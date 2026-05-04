//! File system watcher for incremental index updates.
//!
//! Uses the `notify` crate to watch the configured scan directories.
//! When files are created or modified, they are queued, run through a
//! **stability check** (sampled twice with a wait), then run through a
//! **content-hash short-circuit** (skip if bytes are identical to what
//! was indexed last time), and finally dispatched to the worker pool.

use crate::{
    config::Config,
    error::Result,
    indexer::{scanner, worker_pool},
    search::vector_index::VectorIndex,
    storage::database::DbPool,
};
use notify::{
    event::{CreateKind, ModifyKind, RemoveKind},
    Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tokio::sync::mpsc;

/// Per-path stability tracking state.  A file enters `pending` on its first
/// notify event; on each tick we re-sample its size+mtime and only dispatch
/// once two consecutive samples agree.  This prevents PDFium from being
/// pointed at a half-flushed file (the most common failure mode for the
/// "designer saves a blank, then fills it in" workflow).
#[derive(Debug, Clone)]
struct PendingState {
    last_size: u64,
    last_mtime: i64,
    /// Number of stability checks performed so far.
    attempts: u32,
}

impl Default for PendingState {
    fn default() -> Self {
        Self {
            last_size: u64::MAX, // sentinel: forces "not stable" on first compare
            last_mtime: i64::MIN,
            attempts: 0,
        }
    }
}

/// Start watching all configured scan directories for file changes.
///
/// Returns a `RecommendedWatcher` handle; dropping it stops the watcher.
/// Changed/created PDF and AI files are re-indexed automatically.
pub async fn start_watcher(
    config: Arc<Config>,
    pool: DbPool,
    index: Arc<VectorIndex>,
) -> Result<RecommendedWatcher> {
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
    let (del_tx, mut del_rx) = mpsc::unbounded_channel::<PathBuf>();

    let settle_secs = config.indexer.watcher_settle_secs.max(1);
    let min_bytes = config.indexer.watcher_min_bytes;
    let max_retries = config.indexer.watcher_max_retries;

    // Spawn a task that drains the channel and re-indexes changed files
    let config_clone = config.clone();
    let pool_clone = pool.clone();
    let index_clone = index.clone();

    tokio::spawn(async move {
        let mut pending: HashMap<PathBuf, PendingState> = HashMap::new();
        let mut pending_deletes: Vec<PathBuf> = Vec::new();
        let mut interval = tokio::time::interval(Duration::from_secs(settle_secs));

        loop {
            tokio::select! {
                Some(path) = rx.recv() => {
                    pending.entry(path).or_insert_with(PendingState::default);
                }
                Some(path) = del_rx.recv() => {
                    // If the file was awaiting stability and got removed,
                    // drop it from `pending` first so we don't re-dispatch.
                    pending.remove(&path);
                    if !pending_deletes.contains(&path) {
                        pending_deletes.push(path);
                    }
                }
                _ = interval.tick() => {
                    // ── Deletions first ──────────────────────────────────────
                    if !pending_deletes.is_empty() {
                        let batch = std::mem::take(&mut pending_deletes);
                        tracing::info!("Removing {} deleted file(s) from index", batch.len());
                        for path in batch {
                            let path_str = path.to_string_lossy().to_string();
                            match crate::storage::database::delete_file_by_path(
                                &pool_clone, &path_str,
                            ).await {
                                Ok(vector_ids) => {
                                    if !vector_ids.is_empty() {
                                        let vids: Vec<u64> = vector_ids
                                            .into_iter()
                                            .map(|v| v as u64)
                                            .collect();
                                        if let Err(e) = index_clone.remove(&vids) {
                                            tracing::warn!(
                                                "Failed to remove vectors for {}: {}",
                                                path_str, e
                                            );
                                        }
                                    }
                                    tracing::info!("Removed from index: {}", path_str);
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to delete DB record for {}: {}",
                                        path_str, e
                                    );
                                }
                            }
                        }
                        if let Err(e) = index_clone.save() {
                            tracing::warn!("Failed to save vector index after deletions: {}", e);
                        }
                        index_clone.trigger_rebuild();
                    }

                    if pending.is_empty() {
                        continue;
                    }

                    // ── Stability check ─────────────────────────────────────
                    // Walk every pending path, re-sample size+mtime.  Only
                    // dispatch ones whose two consecutive samples agree AND
                    // whose size meets `min_bytes`.
                    let snapshot: Vec<(PathBuf, PendingState)> = pending
                        .iter()
                        .map(|(p, s)| (p.clone(), s.clone()))
                        .collect();

                    let mut ready: Vec<PathBuf> = Vec::new();
                    for (path, prev) in snapshot {
                        let meta = match std::fs::metadata(&path) {
                            Ok(m) => m,
                            Err(_) => {
                                // Vanished mid-flight — drop it; a Remove event
                                // (if any) will arrive separately.
                                pending.remove(&path);
                                continue;
                            }
                        };

                        let size = meta.len();
                        let mtime = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);

                        // Empty / placeholder file: keep waiting unless we've
                        // exhausted retries, in which case drop it (the next
                        // periodic rescan will pick it up if it ever grows).
                        if size < min_bytes {
                            let entry = pending.get_mut(&path).expect("present");
                            entry.last_size = size;
                            entry.last_mtime = mtime;
                            entry.attempts += 1;
                            if entry.attempts >= max_retries {
                                tracing::debug!(
                                    "Watcher dropping persistently-empty file (size={} < {}): {}",
                                    size, min_bytes, path.display()
                                );
                                pending.remove(&path);
                            }
                            continue;
                        }

                        // Stable iff this sample matches the previous one.
                        if prev.attempts > 0 && prev.last_size == size && prev.last_mtime == mtime {
                            ready.push(path.clone());
                            pending.remove(&path);
                            continue;
                        }

                        // Not yet stable — record this sample and wait one more tick.
                        let entry = pending.get_mut(&path).expect("present");
                        entry.last_size = size;
                        entry.last_mtime = mtime;
                        entry.attempts += 1;
                        if entry.attempts >= max_retries {
                            tracing::warn!(
                                "Watcher giving up on unstable file after {} retries: {}",
                                entry.attempts, path.display()
                            );
                            pending.remove(&path);
                        }
                    }

                    if ready.is_empty() {
                        continue;
                    }

                    // ── Hash short-circuit + DiscoveredFile build ───────────
                    // For each stable file: hash its bytes, compare against the
                    // DB's last-known hash; if equal, just bump indexed_at and
                    // skip the worker pool entirely.
                    let mut to_dispatch: Vec<scanner::DiscoveredFile> = Vec::new();
                    for path in ready {
                        let ext = match path.extension().map(|e| e.to_string_lossy().to_lowercase()) {
                            Some(e) => e,
                            None => continue,
                        };
                        let file_type = match ext.as_str() {
                            "pdf" => "pdf",
                            "ai" => "ai",
                            _ => continue,
                        };

                        let meta = match std::fs::metadata(&path) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        let file_size = Some(meta.len());
                        let modified_at = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs() as i64);
                        let filename = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();

                        let hash = match scanner::hash_file_sha1(&path) {
                            Some(h) => h,
                            None => {
                                // Hashing failed (file locked / vanished) —
                                // re-queue for another settling pass.
                                pending
                                    .entry(path.clone())
                                    .or_insert_with(PendingState::default);
                                continue;
                            }
                        };

                        let path_str = path.to_string_lossy().to_string();
                        let existing = crate::storage::database::get_file_by_path(
                            &pool_clone, &path_str,
                        )
                        .await
                        .ok()
                        .flatten();

                        if let Some(rec) = existing.as_ref() {
                            if rec.content_sha1.as_deref() == Some(hash.as_str())
                                && rec.indexed_at.is_some()
                            {
                                // Bytes haven't changed since last successful
                                // index — refresh the file_size / modified_at
                                // (so the next mtime-only scan won't re-trigger)
                                // and move on.
                                if let Err(e) = crate::storage::database::upsert_file(
                                    &pool_clone,
                                    &path_str,
                                    &filename,
                                    file_type,
                                    file_size.map(|s| s as i64),
                                    modified_at,
                                ).await {
                                    tracing::warn!(
                                        "Failed to refresh metadata for unchanged file {}: {}",
                                        path_str, e
                                    );
                                }
                                tracing::debug!(
                                    "Watcher: bytes unchanged, skipping re-index: {}",
                                    path_str
                                );
                                continue;
                            }
                        }

                        to_dispatch.push(scanner::DiscoveredFile {
                            path,
                            filename,
                            file_type: file_type.to_string(),
                            file_size,
                            modified_at,
                            content_sha1: Some(hash),
                        });
                    }

                    if to_dispatch.is_empty() {
                        continue;
                    }

                    tracing::info!("Re-indexing {} changed file(s)", to_dispatch.len());

                    let cfg = config_clone.clone();
                    let p = pool_clone.clone();
                    let i = index_clone.clone();
                    let progress = worker_pool::IndexProgress::new(to_dispatch.len() as u64);

                    tokio::task::spawn_blocking(move || {
                        worker_pool::run_batch(to_dispatch, p, i, cfg, progress);
                    });
                }
            }
        }
    });

    // Build and start the OS-level watcher
    let tx_clone = tx.clone();
    let del_tx_clone = del_tx.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        if let Ok(event) = res {
            match event.kind {
                EventKind::Create(CreateKind::File)
                | EventKind::Modify(ModifyKind::Data(_))
                | EventKind::Modify(ModifyKind::Any) => {
                    for path in event.paths {
                        let ext = path
                            .extension()
                            .map(|e| e.to_string_lossy().to_lowercase())
                            .unwrap_or_default();
                        if ext == "pdf" || ext == "ai" {
                            let _ = tx_clone.send(path);
                        }
                    }
                }
                EventKind::Remove(RemoveKind::File)
                | EventKind::Remove(RemoveKind::Any) => {
                    for path in event.paths {
                        let ext = path
                            .extension()
                            .map(|e| e.to_string_lossy().to_lowercase())
                            .unwrap_or_default();
                        if ext == "pdf" || ext == "ai" {
                            let _ = del_tx_clone.send(path);
                        }
                    }
                }
                _ => {}
            }
        }
    })
    .map_err(|e| crate::error::AppError::Other(anyhow::anyhow!("Watcher error: {}", e)))?;

    for dir in &config.paths.scan_dirs {
        if dir.exists() {
            watcher
                .watch(dir, RecursiveMode::Recursive)
                .map_err(|e| {
                    crate::error::AppError::Other(anyhow::anyhow!(
                        "Cannot watch {}: {}",
                        dir.display(),
                        e
                    ))
                })?;
            tracing::info!("Watching directory for changes: {}", dir.display());
        }
    }

    Ok(watcher)
}
