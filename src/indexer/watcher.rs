//! File system watcher for incremental index updates.
//!
//! Uses the `notify` crate to watch the configured scan directories.
//! When files are created or modified, they are queued for re-indexing.

use crate::{
    config::Config,
    embedder::clip::ClipEmbedder,
    error::Result,
    indexer::worker_pool,
    search::vector_index::VectorIndex,
    storage::{database::DbPool, thumbnail::ThumbnailStore},
};
use notify::{
    event::{CreateKind, ModifyKind, RemoveKind},
    Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::{
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tokio::sync::mpsc;

/// Start watching all configured scan directories for file changes.
///
/// Returns a `RecommendedWatcher` handle; dropping it stops the watcher.
/// Changed/created PDF and AI files are re-indexed automatically.
pub async fn start_watcher(
    config: Arc<Config>,
    pool: DbPool,
    clip: Arc<ClipEmbedder>,
    index: Arc<VectorIndex>,
    thumb_store: Arc<ThumbnailStore>,
) -> Result<RecommendedWatcher> {
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
    let (del_tx, mut del_rx) = mpsc::unbounded_channel::<PathBuf>();

    // Spawn a task that drains the channel and re-indexes changed files
    let config_clone = config.clone();
    let pool_clone = pool.clone();
    let clip_clone = clip.clone();
    let index_clone = index.clone();
    let thumb_clone = thumb_store.clone();

    tokio::spawn(async move {
        // Debounce: collect events for up to 2s before triggering indexing
        let mut pending: Vec<PathBuf> = Vec::new();
        let mut pending_deletes: Vec<PathBuf> = Vec::new();
        let mut interval = tokio::time::interval(Duration::from_secs(2));

        loop {
            tokio::select! {
                Some(path) = rx.recv() => {
                    if !pending.contains(&path) {
                        pending.push(path);
                    }
                }
                Some(path) = del_rx.recv() => {
                    if !pending_deletes.contains(&path) {
                        pending_deletes.push(path);
                    }
                }
                _ = interval.tick() => {
                    // Handle deletions first
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
                        // Save vector index after removals and trigger background rebuild
                        if let Err(e) = index_clone.save() {
                            tracing::warn!("Failed to save vector index after deletions: {}", e);
                        }
                        index_clone.trigger_rebuild();
                    }

                    // Handle new/modified files
                    if pending.is_empty() {
                        continue;
                    }
                    let batch = std::mem::take(&mut pending);
                    tracing::info!("Re-indexing {} changed file(s)", batch.len());

                    let discovered: Vec<_> = batch
                        .into_iter()
                        .filter_map(|p| {
                            let ext = p.extension()?.to_string_lossy().to_lowercase();
                            let file_type = match ext.as_str() {
                                "pdf" => "pdf",
                                "ai" => "ai",
                                _ => return None,
                            };
                            let filename = p
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            let meta = std::fs::metadata(&p).ok();
                            let file_size = meta.as_ref().map(|m| m.len());
                            let modified_at = meta
                                .as_ref()
                                .and_then(|m| m.modified().ok())
                                .and_then(|t| {
                                    t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()
                                })
                                .map(|d| d.as_secs() as i64);
                            Some(crate::indexer::scanner::DiscoveredFile {
                                path: p,
                                filename,
                                file_type: file_type.to_string(),
                                file_size,
                                modified_at,
                            })
                        })
                        .collect();

                    if discovered.is_empty() {
                        continue;
                    }

                    let cfg = config_clone.clone();
                    let p = pool_clone.clone();
                    let c = clip_clone.clone();
                    let i = index_clone.clone();
                    let t = thumb_clone.clone();
                    let progress = worker_pool::IndexProgress::new(discovered.len() as u64);

                    tokio::task::spawn_blocking(move || {
                        worker_pool::run_batch(discovered, p, c, i, t, cfg, progress);
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