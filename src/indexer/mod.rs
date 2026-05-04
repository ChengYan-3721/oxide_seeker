//! Indexing subsystem: scan -> filter -> render -> embed -> store.

pub mod filter;
pub mod pdf_processor;
pub mod scanner;
pub mod subprocess;
pub mod watcher;
pub mod worker_pool;

pub use worker_pool::IndexProgress;

use crate::{
    config::Config,
    error::Result,
    search::vector_index::VectorIndex,
    storage::database::DbPool,
};
use std::sync::Arc;

/// Kick off a full (or incremental) index of all configured scan directories.
///
/// - Scans the filesystem for PDF/AI files
/// - Skips files that haven't changed since last index
/// - Spawns the worker pool to process new/changed files
///
/// `progress` is the caller-owned Arc that is also passed to the WebSocket
/// handler so both share the same live counters.  The total is updated here
/// once the file list is known.
///
/// PDF rendering and CLIP inference run in worker subprocesses (see
/// [`subprocess::WorkerProcess`]), so the parent never needs a `ClipEmbedder`
/// or `ThumbnailStore` of its own — the model path and thumbnails dir come
/// straight out of `config`.
pub async fn start_full_index(
    config: Arc<Config>,
    pool: DbPool,
    index: Arc<VectorIndex>,
    progress: Arc<IndexProgress>,
) -> Result<()> {
    // 1. Scan directories
    let all_files = {
        let dirs = config.paths.scan_dirs.clone();
        let max_depth = config.indexer.effective_max_scan_depth();
        tokio::task::spawn_blocking(move || scanner::scan_directories(&dirs, max_depth))
            .await
            .map_err(|e| crate::error::AppError::Other(anyhow::anyhow!("Scan task panicked: {}", e)))?
    };

    // 1b. Clean up DB records for files that no longer exist on disk
    let discovered_paths: std::collections::HashSet<String> = all_files
        .iter()
        .map(|f| f.path.to_string_lossy().into_owned())
        .collect();
    let db_files = crate::storage::database::get_all_file_paths(&pool).await?;
    for db_path in db_files {
        if !discovered_paths.contains(&db_path) {
            tracing::info!("File removed from disk, cleaning index: {}", db_path);
            let vector_ids = crate::storage::database::delete_file_by_path(&pool, &db_path).await?;
            if !vector_ids.is_empty() {
                let vids: Vec<u64> = vector_ids.into_iter().map(|v| v as u64).collect();
                if let Err(e) = index.remove(&vids) {
                    tracing::warn!("Failed to remove stale vectors for {}: {}", db_path, e);
                }
            }
        }
    }

    // If stale entries were removed, save and trigger a background rebuild
    // so that searches don't pay the rebuild cost later.
    {
        let idx = index.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to save vector index after stale cleanup: {}", e);
            }
            idx.trigger_rebuild();
        })
        .await
        .ok();
    }

    // 2. Filter to only files needing (re-)indexing
    // Files whose `crash_attempts` counter reached this threshold on previous
    // runs are presumed to be FFI poison-pills (pdfium / onnxruntime raising
    // a structured exception that catch_unwind cannot catch).  We auto-blacklist
    // them with a clear `exclusion_reason` so operators can see what was
    // skipped and why.
    const CRASH_ATTEMPT_THRESHOLD: i64 = 2;

    let mut to_index = Vec::new();
    for file in all_files {
        match crate::storage::database::get_file_by_path(
            &pool,
            &file.path.to_string_lossy(),
        )
        .await?
        {
            Some(rec) if rec.crash_attempts >= CRASH_ATTEMPT_THRESHOLD => {
                if rec.is_excluded == 0 {
                    let reason = format!(
                        "Process crashed {} times while indexing this file (suspected FFI poison pill)",
                        rec.crash_attempts
                    );
                    tracing::warn!(
                        "Auto-excluding file after {} crashes: {} ({})",
                        rec.crash_attempts,
                        file.path.display(),
                        reason
                    );
                    crate::storage::database::mark_file_excluded_with_reason(
                        &pool,
                        rec.id,
                        &reason,
                    )
                    .await?;
                }
            }
            Some(rec) if !scanner::needs_reindex(&file.path, rec.indexed_at) => {
                // File is up-to-date; skip
            }
            Some(rec) if rec.is_excluded == 1 && !scanner::needs_reindex(&file.path, rec.modified_at) => {
                // File was previously excluded and hasn't been modified since;
                // no point re-processing it — it will be excluded again.
            }
            existing => {
                // The file's mtime moved (or it's new).  Compute its content
                // hash and short-circuit when the bytes match what we already
                // indexed — this is the strong-idempotency layer that protects
                // against "save with no edits" / antivirus rewrites / clock
                // drift on network shares.
                let hash = scanner::hash_file_sha1(&file.path);

                if let (Some(rec), Some(h)) = (existing.as_ref(), hash.as_ref()) {
                    if rec.content_sha1.as_deref() == Some(h.as_str())
                        && rec.indexed_at.is_some()
                    {
                        // Bytes are identical — refresh metadata so this file
                        // doesn't keep failing the cheap mtime check on every
                        // rescan, and skip the worker pool.
                        if let Err(e) = crate::storage::database::upsert_file(
                            &pool,
                            &file.path.to_string_lossy(),
                            &file.filename,
                            &file.file_type,
                            file.file_size.map(|s| s as i64),
                            file.modified_at,
                        ).await {
                            tracing::warn!(
                                "Failed to refresh metadata for unchanged file {}: {}",
                                file.path.display(), e
                            );
                        }
                        // Bump indexed_at to suppress future mtime-based
                        // re-triggers for this exact byte content.
                        if let Err(e) = crate::storage::database::mark_file_indexed(
                            &pool, rec.id, rec.page_count,
                        ).await {
                            tracing::warn!(
                                "Failed to refresh indexed_at for unchanged file {}: {}",
                                file.path.display(), e
                            );
                        }
                        continue;
                    }
                }

                to_index.push(scanner::DiscoveredFile {
                    content_sha1: hash,
                    ..file
                });
            }
        }
    }

    tracing::info!("{} files queued for indexing", to_index.len());

    // Update the shared progress total now that we know how many files there are
    progress.total.store(to_index.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let progress_clone = progress.clone();

    // 3. Run the worker pool in a blocking task (CPU-intensive)
    tokio::task::spawn_blocking(move || {
        worker_pool::run_batch(to_index, pool, index, config, progress_clone);
    });

    Ok(())
}