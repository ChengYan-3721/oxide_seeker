//! Indexing subsystem: scan -> filter -> render -> embed -> store.

pub mod filter;
pub mod ocr_backfill;
pub mod pdf_processor;
pub mod scanner;
pub mod subprocess;
pub mod watcher;
pub mod worker_pool;

pub use worker_pool::IndexProgress;

use crate::{
    config::Config,
    error::Result,
    search::{phash_store::PhashStore, vector_index::VectorIndex},
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
/// PDF rendering and encoder inference run in worker subprocesses (see
/// [`subprocess::WorkerProcess`]), so the parent never needs an embedder
/// or `ThumbnailStore` of its own — the model path and thumbnails dir come
/// straight out of `config`.
pub async fn start_full_index(
    config: Arc<Config>,
    pool: DbPool,
    index: Arc<VectorIndex>,
    phash_store: Arc<PhashStore>,
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
            let region_ids = crate::storage::database::delete_file_by_path(&pool, &db_path).await?;
            if !region_ids.is_empty() {
                let vids: Vec<u64> = region_ids.iter().map(|v| *v as u64).collect();
                if let Err(e) = index.remove(&vids) {
                    tracing::warn!("Failed to remove stale vectors for {}: {}", db_path, e);
                }
                phash_store.remove(&region_ids);
            }
        }
    }

    // Persist the tombstones added during stale cleanup.
    {
        let idx = index.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to save vector index after stale cleanup: {}", e);
            }
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

    // Bulk-load every existing files-table row in a single query.  A scan
    // that finds 300k files would otherwise issue 300k async SQLite queries
    // here — each cheap individually, but their cumulative latency makes the
    // CLI look frozen for many minutes between "Scan complete" and the first
    // worker-pool log line.  In-memory map lookups are O(1) and microsecond-
    // scale.
    let known_records = crate::storage::database::get_all_file_records(&pool).await?;
    let known_by_path: std::collections::HashMap<String, crate::storage::database::FileRecord> =
        known_records
            .into_iter()
            .map(|r| (r.path.clone(), r))
            .collect();
    tracing::info!(
        "Loaded {} existing file records for filter stage",
        known_by_path.len()
    );

    let mut to_index = Vec::new();
    let total_to_filter = all_files.len();
    let mut scanned = 0usize;
    let mut hashed = 0usize;
    for file in all_files {
        scanned += 1;
        if scanned % 10_000 == 0 {
            tracing::info!(
                "Filter progress: {}/{} scanned, {} queued, {} byte-hashed",
                scanned,
                total_to_filter,
                to_index.len(),
                hashed
            );
        }

        let path_str = file.path.to_string_lossy().into_owned();
        match known_by_path.get(&path_str) {
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
            Some(rec)
                if rec.is_excluded == 1
                    && !scanner::needs_reindex(&file.path, rec.modified_at) =>
            {
                // File was previously excluded and hasn't been modified since;
                // no point re-processing it — it will be excluded again.
            }
            Some(rec) => {
                // The file has a row AND its mtime moved.  Compute the
                // content hash so we can short-circuit "mtime moved but
                // bytes are identical" — the bulk of cases for design files
                // re-saved through Illustrator without real edits.
                hashed += 1;
                let hash = scanner::hash_file_sha1(&file.path);
                if let Some(h) = hash.as_ref() {
                    if rec.content_sha1.as_deref() == Some(h.as_str())
                        && rec.indexed_at.is_some()
                    {
                        // Bytes are identical — refresh metadata so this file
                        // doesn't keep failing the cheap mtime check on every
                        // rescan, and skip the worker pool.
                        if let Err(e) = crate::storage::database::upsert_file(
                            &pool,
                            &path_str,
                            &file.filename,
                            &file.file_type,
                            file.file_size.map(|s| s as i64),
                            file.modified_at,
                        )
                        .await
                        {
                            tracing::warn!(
                                "Failed to refresh metadata for unchanged file {}: {}",
                                file.path.display(),
                                e
                            );
                        }
                        if let Err(e) = crate::storage::database::mark_file_indexed(
                            &pool,
                            rec.id,
                            rec.page_count,
                        )
                        .await
                        {
                            tracing::warn!(
                                "Failed to refresh indexed_at for unchanged file {}: {}",
                                file.path.display(),
                                e
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
            None => {
                // Brand-new file — no prior row, so a content hash would
                // have nothing to compare against.  Skip the SHA-1 (which
                // would cost a full sequential read of every file on disk —
                // hours on a 300k-file network share) and let the worker
                // pool compute it on first index instead.
                to_index.push(file);
            }
        }
    }

    tracing::info!(
        "Filter complete: {} files queued for indexing ({} byte-hashed during filter)",
        to_index.len(),
        hashed
    );

    // Update the shared progress total now that we know how many files there are
    progress.total.store(to_index.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let progress_clone = progress.clone();

    // 3. Run the worker pool in a blocking task (CPU-intensive)
    tokio::task::spawn_blocking(move || {
        worker_pool::run_batch(to_index, pool, index, phash_store, config, progress_clone);
    });

    Ok(())
}