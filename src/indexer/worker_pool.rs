//! Concurrent indexing worker pool — out-of-process variant.
//!
//! Architecture
//! ------------
//! * A `crossbeam-channel` bounded queue is fed by the scanner.
//! * A fixed-size rayon thread pool drains the queue.  Each worker thread
//!   owns one long-lived [`WorkerProcess`] subprocess (the same binary
//!   re-launched with `--worker-mode`) that does the actual pdfium / CLIP
//!   work.  The parent thread:
//!   1. Upserts the file row and bumps the persistent `crash_attempts`
//!      counter (durable in WAL — survives an FFI crash).
//!   2. Sends the file to the subprocess and waits for one response.
//!   3. On `Indexed`, allocates HNSW ids, inserts the vectors, and persists
//!      the page rows in a single SQLite transaction.
//!   4. On `Excluded`, marks the file row excluded.
//!   5. On a Rust-level `Error`, increments the failed counter and resets
//!      the crash attempt counter (this kind of failure is recoverable).
//!   6. On a broken pipe / EOF (subprocess crashed), respawns a fresh
//!      subprocess and **leaves the crash counter incremented** so the next
//!      run's filter step can apply the auto-blacklist threshold.
//!   7. Periodically saves the vector index (crash-recovery checkpoint).

use crate::{
    config::Config,
    error::Result,
    indexer::{
        scanner::DiscoveredFile,
        subprocess::{PageData, ProcessRequest, ProcessResponse, WorkerProcess},
    },
    search::vector_index::{VectorId, VectorIndex},
    storage::database::{self, DbPool, PageUpsert},
};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

/// Live counters updated by worker threads; readable without a lock.
pub struct IndexProgress {
    pub processed: AtomicU64,
    pub failed: AtomicU64,
    pub excluded: AtomicU64,
    pub total: AtomicU64,
    /// Set to `true` once `run_batch` has fully completed (all workers done).
    pub finished: AtomicBool,
}

impl IndexProgress {
    pub fn new(total: u64) -> Arc<Self> {
        Arc::new(Self {
            processed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            excluded: AtomicU64::new(0),
            total: AtomicU64::new(total),
            finished: AtomicBool::new(false),
        })
    }

    /// Fraction of files processed (0.0 – 100.0).
    pub fn percent(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let done = self.processed.load(Ordering::Relaxed)
            + self.failed.load(Ordering::Relaxed)
            + self.excluded.load(Ordering::Relaxed);
        (done as f64 / total as f64 * 100.0).min(100.0)
    }
}

/// Run full indexing of `files` using a thread pool of `worker_threads`
/// threads, each driving its own subprocess.
///
/// This function blocks until all files are processed. Call it from a
/// `tokio::task::spawn_blocking` context.
pub fn run_batch(
    files: Vec<DiscoveredFile>,
    pool: DbPool,
    index: Arc<VectorIndex>,
    config: Arc<Config>,
    progress: Arc<IndexProgress>,
) {
    let n = files.len();
    progress.total.store(n as u64, Ordering::Relaxed);

    if n == 0 {
        tracing::info!("No files to index.");
        progress.finished.store(true, Ordering::Release);
        return;
    }

    tracing::info!("Starting batch index of {} files (out-of-process)", n);

    // Bounded channel prevents unbounded memory growth.
    let (tx, rx): (Sender<DiscoveredFile>, Receiver<DiscoveredFile>) = bounded(64);

    // Producer thread — feeds files into the channel.
    std::thread::spawn(move || {
        for f in files {
            let _ = tx.send(f);
        }
    });

    let num_threads = config.indexer.worker_threads;
    let pool_rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build rayon thread pool");

    let model_path = config.paths.model_path.clone();
    let thumbnails_dir = config.thumbnails_dir();

    pool_rayon.scope(|scope| {
        for worker_idx in 0..num_threads {
            let rx = rx.clone();
            let db_pool = pool.clone();
            let index = index.clone();
            let progress = progress.clone();
            let model_path = model_path.clone();
            let thumbnails_dir = thumbnails_dir.clone();

            scope.spawn(move |_| {
                run_worker(
                    worker_idx,
                    rx,
                    db_pool,
                    index,
                    progress,
                    model_path,
                    thumbnails_dir,
                    n,
                );
            });
        }
    });

    // Final authoritative save.
    if let Err(e) = index.save() {
        tracing::error!("Failed to save vector index after batch: {}", e);
    }

    // Reconcile counters to ensure 100% is reached.
    let p = progress.processed.load(Ordering::Relaxed);
    let e = progress.excluded.load(Ordering::Relaxed);
    let f = progress.failed.load(Ordering::Relaxed);
    let actual_done = p + e + f;
    let total = progress.total.load(Ordering::Relaxed);

    if actual_done < total {
        let missing = total - actual_done;
        progress.failed.fetch_add(missing, Ordering::Relaxed);
    } else if actual_done > total && total > 0 {
        progress.total.store(actual_done, Ordering::Relaxed);
    }

    progress.finished.store(true, Ordering::Release);

    tracing::info!(
        "Batch indexing complete: {} indexed, {} excluded, {} failed (total {})",
        progress.processed.load(Ordering::Relaxed),
        progress.excluded.load(Ordering::Relaxed),
        progress.failed.load(Ordering::Relaxed),
        progress.total.load(Ordering::Relaxed),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    worker_idx: usize,
    rx: Receiver<DiscoveredFile>,
    db_pool: DbPool,
    index: Arc<VectorIndex>,
    progress: Arc<IndexProgress>,
    model_path: PathBuf,
    thumbnails_dir: PathBuf,
    total_files: usize,
) {
    // Single-threaded tokio runtime per worker for async DB calls.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Worker tokio runtime");

    let mut subproc = match WorkerProcess::spawn(&model_path, &thumbnails_dir, worker_idx) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(
                "[worker {}] Failed to spawn indexing subprocess: {}",
                worker_idx, e
            );
            return;
        }
    };

    while let Ok(file) = rx.recv() {
        // Diagnostic breadcrumb so the operator can see in-flight files
        // after a parent-side panic.  (Subprocess crashes also log here
        // before respawn.)
        let _ = std::fs::write(
            format!("./logs/last_{}.txt", worker_idx),
            file.path.to_string_lossy().as_bytes(),
        );

        // Upsert + bump counter on the parent side — the child never touches
        // the DB.
        let file_id = match rt.block_on(database::upsert_file(
            &db_pool,
            &file.path.to_string_lossy(),
            &file.filename,
            &file.file_type,
            file.file_size.map(|s| s as i64),
            file.modified_at,
        )) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(
                    "Failed to upsert file row for {}: {}",
                    file.path.display(),
                    e
                );
                progress.failed.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };

        if let Err(e) = rt.block_on(database::bump_crash_attempts(&db_pool, file_id)) {
            tracing::warn!(
                "Failed to bump crash_attempts for {}: {}",
                file.path.display(),
                e
            );
        }

        let request = ProcessRequest {
            file_path: file.path.clone(),
            file_id,
        };

        let mut should_reset_counter = true;
        match subproc.process(&request) {
            Ok(ProcessResponse::Indexed { pages }) => {
                handle_indexed(&rt, &db_pool, &index, file_id, &file.path, &pages, &progress);
            }
            Ok(ProcessResponse::Excluded { reason }) => {
                if let Err(e) = rt.block_on(database::mark_file_excluded(&db_pool, file_id)) {
                    tracing::warn!(
                        "Failed to mark file excluded {}: {}",
                        file.path.display(),
                        e
                    );
                }
                tracing::debug!(
                    "Excluded (imposition): {} — {}",
                    file.path.display(),
                    reason
                );
                progress.excluded.fetch_add(1, Ordering::Relaxed);
            }
            Ok(ProcessResponse::Error { msg }) => {
                tracing::warn!("Failed to index {}: {}", file.path.display(), msg);
                progress.failed.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                // Broken pipe / EOF — subprocess died on this file.  Leave
                // the crash counter bumped so the next run's filter can
                // auto-blacklist the file once the threshold is reached.
                tracing::error!(
                    "[worker {}] subprocess crashed on {}: {}",
                    worker_idx,
                    file.path.display(),
                    e
                );
                progress.failed.fetch_add(1, Ordering::Relaxed);
                should_reset_counter = false;

                subproc = match WorkerProcess::spawn(&model_path, &thumbnails_dir, worker_idx) {
                    Ok(w) => w,
                    Err(spawn_err) => {
                        tracing::error!(
                            "[worker {}] failed to respawn subprocess, exiting: {}",
                            worker_idx, spawn_err
                        );
                        return;
                    }
                };
            }
        }

        if should_reset_counter {
            if let Err(e) = rt.block_on(database::reset_crash_attempts(&db_pool, file_id)) {
                tracing::warn!(
                    "Failed to reset crash_attempts for {}: {}",
                    file.path.display(),
                    e
                );
            }
        }

        let done = progress.processed.load(Ordering::Relaxed)
            + progress.failed.load(Ordering::Relaxed)
            + progress.excluded.load(Ordering::Relaxed);

        if done % 500 == 0 {
            tracing::info!(
                "Progress: {}/{} files ({:.1}%)",
                done,
                total_files,
                progress.percent()
            );
            // Checkpoint the vector index (cheap on hnsw_rs).
            if let Err(e) = index.save() {
                tracing::warn!("Failed to save vector index: {}", e);
            }
        }
    }
}

fn handle_indexed(
    rt: &tokio::runtime::Runtime,
    pool: &DbPool,
    index: &VectorIndex,
    file_id: i64,
    file_path: &std::path::Path,
    pages: &[PageData],
    progress: &IndexProgress,
) {
    if pages.is_empty() {
        if let Err(e) = rt.block_on(database::mark_file_indexed(pool, file_id, 0)) {
            tracing::warn!(
                "Failed to mark file indexed {}: {}",
                file_path.display(),
                e
            );
            progress.failed.fetch_add(1, Ordering::Relaxed);
        } else {
            progress.processed.fetch_add(1, Ordering::Relaxed);
        }
        return;
    }

    // Allocate HNSW ids and insert the whole batch.
    let vector_ids: Vec<VectorId> = (0..pages.len()).map(|_| index.next_id()).collect();
    let vec_batch: Vec<(VectorId, Vec<f32>)> = vector_ids
        .iter()
        .zip(pages.iter())
        .map(|(id, p)| (*id, p.vector.clone()))
        .collect();

    if let Err(e) = index.add_batch(&vec_batch) {
        tracing::warn!(
            "HNSW insert failed for {}: {}",
            file_path.display(),
            e
        );
        progress.failed.fetch_add(1, Ordering::Relaxed);
        return;
    }

    // Persist page rows in a single transaction.
    let rows: Vec<PageUpsert<'_>> = pages
        .iter()
        .enumerate()
        .map(|(i, p)| PageUpsert {
            page_num: p.page_num as i64,
            phash: Some(p.phash.as_str()),
            vector_id: Some(vector_ids[i] as i64),
            thumb_path: Some(p.thumb_relative_path.as_str()),
            width_px: Some(p.width_px as i64),
            height_px: Some(p.height_px as i64),
        })
        .collect();

    let outcome: Result<()> = rt.block_on(async {
        database::upsert_pages_batch(pool, file_id, &rows).await?;
        database::mark_file_indexed(pool, file_id, pages.len() as i64).await?;
        Ok(())
    });

    match outcome {
        Ok(()) => {
            tracing::debug!(
                "Indexed: {} ({} pages)",
                file_path.display(),
                pages.len()
            );
            progress.processed.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            tracing::warn!(
                "Failed to persist indexed pages for {}: {}",
                file_path.display(),
                e
            );
            progress.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}
