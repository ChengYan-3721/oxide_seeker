//! Concurrent indexing worker pool.
//!
//! Architecture
//! ------------
//! * A `crossbeam-channel` bounded queue is fed by the scanner.
//! * A fixed-size rayon thread pool drains the queue.  Each worker:
//!   1. Creates its own **dedicated `ClipSession`** — no `Mutex`, no
//!      contention across threads.
//!   2. Opens a PDFium handle (PDFium is `!Send`, one per thread).
//!   3. For each file:
//!      * renders all pages
//!      * saves every page's thumbnail
//!      * computes every page's pHash
//!      * **batches all pages through CLIP in one ONNX forward pass**
//!      * inserts the batch of vectors into the HNSW index
//!      * **upserts every page row inside a single SQLite transaction**
//!   4. Periodically saves the vector index (crash-recovery checkpoint).

use crate::{
    config::Config,
    embedder::clip::{ClipEmbedder, ClipSession},
    embedder::phash::compute_phash,
    error::Result,
    indexer::{pdf_processor, scanner::DiscoveredFile},
    search::vector_index::{VectorId, VectorIndex},
    storage::{
        database::{self, DbPool, PageUpsert},
        thumbnail::{ThumbnailStore, THUMB_SIZE},
    },
};
use crossbeam_channel::{bounded, Receiver, Sender};
use pdfium_render::prelude::Pdfium;
use std::panic::AssertUnwindSafe;
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

/// Run full indexing of `files` using a thread pool of `worker_threads` threads.
///
/// This function blocks until all files are processed. Call it from a
/// `tokio::task::spawn_blocking` context.
pub fn run_batch(
    files: Vec<DiscoveredFile>,
    pool: DbPool,
    clip: Arc<ClipEmbedder>,
    index: Arc<VectorIndex>,
    thumb_store: Arc<ThumbnailStore>,
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

    tracing::info!("Starting batch index of {} files", n);

    // Bounded channel prevents unbounded memory growth
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

    pool_rayon.scope(|scope| {
        for worker_idx in 0..num_threads {
            let rx = rx.clone();
            let db_pool = pool.clone();
            let clip = clip.clone();
            let index = index.clone();
            let thumb_store = thumb_store.clone();
            let config = config.clone();
            let progress = progress.clone();

            scope.spawn(move |_| {
                // One PDFium per worker thread (PDFium is !Send).
                let pdfium = match pdf_processor::init_pdfium() {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("[worker {}] Failed to init PDFium: {}", worker_idx, e);
                        return;
                    }
                };

                // One CLIP session per worker — true parallel inference.
                // Per-session intra-op threads = 1 so the outer rayon pool
                // supplies all parallelism.  If workers exceed core count,
                // ORT's internal scheduler still cooperates, but the default
                // minimises oversubscription.
                let mut clip_session = match clip.new_session(1) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!(
                            "[worker {}] Failed to build CLIP session: {}",
                            worker_idx, e
                        );
                        return;
                    }
                };

                // Single-threaded tokio runtime per worker for async DB calls.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Worker tokio runtime");

                while let Ok(file) = rx.recv() {
                    // `catch_unwind` prevents a panic in pdfium-render / hnsw_rs
                    // from unwinding the rayon scope and bypassing the final
                    // `progress.finished = true` — which would leave the UI
                    // progress bar stuck forever.
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        rt.block_on(process_one_file(
                            &file,
                            &pdfium,
                            &db_pool,
                            &mut clip_session,
                            &index,
                            &thumb_store,
                            &config,
                        ))
                    }));

                    match result {
                        Ok(Ok(FileOutcome::Indexed)) => {
                            progress.processed.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(Ok(FileOutcome::Excluded)) => {
                            progress.excluded.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("Failed to index {}: {}", file.path.display(), e);
                            progress.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(panic_payload) => {
                            let msg = panic_payload
                                .downcast_ref::<&'static str>()
                                .map(|s| s.to_string())
                                .or_else(|| {
                                    panic_payload.downcast_ref::<String>().cloned()
                                })
                                .unwrap_or_else(|| "<non-string panic>".to_string());
                            tracing::error!(
                                "Panic while indexing {}: {}",
                                file.path.display(),
                                msg
                            );
                            progress.failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    let done = progress.processed.load(Ordering::Relaxed)
                        + progress.failed.load(Ordering::Relaxed)
                        + progress.excluded.load(Ordering::Relaxed);

                    if done % 500 == 0 {
                        tracing::info!(
                            "Progress: {}/{} files ({:.1}%)",
                            done,
                            n,
                            progress.percent()
                        );
                        // Checkpoint the vector index (cheap on hnsw_rs).
                        if let Err(e) = index.save() {
                            tracing::warn!("Failed to save vector index: {}", e);
                        }
                    }
                }
            });
        }
    });

    // Final authoritative save
    if let Err(e) = index.save() {
        tracing::error!("Failed to save vector index after batch: {}", e);
    }

    // Reconcile counters to ensure 100% is reached
    let p = progress.processed.load(Ordering::Relaxed);
    let e = progress.excluded.load(Ordering::Relaxed);
    let f = progress.failed.load(Ordering::Relaxed);
    let actual_done = p + e + f;
    let total = progress.total.load(Ordering::Relaxed);

    if actual_done < total {
        let missing = total - actual_done;
        // Add any missing counts to failed to ensure p + e + f == total
        progress.failed.fetch_add(missing, Ordering::Relaxed);
    } else if actual_done > total && total > 0 {
        // This shouldn't happen, but for safety:
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

enum FileOutcome {
    Indexed,
    Excluded,
}

/// Process a single file: render → batched embed → batched DB write.
async fn process_one_file(
    file: &DiscoveredFile,
    pdfium: &Pdfium,
    pool: &DbPool,
    clip_session: &mut ClipSession,
    index: &VectorIndex,
    thumb_store: &ThumbnailStore,
    _config: &Config,
) -> Result<FileOutcome> {
    // ── 1. File row ───────────────────────────────────────────────────────────
    let file_id = database::upsert_file(
        pool,
        &file.path.to_string_lossy(),
        &file.filename,
        &file.file_type,
        file.file_size.map(|s| s as i64),
        file.modified_at,
    )
    .await?;

    // ── 2. Render + imposition filter ────────────────────────────────────────
    let processed = pdf_processor::process_file(&file.path, pdfium)?;

    if processed.excluded {
        database::mark_file_excluded(pool, file_id).await?;
        tracing::debug!(
            "Excluded (imposition): {} — {}",
            file.path.display(),
            processed.exclusion_reason.as_deref().unwrap_or("?")
        );
        return Ok(FileOutcome::Excluded);
    }

    if processed.pages.is_empty() {
        database::mark_file_indexed(pool, file_id, 0).await?;
        return Ok(FileOutcome::Indexed);
    }

    // ── 3. Thumbnails (sequential — disk-bound, cheap per image) ─────────────
    let mut thumb_paths = Vec::with_capacity(processed.pages.len());
    for p in &processed.pages {
        let t =
            thumb_store.save_thumbnail(&p.image, file_id, p.page_num as i64, THUMB_SIZE)?;
        thumb_paths.push(t);
    }

    // ── 4. pHashes (CPU-bound, short, fine to do sequentially) ───────────────
    let phashes: Vec<String> = processed.pages.iter().map(|p| compute_phash(&p.image)).collect();

    // ── 5. CLIP in a single forward pass across every page ───────────────────
    let img_refs: Vec<&image::DynamicImage> =
        processed.pages.iter().map(|p| &p.image).collect();
    let vectors = clip_session.encode_batch(&img_refs)?;

    if vectors.len() != processed.pages.len() {
        return Err(crate::error::AppError::Search(format!(
            "CLIP batch size mismatch: expected {}, got {}",
            processed.pages.len(),
            vectors.len()
        )));
    }

    // ── 6. Allocate ids and insert the whole batch into HNSW at once ─────────
    let vector_ids: Vec<VectorId> = (0..processed.pages.len()).map(|_| index.next_id()).collect();
    let vec_batch: Vec<(VectorId, Vec<f32>)> = vector_ids
        .iter()
        .zip(vectors.into_iter())
        .map(|(id, v)| (*id, v))
        .collect();
    index.add_batch(&vec_batch)?;

    // ── 7. Single transaction for every page row ─────────────────────────────
    let rows: Vec<PageUpsert<'_>> = processed
        .pages
        .iter()
        .enumerate()
        .map(|(i, p)| PageUpsert {
            page_num: p.page_num as i64,
            phash: Some(phashes[i].as_str()),
            vector_id: Some(vector_ids[i] as i64),
            thumb_path: Some(thumb_paths[i].as_str()),
            width_px: Some(p.width_px as i64),
            height_px: Some(p.height_px as i64),
        })
        .collect();
    database::upsert_pages_batch(pool, file_id, &rows).await?;

    database::mark_file_indexed(pool, file_id, processed.page_count as i64).await?;
    tracing::debug!(
        "Indexed: {} ({} pages)",
        file.path.display(),
        processed.page_count
    );

    Ok(FileOutcome::Indexed)
}
