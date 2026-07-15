//! Background OCR backfill: renders and recognises pages that lack a
//! `page_ocr` row, through a crash-isolated worker subprocess.
//!
//! Design notes
//! ------------
//! * **Subprocess, not in-process.**  The first deployment ran pdfium + OCR
//!   on a thread inside the server and hung forever on page 8 — an FFI call
//!   that neither returned nor crashed.  All FFI work now goes through the
//!   same `--worker-mode` child the indexer uses; the child carries a
//!   watchdog that `exit(3)`s past a per-request deadline, which surfaces
//!   here as a broken pipe → record the page as done (empty text), respawn,
//!   move on.  No page can pin the queue.
//! * **The loop is the only OCR producer.**  New pages are not OCR'd inside
//!   the indexing pipeline; they surface here through the missing-row query.
//!   One code path, uniform for initial backfill and steady-state.
//! * A `pages.id` cursor keeps the claim query cheap as coverage grows; it
//!   resets when a sweep drains so new/re-indexed pages get picked up.

use crate::{
    config::Config,
    indexer::subprocess::WorkerProcess,
    storage::database::{self, DbPool},
};
use std::path::Path;
use std::sync::Arc;

/// Pages claimed per DB round-trip.
const BATCH: i64 = 200;
/// Idle poll interval once the queue is drained.
const IDLE_POLL_SECS: u64 = 300;

/// Spawn the backfill loop on a dedicated blocking thread.  Returns
/// immediately; the loop runs for the process lifetime.
pub fn spawn(config: Arc<Config>, pool: DbPool) {
    tokio::task::spawn_blocking(move || {
        // Own runtime for the async DB calls, same pattern as worker_pool.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("OCR backfill runtime");

        let spawn_worker = || {
            WorkerProcess::spawn(
                &config.paths.model_path,
                &config.thumbnails_dir(),
                &config.paths.ocr_det_path,
                &config.paths.ocr_rec_path,
                usize::MAX, // log tag distinguishing the OCR worker from index workers
                config.indexer.tiles_enabled,
            )
        };
        let mut worker = match spawn_worker() {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("OCR backfill disabled (worker spawn failed: {}).", e);
                return;
            }
        };

        tracing::info!("OCR backfill loop started (subprocess-isolated)");
        let mut after_id = 0i64;
        // Tracks the busy→idle transition so completion is logged exactly
        // once per sweep — at warn level, because the rolling log file only
        // records warn+ and this is the operator's "backfill done" signal.
        let mut was_busy = false;
        loop {
            let batch = match rt.block_on(database::claim_ocr_batch(&pool, after_id, BATCH)) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("OCR backfill claim failed: {}", e);
                    std::thread::sleep(std::time::Duration::from_secs(60));
                    continue;
                }
            };
            if batch.is_empty() {
                // Sweep drained.  Mark pages the claim query can never reach
                // (excluded / unindexed files that still own pages rows) so
                // the pending counter converges to zero instead of stalling.
                match rt.block_on(database::mark_orphan_pages_ocr_done(&pool)) {
                    Ok(n) if n > 0 => {
                        tracing::info!("OCR backfill: marked {} orphan pages as done", n)
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!("OCR orphan-page marking failed: {}", e),
                }
                if was_busy {
                    was_busy = false;
                    match rt.block_on(database::ocr_progress(&pool)) {
                        Ok((done, pending)) => tracing::warn!(
                            "OCR backfill sweep complete: {} pages done, {} pending",
                            done,
                            pending
                        ),
                        Err(_) => tracing::warn!("OCR backfill sweep complete"),
                    }
                }
                after_id = 0;
                std::thread::sleep(std::time::Duration::from_secs(IDLE_POLL_SECS));
                continue;
            }
            was_busy = true;
            after_id = batch.last().map(|(id, _, _)| *id).unwrap_or(after_id);

            let mut done = 0usize;
            for (page_id, path, page_num) in &batch {
                let text = match worker.process_ocr(Path::new(path), *page_num) {
                    Ok(t) => t,
                    Err(e) => {
                        // Child died (crash or watchdog timeout).  Record the
                        // page as done-with-empty-text so it cannot pin the
                        // queue, then respawn for the rest of the batch.
                        tracing::warn!(
                            "OCR worker died on {}#{} ({}); recording empty text and respawning",
                            path,
                            page_num,
                            e
                        );
                        worker = match spawn_worker() {
                            Ok(w) => w,
                            Err(se) => {
                                tracing::warn!(
                                    "OCR backfill stopping (worker respawn failed: {})",
                                    se
                                );
                                return;
                            }
                        };
                        String::new()
                    }
                };
                if let Err(e) = rt.block_on(database::upsert_page_ocr(&pool, *page_id, &text)) {
                    tracing::warn!("OCR result write failed for page {}: {}", page_id, e);
                } else {
                    done += 1;
                }
            }
            tracing::info!(
                "OCR backfill: +{} pages this batch (cursor at page {})",
                done,
                after_id
            );
        }
    });
}
