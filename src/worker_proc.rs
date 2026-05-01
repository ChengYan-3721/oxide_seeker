//! Indexing-worker subprocess entry point.
//!
//! The same `oxide_seeker` binary, re-launched with [`crate::WORKER_MODE_FLAG`],
//! enters this loop instead of the main UI / web server.  Responsibilities:
//!
//! * Load pdfium and the CLIP ONNX session once.
//! * Read length-prefixed bincode frames from stdin.
//! * For each [`ProcessRequest`], render the PDF, embed every page, save the
//!   thumbnail to disk, and return a [`ProcessResponse`] with the data the
//!   parent needs to update its DB and HNSW index.
//! * Exit cleanly when stdin closes (parent shut down).
//!
//! Crash containment is the entire point: a structured exception inside
//! pdfium / onnxruntime kills only this process.  The parent observes the
//! broken pipe, logs the failure, increments the per-file `crash_attempts`
//! counter (via its own DB connection — the child never touches the DB),
//! and respawns a fresh subprocess for the next file.

use crate::{
    crash_handler,
    embedder::{
        clip::{ClipEmbedder, ClipSession},
        phash::compute_phash,
    },
    indexer::{
        pdf_processor,
        subprocess::{
            read_frame, write_frame, PageData, ProcessRequest, ProcessResponse, WorkerInit,
        },
    },
    storage::thumbnail::{ThumbnailStore, THUMB_SIZE},
};
use std::io::{BufReader, BufWriter, Write};

/// Synchronous entry point invoked from `main()` when `--worker-mode` is set.
pub fn run() -> anyhow::Result<()> {
    // Crash-log first so a SEH inside the loop still hits crash.log with a
    // distinguishing tag.
    crash_handler::install(std::path::Path::new("./logs"), "worker");

    // Stdin/stdout: take exclusive locks once so we can pass `&mut` references
    // to the framing helpers without re-locking per call.
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdin_buf = BufReader::new(stdin.lock());
    let mut stdout_buf = BufWriter::new(stdout.lock());

    // Initial frame from parent: model path + thumbnails dir + worker idx.
    let init: WorkerInit = match read_frame(&mut stdin_buf)? {
        Some(v) => v,
        None => {
            // Parent died before sending init; nothing to do.
            return Ok(());
        }
    };

    eprintln!(
        "[worker {}] starting (model={}, thumbnails={})",
        init.worker_idx,
        init.model_path.display(),
        init.thumbnails_dir.display(),
    );

    // One pdfium + one CLIP session per child.  Both are heavy to initialise
    // (~600ms for the ONNX session) so the long-lived child amortises that
    // cost across thousands of files.
    let pdfium = pdf_processor::init_pdfium()?;
    let clip = ClipEmbedder::load(&init.model_path)?;
    let mut clip_session = clip.new_session(1)?;
    let thumb_store = ThumbnailStore::new_sync(&init.thumbnails_dir)?;

    eprintln!("[worker {}] ready", init.worker_idx);

    // Request loop.  read_frame returns Ok(None) on clean EOF — parent closed
    // its end of stdin, time to exit.
    while let Some(req) = read_frame::<_, ProcessRequest>(&mut stdin_buf)? {
        let resp = handle_one(&req, &pdfium, &mut clip_session, &thumb_store);
        write_frame(&mut stdout_buf, &resp)?;
        stdout_buf.flush()?;
    }

    Ok(())
}

/// Process a single request, converting any Rust-level error into
/// [`ProcessResponse::Error`].  An unrecoverable FFI structured exception
/// terminates this function (and the whole process); the parent will see
/// the broken pipe and respawn.
fn handle_one(
    req: &ProcessRequest,
    pdfium: &pdfium_render::prelude::Pdfium,
    clip_session: &mut ClipSession,
    thumb_store: &ThumbnailStore,
) -> ProcessResponse {
    match render_and_embed(req, pdfium, clip_session, thumb_store) {
        Ok(Some(pages)) => ProcessResponse::Indexed { pages },
        Ok(None) => ProcessResponse::Excluded {
            reason: "imposition rule (XMP egExtFL:files)".to_string(),
        },
        Err(e) => ProcessResponse::Error { msg: e.to_string() },
    }
}

/// Returns `Ok(Some(pages))` for an indexable file, `Ok(None)` for a file
/// excluded by the imposition rule, `Err(_)` for a Rust-level error.
fn render_and_embed(
    req: &ProcessRequest,
    pdfium: &pdfium_render::prelude::Pdfium,
    clip_session: &mut ClipSession,
    thumb_store: &ThumbnailStore,
) -> anyhow::Result<Option<Vec<PageData>>> {
    let processed = pdf_processor::process_file(&req.file_path, pdfium)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if processed.excluded {
        return Ok(None);
    }
    if processed.pages.is_empty() {
        return Ok(Some(vec![]));
    }

    // Save thumbnails (sync; cheap per-image disk write).
    let mut thumb_paths = Vec::with_capacity(processed.pages.len());
    for p in &processed.pages {
        let t = thumb_store
            .save_thumbnail(&p.image, req.file_id, p.page_num as i64, THUMB_SIZE)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        thumb_paths.push(t);
    }

    // pHashes — sequential.
    let phashes: Vec<String> = processed
        .pages
        .iter()
        .map(|p| compute_phash(&p.image))
        .collect();

    // Single batched CLIP forward pass.
    let img_refs: Vec<&image::DynamicImage> =
        processed.pages.iter().map(|p| &p.image).collect();
    let vectors = clip_session
        .encode_batch(&img_refs)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if vectors.len() != processed.pages.len() {
        return Err(anyhow::anyhow!(
            "CLIP batch size mismatch: expected {}, got {}",
            processed.pages.len(),
            vectors.len()
        ));
    }

    let mut pages = Vec::with_capacity(processed.pages.len());
    for (i, p) in processed.pages.iter().enumerate() {
        pages.push(PageData {
            page_num: p.page_num,
            width_px: p.width_px,
            height_px: p.height_px,
            phash: phashes[i].clone(),
            vector: vectors[i].clone(),
            thumb_relative_path: thumb_paths[i].clone(),
        });
    }
    Ok(Some(pages))
}
