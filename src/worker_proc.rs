//! Indexing-worker subprocess entry point.
//!
//! The same `oxide_seeker` binary, re-launched with [`crate::WORKER_MODE_FLAG`],
//! enters this loop instead of the main UI / web server.  Responsibilities:
//!
//! * Load pdfium and the vision-encoder ONNX session once.
//! * Read length-prefixed bincode frames from stdin.
//! * For each [`ProcessRequest`], render the PDF, embed every page **plus its
//!   overlapping tiles**, save the thumbnail to disk, and return a
//!   [`ProcessResponse`] with the data the parent needs to update its DB,
//!   HNSW index, and pHash store.
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
        phash::compute_phash,
        vision::{VisionEmbedder, VisionSession},
    },
    indexer::{
        pdf_processor::{self, RenderedPage},
        subprocess::{
            read_frame, write_frame, PageData, ProcessRequest, ProcessResponse, RegionBBox,
            RegionData, RegionKind, WorkerInit, WorkerRequest, WorkerResponse,
        },
    },
    ocr::OcrEngine,
    storage::thumbnail::{ThumbnailStore, THUMB_SIZE},
};
use image::DynamicImage;
use std::io::{BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};

/// Cap on how many crops are pushed to ONNX in a single forward pass.
///
/// With tiles enabled a single 30-page PDF expands to `30 × 10 = 300` crops.
/// Letting that all hit the model in one tensor would peak ≈ 180 MB just for
/// the input batch and starve the rest of the worker.  32 is small enough to
/// stay friendly on the smallest deployment profile while still amortising
/// most of the per-call ONNX overhead.
const ENCODE_BATCH_CHUNK: usize = 32;

/// Watchdog deadlines per request kind.  FFI code (pdfium, onnxruntime) can
/// hang without crashing — observed in production on a poison page rendered
/// at OCR resolution.  A hung child would block the parent's synchronous
/// pipe read forever, so the child self-destructs past the deadline and the
/// parent handles it like any crash (respawn, mark failed, move on).
const INDEX_DEADLINE_SECS: u64 = 600; // large multi-page files are slow but finite
const OCR_DEADLINE_SECS: u64 = 120; // one page render + detect + recognise

/// Unix-epoch deadline for the in-flight request; 0 = idle.
static WATCHDOG_DEADLINE: AtomicU64 = AtomicU64::new(0);

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn watchdog_arm(seconds: u64) {
    WATCHDOG_DEADLINE.store(now_epoch() + seconds, Ordering::Release);
}

fn watchdog_disarm() {
    WATCHDOG_DEADLINE.store(0, Ordering::Release);
}

fn spawn_watchdog(worker_idx: usize) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let deadline = WATCHDOG_DEADLINE.load(Ordering::Acquire);
        if deadline != 0 && now_epoch() > deadline {
            eprintln!(
                "[worker {}] watchdog: request exceeded its deadline — exiting so the parent can respawn",
                worker_idx
            );
            std::process::exit(3);
        }
    });
}

/// Overlapping-tile geometry: tile edge = ½ page, stride = ¼ page → 3
/// positions per axis, 9 tiles, adjacent tiles overlap by 50 %.
///
/// Coverage guarantee: any query region smaller than ¼ of the page area is
/// *fully contained* in at least one tile (nearest tile centre is at most ⅛
/// page away in each axis).  Regions between ¼ and ½ land mostly inside some
/// tile; anything larger is covered by the full-page embedding.  This is why
/// the non-overlapping grid of v1 was replaced: a target straddling a grid
/// boundary was split across tiles and matched none of them.
const TILE_POSITIONS: u32 = 3;
const TILE_FRACTION: f32 = 0.5;
const TILE_STRIDE: f32 = 0.25;

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
        "[worker {}] starting (model={}, thumbnails={}, tiles={})",
        init.worker_idx,
        init.model_path.display(),
        init.thumbnails_dir.display(),
        init.tiles_enabled,
    );

    // One pdfium instance per child; vision and OCR engines load lazily on
    // first use so an OCR-only worker never pays the encoder's footprint
    // (and vice versa).
    let pdfium = pdf_processor::init_pdfium()?;
    let embedder = VisionEmbedder::load(&init.model_path).ok();
    let mut vision_session: Option<VisionSession> = None;
    let mut ocr_engine: Option<OcrEngine> = None;
    let thumb_store = ThumbnailStore::new_sync(&init.thumbnails_dir)?;
    let tiles_enabled = init.tiles_enabled;

    spawn_watchdog(init.worker_idx);
    eprintln!("[worker {}] ready", init.worker_idx);

    // Request loop.  read_frame returns Ok(None) on clean EOF — parent closed
    // its end of stdin, time to exit.
    while let Some(req) = read_frame::<_, WorkerRequest>(&mut stdin_buf)? {
        let resp = match req {
            WorkerRequest::Index(req) => {
                watchdog_arm(INDEX_DEADLINE_SECS);
                // Breadcrumb before the FFI work so a watchdog exit still
                // tells the operator which file was in flight.
                eprintln!(
                    "[worker {}] index {}",
                    init.worker_idx,
                    req.file_path.display()
                );
                if vision_session.is_none() {
                    match embedder
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("vision model unavailable"))
                        .and_then(|e| e.new_session(1).map_err(|e| anyhow::anyhow!("{}", e)))
                    {
                        Ok(s) => vision_session = Some(s),
                        Err(e) => {
                            watchdog_disarm();
                            write_frame(
                                &mut stdout_buf,
                                &WorkerResponse::Index(ProcessResponse::Error {
                                    msg: format!("vision session init failed: {}", e),
                                }),
                            )?;
                            stdout_buf.flush()?;
                            continue;
                        }
                    }
                }
                let r = handle_one(
                    &req,
                    &pdfium,
                    vision_session.as_mut().expect("initialised above"),
                    &thumb_store,
                    tiles_enabled,
                );
                WorkerResponse::Index(r)
            }
            WorkerRequest::OcrPage {
                file_path,
                page_num,
            } => {
                watchdog_arm(OCR_DEADLINE_SECS);
                eprintln!(
                    "[worker {}] ocr {}#{}",
                    init.worker_idx,
                    file_path.display(),
                    page_num
                );
                if ocr_engine.is_none() {
                    match OcrEngine::load(&init.ocr_det_path, &init.ocr_rec_path) {
                        Ok(e) => ocr_engine = Some(e),
                        Err(e) => {
                            // Parent shouldn't send OCR work without models,
                            // but degrade to empty text anyway.
                            eprintln!(
                                "[worker {}] OCR engine load failed: {}",
                                init.worker_idx, e
                            );
                            watchdog_disarm();
                            write_frame(
                                &mut stdout_buf,
                                &WorkerResponse::OcrPage { text: String::new() },
                            )?;
                            stdout_buf.flush()?;
                            continue;
                        }
                    }
                }
                let text = ocr_page(
                    &pdfium,
                    ocr_engine.as_mut().expect("initialised above"),
                    &file_path,
                    page_num,
                )
                .unwrap_or_else(|e| {
                    eprintln!(
                        "[worker {}] OCR failed for {}#{}: {}",
                        init.worker_idx,
                        file_path.display(),
                        page_num,
                        e
                    );
                    String::new()
                });
                WorkerResponse::OcrPage { text }
            }
        };
        watchdog_disarm();
        write_frame(&mut stdout_buf, &resp)?;
        stdout_buf.flush()?;
    }

    Ok(())
}

/// Long-edge render size for OCR: small body text is unreadable at the
/// 640 px index-time render.
const OCR_RENDER_PX: i32 = 1280;

/// Render one page at OCR resolution and extract its text.
fn ocr_page(
    pdfium: &pdfium_render::prelude::Pdfium,
    engine: &mut OcrEngine,
    path: &std::path::Path,
    page_num: i64,
) -> anyhow::Result<String> {
    use pdfium_render::prelude::*;
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| anyhow::anyhow!("open {:?}: {}", path, e))?;
    let page = doc
        .pages()
        .get((page_num - 1).max(0) as u16)
        .map_err(|e| anyhow::anyhow!("page {} of {:?}: {}", page_num, path, e))?;

    let page_w = page.width().value as f64;
    let page_h = page.height().value as f64;
    let (tw, th) = if page_w <= 0.0 || page_h <= 0.0 {
        (OCR_RENDER_PX, OCR_RENDER_PX)
    } else {
        let scale = OCR_RENDER_PX as f64 / page_w.max(page_h);
        (
            ((page_w * scale).round() as i32).max(1),
            ((page_h * scale).round() as i32).max(1),
        )
    };
    let cfg = PdfRenderConfig::new()
        .set_target_width(tw)
        .set_target_height(th)
        .render_form_data(false)
        .render_annotations(false);
    let bitmap = page
        .render_with_config(&cfg)
        .map_err(|e| anyhow::anyhow!("render {:?} p{}: {}", path, page_num, e))?;
    let img = pdf_processor::bitmap_to_dynamic_image(&bitmap)?;

    Ok(engine.recognize_to_text(&img)?)
}

/// Process a single request, converting any Rust-level error into
/// [`ProcessResponse::Error`].  An unrecoverable FFI structured exception
/// terminates this function (and the whole process); the parent will see
/// the broken pipe and respawn.
fn handle_one(
    req: &ProcessRequest,
    pdfium: &pdfium_render::prelude::Pdfium,
    session: &mut VisionSession,
    thumb_store: &ThumbnailStore,
    tiles_enabled: bool,
) -> ProcessResponse {
    match render_and_embed(req, pdfium, session, thumb_store, tiles_enabled) {
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
    session: &mut VisionSession,
    thumb_store: &ThumbnailStore,
    tiles_enabled: bool,
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

    // ── Crop expansion ────────────────────────────────────────────────────
    // For each page, emit the full-page image followed by 9 overlapping
    // tiles.  We track per-page region offsets so we can split the flat
    // encoder output back into the right `RegionData` slots.
    let regions_per_page = if tiles_enabled {
        1 + (TILE_POSITIONS * TILE_POSITIONS) as usize
    } else {
        1
    };

    let mut crops: Vec<DynamicImage> =
        Vec::with_capacity(processed.pages.len() * regions_per_page);
    let mut metas: Vec<(RegionKind, u32, RegionBBox)> =
        Vec::with_capacity(processed.pages.len() * regions_per_page);
    let mut page_offsets: Vec<usize> = Vec::with_capacity(processed.pages.len() + 1);

    for p in &processed.pages {
        page_offsets.push(crops.len());
        push_full_region(&mut crops, &mut metas, p);
        if tiles_enabled {
            push_overlap_tiles(&mut crops, &mut metas, p);
        }
    }
    page_offsets.push(crops.len());

    // pHash: fast on CPU, sequential is fine.
    let phashes: Vec<u64> = crops.iter().map(compute_phash).collect();

    // Encoder: chunked batched forward to bound peak memory.
    let vectors = encode_in_chunks(session, &crops)?;
    if vectors.len() != crops.len() {
        return Err(anyhow::anyhow!(
            "Encoder batch size mismatch: expected {}, got {}",
            crops.len(),
            vectors.len()
        ));
    }

    // ── Reassemble per-page ─────────────────────────────────────────────────
    let mut pages = Vec::with_capacity(processed.pages.len());
    for (page_idx, p) in processed.pages.iter().enumerate() {
        let start = page_offsets[page_idx];
        let end = page_offsets[page_idx + 1];
        let mut regions = Vec::with_capacity(end - start);
        for global_idx in start..end {
            let (kind, index, bbox) = metas[global_idx];
            regions.push(RegionData {
                kind,
                index,
                bbox,
                phash: phashes[global_idx],
                vector: vectors[global_idx].clone(),
            });
        }
        pages.push(PageData {
            page_num: p.page_num,
            width_px: p.width_px,
            height_px: p.height_px,
            thumb_relative_path: thumb_paths[page_idx].clone(),
            regions,
        });
    }
    Ok(Some(pages))
}

fn push_full_region(
    crops: &mut Vec<DynamicImage>,
    metas: &mut Vec<(RegionKind, u32, RegionBBox)>,
    p: &RenderedPage,
) {
    crops.push(p.image.clone());
    metas.push((
        RegionKind::Full,
        0,
        RegionBBox {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        },
    ));
}

/// Cut the 3×3 overlapping tiles (edge = ½ page, stride = ¼ page).
fn push_overlap_tiles(
    crops: &mut Vec<DynamicImage>,
    metas: &mut Vec<(RegionKind, u32, RegionBBox)>,
    p: &RenderedPage,
) {
    let img_w = p.image.width();
    let img_h = p.image.height();
    if img_w == 0 || img_h == 0 {
        return;
    }
    for row in 0..TILE_POSITIONS {
        for col in 0..TILE_POSITIONS {
            let x_norm = col as f32 * TILE_STRIDE;
            let y_norm = row as f32 * TILE_STRIDE;
            // Pixel rect for this tile.  Round the far edge rather than the
            // size so the last tile always reaches the page border exactly.
            let x_px = (x_norm * img_w as f32).floor() as u32;
            let y_px = (y_norm * img_h as f32).floor() as u32;
            let end_x = ((x_norm + TILE_FRACTION).min(1.0) * img_w as f32).round() as u32;
            let end_y = ((y_norm + TILE_FRACTION).min(1.0) * img_h as f32).round() as u32;
            let w_px = end_x.saturating_sub(x_px).max(1);
            let h_px = end_y.saturating_sub(y_px).max(1);
            let tile = p.image.crop_imm(x_px, y_px, w_px, h_px);
            crops.push(tile);
            metas.push((
                RegionKind::Tile,
                row * TILE_POSITIONS + col,
                RegionBBox {
                    x: x_norm,
                    y: y_norm,
                    w: TILE_FRACTION,
                    h: TILE_FRACTION,
                },
            ));
        }
    }
}

/// Run encoder inference over `crops` in chunks of [`ENCODE_BATCH_CHUNK`] to
/// bound peak tensor memory on large multi-page documents.
fn encode_in_chunks(
    session: &mut VisionSession,
    crops: &[DynamicImage],
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(crops.len());
    for chunk in crops.chunks(ENCODE_BATCH_CHUNK) {
        let refs: Vec<&DynamicImage> = chunk.iter().collect();
        let vecs = session
            .encode_batch(&refs)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        out.extend(vecs);
    }
    Ok(out)
}
