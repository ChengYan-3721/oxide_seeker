//! Search engine: orchestrates the pHash store + ANN vector search + ranking.

pub mod phash_store;
pub mod ranker;
pub mod vector_index;

pub use phash_store::PhashStore;
pub use ranker::SearchResult;
pub use vector_index::VectorIndex;

use crate::{
    config::SearchConfig,
    embedder::{
        phash::compute_phash,
        vision::{VisionEmbedder, VisionSession},
    },
    error::{AppError, Result},
    ocr::OcrEngine,
    storage::database::{self, DbPool},
};
use image::{DynamicImage, GenericImageView};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use vector_index::{VectorId, VectorMatch};

/// Fraction of the query image's shorter axis that the secondary "centre
/// crop" query keeps.  A value of 0.7 zooms into the central 70 × 70 % of the
/// frame — useful when the user uploaded a screenshot whose subject sits in
/// the middle of a large transparent / blank margin.  Combined with the
/// letterbox pre-processing this gives us two complementary recall channels
/// (whole-frame match + zoomed match) without any extra disk-side cost.
const CENTER_CROP_FRACTION: f32 = 0.7;

/// ANN over-fetch: regions retrieved per query vector before aggregation.
/// Ten region rows collapse to one page, so recalling K pages needs roughly
/// 10–25× K region hits; 500 keeps tail recall healthy at negligible cost
/// (HNSW at ef=256 returns 500 neighbours in tens of milliseconds).
const MIN_SEARCH_K: usize = 500;

/// Page-level candidates fetched from the FTS text channel.
const FTS_CANDIDATES: usize = 500;

/// Queries whose recognised text is shorter than this carry no useful text
/// signal (pure-graphics screenshots) — the FTS pass is skipped.
const MIN_QUERY_TEXT_CHARS: usize = 4;

/// Hard ceiling on query-image resolution.  Beyond this the request is
/// rejected as a client error rather than decoded — a guard against
/// accidental multi-hundred-megapixel uploads (scanner TIFFs, PDF exports)
/// that would OOM the decode.  60 MP comfortably covers any real screenshot
/// or phone photo while capping a full decode near ~240 MB.
const MAX_QUERY_PIXELS: u64 = 60_000_000;

/// Long edge every query is downscaled to before the pipeline runs.  The
/// encoder letterboxes to 224 px and OCR detection caps at 1280 px, so 1600
/// preserves all usable detail while keeping the several in-pipeline clones
/// cheap.
const QUERY_LONG_EDGE_PX: u32 = 1600;

/// Batch size when streaming stored vectors out of SQLite during an index
/// rebuild.  10 k × 384-dim f32 ≈ 15 MB per batch — small enough to keep
/// peak memory flat, large enough to amortise query overhead.
const REBUILD_BATCH: i64 = 10_000;

/// Shared search engine state — cheaply cloneable via `Arc` internals.
#[derive(Clone)]
pub struct SearchEngine {
    pool: DbPool,
    /// One dedicated encoder session for query-side inference.  Guarded by a
    /// `Mutex` because `ort::Session::run` needs `&mut self`; a single
    /// session is plenty for the low-QPS search path and uses all CPU cores
    /// for each individual request.
    session: Arc<Mutex<VisionSession>>,
    /// Query-side OCR engine; `None` disables the text channel entirely
    /// (models absent → pure visual search, exactly the pre-P4 behaviour).
    ocr: Option<Arc<Mutex<OcrEngine>>>,
    index: Arc<VectorIndex>,
    phash_store: Arc<PhashStore>,
    config: SearchConfig,
}

impl SearchEngine {
    pub fn new(
        pool: DbPool,
        embedder: &VisionEmbedder,
        ocr: Option<Arc<Mutex<OcrEngine>>>,
        index: Arc<VectorIndex>,
        phash_store: Arc<PhashStore>,
        config: SearchConfig,
    ) -> Result<Self> {
        // Single query session: give it every available CPU core so a single
        // request is fast, since they arrive one at a time over the LAN.
        let session = embedder.new_session(num_cpus::get())?;
        Ok(Self {
            pool,
            session: Arc::new(Mutex::new(session)),
            ocr,
            index,
            phash_store,
            config,
        })
    }

    /// Run a full image search pipeline:
    ///
    /// 1. Encode query image at two scales (whole image + centre crop) →
    ///    two embeddings in one batched forward
    /// 2. Compute the query pHash
    /// 3. ANN search per query vector, merge by best distance
    /// 4. Parallel Hamming scan of the in-memory pHash store
    /// 5. Aggregate by page, fuse signals, return Top-K
    pub async fn search_image(
        &self,
        img: &DynamicImage,
        top_k: Option<usize>,
    ) -> Result<SearchResponse> {
        let top_k = top_k.unwrap_or(self.config.default_top_k);
        let t0 = Instant::now();

        // ── Step 1: encoder and query-side OCR run in PARALLEL ─────────────
        // Both are CPU-heavy blocking stages (~400ms each on the target box);
        // overlapping them keeps the text channel nearly latency-free.
        let session = self.session.clone();
        let img_main = img.clone();
        let img_center = self
            .config
            .query_center_crop
            .then(|| center_crop(img, CENTER_CROP_FRACTION));
        let n_expected = 1 + img_center.is_some() as usize;
        let encode_task = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let mut guard = session.lock();
            let mut refs: Vec<&DynamicImage> = vec![&img_main];
            if let Some(c) = img_center.as_ref() {
                refs.push(c);
            }
            guard.encode_batch(&refs)
        });

        let ocr_task = self.ocr.clone().map(|ocr| {
            let img_ocr = img.clone();
            tokio::task::spawn_blocking(move || -> Result<String> {
                let mut guard = ocr.lock();
                guard.recognize_to_text(&img_ocr)
            })
        });

        let vectors = encode_task
            .await
            .map_err(|e| AppError::Other(anyhow::anyhow!("Encoder task panicked: {}", e)))??;
        if vectors.len() != n_expected {
            return Err(AppError::Other(anyhow::anyhow!(
                "Expected {} query vectors, got {}",
                n_expected,
                vectors.len()
            )));
        }
        let encode_ms = t0.elapsed().as_millis() as u64;

        // Join the OCR pass (usually already finished — it overlaps encode).
        // Both a Rust-level error AND a panic inside the OCR stack degrade to
        // visual-only search: the text channel is additive and must never be
        // able to fail a request (a det-postprocessing panic once aborted the
        // whole evaluation run through this join).
        let query_text = match ocr_task {
            Some(task) => match task.await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    tracing::warn!("Query OCR failed (continuing without text): {}", e);
                    String::new()
                }
                Err(join_err) => {
                    tracing::warn!(
                        "Query OCR panicked (continuing without text): {}",
                        join_err
                    );
                    String::new()
                }
            },
            None => String::new(),
        };
        let ocr_ms = (t0.elapsed().as_millis() as u64).saturating_sub(encode_ms);

        // ── Step 2: query pHash ────────────────────────────────────────────
        let t_phash = Instant::now();
        let img_clone2 = img.clone();
        let query_phash = tokio::task::spawn_blocking(move || compute_phash(&img_clone2))
            .await
            .map_err(|e| AppError::Other(anyhow::anyhow!("pHash task panicked: {}", e)))?;

        // ── Step 3: ANN search per query vector, then merge ────────────────
        let t_ann = Instant::now();
        let phash_hash_ms = t_ann.duration_since(t_phash).as_millis() as u64;
        let search_k = (top_k * 25).max(MIN_SEARCH_K);
        let index_ref = self.index.clone();
        let vector_matches = tokio::task::spawn_blocking(move || -> Result<Vec<VectorMatch>> {
            let mut all = Vec::with_capacity(vectors.len() * search_k);
            for v in &vectors {
                let m = index_ref.search(v, search_k)?;
                all.extend(m);
            }
            Ok(merge_matches(all))
        })
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!("Vector search task panicked: {}", e)))??;
        let ann_ms = t_ann.elapsed().as_millis() as u64;

        // ── Step 4: in-memory pHash scan (rayon inside) ─────────────────────
        let t_scan = Instant::now();
        let store = self.phash_store.clone();
        let threshold = self.config.phash_threshold;
        let phash_candidates =
            tokio::task::spawn_blocking(move || store.find_candidates(query_phash, threshold))
                .await
                .map_err(|e| AppError::Other(anyhow::anyhow!("pHash scan panicked: {}", e)))?;
        let phash_ms = phash_hash_ms + t_scan.elapsed().as_millis() as u64;

        // ── Step 4b: FTS text candidates from the query's recognised text ──
        let text_candidates: Vec<(i64, f32)> =
            if query_text.trim().chars().count() >= MIN_QUERY_TEXT_CHARS {
                match build_fts_query(&query_text) {
                    Some(expr) => {
                        let raw =
                            database::search_ocr(&self.pool, &expr, FTS_CANDIDATES as i64)
                                .await
                                .unwrap_or_else(|e| {
                                    tracing::warn!("FTS query failed: {}", e);
                                    vec![]
                                });
                        normalize_bm25(raw)
                    }
                    None => vec![],
                }
            } else {
                vec![]
            };

        // ── Step 5: aggregate + fuse + enrich ──────────────────────────────
        let t_rank = Instant::now();
        let results = ranker::rank_results(
            &self.pool,
            vector_matches,
            phash_candidates,
            text_candidates,
            top_k,
            self.config.similarity_threshold,
            ranker::FusionWeights {
                vector: self.config.weight_vector,
                phash: self.config.weight_phash,
                text: self.config.weight_text,
            },
        )
        .await?;
        let rank_ms = t_rank.elapsed().as_millis() as u64;

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        let timing = SearchTiming {
            encode_ms,
            ocr_ms,
            ann_ms,
            phash_ms,
            rank_ms,
        };
        tracing::info!(
            results = results.len(),
            elapsed_ms,
            encode_ms = timing.encode_ms,
            ocr_ms = timing.ocr_ms,
            ann_ms = timing.ann_ms,
            phash_ms = timing.phash_ms,
            rank_ms = timing.rank_ms,
            "Search completed"
        );

        Ok(SearchResponse {
            results,
            search_time_ms: elapsed_ms,
            timing,
        })
    }

    /// Convenience: search from raw image bytes (PNG/JPEG/WebP).
    ///
    /// Large queries are normalised here, at the single entry point both the
    /// upload and clipboard handlers funnel through:
    ///
    /// 1. Dimensions are probed from the header (no pixel allocation).
    ///    Anything past [`MAX_QUERY_PIXELS`] is rejected as a client error —
    ///    the default `image` decoder limits used to turn such files into
    ///    opaque 500s.
    /// 2. Within bounds, decoding runs with explicit limits sized to the
    ///    probe, then the image is downscaled to [`QUERY_LONG_EDGE_PX`].
    ///    Nothing downstream needs more (OCR det caps at 1280 px), and the
    ///    pipeline clones the query several times — a 100-megapixel original
    ///    would drag ~400 MB through encode/OCR/pHash for zero gain.
    pub async fn search_bytes(
        &self,
        data: &[u8],
        top_k: Option<usize>,
    ) -> Result<SearchResponse> {
        let cursor = std::io::Cursor::new(data);
        let reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| AppError::InvalidRequest(format!("Unrecognised image data: {}", e)))?;
        let (w, h) = reader
            .into_dimensions()
            .map_err(|e| AppError::InvalidRequest(format!("Cannot read image header: {}", e)))?;
        let pixels = w as u64 * h as u64;
        if pixels > MAX_QUERY_PIXELS {
            return Err(AppError::InvalidRequest(format!(
                "Query image is too large ({}×{} ≈ {} MP; limit {} MP). \
                 Please downscale before uploading.",
                w,
                h,
                pixels / 1_000_000,
                MAX_QUERY_PIXELS / 1_000_000
            )));
        }

        // Second reader for the actual decode (into_dimensions consumes the
        // first).  Limits are lifted — the pre-check above already bounds
        // the allocation, and the default limits are what produced the old
        // "large image → 500" failure.
        let cursor = std::io::Cursor::new(data);
        let mut reader = image::ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| AppError::InvalidRequest(format!("Unrecognised image data: {}", e)))?;
        reader.no_limits();
        let img = reader
            .decode()
            .map_err(|e| AppError::InvalidRequest(format!("Image decode failed: {}", e)))?;

        let img = if img.width().max(img.height()) > QUERY_LONG_EDGE_PX {
            let t = Instant::now();
            let scaled = img.resize(
                QUERY_LONG_EDGE_PX,
                QUERY_LONG_EDGE_PX,
                image::imageops::FilterType::Triangle,
            );
            tracing::info!(
                "Query downscaled {}×{} → {}×{} in {}ms",
                img.width(),
                img.height(),
                scaled.width(),
                scaled.height(),
                t.elapsed().as_millis()
            );
            scaled
        } else {
            img
        };

        self.search_image(&img, top_k).await
    }
}

/// Rebuild the HNSW graph from the vectors persisted in SQLite when the
/// on-disk dump is missing or clearly out of sync with the database.
///
/// This is what makes the ANN index a *cache*: deleting the `vectors.*`
/// files (corruption, hyperparameter change, tombstone compaction) costs a
/// stream through the `regions` table — roughly 15–25 minutes for 6 M
/// vectors — instead of a full re-embedding run.
pub async fn rebuild_index_if_needed(pool: &DbPool, index: &Arc<VectorIndex>) -> Result<()> {
    let db_count = database::count_regions_with_vectors(pool).await?;
    let graph_count = index.len() as i64;

    // The graph can legitimately trail the DB a little (crash between commit
    // and insert), but an empty/heavily-lagging graph means rebuild.
    if db_count == 0 || graph_count * 2 >= db_count {
        return Ok(());
    }

    tracing::warn!(
        "HNSW graph has {} points but the DB holds {} vectors — rebuilding from DB",
        graph_count,
        db_count
    );

    let t0 = Instant::now();
    let mut after_id = 0i64;
    let mut restored = 0usize;
    let mut skipped_nonfinite = 0usize;
    loop {
        let batch = database::load_vector_batch(pool, after_id, REBUILD_BATCH).await?;
        if batch.is_empty() {
            break;
        }
        after_id = batch.last().map(|(id, _)| *id).unwrap_or(after_id);
        // Drop non-finite vectors: one NaN inside the graph poisons every
        // later distance sort (non-total order → panic).  Legacy rows written
        // before the encoder-side sanitiser get scrubbed here — deleting the
        // dump files and letting this rebuild run IS the cleanup procedure.
        let insert: Vec<(VectorId, Vec<f32>)> = batch
            .into_iter()
            .filter(|(_, v)| {
                let ok = v.iter().all(|x| x.is_finite());
                if !ok {
                    skipped_nonfinite += 1;
                }
                ok
            })
            .map(|(id, v)| (id as VectorId, v))
            .collect();
        restored += insert.len();
        let idx = index.clone();
        tokio::task::spawn_blocking(move || idx.add_batch(&insert))
            .await
            .map_err(|e| AppError::Other(anyhow::anyhow!("rebuild task panicked: {}", e)))??;
        if restored % 100_000 < REBUILD_BATCH as usize {
            tracing::info!("HNSW rebuild progress: {}/{} vectors", restored, db_count);
        }
    }
    index.save()?;
    if skipped_nonfinite > 0 {
        tracing::warn!(
            "HNSW rebuild skipped {} non-finite vectors (poisoned rows from a \
             pre-sanitiser index run; affected regions rank by pHash/text only)",
            skipped_nonfinite
        );
    }
    tracing::info!(
        "HNSW rebuild complete: {} vectors in {:.1}s",
        restored,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Build an FTS5 MATCH expression from OCR'd query text: whitespace-split
/// segments of ≥3 chars (the trigram tokenizer's minimum), quoted and
/// escaped, OR-combined.  Returns `None` when nothing usable remains.
fn build_fts_query(text: &str) -> Option<String> {
    /// Cap on OR terms — a text-dense screenshot could otherwise produce a
    /// pathological query; the longest segments carry the most signal.
    const MAX_SEGMENTS: usize = 16;

    let mut segments: Vec<&str> = text
        .split_whitespace()
        .filter(|s| s.chars().count() >= 3)
        .collect();
    if segments.is_empty() {
        return None;
    }
    // Longest first — product codes and full names beat stray fragments.
    segments.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    segments.truncate(MAX_SEGMENTS);

    Some(
        segments
            .iter()
            .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// Min-max normalise SQLite bm25 ranks (lower = better) into `[0, 1]`
/// text scores (higher = better) within the candidate set.  When all
/// candidates rank equally (including the single-candidate case) they are
/// all best matches — score 1.0, not 0.
fn normalize_bm25(raw: Vec<(i64, f64)>) -> Vec<(i64, f32)> {
    if raw.is_empty() {
        return vec![];
    }
    let min = raw.iter().map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
    let max = raw.iter().map(|(_, r)| *r).fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;
    if span < 1e-9 {
        return raw.into_iter().map(|(id, _)| (id, 1.0)).collect();
    }
    raw.into_iter()
        .map(|(id, r)| (id, ((max - r) / span) as f32))
        .collect()
}

/// Crop the centre `frac × frac` region of `img`.  `frac` is clamped to
/// `(0.0, 1.0]`; `frac = 1.0` returns a clone of the whole image.
fn center_crop(img: &DynamicImage, frac: f32) -> DynamicImage {
    let frac = frac.clamp(0.05, 1.0);
    if frac >= 0.999 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }
    let new_w = ((w as f32 * frac).round() as u32).max(1);
    let new_h = ((h as f32 * frac).round() as u32).max(1);
    let x = (w - new_w) / 2;
    let y = (h - new_h) / 2;
    img.crop_imm(x, y, new_w, new_h)
}

/// Collapse multiple per-query [`VectorMatch`] lists into one, keeping the
/// best (lowest) distance for each `vector_id`.  Used to fuse hits from the
/// whole-image and centre-crop query passes — a vector that scores well for
/// either pass deserves to be considered a candidate.
fn merge_matches(matches: Vec<VectorMatch>) -> Vec<VectorMatch> {
    let mut best: HashMap<VectorId, VectorMatch> = HashMap::with_capacity(matches.len());
    for m in matches {
        match best.get(&m.vector_id) {
            Some(prev) if prev.distance <= m.distance => {}
            _ => {
                best.insert(m.vector_id, m);
            }
        }
    }
    best.into_values().collect()
}

/// Per-stage latency breakdown for one search, in milliseconds.  Serialised
/// with the response so the evaluation harness can aggregate percentiles per
/// stage; the web frontend simply ignores the extra field.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct SearchTiming {
    /// Query-image encoding (one or two scales, one batched forward).
    pub encode_ms: u64,
    /// Extra wall-clock spent waiting for query-side OCR *beyond* the encode
    /// stage it overlaps with (0 when OCR finishes first or is disabled).
    pub ocr_ms: u64,
    /// HNSW search over the query vector(s), including merge.
    pub ann_ms: u64,
    /// Query pHash computation + in-memory Hamming scan.
    pub phash_ms: u64,
    /// Aggregation, fusion scoring, and metadata fetch (SQLite).
    pub rank_ms: u64,
}

/// Response payload returned to HTTP handlers.
#[derive(Debug, serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub search_time_ms: u64,
    pub timing: SearchTiming,
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn solid(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, image::Rgb([200, 100, 50])))
    }

    #[test]
    fn center_crop_full_returns_clone() {
        let img = solid(100, 80);
        let c = center_crop(&img, 1.0);
        assert_eq!(c.dimensions(), img.dimensions());
    }

    #[test]
    fn center_crop_seven_tenths() {
        let img = solid(100, 80);
        let c = center_crop(&img, 0.7);
        assert_eq!(c.dimensions(), (70, 56));
    }

    #[test]
    fn merge_matches_keeps_best_distance_per_id() {
        let a = VectorMatch { vector_id: 1, distance: 0.5 };
        let b = VectorMatch { vector_id: 1, distance: 0.2 };
        let c = VectorMatch { vector_id: 2, distance: 0.4 };
        let merged = merge_matches(vec![a, b, c]);
        let by_id: HashMap<u64, f32> =
            merged.into_iter().map(|m| (m.vector_id, m.distance)).collect();
        assert!((by_id[&1] - 0.2).abs() < 1e-6);
        assert!((by_id[&2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn fts_query_filters_short_segments_and_escapes() {
        let q = build_fts_query("VUK1457-115T 金田 ab \"x\" 峨眉钰泉");
        let q = q.expect("has usable segments");
        assert!(q.contains("\"VUK1457-115T\""));
        assert!(q.contains("\"峨眉钰泉\""));
        assert!(!q.contains("\"ab\""), "2-char segment must be dropped: {}", q);
        assert!(q.contains(" OR "));
    }

    #[test]
    fn fts_query_empty_for_pure_graphics() {
        assert!(build_fts_query("ab x  ").is_none());
        assert!(build_fts_query("").is_none());
    }

    #[test]
    fn bm25_normalisation_maps_best_to_one() {
        // SQLite bm25 is lower-is-better (often negative).
        let out = normalize_bm25(vec![(1, -7.0), (2, -3.0), (3, -1.0)]);
        let m: HashMap<i64, f32> = out.into_iter().collect();
        assert!((m[&1] - 1.0).abs() < 1e-6, "best rank → 1.0");
        assert!((m[&3] - 0.0).abs() < 1e-6, "worst rank → 0.0");
        assert!(m[&2] > 0.0 && m[&2] < 1.0);
    }

    #[test]
    fn bm25_single_candidate_scores_one() {
        let out = normalize_bm25(vec![(9, -2.5)]);
        assert!((out[0].1 - 1.0).abs() < 1e-3);
    }

    /// A tiny PNG whose *header* claims a huge canvas is rejected before any
    /// pixel buffer is allocated — the guard is a client error (400-class),
    /// never the opaque 500 large uploads used to produce.  We build the
    /// engine-free normalisation path by calling the pixel check directly.
    #[test]
    fn oversized_query_is_rejected_by_pixel_guard() {
        // 10000×10000 = 100 MP > MAX_QUERY_PIXELS (60 MP).
        let pixels = 10_000u64 * 10_000;
        assert!(pixels > MAX_QUERY_PIXELS);
        // A within-limit 4K frame passes.
        assert!((3840u64 * 2160) < MAX_QUERY_PIXELS);
    }

    #[test]
    fn query_downscale_threshold_is_sane() {
        // The downscale target must exceed the OCR detector's 1280-px cap so
        // text detection still sees full detail after the query resize.
        assert!(QUERY_LONG_EDGE_PX > 1280);
    }
}
