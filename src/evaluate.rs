//! Offline retrieval-quality evaluation harness (`--evaluate` CLI mode).
//!
//! Measures how well the search pipeline retrieves the *source page* of a
//! synthetic query screenshot.  Real query logs don't exist yet, so queries
//! are generated from the indexed corpus itself:
//!
//! 1. Sample N already-indexed pages from SQLite.
//! 2. Re-render each page with PDFium at a resolution *different* from the
//!    index-side render (default 900 px long edge vs. 512 px at index time),
//!    so query and index pixels never come from the same bitmap.
//! 3. Cut M random crops per page (area 8–90 % of the page, jittered aspect
//!    ratio) and degrade them like a real screenshot: random rescale,
//!    brightness shift, JPEG round-trip.
//! 4. Run each crop through the full [`SearchEngine`] pipeline and record the
//!    rank of the source page.
//!
//! Reported metrics: Recall@1/5/10, MRR, latency P50/P95 — overall and
//! bucketed by crop area (small/medium/large), since partial-screenshot
//! recall is the primary product concern.
//!
//! The JSON report is written next to the CWD so runs can be diffed across
//! model/parameter changes (`--label` tags the run).
//!
//! Usage:
//! ```text
//! oxide_seeker --evaluate [--config config.toml] [--samples 300]
//!              [--queries-per-page 3] [--seed 42] [--render-px 900]
//!              [--label clip-baseline] [--out eval_report.json]
//! ```

use crate::{
    config::Config,
    embedder::vision::VisionEmbedder,
    indexer::pdf_processor,
    search::{PhashStore, SearchEngine, VectorIndex},
    storage::database,
};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, GenericImageView};
use pdfium_render::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::path::{Path, PathBuf};

// ── CLI options ──────────────────────────────────────────────────────────────

struct EvalOpts {
    config_path: PathBuf,
    samples: usize,
    queries_per_page: usize,
    seed: u64,
    render_px: i32,
    label: String,
    out_path: PathBuf,
}

impl EvalOpts {
    fn parse() -> Self {
        let mut opts = Self {
            config_path: PathBuf::from("config.toml"),
            samples: 300,
            queries_per_page: 3,
            seed: 42,
            render_px: 900,
            label: "unlabeled".to_string(),
            out_path: PathBuf::from("eval_report.json"),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut take = |target: &mut String| {
                if let Some(v) = args.next() {
                    *target = v;
                }
            };
            let mut s = String::new();
            match arg.as_str() {
                "--config" => {
                    take(&mut s);
                    opts.config_path = PathBuf::from(&s);
                }
                "--samples" => {
                    take(&mut s);
                    opts.samples = s.parse().unwrap_or(opts.samples);
                }
                "--queries-per-page" => {
                    take(&mut s);
                    opts.queries_per_page = s.parse().unwrap_or(opts.queries_per_page);
                }
                "--seed" => {
                    take(&mut s);
                    opts.seed = s.parse().unwrap_or(opts.seed);
                }
                "--render-px" => {
                    take(&mut s);
                    opts.render_px = s.parse().unwrap_or(opts.render_px);
                }
                "--label" => {
                    take(&mut s);
                    opts.label = s.clone();
                }
                "--out" => {
                    take(&mut s);
                    opts.out_path = PathBuf::from(&s);
                }
                _ => {}
            }
        }
        opts
    }
}

// ── Synthetic query generation ───────────────────────────────────────────────

/// Crop-area buckets used for reporting.  Partial screenshots (small crops)
/// are the primary use case, so quality regressions there must be visible
/// separately from easy whole-page queries.
const BUCKET_SMALL_MAX: f32 = 0.25;
const BUCKET_MEDIUM_MAX: f32 = 0.60;

fn bucket_name(area_ratio: f32) -> &'static str {
    if area_ratio < BUCKET_SMALL_MAX {
        "small(<25%)"
    } else if area_ratio < BUCKET_MEDIUM_MAX {
        "medium(25-60%)"
    } else {
        "large(>60%)"
    }
}

/// One synthetic query: a degraded random crop plus its ground truth.
struct SynthQuery {
    image: DynamicImage,
    /// Actual crop area as a fraction of the source page.
    area_ratio: f32,
}

/// Cut a random crop out of `page` and degrade it like a real screenshot.
fn synth_query(rng: &mut StdRng, page: &DynamicImage) -> SynthQuery {
    let (w, h) = page.dimensions();

    // Target area fraction — biased towards partial crops, which matches the
    // "mostly local screenshots" usage reported for this deployment.
    let area: f32 = rng.gen_range(0.08..0.90);
    let aspect_jitter: f32 = rng.gen_range(0.75..1.3333);

    // Edge lengths whose product ≈ area × (w × h), warped by aspect jitter.
    let base = (area as f64 * w as f64 * h as f64).sqrt();
    let cw = ((base * (aspect_jitter as f64).sqrt()).round() as u32).clamp(16, w);
    let ch = ((base / (aspect_jitter as f64).sqrt()).round() as u32).clamp(16, h);

    let x = if w > cw { rng.gen_range(0..=w - cw) } else { 0 };
    let y = if h > ch { rng.gen_range(0..=h - ch) } else { 0 };
    let crop = page.crop_imm(x, y, cw, ch);
    let actual_area = (cw as f32 * ch as f32) / (w as f32 * h as f32);

    // Rescale to a plausible screenshot size (long edge 320–900 px).
    let target_long: u32 = rng.gen_range(320..=900);
    let scale = target_long as f32 / cw.max(ch) as f32;
    let nw = ((cw as f32 * scale).round() as u32).max(16);
    let nh = ((ch as f32 * scale).round() as u32).max(16);
    let resized = crop.resize_exact(nw, nh, image::imageops::FilterType::Triangle);

    // Brightness shift ±10 % of full scale.
    let brightened = resized.brighten(rng.gen_range(-25..=25));

    // JPEG round-trip (quality 70–90) — screenshot apps and chat clients
    // recompress aggressively; this is the single most realistic degradation.
    let quality: u8 = rng.gen_range(70..=90);
    let degraded = jpeg_roundtrip(&brightened, quality).unwrap_or(brightened);

    SynthQuery {
        image: degraded,
        area_ratio: actual_area,
    }
}

fn jpeg_roundtrip(img: &DynamicImage, quality: u8) -> Option<DynamicImage> {
    let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
    let mut buf = Vec::new();
    rgb.write_with_encoder(JpegEncoder::new_with_quality(&mut buf, quality))
        .ok()?;
    image::load_from_memory(&buf).ok()
}

// ── Page rendering ───────────────────────────────────────────────────────────

/// Render a single page at `target_px` long edge.  Mirrors the sizing logic
/// of `pdf_processor::process_file` but renders only the requested page and
/// at an evaluation-specific resolution.
fn render_single_page(
    pdfium: &Pdfium,
    path: &Path,
    page_num: i64,
    target_px: i32,
) -> anyhow::Result<DynamicImage> {
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| anyhow::anyhow!("open {:?}: {}", path, e))?;
    let page = doc
        .pages()
        .get((page_num - 1).max(0) as u16)
        .map_err(|e| anyhow::anyhow!("page {} of {:?}: {}", page_num, path, e))?;

    let page_w_pt = page.width().value as f64;
    let page_h_pt = page.height().value as f64;
    let (tw, th) = if page_w_pt <= 0.0 || page_h_pt <= 0.0 {
        (target_px, target_px)
    } else {
        let scale = target_px as f64 / page_w_pt.max(page_h_pt);
        (
            ((page_w_pt * scale).round() as i32).max(1),
            ((page_h_pt * scale).round() as i32).max(1),
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
    Ok(pdf_processor::bitmap_to_dynamic_image(&bitmap)?)
}

// ── Metrics ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct BucketMetrics {
    queries: usize,
    recall_at_1: f32,
    recall_at_5: f32,
    recall_at_10: f32,
    mrr: f32,
}

#[derive(Default)]
struct BucketAccum {
    ranks: Vec<Option<usize>>,
}

impl BucketAccum {
    fn push(&mut self, rank: Option<usize>) {
        self.ranks.push(rank);
    }

    fn finish(&self) -> BucketMetrics {
        let n = self.ranks.len().max(1) as f32;
        let hits_at = |k: usize| {
            self.ranks.iter().filter(|r| matches!(r, Some(v) if *v <= k)).count() as f32 / n
        };
        let mrr = self
            .ranks
            .iter()
            .map(|r| r.map(|v| 1.0 / v as f32).unwrap_or(0.0))
            .sum::<f32>()
            / n;
        BucketMetrics {
            queries: self.ranks.len(),
            recall_at_1: hits_at(1),
            recall_at_5: hits_at(5),
            recall_at_10: hits_at(10),
            mrr,
        }
    }
}

#[derive(Debug, Serialize)]
struct MissRecord {
    path: String,
    page_num: i64,
    area_ratio: f32,
}

/// P50/P95 pair for one pipeline stage.
#[derive(Debug, Serialize)]
struct StageLatency {
    p50: u64,
    p95: u64,
}

/// Similarity-score distributions used to calibrate `similarity_threshold`.
///
/// A good threshold sits between the *hit floor* (below which real targets
/// start getting cut) and the *noise ceiling* (top-1 scores of queries whose
/// target was never found — pure lookalike noise).  `suggested` keeps ≥95 %
/// of hits while trimming as much noise as that allows.
#[derive(Debug, Serialize)]
struct ScoreCalibration {
    /// Similarity of the true target when it was found, percentiles over hits.
    hit_target_sim_p05: f32,
    hit_target_sim_p25: f32,
    hit_target_sim_p50: f32,
    /// Top-1 similarity of queries whose target was NOT in the results.
    miss_top1_sim_p50: f32,
    miss_top1_sim_p95: f32,
    /// Threshold that preserves 95 % of hits (= hit_target_sim_p05).
    suggested_threshold: f32,
}

fn score_percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Collects per-stage timings across all queries and reduces to percentiles.
#[derive(Default)]
struct StageAccum {
    encode: Vec<u64>,
    ocr: Vec<u64>,
    ann: Vec<u64>,
    phash: Vec<u64>,
    rank: Vec<u64>,
}

impl StageAccum {
    fn push(&mut self, t: &crate::search::SearchTiming) {
        self.encode.push(t.encode_ms);
        self.ocr.push(t.ocr_ms);
        self.ann.push(t.ann_ms);
        self.phash.push(t.phash_ms);
        self.rank.push(t.rank_ms);
    }

    fn finish(mut self) -> std::collections::BTreeMap<String, StageLatency> {
        let mut out = std::collections::BTreeMap::new();
        for (name, v) in [
            ("encode", &mut self.encode),
            ("ocr", &mut self.ocr),
            ("ann", &mut self.ann),
            ("phash", &mut self.phash),
            ("rank", &mut self.rank),
        ] {
            v.sort_unstable();
            out.insert(
                name.to_string(),
                StageLatency {
                    p50: percentile(v, 0.50),
                    p95: percentile(v, 0.95),
                },
            );
        }
        out
    }
}

#[derive(Debug, Serialize)]
struct EvalReport {
    label: String,
    timestamp: String,
    model_path: String,
    similarity_threshold: f32,
    phash_threshold: u32,
    /// Whether the second (centre-crop) query vector was enabled — affects
    /// both latency and recall, so it must be visible when diffing runs.
    query_center_crop: bool,
    samples_requested: usize,
    pages_rendered: usize,
    pages_skipped: usize,
    queries_per_page: usize,
    seed: u64,
    render_px: i32,
    overall: BucketMetrics,
    buckets: std::collections::BTreeMap<String, BucketMetrics>,
    latency_ms_p50: u64,
    latency_ms_p95: u64,
    latency_ms_mean: u64,
    /// Per-stage latency breakdown (encode / ann / phash / rank), so a slow
    /// P50 can be attributed without re-running under a profiler.
    stage_latency_ms: std::collections::BTreeMap<String, StageLatency>,
    /// Similarity-score distributions for threshold calibration.
    score_calibration: ScoreCalibration,
    /// First 50 misses for manual inspection.
    misses: Vec<MissRecord>,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Entry point ──────────────────────────────────────────────────────────────

/// Run the evaluation harness.  Called from `main` when `--evaluate` is
/// present; never starts the indexer, watcher, or web server.
pub async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let opts = EvalOpts::parse();
    let config = Config::load(&opts.config_path)?;

    tracing::info!(
        "Evaluation run '{}': {} pages × {} queries, seed={}, render_px={}",
        opts.label,
        opts.samples,
        opts.queries_per_page,
        opts.seed,
        opts.render_px
    );

    // ── Bring up the search stack (no indexer / watcher / web) ────────────
    let pool = database::init_pool(&config.db_path()).await?;
    let vector_index = VectorIndex::open(&config.vector_index_path())?;
    crate::search::rebuild_index_if_needed(&pool, &vector_index).await?;
    let phash_store = std::sync::Arc::new(PhashStore::new());
    phash_store.replace_all(database::load_phash_entries(&pool).await?);
    let embedder = VisionEmbedder::load(&config.paths.model_path)?;
    // OCR mirrors production wiring: enabled when the models are present.
    // Note the text channel only contributes once the server-side backfill
    // has coverage — evaluating against a freshly-migrated DB measures the
    // visual channels alone.
    let ocr_engine = match crate::ocr::OcrEngine::load(
        &config.paths.ocr_det_path,
        &config.paths.ocr_rec_path,
    ) {
        Ok(e) => {
            tracing::info!("evaluation: OCR text channel enabled");
            Some(std::sync::Arc::new(parking_lot::Mutex::new(e)))
        }
        Err(e) => {
            tracing::warn!("evaluation: OCR text channel disabled ({})", e);
            None
        }
    };
    let engine = SearchEngine::new(
        pool.clone(),
        &embedder,
        ocr_engine,
        vector_index.clone(),
        phash_store,
        config.search.clone(),
    )?;

    // ── Sample ground-truth pages ──────────────────────────────────────────
    // Pull the full candidate list in a DETERMINISTIC order and sample with
    // the seeded RNG.  The first version used SQL `ORDER BY RANDOM()`, which
    // ignores --seed: two runs compared different page sets, so small
    // parameter deltas drowned in ±few-percent sampling noise.  With the
    // seeded shuffle, identical (seed, corpus) → identical queries.
    let mut rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT f.path, p.page_num FROM pages p \
         JOIN files f ON f.id = p.file_id \
         WHERE f.is_excluded = 0 AND f.indexed_at IS NOT NULL \
         ORDER BY f.path, p.page_num",
    )
    .fetch_all(&pool)
    .await?;

    let mut rng = StdRng::seed_from_u64(opts.seed);
    {
        use rand::seq::SliceRandom;
        rows.shuffle(&mut rng);
    }
    rows.truncate(opts.samples);

    if rows.is_empty() {
        anyhow::bail!(
            "No indexed pages found in {} — run the indexer first",
            config.db_path().display()
        );
    }
    tracing::info!("Sampled {} pages from the index", rows.len());

    let pdfium = pdf_processor::init_pdfium()?;

    // ── Render → synthesize → search → record rank ────────────────────────
    let mut overall = BucketAccum::default();
    let mut buckets: std::collections::BTreeMap<&'static str, BucketAccum> =
        std::collections::BTreeMap::new();
    let mut latencies: Vec<u64> = Vec::new();
    let mut stages = StageAccum::default();
    let mut hit_target_sims: Vec<f32> = Vec::new();
    let mut miss_top1_sims: Vec<f32> = Vec::new();
    let mut misses: Vec<MissRecord> = Vec::new();
    let mut pages_rendered = 0usize;
    let mut pages_skipped = 0usize;

    for (i, (path, page_num)) in rows.iter().enumerate() {
        let page_img = match render_single_page(&pdfium, Path::new(path), *page_num, opts.render_px)
        {
            Ok(img) => img,
            Err(e) => {
                // Files move / get deleted between indexing and evaluation —
                // skip rather than abort, but keep count.
                tracing::debug!("skip {}#{}: {}", path, page_num, e);
                pages_skipped += 1;
                continue;
            }
        };
        pages_rendered += 1;

        for _ in 0..opts.queries_per_page {
            let q = synth_query(&mut rng, &page_img);
            let response = engine.search_image(&q.image, Some(10)).await?;
            latencies.push(response.search_time_ms);
            stages.push(&response.timing);

            let rank = response
                .results
                .iter()
                .position(|r| r.file_path == *path && r.page_num == *page_num)
                .map(|p| p + 1);

            // Score-distribution samples for threshold calibration.
            match rank {
                Some(r) => hit_target_sims.push(response.results[r - 1].similarity),
                None => {
                    if let Some(top1) = response.results.first() {
                        miss_top1_sims.push(top1.similarity);
                    }
                }
            }

            overall.push(rank);
            buckets
                .entry(bucket_name(q.area_ratio))
                .or_default()
                .push(rank);

            if rank.is_none() && misses.len() < 50 {
                misses.push(MissRecord {
                    path: path.clone(),
                    page_num: *page_num,
                    area_ratio: q.area_ratio,
                });
            }
        }

        if (i + 1) % 25 == 0 {
            tracing::info!("progress: {}/{} pages", i + 1, rows.len());
        }
    }

    // ── Report ─────────────────────────────────────────────────────────────
    latencies.sort_unstable();
    let mean = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / latencies.len() as u64
    };

    hit_target_sims.sort_unstable_by(f32::total_cmp);
    miss_top1_sims.sort_unstable_by(f32::total_cmp);
    let score_calibration = ScoreCalibration {
        hit_target_sim_p05: score_percentile(&hit_target_sims, 0.05),
        hit_target_sim_p25: score_percentile(&hit_target_sims, 0.25),
        hit_target_sim_p50: score_percentile(&hit_target_sims, 0.50),
        miss_top1_sim_p50: score_percentile(&miss_top1_sims, 0.50),
        miss_top1_sim_p95: score_percentile(&miss_top1_sims, 0.95),
        suggested_threshold: score_percentile(&hit_target_sims, 0.05),
    };

    let report = EvalReport {
        label: opts.label.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_path: config.paths.model_path.display().to_string(),
        similarity_threshold: config.search.similarity_threshold,
        phash_threshold: config.search.phash_threshold,
        query_center_crop: config.search.query_center_crop,
        samples_requested: opts.samples,
        pages_rendered,
        pages_skipped,
        queries_per_page: opts.queries_per_page,
        seed: opts.seed,
        render_px: opts.render_px,
        overall: overall.finish(),
        buckets: buckets
            .iter()
            .map(|(k, v)| (k.to_string(), v.finish()))
            .collect(),
        latency_ms_p50: percentile(&latencies, 0.50),
        latency_ms_p95: percentile(&latencies, 0.95),
        latency_ms_mean: mean,
        stage_latency_ms: stages.finish(),
        score_calibration,
        misses,
    };

    print_report(&report);
    std::fs::write(&opts.out_path, serde_json::to_string_pretty(&report)?)?;
    tracing::info!("Report written to {}", opts.out_path.display());

    Ok(())
}

fn print_report(r: &EvalReport) {
    println!("\n═══ Evaluation: {} ═══", r.label);
    println!(
        "pages rendered/skipped: {}/{}   queries: {}",
        r.pages_rendered, r.pages_skipped, r.overall.queries
    );
    println!(
        "latency ms  p50={}  p95={}  mean={}",
        r.latency_ms_p50, r.latency_ms_p95, r.latency_ms_mean
    );
    for (name, s) in &r.stage_latency_ms {
        println!("  stage {:<8} p50={:>5}  p95={:>5}", name, s.p50, s.p95);
    }
    let c = &r.score_calibration;
    println!(
        "score calib  hit-target p05/p25/p50 = {:.3}/{:.3}/{:.3}   miss-top1 p50/p95 = {:.3}/{:.3}",
        c.hit_target_sim_p05, c.hit_target_sim_p25, c.hit_target_sim_p50,
        c.miss_top1_sim_p50, c.miss_top1_sim_p95
    );
    println!("suggested similarity_threshold = {:.3}", c.suggested_threshold);
    println!("\n{:<16} {:>8} {:>6} {:>6} {:>6} {:>6}", "bucket", "queries", "R@1", "R@5", "R@10", "MRR");
    let row = |name: &str, m: &BucketMetrics| {
        println!(
            "{:<16} {:>8} {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
            name, m.queries, m.recall_at_1, m.recall_at_5, m.recall_at_10, m.mrr
        );
    };
    row("overall", &r.overall);
    for (name, m) in &r.buckets {
        row(name, m);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn gradient(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        }))
    }

    #[test]
    fn synth_query_dimensions_and_area() {
        let page = gradient(900, 640);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let q = synth_query(&mut rng, &page);
            assert!(q.area_ratio > 0.0 && q.area_ratio <= 1.0);
            let (w, h) = q.image.dimensions();
            assert!(w >= 16 && h >= 16);
            assert!(w <= 1100 && h <= 1100, "resize target should bound size");
        }
    }

    #[test]
    fn synth_query_is_deterministic_per_seed() {
        let page = gradient(400, 300);
        let mut a = StdRng::seed_from_u64(99);
        let mut b = StdRng::seed_from_u64(99);
        let qa = synth_query(&mut a, &page);
        let qb = synth_query(&mut b, &page);
        assert_eq!(qa.area_ratio, qb.area_ratio);
        assert_eq!(qa.image.dimensions(), qb.image.dimensions());
    }

    #[test]
    fn bucket_boundaries() {
        assert_eq!(bucket_name(0.10), "small(<25%)");
        assert_eq!(bucket_name(0.40), "medium(25-60%)");
        assert_eq!(bucket_name(0.80), "large(>60%)");
    }

    #[test]
    fn percentile_basics() {
        let v = vec![10, 20, 30, 40, 100];
        assert_eq!(percentile(&v, 0.5), 30);
        assert_eq!(percentile(&v, 0.95), 100);
        assert_eq!(percentile(&[], 0.5), 0);
    }
}
