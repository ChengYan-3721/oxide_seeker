//! Vision encoder inference via ONNX Runtime (DINOv2 ViT-S/14 by default).
//!
//! The engine is model-agnostic: any ONNX image encoder with input
//! `pixel_values` `[N, 3, 224, 224]` (f32, ImageNet-normalised — see
//! `image_prep`) and a `[N, D]` embedding output works.  The embedding
//! dimension `D` is inferred from the output tensor at run time, so swapping
//! `dinov2_vits14.onnx` (384-d) for `dinov2_vitb14.onnx` (768-d) requires a
//! full re-index but no code change.
//!
//! Two types, clean split:
//!
//! * [`VisionEmbedder`] — a cheap, `Clone`-able *factory* that holds the
//!   model path.  Share it freely via `Arc`.
//! * [`VisionSession`] — an *owned* ONNX Runtime session.  `run()` requires
//!   `&mut self`, so a session is a single-thread resource: the indexer
//!   workers each create their own session (true parallel inference), and
//!   the search engine guards one session behind a `Mutex` (low QPS).
//!
//! # Model export (one-time setup)
//! ```bash
//! python scripts/export_dinov2.py            # → models/dinov2_vits14.onnx
//! #                                            + models/dinov2_vits14_int8.onnx
//! ```
//! Copy the chosen file next to the exe and point `paths.model_path` at it.
//! The INT8 file loads identically and runs 2-3× faster on CPU.

use crate::{
    embedder::image_prep::{preprocess_for_model, MODEL_INPUT_SIZE},
    error::{AppError, Result},
};
use image::DynamicImage;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::{Path, PathBuf};

// ── Factory ──────────────────────────────────────────────────────────────────

/// Cheap handle that knows where the encoder model lives.
///
/// Cloning and `Arc`-sharing is free; each caller that needs to run
/// inference asks the embedder for a fresh [`VisionSession`].
#[derive(Clone)]
pub struct VisionEmbedder {
    model_path: PathBuf,
}

impl VisionEmbedder {
    /// Validate the model file and return a factory.
    ///
    /// No session is kept on `VisionEmbedder` — eager validation opens a
    /// probe session and drops it, so callers get an early failure with a
    /// useful error message instead of a mysterious panic at first inference.
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(AppError::Config(format!(
                "Vision model not found at {}. \
                 Export with: python scripts/export_dinov2.py \
                 then copy models/dinov2_vits14.onnx → {}",
                model_path.display(),
                model_path.display()
            )));
        }

        tracing::info!("Validating vision model at {}", model_path.display());
        // Probe — throw-away session; confirms the model is readable and the
        // ort dylib resolved correctly.
        let _probe = build_session(model_path, 1)?;
        tracing::info!("Vision model validated");

        Ok(Self {
            model_path: model_path.to_path_buf(),
        })
    }

    /// Allocate a new, independent inference session.
    ///
    /// `intra_threads = 0` picks `num_cpus::get()`.  For the indexer, call
    /// with `1` (let the outer worker pool supply parallelism).  For
    /// single-query search call with a larger value so a single request can
    /// use all cores.
    pub fn new_session(&self, intra_threads: usize) -> Result<VisionSession> {
        let threads = if intra_threads == 0 {
            num_cpus::get()
        } else {
            intra_threads
        };
        let session = build_session(&self.model_path, threads)?;
        Ok(VisionSession { session })
    }
}

fn build_session(path: &Path, intra_threads: usize) -> Result<Session> {
    Session::builder()
        .map_err(|e| AppError::Onnx(e.into()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| AppError::Onnx(e.into()))?
        .with_intra_threads(intra_threads)
        .map_err(|e| AppError::Onnx(e.into()))?
        .commit_from_file(path)
        .map_err(|e| AppError::Onnx(e.into()))
}

// ── Owned session ────────────────────────────────────────────────────────────

/// A single ONNX Runtime session.  Not `Sync` (ORT `Session::run` takes
/// `&mut self`) — hand it off to a single thread or guard with a `Mutex`.
pub struct VisionSession {
    session: Session,
}

impl VisionSession {
    /// Encode a single image → L2-normalised embedding.
    pub fn encode_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        let tensor = preprocess_for_model(img);
        self.run_single(tensor)
    }

    /// Encode `imgs.len()` images in a single ONNX forward pass.
    ///
    /// The export has a dynamic batch dimension, so any batch size works.
    /// Accepts borrowed image references so the caller does not have to
    /// clone out of a surrounding struct (e.g. `RenderedPage`).
    pub fn encode_batch(&mut self, imgs: &[&DynamicImage]) -> Result<Vec<Vec<f32>>> {
        if imgs.is_empty() {
            return Ok(vec![]);
        }
        if imgs.len() == 1 {
            return Ok(vec![self.encode_image(imgs[0])?]);
        }

        let n = imgs.len();
        let h = MODEL_INPUT_SIZE as usize;
        let w = MODEL_INPUT_SIZE as usize;
        let mut batch = ndarray::Array4::<f32>::zeros([n, 3, h, w]);

        for (i, img) in imgs.iter().enumerate() {
            let single = preprocess_for_model(img); // [1, 3, H, W]
            batch
                .slice_mut(ndarray::s![i..i + 1, .., .., ..])
                .assign(&single);
        }

        let shape: Vec<i64> = batch.shape().iter().map(|&d| d as i64).collect();
        let flat: Vec<f32> = batch.into_raw_vec();
        let input_tensor =
            Tensor::from_array((shape, flat.into_boxed_slice())).map_err(AppError::Onnx)?;

        let inputs = ort::inputs!["pixel_values" => input_tensor];
        let outputs = self.session.run(inputs).map_err(AppError::Onnx)?;

        let (_shape, data) = outputs[embedding_key(&outputs)?]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;

        // Split flat `[N, D]` row data into N L2-normalised vectors.
        let total = data.len();
        if total % n != 0 {
            return Err(AppError::Onnx(ort::Error::new(format!(
                "Batch output not divisible by batch size: total={}, n={}",
                total, n
            ))));
        }
        let dim = total / n;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            out.push(l2_normalize(&data[start..end]));
        }
        Ok(out)
    }

    fn run_single(&mut self, tensor: ndarray::Array4<f32>) -> Result<Vec<f32>> {
        let shape: Vec<i64> = tensor.shape().iter().map(|&d| d as i64).collect();
        let flat: Vec<f32> = tensor.into_raw_vec();
        let input_tensor =
            Tensor::from_array((shape, flat.into_boxed_slice())).map_err(AppError::Onnx)?;

        let inputs = ort::inputs!["pixel_values" => input_tensor];
        let outputs = self.session.run(inputs).map_err(AppError::Onnx)?;

        let (_shape, data) = outputs[embedding_key(&outputs)?]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;

        Ok(l2_normalize(data))
    }
}

/// Resolve the embedding output name.  Prefers the names our export scripts
/// use (`embedding` for DINOv2, `image_embeds` for the legacy CLIP export),
/// falling back to the model's first output.
fn embedding_key<'a>(outputs: &'a ort::session::SessionOutputs<'_>) -> Result<&'a str> {
    for key in ["embedding", "image_embeds"] {
        if outputs.contains_key(key) {
            return Ok(key);
        }
    }
    outputs
        .keys()
        .next()
        .ok_or_else(|| AppError::Onnx(ort::Error::new("No outputs from vision model")))
}

// ── Vector helpers ───────────────────────────────────────────────────────────

/// L2-normalise a vector so its f32 norm is strictly below 1, and sanitise
/// non-finite values.
///
/// Cosine similarity on unit vectors collapses to a dot product, which is
/// why the HNSW index uses `DistDot`.  A naive `v / ||v||` in pure f32 can
/// leave the result with norm a few ULPs above 1 (f32 sum-of-squares is
/// lossy); combined with the ~`n · f32::EPSILON` rounding error of the
/// downstream dot product, this trips the strict `1 − dot >= 0` assertion
/// inside `anndists::DistDot::scalar_dot_f32` and panics mid-insert.
///
/// **NaN/Inf handling**: quantised models can emit non-finite outputs on
/// pathological inputs (observed in production with the INT8 export on a
/// degenerate page render).  A NaN that reaches the HNSW graph poisons every
/// later distance sort with a non-total order — Rust's sort then panics and
/// takes the whole indexing batch down.  Such vectors are replaced with the
/// zero vector: dot = 0 against everything, so the region simply never ranks.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    const SAFETY_SHRINK: f64 = 1.0 - 1e-4;
    if v.iter().any(|x| !x.is_finite()) {
        tracing::warn!(
            "Encoder produced a non-finite embedding ({} dims) — replaced with zeros",
            v.len()
        );
        return vec![0.0; v.len()];
    }
    let sq_sum: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if sq_sum < 1e-20 {
        return v.to_vec();
    }
    let scale = (SAFETY_SHRINK / sq_sum.sqrt()) as f32;
    v.iter().map(|&x| x * scale).collect()
}

/// Cosine similarity between two L2-normalised vectors — = dot product.
#[allow(dead_code)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    /// End-to-end smoke test against the real exported model.  Validates the
    /// whole contract in one shot: input name, output name, embedding dim,
    /// L2 normalisation, and batch-vs-single consistency.
    ///
    /// Ignored by default because it needs `models/dinov2_vits14.onnx`.
    /// IMPORTANT: set ORT_DYLIB_PATH explicitly — the test exe lives in
    /// `target/debug/deps/` with no onnxruntime.dll beside it, and the DLL
    /// search order then finds the ancient Windows-ML copy in System32,
    /// which hangs the process inside ort's version handshake:
    /// ```text
    /// __COMPAT_LAYER=RunAsInvoker ORT_DYLIB_PATH=$PWD/onnxruntime.dll \
    ///     cargo test dinov2_end_to_end -- --ignored
    /// ```
    #[test]
    #[ignore = "requires models/dinov2_vits14.onnx + ORT_DYLIB_PATH (see doc comment)"]
    fn dinov2_end_to_end_smoke() {
        let model = Path::new("models/dinov2_vits14.onnx");
        let embedder = VisionEmbedder::load(model).expect("model must load");
        let mut session = embedder.new_session(2).expect("session");

        let gradient = DynamicImage::ImageRgb8(RgbImage::from_fn(400, 300, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x * y) % 256) as u8])
        }));
        let solid = DynamicImage::ImageRgb8(RgbImage::from_pixel(
            320,
            240,
            image::Rgb([230, 230, 230]),
        ));

        let v1 = session.encode_image(&gradient).expect("encode gradient");
        let v1_again = session.encode_image(&gradient).expect("encode again");
        let v2 = session.encode_image(&solid).expect("encode solid");

        assert_eq!(v1.len(), 384, "ViT-S/14 embedding dim");
        let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-2, "should be ~unit norm, got {}", norm);

        let same = cosine_similarity(&v1, &v1_again);
        let diff = cosine_similarity(&v1, &v2);
        assert!(same > 0.999, "identical input must embed identically: {}", same);
        assert!(
            diff < same - 0.05,
            "different images should be farther apart: same={}, diff={}",
            same,
            diff
        );

        // Batch path must agree with the single path.
        let batch = session
            .encode_batch(&[&gradient, &solid])
            .expect("batch encode");
        assert_eq!(batch.len(), 2);
        assert!(cosine_similarity(&batch[0], &v1) > 0.999);
        assert!(cosine_similarity(&batch[1], &v2) > 0.999);
    }
}
