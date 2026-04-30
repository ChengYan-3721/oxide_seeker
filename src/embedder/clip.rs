//! CLIP ViT-B/32 visual encoder inference via ONNX Runtime.
//!
//! Two types, clean split:
//!
//! * [`ClipEmbedder`] — a cheap, `Clone`-able *factory* that holds the model
//!   path.  Share it freely via `Arc`.
//! * [`ClipSession`] — an *owned* ONNX Runtime session.  `run()` requires
//!   `&mut self`, so a session is a single-thread resource: the indexer
//!   workers each create their own session (true parallel inference), and
//!   the search engine guards one session behind a `Mutex` (low QPS).
//!
//! This fixes the previous design where every worker thread serialised
//! through a single `Arc<Mutex<Session>>`, completely defeating the rayon
//! pool's parallelism for the CLIP stage.
//!
//! # Model export (one-time setup)
//! ```bash
//! pip install transformers optimum[onnxruntime]
//! optimum-cli export onnx --model openai/clip-vit-base-patch32 --task feature-extraction clip_onnx/
//! # Copy clip_onnx/vision_model.onnx → models/clip_visual.onnx
//! ```
//!
//! ## Optional: INT8 quantisation
//! ```bash
//! optimum-cli onnxruntime quantize --avx512 --onnx_model clip_onnx/vision_model.onnx -o clip_int8/
//! # Copy clip_int8/vision_model_quantized.onnx → models/clip_visual_int8.onnx
//! ```
//! Load the quantised file the same way — no code changes needed.  Expect
//! ~2-3× CPU speedup and ~4× smaller session with <1% accuracy loss on
//! natural images.

use crate::{
    embedder::image_prep::{preprocess_for_clip, CLIP_INPUT_SIZE},
    error::{AppError, Result},
};
use image::DynamicImage;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::{Path, PathBuf};

/// Dimensionality of CLIP ViT-B/32 image embeddings.
#[allow(dead_code)]
pub const CLIP_EMBEDDING_DIM: usize = 512;

// ── Factory ──────────────────────────────────────────────────────────────────

/// Cheap handle that knows where the CLIP model lives.
///
/// Cloning and `Arc`-sharing is free; each caller that needs to run
/// inference asks the embedder for a fresh [`ClipSession`].
#[derive(Clone)]
pub struct ClipEmbedder {
    model_path: PathBuf,
}

impl ClipEmbedder {
    /// Validate the model file and return a factory.
    ///
    /// No session is kept on `ClipEmbedder` — eager validation opens a probe
    /// session and drops it, so callers get an early failure with a useful
    /// error message instead of a mysterious panic at first inference.
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(AppError::Config(format!(
                "CLIP model not found at {}. \
                 Export with: optimum-cli export onnx --model openai/clip-vit-base-patch32 \
                 --task feature-extraction clip_onnx/ \
                 then copy clip_onnx/vision_model.onnx → {}",
                model_path.display(),
                model_path.display()
            )));
        }

        tracing::info!("Validating CLIP model at {}", model_path.display());
        // Probe — throw-away session; confirms the model is readable and the
        // ort dylib resolved correctly.
        let _probe = build_session(model_path, 1)?;
        tracing::info!("CLIP model validated");

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
    pub fn new_session(&self, intra_threads: usize) -> Result<ClipSession> {
        let threads = if intra_threads == 0 {
            num_cpus::get()
        } else {
            intra_threads
        };
        let session = build_session(&self.model_path, threads)?;
        Ok(ClipSession { session })
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

/// A single ONNX Runtime CLIP session.  Not `Sync` (ORT `Session::run` takes
/// `&mut self`) — hand it off to a single thread or guard with a `Mutex`.
pub struct ClipSession {
    session: Session,
}

impl ClipSession {
    /// Encode a single image → 512-dim L2-normalised embedding.
    pub fn encode_image(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        let tensor = preprocess_for_clip(img);
        self.run_single(tensor)
    }

    /// Encode `imgs.len()` images in a single ONNX forward pass.
    ///
    /// The CLIP ONNX export typically has a dynamic batch dimension, so any
    /// batch size works.  Accepts borrowed image references so the caller
    /// does not have to clone out of a surrounding struct (e.g. `RenderedPage`).
    pub fn encode_batch(&mut self, imgs: &[&DynamicImage]) -> Result<Vec<Vec<f32>>> {
        if imgs.is_empty() {
            return Ok(vec![]);
        }
        if imgs.len() == 1 {
            return Ok(vec![self.encode_image(imgs[0])?]);
        }

        let n = imgs.len();
        let h = CLIP_INPUT_SIZE as usize;
        let w = CLIP_INPUT_SIZE as usize;
        let mut batch = ndarray::Array4::<f32>::zeros([n, 3, h, w]);

        for (i, img) in imgs.iter().enumerate() {
            let single = preprocess_for_clip(img); // [1, 3, H, W]
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

        let embedding_key = if outputs.contains_key("image_embeds") {
            "image_embeds"
        } else {
            outputs
                .keys()
                .next()
                .ok_or_else(|| AppError::Onnx(ort::Error::new("No outputs from CLIP model")))?
        };

        let (_shape, data) = outputs[embedding_key]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;

        // Split flat `[N, D]` row data into N L2-normalised vectors.
        let total = data.len();
        if total % n != 0 {
            return Err(AppError::Onnx(ort::Error::new(format!(
                "CLIP batch output not divisible by batch size: total={}, n={}",
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

        let embedding_key = if outputs.contains_key("image_embeds") {
            "image_embeds"
        } else {
            outputs
                .keys()
                .next()
                .ok_or_else(|| AppError::Onnx(ort::Error::new("No outputs from CLIP model")))?
        };

        let (_shape, data) = outputs[embedding_key]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;

        Ok(l2_normalize(data))
    }
}

// ── Vector helpers ───────────────────────────────────────────────────────────

/// L2-normalise a vector so its f32 norm is strictly below 1.
///
/// Cosine similarity on unit vectors collapses to a dot product, which is
/// why the HNSW index uses `DistDot`.  A naive `v / ||v||` in pure f32 can
/// leave the result with norm a few ULPs above 1 (f32 sum-of-squares is
/// lossy); combined with the ~`n · f32::EPSILON` rounding error of the
/// downstream dot product, this trips the strict `1 − dot >= 0` assertion
/// inside `anndists::DistDot::scalar_dot_f32` and panics mid-insert.
///
/// We therefore:
///   1. accumulate the squared norm in f64 for precision, and
///   2. apply a tiny safety shrink so every returned vector has norm
///      ≲ `1 − SAFETY_SHRINK`.  The margin comfortably exceeds the
///      ~`512 · f32::EPSILON ≈ 6e-5` worst-case dot-product error for
///      CLIP ViT-B/32 embeddings while staying well below search-relevant
///      precision.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    const SAFETY_SHRINK: f64 = 1.0 - 1e-4;
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
