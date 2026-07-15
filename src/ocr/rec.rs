//! CRNN/SVTR text recognition: strip preprocessing, batched inference, and
//! greedy CTC decoding against the vocabulary embedded in the ONNX model's
//! `character` metadata.

use crate::error::{AppError, Result};
use crate::ocr::OcrLine;
use image::{DynamicImage, GenericImageView};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

/// Fixed input height for PP-OCRv3/v4 recognition.
const REC_HEIGHT: u32 = 48;
/// Maximum input width; longer strips are squeezed (PaddleOCR behaviour).
const REC_MAX_WIDTH: u32 = 320;
/// Strips per ONNX forward — bounds peak tensor memory.
const REC_BATCH: usize = 16;

pub struct Recognizer {
    session: Session,
    /// CTC vocabulary: index 0 is blank; `vocab[i]` maps class `i`.
    /// Built as blank + metadata characters + trailing space.
    vocab: Vec<String>,
}

impl Recognizer {
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| AppError::Onnx(e.into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::Onnx(e.into()))?
            .with_intra_threads(num_cpus::get().min(4))
            .map_err(|e| AppError::Onnx(e.into()))?
            .commit_from_file(path)
            .map_err(|e| AppError::Onnx(e.into()))?;

        // Vocabulary ships inside the model: one character per line under the
        // `character` custom-metadata key (PaddleOCR export convention).
        let character = session
            .metadata()
            .map_err(AppError::Onnx)?
            .custom("character")
            .ok_or_else(|| {
                AppError::Config(
                    "OCR rec model lacks the 'character' metadata (not a PaddleOCR export?)"
                        .to_string(),
                )
            })?;

        let mut vocab: Vec<String> = Vec::with_capacity(6700);
        vocab.push(String::new()); // index 0 = CTC blank
        for line in character.lines() {
            vocab.push(line.to_string());
        }
        vocab.push(" ".to_string()); // use_space_char=True convention
        Ok(Self { session, vocab })
    }

    /// Recognise a batch of detected line crops.  Output order matches input.
    pub fn recognize_batch(&mut self, crops: &[DynamicImage]) -> Result<Vec<OcrLine>> {
        let mut out = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(REC_BATCH) {
            out.extend(self.run_chunk(chunk)?);
        }
        Ok(out)
    }

    fn run_chunk(&mut self, crops: &[DynamicImage]) -> Result<Vec<OcrLine>> {
        if crops.is_empty() {
            return Ok(vec![]);
        }

        // Resize every strip to height 48, proportional width capped at 320.
        let resized: Vec<image::RgbImage> = crops
            .iter()
            .map(|c| {
                let (w, h) = c.dimensions();
                let ratio = w as f32 / h.max(1) as f32;
                let rw = ((REC_HEIGHT as f32 * ratio).ceil() as u32)
                    .clamp(8, REC_MAX_WIDTH);
                c.resize_exact(rw, REC_HEIGHT, image::imageops::FilterType::Triangle)
                    .to_rgb8()
            })
            .collect();

        // Right-pad to the widest strip in the batch (zeros == mid-grey after
        // the [-1,1] normalisation used by PaddleOCR rec).
        let n = resized.len();
        let max_w = resized.iter().map(|r| r.width()).max().unwrap_or(8) as usize;
        let h = REC_HEIGHT as usize;
        let mut input = vec![0f32; n * 3 * h * max_w];
        for (i, strip) in resized.iter().enumerate() {
            for (x, y, p) in strip.enumerate_pixels() {
                let (xi, yi) = (x as usize, y as usize);
                for c in 0..3 {
                    input[i * 3 * h * max_w + c * h * max_w + yi * max_w + xi] =
                        (p.0[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        let tensor = Tensor::from_array((
            vec![n as i64, 3, h as i64, max_w as i64],
            input.into_boxed_slice(),
        ))
        .map_err(AppError::Onnx)?;
        let outputs = self
            .session
            .run(ort::inputs!["x" => tensor])
            .map_err(AppError::Onnx)?;
        let (shape, probs) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;
        // [N, T, C] softmax probabilities.
        let (t_len, n_class) = (shape[1] as usize, shape[2] as usize);

        let mut lines = Vec::with_capacity(n);
        for i in 0..n {
            lines.push(ctc_greedy(
                &self.vocab,
                &probs[i * t_len * n_class..(i + 1) * t_len * n_class],
                t_len,
                n_class,
            ));
        }
        Ok(lines)
    }
}

/// Greedy CTC: per-step argmax, collapse repeats, drop blanks (class 0);
/// score is the mean probability of the emitted characters.
fn ctc_greedy(vocab: &[String], probs: &[f32], t_len: usize, n_class: usize) -> OcrLine {
    let mut text = String::new();
    let mut score_sum = 0f32;
    let mut emitted = 0usize;
    let mut prev_class = 0usize;
    for t in 0..t_len {
        let row = &probs[t * n_class..(t + 1) * n_class];
        let (best, best_p) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, p)| (i, *p))
            .unwrap_or((0, 0.0));
        if best != 0 && best != prev_class {
            if let Some(ch) = vocab.get(best) {
                text.push_str(ch);
                score_sum += best_p;
                emitted += 1;
            }
        }
        prev_class = best;
    }
    OcrLine {
        score: if emitted > 0 { score_sum / emitted as f32 } else { 0.0 },
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(chars: &[&str]) -> Vec<String> {
        std::iter::once(String::new()) // index 0 = blank
            .chain(chars.iter().map(|s| s.to_string()))
            .collect()
    }

    fn one_hot(t_steps: &[usize], n_class: usize) -> Vec<f32> {
        let mut v = vec![0f32; t_steps.len() * n_class];
        for (t, &c) in t_steps.iter().enumerate() {
            v[t * n_class + c] = 1.0;
        }
        v
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        let v = vocab(&["A", "B", "C"]);
        // blank=0, A=1, B=2, C=3.  Sequence: A A blank A B B blank C
        let steps = [1usize, 1, 0, 1, 2, 2, 0, 3];
        let probs = one_hot(&steps, 4);
        let line = ctc_greedy(&v, &probs, steps.len(), 4);
        assert_eq!(line.text, "AABC");
        assert!((line.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ctc_all_blank_is_empty_zero_score() {
        let v = vocab(&["A"]);
        let steps = [0usize, 0, 0];
        let probs = one_hot(&steps, 2);
        let line = ctc_greedy(&v, &probs, steps.len(), 2);
        assert_eq!(line.text, "");
        assert_eq!(line.score, 0.0);
    }
}
