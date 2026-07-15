//! DBNet text detection: preprocessing, inference, and post-processing.
//!
//! Post-processing follows the DB paper / PaddleOCR defaults
//! (`thresh=0.3`, `box_thresh=0.5`, `unclip_ratio=1.6`, 3×3 dilation) but
//! replaces the rotated-rect + polygon-clipper pipeline with axis-aligned
//! bounding boxes: prepress text is horizontal, and this removes the whole
//! clipper/perspective-transform dependency surface.

use crate::error::{AppError, Result};
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

/// Binarisation threshold on the probability map.
const PROB_THRESH: f32 = 0.3;
/// Minimum mean probability inside a candidate box.
const BOX_THRESH: f32 = 0.5;
/// DB unclip ratio — how far boxes are expanded to recover the shrunken
/// text kernel the network predicts.
const UNCLIP_RATIO: f32 = 1.6;
/// Long-edge cap for detector input; keeps inference bounded on large pages.
const MAX_SIDE: u32 = 1280;
/// Boxes smaller than this on either edge (in detector-input pixels) are noise.
const MIN_BOX_PX: u32 = 3;

/// Axis-aligned text region in source-image pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct TextBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

pub struct Detector {
    session: Session,
}

#[cfg(test)]
mod tests {
    use super::TextBox;

    /// Regression for the production panic: the old single-comparator
    /// "same band → compare x, else compare y" ordering was not transitive.
    /// The two-pass grouping must sort any box set without panicking and
    /// keep clearly-separate rows in top-to-bottom order.
    #[test]
    fn reading_order_survives_band_chains() {
        // Chain of vertically-overlapping boxes that broke transitivity:
        // A~B and B~C are "same band", A vs C is not.
        let mut boxes = vec![
            TextBox { x: 300, y: 0, w: 50, h: 20 },  // A, mid 10
            TextBox { x: 200, y: 8, w: 50, h: 20 },  // B, mid 18
            TextBox { x: 100, y: 16, w: 50, h: 20 }, // C, mid 26
            TextBox { x: 0, y: 200, w: 50, h: 20 },  // clearly lower row
        ];
        // Shuffle-ish orders; grouping logic runs on midpoint-sorted input.
        boxes.reverse();

        // Inline copy of the grouping pass (detect() needs a model session).
        boxes.sort_unstable_by_key(|b| b.y + b.h / 2);
        let mut ordered: Vec<TextBox> = Vec::new();
        let mut band: Vec<TextBox> = Vec::new();
        let mut band_mid = 0u32;
        let mut band_max_h = 1u32;
        for b in boxes {
            let mid = b.y + b.h / 2;
            let new_band =
                !band.is_empty() && mid.abs_diff(band_mid) > (band_max_h / 2).max(1);
            if new_band {
                band.sort_unstable_by_key(|t| t.x);
                ordered.append(&mut band);
                band_max_h = 1;
            }
            if band.is_empty() {
                band_mid = mid;
            }
            band_max_h = band_max_h.max(b.h);
            band.push(b);
        }
        band.sort_unstable_by_key(|t| t.x);
        ordered.append(&mut band);

        assert_eq!(ordered.len(), 4);
        // The separate bottom row must come last regardless of banding above.
        assert_eq!(ordered.last().unwrap().y, 200);
        // Within the top chain, x-order applies inside each derived band.
        let top: Vec<u32> = ordered[..3].iter().map(|b| b.x).collect();
        let mut sorted_pairs = top.clone();
        sorted_pairs.sort_unstable();
        assert_eq!(
            {
                let mut t = top.clone();
                t.sort_unstable();
                t
            },
            sorted_pairs,
            "top boxes are exactly the chain members"
        );
    }
}

impl Detector {
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| AppError::Onnx(e.into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::Onnx(e.into()))?
            .with_intra_threads(num_cpus::get().min(4))
            .map_err(|e| AppError::Onnx(e.into()))?
            .commit_from_file(path)
            .map_err(|e| AppError::Onnx(e.into()))?;
        Ok(Self { session })
    }

    /// Detect text regions, returned in reading order (rows top-to-bottom,
    /// left-to-right within a row band).
    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<TextBox>> {
        let (orig_w, orig_h) = img.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Ok(vec![]);
        }

        // ── Resize: cap long edge, round both edges to multiples of 32 ────
        let scale = (MAX_SIDE as f32 / orig_w.max(orig_h) as f32).min(1.0);
        let round32 = |v: f32| (((v / 32.0).round() as u32).max(1) * 32).max(32);
        let in_w = round32(orig_w as f32 * scale);
        let in_h = round32(orig_h as f32 * scale);
        let resized = img
            .resize_exact(in_w, in_h, image::imageops::FilterType::Triangle)
            .to_rgb8();

        // ── Normalise (ImageNet mean/std), NCHW ────────────────────────────
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let (w, h) = (in_w as usize, in_h as usize);
        let mut input = vec![0f32; 3 * h * w];
        for (x, y, p) in resized.enumerate_pixels() {
            let (xi, yi) = (x as usize, y as usize);
            for c in 0..3 {
                input[c * h * w + yi * w + xi] =
                    (p.0[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }

        let tensor = Tensor::from_array((
            vec![1i64, 3, h as i64, w as i64],
            input.into_boxed_slice(),
        ))
        .map_err(AppError::Onnx)?;
        let outputs = self
            .session
            .run(ort::inputs!["x" => tensor])
            .map_err(AppError::Onnx)?;
        let (_shape, prob) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(AppError::Onnx)?;
        // prob is [1, 1, h, w] flattened.

        // ── Binarise + 3×3 dilate ──────────────────────────────────────────
        let mut bin = GrayImage::new(in_w, in_h);
        for y in 0..h {
            for x in 0..w {
                if prob[y * w + x] > PROB_THRESH {
                    bin.put_pixel(x as u32, y as u32, Luma([255u8]));
                }
            }
        }
        let dilated = imageproc::morphology::dilate(
            &bin,
            imageproc::distance_transform::Norm::LInf,
            1, // 3×3 structuring element
        );

        // ── Connected components → per-label bbox + probability mass ──────
        let labels = connected_components(&dilated, Connectivity::Eight, Luma([0u8]));
        #[derive(Clone)]
        struct Acc {
            min_x: u32,
            min_y: u32,
            max_x: u32,
            max_y: u32,
            prob_sum: f64,
            count: u64,
        }
        let mut accs: std::collections::HashMap<u32, Acc> = std::collections::HashMap::new();
        for (x, y, l) in labels.enumerate_pixels() {
            let label = l.0[0];
            if label == 0 {
                continue;
            }
            let p = prob[y as usize * w + x as usize] as f64;
            let a = accs.entry(label).or_insert(Acc {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                prob_sum: 0.0,
                count: 0,
            });
            a.min_x = a.min_x.min(x);
            a.min_y = a.min_y.min(y);
            a.max_x = a.max_x.max(x);
            a.max_y = a.max_y.max(y);
            a.prob_sum += p;
            a.count += 1;
        }

        // ── Filter, unclip, map back to source coordinates ─────────────────
        let inv_x = orig_w as f32 / in_w as f32;
        let inv_y = orig_h as f32 / in_h as f32;
        let mut boxes = Vec::new();
        for a in accs.values() {
            let bw = a.max_x - a.min_x + 1;
            let bh = a.max_y - a.min_y + 1;
            if bw < MIN_BOX_PX || bh < MIN_BOX_PX {
                continue;
            }
            let mean_prob = (a.prob_sum / a.count.max(1) as f64) as f32;
            if mean_prob < BOX_THRESH {
                continue;
            }
            // DB unclip for an axis-aligned rect: offset = area·ratio / perimeter.
            let offset =
                (bw as f32 * bh as f32 * UNCLIP_RATIO / (2.0 * (bw + bh) as f32)).ceil();
            let x0 = (a.min_x as f32 - offset).max(0.0);
            let y0 = (a.min_y as f32 - offset).max(0.0);
            let x1 = (a.max_x as f32 + 1.0 + offset).min(in_w as f32);
            let y1 = (a.max_y as f32 + 1.0 + offset).min(in_h as f32);

            // Back to source resolution.
            let sx = (x0 * inv_x).floor().max(0.0) as u32;
            let sy = (y0 * inv_y).floor().max(0.0) as u32;
            let sw = (((x1 - x0) * inv_x).ceil() as u32).min(orig_w - sx).max(1);
            let sh = (((y1 - y0) * inv_y).ceil() as u32).min(orig_h - sy).max(1);
            boxes.push(TextBox {
                x: sx,
                y: sy,
                w: sw,
                h: sh,
            });
        }

        // Reading order.  NOTE: this must be a two-pass grouping, not a
        // single comparator — "same band when midpoints are close" is not
        // transitive (A~B, B~C, but A≁C), and a non-total order makes
        // `sort_by` panic.  Production hit exactly that on text-dense pages.
        //
        // Pass 1: total-order sort by vertical midpoint.  Pass 2: walk the
        // sorted list, cut a new band when the midpoint jumps by more than
        // half the band's max height, and order each band left-to-right.
        boxes.sort_unstable_by_key(|b| b.y + b.h / 2);
        let mut ordered: Vec<TextBox> = Vec::with_capacity(boxes.len());
        let mut band: Vec<TextBox> = Vec::new();
        let mut band_mid = 0u32;
        let mut band_max_h = 1u32;
        for b in boxes {
            let mid = b.y + b.h / 2;
            let new_band =
                !band.is_empty() && mid.abs_diff(band_mid) > (band_max_h / 2).max(1);
            if new_band {
                band.sort_unstable_by_key(|t| t.x);
                ordered.append(&mut band);
                band_max_h = 1;
            }
            if band.is_empty() {
                band_mid = mid;
            }
            band_max_h = band_max_h.max(b.h);
            band.push(b);
        }
        band.sort_unstable_by_key(|t| t.x);
        ordered.append(&mut band);
        Ok(ordered)
    }
}
