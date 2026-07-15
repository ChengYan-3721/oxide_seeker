//! Image pre-processing for the vision encoder (DINOv2 ViT-S/14).
//!
//! DINOv2 expects:
//!   - Input size: a multiple of the 14-px patch — we use 224 × 224
//!   - Pixel values normalised with the standard ImageNet mean/std:
//!       mean = [0.485, 0.456, 0.406]
//!       std  = [0.229, 0.224, 0.225]
//!   - Channel order: RGB (not BGR)
//!   - Layout: NCHW (batch × channels × height × width) as f32
//!
//! ## Letterbox vs. center-crop
//!
//! The standard eval pipeline does *resize-shorter-edge + center crop*, which
//! permanently discards the long-edge margins.  For our task that's a recall
//! killer: a query screenshot whose aspect ratio is portrait against an A4
//! (≈ 1:1.41) library page would have its top/bottom shaved off at 224×224 —
//! losing exactly the kind of detail a user is searching for.
//!
//! We therefore *letterbox*: scale uniformly so the long edge fits 224, then
//! pad the short edge with the ImageNet mean colour (which becomes 0 after
//! normalisation, so the padding is effectively neutral for the network).
//! Index-side tiles and query images go through the identical function, so
//! embeddings stay comparable.

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::Array4;

/// Model input resolution (must be a multiple of the DINOv2 patch size 14).
pub const MODEL_INPUT_SIZE: u32 = 224;

/// ImageNet normalisation constants used by DINOv2.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Letterbox `img` into a `size × size` RGB canvas, preserving aspect ratio.
///
/// Resizes uniformly so the longest edge hits `size`, centres the result, and
/// fills the remaining strips with the ImageNet mean colour (≈ neutral grey).
pub fn letterbox_to_square(img: &DynamicImage, size: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return solid_canvas(size);
    }

    // Scale so that max(w', h') == size while preserving aspect ratio.
    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let new_w = ((w as f32 * scale).round() as u32).max(1).min(size);
    let new_h = ((h as f32 * scale).round() as u32).max(1).min(size);

    let resized = image::imageops::resize(
        &img.to_rgb8(),
        new_w,
        new_h,
        image::imageops::FilterType::Lanczos3,
    );

    let mut canvas = solid_canvas(size);
    let dx = ((size - new_w) / 2) as i64;
    let dy = ((size - new_h) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, dx, dy);
    canvas
}

/// Build a `size × size` RGB image filled with the ImageNet mean pixel.
fn solid_canvas(size: u32) -> RgbImage {
    // ImageNet mean expressed in 0..255 space, matching what `to_rgb8` emits.
    let fill = Rgb([
        (MEAN[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (MEAN[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (MEAN[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]);
    RgbImage::from_pixel(size, size, fill)
}

/// Preprocess `img` for the vision encoder and return the input tensor
/// `[1, 3, 224, 224]`.  Internally letterboxes to preserve aspect ratio.
pub fn preprocess_for_model(img: &DynamicImage) -> Array4<f32> {
    let canvas = letterbox_to_square(img, MODEL_INPUT_SIZE);
    rgb_to_model_tensor(&canvas)
}

/// Build the normalised NCHW tensor from an already-letterboxed RGB image.
pub fn rgb_to_model_tensor(canvas: &RgbImage) -> Array4<f32> {
    let h = MODEL_INPUT_SIZE as usize;
    let w = MODEL_INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, h, w]);

    // The canvas should already be 224x224 — but be defensive against callers
    // who hand us something mis-sized.  Out-of-bounds reads would otherwise
    // panic; clamping degrades to a centred crop, which is at least sensible.
    let cw = canvas.width().min(MODEL_INPUT_SIZE);
    let ch = canvas.height().min(MODEL_INPUT_SIZE);
    for y in 0..ch as usize {
        for x in 0..cw as usize {
            let pixel = canvas.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let raw = pixel.0[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (raw - MEAN[c]) / STD[c];
            }
        }
    }

    tensor
}
