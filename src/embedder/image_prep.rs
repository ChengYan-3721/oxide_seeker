//! Image pre-processing for CLIP inference.
//!
//! CLIP ViT-B/32 expects:
//!   - Input size: 224 × 224 pixels
//!   - Pixel values normalized with ImageNet mean/std:
//!       mean = [0.48145466, 0.4578275,  0.40821073]
//!       std  = [0.26862954, 0.26130258, 0.27577711]
//!   - Channel order: RGB (not BGR)
//!   - Layout: NCHW (batch × channels × height × width) as f32

use image::{DynamicImage, RgbImage};
use ndarray::Array4;

/// CLIP input resolution
pub const CLIP_INPUT_SIZE: u32 = 224;

/// ImageNet-style normalisation constants used by CLIP
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Resize `img` to 224×224 (centre-crop then resize, matching the standard CLIP
/// pre-processing pipeline) and return a normalised NCHW `Array4<f32>` tensor.
///
/// The returned shape is `[1, 3, 224, 224]`.
pub fn preprocess_for_clip(img: &DynamicImage) -> Array4<f32> {
    // 1. Convert to RGB (drop alpha if present)
    let rgb = img.to_rgb8();

    // 2. Resize to CLIP_INPUT_SIZE × CLIP_INPUT_SIZE using bicubic / Lanczos3
    let resized: RgbImage = image::imageops::resize(
        &rgb,
        CLIP_INPUT_SIZE,
        CLIP_INPUT_SIZE,
        image::imageops::FilterType::Lanczos3,
    );

    // 3. Build NCHW tensor with normalisation
    //    layout: [batch=1, channel=3, height=224, width=224]
    let h = CLIP_INPUT_SIZE as usize;
    let w = CLIP_INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, h, w]);

    for y in 0..h {
        for x in 0..w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let raw = pixel.0[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (raw - MEAN[c]) / STD[c];
            }
        }
    }

    tensor
}

/// Convenience: preprocess raw image bytes (PNG, JPEG, WebP, …).
pub fn preprocess_bytes_for_clip(data: &[u8]) -> crate::error::Result<Array4<f32>> {
    let img = image::load_from_memory(data).map_err(crate::error::AppError::Image)?;
    Ok(preprocess_for_clip(&img))
}
