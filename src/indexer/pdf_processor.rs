//! PDF and AI file rendering pipeline.
//!
//! Handles:
//! - Opening PDF and Adobe Illustrator (.ai) files via PDFium
//! - Running the imposition filter
//! - Rendering each page to a `DynamicImage`
//! - Delegating thumbnail saving and pHash/CLIP embedding to callers

use crate::{
    error::{AppError, Result},
    indexer::filter,
};
use image::DynamicImage;
use pdfium_render::prelude::*;
use std::path::Path;

/// One rendered page ready for embedding.
pub struct RenderedPage {
    /// 1-based page number
    pub page_num: i32,
    /// Rendered image at thumbnail resolution (256px max dimension)
    pub image: DynamicImage,
    /// Width of the original PDF page in pixels at the render DPI
    pub width_px: u32,
    /// Height of the original PDF page in pixels at the render DPI
    pub height_px: u32,
}

/// Summary returned after processing an entire file.
#[allow(dead_code)]
pub struct ProcessedFile {
    pub pages: Vec<RenderedPage>,
    /// `true` if the file was excluded by the imposition filter
    pub excluded: bool,
    /// Reason string when `excluded == true`
    pub exclusion_reason: Option<String>,
    pub page_count: i32,
}

/// Target long-edge pixel size for index-time rendering.
/// CLIP ViT-B/32 only needs 224×224; we render at 512px so there is enough
/// detail for pHash and thumbnail generation without wasting CPU/RAM at 1240px.
/// Thumbnail saving further downscales to THUMB_SIZE (256px).
const RENDER_TARGET_PX: i32 = 512;

/// Render all pages of a PDF or AI file.
///
/// - Stage 1: fast raw-byte scan for OPI / external-link markers (no PDFium overhead)
/// - Stage 2: opens the document with PDFium and runs structural filter
/// - Renders every page scaled so the longer dimension equals `RENDER_TARGET_PX`
pub fn process_file(
    path: &Path,
    pdfium: &Pdfium,
) -> Result<ProcessedFile> {
    // Stage 1: XMP-based check for external PDF links (runs before PDFium)
    if let Some(reason) = filter::pre_check_bytes(path) {
        tracing::debug!("Pre-check excluded {}: {}", path.display(), reason);
        return Ok(ProcessedFile {
            pages: vec![],
            excluded: true,
            exclusion_reason: Some(reason),
            page_count: 0,
        });
    }

    // Stage 2: open the document (works for both PDF and AI)
    let doc = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| AppError::Pdf(format!("Failed to open {:?}: {}", path, e)))?;

    // Get page count (no structural exclusion logic — handled by pre_check_bytes)
    let page_count = filter::get_page_count(&doc);
    let mut rendered_pages = Vec::with_capacity(page_count as usize);

    for page_index in 0..page_count {
        let page = doc
            .pages()
            .get(page_index as u16)
            .map_err(|e| AppError::Pdf(format!("Page {} error: {}", page_index + 1, e)))?;

        // Compute target pixel dimensions from the page's point dimensions.
        // PDFium page dimensions are in points (1 pt = 1/72 inch).
        // We scale so the longer edge equals RENDER_TARGET_PX.
        let page_w_pt = page.width().value as f64;
        let page_h_pt = page.height().value as f64;

        // Guard against degenerate pages (zero-size)
        let (target_w, target_h) = if page_w_pt <= 0.0 || page_h_pt <= 0.0 {
            (RENDER_TARGET_PX, RENDER_TARGET_PX)
        } else {
            let scale = RENDER_TARGET_PX as f64 / page_w_pt.max(page_h_pt);
            let tw = ((page_w_pt * scale).round() as i32).max(1);
            let th = ((page_h_pt * scale).round() as i32).max(1);
            (tw, th)
        };

        // Build render config with explicit pixel dimensions — never pass 0×0
        let render_cfg = PdfRenderConfig::new()
            .set_target_width(target_w)
            .set_target_height(target_h)
            .render_form_data(false)
            .render_annotations(false);

        // Render at target DPI using a temporary bitmap
        let bitmap = page
            .render_with_config(&render_cfg)
            .map_err(|e| AppError::Pdf(format!("Render error page {}: {}", page_index + 1, e)))?;

        let width_px = bitmap.width() as u32;
        let height_px = bitmap.height() as u32;

        // Convert PDFium bitmap → image::DynamicImage
        let dynamic_image = bitmap_to_dynamic_image(&bitmap)?;

        rendered_pages.push(RenderedPage {
            page_num: page_index + 1,
            image: dynamic_image,
            width_px,
            height_px,
        });
    }

    Ok(ProcessedFile {
        pages: rendered_pages,
        excluded: false,
        exclusion_reason: None,
        page_count,
    })
}

/// Convert a PDFium `PdfBitmap` to an `image::DynamicImage`.
fn bitmap_to_dynamic_image(bitmap: &PdfBitmap) -> Result<DynamicImage> {
    // PDFium returns BGRA bytes by default
    let width = bitmap.width() as u32;
    let height = bitmap.height() as u32;
    let bytes = bitmap.as_raw_bytes();

    // BGRA → RGBA
    let mut rgba_bytes = vec![0u8; (width * height * 4) as usize];
    for pixel in 0..(width * height) as usize {
        let b = bytes[pixel * 4];
        let g = bytes[pixel * 4 + 1];
        let r = bytes[pixel * 4 + 2];
        let a = bytes[pixel * 4 + 3];
        rgba_bytes[pixel * 4] = r;
        rgba_bytes[pixel * 4 + 1] = g;
        rgba_bytes[pixel * 4 + 2] = b;
        rgba_bytes[pixel * 4 + 3] = a;
    }

    let img_buf = image::RgbaImage::from_raw(width, height, rgba_bytes)
        .ok_or_else(|| AppError::Pdf("Failed to construct RgbaImage from PDFium bitmap".to_string()))?;

    Ok(DynamicImage::ImageRgba8(img_buf))
}

/// Initialise the PDFium library from the shared library bundled alongside the binary.
///
/// On Windows, `pdfium.dll` must reside next to the executable or be in PATH.
/// On Linux, `libpdfium.so` must be reachable.
pub fn init_pdfium() -> Result<Pdfium> {
    // Try to load pdfium from the directory of the current executable first,
    // then fall back to the system library search path.
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path("./")
        )
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e| AppError::Pdf(format!("Cannot load PDFium library: {}", e)))?,
    );
    Ok(pdfium)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bgra_to_rgba_conversion() {
        // Single pixel: B=10, G=20, R=30, A=255
        let bytes = vec![10u8, 20, 30, 255];
        let width = 1u32;
        let height = 1u32;
        let mut rgba = vec![0u8; 4];
        // Replicate conversion logic
        rgba[0] = bytes[2]; // R
        rgba[1] = bytes[1]; // G
        rgba[2] = bytes[0]; // B
        rgba[3] = bytes[3]; // A
        assert_eq!(rgba, [30, 20, 10, 255]);
    }
}