//! Thumbnail generation and caching.
//!
//! Thumbnails are persisted as lossless WebP rather than JPEG.  For the
//! predominantly flat-colour, geometric content of design files (logos,
//! layouts, vector art rasterised from AI/PDF) lossless WebP usually beats
//! JPEG q=85 by 20-40 % on disk while introducing zero compression
//! artefacts — artefacts that would otherwise bias pHash matching slightly
//! between the index build and any later re-index.

use crate::error::{AppError, Result};
use image::{DynamicImage, ImageFormat};
use std::path::{Path, PathBuf};

/// Maximum dimension (px) for index-time thumbnails stored on disk.
pub const THUMB_SIZE: u32 = 256;

/// Maximum dimension (px) for higher-quality preview thumbnails (reserved
/// for future on-demand preview generation).
#[allow(dead_code)]
pub const PREVIEW_SIZE: u32 = 512;

/// File extension used for new thumbnails.  Old `.jpg` files written by
/// previous builds remain readable via `ServeDir` — the relative path stored
/// in `pages.thumb_path` is authoritative.
pub const THUMB_EXT: &str = "webp";

/// Manages thumbnail generation and storage under a root directory.
pub struct ThumbnailStore {
    root: PathBuf,
}

impl ThumbnailStore {
    /// Create a new store rooted at `dir`.  The directory is created if it
    /// does not exist.
    pub async fn new(dir: &Path) -> Result<Self> {
        tokio::fs::create_dir_all(dir).await?;
        Ok(Self {
            root: dir.to_path_buf(),
        })
    }

    /// Synchronous constructor for environments without a tokio runtime
    /// (notably the indexing worker subprocess, which has no async surface
    /// of its own).
    pub fn new_sync(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            root: dir.to_path_buf(),
        })
    }

    /// Generate a thumbnail, save it to disk, and return the relative path.
    ///
    /// The file name is derived from `file_id` and `page_num` so re-indexing
    /// overwrites the previous thumbnail deterministically.
    ///
    /// If a thumbnail with the legacy `.jpg` extension exists it is
    /// removed best-effort — keeping the older copy would leave a stale
    /// file whose URL no longer matches the DB row.
    pub fn save_thumbnail(
        &self,
        img: &DynamicImage,
        file_id: i64,
        page_num: i64,
        max_size: u32,
    ) -> Result<String> {
        let thumb = resize_contain(img, max_size);
        let relative = format!("{}_{}.{}", file_id, page_num, THUMB_EXT);
        let abs_path = self.root.join(&relative);

        // Best-effort removal of the legacy .jpg variant.  Ignore failures:
        // if it's missing (common) or unlinked by something else, there is
        // nothing for us to clean up.
        let legacy_jpg = self.root.join(format!("{}_{}.jpg", file_id, page_num));
        let _ = std::fs::remove_file(&legacy_jpg);

        thumb
            .save_with_format(&abs_path, ImageFormat::WebP)
            .map_err(AppError::Image)?;
        Ok(relative)
    }

    /// Return the absolute path for a thumbnail given its stored relative
    /// path.
    #[allow(dead_code)]
    pub fn abs_path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    /// Delete a thumbnail file (best-effort; ignores not-found errors).
    #[allow(dead_code)]
    pub async fn delete(&self, relative: &str) {
        let path = self.root.join(relative);
        let _ = tokio::fs::remove_file(&path).await;
    }
}

/// Resize `img` so that neither dimension exceeds `max_size`, preserving
/// aspect ratio.  Uses `Lanczos3` for high-quality downscaling.
pub fn resize_contain(img: &DynamicImage, max_size: u32) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    if w <= max_size && h <= max_size {
        return img.clone();
    }
    let scale = (max_size as f32 / w as f32).min(max_size as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    img.resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
}

/// Decode raw image bytes (PNG, JPEG, WebP, …) into a `DynamicImage`.
#[allow(dead_code)]
pub fn decode_image_bytes(data: &[u8]) -> Result<DynamicImage> {
    image::load_from_memory(data).map_err(AppError::Image)
}
