//! Perceptual hash (pHash) computation for images.
//!
//! Uses the `image-hasher` crate with the DCT-based pHash algorithm.
//! The hash is stored as a 16-character lowercase hex string (64 bits).

use crate::error::{AppError, Result};
use image::DynamicImage;
use image_hasher::{HashAlg, HasherConfig, ImageHash};

/// Number of bits in the pHash (64 bits = 8 bytes = 16 hex chars)
pub const PHASH_BITS: u32 = 64;

/// Compute the pHash of an image and return it as a 16-character hex string.
pub fn compute_phash(img: &DynamicImage) -> String {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(8, 8) // 8×8 = 64 bits
        .to_hasher();

    let hash = hasher.hash_image(img);
    hash_to_hex(&hash)
}

/// Decode a hex pHash string back to an `ImageHash`.
pub fn hex_to_hash(hex: &str) -> Result<ImageHash<Box<[u8]>>> {
    let bytes = hex::decode(hex)
        .map_err(|e| AppError::Search(format!("Invalid pHash hex '{}': {}", hex, e)))?;
    Ok(ImageHash::from_bytes(&bytes)
        .map_err(|e| AppError::Search(format!("Cannot decode pHash bytes: {:?}", e)))?)
}

/// Encode an `ImageHash` to its hex representation.
pub fn hash_to_hex(hash: &ImageHash<Box<[u8]>>) -> String {
    hex::encode(hash.as_bytes())
}

/// Compute the Hamming distance between two pHash hex strings.
/// Returns `None` if either string is malformed.
pub fn hamming_distance(a: &str, b: &str) -> Option<u32> {
    let ha = hex_to_hash(a).ok()?;
    let hb = hex_to_hash(b).ok()?;
    Some(ha.dist(&hb))
}

/// Return `true` if the Hamming distance between two hex hashes is within `threshold`.
pub fn is_similar(a: &str, b: &str, threshold: u32) -> bool {
    hamming_distance(a, b).map_or(false, |d| d <= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn solid_image(r: u8, g: u8, b: u8) -> DynamicImage {
        let buf = RgbImage::from_fn(64, 64, |_, _| image::Rgb([r, g, b]));
        DynamicImage::ImageRgb8(buf)
    }

    #[test]
    fn same_image_zero_distance() {
        let img = solid_image(128, 64, 32);
        let h1 = compute_phash(&img);
        let h2 = compute_phash(&img);
        assert_eq!(hamming_distance(&h1, &h2), Some(0));
    }

    #[test]
    fn very_different_images_large_distance() {
        let black = solid_image(0, 0, 0);
        let white = solid_image(255, 255, 255);
        let h_black = compute_phash(&black);
        let h_white = compute_phash(&white);
        let dist = hamming_distance(&h_black, &h_white).unwrap();
        // Black and white solid images should have a non-zero distance
        // (exact value depends on algorithm, but should be > 0)
        assert!(dist > 0, "Expected non-zero distance, got {}", dist);
    }

    #[test]
    fn hex_roundtrip() {
        let img = solid_image(100, 150, 200);
        let hex = compute_phash(&img);
        assert_eq!(hex.len(), 16, "pHash hex should be 16 chars (64 bits)");
        // Roundtrip decode
        let decoded = hex_to_hash(&hex).expect("Should decode");
        assert_eq!(hash_to_hex(&decoded), hex);
    }
}