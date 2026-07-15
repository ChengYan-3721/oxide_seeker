//! Perceptual hash (pHash) computation for images.
//!
//! Uses the `image-hasher` crate with the DoubleGradient algorithm at
//! 8×8 = 64 bits.  Hashes are handled as raw `u64` throughout — SQLite stores
//! them as a bit-cast `i64` and the in-memory pHash store scans them with
//! XOR + popcount, so no hex round-trips exist anywhere on the hot path.

use image::DynamicImage;
use image_hasher::{HashAlg, HasherConfig};

/// Compute the 64-bit pHash of an image.
pub fn compute_phash(img: &DynamicImage) -> u64 {
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(8, 8) // 8×8 = 64 bits
        .to_hasher();

    let hash = hasher.hash_image(img);
    let bytes = hash.as_bytes();
    // DoubleGradient at 8×8 yields exactly 8 bytes; be defensive anyway.
    let mut buf = [0u8; 8];
    for (i, b) in bytes.iter().take(8).enumerate() {
        buf[i] = *b;
    }
    u64::from_le_bytes(buf)
}

/// Hamming distance between two 64-bit pHashes.
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
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
        assert_eq!(hamming_distance(compute_phash(&img), compute_phash(&img)), 0);
    }

    #[test]
    fn gradient_images_nonzero_distance() {
        let a = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, _| {
            image::Rgb([(x * 4) as u8, 0, 0])
        }));
        let b = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |_, y| {
            image::Rgb([0, (y * 4) as u8, 0])
        }));
        assert!(hamming_distance(compute_phash(&a), compute_phash(&b)) > 0);
    }

    #[test]
    fn db_bitcast_roundtrip_preserves_high_bit() {
        // regions.phash stores the u64 bit-cast to SQLite's signed i64;
        // make sure hashes with the sign bit set survive the round-trip.
        let h: u64 = 0xFFFF_0000_DEAD_BEEF;
        assert_eq!((h as i64) as u64, h);
    }
}
