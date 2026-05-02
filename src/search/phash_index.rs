//! pHash-based candidate pre-filter.
//!
//! Loads all non-null pHash values from SQLite and computes Hamming distances
//! in Rust. Pages within the configured threshold are returned as candidates
//! for the CLIP re-ranking stage.

use crate::{
    embedder::phash,
    error::Result,
    storage::database::{self, DbPool},
};

/// Hard cap on how many pHash candidates we hand to the ranker.
///
/// Low-entropy queries — colour charts, GMG colour-management test pages,
/// blank/solid-fill pages — produce pHashes that are within Hamming threshold
/// of tens of thousands of unrelated library pages.  The ranker would then
/// issue a single `SELECT ... WHERE id IN (?,?,...)` with that many binds,
/// blowing past SQLite's `SQLITE_MAX_VARIABLE_NUMBER`.  The downstream code
/// also chunks its `IN (...)` queries, but capping here keeps the work bounded
/// upstream so we don't waste cycles ranking obvious noise.
const MAX_PHASH_CANDIDATES: usize = 5_000;

/// A candidate page found by pHash matching.
#[derive(Debug, Clone)]
pub struct PHashCandidate {
    /// `pages.id`
    pub page_id: i64,
    /// Hamming distance (0–64; lower = more similar)
    pub distance: u32,
}

/// Find all pages whose pHash is within `max_distance` Hamming bits of `query_hash`.
///
/// Loads all stored pHashes from SQLite (typically cached in the OS page cache)
/// and computes distances in-process.  For 300 k pages the working set is only
/// ~2.4 MB (300 000 × 8 bytes) so this fits comfortably in L3 cache.
///
/// Results are sorted by ascending distance and truncated to
/// [`MAX_PHASH_CANDIDATES`] to keep low-entropy queries (e.g. colour charts)
/// from flooding the ranker.
pub async fn find_candidates(
    pool: &DbPool,
    query_hash: &str,
    max_distance: u32,
) -> Result<Vec<PHashCandidate>> {
    // Fetch all (page_id, phash_hex) pairs
    let all = database::find_pages_by_phash_candidates(pool).await?;

    let mut candidates: Vec<PHashCandidate> = all
        .into_iter()
        .filter_map(|(page_id, stored_hash)| {
            let dist = phash::hamming_distance(query_hash, &stored_hash)?;
            if dist <= max_distance {
                Some(PHashCandidate {
                    page_id,
                    distance: dist,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by ascending Hamming distance, then keep only the closest matches.
    candidates.sort_unstable_by_key(|c| c.distance);
    if candidates.len() > MAX_PHASH_CANDIDATES {
        tracing::debug!(
            total = candidates.len(),
            kept = MAX_PHASH_CANDIDATES,
            "pHash candidate flood — truncating to closest matches"
        );
        candidates.truncate(MAX_PHASH_CANDIDATES);
    }

    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::phash::compute_phash;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn identical_hash_zero_distance() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |_, _| {
            image::Rgb([100u8, 150, 200])
        }));
        let h = compute_phash(&img);
        assert_eq!(phash::hamming_distance(&h, &h), Some(0));
    }
}