//! In-memory pHash store with parallel Hamming scan.
//!
//! v1 pulled every `(id, phash_hex)` row out of SQLite **on every query** —
//! at 6 M region rows that alone took seconds and allocated a String per
//! row.  v2 keeps the whole table resident: 6 M × 16 B = ~96 MB, scanned
//! with XOR + popcount via rayon in single-digit milliseconds.
//!
//! Lifecycle
//! ---------
//! * `load` — bulk-populated from `regions` at startup.
//! * `add_batch` / `remove` — incrementally maintained by the indexer and
//!   watcher alongside the HNSW index (the two stores deliberately share the
//!   same id space: `regions.id`).
//!
//! Because every region row (full page + 9 overlapping tiles) carries its
//! own pHash, a *partial* screenshot that lines up with any tile position
//! now gets near-duplicate treatment too — v1's single whole-page hash only
//! ever matched whole-page queries.

use crate::embedder::phash::hamming_distance;
use parking_lot::RwLock;
use rayon::prelude::*;

/// A candidate region found by pHash matching.
#[derive(Debug, Clone)]
pub struct PhashCandidate {
    /// `regions.id`
    pub region_id: i64,
    /// Hamming distance (0–64; lower = more similar)
    pub distance: u32,
}

/// Hard cap on how many pHash candidates we hand to the ranker.
///
/// Low-entropy queries — colour charts, blank/solid-fill pages — produce
/// pHashes that are within Hamming threshold of tens of thousands of
/// unrelated regions.  Capping keeps the ranker's DB lookups bounded.
const MAX_CANDIDATES: usize = 2_000;

/// Shared, incrementally-maintained pHash table.
#[derive(Default)]
pub struct PhashStore {
    /// `(region_id, phash)` pairs.  A flat Vec keeps the scan cache-friendly;
    /// removals are rare (file deletions) and handled with `retain`.
    entries: RwLock<Vec<(i64, u64)>>,
}

impl PhashStore {
    /// Build an empty store; populate with [`replace_all`](Self::replace_all).
    pub fn new() -> Self {
        Self::default()
    }

    /// Swap in the full table (startup bulk load).
    pub fn replace_all(&self, entries: Vec<(i64, u64)>) {
        let mut guard = self.entries.write();
        *guard = entries;
        tracing::info!("pHash store loaded: {} entries", guard.len());
    }

    /// Append freshly-indexed regions.
    pub fn add_batch(&self, batch: &[(i64, u64)]) {
        if batch.is_empty() {
            return;
        }
        self.entries.write().extend_from_slice(batch);
    }

    /// Drop regions whose ids appear in `ids` (file deleted / re-indexed).
    /// O(n) over the table, but deletions are rare and 6 M-entry retain is
    /// tens of milliseconds.
    pub fn remove(&self, ids: &[i64]) {
        if ids.is_empty() {
            return;
        }
        let victims: std::collections::HashSet<i64> = ids.iter().copied().collect();
        self.entries.write().retain(|(id, _)| !victims.contains(id));
    }

    /// Number of entries (for logging / status).
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Find all regions within `max_distance` Hamming bits of `query`,
    /// sorted by ascending distance and truncated to [`MAX_CANDIDATES`].
    pub fn find_candidates(&self, query: u64, max_distance: u32) -> Vec<PhashCandidate> {
        let guard = self.entries.read();
        let mut candidates: Vec<PhashCandidate> = guard
            .par_iter()
            .filter_map(|&(region_id, hash)| {
                let dist = hamming_distance(query, hash);
                if dist <= max_distance {
                    Some(PhashCandidate {
                        region_id,
                        distance: dist,
                    })
                } else {
                    None
                }
            })
            .collect();
        drop(guard);

        candidates.sort_unstable_by_key(|c| c.distance);
        if candidates.len() > MAX_CANDIDATES {
            tracing::debug!(
                total = candidates.len(),
                kept = MAX_CANDIDATES,
                "pHash candidate flood — truncating to closest matches"
            );
            candidates.truncate(MAX_CANDIDATES);
        }
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exact_and_near_matches_sorted() {
        let store = PhashStore::new();
        store.replace_all(vec![
            (1, 0b0000),
            (2, 0b0001),          // dist 1
            (3, 0b0111),          // dist 3
            (4, u64::MAX),        // dist 64
        ]);
        let hits = store.find_candidates(0b0000, 4);
        let ids: Vec<i64> = hits.iter().map(|c| c.region_id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(hits[0].distance, 0);
        assert_eq!(hits[2].distance, 3);
    }

    #[test]
    fn add_and_remove_roundtrip() {
        let store = PhashStore::new();
        store.add_batch(&[(10, 0xAA), (11, 0xBB), (12, 0xCC)]);
        assert_eq!(store.len(), 3);
        store.remove(&[11]);
        assert_eq!(store.len(), 2);
        let hits = store.find_candidates(0xBB, 0);
        assert!(hits.is_empty(), "removed entry must not match");
    }
}
