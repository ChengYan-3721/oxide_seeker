//! Result merging and ranking.
//!
//! Candidates flow in from two recall channels (CLIP vector search, pHash
//! pre-filter), are de-duplicated by `page_id`, then scored by a
//! multi-signal re-ranker before the Top-K are enriched with file metadata
//! and returned.
//!
//! Re-ranking fuses three signals:
//!
//! * **CLIP cosine similarity** — primary.  Robust to colour, scale, and
//!   partial-crop variation in the query screenshot.
//! * **pHash proximity** — secondary.  Strongly discriminative for
//!   near-duplicates / exact-crop matches that CLIP may under-score when
//!   the query is aggressively cropped or downscaled.
//! * **Page-position bias** — weak.  First pages (cover art, product hero
//!   shots) are slightly boosted; design libraries are heavily
//!   front-loaded and users tend to search for those.
//!
//! Weights are tunable per-call via [`FusionWeights`]; the default is tuned
//! for prepress / design-asset libraries.

use crate::{
    error::Result,
    search::{phash_index::PHashCandidate, vector_index::VectorMatch},
    storage::database::{self, DbPool, FileRecord, PageRecord},
};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Fusion weights ───────────────────────────────────────────────────────────

/// Weights for the multi-signal re-ranker.  Components sum to ~1.0; the
/// `position_bonus` is added separately as a small bias rather than a
/// convex component.
#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    /// Weight on CLIP cosine similarity, in `[0, 1]`.
    pub clip: f32,
    /// Weight on pHash-derived similarity (`1 − hamming/64`), in `[0, 1]`.
    pub phash: f32,
    /// Additive bonus granted when the page is the first page of its file.
    pub position_bonus: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        // Tuned for design-asset libraries: CLIP dominates, pHash acts as
        // tiebreaker for near-duplicates, first-page gets a whisker of lift.
        Self {
            clip: 0.82,
            phash: 0.15,
            position_bonus: 0.03,
        }
    }
}

impl FusionWeights {
    /// Compute the composite rank score from the raw signals.
    #[inline]
    fn score(
        &self,
        clip_sim: Option<f32>,
        phash_distance: Option<u32>,
        page_num: i64,
    ) -> f32 {
        let c = clip_sim.unwrap_or(0.0).clamp(0.0, 1.0);
        // Distance 0 → similarity 1.0; distance 64 → similarity 0.0.
        let p = phash_distance
            .map(|d| 1.0 - (d as f32 / 64.0).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let pos = if page_num == 1 { self.position_bonus } else { 0.0 };
        (self.clip * c + self.phash * p + pos).clamp(0.0, 1.0)
    }
}

// ── Public result type ──────────────────────────────────────────────────────

/// A single ranked search result returned to the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Absolute path to the file on the server.
    pub file_path: String,
    pub filename: String,
    pub file_type: String,
    /// 1-based page number within the file.
    pub page_num: i64,
    /// CLIP cosine similarity in `[0.0, 1.0]` (0 if the page matched only by pHash).
    pub similarity: f32,
    /// Composite re-rank score in `[0.0, 1.0]` — this is what sorts the list.
    pub score: f32,
    /// pHash Hamming distance (0–64); `None` if only matched by vector search.
    pub phash_distance: Option<u32>,
    /// URL path for the thumbnail image, e.g. `/thumbnails/42_1.webp`.
    pub thumbnail_url: Option<String>,
    pub file_size: Option<i64>,
    pub page_count: i64,
    pub modified_at: Option<String>,
}

// ── Internal merge record ───────────────────────────────────────────────────

#[derive(Debug)]
struct PageCandidate {
    page_id: i64,
    vector_similarity: Option<f32>,
    phash_distance: Option<u32>,
}

/// Merge CLIP vector matches and pHash candidates, multi-signal re-rank,
/// fetch metadata, and return the top `top_k` results sorted by composite
/// score (descending).
pub async fn rank_results(
    pool: &DbPool,
    vector_matches: Vec<VectorMatch>,
    phash_candidates: Vec<PHashCandidate>,
    top_k: usize,
    similarity_threshold: f32,
    weights: FusionWeights,
) -> Result<Vec<SearchResult>> {
    // ── 1. Build page_id → candidate map ──────────────────────────────────────
    let mut page_map: HashMap<i64, PageCandidate> = HashMap::new();

    // Bulk-fetch pages by vector_id so we can hand back the page_id
    let vector_ids: Vec<i64> = vector_matches.iter().map(|m| m.vector_id as i64).collect();

    if !vector_ids.is_empty() {
        let pages = database::get_pages_by_vector_ids(pool, &vector_ids).await?;
        let vid_to_page: HashMap<i64, PageRecord> = pages
            .into_iter()
            .filter_map(|p| p.vector_id.map(|vid| (vid, p)))
            .collect();

        for vm in &vector_matches {
            let vid = vm.vector_id as i64;
            if let Some(page) = vid_to_page.get(&vid) {
                let sim = vm.similarity();
                if sim >= similarity_threshold {
                    page_map
                        .entry(page.id)
                        .and_modify(|c| c.vector_similarity = Some(sim))
                        .or_insert(PageCandidate {
                            page_id: page.id,
                            vector_similarity: Some(sim),
                            phash_distance: None,
                        });
                }
            }
        }
    }

    // Merge pHash candidates (overlap with CLIP matches updates, not replaces)
    for pc in &phash_candidates {
        page_map
            .entry(pc.page_id)
            .and_modify(|c| c.phash_distance = Some(pc.distance))
            .or_insert(PageCandidate {
                page_id: pc.page_id,
                vector_similarity: None,
                phash_distance: Some(pc.distance),
            });
    }

    if page_map.is_empty() {
        return Ok(vec![]);
    }

    // ── 2. Bulk fetch the PageRecords we still need (for page_num + file_id) ─
    // We don't have PageRecord for pHash-only hits yet.  Collect every
    // candidate's page_id and resolve in one round-trip.
    let all_page_ids: Vec<i64> = page_map.keys().copied().collect();
    let page_records = fetch_pages_by_ids(pool, &all_page_ids).await?;
    let page_by_id: HashMap<i64, PageRecord> =
        page_records.into_iter().map(|p| (p.id, p)).collect();

    // ── 3. Score + rank ───────────────────────────────────────────────────────
    let mut ranked: Vec<(f32, PageCandidate)> = page_map
        .into_values()
        .filter_map(|c| {
            let rec = page_by_id.get(&c.page_id)?;
            let score = weights.score(c.vector_similarity, c.phash_distance, rec.page_num);
            Some((score, c))
        })
        .collect();

    ranked.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_k);

    if ranked.is_empty() {
        return Ok(vec![]);
    }

    // ── 4. Bulk fetch the FileRecords for Top-K ──────────────────────────────
    let file_ids: Vec<i64> = ranked
        .iter()
        .filter_map(|(_, c)| page_by_id.get(&c.page_id).map(|p| p.file_id))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let file_records = fetch_files_by_ids(pool, &file_ids).await?;
    let file_by_id: HashMap<i64, FileRecord> =
        file_records.into_iter().map(|f| (f.id, f)).collect();

    // ── 5. Assemble response ─────────────────────────────────────────────────
    let mut results = Vec::with_capacity(ranked.len());
    for (score, candidate) in ranked {
        let page = match page_by_id.get(&candidate.page_id) {
            Some(p) => p,
            None => continue,
        };
        let file = match file_by_id.get(&page.file_id) {
            Some(f) => f,
            None => continue,
        };

        let thumbnail_url = page
            .thumb_path
            .as_ref()
            .map(|p| format!("/thumbnails/{}", p));

        let modified_at = file.modified_at.map(|ts| {
            Utc.timestamp_opt(ts, 0)
                .single()
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()
        });

        results.push(SearchResult {
            file_path: file.path.clone(),
            filename: file.filename.clone(),
            file_type: file.file_type.clone(),
            page_num: page.page_num,
            similarity: candidate.vector_similarity.unwrap_or(0.0),
            score,
            phash_distance: candidate.phash_distance,
            thumbnail_url,
            file_size: file.file_size,
            page_count: file.page_count,
            modified_at,
        });
    }

    Ok(results)
}

/// Fetch multiple page records by their primary key IDs.
async fn fetch_pages_by_ids(pool: &DbPool, ids: &[i64]) -> Result<Vec<PageRecord>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: String = ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT * FROM pages WHERE id IN ({})", placeholders);
    let mut query = sqlx::query_as::<_, PageRecord>(&sql);
    for id in ids {
        query = query.bind(id);
    }
    Ok(query.fetch_all(pool).await?)
}

/// Fetch multiple file records by their primary key IDs.
async fn fetch_files_by_ids(pool: &DbPool, ids: &[i64]) -> Result<Vec<FileRecord>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: String = ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT * FROM files WHERE id IN ({})", placeholders);
    let mut query = sqlx::query_as::<_, FileRecord>(&sql);
    for id in ids {
        query = query.bind(id);
    }
    Ok(query.fetch_all(pool).await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_score_clip_dominates() {
        let w = FusionWeights::default();
        // Two candidates, same pHash.  CLIP should rank higher similarity first.
        let a = w.score(Some(0.9), Some(5), 3);
        let b = w.score(Some(0.5), Some(5), 3);
        assert!(a > b, "higher CLIP sim should score higher: a={}, b={}", a, b);
    }

    #[test]
    fn first_page_bonus_applies() {
        let w = FusionWeights::default();
        let cover = w.score(Some(0.7), None, 1);
        let inner = w.score(Some(0.7), None, 5);
        assert!(cover > inner, "first page should outrank identical-similarity inner");
        assert!((cover - inner - w.position_bonus).abs() < 1e-6);
    }

    #[test]
    fn phash_can_break_tie() {
        let w = FusionWeights::default();
        let near_dup = w.score(Some(0.7), Some(0), 3);
        let far = w.score(Some(0.7), Some(32), 3);
        assert!(near_dup > far, "smaller pHash distance should win the tie");
    }

    #[test]
    fn score_never_exceeds_one() {
        let w = FusionWeights::default();
        let s = w.score(Some(1.0), Some(0), 1);
        assert!(s <= 1.0);
    }
}
