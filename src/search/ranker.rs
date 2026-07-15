//! Result merging and ranking.
//!
//! Candidates flow in from two recall channels (ANN vector search over
//! region embeddings, in-memory pHash scan), are aggregated by **page**
//! (multiple region rows — the full page and its 9 overlapping tiles —
//! collapse to a single result), then scored by a two-signal fusion before
//! the Top-K are enriched with file metadata and returned.
//!
//! Signals:
//!
//! * **Embedding cosine similarity** — primary.  Max over the page's region
//!   rows: a tile match counts as a page match, which is what makes partial
//!   screenshots recall whole pages.
//! * **pHash proximity** — secondary tiebreaker.  Strongly discriminative for
//!   near-duplicates; per-region hashes mean a screenshot aligned with any
//!   tile position also benefits.  Min Hamming distance over the page's
//!   regions.
//!
//! v1 additionally had first-page and tile-hit bonuses.  Both were removed:
//! with overlapping tiles a "tile hit" carries no extra information (most
//! pages hit through tiles now), and the first-page bias was never validated
//! against an evaluation set.  Weights live in [`FusionWeights`] and should
//! be re-tuned via `--evaluate` when the model or corpus changes.

use crate::{
    error::Result,
    search::{phash_store::PhashCandidate, vector_index::VectorMatch},
    storage::database::{self, DbPool, FileRecord, PageRecord},
};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Fusion weights ───────────────────────────────────────────────────────────

/// Weights for the three-signal re-ranker.
#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    /// Weight on embedding cosine similarity, in `[0, 1]`.
    pub vector: f32,
    /// Weight on pHash-derived similarity (`1 − hamming/64`), in `[0, 1]`.
    pub phash: f32,
    /// Weight on the OCR/FTS text score (bm25 min-max normalised within the
    /// candidate set), in `[0, 1]`.  Zero when the OCR channel is disabled
    /// or the query contains no readable text.
    pub text: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        // Starting point for DINOv2 + OCR; re-tune with the evaluation
        // harness.  The vector signal dominates; text breaks same-layout
        // ties (the visual channel's blind spot); pHash breaks near-dup ties.
        Self {
            vector: 0.70,
            phash: 0.10,
            text: 0.20,
        }
    }
}

impl FusionWeights {
    /// Compute the composite rank score from the raw signals.
    #[inline]
    fn score(
        &self,
        vector_sim: Option<f32>,
        phash_distance: Option<u32>,
        text_score: Option<f32>,
    ) -> f32 {
        let v = vector_sim.unwrap_or(0.0).clamp(0.0, 1.0);
        // Distance 0 → similarity 1.0; distance 64 → similarity 0.0.
        let p = phash_distance
            .map(|d| 1.0 - (d as f32 / 64.0).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let t = text_score.unwrap_or(0.0).clamp(0.0, 1.0);
        (self.vector * v + self.phash * p + self.text * t).clamp(0.0, 1.0)
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
    /// Embedding cosine similarity in `[0.0, 1.0]` (max over the page's
    /// region rows; 0 if the page matched only by pHash).
    pub similarity: f32,
    /// Composite re-rank score in `[0.0, 1.0]` — this is what sorts the list.
    pub score: f32,
    /// pHash Hamming distance (0–64); `None` if only matched by vector search.
    /// Min over the page's region rows.
    pub phash_distance: Option<u32>,
    /// URL path for the thumbnail image, e.g. `/thumbnails/42_1.webp`.
    pub thumbnail_url: Option<String>,
    pub file_size: Option<i64>,
    pub page_count: i64,
    pub modified_at: Option<String>,
}

// ── Internal aggregate ──────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct PageAggregate {
    /// Best (max) embedding similarity across all region rows of this page.
    vector_sim: Option<f32>,
    /// Best (min) pHash distance across all region rows of this page.
    phash_dist: Option<u32>,
    /// Normalised FTS text score (page-level; one value per page).
    text_score: Option<f32>,
}

impl PageAggregate {
    fn update_vector(&mut self, sim: f32) {
        match self.vector_sim {
            Some(prev) if prev >= sim => {}
            _ => self.vector_sim = Some(sim),
        }
    }

    fn update_phash(&mut self, dist: u32) {
        self.phash_dist = Some(match self.phash_dist {
            Some(prev) => prev.min(dist),
            None => dist,
        });
    }
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Merge vector matches, pHash candidates, and OCR text candidates; fuse
/// scores per page; fetch metadata; and return the top `top_k` results
/// sorted by composite score (descending).
///
/// `text_candidates` carries `(page_id, normalised_score)` pairs from the
/// FTS channel.  Text hits are an independent recall source: a page found
/// only by text still enters the ranking — that is the mechanism that lets
/// same-layout files win on wording when the visual channels tie.
pub async fn rank_results(
    pool: &DbPool,
    vector_matches: Vec<VectorMatch>,
    phash_candidates: Vec<PhashCandidate>,
    text_candidates: Vec<(i64, f32)>,
    top_k: usize,
    similarity_threshold: f32,
    weights: FusionWeights,
) -> Result<Vec<SearchResult>> {
    // ── 1. Resolve region ids → (page_id, file_id, page_num) in one query ──
    let mut all_region_ids: Vec<i64> = Vec::with_capacity(
        vector_matches.len() + phash_candidates.len(),
    );
    all_region_ids.extend(vector_matches.iter().map(|m| m.vector_id as i64));
    all_region_ids.extend(phash_candidates.iter().map(|c| c.region_id));
    all_region_ids.sort_unstable();
    all_region_ids.dedup();

    if all_region_ids.is_empty() && text_candidates.is_empty() {
        return Ok(vec![]);
    }

    let hits = database::get_region_hits(pool, &all_region_ids).await?;
    // region_id → (page_id, file_id, page_num)
    let region_to_page: HashMap<i64, (i64, i64, i64)> = hits
        .into_iter()
        .map(|h| (h.region_id, (h.page_id, h.file_id, h.page_num)))
        .collect();

    // ── 2. Aggregate by page ────────────────────────────────────────────────
    // page_id → (aggregate, file_id, page_num)
    let mut page_map: HashMap<i64, (PageAggregate, i64, i64)> = HashMap::new();

    for vm in &vector_matches {
        let sim = vm.similarity();
        if sim < similarity_threshold {
            continue;
        }
        if let Some(&(page_id, file_id, page_num)) = region_to_page.get(&(vm.vector_id as i64)) {
            page_map
                .entry(page_id)
                .or_insert_with(|| (PageAggregate::default(), file_id, page_num))
                .0
                .update_vector(sim);
        }
    }

    for pc in &phash_candidates {
        if let Some(&(page_id, file_id, page_num)) = region_to_page.get(&pc.region_id) {
            page_map
                .entry(page_id)
                .or_insert_with(|| (PageAggregate::default(), file_id, page_num))
                .0
                .update_phash(pc.distance);
        }
    }

    // Text candidates address pages directly.  Pages already recalled by the
    // visual channels just get their text score set; text-only pages need
    // their (file_id, page_num) resolved before they can join the map.
    let mut text_only: Vec<(i64, f32)> = Vec::new();
    for &(page_id, score) in &text_candidates {
        match page_map.get_mut(&page_id) {
            Some((agg, _, _)) => agg.text_score = Some(score),
            None => text_only.push((page_id, score)),
        }
    }
    if !text_only.is_empty() {
        let ids: Vec<i64> = text_only.iter().map(|(id, _)| *id).collect();
        let recs = database::get_pages_by_ids(pool, &ids).await?;
        let by_id: HashMap<i64, &PageRecord> = recs.iter().map(|p| (p.id, p)).collect();
        for (page_id, score) in text_only {
            if let Some(p) = by_id.get(&page_id) {
                page_map
                    .entry(page_id)
                    .or_insert_with(|| (PageAggregate::default(), p.file_id, p.page_num))
                    .0
                    .text_score = Some(score);
            }
        }
    }

    if page_map.is_empty() {
        return Ok(vec![]);
    }

    // ── 3. Score + rank ──────────────────────────────────────────────────────
    let mut ranked: Vec<(f32, i64, PageAggregate, i64, i64)> = page_map
        .into_iter()
        .map(|(page_id, (agg, file_id, page_num))| {
            let score = weights.score(agg.vector_sim, agg.phash_dist, agg.text_score);
            (score, page_id, agg, file_id, page_num)
        })
        .collect();

    // total_cmp, not partial_cmp-with-Equal-fallback: a stray NaN score would
    // otherwise violate sort's total-order contract and panic the request.
    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(top_k);

    // ── 4. Bulk-fetch surviving pages + owning files ────────────────────────
    let page_ids: Vec<i64> = ranked.iter().map(|(_, pid, ..)| *pid).collect();
    let file_ids: Vec<i64> = ranked
        .iter()
        .map(|(_, _, _, fid, _)| *fid)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let pages = database::get_pages_by_ids(pool, &page_ids).await?;
    let page_by_id: HashMap<i64, PageRecord> = pages.into_iter().map(|p| (p.id, p)).collect();

    let files = database::get_files_by_ids(pool, &file_ids).await?;
    let file_by_id: HashMap<i64, FileRecord> = files.into_iter().map(|f| (f.id, f)).collect();

    // ── 5. Assemble response ─────────────────────────────────────────────────
    let mut results = Vec::with_capacity(ranked.len());
    for (score, page_id, agg, file_id, page_num) in ranked {
        let file = match file_by_id.get(&file_id) {
            Some(f) => f,
            None => continue,
        };
        let thumb = page_by_id
            .get(&page_id)
            .and_then(|p| p.thumb_path.as_ref())
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
            page_num,
            similarity: agg.vector_sim.unwrap_or(0.0),
            score,
            phash_distance: agg.phash_dist,
            thumbnail_url: thumb,
            file_size: file.file_size,
            page_count: file.page_count,
            modified_at,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_score_vector_dominates() {
        let w = FusionWeights::default();
        let a = w.score(Some(0.9), Some(5), None);
        let b = w.score(Some(0.5), Some(5), None);
        assert!(a > b, "higher similarity should score higher: a={}, b={}", a, b);
    }

    #[test]
    fn phash_can_break_tie() {
        let w = FusionWeights::default();
        let near_dup = w.score(Some(0.7), Some(0), None);
        let far = w.score(Some(0.7), Some(32), None);
        assert!(near_dup > far, "smaller pHash distance should win the tie");
    }

    #[test]
    fn text_breaks_same_layout_tie() {
        let w = FusionWeights::default();
        let with_text = w.score(Some(0.85), None, Some(1.0));
        let without = w.score(Some(0.88), None, None);
        assert!(
            with_text > without,
            "a strong text match should outrank a slightly better visual-only match"
        );
    }

    #[test]
    fn text_only_hit_scores_nonzero() {
        let w = FusionWeights::default();
        let s = w.score(None, None, Some(1.0));
        assert!((s - w.text).abs() < 1e-6, "text-only pages ride on the text weight alone");
    }

    #[test]
    fn score_never_exceeds_one() {
        let w = FusionWeights::default();
        assert!(w.score(Some(1.0), Some(0), Some(1.0)) <= 1.0);
    }

    #[test]
    fn aggregate_takes_max_vector_min_phash() {
        let mut agg = PageAggregate::default();
        agg.update_vector(0.6);
        agg.update_vector(0.8);
        agg.update_vector(0.7);
        assert_eq!(agg.vector_sim, Some(0.8));

        agg.update_phash(20);
        agg.update_phash(8);
        agg.update_phash(15);
        assert_eq!(agg.phash_dist, Some(8));
    }
}
