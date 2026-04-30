//! Search engine: orchestrates pHash pre-filter + CLIP vector search + ranking.

pub mod phash_index;
pub mod ranker;
pub mod vector_index;

pub use ranker::SearchResult;
pub use vector_index::VectorIndex;

use crate::{
    config::SearchConfig,
    embedder::{
        clip::{ClipEmbedder, ClipSession},
        phash::compute_phash,
    },
    error::{AppError, Result},
    storage::database::DbPool,
};
use image::DynamicImage;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

/// Shared search engine state — cheaply cloneable via `Arc` internals.
#[derive(Clone)]
pub struct SearchEngine {
    pool: DbPool,
    /// One dedicated CLIP session for query-side inference.  Guarded by a
    /// `Mutex` because `ort::Session::run` needs `&mut self`; a single
    /// session is plenty for the low-QPS search path and uses all CPU cores
    /// for each individual request.
    clip_session: Arc<Mutex<ClipSession>>,
    index: Arc<VectorIndex>,
    config: SearchConfig,
}

impl SearchEngine {
    pub fn new(
        pool: DbPool,
        clip: &ClipEmbedder,
        index: Arc<VectorIndex>,
        config: SearchConfig,
    ) -> Result<Self> {
        // Single query session: give it every available CPU core so a single
        // request is fast, since they arrive one at a time over the LAN.
        let session = clip.new_session(num_cpus::get())?;
        Ok(Self {
            pool,
            clip_session: Arc::new(Mutex::new(session)),
            index,
            config,
        })
    }

    /// Run a full image search pipeline:
    ///
    /// 1. Encode query image with CLIP → 512-dim vector
    /// 2. Compute pHash
    /// 3. Run pHash pre-filter (Hamming distance)
    /// 4. Run HNSW vector search (Top-K × 5 candidates)
    /// 5. Merge, re-rank, and return Top-K results
    pub async fn search_image(
        &self,
        img: &DynamicImage,
        top_k: Option<usize>,
    ) -> Result<SearchResponse> {
        let top_k = top_k.unwrap_or(self.config.default_top_k);
        let t0 = Instant::now();

        // ── Step 1: Encode with CLIP ──────────────────────────────────────────
        let session = self.clip_session.clone();
        let img_clone = img.clone();
        let vector = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let mut guard = session.lock();
            guard.encode_image(&img_clone)
        })
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!("CLIP task panicked: {}", e)))??;

        // ── Step 2: Compute pHash ─────────────────────────────────────────────
        let img_clone2 = img.clone();
        let query_phash = tokio::task::spawn_blocking(move || compute_phash(&img_clone2))
            .await
            .map_err(|e| AppError::Other(anyhow::anyhow!("pHash task panicked: {}", e)))?;

        // ── Step 3: pHash pre-filter ──────────────────────────────────────────
        let phash_candidates = phash_index::find_candidates(
            &self.pool,
            &query_phash,
            self.config.phash_threshold,
        )
        .await?;

        // ── Step 4: HNSW vector search ────────────────────────────────────────
        // Over-fetch so the re-ranker has headroom.
        let search_k = (top_k * 5).max(50);
        let index_ref = self.index.clone();
        let vec_clone = vector.clone();
        let vector_matches = tokio::task::spawn_blocking(move || {
            index_ref.search(&vec_clone, search_k)
        })
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!("Vector search task panicked: {}", e)))??;

        // ── Step 5: Merge + multi-signal re-rank ──────────────────────────────
        let results = ranker::rank_results(
            &self.pool,
            vector_matches,
            phash_candidates,
            top_k,
            self.config.similarity_threshold,
            ranker::FusionWeights::default(),
        )
        .await?;

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        tracing::info!(
            results = results.len(),
            elapsed_ms,
            "Search completed"
        );

        Ok(SearchResponse {
            results,
            search_time_ms: elapsed_ms,
        })
    }

    /// Convenience: search from raw image bytes (PNG/JPEG/WebP).
    pub async fn search_bytes(
        &self,
        data: &[u8],
        top_k: Option<usize>,
    ) -> Result<SearchResponse> {
        let img = image::load_from_memory(data).map_err(AppError::Image)?;
        self.search_image(&img, top_k).await
    }
}

/// Response payload returned to HTTP handlers.
#[derive(Debug, serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub search_time_ms: u64,
}
