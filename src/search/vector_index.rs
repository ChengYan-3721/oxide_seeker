//! HNSW vector index backed by `hnsw_rs` (pure-Rust, SIMD-optimised).
//!
//! Design highlights
//! -----------------
//! * **True incremental insertion.**  `Hnsw::insert` is O(log N); we never
//!   rebuild the graph on `add`.  Searches see newly-inserted vectors
//!   immediately — no `trigger_rebuild` / background snapshot dance.
//! * **Zero duplicate storage.**  hnsw_rs owns every vector inside its graph
//!   (`PointData::V(Vec<T>)`).  We do **not** keep a secondary `entries` copy,
//!   which previously doubled RAM usage at ~1 GB per million 512-dim vectors.
//! * **Tombstone-based deletion.**  hnsw_rs does not support in-place removal,
//!   so `remove` adds ids to a `HashSet`.  Searches over-fetch and filter
//!   deleted ids at query time.  Tombstone growth is bounded in practice
//!   (users delete files occasionally, not en masse).
//! * **Pre-normalised dot distance.**  `ClipEmbedder` L2-normalises every
//!   vector, so we use `DistDot` (= 1 − cosine-similarity for unit vectors),
//!   which skips the redundant norm calculation inside cosine distance.
//! * **Persistence** uses hnsw_rs's native `file_dump` / `HnswIo::load_hnsw`,
//!   producing `{basename}.hnsw.graph` + `{basename}.hnsw.data`.  A small
//!   sidecar `{basename}.meta.bin` records `next_id` and the tombstone set.
//!
//! Hyperparameters (tuned for 512-dim CLIP embeddings, ~200k-1M vectors):
//! `M=16`, `ef_construction=200`, `ef_search=64`.  Increase `ef_search`
//! (via over-fetching) when many tombstones are present.

use crate::error::{AppError, Result};
use hnsw_rs::{api::AnnT, hnsw::Hnsw, hnswio::HnswIo, prelude::DistDot};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// A vector id is the integer stored in `pages.vector_id`.
pub type VectorId = u64;

/// Top-K search result.
#[derive(Debug, Clone)]
pub struct VectorMatch {
    pub vector_id: VectorId,
    /// Cosine distance (lower = more similar).  `0.0` = identical.
    pub distance: f32,
}

impl VectorMatch {
    /// Convert distance to similarity in `[0, 1]`.
    pub fn similarity(&self) -> f32 {
        (1.0 - self.distance).clamp(0.0, 1.0)
    }
}

// ── HNSW hyperparameters ─────────────────────────────────────────────────────

/// M in HNSW — maximum outgoing edges per node per layer.
const MAX_NB_CONNECTION: usize = 16;
/// ef during graph construction (higher = more accurate, slower build).
const EF_CONSTRUCTION: usize = 200;
/// Maximum layer count.  16 supports ~16M points comfortably.
const MAX_LAYER: usize = 16;
/// Allocation hint — grows automatically beyond this.
const DEFAULT_CAPACITY: usize = 500_000;
/// ef during search (higher = more accurate, slower).
const EF_SEARCH: usize = 64;

// ── File naming ──────────────────────────────────────────────────────────────

const GRAPH_SUFFIX: &str = ".hnsw.graph";
const DATA_SUFFIX: &str = ".hnsw.data";
const META_SUFFIX: &str = ".meta.bin";

/// Sidecar metadata persisted alongside the HNSW graph.
#[derive(Serialize, Deserialize, Default)]
struct VectorMeta {
    next_id: u64,
    deleted: Vec<u64>,
}

/// Thread-safe HNSW vector index with incremental inserts and tombstone deletes.
pub struct VectorIndex {
    /// Owning HNSW graph.  `hnsw_rs` guards internal mutation via `RwLock`s,
    /// so `insert` / `search` can be called concurrently from `&Self`.
    hnsw: Hnsw<'static, f32, DistDot>,
    /// Tombstone set — ids that have been logically removed but may still
    /// appear in graph traversals.  Filtered out at search time.
    deleted: RwLock<HashSet<VectorId>>,
    /// Next allocatable id (strictly monotonic across the lifetime of the DB).
    next_id: AtomicU64,
    /// Directory containing the dump files.
    dir: PathBuf,
    /// Base filename without extension (e.g. `"vectors"`).
    basename: String,
}

impl VectorIndex {
    /// Open the index whose dump files live next to `path_base`.
    ///
    /// The parent directory of `path_base` is used as the storage dir; its
    /// file stem is used as the basename.  For example,
    /// `data/vectors.usearch` → dir=`data/`, basename=`vectors`.
    pub fn open(path_base: &Path) -> Result<Arc<Self>> {
        let dir = path_base
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        std::fs::create_dir_all(&dir).map_err(AppError::Io)?;

        let basename = path_base
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("vectors")
            .to_string();

        let graph_path = dir.join(format!("{}{}", basename, GRAPH_SUFFIX));
        let data_path = dir.join(format!("{}{}", basename, DATA_SUFFIX));
        let meta_path = dir.join(format!("{}{}", basename, META_SUFFIX));

        let (hnsw, meta) = if graph_path.exists() {
            tracing::info!("Loading HNSW index from {}", graph_path.display());
            // `HnswIo::load_hnsw` returns `Hnsw<'b, _, _>` with `'a: 'b` where
            // `'a` is the borrow of `self`.  By leaking the `HnswIo` we lift
            // `'a` to `'static`, letting us store the returned `Hnsw` as
            // `Hnsw<'static, _, _>` without lifetime gymnastics.  The leak is
            // a one-off at startup (a few KB).
            let io: &'static mut HnswIo =
                Box::leak(Box::new(HnswIo::new(&dir, &basename)));
            match io.load_hnsw::<f32, DistDot>() {
                Ok(hnsw) => {
                    let meta: VectorMeta = std::fs::read(&meta_path)
                        .ok()
                        .and_then(|bytes| bincode::deserialize(&bytes).ok())
                        .unwrap_or_default();
                    tracing::info!(
                        "HNSW loaded: {} points, {} tombstones, next_id={}",
                        hnsw.get_nb_point(),
                        meta.deleted.len(),
                        meta.next_id,
                    );
                    (hnsw, meta)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load HNSW index ({}). Treating dump as corrupted and recreating: {}",
                        e,
                        graph_path.display()
                    );

                    for p in [&graph_path, &data_path, &meta_path] {
                        if p.exists() {
                            if let Err(remove_err) = std::fs::remove_file(p) {
                                return Err(AppError::VectorIndex(format!(
                                    "Failed to load HNSW graph: {}; also failed to delete corrupted file {}: {}",
                                    e,
                                    p.display(),
                                    remove_err
                                )));
                            }
                            tracing::warn!("Deleted corrupted index file: {}", p.display());
                        }
                    }

                    tracing::info!("Creating fresh HNSW index at {}", dir.display());
                    let hnsw = Hnsw::<f32, DistDot>::new(
                        MAX_NB_CONNECTION,
                        DEFAULT_CAPACITY,
                        MAX_LAYER,
                        EF_CONSTRUCTION,
                        DistDot {},
                    );
                    (hnsw, VectorMeta { next_id: 1, deleted: vec![] })
                }
            }
        } else {
            tracing::info!("Creating fresh HNSW index at {}", dir.display());
            let hnsw = Hnsw::<f32, DistDot>::new(
                MAX_NB_CONNECTION,
                DEFAULT_CAPACITY,
                MAX_LAYER,
                EF_CONSTRUCTION,
                DistDot {},
            );
            (hnsw, VectorMeta { next_id: 1, deleted: vec![] })
        };

        let deleted: HashSet<VectorId> = meta.deleted.iter().copied().collect();
        let next = meta.next_id.max(1);

        Ok(Arc::new(Self {
            hnsw,
            deleted: RwLock::new(deleted),
            next_id: AtomicU64::new(next),
            dir,
            basename,
        }))
    }

    /// Add a single vector with the given `id`.  Searchable immediately.
    #[allow(dead_code)]
    pub fn add(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        // hnsw_rs clones the slice into an owned `Vec<T>` internally, so
        // we can safely pass a borrowed view without holding a duplicate.
        self.hnsw.insert((vector, id as usize));
        Ok(())
    }

    /// Add a batch of vectors in a single call (uses hnsw_rs parallel insert
    /// when the batch is large enough to amortise its locking overhead).
    pub fn add_batch(&self, batch: &[(VectorId, Vec<f32>)]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        // Build `(&Vec<f32>, usize)` tuples expected by `parallel_insert`.
        // NOTE: `parallel_insert` acquires a write lock once per layer and
        // inserts in parallel, which beats N single inserts when N >= ~64.
        if batch.len() >= 64 {
            let view: Vec<(&Vec<f32>, usize)> =
                batch.iter().map(|(id, v)| (v, *id as usize)).collect();
            self.hnsw.parallel_insert(&view);
        } else {
            for (id, v) in batch {
                self.hnsw.insert((v.as_slice(), *id as usize));
            }
        }
        Ok(())
    }

    /// Mark vector ids as deleted.  Does not reclaim graph memory; repeated
    /// mass-deletion should be followed by a cold rebuild (delete the dump
    /// files and re-index) to compact the graph.
    pub fn remove(&self, ids: &[VectorId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut del = self.deleted.write();
        for &id in ids {
            del.insert(id);
        }
        Ok(())
    }

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// When tombstones are present we over-fetch so that, after filtering,
    /// we still return up to `k` live neighbours.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorMatch>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let total = self.hnsw.get_nb_point();
        if total == 0 {
            return Ok(vec![]);
        }
        let deleted = self.deleted.read();
        let nb_tomb = deleted.len();
        // Heuristic: fetch k + min(tombstones, 2k) up to total, so a purge of
        // many tombstones doesn't starve the result set.
        let fetch_k = k
            .saturating_add(nb_tomb.min(k.saturating_mul(2)))
            .min(total);
        let ef = EF_SEARCH.max(fetch_k);
        let neighbours = self.hnsw.search(query, fetch_k, ef);

        Ok(neighbours
            .into_iter()
            .filter(|n| !deleted.contains(&(n.d_id as VectorId)))
            .take(k)
            .map(|n| VectorMatch {
                vector_id: n.d_id as VectorId,
                distance: n.distance,
            })
            .collect())
    }

    /// Persist the index to disk (graph + data + sidecar meta).
    ///
    /// Uses hnsw_rs's read-lock-only dump path, so it is safe to call
    /// concurrently with `add` / `search` — inserts just briefly wait on the
    /// internal RwLock.
    ///
    /// Workaround for hnsw_rs 0.3.4: `HnswIo::load_hnsw` unconditionally sets
    /// the internal `datamap_opt` flag to `true`, which makes `file_dump` pick
    /// the "do not overwrite" code path on every subsequent save and emit
    /// uniquely-suffixed files like `vectors-NNNN.hnsw.{graph,data}` instead of
    /// updating the canonical ones.  We dump to a temp basename and rename on
    /// success so the canonical pair always reflects the latest state.
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(AppError::Io)?;

        // hnsw_rs panics / errors when dumping an HNSW with no entry point.
        // Persist meta only so `next_id` survives restarts in this case.
        if self.hnsw.get_nb_point() == 0 {
            self.write_meta()?;
            return Ok(());
        }

        let tmp_basename = format!("{}.new", self.basename);
        let tmp_graph = self.dir.join(format!("{}{}", tmp_basename, GRAPH_SUFFIX));
        let tmp_data = self.dir.join(format!("{}{}", tmp_basename, DATA_SUFFIX));
        // Clean leftovers from a previous failed save so DumpInit doesn't
        // generate yet another random-suffixed filename.
        let _ = std::fs::remove_file(&tmp_graph);
        let _ = std::fs::remove_file(&tmp_data);

        let actual_basename = self
            .hnsw
            .file_dump(&self.dir, &tmp_basename)
            .map_err(|e| AppError::VectorIndex(format!("file_dump failed: {}", e)))?;

        if actual_basename != tmp_basename {
            // hnsw_rs still picked a unique suffix despite our fresh tmp name
            // (e.g. a concurrent save raced with us). Clean up and bail.
            let stray_graph = self.dir.join(format!("{}{}", actual_basename, GRAPH_SUFFIX));
            let stray_data = self.dir.join(format!("{}{}", actual_basename, DATA_SUFFIX));
            let _ = std::fs::remove_file(&stray_graph);
            let _ = std::fs::remove_file(&stray_data);
            return Err(AppError::VectorIndex(format!(
                "file_dump generated unexpected basename: {}",
                actual_basename
            )));
        }

        let final_graph = self.dir.join(format!("{}{}", self.basename, GRAPH_SUFFIX));
        let final_data = self.dir.join(format!("{}{}", self.basename, DATA_SUFFIX));

        // std::fs::rename on Windows fails if the destination exists, so
        // remove first then rename.
        if final_graph.exists() {
            std::fs::remove_file(&final_graph).map_err(AppError::Io)?;
        }
        if final_data.exists() {
            std::fs::remove_file(&final_data).map_err(AppError::Io)?;
        }
        std::fs::rename(&tmp_graph, &final_graph).map_err(AppError::Io)?;
        std::fs::rename(&tmp_data, &final_data).map_err(AppError::Io)?;

        self.write_meta()?;

        tracing::debug!(
            "Vector index saved ({} points, {} tombstones)",
            self.hnsw.get_nb_point(),
            self.deleted.read().len(),
        );
        Ok(())
    }

    fn write_meta(&self) -> Result<()> {
        let meta = VectorMeta {
            next_id: self.next_id.load(Ordering::Acquire),
            deleted: self.deleted.read().iter().copied().collect(),
        };
        let bytes = bincode::serialize(&meta)
            .map_err(|e| AppError::VectorIndex(format!("meta serialise: {}", e)))?;
        let meta_path = self.dir.join(format!("{}{}", self.basename, META_SUFFIX));
        std::fs::write(&meta_path, bytes).map_err(AppError::Io)?;
        Ok(())
    }

    /// Allocate a fresh, unique vector id.
    pub fn next_id(&self) -> VectorId {
        self.next_id.fetch_add(1, Ordering::AcqRel)
    }

    /// Number of live (non-tombstoned) vectors.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.hnsw
            .get_nb_point()
            .saturating_sub(self.deleted.read().len())
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Always `true` — hnsw_rs is query-ready after every insert; there is
    /// no background rebuild stage.  Kept for API compatibility.
    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        true
    }

    /// No-op — kept for API compatibility with the previous rebuild-based
    /// implementation.
    pub fn trigger_rebuild(self: &Arc<Self>) {}

    /// No-op — inserts go straight into the live graph.
    #[allow(dead_code)]
    pub fn rebuild_sync(&self) -> Result<()> {
        Ok(())
    }
}
