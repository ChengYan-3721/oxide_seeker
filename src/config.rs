use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Context, Result};

/// Top-level application configuration loaded from `config.toml`
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub paths: PathsConfig,
    pub indexer: IndexerConfig,
    pub search: SearchConfig,
    pub filter: FilterConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Bind address, e.g. "0.0.0.0"
    pub host: String,
    /// Listen port, e.g. 8080
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathsConfig {
    /// Directories to scan for PDF/AI files
    pub scan_dirs: Vec<PathBuf>,
    /// Root directory for database, thumbnails, and vector index
    pub data_dir: PathBuf,
    /// Path to the vision encoder ONNX model file
    pub model_path: PathBuf,
    /// PP-OCR detector model.  If the file is missing the OCR text channel
    /// is disabled (visual-only search) with a startup warning.
    #[serde(default = "PathsConfig::default_ocr_det_path")]
    pub ocr_det_path: PathBuf,
    /// PP-OCR recognizer model (vocabulary embedded in its metadata).
    #[serde(default = "PathsConfig::default_ocr_rec_path")]
    pub ocr_rec_path: PathBuf,
}

impl PathsConfig {
    fn default_ocr_det_path() -> PathBuf {
        PathBuf::from("./ppocr_det.onnx")
    }

    fn default_ocr_rec_path() -> PathBuf {
        PathBuf::from("./ppocr_rec.onnx")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexerConfig {
    /// Number of parallel worker threads (recommended: CPU cores / 2)
    pub worker_threads: usize,
    /// Batch size for CLIP inference
    pub batch_size: usize,
    /// Enable filesystem watcher for incremental updates
    pub watch_enabled: bool,
    /// DPI for PDF page rendering (higher = better quality, slower)
    pub render_dpi: f32,
    /// Maximum directory recursion depth, counted from each `paths.scan_dirs`
    /// entry (the entry itself is depth 0, its children depth 1, etc.).  Files
    /// deeper than this are ignored.  Hard-capped at
    /// [`MAX_SCAN_DEPTH_LIMIT`]; values above the cap are silently clamped to
    /// avoid runaway recursion on pathological hierarchies (e.g. `node_modules`).
    #[serde(default = "IndexerConfig::default_max_scan_depth")]
    pub max_scan_depth: u32,
    /// Interval (seconds) at which the watchdog re-runs `start_full_index`
    /// to catch events the OS-level watcher missed (common on SMB / network
    /// shares).  Set to `0` to disable.  Default: 1800 (30 min).
    #[serde(default = "IndexerConfig::default_rescan_interval_secs")]
    pub rescan_interval_secs: u64,
    /// When the watcher receives a Modify event, it samples size+mtime,
    /// waits this many seconds, then samples again.  The file is only
    /// dispatched to the worker pool when two consecutive samples agree
    /// (i.e. the writing application has finished flushing).  Default: 3.
    #[serde(default = "IndexerConfig::default_watcher_settle_secs")]
    pub watcher_settle_secs: u64,
    /// Files smaller than this are treated as empty placeholders by the
    /// watcher and held back until they grow.  Helps with the common
    /// "designer saves a blank file, then fills it in" workflow without
    /// burning a `crash_attempts` slot on the empty version.  Default: 1024.
    #[serde(default = "IndexerConfig::default_watcher_min_bytes")]
    pub watcher_min_bytes: u64,
    /// Maximum number of stability re-checks a single file may undergo
    /// before the watcher drops it from the pending queue (it'll be picked
    /// up again on the next periodic rescan).  Default: 20 — combined with
    /// `watcher_settle_secs=3` this gives a one-minute settling window.
    #[serde(default = "IndexerConfig::default_watcher_max_retries")]
    pub watcher_max_retries: u32,
    /// When `true` (default), each page is indexed as the full-page embedding
    /// **plus 9 overlapping tiles** (tile edge = ½ page, stride = ¼ page), so
    /// a query that only matches part of the artwork can still recall the
    /// page.  Costs 10× the embeddings per page; disable only for very large
    /// corpora where partial-screenshot queries don't matter.
    #[serde(default = "IndexerConfig::default_tiles_enabled")]
    pub tiles_enabled: bool,
}

/// Hard cap for `IndexerConfig::max_scan_depth`.  Picked at 32 because:
///   * Windows MAX_PATH (~260) divided by a typical 8-char folder name puts
///     a sane upper bound around 32 levels;
///   * scanning beyond 32 levels almost always indicates a symlink loop or
///     content-addressable storage that should be excluded explicitly, not
///     traversed.
pub const MAX_SCAN_DEPTH_LIMIT: u32 = 32;

impl IndexerConfig {
    fn default_max_scan_depth() -> u32 {
        3
    }

    fn default_rescan_interval_secs() -> u64 {
        1800
    }

    fn default_watcher_settle_secs() -> u64 {
        3
    }

    fn default_watcher_min_bytes() -> u64 {
        1024
    }

    fn default_watcher_max_retries() -> u32 {
        20
    }

    fn default_tiles_enabled() -> bool {
        true
    }

    /// Return the configured depth, clamped to [`MAX_SCAN_DEPTH_LIMIT`].
    /// A depth of `0` means "scan the root directory only", so we keep the
    /// raw value as-is at the lower bound.
    pub fn effective_max_scan_depth(&self) -> u32 {
        self.max_scan_depth.min(MAX_SCAN_DEPTH_LIMIT)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_top_k: usize,
    /// Minimum cosine similarity threshold (0.0–1.0)
    pub similarity_threshold: f32,
    /// Maximum pHash Hamming distance (0–64, lower = stricter)
    pub phash_threshold: u32,
    /// When `true`, the query image is encoded at two scales (whole frame +
    /// centre 70 % crop) and the ANN hits are merged — a recall aid for
    /// screenshots with large blank margins, at the cost of doubling encoder
    /// time (the dominant latency stage).  Toggle and compare with
    /// `--evaluate` on the same seed to decide whether the second pass pays
    /// for itself on your corpus.
    #[serde(default = "SearchConfig::default_query_center_crop")]
    pub query_center_crop: bool,
    /// Fusion weight on embedding cosine similarity.  The three weights are
    /// re-tunable per corpus via `--evaluate`; the defaults come from the
    /// 2026-07 calibration run (visual dominates, text breaks same-layout
    /// ties, pHash breaks near-duplicate ties).
    #[serde(default = "SearchConfig::default_weight_vector")]
    pub weight_vector: f32,
    /// Fusion weight on pHash proximity.
    #[serde(default = "SearchConfig::default_weight_phash")]
    pub weight_phash: f32,
    /// Fusion weight on the OCR/FTS text score.
    #[serde(default = "SearchConfig::default_weight_text")]
    pub weight_text: f32,
}

impl SearchConfig {
    fn default_query_center_crop() -> bool {
        true
    }

    fn default_weight_vector() -> f32 {
        0.70
    }

    fn default_weight_phash() -> f32 {
        0.10
    }

    fn default_weight_text() -> f32 {
        0.20
    }
}

/// Filter configuration is intentionally empty — the sole exclusion
/// criterion (XMP `egExtFL:files` containing `.pdf` references) requires
/// no user-tunable parameters.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct FilterConfig {}

impl Config {
    /// Load configuration from a TOML file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Load with fallback to built-in defaults when no config file exists.
    pub fn load_or_default(path: &std::path::Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            tracing::warn!(
                "Config file not found at {}, using built-in defaults",
                path.display()
            );
            Ok(Self::default())
        }
    }

    /// Validate that required paths are sensible.
    fn validate(&self) -> Result<()> {
        if self.indexer.worker_threads == 0 {
            anyhow::bail!("indexer.worker_threads must be >= 1");
        }
        if self.indexer.batch_size == 0 {
            anyhow::bail!("indexer.batch_size must be >= 1");
        }
        Ok(())
    }

    /// Resolved path to the SQLite database file.
    pub fn db_path(&self) -> PathBuf {
        self.paths.data_dir.join("index.db")
    }

    /// Resolved base path for the HNSW vector index dump files.
    pub fn vector_index_path(&self) -> PathBuf {
        self.paths.data_dir.join("vectors.hnsw")
    }

    /// Resolved path to the thumbnails directory.
    pub fn thumbnails_dir(&self) -> PathBuf {
        self.paths.data_dir.join("thumbnails")
    }
}

impl Default for Config {
    fn default() -> Self {
        let worker_threads = (num_cpus::get() / 2).max(2);
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
            },
            paths: PathsConfig {
                scan_dirs: vec![],
                data_dir: PathBuf::from("data"),
                model_path: PathBuf::from("models/dinov2_vits14.onnx"),
                ocr_det_path: PathsConfig::default_ocr_det_path(),
                ocr_rec_path: PathsConfig::default_ocr_rec_path(),
            },
            indexer: IndexerConfig {
                worker_threads,
                batch_size: 8,
                watch_enabled: true,
                render_dpi: 150.0,
                max_scan_depth: IndexerConfig::default_max_scan_depth(),
                rescan_interval_secs: IndexerConfig::default_rescan_interval_secs(),
                watcher_settle_secs: IndexerConfig::default_watcher_settle_secs(),
                watcher_min_bytes: IndexerConfig::default_watcher_min_bytes(),
                watcher_max_retries: IndexerConfig::default_watcher_max_retries(),
                tiles_enabled: IndexerConfig::default_tiles_enabled(),
            },
            search: SearchConfig {
                default_top_k: 20,
                // DINOv2 similarity distribution is much wider than CLIP's —
                // a lenient floor only trims obvious noise; ranking does the
                // real work.  Re-calibrate with `--evaluate` after indexing.
                similarity_threshold: 0.30,
                phash_threshold: 12,
                query_center_crop: SearchConfig::default_query_center_crop(),
                weight_vector: SearchConfig::default_weight_vector(),
                weight_phash: SearchConfig::default_weight_phash(),
                weight_text: SearchConfig::default_weight_text(),
            },
            filter: FilterConfig::default(),
        }
    }
}