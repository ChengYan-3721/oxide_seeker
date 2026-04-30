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
    /// Path to the CLIP visual encoder ONNX model file
    pub model_path: PathBuf,
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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_top_k: usize,
    /// Minimum cosine similarity threshold (0.0–1.0)
    pub similarity_threshold: f32,
    /// Maximum pHash Hamming distance (0–64, lower = stricter)
    pub phash_threshold: u32,
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

    /// Resolved path to the usearch vector index file.
    pub fn vector_index_path(&self) -> PathBuf {
        self.paths.data_dir.join("vectors.usearch")
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
                model_path: PathBuf::from("models/clip_visual.onnx"),
            },
            indexer: IndexerConfig {
                worker_threads,
                batch_size: 8,
                watch_enabled: true,
                render_dpi: 150.0,
            },
            search: SearchConfig {
                default_top_k: 20,
                similarity_threshold: 0.65,
                phash_threshold: 12,
            },
            filter: FilterConfig::default(),
        }
    }
}