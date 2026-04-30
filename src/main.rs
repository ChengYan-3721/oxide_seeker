//! OxideSeeker — Visual file search for prepress design files.
//!
//! Usage:
//!   oxide_seeker [--config <path>]
//!
//! Defaults to `config.toml` in the current directory.

mod config;
mod embedder;
mod error;
mod indexer;
mod search;
mod storage;
mod web;

use crate::{
    config::Config,
    embedder::clip::ClipEmbedder,
    search::{SearchEngine, VectorIndex},
    storage::{
        database,
        thumbnail::ThumbnailStore,
    },
};
use std::{path::PathBuf, sync::Arc};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_timer(time::LocalTime::rfc_3339())
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("oxide_seeker=info,warn")),
        )
        .with_target(false)
        .compact()
        .init();

    // Config
    let config_path = parse_config_arg();
    let config = Config::load_or_default(&config_path)?;
    let config = Arc::new(config);

    tracing::info!("OxideSeeker v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Data directory: {}", config.paths.data_dir.display());

    // Storage init
    let pool = database::init_pool(&config.db_path()).await?;

    let thumb_store = Arc::new(
        ThumbnailStore::new(&config.thumbnails_dir()).await?,
    );

    // Vector index
    let vector_index = VectorIndex::open(&config.vector_index_path())?;

    // CLIP model
    let clip = match ClipEmbedder::load(&config.paths.model_path) {
        Ok(c) => {
            tracing::info!("CLIP model loaded from {}", config.paths.model_path.display());
            Arc::new(c)
        }
        Err(e) => {
            tracing::warn!(
                "CLIP model not available ({}). Vector search disabled. \
                 Only pHash search will work until the model is placed at: {}",
                e,
                config.paths.model_path.display()
            );
            // Proceed without CLIP -- pHash-only mode
            // For a production deployment this should be a hard error.
            // Here we allow startup so the operator can place the model file.
            return Err(e.into());
        }
    };

    // Search engine
    let engine = SearchEngine::new(
        pool.clone(),
        &clip,
        vector_index.clone(),
        config.search.clone(),
    )?;

    // Create the shared progress tracker BEFORE starting the indexer so that
    // the exact same Arc is passed to both the indexer worker pool and the
    // WebSocket handler.  Previously a fresh IndexProgress::new(0) was created
    // after the indexer returned, giving the WS handler a permanently-zero stub.
    let progress = indexer::IndexProgress::new(0);

    // Initial full index
    if config.paths.scan_dirs.is_empty() {
        tracing::warn!(
            "No scan directories configured (paths.scan_dirs is empty). \
             Edit config.toml to add directories to index."
        );
    } else {
        indexer::start_full_index(
            config.clone(),
            pool.clone(),
            clip.clone(),
            vector_index.clone(),
            thumb_store.clone(),
            progress.clone(),
        )
        .await?;

        tracing::info!(
            "Full index started: {} files queued",
            progress.total.load(std::sync::atomic::Ordering::Relaxed)
        );

        // File watcher (incremental updates)
        if config.indexer.watch_enabled {
            let _watcher = indexer::watcher::start_watcher(
                config.clone(),
                pool.clone(),
                clip.clone(),
                vector_index.clone(),
                thumb_store.clone(),
            )
            .await?;

            tracing::info!("File watcher active");

            // Keep the watcher alive for the lifetime of the process by leaking
            // it into a Box -- it will be dropped when the process exits.
            Box::leak(Box::new(_watcher));
        }
    }

    // Web server
    // `progress` is the same Arc shared with the indexer worker pool so
    // WebSocket clients see live counters.
    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/web/static");

    // Thumbnails are stored under data_dir/thumbnails at runtime; pass the
    // resolved path so the static file service points to the right directory.
    let thumbnails_dir = config.thumbnails_dir();

    let router = web::build_router(
        engine,
        pool.clone(),
        progress,
        &static_dir,
        &thumbnails_dir,
        config_path,
        clip,
        vector_index,
        thumb_store,
    );

    web::run_server(router, &config).await?;

    Ok(())
}

/// Parse `--config <path>` CLI argument, defaulting to `config.toml`.
fn parse_config_arg() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        } else if let Some(path) = arg.strip_prefix("--config=") {
            return PathBuf::from(path);
        }
    }
    PathBuf::from("config.toml")
}
