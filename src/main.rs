//! OxideSeeker — Visual file search for prepress design files.
//!
//! Usage:
//!   oxide_seeker [--config <path>]
//!
//! Defaults to `config.toml` in the current directory.

mod config;
mod crash_handler;
mod embedder;
mod error;
mod indexer;
mod search;
mod storage;
mod license;
mod web;
mod worker_proc;

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
use single_instance::SingleInstance;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// Sentinel CLI flag that re-launches this same binary as an indexing worker
/// subprocess.  Detected before the tokio runtime / single-instance check so
/// children don't fight the parent for the global lock.
pub const WORKER_MODE_FLAG: &str = "--worker-mode";

fn main() -> anyhow::Result<()> {
    // Re-execute paths first: if we were spawned as a worker subprocess, run
    // the synchronous request loop and exit when stdin closes.  Done before
    // any tokio / single-instance / chdir setup since those would interfere
    // with parent-controlled stdin/stdout pipes.
    if std::env::args().any(|a| a == WORKER_MODE_FLAG) {
        // Set working directory to the exe folder so relative paths used by
        // pdfium / models behave the same as in the parent.
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                let _ = std::env::set_current_dir(dir);
            }
        }
        return worker_proc::run();
    }

    // Build a multi-thread tokio runtime by hand so the worker-mode early
    // return above doesn't pay the cost.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    // 检查程序是否已经在运行 (使用唯一的字符串标识)
    let instance = SingleInstance::new("oxide_seeker_lock")?;
    if !instance.is_single() {
        return Ok(());
    }

    // 设置工作目录为 exe 所在目录 (解决相对路径问题，如找不到 config.toml)，否则开机启动会失败
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let _ = std::env::set_current_dir(dir);
        }
    }

    // Logging: console (info+) + rolling file (warn+, daily, keep 7 days).
    // The file layer writes synchronously: a non_blocking worker thread would
    // be killed by an FFI segfault before flushing its buffer, so warn+ records
    // emitted near a crash would be lost.
    let log_dir = std::path::Path::new("./logs");
    std::fs::create_dir_all(log_dir).ok();

    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("oxide_seeker")
        .filename_suffix("log")
        .max_log_files(7)
        .build(log_dir)
        .expect("failed to initialise rolling file logger");

    let console_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("oxide_seeker=info,warn"));
    let file_filter = EnvFilter::new("warn");

    let console_layer = tracing_subscriber::fmt::layer()
        .with_timer(time::LocalTime::rfc_3339())
        .with_target(false)
        .compact()
        .with_filter(console_filter);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_timer(time::LocalTime::rfc_3339())
        .with_target(false)
        .with_ansi(false)
        .with_writer(file_appender)
        .with_filter(file_filter);

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    // Crash recording: panic hook + Windows SEH filter writing to crash.log.
    crash_handler::install(log_dir, "parent");

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

    // License state init
    let license_state = license::evaluate_and_persist(&pool).await?;
    tracing::info!("License status: {} - {}", license_state.status, license_state.message);

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
            vector_index.clone(),
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
                vector_index.clone(),
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
    let thumbnails_dir = config.thumbnails_dir();

    let router = web::build_router(
        engine,
        pool.clone(),
        progress,
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
