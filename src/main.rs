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
mod evaluate;
mod indexer;
#[cfg(windows)]
mod job_object;
mod ocr;
mod search;
mod storage;
mod license;
mod web;
mod worker_proc;

use crate::{
    config::Config,
    embedder::vision::VisionEmbedder,
    search::{PhashStore, SearchEngine, VectorIndex},
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

/// CLI flag passed by `oxide_seeker_service.exe` when it spawns this binary.
/// Enables the stdin watchdog: when the service drops the child's stdin
/// pipe, the watchdog triggers `process::exit(0)` so the Job Object cleans
/// up the worker subprocesses.
pub const SERVICE_MODE_FLAG: &str = "--service-mode";

/// Global Job Object that owns every `--worker-mode` subprocess spawned by
/// this main process.  When the main process exits for any reason (clean
/// shutdown, panic, SEH crash, `TerminateProcess` from the service wrapper),
/// the kernel closes this handle and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`
/// terminates every worker synchronously.  No orphan workers can survive.
#[cfg(windows)]
pub static WORKER_JOB: once_cell::sync::OnceCell<job_object::JobObject> =
    once_cell::sync::OnceCell::new();

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

    // Offline evaluation harness: measures retrieval quality against
    // synthetic queries cut from the indexed corpus.  Runs the search stack
    // only — no indexer, watcher, web server, or single-instance lock (it is
    // expected to run alongside a live service instance).  CWD is left
    // untouched so `--config`/`--out` resolve relative to the caller's shell.
    if std::env::args().any(|a| a == "--evaluate") {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        return runtime.block_on(evaluate::run());
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

    // Worker process containment: create the parent's Job Object as early as
    // possible so every subsequent worker spawn can join it.  Workers added
    // here die automatically when this process exits — no orphans regardless
    // of how the death happened.
    #[cfg(windows)]
    match job_object::JobObject::new_kill_on_close() {
        Ok(job) => {
            let _ = WORKER_JOB.set(job);
        }
        Err(e) => {
            tracing::warn!(
                "Failed to create worker Job Object ({}). Workers may outlive parent on crash.",
                e
            );
        }
    }

    // When launched by the Windows service wrapper, watch stdin for EOF.
    // The service's `stop_oxide_seeker` drops the child stdin pipe to
    // request a graceful shutdown; here we just exit, which tears down
    // workers via the Job Object above.
    if std::env::args().any(|a| a == SERVICE_MODE_FLAG) {
        std::thread::spawn(|| {
            use std::io::Read;
            let mut buf = [0u8; 64];
            let stdin = std::io::stdin();
            let mut handle = stdin.lock();
            loop {
                match handle.read(&mut buf) {
                    Ok(0) => break,
                    Err(_) => break,
                    Ok(_) => continue,
                }
            }
            tracing::info!("Service shutdown signal received (parent stdin closed); exiting");
            std::process::exit(0);
        });
    }

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

    // Vector index (HNSW dump; rebuilt from DB below if missing/lagging)
    let vector_index = VectorIndex::open(&config.vector_index_path())?;

    // Rebuild the ANN graph from the vectors persisted in SQLite when the
    // dump is missing or clearly behind the DB (corruption, deleted files,
    // tombstone compaction).  Streams blobs — no model inference involved.
    search::rebuild_index_if_needed(&pool, &vector_index).await?;

    // In-memory pHash table (one u64 per region, scanned with rayon)
    let phash_store = Arc::new(PhashStore::new());
    phash_store.replace_all(database::load_phash_entries(&pool).await?);

    // Vision encoder model
    let embedder = match VisionEmbedder::load(&config.paths.model_path) {
        Ok(c) => {
            tracing::info!("Vision model loaded from {}", config.paths.model_path.display());
            Arc::new(c)
        }
        Err(e) => {
            tracing::warn!(
                "Vision model not available ({}). \
                 Place the exported model at: {}",
                e,
                config.paths.model_path.display()
            );
            return Err(e.into());
        }
    };

    // Query-side OCR engine — optional: absent models degrade to
    // visual-only search with a warning.
    let ocr_engine = match ocr::OcrEngine::load(
        &config.paths.ocr_det_path,
        &config.paths.ocr_rec_path,
    ) {
        Ok(e) => {
            tracing::info!("OCR text channel enabled");
            Some(Arc::new(parking_lot::Mutex::new(e)))
        }
        Err(e) => {
            tracing::warn!("OCR text channel disabled ({}).", e);
            None
        }
    };

    // Search engine
    let engine = SearchEngine::new(
        pool.clone(),
        &embedder,
        ocr_engine.clone(),
        vector_index.clone(),
        phash_store.clone(),
        config.search.clone(),
    )?;

    // Background OCR backfill: renders + recognises pages missing a
    // page_ocr row.  No-op (with a warning) when the models are absent.
    if ocr_engine.is_some() {
        indexer::ocr_backfill::spawn(config.clone(), pool.clone());
    }

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
            phash_store.clone(),
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
                phash_store.clone(),
            )
            .await?;

            tracing::info!("File watcher active");

            // Keep the watcher alive for the lifetime of the process by leaking
            // it into a Box -- it will be dropped when the process exits.
            Box::leak(Box::new(_watcher));
        }

        // Periodic full rescan — safety net for events the OS-level watcher
        // missed.  notify on SMB / network shares is famously unreliable, and
        // the watcher's settle-check can also drop a file if the writing app
        // takes longer than `watcher_max_retries * watcher_settle_secs` to
        // finish flushing.  This loop re-runs `start_full_index`, which is
        // cheap when nothing changed (mtime + hash short-circuits) and
        // automatically rebuilds anything the watcher dropped.
        let rescan_secs = config.indexer.rescan_interval_secs;
        if rescan_secs > 0 {
            let cfg = config.clone();
            let p = pool.clone();
            let v = vector_index.clone();
            let ph = phash_store.clone();
            let prog = progress.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(std::time::Duration::from_secs(rescan_secs));
                tick.tick().await; // skip the immediate first tick
                loop {
                    tick.tick().await;
                    if !prog.finished.load(std::sync::atomic::Ordering::Acquire) {
                        tracing::debug!(
                            "Periodic rescan: previous batch still running, deferring"
                        );
                        continue;
                    }
                    tracing::info!("Periodic rescan starting");
                    if let Err(e) = indexer::start_full_index(
                        cfg.clone(),
                        p.clone(),
                        v.clone(),
                        ph.clone(),
                        prog.clone(),
                    )
                    .await
                    {
                        tracing::warn!("Periodic rescan failed: {}", e);
                    }
                }
            });
            tracing::info!("Periodic rescan enabled (every {}s)", rescan_secs);
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
        embedder,
        vector_index,
        phash_store,
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
