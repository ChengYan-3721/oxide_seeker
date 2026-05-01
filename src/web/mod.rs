//! Axum web server: routes, middleware, and static file serving.

pub mod handlers;
pub mod ws_handler;

use crate::{
    config::Config,
    indexer::IndexProgress,
    search::{SearchEngine, vector_index::VectorIndex},
    storage::{database::DbPool, thumbnail::ThumbnailStore},
    embedder::clip::ClipEmbedder,
   web::{
       handlers::{
           get_config, get_license_status, index_status, search_clipboard, search_upload,
           update_config, update_license, AppState,
       },
       ws_handler::{ws_handler, WsState},
   },
};
use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use std::{net::SocketAddr, path::Path, sync::Arc};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};

/// Build the Axum application router with all routes configured.
pub fn build_router(
    engine: SearchEngine,
    pool: DbPool,
    progress: Arc<IndexProgress>,
    static_dir: &Path,
    thumbnails_dir: &Path,
    config_path: std::path::PathBuf,
    clip: Arc<ClipEmbedder>,
    vector_index: Arc<VectorIndex>,
    thumb_store: Arc<ThumbnailStore>,
) -> Router {
    let app_state = AppState {
        engine,
        pool: pool.clone(),
        progress: progress.clone(),
        config_path,
        clip,
        vector_index,
        thumb_store,
    };

    let ws_state = WsState {
        pool,
        progress,
    };

    // CORS: allow all origins on the LAN (internal tool)
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // API routes
        .route("/api/search", post(search_upload))
        .route("/api/search/clipboard", post(search_clipboard))
       .route("/api/index/status", get(index_status))
       .route("/api/config", get(get_config).post(update_config))
       .route("/api/license", get(get_license_status).post(update_license))
        .with_state(app_state)
        .layer(DefaultBodyLimit::max(20 * 1024 * 1024))
        // WebSocket route (different state type)
        .route("/ws/progress", get(ws_handler))
        .with_state(ws_state)
        // Static files
        .nest_service("/thumbnails", ServeDir::new(thumbnails_dir))
        .nest_service("/static", ServeDir::new(static_dir))
        .route("/", get(serve_index))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}

/// Serve the main HTML page.
async fn serve_index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("static/index.html"))
}

/// Start the HTTP server, binding to `host:port` from config.
pub async fn run_server(
    router: Router,
    config: &Config,
) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;

    tracing::info!("OxideSeeker listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener, 
        router.into_make_service_with_connect_info::<SocketAddr>()
    ).await?;
    Ok(())
}