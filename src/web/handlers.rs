//! Axum HTTP request handlers.

use crate::{
    error::{AppError, Result},
    indexer::{self, IndexProgress},
    search::{SearchEngine, SearchResponse, vector_index::VectorIndex},
    storage::{database::{self, DbPool}, thumbnail::ThumbnailStore},
    embedder::clip::ClipEmbedder,
};
use axum::{
    body::Bytes,
    extract::{Multipart, State},
    Json,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;

/// Maximum upload size: 20 MB
const MAX_UPLOAD_BYTES: usize = 20 * 1024 * 1024;

// ── Shared handler state ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub engine: SearchEngine,
    pub pool: DbPool,
    pub progress: Arc<IndexProgress>,
    pub config_path: PathBuf,
    pub clip: Arc<ClipEmbedder>,
    pub vector_index: Arc<VectorIndex>,
    pub thumb_store: Arc<ThumbnailStore>,
}

// ── POST /api/search  (multipart) ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SearchQuery {
    pub top_k: Option<usize>,
}

/// Upload an image file and return the top-K most similar pages.
pub async fn search_upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<SearchResponseBody>> {
    let mut image_bytes: Option<Bytes> = None;
    let mut top_k: Option<usize> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::InvalidRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" => {
                // Validate content-type
                let ct = field
                    .content_type()
                    .unwrap_or("")
                    .to_string();
                if !ct.starts_with("image/") && !ct.is_empty() {
                    return Err(AppError::UnsupportedMediaType { mime: ct });
                }
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::InvalidRequest(e.to_string()))?;
                if bytes.len() > MAX_UPLOAD_BYTES {
                    return Err(AppError::PayloadTooLarge {
                        max_bytes: MAX_UPLOAD_BYTES,
                    });
                }
                image_bytes = Some(bytes);
            }
            "top_k" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| AppError::InvalidRequest(e.to_string()))?;
                top_k = v.parse::<usize>().ok();
            }
            _ => {}
        }
    }

    let bytes = image_bytes
        .ok_or_else(|| AppError::InvalidRequest("Missing 'image' field in multipart".into()))?;

    let resp = state.engine.search_bytes(&bytes, top_k).await?;
    Ok(Json(resp.into()))
}

// ── POST /api/search/clipboard  (base64 JSON) ────────────────────────────────

#[derive(Deserialize)]
pub struct ClipboardRequest {
    /// `data:image/png;base64,<data>` or raw base64
    pub image_base64: String,
    pub top_k: Option<usize>,
}

/// Accept a base64-encoded image (clipboard paste) and search.
pub async fn search_clipboard(
    State(state): State<AppState>,
    Json(req): Json<ClipboardRequest>,
) -> Result<Json<SearchResponseBody>> {
    // Strip the data-URI prefix if present
    let raw = if let Some(pos) = req.image_base64.find(',') {
        &req.image_base64[pos + 1..]
    } else {
        &req.image_base64
    };

    let bytes = B64
        .decode(raw.trim())
        .map_err(|e| AppError::InvalidRequest(format!("Invalid base64: {}", e)))?;

    if bytes.len() > MAX_UPLOAD_BYTES {
        return Err(AppError::PayloadTooLarge {
            max_bytes: MAX_UPLOAD_BYTES,
        });
    }

    let resp = state.engine.search_bytes(&bytes, req.top_k).await?;
    Ok(Json(resp.into()))
}

// ── GET /api/index/status ─────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct IndexStatusResponse {
    pub status: String,
    pub total_files: i64,
    pub indexed_files: i64,
    pub excluded_files: i64,
    pub failed_files: i64,
    pub total_pages: i64,
    pub progress_percent: f64,
}

pub async fn index_status(State(state): State<AppState>) -> Result<Json<IndexStatusResponse>> {
    let stats = database::get_index_stats(&state.pool).await?;
    
    // Use memory counters for live progress, as they are the source of truth for the current batch
    let processed = state.progress.processed.load(std::sync::atomic::Ordering::Relaxed);
    let excluded = state.progress.excluded.load(std::sync::atomic::Ordering::Relaxed);
    let failed = state.progress.failed.load(std::sync::atomic::Ordering::Relaxed);
    let total = state.progress.total.load(std::sync::atomic::Ordering::Relaxed);
    let is_finished = state.progress.finished.load(std::sync::atomic::Ordering::Acquire);

    let progress_percent = if is_finished {
        100.0
    } else if total == 0 {
        if stats.total_files > 0 { 100.0 } else { 0.0 }
    } else {
        let done = processed + excluded + failed;
        (done as f64 / total as f64 * 100.0).min(100.0)
    };

    let status = if is_finished || progress_percent >= 100.0 {
        "ready"
    } else {
        "indexing"
    };

    Ok(Json(IndexStatusResponse {
        status: status.to_string(),
        total_files: stats.total_files,
        indexed_files: stats.indexed_files,
        excluded_files: stats.excluded_files,
        failed_files: stats.failed_files,
        total_pages: stats.total_pages,
        progress_percent,
    }))
}

#[derive(Deserialize, Serialize)]
pub struct ConfigResponse {
    pub scan_dirs: Vec<String>,
    pub is_local: bool,
}

#[derive(Deserialize)]
pub struct UpdateConfig {
    pub scan_dirs: Vec<String>,
}

/// Check if a request is from the local machine.
///
/// Returns true if the connection comes from a loopback IP (covering both
/// IPv4 `127.0.0.0/8` and IPv6 `::1`, plus IPv4-mapped IPv6 like
/// `::ffff:127.0.0.1`), OR if the `Host` header indicates a localhost
/// hostname. The hostname check fixes scenarios where Windows / virtual
/// adapters / hosts-file overrides make the peer IP non-loopback even when
/// the user is accessing via `http://localhost`.
fn is_local_request(addr: std::net::SocketAddr, host_header: Option<&str>) -> bool {
    let ip = addr.ip();
    let is_loopback_ip = match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback() || {
                // IPv4-mapped IPv6: ::ffff:a.b.c.d where a == 127
                let s = v6.segments();
                s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 && s[4] == 0
                    && s[5] == 0xffff
                    && (s[6] >> 8) == 127
            }
        }
    };
    if is_loopback_ip {
        return true;
    }

    if let Some(host) = host_header {
        let hostname = extract_hostname(host).to_ascii_lowercase();
        if matches!(hostname.as_str(), "localhost" | "127.0.0.1" | "::1") {
            return true;
        }
    }

    tracing::warn!("Non-local access attempt from IP: {} (host header: {:?})", ip, host_header);
    false
}

/// Extract the hostname portion of a `Host` header value, stripping the port
/// and IPv6 brackets if present. Examples:
/// - `localhost:7788` -> `localhost`
/// - `127.0.0.1`      -> `127.0.0.1`
/// - `[::1]:7788`     -> `::1`
fn extract_hostname(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return &rest[..end];
        }
    }
    host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host)
}

pub async fn get_config(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<ConfigResponse>> {
    let content = std::fs::read_to_string(&state.config_path)
        .map_err(|e| {
            tracing::error!("Failed to read config at {:?}: {}", state.config_path, e);
            AppError::Other(anyhow::anyhow!("Failed to read config: {}", e))
        })?;
    let config: crate::config::Config = toml::from_str(&content)
        .map_err(|e| {
            tracing::error!("Failed to parse config at {:?}: {}", state.config_path, e);
            AppError::Other(anyhow::anyhow!("Failed to parse config: {}", e))
        })?;

    let host = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok());
    let is_local = is_local_request(addr, host);
    tracing::info!("is_local for {} (host={:?}): {}", addr, host, is_local);

    Ok(Json(ConfigResponse {
        scan_dirs: config.paths.scan_dirs.into_iter().map(|p| p.to_string_lossy().to_string()).collect(),
        is_local,
    }))
}

pub async fn update_config(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<UpdateConfig>,
) -> Result<Json<serde_json::Value>> {
    let host = headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok());
    if !is_local_request(addr, host) {
        return Err(AppError::InvalidRequest("Only local connections can change configuration".into()));
    }

    tracing::info!("Updating config with scan_dirs: {:?}", req.scan_dirs);

    // 1. Update config.toml
    let content = std::fs::read_to_string(&state.config_path)
        .map_err(|e| {
            tracing::error!("Failed to read config for update: {}", e);
            AppError::Other(anyhow::anyhow!("Failed to read config: {}", e))
        })?;
    let mut config_val: toml::Value = toml::from_str(&content)
        .map_err(|e| {
            tracing::error!("Failed to parse config for update: {}", e);
            AppError::Other(anyhow::anyhow!("Failed to parse config: {}", e))
        })?;
    
    // Simple update logic (assuming specific structure)
    if let Some(paths) = config_val.get_mut("paths") {
        if let Some(paths_table) = paths.as_table_mut() {
            paths_table.insert("scan_dirs".to_string(), toml::Value::try_from(&req.scan_dirs)
                .map_err(|e| {
                    tracing::error!("Failed to convert scan_dirs to toml: {}", e);
                    AppError::Other(anyhow::anyhow!("Failed to convert scan_dirs: {}", e))
                })?);
        } else {
            tracing::error!("'paths' in config is not a table");
            return Err(AppError::Other(anyhow::anyhow!("'paths' in config is not a table")));
        }
    } else {
        tracing::error!("'paths' section not found in config");
        return Err(AppError::Other(anyhow::anyhow!("'paths' section not found in config")));
    }

    let new_toml = toml::to_string_pretty(&config_val)
        .map_err(|e| {
            tracing::error!("Failed to serialize new config to TOML: {}", e);
            AppError::Other(anyhow::anyhow!("Failed to serialize config: {}", e))
        })?;
    std::fs::write(&state.config_path, new_toml)
        .map_err(|e| {
            tracing::error!("Failed to write config at {:?}: {}", state.config_path, e);
            AppError::Other(anyhow::anyhow!("Failed to write config: {}", e))
        })?;

    // 2. Trigger re-index without restart
    let new_config: crate::config::Config = toml::from_str(&toml::to_string(&config_val).unwrap()).unwrap();
    let config_arc = Arc::new(new_config);
    
    // Reset progress for new index
    state.progress.processed.store(0, std::sync::atomic::Ordering::SeqCst);
    state.progress.excluded.store(0, std::sync::atomic::Ordering::SeqCst);
    state.progress.failed.store(0, std::sync::atomic::Ordering::SeqCst);
    state.progress.total.store(0, std::sync::atomic::Ordering::SeqCst);
    state.progress.finished.store(false, std::sync::atomic::Ordering::Release);

    let pool = state.pool.clone();
    let clip = state.clip.clone();
    let vector_index = state.vector_index.clone();
    let thumb_store = state.thumb_store.clone();
    let progress = state.progress.clone();

    tokio::spawn(async move {
        if let Err(e) = indexer::start_full_index(
            config_arc,
            pool,
            clip,
            vector_index,
            thumb_store,
            progress,
        ).await {
            tracing::error!("Failed to trigger re-index after config update: {}", e);
        }
    });

    tracing::info!("Config updated successfully, re-index triggered");
    Ok(Json(serde_json::json!({"status": "ok", "message": "Configuration updated. Re-indexing started."})))
}

// ── Response type conversion ──────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SearchResponseBody {
    pub results: Vec<crate::search::ranker::SearchResult>,
    pub total: usize,
    pub search_time_ms: u64,
}

impl From<SearchResponse> for SearchResponseBody {
    fn from(r: SearchResponse) -> Self {
        let total = r.results.len();
        Self {
            results: r.results,
            total,
            search_time_ms: r.search_time_ms,
        }
    }
}