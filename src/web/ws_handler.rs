//! WebSocket handler for real-time index progress push.
//!
//! Clients connect to `GET /ws/progress` and receive JSON messages whenever
//! the indexing state changes, as well as periodic heartbeats.

use crate::{
    indexer::IndexProgress,
    storage::database::{self, DbPool},
};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::time::{interval, Duration};

/// Shared application state threaded into the Axum router.
#[derive(Clone)]
pub struct WsState {
    pub pool: DbPool,
    pub progress: Arc<IndexProgress>,
}

/// The JSON message sent to WebSocket clients.
#[derive(Debug, Serialize)]
pub struct ProgressMessage {
    #[serde(rename = "type")]
    pub msg_type: &'static str,
    pub status: &'static str,
    pub total_files: u64,
    pub indexed_files: u64,
    pub excluded_files: u64,
    pub failed_files: u64,
    pub total_pages: u64,
    pub progress_percent: f64,
}

/// Axum handler: upgrades the HTTP connection to WebSocket.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Drives the WebSocket connection: sends progress updates every 2 seconds.
async fn handle_socket(mut socket: WebSocket, state: WsState) {
    let mut tick = interval(Duration::from_secs(2));

    loop {
        tick.tick().await;

        // Fetch live stats from DB for persistent counters (files, pages)
        let db_stats = match database::get_index_stats(&state.pool).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("WS: failed to fetch index stats: {}", e);
                continue;
            }
        };

        // Live memory counters for indexing progress
        let processed = state.progress.processed.load(std::sync::atomic::Ordering::Relaxed);
        let excluded = state.progress.excluded.load(std::sync::atomic::Ordering::Relaxed);
        let failed = state.progress.failed.load(std::sync::atomic::Ordering::Relaxed);
        let total = state.progress.total.load(std::sync::atomic::Ordering::Relaxed);
        let is_finished = state.progress.finished.load(std::sync::atomic::Ordering::Acquire);

        // Progress percent: use memory counters while indexing; 100% if finished.
        let progress_percent = if is_finished {
            100.0_f64
        } else if total == 0 {
            if db_stats.total_files > 0 { 100.0 } else { 0.0 }
        } else {
            let done = processed + excluded + failed;
            (done as f64 / total as f64 * 100.0).min(100.0)
        };

        let status = if is_finished || progress_percent >= 100.0 {
            "ready"
        } else {
            "indexing"
        };

        let msg = ProgressMessage {
            msg_type: "status",
            status,
            total_files: db_stats.total_files as u64,
            indexed_files: db_stats.indexed_files as u64,
            excluded_files: db_stats.excluded_files as u64,
            failed_files: db_stats.failed_files as u64,
            total_pages: db_stats.total_pages as u64,
            progress_percent,
        };

        let json = match serde_json::to_string(&msg) {
            Ok(j) => j,
            Err(_) => continue,
        };

        if socket.send(Message::Text(json.into())).await.is_err() {
            // Client disconnected
            break;
        }
    }
}