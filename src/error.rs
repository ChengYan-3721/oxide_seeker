use thiserror::Error;

/// Unified application error type
#[derive(Debug, Error)]
pub enum AppError {
    // ── Storage ─────────────────────────────────────────────────────────────
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Database migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    // ── I/O ─────────────────────────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // ── Image processing ─────────────────────────────────────────────────────
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    // ── PDF processing ───────────────────────────────────────────────────────
    #[error("PDF error: {0}")]
    Pdf(String),

    // ── ONNX / CLIP ──────────────────────────────────────────────────────────
    #[error("ONNX runtime error: {0}")]
    Onnx(#[from] ort::Error),

    // ── Vector index ─────────────────────────────────────────────────────────
    #[error("Vector index error: {0}")]
    VectorIndex(String),

    // ── Configuration ────────────────────────────────────────────────────────
    #[error("Configuration error: {0}")]
    Config(String),

    // ── Web / HTTP ───────────────────────────────────────────────────────────
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Payload too large (max {max_bytes} bytes)")]
    PayloadTooLarge { max_bytes: usize },

    #[error("Unsupported media type: {mime}")]
    UnsupportedMediaType { mime: String },

    // ── Generic ────────────────────────────────────────────────────────���─────
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience alias
pub type Result<T, E = AppError> = std::result::Result<T, E>;

// ── Axum IntoResponse ────────────────────────────────────────────────────────

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::InvalidRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::PayloadTooLarge { max_bytes } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Payload too large (max {} bytes)", max_bytes),
            ),
            AppError::UnsupportedMediaType { mime } => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("Unsupported media type: {}", mime),
            ),
            _ => {
                tracing::error!(error = %self, "Internal server error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                )
            }
        };

        let body = Json(json!({
            "error": message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}