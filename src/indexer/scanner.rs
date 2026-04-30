//! File system scanner that discovers PDF and AI files under configured directories.
//!
//! Provides both a one-shot full scan and a helper to check whether a known file
//! has been modified since it was last indexed.

use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

/// Metadata for a discovered design file.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub filename: String,
    /// File type: `"pdf"` or `"ai"`
    pub file_type: String,
    pub file_size: Option<u64>,
    /// Modification time as Unix timestamp (seconds)
    pub modified_at: Option<i64>,
}

/// Scan every directory in `scan_dirs` recursively and return all PDF/AI files found.
///
/// Symbolic links are followed once (to avoid infinite loops).
pub fn scan_directories(scan_dirs: &[PathBuf]) -> Vec<DiscoveredFile> {
    let mut files = Vec::new();

    for root in scan_dirs {
        if !root.exists() {
            tracing::warn!("Scan directory does not exist: {}", root.display());
            continue;
        }

        tracing::info!("Scanning directory: {}", root.display());

        for entry in WalkDir::new(root)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| {
                match e {
                    Ok(entry) => Some(entry),
                    Err(err) => {
                        tracing::warn!("Scan error: {}", err);
                        None
                    }
                }
            })
        {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path().to_path_buf();
            let file_type = match classify_file(&path) {
                Some(t) => t,
                None => continue, // not PDF or AI
            };

            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            let meta = entry.metadata().ok();
            let file_size = meta.as_ref().map(|m| m.len());
            let modified_at = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64);

            files.push(DiscoveredFile {
                path,
                filename,
                file_type,
                file_size: file_size.map(|s| s as u64),
                modified_at,
            });
        }
    }

    tracing::info!("Scan complete: {} files found", files.len());
    files
}

/// Return `"pdf"`, `"ai"`, or `None` based on the file extension.
pub fn classify_file(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    match ext.as_str() {
        "pdf" => Some("pdf".to_string()),
        "ai" => Some("ai".to_string()),
        _ => None,
    }
}

/// Check whether the file at `path` has been modified since `last_indexed_at`.
///
/// Returns `true` if the file should be re-indexed (modified or not yet indexed).
pub fn needs_reindex(path: &Path, last_indexed_at: Option<i64>) -> bool {
    let Some(indexed_ts) = last_indexed_at else {
        return true; // never indexed
    };

    let current_mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);

    match current_mtime {
        Some(mtime) => mtime > indexed_ts,
        None => false, // can't read mtime, skip
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classify_pdf() {
        assert_eq!(
            classify_file(&PathBuf::from("design.PDF")),
            Some("pdf".to_string())
        );
    }

    #[test]
    fn classify_ai() {
        assert_eq!(
            classify_file(&PathBuf::from("logo.ai")),
            Some("ai".to_string())
        );
    }

    #[test]
    fn classify_unknown() {
        assert_eq!(classify_file(&PathBuf::from("image.png")), None);
        assert_eq!(classify_file(&PathBuf::from("document.docx")), None);
    }

    #[test]
    fn needs_reindex_when_never_indexed() {
        // A file that has never been indexed always needs reindexing
        let path = PathBuf::from("nonexistent_file.pdf");
        assert!(needs_reindex(&path, None));
    }
}