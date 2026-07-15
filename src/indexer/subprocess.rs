//! Out-of-process indexing worker (parent-side client + shared protocol types).
//!
//! Why
//! ---
//! pdfium and ONNX Runtime can both raise unrecoverable structured exceptions
//! on malformed input (the `0xE0000008` we saw in production was pdfium
//! aborting from inside `KERNELBASE`).  Rust `catch_unwind` does not catch
//! SEH, so a single bad file would otherwise terminate the entire indexer.
//!
//! Architecture
//! ------------
//! Each parent indexing thread keeps one long-lived child process — the same
//! binary re-invoked with `--worker-mode`.  The child loads pdfium + the CLIP
//! ONNX session once, then handles request frames over stdin/stdout in a
//! synchronous loop.  When the child crashes the parent observes the broken
//! pipe, logs the failure, and respawns a fresh child for the next file.
//!
//! Wire format
//! -----------
//! Length-prefixed bincode frames.  Every direction:
//! ```text
//!   [u32 LE: payload_size][bincode-serialised payload]
//! ```
//! The first frame the parent sends is a [`WorkerInit`].  After that it sends
//! [`ProcessRequest`]s and expects one [`ProcessResponse`] per request.

use crate::error::{AppError, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Parameters the parent passes to the child once after spawning.
#[derive(Serialize, Deserialize, Debug)]
pub struct WorkerInit {
    pub model_path: PathBuf,
    pub thumbnails_dir: PathBuf,
    /// PP-OCR model paths; loaded lazily on the first [`WorkerRequest::OcrPage`].
    pub ocr_det_path: PathBuf,
    pub ocr_rec_path: PathBuf,
    /// Used purely for logging on the child side so log lines from different
    /// workers can be told apart.
    pub worker_idx: usize,
    /// When `true`, each page additionally emits 9 overlapping tiles
    /// (size = ½ page, stride = ¼ page).  See `IndexerConfig::tiles_enabled`.
    pub tiles_enabled: bool,
}

/// One unit of work for the child.  Every variant is watchdog-protected on
/// the child side: an FFI call that neither returns nor crashes (observed
/// with pdfium on a poison page at OCR resolution) makes the child
/// `exit(3)` after a deadline, which the parent sees as a broken pipe and
/// handles like any crash — respawn and move on.
#[derive(Serialize, Deserialize, Debug)]
pub enum WorkerRequest {
    /// Render + embed + pHash a whole file (the indexing pipeline).
    Index(ProcessRequest),
    /// Render one page at OCR resolution and extract its text.
    OcrPage { file_path: PathBuf, page_num: i64 },
}

/// Response for a [`WorkerRequest`], same order.
#[derive(Serialize, Deserialize, Debug)]
pub enum WorkerResponse {
    Index(ProcessResponse),
    /// Extracted text ("" when the page has none / OCR failed softly).
    OcrPage { text: String },
}

/// One file the parent wants processed.
#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessRequest {
    pub file_path: PathBuf,
    /// Pre-allocated DB row id.  The child uses it to name the thumbnail file
    /// it writes; the parent uses it as the foreign key for the pages rows it
    /// inserts after receiving the response.
    pub file_id: i64,
}

/// Outcome of processing a single file.  Splits the historical
/// `Result<FileOutcome, AppError>` into a serialisable enum.
#[derive(Serialize, Deserialize, Debug)]
pub enum ProcessResponse {
    /// File matched the imposition exclusion rule.
    Excluded { reason: String },
    /// File processed successfully; one page entry per rendered page.
    Indexed { pages: Vec<PageData> },
    /// Recoverable Rust-level error (malformed PDF, image decode failure,
    /// etc.).  The parent treats this as a normal failure and moves on.
    Error { msg: String },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PageData {
    pub page_num: i32,
    pub width_px: u32,
    pub height_px: u32,
    /// Path of the saved thumbnail relative to the thumbnails dir.  Child
    /// writes the file directly to disk so the parent does not have to ship
    /// thumbnail bytes back through the pipe.
    pub thumb_relative_path: String,
    /// Per-region embeddings.  Always at least one entry — the full-page
    /// region — followed by 9 overlapping tiles when tiles are enabled.
    /// The parent persists one `regions` row per entry.
    pub regions: Vec<RegionData>,
}

/// Sub-region of a page that gets its own pHash + embedding.
#[derive(Serialize, Deserialize, Debug)]
pub struct RegionData {
    pub kind: RegionKind,
    /// Row-major position within the tile grid (0 for full-page rows).
    pub index: u32,
    /// Normalised bbox in page coordinates (`x`, `y`, `w`, `h`, all in
    /// `[0, 1]`).  The full-page region is `(0, 0, 1, 1)`.
    pub bbox: RegionBBox,
    /// 64-bit perceptual hash of this region's pixels.
    pub phash: u64,
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Whole-page embedding.  Always present.
    Full,
    /// One of the 9 overlapping tiles (size = ½ page, stride = ¼ page).
    Tile,
}

impl RegionKind {
    /// Lower-case string used in the `regions.kind` SQL column.
    pub fn as_sql(self) -> &'static str {
        match self {
            RegionKind::Full => "full",
            RegionKind::Tile => "tile",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct RegionBBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

// ── Framing helpers ──────────────────────────────────────────────────────────

/// Write a length-prefixed bincode frame.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value).map_err(|e| {
        AppError::Other(anyhow::anyhow!("subprocess serialise: {}", e))
    })?;
    let len = u32::try_from(bytes.len()).map_err(|_| {
        AppError::Other(anyhow::anyhow!(
            "subprocess frame too large: {} bytes",
            bytes.len()
        ))
    })?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Read a length-prefixed bincode frame.  Returns `Ok(None)` on clean EOF
/// (the peer closed the pipe before sending anything), `Err(_)` on partial
/// frames or deserialisation failures.
pub fn read_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(AppError::Io(e)),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    let value = bincode::deserialize(&payload).map_err(|e| {
        AppError::Other(anyhow::anyhow!("subprocess deserialise: {}", e))
    })?;
    Ok(Some(value))
}

// ── Parent-side client ───────────────────────────────────────────────────────

/// Long-lived handle to one indexing worker subprocess.  Drop kills the
/// child and waits for it.
#[allow(dead_code)]
pub struct WorkerProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    pub worker_idx: usize,
}

impl WorkerProcess {
    /// Spawn `oxide_seeker --worker-mode` and send the `WorkerInit` frame.
    /// The returned handle is ready to call [`process`](Self::process).
    pub fn spawn(
        model_path: &Path,
        thumbnails_dir: &Path,
        ocr_det_path: &Path,
        ocr_rec_path: &Path,
        worker_idx: usize,
        tiles_enabled: bool,
    ) -> Result<Self> {
        // Re-exec the same binary so deployment stays a single file.
        let exe = std::env::current_exe().map_err(AppError::Io)?;

        let mut child = Command::new(&exe)
            .arg(crate::WORKER_MODE_FLAG)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr inherits the parent's so worker tracing / panic output
            // shows up on the same console.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                AppError::Other(anyhow::anyhow!(
                    "Failed to spawn worker subprocess ({}): {}",
                    exe.display(),
                    e
                ))
            })?;

        // Tie the worker's lifetime to the parent's Job Object.  If the
        // parent dies before reaching its own Drop / kill code path (panic,
        // SEH crash, kill from the service wrapper), the kernel closes the
        // Job handle and the worker is terminated synchronously.  Done
        // before the Init frame so a concurrent parent-death window can't
        // leave a worker mid-handshake without Job protection.
        #[cfg(windows)]
        if let Some(job) = crate::WORKER_JOB.get() {
            if let Err(e) = job.assign_child(&child) {
                tracing::warn!(
                    "[worker {}] Failed to assign worker to Job Object: {}. Worker may outlive parent on crash.",
                    worker_idx,
                    e
                );
            }
        }

        let stdin = child.stdin.take().ok_or_else(|| {
            AppError::Other(anyhow::anyhow!("worker stdin was not piped"))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AppError::Other(anyhow::anyhow!("worker stdout was not piped"))
        })?;

        let mut wp = Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            worker_idx,
        };

        let init = WorkerInit {
            model_path: model_path.to_path_buf(),
            thumbnails_dir: thumbnails_dir.to_path_buf(),
            ocr_det_path: ocr_det_path.to_path_buf(),
            ocr_rec_path: ocr_rec_path.to_path_buf(),
            worker_idx,
            tiles_enabled,
        };
        write_frame(&mut wp.stdin, &init)?;
        Ok(wp)
    }

    /// Send one indexing request and wait for the matching response.  Errors
    /// here indicate the subprocess died (broken pipe / EOF / watchdog
    /// exit) and the parent should respawn.
    pub fn process(&mut self, req: &ProcessRequest) -> Result<ProcessResponse> {
        write_frame(
            &mut self.stdin,
            &WorkerRequest::Index(ProcessRequest {
                file_path: req.file_path.clone(),
                file_id: req.file_id,
            }),
        )?;
        let resp: Option<WorkerResponse> = read_frame(&mut self.stdout)?;
        match resp {
            Some(WorkerResponse::Index(r)) => Ok(r),
            Some(other) => Err(AppError::Other(anyhow::anyhow!(
                "worker returned mismatched response type: {:?}",
                other
            ))),
            None => Err(AppError::Other(anyhow::anyhow!(
                "worker subprocess closed pipe without responding"
            ))),
        }
    }

    /// Send one OCR request and wait for the text.  Same failure semantics
    /// as [`process`](Self::process).
    pub fn process_ocr(&mut self, file_path: &Path, page_num: i64) -> Result<String> {
        write_frame(
            &mut self.stdin,
            &WorkerRequest::OcrPage {
                file_path: file_path.to_path_buf(),
                page_num,
            },
        )?;
        let resp: Option<WorkerResponse> = read_frame(&mut self.stdout)?;
        match resp {
            Some(WorkerResponse::OcrPage { text }) => Ok(text),
            Some(other) => Err(AppError::Other(anyhow::anyhow!(
                "worker returned mismatched response type: {:?}",
                other
            ))),
            None => Err(AppError::Other(anyhow::anyhow!(
                "worker subprocess closed pipe without responding"
            ))),
        }
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        // Best-effort shutdown.  The child also exits naturally when stdin
        // closes (which happens automatically when this struct is dropped),
        // but we belt-and-brace with kill() so a hung child can't outlive
        // the parent.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
