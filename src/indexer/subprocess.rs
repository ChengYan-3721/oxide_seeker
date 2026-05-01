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
    /// Used purely for logging on the child side so log lines from different
    /// workers can be told apart.
    pub worker_idx: usize,
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
    pub phash: String,
    pub vector: Vec<f32>,
    /// Path of the saved thumbnail relative to the thumbnails dir.  Child
    /// writes the file directly to disk so the parent does not have to ship
    /// thumbnail bytes back through the pipe.
    pub thumb_relative_path: String,
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
    pub fn spawn(model_path: &Path, thumbnails_dir: &Path, worker_idx: usize) -> Result<Self> {
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
            worker_idx,
        };
        write_frame(&mut wp.stdin, &init)?;
        Ok(wp)
    }

    /// Send one request and wait for the matching response.  Errors here
    /// indicate the subprocess died (broken pipe / EOF) and the parent
    /// should respawn.
    pub fn process(&mut self, req: &ProcessRequest) -> Result<ProcessResponse> {
        write_frame(&mut self.stdin, req)?;
        let resp: Option<ProcessResponse> = read_frame(&mut self.stdout)?;
        resp.ok_or_else(|| {
            AppError::Other(anyhow::anyhow!(
                "worker subprocess closed pipe without responding"
            ))
        })
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
