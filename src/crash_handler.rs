//! Process-wide crash recording.
//!
//! Two paths funnel into a single `./logs/crash.log`:
//!
//! 1. **Rust panics.**  We replace the default panic hook with one that
//!    records the panic message + a forced backtrace to both `tracing` and
//!    `crash.log`, then chains to whatever hook was previously installed.
//!
//! 2. **Windows structured exceptions** (FFI segfaults inside ONNX Runtime,
//!    pdfium, native image decoders, …).  These never raise a Rust panic,
//!    so we register a top-level filter via `SetUnhandledExceptionFilter`.
//!    The filter writes a synchronous record (no allocator path more than
//!    a single `OpenOptions`) and then returns `EXCEPTION_CONTINUE_SEARCH`,
//!    so Windows still produces a WER report / minidump (if configured).
//!
//! Both the main process and any subprocess (`--worker-mode`) call
//! [`install`] at startup; multi-process append to the same log file is safe
//! because Win32 file handles serialise concurrent O_APPEND writes.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static CRASH_LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Append a record to the crash log.  Designed to be safe to call from a
/// panic / structured-exception handler: no panics, minimal allocation, no
/// locking.  Best-effort — silently drops the record if the file cannot be
/// opened (e.g. the disk is full).
pub fn append(msg: &str) {
    use std::io::Write;
    let Some(path) = CRASH_LOG_PATH.get() else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(msg.as_bytes());
        if !msg.ends_with('\n') {
            let _ = f.write_all(b"\n");
        }
        let _ = f.flush();
    }
}

/// Initialise the crash log path, install the panic hook, and (on Windows)
/// register the structured-exception filter.  Idempotent in spirit — the
/// crash-log path is only set the first time, but the panic hook chain and
/// SEH filter are re-registered on every call (so the most-recent caller's
/// `tag` shows up in records).
///
/// `tag` is included in each crash record to disambiguate the parent process
/// from its `--worker-mode` children when both append to the same file.
pub fn install(log_dir: &Path, tag: &'static str) {
    std::fs::create_dir_all(log_dir).ok();
    let _ = CRASH_LOG_PATH.set(log_dir.join("crash.log"));

    install_panic_hook(tag);
    install_seh_handler(tag);
}

fn install_panic_hook(tag: &'static str) {
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let timestamp = chrono::Local::now().to_rfc3339();
        let msg = format!("{} [{}] PANIC {info}\n{backtrace}", timestamp, tag);
        tracing::error!("{}", msg);
        append(&msg);
        default_panic(info);
    }));
}

#[cfg(windows)]
fn install_seh_handler(tag: &'static str) {
    static SEH_TAG: OnceLock<&'static str> = OnceLock::new();
    let _ = SEH_TAG.set(tag);

    unsafe extern "system" fn filter(
        info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use std::fmt::Write as _;

        let tag = SEH_TAG.get().copied().unwrap_or("unknown");
        let mut msg = String::new();
        let _ = writeln!(
            msg,
            "{} [{}] FATAL Windows structured exception",
            chrono::Local::now().to_rfc3339(),
            tag,
        );

        if !info.is_null() {
            let record = unsafe { (*info).ExceptionRecord };
            if !record.is_null() {
                let code = unsafe { (*record).ExceptionCode } as u32;
                let addr = unsafe { (*record).ExceptionAddress } as usize;
                let flags = unsafe { (*record).ExceptionFlags };
                let _ = writeln!(
                    msg,
                    "  code=0x{:08X} ({})  address=0x{:016X}  flags=0x{:X}  thread={:?}",
                    code,
                    seh_code_name(code),
                    addr,
                    flags,
                    std::thread::current().id(),
                );
            }
        }

        eprintln!("{}", msg);
        append(&msg);
        // EXCEPTION_CONTINUE_SEARCH: let Windows Error Reporting and the
        // default termination path proceed.
        0
    }

    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(Some(filter));
    }
}

#[cfg(not(windows))]
fn install_seh_handler(_tag: &'static str) {}

#[cfg(windows)]
fn seh_code_name(code: u32) -> &'static str {
    match code {
        0xC0000005 => "ACCESS_VIOLATION",
        0xC000001D => "ILLEGAL_INSTRUCTION",
        0xC0000025 => "NONCONTINUABLE_EXCEPTION",
        0xC000008C => "ARRAY_BOUNDS_EXCEEDED",
        0xC0000094 => "INT_DIVIDE_BY_ZERO",
        0xC0000095 => "INT_OVERFLOW",
        0xC00000FD => "STACK_OVERFLOW",
        0xC0000374 => "HEAP_CORRUPTION",
        0xC0000409 => "STACK_BUFFER_OVERRUN",
        0xC0000417 => "INVALID_CRT_PARAMETER",
        0xE0000008 => "RAISE_EXCEPTION_8 (likely pdfium internal abort)",
        0xE06D7363 => "C++ EXCEPTION",
        _ => "UNKNOWN",
    }
}
