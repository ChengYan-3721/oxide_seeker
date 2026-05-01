//! Windows Job Object — cascading process termination.
//!
//! A Job Object created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` kills every
//! process inside it the moment the last handle to the Job is closed.  The
//! handle is closed automatically when the holding process exits — whether
//! gracefully, via a Rust panic, or via `TerminateProcess` from outside.
//!
//! Two Jobs run nested in this codebase:
//!
//! 1. The Windows service (`oxide_seeker_service.exe`) wraps the main
//!    `oxide_seeker.exe` process — guarantees the whole tree dies when the
//!    service is stopped, even before the main process has a chance to spin
//!    up its own Job.
//! 2. The main process wraps every `--worker-mode` indexing subprocess —
//!    guarantees workers die when the main process dies, regardless of how
//!    that death happened (SEH, panic, kill from the service).
//!
//! Nested Jobs are supported on Windows 7 / Server 2008 R2 and later.

#![cfg(windows)]

use std::io;
use std::os::windows::io::AsRawHandle;
use std::process::Child;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Owns a Job Object handle.  Drop closes the handle, which (with
/// `KILL_ON_JOB_CLOSE`) terminates every member process synchronously.
pub struct JobObject {
    handle: HANDLE,
}

impl JobObject {
    /// Create an unnamed Job configured so closing the last handle kills
    /// every member process.
    pub fn new_kill_on_close() -> io::Result<Self> {
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let ok = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            let err = io::Error::last_os_error();
            unsafe {
                CloseHandle(handle);
            }
            return Err(err);
        }

        Ok(Self { handle })
    }

    /// Add an already-spawned child process to the Job.  Idempotent failures
    /// (already in another Job that disallows breakaway) propagate as
    /// `io::Error`.
    pub fn assign_child(&self, child: &Child) -> io::Result<()> {
        let proc_handle = child.as_raw_handle() as HANDLE;
        let ok = unsafe { AssignProcessToJobObject(self.handle, proc_handle) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for JobObject {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

unsafe impl Send for JobObject {}
unsafe impl Sync for JobObject {}
