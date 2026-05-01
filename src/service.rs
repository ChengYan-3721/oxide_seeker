#[macro_use]
extern crate windows_service;
use single_instance::SingleInstance;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::{
    error::Error,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread::{self},
    time::Duration,
};
use windows_service::{
    service::{
        ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
        ServiceType,
    },
    service_control_handler::{self, ServiceControlHandlerResult},
    service_dispatcher,
};

#[path = "job_object.rs"]
mod job_object;

define_windows_service!(ffi_service_main, my_service_main);
const SERVICE_NAME: &str = "OxideSeeker";
/// Mirrors `oxide_seeker::SERVICE_MODE_FLAG`.  Kept as a literal here so the
/// service binary doesn't need to depend on the main crate as a library.
const SERVICE_MODE_FLAG: &str = "--service-mode";
/// Time we give the main process to drain in-flight DB writes after we close
/// its stdin (graceful-shutdown signal) before we fall back to TerminateProcess.
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    // 检查程序是否已经在运行 (使用唯一的字符串标识)
    let instance = SingleInstance::new("oxide_seeker_service_lock").unwrap();
    if !instance.is_single() {
        return;
    }

    // 1. 设置工作目录为 exe 所在目录 (解决相对路径问题，如找不到 config.toml)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let _ = std::env::set_current_dir(dir);
        }
    }

    // 2. 注册 Panic Hook (捕获崩溃信息到文件)
    std::panic::set_hook(Box::new(|info| {
        let msg = match info.payload().downcast_ref::<&str>() {
            Some(s) => *s,
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => &**s,
                None => "Box<Any>",
            },
        };
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());
        let err_msg = format!("Panic occurred at {}: {}\n", location, msg);

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("./logs/service_panic.log")
        {
            let _ = writeln!(file, "{}", err_msg);
        }
    }));

    // 尝试作为 Windows 服务运行
    if service_dispatcher::start(SERVICE_NAME, ffi_service_main).is_err() {
        let _ = run_service();
    }
}

fn my_service_main(_arguments: Vec<OsString>) {
    if let Ok(status_handle) =
        service_control_handler::register(SERVICE_NAME, move |control_event| {
            match control_event {
                ServiceControl::Stop => {
                    // 收到停止信号，设置 RUNNING 为 false
                    RUNNING.store(false, Ordering::Relaxed);
                    ServiceControlHandlerResult::NoError
                }
                ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
                _ => ServiceControlHandlerResult::NotImplemented,
            }
        })
    {
        // 报告服务状态为 Running
        let _ = status_handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        });

        // 启动实际的业务逻辑 (阻塞运行)
        let _ = run_service();

        // 报告服务状态为 Stopped
        let _ = status_handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Stopped,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        });
    }
}

pub fn run_service() -> Result<(), Box<dyn Error>> {
    let mut child: Option<Child> = None;
    let target_exe = get_target_exe_path()?;

    // Outer Job Object: wraps oxide_seeker.exe.  If this service process is
    // killed (Stop, crash, end-task), the kernel closes the Job handle and
    // every member — main process plus its workers, since nested Jobs
    // propagate — is terminated.  Belt-and-braces with the inner Job the
    // main process maintains over its own workers.
    let job = job_object::JobObject::new_kill_on_close()
        .map_err(|e| format!("创建 Job Object 失败: {}", e))?;

    while RUNNING.load(Ordering::Relaxed) {
        let need_start = match child.as_mut() {
            Some(process) => process.try_wait()?.is_some(),
            None => true,
        };

        if need_start {
            child = Some(start_oxide_seeker(&target_exe, &job)?);
        }

        thread::sleep(Duration::from_millis(100));
    }

    if let Some(mut process) = child {
        stop_oxide_seeker(&mut process)?;
    }

    Ok(())
}

fn get_target_exe_path() -> Result<PathBuf, Box<dyn Error>> {
    let service_exe = std::env::current_exe()?;
    let dir = service_exe
        .parent()
        .ok_or_else(|| "无法获取服务程序所在目录".to_string())?;
    Ok(dir.join("oxide_seeker.exe"))
}

fn start_oxide_seeker(
    exe_path: &PathBuf,
    job: &job_object::JobObject,
) -> Result<Child, Box<dyn Error>> {
    if !exe_path.exists() {
        return Err(format!("目标程序不存在: {}", exe_path.display()).into());
    }

    // stdin is piped so we can close it as a graceful-shutdown signal in
    // stop_oxide_seeker.  stdout / stderr stay null — the main process
    // writes its own logs.
    let child = Command::new(exe_path)
        .arg(SERVICE_MODE_FLAG)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    // Assign immediately so even if the main process crashes during early
    // init (before it builds its own worker Job) the service Job still
    // owns it.  Failure here means we'd lose cascading shutdown, so we
    // tear the half-spawned child down rather than continue blindly.
    if let Err(e) = job.assign_child(&child) {
        let mut child = child;
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("将 oxide_seeker.exe 加入 Job Object 失败: {}", e).into());
    }

    Ok(child)
}

fn stop_oxide_seeker(child: &mut Child) -> Result<(), Box<dyn Error>> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    // 1. Graceful: drop stdin so the main process's stdin watchdog observes
    //    EOF and exits cleanly (its own Job kills the workers on the way
    //    out).
    drop(child.stdin.take());

    // 2. Wait briefly for the child to honour the signal.
    let deadline = Instant::now() + GRACEFUL_SHUTDOWN_TIMEOUT;
    while Instant::now() < deadline {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }

    // 3. Hard kill as a fallback.  The outer Job Object will also tear
    //    down any workers that managed to escape the inner Job, so this
    //    is just to make the wait below return promptly.
    let _ = child.kill();
    let _ = child.wait();
    Ok(())
}
