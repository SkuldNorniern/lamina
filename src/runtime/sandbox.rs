//! Sandbox for executing JIT-compiled code with resource constraints.
//!
//! The sandbox provides:
//! - A timeout enforced by running the JIT function on a worker thread and
//!   blocking the calling thread with `recv_timeout`.
//! - A MIR-level static analysis pass that rejects modules containing
//!   calls to syscall intrinsics or known I/O symbols when the corresponding
//!   capability is disabled in `SandboxConfig`.
//! - A basic memory-budget check based on estimated code + stack size.

use crate::error::LaminaError;
use crate::mir::{Instruction, Module as MirModule};
use crate::runtime::compiler::RuntimeCompiler;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::time::Duration;

/// Configuration for the sandbox execution environment.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum memory the JIT-compiled code may use (bytes). `None` disables the limit.
    pub max_memory: Option<usize>,
    /// Wall-clock timeout for a single execution (milliseconds). `None` disables the limit.
    pub max_execution_time: Option<u64>,
    /// Allow calls to raw system-call intrinsics (`syscall`, `sysenter`).
    pub allow_syscalls: bool,
    /// Allow calls to file-I/O symbols (`open`, `read`, `write`, `close`, …).
    pub allow_file_io: bool,
    /// Allow calls to network symbols (`socket`, `connect`, `send`, `recv`, …).
    pub allow_network: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory: Some(64 * 1024 * 1024), // 64 MB
            max_execution_time: Some(5_000),    // 5 s
            allow_syscalls: false,
            allow_file_io: false,
            allow_network: false,
        }
    }
}

impl SandboxConfig {
    /// Maximally restrictive: no syscalls, no I/O, 1-second timeout, 16 MB cap.
    pub fn restrictive() -> Self {
        Self {
            max_memory: Some(16 * 1024 * 1024),
            max_execution_time: Some(1_000),
            allow_syscalls: false,
            allow_file_io: false,
            allow_network: false,
        }
    }

    /// Permissive: all capabilities enabled, no limits.
    pub fn permissive() -> Self {
        Self {
            max_memory: None,
            max_execution_time: None,
            allow_syscalls: true,
            allow_file_io: true,
            allow_network: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Known disallowed symbol sets
// ---------------------------------------------------------------------------

/// Symbols that bypass the OS permission model via raw syscall instructions.
const SYSCALL_SYMBOLS: &[&str] = &["syscall", "sysenter", "int80", "__syscall"];

/// Common POSIX file-I/O symbols.
const FILE_IO_SYMBOLS: &[&str] = &[
    "open", "openat", "creat", "read", "write", "close", "fopen", "fclose", "fread", "fwrite",
    "fseek", "ftell", "fflush", "rename", "unlink", "mkdir", "rmdir", "stat", "fstat", "lstat",
    "access", "chmod", "chown", "truncate", "readdir", "opendir", "closedir",
];

/// Common network symbols.
const NETWORK_SYMBOLS: &[&str] = &[
    "socket",
    "bind",
    "listen",
    "accept",
    "connect",
    "send",
    "recv",
    "sendto",
    "recvfrom",
    "gethostbyname",
    "getaddrinfo",
    "freeaddrinfo",
    "inet_addr",
    "inet_pton",
    "setsockopt",
    "getsockopt",
    "shutdown",
    "poll",
    "select",
    "epoll_create",
    "epoll_ctl",
    "epoll_wait",
];

// ---------------------------------------------------------------------------
// MIR static analysis
// ---------------------------------------------------------------------------

/// Analyse every `Call`/`TailCall` in `module` and return a list of
/// violation messages based on `config`.
fn analyse_module(module: &MirModule, config: &SandboxConfig) -> Vec<String> {
    let mut violations = Vec::new();

    for (func_name, func) in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                let callee = match inst {
                    Instruction::Call { name, .. } => Some(name.as_str()),
                    Instruction::TailCall { name, .. } => Some(name.as_str()),
                    _ => None,
                };
                let Some(name) = callee else { continue };

                if !config.allow_syscalls && SYSCALL_SYMBOLS.contains(&name) {
                    violations.push(format!(
                        "function '{func_name}': calls disallowed syscall intrinsic '{name}'"
                    ));
                }
                if !config.allow_file_io && FILE_IO_SYMBOLS.contains(&name) {
                    violations.push(format!(
                        "function '{func_name}': calls disallowed file-I/O symbol '{name}'"
                    ));
                }
                if !config.allow_network && NETWORK_SYMBOLS.contains(&name) {
                    violations.push(format!(
                        "function '{func_name}': calls disallowed network symbol '{name}'"
                    ));
                }
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// Sandbox for safely executing JIT-compiled code.
pub struct Sandbox {
    compiler: RuntimeCompiler,
    config: SandboxConfig,
}

impl Sandbox {
    /// Create a new sandbox with the given configuration.
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
        config: SandboxConfig,
    ) -> Self {
        Self {
            compiler: RuntimeCompiler::new(target_arch, target_os),
            config,
        }
    }

    /// Compile and execute `function_name` from `module` inside the sandbox.
    ///
    /// Execution is isolated on a worker thread so that a timeout can be
    /// applied without blocking the caller indefinitely.  The function must
    /// take no arguments and return a value convertible to `i64` (the type
    /// used internally by Lamina's calling convention).
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - The module contains calls that violate `SandboxConfig` restrictions.
    /// - The estimated memory budget is exceeded.
    /// - Compilation fails.
    /// - The execution timeout is exceeded.
    pub fn execute_i64(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<i64, LaminaError> {
        self.validate_module(module)?;

        #[cfg(feature = "encoder")]
        {
            // Compile to executable memory on the calling thread (compilation is
            // synchronous and does not run user code).
            let memory = self
                .compiler
                .compile(module, Some(function_name))
                .map_err(|e| {
                    LaminaError::ValidationError(format!("Sandbox compilation failed: {e}"))
                })?;

            // SAFETY: `memory` is kept alive for the entire duration of the
            // execution below; the function pointer is only used within this scope.
            let fn_ptr: unsafe extern "C" fn() -> i64 = unsafe {
                let raw = memory.code_start();
                if raw.is_null() {
                    return Err(LaminaError::ValidationError(
                        "Sandbox: compiled function pointer is null".to_string(),
                    ));
                }
                std::mem::transmute(raw)
            };

            match self.config.max_execution_time {
                None => {
                    // No timeout — call directly.
                    let result = unsafe { fn_ptr() };
                    Ok(result)
                }
                Some(timeout_ms) => {
                    // Run on a dedicated thread and wait for the result.
                    let (tx, rx) = std::sync::mpsc::channel::<Result<i64, String>>();
                    // SAFETY: We transmit the raw function pointer as a usize so it
                    // can cross the thread boundary without a Send bound.  The
                    // `memory` object is not moved into the thread — it remains on
                    // this stack frame (and therefore alive) until `recv_timeout`
                    // returns, which is before `memory` is dropped.
                    let fn_addr = fn_ptr as usize;
                    std::thread::spawn(move || {
                        let f: unsafe extern "C" fn() -> i64 =
                            unsafe { std::mem::transmute(fn_addr) };
                        let result = unsafe { f() };
                        let _ = tx.send(Ok(result));
                    });

                    rx.recv_timeout(Duration::from_millis(timeout_ms))
                        .map_err(|_| {
                            LaminaError::ValidationError(format!(
                                "Sandbox: execution timed out after {timeout_ms} ms"
                            ))
                        })?
                        .map_err(LaminaError::ValidationError)
                }
            }
        }
        #[cfg(not(feature = "encoder"))]
        {
            let _ = (module, function_name);
            Err(LaminaError::ValidationError(
                "Sandbox execution requires the 'encoder' feature".to_string(),
            ))
        }
    }

    fn validate_module(&self, module: &MirModule) -> Result<(), LaminaError> {
        // Static analysis: check for disallowed calls.
        let violations = analyse_module(module, &self.config);
        if !violations.is_empty() {
            return Err(LaminaError::ValidationError(format!(
                "Sandbox policy violation(s):\n  {}",
                violations.join("\n  ")
            )));
        }

        // Memory budget: rough estimate based on function count × average size.
        if let Some(max_mem) = self.config.max_memory {
            // Each function averages ~4 KB of code plus 8 KB of stack.
            let estimated = module.functions.len() * (4 * 1024 + 8 * 1024);
            if estimated > max_mem {
                return Err(LaminaError::ValidationError(format!(
                    "Sandbox: estimated memory {estimated} B exceeds limit {max_mem} B"
                )));
            }
        }

        Ok(())
    }

    /// Update the sandbox configuration.
    pub fn set_config(&mut self, config: SandboxConfig) {
        self.config = config;
    }

    /// Get the current sandbox configuration.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, MirType, ScalarType};

    fn empty_module() -> MirModule {
        MirModule::new("sandbox_test")
    }

    fn module_with_call(callee: &str) -> MirModule {
        let mut m = MirModule::new("sandbox_call_test");
        let f = FunctionBuilder::new("main_fn")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Call {
                name: callee.to_string(),
                args: vec![],
                ret: None,
            })
            .instr(Instruction::Ret { value: None })
            .build();
        m.add_function(f);
        m
    }

    #[test]
    fn test_analysis_empty_module_no_violations() {
        let m = empty_module();
        let violations = analyse_module(&m, &SandboxConfig::default());
        assert!(violations.is_empty());
    }

    #[test]
    fn test_analysis_blocks_syscall() {
        let m = module_with_call("syscall");
        let violations = analyse_module(&m, &SandboxConfig::default());
        assert!(!violations.is_empty());
        assert!(violations[0].contains("syscall"));
    }

    #[test]
    fn test_analysis_allows_syscall_when_permitted() {
        let m = module_with_call("syscall");
        let config = SandboxConfig {
            allow_syscalls: true,
            ..SandboxConfig::default()
        };
        let violations = analyse_module(&m, &config);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_analysis_blocks_file_io() {
        let m = module_with_call("open");
        let violations = analyse_module(&m, &SandboxConfig::default());
        assert!(!violations.is_empty());
        assert!(violations[0].contains("open"));
    }

    #[test]
    fn test_analysis_blocks_network() {
        let m = module_with_call("socket");
        let violations = analyse_module(&m, &SandboxConfig::default());
        assert!(!violations.is_empty());
        assert!(violations[0].contains("socket"));
    }

    #[test]
    fn test_validate_rejects_policy_violation() {
        let m = module_with_call("connect");
        let arch = TargetArchitecture::X86_64;
        let os = TargetOperatingSystem::Linux;
        let sandbox = Sandbox::new(arch, os, SandboxConfig::default());
        let result = sandbox.validate_module(&m);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_accepts_clean_module() {
        let m = empty_module();
        let arch = TargetArchitecture::X86_64;
        let os = TargetOperatingSystem::Linux;
        let sandbox = Sandbox::new(arch, os, SandboxConfig::default());
        assert!(sandbox.validate_module(&m).is_ok());
    }

    #[test]
    fn test_config_restrictive() {
        let c = SandboxConfig::restrictive();
        assert!(!c.allow_syscalls);
        assert!(!c.allow_file_io);
        assert!(!c.allow_network);
        assert_eq!(c.max_execution_time, Some(1_000));
    }

    #[test]
    fn test_config_permissive() {
        let c = SandboxConfig::permissive();
        assert!(c.allow_syscalls);
        assert!(c.allow_file_io);
        assert!(c.allow_network);
        assert!(c.max_memory.is_none());
    }
}
