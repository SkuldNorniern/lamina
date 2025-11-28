//! x86_64 ABI utilities for different platforms.

use crate::target::TargetOperatingSystem;

/// Platform-specific ABI utilities for x86_64 code generation.
pub struct X86ABI {
    target_os: TargetOperatingSystem,
}

impl X86ABI {
    /// Creates a new ABI instance for the specified target OS.
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Returns the mangled function name with platform-specific prefix.
    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOperatingSystem::MacOS => {
                if name == "main" {
                    "_main".to_string()
                } else {
                    format!("_{}", name)
                }
            }
            TargetOperatingSystem::Windows => {
                if name == "main" {
                    "main".to_string()
                } else {
                    name.to_string()
                }
            }
            _ => name.to_string(),
        }
    }

    /// Returns the global declaration directive for the main function.
    pub fn get_main_global(&self) -> &'static str {
        match self.target_os {
            TargetOperatingSystem::Windows => ".globl main",
            _ => ".globl main",
        }
    }

    /// Returns the argument registers for the target OS ABI.
    ///
    /// - Windows x64: rcx, rdx, r8, r9 (first 4 arguments)
    /// - System V AMD64: rdi, rsi, rdx, rcx, r8, r9 (first 6 arguments)
    pub fn arg_registers(&self) -> &'static [&'static str] {
        match self.target_os {
            TargetOperatingSystem::Windows => &["rcx", "rdx", "r8", "r9"],
            _ => &["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        }
    }

    /// System V AMD64 ABI argument registers.
    ///
    /// Deprecated: use `arg_registers()` instead for platform-aware register selection.
    pub const ARG_REGISTERS: &'static [&'static str] = &["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

    /// Caller-saved registers that must be preserved by the caller if live across function calls.
    pub const CALLER_SAVED_REGISTERS: &'static [&'static str] =
        &["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"];

    /// Callee-saved registers that are preserved by called functions.
    pub const CALLEE_SAVED_REGISTERS: &'static [&'static str] =
        &["rbx", "rbp", "r12", "r13", "r14", "r15"];
}
