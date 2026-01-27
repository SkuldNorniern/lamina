//! x86_64 ABI utilities for different platforms.
//!
//! # x86_64 ABI Documentation
//!
//! ## System V AMD64 (Linux, macOS)
//!
//! **Argument Registers** (first 6 arguments):
//! 1. `%rdi` - 1st argument
//! 2. `%rsi` - 2nd argument
//! 3. `%rdx` - 3rd argument
//! 4. `%rcx` - 4th argument
//! 5. `%r8` - 5th argument
//! 6. `%r9` - 6th argument
//!
//! **Stack Arguments**: 7th argument and beyond are passed on the stack, pushed right-to-left.
//!
//! **Return Register**: `%rax` for integer returns, `%xmm0` for floating-point returns.
//!
//! **Caller-Saved Registers**: `%rax`, `%rcx`, `%rdx`, `%rsi`, `%rdi`, `%r8`, `%r9`, `%r10`, `%r11`
//!
//! **Callee-Saved Registers**: `%rbx`, `%rbp`, `%r12`, `%r13`, `%r14`, `%r15`
//!
//! **Stack Alignment**: 16-byte aligned at function entry.
//!
//! ## Microsoft x64 (Windows)
//!
//! **Argument Registers** (first 4 arguments):
//! 1. `%rcx` - 1st argument
//! 2. `%rdx` - 2nd argument
//! 3. `%r8` - 3rd argument
//! 4. `%r9` - 4th argument
//!
//! **Shadow Space**: 32 bytes of shadow space allocated before stack arguments.
//!
//! **Stack Arguments**: 5th argument and beyond are passed on the stack in shadow space.
//!
//! **Return Register**: `%rax` for integer returns, `%xmm0` for floating-point returns.

use crate::mir_codegen::abi::{Abi, common_call_stub, mangle_macos_name};
use lamina_platform::TargetOperatingSystem;

/// Platform-specific ABI utilities for x86_64 code generation.
///
/// Supports both System V AMD64 (Linux, macOS) and Microsoft x64 (Windows) calling conventions.
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
            TargetOperatingSystem::MacOS => mangle_macos_name(name),
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

impl Abi for X86ABI {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        X86ABI::mangle_function_name(self, name)
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }
}
