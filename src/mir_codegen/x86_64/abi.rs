use crate::target::TargetOperatingSystem;

/// x86_64 ABI utilities for different platforms
pub struct X86ABI {
    target_os: TargetOperatingSystem,
}

impl X86ABI {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Get the appropriate function name with platform-specific prefix
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
                // Windows x64: no underscore prefix, but main is special
                if name == "main" {
                    "main".to_string()
                } else {
                    name.to_string()
                }
            }
            _ => name.to_string(),
        }
    }

    /// Get the appropriate global declaration for main
    pub fn get_main_global(&self) -> &'static str {
        match self.target_os {
            TargetOperatingSystem::Windows => ".globl main",
            _ => ".globl main",
        }
    }

    /// Get argument registers based on the target OS ABI
    pub fn arg_registers(&self) -> &'static [&'static str] {
        match self.target_os {
            TargetOperatingSystem::Windows => {
                // Microsoft x64 ABI: rcx, rdx, r8, r9 (first 4 args)
                &["rcx", "rdx", "r8", "r9"]
            }
            _ => {
                // System V AMD64 ABI: rdi, rsi, rdx, rcx, r8, r9 (first 6 args)
                &["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            }
        }
    }

    /// System V AMD64 ABI Argument Registers (deprecated, use arg_registers() instead)
    pub const ARG_REGISTERS: &'static [&'static str] = &["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

    /// Caller-saved registers that must be preserved across calls if live
    pub const CALLER_SAVED_REGISTERS: &'static [&'static str] =
        &["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"];

    /// Callee-saved registers (preserved by the callee)
    pub const CALLEE_SAVED_REGISTERS: &'static [&'static str] =
        &["rbx", "rbp", "r12", "r13", "r14", "r15"];
}
