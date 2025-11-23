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
            _ => name.to_string(),
        }
    }

    /// Get the appropriate global declaration for main
    pub fn get_main_global(&self) -> &'static str {
        ".globl main"
    }

    /// System V AMD64 ABI Argument Registers
    pub const ARG_REGISTERS: &'static [&'static str] = &["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

    /// Caller-saved registers that must be preserved across calls if live
    pub const CALLER_SAVED_REGISTERS: &'static [&'static str] = &[
        "rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
    ];

    /// Callee-saved registers (preserved by the callee)
    pub const CALLEE_SAVED_REGISTERS: &'static [&'static str] = &[
        "rbx", "rbp", "r12", "r13", "r14", "r15",
    ];
}
