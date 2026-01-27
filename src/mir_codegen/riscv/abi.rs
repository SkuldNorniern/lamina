//! RISC-V ABI utilities for symbol naming and calling conventions.
//!
//! # RISC-V ABI Documentation
//!
//! ## RISC-V Calling Convention (RV64)
//!
//! **Argument Registers** (first 8 arguments):
//! 1. `a0` - 1st argument
//! 2. `a1` - 2nd argument
//! 3. `a2` - 3rd argument
//! 4. `a3` - 4th argument
//! 5. `a4` - 5th argument
//! 6. `a5` - 6th argument
//! 7. `a6` - 7th argument
//! 8. `a7` - 8th argument
//!
//! **Stack Arguments**: 9th argument and beyond are passed on the stack, 16-byte aligned.
//!
//! **Return Register**: `a0` for integer returns, `fa0` for floating-point returns.
//!
//! **Caller-Saved Registers**: `a0-a7`, `t0-t6`
//!
//! **Callee-Saved Registers**: `s0-s11`, `ra` (return address)
//!
//! **Stack Alignment**: 16-byte aligned at function entry.

use crate::mir_codegen::abi::{Abi, common_call_stub, mangle_macos_name};
use lamina_platform::TargetOperatingSystem;

/// RISC-V ABI utilities.
///
/// Implements the RISC-V calling convention for RV64.
pub struct RiscVAbi {
    target_os: TargetOperatingSystem,
}

impl RiscVAbi {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Get the appropriate function name with platform-specific prefix
    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOperatingSystem::MacOS => mangle_macos_name(name),
            _ => name.to_string(),
        }
    }

    /// Get the appropriate global declaration for main
    pub fn get_main_global(&self) -> &'static str {
        ".globl main"
    }

    /// Get the data section directive
    pub fn get_data_section(&self) -> &'static str {
        ".data"
    }

    /// Get the text section directive
    pub fn get_text_section(&self) -> &'static str {
        ".text"
    }

    /// Get the format string for printing integers
    pub fn get_print_format(&self) -> &'static str {
        match self.target_os {
            TargetOperatingSystem::MacOS => "__mir_fmt_int: .asciz \"%lld\\n\"",
            _ => ".L_mir_fmt_int: .string \"%lld\\n\"",
        }
    }

    /// RISC-V calling convention argument registers (first 8 arguments)
    pub const ARG_REGISTERS: &'static [&'static str] =
        &["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"];

    /// Map well-known intrinsic/runtime names to platform symbol stubs
    pub fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }
}

impl Abi for RiscVAbi {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        RiscVAbi::mangle_function_name(self, name)
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        RiscVAbi::call_stub(self, name)
    }
}
