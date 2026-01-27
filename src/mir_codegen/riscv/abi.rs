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
//! ## Example: Function Call Codegen
//!
//! ```text
//! # Calling function foo(x, y, z) where x, y, z are in t0, t1, t2
//! mv a0, t0        # First argument in a0
//! mv a1, t1        # Second argument in a1
//! mv a2, t2        # Third argument in a2
//! call foo         # Call function (or _foo on macOS)
//! mv t3, a0        # Save return value from a0
//!
//! # Calling function with 10 arguments (8 in registers, 2 on stack)
//! mv a0, t0        # arg1
//! mv a1, t1        # arg2
//! # ... args 3-8 in a2-a7
//! addi sp, sp, -16 # Allocate stack space (16-byte aligned)
//! sd t8, 0(sp)     # arg9 on stack
//! sd t9, 8(sp)     # arg10 on stack
//! call bar
//! addi sp, sp, 16  # Clean up stack
//! ```
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riscv_abi_new() {
        let abi = RiscVAbi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.target_os(), TargetOperatingSystem::Linux);

        let abi_macos = RiscVAbi::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.target_os(), TargetOperatingSystem::MacOS);
    }

    #[test]
    fn test_riscv_mangle_function_name() {
        let abi_linux = RiscVAbi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi_linux.mangle_function_name("main"), "main");
        assert_eq!(abi_linux.mangle_function_name("foo"), "foo");
        assert_eq!(abi_linux.mangle_function_name("my_func"), "my_func");

        let abi_macos = RiscVAbi::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.mangle_function_name("main"), "_main");
        assert_eq!(abi_macos.mangle_function_name("foo"), "_foo");
        assert_eq!(abi_macos.mangle_function_name("my_func"), "_my_func");
    }

    #[test]
    fn test_riscv_call_stub() {
        let abi_linux = RiscVAbi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi_linux.call_stub("print"), Some("printf".to_string()));
        assert_eq!(abi_linux.call_stub("malloc"), Some("malloc".to_string()));
        assert_eq!(abi_linux.call_stub("dealloc"), Some("free".to_string()));
        assert_eq!(abi_linux.call_stub("unknown"), None);

        let abi_macos = RiscVAbi::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.call_stub("print"), Some("_printf".to_string()));
        assert_eq!(abi_macos.call_stub("malloc"), Some("_malloc".to_string()));
        assert_eq!(abi_macos.call_stub("dealloc"), Some("_free".to_string()));
    }

    #[test]
    fn test_riscv_arg_registers() {
        assert_eq!(RiscVAbi::ARG_REGISTERS.len(), 8);
        assert_eq!(RiscVAbi::ARG_REGISTERS[0], "a0");
        assert_eq!(RiscVAbi::ARG_REGISTERS[7], "a7");
    }

    #[test]
    fn test_riscv_get_main_global() {
        let abi = RiscVAbi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.get_main_global(), ".globl main");
    }

    #[test]
    fn test_riscv_get_print_format() {
        let abi_linux = RiscVAbi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi_linux.get_print_format(), ".L_mir_fmt_int: .string \"%lld\\n\"");

        let abi_macos = RiscVAbi::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.get_print_format(), "__mir_fmt_int: .asciz \"%lld\\n\"");
    }
}
