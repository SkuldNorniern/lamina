//! AArch64 ABI utilities for symbol naming and calling conventions.
//!
//! # AArch64 ABI Documentation
//!
//! ## AAPCS64 (Linux, macOS)
//!
//! **Argument Registers** (first 8 arguments):
//! 1. `x0` - 1st argument
//! 2. `x1` - 2nd argument
//! 3. `x2` - 3rd argument
//! 4. `x3` - 4th argument
//! 5. `x4` - 5th argument
//! 6. `x5` - 6th argument
//! 7. `x6` - 7th argument
//! 8. `x7` - 8th argument
//!
//! **Stack Arguments**: 9th argument and beyond are passed on the stack, 16-byte aligned.
//!
//! **Return Register**: `x0` for integer returns, `v0` (or `d0`) for floating-point returns.
//!
//! ## Example: Function Call Codegen
//!
//! ```text
//! # Calling function foo(x, y, z) where x, y, z are in x10, x11, x12
//! mov x0, x10      # First argument in x0
//! mov x1, x11      # Second argument in x1
//! mov x2, x12      # Third argument in x2
//! bl foo           # Call function (or _foo on macOS)
//! mov x13, x0      # Save return value from x0
//!
//! # Calling function with 10 arguments (8 in registers, 2 on stack)
//! mov x0, x10      # arg1
//! mov x1, x11      # arg2
//! # ... args 3-8 in x2-x7
//! sub sp, sp, #16  # Allocate stack space (16-byte aligned)
//! str x18, [sp]    # arg9 on stack
//! str x19, [sp, #8] # arg10 on stack
//! bl bar
//! add sp, sp, #16  # Clean up stack
//! ```
//!
//! **Caller-Saved Registers**: `x0-x18` (x16, x17 are IP0, IP1)
//!
//! **Callee-Saved Registers**: `x19-x30` (x29 is FP, x30 is LR)
//!
//! **Stack Alignment**: 16-byte aligned at function entry.

use crate::mir_codegen::abi::{Abi, common_call_stub, mangle_macos_name};
use lamina_platform::TargetOperatingSystem;

/// Platform-specific ABI utilities for AArch64 code generation.
///
/// Implements the AAPCS64 (ARM Architecture Procedure Call Standard 64-bit) calling convention.
pub struct AArch64ABI {
    target_os: TargetOperatingSystem,
}

impl AArch64ABI {
    /// Creates a new ABI instance for the specified target OS.
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Returns the mangled function name with platform-specific prefix.
    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOperatingSystem::MacOS => mangle_macos_name(name),
            _ => name.to_string(),
        }
    }

    /// Returns the global declaration directive for a function.
    pub fn get_global_directive(&self, func_name: &str) -> Option<String> {
        match self.target_os {
            TargetOperatingSystem::MacOS => Some(format!(".globl _{}", func_name)),
            _ => Some(format!(".globl {}", func_name)),
        }
    }

    /// Maps well-known intrinsic/runtime names to platform symbol stubs.
    pub fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }

    /// AArch64 AAPCS64 calling convention argument registers (first 8 arguments).
    pub const ARG_REGISTERS: &'static [&'static str] =
        &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];

    /// Caller-saved registers that must be preserved by the caller if live across function calls.
    pub const CALLER_SAVED_REGISTERS: &'static [&'static str] = &[
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
        "x14", "x15", "x16", "x17",
    ];

    /// Callee-saved registers that are preserved by called functions.
    pub const CALLEE_SAVED_REGISTERS: &'static [&'static str] = &[
        "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
    ];

    /// Returns the target operating system for this ABI instance.
    pub fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }
}

impl Abi for AArch64ABI {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        AArch64ABI::mangle_function_name(self, name)
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        AArch64ABI::call_stub(self, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aarch64_abi_new() {
        let abi = AArch64ABI::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.target_os(), TargetOperatingSystem::Linux);

        let abi_macos = AArch64ABI::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.target_os(), TargetOperatingSystem::MacOS);
    }

    #[test]
    fn test_aarch64_mangle_function_name() {
        let abi_linux = AArch64ABI::new(TargetOperatingSystem::Linux);
        assert_eq!(abi_linux.mangle_function_name("main"), "main");
        assert_eq!(abi_linux.mangle_function_name("foo"), "foo");
        assert_eq!(abi_linux.mangle_function_name("my_func"), "my_func");

        let abi_macos = AArch64ABI::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.mangle_function_name("main"), "_main");
        assert_eq!(abi_macos.mangle_function_name("foo"), "_foo");
        assert_eq!(abi_macos.mangle_function_name("my_func"), "_my_func");
    }

    #[test]
    fn test_aarch64_call_stub() {
        let abi_linux = AArch64ABI::new(TargetOperatingSystem::Linux);
        assert_eq!(abi_linux.call_stub("print"), Some("printf".to_string()));
        assert_eq!(abi_linux.call_stub("malloc"), Some("malloc".to_string()));
        assert_eq!(abi_linux.call_stub("dealloc"), Some("free".to_string()));
        assert_eq!(abi_linux.call_stub("unknown"), None);

        let abi_macos = AArch64ABI::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.call_stub("print"), Some("_printf".to_string()));
        assert_eq!(abi_macos.call_stub("malloc"), Some("_malloc".to_string()));
    }

    #[test]
    fn test_aarch64_arg_registers() {
        assert_eq!(AArch64ABI::ARG_REGISTERS.len(), 8);
        assert_eq!(AArch64ABI::ARG_REGISTERS[0], "x0");
        assert_eq!(AArch64ABI::ARG_REGISTERS[7], "x7");
    }

    #[test]
    fn test_aarch64_caller_saved_registers() {
        assert_eq!(AArch64ABI::CALLER_SAVED_REGISTERS.len(), 18);
        assert!(AArch64ABI::CALLER_SAVED_REGISTERS.contains(&"x0"));
        assert!(AArch64ABI::CALLER_SAVED_REGISTERS.contains(&"x15"));
    }

    #[test]
    fn test_aarch64_callee_saved_registers() {
        assert_eq!(AArch64ABI::CALLEE_SAVED_REGISTERS.len(), 12);
        assert!(AArch64ABI::CALLEE_SAVED_REGISTERS.contains(&"x19"));
        assert!(AArch64ABI::CALLEE_SAVED_REGISTERS.contains(&"x30"));
    }

    #[test]
    fn test_aarch64_get_global_directive() {
        let abi_linux = AArch64ABI::new(TargetOperatingSystem::Linux);
        assert_eq!(
            abi_linux.get_global_directive("main"),
            Some(".globl main".to_string())
        );
        assert_eq!(
            abi_linux.get_global_directive("foo"),
            Some(".globl foo".to_string())
        );

        let abi_macos = AArch64ABI::new(TargetOperatingSystem::MacOS);
        assert_eq!(
            abi_macos.get_global_directive("main"),
            Some(".globl _main".to_string())
        );
        assert_eq!(
            abi_macos.get_global_directive("foo"),
            Some(".globl _foo".to_string())
        );
    }
}
