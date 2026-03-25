//! PowerPC64 ELFv2 ABI (Linux ppc64le).
//!
//! ## Calling Convention Summary
//!
//! - Integer arguments: r3–r10 (8 registers). Overflow args go on the stack.
//! - Integer return value: r3.
//! - Float arguments: f1–f13. Float return value: f1.
//! - Caller-saved: r0, r3–r12, f0–f13, lr.
//! - Callee-saved: r14–r31, f14–f31, cr2–cr4.
//! - Stack pointer: r1 (must be 16-byte aligned at call sites).
//! - TOC pointer: r2 (do not clobber).

use crate::mir_codegen::abi::{Abi, common_call_stub, mangle_macos_name};
use lamina_platform::TargetOperatingSystem;

pub struct Ppc64Abi {
    target_os: TargetOperatingSystem,
}

impl Ppc64Abi {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    pub fn mangle_function_name(&self, name: &str) -> String {
        match self.target_os {
            TargetOperatingSystem::MacOS => mangle_macos_name(name),
            _ => name.to_string(),
        }
    }

    pub fn get_main_global(&self) -> &'static str {
        ".globl main"
    }

    pub fn get_data_section(&self) -> &'static str {
        ".data"
    }

    pub fn get_text_section(&self) -> &'static str {
        ".text"
    }

    pub fn get_print_format(&self) -> &'static str {
        ".L_mir_fmt_int: .string \"%lld\\n\""
    }

    pub const ARG_REGISTERS: &'static [&'static str] =
        &["3", "4", "5", "6", "7", "8", "9", "10"];

    pub fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }
}

impl Abi for Ppc64Abi {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        Ppc64Abi::mangle_function_name(self, name)
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        Ppc64Abi::call_stub(self, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppc64_mangle_linux() {
        let abi = Ppc64Abi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.mangle_function_name("main"), "main");
        assert_eq!(abi.mangle_function_name("foo"), "foo");
    }

    #[test]
    fn test_ppc64_call_stub_linux() {
        let abi = Ppc64Abi::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.call_stub("print"), Some("printf".to_string()));
        assert_eq!(abi.call_stub("malloc"), Some("malloc".to_string()));
        assert_eq!(abi.call_stub("unknown"), None);
    }

    #[test]
    fn test_ppc64_arg_registers() {
        assert_eq!(Ppc64Abi::ARG_REGISTERS.len(), 8);
        assert_eq!(Ppc64Abi::ARG_REGISTERS[0], "3");
    }
}
