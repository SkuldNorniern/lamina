//! Common ABI trait and utilities for code generation backends.
//!
//! A trait that all ABI implementations must follow,
//! reducing code duplication and giving a consistent interface.
//!
//! # Application Binary Interface (ABI) Overview
//!
//! Lamina uses platform C ABIs for function calls. The ABI implementation is
//! abstracted through the `Abi` trait, which gives a consistent interface
//! across all backends.
//!
//! ## Common ABI Contract
//!
//! ### Function Name Mangling
//!
//! All backends apply platform-specific symbol mangling:
//! - **macOS**: Functions are prefixed with underscore (e.g., `main` → `_main`)
//! - **Linux/Windows**: Functions use their original names (e.g., `main` → `main`)
//!
//! The `mangle_function_name()` method handles this automatically.
//!
//! ### Argument Passing
//!
//! - Arguments are passed using the platform C ABI register order
//! - Overflow arguments (beyond register capacity) are spilled to the stack
//! - Integer arguments use 64-bit slots, even for narrower integer types (i8, i16, i32)
//! - Floating-point arguments follow platform-specific rules
//!
//! **Example: Function call with 3 arguments (RISC-V)**
//! ```text
//! # Arguments: a0=arg1, a1=arg2, a2=arg3
//! mv a0, t0        # Load arg1 to a0
//! mv a1, t1        # Load arg2 to a1
//! mv a2, t2        # Load arg3 to a2
//! call my_function # Call function
//! ```
//!
//! **Example: Function call with 10 arguments (x86_64 System V)**
//! ```text
//! # First 6 args in registers: rdi, rsi, rdx, rcx, r8, r9
//! # Remaining 4 args on stack (16-byte aligned)
//! mov %rdi, %rax   # arg1
//! mov %rsi, %rbx   # arg2
//! # ... args 3-6 in rdx, rcx, r8, r9
//! push %r10        # arg7 on stack
//! push %r11        # arg8 on stack
//! push %r12        # arg9 on stack
//! push %r13        # arg10 on stack
//! call my_function
//! add $32, %rsp    # Clean up stack
//! ```
//!
//! ### Return Values
//!
//! - Integer return values use the C ABI integer return register
//! - Floating-point return values use the FP return register (if applicable)
//! - Void functions return nothing
//!
//! **Example: Return value handling**
//! ```text
//! # RISC-V: return value in a0 (integer) or fa0 (float)
//! call my_function
//! mv t0, a0        # Save integer return value
//!
//! # x86_64: return value in rax (integer) or xmm0 (float)
//! call my_function
//! mov %rax, %rbx   # Save integer return value
//! ```
//!
//! ### Builtin Functions
//!
//! Builtin functions can be overridden via the `LAMINA_BUILTINS` environment variable:
//!
//! ```bash
//! LAMINA_BUILTINS="print=printf,malloc=jemalloc_malloc,dealloc=jemalloc_free"
//! ```
//!
//! Default builtins:
//! - `print` → `printf` (or `_printf` on macOS)
//! - `malloc` → `malloc` (or `_malloc` on macOS)
//! - `dealloc` → `free` (or `_free` on macOS)
//!
//! ## Backend-Specific ABIs
//!
//! See the documentation in each backend's ABI module:
//! - `x86_64::abi` - System V AMD64 and Microsoft x64
//! - `arm::aarch64::abi` - AAPCS64
//! - `riscv::abi` - RISC-V calling convention
//! - `wasm::abi` - WebAssembly stack-based calling

use lamina_platform::TargetOperatingSystem;
use std::collections::HashMap;
use std::env;

pub struct BuiltinLibrary {
    target_os: TargetOperatingSystem,
    overrides: HashMap<String, String>,
}

impl BuiltinLibrary {
    pub fn from_env(target_os: TargetOperatingSystem) -> Self {
        let overrides = builtin_overrides_from_env();
        Self {
            target_os,
            overrides,
        }
    }

    pub fn resolve(&self, name: &str) -> Option<String> {
        self.overrides
            .get(name)
            .cloned()
            .or_else(|| self.default_symbol(name))
    }

    fn default_symbol(&self, name: &str) -> Option<String> {
        match name {
            "print" => Some(get_printf_symbol(self.target_os).to_string()),
            "malloc" => Some(get_malloc_symbol(self.target_os).to_string()),
            "dealloc" => Some(get_free_symbol(self.target_os).to_string()),
            _ => None,
        }
    }
}

fn builtin_overrides_from_env() -> HashMap<String, String> {
    let mut overrides = HashMap::new();
    if let Ok(value) = env::var("LAMINA_BUILTINS") {
        for entry in value.split(',') {
            let mut parts = entry.splitn(2, '=');
            let name = parts.next().unwrap_or("").trim();
            let symbol = parts.next().unwrap_or("").trim();
            if !name.is_empty() && !symbol.is_empty() {
                overrides.insert(name.to_string(), symbol.to_string());
            }
        }
    }
    overrides
}

/// Parses the `LAMINA_BUILTINS` override map.
///
/// Format: `LAMINA_BUILTINS="print=printf,malloc=jemalloc_malloc"`.
/// Entries are comma-separated key/value pairs, where each key is the Lamina
/// builtin name and each value is the platform symbol to link against.

/// Trait defining the common interface for all ABI implementations.
///
/// This contract covers name mangling, builtin symbol mapping, and the
/// minimal calling convention expectations shared by MIR backends.
///
/// ## Calling Convention Assumptions
/// - Arguments are passed using the platform C ABI register order, with
///   overflow arguments spilled to the stack.
/// - Integer arguments use 64-bit slots, even for narrower integer types.
/// - Return values use the C ABI integer return register when present.
///
/// ## Builtins
/// - Builtins are resolved by `call_stub`, which can be overridden with
///   the `LAMINA_BUILTINS` environment variable.
/// - The default builtin map recognizes `print`, `malloc`, and `dealloc`.
///
/// ## Naming
/// - Backends must apply platform-specific symbol mangling, such as
///   the underscore prefix on macOS.
pub trait Abi {
    /// Returns the target operating system for this ABI instance.
    fn target_os(&self) -> TargetOperatingSystem;

    /// Returns the mangled function name with platform-specific prefix.
    ///
    /// This method handles platform-specific symbol naming conventions,
    /// such as the underscore prefix on macOS.
    fn mangle_function_name(&self, name: &str) -> String;

    fn builtin_library(&self) -> BuiltinLibrary {
        BuiltinLibrary::from_env(self.target_os())
    }

    /// Maps well-known intrinsic/runtime names to platform symbol stubs.
    ///
    /// Returns `None` if the name is not a known intrinsic that needs
    /// special handling. This lets backends map internal names
    /// (like "print") to platform-specific symbols (like "_printf" on macOS).
    ///
    /// Default implementation returns `None` for all names, meaning
    /// no special mapping is needed.
    fn call_stub(&self, name: &str) -> Option<String> {
        self.builtin_library().resolve(name)
    }
}

/// Helper function to mangle function names for macOS (adds underscore prefix).
///
/// This is a common pattern used by multiple backends, so we provide
/// it as a shared utility function.
pub fn mangle_macos_name(name: &str) -> String {
    if name == "main" {
        "_main".to_string()
    } else {
        format!("_{}", name)
    }
}

/// Helper function to get the platform-specific printf symbol name.
///
/// Returns "_printf" on macOS, "printf" on other platforms.
pub fn get_printf_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_printf",
        _ => "printf",
    }
}

/// Helper function to get the platform-specific malloc symbol name.
///
/// Returns "_malloc" on macOS, "malloc" on other platforms.
pub fn get_malloc_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_malloc",
        _ => "malloc",
    }
}

/// Helper function to get the platform-specific free symbol name.
///
/// Returns "_free" on macOS, "free" on other platforms.
pub fn get_free_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        TargetOperatingSystem::MacOS => "_free",
        _ => "free",
    }
}

/// Helper function to create a common call_stub implementation.
///
/// This handles the common pattern of mapping "print", "malloc", and "dealloc"
/// to their platform-specific symbols. Backends can use this to reduce
/// duplication in their `call_stub` implementations.
pub fn common_call_stub(name: &str, target_os: TargetOperatingSystem) -> Option<String> {
    BuiltinLibrary::from_env(target_os).resolve(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mangle_macos_name() {
        assert_eq!(mangle_macos_name("main"), "_main");
        assert_eq!(mangle_macos_name("foo"), "_foo");
        assert_eq!(mangle_macos_name("my_function"), "_my_function");
    }

    #[test]
    fn test_get_printf_symbol() {
        assert_eq!(get_printf_symbol(TargetOperatingSystem::MacOS), "_printf");
        assert_eq!(get_printf_symbol(TargetOperatingSystem::Linux), "printf");
        assert_eq!(get_printf_symbol(TargetOperatingSystem::Windows), "printf");
    }

    #[test]
    fn test_get_malloc_symbol() {
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::MacOS), "_malloc");
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::Linux), "malloc");
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::Windows), "malloc");
    }

    #[test]
    fn test_get_free_symbol() {
        assert_eq!(get_free_symbol(TargetOperatingSystem::MacOS), "_free");
        assert_eq!(get_free_symbol(TargetOperatingSystem::Linux), "free");
        assert_eq!(get_free_symbol(TargetOperatingSystem::Windows), "free");
    }

    #[test]
    fn test_builtin_library_default_symbols() {
        let lib_macos = BuiltinLibrary::from_env(TargetOperatingSystem::MacOS);
        assert_eq!(lib_macos.resolve("print"), Some("_printf".to_string()));
        assert_eq!(lib_macos.resolve("malloc"), Some("_malloc".to_string()));
        assert_eq!(lib_macos.resolve("dealloc"), Some("_free".to_string()));

        let lib_linux = BuiltinLibrary::from_env(TargetOperatingSystem::Linux);
        assert_eq!(lib_linux.resolve("print"), Some("printf".to_string()));
        assert_eq!(lib_linux.resolve("malloc"), Some("malloc".to_string()));
        assert_eq!(lib_linux.resolve("dealloc"), Some("free".to_string()));
    }

    #[test]
    fn test_builtin_library_unknown_symbol() {
        let lib = BuiltinLibrary::from_env(TargetOperatingSystem::Linux);
        assert_eq!(lib.resolve("unknown_function"), None);
    }

    #[test]
    fn test_common_call_stub() {
        assert_eq!(
            common_call_stub("print", TargetOperatingSystem::MacOS),
            Some("_printf".to_string())
        );
        assert_eq!(
            common_call_stub("print", TargetOperatingSystem::Linux),
            Some("printf".to_string())
        );
        assert_eq!(
            common_call_stub("malloc", TargetOperatingSystem::MacOS),
            Some("_malloc".to_string())
        );
        assert_eq!(
            common_call_stub("dealloc", TargetOperatingSystem::Linux),
            Some("free".to_string())
        );
        assert_eq!(
            common_call_stub("unknown", TargetOperatingSystem::Linux),
            None
        );
    }
}
