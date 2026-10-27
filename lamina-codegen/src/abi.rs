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
//! Every runtime helper the compiler calls is a [`RuntimeSymbol`]. Keys are the
//! C names (`malloc`, `free`, `printf`, `fflush`, `exit`, `memcpy`, `memmove`,
//! `memset`), plus the legacy aliases `print` (printf) and `dealloc` (free).
//! Platform decoration (`_printf` on macOS) is applied to the override as well.
//!
//! ## Backend-Specific ABIs
//!
//! See the documentation in each backend's ABI module:
//! - `x86_64::abi` - System V AMD64 and Microsoft x64
//! - `aarch64::abi` - AAPCS64
//! - `riscv::abi` - RISC-V calling convention
//! - `wasm::abi` - WebAssembly stack-based calling

use crate::target_support::os_uses_macho;
use lamina_platform::TargetOperatingSystem;
use std::{collections::HashMap, env};

/// Runtime helpers the compiler calls by name.
///
/// Each variant names a C symbol. Emission sites use the variant, never a string
/// literal, so `LAMINA_BUILTINS` can retarget every one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeSymbol {
    Malloc,
    Free,
    Printf,
    Fflush,
    Exit,
    Memcpy,
    Memmove,
    Memset,
}

impl RuntimeSymbol {
    pub const ALL: [RuntimeSymbol; 8] = [
        RuntimeSymbol::Malloc,
        RuntimeSymbol::Free,
        RuntimeSymbol::Printf,
        RuntimeSymbol::Fflush,
        RuntimeSymbol::Exit,
        RuntimeSymbol::Memcpy,
        RuntimeSymbol::Memmove,
        RuntimeSymbol::Memset,
    ];

    /// The undecorated C symbol name.
    pub const fn c_name(self) -> &'static str {
        match self {
            RuntimeSymbol::Malloc => "malloc",
            RuntimeSymbol::Free => "free",
            RuntimeSymbol::Printf => "printf",
            RuntimeSymbol::Fflush => "fflush",
            RuntimeSymbol::Exit => "exit",
            RuntimeSymbol::Memcpy => "memcpy",
            RuntimeSymbol::Memmove => "memmove",
            RuntimeSymbol::Memset => "memset",
        }
    }

    /// Looks up a symbol by its C name, as MIR call sites spell it.
    pub fn from_c_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|symbol| symbol.c_name() == name)
    }

    /// Looks up a symbol by a `LAMINA_BUILTINS` key or builtin name.
    ///
    /// Accepts the C name plus the legacy keys `print` (printf) and `dealloc` (free).
    pub fn from_builtin_name(name: &str) -> Option<Self> {
        match name {
            "dealloc" => Some(RuntimeSymbol::Free),
            "print" => Some(RuntimeSymbol::Printf),
            _ => Self::from_c_name(name),
        }
    }
}

/// Resolves runtime helper names to the symbols the emitted code must reference.
///
/// Overrides name the C symbol. Platform decoration (the Mach-O leading
/// underscore) is applied on top, the same way it is for every other C symbol.
pub struct BuiltinLibrary {
    target_os: TargetOperatingSystem,
    overrides: HashMap<RuntimeSymbol, String>,
}

impl BuiltinLibrary {
    pub fn new(
        target_os: TargetOperatingSystem,
        overrides: HashMap<RuntimeSymbol, String>,
    ) -> Self {
        Self {
            target_os,
            overrides,
        }
    }

    pub fn from_env(target_os: TargetOperatingSystem) -> Self {
        Self::new(target_os, builtin_overrides_from_env())
    }

    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }

    /// The undecorated C name for a symbol. MIR call sites carry this name and the
    /// backend's function mangling adds any platform prefix.
    pub fn c_name(&self, symbol: RuntimeSymbol) -> &str {
        self.overrides
            .get(&symbol)
            .map_or(symbol.c_name(), String::as_str)
    }

    /// The final symbol to reference from directly emitted code.
    pub fn symbol(&self, symbol: RuntimeSymbol) -> String {
        decorate_symbol(self.target_os, self.c_name(symbol))
    }

    pub fn resolve(&self, name: &str) -> Option<String> {
        RuntimeSymbol::from_builtin_name(name).map(|symbol| self.symbol(symbol))
    }
}

fn decorate_symbol(target_os: TargetOperatingSystem, name: &str) -> String {
    if os_uses_macho(target_os) {
        format!("_{name}")
    } else {
        name.to_string()
    }
}

/// Parses `name=symbol,name=symbol`. Entries with an unknown name or an empty side are skipped.
pub fn parse_builtin_overrides(value: &str) -> HashMap<RuntimeSymbol, String> {
    let mut overrides = HashMap::new();
    for entry in value.split(',') {
        let mut parts = entry.splitn(2, '=');
        let name = parts.next().unwrap_or("").trim();
        let symbol = parts.next().unwrap_or("").trim();
        if symbol.is_empty() {
            continue;
        }
        if let Some(runtime) = RuntimeSymbol::from_builtin_name(name) {
            overrides.insert(runtime, symbol.to_string());
        }
    }
    overrides
}

fn builtin_overrides_from_env() -> HashMap<RuntimeSymbol, String> {
    env::var("LAMINA_BUILTINS")
        .map(|value| parse_builtin_overrides(&value))
        .unwrap_or_default()
}

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
/// - The builtin map recognizes every [`RuntimeSymbol`].
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
        format!("_{name}")
    }
}

/// Helper function to get the platform-specific printf symbol name.
///
/// Returns "_printf" on macOS, "printf" on other platforms.
pub fn get_printf_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        os if os_uses_macho(os) => "_printf",
        _ => "printf",
    }
}

/// Helper function to get the platform-specific malloc symbol name.
///
/// Returns "_malloc" on macOS, "malloc" on other platforms.
pub fn get_malloc_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        os if os_uses_macho(os) => "_malloc",
        _ => "malloc",
    }
}

/// Helper function to get the platform-specific free symbol name.
///
/// Returns "_free" on macOS, "free" on other platforms.
pub fn get_free_symbol(target_os: TargetOperatingSystem) -> &'static str {
    match target_os {
        os if os_uses_macho(os) => "_free",
        _ => "free",
    }
}

/// Helper function to create a common call_stub implementation.
///
/// Resolves a builtin name through the environment-configured [`BuiltinLibrary`].
/// Backends can use this to reduce duplication in their `call_stub` implementations.
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
        assert_eq!(get_printf_symbol(TargetOperatingSystem::IOS), "_printf");
        assert_eq!(
            get_printf_symbol(TargetOperatingSystem::VisionOS),
            "_printf"
        );
        assert_eq!(get_printf_symbol(TargetOperatingSystem::Linux), "printf");
        assert_eq!(get_printf_symbol(TargetOperatingSystem::Windows), "printf");
    }

    #[test]
    fn test_get_malloc_symbol() {
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::MacOS), "_malloc");
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::IOS), "_malloc");
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::Linux), "malloc");
        assert_eq!(get_malloc_symbol(TargetOperatingSystem::Windows), "malloc");
    }

    #[test]
    fn test_get_free_symbol() {
        assert_eq!(get_free_symbol(TargetOperatingSystem::MacOS), "_free");
        assert_eq!(get_free_symbol(TargetOperatingSystem::TvOS), "_free");
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

#[cfg(test)]
mod runtime_symbol_tests {
    use super::*;

    fn all_overridden(os: TargetOperatingSystem) -> BuiltinLibrary {
        let overrides = RuntimeSymbol::ALL
            .into_iter()
            .map(|symbol| (symbol, format!("k_{}", symbol.c_name())))
            .collect();
        BuiltinLibrary::new(os, overrides)
    }

    #[test]
    fn default_symbols_keep_c_names_and_platform_decoration() {
        let linux = BuiltinLibrary::new(TargetOperatingSystem::Linux, HashMap::new());
        let macos = BuiltinLibrary::new(TargetOperatingSystem::MacOS, HashMap::new());
        assert!(!linux.has_overrides());
        for symbol in RuntimeSymbol::ALL {
            assert_eq!(linux.c_name(symbol), symbol.c_name());
            assert_eq!(linux.symbol(symbol), symbol.c_name());
            assert_eq!(macos.c_name(symbol), symbol.c_name());
            assert_eq!(macos.symbol(symbol), format!("_{}", symbol.c_name()));
        }
    }

    #[test]
    fn overrides_replace_every_symbol() {
        for os in [TargetOperatingSystem::Linux, TargetOperatingSystem::MacOS] {
            let lib = all_overridden(os);
            assert!(lib.has_overrides());
            for symbol in RuntimeSymbol::ALL {
                let expected_c = format!("k_{}", symbol.c_name());
                assert_eq!(lib.c_name(symbol), expected_c);
                assert_eq!(
                    lib.resolve(symbol.c_name()),
                    Some(decorate_symbol(os, &expected_c))
                );
            }
        }
    }

    #[test]
    fn resolve_maps_legacy_builtin_keys() {
        let lib = BuiltinLibrary::new(
            TargetOperatingSystem::Linux,
            parse_builtin_overrides("print=k_print,dealloc=k_free"),
        );
        assert_eq!(lib.resolve("print"), Some("k_print".to_string()));
        assert_eq!(lib.resolve("dealloc"), Some("k_free".to_string()));
        assert_eq!(lib.resolve("free"), Some("k_free".to_string()));
        assert_eq!(lib.resolve("write"), None);
    }

    #[test]
    fn parse_builtin_overrides_skips_unknown_and_empty_entries() {
        let parsed = parse_builtin_overrides(" malloc = k_alloc ,bogus=x,memset=,=y,exit=k_exit");
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            parsed.get(&RuntimeSymbol::Malloc).map(String::as_str),
            Some("k_alloc")
        );
        assert_eq!(
            parsed.get(&RuntimeSymbol::Exit).map(String::as_str),
            Some("k_exit")
        );
    }

    #[test]
    fn from_c_name_round_trips_every_symbol() {
        for symbol in RuntimeSymbol::ALL {
            assert_eq!(RuntimeSymbol::from_c_name(symbol.c_name()), Some(symbol));
            assert_eq!(
                RuntimeSymbol::from_builtin_name(symbol.c_name()),
                Some(symbol)
            );
        }
        assert_eq!(RuntimeSymbol::from_c_name("print"), None);
    }
}
