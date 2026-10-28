//! WASM ABI utilities
//!
//! WASM has a more standardized ABI compared to native platforms,
//! but we still need utilities for function naming and module structure.

use crate::abi::Abi;
use lamina_mir::MirType;
use lamina_platform::TargetOperatingSystem;

/// Platform-specific ABI utilities for WASM code generation.
pub struct WasmABI {
    target_os: TargetOperatingSystem,
}

impl WasmABI {
    /// Creates a new ABI instance for the specified target OS.
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    /// Get the WASM function name (currently just the original name).
    /// WASM doesn't require name mangling, but this method exists for consistency
    /// with other backends and future extensibility.
    pub fn mangle_function_name(&self, name: &str) -> String {
        name.to_string()
    }

    /// Helpers the backend calls but emits no body for, each paired with the symbol a
    /// call site names. A module that calls none of them imports none, so it needs no
    /// host beyond what its own code asks for.
    pub fn get_runtime_imports(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "$writebyte",
                "(import \"env\" \"writebyte\" (func $writebyte (param i64) (result i64)))",
            ),
            (
                "$readbyte",
                "(import \"env\" \"readbyte\" (func $readbyte (result i64)))",
            ),
            (
                "$malloc",
                "(import \"env\" \"malloc\" (func $malloc (param i64) (result i64)))",
            ),
            (
                "$free",
                "(import \"env\" \"free\" (func $free (param i64)))",
            ),
            (
                "$write",
                "(import \"env\" \"write\" (func $write (param i64 i64) (result i64)))",
            ),
            (
                "$read",
                "(import \"env\" \"read\" (func $read (param i64 i64) (result i64)))",
            ),
            (
                "$memcpy",
                "(import \"env\" \"memcpy\" (func $memcpy (param i64 i64 i64)))",
            ),
            (
                "$memmove",
                "(import \"env\" \"memmove\" (func $memmove (param i64 i64 i64)))",
            ),
            (
                "$memset",
                "(import \"env\" \"memset\" (func $memset (param i64 i64 i64)))",
            ),
        ]
    }

    /// Get the WASM type for MIR types (currently only i64).
    pub fn get_wasm_type(&self, ty: &MirType) -> &'static str {
        let _ = ty;
        "i64"
    }

    /// Generate WASM local variable declaration.
    pub fn generate_local_decl(&self, index: usize) -> String {
        format!("    (local $l{index} i64)")
    }
}

impl Abi for WasmABI {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        WasmABI::mangle_function_name(self, name)
    }
}
