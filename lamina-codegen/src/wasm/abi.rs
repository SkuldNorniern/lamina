//! WASM ABI utilities
//!
//! WASM has a more standardized ABI compared to native platforms,
//! but we still need utilities for function naming and module structure.

use crate::abi::Abi;
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

    /// Get the WASM import for the print function.
    pub fn get_print_import(&self) -> &'static str {
        "(import \"console\" \"log\" (func $log (param i64)))"
    }

    /// Get the WASM type for MIR types (currently only i64).
    pub fn get_wasm_type(&self, ty: &lamina_mir::MirType) -> &'static str {
        match ty {
            lamina_mir::MirType::Scalar(lamina_mir::ScalarType::I64) => "i64",
            // For now, everything is i64 in WASM
            _ => "i64",
        }
    }

    /// Generate WASM global variable declaration for virtual registers.
    pub fn generate_global_decl(&self, index: usize) -> String {
        format!("  (global $vreg{} (mut i64) (i64.const 0))", index)
    }

    /// Generate WASM local variable declaration.
    pub fn generate_local_decl(&self, index: usize) -> String {
        format!("    (local $l{} i64)", index)
    }
}

impl Abi for WasmABI {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        WasmABI::mangle_function_name(self, name)
    }

    // WASM doesn't need call_stub since it uses imports, so we use the default implementation
}
