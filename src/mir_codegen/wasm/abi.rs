/// WASM ABI utilities
///
/// WASM has a more standardized ABI compared to native platforms,
/// but we still need utilities for function naming and module structure.
pub struct WasmABI;

impl WasmABI {
    /// Get the WASM function name (currently just the original name)
    pub fn mangle_function_name(name: &str) -> String {
        name.to_string()
    }

    /// Get the WASM import for the print function
    pub fn get_print_import() -> &'static str {
        "(import \"console\" \"log\" (func $log (param i64)))"
    }

    /// Get the WASM type for MIR types (currently only i64)
    pub fn get_wasm_type(ty: &crate::mir::MirType) -> &'static str {
        match ty {
            crate::mir::MirType::Scalar(crate::mir::ScalarType::I64) => "i64",
            // For now, everything is i64 in WASM
            _ => "i64",
        }
    }

    /// Generate WASM global variable declaration for virtual registers
    pub fn generate_global_decl(index: usize) -> String {
        format!("  (global $vreg{} (mut i64) (i64.const 0))", index)
    }

    /// Generate WASM local variable declaration
    pub fn generate_local_decl(index: usize) -> String {
        format!("    (local $l{} i64)", index)
    }
}

