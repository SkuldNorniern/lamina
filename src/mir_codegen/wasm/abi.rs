//! WASM ABI utilities
//!
//! # WebAssembly ABI Documentation
//!
//! WebAssembly uses a stack-based calling convention, not register-based.
//!
//! **Arguments**: Passed on the stack in order (first argument on top).
//!
//! **Return Values**: Left on the stack after function execution.
//!
//! **Locals**: Function parameters become local variables.
//!
//! **Memory**: Linear memory is used for heap allocations.
//!
//! Unlike native platforms, WASM doesn't require name mangling or register allocation
//! for function calls. Parameters are passed as locals and return values are left
//! on the evaluation stack.
//!
//! ## Example: Function Call Codegen
//!
//! ```text
//! # Calling function foo(x, y, z) where x, y, z are in locals $l0, $l1, $l2
//! local.get $l0    # Push first argument
//! local.get $l1    # Push second argument
//! local.get $l2    # Push third argument
//! call $foo        # Call function (parameters consumed from stack)
//! local.set $l3    # Save return value (left on stack)
//!
//! # WASM function signature: (func $foo (param i64) (param i64) (param i64) (result i64))
//! ```

use crate::mir_codegen::abi::Abi;
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
    pub fn get_wasm_type(&self, ty: &crate::mir::MirType) -> &'static str {
        match ty {
            crate::mir::MirType::Scalar(crate::mir::ScalarType::I64) => "i64",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_abi_new() {
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.target_os(), TargetOperatingSystem::Linux);

        let abi_macos = WasmABI::new(TargetOperatingSystem::MacOS);
        assert_eq!(abi_macos.target_os(), TargetOperatingSystem::MacOS);
    }

    #[test]
    fn test_wasm_mangle_function_name() {
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        // WASM doesn't mangle names
        assert_eq!(abi.mangle_function_name("main"), "main");
        assert_eq!(abi.mangle_function_name("foo"), "foo");
        assert_eq!(abi.mangle_function_name("_main"), "_main");

        let abi_macos = WasmABI::new(TargetOperatingSystem::MacOS);
        // WASM doesn't mangle names even on macOS
        assert_eq!(abi_macos.mangle_function_name("main"), "main");
        assert_eq!(abi_macos.mangle_function_name("foo"), "foo");
    }

    #[test]
    fn test_wasm_get_print_import() {
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        assert_eq!(
            abi.get_print_import(),
            "(import \"console\" \"log\" (func $log (param i64)))"
        );
    }

    #[test]
    fn test_wasm_get_wasm_type() {
        use crate::mir::{MirType, ScalarType};
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        assert_eq!(
            abi.get_wasm_type(&MirType::Scalar(ScalarType::I64)),
            "i64"
        );
        // Currently everything maps to i64
        assert_eq!(
            abi.get_wasm_type(&MirType::Scalar(ScalarType::I32)),
            "i64"
        );
    }

    #[test]
    fn test_wasm_generate_global_decl() {
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        assert_eq!(
            abi.generate_global_decl(0),
            "  (global $vreg0 (mut i64) (i64.const 0))"
        );
        assert_eq!(
            abi.generate_global_decl(42),
            "  (global $vreg42 (mut i64) (i64.const 0))"
        );
    }

    #[test]
    fn test_wasm_generate_local_decl() {
        let abi = WasmABI::new(TargetOperatingSystem::Linux);
        assert_eq!(abi.generate_local_decl(0), "    (local $l0 i64)");
        assert_eq!(abi.generate_local_decl(5), "    (local $l5 i64)");
    }
}
