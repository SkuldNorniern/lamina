//! Lamina IR inline macro (`lamina!`)
//!
//! Provides a macro for embedding Lamina IR code directly in Rust, similar to `asm!`.
//!
//! # Example
//!
//! ```rust
//! use lamina::lamina;
//!
//! let add_fn = lamina!(
//!     r#"
//!     fn @add(i64 %a, i64 %b) -> i64 {
//!         entry:
//!             %res = add.i64 %a, %b
//!             ret.i64 %res
//!     }
//!     "#
//! );
//!
//! let result = unsafe { add_fn(10, 20) };
//! assert_eq!(result, 30);
//! ```

use crate::error::LaminaError;
use crate::mir::codegen::from_ir;
use crate::parser::parse_module;
use lamina_platform::{detect_host, TargetArchitecture, TargetOperatingSystem};

/// Compiles Lamina IR code at runtime and returns a function pointer.
///
/// This function parses, lowers to MIR, and JIT compiles the provided IR code.
/// The function name must be specified with `@` prefix (e.g., `@add`).
///
/// # Arguments
///
/// * `ir_code` - String containing Lamina IR code
/// * `function_name` - Name of the function to compile (with or without `@` prefix)
/// * `codegen_units` - Number of parallel compilation units (default: 1)
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - `Ok(function_ptr)` - A function pointer to the compiled code
/// - `Err(LaminaError)` - If parsing, lowering, or compilation fails
///
/// # Safety
///
/// The returned function pointer is unsafe to call. The caller must ensure:
/// 1. The function signature matches the expected signature
/// 2. The memory remains valid for the lifetime of the function pointer
/// 3. Arguments match the function's parameter types
pub fn compile_lir_function(
    ir_code: &str,
    function_name: &str,
    codegen_units: usize,
) -> Result<unsafe extern "C" fn() -> i64, LaminaError> {
    let host = detect_host();
    let target_arch = host.arch;
    let target_os = host.os;

    let ir_module = parse_module(ir_code)?;
    let mir_module = from_ir(&ir_module, "lir_macro")?;

    let runtime_result = crate::runtime::compile_to_runtime(
        &mir_module,
        target_arch,
        target_os,
        Some(function_name),
    )?;

    unsafe {
        let ptr: *const unsafe extern "C" fn() -> i64 = runtime_result.function_ptr as *const _;
        Ok(*ptr)
    }
}

/// Compiles Lamina IR code for a function that takes i64 arguments and returns i64.
///
/// Similar to `compile_lir_function` but with a typed signature.
pub fn compile_lir_function_i64(
    ir_code: &str,
    function_name: &str,
    codegen_units: usize,
) -> Result<unsafe extern "C" fn(i64) -> i64, LaminaError> {
    let host = detect_host();
    let target_arch = host.arch;
    let target_os = host.os;

    let ir_module = parse_module(ir_code)?;
    let mir_module = from_ir(&ir_module, "lir_macro")?;

    let runtime_result = crate::runtime::compile_to_runtime(
        &mir_module,
        target_arch,
        target_os,
        Some(function_name),
    )?;

    unsafe {
        let ptr: *const unsafe extern "C" fn(i64) -> i64 = runtime_result.function_ptr as *const _;
        Ok(*ptr)
    }
}

/// Compiles Lamina IR code for a function that takes two i64 arguments and returns i64.
pub fn compile_lir_function_i64_i64(
    ir_code: &str,
    function_name: &str,
    codegen_units: usize,
) -> Result<unsafe extern "C" fn(i64, i64) -> i64, LaminaError> {
    let host = detect_host();
    let target_arch = host.arch;
    let target_os = host.os;

    let ir_module = parse_module(ir_code)?;
    let mir_module = from_ir(&ir_module, "lir_macro")?;

    let runtime_result = crate::runtime::compile_to_runtime(
        &mir_module,
        target_arch,
        target_os,
        Some(function_name),
    )?;

    unsafe {
        let ptr: *const unsafe extern "C" fn(i64, i64) -> i64 = runtime_result.function_ptr as *const _;
        Ok(*ptr)
    }
}
