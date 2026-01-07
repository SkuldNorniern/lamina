//! Helper functions for the `lir!` macro
//!
//! These functions are used internally by the macro to compile Lamina IR code.

use crate::error::LaminaError;
use crate::mir::codegen::from_ir;
use crate::parser::parse_module;
use lamina_platform::Target;

/// Compiles Lamina IR code at runtime and returns a function pointer.
///
/// This is a helper function used by the `lir!` macro. It parses, lowers to MIR,
/// and JIT compiles the provided IR code.
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
pub fn compile_lir_internal(
    ir_code: &str,
    function_name: &str,
    codegen_units: usize,
) -> Result<*const u8, LaminaError> {
    let host = Target::detect_host();
    let target_arch = host.architecture;
    let target_os = host.operating_system;

    let ir_module = parse_module(ir_code)?;
    let mir_module = from_ir(&ir_module, "lir_macro")?;

    let runtime_result = crate::runtime::compile_to_runtime(
        &mir_module,
        target_arch,
        target_os,
        Some(function_name),
    )?;

    Ok(runtime_result.function_ptr)
}

/// Extracts the function name from Lamina IR code string.
///
/// Looks for patterns like `fn @name(` or `fn name(` and returns the name.
///
/// # Arguments
///
/// * `ir_code` - String containing Lamina IR code
///
/// # Returns
///
/// Returns `Some(name)` if a function name is found, `None` otherwise.
/// The name will NOT include the `@` prefix (caller should add it if needed).
pub fn extract_function_name(ir_code: &str) -> Option<&str> {
    // Look for "fn @" pattern
    if let Some(start) = ir_code.find("fn @") {
        let after_fn = &ir_code[start + 4..];
        if let Some(end) = after_fn.find(|c: char| c == '(' || c.is_whitespace()) {
            let name = &after_fn[..end];
            // Return name with @ prefix
            return Some(name);
        }
    }
    
    // Fallback: look for "fn " pattern (without @)
    if let Some(start) = ir_code.find("fn ") {
        let after_fn = &ir_code[start + 3..];
        if let Some(end) = after_fn.find(|c: char| c == '(' || c.is_whitespace()) {
            let name = &after_fn[..end];
            // Return as-is
            return Some(name);
        }
    }
    
    None
}

