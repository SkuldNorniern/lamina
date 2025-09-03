pub mod codegen;
pub mod error;
pub mod ir;
pub mod parser;

use std::io::Write;

// Re-export core IR structures for easier access
pub use codegen::generate_x86_64_assembly;
pub use error::{LaminaError, Result};
pub use ir::{
    function::{BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature},
    instruction::{AllocType, BinaryOp, CmpOp, Instruction},
    module::{GlobalDeclaration, Module, TypeDeclaration},
    types::{Identifier, Label, Literal, PrimitiveType, StructField, Type, Value},
};

pub const HOST_ARCH_LIST: &[&str] = &[
    "x86_64_unknown",
    "x86_64_linux",
    "x86_64_windows",
    "x86_64_macos",
    "aarch64_unknown",
    "aarch64_macos",
    "aarch64_linux",
    "aarch64_windows",
];

/// Detect the host system's architecture.
///
/// Returns a string representing the detected architecture and host system: "x86_64" or "aarch64".
/// 
/// Target List:
/// x86_64_unknown
/// x86_64_linux
/// x86_64_windows
/// x86_64_macos - since Intel mac is heading to a end, it's not supported as much as aarch64 
/// aarch64_unknown
/// aarch64_macos
///
/// Falls back to "x86_64" if detection fails.
pub fn detect_host_architecture() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_os = "macos")]
        {
            return "x86_64_macos";
        }
        #[cfg(target_os = "linux")]
        {
            return "x86_64_linux";
        }
        #[cfg(target_os = "windows")]
        {
            return "x86_64_windows";
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            return "x86_64_unknown";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_os = "macos")]
        {
            return "aarch64_macos"
        }
        #[cfg(target_os = "linux")]
        {
            return "aarch64_linux";
        }
        #[cfg(target_os = "windows")]
        {
            return "aarch64_windows";
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            return "aarch64_unknown";
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Default to x86_64 for unsupported architectures
        return "x86_64_unknown";
    }
    // FEAT:TODO Add support for other architectures (RISC-V, etc.)
}

/// Parses Lamina IR text and generates assembly code using the host system's architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
pub fn compile_lamina_ir_to_assembly<W: Write>(input_ir: &str, output_asm: &mut W) -> Result<()> {
    // Use the host architecture as the default target
    let target = detect_host_architecture();
    compile_lamina_ir_to_target_assembly(input_ir, output_asm, target)
}

/// Parses Lamina IR text and generates assembly code for a specific target architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
/// * `target` - A string slice specifying the target architecture (e.g., "x86_64", "aarch64").
///
/// # Returns
/// * `Result<()>` - Ok if compilation succeeds, Err with error information otherwise.
pub fn compile_lamina_ir_to_target_assembly<W: Write>(
    input_ir: &str,
    output_asm: &mut W,
    target: &str,
) -> Result<()> {
    // 1. Parse the input string into an IR Module
    let module = parser::parse_module(input_ir)?;

    // 2. Generate assembly for the specified target
    match target {
        "x86_64_unknown" => codegen::generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_macos" => codegen::generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_linux" => codegen::generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_windows" => codegen::generate_x86_64_assembly(&module, output_asm)?,
        // FEAT:TODO Add per-target generation refinements for macOS/Linux/Windows
        "aarch64_unknown" => codegen::generate_aarch64_assembly(&module, output_asm)?,
        "aarch64_macos" => codegen::generate_aarch64_assembly(&module, output_asm)?,
        "aarch64_linux" => codegen::generate_aarch64_assembly(&module, output_asm)?,
        "aarch64_windows" => codegen::generate_aarch64_assembly(&module, output_asm)?,
        _ => {
            return Err(error::LaminaError::CodegenError(format!(
                "Unsupported target architecture: {}",
                target
            )));
        }
    }

    Ok(())
}
