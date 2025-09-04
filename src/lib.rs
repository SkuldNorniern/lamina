//! # Lamina IR Compiler Library
//!
//! Lamina is an intermediate representation (IR) compiler library that provides a
//! high-level, safe way to generate assembly code for multiple target architectures.
//!
//! ## Main Usage Pattern
//!
//! The primary way to use Lamina is through the **builder API** in the `ir::builder` module.
//! This fluent API allows you to construct IR code programmatically without dealing
//! with low-level instruction objects directly.
//!
//! ## Quick Start
//!
//! ```rust
//! use lamina::{IRBuilder, Type, PrimitiveType, BinaryOp, var, i32};
//!
//! // Create a builder and define a simple function
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function("add", Type::Primitive(PrimitiveType::I32))
//!     .binary(
//!         BinaryOp::Add,
//!         "result",
//!         PrimitiveType::I32,
//!         var("a"),
//!         i32(42)
//!     )
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//!
//! let module = builder.build();
//!
//! // Compile to assembly (for host architecture)
//! let ir_code = r#"
//! fn @add(i32 %a, i32 %b) -> i32 {
//!   entry:
//!     %result = add.i32 %a, %b
//!     ret.i32 %result
//! }
//! "#;
//! let mut output = Vec::new();
//! lamina::compile_lamina_ir_to_assembly(ir_code, &mut output).unwrap();
//! ```
//!
//! ## Architecture Support
//!
//! Lamina currently supports:
//! - **x86_64**: Linux, macOS, Windows, and unknown targets
//! - **aarch64**: Linux, macOS, Windows, and unknown targets
//!
//! ## Module Structure
//!
//! - `ir::builder`: Main API for constructing IR code (recommended)
//! - `ir`: Core IR data structures and types
//! - `parser`: Parse Lamina IR text format into modules
//! - `codegen`: Generate assembly code from IR modules
//! - `error`: Error types and Result aliases
//!
//! ## Alternative Usage
//!
//! While the builder API is recommended, you can also:
//! - Parse IR from text using `parser::parse_module()`
//! - Construct IR structures manually using the types in `ir`
//! - Generate assembly directly using the codegen functions

pub mod codegen;
pub mod error;
pub mod ir;
pub mod parser;

use std::io::Write;

// Re-export core IR structures and builder API for easier access
//
// These re-exports allow users to access the most commonly used types and functions
// without needing to know the internal module structure. The builder API in particular
// is the recommended way to construct Lamina IR code programmatically.

/// Builder API - The primary way to construct Lamina IR code
pub use ir::builder::{IRBuilder, var, i8, i32, i64, f32, bool, string, global};

/// Error handling types
pub use error::{LaminaError, Result};

/// Core IR data structures
pub use ir::{
    function::{BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature},
    instruction::{AllocType, BinaryOp, CmpOp, Instruction},
    module::{GlobalDeclaration, Module, TypeDeclaration},
    types::{Identifier, Label, Literal, PrimitiveType, StructField, Type, Value},
};

// Internal use - not typically needed by end users
use codegen::CodegenError;
pub use codegen::generate_x86_64_assembly;

/// List of supported target architectures for compilation.
///
/// These strings can be passed to `compile_lamina_ir_to_target_assembly()`
/// to specify the target platform for assembly generation.
///
/// Current supported targets:
/// - **x86_64 variants**: `x86_64_unknown`, `x86_64_linux`, `x86_64_windows`, `x86_64_macos`
/// - **aarch64 variants**: `aarch64_unknown`, `aarch64_macos`, `aarch64_linux`, `aarch64_windows`
///
/// Note: While multiple target strings exist for each architecture family,
/// the current implementation uses the same code generation logic for all
/// variants within an architecture family. Future versions may add
/// target-specific optimizations.
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

/// Detect the host system's architecture and operating system combination.
///
/// This function uses compile-time feature detection to determine the current
/// platform's architecture and OS, returning a target string that can be used
/// with `compile_lamina_ir_to_target_assembly()`.
///
/// # Returns
///
/// A string from `HOST_ARCH_LIST` representing the detected target:
/// - `"x86_64_linux"` on x86_64 Linux systems
/// - `"x86_64_macos"` on x86_64 macOS systems
/// - `"x86_64_windows"` on x86_64 Windows systems
/// - `"x86_64_unknown"` on x86_64 systems with unrecognized OS
/// - `"aarch64_linux"` on ARM64 Linux systems
/// - `"aarch64_macos"` on ARM64 macOS systems
/// - `"aarch64_windows"` on ARM64 Windows systems
/// - `"aarch64_unknown"` on ARM64 systems with unrecognized OS
///
/// # Panics
///
/// This function does not panic. It falls back to `"x86_64_unknown"` for
/// any unsupported architecture/OS combinations.
///
/// # Examples
///
/// ```rust
/// use lamina::detect_host_architecture;
///
/// let target = detect_host_architecture();
/// println!("Current platform: {}", target);
///
/// // Use with compilation
/// let ir_code = r#"
/// fn @main() -> i32 {
///   entry:
///     ret.i32 42
/// }
/// "#;
/// let mut output = Vec::new();
/// lamina::compile_lamina_ir_to_target_assembly(ir_code, &mut output, target).unwrap();
/// ```
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
            "aarch64_macos"
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

/// Parse Lamina IR text and generate assembly code for the host system's architecture.
///
/// This is the most convenient way to compile Lamina IR code, as it automatically
/// detects the current platform and generates appropriate assembly code.
///
/// # Arguments
/// * `input_ir` - A string slice containing valid Lamina IR code
/// * `output_asm` - A mutable writer where the generated assembly will be written
///
/// # Returns
/// * `Result<()>` - Ok if compilation succeeds, Err with detailed error information otherwise
///
/// # Errors
/// This function can fail if:
/// - The input IR code is malformed or invalid
/// - The detected host architecture is not supported
/// - Writing to the output fails
///
/// # Examples
///
/// ```rust
/// use lamina::compile_lamina_ir_to_assembly;
/// use std::io::stdout;
///
/// // Example Lamina IR code (this is a simplified example)
/// let ir_code = r#"
/// function @add(i32, i32) -> i32 {
///   %result = add i32 %0, %1
///   ret i32 %result
/// }
/// "#;
///
/// // Compile to assembly for the current platform
/// match compile_lamina_ir_to_assembly(ir_code, &mut stdout()) {
///     Ok(()) => println!("Compilation successful!"),
///     Err(e) => eprintln!("Compilation failed: {}", e),
/// }
/// ```
pub fn compile_lamina_ir_to_assembly<W: Write>(input_ir: &str, output_asm: &mut W) -> Result<()> {
    // Use the host architecture as the default target
    let target = detect_host_architecture();
    compile_lamina_ir_to_target_assembly(input_ir, output_asm, target)
}

/// Parse Lamina IR text and generate assembly code for a specific target architecture.
///
/// This function provides full control over the compilation target, allowing you to
/// cross-compile for different architectures. Use `HOST_ARCH_LIST` to see all
/// supported targets, or call `detect_host_architecture()` for the current platform.
///
/// # Arguments
/// * `input_ir` - A string slice containing valid Lamina IR code
/// * `output_asm` - A mutable writer where the generated assembly will be written
/// * `target` - Target architecture string from `HOST_ARCH_LIST` (e.g., `"x86_64_linux"`, `"aarch64_macos"`)
///
/// # Returns
/// * `Result<()>` - Ok if compilation succeeds, Err with detailed error information otherwise
///
/// # Errors
/// This function can fail if:
/// - The input IR code is malformed or invalid
/// - The specified target architecture is not supported (see `HOST_ARCH_LIST`)
/// - Writing to the output fails
/// - Internal compilation errors occur
///
/// # Examples
///
/// ```rust
/// use lamina::{compile_lamina_ir_to_target_assembly, HOST_ARCH_LIST};
/// use std::io::stdout;
///
/// // Example Lamina IR code
/// let ir_code = r#"
/// function @main() -> i32 {
///   ret i32 42
/// }
/// "#;
///
/// // Compile for a specific target
/// let target = "x86_64_linux";
/// let mut output = Vec::new();
///
/// match compile_lamina_ir_to_target_assembly(ir_code, &mut output, target) {
///     Ok(()) => {
///         let assembly = String::from_utf8_lossy(&output);
///         println!("Generated assembly:\n{}", assembly);
///     }
///     Err(e) => eprintln!("Compilation failed: {}", e),
/// }
///
/// // List all supported targets
/// println!("Supported targets: {:?}", HOST_ARCH_LIST);
/// ```
///
/// # Cross-compilation
///
/// This function enables cross-compilation by allowing you to specify any supported target:
///
/// ```rust
/// use lamina::compile_lamina_ir_to_target_assembly;
///
/// // Compile ARM64 code on an x86_64 host
/// let ir_code = r#"
/// fn @main() -> i32 {
///   entry:
///     ret.i32 42
/// }
/// "#;
/// let target = "aarch64_linux";
/// let mut output = Vec::new();
/// compile_lamina_ir_to_target_assembly(ir_code, &mut output, target).unwrap();
/// ```
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
            return Err(error::LaminaError::CodegenError(
                CodegenError::UnsupportedFeature(codegen::FeatureType::Custom(format!(
                    "Unsupported target architecture: {}",
                    target
                ))),
            ));
        }
    }

    Ok(())
}
