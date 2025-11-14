//! # Lamina Compiler Library
//!
//! Lamina is a modern compiler that generates efficient machine code from a high-level
//! intermediate representation (IR). It supports multiple target architectures and provides
//! a comprehensive set of tools for building compilers, interpreters, and language runtimes.
//!
//! ## Overview
//!
//! Lamina consists of several key components:
//!
//! - **IR (Intermediate Representation)**: A low-level, architecture-agnostic representation
//!   of programs that serves as the bridge between high-level source code and machine code.
//! - **Parser**: Converts text-based IR into structured data that can be processed by the compiler.
//! - **Code Generator**: Translates IR into native assembly code for various target architectures.
//! - **Error Handling**: Comprehensive error reporting and recovery mechanisms.
//!
//! ## Quick Start
//!
//! ```rust
//! use lamina::{compile_lamina_ir_to_assembly};
//! use lamina::target::{Target, architecture_name};
//! use std::io::Write;
//!
//! // Detect the host architecture
//! let target = Target::detect_host();
//! println!("Host target: {}", target);
//!
//! // Or just get the architecture name
//! let arch = architecture_name();
//! println!("Architecture: {}", arch);
//!
//! // Compile IR to assembly
//! let ir_code = r#"
//! fn @main() -> i64 {
//!   entry:
//!     %result = add.i64 42, 8
//!     ret.i64 %result
//! }
//! "#;
//!
//! let mut assembly = Vec::new();
//! compile_lamina_ir_to_assembly(ir_code, &mut assembly)?;
//! println!("Generated assembly:\n{}", String::from_utf8(assembly)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Architecture Support
//!
//! Lamina currently supports the following target architectures:
//!
//! - **x86_64**: Intel/AMD 64-bit processors
//!   - `x86_64_unknown` - Generic x86_64
//!   - `x86_64_linux` - Linux x86_64
//!   - `x86_64_windows` - Windows x86_64
//!   - `x86_64_macos` - macOS x86_64 (Intel Macs)
//!
//! - **AArch64**: ARM 64-bit processors
//!   - `aarch64_unknown` - Generic AArch64
//!   - `aarch64_linux` - Linux AArch64
//!   - `aarch64_windows` - Windows AArch64
//!   - `aarch64_macos` - macOS AArch64 (Apple Silicon)
//!
//! ## Core Modules
//!
//! ### IR (Intermediate Representation)
//!
//! The IR module provides the fundamental data structures for representing programs:
//!
//! - **Types**: Primitive types, structs, arrays, tuples, and function signatures
//! - **Instructions**: Arithmetic, memory operations, control flow, and function calls
//! - **Functions**: Complete function definitions with basic blocks
//! - **Modules**: Top-level containers for functions, types, and globals
//! - **Builder**: Fluent API for programmatically constructing IR
//!
//! ### Code Generation
//!
//! The codegen module translates IR into native assembly:
//!
//! - **Architecture-specific backends**: Separate implementations for x86_64 and AArch64
//! - **Register allocation**: Efficient use of target architecture registers
//! - **Instruction selection**: Optimal instruction choice for IR operations
//! - **ABI compliance**: Proper calling conventions and stack management
//!
//! ### Error Handling
//!
//! Comprehensive error reporting with detailed context:
//!
//! - **Parse errors**: Syntax and semantic errors in IR input
//! - **Codegen errors**: Architecture-specific compilation issues
//! - **Type errors**: Type checking and validation failures
//! - **Memory errors**: Stack overflow, invalid memory access, etc.
//!
//! ## Memory Management
//!
//! Lamina provides sophisticated memory management capabilities:
//!
//! - **Stack allocation**: Fast, automatic memory management for local variables
//! - **Heap allocation**: Manual memory management for persistent data
//! - **Pointer arithmetic**: Safe array and struct field access
//! - **Memory safety**: Bounds checking and validation (where possible)
//!
//! ## Examples
//!
//! ### Basic Arithmetic
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina::ir::builder::{var, i32, i64};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("add_numbers", Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(32))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("sum"));
//! ```
//!
//! ### Memory Operations
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("memory_demo", Type::Void)
//!     .alloc_stack("local", Type::Primitive(PrimitiveType::I32))
//!     .store(Type::Primitive(PrimitiveType::I32), var("local"), i32(42))
//!     .load("value", Type::Primitive(PrimitiveType::I32), var("local"))
//!     .print(var("value"))
//!     .ret_void();
//! ```
//!
//! ### Complete Builder Example: Memory Workflow
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina::ir::builder::{var, i32};
//!
//! // Create a new IR builder
//! let mut builder = IRBuilder::new();
//!
//! // Define a function that demonstrates memory operations
//! builder
//!     .function_with_params("memory_workflow", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "input",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!
//!     // Step 1: Allocate memory on stack
//!     .alloc_stack("buffer", Type::Primitive(PrimitiveType::I32))
//!
//!     // Step 2: Store the input value in our buffer
//!     .store(Type::Primitive(PrimitiveType::I32), var("buffer"), var("input"))
//!
//!     // Step 3: Load the value back from memory
//!     .load("loaded", Type::Primitive(PrimitiveType::I32), var("buffer"))
//!
//!     // Step 4: Perform arithmetic on the loaded value
//!     .binary(BinaryOp::Add, "result", PrimitiveType::I32, var("loaded"), i32(10))
//!
//!     // Step 5: Store the result back to memory
//!     .store(Type::Primitive(PrimitiveType::I32), var("buffer"), var("result"))
//!
//!     // Step 6: Load and return the final value
//!     .load("final", Type::Primitive(PrimitiveType::I32), var("buffer"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("final"));
//!
//! // Build the module
//! let module = builder.build();
//!
//! // The module now contains our memory_workflow function
//! assert!(module.functions.contains_key("memory_workflow"));
//! ```
//!
//! ### Advanced Builder Example: Control Flow with Memory
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp, CmpOp};
//! use lamina::ir::builder::{var, i32, string};
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function_with_params("process_data", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "data",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Void)
//!
//!     // Allocate memory for processing
//!     .alloc_stack("temp", Type::Primitive(PrimitiveType::I32))
//!     .store(Type::Primitive(PrimitiveType::I32), var("temp"), var("data"))
//!
//!     // Check if data is positive
//!     .cmp(CmpOp::Gt, "is_positive", PrimitiveType::I32, var("data"), i32(0))
//!     .branch(var("is_positive"), "positive_path", "negative_path")
//!
//!     // Positive path: double the value
//!     .block("positive_path")
//!     .load("current", Type::Primitive(PrimitiveType::I32), var("temp"))
//!     .binary(BinaryOp::Mul, "doubled", PrimitiveType::I32, var("current"), i32(2))
//!     .store(Type::Primitive(PrimitiveType::I32), var("temp"), var("doubled"))
//!     .print(string("Processed positive value"))
//!     .jump("cleanup")
//!
//!     // Negative path: take absolute value
//!     .block("negative_path")
//!     .load("current", Type::Primitive(PrimitiveType::I32), var("temp"))
//!     .binary(BinaryOp::Sub, "abs", PrimitiveType::I32, i32(0), var("current"))
//!     .store(Type::Primitive(PrimitiveType::I32), var("temp"), var("abs"))
//!     .print(string("Processed negative value"))
//!     .jump("cleanup")
//!
//!     // Cleanup: print final result
//!     .block("cleanup")
//!     .load("final_result", Type::Primitive(PrimitiveType::I32), var("temp"))
//!     .print(var("final_result"))
//!     .ret_void();
//!
//! let module = builder.build();
//! ```
//!
//! ### Control Flow
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, CmpOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("conditional", Type::Primitive(PrimitiveType::I32))
//!     .cmp(CmpOp::Lt, "is_negative", PrimitiveType::I32, var("x"), i32(0))
//!     .branch(var("is_negative"), "negative", "positive")
//!     .block("negative")
//!     .ret(Type::Primitive(PrimitiveType::I32), i32(-1))
//!     .block("positive")
//!     .ret(Type::Primitive(PrimitiveType::I32), i32(1));
//! ```
//!
//! ## Performance Considerations
//!
//! - **Zero-copy parsing**: IR uses string references to avoid unnecessary allocations
//! - **Efficient data structures**: HashMaps for O(1) lookups of functions and types
//! - **SSA form**: Single Static Assignment form enables powerful optimizations
//! - **Architecture-specific optimizations**: Tailored code generation for each target
//!
//! ## Thread Safety
//!
//! Lamina's IR structures are designed to be thread-safe when used correctly:
//!
//! - **Immutable by default**: IR structures are typically built once and then read-only
//! - **Copy semantics**: Most IR types implement `Clone` for easy duplication
//! - **Lifetime management**: Uses Rust's lifetime system to ensure memory safety
//!
//! ## Error Recovery
//!
//! The compiler provides detailed error messages to help with debugging:
//!
//! - **Source location**: Errors include line and column information
//! - **Context information**: Additional details about what went wrong
//! - **Suggestions**: Helpful hints for fixing common issues
//! - **Multiple errors**: Reports all errors found, not just the first one
//!
//! ## Future Roadmap
//!
//! - **Additional architectures**: RISC-V, WebAssembly, and more
//! - **Optimization passes**: Dead code elimination, constant folding, etc.
//! - **Debug information**: Source-level debugging support
//! - **Standard library**: Common functions and data structures
//! - **Language bindings**: C, Python, JavaScript, and other language interfaces

pub mod codegen;
pub mod mir_codegen;
pub mod target;

pub mod error;
pub mod ir;
pub mod mir;
pub mod parser;

use std::io::Write;

// Re-export core IR structures for easier access
use codegen::CodegenError;
pub use codegen::generate_x86_64_assembly;
pub use error::LaminaError;
pub use ir::{
    function::{BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature},
    instruction::{AllocType, BinaryOp, CmpOp, Instruction},
    module::{GlobalDeclaration, Module, TypeDeclaration},
    types::{Identifier, Label, Literal, PrimitiveType, StructField, Type, Value},
};
pub use mir_codegen::{
    generate_mir_to_aarch64, generate_mir_to_riscv, generate_mir_to_wasm, generate_mir_to_x86_64,
};


/// Parses Lamina IR text and generates assembly code using the host system's architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
pub fn compile_lamina_ir_to_assembly<W: Write>(
    input_ir: &str,
    output_asm: &mut W,
) -> std::result::Result<(), LaminaError> {
    // Use the host target as the default target
    let target = target::Target::detect_host().to_str();
    compile_lamina_ir_to_target_assembly(input_ir, output_asm, &target)
}

/// Parses Lamina IR text and generates assembly code for a specific target architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
/// * `target` - A string slice specifying the target architecture (e.g., "x86_64", "aarch64").
///
/// # Returns
/// * `Result<(),LaminaError>` - Ok if compilation succeeds, Err with error information otherwise.
pub fn compile_lamina_ir_to_target_assembly<W: Write>(
    input_ir: &str,
    output_asm: &mut W,
    target: &str,
) -> std::result::Result<(), LaminaError> {
    // 1. Parse the input string into an IR Module
    let module = parser::parse_module(input_ir)?;

    // 2. Generate assembly for the specified target
    match target {
        "x86_64_unknown" => generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_macos" => generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_linux" => generate_x86_64_assembly(&module, output_asm)?,
        "x86_64_windows" => generate_x86_64_assembly(&module, output_asm)?,
        // FEAT:TODO Add per-target generation refinements for macOS/Linux/Windows
        "aarch64_unknown" => {
            let mir_module = mir::codegen::from_ir(&module, "module")?;
            mir_codegen::generate_mir_to_aarch64(&mir_module, output_asm, target::TargetOperatingSystem::Linux)?;
        }
        "aarch64_macos" => {
            let mir_module = mir::codegen::from_ir(&module, "module")?;
            mir_codegen::generate_mir_to_aarch64(&mir_module, output_asm, target::TargetOperatingSystem::MacOS)?;
        }
        "aarch64_linux" => {
            let mir_module = mir::codegen::from_ir(&module, "module")?;
            mir_codegen::generate_mir_to_aarch64(&mir_module, output_asm, target::TargetOperatingSystem::Linux)?;
        }
        "aarch64_windows" => {
            let mir_module = mir::codegen::from_ir(&module, "module")?;
            mir_codegen::generate_mir_to_aarch64(&mir_module, output_asm, target::TargetOperatingSystem::Windows)?;
        }
        // RISC-V targets
        "riscv32_unknown" => codegen::generate_riscv32_assembly(&module, output_asm)?,
        "riscv64_unknown" => codegen::generate_riscv64_assembly(&module, output_asm)?,
        "riscv128_unknown" => codegen::generate_riscv128_assembly(&module, output_asm)?,
        _ => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                codegen::FeatureType::Custom(format!(
                    "Unsupported target architecture: {}",
                    target
                )),
            )));
        }
    }

    Ok(())
}
