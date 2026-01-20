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
//! use lamina::target::Target;
//! use std::io::Write;
//!
//! // Detect the host architecture
//! let target = Target::detect_host();
//! println!("Host target: {}", target);
//! println!("Architecture: {}", target.architecture);
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
//! compile_lamina_ir_to_assembly(ir_code, &mut assembly, 1)?;
//! println!("Generated assembly:\n{}", String::from_utf8(assembly)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Architecture Support
//!
//! Lamina currently supports the following target architectures:
//!
//! - **x86_64**: Intel/AMD 64-bit processors
//!   - `x86_64_unknown` - Generic x86_64 (uses ELF conventions for compatibility)
//!   - `x86_64_linux` - Linux x86_64
//!   - `x86_64_windows` - Windows x86_64
//!   - `x86_64_macos` - macOS x86_64 (Intel Macs)
//!
//! - **AArch64**: ARM 64-bit processors
//!   - `aarch64_unknown` - Generic AArch64 (uses ELF conventions for compatibility)
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
//!             ty: Type::Primitive(PrimitiveType::I32),
//!             annotations: vec![]
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
//!             ty: Type::Primitive(PrimitiveType::I32),
//!             annotations: vec![]
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
//! ## Nightly Features
//!
//! Lamina includes experimental features that are gated behind the `nightly` feature flag:
//!
//! - **Atomic Operations**: Thread-safe memory operations with memory ordering constraints
//! - **Module Annotations**: Module-level attributes for optimization and debugging hints
//! - **Experimental Targets**: Additional target architectures (e.g., RISC-V 128-bit)
//!
//! To enable nightly features, compile with:
//!
//! ```toml
//! [dependencies]
//! lamina = { version = "0.0.7", features = ["nightly"] }
//! ```
//!
//! **Note**: Nightly features are experimental and may change or be removed in future versions.
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
pub mod target; // Re-exports from lamina-platform for backward compatibility

pub mod error;
pub mod ir;
pub mod mir; // Re-exports from lamina-mir for backward compatibility
pub mod parser;
pub mod runtime;

/// Macro for inline Lamina IR code compilation (similar to `asm!`).
///
/// This macro allows you to write Lamina IR code directly in Rust and have it
/// compiled to native code at runtime. The compiled function can then be called
/// from Rust code.
///
/// # Syntax
///
/// ```rust
/// use lamina::lamina;
///
/// // Basic usage - function name is extracted from the IR
/// let add_fn = lamina!(
///     r#"
///     fn @add(i64 %a, i64 %b) -> i64 {
///         entry:
///             %res = add.i64 %a, %b
///             ret.i64 %res
///     }
///     "#
/// );
///
/// // Call the compiled function (returns Option<i64>)
/// let result = add_fn(&[10, 20]);
/// assert_eq!(result, Some(30));
/// ```
///
/// # Function Name Extraction
///
/// The macro automatically extracts the function name from the IR code.
/// The function must be named with the `@` prefix (e.g., `@add`).
///
/// # Safety
///
/// The returned function pointer is unsafe to call. The caller must ensure:
/// 1. The function signature matches the expected signature
/// 2. Arguments match the function's parameter types
/// 3. The function is only called on the target architecture it was compiled for
///
/// # Limitations
///
/// - Currently only supports functions with i64 parameters and i64 return type
/// - Requires the `encoder` feature to be enabled
/// - Compilation happens at runtime, not compile-time
#[macro_export]
#[cfg(feature = "encoder")]
macro_rules! lamina {
    ($ir_code:literal) => {{
        use $crate::runtime::compile_lir_internal;

        let initialization = (||
            -> Result<
                (
                    $crate::mir::Signature,
                    *const u8,
                    *mut $crate::runtime::ExecutableMemory,
                    usize,
                ),
                String,
            > {
            let raw_name = $crate::runtime::macro_helpers::extract_function_name($ir_code)
                .ok_or_else(|| {
                    "lamina!: Could not extract function name from IR code. Function must be named with @ prefix (e.g., @add)"
                        .to_string()
                })?;

            let function_name: &'static str = if raw_name.starts_with('@') {
                raw_name
            } else {
                let owned_name = format!("@{}", raw_name);
                Box::leak(owned_name.into_boxed_str())
            };

            let runtime_result = compile_lir_internal($ir_code, function_name, 1)
                .map_err(|e| format!("lamina!: Failed to compile IR code: {}", e))?;

            let memory_handle = Box::leak(Box::new(runtime_result.memory));
            let function_ptr = runtime_result.function_ptr;

            let ir_module = $crate::parser::parse_module($ir_code)
                .map_err(|e| format!("lamina!: Failed to parse IR: {}", e))?;
            let mir_module = $crate::mir::codegen::from_ir(&ir_module, "lamina_macro")
                .map_err(|e| format!("lamina!: Failed to lower to MIR: {}", e))?;

            let func = mir_module
                .functions
                .get(function_name)
                .or_else(|| {
                    if function_name.starts_with('@') {
                        mir_module.functions.get(&function_name[1..])
                    } else {
                        let name_with_at = format!("@{}", function_name);
                        mir_module.functions.get(&name_with_at)
                    }
                })
                .or_else(|| {
                    mir_module.functions.values().find(|f| {
                        f.sig.name == function_name
                            || f.sig.name == format!("@{}", function_name)
                            || (function_name.starts_with('@') && f.sig.name == &function_name[1..])
                    })
                })
                .ok_or_else(|| {
                    let available = mir_module.function_names().join(", ");
                    format!(
                        "lamina!: Function '{}' not found in IR code. Available functions: [{}]",
                        function_name, available
                    )
                })?;

            let param_count = func.sig.params.len();
            let signature = func.sig.clone();

            Ok((signature, function_ptr, memory_handle, param_count))
        })();

        let jit_handle = match initialization {
            Ok(handle) => Some(handle),
            Err(error_message) => {
                eprintln!("{}", error_message);
                None
            }
        };

        move |args: &[i64]| -> Option<i64> {
            let (signature, function_ptr, memory_handle, param_count) = jit_handle.as_ref()?;
            let _memory_ref = memory_handle;
            if args.len() != *param_count {
                return None;
            }
            match $crate::runtime::execute_jit_function(
                signature,
                *function_ptr,
                Some(args),
                false,
            ) {
                Ok(result) => result,
                Err(_) => None,
            }
        }
    }};
}

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
pub use mir_codegen::generate_mir_to_target;

/// Parses Lamina IR text and generates assembly code using the host system's architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
/// * `codegen_units` - Number of parallel compilation threads (default: 1).
pub fn compile_lamina_ir_to_assembly<W: Write>(
    input_ir: &str,
    output_asm: &mut W,
    codegen_units: usize,
) -> std::result::Result<(), LaminaError> {
    let target = lamina_platform::Target::detect_host().to_str();
    compile_lamina_ir_to_target_assembly(input_ir, output_asm, &target, codegen_units)
}

/// Parses Lamina IR text and generates assembly code for a specific target architecture.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
/// * `target` - A string slice specifying the target architecture (e.g., "x86_64", "aarch64").
/// * `codegen_units` - Number of parallel compilation threads.
///
/// # Returns
/// * `Result<(),LaminaError>` - Ok if compilation succeeds, Err with error information otherwise.
pub fn compile_lamina_ir_to_target_assembly<W: Write>(
    input_ir: &str,
    output_asm: &mut W,
    target: &str,
    codegen_units: usize,
) -> std::result::Result<(), LaminaError> {
    let module = parser::parse_module(input_ir)?;

    use std::str::FromStr;
    let target_obj = lamina_platform::Target::from_str(target)
        .map_err(|e| LaminaError::ValidationError(format!("Invalid target '{}': {}", target, e)))?;
    let mir_module = mir::codegen::from_ir(&module, "module")?;

    match target_obj.architecture {
        lamina_platform::TargetArchitecture::X86_64
        | lamina_platform::TargetArchitecture::Aarch64
        | lamina_platform::TargetArchitecture::Wasm32
        | lamina_platform::TargetArchitecture::Wasm64
        | lamina_platform::TargetArchitecture::Riscv32
        | lamina_platform::TargetArchitecture::Riscv64 => {
            mir_codegen::generate_mir_to_target(
                &mir_module,
                output_asm,
                target_obj.architecture,
                target_obj.operating_system,
                codegen_units,
            )?;
        }
        #[cfg(feature = "nightly")]
        lamina_platform::TargetArchitecture::Riscv128 => {
            mir_codegen::generate_mir_to_target(
                &mir_module,
                output_asm,
                target_obj.architecture,
                target_obj.operating_system,
                codegen_units,
            )?;
        }
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
