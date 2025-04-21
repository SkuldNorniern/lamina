#![allow(dead_code)] // Allow unused code for now

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

/// Parses Lamina IR text and generates assembly code.
///
/// # Arguments
/// * `input_ir` - A string slice containing the Lamina IR code.
/// * `output_asm` - A mutable writer where the generated assembly will be written.
pub fn compile_lamina_ir_to_assembly<W: Write>(input_ir: &str, output_asm: &mut W) -> Result<()> {
    // 1. Parse the input string into an IR Module
    let module = parser::parse_module(input_ir)?;

    // 2. Generate assembly from the IR Module
    codegen::generate_x86_64_assembly(&module, output_asm)?;

    // Keep this minimal for library code
    // println!("Lamina IR compilation steps outlined (parsing and codegen placeholders called).");
    Ok(())
}

