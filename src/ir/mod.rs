//! # Lamina IR
//!
//! The Lamina IR is a low-level intermediate representation used by the Lamina compiler.
//! It is used to represent the program in a way that is easy to compile to machine code.
//!
//! ## Modules
//!
//! - `builder`: A builder for constructing IR modules.
//!
//! - `function`: A module for representing functions.
//! - `instruction`: A module for representing instructions.
//! - `module`: A module for representing modules.
//! - `types`: A module for representing types.
//!
//! ## Structures
//!
//! - `BasicBlock`: A basic block in a function.
//! - `Function`: A function in the IR.
//! - `FunctionAnnotation`: An annotation for a function.
//! - `FunctionParameter`: A parameter for a function.
//! - `FunctionSignature`: A signature for a function.
//! - `Instruction`: An instruction in a basic block.
//! - `Module`: A module in the IR.
//! - `TypeDeclaration`: A type declaration in the IR.
//! - `Identifier`: An identifier in the IR.
//! - `Literal`: A literal in the IR.
//! - `PrimitiveType`: A primitive type in the IR.
//! - `Type`: A type in the IR.
//! - `Value`: A value in the IR.
pub mod builder;
pub mod function;
pub mod instruction;
pub mod module;
pub mod types;

// Re-export the IR structures
pub use builder::IRBuilder;
pub use function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
};
pub use instruction::{AllocType, BinaryOp, CmpOp, Instruction};
pub use module::{GlobalDeclaration, Module, TypeDeclaration};
pub use types::{Identifier, Literal, PrimitiveType, StructField, Type, Value};
