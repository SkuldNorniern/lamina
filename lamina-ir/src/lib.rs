//! `lamina-ir` — Lamina Intermediate Representation.
//!
//! This crate is the canonical home of the Lamina IR: the high-level, SSA-based
//! representation produced by the Lamina front-end before lowering to MIR.
//!
//! ## Layer overview
//!
//! ```text
//! Lamina source
//!      │
//!      ▼  (parse + type-check)
//!   Lamina IR  ◄── this crate
//!      │
//!      ▼  (IR → MIR lowering)
//!   lamina-mir
//!      │
//!      ▼  (code generation)
//!   Target assembly / binary
//! ```
pub mod builder;
pub mod function;
pub mod instruction;
pub mod module;
pub mod owned;
pub mod types;

pub use builder::IRBuilder;
pub use function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
};
pub use instruction::{
    AllocType, BinaryOp, CmpOp, Instruction, assignment_opcode_names, non_assignment_opcode_names,
};
#[cfg(feature = "nightly")]
pub use instruction::{AtomicBinOp, MemoryOrdering, SimdOp};
#[cfg(feature = "nightly")]
pub use module::ModuleAnnotation;
pub use module::{GlobalDeclaration, Module, TypeDeclaration};
pub use owned::{OwnedIRBuilder, OwnedParam, OwnedStructField, OwnedType, OwnedValue};
pub use types::{Identifier, Literal, PrimitiveType, StructField, Type, Value};
