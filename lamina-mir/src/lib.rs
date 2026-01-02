//! lamina-mir - MIR (Machine Intermediate Representation) type definitions
//!
//! This crate provides the core type definitions for LUMIR (Lamina Unified Machine
//! Intermediate Representation), including modules, functions, instructions,
//! registers, and types.
//!
//! ## Architecture
//!
//! LUMIR is a low-level, machine-friendly layer produced after IR Processing.
//! It is assembly-like, easy to apply optimizations, and straightforward to lower
//! into target assembly.
//!
//! ## Usage
//!
//! ```rust
//! use lamina_mir::{Module, Function, Instruction, Register, MirType};
//! ```

pub mod block;
pub mod function;
pub mod instruction;
pub mod module;
pub mod register;
pub mod simd;
pub mod types;

// Re-exports for convenience
pub use block::Block;
pub use function::{Function, FunctionBuilder, Parameter, Signature};
pub use instruction::{
    AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction, IntBinOp, IntCmpOp,
    MemoryAttrs, Operand, VectorOp,
};
#[cfg(feature = "nightly")]
pub use instruction::{AtomicBinOp, MemoryOrdering, SimdOp};
pub use module::{Global, Module, ModuleBuilder};
pub use register::{PhysicalReg, Register, RegisterClass, VirtualReg, VirtualRegAllocator};
pub use types::{MirType, ScalarType, VectorLane, VectorType};
