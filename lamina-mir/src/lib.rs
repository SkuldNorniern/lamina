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

pub mod asm;
pub mod block;
pub mod function;
pub mod gpu;
pub mod instruction;
pub mod loc;
pub mod module;
pub mod parallel;
pub mod register;
pub mod simd;
pub mod ssa;
pub mod types;
pub mod verify;

// Re-exports for convenience
pub use block::Block;
pub use function::{
    CodePlacement, Function, FunctionBuilder, KernelMetadata, Parameter, Signature,
    SymbolVisibility,
};
pub use instruction::{
    AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction, IntBinOp, IntCmpOp,
    IntOverflowOp, IntRotOp, IntUnOp, MemoryAttrs, Operand, VectorOp,
};
pub use instruction::{
    AsmOperand, AtomicBinOp, MemoryOrdering, index_scale_shift, io_intrinsic_result_type,
};
pub use loc::{Located, SourceLoc};
pub use module::{Global, GlobalRelocation, Module, ModuleBuilder, RelocationModel, TlsModel};
pub use register::{PhysicalReg, Register, RegisterClass, VirtualReg, VirtualRegAllocator};
#[cfg(feature = "nightly")]
pub use simd::{
    extract_vector_type, get_lane_type, is_float_vector, is_integer_vector, lane_count,
    vector_alignment, vector_size_bytes,
};
pub use types::{
    AddressSpace, Aggregate, AggregateClass, AggregateShape, MirType, ScalarType, Signedness,
    VectorLane, VectorType,
};
pub use verify::{verify_function, verify_function_in_pipeline, verify_module_in_pipeline};
