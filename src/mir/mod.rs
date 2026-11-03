//! MIR (Machine Intermediate Representation) module
//!
//! Re-exports MIR types from `lamina-mir`, by name rather than by glob, so that an
//! item added to `lamina-mir` does not silently join this crate's public API.
//! It also includes lamina-specific modules like codegen (IR → MIR conversion)
//! and transform (optimizations).

pub use lamina_mir::{
    AddressMode, AddressSpace, Aggregate, AggregateClass, AggregateShape, AsmOperand, AtomicBinOp,
    Block, CodePlacement, FloatBinOp, FloatCmpOp, FloatUnOp, Function, FunctionBuilder, Global,
    GlobalRelocation, Immediate, Instruction, IntBinOp, IntCmpOp, IntOverflowOp, IntRotOp, IntUnOp,
    KernelMetadata, Located, MemoryAttrs, MemoryOrdering, MirType, Module, ModuleBuilder, Operand,
    Parameter, PhysicalReg, Register, RegisterClass, RelocationModel, ScalarType, Signature,
    Signedness, SourceLoc, SymbolVisibility, TlsModel, VectorLane, VectorOp, VectorType,
    VirtualReg, VirtualRegAllocator, index_scale_shift, io_intrinsic_result_type, verify_function,
    verify_function_in_pipeline, verify_module_in_pipeline,
};
pub use lamina_mir::{
    asm, block, function, gpu, instruction, loc, module, parallel, register, simd, ssa, types,
    verify,
};
#[cfg(feature = "nightly")]
pub use lamina_mir::{
    extract_vector_type, get_lane_type, is_float_vector, is_integer_vector, lane_count,
    vector_alignment, vector_size_bytes,
};

// Keep lamina-specific modules
pub mod codegen; // IR → MIR conversion (lamina-specific)
pub mod transform; // Optimizations (lamina-specific)

#[cfg(test)]
mod ssa_corpus_tests;

// Re-export transform types for convenience
pub use transform::{
    DeadCodeElimination, FunctionInlining, LoopInvariantCodeMotion, LoopUnrolling, ModuleInlining,
    Peephole, Transform, TransformError, TransformPipeline, TransformStats,
};
