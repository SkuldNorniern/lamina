//! MIR (Machine Intermediate Representation) module
//!
//! Re-exports MIR types from `lamina-mir` for backward compatibility.
//! It also includes lamina-specific modules like codegen (IR → MIR conversion)
//! and transform (optimizations).

// Re-export all MIR types from lamina-mir
pub use lamina_mir::*;

// Keep lamina-specific modules
pub mod codegen; // IR → MIR conversion (lamina-specific)
pub mod transform; // Optimizations (lamina-specific)

// Re-export transform types for convenience
pub use transform::{
    DeadCodeElimination, FunctionInlining, LoopInvariantCodeMotion, LoopUnrolling, ModuleInlining,
    Peephole, Transform, TransformPipeline, TransformStats,
};
