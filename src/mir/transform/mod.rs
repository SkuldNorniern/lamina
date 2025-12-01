//! MIR transformation passes for optimization and code generation preparation.
//!
//! Transforms operate on the MIR representation and can be composed into pipelines.
//! Each transform implements the `Transform` trait and can be applied to functions
//! or entire modules.

mod addressing;
mod branch_opt;
mod cfg;
mod deadcode;
mod inline;
mod loop_opt;
mod memory;
mod motion;
mod peephole;
pub mod sanity;
mod scheduling;
mod strength_reduction;
mod tail_call;

pub use addressing::AddressingCanonicalization;
pub use branch_opt::BranchOptimization;
pub use cfg::{CfgSimplify, JumpThreading};
pub use deadcode::DeadCodeElimination;
pub use inline::{FunctionInlining, ModuleInlining};
pub use loop_opt::{LoopFusion, LoopInvariantCodeMotion, LoopUnrolling};
pub use memory::MemoryOptimization;
pub use motion::{CommonSubexpressionElimination, ConstantFolding, CopyPropagation};
pub use peephole::Peephole;
pub use scheduling::InstructionScheduling;
pub use strength_reduction::StrengthReduction;
pub use tail_call::TailCallOptimization;

/// Trait for MIR transformation passes.
pub trait Transform {
    /// Unique name for this transform
    fn name(&self) -> &'static str;

    /// Description of what this transform does
    fn description(&self) -> &'static str;

    /// Category of this transform
    fn category(&self) -> TransformCategory;

    /// Level of this transform
    fn level(&self) -> TransformLevel;

    /// Apply this transform to a function. Returns true if any changes were made.
    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String>;
}

/// Stability level of a transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransformLevel {
    /// Deprecated transform that will be removed.
    Deprecated,
    /// Experimental transform that may change or have bugs.
    Experimental,
    /// Well-tested and production-ready transform.
    Stable,
}

impl TransformLevel {
    /// Check if this level should be included at the given optimization level
    ///
    /// - `-O0`: No transforms
    /// - `-O1`: `Stable` transforms
    /// - `-O2`: `Stable` + `Experimental` transforms
    /// - `-O3`: All transforms (including `Deprecated` for compatibility)
    pub fn is_enabled_at_opt_level(self, opt_level: u8) -> bool {
        match (self, opt_level) {
            (TransformLevel::Deprecated, 3..) => true, // Only at -O3+
            (TransformLevel::Experimental, 2..) => true, // At -O2+
            (TransformLevel::Stable, 1..) => true,     // At -O1+
            _ => false,
        }
    }
}

/// Statistics about a transform run
#[derive(Debug, Default)]
pub struct TransformStats {
    /// Number of transforms that were run
    pub transforms_run: usize,
    /// Number of transforms that made changes
    pub transforms_changed: usize,
    /// Total number of iterations performed (for fixed-point transforms)
    pub iterations: usize,
}

/// A pipeline of transforms that can be applied to MIR
pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
    /// Safety limit for total iterations across all transforms
    max_total_iterations: usize,
}

impl TransformPipeline {
    /// Create a new empty transform pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            max_total_iterations: 1000, // Safety limit to prevent infinite loops
        }
    }

    /// Add a transform to the pipeline
    pub fn add_transform<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Create a default optimization pipeline for the given optimization level
    pub fn default_for_opt_level(opt_level: u8) -> Self {
        let mut pipeline = Self::new();

        if opt_level == 0 {
            return pipeline;
        }

        if opt_level >= 1 {
            pipeline = pipeline.add_transform(CfgSimplify);
            pipeline = pipeline.add_transform(JumpThreading);
            // BranchOptimization disabled at O1 - too aggressive, can break code
            // pipeline = pipeline.add_transform(BranchOptimization);
        }

        if opt_level >= 2 {
            // O2 transforms disabled - investigating infinite loops in generated code
            // ConstantFolding disabled - may be incorrectly folding loop conditions
            pipeline = pipeline.add_transform(ConstantFolding);
            // AddressingCanonicalization disabled - x86_64 backend doesn't support BaseIndexScale
            // pipeline = pipeline.add_transform(AddressingCanonicalization);
            // MemoryOptimization disabled - investigating hangs
            // pipeline = pipeline.add_transform(MemoryOptimization);
            // DeadCodeElimination disabled - intra-block analysis can incorrectly remove needed code
            // pipeline = pipeline.add_transform(DeadCodeElimination);
            // TailCallOptimization disabled - can create infinite loops if misapplied
            pipeline = pipeline.add_transform(TailCallOptimization);
            // Peephole disabled - comparison optimizations can break loop termination
            pipeline = pipeline.add_transform(Peephole);
            // CopyPropagation disabled - can cause cycles with other transforms
            pipeline = pipeline.add_transform(CopyPropagation);
        }

        if opt_level >= 3 {
            pipeline = pipeline.add_transform(StrengthReduction);
            pipeline = pipeline.add_transform(FunctionInlining);
            // DeadCodeElimination disabled - intra-block analysis is unsafe without inter-block liveness
            // pipeline = pipeline.add_transform(DeadCodeElimination);
            // CSE re-enabled with conservative settings
            pipeline = pipeline.add_transform(CommonSubexpressionElimination);
            // InstructionScheduling re-enabled (currently no-op but safe)
            pipeline = pipeline.add_transform(InstructionScheduling);
            // LoopUnrolling disabled - causing correctness issues
            // pipeline = pipeline.add_transform(LoopUnrolling);
            // LoopFusion still disabled due to complexity
            // pipeline = pipeline.add_transform(LoopFusion);
        }

        pipeline
    }

    /// Apply all transforms in the pipeline to a function
    pub fn apply_to_function(
        &self,
        func: &mut crate::mir::Function,
    ) -> Result<TransformStats, String> {
        // Safety check: prevent transforms on extremely large functions
        let total_instructions: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
        const MAX_INSTRUCTIONS: usize = 100_000;
        if total_instructions > MAX_INSTRUCTIONS {
            return Err(format!(
                "Function too large for transforms ({} instructions, max {})",
                total_instructions, MAX_INSTRUCTIONS
            ));
        }

        let mut stats = TransformStats::default();
        let mut total_iterations = 0;
        
        // Single-pass only - no fixed-point iteration to prevent infinite loops
        // Individual transforms can do their own fixed-point iteration if needed
        for transform in &self.transforms {
            if total_iterations >= self.max_total_iterations {
                return Err(format!(
                    "Transform pipeline exceeded maximum iterations ({}), possible infinite loop in transform '{}'",
                    self.max_total_iterations,
                    transform.name()
                ));
            }

            stats.transforms_run += 1;
            match transform.apply(func) {
                Ok(changed) => {
                    if changed {
                        stats.transforms_changed += 1;
                    }
                }
                Err(e) => {
                    return Err(format!("Transform '{}' failed: {}", transform.name(), e));
                }
            }

            total_iterations += 1;
        }

        stats.iterations = total_iterations;
        Ok(stats)
    }

    /// Apply all transforms in the pipeline to a module (all functions)
    pub fn apply_to_module(
        &self,
        module: &mut crate::mir::Module,
    ) -> Result<TransformStats, String> {
        let mut total_stats = TransformStats::default();
        let mut failed_functions = Vec::new();

        for (func_name, func) in module.functions.iter_mut() {
            match self.apply_to_function(func) {
                Ok(func_stats) => {
                    total_stats.transforms_run += func_stats.transforms_run;
                    total_stats.transforms_changed += func_stats.transforms_changed;
                    total_stats.iterations += func_stats.iterations;
                }
                Err(e) => {
                    // Collect errors but continue processing other functions
                    failed_functions.push((func_name.clone(), e));
                }
            }
        }

        // If all functions failed, return error. Otherwise, log warnings but succeed.
        if !failed_functions.is_empty() && total_stats.transforms_run == 0 {
            return Err(format!(
                "All functions failed transforms: {}",
                failed_functions.iter()
                    .map(|(name, err)| format!("{}: {}", name, err))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Log warnings for failed functions but don't fail the entire pipeline
        if !failed_functions.is_empty() {
            eprintln!("Warning: {} function(s) failed transforms:", failed_functions.len());
            for (name, err) in &failed_functions {
                eprintln!("  {}: {}", name, err);
            }
        }

        Ok(total_stats)
    }

    /// Get the number of transforms in this pipeline
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Function, FunctionBuilder, MirType, ScalarType, VirtualReg};

    #[test]
    fn test_transform_pipeline_empty() {
        let pipeline = TransformPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    #[test]
    fn test_transform_pipeline_add_transform() {
        let pipeline = TransformPipeline::new()
            .add_transform(Peephole)
            .add_transform(DeadCodeElimination);

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_transform_pipeline_apply_to_function() {
        // Create a simple function with some dead code
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::new().add_transform(Peephole);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert_eq!(stats.transforms_run, 1);
    }

    #[test]
    fn test_transform_pipeline_apply_to_module() {
        // Create a simple module
        let module = crate::mir::Module::new("test");
        let mut module = module;
        let pipeline = TransformPipeline::new().add_transform(Peephole);
        let stats = pipeline.apply_to_module(&mut module).unwrap();
        assert_eq!(stats.transforms_run, 0); // No functions in module
    }
}

/// Categories of transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformCategory {
    /// Removes code that doesn't affect program output.
    DeadCodeElimination,
    /// Expands function calls inline for performance.
    Inlining,
    /// Evaluates constant expressions at compile time.
    ConstantFolding,
    /// Replaces variable copies with their source values.
    CopyPropagation,
    /// Chooses optimal machine instructions.
    InstructionSelection,
    /// Optimizes control flow patterns.
    ControlFlowOptimization,
    /// Optimizes arithmetic operations.
    ArithmeticOptimization,
    /// Optimizes memory access patterns.
    MemoryOptimization,
}
