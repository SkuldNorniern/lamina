mod addressing;
mod cfg;
mod deadcode;
mod inline;
mod loop_opt;
mod motion;
mod peephole;
mod strength_reduction;
mod tail_call;
mod memory;

// Re-export transforms for easy access
pub use addressing::AddressingCanonicalization;
pub use cfg::{CfgSimplify, JumpThreading};
pub use deadcode::DeadCodeElimination;
pub use inline::{FunctionInlining, ModuleInlining};
pub use loop_opt::{LoopFusion, LoopInvariantCodeMotion, LoopUnrolling};
pub use motion::{CommonSubexpressionElimination, ConstantFolding, CopyPropagation};
pub use memory::MemoryOptimization;
pub use peephole::Peephole;
pub use strength_reduction::StrengthReduction;
pub use tail_call::TailCallOptimization;

// Each transform operates on the MIR representation to optimize or
// prepare code for code generation. Transforms are composable and
// can be arranged in pipelines.
//
// ## Example Usage
//
// ```rust
// use lamina::mir::{Module, Function, TransformPipeline, Peephole, DeadCodeElimination};
//
// // Parse IR and convert to MIR
// let ir_mod = /* parse your IR */;
// let mut mir_mod = lamina::mir::codegen::from_ir(&ir_mod, "example").unwrap();
//
// // Create and run optimization pipeline
// let pipeline = TransformPipeline::default_for_opt_level(1);
// let stats = pipeline.apply_to_module(&mut mir_mod).unwrap();
//
// println!("Ran {} transforms, {} made changes",
//          stats.transforms_run, stats.transforms_changed);
// ```
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

// Stability level of a transform
//
// This indicates the maturity and reliability of the transform,
// allowing users to choose appropriate optimization levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransformLevel {
    // Transform is deprecated and will be removed
    Deprecated,
    // Transform is experimental and may change or have bugs
    Experimental,
    // Transform is well-tested and production-ready
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
}

impl TransformPipeline {
    /// Create a new empty transform pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
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
            return pipeline; // No transforms at -O0
        }

        // At -O1: only stable, conservative transforms
        if opt_level >= 1 {
            // CFG cleanups
            pipeline = pipeline.add_transform(CfgSimplify);
            pipeline = pipeline.add_transform(JumpThreading);
        }

        // At -O2: add experimental but generally safe optimizations
        if opt_level >= 2 {
            // Memory and addressing canonicalization
            pipeline = pipeline.add_transform(MemoryOptimization);
            pipeline = pipeline.add_transform(AddressingCanonicalization);
            // Loop and call improvements (now with safety limits)
            pipeline = pipeline.add_transform(LoopInvariantCodeMotion);
            pipeline = pipeline.add_transform(TailCallOptimization);
            // Local algebraic rewrites (now safe for IntCmp folding)
            pipeline = pipeline.add_transform(Peephole);
        }

        // At -O3: high-cost transforms
        if opt_level >= 3 {
            // Strength reduction and inlining added at highest level
            pipeline = pipeline.add_transform(StrengthReduction);
            pipeline = pipeline.add_transform(FunctionInlining);
            pipeline = pipeline.add_transform(ConstantFolding);
            pipeline = pipeline.add_transform(CopyPropagation);
            pipeline = pipeline.add_transform(CommonSubexpressionElimination);
            // Potential future:
            // pipeline = pipeline.add_transform(LoopUnrolling::default());
            // pipeline = pipeline.add_transform(LoopFusion::default());
        }

        pipeline
    }

    /// Apply all transforms in the pipeline to a function
    pub fn apply_to_function(
        &self,
        func: &mut crate::mir::Function,
    ) -> Result<TransformStats, String> {
        let mut stats = TransformStats::default();

        for transform in &self.transforms {
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
        }

        Ok(stats)
    }

    /// Apply all transforms in the pipeline to a module (all functions)
    pub fn apply_to_module(
        &self,
        module: &mut crate::mir::Module,
    ) -> Result<TransformStats, String> {
        let mut total_stats = TransformStats::default();

        for func in module.functions.values_mut() {
            let func_stats = self.apply_to_function(func)?;
            total_stats.transforms_run += func_stats.transforms_run;
            total_stats.transforms_changed += func_stats.transforms_changed;
            total_stats.iterations += func_stats.iterations;
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
    fn test_transform_pipeline_default_for_opt_level() {
        // -O0 should have no transforms
        let pipeline = TransformPipeline::default_for_opt_level(0);
        assert!(pipeline.is_empty());

        // -O1 should have peephole and dead code elimination
        let pipeline = TransformPipeline::default_for_opt_level(1);
        assert_eq!(pipeline.len(), 2);

        // -O2 should have peephole, dead code elimination, loop invariant code motion, strength reduction, and tail call optimization
        let pipeline = TransformPipeline::default_for_opt_level(2);
        assert_eq!(pipeline.len(), 6);

        // -O3 should have all transforms including function inlining and motion optimizations
        let pipeline = TransformPipeline::default_for_opt_level(3);
        assert_eq!(pipeline.len(), 13);
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

        // Should not fail
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert_eq!(stats.transforms_run, 1);
    }

    #[test]
    fn test_transform_pipeline_apply_to_module() {
        // Create a simple module
        let module = crate::mir::Module::new("test");
        let mut module = module;
        let pipeline = TransformPipeline::new().add_transform(Peephole);

        // Should not fail
        let stats = pipeline.apply_to_module(&mut module).unwrap();
        assert_eq!(stats.transforms_run, 0); // No functions in module
    }
}

// Categories of transformations
//
// These categorize transforms by their primary purpose, helping with
// organizing optimization passes and understanding what each does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformCategory {
    // Removes code that doesn't affect program output
    DeadCodeElimination,
    // Expands function calls inline for performance
    Inlining,
    // Evaluates constant expressions at compile time
    ConstantFolding,
    // Replaces variable copies with their source values
    CopyPropagation,
    // Chooses optimal machine instructions
    InstructionSelection,
    // Optimizes control flow patterns
    ControlFlowOptimization,
    // Optimizes arithmetic operations
    ArithmeticOptimization,
    // Optimizes memory access patterns
    MemoryOptimization,
}
