//! MIR transformation passes for optimization and code generation preparation.
//!
//! Transforms operate on the MIR representation and can be composed into pipelines.
//! Each transform implements the `Transform` trait and can be applied to functions
//! or entire modules.

mod addressing;
mod branch_opt;
mod cfg;
mod constant_folding;
mod copy_propagation;
mod cse;
mod deadcode;
mod inline;
mod loop_opt;
mod memory;
mod peephole;
pub mod sanity;
mod scheduling;
mod strength_reduction;
mod tail_call;
#[cfg(feature = "nightly")]
mod vectorization;

pub use addressing::AddressingCanonicalization;
pub use branch_opt::BranchOptimization;
pub use cfg::{CfgSimplify, JumpThreading};
pub use constant_folding::ConstantFolding;
pub use copy_propagation::CopyPropagation;
pub use cse::CommonSubexpressionElimination;
pub use deadcode::DeadCodeElimination;
pub use inline::{FunctionInlining, ModuleInlining};
pub use loop_opt::{LoopInvariantCodeMotion, LoopUnrolling};
pub use memory::MemoryOptimization;
pub use peephole::Peephole;
pub use scheduling::InstructionScheduling;
pub use strength_reduction::StrengthReduction;
pub use tail_call::TailCallOptimization;
#[cfg(feature = "nightly")]
pub use vectorization::AutoVectorization;

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

    pub fn transform_names(&self) -> Vec<&'static str> {
        self.transforms
            .iter()
            .map(|transform| transform.name())
            .collect()
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
            // NOTE: BranchOptimization disabled - causes infinite loop at O3
            // pipeline = pipeline.add_transform(BranchOptimization);
        }

        if opt_level >= 2 {
            pipeline = pipeline.add_transform(ConstantFolding);
            // NOTE: CopyPropagation disabled - causes infinite loop at O3
            // pipeline = pipeline.add_transform(CopyPropagation);
            pipeline = pipeline.add_transform(MemoryOptimization);
            pipeline = pipeline.add_transform(DeadCodeElimination);
            pipeline = pipeline.add_transform(TailCallOptimization);
            pipeline = pipeline.add_transform(Peephole);
        }

        if opt_level >= 3 {
            pipeline = pipeline.add_transform(StrengthReduction);
            pipeline = pipeline.add_transform(DeadCodeElimination);
            pipeline = pipeline.add_transform(CommonSubexpressionElimination);
            pipeline = pipeline.add_transform(InstructionScheduling);

            #[cfg(feature = "nightly")]
            {
                pipeline = pipeline.add_transform(AutoVectorization);
            }
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
                failed_functions
                    .iter()
                    .map(|(name, err)| format!("{}: {}", name, err))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Log warnings for failed functions but don't fail the entire pipeline
        if !failed_functions.is_empty() {
            eprintln!(
                "Warning: {} function(s) failed transforms:",
                failed_functions.len()
            );
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, MirType, ScalarType, VirtualReg};

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

    #[test]
    fn test_phase_ordering_o1_passes() {
        // Test that O1 passes work together
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::default_for_opt_level(1);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert!(stats.transforms_run > 0);
    }

    #[test]
    fn test_phase_ordering_o2_passes() {
        // Test that O2 passes work together
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::default_for_opt_level(2);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert!(stats.transforms_run > 0);
    }

    #[test]
    fn test_phase_ordering_o3_passes() {
        // Test that O3 passes work together
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::default_for_opt_level(3);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert!(stats.transforms_run > 0);
    }

    #[test]
    fn test_phase_ordering_cfg_then_constant_folding() {
        // Test that CFG simplification before constant folding works
        use crate::mir::instruction::{Instruction, IntBinOp, Operand};
        use crate::mir::register::Register;

        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(1)),
                lhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(5)),
                rhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(3)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(1)))),
            })
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::new()
            .add_transform(CfgSimplify)
            .add_transform(ConstantFolding);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert!(stats.transforms_run == 2);
    }

    #[test]
    fn test_phase_ordering_dce_after_optimizations() {
        // Test that DCE after other optimizations works correctly
        use crate::mir::instruction::{Instruction, IntBinOp, Operand};
        use crate::mir::register::Register;

        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(1)),
                lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(0))),
                rhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(0)))),
            })
            .build();

        let mut func = func;
        let pipeline = TransformPipeline::new()
            .add_transform(Peephole)
            .add_transform(DeadCodeElimination);
        let stats = pipeline.apply_to_function(&mut func).unwrap();
        assert!(stats.transforms_run == 2);
    }

    #[test]
    fn test_regression_branch_optimization_disabled() {
        // Regression test: BranchOptimization is disabled due to infinite loop bug
        let pipeline_o1 = TransformPipeline::default_for_opt_level(1);
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        let names_o1: Vec<&str> = pipeline_o1.transform_names();
        let names_o2: Vec<&str> = pipeline_o2.transform_names();
        let names_o3: Vec<&str> = pipeline_o3.transform_names();

        // BranchOptimization should NOT be in any default pipeline until bug is fixed
        assert!(!names_o1.contains(&"branch_optimization"));
        assert!(!names_o2.contains(&"branch_optimization"));
        assert!(!names_o3.contains(&"branch_optimization"));
    }

    #[test]
    fn test_regression_copy_propagation_disabled() {
        // Regression test: CopyPropagation is disabled due to infinite loop bug
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        let names_o2: Vec<&str> = pipeline_o2.transform_names();
        let names_o3: Vec<&str> = pipeline_o3.transform_names();

        // CopyPropagation should NOT be in default pipelines until bug is fixed
        assert!(!names_o2.contains(&"copy_propagation"));
        assert!(!names_o3.contains(&"copy_propagation"));
    }

    #[test]
    fn test_regression_function_inlining_disabled() {
        // Regression test: FunctionInlining is disabled, verify it's not in default pipeline
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);
        let names_o3: Vec<&str> = pipeline_o3.transform_names();

        // FunctionInlining should not be in default O3 pipeline
        assert!(!names_o3.contains(&"function_inlining"));
    }

    #[test]
    fn test_regression_addressing_canonicalization_disabled() {
        // Regression test: AddressingCanonicalization is disabled
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        let names_o2: Vec<&str> = pipeline_o2.transform_names();
        let names_o3: Vec<&str> = pipeline_o3.transform_names();

        // AddressingCanonicalization should not be in default pipelines
        assert!(!names_o2.contains(&"addressing_canonicalization"));
        assert!(!names_o3.contains(&"addressing_canonicalization"));
    }
}

/// Categories of transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformCategory {
    /// Removes code that doesn't affect program output.
    DeadCodeElimination,
    /// Expands function calls inline.
    Inlining,
    /// Evaluates constant expressions at compile time.
    ConstantFolding,
    /// Replaces variable copies with their source values.
    CopyPropagation,
    /// Chooses machine instructions.
    InstructionSelection,
    /// Optimizes control flow.
    ControlFlowOptimization,
    /// Optimizes arithmetic.
    ArithmeticOptimization,
    /// Optimizes memory access.
    MemoryOptimization,
    /// Vectorizes scalar operations using SIMD.
    Vectorization,
}
