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
pub(crate) use cfg::{calculate_dominators, compute_back_edge_headers};
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

use crate::mir::{Function, Module};
use std::fmt;

#[cfg(feature = "nightly")]
pub use vectorization::AutoVectorization;

/// Error returned by transform passes.
#[derive(Debug)]
pub enum TransformError {
    FunctionTooLarge {
        pass: &'static str,
        count: usize,
        limit: usize,
    },
    BlockTooLarge {
        label: String,
        count: usize,
        limit: usize,
    },
    WouldDestroyFunction,
    ConvergenceFailed,
    Unsupported(String),
    InvalidState(&'static str),
    /// Function exceeded the pipeline-wide instruction budget.
    PipelineFunctionTooLarge {
        count: usize,
        limit: usize,
    },
    /// Pipeline ran too many iterations; a pass likely fails to converge.
    PipelineStalled {
        pass: &'static str,
        limit: usize,
    },
    /// A named pass returned an error.
    PassFailed {
        pass: &'static str,
        source: Box<TransformError>,
    },
    /// Every function in the module failed to transform.
    AllFunctionsFailed {
        failures: Vec<(String, TransformError)>,
    },
    /// A control-flow edge targets a block that does not exist.
    InvalidCfg {
        block: String,
        target: String,
        edge: &'static str,
    },
    /// Module exceeded the inliner's instruction budget.
    InliningModuleTooLarge {
        count: usize,
        limit: usize,
    },
    /// Inlining did not reach a fixed point within the iteration limit.
    InliningStalled {
        limit: usize,
    },
    /// Call site argument count did not match the callee signature.
    ParamCountMismatch {
        expected: usize,
        got: usize,
    },
    /// A referenced function or block was not present in the module.
    NotFound {
        kind: &'static str,
        name: String,
    },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionTooLarge { pass, count, limit } => {
                write!(
                    f,
                    "Function too large for {pass} ({count} blocks, max {limit})"
                )
            }
            Self::BlockTooLarge {
                label,
                count,
                limit,
            } => {
                write!(
                    f,
                    "Block '{label}' too large ({count} instructions, max {limit})"
                )
            }
            Self::WouldDestroyFunction => write!(f, "transform would remove all blocks"),
            Self::ConvergenceFailed => write!(f, "liveness analysis failed to converge"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::PipelineFunctionTooLarge { count, limit } => {
                write!(
                    f,
                    "function too large for transforms ({count} instructions, max {limit})"
                )
            }
            Self::PipelineStalled { pass, limit } => {
                write!(
                    f,
                    "transform pipeline exceeded {limit} iterations, possible infinite loop in pass '{pass}'"
                )
            }
            Self::PassFailed { pass, source } => write!(f, "transform '{pass}' failed: {source}"),
            Self::AllFunctionsFailed { failures } => {
                write!(f, "all functions failed transforms: ")?;
                for (i, (name, err)) in failures.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {err}")?;
                }
                Ok(())
            }
            Self::InvalidCfg {
                block,
                target,
                edge,
            } => {
                write!(
                    f,
                    "invalid CFG: block '{block}' {edge} targets missing block '{target}'"
                )
            }
            Self::InliningModuleTooLarge { count, limit } => {
                write!(
                    f,
                    "module too large for inlining ({count} instructions, max {limit})"
                )
            }
            Self::InliningStalled { limit } => {
                write!(f, "inlining did not converge after {limit} iterations")
            }
            Self::ParamCountMismatch { expected, got } => {
                write!(
                    f,
                    "parameter count mismatch: expected {expected}, got {got}"
                )
            }
            Self::NotFound { kind, name } => write!(f, "{kind} '{name}' not found"),
        }
    }
}

/// Reject functions too large for a pass to process in reasonable time.
///
/// Pass `usize::MAX` for `max_instructions_per_block` to skip the per-block check.
pub(crate) fn check_function_size(
    func: &Function,
    pass: &'static str,
    max_blocks: usize,
    max_instructions_per_block: usize,
) -> Result<(), TransformError> {
    if func.blocks.len() > max_blocks {
        return Err(TransformError::FunctionTooLarge {
            pass,
            count: func.blocks.len(),
            limit: max_blocks,
        });
    }
    for block in &func.blocks {
        if block.instructions.len() > max_instructions_per_block {
            return Err(TransformError::BlockTooLarge {
                label: block.label.clone(),
                count: block.instructions.len(),
                limit: max_instructions_per_block,
            });
        }
    }
    Ok(())
}

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
    fn apply(&self, func: &mut Function) -> Result<bool, TransformError>;
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
        matches!(
            (self, opt_level),
            (TransformLevel::Deprecated, 3..)
                | (TransformLevel::Experimental, 2..)
                | (TransformLevel::Stable, 1..)
        )
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

pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
    max_total_iterations: usize,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            max_total_iterations: 1000,
        }
    }

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

    /// Build the standard optimization pipeline for the given level.
    ///
    /// | Level | Transforms added |
    /// |-------|-----------------|
    /// | O0    | (none) |
    /// | O1    | CFG simplify, jump threading, branch opt, DCE |
    /// | O2    | + constant folding, copy propagation, memory opt, tail-call opt, peephole |
    /// | O3    | + LICM, strength reduction, CSE, instruction scheduling |
    pub fn default_for_opt_level(opt_level: u8) -> Self {
        let mut pipeline = Self::new();

        if opt_level == 0 {
            return pipeline;
        }

        if opt_level >= 1 {
            pipeline = pipeline.add_transform(CfgSimplify);
            pipeline = pipeline.add_transform(JumpThreading);
            pipeline = pipeline.add_transform(BranchOptimization);
            pipeline = pipeline.add_transform(DeadCodeElimination);
        }

        if opt_level >= 2 {
            pipeline = pipeline.add_transform(ConstantFolding);
            pipeline = pipeline.add_transform(CopyPropagation);
            pipeline = pipeline.add_transform(MemoryOptimization);
            pipeline = pipeline.add_transform(DeadCodeElimination);
            pipeline = pipeline.add_transform(TailCallOptimization);
            pipeline = pipeline.add_transform(Peephole);
        }

        if opt_level >= 3 {
            pipeline = pipeline.add_transform(LoopInvariantCodeMotion);
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

    pub fn apply_to_function(&self, func: &mut Function) -> Result<TransformStats, TransformError> {
        let total_instructions: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
        const MAX_INSTRUCTIONS: usize = 100_000;
        if total_instructions > MAX_INSTRUCTIONS {
            return Err(TransformError::PipelineFunctionTooLarge {
                count: total_instructions,
                limit: MAX_INSTRUCTIONS,
            });
        }

        let mut stats = TransformStats::default();
        let mut total_iterations = 0;

        for transform in &self.transforms {
            if total_iterations >= self.max_total_iterations {
                return Err(TransformError::PipelineStalled {
                    pass: transform.name(),
                    limit: self.max_total_iterations,
                });
            }

            stats.transforms_run += 1;
            let changed = transform
                .apply(func)
                .map_err(|e| TransformError::PassFailed {
                    pass: transform.name(),
                    source: Box::new(e),
                })?;
            if changed {
                stats.transforms_changed += 1;
            }

            total_iterations += 1;
        }

        stats.iterations = total_iterations;
        Ok(stats)
    }

    pub fn apply_to_module(&self, module: &mut Module) -> Result<TransformStats, TransformError> {
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
                    failed_functions.push((func_name.clone(), e));
                }
            }
        }

        if !failed_functions.is_empty() && total_stats.transforms_run == 0 {
            return Err(TransformError::AllFunctionsFailed {
                failures: failed_functions,
            });
        }

        if !failed_functions.is_empty() {
            eprintln!(
                "Warning: {} function(s) failed transforms:",
                failed_functions.len()
            );
            for (name, err) in &failed_functions {
                eprintln!("  {name}: {err}");
            }
        }

        Ok(total_stats)
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

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
pub(crate) mod test_utils {
    use crate::mir::{Block, Function};

    /// Retrieves a block by label, panicking with the label name if not found.
    // SAFETY: test helper — label must exist in the function's block list.
    pub fn get_block<'a>(func: &'a Function, label: &str) -> &'a Block {
        func.get_block(label)
            .unwrap_or_else(|| panic!("block '{label}' not found in function"))
    }

    /// Applies a transform pass, panicking on error, and returns whether anything changed.
    // SAFETY: test helper — pass must be valid for the function's IR.
    pub fn apply_pass(pass: &impl super::Transform, func: &mut Function) -> bool {
        pass.apply(func)
            .unwrap_or_else(|e| panic!("transform failed: {e}"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::instruction::{Immediate, Instruction, IntBinOp, Operand};
    use crate::mir::register::Register;
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
        let mut module = Module::new("test");
        let pipeline = TransformPipeline::new().add_transform(Peephole);
        let stats = pipeline.apply_to_module(&mut module).unwrap();
        assert_eq!(stats.transforms_run, 0);
    }

    #[test]
    fn test_phase_ordering_o1_passes() {
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
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(1)),
                lhs: Operand::Immediate(Immediate::I64(5)),
                rhs: Operand::Immediate(Immediate::I64(3)),
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
        assert_eq!(stats.transforms_run, 2);
    }

    #[test]
    fn test_phase_ordering_dce_after_optimizations() {
        let func = FunctionBuilder::new("test")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(VirtualReg::gpr(1)),
                lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(0))),
                rhs: Operand::Immediate(Immediate::I64(0)),
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
        assert_eq!(stats.transforms_run, 2);
    }

    #[test]
    fn test_branch_optimization_enabled() {
        let pipeline_o1 = TransformPipeline::default_for_opt_level(1);
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        assert!(
            pipeline_o1
                .transform_names()
                .contains(&"branch_optimization")
        );
        assert!(
            pipeline_o2
                .transform_names()
                .contains(&"branch_optimization")
        );
        assert!(
            pipeline_o3
                .transform_names()
                .contains(&"branch_optimization")
        );
    }

    #[test]
    fn test_copy_propagation_enabled() {
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        assert!(pipeline_o2.transform_names().contains(&"copy_propagation"));
        assert!(pipeline_o3.transform_names().contains(&"copy_propagation"));
    }

    #[test]
    fn test_regression_function_inlining_disabled() {
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);
        assert!(!pipeline_o3.transform_names().contains(&"function_inlining"));
    }

    #[test]
    fn test_regression_addressing_canonicalization_disabled() {
        let pipeline_o2 = TransformPipeline::default_for_opt_level(2);
        let pipeline_o3 = TransformPipeline::default_for_opt_level(3);

        assert!(
            !pipeline_o2
                .transform_names()
                .contains(&"addressing_canonicalization")
        );
        assert!(
            !pipeline_o3
                .transform_names()
                .contains(&"addressing_canonicalization")
        );
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
