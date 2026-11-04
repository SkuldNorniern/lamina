//! MIR transformation passes for optimization and code generation preparation.
//!
//! Transforms operate on the MIR representation and can be composed into pipelines.
//! Each transform implements the `Transform` trait and can be applied to functions
//! or entire modules.

mod addressing;
mod alias;
#[cfg(test)]
mod alias_tests;
mod branch_opt;
mod cfg;
mod constant_folding;
mod copy_propagation;
mod cse;
mod deadcode;
mod induction_strength_reduction;
mod inline;
#[cfg(test)]
mod inline_asm_tests;
mod loop_opt;
mod loop_rotation;
mod mem2reg;
mod memory;
mod peephole;
mod reduction_exit;
pub mod sanity;
mod scheduling;
mod sroa;
mod strength_reduction;
mod tail_call;
mod unroll;
#[cfg(feature = "nightly")]
mod vectorization;

pub use addressing::AddressingCanonicalization;
pub use branch_opt::BranchOptimization;
pub use cfg::{CfgSimplify, JumpThreading};
pub(crate) use cfg::{Dominators, calculate_dominators, compute_back_edge_headers};
pub use constant_folding::ConstantFolding;
pub use copy_propagation::CopyPropagation;
pub use cse::CommonSubexpressionElimination;
pub use deadcode::{DeadCodeElimination, remove_dead_pure_calls, truncate_after_noreturn_calls};
pub use induction_strength_reduction::InductionStrengthReduction;
pub use inline::{FunctionInlining, ModuleInlining};
pub use loop_opt::LoopInvariantCodeMotion;
pub use loop_rotation::LoopRotation;
pub use mem2reg::Mem2Reg;
pub use memory::MemoryOptimization;
pub use peephole::Peephole;
pub use reduction_exit::ReductionExit;
pub use scheduling::InstructionScheduling;
pub use sroa::ScalarReplacement;
pub use strength_reduction::StrengthReduction;
pub use tail_call::TailCallOptimization;
pub use unroll::LoopUnrolling;

use lamina_platform::{TargetArchitecture, X86CpuLevel};

use crate::mir::{
    Aggregate, Function, MirType, Module, Signature,
    ssa::{SsaFunction, destroy},
    verify_function_in_pipeline,
};
use std::collections::HashMap;
use std::fmt;

/// What a function hands back to its caller: the scalar type (None for void) and
/// whether the value is an aggregate copied through the caller's slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReturnShape {
    pub ty: Option<MirType>,
    pub aggregate: bool,
}

/// One parameter as a caller passes it: its type and, for an aggregate passed by value,
/// the aggregate's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamShape {
    pub ty: MirType,
    pub aggregate: Option<Aggregate>,
}

/// The signature a call has to match, as the callee declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalleeShape {
    pub ret: ReturnShape,
    pub params: Vec<ParamShape>,
    pub variadic: bool,
}

impl CalleeShape {
    pub fn of(sig: &Signature) -> Self {
        Self {
            ret: ReturnShape {
                ty: sig.ret_ty,
                aggregate: sig.ret_aggregate.is_some(),
            },
            params: sig
                .params
                .iter()
                .map(|param| ParamShape {
                    ty: param.ty,
                    aggregate: param.aggregate,
                })
                .collect(),
            variadic: sig.variadic,
        }
    }
}

/// Signature of every function in a module, keyed by name.
pub type CalleeSignatures = HashMap<String, CalleeShape>;

#[cfg(feature = "nightly")]
#[allow(unused_imports)]
pub use vectorization::{AutoVectorization, VectorUnit};

/// Error returned by transform passes.
#[derive(Debug)]
pub enum TransformError {
    WouldDestroyFunction,
    Unsupported(String),
    InvalidState(&'static str),
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
    /// A transform produced MIR that failed verification.
    VerificationFailed {
        errors: Vec<String>,
    },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WouldDestroyFunction => write!(f, "transform would remove all blocks"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
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
            Self::VerificationFailed { errors } => {
                write!(f, "MIR verification failed: {}", errors.join("; "))
            }
        }
    }
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

    /// Apply this transform to a function within a module, given the return types of
    /// the module's functions.
    fn apply_with_callees(
        &self,
        func: &mut Function,
        _callees: &CalleeSignatures,
    ) -> Result<bool, TransformError> {
        self.apply(func)
    }

    /// The SSA entry point of this pass, when it has one. A pass that answers `Some`
    /// runs on the pipeline's graph and never sees mutable MIR.
    fn ssa(&self) -> Option<&dyn SsaTransform> {
        None
    }
}

/// A pass that reads and rewrites the optimizing SSA form the pipeline owns.
pub trait SsaTransform {
    fn apply_ssa(&self, graph: &mut SsaFunction) -> Result<bool, TransformError>;
}

/// The one authoritative form of a function inside the optimizing region. The graph is
/// built the first time an SSA pass asks for it and lowered again the first time a
/// mutable pass does, so construction and destruction belong to the pipeline rather
/// than to any pass.
struct OptimizingForm<'a> {
    func: &'a mut Function,
    graph: Option<SsaFunction>,
    callees: &'a CalleeSignatures,
    unbuildable: bool,
}

impl<'a> OptimizingForm<'a> {
    fn new(func: &'a mut Function, callees: &'a CalleeSignatures) -> Self {
        Self {
            func,
            graph: None,
            callees,
            unbuildable: false,
        }
    }

    fn ssa(&mut self) -> Option<&mut SsaFunction> {
        if self.graph.is_none() && !self.unbuildable {
            let callees = self.callees;
            let returns = |name: &str| callees.get(name).and_then(|shape| shape.ret.ty);
            match SsaFunction::build(self.func, &returns) {
                Ok(graph) => self.graph = Some(graph),
                Err(_) => self.unbuildable = true,
            }
        }
        self.graph.as_mut()
    }

    fn mutable(&mut self) -> Result<&mut Function, TransformError> {
        if let Some(graph) = self.graph.take() {
            *self.func = destroy(&graph).map_err(|e| TransformError::Unsupported(e.to_string()))?;
            self.unbuildable = false;
        }
        Ok(self.func)
    }

    fn verify(&self, pass: &'static str) -> Result<(), TransformError> {
        let fail = |source| TransformError::PassFailed {
            pass,
            source: Box::new(source),
        };
        match &self.graph {
            Some(graph) => graph.verify().map_err(|violations| {
                let errors = violations.iter().map(|v| v.message.clone()).collect();
                fail(TransformError::VerificationFailed { errors })
            }),
            None => verify_function_in_pipeline(self.func)
                .map_err(|errors| fail(TransformError::VerificationFailed { errors })),
        }
    }
}

/// The vector unit this compiler can target for `target`, if any. Only x86-64 is
/// wired up; every other architecture answers `None` and is left untouched.
#[cfg(feature = "nightly")]
pub fn vector_unit_for(target: TargetArchitecture, level: X86CpuLevel) -> Option<VectorUnit> {
    matches!(target, TargetArchitecture::X86_64).then_some(VectorUnit::X86Sse2(level))
}

/// Without the nightly feature there is no vectorizer to configure.
#[cfg(not(feature = "nightly"))]
pub fn vector_unit_for(_target: TargetArchitecture, _level: X86CpuLevel) -> Option<()> {
    None
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

/// Statistics about a transform run
#[derive(Debug, Default)]
pub struct TransformStats {
    /// Number of transforms that were run
    pub transforms_run: usize,
    /// Number of transforms that made changes
    pub transforms_changed: usize,
    /// Total number of iterations performed (for fixed-point transforms)
    pub iterations: usize,
    pub functions_inlined: usize,
}

/// An ordered run of transforms, applied until they stop changing the function or the
/// iteration budget runs out.
pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
    max_total_iterations: usize,
    inline: bool,
    rotate_loops: bool,
    #[cfg(feature = "nightly")]
    vector: Option<VectorUnit>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            max_total_iterations: 1000,
            inline: false,
            rotate_loops: false,
            #[cfg(feature = "nightly")]
            vector: None,
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
    /// | O2    | + splitting of field-accessed stack slots (sroa), promotion of stack slots to registers (mem2reg), constant folding, copy propagation, memory opt, tail-call opt, peephole, module inlining, bounded unrolling |
    /// | O3    | + LICM, strength reduction, CSE, instruction scheduling, loop rotation |
    pub fn default_for_opt_level(opt_level: u8) -> Self {
        let mut pipeline = Self::new();
        pipeline.inline = opt_level >= 2;

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
            pipeline = pipeline.add_transform(ScalarReplacement);
            pipeline = pipeline.add_transform(Mem2Reg);
            pipeline = pipeline.add_transform(ConstantFolding);
            pipeline = pipeline.add_transform(CopyPropagation);
            pipeline = pipeline.add_transform(MemoryOptimization);
            pipeline = pipeline.add_transform(DeadCodeElimination);
            pipeline = pipeline.add_transform(TailCallOptimization);
            pipeline = pipeline.add_transform(Peephole);
        }

        if opt_level >= 3 {
            pipeline = pipeline.add_transform(LoopInvariantCodeMotion);
            pipeline = pipeline.add_transform(InductionStrengthReduction);
            pipeline = pipeline.add_transform(ReductionExit);
            pipeline = pipeline.add_transform(StrengthReduction);
            pipeline = pipeline.add_transform(DeadCodeElimination);
            pipeline = pipeline.add_transform(CommonSubexpressionElimination);
            pipeline = pipeline.add_transform(InstructionScheduling);
        }

        if opt_level >= 2 {
            pipeline = pipeline.add_transform(LoopUnrolling);
        }

        pipeline.rotate_loops = opt_level >= 3;

        pipeline
    }

    /// The standard pipeline, plus vectorization at O3 when the target has a vector
    /// unit this compiler knows how to use. `None` leaves the output identical to
    /// [`Self::default_for_opt_level`].
    #[cfg(feature = "nightly")]
    pub fn default_for_opt_level_on(opt_level: u8, unit: Option<VectorUnit>) -> Self {
        let mut pipeline = Self::default_for_opt_level(opt_level);
        pipeline.vector = unit.filter(|_| opt_level >= 3);
        pipeline
    }

    /// Without the nightly feature there is no vectorizer, so the unit is ignored.
    #[cfg(not(feature = "nightly"))]
    pub fn default_for_opt_level_on(opt_level: u8, _unit: Option<()>) -> Self {
        Self::default_for_opt_level(opt_level)
    }

    /// The pipeline for one target. AArch64 reads an indexed address in the load itself.
    /// Only backends tested with these addresses at optimization levels get them.
    pub fn for_target(opt_level: u8, target: TargetArchitecture, level: X86CpuLevel) -> Self {
        let mut pipeline =
            Self::default_for_opt_level_on(opt_level, vector_unit_for(target, level));
        if opt_level >= 2 && target == TargetArchitecture::Aarch64 {
            pipeline = pipeline
                .add_transform(AddressingCanonicalization)
                .add_transform(DeadCodeElimination);
        }
        pipeline
    }

    /// Runs each pass over one function once, in the order they were added. Repeating
    /// the list until nothing changes costs about half again as much and changed no
    /// output across the testcases, so the order the pipeline is built in is what
    /// decides what a pass sees.
    pub fn apply_to_function(&self, func: &mut Function) -> Result<TransformStats, TransformError> {
        self.apply_to_function_with(func, &CalleeSignatures::new())
    }

    fn apply_to_function_with(
        &self,
        func: &mut Function,
        callees: &CalleeSignatures,
    ) -> Result<TransformStats, TransformError> {
        let mut stats = TransformStats::default();
        let mut total_iterations = 0;
        let mut form = OptimizingForm::new(func, callees);

        for transform in &self.transforms {
            if total_iterations >= self.max_total_iterations {
                return Err(TransformError::PipelineStalled {
                    pass: transform.name(),
                    limit: self.max_total_iterations,
                });
            }

            stats.transforms_run += 1;
            let failed = |e| TransformError::PassFailed {
                pass: transform.name(),
                source: Box::new(e),
            };
            let changed = match transform.ssa() {
                Some(pass) => match form.ssa() {
                    Some(graph) => pass.apply_ssa(graph).map_err(failed)?,
                    None => false,
                },
                None => {
                    let func = form.mutable()?;
                    transform
                        .apply_with_callees(func, callees)
                        .map_err(failed)?
                }
            };
            if changed {
                stats.transforms_changed += 1;
            }

            if cfg!(debug_assertions) {
                form.verify(transform.name())?;
            }

            total_iterations += 1;
        }

        let func = form.mutable()?;
        verify_function_in_pipeline(func)
            .map_err(|errors| TransformError::VerificationFailed { errors })?;
        stats.iterations = total_iterations;
        Ok(stats)
    }

    /// Runs the pipeline over every function, then inlines across them when the
    /// pipeline was built with inlining on.
    pub fn apply_to_module(&self, module: &mut Module) -> Result<TransformStats, TransformError> {
        if !self.transforms.is_empty() {
            truncate_after_noreturn_calls(module);
            remove_dead_pure_calls(module);
        }
        let mut total_stats = self.apply_function_stages(module)?;
        if self.inline {
            total_stats.functions_inlined = ModuleInlining::new().inline_functions(module)?;
        }
        if total_stats.functions_inlined > 0 {
            let stats = self.apply_function_stages(module)?;
            total_stats.transforms_run += stats.transforms_run;
            total_stats.transforms_changed += stats.transforms_changed;
            total_stats.iterations += stats.iterations;
        }
        if self.rotate_loops {
            for func in module.functions.values_mut() {
                LoopRotation.apply(func)?;
            }
            module
                .validate_in_pipeline()
                .map_err(|errors| TransformError::VerificationFailed { errors })?;
        }
        #[cfg(feature = "nightly")]
        if let Some(unit) = self.vector {
            let pass = AutoVectorization::new(unit);
            for func in module.functions.values_mut() {
                if pass.apply(func)? {
                    DeadCodeElimination.apply(func)?;
                }
            }
            module
                .validate_in_pipeline()
                .map_err(|errors| TransformError::VerificationFailed { errors })?;
        }
        Ok(total_stats)
    }

    fn apply_function_stages(&self, module: &mut Module) -> Result<TransformStats, TransformError> {
        let mut total_stats = TransformStats::default();
        let mut failed_functions = Vec::new();
        let callees: CalleeSignatures = module
            .functions
            .iter()
            .map(|(name, func)| (name.clone(), CalleeShape::of(&func.sig)))
            .collect();

        for (func_name, func) in module.functions.iter_mut() {
            match self.apply_to_function_with(func, &callees) {
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

        module
            .validate_in_pipeline()
            .map_err(|errors| TransformError::VerificationFailed { errors })?;
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
    use crate::mir::{
        FunctionBuilder, MirType, ScalarType, VirtualReg,
        instruction::{Immediate, Instruction, IntBinOp, Operand},
        register::Register,
    };
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

    #[test]
    fn oversized_blocks_are_optimized_at_every_level() {
        let mut ir = String::from("fn @main() -> i64 {\nentry:\n  %v0 = add.i64 1, 0\n");
        for i in 1..1600 {
            ir.push_str(&format!("  %v{i} = add.i64 %v{}, {}\n", i - 1, i % 7 + 1));
        }
        ir.push_str("  ret.i64 %v1599\n}\n");
        let parsed = parse_module(&ir).unwrap();
        for level in 0..=3 {
            let mut module = from_ir_x86_64(&parsed, "big").unwrap();
            TransformPipeline::default_for_opt_level(level)
                .apply_to_module(&mut module)
                .unwrap_or_else(|e| panic!("O{level} failed on a 1600-instruction block: {e}"));
        }
    }

    #[test]
    fn module_inlining_runs_only_at_o2_and_reoptimizes_copies() {
        let ir = parse_module(
            "fn @identity(i64 %x) -> i64 {
entry:
ret.i64 %x
}
fn @caller(i64 %x) -> i64 {
entry:
%r = call @identity(%x)
%s = add.i64 %r, 1
ret.i64 %s
}",
        )
        .unwrap();
        for level in 0..=3 {
            let mut module = from_ir_x86_64(&ir, "pipeline").unwrap();
            let stats = TransformPipeline::default_for_opt_level(level)
                .apply_to_module(&mut module)
                .unwrap();
            assert_eq!(stats.functions_inlined, usize::from(level >= 2));
            if level >= 2 {
                let caller = &module.functions["caller"];
                assert!(
                    caller
                        .blocks
                        .iter()
                        .flat_map(|b| &b.instructions)
                        .all(|i| !matches!(
                            &i.instr,
                            Instruction::Copy { .. } | Instruction::Call { .. }
                        ))
                );
                assert!(caller.blocks.iter().flat_map(|b| &b.instructions).any(|i| matches!(&i.instr, Instruction::IntBinary { lhs: Operand::Register(reg), .. } if reg == &caller.sig.params[0].reg)));
            }
        }
    }

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
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
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
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
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
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
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
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
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
        pipeline.apply_to_function(&mut func).unwrap();
        let folded = func.blocks[0].instructions.iter().any(|inst| {
            matches!(
                &inst.instr,
                Instruction::Copy {
                    src: Operand::Immediate(Immediate::I64(8)),
                    ..
                }
            )
        });
        assert!(folded, "{:?}", func.blocks[0].instructions);
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
        pipeline.apply_to_function(&mut func).unwrap();
        let dead = func.blocks[0]
            .instructions
            .iter()
            .any(|inst| matches!(&inst.instr, Instruction::IntBinary { .. }));
        assert!(!dead, "{:?}", func.blocks[0].instructions);
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
