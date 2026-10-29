//! Loop optimization transforms for MIR.

#[cfg(test)]
use crate::mir::{Block, Register};
use crate::mir::{
    Function, Instruction, MirType,
    ssa::{SsaFunction, destroy},
    transform::{
        Dominators, SsaTransform, Transform, TransformCategory, TransformError, TransformLevel,
        calculate_dominators,
    },
};
#[cfg(test)]
use lamina_mir::Located;
use lamina_platform::PointerWidth;
use std::collections::{HashMap, HashSet, hash_map::Entry};

/// Loop invariant code motion that moves loop-invariant computations outside loops.
#[derive(Default)]
pub struct LoopInvariantCodeMotion;

impl Transform for LoopInvariantCodeMotion {
    fn name(&self) -> &'static str {
        "loop_invariant_code_motion"
    }

    fn description(&self) -> &'static str {
        "Moves loop-invariant computations outside loops"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let no_callee = |_: &str| -> Option<MirType> { None };
        let Ok(mut graph) = SsaFunction::build(func, &no_callee) else {
            return Ok(false);
        };
        if !self.apply_ssa(&mut graph)? {
            return Ok(false);
        }
        *func = destroy(&graph).map_err(|e| TransformError::Unsupported(e.to_string()))?;
        Ok(true)
    }

    fn ssa(&self) -> Option<&dyn SsaTransform> {
        Some(self)
    }
}

impl SsaTransform for LoopInvariantCodeMotion {
    fn apply_ssa(&self, graph: &mut SsaFunction) -> Result<bool, TransformError> {
        ssa_licm::run(self, graph)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests_licm {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn loop_discovery_follows_block_order() {
        let mut function = FunctionBuilder::new("nested")
            .block("entry")
            .instr(Instruction::Jmp {
                target: "outer".into(),
            })
            .block("outer")
            .instr(Instruction::Jmp {
                target: "inner".into(),
            })
            .block("inner")
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "inner".into(),
                false_target: "latch".into(),
            })
            .block("latch")
            .instr(Instruction::Jmp {
                target: "outer".into(),
            })
            .build();
        for _ in 0..32 {
            let headers: Vec<_> = LoopInvariantCodeMotion
                .find_loops(&function)
                .into_iter()
                .map(|info| info.header)
                .collect();
            assert_eq!(headers, ["outer", "inner"]);
        }
        function.blocks.swap(1, 2);
        let headers: Vec<_> = LoopInvariantCodeMotion
            .find_loops(&function)
            .into_iter()
            .map(|info| info.header)
            .collect();
        assert_eq!(headers, ["inner", "outer"]);
    }

    #[test]
    fn test_licm_basic() {
        // Create a loop with invariant calculation
        // entry:
        //   jmp loop
        // loop:
        //   v0 = add v0, 1  (induction)
        //   v1 = add 5, 10  (invariant)
        //   v2 = lt v0, v1
        //   br v2, loop, exit
        // exit:
        //   ret

        let mut func = FunctionBuilder::new("test_licm")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            }) // init v0
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            // Invariant: 5 + 10
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(5)),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(1).into()),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("LICM failed");

        assert!(changed);

        // Loop block should no longer have the 5+10 addition
        let loop_block = func.get_block("loop").expect("loop block exists");
        let has_invariant = loop_block.instructions.iter().any(|i| {
            if let Instruction::IntBinary { lhs, rhs, .. } = &i.instr {
                matches!(lhs, Operand::Immediate(Immediate::I64(5)))
                    && matches!(rhs, Operand::Immediate(Immediate::I64(10)))
            } else {
                false
            }
        });
        assert!(
            !has_invariant,
            "Invariant instruction should be moved out of loop"
        );

        let pre_header = func.get_block("entry").expect("pre-header");
        let has_moved = pre_header.instructions.iter().any(|i| {
            if let Instruction::IntBinary { lhs, rhs, .. } = &i.instr {
                matches!(lhs, Operand::Immediate(Immediate::I64(5)))
                    && matches!(rhs, Operand::Immediate(Immediate::I64(10)))
            } else {
                false
            }
        });
        assert!(has_moved, "Invariant instruction should be in pre-header");
    }
}

impl LoopInvariantCodeMotion {
    pub(super) fn find_loops(&self, func: &Function) -> Vec<LoopInfo> {
        let mut loops = Vec::new();

        let dominators = calculate_dominators(func);
        let mut preds: HashMap<&str, Vec<&str>> = HashMap::new();
        for block in &func.blocks {
            for succ in block.successors() {
                preds.entry(succ).or_default().push(block.label.as_str());
            }
        }

        for block in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } = &instr.instr
                {
                    // Check for back edges: target dominates source
                    if self.is_back_edge(&dominators, &block.label, true_target)
                        && let Some(loop_info) =
                            self.analyze_loop(true_target, &block.label, &preds)
                    {
                        loops.push(loop_info);
                    }
                    if self.is_back_edge(&dominators, &block.label, false_target)
                        && let Some(loop_info) =
                            self.analyze_loop(false_target, &block.label, &preds)
                    {
                        loops.push(loop_info);
                    }
                }
                if let Instruction::Jmp { target } = &instr.instr
                    && self.is_back_edge(&dominators, &block.label, target)
                    && let Some(loop_info) = self.analyze_loop(target, &block.label, &preds)
                {
                    loops.push(loop_info);
                }
            }
        }

        // Merge loops with the same header to handle multiple backedges
        let mut merged_loops: HashMap<String, HashSet<String>> = HashMap::new();

        for loop_info in loops {
            match merged_loops.entry(loop_info.header) {
                Entry::Vacant(e) => {
                    e.insert(loop_info.blocks);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().extend(loop_info.blocks);
                }
            }
        }

        func.blocks
            .iter()
            .filter_map(|block| {
                merged_loops.remove(&block.label).map(|blocks| LoopInfo {
                    header: block.label.clone(),
                    blocks,
                })
            })
            .collect()
    }

    fn is_back_edge(&self, dominators: &Dominators, from: &str, to: &str) -> bool {
        // A back edge exists if the target (header) dominates the source
        dominators.dominates(to, from)
    }

    fn analyze_loop(
        &self,
        header: &str,
        back_edge_source: &str,
        preds: &HashMap<&str, Vec<&str>>,
    ) -> Option<LoopInfo> {
        // Natural loop: the header plus every block that reaches the back edge source
        // without passing through the header.
        let mut loop_blocks = HashSet::new();
        let mut to_visit = Vec::new();
        let mut visited = HashSet::new();

        if header != back_edge_source {
            to_visit.push(back_edge_source.to_string());
        }
        loop_blocks.insert(header.to_string());
        loop_blocks.insert(back_edge_source.to_string());

        while let Some(block_label) = to_visit.pop() {
            if visited.contains(&block_label) {
                continue;
            }
            visited.insert(block_label.clone());
            loop_blocks.insert(block_label.clone());

            for &pred in preds.get(block_label.as_str()).into_iter().flatten() {
                if pred != header {
                    to_visit.push(pred.to_string());
                }
            }
        }

        // Add the header if not already included
        loop_blocks.insert(header.to_string());

        if !loop_blocks.contains(header) || !loop_blocks.contains(back_edge_source) {
            return None;
        }

        Some(LoopInfo {
            header: header.to_string(),
            blocks: loop_blocks,
        })
    }

    fn has_side_effects(&self, instr: &Instruction, pointer: PointerWidth) -> bool {
        instr.may_trap_integer(pointer)
            || matches!(
                instr,
                Instruction::Load { .. }
                    | Instruction::Store { .. }
                    | Instruction::AtomicLoad { .. }
                    | Instruction::AtomicStore { .. }
                    | Instruction::AtomicBinary { .. }
                    | Instruction::AtomicCompareExchange { .. }
                    | Instruction::Fence { .. }
                    | Instruction::InlineAsm { .. }
                    | Instruction::VaStart { .. }
                    | Instruction::VaArg { .. }
                    | Instruction::VaEnd { .. }
                    | Instruction::VaCopy { .. }
                    | Instruction::Call { .. }
                    | Instruction::CallIndirect { .. }
                    | Instruction::TailCall { .. }
                    | Instruction::Ret { .. }
            )
            || self.touches_memory_under_nightly(instr)
    }

    #[cfg(feature = "nightly")]
    fn touches_memory_under_nightly(&self, instr: &Instruction) -> bool {
        matches!(
            instr,
            Instruction::SimdLoad { .. } | Instruction::SimdStore { .. }
        )
    }

    #[cfg(not(feature = "nightly"))]
    fn touches_memory_under_nightly(&self, _instr: &Instruction) -> bool {
        false
    }
}

#[derive(Debug)]
pub(super) struct LoopInfo {
    pub(super) header: String,
    pub(super) blocks: HashSet<String>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::mir::transform::LoopUnrolling;
    use crate::mir::{
        AddressMode, AtomicBinOp, FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MemoryAttrs,
        MemoryOrdering, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_licm_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let result = licm.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No changes expected
    }

    #[test]
    fn test_licm_no_loops() {
        // Function with no loops - nothing to optimize
        let mut func = FunctionBuilder::new("no_loop")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_licm_keeps_register_read_before_its_definition() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut func = FunctionBuilder::new("read_before_def")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(add(1, imm(5), imm(0)))
            .instr(add(2, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(3).into(),
                lhs: reg(0),
                rhs: imm(3),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(3).into(),
                true_target: "body".to_string(),
                false_target: "exit".to_string(),
            })
            .block("body")
            .instr(add(2, reg(2), reg(1)))
            .instr(add(1, imm(2), imm(3)))
            .instr(add(0, reg(0), imm(1)))
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(reg(2)),
            })
            .build();

        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");

        let defines_v1_constant = |instr: &Located| {
            matches!(
                &instr.instr,
                Instruction::IntBinary {
                    dst: Register::Virtual(v),
                    lhs: Operand::Immediate(Immediate::I64(2)),
                    ..
                } if *v == VirtualReg::gpr(1)
            )
        };
        let hoisted = func
            .get_block("loop_pre")
            .is_some_and(|pre| pre.instructions.iter().any(defines_v1_constant));
        assert!(!hoisted);
        let body = func.get_block("body").expect("body block exists");
        assert!(body.instructions.iter().any(defines_v1_constant));
    }

    #[test]
    fn test_licm_self_loop_does_not_capture_predecessors() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut func = FunctionBuilder::new("self_loop_predecessor")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(9).into(),
                lhs: reg(0),
                rhs: imm(1),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(9).into(),
                true_target: "then".to_string(),
                false_target: "hdr".to_string(),
            })
            .block("then")
            .instr(add(5, imm(7), imm(0)))
            .instr(add(6, reg(5), imm(1)))
            .instr(Instruction::Jmp {
                target: "hdr".to_string(),
            })
            .block("hdr")
            .instr(add(1, reg(1), imm(1)))
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(2).into(),
                lhs: reg(1),
                rhs: imm(4),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "hdr".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(imm(0)),
            })
            .build();

        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");

        let then_block = func.get_block("then").expect("then block exists");
        assert!(then_block.instructions.iter().any(|i| matches!(
            &i.instr,
            Instruction::IntBinary {
                dst: Register::Virtual(v),
                ..
            } if *v == VirtualReg::gpr(5)
        )));
    }

    #[test]
    fn test_licm_nested_loops_keep_inner_hoisted_values_in_outer_scope() {
        use crate::runtime::executor::interpret_mir_function;

        let i64_ty = MirType::Scalar(ScalarType::I64);
        let add = |dst: u32, lhs: Operand, rhs: Operand| Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: VirtualReg::gpr(dst).into(),
            lhs,
            rhs,
        };
        let less = |dst: u32, lhs: u32, bound: i64| Instruction::IntCmp {
            op: IntCmpOp::SLt,
            ty: MirType::Scalar(ScalarType::I1),
            dst: VirtualReg::gpr(dst).into(),
            lhs: Operand::Register(VirtualReg::gpr(lhs).into()),
            rhs: Operand::Immediate(Immediate::I64(bound)),
        };
        let reg = |id: u32| Operand::Register(VirtualReg::gpr(id).into());
        let imm = |v: i64| Operand::Immediate(Immediate::I64(v));
        let mut inner_body = FunctionBuilder::new("nested_hoist")
            .returns(i64_ty)
            .block("entry")
            .instr(add(0, imm(0), imm(0)))
            .instr(add(1, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "outer".to_string(),
            })
            .block("inner")
            .instr(less(15, 12, 3))
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(15).into(),
                true_target: "inner_body".to_string(),
                false_target: "outer_latch".to_string(),
            })
            .block("inner_body")
            .instr(add(2, imm(1), imm(0)));
        for (dst, src) in [(3, 2), (4, 3), (6, 4), (7, 6), (9, 7), (10, 9), (11, 10)] {
            inner_body = inner_body.instr(add(dst, reg(src), imm(1)));
        }
        let mut func = inner_body
            .instr(add(1, reg(1), reg(11)))
            .instr(add(12, reg(12), imm(1)))
            .instr(Instruction::Jmp {
                target: "inner".to_string(),
            })
            .block("outer")
            .instr(less(14, 0, 2))
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(14).into(),
                true_target: "outer_body".to_string(),
                false_target: "exit".to_string(),
            })
            .block("outer_body")
            .instr(add(12, imm(0), imm(0)))
            .instr(Instruction::Jmp {
                target: "inner".to_string(),
            })
            .block("outer_latch")
            .instr(add(0, reg(0), imm(1)))
            .instr(Instruction::Jmp {
                target: "outer".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(reg(1)),
            })
            .build();

        let expected = interpret_mir_function(&func, &[]).expect("interpret before LICM");
        LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("LICM failed");
        let actual = interpret_mir_function(&func, &[]).ok().flatten();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_licm_side_effect_not_moved() {
        // Store inside loop should NOT be moved out
        let mut func = FunctionBuilder::new("side_effect")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::Store {
                ty: MirType::Scalar(ScalarType::I64),
                src: Operand::Immediate(Immediate::I64(42)),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(0).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let licm = LoopInvariantCodeMotion;
        let changed = licm.apply(&mut func).expect("should succeed");
        // Store has side effects - should not be moved
        assert!(!changed);
    }

    #[test]
    fn test_licm_atomic_load_not_moved() {
        let mut func = FunctionBuilder::new("atomic_load_in_loop")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::AtomicLoad {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                ordering: MemoryOrdering::SeqCst,
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "exit".to_string(),
                false_target: "loop".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let changed = LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_licm_atomic_rmw_not_moved() {
        let mut func = FunctionBuilder::new("atomic_rmw_in_loop")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::AtomicBinary {
                op: AtomicBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 0,
                    address_space: Default::default(),
                },
                value: Operand::Immediate(Immediate::I64(1)),
                ordering: MemoryOrdering::SeqCst,
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "exit".to_string(),
                false_target: "loop".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let changed = LoopInvariantCodeMotion
            .apply(&mut func)
            .expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_loop_unrolling_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let unroll = LoopUnrolling;
        let result = unroll.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_loop_unrolling_no_loops() {
        let mut func = FunctionBuilder::new("no_loop")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let unroll = LoopUnrolling;
        let changed = unroll.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_loop_unrolling_large_bound_not_unrolled() {
        // Loop with bound > 32 should not be unrolled
        let mut func = FunctionBuilder::new("large_bound")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: MirType::Scalar(ScalarType::I1),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(100)), // > 32
            })
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(1).into(),
                true_target: "loop".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let unroll = LoopUnrolling;
        let changed = unroll.apply(&mut func).expect("should succeed");
        // Loop with large bound should not be unrolled
        assert!(!changed);
        // Original loop should still exist
        assert!(func.get_block("loop").is_some());
    }

    #[test]
    fn test_licm_stress_no_infinite_loop() {
        // Create function with nested-looking structure but no actual loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Add many blocks in linear fashion
        for i in 0..50 {
            let label = format!("block{}", i);
            let next_label = if i < 49 {
                format!("block{}", i + 1)
            } else {
                "exit".to_string()
            };

            let mut block = Block::new(&label);
            block.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(i).into(),
                lhs: Operand::Immediate(Immediate::I64(i as i64)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            });
            block.push(Instruction::Jmp { target: next_label });
            func.add_block(block);
        }

        let mut exit_block = Block::new("exit");
        exit_block.push(Instruction::Ret { value: None });
        func.add_block(exit_block);

        // Update entry to jump to block0
        func.blocks[0]
            .instructions
            .push(Located::from(Instruction::Jmp {
                target: "block0".to_string(),
            }));

        let licm = LoopInvariantCodeMotion;
        let result = licm.apply(&mut func);
        assert!(result.is_ok());
    }

    #[test]
    fn test_loop_unrolling_stress_no_infinite_loop() {
        // Many simple blocks should not cause infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        // Add 100 independent blocks
        for i in 0..100 {
            let label = format!("dead_block_{}", i);
            let mut block = Block::new(&label);
            block.push(Instruction::Ret { value: None });
            func.add_block(block);
        }

        let unroll = LoopUnrolling;
        let result = unroll.apply(&mut func);
        assert!(result.is_ok());
    }
}

/// Loop invariant code motion over the pipeline's SSA graph. A value has exactly one
/// definition, so a computation whose operands are all defined in a block that dominates
/// the header can move to a preheader without the availability proof the mutable form
/// needs for a register that may be written again.
mod ssa_licm {
    use super::LoopInvariantCodeMotion;
    use crate::mir::{
        Instruction, Register,
        ssa::{BlockId, SsaBlock, SsaEdge, SsaFunction, SsaInst, ValueDef, ValueId},
        transform::TransformError,
    };
    use lamina_mir::Located;
    use std::collections::{HashMap, HashSet};

    const MAX_LOOPS: usize = 10;
    const MAX_PASSES_PER_LOOP: usize = 5;
    const MAX_HOISTED: usize = 20;

    struct Loop {
        header: usize,
        body: HashSet<usize>,
    }

    pub(super) fn run(
        licm: &LoopInvariantCodeMotion,
        graph: &mut SsaFunction,
    ) -> Result<bool, TransformError> {
        let mut changed = false;
        let headers: Vec<usize> = loops(graph)
            .into_iter()
            .take(MAX_LOOPS)
            .map(|info| info.header)
            .collect();
        for header in headers {
            for _ in 0..MAX_PASSES_PER_LOOP {
                let Some(info) = loops(graph)
                    .into_iter()
                    .find(|candidate| candidate.header == header)
                else {
                    break;
                };
                if !hoist(licm, graph, &info) {
                    break;
                }
                changed = true;
            }
        }
        Ok(changed)
    }

    /// Natural loops, one per header, in emission order.
    fn loops(graph: &SsaFunction) -> Vec<Loop> {
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); graph.blocks.len()];
        for (b, block) in graph.blocks.iter().enumerate() {
            for edge in &block.edges {
                preds[edge.target.0 as usize].push(b);
            }
        }
        let mut bodies: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (b, block) in graph.blocks.iter().enumerate() {
            if !graph.dominators.is_reachable(b) {
                continue;
            }
            for edge in &block.edges {
                let header = edge.target.0 as usize;
                if !graph.dominators.dominates(header, b) {
                    continue;
                }
                let body = bodies.entry(header).or_default();
                body.insert(header);
                body.insert(b);
                let mut stack = vec![b];
                while let Some(block) = stack.pop() {
                    if block == header {
                        continue;
                    }
                    for &pred in &preds[block] {
                        if pred != header && body.insert(pred) {
                            stack.push(pred);
                        }
                    }
                }
            }
        }
        let mut out: Vec<Loop> = bodies
            .into_iter()
            .map(|(header, body)| Loop { header, body })
            .collect();
        out.sort_by_key(|info| graph.blocks[info.header].order);
        out
    }

    fn defining_block(graph: &SsaFunction, value: ValueId) -> usize {
        match graph.values[value.0 as usize].def {
            ValueDef::BlockParam { block, .. } | ValueDef::Inst { block, .. } => block.0 as usize,
        }
    }

    /// Blocks outside the loop with an edge into the header.
    fn entries(graph: &SsaFunction, info: &Loop) -> Vec<usize> {
        (0..graph.blocks.len())
            .filter(|b| {
                !info.body.contains(b)
                    && graph.dominators.is_reachable(*b)
                    && graph.blocks[*b]
                        .edges
                        .iter()
                        .any(|edge| edge.target.0 as usize == info.header)
            })
            .collect()
    }

    /// A block already standing alone in front of the header, which hoisted code can
    /// join instead of adding another one.
    fn existing_preheader(graph: &SsaFunction, outside: &[usize]) -> Option<usize> {
        let &[pre] = outside else {
            return None;
        };
        let block = graph.blocks.get(pre)?;
        (pre != 0
            && block.edges.len() == 1
            && matches!(
                block.insts.last().map(|inst| &*inst.instruction),
                Some(Instruction::Jmp { .. })
            ))
        .then_some(pre)
    }

    fn hoist(licm: &LoopInvariantCodeMotion, graph: &mut SsaFunction, info: &Loop) -> bool {
        let outside = entries(graph, info);
        if outside.is_empty() {
            return false;
        }
        let pointer = graph.shell.pointer_width;
        let mut carried: HashMap<Register, usize> = HashMap::new();
        for value in &graph.values {
            *carried.entry(value.register.clone()).or_insert(0) += 1;
        }

        let mut body: Vec<usize> = info.body.iter().copied().collect();
        body.sort_by_key(|&b| graph.blocks[b].order);
        let mut picked: Vec<(usize, usize)> = Vec::new();
        'blocks: for &b in &body {
            for (i, inst) in graph.blocks[b].insts.iter().enumerate() {
                let Some(defined) = inst.defines else {
                    continue;
                };
                if licm.has_side_effects(&inst.instruction, pointer) {
                    continue;
                }
                if carried.get(&graph.values[defined.0 as usize].register) != Some(&1) {
                    continue;
                }
                let available = inst.uses.iter().all(|&used| {
                    let from = defining_block(graph, used);
                    from != info.header && graph.dominators.dominates(from, info.header)
                });
                if !available {
                    continue;
                }
                picked.push((b, i));
                if picked.len() >= MAX_HOISTED {
                    break 'blocks;
                }
            }
        }
        if picked.is_empty() {
            return false;
        }

        let mut moved: Vec<SsaInst> = Vec::with_capacity(picked.len());
        for &(b, i) in picked.iter().rev() {
            moved.push(graph.blocks[b].insts.remove(i));
        }
        moved.reverse();

        match existing_preheader(graph, &outside) {
            Some(pre) => {
                let at = graph.blocks[pre].insts.len() - 1;
                graph.blocks[pre].insts.splice(at..at, moved);
                reindex(graph);
            }
            None => {
                add_preheader(graph, info, &outside, moved);
                reindex(graph);
                graph.recompute_dominators();
            }
        }
        true
    }

    fn add_preheader(
        graph: &mut SsaFunction,
        info: &Loop,
        outside: &[usize],
        mut insts: Vec<SsaInst>,
    ) {
        let preheader = BlockId(graph.blocks.len() as u32);
        let header_label = graph.blocks[info.header].label.clone();
        let label = graph.fresh_label(&format!("{header_label}_pre"));
        let header_params = graph.blocks[info.header].params.clone();
        let mut params = Vec::with_capacity(header_params.len());
        for (index, &param) in header_params.iter().enumerate() {
            let value = &graph.values[param.0 as usize];
            let (ty, register) = (value.ty, value.register.clone());
            params.push(graph.add_value(
                ValueDef::BlockParam {
                    block: preheader,
                    index: index as u32,
                },
                ty,
                register,
            ));
        }
        insts.push(SsaInst {
            source: 0,
            defines: None,
            uses: Vec::new(),
            instruction: Located::from(Instruction::Jmp {
                target: header_label.clone(),
            }),
        });
        let order = graph.blocks[info.header].order;
        graph.blocks.push(SsaBlock {
            label: label.clone(),
            source: None,
            order: (order.0, order.1 - 1),
            params: params.clone(),
            insts,
            edges: vec![SsaEdge {
                target: BlockId(info.header as u32),
                args: params,
            }],
        });

        for &b in outside {
            for e in 0..graph.blocks[b].edges.len() {
                if graph.blocks[b].edges[e].target.0 as usize != info.header {
                    continue;
                }
                graph.blocks[b].edges[e].target = preheader;
                if let Some(terminator) = graph.blocks[b].insts.last_mut()
                    && let Some(target) = terminator.instruction.successors_mut().get_mut(e)
                {
                    **target = label.clone();
                }
            }
        }
        if graph.shell.entry == header_label {
            graph.shell.entry = label;
        }
    }

    /// Restores every value's recorded definition site after blocks or instructions moved.
    fn reindex(graph: &mut SsaFunction) {
        for b in 0..graph.blocks.len() {
            for index in 0..graph.blocks[b].params.len() {
                let param = graph.blocks[b].params[index];
                graph.values[param.0 as usize].def = ValueDef::BlockParam {
                    block: BlockId(b as u32),
                    index: index as u32,
                };
            }
            for i in 0..graph.blocks[b].insts.len() {
                if let Some(defined) = graph.blocks[b].insts[i].defines {
                    graph.values[defined.0 as usize].def = ValueDef::Inst {
                        block: BlockId(b as u32),
                        inst: i as u32,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests_ssa_licm {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, IntCmpOp, Operand, ScalarType, VirtualReg,
        transform::test_utils::get_block, verify_function_in_pipeline,
    };

    fn gpr(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }

    fn i64t() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    /// entry -> head -> left -> join -> head, with the invariant `v4 = v0 + 1` computed
    /// in `left` and read in `join`.
    fn diamond_loop() -> Function {
        FunctionBuilder::new("diamond")
            .param(gpr(0), i64t())
            .returns(i64t())
            .block("entry")
            .instr(Instruction::Copy {
                ty: i64t(),
                dst: gpr(1),
                src: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Copy {
                ty: i64t(),
                dst: gpr(3),
                src: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "head".into(),
            })
            .block("head")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: i64t(),
                dst: gpr(2),
                lhs: Operand::Register(gpr(1)),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Br {
                cond: gpr(2),
                true_target: "left".into(),
                false_target: "done".into(),
            })
            .block("left")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64t(),
                dst: gpr(4),
                lhs: Operand::Register(gpr(0)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Jmp {
                target: "join".into(),
            })
            .block("join")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64t(),
                dst: gpr(1),
                lhs: Operand::Register(gpr(1)),
                rhs: Operand::Register(gpr(4)),
            })
            .instr(Instruction::Jmp {
                target: "head".into(),
            })
            .block("done")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(gpr(1))),
            })
            .build()
    }

    fn computes(block: &Block, register: Register) -> bool {
        block.instructions.iter().any(
            |instr| matches!(&instr.instr, Instruction::IntBinary { dst, .. } if *dst == register),
        )
    }

    #[test]
    fn an_invariant_read_in_another_block_leaves_the_loop() {
        let mut func = diamond_loop();
        assert!(
            LoopInvariantCodeMotion
                .apply(&mut func)
                .expect("licm runs on the diamond")
        );
        assert!(!computes(get_block(&func, "left"), gpr(4)));
        assert!(computes(get_block(&func, "entry"), gpr(4)));
        verify_function_in_pipeline(&func).expect("hoisted function verifies");
    }

    #[test]
    fn a_block_already_standing_in_front_of_the_header_takes_the_hoisted_code() {
        let mut func = diamond_loop();
        let before = func.blocks.len();
        assert!(LoopInvariantCodeMotion.apply(&mut func).expect("licm runs"));
        assert_eq!(func.blocks.len(), before);
    }

    #[test]
    fn a_loop_whose_header_is_the_entry_gets_a_preheader() {
        let mut func = FunctionBuilder::new("entry_header")
            .param(gpr(0), i64t())
            .returns(i64t())
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64t(),
                dst: gpr(1),
                lhs: Operand::Register(gpr(0)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: i64t(),
                dst: gpr(2),
                lhs: Operand::Register(gpr(1)),
                rhs: Operand::Immediate(Immediate::I64(10)),
            })
            .instr(Instruction::Br {
                cond: gpr(2),
                true_target: "entry".into(),
                false_target: "done".into(),
            })
            .block("done")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(gpr(1))),
            })
            .build();
        assert!(LoopInvariantCodeMotion.apply(&mut func).expect("licm runs"));
        assert_ne!(func.entry, "entry");
        let entry = func.entry.clone();
        assert!(computes(get_block(&func, &entry), gpr(1)));
        verify_function_in_pipeline(&func).expect("hoisted function verifies");
    }
}
