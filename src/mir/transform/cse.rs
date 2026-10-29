//! Common Subexpression Elimination (CSE) transform for MIR.
//!
//! Runs on the SSA value graph. A pure arithmetic or comparison whose operands are the
//! same values as an earlier, dominating one becomes a copy of that result. The earlier
//! result's register must still hold it at the duplicate, which is checked against every
//! other definition of that register.

use crate::mir::{
    Function, Instruction, MirType, Operand, Register, ScalarType,
    ssa::{SsaFunction, ValueDef, ValueId, destroy},
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_mir::Located;
use std::{collections::HashMap, mem};

/// An expression's text and the values it reads.
type Expr = (String, Vec<ValueId>);
/// A duplicate's block, slot index, the earlier value it copies, and the copy.
type Rewrite = (usize, usize, ValueId, Instruction);

/// A position in the graph: a block index and a slot. Slot `i + 1` is instruction `i`.
/// `EDGE_MOVE` is the end of a block, where an edge writes the registers of its target's
/// parameters.
type Site = (usize, usize);
const EDGE_MOVE: usize = usize::MAX;

/// Common Subexpression Elimination (CSE)
/// Replaces a pure computation with a copy of an earlier identical one that dominates it.
#[derive(Default)]
pub struct CommonSubexpressionElimination;

impl Transform for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "common_subexpression_elimination"
    }

    fn description(&self) -> &'static str {
        "Replaces repeated pure computations with copies of a dominating result"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl CommonSubexpressionElimination {
    /// Extract the type from an instruction for use in replacement instructions
    fn extract_instruction_type(&self, instr: &Instruction) -> MirType {
        match instr {
            Instruction::Copy { ty, .. }
            | Instruction::IntBinary { ty, .. }
            | Instruction::FloatBinary { ty, .. }
            | Instruction::FloatUnary { ty, .. }
            | Instruction::Select { ty, .. }
            | Instruction::Load { ty, .. }
            | Instruction::Store { ty, .. }
            | Instruction::VectorOp { ty, .. } => *ty,
            Instruction::IntCmp { .. } | Instruction::FloatCmp { .. } => {
                MirType::Scalar(ScalarType::I1)
            }
            _ => MirType::Scalar(ScalarType::I64),
        }
    }

    fn expr_key(&self, instr: &Instruction) -> Option<String> {
        match instr {
            Instruction::IntBinary {
                op, ty, lhs, rhs, ..
            } => Some(format!("IntBinary_{op:?}_{ty:?}_{lhs:?}_{rhs:?}")),
            Instruction::FloatBinary {
                op, ty, lhs, rhs, ..
            } => Some(format!("FloatBinary_{op:?}_{ty:?}_{lhs:?}_{rhs:?}")),
            Instruction::IntCmp {
                op, ty, lhs, rhs, ..
            } => Some(format!("IntCmp_{op:?}_{ty:?}_{lhs:?}_{rhs:?}")),
            Instruction::FloatCmp {
                op, ty, lhs, rhs, ..
            } => Some(format!("FloatCmp_{op:?}_{ty:?}_{lhs:?}_{rhs:?}")),
            _ => None,
        }
    }

    /// A computation over immediates alone. Recomputing it at each use is free, while
    /// sharing its first result would keep that register live across the function.
    fn reads_only_immediates(instr: &Instruction) -> bool {
        let (lhs, rhs) = match instr {
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => (lhs, rhs),
            _ => return false,
        };
        matches!(lhs, Operand::Immediate(_)) && matches!(rhs, Operand::Immediate(_))
    }

    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let no_callee = |_: &str| -> Option<MirType> { None };
        let Ok(mut graph) = SsaFunction::build(func, &no_callee) else {
            return Ok(false);
        };
        verify_graph(&graph, "before CSE")?;

        let reach = reachability(&graph);
        let defs = definitions(&graph);
        let mut available: HashMap<Expr, Vec<(ValueId, Site)>> = HashMap::new();
        let mut rewrites: Vec<Rewrite> = Vec::new();

        for (b, block) in graph.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                let (Some(defined), Some(text)) = (inst.defines, self.expr_key(&inst.instruction))
                else {
                    continue;
                };
                if Self::reads_only_immediates(&inst.instruction) {
                    continue;
                }
                let site = (b, i + 1);
                let key = (text, inst.uses.clone());
                let dst = &graph.values[defined.0 as usize].register;

                let earlier = available.get(&key).and_then(|candidates| {
                    candidates.iter().find(|&&(_, from)| {
                        if from.0 == b {
                            from.1 < site.1
                        } else {
                            graph.dominators.dominates(from.0, b)
                        }
                    })
                });
                match earlier {
                    Some(&(src, from)) => {
                        let src_reg = &graph.values[src.0 as usize].register;
                        let clobbered = defs.get(src_reg).is_some_and(|sites| {
                            sites
                                .iter()
                                .any(|&w| w != from && between(&reach, w, from, site))
                        });
                        if clobbered || src_reg == dst {
                            available.entry(key).or_default().push((defined, site));
                            continue;
                        }
                        let copy = Instruction::Copy {
                            ty: self.extract_instruction_type(&inst.instruction),
                            dst: dst.clone(),
                            src: Operand::Register(src_reg.clone()),
                        };
                        rewrites.push((b, i, src, copy));
                    }
                    None => available.entry(key).or_default().push((defined, site)),
                }
            }
        }

        if rewrites.is_empty() {
            return Ok(false);
        }
        for (b, i, src, copy) in rewrites {
            let inst = &mut graph.blocks[b].insts[i];
            inst.instruction = Located::at(copy, inst.instruction.loc);
            inst.uses = vec![src];
        }
        verify_graph(&graph, "after CSE")?;
        *func = destroy(&graph).map_err(|e| TransformError::Unsupported(e.to_string()))?;
        Ok(true)
    }
}

fn verify_graph(graph: &SsaFunction, stage: &str) -> Result<(), TransformError> {
    #[cfg(any(debug_assertions, feature = "ssa-round-trip"))]
    graph.verify().map_err(|violations| {
        let lines: Vec<&str> = violations.iter().map(|v| v.message.as_str()).collect();
        TransformError::Unsupported(format!("SSA verification {stage}: {}", lines.join("; ")))
    })?;
    #[cfg(not(any(debug_assertions, feature = "ssa-round-trip")))]
    let _ = (graph, stage);
    Ok(())
}

/// `reach[a][b]`: a path of at least one edge runs from block `a` to block `b`.
fn reachability(graph: &SsaFunction) -> Vec<Vec<bool>> {
    let n = graph.blocks.len();
    let mut reach = vec![vec![false; n]; n];
    for (start, row) in reach.iter_mut().enumerate() {
        let mut stack: Vec<usize> = graph.blocks[start]
            .edges
            .iter()
            .map(|e| e.target.0 as usize)
            .collect();
        while let Some(b) = stack.pop() {
            if !mem::replace(&mut row[b], true) {
                stack.extend(graph.blocks[b].edges.iter().map(|e| e.target.0 as usize));
            }
        }
    }
    reach
}

/// Every site that writes a register: each instruction result, and each edge whose
/// argument lives in a different register than the target parameter it feeds.
fn definitions(graph: &SsaFunction) -> HashMap<Register, Vec<Site>> {
    let mut defs: HashMap<Register, Vec<Site>> = HashMap::new();
    for value in &graph.values {
        if let ValueDef::Inst { block, inst } = value.def {
            defs.entry(value.register.clone())
                .or_default()
                .push((block.0 as usize, inst as usize + 1));
        }
    }
    for (b, block) in graph.blocks.iter().enumerate() {
        for edge in &block.edges {
            let target = &graph.blocks[edge.target.0 as usize];
            for (&param, &arg) in target.params.iter().zip(&edge.args) {
                let param = &graph.values[param.0 as usize];
                if graph.values[arg.0 as usize].register != param.register {
                    defs.entry(param.register.clone())
                        .or_default()
                        .push((b, EDGE_MOVE));
                }
            }
        }
    }
    defs
}

/// Whether a write at `w` can run after `from` and before `to` on some path.
fn between(reach: &[Vec<bool>], w: Site, from: Site, to: Site) -> bool {
    let after_from = (w.0 == from.0 && w.1 > from.1) || reach[from.0][w.0];
    let before_to = (w.0 == to.0 && to.1 > w.1) || reach[w.0][to.0];
    after_from && before_to
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FloatBinOp, FloatCmpOp, FunctionBuilder, Immediate, IntBinOp, ScalarType, VirtualReg,
        transform::test_utils::get_block,
    };
    use lamina_mir::Located;

    #[test]
    fn test_cse_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let cse = CommonSubexpressionElimination;
        let result = cse.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn cse_float_comparison_reuse_has_i1_result_type() {
        let f64_ty = MirType::Scalar(ScalarType::F64);
        let i1_ty = MirType::Scalar(ScalarType::I1);
        let lhs: Register = VirtualReg::fpr(0).into();
        let rhs: Register = VirtualReg::fpr(1).into();
        let comparison = Instruction::FloatCmp {
            op: FloatCmpOp::Eq,
            ty: f64_ty,
            dst: VirtualReg::gpr(2).into(),
            lhs: Operand::Register(lhs),
            rhs: Operand::Register(rhs),
        };

        assert_eq!(
            CommonSubexpressionElimination.extract_instruction_type(&comparison),
            i1_ty
        );
    }

    #[test]
    fn test_cse_no_duplicate_expressions() {
        // No duplicate expressions, nothing to eliminate
        let mut func = FunctionBuilder::new("no_dups")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Sub,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(3)),
                rhs: Operand::Immediate(Immediate::I64(4)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination;
        let changed = cse.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_cse_with_intervening_redefinition() {
        // CSE should not reuse expression if operand was redefined
        let mut func = FunctionBuilder::new("redef")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // v0 = v1 + v2
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            // Redefine v1
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(99)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // v3 = v1 + v2 (should NOT reuse v0 because v1 was redefined)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(3).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination;
        let changed = cse.apply(&mut func).expect("should succeed");

        // Should NOT eliminate the second v1 + v2 since v1 was redefined
        assert!(!changed);
    }

    #[test]
    fn test_cse_duplicate_expression_does_not_crash() {
        // Test that CSE handles duplicate expressions without panicking
        // CSE may or may not optimize depending on block size and loop detection
        let mut func = FunctionBuilder::new("dup")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Register(VirtualReg::gpr(11).into()),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Register(VirtualReg::gpr(11).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination;
        let result = cse.apply(&mut func);
        // Main test: CSE should not panic or error
        assert!(result.is_ok());
        // Function should still be valid (have blocks and instructions)
        assert!(!func.blocks.is_empty());
        let entry = get_block(&func, "entry");
        assert!(!entry.instructions.is_empty());
    }

    #[test]
    fn test_cse_stress_no_infinite_loop() {
        // Stress test with many instructions to ensure no infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        for i in 0..200 {
            func.blocks[0].instructions.insert(
                0,
                Located::from(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i).into(),
                    lhs: Operand::Immediate(Immediate::I64(i as i64)),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                }),
            );
        }
        func.blocks[0]
            .instructions
            .push(Located::from(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            }));

        let cse = CommonSubexpressionElimination;
        let result = cse.apply(&mut func);
        assert!(result.is_ok());
    }

    fn int_add(ty: ScalarType, dst: u32, lhs: u32, rhs: u32) -> Instruction {
        Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ty),
            dst: VirtualReg::gpr(dst).into(),
            lhs: Operand::Register(VirtualReg::gpr(lhs).into()),
            rhs: Operand::Register(VirtualReg::gpr(rhs).into()),
        }
    }

    #[test]
    fn cse_eliminates_identical_repeated_expression() {
        let mut func = FunctionBuilder::new("repeat")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(int_add(ScalarType::I64, 0, 10, 11))
            .instr(int_add(ScalarType::I64, 1, 10, 11))
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let changed = CommonSubexpressionElimination
            .apply(&mut func)
            .expect("should succeed");
        assert!(changed);

        let entry = get_block(&func, "entry");
        let first: Register = VirtualReg::gpr(0).into();
        let second: Register = VirtualReg::gpr(1).into();
        match &entry.instructions[1].instr {
            Instruction::Copy {
                ty,
                dst,
                src: Operand::Register(src),
            } => {
                assert_eq!(*ty, MirType::Scalar(ScalarType::I64));
                assert_eq!(*dst, second);
                assert_eq!(*src, first);
            }
            other => panic!("expected Copy for repeated add, got {other:?}"),
        }
    }

    #[test]
    fn cse_does_not_merge_same_shape_ops_of_different_types() {
        let mut func = FunctionBuilder::new("typed")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(int_add(ScalarType::I8, 0, 10, 11))
            .instr(int_add(ScalarType::I32, 1, 10, 11))
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let changed = CommonSubexpressionElimination
            .apply(&mut func)
            .expect("should succeed");
        assert!(!changed);

        let entry = get_block(&func, "entry");
        assert!(matches!(
            &entry.instructions[1].instr,
            Instruction::IntBinary {
                ty: MirType::Scalar(ScalarType::I32),
                ..
            }
        ));
    }

    fn imm_copy(dst: u32, value: i64) -> Instruction {
        Instruction::Copy {
            ty: MirType::Scalar(ScalarType::I64),
            dst: VirtualReg::gpr(dst).into(),
            src: Operand::Immediate(Immediate::I64(value)),
        }
    }

    #[test]
    fn cse_reuses_result_of_dominating_block() {
        let mut func = FunctionBuilder::new("dominated")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(int_add(ScalarType::I64, 0, 10, 11))
            .instr(Instruction::Jmp {
                target: "next".to_string(),
            })
            .block("next")
            .instr(int_add(ScalarType::I64, 1, 10, 11))
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        assert!(
            CommonSubexpressionElimination
                .apply(&mut func)
                .expect("should succeed")
        );
        let first: Register = VirtualReg::gpr(0).into();
        let next = get_block(&func, "next");
        assert!(matches!(
            &next.instructions[0].instr,
            Instruction::Copy { src: Operand::Register(src), .. } if *src == first
        ));
    }

    #[test]
    fn cse_refuses_when_result_register_is_overwritten_on_the_way() {
        let mut func = FunctionBuilder::new("clobbered")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(int_add(ScalarType::I64, 0, 10, 11))
            .instr(Instruction::Jmp {
                target: "next".to_string(),
            })
            .block("next")
            .instr(imm_copy(0, 99))
            .instr(int_add(ScalarType::I64, 1, 10, 11))
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        assert!(
            !CommonSubexpressionElimination
                .apply(&mut func)
                .expect("should succeed")
        );
    }

    #[test]
    fn cse_reuses_within_a_loop_header() {
        let mut func = FunctionBuilder::new("header")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Jmp {
                target: "head".to_string(),
            })
            .block("head")
            .instr(imm_copy(2, 0))
            .instr(int_add(ScalarType::I64, 0, 10, 11))
            .instr(int_add(ScalarType::I64, 1, 10, 11))
            .instr(Instruction::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "head".to_string(),
                false_target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        assert!(
            CommonSubexpressionElimination
                .apply(&mut func)
                .expect("should succeed")
        );
        let head = get_block(&func, "head");
        assert!(matches!(
            &head.instructions[2].instr,
            Instruction::Copy {
                src: Operand::Register(_),
                ..
            }
        ));
    }

    fn float_div(ty: ScalarType, dst: u32, lhs: Immediate, rhs: Immediate) -> Instruction {
        Instruction::FloatBinary {
            op: FloatBinOp::FDiv,
            ty: MirType::Scalar(ty),
            dst: VirtualReg::fpr(dst).into(),
            lhs: Operand::Immediate(lhs),
            rhs: Operand::Immediate(rhs),
        }
    }

    #[test]
    fn cse_leaves_immediate_only_expressions_alone() {
        let mut func = FunctionBuilder::new("consts")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(float_div(
                ScalarType::F32,
                0,
                Immediate::F32(0.0),
                Immediate::F32(0.0),
            ))
            .instr(float_div(
                ScalarType::F32,
                1,
                Immediate::F32(0.0),
                Immediate::F32(0.0),
            ))
            .instr(float_div(
                ScalarType::F64,
                2,
                Immediate::F64(1.0),
                Immediate::F64(0.0),
            ))
            .instr(float_div(
                ScalarType::F64,
                3,
                Immediate::F64(1.0),
                Immediate::F64(0.0),
            ))
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        assert!(
            !CommonSubexpressionElimination
                .apply(&mut func)
                .expect("should succeed")
        );
        let entry = get_block(&func, "entry");
        assert!(
            entry
                .instructions
                .iter()
                .take(4)
                .all(|instr| matches!(&instr.instr, Instruction::FloatBinary { .. })),
            "{:?}",
            entry.instructions
        );
    }
}
