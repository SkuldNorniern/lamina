use std::collections::{HashMap, HashSet};

use crate::mir::{
    Block, Function, Instruction, Register, VirtualReg,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, calculate_dominators,
        inline::ModuleInlining,
    },
};

/// Copies a small pure loop test into each back edge. The back edge then branches on the
/// test itself, so an iteration takes one branch instead of a jump to the header followed
/// by a second branch out of it.
#[derive(Default)]
pub struct LoopRotation;

const MAX_TEST_INSTRUCTIONS: usize = 8;

impl Transform for LoopRotation {
    fn name(&self) -> &'static str {
        "loop_rotation"
    }

    fn description(&self) -> &'static str {
        "Copies a pure loop test into each back edge so the back edge branches on the test"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;
        let mut rotated: HashSet<String> = HashSet::new();
        let mut next_id = next_virtual_id(func);
        let headers: Vec<String> = func
            .blocks
            .iter()
            .filter(|b| is_test_block(b))
            .map(|b| b.label.clone())
            .collect();
        for header in headers {
            let Some(test) = func.get_block(&header).map(|b| b.instructions.clone()) else {
                continue;
            };
            let dominators = calculate_dominators(func);
            let latches: Vec<usize> = func
                .blocks
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    b.label != header
                        && matches!(b.instructions.last(), Some(Instruction::Jmp { target }) if *target == header)
                        && dominators.dominates(&header, &b.label)
                })
                .map(|(index, _)| index)
                .collect();
            if latches.is_empty() {
                continue;
            }
            let local = local_temporaries(func, &header, &test);
            for index in latches {
                let copy = fresh_copy(&test, &local, &mut next_id);
                let block = &mut func.blocks[index];
                block.instructions.pop();
                block.instructions.extend(copy);
                changed = true;
                rotated.insert(header.clone());
            }
        }
        if changed {
            remove_orphaned_headers(func, &rotated);
        }
        Ok(changed)
    }
}

fn next_virtual_id(func: &Function) -> u32 {
    func.blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .flat_map(|i| i.def_reg().into_iter().chain(i.use_regs()))
        .chain(func.sig.params.iter().map(|p| &p.reg))
        .filter_map(|r| match r {
            Register::Virtual(v) => Some(v.id),
            _ => None,
        })
        .max()
        .map_or(0, |max| max + 1)
}

/// Registers the test defines before any use and never mentions outside itself. A copy
/// may rename these without changing what any other block reads.
fn local_temporaries(func: &Function, header: &str, test: &[Instruction]) -> HashSet<Register> {
    let mut seen: HashSet<&Register> = HashSet::new();
    let mut local: HashSet<Register> = HashSet::new();
    let mut excluded: HashSet<&Register> = HashSet::new();
    for instr in test {
        for reg in instr.use_regs() {
            if !seen.contains(reg) {
                excluded.insert(reg);
            }
        }
        if let Some(reg) = instr.def_reg() {
            if !seen.contains(reg) && matches!(reg, Register::Virtual(_)) {
                local.insert(reg.clone());
            }
            seen.insert(reg);
        }
    }
    local.retain(|reg| {
        !excluded.contains(reg)
            && func.blocks.iter().filter(|b| b.label != header).all(|b| {
                b.instructions
                    .iter()
                    .all(|i| i.def_reg() != Some(reg) && !i.use_regs().contains(&reg))
            })
    });
    local
}

fn fresh_copy(
    test: &[Instruction],
    local: &HashSet<Register>,
    next_id: &mut u32,
) -> Vec<Instruction> {
    let mut renamed: HashMap<Register, Register> = HashMap::new();
    for reg in local {
        if let Register::Virtual(v) = reg {
            renamed.insert(
                reg.clone(),
                Register::Virtual(VirtualReg::new(*next_id, v.class)),
            );
            *next_id += 1;
        }
    }
    test.iter()
        .cloned()
        .map(|mut instr| {
            ModuleInlining::map_registers(&mut instr, &mut |reg: &Register| {
                renamed.get(reg).cloned().unwrap_or_else(|| reg.clone())
            });
            instr
        })
        .collect()
}

fn is_test_block(block: &Block) -> bool {
    match block.instructions.split_last() {
        Some((Instruction::Br { .. }, body)) => {
            body.len() <= MAX_TEST_INSTRUCTIONS && body.iter().all(is_duplicable)
        }
        _ => false,
    }
}

fn is_duplicable(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::Copy { .. }
            | Instruction::IntBinary { .. }
            | Instruction::IntCmp { .. }
            | Instruction::IntUnary { .. }
            | Instruction::IntConvert { .. }
            | Instruction::FloatCmp { .. }
            | Instruction::Select { .. }
    )
}

fn remove_orphaned_headers(func: &mut Function, rotated: &HashSet<String>) {
    let referenced: HashSet<String> = func
        .blocks
        .iter()
        .flat_map(|b| b.successors())
        .map(str::to_owned)
        .collect();
    let entry = func.entry.clone();
    func.blocks.retain(|b| {
        b.label == entry || !rotated.contains(&b.label) || referenced.contains(&b.label)
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MirType, Operand, ScalarType, VirtualReg,
        transform::test_utils::{apply_pass, get_block},
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn reg(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }

    fn loop_with_test(test: Vec<Instruction>) -> Function {
        let mut head = test;
        head.push(Instruction::Br {
            cond: reg(1),
            true_target: "body".to_owned(),
            false_target: "exit".to_owned(),
        });
        let mut builder = FunctionBuilder::new("f")
            .param(reg(0), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: "head".to_owned(),
            })
            .block("head");
        for instr in head {
            builder = builder.instr(instr);
        }
        builder
            .block("body")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64(),
                dst: reg(0),
                lhs: Operand::Register(reg(0)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Jmp {
                target: "head".to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(reg(0))),
            })
            .build()
    }

    fn compare_test() -> Vec<Instruction> {
        vec![Instruction::IntCmp {
            op: IntCmpOp::SLt,
            ty: i64(),
            dst: reg(1),
            lhs: Operand::Register(reg(0)),
            rhs: Operand::Immediate(Immediate::I64(10)),
        }]
    }

    #[test]
    fn pure_test_is_copied_into_back_edge() {
        let mut func = loop_with_test(compare_test());
        assert!(apply_pass(&LoopRotation, &mut func));
        let body = get_block(&func, "body");
        assert!(matches!(
            body.instructions.last(),
            Some(Instruction::Br { true_target, false_target, .. })
                if true_target == "body" && false_target == "exit"
        ));
        assert!(
            body.instructions
                .iter()
                .any(|i| matches!(i, Instruction::IntCmp { .. }))
        );
    }

    #[test]
    fn copied_temporary_gets_a_fresh_register() {
        let mut func = loop_with_test(compare_test());
        apply_pass(&LoopRotation, &mut func);
        let body = get_block(&func, "body");
        let Some(Instruction::Br { cond, .. }) = body.instructions.last() else {
            panic!("back edge must end in a branch");
        };
        assert_ne!(cond, &reg(1));
    }

    #[test]
    fn test_with_call_is_not_copied() {
        let mut func = loop_with_test(vec![Instruction::Call {
            name: "probe".to_owned(),
            args: vec![Operand::Register(reg(0))],
            ret: Some(reg(1)),
        }]);
        assert!(!apply_pass(&LoopRotation, &mut func));
        assert!(matches!(
            get_block(&func, "body").instructions.last(),
            Some(Instruction::Jmp { .. })
        ));
    }
}
