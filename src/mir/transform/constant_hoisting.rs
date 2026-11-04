//! Loads large constants once, ahead of the loop that uses them.
//!
//! AArch64 rebuilds a constant it cannot encode in the instruction each time the
//! instruction runs, with up to four moves. In a loop that is a large part of the body.

use lamina_mir::Located;
use std::collections::{HashMap, HashSet};

use crate::mir::{
    Function, Immediate, Instruction, IntBinOp, MirType, Operand, Register, ScalarType, VirtualReg,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, calculate_dominators,
        loop_rotation::next_virtual_id,
    },
};

#[derive(Default)]
pub struct ConstantHoisting;

/// Past this many a loop runs short of registers, which costs more than the moves.
const MAX_PER_LOOP: usize = 6;

const WORD: MirType = MirType::Scalar(ScalarType::I64);

impl Transform for ConstantHoisting {
    fn name(&self) -> &'static str {
        "constant_hoisting"
    }

    fn description(&self) -> &'static str {
        "Moves constants an instruction cannot encode out of loops"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;
        let mut next_id = next_virtual_id(func);
        let mut loops = natural_loops(func);
        // Inner loops first, a smaller loop runs more often.
        loops.sort_by_key(|(_, body)| body.len());
        for (head, body) in &loops {
            changed |= hoist_from_loop(func, head, body, &mut next_id);
        }
        Ok(changed)
    }
}

fn successors(instr: &Instruction) -> Vec<&str> {
    match instr {
        Instruction::Jmp { target } => vec![target],
        Instruction::Br {
            true_target,
            false_target,
            ..
        } => vec![true_target, false_target],
        Instruction::Switch { cases, default, .. } => cases
            .iter()
            .map(|(_, target)| target.as_str())
            .chain([default.as_str()])
            .collect(),
        _ => Vec::new(),
    }
}

fn predecessors(func: &Function) -> HashMap<String, Vec<String>> {
    let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
    for block in &func.blocks {
        if let Some(terminator) = block.terminator() {
            for target in successors(terminator) {
                predecessors
                    .entry(target.to_string())
                    .or_default()
                    .push(block.label.clone());
            }
        }
    }
    predecessors
}

/// Each loop header with the blocks of its loop, back edges to one header merged.
fn natural_loops(func: &Function) -> Vec<(String, HashSet<String>)> {
    let dominators = calculate_dominators(func);
    let predecessors = predecessors(func);
    let mut loops: HashMap<String, HashSet<String>> = HashMap::new();
    for block in &func.blocks {
        let Some(terminator) = block.terminator() else {
            continue;
        };
        for head in successors(terminator) {
            if !dominators.dominates(head, &block.label) {
                continue;
            }
            let body = loops.entry(head.to_string()).or_default();
            body.insert(head.to_string());
            let mut pending = vec![block.label.clone()];
            while let Some(label) = pending.pop() {
                if body.insert(label.clone())
                    && let Some(before) = predecessors.get(&label)
                {
                    pending.extend(before.iter().cloned());
                }
            }
        }
    }
    let mut loops: Vec<_> = loops.into_iter().collect();
    // The map's order must not change the code.
    loops.sort_by(|a, b| a.0.cmp(&b.0));
    loops
}

fn fits_arithmetic(value: i64) -> bool {
    let magnitude = value.unsigned_abs();
    magnitude <= 0xfff || (magnitude & 0xfff == 0 && magnitude >> 12 <= 0xfff)
}

fn is_constant_that_fits(operand: &Operand) -> bool {
    matches!(operand, Operand::Immediate(Immediate::I64(v)) if fits_arithmetic(*v))
}

fn operands_to_hoist(instr: &mut Instruction) -> Vec<&mut Operand> {
    let (always, ty, lhs, rhs) = match instr {
        Instruction::IntBinary {
            op, ty, lhs, rhs, ..
        } => match op {
            // Not division: a constant divisor spares the backend its zero and overflow checks.
            IntBinOp::Mul => (true, ty, lhs, rhs),
            IntBinOp::Add | IntBinOp::Sub => (false, ty, lhs, rhs),
            _ => return Vec::new(),
        },
        Instruction::IntCmp { ty, lhs, rhs, .. } => (false, ty, lhs, rhs),
        _ => return Vec::new(),
    };
    if *ty != WORD {
        return Vec::new();
    }
    [lhs, rhs]
        .into_iter()
        .filter(|operand| always || !is_constant_that_fits(operand))
        .collect()
}

fn hoist_from_loop(
    func: &mut Function,
    head: &str,
    body: &HashSet<String>,
    next_id: &mut u32,
) -> bool {
    let predecessors = predecessors(func);
    let outside: Vec<&String> = predecessors
        .get(head)
        .map(|labels| {
            labels
                .iter()
                .filter(|label| !body.contains(*label))
                .collect()
        })
        .unwrap_or_default();
    let [preheader] = outside.as_slice() else {
        return false;
    };
    let preheader = (*preheader).clone();

    let mut homes: Vec<(i64, Register)> = Vec::new();
    for block in func.blocks.iter_mut().filter(|b| body.contains(&b.label)) {
        for located in &mut block.instructions {
            for operand in operands_to_hoist(&mut located.instr) {
                let Operand::Immediate(Immediate::I64(value)) = operand else {
                    continue;
                };
                let value = *value;
                let home = match homes.iter().find(|(known, _)| *known == value) {
                    Some((_, home)) => home.clone(),
                    None if homes.len() < MAX_PER_LOOP => {
                        let home = Register::Virtual(VirtualReg::gpr(*next_id));
                        *next_id += 1;
                        homes.push((value, home.clone()));
                        home
                    }
                    None => continue,
                };
                *operand = Operand::Register(home);
            }
        }
    }
    if homes.is_empty() {
        return false;
    }

    let Some(block) = func.blocks.iter_mut().find(|b| b.label == preheader) else {
        return false;
    };
    let before_terminator = block.instructions.len().saturating_sub(1);
    for (value, home) in homes {
        block.instructions.insert(
            before_terminator,
            Located::from(Instruction::Copy {
                ty: WORD,
                dst: home,
                src: Operand::Immediate(Immediate::I64(value)),
            }),
        );
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{Block, IntCmpOp, Signature, verify_function_in_pipeline};

    fn reg(id: u32) -> Register {
        Register::Virtual(VirtualReg::gpr(id))
    }

    fn counting_loop(step: i64, limit: i64) -> Function {
        let mut func = Function::new(Signature::new("count").with_return(WORD));
        let mut entry = Block::new("entry");
        entry.push(Instruction::Copy {
            ty: WORD,
            dst: reg(0),
            src: Operand::Immediate(Immediate::I64(0)),
        });
        entry.push(Instruction::Jmp {
            target: "body".to_string(),
        });
        let mut body = Block::new("body");
        body.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: WORD,
            dst: reg(1),
            lhs: Operand::Register(reg(0)),
            rhs: Operand::Immediate(Immediate::I64(step)),
        });
        body.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: WORD,
            dst: reg(0),
            lhs: Operand::Register(reg(0)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        body.push(Instruction::IntCmp {
            op: IntCmpOp::SLt,
            ty: WORD,
            dst: reg(2),
            lhs: Operand::Register(reg(0)),
            rhs: Operand::Immediate(Immediate::I64(limit)),
        });
        body.push(Instruction::Br {
            cond: reg(2),
            true_target: "body".to_string(),
            false_target: "done".to_string(),
        });
        let mut done = Block::new("done");
        done.push(Instruction::Ret {
            value: Some(Operand::Register(reg(1))),
        });
        func.add_block(entry);
        func.add_block(body);
        func.add_block(done);
        func
    }

    fn constants_in(func: &Function, label: &str) -> usize {
        let block = func.blocks.iter().find(|b| b.label == label).unwrap();
        block
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    &i.instr,
                    Instruction::Copy {
                        src: Operand::Immediate(_),
                        ..
                    }
                )
            })
            .count()
    }

    #[test]
    fn large_constants_move_ahead_of_the_loop() {
        let mut func = counting_loop(1103515245, 100000000);
        assert!(ConstantHoisting.apply(&mut func).unwrap());
        // The counter's own start, the multiplier and the limit.
        assert_eq!(constants_in(&func, "entry"), 3);
        verify_function_in_pipeline(&func).unwrap();
        let entry = func.blocks.iter().find(|b| b.label == "entry").unwrap();
        assert!(entry.terminator().is_some());
    }

    #[test]
    fn repeated_constant_shares_one_register() {
        let mut func = counting_loop(100000000, 100000000);
        ConstantHoisting.apply(&mut func).unwrap();
        assert_eq!(constants_in(&func, "entry"), 2);
    }

    #[test]
    fn small_compare_constant_stays_in_the_instruction() {
        let mut func = counting_loop(3, 4095);
        ConstantHoisting.apply(&mut func).unwrap();
        // Only the multiply needs a register.
        assert_eq!(constants_in(&func, "entry"), 2);
    }

    #[test]
    fn divisor_stays_a_constant() {
        let mut func = counting_loop(3, 4095);
        let body = func.blocks.iter_mut().find(|b| b.label == "body").unwrap();
        let Instruction::IntBinary { op, .. } = &mut body.instructions[0].instr else {
            panic!("the loop starts with a multiply");
        };
        *op = IntBinOp::SRem;
        assert!(!ConstantHoisting.apply(&mut func).unwrap());
    }

    #[test]
    fn loop_with_two_ways_in_is_left_alone() {
        let mut func = counting_loop(1103515245, 100000000);
        let entry = func.blocks.iter_mut().find(|b| b.label == "entry").unwrap();
        entry.instructions.pop();
        entry.push(Instruction::Br {
            cond: reg(0),
            true_target: "body".to_string(),
            false_target: "other".to_string(),
        });
        let mut other = Block::new("other");
        other.push(Instruction::Jmp {
            target: "body".to_string(),
        });
        func.add_block(other);
        assert!(!ConstantHoisting.apply(&mut func).unwrap());
    }
}
