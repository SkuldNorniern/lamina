use std::collections::HashSet;

use crate::mir::{
    Block, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register,
    ScalarType, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};

#[derive(Default)]
pub struct ReductionExit;

impl Transform for ReductionExit {
    fn name(&self) -> &'static str {
        "reduction_exit"
    }

    fn description(&self) -> &'static str {
        "Replace guarded unit-step linear sum loops with closed forms"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;
        loop {
            let shape = LoopInvariantCodeMotion
                .find_loops(func)
                .iter()
                .find_map(|info| match_loop(func, info));
            let Some(shape) = shape else { break };
            let Some(mut next) = fresh_register_base(func) else {
                break;
            };
            rewrite(func, &shape, &mut next);
            changed = true;
        }
        Ok(changed)
    }
}

/// A rotated loop `body -> test -> body` where `body` adds the summand into the total,
/// bumps the summand by an invariant step and increments the induction variable by one,
/// and `test` exits when `iv >= end` (signed). The loop is entered from exactly one
/// predecessor whose path proves `iv < end` on entry.
struct Shape {
    entry: String,
    body: String,
    test: String,
    exit: String,
    ty: MirType,
    total: Register,
    summand: Register,
    delta: Register,
    iv: Register,
    end: Register,
}

fn match_loop(func: &Function, info: &LoopInfo) -> Option<Shape> {
    if info.blocks.len() != 2 {
        return None;
    }
    let mut labels = info.blocks.iter().map(String::as_str);
    let (a, b) = (labels.next()?, labels.next()?);
    match_pair(func, info, a, b).or_else(|| match_pair(func, info, b, a))
}

fn match_pair(func: &Function, info: &LoopInfo, body: &str, test: &str) -> Option<Shape> {
    let body_block = func.get_block(body)?;
    let test_block = func.get_block(test)?;

    let [cmp, br] = test_block.instructions.as_slice() else {
        return None;
    };
    let Instruction::IntCmp {
        op: IntCmpOp::SGe,
        ty,
        dst: flag,
        lhs: Operand::Register(iv),
        rhs: Operand::Register(end),
    } = cmp
    else {
        return None;
    };
    let Instruction::Br {
        cond,
        true_target: exit,
        false_target,
    } = br
    else {
        return None;
    };
    if cond != flag || false_target != body || info.blocks.contains(exit.as_str()) {
        return None;
    }
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64)
    ) {
        return None;
    }

    let (last, updates) = body_block.instructions.split_last()?;
    let Instruction::Jmp { target } = last else {
        return None;
    };
    if target != test {
        return None;
    }
    let [first, second, third] = updates else {
        return None;
    };

    let mut inc = None;
    let mut adds = Vec::new();
    for (index, instr) in [first, second, third].into_iter().enumerate() {
        let Instruction::IntBinary {
            op: IntBinOp::Add,
            dst,
            ty: add_ty,
            lhs,
            rhs,
        } = instr
        else {
            return None;
        };
        if add_ty != ty {
            return None;
        }
        let other = match (lhs, rhs) {
            (Operand::Register(r), other) if r == dst => other,
            (other, Operand::Register(r)) if r == dst => other,
            _ => return None,
        };
        match other {
            Operand::Immediate(_) if other.integer_constant() == Some(1) => {
                if inc.replace(dst.clone()).is_some() {
                    return None;
                }
            }
            Operand::Register(other) if other != dst => {
                adds.push((index, dst.clone(), other.clone()));
            }
            _ => return None,
        }
    }
    let inc_dst = inc?;
    let [(i1, d1, o1), (i2, d2, o2)] = adds.as_slice() else {
        return None;
    };
    let ((sum_index, total, summand), (bump_index, bumped, delta)) =
        if o1 == d2 && o2 != d1 && o2 != d2 {
            ((*i1, d1.clone(), o1.clone()), (*i2, d2.clone(), o2.clone()))
        } else if o2 == d1 && o1 != d1 && o1 != d2 {
            ((*i2, d2.clone(), o2.clone()), (*i1, d1.clone(), o1.clone()))
        } else {
            return None;
        };
    if sum_index > bump_index || inc_dst != *iv || bumped != summand {
        return None;
    }
    if [&total, &summand, iv].contains(&flag) || total == *iv || summand == *iv {
        return None;
    }
    for invariant in [end, &delta] {
        if [&total, &summand, iv, flag].contains(&invariant) {
            return None;
        }
    }

    let loop_blocks = [body, test];
    let uses_flag_elsewhere = func
        .blocks
        .iter()
        .any(|b| b.label != test && b.instructions.iter().any(|i| i.use_regs().contains(&flag)));
    if uses_flag_elsewhere {
        return None;
    }

    let test_preds = predecessors(func, test);
    if test_preds != [body] {
        return None;
    }
    let outside: Vec<&str> = predecessors(func, body)
        .into_iter()
        .filter(|p| *p != test)
        .collect();
    let [entry] = outside.as_slice() else {
        return None;
    };
    if loop_blocks.contains(entry) {
        return None;
    }
    let entry_block = func.get_block(entry)?;
    if !matches!(entry_block.terminator(), Some(Instruction::Jmp { target }) if target == body) {
        return None;
    }
    if !guarded_entry(func, entry, &info.blocks, iv, end, ty) {
        return None;
    }

    Some(Shape {
        entry: (*entry).to_owned(),
        body: body.to_owned(),
        test: test.to_owned(),
        exit: exit.clone(),
        ty: *ty,
        total,
        summand,
        delta,
        iv: iv.clone(),
        end: end.clone(),
    })
}

fn predecessors<'a>(func: &'a Function, label: &str) -> Vec<&'a str> {
    func.blocks
        .iter()
        .filter(|b| b.successors().contains(&label))
        .map(|b| b.label.as_str())
        .collect()
}

fn defines(instr: &Instruction, reg: &Register) -> bool {
    instr.def_reg() == Some(reg)
}

fn guarded_entry(
    func: &Function,
    entry: &str,
    loop_blocks: &HashSet<String>,
    iv: &Register,
    end: &Register,
    ty: &MirType,
) -> bool {
    let mut current = entry.to_owned();
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return false;
        }
        let Some(block) = func.get_block(&current) else {
            return false;
        };
        if block
            .instructions
            .iter()
            .any(|i| defines(i, iv) || defines(i, end))
        {
            return false;
        }
        let preds = predecessors(func, &current);
        let [pred] = preds.as_slice() else {
            return false;
        };
        if loop_blocks.contains(*pred) {
            return false;
        }
        let Some(pred_block) = func.get_block(pred) else {
            return false;
        };
        match pred_block.terminator() {
            Some(Instruction::Jmp { target }) if *target == current => {
                current = (*pred).to_owned();
            }
            Some(Instruction::Br {
                cond,
                true_target,
                false_target,
            }) => {
                let continues_on_true = *true_target == current && *false_target != current;
                let continues_on_false = *false_target == current && *true_target != current;
                let Some(cmp_index) = pred_block
                    .instructions
                    .iter()
                    .rposition(|i| i.def_reg() == Some(cond))
                else {
                    return false;
                };
                let Instruction::IntCmp {
                    op,
                    ty: cmp_ty,
                    dst,
                    lhs,
                    rhs,
                } = &pred_block.instructions[cmp_index]
                else {
                    return false;
                };
                if cmp_ty != ty
                    || dst == iv
                    || dst == end
                    || *lhs != Operand::Register(iv.clone())
                    || *rhs != Operand::Register(end.clone())
                {
                    return false;
                }
                let holds = (continues_on_false && *op == IntCmpOp::SGe)
                    || (continues_on_true && *op == IntCmpOp::SLt);
                if !holds {
                    return false;
                }
                return pred_block.instructions[cmp_index + 1..]
                    .iter()
                    .all(|i| !defines(i, iv) && !defines(i, end));
            }
            _ => return false,
        }
    }
}

fn fresh_register_base(func: &Function) -> Option<u32> {
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
        .unwrap_or(0)
        .checked_add(1)
}

fn immediate(ty: &MirType, value: i64) -> Operand {
    Operand::Immediate(match ty {
        MirType::Scalar(ScalarType::I8) => Immediate::I8(value as i8),
        MirType::Scalar(ScalarType::I16) => Immediate::I16(value as i16),
        MirType::Scalar(ScalarType::I32) => Immediate::I32(value as i32),
        _ => Immediate::I64(value),
    })
}

fn binary(op: IntBinOp, ty: &MirType, dst: &Register, lhs: Operand, rhs: Operand) -> Instruction {
    Instruction::IntBinary {
        op,
        ty: *ty,
        dst: dst.clone(),
        lhs,
        rhs,
    }
}

/// With `n = end - iv` (exact as an unsigned value because `iv < end`), the body runs `n`
/// times and the sum adds `n*P + C(n,2)*D` to the total. `C(n,2)` halves whichever of
/// `n` and `n-1` is even before the wrapping multiply.
fn rewrite(func: &mut Function, shape: &Shape, next: &mut u32) {
    let ty = shape.ty;
    let mut fresh = || {
        let reg: Register = VirtualReg::gpr(*next).into();
        *next += 1;
        reg
    };
    let n = fresh();
    let n_minus_one = fresh();
    let parity = fresh();
    let flip = fresh();
    let halved_n = fresh();
    let halved_pred = fresh();
    let pair = fresh();
    let linear = fresh();
    let pair_delta = fresh();
    let step_delta = fresh();
    let partial = fresh();

    let reg = Operand::Register;
    let one = immediate(&ty, 1);
    let sequence = [
        binary(
            IntBinOp::Sub,
            &ty,
            &n,
            reg(shape.end.clone()),
            reg(shape.iv.clone()),
        ),
        binary(
            IntBinOp::Sub,
            &ty,
            &n_minus_one,
            reg(n.clone()),
            one.clone(),
        ),
        binary(IntBinOp::And, &ty, &parity, reg(n.clone()), one.clone()),
        binary(IntBinOp::Xor, &ty, &flip, reg(parity.clone()), one),
        binary(IntBinOp::LShr, &ty, &halved_n, reg(n.clone()), reg(flip)),
        binary(
            IntBinOp::LShr,
            &ty,
            &halved_pred,
            reg(n_minus_one),
            reg(parity),
        ),
        binary(IntBinOp::Mul, &ty, &pair, reg(halved_n), reg(halved_pred)),
        binary(
            IntBinOp::Mul,
            &ty,
            &linear,
            reg(n.clone()),
            reg(shape.summand.clone()),
        ),
        binary(
            IntBinOp::Mul,
            &ty,
            &pair_delta,
            reg(pair),
            reg(shape.delta.clone()),
        ),
        binary(
            IntBinOp::Mul,
            &ty,
            &step_delta,
            reg(n),
            reg(shape.delta.clone()),
        ),
        binary(
            IntBinOp::Add,
            &ty,
            &partial,
            reg(shape.total.clone()),
            reg(linear),
        ),
        binary(
            IntBinOp::Add,
            &ty,
            &shape.total,
            reg(partial),
            reg(pair_delta),
        ),
        binary(
            IntBinOp::Add,
            &ty,
            &shape.summand,
            reg(shape.summand.clone()),
            reg(step_delta),
        ),
        Instruction::Copy {
            ty,
            dst: shape.iv.clone(),
            src: reg(shape.end.clone()),
        },
        Instruction::Jmp {
            target: shape.exit.clone(),
        },
    ];

    if let Some(entry) = func.get_block_mut(&shape.entry) {
        entry.instructions.pop();
        entry.instructions.extend(sequence);
    }
    func.blocks
        .retain(|b: &Block| b.label != shape.body && b.label != shape.test);
}
