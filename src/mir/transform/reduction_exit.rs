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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{mir::FunctionBuilder, runtime::executor::interpret_mir_function};

    fn reg(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }
    fn op(id: u32) -> Operand {
        Operand::Register(reg(id))
    }
    fn add(ty: &MirType, dst: u32, lhs: Operand, rhs: Operand) -> Instruction {
        binary(IntBinOp::Add, ty, &reg(dst), lhs, rhs)
    }
    fn jmp(target: &str) -> Instruction {
        Instruction::Jmp {
            target: target.into(),
        }
    }
    fn cmp_sge(ty: &MirType, dst: u32) -> Instruction {
        Instruction::IntCmp {
            op: IntCmpOp::SGe,
            ty: *ty,
            dst: reg(dst),
            lhs: op(0),
            rhs: op(1),
        }
    }

    #[derive(Clone)]
    struct Variant {
        step: i64,
        guard: bool,
        second_entry: bool,
        extra_body: Option<Instruction>,
    }

    impl Variant {
        fn standard() -> Self {
            Self {
                step: 1,
                guard: true,
                second_entry: false,
                extra_body: None,
            }
        }
    }

    fn linear_loop(ty: MirType, variant: &Variant) -> Function {
        let mut body = Vec::new();
        if let Some(extra) = &variant.extra_body {
            body.push(extra.clone());
        }
        body.extend([
            add(&ty, 4, op(4), op(2)),
            add(&ty, 0, op(0), immediate(&ty, variant.step)),
            add(&ty, 2, op(2), op(3)),
            jmp("test"),
        ]);
        let entry_branch = if variant.guard {
            vec![
                cmp_sge(&ty, 5),
                Instruction::Br {
                    cond: reg(5),
                    true_target: "exit".into(),
                    false_target: "pre".into(),
                },
            ]
        } else {
            vec![jmp("pre")]
        };
        let mut builder = FunctionBuilder::new("linear")
            .param(reg(0), ty)
            .param(reg(1), ty)
            .param(reg(2), ty)
            .param(reg(3), ty)
            .param(reg(4), ty)
            .returns(ty)
            .block("entry");
        for instr in entry_branch {
            builder = builder.instr(instr);
        }
        builder = builder.block("pre").instr(jmp("body")).block("body");
        for instr in body {
            builder = builder.instr(instr);
        }
        builder = builder
            .block("test")
            .instr(cmp_sge(&ty, 6))
            .instr(Instruction::Br {
                cond: reg(6),
                true_target: "exit".into(),
                false_target: "body".into(),
            });
        if variant.second_entry {
            builder = builder.block("alt").instr(jmp("body"));
        }
        builder
            .block("exit")
            .instr(binary(
                IntBinOp::Mul,
                &ty,
                &reg(7),
                op(2),
                immediate(&ty, 3),
            ))
            .instr(add(&ty, 8, op(4), op(7)))
            .instr(binary(
                IntBinOp::Mul,
                &ty,
                &reg(9),
                op(0),
                immediate(&ty, 7),
            ))
            .instr(add(&ty, 10, op(8), op(9)))
            .instr(Instruction::Ret {
                value: Some(op(10)),
            })
            .build()
    }

    fn run(func: &Function, args: &[i64; 5]) -> Option<i64> {
        interpret_mir_function(func, args).unwrap()
    }

    fn differential(ty: MirType, variant: &Variant, cases: &[[i64; 5]]) -> bool {
        let original = linear_loop(ty, variant);
        let mut transformed = original.clone();
        ReductionExit.apply(&mut transformed).unwrap();
        for args in cases {
            assert_eq!(run(&transformed, args), run(&original, args), "{args:?}");
        }
        transformed.get_block("body").is_none()
    }

    fn accumulators() -> [[i64; 3]; 3] {
        [[-7, 13, 5], [100, -99, -3], [0, 1, 0]]
    }

    #[test]
    fn fires_on_guarded_loop_for_every_i8_span() {
        let mut cases = Vec::new();
        for j0 in i64::from(i8::MIN)..=i64::from(i8::MAX) {
            for end in i64::from(i8::MIN)..=i64::from(i8::MAX) {
                let [p, d, t] = accumulators()[0];
                cases.push([j0, end, p, d, t]);
            }
        }
        assert!(differential(MirType::i8(), &Variant::standard(), &cases));
    }

    #[test]
    fn fires_for_every_accumulator_at_i8_trip_counts_that_wrap() {
        let mut cases = Vec::new();
        for [p, d, t] in accumulators() {
            for j0 in [-128, -1, 0, 1, 27, 127] {
                for end in [j0, j0 + 1, j0 + 100, j0 + 127, j0 + 128, 127] {
                    if (-128..=127).contains(&end) {
                        cases.push([j0, end, p, d, t]);
                    }
                }
            }
        }
        assert!(differential(MirType::i8(), &Variant::standard(), &cases));
    }

    #[test]
    fn fires_at_wider_widths_and_extreme_bounds() {
        let i16_cases: Vec<[i64; 5]> = [
            (-32768, 32767),
            (-32768, -32768),
            (-1, 32767),
            (-5000, 30000),
            (32766, 32767),
            (0, 1),
            (100, 101),
            (-3, -4),
        ]
        .into_iter()
        .flat_map(|(j0, end)| {
            accumulators()
                .into_iter()
                .map(move |[p, d, t]| [j0, end, p, d, t])
        })
        .collect();
        assert!(differential(
            MirType::i16(),
            &Variant::standard(),
            &i16_cases
        ));

        let i64_cases: Vec<[i64; 5]> = [
            (-5, 7),
            (0, 0),
            (3, 3),
            (7, 2),
            (i64::MIN, i64::MIN + 3),
            (i64::MAX - 5, i64::MAX),
            (-1000, 1000),
        ]
        .into_iter()
        .flat_map(|(j0, end)| {
            [[i64::MAX - 3, (1i64 << 40) + 7, 9], [-7, 13, 5]]
                .into_iter()
                .map(move |[p, d, t]| [j0, end, p, d, t])
        })
        .collect();
        assert!(differential(
            MirType::i64(),
            &Variant::standard(),
            &i64_cases
        ));
    }

    #[test]
    fn refuses_non_unit_and_negative_steps() {
        let variant = Variant {
            step: 2,
            ..Variant::standard()
        };
        let cases = [[0, 5, -7, 13, 5], [5, 3, 1, 2, 3], [9, 9, 4, 4, 4]];
        assert!(!differential(MirType::i64(), &variant, &cases));
        let variant = Variant {
            step: -1,
            ..Variant::standard()
        };
        let cases = [[5, 3, 1, 2, 3], [9, 9, 4, 4, 4]];
        assert!(!differential(MirType::i64(), &variant, &cases));
    }

    #[test]
    fn refuses_when_body_observes_the_accumulator() {
        let variant = Variant {
            extra_body: Some(add(&MirType::i64(), 11, op(4), immediate_operand(1))),
            ..Variant::standard()
        };
        let cases = [[0, 5, -7, 13, 5], [5, 3, 1, 2, 3]];
        assert!(!differential(MirType::i64(), &variant, &cases));
    }

    #[test]
    fn refuses_body_that_may_trap() {
        let variant = Variant {
            extra_body: Some(binary(
                IntBinOp::UDiv,
                &MirType::i64(),
                &reg(11),
                op(4),
                op(3),
            )),
            ..Variant::standard()
        };
        let cases = [[0, 5, -7, 13, 5], [5, 3, 1, 2, 3]];
        assert!(!differential(MirType::i64(), &variant, &cases));
    }

    #[test]
    fn refuses_entry_without_a_proven_bound() {
        let variant = Variant {
            guard: false,
            ..Variant::standard()
        };
        let cases = [[5, 3, -7, 13, 5], [0, 5, 1, 2, 3], [9, 9, 4, 4, 4]];
        assert!(!differential(MirType::i64(), &variant, &cases));
    }

    #[test]
    fn refuses_a_second_entry_edge() {
        let variant = Variant {
            second_entry: true,
            ..Variant::standard()
        };
        let cases = [[0, 5, -7, 13, 5], [5, 3, 1, 2, 3]];
        assert!(!differential(MirType::i64(), &variant, &cases));
    }

    fn immediate_operand(value: i64) -> Operand {
        Operand::Immediate(Immediate::I64(value))
    }
}
