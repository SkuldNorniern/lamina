use std::collections::{HashMap, HashSet};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};
use crate::mir::{
    Block, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register,
    ScalarType, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};

#[derive(Default)]
pub struct LoopUnrolling;

impl Transform for LoopUnrolling {
    fn name(&self) -> &'static str {
        "loop_unrolling"
    }
    fn description(&self) -> &'static str {
        "Unroll bounded counting loops with value renaming"
    }
    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }
    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }
    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;
        for _ in 0..16 {
            let candidate = LoopInvariantCodeMotion
                .find_loops(func)
                .iter()
                .find_map(|info| analyze(func, info));
            let Some(shape) = candidate else {
                break;
            };
            let count = constant_trip_count(func, &shape);
            if let Some(count) =
                count.filter(|n| *n <= 32 && shape.body.len().saturating_mul(*n as usize) <= 192)
            {
                let Some(mut next) = register_base(func) else {
                    break;
                };
                full_unroll(func, &shape, count, &mut next);
            } else {
                let Some(mut next) = register_base(func) else {
                    break;
                };
                partial_unroll(func, &shape, 8, &mut next);
            }
            changed = true;
        }
        Ok(changed)
    }
}

struct Shape {
    header: String,
    body: Vec<Instruction>,
    labels: HashSet<String>,
    iv: Register,
    end: Operand,
    inclusive: bool,
    entry: String,
}

fn analyze(func: &Function, info: &LoopInfo) -> Option<Shape> {
    if info.header.starts_with("__unroll_") {
        return None;
    }
    if info.blocks.len() < 2 || info.blocks.len() > 8 {
        return None;
    }
    let header = func.get_block(&info.header)?;
    let [
        Instruction::IntCmp {
            op,
            ty: MirType::Scalar(ScalarType::I64),
            dst,
            lhs: Operand::Register(iv),
            rhs: end,
        },
        Instruction::Br {
            cond,
            true_target,
            false_target,
        },
    ] = header.instructions.as_slice()
    else {
        return None;
    };
    if cond != dst || iv == dst {
        return None;
    }
    let (first, inclusive) = match op {
        IntCmpOp::SLt if !info.blocks.contains(false_target) => (true_target, false),
        IntCmpOp::SLe if !info.blocks.contains(false_target) => (true_target, true),
        IntCmpOp::SGe if !info.blocks.contains(true_target) => (false_target, false),
        IntCmpOp::SGt if !info.blocks.contains(true_target) => (false_target, true),
        _ => return None,
    };
    let outside: Vec<_> = func
        .blocks
        .iter()
        .filter(|b| {
            !info.blocks.contains(&b.label) && b.successors().contains(&info.header.as_str())
        })
        .collect();
    let [entry] = outside.as_slice() else {
        return None;
    };
    if entry.label.starts_with("__unroll_") {
        return None;
    }
    let mut labels = HashSet::from([info.header.clone()]);
    let mut body = Vec::new();
    let mut current = first.clone();
    while current != info.header {
        if !info.blocks.contains(&current) || !labels.insert(current.clone()) {
            return None;
        }
        let block = func.get_block(&current)?;
        let (Instruction::Jmp { target }, instructions) = block.instructions.split_last()? else {
            return None;
        };
        for i in instructions {
            if !matches!(
                i,
                Instruction::IntBinary {
                    ty: MirType::Scalar(ScalarType::I64),
                    ..
                } | Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::I64),
                    ..
                }
            ) {
                return None;
            }
            if i.def_reg()
                .into_iter()
                .chain(i.use_regs())
                .any(|r| !r.is_virtual())
            {
                return None;
            }
        }
        body.extend_from_slice(instructions);
        current = target.clone();
    }
    if labels != info.blocks || body.len() > 48 {
        return None;
    }
    for b in &func.blocks {
        if !labels.contains(&b.label)
            && b.successors()
                .iter()
                .any(|s| labels.contains(*s) && *s != info.header)
        {
            return None;
        }
    }
    let updates: Vec<_> = body.iter().filter(|i| i.def_reg() == Some(iv)).collect();
    let [
        Instruction::IntBinary {
            op: IntBinOp::Add,
            lhs: Operand::Register(src),
            rhs,
            ..
        },
    ] = updates.as_slice()
    else {
        return None;
    };
    if src != iv || rhs.integer_constant() != Some(1) {
        return None;
    }
    if body.iter().any(|i| {
        i.def_reg() == Some(dst) || end.as_register().is_some_and(|r| i.def_reg() == Some(r))
    }) {
        return None;
    }
    Some(Shape {
        header: info.header.clone(),
        body,
        labels,
        iv: iv.clone(),
        end: end.clone(),
        inclusive,
        entry: entry.label.clone(),
    })
}

fn register_base(func: &Function) -> Option<u32> {
    let next = func
        .blocks
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
        .checked_add(1)?;
    next.checked_add(100_000)?;
    Some(next)
}

fn fresh(next: &mut u32) -> Register {
    let reg = VirtualReg::gpr(*next).into();
    *next += 1;
    reg
}
fn imm(n: i64) -> Operand {
    Operand::Immediate(Immediate::I64(n))
}
fn binary(op: IntBinOp, dst: &Register, lhs: Operand, rhs: Operand) -> Instruction {
    Instruction::IntBinary {
        op,
        ty: MirType::Scalar(ScalarType::I64),
        dst: dst.clone(),
        lhs,
        rhs,
    }
}
fn compare(op: IntCmpOp, dst: &Register, lhs: Operand, rhs: Operand) -> Instruction {
    Instruction::IntCmp {
        op,
        ty: MirType::Scalar(ScalarType::I64),
        dst: dst.clone(),
        lhs,
        rhs,
    }
}
fn jump(target: &str) -> Instruction {
    Instruction::Jmp {
        target: target.into(),
    }
}
fn branch(cond: &Register, yes: &str, no: &str) -> Instruction {
    Instruction::Br {
        cond: cond.clone(),
        true_target: yes.into(),
        false_target: no.into(),
    }
}

fn partial_unroll(func: &mut Function, shape: &Shape, factor: usize, next: &mut u32) {
    let mut prefix = format!("__unroll_{}", shape.header);
    while func.blocks.iter().any(|b| b.label.starts_with(&prefix)) {
        prefix.push('_');
    }
    let label = |suffix: &str| format!("{prefix}_{suffix}");
    let dispatch = label("dispatch");
    let overflow = label("overflow");
    let setup = label("setup");
    let procheck = label("procheck");
    let probody = label("probody");
    let maincheck = label("maincheck");
    let mainbody = label("mainbody");
    let trip = fresh(next);
    let remainder = fresh(next);
    let flag = fresh(next);
    let mut guard = func
        .get_block(&shape.header)
        .map(|b| b.instructions.clone())
        .unwrap_or_default();
    if let Some(Instruction::Br {
        true_target,
        false_target,
        ..
    }) = guard.last_mut()
    {
        let target = if shape.labels.contains(true_target) {
            true_target
        } else {
            false_target
        };
        *target = if shape.inclusive {
            overflow.clone()
        } else {
            setup.clone()
        };
    }
    for b in &mut func.blocks {
        if shape.labels.contains(&b.label) {
            continue;
        }
        for i in &mut b.instructions {
            match i {
                Instruction::Jmp { target } if *target == shape.header => {
                    *target = dispatch.clone()
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if *true_target == shape.header {
                        *true_target = dispatch.clone();
                    }
                    if *false_target == shape.header {
                        *false_target = dispatch.clone();
                    }
                }
                _ => {}
            }
        }
    }
    let mut add_block = |name: String, instructions: Vec<Instruction>| {
        func.blocks.push(Block {
            label: name,
            instructions,
        })
    };
    add_block(dispatch, guard);
    if shape.inclusive {
        add_block(
            overflow,
            vec![
                compare(IntCmpOp::Eq, &flag, shape.end.clone(), imm(i64::MAX)),
                branch(&flag, &shape.header, &setup),
            ],
        );
    }
    let mut setup_code = vec![binary(
        IntBinOp::Sub,
        &trip,
        shape.end.clone(),
        Operand::Register(shape.iv.clone()),
    )];
    if shape.inclusive {
        setup_code.push(binary(IntBinOp::Add, &trip, trip.clone().into(), imm(1)));
    }
    setup_code.extend([
        binary(
            IntBinOp::And,
            &remainder,
            trip.clone().into(),
            imm(factor as i64 - 1),
        ),
        binary(
            IntBinOp::Sub,
            &trip,
            trip.clone().into(),
            remainder.clone().into(),
        ),
        jump(&procheck),
    ]);
    add_block(setup, setup_code);
    add_block(
        procheck.clone(),
        vec![
            compare(IntCmpOp::Ne, &flag, remainder.clone().into(), imm(0)),
            branch(&flag, &probody, &maincheck),
        ],
    );
    let mut code = renamed_body(shape, 1, next);
    code.extend([
        binary(IntBinOp::Sub, &remainder, remainder.clone().into(), imm(1)),
        jump(&procheck),
    ]);
    add_block(probody, code);
    add_block(
        maincheck.clone(),
        vec![
            compare(IntCmpOp::Ne, &flag, trip.clone().into(), imm(0)),
            branch(&flag, &mainbody, &shape.header),
        ],
    );
    let mut code = renamed_body(shape, factor, next);
    code.extend([
        binary(
            IntBinOp::Sub,
            &trip,
            trip.clone().into(),
            imm(factor as i64),
        ),
        jump(&maincheck),
    ]);
    add_block(mainbody, code);
}

fn renamed_body(shape: &Shape, count: usize, next: &mut u32) -> Vec<Instruction> {
    let first_fresh = *next;
    let mut values: HashMap<Register, Operand> = HashMap::new();
    let mut definitions = Vec::new();
    let mut code = Vec::new();
    let only_increment = shape
        .body
        .iter()
        .all(|i| i.def_reg() == Some(&shape.iv) || !i.use_regs().contains(&&shape.iv));
    for _ in 0..count {
        for i in &shape.body {
            let Some(dst) = i.def_reg() else {
                continue;
            };
            if !definitions.contains(dst) {
                definitions.push(dst.clone());
            }
            if only_increment && dst == &shape.iv {
                continue;
            }
            let resolve = |o: &Operand| {
                o.as_register()
                    .and_then(|r| values.get(r))
                    .cloned()
                    .unwrap_or_else(|| o.clone())
            };
            let (op, lhs, rhs) = match i {
                Instruction::IntBinary { op, lhs, rhs, .. } => (*op, resolve(lhs), resolve(rhs)),
                Instruction::Copy { src, .. } => (IntBinOp::Add, resolve(src), imm(0)),
                _ => continue,
            };
            if matches!(
                op,
                IntBinOp::Add | IntBinOp::Sub | IntBinOp::Or | IntBinOp::Xor
            ) && rhs.integer_constant() == Some(0)
            {
                values.insert(dst.clone(), lhs);
            } else {
                let result = fresh(next);
                code.push(binary(op, &result, lhs, rhs));
                values.insert(dst.clone(), result.into());
            }
        }
    }
    if only_increment && count > 0 {
        let result = fresh(next);
        code.push(binary(
            IntBinOp::Add,
            &result,
            shape.iv.clone().into(),
            imm(count as i64),
        ));
        values.insert(shape.iv.clone(), result.into());
    }
    let mut copies = Vec::new();
    for dst in definitions {
        if let Some(mut src) = values.remove(&dst) {
            if let Operand::Register(r) = &src {
                if !matches!(r, Register::Virtual(v) if v.id >= first_fresh) {
                    let saved = fresh(next);
                    code.push(Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: saved.clone(),
                        src,
                    });
                    src = saved.into();
                }
            }
            copies.push(Instruction::Copy {
                ty: MirType::Scalar(ScalarType::I64),
                dst,
                src,
            });
        }
    }
    code.extend(copies);
    code
}

fn full_unroll(func: &mut Function, shape: &Shape, count: u64, next: &mut u32) {
    let mut instructions = renamed_body(shape, count as usize, next);
    if let Some(header) = func.get_block_mut(&shape.header) {
        instructions.extend_from_slice(&header.instructions);
        if let Some(Instruction::Br {
            true_target,
            false_target,
            ..
        }) = instructions.last()
        {
            let exit = if shape.labels.contains(true_target) {
                false_target
            } else {
                true_target
            }
            .clone();
            instructions.pop();
            instructions.push(Instruction::Jmp { target: exit });
        }
        header.instructions = instructions;
    }
    func.blocks
        .retain(|b| b.label == shape.header || !shape.labels.contains(&b.label));
}

fn constant_trip_count(func: &Function, shape: &Shape) -> Option<u64> {
    let entry = func.get_block(&shape.entry)?;
    let mut values: HashMap<Register, i64> = HashMap::new();
    for i in &entry.instructions {
        let value = |o: &Operand| {
            o.integer_constant()
                .or_else(|| o.as_register().and_then(|r| values.get(r).copied()))
        };
        let result = match i {
            Instruction::Copy { src, .. } => value(src),
            Instruction::IntBinary {
                op, ty, lhs, rhs, ..
            } => value(lhs)
                .zip(value(rhs))
                .and_then(|(a, b)| op.evaluate(*ty, a, b)),
            _ => None,
        };
        if let Some(dst) = i.def_reg() {
            values.remove(dst);
            if let Some(v) = result {
                values.insert(dst.clone(), v);
            }
        }
    }
    let start = *values.get(&shape.iv)?;
    let end = shape
        .end
        .integer_constant()
        .or_else(|| values.get(shape.end.as_register()?).copied())?;
    if shape.inclusive && end == i64::MAX {
        return None;
    }
    let count = i128::from(end) - i128::from(start) + i128::from(shape.inclusive);
    u64::try_from(count.max(0)).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{mir::codegen::from_ir, parser::parse_module};

    fn counting(start: i64, end: &str, comparison: &str) -> Function {
        let source = format!(
            "fn @f(i64 %n) -> i64 {{
entry:
%i = add.i64 {start}, 0
%a = add.i64 0, 0
%b = add.i64 1, 0
jmp test
test:
%c = {comparison}.i64 %i, {end}
br %c, body, exit
body:
%t = add.i64 %a, %b
%a = add.i64 %b, 0
%b = add.i64 %t, 0
%i = add.i64 %i, 1
jmp test
exit:
ret.i64 %b
}}"
        );
        from_ir(&parse_module(&source).unwrap(), "unroll")
            .unwrap()
            .functions
            .remove("f")
            .unwrap()
    }

    #[test]
    fn renaming_removes_rotation_copies_between_iterations() {
        let func = counting(0, "%n", "lt");
        let shape = analyze(&func, &LoopInvariantCodeMotion.find_loops(&func)[0]).unwrap();
        let mut next = register_base(&func).unwrap();
        let body = renamed_body(&shape, 8, &mut next);
        let arithmetic: Vec<_> = body
            .iter()
            .filter(|i| matches!(i, Instruction::IntBinary { .. }))
            .collect();
        assert_eq!(arithmetic.len(), 9);
        assert!(body[..8].iter().all(|i| matches!(
            i,
            Instruction::IntBinary {
                op: IntBinOp::Add,
                ..
            }
        )));
        let mut values = HashSet::new();
        for i in &body[..8] {
            assert!(values.insert(i.def_reg().unwrap()));
        }
    }

    #[test]
    fn partial_unrolling_preserves_remainders_and_signed_spans() {
        for cmp in ["lt", "le"] {
            let original = counting(-3, "%n", cmp);
            let mut func = original.clone();
            assert!(LoopUnrolling.apply(&mut func).unwrap());
            assert!(!LoopUnrolling.apply(&mut func).unwrap());
            crate::mir::verify::verify_function_in_pipeline(&func).unwrap();
            for n in [-9, -3, -2, 3, 4, 5, 6, 13, 29, 30] {
                assert_eq!(
                    crate::runtime::executor::interpret_mir_function(&func, &[n]).unwrap(),
                    crate::runtime::executor::interpret_mir_function(&original, &[n]).unwrap(),
                    "{cmp} {n}"
                );
            }
        }
    }

    #[test]
    fn full_unrolling_preserves_zero_and_nonzero_trips() {
        for count in [0, 1, 7, 8, 9, 16, 32] {
            let mut func = counting(3, &(3 + count).to_string(), "lt");
            let before = crate::runtime::executor::interpret_mir_function(&func, &[0]).unwrap();
            assert!(LoopUnrolling.apply(&mut func).unwrap());
            assert!(LoopInvariantCodeMotion.find_loops(&func).is_empty());
            assert_eq!(
                crate::runtime::executor::interpret_mir_function(&func, &[0]).unwrap(),
                before
            );
        }
    }

    #[test]
    fn trip_counts_respect_start_polarity_and_empty_ranges() {
        for (start, end, cmp, expected) in [
            (3, "7", "lt", 4),
            (3, "7", "le", 5),
            (7, "3", "lt", 0),
            (7, "7", "lt", 0),
            (7, "7", "le", 1),
            (-5, "2", "lt", 7),
        ] {
            let func = counting(start, end, cmp);
            let loops = LoopInvariantCodeMotion.find_loops(&func);
            let shape = analyze(&func, &loops[0]).unwrap();
            assert_eq!(constant_trip_count(&func, &shape), Some(expected));
        }
        let func = counting(0, "9223372036854775807", "le");
        let shape = analyze(&func, &LoopInvariantCodeMotion.find_loops(&func)[0]).unwrap();
        assert_eq!(constant_trip_count(&func, &shape), None);
    }
}
