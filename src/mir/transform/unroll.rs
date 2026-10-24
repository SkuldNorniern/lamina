use std::collections::{HashMap, HashSet};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};
use crate::mir::{
    Function, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register, ScalarType,
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
        for info in LoopInvariantCodeMotion.find_loops(func) {
            if let Some(shape) = analyze(func, &info) {
                let _ = constant_trip_count(func, &shape);
            }
        }
        Ok(false)
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
