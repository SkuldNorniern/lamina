use std::collections::{HashMap, HashSet};

use crate::mir::{
    Block, Function, Immediate, Instruction, IntBinOp, MirType, Operand, Register, ScalarType,
    VirtualReg,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, calculate_dominators,
    },
};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};

#[derive(Default)]
pub struct InductionStrengthReduction;

impl Transform for InductionStrengthReduction {
    fn name(&self) -> &'static str {
        "induction_strength_reduction"
    }

    fn description(&self) -> &'static str {
        "Replace loop induction products with wrapping recurrences"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut loops = LoopInvariantCodeMotion.find_loops(func);
        loops.sort_by(|a, b| (a.blocks.len(), &a.header).cmp(&(b.blocks.len(), &b.header)));
        let mut changed = false;
        for original in loops {
            if let Some(info) = LoopInvariantCodeMotion
                .find_loops(func)
                .into_iter()
                .find(|info| info.header == original.header)
            {
                while reduce_loop(func, &info) {
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
}

fn integer_constant(operand: &Operand) -> Option<i64> {
    match operand {
        Operand::Immediate(Immediate::I8(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I16(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I32(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I64(v)) => Some(*v),
        _ => None,
    }
}

fn invariant(operand: &Operand, defs: &HashMap<Register, Vec<(usize, usize)>>) -> bool {
    match operand {
        Operand::Immediate(_) => true,
        Operand::Register(reg) => reg.is_virtual() && !defs.contains_key(reg),
    }
}

/// Registers a block defines whose every use sits in that same block, so a clone of the
/// block can rename them without touching anything else.
fn block_local_defs(func: &Function, block: &Block) -> Vec<Register> {
    let mut defined: Vec<Register> = Vec::new();
    for instr in &block.instructions {
        if let Some(Register::Virtual(v)) = instr.def_reg()
            && !defined.iter().any(|d| d == &Register::Virtual(*v))
        {
            defined.push(Register::Virtual(*v));
        }
    }
    defined.retain(|reg| {
        func.blocks.iter().all(|other| {
            other.label == block.label
                || other.instructions.iter().all(|instr| {
                    !instr.use_regs().contains(&reg)
                        && instr.def_reg() != Some(reg)
                        && !matches!(instr, Instruction::StackAddr { slot, .. }
                            if Register::Virtual(*slot) == *reg)
                })
        })
    });
    defined
}

/// Replace every mention of `from` with `to` inside one block.
fn rename_in_block(block: &mut Block, from: &Register, to: &Register) {
    for instr in &mut block.instructions {
        super::inline::ModuleInlining::map_registers(instr, &mut |reg: &Register| {
            if reg == from { to.clone() } else { reg.clone() }
        });
    }
}

fn reduce_loop(func: &mut Function, info: &LoopInfo) -> bool {
    let Some(header_idx) = func.blocks.iter().position(|b| b.label == info.header) else {
        return false;
    };
    if func.entry == info.header {
        return false;
    }
    let dominators = calculate_dominators(func);
    let dominates = |a: &str, b: &str| dominators.get(b).is_some_and(|ds| ds.contains(a));
    let latches: Vec<_> = func
        .blocks
        .iter()
        .filter(|b| {
            info.blocks.contains(&b.label) && b.successors().contains(&info.header.as_str())
        })
        .map(|b| b.label.clone())
        .collect();
    let nested: HashSet<_> = LoopInvariantCodeMotion
        .find_loops(func)
        .into_iter()
        .filter(|other| {
            other.header != info.header
                && info.blocks.contains(&other.header)
                && other.blocks != info.blocks
        })
        .flat_map(|other| other.blocks)
        .collect();
    let mut defs: HashMap<Register, Vec<(usize, usize)>> = HashMap::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        if info.blocks.contains(&block.label) {
            for (ii, instr) in block.instructions.iter().enumerate() {
                if let Some(reg) = instr.def_reg() {
                    defs.entry(reg.clone()).or_default().push((bi, ii));
                }
            }
        }
    }
    let Some(mut next_id) = func
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
        .checked_add(1)
    else {
        return false;
    };
    let mut init = Vec::new();
    let mut product_block = None;
    let mut replacements = HashMap::new();
    let mut removed = HashSet::new();
    let mut increments: HashMap<(usize, usize), Vec<Instruction>> = HashMap::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        if !info.blocks.contains(&block.label)
            || bi == header_idx
            || product_block.is_some_and(|chosen| chosen != bi)
        {
            continue;
        }
        for (ii, instr) in block.instructions.iter().enumerate() {
            let Instruction::IntBinary {
                op: IntBinOp::Mul,
                dst,
                ty,
                lhs,
                rhs,
            } = instr
            else {
                continue;
            };
            if !matches!(
                ty,
                MirType::Scalar(
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                )
            ) {
                continue;
            }
            let (iv, multiplier) = match (lhs, rhs) {
                (Operand::Register(iv), m) if invariant(m, &defs) => (iv, m),
                (m, Operand::Register(iv)) if invariant(m, &defs) => (iv, m),
                _ => continue,
            };
            let mut offset = None;
            let iv = if let Some(locations) = defs.get(iv)
                && locations.len() == 1
                && let (ab, ai) = locations[0]
                && ab == bi
                && ai < ii
                && let Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: at,
                    lhs: al,
                    rhs: ar,
                    ..
                } = &block.instructions[ai]
                && at == ty
            {
                match (al, ar) {
                    (Operand::Register(base), addend) | (addend, Operand::Register(base))
                        if base != iv && invariant(addend, &defs) =>
                    {
                        offset = Some(addend.clone());
                        base
                    }
                    _ => iv,
                }
            } else {
                iv
            };
            let Some(locations) = defs.get(iv) else {
                continue;
            };
            if !iv.is_virtual() || locations.len() != 1 {
                continue;
            }
            // Shared induction registers can extend recurrences across sibling loops.
            if func.blocks.iter().any(|b| {
                !info.blocks.contains(&b.label)
                    && !dominates(&b.label, &block.label)
                    && b.instructions
                        .iter()
                        .any(|i| i.def_reg() == Some(iv) && i.use_regs().contains(&iv))
            }) {
                continue;
            }
            let (ub, ui) = locations[0];
            let update_block = &func.blocks[ub];
            let Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: update_ty,
                lhs: ul,
                rhs: ur,
                ..
            } = &update_block.instructions[ui]
            else {
                continue;
            };
            let step = match (ul, ur) {
                (Operand::Register(r), c) | (c, Operand::Register(r))
                    if r == iv && invariant(c, &defs) =>
                {
                    c
                }
                _ => continue,
            };
            if update_ty != ty
                || ub == header_idx
                || nested.contains(&update_block.label)
                || latches.is_empty()
                || !latches.iter().all(|l| dominates(&update_block.label, l))
                || !dominates(&block.label, &update_block.label)
                || (bi == ub && ii >= ui)
                || (bi != ub && dominates(&update_block.label, &block.label))
            {
                continue;
            }
            // No path may revisit the product after updating the induction variable.
            let mut pending: Vec<_> = update_block
                .successors()
                .into_iter()
                .filter(|s| *s != info.header)
                .map(str::to_owned)
                .collect();
            let mut seen = HashSet::new();
            while let Some(label) = pending.pop() {
                if info.blocks.contains(&label)
                    && label != info.header
                    && seen.insert(label.clone())
                    && let Some(b) = func.get_block(&label)
                {
                    pending.extend(
                        b.successors()
                            .into_iter()
                            .filter(|s| *s != info.header)
                            .map(str::to_owned),
                    );
                }
            }
            if seen.contains(&block.label) || seen.contains(&update_block.label) {
                continue;
            }
            let Some(following_id) = next_id.checked_add(2) else {
                continue;
            };
            product_block = Some(bi);
            let affine = match block.instructions.get(ii + 1) {
                Some(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    dst: ad,
                    ty: at,
                    lhs: al,
                    rhs: ar,
                }) if ad == dst && at == ty => match (al, ar) {
                    (Operand::Register(r), a) | (a, Operand::Register(r))
                        if r == dst && invariant(a, &defs) =>
                    {
                        Some(a.clone())
                    }
                    _ => None,
                },
                _ => None,
            };
            let reuse = defs
                .get(dst)
                .is_some_and(|locations| locations.len() == 1 + usize::from(affine.is_some()))
                && func.blocks.iter().enumerate().all(|(use_bi, b)| {
                    b.instructions.iter().enumerate().all(|(use_ii, i)| {
                        !i.use_regs().contains(&dst)
                            || (info.blocks.contains(&b.label)
                                && dominates(&block.label, &b.label)
                                && !seen.contains(&b.label)
                                && (use_bi != bi || use_ii > ii)
                                && (use_bi != ub || use_ii < ui))
                    })
                });
            let accumulator: Register = if reuse {
                dst.clone()
            } else {
                VirtualReg::gpr(next_id).into()
            };
            let delta: Register = VirtualReg::gpr(next_id + 1).into();
            next_id = following_id;
            let initial = if let Some(offset) = offset {
                init.push(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    dst: accumulator.clone(),
                    ty: *ty,
                    lhs: Operand::Register(iv.clone()),
                    rhs: offset,
                });
                Operand::Register(accumulator.clone())
            } else {
                Operand::Register(iv.clone())
            };
            init.push(Instruction::IntBinary {
                op: IntBinOp::Mul,
                dst: accumulator.clone(),
                ty: *ty,
                lhs: initial,
                rhs: multiplier.clone(),
            });
            if let Some(offset) = affine {
                init.push(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    dst: accumulator.clone(),
                    ty: *ty,
                    lhs: Operand::Register(accumulator.clone()),
                    rhs: offset,
                });
                removed.insert((bi, ii + 1));
            }
            let delta_operand = if matches!(
                step,
                Operand::Immediate(
                    Immediate::I8(1) | Immediate::I16(1) | Immediate::I32(1) | Immediate::I64(1)
                )
            ) {
                multiplier.clone()
            } else {
                let calculation = Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    dst: delta.clone(),
                    ty: *ty,
                    lhs: multiplier.clone(),
                    rhs: step.clone(),
                };
                if integer_constant(step).is_some_and(|v| v > 0 && (v & (v - 1)) == 0) {
                    increments.entry((ub, ui)).or_default().push(calculation);
                } else {
                    init.push(calculation);
                }
                Operand::Register(delta)
            };
            if reuse {
                removed.insert((bi, ii));
            } else {
                replacements.insert(
                    (bi, ii),
                    Instruction::Copy {
                        dst: dst.clone(),
                        ty: *ty,
                        src: Operand::Register(accumulator.clone()),
                    },
                );
            }
            increments
                .entry((ub, ui))
                .or_default()
                .push(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    dst: accumulator.clone(),
                    ty: *ty,
                    lhs: Operand::Register(accumulator),
                    rhs: delta_operand,
                });
        }
    }
    if init.is_empty() {
        return false;
    }
    let unique_label = |suffix: &str| {
        let mut label = format!("{}_{}", info.header, suffix);
        while func.get_block(&label).is_some() {
            label.push('_');
        }
        label
    };
    let product_label = func.blocks[product_block.unwrap_or(header_idx)]
        .label
        .clone();
    let Some(prefix) = entry_prefix(func, info, &product_label) else {
        return false;
    };
    if func.blocks.iter().any(|b| {
        !info.blocks.contains(&b.label)
            && b.successors()
                .iter()
                .any(|s| info.blocks.contains(*s) && !prefix.contains(*s) && *s != product_label)
    }) {
        return false;
    }
    let prefix = if func.blocks.iter().any(|b| {
        !info.blocks.contains(&b.label) && b.successors().iter().any(|s| prefix.contains(*s))
    }) {
        prefix
    } else {
        HashSet::new()
    };
    let init_label = unique_label("sr_init");
    let labels: HashMap<_, _> = prefix
        .iter()
        .map(|label| {
            let mut fresh = format!("{label}_sr_first");
            while func.get_block(&fresh).is_some() {
                fresh.push('_');
            }
            (label.clone(), fresh)
        })
        .collect();
    let mut first_blocks = Vec::new();
    for block in &func.blocks {
        if let Some(label) = labels.get(&block.label) {
            let mut first = block.clone();
            first.label = label.clone();
            // Peeling clones a block verbatim, which would give a second definition to
            // every value it defines. For one that never escapes the block, that extra
            // definition only hides it from later passes, so the copy gets its own.
            for local in block_local_defs(func, block) {
                let fresh: Register = VirtualReg::gpr(next_id).into();
                next_id = match next_id.checked_add(1) {
                    Some(id) => id,
                    None => return false,
                };
                rename_in_block(&mut first, &local, &fresh);
            }
            redirect(&mut first, &|target| {
                if target == product_label {
                    Some(init_label.clone())
                } else {
                    labels.get(target).cloned()
                }
            });
            first_blocks.push(first);
        }
    }
    init.push(Instruction::Jmp {
        target: product_label.clone(),
    });
    for (bi, block) in func.blocks.iter_mut().enumerate() {
        let mut instructions = Vec::new();
        for (ii, instr) in block.instructions.iter().enumerate() {
            if removed.contains(&(bi, ii)) {
                continue;
            }
            instructions.push(
                replacements
                    .remove(&(bi, ii))
                    .unwrap_or_else(|| instr.clone()),
            );
            if let Some(updates) = increments.remove(&(bi, ii)) {
                instructions.extend(updates);
            }
        }
        block.instructions = instructions;
        if !info.blocks.contains(&block.label) {
            redirect(block, &|target| {
                if target == product_label {
                    Some(init_label.clone())
                } else {
                    labels.get(target).cloned()
                }
            });
        }
    }
    let init_block = Block {
        label: init_label,
        instructions: init,
    };
    let product_idx = product_block.unwrap_or(header_idx);
    if product_idx >= header_idx {
        func.blocks.insert(product_idx, init_block);
        func.blocks.splice(header_idx..header_idx, first_blocks);
    } else {
        func.blocks.splice(header_idx..header_idx, first_blocks);
        func.blocks.insert(product_idx, init_block);
    }
    true
}

fn redirect(block: &mut Block, replace: &impl Fn(&str) -> Option<String>) {
    for instr in &mut block.instructions {
        let target = |label: &mut String| {
            if let Some(new) = replace(label) {
                *label = new;
            }
        };
        match instr {
            Instruction::Jmp { target: label } => target(label),
            Instruction::Br {
                true_target,
                false_target,
                ..
            } => {
                target(true_target);
                target(false_target);
            }
            Instruction::Switch { cases, default, .. } => {
                target(default);
                for (_, label) in cases {
                    target(label);
                }
            }
            _ => {}
        }
    }
}

fn entry_prefix(func: &Function, info: &LoopInfo, product: &str) -> Option<HashSet<String>> {
    fn visit(
        func: &Function,
        info: &LoopInfo,
        product: &str,
        label: &str,
        active: &mut HashSet<String>,
        done: &mut HashSet<String>,
    ) -> Option<()> {
        if label == product || !info.blocks.contains(label) || done.contains(label) {
            return Some(());
        }
        if !active.insert(label.to_owned()) || active.len() + done.len() > 12 {
            return None;
        }
        for next in func.get_block(label)?.successors() {
            visit(func, info, product, next, active, done)?;
        }
        active.remove(label);
        done.insert(label.to_owned());
        Some(())
    }
    let mut done = HashSet::new();
    visit(
        func,
        info,
        product,
        &info.header,
        &mut HashSet::new(),
        &mut done,
    )?;
    Some(done)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, Immediate, IntCmpOp};

    fn reg(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }
    fn operand(id: u32) -> Operand {
        Operand::Register(reg(id))
    }
    fn constant(n: i64) -> Operand {
        Operand::Immediate(Immediate::I64(n))
    }
    fn binary(op: IntBinOp, dst: u32, ty: MirType, lhs: Operand, rhs: Operand) -> Instruction {
        Instruction::IntBinary {
            op,
            dst: reg(dst),
            ty,
            lhs,
            rhs,
        }
    }

    fn example(ty: MirType, trips: i64) -> Function {
        let start = match ty {
            MirType::Scalar(ScalarType::I8) => 250,
            MirType::Scalar(ScalarType::I16) => 65_530,
            MirType::Scalar(ScalarType::I32) => 4_294_967_290,
            _ => 120,
        };
        FunctionBuilder::new("recurrence")
            .returns(ty)
            .block("entry")
            .instr(Instruction::Copy {
                dst: reg(0),
                ty,
                src: constant(start),
            })
            .instr(Instruction::Copy {
                dst: reg(1),
                ty,
                src: constant(0),
            })
            .instr(Instruction::Copy {
                dst: reg(2),
                ty,
                src: constant(0),
            })
            .instr(Instruction::Jmp {
                target: "header".into(),
            })
            .block("header")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::ULt,
                dst: reg(3),
                ty,
                lhs: operand(1),
                rhs: constant(trips),
            })
            .instr(Instruction::Br {
                cond: reg(3),
                true_target: "body".into(),
                false_target: "exit".into(),
            })
            .block("body")
            .instr(binary(IntBinOp::Mul, 4, ty, operand(0), constant(7)))
            .instr(binary(IntBinOp::Add, 4, ty, operand(4), constant(11)))
            .instr(binary(IntBinOp::Add, 2, ty, operand(2), operand(4)))
            .instr(binary(IntBinOp::Add, 0, ty, operand(0), constant(3)))
            .instr(binary(IntBinOp::Add, 1, ty, operand(1), constant(1)))
            .instr(Instruction::Jmp {
                target: "header".into(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(operand(2)),
            })
            .build()
    }

    fn execute(func: &Function) -> (u64, usize, Vec<u64>) {
        let mut values = HashMap::<Register, u64>::new();
        let mut label = func.entry.clone();
        let mut multiplies = 0;
        let mut products = Vec::new();
        for _ in 0..10000 {
            let block = func.get_block(&label).unwrap();
            for instr in &block.instructions {
                let read = |o: &Operand| match o {
                    Operand::Register(r) => *values.get(r).expect("defined operand"),
                    Operand::Immediate(Immediate::I64(v)) => *v as u64,
                    _ => panic!("unsupported operand"),
                };
                let mask = |ty: &MirType, v: u64| {
                    let bits = ty.size_bytes() * 8;
                    if bits == 64 {
                        v
                    } else {
                        v & ((1u64 << bits) - 1)
                    }
                };
                match instr {
                    Instruction::Copy { dst, ty, src } => {
                        let v = mask(ty, read(src));
                        values.insert(dst.clone(), v);
                    }
                    Instruction::IntBinary {
                        op,
                        dst,
                        ty,
                        lhs,
                        rhs,
                    } => {
                        let (a, b) = (read(lhs), read(rhs));
                        let v = match op {
                            IntBinOp::Add => a.wrapping_add(b),
                            IntBinOp::Mul => {
                                multiplies += 1;
                                a.wrapping_mul(b)
                            }
                            _ => panic!("unsupported operation"),
                        };
                        values.insert(dst.clone(), mask(ty, v));
                    }
                    Instruction::IntCmp { dst, lhs, rhs, .. } => {
                        let v = u64::from(read(lhs) < read(rhs));
                        values.insert(dst.clone(), v);
                    }
                    Instruction::Jmp { target } => {
                        label = target.clone();
                        break;
                    }
                    Instruction::Br {
                        cond,
                        true_target,
                        false_target,
                    } => {
                        label = if values[cond] != 0 {
                            true_target
                        } else {
                            false_target
                        }
                        .clone();
                        break;
                    }
                    Instruction::Ret { value } => {
                        return (read(value.as_ref().unwrap()), multiplies, products);
                    }
                    _ => panic!("unsupported instruction"),
                }
                if instr.def_reg() == Some(&reg(2)) {
                    products.push(values[&reg(2)]);
                }
            }
        }
        panic!("execution did not terminate")
    }

    #[test]
    fn reduces_induction_product_and_affine_expression() {
        let mut func = example(MirType::i64(), 20);
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        assert_eq!(actual.1, 2);
        assert!(
            !func
                .get_block("body")
                .unwrap()
                .instructions
                .iter()
                .any(|i| matches!(
                    i,
                    Instruction::IntBinary {
                        op: IntBinOp::Mul,
                        ..
                    }
                ))
        );
    }

    #[test]
    fn recurrence_wraps_at_original_width() {
        for ty in [MirType::i8(), MirType::i16(), MirType::i32()] {
            let mut func = example(ty, 100);
            let expected = execute(&func);
            assert!(InductionStrengthReduction.apply(&mut func).unwrap());
            let actual = execute(&func);
            assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        }
    }

    #[test]
    fn zero_trip_loop_skips_initial_products() {
        let mut func = example(MirType::i64(), 0);
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        assert_eq!(execute(&func), expected);
        assert_eq!(expected.1, 0);
    }

    #[test]
    fn guarded_body_skips_products_and_undefined_induction_input() {
        let mut func = example(MirType::i64(), 10);
        func.get_block_mut("entry").unwrap().instructions.remove(0);
        func.get_block_mut("entry").unwrap().instructions.insert(
            0,
            Instruction::Copy {
                dst: reg(5),
                ty: MirType::i64(),
                src: constant(0),
            },
        );
        if let Instruction::Br { true_target, .. } = func
            .get_block_mut("header")
            .unwrap()
            .instructions
            .last_mut()
            .unwrap()
        {
            *true_target = "gate".into();
        }
        func.blocks.push(Block {
            label: "gate".into(),
            instructions: vec![Instruction::Br {
                cond: reg(5),
                true_target: "body".into(),
                false_target: "exit".into(),
            }],
        });
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        assert_eq!(execute(&func), expected);
        assert_eq!(expected.1, 0);
    }

    #[test]
    fn invariant_register_multiplier_and_step() {
        let mut func = example(MirType::i8(), 100);
        for (id, value) in [(6, 7), (7, 3)] {
            func.get_block_mut("entry").unwrap().instructions.insert(
                0,
                Instruction::Copy {
                    dst: reg(id),
                    ty: MirType::i8(),
                    src: constant(value),
                },
            );
        }
        let body = func.get_block_mut("body").unwrap();
        if let Instruction::IntBinary { rhs, .. } = &mut body.instructions[0] {
            *rhs = operand(6);
        }
        if let Instruction::IntBinary { rhs, .. } = &mut body.instructions[3] {
            *rhs = operand(7);
        }
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        assert_eq!(actual.1, 2);
    }

    #[test]
    fn preserves_product_used_after_loop() {
        let mut func = example(MirType::i64(), 10);
        *func
            .get_block_mut("exit")
            .unwrap()
            .instructions
            .last_mut()
            .unwrap() = Instruction::Ret {
            value: Some(operand(4)),
        };
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
    }

    fn two_level(trips: i64, escapes: bool) -> Function {
        let ty = MirType::i64();
        let mut func = example(ty, trips);
        func.get_block_mut("entry").unwrap().instructions.insert(
            0,
            Instruction::Copy {
                dst: reg(5),
                ty,
                src: constant(99),
            },
        );
        for (id, value) in [(7, 7), (8, 5), (9, 3)] {
            func.get_block_mut("entry").unwrap().instructions.insert(
                0,
                Instruction::Copy {
                    dst: reg(id),
                    ty,
                    src: constant(value),
                },
            );
        }
        let body = func.get_block_mut("body").unwrap();
        body.instructions[0] = binary(IntBinOp::Mul, 4, ty, operand(0), operand(7));
        body.instructions[3] = binary(IntBinOp::Add, 0, ty, operand(0), operand(9));
        body.instructions[1] = binary(IntBinOp::Add, 6, ty, operand(4), constant(11));
        body.instructions
            .insert(2, binary(IntBinOp::Mul, 5, ty, operand(6), operand(8)));
        body.instructions[3] = binary(IntBinOp::Add, 2, ty, operand(2), operand(5));
        if escapes {
            func.get_block_mut("exit").unwrap().instructions[0] = Instruction::Ret {
                value: Some(operand(5)),
            };
        }
        func
    }

    #[test]
    fn reduces_two_level_products_to_fixed_point() {
        for (trips, escapes) in [(20, false), (20, true), (0, false), (0, true)] {
            let mut func = two_level(trips, escapes);
            let expected = execute(&func);
            assert!(InductionStrengthReduction.apply(&mut func).unwrap());
            let actual = execute(&func);
            assert_eq!((actual.0, actual.2), (expected.0, expected.2));
            assert_eq!(actual.1, if trips == 0 { 0 } else { 4 });
            assert!(
                !func
                    .get_block("body")
                    .unwrap()
                    .instructions
                    .iter()
                    .any(|i| {
                        matches!(
                            i,
                            Instruction::IntBinary {
                                op: IntBinOp::Mul,
                                ..
                            }
                        )
                    })
            );
            assert!(!InductionStrengthReduction.apply(&mut func).unwrap());
        }
    }

    fn two_level_blocks() -> Function {
        let ty = MirType::i64();
        let mut func = example(ty, 20);
        let body = func.get_block_mut("body").unwrap();
        let mut tail = body.instructions.split_off(2);
        tail[0] = binary(IntBinOp::Add, 2, ty, operand(2), operand(5));
        tail.insert(0, binary(IntBinOp::Mul, 5, ty, operand(4), constant(5)));
        body.instructions.push(Instruction::Jmp {
            target: "product".into(),
        });
        func.blocks.push(Block {
            label: "product".into(),
            instructions: tail,
        });
        func
    }

    #[test]
    fn dce_removes_unused_two_level_recurrences() {
        let mut func = two_level_blocks();
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        assert!(super::super::DeadCodeElimination.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        assert_eq!(
            func.get_block("body").unwrap().instructions,
            vec![Instruction::Jmp {
                target: "product".into(),
            }]
        );
        let updates: Vec<_> = func
            .get_block("product")
            .unwrap()
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instruction::IntBinary { dst, .. } => Some(dst.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(updates.len(), 3);
        assert!(updates.contains(&reg(1)));
        assert!(updates.contains(&reg(2)));
        assert!(!super::super::DeadCodeElimination.apply(&mut func).unwrap());
    }

    #[test]
    fn reduces_dependent_product_in_later_block() {
        let mut func = two_level_blocks();
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        assert_eq!(actual.1, 4);
        assert!(!InductionStrengthReduction.apply(&mut func).unwrap());
    }

    #[test]
    fn repeated_peeling_keeps_local_comparisons_unique() {
        let mut func = two_level_blocks();
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        for instr in func.blocks.iter().flat_map(|b| &b.instructions) {
            if let Instruction::IntCmp { dst, .. } = instr {
                assert_eq!(
                    func.blocks
                        .iter()
                        .flat_map(|b| &b.instructions)
                        .filter(|i| i.def_reg() == Some(dst))
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn rejects_multiple_induction_definitions() {
        let mut func = example(MirType::i64(), 10);
        func.get_block_mut("body").unwrap().instructions.insert(
            4,
            binary(IntBinOp::Add, 0, MirType::i64(), operand(0), constant(1)),
        );
        assert!(!InductionStrengthReduction.apply(&mut func).unwrap());
    }

    #[test]
    fn rejects_width_mismatch() {
        let mut func = example(MirType::i8(), 10);
        if let Instruction::IntBinary { ty, .. } =
            &mut func.get_block_mut("body").unwrap().instructions[3]
        {
            *ty = MirType::i64();
        }
        assert!(!InductionStrengthReduction.apply(&mut func).unwrap());
    }

    #[test]
    fn handles_multiple_back_edges_after_shared_update() {
        let mut func = example(MirType::i64(), 10);
        *func
            .get_block_mut("body")
            .unwrap()
            .instructions
            .last_mut()
            .unwrap() = Instruction::Br {
            cond: reg(3),
            true_target: "left".into(),
            false_target: "right".into(),
        };
        for label in ["left", "right"] {
            func.blocks.push(Block {
                label: label.into(),
                instructions: vec![Instruction::Jmp {
                    target: "header".into(),
                }],
            });
        }
        let expected = execute(&func);
        assert!(InductionStrengthReduction.apply(&mut func).unwrap());
        let actual = execute(&func);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
    }

    #[test]
    fn rejects_back_edge_bypassing_update() {
        let mut func = example(MirType::i64(), 10);
        func.get_block_mut("body").unwrap().instructions.truncate(3);
        func.get_block_mut("body")
            .unwrap()
            .instructions
            .push(Instruction::Br {
                cond: reg(3),
                true_target: "update".into(),
                false_target: "header".into(),
            });
        func.blocks.push(Block {
            label: "update".into(),
            instructions: vec![
                binary(IntBinOp::Add, 0, MirType::i64(), operand(0), constant(1)),
                Instruction::Jmp {
                    target: "header".into(),
                },
            ],
        });
        assert!(!InductionStrengthReduction.apply(&mut func).unwrap());
    }
}
