use std::collections::{HashMap, HashSet};

use crate::mir::{
    Block, Function, Immediate, Instruction, IntBinOp, MirType, Operand, Register, ScalarType,
    VirtualReg,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel, calculate_dominators,
    },
};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};

mod polynomial;

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
                while reduce_loop(func, &info)? {
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

fn following_offset(
    block: &Block,
    ii: usize,
    dst: &Register,
    ty: &MirType,
    defs: &HashMap<Register, Vec<(usize, usize)>>,
) -> Option<Operand> {
    match block.instructions.get(ii + 1) {
        Some(Instruction::IntBinary {
            op: IntBinOp::Add,
            dst: ad,
            ty: at,
            lhs: al,
            rhs: ar,
        }) if ad == dst && at == ty => match (al, ar) {
            (Operand::Register(r), a) | (a, Operand::Register(r))
                if r == dst && invariant(a, defs) =>
            {
                Some(a.clone())
            }
            _ => None,
        },
        _ => None,
    }
}

struct Legality<'a> {
    func: &'a Function,
    info: &'a LoopInfo,
    header_idx: usize,
    dominators: &'a HashMap<String, HashSet<String>>,
    latches: &'a [String],
    nested: &'a HashSet<String>,
}

impl Legality<'_> {
    fn update(
        &self,
        iv: &Register,
        product: (usize, usize),
        update: (usize, usize),
    ) -> Option<HashSet<String>> {
        let dominates = |a: &str, b: &str| self.dominators.get(b).is_some_and(|ds| ds.contains(a));
        let (bi, ii) = product;
        let (ub, ui) = update;
        let block = &self.func.blocks[bi];
        let update_block = &self.func.blocks[ub];
        if self.func.blocks.iter().any(|b| {
            !self.info.blocks.contains(&b.label)
                && !dominates(&b.label, &block.label)
                && b.instructions
                    .iter()
                    .any(|i| i.def_reg() == Some(iv) && i.use_regs().contains(&iv))
        }) || ub == self.header_idx
            || self.nested.contains(&update_block.label)
            || self.latches.is_empty()
            || !self
                .latches
                .iter()
                .all(|l| dominates(&update_block.label, l))
            || !dominates(&block.label, &update_block.label)
            || (bi == ub && ii >= ui)
            || (bi != ub && dominates(&update_block.label, &block.label))
        {
            return None;
        }
        let mut pending: Vec<_> = update_block
            .successors()
            .into_iter()
            .filter(|s| *s != self.info.header)
            .map(str::to_owned)
            .collect();
        let mut seen = HashSet::new();
        while let Some(label) = pending.pop() {
            if self.info.blocks.contains(&label)
                && label != self.info.header
                && seen.insert(label.clone())
                && let Some(b) = self.func.get_block(&label)
            {
                pending.extend(
                    b.successors()
                        .into_iter()
                        .filter(|s| *s != self.info.header)
                        .map(str::to_owned),
                );
            }
        }
        if seen.contains(&block.label) || seen.contains(&update_block.label) {
            return None;
        }
        Some(seen)
    }
}

#[allow(clippy::too_many_arguments)]
fn reusable(
    func: &Function,
    info: &LoopInfo,
    defs: &HashMap<Register, Vec<(usize, usize)>>,
    dominators: &HashMap<String, HashSet<String>>,
    dst: &Register,
    product: (usize, usize),
    update: (usize, usize),
    seen: &HashSet<String>,
    affine: bool,
) -> bool {
    let (bi, ii) = product;
    let (ub, ui) = update;
    let block = &func.blocks[bi];
    defs.get(dst)
        .is_some_and(|locations| locations.len() == 1 + usize::from(affine))
        && func.blocks.iter().enumerate().all(|(use_bi, b)| {
            b.instructions.iter().enumerate().all(|(use_ii, i)| {
                !i.use_regs().contains(&dst)
                    || (info.blocks.contains(&b.label)
                        && dominators
                            .get(&b.label)
                            .is_some_and(|ds| ds.contains(&block.label))
                        && !seen.contains(&b.label)
                        && (use_bi != bi || use_ii > ii)
                        && (use_bi != ub || use_ii < ui))
            })
        })
}

fn reduce_loop(func: &mut Function, info: &LoopInfo) -> Result<bool, TransformError> {
    let Some(header_idx) = func.blocks.iter().position(|b| b.label == info.header) else {
        return Ok(false);
    };
    if func.entry == info.header {
        return Ok(false);
    }
    let dominators = calculate_dominators(func);
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
        return Ok(false);
    };
    let legality = Legality {
        func,
        info,
        header_idx,
        dominators: &dominators,
        latches: &latches,
        nested: &nested,
    };
    let mut quadratic = false;
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
            let mut analysis = polynomial::Analysis::new(func, info, &defs, (bi, ii), *ty);
            if let Some(polynomial) = analysis.product(lhs, rhs) {
                let Some((_, update, _)) = polynomial.updates.first() else {
                    continue;
                };
                let update = *update;
                if !polynomial
                    .updates
                    .iter()
                    .all(|(reg, position, _)| legality.update(reg, (bi, ii), *position).is_some())
                {
                    continue;
                }
                let Some(seen) = legality.update(&polynomial.updates[0].0, (bi, ii), update) else {
                    continue;
                };
                let affine = following_offset(block, ii, dst, ty, &defs);
                let reuse = reusable(
                    func,
                    info,
                    &defs,
                    &dominators,
                    dst,
                    (bi, ii),
                    update,
                    &seen,
                    affine.is_some(),
                );
                let mut fresh = next_id;
                let Some(after_state) = fresh.checked_add(2) else {
                    continue;
                };
                let accumulator: Register = if reuse {
                    dst.clone()
                } else {
                    VirtualReg::gpr(fresh).into()
                };
                let delta: Register = VirtualReg::gpr(fresh + 1).into();
                fresh = after_state;
                let mut initializers = Vec::new();
                let Some([p, d, e]) = analysis.emit(&polynomial, &mut fresh, &mut initializers)
                else {
                    continue;
                };
                initializers.push(Instruction::Copy {
                    dst: accumulator.clone(),
                    ty: *ty,
                    src: p,
                });
                if let Some(offset) = affine {
                    initializers.push(Instruction::IntBinary {
                        op: IntBinOp::Add,
                        dst: accumulator.clone(),
                        ty: *ty,
                        lhs: Operand::Register(accumulator.clone()),
                        rhs: offset,
                    });
                    removed.insert((bi, ii + 1));
                }
                initializers.push(Instruction::Copy {
                    dst: delta.clone(),
                    ty: *ty,
                    src: d,
                });
                quadratic = true;
                init.extend(initializers);
                next_id = fresh;
                product_block = Some(bi);
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
                increments.entry(update).or_default().extend([
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        dst: accumulator.clone(),
                        ty: *ty,
                        lhs: Operand::Register(accumulator),
                        rhs: Operand::Register(delta.clone()),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        dst: delta.clone(),
                        ty: *ty,
                        lhs: Operand::Register(delta),
                        rhs: e,
                    },
                ]);
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
            if update_ty != ty {
                continue;
            }
            let Some(seen) = legality.update(iv, (bi, ii), (ub, ui)) else {
                continue;
            };
            let Some(following_id) = next_id.checked_add(2) else {
                continue;
            };
            product_block = Some(bi);
            let affine = following_offset(block, ii, dst, ty, &defs);
            let reuse = reusable(
                func,
                info,
                &defs,
                &dominators,
                dst,
                (bi, ii),
                (ub, ui),
                &seen,
                affine.is_some(),
            );
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
        return Ok(false);
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
        return Ok(false);
    };
    if func.blocks.iter().any(|b| {
        !info.blocks.contains(&b.label)
            && b.successors()
                .iter()
                .any(|s| info.blocks.contains(*s) && !prefix.contains(*s) && *s != product_label)
    }) {
        return Ok(false);
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
                    None => return Ok(false),
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
    if quadratic {
        super::DeadCodeElimination.apply(func)?;
    }
    Ok(true)
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
    use crate::{
        mir::{FunctionBuilder, Immediate, IntCmpOp},
        runtime::executor::interpret_mir_function,
    };

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
    fn quadratic(ty: MirType, trips: i64, a: i64, b: i64, square: bool) -> Function {
        let mut func = example(ty, trips);
        func.get_block_mut("entry").unwrap().instructions.insert(
            0,
            Instruction::Copy {
                dst: reg(6),
                ty,
                src: constant(13),
            },
        );
        let body = func.get_block_mut("body").unwrap();
        body.instructions[0] = binary(
            IntBinOp::Mul,
            4,
            ty,
            operand(0),
            operand(if square { 0 } else { 6 }),
        );
        body.instructions[3] = binary(IntBinOp::Add, 0, ty, operand(0), constant(a));
        body.instructions
            .insert(4, binary(IntBinOp::Add, 6, ty, operand(6), constant(b)));
        func
    }

    fn equivalent(mut func: Function, reduce: bool) -> Function {
        let oracle = interpret_mir_function(&func, &[]).unwrap();
        let expected = execute(&func);
        assert_eq!(InductionStrengthReduction.apply(&mut func).unwrap(), reduce);
        let actual = execute(&func);
        assert_eq!(interpret_mir_function(&func, &[]).unwrap(), oracle);
        assert_eq!((actual.0, actual.2), (expected.0, expected.2));
        func
    }

    #[test]
    fn quadratic_differences_preserve_every_observed_sum_at_each_width() {
        for ty in [
            MirType::i8(),
            MirType::i16(),
            MirType::i32(),
            MirType::i64(),
        ] {
            for trips in [0, 1, 2, 19, 100] {
                for (a, b) in [(1, 1), (-3, 7), (5, -2), (127, 65), (0, -1)] {
                    for square in [false, true] {
                        let original = quadratic(ty, trips, a, b, square);
                        let expected = execute(&original);
                        let reduced = equivalent(original, true);
                        assert!(execute(&reduced).1 <= if trips == 0 { 0 } else { 4 });
                        assert_eq!(execute(&reduced).2, expected.2);
                        assert!(
                            !reduced
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
                    }
                }
            }
        }
    }

    #[test]
    fn quadratic_sums_offsets_and_invariant_steps_preserve_observations() {
        let ty = MirType::i8();
        let mut func = quadratic(ty, 45, -7, 11, false);
        let entry = func.get_block_mut("entry").unwrap();
        for (id, value) in [(12, -7), (13, 11), (14, 93)] {
            entry.instructions.insert(
                0,
                Instruction::Copy {
                    dst: reg(id),
                    ty,
                    src: constant(value),
                },
            );
        }
        let body = func.get_block_mut("body").unwrap();
        body.instructions[3] = binary(IntBinOp::Add, 0, ty, operand(0), operand(12));
        body.instructions[4] = binary(IntBinOp::Add, 6, ty, operand(6), operand(13));
        body.instructions[0] = binary(IntBinOp::Mul, 4, ty, operand(11), operand(6));
        body.instructions
            .insert(0, binary(IntBinOp::Add, 11, ty, operand(10), operand(14)));
        body.instructions
            .insert(0, binary(IntBinOp::Add, 10, ty, operand(0), operand(6)));
        equivalent(func, true);
    }

    #[test]
    fn quadratic_rewrite_preserves_escaping_factors_and_product() {
        for trips in [0, 1, 2, 37] {
            for escaped in [0, 6, 4] {
                let ty = MirType::i64();
                let mut func = quadratic(ty, trips, -3, 8, false);
                func.get_block_mut("entry").unwrap().instructions.insert(
                    0,
                    Instruction::Copy {
                        dst: reg(4),
                        ty,
                        src: constant(99),
                    },
                );
                func.get_block_mut("exit").unwrap().instructions.insert(
                    0,
                    binary(IntBinOp::Add, 2, ty, operand(2), operand(escaped)),
                );
                equivalent(func, true);
            }
        }
    }

    #[test]
    fn quadratic_rewrite_preserves_uses_before_and_after_factor_updates() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 27, 5, -3, false);
        let body = func.get_block_mut("body").unwrap();
        body.instructions
            .insert(3, binary(IntBinOp::Add, 2, ty, operand(2), operand(0)));
        body.instructions
            .insert(5, binary(IntBinOp::Add, 2, ty, operand(2), operand(0)));
        body.instructions
            .insert(7, binary(IntBinOp::Add, 2, ty, operand(2), operand(6)));
        equivalent(func, true);
    }

    #[test]
    fn quadratic_rewrite_removes_only_unused_factor_state() {
        let ty = MirType::i64();
        let func = equivalent(quadratic(ty, 20, 3, 7, false), true);
        let body = func.get_block("body").unwrap();
        assert!(
            !body
                .instructions
                .iter()
                .any(|i| i.def_reg() == Some(&reg(0)) || i.def_reg() == Some(&reg(6)))
        );
        let mut func = quadratic(ty, 20, 3, 7, false);
        func.get_block_mut("exit")
            .unwrap()
            .instructions
            .insert(0, binary(IntBinOp::Add, 2, ty, operand(2), operand(6)));
        let func = equivalent(func, true);
        assert!(
            func.get_block("body")
                .unwrap()
                .instructions
                .iter()
                .any(|i| i.def_reg() == Some(&reg(6)))
        );
    }

    #[test]
    fn quadratic_guarded_entry_skips_uninitialized_factors() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, 7, false);
        func.get_block_mut("entry")
            .unwrap()
            .instructions
            .retain(|i| i.def_reg() != Some(&reg(0)) && i.def_reg() != Some(&reg(6)));
        func.get_block_mut("entry").unwrap().instructions.insert(
            0,
            Instruction::Copy {
                dst: reg(5),
                ty,
                src: constant(0),
            },
        );
        if let Instruction::Br { true_target, .. } =
            &mut func.get_block_mut("header").unwrap().instructions[1]
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
        equivalent(func, true);
    }

    #[test]
    fn quadratic_shared_latches_require_both_updates() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, 7, false);
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
        equivalent(func.clone(), true);
        let body = func.get_block_mut("body").unwrap();
        let y_update = body.instructions.remove(4);
        func.get_block_mut("left")
            .unwrap()
            .instructions
            .insert(0, y_update);
        equivalent(func, false);
    }

    #[test]
    fn quadratic_discovery_rejects_ambiguous_definition_epochs() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, 7, false);
        func.get_block_mut("body")
            .unwrap()
            .instructions
            .insert(4, binary(IntBinOp::Add, 0, ty, operand(0), constant(1)));
        equivalent(func, false);
        let mut func = quadratic(ty, 20, 3, 7, false);
        let body = func.get_block_mut("body").unwrap();
        let update = body.instructions.remove(3);
        body.instructions.insert(0, update);
        equivalent(func, false);
    }

    #[test]
    fn polynomial_discovery_caps_degree_and_expansion() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 2, 3, 7, false);
        let body = func.get_block_mut("body").unwrap();
        body.instructions[0] = binary(IntBinOp::Mul, 4, ty, operand(10), operand(6));
        body.instructions
            .insert(0, binary(IntBinOp::Mul, 10, ty, operand(0), operand(6)));
        let defs = HashMap::from([
            (reg(0), vec![(2, 4)]),
            (reg(6), vec![(2, 5)]),
            (reg(10), vec![(2, 0)]),
        ]);
        let info = LoopInvariantCodeMotion.find_loops(&func).remove(0);
        let mut analysis = polynomial::Analysis::new(&func, &info, &defs, (2, 1), ty);
        assert!(analysis.product(&operand(10), &operand(6)).is_none());
        let body = func.get_block_mut("body").unwrap();
        body.instructions.remove(0);
        body.instructions[0] = binary(IntBinOp::Mul, 4, ty, operand(199), operand(6));
        let mut defs = HashMap::new();
        for id in 10..200 {
            let source = if id == 10 { 0 } else { id - 1 };
            body.instructions.insert(
                (id - 10) as usize,
                binary(IntBinOp::Add, id, ty, operand(source), operand(source)),
            );
            defs.insert(reg(id), vec![(2, (id - 10) as usize)]);
        }
        defs.insert(reg(0), vec![(2, 193)]);
        defs.insert(reg(6), vec![(2, 194)]);
        let mut analysis = polynomial::Analysis::new(&func, &info, &defs, (2, 190), ty);
        assert!(analysis.product(&operand(199), &operand(6)).is_none());
    }
    #[test]
    fn quadratic_product_copies_preserve_observations_after_state_advance() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, -7, false);
        let body = func.get_block_mut("body").unwrap();
        body.instructions
            .push(binary(IntBinOp::Add, 2, ty, operand(2), operand(4)));
        let last = body.instructions.len() - 1;
        body.instructions.swap(last, last - 1);
        let func = equivalent(func, true);
        assert!(
            func.get_block("body")
                .unwrap()
                .instructions
                .iter()
                .any(|i| { matches!(i, Instruction::Copy { dst, .. } if *dst == reg(4)) })
        );
    }

    #[test]
    fn quadratic_product_reuses_destination_when_every_use_precedes_update() {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, -7, false);
        func.get_block_mut("body").unwrap().instructions.remove(1);
        let func = equivalent(func, true);
        assert!(
            !func
                .get_block("body")
                .unwrap()
                .instructions
                .iter()
                .any(|i| { matches!(i, Instruction::Copy { dst, .. } if *dst == reg(4)) })
        );
    }

    fn quadratic_separate_updates() -> Function {
        let ty = MirType::i64();
        let mut func = quadratic(ty, 20, 3, 7, false);
        let body = func.get_block_mut("body").unwrap();
        let mut tail = body.instructions.split_off(3);
        let y_tail = tail.split_off(1);
        tail.push(Instruction::Jmp {
            target: "y_update".into(),
        });
        body.instructions.push(Instruction::Jmp {
            target: "x_update".into(),
        });
        func.blocks.push(Block {
            label: "x_update".into(),
            instructions: tail,
        });
        func.blocks.push(Block {
            label: "y_update".into(),
            instructions: y_tail,
        });
        func
    }

    #[test]
    fn quadratic_factors_can_advance_in_separate_dominating_blocks() {
        equivalent(quadratic_separate_updates(), true);
    }

    #[test]
    fn quadratic_rewrite_rejects_paths_revisiting_product_after_update() {
        let mut func = quadratic_separate_updates();
        *func
            .get_block_mut("y_update")
            .unwrap()
            .instructions
            .last_mut()
            .unwrap() = Instruction::Br {
            cond: reg(3),
            true_target: "header".into(),
            false_target: "body".into(),
        };
        equivalent(func, false);
    }

    #[test]
    fn quadratic_rewrite_rejects_nested_factor_updates() {
        let mut func = quadratic_separate_updates();
        *func
            .get_block_mut("y_update")
            .unwrap()
            .instructions
            .last_mut()
            .unwrap() = Instruction::Br {
            cond: reg(3),
            true_target: "header".into(),
            false_target: "y_update".into(),
        };
        equivalent(func, false);
    }

    #[test]
    fn quadratic_rewrite_rejects_factor_width_mismatch_and_header_products() {
        let mut func = quadratic(MirType::i64(), 20, 3, 7, false);
        if let Instruction::IntBinary { ty, .. } =
            &mut func.get_block_mut("body").unwrap().instructions[4]
        {
            *ty = MirType::i32();
        }
        equivalent(func, false);
        let mut func = quadratic(MirType::i64(), 20, 3, 7, false);
        let product = func.get_block_mut("body").unwrap().instructions.remove(0);
        func.get_block_mut("header")
            .unwrap()
            .instructions
            .insert(0, product);
        equivalent(func, false);
    }
    #[test]
    fn quadratic_differences_may_wrap_when_observed_products_fit() {
        for (ty, step) in [
            (MirType::i8(), 8),
            (MirType::i16(), 128),
            (MirType::i32(), 32768),
        ] {
            let mut func = quadratic(ty, 2, step, step + 1, false);
            for instruction in &mut func.get_block_mut("entry").unwrap().instructions {
                if let Instruction::Copy { dst, src, .. } = instruction
                    && (*dst == reg(0) || *dst == reg(6))
                {
                    *src = constant(1);
                }
            }
            equivalent(func, true);
        }
    }
}
