//! Loop vectorization for MIR, enabled only where the alias analysis proves the
//! transform cannot change what the program reads or writes.
//!
//! The one shape this pass accepts is a self-looping block whose single load and
//! single store cover the same bytes (`AliasResult::SameRange`), whose address
//! advances by exactly one element per iteration, and whose trip count is a
//! constant multiple of the vector width. Widening such a loop touches exactly
//! the bytes the scalar loop already touched, in the same order per lane, so no
//! cross-iteration dependence can exist. Anything else stays scalar.

use crate::mir::transform::alias::{AliasResult, alias, scan_block};
use crate::mir::transform::{Dominators, calculate_dominators};
use crate::mir::{
    AddressMode, AddressSpace, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType,
    Operand, Register, ScalarType, VectorLane, VectorOp, VectorType, VirtualReg,
    instruction::FloatBinOp,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_mir::Located;
use lamina_platform::X86CpuLevel;
use std::collections::{HashMap, HashSet};

/// Vector unit a function is being compiled for. Only the x86-64 SSE2 unit is
/// wired up; every other target keeps the pass switched off and so is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorUnit {
    X86Sse2(X86CpuLevel),
}

/// Vectorizes self-looping element-wise loops whose accesses the alias analysis
/// proves identical.
#[derive(Clone, Copy, Debug)]
pub struct AutoVectorization {
    unit: VectorUnit,
}

impl AutoVectorization {
    pub fn new(unit: VectorUnit) -> Self {
        Self { unit }
    }
}

impl Transform for AutoVectorization {
    fn name(&self) -> &'static str {
        "auto_vectorization"
    }

    fn description(&self) -> &'static str {
        "Widen element-wise self-loops whose load and store provably cover the same bytes"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::Vectorization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        let Some(mut next) = register_base(func) else {
            return Ok(false);
        };
        let dominators = calculate_dominators(func);
        let labels: Vec<String> = func.blocks.iter().map(|b| b.label.clone()).collect();
        let mut changed = false;
        for label in labels {
            if let Some(plan) = self.plan(func, &dominators, &label) {
                rewrite(func, &plan, &mut next);
                changed = true;
            }
        }
        Ok(changed)
    }
}

/// Value of a register in this block as `base + step * iv + offset`, where `iv`
/// is the induction variable's value on entry to the visit and `base` names a
/// register the block never defines.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Affine {
    base: Option<u32>,
    step: i64,
    offset: i64,
}

impl Affine {
    fn constant(value: i64) -> Self {
        Self {
            base: None,
            step: 0,
            offset: value,
        }
    }

    fn add(self, other: Self) -> Option<Self> {
        let base = match (self.base, other.base) {
            (Some(_), Some(_)) => return None,
            (left, right) => left.or(right),
        };
        Some(Self {
            base,
            step: self.step.checked_add(other.step)?,
            offset: self.offset.checked_add(other.offset)?,
        })
    }

    fn sub(self, other: Self) -> Option<Self> {
        if other.base.is_some() {
            return None;
        }
        Some(Self {
            base: self.base,
            step: self.step.checked_sub(other.step)?,
            offset: self.offset.checked_sub(other.offset)?,
        })
    }

    fn scale(self, factor: i64) -> Option<Self> {
        if self.base.is_some() {
            return None;
        }
        Some(Self {
            base: None,
            step: self.step.checked_mul(factor)?,
            offset: self.offset.checked_mul(factor)?,
        })
    }
}

struct Plan {
    block: String,
    load_idx: usize,
    compute_idx: usize,
    store_idx: usize,
    increment_idx: usize,
    lane: VectorLane,
    op: VectorOp,
    rhs: Operand,
    iv: Register,
    factor: i64,
    address: Register,
    offset: i16,
    address_space: AddressSpace,
}

impl AutoVectorization {
    fn plan(&self, func: &Function, dominators: &Dominators, label: &str) -> Option<Plan> {
        let block = func.get_block(label)?;
        let (terminator, body) = block.instructions.split_last()?;
        let Instruction::Br {
            cond,
            true_target,
            false_target,
        } = &terminator.instr
        else {
            return None;
        };
        if true_target != label || false_target == label {
            return None;
        }

        let defined: HashSet<Register> = body.iter().filter_map(|i| i.def_reg().cloned()).collect();

        let (increment_idx, iv) = single_increment(body, &defined)?;
        let bound = loop_bound(body, cond, &iv, increment_idx)?;
        let (start, init) = entry_value(func, dominators, label, &iv)?;
        if reenters_without(func, label, init) {
            return None;
        }
        let trips = bound.checked_sub(start)?;

        let (load_idx, store_idx) = memory_pair(body)?;
        let accesses = scan_block(block, &func.stack_reservations);
        let (Some(read), Some(write)) = (accesses[load_idx], accesses[store_idx]) else {
            return None;
        };
        if alias(&read, &write) != AliasResult::SameRange {
            return None;
        }

        let (
            Instruction::Load {
                ty: load_ty,
                dst: loaded,
                addr: load_addr,
                attrs: load_attrs,
            },
            Instruction::Store {
                ty: store_ty,
                src: stored,
                addr: store_addr,
                attrs: store_attrs,
            },
        ) = (&body[load_idx].instr, &body[store_idx].instr)
        else {
            return None;
        };
        if load_attrs.volatile || store_attrs.volatile || load_ty != store_ty {
            return None;
        }
        let MirType::Scalar(element) = *load_ty else {
            return None;
        };
        let (address, offset, address_space) = same_place(load_addr, store_addr)?;

        let compute_idx = (load_idx + 1..store_idx).find(|i| {
            body[*i]
                .def_reg()
                .is_some_and(|dst| Some(dst) == stored.as_register())
        })?;
        let (op, lhs, rhs) = arithmetic(&body[compute_idx].instr, element)?;
        if lhs.as_register() != Some(loaded) {
            return None;
        }
        if !single_use_inside(func, label, loaded, load_idx, compute_idx)
            || !single_use_inside(func, label, stored.as_register()?, compute_idx, store_idx)
        {
            return None;
        }
        if rhs
            .as_register()
            .is_some_and(|reg| defined.contains(reg) || !reg.is_virtual())
        {
            return None;
        }

        let lane = lane_of(element)?;
        let width = i64::from(lane_bytes(lane));
        let factor = 16 / width;
        if trips < factor || trips % factor != 0 {
            return None;
        }
        if !self.profitable(op, lane) {
            return None;
        }

        let values = affine_values(body, &iv, &defined);
        let place = values.get(&register_id(&address)?)?;
        if place.base.is_none() || place.step != width {
            return None;
        }

        let compare_idx = body
            .iter()
            .position(|inst| matches!(inst.instr, Instruction::IntCmp { .. }))?;
        let mut accounted =
            HashSet::from([load_idx, compute_idx, store_idx, increment_idx, compare_idx]);
        for (index, inst) in body.iter().enumerate() {
            if inst
                .def_reg()
                .and_then(register_id)
                .is_some_and(|id| values.contains_key(&id))
            {
                accounted.insert(index);
            }
        }
        if accounted.len() != body.len() {
            return None;
        }
        let chain_or_control = |index: usize, inst: &Located| {
            index == increment_idx
                || index == compare_idx
                || inst
                    .def_reg()
                    .and_then(register_id)
                    .is_some_and(|id| values.contains_key(&id))
        };
        if body
            .iter()
            .enumerate()
            .any(|(index, inst)| inst.use_regs().contains(&&iv) && !chain_or_control(index, inst))
        {
            return None;
        }
        if escapes(func, label, &defined, &iv) {
            return None;
        }

        Some(Plan {
            block: label.to_owned(),
            load_idx,
            compute_idx,
            store_idx,
            increment_idx,
            lane,
            op,
            rhs: rhs.clone(),
            iv,
            factor,
            address: address.clone(),
            offset,
            address_space,
        })
    }

    /// The cost model from Plan/vector-cost-model.md: a 64-bit lane multiply is
    /// emulated from `pmuludq` below AVX-512DQ and measured slower than the
    /// scalar loop, so it is refused there.
    fn profitable(&self, op: VectorOp, lane: VectorLane) -> bool {
        let VectorUnit::X86Sse2(level) = self.unit;
        if op == VectorOp::VMul && matches!(lane, VectorLane::I8 | VectorLane::I16) {
            return false;
        }
        if op == VectorOp::VMul && lane == VectorLane::I32 && level < X86CpuLevel::V2 {
            return false;
        }
        !(op == VectorOp::VMul && lane == VectorLane::I64 && level < X86CpuLevel::V4)
    }
}

fn register_id(reg: &Register) -> Option<u32> {
    match reg {
        Register::Virtual(v) => Some(v.id),
        Register::Physical(_) => None,
    }
}

fn lane_of(scalar: ScalarType) -> Option<VectorLane> {
    match scalar {
        ScalarType::I8 => Some(VectorLane::I8),
        ScalarType::I16 => Some(VectorLane::I16),
        ScalarType::I32 => Some(VectorLane::I32),
        ScalarType::I64 => Some(VectorLane::I64),
        ScalarType::F32 => Some(VectorLane::F32),
        ScalarType::F64 => Some(VectorLane::F64),
        _ => None,
    }
}

fn lane_bytes(lane: VectorLane) -> u8 {
    match lane {
        VectorLane::I8 => 1,
        VectorLane::I16 => 2,
        VectorLane::I32 | VectorLane::F32 => 4,
        VectorLane::I64 | VectorLane::F64 => 8,
    }
}

/// `dst = iv + 1`, the only such self-increment in the block, and the only place the
/// block writes `iv` at all. A second write would advance the induction variable by
/// something other than one element per iteration while this still read as a unit stride.
fn single_increment(body: &[Located], defined: &HashSet<Register>) -> Option<(usize, Register)> {
    let mut found = None;
    for (index, inst) in body.iter().enumerate() {
        let Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst,
            lhs: Operand::Register(lhs),
            rhs: Operand::Immediate(Immediate::I64(1)),
        } = &inst.instr
        else {
            continue;
        };
        if dst != lhs || !dst.is_virtual() {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((index, dst.clone()));
    }
    let (index, iv) = found?;
    if body.iter().filter(|i| i.def_reg() == Some(&iv)).count() != 1 {
        return None;
    }
    defined.contains(&iv).then_some((index, iv))
}

/// `cond = iv <signed N` placed after the increment, with `N` a constant.
fn loop_bound(body: &[Located], cond: &Register, iv: &Register, increment: usize) -> Option<i64> {
    let mut bound = None;
    for (index, inst) in body.iter().enumerate() {
        let Instruction::IntCmp {
            op: IntCmpOp::SLt,
            ty: MirType::Scalar(ScalarType::I64),
            dst,
            lhs: Operand::Register(lhs),
            rhs: Operand::Immediate(Immediate::I64(end)),
        } = &inst.instr
        else {
            if matches!(inst.instr, Instruction::IntCmp { .. }) {
                return None;
            }
            continue;
        };
        if dst != cond || lhs != iv || index < increment {
            return None;
        }
        if bound.is_some() {
            return None;
        }
        bound = Some(*end);
    }
    bound
}

/// The constant the induction variable holds on entry to the loop: the only
/// definition outside the loop block must be that constant, and must dominate.
fn entry_value<'a>(
    func: &'a Function,
    dominators: &Dominators,
    label: &str,
    iv: &Register,
) -> Option<(i64, &'a str)> {
    let mut start = None;
    for block in &func.blocks {
        if block.label == label {
            continue;
        }
        for inst in &block.instructions {
            if inst.def_reg() != Some(iv) {
                continue;
            }
            if start.is_some() || !dominators.dominates(&block.label, label) {
                return None;
            }
            start = match &inst.instr {
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::I64),
                    src: Operand::Immediate(Immediate::I64(value)),
                    ..
                } => Some((*value, block.label.as_str())),
                _ => return None,
            };
        }
    }
    start
}

/// Whether control can reach `label` again without passing `init`, which would
/// start the loop at an induction value this pass did not predict. `init`
/// dominates `label`, so refusing this also fixes the value on every entry.
fn reenters_without(func: &Function, label: &str, init: &str) -> bool {
    let Some(block) = func.get_block(label) else {
        return true;
    };
    let mut seen: HashSet<&str> = HashSet::new();
    let mut queue: Vec<&str> = block
        .successors()
        .into_iter()
        .filter(|next| *next != label)
        .collect();
    while let Some(next) = queue.pop() {
        if next == label {
            return true;
        }
        if next == init || !seen.insert(next) {
            continue;
        }
        match func.get_block(next) {
            Some(block) => queue.extend(block.successors()),
            None => return true,
        }
    }
    false
}

/// Exactly one load and one store, the load first, and no other memory access.
fn memory_pair(body: &[Located]) -> Option<(usize, usize)> {
    let mut load = None;
    let mut store = None;
    for (index, inst) in body.iter().enumerate() {
        let slot = match &inst.instr {
            Instruction::Load { .. } => &mut load,
            Instruction::Store { .. } => &mut store,
            _ => continue,
        };
        if slot.replace(index).is_some() {
            return None;
        }
    }
    let (load, store) = (load?, store?);
    (load < store).then_some((load, store))
}

fn same_place(load: &AddressMode, store: &AddressMode) -> Option<(Register, i16, AddressSpace)> {
    match (load, store) {
        (
            AddressMode::BaseOffset {
                base: lb,
                offset: lo,
                address_space: ls,
            },
            AddressMode::BaseOffset {
                base: sb,
                offset: so,
                address_space: ss,
            },
        ) if lb == sb && lo == so && ls == ss && *ls == AddressSpace::default() => {
            Some((lb.clone(), *lo, *ls))
        }
        _ => None,
    }
}

fn arithmetic(inst: &Instruction, element: ScalarType) -> Option<(VectorOp, &Operand, &Operand)> {
    match inst {
        Instruction::IntBinary {
            op,
            ty: MirType::Scalar(ty),
            lhs,
            rhs,
            ..
        } if *ty == element && integer(element) => {
            let op = match op {
                IntBinOp::Add => VectorOp::VAdd,
                IntBinOp::Sub => VectorOp::VSub,
                IntBinOp::Mul => VectorOp::VMul,
                IntBinOp::And => VectorOp::VAnd,
                IntBinOp::Or => VectorOp::VOr,
                IntBinOp::Xor => VectorOp::VXor,
                _ => return None,
            };
            Some((op, lhs, rhs))
        }
        Instruction::FloatBinary {
            op,
            ty: MirType::Scalar(ty),
            lhs,
            rhs,
            ..
        } if *ty == element && matches!(element, ScalarType::F32 | ScalarType::F64) => {
            let op = match op {
                FloatBinOp::FAdd => VectorOp::VAdd,
                FloatBinOp::FSub => VectorOp::VSub,
                FloatBinOp::FMul => VectorOp::VMul,
                FloatBinOp::FDiv => return None,
            };
            Some((op, lhs, rhs))
        }
        _ => None,
    }
}

/// `reg` is defined at `def`, used only at `use_at`, and nowhere else at all.
fn single_use_inside(
    func: &Function,
    label: &str,
    reg: &Register,
    def: usize,
    use_at: usize,
) -> bool {
    let mut seen = 0;
    for block in &func.blocks {
        for (index, inst) in block.instructions.iter().enumerate() {
            if inst.use_regs().contains(&reg) {
                if block.label != label || index != use_at {
                    return false;
                }
                seen += 1;
            }
            if inst.def_reg() == Some(reg) && (block.label != label || index != def) {
                return false;
            }
        }
    }
    seen == 1
}

/// Any register the block defines, apart from the induction variable, that some
/// other block reads.
fn escapes(func: &Function, label: &str, defined: &HashSet<Register>, iv: &Register) -> bool {
    func.blocks.iter().any(|block| {
        block.label != label
            && block.instructions.iter().any(|inst| {
                inst.use_regs()
                    .iter()
                    .any(|reg| *reg != iv && defined.contains(*reg))
            })
    })
}

/// Registers whose value in this visit is an affine function of the induction
/// variable's entry value. Uses are read before the instruction's definition.
fn affine_values(
    body: &[Located],
    iv: &Register,
    defined: &HashSet<Register>,
) -> HashMap<u32, Affine> {
    let mut values: HashMap<u32, Affine> = HashMap::new();
    if let Some(id) = register_id(iv) {
        values.insert(
            id,
            Affine {
                base: None,
                step: 1,
                offset: 0,
            },
        );
    }
    let read = |values: &HashMap<u32, Affine>, operand: &Operand| -> Option<Affine> {
        match operand {
            Operand::Immediate(Immediate::I64(value)) => Some(Affine::constant(*value)),
            Operand::Immediate(_) => None,
            Operand::Register(reg) => {
                let id = register_id(reg)?;
                match values.get(&id) {
                    Some(value) => Some(*value),
                    None if !defined.contains(reg) => Some(Affine {
                        base: Some(id),
                        step: 0,
                        offset: 0,
                    }),
                    None => None,
                }
            }
        }
    };
    for inst in body {
        let Some(dst) = inst.def_reg().and_then(register_id) else {
            continue;
        };
        let value = match &inst.instr {
            Instruction::Copy {
                ty: MirType::Scalar(ScalarType::I64 | ScalarType::Ptr),
                src,
                ..
            } => read(&values, src),
            Instruction::IntConvert { from, to, src, .. } if wide(*from) && wide(*to) => {
                read(&values, src)
            }
            Instruction::Lea { base, offset, .. } => {
                read(&values, &Operand::Register(base.clone()))
                    .and_then(|value| value.add(Affine::constant(i64::from(*offset))))
            }
            Instruction::IntBinary {
                op,
                ty: MirType::Scalar(ScalarType::I64 | ScalarType::Ptr),
                lhs,
                rhs,
                ..
            } => {
                let (left, right) = (read(&values, lhs), read(&values, rhs));
                match (op, left, right) {
                    (IntBinOp::Add, Some(left), Some(right)) => left.add(right),
                    (IntBinOp::Sub, Some(left), Some(right)) => left.sub(right),
                    (IntBinOp::Mul, Some(left), Some(right)) => match (left.base, right.base) {
                        (None, None) if left.step == 0 => right.scale(left.offset),
                        (None, None) if right.step == 0 => left.scale(right.offset),
                        _ => None,
                    },
                    (IntBinOp::Shl, Some(left), Some(right))
                        if right.base.is_none()
                            && right.step == 0
                            && (0..32).contains(&right.offset) =>
                    {
                        left.scale(1i64 << right.offset)
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        match value {
            Some(value) => values.insert(dst, value),
            None => values.remove(&dst),
        };
    }
    values
}

fn integer(ty: ScalarType) -> bool {
    matches!(
        ty,
        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
    )
}

fn wide(ty: MirType) -> bool {
    matches!(
        ty,
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_))
    )
}

fn register_base(func: &Function) -> Option<u32> {
    func.blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .flat_map(|i| i.def_reg().into_iter().chain(i.use_regs()))
        .chain(func.sig.params.iter().map(|p| &p.reg))
        .filter_map(register_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
}

fn rewrite(func: &mut Function, plan: &Plan, next: &mut u32) {
    let vector = MirType::Vector(VectorType::V128(plan.lane));
    let loaded = Register::Virtual(VirtualReg::vec(*next));
    let widened = Register::Virtual(VirtualReg::vec(*next + 1));
    let splat = Register::Virtual(VirtualReg::vec(*next + 2));
    *next += 3;

    let Some(block) = func.get_block_mut(&plan.block) else {
        return;
    };
    let address = AddressMode::BaseOffset {
        base: plan.address.clone(),
        offset: plan.offset,
        address_space: plan.address_space,
    };

    block.instructions[plan.load_idx].instr = Instruction::Load {
        ty: vector,
        dst: loaded.clone(),
        addr: address.clone(),
        attrs: Default::default(),
    };
    block.instructions[plan.compute_idx].instr = Instruction::VectorOp {
        op: plan.op,
        ty: vector,
        dst: widened.clone(),
        operands: vec![Operand::Register(loaded), Operand::Register(splat.clone())],
    };
    block.instructions[plan.store_idx].instr = Instruction::Store {
        ty: vector,
        src: Operand::Register(widened),
        addr: address,
        attrs: Default::default(),
    };
    block.instructions[plan.increment_idx].instr = Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: MirType::Scalar(ScalarType::I64),
        dst: plan.iv.clone(),
        lhs: Operand::Register(plan.iv.clone()),
        rhs: Operand::Immediate(Immediate::I64(plan.factor)),
    };

    block.instructions.insert(
        plan.load_idx,
        Instruction::VectorOp {
            op: VectorOp::VSplat,
            ty: vector,
            dst: splat,
            operands: vec![plan.rhs.clone()],
        }
        .into(),
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mir::transform::TransformPipeline;
    use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};

    /// Runs the real O3 pipeline, so a test sees the MIR the pass sees in a build.
    fn vector_ops(source: &str, level: X86CpuLevel) -> usize {
        let mut module = from_ir_x86_64(&parse_module(source).unwrap(), "vectorization").unwrap();
        TransformPipeline::default_for_opt_level_on(3, Some(VectorUnit::X86Sse2(level)))
            .apply_to_module(&mut module)
            .unwrap();
        module
            .functions
            .values()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i.instr, Instruction::VectorOp { .. }))
            .count()
    }

    fn loop_over(body: &str) -> String {
        format!(
            "fn @f() -> i64 {{
entry:
  %a = stackalloc.[64 x i64]
  %b = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 64
  br %more, body, done
body:
{body}
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}}"
        )
    }

    #[test]
    fn widens_a_loop_that_rewrites_its_own_elements() {
        let source = loop_over(
            "  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z",
        );
        assert!(vector_ops(&source, X86CpuLevel::V2) > 0);
    }

    #[test]
    fn refuses_a_store_that_overlaps_the_next_load() {
        let source = loop_over(
            "  %src = getelem.i64 %a, %j
  %j1 = add.i64 %j, 1
  %dst = getelem.i64 %a, %j1
  %x = load.i64 %src
  %z = add.i64 %x, 5
  store.i64 %dst, %z",
        );
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_second_array_the_analysis_cannot_separate() {
        let source = loop_over(
            "  %q = getelem.i64 %a, %j
  %r = getelem.i64 %b, %j
  %x = load.i64 %q
  %y = load.i64 %r
  %z = add.i64 %x, %y
  store.i64 %q, %z",
        );
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_stride_wider_than_one_element() {
        let source = "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 32
  br %more, body, done
body:
  %idx = mul.i64 %j, 2
  %q = getelem.i64 %a, %idx
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_trip_count_the_vector_width_does_not_divide() {
        let source = loop_over(
            "  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z",
        )
        .replace("lt.i64 %j, 64", "lt.i64 %j, 63");
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_volatile_accesses() {
        let source = loop_over(
            "  %q = getelem.i64 %a, %j
  %x = volatile load.i64 %q
  %z = add.i64 %x, 5
  volatile store.i64 %q, %z",
        );
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_loop_whose_bound_is_not_known() {
        let source = "fn @f(i64 %n) -> i64 {
entry:
  %a = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, %n
  br %more, body, done
body:
  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn a_64_bit_lane_multiply_waits_for_a_native_one() {
        let source = loop_over(
            "  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = mul.i64 %x, 3
  store.i64 %q, %z",
        );
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
        assert_eq!(vector_ops(&source, X86CpuLevel::V3), 0);
        assert!(vector_ops(&source, X86CpuLevel::V4) > 0);
    }

    #[test]
    fn refuses_a_store_that_overlaps_the_previous_load() {
        let source = loop_over(
            "  %dst = getelem.i64 %a, %j
  %j1 = add.i64 %j, 1
  %src = getelem.i64 %a, %j1
  %x = load.i64 %src
  %z = add.i64 %x, 3
  store.i64 %dst, %z",
        );
        assert_eq!(vector_ops(&source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_loop_carried_scalar() {
        let source = "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  %carry = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 64
  br %more, body, done
body:
  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, %carry
  store.i64 %q, %z
  %carry = add.i64 %z, 0
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_two_pointer_parameters_the_analysis_cannot_separate() {
        let source = "fn @f(ptr %dst, ptr %src) -> i64 {
entry:
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 64
  br %more, body, done
body:
  %pd = getelem.i64 %dst, %j
  %ps = getelem.i64 %src, %j
  %x = load.i64 %ps
  %z = add.i64 %x, 3
  store.i64 %pd, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_float_reduction() {
        let source = "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x f64]
  %k = add.i64 0, 0
  %t = add.f64 0.0, 0.0
  jmp head
head:
  %more = lt.i64 %k, 64
  br %more, body, done
body:
  %q = getelem.f64 %a, %k
  %w = load.f64 %q
  %t = add.f64 %t, %w
  %k = add.i64 %k, 1
  jmp head
done:
  %ti = fptoint.f64.i64 %t
  ret.i64 %ti
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_second_write_to_the_induction_variable() {
        let source = "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 60
  br %more, body, done
body:
  %j = add.i64 %j, 2
  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }

    #[test]
    fn refuses_a_float_division() {
        let source = "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x f64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 64
  br %more, body, done
body:
  %q = getelem.f64 %a, %j
  %x = load.f64 %q
  %z = div.f64 %x, 2.0
  store.f64 %q, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}";
        assert_eq!(vector_ops(source, X86CpuLevel::V2), 0);
    }
}
