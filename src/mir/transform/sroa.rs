//! Splits stack slots into one slot per field.
//!
//! A slot is a candidate when a `lea` derives one of its addresses. It is split only when
//! every use of every derived address is a non-volatile, non-atomic load or store at a
//! fixed offset, the accesses are scalar and fit in the slot, and no two accesses share
//! bytes without sharing an offset and type. Each distinct field becomes its own slot,
//! addressed directly by one fresh pointer, so mem2reg can promote the pieces. Any other
//! slot is left unchanged.

use crate::mir::{
    AddressMode, AddressSpace, Block, Function, Instruction, MirType, Operand, Register,
    ScalarType, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::mem;

/// Splits stack slots accessed at constant field offsets into one slot per field.
#[derive(Default)]
pub struct ScalarReplacement;

impl Transform for ScalarReplacement {
    fn name(&self) -> &'static str {
        "sroa"
    }

    fn description(&self) -> &'static str {
        "Splits stack slots accessed at constant field offsets into one slot per field"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::MemoryOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        Ok(split(func))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    EscapeCall,
    EscapeStore,
    EscapeOther,
    DynamicOffset,
    Volatile,
    Atomic,
    NonScalar,
    AddressSpace,
    MixedType,
    Overlap,
    OutOfBounds,
    MultipleDefs,
    SlotNamed,
    MultipleStackAddrs,
}

enum Use {
    Lea,
    Access { offset: i64, ty: MirType },
}

#[derive(Default)]
struct SlotInfo {
    stack_addrs: usize,
    pointers: Vec<Register>,
    has_lea: bool,
    accesses: Vec<(i64, MirType)>,
    refusal: Option<Refusal>,
}

fn scalar_size(ty: MirType) -> Option<i64> {
    match ty {
        MirType::Scalar(ScalarType::I8) => Some(1),
        MirType::Scalar(ScalarType::I16) => Some(2),
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => Some(4),
        MirType::Scalar(ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr) => Some(8),
        _ => None,
    }
}

fn constant_offset(addr: &AddressMode, pointer: &Register) -> Result<i64, Refusal> {
    match addr {
        AddressMode::BaseOffset {
            base,
            offset,
            address_space,
        } if base == pointer => {
            if *address_space == AddressSpace::default() {
                Ok(i64::from(*offset))
            } else {
                Err(Refusal::AddressSpace)
            }
        }
        AddressMode::BaseIndexScale { base, .. } if base == pointer => Err(Refusal::DynamicOffset),
        _ => Err(Refusal::EscapeOther),
    }
}

/// Classifies one instruction that uses `pointer`, which it names `count` times.
fn classify(inst: &Instruction, pointer: &Register, count: usize) -> Result<Use, Refusal> {
    if count != 1 {
        return Err(Refusal::EscapeOther);
    }
    match inst {
        Instruction::Lea { base, .. } if base == pointer => Ok(Use::Lea),
        Instruction::Load {
            ty, addr, attrs, ..
        } => {
            let offset = constant_offset(addr, pointer)?;
            if attrs.volatile {
                return Err(Refusal::Volatile);
            }
            Ok(Use::Access { offset, ty: *ty })
        }
        Instruction::Store {
            ty,
            src,
            addr,
            attrs,
        } => {
            if matches!(src, Operand::Register(r) if r == pointer) {
                return Err(Refusal::EscapeStore);
            }
            let offset = constant_offset(addr, pointer)?;
            if attrs.volatile {
                return Err(Refusal::Volatile);
            }
            Ok(Use::Access { offset, ty: *ty })
        }
        Instruction::AtomicLoad { .. }
        | Instruction::AtomicStore { .. }
        | Instruction::AtomicBinary { .. }
        | Instruction::AtomicCompareExchange { .. } => Err(Refusal::Atomic),
        Instruction::Call { .. }
        | Instruction::CallIndirect { .. }
        | Instruction::TailCall { .. } => Err(Refusal::EscapeCall),
        _ => Err(Refusal::EscapeOther),
    }
}

fn record_refusal(slots: &mut HashMap<VirtualReg, SlotInfo>, slot: VirtualReg, why: Refusal) {
    let info = slots.entry(slot).or_default();
    if info.refusal.is_none() {
        info.refusal = Some(why);
    }
}

/// Field layout of a slot (byte offset and type of each distinct field), or the reason
/// the slot must stay whole.
fn fields_of(
    func: &Function,
    slot: VirtualReg,
    info: &SlotInfo,
    mentioned: &HashSet<Register>,
    defs: &HashMap<Register, usize>,
) -> Result<Vec<(i64, MirType)>, Refusal> {
    if info.stack_addrs != 1 {
        return Err(Refusal::MultipleStackAddrs);
    }
    if mentioned.contains(&Register::Virtual(slot)) {
        return Err(Refusal::SlotNamed);
    }
    if info
        .pointers
        .iter()
        .any(|pointer| defs.get(pointer).copied().unwrap_or(0) != 1)
    {
        return Err(Refusal::MultipleDefs);
    }
    let words = func.stack_reservations.get(&slot).copied().unwrap_or(0);
    let reserved = 8 * (1 + i64::try_from(words).unwrap_or(i64::MAX / 8));

    let mut fields: BTreeMap<i64, MirType> = BTreeMap::new();
    for &(offset, ty) in &info.accesses {
        let size = scalar_size(ty).ok_or(Refusal::NonScalar)?;
        if offset < 0 || offset + size > reserved {
            return Err(Refusal::OutOfBounds);
        }
        if let Some(existing) = fields.insert(offset, ty)
            && existing != ty
        {
            return Err(Refusal::MixedType);
        }
    }
    let mut end = i64::MIN;
    for (&offset, &ty) in &fields {
        if offset < end {
            return Err(Refusal::Overlap);
        }
        end = offset + scalar_size(ty).unwrap_or(0);
    }
    Ok(fields.into_iter().collect())
}

fn split(func: &mut Function) -> bool {
    let mut defs: HashMap<Register, usize> = HashMap::new();
    let mut mentioned: HashSet<Register> = HashSet::new();
    let mut stack_addrs: Vec<(Register, VirtualReg)> = Vec::new();

    for param in &func.sig.params {
        *defs.entry(param.reg.clone()).or_default() += 1;
        mentioned.insert(param.reg.clone());
    }
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(dst) = inst.def_reg() {
                *defs.entry(dst.clone()).or_default() += 1;
                mentioned.insert(dst.clone());
            }
            for used in inst.use_regs() {
                mentioned.insert(used.clone());
            }
            if let Instruction::StackAddr { dst, slot } = inst {
                stack_addrs.push((dst.clone(), *slot));
            }
        }
    }
    if stack_addrs.is_empty() {
        return false;
    }

    let mut derived: HashMap<Register, (VirtualReg, i64)> = HashMap::new();
    for (dst, slot) in &stack_addrs {
        derived.insert(dst.clone(), (*slot, 0));
    }
    loop {
        let mut grew = false;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Instruction::Lea { dst, base, offset } = inst
                    && !derived.contains_key(dst)
                    && let Some(&(slot, base_offset)) = derived.get(base)
                {
                    derived.insert(dst.clone(), (slot, base_offset + i64::from(*offset)));
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }

    let mut slots: HashMap<VirtualReg, SlotInfo> = HashMap::new();
    for (slot_reg, (slot, _)) in &derived {
        slots
            .entry(*slot)
            .or_default()
            .pointers
            .push(slot_reg.clone());
    }
    for (_, slot) in &stack_addrs {
        slots.entry(*slot).or_default().stack_addrs += 1;
    }

    for block in &func.blocks {
        for inst in &block.instructions {
            let uses = inst.use_regs();
            for &reg in &uses {
                let Some(&(slot, base_offset)) = derived.get(reg) else {
                    continue;
                };
                let count = uses.iter().filter(|r| **r == reg).count();
                match classify(inst, reg, count) {
                    Ok(Use::Lea) => {
                        slots.entry(slot).or_default().has_lea = true;
                    }
                    Ok(Use::Access { offset, ty }) => {
                        slots
                            .entry(slot)
                            .or_default()
                            .accesses
                            .push((base_offset + offset, ty));
                    }
                    Err(why) => record_refusal(&mut slots, slot, why),
                }
            }
        }
    }

    let mut candidates: Vec<VirtualReg> = slots
        .iter()
        .filter(|(_, info)| info.has_lea)
        .map(|(slot, _)| *slot)
        .collect();
    candidates.sort_by_key(|slot| (slot.id, slot.class));

    let mut layouts: HashMap<VirtualReg, Vec<(i64, MirType)>> = HashMap::new();
    for slot in &candidates {
        let Some(info) = slots.get(slot) else {
            continue;
        };
        if info.refusal.is_some() {
            continue;
        }
        if let Ok(fields) = fields_of(func, *slot, info, &mentioned, &defs)
            && !fields.is_empty()
        {
            layouts.insert(*slot, fields);
        }
    }
    if layouts.is_empty() {
        return false;
    }

    let mut next_id = mentioned
        .iter()
        .filter_map(|reg| match reg {
            Register::Virtual(v) => Some(v.id),
            Register::Physical(_) => None,
        })
        .chain(stack_addrs.iter().map(|(_, slot)| slot.id))
        .chain(func.stack_reservations.keys().map(|slot| slot.id))
        .max()
        .map_or(0, |max| max + 1);

    let mut piece_of: HashMap<(VirtualReg, i64), (VirtualReg, Register)> = HashMap::new();
    let mut pieces_by_slot: HashMap<VirtualReg, Vec<(VirtualReg, Register)>> = HashMap::new();
    for slot in &candidates {
        let Some(fields) = layouts.get(slot) else {
            continue;
        };
        for (offset, _) in fields {
            let piece = (
                VirtualReg::gpr(next_id),
                Register::Virtual(VirtualReg::gpr(next_id + 1)),
            );
            next_id += 2;
            piece_of.insert((*slot, *offset), piece.clone());
            pieces_by_slot.entry(*slot).or_default().push(piece);
        }
    }

    let split_slots: HashSet<VirtualReg> = layouts.keys().copied().collect();
    for slot in &split_slots {
        func.stack_reservations.remove(slot);
    }

    let retarget = |addr: AddressMode| -> AddressMode {
        if let AddressMode::BaseOffset { base, offset, .. } = &addr
            && let Some(&(slot, base_offset)) = derived.get(base)
            && split_slots.contains(&slot)
            && let Some((_, pointer)) = piece_of.get(&(slot, base_offset + i64::from(*offset)))
        {
            return AddressMode::BaseOffset {
                address_space: AddressSpace::default(),
                base: pointer.clone(),
                offset: 0,
            };
        }
        addr
    };
    let in_split = |reg: &Register| {
        derived
            .get(reg)
            .is_some_and(|(slot, _)| split_slots.contains(slot))
    };

    for block in &mut func.blocks {
        rewrite_block(block, &split_slots, &pieces_by_slot, &in_split, &retarget);
    }
    true
}

fn rewrite_block(
    block: &mut Block,
    split_slots: &HashSet<VirtualReg>,
    pieces_by_slot: &HashMap<VirtualReg, Vec<(VirtualReg, Register)>>,
    in_split: &impl Fn(&Register) -> bool,
    retarget: &impl Fn(AddressMode) -> AddressMode,
) {
    let instructions = mem::take(&mut block.instructions);
    let mut rewritten = Vec::with_capacity(instructions.len());
    for inst in instructions {
        match inst {
            Instruction::StackAddr { slot, .. } if split_slots.contains(&slot) => {
                for (piece_slot, pointer) in pieces_by_slot.get(&slot).into_iter().flatten() {
                    rewritten.push(Instruction::StackAddr {
                        dst: pointer.clone(),
                        slot: *piece_slot,
                    });
                }
            }
            Instruction::Lea { ref dst, .. } if in_split(dst) => {}
            Instruction::Load {
                ty,
                dst,
                addr,
                attrs,
            } => rewritten.push(Instruction::Load {
                ty,
                dst,
                addr: retarget(addr),
                attrs,
            }),
            Instruction::Store {
                ty,
                src,
                addr,
                attrs,
            } => rewritten.push(Instruction::Store {
                ty,
                src,
                addr: retarget(addr),
                attrs,
            }),
            other => rewritten.push(other),
        }
    }
    block.instructions = rewritten;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, MemoryAttrs, MemoryOrdering,
        transform::{Mem2Reg, test_utils::apply_pass},
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn i32() -> MirType {
        MirType::Scalar(ScalarType::I32)
    }

    fn slot() -> VirtualReg {
        VirtualReg::gpr(0)
    }

    fn ptr() -> Register {
        VirtualReg::gpr(1).into()
    }

    fn cond() -> Register {
        VirtualReg::gpr(2).into()
    }

    fn value(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }

    fn imm(v: i64) -> Operand {
        Operand::Immediate(Immediate::I64(v))
    }

    fn at(base: Register, offset: i16) -> AddressMode {
        AddressMode::BaseOffset {
            base,
            offset,
            address_space: Default::default(),
        }
    }

    fn volatile() -> MemoryAttrs {
        MemoryAttrs {
            volatile: true,
            ..MemoryAttrs::default()
        }
    }

    fn stack_addr() -> Instruction {
        Instruction::StackAddr {
            dst: ptr(),
            slot: slot(),
        }
    }

    fn lea(dst: Register, base: Register, offset: i32) -> Instruction {
        Instruction::Lea { dst, base, offset }
    }

    fn store(ty: MirType, src: Operand, addr: AddressMode) -> Instruction {
        Instruction::Store {
            ty,
            src,
            addr,
            attrs: MemoryAttrs::default(),
        }
    }

    fn load(ty: MirType, dst: Register, addr: AddressMode) -> Instruction {
        Instruction::Load {
            ty,
            dst,
            addr,
            attrs: MemoryAttrs::default(),
        }
    }

    fn ret(reg: Register) -> Instruction {
        Instruction::Ret {
            value: Some(Operand::Register(reg)),
        }
    }

    fn count<F: Fn(&Instruction) -> bool>(func: &Function, f: F) -> usize {
        func.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter(|i| f(i))
            .count()
    }

    fn has_lea(func: &Function) -> bool {
        count(func, |i| matches!(i, Instruction::Lea { .. })) > 0
    }

    fn stack_addr_slots(func: &Function) -> Vec<VirtualReg> {
        func.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .filter_map(|i| match i {
                Instruction::StackAddr { slot, .. } => Some(*slot),
                _ => None,
            })
            .collect()
    }

    fn two_field_struct(extra: Vec<Instruction>) -> Function {
        let mut builder = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), 4))
            .instr(store(
                i32(),
                Operand::Immediate(Immediate::I32(1)),
                at(ptr(), 0),
            ))
            .instr(store(
                i32(),
                Operand::Immediate(Immediate::I32(2)),
                at(value(3), 0),
            ));
        for inst in extra {
            builder = builder.instr(inst);
        }
        builder
            .instr(load(i32(), value(4), at(ptr(), 0)))
            .instr(load(i32(), value(5), at(value(3), 0)))
            .instr(ret(value(4)))
            .build()
    }

    #[test]
    fn splits_fields_so_mem2reg_promotes_them() {
        let mut func = two_field_struct(Vec::new());

        assert!(apply_pass(&ScalarReplacement, &mut func));
        assert!(!has_lea(&func));
        assert_eq!(stack_addr_slots(&func).len(), 2);
        assert!(!stack_addr_slots(&func).contains(&slot()));

        assert!(apply_pass(&Mem2Reg, &mut func));
        assert_eq!(stack_addr_slots(&func).len(), 0);
        assert_eq!(
            count(&func, |i| matches!(
                i,
                Instruction::Load { .. } | Instruction::Store { .. }
            )),
            0
        );
    }

    #[test]
    fn leaves_slot_without_lea_alone() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr(), 0)))
            .instr(load(i64(), value(3), at(ptr(), 0)))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_pointer_passed_to_call() {
        let mut func = two_field_struct(vec![Instruction::Call {
            name: "sink".into(),
            args: vec![Operand::Register(value(3))],
            ret: None,
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
        assert!(has_lea(&func));
    }

    #[test]
    fn refuses_field_pointer_stored_to_memory() {
        let mut func = two_field_struct(vec![store(
            MirType::Scalar(ScalarType::Ptr),
            Operand::Register(value(3)),
            at(cond(), 0),
        )]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
        assert!(has_lea(&func));
    }

    #[test]
    fn refuses_pointer_copied_to_register() {
        let mut func = two_field_struct(vec![Instruction::Copy {
            ty: MirType::Scalar(ScalarType::Ptr),
            dst: value(6),
            src: Operand::Register(value(3)),
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_pointer_returned() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), 4))
            .instr(store(
                i32(),
                Operand::Immediate(Immediate::I32(1)),
                at(value(3), 0),
            ))
            .instr(Instruction::Ret {
                value: Some(Operand::Register(ptr())),
            })
            .build();
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_non_constant_index() {
        let mut func = two_field_struct(vec![load(
            i32(),
            value(6),
            AddressMode::BaseIndexScale {
                address_space: Default::default(),
                base: value(3),
                index: cond(),
                scale: 4,
                offset: 0,
            },
        )]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_volatile_field() {
        let mut func = two_field_struct(vec![Instruction::Store {
            ty: i32(),
            src: imm(9),
            addr: at(value(3), 0),
            attrs: volatile(),
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_atomic_field() {
        let mut func = two_field_struct(vec![Instruction::AtomicLoad {
            ty: i32(),
            dst: value(6),
            addr: at(value(3), 0),
            ordering: MemoryOrdering::SeqCst,
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_overlapping_fields() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), 2))
            .instr(store(i64(), imm(1), at(ptr(), 0)))
            .instr(load(i32(), value(4), at(value(3), 0)))
            .instr(ret(value(4)))
            .build();
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_mixed_type_at_one_offset() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), 0))
            .instr(store(i64(), imm(1), at(value(3), 0)))
            .instr(load(
                MirType::Scalar(ScalarType::Ptr),
                value(4),
                at(ptr(), 0),
            ))
            .instr(ret(value(4)))
            .build();
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_access_past_slot_end() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), 8))
            .instr(store(i64(), imm(1), at(value(3), 0)))
            .instr(ret(value(3)))
            .build();
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_negative_displacement() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(lea(value(3), ptr(), -4))
            .instr(store(
                i32(),
                Operand::Immediate(Immediate::I32(1)),
                at(value(3), 0),
            ))
            .instr(load(i32(), value(4), at(value(3), 0)))
            .instr(ret(value(4)))
            .build();
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_slot_named_as_register() {
        let mut func = two_field_struct(vec![Instruction::Copy {
            ty: i64(),
            dst: value(6),
            src: Operand::Register(Register::Virtual(slot())),
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }

    #[test]
    fn refuses_second_stack_address_of_slot() {
        let mut func = two_field_struct(vec![Instruction::StackAddr {
            dst: value(6),
            slot: slot(),
        }]);
        assert!(!apply_pass(&ScalarReplacement, &mut func));
    }
}
