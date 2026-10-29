//! Promotes stack slots into registers.
//!
//! A slot is promoted when every use of its address is a whole-slot, offset-zero load or
//! store of one scalar type, the slot is never read before it is written on some path,
//! and no access is volatile. A promoted slot becomes one virtual register: each store
//! copies into it and each load copies out of it. MIR registers may be defined on several
//! control-flow edges (phi lowering does this), so no merge placement is needed.

use crate::mir::{
    AddressMode, AddressSpace, Block, Function, Instruction, MirType, Operand, Register,
    RegisterClass, ScalarType, VirtualReg,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_mir::Located;
use std::collections::{HashMap, HashSet};
use std::mem;

/// Promotes scalar stack slots to registers.
#[derive(Default)]
pub struct Mem2Reg;

impl Transform for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn description(&self) -> &'static str {
        "Promotes scalar stack slots that are only loaded and stored whole into registers"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::MemoryOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        Ok(promote(func))
    }
}

#[derive(Default)]
struct SlotInfo {
    pointers: HashSet<Register>,
    ty: Option<MirType>,
    viable: bool,
}

fn promotable_type(ty: MirType) -> bool {
    matches!(
        ty,
        MirType::Scalar(
            ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::F32
                | ScalarType::F64
                | ScalarType::Ptr
        )
    )
}

fn register_class(ty: MirType) -> RegisterClass {
    match ty {
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => RegisterClass::Fpr,
        _ => RegisterClass::Gpr,
    }
}

fn direct_access_base(addr: &AddressMode) -> Option<&Register> {
    match addr {
        AddressMode::BaseOffset {
            base,
            offset: 0,
            address_space,
        } if *address_space == AddressSpace::default() => Some(base),
        _ => None,
    }
}

/// Returns the (type, volatile) of a load or store that addresses `pointer` directly.
/// Any other instruction that names `pointer` is a use the pass cannot follow.
fn direct_access(inst: &Instruction, pointer: &Register, uses: usize) -> Option<(MirType, bool)> {
    if uses != 1 {
        return None;
    }
    match inst {
        Instruction::Load {
            ty, addr, attrs, ..
        } if direct_access_base(addr) == Some(pointer) => Some((*ty, attrs.volatile)),
        Instruction::Store {
            ty, addr, attrs, ..
        } if direct_access_base(addr) == Some(pointer) => Some((*ty, attrs.volatile)),
        _ => None,
    }
}

fn promote(func: &mut Function) -> bool {
    let mut slots: HashMap<VirtualReg, SlotInfo> = HashMap::new();
    let mut defs: HashMap<Register, usize> = HashMap::new();
    let mut mentioned: HashSet<Register> = HashSet::new();

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
            if let Instruction::StackAddr { dst, slot } = &inst.instr {
                slots
                    .entry(*slot)
                    .or_insert_with(|| SlotInfo {
                        viable: true,
                        ..SlotInfo::default()
                    })
                    .pointers
                    .insert(dst.clone());
            }
        }
    }
    if slots.is_empty() {
        return false;
    }

    for (slot, info) in &mut slots {
        if func.stack_reservations.contains_key(slot)
            || mentioned.contains(&Register::Virtual(*slot))
        {
            info.viable = false;
        }
        for pointer in &info.pointers {
            if defs.get(pointer).copied().unwrap_or(0) != 1 {
                info.viable = false;
            }
        }
    }

    let mut pointer_slot: HashMap<Register, VirtualReg> = HashMap::new();
    for (slot, info) in &slots {
        for pointer in &info.pointers {
            pointer_slot.insert(pointer.clone(), *slot);
        }
    }

    for block in &func.blocks {
        for inst in &block.instructions {
            let uses = inst.use_regs();
            for (pointer, slot) in &pointer_slot {
                let count = uses.iter().filter(|r| **r == pointer).count();
                if count == 0 {
                    continue;
                }
                let Some(info) = slots.get_mut(slot) else {
                    continue;
                };
                match direct_access(inst, pointer, count) {
                    Some((ty, volatile)) => {
                        if volatile || !promotable_type(ty) {
                            info.viable = false;
                        }
                        match info.ty {
                            None => info.ty = Some(ty),
                            Some(existing) if existing != ty => info.viable = false,
                            Some(_) => {}
                        }
                    }
                    None => info.viable = false,
                }
            }
        }
    }

    let mut viable: Vec<VirtualReg> = slots
        .iter()
        .filter(|(_, info)| info.viable)
        .map(|(slot, _)| *slot)
        .collect();
    viable.sort_by_key(|slot| (slot.id, slot.class));
    let index: HashMap<VirtualReg, usize> = viable
        .iter()
        .enumerate()
        .map(|(i, slot)| (*slot, i))
        .collect();
    if viable.is_empty() {
        return false;
    }

    let entry_index = func.blocks.iter().position(|b| b.label == func.entry);
    let Some(entry_index) = entry_index else {
        return false;
    };
    let label_index: HashMap<&str, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
    for (i, block) in func.blocks.iter().enumerate() {
        for succ in block.successors() {
            if let Some(&j) = label_index.get(succ) {
                preds[j].push(i);
            }
        }
    }

    let slot_of_pointer = |reg: &Register| -> Option<usize> {
        pointer_slot
            .get(reg)
            .and_then(|slot| index.get(slot).copied())
    };
    let n = viable.len();
    let mut gen_: Vec<Vec<bool>> = vec![vec![false; n]; func.blocks.len()];
    for (b, block) in func.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Instruction::Store { addr, .. } = &inst.instr
                && let Some(base) = direct_access_base(addr)
                && let Some(s) = slot_of_pointer(base)
            {
                gen_[b][s] = true;
            }
        }
    }

    // Must-analysis: a slot is assigned at a point when every path from entry stores it.
    let nb = func.blocks.len();
    let mut inn: Vec<Vec<bool>> = vec![vec![true; n]; nb];
    let mut out: Vec<Vec<bool>> = vec![vec![true; n]; nb];
    loop {
        let mut changed = false;
        for b in 0..nb {
            let new_in: Vec<bool> = if b == entry_index {
                vec![false; n]
            } else if preds[b].is_empty() {
                vec![true; n]
            } else {
                (0..n)
                    .map(|s| preds[b].iter().all(|&p| out[p][s]))
                    .collect()
            };
            let new_out: Vec<bool> = (0..n).map(|s| new_in[s] || gen_[b][s]).collect();
            if new_in != inn[b] || new_out != out[b] {
                changed = true;
            }
            inn[b] = new_in;
            out[b] = new_out;
        }
        if !changed {
            break;
        }
    }

    let mut refused: HashSet<usize> = HashSet::new();
    for (b, block) in func.blocks.iter().enumerate() {
        let mut assigned = inn[b].clone();
        for inst in &block.instructions {
            match &inst.instr {
                Instruction::Load { addr, .. } => {
                    if let Some(base) = direct_access_base(addr)
                        && let Some(s) = slot_of_pointer(base)
                        && !assigned[s]
                    {
                        refused.insert(s);
                    }
                }
                Instruction::Store { addr, .. } => {
                    if let Some(base) = direct_access_base(addr)
                        && let Some(s) = slot_of_pointer(base)
                    {
                        assigned[s] = true;
                    }
                }
                _ => {}
            }
        }
    }

    let mut next_id = mentioned
        .iter()
        .filter_map(|reg| match reg {
            Register::Virtual(v) => Some(v.id),
            Register::Physical(_) => None,
        })
        .chain(viable.iter().map(|slot| slot.id))
        .chain(func.stack_reservations.keys().map(|slot| slot.id))
        .max()
        .map_or(0, |max| max + 1);

    let mut promoted: HashSet<VirtualReg> = HashSet::new();
    let mut slot_reg: HashMap<VirtualReg, Register> = HashMap::new();
    for (i, slot) in viable.iter().enumerate() {
        if refused.contains(&i) {
            continue;
        }
        promoted.insert(*slot);
        if let Some(ty) = slots[slot].ty {
            let reg = Register::Virtual(VirtualReg::new(next_id, register_class(ty)));
            next_id += 1;
            slot_reg.insert(*slot, reg);
        }
    }
    if promoted.is_empty() {
        return false;
    }

    let promoted_pointer: HashMap<Register, VirtualReg> = pointer_slot
        .into_iter()
        .filter(|(_, slot)| promoted.contains(slot))
        .collect();
    let promoted_access = |addr: &AddressMode| -> Option<VirtualReg> {
        direct_access_base(addr).and_then(|base| promoted_pointer.get(base).copied())
    };

    for slot in &promoted {
        func.stack_alignments.remove(slot);
    }
    for block in &mut func.blocks {
        rewrite_block(block, &promoted, &promoted_access, &slot_reg);
    }
    true
}

fn rewrite_block(
    block: &mut Block,
    promoted: &HashSet<VirtualReg>,
    promoted_access: &impl Fn(&AddressMode) -> Option<VirtualReg>,
    slot_reg: &HashMap<VirtualReg, Register>,
) {
    let instructions = mem::take(&mut block.instructions);
    block.instructions = instructions
        .into_iter()
        .filter_map(|inst| {
            let loc = inst.loc;
            let replacement = match inst.instr {
                Instruction::StackAddr { slot, .. } if promoted.contains(&slot) => None,
                Instruction::Load {
                    ty,
                    dst,
                    addr,
                    attrs,
                } => match promoted_access(&addr) {
                    Some(slot) => {
                        let mut copy = Instruction::Copy {
                            ty,
                            dst,
                            src: Operand::Register(slot_reg[&slot].clone()),
                        };
                        copy.normalize_integer_immediates(None);
                        Some(copy)
                    }
                    None => Some(Instruction::Load {
                        ty,
                        dst,
                        addr,
                        attrs,
                    }),
                },
                Instruction::Store {
                    ty,
                    src,
                    addr,
                    attrs,
                } => match promoted_access(&addr) {
                    Some(slot) => {
                        let mut copy = Instruction::Copy {
                            ty,
                            dst: slot_reg[&slot].clone(),
                            src,
                        };
                        copy.normalize_integer_immediates(None);
                        Some(copy)
                    }
                    None => Some(Instruction::Store {
                        ty,
                        src,
                        addr,
                        attrs,
                    }),
                },
                other => Some(other),
            }?;
            Some(Located::at(replacement, loc))
        })
        .collect();
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, IntCmpOp, MemoryAttrs, MemoryOrdering,
        transform::test_utils::{apply_pass, get_block},
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

    fn at(base: Register) -> AddressMode {
        AddressMode::BaseOffset {
            base,
            offset: 0,
            address_space: Default::default(),
        }
    }

    fn plain() -> MemoryAttrs {
        MemoryAttrs::default()
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

    fn store(ty: MirType, src: Operand, addr: AddressMode) -> Instruction {
        Instruction::Store {
            ty,
            src,
            addr,
            attrs: plain(),
        }
    }

    fn load(ty: MirType, dst: Register, addr: AddressMode) -> Instruction {
        Instruction::Load {
            ty,
            dst,
            addr,
            attrs: plain(),
        }
    }

    fn ret(reg: Register) -> Instruction {
        Instruction::Ret {
            value: Some(Operand::Register(reg)),
        }
    }

    fn has_stack_addr(func: &Function) -> bool {
        func.blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(&i.instr, Instruction::StackAddr { .. }))
    }

    fn has_memory_access(func: &Function) -> bool {
        func.blocks.iter().flat_map(|b| &b.instructions).any(|i| {
            matches!(
                &i.instr,
                Instruction::Load { .. } | Instruction::Store { .. }
            )
        })
    }

    fn copy_src_reg(block: &Block, index: usize) -> Register {
        match &block.instructions[index].instr {
            Instruction::Copy {
                src: Operand::Register(r),
                ..
            } => r.clone(),
            other => panic!("expected register copy, got {other}"),
        }
    }

    #[test]
    fn promotes_slot_across_branch() {
        let mut func = FunctionBuilder::new("f")
            .param(cond(), i64())
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(Instruction::Br {
                cond: cond(),
                true_target: "then".into(),
                false_target: "join".into(),
            })
            .block("then")
            .instr(store(i64(), imm(2), at(ptr())))
            .instr(Instruction::Jmp {
                target: "join".into(),
            })
            .block("join")
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(apply_pass(&Mem2Reg, &mut func));
        assert!(!has_stack_addr(&func));
        assert!(!has_memory_access(&func));

        let entry_dst = match &get_block(&func, "entry").instructions[0].instr {
            Instruction::Copy { dst, .. } => dst.clone(),
            other => panic!("expected copy, got {other}"),
        };
        let then_dst = match &get_block(&func, "then").instructions[0].instr {
            Instruction::Copy { dst, .. } => dst.clone(),
            other => panic!("expected copy, got {other}"),
        };
        assert_eq!(entry_dst, then_dst);
        assert_eq!(copy_src_reg(get_block(&func, "join"), 0), entry_dst);
    }

    #[test]
    fn promotes_slot_across_loop() {
        let mut func = FunctionBuilder::new("f")
            .param(cond(), i64())
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(0), at(ptr())))
            .instr(Instruction::Jmp {
                target: "head".into(),
            })
            .block("head")
            .instr(load(i64(), value(3), at(ptr())))
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64(),
                dst: value(4),
                lhs: Operand::Register(value(3)),
                rhs: imm(1),
            })
            .instr(store(i64(), Operand::Register(value(4)), at(ptr())))
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: i64(),
                dst: cond(),
                lhs: Operand::Register(value(4)),
                rhs: imm(10),
            })
            .instr(Instruction::Br {
                cond: cond(),
                true_target: "head".into(),
                false_target: "exit".into(),
            })
            .block("exit")
            .instr(load(i64(), value(5), at(ptr())))
            .instr(ret(value(5)))
            .build();

        assert!(apply_pass(&Mem2Reg, &mut func));
        assert!(!has_stack_addr(&func));
        assert!(!has_memory_access(&func));
    }

    #[test]
    fn refuses_address_passed_to_call() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(Instruction::Call {
                name: "sink".into(),
                args: vec![Operand::Register(ptr())],
                ret: None,
            })
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_address_copied_to_register() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(Instruction::Copy {
                ty: MirType::Scalar(ScalarType::Ptr),
                dst: value(3),
                src: Operand::Register(ptr()),
            })
            .instr(store(i64(), imm(1), at(value(3))))
            .instr(load(i64(), value(4), at(ptr())))
            .instr(ret(value(4)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_address_stored_to_memory() {
        let mut func = FunctionBuilder::new("f")
            .param(cond(), i64())
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), Operand::Register(ptr()), at(cond())))
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_address_stored_into_itself() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(
                MirType::Scalar(ScalarType::Ptr),
                Operand::Register(ptr()),
                at(ptr()),
            ))
            .instr(load(MirType::Scalar(ScalarType::Ptr), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_volatile_store() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(Instruction::Store {
                ty: i64(),
                src: imm(1),
                addr: at(ptr()),
                attrs: volatile(),
            })
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_volatile_load() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(Instruction::Load {
                ty: i64(),
                dst: value(3),
                addr: at(ptr()),
                attrs: volatile(),
            })
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_atomic_access() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(Instruction::AtomicLoad {
                ty: i64(),
                dst: value(3),
                addr: at(ptr()),
                ordering: MemoryOrdering::SeqCst,
            })
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_mixed_width_accesses() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(
                i32(),
                Operand::Immediate(Immediate::I32(1)),
                at(ptr()),
            ))
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_partial_access() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(load(
                i64(),
                value(3),
                AddressMode::BaseOffset {
                    base: ptr(),
                    offset: 4,
                    address_space: Default::default(),
                },
            ))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_indexed_access() {
        let mut func = FunctionBuilder::new("f")
            .param(cond(), i64())
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(load(
                i64(),
                value(3),
                AddressMode::BaseIndexScale {
                    address_space: Default::default(),
                    base: ptr(),
                    index: cond(),
                    scale: 1,
                    offset: 0,
                },
            ))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_read_before_write_on_some_path() {
        let mut func = FunctionBuilder::new("f")
            .param(cond(), i64())
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(Instruction::Br {
                cond: cond(),
                true_target: "then".into(),
                false_target: "join".into(),
            })
            .block("then")
            .instr(store(i64(), imm(2), at(ptr())))
            .instr(Instruction::Jmp {
                target: "join".into(),
            })
            .block("join")
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_read_before_write_in_block() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(load(i64(), value(3), at(ptr())))
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(ret(value(3)))
            .build();

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }

    #[test]
    fn refuses_multi_slot_reservation() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(stack_addr())
            .instr(store(i64(), imm(1), at(ptr())))
            .instr(load(i64(), value(3), at(ptr())))
            .instr(ret(value(3)))
            .build();
        func.stack_reservations.insert(slot(), 1);

        assert!(!apply_pass(&Mem2Reg, &mut func));
        assert!(has_stack_addr(&func));
    }
}
