//! Memory optimization transforms for MIR.

use crate::mir::{
    AddressMode, Function, Instruction, MirType, Operand, Register,
    transform::{
        Transform, TransformCategory, TransformError, TransformLevel,
        alias::{Access, AliasResult, alias, scan_block},
    },
};

const MAX_CACHED: usize = 64;

struct Cached<V> {
    base: Register,
    access: Access,
    ty: MirType,
    value: V,
}

fn same_range<V>(entry: &Cached<V>, ty: MirType, access: &Access) -> bool {
    entry.ty == ty && alias(&entry.access, access) == AliasResult::SameRange
}

fn may_overlap<V>(entries: &mut Vec<Cached<V>>, access: &Access) {
    entries.retain(|entry| alias(&entry.access, access) == AliasResult::NoAlias);
}

/// Memory optimization transform performing redundant load elimination.
#[derive(Default)]
pub struct MemoryOptimization;

impl Transform for MemoryOptimization {
    fn name(&self) -> &'static str {
        "memory_optimization"
    }

    fn description(&self) -> &'static str {
        "Optimizes memory operations including redundant load elimination"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::MemoryOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl MemoryOptimization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        if self.forward_redundant_loads(func) {
            changed = true;
        }

        Ok(changed)
    }

    /// Redundant load elimination and store-to-load forwarding within basic blocks.
    ///
    /// A store only drops the remembered entries it may overlap, so entries for
    /// disjoint bytes survive it. Calls, atomics, volatile and unknown accesses drop
    /// everything.
    fn forward_redundant_loads(&self, func: &mut Function) -> bool {
        let mut changed = false;
        let Function {
            blocks,
            stack_reservations,
            ..
        } = func;

        for block in blocks.iter_mut() {
            let accesses = scan_block(block, stack_reservations);
            let mut stores: Vec<Cached<Operand>> = Vec::new();
            let mut loads: Vec<Cached<Register>> = Vec::new();

            for (inst, access) in block.instructions.iter_mut().zip(accesses) {
                // What the caches hold is only good while the registers they name still
                // hold the same values, so a definition drops every entry naming it.
                if let Some(defined) = inst.def_reg().cloned() {
                    stores.retain(|entry| {
                        entry.base != defined
                            && !matches!(&entry.value, Operand::Register(reg) if *reg == defined)
                    });
                    loads.retain(|entry| entry.base != defined && entry.value != defined);
                }

                let mut replacement: Option<Instruction> = None;
                match &inst.instr {
                    Instruction::Store {
                        ty,
                        src,
                        addr: AddressMode::BaseOffset { base, .. },
                        attrs,
                    } if !attrs.volatile => match access {
                        Some(access) if access.base_known() => {
                            may_overlap(&mut stores, &access);
                            may_overlap(&mut loads, &access);
                            stores.push(Cached {
                                base: base.clone(),
                                access,
                                ty: *ty,
                                value: src.clone(),
                            });
                            if stores.len() + loads.len() > MAX_CACHED {
                                stores.clear();
                                loads.clear();
                            }
                        }
                        _ => {
                            stores.clear();
                            loads.clear();
                        }
                    },
                    Instruction::Load {
                        ty,
                        dst,
                        addr: AddressMode::BaseOffset { base, .. },
                        attrs,
                    } if !attrs.volatile => match access {
                        Some(access) if access.base_known() => {
                            let source = stores
                                .iter()
                                .find(|entry| same_range(entry, *ty, &access))
                                .map(|entry| entry.value.clone())
                                .or_else(|| {
                                    loads
                                        .iter()
                                        .find(|entry| same_range(entry, *ty, &access))
                                        .map(|entry| Operand::Register(entry.value.clone()))
                                });
                            if let Some(src) = source {
                                replacement = Some(Instruction::Copy {
                                    ty: *ty,
                                    dst: dst.clone(),
                                    src,
                                });
                            }
                            loads.retain(|entry| {
                                entry.ty != *ty
                                    || alias(&entry.access, &access) != AliasResult::SameRange
                            });
                            loads.push(Cached {
                                base: base.clone(),
                                access,
                                ty: *ty,
                                value: dst.clone(),
                            });
                            if stores.len() + loads.len() > MAX_CACHED {
                                stores.clear();
                                loads.clear();
                            }
                        }
                        _ => {
                            stores.clear();
                            loads.clear();
                        }
                    },
                    Instruction::Store { .. } | Instruction::Load { .. } => {
                        stores.clear();
                        loads.clear();
                    }
                    Instruction::Call { .. }
                    | Instruction::CallIndirect { .. }
                    | Instruction::TailCall { .. } => {
                        // Calls may read/write memory; conservatively invalidate
                        stores.clear();
                        loads.clear();
                    }
                    Instruction::AtomicLoad { .. }
                    | Instruction::AtomicStore { .. }
                    | Instruction::AtomicBinary { .. }
                    | Instruction::AtomicCompareExchange { .. }
                    | Instruction::VaStart { .. }
                    | Instruction::VaArg { .. }
                    | Instruction::VaEnd { .. }
                    | Instruction::VaCopy { .. }
                    | Instruction::InlineAsm { .. }
                    | Instruction::Fence { .. } => {
                        stores.clear();
                        loads.clear();
                    }
                    _ => {}
                }

                if let Some(new_inst) = replacement {
                    *inst = new_inst.into();
                    changed = true;
                }
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        AddressMode, AtomicBinOp, FunctionBuilder, Immediate, MemoryAttrs, MemoryOrdering, MirType,
        Operand, ScalarType, VirtualReg, transform::test_utils::apply_pass,
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn base_offset(base: Register, offset: i16) -> AddressMode {
        AddressMode::BaseOffset {
            base,
            offset,
            address_space: Default::default(),
        }
    }

    fn non_volatile() -> MemoryAttrs {
        MemoryAttrs {
            volatile: false,
            ..MemoryAttrs::default()
        }
    }

    fn volatile() -> MemoryAttrs {
        MemoryAttrs {
            volatile: true,
            ..MemoryAttrs::default()
        }
    }

    #[test]
    fn redundant_load_replaced_with_copy() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst1: Register = VirtualReg::gpr(1).into();
        let dst2: Register = VirtualReg::gpr(2).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst1.clone(),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst2.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst2)),
            })
            .build();

        let changed = apply_pass(&MemoryOptimization, &mut func);
        assert!(changed, "second load should have been replaced");
        let entry = func.get_block("entry").unwrap();
        let second = &entry.instructions[1];
        assert!(matches!(&second.instr, Instruction::Copy { .. }));
    }

    #[test]
    fn store_to_load_forwarded() {
        let base: Register = VirtualReg::gpr(0).into();
        let src: Operand = Operand::Immediate(Immediate::I64(99));
        let dst: Register = VirtualReg::gpr(1).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Store {
                ty: i64(),
                src: src.clone(),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        let changed = apply_pass(&MemoryOptimization, &mut func);
        assert!(
            changed,
            "load after store to same address should be forwarded"
        );
        let entry = func.get_block("entry").unwrap();
        assert!(matches!(
            &entry.instructions[1].instr,
            Instruction::Copy { .. }
        ));
    }

    #[test]
    fn aliasing_stores_and_mismatched_types_prevent_forwarding() {
        for (second_base, offset, load_ty, store_ty) in [
            (1, 0, i64(), i64()),
            (0, 4, i64(), MirType::Scalar(ScalarType::I32)),
            (0, 0, MirType::Scalar(ScalarType::I32), i64()),
        ] {
            let base: Register = VirtualReg::gpr(0).into();
            let mut func = FunctionBuilder::new("f")
                .block("entry")
                .instr(Instruction::Store {
                    ty: i64(),
                    src: Operand::Immediate(Immediate::I64(99)),
                    addr: base_offset(base.clone(), 0),
                    attrs: non_volatile(),
                })
                .instr(Instruction::Store {
                    ty: store_ty,
                    src: Operand::Register(VirtualReg::gpr(2).into()),
                    addr: base_offset(VirtualReg::gpr(second_base).into(), offset),
                    attrs: non_volatile(),
                })
                .instr(Instruction::Load {
                    ty: load_ty,
                    dst: VirtualReg::gpr(3).into(),
                    addr: base_offset(base, 0),
                    attrs: non_volatile(),
                })
                .build();
            assert!(!apply_pass(&MemoryOptimization, &mut func));
            assert!(matches!(
                &func.blocks[0].instructions[2].instr,
                Instruction::Load { .. }
            ));
        }
    }

    #[test]
    fn mixed_width_loads_prevent_redundant_load_forwarding() {
        let base: Register = VirtualReg::gpr(0).into();
        let mut func = FunctionBuilder::new("f")
            .block("entry")
            .instr(Instruction::Load {
                ty: i64(),
                dst: VirtualReg::gpr(1).into(),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I32),
                dst: VirtualReg::gpr(2).into(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .build();
        assert!(!apply_pass(&MemoryOptimization, &mut func));
    }

    #[test]
    fn volatile_load_not_forwarded() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst1: Register = VirtualReg::gpr(1).into();
        let dst2: Register = VirtualReg::gpr(2).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst1,
                addr: base_offset(base.clone(), 0),
                attrs: volatile(),
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst2.clone(),
                addr: base_offset(base, 0),
                attrs: volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst2)),
            })
            .build();

        let changed = apply_pass(&MemoryOptimization, &mut func);
        assert!(!changed, "volatile loads must not be eliminated");
    }

    #[test]
    fn store_not_forwarded_to_volatile_load() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst: Register = VirtualReg::gpr(1).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Store {
                ty: i64(),
                src: Operand::Immediate(Immediate::I64(99)),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst.clone(),
                addr: base_offset(base, 0),
                attrs: volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        assert!(!apply_pass(&MemoryOptimization, &mut func));
        assert!(matches!(
            &func.blocks[0].instructions[1].instr,
            Instruction::Load { .. }
        ));
    }

    #[test]
    fn volatile_store_not_forwarded_to_load() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst: Register = VirtualReg::gpr(1).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Store {
                ty: i64(),
                src: Operand::Immediate(Immediate::I64(99)),
                addr: base_offset(base.clone(), 0),
                attrs: volatile(),
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        assert!(!apply_pass(&MemoryOptimization, &mut func));
        assert!(matches!(
            &func.blocks[0].instructions[1].instr,
            Instruction::Load { .. }
        ));
    }

    #[test]
    fn call_invalidates_store_forwarding() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst: Register = VirtualReg::gpr(1).into();
        let result: Register = VirtualReg::gpr(2).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Store {
                ty: i64(),
                src: Operand::Immediate(Immediate::I64(7)),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Call {
                ret: Some(result),
                name: "side_effect".to_owned(),
                args: vec![],
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        let changed = apply_pass(&MemoryOptimization, &mut func);
        assert!(!changed, "call should have invalidated forwarding state");
        let entry = func.get_block("entry").unwrap();
        assert!(matches!(
            &entry.instructions[2].instr,
            Instruction::Load { .. }
        ));
    }

    #[test]
    fn load_is_not_forwarded_across_atomic_exchange() {
        let base: Register = VirtualReg::gpr(0).into();
        let dst: Register = VirtualReg::gpr(1).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Store {
                ty: i64(),
                src: Operand::Immediate(Immediate::I64(1)),
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::AtomicBinary {
                op: AtomicBinOp::Exchange,
                ty: i64(),
                dst: VirtualReg::gpr(2).into(),
                addr: base_offset(base.clone(), 0),
                value: Operand::Immediate(Immediate::I64(7)),
                ordering: MemoryOrdering::SeqCst,
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: dst.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        assert!(!apply_pass(&MemoryOptimization, &mut func));
        assert!(matches!(
            &func.blocks[0].instructions[2].instr,
            Instruction::Load { .. }
        ));
    }

    #[test]
    fn load_is_not_merged_across_a_fence() {
        let base: Register = VirtualReg::gpr(0).into();
        let first: Register = VirtualReg::gpr(1).into();
        let second: Register = VirtualReg::gpr(2).into();
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Load {
                ty: i64(),
                dst: first,
                addr: base_offset(base.clone(), 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Fence {
                ordering: MemoryOrdering::SeqCst,
            })
            .instr(Instruction::Load {
                ty: i64(),
                dst: second.clone(),
                addr: base_offset(base, 0),
                attrs: non_volatile(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(second)),
            })
            .build();

        assert!(!apply_pass(&MemoryOptimization, &mut func));
        assert!(matches!(
            &func.blocks[0].instructions[2].instr,
            Instruction::Load { .. }
        ));
    }

    fn stack_addr(dst: u32, slot: u32) -> Instruction {
        Instruction::StackAddr {
            dst: VirtualReg::gpr(dst).into(),
            slot: VirtualReg::gpr(100 + slot),
        }
    }

    fn store_imm(base: u32, offset: i16, value: i64) -> Instruction {
        Instruction::Store {
            ty: i64(),
            src: Operand::Immediate(Immediate::I64(value)),
            addr: base_offset(VirtualReg::gpr(base).into(), offset),
            attrs: non_volatile(),
        }
    }

    fn load_into(dst: u32, base: u32, offset: i16) -> Instruction {
        Instruction::Load {
            ty: i64(),
            dst: VirtualReg::gpr(dst).into(),
            addr: base_offset(VirtualReg::gpr(base).into(), offset),
            attrs: non_volatile(),
        }
    }

    fn forwarded_block(instructions: Vec<Instruction>) -> Vec<Instruction> {
        forwarded_block_with(instructions, &[])
    }

    fn forwarded_block_with(
        instructions: Vec<Instruction>,
        words: &[(u32, usize)],
    ) -> Vec<Instruction> {
        let mut func = FunctionBuilder::new("f").block("entry");
        for inst in instructions {
            func = func.instr(inst);
        }
        let mut func = func.build();
        for &(slot, extra) in words {
            func.stack_reservations
                .insert(VirtualReg::gpr(100 + slot), extra);
        }
        apply_pass(&MemoryOptimization, &mut func);
        func.blocks[0]
            .instructions
            .iter()
            .map(|i| i.instr.clone())
            .collect()
    }

    #[test]
    fn store_to_disjoint_field_keeps_forwarding() {
        let out = forwarded_block_with(
            vec![
                stack_addr(0, 0),
                store_imm(0, 0, 7),
                store_imm(0, 8, 8),
                load_into(1, 0, 0),
            ],
            &[(0, 1)],
        );
        assert!(matches!(out[3], Instruction::Copy { .. }));
    }

    #[test]
    fn store_to_another_slot_keeps_forwarding() {
        let out = forwarded_block(vec![
            stack_addr(0, 0),
            stack_addr(1, 1),
            store_imm(0, 0, 7),
            store_imm(1, 0, 8),
            load_into(2, 0, 0),
        ]);
        assert!(matches!(out[4], Instruction::Copy { .. }));
    }

    #[test]
    fn overlapping_field_store_kills_forwarding() {
        let out = forwarded_block(vec![
            stack_addr(0, 0),
            store_imm(0, 0, 7),
            store_imm(0, 4, 8),
            load_into(1, 0, 0),
        ]);
        assert!(matches!(out[3], Instruction::Load { .. }));
    }

    #[test]
    fn store_through_unrelated_pointer_kills_forwarding() {
        let out = forwarded_block(vec![
            store_imm(0, 0, 7),
            store_imm(1, 8, 8),
            load_into(2, 0, 0),
        ]);
        assert!(matches!(out[2], Instruction::Load { .. }));
    }

    #[test]
    fn redefined_stored_value_is_not_forwarded() {
        let out = forwarded_block(vec![
            stack_addr(0, 0),
            Instruction::Copy {
                ty: i64(),
                dst: VirtualReg::gpr(5).into(),
                src: Operand::Immediate(Immediate::I64(5)),
            },
            Instruction::Store {
                ty: i64(),
                src: Operand::Register(VirtualReg::gpr(5).into()),
                addr: base_offset(VirtualReg::gpr(0).into(), 0),
                attrs: non_volatile(),
            },
            Instruction::Copy {
                ty: i64(),
                dst: VirtualReg::gpr(5).into(),
                src: Operand::Immediate(Immediate::I64(9)),
            },
            store_imm(0, 8, 1),
            load_into(1, 0, 0),
        ]);
        assert!(matches!(out[5], Instruction::Load { .. }));
    }
}
