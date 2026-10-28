//! Memory optimization transforms for MIR.

use crate::mir::{
    AddressMode, AddressSpace, Function, Instruction, MirType, Operand, Register,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::collections::HashMap;

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
    fn forward_redundant_loads(&self, func: &mut Function) -> bool {
        let mut changed = false;

        for block in &mut func.blocks {
            let mut last_store: HashMap<(Register, i16, AddressSpace, MirType), Operand> =
                HashMap::new();
            let mut last_load: HashMap<(Register, i16, AddressSpace, MirType), Register> =
                HashMap::new();

            for inst in &mut block.instructions {
                // What the maps hold is only good while the registers it names still
                // hold the same values, so a definition drops every entry naming it.
                if let Some(defined) = inst.def_reg() {
                    last_store.retain(|(base, ..), stored| {
                        base != defined
                            && !matches!(stored, Operand::Register(reg) if reg == defined)
                    });
                    last_load.retain(|(base, ..), loaded| base != defined && loaded != defined);
                }

                let mut replacement: Option<Instruction> = None;
                match &*inst {
                    Instruction::Store {
                        ty,
                        src,
                        addr,
                        attrs,
                    } => match addr {
                        AddressMode::BaseOffset {
                            base,
                            offset,
                            address_space,
                        } if !attrs.volatile => {
                            last_store.clear();
                            last_store
                                .insert((base.clone(), *offset, *address_space, *ty), src.clone());
                            last_load.clear();
                        }
                        _ => {
                            last_store.clear();
                            last_load.clear();
                        }
                    },
                    Instruction::Load {
                        ty,
                        dst,
                        addr,
                        attrs,
                    } => {
                        // Skip volatile loads
                        if attrs.volatile {
                            last_store.clear();
                            last_load.clear();
                        } else if let AddressMode::BaseOffset {
                            base,
                            offset,
                            address_space,
                        } = addr
                        {
                            let key = (base.clone(), *offset, *address_space, *ty);
                            if let Some(stored_val) = last_store.get(&key) {
                                // store-to-load forwarding: replace load with a copy of stored_val
                                replacement = Some(Instruction::Copy {
                                    ty: *ty,
                                    dst: dst.clone(),
                                    src: stored_val.clone(),
                                });
                                // record latest load destination
                                last_load.insert(key, dst.clone());
                            } else if let Some(prev_reg) = last_load.get(&key) {
                                // Redundant load: reuse previous loaded register
                                replacement = Some(Instruction::Copy {
                                    ty: *ty,
                                    dst: dst.clone(),
                                    src: Operand::Register(prev_reg.clone()),
                                });
                                // record latest load destination
                                last_load.insert(key, dst.clone());
                            } else {
                                // first load of this address
                                last_load.insert(key, dst.clone());
                            }
                        } else {
                            // Unknown addressing, be conservative
                            last_store.clear();
                            last_load.clear();
                        }
                    }
                    Instruction::Call { .. }
                    | Instruction::CallIndirect { .. }
                    | Instruction::TailCall { .. } => {
                        // Calls may read/write memory; conservatively invalidate
                        last_store.clear();
                        last_load.clear();
                    }
                    Instruction::AtomicLoad { .. }
                    | Instruction::AtomicStore { .. }
                    | Instruction::AtomicBinary { .. }
                    | Instruction::AtomicCompareExchange { .. }
                    | Instruction::VaStart { .. }
                    | Instruction::VaArg { .. }
                    | Instruction::VaEnd { .. }
                    | Instruction::VaCopy { .. }
                    | Instruction::Fence { .. } => {
                        last_store.clear();
                        last_load.clear();
                    }
                    _ => {}
                }

                if let Some(new_inst) = replacement {
                    *inst = new_inst;
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
        assert!(matches!(second, Instruction::Copy { .. }));
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
        assert!(matches!(&entry.instructions[1], Instruction::Copy { .. }));
    }

    #[test]
    fn aliasing_stores_and_mismatched_types_prevent_forwarding() {
        for (second_base, offset, load_ty, store_ty) in [
            (1, 0, i64(), i64()),
            (0, 4, i64(), MirType::i32()),
            (0, 0, MirType::i32(), i64()),
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
                func.blocks[0].instructions[2],
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
                ty: MirType::i32(),
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
            func.blocks[0].instructions[1],
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
            func.blocks[0].instructions[1],
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
        assert!(matches!(&entry.instructions[2], Instruction::Load { .. }));
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
            func.blocks[0].instructions[2],
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
            func.blocks[0].instructions[2],
            Instruction::Load { .. }
        ));
    }
}
