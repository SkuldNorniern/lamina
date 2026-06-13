//! Memory optimization transforms for MIR.

use crate::mir::instruction::Immediate;
use crate::mir::transform::{Transform, TransformCategory, TransformError, TransformLevel};
use crate::mir::{AddressMode, Function, Instruction, IntBinOp, Operand, Register};
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
            let mut last_store: HashMap<(Register, i16), Operand> = HashMap::new();
            let mut last_load: HashMap<(Register, i16), Register> = HashMap::new();

            for inst in &mut block.instructions {
                let mut replacement: Option<Instruction> = None;
                match &*inst {
                    Instruction::Store {
                        src, addr, attrs, ..
                    } => match addr {
                        AddressMode::BaseOffset { base, offset } if !attrs.volatile => {
                            last_store.insert((base.clone(), *offset), src.clone());
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
                        } else if let AddressMode::BaseOffset { base, offset } = addr {
                            let key = (base.clone(), *offset);
                            if let Some(stored_val) = last_store.get(&key) {
                                // store-to-load forwarding: replace load with a copy of stored_val
                                replacement = Some(Instruction::IntBinary {
                                    op: IntBinOp::Add,
                                    ty: *ty,
                                    dst: dst.clone(),
                                    lhs: stored_val.clone(),
                                    rhs: Operand::Immediate(Immediate::I64(0)),
                                });
                                // record latest load destination
                                last_load.insert(key, dst.clone());
                            } else if let Some(prev_reg) = last_load.get(&key) {
                                // Redundant load: reuse previous loaded register
                                replacement = Some(Instruction::IntBinary {
                                    op: IntBinOp::Add,
                                    ty: *ty,
                                    dst: dst.clone(),
                                    lhs: Operand::Register(prev_reg.clone()),
                                    rhs: Operand::Immediate(Immediate::I64(0)),
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
                    Instruction::Call { .. } | Instruction::TailCall { .. } => {
                        // Calls may read/write memory; conservatively invalidate
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
    use crate::mir::transform::test_utils::apply_pass;
    use crate::mir::{
        AddressMode, FunctionBuilder, Immediate, IntBinOp, MemoryAttrs, MirType, Operand,
        ScalarType, VirtualReg,
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn base_offset(base: Register, offset: i16) -> AddressMode {
        AddressMode::BaseOffset { base, offset }
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
        assert!(matches!(
            second,
            Instruction::IntBinary {
                op: IntBinOp::Add,
                ..
            }
        ));
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
            &entry.instructions[1],
            Instruction::IntBinary {
                op: IntBinOp::Add,
                ..
            }
        ));
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
}
