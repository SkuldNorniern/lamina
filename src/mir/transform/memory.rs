//! Memory optimization transforms for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Function, Instruction, Operand, Register};
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

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl MemoryOptimization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        if self.forward_redundant_loads(func) {
            changed = true;
        }

        Ok(changed)
    }

    /// Redundant load elimination and store-to-load forwarding within basic blocks.
    fn forward_redundant_loads(&self, func: &mut Function) -> bool {
        use crate::mir::AddressMode;
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
                        AddressMode::BaseIndexScale { .. } => {
                            last_store.clear();
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
                                    op: crate::mir::IntBinOp::Add,
                                    ty: *ty,
                                    dst: dst.clone(),
                                    lhs: stored_val.clone(),
                                    rhs: Operand::Immediate(
                                        crate::mir::instruction::Immediate::I64(0),
                                    ),
                                });
                                // record latest load destination
                                last_load.insert(key, dst.clone());
                            } else if let Some(prev_reg) = last_load.get(&key) {
                                // Redundant load: reuse previous loaded register
                                replacement = Some(Instruction::IntBinary {
                                    op: crate::mir::IntBinOp::Add,
                                    ty: *ty,
                                    dst: dst.clone(),
                                    lhs: Operand::Register(prev_reg.clone()),
                                    rhs: Operand::Immediate(
                                        crate::mir::instruction::Immediate::I64(0),
                                    ),
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
