use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Operand, Register};
use std::collections::{HashMap, HashSet};

/// Memory Optimization Transform
/// Performs various memory-related optimizations including dead store elimination
#[derive(Default)]
pub struct MemoryOptimization;

impl Transform for MemoryOptimization {
    fn name(&self) -> &'static str {
        "memory_optimization"
    }

    fn description(&self) -> &'static str {
        "Optimizes memory operations including dead store elimination"
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

        // Apply dead store elimination
        if self.eliminate_dead_stores(func) {
            changed = true;
        }

        Ok(changed)
    }

    /// Dead Store Elimination
    /// Removes stores to variables that are never read before being overwritten or going out of scope
    fn eliminate_dead_stores(&self, func: &mut Function) -> bool {
        let mut changed = false;

        // For each block, perform dead store elimination
        for block in &mut func.blocks {
            if self.eliminate_dead_stores_in_block(block) {
                changed = true;
            }
        }

        changed
    }

    fn eliminate_dead_stores_in_block(&self, block: &mut Block) -> bool {
        let mut changed = false;
        let mut live_stores: HashMap<Register, usize> = HashMap::new(); // Register -> last store position
        let mut dead_stores = HashSet::new();

        // First pass: identify all loads (uses) of registers
        let mut used_registers = HashSet::new();
        for (idx, instr) in block.instructions.iter().enumerate() {
            match instr {
                Instruction::Load { dst, .. } => {
                    used_registers.insert(dst.clone());
                }
                Instruction::Call { ret: Some(dst), .. } => {
                    used_registers.insert(dst.clone());
                }
                Instruction::IntBinary { dst, lhs, rhs, .. }
                | Instruction::FloatBinary { dst, lhs, rhs, .. } => {
                    // Check if operands are registers and mark them as used
                    if let Operand::Register(reg) = lhs {
                        used_registers.insert(reg.clone());
                    }
                    if let Operand::Register(reg) = rhs {
                        used_registers.insert(reg.clone());
                    }
                }
                Instruction::IntCmp { dst, lhs, rhs, .. }
                | Instruction::FloatCmp { dst, lhs, rhs, .. } => {
                    if let Operand::Register(reg) = lhs {
                        used_registers.insert(reg.clone());
                    }
                    if let Operand::Register(reg) = rhs {
                        used_registers.insert(reg.clone());
                    }
                }
                Instruction::Br { cond, .. } => {
                    used_registers.insert(cond.clone());
                }
                Instruction::Ret { value: Some(cond) } => {
                    if let Operand::Register(reg) = cond {
                        used_registers.insert(reg.clone());
                    }
                }
                Instruction::Select { cond, true_val, false_val, .. } => {
                    used_registers.insert(cond.clone());
                    if let Operand::Register(reg) = true_val {
                        used_registers.insert(reg.clone());
                    }
                    if let Operand::Register(reg) = false_val {
                        used_registers.insert(reg.clone());
                    }
                }
                _ => {}
            }
        }

        // Second pass: identify dead stores
        // A store is dead if:
        // 1. It's a store to a register
        // 2. The register is never used after this store
        // 3. The register is not used in any control flow (branches, returns)

        for (idx, instr) in block.instructions.iter().enumerate() {
            match instr {
                Instruction::IntBinary { dst, .. }
                | Instruction::FloatBinary { dst, .. }
                | Instruction::FloatUnary { dst, .. }
                | Instruction::IntCmp { dst, .. }
                | Instruction::FloatCmp { dst, .. }
                | Instruction::Load { dst, .. }
                | Instruction::Lea { dst, .. }
                | Instruction::Select { dst, .. } => {
                    // This defines a register, so any previous store to it is dead
                    // unless this register was used somewhere
                    if let Some(prev_store_idx) = live_stores.get(dst) {
                        if !used_registers.contains(dst) {
                            // This register is never used, so the store is dead
                            dead_stores.insert(*prev_store_idx);
                        }
                    }
                    // Remove from live stores since it's been redefined
                    live_stores.remove(dst);
                }
                Instruction::Call { ret: Some(dst), .. } => {
                    // Function call result - similar to above
                    if let Some(prev_store_idx) = live_stores.get(dst) {
                        if !used_registers.contains(dst) {
                            dead_stores.insert(*prev_store_idx);
                        }
                    }
                    live_stores.remove(dst);
                }
                Instruction::Store { .. } => {
                    // For now, we don't eliminate memory stores as they're harder to track
                    // and may have side effects. Focus on register dead stores.
                }
                _ => {}
            }
        }

        // Remove dead stores (in reverse order to maintain indices)
        let mut indices_to_remove: Vec<usize> = dead_stores.into_iter().collect();
        indices_to_remove.sort_by(|a, b| b.cmp(a)); // Sort in reverse order

        for idx in indices_to_remove {
            if idx < block.instructions.len() {
                block.instructions.remove(idx);
                changed = true;
            }
        }

        changed
    }
}
