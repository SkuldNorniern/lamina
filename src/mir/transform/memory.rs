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

        // Apply dead store elimination - DISABLED (too aggressive)
        // if self.eliminate_dead_stores(func) {
        //     changed = true;
        // }

        // Apply simple redundant load/store forwarding
        if self.forward_redundant_loads(func) {
            changed = true;
        }

        // Apply memory access pattern optimizations for matrix operations
        if self.optimize_memory_access_patterns(func) {
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
        // DISABLED: Dead store elimination can be unsafe in the presence of loops
        // and complex control flow. The analysis is too conservative and may remove
        // stores that are needed across loop iterations or complex control flow.
        false
    }

    /// Redundant Load Elimination / Store-to-Load Forwarding (intra-block, conservative)
    fn forward_redundant_loads(&self, func: &mut Function) -> bool {
        use crate::mir::AddressMode;
        let mut changed = false;

        for block in &mut func.blocks {
            // Maps (base, offset) -> last stored Operand in this block
            let mut last_store: HashMap<(Register, i16), Operand> = HashMap::new();
            // Maps (base, offset) -> register holding last loaded value in this block
            let mut last_load: HashMap<(Register, i16), Register> = HashMap::new();

            for inst in &mut block.instructions {
                // Compute replacement instruction, if any, without holding a mutable borrow
                let mut replacement: Option<Instruction> = None;
                match &*inst {
                    Instruction::Store { src, addr, attrs, .. } => {
                        // Invalidate on volatile or unknown addressing
                        match addr {
                            AddressMode::BaseOffset { base, offset } if !attrs.volatile => {
                                // Record last store and invalidate prior load
                                last_store.insert((base.clone(), *offset), src.clone());
                                last_load.remove(&(base.clone(), *offset));
                            }
                            _ => {
                                // Be conservative on unknown addressing
                                last_store.clear();
                                last_load.clear();
                            }
                        }
                    }
                    Instruction::Load { ty, dst, addr, attrs } => {
                        // Skip volatile loads
                        if attrs.volatile {
                            last_store.clear();
                            last_load.clear();
                        } else {
                            if let AddressMode::BaseOffset { base, offset } = addr {
                                let key = (base.clone(), *offset);
                                if let Some(stored_val) = last_store.get(&key) {
                                    // store-to-load forwarding: replace load with a copy of stored_val
                                    replacement = Some(Instruction::IntBinary {
                                        op: crate::mir::IntBinOp::Add,
                                        ty: *ty,
                                        dst: dst.clone(),
                                        lhs: stored_val.clone(),
                                        rhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(0)),
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
                                        rhs: Operand::Immediate(crate::mir::instruction::Immediate::I64(0)),
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

    /// Optimize memory access patterns for matrix operations
    /// This is a placeholder for future enhancements like:
    /// - Reordering loads/stores for better cache locality
    /// - Adding prefetch hints
    /// - Optimizing strided access patterns
    fn optimize_memory_access_patterns(&self, _func: &mut Function) -> bool {
        // TODO: Implement memory access pattern optimizations
        // For now, this is a no-op but serves as a framework for future work
        false
    }
}
