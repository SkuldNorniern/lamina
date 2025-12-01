//! Address mode canonicalization transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{AddressMode, Function, Instruction, Operand, Register};

/// Canonicalizes address formation patterns into BaseIndexScale addressing.
#[derive(Default)]
pub struct AddressingCanonicalization;

impl Transform for AddressingCanonicalization {
    fn name(&self) -> &'static str {
        "addressing_canonicalization"
    }

    fn description(&self) -> &'static str {
        "Canonicalizes address formation into BaseIndexScale addressing where possible"
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

impl AddressingCanonicalization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit function size
        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 1_000;

        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for addressing canonicalization ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        for block in &func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                return Err(format!(
                    "Block '{}' too large for addressing canonicalization ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_INSTRUCTIONS_PER_BLOCK
                ));
            }
        }

        let mut changed = false;

        for block in &mut func.blocks {
            // Build a simple def map for this block: reg -> (idx of instr)
            let mut def_index: std::collections::HashMap<Register, usize> =
                std::collections::HashMap::new();
            for (i, instr) in block.instructions.iter().enumerate() {
                if let Some(reg) = instr.def_reg() {
                    def_index.insert(reg.clone(), i);
                }
            }

            let len = block.instructions.len();
            // Safety: limit number of rewrites per block to prevent excessive changes
            const MAX_REWRITES_PER_BLOCK: usize = 50;
            let mut rewrites_this_block = 0;

            for i in 0..len {
                if rewrites_this_block >= MAX_REWRITES_PER_BLOCK {
                    break; // Stop if we've made too many changes
                }

                // Phase 1: analyze immutably to compute potential new addressing mode
                let new_addr_mode: Option<AddressMode> = match &block.instructions[i] {
                    Instruction::Load { addr, .. } => {
                        let mut addr_clone = addr.clone();
                        if self.try_rewrite_addr(&mut addr_clone, &def_index, &block.instructions) {
                            Some(addr_clone)
                        } else {
                            None
                        }
                    }
                    Instruction::Store { addr, .. } => {
                        let mut addr_clone = addr.clone();
                        if self.try_rewrite_addr(&mut addr_clone, &def_index, &block.instructions) {
                            Some(addr_clone)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                // Phase 2: apply mutation
                if let Some(new_mode) = new_addr_mode {
                    match &mut block.instructions[i] {
                        Instruction::Load { addr, .. } => {
                            *addr = new_mode;
                            changed = true;
                            rewrites_this_block += 1;
                        }
                        Instruction::Store { addr, .. } => {
                            *addr = new_mode;
                            changed = true;
                            rewrites_this_block += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(changed)
    }

    fn try_rewrite_addr(
        &self,
        addr: &mut AddressMode,
        def_index: &std::collections::HashMap<Register, usize>,
        instructions: &[Instruction],
    ) -> bool {
        match addr {
            AddressMode::BaseOffset { base, offset } => {
                // Safety: only rewrite if we can find the definition
                // and it's a simple pattern we can verify
                if let Some(def_pos) = def_index.get(base) {
                    // Safety: ensure def_pos is valid
                    if *def_pos >= instructions.len() {
                        return false;
                    }

                    if let Some((base_reg, index_reg, scale)) = self.match_add_scaled_index(
                        &instructions[*def_pos],
                        def_index,
                        instructions,
                    ) {
                        // Keep original offset, scale is guaranteed in {1,2,4,8}
                        // Only convert if offset fits in i8 range to avoid information loss
                        if (*offset >= i8::MIN as i16) && (*offset <= i8::MAX as i16) {
                            // Safety: verify base_reg and index_reg are actually defined
                            // and not the same register (which could cause issues)
                            if base_reg == index_reg {
                                return false; // Don't rewrite if base == index
                            }

                            let clamped_off = *offset as i8;
                            *addr = AddressMode::BaseIndexScale {
                                base: base_reg,
                                index: index_reg,
                                scale,
                                offset: clamped_off,
                            };
                            return true;
                        }
                    }
                }
            }
            AddressMode::BaseIndexScale { .. } => {
                // Already canonicalized, don't change
            }
        }
        false
    }

    fn match_add_scaled_index(
        &self,
        def_instr: &Instruction,
        def_index: &std::collections::HashMap<Register, usize>,
        instructions: &[Instruction],
    ) -> Option<(Register, Register, u8)> {
        use crate::mir::IntBinOp;
        use crate::mir::instruction::Immediate;

        // Check if a register is defined by shift-left with small power-of-two scale
        fn scaled_from_shift(
            r: &Register,
            def_index: &std::collections::HashMap<Register, usize>,
            instructions: &[Instruction],
        ) -> Option<(Register, u8)> {
            if let Some(&pos) = def_index.get(r)
                && let Instruction::IntBinary {
                    op: IntBinOp::Shl,
                    lhs,
                    rhs,
                    ..
                } = &instructions[pos]
                && let Operand::Register(idx) = lhs
                && let Operand::Immediate(Immediate::I64(shift)) = rhs
                && (0..=3).contains(&{ *shift })
            {
                return Some((idx.clone(), 1u8 << (*shift as u8)));
            }
            None
        }

        // Check if a register is defined by mul-by-const in {1,2,4,8}
        fn scaled_from_mul(
            r: &Register,
            def_index: &std::collections::HashMap<Register, usize>,
            instructions: &[Instruction],
        ) -> Option<(Register, u8)> {
            if let Some(&pos) = def_index.get(r)
                && let Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    lhs,
                    rhs,
                    ..
                } = &instructions[pos]
                && let Operand::Register(idx) = lhs
                && let Operand::Immediate(Immediate::I64(scale)) = rhs
            {
                match *scale {
                    1 | 2 | 4 | 8 => return Some((idx.clone(), *scale as u8)),
                    _ => {}
                }
            }
            None
        }

        // Helper to detect base + scaled(idx)
        fn try_base_plus_scaled(
            base_op: &Operand,
            other_op: &Operand,
            def_index: &std::collections::HashMap<Register, usize>,
            instructions: &[Instruction],
        ) -> Option<(Register, Register, u8)> {
            if let Operand::Register(base_reg) = base_op
                && let Operand::Register(r2) = other_op
                && let Some((idx, scale)) = scaled_from_shift(r2, def_index, instructions)
                    .or_else(|| scaled_from_mul(r2, def_index, instructions))
            {
                return Some((base_reg.clone(), idx, scale));
            }
            None
        }

        if let Instruction::IntBinary {
            op: IntBinOp::Add,
            lhs,
            rhs,
            ..
        } = def_instr
        {
            if let Some(m) = try_base_plus_scaled(lhs, rhs, def_index, instructions) {
                return Some(m);
            }
            if let Some(m) = try_base_plus_scaled(rhs, lhs, def_index, instructions) {
                return Some(m);
            }
        }

        None
    }
}
