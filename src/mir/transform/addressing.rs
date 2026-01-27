//! Address mode canonicalization transform for MIR.
//!
//! This transform converts `BaseOffset` addressing patterns like:
//!   `base = arr + (idx * scale)` followed by `load [base + offset]`
//! Into the `BaseIndexScale` addressing mode:
//!   `load [arr + idx * scale + offset]`
//!
//! This helps on x86_64 which has native support for complex addressing modes.

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        Block, FunctionBuilder, Immediate, IntBinOp, MemoryAttrs, MirType, Operand, ScalarType,
        VirtualReg,
    };

    #[test]
    fn test_addressing_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = AddressingCanonicalization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No changes expected
    }

    #[test]
    fn test_addressing_no_memory_ops() {
        // Function with no loads/stores - nothing to canonicalize
        let mut func = FunctionBuilder::new("no_mem")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_addressing_base_offset_unchanged() {
        // Simple BaseOffset that doesn't match scaled pattern - should not change
        let mut func = FunctionBuilder::new("simple_load")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(0).into(),
                    offset: 8,
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_addressing_canonicalize_shift_pattern() {
        // Pattern: ptr = base + (idx << 3), load [ptr + 0]
        // Should become: load [base + idx * 8 + 0]
        let mut func = FunctionBuilder::new("shift_pattern")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64)) // base
            .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64)) // idx
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // v2 = idx << 3 (multiply by 8)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Shl,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(3)),
            })
            // v3 = base + v2
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            // v4 = load [v3 + 0]
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(3).into(),
                    offset: 0,
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Verify the addressing mode was changed to BaseIndexScale
        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[2] {
            Instruction::Load { addr, .. } => {
                assert!(
                    matches!(addr, AddressMode::BaseIndexScale { scale: 8, .. }),
                    "Expected BaseIndexScale with scale=8, got {:?}",
                    addr
                );
            }
            _ => panic!("Expected Load instruction"),
        }
    }

    #[test]
    fn test_addressing_canonicalize_mul_pattern() {
        // Pattern: ptr = base + (idx * 4), load [ptr + 0]
        // Should become: load [base + idx * 4 + 0]
        let mut func = FunctionBuilder::new("mul_pattern")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64)) // base
            .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64)) // idx
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // v2 = idx * 4
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(4)),
            })
            // v3 = base + v2
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            // v4 = load [v3 + 0]
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(3).into(),
                    offset: 0,
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[2] {
            Instruction::Load { addr, .. } => {
                assert!(
                    matches!(addr, AddressMode::BaseIndexScale { scale: 4, .. }),
                    "Expected BaseIndexScale with scale=4, got {:?}",
                    addr
                );
            }
            _ => panic!("Expected Load instruction"),
        }
    }

    #[test]
    fn test_addressing_invalid_scale_not_changed() {
        // Pattern with scale=3 (not power of 2) - should NOT be canonicalized
        let mut func = FunctionBuilder::new("invalid_scale")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(3)), // Invalid scale
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(3).into(),
                    offset: 0,
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed); // Scale=3 is not valid for BaseIndexScale
    }

    #[test]
    fn test_addressing_large_offset_not_changed() {
        // Offset > i8::MAX should not be canonicalized (to avoid info loss)
        let mut func = FunctionBuilder::new("large_offset")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Shl,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(3)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::Load {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(3).into(),
                    offset: 256, // > i8::MAX
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(!changed); // Offset too large
    }

    #[test]
    fn test_addressing_store_also_canonicalized() {
        // Store instructions should also be canonicalized
        let mut func = FunctionBuilder::new("store_pattern")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Shl,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(2)), // *4
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::Store {
                ty: MirType::Scalar(ScalarType::I64),
                src: Operand::Immediate(Immediate::I64(42)),
                addr: AddressMode::BaseOffset {
                    base: VirtualReg::gpr(3).into(),
                    offset: 0,
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = AddressingCanonicalization::default();
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[2] {
            Instruction::Store { addr, .. } => {
                assert!(
                    matches!(addr, AddressMode::BaseIndexScale { scale: 4, .. }),
                    "Expected BaseIndexScale with scale=4, got {:?}",
                    addr
                );
            }
            _ => panic!("Expected Store instruction"),
        }
    }

    #[test]
    fn test_addressing_stress_no_infinite_loop() {
        // Many loads to ensure no infinite loop or excessive rewrites
        let mut func = FunctionBuilder::new("stress")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        // Add 100 simple loads
        for i in 0..100u32 {
            func.blocks[0].instructions.insert(
                0,
                Instruction::Load {
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i + 10).into(),
                    addr: AddressMode::BaseOffset {
                        base: VirtualReg::gpr(0).into(),
                        offset: (i as i16) * 8,
                    },
                    attrs: MemoryAttrs::default(),
                },
            );
        }
        func.blocks[0].instructions.push(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        });

        let pass = AddressingCanonicalization::default();
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
    }
}
