//! Address mode canonicalization transform for MIR.
//!
//! This transform converts `BaseOffset` addressing patterns like:
//!   `base = arr + (idx * scale)` followed by `load [base + offset]`
//! Into the `BaseIndexScale` addressing mode:
//!   `load [arr + idx * scale + offset]`
//!
//! This helps on x86_64 which has native support for complex addressing modes.

use crate::mir::{
    AddressMode, Function, Instruction, IntBinOp, Operand, Register,
    instruction::Immediate,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use lamina_mir::Located;

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

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl AddressingCanonicalization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        for block in &mut func.blocks {
            for i in 0..block.instructions.len() {
                let new_addr_mode = match &block.instructions[i].instr {
                    Instruction::Load { addr, .. } | Instruction::Store { addr, .. } => {
                        canonical_address(addr, i, &block.instructions)
                    }
                    _ => None,
                };

                if let Some(new_mode) = new_addr_mode {
                    match &mut block.instructions[i].instr {
                        Instruction::Load { addr, .. } | Instruction::Store { addr, .. } => {
                            *addr = new_mode;
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(changed)
    }
}

/// The `BaseIndexScale` form of the address used by the access at `at`, when the
/// registers it names hold the same values there as they did when the address was formed.
fn canonical_address(
    addr: &AddressMode,
    at: usize,
    instructions: &[Located],
) -> Option<AddressMode> {
    let AddressMode::BaseOffset {
        base,
        offset,
        address_space,
    } = addr
    else {
        return None;
    };
    let offset = i8::try_from(*offset).ok()?;
    let formed = last_definition(instructions, base, at)?;
    let Instruction::IntBinary {
        op: IntBinOp::Add,
        lhs,
        rhs,
        ..
    } = &instructions[formed].instr
    else {
        return None;
    };

    [(lhs, rhs), (rhs, lhs)]
        .into_iter()
        .find_map(|(base_op, scaled_op)| {
            let (Operand::Register(base_reg), Operand::Register(scaled_reg)) = (base_op, scaled_op)
            else {
                return None;
            };
            let (scaled_at, index, scale) = scaled_index(instructions, scaled_reg, formed)?;
            if base_reg == &index
                || redefined_between(instructions, base_reg, formed, at)
                || redefined_between(instructions, &index, scaled_at, at)
            {
                return None;
            }
            Some(AddressMode::BaseIndexScale {
                base: base_reg.clone(),
                index,
                scale,
                offset,
                address_space: *address_space,
            })
        })
}

/// The register's value is `1 << shift` or `multiplier` times `index`, as set by the
/// definition at the returned position, which is the last one before `before`.
fn scaled_index(
    instructions: &[Located],
    reg: &Register,
    before: usize,
) -> Option<(usize, Register, u8)> {
    let at = last_definition(instructions, reg, before)?;
    let (index, scale) = match &instructions[at].instr {
        Instruction::IntBinary {
            op: IntBinOp::Shl,
            lhs: Operand::Register(index),
            rhs: Operand::Immediate(Immediate::I64(shift)),
            ..
        } if (0..=3).contains(shift) => (index, 1u8 << *shift),
        Instruction::IntBinary {
            op: IntBinOp::Mul,
            lhs: Operand::Register(index),
            rhs: Operand::Immediate(Immediate::I64(scale @ (1 | 2 | 4 | 8))),
            ..
        } => (index, *scale as u8),
        _ => return None,
    };
    Some((at, index.clone(), scale))
}

fn last_definition(instructions: &[Located], reg: &Register, before: usize) -> Option<usize> {
    (0..before)
        .rev()
        .find(|&i| instructions[i].def_reg() == Some(reg))
}

fn redefined_between(instructions: &[Located], reg: &Register, from: usize, to: usize) -> bool {
    instructions[from..to]
        .iter()
        .any(|instruction| instruction.def_reg() == Some(reg))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, MemoryAttrs, MirType, ScalarType, VirtualReg,
        transform::test_utils::get_block, verify_function_in_pipeline,
    };

    #[test]
    fn test_addressing_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = AddressingCanonicalization;
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

        let pass = AddressingCanonicalization;
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
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let pass = AddressingCanonicalization;
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
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization;
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Verify the addressing mode was changed to BaseIndexScale
        let entry = get_block(&func, "entry");
        match &entry.instructions[2].instr {
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
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization;
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        let entry = get_block(&func, "entry");
        match &entry.instructions[2].instr {
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
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization;
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
                    offset: 256,
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let pass = AddressingCanonicalization;
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
                    address_space: Default::default(),
                },
                attrs: MemoryAttrs::default(),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        let pass = AddressingCanonicalization;
        let changed = pass.apply(&mut func).expect("should succeed");
        assert!(changed);

        let entry = get_block(&func, "entry");
        match &entry.instructions[2].instr {
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
                Located::from(Instruction::Load {
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i + 10).into(),
                    addr: AddressMode::BaseOffset {
                        base: VirtualReg::gpr(0).into(),
                        offset: (i as i16) * 8,
                        address_space: Default::default(),
                    },
                    attrs: MemoryAttrs::default(),
                }),
            );
        }
        func.blocks[0]
            .instructions
            .push(Located::from(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            }));

        let pass = AddressingCanonicalization;
        let result = pass.apply(&mut func);
        assert!(result.is_ok());
    }

    fn base_index_candidate(
        instructions_before: Vec<Instruction>,
        instructions_after: Vec<Instruction>,
    ) -> Function {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let reg = |id| Register::from(VirtualReg::gpr(id));
        let mut func = FunctionBuilder::new("candidate")
            .param(reg(0), i64_ty)
            .param(reg(1), i64_ty)
            .param(reg(3), i64_ty)
            .returns(i64_ty)
            .block("entry")
            .build();
        func.blocks[0]
            .instructions
            .extend(instructions_before.into_iter().map(Located::from));
        func.blocks[0]
            .instructions
            .extend(instructions_after.into_iter().map(Located::from));
        func
    }

    fn address_of(base: u32, offset: i16) -> AddressMode {
        AddressMode::BaseOffset {
            base: VirtualReg::gpr(base).into(),
            offset,
            address_space: Default::default(),
        }
    }

    #[test]
    fn access_is_not_rewritten_when_its_base_is_redefined_after_the_address_is_formed() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let reg = |id| Register::from(VirtualReg::gpr(id));
        let mut func = base_index_candidate(
            vec![
                Instruction::IntBinary {
                    op: IntBinOp::Shl,
                    ty: i64_ty,
                    dst: reg(2),
                    lhs: Operand::Register(reg(1)),
                    rhs: Operand::Immediate(Immediate::I64(3)),
                },
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: i64_ty,
                    dst: reg(4),
                    lhs: Operand::Register(reg(0)),
                    rhs: Operand::Register(reg(2)),
                },
                Instruction::Load {
                    ty: i64_ty,
                    dst: reg(5),
                    addr: address_of(4, 0),
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Copy {
                    ty: i64_ty,
                    dst: reg(0),
                    src: Operand::Register(reg(3)),
                },
            ],
            vec![
                Instruction::Load {
                    ty: i64_ty,
                    dst: reg(6),
                    addr: address_of(4, 0),
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Ret {
                    value: Some(Operand::Register(reg(6))),
                },
            ],
        );
        verify_function_in_pipeline(&func).expect("input is valid MIR");
        AddressingCanonicalization
            .apply(&mut func)
            .expect("should succeed");
        let entry = get_block(&func, "entry");
        assert!(
            matches!(
                &entry.instructions[4].instr,
                Instruction::Load {
                    addr: AddressMode::BaseOffset { .. },
                    ..
                }
            ),
            "{:?}",
            entry.instructions[4]
        );
    }

    #[test]
    fn access_is_not_rewritten_from_a_definition_that_follows_it() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let reg = |id| Register::from(VirtualReg::gpr(id));
        let mut func = base_index_candidate(
            vec![Instruction::Load {
                ty: i64_ty,
                dst: reg(5),
                addr: address_of(3, 0),
                attrs: MemoryAttrs::default(),
            }],
            vec![
                Instruction::IntBinary {
                    op: IntBinOp::Shl,
                    ty: i64_ty,
                    dst: reg(2),
                    lhs: Operand::Register(reg(1)),
                    rhs: Operand::Immediate(Immediate::I64(3)),
                },
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: i64_ty,
                    dst: reg(3),
                    lhs: Operand::Register(reg(0)),
                    rhs: Operand::Register(reg(2)),
                },
                Instruction::Ret {
                    value: Some(Operand::Register(reg(5))),
                },
            ],
        );
        verify_function_in_pipeline(&func).expect("input is valid MIR");
        AddressingCanonicalization
            .apply(&mut func)
            .expect("should succeed");
        let entry = get_block(&func, "entry");
        assert!(
            matches!(
                &entry.instructions[0].instr,
                Instruction::Load {
                    addr: AddressMode::BaseOffset { .. },
                    ..
                }
            ),
            "{:?}",
            entry.instructions[0]
        );
    }
}
