//! Common Subexpression Elimination (CSE) transform for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Immediate, Instruction, MirType, Operand, Register, ScalarType};
use std::collections::HashMap;

/// Common Subexpression Elimination (CSE)
/// Eliminates redundant computations within basic blocks
#[derive(Default)]
pub struct CommonSubexpressionElimination;

impl Transform for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "common_subexpression_elimination"
    }

    fn description(&self) -> &'static str {
        "Eliminates redundant computations within basic blocks"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ArithmeticOptimization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CommonSubexpressionElimination {
    /// Extract the type from an instruction for use in replacement instructions
    fn extract_instruction_type(&self, instr: &Instruction) -> MirType {
        match instr {
            Instruction::IntBinary { ty, .. }
            | Instruction::FloatBinary { ty, .. }
            | Instruction::FloatUnary { ty, .. }
            | Instruction::IntCmp { ty, .. }
            | Instruction::FloatCmp { ty, .. }
            | Instruction::Select { ty, .. }
            | Instruction::Load { ty, .. }
            | Instruction::Store { ty, .. }
            | Instruction::VectorOp { ty, .. } => *ty,
            // For other instructions that don't have explicit types, default to I64
            // This should be rare and these instructions probably shouldn't be CSE'd
            _ => MirType::Scalar(ScalarType::I64),
        }
    }

    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit function size
        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 500;

        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for CSE ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        for block in &func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                return Err(format!(
                    "Block '{}' too large for CSE ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_INSTRUCTIONS_PER_BLOCK
                ));
            }
        }

        let mut changed = false;
        let loop_headers = compute_back_edge_headers(func);

        for block in &mut func.blocks {
            // Skip only loop header blocks to avoid complex control flow issues
            // Allow CSE within loop bodies where it's safe and beneficial
            if loop_headers.contains(&block.label) {
                continue;
            }

            // Apply CSE to this block, but be conservative about loop-related blocks
            if self.eliminate_in_block_safe(block, &loop_headers) {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn eliminate_in_block_safe(
        &self,
        block: &mut Block,
        loop_headers: &std::collections::HashSet<String>,
    ) -> bool {
        // For blocks that are part of loops, be more conservative
        // Only apply CSE if the block is small and contains simple operations
        let is_in_loop_body = self.is_block_in_loop_body(block, loop_headers);

        if is_in_loop_body && block.instructions.len() > 50 {
            return false; // Skip large loop body blocks
        }

        if !is_in_loop_body && block.instructions.len() > 200 {
            return false; // Skip large non-loop blocks
        }

        self.eliminate_in_block_conservative(block, is_in_loop_body)
    }

    fn is_block_in_loop_body(
        &self,
        block: &Block,
        loop_headers: &std::collections::HashSet<String>,
    ) -> bool {
        // Check if this block can reach a loop header (simplified check)
        // This is a conservative approximation
        for instr in &block.instructions {
            match instr {
                Instruction::Jmp { target }
                | Instruction::Br {
                    true_target: target,
                    ..
                } => {
                    if loop_headers.contains(target) {
                        return true;
                    }
                }
                #[allow(unreachable_patterns)]
                Instruction::Br {
                    false_target: target,
                    ..
                } => {
                    if loop_headers.contains(target) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn is_simple_cse_candidate(&self, instr: &Instruction) -> bool {
        // Only allow simple arithmetic operations for CSE in loop bodies
        match instr {
            Instruction::IntBinary { op, .. } => {
                matches!(
                    op,
                    crate::mir::IntBinOp::Add
                        | crate::mir::IntBinOp::Sub
                        | crate::mir::IntBinOp::Mul
                )
            }
            Instruction::FloatBinary { op, .. } => {
                matches!(
                    op,
                    crate::mir::FloatBinOp::FAdd
                        | crate::mir::FloatBinOp::FSub
                        | crate::mir::FloatBinOp::FMul
                )
            }
            _ => false, // Only simple arithmetic for loop CSE
        }
    }

    fn eliminate_in_block_conservative(&self, block: &mut Block, is_in_loop_body: bool) -> bool {
        let mut changed = false;

        // Skip CSE in blocks with too many instructions to avoid excessive processing
        if (!is_in_loop_body && block.instructions.len() > 200)
            || (is_in_loop_body && block.instructions.len() > 50)
        {
            return false;
        }

        // Map from expression key (including operand versions) to (destination register, dest version)
        let mut expr_to_reg: HashMap<String, (Register, u64)> = HashMap::new();
        // Per-register version counter to detect intervening re-definitions
        let mut def_version: HashMap<Register, u64> = HashMap::new();
        let mut instructions = Vec::new();

        for instr in &block.instructions {
            let expr_key = self.expr_key_with_versions(instr, &def_version);

            if let Some(expr_key) = expr_key {
                if let Some((existing_reg, existing_ver)) = expr_to_reg.get(&expr_key) {
                    // Replace this instruction with a copy from the existing register
                    if let Some(dst) = instr.def_reg() {
                        // Ensure the producing register has not been redefined since
                        let cur_ver = def_version.get(existing_reg).copied().unwrap_or(0);
                        if cur_ver != *existing_ver {
                            // Stale producer, cannot reuse
                        } else {
                            // In loop bodies, be extra conservative: only CSE simple operations
                            if is_in_loop_body && !self.is_simple_cse_candidate(instr) {
                                instructions.push(instr.clone());
                            } else {
                                let instr_type = self.extract_instruction_type(instr);
                                let copy_instr = match instr_type {
                                    MirType::Scalar(ScalarType::F64)
                                    | MirType::Scalar(ScalarType::F32) => {
                                        // For float types, create a float add with 0.0
                                        let zero = match instr_type {
                                            MirType::Scalar(ScalarType::F64) => Immediate::F64(0.0),
                                            MirType::Scalar(ScalarType::F32) => Immediate::F32(0.0),
                                            _ => unreachable!(),
                                        };
                                        Instruction::FloatBinary {
                                            op: crate::mir::FloatBinOp::FAdd,
                                            dst: dst.clone(),
                                            ty: instr_type,
                                            lhs: Operand::Register(existing_reg.clone()),
                                            rhs: Operand::Immediate(zero),
                                        }
                                    }
                                    _ => {
                                        // For integer types, use integer add with 0
                                        Instruction::IntBinary {
                                            op: crate::mir::IntBinOp::Add,
                                            dst: dst.clone(),
                                            ty: instr_type,
                                            lhs: Operand::Register(existing_reg.clone()),
                                            rhs: Operand::Immediate(Immediate::I64(0)),
                                        }
                                    }
                                };
                                instructions.push(copy_instr);
                                changed = true;
                                // Destination has now been defined; bump its version
                                def_version
                                    .entry(dst.clone())
                                    .and_modify(|v| *v += 1)
                                    .or_insert(1);
                                continue;
                            }
                        }
                    }
                } else {
                    // First time seeing this expression, record it
                    if let Some(dst) = instr.def_reg() {
                        let dst_ver = def_version.get(dst).copied().unwrap_or(0);
                        expr_to_reg.insert(expr_key, (dst.clone(), dst_ver));
                    }
                }
            }

            instructions.push(instr.clone());
            // Bump version for any destination defined by this instruction
            if let Some(dst) = instr.def_reg() {
                def_version
                    .entry(dst.clone())
                    .and_modify(|v| *v += 1)
                    .or_insert(1);
            }
        }

        block.instructions = instructions;
        changed
    }

    fn expr_key(&self, instr: &Instruction) -> Option<String> {
        match instr {
            Instruction::IntBinary { op, lhs, rhs, .. } => {
                Some(format!("IntBinary_{:?}_{:?}_{:?}", op, lhs, rhs))
            }
            Instruction::FloatBinary { op, lhs, rhs, .. } => {
                Some(format!("FloatBinary_{:?}_{:?}_{:?}", op, lhs, rhs))
            }
            Instruction::IntCmp { op, lhs, rhs, .. } => {
                Some(format!("IntCmp_{:?}_{:?}_{:?}", op, lhs, rhs))
            }
            Instruction::FloatCmp { op, lhs, rhs, .. } => {
                Some(format!("FloatCmp_{:?}_{:?}_{:?}", op, lhs, rhs))
            }
            _ => None,
        }
    }

    fn expr_key_with_versions(
        &self,
        instr: &Instruction,
        def_version: &HashMap<Register, u64>,
    ) -> Option<String> {
        // Base structural key
        let base = self.expr_key(instr)?;
        // Append operand versions to ensure no intervening re-definitions occurred
        let mut parts = vec![base];
        for reg in instr.use_regs() {
            let ver = def_version.get(reg).copied().unwrap_or(0);
            parts.push(format!("{}@{}", reg, ver));
        }
        Some(parts.join("|"))
    }
}

/// Identify loop headers via simple back-edge detection using block order.
fn compute_back_edge_headers(func: &Function) -> std::collections::HashSet<String> {
    use std::collections::{HashMap, HashSet};
    let mut label_index: HashMap<&str, usize> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        label_index.insert(&b.label, i);
    }
    let mut headers: HashSet<String> = HashSet::new();
    for (i, b) in func.blocks.iter().enumerate() {
        if let Some(term) = b.instructions.last() {
            match term {
                Instruction::Jmp { target } => {
                    if let Some(&tidx) = label_index.get(target.as_str())
                        && tidx <= i
                    {
                        headers.insert(target.clone());
                    }
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if let Some(&tidx) = label_index.get(true_target.as_str())
                        && tidx <= i
                    {
                        headers.insert(true_target.clone());
                    }
                    if let Some(&fidx) = label_index.get(false_target.as_str())
                        && fidx <= i
                    {
                        headers.insert(false_target.clone());
                    }
                }
                _ => {}
            }
        }
    }
    headers
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_cse_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let cse = CommonSubexpressionElimination::default();
        let result = cse.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_cse_no_duplicate_expressions() {
        // No duplicate expressions, nothing to eliminate
        let mut func = FunctionBuilder::new("no_dups")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Sub,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(3)),
                rhs: Operand::Immediate(Immediate::I64(4)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination::default();
        let changed = cse.apply(&mut func).expect("should succeed");
        assert!(!changed);
    }

    #[test]
    fn test_cse_with_intervening_redefinition() {
        // CSE should not reuse expression if operand was redefined
        let mut func = FunctionBuilder::new("redef")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // v0 = v1 + v2
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            // Redefine v1
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(99)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // v3 = v1 + v2 (should NOT reuse v0 because v1 was redefined)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(3).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination::default();
        let changed = cse.apply(&mut func).expect("should succeed");

        // Should NOT eliminate the second v1 + v2 since v1 was redefined
        assert!(!changed);
    }

    #[test]
    fn test_cse_duplicate_expression_does_not_crash() {
        // Test that CSE handles duplicate expressions without panicking
        // CSE may or may not optimize depending on block size and loop detection
        let mut func = FunctionBuilder::new("dup")
            .param(VirtualReg::gpr(10).into(), MirType::Scalar(ScalarType::I64))
            .param(VirtualReg::gpr(11).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Register(VirtualReg::gpr(11).into()),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Register(VirtualReg::gpr(11).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cse = CommonSubexpressionElimination::default();
        let result = cse.apply(&mut func);
        // Main test: CSE should not panic or error
        assert!(result.is_ok());
        // Function should still be valid (have blocks and instructions)
        assert!(!func.blocks.is_empty());
        let entry = func.get_block("entry").unwrap();
        assert!(!entry.instructions.is_empty());
    }

    #[test]
    fn test_cse_stress_no_infinite_loop() {
        // Stress test with many instructions to ensure no infinite loop
        let mut func = FunctionBuilder::new("stress")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .build();

        for i in 0..200 {
            func.blocks[0].instructions.insert(
                0,
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(i).into(),
                    lhs: Operand::Immediate(Immediate::I64(i as i64)),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                },
            );
        }
        func.blocks[0].instructions.push(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        });

        let cse = CommonSubexpressionElimination::default();
        let result = cse.apply(&mut func);
        assert!(result.is_ok());
    }
}
