//! Code motion and propagation transforms for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Immediate, Instruction, MirType, Operand, Register, ScalarType};
use std::collections::HashMap;

/// Copy propagation transform that replaces variable uses with their source values.
#[derive(Default)]
pub struct CopyPropagation;

impl Transform for CopyPropagation {
    fn name(&self) -> &'static str {
        "copy_propagation"
    }

    fn description(&self) -> &'static str {
        "Replaces variable uses with their source values"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::CopyPropagation
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CopyPropagation {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        // Safety check: limit function size
        const MAX_BLOCKS: usize = 500;
        const MAX_INSTRUCTIONS_PER_BLOCK: usize = 1_000;

        if func.blocks.len() > MAX_BLOCKS {
            return Err(format!(
                "Function too large for copy propagation ({} blocks, max {})",
                func.blocks.len(),
                MAX_BLOCKS
            ));
        }

        for block in &func.blocks {
            if block.instructions.len() > MAX_INSTRUCTIONS_PER_BLOCK {
                return Err(format!(
                    "Block '{}' too large for copy propagation ({} instructions, max {})",
                    block.label,
                    block.instructions.len(),
                    MAX_INSTRUCTIONS_PER_BLOCK
                ));
            }
        }

        let mut changed = false;

        // Apply copy propagation within each basic block (safe: no control flow issues)
        for block in &mut func.blocks {
            if self.propagate_copies_in_block(block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn propagate_copies_in_block(&self, block: &mut Block) -> bool {
        let mut changed = false;
        let mut value_map = HashMap::new();

        let mut propagations_this_block = 0;
        let max_propagations_per_block = 50;

        let mut new_instructions = Vec::new();

        for instr in &block.instructions {
            let mut new_instr = instr.clone();

            if propagations_this_block < max_propagations_per_block
                && self.propagate_copies(&mut new_instr, &value_map)
            {
                changed = true;
                propagations_this_block += 1;
            }

            if let Some(def_reg) = new_instr.def_reg() {
                value_map.remove(def_reg);
            }
            if let Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst,
                lhs,
                rhs,
                ..
            } = &new_instr
                && let Operand::Immediate(Immediate::I64(0)) = rhs
                && let Operand::Register(src_reg) = lhs
            {
                value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
            }
            match &new_instr {
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Sub,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y - 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Mul,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y * 1 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(1)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                    // Check for x = 1 * y (copy pattern)
                    if let Operand::Immediate(Immediate::I64(1)) = lhs
                        && let Operand::Register(src_reg) = rhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Or,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y | 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                    // Check for x = 0 | y (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = lhs
                        && let Operand::Register(src_reg) = rhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::And,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y & -1 (copy pattern, since -1 is all bits set)
                    if let Operand::Immediate(Immediate::I64(-1)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Xor,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check for x = y ^ 0 (copy pattern)
                    if let Operand::Immediate(Immediate::I64(0)) = rhs
                        && let Operand::Register(src_reg) = lhs
                    {
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
                }
                _ => {}
            }

            new_instructions.push(new_instr);
        }

        block.instructions = new_instructions;
        changed
    }

    fn propagate_copies(
        &self,
        instr: &mut Instruction,
        value_map: &HashMap<Register, Operand>,
    ) -> bool {
        let mut changed = false;

        match instr {
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => {
                changed |= self.replace_operand(lhs, value_map);
                changed |= self.replace_operand(rhs, value_map);
            }
            Instruction::FloatUnary { src, .. } => {
                changed |= self.replace_operand(src, value_map);
            }
            Instruction::Select {
                cond: _,
                true_val,
                false_val,
                ..
            } => {
                changed |= self.replace_operand(true_val, value_map);
                changed |= self.replace_operand(false_val, value_map);
            }
            Instruction::Load { addr, .. } => {
                if let crate::mir::AddressMode::BaseOffset { base, offset: _ } = addr
                    && let Some(new_base) = value_map.get(base)
                    && let Operand::Register(new_reg) = new_base
                {
                    *base = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Store { src, addr, .. } => {
                changed |= self.replace_operand(src, value_map);
                if let crate::mir::AddressMode::BaseOffset { base, offset: _ } = addr
                    && let Some(new_base) = value_map.get(base)
                    && let Operand::Register(new_reg) = new_base
                {
                    *base = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    changed |= self.replace_operand(arg, value_map);
                }
            }
            Instruction::Br { cond, .. } => {
                if let Some(new_cond) = value_map.get(cond)
                    && let Operand::Register(new_reg) = new_cond
                {
                    *cond = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Switch { value, .. } => {
                if let Some(new_value) = value_map.get(value)
                    && let Operand::Register(new_reg) = new_value
                {
                    *value = new_reg.clone();
                    changed = true;
                }
            }
            Instruction::Ret { value: Some(val) } => {
                changed |= self.replace_operand(val, value_map);
            }
            _ => {}
        }

        changed
    }

    fn replace_operand(
        &self,
        operand: &mut Operand,
        value_map: &HashMap<Register, Operand>,
    ) -> bool {
        if let Operand::Register(reg) = operand
            && let Some(replacement) = value_map.get(reg)
        {
            *operand = replacement.clone();
            return true;
        }
        false
    }
}

/// Constant folding that evaluates constant expressions at compile time.
#[derive(Default)]
pub struct ConstantFolding;

impl Transform for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn description(&self) -> &'static str {
        "Evaluates constant expressions at compile time"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ConstantFolding
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl ConstantFolding {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                if self.try_fold_constants(instr) {
                    changed = true;
                }
            }
        }

        Ok(changed)
    }

    fn try_fold_constants(&self, instr: &mut Instruction) -> bool {
        if let Instruction::IntBinary {
            op, dst, lhs, rhs, ..
        } = instr
            && let (Some(lhs_val), Some(rhs_val)) =
                (self.extract_constant(lhs), self.extract_constant(rhs))
        {
            let result = match op {
                crate::mir::IntBinOp::Add => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_add(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::Sub => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_sub(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::Mul => {
                    // Use checked arithmetic to prevent overflow panics/wrapping
                    match lhs_val.checked_mul(rhs_val) {
                        Some(res) => res,
                        None => return false, // Skip folding on overflow
                    }
                }
                crate::mir::IntBinOp::UDiv if rhs_val != 0 => {
                    // Cast to u64 for proper unsigned division semantics
                    let lhs_u = lhs_val as u64;
                    let rhs_u = rhs_val as u64;
                    (lhs_u / rhs_u) as i64
                }
                crate::mir::IntBinOp::SDiv if rhs_val != 0 => {
                    // Check for overflow: i64::MIN / -1
                    if lhs_val == i64::MIN && rhs_val == -1 {
                        return false; // Skip folding to avoid overflow
                    }
                    // Keep signed division for SDiv
                    lhs_val / rhs_val
                }
                crate::mir::IntBinOp::URem if rhs_val != 0 => {
                    // Cast to u64 for proper unsigned remainder semantics
                    let lhs_u = lhs_val as u64;
                    let rhs_u = rhs_val as u64;
                    (lhs_u % rhs_u) as i64
                }
                crate::mir::IntBinOp::SRem if rhs_val != 0 => {
                    // Keep signed remainder for SRem
                    lhs_val % rhs_val
                }
                _ => return false,
            };

            // Replace the instruction with a load immediate
            *instr = Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst: dst.clone(),
                ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                lhs: Operand::Immediate(Immediate::I64(result)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            };
            return true;
        }
        false
    }

    fn extract_constant(&self, operand: &Operand) -> Option<i64> {
        match operand {
            Operand::Immediate(Immediate::I64(val)) => Some(*val),
            _ => None,
        }
    }
}

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
    fn test_copy_propagation_basic() {
        // Test basic copy propagation within a block
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Copy: v1 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // Should be replaced: v2 = v1 + 0 -> v2 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let cp = CopyPropagation::default();
        let changed = cp
            .apply(&mut func)
            .expect("Copy propagation should succeed");

        assert!(changed);
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 3);

        // Check that the second instruction was modified to use v0 directly
        match &entry.instructions[1] {
            Instruction::IntBinary { dst, lhs, rhs, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_signed_division_overflow_prevention() {
        // Test that i64::MIN / -1 is prevented (would overflow)
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::SDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(-1)),
        });
        func.add_block(bb);

        let pass = ConstantFolding::default();
        let changed = pass.try_fold_constants(&mut func.blocks[0].instructions[0]);
        // Should NOT change due to overflow prevention
        assert!(!changed);
    }

    #[test]
    fn test_copy_propagation_register_redefinition() {
        // Test that copy mappings are invalidated when registers are redefined
        let func = FunctionBuilder::new("test")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            // Copy: v1 = v0 + 0
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            // Redefine v1: v1 = v3 + v4 (not a copy)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(3).into()),
                rhs: Operand::Register(VirtualReg::gpr(4).into()),
            })
            // Should NOT be replaced: v2 = v1 + 0 should stay as v2 = v1 + 0
            // (not become v2 = v0 + 0 since v1 was redefined)
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let mut func = func;
        let cp = CopyPropagation::default();
        let changed = cp
            .apply(&mut func)
            .expect("Copy propagation should succeed");

        // Should have made changes (first copy propagation), but not invalid ones
        assert!(changed);
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 4);

        // Check that the third instruction was NOT modified (still uses v1)
        match &entry.instructions[2] {
            Instruction::IntBinary { dst, lhs, rhs, .. } => {
                assert_eq!(dst, &VirtualReg::gpr(2).into());
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(1).into())); // Should still be v1
                assert_eq!(rhs, &Operand::Immediate(Immediate::I64(0)));
            }
            _ => panic!("Expected IntBinary instruction"),
        }
    }

    #[test]
    fn test_copy_propagation_empty_function() {
        let mut func = FunctionBuilder::new("empty")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_copy_propagation_no_infinite_loop() {
        // Ensure copy propagation terminates on circular-looking patterns
        let mut func = FunctionBuilder::new("circular")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // Redefine v0
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        // Run multiple times to ensure no infinite loop
        for _ in 0..10 {
            let _ = cp.apply(&mut func);
        }
    }

    #[test]
    fn test_constant_folding_division_by_zero() {
        // Division by zero should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(100)),
            rhs: Operand::Immediate(Immediate::I64(0)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed);
    }

    #[test]
    fn test_constant_folding_overflow_sub() {
        // Subtraction overflow should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Sub,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MIN)),
            rhs: Operand::Immediate(Immediate::I64(1)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed); // Overflow, no fold
    }

    #[test]
    fn test_constant_folding_overflow_mul() {
        // Multiplication overflow should not be folded
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::Mul,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(i64::MAX)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(!changed); // Overflow, no fold
    }

    #[test]
    fn test_constant_folding_unsigned_division() {
        // Unsigned division: large positive / 2 should fold correctly
        let mut func = Function::new(crate::mir::function::Signature::new("f"))
            .with_entry("entry".to_string());
        let mut bb = Block::new("entry");
        bb.push(Instruction::IntBinary {
            op: IntBinOp::UDiv,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(-2)), // Interpreted as u64::MAX - 1
            rhs: Operand::Immediate(Immediate::I64(2)),
        });
        func.add_block(bb);

        let cf = ConstantFolding::default();
        let changed = cf.apply(&mut func).expect("should not panic");
        assert!(changed);

        // Check the result: (u64::MAX - 1) / 2 = 0x7FFFFFFFFFFFFFFE
        match &func.blocks[0].instructions[0] {
            Instruction::IntBinary { lhs, .. } => {
                let expected = ((-2i64 as u64) / 2) as i64;
                assert_eq!(lhs, &Operand::Immediate(Immediate::I64(expected)));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

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

    #[test]
    fn test_copy_propagation_mul_by_one() {
        // x * 1 is a copy pattern
        let mut func = FunctionBuilder::new("mul_one")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(5)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(2).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let changed = cp.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Second instruction should now use v0 directly
        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[1] {
            Instruction::IntBinary { lhs, .. } => {
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_copy_propagation_loop_counter_pattern() {
        // %i = add.i64 0, 0 (loop counter init)
        // followed by %i = add.i64 %i, 1 (loop increment)
        // Copy propagation should handle the redefinition correctly
        let mut func = FunctionBuilder::new("loop_counter")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %i = 0 + 0
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "loop".to_string(),
            })
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %i = %i + 1 (redefine same reg)
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should not crash on redefined registers
    }

    #[test]
    fn test_copy_propagation_block_size_pattern() {
        // %block_size = add.i64 8, 0 (constant via add)
        // This is a copy pattern that should be propagated
        let mut func = FunctionBuilder::new("block_size")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %block_size_i = 8 + 0
                lhs: Operand::Immediate(Immediate::I64(8)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(), // %i_end = %i_block + %block_size_i
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Register(VirtualReg::gpr(0).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
    }

    #[test]
    fn test_copy_propagation_chained_adds_zero() {
        // Multiple %x = add.i64 %y, 0 in sequence
        let mut func = FunctionBuilder::new("chained_zero")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(), // %a = %src + 0
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(), // %b = %a + 0
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(), // %c = %b + 0
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(4).into(), // use %c
                lhs: Operand::Register(VirtualReg::gpr(3).into()),
                rhs: Operand::Immediate(Immediate::I64(100)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(4).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let changed = cp.apply(&mut func).expect("should succeed");
        assert!(changed);

        // Final add should use v0 directly (all copies should chain)
        let entry = func.get_block("entry").unwrap();
        match &entry.instructions[3] {
            Instruction::IntBinary { lhs, .. } => {
                // After propagation, should trace back to v0
                assert_eq!(lhs, &Operand::Register(VirtualReg::gpr(0).into()));
            }
            _ => panic!("Expected IntBinary"),
        }
    }

    #[test]
    fn test_copy_propagation_same_reg_redefined_in_block() {
        // %a_elem = mul %i %k; %a_elem = add %a_elem 1
        // Same register redefined - must NOT propagate old value after redefinition
        let mut func = FunctionBuilder::new("redefine_same")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = 0 + 0 (creates copy mapping)
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Mul,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = %i * %k (REDEFINE - invalidates mapping)
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Register(VirtualReg::gpr(2).into()),
            })
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %x = %x + 1 (should use the mul result)
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should not crash and should handle redefinitions correctly
    }

    #[test]
    fn test_copy_propagation_unroll_pattern() {
        // Unrolled loop with many similar instructions
        // %j_plus_1 = add %j 1; %j_plus_2 = add %j 2; etc.
        let mut func = FunctionBuilder::new("unroll_pattern")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %j = %j_block + 0
                lhs: Operand::Register(VirtualReg::gpr(10).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .build();

        // Add 16 unrolled iterations
        let entry = func.blocks.get_mut(0).unwrap();
        for i in 1..=16 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(i).into(), // %j_plus_i = %j + i
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(i as i64)),
            });
        }

        // Use all the computed values
        let mut sum_reg = VirtualReg::gpr(1);
        for i in 2..=16 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(50 + i).into(),
                lhs: Operand::Register(sum_reg.into()),
                rhs: Operand::Register(VirtualReg::gpr(i).into()),
            });
            sum_reg = VirtualReg::gpr(50 + i);
        }

        entry.push(Instruction::Ret {
            value: Some(Operand::Register(sum_reg.into())),
        });

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        let changed = result.unwrap();
        assert!(changed); // Should propagate v10 -> v0
    }

    #[test]
    fn test_copy_propagation_accumulator_pattern() {
        // %total_sum = add %total_sum %product
        // Accumulator that gets updated repeatedly
        let mut func = FunctionBuilder::new("accumulator")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %total_sum = 0 + 0
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .build();

        let entry = func.blocks.get_mut(0).unwrap();
        // Simulate 8 accumulation steps
        for i in 1..=8 {
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(), // %total_sum = %total_sum + %product_i
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Register(VirtualReg::gpr(100 + i).into()),
            });
        }

        entry.push(Instruction::Ret {
            value: Some(Operand::Register(VirtualReg::gpr(0).into())),
        });

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Should handle accumulator pattern without issues
    }

    #[test]
    fn test_copy_propagation_many_blocks_with_copies() {
        // Pattern: Multiple blocks each with copy patterns
        // Tests cross-block isolation (copies don't propagate across blocks)
        let mut func = FunctionBuilder::new("multi_block_copies")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "block1".to_string(),
            })
            .block("block1")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(1).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Jmp {
                target: "block2".to_string(),
            })
            .block("block2")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::gpr(2).into()),
                rhs: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(3).into())),
            })
            .build();

        let cp = CopyPropagation::default();
        let result = cp.apply(&mut func);
        assert!(result.is_ok());
        // Each block should handle its own copies independently
    }
}
