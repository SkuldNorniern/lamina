use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Immediate, Instruction, MirType, Operand, Register, ScalarType};
use std::collections::HashMap;

/// Copy Propagation Transform
/// Replaces variable uses with their source values when safe to do so
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
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl CopyPropagation {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
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

        // Process instructions in the block, building and using the value map
        let mut new_instructions = Vec::new();

        for instr in &block.instructions {
            let mut new_instr = instr.clone();

            // First, propagate any known copies into this instruction
            if self.propagate_copies(&mut new_instr, &value_map) {
                changed = true;
            }

            // Then, check if this instruction defines a copy that we should track
            if let Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                dst,
                lhs,
                rhs,
                ..
            } = &new_instr
            {
                // Check for x = y + 0 (copy pattern)
                if let Operand::Immediate(Immediate::I64(0)) = rhs
                    && let Operand::Register(src_reg) = lhs {
                        // Record this copy for future propagation within this block
                        value_map.insert(dst.clone(), Operand::Register(src_reg.clone()));
                    }
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
                        && let Operand::Register(new_reg) = new_base {
                            *base = new_reg.clone();
                            changed = true;
                        }
            }
            Instruction::Store { src, addr, .. } => {
                changed |= self.replace_operand(src, value_map);
                if let crate::mir::AddressMode::BaseOffset { base, offset: _ } = addr
                    && let Some(new_base) = value_map.get(base)
                        && let Operand::Register(new_reg) = new_base {
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
                    && let Operand::Register(new_reg) = new_cond {
                        *cond = new_reg.clone();
                        changed = true;
                    }
            }
            Instruction::Switch { value, .. } => {
                if let Some(new_value) = value_map.get(value)
                    && let Operand::Register(new_reg) = new_value {
                        *value = new_reg.clone();
                        changed = true;
                    }
            }
            Instruction::Ret { value } => {
                if let Some(val) = value {
                    changed |= self.replace_operand(val, value_map);
                }
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
            && let Some(replacement) = value_map.get(reg) {
                *operand = replacement.clone();
                return true;
            }
        false
    }

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
            | Instruction::VectorOp { ty, .. } => ty.clone(),
            // For other instructions that don't have explicit types, default to I64
            // This should be rare and these instructions probably shouldn't be CSE'd
            _ => MirType::Scalar(ScalarType::I64),
        }
    }
}

/// Constant Folding Transform
/// Evaluates constant expressions at compile time
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
        TransformLevel::Experimental
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
                    crate::mir::IntBinOp::Add => lhs_val + rhs_val,
                    crate::mir::IntBinOp::Sub => lhs_val - rhs_val,
                    crate::mir::IntBinOp::Mul => lhs_val * rhs_val,
                    crate::mir::IntBinOp::UDiv if rhs_val != 0 => lhs_val / rhs_val,
                    crate::mir::IntBinOp::SDiv if rhs_val != 0 => lhs_val / rhs_val,
                    crate::mir::IntBinOp::URem if rhs_val != 0 => lhs_val % rhs_val,
                    crate::mir::IntBinOp::SRem if rhs_val != 0 => lhs_val % rhs_val,
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
        TransformLevel::Experimental
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
            | Instruction::VectorOp { ty, .. } => ty.clone(),
            // For other instructions that don't have explicit types, default to I64
            // This should be rare and these instructions probably shouldn't be CSE'd
            _ => MirType::Scalar(ScalarType::I64),
        }
    }

    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        for block in &mut func.blocks {
            if self.eliminate_in_block(block) {
                changed = true;
            }
        }

        Ok(changed)
    }

    fn eliminate_in_block(&self, block: &mut Block) -> bool {
        let mut changed = false;
        let mut expr_to_reg: HashMap<String, Register> = HashMap::new();
        let mut instructions = Vec::new();

        for instr in &block.instructions {
            let expr_key = self.expr_key(instr);

            if let Some(expr_key) = expr_key {
                if let Some(existing_reg) = expr_to_reg.get(&expr_key) {
                    // Replace this instruction with a copy from the existing register
                    if let Some(dst) = instr.def_reg() {
                        let instr_type = self.extract_instruction_type(instr);
                        let copy_instr = Instruction::IntBinary {
                            op: crate::mir::IntBinOp::Add,
                            dst: dst.clone(),
                            ty: instr_type,
                            lhs: Operand::Register(existing_reg.clone()),
                            rhs: Operand::Immediate(Immediate::I64(0)),
                        };
                        instructions.push(copy_instr);
                        changed = true;
                        continue;
                    }
                } else {
                    // First time seeing this expression, record it
                    if let Some(dst) = instr.def_reg() {
                        expr_to_reg.insert(expr_key, dst.clone());
                    }
                }
            }

            instructions.push(instr.clone());
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
}
