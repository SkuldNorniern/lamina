//! basic blocks for LUMIR
//!
//! A basic block is a sequence of instructions with a single entry and exit point.
use super::instruction::Instruction;
use std::fmt;

/// Block in LUMIR
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// Block label/name
    pub label: String,

    /// Instructions in this block
    pub instructions: Vec<Instruction>,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for instr in &self.instructions {
            writeln!(f, "  {}", instr)?;
        }
        Ok(())
    }
}

impl Block {
    /// Create a new basic block with the given label
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            instructions: Vec::new(),
        }
    }

    /// Add an instruction to this block
    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    /// Get the terminator instruction (last instruction)
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last()
    }

    /// Check if this block has a terminator
    pub fn has_terminator(&self) -> bool {
        self.terminator()
            .map(|i| i.is_terminator())
            .unwrap_or(false)
    }

    /// Get all non-terminator instructions
    pub fn body(&self) -> &[Instruction] {
        if self.has_terminator() {
            &self.instructions[..self.instructions.len() - 1]
        } else {
            &self.instructions
        }
    }

    /// Number of instructions in this block
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if this block is empty
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Get successor block labels
    pub fn successors(&self) -> Vec<String> {
        match self.terminator() {
            Some(Instruction::Jmp { target }) => vec![target.clone()],
            Some(Instruction::Br {
                true_target,
                false_target,
                ..
            }) => vec![true_target.clone(), false_target.clone()],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::instruction::{Immediate, Instruction, IntBinOp, Operand};
    use crate::mir::register::{Register, VirtualReg};
    use crate::mir::types::{MirType, ScalarType};

    #[test]
    fn test_basic_block_creation() {
        let bb = Block::new("entry");
        assert_eq!(bb.label, "entry");
        assert!(bb.is_empty());
        assert!(!bb.has_terminator());
    }

    #[test]
    fn test_basic_block_push() {
        let mut bb = Block::new("bb0");

        let add = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        };

        bb.push(add);
        assert_eq!(bb.len(), 1);
    }

    #[test]
    fn test_basic_block_terminator() {
        let mut bb = Block::new("bb0");

        let add = Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: MirType::Scalar(ScalarType::I64),
            dst: Register::Virtual(VirtualReg::gpr(0)),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        };

        let ret = Instruction::Ret { value: None };

        bb.push(add.clone());
        assert!(!bb.has_terminator());

        bb.push(ret);
        assert!(bb.has_terminator());
        assert_eq!(bb.body().len(), 1);
    }
}
