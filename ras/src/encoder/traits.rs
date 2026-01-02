//! Instruction encoder trait
//!
//! This trait defines the interface for encoding instructions to binary machine code.

use crate::error::RasError;

/// Parsed instruction from assembly (simplified for now)
#[derive(Debug, Clone)]
pub struct ParsedInstruction {
    pub opcode: String,
    pub operands: Vec<String>,
}

/// Trait for encoding instructions to binary machine code
pub trait InstructionEncoder {
    /// Encode a parsed instruction to binary
    fn encode_instruction(
        &mut self,
        inst: &ParsedInstruction,
    ) -> Result<Vec<u8>, RasError>;

    /// Get current code position (for relocations)
    fn current_position(&self) -> usize;
}

