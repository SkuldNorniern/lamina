//! I/O operations for IR builder.
//!
//! This module provides raw byte input/output: reading from stdin and writing to stdout.

use crate::{builder::IRBuilder, instruction::Instruction, types::Value};

impl<'a> IRBuilder<'a> {
    /// Writes a buffer to stdout (raw syscall)
    pub fn write(&mut self, buffer: Value<'a>, size: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::Write {
            buffer,
            size,
            result,
        })
    }

    /// Reads from stdin into a buffer (raw syscall)
    pub fn read(&mut self, buffer: Value<'a>, size: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::Read {
            buffer,
            size,
            result,
        })
    }

    /// Writes a single byte to stdout (raw syscall)
    pub fn write_byte(&mut self, value: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::WriteByte { value, result })
    }

    /// Reads a single byte from stdin (raw syscall)
    pub fn read_byte(&mut self, result: &'a str) -> &mut Self {
        self.inst(Instruction::ReadByte { result })
    }
}
