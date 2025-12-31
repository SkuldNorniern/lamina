//! I/O operations for IR builder.
//!
//! This module provides methods for input/output operations including reading
//! from stdin, writing to stdout, and printing values for debugging.

use super::IRBuilder;
use crate::ir::instruction::Instruction;
use crate::ir::types::Value;

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

    /// Writes the value stored at a pointer location to stdout (I/O operation)
    pub fn write_ptr(&mut self, ptr: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::WritePtr { ptr, result })
    }

    /// Creates a print instruction for debugging
    pub fn print(&mut self, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Print { value })
    }
}
