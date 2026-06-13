//! x86_64 stack frame management utilities.

use std::io::{Error, Write};

/// Stack frame utilities for x86_64 code generation.
pub struct X86Frame;

impl X86Frame {
    /// Generates the function prologue: saves frame pointer and allocates stack space.
    pub fn generate_prologue<W: Write>(writer: &mut W, stack_size: usize) -> Result<(), Error> {
        writeln!(writer, "    pushq %rbp")?;
        writeln!(writer, "    movq %rsp, %rbp")?;
        if stack_size > 0 {
            writeln!(writer, "    subq ${stack_size}, %rsp")?;
        }
        Ok(())
    }

    /// Generates the function epilogue: restores stack and frame pointer, then returns.
    pub fn generate_epilogue<W: Write>(writer: &mut W, stack_size: usize) -> Result<(), Error> {
        if stack_size > 0 {
            writeln!(writer, "    addq ${stack_size}, %rsp")?;
        }
        writeln!(writer, "    popq %rbp")?;
        writeln!(writer, "    ret")?;
        Ok(())
    }

    /// Calculates the stack slot offset from RBP for a given slot index.
    pub fn calculate_stack_offset(slot_index: usize) -> i32 {
        -((slot_index as i32 + 1) * 8)
    }
}
