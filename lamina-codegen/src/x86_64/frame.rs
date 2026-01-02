//! x86_64 stack frame management utilities.

/// Stack frame utilities for x86_64 code generation.
pub struct X86Frame;

impl X86Frame {
    /// Generates the function prologue: saves frame pointer and allocates stack space.
    pub fn generate_prologue<W: std::io::Write>(
        writer: &mut W,
        stack_size: usize,
    ) -> Result<(), std::io::Error> {
        writeln!(writer, "    pushq %rbp")?;
        writeln!(writer, "    movq %rsp, %rbp")?;
        if stack_size > 0 {
            writeln!(writer, "    subq ${}, %rsp", stack_size)?;
        }
        Ok(())
    }

    /// Generates the function epilogue: restores stack and frame pointer, then returns.
    pub fn generate_epilogue<W: std::io::Write>(
        writer: &mut W,
        stack_size: usize,
    ) -> Result<(), std::io::Error> {
        if stack_size > 0 {
            writeln!(writer, "    addq ${}, %rsp", stack_size)?;
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
