/// RISC-V stack frame management utilities
pub struct RiscVFrame;

impl RiscVFrame {
    /// Generate function prologue
    pub fn generate_prologue<W: std::io::Write>(
        writer: &mut W,
        stack_size: usize,
    ) -> Result<(), std::io::Error> {
        // Save return address and frame pointer
        writeln!(writer, "    addi sp, sp, -16")?;
        writeln!(writer, "    sd ra, 8(sp)")?;
        writeln!(writer, "    sd fp, 0(sp)")?;
        writeln!(writer, "    addi fp, sp, 16")?;

        // Allocate stack space for local variables if needed
        if stack_size > 0 {
            writeln!(writer, "    addi sp, sp, -{}", stack_size)?;
        }
        Ok(())
    }

    /// Generate function epilogue
    pub fn generate_epilogue<W: std::io::Write>(
        writer: &mut W,
        stack_size: usize,
    ) -> Result<(), std::io::Error> {
        // Deallocate stack space for local variables if needed
        if stack_size > 0 {
            writeln!(writer, "    addi sp, sp, {}", stack_size)?;
        }

        // Restore return address and frame pointer
        writeln!(writer, "    ld ra, -8(fp)")?;
        writeln!(writer, "    ld fp, -16(fp)")?;
        writeln!(writer, "    addi sp, sp, 16")?;
        writeln!(writer, "    ret")?;
        Ok(())
    }

    /// Calculate stack slot offset from frame pointer (fp)
    pub fn calculate_stack_offset(slot_index: usize) -> i32 {
        -((slot_index as i32 + 1) * 8)
    }
}

