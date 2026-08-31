/// RISC-V stack frame management utilities
use std::io::Error;
use std::io::Write;

pub struct RiscVFrame;

/// Bytes the prologue reserves for the saved `ra` and `fp` pair.
const SAVED_PAIR_BYTES: i32 = 16;

impl RiscVFrame {
    /// Round the locals area up so `sp` keeps the 16-byte alignment the ABI requires.
    ///
    /// Callers size this as `slots * 8`, so an odd slot count would otherwise leave `sp`
    /// 8-byte aligned for the whole call.
    fn locals_bytes(stack_size: usize) -> usize {
        stack_size.div_ceil(16) * 16
    }

    /// Generate function prologue
    pub fn generate_prologue<W: Write>(writer: &mut W, stack_size: usize) -> Result<(), Error> {
        // Save return address and frame pointer
        writeln!(writer, "    addi sp, sp, -16")?;
        writeln!(writer, "    sd ra, 8(sp)")?;
        writeln!(writer, "    sd fp, 0(sp)")?;
        writeln!(writer, "    addi fp, sp, 16")?;

        // Allocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, -{locals}")?;
        }
        Ok(())
    }

    /// Generate function epilogue
    pub fn generate_epilogue<W: Write>(writer: &mut W, stack_size: usize) -> Result<(), Error> {
        // Deallocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }

        // Restore return address and frame pointer
        writeln!(writer, "    ld ra, -8(fp)")?;
        writeln!(writer, "    ld fp, -16(fp)")?;
        writeln!(writer, "    addi sp, sp, 16")?;
        writeln!(writer, "    ret")?;
        Ok(())
    }

    /// Like [`Self::generate_epilogue`], but jump to `target_sym` instead of `ret` (tail call).
    pub fn generate_tail_epilogue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        target_sym: &str,
    ) -> Result<(), Error> {
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }
        writeln!(writer, "    ld ra, -8(fp)")?;
        writeln!(writer, "    ld fp, -16(fp)")?;
        writeln!(writer, "    addi sp, sp, 16")?;
        writeln!(writer, "    j {target_sym}")?;
        Ok(())
    }

    /// Offset of a local slot from `fp`.
    ///
    /// The prologue leaves `fp` at the caller's `sp`, with `ra` at `fp-8` and the caller's
    /// `fp` at `fp-16`. Locals start below that pair. This used to return `fp-8` for slot
    /// 0 and `fp-16` for slot 1, so the first local overwrote the return address and the
    /// second overwrote the saved frame pointer.
    pub fn calculate_stack_offset(slot_index: usize) -> i32 {
        -(SAVED_PAIR_BYTES + (slot_index as i32 + 1) * 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locals_never_land_on_the_saved_pair() {
        // fp-8 holds ra and fp-16 holds the caller's fp, so no local may sit there.
        for slot in 0..8 {
            let off = RiscVFrame::calculate_stack_offset(slot);
            assert!(off <= -24, "slot {slot} at fp{off} overlaps the saved pair");
            assert_eq!(off % 8, 0, "slot {slot} at fp{off} is not 8-byte aligned");
        }
        assert_eq!(RiscVFrame::calculate_stack_offset(0), -24);
        assert_eq!(RiscVFrame::calculate_stack_offset(1), -32);
    }

    #[test]
    fn locals_area_keeps_sp_16_byte_aligned() {
        // Callers pass slots * 8, so an odd count would misalign sp for the whole call.
        assert_eq!(RiscVFrame::locals_bytes(0), 0);
        assert_eq!(RiscVFrame::locals_bytes(8), 16);
        assert_eq!(RiscVFrame::locals_bytes(16), 16);
        assert_eq!(RiscVFrame::locals_bytes(24), 32);
    }

    #[test]
    fn prologue_and_epilogue_move_sp_by_the_same_amount() {
        let mut pro = Vec::new();
        let mut epi = Vec::new();
        RiscVFrame::generate_prologue(&mut pro, 24).expect("prologue");
        RiscVFrame::generate_epilogue(&mut epi, 24).expect("epilogue");
        let pro = String::from_utf8(pro).expect("utf8");
        let epi = String::from_utf8(epi).expect("utf8");
        assert!(pro.contains("addi sp, sp, -32"), "prologue was:\n{pro}");
        assert!(epi.contains("addi sp, sp, 32"), "epilogue was:\n{epi}");
    }
}
