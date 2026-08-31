/// RISC-V stack frame management utilities
use crate::riscv::target::RiscVTarget;
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
    pub fn generate_prologue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        target: RiscVTarget,
    ) -> Result<(), Error> {
        // Reserve 16 either way, because both ABIs want sp 16-byte aligned. On rv32 the
        // pair only fills the top 8 bytes of that and the rest is padding.
        let w = target.word_bytes();
        let sw = target.store_word();
        writeln!(writer, "    addi sp, sp, -{SAVED_PAIR_BYTES}")?;
        writeln!(writer, "    {sw} ra, {}(sp)", SAVED_PAIR_BYTES - w)?;
        writeln!(writer, "    {sw} fp, {}(sp)", SAVED_PAIR_BYTES - 2 * w)?;
        writeln!(writer, "    addi fp, sp, {SAVED_PAIR_BYTES}")?;

        // Allocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, -{locals}")?;
        }
        Ok(())
    }

    /// Generate function epilogue
    pub fn generate_epilogue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        target: RiscVTarget,
    ) -> Result<(), Error> {
        // Deallocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }

        // Restore return address and frame pointer
        let w = target.word_bytes();
        let lw = target.load_word();
        writeln!(writer, "    {lw} ra, {}(fp)", -w)?;
        writeln!(writer, "    {lw} fp, {}(fp)", -2 * w)?;
        writeln!(writer, "    addi sp, sp, {SAVED_PAIR_BYTES}")?;
        writeln!(writer, "    ret")?;
        Ok(())
    }

    /// Like [`Self::generate_epilogue`], but jump to `target_sym` instead of `ret` (tail call).
    pub fn generate_tail_epilogue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        target_sym: &str,
        target: RiscVTarget,
    ) -> Result<(), Error> {
        let locals = Self::locals_bytes(stack_size);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }
        let w = target.word_bytes();
        let lw = target.load_word();
        writeln!(writer, "    {lw} ra, {}(fp)", -w)?;
        writeln!(writer, "    {lw} fp, {}(fp)", -2 * w)?;
        writeln!(writer, "    addi sp, sp, {SAVED_PAIR_BYTES}")?;
        writeln!(writer, "    j {target_sym}")?;
        Ok(())
    }

    /// Offset of a local slot from `fp`.
    ///
    /// The prologue leaves `fp` at the caller's `sp`, with `ra` at `fp-8` and the caller's
    /// `fp` at `fp-16`. Locals start below that pair. This used to return `fp-8` for slot
    /// 0 and `fp-16` for slot 1, so the first local overwrote the return address and the
    /// second overwrote the saved frame pointer.
    pub fn calculate_stack_offset(slot_index: usize, target: RiscVTarget) -> i32 {
        -(SAVED_PAIR_BYTES + (slot_index as i32 + 1) * target.word_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riscv::target::Xlen;

    #[test]
    fn locals_never_land_on_the_saved_pair() {
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        // fp-8 holds ra and fp-16 holds the caller's fp, so no local may sit there.
        for slot in 0..8 {
            let off = RiscVFrame::calculate_stack_offset(slot, rv64);
            assert!(off <= -24, "slot {slot} at fp{off} overlaps the saved pair");
            assert_eq!(off % 8, 0, "slot {slot} at fp{off} is not 8-byte aligned");
        }
        assert_eq!(RiscVFrame::calculate_stack_offset(0, rv64), -24);
        assert_eq!(RiscVFrame::calculate_stack_offset(1, rv64), -32);
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
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        RiscVFrame::generate_prologue(&mut pro, 24, rv64).expect("prologue");
        RiscVFrame::generate_epilogue(&mut epi, 24, rv64).expect("epilogue");
        let pro = String::from_utf8(pro).expect("utf8");
        let epi = String::from_utf8(epi).expect("utf8");
        assert!(pro.contains("addi sp, sp, -32"), "prologue was:\n{pro}");
        assert!(epi.contains("addi sp, sp, 32"), "epilogue was:\n{epi}");
    }
}
