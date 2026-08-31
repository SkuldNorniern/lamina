/// RISC-V stack frame management utilities
use crate::riscv::target::RiscVTarget;
use std::io::{Error, Write};

pub struct RiscVFrame;

/// Bytes the prologue reserves for the saved `ra` and `fp` pair.
const SAVED_PAIR_BYTES: i32 = 16;

impl RiscVFrame {
    /// Round the locals area up so `sp` keeps the 16-byte alignment the ABI requires.
    ///
    /// Callers size this as `slots * 8`, so an odd slot count would otherwise leave `sp`
    /// 8-byte aligned for the whole call.
    fn locals_bytes(
        stack_size: usize,
        callee_saved_registers: &[&str],
        target: RiscVTarget,
    ) -> usize {
        let callee_saved_bytes = callee_saved_registers.len() * target.word_bytes() as usize;
        (stack_size + callee_saved_bytes).div_ceil(16) * 16
    }

    fn callee_saved_offset(stack_size: usize, index: usize, target: RiscVTarget) -> i32 {
        -(SAVED_PAIR_BYTES + stack_size as i32 + (index as i32 + 1) * target.word_bytes())
    }

    /// Generate function prologue
    pub fn generate_prologue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        callee_saved_registers: &[&str],
        target: RiscVTarget,
    ) -> Result<(), Error> {
        // 16 either way: both ABIs want sp 16-byte aligned.
        let w = target.word_bytes();
        let sw = target.store_word();
        writeln!(writer, "    addi sp, sp, -{SAVED_PAIR_BYTES}")?;
        writeln!(writer, "    {sw} ra, {}(sp)", SAVED_PAIR_BYTES - w)?;
        writeln!(writer, "    {sw} fp, {}(sp)", SAVED_PAIR_BYTES - 2 * w)?;
        writeln!(writer, "    addi fp, sp, {SAVED_PAIR_BYTES}")?;

        // Allocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size, callee_saved_registers, target);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, -{locals}")?;
        }
        for (index, register) in callee_saved_registers.iter().enumerate() {
            let offset = Self::callee_saved_offset(stack_size, index, target);
            writeln!(writer, "    {sw} {register}, {offset}(fp)")?;
        }
        Ok(())
    }

    /// Generate function epilogue
    pub fn generate_epilogue<W: Write>(
        writer: &mut W,
        stack_size: usize,
        callee_saved_registers: &[&str],
        target: RiscVTarget,
    ) -> Result<(), Error> {
        let lw = target.load_word();
        for (index, register) in callee_saved_registers.iter().enumerate().rev() {
            let offset = Self::callee_saved_offset(stack_size, index, target);
            writeln!(writer, "    {lw} {register}, {offset}(fp)")?;
        }

        // Deallocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size, callee_saved_registers, target);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }

        // Restore return address and frame pointer
        let w = target.word_bytes();
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
        callee_saved_registers: &[&str],
        target_sym: &str,
        target: RiscVTarget,
    ) -> Result<(), Error> {
        let lw = target.load_word();
        for (index, register) in callee_saved_registers.iter().enumerate().rev() {
            let offset = Self::callee_saved_offset(stack_size, index, target);
            writeln!(writer, "    {lw} {register}, {offset}(fp)")?;
        }
        let locals = Self::locals_bytes(stack_size, callee_saved_registers, target);
        if locals > 0 {
            writeln!(writer, "    addi sp, sp, {locals}")?;
        }
        let w = target.word_bytes();
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
    use std::error::Error;

    #[test]
    fn locals_never_land_on_the_saved_pair() {
        let rv64 = RiscVTarget::general(Xlen::Rv64);
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
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        assert_eq!(RiscVFrame::locals_bytes(0, &[], rv64), 0);
        assert_eq!(RiscVFrame::locals_bytes(8, &[], rv64), 16);
        assert_eq!(RiscVFrame::locals_bytes(16, &[], rv64), 16);
        assert_eq!(RiscVFrame::locals_bytes(24, &[], rv64), 32);
        assert_eq!(RiscVFrame::locals_bytes(16, &["s1"], rv64), 32);
    }

    #[test]
    fn prologue_and_epilogue_move_sp_by_the_same_amount() {
        let mut pro = Vec::new();
        let mut epi = Vec::new();
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        RiscVFrame::generate_prologue(&mut pro, 24, &[], rv64).expect("prologue");
        RiscVFrame::generate_epilogue(&mut epi, 24, &[], rv64).expect("epilogue");
        let pro = String::from_utf8(pro).expect("utf8");
        let epi = String::from_utf8(epi).expect("utf8");
        assert!(pro.contains("addi sp, sp, -32"), "prologue was:\n{pro}");
        assert!(epi.contains("addi sp, sp, 32"), "epilogue was:\n{epi}");
    }

    #[test]
    fn callee_saved_registers_have_frame_slots_and_are_restored() -> Result<(), Box<dyn Error>> {
        let mut prologue = Vec::new();
        let mut epilogue = Vec::new();
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        RiscVFrame::generate_prologue(&mut prologue, 16, &["s1"], rv64)?;
        RiscVFrame::generate_epilogue(&mut epilogue, 16, &["s1"], rv64)?;
        let prologue = String::from_utf8(prologue)?;
        let epilogue = String::from_utf8(epilogue)?;
        assert!(prologue.contains("addi sp, sp, -32"));
        assert!(prologue.contains("sd s1, -40(fp)"));
        assert!(epilogue.contains("ld s1, -40(fp)"));
        assert!(epilogue.contains("addi sp, sp, 32"));
        Ok(())
    }
}
