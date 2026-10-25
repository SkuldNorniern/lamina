/// RISC-V stack frame management utilities
use crate::riscv::target::RiscVTarget;
use std::io::{Error, Write};

pub struct RiscVFrame;

/// Bytes the prologue reserves for the saved `ra` and `fp` pair.
const SAVED_PAIR_BYTES: i32 = 16;

/// Range of a 12-bit signed immediate, as used by `addi` and the load and store offsets.
const IMM12_MIN: i64 = -2048;
const IMM12_MAX: i64 = 2047;

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
            Self::adjust_sp(writer, -(locals as i64))?;
        }
        for (index, register) in callee_saved_registers.iter().enumerate() {
            let offset = Self::callee_saved_offset(stack_size, index, target);
            Self::fp_store(writer, sw, register, offset, "t0")?;
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
            Self::fp_load(writer, lw, register, offset, register)?;
        }

        // Deallocate stack space for local variables if needed
        let locals = Self::locals_bytes(stack_size, callee_saved_registers, target);
        if locals > 0 {
            Self::adjust_sp(writer, locals as i64)?;
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
            Self::fp_load(writer, lw, register, offset, register)?;
        }
        let locals = Self::locals_bytes(stack_size, callee_saved_registers, target);
        if locals > 0 {
            Self::adjust_sp(writer, locals as i64)?;
        }
        let w = target.word_bytes();
        writeln!(writer, "    {lw} ra, {}(fp)", -w)?;
        writeln!(writer, "    {lw} fp, {}(fp)", -2 * w)?;
        writeln!(writer, "    addi sp, sp, {SAVED_PAIR_BYTES}")?;
        writeln!(writer, "    j {target_sym}")?;
        Ok(())
    }

    /// Move `sp` by `delta` bytes. Deltas outside the 12-bit range go through `t0`,
    /// which the prologue, epilogue and call sequences do not hold a value in.
    pub fn adjust_sp<W: Write>(writer: &mut W, delta: i64) -> Result<(), Error> {
        if (IMM12_MIN..=IMM12_MAX).contains(&delta) {
            writeln!(writer, "    addi sp, sp, {delta}")
        } else {
            Self::load_imm(writer, "t0", delta)?;
            writeln!(writer, "    add sp, sp, t0")
        }
    }

    /// Emit `{mnemonic} {reg}, {offset}(fp)`, building the address in `scratch` when the
    /// offset does not fit the 12-bit immediate. `reg` is overwritten by a load, so it may
    /// serve as the scratch for one.
    pub fn fp_load<W: Write>(
        writer: &mut W,
        mnemonic: &str,
        reg: &str,
        offset: i32,
        scratch: &str,
    ) -> Result<(), Error> {
        if (IMM12_MIN..=IMM12_MAX).contains(&i64::from(offset)) {
            writeln!(writer, "    {mnemonic} {reg}, {offset}(fp)")
        } else {
            Self::load_imm(writer, scratch, i64::from(offset))?;
            writeln!(writer, "    add {scratch}, fp, {scratch}")?;
            writeln!(writer, "    {mnemonic} {reg}, 0({scratch})")
        }
    }

    /// Store `reg` at `{offset}(fp)`. A store must not clobber `reg`, so when the offset is
    /// out of range and `reg` is `t0`, `fp` itself is walked to the slot and back.
    pub fn fp_store<W: Write>(
        writer: &mut W,
        mnemonic: &str,
        reg: &str,
        offset: i32,
        scratch: &str,
    ) -> Result<(), Error> {
        if (IMM12_MIN..=IMM12_MAX).contains(&i64::from(offset)) {
            writeln!(writer, "    {mnemonic} {reg}, {offset}(fp)")
        } else if reg != scratch {
            Self::fp_load(writer, mnemonic, reg, offset, scratch)
        } else {
            // fp + (offset + 2048) + (-2048) is fp + offset.
            let walk = i64::from(offset) + 2048;
            Self::walk_fp(writer, walk)?;
            writeln!(writer, "    {mnemonic} {reg}, -2048(fp)")?;
            Self::walk_fp(writer, -walk)
        }
    }

    /// Materialise `fp + offset` into `dst`.
    pub fn fp_address<W: Write>(writer: &mut W, dst: &str, offset: i32) -> Result<(), Error> {
        if (IMM12_MIN..=IMM12_MAX).contains(&i64::from(offset)) {
            writeln!(writer, "    addi {dst}, fp, {offset}")
        } else {
            Self::load_imm(writer, dst, i64::from(offset))?;
            writeln!(writer, "    add {dst}, fp, {dst}")
        }
    }

    /// Load `value` into `reg`. ras only encodes `li` for 12-bit immediates, so larger values
    /// are built from `slli` and `addi` steps.
    pub fn load_imm<W: Write>(writer: &mut W, reg: &str, value: i64) -> Result<(), Error> {
        if (IMM12_MIN..=IMM12_MAX).contains(&value) {
            return writeln!(writer, "    li {reg}, {value}");
        }
        let low = ((value & 0xfff) ^ 0x800) - 0x800;
        Self::load_imm(writer, reg, (value - low) >> 12)?;
        writeln!(writer, "    slli {reg}, {reg}, 12")?;
        if low != 0 {
            writeln!(writer, "    addi {reg}, {reg}, {low}")?;
        }
        Ok(())
    }

    fn walk_fp<W: Write>(writer: &mut W, mut delta: i64) -> Result<(), Error> {
        while delta != 0 {
            let step = delta.clamp(IMM12_MIN, IMM12_MAX);
            writeln!(writer, "    addi fp, fp, {step}")?;
            delta -= step;
        }
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
    fn frame_beyond_the_immediate_range_is_adjusted_through_t0() -> Result<(), Box<dyn Error>> {
        let rv64 = RiscVTarget::general(Xlen::Rv64);
        let mut prologue = Vec::new();
        RiscVFrame::generate_prologue(&mut prologue, 40000, &[], rv64)?;
        let prologue = String::from_utf8(prologue)?;
        assert!(prologue.contains("    add sp, sp, t0\n"), "{prologue}");
        assert_eq!(eval_li_sequence(&prologue, "t0"), -40000, "{prologue}");
        assert!(!prologue.contains("addi sp, sp, -40000"), "{prologue}");
        Ok(())
    }

    /// Runs the `li`, `slli` and `addi` lines that load `reg`, to check the value they build.
    fn eval_li_sequence(asm: &str, reg: &str) -> i64 {
        let mut value = 0i64;
        for line in asm.lines().map(str::trim) {
            let (op, args) = line.split_once(' ').unwrap_or((line, ""));
            let mut parts = args.split(", ");
            if parts.next() != Some(reg) {
                continue;
            }
            match op {
                "li" => value = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "slli" => {
                    let shift: u32 = parts.nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                    value = value.wrapping_shl(shift);
                }
                "addi" => {
                    let imm: i64 = parts.nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                    value = value.wrapping_add(imm);
                }
                _ => {}
            }
        }
        value
    }

    #[test]
    fn load_imm_builds_every_tested_value() -> Result<(), Box<dyn Error>> {
        for value in [
            0,
            1,
            -1,
            2047,
            -2048,
            2048,
            -2049,
            40016,
            -40016,
            1 << 31,
            i32::MIN as i64,
            i64::MAX,
            i64::MIN,
        ] {
            let mut out = Vec::new();
            RiscVFrame::load_imm(&mut out, "t0", value)?;
            let asm = String::from_utf8(out)?;
            assert_eq!(eval_li_sequence(&asm, "t0"), value, "{asm}");
        }
        Ok(())
    }

    #[test]
    fn out_of_range_fp_access_builds_the_address_in_scratch() -> Result<(), Box<dyn Error>> {
        let mut out = Vec::new();
        RiscVFrame::fp_store(&mut out, "sd", "a0", -40016, "t0")?;
        RiscVFrame::fp_load(&mut out, "ld", "a1", -40016, "a1")?;
        let out = String::from_utf8(out)?;
        assert_eq!(
            out,
            concat!(
                "    li t0, -10\n    slli t0, t0, 12\n    addi t0, t0, 944\n",
                "    add t0, fp, t0\n    sd a0, 0(t0)\n",
                "    li a1, -10\n    slli a1, a1, 12\n    addi a1, a1, 944\n",
                "    add a1, fp, a1\n    ld a1, 0(a1)\n",
            )
        );
        Ok(())
    }

    #[test]
    fn out_of_range_store_of_the_scratch_register_walks_fp_and_back() -> Result<(), Box<dyn Error>>
    {
        let mut out = Vec::new();
        RiscVFrame::fp_store(&mut out, "sd", "t0", -40016, "t0")?;
        let out = String::from_utf8(out)?;
        let walked: i64 = out
            .lines()
            .filter_map(|line| line.trim().strip_prefix("addi fp, fp, "))
            .map(|v| v.parse::<i64>())
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum();
        assert_eq!(walked, 0, "fp must return to its value: {out}");
        assert!(out.contains("sd t0, -2048(fp)"), "{out}");
        assert!(
            out.lines()
                .filter_map(|line| line.trim().strip_prefix("addi fp, fp, "))
                .all(|v| (-2048..=2047).contains(&v.parse::<i64>().unwrap_or(i64::MAX))),
            "{out}"
        );
        Ok(())
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
