//! PowerPC64 ELFv2 stack frame management.
//!
//! ELFv2 minimum frame layout:
//!   [sp+0]   back-chain pointer (previous sp)
//!   [sp+16]  LR save area
//!   [sp+32+] local variables
//!
//! The frame is grown by subtracting from r1 and the back-chain must be
//! written before we clobber anything.

use std::io::{self, Write};

/// Minimum linkage area size (back-chain + cr save + LR save + padding).
const LINKAGE_SIZE: usize = 32;

/// Range of the signed 16-bit displacement of D-form loads, stores and `addi`.
const IMM16_MIN: i64 = -32768;
const IMM16_MAX: i64 = 32767;

pub struct Ppc64Frame;

impl Ppc64Frame {
    /// Total stack frame size (linkage + locals), 16-byte aligned.
    pub fn aligned_frame_size(local_bytes: usize) -> usize {
        ((LINKAGE_SIZE + local_bytes) + 15) & !15
    }

    /// Calculate the stack offset for local slot `n` (0-based).
    ///
    /// Slots start above the linkage area and grow upward in address
    /// (i.e., at higher positive offsets from the old sp stored as
    /// back-chain).  In the prologue we subtract `frame_size` from r1,
    /// so slot `n` is at `LINKAGE_SIZE + n*8` from the new r1.
    pub fn calculate_stack_offset(slot: usize) -> i32 {
        (LINKAGE_SIZE + slot * 8) as i32
    }

    /// Emit the function prologue.
    ///
    /// Saves LR, creates the stack frame, stores callee-saved r14-r31
    /// if needed (we don't track that here, so we just save LR and
    /// update the back-chain).
    pub fn generate_prologue<W: Write>(writer: &mut W, local_bytes: usize) -> io::Result<()> {
        let frame_size = Self::aligned_frame_size(local_bytes) as i64;
        writeln!(writer, "    mflr 0")?;
        writeln!(writer, "    std 0, 16(1)")?; // Save LR to LR save slot
        if -frame_size >= IMM16_MIN {
            writeln!(writer, "    stdu 1, -{frame_size}(1)")?; // Allocate frame + store back-chain
        } else {
            // stdux stores r1 at r1 + r12 and moves r1 there, which is stdu with a register offset.
            load_imm(writer, "12", -frame_size)?;
            writeln!(writer, "    stdux 1, 1, 12")?;
        }
        Ok(())
    }

    /// Emit the function epilogue and return.
    pub fn generate_epilogue<W: Write>(writer: &mut W, local_bytes: usize) -> io::Result<()> {
        let frame_size = Self::aligned_frame_size(local_bytes) as i64;
        adjust_sp(writer, frame_size)?; // Deallocate frame
        writeln!(writer, "    ld 0, 16(1)")?; // Restore LR
        writeln!(writer, "    mtlr 0")?;
        writeln!(writer, "    blr")?;
        Ok(())
    }

    /// Deallocate the frame and restore LR, then the caller must branch to the tail target
    /// (no `blr` — that would return to this function instead of tail-calling).
    pub fn generate_tail_epilogue<W: Write>(writer: &mut W, local_bytes: usize) -> io::Result<()> {
        let frame_size = Self::aligned_frame_size(local_bytes) as i64;
        adjust_sp(writer, frame_size)?;
        writeln!(writer, "    ld 0, 16(1)")?;
        writeln!(writer, "    mtlr 0")?;
        Ok(())
    }
}

/// Load `value` into the GPR `reg`. Values outside the signed 16-bit range take two
/// instructions. The high half is added to `reg` with `addis`, which reads `r0` as zero,
/// so `r0` is loaded with `lis` and `ori` and only takes non-negative values.
pub fn load_imm<W: Write>(writer: &mut W, reg: &str, value: i64) -> io::Result<()> {
    let out_of_range = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("PowerPC64 immediate {value} does not fit a 32-bit load"),
        )
    };
    if (IMM16_MIN..=IMM16_MAX).contains(&value) {
        return writeln!(writer, "    li {reg}, {value}");
    }
    if reg == "0" {
        if !(0..(1 << 31)).contains(&value) {
            return Err(out_of_range());
        }
        writeln!(writer, "    lis 0, {}", value >> 16)?;
        return writeln!(writer, "    ori 0, 0, {}", value & 0xffff);
    }
    let low = i64::from((value & 0xffff) as u16 as i16);
    let high = (value - low) >> 16;
    if i32::try_from(value).is_err() || !(IMM16_MIN..=IMM16_MAX).contains(&high) {
        return Err(out_of_range());
    }
    writeln!(writer, "    li {reg}, {low}")?;
    writeln!(writer, "    addis {reg}, {reg}, {high}")
}

/// Load or store `reg` at `offset(1)`. An offset past the signed 16-bit range is built in
/// `r0` and used as the index of the X-form instruction. `r12` stands in when `reg` is `r0`.
pub fn r1_mem<W: Write>(writer: &mut W, mnemonic: &str, reg: &str, offset: i32) -> io::Result<()> {
    if (IMM16_MIN..=IMM16_MAX).contains(&i64::from(offset)) {
        return writeln!(writer, "    {mnemonic} {reg}, {offset}(1)");
    }
    let scratch = if reg == "0" { "12" } else { "0" };
    load_imm(writer, scratch, i64::from(offset))?;
    writeln!(writer, "    {mnemonic}x {reg}, 1, {scratch}")
}

/// Materialise `r1 + offset` into `dst`.
pub fn r1_address<W: Write>(writer: &mut W, dst: &str, offset: i32) -> io::Result<()> {
    if (IMM16_MIN..=IMM16_MAX).contains(&i64::from(offset)) {
        return writeln!(writer, "    addi {dst}, 1, {offset}");
    }
    load_imm(writer, dst, i64::from(offset))?;
    writeln!(writer, "    add {dst}, 1, {dst}")
}

/// Move `r1` by `delta`. Deltas past the 16-bit range go through `r12`.
fn adjust_sp<W: Write>(writer: &mut W, delta: i64) -> io::Result<()> {
    if (IMM16_MIN..=IMM16_MAX).contains(&delta) {
        writeln!(writer, "    addi 1, 1, {delta}")
    } else {
        load_imm(writer, "12", delta)?;
        writeln!(writer, "    add 1, 1, 12")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_offset_slot_0() {
        assert_eq!(Ppc64Frame::calculate_stack_offset(0), LINKAGE_SIZE as i32);
    }

    #[test]
    fn test_stack_offset_slot_1() {
        assert_eq!(
            Ppc64Frame::calculate_stack_offset(1),
            (LINKAGE_SIZE + 8) as i32
        );
    }

    #[test]
    fn test_prologue_emits_stdu() {
        let mut buf = Vec::new();
        Ppc64Frame::generate_prologue(&mut buf, 0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("stdu"), "Expected stdu in prologue: {s}");
        assert!(s.contains("mflr"), "Expected mflr in prologue: {s}");
    }

    #[test]
    fn test_large_frame_is_allocated_with_stdux() {
        let mut buf = Vec::new();
        Ppc64Frame::generate_prologue(&mut buf, 40000).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(!s.contains("stdu 1"), "{s}");
        assert!(s.contains("stdux 1, 1, 12"), "{s}");
        assert_eq!(ppc_value_of(&s, "12"), -40032, "{s}");
    }

    #[test]
    fn test_large_frame_is_released_through_r12() {
        let mut buf = Vec::new();
        Ppc64Frame::generate_epilogue(&mut buf, 40000).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("add 1, 1, 12"), "{s}");
        assert_eq!(ppc_value_of(&s, "12"), 40032, "{s}");
    }

    /// Replays the `li`, `lis`, `addis` and `ori` lines that load `reg`.
    fn ppc_value_of(asm: &str, reg: &str) -> i64 {
        let mut value = 0i64;
        for line in asm.lines().map(str::trim) {
            let (op, args) = line.split_once(' ').unwrap_or((line, ""));
            let parts: Vec<&str> = args.split(", ").collect();
            match (op, parts.as_slice()) {
                ("li", [r, v]) if *r == reg => value = v.parse().unwrap(),
                ("addis", [r, _, v]) if *r == reg => value += v.parse::<i64>().unwrap() << 16,
                _ => {}
            }
        }
        value
    }

    #[test]
    fn test_load_imm_covers_the_32_bit_range() {
        for value in [
            0i64,
            1,
            -1,
            32767,
            -32768,
            32768,
            40032,
            65536,
            0x7FFF_0000,
            -40032,
        ] {
            let mut buf = Vec::new();
            load_imm(&mut buf, "12", value).unwrap();
            let s = String::from_utf8(buf).unwrap();
            assert_eq!(ppc_value_of(&s, "12"), value, "{s}");
        }
    }

    #[test]
    fn test_epilogue_emits_blr() {
        let mut buf = Vec::new();
        Ppc64Frame::generate_epilogue(&mut buf, 0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("blr"), "Expected blr in epilogue: {s}");
        assert!(s.contains("mtlr"), "Expected mtlr in epilogue: {s}");
    }

    #[test]
    fn test_tail_epilogue_has_no_blr() {
        let mut buf = Vec::new();
        Ppc64Frame::generate_tail_epilogue(&mut buf, 0).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(!s.contains("blr"), "tail epilogue must not return: {s}");
        assert!(s.contains("mtlr"), "Expected mtlr: {s}");
    }
}
