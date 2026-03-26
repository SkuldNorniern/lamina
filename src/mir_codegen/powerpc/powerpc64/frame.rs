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
        let frame_size = Self::aligned_frame_size(local_bytes);
        writeln!(writer, "    mflr 0")?;
        writeln!(writer, "    std 0, 16(1)")?;        // Save LR to LR save slot
        writeln!(writer, "    stdu 1, -{}(1)", frame_size)?; // Allocate frame + store back-chain
        Ok(())
    }

    /// Emit the function epilogue and return.
    pub fn generate_epilogue<W: Write>(writer: &mut W, local_bytes: usize) -> io::Result<()> {
        let frame_size = Self::aligned_frame_size(local_bytes);
        writeln!(writer, "    addi 1, 1, {}", frame_size)?; // Deallocate frame
        writeln!(writer, "    ld 0, 16(1)")?;               // Restore LR
        writeln!(writer, "    mtlr 0")?;
        writeln!(writer, "    blr")?;
        Ok(())
    }

    /// Deallocate the frame and restore LR, then the caller must branch to the tail target
    /// (no `blr` — that would return to this function instead of tail-calling).
    pub fn generate_tail_epilogue<W: Write>(writer: &mut W, local_bytes: usize) -> io::Result<()> {
        let frame_size = Self::aligned_frame_size(local_bytes);
        writeln!(writer, "    addi 1, 1, {}", frame_size)?;
        writeln!(writer, "    ld 0, 16(1)")?;
        writeln!(writer, "    mtlr 0")?;
        Ok(())
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
