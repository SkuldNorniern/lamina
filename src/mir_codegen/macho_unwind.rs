//! Mach-O `__LD,__compact_unwind` entries.
//!
//! Darwin's unwinder looks a return address up in `__unwind_info`, which the linker builds
//! from the 32-byte entries in this section: function address, length, a packed encoding
//! of the frame shape, personality and LSDA. The encoding has room for one of a handful of
//! shapes and nothing else, so a frame it cannot name exactly gets no entry at all. An
//! absent entry stops an unwind; a wrong one resumes at a wrong address.

use std::io::{self, Write};

use lamina_platform::TargetOperatingSystem;

use lamina_codegen::os_uses_macho;

const UNWIND_X86_64_MODE_RBP_FRAME: u32 = 0x0100_0000;
const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;

/// The five slots the x86_64 frame encoding has for callee-saved registers.
const X86_64_FRAME_REGISTERS: usize = 5;

/// `UNWIND_X86_64_RBP_FRAME_OFFSET` is eight bits of word count below `rbp`.
const X86_64_FRAME_OFFSET_MAX: usize = 0xff;

/// The frame record `stp x29, x30, [sp, #-16]!` the arm64 frame encoding assumes.
const AARCH64_FRAME_RECORD_BYTES: i32 = 16;

/// The register numbers the x86_64 frame encoding can name. `rbp` is implied by the mode
/// and has no slot of its own.
fn x86_64_register_number(name: &str) -> Option<u32> {
    Some(match name {
        "rbx" => 1,
        "r12" => 2,
        "r13" => 3,
        "r14" => 4,
        "r15" => 5,
        _ => return None,
    })
}

/// The compact encoding for an x86_64 frame, or `None` when the shape has no exact one.
///
/// The prologue is `push rbp`, `mov rsp, rbp`, one stack allocation, then a `mov` per
/// callee-saved register. `UNWIND_X86_64_MODE_RBP_FRAME` restores `rsp` from `rbp`, so the
/// allocation needs no description and its size does not matter. What it does constrain is
/// the saved registers: five slots, read upwards from `rbp - 8 * offset`, each naming one
/// of `rbx` and `r12` through `r15`. A function that saves more than five, saves one the
/// encoding cannot name, or puts the lowest slot further than 255 words below `rbp`, gets
/// no entry.
pub fn plan_x86_64(
    target_os: TargetOperatingSystem,
    naked: bool,
    has_frame: bool,
    saved_regs: &[&str],
    total_slots: usize,
) -> Option<u32> {
    if !os_uses_macho(target_os) || naked || !has_frame {
        return None;
    }
    if saved_regs.is_empty() {
        return Some(UNWIND_X86_64_MODE_RBP_FRAME);
    }
    if saved_regs.len() > X86_64_FRAME_REGISTERS {
        return None;
    }
    // Register `index` sits at `rbp - 8 * (total_slots + index + 1)`, so the deepest is the
    // last one, and the encoding reads upwards from there.
    let offset = total_slots + saved_regs.len();
    if offset > X86_64_FRAME_OFFSET_MAX {
        return None;
    }
    let mut registers = 0u32;
    for (slot, name) in saved_regs.iter().rev().enumerate() {
        registers |= x86_64_register_number(name)? << (3 * slot);
    }
    Some(UNWIND_X86_64_MODE_RBP_FRAME | (offset as u32) << 16 | registers)
}

/// The compact encoding for an AArch64 frame, or `None` when the shape has no exact one.
///
/// `UNWIND_ARM64_MODE_FRAME` describes the standard frame record and, optionally, register
/// pairs stored in a fixed order immediately below `x29`: `x19`/`x20` at `x29 - 8` and
/// `x29 - 16`, then `x21`/`x22`, and so on. This backend stores its callee-saved registers
/// at the bottom of the frame instead, ascending, and not in pairs, so a function that
/// saves any of them cannot be described and gets no entry. A grown frame record, which
/// only a Windows variadic function has, is not the shape the encoding assumes either.
pub fn plan_aarch64(
    target_os: TargetOperatingSystem,
    frame_record: i32,
    callee_saved: &[&'static str],
) -> Option<u32> {
    let describable = os_uses_macho(target_os)
        && callee_saved.is_empty()
        && frame_record == AARCH64_FRAME_RECORD_BYTES;
    describable.then_some(UNWIND_ARM64_MODE_FRAME)
}

/// The end of a function's instructions, which the assembler subtracts to get its length.
/// Mach-O keeps an `L`-prefixed local out of the symbol table.
pub fn function_end_label(label: &str) -> String {
    format!("Lcompact_unwind_end_{label}")
}

/// Emit one entry, then return to `.text` for whatever follows.
pub fn emit<W: Write>(writer: &mut W, label: &str, encoding: u32) -> io::Result<()> {
    writeln!(writer, ".section __LD,__compact_unwind,regular,debug")?;
    writeln!(writer, "    .p2align 3")?;
    writeln!(writer, "    .quad {label}")?;
    writeln!(writer, "    .long {}-{label}", function_end_label(label))?;
    writeln!(writer, "    .long {encoding}")?;
    writeln!(writer, "    .quad 0")?;
    writeln!(writer, "    .quad 0")?;
    writeln!(writer, ".text")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MACOS: TargetOperatingSystem = TargetOperatingSystem::MacOS;

    #[test]
    fn a_frameless_or_naked_x86_64_function_gets_no_entry() {
        assert_eq!(plan_x86_64(MACOS, false, false, &[], 0), None);
        assert_eq!(plan_x86_64(MACOS, true, true, &[], 0), None);
    }

    #[test]
    fn no_entry_is_planned_off_mach_o() {
        for os in [
            TargetOperatingSystem::Linux,
            TargetOperatingSystem::Windows,
            TargetOperatingSystem::FreeBSD,
        ] {
            assert_eq!(plan_x86_64(os, false, true, &[], 0), None);
            assert_eq!(plan_aarch64(os, 16, &[]), None);
        }
    }

    #[test]
    fn a_plain_x86_64_frame_is_the_bare_mode() {
        assert_eq!(
            plan_x86_64(MACOS, false, true, &[], 9),
            Some(UNWIND_X86_64_MODE_RBP_FRAME)
        );
    }

    #[test]
    fn x86_64_saved_registers_read_upwards_from_the_deepest_slot() {
        // rbx at rbp-24, r12 at rbp-32: the deepest is four words down, and the slots run
        // r12 then rbx from there.
        let encoding = plan_x86_64(MACOS, false, true, &["rbx", "r12"], 2).expect("described");
        assert_eq!(encoding & 0x0f00_0000, UNWIND_X86_64_MODE_RBP_FRAME);
        assert_eq!(encoding >> 16 & 0xff, 4, "frame offset in words");
        assert_eq!(encoding & 0x7fff, 2 | 1 << 3, "r12 then rbx");
    }

    #[test]
    fn an_x86_64_register_the_encoding_cannot_name_gets_no_entry() {
        assert_eq!(plan_x86_64(MACOS, false, true, &["rsi"], 0), None);
    }

    #[test]
    fn more_than_five_x86_64_saved_registers_gets_no_entry() {
        let saved = ["rbx", "r12", "r13", "r14", "r15", "rbx"];
        assert_eq!(plan_x86_64(MACOS, false, true, &saved, 0), None);
        assert!(plan_x86_64(MACOS, false, true, &saved[..5], 0).is_some());
    }

    #[test]
    fn an_x86_64_slot_past_the_offset_field_gets_no_entry() {
        assert_eq!(
            plan_x86_64(MACOS, false, true, &["rbx"], X86_64_FRAME_OFFSET_MAX),
            None
        );
        assert!(plan_x86_64(MACOS, false, true, &["rbx"], X86_64_FRAME_OFFSET_MAX - 1).is_some());
    }

    #[test]
    fn an_aarch64_frame_record_alone_is_describable() {
        assert_eq!(plan_aarch64(MACOS, 16, &[]), Some(UNWIND_ARM64_MODE_FRAME));
    }

    #[test]
    fn aarch64_callee_saved_registers_or_a_grown_record_get_no_entry() {
        assert_eq!(plan_aarch64(MACOS, 16, &["x19"]), None);
        assert_eq!(plan_aarch64(MACOS, 80, &[]), None);
    }

    #[test]
    fn an_entry_is_thirty_two_bytes_of_address_length_encoding_and_two_nulls() {
        let mut out = Vec::new();
        emit(&mut out, "_f", UNWIND_ARM64_MODE_FRAME).expect("emit");
        let asm = String::from_utf8(out).expect("utf8");
        assert!(asm.contains("\n    .quad _f\n"), "{asm}");
        assert!(
            asm.contains("\n    .long Lcompact_unwind_end__f-_f\n    .long 67108864\n"),
            "{asm}"
        );
        assert!(asm.ends_with("    .quad 0\n    .quad 0\n.text\n"), "{asm}");
    }
}
