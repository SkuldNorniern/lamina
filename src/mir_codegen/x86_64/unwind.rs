//! Windows x64 `.pdata` and `.xdata` unwind tables.
//!
//! Windows does not walk frame pointers: `RtlLookupFunctionEntry` finds a `RUNTIME_FUNCTION`
//! in `.pdata` and the `UNWIND_INFO` it points at in `.xdata` replays the prologue backwards.
//! A function with no entry is read as a leaf, so leaving a function out is inert while a
//! record that does not match its prologue sends the unwinder to a wrong address.

use std::io::{self, Write};

use lamina_codegen::x86_64::X86Frame;
use lamina_platform::TargetOperatingSystem;

const UWOP_PUSH_NONVOL: u8 = 0;
const UWOP_ALLOC_LARGE: u8 = 1;
const UWOP_ALLOC_SMALL: u8 = 2;
const UWOP_SET_FPREG: u8 = 3;
const UWOP_SAVE_NONVOL: u8 = 4;

const REG_RBP: u8 = 5;

/// Read-only initialized data, as cl and clang mark both tables. COFF letters read
/// `d` and `r` and ignore `a`; an assembler that reads ELF letters needs the `a` to
/// keep the section allocated.
const TABLE_FLAGS: &str = "adr";

/// `pushq %rbp` is one byte and `movq %rsp, %rbp` is three.
const PUSH_RBP_END: u8 = 1;
const MOVE_RBP_END: u8 = 4;

const ALLOC_SMALL_MAX: usize = 128;
const ALLOC_LARGE_SCALED_MAX: usize = 512 * 1024 - 8;

/// `UNWIND_INFO` holds the frame offset in four bits, scaled by sixteen.
const FRAME_OFFSET_MAX: usize = 240;

/// The register numbers `UNWIND_CODE` operation info uses.
fn register_number(name: &str) -> Option<u8> {
    Some(match name {
        "rax" => 0,
        "rcx" => 1,
        "rdx" => 2,
        "rbx" => 3,
        "rsp" => 4,
        "rbp" => 5,
        "rsi" => 6,
        "rdi" => 7,
        "r8" => 8,
        "r9" => 9,
        "r10" => 10,
        "r11" => 11,
        "r12" => 12,
        "r13" => 13,
        "r14" => 14,
        "r15" => 15,
        _ => return None,
    })
}

/// How the stack allocation at the end of the prologue is described.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Alloc {
    None,
    Small(u8),
    Scaled(u16),
    Wide(u32),
}

impl Alloc {
    fn of(frame: usize) -> Option<Self> {
        if frame == 0 {
            return Some(Self::None);
        }
        if !frame.is_multiple_of(8) {
            return None;
        }
        if frame <= ALLOC_SMALL_MAX {
            return Some(Self::Small((frame / 8 - 1) as u8));
        }
        if frame <= ALLOC_LARGE_SCALED_MAX {
            return Some(Self::Scaled((frame / 8) as u16));
        }
        u32::try_from(frame).ok().map(Self::Wide)
    }

    fn slots(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Small(_) => 1,
            Self::Scaled(_) => 2,
            Self::Wide(_) => 3,
        }
    }
}

/// A nonvolatile register stored into the frame with a `mov`, and where it landed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Save {
    register: u8,
    /// Distance above the frame base, in eight-byte words.
    words: u16,
}

/// A function's unwind record, or the decision to leave it out of the table.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Unwind {
    /// No record. Windows reads a missing entry as a leaf frame.
    Omitted,
    Described {
        alloc: Alloc,
        /// Scaled frame offset: the frame base sits this many sixteens below `rbp`.
        frame_offset: u8,
        saves: Vec<Save>,
    },
}

impl Unwind {
    pub(super) fn is_described(&self) -> bool {
        matches!(self, Self::Described { .. })
    }

    /// Whether the record names the register stores that follow the allocation, so the
    /// caller knows to label them.
    pub(super) fn describes_saves(&self) -> bool {
        matches!(self, Self::Described { saves, .. } if !saves.is_empty())
    }
}

/// Decide how to describe a frame, or that it cannot be described exactly.
///
/// The prologue is `push rbp`, `mov rsp, rbp`, one stack allocation, then a `mov` per
/// callee-saved register into a slot below `rbp`. The page-loop probe leaves `rsp` exactly
/// one frame below its entry value, so it needs no shape of its own; only the prologue's
/// length differs, and the assembler measures that.
///
/// Two descriptions are exact, and which one applies turns on the frame size:
///
/// * A frame of at most 240 bytes puts the frame base at `rbp - frame`, which is `rsp`
///   once the allocation is done. `UWOP_SAVE_NONVOL` offsets run upwards from there, so
///   every register store is named.
/// * A larger frame cannot hold its size in the four-bit frame offset, so the frame base
///   is `rbp` itself and the stores, which sit below `rbp`, have no unsigned offset. Such
///   a function is described only when it stores nothing, and left out otherwise.
pub(super) fn plan(
    target_os: TargetOperatingSystem,
    naked: bool,
    has_frame: bool,
    saved_regs: &[&str],
    total_slots: usize,
    stack_size: usize,
) -> Unwind {
    if target_os != TargetOperatingSystem::Windows || naked || !has_frame {
        return Unwind::Omitted;
    }
    let frame = X86Frame::aligned_frame_size(stack_size);
    if frame > i32::MAX as usize {
        return Unwind::Omitted;
    }
    let Some(alloc) = Alloc::of(frame) else {
        return Unwind::Omitted;
    };
    if saved_regs.is_empty() {
        return Unwind::Described {
            alloc,
            frame_offset: 0,
            saves: Vec::new(),
        };
    }
    if frame > FRAME_OFFSET_MAX {
        return Unwind::Omitted;
    }
    let mut saves = Vec::with_capacity(saved_regs.len());
    for (index, name) in saved_regs.iter().enumerate() {
        let Some(register) = register_number(name) else {
            return Unwind::Omitted;
        };
        // The slot is at -(8 * (total_slots + index + 1)) from rbp, and the frame base is
        // at -frame, so the distance above the frame base is the difference.
        let below = 8 * (total_slots + index + 1);
        let Some(above) = frame.checked_sub(below) else {
            return Unwind::Omitted;
        };
        saves.push(Save {
            register,
            words: (above / 8) as u16,
        });
    }
    Unwind::Described {
        alloc,
        frame_offset: (frame / 16) as u8,
        saves,
    }
}

pub(super) fn alloc_end_label(label: &str) -> String {
    format!(".Lxunwind_alloc_{label}")
}

pub(super) fn save_label(label: &str, index: usize) -> String {
    format!(".Lxunwind_save{index}_{label}")
}

pub(super) fn prologue_end_label(label: &str) -> String {
    format!(".Lxunwind_prologue_{label}")
}

pub(super) fn function_end_label(label: &str) -> String {
    format!(".Lxunwind_end_{label}")
}

fn info_label(label: &str) -> String {
    format!(".Lxunwind_info_{label}")
}

/// Emit a function's `UNWIND_INFO` and its `RUNTIME_FUNCTION`, then return to `.text`.
///
/// `.pdata` is left unsorted here; the linker orders the table.
pub(super) fn emit<W: Write>(writer: &mut W, label: &str, unwind: &Unwind) -> io::Result<()> {
    let Unwind::Described {
        alloc,
        frame_offset,
        saves,
    } = unwind
    else {
        return Ok(());
    };
    let alloc_end = alloc_end_label(label);
    let info = info_label(label);
    let codes = alloc.slots() + 2 + 2 * saves.len() as u8;
    // A frame base below rbp is only reachable once the allocation has run, so the frame
    // register is declared established there rather than at the `mov`.
    let set_fpreg_at = if *frame_offset == 0 {
        format!("{MOVE_RBP_END}")
    } else {
        format!("{alloc_end}-{label}")
    };

    writeln!(writer, ".section .xdata,\"{TABLE_FLAGS}\"")?;
    writeln!(writer, "    .p2align 2")?;
    writeln!(writer, "{info}:")?;
    writeln!(writer, "    .byte 1")?;
    writeln!(writer, "    .byte {}-{label}", prologue_end_label(label))?;
    writeln!(writer, "    .byte {codes}")?;
    writeln!(writer, "    .byte {}", (frame_offset << 4) | REG_RBP)?;
    for (index, save) in saves.iter().enumerate().rev() {
        writeln!(writer, "    .byte {}-{label}", save_label(label, index))?;
        writeln!(
            writer,
            "    .byte {}",
            (save.register << 4) | UWOP_SAVE_NONVOL
        )?;
        writeln!(writer, "    .short {}", save.words)?;
    }
    let set_fpreg = |writer: &mut W| -> io::Result<()> {
        writeln!(writer, "    .byte {set_fpreg_at}")?;
        writeln!(writer, "    .byte {UWOP_SET_FPREG}")
    };
    // The unwinder replays these in array order. A frame base below rbp has to be
    // restored before the allocation is given back, or the allocation is applied twice;
    // a frame base at rbp is reached by giving the allocation back first.
    let base_below_rbp = *frame_offset != 0;
    if base_below_rbp {
        set_fpreg(writer)?;
    }
    match alloc {
        Alloc::None => {}
        Alloc::Small(words) => {
            writeln!(writer, "    .byte {alloc_end}-{label}")?;
            writeln!(writer, "    .byte {}", (words << 4) | UWOP_ALLOC_SMALL)?;
        }
        Alloc::Scaled(words) => {
            writeln!(writer, "    .byte {alloc_end}-{label}")?;
            writeln!(writer, "    .byte {UWOP_ALLOC_LARGE}")?;
            writeln!(writer, "    .short {words}")?;
        }
        Alloc::Wide(bytes) => {
            writeln!(writer, "    .byte {alloc_end}-{label}")?;
            writeln!(writer, "    .byte {}", (1 << 4) | UWOP_ALLOC_LARGE)?;
            writeln!(writer, "    .short {}", bytes & 0xffff)?;
            writeln!(writer, "    .short {}", bytes >> 16)?;
        }
    }
    if !base_below_rbp {
        set_fpreg(writer)?;
    }
    writeln!(writer, "    .byte {PUSH_RBP_END}")?;
    writeln!(writer, "    .byte {}", (REG_RBP << 4) | UWOP_PUSH_NONVOL)?;
    if !codes.is_multiple_of(2) {
        writeln!(writer, "    .short 0")?;
    }

    writeln!(writer, ".section .pdata,\"{TABLE_FLAGS}\"")?;
    writeln!(writer, "    .p2align 2")?;
    writeln!(writer, "    .long {label}")?;
    writeln!(writer, "    .long {}", function_end_label(label))?;
    writeln!(writer, "    .long {info}")?;
    writeln!(writer, ".text")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOWS: TargetOperatingSystem = TargetOperatingSystem::Windows;

    fn frameful(frame: usize) -> Unwind {
        plan(WINDOWS, false, true, &[], 0, frame)
    }

    fn body(unwind: &Unwind) -> String {
        assert!(unwind.is_described(), "{unwind:?} has no record");
        let mut out = Vec::new();
        emit(&mut out, "fn", unwind).expect("emit");
        String::from_utf8(out).expect("utf8")
    }

    #[test]
    fn a_frameless_function_gets_no_record() {
        assert_eq!(plan(WINDOWS, false, false, &[], 0, 0), Unwind::Omitted);
    }

    #[test]
    fn a_naked_function_gets_no_record() {
        assert_eq!(plan(WINDOWS, true, true, &[], 0, 64), Unwind::Omitted);
    }

    #[test]
    fn no_table_is_emitted_off_windows() {
        for os in [
            TargetOperatingSystem::Linux,
            TargetOperatingSystem::MacOS,
            TargetOperatingSystem::FreeBSD,
        ] {
            assert_eq!(plan(os, false, true, &[], 0, 64), Unwind::Omitted);
        }
    }

    #[test]
    fn a_small_frame_scales_by_eight_and_biases_by_one() {
        assert_eq!(Alloc::of(8), Some(Alloc::Small(0)));
        assert_eq!(Alloc::of(64), Some(Alloc::Small(7)));
        assert_eq!(Alloc::of(ALLOC_SMALL_MAX), Some(Alloc::Small(15)));
        // Eight words of frame, so the operation info is seven: (7 << 4) | 2.
        let text = body(&frameful(64));
        assert!(text.contains("    .byte 114\n"), "{text}");
    }

    #[test]
    fn a_large_frame_stores_its_size_in_words() {
        assert_eq!(Alloc::of(ALLOC_SMALL_MAX + 8), Some(Alloc::Scaled(17)));
        assert_eq!(
            Alloc::of(ALLOC_LARGE_SCALED_MAX),
            Some(Alloc::Scaled(65535))
        );
        assert!(body(&frameful(4096)).contains("    .short 512\n"));
    }

    #[test]
    fn a_frame_past_the_scaled_range_stores_its_byte_count() {
        let frame = ALLOC_LARGE_SCALED_MAX + 8;
        assert_eq!(Alloc::of(frame), Some(Alloc::Wide(frame as u32)));
        let text = body(&frameful(frame));
        // (1 << 4) | UWOP_ALLOC_LARGE, then the byte count, low half first.
        assert!(text.contains("    .byte 17\n"), "{text}");
        assert!(text.contains("    .short 0\n"), "{text}");
        assert!(text.contains("    .short 8\n"), "{text}");
    }

    #[test]
    fn a_frame_too_large_to_subtract_gets_no_record() {
        let frame = i32::MAX as usize + 16;
        assert_eq!(plan(WINDOWS, false, true, &[], 0, frame), Unwind::Omitted);
    }

    #[test]
    fn an_empty_frame_sets_the_frame_register_at_the_move() {
        let text = body(&frameful(0));
        assert!(text.contains("    .byte 2\n"), "{text}");
        assert!(text.contains("    .byte 5\n"), "{text}");
        assert!(text.contains("    .byte 4\n    .byte 3\n"), "{text}");
    }

    #[test]
    fn a_frame_without_stores_keeps_the_frame_base_at_rbp() {
        let Unwind::Described { frame_offset, .. } = frameful(64) else {
            panic!("no record");
        };
        assert_eq!(frame_offset, 0);
    }

    #[test]
    fn stores_are_measured_upwards_from_the_frame_base() {
        // One local slot, then rdi and rsi below it, in a 48-byte frame: the frame base is
        // 48 below rbp, so the slots at -16 and -24 sit 32 and 24 bytes above it.
        let unwind = plan(WINDOWS, false, true, &["rdi", "rsi"], 1, 48);
        let Unwind::Described {
            frame_offset,
            saves,
            ..
        } = &unwind
        else {
            panic!("no record");
        };
        assert_eq!(*frame_offset, 3);
        assert_eq!(
            saves,
            &[
                Save {
                    register: 7,
                    words: 4
                },
                Save {
                    register: 6,
                    words: 3
                },
            ]
        );
        let text = body(&unwind);
        // Latest store first, so rsi precedes rdi in the array.
        let rsi = text.find("    .byte 100\n").expect("rsi save");
        let rdi = text.find("    .byte 116\n").expect("rdi save");
        assert!(rsi < rdi, "{text}");
        assert!(text.contains("    .short 3\n"), "{text}");
        assert!(text.contains("    .short 4\n"), "{text}");
        // Frame register rbp with a frame offset of three.
        assert!(text.contains("    .byte 53\n"), "{text}");
        assert!(
            text.contains("    .byte .Lxunwind_alloc_fn-fn\n    .byte 3\n"),
            "{text}"
        );
    }

    #[test]
    fn a_large_frame_with_stores_is_left_out() {
        assert_eq!(
            plan(WINDOWS, false, true, &["rbx"], 2, FRAME_OFFSET_MAX + 16),
            Unwind::Omitted
        );
    }

    #[test]
    fn the_largest_describable_frame_with_stores_is_two_hundred_and_forty() {
        let unwind = plan(WINDOWS, false, true, &["rbx"], 1, FRAME_OFFSET_MAX);
        let Unwind::Described { frame_offset, .. } = unwind else {
            panic!("no record");
        };
        assert_eq!(frame_offset, 15);
    }

    #[test]
    fn the_code_array_is_padded_to_four_bytes() {
        assert!(body(&frameful(64)).contains("    .short 0\n"));
        assert!(!body(&frameful(0)).contains("    .short 0\n"));
        assert!(!body(&frameful(4096)).contains("    .short 0\n"));
    }

    #[test]
    fn the_record_names_the_function_and_its_unwind_info() {
        let text = body(&frameful(64));
        assert!(text.contains(".section .xdata,\"adr\"\n"), "{text}");
        assert!(text.contains(".section .pdata,\"adr\"\n"), "{text}");
        assert!(text.contains("    .long fn\n"), "{text}");
        assert!(text.contains("    .long .Lxunwind_end_fn\n"), "{text}");
        assert!(text.contains("    .long .Lxunwind_info_fn\n"), "{text}");
        assert!(text.ends_with(".text\n"), "{text}");
    }

    #[test]
    fn the_prologue_length_is_left_to_the_assembler() {
        let text = body(&frameful(64));
        assert!(
            text.contains("    .byte .Lxunwind_prologue_fn-fn\n"),
            "{text}"
        );
        assert!(text.contains("    .byte .Lxunwind_alloc_fn-fn\n"), "{text}");
    }

    #[test]
    fn a_frame_base_below_rbp_is_restored_before_the_allocation_is_given_back() {
        // SET_FPREG assigns rsp from the frame register, so with the base below rbp it
        // has to run first: the allocation then brings rsp back up to the saved rbp.
        let text = body(&plan(WINDOWS, false, true, &["rbx"], 1, 48));
        let set_fpreg = text
            .find("    .byte .Lxunwind_alloc_fn-fn\n    .byte 3\n")
            .expect("set_fpreg");
        let alloc = text
            .find("    .byte .Lxunwind_alloc_fn-fn\n    .byte 82\n")
            .expect("alloc");
        assert!(set_fpreg < alloc, "{text}");
    }

    #[test]
    fn a_frame_base_at_rbp_is_reached_after_the_allocation_is_given_back() {
        let text = body(&frameful(64));
        let set_fpreg = text.find("    .byte 4\n    .byte 3\n").expect("set_fpreg");
        let alloc = text
            .find("    .byte .Lxunwind_alloc_fn-fn\n    .byte 114\n")
            .expect("alloc");
        assert!(alloc < set_fpreg, "{text}");
    }

    #[test]
    fn only_a_record_with_stores_asks_for_store_labels() {
        assert!(!frameful(64).describes_saves());
        assert!(plan(WINDOWS, false, true, &["rbx"], 1, 48).describes_saves());
    }
}
