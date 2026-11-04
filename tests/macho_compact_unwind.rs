#![allow(clippy::expect_used)]

#[path = "common/macho.rs"]
mod macho;

use lamina::api::{ObjectBuild, Target, compile_module_to_object, compile_module_to_object_with};
use lamina::parser::parse_module;
use macho::MachO;
use std::str::FromStr;

const ENTRY_BYTES: usize = 32;
const UNWIND_X86_64_MODE_RBP_FRAME: u32 = 0x0100_0000;
const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;
const S_ATTR_DEBUG: u32 = 0x0200_0000;
const RELOC_UNSIGNED: u32 = 0;

const CALLS_IR: &str = r#"
fn @callee(i64 %a) -> i64 {
entry:
    %r = add.i64 %a, 1
    ret.i64 %r
}

fn @main() -> i64 {
entry:
    %x = call @callee(41)
    ret.i64 %x
}
"#;

fn object(target: &str, ir: &str) -> Vec<u8> {
    let target = Target::from_str(target).expect("target string parses");
    let module = parse_module(ir).expect("module parses");
    compile_module_to_object(&module, target).expect("object bytes")
}

/// Register pressure, which is what puts a callee-saved register in the frame at all,
/// only arises once the global allocator runs.
fn optimized_object(target: &str, ir: &str) -> Vec<u8> {
    let target = Target::from_str(target).expect("target string parses");
    let module = parse_module(ir).expect("module parses");
    let mut build = ObjectBuild::default();
    build.opt_level = 2;
    compile_module_to_object_with(&module, target, &build).expect("object bytes")
}

const FAR_FRAME_IR: &str = r#"
fn @farframe(i64 %seed) -> i64 {
entry:
  %buf = stackalloc.[4096 x i64]
  %i = stackalloc.i64
  store.i64 %i, 0
  jmp fill

fill:
  %iv = load.i64 %i
  %go = lt.i64 %iv, 4096
  br %go, fill_body, done

fill_body:
  %slot = getelem.i64 %buf, %iv
  %v1 = add.i64 %iv, %seed
  %v2 = mul.i64 %v1, 3
  %v3 = xor.i64 %v2, %v1
  %v4 = mul.i64 %v3, 5
  %v5 = xor.i64 %v4, %v2
  %v6 = mul.i64 %v5, 7
  %v7 = xor.i64 %v6, %v3
  %v8 = add.i64 %v7, %v4
  %v9 = xor.i64 %v8, %v5
  %v10 = add.i64 %v9, %v6
  store.i64 %slot, %v10
  %in = add.i64 %iv, 1
  store.i64 %i, %in
  jmp fill

done:
  %last = getelem.i64 %buf, 4095
  %out = load.i64 %last
  ret.i64 %out
}
"#;

struct Entry {
    function: String,
    length: u32,
    encoding: u32,
}

fn entries(object: &MachO) -> Vec<Entry> {
    let Some(section) = object.find_section("__LD", "__compact_unwind") else {
        return Vec::new();
    };
    assert_eq!(
        section.flags & S_ATTR_DEBUG,
        S_ATTR_DEBUG,
        "the section is marked debug so the linker consumes it rather than copying it"
    );
    assert_eq!(section.align, 3, "entries are eight-byte aligned");
    assert_eq!(
        section.size as usize % ENTRY_BYTES,
        0,
        "the section is a whole number of entries"
    );
    let count = section.size as usize / ENTRY_BYTES;
    (0..count)
        .map(|index| {
            let at = index * ENTRY_BYTES;
            let reloc = section
                .relocs
                .iter()
                .find(|r| r.address as usize == at)
                .expect("the function address is a relocation");
            assert_eq!(reloc.kind, RELOC_UNSIGNED);
            assert_eq!(reloc.length, 3, "an eight-byte pointer");
            assert!(!reloc.pcrel);
            for field in [at + 16, at + 24] {
                assert!(
                    !section.relocs.iter().any(|r| r.address as usize == field),
                    "personality and LSDA are null, not relocated"
                );
                assert_eq!(
                    u64::from_le_bytes(
                        section.data[field..field + 8]
                            .try_into()
                            .expect("eight bytes")
                    ),
                    0
                );
            }
            Entry {
                function: object.symbol_name(reloc.symbolnum).to_string(),
                length: u32::from_le_bytes(
                    section.data[at + 8..at + 12]
                        .try_into()
                        .expect("four bytes"),
                ),
                encoding: u32::from_le_bytes(
                    section.data[at + 12..at + 16]
                        .try_into()
                        .expect("four bytes"),
                ),
            }
        })
        .collect()
}

/// A length of zero would make the linker's range for the function empty, so every entry
/// has to carry the measured distance to the end of its instructions.
fn assert_lengths_cover_the_text(object: &MachO, entries: &[Entry]) {
    let text = object.section("__TEXT", "__text");
    for entry in entries {
        let symbol = object.symbol(&entry.function);
        let start = symbol.n_value - text.addr;
        assert!(entry.length > 0, "{} has a length", entry.function);
        assert!(
            start + u64::from(entry.length) <= text.size,
            "{} runs past __text",
            entry.function
        );
    }
}

#[test]
fn x86_64_frames_are_described_as_rbp_frames() {
    let object = MachO::parse(&object("x86_64_macos", CALLS_IR));
    let entries = entries(&object);
    assert_lengths_cover_the_text(&object, &entries);
    let main = entries
        .iter()
        .find(|e| e.function == "_main")
        .expect("_main has a frame and an entry");
    assert_eq!(
        main.encoding & 0x0f00_0000,
        UNWIND_X86_64_MODE_RBP_FRAME,
        "mode"
    );
    // `_callee` is a leaf with no frame, so it is described by nothing at all.
    assert!(
        !entries.iter().any(|e| e.function == "_callee"),
        "a frameless function gets no entry"
    );
}

#[test]
fn an_x86_64_saved_register_is_named_at_its_own_slot() {
    let object = MachO::parse(&object("x86_64_macos", CALLS_IR));
    let main = entries(&object)
        .into_iter()
        .find(|e| e.function == "_main")
        .expect("_main entry");
    let offset = main.encoding >> 16 & 0xff;
    let registers = main.encoding & 0x7fff;
    assert_eq!(
        (registers != 0),
        (offset != 0),
        "a named register needs a frame offset and the other way round"
    );
    if registers != 0 {
        assert_eq!(registers & 0x7, 1, "the deepest slot holds rbx");
        assert_eq!(offset, 1, "rbx sits one word below rbp");
    }
}

#[test]
fn aarch64_frames_are_described_as_standard_frame_records() {
    let object = MachO::parse(&object("aarch64_macos", CALLS_IR));
    for entry in entries(&object) {
        assert_eq!(
            entry.encoding, UNWIND_ARM64_MODE_FRAME,
            "{} is a bare frame record",
            entry.function
        );
    }
}

/// The frame size itself never disqualifies an `rbp` frame: the mode restores `rsp` from
/// `rbp`, so the allocation needs no description.
#[test]
fn a_deep_frame_with_nothing_saved_is_still_described() {
    const DEEP: &str = r#"
fn @deep(i64 %seed) -> i64 {
entry:
    %buf = stackalloc.[4096 x i64]
    %slot = getelem.i64 %buf, 7
    store.i64 %slot, %seed
    %v = load.i64 %slot
    ret.i64 %v
}
"#;
    let object = MachO::parse(&object("x86_64_macos", DEEP));
    let deep = entries(&object)
        .into_iter()
        .find(|e| e.function == "_deep")
        .expect("_deep entry");
    assert_eq!(deep.encoding, UNWIND_X86_64_MODE_RBP_FRAME, "nothing saved");
}

/// A saved register further below `rbp` than the eight-bit offset field reaches cannot be
/// named, so the function is left out rather than described approximately.
#[test]
fn a_saved_register_out_of_reach_of_the_offset_field_gets_no_entry() {
    let object = MachO::parse(&optimized_object("x86_64_macos", FAR_FRAME_IR));
    assert!(
        !entries(&object).iter().any(|e| e.function == "_farframe"),
        "the deep frame with saved registers is left out"
    );
}

#[test]
fn no_compact_unwind_section_is_emitted_off_mach_o() {
    for target in ["x86_64_linux", "aarch64_linux", "x86_64_windows"] {
        let object = object(target, CALLS_IR);
        assert!(
            !object.windows(16).any(|w| w == b"__compact_unwind"),
            "{target} carries no compact unwind section"
        );
    }
}
