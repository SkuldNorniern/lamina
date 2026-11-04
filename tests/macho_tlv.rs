#![allow(clippy::expect_used)]

#[path = "common/macho.rs"]
mod macho;

use lamina::api::{Target, compile_module_to_object};
use lamina::parser::parse_module;
use macho::{MachO, Section};
use std::str::FromStr;

const TLS_IR: &str = r#"
@export @thread_local global $counter: i64 = 7
@thread_local global $hidden: i64 = 0

fn @main() -> i64 {
entry:
    %a = load.i64 $counter
    %b = load.i64 $hidden
    %s = add.i64 %a, %b
    ret.i64 %s
}
"#;

const ARM64_RELOC_UNSIGNED: u32 = 0;
const S_THREAD_LOCAL_REGULAR: u32 = 0x11;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_THREAD_LOCAL_VARIABLES: u32 = 0x13;

fn object() -> Vec<u8> {
    let target = Target::from_str("aarch64_macos").expect("target string parses");
    let module = parse_module(TLS_IR).expect("module parses");
    compile_module_to_object(&module, target).expect("object bytes")
}

fn word_at(section: &Section, offset: u64) -> u64 {
    let at = offset as usize;
    u64::from_le_bytes(
        section.data[at..at + 8]
            .try_into()
            .expect("eight bytes of descriptor"),
    )
}

fn pointer_reloc<'a>(object: &'a MachO, section: &Section, offset: u64) -> (&'a str, bool) {
    let reloc = section
        .relocs
        .iter()
        .find(|r| u64::from(r.address) == offset)
        .unwrap_or_else(|| panic!("no relocation at offset {offset} of {}", section.name));
    assert_eq!(reloc.kind, ARM64_RELOC_UNSIGNED, "pointer relocation");
    assert_eq!(reloc.length, 3, "eight-byte relocation");
    assert!(!reloc.pcrel, "pointer relocation is absolute");
    (object.symbol_name(reloc.symbolnum), reloc.external)
}

/// The descriptor's first word is read and called by every reference to the variable, so
/// it must be a relocation against the dyld thunk and never a stored value.
#[test]
fn a_thread_local_descriptor_starts_with_a_relocation_to_tlv_bootstrap() {
    let object = MachO::parse(&object());
    let vars = object.section("__DATA", "__thread_vars");
    assert_eq!(vars.flags & 0xff, S_THREAD_LOCAL_VARIABLES);
    for name in ["_counter", "_hidden"] {
        let symbol = object.symbol(name);
        let base = symbol.n_value - vars.addr;
        let (target, external) = pointer_reloc(&object, vars, base);
        assert_eq!(target, "__tlv_bootstrap", "{name} thunk word");
        assert!(external, "__tlv_bootstrap is undefined here");
        assert_eq!(word_at(vars, base), 0, "{name} thunk word holds no value");
        assert_eq!(word_at(vars, base + 8), 0, "{name} key word is zero");
    }
}

#[test]
fn the_exported_symbol_is_the_descriptor_and_the_storage_is_file_local() {
    let object = MachO::parse(&object());
    let exported = object.symbol("_counter");
    assert!(exported.external(), "_counter is exported");
    assert!(exported.defined_in_section(), "_counter is defined here");
    let section = object.section_of(exported);
    assert_eq!(
        (section.segment.as_str(), section.name.as_str()),
        ("__DATA", "__thread_vars"),
        "_counter addresses the descriptor"
    );

    let storage = object.symbol("_counter$tlv$init");
    assert!(!storage.external(), "the storage is never exported");
    let section = object.section_of(storage);
    assert_eq!(section.flags & 0xff, S_THREAD_LOCAL_REGULAR);
    assert_eq!(
        word_at(section, storage.n_value - section.addr),
        7,
        "the storage holds the initial value"
    );
}

#[test]
fn a_descriptor_points_at_its_own_storage() {
    let object = MachO::parse(&object());
    let vars = object.section("__DATA", "__thread_vars");
    for (name, storage) in [
        ("_counter", "_counter$tlv$init"),
        ("_hidden", "_hidden$tlv$init"),
    ] {
        let base = object.symbol(name).n_value - vars.addr;
        let (target, _) = pointer_reloc(&object, vars, base + 16);
        assert_eq!(target, storage, "{name} descriptor names its storage");
    }
}

/// An unexported thread local still needs its descriptor, because the reference sequence
/// is the same; only the symbol's linkage differs.
#[test]
fn an_unexported_thread_local_keeps_a_file_local_descriptor() {
    let object = MachO::parse(&object());
    let hidden = object.symbol("_hidden");
    assert!(!hidden.external(), "_hidden is not exported");
    assert_eq!(object.section_of(hidden).name, "__thread_vars");
    let storage = object.symbol("_hidden$tlv$init");
    assert!(!storage.external());
    assert_eq!(
        object.section_of(storage).flags & 0xff,
        S_THREAD_LOCAL_ZEROFILL,
        "a zero initializer goes to __thread_bss"
    );
}
