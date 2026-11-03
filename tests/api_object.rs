#![allow(clippy::expect_used)]

use lamina::api::{ObjectError, Target, compile_module_to_object};
use lamina::parser::parse_module;
use lamina::target::TargetArchitecture;
use std::str::{self, FromStr};

const MAIN_IR: &str = r#"
fn @main() -> i64 {
entry:
    %result = add.i64 42, 8
    ret.i64 %result
}
"#;

fn compile(target: &str) -> Result<Vec<u8>, ObjectError> {
    let target = Target::from_str(target).expect("target string parses");
    let module = parse_module(MAIN_IR).expect("module parses");
    compile_module_to_object(&module, target)
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

fn cstr_at(bytes: &[u8], at: usize) -> &str {
    let end = bytes[at..]
        .iter()
        .position(|&b| b == 0)
        .expect("nul terminated");
    str::from_utf8(&bytes[at..at + end]).expect("utf-8 symbol name")
}

const ELF_EM_X86_64: u16 = 62;
const ELF_EM_AARCH64: u16 = 183;
const ELF_EM_RISCV: u16 = 243;

fn assert_elf_symbol(object: &[u8], machine: u16, name: &str) {
    assert_eq!(&object[..4], b"\x7fELF", "ELF magic");
    assert_eq!(object[4], 2, "ELFCLASS64");
    assert_eq!(object[5], 1, "ELFDATA2LSB");
    assert_eq!(u16_at(object, 16), 1, "e_type is ET_REL");
    assert_eq!(u16_at(object, 18), machine, "e_machine");

    let shoff = u64_at(object, 0x28) as usize;
    let shentsize = u16_at(object, 0x3A) as usize;
    let shnum = u16_at(object, 0x3C) as usize;
    let section = |index: usize| shoff + index * shentsize;

    let symtab = (0..shnum)
        .find(|&i| u32_at(object, section(i) + 4) == 2)
        .expect("ELF object has a .symtab");
    let sym_off = u64_at(object, section(symtab) + 0x18) as usize;
    let sym_size = u64_at(object, section(symtab) + 0x20) as usize;
    let strtab = u32_at(object, section(symtab) + 0x28) as usize;
    let str_off = u64_at(object, section(strtab) + 0x18) as usize;

    let found = (0..sym_size / 24).any(|i| {
        let entry = sym_off + i * 24;
        let name_at = str_off + u32_at(object, entry) as usize;
        let shndx = u16_at(object, entry + 6);
        cstr_at(object, name_at) == name && shndx != 0
    });
    assert!(found, "defined symbol {name} in ELF .symtab");
}

fn assert_macho_symbol(object: &[u8], cpu_type: u32, name: &str) {
    assert_eq!(u32_at(object, 0), 0xFEED_FACF, "MH_MAGIC_64");
    assert_eq!(u32_at(object, 4), cpu_type, "cputype");
    assert_eq!(u32_at(object, 12), 1, "MH_OBJECT filetype");

    let ncmds = u32_at(object, 16) as usize;
    let mut at = 32;
    let mut found = false;
    for _ in 0..ncmds {
        let cmd = u32_at(object, at);
        let cmdsize = u32_at(object, at + 4) as usize;
        if cmd == 0x2 {
            let symoff = u32_at(object, at + 8) as usize;
            let nsyms = u32_at(object, at + 12) as usize;
            let stroff = u32_at(object, at + 16) as usize;
            found = (0..nsyms).any(|i| {
                let entry = symoff + i * 16;
                let name_at = stroff + u32_at(object, entry) as usize;
                let n_sect = object[entry + 5];
                cstr_at(object, name_at) == name && n_sect != 0
            });
        }
        at += cmdsize;
    }
    assert!(found, "defined symbol {name} in Mach-O symtab");
}

fn assert_coff_symbol(object: &[u8], machine: u16, name: &str) {
    assert_eq!(u16_at(object, 0), machine, "COFF machine");

    let sym_off = u32_at(object, 8) as usize;
    let sym_count = u32_at(object, 12) as usize;
    let str_off = sym_off + sym_count * 18;

    let found = (0..sym_count).any(|i| {
        let entry = sym_off + i * 18;
        let section_number = u16_at(object, entry + 12) as i16;
        let symbol_name = if u32_at(object, entry) == 0 {
            cstr_at(object, str_off + u32_at(object, entry + 4) as usize)
        } else {
            cstr_at(object, entry)
        };
        symbol_name == name && section_number > 0
    });
    assert!(found, "defined symbol {name} in COFF symbol table");
}

#[test]
fn x86_64_linux_yields_elf_relocatable_with_main() {
    let object = compile("x86_64_linux").expect("object bytes");
    assert_elf_symbol(&object, ELF_EM_X86_64, "main");
}

#[test]
fn aarch64_linux_yields_elf_relocatable_with_main() {
    let object = compile("aarch64_linux").expect("object bytes");
    assert_elf_symbol(&object, ELF_EM_AARCH64, "main");
}

#[test]
fn riscv64_linux_yields_elf_relocatable_with_main() {
    let object = compile("riscv64_linux").expect("object bytes");
    assert_elf_symbol(&object, ELF_EM_RISCV, "main");
}

#[test]
fn x86_64_macos_yields_macho_relocatable_with_main() {
    let object = compile("x86_64_macos").expect("object bytes");
    assert_macho_symbol(&object, 0x0100_0007, "_main");
}

#[test]
fn aarch64_macos_yields_macho_relocatable_with_main() {
    let object = compile("aarch64_macos").expect("object bytes");
    assert_macho_symbol(&object, 0x0100_000C, "_main");
}

#[test]
fn x86_64_windows_yields_coff_with_main() {
    let object = compile("x86_64_windows").expect("object bytes");
    assert_coff_symbol(&object, 0x8664, "main");
}

#[test]
fn target_without_object_writer_is_refused_with_typed_error() {
    let error = compile("aarch64_windows").expect_err("COFF writer is x86_64 only");
    assert!(
        matches!(error, ObjectError::UnsupportedTarget(target) if target.architecture == TargetArchitecture::Aarch64),
        "got {error:?}"
    );
}
