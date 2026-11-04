#![allow(clippy::expect_used)]

use lamina::api::{Target, compile_module_to_object};
use lamina::parser::parse_module;
use std::str::FromStr;

const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_ALIGN_4BYTES: u32 = 0x0030_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;

const FRAME_IR: &str = r#"
fn @callee(i64 %a) -> i64 {
entry:
    %r = add.i64 %a, 1
    ret.i64 %r
}

fn @main() -> i64 {
entry:
    %slot = stackalloc.i64
    store.i64 %slot, 41
    %v = load.i64 %slot
    %x = call @callee(%v)
    ret.i64 %x
}
"#;

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// The offset of the section header named `name`.
fn section_header(object: &[u8], name: &str) -> usize {
    let count = usize::from(u16_at(object, 2));
    (0..count)
        .map(|i| 20 + i * 40)
        .find(|&at| {
            let raw = &object[at..at + 8];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
            raw[..end] == *name.as_bytes()
        })
        .unwrap_or_else(|| panic!("{name} section in COFF object"))
}

fn windows_object(ir: &str) -> Vec<u8> {
    let target = Target::from_str("x86_64_windows").expect("target string parses");
    let module = parse_module(ir).expect("module parses");
    compile_module_to_object(&module, target).expect("COFF object")
}

#[test]
fn unwind_tables_are_read_only_data_as_cl_writes_them() {
    let object = windows_object(FRAME_IR);
    // link.exe warns of any .pdata or .xdata whose attributes differ from the CRT's.
    for name in [".pdata", ".xdata"] {
        let header = section_header(&object, name);
        assert_eq!(
            u32_at(&object, header + 36),
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_MEM_READ,
            "{name}"
        );
    }
}
