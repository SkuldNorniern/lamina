//! What a Windows debugger is told about a `-g` build.
//!
//! No PDB is written by the compiler: `.debug$S` and `.debug$T` go into the COFF
//! object and the linker gathers them, which is what `link.exe` and `lld-link` are
//! for. The end-to-end assertion here is therefore that `lld-link` builds a PDB from
//! our objects and that the PDB answers the questions a debugger asks of it.
#![allow(clippy::expect_used)]

use lamina::api::{ObjectBuild, Target, compile_module_to_object_with};
use lamina::parser::parse_module;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;
use std::{env, fs};

mod common;
use common::TempDir;

const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
  entry:
    %r = mul.i64 %n, 3
    ret.i64 %r
}

fn @main() -> i64 {
  entry:
    %v = call @triple(7)
    ret.i64 0
}
"#;

fn object(debug: bool) -> Vec<u8> {
    let module = parse_module(SOURCE).expect("parse");
    let target = Target::from_str("x86_64_windows").expect("target");
    let mut build = ObjectBuild::default();
    build.debug_lines = debug;
    compile_module_to_object_with(&module, target, &build).expect("compile")
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn section(object: &[u8], name: &str) -> Option<Vec<u8>> {
    let count = usize::from(u16_at(object, 2));
    let at = (0..count).map(|i| 20 + i * 40).find(|&at| {
        let raw = &object[at..at + 8];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
        raw[..end] == *name.as_bytes()
    })?;
    let size = u32_at(object, at + 16) as usize;
    let start = u32_at(object, at + 20) as usize;
    Some(object[start..start + size].to_vec())
}

#[test]
fn a_build_without_g_carries_no_codeview() {
    let object = object(false);
    assert!(section(&object, ".debug$S").is_none());
    assert!(section(&object, ".debug$T").is_none());
}

#[test]
fn a_g_build_carries_both_codeview_sections() {
    let object = object(true);
    for name in [".debug$S", ".debug$T"] {
        let section = section(&object, name).unwrap_or_else(|| panic!("no {name}"));
        assert_eq!(u32_at(&section, 0), 4, "{name} is not CV_SIGNATURE_C13");
    }
}

/// Mirrors the DWARF side: nothing outside the backends knows where a value lives
/// after register allocation, so no `S_LOCAL` and no `S_DEFRANGE_*` may appear. A
/// future guess trips this before it reaches a debugger.
#[test]
fn no_variable_location_is_stated() {
    let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"]) else {
        return;
    };
    let dir = TempDir::new("codeview-locations");
    let path = dir.path().join("program.obj");
    fs::write(&path, object(true)).expect("write");
    let dump = dump(&pdbutil, &["dump", "--symbols"], &path);
    assert!(dump.contains("S_GPROC32_ID"), "{dump}");
    assert!(!dump.contains("S_LOCAL"), "{dump}");
    assert!(!dump.contains("S_DEFRANGE"), "{dump}");
}

/// The whole point of emitting sections instead of writing a PDB: a linker has to be
/// able to make one out of them. If this stops holding, leaving PDB generation to the
/// linker is no longer a working plan.
#[test]
fn lld_link_builds_a_pdb_that_names_our_functions() {
    let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"]) else {
        return;
    };
    let Some(lld) = tool(&["lld-link", "lld-link-18", "lld-link-19"]) else {
        return;
    };
    let dir = TempDir::new("codeview-pdb");
    let object_path = dir.path().join("program.obj");
    let exe = dir.path().join("program.exe");
    let pdb = dir.path().join("program.pdb");
    fs::write(&object_path, object(true)).expect("write");

    let link = Command::new(lld)
        .args([
            "/nologo",
            "/entry:main",
            "/subsystem:console",
            "/nodefaultlib",
            "/debug",
        ])
        .arg(format!("/pdb:{}", pdb.display()))
        .arg(format!("/out:{}", exe.display()))
        .arg(&object_path)
        .output()
        .expect("lld-link");
    assert!(
        link.status.success(),
        "{}",
        String::from_utf8_lossy(&link.stderr)
    );
    assert!(pdb.exists(), "lld-link produced no PDB");

    let dump = dump(
        &pdbutil,
        &["dump", "--symbols", "--types", "--ids", "-l"],
        &pdb,
    );
    // A procedure that reached the PDB has an address in a section of the image,
    // which only happens if the object's relocations were understood.
    assert!(dump.contains("S_GPROC32 [size = 44] `main`"), "{dump}");
    assert!(dump.contains("`triple`"), "{dump}");
    assert!(dump.contains("LF_FUNC_ID"), "{dump}");
    assert!(!dump.contains("addr = 0000:0000"), "{dump}");
    // The line table survived, mapping image offsets to the lines `triple`'s body is
    // on, which is what a breakpoint is placed by.
    assert!(dump.contains("line/addr entries = 2"), "{dump}");
    assert!(dump.contains("     3 "), "{dump}");
    assert!(!dump.contains("S_DEFRANGE"), "{dump}");
}

fn dump(tool: &str, args: &[&str], path: &Path) -> String {
    let output = Command::new(tool)
        .args(args)
        .arg(path)
        .output()
        .expect("dump");
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{text}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    text
}

/// Skips when the tool is missing, unless the suite was told to require it.
fn tool(names: &[&str]) -> Option<String> {
    let found = names.iter().find(|name| {
        Command::new(name)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    });
    if let Some(name) = found {
        return Some((*name).to_string());
    }
    let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
    assert!(
        !required,
        "none of {names:?} is installed (LAMINA_REQUIRE_TOOLS=1)"
    );
    None
}
