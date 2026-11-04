//! A `-g` Mach-O object, read back with the LLVM tools.
//!
//! There is no Mac here. Every assertion is structural, and the strongest of them is
//! that `dsymutil` itself links this object's DWARF when it is handed a debug map.
//! Nothing below claims a debugger stopped at a breakpoint on Darwin.
#![allow(clippy::expect_used)]

use lamina::api::{ObjectBuild, Target, compile_module_to_object_with};
use lamina::parser::parse_module;
use std::env;
use std::fs;
use std::process::Command;
use std::str::FromStr;

mod common;
use common::TempDir;

const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
  entry:
    %r = mul.i64 %n, 3
    ret.i64 %r
}

fn @scale(f64 %x, u32 %k) -> f64 {
  entry:
    ret.f64 %x
}

fn @main() -> i64 {
  entry:
    %v = call @triple(7)
    ret.i64 %v
}
"#;

/// Skips when the tool is missing, unless the suite was told to require it.
fn tool(tool: &str) -> Option<String> {
    for name in [tool.to_string(), format!("{tool}-18"), format!("{tool}-19")] {
        let found = Command::new(&name)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if found {
            return Some(name);
        }
    }
    let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
    assert!(
        !required,
        "{tool} is not installed (LAMINA_REQUIRE_TOOLS=1)"
    );
    None
}

fn object(target: &str, debug: bool) -> Vec<u8> {
    let target = Target::from_str(target).expect("target string parses");
    let module = parse_module(SOURCE).expect("module parses");
    let mut build = ObjectBuild::default();
    build.debug_lines = debug;
    compile_module_to_object_with(&module, target, &build).expect("object bytes")
}

fn dwarfdump(target: &str, switch: &str) -> Option<String> {
    let dwarfdump = tool("llvm-dwarfdump")?;
    let dir = TempDir::new("macho-debug-info");
    let path = dir.path().join("out.o");
    fs::write(&path, object(target, true)).expect("write object");
    let output = Command::new(dwarfdump)
        .arg(switch)
        .arg(&path)
        .output()
        .expect("llvm-dwarfdump");
    assert!(output.status.success(), "{switch} on {target}");
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[test]
fn both_mach_o_targets_pass_the_dwarf_verifier() {
    for target in ["x86_64_macos", "aarch64_macos"] {
        let Some(text) = dwarfdump(target, "--verify") else {
            return;
        };
        assert!(text.contains("No errors."), "{target}: {text}");
    }
}

#[test]
fn every_function_with_a_body_is_a_subprogram_at_its_own_address() {
    for target in ["x86_64_macos", "aarch64_macos"] {
        let Some(dump) = dwarfdump(target, "--debug-info") else {
            return;
        };
        assert_eq!(dump.matches("DW_TAG_subprogram").count(), 3, "{dump}");
        for (name, line) in [("triple", 1), ("scale", 7), ("main", 12)] {
            let die = dump
                .split("DW_TAG_subprogram")
                .find(|part| part.contains(&format!("DW_AT_name\t(\"{name}\")")))
                .unwrap_or_else(|| panic!("no subprogram for {name} in {dump}"));
            assert!(die.contains(&format!("DW_AT_decl_line\t({line})")), "{die}");
            assert!(die.contains("DW_AT_frame_base"), "{die}");
            assert!(die.contains("DW_AT_low_pc"), "{die}");
        }
        // The underscore belongs to the object label, not to what a debugger shows.
        assert!(!dump.contains("\"_main\""), "{dump}");
    }
}

/// The property the ELF path holds, held here too: nothing outside the backends knows
/// where the allocator put a value, and a guess makes a debugger print a wrong number.
#[test]
fn no_parameter_is_given_a_location() {
    for target in ["x86_64_macos", "aarch64_macos"] {
        let Some(dump) = dwarfdump(target, "--debug-info") else {
            return;
        };
        assert!(!dump.contains("DW_AT_location"), "{target}: {dump}");
        assert!(!dump.contains("DW_TAG_variable"), "{target}: {dump}");
    }
}

#[test]
fn the_line_table_reaches_the_source_lines_of_every_function() {
    let Some(dump) = dwarfdump("x86_64_macos", "--debug-line") else {
        return;
    };
    for line in ["     3 ", "     9 ", "    14 "] {
        assert!(dump.contains(line), "no row for line{line}in {dump}");
    }
}

/// Debug info is opt-in: a build without `-g` must be the same object it always was.
#[test]
fn a_build_without_debug_lines_is_unchanged() {
    for target in ["x86_64_macos", "aarch64_macos"] {
        let plain = object(target, false);
        assert!(
            !String::from_utf8_lossy(&plain).contains("__DWARF"),
            "{target} carries debug sections without -g"
        );
    }
}
