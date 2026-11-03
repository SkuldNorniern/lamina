#![allow(clippy::expect_used)]

use super::parse_module;
use crate::{ir::SourceLoc, mir::codegen::from_ir_x86_64};
use lamina_platform::DEFAULT_SOURCE_FILE;
use std::num::NonZeroU32;

const TWO_FILE_SOURCE: &str = "\
.file 2 \"util.c\"
fn @main() -> i64 {
entry:
    .loc 2 7 3
    %a = add.i64 1, 2
    %b = add.i64 %a, 3
    ret.i64 %b
}
";

const PLAIN_SOURCE: &str = "\
fn @main() -> i64 {
entry:
    %a = add.i64 1, 2
    ret.i64 %a
}
";

fn instruction_locs(source: &str) -> Vec<Option<SourceLoc>> {
    let module = parse_module(source).expect("parse");
    let function = module.functions.values().next().expect("one function");
    let block = function.basic_blocks.values().next().expect("one block");
    block.instruction_locs.clone()
}

fn parse_error(source: &str) -> String {
    parse_module(source)
        .expect_err("source should be rejected")
        .to_string()
}

#[test]
fn source_without_location_directives_stays_in_the_default_file() {
    let locs = instruction_locs(PLAIN_SOURCE);
    assert_eq!(
        locs,
        [SourceLoc::new(3, 5), SourceLoc::new(4, 5)],
        "{locs:?}"
    );
    assert!(
        locs.iter()
            .flatten()
            .all(|loc| loc.file == DEFAULT_SOURCE_FILE)
    );
}

#[test]
fn loc_directive_states_the_file_and_position_of_the_next_instruction() {
    let locs = instruction_locs(TWO_FILE_SOURCE);
    assert_eq!(
        locs,
        [
            SourceLoc::in_file(2, 7, 3),
            SourceLoc::new(6, 5),
            SourceLoc::new(7, 5),
        ]
    );
}

#[test]
fn loc_directive_applies_only_to_the_instruction_after_it() {
    let source = "\
fn @main() -> i64 {
entry:
    .loc 4 9 1
    %a = add.i64 1, 2
    %b = add.i64 %a, 3
    ret.i64 %b
}
.file 4 \"gen.lamina\"
";
    let locs = instruction_locs(source);
    assert_eq!(locs[0], SourceLoc::in_file(4, 9, 1));
    assert_eq!(locs[1], SourceLoc::new(5, 5));
}

#[test]
fn file_declared_after_its_function_is_accepted() {
    let source = format!("{TWO_FILE_SOURCE}.file 3 \"late.c\"\n");
    let module = parse_module(&source).expect("parse");
    let two = NonZeroU32::new(2).expect("2");
    let three = NonZeroU32::new(3).expect("3");
    assert_eq!(module.source_files[&two], "util.c");
    assert_eq!(module.source_files[&three], "late.c");
}

#[test]
fn file_table_is_carried_into_the_mir_module() {
    let module = parse_module(TWO_FILE_SOURCE).expect("parse");
    let mir = from_ir_x86_64(&module, "files").expect("lower");
    let id = NonZeroU32::new(2).expect("2");
    assert_eq!(
        mir.source_files.get(&id).map(String::as_str),
        Some("util.c")
    );
    assert_eq!(mir.source_files.len(), 1);
}

#[test]
fn file_path_escapes_resolve_backslash_and_quote() {
    let source = "\
.file 2 \"C:\\\\src\\\\main.c\"
.file 3 \"say \\\"hi\\\"\"
fn @main() -> i64 {
entry:
    ret.i64 0
}
";
    let module = parse_module(source).expect("parse");
    let two = NonZeroU32::new(2).expect("2");
    let three = NonZeroU32::new(3).expect("3");
    assert_eq!(module.source_files[&two], "C:\\src\\main.c");
    assert_eq!(module.source_files[&three], "say \"hi\"");
}

#[test]
fn unsupported_file_path_escape_is_rejected() {
    let error = parse_error(
        "\
.file 2 \"C:\\src\\main.c\"
fn @main() -> i64 {
entry:
    ret.i64 0
}
",
    );
    assert!(error.contains("Unsupported escape"), "{error}");
}

#[test]
fn loc_directive_must_be_followed_by_an_instruction() {
    let error = parse_error(
        "\
fn @main() -> i64 {
entry:
    .loc 1 3 1
}
",
    );
    assert!(
        error.contains("must be followed by an instruction"),
        "{error}"
    );
}

#[test]
fn loc_naming_an_undeclared_file_is_rejected() {
    let error = parse_error(
        "\
fn @main() -> i64 {
entry:
    .loc 3 2 1
    ret.i64 0
}
",
    );
    assert!(error.contains("undeclared file id 3"), "{error}");
}

#[test]
fn duplicate_file_id_is_rejected() {
    let error = parse_error(
        "\
.file 2 \"a.c\"
.file 2 \"b.c\"
fn @main() -> i64 {
entry:
    ret.i64 0
}
",
    );
    assert!(error.contains("Duplicate file id: 2"), "{error}");
}

#[test]
fn file_and_line_zero_are_rejected() {
    let zero_file = parse_error(
        "\
.file 0 \"a.c\"
fn @main() -> i64 {
entry:
    ret.i64 0
}
",
    );
    assert!(zero_file.contains("Invalid file id: 0"), "{zero_file}");

    let zero_line = parse_error(
        "\
.file 2 \"a.c\"
fn @main() -> i64 {
entry:
    .loc 2 0 1
    ret.i64 0
}
",
    );
    assert!(zero_line.contains("Invalid '.loc'"), "{zero_line}");
}
