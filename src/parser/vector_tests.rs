#![allow(clippy::expect_used, clippy::unwrap_used)]

use crate::ir::{BinaryOp, Instruction, PrimitiveType, Type};
use crate::parser::parse_module;

fn body(instructions: &str) -> String {
    format!("fn @f(ptr %p) -> i64 {{\nentry:\n{instructions}\n  ret.i64 0\n}}\n")
}

fn refusal(instructions: &str) -> String {
    match parse_module(&body(instructions)) {
        Ok(_) => panic!("accepted: {instructions}"),
        Err(e) => e.to_string(),
    }
}

fn only_instruction(source: &str) -> Instruction<'static> {
    let source: &'static str = Box::leak(source.to_owned().into_boxed_str());
    let module = Box::leak(Box::new(parse_module(source).expect("parse")));
    module
        .functions
        .values()
        .next()
        .expect("function")
        .basic_blocks
        .values()
        .next()
        .expect("block")
        .instructions
        .first()
        .expect("instruction")
        .clone()
}

#[test]
fn vector_arithmetic_reuses_the_scalar_opcode() {
    let instruction =
        only_instruction("fn @f() -> i64 {\nentry:\n  %c = add.<4 x i32> 0, 0\n  ret.i64 0\n}\n");
    let Instruction::VectorBinary {
        op,
        element_type,
        lanes,
        ..
    } = instruction
    else {
        panic!("expected VectorBinary, got {instruction}");
    };
    assert_eq!(op, BinaryOp::Add);
    assert_eq!(element_type, PrimitiveType::I32);
    assert_eq!(lanes, 4);
}

#[test]
fn a_vector_prints_the_way_it_was_written() {
    let source = "fn @f(ptr %p) -> i64 {
entry:
  %a = load.<4 x i32> %p
  %b = load.<4 x i32> %p
  %c = add.<4 x i32> %a, %b
  %v = splat.<4 x i32> 7
  %x = extractlane.<4 x i32> %v, 3
  %w = insertlane.<4 x i32> %v, 9, 1
  %s = shuffle.<4 x i32> %a, %b, [0, 5, 2, 7]
  ret.i64 0
}
";
    let module = parse_module(source).expect("parse");
    let printed: Vec<String> = module
        .functions
        .values()
        .next()
        .unwrap()
        .basic_blocks
        .values()
        .next()
        .unwrap()
        .instructions
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        printed,
        vec![
            "%a = load.<4 x i32> %p",
            "%b = load.<4 x i32> %p",
            "%c = add.<4 x i32> %a, %b",
            "%v = splat.<4 x i32> 7",
            "%x = extractlane.<4 x i32> %v, 3",
            "%w = insertlane.<4 x i32> %v, 9, 1",
            "%s = shuffle.<4 x i32> %a, %b, [0, 5, 2, 7]",
            "ret.i64 0",
        ]
    );
}

#[test]
fn load_and_store_take_a_vector_suffix() {
    let instruction = only_instruction(&body("  %v = load.<4 x i32> %p"));
    let Instruction::Load { ty, .. } = instruction else {
        panic!("expected Load");
    };
    assert_eq!(
        ty,
        Type::Vector {
            element_type: PrimitiveType::I32,
            lanes: 4
        }
    );
}

#[test]
fn a_shuffle_mask_chosen_at_run_time_is_refused() {
    let message = refusal("  %s = shuffle.<4 x i32> %p, %p, %p");
    assert!(
        message.contains("shuffle takes a constant lane mask"),
        "{message}"
    );
    let inside = refusal("  %s = shuffle.<4 x i32> %p, %p, [0, %p, 2, 3]");
    assert!(inside.contains("constant lane mask"), "{inside}");
}

#[test]
fn a_lane_index_chosen_at_run_time_is_refused() {
    for opcode in ["extractlane", "insertlane"] {
        let text = if opcode == "extractlane" {
            "  %x = extractlane.<4 x i32> %p, %p".to_string()
        } else {
            "  %x = insertlane.<4 x i32> %p, 1, %p".to_string()
        };
        let message = refusal(&text);
        assert!(
            message.contains("takes a constant lane index"),
            "{opcode}: {message}"
        );
    }
}

#[test]
fn a_vector_that_fills_no_register_is_refused_by_name() {
    let message = refusal("  %v = splat.<8 x i64> 1");
    assert!(message.contains("512 bits"), "{message}");
    assert!(message.contains("no vector register width"), "{message}");
    assert!(message.contains("<2 x i64>"), "{message}");
}

#[test]
fn a_lane_type_with_no_machine_form_is_refused_by_name() {
    for ty in ["<4 x bool>", "<2 x ptr>", "<2 x i128>"] {
        let message = refusal(&format!("  %v = load.{ty} %p"));
        assert!(message.contains("no vector lane form"), "{ty}: {message}");
    }
}

#[test]
fn a_vector_may_not_appear_in_a_declaration() {
    let cases = [
        "fn @f(<4 x i32> %v) -> i64 {\nentry:\n  ret.i64 0\n}\n".to_string(),
        "fn @f() -> <4 x i32> {\nentry:\n  %v = splat.<4 x i32> 0\n  ret.<4 x i32> %v\n}\n"
            .to_string(),
        "global $g: <4 x i32>\nfn @f() -> i64 {\nentry:\n  ret.i64 0\n}\n".to_string(),
        "type !T = struct { v: <4 x i32> }\nfn @f() -> i64 {\nentry:\n  ret.i64 0\n}\n".to_string(),
        body("  %v = stackalloc.<4 x i32>"),
    ];
    for source in cases {
        let message = match parse_module(&source) {
            Ok(_) => panic!("accepted a declared vector:\n{source}"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("names a vector type"),
            "{source}\n{message}"
        );
    }
}

#[test]
fn operations_with_no_vector_form_are_refused_by_name() {
    for op in ["div", "rem", "rotl", "rotr"] {
        let message = refusal(&format!("  %c = {op}.<4 x i32> %p, %p"));
        assert!(message.contains("has no vector form"), "{op}: {message}");
    }
}

#[test]
fn a_vector_shift_reads_its_form_from_the_count() {
    for op in ["shl", "shr"] {
        let uniform = format!(
            "fn @f(ptr %p, i64 %n) -> i64 {{\nentry:\n  %a = load.<4 x i32> %p\n  %c = {op}.<4 x i32> %a, %n\n  store.<4 x i32> %p, %c\n  ret.i64 0\n}}\n"
        );
        parse_module(&uniform).unwrap_or_else(|e| panic!("{op} by a scalar: {e}"));
        let per_lane = format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n  %a = load.<4 x i32> %p\n  %b = load.<4 x i32> %p\n  %c = {op}.<4 x i32> %a, %b\n  store.<4 x i32> %p, %c\n  ret.i64 0\n}}\n"
        );
        parse_module(&per_lane).unwrap_or_else(|e| panic!("{op} by a vector: {e}"));
        let literal = format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n  %a = load.<4 x i32> %p\n  %c = {op}.<4 x i32> %a, 3\n  store.<4 x i32> %p, %c\n  ret.i64 0\n}}\n"
        );
        parse_module(&literal).unwrap_or_else(|e| panic!("{op} by a literal: {e}"));
    }
}

#[test]
fn a_vector_shift_by_something_that_is_neither_is_refused_by_name() {
    for op in ["shl", "shr"] {
        let message = refusal(&format!("  %c = {op}.<4 x i32> %p, %p"));
        assert!(
            message.contains("takes a ptr as its shift count"),
            "{message}"
        );
        let shape = match parse_module(&format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n  %a = load.<4 x i32> %p\n  %b = load.<2 x i64> %p\n  %c = {op}.<4 x i32> %a, %b\n  ret.i64 0\n}}\n"
        )) {
            Ok(_) => panic!("accepted a count of another shape"),
            Err(e) => e.to_string(),
        };
        assert!(
            shape.contains("shifts <4 x i32> by a <2 x i64> count"),
            "{shape}"
        );
    }
}

#[test]
fn a_lane_outside_the_vector_is_refused() {
    assert!(
        refusal("  %x = extractlane.<4 x i32> %p, 4").contains("reads lane 4"),
        "extractlane out of range accepted"
    );
    assert!(
        refusal("  %x = insertlane.<4 x i32> %p, 1, 9").contains("writes lane 9"),
        "insertlane out of range accepted"
    );
    assert!(
        refusal("  %s = shuffle.<4 x i32> %p, %p, [0, 1, 2, 8]").contains("selects lane 8"),
        "shuffle index out of range accepted"
    );
    assert!(
        refusal("  %s = shuffle.<4 x i32> %p, %p, [0, 1]").contains("2 mask entries"),
        "short shuffle mask accepted"
    );
}

#[test]
fn an_array_type_is_still_an_array() {
    let instruction = only_instruction(&body("  %a = stackalloc.[4 x i32]"));
    let Instruction::Alloc { allocated_ty, .. } = instruction else {
        panic!("expected Alloc");
    };
    assert_eq!(
        allocated_ty,
        Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: 4
        }
    );
}
