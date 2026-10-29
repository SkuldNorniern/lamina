//! Lamina IR parser.
//!
//! Parsing functionality for Lamina IR text into structured
//! Module representations.

mod functions;
mod globals;
#[cfg(test)]
mod inline_asm_tests;
mod instructions;
pub mod state;
mod types;
#[cfg(test)]
mod validation_tests;
mod values;

use std::{mem, str::from_utf8};

use self::{
    functions::parse_function_def,
    globals::{parse_global_annotations, parse_global_declaration},
    state::ParserState,
    types::parse_type_declaration,
};
#[cfg(feature = "nightly")]
use crate::ir::ModuleAnnotation;
use crate::{
    LaminaError,
    ir::{
        AllocType, Module, PrimitiveType, Type, assignment_opcode_names,
        non_assignment_opcode_names,
    },
    parser::values::{integer_literal, literal_integer},
};

/// Calculates the Levenshtein edit distance between two strings.
///
/// This function computes the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to transform one string
/// into another. The comparison is case-insensitive to improve typo detection.
///
/// # Arguments
///
/// * `s1` - First string to compare
/// * `s2` - Second string to compare
/// * `max_distance` - Maximum distance to consider (for early termination optimization)
///
/// # Returns
///
/// The edit distance between the two strings, or `max_distance + 1` if the
/// distance exceeds `max_distance` (for early termination).
///
/// # Examples
///
/// ```
/// # use lamina::parser::edit_distance;
/// assert_eq!(edit_distance("inline", "inlien", None), 2);
/// assert_eq!(edit_distance("export", "EXPORT", None), 0); // case-insensitive
/// assert_eq!(edit_distance("extern", "external", Some(1)), 2); // exceeds max
/// ```
pub fn edit_distance(s1: &str, s2: &str, max_distance: Option<usize>) -> usize {
    // Normalize to lowercase for case-insensitive comparison
    let s1_lower: Vec<char> = s1.to_lowercase().chars().collect();
    let s2_lower: Vec<char> = s2.to_lowercase().chars().collect();
    let m = s1_lower.len();
    let n = s2_lower.len();

    // Early exit for empty strings
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Early exit if length difference exceeds max_distance
    if let Some(max) = max_distance {
        let len_diff = m.abs_diff(n);
        if len_diff > max {
            return max + 1;
        }
    }

    // Use space-optimized DP: only store two rows at a time
    // This reduces space complexity from O(m*n) to O(min(m,n))
    let (shorter, longer) = if m <= n {
        (&s1_lower, &s2_lower)
    } else {
        (&s2_lower, &s1_lower)
    };
    let short_len = shorter.len();
    let long_len = longer.len();

    // Previous row (dp[i-1])
    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    // Current row (dp[i])
    let mut curr_row = vec![0; short_len + 1];

    for i in 1..=long_len {
        curr_row[0] = i;

        for j in 1..=short_len {
            // Cost is 0 if characters match, 1 otherwise
            let cost = if longer[i - 1] == shorter[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }

        if let Some(max) = max_distance
            && curr_row.iter().all(|&d| d > max)
        {
            return max + 1;
        }

        // Swap rows for next iteration
        mem::swap(&mut prev_row, &mut curr_row);
    }

    let distance = prev_row[short_len];
    match max_distance {
        Some(max) if distance > max => max + 1,
        _ => distance,
    }
}

/// Collect close matches from `valid` for `input` (Levenshtein distance ≤ 2),
/// sorted nearest-first.
pub fn suggest_alternatives<'a>(input: &str, valid: &[&'a str]) -> Vec<&'a str> {
    const MAX_TYPO_DISTANCE: usize = 2;
    let mut suggestions: Vec<&str> = valid
        .iter()
        .copied()
        .filter(|&s| edit_distance(input, s, Some(MAX_TYPO_DISTANCE)) <= MAX_TYPO_DISTANCE)
        .collect();
    suggestions.sort_by_key(|&s| edit_distance(input, s, None));
    suggestions
}

/// Returns all valid primitive type names as strings.
///
/// This function delegates to the IR module so parser error messages
/// stay in sync with the actual type system.
pub fn get_primitive_type_names() -> &'static [&'static str] {
    PrimitiveType::all_names()
}

/// Returns all valid allocation type names as strings.
///
/// This function delegates to the IR module so parser error messages
/// stay in sync with the actual type system.
pub fn get_alloc_type_names() -> &'static [&'static str] {
    AllocType::all_names()
}

/// Returns all function annotation keywords recognized by the parser.
///
/// Update this list when adding new annotations to the parser match arm.
pub fn get_annotation_names() -> &'static [&'static str] {
    &[
        "kernel",
        "inline",
        "export",
        "extern",
        "noreturn",
        "noinline",
        "cold",
        "hot",
        "pure",
        "internal",
        "private",
        "hidden",
        "protected",
        "weak",
        "section",
        "align",
        "naked",
        "no_red_zone",
    ]
}

/// Returns all valid instruction opcodes that can appear after an assignment.
///
/// These are instructions that produce a result value (e.g., `%result = add.i32 ...`).
/// This function delegates to the IR module so parser error messages
/// stay in sync with the actual instruction set.
pub fn get_assignment_opcode_names() -> &'static [&'static str] {
    assignment_opcode_names()
}

/// Returns all valid instruction opcodes that don't require an assignment.
///
/// These are instructions that don't produce a result value (e.g., `ret.void`, `jmp label`).
/// This function delegates to the IR module so parser error messages
/// stay in sync with the actual instruction set.
pub fn get_non_assignment_opcode_names() -> &'static [&'static str] {
    non_assignment_opcode_names()
}

const MODULE_ANNOTATION_NAMES: &[&str] = &["pic", "pie"];

const REMOVED_MODULE_ANNOTATIONS: &[(&str, &str)] = &[
    ("optimize_speed", "use --opt-level (0-3)"),
    ("optimize_size", "use --opt-level (0-3)"),
    ("debug", "use -g"),
    ("strip", "use the strip tool on the output"),
    ("target_triple", "use --target"),
];

/// The keyword that follows the leading run of `@name` / `@name(...)` annotations,
/// found without consuming anything.
fn leading_item_keyword<'s>(state: &ParserState<'s>) -> Result<&'s str, LaminaError> {
    let mut probe = state.clone();
    loop {
        probe.skip_whitespace_and_comments();
        if probe.current_char() != Some('@') {
            break;
        }
        probe.advance();
        probe.parse_identifier_str()?;
        if probe.current_char() == Some('(') {
            skip_annotation_argument(&mut probe)?;
        }
    }
    probe.skip_whitespace_and_comments();
    Ok(probe.peek_slice(6).unwrap_or(""))
}

fn skip_annotation_argument(probe: &mut ParserState<'_>) -> Result<(), LaminaError> {
    probe.expect_char('(')?;
    loop {
        match probe.current_char() {
            None => return Err(probe.error("Expected ')' to close the annotation")),
            Some(')') => {
                probe.advance();
                return Ok(());
            }
            Some('"') => {
                probe.parse_string_literal()?;
            }
            Some(_) => probe.advance(),
        }
    }
}

/// The name after a leading `@`, when the upcoming token is an `@` followed by an identifier.
fn upcoming_annotation_name<'s>(state: &'s ParserState<'_>) -> &'s str {
    let bytes = state.bytes();
    let start = state.position() + 1;
    let Some(rest) = bytes.get(start..) else {
        return "";
    };
    let len = rest
        .iter()
        .position(|b| !(b.is_ascii_alphanumeric() || *b == b'_'))
        .unwrap_or(rest.len());
    from_utf8(&rest[..len]).unwrap_or("")
}

fn reject_removed_module_annotation(state: &ParserState<'_>) -> Result<(), LaminaError> {
    if state.current_char() != Some('@') {
        return Ok(());
    }
    let name = upcoming_annotation_name(state);
    match REMOVED_MODULE_ANNOTATIONS
        .iter()
        .find(|(removed, _)| *removed == name)
    {
        Some((removed, replacement)) => {
            Err(state.error(format!("@{removed} was removed; {replacement}")))
        }
        None => Ok(()),
    }
}

#[cfg(feature = "nightly")]
fn parse_module_annotation_if_present(
    state: &mut ParserState<'_>,
    module: &mut Module<'_>,
) -> Result<bool, LaminaError> {
    reject_removed_module_annotation(state)?;
    if state.current_char() != Some('@') {
        return Ok(false);
    }
    let name = upcoming_annotation_name(state).to_string();
    if !MODULE_ANNOTATION_NAMES.contains(&name.as_str()) {
        return Ok(false);
    }
    state.advance();
    state.parse_identifier_str()?;
    let annotation = parse_module_annotation_body(state, &name)?;
    if module
        .annotations
        .iter()
        .any(|existing| mem::discriminant(existing) == mem::discriminant(&annotation))
    {
        return Err(state.error(format!("Duplicate module annotation: @{name}")));
    }
    if module
        .annotations
        .iter()
        .any(|existing| module_annotations_conflict(existing, &annotation))
    {
        return Err(state.error(format!(
            "Conflicting module annotations: @{name} cannot be combined with another module annotation"
        )));
    }
    module.annotations.push(annotation);
    Ok(true)
}

#[cfg(not(feature = "nightly"))]
fn parse_module_annotation_if_present(
    state: &mut ParserState<'_>,
    _module: &mut Module<'_>,
) -> Result<bool, LaminaError> {
    reject_removed_module_annotation(state)?;
    if state.current_char() != Some('@') {
        return Ok(false);
    }
    let name = upcoming_annotation_name(state);
    if MODULE_ANNOTATION_NAMES.contains(&name) {
        return Err(state.error(format!(
            "Module annotation @{name} is only available with the nightly feature"
        )));
    }
    Ok(false)
}

#[cfg(feature = "nightly")]
fn parse_module_annotation_body(
    state: &mut ParserState<'_>,
    name: &str,
) -> Result<ModuleAnnotation, LaminaError> {
    match name {
        "pic" => Ok(ModuleAnnotation::PositionIndependentCode),
        "pie" => Ok(ModuleAnnotation::PositionIndependentExecutable),
        _ => Err(state.error(format!("Unknown module annotation @{name}"))),
    }
}

#[cfg(feature = "nightly")]
fn module_annotations_conflict(a: &ModuleAnnotation, b: &ModuleAnnotation) -> bool {
    use ModuleAnnotation::{PositionIndependentCode, PositionIndependentExecutable};
    matches!(
        (a, b),
        (PositionIndependentCode, PositionIndependentExecutable)
            | (PositionIndependentExecutable, PositionIndependentCode)
    )
}

fn check_call_sites(state: &mut ParserState<'_>, module: &Module<'_>) -> Result<(), LaminaError> {
    for site in mem::take(&mut state.call_sites) {
        if module.functions.contains_key(site.callee) || site.callee.starts_with("__lamina_gpu_") {
            continue;
        }
        state.set_position(site.position);
        return Err(state.error(format!(
            "Call to undefined function @{}\n  Hint: Define 'fn @{}(...)', or declare it with '@extern' before calling it",
            site.callee, site.callee
        )));
    }
    Ok(())
}

fn check_untyped_call_literals(
    state: &mut ParserState<'_>,
    module: &Module<'_>,
) -> Result<(), LaminaError> {
    for arg in mem::take(&mut state.untyped_call_literals) {
        let Some(func) = module.functions.get(arg.callee) else {
            continue;
        };
        let Some(param) = func.signature.params.get(arg.index) else {
            continue;
        };
        let Type::Primitive(param_ty) = param.ty else {
            continue;
        };
        let Some(value) = literal_integer(&arg.value) else {
            continue;
        };
        let message = if param_ty == PrimitiveType::Bool {
            "Integer literal cannot be passed to a bool parameter\n  Hint: Write 'true' or 'false'"
                .to_string()
        } else {
            match integer_literal(value, param_ty) {
                Ok(_) => continue,
                Err(message) => message,
            }
        };
        state.set_position(arg.position);
        return Err(state.error(message));
    }
    Ok(())
}

fn check_typed_call_args(
    state: &mut ParserState<'_>,
    module: &Module<'_>,
) -> Result<(), LaminaError> {
    for call in mem::take(&mut state.typed_call_args) {
        let Some(func) = module.functions.get(call.callee) else {
            continue;
        };
        let Some(param) = func.signature.params.get(call.index) else {
            continue;
        };
        if param.ty != call.ty {
            let message = format!(
                "Argument {} of call to @{} has type {}, but parameter %{} is declared {}\n  Hint: Write '{} <value>' to match the parameter, or drop the type",
                call.index + 1,
                call.callee,
                call.ty,
                param.name,
                param.ty,
                param.ty
            );
            state.set_position(call.position);
            return Err(state.error(message));
        }
    }
    Ok(())
}

/// Parses a string containing Lamina IR text into a Module.
pub fn parse_module(input: &str) -> Result<Module<'_>, LaminaError> {
    let mut state = ParserState::new(input);
    let mut module = Module::new();

    loop {
        state.skip_whitespace_and_comments();
        if state.is_eof() {
            break;
        }

        if parse_module_annotation_if_present(&mut state, &mut module)? {
            continue;
        }

        let keyword_slice = state.peek_slice(6).unwrap_or("");
        let item_keyword = leading_item_keyword(&state)?;

        if keyword_slice.starts_with("type") {
            let decl = parse_type_declaration(&mut state)?;
            let name = decl.name;
            if module.type_declarations.insert(name, decl).is_some() {
                return Err(state.error(format!(
                    "Duplicate type declaration: !{name}\n  Hint: Each type can only be declared once"
                )));
            }
        } else if item_keyword.starts_with("global") {
            let annotations = parse_global_annotations(&mut state)?;
            let decl =
                parse_global_declaration(&mut state, annotations, &module.type_declarations)?;
            let name = decl.name;
            if module.global_declarations.insert(name, decl).is_some() {
                return Err(state.error(format!(
                    "Duplicate global declaration: ${name}\n  Hint: Each global can only be declared once"
                )));
            }
        } else if keyword_slice.starts_with("fn") || keyword_slice.starts_with('@') {
            let func = parse_function_def(&mut state)?;
            let name = func.name;
            if module.functions.insert(name, func).is_some() {
                return Err(state.error(format!(
                    "Duplicate function definition: @{name}\n  Hint: Each function can only be defined once"
                )));
            }
        } else {
            let token = state.peek_slice(20).unwrap_or("");
            let suggestions = if token.starts_with("type") {
                "Did you mean 'type'? (Note: keywords are case-sensitive)"
            } else if token.starts_with("global") {
                "Did you mean 'global'?"
            } else if token.starts_with("fn") || token.starts_with('@') {
                "Did you mean 'fn' or '@' for a function?"
            } else {
                "Expected one of: 'type', 'global', 'fn', or '@' (for function)"
            };

            return Err(state.error(format!(
                "Unexpected token at top level: {token:?}\n  Hint: {suggestions}"
            )));
        }
    }

    check_call_sites(&mut state, &module)?;
    check_typed_call_args(&mut state, &module)?;
    check_untyped_call_literals(&mut state, &module)?;
    module
        .validate()
        .map_err(|errors| LaminaError::ValidationError(errors.join("\n")))?;
    Ok(module)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ir::{
        AtomicBinOp, BinaryOp, CmpOp, GlobalInit, IRBuilder, Instruction, Literal, MemoryOrdering,
        Type, Value,
        builder::{i32 as ir_i32, i64 as ir_i64},
    };
    use std::fs;

    #[test]
    fn test_parse_simple_add_function() {
        let input = r#"
            fn @add(i32 %a, i32 %b) -> i32 {
              entry:
                %sum = add.i32 %a, %b
                ret.i32 %sum
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.functions.len(), 1);
        let func = module
            .functions
            .get("add")
            .expect("Function @add not found");

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.params[0].name, "a");
        assert_eq!(
            func.signature.params[0].ty,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.signature.params[1].name, "b");
        assert_eq!(
            func.signature.params[1].ty,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(
            func.signature.return_type,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.basic_blocks.len(), 1);
        assert_eq!(func.entry_block, "entry");

        let entry_block = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");
        assert_eq!(entry_block.instructions.len(), 2);

        match &entry_block.instructions[0] {
            Instruction::Binary {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*result, "sum");
                assert_eq!(*ty, PrimitiveType::I32);
                assert_eq!(*lhs, Value::Variable("a"));
                assert_eq!(*rhs, Value::Variable("b"));
            }
            _ => panic!("Unexpected instruction: {:?}", entry_block.instructions[0]),
        }

        match &entry_block.instructions[1] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                assert_eq!(value.as_ref(), Some(&Value::Variable("sum")));
            }
            _ => panic!("Unexpected instruction: {:?}", entry_block.instructions[1]),
        }
    }

    #[test]
    fn test_parse_type_declarations() {
        let input = r#"
            type !Vec2 = struct { x: f32, y: f32 }
            type !Matrix = [4 x i32]
        "#;
        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.type_declarations.len(), 2);

        let vec2 = module.type_declarations.get("Vec2").unwrap();
        assert_eq!(vec2.name, "Vec2");
        if let Type::Struct { fields, .. } = &vec2.ty {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "x");
            assert_eq!(fields[0].ty, Type::Primitive(PrimitiveType::F32));
            assert_eq!(fields[1].name, "y");
            assert_eq!(fields[1].ty, Type::Primitive(PrimitiveType::F32));
        } else {
            panic!("Expected struct for Vec2");
        }

        let matrix = module.type_declarations.get("Matrix").unwrap();
        assert_eq!(matrix.name, "Matrix");
        if let Type::Array { element_type, size } = &matrix.ty {
            assert_eq!(*size, 4);
            assert_eq!(element_type.as_ref(), &Type::Primitive(PrimitiveType::I32));
        } else {
            panic!("Expected array for Matrix");
        }
    }

    #[test]
    fn test_parse_exported_global_declaration() {
        let module = parse_module("@export global $g: f64 = 1.5\nglobal $h: i32 = 2")
            .expect("Parsing failed");
        assert!(module.global_declarations["g"].exported);
        assert!(!module.global_declarations["h"].exported);
    }

    #[test]
    fn instruction_operand_literal_must_fit_its_stated_type() {
        let error = |source: &str| match parse_module(source) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected a range error in {source}"),
        };
        let ret_i8 = error("fn @f() -> i8 {\nentry:\n    ret.i8 300\n}");
        assert!(ret_i8.contains("300") && ret_i8.contains("i8"), "{ret_i8}");
        let add_i16 =
            error("fn @f(i16 %a) -> i16 {\nentry:\n    %r = add.i16 %a, 40000\n    ret.i16 %r\n}");
        assert!(
            add_i16.contains("40000") && add_i16.contains("i16"),
            "{add_i16}"
        );
        assert!(error("fn @f() -> u8 {\nentry:\n    ret.u8 -1\n}").contains("u8"));
    }

    #[test]
    fn large_u64_literal_is_accepted_in_global_and_instruction() {
        let module =
            parse_module("global $big: u64 = 18446744073709551615").expect("u64 global parses");
        assert_eq!(
            module.global_declarations["big"].initializer,
            Some(GlobalInit::Value(Value::Constant(Literal::U64(u64::MAX))))
        );

        let module = parse_module("fn @f() -> u64 {\nentry:\n    ret.u64 18446744073709551615\n}")
            .expect("u64 operand parses");
        let instructions = &module.functions["f"].basic_blocks["entry"].instructions;
        assert!(matches!(
            instructions.last(),
            Some(Instruction::Ret {
                value: Some(Value::Constant(Literal::U64(u64::MAX))),
                ..
            })
        ));
    }

    #[test]
    fn global_initializer_takes_width_of_declared_type() {
        let input = r#"
            global $a: u64 = 9
            global $b: i16 = -3
            global $c: u8 = 200
        "#;
        let module = parse_module(input).expect("Parsing failed");
        let literal = |name: &str| {
            module.global_declarations[name]
                .initializer
                .clone()
                .expect("initializer")
        };
        assert_eq!(
            literal("a"),
            GlobalInit::Value(Value::Constant(Literal::U64(9)))
        );
        assert_eq!(
            literal("b"),
            GlobalInit::Value(Value::Constant(Literal::I16(-3)))
        );
        assert_eq!(
            literal("c"),
            GlobalInit::Value(Value::Constant(Literal::U8(200)))
        );

        assert!(parse_module("global $d: u8 = 300").is_err());
        assert!(parse_module("global $e: u32 = -1").is_err());
    }

    #[test]
    fn test_parse_global_align_annotation() {
        let module =
            parse_module("@align(64)\nglobal $arena: [1048576 x i8]").expect("Parsing failed");
        let arena = module.global_declarations.get("arena").unwrap();
        assert_eq!(arena.align, Some(64));
        assert!(!arena.exported);
    }

    #[test]
    fn test_parse_global_export_and_align_in_either_order() {
        let input = "@export\n@align(16)\nglobal $v: [4 x i64] = [1, 2, 3, 4]";
        let module = parse_module(input).expect("Parsing failed");
        let v = module.global_declarations.get("v").unwrap();
        assert!(v.exported);
        assert_eq!(v.align, Some(16));

        let swapped = "@align(16)\n@export\nglobal $v: [4 x i64] = [1, 2, 3, 4]";
        let module = parse_module(swapped).expect("Parsing failed");
        let v = module.global_declarations.get("v").unwrap();
        assert!(v.exported);
        assert_eq!(v.align, Some(16));
    }

    #[test]
    fn test_parse_global_align_rejects_bad_values() {
        let not_power = parse_module("@align(3)\nglobal $x: i32 = 1")
            .expect_err("non-power-of-two alignment should be rejected")
            .to_string();
        assert!(
            not_power.contains("Invalid alignment: @align(3)"),
            "{not_power}"
        );

        let too_big = parse_module("@align(4294967296)\nglobal $x: i32 = 1")
            .expect_err("alignment wider than u32 should be rejected")
            .to_string();
        assert!(too_big.contains("Invalid alignment"), "{too_big}");

        let duplicate = parse_module("@align(8)\n@align(16)\nglobal $x: i32 = 1")
            .expect_err("a second @align should be rejected")
            .to_string();
        assert!(
            duplicate.contains("Duplicate annotation: @align"),
            "{duplicate}"
        );

        let misspelled = parse_module("@alignn(16)\nglobal $x: i32 = 1")
            .expect_err("an unknown global annotation should be rejected")
            .to_string();
        assert!(misspelled.contains("Did you mean @align?"), "{misspelled}");
    }

    #[test]
    fn test_parse_global_declarations() {
        let input = r#"
            global $count: i64 = 10
            global $message: [12 x bool] = "hello world"
            global $uninit_ptr: ptr
        "#;
        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.global_declarations.len(), 3);

        let count = module.global_declarations.get("count").unwrap();
        assert_eq!(count.name, "count");
        assert_eq!(count.ty, Type::Primitive(PrimitiveType::I64));
        assert_eq!(
            count.initializer.as_ref(),
            Some(&GlobalInit::Value(Value::Constant(Literal::I64(10))))
        );

        let message = module.global_declarations.get("message").unwrap();
        assert_eq!(message.name, "message");
        if let Type::Array { element_type, size } = &message.ty {
            assert_eq!(*size, 12);
            assert_eq!(element_type.as_ref(), &Type::Primitive(PrimitiveType::Bool));
        } else {
            panic!("Expected array for message");
        }
        assert_eq!(
            message.initializer.as_ref(),
            Some(&GlobalInit::Value(Value::Constant(Literal::String(
                "hello world"
            ))))
        );

        let uninit = module.global_declarations.get("uninit_ptr").unwrap();
        assert_eq!(uninit.name, "uninit_ptr");
        assert_eq!(uninit.ty, Type::Primitive(PrimitiveType::Ptr));
        assert!(uninit.initializer.is_none());
    }

    #[test]
    fn test_parse_comments_and_whitespace() {
        let input = r#"
            # This is a comment
            fn @test() -> void { # Another comment
               entry: # Block label comment
                  # Instruction comment
                  ret.void # Trailing comment
            } # Final comment


        "#;
        let module = parse_module(input).expect("Parsing should succeed");
        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test").unwrap();
        assert_eq!(func.basic_blocks.len(), 1);
        let block = func.basic_blocks.get("entry").unwrap();
        assert_eq!(block.instructions.len(), 1);
        assert!(matches!(
            block.instructions[0],
            Instruction::Ret {
                ty: Type::Void,
                value: None
            }
        ));
    }

    #[test]
    fn test_non_ascii_identifier_is_rejected_without_panic() {
        let input = "fn @m\u{e4}in() -> i64 {\n  entry:\n    ret.i64 0\n}\n";
        let result = parse_module(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_context_truncates_on_char_boundary() {
        let long_line = format!("    %a = add.i64 1, 1 {} ret.i64 0", "\u{e9}".repeat(60));
        let input = format!("fn @main() -> i64 {{\n  entry:\n{long_line}\n}}\n");
        let result = parse_module(&input);
        assert!(result.is_err());
    }

    #[test]
    fn caret_stays_inside_windowed_error_line() {
        let nested = format!("{}i64{}", "struct { x: ".repeat(1000), " }".repeat(1000));
        let input = format!(
            "fn @main() -> i64 {{\n  entry:\n    %a = stackalloc.{nested}\n    ret.i64 0\n}}\n"
        );
        let err = parse_module(&input).expect_err("deep type").to_string();
        let caret = err
            .lines()
            .last()
            .expect("caret line")
            .trim_end_matches('^');
        let context = err
            .lines()
            .nth_back(1)
            .expect("context line")
            .trim_start_matches(' ');
        assert!(caret.len() <= context.len() + 2, "{err}");
    }

    #[test]
    fn test_deeply_nested_type_is_rejected_without_overflow() {
        let nested = format!("{}i64{}", "struct { x: ".repeat(1000), " }".repeat(1000));
        let input = format!(
            "fn @main() -> i64 {{\n  entry:\n    %a = stackalloc.{nested}\n    ret.i64 0\n}}\n"
        );
        let err = parse_module(&input).expect_err("deep type");
        assert!(err.to_string().contains("Type nesting exceeds"));
    }

    #[test]
    fn test_instruction_after_terminator_is_rejected() {
        let input = "fn @main() -> i64 {\n  entry:\n    ret.i64 0\n    ret.i64 1\n}\n";
        let err = parse_module(input).expect_err("instruction after terminator");
        assert!(
            err.to_string()
                .contains("after the terminator of block 'entry'")
        );
    }

    #[test]
    fn test_bare_ret_names_type_suffix() {
        let input = "fn @main() -> i64 {\n  entry:\n    ret\n}\n";
        let err = parse_module(input).expect_err("bare ret");
        assert!(err.to_string().contains("'ret.void'"));
    }

    #[test]
    fn test_parse_block_missing_terminator() {
        let input = r#"
            fn @bad() -> i32 {
                entry:
                    %a = add.i32 1, 2
            }
        "#;
        let result = parse_module(input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("must end with a terminator instruction")
            );
        }
    }

    #[test]
    fn test_parse_function_with_multiple_blocks() {
        let input = r#"
            fn @conditional(i32 %x) -> i32 {
              entry:
                %is_pos = gt.i32 %x, 0
                br %is_pos, positive, negative
              
              positive:
                ret.i32 1
                
              negative:
                ret.i32 -42
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module
            .functions
            .get("conditional")
            .expect("Function not found");

        assert_eq!(func.basic_blocks.len(), 3);
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("positive"));
        assert!(func.basic_blocks.contains_key("negative"));

        let positive = func.basic_blocks.get("positive").unwrap();
        let negative = func.basic_blocks.get("negative").unwrap();

        match &positive.instructions[0] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                if let Some(Value::Constant(Literal::I32(val))) = value {
                    assert_eq!(*val, 1);
                } else {
                    panic!("Expected constant 1");
                }
            }
            _ => panic!("Expected ret instruction"),
        }

        match &negative.instructions[0] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                if let Some(Value::Constant(Literal::I32(val))) = value {
                    assert_eq!(*val, -42);
                } else {
                    panic!("Expected constant -42");
                }
            }
            _ => panic!("Expected ret instruction"),
        }
    }

    #[test]
    fn test_parse_array_and_struct_operations() {
        let input = r#"
            fn @test_composite() -> ptr {
              entry:
                %arr_ptr = stackalloc.[4 x i32]
                %elem_ptr = getelem.i32 %arr_ptr, 2
                %elem = load.i32 %elem_ptr
                
                %struct_ptr = stackalloc.struct { x: i32, y: i32 }
                %y_ptr = getfieldptr.struct { x: i32, y: i32 } %struct_ptr, 1
                %y_val = load.i32 %y_ptr
                
                ret.ptr %struct_ptr
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module
            .functions
            .get("test_composite")
            .expect("Function not found");

        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");
        assert_eq!(entry.instructions.len(), 7); // stackalloc, getelem, load, stackalloc, getfieldptr, load, ret

        match &entry.instructions[0] {
            Instruction::Alloc {
                result,
                allocated_ty,
                alloc_type,
                ..
            } => {
                assert_eq!(*result, "arr_ptr");
                if let Type::Array { element_type, size } = allocated_ty {
                    assert_eq!(**element_type, Type::Primitive(PrimitiveType::I32));
                    assert_eq!(*size, 4);
                } else {
                    panic!("Expected array type");
                }
                assert_eq!(*alloc_type, AllocType::Stack);
            }
            _ => panic!("Expected alloc instruction"),
        }

        match &entry.instructions[1] {
            Instruction::GetElemPtr {
                result,
                array_ptr,
                index,
                element_type: _,
            } => {
                assert_eq!(*result, "elem_ptr");
                assert_eq!(*array_ptr, Value::Variable("arr_ptr"));
                assert_eq!(*index, Value::Constant(Literal::I32(2)));
            }
            _ => panic!("Expected getelem instruction"),
        }
    }

    #[test]
    fn test_parse_function_call() {
        let input = r#"
            fn @callee(i32 %x, i32 %y) -> i32 {
              entry:
                %sum = add.i32 %x, %y
                ret.i32 %sum
            }
            
            fn @caller() -> i32 {
              entry:
                %result = call @callee(5, 10)
                ret.i32 %result
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");

        // Check caller function has the correct call instruction
        let caller = module
            .functions
            .get("caller")
            .expect("Caller function not found");
        let entry = caller
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        match &entry.instructions[0] {
            Instruction::Call {
                result,
                func_name,
                args,
            } => {
                let result_str: &str = result.as_ref().unwrap();
                let func_str: &str = func_name;
                assert_eq!(result_str, "result");
                assert_eq!(func_str, "callee");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Value::Constant(Literal::I32(5)));
                assert_eq!(args[1], Value::Constant(Literal::I32(10)));
            }
            _ => panic!("Expected call instruction"),
        }
    }

    #[test]
    fn test_parse_comparison_operators() {
        let input = r#"
            fn @test_comparisons(i32 %a, i32 %b) -> i32 {
              entry:
                %eq = eq.i32 %a, %b
                %ne = ne.i32 %a, %b
                %lt = lt.i32 %a, %b
                %le = le.i32 %a, %b
                %gt = gt.i32 %a, %b
                %ge = ge.i32 %a, %b
                
                ret.i32 0
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module
            .functions
            .get("test_comparisons")
            .expect("Function not found");
        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        assert_eq!(entry.instructions.len(), 7); // 6 comparisons + ret

        match &entry.instructions[0] {
            Instruction::Cmp {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => {
                assert_eq!(*op, CmpOp::Eq);
                assert_eq!(*result, "eq");
                assert_eq!(*ty, PrimitiveType::I32);
                assert_eq!(*lhs, Value::Variable("a"));
                assert_eq!(*rhs, Value::Variable("b"));
            }
            _ => panic!("Expected comparison instruction"),
        }

        match &entry.instructions[1] {
            Instruction::Cmp { op, result, .. } => {
                assert_eq!(*op, CmpOp::Ne);
                assert_eq!(*result, "ne");
            }
            _ => panic!("Expected comparison instruction"),
        }
    }

    fn extend_line_error(line: &str) -> String {
        let input =
            format!("fn @f(i8 %x, u8 %y, bool %b) -> i64 {{\nentry:\n  {line}\n  ret.i64 0\n}}\n");
        parse_module(&input)
            .expect_err("extend line should be rejected")
            .to_string()
    }

    #[test]
    fn removed_zext_and_sext_point_at_extend() {
        for line in [
            "%r = zext.u8.i64 %y",
            "%r = sext.i8.i64 %x",
            "%r = zext.i8.i64 %x",
        ] {
            let err = extend_line_error(line);
            assert!(err.contains("Unknown instruction opcode"), "{line}: {err}");
            assert!(
                err.contains("extend.<source-type>.<target-type>"),
                "{line}: {err}"
            );
        }
    }

    #[test]
    fn extend_rejects_narrowing_and_equal_width() {
        let narrowing = extend_line_error("%r = extend.i64.i32 %x");
        assert!(
            narrowing.contains("narrows") && narrowing.contains("trunc.i64.i32"),
            "{narrowing}"
        );
        let equal = extend_line_error("%r = extend.u8.u8 %y");
        assert!(
            equal.contains("keeps the width") && equal.contains("bitcast.u8.u8"),
            "{equal}"
        );
    }

    #[test]
    fn extend_rejects_float_and_pointer_sources() {
        let float = extend_line_error("%r = extend.f32.i64 %x");
        assert!(float.contains("integer or bool"), "{float}");
        let pointer = extend_line_error("%r = extend.ptr.i64 %x");
        assert!(pointer.contains("integer or bool"), "{pointer}");
    }

    #[test]
    fn extend_picks_sign_from_source_type() {
        let input = "fn @f(i8 %x, u8 %y, bool %b) -> i64 {\nentry:\n  %a = extend.i8.i64 %x\n  %c = extend.u8.i64 %y\n  %d = extend.bool.i64 %b\n  ret.i64 0\n}\n";
        let module = parse_module(input).expect("parse");
        let entry = &module.functions["f"].basic_blocks["entry"];
        assert!(matches!(
            entry.instructions[0],
            Instruction::SignExtend { .. }
        ));
        assert!(matches!(
            entry.instructions[1],
            Instruction::ZeroExtend { .. }
        ));
        assert!(matches!(
            entry.instructions[2],
            Instruction::ZeroExtend { .. }
        ));
    }

    #[test]
    fn test_parse_u8_type_and_extend() {
        let input = r#"
            fn @test_u8() -> i64 {
              entry:
                %u8_val = add.u8 10, 5
                %extended = extend.u8.i64 %u8_val
                ret.i64 %extended
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("test_u8").expect("Function not found");
        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        match &entry.instructions[0] {
            Instruction::Binary { op, ty, .. } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*ty, PrimitiveType::U8);
            }
            _ => panic!("Expected Binary instruction"),
        }

        match &entry.instructions[1] {
            Instruction::ZeroExtend {
                source_type,
                target_type,
                ..
            } => {
                assert_eq!(*source_type, PrimitiveType::U8);
                assert_eq!(*target_type, PrimitiveType::I64);
            }
            _ => panic!("Expected ZeroExtend instruction"),
        }
    }

    #[test]
    fn test_parse_tensor_benchmark() -> Result<(), LaminaError> {
        // Load the benchmark Lamina code
        let source = fs::read_to_string("benchmarks/2Dmatmul/2Dmatmul.lamina").map_err(|e| {
            LaminaError::ParsingError(format!("Failed to read benchmark file: {}", e))
        })?;

        let module: Module = parse_module(&source)?;

        // Basic Assertions on the parsed module
        assert_eq!(
            module.type_declarations.len(),
            0,
            "Should have no type declarations"
        );
        assert_eq!(
            module.global_declarations.len(),
            0,
            "Should have no global declarations"
        );
        let declared = source
            .lines()
            .filter(|l| l.trim_start().starts_with("fn @"))
            .count();
        assert_eq!(
            module.functions.len(),
            declared,
            "every declared function should parse"
        );

        // Check @main function details
        let main_func = module
            .functions
            .get("main")
            .expect("Missing @main function");
        assert_eq!(main_func.name, "main");
        assert!(
            main_func.signature.params.is_empty(),
            "@main should have no parameters"
        );
        assert_eq!(
            main_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64),
            "@main should return i64"
        );
        assert_eq!(
            main_func.basic_blocks.len(),
            1,
            "@main should have 1 basic block"
        );
        let entry_block = main_func
            .basic_blocks
            .get("entry")
            .expect("@main missing entry block");
        assert!(
            entry_block.instructions.len() > 5,
            "@main entry block should have several instructions"
        );
        assert!(
            matches!(
                entry_block.instructions.last(),
                Some(Instruction::Ret { .. })
            ),
            "@main should end with ret"
        );

        // Check @matmul_2d function details (more complex)
        let matmul_func = module
            .functions
            .get("matmul_2d")
            .expect("Missing @matmul_2d function");
        assert_eq!(matmul_func.name, "matmul_2d");
        assert_eq!(
            matmul_func.signature.params.len(),
            3,
            "@matmul_2d should have 3 parameters"
        );
        assert_eq!(
            matmul_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64),
            "@matmul_2d should return i64"
        );

        // Check that it has several basic blocks (it's a complex function)
        assert!(
            matmul_func.basic_blocks.len() >= 7,
            "@matmul_2d should have at least 7 basic blocks"
        );
        assert!(matmul_func.basic_blocks.contains_key("entry"));

        // Check for get_matrix_a_element function
        let get_a_func = module
            .functions
            .get("get_matrix_a_element")
            .expect("Missing @get_matrix_a_element function");
        assert!(
            get_a_func
                .signature
                .params
                .iter()
                .all(|p| matches!(p.ty, Type::Primitive(PrimitiveType::I64)))
        );

        // Check for get_matrix_b_element function
        let get_b_func = module
            .functions
            .get("get_matrix_b_element")
            .expect("Missing @get_matrix_b_element function");
        assert!(
            get_b_func
                .signature
                .params
                .iter()
                .all(|p| matches!(p.ty, Type::Primitive(PrimitiveType::I64)))
        );
        assert_eq!(
            get_b_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64)
        );
        assert!(get_b_func.basic_blocks.contains_key("entry"));

        Ok(())
    }

    #[test]
    fn test_parse_error_recovery() -> Result<(), LaminaError> {
        // Test that parser gives helpful error messages for common mistakes and handles valid edge cases

        // Missing function body should fail
        let source_missing_body = "fn @test() -> i64";
        assert!(parse_module(source_missing_body).is_err());

        // Malformed instruction (missing operand) should fail
        let source_malformed = r#"
fn @test() -> i64 {
  entry:
    %result = add.i64 42
    ret.i64 %result
}
"#;
        assert!(parse_module(source_malformed).is_err());

        // A use of a variable that is never defined is rejected by validation
        let source_undefined_var = r#"
fn @test() -> i64 {
  entry:
    %result = add.i64 %undefined, 42
    ret.i64 %result
}
"#;
        let err = parse_module(source_undefined_var).expect_err("undefined variable");
        assert!(err.to_string().contains("'%undefined'"));

        Ok(())
    }

    #[test]
    fn test_parse_complex_expressions() -> Result<(), LaminaError> {
        let source = r#"
fn @complex_math(i64 %a, i64 %b, i64 %c) -> i64 {
  entry:
    %temp1 = mul.i64 %a, %b
    %temp2 = add.i64 %temp1, %c
    %temp3 = sub.i64 %temp2, 10
    %result = div.i64 %temp3, 2
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("complex_math").unwrap();
        assert_eq!(func.signature.params.len(), 3);
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_parse_nested_calls() -> Result<(), LaminaError> {
        let source = r#"
fn @outer() -> i64 {
  entry:
    %result = call @middle()
    ret.i64 %result
}

fn @middle() -> i64 {
  entry:
    %result = call @inner(42)
    ret.i64 %result
}

fn @inner(i64 %x) -> i64 {
  entry:
    %result = add.i64 %x, 1
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 3);

        // Check that all functions exist and have correct signatures
        assert!(module.functions.contains_key("outer"));
        assert!(module.functions.contains_key("middle"));
        assert!(module.functions.contains_key("inner"));

        let inner_func = module.functions.get("inner").unwrap();
        assert_eq!(inner_func.signature.params.len(), 1);

        Ok(())
    }

    #[test]
    fn test_parse_copy_complex() -> Result<(), LaminaError> {
        let source = r#"
fn @test_copy_complex(i64 %cond1, i64 %cond2) -> i64 {
  entry:
    %is_true1 = ne.i64 %cond1, 0
    br %is_true1, block_a, block_b

  block_a:
    %val_a = add.i64 %cond2, 10
    %intermediate = copy.i64 %val_a
    jmp merge1

  block_b:
    %val_b = mul.i64 %cond2, 2
    %intermediate = copy.i64 %val_b
    jmp merge1

  merge1:
    %is_true2 = gt.i64 %intermediate, 5
    br %is_true2, block_c, block_d

  block_c:
    %final_c = add.i64 %intermediate, 100
    %result = copy.i64 %final_c
    jmp merge2

  block_d:
    %final_d = sub.i64 %intermediate, 50
    %result = copy.i64 %final_d
    jmp merge2

  merge2:
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_copy_complex").unwrap();

        // Should have 7 basic blocks
        assert_eq!(func.basic_blocks.len(), 7);

        // Check that all expected blocks exist
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("block_a"));
        assert!(func.basic_blocks.contains_key("block_b"));
        assert!(func.basic_blocks.contains_key("merge1"));
        assert!(func.basic_blocks.contains_key("block_c"));
        assert!(func.basic_blocks.contains_key("block_d"));
        assert!(func.basic_blocks.contains_key("merge2"));

        Ok(())
    }

    #[test]
    fn test_phi_is_removed_in_favor_of_copy() {
        let source = "fn @f(i64 %c) -> i64 {\nentry:\n  br %c, a, b\na:\n  jmp j\nb:\n  jmp j\nj:\n  %x = phi.i64 [1, a], [2, b]\n  ret.i64 %x\n}\n";
        let err = parse_module(source).expect_err("phi must not parse");
        let message = format!("{err}");
        assert!(message.contains("'phi'"), "{message}");
        assert!(message.contains("copy."), "{message}");
        assert!(message.contains("predecessor"), "{message}");
    }

    #[test]
    fn test_parse_copy_scalar_types() -> Result<(), LaminaError> {
        let source = r#"
fn @f(i64 %n, ptr %p) -> i64 {
entry:
  %a = copy.i64 %n
  %b = copy.f64 2.5
  %c = copy.ptr %p
  %d = copy.bool true
  %e = copy.i128 -170141183460469231731687303715884105000
  %g = copy.u128 340282366920938463463374607431768211455
  ret.i64 %a
}
"#;
        let module = parse_module(source)?;
        let func = module.functions.get("f").unwrap();
        let instructions = &func.basic_blocks["entry"].instructions;
        assert_eq!(instructions.len(), 7);
        for instruction in &instructions[..6] {
            assert!(
                matches!(instruction, Instruction::Copy { .. }),
                "{instruction:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_parse_memory_operations_comprehensive() -> Result<(), LaminaError> {
        let source = r#"
@extern
fn @malloc(i64 %n) -> ptr
@extern
fn @free(ptr %p) -> void

fn @test_memory_ops() -> i64 {
  entry:
    %ptr1 = call @malloc(8)
    %ptr2 = call @malloc(8)
    store.i64 %ptr1, 42
    store.i64 %ptr2, 24
    %val1 = load.i64 %ptr1
    %val2 = load.i64 %ptr2
    %sum = add.i64 %val1, %val2
    call @free(%ptr1)
    call @free(%ptr2)
    ret.i64 %sum
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 3);
        let func = module.functions.get("test_memory_ops").unwrap();

        assert_eq!(func.basic_blocks["entry"].instructions.len(), 10);

        Ok(())
    }

    #[test]
    fn test_parse_large_function() -> Result<(), LaminaError> {
        // Test that parser can handle functions with many instructions and blocks
        let mut source = String::from(
            r#"
fn @large_function(i64 %input) -> i64 {
  entry:
"#,
        );

        // Add many instructions
        for i in 0..50 {
            source.push_str(&format!("    %temp{} = add.i64 %input, {}\n", i, i));
        }

        source.push_str("    ret.i64 %temp49\n}");

        let module = parse_module(&source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("large_function").unwrap();

        // Should have 51 instructions (50 adds + 1 ret)
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 51);

        Ok(())
    }

    #[test]
    fn test_parse_empty_blocks() -> Result<(), LaminaError> {
        let source = r#"
fn @test_empty_blocks() -> i64 {
  entry:
    jmp block1

  block1:
    jmp block2

  block2:
    ret.i64 42
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_empty_blocks").unwrap();

        assert_eq!(func.basic_blocks.len(), 3);

        // Check that blocks exist even if they only contain jumps
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("block1"));
        assert!(func.basic_blocks.contains_key("block2"));

        Ok(())
    }

    #[test]
    fn test_parse_instruction_variations() -> Result<(), LaminaError> {
        let source = r#"
fn @test_instructions() -> i64 {
  entry:
    %a = add.i64 1, 2
    %b = sub.i64 %a, 3
    %c = mul.i64 %b, 4
    %d = div.i64 %c, 2
    %e = rem.i64 %d, 3
    %f = eq.i64 %e, 0
    %g = ne.i64 %f, 0
    %h = lt.i64 %g, 1
    %i = le.i64 %h, 1
    %j = gt.i64 %i, 0
    %k = ge.i64 %j, 0
    %out = writebyte %k
    ret.i64 %k
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_instructions").unwrap();

        // Should have 13 instructions (11 computations + 1 writebyte + 1 ret)
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 13);

        Ok(())
    }

    #[test]
    fn test_parse_minimal_programs() -> Result<(), LaminaError> {
        // Test very simple programs that should parse correctly and produce correct IR

        // Single instruction function
        let source1 = r#"
fn @minimal() -> i64 {
  entry:
    ret.i64 0
}
"#;
        let mut builder1 = IRBuilder::new();
        builder1
            .function("minimal", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(0));
        let expected1 = builder1.build();
        let parsed1 = parse_module(source1)?;
        assert_eq!(parsed1, expected1);

        // Function with just writebyte
        let source2 = r#"
fn @write_only() -> i64 {
  entry:
    %w = writebyte 42
    ret.i64 0
}
"#;
        let mut builder2 = IRBuilder::new();
        builder2
            .function("write_only", Type::Primitive(PrimitiveType::I64))
            .write_byte(ir_i32(42), "w")
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(0));
        let expected2 = builder2.build();
        let parsed2 = parse_module(source2)?;
        assert_eq!(parsed2, expected2);

        // Empty function (just return)
        let source3 = r#"
fn @empty() -> i64 {
  entry:
    ret.i64 42
}
"#;
        let mut builder3 = IRBuilder::new();
        builder3
            .function("empty", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(42));
        let expected3 = builder3.build();
        let parsed3 = parse_module(source3)?;
        assert_eq!(parsed3, expected3);

        Ok(())
    }

    #[test]
    fn test_parse_whitespace_tolerance() -> Result<(), LaminaError> {
        // Expected IR structure for all test cases
        let mut builder = IRBuilder::new();
        builder
            .function("test", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(0));

        let expected_module = builder.build();

        // Test that parser handles various whitespace patterns and produces correct IR

        // Normal spacing should work and produce correct IR
        let source_normal = r#"
fn @test() -> i64 {
  entry:
    ret.i64 0
}
"#;
        let parsed_normal = parse_module(source_normal)?;
        assert_eq!(parsed_normal, expected_module);

        // Extra spaces should work and produce the same correct IR
        let source_extra_spaces = r#"
fn   @test(  )   ->   i64   {
    entry   :
        ret.i64   0
}
"#;
        let parsed_extra_spaces = parse_module(source_extra_spaces)?;
        assert_eq!(parsed_extra_spaces, expected_module);

        // Minimal spacing should work and produce the same correct IR
        let source_minimal = r#"fn @test() -> i64 {
entry:
ret.i64 0
}"#;
        let parsed_minimal = parse_module(source_minimal)?;
        assert_eq!(parsed_minimal, expected_module);

        Ok(())
    }

    #[test]
    fn trailing_dots_mark_extern_declaration_variadic() -> Result<(), LaminaError> {
        let module = parse_module("@extern\nfn @printf(ptr %fmt, ...) -> i32\n")?;
        let printf = &module.functions["printf"];
        assert!(printf.signature.variadic);
        assert_eq!(printf.signature.params.len(), 1);

        let plain = parse_module("fn @main() -> i64 {\nentry:\n  ret.i64 0\n}\n")?;
        assert!(!plain.functions["main"].signature.variadic);
        Ok(())
    }

    #[test]
    fn variadic_parameter_list_may_be_defined_with_a_body() {
        let defined = parse_module("fn @f(i64 %a, ...) -> i64 {\nentry:\n  ret.i64 %a\n}\n")
            .expect("a variadic definition parses");
        assert!(defined.functions["f"].signature.variadic);

        let misplaced = parse_module("@extern\nfn @f(..., i64 %a) -> i64\n");
        assert!(
            format!("{misplaced:?}").contains("must be the last parameter"),
            "{misplaced:?}"
        );
    }

    fn entry_body(line: &str) -> String {
        format!(
            "fn @main(ptr %ptr, ptr %buf, ptr %p, i64 %addr) -> i64 {{\n  entry:\n    {line}\n    ret.i64 0\n}}\n"
        )
    }

    fn entry_instructions(input: &str) -> Vec<Instruction<'_>> {
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("main").expect("Function not found");
        func.basic_blocks
            .get("entry")
            .expect("Entry block not found")
            .instructions
            .clone()
    }

    fn parse_error(line: &str) -> String {
        parse_module(&entry_body(line))
            .expect_err("old spelling should be rejected")
            .to_string()
    }

    #[test]
    fn test_ptrtoint_and_inttoptr_take_suffix_type() {
        let input = entry_body("%a = ptrtoint.i64 %ptr");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::PtrToInt {
                target_type: PrimitiveType::I64,
                ..
            }
        ));

        let input = entry_body("%ip = inttoptr.i64 %addr");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::IntToPtr {
                target_type: PrimitiveType::I64,
                ..
            }
        ));
    }

    #[test]
    fn test_old_ptrtoint_and_inttoptr_spelling_is_rejected() {
        let err = parse_error("%a = ptrtoint %ptr, .i64");
        assert!(err.contains("ptrtoint.i64 %ptr"), "{err}");

        let err = parse_error("%p = inttoptr %addr, .i64");
        assert!(err.contains("inttoptr.i64 %addr"), "{err}");
    }

    #[test]
    fn test_getelem_takes_element_type_in_suffix() {
        let input = entry_body("%e = getelem.i8 %buf, 0");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::GetElemPtr {
                element_type: Type::Primitive(PrimitiveType::I8),
                ..
            }
        ));

        let input = entry_body("%e = getelem.[3 x i32] %buf, 1");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::GetElemPtr {
                element_type: Type::Array { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_old_getelem_spellings_are_rejected() {
        let err = parse_error("%e = getelem.ptr %buf, 0, i8");
        assert!(err.contains("getelem.<type> <ptr>, <index>"), "{err}");

        let err = parse_error("%e = getelementptr %buf, 0, char");
        assert!(err.contains("getelem.<type> <ptr>, <index>"), "{err}");
    }

    fn call_module(call: &str) -> String {
        format!(
            "fn @main(i64 %p) -> i64 {{\nentry:\n  %v = {call}\n  ret.i64 0\n}}\n\
             fn @f(i64 %x) -> i64 {{\nentry:\n  ret.i64 %x\n}}\n\
             fn @g(i64 %a, i64 %b) -> i64 {{\nentry:\n  ret.i64 %a\n}}\n\
             @extern\nfn @printf(ptr %fmt, ...) -> i32\n"
        )
    }

    fn call_error(call: &str) -> String {
        parse_module(&call_module(call))
            .expect_err("call should be rejected")
            .to_string()
    }

    fn entry_call(input: &str) -> Instruction<'_> {
        let module = parse_module(input).expect("Parsing failed");
        module.functions["main"].basic_blocks["entry"].instructions[0].clone()
    }

    #[test]
    fn typed_call_argument_parses_like_untyped() {
        let body = "fn @f(i32 %x) -> i32 {\nentry:\n  ret.i32 %x\n}\n\
                    fn @main() -> i32 {\nentry:\n  %v = call @f({arg})\n  ret.i32 %v\n}\n";
        let typed_source = body.replace("{arg}", "i32 7");
        let untyped_source = body.replace("{arg}", "7");
        let typed = parse_module(&typed_source).expect("typed call");
        let untyped = parse_module(&untyped_source).expect("untyped call");
        assert_eq!(typed, untyped);
    }

    #[test]
    fn typed_call_argument_literal_takes_its_stated_width() {
        let source = call_module("call @f(i64 7)");
        let Instruction::Call { args, .. } = &entry_call(&source) else {
            panic!("expected a call");
        };
        assert_eq!(args[0], Value::Constant(Literal::I64(7)));
    }

    #[test]
    fn typed_and_untyped_arguments_mix_in_one_call() {
        let source_mixed = call_module("call @g(7, i64 %p)");
        let mixed = parse_module(&source_mixed).expect("mixed call");
        let source_untyped = call_module("call @g(7, %p)");
        let untyped = parse_module(&source_untyped).expect("untyped call");
        assert_eq!(mixed, untyped);
    }

    #[test]
    fn typed_call_argument_is_checked_against_declared_parameter() {
        let err = call_error("call @f(i32 7)");
        assert!(
            err.contains("Argument 1 of call to @f has type i32, but parameter %x is declared i64"),
            "{err}"
        );
        assert!(err.contains("Write 'i64 <value>'"), "{err}");
    }

    #[test]
    fn typed_call_argument_checks_position_past_the_first() {
        let err = call_error("call @g(i64 1, i32 2)");
        assert!(
            err.contains("Argument 2 of call to @g has type i32, but parameter %b is declared i64"),
            "{err}"
        );
    }

    #[test]
    fn typed_call_argument_to_callee_defined_later_is_checked() {
        let source = "fn @main() -> i64 {\nentry:\n  %v = call @later(bool true)\n  ret.i64 0\n}\n\
                      fn @later(i64 %x) -> i64 {\nentry:\n  ret.i64 %x\n}\n";
        let err = parse_module(source).expect_err("bool for i64 must be rejected");
        assert!(
            err.to_string()
                .contains("has type bool, but parameter %x is declared i64"),
            "{err}"
        );
    }

    #[test]
    fn typed_call_argument_to_extern_is_checked_and_variadic_tail_is_free() {
        let source_ok = call_module("call @printf(ptr $fmt, f64 1.5, i64 %p)");
        let ok = parse_module(&source_ok);
        assert!(ok.is_ok(), "{ok:?}");

        let err = call_error("call @printf(i32 7, i64 1)");
        assert!(
            err.contains(
                "Argument 1 of call to @printf has type i32, but parameter %fmt is declared ptr"
            ),
            "{err}"
        );
    }

    #[test]
    fn typed_call_arguments_display_in_canonical_untyped_form() {
        let typed_source = call_module("call @g(i64 7, i64 %p)");
        let untyped_source = call_module("call @g(7, %p)");
        let typed = parse_module(&typed_source).expect("typed call");
        let untyped = parse_module(&untyped_source).expect("untyped call");
        let typed_text = typed.to_string();
        assert_eq!(typed_text, untyped.to_string());
        assert!(typed_text.contains("%v = call @g(7, %p)"), "{typed_text}");
        assert!(!typed_text.contains("i64 7"), "{typed_text}");
    }

    #[test]
    fn call_argument_missing_value_names_callptr() {
        let err = call_error("call @f(i)");
        assert!(err.contains("optionally preceded by their type"), "{err}");
        assert!(err.contains("callptr.<ret-ty>"), "{err}");
    }

    #[test]
    fn getelem_and_getfieldptr_name_each_other_in_their_errors() {
        let err = parse_error("%e = getelem %buf, 0");
        assert!(
            err.contains("getfieldptr.<struct-type> <ptr>, <index>"),
            "{err}"
        );
        assert!(err.contains("element size"), "{err}");

        let err = parse_error("%f = getfieldptr.i64 %buf, 0");
        assert!(err.contains("needs a struct layout, found i64"), "{err}");
    }

    #[test]
    fn test_removed_char_tuple_and_writeptr_are_rejected() {
        let err = parse_error("%e = add.char 1, 2");
        assert!(err.contains("'i8'"), "{err}");

        let err = parse_error("%e = stackalloc.char");
        assert!(err.contains("'i8'"), "{err}");

        let err = parse_error("%e = writeptr %buf");
        assert!(err.contains("writebyte"), "{err}");

        let err = parse_error("print %e");
        assert!(err.contains("writebyte"), "{err}");
        assert!(err.contains("'write <ptr>, <size>'"), "{err}");

        let err = parse_module(
            "fn @main() -> i64 {\n  entry:\n    %t = stackalloc.tuple i32, f64\n    ret.i64 0\n}\n",
        )
        .expect_err("tuple type should be rejected")
        .to_string();
        assert!(err.contains("struct"), "{err}");
    }

    #[test]
    fn test_getfield_alias_and_untyped_are_rejected() {
        let err = parse_error("%f = getfieldptr %p, 2");
        assert!(
            err.contains("getfieldptr.<struct-type> <ptr>, <index>"),
            "{err}"
        );
        assert!(err.contains("e.g. 'getfieldptr.!Pair %p, 1'"), "{err}");

        let err = parse_error("%f = getfield %p, 2");
        assert!(
            err.contains("getfieldptr.<struct-type> <ptr>, <index>"),
            "{err}"
        );

        let err = parse_error("%f = getfieldptr.ptr %p, 2");
        assert!(err.contains("needs a struct layout, found ptr"), "{err}");
    }

    #[test]
    fn test_getfieldptr_carries_its_layout() {
        let input = entry_body("%f = getfieldptr.struct { a: i8, b: i32 } %p, 1");
        assert!(matches!(
            &entry_instructions(&input)[0],
            Instruction::GetFieldPtr { field_index: 1, struct_type: Type::Struct { fields, .. }, .. }
                if fields.len() == 2
        ));
    }

    #[test]
    fn test_packed_struct_parses_as_packed() {
        let input = entry_body("%f = getfieldptr.packed struct { a: i8, b: i32 } %p, 1");
        assert!(matches!(
            &entry_instructions(&input)[0],
            Instruction::GetFieldPtr {
                field_index: 1,
                struct_type: Type::Struct { packed: true, .. },
                ..
            }
        ));
    }

    #[test]
    fn test_packed_applies_only_to_structs() {
        let err = parse_error("%p = stackalloc.packed [2 x i8]");
        assert!(err.contains("'packed' applies to a struct only"), "{err}");
    }

    #[test]
    fn test_stackalloc_parse() {
        let input = entry_body("%p = stackalloc.i64");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::Alloc {
                alloc_type: AllocType::Stack,
                allocated_ty: Type::Primitive(PrimitiveType::I64),
                ..
            }
        ));

        let input = entry_body("%s = stackalloc.struct { a: i64 }");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::Alloc {
                alloc_type: AllocType::Stack,
                allocated_ty: Type::Struct { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_stackalloc_align_modifier() {
        let input = entry_body("%p = stackalloc.i64 align 16");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::Alloc {
                allocated_ty: Type::Primitive(PrimitiveType::I64),
                align: Some(16),
                ..
            }
        ));

        let input = entry_body("%a = stackalloc.[16 x f64] align 8");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::Alloc { align: Some(8), .. }
        ));

        let input = entry_body("%p = stackalloc.i64 align 32");
        assert!(matches!(
            entry_instructions(&input)[0],
            Instruction::Alloc {
                align: Some(32),
                ..
            }
        ));
    }

    #[test]
    fn test_stackalloc_without_align_leaves_next_line_alone() {
        let input = entry_body("%p = stackalloc.i64");
        let instructions = entry_instructions(&input);
        assert_eq!(instructions.len(), 2);
        assert!(matches!(
            instructions[0],
            Instruction::Alloc { align: None, .. }
        ));
        assert!(matches!(instructions[1], Instruction::Ret { .. }));

        let input = "fn @main() -> i64 {\n  entry:\n    %p = stackalloc.i64\n    jmp align\n  align:\n    ret.i64 0\n}\n";
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("main").expect("Function not found");
        assert!(func.basic_blocks.contains_key("align"));
    }

    #[test]
    fn test_stackalloc_align_rejects_non_power_of_two() {
        let err = parse_error("%p = stackalloc.i64 align 3");
        assert!(err.contains("Invalid alignment: align 3"), "{err}");
        assert!(err.contains("power of two"), "{err}");
    }

    #[test]
    fn test_old_alloc_and_dealloc_spellings_are_rejected() {
        let err = parse_error("%p = alloc.stack i64");
        assert!(err.contains("stackalloc.<type>"), "{err}");

        let err = parse_error("%h = alloc.heap [16 x i8]");
        assert!(err.contains("'stackalloc.<type>'"), "{err}");

        let err = parse_error("dealloc.heap %h");
        assert!(err.contains("call @free(<ptr>)"), "{err}");
    }

    #[test]
    fn test_removed_heap_and_memory_spellings_name_their_replacement() {
        let err = parse_error("%h = heapalloc.i64");
        assert!(
            err.contains("Unknown instruction opcode 'heapalloc'"),
            "{err}"
        );
        assert!(err.contains("call @malloc(<size>)"), "{err}");

        let err = parse_error("heapfree %h");
        assert!(
            err.contains("Unknown instruction opcode 'heapfree'"),
            "{err}"
        );
        assert!(err.contains("call @free(<ptr>)"), "{err}");

        let err = parse_error("memcpy %d, %s, 8");
        assert!(err.contains("Unknown instruction opcode 'memcpy'"), "{err}");
        assert!(err.contains("call @memcpy(<dst>, <src>, <size>)"), "{err}");

        let err = parse_error("memmove %d, %s, 8");
        assert!(
            err.contains("Unknown instruction opcode 'memmove'"),
            "{err}"
        );
        assert!(err.contains("call @memmove(<dst>, <src>, <size>)"), "{err}");

        let err = parse_error("memset %d, 0, 8");
        assert!(err.contains("Unknown instruction opcode 'memset'"), "{err}");
        assert!(
            err.contains("call @memset(<dst>, <value>, <size>)"),
            "{err}"
        );
    }

    #[test]
    fn test_opcode_name_lists_follow_new_spellings() {
        let assignment = get_assignment_opcode_names();
        assert!(assignment.contains(&"stackalloc"));
        assert!(!assignment.contains(&"heapalloc"));
        assert!(!assignment.contains(&"alloc"));
        assert!(!assignment.contains(&"getfield"));
        assert!(!assignment.contains(&"getelementptr"));

        let non_assignment = get_non_assignment_opcode_names();
        assert!(!non_assignment.contains(&"heapfree"));
        assert!(!non_assignment.contains(&"dealloc"));
    }

    #[test]
    fn test_display_of_changed_instructions_round_trips() {
        let source = "fn @malloc(i64 %n) -> ptr {\nentry:\n  %p = stackalloc.i64\n  ret.ptr %p\n}\nfn @free(ptr %p) -> void {\nentry:\n  ret.void\n}\nfn @cb(f32 %x, i64 %n) -> i64 {
entry:
  ret.i64 %n
}

fn @sink(f32 %x) -> void {
entry:
  ret.void
}

fn @main() -> i64 {
entry:
  %p = stackalloc.i64
  %h = call @malloc(16)
  %s = stackalloc.struct { a: i64, b: i64 }
  %f = getfieldptr.struct { a: i64, b: i64 } %s, 1
  %e = getelem.i8 %h, 0
  %a = ptrtoint.i64 %p
  %q = inttoptr.i64 %a
  store.i64 %q, 7
  call @free(%h)
  %fp = funcaddr @cb
  %cr = callptr.i64 %fp(f32 1.5, i64 %a)
  %vp = funcaddr @sink
  callptr.void %vp(f32 2.5)
  ret.i64 0
}
";
        let module = parse_module(source).expect("parse changed instructions");
        let printed = module.to_string();
        let reparsed = parse_module(&printed).expect("parse printed module");
        assert_eq!(reparsed.to_string(), printed);
        assert!(printed.contains("callptr.i64 %fp(f32 1.5, i64 %a)"));
    }

    #[test]
    fn test_untyped_callptr_arguments_are_rejected() {
        let source = "fn @cb(i64 %n) -> i64 {
entry:
  ret.i64 %n
}

fn @main() -> i64 {
entry:
  %fp = funcaddr @cb
  %r = callptr.i64 %fp(7)
  ret.i64 0
}
";
        let err = parse_module(source).expect_err("untyped callptr argument must be rejected");
        let text = err.to_string();
        assert!(
            text.contains("callptr arguments must each carry a type"),
            "{text}"
        );
        assert!(
            text.contains("Hint: Did you mean 'callptr.<ret-ty> <ptr>(<ty> <arg>, <ty> <arg>)'"),
            "{text}"
        );
    }

    #[test]
    fn owned_ir_builder_output_round_trips_through_parser() {
        use crate::ir::{OwnedIRBuilder, OwnedParam, OwnedStructField, OwnedType, OwnedValue};

        let i32t = OwnedType::Primitive(PrimitiveType::I32);
        let i64t = OwnedType::Primitive(PrimitiveType::I64);
        let f64t = OwnedType::Primitive(PrimitiveType::F64);
        let pair = OwnedType::Named("Pair".to_string());
        let var = |n: &str| OwnedValue::Variable(n.to_string());
        let param = |n: &str, ty: &OwnedType| OwnedParam {
            name: n.to_string(),
            ty: ty.clone(),
        };

        let mut b = OwnedIRBuilder::new();
        b.function_with_params("helper", vec![param("x", &i64t)], i64t.clone())
            .ret(&i64t, &var("x"));
        b.function("vfn", OwnedType::Void).ret_void();
        b.external_function("ext", vec![param("c", &i32t)], i32t.clone());
        let ptr_t = OwnedType::Primitive(PrimitiveType::Ptr);
        b.external_function("malloc", vec![param("n", &i64t)], ptr_t.clone());
        b.external_function("free", vec![param("p", &ptr_t)], OwnedType::Void);
        b.external_function(
            "memcpy",
            vec![param("d", &ptr_t), param("s", &ptr_t), param("n", &i64t)],
            ptr_t.clone(),
        );
        b.external_function(
            "memmove",
            vec![param("d", &ptr_t), param("s", &ptr_t), param("n", &i64t)],
            ptr_t.clone(),
        );
        b.external_function(
            "memset",
            vec![param("d", &ptr_t), param("v", &i32t), param("n", &i64t)],
            ptr_t.clone(),
        );
        b.function_with_params(
            "main",
            vec![param("a", &i64t), param("f", &f64t), param("p", &pair)],
            i64t.clone(),
        )
        .binary(
            BinaryOp::Add,
            "sum",
            PrimitiveType::I64,
            &var("a"),
            &OwnedValue::I64(1),
        )
        .binary(
            BinaryOp::Mul,
            "fm",
            PrimitiveType::F64,
            &var("f"),
            &OwnedValue::F64(1.0),
        )
        .cmp(
            CmpOp::Lt,
            "lt",
            PrimitiveType::I64,
            &var("sum"),
            &OwnedValue::I64(10),
        )
        .branch(&var("lt"), "then", "other")
        .block("then")
        .copy("m", &i64t, &var("sum"))
        .jump("merge")
        .block("other")
        .copy("m", &i64t, &OwnedValue::I64(0))
        .jump("merge")
        .block("merge")
        .stackalloc("s", &i64t)
        .stackalloc("ps", &pair)
        .stackalloc(
            "ss",
            &OwnedType::Struct(vec![OwnedStructField {
                name: "a".to_string(),
                ty: i64t.clone(),
            }]),
        )
        .stackalloc(
            "st",
            &OwnedType::Struct(vec![
                OwnedStructField {
                    name: "a".to_string(),
                    ty: i64t.clone(),
                },
                OwnedStructField {
                    name: "b".to_string(),
                    ty: f64t.clone(),
                },
            ]),
        )
        .call(Some("h"), "malloc", &[OwnedValue::I64(8)])
        .store(&i64t, &var("s"), &var("sum"))
        .load("l", &i64t, &var("s"))
        .load("gl", &i64t, &OwnedValue::Global("g".to_string()))
        .getfieldptr("fp", &var("ps"), pair.clone(), 1)
        .store(&i64t, &var("fp"), &OwnedValue::I64(3))
        .getelem("ge", &var("s"), &OwnedValue::I32(1), i32t.clone())
        .getelem(
            "ga",
            &var("s"),
            &OwnedValue::I32(1),
            OwnedType::Array {
                element_type: Box::new(i32t.clone()),
                size: 3,
            },
        )
        .ptrtoint("pi", &var("s"), PrimitiveType::I64)
        .inttoptr("ip", &var("pi"), PrimitiveType::Ptr)
        .zext("z", PrimitiveType::I8, PrimitiveType::I32, &var("l"))
        .sext("sx", PrimitiveType::I8, PrimitiveType::I64, &var("l"))
        .trunc("tr", PrimitiveType::I64, PrimitiveType::I32, &var("sum"))
        .bitcast("bc", PrimitiveType::I64, PrimitiveType::F64, &var("sum"))
        .select("sel", &i64t, &var("lt"), &var("sum"), &OwnedValue::I64(0))
        .call(None, "memcpy", &[var("s"), var("s"), OwnedValue::I64(8)])
        .call(None, "memmove", &[var("s"), var("s"), OwnedValue::I64(8)])
        .call(
            None,
            "memset",
            &[var("s"), OwnedValue::I32(0), OwnedValue::I64(8)],
        )
        .write("w", &var("s"), &OwnedValue::I64(8))
        .read("rd", &var("s"), &OwnedValue::I64(8))
        .write_byte("wb", &OwnedValue::I32(65))
        .read_byte("rb")
        .inttofp("if", PrimitiveType::I64, PrimitiveType::F64, &var("sum"))
        .inttofp(
            "uf",
            PrimitiveType::U8,
            PrimitiveType::F32,
            &OwnedValue::U8(3),
        )
        .fptoint("fi", PrimitiveType::F64, PrimitiveType::U32, &var("f"))
        .fptoint(
            "si",
            PrimitiveType::F32,
            PrimitiveType::I8,
            &OwnedValue::F32(2.5),
        )
        .fpcast("fc", PrimitiveType::F64, PrimitiveType::F32, &var("f"))
        .fpcast(
            "fw",
            PrimitiveType::F32,
            PrimitiveType::F64,
            &OwnedValue::F32(1.5),
        )
        .neg("ng", PrimitiveType::I64, &var("sum"))
        .neg("nf", PrimitiveType::F64, &var("f"))
        .not("nt", PrimitiveType::U16, &OwnedValue::U16(1))
        .sqrt("sq", PrimitiveType::F64, &var("f"))
        .funcaddr("fa", "helper")
        .callptr(
            Some("ci"),
            &i64t,
            &var("fa"),
            &[(i64t.clone(), var("sum"))],
            None,
        )
        .callptr(
            None,
            &OwnedType::Void,
            &var("fa"),
            &[(i32t.clone(), OwnedValue::I32(1)), (f64t.clone(), var("f"))],
            Some(1),
        )
        .call(Some("c"), "helper", &[var("sum")])
        .call(None, "helper", &[OwnedValue::I64(1)])
        .write_byte("w", &var("c"))
        .call(None, "free", &[var("h")])
        .switch(
            PrimitiveType::I64,
            &var("sum"),
            "end",
            &[(1, "a"), (2, "b")],
        )
        .block("a")
        .jump("end")
        .block("b")
        .jump("end")
        .block("end")
        .ret(&i64t, &var("sum"));

        let text = format!(
            "type !Pair = struct {{ x: i64, y: i64 }}\nglobal $g: i64 = 5\n\n{}",
            b.build_ir_text()
        );
        parse_module(&text).unwrap_or_else(|e| panic!("{e}\n---\n{text}"));
    }

    #[test]
    fn builder_conversion_unary_and_call_instructions_round_trip_through_parser() {
        use crate::ir::builder::{i64 as ir_i64_value, var};

        let i64_ty = Type::Primitive(PrimitiveType::I64);
        let f64_ty = Type::Primitive(PrimitiveType::F64);
        let f32_ty = Type::Primitive(PrimitiveType::F32);
        let i32_ty = Type::Primitive(PrimitiveType::I32);
        let ptr_ty = Type::Primitive(PrimitiveType::Ptr);

        let mut b = IRBuilder::new();
        b.function("f", i64_ty.clone())
            .inttofp("a", PrimitiveType::I64, PrimitiveType::F64, var("x"))
            .inttofp("b", PrimitiveType::U32, PrimitiveType::F32, var("x"))
            .fptoint("c", PrimitiveType::F64, PrimitiveType::I8, var("x"))
            .fptoint("d", PrimitiveType::F32, PrimitiveType::U64, var("x"))
            .fpcast("e", PrimitiveType::F64, PrimitiveType::F32, var("x"))
            .fpcast("g", PrimitiveType::F32, PrimitiveType::F64, var("x"))
            .neg("h", PrimitiveType::I32, var("x"))
            .neg("h2", PrimitiveType::F32, var("x"))
            .not("i", PrimitiveType::U8, var("x"))
            .sqrt("j", PrimitiveType::F64, var("x"))
            .funcaddr("k", "helper")
            .callptr(
                Some("l"),
                i64_ty.clone(),
                var("k"),
                vec![(f32_ty.clone(), var("fx")), (i64_ty.clone(), var("x"))],
                None,
            )
            .callptr(
                None,
                Type::Void,
                var("k"),
                vec![(ptr_ty.clone(), var("fmt")), (f64_ty.clone(), var("x"))],
                Some(1),
            )
            .callptr(
                Some("m"),
                i32_ty.clone(),
                var("k"),
                vec![(ptr_ty.clone(), var("fmt"))],
                Some(1),
            )
            .ret(i64_ty.clone(), ir_i64_value(0));

        let module = b.build();
        let built = module.functions["f"].basic_blocks["entry"]
            .instructions
            .clone();
        let body: Vec<String> = built.iter().map(|inst| format!("  {inst}")).collect();
        let text = format!(
            "fn @f(i64 %x, ptr %fmt, f32 %fx) -> i64 {{\nentry:\n{}\n}}",
            body.join("\n")
        );
        let parsed = parse_module(&text).unwrap_or_else(|e| panic!("{e}\n---\n{text}"));
        assert_eq!(
            parsed.functions["f"].basic_blocks["entry"].instructions,
            built
        );
    }

    fn atomic_entry(lines: &[String]) -> String {
        format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n{}\n ret.i64 0\n}}",
            lines
                .iter()
                .map(|line| format!(" {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn atomic_parse_error(line: &str) -> String {
        parse_module(&atomic_entry(&[line.to_string()]))
            .expect_err("atomic source should be rejected")
            .to_string()
    }

    #[test]
    fn atomic_operations_parse_and_display_round_trip() {
        let orderings = ["relaxed", "acquire", "release", "acqrel", "seqcst"];
        let mut lines = Vec::new();
        let mut expected = Vec::new();
        for (index, ordering) in orderings.iter().enumerate() {
            for op in [
                "atomic_add",
                "atomic_sub",
                "atomic_and",
                "atomic_or",
                "atomic_xor",
                "atomic_exchange",
            ] {
                let line = format!("%r{index}_{op} = {op}.i64 %p, 5, {ordering}");
                expected.push(line.clone());
                lines.push(line);
            }
            if !matches!(*ordering, "release" | "acqrel") {
                let line = format!("%load{index} = atomic_load.i64 %p, {ordering}");
                expected.push(line.clone());
                lines.push(line);
            }
            if !matches!(*ordering, "acquire" | "acqrel") {
                let line = format!("atomic_store.i64 %p, 5, {ordering}");
                expected.push(line.clone());
                lines.push(line);
            }
            let line = format!("fence {ordering}");
            expected.push(line.clone());
            lines.push(line);
        }
        lines.push("%narrow = atomic_add.u8 %p, 1, seqcst".to_string());
        expected.push("%narrow = atomic_add.u8 %p, 1, seqcst".to_string());

        let source = atomic_entry(&lines);
        let module = parse_module(&source).expect("parse atomic operations");
        let rendered: Vec<String> = module.functions["f"].basic_blocks["entry"]
            .instructions
            .iter()
            .take(expected.len())
            .map(|i| i.to_string())
            .collect();
        assert_eq!(rendered, expected);

        let rerendered_source = atomic_entry(&rendered);
        let reparsed = parse_module(&rerendered_source).expect("reparse rendered atomics");
        let reparsed_text: Vec<String> = reparsed.functions["f"].basic_blocks["entry"]
            .instructions
            .iter()
            .take(expected.len())
            .map(|i| i.to_string())
            .collect();
        assert_eq!(reparsed_text, expected);
    }

    #[test]
    fn atomic_cmpxchg_parses_and_displays_round_trip() {
        let success = ["relaxed", "acquire", "release", "acqrel", "seqcst"];
        let failure = ["relaxed", "acquire", "seqcst"];
        let mut expected = Vec::new();
        for s in success {
            for f in failure {
                expected.push(format!("%old = atomic_cmpxchg.i32 %p, 1, 2, {s}, {f}"));
            }
        }
        expected.push("%old = atomic_cmpxchg.u8 %p, 255, 0, seqcst, seqcst".to_string());

        let source = atomic_entry(&expected);
        let module = parse_module(&source).expect("parse cmpxchg");
        let rendered: Vec<String> = module.functions["f"].basic_blocks["entry"]
            .instructions
            .iter()
            .take(expected.len())
            .map(|i| i.to_string())
            .collect();
        assert_eq!(rendered, expected);

        let rerendered_source = atomic_entry(&rendered);
        let reparsed = parse_module(&rerendered_source).expect("reparse cmpxchg");
        let reparsed_text: Vec<String> = reparsed.functions["f"].basic_blocks["entry"]
            .instructions
            .iter()
            .take(expected.len())
            .map(|i| i.to_string())
            .collect();
        assert_eq!(reparsed_text, expected);
    }

    #[test]
    fn atomic_cmpxchg_builds_its_instruction_variant() {
        let source =
            atomic_entry(&["%old = atomic_cmpxchg.i64 %p, 7, 8, acquire, relaxed".to_string()]);
        let module = parse_module(&source).expect("parse cmpxchg");
        assert!(matches!(
            &module.functions["f"].basic_blocks["entry"].instructions[0],
            Instruction::AtomicCompareExchange {
                success_ordering: MemoryOrdering::Acquire,
                failure_ordering: MemoryOrdering::Relaxed,
                ..
            }
        ));
    }

    #[test]
    fn atomic_operations_build_their_instruction_variants() {
        let source = atomic_entry(&[
            "%old = atomic_exchange.u32 %p, 9, acqrel".to_string(),
            "atomic_store.i8 %p, 1, release".to_string(),
        ]);
        let module = parse_module(&source).expect("parse atomics");
        let instructions = &module.functions["f"].basic_blocks["entry"].instructions;
        assert!(matches!(
            &instructions[0],
            Instruction::AtomicBinary {
                op: AtomicBinOp::Exchange,
                ordering: MemoryOrdering::AcqRel,
                ..
            }
        ));
        assert!(matches!(
            &instructions[1],
            Instruction::AtomicStore {
                ordering: MemoryOrdering::Release,
                ..
            }
        ));
    }

    #[test]
    fn atomic_operations_reject_invalid_orderings_and_shapes() {
        for (line, expected) in [
            (
                "%r = atomic_load.i64 %p, release",
                "atomic_load cannot use release ordering",
            ),
            (
                "%r = atomic_load.i64 %p, acqrel",
                "atomic_load cannot use acqrel ordering",
            ),
            (
                "atomic_store.i64 %p, 5, acquire",
                "atomic_store cannot use acquire ordering",
            ),
            (
                "atomic_store.i64 %p, 5, acqrel",
                "atomic_store cannot use acqrel ordering",
            ),
            (
                "%r = atomic_add.i64 %p, 5, sequential",
                "Unknown memory ordering 'sequential'",
            ),
            ("%r = atomic_add.i64 %p, 5", "Expected character ','"),
            (
                "%r = atomic_add.f64 %p, 5.0, seqcst",
                "integer or pointer type",
            ),
            (
                "%r, %s = atomic_cmpxchg.i64 %p, 1, 2, seqcst, seqcst",
                "two results are not supported yet",
            ),
            (
                "%r = atomic_cmpxchg.i64 %p, 1, 2, seqcst, release",
                "atomic_cmpxchg cannot use release failure ordering",
            ),
            (
                "%r = atomic_cmpxchg.i64 %p, 1, 2, seqcst, acqrel",
                "atomic_cmpxchg cannot use acqrel failure ordering",
            ),
            (
                "atomic_cmpxchg.i64 %p, 1, 2, seqcst, seqcst",
                "atomic_cmpxchg produces a value",
            ),
            (
                "%r = atomic_cmpxchg.f64 %p, 1.0, 2.0, seqcst, seqcst",
                "integer or pointer type",
            ),
            (
                "atomic_add.i64 %p, 5, seqcst",
                "atomic_add produces a value",
            ),
            (
                "%r = atomic_store.i64 %p, 5, seqcst",
                "atomic_store produces no result",
            ),
            ("%r = fence seqcst", "fence produces no result"),
        ] {
            let message = atomic_parse_error(line);
            assert!(message.contains(expected), "{line}: {message}");
        }
    }

    #[test]
    fn volatile_prefix_parses_and_displays_on_load_and_store() {
        let source = "fn @f(ptr %p) -> i64 {\nentry:\n %x = volatile load.i64 %p\n volatile store.i64 %p, 5\n %y = load.i64 %p\n store.i64 %p, 6\n ret.i64 %x\n}";
        let module = parse_module(source).expect("parse volatile access");
        let instructions = &module.functions["f"].basic_blocks["entry"].instructions;
        let rendered: Vec<String> = instructions.iter().map(|i| i.to_string()).collect();
        assert_eq!(
            rendered[..4],
            [
                "%x = volatile load.i64 %p",
                "volatile store.i64 %p, 5",
                "%y = load.i64 %p",
                "store.i64 %p, 6",
            ]
        );
        assert!(matches!(
            instructions[0],
            Instruction::Load { volatile: true, .. }
        ));
        assert!(matches!(
            instructions[2],
            Instruction::Load {
                volatile: false,
                ..
            }
        ));
        let reparsed_source = format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n {}\n ret.i64 %x\n}}",
            rendered[..4].join("\n ")
        );
        let reparsed = parse_module(&reparsed_source).expect("reparse rendered volatile access");
        let reparsed_text: Vec<String> = reparsed.functions["f"].basic_blocks["entry"]
            .instructions
            .iter()
            .map(|i| i.to_string())
            .collect();
        assert_eq!(reparsed_text[..4], rendered[..4]);
    }

    #[test]
    fn volatile_prefix_rejects_opcodes_other_than_load_and_store() {
        for (source, expected) in [
            (
                "fn @f() -> i64 { entry: %x = volatile add.i64 1, 2 ret.i64 %x }",
                "applies to load and store only",
            ),
            (
                "fn @f(ptr %p) -> i64 { entry: volatile ret.i64 0 }",
                "applies to load and store only",
            ),
            (
                "fn @f(ptr %p) -> i64 { entry: %x = volatile store.i64 %p, 5 ret.i64 0 }",
                "store produces no result",
            ),
        ] {
            let Err(error) = parse_module(source) else {
                panic!("volatile must be rejected: {source}")
            };
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}

#[cfg(test)]
mod surface_tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ir::Instruction;

    fn round_trip(source: &str) {
        let module = parse_module(source).expect("parse canonical source");
        assert_eq!(module.to_string(), source);
    }

    fn rejects(source: &str, expected: &str) {
        let error = parse_module(source).expect_err("source should be rejected");
        assert!(error.to_string().contains(expected), "{error}");
    }

    #[test]
    fn wide_integer_types_round_trip() {
        round_trip(
            "fn @wide(i128 %x, u128 %y) -> u128 {\n\
             entry:\n  \
             %s = add.i128 %x, -170141183460469231731687303715884105728\n  \
             %d = sub.u128 %y, 340282366920938463463374607431768211455\n  \
             %m = mul.i128 %s, 18446744073709551617\n  \
             %q = div.i128 %m, %x\n  \
             %r = rem.u128 %y, 3\n  \
             %l = lt.i128 %q, 0\n  \
             %h = shr.u128 %y, 64\n  \
             %z = extend.u64.u128 %h\n  \
             %t = trunc.i128.i32 %q\n  \
             %f = inttofp.i128.f64 %x\n  \
             ret.u128 %z\n\
             }\n",
        );
    }

    #[test]
    fn wide_integer_literals_respect_their_width() {
        rejects(
            "fn @f() -> i128 {\nentry:\n  %x = add.i128 170141183460469231731687303715884105728, 0\n  ret.i128 %x\n}\n",
            "too large",
        );
        rejects(
            "fn @f() -> u128 {\nentry:\n  %x = add.u128 -1, 0\n  ret.u128 %x\n}\n",
            "Expected an unsigned integer literal",
        );
    }

    #[test]
    fn bit_intrinsics_round_trip() {
        round_trip(
            "fn @f(i64 %x, i64 %n, u8 %b, u32 %w) -> i64 {\n\
             entry:\n  \
             %p = popcount.i64 %x\n  \
             %c = ctlz.i64 %x\n  \
             %t = cttz.u8 %b\n  \
             %s = bswap.u32 %w\n  \
             %l = rotl.i64 %x, %n\n  \
             %r = rotr.u8 %b, 3\n  \
             ret.i64 %p\n\
             }\n",
        );
    }

    #[test]
    fn variadic_instructions_round_trip() {
        round_trip(
            "fn @f(i64 %n, ...) -> i64 {\n\
             entry:\n  \
             %ap = stackalloc.valist\n  \
             %bp = stackalloc.valist\n  \
             va_start %ap\n  \
             %x = va_arg.i64 %ap\n  \
             %d = va_arg.f64 %ap\n  \
             va_copy %bp, %ap\n  \
             va_end %ap\n  \
             ret.i64 %x\n\
             }\n",
        );
    }

    #[test]
    fn bit_intrinsics_reject_non_integer_types() {
        rejects(
            "fn @f(f64 %x) -> f64 {\nentry:\n  %r = popcount.f64 %x\n  ret.f64 %r\n}\n",
            "popcount takes an integer type",
        );
        rejects(
            "fn @f(f32 %x, i32 %n) -> f32 {\nentry:\n  %r = rotl.f32 %x, %n\n  ret.f32 %r\n}\n",
            "rotl takes an integer type",
        );
        rejects(
            "fn @f(ptr %x) -> i64 {\nentry:\n  %r = bswap.ptr %x\n  ret.i64 0\n}\n",
            "bswap takes an integer type",
        );
    }

    #[test]
    fn unreachable_round_trips_as_terminator() {
        round_trip("fn @f() -> void {\nentry:\n  unreachable\n}\n");
        let module = parse_module(
            "fn @f(i64 %x) -> i64 {\nentry:\n  %c = eq.i64 %x, 0\n  br %c, zero, other\nzero:\n  ret.i64 1\nother:\n  unreachable\n}\n",
        )
        .expect("parse unreachable block");
        assert!(matches!(
            module.functions["f"].basic_blocks["other"]
                .instructions
                .last(),
            Some(Instruction::Unreachable)
        ));
    }

    #[test]
    fn function_placement_annotations_round_trip() {
        round_trip(
            "@weak\n@section(\".text.hot\")\n@align(32)\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
        );
    }

    #[test]
    fn function_placement_annotations_reject_bad_arguments() {
        rejects(
            "@align(3)\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "power of two",
        );
        rejects(
            "@section(\"\")\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "Invalid section name",
        );
        rejects(
            "@section(\".a\")\n@section(\".b\")\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "Conflicting annotations",
        );
        rejects(
            "@align(8)\n@align(16)\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "Conflicting annotations",
        );
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn module_annotations_round_trip() {
        round_trip("@pic\n\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n");
        round_trip("@pie\n\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n");
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn module_annotations_reject_conflicts() {
        rejects(
            "@pic\n@pie\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "Conflicting module annotations",
        );
        rejects(
            "@pic\n@pic\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "Duplicate module annotation",
        );
    }

    #[test]
    fn removed_module_annotations_are_rejected() {
        let cases = [
            ("optimize_speed", "use --opt-level (0-3)"),
            ("optimize_size", "use --opt-level (0-3)"),
            ("debug", "use -g"),
            ("strip", "use the strip tool on the output"),
            ("target_triple", "use --target"),
        ];
        for (name, replacement) in cases {
            let source = if name == "target_triple" {
                format!(
                    "@{name}(\"x86_64-unknown-linux-gnu\")\nfn @f() -> i64 {{\nentry:\n  ret.i64 1\n}}\n"
                )
            } else {
                format!("@{name}\nfn @f() -> i64 {{\nentry:\n  ret.i64 1\n}}\n")
            };
            rejects(&source, &format!("@{name} was removed; {replacement}"));
        }
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn module_annotations_require_nightly() {
        rejects(
            "@pic\nfn @f() -> i64 {\nentry:\n  ret.i64 1\n}\n",
            "only available with the nightly feature",
        );
    }
}

#[cfg(test)]
mod bit_builder_tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ir::{
        FunctionParameter, IRBuilder, Instruction, OwnedIRBuilder, OwnedParam, OwnedType,
        OwnedValue, Type, builder::var,
    };
    use std::panic;

    #[test]
    fn bit_intrinsic_builders_agree_with_text_syntax() {
        let i64_ty = Type::Primitive(PrimitiveType::I64);
        let mut b = IRBuilder::new();
        b.function_with_params(
            "f",
            vec![
                FunctionParameter {
                    name: "x",
                    ty: i64_ty.clone(),
                    annotations: vec![],
                },
                FunctionParameter {
                    name: "n",
                    ty: i64_ty.clone(),
                    annotations: vec![],
                },
            ],
            i64_ty.clone(),
        )
        .popcount("p", PrimitiveType::I64, var("x"))
        .ctlz("l", PrimitiveType::U8, var("x"))
        .cttz("t", PrimitiveType::I32, var("x"))
        .bswap("s", PrimitiveType::U16, var("x"))
        .rotl("a", PrimitiveType::I64, var("x"), var("n"))
        .rotr("b", PrimitiveType::U32, var("x"), var("n"))
        .unreachable();
        let built = b.build();
        let built = built.functions["f"].basic_blocks["entry"]
            .instructions
            .clone();

        let mut o = OwnedIRBuilder::new();
        let x = OwnedValue::Variable("x".to_string());
        let n = OwnedValue::Variable("n".to_string());
        let i64_owned = OwnedType::Primitive(PrimitiveType::I64);
        o.function_with_params(
            "f",
            vec![
                OwnedParam {
                    name: "x".to_string(),
                    ty: i64_owned.clone(),
                },
                OwnedParam {
                    name: "n".to_string(),
                    ty: i64_owned.clone(),
                },
            ],
            i64_owned,
        )
        .popcount("p", PrimitiveType::I64, &x)
        .ctlz("l", PrimitiveType::U8, &x)
        .cttz("t", PrimitiveType::I32, &x)
        .bswap("s", PrimitiveType::U16, &x)
        .rotl("a", PrimitiveType::I64, &x, &n)
        .rotr("b", PrimitiveType::U32, &x, &n)
        .unreachable();
        let text = o.build_ir_text();
        let parsed = parse_module(&text).unwrap_or_else(|e| panic!("{e}\n---\n{text}"));
        assert_eq!(
            parsed.functions["f"].basic_blocks["entry"].instructions,
            built
        );
        assert!(matches!(built.last(), Some(Instruction::Unreachable)));
    }

    #[test]
    fn bit_intrinsic_builders_reject_floats() {
        let result = panic::catch_unwind(|| {
            let mut b = IRBuilder::new();
            b.function("f", Type::Void)
                .popcount("p", PrimitiveType::F64, var("x"));
        });
        assert!(result.is_err());
    }

    #[test]
    fn redefinition_with_different_type_is_rejected_at_parse() {
        let print_reuse = r#"
fn @neg() -> i64 { entry: %x = add.i64 0, -5  ret.i64 %x }
fn @big() -> u64 { entry: %y = add.u64 0, 3   ret.u64 %y }
fn @main() -> i64 {
  entry:
    %v = call @neg()
    %w = writebyte %v
    %v = call @big()
    %x = writebyte %v
    ret.i64 0
}
"#;
        let err = parse_module(print_reuse).expect_err("type-changing redefinition");
        let message = err.to_string();
        assert!(message.contains("'%v'"), "{message}");
        assert!(message.contains("first defined as i64"), "{message}");
        assert!(message.contains("redefined as u64"), "{message}");

        let wrong_answer = r#"
fn @main() -> i64 {
  entry:
    %v = add.i128 0, 5
    %v = add.i64 0, 7
    %w = add.i128 %v, 1
    %lo = trunc.i128.i64 %w
    ret.i64 %lo
}
"#;
        let err = parse_module(wrong_answer).expect_err("i128 then i64");
        assert!(err.to_string().contains("'%v'"));
    }

    #[test]
    fn same_type_reassignment_still_parses() {
        let source = r#"
fn @main() -> i64 {
  entry:
    %v = add.i64 0, 5
    %v = add.i64 %v, 7
    %w = writebyte %v
    ret.i64 %v
}
"#;
        assert!(parse_module(source).is_ok());
    }
}
