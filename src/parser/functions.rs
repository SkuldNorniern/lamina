//! Function parsing for Lamina IR.

use crate::{
    Label, LaminaError,
    ir::{
        BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
        Instruction, SourceLoc, Type,
    },
    parser::{instructions::parse_instruction, state::ParserState, types::parse_type},
};
use lamina_ir::function::VariableAnnotation;
use std::collections::{HashMap, HashSet};

/// Parses the `(n)` of an `@align(n)` annotation. `n` must be a power of two that fits u32.
pub fn parse_alignment_argument(state: &mut ParserState<'_>) -> Result<u32, LaminaError> {
    state.expect_char('(')?;
    let bytes = state.parse_integer()?;
    state.expect_char(')')?;
    alignment_from_bytes(bytes).ok_or_else(|| {
        state.error(format!(
            "Invalid alignment: @align({bytes})\n  Hint: Alignment must be a power of two, e.g. @align(16)"
        ))
    })
}

/// The alignment in bytes when `bytes` is a power of two that fits in a `u32`.
pub fn alignment_from_bytes(bytes: i64) -> Option<u32> {
    u32::try_from(bytes)
        .ok()
        .filter(|alignment| alignment.is_power_of_two())
}

/// The hint for an unknown annotation: a did-you-mean when a name is close, else the full list.
pub fn unknown_annotation_hint(name: &str, valid: &[&str]) -> String {
    let suggestions = super::suggest_alternatives(name, valid);
    if !suggestions.is_empty() {
        format!("Did you mean @{}?", suggestions.join(" or @"))
    } else {
        let list = valid
            .iter()
            .map(|s| format!("@{s}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Valid annotations are: {list}")
    }
}

/// Parses function annotations.
pub fn parse_annotations(
    state: &mut ParserState<'_>,
) -> Result<Vec<FunctionAnnotation>, LaminaError> {
    let mut annotations = Vec::new();
    let mut seen = HashSet::new();

    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('@') {
            state.advance();
            let name = state.parse_identifier_str()?;
            let annotation = match name {
                "kernel" => FunctionAnnotation::Kernel,
                "inline" => FunctionAnnotation::Inline,
                "export" => FunctionAnnotation::Export,
                "extern" => FunctionAnnotation::Extern,
                "noreturn" => FunctionAnnotation::NoReturn,
                "noinline" => FunctionAnnotation::NoInline,
                "cold" => FunctionAnnotation::Cold,
                "hot" => FunctionAnnotation::Hot,
                "pure" => FunctionAnnotation::Pure,
                "const" => {
                    return Err(state.error(
                        "@const was removed\n  Hint: Use @pure, which marks a function with no side effects",
                    ));
                }
                "internal" => FunctionAnnotation::Internal,
                "private" => FunctionAnnotation::Private,
                "hidden" => FunctionAnnotation::Hidden,
                "protected" => FunctionAnnotation::Protected,
                "weak" => FunctionAnnotation::Weak,
                "section" => {
                    state.expect_char('(')?;
                    let name = state.parse_string_literal()?;
                    state.expect_char(')')?;
                    if name.is_empty() || name.contains('\\') {
                        return Err(state.error(
                            "Invalid section name\n  Hint: Give a non-empty name without backslashes, e.g. @section(\".text.hot\")",
                        ));
                    }
                    FunctionAnnotation::Section(name.to_string())
                }
                "align" => FunctionAnnotation::Align(parse_alignment_argument(state)?),
                "naked" => FunctionAnnotation::Naked,
                "no_red_zone" => {
                    return Err(state.error(
                        "@no_red_zone was removed\n  Hint: Delete it. Nothing replaces it: the x86_64 backend never uses the red zone, and other targets have none",
                    ));
                }
                _ => {
                    let hint = unknown_annotation_hint(name, super::get_annotation_names());
                    return Err(state.error(format!(
                        "Unknown function annotation: @{name}\n  Hint: {hint}"
                    )));
                }
            };

            if let Some(other) = seen.iter().find(|other| conflicts(&annotation, other)) {
                return Err(state.error(format!(
                    "Conflicting annotations: {annotation} and {other} cannot be used together"
                )));
            }

            if !seen.insert(annotation.clone()) {
                return Err(state.error(format!("Duplicate annotation: @{name}")));
            }

            if annotation == FunctionAnnotation::Inline
                && seen.contains(&FunctionAnnotation::NoInline)
            {
                return Err(state.error(
                    "Conflicting annotations: @inline and @noinline cannot be used together",
                ));
            }
            if annotation == FunctionAnnotation::NoInline
                && seen.contains(&FunctionAnnotation::Inline)
            {
                return Err(state.error(
                    "Conflicting annotations: @noinline and @inline cannot be used together",
                ));
            }
            if annotation == FunctionAnnotation::Extern
                && seen.contains(&FunctionAnnotation::Export)
            {
                return Err(state.error(
                    "Conflicting annotations: @extern and @export cannot be used together",
                ));
            }
            if annotation == FunctionAnnotation::Export
                && seen.contains(&FunctionAnnotation::Extern)
            {
                return Err(state.error(
                    "Conflicting annotations: @export and @extern cannot be used together",
                ));
            }

            annotations.push(annotation);
        } else {
            break;
        }
    }
    Ok(annotations)
}

fn conflicts(a: &FunctionAnnotation, b: &FunctionAnnotation) -> bool {
    use FunctionAnnotation::{
        Align, Cold, Export, Hidden, Hot, Inline, Internal, Naked, Private, Protected, Pure,
        Section,
    };
    let visibility = |x: &FunctionAnnotation| matches!(x, Internal | Private | Hidden | Protected);
    let local = |x: &FunctionAnnotation| matches!(x, Internal | Private);
    a != b
        && ((visibility(a) && visibility(b))
            || (matches!((a, b), (Cold, Hot) | (Hot, Cold)))
            || matches!((a, b), (Section(_), Section(_)) | (Align(_), Align(_)))
            || (local(a) && *b == Export)
            || (*a == Export && local(b))
            || matches!((a, b), (Naked, Inline | Pure) | (Inline | Pure, Naked)))
}

/// Parses a function definition.
pub fn parse_function_def<'a>(state: &mut ParserState<'a>) -> Result<Function<'a>, LaminaError> {
    let annotations = parse_annotations(state)?;
    state.skip_whitespace_and_comments();
    let decl_loc = state.source_loc();
    state.consume_keyword("fn")?;
    let name = state.parse_function_identifier()?;
    let signature = parse_fn_signature(state)?;

    state.skip_whitespace_and_comments();
    let declaration_only =
        annotations.contains(&FunctionAnnotation::Extern) && state.current_char() != Some('{');
    if declaration_only {
        return Ok(Function {
            name,
            signature,
            annotations,
            basic_blocks: HashMap::new(),
            entry_block: "",
            decl_loc,
        });
    }

    if annotations.contains(&FunctionAnnotation::Extern) {
        return Err(state.error(format!(
            "@extern function '{name}' has a body\n  Hint: @extern declares a symbol defined elsewhere; drop the body to declare it, or drop @extern to define it here"
        )));
    }

    state.expect_char('{')?;

    let mut basic_blocks = HashMap::new();
    let mut entry_block_label: Option<Label<'a>> = None;

    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('}') {
            state.advance();
            break;
        }

        let (label, block) = parse_basic_block(state)?;

        if entry_block_label.is_none() {
            entry_block_label = Some(label);
        }

        if basic_blocks.insert(label, block).is_some() {
            return Err(state.error(format!("Redefinition of basic block label: '{label}'")));
        }
    }

    let entry_block = entry_block_label
        .ok_or_else(|| state.error("Function must have at least one basic block"))?;

    if annotations.contains(&FunctionAnnotation::Naked) {
        validate_naked_body(state, name, &signature, &basic_blocks)?;
    }

    Ok(Function {
        name,
        signature,
        annotations,
        basic_blocks,
        entry_block,
        decl_loc,
    })
}

/// Rejects a `@naked` function whose body would need a prologue, epilogue or values.
fn validate_naked_body(
    state: &ParserState<'_>,
    name: &str,
    signature: &FunctionSignature<'_>,
    basic_blocks: &HashMap<Label<'_>, BasicBlock<'_>>,
) -> Result<(), LaminaError> {
    if !signature.params.is_empty() || signature.variadic {
        return Err(state.error(format!(
            "@naked function '{name}' takes parameters\n  Hint: A naked function has no argument setup; read the registers in inline asm"
        )));
    }
    if signature.return_type != Type::Void {
        return Err(state.error(format!(
            "@naked function '{name}' returns a value\n  Hint: A naked function returns void; the asm writes the result where it needs to go"
        )));
    }
    let [block] = basic_blocks.values().collect::<Vec<_>>()[..] else {
        return Err(state.error(format!(
            "@naked function '{name}' must have exactly one basic block\n  Hint: Write the whole body as one block of inline asm ending in unreachable"
        )));
    };
    let (last, body) = block
        .instructions
        .split_last()
        .ok_or_else(|| state.error(format!("@naked function '{name}' has an empty body")))?;
    if !matches!(last, Instruction::Unreachable) {
        return Err(state.error(format!(
            "@naked function '{name}' must end in unreachable\n  Hint: Let the asm leave the function (iret, jmp, hlt loop); unreachable marks the end"
        )));
    }
    for inst in body {
        match inst {
            Instruction::InlineAsm {
                result: None,
                operands,
                ..
            } if operands.is_empty() => {}
            Instruction::InlineAsm { .. } => {
                return Err(state.error(format!(
                    "@naked function '{name}' uses inline asm with operands or a result\n  Hint: A naked body computes no IR values; name the registers in the template"
                )));
            }
            other => {
                return Err(state.error(format!(
                    "@naked function '{name}' contains '{other}'\n  Hint: A naked body is inline asm only, because any other instruction expects a frame"
                )));
            }
        }
    }
    Ok(())
}

/// Parses a function signature.
pub fn parse_fn_signature<'a>(
    state: &mut ParserState<'a>,
) -> Result<FunctionSignature<'a>, LaminaError> {
    state.expect_char('(')?;
    let (params, variadic) = parse_param_list(state)?;
    state.expect_char(')')?;
    state.consume_keyword("->")?;
    let sret = consume_return_marker(state)?;
    let return_type = parse_type(state)?;
    Ok(FunctionSignature {
        params,
        return_type,
        variadic,
        sret,
    })
}

/// Consumes a `sret` marker sitting between `->` and the return type.
fn consume_return_marker(state: &mut ParserState<'_>) -> Result<bool, LaminaError> {
    state.skip_whitespace_and_comments();
    let saved = state.position();
    if !matches!(state.current_char(), Some(c) if c.is_ascii_alphabetic() || c == '_') {
        return Ok(false);
    }
    let word = state.parse_identifier_str()?;
    if word != "sret" {
        state.set_position(saved);
        return Ok(false);
    }
    Ok(true)
}

/// Parses a parameter list, and whether it ends in `...`.
pub fn parse_param_list<'a>(
    state: &mut ParserState<'a>,
) -> Result<(Vec<FunctionParameter<'a>>, bool), LaminaError> {
    let mut params = Vec::new();
    let mut param_names = HashSet::new();

    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }

        if state.peek_slice(3) == Some("...") {
            state.advance_by(3);
            state.skip_whitespace_and_comments();
            if state.current_char() != Some(')') {
                return Err(state.error("... must be the last parameter"));
            }
            return Ok((params, true));
        }

        let param_ty = parse_type(state)?;
        let annotations = parse_param_annotations(state)?;
        let param_name = state.parse_value_identifier()?;

        if !param_names.insert(param_name) {
            return Err(state.error(format!("Duplicate parameter name: %{param_name}")));
        }

        params.push(FunctionParameter {
            name: param_name,
            ty: param_ty,
            annotations,
        });

        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }
        state.expect_char(',')?;
    }
    Ok((params, false))
}

const PARAM_ANNOTATIONS: &[&str] = &["byval"];

/// Parses the attributes a parameter carries between its type and its name.
fn parse_param_annotations(
    state: &mut ParserState<'_>,
) -> Result<Vec<VariableAnnotation>, LaminaError> {
    let mut annotations = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if !matches!(state.current_char(), Some(c) if c.is_ascii_alphabetic() || c == '_') {
            return Ok(annotations);
        }
        let name = state.parse_identifier_str()?;
        let annotation = match name {
            "byval" => {
                state.expect_char('(')?;
                let size = state.parse_integer()?;
                state.expect_char(',')?;
                let align = state.parse_integer()?;
                state.expect_char(')')?;
                let size = u64::try_from(size).ok().filter(|size| *size > 0);
                let align = u64::try_from(align)
                    .ok()
                    .filter(|align| align.is_power_of_two());
                match (size, align) {
                    (Some(size), Some(align)) => VariableAnnotation::ByVal { size, align },
                    _ => {
                        return Err(state.error(
                            "Invalid byval arguments\n  Hint: Write byval(size, align) with a positive size and a power-of-two alignment, e.g. byval(16, 8)",
                        ));
                    }
                }
            }
            _ => {
                let suggestions = super::suggest_alternatives(name, PARAM_ANNOTATIONS);
                let hint = if suggestions.is_empty() {
                    "Valid parameter attributes are: byval(size, align)".to_string()
                } else {
                    format!("Did you mean {}?", suggestions.join(" or "))
                };
                return Err(state.error(format!(
                    "Unknown parameter attribute: {name}\n  Hint: {hint}"
                )));
            }
        };
        if annotations.contains(&annotation) {
            return Err(state.error(format!("Duplicate parameter attribute: {name}")));
        }
        annotations.push(annotation);
    }
}

fn at_location_directive(state: &ParserState<'_>) -> bool {
    state.peek_slice(4) == Some(".loc")
        && state
            .peek_char(4)
            .is_some_and(|ch| ch.is_ascii_whitespace())
}

fn parse_location_directive(state: &mut ParserState<'_>) -> Result<SourceLoc, LaminaError> {
    state.consume_keyword(".loc")?;
    let file = parse_location_number(state, "file id")?;
    let line = parse_location_number(state, "line")?;
    let column = parse_location_number(state, "column")?;
    SourceLoc::in_file(file, line, column).ok_or_else(|| {
        state.error(
            "Invalid '.loc': file ids and line numbers start at 1\n  Hint: write '.loc <file> <line> <column>'",
        )
    })
}

pub fn parse_location_number(state: &mut ParserState<'_>, what: &str) -> Result<u32, LaminaError> {
    let value = state.parse_u64()?;
    u32::try_from(value).map_err(|_| {
        state.error(format!(
            "Invalid {what}: {value}\n  Hint: {what} must fit in 32 bits"
        ))
    })
}

/// Parses a basic block.
///
/// A basic block consists of a label followed by a colon, then zero or more
/// instructions ending with a terminator (ret, jmp, or br).
pub fn parse_basic_block<'a>(
    state: &mut ParserState<'a>,
) -> Result<(Label<'a>, BasicBlock<'a>), LaminaError> {
    let label = state.parse_label_identifier()?;
    state.expect_char(':')?;

    let mut instructions = Vec::new();
    let mut instruction_locs = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.at_block_label() {
            break;
        }

        if state.current_char() == Some('}') {
            break;
        }

        let explicit_loc = if at_location_directive(state) {
            let loc = parse_location_directive(state)?;
            state.skip_whitespace_and_comments();
            if state.at_block_label() || state.current_char() == Some('}') {
                return Err(state.error(
                    "A '.loc' must be followed by an instruction\n  Hint: put '.loc <file> <line> <column>' on the line before the instruction it positions",
                ));
            }
            Some(loc)
        } else {
            None
        };

        let loc = explicit_loc.or_else(|| state.source_loc());
        let instruction = parse_instruction(state)?;
        let is_terminator = instruction.is_terminator();
        instructions.push(instruction);
        instruction_locs.push(loc);

        if is_terminator {
            state.skip_whitespace_and_comments();
            if !state.is_eof() && state.current_char() != Some('}') && !state.at_block_label() {
                return Err(state.error(format!(
                    "Instruction after the terminator of block '{label}'\n  Hint: A block ends at its first terminator (ret, jmp, br); start a new labeled block for the instruction"
                )));
            }
            break;
        }
    }

    if instructions.is_empty() || instructions.last().is_none_or(|last| !last.is_terminator()) {
        return Err(state.error(format!(
            "Basic block '{label}' must end with a terminator instruction (ret, jmp, br)"
        )));
    }

    Ok((
        label,
        BasicBlock {
            instructions,
            instruction_locs,
        },
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use crate::{ir::FunctionAnnotation, parser::parse_module};

    fn annotations_of(source: &str) -> Vec<FunctionAnnotation> {
        let module = parse_module(source).expect("parse annotated function");
        module
            .functions
            .values()
            .next()
            .expect("one function")
            .annotations
            .clone()
    }

    #[test]
    fn const_annotation_is_removed_in_favour_of_pure() {
        let source = "@const\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n";
        let err = parse_module(source)
            .expect_err("@const should be rejected")
            .to_string();
        assert!(err.contains("@const was removed"), "{err}");
        assert!(err.contains("@pure"), "{err}");
    }

    #[test]
    fn extern_with_a_body_is_rejected() {
        let source = "@extern\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n";
        let err = parse_module(source)
            .expect_err("@extern names a symbol defined elsewhere")
            .to_string();
        assert!(err.contains("has a body"), "{err}");
        parse_module("@extern\nfn @g() -> i64\n").expect("@extern without a body declares");
    }

    #[test]
    fn visibility_and_placement_annotations_parse() {
        let cases = [
            ("hot", FunctionAnnotation::Hot),
            ("pure", FunctionAnnotation::Pure),
            ("internal", FunctionAnnotation::Internal),
            ("private", FunctionAnnotation::Private),
            ("hidden", FunctionAnnotation::Hidden),
            ("protected", FunctionAnnotation::Protected),
        ];
        for (name, expected) in cases {
            let source = format!("@{name}\nfn @f() -> i64 {{\nentry:\n ret.i64 1\n}}\n");
            assert_eq!(annotations_of(&source), vec![expected], "@{name}");
        }
    }

    #[test]
    fn conflicting_annotations_are_rejected() {
        let cases = [
            "@internal\n@hidden\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n",
            "@cold\n@hot\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n",
            "@private\n@export\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n",
        ];
        for source in cases {
            let error = parse_module(source).expect_err("conflicting annotations");
            assert!(
                error.to_string().contains("Conflicting annotations"),
                "{error}"
            );
        }
    }

    #[test]
    fn hidden_and_protected_may_combine_with_export() {
        let source = "@export\n@hidden\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n";
        assert_eq!(annotations_of(source).len(), 2);
    }

    fn naked_error(source: &str) -> String {
        parse_module(source)
            .expect_err("naked body must be rejected")
            .to_string()
    }

    #[test]
    fn naked_with_asm_body_parses() {
        let source =
            "@naked\nfn @isr() -> void {\nentry:\n    asm.void \"cli; hlt\"\n    unreachable\n}\n";
        assert_eq!(annotations_of(source), vec![FunctionAnnotation::Naked]);
    }

    #[test]
    fn no_red_zone_is_removed_with_no_replacement() {
        let source = "@no_red_zone\nfn @f() -> i64 {\nentry:\n ret.i64 1\n}\n";
        let err = parse_module(source)
            .expect_err("@no_red_zone should be rejected")
            .to_string();
        assert!(err.contains("@no_red_zone was removed"), "{err}");
        assert!(err.contains("Nothing replaces it"), "{err}");
    }

    #[test]
    fn naked_body_rejects_ir_instructions_and_terminators() {
        let cases = [
            "@naked\nfn @f() -> void {\nentry:\n    asm.void \"cli\"\n    ret.void\n}\n",
            "@naked\nfn @f() -> void {\nentry:\n    %x = add.i64 1, 2\n    unreachable\n}\n",
            "@naked\nfn @f() -> void {\nentry:\n    asm.void \"cli\"\n    jmp next\nnext:\n    unreachable\n}\n",
        ];
        for source in cases {
            let message = naked_error(source);
            assert!(message.contains("@naked function 'f'"), "{message}");
        }
    }

    #[test]
    fn naked_body_must_end_in_unreachable() {
        let message = naked_error(
            "@naked\nfn @f() -> void {\nentry:\n    asm.void \"hlt\"\n    ret.void\n}\n",
        );
        assert!(message.contains("must end in unreachable"), "{message}");
    }

    #[test]
    fn naked_signature_must_take_no_parameters_and_return_void() {
        let with_param = naked_error(
            "@naked\nfn @f(i64 %a) -> void {\nentry:\n    asm.void \"cli\"\n    unreachable\n}\n",
        );
        assert!(with_param.contains("takes parameters"), "{with_param}");
        let with_return = naked_error(
            "@naked\nfn @f() -> i64 {\nentry:\n    asm.void \"cli\"\n    unreachable\n}\n",
        );
        assert!(with_return.contains("returns a value"), "{with_return}");
    }

    #[test]
    fn naked_asm_with_operands_or_result_is_rejected() {
        let operands = naked_error(
            "@naked\nfn @f() -> void {\nentry:\n    %v = asm.i64 \"movq $$1, $0\", out(rax)\n    unreachable\n}\n",
        );
        assert!(operands.contains("operands or a result"), "{operands}");
    }

    #[test]
    fn naked_conflicts_with_inline_and_pure() {
        for other in ["@inline", "@pure"] {
            let source =
                format!("@naked\n{other}\nfn @f() -> void {{\nentry:\n    unreachable\n}}\n");
            let message = naked_error(&source);
            assert!(message.contains("Conflicting annotations"), "{message}");
        }
    }
}
