//! Function parsing for Lamina IR.

use crate::{
    Label, LaminaError,
    ir::{BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature},
    parser::{instructions::parse_instruction, state::ParserState, types::parse_type},
};
use std::collections::{HashMap, HashSet};

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
                "const" => FunctionAnnotation::Const,
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
                "align" => {
                    state.expect_char('(')?;
                    let bytes = state.parse_integer()?;
                    state.expect_char(')')?;
                    match u32::try_from(bytes) {
                        Ok(alignment) if alignment.is_power_of_two() => {
                            FunctionAnnotation::Align(alignment)
                        }
                        _ => {
                            return Err(state.error(format!(
                                "Invalid alignment: @align({bytes})\n  Hint: Alignment must be a power of two, e.g. @align(16)"
                            )));
                        }
                    }
                }
                _ => {
                    let valid = super::get_annotation_names();
                    let suggestions = super::suggest_alternatives(name, valid);

                    let hint = if !suggestions.is_empty() {
                        if suggestions.len() == 1 {
                            format!("Did you mean @{}?", suggestions[0])
                        } else {
                            format!("Did you mean @{}?", suggestions.join(" or @"))
                        }
                    } else {
                        let list = valid
                            .iter()
                            .map(|s| format!("@{s}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("Valid annotations are: {list}")
                    };

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
        Align, Cold, Export, Hidden, Hot, Internal, Private, Protected, Section,
    };
    let visibility = |x: &FunctionAnnotation| matches!(x, Internal | Private | Hidden | Protected);
    let local = |x: &FunctionAnnotation| matches!(x, Internal | Private);
    a != b
        && ((visibility(a) && visibility(b))
            || (matches!((a, b), (Cold, Hot) | (Hot, Cold)))
            || matches!((a, b), (Section(_), Section(_)) | (Align(_), Align(_)))
            || (local(a) && *b == Export)
            || (*a == Export && local(b)))
}

/// Parses a function definition.
pub fn parse_function_def<'a>(state: &mut ParserState<'a>) -> Result<Function<'a>, LaminaError> {
    let annotations = parse_annotations(state)?;
    state.consume_keyword("fn")?;
    let name = state.parse_function_identifier()?;
    let signature = parse_fn_signature(state)?;

    state.skip_whitespace_and_comments();
    let declaration_only =
        annotations.contains(&FunctionAnnotation::Extern) && state.current_char() != Some('{');
    if signature.variadic && !declaration_only {
        return Err(state.error(format!(
            "Variadic parameter list on @{name}, which has a body\n  Hint: Lamina cannot define a variadic function; only an @extern declaration may end in ..."
        )));
    }

    if declaration_only {
        return Ok(Function {
            name,
            signature,
            annotations,
            basic_blocks: HashMap::new(),
            entry_block: "",
        });
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

    Ok(Function {
        name,
        signature,
        annotations,
        basic_blocks,
        entry_block,
    })
}

/// Parses a function signature.
pub fn parse_fn_signature<'a>(
    state: &mut ParserState<'a>,
) -> Result<FunctionSignature<'a>, LaminaError> {
    state.expect_char('(')?;
    let (params, variadic) = parse_param_list(state)?;
    state.expect_char(')')?;
    state.consume_keyword("->")?;
    let return_type = parse_type(state)?;
    Ok(FunctionSignature {
        params,
        return_type,
        variadic,
    })
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
        let param_name = state.parse_value_identifier()?;

        if !param_names.insert(param_name) {
            return Err(state.error(format!("Duplicate parameter name: %{param_name}")));
        }

        params.push(FunctionParameter {
            name: param_name,
            ty: param_ty,
            annotations: vec![],
        });

        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }
        state.expect_char(',')?;
    }
    Ok((params, false))
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
    loop {
        state.skip_whitespace_and_comments();
        if state.at_block_label() {
            break;
        }

        if state.current_char() == Some('}') {
            break;
        }

        let instruction = parse_instruction(state)?;
        let is_terminator = instruction.is_terminator();
        instructions.push(instruction);

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

    Ok((label, BasicBlock { instructions }))
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
    fn visibility_and_placement_annotations_parse() {
        let cases = [
            ("hot", FunctionAnnotation::Hot),
            ("pure", FunctionAnnotation::Pure),
            ("const", FunctionAnnotation::Const),
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
}
