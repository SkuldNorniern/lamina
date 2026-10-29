//! Global variable parsing for Lamina IR.

use std::collections::HashMap;

use crate::{
    LaminaError,
    ir::{
        GlobalDeclaration, GlobalInit, Identifier, Literal, PrimitiveType, TlsModel, Type,
        TypeDeclaration, Value, types::resolve_type_names,
    },
    parser::{
        functions::{parse_alignment_argument, unknown_annotation_hint},
        state::ParserState,
        types::parse_type,
        values::{integer_bounds, integer_literal},
    },
};

const GLOBAL_ANNOTATION_NAMES: &[&str] = &["export", "align", "thread_local"];

/// The annotations written before a `global`.
#[derive(Debug, Default)]
pub struct GlobalAnnotations {
    pub exported: bool,
    pub align: Option<u32>,
    pub thread_local: Option<TlsModel>,
}

/// Parses the `@name` annotations that precede a `global`, in any order.
pub fn parse_global_annotations(
    state: &mut ParserState<'_>,
) -> Result<GlobalAnnotations, LaminaError> {
    let mut annotations = GlobalAnnotations::default();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() != Some('@') {
            break;
        }
        state.advance();
        let name = state.parse_identifier_str()?;
        match name {
            "export" => {
                if annotations.exported {
                    return Err(state.error("Duplicate annotation: @export"));
                }
                annotations.exported = true;
            }
            "align" => {
                if annotations.align.is_some() {
                    return Err(state.error("Duplicate annotation: @align"));
                }
                annotations.align = Some(parse_alignment_argument(state)?);
            }
            "thread_local" => {
                if annotations.thread_local.is_some() {
                    return Err(state.error("Duplicate annotation: @thread_local"));
                }
                annotations.thread_local = Some(parse_tls_model_argument(state)?);
            }
            _ => {
                let hint = unknown_annotation_hint(name, GLOBAL_ANNOTATION_NAMES);
                return Err(state.error(format!(
                    "Unknown global annotation: @{name}\n  Hint: {hint}"
                )));
            }
        }
    }
    Ok(annotations)
}

/// Parses a global declaration, given the annotations already read before its `global`.
pub fn parse_global_declaration<'a>(
    state: &mut ParserState<'a>,
    annotations: GlobalAnnotations,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Result<GlobalDeclaration<'a>, LaminaError> {
    let GlobalAnnotations {
        exported,
        align,
        thread_local,
    } = annotations;
    state.consume_keyword("global")?;
    let name = state.parse_global_identifier()?;
    state.expect_char(':')?;
    let ty = parse_type(state)?;

    state.skip_whitespace_and_comments();
    let initializer = if state.current_char() == Some('=') {
        state.advance();
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('[') || state.current_char() == Some('{') {
            let resolved = resolve_type_names(&ty, types).unwrap_or_else(|| ty.clone());
            Some(GlobalInit::List(parse_initializer_list(state, &resolved)?))
        } else {
            Some(GlobalInit::Value(parse_value_with_type_hint(state, &ty)?))
        }
    } else {
        None
    };

    Ok(GlobalDeclaration {
        name,
        ty,
        initializer,
        exported,
        align,
        thread_local,
    })
}

/// Parses the optional `(model)` after `@thread_local`. Local-exec is the default
/// because it needs nothing of the dynamic loader; the two dynamic models are named
/// and refused rather than silently downgraded.
fn parse_tls_model_argument(state: &mut ParserState<'_>) -> Result<TlsModel, LaminaError> {
    state.skip_inline_space();
    if state.current_char() != Some('(') {
        return Ok(TlsModel::LocalExec);
    }
    state.advance();
    state.skip_whitespace_and_comments();
    let name = state.parse_identifier_str()?;
    state.skip_whitespace_and_comments();
    state.expect_char(')')?;
    match name {
        "local_exec" => Ok(TlsModel::LocalExec),
        "initial_exec" => Ok(TlsModel::InitialExec),
        "general_dynamic" | "local_dynamic" => Err(state.error(format!(
            "@thread_local({name}) is not supported: {name} resolves through a __tls_get_addr call, and lamina has no TLS runtime\n  Hint: Use @thread_local(local_exec) in an executable that defines the symbol, or @thread_local(initial_exec) when it may come from a shared object loaded at startup"
        ))),
        _ => Err(state.error(format!(
            "Unknown TLS model '{name}'\n  Hint: Valid models are local_exec and initial_exec"
        ))),
    }
}

/// Parses `[a, b, c]` for an array global, or `{a, b, c}` for a struct one.
fn parse_initializer_list<'a>(
    state: &mut ParserState<'a>,
    ty: &Type<'a>,
) -> Result<Vec<Literal<'a>>, LaminaError> {
    let (open, close, uniform, hints, capacity) = match ty {
        Type::Array { element_type, size } => (
            '[',
            ']',
            Some(element_type.as_ref().clone()),
            Vec::new(),
            usize::try_from(*size).unwrap_or(usize::MAX),
        ),
        Type::Struct { fields, .. } => (
            '{',
            '}',
            None,
            fields.iter().map(|field| field.ty.clone()).collect(),
            fields.len(),
        ),
        _ => {
            return Err(state.error(format!(
                "An element list initializes an array or struct, but this global has type {ty}"
            )));
        }
    };
    state.expect_char(open)?;
    let mut elements = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(close) {
            state.advance();
            break;
        }
        let hint = match &uniform {
            Some(element) => element,
            None => hints.get(elements.len()).ok_or_else(|| {
                state.error(format!("{capacity} elements fit here, this is one more"))
            })?,
        };
        if elements.len() >= capacity {
            return Err(state.error(format!("{capacity} elements fit here, this is one more")));
        }
        let value = parse_value_with_type_hint(state, hint)?;
        let Value::Constant(literal) = value else {
            return Err(state.error("An element list takes literals only"));
        };
        elements.push(literal);
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(close) {
            state.advance();
            break;
        }
        state.expect_char(',')?;
    }
    Ok(elements)
}

/// Parses a value using type hints for integer literals.
pub fn parse_value_with_type_hint<'a>(
    state: &mut ParserState<'a>,
    type_hint: &Type<'a>,
) -> Result<Value<'a>, LaminaError> {
    let start_pos = state.position();
    state.skip_whitespace_and_comments();

    match state.current_char() {
        Some('%') => {
            state.advance();
            let name = state.parse_identifier_str()?;
            Ok(Value::Variable(name))
        }
        Some('$') => {
            state.advance();
            let name = state.parse_identifier_str()?;
            Ok(Value::Global(name))
        }
        Some('"') => {
            let string_value = state.parse_string_literal()?;

            match type_hint {
                Type::Array { element_type, .. } => match element_type.as_ref() {
                    Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::Bool) => {
                        Ok(Value::Constant(Literal::String(string_value)))
                    }
                    _ => Err(state.error(format!(
                        "String literal is not compatible with type hint: {type_hint:?}"
                    ))),
                },
                _ => Err(state.error(format!(
                    "String literal is not compatible with type hint: {type_hint:?}"
                ))),
            }
        }
        Some('t') => {
            if state.peek_slice(4) == Some("true") {
                state.advance_by(4);

                match type_hint {
                    Type::Primitive(PrimitiveType::Bool) => {
                        Ok(Value::Constant(Literal::Bool(true)))
                    }
                    _ => Err(state.error(format!(
                        "Boolean literal not compatible with type hint: {type_hint:?}"
                    ))),
                }
            } else {
                let found = state.peek_slice(10).unwrap_or("");
                let hint = if super::edit_distance(found, "true", Some(2)) <= 2 {
                    "Did you mean 'true'?".to_string()
                } else {
                    "Expected 'true' boolean literal".to_string()
                };

                Err(state.error(format!("Expected 'true'\n  Hint: {hint}")))
            }
        }
        Some('f') => {
            if state.peek_slice(5) == Some("false") {
                state.advance_by(5);

                match type_hint {
                    Type::Primitive(PrimitiveType::Bool) => {
                        Ok(Value::Constant(Literal::Bool(false)))
                    }
                    _ => Err(state.error(format!(
                        "Boolean literal not compatible with type hint: {type_hint:?}"
                    ))),
                }
            } else {
                let found = state.peek_slice(10).unwrap_or("");
                let hint = if super::edit_distance(found, "false", Some(2)) <= 2 {
                    "Did you mean 'false'?".to_string()
                } else {
                    "Expected 'false' boolean literal".to_string()
                };

                Err(state.error(format!("Expected 'false'\n  Hint: {hint}")))
            }
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            if matches!(type_hint, Type::Primitive(PrimitiveType::F32)) {
                if let Ok(f_val) = state.parse_float() {
                    return Ok(Value::Constant(Literal::F32(f_val as f32)));
                }

                state.set_position(start_pos);
                if let Ok(i_val) = state.parse_integer() {
                    return Ok(Value::Constant(Literal::F32(i_val as f32)));
                }

                return Err(state.error("Expected float literal for F32 hint"));
            }

            if matches!(type_hint, Type::Primitive(PrimitiveType::F64)) {
                if let Ok(f_val) = state.parse_float() {
                    return Ok(Value::Constant(Literal::F64(f_val)));
                }

                state.set_position(start_pos);
                if let Ok(i_val) = state.parse_integer() {
                    return Ok(Value::Constant(Literal::F64(i_val as f64)));
                }

                return Err(state.error("Expected float literal for F64 hint"));
            }

            if matches!(type_hint, Type::Primitive(PrimitiveType::Bool)) {
                return Err(state.error(
                    "Integer literal cannot be used with bool type\n  Hint: Write 'true' or 'false'",
                ));
            }

            if let Type::Primitive(p) = type_hint
                && let Some(literal) = parse_integer_literal(state, *p)?
            {
                return Ok(Value::Constant(literal));
            }

            state.set_position(start_pos);
            if let Ok(i_val) = state.parse_integer() {
                return if i_val >= i32::MIN as i64 && i_val <= i32::MAX as i64 {
                    Ok(Value::Constant(Literal::I32(i_val as i32)))
                } else {
                    Ok(Value::Constant(Literal::I64(i_val)))
                };
            }

            state.set_position(start_pos);
            if let Ok(f_val) = state.parse_float() {
                return Ok(Value::Constant(Literal::F64(f_val)));
            }

            Err(state.error("Expected a numeric literal"))
        }
        _ => Err(state.error("Expected value (%, @, literal)")),
    }
}

/// Parses an integer literal at the declared width. Returns `None` when the hint is
/// not an integer type, so the caller can fall back to inference.
fn parse_integer_literal<'a>(
    state: &mut ParserState<'a>,
    hint: PrimitiveType,
) -> Result<Option<Literal<'a>>, LaminaError> {
    if integer_bounds(hint).is_none() {
        return Ok(None);
    }

    // Only this literal's own characters decide whether it is a float; a later
    // element in a list carries its own point.
    let token = state.peek_slice(40).unwrap_or("");
    let digits = token
        .strip_prefix('-')
        .unwrap_or(token)
        .trim_start_matches(|c: char| c.is_ascii_digit());
    if digits.starts_with('.') {
        return Err(state.error(format!(
            "Float literal cannot be used with {hint} type hint"
        )));
    }

    if hint == PrimitiveType::U128 {
        return Ok(Some(Literal::U128(state.parse_wide_unsigned()?)));
    }
    let value = state.parse_wide_integer()?;
    integer_literal(value, hint).map_err(|message| state.error(message))
}
