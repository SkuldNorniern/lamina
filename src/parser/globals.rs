//! Global variable parsing for Lamina IR.

use std::collections::HashMap;

use crate::{
    LaminaError,
    ir::{
        GlobalDeclaration, GlobalInit, Identifier, Literal, PrimitiveType, Type, TypeDeclaration,
        Value, types::resolve_type_names,
    },
    parser::{state::ParserState, types::parse_type},
};

/// Parses a global declaration.
pub fn parse_global_declaration<'a>(
    state: &mut ParserState<'a>,
    exported: bool,
    types: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Result<GlobalDeclaration<'a>, LaminaError> {
    state.consume_keyword("global")?;
    let name = state.parse_type_identifier()?;
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
    })
}

/// Parses `[a, b, c]` for an array global, or `{a, b, c}` for a struct or tuple one.
fn parse_initializer_list<'a>(
    state: &mut ParserState<'a>,
    ty: &Type<'a>,
) -> Result<Vec<Literal<'a>>, LaminaError> {
    let (open, close, hints, capacity) = match ty {
        Type::Array { element_type, size } => (
            '[',
            ']',
            vec![element_type.as_ref().clone(); *size as usize],
            *size as usize,
        ),
        Type::Struct(fields) => (
            '{',
            '}',
            fields.iter().map(|field| field.ty.clone()).collect(),
            fields.len(),
        ),
        Type::Tuple(elements) => ('{', '}', elements.clone(), elements.len()),
        _ => {
            return Err(state.error(format!(
                "An element list initializes an array, struct or tuple, but this global has type {ty}"
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
        let hint = hints.get(elements.len()).ok_or_else(|| {
            state.error(format!("{capacity} elements fit here, this is one more"))
        })?;
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
        Some('@') => {
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
    let (name, min, max) = match hint {
        PrimitiveType::I8 => ("i8", i8::MIN as i128, i8::MAX as i128),
        PrimitiveType::I16 => ("i16", i16::MIN as i128, i16::MAX as i128),
        PrimitiveType::I32 => ("i32", i32::MIN as i128, i32::MAX as i128),
        PrimitiveType::I64 => ("i64", i64::MIN as i128, i64::MAX as i128),
        PrimitiveType::U8 => ("u8", 0, u8::MAX as i128),
        PrimitiveType::U16 => ("u16", 0, u16::MAX as i128),
        PrimitiveType::U32 => ("u32", 0, u32::MAX as i128),
        PrimitiveType::U64 => ("u64", 0, u64::MAX as i128),
        _ => return Ok(None),
    };

    // Only this literal's own characters decide whether it is a float; a later
    // element in a list carries its own point.
    let token = state.peek_slice(24).unwrap_or("");
    let digits = token
        .strip_prefix('-')
        .unwrap_or(token)
        .trim_start_matches(|c: char| c.is_ascii_digit());
    if digits.starts_with('.') {
        return Err(state.error(format!(
            "Float literal cannot be used with {name} type hint"
        )));
    }

    let value = i128::from(state.parse_integer()?);
    if value < min || value > max {
        return Err(state.error(format!(
            "Integer literal {value} out of range for {name} (must be between {min} and {max})"
        )));
    }

    Ok(Some(match hint {
        PrimitiveType::I8 => Literal::I8(value as i8),
        PrimitiveType::I16 => Literal::I16(value as i16),
        PrimitiveType::I32 => Literal::I32(value as i32),
        PrimitiveType::I64 => Literal::I64(value as i64),
        PrimitiveType::U8 => Literal::U8(value as u8),
        PrimitiveType::U16 => Literal::U16(value as u16),
        PrimitiveType::U32 => Literal::U32(value as u32),
        _ => Literal::U64(value as u64),
    }))
}
