//! Global variable parsing for Lamina IR.

use super::state::ParserState;
use super::types::parse_type;
use crate::{GlobalDeclaration, LaminaError, Literal, PrimitiveType, Type, Value};

/// Parses a global declaration.
pub fn parse_global_declaration<'a>(
    state: &mut ParserState<'a>,
) -> Result<GlobalDeclaration<'a>, LaminaError> {
    state.consume_keyword("global")?;
    let name = state.parse_type_identifier()?;
    state.expect_char(':')?;
    let ty = parse_type(state)?;

    state.skip_whitespace_and_comments();
    let initializer = if state.current_char() == Some('=') {
        state.advance();
        let value = parse_value_with_type_hint(state, &ty)?;
        Some(value)
    } else {
        None
    };

    Ok(GlobalDeclaration {
        name,
        ty,
        initializer,
    })
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
                        "String literal is not compatible with type hint: {:?}\n  Hint: String literals can only be used with array types containing i8 or bool elements",
                        type_hint
                    ))),
                },
                _ => Err(state.error(format!(
                    "String literal is not compatible with type hint: {:?}",
                    type_hint
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
                        "Boolean literal not compatible with type hint: {:?}\n  Hint: Boolean literals (true/false) can only be used with bool type",
                        type_hint
                    ))),
                }
            } else {
                Err(state.error("Expected 'true'".to_string()))
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
                        "Boolean literal not compatible with type hint: {:?}\n  Hint: Boolean literals (true/false) can only be used with bool type",
                        type_hint
                    ))),
                }
            } else {
                Err(state.error("Expected 'false'".to_string()))
            }
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            if matches!(type_hint, Type::Primitive(PrimitiveType::F32)) {
                if let Ok(f_val) = state.parse_float() {
                    return Ok(Value::Constant(Literal::F32(f_val)));
                }

                state.set_position(start_pos);
                if let Ok(i_val) = state.parse_integer() {
                    return Ok(Value::Constant(Literal::F32(i_val as f32)));
                }

                return Err(state.error("Expected float literal for F32 hint".to_string()));
            }

            match type_hint {
                Type::Primitive(PrimitiveType::I8) => {
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                    .error("Float literal cannot be used with I8 type hint\n  Hint: Use an integer literal (e.g., 42) instead of a float (e.g., 42.0) for integer types".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    if i_val < i8::MIN as i64 || i_val > i8::MAX as i64 {
                        return Err(
                        state.error(format!(
                            "Integer literal {} out of range for i8\n  Hint: i8 values must be between {} and {}",
                            i_val, i8::MIN, i8::MAX
                        ))
                    );
                    }
                    Ok(Value::Constant(Literal::I8(i_val as i8)))
                }
                Type::Primitive(PrimitiveType::I32) => {
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                            .error("Float literal cannot be used with I32 type hint".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    if i_val < i32::MIN as i64 || i_val > i32::MAX as i64 {
                        return Err(
                            state.error(format!(
                                "Integer literal {} out of range for i32\n  Hint: i32 values must be between {} and {}",
                                i_val, i32::MIN, i32::MAX
                            ))
                        );
                    }
                    Ok(Value::Constant(Literal::I32(i_val as i32)))
                }
                Type::Primitive(PrimitiveType::I64) => {
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                            .error("Float literal cannot be used with I64 type hint".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    Ok(Value::Constant(Literal::I64(i_val)))
                }
                _ => {
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
                        return Ok(Value::Constant(Literal::F32(f_val)));
                    }

                    Err(state.error("Expected a numeric literal\n  Hint: Numeric literals can be integers (42, -10) or floats (3.14, -0.5)".to_string()))
                }
            }
        }
        _ => Err(state.error("Expected value (%, @, literal)".to_string())),
    }
}
