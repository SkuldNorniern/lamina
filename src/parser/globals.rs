use crate::{GlobalDeclaration, LaminaError, Literal, PrimitiveType, Type, Value};
use super::state::ParserState;
use super::types::parse_type;

pub fn parse_global_declaration<'a>(
    state: &mut ParserState<'a>,
) -> Result<GlobalDeclaration<'a>, LaminaError> {
    state.consume_keyword("global")?;
    let name = state.parse_type_identifier()?; // Globals start with @
    state.expect_char(':')?;
    let ty = parse_type(state)?;

    state.skip_whitespace_and_comments();
    let initializer = if state.current_char() == Some('=') {
        state.advance(); // Consume '='
        // For global initializer, use the appropriate literal type based on declared type
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

// Parse a value potentially using type hints for integer literals
pub fn parse_value_with_type_hint<'a>(
    state: &mut ParserState<'a>,
    type_hint: &Type<'a>,
) -> Result<Value<'a>, LaminaError> {
    let start_pos = state.position(); // Remember position for backtracking
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

            // Check if the type hint is compatible with a string literal
            match type_hint {
                Type::Array { element_type, .. } => {
                    // Allow string literals for i8 arrays (standard string representation)
                    // and also bool arrays (as per requirement)
                    match element_type.as_ref() {
                        Type::Primitive(PrimitiveType::I8)
                        | Type::Primitive(PrimitiveType::Bool) => {
                            Ok(Value::Constant(Literal::String(string_value)))
                        }
                        _ => Err(state.error(format!(
                            "String literal is not compatible with type hint: {:?}",
                            type_hint
                        ))),
                    }
                }
                _ => Err(state.error(format!(
                    "String literal is not compatible with type hint: {:?}",
                    type_hint
                ))),
            }
        }
        Some('t') => {
            // Try to parse 'true'
            if state.peek_slice(4) == Some("true") {
                state.advance_by(4);

                // Check if hint is compatible with boolean
                match type_hint {
                    Type::Primitive(PrimitiveType::Bool) => {
                        Ok(Value::Constant(Literal::Bool(true)))
                    }
                    _ => Err(state.error(format!(
                        "Boolean literal not compatible with type hint: {:?}",
                        type_hint
                    ))),
                }
            } else {
                Err(state.error("Expected 'true'".to_string()))
            }
        }
        Some('f') => {
            // Try to parse 'false'
            if state.peek_slice(5) == Some("false") {
                state.advance_by(5);

                // Check if hint is compatible with boolean
                match type_hint {
                    Type::Primitive(PrimitiveType::Bool) => {
                        Ok(Value::Constant(Literal::Bool(false)))
                    }
                    _ => Err(state.error(format!(
                        "Boolean literal not compatible with type hint: {:?}",
                        type_hint
                    ))),
                }
            } else {
                Err(state.error("Expected 'false'".to_string()))
            }
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            // Try handling numbers based on hint

            // Handle float hint - try parse as float first
            if matches!(type_hint, Type::Primitive(PrimitiveType::F32)) {
                if let Ok(f_val) = state.parse_float() {
                    return Ok(Value::Constant(Literal::F32(f_val)));
                }

                // If float parse failed, try integer and convert
                state.set_position(start_pos);
                if let Ok(i_val) = state.parse_integer() {
                    return Ok(Value::Constant(Literal::F32(i_val as f32)));
                }

                return Err(state.error("Expected float literal for F32 hint".to_string()));
            }

            // For integer type hints, parse as integer
            match type_hint {
                Type::Primitive(PrimitiveType::I8) => {
                    // For i8 hint, ensure we're parsing an integer, not a float
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                            .error("Float literal cannot be used with I8 type hint".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    // Check range for i8
                    if i_val < i8::MIN as i64 || i_val > i8::MAX as i64 {
                        return Err(
                            state.error(format!("Integer literal {} out of range for i8", i_val))
                        );
                    }
                    Ok(Value::Constant(Literal::I8(i_val as i8)))
                }
                Type::Primitive(PrimitiveType::I32) => {
                    // For i32 hint, ensure we're parsing an integer, not a float
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                            .error("Float literal cannot be used with I32 type hint".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    // Check range for i32
                    if i_val < i32::MIN as i64 || i_val > i32::MAX as i64 {
                        return Err(
                            state.error(format!("Integer literal {} out of range for i32", i_val))
                        );
                    }
                    Ok(Value::Constant(Literal::I32(i_val as i32)))
                }
                Type::Primitive(PrimitiveType::I64) => {
                    // For i64 hint, ensure we're parsing an integer, not a float
                    let peek_string = state.peek_slice(20).unwrap_or("");
                    if peek_string.contains('.') {
                        return Err(state
                            .error("Float literal cannot be used with I64 type hint".to_string()));
                    }

                    let i_val = state.parse_integer()?;
                    Ok(Value::Constant(Literal::I64(i_val)))
                }
                // Default case - try best-effort approach
                _ => {
                    // Try integer first
                    state.set_position(start_pos);
                    if let Ok(i_val) = state.parse_integer() {
                        return if i_val >= i32::MIN as i64 && i_val <= i32::MAX as i64 {
                            Ok(Value::Constant(Literal::I32(i_val as i32)))
                        } else {
                            Ok(Value::Constant(Literal::I64(i_val)))
                        };
                    }

                    // Reset position and try float
                    state.set_position(start_pos);
                    if let Ok(f_val) = state.parse_float() {
                        return Ok(Value::Constant(Literal::F32(f_val)));
                    }

                    // If we get here, both parse attempts failed
                    Err(state.error("Expected numeric literal".to_string()))
                }
            }
        }
        _ => Err(state.error("Expected value (%, @, literal)".to_string())),
    }
}

