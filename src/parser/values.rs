//! Value parsing for Lamina IR.

use crate::{
    LaminaError,
    ir::{Literal, PrimitiveType, Value},
    parser::state::ParserState,
};

/// Inclusive bounds of an integer type, or `None` for a type that is not an integer.
///
/// Literals are read as `i128`, so these bounds cover `u128` only up to `i128::MAX`.
/// The inclusive range an integer type holds, or `None` when it is not an integer.
pub fn integer_bounds(ty: PrimitiveType) -> Option<(i128, i128)> {
    ty.integer_bounds()
}

/// The literal of integer type `ty` holding `value`. `Ok(None)` when `ty` is not an
/// integer type; `Err` with the range message when `value` does not fit.
pub fn integer_literal(value: i128, ty: PrimitiveType) -> Result<Option<Literal<'static>>, String> {
    let Some((min, max)) = integer_bounds(ty) else {
        return Ok(None);
    };
    if value < min || value > max {
        return Err(format!(
            "Integer literal {value} out of range for {ty} (must be between {min} and {max})"
        ));
    }
    Ok(Some(match ty {
        PrimitiveType::I8 => Literal::I8(value as i8),
        PrimitiveType::I16 => Literal::I16(value as i16),
        PrimitiveType::I32 => Literal::I32(value as i32),
        PrimitiveType::I64 => Literal::I64(value as i64),
        PrimitiveType::U8 => Literal::U8(value as u8),
        PrimitiveType::U16 => Literal::U16(value as u16),
        PrimitiveType::U32 => Literal::U32(value as u32),
        PrimitiveType::U64 => Literal::U64(value as u64),
        PrimitiveType::I128 => Literal::I128(value),
        PrimitiveType::U128 => Literal::U128(value as u128),
        PrimitiveType::F32 | PrimitiveType::F64 | PrimitiveType::Bool | PrimitiveType::Ptr => {
            return Ok(None);
        }
    }))
}

/// Parses a value: literal, variable (%name), or global ($name).
pub fn parse_value<'a>(state: &mut ParserState<'a>) -> Result<Value<'a>, LaminaError> {
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
            Ok(Value::Constant(Literal::String(string_value)))
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            let peek_pos = state.position();
            let mut temp_pos = peek_pos;
            let mut has_digits = false;
            let bytes = state.bytes();

            if !state.is_eof() && bytes.get(temp_pos) == Some(&b'-') {
                temp_pos += 1;
            }

            while temp_pos < bytes.len() && bytes[temp_pos].is_ascii_digit() {
                has_digits = true;
                temp_pos += 1;
            }

            let looks_like_float = has_digits
                && temp_pos < bytes.len()
                && bytes[temp_pos] == b'.'
                && temp_pos + 1 < bytes.len()
                && bytes[temp_pos + 1].is_ascii_digit();

            // A digit run with no '.' or exponent is an integer literal. If it
            // does not fit, that is an error: falling through to the float
            // parser would silently turn it into an f64.
            let is_pure_integer = has_digits
                && (temp_pos >= bytes.len() || !matches!(bytes[temp_pos], b'.' | b'e' | b'E'));

            if looks_like_float && let Ok(f_val) = state.parse_float() {
                return Ok(Value::Constant(Literal::F64(f_val)));
            }

            match state.parse_integer() {
                Ok(i_val) => {
                    return if i_val >= i32::MIN as i64 && i_val <= i32::MAX as i64 {
                        Ok(Value::Constant(Literal::I32(i_val as i32)))
                    } else {
                        Ok(Value::Constant(Literal::I64(i_val)))
                    };
                }
                Err(e) if is_pure_integer => {
                    // Values above i64::MAX are still valid u64 literals.
                    state.set_position(start_pos);
                    if let Ok(u_val) = state.parse_u64() {
                        return Ok(Value::Constant(Literal::U64(u_val)));
                    }
                    state.set_position(start_pos);
                    return Err(e);
                }
                Err(_) => {}
            }

            state.set_position(start_pos);
            if let Ok(f_val) = state.parse_float() {
                return Ok(Value::Constant(Literal::F64(f_val)));
            }

            Err(state.error("Expected a numeric literal"))
        }
        Some('t') => {
            if state.peek_slice(4) == Some("true") {
                state.advance_by(4);
                return Ok(Value::Constant(Literal::Bool(true)));
            }
            let found = state.peek_slice(10).unwrap_or("");
            let hint = if super::edit_distance(found, "true", Some(2)) <= 2 {
                "Did you mean 'true'?".to_string()
            } else {
                "Expected 'true' boolean literal".to_string()
            };

            Err(state.error(format!(
                "{}\n  Hint: {}",
                "Expected 'true' boolean literal", hint
            )))
        }
        Some('f') => {
            if state.peek_slice(5) == Some("false") {
                state.advance_by(5);
                return Ok(Value::Constant(Literal::Bool(false)));
            }
            let found = state.peek_slice(10).unwrap_or("");
            let hint = if super::edit_distance(found, "false", Some(2)) <= 2 {
                "Did you mean 'false'?".to_string()
            } else {
                "Expected 'false' boolean literal".to_string()
            };

            Err(state.error(format!(
                "{}\n  Hint: {}",
                "Expected 'false' boolean literal", hint
            )))
        }
        Some('@') => Err(state.error(
            "Expected a value, but found '@'\n  Hint: A global is $name; the address of a function is 'funcaddr @name'",
        )),
        _ => {
            let found = state
                .current_char()
                .map_or_else(|| "end of input".to_string(), |c| format!("'{c}'"));
            Err(state.error(format!("Expected a value, but found {found}")))
        }
    }
}
