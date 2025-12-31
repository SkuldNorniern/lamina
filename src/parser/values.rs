//! Value parsing for Lamina IR.

use super::state::ParserState;
use crate::{LaminaError, Literal, Value};

/// Parses a value: literal, variable (%name), or global (@name).
pub fn parse_value<'a>(state: &mut ParserState<'a>) -> Result<Value<'a>, LaminaError> {
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

            if looks_like_float && let Ok(f_val) = state.parse_float() {
                return Ok(Value::Constant(Literal::F32(f_val)));
            }

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
        Some('t') => {
            if state.peek_slice(4) == Some("true") {
                state.advance_by(4);
                return Ok(Value::Constant(Literal::Bool(true)));
            }
            Err(state.error(
                "Expected 'true' boolean literal\n  Hint: Boolean literals are 'true' or 'false'"
                    .to_string(),
            ))
        }
        Some('f') => {
            if state.peek_slice(5) == Some("false") {
                state.advance_by(5);
                return Ok(Value::Constant(Literal::Bool(false)));
            }
            Err(state.error(
                "Expected 'false' boolean literal\n  Hint: Boolean literals are 'true' or 'false'"
                    .to_string(),
            ))
        }
        _ => {
            let found = state
                .current_char()
                .map(|c| format!("'{}'", c))
                .unwrap_or_else(|| "end of input".to_string());
            Err(state.error(format!(
                "Expected a value, but found {}\n  Hint: Values can be:\n    - Variables: %name\n    - Globals: @name\n    - Literals: numbers (42, -10), strings (\"hello\"), booleans (true, false)",
                found
            )))
        }
    }
}
