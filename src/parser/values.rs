use crate::{LaminaError, Literal, Value};
use super::state::ParserState;

// Parses a value: literal or %variable or @global
// This version tries I32 for integer literals first, but will use I64 for larger values.
pub fn parse_value<'a>(state: &mut ParserState<'a>) -> Result<Value<'a>, LaminaError> {
    // Special case for numeric literals without a context
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
            // String literals need array type hints, so just use a default I8 array
            let string_value = state.parse_string_literal()?;
            Ok(Value::Constant(Literal::String(string_value)))
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            // Check if this looks like a float (has a dot after digits)
            // Peek ahead to see if there's a dot followed by digits
            let peek_pos = state.position();
            let mut temp_pos = peek_pos;
            let mut has_digits = false;
            let bytes = state.bytes();
            
            // Skip negative sign if present
            if !state.is_eof() && bytes.get(temp_pos) == Some(&b'-') {
                temp_pos += 1;
            }
            
            // Check for digits
            while temp_pos < bytes.len() && bytes[temp_pos].is_ascii_digit() {
                has_digits = true;
                temp_pos += 1;
            }
            
            // Check if there's a dot followed by digits (indicating a float)
            let looks_like_float = has_digits 
                && temp_pos < bytes.len() 
                && bytes[temp_pos] == b'.'
                && temp_pos + 1 < bytes.len()
                && bytes[temp_pos + 1].is_ascii_digit();
            
            // If it looks like a float, parse as float directly
            if looks_like_float {
                if let Ok(f_val) = state.parse_float() {
                    return Ok(Value::Constant(Literal::F32(f_val)));
                }
            }
            
            // Otherwise, try parsing as integer first
            if let Ok(i_val) = state.parse_integer() {
                return if i_val >= i32::MIN as i64 && i_val <= i32::MAX as i64 {
                    Ok(Value::Constant(Literal::I32(i_val as i32)))
                } else {
                    // If it doesn't fit in I32, use I64
                    Ok(Value::Constant(Literal::I64(i_val)))
                };
            }

            // Reset position and try float as fallback
            state.set_position(start_pos);
            if let Ok(f_val) = state.parse_float() {
                return Ok(Value::Constant(Literal::F32(f_val)));
            }

            Err(state.error("Expected numeric literal".to_string()))
        }
        Some('t') => {
            // Try parsing boolean 'true'
            if state.peek_slice(4) == Some("true") {
                state.advance_by(4);
                return Ok(Value::Constant(Literal::Bool(true)));
            }
            Err(state.error("Expected value".to_string()))
        }
        Some('f') => {
            // Try parsing boolean 'false'
            if state.peek_slice(5) == Some("false") {
                state.advance_by(5);
                return Ok(Value::Constant(Literal::Bool(false)));
            }
            Err(state.error("Expected value".to_string()))
        }
        _ => Err(state.error("Expected value (%, @, literal)".to_string())),
    }
}

