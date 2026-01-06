//! Type parsing for Lamina IR.

use super::state::ParserState;
use crate::{LaminaError, PrimitiveType, StructField, Type, TypeDeclaration};

/// Parses a type declaration.
pub fn parse_type_declaration<'a>(
    state: &mut ParserState<'a>,
) -> Result<TypeDeclaration<'a>, LaminaError> {
    state.consume_keyword("type")?;
    let name = state.parse_type_identifier()?;
    state.expect_char('=')?;
    let ty = parse_composite_type(state)?;
    Ok(TypeDeclaration { name, ty })
}

/// Parses a composite type (struct or array).
pub fn parse_composite_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    let keyword_slice = state.peek_slice(6).unwrap_or("");
    if keyword_slice.starts_with("struct") {
        state.consume_keyword("struct")?;
        state.expect_char('{')?;
        let mut fields = Vec::new();
        let mut field_names = std::collections::HashSet::new();
        
        loop {
            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') {
                if fields.is_empty() {
                    return Err(state.error("Struct type must have at least one field\n  Hint: Empty structs are not allowed. Add at least one field (e.g., 'struct { x: i32 }')".to_string()));
                }
                state.advance();
                break;
            }
            let field_name = state.parse_identifier_str()?;
            
            // Check for duplicate field names
            if !field_names.insert(field_name) {
                return Err(state.error(format!(
                    "Duplicate struct field name: '{}'\n  Hint: Each field in a struct must have a unique name",
                    field_name
                )));
            }
            
            state.expect_char(':')?;
            let field_ty = parse_type(state)?;
            fields.push(StructField {
                name: field_name,
                ty: field_ty,
            });

            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') {
                state.advance();
                break;
            }
            state.expect_char(',')?;
        }
        Ok(Type::Struct(fields))
    } else if state.current_char() == Some('[') {
        state.expect_char('[')?;
        let size_val = state.parse_integer()?;
        
        if size_val < 0 {
            return Err(state.error(format!(
                "Invalid array size: {}",
                size_val
            )));
        }
        
        let size = size_val as u64;
        state.consume_keyword("x")?;
        let elem_type = parse_type(state)?;
        state.expect_char(']')?;
        Ok(Type::Array {
            element_type: Box::new(elem_type),
            size,
        })
        } else {
            let found = state.peek_slice(20).unwrap_or("");
            Err(state.error(format!(
                "Expected 'struct' or '[' for composite type, but found '{}'",
                found
            )))
        }
}

/// Parses a type (primitive, composite, named, or tuple).
pub fn parse_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    match state.current_char() {
        Some('@') => Ok(Type::Named(state.parse_type_identifier()?)),
        Some('[') => parse_composite_type(state),
        Some('s') if state.peek_slice(6) == Some("struct") => parse_composite_type(state),
        Some('t') if state.peek_slice(5) == Some("tuple") => {
            state.consume_keyword("tuple")?;
            state.skip_whitespace_and_comments();
            let mut element_types = Vec::new();
            loop {
                let elem_type = parse_type(state)?;
                element_types.push(elem_type);
                state.skip_whitespace_and_comments();
                if state.current_char() != Some(',') {
                    break;
                }
                state.expect_char(',')?;
                state.skip_whitespace_and_comments();
            }
            Ok(Type::Tuple(element_types))
        }
        _ => {
            let potential_primitive = state.parse_identifier_str()?;
            match potential_primitive {
                "i8" => Ok(Type::Primitive(PrimitiveType::I8)),
                "i16" => Ok(Type::Primitive(PrimitiveType::I16)),
                "i32" => Ok(Type::Primitive(PrimitiveType::I32)),
                "i64" => Ok(Type::Primitive(PrimitiveType::I64)),
                "u8" => Ok(Type::Primitive(PrimitiveType::U8)),
                "u16" => Ok(Type::Primitive(PrimitiveType::U16)),
                "u32" => Ok(Type::Primitive(PrimitiveType::U32)),
                "u64" => Ok(Type::Primitive(PrimitiveType::U64)),
                "f32" => Ok(Type::Primitive(PrimitiveType::F32)),
                "f64" => Ok(Type::Primitive(PrimitiveType::F64)),
                "bool" => Ok(Type::Primitive(PrimitiveType::Bool)),
                "char" => Ok(Type::Primitive(PrimitiveType::Char)),
                "ptr" => Ok(Type::Primitive(PrimitiveType::Ptr)),
                "void" => Ok(Type::Void),
                _ => {
                    let primitive_types = super::get_primitive_type_names();
                    // Include "void" as it's a valid type identifier but not a primitive type
                    let all_type_names: Vec<&str> = primitive_types.iter().copied().chain(std::iter::once("void")).collect();
                    let mut suggestions = Vec::new();
                    const MAX_TYPO_DISTANCE: usize = 2;
                    
                    for valid in &all_type_names {
                        let distance = super::edit_distance(potential_primitive, valid, Some(MAX_TYPO_DISTANCE));
                        if distance <= MAX_TYPO_DISTANCE {
                            suggestions.push(*valid);
                        }
                    }
                    
                    suggestions.sort_by_key(|&s| super::edit_distance(potential_primitive, s, None));
                    
                    let hint = if !suggestions.is_empty() {
                        if suggestions.len() == 1 {
                            format!("Did you mean '{}'?", suggestions[0])
                        } else {
                            format!("Did you mean one of: {}?", suggestions.iter().take(3).map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", "))
                        }
                    } else {
                        format!("Valid type identifiers include: {}, void, or named types starting with @", primitive_types.join(", "))
                    };
                    
                    Err(state.error(format!(
                        "Unknown type identifier: '{}'\n  Hint: {}",
                        potential_primitive, hint
                    )))
                }
            }
        }
    }
}
