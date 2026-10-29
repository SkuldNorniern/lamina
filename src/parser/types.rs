//! Type parsing for Lamina IR.

use std::{collections::HashSet, iter};

use crate::{
    LaminaError,
    ir::{PrimitiveType, StructField, Type, TypeDeclaration},
    parser::state::ParserState,
};

pub fn removed_char_error(state: &ParserState<'_>) -> LaminaError {
    state.error(
        "Type 'char' was removed\n  Hint: Use 'i8' for a signed byte or 'u8' for an unsigned byte",
    )
}

/// Parses a type declaration.
pub fn parse_type_declaration<'a>(
    state: &mut ParserState<'a>,
) -> Result<TypeDeclaration<'a>, LaminaError> {
    state.consume_keyword("type")?;
    let name = state.parse_named_type_identifier()?;
    state.expect_char('=')?;
    let ty = parse_composite_type(state)?;
    Ok(TypeDeclaration { name, ty })
}

/// Parses a composite type (struct or array). A struct may be prefixed with `packed`,
/// which removes the padding between its fields.
pub fn parse_composite_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    let packed = state.peek_slice(6) == Some("packed");
    if packed {
        state.consume_keyword("packed")?;
        state.skip_whitespace_and_comments();
        if state.peek_slice(6) != Some("struct") {
            return Err(state.error(
                "'packed' applies to a struct only\n  Hint: Write 'packed struct { x: i8, y: i32 }'",
            ));
        }
    }
    let keyword_slice = state.peek_slice(6).unwrap_or("");
    if keyword_slice.starts_with("struct") {
        state.consume_keyword("struct")?;
        state.expect_char('{')?;
        let mut fields = Vec::new();
        let mut field_names = HashSet::new();

        loop {
            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') {
                if fields.is_empty() {
                    return Err(state.error("Struct type must have at least one field\n  Hint: Empty structs are not allowed. Add at least one field (e.g., 'struct { x: i32 }')"));
                }
                state.advance();
                break;
            }
            let field_name = state.parse_identifier_str()?;

            // Check for duplicate field names
            if !field_names.insert(field_name) {
                return Err(state.error(format!(
                    "Duplicate struct field name: '{field_name}'\n  Hint: Each field in a struct must have a unique name"
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
        Ok(Type::Struct { fields, packed })
    } else if state.current_char() == Some('[') {
        state.expect_char('[')?;
        let size_val = state.parse_integer()?;

        if size_val < 0 {
            return Err(state.error(format!("Invalid array size: {size_val}")));
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
            "Expected 'struct' or '[' for composite type, but found '{found}'"
        )))
    }
}

/// Parses a type (primitive, composite, or named).
pub fn parse_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.enter_type()?;
    let ty = parse_type_inner(state);
    state.leave_type();
    ty
}

fn parse_type_inner<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    match state.current_char() {
        Some('!') => Ok(Type::Named(state.parse_named_type_identifier()?)),
        Some('[') => parse_composite_type(state),
        Some('s') if state.peek_slice(6) == Some("struct") => parse_composite_type(state),
        Some('p') if state.peek_slice(6) == Some("packed") => parse_composite_type(state),
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
                "i128" => Ok(Type::Primitive(PrimitiveType::I128)),
                "u128" => Ok(Type::Primitive(PrimitiveType::U128)),
                "f32" => Ok(Type::Primitive(PrimitiveType::F32)),
                "f64" => Ok(Type::Primitive(PrimitiveType::F64)),
                "bool" => Ok(Type::Primitive(PrimitiveType::Bool)),
                "char" => Err(removed_char_error(state)),
                "tuple" => Err(state.error(
                    "Type 'tuple' was removed\n  Hint: Use a struct with named fields, e.g. 'struct { a: i32, b: f64 }'",
                )),
                "ptr" => Ok(Type::Primitive(PrimitiveType::Ptr)),
                "void" => Ok(Type::Void),
        "valist" => Ok(Type::VaList),
                _ => {
                    let primitive_types = super::get_primitive_type_names();
                    // Include "void" as it's a valid type identifier but not a primitive type
                    let all_type_names: Vec<&str> = primitive_types
                        .iter()
                        .copied()
                        .chain(iter::once("void"))
                        .collect();
                    let suggestions =
                        super::suggest_alternatives(potential_primitive, &all_type_names);

                    let hint = if !suggestions.is_empty() {
                        if suggestions.len() == 1 {
                            format!("Did you mean '{}'?", suggestions[0])
                        } else {
                            format!(
                                "Did you mean one of: {}?",
                                suggestions
                                    .iter()
                                    .take(3)
                                    .map(|s| format!("'{s}'"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        }
                    } else {
                        format!(
                            "Valid type identifiers include: {}, void, or named types starting with @",
                            primitive_types.join(", ")
                        )
                    };

                    Err(state.error(format!(
                        "Unknown type identifier: '{potential_primitive}'\n  Hint: {hint}"
                    )))
                }
            }
        }
    }
}
