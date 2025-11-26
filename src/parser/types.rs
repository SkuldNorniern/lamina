use super::state::ParserState;
use crate::{LaminaError, PrimitiveType, StructField, Type, TypeDeclaration};

pub fn parse_type_declaration<'a>(
    state: &mut ParserState<'a>,
) -> Result<TypeDeclaration<'a>, LaminaError> {
    state.consume_keyword("type")?;
    let name = state.parse_type_identifier()?;
    state.expect_char('=')?;
    let ty = parse_composite_type(state)?;
    Ok(TypeDeclaration { name, ty })
}

pub fn parse_composite_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    let keyword_slice = state.peek_slice(6).unwrap_or(""); // Peek "struct"
    if keyword_slice.starts_with("struct") {
        // Struct: struct { name: type, ... }
        state.consume_keyword("struct")?;
        state.expect_char('{')?;
        let mut fields = Vec::new();
        loop {
            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') {
                state.advance();
                break;
            }
            let field_name = state.parse_identifier_str()?;
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
            state.expect_char(',')?; // Expect comma or closing brace
        }
        Ok(Type::Struct(fields))
    } else if state.current_char() == Some('[') {
        // Array: [ size x type ]
        state.expect_char('[')?;
        let size = state.parse_integer()? as u64;
        state.consume_keyword("x")?;
        let elem_type = parse_type(state)?;
        state.expect_char(']')?;
        Ok(Type::Array {
            element_type: Box::new(elem_type),
            size,
        })
    } else {
        Err(state.error("Expected 'struct' or '[' for composite type".to_string()))
    }
}

pub fn parse_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>, LaminaError> {
    state.skip_whitespace_and_comments();
    match state.current_char() {
        Some('@') => {
            // Named type: @Name
            Ok(Type::Named(state.parse_type_identifier()?))
        }
        Some('[') => {
            // Inline Array Type
            parse_composite_type(state) // Delegate to handle arrays
        }
        Some('s') if state.peek_slice(6) == Some("struct") => {
            // Inline Struct Type
            parse_composite_type(state) // Delegate to handle structs
        }
        Some('t') if state.peek_slice(5) == Some("tuple") => {
            // Tuple type: tuple type1, type2, ...
            state.consume_keyword("tuple")?;
            state.skip_whitespace_and_comments();
            let mut element_types = Vec::new();
            loop {
                let elem_type = parse_type(state)?;
                element_types.push(elem_type);
                state.skip_whitespace_and_comments();
                if state.current_char() != Some(',') {
                    break; // No more elements
                }
                state.expect_char(',')?;
                state.skip_whitespace_and_comments();
            }
            Ok(Type::Tuple(element_types))
        }
        _ => {
            // Try matching primitive types
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
                _ => Err(state.error(format!(
                    "Unknown or unexpected type identifier: {}",
                    potential_primitive
                ))),
            }
        }
    }
}
