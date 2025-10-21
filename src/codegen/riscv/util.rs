use super::state::{CodegenState, FunctionContext};
use crate::codegen::{CodegenError, LiteralType};
use crate::{LaminaError, Literal, PrimitiveType, Result, Type, Value};

// Convert an IR Type into RISC-V storage width in bytes and a directive for data sections
pub fn get_type_size_directive_and_bytes(ty: &Type<'_>) -> Result<(&'static str, u64)> {
    match ty {
        Type::Primitive(pt) => match pt {
            PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => {
                Ok((".byte", 1))
            }
            PrimitiveType::I16 | PrimitiveType::U16 => Ok((".half", 2)),
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => Ok((".word", 4)),
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 | PrimitiveType::Ptr => {
                Ok((".dword", 8))
            }
        },
        Type::Array { element_type, size } => {
            let (_, elem) = get_type_size_directive_and_bytes(element_type)?;
            Ok((".space", elem * size))
        }
        Type::Struct(fields) => {
            let mut total_size = 0u64;
            for field in fields {
                let (_, field_size) = get_type_size_directive_and_bytes(&field.ty)?;
                total_size += field_size;
            }
            Ok((".space", total_size))
        }
        Type::Tuple(_) => Err(LaminaError::CodegenError(CodegenError::TupleNotImplemented)),
        Type::Named(_) => Err(LaminaError::CodegenError(
            CodegenError::NamedTypeNotImplemented,
        )),
        Type::Void => Err(LaminaError::CodegenError(CodegenError::VoidTypeSize)),
    }
}

// Get operand string for an IR value in the context of RISC-V codegen (very basic)
pub fn get_value_operand_asm<'a>(
    value: &Value<'a>,
    _state: &CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
) -> Result<String> {
    match value {
        Value::Constant(literal) => match literal {
            Literal::I32(v) => Ok(format!("{}", v)),
            Literal::I64(v) => Ok(format!("{}", v)),
            Literal::Bool(v) => Ok(if *v { "1".to_string() } else { "0".to_string() }),
            Literal::F32(v) => Ok(format!("{}", v.to_bits() as i32)),
            Literal::I8(v) => Ok(format!("{}", v)),
            Literal::U8(v) => Ok(format!("{}", v)),
            Literal::I16(v) => Ok(format!("{}", v)),
            Literal::U16(v) => Ok(format!("{}", v)),
            Literal::U32(v) => Ok(format!("{}", v)),
            Literal::U64(v) => Ok(format!("{}", v)),
            Literal::F64(_v) => Err(LaminaError::CodegenError(
                CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::F64),
            )),
            _ => Err(LaminaError::CodegenError(
                CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::Unknown(format!(
                    "{:?}",
                    value
                ))),
            )),
        },
        Value::Variable(name) => Ok(func_ctx.get_value_location(name)?.to_operand_string()),
        Value::Global(name) => Ok(format!("{}", name)),
    }
}
