use super::state::{CodegenState, FunctionContext};
use crate::codegen::{CodegenError, LiteralType};
use crate::{LaminaError, Literal, PrimitiveType, Result, Type, Value};

// Convert an IR Type into AArch64 storage width in bytes and a directive for data sections
pub fn get_type_size_directive_and_bytes(ty: &Type<'_>) -> Result<(&'static str, u64)> {
    match ty {
        Type::Primitive(pt) => match pt {
            PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => {
                Ok((".byte", 1))
            }
            PrimitiveType::I16 | PrimitiveType::U16 => Ok((".hword", 2)),
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => Ok((".word", 4)),
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 | PrimitiveType::Ptr => {
                Ok((".xword", 8))
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
                // For simplicity, assume no padding (fields are packed)
                // In a real implementation, we'd need to handle alignment
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

// Get operand string for an IR value in the context of AArch64 codegen
pub fn get_value_operand_asm<'a>(
    value: &Value<'a>,
    state: &CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
) -> Result<String> {
    match value {
        Value::Constant(literal) => match literal {
            Literal::I32(v) => Ok(format!("#{}", v)),
            Literal::I64(v) => Ok(format!("#{}", v)),
            Literal::Bool(v) => Ok(format!("#{}", if *v { 1 } else { 0 })),
            // F32 literals - convert to integer representation
            Literal::F32(v) => {
                // Convert f32 to raw bits and treat as i32
                let bits = v.to_bits() as i32;
                Ok(format!("#{}", bits))
            }
            // POTENTIAL BUG: String literals not supported in operands - requires global variable workaround
            Literal::String(_) => Err(LaminaError::CodegenError(
                CodegenError::StringLiteralRequiresGlobal,
            )),
            Literal::I8(v) => Ok(format!("#{}", v)),
            Literal::U8(v) => Ok(format!("#{}", v)),
            Literal::I16(v) => Ok(format!("#{}", v)),
            Literal::U16(v) => Ok(format!("#{}", v)),
            Literal::U32(v) => Ok(format!("#{}", v)),
            Literal::U64(v) => Ok(format!("#{}", v)),
            Literal::F64(_v) => {
                // Temporarily disable F64 literals
                Err(LaminaError::CodegenError(
                    CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::Unknown(
                        "F64".to_string(),
                    )),
                ))
            }
            _ => Err(LaminaError::CodegenError(
                CodegenError::UnsupportedLiteralTypeInGlobal(LiteralType::Unknown(format!(
                    "{:?}",
                    value
                ))),
            )),
        },
        Value::Variable(name) => {
            let location = func_ctx.get_value_location(name)?;
            Ok(location.to_operand_string())
        }
        Value::Global(name) => {
            let asm_label = state.global_layout.get(name).ok_or_else(|| {
                LaminaError::CodegenError(CodegenError::GlobalNotFound(name.to_string()))
            })?;
            // ADRP/ADD sequence is used in code emission, here return label placeholder
            Ok(format!("{}(adrp+add)", asm_label))
        }
    }
}

// Utility: materialize an address of a label into a register using ADRP+ADD
// Returns instructions as strings to be emitted
pub fn materialize_label_address(dest_reg: &str, label: &str) -> [String; 2] {
    [
        format!("        adrp {}, {}@PAGE", dest_reg, label),
        format!("        add {}, {}, {}@PAGEOFF", dest_reg, dest_reg, label),
    ]
}
