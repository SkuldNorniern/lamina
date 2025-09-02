use super::state::{CodegenState, FunctionContext};
use crate::{LaminaError, Literal, PrimitiveType, Result, Type, Value};

// Convert an IR Type into AArch64 storage width in bytes and a directive for data sections
pub fn get_type_size_directive_and_bytes(ty: &Type<'_>) -> Result<(&'static str, u64)> {
    match ty {
        Type::Primitive(pt) => match pt {
            PrimitiveType::I8 | PrimitiveType::Bool => Ok((".byte", 1)),
            PrimitiveType::I32 | PrimitiveType::F32 => Ok((".word", 4)),
            PrimitiveType::I64 | PrimitiveType::Ptr => Ok((".xword", 8)),
            _ => return Err(LaminaError::CodegenError(format!("Unsupported primitive type: {:?}", pt))),
        },
        Type::Array { element_type, size } => {
            let (_, elem) = get_type_size_directive_and_bytes(element_type)?;
            Ok((".space", elem * size))
        }
        Type::Struct(_) => Err(LaminaError::CodegenError(
            "Struct size calculation not implemented yet".to_string(),
        )),
        Type::Tuple(_) => Err(LaminaError::CodegenError(
            "Tuple size calculation not implemented yet".to_string(),
        )),
        Type::Named(_) => Err(LaminaError::CodegenError(
            "Named type size calculation requires lookup (not implemented yet)".to_string(),
        )),
        Type::Void => Err(LaminaError::CodegenError(
            "Cannot get size of void type".to_string(),
        )),
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
            Literal::F32(_v) => Err(LaminaError::CodegenError(
                "f32 literal operand not implemented".to_string(),
            )),
            Literal::String(_) => Err(LaminaError::CodegenError(
                "String literal operand requires label (use global var)".to_string(),
            )),
            Literal::I8(v) => Ok(format!("#{}", v)),
            _ => return Err(LaminaError::CodegenError(format!("Unsupported literal type for operand: {:?}", value))),
        },
        Value::Variable(name) => {
            let location = func_ctx.get_value_location(name)?;
            Ok(location.to_operand_string())
        }
        Value::Global(name) => {
            let asm_label = state.global_layout.get(name).ok_or_else(|| {
                LaminaError::CodegenError(format!("Global '{}' not found in layout map", name))
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




