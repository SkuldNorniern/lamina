use super::state::{CodegenState, FunctionContext};
use crate::codegen::CodegenError;
use crate::{LaminaError, Literal, PrimitiveType, Type, Value};
use std::result::Result;

// Helper to get assembly size directive and size in bytes (simplified)
pub fn get_type_size_directive_and_bytes(ty: &Type<'_>) -> Result<(&'static str, u64), LaminaError> {
    match ty {
        Type::Primitive(pt) => match pt {
            PrimitiveType::I8 => Ok((".byte", 1)),
            PrimitiveType::I16 => Ok((".word", 2)),
            PrimitiveType::I32 => Ok((".long", 4)),
            PrimitiveType::I64 => Ok((".quad", 8)),
            PrimitiveType::U8 => Ok((".byte", 1)),
            PrimitiveType::U16 => Ok((".word", 2)),
            PrimitiveType::U32 => Ok((".long", 4)),
            PrimitiveType::U64 => Ok((".quad", 8)),
            PrimitiveType::F32 => Ok((".long", 4)), // Assuming 32-bit float
            PrimitiveType::F64 => Ok((".quad", 8)), // 64-bit float
            PrimitiveType::Bool => Ok((".byte", 1)),
            PrimitiveType::Char => Ok((".byte", 1)), // Char is 8-bit
            PrimitiveType::Ptr => Ok((".quad", 8)),  // Assuming 64-bit pointers
        },
        Type::Array { element_type, size } => {
            let (_, elem_size) = get_type_size_directive_and_bytes(element_type)?;
            Ok((".space", elem_size * size)) // Use .space for aggregate types, directive isn't really used here
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

// Helper to get the assembly operand string for an IR value
pub fn get_value_operand_asm<'a>(
    value: &Value<'a>,
    state: &CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
) -> Result<String, LaminaError> {
    match value {
        Value::Constant(literal) => match literal {
            Literal::I8(v) => Ok(format!("${}", v)),
            Literal::I16(v) => Ok(format!("${}", v)),
            Literal::I32(v) => Ok(format!("${}", v)),
            Literal::I64(v) => Ok(format!("${}", v)),
            Literal::U8(v) => Ok(format!("${}", v)),
            Literal::U16(v) => Ok(format!("${}", v)),
            Literal::U32(v) => Ok(format!("${}", v)),
            Literal::U64(v) => Ok(format!("${}", v)),
            Literal::F32(v) => {
                // For now, convert to integer representation
                // This is a temporary implementation - proper float constant handling
                // should use .float/.double directives in data section
                let bits = v.to_bits() as i64;
                Ok(format!("${}", bits))
            }
            Literal::F64(v) => {
                // For now, convert to integer representation
                // This is a temporary implementation - proper float constant handling
                // should use .float/.double directives in data section
                let bits = v.to_bits() as i64;
                Ok(format!("${}", bits))
            }
            Literal::Bool(v) => Ok(format!("${}", if *v { 1 } else { 0 })),
            Literal::Char(c) => Ok(format!("${}", *c as u8)),
            Literal::String(_) => Err(LaminaError::CodegenError(
                CodegenError::StringLiteralRequiresGlobal,
            )),
        },
        Value::Variable(name) => {
            // Use the location assigned during precompute/prologue
            let location = func_ctx.get_value_location(name)?;
            Ok(location.to_operand_string())
        }
        Value::Global(name) => {
            let asm_label = state.global_layout.get(name).ok_or_else(|| {
                LaminaError::CodegenError(CodegenError::GlobalNotFound(name.to_string()))
            })?;
            Ok(format!("{}(%rip)", asm_label))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::x86_64::state::ValueLocation;
    use crate::ir::types::*;

    #[test]
    fn test_get_type_size_directive_and_bytes_primitives() {
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::I8)).unwrap(),
            (".byte", 1)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::I16)).unwrap(),
            (".word", 2)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::I32)).unwrap(),
            (".long", 4)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::I64)).unwrap(),
            (".quad", 8)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::U8)).unwrap(),
            (".byte", 1)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::U16)).unwrap(),
            (".word", 2)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::U32)).unwrap(),
            (".long", 4)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::U64)).unwrap(),
            (".quad", 8)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::F32)).unwrap(),
            (".long", 4)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::F64)).unwrap(),
            (".quad", 8)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Bool)).unwrap(),
            (".byte", 1)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Char)).unwrap(),
            (".byte", 1)
        );
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Primitive(PrimitiveType::Ptr)).unwrap(),
            (".quad", 8)
        );
    }

    #[test]
    fn test_get_type_size_directive_and_bytes_array() {
        let arr_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: 10,
        };
        assert_eq!(
            get_type_size_directive_and_bytes(&arr_type).unwrap(),
            (".space", 40)
        );
        let arr_type_ptr = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::Ptr)),
            size: 5,
        };
        assert_eq!(
            get_type_size_directive_and_bytes(&arr_type_ptr).unwrap(),
            (".space", 40)
        );
    }

    #[test]
    fn test_get_type_size_directive_and_bytes_invalid() {
        // Empty struct is now valid (size 0)
        assert_eq!(
            get_type_size_directive_and_bytes(&Type::Struct(vec![])).unwrap(),
            (".space", 0)
        );
        assert!(get_type_size_directive_and_bytes(&Type::Tuple(vec![])).is_err());
        assert!(get_type_size_directive_and_bytes(&Type::Named("MyType")).is_err());
        assert!(get_type_size_directive_and_bytes(&Type::Void).is_err());
    }

    // --- Tests for get_value_operand_asm ---

    // Helper to create mock state and context for get_value_operand_asm tests
    fn setup_test_state_context<'a>() -> (CodegenState<'a>, FunctionContext<'a>) {
        let mut state = CodegenState::new();
        state
            .global_layout
            .insert("my_global_var", "my_global_var_label".to_string());
        state
            .global_layout
            .insert("another_global", "another_global_label".to_string());

        let mut func_ctx = FunctionContext::new();
        // Pre-assign locations for some variables
        func_ctx
            .value_locations
            .insert("local1", ValueLocation::StackOffset(-8));
        func_ctx
            .value_locations
            .insert("param1", ValueLocation::Register("%rdi".to_string()));
        func_ctx
            .value_locations
            .insert("temp_ptr", ValueLocation::StackOffset(-16));

        (state, func_ctx)
    }

    #[test]
    fn test_get_value_operand_asm_constants() {
        let (state, func_ctx) = setup_test_state_context();
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::I8(123)), &state, &func_ctx).unwrap(),
            "$123"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::I16(-456)), &state, &func_ctx).unwrap(),
            "$-456"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::I32(123)), &state, &func_ctx).unwrap(),
            "$123"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::I64(-5)), &state, &func_ctx).unwrap(),
            "$-5"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::U8(123)), &state, &func_ctx).unwrap(),
            "$123"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::U16(456)), &state, &func_ctx).unwrap(),
            "$456"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::U32(123)), &state, &func_ctx).unwrap(),
            "$123"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::U64(456)), &state, &func_ctx).unwrap(),
            "$456"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::Bool(true)), &state, &func_ctx)
                .unwrap(),
            "$1"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::Bool(false)), &state, &func_ctx)
                .unwrap(),
            "$0"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::Char('A')), &state, &func_ctx).unwrap(),
            "$65"
        );
        // F32 and F64 literals are now supported (converted to integer bit representation)
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::F32(1.0)), &state, &func_ctx).unwrap(),
            "$1065353216" // 1.0f32 in IEEE 754 bit representation
        );
        assert_eq!(
            get_value_operand_asm(&Value::Constant(Literal::F64(2.5)), &state, &func_ctx).unwrap(),
            "$4612811918334230528" // 2.5f64 in IEEE 754 bit representation
        );
        assert!(
            get_value_operand_asm(&Value::Constant(Literal::String("hi")), &state, &func_ctx)
                .is_err()
        );
    }

    #[test]
    fn test_get_value_operand_asm_variable() {
        let (state, func_ctx) = setup_test_state_context();
        // Variable on stack
        assert_eq!(
            get_value_operand_asm(&Value::Variable("local1"), &state, &func_ctx).unwrap(),
            "-8(%rbp)"
        );
        // Variable in register (parameter)
        assert_eq!(
            get_value_operand_asm(&Value::Variable("param1"), &state, &func_ctx).unwrap(),
            "%rdi"
        );
        // Variable not found (should error)
        assert!(get_value_operand_asm(&Value::Variable("nonexistent"), &state, &func_ctx).is_err());
    }

    #[test]
    fn test_get_value_operand_asm_global() {
        let (state, func_ctx) = setup_test_state_context();
        assert_eq!(
            get_value_operand_asm(&Value::Global("my_global_var"), &state, &func_ctx).unwrap(),
            "my_global_var_label(%rip)"
        );
        assert_eq!(
            get_value_operand_asm(&Value::Global("another_global"), &state, &func_ctx).unwrap(),
            "another_global_label(%rip)"
        );
        // Global not found (should error)
        assert!(
            get_value_operand_asm(&Value::Global("unknown_global"), &state, &func_ctx).is_err()
        );
    }
}
