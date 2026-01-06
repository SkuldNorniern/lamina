use crate::{LaminaError, PrimitiveType, Type};
use std::result::Result;

/// Get type size in bytes for any architecture
pub fn get_type_size_bytes(ty: &Type<'_>) -> Result<u64, LaminaError> {
    match ty {
        Type::Primitive(pt) => Ok(match pt {
            PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 | PrimitiveType::Ptr => 8,
        }),
        Type::Array { element_type, size } => {
            let elem_size = get_type_size_bytes(element_type)?;
            Ok(elem_size * size)
        }
        Type::Struct(_) => Err(crate::LaminaError::CodegenError(
            crate::codegen::CodegenError::StructNotImplemented,
        )),
        Type::Tuple(_) => Err(crate::LaminaError::CodegenError(
            crate::codegen::CodegenError::TupleNotImplemented,
        )),
        Type::Named(_) => Err(crate::LaminaError::CodegenError(
            crate::codegen::CodegenError::NamedTypeNotImplemented,
        )),
        #[cfg(feature = "nightly")]
        Type::Vector { element_type, lanes } => {
            let elem_size = match element_type {
                PrimitiveType::I8 | PrimitiveType::U8 => 1,
                PrimitiveType::I16 | PrimitiveType::U16 => 2,
                PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
                _ => return Err(crate::LaminaError::CodegenError(
                    crate::codegen::CodegenError::InvalidVectorElementType,
                )),
            };
            Ok(elem_size * (*lanes as u64))
        }
        Type::Void => Err(crate::LaminaError::CodegenError(
            crate::codegen::CodegenError::VoidTypeSize,
        )),
    }
}

/// Calculate required alignment for a type
pub fn get_type_alignment(ty: &Type<'_>) -> Result<u64, LaminaError> {
    match ty {
        Type::Primitive(pt) => Ok(match pt {
            PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool | PrimitiveType::Char => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 | PrimitiveType::Ptr => 8,
        }),
        Type::Array { element_type, .. } => get_type_alignment(element_type),
        Type::Struct(_) => Ok(8), // Default to 8-byte alignment for structs
        Type::Tuple(_) => Ok(8),  // Default to 8-byte alignment for tuples
        Type::Named(_) => Ok(8),  // Default to 8-byte alignment for named types
        #[cfg(feature = "nightly")]
        Type::Vector { element_type, lanes } => {
            // Vector alignment is typically the vector size (128-bit = 16 bytes, 256-bit = 32 bytes)
            let elem_size = match element_type {
                PrimitiveType::I8 | PrimitiveType::U8 => 1,
                PrimitiveType::I16 | PrimitiveType::U16 => 2,
                PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
                _ => return Err(crate::LaminaError::CodegenError(
                    crate::codegen::CodegenError::InvalidVectorElementType,
                )),
            };
            let vector_size = elem_size * (*lanes as u64);
            // Align to vector size (typically 16 or 32 bytes)
            Ok(vector_size.min(32).max(16))
        }
        Type::Void => Err(crate::LaminaError::CodegenError(
            crate::codegen::CodegenError::VoidTypeSize,
        )),
    }
}

/// Align a value to the specified alignment boundary
pub fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Align a signed value to the specified alignment boundary
pub fn align_to_signed(value: i64, alignment: i64) -> i64 {
    if value >= 0 {
        ((value + alignment - 1) / alignment) * alignment
    } else {
        // For negative values, align downward (away from zero)
        ((value - alignment + 1) / alignment) * alignment
    }
}

/// Check if a value is properly aligned
pub fn is_aligned(value: u64, alignment: u64) -> bool {
    value & (alignment - 1) == 0
}

/// Escape string for assembly output
pub fn escape_asm_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\t' => escaped.push_str("\\t"),
            '\r' => escaped.push_str("\\r"),
            '\0' => escaped.push_str("\\0"),
            c if c.is_control() => escaped.push_str(&format!("\\{:03o}", c as u8)),
            c => escaped.push(c),
        }
    }
    escaped
}

/// Generate a unique temporary name
pub fn generate_temp_name(prefix: &str, counter: &mut u32) -> String {
    let name = format!("{}_{}", prefix, counter);
    *counter += 1;
    name
}

/// Check if an immediate value fits in the specified number of bits (signed)
pub fn fits_in_signed_bits(value: i64, bits: u32) -> bool {
    let min_value = -(1i64 << (bits - 1));
    let max_value = (1i64 << (bits - 1)) - 1;
    value >= min_value && value <= max_value
}

/// Check if an immediate value fits in the specified number of bits (unsigned)
pub fn fits_in_unsigned_bits(value: u64, bits: u32) -> bool {
    if bits >= 64 {
        true
    } else {
        value < (1u64 << bits)
    }
}

/// Calculate the number of bits needed to represent a value
pub fn bits_needed_for_value(value: u64) -> u32 {
    if value == 0 {
        1
    } else {
        64 - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_get_type_size_bytes() {
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::I8)).unwrap(),
            1
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::I16)).unwrap(),
            2
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::I32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::I64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::U8)).unwrap(),
            1
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::U16)).unwrap(),
            2
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::U32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::U64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::F32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::F64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::Bool)).unwrap(),
            1
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::Char)).unwrap(),
            1
        );
        assert_eq!(
            get_type_size_bytes(&Type::Primitive(PrimitiveType::Ptr)).unwrap(),
            8
        );

        let array_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: 10,
        };
        assert_eq!(get_type_size_bytes(&array_type).unwrap(), 40);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_get_type_alignment() {
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::I8)).unwrap(),
            1
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::I16)).unwrap(),
            2
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::I32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::I64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::U8)).unwrap(),
            1
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::U16)).unwrap(),
            2
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::U32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::U64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::F32)).unwrap(),
            4
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::F64)).unwrap(),
            8
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::Bool)).unwrap(),
            1
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::Char)).unwrap(),
            1
        );
        assert_eq!(
            get_type_alignment(&Type::Primitive(PrimitiveType::Ptr)).unwrap(),
            8
        );
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(7, 8), 8);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(9, 8), 16);
        assert_eq!(align_to(0, 8), 0);
    }

    #[test]
    fn test_align_to_signed() {
        assert_eq!(align_to_signed(7, 8), 8);
        assert_eq!(align_to_signed(8, 8), 8);
        assert_eq!(align_to_signed(-7, 8), -8);
        assert_eq!(align_to_signed(-8, 8), -8);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0, 8));
        assert!(is_aligned(8, 8));
        assert!(is_aligned(16, 8));
        assert!(!is_aligned(7, 8));
        assert!(!is_aligned(9, 8));
    }

    #[test]
    fn test_escape_asm_string() {
        assert_eq!(escape_asm_string("simple"), "simple");
        assert_eq!(escape_asm_string("with\"quote"), "with\\\"quote");
        assert_eq!(escape_asm_string("with\\backslash"), "with\\\\backslash");
        assert_eq!(escape_asm_string("new\nline"), "new\\nline");
        assert_eq!(escape_asm_string("tab\ttab"), "tab\\ttab");
    }

    #[test]
    fn test_generate_temp_name() {
        let mut counter = 0;
        assert_eq!(generate_temp_name("temp", &mut counter), "temp_0");
        assert_eq!(generate_temp_name("temp", &mut counter), "temp_1");
        assert_eq!(generate_temp_name("var", &mut counter), "var_2");
    }

    #[test]
    fn test_fits_in_signed_bits() {
        assert!(fits_in_signed_bits(127, 8));
        assert!(fits_in_signed_bits(-128, 8));
        assert!(!fits_in_signed_bits(128, 8));
        assert!(!fits_in_signed_bits(-129, 8));

        assert!(fits_in_signed_bits(32767, 16));
        assert!(fits_in_signed_bits(-32768, 16));
        assert!(!fits_in_signed_bits(32768, 16));
        assert!(!fits_in_signed_bits(-32769, 16));
    }

    #[test]
    fn test_fits_in_unsigned_bits() {
        assert!(fits_in_unsigned_bits(255, 8));
        assert!(!fits_in_unsigned_bits(256, 8));

        assert!(fits_in_unsigned_bits(65535, 16));
        assert!(!fits_in_unsigned_bits(65536, 16));
    }

    #[test]
    fn test_bits_needed_for_value() {
        assert_eq!(bits_needed_for_value(0), 1);
        assert_eq!(bits_needed_for_value(1), 1);
        assert_eq!(bits_needed_for_value(2), 2);
        assert_eq!(bits_needed_for_value(3), 2);
        assert_eq!(bits_needed_for_value(4), 3);
        assert_eq!(bits_needed_for_value(255), 8);
        assert_eq!(bits_needed_for_value(256), 9);
    }
}
