//! Type mapping from IR to MIR.
//!
//! Functions to convert IR types to MIR types.
//! The mapping process handles primitive type conversions and validates
//! that only supported types are converted.

use crate::{
    ir::types::{PrimitiveType as IRPrim, Type as IRType, align_of_type, sizeof_type},
    mir::{
        codegen::error::FromIRError,
        types::{Aggregate, AggregateClass, MirType, ScalarType, Signedness},
    },
};

/// Largest aggregate passed in registers. Beyond this every target in the table puts
/// the value in memory, except ppc64, which is why ppc64 rejects aggregates for now.
pub const MAX_REGISTER_AGGREGATE: u64 = 16;

pub fn map_ir_prim(p: IRPrim) -> Result<MirType, FromIRError> {
    let scalar = match p {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Char => ScalarType::I8,
        IRPrim::I16 | IRPrim::U16 => ScalarType::I16,
        IRPrim::I32 | IRPrim::U32 => ScalarType::I32,
        IRPrim::I64 | IRPrim::U64 => ScalarType::I64,
        IRPrim::F32 => ScalarType::F32,
        IRPrim::F64 => ScalarType::F64,
        IRPrim::Bool => ScalarType::I1,
        IRPrim::Ptr => ScalarType::Ptr,
    };
    Ok(MirType::Scalar(scalar))
}

pub fn signedness_of(ty: &IRType<'_>) -> Signedness {
    match ty {
        IRType::Primitive(
            IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64 | IRPrim::Bool | IRPrim::Ptr,
        ) => Signedness::Unsigned,
        _ => Signedness::Signed,
    }
}

/// Maps an IR type to a MIR type.
///
/// Currently only primitive types are supported. Composite types (structs,
/// arrays, tuples) will return an error as they require additional lowering.
pub fn map_ir_type(ty: &IRType<'_>) -> Result<MirType, FromIRError> {
    match ty {
        IRType::Primitive(p) => map_ir_prim(*p),
        _ => Err(FromIRError::UnsupportedType),
    }
}

/// Whether an IR type is an aggregate, after named types have been resolved.
pub fn is_aggregate(ty: &IRType<'_>) -> bool {
    matches!(
        ty,
        IRType::Struct(_) | IRType::Tuple(_) | IRType::Array { .. }
    )
}

fn integer_fields_only(ty: &IRType<'_>) -> bool {
    match ty {
        IRType::Primitive(p) => !matches!(p, IRPrim::F32 | IRPrim::F64),
        IRType::Struct(fields) => fields.iter().all(|field| integer_fields_only(&field.ty)),
        IRType::Tuple(elements) => elements.iter().all(integer_fields_only),
        IRType::Array { element_type, .. } => integer_fields_only(element_type),
        _ => false,
    }
}

/// Classifies an aggregate crossing a function boundary. `what` names the value for
/// the error text. Scalars return `Ok(None)`; aggregates outside the supported subset
/// are an error rather than a wrong ABI.
pub fn classify_aggregate(ty: &IRType<'_>, what: &str) -> Result<Option<Aggregate>, FromIRError> {
    if !is_aggregate(ty) {
        return Ok(None);
    }
    if !integer_fields_only(ty) {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} has a float field; only integer aggregates cross a boundary so far"
        )));
    }
    let size = sizeof_type(ty).ok_or(FromIRError::UnsupportedType)?;
    if size == 0 || size > MAX_REGISTER_AGGREGATE {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} is {size} bytes; only 1 to {MAX_REGISTER_AGGREGATE} bytes cross a boundary so far"
        )));
    }
    Ok(Some(Aggregate {
        size: size as usize,
        align: align_of_type(ty) as usize,
        class: AggregateClass::Integer,
    }))
}
