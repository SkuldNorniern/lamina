//! Type mapping from IR to MIR.
//!
//! Functions to convert IR types to MIR types.
//! The mapping process handles primitive type conversions and validates
//! that only supported types are converted.

use crate::{
    ir::types::{PrimitiveType as IRPrim, Type as IRType, align_of_type, sizeof_type},
    mir::{
        codegen::error::FromIRError,
        types::{
            Aggregate, AggregateClass, AggregateShape, MirType, ScalarType, Signedness, VectorLane,
            VectorType,
        },
    },
};
use lamina_platform::PointerWidth;

/// Largest aggregate passed in registers. Beyond this every target in the table puts
/// the value in memory, except ppc64le, which is why ppc64le rejects aggregates for now.
pub const MAX_REGISTER_AGGREGATE: u64 = 16;

/// Maps an IR primitive to the MIR type that holds it. Signedness is dropped, because
/// MIR carries it on the operation rather than the value.
pub fn map_ir_prim(p: IRPrim) -> Result<MirType, FromIRError> {
    let scalar = match p {
        IRPrim::I8 | IRPrim::U8 => ScalarType::I8,
        IRPrim::I16 | IRPrim::U16 => ScalarType::I16,
        IRPrim::I32 | IRPrim::U32 => ScalarType::I32,
        IRPrim::I64 | IRPrim::U64 => ScalarType::I64,
        IRPrim::F32 => ScalarType::F32,
        IRPrim::F64 => ScalarType::F64,
        IRPrim::Bool => ScalarType::I1,
        IRPrim::Ptr => ScalarType::Ptr,
        IRPrim::I128 | IRPrim::U128 => return Err(FromIRError::UnsupportedType),
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
/// Currently only primitive types are supported. Composite types (structs
/// and arrays) will return an error as they require additional lowering.
pub fn map_ir_type(ty: &IRType<'_>) -> Result<MirType, FromIRError> {
    match ty {
        IRType::Primitive(p) => map_ir_prim(*p),
        IRType::Vector {
            element_type,
            lanes,
        } => map_ir_vector(*element_type, *lanes),
        _ => Err(FromIRError::UnsupportedType),
    }
}

/// Maps `<lanes x element_type>` to the vector register that holds it. A shape that
/// fills no register is refused rather than split across several.
pub fn map_ir_vector(element_type: IRPrim, lanes: u32) -> Result<MirType, FromIRError> {
    let lane = match element_type {
        IRPrim::I8 | IRPrim::U8 => VectorLane::I8,
        IRPrim::I16 | IRPrim::U16 => VectorLane::I16,
        IRPrim::I32 | IRPrim::U32 => VectorLane::I32,
        IRPrim::I64 | IRPrim::U64 => VectorLane::I64,
        IRPrim::F32 => VectorLane::F32,
        IRPrim::F64 => VectorLane::F64,
        IRPrim::Bool | IRPrim::Ptr | IRPrim::I128 | IRPrim::U128 => {
            return Err(FromIRError::UnsupportedType);
        }
    };
    let bits = lanes
        .checked_mul(lane_bits(lane))
        .ok_or(FromIRError::UnsupportedType)?;
    Ok(MirType::Vector(match bits {
        128 => VectorType::V128(lane),
        256 => VectorType::V256(lane),
        _ => return Err(FromIRError::UnsupportedType),
    }))
}

const fn lane_bits(lane: VectorLane) -> u32 {
    match lane {
        VectorLane::I8 => 8,
        VectorLane::I16 => 16,
        VectorLane::I32 | VectorLane::F32 => 32,
        VectorLane::I64 | VectorLane::F64 => 64,
    }
}

/// Whether an IR type is an aggregate, after named types have been resolved.
pub fn is_aggregate(ty: &IRType<'_>) -> bool {
    matches!(ty, IRType::Struct { .. } | IRType::Array { .. })
}

fn integer_fields_only(ty: &IRType<'_>) -> bool {
    match ty {
        IRType::Primitive(p) => !matches!(p, IRPrim::F32 | IRPrim::F64),
        IRType::Struct { fields, .. } => fields.iter().all(|field| integer_fields_only(&field.ty)),
        IRType::Array { element_type, .. } => integer_fields_only(element_type),
        _ => false,
    }
}

/// Classifies an aggregate crossing a function boundary. `what` names the value for
/// the error text. Scalars return `Ok(None)`; aggregates outside the supported subset
/// are an error rather than a wrong ABI.
pub fn classify_aggregate(
    ty: &IRType<'_>,
    what: &str,
    pointer: PointerWidth,
    int64_regs: bool,
) -> Result<Option<Aggregate>, FromIRError> {
    classify_aggregate_within(ty, what, None, pointer, int64_regs)
}

/// The ABI of the integer that takes two registers, crossing a boundary: two words,
/// aligned to its own size, one INTEGER aggregate, as System V classifies `__int128`.
pub fn wide_aggregate(int64_regs: bool) -> Aggregate {
    let size = if int64_regs { 16 } else { 8 };
    Aggregate {
        size,
        align: size,
        class: AggregateClass::Integer,
        shape: AggregateShape::WideScalar,
    }
}

fn classify_aggregate_within(
    ty: &IRType<'_>,
    what: &str,
    max: Option<u64>,
    pointer: PointerWidth,
    int64_regs: bool,
) -> Result<Option<Aggregate>, FromIRError> {
    if matches!(ty, IRType::Primitive(IRPrim::I128 | IRPrim::U128)) {
        if !int64_regs {
            return Err(FromIRError::UnsupportedType);
        }
        return Ok(Some(wide_aggregate(int64_regs)));
    }
    if !int64_regs && matches!(ty, IRType::Primitive(IRPrim::I64 | IRPrim::U64)) {
        return Ok(Some(wide_aggregate(int64_regs)));
    }
    if !is_aggregate(ty) {
        return Ok(None);
    }
    if !integer_fields_only(ty) {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} has a float field; only integer aggregates cross a boundary so far"
        )));
    }
    let size = sizeof_type(ty, pointer).ok_or(FromIRError::UnsupportedType)?;
    if size == 0 {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} is empty; an aggregate crossing a boundary holds at least one byte"
        )));
    }
    if let Some(max) = max
        && size > max
    {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} is {size} bytes; only 1 to {max} bytes come back in registers so far"
        )));
    }
    Ok(Some(Aggregate {
        size: size as usize,
        align: align_of_type(ty, pointer) as usize,
        class: AggregateClass::Integer,
        shape: AggregateShape::Composite,
    }))
}
