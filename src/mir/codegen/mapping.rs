//! Type mapping from IR to MIR.
//!
//! Functions to convert IR types to MIR types.
//! The mapping process handles primitive type conversions and validates
//! that only supported types are converted.

use crate::{
    ir::types::{
        PrimitiveType as IRPrim, Type as IRType, align_of_type, sizeof_type,
        struct_field_byte_offset,
    },
    mir::{
        codegen::error::FromIRError,
        types::{
            Aggregate, AggregateClass, AggregateShape, EightbyteClass, FloatMember, MirType,
            ScalarType, Signedness, VectorLane, VectorType,
        },
    },
};
use lamina_platform::{FloatAggregateAbi, TargetDataLayout};

/// Largest aggregate passed in registers. Beyond this size every target in the table,
/// ppc64le included, passes and returns the value in memory through a hidden pointer.
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

const MAX_HOMOGENEOUS_MEMBERS: usize = 4;

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

/// Classifies an aggregate crossing a function boundary. `what` names the value for
/// the error text. Scalars return `Ok(None)`; aggregates outside the supported subset
/// are an error rather than a wrong ABI.
pub fn classify_aggregate(
    ty: &IRType<'_>,
    what: &str,
    layout: TargetDataLayout,
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
    let integer_only = integer_fields_only(ty);
    let size = sizeof_type(ty, layout).ok_or(FromIRError::UnsupportedType)?;
    if size == 0 {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} is empty; an aggregate crossing a boundary holds at least one byte"
        )));
    }
    let class = if integer_only {
        AggregateClass::Integer
    } else {
        match layout.float_aggregates {
            FloatAggregateAbi::Refused => {
                return Err(FromIRError::UnsupportedAggregate(format!(
                    "{what} has a float field, which this target's backend does not place yet"
                )));
            }
            FloatAggregateAbi::SysV => sysv_eightbyte_class(ty, size, what, layout)?,
            FloatAggregateAbi::Aapcs64 => aapcs64_class(ty, size, what, layout)?,
        }
    };
    Ok(Some(Aggregate {
        size: size as usize,
        align: align_of_type(ty, layout) as usize,
        class,
        shape: AggregateShape::Composite,
    }))
}

struct Leaf {
    offset: u64,
    size: u64,
    float: bool,
    member: Option<FloatMember>,
    unaligned: bool,
}

fn aapcs64_class(
    ty: &IRType<'_>,
    size: u64,
    what: &str,
    layout: TargetDataLayout,
) -> Result<AggregateClass, FromIRError> {
    let mut leaves = Vec::new();
    collect_leaves(ty, 0, what, layout, &mut leaves)?;
    let Some(first) = leaves.first().and_then(|leaf| leaf.member) else {
        return Ok(AggregateClass::Integer);
    };
    let homogeneous = leaves.len() <= MAX_HOMOGENEOUS_MEMBERS
        && leaves.iter().all(|leaf| leaf.member == Some(first));
    if !homogeneous {
        return Ok(AggregateClass::Integer);
    }
    if leaves.iter().any(|leaf| leaf.unaligned) {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} has a misaligned float member, which no float placement covers"
        )));
    }
    let members = leaves.len() as u64;
    if members * first.size_bytes() as u64 != size {
        return Err(FromIRError::UnsupportedAggregate(format!(
            "{what} is a float aggregate with padding, which no float placement covers"
        )));
    }
    Ok(AggregateClass::Homogeneous {
        member: first,
        count: members as u8,
    })
}

fn sysv_eightbyte_class(
    ty: &IRType<'_>,
    size: u64,
    what: &str,
    layout: TargetDataLayout,
) -> Result<AggregateClass, FromIRError> {
    if size > MAX_REGISTER_AGGREGATE {
        return Ok(AggregateClass::Memory);
    }
    let mut leaves = Vec::new();
    collect_leaves(ty, 0, what, layout, &mut leaves)?;
    if leaves.iter().any(|leaf| leaf.unaligned) {
        return Ok(AggregateClass::Memory);
    }
    let mut seen: [Option<EightbyteClass>; 2] = [None; 2];
    for leaf in &leaves {
        let class = if leaf.float {
            EightbyteClass::Sse
        } else {
            EightbyteClass::Integer
        };
        let first = (leaf.offset / 8) as usize;
        let last = ((leaf.offset + leaf.size - 1) / 8) as usize;
        for word in first..=last {
            let entry = seen.get_mut(word).ok_or_else(|| {
                FromIRError::UnsupportedAggregate(format!("{what} is over sixteen bytes"))
            })?;
            *entry = Some(match (*entry, class) {
                (Some(EightbyteClass::Integer), _) | (_, EightbyteClass::Integer) => {
                    EightbyteClass::Integer
                }
                _ => EightbyteClass::Sse,
            });
        }
    }
    let words = size.div_ceil(8) as usize;
    let mut classes = [EightbyteClass::Integer; 2];
    for (word, class) in classes.iter_mut().enumerate().take(words) {
        *class = seen[word].ok_or_else(|| {
            FromIRError::UnsupportedAggregate(format!(
                "{what} has an eight-byte word that holds no field"
            ))
        })?;
    }
    Ok(AggregateClass::Eightbytes(classes))
}

fn collect_leaves(
    ty: &IRType<'_>,
    offset: u64,
    what: &str,
    layout: TargetDataLayout,
    leaves: &mut Vec<Leaf>,
) -> Result<(), FromIRError> {
    match ty {
        IRType::Primitive(p) => {
            let size = sizeof_type(ty, layout).ok_or(FromIRError::UnsupportedType)?;
            let align = align_of_type(ty, layout);
            let member = match p {
                IRPrim::F32 => Some(FloatMember::F32),
                IRPrim::F64 => Some(FloatMember::F64),
                _ => None,
            };
            leaves.push(Leaf {
                offset,
                size,
                float: member.is_some(),
                member,
                unaligned: !offset.is_multiple_of(align),
            });
            Ok(())
        }
        IRType::Array { element_type, size } => {
            let element_size =
                sizeof_type(element_type, layout).ok_or(FromIRError::UnsupportedType)?;
            for index in 0..*size {
                collect_leaves(
                    element_type,
                    offset + index * element_size,
                    what,
                    layout,
                    leaves,
                )?;
            }
            Ok(())
        }
        IRType::Struct { fields, packed } => {
            for (index, field) in fields.iter().enumerate() {
                let at = struct_field_byte_offset(fields, *packed, index, layout)
                    .ok_or(FromIRError::UnsupportedType)?;
                collect_leaves(&field.ty, offset + at, what, layout, leaves)?;
            }
            Ok(())
        }
        IRType::Vector { .. } => Err(FromIRError::UnsupportedAggregate(format!(
            "{what} has a vector field, which no aggregate classification covers"
        ))),
        _ => Err(FromIRError::UnsupportedType),
    }
}
