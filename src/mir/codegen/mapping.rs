use super::error::FromIRError;
use crate::mir::types::{MirType, ScalarType};

pub fn map_ir_prim(p: crate::ir::types::PrimitiveType) -> Result<MirType, FromIRError> {
    use crate::ir::types::PrimitiveType as IRPrim;
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

pub fn map_ir_type(ty: &crate::ir::types::Type<'_>) -> Result<MirType, FromIRError> {
    use crate::ir::types::Type as IRType;
    match ty {
        IRType::Primitive(p) => map_ir_prim(*p),
        _ => Err(FromIRError::UnsupportedType),
    }
}


