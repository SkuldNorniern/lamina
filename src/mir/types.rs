/// Type system for LUMIR
///
/// LUMIR types are lower-level than IR types, focused on machine representation.
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Ptr,
    I1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorLane {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorType {
    V128(VectorLane),
    V256(VectorLane),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirType {
    Scalar(ScalarType),
    Vector(VectorType),
}

impl MirType {
    /// Size of the type in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            MirType::Scalar(s) => match s {
                ScalarType::I1 => 1,
                ScalarType::I8 => 1,
                ScalarType::I16 => 2,
                ScalarType::I32 => 4,
                ScalarType::I64 => 8,
                ScalarType::F32 => 4,
                ScalarType::F64 => 8,
                ScalarType::Ptr => 8, // Assume 64-bit pointers for now
            },
            MirType::Vector(v) => match v {
                VectorType::V128(_) => 16,
                VectorType::V256(_) => 32,
            },
        }
    }

    /// Alignment requirement in bytes
    pub fn alignment(&self) -> usize {
        match self {
            MirType::Scalar(s) => match s {
                ScalarType::I1 | ScalarType::I8 => 1,
                ScalarType::I16 => 2,
                ScalarType::I32 | ScalarType::F32 => 4,
                ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr => 8,
            },
            MirType::Vector(v) => match v {
                VectorType::V128(_) => 16,
                VectorType::V256(_) => 32,
            },
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            MirType::Scalar(ScalarType::F32 | ScalarType::F64)
                | MirType::Vector(VectorType::V128(VectorLane::F32 | VectorLane::F64))
                | MirType::Vector(VectorType::V256(VectorLane::F32 | VectorLane::F64))
        )
    }

    /// Check if this is a vector type
    pub fn is_vector(&self) -> bool {
        matches!(self, MirType::Vector(_))
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::I1 => write!(f, "i1"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::Ptr => write!(f, "ptr"),
        }
    }
}

impl fmt::Display for VectorLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorLane::I8 => write!(f, "i8"),
            VectorLane::I16 => write!(f, "i16"),
            VectorLane::I32 => write!(f, "i32"),
            VectorLane::I64 => write!(f, "i64"),
            VectorLane::F32 => write!(f, "f32"),
            VectorLane::F64 => write!(f, "f64"),
        }
    }
}

impl fmt::Display for VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorType::V128(lane) => write!(f, "v128<{}>", lane),
            VectorType::V256(lane) => write!(f, "v256<{}>", lane),
        }
    }
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::Scalar(s) => write!(f, "{}", s),
            MirType::Vector(v) => write!(f, "{}", v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_sizes() {
        assert_eq!(MirType::Scalar(ScalarType::I8).size_bytes(), 1);
        assert_eq!(MirType::Scalar(ScalarType::I32).size_bytes(), 4);
        assert_eq!(MirType::Scalar(ScalarType::I64).size_bytes(), 8);
    }

    #[test]
    fn test_vector_sizes() {
        assert_eq!(
            MirType::Vector(VectorType::V128(VectorLane::I32)).size_bytes(),
            16
        );
        assert_eq!(
            MirType::Vector(VectorType::V256(VectorLane::F64)).size_bytes(),
            32
        );
    }

    #[test]
    fn test_float_detection() {
        assert!(MirType::Scalar(ScalarType::F32).is_float());
        assert!(MirType::Scalar(ScalarType::F64).is_float());
        assert!(!MirType::Scalar(ScalarType::I32).is_float());
    }
}
