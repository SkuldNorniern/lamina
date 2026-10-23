//! Type system for LUMIR.
//!
//! LUMIR types are lower-level than IR types, focused on machine representation
//! and efficient code generation. The type system supports:
//!
//! - **Scalar types**: Integer and floating-point types of various sizes
//! - **Vector types**: SIMD vector types for parallel operations
//! - **Pointer types**: Memory addresses
//!
//! Integer scalars carry width-specific bit patterns without signedness. Memory loads
//! zero-extend integer bits when host registers are wider than the access. Signed
//! interpretation belongs to signed operations and explicit `IntConvert` extension.
//!
//! ## Type Sizes
//!
//! All type sizes are fixed and known at compile time, enabling efficient
//! memory layout and register allocation decisions.
use std::fmt;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    #[default]
    Generic,
    Global,
    Region,
    Local,
    Constant,
    Private,
    Parameter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Ptr,
    PtrIn(AddressSpace),
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
    pub fn signed_integer(self, value: i64) -> i64 {
        let shift = 64 - self.size_bytes() * 8;
        (value << shift) >> shift
    }

    pub fn unsigned_integer(self, value: i64) -> u64 {
        let shift = 64 - self.size_bytes() * 8;
        ((value as u64) << shift) >> shift
    }

    /// Size of the type in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            MirType::Scalar(s) => match s {
                ScalarType::I1 | ScalarType::I8 => 1,
                ScalarType::I16 => 2,
                ScalarType::I32 | ScalarType::F32 => 4,
                ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr | ScalarType::PtrIn(_) => 8,
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
                ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr | ScalarType::PtrIn(_) => 8,
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

    /// Shorthand for `MirType::Scalar(ScalarType::I1)`.
    pub fn i1() -> Self {
        MirType::Scalar(ScalarType::I1)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::I8)`.
    pub fn i8() -> Self {
        MirType::Scalar(ScalarType::I8)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::I16)`.
    pub fn i16() -> Self {
        MirType::Scalar(ScalarType::I16)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::I32)`.
    pub fn i32() -> Self {
        MirType::Scalar(ScalarType::I32)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::I64)`.
    pub fn i64() -> Self {
        MirType::Scalar(ScalarType::I64)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::F32)`.
    pub fn f32() -> Self {
        MirType::Scalar(ScalarType::F32)
    }

    /// Shorthand for `MirType::Scalar(ScalarType::F64)`.
    pub fn f64() -> Self {
        MirType::Scalar(ScalarType::F64)
    }

    pub fn ptr_in(space: AddressSpace) -> Self {
        Self::Scalar(ScalarType::PtrIn(space))
    }

    /// Shorthand for `MirType::Scalar(ScalarType::Ptr)`.
    pub fn ptr() -> Self {
        MirType::Scalar(ScalarType::Ptr)
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
            ScalarType::PtrIn(space) => write!(f, "ptr<{space:?}>"),
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
            VectorType::V128(lane) => write!(f, "v128<{lane}>"),
            VectorType::V256(lane) => write!(f, "v256<{lane}>"),
        }
    }
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::Scalar(s) => write!(f, "{s}"),
            MirType::Vector(v) => write!(f, "{v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_spaces_preserve_layout() {
        for space in [
            AddressSpace::Generic,
            AddressSpace::Global,
            AddressSpace::Region,
            AddressSpace::Local,
            AddressSpace::Constant,
            AddressSpace::Private,
            AddressSpace::Parameter,
        ] {
            let ty = MirType::ptr_in(space);
            assert_eq!(ty.size_bytes(), MirType::ptr().size_bytes());
            assert_eq!(ty.alignment(), MirType::ptr().alignment());
        }
        assert_ne!(
            MirType::ptr_in(AddressSpace::Global),
            MirType::ptr_in(AddressSpace::Local)
        );
    }

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
