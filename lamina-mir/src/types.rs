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

use lamina_platform::PointerWidth;

/// Which memory a pointer addresses. Only the gpu backends distinguish them; every
/// other target treats them all as `Generic`.
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

/// Signedness of an integer crossing a function boundary. Instructions stay
/// signedness-free; this tells external call sites how to widen narrow arguments
/// and results to the width the C ABI expects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Signedness {
    #[default]
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EightbyteClass {
    Integer,
    Sse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatMember {
    F32,
    F64,
}

impl FloatMember {
    pub fn size_bytes(self) -> usize {
        match self {
            FloatMember::F32 => 4,
            FloatMember::F64 => 8,
        }
    }
}

/// One member an ABI hands a register of its own, with the byte offset it sits at in
/// the aggregate it was flattened out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlatField {
    pub offset: u32,
    pub bytes: u32,
    /// The floating-point type this member is, or `None` when it is an integer that
    /// takes a general register.
    pub float: Option<FloatMember>,
}

/// Field makeup of an aggregate crossing a function boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateClass {
    Integer,
    Eightbytes([EightbyteClass; 2]),
    Memory,
    Homogeneous {
        member: FloatMember,
        count: u8,
    },
    /// One or two flattened members, each in a register of its own class. The RISC-V
    /// hard-float convention, and the single floating-point member that the
    /// z/Architecture ABI treats as that member.
    Flattened {
        first: FlatField,
        second: Option<FlatField>,
    },
}

impl AggregateClass {
    pub fn flattened(&self) -> Option<(FlatField, Option<FlatField>)> {
        match self {
            AggregateClass::Flattened { first, second } => Some((*first, *second)),
            _ => None,
        }
    }
}

/// What a value of aggregate shape actually is. A [`AggregateShape::Composite`] is a
/// struct or an array the author wrote; a [`AggregateShape::WideScalar`] is one
/// integer that legalization split across n registers because no register on the
/// target is wide enough to hold it.
///
/// The two reach a backend with the same size, alignment and class, and several ABIs
/// place them differently: AAPCS returns a composite over four bytes in memory while
/// a 64-bit integer comes back in `r0:r1`, and cdecl returns a struct through a
/// hidden pointer while a 64-bit integer comes back in `%edx:%eax`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AggregateShape {
    #[default]
    Composite,
    WideScalar,
}

/// An aggregate passed by value. The parameter register holds the address of the
/// callee's own copy; size and class tell a backend which registers carry it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aggregate {
    pub size: usize,
    pub align: usize,
    pub class: AggregateClass,
    pub shape: AggregateShape,
}

impl Aggregate {
    /// Whether this is a struct or an array rather than one legalized integer.
    pub fn is_composite(&self) -> bool {
        self.shape == AggregateShape::Composite
    }

    /// Whether this is one integer that legalization split across registers.
    pub fn is_wide_scalar(&self) -> bool {
        self.shape == AggregateShape::WideScalar
    }

    /// Eight-byte words the aggregate occupies in registers or on the stack.
    pub fn words(&self) -> usize {
        self.size.div_ceil(8)
    }

    pub fn eightbyte_class(&self, word: usize) -> EightbyteClass {
        match self.class {
            AggregateClass::Eightbytes(classes) => classes[word],
            AggregateClass::Homogeneous { .. } => EightbyteClass::Sse,
            AggregateClass::Integer | AggregateClass::Memory | AggregateClass::Flattened { .. } => {
                EightbyteClass::Integer
            }
        }
    }

    pub fn homogeneous_members(&self) -> Option<(FloatMember, usize)> {
        match self.class {
            AggregateClass::Homogeneous { member, count } => Some((member, count as usize)),
            AggregateClass::Integer
            | AggregateClass::Eightbytes(_)
            | AggregateClass::Memory
            | AggregateClass::Flattened { .. } => None,
        }
    }

    /// Register-sized words the aggregate occupies on a target of this pointer width.
    /// A 32-bit target holds twice as many of them as [`Aggregate::words`] counts.
    pub fn words_of(&self, pointer: PointerWidth) -> usize {
        self.size.div_ceil(pointer.bytes())
    }

    /// The largest aggregate a target of this pointer width passes in registers:
    /// two words, as every ABI in the table classifies it.
    pub fn max_in_registers(pointer: PointerWidth) -> u64 {
        2 * pointer.bytes() as u64
    }
}

/// A value that fits in one register. `I1` is a one-bit boolean, which the backends
/// still hold in a full register.
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

/// Element type of a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorLane {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

/// A vector, named by its total width and its lane type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorType {
    V128(VectorLane),
    V256(VectorLane),
}

/// Every type MIR can name. MIR is a register machine, so an aggregate is not one of
/// them: an aggregate crossing a function boundary travels as a pointer plus an
/// [`Aggregate`] marker on the parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirType {
    Scalar(ScalarType),
    Vector(VectorType),
}

impl MirType {
    /// Reads `value` as this type's two's complement, sign-extended to 64 bits.
    pub fn signed_integer(self, value: i64, pointer: PointerWidth) -> i64 {
        let shift = 64 - self.size_bytes(pointer) * 8;
        (value << shift) >> shift
    }

    /// Reads `value` as this type's unsigned bits, zero-extended to 64 bits.
    pub fn unsigned_integer(self, value: i64, pointer: PointerWidth) -> u64 {
        let shift = 64 - self.size_bytes(pointer) * 8;
        ((value as u64) << shift) >> shift
    }

    /// Size of the type in bytes
    pub fn size_bytes(&self, pointer: PointerWidth) -> usize {
        match self {
            MirType::Scalar(s) => match s {
                ScalarType::I1 | ScalarType::I8 => 1,
                ScalarType::I16 => 2,
                ScalarType::I32 | ScalarType::F32 => 4,
                ScalarType::I64 | ScalarType::F64 => 8,
                ScalarType::Ptr | ScalarType::PtrIn(_) => pointer.bytes(),
            },
            MirType::Vector(v) => match v {
                VectorType::V128(_) => 16,
                VectorType::V256(_) => 32,
            },
        }
    }

    /// Alignment requirement in bytes
    pub fn alignment(&self, pointer: PointerWidth) -> usize {
        match self {
            MirType::Scalar(s) => match s {
                ScalarType::I1 | ScalarType::I8 => 1,
                ScalarType::I16 => 2,
                ScalarType::I32 | ScalarType::F32 => 4,
                ScalarType::I64 | ScalarType::F64 => 8,
                ScalarType::Ptr | ScalarType::PtrIn(_) => pointer.bytes(),
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

    /// A pointer into `space`.
    pub fn ptr_in(space: AddressSpace) -> Self {
        Self::Scalar(ScalarType::PtrIn(space))
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
            assert_eq!(
                ty.size_bytes(PointerWidth::Bits64),
                MirType::Scalar(ScalarType::Ptr).size_bytes(PointerWidth::Bits64)
            );
            assert_eq!(
                ty.alignment(PointerWidth::Bits64),
                MirType::Scalar(ScalarType::Ptr).alignment(PointerWidth::Bits64)
            );
        }
        assert_ne!(
            MirType::ptr_in(AddressSpace::Global),
            MirType::ptr_in(AddressSpace::Local)
        );
    }

    #[test]
    fn test_scalar_sizes() {
        let w = PointerWidth::Bits64;
        assert_eq!(MirType::Scalar(ScalarType::I8).size_bytes(w), 1);
        assert_eq!(MirType::Scalar(ScalarType::I32).size_bytes(w), 4);
        assert_eq!(MirType::Scalar(ScalarType::I64).size_bytes(w), 8);
    }

    #[test]
    fn pointer_size_and_alignment_follow_pointer_width() {
        for ty in [
            MirType::Scalar(ScalarType::Ptr),
            MirType::ptr_in(AddressSpace::Local),
        ] {
            assert_eq!(ty.size_bytes(PointerWidth::Bits32), 4);
            assert_eq!(ty.alignment(PointerWidth::Bits32), 4);
            assert_eq!(ty.size_bytes(PointerWidth::Bits64), 8);
            assert_eq!(ty.alignment(PointerWidth::Bits64), 8);
        }
        assert_eq!(
            MirType::Scalar(ScalarType::I64).size_bytes(PointerWidth::Bits32),
            8
        );
        assert_eq!(
            MirType::Scalar(ScalarType::I64).alignment(PointerWidth::Bits32),
            8
        );
    }

    #[test]
    fn test_vector_sizes() {
        assert_eq!(
            MirType::Vector(VectorType::V128(VectorLane::I32)).size_bytes(PointerWidth::Bits64),
            16
        );
        assert_eq!(
            MirType::Vector(VectorType::V256(VectorLane::F64)).size_bytes(PointerWidth::Bits64),
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
