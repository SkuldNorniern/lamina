//! # Type System
//!
//! Complete type system for the Lamina IR, including
//! primitive types, composite types, and value representations.
//!
//! ## Type Categories
//!
//! ### Primitive Types
//! - **Integers**: Signed and unsigned integers of various sizes
//! - **Floats**: Single and double precision floating point numbers
//! - **Booleans**: True/false values
//! - **Pointers**: Generic pointer type for memory addresses
//!
//! ### Composite Types
//! - **Arrays**: Fixed-size sequences of elements of the same type
//! - **Structs**: Named collections of fields with different types
//! - **Named Types**: User-defined type aliases
//!
//! ### Special Types
//! - **Void**: Represents the absence of a value (for functions that don't return)
//!
//! ## Value Representation
//!
//! Values in the IR can be:
//! - **Variables**: named values like `%result`
//! - **Constants**: Literal values like `42`, `true`, `"hello"`
//! - **Globals**: Global variables like `@message`
//!
//! ## FEAT:TODO - Missing Features Compared to LLVM IR
//!
//! ### Type System Gaps:
//! - **Function types**: First-class function types like `i32 (i32, i32)*`
//! - **Address spaces**: Memory spaces like `i32 addrspace(1)*` for GPUs
//! - **Packed structs**: `<{ i32, i8 }>` for compact memory layout
//! - **Union types**: Sum types for different representations
//! - **Opaque types**: Forward declarations like `type opaque`
//! - **Type aliases**: Named type abbreviations
//! - **Metadata types**: Types for debug and optimization info
//!
//! ### Missing / Planned Instructions:
//! - **Floating point**: additional ops beyond the existing arithmetic and comparisons
//! - **Atomic operations**: richer `atomicrmw`/`cmpxchg`/`fence` variants
//! - **Exception handling**: `invoke`, `landingpad`, `resume`
//! - **Control flow**: `indirectbr`
//! - **Vector operations**: `div`, `rem`, the rotates and the shifts have no `<N x T>` form
//!
//! ### Implemented (previously listed as missing):
//! - `switch` — full multi-way branch lowered to cmp/jmp chains in all backends
//! - Vector types, `<N x T>`, as an instruction's type suffix. See `Plan/simd-surface.md`
//!
//! ### Advanced Features:
//! - **Debug information**: `!dbg`, `!llvm.dbg.value`
//! - **Module features**: `comdat`, `section`, `alignment`
//! - **Optimization hints**: `!invariant.load`, `!dereferenceable`
//! - **Target intrinsics**: Architecture-specific operations
//! - **Profile-guided optimization**: Branch weights and profiling data
//!
//! ## IMPLEMENTED: Raw I/O Support
//!
//! Lamina provides raw I/O operations for direct system-level
//! input and output without stdlib dependencies:
//!
//! ### I/O Instruction Categories:
//!
//! #### **Byte-Level I/O**:
//! - **`writebyte`**: Write single byte (low 8 bits) to stdout; canonical for portable output
//! - **`readbyte`**: Read single byte from stdin
//! - **Use case**: Character I/O, text output, keyboard input
//!
//! #### **Buffer-Level I/O**:
//! - **`write`**: Write buffer contents to stdout
//! - **`read`**: Read data into buffer from stdin
//! - **Use case**: Bulk data transfer, file operations, network I/O
//!
//! ## Type Safety
//!
//! The type system provides:
//! - **Type checking**: Operations are only performed on compatible types
//! - **Memory safety**: Proper handling of pointers and memory access
//! - **Correctness**: Many classes of errors are caught at compile time
//!
//! ## Integer Semantics
//!
//! Integer addition, subtraction and multiplication wrap modulo `2^B`, where
//! `B` is the operand width, for both signed and unsigned values. Shift counts
//! are masked with `B - 1`. Values are concrete; arithmetic has no deferred errors.
//!
//! A trap stops execution at the offending instruction:
//! - Integer division or remainder by zero
//! - Signed division or remainder of the minimum value by `-1`
//! - Null-pointer dereference (`Load` / `Store` through a null `Ptr`)
//! - Out-of-bounds memory access (if bounds information is available)
//!
//! ## Examples
//!
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, StructField, BinaryOp, CmpOp};
//! use lamina_ir::builder::{var, i32, bool};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("type_demo", Type::Void)
//!     // Primitive types
//!     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
//!     // Array type
//!     .stackalloc("arr", Type::Array {
//!         element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
//!         size: 10
//!     })
//!     // Struct type
//!     .stackalloc("point", Type::Struct { fields: vec![
//!         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
//!         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) }
//!     ], packed: false })
//!     .ret_void();
//! ```

use lamina_platform::{PointerWidth, TargetDataLayout, VaListAbi, any_vector_register_bits};

use crate::module::TypeDeclaration;
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

/// A type alias for identifiers in the IR.
///
/// Uses `&'a str` to avoid allocating `String` objects
/// and memory efficiency. The lifetime parameter keeps identifiers
/// valid as long as the source text they were parsed from.
///
/// Identifiers are used for variable names, function names, and other symbols.
/// Using string references avoids unnecessary allocations and keeps the IR lightweight.
pub type Identifier<'a> = &'a str;
/// A type alias for basic block labels in the IR.
///
/// Labels are used to identify basic blocks for control flow operations
/// like branches and jumps.
pub type Label<'a> = &'a str;

/// Primitive types in the Lamina IR.
///
/// Primitive types are the fundamental building blocks of the type system.
/// They represent basic data types that can be directly operated on by
/// the target architecture.
///
/// # Integer Types
///
/// - **Signed integers**: `I8`, `I16`, `I32`, `I64`
/// - **Unsigned integers**: `U8`, `U16`, `U32`, `U64`
///
/// # Floating Point Types
///
/// - **Single precision**: `F32` (32-bit IEEE 754)
/// - **Double precision**: `F64` (64-bit IEEE 754)
///
/// # Other Types
///
/// - **Boolean**: `Bool` (true/false)
/// - **Pointer**: `Ptr` (generic pointer type)
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp, CmpOp};
/// use lamina_ir::builder::{var, i32, bool};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("primitives", Type::Void)
///     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
///     .cmp(CmpOp::Gt, "is_positive", PrimitiveType::I32, var("sum"), i32(0))
///     .ret_void();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    /// 8-bit signed integer (-128 to 127)
    I8,
    /// 16-bit signed integer (-32,768 to 32,767)
    I16,
    /// 32-bit signed integer (-2,147,483,648 to 2,147,483,647)
    I32,
    /// 64-bit signed integer (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
    I64,
    /// 8-bit unsigned integer (0 to 255)
    U8,
    /// 16-bit unsigned integer (0 to 65,535)
    U16,
    /// 32-bit unsigned integer (0 to 4,294,967,295)
    U32,
    /// 64-bit unsigned integer (0 to 18,446,744,073,709,551,615)
    U64,
    /// 128-bit signed integer, two's complement. Lowered to a pair of 64-bit words
    /// before any backend sees it. Arithmetic wraps; division traps like `I64`.
    I128,
    /// 128-bit unsigned integer. Lowered to a pair of 64-bit words before any backend
    /// sees it.
    U128,
    /// 32-bit floating point number (IEEE 754 single precision)
    F32,
    /// 64-bit floating point number (IEEE 754 double precision)
    F64,
    /// Boolean value (true or false)
    Bool,
    /// Generic pointer type for memory addresses
    Ptr,
}

/// One named field of a struct type. Fields keep their order, which fixes the layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField<'a> {
    pub name: Identifier<'a>,
    pub ty: Type<'a>,
}

/// Every type the IR can name. A `Named` type stands for a declaration elsewhere in
/// the module and is resolved with
/// [`resolve_type_names`](crate::types::resolve_type_names) before layout or lowering.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type<'a> {
    Primitive(PrimitiveType),
    Named(Identifier<'a>), // `@Vec2`
    Array {
        element_type: Box<Type<'a>>,
        size: u64,
    },
    /// A fixed-width machine vector, written `<N x T>`. It names a value held in one
    /// vector register and never has a memory layout of its own: it may appear as an
    /// instruction's type suffix and nowhere else. See `Plan/simd-surface.md`.
    Vector {
        element_type: PrimitiveType,
        lanes: u32,
    },
    /// A struct. Fields are laid out in order. With `packed` set there is no padding
    /// between fields and the struct's own alignment is 1, as in C `__attribute__((packed))`.
    Struct {
        fields: Vec<StructField<'a>>,
        packed: bool,
    },
    Void,
    /// A `va_list` cursor over variadic arguments, written `valist`. It is laid out as
    /// the x86_64 System V `va_list`: 24 bytes, 8-byte aligned.
    VaList,
}

impl From<PrimitiveType> for Type<'_> {
    fn from(ty: PrimitiveType) -> Self {
        Self::Primitive(ty)
    }
}

/// A constant written in the source. The variant is the literal's own type, which the
/// parser picks from the type it is used at, not from the digits.
#[derive(Debug, Clone, PartialEq)] // Float comparison requires PartialEq only
pub enum Literal<'a> {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I128(i128),
    U128(u128),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(&'a str),
}

/// An operand: a named value, a constant, or the address of a global.
#[derive(Debug, Clone, PartialEq)] // Removed Eq, Hash because f32 doesn't support them
pub enum Value<'a> {
    Variable(Identifier<'a>),
    Constant(Literal<'a>),
    Global(Identifier<'a>),
}

impl<'a> Literal<'a> {
    /// The value of an integer literal, or `None` for any other literal.
    pub fn as_integer(&self) -> Option<i128> {
        Some(match *self {
            Literal::I8(v) => v.into(),
            Literal::I16(v) => v.into(),
            Literal::I32(v) => v.into(),
            Literal::I64(v) => v.into(),
            Literal::U8(v) => v.into(),
            Literal::U16(v) => v.into(),
            Literal::U32(v) => v.into(),
            Literal::U64(v) => v.into(),
            Literal::I128(v) => v,
            Literal::U128(v) => return i128::try_from(v).ok(),
            Literal::F32(_) | Literal::F64(_) | Literal::Bool(_) | Literal::String(_) => {
                return None;
            }
        })
    }
}

/// What a value of a primitive type occupies, with signedness dropped.
///
/// Signedness belongs to the operation rather than to the value, so `u64` and `i64`
/// are one class: either may stand where the other is stated. A width is not a class
/// of its own, so `i32` and `i64` are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueClass {
    Integer(u32),
    Float(u32),
    Bool,
    Pointer,
}

impl PrimitiveType {
    /// The class of register a value of this type occupies.
    pub const fn value_class(self) -> ValueClass {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => ValueClass::Integer(8),
            PrimitiveType::I16 | PrimitiveType::U16 => ValueClass::Integer(16),
            PrimitiveType::I32 | PrimitiveType::U32 => ValueClass::Integer(32),
            PrimitiveType::I64 | PrimitiveType::U64 => ValueClass::Integer(64),
            PrimitiveType::I128 | PrimitiveType::U128 => ValueClass::Integer(128),
            PrimitiveType::F32 => ValueClass::Float(32),
            PrimitiveType::F64 => ValueClass::Float(64),
            PrimitiveType::Bool => ValueClass::Bool,
            PrimitiveType::Ptr => ValueClass::Pointer,
        }
    }

    /// The inclusive range this integer type holds, or `None` when it is not an integer.
    pub fn integer_bounds(self) -> Option<(i128, i128)> {
        Some(match self {
            PrimitiveType::I8 => (i8::MIN.into(), i8::MAX.into()),
            PrimitiveType::I16 => (i16::MIN.into(), i16::MAX.into()),
            PrimitiveType::I32 => (i32::MIN.into(), i32::MAX.into()),
            PrimitiveType::I64 => (i64::MIN.into(), i64::MAX.into()),
            PrimitiveType::U8 => (0, u8::MAX.into()),
            PrimitiveType::U16 => (0, u16::MAX.into()),
            PrimitiveType::U32 => (0, u32::MAX.into()),
            PrimitiveType::U64 => (0, u64::MAX.into()),
            PrimitiveType::I128 => (i128::MIN, i128::MAX),
            PrimitiveType::U128 => (0, i128::MAX),
            PrimitiveType::F32 | PrimitiveType::F64 | PrimitiveType::Bool | PrimitiveType::Ptr => {
                return None;
            }
        })
    }

    /// Returns all valid primitive type names as strings.
    pub fn all_names() -> &'static [&'static str] {
        const fn name_for(ty: PrimitiveType) -> &'static str {
            ty.as_str()
        }
        const NAMES: &[&str] = &[
            name_for(PrimitiveType::I8),
            name_for(PrimitiveType::I16),
            name_for(PrimitiveType::I32),
            name_for(PrimitiveType::I64),
            name_for(PrimitiveType::U8),
            name_for(PrimitiveType::U16),
            name_for(PrimitiveType::U32),
            name_for(PrimitiveType::U64),
            name_for(PrimitiveType::I128),
            name_for(PrimitiveType::U128),
            name_for(PrimitiveType::F32),
            name_for(PrimitiveType::F64),
            name_for(PrimitiveType::Bool),
            name_for(PrimitiveType::Ptr),
        ];
        NAMES
    }

    /// Returns the string representation of this primitive type.
    pub const fn as_str(&self) -> &'static str {
        match self {
            PrimitiveType::I8 => "i8",
            PrimitiveType::I16 => "i16",
            PrimitiveType::I32 => "i32",
            PrimitiveType::I64 => "i64",
            PrimitiveType::U8 => "u8",
            PrimitiveType::U16 => "u16",
            PrimitiveType::U32 => "u32",
            PrimitiveType::U64 => "u64",
            PrimitiveType::I128 => "i128",
            PrimitiveType::U128 => "u128",
            PrimitiveType::F32 => "f32",
            PrimitiveType::F64 => "f64",
            PrimitiveType::Bool => "bool",
            PrimitiveType::Ptr => "ptr",
        }
    }
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Primitive(pt) => write!(f, "{pt}"),
            Type::Named(id) => write!(f, "!{id}"),
            Type::Array { element_type, size } => write!(f, "[{size} x {element_type}]"),
            Type::Vector {
                element_type,
                lanes,
            } => write!(f, "<{lanes} x {element_type}>"),
            Type::Struct { fields, packed } => {
                if *packed {
                    write!(f, "packed ")?;
                }
                write!(f, "struct {{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    write!(f, "{}: {}", field.name, field.ty)?;
                    if i < fields.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " }}")
            }
            Type::Void => write!(f, "void"),
            Type::VaList => write!(f, "valist"),
        }
    }
}

impl fmt::Display for Literal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::I8(v) => write!(f, "{v}"),
            Literal::I16(v) => write!(f, "{v}"),
            Literal::I32(v) => write!(f, "{v}"),
            Literal::I64(v) => write!(f, "{v}"),
            Literal::U8(v) => write!(f, "{v}"),
            Literal::U16(v) => write!(f, "{v}"),
            Literal::U32(v) => write!(f, "{v}"),
            Literal::U64(v) => write!(f, "{v}"),
            Literal::I128(v) => write!(f, "{v}"),
            Literal::U128(v) => write!(f, "{v}"),
            Literal::F32(v) => write_float(f, v, v.is_finite()),
            Literal::F64(v) => write_float(f, v, v.is_finite()),
            Literal::Bool(v) => write!(f, "{v}"),
            Literal::String(s) => write!(f, "\"{s}\""), // Note: needs escaping for proper output
        }
    }
}

/// Writes a float so that reading the text back yields a float literal again: Rust
/// prints `1.0` as `1`, which reads as an integer.
fn write_float(f: &mut fmt::Formatter<'_>, value: &dyn fmt::Display, finite: bool) -> fmt::Result {
    let text = format!("{value}");
    if finite && !text.contains(['.', 'e', 'E']) {
        write!(f, "{text}.0")
    } else {
        write!(f, "{text}")
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Variable(id) => write!(f, "%{id}"),
            Value::Constant(lit) => write!(f, "{lit}"),
            Value::Global(id) => write!(f, "${id}"),
        }
    }
}

/// Returns the ABI alignment of a primitive type in bytes.
pub const fn align_of_primitive(ty: PrimitiveType, pointer: PointerWidth) -> u64 {
    match ty {
        PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool => 1,
        PrimitiveType::I16 | PrimitiveType::U16 => 2,
        PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
        PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
        PrimitiveType::Ptr => pointer.bytes() as u64,
        PrimitiveType::I128 | PrimitiveType::U128 => 16,
    }
}

/// Returns the size in bytes of a primitive type.
pub const fn sizeof_primitive(ty: PrimitiveType, pointer: PointerWidth) -> u64 {
    match ty {
        PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool => 1,
        PrimitiveType::I16 | PrimitiveType::U16 => 2,
        PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
        PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 8,
        PrimitiveType::Ptr => pointer.bytes() as u64,
        PrimitiveType::I128 | PrimitiveType::U128 => 16,
    }
}

/// The lane width in bits of a type that may be a vector lane, or `None` for one that
/// may not. `bool`, `ptr` and the 128-bit integers have no lane form.
pub const fn vector_lane_bits(element_type: PrimitiveType) -> Option<u32> {
    Some(match element_type {
        PrimitiveType::I8 | PrimitiveType::U8 => 8,
        PrimitiveType::I16 | PrimitiveType::U16 => 16,
        PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 32,
        PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => 64,
        PrimitiveType::Bool | PrimitiveType::Ptr | PrimitiveType::I128 | PrimitiveType::U128 => {
            return None;
        }
    })
}

/// Checks `<lanes x element_type>` against the lane types and register widths that
/// exist, returning the message to report when it names neither. `what` places the
/// type for the reader.
pub fn check_vector_shape(
    element_type: PrimitiveType,
    lanes: u32,
    what: &str,
) -> Result<(), String> {
    let Some(bits) = vector_lane_bits(element_type) else {
        return Err(format!(
            "{what} has element type {element_type}, which has no vector lane form\n  Hint: vector lanes are i8, i16, i32, i64, their unsigned forms, f32 and f64"
        ));
    };
    let widths = any_vector_register_bits();
    let total = lanes.checked_mul(bits).unwrap_or(0);
    if !widths.contains(&total) {
        let named = widths
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let smallest = widths.first().copied().unwrap_or(bits);
        return Err(format!(
            "<{lanes} x {element_type}> in {what} is {total} bits, which is no vector register width\n  Hint: a vector fills one register; widths are {named} bits, so write <{} x {element_type}> instead of splitting the value yourself",
            smallest / bits
        ));
    }
    Ok(())
}

/// Resolves named types throughout an aggregate, rejecting unknown names and cycles.
pub fn resolve_type_names<'a>(
    ty: &Type<'a>,
    declarations: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
) -> Option<Type<'a>> {
    fn resolve<'a>(
        ty: &Type<'a>,
        declarations: &HashMap<Identifier<'a>, TypeDeclaration<'a>>,
        visiting: &mut HashSet<Identifier<'a>>,
    ) -> Option<Type<'a>> {
        Some(match ty {
            Type::Named(name) => {
                if !visiting.insert(name) {
                    return None;
                }
                let resolved = resolve(&declarations.get(name)?.ty, declarations, visiting)?;
                visiting.remove(name);
                resolved
            }
            Type::Array { element_type, size } => Type::Array {
                element_type: Box::new(resolve(element_type, declarations, visiting)?),
                size: *size,
            },
            Type::Struct { fields, packed } => Type::Struct {
                fields: fields
                    .iter()
                    .map(|field| {
                        Some(StructField {
                            name: field.name,
                            ty: resolve(&field.ty, declarations, visiting)?,
                        })
                    })
                    .collect::<Option<_>>()?,
                packed: *packed,
            },
            other => other.clone(),
        })
    }
    resolve(ty, declarations, &mut HashSet::new())
}

fn round_up(value: u64, align: u64) -> Option<u64> {
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

fn aggregate_layout<'a>(
    fields: impl Iterator<Item = &'a Type<'a>>,
    packed: bool,
    layout: TargetDataLayout,
) -> Option<(u64, u64)> {
    let mut size = 0u64;
    let mut alignment = 1;
    for field in fields {
        let (field_size, field_alignment) = type_layout(field, layout)?;
        if !packed {
            alignment = alignment.max(field_alignment);
            size = round_up(size, field_alignment)?;
        }
        size = size.checked_add(field_size)?;
    }
    Some((round_up(size, alignment)?, alignment))
}

fn type_layout(ty: &Type<'_>, layout: TargetDataLayout) -> Option<(u64, u64)> {
    let pointer = layout.pointer_width;
    match ty {
        Type::Primitive(p) => {
            let alignment = match p {
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => layout.align_64,
                _ => align_of_primitive(*p, pointer),
            };
            Some((sizeof_primitive(*p, pointer), alignment))
        }
        Type::Array { element_type, size } => {
            let (element_size, alignment) = type_layout(element_type, layout)?;
            Some((element_size.checked_mul(*size)?, alignment))
        }
        Type::Struct { fields, packed } => {
            aggregate_layout(fields.iter().map(|field| &field.ty), *packed, layout)
        }
        Type::VaList => match layout.va_list {
            VaListAbi::Defined { size, alignment } => Some((size, alignment)),
            VaListAbi::Undefined => None,
        },
        _ => None,
    }
}

/// Returns the size in bytes, or `None` for unsized types or overflowing layouts.
pub fn sizeof_type(ty: &Type<'_>, layout: TargetDataLayout) -> Option<u64> {
    type_layout(ty, layout).map(|(size, _)| size)
}

/// Returns the ABI alignment in bytes of an IR type.
pub fn align_of_type(ty: &Type<'_>, layout: TargetDataLayout) -> u64 {
    type_layout(ty, layout).map_or(8, |(_, alignment)| alignment)
}

/// Computes the byte offset of `field_index` within a struct.
///
/// Returns `None` if `field_index` is out of bounds or a field size cannot be determined.
/// A normal struct aligns each field to its natural ABI alignment, inserting padding as
/// C `struct` layout does with default alignment. A packed struct places each field at
/// the running sum of the sizes before it.
///
/// # Example
/// For `struct { a: i8, b: i32, c: i64 }`:
/// - field 0 (`a`): offset 0
/// - field 1 (`b`): offset 4  (3 bytes of padding after `a`)
/// - field 2 (`c`): offset 8
///
/// For `packed struct { a: i8, b: i32, c: i64 }`:
/// - field 0 (`a`): offset 0
/// - field 1 (`b`): offset 1
/// - field 2 (`c`): offset 5
pub fn struct_field_byte_offset(
    fields: &[StructField<'_>],
    packed: bool,
    field_index: usize,
    layout: TargetDataLayout,
) -> Option<u64> {
    if field_index >= fields.len() {
        return None;
    }
    let mut offset = 0u64;
    for (index, field) in fields.iter().enumerate() {
        if !packed {
            offset = round_up(offset, align_of_type(&field.ty, layout))?;
        }
        if index == field_index {
            return Some(offset);
        }
        offset = offset.checked_add(sizeof_type(&field.ty, layout)?)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamina_platform::TargetArchitecture;

    fn l64() -> TargetDataLayout {
        TargetArchitecture::X86_64.data_layout()
    }

    fn l32() -> TargetDataLayout {
        TargetArchitecture::I686.data_layout()
    }

    #[test]
    fn valist_layout_is_a_property_of_the_target() {
        let expected = [
            (TargetArchitecture::X86_64, Some((24, 8))),
            (TargetArchitecture::Aarch64, Some((32, 8))),
            (TargetArchitecture::S390x, Some((32, 8))),
            (TargetArchitecture::Riscv64, Some((8, 8))),
            (TargetArchitecture::PowerPC64, Some((8, 8))),
            (TargetArchitecture::PowerPC64Be, Some((8, 8))),
            (TargetArchitecture::Mips64, Some((8, 8))),
            (TargetArchitecture::LoongArch64, Some((8, 8))),
            (TargetArchitecture::Wasm64, Some((8, 8))),
            (TargetArchitecture::I686, Some((4, 4))),
            (TargetArchitecture::Armv7, Some((4, 4))),
            (TargetArchitecture::Thumbv7m, Some((4, 4))),
            (TargetArchitecture::Riscv32, Some((4, 4))),
            (TargetArchitecture::Wasm32, Some((4, 4))),
            (TargetArchitecture::Nvptx64, None),
            (TargetArchitecture::Amdgcn, None),
        ];
        for (arch, want) in expected {
            let layout = arch.data_layout();
            assert_eq!(
                sizeof_type(&Type::VaList, layout)
                    .map(|size| (size, align_of_type(&Type::VaList, layout))),
                want,
                "{arch}"
            );
        }
    }

    #[test]
    fn valist_inside_a_struct_follows_the_target_too() {
        let holder = Type::Struct {
            fields: vec![
                StructField {
                    name: "flag",
                    ty: Type::Primitive(PrimitiveType::I8),
                },
                StructField {
                    name: "ap",
                    ty: Type::VaList,
                },
            ],
            packed: false,
        };
        assert_eq!(
            sizeof_type(&holder, TargetArchitecture::X86_64.data_layout()),
            Some(32)
        );
        assert_eq!(
            sizeof_type(&holder, TargetArchitecture::S390x.data_layout()),
            Some(40)
        );
        assert_eq!(
            sizeof_type(&holder, TargetArchitecture::Riscv64.data_layout()),
            Some(16)
        );
        assert_eq!(
            sizeof_type(&holder, TargetArchitecture::Nvptx64.data_layout()),
            None
        );
    }

    #[test]
    fn test_display_primitive_type() {
        assert_eq!(format!("{}", PrimitiveType::I8), "i8");
        assert_eq!(format!("{}", PrimitiveType::I16), "i16");
        assert_eq!(format!("{}", PrimitiveType::I32), "i32");
        assert_eq!(format!("{}", PrimitiveType::I64), "i64");
        assert_eq!(format!("{}", PrimitiveType::U8), "u8");
        assert_eq!(format!("{}", PrimitiveType::U16), "u16");
        assert_eq!(format!("{}", PrimitiveType::U32), "u32");
        assert_eq!(format!("{}", PrimitiveType::U64), "u64");
        assert_eq!(format!("{}", PrimitiveType::F32), "f32");
        assert_eq!(format!("{}", PrimitiveType::F64), "f64");
        assert_eq!(format!("{}", PrimitiveType::Bool), "bool");
        assert_eq!(format!("{}", PrimitiveType::Ptr), "ptr");
    }

    #[test]
    fn test_display_type() {
        assert_eq!(format!("{}", Type::Primitive(PrimitiveType::I64)), "i64");
        assert_eq!(format!("{}", Type::Named("MyStruct")), "!MyStruct");
        let arr_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
            size: 10,
        };
        assert_eq!(format!("{}", arr_type), "[10 x i8]");
        let fields = vec![
            StructField {
                name: "x",
                ty: Type::Primitive(PrimitiveType::F32),
            },
            StructField {
                name: "y",
                ty: Type::Primitive(PrimitiveType::F32),
            },
        ];
        let struct_type = Type::Struct {
            fields: fields.clone(),
            packed: false,
        };
        assert_eq!(format!("{}", struct_type), "struct { x: f32, y: f32 }");
        let packed_type = Type::Struct {
            fields,
            packed: true,
        };
        assert_eq!(
            format!("{}", packed_type),
            "packed struct { x: f32, y: f32 }"
        );
        assert_eq!(format!("{}", Type::Void), "void");
        {
            let vec_type = Type::Vector {
                element_type: PrimitiveType::I32,
                lanes: 4,
            };
            assert_eq!(format!("{}", vec_type), "<4 x i32>");
            let vec_f32 = Type::Vector {
                element_type: PrimitiveType::F32,
                lanes: 8,
            };
            assert_eq!(format!("{}", vec_f32), "<8 x f32>");
        }
    }

    #[test]
    fn pointer_fields_take_the_width_of_the_target() {
        assert_eq!(
            sizeof_primitive(PrimitiveType::Ptr, PointerWidth::Bits32),
            4
        );
        assert_eq!(
            align_of_primitive(PrimitiveType::Ptr, PointerWidth::Bits32),
            4
        );
        assert_eq!(
            sizeof_primitive(PrimitiveType::Ptr, PointerWidth::Bits64),
            8
        );
        let pointer_and_byte = Type::Struct {
            fields: vec![
                StructField {
                    name: "p",
                    ty: Type::Primitive(PrimitiveType::Ptr),
                },
                StructField {
                    name: "b",
                    ty: Type::Primitive(PrimitiveType::I8),
                },
            ],
            packed: false,
        };
        assert_eq!(sizeof_type(&pointer_and_byte, l32()), Some(8));
        assert_eq!(align_of_type(&pointer_and_byte, l32()), 4);
        assert_eq!(sizeof_type(&pointer_and_byte, l64()), Some(16));
        assert_eq!(
            struct_field_byte_offset(
                match &pointer_and_byte {
                    Type::Struct { fields, .. } => fields,
                    _ => unreachable!(),
                },
                false,
                1,
                l32(),
            ),
            Some(4)
        );
    }

    #[test]
    fn test_packed_struct_offsets_and_size() {
        let byte_word = [
            StructField {
                name: "a",
                ty: Type::Primitive(PrimitiveType::I8),
            },
            StructField {
                name: "b",
                ty: Type::Primitive(PrimitiveType::I32),
            },
        ];
        assert_eq!(
            struct_field_byte_offset(&byte_word, false, 1, l64()),
            Some(4)
        );
        assert_eq!(
            struct_field_byte_offset(&byte_word, true, 1, l64()),
            Some(1)
        );
        let packed = Type::Struct {
            fields: byte_word.to_vec(),
            packed: true,
        };
        assert_eq!(sizeof_type(&packed, l64()), Some(5));
        assert_eq!(align_of_type(&packed, l64()), 1);
        let padded = Type::Struct {
            fields: byte_word.to_vec(),
            packed: false,
        };
        assert_eq!(sizeof_type(&padded, l64()), Some(8));
        assert_eq!(align_of_type(&padded, l64()), 4);
    }

    #[test]
    fn test_packed_struct_odd_tail_and_nesting() {
        let tail = [
            StructField {
                name: "a",
                ty: Type::Primitive(PrimitiveType::I64),
            },
            StructField {
                name: "b",
                ty: Type::Primitive(PrimitiveType::I8),
            },
        ];
        let packed_tail = Type::Struct {
            fields: tail.to_vec(),
            packed: true,
        };
        assert_eq!(sizeof_type(&packed_tail, l64()), Some(9));
        let outer = [
            StructField {
                name: "c",
                ty: Type::Primitive(PrimitiveType::I8),
            },
            StructField {
                name: "tail",
                ty: packed_tail,
            },
        ];
        assert_eq!(struct_field_byte_offset(&outer, false, 1, l64()), Some(1));
        assert_eq!(
            sizeof_type(
                &Type::Struct {
                    fields: outer.to_vec(),
                    packed: false,
                },
                l64()
            ),
            Some(10)
        );
        assert_eq!(struct_field_byte_offset(&outer, false, 2, l64()), None);
    }

    #[test]
    fn test_display_literal() {
        assert_eq!(format!("{}", Literal::I8(123)), "123");
        assert_eq!(format!("{}", Literal::I16(-456)), "-456");
        assert_eq!(format!("{}", Literal::I32(123)), "123");
        assert_eq!(format!("{}", Literal::I64(-456)), "-456");
        assert_eq!(format!("{}", Literal::U8(123)), "123");
        assert_eq!(format!("{}", Literal::U16(456)), "456");
        assert_eq!(format!("{}", Literal::U32(123)), "123");
        assert_eq!(format!("{}", Literal::U64(456)), "456");
        assert_eq!(format!("{}", Literal::F32(1.25)), "1.25");
        assert_eq!(format!("{}", Literal::F64(2.5)), "2.5");
        assert_eq!(format!("{}", Literal::Bool(true)), "true");
        assert_eq!(format!("{}", Literal::Bool(false)), "false");
        assert_eq!(format!("{}", Literal::String("hello")), "\"hello\"");
    }

    #[test]
    fn test_display_value() {
        assert_eq!(format!("{}", Value::Variable("tmp1")), "%tmp1");
        assert_eq!(format!("{}", Value::Constant(Literal::I32(42))), "42");
        assert_eq!(
            format!("{}", Value::Constant(Literal::String("world"))),
            "\"world\""
        );
        assert_eq!(
            format!("{}", Value::Global("my_global_var")),
            "$my_global_var"
        );
    }

    #[test]
    fn eight_byte_scalars_take_the_target_alignment() {
        // gcc -m32 lays `struct { int a; long long b; int c; }` out as 0, 4, 12 in 16
        // bytes; every other target aligns b to 8.
        for wide in [PrimitiveType::I64, PrimitiveType::U64, PrimitiveType::F64] {
            let fields = vec![
                StructField {
                    name: "a",
                    ty: Type::Primitive(PrimitiveType::I32),
                },
                StructField {
                    name: "b",
                    ty: Type::Primitive(wide),
                },
                StructField {
                    name: "c",
                    ty: Type::Primitive(PrimitiveType::I32),
                },
            ];
            let offsets = |layout| {
                (0..3)
                    .map(|index| struct_field_byte_offset(&fields, false, index, layout))
                    .collect::<Vec<_>>()
            };
            let ty = Type::Struct {
                fields: fields.clone(),
                packed: false,
            };
            assert_eq!(offsets(l32()), [Some(0), Some(4), Some(12)]);
            assert_eq!(sizeof_type(&ty, l32()), Some(16));
            assert_eq!(align_of_type(&ty, l32()), 4);
            for arch in [TargetArchitecture::X86_64, TargetArchitecture::Armv7] {
                let layout = arch.data_layout();
                assert_eq!(offsets(layout), [Some(0), Some(8), Some(16)], "{arch:?}");
                assert_eq!(sizeof_type(&ty, layout), Some(24), "{arch:?}");
                assert_eq!(align_of_type(&ty, layout), 8, "{arch:?}");
            }
        }
    }
}

/// Which of the two vector shifts `shl`/`shr` names, decided by the count operand's
/// type: a scalar count shifts every lane by the same amount, a count vector of the
/// same shape shifts each lane by its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorShiftForm {
    Uniform,
    PerLane,
}

/// The form `rhs` names as the count of a `<lanes x element_type>` shift, or why it
/// names neither. The returned message is the reason alone; the caller says where.
pub fn classify_vector_shift<'a>(
    element_type: PrimitiveType,
    lanes: u32,
    rhs: &Value<'a>,
    env: &HashMap<Identifier<'a>, Type<'a>>,
) -> Result<VectorShiftForm, String> {
    if !is_integer_lane(element_type) {
        return Err(format!(
            "shifts {element_type} lanes\n  Hint: a shift is an integer operation; there is no vector shift of float lanes on any target here"
        ));
    }
    let hint = format!(
        "\n  Hint: a scalar count shifts every lane by the same amount; a <{lanes} x {element_type}> count shifts each lane by its own"
    );
    let name = match rhs {
        Value::Constant(literal) => {
            return if literal.as_integer().is_some() {
                Ok(VectorShiftForm::Uniform)
            } else {
                Err(format!("takes a {literal} as its shift count{hint}"))
            };
        }
        Value::Global(name) => {
            return Err(format!(
                "takes the address of the global ${name} as its shift count{hint}"
            ));
        }
        Value::Variable(name) => name,
    };
    match env.get(name) {
        Some(Type::Vector {
            element_type: count_element,
            lanes: count_lanes,
        }) => {
            if *count_lanes == lanes
                && is_integer_lane(*count_element)
                && vector_lane_bits(*count_element) == vector_lane_bits(element_type)
            {
                Ok(VectorShiftForm::PerLane)
            } else {
                Err(format!(
                    "shifts <{lanes} x {element_type}> by a <{count_lanes} x {count_element}> count{hint}"
                ))
            }
        }
        Some(Type::Primitive(count)) if is_integer_lane(*count) => Ok(VectorShiftForm::Uniform),
        Some(ty) => Err(format!("takes a {ty} as its shift count{hint}")),
        None => Err(format!(
            "takes %{name} as its shift count, whose type is not stated here{hint}"
        )),
    }
}

fn is_integer_lane(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::U8
            | PrimitiveType::U16
            | PrimitiveType::U32
            | PrimitiveType::U64
    )
}
