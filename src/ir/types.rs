//! # Type System
//!
//! This module defines the complete type system for the Lamina IR, including
//! primitive types, composite types, and value representations.
//!
//! ## Type Categories
//!
//! ### Primitive Types
//! - **Integers**: Signed and unsigned integers of various sizes
//! - **Floats**: Single and double precision floating point numbers
//! - **Booleans**: True/false values
//! - **Characters**: Single character values
//! - **Pointers**: Generic pointer type for memory addresses
//!
//! ### Composite Types
//! - **Arrays**: Fixed-size sequences of elements of the same type
//! - **Structs**: Named collections of fields with different types
//! - **Tuples**: Anonymous collections of values with different types
//! - **Named Types**: User-defined type aliases
//!
//! ### Special Types
//! - **Void**: Represents the absence of a value (for functions that don't return)
//!
//! ## Value Representation
//!
//! Values in the IR can be:
//! - **Variables**: SSA variables like `%result`
//! - **Constants**: Literal values like `42`, `true`, `"hello"`
//! - **Globals**: Global variables like `@message`
//!
//! ## FEAT:TODO - Missing Features Compared to LLVM IR
//!
//! ### Type System Gaps:
//! - **Vector/SIMD types**: `<4 x i32>` for SIMD operations
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
//! - **Memory intrinsics**: `memcpy`, `memset`, `memmove`
//! - **Atomic operations**: richer `atomicrmw`/`cmpxchg`/`fence` variants
//! - **Exception handling**: `invoke`, `landingpad`, `resume`
//! - **Control flow**: `switch`, `indirectbr`
//! - **Vector operations**: additional SIMD arithmetic and operations (kept behind `nightly`)
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
//! Lamina provides comprehensive raw I/O operations for direct system-level
//! input and output without stdlib dependencies:
//!
//! ### I/O Instruction Categories:
//!
//! #### **Byte-Level I/O**:
//! - **`writebyte`**: Write single ASCII character to stdout
//! - **`readbyte`**: Read single byte from stdin
//! - **Use case**: Character I/O, simple text output, keyboard input
//!
//! #### **Buffer-Level I/O**:
//! - **`write`**: Write buffer contents to stdout
//! - **`read`**: Read data into buffer from stdin
//! - **Use case**: Bulk data transfer, file operations, network I/O
//!
//! #### **Memory-Value I/O** (NEW):
//! - **`writeptr`**: Write VALUE stored at pointer to stdout
//!   //! - **Use case**: Memory inspection, binary data output, serialization
//!
//! ### WritePtr Semantics (IMPORTANT):
//!
//! **What it does:**
//! ```text
//! Memory: [42] <- pointer points here
//! writeptr(pointer) -> outputs: 42 (binary)
//! ```
//!
//! **NOT the pointer address, but the VALUE stored at that address!**
//!
//! **Example:**
//! ```lamina
//! %buffer = alloc.stack i32      # Allocate memory
//! store.i32 %buffer, 72          # Store 72 ('H' in ASCII)
//! %result = writeptr %buffer     # Outputs: 'H' (binary 72)
//! ```
//!
//! **Common misconception:**
//! - `writeptr(ptr)` does NOT write the pointer address
//! - `writeptr(ptr)` writes the VALUE stored at that address
//! - Think of it as "write pointer's value", not "write pointer"
//!
//! ### I/O Operation Comparison:
//!
//! | Instruction | Input/Output | Data Format | Use Case |
//! |-------------|--------------|-------------|----------|
//! | `writebyte` | Single char | ASCII | Simple text |
//! | `write` | Buffer | Raw bytes | Bulk data |
//! | `writeptr` | Memory value | Binary | Memory inspection |
//!
//! ### Memory Workflow Example:
//!
//! ```lamina
//! # Step 1: Allocate memory
//! %data = alloc.stack i32
//!
//! # Step 2: Store a value
//! store.i32 %data, 65  # ASCII 'A'
//!
//! # Step 3: Load the value (optional, for verification)
//! %value = load.i32 %data
//!
//! # Step 4: Output the stored value
//! %bytes = writeptr %data  # Outputs: 'A'
//! ```
//!
//! This workflow demonstrates the complete memory manipulation cycle
//! that forms the foundation of all data processing in Lamina.
//!
//! ## Type Safety
//!
//! The type system ensures:
//! - **Type checking**: Operations are only performed on compatible types
//! - **Memory safety**: Proper handling of pointers and memory access
//! - **Correctness**: Many classes of errors are caught at compile time
//!
//! ## Examples
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, StructField, BinaryOp, CmpOp};
//! use lamina::ir::builder::{var, i32, bool};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("type_demo", Type::Void)
//!     // Primitive types
//!     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
//!     // Array type
//!     .alloc_stack("arr", Type::Array {
//!         element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
//!         size: 10
//!     })
//!     // Struct type
//!     .alloc_stack("point", Type::Struct(vec![
//!         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
//!         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) }
//!     ]))
//!     .ret_void();
//! ```

use std::fmt;

/// A type alias for identifiers in the IR.
///
/// Uses `&'a str` to avoid allocating `String` objects, improving performance
/// and memory efficiency. The lifetime parameter ensures that identifiers remain
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
/// - **Character**: `Char` (8-bit character)
/// - **Pointer**: `Ptr` (generic pointer type)
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp, CmpOp};
/// use lamina::ir::builder::{var, i32, bool};
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
    /// 32-bit floating point number (IEEE 754 single precision)
    F32,
    /// 64-bit floating point number (IEEE 754 double precision)
    F64,
    /// Boolean value (true or false)
    Bool,
    /// Single character (8-bit)
    Char, // Single character (8-bit)
    /// Generic pointer type for memory addresses
    Ptr, // Generic pointer type
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField<'a> {
    pub name: Identifier<'a>,
    pub ty: Type<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type<'a> {
    Primitive(PrimitiveType),
    Named(Identifier<'a>), // Like "@Vec2"
    // Using Box to avoid recursive type definition issues
    Array {
        element_type: Box<Type<'a>>,
        size: u64,
    },
    #[cfg(feature = "nightly")]
    /// SIMD vector type: `<N x T>` where N is the number of lanes and T is the element type
    /// Examples: `<4 x i32>`, `<8 x f32>`, `<2 x i64>`
    Vector {
        element_type: PrimitiveType,
        lanes: u32,
    },
    Struct(Vec<StructField<'a>>),
    Tuple(Vec<Type<'a>>),
    Void, // For functions that don't return a value
}

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
    F32(f32),
    F64(f64),
    Bool(bool),
    Char(char),
    String(&'a str),
    // Null pointers or other constants might be needed later
}

#[derive(Debug, Clone, PartialEq)] // Removed Eq, Hash because f32 doesn't support them
pub enum Value<'a> {
    // SSA register/variable, like "%result"
    Variable(Identifier<'a>),
    // Literal values used directly in instructions
    Constant(Literal<'a>),
    // Reference to a global variable, like "@message"
    Global(Identifier<'a>),
}

// --- Display implementations for better readability ---

impl PrimitiveType {
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
            name_for(PrimitiveType::F32),
            name_for(PrimitiveType::F64),
            name_for(PrimitiveType::Bool),
            name_for(PrimitiveType::Char),
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
            PrimitiveType::F32 => "f32",
            PrimitiveType::F64 => "f64",
            PrimitiveType::Bool => "bool",
            PrimitiveType::Char => "char",
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
            Type::Primitive(pt) => write!(f, "{}", pt),
            Type::Named(id) => write!(f, "@{}", id),
            Type::Array { element_type, size } => write!(f, "[{} x {}]", size, element_type),
            #[cfg(feature = "nightly")]
            Type::Vector {
                element_type,
                lanes,
            } => write!(f, "<{} x {}>", lanes, element_type),
            Type::Struct(fields) => {
                write!(f, "struct {{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    write!(f, "{}: {}", field.name, field.ty)?;
                    if i < fields.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " }}")
            }
            Type::Tuple(types) => {
                write!(f, "tuple(")?;
                for (i, ty) in types.iter().enumerate() {
                    write!(f, "{}", ty)?;
                    if i < types.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
            Type::Void => write!(f, "void"),
        }
    }
}

impl fmt::Display for Literal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::I8(v) => write!(f, "{}", v),
            Literal::I16(v) => write!(f, "{}", v),
            Literal::I32(v) => write!(f, "{}", v),
            Literal::I64(v) => write!(f, "{}", v),
            Literal::U8(v) => write!(f, "{}", v),
            Literal::U16(v) => write!(f, "{}", v),
            Literal::U32(v) => write!(f, "{}", v),
            Literal::U64(v) => write!(f, "{}", v),
            Literal::F32(v) => write!(f, "{}", v),
            Literal::F64(v) => write!(f, "{}", v),
            Literal::Bool(v) => write!(f, "{}", v),
            Literal::Char(c) => write!(f, "'{}'", c),
            Literal::String(s) => write!(f, "\"{}\"", s), // Note: needs escaping for proper output
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Variable(id) => write!(f, "%{id}"),
            Value::Constant(lit) => write!(f, "{}", lit),
            Value::Global(id) => write!(f, "@{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(format!("{}", PrimitiveType::Char), "char");
        assert_eq!(format!("{}", PrimitiveType::Ptr), "ptr");
    }

    #[test]
    fn test_display_type() {
        assert_eq!(format!("{}", Type::Primitive(PrimitiveType::I64)), "i64");
        assert_eq!(format!("{}", Type::Named("MyStruct")), "@MyStruct");
        let arr_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
            size: 10,
        };
        assert_eq!(format!("{}", arr_type), "[10 x i8]");
        let struct_type = Type::Struct(vec![
            StructField {
                name: "x",
                ty: Type::Primitive(PrimitiveType::F32),
            },
            StructField {
                name: "y",
                ty: Type::Primitive(PrimitiveType::F32),
            },
        ]);
        assert_eq!(format!("{}", struct_type), "struct { x: f32, y: f32 }");
        let tuple_type = Type::Tuple(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Named("MyBool"),
        ]);
        assert_eq!(format!("{}", tuple_type), "tuple(i32, @MyBool)");
        assert_eq!(format!("{}", Type::Void), "void");
        #[cfg(feature = "nightly")]
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
        assert_eq!(format!("{}", Literal::Char('A')), "'A'");
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
            "@my_global_var"
        );
    }
}
