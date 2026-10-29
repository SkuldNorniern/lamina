//! # Instruction Definitions
//!
//! This module defines all instruction types in the Lamina IR. Instructions are
//! the fundamental operations that can be performed on values and represent the
//! atomic units of computation.
//!
//! ## Instruction Categories
//!
//! ### Arithmetic and Logic
//! - **Binary operations**: Addition, subtraction, multiplication, division
//! - **Comparison operations**: Equality, ordering, logical comparisons
//! - **Type conversions**: Zero-extension, type casting
//!
//! ### Memory Operations
//! - **Allocation**: Stack memory allocation
//! - **Access**: Loading and storing values to/from memory
//! - **Pointer arithmetic**: Array indexing, struct field access
//!
//! ### Control Flow
//! - **Branches**: Conditional jumps based on boolean values
//! - **Jumps**: Unconditional jumps to other basic blocks
//! - **Function calls**: Invoking other functions
//! - **Returns**: Exiting functions with optional return values
//!
//! ### Debug and I/O
//! - **Phi nodes**: optional value merging from different control paths
//!
//! ## Values
//!
//! Values are named and typed. A name may be assigned again, but not with a
//! different type. Control flow is explicit through blocks; `phi` merges values
//! from incoming edges and is optional.
//!
//! ## Example
//!
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina_ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("example", Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
//!     .binary(BinaryOp::Mul, "result", PrimitiveType::I32, var("sum"), i32(2))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//! ```

use std::fmt;

use crate::types::{Identifier, Label, Literal, PrimitiveType, Type, Value};

/// Memory ordering constraints for atomic operations.
///
/// These specify the ordering guarantees for atomic operations relative to
/// other memory operations in the same thread and across threads.
///
/// # Memory Ordering Levels
///
/// - **Relaxed**: No ordering constraints with other operations
/// - **Acquire**: Subsequent loads cannot move before this operation
/// - **Release**: Previous stores cannot move after this operation
/// - **AcqRel**: Both acquire and release semantics
/// - **SeqCst**: Sequentially consistent (strongest ordering)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrdering {
    /// Relaxed ordering - no synchronization or ordering constraints
    Relaxed,
    /// Acquire ordering - prevents reordering of subsequent loads
    Acquire,
    /// Release ordering - prevents reordering of previous stores
    Release,
    /// Acquire-release ordering - both acquire and release semantics
    AcqRel,
    /// Sequentially consistent - strongest ordering guarantee
    SeqCst,
}

impl fmt::Display for MemoryOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryOrdering::Relaxed => write!(f, "relaxed"),
            MemoryOrdering::Acquire => write!(f, "acquire"),
            MemoryOrdering::Release => write!(f, "release"),
            MemoryOrdering::AcqRel => write!(f, "acqrel"),
            MemoryOrdering::SeqCst => write!(f, "seqcst"),
        }
    }
}

/// SIMD (Single Instruction, Multiple Data) vector operations.
///
/// These operations perform parallel computations on multiple data elements
/// simultaneously, providing significant performance improvements for
/// data-parallel algorithms.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    /// Vector addition: `result = lhs + rhs` (element-wise)
    Add,
    /// Vector subtraction: `result = lhs - rhs` (element-wise)
    Sub,
    /// Vector multiplication: `result = lhs * rhs` (element-wise)
    Mul,
    /// Vector division: `result = lhs / rhs` (element-wise)
    Div,
    /// Vector minimum: `result = min(lhs, rhs)` (element-wise)
    Min,
    /// Vector maximum: `result = max(lhs, rhs)` (element-wise)
    Max,
    /// Vector absolute value: `result = abs(value)`
    Abs,
    /// Vector negation: `result = -value`
    Neg,
    /// Vector square root: `result = sqrt(value)`
    Sqrt,
    /// Vector fused multiply-add: `result = (lhs * rhs) + acc`
    Fma,
    /// Vector shuffle/rearrange elements according to mask
    Shuffle,
    /// Vector extract lane: extract single element from vector
    ExtractLane,
    /// Vector insert lane: insert single element into vector
    InsertLane,
    /// Vector splat: broadcast scalar to all vector lanes
    Splat,
}

#[cfg(feature = "nightly")]
impl fmt::Display for SimdOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdOp::Add => write!(f, "simd_add"),
            SimdOp::Sub => write!(f, "simd_sub"),
            SimdOp::Mul => write!(f, "simd_mul"),
            SimdOp::Div => write!(f, "simd_div"),
            SimdOp::Min => write!(f, "simd_min"),
            SimdOp::Max => write!(f, "simd_max"),
            SimdOp::Abs => write!(f, "simd_abs"),
            SimdOp::Neg => write!(f, "simd_neg"),
            SimdOp::Sqrt => write!(f, "simd_sqrt"),
            SimdOp::Fma => write!(f, "simd_fma"),
            SimdOp::Shuffle => write!(f, "simd_shuffle"),
            SimdOp::ExtractLane => write!(f, "simd_extract_lane"),
            SimdOp::InsertLane => write!(f, "simd_insert_lane"),
            SimdOp::Splat => write!(f, "simd_splat"),
        }
    }
}

/// Atomic binary operations for concurrent programming.
///
/// These operations perform read-modify-write sequences atomically,
/// for thread-safe updates to shared memory locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicBinOp {
    /// Atomic addition: `*ptr += value`
    Add,
    /// Atomic subtraction: `*ptr -= value`
    Sub,
    /// Atomic bitwise AND: `*ptr &= value`
    And,
    /// Atomic bitwise OR: `*ptr |= value`
    Or,
    /// Atomic bitwise XOR: `*ptr ^= value`
    Xor,
    /// Atomic exchange: `result = *ptr; *ptr = value`
    Exchange,
    /// Atomic compare-exchange: `if *ptr == expected { *ptr = value; return true } else { return false }`
    CompareExchange,
}

impl fmt::Display for AtomicBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicBinOp::Add => write!(f, "atomic_add"),
            AtomicBinOp::Sub => write!(f, "atomic_sub"),
            AtomicBinOp::And => write!(f, "atomic_and"),
            AtomicBinOp::Or => write!(f, "atomic_or"),
            AtomicBinOp::Xor => write!(f, "atomic_xor"),
            AtomicBinOp::Exchange => write!(f, "atomic_exchange"),
            AtomicBinOp::CompareExchange => write!(f, "atomic_cmpxchg"),
        }
    }
}

/// Binary arithmetic and bitwise operations.
///
/// These operations take two operands and produce a single result.
/// All operands and the result must have the same primitive type.
///
/// Arithmetic operations are defined for integer and floating types,
/// while bitwise operations are defined only for integer-like types
/// (signed/unsigned integers, booleans, and pointers).
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina_ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("math", Type::Primitive(PrimitiveType::I32))
///     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
///     .binary(BinaryOp::And, "masked", PrimitiveType::I32, var("sum"), i32(0xff))
///     .ret(Type::Primitive(PrimitiveType::I32), var("masked"));
/// ```
#[derive(Debug, Clone, PartialEq)] // Cannot derive Eq due to f32 in Value::Constant
pub enum BinaryOp {
    /// Addition: `result = lhs + rhs`. Integer results wrap.
    Add,
    /// Subtraction: `result = lhs - rhs`. Integer results wrap.
    Sub,
    /// Multiplication: `result = lhs * rhs`. Integer results wrap.
    Mul,
    /// Division: `result = lhs / rhs`. Integers trap on zero and signed `MIN / -1`.
    Div,
    /// Remainder: `result = lhs % rhs`. Integers trap on zero and signed `MIN % -1`.
    Rem,
    /// Bitwise AND: `result = lhs & rhs`
    ///
    /// Only valid for integer-like types (signed/unsigned integers, booleans, and pointers).
    And,
    /// Bitwise OR: `result = lhs | rhs`
    ///
    /// Only valid for integer-like types (signed/unsigned integers, booleans, and pointers).
    Or,
    /// Bitwise XOR: `result = lhs ^ rhs`
    ///
    /// Only valid for integer-like types (signed/unsigned integers, booleans, and pointers).
    Xor,
    /// Logical left shift: `result = lhs << rhs`
    ///
    /// Only valid for integer-like types. The count is masked with `bit_width(lhs) - 1`.
    Shl,
    /// Arithmetic right shift: `result = lhs >> rhs`
    ///
    /// Interprets `lhs` as a two's-complement signed integer and propagates the sign bit.
    /// Only valid for signed integer types. The count is masked with `bit_width(lhs) - 1`.
    ///
    /// For unsigned integers, consider using logical right shift in MIR (`IntBinOp::LShr`)
    /// or an explicit mask after an arithmetic shift.
    Shr,
    /// Rotate left: the bits shifted out at the top re-enter at the bottom.
    ///
    /// Only valid for integer types. The count is taken modulo `bit_width(lhs)`, so a
    /// count of zero (or a multiple of the width) returns `lhs` unchanged.
    Rotl,
    /// Rotate right: the bits shifted out at the bottom re-enter at the top.
    ///
    /// Only valid for integer types. The count is taken modulo `bit_width(lhs)`, so a
    /// count of zero (or a multiple of the width) returns `lhs` unchanged.
    Rotr,
}

impl BinaryOp {
    /// Returns all valid binary operation names as strings.
    pub fn all_names() -> &'static [&'static str] {
        const fn name_for(op: BinaryOp) -> &'static str {
            op.as_str()
        }
        const NAMES: &[&str] = &[
            name_for(BinaryOp::Add),
            name_for(BinaryOp::Sub),
            name_for(BinaryOp::Mul),
            name_for(BinaryOp::Div),
            name_for(BinaryOp::Rem),
            name_for(BinaryOp::And),
            name_for(BinaryOp::Or),
            name_for(BinaryOp::Xor),
            name_for(BinaryOp::Shl),
            name_for(BinaryOp::Shr),
            name_for(BinaryOp::Rotl),
            name_for(BinaryOp::Rotr),
        ];
        NAMES
    }

    /// Returns the string representation of this binary operation.
    pub const fn as_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Rem => "rem",
            BinaryOp::And => "and",
            BinaryOp::Or => "or",
            BinaryOp::Xor => "xor",
            BinaryOp::Shl => "shl",
            BinaryOp::Shr => "shr",
            BinaryOp::Rotl => "rotl",
            BinaryOp::Rotr => "rotr",
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Operations on a single value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation: `result = -value`. Integer results wrap, so negating
    /// the minimum of a signed type yields that minimum again. On a float this
    /// flips the sign bit, including on NaN and on zero.
    Neg,
    /// Bitwise complement: `result = !value`. Integer types only.
    Not,
    /// Square root, rounded to nearest with ties to even. Float types only.
    /// A negative input other than `-0.0` yields NaN.
    Sqrt,
    /// Population count: `result` is the number of set bits in `value`.
    ///
    /// Integer types only. Only the bits within the width of `ty` are counted, so
    /// `popcount.i8 -1` yields 8. The result has type `ty` and lies in `0..=bit_width`.
    Popcount,
    /// Count leading zeros: the number of zero bits above the most significant set bit,
    /// counted within the width of `ty`.
    ///
    /// Integer types only. `ctlz` of zero is the bit width of `ty`, not an undefined value.
    Ctlz,
    /// Count trailing zeros: the number of zero bits below the least significant set bit,
    /// counted within the width of `ty`.
    ///
    /// Integer types only. `cttz` of zero is the bit width of `ty`, not an undefined value.
    Cttz,
    /// Byte swap: reverses the order of the bytes of `value`.
    ///
    /// Integer types only. `bswap.i8` and `bswap.u8` return the value unchanged.
    Bswap,
}

impl UnaryOp {
    pub fn all_names() -> &'static [&'static str] {
        &["neg", "not", "sqrt", "popcount", "ctlz", "cttz", "bswap"]
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Not => "not",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Popcount => "popcount",
            UnaryOp::Ctlz => "ctlz",
            UnaryOp::Cttz => "cttz",
            UnaryOp::Bswap => "bswap",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Comparison operations that produce boolean results.
///
/// These operations compare two values and return a boolean indicating
/// whether the comparison is true or false.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, CmpOp};
/// use lamina_ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("compare", Type::Primitive(PrimitiveType::Bool))
///     .cmp(CmpOp::Gt, "is_greater", PrimitiveType::I32, var("a"), var("b"))
///     .ret(Type::Primitive(PrimitiveType::Bool), var("is_greater"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum CmpOp {
    /// Equality: `lhs == rhs`
    Eq,
    /// Inequality: `lhs != rhs`
    Ne,
    /// Greater than: `lhs > rhs`
    Gt,
    /// Greater than or equal: `lhs >= rhs`
    Ge,
    /// Less than: `lhs < rhs`
    Lt,
    /// Less than or equal: `lhs <= rhs`
    Le, // Equality, Non-equality, Greater than, etc.
}

impl CmpOp {
    /// Returns all valid comparison operation names as strings.
    pub fn all_names() -> &'static [&'static str] {
        const fn name_for(op: CmpOp) -> &'static str {
            op.as_str()
        }
        const NAMES: &[&str] = &[
            name_for(CmpOp::Eq),
            name_for(CmpOp::Ne),
            name_for(CmpOp::Gt),
            name_for(CmpOp::Ge),
            name_for(CmpOp::Lt),
            name_for(CmpOp::Le),
        ];
        NAMES
    }

    /// Returns the string representation of this comparison operation.
    pub const fn as_str(&self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Memory allocation strategies.
///
/// Determines where and how memory is allocated for variables.
/// This affects the lifetime and performance characteristics of the allocation.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{IRBuilder, Type, PrimitiveType};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("memory", Type::Void)
///     .stackalloc("local", Type::Primitive(PrimitiveType::I32));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocType {
    /// Stack allocation - automatic cleanup, fast, limited lifetime
    Stack,
}

/// A single instruction in a basic block.
///
/// # Precondition conventions
///
/// One operand of an inline assembly instruction. `$n` placeholders in the template name
/// operands by their position in the list.
#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand<'a> {
    /// `in(location) %value`: the value is placed in `location` before the template runs.
    In { location: &'a str, value: Value<'a> },
    /// `out(location)`: the template writes `location`. The instruction's result is read
    /// from it.
    Out { location: &'a str },
}

/// Operands must satisfy the documented types. Integer arithmetic wraps and
/// invalid integer division traps; see [`crate::types`] for the value contract.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction<'a> {
    /// Perform a binary arithmetic or bitwise operation.
    ///
    /// # Preconditions
    /// - `lhs` and `rhs` must have type `ty`.
    /// - Integer Add/Sub/Mul wrap modulo `2^bit_width(ty)`.
    /// - Shift counts are masked with `bit_width(ty) - 1`.
    /// - Integer Div/Rem trap on zero divisors and on signed `MIN / -1`.
    Binary {
        op: BinaryOp,
        result: Identifier<'a>,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    },
    /// Apply an operation to one value.
    ///
    /// # Preconditions
    /// - `value` must have type `ty`.
    /// - `Not` takes an integer type, `Sqrt` a float type, `Neg` either.
    Unary {
        op: UnaryOp,
        result: Identifier<'a>,
        ty: PrimitiveType,
        value: Value<'a>,
    },
    /// Compare two values and produce a boolean result.
    ///
    /// # Preconditions
    /// - `lhs` and `rhs` must have type `ty`.
    /// - Ordered floating-point comparisons on NaN produce `false`.
    Cmp {
        op: CmpOp,
        result: Identifier<'a>,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    },
    /// Zero-extend an unsigned integer or bool to a larger integer type.
    ///
    /// Written `extend.<unsigned-or-bool>.<target>` in source. A bool source maps
    /// false and true to 0 and 1.
    ///
    /// # Preconditions
    /// - `value` must have type `source_type`.
    /// - `source_type` must be an unsigned integer or bool, strictly smaller than
    ///   `target_type` in bit-width.
    ZeroExtend {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Select between two values based on a boolean condition.
    ///
    /// This is the value form of a conditional expression:
    /// `result = cond ? true_val : false_val`.
    /// It avoids introducing extra basic blocks compared to `br` + `phi`.
    Select {
        result: Identifier<'a>,
        ty: Type<'a>,
        cond: Value<'a>,      // Must be a boolean value
        true_val: Value<'a>,  // Value when condition is true
        false_val: Value<'a>, // Value when condition is false
    },
    /// Truncate an integer to a smaller integer type by discarding high bits.
    ///
    /// The source and target types must be integer-like, and the target must be
    /// strictly smaller than the source in bit-width.
    Trunc {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Sign-extend a signed integer value to a larger integer type.
    ///
    /// Written `extend.<signed>.<target>` in source. The source and target types must
    /// be integers, the source signed, and the target strictly larger in bit-width.
    SignExtend {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Reinterpret the bit pattern of a value as another primitive type of equal size.
    ///
    /// This does not change any bits, only the type used for subsequent operations.
    /// For example, `bitcast.f32.i32` reinterprets a 32-bit float as a 32-bit integer.
    Bitcast {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Convert an integer to a floating-point type.
    ///
    /// Written `inttofp.<source>.<target>`. `signed` follows `source_type`: an
    /// `i8`, `i16`, `i32` or `i64` is read as two's complement, a `u8`, `u16`,
    /// `u32` or `u64` as unsigned. `target_type` is `f32` or `f64`.
    ///
    /// Rounding is IEEE 754 round-to-nearest, ties-to-even. Values that are
    /// representable are converted exactly. Conversion never traps and never
    /// produces NaN or infinity, because every integer of at most 64 bits is
    /// finite and below the overflow threshold of `f32`.
    IntToFloat {
        result: Identifier<'a>,
        signed: bool,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Convert a floating-point value to an integer, truncating toward zero.
    ///
    /// Written `fptoint.<source>.<target>`. `signed` follows `target_type`: an
    /// `i8`, `i16`, `i32` or `i64` target is signed, a `u8`, `u16`, `u32` or
    /// `u64` target unsigned. `source_type` is `f32` or `f64`.
    ///
    /// This operation saturates and never traps. It does not depend on the
    /// hardware instruction used by a backend:
    /// - NaN produces `0`.
    /// - A value whose truncation lies above the largest representable target
    ///   value produces that maximum, including `+inf`.
    /// - For signed targets, a value whose truncation lies below the smallest
    ///   representable target value produces the minimum, including `-inf`.
    /// - For unsigned targets, a value whose truncation is negative produces `0`,
    ///   including `-inf`. Values in `(-1, 0)` truncate to `0`.
    /// - Values in range truncate toward zero: `fptoint.f64.i8` of `-1.9` is `-1`.
    FloatToInt {
        result: Identifier<'a>,
        signed: bool,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Convert between the floating-point types.
    ///
    /// Written `fpcast.<source>.<target>`. `f32` to `f64` widens exactly. `f64`
    /// to `f32` narrows, rounding to nearest with ties to even; a magnitude above the largest `f32`
    /// becomes an infinity of the same sign, and one below the smallest subnormal
    /// becomes a zero of the same sign. NaN stays NaN.
    FloatConvert {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    /// Address of a function, as a pointer value.
    FuncAddr {
        result: Identifier<'a>,
        func_name: Identifier<'a>,
    },
    /// Call through a function pointer.
    ///
    /// Each argument carries the type the call site declares for it, written like a
    /// function signature: `callptr.i64 %f(f32 1.5, i64 %n)`. That type is what a
    /// literal operand is materialised as and what the ABI classifies the argument by.
    ///
    /// A `...` in the argument list marks where the callee's named parameters end:
    /// `callptr.i32 %f(ptr %fmt, ..., f64 %x)` passes `%x` under the variadic rules.
    ///
    /// # Preconditions
    /// - `target` must hold the address of a function whose signature matches the
    ///   argument types given and `return_type`.
    CallIndirect {
        result: Option<Identifier<'a>>,
        target: Value<'a>,
        args: Vec<(Type<'a>, Value<'a>)>,
        /// Number of arguments the callee names, when `...` marks the rest as
        /// variadic. `None` when the call is not variadic.
        variadic_fixed: Option<usize>,
        return_type: Type<'a>,
    },
    // --- Control Flow ---
    /// Multi-way branch based on an integer or boolean value.
    ///
    /// This is the structured form of a jump table. The `value` is compared
    /// against each case literal; on the first match, control transfers to the
    /// corresponding label. If no case matches, control transfers to `default`.
    Switch {
        ty: PrimitiveType,
        value: Value<'a>,
        /// Default target label when no case matches.
        default: Label<'a>,
        /// Case list as pairs of (literal, target_label).
        cases: Vec<(Literal<'a>, Label<'a>)>,
    },
    /// Conditional branch to one of two target blocks.
    ///
    /// # Preconditions
    /// - `condition` must have type `bool`.
    /// - Both `true_label` and `false_label` must name blocks in the same
    ///   function.
    Br {
        condition: Value<'a>,
        true_label: Label<'a>,
        false_label: Label<'a>,
    },
    /// Unconditional jump to `target_label`.
    ///
    /// # Preconditions
    /// - `target_label` must name a block in the same function.
    Jmp { target_label: Label<'a> },
    /// Return from the current function.
    ///
    /// # Preconditions
    /// - If `ty` is `Type::Void`, `value` must be `None`.
    /// - If `ty` is non-void, `value` must be `Some(_)` with a value of type `ty`.
    /// - The return type must match the enclosing function's declared return type.
    Ret {
        ty: Type<'a>,
        value: Option<Value<'a>>,
    },
    /// Marks a point that control flow must never reach. Terminates a basic block.
    ///
    /// # Preconditions
    /// - Execution must never reach this instruction. Reaching it is undefined
    ///   behavior; the x86_64 backend traps instead.
    Unreachable,
    // --- Memory Operations ---
    /// Allocate memory for a value of type `allocated_ty`.
    ///
    /// # Preconditions
    /// - The allocation is automatically freed when the function returns.
    ///   Memory must be initialized before reading.
    /// - The result is always of type `ptr` and is guaranteed non-null.
    Alloc {
        result: Identifier<'a>,
        alloc_type: AllocType,
        allocated_ty: Type<'a>,
        align: Option<u32>,
    },
    /// Load a value of type `ty` from the address in `ptr`.
    ///
    /// # Preconditions
    /// - `ptr` must have type `ptr` and must point to memory of type `ty`.
    /// - `ptr` must not be null — a null pointer causes a **trap**.
    /// - The memory at `ptr` must be initialized.
    /// - The memory must be accessible (allocated and not freed).
    ///
    /// When `volatile` is set, the access must happen exactly once, must not be
    /// removed, and must not be reordered with other volatile accesses.
    Load {
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        volatile: bool,
    },
    /// Store a `value` of type `ty` to the address in `ptr`.
    ///
    /// # Preconditions
    /// - `ptr` must have type `ptr` and point to memory of type `ty`.
    /// - `ptr` must not be null — a null pointer causes a **trap**.
    /// - `value` must have type `ty`.
    ///
    /// When `volatile` is set, the access must happen exactly once, must not be
    /// removed, and must not be reordered with other volatile accesses.
    Store {
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        volatile: bool,
    },
    // --- Atomic Operations ---
    AtomicLoad {
        // Atomic load with memory ordering
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        ordering: MemoryOrdering,
    },
    AtomicStore {
        // Atomic store with memory ordering
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    },
    AtomicBinary {
        // Atomic read-modify-write operation
        op: AtomicBinOp,
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    },
    AtomicCompareExchange {
        // Atomic compare-exchange. `result` is the value that was in memory before
        // the operation; the exchange happened exactly when it equals `expected`.
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        expected: Value<'a>,
        desired: Value<'a>,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },
    Fence {
        // Memory fence/barrier
        ordering: MemoryOrdering,
    },
    /// Inline assembly, always treated as side-effecting.
    InlineAsm {
        result: Option<(Identifier<'a>, PrimitiveType)>,
        template: &'a str,
        operands: Vec<AsmOperand<'a>>,
        clobbers: Vec<&'a str>,
    },
    // --- SIMD Operations ---
    #[cfg(feature = "nightly")]
    SimdBinary {
        // SIMD binary operation: element-wise computation
        op: SimdOp,
        result: Identifier<'a>,
        vector_type: Type<'a>, // SIMD vector type (e.g., v4f32, v8i32)
        lhs: Value<'a>,
        rhs: Value<'a>,
    },
    #[cfg(feature = "nightly")]
    SimdUnary {
        // SIMD unary operation: single operand vector computation
        op: SimdOp,
        result: Identifier<'a>,
        vector_type: Type<'a>,
        operand: Value<'a>,
    },
    #[cfg(feature = "nightly")]
    SimdTernary {
        // SIMD ternary operation (e.g., fused multiply-add)
        op: SimdOp,
        result: Identifier<'a>,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        acc: Value<'a>, // accumulator/third operand
    },
    #[cfg(feature = "nightly")]
    SimdShuffle {
        // SIMD shuffle/rearrange elements
        result: Identifier<'a>,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        mask: Value<'a>, // shuffle mask (constant or runtime)
    },
    #[cfg(feature = "nightly")]
    SimdExtract {
        // Extract single element from SIMD vector
        result: Identifier<'a>,
        scalar_type: PrimitiveType, // Type of extracted element
        vector: Value<'a>,
        lane_index: Value<'a>, // Which lane to extract (0-based)
    },
    #[cfg(feature = "nightly")]
    SimdInsert {
        // Insert single element into SIMD vector
        result: Identifier<'a>,
        vector_type: Type<'a>,
        vector: Value<'a>,
        scalar: Value<'a>,     // Element to insert
        lane_index: Value<'a>, // Which lane to insert into (0-based)
    },
    #[cfg(feature = "nightly")]
    SimdLoad {
        // Load SIMD vector from memory
        result: Identifier<'a>,
        vector_type: Type<'a>,
        ptr: Value<'a>,
        alignment: Option<u32>, // Optional alignment hint
    },
    #[cfg(feature = "nightly")]
    SimdStore {
        // Store SIMD vector to memory
        vector_type: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        alignment: Option<u32>, // Optional alignment hint
    },
    // --- Composite Type Operations ---
    GetFieldPtr {
        // Get pointer to struct field
        result: Identifier<'a>, // Result is always ptr
        struct_ptr: Value<'a>,  // Must be a ptr to a struct or named struct type
        /// Layout the field offset is computed from: a struct, or a named type that
        /// resolves to one.
        struct_type: Type<'a>,
        field_index: usize,
    },
    GetElemPtr {
        // Get pointer to array element
        result: Identifier<'a>, // Result is always ptr
        array_ptr: Value<'a>,   // Must be a ptr to an array or named array type
        index: Value<'a>,       // Must be an integer type
        element_type: Type<'a>,
    },
    // Convert pointer to integer for pointer arithmetic
    PtrToInt {
        result: Identifier<'a>,
        ptr_value: Value<'a>,
        target_type: PrimitiveType, // Usually i64 for addresses
    },
    // Convert integer back to pointer
    IntToPtr {
        result: Identifier<'a>,
        int_value: Value<'a>,
        target_type: PrimitiveType, // The pointer type (usually ptr)
    },
    // --- Function Calls ---
    Call {
        result: Option<Identifier<'a>>, // None if function returns void
        func_name: Identifier<'a>,      // Name of the function to call (e.g., "@add")
        args: Vec<Value<'a>>,
        // Return type needs to be known from function signature context
    },
    // --- Phi ---
    Phi {
        result: Identifier<'a>,
        ty: Type<'a>,
        // Pairs of (value, predecessor_label)
        incoming: Vec<(Value<'a>, Label<'a>)>,
    },
    // --- I/O Operations ---
    Write {
        buffer: Value<'a>,      // Pointer to buffer to write
        size: Value<'a>,        // Size of buffer in bytes
        result: Identifier<'a>, // Result (syscall return value)
    },
    Read {
        buffer: Value<'a>,      // Pointer to buffer to read into
        size: Value<'a>,        // Maximum bytes to read
        result: Identifier<'a>, // Result (bytes read, or -1 on error)
    },
    WriteByte {
        value: Value<'a>,       // Single byte value to write
        result: Identifier<'a>, // Result (1 on success, -1 on error)
    },
    ReadByte {
        result: Identifier<'a>, // Result byte read (-1 on error)
    },

    // --- Variadic arguments ---
    /// Initialize the `valist` at `ap` so it reads the arguments past the named
    /// parameters of the enclosing variadic function.
    ///
    /// # Preconditions
    /// - `ap` must have type `ptr` and point to a `valist`.
    /// - The enclosing function must be variadic.
    VaStart { ap: Value<'a> },
    /// Read the next variadic argument as `ty` through the `valist` at `ap`, and
    /// advance `ap` past it.
    ///
    /// # Preconditions
    /// - `ap` must have type `ptr` and point to an initialized `valist`.
    /// - `ty` must be an integer, pointer or float type. Aggregates are not supported.
    VaArg {
        result: Identifier<'a>,
        ty: PrimitiveType,
        ap: Value<'a>,
    },
    /// End the lifetime of the `valist` at `ap`. Platforms that allocate state for a
    /// `valist` release it here; on x86_64 SysV this does nothing.
    VaEnd { ap: Value<'a> },
    /// Copy the `valist` at `src` into the `valist` at `dst`, so both advance
    /// independently from the same point.
    VaCopy { dst: Value<'a>, src: Value<'a> },
}

// --- Display Implementations ---

impl AllocType {
    /// Returns all valid allocation type names as strings.
    pub fn all_names() -> &'static [&'static str] {
        const fn name_for(alloc: AllocType) -> &'static str {
            alloc.as_str()
        }
        const NAMES: &[&str] = &[name_for(AllocType::Stack)];
        NAMES
    }

    /// Returns the string representation of this allocation type.
    pub const fn as_str(&self) -> &'static str {
        match self {
            AllocType::Stack => "stack",
        }
    }
}

impl fmt::Display for AllocType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Returns all valid instruction opcodes that can appear after an assignment.
///
/// These are instructions that produce a result value (e.g., `%result = add.i32 ...`).
/// The list combines dynamically generated names from BinaryOp and CmpOp enums with
/// instruction-specific opcodes.
///
/// This function keeps parser error messages in sync with the actual
/// instruction set. When adding new variants:
/// - For BinaryOp: add to BinaryOp::ALL_VARIANTS (automatically appears here)
/// - For CmpOp: add to CmpOp::ALL_VARIANTS (automatically appears here)
/// - For other instructions: add to INSTRUCTION_ASSIGNMENT_OPS below
pub fn assignment_opcode_names() -> &'static [&'static str] {
    // Instruction-specific opcodes that produce results
    // These correspond to Instruction enum variants that have a `result` field
    #[allow(dead_code)]
    const INSTRUCTION_ASSIGNMENT_OPS: &[&str] = &[
        // Type conversion operations (from Instruction::ZeroExtend, etc.)
        "extend",
        "trunc",
        "bitcast",
        "inttofp",
        "fptoint",
        "fpcast",
        "neg",
        "not",
        "sqrt",
        "popcount",
        "ctlz",
        "cttz",
        "bswap",
        "funcaddr",
        "callptr",
        "select",
        // Memory operations (from Instruction::Alloc, Instruction::Load, etc.)
        "stackalloc",
        "load",
        "volatile",
        "getfieldptr",
        "getelem",
        // Pointer operations (from Instruction::PtrToInt, Instruction::IntToPtr)
        "ptrtoint",
        "inttoptr",
        // Function calls (from Instruction::Call - can have result)
        "call",
        "asm",
        // Edge merge (from Instruction::Phi)
        "phi",
        // I/O operations (from Instruction::Write, Instruction::Read, etc. - with result)
        "write",
        "read",
        "writebyte",
        "readbyte",
        // Atomic read-modify-write and load (from Instruction::AtomicBinary, Instruction::AtomicLoad)
        "atomic_load",
        "atomic_add",
        "atomic_sub",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
        "atomic_exchange",
        "atomic_cmpxchg",
        // Variadic argument read (from Instruction::VaArg)
        "va_arg",
    ];

    // Combine BinaryOp and CmpOp (generated from enums) with instruction-specific ops
    // Note: We can't concatenate arrays at compile time, so we manually list them.
    // However, BinaryOp and CmpOp names come from their enums via all_names(),
    // so those stay automatically in sync.
    &[
        // From BinaryOp enum (dynamically generated via BinaryOp::all_names())
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "rotl",
        "rotr",
        // From CmpOp enum (dynamically generated via CmpOp::all_names())
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        // Instruction-specific (manually maintained, corresponds to Instruction enum variants)
        "extend",
        "trunc",
        "bitcast",
        "inttofp",
        "fptoint",
        "fpcast",
        "neg",
        "not",
        "sqrt",
        "popcount",
        "ctlz",
        "cttz",
        "bswap",
        "funcaddr",
        "callptr",
        "select",
        "stackalloc",
        "load",
        "volatile",
        "getfieldptr",
        "getelem",
        "ptrtoint",
        "inttoptr",
        "call",
        "phi",
        "write",
        "read",
        "writebyte",
        "readbyte",
    ]
}

/// Returns all valid instruction opcodes that don't require an assignment.
///
/// These are instructions that don't produce a result value (e.g., `ret.void`, `jmp label`).
/// The list corresponds to Instruction enum variants that don't have a `result` field.
///
/// This function keeps parser error messages in sync with the actual
/// instruction set. When adding new Instruction variants without results, add them here.
pub fn non_assignment_opcode_names() -> &'static [&'static str] {
    // These correspond to Instruction enum variants without a `result` field
    &[
        // Control flow (from Instruction::Ret, Instruction::Jmp, Instruction::Br, Instruction::Switch)
        "ret",
        "jmp",
        "br",
        "switch",
        "unreachable",
        // Function calls (from Instruction::Call - without result)
        "call",
        "callptr",
        // Memory operations (from Instruction::Store, etc.)
        "store",
        "volatile",
        // I/O operations (from Instruction::Write, Instruction::Read, etc. - without result)
        "write",
        "read",
        "writebyte",
        "readbyte",
        // Variadic arguments (from Instruction::VaStart, VaEnd, VaCopy)
        "va_start",
        "va_end",
        "va_copy",
        // Atomic store and fence (from Instruction::AtomicStore, Instruction::Fence)
        "atomic_store",
        "fence",
        // Inline assembly (from Instruction::InlineAsm)
        "asm",
    ]
}

impl<'a> Instruction<'a> {
    /// Returns true if this instruction is a basic-block terminator
    /// (i.e. it transfers control flow: `ret`, `jmp`, or `br`).
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Ret { .. }
                | Instruction::Jmp { .. }
                | Instruction::Br { .. }
                | Instruction::Switch { .. }
                | Instruction::Unreachable
        )
    }

    /// The type of the value this instruction defines, when the instruction states it.
    ///
    /// `None` for instructions whose result type depends on context outside the
    /// instruction (calls, SIMD, aggregates) and for instructions that define nothing.
    pub fn result_type(&self) -> Option<Type<'a>> {
        let prim = |p: &PrimitiveType| Some(Type::Primitive(*p));
        match self {
            Instruction::Binary { ty, .. } | Instruction::Unary { ty, .. } => prim(ty),
            Instruction::Cmp { .. } => prim(&PrimitiveType::Bool),
            Instruction::ZeroExtend { target_type, .. }
            | Instruction::SignExtend { target_type, .. }
            | Instruction::Trunc { target_type, .. }
            | Instruction::Bitcast { target_type, .. }
            | Instruction::IntToFloat { target_type, .. }
            | Instruction::FloatToInt { target_type, .. }
            | Instruction::FloatConvert { target_type, .. }
            | Instruction::PtrToInt { target_type, .. }
            | Instruction::IntToPtr { target_type, .. } => prim(target_type),
            Instruction::Alloc { .. }
            | Instruction::GetElemPtr { .. }
            | Instruction::GetFieldPtr { .. } => prim(&PrimitiveType::Ptr),
            Instruction::Select { ty, .. }
            | Instruction::Load { ty, .. }
            | Instruction::Phi { ty, .. }
            | Instruction::AtomicLoad { ty, .. }
            | Instruction::AtomicBinary { ty, .. } => Some(ty.clone()),
            Instruction::VaArg { ty, .. } => prim(ty),
            Instruction::InlineAsm {
                result: Some((_, ty)),
                ..
            } => prim(ty),
            _ => None,
        }
    }

    /// Returns the name of the value defined by this instruction, if any.
    ///
    /// Only assignment-style instructions (`Binary`, `Cmp`, `Alloc`, `Load`, `Call` with
    /// a result, etc.) define a variable. Terminators and stores do not.
    pub fn defined_var(&self) -> Option<&'a str> {
        match self {
            Instruction::Binary { result, .. }
            | Instruction::Cmp { result, .. }
            | Instruction::Alloc { result, .. }
            | Instruction::Load { result, .. }
            | Instruction::ZeroExtend { result, .. }
            | Instruction::SignExtend { result, .. }
            | Instruction::Trunc { result, .. }
            | Instruction::Bitcast { result, .. }
            | Instruction::IntToFloat { result, .. }
            | Instruction::FloatToInt { result, .. }
            | Instruction::FloatConvert { result, .. }
            | Instruction::Unary { result, .. }
            | Instruction::FuncAddr { result, .. }
            | Instruction::Select { result, .. }
            | Instruction::GetElemPtr { result, .. }
            | Instruction::GetFieldPtr { result, .. }
            | Instruction::WriteByte { result, .. }
            | Instruction::ReadByte { result, .. }
            | Instruction::Write { result, .. }
            | Instruction::Read { result, .. }
            | Instruction::Phi { result, .. }
            | Instruction::PtrToInt { result, .. }
            | Instruction::IntToPtr { result, .. }
            | Instruction::AtomicLoad { result, .. }
            | Instruction::AtomicBinary { result, .. }
            | Instruction::AtomicCompareExchange { result, .. }
            | Instruction::VaArg { result, .. }
            | Instruction::InlineAsm {
                result: Some((result, _)),
                ..
            }
            | Instruction::Call {
                result: Some(result),
                ..
            }
            | Instruction::CallIndirect {
                result: Some(result),
                ..
            } => Some(result),
            _ => None,
        }
    }

    /// Returns the names of all values *used* (read) by this instruction.
    ///
    /// Globals (prefixed `@`) and literal constants are not named values and
    /// are not returned.
    pub fn used_vars(&self) -> Vec<&'a str> {
        let mut vars: Vec<&'a str> = Vec::new();

        let push_value = |v: &Value<'a>, vars: &mut Vec<&'a str>| {
            if let Value::Variable(name) = v {
                vars.push(name);
            }
        };

        match self {
            Instruction::Binary { lhs, rhs, .. } | Instruction::Cmp { lhs, rhs, .. } => {
                push_value(lhs, &mut vars);
                push_value(rhs, &mut vars);
            }
            Instruction::Store { ptr, value, .. } => {
                push_value(ptr, &mut vars);
                push_value(value, &mut vars);
            }
            Instruction::Load { ptr, .. } => push_value(ptr, &mut vars),
            Instruction::Alloc { .. } => {}
            Instruction::VaStart { ap }
            | Instruction::VaEnd { ap }
            | Instruction::VaArg { ap, .. } => {
                push_value(ap, &mut vars);
            }
            Instruction::VaCopy { dst, src } => {
                push_value(dst, &mut vars);
                push_value(src, &mut vars);
            }
            Instruction::InlineAsm { operands, .. } => {
                for operand in operands {
                    if let AsmOperand::In { value, .. } = operand {
                        push_value(value, &mut vars);
                    }
                }
            }
            Instruction::Br { condition, .. } => push_value(condition, &mut vars),
            Instruction::Ret { value: Some(v), .. } => push_value(v, &mut vars),
            Instruction::Ret { value: None, .. }
            | Instruction::Jmp { .. }
            | Instruction::Unreachable => {}
            Instruction::Switch { value, .. }
            | Instruction::WriteByte { value, .. }
            | Instruction::ZeroExtend { value, .. }
            | Instruction::SignExtend { value, .. }
            | Instruction::Trunc { value, .. }
            | Instruction::Bitcast { value, .. }
            | Instruction::IntToFloat { value, .. }
            | Instruction::FloatToInt { value, .. }
            | Instruction::FloatConvert { value, .. }
            | Instruction::Unary { value, .. } => push_value(value, &mut vars),
            Instruction::CallIndirect { target, args, .. } => {
                push_value(target, &mut vars);
                for (_, arg) in args {
                    push_value(arg, &mut vars);
                }
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    push_value(arg, &mut vars);
                }
            }
            Instruction::GetElemPtr {
                array_ptr, index, ..
            } => {
                push_value(array_ptr, &mut vars);
                push_value(index, &mut vars);
            }
            Instruction::GetFieldPtr { struct_ptr, .. } => push_value(struct_ptr, &mut vars),
            Instruction::PtrToInt { ptr_value, .. } => push_value(ptr_value, &mut vars),
            Instruction::IntToPtr { int_value, .. } => push_value(int_value, &mut vars),
            Instruction::Select {
                cond,
                true_val,
                false_val,
                ..
            } => {
                push_value(cond, &mut vars);
                push_value(true_val, &mut vars);
                push_value(false_val, &mut vars);
            }
            Instruction::Write { buffer, size, .. } | Instruction::Read { buffer, size, .. } => {
                push_value(buffer, &mut vars);
                push_value(size, &mut vars);
            }
            Instruction::AtomicLoad { ptr, .. } => push_value(ptr, &mut vars),
            Instruction::AtomicStore { ptr, value, .. }
            | Instruction::AtomicBinary { ptr, value, .. } => {
                push_value(ptr, &mut vars);
                push_value(value, &mut vars);
            }
            Instruction::AtomicCompareExchange {
                ptr,
                expected,
                desired,
                ..
            } => {
                push_value(ptr, &mut vars);
                push_value(expected, &mut vars);
                push_value(desired, &mut vars);
            }
            Instruction::Phi { incoming, .. } => {
                for (v, _) in incoming {
                    push_value(v, &mut vars);
                }
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary { lhs, rhs, .. } => {
                push_value(lhs, &mut vars);
                push_value(rhs, &mut vars);
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary { operand, .. } => push_value(operand, &mut vars),
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary { lhs, rhs, acc, .. } => {
                push_value(lhs, &mut vars);
                push_value(rhs, &mut vars);
                push_value(acc, &mut vars);
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdShuffle { lhs, rhs, mask, .. } => {
                push_value(lhs, &mut vars);
                push_value(rhs, &mut vars);
                push_value(mask, &mut vars);
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdExtract {
                vector, lane_index, ..
            } => {
                push_value(vector, &mut vars);
                push_value(lane_index, &mut vars);
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdInsert {
                vector,
                scalar,
                lane_index,
                ..
            } => {
                push_value(vector, &mut vars);
                push_value(scalar, &mut vars);
                push_value(lane_index, &mut vars);
            }
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad { ptr, .. } => push_value(ptr, &mut vars),
            #[cfg(feature = "nightly")]
            Instruction::SimdStore { ptr, value, .. } => {
                push_value(ptr, &mut vars);
                push_value(value, &mut vars);
            }
            _ => {}
        }

        vars
    }
}

impl fmt::Display for Instruction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Binary {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => write!(f, "%{result} = {op}.{ty} {lhs}, {rhs}"),
            Instruction::Cmp {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => write!(f, "%{result} = {op}.{ty} {lhs}, {rhs}"),
            Instruction::ZeroExtend {
                result,
                source_type,
                target_type,
                value,
            } => write!(f, "%{result} = extend.{source_type}.{target_type} {value}"),
            Instruction::Select {
                result,
                ty,
                cond,
                true_val,
                false_val,
            } => write!(f, "%{result} = select.{ty} {cond}, {true_val}, {false_val}"),
            Instruction::Trunc {
                result,
                source_type,
                target_type,
                value,
            } => write!(f, "%{result} = trunc.{source_type}.{target_type} {value}"),
            Instruction::SignExtend {
                result,
                source_type,
                target_type,
                value,
            } => write!(f, "%{result} = extend.{source_type}.{target_type} {value}"),
            Instruction::Bitcast {
                result,
                source_type,
                target_type,
                value,
            } => write!(f, "%{result} = bitcast.{source_type}.{target_type} {value}"),
            Instruction::IntToFloat {
                result,
                signed: _,
                source_type,
                target_type,
                value,
            } => {
                write!(f, "%{result} = inttofp.{source_type}.{target_type} {value}")
            }
            Instruction::FloatToInt {
                result,
                signed: _,
                source_type,
                target_type,
                value,
            } => {
                write!(f, "%{result} = fptoint.{source_type}.{target_type} {value}")
            }
            Instruction::FloatConvert {
                result,
                source_type,
                target_type,
                value,
            } => {
                write!(f, "%{result} = fpcast.{source_type}.{target_type} {value}")
            }
            Instruction::Unary {
                op,
                result,
                ty,
                value,
            } => write!(f, "%{result} = {op}.{ty} {value}"),
            Instruction::FuncAddr { result, func_name } => {
                write!(f, "%{result} = funcaddr @{func_name}")
            }
            Instruction::CallIndirect {
                result,
                target,
                args,
                variadic_fixed,
                return_type,
            } => {
                let mut args: Vec<String> =
                    args.iter().map(|(ty, v)| format!("{ty} {v}")).collect();
                if let Some(fixed) = variadic_fixed {
                    args.insert((*fixed).min(args.len()), "...".to_string());
                }
                match result {
                    Some(result) => write!(
                        f,
                        "%{result} = callptr.{return_type} {target}({})",
                        args.join(", ")
                    ),
                    None => write!(f, "callptr.{return_type} {target}({})", args.join(", ")),
                }
            }
            Instruction::Switch {
                ty,
                value,
                default,
                cases,
            } => {
                write!(f, "switch.{ty} {value}, {default}")?;
                for (lit, label) in cases.iter() {
                    write!(f, ", [{lit}, {label}]")?;
                }
                Ok(())
            }
            Instruction::Br {
                condition,
                true_label,
                false_label,
            } => write!(f, "br {condition}, {true_label}, {false_label}"),
            Instruction::Jmp { target_label } => write!(f, "jmp {target_label}"),
            Instruction::Unreachable => write!(f, "unreachable"),
            Instruction::Ret { ty, value } => match value {
                Some(v) => write!(f, "ret.{ty} {v}"),
                None => write!(f, "ret.void"),
            },
            Instruction::Alloc {
                result,
                alloc_type,
                allocated_ty,
                align,
            } => match alloc_type {
                AllocType::Stack => {
                    write!(f, "%{result} = stackalloc.{allocated_ty}")?;
                    if let Some(align) = align {
                        write!(f, " align {align}")?;
                    }
                    Ok(())
                }
            },
            Instruction::Load {
                result,
                ty,
                ptr,
                volatile,
            } => {
                let prefix = if *volatile { "volatile " } else { "" };
                write!(f, "%{result} = {prefix}load.{ty} {ptr}")
            }
            Instruction::Store {
                ty,
                ptr,
                value,
                volatile,
            } => {
                let prefix = if *volatile { "volatile " } else { "" };
                write!(f, "{prefix}store.{ty} {ptr}, {value}")
            }
            Instruction::AtomicLoad {
                result,
                ty,
                ptr,
                ordering,
            } => write!(f, "%{result} = atomic_load.{ty} {ptr}, {ordering}"),
            Instruction::AtomicStore {
                ty,
                ptr,
                value,
                ordering,
            } => write!(f, "atomic_store.{ty} {ptr}, {value}, {ordering}"),
            Instruction::AtomicBinary {
                op,
                result,
                ty,
                ptr,
                value,
                ordering,
            } => write!(f, "%{result} = {op}.{ty} {ptr}, {value}, {ordering}"),
            Instruction::AtomicCompareExchange {
                result,
                ty,
                ptr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => write!(
                f,
                "%{result} = atomic_cmpxchg.{ty} {ptr}, {expected}, {desired}, {success_ordering}, {failure_ordering}"
            ),
            Instruction::Fence { ordering } => write!(f, "fence {ordering}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary {
                op,
                result,
                vector_type,
                lhs,
                rhs,
            } => write!(f, "%{result} = {op}.{vector_type} {lhs}, {rhs}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary {
                op,
                result,
                vector_type,
                operand,
            } => write!(f, "%{result} = {op}.{vector_type} {operand}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary {
                op,
                result,
                vector_type,
                lhs,
                rhs,
                acc,
            } => write!(f, "%{result} = {op}.{vector_type} {lhs}, {rhs}, {acc}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdShuffle {
                result,
                vector_type,
                lhs,
                rhs,
                mask,
            } => write!(f, "%{result} = shuffle.{vector_type} {lhs}, {rhs}, {mask}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdExtract {
                result,
                scalar_type,
                vector,
                lane_index,
            } => write!(
                f,
                "%{result} = extract_lane.{scalar_type} {vector}, {lane_index}"
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdInsert {
                result,
                vector_type,
                vector,
                scalar,
                lane_index,
            } => write!(
                f,
                "%{result} = insert_lane.{vector_type} {vector}, {scalar}, {lane_index}"
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad {
                result,
                vector_type,
                ptr,
                alignment: _,
            } => write!(f, "%{result} = load_simd.{vector_type} {ptr}"),
            #[cfg(feature = "nightly")]
            Instruction::SimdStore {
                vector_type,
                ptr,
                value,
                alignment: _,
            } => write!(f, "store_simd.{vector_type} {ptr}, {value}"),
            Instruction::GetFieldPtr {
                result,
                struct_ptr,
                struct_type,
                field_index,
            } => write!(
                f,
                "%{result} = getfieldptr.{struct_type} {struct_ptr}, {field_index}"
            ),
            Instruction::GetElemPtr {
                result,
                array_ptr,
                index,
                element_type,
            } => write!(f, "%{result} = getelem.{element_type} {array_ptr}, {index}"),
            Instruction::PtrToInt {
                result,
                ptr_value,
                target_type,
            } => write!(f, "%{result} = ptrtoint.{target_type} {ptr_value}"),
            Instruction::IntToPtr {
                result,
                int_value,
                target_type,
            } => write!(f, "%{result} = inttoptr.{target_type} {int_value}"),
            Instruction::Call {
                result,
                func_name,
                args,
            } => {
                if let Some(res) = result {
                    write!(f, "%{res} = call @{func_name}(")?;
                } else {
                    write!(f, "call @{func_name}(")?;
                }
                for (i, arg) in args.iter().enumerate() {
                    write!(f, "{arg}")?;
                    if i < args.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
            Instruction::Phi {
                result,
                ty,
                incoming,
            } => {
                write!(f, "%{result} = phi.{ty} ")?;
                for (i, (val, label)) in incoming.iter().enumerate() {
                    write!(f, "[{val}, {label}]")?;
                    if i < incoming.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                Ok(())
            }
            Instruction::Write {
                buffer,
                size,
                result,
            } => write!(f, "%{result} = write {buffer}, {size}"),
            Instruction::Read {
                buffer,
                size,
                result,
            } => write!(f, "%{result} = read {buffer}, {size}"),
            Instruction::WriteByte { value, result } => {
                write!(f, "%{result} = writebyte {value}")
            }
            Instruction::ReadByte { result } => write!(f, "%{result} = readbyte"),
            Instruction::VaStart { ap } => write!(f, "va_start {ap}"),
            Instruction::VaArg { result, ty, ap } => {
                write!(f, "%{result} = va_arg.{ty} {ap}")
            }
            Instruction::VaEnd { ap } => write!(f, "va_end {ap}"),
            Instruction::VaCopy { dst, src } => write!(f, "va_copy {dst}, {src}"),
            Instruction::InlineAsm {
                result,
                template,
                operands,
                clobbers,
            } => {
                match result {
                    Some((name, ty)) => write!(f, "%{name} = asm.{ty} \"{template}\"")?,
                    None => write!(f, "asm.void \"{template}\"")?,
                }
                for operand in operands {
                    match operand {
                        AsmOperand::In { location, value } => {
                            write!(f, ", in({location}) {value}")?
                        }
                        AsmOperand::Out { location } => write!(f, ", out({location})")?,
                    }
                }
                if !clobbers.is_empty() {
                    write!(f, ", clobber({})", clobbers.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Literal, PrimitiveType, Type, Value}; // Assuming crate root

    #[test]
    fn test_display_binary_op() {
        assert_eq!(format!("{}", BinaryOp::Add), "add");
        assert_eq!(format!("{}", BinaryOp::Sub), "sub");
        assert_eq!(format!("{}", BinaryOp::Mul), "mul");
        assert_eq!(format!("{}", BinaryOp::Div), "div");
        assert_eq!(format!("{}", BinaryOp::Rem), "rem");
        assert_eq!(format!("{}", BinaryOp::And), "and");
        assert_eq!(format!("{}", BinaryOp::Or), "or");
        assert_eq!(format!("{}", BinaryOp::Xor), "xor");
        assert_eq!(format!("{}", BinaryOp::Shl), "shl");
        assert_eq!(format!("{}", BinaryOp::Shr), "shr");
    }

    #[test]
    fn test_display_cmp_op() {
        assert_eq!(format!("{}", CmpOp::Eq), "eq");
        assert_eq!(format!("{}", CmpOp::Ne), "ne");
        assert_eq!(format!("{}", CmpOp::Gt), "gt");
        assert_eq!(format!("{}", CmpOp::Ge), "ge");
        assert_eq!(format!("{}", CmpOp::Lt), "lt");
        assert_eq!(format!("{}", CmpOp::Le), "le");
    }

    #[test]
    fn test_display_alloc_type() {
        assert_eq!(format!("{}", AllocType::Stack), "stack");
    }

    #[test]
    fn test_display_instruction() {
        // Binary
        let instr1 = Instruction::Binary {
            op: BinaryOp::Add,
            result: "res1",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("a"),
            rhs: Value::Constant(Literal::I32(5)),
        };
        assert_eq!(format!("{}", instr1), "%res1 = add.i32 %a, 5");

        // Cmp
        let instr2 = Instruction::Cmp {
            op: CmpOp::Lt,
            result: "cond",
            ty: PrimitiveType::F32,
            lhs: Value::Variable("x"),
            rhs: Value::Constant(Literal::F32(0.0)),
        };
        assert_eq!(format!("{}", instr2), "%cond = lt.f32 %x, 0"); // Note: f32 display might vary

        // Br
        let instr3 = Instruction::Br {
            condition: Value::Variable("cond"),
            true_label: "if_true",
            false_label: "if_false",
        };
        assert_eq!(format!("{}", instr3), "br %cond, if_true, if_false");

        // Jmp
        let instr4 = Instruction::Jmp {
            target_label: "loop_start",
        };
        assert_eq!(format!("{}", instr4), "jmp loop_start");

        // Ret (Void)
        let instr5 = Instruction::Ret {
            ty: Type::Void,
            value: None,
        };
        assert_eq!(format!("{}", instr5), "ret.void");

        // Ret (Value)
        let instr6 = Instruction::Ret {
            ty: Type::Primitive(PrimitiveType::I64),
            value: Some(Value::Variable("final_result")),
        };
        assert_eq!(format!("{}", instr6), "ret.i64 %final_result");

        // Alloc (Stack)
        let instr7 = Instruction::Alloc {
            result: "ptr1",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I32),
            align: None,
        };
        assert_eq!(format!("{}", instr7), "%ptr1 = stackalloc.i32");
        let instr8 = Instruction::Alloc {
            result: "ptr2",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
            align: Some(16),
        };
        assert_eq!(format!("{}", instr8), "%ptr2 = stackalloc.i64 align 16");

        // Load
        let instr9 = Instruction::Load {
            result: "loaded_val",
            ty: Type::Primitive(PrimitiveType::F32),
            ptr: Value::Variable("some_ptr"),
            volatile: false,
        };
        assert_eq!(format!("{}", instr9), "%loaded_val = load.f32 %some_ptr");

        let instr10 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("bool_ptr"),
            value: Value::Constant(Literal::Bool(true)),
            volatile: false,
        };
        assert_eq!(format!("{}", instr10), "store.bool %bool_ptr, true");

        let volatile_load = Instruction::Load {
            result: "v",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("p"),
            volatile: true,
        };
        assert_eq!(format!("{}", volatile_load), "%v = volatile load.i64 %p");

        let volatile_store = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("p"),
            value: Value::Constant(Literal::I64(5)),
            volatile: true,
        };
        assert_eq!(format!("{}", volatile_store), "volatile store.i64 %p, 5");

        // GetFieldPtr
        let instr11 = Instruction::GetFieldPtr {
            result: "field_ptr",
            struct_ptr: Value::Variable("struct_instance_ptr"),
            struct_type: Type::Named("Pair"),
            field_index: 1,
        };
        assert_eq!(
            format!("{}", instr11),
            "%field_ptr = getfieldptr.!Pair %struct_instance_ptr, 1"
        );

        // GetElemPtr
        let instr12 = Instruction::GetElemPtr {
            result: "elem_ptr",
            array_ptr: Value::Variable("array_data_ptr"),
            index: Value::Constant(Literal::I64(3)),
            element_type: Type::Primitive(PrimitiveType::I64),
        };
        assert_eq!(
            format!("{}", instr12),
            "%elem_ptr = getelem.i64 %array_data_ptr, 3"
        );

        // Call (void)
        let instr13 = Instruction::Call {
            result: None,
            func_name: "print_message",
            args: vec![Value::Constant(Literal::String("test"))],
        };
        assert_eq!(format!("{}", instr13), "call @print_message(\"test\")"); // Escaped quotes

        // Call (with result)
        let instr14 = Instruction::Call {
            result: Some("sum"),
            func_name: "calculate",
            args: vec![Value::Variable("x"), Value::Variable("y")],
        };
        assert_eq!(format!("{}", instr14), "%sum = call @calculate(%x, %y)");

        // Phi
        let instr15 = Instruction::Phi {
            result: "merged_val",
            ty: Type::Primitive(PrimitiveType::I32),
            incoming: vec![
                (Value::Variable("val1"), "label1"),
                (Value::Constant(Literal::I32(10)), "label2"),
            ],
        };
        assert_eq!(
            format!("{}", instr15),
            "%merged_val = phi.i32 [%val1, label1], [10, label2]"
        );

        // ZeroExtend
        let instr17 = Instruction::ZeroExtend {
            result: "extended_val",
            source_type: PrimitiveType::U8,
            target_type: PrimitiveType::I64,
            value: Value::Variable("byte_val"),
        };
        assert_eq!(
            format!("{}", instr17),
            "%extended_val = extend.u8.i64 %byte_val"
        );

        // Select
        let instr_select = Instruction::Select {
            result: "sel",
            ty: Type::Primitive(PrimitiveType::I32),
            cond: Value::Variable("cond"),
            true_val: Value::Variable("a"),
            false_val: Value::Variable("b"),
        };
        assert_eq!(
            format!("{}", instr_select),
            "%sel = select.i32 %cond, %a, %b"
        );

        // Trunc
        let instr_trunc = Instruction::Trunc {
            result: "trunc_val",
            source_type: PrimitiveType::I64,
            target_type: PrimitiveType::I32,
            value: Value::Variable("wide_val"),
        };
        assert_eq!(
            format!("{}", instr_trunc),
            "%trunc_val = trunc.i64.i32 %wide_val"
        );

        // SignExtend
        let instr_sext = Instruction::SignExtend {
            result: "sext_val",
            source_type: PrimitiveType::I32,
            target_type: PrimitiveType::I64,
            value: Value::Variable("narrow_val"),
        };
        assert_eq!(
            format!("{}", instr_sext),
            "%sext_val = extend.i32.i64 %narrow_val"
        );

        // Bitcast
        let instr_bitcast = Instruction::Bitcast {
            result: "cast_val",
            source_type: PrimitiveType::I32,
            target_type: PrimitiveType::F32,
            value: Value::Variable("bits"),
        };
        assert_eq!(
            format!("{}", instr_bitcast),
            "%cast_val = bitcast.i32.f32 %bits"
        );

        // Switch
        let instr_switch = Instruction::Switch {
            ty: PrimitiveType::I32,
            value: Value::Variable("x"),
            default: "default",
            cases: vec![(Literal::I32(0), "zero"), (Literal::I32(1), "one")],
        };
        assert_eq!(
            format!("{}", instr_switch),
            "switch.i32 %x, default, [0, zero], [1, one]"
        );
    }
}
