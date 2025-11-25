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
//! - **Allocation**: Stack and heap memory allocation
//! - **Access**: Loading and storing values to/from memory
//! - **Pointer arithmetic**: Array indexing, struct field access
//! - **Deallocation**: Freeing heap-allocated memory
//!
//! ### Control Flow
//! - **Branches**: Conditional jumps based on boolean values
//! - **Jumps**: Unconditional jumps to other basic blocks
//! - **Function calls**: Invoking other functions
//! - **Returns**: Exiting functions with optional return values
//!
//! ### Debug and I/O
//! - **Print**: Debug output for development and testing
//! - **Phi nodes**: SSA form value merging from different control paths
//!
//! ## SSA Form
//!
//! All instructions follow Single Static Assignment (SSA) form:
//! - Each variable is assigned exactly once
//! - Instructions produce new values rather than modifying existing ones
//! - This enables powerful optimizations and simplifies analysis
//!
//! ## Example
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("example", Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
//!     .binary(BinaryOp::Mul, "result", PrimitiveType::I32, var("sum"), i32(2))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//! ```

use std::fmt;

use super::types::{Identifier, Label, PrimitiveType, Type, Value};

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
#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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
/// ensuring thread-safe updates to shared memory locations.
#[cfg(feature = "nightly")]
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

#[cfg(feature = "nightly")]
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
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
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
    /// Addition: `result = lhs + rhs`
    Add,
    /// Subtraction: `result = lhs - rhs`
    Sub,
    /// Multiplication: `result = lhs * rhs`
    Mul,
    /// Division: `result = lhs / rhs`
    Div,
    /// Remainder: `result = lhs % rhs`
    Rem,
    /// Bitwise AND: `result = lhs & rhs`
    And,
    /// Bitwise OR: `result = lhs | rhs`
    Or,
    /// Bitwise XOR: `result = lhs ^ rhs`
    Xor,
    /// Logical shift left: `result = lhs << rhs`
    Shl,
    /// Arithmetic shift right: `result = lhs >> rhs`
    ///
    /// For signed integers this preserves the sign bit. For unsigned
    /// integers the semantics match the target's arithmetic right shift.
    Shr,
}

/// Comparison operations that produce boolean results.
///
/// These operations compare two values and return a boolean indicating
/// whether the comparison is true or false.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, CmpOp};
/// use lamina::ir::builder::{var, i32};
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

/// Memory allocation strategies.
///
/// Determines where and how memory is allocated for variables.
/// This affects the lifetime and performance characteristics of the allocation.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, AllocType};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("memory", Type::Void)
///     .alloc_stack("local", Type::Primitive(PrimitiveType::I32))  // Stack allocation
///     .alloc_heap("persistent", Type::Primitive(PrimitiveType::I32)); // Heap allocation
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocType {
    /// Stack allocation - automatic cleanup, fast, limited lifetime
    Stack,
    /// Heap allocation - manual cleanup, slower, persistent
    Heap,
}

// Represents a single instruction in a basic block
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction<'a> {
    // --- Arithmetic and Logic ---
    Binary {
        op: BinaryOp,
        result: Identifier<'a>,
        ty: PrimitiveType, // Assuming binary ops work on primitives
        lhs: Value<'a>,
        rhs: Value<'a>,
    },
    // --- Comparison ---
    Cmp {
        op: CmpOp,
        result: Identifier<'a>,
        ty: PrimitiveType, // Type being compared
        lhs: Value<'a>,
        rhs: Value<'a>,
    },
    // --- Type Conversion ---
    ZeroExtend {
        result: Identifier<'a>,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    },
    // --- Control Flow ---
    Br {
        // Conditional branch
        condition: Value<'a>, // Must be bool type
        true_label: Label<'a>,
        false_label: Label<'a>,
    },
    Jmp {
        // Unconditional jump
        target_label: Label<'a>,
    },
    Ret {
        // Return from function
        ty: Type<'a>,             // Use Type::Void for void returns
        value: Option<Value<'a>>, // None if ty is Void
    },
    // --- Memory Operations ---
    Alloc {
        result: Identifier<'a>, // Result is always a ptr
        alloc_type: AllocType,
        allocated_ty: Type<'a>, // Type of the memory being allocated
    },
    Load {
        result: Identifier<'a>,
        ty: Type<'a>,   // Type being loaded
        ptr: Value<'a>, // Must be a ptr type
    },
    Store {
        ty: Type<'a>,   // Type being stored
        ptr: Value<'a>, // Must be a ptr type
        value: Value<'a>,
    },
    Dealloc {
        // Optional Heap Deallocation
        ptr: Value<'a>, // Must be a ptr from alloc.heap
    },
    // --- Atomic Operations ---
    #[cfg(feature = "nightly")]
    AtomicLoad {
        // Atomic load with memory ordering
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        ordering: MemoryOrdering,
    },
    #[cfg(feature = "nightly")]
    AtomicStore {
        // Atomic store with memory ordering
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    },
    #[cfg(feature = "nightly")]
    AtomicBinary {
        // Atomic read-modify-write operation
        op: AtomicBinOp,
        result: Identifier<'a>,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    },
    #[cfg(feature = "nightly")]
    AtomicCompareExchange {
        // Atomic compare-exchange operation
        result: Identifier<'a>,
        success: Identifier<'a>, // Boolean indicating if exchange succeeded
        ty: Type<'a>,
        ptr: Value<'a>,
        expected: Value<'a>,
        desired: Value<'a>,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    },
    #[cfg(feature = "nightly")]
    Fence {
        // Memory fence/barrier
        ordering: MemoryOrdering,
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
        field_index: usize,
    },
    GetElemPtr {
        // Get pointer to array element
        result: Identifier<'a>,      // Result is always ptr
        array_ptr: Value<'a>,        // Must be a ptr to an array or named array type
        index: Value<'a>,            // Must be an integer type
        element_type: PrimitiveType, // Type of elements in the array (for proper sizing)
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
    Tuple {
        // Create a tuple
        result: Identifier<'a>,
        elements: Vec<Value<'a>>,
    },
    ExtractTuple {
        // Extract element from tuple
        result: Identifier<'a>,
        tuple_val: Value<'a>, // Must be a tuple type
        index: usize,
    },
    // --- Function Calls ---
    Call {
        result: Option<Identifier<'a>>, // None if function returns void
        func_name: Identifier<'a>,      // Name of the function to call (e.g., "@add")
        args: Vec<Value<'a>>,
        // Return type needs to be known from function signature context
    },
    // --- SSA ---
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
    WritePtr {
        ptr: Value<'a>,         // Pointer whose address/value to write to output
        result: Identifier<'a>, // Result (bytes written on success, -1 on error)
    },

    // --- Debugging ---
    Print {
        value: Value<'a>, // Value to print (currently assumes i64 for printf)
    },
}

// --- Display Implementations ---

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
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
            }
        )
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CmpOp::Eq => "eq",
                CmpOp::Ne => "ne",
                CmpOp::Gt => "gt",
                CmpOp::Ge => "ge",
                CmpOp::Lt => "lt",
                CmpOp::Le => "le",
            }
        )
    }
}

impl fmt::Display for AllocType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                AllocType::Stack => "stack",
                AllocType::Heap => "heap",
            }
        )
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
            } => write!(f, "%{} = {}.{} {}, {}", result, op, ty, lhs, rhs),
            Instruction::Cmp {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => write!(f, "%{} = {}.{} {}, {}", result, op, ty, lhs, rhs),
            Instruction::ZeroExtend {
                result,
                source_type,
                target_type,
                value,
            } => write!(
                f,
                "%{} = zext.{}.{} {}",
                result, source_type, target_type, value
            ),
            Instruction::Br {
                condition,
                true_label,
                false_label,
            } => write!(f, "br {}, {}, {}", condition, true_label, false_label),
            Instruction::Jmp { target_label } => write!(f, "jmp {}", target_label),
            Instruction::Ret { ty, value } => match value {
                Some(v) => write!(f, "ret.{} {}", ty, v),
                None => write!(f, "ret.void"),
            },
            Instruction::Alloc {
                result,
                alloc_type,
                allocated_ty,
            } => write!(f, "%{} = alloc.ptr.{} {}", result, alloc_type, allocated_ty),
            Instruction::Load { result, ty, ptr } => write!(f, "%{} = load.{} {}", result, ty, ptr),
            Instruction::Store { ty, ptr, value } => write!(f, "store.{} {}, {}", ty, ptr, value),
            Instruction::Dealloc { ptr } => write!(f, "dealloc.heap {}", ptr),
            #[cfg(feature = "nightly")]
            Instruction::AtomicLoad {
                result,
                ty,
                ptr,
                ordering,
            } => write!(f, "%{} = atomic_load.{} {}, {}", result, ty, ptr, ordering),
            #[cfg(feature = "nightly")]
            Instruction::AtomicStore {
                ty,
                ptr,
                value,
                ordering,
            } => write!(f, "atomic_store.{} {}, {}, {}", ty, ptr, value, ordering),
            #[cfg(feature = "nightly")]
            Instruction::AtomicBinary {
                op,
                result,
                ty,
                ptr,
                value,
                ordering,
            } => write!(
                f,
                "%{} = {}.{} {}, {}, {}",
                result, op, ty, ptr, value, ordering
            ),
            #[cfg(feature = "nightly")]
            Instruction::AtomicCompareExchange {
                result,
                success,
                ty,
                ptr,
                expected,
                desired,
                success_ordering,
                failure_ordering,
            } => write!(
                f,
                "%{}, %{} = atomic_cmpxchg.{} {}, {}, {}, {}, {}",
                result, success, ty, ptr, expected, desired, success_ordering, failure_ordering
            ),
            #[cfg(feature = "nightly")]
            Instruction::Fence { ordering } => write!(f, "fence {}", ordering),
            #[cfg(feature = "nightly")]
            Instruction::SimdBinary {
                op,
                result,
                vector_type,
                lhs,
                rhs,
            } => write!(f, "%{} = {}.{} {}, {}", result, op, vector_type, lhs, rhs),
            #[cfg(feature = "nightly")]
            Instruction::SimdUnary {
                op,
                result,
                vector_type,
                operand,
            } => write!(f, "%{} = {}.{} {}", result, op, vector_type, operand),
            #[cfg(feature = "nightly")]
            Instruction::SimdTernary {
                op,
                result,
                vector_type,
                lhs,
                rhs,
                acc,
            } => write!(
                f,
                "%{} = {}.{} {}, {}, {}",
                result, op, vector_type, lhs, rhs, acc
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdShuffle {
                result,
                vector_type,
                lhs,
                rhs,
                mask,
            } => write!(
                f,
                "%{} = shuffle.{} {}, {}, {}",
                result, vector_type, lhs, rhs, mask
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdExtract {
                result,
                scalar_type,
                vector,
                lane_index,
            } => write!(
                f,
                "%{} = extract_lane.{} {}, {}",
                result, scalar_type, vector, lane_index
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
                "%{} = insert_lane.{} {}, {}, {}",
                result, vector_type, vector, scalar, lane_index
            ),
            #[cfg(feature = "nightly")]
            Instruction::SimdLoad {
                result,
                vector_type,
                ptr,
                alignment: _,
            } => write!(f, "%{} = load_simd.{} {}", result, vector_type, ptr),
            #[cfg(feature = "nightly")]
            Instruction::SimdStore {
                vector_type,
                ptr,
                value,
                alignment: _,
            } => write!(f, "store_simd.{} {}, {}", vector_type, ptr, value),
            Instruction::GetFieldPtr {
                result,
                struct_ptr,
                field_index,
            } => write!(
                f,
                "%{} = getfield.ptr {}, {}",
                result, struct_ptr, field_index
            ),
            Instruction::GetElemPtr {
                result,
                array_ptr,
                index,
                element_type,
            } => write!(
                f,
                "%{} = getelem.ptr {}, {}, {}",
                result, array_ptr, index, element_type
            ),
            Instruction::PtrToInt {
                result,
                ptr_value,
                target_type,
            } => write!(f, "%{} = ptrtoint {}, {}", result, ptr_value, target_type),
            Instruction::IntToPtr {
                result,
                int_value,
                target_type,
            } => write!(f, "%{} = inttoptr {}, {}", result, int_value, target_type),
            Instruction::Tuple { result, elements } => {
                write!(f, "%{} = tuple", result)?;
                for elem in elements {
                    write!(f, ", {}", elem)?;
                }
                Ok(())
            }
            Instruction::ExtractTuple {
                result,
                tuple_val,
                index,
            } => write!(f, "%{} = extract.tuple {}, {}", result, tuple_val, index),
            Instruction::Call {
                result,
                func_name,
                args,
            } => {
                if let Some(res) = result {
                    write!(f, "%{} = call @{}(", res, func_name)?;
                } else {
                    write!(f, "call @{}(", func_name)?;
                }
                for (i, arg) in args.iter().enumerate() {
                    write!(f, "{}", arg)?;
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
                write!(f, "%{} = phi.{} ", result, ty)?;
                for (i, (val, label)) in incoming.iter().enumerate() {
                    write!(f, "[{}, {}]", val, label)?;
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
            } => write!(f, "%{} = write {}, {}", result, buffer, size),
            Instruction::Read {
                buffer,
                size,
                result,
            } => write!(f, "%{} = read {}, {}", result, buffer, size),
            Instruction::WriteByte { value, result } => {
                write!(f, "%{} = writebyte {}", result, value)
            }
            Instruction::ReadByte { result } => write!(f, "%{} = readbyte", result),
            Instruction::WritePtr { ptr, result } => write!(f, "%{} = writeptr {}", result, ptr),
            Instruction::Print { value } => write!(f, "print {}", value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{Literal, PrimitiveType, StructField, Type, Value}; // Assuming crate root

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
        assert_eq!(format!("{}", AllocType::Heap), "heap");
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
        };
        assert_eq!(format!("{}", instr7), "%ptr1 = alloc.ptr.stack i32");

        // Alloc (Heap)
        let struct_type = Type::Struct(vec![StructField {
            name: "a",
            ty: Type::Primitive(PrimitiveType::I32),
        }]);
        let instr8 = Instruction::Alloc {
            result: "heap_ptr",
            alloc_type: AllocType::Heap,
            allocated_ty: struct_type.clone(), // Clone necessary if used later
        };
        assert_eq!(
            format!("{}", instr8),
            "%heap_ptr = alloc.ptr.heap struct { a: i32 }"
        );

        // Load
        let instr9 = Instruction::Load {
            result: "loaded_val",
            ty: Type::Primitive(PrimitiveType::F32),
            ptr: Value::Variable("some_ptr"),
        };
        assert_eq!(format!("{}", instr9), "%loaded_val = load.f32 %some_ptr");

        // Store
        let instr10 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("bool_ptr"),
            value: Value::Constant(Literal::Bool(true)),
        };
        assert_eq!(format!("{}", instr10), "store.bool %bool_ptr, true");

        // GetFieldPtr
        let instr11 = Instruction::GetFieldPtr {
            result: "field_ptr",
            struct_ptr: Value::Variable("struct_instance_ptr"),
            field_index: 1,
        };
        assert_eq!(
            format!("{}", instr11),
            "%field_ptr = getfield.ptr %struct_instance_ptr, 1"
        );

        // GetElemPtr
        let instr12 = Instruction::GetElemPtr {
            result: "elem_ptr",
            array_ptr: Value::Variable("array_data_ptr"),
            index: Value::Constant(Literal::I64(3)),
            element_type: PrimitiveType::I64,
        };
        assert_eq!(
            format!("{}", instr12),
            "%elem_ptr = getelem.ptr %array_data_ptr, 3, i64"
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

        // Print
        let instr16 = Instruction::Print {
            value: Value::Variable("debug_val"),
        };
        assert_eq!(format!("{}", instr16), "print %debug_val");

        // ZeroExtend
        let instr17 = Instruction::ZeroExtend {
            result: "extended_val",
            source_type: PrimitiveType::I8,
            target_type: PrimitiveType::I64,
            value: Value::Variable("byte_val"),
        };
        assert_eq!(
            format!("{}", instr17),
            "%extended_val = zext.i8.i64 %byte_val"
        );

        // Tuple
        let instr18 = Instruction::Tuple {
            result: "my_tuple",
            elements: vec![Value::Constant(Literal::I32(1)), Value::Variable("v2")],
        };
        assert_eq!(format!("{}", instr18), "%my_tuple = tuple, 1, %v2");

        // ExtractTuple
        let instr19 = Instruction::ExtractTuple {
            result: "elem0",
            tuple_val: Value::Variable("my_tuple"),
            index: 0,
        };
        assert_eq!(
            format!("{}", instr19),
            "%elem0 = extract.tuple %my_tuple, 0"
        );

        // Dealloc
        let instr20 = Instruction::Dealloc {
            ptr: Value::Variable("heap_ptr"),
        };
        assert_eq!(format!("{}", instr20), "dealloc.heap %heap_ptr");
    }
}



