//! # Lamina Intermediate Representation (IR)
//!
//! The Lamina IR is a low-level, architecture-agnostic intermediate representation
//! designed to bridge high-level source code and machine code. It provides a clean,
//! efficient representation that enables powerful optimizations and easy code generation
//! for multiple target architectures.
//!
//! ## Design Principles
//!
//! - **SSA Form**: Single Static Assignment form ensures each variable is assigned exactly once
//! - **Type Safety**: Strong typing system prevents many classes of errors
//! - **Architecture Agnostic**: IR operations are independent of target architecture
//! - **Zero-Copy**: Uses string references to avoid unnecessary memory allocations
//! - **Extensible**: Easy to add new instruction types and operations
//!
//! ## Core Concepts
//!
//! ### Values and Types
//!
//! The IR operates on **values** that have specific **types**. Values can be:
//! - **Variables**: SSA variables like `%result`
//! - **Constants**: Literal values like `42`, `true`, `"hello"`
//! - **Globals**: Global variables like `@message`
//!
//! Types include primitives (`i32`, `f64`, `bool`), composite types (structs, arrays),
//! and function types.
//!
//! ### Instructions
//!
//! Instructions perform operations on values and produce new values. They are organized into:
//! - **Arithmetic**: Addition, subtraction, multiplication, division
//! - **Comparison**: Equality, ordering, logical operations
//! - **Memory**: Allocation, loading, storing, pointer arithmetic
//! - **Control Flow**: Branches, jumps, function calls, returns
//! - **Type Conversion**: Zero-extension, type casting
//!
//! ### Functions and Basic Blocks
//!
//! Functions contain **basic blocks**, which are sequences of instructions ending with
//! a terminator (branch, jump, or return). This structure enables:
//! - **Control flow analysis**: Easy to determine program flow
//! - **Optimization**: Block-level optimizations are straightforward
//! - **Code generation**: Natural mapping to assembly code
//!
//! ## Module Organization
//!
//! ### [`builder`] - IR Construction
//!
//! The `IRBuilder` provides a fluent API for programmatically constructing IR modules.
//! It handles the complexity of managing functions, basic blocks, and instruction sequences.
//!
//! ```rust
//! use lamina::{IRBuilder, Type, PrimitiveType, var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("add", Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "result", PrimitiveType::I32, i32(10), i32(20))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//! ```
//!
//! ### [`function`] - Function Representation
//!
//! Defines the structure of functions, including:
//! - **Function signatures**: Parameters and return types
//! - **Basic blocks**: Sequences of instructions
//! - **Annotations**: Metadata like `@inline`, `@export`
//! - **Control flow**: How blocks connect via branches and jumps
//!
//! ### [`instruction`] - Instruction Definitions
//!
//! Contains all instruction types and their operands:
//! - **Binary operations**: Arithmetic and logical operations
//! - **Memory operations**: Allocation, access, and deallocation
//! - **Control flow**: Branches, jumps, and function calls
//! - **Type operations**: Conversions and extensions
//!
//! ### [`module`] - Top-Level Organization
//!
//! Represents complete IR modules containing:
//! - **Functions**: All function definitions
//! - **Type declarations**: Named type definitions
//! - **Global variables**: Module-level data
//! - **Metadata**: Module-level annotations and attributes
//!
//! ### [`types`] - Type System
//!
//! Defines the complete type system:
//! - **Primitive types**: Integers, floats, booleans, pointers
//! - **Composite types**: Structs, arrays, tuples
//! - **Function types**: Parameter and return type specifications
//! - **Value representations**: How values are stored and manipulated
//!
//! ## Memory Management
//!
//! The IR supports two memory allocation strategies:
//!
//! ### Stack Allocation
//! - **Automatic cleanup**: Memory is freed when function returns
//! - **Fast allocation**: Minimal overhead
//! - **Limited lifetime**: Only valid within function scope
//! - **Use cases**: Local variables, temporary data
//!
//! ### Heap Allocation
//! - **Manual management**: Must explicitly deallocate
//! - **Persistent**: Survives function returns
//! - **Flexible**: Can be passed between functions
//! - **Use cases**: Dynamic data structures, long-lived data
//!
//! ## Example IR Program
//!
//! ```lamina
//! // Function that computes factorial
//! fn @factorial(i64 %n) -> i64 {
//!   entry:
//!     %is_zero = eq.i64 %n, 0
//!     br %is_zero, base_case, recursive_case
//!   
//!   base_case:
//!     ret.i64 1
//!   
//!   recursive_case:
//!     %n_minus_1 = sub.i64 %n, 1
//!     %factorial_n_minus_1 = call @factorial(%n_minus_1)
//!     %result = mul.i64 %n, %factorial_n_minus_1
//!     ret.i64 %result
//! }
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Zero-copy parsing**: String references avoid allocations
//! - **Efficient lookups**: HashMaps provide O(1) access
//! - **Cache-friendly**: Contiguous memory layout for instructions
//! - **Optimization-ready**: SSA form enables many optimizations
//!
//! ## Thread Safety
//!
//! IR structures are designed for safe concurrent access:
//! - **Immutable by default**: Built once, read many times
//! - **Copy semantics**: Easy to duplicate for analysis
//! - **Lifetime management**: Rust's ownership system prevents use-after-free
//!
//! ## Future Extensions
//!
//! The IR is designed to be extensible:
//! - **New instruction types**: Easy to add new operations
//! - **Additional types**: Support for new data types
//! - **Optimization hints**: Metadata for better code generation
//! - **Debug information**: Source-level debugging support
pub mod builder;
pub mod function;
pub mod instruction;
pub mod module;
pub mod types;

// Re-export the IR structures
pub use builder::IRBuilder;
pub use function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
};
pub use instruction::{AllocType, BinaryOp, CmpOp, Instruction};
pub use module::{GlobalDeclaration, Module, TypeDeclaration};
pub use types::{Identifier, Literal, PrimitiveType, StructField, Type, Value};
