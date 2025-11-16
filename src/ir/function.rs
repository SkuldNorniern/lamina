//! # Function Representation
//!
//! This module defines the structures for representing functions in the Lamina IR.
//! Functions are the primary unit of code organization and execution, containing
//! basic blocks which hold the actual instructions.
//!
//! ## Key Concepts
//!
//! - **Functions**: Complete function definitions with signatures and implementations
//! - **Basic Blocks**: Linear sequences of instructions ending with a terminator
//! - **Function Signatures**: Parameter types and return types
//! - **Annotations**: Metadata that affects function behavior and optimization
//! - **SSA Variables**: Single Static Assignment form for all variables
//!
//! ## Function Structure
//!
//! A function consists of:
//! 1. **Signature** (`FunctionSignature`): Parameters and return type
//! 2. **Basic Blocks** (`HashMap<&str, BasicBlock>`): The actual implementation
//! 3. **Entry Block** (`&str`): The first block to execute when called
//! 4. **Annotations** (`Vec<FunctionAnnotation>`): Metadata like `@inline`, `@export`
//!
//! ## Function Signature
//!
//! ```rust
//! // Example function signature in source language
//! fn add_numbers(a: i32, b: i32) -> i32 { a + b }
//!
//! // Corresponding IR structure
//! # use lamina::ir::{Type, PrimitiveType};
//! # use lamina::ir::function::{FunctionSignature, FunctionParameter};
//! let signature = FunctionSignature {
//!     params: vec![
//!         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) },
//!         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32) }
//!     ],
//!     return_type: Type::Primitive(PrimitiveType::I32)
//! };
//! ```
//!
//! ## Basic Block Organization
//!
//! ### Block Structure
//! ```text
//! block_name:
//!     instruction1
//!     instruction2
//!     ...
//!     terminator_instruction
//! ```
//!
//! ### Terminator Instructions
//! Each basic block must end with a **terminator instruction**:
//! - `Br` (conditional branch): `br %condition, "true_block", "false_block"`
//! - `Jmp` (unconditional jump): `jmp "target_block"`
//! - `Ret` (return): `ret.i32 %value` or `ret.void`
//!
//! ### Control Flow Rules
//! - Instructions within a block execute sequentially
//! - Control flow only changes at block boundaries
//! - No jumps into the middle of blocks
//! - All paths must eventually reach a terminator
//!
//! ## Function Annotations
//!
//! ### Built-in Annotations
//! - **`@inline`**: Suggest the function should be inlined
//! - **`@export`**: Make the function visible to other modules
//! - **`@internal`**: Function is only used within this module
//! - **`@noreturn`**: Function never returns (e.g., exits the program)
//!
//! ### Custom Annotations
//! Functions can have custom annotations for:
//! - **Optimization hints**: `@optimize_speed`, `@optimize_size`
//! - **Debugging**: `@debug_info`, `@trace_calls`
//! - **Security**: `@trusted`, `@untrusted`
//! - **Platform-specific**: `@windows_only`, `@linux_only`
//!
//! ## Variable Management (SSA Form)
//!
//! ### Single Static Assignment
//! - Each variable is assigned exactly **once** in its lifetime
//! - Variables are immutable after assignment
//! - Enables powerful optimizations and analysis
//!
//! ### Variable Naming Convention
//! ```text
//! %result      # Standard variable
//! %temp_1      # Temporary variable
//! %cond        # Condition variable
//! %ptr         # Pointer variable
//! ```
//!
//! ### Variable Scoping
//! - Variables are scoped to their function
//! - Variables from different blocks can be merged using `phi` nodes
//! - No global variable shadowing within functions
//!
//! ## Function Creation Patterns
//!
//! ### Simple Function (No Parameters)
//! ```rust
//! use lamina::ir::{IRBuilder, Type};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("hello", Type::Void)
//!     .print(lamina::ir::builder::string("Hello, World!"))
//!     .ret_void();
//! ```
//!
//! ### Function with Parameters
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function_with_params("add", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "a",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         },
//!         lamina::ir::FunctionParameter {
//!             name: "b",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "result", PrimitiveType::I32, var("a"), var("b"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//! ```
//!
//! ### Function with Control Flow
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, CmpOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("max", Type::Primitive(PrimitiveType::I32))
//!     .cmp(CmpOp::Gt, "a_greater", PrimitiveType::I32, var("a"), var("b"))
//!     .branch(var("a_greater"), "return_a", "return_b")
//!     .block("return_a")
//!     .ret(Type::Primitive(PrimitiveType::I32), var("a"))
//!     .block("return_b")
//!     .ret(Type::Primitive(PrimitiveType::I32), var("b"));
//! ```
//!
//! ## Function Calling Convention
//!
//! ### Parameter Passing
//! - Parameters are passed by value
//! - Complex types (structs, arrays) are passed by reference
//! - Variable argument functions are supported
//!
//! ### Return Value Handling
//! - Simple types returned in registers
//! - Complex types returned via implicit reference parameter
//! - Void functions have no return value
//!
//! ### Stack Frame Management
//! - Functions automatically manage their stack frames
//! - Local variables allocated on stack
//! - Stack frame cleaned up on function return
//!
//! ## Advanced Features
//!
//! ### Exception Handling
//! ```rust
//! # use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! // Functions can have exception handling blocks
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("safe_divide", Type::Primitive(PrimitiveType::I32))
//!     .block("try");
//!     // ... division code ...
//! builder.block("catch");
//!     // ... exception handling ...
//! ```
//!
//! ### Tail Recursion
//! ```rust
//! # use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! // Functions can be optimized for tail recursion
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("factorial", Type::Primitive(PrimitiveType::I32));
//!     // ... tail recursive implementation ...
//! ```
//!
//! ## Examples
//!
//! ### Complete Function Example
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, CmpOp, BinaryOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function_with_params("absolute_value", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "x",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!     .annotate(lamina::ir::FunctionAnnotation::Inline)
//!     .cmp(CmpOp::Lt, "is_negative", PrimitiveType::I32, var("x"), i32(0))
//!     .branch(var("is_negative"), "negate", "return_x")
//!     .block("negate")
//!     .binary(BinaryOp::Sub, "result", PrimitiveType::I32, i32(0), var("x"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"))
//!     .block("return_x")
//!     .ret(Type::Primitive(PrimitiveType::I32), var("x"));
//!
//! let module = builder.build();
//! ```

use std::collections::HashMap; // Using HashMap for basic blocks
use std::fmt;

use super::instruction::Instruction;
use super::types::{Identifier, Label, Type};

/// Function annotations that provide metadata about function behavior.
///
/// Annotations are hints to the compiler about how to handle a function.
/// They can affect optimization decisions, calling conventions, and code generation.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, FunctionAnnotation};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("fast_calc", Type::Void)
///     .annotate(FunctionAnnotation::Inline)  // Hint to inline this function
///     .annotate(FunctionAnnotation::Cold);   // Function is rarely called
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAnnotation {
    /// Hint to the compiler to inline this function at call sites.
    ///
    /// Inlining can improve performance by eliminating function call overhead,
    /// but may increase code size. Use for small, frequently called functions.
    Inline,

    /// Mark this function as exported (visible to other modules).
    ///
    /// Exported functions can be called from other modules or linked against
    /// by external code. This is typically used for public APIs.
    Export,

    /// Mark this function as having external linkage (imported from another module).
    ///
    /// External functions are declarations without implementation. They represent
    /// functions defined in other modules or external libraries.
    Extern,

    /// Indicate that this function never returns normally.
    ///
    /// Functions marked with `NoReturn` always terminate the program (e.g., `exit`, `panic`).
    /// This allows the compiler to optimize control flow and eliminate unreachable code.
    NoReturn,

    /// Hint to the compiler to never inline this function.
    ///
    /// Use for large functions or when you want to preserve the function call
    /// for debugging or profiling purposes.
    NoInline,

    /// Mark this function as "cold" (rarely executed).
    ///
    /// Cold functions are optimized for size rather than speed, and may be
    /// placed in a separate code section to improve instruction cache usage.
    Cold,

    /// Mark this function as "hot" (frequently executed).
    ///
    /// Hot functions are aggressively optimized for performance and may be
    /// placed in hot code sections for better instruction cache usage.
    Hot,

    /// Mark this function as pure (has no side effects and depends only on inputs).
    ///
    /// Pure functions can be optimized more aggressively and may be subject to
    /// common subexpression elimination, memoization, or other optimizations.
    Pure,

    /// Mark this function as const (can be evaluated at compile time).
    ///
    /// Const functions have no side effects and can be evaluated during compilation.
    /// This enables compile-time function evaluation and constant folding.
    Const,

    /// Mark this function as having internal linkage (private to this module).
    ///
    /// Internal functions are not visible outside the current module and cannot
    /// be called from other modules. This allows more aggressive optimizations.
    Internal,

    /// Mark this function as having private linkage (ELF-specific).
    ///
    /// Private symbols are not exported in the dynamic symbol table and are
    /// only visible within the same shared object. This is stronger than internal linkage.
    Private,

    /// Mark this function as having hidden visibility (ELF-specific).
    ///
    /// Hidden symbols are not exported but may be accessed from other components
    /// within the same shared object. This provides better optimization opportunities
    /// than protected visibility.
    Hidden,

    /// Mark this function as having protected visibility (ELF-specific).
    ///
    /// Protected symbols are exported but can only be preempted by symbols
    /// from the same shared object. This allows intra-module function calls to
    /// use direct references while still allowing inter-module calls.
    Protected,

    /// Mark this function as having weak linkage.
    ///
    /// Weak symbols can be overridden by stronger definitions. If multiple
    /// definitions exist, the strongest one is used.
    Weak,

    /// Use the C calling convention (system default).
    ///
    /// The C calling convention is the most common and portable convention,
    /// used for interfacing with C libraries and system calls.
    CCc,

    /// Use the fastcall calling convention (first few arguments in registers).
    ///
    /// Fastcall passes the first few arguments in registers rather than on the stack,
    /// which can improve performance for functions with few arguments.
    CCfast,

    /// Use the cold calling convention (function is rarely called).
    ///
    /// Cold calling convention optimizes for the case where the function is rarely executed,
    /// potentially using slower but more compact calling sequences.
    CCcold,

    /// Use the preserve_most calling convention (preserves most registers).
    ///
    /// This convention preserves most registers across the call, requiring the caller
    /// to save fewer registers. Useful for functions that call many other functions.
    CCpreserveMost,

    /// Use the preserve_all calling convention (preserves all registers).
    ///
    /// This convention preserves all registers across the call, minimizing the caller's
    /// register save/restore overhead. Useful for leaf functions or hot paths.
    CCpreserveAll,

    /// Use the swift calling convention (for Swift interoperability).
    ///
    /// Swift calling convention is used for interfacing with Swift code,
    /// with special handling for Swift types and error propagation.
    CCswift,

    /// Use the tail calling convention (enables tail call optimization).
    ///
    /// Tail calling convention allows the compiler to optimize tail recursive calls
    /// and function calls in tail position into jumps instead of calls.
    CCtail,

    /// Specify a custom calling convention by name.
    ///
    /// For calling conventions not covered by the above variants, you can specify
    /// a custom convention by name (e.g., "vectorcall", "thiscall", etc.).
    CallingConvention(String),

    /// Specify that this function should be placed in a specific section.
    ///
    /// Section placement can affect memory layout, performance, and linking.
    /// Common sections include ".text", ".text.hot", ".text.cold", etc.
    Section(String),

    /// Specify the minimum alignment for this function in bytes.
    ///
    /// Function alignment can improve performance by ensuring the function
    /// starts at an address that's optimal for the target architecture.
    Align(u32),

    /// Mark this function as unsafe (bypasses safety checks).
    ///
    /// Unsafe functions may perform operations that violate memory safety
    /// or other invariants. Use with caution and only when necessary.
    Unsafe,

    /// Mark this function as deprecated (should not be used).
    ///
    /// Deprecated functions may be removed in future versions. The compiler
    /// may emit warnings when these functions are called.
    Deprecated(Option<String>),
}

/// Attributes that can be applied to function parameters and variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariableAnnotation {
    /// Mark this parameter as non-null (never accepts null pointers).
    ///
    /// This allows the compiler to optimize away null checks and enables
    /// more aggressive optimizations for pointer operations.
    NonNull,

    /// Mark this parameter as read-only (not modified by the function).
    ///
    /// Read-only parameters can be passed by reference without copying,
    /// and the compiler can optimize based on the immutability guarantee.
    ReadOnly,

    /// Mark this parameter as write-only (only written to, never read from).
    ///
    /// Write-only parameters are typically used for output parameters.
    /// The compiler can optimize away reads from these parameters.
    WriteOnly,

    /// Specify the alignment requirement for this parameter in bytes.
    ///
    /// Parameter alignment affects how the parameter is passed and stored.
    /// This can improve performance on some architectures.
    Align(u32),

    /// Mark this parameter as sensitive (contains security-critical data).
    ///
    /// Sensitive parameters may receive special treatment such as avoiding
    /// storage in registers or ensuring they are zeroed after use.
    Sensitive,

    /// Specify that this parameter should be passed in a specific register.
    ///
    /// Register assignment can improve performance but reduces flexibility.
    /// Only use when you know the target ABI well.
    Register(String),

    /// Mark this parameter as unused (not used by the function).
    ///
    /// Unused parameters may be optimized away or may indicate API compatibility.
    Unused,
}

/// Represents a function parameter with its name, type, and attributes.
///
/// Function parameters are the inputs to a function and are bound to values
/// when the function is called. Each parameter has a unique name within the
/// function scope, a specific type, and optional attributes that provide
/// additional information to the compiler.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{FunctionParameter, VariableAnnotation, Type, PrimitiveType};
///
/// let param = FunctionParameter {
///     name: "data",
///     ty: Type::Primitive(PrimitiveType::Ptr),
///     annotations: vec![VariableAnnotation::NonNull, VariableAnnotation::ReadOnly],
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParameter<'a> {
    /// The parameter name (without the `%` prefix)
    pub name: Identifier<'a>,
    /// The parameter's type
    pub ty: Type<'a>,
    /// Optional attributes that provide additional information about the parameter
    pub annotations: Vec<VariableAnnotation>,
}

/// Represents the signature of a function (parameters and return type).
///
/// A function signature defines the interface of a function, including what
/// parameters it accepts and what type it returns. This is used for type
/// checking and to ensure function calls are correct.
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{FunctionSignature, FunctionParameter, Type, PrimitiveType};
///
/// let signature = FunctionSignature {
///     params: vec![
///         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) },
///         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32) },
///     ],
///     return_type: Type::Primitive(PrimitiveType::I32),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature<'a> {
    /// List of function parameters in order
    pub params: Vec<FunctionParameter<'a>>,
    /// The return type of the function (`Type::Void` if it doesn't return)
    pub return_type: Type<'a>, // Type::Void if it doesn't return
}

/// Represents a basic block within a function.
///
/// A basic block is a sequence of instructions that execute sequentially
/// and end with a terminator instruction (branch, jump, or return). This
/// structure enables control flow analysis and optimization.
///
/// # Invariants
///
/// - The last instruction MUST be a terminator (`Br`, `Jmp`, or `Ret`)
/// - Instructions within a block execute in order
/// - Control flow only changes at block boundaries
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("example", Type::Void)
///     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
///     .print(var("sum"))
///     .ret_void();
///
/// let module = builder.build();
/// let func = &module.functions["example"];
/// let entry_block = &func.basic_blocks["entry"];
/// // entry_block.instructions contains: [Binary, Print, Ret]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock<'a> {
    /// Instructions in this block (must end with a terminator)
    pub instructions: Vec<Instruction<'a>>,
    // Note: The last instruction MUST be a terminator (Br, Jmp, Ret)
}

/// Represents a complete function definition.
///
/// A function contains its signature, implementation (basic blocks), and
/// metadata (annotations). Functions are the primary unit of code organization
/// in the IR.
///
/// # Structure
///
/// - **Name**: Unique identifier for the function
/// - **Signature**: Parameters and return type
/// - **Basic Blocks**: The actual implementation
/// - **Entry Block**: The first block to execute
/// - **Annotations**: Metadata affecting behavior
///
/// # Examples
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("add", Type::Primitive(PrimitiveType::I32))
///     .binary(BinaryOp::Add, "result", PrimitiveType::I32, i32(10), i32(20))
///     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
///
/// let module = builder.build();
/// let func = &module.functions["add"];
/// // func contains the complete function definition
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Function<'a> {
    /// The function name (without the `@` prefix)
    pub name: Identifier<'a>,
    /// The function's signature (parameters and return type)
    pub signature: FunctionSignature<'a>,
    /// Annotations that affect function behavior
    pub annotations: Vec<FunctionAnnotation>,
    /// Basic blocks in this function, keyed by their label
    ///
    /// Using a HashMap allows efficient lookup by label name. The order
    /// of blocks is implicitly defined by the control flow (branches and jumps).
    pub basic_blocks: HashMap<Label<'a>, BasicBlock<'a>>,
    /// Label of the first block to execute when the function is called
    pub entry_block: Label<'a>, // Label of the first block to execute
}

// --- Display Implementations ---

impl fmt::Display for FunctionAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionAnnotation::Inline => write!(f, "@inline"),
            FunctionAnnotation::Export => write!(f, "@export"),
            FunctionAnnotation::Extern => write!(f, "@extern"),
            FunctionAnnotation::NoReturn => write!(f, "@noreturn"),
            FunctionAnnotation::NoInline => write!(f, "@noinline"),
            FunctionAnnotation::Cold => write!(f, "@cold"),
            FunctionAnnotation::Hot => write!(f, "@hot"),
            FunctionAnnotation::Pure => write!(f, "@pure"),
            FunctionAnnotation::Const => write!(f, "@const"),
            FunctionAnnotation::Internal => write!(f, "@internal"),
            FunctionAnnotation::Private => write!(f, "@private"),
            FunctionAnnotation::Hidden => write!(f, "@hidden"),
            FunctionAnnotation::Protected => write!(f, "@protected"),
            FunctionAnnotation::Weak => write!(f, "@weak"),
            FunctionAnnotation::CCc => write!(f, "@cc_c"),
            FunctionAnnotation::CCfast => write!(f, "@cc_fast"),
            FunctionAnnotation::CCcold => write!(f, "@cc_cold"),
            FunctionAnnotation::CCpreserveMost => write!(f, "@cc_preserve_most"),
            FunctionAnnotation::CCpreserveAll => write!(f, "@cc_preserve_all"),
            FunctionAnnotation::CCswift => write!(f, "@cc_swift"),
            FunctionAnnotation::CCtail => write!(f, "@cc_tail"),
            FunctionAnnotation::CallingConvention(cc) => write!(f, "@calling_convention({})", cc),
            FunctionAnnotation::Section(section) => write!(f, "@section({})", section),
            FunctionAnnotation::Align(alignment) => write!(f, "@align({})", alignment),
            FunctionAnnotation::Unsafe => write!(f, "@unsafe"),
            FunctionAnnotation::Deprecated(None) => write!(f, "@deprecated"),
            FunctionAnnotation::Deprecated(Some(msg)) => write!(f, "@deprecated({})", msg),
        }
    }
}

impl fmt::Display for VariableAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariableAnnotation::NonNull => write!(f, "nonnull"),
            VariableAnnotation::ReadOnly => write!(f, "readonly"),
            VariableAnnotation::WriteOnly => write!(f, "writeonly"),
            VariableAnnotation::Align(alignment) => write!(f, "align({})", alignment),
            VariableAnnotation::Sensitive => write!(f, "sensitive"),
            VariableAnnotation::Register(reg) => write!(f, "register({})", reg),
            VariableAnnotation::Unused => write!(f, "unused"),
        }
    }
}

impl fmt::Display for FunctionParameter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.annotations.is_empty() {
            write!(f, "{} %{}", self.ty, self.name)
        } else {
            write!(f, "{} ", self.ty)?;
            for (i, annotation) in self.annotations.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", annotation)?;
            }
            write!(f, " %{}", self.name)
        }
    }
}

impl fmt::Display for FunctionSignature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.params.iter().enumerate() {
            write!(f, "{}", param)?;
            if i < self.params.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ") -> {}", self.return_type)
    }
}

impl fmt::Display for BasicBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instr in &self.instructions {
            writeln!(f, "  {}", instr)?;
        }
        Ok(())
    }
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for annotation in &self.annotations {
            writeln!(f, "{}", annotation)?;
        }
        writeln!(f, "fn @{}{} {{", self.name, self.signature)?;

        // We need a consistent order for printing blocks. Let's try to print entry first.
        if let Some(entry) = self.basic_blocks.get(self.entry_block) {
            writeln!(f, "{}:", self.entry_block)?;
            write!(f, "{}", entry)?;
        }

        // Print remaining blocks (order might not be source order, but good enough for display)
        for (label, block) in &self.basic_blocks {
            if *label != self.entry_block {
                writeln!(f, "{}:", label)?;
                write!(f, "{}", block)?;
            }
        }

        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instruction::{AllocType, BinaryOp, Instruction};
    use crate::ir::types::{Literal, PrimitiveType, Type, Value};
    use std::collections::HashMap;

    #[test]
    fn test_display_function_annotation() {
        assert_eq!(format!("{}", FunctionAnnotation::Inline), "@inline");
        assert_eq!(format!("{}", FunctionAnnotation::Export), "@export");
        assert_eq!(format!("{}", FunctionAnnotation::NoReturn), "@noreturn");
        assert_eq!(format!("{}", FunctionAnnotation::NoInline), "@noinline");
        assert_eq!(format!("{}", FunctionAnnotation::Cold), "@cold");
    }

    #[test]
    fn test_display_function_parameter() {
        let param = FunctionParameter {
            name: "count",
            ty: Type::Primitive(PrimitiveType::I64),
            annotations: vec![],
        };
        assert_eq!(format!("{}", param), "i64 %count");
    }

    #[test]
    fn test_display_function_signature() {
        // No params, void return
        let sig1 = FunctionSignature {
            params: vec![],
            return_type: Type::Void,
        };
        assert_eq!(format!("{}", sig1), "() -> void");

        // One param, primitive return
        let sig2 = FunctionSignature {
            params: vec![FunctionParameter {
                name: "input",
                ty: Type::Primitive(PrimitiveType::F32),
                annotations: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::F32),
        };
        assert_eq!(format!("{}", sig2), "(f32 %input) -> f32");

        // Multiple params, named return
        let sig3 = FunctionSignature {
            params: vec![
                FunctionParameter {
                    name: "a",
                    ty: Type::Primitive(PrimitiveType::I32),
                    annotations: vec![],
                },
                FunctionParameter {
                    name: "b",
                    ty: Type::Named("MyType"),
                    annotations: vec![],
                },
            ],
            return_type: Type::Named("ResultType"),
        };
        assert_eq!(format!("{}", sig3), "(i32 %a, @MyType %b) -> @ResultType");
    }

    #[test]
    fn test_display_basic_block() {
        let block = BasicBlock {
            instructions: vec![
                Instruction::Alloc {
                    result: "p",
                    alloc_type: AllocType::Stack,
                    allocated_ty: Type::Primitive(PrimitiveType::I32),
                },
                Instruction::Store {
                    ty: Type::Primitive(PrimitiveType::I32),
                    ptr: Value::Variable("p"),
                    value: Value::Constant(Literal::I32(10)),
                },
                Instruction::Jmp {
                    target_label: "next_block",
                },
            ],
        };
        // Expect leading spaces on ALL instruction lines due to writeln!
        let expected_output = "  %p = alloc.ptr.stack i32\n  store.i32 %p, 10\n  jmp next_block\n";
        assert_eq!(format!("{}", block), expected_output);
    }

    #[test]
    fn test_display_function() {
        let sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "x",
                ty: Type::Primitive(PrimitiveType::I32),
                annotations: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };

        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "tmp",
                        ty: PrimitiveType::I32,
                        lhs: Value::Variable("x"),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Ret {
                        ty: Type::Primitive(PrimitiveType::I32),
                        value: Some(Value::Variable("tmp")),
                    },
                ],
            },
        );

        let func = Function {
            name: "increment",
            signature: sig,
            annotations: vec![FunctionAnnotation::Inline, FunctionAnnotation::Export],
            basic_blocks: blocks,
            entry_block: "entry",
        };

        // Match actual output - no indentation for instructions
        let expected_output = "@inline\n@export\nfn @increment(i32 %x) -> i32 {\nentry:\n  %tmp = add.i32 %x, 1\n  ret.i32 %tmp\n}\n";

        assert_eq!(format!("{}", func), expected_output);
    }

    #[test]
    fn test_display_function_multiple_blocks() {
        let sig = FunctionSignature {
            params: vec![FunctionParameter {
                name: "cond",
                ty: Type::Primitive(PrimitiveType::Bool),
                annotations: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::I32),
        };

        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Br {
                    condition: Value::Variable("cond"),
                    true_label: "if_true",
                    false_label: "if_false",
                }],
            },
        );
        blocks.insert(
            "if_true",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(1))),
                }],
            },
        );
        blocks.insert(
            "if_false",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(0))),
                }],
            },
        );

        let func = Function {
            name: "branch_test",
            signature: sig,
            annotations: vec![],
            basic_blocks: blocks,
            entry_block: "entry",
        };

        // Match actual output - no indentation for instructions
        let output = format!("{}", func);

        let expected_order1 = "fn @branch_test(bool %cond) -> i32 {\nentry:\n  br %cond, if_true, if_false\nif_false:\n  ret.i32 0\nif_true:\n  ret.i32 1\n}\n";

        let expected_order2 = "fn @branch_test(bool %cond) -> i32 {\n  entry:\n    br %cond, if_true, if_false\n  if_true:\n    ret.i32 1\n  if_false:\n    ret.i32 0\n}\n";
        let expected_order3 = "fn @branch_test(bool %cond) -> i32 {\nentry:\n  br %cond, if_true, if_false\nif_true:\n  ret.i32 1\nif_false:\n  ret.i32 0\n}\n";
        let expected_order4 = "fn @branch_test(bool %cond) -> i32 {\n  entry:\n    br %cond, if_true, if_false\n  if_true:\n    ret.i32 1\n  if_false:\n    ret.i32 0\n}\n";

        assert!(
            output == expected_order1
                || output == expected_order2
                || output == expected_order3
                || output == expected_order4,
            "Unexpected output:\n{}\nExpected one of:\n{}\nOR\n{}\n",
            output,
            expected_order1,
            expected_order2
        );
    }
}
