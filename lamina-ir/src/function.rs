//! # Function Representation
//!
//! Structures for representing functions in the Lamina IR.
//! Functions are the primary unit of code organization and execution, containing
//! basic blocks which hold the actual instructions.
//!
//! ## Key Concepts
//!
//! - **Functions**: Complete function definitions with signatures and implementations
//! - **Basic Blocks**: Linear sequences of instructions ending with a terminator
//! - **Function Signatures**: Parameter types and return types
//! - **Annotations**: Metadata that affects function behavior and optimization
//! - **Named Values**: Typed names; a name may be reassigned, but not to a different type
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
//! # use lamina_ir::{Type, PrimitiveType};
//! # use lamina_ir::function::{FunctionSignature, FunctionParameter};
//! let signature = FunctionSignature {
//!     params: vec![
//!         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
//!         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] }
//!     ],
//!     return_type: Type::Primitive(PrimitiveType::I32),
//!     variadic: false,
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
//! ## Variable Management
//!
//! ### Typed Reassignment
//! - A name may be assigned more than once in a function
//! - Every assignment to a name uses the type of its first definition
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
//! - Values from different blocks can be merged by reassigning a name on each path
//! - No global variable shadowing within functions
//!
//! ## Function Creation Patterns
//!
//! ### Simple Function (No Parameters)
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType};
//! use lamina_ir::builder::i32;
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("hello", Type::Void)
//!     .write_byte(i32(52), "w")
//!     .ret_void();
//! ```
//!
//! ### Function with Parameters
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina_ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//! builder
//!     .function_with_params("add", vec![
//!         lamina_ir::FunctionParameter {
//!             name: "a",
//!             ty: Type::Primitive(PrimitiveType::I32),
//!             annotations: vec![]
//!         },
//!         lamina_ir::FunctionParameter {
//!             name: "b",
//!             ty: Type::Primitive(PrimitiveType::I32),
//!             annotations: vec![]
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!     .binary(BinaryOp::Add, "result", PrimitiveType::I32, var("a"), var("b"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//! ```
//!
//! ### Function with Control Flow
//! ```rust
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, CmpOp};
//! use lamina_ir::builder::{var, i32};
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
//! ## Advanced
//!
//! ### Exception Handling
//! ```rust
//! # use lamina_ir::{IRBuilder, Type, PrimitiveType};
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
//! # use lamina_ir::{IRBuilder, Type, PrimitiveType};
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
//! use lamina_ir::{IRBuilder, Type, PrimitiveType, CmpOp, BinaryOp};
//! use lamina_ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function_with_params("absolute_value", vec![
//!         lamina_ir::FunctionParameter {
//!             name: "x",
//!             ty: Type::Primitive(PrimitiveType::I32),
//!             annotations: vec![]
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!     .annotate(lamina_ir::FunctionAnnotation::Inline)
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

use std::{collections::HashMap, fmt, iter};

pub use lamina_platform::SourceLoc;

use crate::{
    instruction::Instruction,
    types::{Identifier, Label, Type},
};

/// Function annotations that provide metadata about function behavior.
///
/// Annotations are hints to the compiler about how to handle a function.
/// They can affect optimization decisions, calling conventions, and code generation.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{IRBuilder, Type, FunctionAnnotation};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("fast_calc", Type::Void)
///     .annotate(FunctionAnnotation::Inline)  // Hint to inline this function
///     .annotate(FunctionAnnotation::Cold);   // Function is rarely called
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAnnotation {
    Kernel,
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
    /// This lets the compiler optimize control flow and eliminate unreachable code.
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
    /// placed in hot code sections to improve instruction cache usage.
    Hot,

    /// Mark this function as pure (has no side effects and depends only on inputs).
    ///
    /// Pure functions can be optimized more aggressively and may be subject to
    /// common subexpression elimination, memoization, or other optimizations.
    Pure,

    /// Mark this function as having internal linkage (private to this module).
    ///
    /// Internal functions are not visible outside the current module and cannot
    /// be called from other modules. This enables more aggressive optimizations.
    Internal,

    /// Mark this function as having private linkage (ELF-specific).
    ///
    /// Private symbols are not exported in the dynamic symbol table and are
    /// only visible within the same shared object. This is stronger than internal linkage.
    Private,

    /// Mark this function as having hidden visibility (ELF-specific).
    ///
    /// Hidden symbols are not exported but may be accessed from other components
    /// within the same shared object. This improves optimization opportunities
    /// than protected visibility.
    Hidden,

    /// Mark this function as having protected visibility (ELF-specific).
    ///
    /// Protected symbols are exported but can only be preempted by symbols
    /// from the same shared object. This lets intra-module function calls
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
    /// Tail calling convention lets the compiler optimize tail recursive calls
    /// and function calls in tail position into jumps instead of calls.
    CCtail,

    /// Specify a custom calling convention by name.
    ///
    /// For calling conventions not covered by the above variants, you can specify
    /// a custom convention by name (e.g., "vectorcall", "thiscall", etc.).
    CallingConvention(String),

    /// Place this function in the named section, written `@section("name")`.
    ///
    /// Section placement can affect memory layout, performance, and linking.
    /// Common sections include ".text", ".text.hot", ".text.cold", etc. Honored on
    /// ELF targets. At most one section annotation is allowed per function.
    Section(String),

    /// Align this function's entry to the given power-of-two byte boundary, written
    /// `@align(n)`.
    ///
    /// Function alignment can improve performance by making the function
    /// starts at an address that works well for the target architecture.
    /// At most one alignment annotation is allowed per function.
    Align(u32),

    /// Emit this function with no prologue or epilogue, written `@naked`.
    ///
    /// The body is exactly the inline assembly the author wrote, so an interrupt entry
    /// point can save and restore its own state. Only a `void` function with no
    /// parameters qualifies, and its body must be inline assembly with no operands,
    /// ending in `unreachable`. Any other instruction is rejected, since it would
    /// expect a frame.
    Naked,

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
    /// This lets the compiler optimize away null checks and enables
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
    /// storage in registers or zeroing them after use.
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

/// A function parameter with its name, type, and attributes.
///
/// Function parameters are the inputs to a function and are bound to values
/// when the function is called. Each parameter has a unique name within the
/// function scope, a specific type, and optional attributes that provide
/// additional information to the compiler.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::function::FunctionParameter;
/// use lamina_ir::{Type, PrimitiveType};
/// use lamina_ir::function::VariableAnnotation;
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

/// The signature of a function (parameters and return type).
///
/// A function signature defines the interface of a function, including what
/// parameters it accepts and what type it returns. This is used for type
/// checking and to ensure function calls are correct.
///
/// # Examples
///
/// ```rust
/// use lamina_ir::{FunctionSignature, FunctionParameter, Type, PrimitiveType};
///
/// let signature = FunctionSignature {
///     params: vec![
///         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
///         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
///     ],
///     return_type: Type::Primitive(PrimitiveType::I32),
///     variadic: false,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature<'a> {
    /// List of function parameters in order
    pub params: Vec<FunctionParameter<'a>>,
    /// The return type of the function (`Type::Void` if it doesn't return)
    pub return_type: Type<'a>, // Type::Void if it doesn't return
    /// Set by a trailing `...`, which only an `@extern` declaration may carry.
    pub variadic: bool,
}

/// A basic block within a function.
///
/// A basic block is a sequence of instructions that execute sequentially
/// and end with a terminator instruction (branch, jump, or return). This
/// structure supports control flow analysis and optimization.
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
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina_ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("example", Type::Void)
///     .binary(BinaryOp::Add, "sum", PrimitiveType::I32, i32(10), i32(20))
///     .write_byte(var("sum"), "w")
///     .ret_void();
///
/// let module = builder.build();
/// let func = &module.functions["example"];
/// let entry_block = &func.basic_blocks["entry"];
/// // entry_block.instructions contains: [Binary, WriteByte, Ret]
/// ```
#[derive(Debug, Clone)]
pub struct BasicBlock<'a> {
    /// Instructions in this block (must end with a terminator)
    pub instructions: Vec<Instruction<'a>>,
    /// Source position of each instruction, parallel to `instructions`. A shorter
    /// vector, or a `None` entry, means the position is unknown.
    pub instruction_locs: Vec<Option<SourceLoc>>,
}

/// Debug positions are metadata, so two blocks holding the same instructions are equal
/// whether or not their positions were recorded.
impl PartialEq for BasicBlock<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.instructions == other.instructions
    }
}

impl<'a> BasicBlock<'a> {
    pub fn new(instructions: Vec<Instruction<'a>>) -> Self {
        Self {
            instructions,
            instruction_locs: Vec::new(),
        }
    }

    /// The source position of instruction `index`, if one was recorded.
    pub fn loc_at(&self, index: usize) -> Option<SourceLoc> {
        self.instruction_locs.get(index).copied().flatten()
    }
}

/// A complete function definition.
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
/// use lamina_ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina_ir::builder::{var, i32};
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
    /// Using a HashMap enables efficient lookup by label name. The order
    /// of blocks is implicitly defined by the control flow (branches and jumps).
    pub basic_blocks: HashMap<Label<'a>, BasicBlock<'a>>,
    /// Label of the first block to execute when the function is called
    pub entry_block: Label<'a>,
}

// ---------------------------------------------------------------------------
// IR Validation
// ---------------------------------------------------------------------------

impl<'a> Function<'a> {
    /// Validate this function's IR for well-formedness.
    ///
    /// Checks performed:
    /// - Entry block exists among the function's basic blocks.
    /// - Every branch target (in `Br`, `Jmp`, `Switch`) names a block that
    ///   exists in this function.
    /// - Every block ends with a terminator (`Br`, `Jmp`, or `Ret`).
    /// - Return instructions carry a value when the return type is non-void,
    ///   and are bare when the return type is void.
    /// - Every read of a variable is assigned on every path from the entry block to
    ///   it. Parameters are assigned on entry. Reads in unreachable blocks see only
    ///   parameters and definitions earlier in the same block.
    ///
    /// Returns `Ok(())` when the function is valid, or a list of diagnostic
    /// strings describing every violation found.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        if self.annotations.contains(&FunctionAnnotation::Extern) && self.basic_blocks.is_empty() {
            return Ok(());
        }

        if !self.basic_blocks.contains_key(self.entry_block) {
            errors.push(format!(
                "entry block '{}' does not exist in function '@{}'",
                self.entry_block, self.name
            ));
        }

        for (label, block) in &self.basic_blocks {
            let last = block.instructions.last();
            let has_terminator = last.is_some_and(Instruction::is_terminator);
            if !has_terminator {
                errors.push(format!(
                    "block '{}' in '@{}' has no terminator instruction",
                    label, self.name
                ));
            }

            for instr in &block.instructions {
                match instr {
                    Instruction::Br {
                        true_label,
                        false_label,
                        ..
                    } => {
                        if !self.basic_blocks.contains_key(true_label) {
                            errors.push(format!(
                                "branch true target '{}' in block '{}' of '@{}' not found",
                                true_label, label, self.name
                            ));
                        }
                        if !self.basic_blocks.contains_key(false_label) {
                            errors.push(format!(
                                "branch false target '{}' in block '{}' of '@{}' not found",
                                false_label, label, self.name
                            ));
                        }
                    }
                    Instruction::Jmp { target_label } => {
                        if !self.basic_blocks.contains_key(target_label) {
                            errors.push(format!(
                                "jump target '{}' in block '{}' of '@{}' not found",
                                target_label, label, self.name
                            ));
                        }
                    }
                    Instruction::Switch { default, cases, .. } => {
                        if !self.basic_blocks.contains_key(default) {
                            errors.push(format!(
                                "switch default target '{}' in block '{}' of '@{}' not found",
                                default, label, self.name
                            ));
                        }
                        for (i, (value, _)) in cases.iter().enumerate() {
                            if cases[..i].iter().any(|(earlier, _)| earlier == value) {
                                errors.push(format!(
                                    "switch in block '{}' of '@{}' repeats case value {}",
                                    label, self.name, value
                                ));
                            }
                        }
                        for (_, case_label) in cases {
                            if !self.basic_blocks.contains_key(case_label) {
                                errors.push(format!(
                                    "switch case target '{}' in block '{}' of '@{}' not found",
                                    case_label, label, self.name
                                ));
                            }
                        }
                    }
                    Instruction::Ret { ty, value } => {
                        // The written type is checked against the signature in
                        // `lamina_ir::calls`, which can resolve a named type.
                        let ret_is_void = matches!(ty, Type::Void);
                        if !ret_is_void && value.is_none() {
                            errors.push(format!(
                                "non-void return in block '{}' of '@{}' has no value",
                                label, self.name
                            ));
                        }
                        if ret_is_void && value.is_some() {
                            errors.push(format!(
                                "void return in block '{}' of '@{}' carries a value",
                                label, self.name
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        self.check_definite_assignment(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn check_definite_assignment(&self, errors: &mut Vec<String>) {
        let mut labels: Vec<Label<'a>> = self.basic_blocks.keys().copied().collect();
        labels.sort_unstable();
        let index: HashMap<Label<'a>, usize> =
            labels.iter().enumerate().map(|(i, l)| (*l, i)).collect();
        let Some(&entry) = index.get(self.entry_block) else {
            return;
        };

        let mut var_index: HashMap<&'a str, usize> = HashMap::new();
        for name in self
            .signature
            .params
            .iter()
            .map(|p| p.name)
            .chain(labels.iter().flat_map(|label| {
                self.basic_blocks[label]
                    .instructions
                    .iter()
                    .filter_map(Instruction::defined_var)
            }))
        {
            let next = var_index.len();
            var_index.entry(name).or_insert(next);
        }
        let words = var_index.len().div_ceil(64);
        let top = vec![u64::MAX; words];
        let mut params = vec![0u64; words];
        for p in &self.signature.params {
            set_bit(&mut params, var_index[p.name]);
        }

        let n = labels.len();
        let succs: Vec<Vec<usize>> = labels
            .iter()
            .map(|label| {
                terminator_targets(&self.basic_blocks[label])
                    .into_iter()
                    .filter_map(|t| index.get(t).copied())
                    .collect()
            })
            .collect();
        let mut reachable = vec![false; n];
        reachable[entry] = true;
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            for &s in &succs[b] {
                if !reachable[s] {
                    reachable[s] = true;
                    stack.push(s);
                }
            }
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in (0..n).filter(|&b| reachable[b]) {
            for &s in &succs[b] {
                preds[s].push(b);
            }
        }

        let block_defs: Vec<Vec<u64>> = labels
            .iter()
            .map(|label| {
                let mut defs = vec![0u64; words];
                for instr in &self.basic_blocks[label].instructions {
                    if let Some(def) = instr.defined_var() {
                        set_bit(&mut defs, var_index[def]);
                    }
                }
                defs
            })
            .collect();

        let mut ins: Vec<Vec<u64>> = vec![top.clone(); n];
        let mut outs: Vec<Vec<u64>> = (0..n).map(|b| or_words(&top, &block_defs[b])).collect();
        ins[entry] = params.clone();
        outs[entry] = or_words(&params, &block_defs[entry]);
        loop {
            let mut changed = false;
            for b in (0..n).filter(|&b| reachable[b]) {
                if b != entry {
                    let mut new_in = top.clone();
                    for &p in &preds[b] {
                        and_assign(&mut new_in, &outs[p]);
                    }
                    if new_in != ins[b] {
                        ins[b] = new_in;
                        changed = true;
                    }
                }
                let new_out = or_words(&ins[b], &block_defs[b]);
                if new_out != outs[b] {
                    outs[b] = new_out;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let assigned = |set: &[u64], name: &str| {
            var_index
                .get(name)
                .is_some_and(|&i| set[i / 64] >> (i % 64) & 1 == 1)
        };

        for (b, label) in labels.iter().enumerate() {
            let block = &self.basic_blocks[label];
            let mut cur = if reachable[b] {
                ins[b].clone()
            } else {
                params.clone()
            };
            for instr in &block.instructions {
                for used in instr.used_vars() {
                    if !assigned(&cur, used) {
                        errors.push(format!(
                            "use of '%{}' in block '{}' of '@{}' is not assigned on every path to it",
                            used, label, self.name
                        ));
                    }
                }
                if let Some(def) = instr.defined_var() {
                    set_bit(&mut cur, var_index[def]);
                }
            }
        }
    }

    /// Reports every redefinition whose type differs from the variable's first definition.
    ///
    /// A name may be reassigned with the same type. Lowering keys type facts by name, so a
    /// reassignment with another type would give later uses the wrong type. `callee_return`
    /// gives the return type of a called function, for call results.
    pub fn duplicate_definition_errors(
        &self,
        callee_return: &dyn Fn(&str) -> Option<Type<'a>>,
    ) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();
        let mut first_defs: HashMap<&str, (Option<Type<'a>>, String)> = self
            .signature
            .params
            .iter()
            .map(|p| {
                (
                    p.name,
                    (
                        Some(p.ty.clone()),
                        format!("the parameters of '@{}'", self.name),
                    ),
                )
            })
            .collect();
        let mut ordered_blocks: Vec<(&Label<'a>, &BasicBlock<'a>)> =
            self.basic_blocks.iter().collect();
        ordered_blocks.sort_by_key(|(label, _)| **label);
        for (label, block) in &ordered_blocks {
            for (index, instr) in block.instructions.iter().enumerate() {
                let Some(def) = instr.defined_var() else {
                    continue;
                };
                let ty = match instr {
                    Instruction::Call { func_name, .. } => callee_return(func_name),
                    other => other.result_type(),
                };
                match first_defs.get(def) {
                    Some((first_ty, first_site)) => {
                        if let (Some(first_ty), Some(ty)) = (first_ty, &ty)
                            && first_ty != ty
                        {
                            errors.push(format!(
                                "duplicate definition of '%{}' in '@{}': first defined as {} in {}, redefined as {} in block '{}' instruction {} (a value may be reassigned, but not with a different type)",
                                def, self.name, first_ty, first_site, ty, label, index + 1
                            ));
                        }
                    }
                    None => {
                        let site = format!("block '{}' instruction {}", label, index + 1);
                        first_defs.insert(def, (ty, site));
                    }
                }
            }
        }

        errors
    }
}

// --- Display Implementations ---

fn terminator_targets<'a>(block: &BasicBlock<'a>) -> Vec<Label<'a>> {
    match block.instructions.last() {
        Some(Instruction::Br {
            true_label,
            false_label,
            ..
        }) => vec![*true_label, *false_label],
        Some(Instruction::Jmp { target_label }) => vec![*target_label],
        Some(Instruction::Switch { default, cases, .. }) => iter::once(*default)
            .chain(cases.iter().map(|(_, label)| *label))
            .collect(),
        _ => Vec::new(),
    }
}

fn set_bit(set: &mut [u64], i: usize) {
    set[i / 64] |= 1 << (i % 64);
}

fn or_words(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(x, y)| x | y).collect()
}

fn and_assign(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x &= y;
    }
}

impl fmt::Display for FunctionAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionAnnotation::Kernel => write!(f, "@kernel"),
            FunctionAnnotation::Inline => write!(f, "@inline"),
            FunctionAnnotation::Export => write!(f, "@export"),
            FunctionAnnotation::Extern => write!(f, "@extern"),
            FunctionAnnotation::NoReturn => write!(f, "@noreturn"),
            FunctionAnnotation::NoInline => write!(f, "@noinline"),
            FunctionAnnotation::Cold => write!(f, "@cold"),
            FunctionAnnotation::Hot => write!(f, "@hot"),
            FunctionAnnotation::Pure => write!(f, "@pure"),
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
            FunctionAnnotation::CallingConvention(cc) => write!(f, "@calling_convention({cc})"),
            FunctionAnnotation::Section(section) => write!(f, "@section(\"{section}\")"),
            FunctionAnnotation::Align(alignment) => write!(f, "@align({alignment})"),
            FunctionAnnotation::Naked => write!(f, "@naked"),
            FunctionAnnotation::Unsafe => write!(f, "@unsafe"),
            FunctionAnnotation::Deprecated(None) => write!(f, "@deprecated"),
            FunctionAnnotation::Deprecated(Some(msg)) => write!(f, "@deprecated({msg})"),
        }
    }
}

impl fmt::Display for VariableAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariableAnnotation::NonNull => write!(f, "nonnull"),
            VariableAnnotation::ReadOnly => write!(f, "readonly"),
            VariableAnnotation::WriteOnly => write!(f, "writeonly"),
            VariableAnnotation::Align(alignment) => write!(f, "align({alignment})"),
            VariableAnnotation::Sensitive => write!(f, "sensitive"),
            VariableAnnotation::Register(reg) => write!(f, "register({reg})"),
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
                write!(f, "{annotation}")?;
            }
            write!(f, " %{}", self.name)
        }
    }
}

impl fmt::Display for FunctionSignature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.params.iter().enumerate() {
            write!(f, "{param}")?;
            if i < self.params.len() - 1 || self.variadic {
                write!(f, ", ")?;
            }
        }
        if self.variadic {
            write!(f, "...")?;
        }
        write!(f, ") -> {}", self.return_type)
    }
}

impl fmt::Display for BasicBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for instr in &self.instructions {
            writeln!(f, "  {instr}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Function<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for annotation in &self.annotations {
            writeln!(f, "{annotation}")?;
        }
        if self.annotations.contains(&FunctionAnnotation::Extern) && self.basic_blocks.is_empty() {
            return writeln!(f, "fn @{}{}", self.name, self.signature);
        }
        writeln!(f, "fn @{}{} {{", self.name, self.signature)?;

        if let Some(entry) = self.basic_blocks.get(self.entry_block) {
            writeln!(f, "{}:", self.entry_block)?;
            write!(f, "{entry}")?;
        }

        let mut labels: Vec<_> = self
            .basic_blocks
            .keys()
            .filter(|label| **label != self.entry_block)
            .collect();
        labels.sort_unstable();
        for label in labels {
            writeln!(f, "{label}:")?;
            write!(f, "{}", self.basic_blocks[label])?;
        }

        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction::{AllocType, BinaryOp, Instruction},
        types::{Literal, PrimitiveType, Type, Value},
    };

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
            variadic: false,
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
            variadic: false,
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
            variadic: false,
        };
        assert_eq!(format!("{}", sig3), "(i32 %a, !MyType %b) -> !ResultType");
    }

    #[test]
    fn test_display_basic_block() {
        let block = BasicBlock {
            instructions: vec![
                Instruction::Alloc {
                    result: "p",
                    alloc_type: AllocType::Stack,
                    allocated_ty: Type::Primitive(PrimitiveType::I32),
                    align: None,
                },
                Instruction::Store {
                    ty: Type::Primitive(PrimitiveType::I32),
                    ptr: Value::Variable("p"),
                    value: Value::Constant(Literal::I32(10)),
                    volatile: false,
                    align: None,
                },
                Instruction::Jmp {
                    target_label: "next_block",
                },
            ],
            instruction_locs: Vec::new(),
        };
        // Expect leading spaces on ALL instruction lines due to writeln!
        let expected_output = "  %p = stackalloc.i32\n  store.i32 %p, 10\n  jmp next_block\n";
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
            variadic: false,
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
                instruction_locs: Vec::new(),
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
            variadic: false,
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
                instruction_locs: Vec::new(),
            },
        );
        blocks.insert(
            "if_true",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(1))),
                }],
                instruction_locs: Vec::new(),
            },
        );
        blocks.insert(
            "if_false",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Constant(Literal::I32(0))),
                }],
                instruction_locs: Vec::new(),
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

    // ---------------------------------------------------------------------------
    // validate() tests
    // ---------------------------------------------------------------------------

    fn simple_func<'a>(
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
        blocks: HashMap<&'a str, BasicBlock<'a>>,
        entry: &'a str,
    ) -> Function<'a> {
        Function {
            name: "test",
            signature: FunctionSignature {
                params,
                return_type,
                variadic: false,
            },
            annotations: vec![],
            basic_blocks: blocks,
            entry_block: entry,
        }
    }

    fn binary_def<'a>(result: &'a str, ty: PrimitiveType) -> Instruction<'a> {
        Instruction::Binary {
            op: BinaryOp::Add,
            result,
            ty,
            lhs: Value::Constant(Literal::I64(1)),
            rhs: Value::Constant(Literal::I64(2)),
        }
    }

    fn ret_void<'a>() -> Instruction<'a> {
        Instruction::Ret {
            ty: Type::Void,
            value: None,
        }
    }

    #[test]
    fn test_same_type_reassignment_is_accepted() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    binary_def("x", PrimitiveType::I32),
                    binary_def("x", PrimitiveType::I32),
                    ret_void(),
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        assert!(f.duplicate_definition_errors(&|_| None).is_empty());
    }

    #[test]
    fn test_redefinition_with_different_type_is_rejected() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    binary_def("x", PrimitiveType::I128),
                    binary_def("x", PrimitiveType::I64),
                    ret_void(),
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.duplicate_definition_errors(&|_| None);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("'%x'"));
        assert!(errs[0].contains("first defined as i128 in block 'entry' instruction 1"));
        assert!(errs[0].contains("redefined as i64 in block 'entry' instruction 2"));
    }

    #[test]
    fn test_redefinition_of_parameter_with_different_type_is_rejected() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![binary_def("a", PrimitiveType::U64), ret_void()],
                instruction_locs: Vec::new(),
            },
        );
        let params = vec![FunctionParameter {
            name: "a",
            ty: Type::Primitive(PrimitiveType::I64),
            annotations: vec![],
        }];
        let f = simple_func(params, Type::Void, blocks, "entry");
        let errs = f.duplicate_definition_errors(&|_| None);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("'%a'"));
        assert!(errs[0].contains("first defined as i64 in the parameters of '@test'"));
    }

    #[test]
    fn test_call_result_redefinition_uses_callee_return_type() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Call {
                        result: Some("v"),
                        func_name: "neg",
                        args: vec![],
                    },
                    Instruction::Call {
                        result: Some("v"),
                        func_name: "big",
                        args: vec![],
                    },
                    ret_void(),
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let returns = |name: &str| match name {
            "neg" => Some(Type::Primitive(PrimitiveType::I64)),
            "big" => Some(Type::Primitive(PrimitiveType::U64)),
            _ => None,
        };
        let errs = f.duplicate_definition_errors(&returns);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("first defined as i64"));
        assert!(errs[0].contains("redefined as u64"));
    }

    #[test]
    fn test_validate_valid_void_function() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Void,
                    value: None,
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        assert!(f.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_entry_block() {
        let f = simple_func(vec![], Type::Void, HashMap::new(), "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("entry block")));
    }

    #[test]
    fn test_validate_missing_terminator() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Binary {
                    op: BinaryOp::Add,
                    result: "x",
                    ty: PrimitiveType::I32,
                    lhs: Value::Constant(Literal::I32(1)),
                    rhs: Value::Constant(Literal::I32(2)),
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("no terminator")));
    }

    #[test]
    fn test_validate_invalid_branch_target() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Jmp {
                    target_label: "nonexistent",
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("nonexistent")));
    }

    #[test]
    fn test_validate_undefined_variable() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "sum",
                        ty: PrimitiveType::I32,
                        lhs: Value::Variable("undefined_var"),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Ret {
                        ty: Type::Void,
                        value: None,
                    },
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("undefined_var")));
    }

    #[test]
    fn test_validate_ssa_define_then_use() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "sum",
                        ty: PrimitiveType::I32,
                        lhs: Value::Constant(Literal::I32(1)),
                        rhs: Value::Constant(Literal::I32(2)),
                    },
                    Instruction::Ret {
                        ty: Type::Primitive(PrimitiveType::I32),
                        value: Some(Value::Variable("sum")),
                    },
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Primitive(PrimitiveType::I32), blocks, "entry");
        assert!(f.validate().is_ok());
    }

    #[test]
    fn test_validate_value_used_in_later_block_is_accepted() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "sum",
                        ty: PrimitiveType::I32,
                        lhs: Value::Constant(Literal::I32(1)),
                        rhs: Value::Constant(Literal::I32(2)),
                    },
                    Instruction::Jmp {
                        target_label: "next",
                    },
                ],
                instruction_locs: Vec::new(),
            },
        );
        blocks.insert(
            "next",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I32),
                    value: Some(Value::Variable("sum")),
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Primitive(PrimitiveType::I32), blocks, "entry");
        assert!(f.validate().is_ok());
    }

    #[test]
    fn test_validate_use_before_definition_in_same_block_fails() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "early",
                        ty: PrimitiveType::I32,
                        lhs: Value::Variable("late"),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Binary {
                        op: BinaryOp::Add,
                        result: "late",
                        ty: PrimitiveType::I32,
                        lhs: Value::Constant(Literal::I32(1)),
                        rhs: Value::Constant(Literal::I32(1)),
                    },
                    Instruction::Ret {
                        ty: Type::Void,
                        value: None,
                    },
                ],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'%late'")));
    }

    #[test]
    fn test_validate_duplicate_switch_case_fails() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Switch {
                    ty: PrimitiveType::I32,
                    value: Value::Constant(Literal::I32(1)),
                    default: "done",
                    cases: vec![(Literal::I32(1), "done"), (Literal::I32(1), "done")],
                }],
                instruction_locs: Vec::new(),
            },
        );
        blocks.insert(
            "done",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Void,
                    value: None,
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Void, blocks, "entry");
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("repeats case value")));
    }

    #[test]
    fn test_validate_pointer_return_for_integer_signature_passes() {
        let mut blocks = HashMap::new();
        blocks.insert(
            "entry",
            BasicBlock {
                instructions: vec![Instruction::Ret {
                    ty: Type::Primitive(PrimitiveType::I64),
                    value: Some(Value::Constant(Literal::I64(0))),
                }],
                instruction_locs: Vec::new(),
            },
        );
        let f = simple_func(vec![], Type::Primitive(PrimitiveType::Ptr), blocks, "entry");
        assert!(f.validate().is_ok());
    }

    fn da_add<'a>(result: &'a str, lhs: Value<'a>) -> Instruction<'a> {
        Instruction::Binary {
            op: BinaryOp::Add,
            result,
            ty: PrimitiveType::I32,
            lhs,
            rhs: Value::Constant(Literal::I32(1)),
        }
    }

    fn da_branch<'a>(true_label: &'a str, false_label: &'a str) -> Instruction<'a> {
        Instruction::Br {
            condition: Value::Constant(Literal::Bool(true)),
            true_label,
            false_label,
        }
    }

    fn da_jump<'a>(target_label: &'a str) -> Instruction<'a> {
        Instruction::Jmp { target_label }
    }

    fn da_ret<'a>(value: Value<'a>) -> Instruction<'a> {
        Instruction::Ret {
            ty: Type::Primitive(PrimitiveType::I32),
            value: Some(value),
        }
    }

    fn da_func<'a>(blocks: Vec<(&'a str, Vec<Instruction<'a>>)>) -> Function<'a> {
        let basic_blocks = blocks
            .into_iter()
            .map(|(label, instructions)| {
                (
                    label,
                    BasicBlock {
                        instructions,
                        instruction_locs: Vec::new(),
                    },
                )
            })
            .collect();
        simple_func(
            vec![],
            Type::Primitive(PrimitiveType::I32),
            basic_blocks,
            "entry",
        )
    }

    #[test]
    fn test_definite_assignment_one_arm_does_not_reach_join() {
        let f = da_func(vec![
            ("entry", vec![da_branch("a", "b")]),
            (
                "a",
                vec![da_add("x", Value::Constant(Literal::I32(1))), da_jump("j")],
            ),
            ("b", vec![da_jump("j")]),
            ("j", vec![da_ret(Value::Variable("x"))]),
        ]);
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'%x'") && e.contains("'j'")));
    }

    #[test]
    fn test_definite_assignment_both_arms_reach_join() {
        let f = da_func(vec![
            ("entry", vec![da_branch("a", "b")]),
            (
                "a",
                vec![da_add("x", Value::Constant(Literal::I32(1))), da_jump("j")],
            ),
            (
                "b",
                vec![da_add("x", Value::Constant(Literal::I32(2))), da_jump("j")],
            ),
            ("j", vec![da_ret(Value::Variable("x"))]),
        ]);
        assert_eq!(f.validate(), Ok(()));
    }

    #[test]
    fn test_definite_assignment_entry_backedge_does_not_initialize() {
        let f = da_func(vec![
            (
                "entry",
                vec![
                    da_add("y", Value::Variable("x")),
                    da_add("x", Value::Constant(Literal::I32(1))),
                    da_branch("entry", "exit"),
                ],
            ),
            ("exit", vec![da_ret(Value::Variable("y"))]),
        ]);
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'%x'")));
    }

    #[test]
    fn test_definite_assignment_loop_defined_value_read_on_first_iteration() {
        let f = da_func(vec![
            ("entry", vec![da_jump("h")]),
            (
                "h",
                vec![
                    da_add("y", Value::Variable("x")),
                    da_add("x", Value::Constant(Literal::I32(1))),
                    da_branch("h", "exit"),
                ],
            ),
            ("exit", vec![da_ret(Value::Variable("y"))]),
        ]);
        let errs = f.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'%x'")));
    }

    #[test]
    fn test_definite_assignment_copy_in_each_predecessor() {
        let copy = |value: Value<'static>| Instruction::Copy {
            result: "p",
            ty: Type::Primitive(PrimitiveType::I32),
            value,
        };
        let f = da_func(vec![
            ("entry", vec![da_branch("a", "b")]),
            (
                "a",
                vec![
                    da_add("x", Value::Constant(Literal::I32(1))),
                    copy(Value::Variable("x")),
                    da_jump("j"),
                ],
            ),
            (
                "b",
                vec![
                    da_add("z", Value::Constant(Literal::I32(2))),
                    copy(Value::Variable("z")),
                    da_jump("j"),
                ],
            ),
            ("j", vec![da_ret(Value::Variable("p"))]),
        ]);
        assert_eq!(f.validate(), Ok(()));
    }

    #[test]
    fn test_definite_assignment_copy_operand_must_be_assigned_on_its_edge() {
        let f = da_func(vec![
            ("entry", vec![da_branch("a", "b")]),
            (
                "a",
                vec![
                    Instruction::Copy {
                        result: "p",
                        ty: Type::Primitive(PrimitiveType::I32),
                        value: Value::Variable("z"),
                    },
                    da_jump("j"),
                ],
            ),
            (
                "b",
                vec![da_add("z", Value::Constant(Literal::I32(2))), da_jump("j")],
            ),
            ("j", vec![da_ret(Value::Variable("p"))]),
        ]);
        assert!(f.validate().is_err());
    }

    #[test]
    fn test_definite_assignment_unreachable_block_does_not_inherit_definitions() {
        let f = da_func(vec![
            (
                "entry",
                vec![
                    da_add("x", Value::Constant(Literal::I32(1))),
                    da_ret(Value::Variable("x")),
                ],
            ),
            (
                "dead",
                vec![
                    da_add("q", Value::Variable("x")),
                    da_ret(Value::Variable("q")),
                ],
            ),
        ]);
        let errs = f.validate().unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.contains("'%x'") && e.contains("'dead'"))
        );
    }
}
