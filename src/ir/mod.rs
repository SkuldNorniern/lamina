//! # Lamina Intermediate Representation (IR)
//!
//! The Lamina IR is a sophisticated, low-level, architecture-agnostic intermediate
//! representation that serves as the critical bridge between high-level programming
//! languages and machine code. It provides a clean, efficient, and extensible
//! representation that enables powerful optimizations and seamless code generation
//! across multiple target architectures.
//!
//! ## Design Philosophy
//!
//! Lamina IR is built on several fundamental design principles:
//!
//! ### Core Design Principles
//!
//! - **SSA Form**: Single Static Assignment ensures each variable is assigned exactly once,
//!   enabling powerful optimizations and simplifying analysis
//! - **Type Safety**: Comprehensive type system prevents many classes of runtime errors
//! - **Architecture Agnostic**: IR operations work identically across all supported platforms
//! - **Zero-Copy**: Uses string references to minimize memory allocations and copying
//! - **Extensible**: Easy to add new instruction types, types, and optimization passes
//! - **Optimization-First**: Designed specifically to enable advanced compiler optimizations
//!
//! ### Technical Foundations
//!
//! - **Memory Efficient**: Minimal runtime overhead with compact representations
//! - **Analysis Friendly**: Structure enables fast static analysis and optimization
//! - **Code Generation Ready**: Natural mapping to assembly instructions
//! - **Debuggable**: Rich metadata support for debugging and profiling
//!
//! ## Core Concepts Deep Dive
//!
//! ### Values and Types System
//!
//! The IR operates on **values** that have specific **types**. This type system is the
//! foundation of Lamina's safety and optimization capabilities.
//!
//! #### Value Types
//!
//! | Value Type | Syntax | Description | Example |
//! |------------|--------|-------------|---------|
//! | **Variables** | `%name` | SSA variables assigned exactly once | `%result`, `%temp` |
//! | **Constants** | `literal` | Immutable literal values | `42`, `true`, `"hello"` |
//! | **Globals** | `@name` | Module-level variables | `@message`, `@counter` |
//! | **Parameters** | `%param` | Function input parameters | `%input`, `%config` |
//!
//! #### Type System Architecture
//!
//! ```text
//! Type Hierarchy:
//! └── Type (enum)
//!     ├── Primitive (i8, i16, i32, i64, f32, f64, bool)
//!     ├── Composite
//!     │   ├── Struct (named fields with types)
//!     │   ├── Array (element_type + size)
//!     │   └── Tuple (anonymous ordered types)
//!     ├── Function (parameters + return type)
//!     └── Pointer (target type + address space)
//! ```
//!
//! #### Type Safety Guarantees
//!
//! - **Compile-time verification**: All operations are type-checked
//! - **Memory safety**: Pointer operations are bounds-checked where possible
//! - **Optimization enabling**: Types guide optimization decisions
//! - **Code generation**: Types determine appropriate assembly instructions
//!
//! ### Instruction Architecture
//!
//! Instructions are the atomic operations that perform computations and control flow.
//! Each instruction produces zero or more new values and may have side effects.
//!
//! #### Instruction Categories
//!
//! ##### Arithmetic Instructions
//! - `add`, `sub`, `mul`, `div` - Basic arithmetic operations
//! - `rem`, `neg` - Remainder and negation
//! - Type-specific variants: `add.i32`, `mul.i64`, etc.
//!
//! ##### Comparison Instructions
//! - `eq`, `ne`, `lt`, `le`, `gt`, `ge` - Ordering comparisons
//! - `cmp` - General comparison returning ordering
//! - Result type is always boolean or ordering enum
//!
//! ##### Memory Instructions
//! - `alloc_stack`, `alloc_heap` - Memory allocation
//! - `load`, `store` - Memory access operations
//! - `getelementptr` - Pointer arithmetic for arrays/structs
//! - `dealloc` - Memory deallocation
//!
//! ##### Control Flow Instructions
//! - `br` - Conditional branch based on boolean condition
//! - `jmp` - Unconditional jump to labeled block
//! - `call` - Function invocation with parameter passing
//! - `ret` - Return from function with optional value
//! - `phi` - SSA phi nodes for merging values from different paths
//!
//! ##### Type Conversion Instructions
//! - `zext` - Zero extension (smaller to larger integer)
//! - `sext` - Sign extension (smaller to larger integer)
//! - `trunc` - Truncation (larger to smaller integer)
//! - `bitcast` - Reinterpretation of bit patterns
//!
//! ##### I/O Instructions
//! - `write`, `read` - Buffer-based I/O operations
//! - `writebyte`, `readbyte` - Single byte I/O
//! - `writeptr` - Pointer-based I/O operations
//! - `print` - Debug output for development
//!
//! ### Functions and Basic Blocks Architecture
//!
//! #### Function Structure
//!
//! ```text
//! Function Layout:
//! └── Function
//!     ├── Signature (parameters + return type)
//!     ├── Entry Block (first block executed)
//!     ├── Basic Blocks (instruction sequences)
//!     │   ├── Block 1: instructions + terminator
//!     │   ├── Block 2: instructions + terminator
//!     │   └── Block N: instructions + terminator
//!     └── Annotations (metadata like @inline, @export)
//! ```
//!
//! #### Basic Block Properties
//!
//! **Definition**: A maximal sequence of instructions with:
//! - **Single entry point**: Only one way to enter the block
//! - **Single exit point**: Exactly one terminator instruction
//! - **No internal branches**: All control flow goes through terminator
//!
//! **Terminator Instructions Required**:
//! - `br condition, true_block, false_block` - Conditional branch
//! - `jmp target_block` - Unconditional jump
//! - `ret value` or `ret.void` - Return from function
//!
//! #### Control Flow Graph (CFG)
//!
//! Functions form a **Control Flow Graph** where:
//! - **Nodes**: Basic blocks containing instructions
//! - **Edges**: Control flow transfers between blocks
//! - **Entry**: Special entry block where execution begins
//! - **Exits**: Blocks ending with return instructions
//!
//! #### SSA Variable Management
//!
//! **Single Static Assignment Rules**:
//! 1. Each variable is defined exactly once
//! 2. Each use refers to exactly one definition
//! 3. Variable definitions dominate all uses
//!
//! **Phi Nodes for Merging**:
//! ```lamina
//! // Merging values from different control paths
//! %result = phi [%value1, "block1"], [%value2, "block2"]
//! ```
//!
//! ### Memory Management System
//!
//! #### Allocation Strategies
//!
//! ##### Stack Allocation (`alloc_stack`)
//! ```rust
//! # use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! // Automatic lifetime management
//! let mut builder = IRBuilder::new();
//! builder.alloc_stack("local", Type::Primitive(PrimitiveType::I32));
//! // Memory automatically freed when function returns
//! ```
//!
//! **Characteristics**:
//! - **Fast**: Minimal allocation overhead
//! - **Automatic**: No manual deallocation needed
//! - **Limited**: Function scope only
//! - **Efficient**: Reuses stack frames
//!
//! ##### Heap Allocation (`alloc_heap`)
//! ```rust
//! # use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! # use lamina::ir::builder::var;
//! // Manual lifetime management
//! let mut builder = IRBuilder::new();
//! builder.alloc_heap("dynamic", Type::Primitive(PrimitiveType::I32));
//! // Must explicitly deallocate when done
//! builder.dealloc(var("dynamic"));
//! ```
//!
//! **Characteristics**:
//! - **Persistent**: Survives function boundaries
//! - **Manual**: Requires explicit deallocation
//! - **Slower**: Heap allocation overhead
//! - **Flexible**: Can be passed between functions
//!
//! #### Memory Access Patterns
//!
//! ##### Direct Memory Access
//! ```lamina
//! %ptr = alloc_stack i32     # Allocate memory
//! store.i32 %ptr, 42         # Store value
//! %value = load.i32 %ptr     # Load value
//! ```
//!
//! ##### Pointer Arithmetic
//! ```lamina
//! %array = alloc_stack [10 x i32]  # Allocate array
//! %element_ptr = getelementptr %array, 5  # Get pointer to index 5
//! store.i32 %element_ptr, 99       # Store at array[5]
//! ```
//!
//! ##### Struct Field Access
//! ```lamina
//! %point = alloc_stack {i32, i32}   # Allocate struct
//! %x_ptr = struct_gep %point, 0    # Get pointer to x field
//! %y_ptr = struct_gep %point, 1    # Get pointer to y field
//! store.i32 %x_ptr, 10             # Set x coordinate
//! store.i32 %y_ptr, 20             # Set y coordinate
//! ```
//!
//! ## Module Organization Deep Dive
//!
//! The Lamina IR is organized as a modular system where each component serves a specific
//! purpose in the compilation pipeline. Understanding this organization is key to
//! effectively using and extending the IR.
//!
//! ### [`builder`] - Fluent IR Construction API
//!
//! The `IRBuilder` is the primary interface for programmatically constructing IR modules.
//! It provides a fluent, type-safe API that handles the complexity of managing functions,
//! basic blocks, and instruction sequences.
//!
//! #### Builder Pattern Usage
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp, CmpOp};
//! use lamina::ir::builder::{var, i32, string};
//!
//! let mut builder = IRBuilder::new();
//!
//! // Build a complete function with control flow
//! builder
//!     .function_with_params("conditional_add", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "a",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         },
//!         lamina::ir::FunctionParameter {
//!             name: "b",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Primitive(PrimitiveType::I32))
//!
//!     // Compare parameters
//!     .cmp(CmpOp::Gt, "a_greater", PrimitiveType::I32, var("a"), var("b"))
//!
//!     // Conditional logic
//!     .branch(var("a_greater"), "add_values", "subtract_values")
//!
//!     .block("add_values")
//!     .binary(BinaryOp::Add, "result", PrimitiveType::I32, var("a"), var("b"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"))
//!
//!     .block("subtract_values")
//!     .binary(BinaryOp::Sub, "result", PrimitiveType::I32, var("a"), var("b"))
//!     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
//!
//! // Build the final module
//! let module = builder.build();
//! ```
//!
//! #### Builder Capabilities
//!
//! - **Type-safe construction**: Compile-time type checking
//! - **Fluent API**: Chain operations naturally
//! - **Automatic SSA**: Manages variable assignments
//! - **Block management**: Handles basic block creation and transitions
//! - **Validation**: Ensures IR correctness during construction
//!
//! ### [`function`] - Function Definition and Control Flow
//!
//! The function module defines the structure of functions and their control flow.
//! Functions are the primary unit of code organization and execution.
//!
//! #### Function Lifecycle
//!
//! ```text
//! Function Creation → Basic Block Management → Instruction Addition → Validation
//!       ↓                    ↓                        ↓                ↓
//!    Signature        Entry Block Creation     SSA Variable      Control Flow
//!    Parameters       Terminator Handling      Management        Analysis
//!    Return Types     Block Transitions        Scope Rules       Optimization
//! ```
//!
//! #### Advanced Function Features
//!
//! ##### Function Annotations
//! ```rust
//! # use lamina::ir::{IRBuilder, Type};
//! # use lamina::ir::function::FunctionAnnotation;
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("optimized_func", Type::Void)
//!     .annotate(FunctionAnnotation::Inline)      // Suggest inlining
//!     .annotate(FunctionAnnotation::NoInline);   // Prevent inlining
//! ```
//!
//! ##### Tail Recursion Optimization
//! ```rust
//! # use lamina::ir::{IRBuilder, Type, PrimitiveType};
//! let mut builder = IRBuilder::new();
//! builder
//!     .function("factorial_tail", Type::Primitive(PrimitiveType::I64));
//!     // Implementation optimized for tail recursion
//! ```
//!
//! ### [`instruction`] - Complete Instruction Set
//!
//! The instruction module contains all atomic operations that can be performed.
//! Instructions are categorized by their primary function and effect.
//!
//! #### Instruction Set Overview
//!
//! | Category | Instructions | Purpose | Example |
//! |----------|-------------|---------|---------|
//! | **Arithmetic** | `add`, `sub`, `mul`, `div`, `rem` | Mathematical operations | `add.i32 %a, %b` |
//! | **Comparison** | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | Value comparisons | `eq.i32 %a, %b` |
//! | **Memory** | `load`, `store`, `alloc_stack`, `alloc_heap` | Memory management | `store.i32 %ptr, 42` |
//! | **Control Flow** | `br`, `jmp`, `call`, `ret`, `phi` | Program flow control | `br %cond, "true", "false"` |
//! | **Type Conversion** | `zext`, `sext`, `trunc`, `bitcast` | Type transformations | `zext.i32.i64 %val` |
//! | **I/O** | `write`, `read`, `print`, `writeptr` | Input/Output operations | `writeptr %ptr` |
//!
//! #### Instruction Execution Model
//!
//! **Atomic Operations**: Each instruction is:
//! - **Indivisible**: Cannot be interrupted or partially executed
//! - **Deterministic**: Same inputs always produce same outputs
//! - **Side-effect aware**: Memory and I/O effects are explicit
//! - **SSA compliant**: Produces new values, doesn't modify existing ones
//!
//! ### [`module`] - Top-Level IR Organization
//!
//! The module represents a complete, self-contained unit of IR that can be:
//! - **Compiled independently**
//! - **Linked with other modules**
//! - **Optimized as a unit**
//! - **Serialized and deserialized**
//!
//! #### Module Contents
//!
//! ```text
//! Module Structure:
//! ├── Functions (HashMap<&str, Function>)
//! │   ├── Entry points and internal functions
//! │   ├── Exported and private functions
//! │   └── Function annotations and metadata
//! ├── Type Declarations (HashMap<&str, TypeDeclaration>)
//! │   ├── Custom struct definitions
//! │   ├── Type aliases and renames
//! │   └── Forward declarations
//! ├── Global Variables (HashMap<&str, GlobalDeclaration>)
//! │   ├── Module-level constants
//! │   ├── Static data with module lifetime
//! │   └── External symbol references
//! └── Metadata
//!     ├── Module name and version
//!     ├── Dependencies and imports
//!     └── Compilation hints
//! ```
//!
//! #### Module Linking and Dependencies
//!
//! Modules can reference symbols from other modules:
//! ```rust
//! # use lamina::ir::{IRBuilder, Type};
//! # use lamina::ir::builder::var;
//! // Module A defines a function
//! let mut builder_a = IRBuilder::new();
//! builder_a.function("public_func", Type::Void);
//!
//! // Module B references it
//! let mut builder_b = IRBuilder::new();
//! builder_b.call(Some("result"), "public_func", vec![]);
//! ```
//!
//! ### [`types`] - Comprehensive Type System
//!
//! The type system is the foundation of Lamina's safety and optimization capabilities.
//! It provides both static guarantees and runtime efficiency.
//!
//! #### Type Hierarchy and Properties
//!
//! | Type Category | Examples | Memory Layout | Operations |
//! |---------------|----------|---------------|------------|
//! | **Primitives** | `i8`, `i32`, `f64`, `bool` | Fixed size | Arithmetic, comparison |
//! | **Pointers** | `i32*`, `void*` | Platform-dependent | Load, store, arithmetic |
//! | **Arrays** | `[10 x i32]` | Contiguous elements | Indexing, slicing |
//! | **Structs** | `{i32, f64}` | Field layout | Field access, construction |
//! | **Functions** | `(i32, i32) -> i32` | Function pointer | Call, return |
//!
//! #### Type System Features
//!
//! - **Memory Safety**: Bounds checking and null pointer prevention
//! - **Optimization**: Types guide code generation and optimization
//! - **Analysis**: Types enable static analysis and verification
//! - **Interoperability**: Compatible with C ABI and other systems
//!
//! ## Advanced Memory Management Patterns
//!
//! ### Memory Allocation Strategies Comparison
//!
//! | Feature | Stack Allocation | Heap Allocation |
//! |---------|------------------|-----------------|
//! | **Speed** | Fast (no syscall) | Slow (system call) |
//! | **Lifetime** | Automatic (function scope) | Manual (explicit dealloc) |
//! | **Size** | Limited (stack size) | Large (heap size) |
//! | **Safety** | Safe (automatic) | Manual (error-prone) |
//! | **Sharing** | Local only | Cross-function |
//! | **Overhead** | Minimal | Allocation headers |
//!
//! ### Advanced Memory Patterns
//!
//! #### Memory Pool Pattern
//! ```lamina
//! // Pre-allocate memory pool
//! %pool = alloc_stack [100 x i32]
//!
//! // Use pool for multiple allocations
//! %item1 = getelementptr %pool, 0   # pool[0]
//! %item2 = getelementptr %pool, 10  # pool[10]
//! %item3 = getelementptr %pool, 20  # pool[20]
//! ```
//!
//! #### Struct with Embedded Arrays
//! ```lamina
//! // Define struct with array field
//! %data = alloc_stack {i32, [5 x i32]}
//!
//! // Access scalar field
//! %count_ptr = struct_gep %data, 0
//! store.i32 %count_ptr, 5
//!
//! // Access array field
//! %array_ptr = struct_gep %data, 1
//! %element_ptr = getelementptr %array_ptr, 2
//! store.i32 %element_ptr, 42
//! ```
//!
//! ## Performance Optimization Features
//!
//! ### Compiler Optimization Opportunities
//!
//! #### SSA-Enabled Optimizations
//! - **Dead Code Elimination**: Remove unused variables and instructions
//! - **Constant Propagation**: Replace variables with known constants
//! - **Common Subexpression Elimination**: Reuse computed values
//! - **Strength Reduction**: Replace expensive operations with cheaper ones
//!
//! #### Memory Optimization
//! - **Stack Slot Coloring**: Reuse stack slots for non-overlapping variables
//! - **Heap Allocation Coalescing**: Combine small allocations
//! - **Memory Access Optimization**: Optimize load/store patterns
//!
//! #### Control Flow Optimization
//! - **Block Merging**: Combine compatible basic blocks
//! - **Jump Threading**: Simplify conditional branches
//! - **Loop Unrolling**: Expand small loops for better ILP
//! - **Tail Call Optimization**: Optimize recursive functions
//!
//! ### Performance Characteristics
//!
//! #### Memory Efficiency
//! - **Zero-copy parsing**: String references avoid allocations
//! - **Compact representation**: Minimal memory overhead for IR structures
//! - **Efficient lookups**: HashMap-based symbol tables with O(1) access
//! - **Cache-friendly layout**: Contiguous memory for instruction sequences
//!
//! #### Analysis Performance
//! - **Fast traversal**: Linear passes over instruction sequences
//! - **Incremental updates**: Local changes don't require global recomputation
//! - **Parallel analysis**: Independent analyses can run concurrently
//! - **Scalable design**: Performance scales with program size
//!
//! ## Thread Safety and Concurrency
//!
//! ### IR Structure Thread Safety
//!
//! #### Immutable by Default
//! ```rust
//! # use lamina::ir::IRBuilder;
//! // IR structures are typically built once and read many times
//! let mut builder = IRBuilder::new();
//! let module = builder.build(); // Immutable after construction
//!
//! // Safe to share across threads
//! std::thread::spawn(move || {
//!     // analyze_module(&module); // Read-only access
//! });
//! ```
//!
//! #### Safe Concurrent Access Patterns
//! - **Analysis threads**: Multiple analysis passes can run simultaneously
//! - **Code generation**: Different targets can generate code in parallel
//! - **Optimization passes**: Independent optimizations can run concurrently
//! - **Module linking**: Multiple modules can be linked in parallel
//!
//! #### Memory Safety Guarantees
//! - **Rust ownership system**: Prevents data races at compile time
//! - **Immutable sharing**: `Arc` and `&` for safe sharing
//! - **No internal mutability**: No `RefCell` or `Mutex` in core structures
//! - **Lifetime management**: Compile-time prevention of use-after-free
//!
//! ## Practical Examples and Use Cases
//!
//! ### Complete Program Example
//!
//! ```lamina
//! // Fibonacci calculator with memory management
//! fn @fibonacci(i32 %n) -> i32 {
//!   entry:
//!     %is_base = cmp.lt.i32 %n, 2
//!     br %is_base, base_case, recursive_case
//!
//!   base_case:
//!     ret.i32 1
//!
//!   recursive_case:
//!     // Allocate memory for intermediate results
//!     %temp = alloc_stack i32
//!
//!     // Calculate fib(n-1)
//!     %n_minus_1 = sub.i32 %n, 1
//!     %fib_n_minus_1 = call @fibonacci(%n_minus_1)
//!
//!     // Calculate fib(n-2)
//!     %n_minus_2 = sub.i32 %n, 2
//!     %fib_n_minus_2 = call @fibonacci(%n_minus_2)
//!
//!     // Sum the results
//!     %result = add.i32 %fib_n_minus_1, %fib_n_minus_2
//!     ret.i32 %result
//! }
//!
//! fn @main() -> void {
//!   entry:
//!     // Calculate fib(10)
//!     %result = call @fibonacci(10)
//!
//!     // Print the result
//!     print %result
//!     ret.void
//! }
//! ```
//!
//! ### Memory Management Example
//!
//! ```rust
//! use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
//! use lamina::ir::builder::{var, i32};
//!
//! let mut builder = IRBuilder::new();
//!
//! builder
//!     .function_with_params("process_data", vec![
//!         lamina::ir::FunctionParameter {
//!             name: "size",
//!             ty: Type::Primitive(PrimitiveType::I32)
//!         }
//!     ], Type::Void)
//!
//!     // Allocate dynamic array on heap
//!     .alloc_heap("array", Type::Array {
//!         element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
//!         size: 100
//!     })
//!
//!     // Process each element
//!     .binary(BinaryOp::Mul, "scaled_size", PrimitiveType::I32, var("size"), i32(2))
//!
//!     // Use the array (simplified)
//!     .print(var("scaled_size"))
//!
//!     // Always clean up heap allocations
//!     .dealloc(var("array"))
//!
//!     .ret_void();
//!
//! let module = builder.build();
//! ```
//!
//! ## Future Extensions and Roadmap
//!
//! ### Planned Features
//!
//! #### Advanced Type System
//! - **Generic types**: Parametric polymorphism
//! - **Union types**: Sum types for different representations
//! - **Trait system**: Interface-based polymorphism
//! - **Associated types**: Type families and relationships
//!
//! #### Enhanced Optimization
//! - **Interprocedural analysis**: Cross-function optimizations
//! - **Profile-guided optimization**: Runtime feedback integration
//! - **Vectorization**: SIMD instruction generation
//! - **Memory optimization**: Advanced alias analysis
//!
//! #### Extended Instruction Set
//! - **Atomic operations**: Thread-safe memory operations
//! - **Vector instructions**: SIMD arithmetic and operations
//! - **Exception handling**: Try/catch semantics
//! - **Coroutine support**: Asynchronous programming primitives
//!
//! #### Advanced Features
//! - **Debug information**: Source-level debugging support
//! - **Metadata annotations**: Rich program metadata
//! - **Serialization**: Binary IR format for fast loading
//! - **Incremental compilation**: Partial recompilation support
//!
//! ### Extensibility Design
//!
//! The IR is designed to be highly extensible:
//!
//! - **New instruction types**: Easy to add through the instruction enum
//! - **Custom optimization passes**: Plugin architecture for analysis and transformation
//! - **Target-specific backends**: Clean separation of architecture-specific code
//! - **User-defined types**: Runtime type system extension
//!
//! This modular design ensures that Lamina IR can evolve with new language features,
//! optimization techniques, and target architectures while maintaining backward compatibility.
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
#[cfg(feature = "nightly")]
pub use instruction::{AtomicBinOp, MemoryOrdering, SimdOp};
#[cfg(feature = "nightly")]
pub use module::ModuleAnnotation;
pub use module::{GlobalDeclaration, Module, TypeDeclaration};
pub use types::{Identifier, Literal, PrimitiveType, StructField, Type, Value};
