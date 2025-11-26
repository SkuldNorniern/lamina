use std::collections::HashMap;

use super::function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
    VariableAnnotation,
};
use super::instruction::{AllocType, BinaryOp, CmpOp, Instruction};
#[cfg(feature = "nightly")]
use super::instruction::{AtomicBinOp, MemoryOrdering, SimdOp};
use super::module::Module;
#[cfg(feature = "nightly")]
use super::module::ModuleAnnotation;
use super::types::{Literal, PrimitiveType, Type, Value};

/// # IR Builder
///
/// A fluent API for programmatically constructing Lamina IR modules.
///
/// ## Overview
///
/// The `IRBuilder` allows you to construct IR code in a safe, programmatic way without
/// having to manually build instruction objects. It provides methods for all IR operations
/// and maintains the context of the current function and basic block.
///
/// ## Basic Usage Pattern
///
/// 1. Create a builder with `IRBuilder::new()`
/// 2. Define a function with `function()` or `function_with_params()`
/// 3. Add basic blocks with `block()`
/// 4. Add instructions to the current block
/// 5. Generate the final module with `build()`
///
/// ## Example
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, FunctionParameter, BinaryOp};
/// use lamina::ir::builder::var;
///
/// let mut builder = IRBuilder::new();
///
/// // Create a function that adds two numbers
/// builder
///     .function_with_params(
///         "add",
///         vec![
///             FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
///             FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
///         ],
///         Type::Primitive(PrimitiveType::I32)
///     )
///     .binary(
///         BinaryOp::Add,
///         "result",
///         PrimitiveType::I32,
///         var("a"),
///         var("b")
///     )
///     .ret(Type::Primitive(PrimitiveType::I32), var("result"));
///
/// let module = builder.build();
/// ```
///
/// ## Memory Management in Lamina IR
///
/// Lamina provides comprehensive memory management capabilities through its IR builder.
/// Understanding these operations is crucial for efficient and safe code generation.
///
/// ### Memory Allocation Strategies
///
/// **Stack Allocation (`alloc_stack`)**:
/// - Fast allocation/deallocation (automatic)
/// - Limited lifetime (function scope)
/// - Ideal for temporary variables and local data
/// - No manual deallocation needed
///
/// **Heap Allocation (`alloc_heap`)**:
/// - Persistent across function calls
/// - Must be manually deallocated with `dealloc()`
/// - Suitable for data that outlives the current function
/// - Enables dynamic data structures
///
/// ### Memory Access Patterns
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("memory_demo", Type::Void)
///     // Allocate on stack
///     .alloc_stack("local_var", Type::Primitive(PrimitiveType::I32))
///     // Store a value
///     .store(Type::Primitive(PrimitiveType::I32), var("local_var"), i32(42))
///     // Load and use it
///     .load("loaded_val", Type::Primitive(PrimitiveType::I32), var("local_var"))
///     .print(var("loaded_val")) // Debug output
///     .ret_void();
/// ```
///
/// ### Pointer Operations
///
/// Lamina supports sophisticated pointer arithmetic for arrays and structures:
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("pointer_demo", Type::Void)
///     // Allocate an array
///     .alloc_stack("arr", Type::Array { element_type: Box::new(Type::Primitive(PrimitiveType::I32)), size: 10 })
///     // Get pointer to element at index 5
///     .getelementptr("elem_ptr", var("arr"), i32(5), PrimitiveType::I32)
///     // Store value at that location
///     .store(Type::Primitive(PrimitiveType::I32), var("elem_ptr"), i32(100))
///     .ret_void();
/// ```
///
/// ### Heap Memory Management
///
/// Always pair heap allocations with deallocations:
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("heap_demo", Type::Void)
///     // Allocate on heap
///     .alloc_heap("heap_data", Type::Primitive(PrimitiveType::I32))
///     // Use the allocated memory
///     .store(Type::Primitive(PrimitiveType::I32), var("heap_data"), i32(999))
///     // Always deallocate when done
///     .dealloc(var("heap_data"))
///     .ret_void();
/// ```
///
/// ### Debugging with Print
///
/// The `print()` instruction is invaluable for debugging IR code:
///
/// ```rust
/// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
/// use lamina::ir::builder::{var, i32, bool, string};
///
/// let mut builder = IRBuilder::new();
/// builder
///     .function("debug_demo", Type::Void)
///     .print(i32(42))           // Print integer
///     .print(bool(true))        // Print boolean
///     .print(string("hello"))   // Print string
///     .print(var("my_var"))     // Print variable value
///     .ret_void();
/// ```
///
/// ### Best Practices
///
/// 1. **Prefer stack allocation** when possible for better performance
/// 2. **Always deallocate heap memory** to prevent memory leaks
/// 3. **Use print() liberally** during development for debugging
/// 4. **Validate pointer arithmetic** to avoid out-of-bounds access
/// 5. **Consider data layout** when using structs and arrays
///
/// ## Method Organization
///
/// The builder methods are organized into several categories:
///
/// ### **Memory Management**
/// - **Allocation**: `alloc_stack()`, `alloc_heap()` - Allocate memory on stack or heap
/// - **Access**: `load()`, `store()` - Read from and write to memory locations
/// - **Deallocation**: `dealloc()` - Free heap-allocated memory
/// - **Pointer arithmetic**: `getelementptr()`, `struct_gep()` - Calculate element/field addresses
///
/// ### **Function Definition**
/// - `function()`, `function_with_params()`, `annotate()` - Define functions and their properties
///
/// ### **Control Flow**
/// - **Blocks**: `block()`, `set_entry_block()` - Manage basic blocks
/// - **Branching**: `branch()`, `jump()`, `ret()`, `ret_void()` - Control flow operations
///
/// ### **Data Operations**
/// - **Arithmetic**: `binary()`, `cmp()` - Mathematical and comparison operations
/// - **Type conversion**: `zext()` - Zero-extension for type widening
/// - **Composite types**: `tuple()`, `extract_tuple()` - Tuple creation and access
///
/// ### **Function Calls**
/// - `call()` - Call other functions
///
/// ### **Debugging & I/O**
/// - `print()` - Debug output for development and testing
///
/// ### **Advanced Features**
/// - **SSA form**: `phi()` - Merge values from different control flow paths
/// - **Helper methods**: `temp_var()` - Generate unique temporary variable names
pub struct IRBuilder<'a> {
    module: Module<'a>,
    current_function: Option<&'a str>,
    current_block: Option<&'a str>,
    block_instructions: HashMap<&'a str, Vec<Instruction<'a>>>,
    function_blocks: HashMap<&'a str, HashMap<&'a str, BasicBlock<'a>>>,
    function_signatures: HashMap<&'a str, FunctionSignature<'a>>,
    function_annotations: HashMap<&'a str, Vec<FunctionAnnotation>>,
    function_entry_blocks: HashMap<&'a str, &'a str>,
    temp_var_counter: usize,
}

impl Default for IRBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IRBuilder<'a> {
    /// Creates a new empty IR builder
    ///
    /// Initializes a builder with no functions or blocks.
    pub fn new() -> Self {
        IRBuilder {
            module: Module::new(),
            current_function: None,
            current_block: None,
            block_instructions: HashMap::new(),
            function_blocks: HashMap::new(),
            function_signatures: HashMap::new(),
            function_annotations: HashMap::new(),
            function_entry_blocks: HashMap::new(),
            temp_var_counter: 0,
        }
    }

    /// Generates a unique temporary variable name
    ///
    /// Returns: A fresh variable name in the format "temp_N" where N is a counter
    ///
    /// Use this when you need a variable but don't care about its specific name.
    /// Each call returns a new name that won't conflict with previous ones.
    pub fn temp_var(&mut self) -> String {
        let var = format!("temp_{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        var
    }

    /// Creates a new function with no parameters
    ///
    /// Parameters:
    /// - `name`: The function name (without @ prefix)
    /// - `return_type`: The function's return type
    ///
    /// This method:
    /// 1. Creates a new function with the given name and return type
    /// 2. Sets this function as the current function context
    /// 3. Creates a default "entry" block
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder.function("main", Type::Void);
    /// ```
    pub fn function(&mut self, name: &'a str, return_type: Type<'a>) -> &mut Self {
        self.function_with_params(name, vec![], return_type)
    }

    /// Creates a new function with parameters
    ///
    /// Parameters:
    /// - `name`: The function name (without @ prefix)
    /// - `params`: Vector of function parameters
    /// - `return_type`: The function's return type
    ///
    /// This is the full version of the function creation method, allowing
    /// specification of parameters. Uses the builder pattern to chain method calls.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, FunctionParameter};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder.function_with_params(
    ///     "add",
    ///     vec![
    ///         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
    ///         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] },
    ///     ],
    ///     Type::Primitive(PrimitiveType::I32)
    /// );
    /// ```
    pub fn function_with_params(
        &mut self,
        name: &'a str,
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
    ) -> &mut Self {
        let signature = FunctionSignature {
            params,
            return_type,
        };

        self.function_signatures.insert(name, signature);
        self.function_blocks.insert(name, HashMap::new());
        self.function_annotations.insert(name, vec![]);
        self.current_function = Some(name);

        // Default entry block
        self.block("entry");
        self.function_entry_blocks.insert(name, "entry");

        self
    }

    /// Creates a function parameter with optional annotations
    ///
    /// Parameters:
    /// - `name`: Parameter name (without % prefix)
    /// - `ty`: Parameter type
    /// - `annotations`: Optional vector of parameter annotations
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::function::VariableAnnotation;
    ///
    /// let mut builder = IRBuilder::new();
    /// let param = builder.param("data", Type::Primitive(PrimitiveType::Ptr), vec![VariableAnnotation::NonNull]);
    /// ```
    pub fn param(
        &mut self,
        name: &'a str,
        ty: Type<'a>,
        annotations: Vec<VariableAnnotation>,
    ) -> FunctionParameter<'a> {
        FunctionParameter {
            name,
            ty,
            annotations,
        }
    }

    /// Creates a simple function parameter without annotations
    ///
    /// Parameters:
    /// - `name`: Parameter name (without % prefix)
    /// - `ty`: Parameter type
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    ///
    /// let mut builder = IRBuilder::new();
    /// let param = builder.param_simple("x", Type::Primitive(PrimitiveType::I32));
    /// ```
    pub fn param_simple(&mut self, name: &'a str, ty: Type<'a>) -> FunctionParameter<'a> {
        self.param(name, ty, vec![])
    }

    /// Adds an annotation to the current function
    ///
    /// Parameters:
    /// - `annotation`: The function annotation to add (e.g., FunctionAnnotation::Inline)
    ///
    /// Function annotations provide metadata about the function.
    /// Must be called after creating a function and before moving to a new function.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, FunctionAnnotation};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("fast_func", Type::Void)
    ///     .annotate(FunctionAnnotation::Inline);
    /// ```
    pub fn annotate(&mut self, annotation: FunctionAnnotation) -> &mut Self {
        if let Some(func_name) = self.current_function
            && let Some(annotations) = self.function_annotations.get_mut(func_name)
        {
            annotations.push(annotation);
        }
        self
    }

    /// Marks the current function as inline
    pub fn inline(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Inline)
    }

    /// Marks the current function as exported
    pub fn export(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Export)
    }

    /// Marks the current function as external (imported)
    pub fn external(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Extern)
    }

    /// Marks the current function as having no return
    pub fn no_return(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::NoReturn)
    }

    /// Marks the current function as cold (rarely executed)
    pub fn cold(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Cold)
    }

    /// Marks the current function as hot (frequently executed)
    pub fn hot(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hot)
    }

    /// Marks the current function as pure (no side effects)
    pub fn pure(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Pure)
    }

    /// Marks the current function as const (compile-time evaluable)
    pub fn const_fn(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Const)
    }

    /// Marks the current function as internal (private to module)
    pub fn internal(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Internal)
    }

    /// Marks the current function as having private linkage (ELF-specific)
    pub fn private(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Private)
    }

    /// Marks the current function as having hidden visibility (ELF-specific)
    pub fn hidden(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hidden)
    }

    /// Marks the current function as having protected visibility (ELF-specific)
    pub fn protected(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Protected)
    }

    /// Marks the current function as unsafe
    pub fn unsafe_fn(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Unsafe)
    }

    /// Sets the C calling convention for the current function (system default).
    pub fn cc_c(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCc)
    }

    /// Sets the fastcall calling convention for the current function.
    pub fn cc_fast(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCfast)
    }

    /// Sets the cold calling convention for the current function.
    pub fn cc_cold(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCcold)
    }

    /// Sets the preserve_most calling convention for the current function.
    pub fn cc_preserve_most(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCpreserveMost)
    }

    /// Sets the preserve_all calling convention for the current function.
    pub fn cc_preserve_all(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCpreserveAll)
    }

    /// Sets the swift calling convention for the current function.
    pub fn cc_swift(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCswift)
    }

    /// Sets the tail calling convention for the current function.
    pub fn cc_tail(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCtail)
    }

    /// Sets a custom calling convention for the current function.
    pub fn calling_convention(&mut self, cc: &str) -> &mut Self {
        self.annotate(FunctionAnnotation::CallingConvention(cc.to_string()))
    }

    /// Sets the section for the current function
    pub fn section(&mut self, section: &str) -> &mut Self {
        self.annotate(FunctionAnnotation::Section(section.to_string()))
    }

    /// Sets the alignment for the current function
    pub fn align(&mut self, alignment: u32) -> &mut Self {
        self.annotate(FunctionAnnotation::Align(alignment))
    }

    /// Annotates the module with a global attribute.
    ///
    /// Module annotations affect how the entire compilation unit is treated.
    /// They control optimization, code generation, and linking behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lamina::ir::{IRBuilder, ModuleAnnotation};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .annotate_module(ModuleAnnotation::PositionIndependentCode)
    ///     .annotate_module(ModuleAnnotation::OptimizeForSpeed);
    /// ```
    #[cfg(feature = "nightly")]
    pub fn annotate_module(&mut self, annotation: ModuleAnnotation) -> &mut Self {
        self.module.annotations.push(annotation);
        self
    }

    /// Enables position-independent code generation for this module.
    ///
    /// PIC allows the code to be loaded at any address in memory, which is
    /// required for shared libraries and improves security through ASLR.
    #[cfg(feature = "nightly")]
    pub fn pic(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentCode)
    }

    /// Enables position-independent executable generation for this module.
    ///
    /// PIE creates executables that can be loaded at random addresses,
    /// providing additional security benefits.
    #[cfg(feature = "nightly")]
    pub fn pie(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentExecutable)
    }

    /// Optimizes this module for execution speed.
    ///
    /// This may increase code size but should improve runtime performance.
    #[cfg(feature = "nightly")]
    pub fn optimize_for_speed(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::OptimizeForSpeed)
    }

    /// Optimizes this module for code size.
    ///
    /// This may reduce performance but will create smaller binaries.
    #[cfg(feature = "nightly")]
    pub fn optimize_for_size(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::OptimizeForSize)
    }

    /// Includes debug information in the compiled output.
    ///
    /// Debug information allows for better debugging and profiling
    /// but increases the size of the final binary.
    #[cfg(feature = "nightly")]
    pub fn include_debug_info(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::IncludeDebugInfo)
    }

    /// Strips debug information and symbols from the compiled output.
    ///
    /// This reduces binary size and removes potentially sensitive information
    /// but makes debugging impossible.
    #[cfg(feature = "nightly")]
    pub fn strip_symbols(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::StripSymbols)
    }

    /// Specifies the target triple for this module.
    ///
    /// The target triple identifies the architecture, vendor, OS, and ABI
    /// for which the code should be compiled (e.g., "x86_64-unknown-linux-gnu").
    #[cfg(feature = "nightly")]
    pub fn target_triple(&mut self, triple: &str) -> &mut Self {
        self.annotate_module(ModuleAnnotation::TargetTriple(triple.to_string()))
    }

    /// Creates a new basic block in the current function
    ///
    /// Parameters:
    /// - `name`: The block label (used for branching)
    ///
    /// Creates a new basic block and sets it as the current block for
    /// subsequent instruction additions. Must be called while a function
    /// is active.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type};
    /// use lamina::ir::builder::var;
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("condition", Type::Void)
    ///     .branch(var("cond"), "then_block", "else_block")
    ///     .block("then_block")
    ///     // Add instructions to then_block
    ///     .block("else_block");
    ///     // Add instructions to else_block
    /// ```
    pub fn block(&mut self, name: &'a str) -> &mut Self {
        if self.current_function.is_some() {
            self.current_block = Some(name);
            self.block_instructions.insert(name, vec![]);
        }
        self
    }

    /// Sets the entry block for the current function
    ///
    /// Parameters:
    /// - `name`: The block name to use as entry point
    ///
    /// By default, the first block created ("entry") will be the entry point.
    /// Use this method to override that behavior.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("custom_entry", Type::Void)
    ///     .block("setup")
    ///     // ... setup instructions
    ///     .block("main_logic")
    ///     // ... main logic
    ///     .set_entry_block("main_logic"); // Skip the setup block when calling
    /// ```
    pub fn set_entry_block(&mut self, name: &'a str) -> &mut Self {
        if let Some(func_name) = self.current_function {
            self.function_entry_blocks.insert(func_name, name);
        }
        self
    }

    /// Adds a raw instruction to the current block
    ///
    /// Parameters:
    /// - `instruction`: The instruction to add
    ///
    /// This is a low-level method, mainly used internally by other builder methods.
    /// You should prefer using the specialized methods unless you need to add a
    /// custom instruction type.
    pub fn inst(&mut self, instruction: Instruction<'a>) -> &mut Self {
        if let (Some(_func_name), Some(block_name)) = (self.current_function, self.current_block)
            && let Some(instructions) = self.block_instructions.get_mut(block_name)
        {
            instructions.push(instruction);
        }
        self
    }

    /// Allocates stack memory (automatic lifetime management)
    ///
    /// # Parameters
    /// - `result`: Name for the pointer variable that will hold the allocation
    /// - `ty`: Type of the data to allocate on the stack
    ///
    /// # Returns
    /// A pointer to the newly allocated memory of the specified type.
    ///
    /// # Important Notes
    /// - **Automatic cleanup**: Memory is automatically freed when function returns
    /// - **Performance**: Fast allocation/deallocation with minimal overhead
    /// - **Lifetime**: Limited to current function scope
    /// - **Size limits**: May have stack size limitations for large allocations
    ///
    /// # Memory Management
    /// Stack allocation requires no explicit deallocation. The memory is automatically
    /// reclaimed when the function returns. This makes it ideal for:
    /// - Local variables
    /// - Temporary buffers
    /// - Function-scoped data structures
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("stack_example", Type::Void)
    ///     // Allocate on stack (automatic cleanup)
    ///     .alloc_stack("local", Type::Primitive(PrimitiveType::I32))
    ///     // Use the memory
    ///     .store(Type::Primitive(PrimitiveType::I32), var("local"), i32(42))
    ///     // Load and use
    ///     .load("value", Type::Primitive(PrimitiveType::I32), var("local"))
    ///     .print(var("value"))
    ///     .ret_void(); // Memory automatically freed here
    /// ```
    pub fn alloc_stack(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Stack,
            allocated_ty: ty,
        })
    }

    /// Allocates heap memory
    ///
    /// # Parameters
    /// - `result`: Name for the pointer variable that will hold the allocation
    /// - `ty`: Type of the data to allocate on the heap
    ///
    /// # Returns
    /// A pointer to the newly allocated memory of the specified type.
    ///
    /// # Important Notes
    /// - **Memory leaks**: Always call `dealloc()` on the returned pointer when done
    /// - **Performance**: Heap allocation is slower than stack allocation
    /// - **Lifetime**: Heap memory persists until explicitly deallocated
    /// - **Thread safety**: Consider synchronization if used in multi-threaded contexts
    ///
    /// # Memory Management Responsibility
    /// Unlike stack allocation, heap allocation requires explicit memory management.
    /// Failing to deallocate heap memory will result in memory leaks.
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("heap_example", Type::Void)
    ///     // Allocate on heap
    ///     .alloc_heap("data", Type::Primitive(PrimitiveType::I32))
    ///     // Use the memory
    ///     .store(Type::Primitive(PrimitiveType::I32), var("data"), i32(42))
    ///     // Always deallocate
    ///     .dealloc(var("data"))
    ///     .ret_void();
    /// ```
    pub fn alloc_heap(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Heap,
            allocated_ty: ty,
        })
    }

    /// Stores a value to memory
    ///
    /// Parameters:
    /// - `ty`: Type of the value being stored
    /// - `ptr`: Pointer to the destination memory
    /// - `val`: Value to store
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Store 42 into ptr
    /// builder.store(
    ///     Type::Primitive(PrimitiveType::I32),
    ///     var("ptr"),
    ///     i32(42)
    /// );
    /// ```
    pub fn store(&mut self, ty: Type<'a>, ptr: Value<'a>, val: Value<'a>) -> &mut Self {
        self.inst(Instruction::Store {
            ty,
            ptr,
            value: val,
        })
    }

    /// Loads a value from memory
    ///
    /// Parameters:
    /// - `result`: Name for the loaded value
    /// - `ty`: Type of the value to load
    /// - `ptr`: Pointer to the source memory
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::var;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Load an i32 from ptr
    /// builder.load("val", Type::Primitive(PrimitiveType::I32), var("ptr"));
    /// ```
    pub fn load(&mut self, result: &'a str, ty: Type<'a>, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Load { result, ty, ptr })
    }

    /// Creates a binary operation instruction
    ///
    /// Parameters:
    /// - `op`: The binary operation (Add, Sub, Mul, Div, etc.)
    /// - `result`: Name for the result variable
    /// - `ty`: Primitive type of the operands and result
    /// - `lhs`: Left-hand side operand
    /// - `rhs`: Right-hand side operand
    ///
    /// # Supported Operations
    /// - **Arithmetic**: `Add`, `Sub`, `Mul`, `Div` (integer division)
    /// - **Special**: `PtrToInt`, `IntToPtr` (pointer conversions)
    ///
    /// # Examples
    ///
    /// ## Basic Arithmetic
    /// ```rust
    /// use lamina::ir::{IRBuilder, BinaryOp, PrimitiveType};
    /// use lamina::ir::builder::{var, i32, i64};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Compute x + y
    /// builder.binary(
    ///     BinaryOp::Add,
    ///     "sum",
    ///     PrimitiveType::I32,
    ///     var("x"),
    ///     var("y")
    /// );
    ///
    /// // Compute x * 2 with constants
    /// builder.binary(
    ///     BinaryOp::Mul,
    ///     "doubled",
    ///     PrimitiveType::I64,
    ///     var("x"),
    ///     i64(2)
    /// );
    /// ```
    ///
    ///
    /// ## Complex Expressions
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("complex_arithmetic", Type::Primitive(PrimitiveType::I32))
    ///     .block("entry")
    ///         // Compute (a + b) * (c - d)
    ///         .binary(BinaryOp::Add, "sum", PrimitiveType::I32, var("a"), var("b"))
    ///         .binary(BinaryOp::Sub, "diff", PrimitiveType::I32, var("c"), var("d"))
    ///         .binary(BinaryOp::Mul, "result", PrimitiveType::I32, var("sum"), var("diff"))
    ///         .ret(Type::Primitive(PrimitiveType::I32), var("result"));
    /// ```
    pub fn binary(
        &mut self,
        op: BinaryOp,
        result: &'a str,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }

    /// Creates a comparison operation instruction
    ///
    /// Parameters:
    /// - `op`: The comparison operation (Eq, Ne, Gt, Ge, Lt, Le)
    /// - `result`: Name for the boolean result variable
    /// - `ty`: Primitive type of the values being compared
    /// - `lhs`: Left-hand side operand
    /// - `rhs`: Right-hand side operand
    ///
    /// Result will be a boolean value (true or false).
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, CmpOp, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Check if x < 10
    /// builder.cmp(
    ///     CmpOp::Lt,
    ///     "is_small",
    ///     PrimitiveType::I32,
    ///     var("x"),
    ///     i32(10)
    /// );
    /// ```
    pub fn cmp(
        &mut self,
        op: CmpOp,
        result: &'a str,
        ty: PrimitiveType,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }

    /// Creates a conditional branch instruction
    ///
    /// Parameters:
    /// - `condition`: Boolean value that determines which branch to take
    /// - `true_label`: Block label to jump to if condition is true
    /// - `false_label`: Block label to jump to if condition is false
    ///
    /// This instruction must be the last one in a block.
    ///
    /// # Control Flow Patterns
    ///
    /// ## Simple Conditional
    /// ```rust
    /// use lamina::ir::{IRBuilder, CmpOp, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Branch based on comparison result
    /// builder
    ///     .cmp(CmpOp::Lt, "is_neg", PrimitiveType::I32, var("x"), i32(0))
    ///     .branch(var("is_neg"), "negative", "non_negative");
    /// ```
    ///
    /// ## Complex Conditional with Multiple Blocks
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp, CmpOp, FunctionParameter};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function_with_params(
    ///         "conditional_logic",
    ///         vec![FunctionParameter { name: "x", ty: Type::Primitive(PrimitiveType::I32), annotations: vec![] }],
    ///         Type::Primitive(PrimitiveType::I32)
    ///     )
    ///     .block("entry")
    ///         // Check if x > 0
    ///         .cmp(CmpOp::Gt, "positive", PrimitiveType::I32, var("x"), i32(0))
    ///         .branch(var("positive"), "positive_branch", "negative_branch")
    ///
    ///     .block("positive_branch")
    ///         // Handle positive case
    ///         .binary(BinaryOp::Mul, "squared", PrimitiveType::I32, var("x"), var("x"))
    ///         .jump("end")
    ///
    ///     .block("negative_branch")
    ///         // Handle non-positive case
    ///         .binary(BinaryOp::Sub, "negated", PrimitiveType::I32, i32(0), var("x"))
    ///         .jump("end")
    ///
    ///     .block("end")
    ///         .ret(Type::Primitive(PrimitiveType::I32), var("squared")); // Note: need phi node for proper SSA
    /// ```
    pub fn branch(
        &mut self,
        condition: Value<'a>,
        true_label: &'a str,
        false_label: &'a str,
    ) -> &mut Self {
        self.inst(Instruction::Br {
            condition,
            true_label,
            false_label,
        })
    }

    /// Creates an unconditional jump instruction
    ///
    /// Parameters:
    /// - `target`: Block label to jump to
    ///
    /// This instruction must be the last one in a block.
    ///
    /// # Jump Patterns
    ///
    /// ## Simple Jump
    /// ```rust
    /// use lamina::ir::IRBuilder;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Jump to the "end" block
    /// builder.jump("end");
    /// ```
    ///
    /// ## Loop Control
    /// ```rust
    /// use lamina::ir::{IRBuilder, CmpOp, PrimitiveType, BinaryOp, Type};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("loop_example", Type::Void)
    ///     .block("loop_start")
    ///         // Loop body
    ///         .binary(BinaryOp::Add, "counter", PrimitiveType::I32, var("counter"), i32(1))
    ///         // Check loop condition
    ///         .cmp(CmpOp::Lt, "continue", PrimitiveType::I32, var("counter"), i32(10))
    ///         .branch(var("continue"), "loop_start", "loop_end")
    ///
    ///     .block("loop_end")
    ///         .ret_void();
    /// ```
    ///
    /// ## Early Return
    /// ```rust
    /// use lamina::ir::{IRBuilder, CmpOp, PrimitiveType, Type, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("early_return", Type::Primitive(PrimitiveType::I32))
    ///     .block("check_input")
    ///         // Check if input is valid
    ///         .cmp(CmpOp::Eq, "is_zero", PrimitiveType::I32, var("input"), i32(0))
    ///         .branch(var("is_zero"), "error_case", "normal_case")
    ///
    ///     .block("error_case")
    ///         // Handle error case
    ///         .jump("return_zero")
    ///
    ///     .block("normal_case")
    ///         // Handle normal case
    ///         .binary(BinaryOp::Mul, "result", PrimitiveType::I32, var("input"), i32(2))
    ///         .jump("end")
    ///
    ///     .block("return_zero")
    ///         .ret(Type::Primitive(PrimitiveType::I32), i32(0))
    ///
    ///     .block("end")
    ///         .ret(Type::Primitive(PrimitiveType::I32), var("result"));
    /// ```
    pub fn jump(&mut self, target: &'a str) -> &mut Self {
        self.inst(Instruction::Jmp {
            target_label: target,
        })
    }

    /// Creates a switch instruction for multi-way branching on an integer value.
    ///
    /// Parameters:
    /// - `ty`: Primitive type of the switched value (typically an integer or bool)
    /// - `value`: Value to inspect
    /// - `default`: Label to jump to if no case matches
    /// - `cases`: Slice of `(Literal, Label)` pairs for each case
    pub fn switch(
        &mut self,
        ty: PrimitiveType,
        value: Value<'a>,
        default: &'a str,
        cases: &[(Literal<'a>, &'a str)],
    ) -> &mut Self {
        let mapped_cases = cases.iter().map(|(lit, lbl)| (lit.clone(), *lbl)).collect();
        self.inst(Instruction::Switch {
            ty,
            value,
            default,
            cases: mapped_cases,
        })
    }

    /// Creates a return instruction with a value
    ///
    /// Parameters:
    /// - `ty`: Type of the returned value
    /// - `value`: Value to return
    ///
    /// This instruction must be the last one in a block.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::var;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Return the computed result
    /// builder.ret(Type::Primitive(PrimitiveType::I32), var("result"));
    /// ```
    pub fn ret(&mut self, ty: Type<'a>, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Ret {
            ty,
            value: Some(value),
        })
    }

    /// Creates a void return instruction (returns nothing)
    ///
    /// Use this for functions that don't return a value.
    /// This instruction must be the last one in a block.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::IRBuilder;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Return from a void function
    /// builder.ret_void();
    /// ```
    pub fn ret_void(&mut self) -> &mut Self {
        self.inst(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    }

    /// Creates a zero-extension instruction
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `source_type`: Original primitive type
    /// - `target_type`: Target primitive type (must be larger than source)
    /// - `value`: Value to extend
    ///
    /// Zero-extension preserves the original value's bits and fills higher bits with zeros.
    /// Typically used to convert between integer types of different sizes.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, PrimitiveType};
    /// use lamina::ir::builder::var;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Extend an i32 to i64
    /// builder.zext(
    ///     "extended",
    ///     PrimitiveType::I32,
    ///     PrimitiveType::I64,
    ///     var("original")
    /// );
    /// ```
    pub fn zext(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates an integer truncation instruction
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `source_type`: Original integer primitive type (wider)
    /// - `target_type`: Target integer primitive type (narrower)
    /// - `value`: Value to truncate
    ///
    /// Truncation keeps the least significant bits of the value and discards
    /// the higher bits. It is typically used to reduce the bit-width of an
    /// integer, e.g. from `i64` to `i32`.
    pub fn trunc(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Trunc {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a sign-extension instruction
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `source_type`: Original signed integer type (narrower)
    /// - `target_type`: Target signed integer type (wider)
    /// - `value`: Value to extend
    ///
    /// Sign-extension interprets the most significant bit of the source as a
    /// sign bit and replicates it into the new high bits of the result.
    pub fn sext(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SignExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a bitcast instruction between equally-sized primitive types.
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `source_type`: Original primitive type
    /// - `target_type`: Target primitive type (must have same bit-width)
    /// - `value`: Value to reinterpret
    pub fn bitcast(
        &mut self,
        result: &'a str,
        source_type: PrimitiveType,
        target_type: PrimitiveType,
        value: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Bitcast {
            result,
            source_type,
            target_type,
            value,
        })
    }

    /// Creates a select instruction (SSA conditional expression).
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `ty`: Type of both `true_val` and `false_val`
    /// - `cond`: Boolean condition
    /// - `true_val`: Value when `cond` is true
    /// - `false_val`: Value when `cond` is false
    pub fn select(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        cond: Value<'a>,
        true_val: Value<'a>,
        false_val: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        })
    }

    /// Gets a pointer to an array element (pointer arithmetic)
    ///
    /// # Parameters
    /// - `result`: Name for the resulting element pointer
    /// - `array_ptr`: Pointer to the base of the array
    /// - `index`: Zero-based index of the element to access
    ///
    /// # Returns
    /// A pointer to the element at the specified index within the array.
    ///
    /// # Memory Safety
    /// - **Bounds checking**: No automatic bounds checking is performed
    /// - **Valid indices**: Ensure `index` is within array bounds
    /// - **Array pointer**: `array_ptr` must point to a valid array
    /// - **Type safety**: Result pointer has the element type, not array type
    ///
    /// # Index Calculation
    /// The operation computes: `result = array_ptr + (index * element_size)`
    /// where `element_size` is determined by the array's element type.
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("array_access", Type::Void)
    ///     // Allocate array of 10 i32 elements
    ///     .alloc_stack("arr", Type::Array {
    ///         element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
    ///         size: 10
    ///     })
    ///     // Get pointer to element at index 5
    ///     .getelementptr("elem_ptr", var("arr"), i32(5), PrimitiveType::I32)
    ///     // Store value at that location
    ///     .store(Type::Primitive(PrimitiveType::I32), var("elem_ptr"), i32(99))
    ///     // Load it back
    ///     .load("value", Type::Primitive(PrimitiveType::I32), var("elem_ptr"))
    ///     .print(var("value")) // Should print 99
    ///     .ret_void();
    /// ```
    pub fn getelementptr(
        &mut self,
        result: &'a str,
        array_ptr: Value<'a>,
        index: Value<'a>,
        element_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        })
    }

    /// Convert pointer to integer for pointer arithmetic
    ///
    /// This instruction extracts the memory address from a pointer as an integer value,
    /// enabling pointer arithmetic operations. On 64-bit systems, this typically converts
    /// to a 64-bit integer representing the absolute memory address.
    ///
    /// # Use Cases
    /// - Pointer arithmetic and address calculations
    /// - Memory address inspection and debugging
    /// - Dynamic memory access patterns
    /// - Brainfuck-style pointer manipulation
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var, i64};
    ///
    /// let mut builder = IRBuilder::new();
    ///
    /// // Simple pointer arithmetic demonstration
    /// builder
    ///     .function("simple_ptr_math", Type::Primitive(PrimitiveType::I64))
    ///     .block("entry")
    ///     // Allocate an i64 variable
    ///     .alloc_stack("value", Type::Primitive(PrimitiveType::I64))
    ///     // Store a value
    ///     .store(Type::Primitive(PrimitiveType::I64), var("value"), i64(42))
    ///     // Convert pointer to integer
    ///     .ptrtoint("addr", var("value"), PrimitiveType::I64)
    ///     // Convert back to pointer
    ///     .inttoptr("new_ptr", var("addr"), PrimitiveType::Ptr)
    ///     // Load the original value
    ///     .load("result", Type::Primitive(PrimitiveType::I64), var("new_ptr"))
    ///     .ret(Type::Primitive(PrimitiveType::I64), var("result"));
    /// ```
    pub fn ptrtoint(
        &mut self,
        result: &'a str,
        ptr_value: Value<'a>,
        target_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::PtrToInt {
            result,
            ptr_value,
            target_type,
        })
    }

    /// Convert integer back to pointer
    ///
    /// This instruction creates a pointer from an integer address value, enabling
    /// dynamic pointer creation and memory access at computed addresses. This is
    /// essential for implementing dynamic memory access patterns and pointer arithmetic.
    ///
    /// # Use Cases
    /// - Creating pointers from computed memory addresses
    /// - Dynamic memory access with calculated offsets
    /// - Implementing low-level memory operations
    /// - Brainfuck interpreter implementation
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var, i64};
    ///
    /// let mut builder = IRBuilder::new();
    ///
    /// // Pointer arithmetic round-trip
    /// builder
    ///     .function("pointer_roundtrip", Type::Primitive(PrimitiveType::I64))
    ///     .block("entry")
    ///     // Allocate a value
    ///     .alloc_stack("val", Type::Primitive(PrimitiveType::I64))
    ///     .store(Type::Primitive(PrimitiveType::I64), var("val"), i64(123))
    ///     // Convert to integer
    ///     .ptrtoint("addr", var("val"), PrimitiveType::I64)
    ///     // Convert back to pointer
    ///     .inttoptr("reconstructed", var("addr"), PrimitiveType::Ptr)
    ///     // Load the original value
    ///     .load("result", Type::Primitive(PrimitiveType::I64), var("reconstructed"))
    ///     .ret(Type::Primitive(PrimitiveType::I64), var("result"));
    /// ```
    ///
    /// # Note on Removed ptradd Instruction
    ///
    /// Initially, a `ptradd` instruction was considered for direct pointer+offset arithmetic.
    /// However, this instruction was removed as unnecessary because the same functionality
    /// can be achieved more efficiently using the `ptrtoint` + `inttoptr` sequence.
    ///
    /// This approach provides more flexibility and doesn't require additional instruction
    /// support in the compiler.
    ///
    /// # Safety Note
    /// Creating pointers from arbitrary integer values can lead to undefined behavior
    /// if the resulting pointer doesn't point to valid memory. Always ensure the
    /// integer address represents a valid memory location that your program has
    /// access to.
    pub fn inttoptr(
        &mut self,
        result: &'a str,
        int_value: Value<'a>,
        target_type: PrimitiveType,
    ) -> &mut Self {
        self.inst(Instruction::IntToPtr {
            result,
            int_value,
            target_type,
        })
    }

    /// Gets a pointer to a struct field (structure field access)
    ///
    /// # Parameters
    /// - `result`: Name for the resulting field pointer
    /// - `struct_ptr`: Pointer to the struct instance
    /// - `field_index`: Zero-based index of the field to access
    ///
    /// # Returns
    /// A pointer to the specified field within the struct.
    ///
    /// # Memory Safety
    /// - **Valid field index**: Must be within the struct's field count
    /// - **Struct pointer**: `struct_ptr` must point to a valid struct instance
    /// - **Type safety**: Result pointer has the field type, not struct type
    ///
    /// # Field Layout
    /// Fields are ordered according to the struct type definition:
    /// ```text
    /// // struct Point { x: i32, y: i32 }
    /// // Field 0: x (i32), Field 1: y (i32)
    /// ```
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, StructField};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("struct_access", Type::Void)
    ///     // Allocate a struct with two i32 fields
    ///     .alloc_stack("point", Type::Struct(vec![
    ///         StructField { name: "x", ty: Type::Primitive(PrimitiveType::I32) },
    ///         StructField { name: "y", ty: Type::Primitive(PrimitiveType::I32) }
    ///     ]))
    ///     // Get pointer to field 0 (x coordinate)
    ///     .struct_gep("x_ptr", var("point"), 0)
    ///     // Store value in x field
    ///     .store(Type::Primitive(PrimitiveType::I32), var("x_ptr"), i32(10))
    ///     // Get pointer to field 1 (y coordinate)
    ///     .struct_gep("y_ptr", var("point"), 1)
    ///     // Store value in y field
    ///     .store(Type::Primitive(PrimitiveType::I32), var("y_ptr"), i32(20))
    ///     .ret_void();
    /// ```
    ///
    /// # Common Use Cases
    /// - Accessing individual fields of struct instances
    /// - Implementing object-oriented patterns in IR
    /// - Manipulating complex data structures
    /// - Field-based data processing
    pub fn struct_gep(
        &mut self,
        result: &'a str,
        struct_ptr: Value<'a>,
        field_index: usize,
    ) -> &mut Self {
        self.inst(Instruction::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
        })
    }

    /// Creates a function call instruction
    ///
    /// Parameters:
    /// - `result`: Optional name for the result (None for void functions)
    /// - `func_name`: Name of the function to call
    /// - `args`: Vector of argument values
    ///
    /// Example:
    /// ```
    /// use lamina::ir::IRBuilder;
    /// use lamina::ir::builder::{var, string};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Call a function that returns a value
    /// builder.call(
    ///     Some("result"),
    ///     "calculate",
    ///     vec![var("x"), var("y")]
    /// );
    ///
    /// // Call a void function
    /// builder.call(None, "print_message", vec![string("Hello")]);
    /// ```
    pub fn call(
        &mut self,
        result: Option<&'a str>,
        func_name: &'a str,
        args: Vec<Value<'a>>,
    ) -> &mut Self {
        self.inst(Instruction::Call {
            result,
            func_name,
            args,
        })
    }

    /// Writes a buffer to stdout (raw syscall)
    ///
    /// # Parameters
    /// - `buffer`: Pointer to the buffer to write
    /// - `size`: Number of bytes to write
    /// - `result`: Variable to store the result (bytes written, or -1 on error)
    ///
    /// # Returns
    /// Number of bytes written on success, -1 on error.
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("write_example", Type::Void)
    ///     // Write 5 bytes from buffer to stdout
    ///     .write(var("buffer"), i32(5), "bytes_written")
    ///     .ret_void();
    /// ```
    pub fn write(&mut self, buffer: Value<'a>, size: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::Write {
            buffer,
            size,
            result,
        })
    }

    /// Reads from stdin into a buffer (raw syscall)
    ///
    /// # Parameters
    /// - `buffer`: Pointer to buffer to read into
    /// - `size`: Maximum number of bytes to read
    /// - `result`: Variable to store the result (bytes read, or -1 on error)
    ///
    /// # Returns
    /// Number of bytes read on success, -1 on error.
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("read_example", Type::Void)
    ///     // Read up to 100 bytes from stdin
    ///     .read(var("buffer"), i32(100), "bytes_read")
    ///     .ret_void();
    /// ```
    pub fn read(&mut self, buffer: Value<'a>, size: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::Read {
            buffer,
            size,
            result,
        })
    }

    /// Writes a single byte to stdout (raw syscall)
    ///
    /// # Parameters
    /// - `value`: Byte value to write (will be truncated to 8 bits)
    /// - `result`: Variable to store the result (1 on success, -1 on error)
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("write_byte_example", Type::Void)
    ///     // Write 'A' (65) to stdout
    ///     .write_byte(i32(65), "success")
    ///     .ret_void();
    /// ```
    pub fn write_byte(&mut self, value: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::WriteByte { value, result })
    }

    /// Reads a single byte from stdin (raw syscall)
    ///
    /// # Parameters
    /// - `result`: Variable to store the result (byte read, or -1 on error)
    ///
    /// # Returns
    /// Byte value (0-255) on success, -1 on error.
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("read_byte_example", Type::Void)
    ///     // Read one byte from stdin
    ///     .read_byte("byte_read")
    ///     .ret_void();
    /// ```
    pub fn read_byte(&mut self, result: &'a str) -> &mut Self {
        self.inst(Instruction::ReadByte { result })
    }

    /// Writes the value stored at a pointer location to stdout (I/O operation)
    ///
    /// # Parameters
    /// - `ptr`: Pointer to the memory location containing the value to write
    /// - `result`: Variable to store the result (bytes written on success, -1 on error)
    ///
    /// # Important Notes
    /// - **Value-based I/O**: Writes the VALUE stored at the pointer, not the pointer address
    /// - **Memory dereference**: Automatically loads the value before writing to stdout
    /// - **Type-aware**: Handles different data types (i8, i32, i64) correctly
    /// - **Raw output**: Produces binary data (not formatted text)
    ///
    /// # Memory Access Pattern
    /// ```text
    /// Memory: [value] <- ptr points here
    /// Output:  value  <- sent to stdout
    /// ```
    ///
    /// # Use Cases
    /// - **Direct memory output**: Writing computed results to stdout
    /// - **Binary data output**: Sending raw values without formatting
    /// - **Memory inspection**: Examining stored values during debugging
    /// - **Data serialization**: Outputting memory contents for external processing
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("memory_output", Type::Void)
    ///     // Allocate memory and store a value
    ///     .alloc_stack("data", Type::Primitive(PrimitiveType::I32))
    ///     .store(Type::Primitive(PrimitiveType::I32), var("data"), i32(42))
    ///     // Write the stored value to stdout (outputs binary 42)
    ///     .write_ptr(var("data"), "bytes_written")
    ///     .ret_void();
    /// ```
    ///
    /// # See Also
    /// - `write()`: Write buffer contents to stdout
    /// - `write_byte()`: Write single byte to stdout
    /// - `load()`: Load value from memory location
    /// - `store()`: Store value to memory location
    pub fn write_ptr(&mut self, ptr: Value<'a>, result: &'a str) -> &mut Self {
        self.inst(Instruction::WritePtr { ptr, result })
    }

    /// Reads binary data from stdin and stores it as a pointer-sized value
    ///
    /// # Parameters
    /// - `result`: Variable to store the binary data read from stdin (as i64)
    ///
    /// # Important Notes
    /// - **Binary input**: Reads raw bytes from stdin (not text)
    /// - **8-byte read**: Always reads exactly 8 bytes (64-bit)
    /// - **No validation**: Assumes input contains valid binary data
    /// - **Blocking operation**: Waits for input if stdin is empty
    ///
    /// # Use Cases
    /// - **Binary data input**: Reading serialized data from stdin
    /// - **Memory initialization**: Loading values from external sources
    /// - **Data deserialization**: Reconstructing values from binary streams
    /// - **Raw input processing**: Handling non-text data from pipes or files
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{var};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("binary_input", Type::Void)
    ///     // Read 8 bytes of binary data from stdin
    ///     .read_byte("input_byte")
    ///     // Use the data (e.g., store it somewhere)
    ///     .alloc_stack("storage", Type::Primitive(PrimitiveType::I8))
    ///     .store(Type::Primitive(PrimitiveType::I8), var("storage"), var("input_byte"))
    ///     .ret_void();
    /// ```
    ///
    /// # Input/Output Behavior
    /// ```text
    /// stdin:  [binary_data] (8 bytes)
    /// Result: input_data = binary_data (as i64)
    /// ```
    ///
    /// # See Also
    /// - `read()`: Read buffer from stdin
    /// - `read_byte()`: Read single byte from stdin
    /// - `write_ptr()`: Write value to stdout    ///
    ///   Creates a print instruction for debugging
    ///
    /// # Parameters
    /// - `value`: The value to print (can be constants, variables, or expressions)
    ///
    /// # Supported Types
    /// The print instruction can output values of any Lamina IR type:
    /// - **Integers**: `i8`, `i32`, `i64` (displayed as decimal numbers)
    /// - **Booleans**: `true` or `false`
    /// - **Strings**: String literals and string variables
    /// - **Pointers**: Memory addresses (displayed in hexadecimal)
    /// - **Variables**: Current values of SSA variables
    ///
    /// # Use Cases
    /// - **Debugging**: Inspect variable values during execution
    /// - **Tracing**: Log program execution flow
    /// - **Testing**: Verify computation results
    /// - **Development**: Understand IR code behavior
    ///
    /// # Performance Note
    /// Print instructions are primarily for development and debugging.
    /// For production I/O, use the raw `write`/`read` instructions for better performance.
    ///
    /// # See Also
    /// - `write()`: Raw buffer write to stdout
    /// - `read()`: Raw buffer read from stdin
    /// - `writebyte()`: Single byte write
    /// - `readbyte()`: Single byte read
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32, bool, string};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("debug_example", Type::Void)
    ///     // Print constants
    ///     .print(i32(42))
    ///     .print(bool(true))
    ///     .print(string("Debug message"))
    ///     // Print variables
    ///     .print(var("result"))
    ///     .ret_void();
    /// ```
    ///
    /// # Output Format
    /// The exact output format depends on the target architecture's runtime,
    /// but typically follows these patterns:
    /// - Integers: `42`, `-15`, `0`
    /// - Booleans: `true`, `false`
    /// - Strings: `hello world`
    /// - Pointers: `0x7fff12345678`
    pub fn print(&mut self, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Print { value })
    }

    /// Creates a tuple from multiple values (composite data construction)
    ///
    /// # Parameters
    /// - `result`: Name for the resulting tuple variable
    /// - `elements`: Vector of values to combine into a tuple
    ///
    /// # Returns
    /// A tuple value containing all the specified elements in order.
    ///
    /// # Tuple Structure
    /// Tuples in Lamina IR are heterogeneous collections that can contain
    /// values of different types. Elements are accessed by index using `extract_tuple()`.
    ///
    /// # Use Cases
    /// - **Multiple return values**: Functions can return multiple values as a tuple
    /// - **Data aggregation**: Combine related values into a single unit
    /// - **Temporary grouping**: Group values for processing
    /// - **Pattern matching**: Enable destructuring operations
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{i32, bool, var};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("tuple_demo", Type::Void)
    ///     // Create a tuple with mixed types
    ///     .tuple("pair", vec![i32(42), bool(true)])
    ///     // Extract first element (index 0)
    ///     .extract_tuple("num", var("pair"), 0)
    ///     // Extract second element (index 1)
    ///     .extract_tuple("flag", var("pair"), 1)
    ///     // Print both values
    ///     .print(var("num"))
    ///     .print(var("flag"))
    ///     .ret_void();
    /// ```
    pub fn tuple(&mut self, result: &'a str, elements: Vec<Value<'a>>) -> &mut Self {
        self.inst(Instruction::Tuple { result, elements })
    }

    /// Extracts a value from a tuple (tuple element access)
    ///
    /// # Parameters
    /// - `result`: Name for the extracted element variable
    /// - `tuple_val`: The tuple value to extract from
    /// - `index`: Zero-based index of the element to extract
    ///
    /// # Returns
    /// The value at the specified index within the tuple.
    ///
    /// # Type Safety
    /// The result type matches the type of the element at the specified index
    /// in the tuple's type definition.
    ///
    /// # Bounds Checking
    /// - **Valid indices**: Index must be within the tuple's element count
    /// - **Runtime behavior**: Out-of-bounds access has undefined behavior
    /// - **Compile-time verification**: Index validity should be verified at IR construction time
    ///
    /// # Examples
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType};
    /// use lamina::ir::builder::{i32, bool, string, var};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("extract_demo", Type::Void)
    ///     // Create tuple: (42, true, "hello")
    ///     .tuple("data", vec![i32(42), bool(true), string("hello")])
    ///     // Extract number (index 0)
    ///     .extract_tuple("num", var("data"), 0)
    ///     // Extract boolean (index 1)
    ///     .extract_tuple("flag", var("data"), 1)
    ///     // Extract string (index 2)
    ///     .extract_tuple("text", var("data"), 2)
    ///     // Use extracted values
    ///     .print(var("num"))
    ///     .print(var("flag"))
    ///     .print(var("text"))
    ///     .ret_void();
    /// ```
    ///
    /// # Common Patterns
    /// - **Destructuring**: Extract multiple elements from a tuple
    /// - **Data processing**: Access individual components of composite data
    /// - **Return value unpacking**: Handle multiple return values from functions
    /// - **Field access**: Simulate struct field access with tuples
    pub fn extract_tuple(
        &mut self,
        result: &'a str,
        tuple_val: Value<'a>,
        index: usize,
    ) -> &mut Self {
        self.inst(Instruction::ExtractTuple {
            result,
            tuple_val,
            index,
        })
    }

    /// Deallocates heap memory
    ///
    /// # Parameters
    /// - `ptr`: Pointer to the heap memory to deallocate (must be from `alloc_heap`)
    ///
    /// # Important Notes
    /// - **Use after alloc_heap**: Only call on pointers returned by `alloc_heap()`
    /// - **Double deallocation**: Never deallocate the same pointer twice
    /// - **Dangling pointers**: Do not use the pointer after deallocation
    /// - **Null pointers**: Behavior undefined for null or invalid pointers
    ///
    /// # Memory Safety
    /// This operation frees the memory pointed to by `ptr`. Using `ptr` after
    /// deallocation will result in undefined behavior. Always ensure:
    /// 1. The pointer came from a previous `alloc_heap()` call
    /// 2. The memory hasn't been deallocated already
    /// 3. No other references to this memory exist
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// builder
    ///     .function("safe_heap_usage", Type::Void)
    ///     // Allocate memory
    ///     .alloc_heap("temp", Type::Primitive(PrimitiveType::I32))
    ///     // Use the memory
    ///     .store(Type::Primitive(PrimitiveType::I32), var("temp"), i32(42))
    ///     // Always clean up
    ///     .dealloc(var("temp"))
    ///     .ret_void();
    /// ```
    pub fn dealloc(&mut self, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Dealloc { ptr })
    }

    /// Performs an atomic load operation with specified memory ordering.
    ///
    /// # Parameters
    /// - `result`: Name for the loaded value
    /// - `ty`: Type of the value being loaded
    /// - `ptr`: Pointer to the atomic location
    /// - `ordering`: Memory ordering constraint
    ///
    /// # Example
    /// ```rust
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, MemoryOrdering};
    /// use lamina::ir::builder::var;
    ///
    /// let mut builder = IRBuilder::new();
    /// builder.atomic_load("value", Type::Primitive(PrimitiveType::I32), var("atomic_ptr"), MemoryOrdering::SeqCst);
    /// ```
    #[cfg(feature = "nightly")]
    pub fn atomic_load(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicLoad {
            result: result.into(),
            ty,
            ptr,
            ordering,
        })
    }

    /// Performs an atomic store operation with specified memory ordering.
    ///
    /// # Parameters
    /// - `ty`: Type of the value being stored
    /// - `ptr`: Pointer to the atomic location
    /// - `value`: Value to store
    /// - `ordering`: Memory ordering constraint
    #[cfg(feature = "nightly")]
    pub fn atomic_store(
        &mut self,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        })
    }

    /// Performs an atomic binary operation (read-modify-write).
    ///
    /// # Parameters
    /// - `op`: Atomic binary operation to perform
    /// - `result`: Name for the result (previous value)
    /// - `ty`: Type of the atomic location
    /// - `ptr`: Pointer to the atomic location
    /// - `value`: Value for the operation
    /// - `ordering`: Memory ordering constraint
    #[cfg(feature = "nightly")]
    pub fn atomic_binary(
        &mut self,
        op: AtomicBinOp,
        result: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicBinary {
            op,
            result: result.into(),
            ty,
            ptr,
            value,
            ordering,
        })
    }

    /// Performs an atomic compare-exchange operation.
    ///
    /// # Parameters
    /// - `result`: Name for the loaded value
    /// - `success`: Name for the success flag (boolean)
    /// - `ty`: Type of the atomic location
    /// - `ptr`: Pointer to the atomic location
    /// - `expected`: Expected value for comparison
    /// - `desired`: Desired value for exchange
    /// - `success_ordering`: Memory ordering on success
    /// - `failure_ordering`: Memory ordering on failure
    #[cfg(feature = "nightly")]
    pub fn atomic_compare_exchange(
        &mut self,
        result: &'a str,
        success: &'a str,
        ty: Type<'a>,
        ptr: Value<'a>,
        expected: Value<'a>,
        desired: Value<'a>,
        success_ordering: MemoryOrdering,
        failure_ordering: MemoryOrdering,
    ) -> &mut Self {
        self.inst(Instruction::AtomicCompareExchange {
            result: result.into(),
            success: success.into(),
            ty,
            ptr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        })
    }

    /// Inserts a memory fence/barrier with specified memory ordering.
    ///
    /// # Parameters
    /// - `ordering`: Memory ordering constraint for the fence
    #[cfg(feature = "nightly")]
    pub fn fence(&mut self, ordering: MemoryOrdering) -> &mut Self {
        self.inst(Instruction::Fence { ordering })
    }

    /// Performs a SIMD binary operation (element-wise).
    ///
    /// # Parameters
    /// - `op`: SIMD operation to perform
    /// - `result`: Name for the result vector
    /// - `vector_type`: Type of the SIMD vector (e.g., v4f32)
    /// - `lhs`: Left-hand side vector operand
    /// - `rhs`: Right-hand side vector operand
    #[cfg(feature = "nightly")]
    pub fn simd_binary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdBinary {
            op,
            result: result.into(),
            vector_type,
            lhs,
            rhs,
        })
    }

    /// Performs a SIMD unary operation.
    ///
    /// # Parameters
    /// - `op`: SIMD operation to perform
    /// - `result`: Name for the result vector
    /// - `vector_type`: Type of the SIMD vector
    /// - `operand`: Vector operand
    #[cfg(feature = "nightly")]
    pub fn simd_unary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        operand: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdUnary {
            op,
            result: result.into(),
            vector_type,
            operand,
        })
    }

    /// Performs a SIMD ternary operation (e.g., fused multiply-add).
    ///
    /// # Parameters
    /// - `op`: SIMD operation to perform
    /// - `result`: Name for the result vector
    /// - `vector_type`: Type of the SIMD vector
    /// - `lhs`: Left-hand side vector operand
    /// - `rhs`: Right-hand side vector operand
    /// - `acc`: Accumulator/third vector operand
    #[cfg(feature = "nightly")]
    pub fn simd_ternary(
        &mut self,
        op: SimdOp,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        acc: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdTernary {
            op,
            result: result.into(),
            vector_type,
            lhs,
            rhs,
            acc,
        })
    }

    /// Performs a SIMD shuffle operation.
    ///
    /// # Parameters
    /// - `result`: Name for the result vector
    /// - `vector_type`: Type of the SIMD vector
    /// - `lhs`: Left-hand side vector operand
    /// - `rhs`: Right-hand side vector operand
    /// - `mask`: Shuffle mask (indices for rearranging elements)
    #[cfg(feature = "nightly")]
    pub fn simd_shuffle(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        lhs: Value<'a>,
        rhs: Value<'a>,
        mask: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdShuffle {
            result: result.into(),
            vector_type,
            lhs,
            rhs,
            mask,
        })
    }

    /// Extracts a single element from a SIMD vector.
    ///
    /// # Parameters
    /// - `result`: Name for the extracted scalar value
    /// - `scalar_type`: Type of the extracted element
    /// - `vector`: Source SIMD vector
    /// - `lane_index`: Which lane to extract (0-based index)
    #[cfg(feature = "nightly")]
    pub fn simd_extract(
        &mut self,
        result: &'a str,
        scalar_type: PrimitiveType,
        vector: Value<'a>,
        lane_index: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdExtract {
            result: result.into(),
            scalar_type,
            vector,
            lane_index,
        })
    }

    /// Inserts a single element into a SIMD vector.
    ///
    /// # Parameters
    /// - `result`: Name for the result vector
    /// - `vector_type`: Type of the SIMD vector
    /// - `vector`: Source SIMD vector
    /// - `scalar`: Scalar value to insert
    /// - `lane_index`: Which lane to insert into (0-based index)
    #[cfg(feature = "nightly")]
    pub fn simd_insert(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        vector: Value<'a>,
        scalar: Value<'a>,
        lane_index: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::SimdInsert {
            result: result.into(),
            vector_type,
            vector,
            scalar,
            lane_index,
        })
    }

    /// Loads a SIMD vector from memory.
    ///
    /// # Parameters
    /// - `result`: Name for the loaded vector
    /// - `vector_type`: Type of the SIMD vector
    /// - `ptr`: Pointer to load from
    /// - `alignment`: Optional alignment hint in bytes
    #[cfg(feature = "nightly")]
    pub fn simd_load(
        &mut self,
        result: &'a str,
        vector_type: Type<'a>,
        ptr: Value<'a>,
        alignment: Option<u32>,
    ) -> &mut Self {
        self.inst(Instruction::SimdLoad {
            result: result.into(),
            vector_type,
            ptr,
            alignment,
        })
    }

    /// Stores a SIMD vector to memory.
    ///
    /// # Parameters
    /// - `vector_type`: Type of the SIMD vector
    /// - `ptr`: Pointer to store to
    /// - `value`: SIMD vector value to store
    /// - `alignment`: Optional alignment hint in bytes
    #[cfg(feature = "nightly")]
    pub fn simd_store(
        &mut self,
        vector_type: Type<'a>,
        ptr: Value<'a>,
        value: Value<'a>,
        alignment: Option<u32>,
    ) -> &mut Self {
        self.inst(Instruction::SimdStore {
            vector_type,
            ptr,
            value,
            alignment,
        })
    }

    /// Creates a phi node for SSA form
    ///
    /// Parameters:
    /// - `result`: Name for the result variable
    /// - `ty`: Type of the values being merged
    /// - `incoming`: Vector of (value, label) pairs
    ///
    /// Phi nodes are used in SSA form to merge values from different basic blocks.
    /// Each incoming pair specifies a value and the predecessor block it comes from.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Merge values from two blocks
    /// builder.phi(
    ///     "merged",
    ///     Type::Primitive(PrimitiveType::I32),
    ///     vec![
    ///         (var("x"), "true_block"),
    ///         (i32(0), "false_block")
    ///     ]
    /// );
    /// ```
    pub fn phi(
        &mut self,
        result: &'a str,
        ty: Type<'a>,
        incoming: Vec<(Value<'a>, &'a str)>,
    ) -> &mut Self {
        self.inst(Instruction::Phi {
            result,
            ty,
            incoming,
        })
    }

    //
    // MODULE FINALIZATION
    //

    /// Finalizes and returns the complete IR module
    ///
    /// This method converts all the accumulated function and block data
    /// into a complete Module object that can be used for code generation,
    /// optimization, or serialization.
    ///
    /// Call this method only once all functions and instructions have been added.
    ///
    /// Returns: A Module object containing all defined functions and blocks
    ///
    /// Example:
    /// ```
    /// use lamina::ir::IRBuilder;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Add all functions and instructions...
    /// let module = builder.build();
    /// // Use the module for analysis, code generation, etc.
    /// ```
    pub fn build(&mut self) -> Module<'a> {
        // Finalize the module by converting all block instructions to basicblocks
        // and all functions to their final representation

        for (func_name, block_map) in &mut self.function_blocks {
            // Convert block instructions to BasicBlocks
            for (block_name, instructions) in self.block_instructions.iter() {
                // Only process blocks for current function
                if self.block_instructions.contains_key(block_name) {
                    block_map.insert(
                        block_name,
                        BasicBlock {
                            instructions: instructions.clone(),
                        },
                    );
                }
            }

            // Create the function and add it to the module
            if let (Some(signature), Some(entry_block)) = (
                self.function_signatures.get(func_name),
                self.function_entry_blocks.get(func_name),
            ) {
                let function = Function {
                    name: func_name,
                    signature: signature.clone(),
                    annotations: self
                        .function_annotations
                        .get(func_name)
                        .cloned()
                        .unwrap_or_else(Vec::new),
                    basic_blocks: block_map.clone(),
                    entry_block,
                };

                self.module.functions.insert(func_name, function);
            }
        }

        self.module.clone()
    }
}

//
// EXTERNAL FUNCTION SUPPORT
//

impl<'a> IRBuilder<'a> {
    /// Declares an external function (imported from another module)
    ///
    /// Parameters:
    /// - `name`: Name of the external function
    /// - `params`: Vector of parameter types and names
    /// - `return_type`: Return type of the function
    ///
    /// External functions are declarations only (no implementation) and are
    /// marked with the Export annotation. This is typically used for declaring
    /// external library functions or functions from other modules.
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, Type, PrimitiveType, FunctionParameter};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Declare printf from libc
    /// builder.external_function(
    ///     "printf",
    ///     vec![FunctionParameter { name: "format", ty: Type::Primitive(PrimitiveType::Ptr), annotations: vec![] }],
    ///     Type::Primitive(PrimitiveType::I32)
    /// );
    /// ```
    pub fn external_function(
        &mut self,
        name: &'a str,
        params: Vec<FunctionParameter<'a>>,
        return_type: Type<'a>,
    ) -> &mut Self {
        let signature = FunctionSignature {
            params,
            return_type,
        };

        self.function_signatures.insert(name, signature);
        self.function_blocks.insert(name, HashMap::new());
        self.function_annotations
            .insert(name, vec![FunctionAnnotation::Extern]);
        self.current_function = Some(name);

        self
    }
}

//
// VALUE FACTORY FUNCTIONS
//
// These utility functions create Value objects for common types.
// They make IR building code cleaner and more readable.
//

/// Creates a variable reference
///
/// Parameter:
/// - `name`: Name of the variable (without % prefix)
///
/// Example: `var("result")` → `%result`
pub fn var(name: &str) -> Value<'_> {
    Value::Variable(name)
}

/// Creates an i8 constant
///
/// Parameter:
/// - `val`: The i8 value
///
/// Example: `i8(42)` → `42`
pub fn i8<'a>(val: i8) -> Value<'a> {
    Value::Constant(Literal::I8(val))
}

/// Creates an i32 constant
///
/// Parameter:
/// - `val`: The i32 value
///
/// Example: `i32(42)` → `42`
pub fn i32<'a>(val: i32) -> Value<'a> {
    Value::Constant(Literal::I32(val))
}

/// Creates an i64 constant
///
/// Parameter:
/// - `val`: The i64 value
///
/// Example: `i64(42)` → `42`
pub fn i64<'a>(val: i64) -> Value<'a> {
    Value::Constant(Literal::I64(val))
}

/// Creates an f32 constant
///
/// Parameter:
/// - `val`: The f32 value
///
/// Example: `f32(3.14)` → `3.14`
pub fn f32<'a>(val: f32) -> Value<'a> {
    Value::Constant(Literal::F32(val))
}

/// Creates a boolean constant
///
/// Parameter:
/// - `val`: The boolean value
///
/// Example: `bool(true)` → `true`
pub fn bool<'a>(val: bool) -> Value<'a> {
    Value::Constant(Literal::Bool(val))
}

/// Creates a string constant
///
/// Parameter:
/// - `val`: The string value
///
/// Example: `string("hello")` → `"hello"`
pub fn string<'a>(val: &'a str) -> Value<'a> {
    Value::Constant(Literal::String(val))
}

/// Creates a global reference
///
/// Parameter:
/// - `name`: Name of the global (without @ prefix)
///
/// Example: `global("message")` → `@message`
pub fn global(name: &str) -> Value<'_> {
    Value::Global(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_function() {
        let mut builder = IRBuilder::new();

        // Build a simple function that adds two integers
        builder
            .function("add", Type::Primitive(PrimitiveType::I32))
            .annotate(FunctionAnnotation::Inline)
            .binary(
                BinaryOp::Add,
                "result",
                PrimitiveType::I32,
                var("a"),
                i32(5),
            )
            .ret(Type::Primitive(PrimitiveType::I32), var("result"));

        let module = builder.build();

        // Assert that the module contains our function
        assert!(module.functions.contains_key("add"));

        // Assert function properties
        let func = &module.functions["add"];
        assert_eq!(func.name, "add");
        assert_eq!(
            func.signature.return_type,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.annotations.len(), 1);
        assert!(matches!(func.annotations[0], FunctionAnnotation::Inline));

        // Assert block properties
        assert!(func.basic_blocks.contains_key("entry"));
        let entry_block = &func.basic_blocks["entry"];
        assert_eq!(entry_block.instructions.len(), 2);

        // Check first instruction is a binary add
        if let Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } = &entry_block.instructions[0]
        {
            assert!(matches!(op, BinaryOp::Add));
            assert_eq!(*result, "result");
            assert_eq!(*ty, PrimitiveType::I32);
            assert_eq!(*lhs, var("a"));
            assert_eq!(*rhs, i32(5));
        } else {
            panic!("Expected Binary instruction");
        }

        // Check second instruction is a return
        if let Instruction::Ret { ty, value } = &entry_block.instructions[1] {
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*value, Some(var("result")));
        } else {
            panic!("Expected Ret instruction");
        }
    }

    #[test]
    fn test_build_func_with_multiple_blocks() {
        let mut builder = IRBuilder::new();

        builder
            .function("conditional", Type::Primitive(PrimitiveType::I32))
            .cmp(CmpOp::Lt, "is_less", PrimitiveType::I32, var("x"), i32(10))
            .branch(var("is_less"), "then_block", "else_block")
            .block("then_block")
            .ret(Type::Primitive(PrimitiveType::I32), i32(1))
            .block("else_block")
            .ret(Type::Primitive(PrimitiveType::I32), i32(2));

        let module = builder.build();
        let func = &module.functions["conditional"];

        // Assert blocks exist
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("then_block"));
        assert!(func.basic_blocks.contains_key("else_block"));

        // Assert block contents
        let entry = &func.basic_blocks["entry"];
        let then_block = &func.basic_blocks["then_block"];
        let else_block = &func.basic_blocks["else_block"];

        assert_eq!(entry.instructions.len(), 2);
        assert_eq!(then_block.instructions.len(), 1);
        assert_eq!(else_block.instructions.len(), 1);

        // Check that branches point to the right blocks
        if let Instruction::Br {
            true_label,
            false_label,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*true_label, "then_block");
            assert_eq!(*false_label, "else_block");
        } else {
            panic!("Expected Br instruction");
        }
    }

    #[test]
    fn test_alloc_store_load_pattern() {
        let mut builder = IRBuilder::new();

        builder
            .function("use_memory", Type::Primitive(PrimitiveType::I32))
            .alloc_stack("ptr", Type::Primitive(PrimitiveType::I32))
            .store(Type::Primitive(PrimitiveType::I32), var("ptr"), i32(42))
            .load("val", Type::Primitive(PrimitiveType::I32), var("ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("val"));

        let module = builder.build();
        let func = &module.functions["use_memory"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 4);

        // Check alloc
        if let Instruction::Alloc {
            result,
            alloc_type,
            allocated_ty,
        } = &entry.instructions[0]
        {
            assert_eq!(*result, "ptr");
            assert!(matches!(alloc_type, AllocType::Stack));
            assert_eq!(*allocated_ty, Type::Primitive(PrimitiveType::I32));
        } else {
            panic!("Expected Alloc instruction");
        }

        // Check store
        if let Instruction::Store { ty, ptr, value } = &entry.instructions[1] {
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*ptr, var("ptr"));
            assert_eq!(*value, i32(42));
        } else {
            panic!("Expected Store instruction");
        }

        // Check load
        if let Instruction::Load { result, ty, ptr } = &entry.instructions[2] {
            assert_eq!(*result, "val");
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*ptr, var("ptr"));
        } else {
            panic!("Expected Load instruction");
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_simd_operations() {
        let mut module = Module::new();
        let mut func = Function::new("test_simd");
        let mut builder = IRBuilder::new(&mut func);

        // Create a SIMD vector type (4 floats)
        let vector_type = Type::Vector {
            element_type: PrimitiveType::F32,
            length: 4,
        };

        // SIMD binary operation
        builder.simd_binary(
            SimdOp::Add,
            "result1",
            vector_type.clone(),
            Value::Register("vec1".into()),
            Value::Register("vec2".into()),
        );

        // SIMD unary operation
        builder.simd_unary(
            SimdOp::Sqrt,
            "result2",
            vector_type.clone(),
            Value::Register("vec3".into()),
        );

        // SIMD extract
        builder.simd_extract(
            "scalar",
            PrimitiveType::F32,
            Value::Register("vec4".into()),
            Value::Literal(Literal::Int(0)),
        );

        // SIMD insert
        builder.simd_insert(
            "result3",
            vector_type.clone(),
            Value::Register("vec5".into()),
            Value::Register("scalar".into()),
            Value::Literal(Literal::Int(1)),
        );

        // SIMD load
        builder.simd_load(
            "loaded_vec",
            vector_type.clone(),
            Value::Register("ptr".into()),
            Some(16),
        );

        // SIMD store
        builder.simd_store(
            vector_type,
            Value::Register("ptr".into()),
            Value::Register("vec_to_store".into()),
            Some(16),
        );

        module.add_function(func);
        assert_eq!(module.functions().len(), 1);
        let func = &module.functions()[0];
        assert_eq!(func.basic_blocks[0].instructions.len(), 6);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_module_annotations() {
        let mut builder = IRBuilder::new();

        // Add various module annotations
        builder
            .annotate_module(ModuleAnnotation::PositionIndependentCode)
            .annotate_module(ModuleAnnotation::OptimizeForSpeed)
            .annotate_module(ModuleAnnotation::IncludeDebugInfo)
            .annotate_module(ModuleAnnotation::TargetTriple(
                "x86_64-unknown-linux-gnu".to_string(),
            ));

        // Also test the convenience methods
        builder
            .pic()
            .pie()
            .optimize_for_size()
            .strip_symbols()
            .target_triple("aarch64-apple-darwin");

        let module = builder.build();

        // Check that all annotations were added
        assert_eq!(module.annotations.len(), 9);

        // Check specific annotations
        assert!(
            module
                .annotations
                .contains(&ModuleAnnotation::PositionIndependentCode)
        );
        assert!(
            module
                .annotations
                .contains(&ModuleAnnotation::PositionIndependentExecutable)
        );
        assert!(
            module
                .annotations
                .contains(&ModuleAnnotation::OptimizeForSpeed)
        );
        assert!(
            module
                .annotations
                .contains(&ModuleAnnotation::OptimizeForSize)
        );
        assert!(
            module
                .annotations
                .contains(&ModuleAnnotation::IncludeDebugInfo)
        );
        assert!(module.annotations.contains(&ModuleAnnotation::StripSymbols));
        assert!(module.annotations.contains(&ModuleAnnotation::TargetTriple(
            "x86_64-unknown-linux-gnu".to_string()
        )));
        assert!(module.annotations.contains(&ModuleAnnotation::TargetTriple(
            "aarch64-apple-darwin".to_string()
        )));

        // Test Display implementation
        assert_eq!(
            format!("{}", ModuleAnnotation::PositionIndependentCode),
            "@pic"
        );
        assert_eq!(
            format!("{}", ModuleAnnotation::OptimizeForSpeed),
            "@optimize_speed"
        );
        assert_eq!(
            format!("{}", ModuleAnnotation::TargetTriple("test".to_string())),
            "@target_triple(test)"
        );
    }
}
