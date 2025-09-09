use std::collections::HashMap;

use super::function::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature,
};
use super::instruction::{AllocType, BinaryOp, CmpOp, Instruction};
use super::module::Module;
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
///             FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) },
///             FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32) },
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
///     .getelementptr("elem_ptr", var("arr"), i32(5))
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
    ///         FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I32) },
    ///         FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I32) },
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
    /// - `op`: The binary operation (Add, Sub, Mul, Div)
    /// - `result`: Name for the result variable
    /// - `ty`: Primitive type of the operands and result
    /// - `lhs`: Left-hand side operand
    /// - `rhs`: Right-hand side operand
    ///
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, BinaryOp, PrimitiveType};
    /// use lamina::ir::builder::var;
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
    /// Example:
    /// ```
    /// use lamina::ir::{IRBuilder, CmpOp, PrimitiveType};
    /// use lamina::ir::builder::{var, i32};
    ///
    /// let mut builder = IRBuilder::new();
    /// // Branch based on comparison result
    /// builder
    ///     .cmp(CmpOp::Lt, "is_neg", PrimitiveType::I32, var("x"), i32(0))
    ///     .branch(var("is_neg"), "negative", "non_negative");
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
    /// Example:
    /// ```
    /// use lamina::ir::IRBuilder;
    ///
    /// let mut builder = IRBuilder::new();
    /// // Jump to the "end" block
    /// builder.jump("end");
    /// ```
    pub fn jump(&mut self, target: &'a str) -> &mut Self {
        self.inst(Instruction::Jmp {
            target_label: target,
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
    ///     .getelementptr("elem_ptr", var("arr"), i32(5))
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
    /// use lamina::ir::{IRBuilder, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i64};
    ///
    /// let mut builder = IRBuilder::new();
    ///
    /// // Main function demonstrating pointer arithmetic
    /// builder
    ///     .function("pointer_arithmetic_demo", PrimitiveType::I64)
    ///     .block("entry")
    ///     // Allocate an array of 5 i64 elements
    ///     .alloc_stack("arr", PrimitiveType::I64, 5)
    ///     // Get pointer to first element
    ///     .getelementptr("base_ptr", var("arr"), i64(0), PrimitiveType::I64)
    ///     // Store values in the array
    ///     .store_i64(var("base_ptr"), i64(10))
    ///     .getelementptr("ptr1", var("arr"), i64(1), PrimitiveType::I64)
    ///     .store_i64(var("ptr1"), i64(20))
    ///     .getelementptr("ptr2", var("arr"), i64(2), PrimitiveType::I64)
    ///     .store_i64(var("ptr2"), i64(30))
    ///     // Convert base pointer to integer for arithmetic
    ///     .ptrtoint("base_addr", var("base_ptr"), PrimitiveType::I64)
    ///     // Calculate address of element at index 3 (8 bytes * 3 = 24 bytes offset)
    ///     .binary(BinaryOp::Add, PrimitiveType::I64, "target_addr", var("base_addr"), i64(24))
    ///     // Convert back to pointer
    ///     .inttoptr("target_ptr", var("target_addr"), PrimitiveType::Ptr)
    ///     // Store value at computed location
    ///     .store_i64(var("target_ptr"), i64(40))
    ///     // Load it back to verify
    ///     .load_i64("result", var("target_ptr"))
    ///     .ret(PrimitiveType::I64, var("result"));
    ///
    /// // Helper function to print numbers (essential for debugging)
    /// builder
    ///     .function("print_number", PrimitiveType::I64)
    ///     .param("num", PrimitiveType::I64)
    ///     .block("entry")
    ///     // Check if number is zero
    ///     .binary(BinaryOp::Equal, PrimitiveType::I64, "is_zero", var("num"), i64(0))
    ///     .br("is_zero", "print_zero", "check_negative")
    ///     .block("print_zero")
    ///     .write_byte(i64(48)) // ASCII '0'
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("check_negative")
    ///     .binary(BinaryOp::LessThan, PrimitiveType::I64, "is_negative", var("num"), i64(0))
    ///     .br("is_negative", "handle_negative", "print_digits")
    ///     .block("handle_negative")
    ///     .write_byte(i64(45)) // ASCII '-'
    ///     .binary(BinaryOp::Subtract, PrimitiveType::I64, "abs_num", i64(0), var("num"))
    ///     .call("dummy", "print_digits", vec![var("abs_num")], PrimitiveType::I64)
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("print_digits")
    ///     .call("dummy", "print_digits", vec![var("num")], PrimitiveType::I64)
    ///     .ret(PrimitiveType::I64, i64(0));
    ///
    /// // Recursive helper for digit printing
    /// builder
    ///     .function("print_digits", PrimitiveType::I64)
    ///     .param("num", PrimitiveType::I64)
    ///     .block("entry")
    ///     .binary(BinaryOp::Equal, PrimitiveType::I64, "is_zero", var("num"), i64(0))
    ///     .br("is_zero", "done", "continue_print")
    ///     .block("continue_print")
    ///     .binary(BinaryOp::Divide, PrimitiveType::I64, "quotient", var("num"), i64(10))
    ///     .binary(BinaryOp::Multiply, PrimitiveType::I64, "temp", var("quotient"), i64(10))
    ///     .binary(BinaryOp::Subtract, PrimitiveType::I64, "remainder", var("num"), var("temp"))
    ///     .call("dummy", "print_digits", vec![var("quotient")], PrimitiveType::I64)
    ///     .binary(BinaryOp::Add, PrimitiveType::I64, "digit", var("remainder"), i64(48))
    ///     .write_byte(var("digit"))
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("done")
    ///     .ret(PrimitiveType::I64, i64(0));
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
    /// use lamina::ir::{IRBuilder, PrimitiveType, BinaryOp};
    /// use lamina::ir::builder::{var, i64};
    ///
    /// let mut builder = IRBuilder::new();
    ///
    /// // Function demonstrating dynamic memory access with pointer arithmetic
    /// builder
    ///     .function("brainfuck_cell_access", PrimitiveType::I64)
    ///     .block("entry")
    ///     // Simulate Brainfuck tape (array of bytes)
    ///     .alloc_stack("tape", PrimitiveType::I8, 10)
    ///     // Initialize some cells
    ///     .getelementptr("cell0", var("tape"), i64(0), PrimitiveType::I8)
    ///     .store_i8(var("cell0"), i64(65)) // 'A'
    ///     .getelementptr("cell5", var("tape"), i64(5), PrimitiveType::I8)
    ///     .store_i8(var("cell5"), i64(66)) // 'B'
    ///     // Simulate Brainfuck pointer movement to cell 5
    ///     .ptrtoint("base_addr", var("cell0"), PrimitiveType::I64)
    ///     .binary(BinaryOp::Add, PrimitiveType::I64, "target_addr", var("base_addr"), i64(5))
    ///     .inttoptr("data_ptr", var("target_addr"), PrimitiveType::Ptr)
    ///     // Load value from dynamically computed location
    ///     .load_i8("value", var("data_ptr"))
    ///     .zext_i8_to_i64("result", var("value"))
    ///     .ret(PrimitiveType::I64, var("result"));
    ///
    /// // Helper function to print numbers (essential for debugging pointer operations)
    /// builder
    ///     .function("print_number", PrimitiveType::I64)
    ///     .param("num", PrimitiveType::I64)
    ///     .block("entry")
    ///     // Handle zero case
    ///     .binary(BinaryOp::Equal, PrimitiveType::I64, "is_zero", var("num"), i64(0))
    ///     .br("is_zero", "print_zero", "check_negative")
    ///     .block("print_zero")
    ///     .write_byte(i64(48)) // ASCII '0'
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("check_negative")
    ///     .binary(BinaryOp::LessThan, PrimitiveType::I64, "is_negative", var("num"), i64(0))
    ///     .br("is_negative", "handle_negative", "print_digits")
    ///     .block("handle_negative")
    ///     .write_byte(i64(45)) // ASCII '-'
    ///     .binary(BinaryOp::Subtract, PrimitiveType::I64, "abs_num", i64(0), var("num"))
    ///     .call("dummy", "print_digits", vec![var("abs_num")], PrimitiveType::I64)
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("print_digits")
    ///     .call("dummy", "print_digits", vec![var("num")], PrimitiveType::I64)
    ///     .ret(PrimitiveType::I64, i64(0));
    ///
    /// // Recursive digit printing helper
    /// builder
    ///     .function("print_digits", PrimitiveType::I64)
    ///     .param("num", PrimitiveType::I64)
    ///     .block("entry")
    ///     .binary(BinaryOp::Equal, PrimitiveType::I64, "is_zero", var("num"), i64(0))
    ///     .br("is_zero", "done", "continue_print")
    ///     .block("continue_print")
    ///     .binary(BinaryOp::Divide, PrimitiveType::I64, "quotient", var("num"), i64(10))
    ///     .binary(BinaryOp::Multiply, PrimitiveType::I64, "temp", var("quotient"), i64(10))
    ///     .binary(BinaryOp::Subtract, PrimitiveType::I64, "remainder", var("num"), var("temp"))
    ///     .call("dummy", "print_digits", vec![var("quotient")], PrimitiveType::I64)
    ///     .binary(BinaryOp::Add, PrimitiveType::I64, "digit", var("remainder"), i64(48))
    ///     .write_byte(var("digit"))
    ///     .ret(PrimitiveType::I64, i64(0))
    ///     .block("done")
    ///     .ret(PrimitiveType::I64, i64(0));
    /// ```
    ///
    /// # Note on Removed ptradd Instruction
    ///
    /// Initially, a `ptradd` instruction was considered for direct pointer+offset arithmetic.
    /// However, this instruction was removed as unnecessary because the same functionality
    /// can be achieved more efficiently using the `ptrtoint` + `inttoptr` sequence:
    ///
    /// ```rust
    /// // Instead of: ptradd result, base_ptr, offset
    /// // Use:
    /// .ptrtoint("temp", base_ptr, PrimitiveType::I64)
    /// .binary(BinaryOp::Add, PrimitiveType::I64, "new_addr", var("temp"), offset)
    /// .inttoptr("result", var("new_addr"), PrimitiveType::Ptr)
    /// ```
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
    ///     .read_ptr("input_data")
    ///     // Use the data (e.g., store it somewhere)
    ///     .alloc_stack("storage", Type::Primitive(PrimitiveType::I64))
    ///     .store(Type::Primitive(PrimitiveType::I64), var("storage"), var("input_data"))
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
    /// - `write_ptr()`: Write value to stdout
    pub fn read_ptr(&mut self, result: &'a str) -> &mut Self {
        self.inst(Instruction::ReadPtr { result })
    }

    /// Creates a print instruction for debugging
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
    ///     vec![FunctionParameter { name: "format", ty: Type::Primitive(PrimitiveType::Ptr) }],
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
            .insert(name, vec![FunctionAnnotation::Export]);
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
}
