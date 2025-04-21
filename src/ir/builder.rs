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
/// ## Method Organization
///
/// The builder methods are organized into several categories:
///
/// - **Function definition**: `function()`, `function_with_params()`, `annotate()`
/// - **Block management**: `block()`, `set_entry_block()`
/// - **Memory operations**: `alloc_stack()`, `alloc_heap()`, `load()`, `store()`, `dealloc()`
/// - **Arithmetic**: `binary()`, `cmp()`
/// - **Control flow**: `branch()`, `jump()`, `ret()`, `ret_void()`
/// - **Pointers**: `getelementptr()`, `struct_gep()`
/// - **Function calls**: `call()`
/// - **SSA form**: `phi()`
/// - **Helper methods**: `temp_var()`
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
    /// builder
    ///     .function("fast_func", Type::Void)
    ///     .annotate(FunctionAnnotation::Inline);
    /// ```
    pub fn annotate(&mut self, annotation: FunctionAnnotation) -> &mut Self {
        if let Some(func_name) = self.current_function {
            if let Some(annotations) = self.function_annotations.get_mut(func_name) {
                annotations.push(annotation);
            }
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
        if let (Some(_func_name), Some(block_name)) = (self.current_function, self.current_block) {
            if let Some(instructions) = self.block_instructions.get_mut(block_name) {
                instructions.push(instruction);
            }
        }
        self
    }

    /// Allocates stack memory
    ///
    /// Parameters:
    /// - `result`: Name for the pointer variable
    /// - `ty`: Type of the allocated memory
    ///
    /// Creates a stack allocation instruction. The result will be a pointer
    /// to memory of the specified type.
    ///
    /// Example:
    /// ```
    /// // Allocate an i32 on the stack
    /// builder.alloc_stack("ptr", Type::Primitive(PrimitiveType::I32));
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
    /// Parameters:
    /// - `result`: Name for the pointer variable
    /// - `ty`: Type of the allocated memory
    ///
    /// Creates a heap allocation instruction. The result will be a pointer
    /// to memory of the specified type. Remember to deallocate with `dealloc`
    /// to avoid memory leaks.
    ///
    /// Example:
    /// ```
    /// // Allocate a struct on the heap
    /// builder.alloc_heap("obj_ptr", struct_type);
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

    /// Gets a pointer to an array element
    ///
    /// Parameters:
    /// - `result`: Name for the resulting pointer
    /// - `array_ptr`: Pointer to the array
    /// - `index`: Index value (must be integer type)
    ///
    /// Computes the address of an element within an array. The index is 0-based.
    ///
    /// Example:
    /// ```
    /// // Get pointer to the third element (index 2)
    /// builder.getelementptr("elem_ptr", var("array_ptr"), i32(2));
    /// ```
    pub fn getelementptr(
        &mut self,
        result: &'a str,
        array_ptr: Value<'a>,
        index: Value<'a>,
    ) -> &mut Self {
        self.inst(Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
        })
    }

    /// Gets a pointer to a struct field
    ///
    /// Parameters:
    /// - `result`: Name for the resulting pointer
    /// - `struct_ptr`: Pointer to the struct
    /// - `field_index`: Field index (0-based)
    ///
    /// Computes the address of a field within a struct. The index corresponds
    /// to the order of fields in the struct definition.
    ///
    /// Example:
    /// ```
    /// // Get pointer to the second field (index 1)
    /// builder.struct_gep("field_ptr", var("struct_ptr"), 1);
    /// ```
    pub fn struct_gep(&mut self, result: &'a str, struct_ptr: Value<'a>, field_index: usize) -> &mut Self {
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

    /// Creates a print instruction (for debugging)
    ///
    /// Parameters:
    /// - `value`: Value to print
    ///
    /// A utility instruction for debugging IR code.
    ///
    /// Example:
    /// ```
    /// // Print the value of x
    /// builder.print(var("x"));
    /// ```
    pub fn print(&mut self, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Print { value })
    }

    /// Creates a tuple from elements
    ///
    /// Parameters:
    /// - `result`: Name for the tuple variable
    /// - `elements`: Vector of values to put in the tuple
    ///
    /// Example:
    /// ```
    /// // Create a tuple of (42, true)
    /// builder.tuple("my_tuple", vec![i32(42), bool(true)]);
    /// ```
    pub fn tuple(&mut self, result: &'a str, elements: Vec<Value<'a>>) -> &mut Self {
        self.inst(Instruction::Tuple {
            result,
            elements,
        })
    }

    /// Extracts a value from a tuple
    ///
    /// Parameters:
    /// - `result`: Name for the extracted value
    /// - `tuple_val`: Tuple to extract from
    /// - `index`: Index of the element to extract (0-based)
    ///
    /// Example:
    /// ```
    /// // Extract the first element (index 0) from a tuple
    /// builder.extract_tuple("first", var("my_tuple"), 0);
    /// ```
    pub fn extract_tuple(&mut self, result: &'a str, tuple_val: Value<'a>, index: usize) -> &mut Self {
        self.inst(Instruction::ExtractTuple {
            result,
            tuple_val,
            index,
        })
    }

    /// Deallocates heap memory
    ///
    /// Parameters:
    /// - `ptr`: Pointer to the memory to free
    ///
    /// Use this to free memory previously allocated with `alloc_heap`.
    ///
    /// Example:
    /// ```
    /// // Free previously allocated memory
    /// builder.dealloc(var("heap_ptr"));
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
        self.function_annotations.insert(name, vec![FunctionAnnotation::Export]);
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
