use std::collections::HashMap;

use super::types::{Type, PrimitiveType, Value, Literal};
use super::instruction::{Instruction, BinaryOp, CmpOp, AllocType};
use super::function::{Function, FunctionSignature, FunctionParameter, BasicBlock, FunctionAnnotation};
use super::module::Module;

/// Builder for creating IR test cases programmatically
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

impl<'a> IRBuilder<'a> {
    /// Create a new IR builder
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

    /// Get a fresh temporary variable name
    pub fn temp_var(&mut self) -> String {
        let var = format!("temp_{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        var
    }

    /// Start a new function definition
    pub fn function(&mut self, name: &'a str, return_type: Type<'a>) -> &mut Self {
        self.function_with_params(name, vec![], return_type)
    }

    /// Start a new function definition with parameters
    pub fn function_with_params(
        &mut self, 
        name: &'a str, 
        params: Vec<FunctionParameter<'a>>, 
        return_type: Type<'a>
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
    
    /// Add an annotation to the current function
    pub fn annotate(&mut self, annotation: FunctionAnnotation) -> &mut Self {
        if let Some(func_name) = self.current_function {
            if let Some(annotations) = self.function_annotations.get_mut(func_name) {
                annotations.push(annotation);
            }
        }
        self
    }
    
    /// Start a new basic block in the current function
    pub fn block(&mut self, name: &'a str) -> &mut Self {
        if self.current_function.is_some() {
            self.current_block = Some(name);
            self.block_instructions.insert(name, vec![]);
        }
        self
    }
    
    /// Set the entry block for the current function
    pub fn set_entry_block(&mut self, name: &'a str) -> &mut Self {
        if let Some(func_name) = self.current_function {
            self.function_entry_blocks.insert(func_name, name);
        }
        self
    }
    
    /// Add an instruction to the current block
    pub fn inst(&mut self, instruction: Instruction<'a>) -> &mut Self {
        if let (Some(_func_name), Some(block_name)) = (self.current_function, self.current_block) {
            if let Some(instructions) = self.block_instructions.get_mut(block_name) {
                instructions.push(instruction);
            }
        }
        self
    }
    
    /// Add an alloc instruction for stack memory
    pub fn alloc_stack(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Stack,
            allocated_ty: ty,
        })
    }
    
    /// Add an alloc instruction for heap memory
    pub fn alloc_heap(&mut self, result: &'a str, ty: Type<'a>) -> &mut Self {
        self.inst(Instruction::Alloc {
            result,
            alloc_type: AllocType::Heap,
            allocated_ty: ty,
        })
    }
    
    /// Add a store instruction
    pub fn store(&mut self, ty: Type<'a>, ptr: Value<'a>, val: Value<'a>) -> &mut Self {
        self.inst(Instruction::Store {
            ty,
            ptr,
            value: val,
        })
    }
    
    /// Add a load instruction
    pub fn load(&mut self, result: &'a str, ty: Type<'a>, ptr: Value<'a>) -> &mut Self {
        self.inst(Instruction::Load {
            result,
            ty,
            ptr,
        })
    }
    
    /// Add a binary operation instruction
    pub fn binary(&mut self, op: BinaryOp, result: &'a str, ty: PrimitiveType, 
                 lhs: Value<'a>, rhs: Value<'a>) -> &mut Self {
        self.inst(Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }
    
    /// Add a comparison instruction
    pub fn cmp(&mut self, op: CmpOp, result: &'a str, ty: PrimitiveType,
              lhs: Value<'a>, rhs: Value<'a>) -> &mut Self {
        self.inst(Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        })
    }
    
    /// Add a conditional branch instruction
    pub fn branch(&mut self, condition: Value<'a>, true_label: &'a str, false_label: &'a str) -> &mut Self {
        self.inst(Instruction::Br {
            condition,
            true_label,
            false_label,
        })
    }
    
    /// Add an unconditional jump instruction
    pub fn jump(&mut self, target: &'a str) -> &mut Self {
        self.inst(Instruction::Jmp {
            target_label: target,
        })
    }
    
    /// Add a return instruction with a value
    pub fn ret(&mut self, ty: Type<'a>, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Ret {
            ty,
            value: Some(value),
        })
    }
    
    /// Add a void return instruction
    pub fn ret_void(&mut self) -> &mut Self {
        self.inst(Instruction::Ret {
            ty: Type::Void,
            value: None,
        })
    }
    
    /// Add a zero extension instruction
    pub fn zext(&mut self, result: &'a str, source_type: PrimitiveType, 
               target_type: PrimitiveType, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        })
    }
    
    /// Add a GetElemPtr instruction
    pub fn getelementptr(&mut self, result: &'a str, array_ptr: Value<'a>, index: Value<'a>) -> &mut Self {
        self.inst(Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
        })
    }
    
    /// Add a call instruction
    pub fn call(&mut self, result: Option<&'a str>, func_name: &'a str, args: Vec<Value<'a>>) -> &mut Self {
        self.inst(Instruction::Call {
            result,
            func_name,
            args,
        })
    }
    
    /// Add a print instruction
    pub fn print(&mut self, value: Value<'a>) -> &mut Self {
        self.inst(Instruction::Print {
            value,
        })
    }
    
    /// Build a complete IR module
    pub fn build(&mut self) -> Module<'a> {
        // Finalize the module by converting all block instructions to basicblocks
        // and all functions to their final representation
        
        for (func_name, block_map) in &mut self.function_blocks {
            // Convert block instructions to BasicBlocks
            for (block_name, instructions) in self.block_instructions.iter() {
                // Only process blocks for current function
                if self.block_instructions.contains_key(block_name) {
                    block_map.insert(block_name, BasicBlock {
                        instructions: instructions.clone(),
                    });
                }
            }
            
            // Create the function and add it to the module
            if let (Some(signature), Some(entry_block)) = (
                self.function_signatures.get(func_name),
                self.function_entry_blocks.get(func_name)
            ) {
                let function = Function {
                    name: func_name,
                    signature: signature.clone(),
                    annotations: self.function_annotations.get(func_name)
                        .cloned().unwrap_or_else(Vec::new),
                    basic_blocks: block_map.clone(),
                    entry_block,
                };
                
                self.module.functions.insert(func_name, function);
            }
        }
        
        self.module.clone()
    }
}

// Factory functions for creating common IR values
pub fn var<'a>(name: &'a str) -> Value<'a> {
    Value::Variable(name)
}

pub fn i32<'a>(val: i32) -> Value<'a> {
    Value::Constant(Literal::I32(val))
}

pub fn i64<'a>(val: i64) -> Value<'a> {
    Value::Constant(Literal::I64(val))
}

pub fn bool<'a>(val: bool) -> Value<'a> {
    Value::Constant(Literal::Bool(val))
}

pub fn global<'a>(name: &'a str) -> Value<'a> {
    Value::Global(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_simple_function() {
        let mut builder = IRBuilder::new();
        
        // Build a simple function that adds two integers
        builder.function("add", Type::Primitive(PrimitiveType::I32))
            .annotate(FunctionAnnotation::Inline)
            .binary(BinaryOp::Add, "result", PrimitiveType::I32, 
                   var("a"), i32(5))
            .ret(Type::Primitive(PrimitiveType::I32), var("result"));
        
        let module = builder.build();
        
        // Assert that the module contains our function
        assert!(module.functions.contains_key("add"));
        
        // Assert function properties
        let func = &module.functions["add"];
        assert_eq!(func.name, "add");
        assert_eq!(func.signature.return_type, Type::Primitive(PrimitiveType::I32));
        assert_eq!(func.annotations.len(), 1);
        assert!(matches!(func.annotations[0], FunctionAnnotation::Inline));
        
        // Assert block properties
        assert!(func.basic_blocks.contains_key("entry"));
        let entry_block = &func.basic_blocks["entry"];
        assert_eq!(entry_block.instructions.len(), 2);
        
        // Check first instruction is a binary add
        if let Instruction::Binary { op, result, ty, lhs, rhs } = &entry_block.instructions[0] {
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
        
        builder.function("conditional", Type::Primitive(PrimitiveType::I32))
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
        if let Instruction::Br { true_label, false_label, .. } = &entry.instructions[1] {
            assert_eq!(*true_label, "then_block");
            assert_eq!(*false_label, "else_block");
        } else {
            panic!("Expected Br instruction");
        }
    }
    
    #[test]
    fn test_alloc_store_load_pattern() {
        let mut builder = IRBuilder::new();
        
        builder.function("use_memory", Type::Primitive(PrimitiveType::I32))
            .alloc_stack("ptr", Type::Primitive(PrimitiveType::I32))
            .store(Type::Primitive(PrimitiveType::I32), var("ptr"), i32(42))
            .load("val", Type::Primitive(PrimitiveType::I32), var("ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("val"));
        
        let module = builder.build();
        let func = &module.functions["use_memory"];
        let entry = &func.basic_blocks["entry"];
        
        assert_eq!(entry.instructions.len(), 4);
        
        // Check alloc
        if let Instruction::Alloc { result, alloc_type, allocated_ty } = &entry.instructions[0] {
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