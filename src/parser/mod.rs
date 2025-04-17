pub mod parser;

// Re-export the main entry point from parser.rs
pub use parser::parse_module;


#[cfg(test)]
mod tests {
    use super::parse_module;
    use crate::{Type, PrimitiveType, Value, Literal, Instruction, BinaryOp, CmpOp, AllocType};

    #[test]
    fn test_parse_simple_add_function() {
        let input = r#"
            fn @add(i32 %a, i32 %b) -> i32 {
              entry:
                %sum = add.i32 %a, %b
                ret.i32 %sum
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("add").expect("Function @add not found");

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.params[0].name, "a");
        assert_eq!(func.signature.params[0].ty, Type::Primitive(PrimitiveType::I32));
        assert_eq!(func.signature.params[1].name, "b");
        assert_eq!(func.signature.params[1].ty, Type::Primitive(PrimitiveType::I32));
        assert_eq!(func.signature.return_type, Type::Primitive(PrimitiveType::I32));
        assert_eq!(func.basic_blocks.len(), 1);
        assert_eq!(func.entry_block, "entry");

        let entry_block = func.basic_blocks.get("entry").expect("Entry block not found");
        assert_eq!(entry_block.instructions.len(), 2);

        match &entry_block.instructions[0] {
            Instruction::Binary {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*result, "sum");
                assert_eq!(*ty, PrimitiveType::I32);
                assert_eq!(*lhs, Value::Variable("a"));
                assert_eq!(*rhs, Value::Variable("b"));
            }
            _ => panic!("Unexpected instruction: {:?}", entry_block.instructions[0]),
        }

        match &entry_block.instructions[1] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                assert_eq!(value.as_ref(), Some(&Value::Variable("sum")));
            }
            _ => panic!("Unexpected instruction: {:?}", entry_block.instructions[1]),
        }
    }

    #[test]
    fn test_parse_type_declarations() {
        let input = r#"
            type @Vec2 = struct { x: f32, y: f32 }
            type @Matrix = [4 x i32]
        "#;
        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.type_declarations.len(), 2);
        
        let vec2 = module.type_declarations.get("Vec2").unwrap();
        assert_eq!(vec2.name, "Vec2");
        if let Type::Struct(fields) = &vec2.ty {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "x");
            assert_eq!(fields[0].ty, Type::Primitive(PrimitiveType::F32));
            assert_eq!(fields[1].name, "y");
            assert_eq!(fields[1].ty, Type::Primitive(PrimitiveType::F32));
        } else {
            panic!("Expected struct for Vec2");
        }

        let matrix = module.type_declarations.get("Matrix").unwrap();
        assert_eq!(matrix.name, "Matrix");
        if let Type::Array{ element_type, size } = &matrix.ty {
            assert_eq!(*size, 4);
            assert_eq!(element_type.as_ref(), &Type::Primitive(PrimitiveType::I32));
        } else {
            panic!("Expected array for Matrix");
        }
    }

    #[test]
    fn test_parse_global_declarations() {
        let input = r#"
            global @count: i64 = 10
            global @message: [12 x bool] = "hello world"
            global @uninit_ptr: ptr
        "#;
        let module = parse_module(input).expect("Parsing failed");

        assert_eq!(module.global_declarations.len(), 3);

        let count = module.global_declarations.get("count").unwrap();
        assert_eq!(count.name, "count");
        assert_eq!(count.ty, Type::Primitive(PrimitiveType::I64));
        assert_eq!(count.initializer.as_ref(), Some(&Value::Constant(Literal::I64(10))));

        let message = module.global_declarations.get("message").unwrap();
        assert_eq!(message.name, "message");
        if let Type::Array{ element_type, size } = &message.ty {
            assert_eq!(*size, 12);
            assert_eq!(element_type.as_ref(), &Type::Primitive(PrimitiveType::Bool));
        } else {
            panic!("Expected array for message");
        }
        assert_eq!(message.initializer.as_ref(), Some(&Value::Constant(Literal::String("hello world"))));

        let uninit = module.global_declarations.get("uninit_ptr").unwrap();
        assert_eq!(uninit.name, "uninit_ptr");
        assert_eq!(uninit.ty, Type::Primitive(PrimitiveType::Ptr));
        assert!(uninit.initializer.is_none());
    }

    #[test]
    fn test_parse_comments_and_whitespace() {
        let input = r#"
            # This is a comment
            fn @test() -> void { # Another comment
               entry: # Block label comment
                  # Instruction comment
                  ret.void # Trailing comment
            } # Final comment


        "#;
        let module = parse_module(input).expect("Parsing should succeed");
        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test").unwrap();
        assert_eq!(func.basic_blocks.len(), 1);
        let block = func.basic_blocks.get("entry").unwrap();
        assert_eq!(block.instructions.len(), 1);
        assert!(matches!(block.instructions[0], Instruction::Ret{ ty: Type::Void, value: None }));
    }

    #[test]
    fn test_parse_block_missing_terminator() {
        let input = r#"
            fn @bad() -> i32 {
                entry:
                    %a = add.i32 1, 2
            }
        "#;
        let result = parse_module(input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("must end with a terminator instruction"));
        }
    }

    #[test]
    fn test_parse_function_with_multiple_blocks() {
        let input = r#"
            fn @conditional(i32 %x) -> i32 {
              entry:
                %is_pos = gt.i32 %x, 0
                br %is_pos, positive, negative
              
              positive:
                ret.i32 1
                
              negative:
                ret.i32 -1
            }
        "#;
        
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("conditional").expect("Function not found");
        
        assert_eq!(func.basic_blocks.len(), 3);
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("positive"));
        assert!(func.basic_blocks.contains_key("negative"));
        
        let positive = func.basic_blocks.get("positive").unwrap();
        let negative = func.basic_blocks.get("negative").unwrap();
        
        match &positive.instructions[0] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                if let Some(Value::Constant(Literal::I32(val))) = value {
                    assert_eq!(*val, 1);
                } else {
                    panic!("Expected constant 1");
                }
            },
            _ => panic!("Expected ret instruction"),
        }
        
        match &negative.instructions[0] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                if let Some(Value::Constant(Literal::I32(val))) = value {
                    assert_eq!(*val, -1);
                } else {
                    panic!("Expected constant -1");
                }
            },
            _ => panic!("Expected ret instruction"),
        }
    }

    #[test]
    fn test_parse_array_and_struct_operations() {
        let input = r#"
            fn @test_composite() -> ptr {
              entry:
                %arr_ptr = alloc.stack [4 x i32]
                %elem_ptr = getelementptr %arr_ptr, 2
                %elem = load.i32 %elem_ptr
                
                %struct_ptr = alloc.stack struct { x: i32, y: i32 }
                %y_ptr = getfieldptr %struct_ptr, 1
                %y_val = load.i32 %y_ptr
                
                ret.ptr %struct_ptr
            }
        "#;
        
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("test_composite").expect("Function not found");
        
        let entry = func.basic_blocks.get("entry").expect("Entry block not found");
        assert_eq!(entry.instructions.len(), 7); // alloc, getelementptr, load, alloc, getfieldptr, load, ret
        
        // Check first instruction is alloc
        match &entry.instructions[0] {
            Instruction::Alloc { result, allocated_ty, alloc_type } => {
                assert_eq!(*result, "arr_ptr");
                if let Type::Array { element_type, size } = allocated_ty {
                    assert_eq!(**element_type, Type::Primitive(PrimitiveType::I32));
                    assert_eq!(*size, 4);
                } else {
                    panic!("Expected array type");
                }
                assert_eq!(*alloc_type, AllocType::Stack);
            },
            _ => panic!("Expected alloc instruction"),
        }
        
        // Check second instruction is getelementptr
        match &entry.instructions[1] {
            Instruction::GetElemPtr { result, array_ptr, index } => {
                assert_eq!(*result, "elem_ptr");
                assert_eq!(*array_ptr, Value::Variable("arr_ptr"));
                assert_eq!(*index, Value::Constant(Literal::I32(2)));
            },
            _ => panic!("Expected getelementptr instruction"),
        }
    }
    
    #[test]
    fn test_parse_function_call() {
        let input = r#"
            fn @callee(i32 %x, i32 %y) -> i32 {
              entry:
                %sum = add.i32 %x, %y
                ret.i32 %sum
            }
            
            fn @caller() -> i32 {
              entry:
                %result = call @callee(5, 10)
                ret.i32 %result
            }
        "#;
        
        let module = parse_module(input).expect("Parsing failed");
        
        // Check caller function has the correct call instruction
        let caller = module.functions.get("caller").expect("Caller function not found");
        let entry = caller.basic_blocks.get("entry").expect("Entry block not found");
        
        match &entry.instructions[0] {
            Instruction::Call { result, func_name, args } => {
                let result_str: &str = result.as_ref().unwrap();
                let func_str: &str = func_name;
                assert_eq!(result_str, "result");
                assert_eq!(func_str, "callee");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Value::Constant(Literal::I32(5)));
                assert_eq!(args[1], Value::Constant(Literal::I32(10)));
            },
            _ => panic!("Expected call instruction"),
        }
    }
    
    #[test]
    fn test_parse_comparison_operators() {
        let input = r#"
            fn @test_comparisons(i32 %a, i32 %b) -> i32 {
              entry:
                %eq = eq.i32 %a, %b
                %ne = ne.i32 %a, %b
                %lt = lt.i32 %a, %b
                %le = le.i32 %a, %b
                %gt = gt.i32 %a, %b
                %ge = ge.i32 %a, %b
                
                ret.i32 0
            }
        "#;
        
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("test_comparisons").expect("Function not found");
        let entry = func.basic_blocks.get("entry").expect("Entry block not found");
        
        assert_eq!(entry.instructions.len(), 7); // 6 comparisons + ret
        
        match &entry.instructions[0] {
            Instruction::Cmp { op, result, ty, lhs, rhs } => {
                assert_eq!(*op, CmpOp::Eq);
                assert_eq!(*result, "eq");
                assert_eq!(*ty, PrimitiveType::I32);
                assert_eq!(*lhs, Value::Variable("a"));
                assert_eq!(*rhs, Value::Variable("b"));
            },
            _ => panic!("Expected comparison instruction"),
        }
        
        match &entry.instructions[1] {
            Instruction::Cmp { op, result, .. } => {
                assert_eq!(*op, CmpOp::Ne);
                assert_eq!(*result, "ne");
            },
            _ => panic!("Expected comparison instruction"),
        }
    }

    #[test]
    fn test_parse_i8_type_and_zext() {
        let input = r#"
            fn @test_i8() -> i64 {
              entry:
                %i8_val = add.i8 10, 5
                %extended = zext.i8.i64 %i8_val
                ret.i64 %extended
            }
        "#;
        
        let module = parse_module(input).expect("Parsing failed");
        let func = module.functions.get("test_i8").expect("Function not found");
        let entry = func.basic_blocks.get("entry").expect("Entry block not found");
        
        // Check the add operation uses I8 type
        match &entry.instructions[0] {
            Instruction::Binary { op, ty, .. } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*ty, PrimitiveType::I8);
            },
            _ => panic!("Expected Binary instruction"),
        }
        
        // Check the zext operation
        match &entry.instructions[1] {
            Instruction::ZeroExtend { source_type, target_type, .. } => {
                assert_eq!(*source_type, PrimitiveType::I8);
                assert_eq!(*target_type, PrimitiveType::I64);
            },
            _ => panic!("Expected ZeroExtend instruction"),
        }
    }
} 