#[cfg(test)]
mod tests {
    use super::super::super::function::FunctionAnnotation;
    #[cfg(feature = "nightly")]
    use super::super::super::instruction::SimdOp;
    use super::super::super::instruction::{AllocType, BinaryOp, CmpOp, Instruction};
    use super::super::super::module::Module;
    #[cfg(feature = "nightly")]
    use super::super::super::module::ModuleAnnotation;
    use super::super::super::types::{Literal, PrimitiveType, Type, Value};
    use super::super::IRBuilder;
    use super::super::values::*;

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
    }

    #[test]
    fn test_function_annotations() {
        let mut builder = IRBuilder::new();

        builder
            .function("test_func", Type::Void)
            .inline()
            .export()
            .hot()
            .pure()
            .cc_c()
            .section(".text.fast")
            .align(16);

        let module = builder.build();
        let func = &module.functions["test_func"];

        assert!(func.annotations.contains(&FunctionAnnotation::Inline));
        assert!(func.annotations.contains(&FunctionAnnotation::Export));
        assert!(func.annotations.contains(&FunctionAnnotation::Hot));
        assert!(func.annotations.contains(&FunctionAnnotation::Pure));
        assert!(func.annotations.contains(&FunctionAnnotation::CCc));
        assert!(
            func.annotations
                .contains(&FunctionAnnotation::Section(".text.fast".to_string()))
        );
        assert!(func.annotations.contains(&FunctionAnnotation::Align(16)));
    }

    #[test]
    fn test_function_with_params() {
        let mut builder = IRBuilder::new();

        let param1 = builder.param_simple("x", Type::Primitive(PrimitiveType::I32));
        let param2 = builder.param_simple("y", Type::Primitive(PrimitiveType::I64));

        builder
            .function_with_params(
                "add",
                vec![param1, param2],
                Type::Primitive(PrimitiveType::I64),
            )
            .zext(
                "x_extended",
                PrimitiveType::I32,
                PrimitiveType::I64,
                var("x"),
            )
            .binary(
                BinaryOp::Add,
                "sum",
                PrimitiveType::I64,
                var("x_extended"),
                var("y"),
            )
            .ret(Type::Primitive(PrimitiveType::I64), var("sum"));

        let module = builder.build();
        let func = &module.functions["add"];

        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.params[0].name, "x");
        assert_eq!(func.signature.params[1].name, "y");
    }

    #[test]
    fn test_all_binary_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("binary_ops", Type::Primitive(PrimitiveType::I32))
            .binary(
                BinaryOp::Add,
                "add_result",
                PrimitiveType::I32,
                i32(10),
                i32(5),
            )
            .binary(
                BinaryOp::Sub,
                "sub_result",
                PrimitiveType::I32,
                i32(10),
                i32(5),
            )
            .binary(
                BinaryOp::Mul,
                "mul_result",
                PrimitiveType::I32,
                i32(10),
                i32(5),
            )
            .binary(
                BinaryOp::Div,
                "div_result",
                PrimitiveType::I32,
                i32(10),
                i32(5),
            )
            .binary(
                BinaryOp::Rem,
                "rem_result",
                PrimitiveType::I32,
                i32(10),
                i32(3),
            )
            .binary(
                BinaryOp::And,
                "and_result",
                PrimitiveType::I32,
                i32(0b1010),
                i32(0b1100),
            )
            .binary(
                BinaryOp::Or,
                "or_result",
                PrimitiveType::I32,
                i32(0b1010),
                i32(0b1100),
            )
            .binary(
                BinaryOp::Xor,
                "xor_result",
                PrimitiveType::I32,
                i32(0b1010),
                i32(0b1100),
            )
            .binary(
                BinaryOp::Shl,
                "shl_result",
                PrimitiveType::I32,
                i32(5),
                i32(2),
            )
            .binary(
                BinaryOp::Shr,
                "shr_result",
                PrimitiveType::I32,
                i32(20),
                i32(2),
            )
            .ret(Type::Primitive(PrimitiveType::I32), var("add_result"));

        let module = builder.build();
        let func = &module.functions["binary_ops"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 11);

        // Verify first instruction is Add
        if let Instruction::Binary { op, .. } = &entry.instructions[0] {
            assert!(matches!(op, BinaryOp::Add));
        } else {
            panic!("Expected Binary instruction");
        }
    }

    #[test]
    fn test_all_comparison_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("cmp_ops", Type::Void)
            .cmp(CmpOp::Eq, "eq_result", PrimitiveType::I32, i32(5), i32(5))
            .cmp(CmpOp::Ne, "ne_result", PrimitiveType::I32, i32(5), i32(3))
            .cmp(CmpOp::Lt, "lt_result", PrimitiveType::I32, i32(3), i32(5))
            .cmp(CmpOp::Le, "le_result", PrimitiveType::I32, i32(5), i32(5))
            .cmp(CmpOp::Gt, "gt_result", PrimitiveType::I32, i32(5), i32(3))
            .cmp(CmpOp::Ge, "ge_result", PrimitiveType::I32, i32(5), i32(5))
            .ret_void();

        let module = builder.build();
        let func = &module.functions["cmp_ops"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 7);

        // Verify comparison operations
        if let Instruction::Cmp { op, .. } = &entry.instructions[0] {
            assert!(matches!(op, CmpOp::Eq));
        }
        if let Instruction::Cmp { op, .. } = &entry.instructions[1] {
            assert!(matches!(op, CmpOp::Ne));
        }
    }

    #[test]
    fn test_type_conversions() {
        let mut builder = IRBuilder::new();

        builder
            .function("conversions", Type::Primitive(PrimitiveType::I64))
            .zext(
                "zext_result",
                PrimitiveType::I32,
                PrimitiveType::I64,
                i32(42),
            )
            .sext(
                "sext_result",
                PrimitiveType::I32,
                PrimitiveType::I64,
                i32(-42),
            )
            .trunc(
                "trunc_result",
                PrimitiveType::I64,
                PrimitiveType::I32,
                i64(1000),
            )
            .bitcast(
                "bitcast_result",
                PrimitiveType::I32,
                PrimitiveType::F32,
                i32(0x40400000),
            )
            .ret(Type::Primitive(PrimitiveType::I64), var("zext_result"));

        let module = builder.build();
        let func = &module.functions["conversions"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 5);

        // Verify zero extension
        if let Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            ..
        } = &entry.instructions[0]
        {
            assert_eq!(*result, "zext_result");
            assert_eq!(*source_type, PrimitiveType::I32);
            assert_eq!(*target_type, PrimitiveType::I64);
        } else {
            panic!("Expected ZeroExtend instruction");
        }
    }

    #[test]
    fn test_pointer_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("pointer_ops", Type::Primitive(PrimitiveType::I32))
            .alloc_stack(
                "array",
                Type::Array {
                    element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                    size: 10,
                },
            )
            .getelementptr("elem_ptr", var("array"), i32(5), PrimitiveType::I32)
            .store(
                Type::Primitive(PrimitiveType::I32),
                var("elem_ptr"),
                i32(42),
            )
            .load(
                "value",
                Type::Primitive(PrimitiveType::I32),
                var("elem_ptr"),
            )
            .ret(Type::Primitive(PrimitiveType::I32), var("value"));

        let module = builder.build();
        let func = &module.functions["pointer_ops"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 5);

        // Verify getelementptr
        if let Instruction::GetElemPtr {
            result,
            index,
            element_type,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "elem_ptr");
            assert_eq!(*index, i32(5));
            assert_eq!(*element_type, PrimitiveType::I32);
        } else {
            panic!("Expected GetElemPtr instruction");
        }
    }

    #[test]
    fn test_struct_gep() {
        let mut builder = IRBuilder::new();

        builder
            .function("struct_ops", Type::Primitive(PrimitiveType::I32))
            .alloc_stack(
                "point",
                Type::Struct(vec![
                    crate::ir::types::StructField {
                        name: "x",
                        ty: Type::Primitive(PrimitiveType::I32),
                    },
                    crate::ir::types::StructField {
                        name: "y",
                        ty: Type::Primitive(PrimitiveType::I32),
                    },
                ]),
            )
            .struct_gep("x_ptr", var("point"), 0)
            .store(Type::Primitive(PrimitiveType::I32), var("x_ptr"), i32(10))
            .struct_gep("y_ptr", var("point"), 1)
            .store(Type::Primitive(PrimitiveType::I32), var("y_ptr"), i32(20))
            .load("x_val", Type::Primitive(PrimitiveType::I32), var("x_ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("x_val"));

        let module = builder.build();
        let func = &module.functions["struct_ops"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 7);

        // Verify struct_gep
        if let Instruction::GetFieldPtr {
            result,
            field_index,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "x_ptr");
            assert_eq!(*field_index, 0);
        } else {
            panic!("Expected GetFieldPtr instruction");
        }
    }

    #[test]
    fn test_heap_allocation() {
        let mut builder = IRBuilder::new();

        builder
            .function("heap_ops", Type::Void)
            .alloc_heap("heap_ptr", Type::Primitive(PrimitiveType::I32))
            .store(
                Type::Primitive(PrimitiveType::I32),
                var("heap_ptr"),
                i32(100),
            )
            .dealloc(var("heap_ptr"))
            .ret_void();

        let module = builder.build();
        let func = &module.functions["heap_ops"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 4);

        // Verify heap allocation
        if let Instruction::Alloc {
            result, alloc_type, ..
        } = &entry.instructions[0]
        {
            assert_eq!(*result, "heap_ptr");
            assert!(matches!(alloc_type, AllocType::Heap));
        } else {
            panic!("Expected Alloc instruction with Heap type");
        }

        // Verify deallocation
        if let Instruction::Dealloc { ptr } = &entry.instructions[2] {
            assert_eq!(*ptr, var("heap_ptr"));
        } else {
            panic!("Expected Dealloc instruction");
        }
    }

    #[test]
    fn test_function_call() {
        let mut builder = IRBuilder::new();

        // Define a helper function
        builder
            .function("helper", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(42));

        // Call it from another function
        builder
            .function("caller", Type::Primitive(PrimitiveType::I32))
            .call(Some("result"), "helper", vec![])
            .ret(Type::Primitive(PrimitiveType::I32), var("result"));

        let module = builder.build();

        assert!(module.functions.contains_key("helper"));
        assert!(module.functions.contains_key("caller"));

        let caller = &module.functions["caller"];
        let entry = &caller.basic_blocks["entry"];

        // Verify call instruction
        if let Instruction::Call {
            result,
            func_name,
            args,
        } = &entry.instructions[0]
        {
            assert_eq!(*result, Some("result"));
            assert_eq!(*func_name, "helper");
            assert_eq!(args.len(), 0);
        } else {
            panic!("Expected Call instruction");
        }
    }

    #[test]
    fn test_external_function() {
        let mut builder = IRBuilder::new();

        builder.external_function("printf", vec![], Type::Void);

        let module = builder.build();
        let func = &module.functions["printf"];

        assert!(func.annotations.contains(&FunctionAnnotation::Extern));
        assert_eq!(func.signature.params.len(), 0);
    }

    #[test]
    fn test_switch_statement() {
        let mut builder = IRBuilder::new();

        builder
            .function("switch_test", Type::Primitive(PrimitiveType::I32))
            .switch(
                PrimitiveType::I32,
                var("value"),
                "default_case",
                &[
                    (Literal::I32(1), "case_1"),
                    (Literal::I32(2), "case_2"),
                    (Literal::I32(3), "case_3"),
                ],
            )
            .block("case_1")
            .ret(Type::Primitive(PrimitiveType::I32), i32(10))
            .block("case_2")
            .ret(Type::Primitive(PrimitiveType::I32), i32(20))
            .block("case_3")
            .ret(Type::Primitive(PrimitiveType::I32), i32(30))
            .block("default_case")
            .ret(Type::Primitive(PrimitiveType::I32), i32(0));

        let module = builder.build();
        let func = &module.functions["switch_test"];

        assert!(func.basic_blocks.contains_key("case_1"));
        assert!(func.basic_blocks.contains_key("case_2"));
        assert!(func.basic_blocks.contains_key("case_3"));
        assert!(func.basic_blocks.contains_key("default_case"));

        let entry = &func.basic_blocks["entry"];
        if let Instruction::Switch {
            ty,
            value,
            default,
            cases,
        } = &entry.instructions[0]
        {
            assert_eq!(*ty, PrimitiveType::I32);
            assert_eq!(*value, var("value"));
            assert_eq!(*default, "default_case");
            assert_eq!(cases.len(), 3);
        } else {
            panic!("Expected Switch instruction");
        }
    }

    #[test]
    fn test_phi_node() {
        let mut builder = IRBuilder::new();

        builder
            .function("phi_test", Type::Primitive(PrimitiveType::I32))
            .cmp(CmpOp::Gt, "cond", PrimitiveType::I32, var("x"), i32(0))
            .branch(var("cond"), "positive", "negative")
            .block("positive")
            .binary(
                BinaryOp::Add,
                "pos_val",
                PrimitiveType::I32,
                var("x"),
                i32(1),
            )
            .jump("merge")
            .block("negative")
            .binary(
                BinaryOp::Sub,
                "neg_val",
                PrimitiveType::I32,
                var("x"),
                i32(1),
            )
            .jump("merge")
            .block("merge")
            .phi(
                "result",
                Type::Primitive(PrimitiveType::I32),
                vec![(var("pos_val"), "positive"), (var("neg_val"), "negative")],
            )
            .ret(Type::Primitive(PrimitiveType::I32), var("result"));

        let module = builder.build();
        let func = &module.functions["phi_test"];
        let merge = &func.basic_blocks["merge"];

        if let Instruction::Phi {
            result,
            ty,
            incoming,
        } = &merge.instructions[0]
        {
            assert_eq!(*result, "result");
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(incoming.len(), 2);
        } else {
            panic!("Expected Phi instruction");
        }
    }

    #[test]
    fn test_select_instruction() {
        let mut builder = IRBuilder::new();

        builder
            .function("select_test", Type::Primitive(PrimitiveType::I32))
            .cmp(CmpOp::Gt, "cond", PrimitiveType::I32, var("a"), var("b"))
            .select(
                "max",
                Type::Primitive(PrimitiveType::I32),
                var("cond"),
                var("a"),
                var("b"),
            )
            .ret(Type::Primitive(PrimitiveType::I32), var("max"));

        let module = builder.build();
        let func = &module.functions["select_test"];
        let entry = &func.basic_blocks["entry"];

        if let Instruction::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "max");
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*cond, var("cond"));
            assert_eq!(*true_val, var("a"));
            assert_eq!(*false_val, var("b"));
        } else {
            panic!("Expected Select instruction");
        }
    }

    #[test]
    fn test_tuple_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("tuple_test", Type::Primitive(PrimitiveType::I32))
            .tuple("my_tuple", vec![i32(10), i64(20), bool(true)])
            .extract_tuple("first", var("my_tuple"), 0)
            .extract_tuple("second", var("my_tuple"), 1)
            .ret(Type::Primitive(PrimitiveType::I32), var("first"));

        let module = builder.build();
        let func = &module.functions["tuple_test"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 4);

        // Verify tuple creation
        if let Instruction::Tuple { result, elements } = &entry.instructions[0] {
            assert_eq!(*result, "my_tuple");
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Tuple instruction");
        }

        // Verify tuple extraction
        if let Instruction::ExtractTuple {
            result,
            tuple_val,
            index,
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "first");
            assert_eq!(*tuple_val, var("my_tuple"));
            assert_eq!(*index, 0);
        } else {
            panic!("Expected ExtractTuple instruction");
        }
    }

    #[test]
    fn test_io_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("io_test", Type::Void)
            .alloc_stack(
                "buffer",
                Type::Array {
                    element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                    size: 100,
                },
            )
            .write(var("buffer"), i32(100), "bytes_written")
            .read(var("buffer"), i32(100), "bytes_read")
            .write_byte(i8(65), "byte_written")
            .read_byte("byte_read")
            .write_ptr(var("buffer"), "ptr_written")
            .print(i32(42))
            .ret_void();

        let module = builder.build();
        let func = &module.functions["io_test"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 8);

        // Verify write
        if let Instruction::Write {
            buffer,
            size,
            result,
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "bytes_written");
            assert_eq!(*size, i32(100));
        } else {
            panic!("Expected Write instruction");
        }

        // Verify print (should be at index 6, ret_void is at 7)
        if let Instruction::Print { value } = &entry.instructions[6] {
            assert_eq!(*value, i32(42));
        } else {
            panic!("Expected Print instruction");
        }
    }

    #[test]
    fn test_ptrtoint_inttoptr() {
        let mut builder = IRBuilder::new();

        builder
            .function("ptr_conv", Type::Primitive(PrimitiveType::I32))
            .alloc_stack("data", Type::Primitive(PrimitiveType::I32))
            .ptrtoint("ptr_as_int", var("data"), PrimitiveType::I64)
            .binary(
                BinaryOp::Add,
                "offset_ptr",
                PrimitiveType::I64,
                var("ptr_as_int"),
                i64(4),
            )
            .inttoptr("new_ptr", var("offset_ptr"), PrimitiveType::I32)
            .load("value", Type::Primitive(PrimitiveType::I32), var("new_ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("value"));

        let module = builder.build();
        let func = &module.functions["ptr_conv"];
        let entry = &func.basic_blocks["entry"];

        // Verify ptrtoint
        if let Instruction::PtrToInt {
            result,
            target_type,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "ptr_as_int");
            assert_eq!(*target_type, PrimitiveType::I64);
        } else {
            panic!("Expected PtrToInt instruction");
        }

        // Verify inttoptr
        if let Instruction::IntToPtr {
            result,
            target_type,
            ..
        } = &entry.instructions[3]
        {
            assert_eq!(*result, "new_ptr");
            assert_eq!(*target_type, PrimitiveType::I32);
        } else {
            panic!("Expected IntToPtr instruction");
        }
    }

    #[test]
    fn test_multiple_functions() {
        let mut builder = IRBuilder::new();

        builder
            .function("func1", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(1));

        builder
            .function("func2", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(2));

        builder
            .function("func3", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(3));

        let module = builder.build();

        assert_eq!(module.functions.len(), 3);
        assert!(module.functions.contains_key("func1"));
        assert!(module.functions.contains_key("func2"));
        assert!(module.functions.contains_key("func3"));
    }

    #[test]
    fn test_temp_var() {
        let mut builder = IRBuilder::new();

        let temp1 = builder.temp_var();
        let temp2 = builder.temp_var();
        let temp3 = builder.temp_var();

        assert_eq!(temp1, "temp_0");
        assert_eq!(temp2, "temp_1");
        assert_eq!(temp3, "temp_2");
        assert_ne!(temp1, temp2);
        assert_ne!(temp2, temp3);
    }

    #[test]
    fn test_set_entry_block() {
        let mut builder = IRBuilder::new();

        builder
            .function("custom_entry", Type::Primitive(PrimitiveType::I32))
            .block("entry")
            .block("actual_start")
            .set_entry_block("actual_start")
            .ret(Type::Primitive(PrimitiveType::I32), i32(42));

        let module = builder.build();
        let func = &module.functions["custom_entry"];

        assert_eq!(func.entry_block, "actual_start");
        assert_ne!(func.entry_block, "entry");
    }

    #[test]
    fn test_void_return() {
        let mut builder = IRBuilder::new();

        builder.function("void_func", Type::Void).ret_void();

        let module = builder.build();
        let func = &module.functions["void_func"];
        let entry = &func.basic_blocks["entry"];

        if let Instruction::Ret { ty, value } = &entry.instructions[0] {
            assert_eq!(*ty, Type::Void);
            assert_eq!(*value, None);
        } else {
            panic!("Expected Ret instruction with void");
        }
    }

    #[test]
    fn test_jump_instruction() {
        let mut builder = IRBuilder::new();

        builder
            .function("jump_test", Type::Void)
            .jump("target_block")
            .block("target_block")
            .ret_void();

        let module = builder.build();
        let func = &module.functions["jump_test"];
        let entry = &func.basic_blocks["entry"];

        if let Instruction::Jmp { target_label } = &entry.instructions[0] {
            assert_eq!(*target_label, "target_block");
        } else {
            panic!("Expected Jmp instruction");
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_atomic_operations() {
        let mut builder = IRBuilder::new();

        use crate::ir::instruction::{AtomicBinOp, MemoryOrdering};

        builder
            .function("atomic_test", Type::Primitive(PrimitiveType::I32))
            .alloc_stack("atomic_var", Type::Primitive(PrimitiveType::I32))
            .atomic_store(
                Type::Primitive(PrimitiveType::I32),
                var("atomic_var"),
                i32(10),
                MemoryOrdering::SeqCst,
            )
            .atomic_load(
                "loaded",
                Type::Primitive(PrimitiveType::I32),
                var("atomic_var"),
                MemoryOrdering::SeqCst,
            )
            .atomic_binary(
                AtomicBinOp::Add,
                "result",
                Type::Primitive(PrimitiveType::I32),
                var("atomic_var"),
                i32(5),
                MemoryOrdering::SeqCst,
            )
            .atomic_compare_exchange(
                "old_val",
                "success",
                Type::Primitive(PrimitiveType::I32),
                var("atomic_var"),
                i32(15),
                i32(20),
                MemoryOrdering::SeqCst,
                MemoryOrdering::Acquire,
            )
            .fence(MemoryOrdering::SeqCst)
            .ret(Type::Primitive(PrimitiveType::I32), var("loaded"));

        let module = builder.build();
        let func = &module.functions["atomic_test"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 7);

        // Verify atomic store
        if let Instruction::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        } = &entry.instructions[1]
        {
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*value, i32(10));
            assert_eq!(*ordering, MemoryOrdering::SeqCst);
        } else {
            panic!("Expected AtomicStore instruction");
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_simd_operations() {
        let mut builder = IRBuilder::new();

        use crate::ir::instruction::SimdOp;

        let vector_type = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::F32)),
            size: 4,
        };

        builder
            .function("simd_test", Type::Void)
            .alloc_stack("vec1", vector_type.clone())
            .alloc_stack("vec2", vector_type.clone())
            .simd_binary(
                SimdOp::Add,
                "result",
                vector_type.clone(),
                var("vec1"),
                var("vec2"),
            )
            .simd_unary(
                SimdOp::Sqrt,
                "sqrt_result",
                vector_type.clone(),
                var("result"),
            )
            .simd_extract("scalar", PrimitiveType::F32, var("sqrt_result"), i32(0))
            .simd_insert(
                "new_vec",
                vector_type.clone(),
                var("sqrt_result"),
                f32(3.14),
                i32(1),
            )
            .simd_load("loaded_vec", vector_type.clone(), var("vec1"), Some(16))
            .simd_store(vector_type.clone(), var("vec1"), var("new_vec"), Some(16))
            .ret_void();

        let module = builder.build();
        let func = &module.functions["simd_test"];
        let entry = &func.basic_blocks["entry"];

        assert_eq!(entry.instructions.len(), 9);

        // Verify SIMD binary operation
        if let Instruction::SimdBinary { op, result, .. } = &entry.instructions[2] {
            assert!(matches!(op, SimdOp::Add));
            assert_eq!(*result, "result");
        } else {
            panic!("Expected SimdBinary instruction");
        }
    }
}
