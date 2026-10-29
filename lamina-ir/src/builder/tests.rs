#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    #[cfg(feature = "nightly")]
    use crate::module::ModuleAnnotation;
    use crate::{
        builder::{IRBuilder, values::*},
        function::FunctionAnnotation,
        instruction::{AllocType, BinaryOp, CmpOp, Instruction, UnaryOp},
        types::{Literal, PrimitiveType, StructField, Type},
    };
    #[cfg(feature = "nightly")]
    use std::f32::consts::PI;

    #[test]
    fn test_build_simple_function() {
        let mut builder = IRBuilder::new();

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

        assert!(module.functions.contains_key("add"));

        let func = &module.functions["add"];
        assert_eq!(func.name, "add");
        assert_eq!(
            func.signature.return_type,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.annotations.len(), 1);
        assert!(matches!(func.annotations[0], FunctionAnnotation::Inline));

        assert!(func.basic_blocks.contains_key("entry"));
        let entry_block = &func.basic_blocks["entry"];
        assert_eq!(entry_block.instructions.len(), 2);

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

        assert!(module.functions.contains_key("conditional"));

        let func = &module.functions["conditional"];

        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("then_block"));
        assert!(func.basic_blocks.contains_key("else_block"));

        let entry = &func.basic_blocks["entry"];
        let then_block = &func.basic_blocks["then_block"];
        let else_block = &func.basic_blocks["else_block"];

        assert_eq!(entry.instructions.len(), 2);
        assert_eq!(then_block.instructions.len(), 1);
        assert_eq!(else_block.instructions.len(), 1);

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
            .stackalloc("ptr", Type::Primitive(PrimitiveType::I32))
            .store(Type::Primitive(PrimitiveType::I32), var("ptr"), i32(42))
            .load("val", Type::Primitive(PrimitiveType::I32), var("ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("val"));

        let module = builder.build();

        assert!(module.functions.contains_key("use_memory"));

        let func = &module.functions["use_memory"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Alloc { .. },
                Instruction::Store { .. },
                Instruction::Load { .. },
                Instruction::Ret { .. },
            ]
        ));

        if let Instruction::Alloc {
            result,
            alloc_type,
            allocated_ty,
            ..
        } = &entry.instructions[0]
        {
            assert_eq!(*result, "ptr");
            assert!(matches!(alloc_type, AllocType::Stack));
            assert_eq!(*allocated_ty, Type::Primitive(PrimitiveType::I32));
        } else {
            panic!("Expected Alloc instruction");
        }

        if let Instruction::Store {
            ty,
            ptr,
            value,
            volatile,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*ptr, var("ptr"));
            assert_eq!(*value, i32(42));
            assert!(!volatile);
        } else {
            panic!("Expected Store instruction");
        }

        if let Instruction::Load {
            result,
            ty,
            ptr,
            volatile,
            ..
        } = &entry.instructions[2]
        {
            assert_eq!(*result, "val");
            assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(*ptr, var("ptr"));
            assert!(!volatile);
        } else {
            panic!("Expected Load instruction");
        }
    }

    #[test]
    fn test_volatile_store_load_builders() {
        let mut builder = IRBuilder::new();

        builder
            .function("f", Type::Primitive(PrimitiveType::I64))
            .stackalloc("p", Type::Primitive(PrimitiveType::I64))
            .volatile_store(Type::Primitive(PrimitiveType::I64), var("p"), i64(5))
            .volatile_load("v", Type::Primitive(PrimitiveType::I64), var("p"))
            .ret(Type::Primitive(PrimitiveType::I64), var("v"));

        let module = builder.build();
        let entry = &module.functions["f"].basic_blocks["entry"];

        assert!(matches!(
            entry.instructions[1],
            Instruction::Store { volatile: true, .. }
        ));
        assert!(matches!(
            entry.instructions[2],
            Instruction::Load { volatile: true, .. }
        ));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_module_annotations() {
        let mut builder = IRBuilder::new();

        builder
            .annotate_module(ModuleAnnotation::PositionIndependentCode)
            .pie();

        let module = builder.build();

        assert_eq!(module.annotations.len(), 2);
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
            .section(".text.fast")
            .align(16)
            .ret_void();

        let module = builder.build();

        assert!(module.functions.contains_key("test_func"));

        let func = &module.functions["test_func"];

        assert!(func.annotations.contains(&FunctionAnnotation::Inline));
        assert!(func.annotations.contains(&FunctionAnnotation::Export));
        assert!(func.annotations.contains(&FunctionAnnotation::Hot));
        assert!(func.annotations.contains(&FunctionAnnotation::Pure));
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
            .sext(
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

        assert!(module.functions.contains_key("add"));

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

        assert!(module.functions.contains_key("binary_ops"));

        let func = &module.functions["binary_ops"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Binary {
                    op: BinaryOp::Add,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Sub,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Mul,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Div,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Rem,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::And,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Or,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Xor,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Shl,
                    ..
                },
                Instruction::Binary {
                    op: BinaryOp::Shr,
                    ..
                },
                Instruction::Ret { .. },
            ]
        ));

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

        assert!(module.functions.contains_key("cmp_ops"));

        let func = &module.functions["cmp_ops"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Cmp { op: CmpOp::Eq, .. },
                Instruction::Cmp { op: CmpOp::Ne, .. },
                Instruction::Cmp { op: CmpOp::Lt, .. },
                Instruction::Cmp { op: CmpOp::Le, .. },
                Instruction::Cmp { op: CmpOp::Gt, .. },
                Instruction::Cmp { op: CmpOp::Ge, .. },
                Instruction::Ret { .. },
            ]
        ));
    }

    #[test]
    fn test_type_conversions() {
        let mut builder = IRBuilder::new();

        builder
            .function("conversions", Type::Primitive(PrimitiveType::I64))
            .zext(
                "zext_result",
                PrimitiveType::U32,
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

        assert!(module.functions.contains_key("conversions"));

        let func = &module.functions["conversions"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::ZeroExtend { .. },
                Instruction::SignExtend { .. },
                Instruction::Trunc { .. },
                Instruction::Bitcast { .. },
                Instruction::Ret { .. },
            ]
        ));

        // Verify zero extension
        if let Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            ..
        } = &entry.instructions[0]
        {
            assert_eq!(*result, "zext_result");
            assert_eq!(*source_type, PrimitiveType::U32);
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
            .stackalloc(
                "array",
                Type::Array {
                    element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                    size: 10,
                },
            )
            .getelem("elem_ptr", var("array"), i32(5), PrimitiveType::I32)
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

        assert!(module.functions.contains_key("pointer_ops"));

        let func = &module.functions["pointer_ops"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Alloc { .. },
                Instruction::GetElemPtr { .. },
                Instruction::Store { .. },
                Instruction::Load { .. },
                Instruction::Ret { .. },
            ]
        ));

        // Verify getelem
        if let Instruction::GetElemPtr {
            result,
            index,
            element_type,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "elem_ptr");
            assert_eq!(*index, i32(5));
            assert_eq!(*element_type, Type::Primitive(PrimitiveType::I32));
        } else {
            panic!("Expected GetElemPtr instruction");
        }
    }

    #[test]
    fn composite_element_pointer_preserves_type() {
        let row = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: 3,
        };
        let mut builder = IRBuilder::new();
        builder
            .function("row", Type::Primitive(PrimitiveType::Ptr))
            .getelem("row_ptr", var("matrix"), i32(1), row.clone())
            .ret_void();
        let module = builder.build();
        let inst = &module.functions["row"].basic_blocks["entry"].instructions[0];
        assert!(
            matches!(inst, Instruction::GetElemPtr { element_type, .. } if *element_type == row)
        );
        assert_eq!(inst.to_string(), "%row_ptr = getelem.[3 x i32] %matrix, 1");
    }

    #[test]
    fn test_getfieldptr() {
        let mut builder = IRBuilder::new();

        let point_ty = Type::Struct {
            fields: vec![
                StructField {
                    name: "x",
                    ty: Type::Primitive(PrimitiveType::I32),
                },
                StructField {
                    name: "y",
                    ty: Type::Primitive(PrimitiveType::I32),
                },
            ],
            packed: false,
        };

        builder
            .function("struct_ops", Type::Primitive(PrimitiveType::I32))
            .stackalloc("point", point_ty.clone())
            .getfieldptr("x_ptr", var("point"), point_ty.clone(), 0)
            .store(Type::Primitive(PrimitiveType::I32), var("x_ptr"), i32(10))
            .getfieldptr("y_ptr", var("point"), point_ty, 1)
            .store(Type::Primitive(PrimitiveType::I32), var("y_ptr"), i32(20))
            .load("x_val", Type::Primitive(PrimitiveType::I32), var("x_ptr"))
            .ret(Type::Primitive(PrimitiveType::I32), var("x_val"));

        let module = builder.build();

        assert!(module.functions.contains_key("struct_ops"));

        let func = &module.functions["struct_ops"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Alloc { .. },
                Instruction::GetFieldPtr { .. },
                Instruction::Store { .. },
                Instruction::GetFieldPtr { .. },
                Instruction::Store { .. },
                Instruction::Load { .. },
                Instruction::Ret { .. },
            ]
        ));

        // Verify getfieldptr
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

        assert!(caller.basic_blocks.contains_key("entry"));

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

        assert!(module.functions.contains_key("printf"));

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

        assert!(module.functions.contains_key("switch_test"));

        let func = &module.functions["switch_test"];

        assert!(func.basic_blocks.contains_key("case_1"));
        assert!(func.basic_blocks.contains_key("case_2"));
        assert!(func.basic_blocks.contains_key("case_3"));
        assert!(func.basic_blocks.contains_key("default_case"));
        assert!(func.basic_blocks.contains_key("entry"));

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

        assert!(module.functions.contains_key("select_test"));

        let func = &module.functions["select_test"];

        assert!(func.basic_blocks.contains_key("entry"));

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
    fn test_io_operations() {
        let mut builder = IRBuilder::new();

        builder
            .function("io_test", Type::Void)
            .stackalloc(
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
            .write_byte(i32(42), "printed")
            .ret_void();

        let module = builder.build();

        assert!(module.functions.contains_key("io_test"));

        let func = &module.functions["io_test"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Alloc { .. },
                Instruction::Write { .. },
                Instruction::Read { .. },
                Instruction::WriteByte { .. },
                Instruction::ReadByte { .. },
                Instruction::WriteByte { .. },
                Instruction::Ret { .. },
            ]
        ));

        // Verify write
        if let Instruction::Write {
            buffer: _,
            size,
            result,
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "bytes_written");
            assert_eq!(*size, i32(100));
        } else {
            panic!("Expected Write instruction");
        }

        // Verify writebyte (should be at index 5, ret_void is at 6)
        if let Instruction::WriteByte { value, result } = &entry.instructions[5] {
            assert_eq!(*value, i32(42));
            assert_eq!(*result, "printed");
        } else {
            panic!("Expected WriteByte instruction");
        }
    }

    #[test]
    fn test_ptrtoint_inttoptr() {
        let mut builder = IRBuilder::new();

        builder
            .function("ptr_conv", Type::Primitive(PrimitiveType::I32))
            .stackalloc("data", Type::Primitive(PrimitiveType::I32))
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

        assert!(module.functions.contains_key("ptr_conv"));

        let func = &module.functions["ptr_conv"];

        assert!(func.basic_blocks.contains_key("entry"));

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
            .jump("actual_start")
            .block("actual_start")
            .set_entry_block("actual_start")
            .ret(Type::Primitive(PrimitiveType::I32), i32(42));

        let module = builder.build();

        assert!(module.functions.contains_key("custom_entry"));

        let func = &module.functions["custom_entry"];

        assert_eq!(func.entry_block, "actual_start");
        assert_ne!(func.entry_block, "entry");
    }

    #[test]
    fn test_void_return() {
        let mut builder = IRBuilder::new();

        builder.function("void_func", Type::Void).ret_void();

        let module = builder.build();

        assert!(module.functions.contains_key("void_func"));

        let func = &module.functions["void_func"];

        assert!(func.basic_blocks.contains_key("entry"));

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

        assert!(module.functions.contains_key("jump_test"));

        let func = &module.functions["jump_test"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        if let Instruction::Jmp { target_label } = &entry.instructions[0] {
            assert_eq!(*target_label, "target_block");
        } else {
            panic!("Expected Jmp instruction");
        }
    }

    #[test]
    fn test_atomic_operations() {
        let mut builder = IRBuilder::new();

        use crate::instruction::{AtomicBinOp, MemoryOrdering};

        builder
            .function("atomic_test", Type::Primitive(PrimitiveType::I32))
            .stackalloc("atomic_var", Type::Primitive(PrimitiveType::I32))
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

        assert!(module.functions.contains_key("atomic_test"));

        let func = &module.functions["atomic_test"];

        assert!(func.basic_blocks.contains_key("entry"));

        let entry = &func.basic_blocks["entry"];

        assert!(matches!(
            entry.instructions.as_slice(),
            [
                Instruction::Alloc { .. },
                Instruction::AtomicStore { .. },
                Instruction::AtomicLoad { .. },
                Instruction::AtomicBinary { .. },
                Instruction::AtomicCompareExchange { .. },
                Instruction::Fence { .. },
                Instruction::Ret { .. },
            ]
        ));

        // Verify atomic store
        if let Instruction::AtomicStore {
            ty,
            ptr: _,
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

    #[test]
    fn test_getfieldptr_keeps_layout() {
        let mut builder = IRBuilder::new();

        let fields = vec![
            StructField {
                name: "flag",
                ty: Type::Primitive(PrimitiveType::I8),
            },
            StructField {
                name: "value",
                ty: Type::Primitive(PrimitiveType::I32),
            },
        ];
        let layout = Type::Struct {
            fields,
            packed: false,
        };

        builder
            .function("typed_gep", Type::Void)
            .stackalloc("s", layout.clone())
            .getfieldptr("value_ptr", var("s"), layout.clone(), 1)
            .ret_void();

        let module = builder.build();
        let entry = &module.functions["typed_gep"].basic_blocks["entry"];

        if let Instruction::GetFieldPtr {
            result,
            field_index,
            struct_type,
            ..
        } = &entry.instructions[1]
        {
            assert_eq!(*result, "value_ptr");
            assert_eq!(*field_index, 1);
            assert_eq!(*struct_type, layout);
        } else {
            panic!("Expected GetFieldPtr for value");
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    #[should_panic(expected = "SIMD instructions have no text form")]
    fn test_simd_has_no_text_form() {
        let mut builder = IRBuilder::new();
        builder.function("simd_test", Type::Void).simd_add(
            "result",
            Type::Primitive(PrimitiveType::F32),
            var("a"),
            var("b"),
        );
    }

    #[test]
    fn test_unary_builders() {
        let mut builder = IRBuilder::new();
        builder
            .function("unary", Type::Primitive(PrimitiveType::I32))
            .neg("n", PrimitiveType::I32, var("a"))
            .not("m", PrimitiveType::U8, var("b"))
            .sqrt("r", PrimitiveType::F64, var("c"))
            .ret(Type::Primitive(PrimitiveType::I32), var("n"));

        let module = builder.build();
        let instructions = &module.functions["unary"].basic_blocks["entry"].instructions;
        assert_eq!(
            instructions[0],
            Instruction::Unary {
                op: UnaryOp::Neg,
                result: "n",
                ty: PrimitiveType::I32,
                value: var("a"),
            }
        );
        assert_eq!(
            instructions[1],
            Instruction::Unary {
                op: UnaryOp::Not,
                result: "m",
                ty: PrimitiveType::U8,
                value: var("b"),
            }
        );
        assert_eq!(
            instructions[2],
            Instruction::Unary {
                op: UnaryOp::Sqrt,
                result: "r",
                ty: PrimitiveType::F64,
                value: var("c"),
            }
        );
    }

    #[test]
    #[should_panic(expected = "not takes an integer type, found f32")]
    fn test_not_rejects_float_type() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::I32))
            .not("m", PrimitiveType::F32, var("b"));
    }

    #[test]
    #[should_panic(expected = "sqrt takes f32 or f64, found i64")]
    fn test_sqrt_rejects_integer_type() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::I32))
            .sqrt("r", PrimitiveType::I64, var("c"));
    }

    #[test]
    fn test_float_conversion_builders() {
        let mut builder = IRBuilder::new();
        builder
            .function("convert", Type::Primitive(PrimitiveType::F64))
            .inttofp("a", PrimitiveType::U32, PrimitiveType::F64, var("x"))
            .fptoint("b", PrimitiveType::F32, PrimitiveType::I8, var("y"))
            .fpcast("c", PrimitiveType::F64, PrimitiveType::F32, var("z"))
            .ret(Type::Primitive(PrimitiveType::F64), var("a"));

        let module = builder.build();
        let instructions = &module.functions["convert"].basic_blocks["entry"].instructions;
        assert_eq!(
            instructions[0],
            Instruction::IntToFloat {
                result: "a",
                signed: false,
                source_type: PrimitiveType::U32,
                target_type: PrimitiveType::F64,
                value: var("x"),
            }
        );
        assert_eq!(
            instructions[1],
            Instruction::FloatToInt {
                result: "b",
                signed: true,
                source_type: PrimitiveType::F32,
                target_type: PrimitiveType::I8,
                value: var("y"),
            }
        );
        assert_eq!(
            instructions[2],
            Instruction::FloatConvert {
                result: "c",
                source_type: PrimitiveType::F64,
                target_type: PrimitiveType::F32,
                value: var("z"),
            }
        );
    }

    #[test]
    fn test_inttofp_signedness_comes_from_source_type() {
        let signed_sources = [
            PrimitiveType::I8,
            PrimitiveType::I16,
            PrimitiveType::I32,
            PrimitiveType::I64,
        ];
        let unsigned_sources = [
            PrimitiveType::U8,
            PrimitiveType::U16,
            PrimitiveType::U32,
            PrimitiveType::U64,
        ];
        let mut builder = IRBuilder::new();
        builder.function("signedness", Type::Primitive(PrimitiveType::F32));
        for ty in signed_sources.into_iter().chain(unsigned_sources) {
            builder.inttofp("r", ty, PrimitiveType::F32, var("v"));
        }
        builder.ret_void();
        let module = builder.build();
        let instructions = &module.functions["signedness"].basic_blocks["entry"].instructions;
        let signs: Vec<bool> = instructions
            .iter()
            .take(instructions.len() - 1)
            .map(|inst| match inst {
                Instruction::IntToFloat { signed, .. } => *signed,
                other => panic!("Expected IntToFloat instruction, found {other:?}"),
            })
            .collect();
        assert_eq!(signs, [true, true, true, true, false, false, false, false]);

        let mut builder = IRBuilder::new();
        builder
            .function("from_u32", Type::Primitive(PrimitiveType::F64))
            .inttofp("u", PrimitiveType::U32, PrimitiveType::F64, var("x"))
            .inttofp("s", PrimitiveType::I32, PrimitiveType::F64, var("x"))
            .ret_void();
        let module = builder.build();
        let instructions = &module.functions["from_u32"].basic_blocks["entry"].instructions;
        assert!(matches!(
            instructions[0],
            Instruction::IntToFloat { signed: false, .. }
        ));
        assert!(matches!(
            instructions[1],
            Instruction::IntToFloat { signed: true, .. }
        ));
    }

    #[test]
    fn test_fptoint_signedness_comes_from_target_type() {
        let mut builder = IRBuilder::new();
        builder
            .function("to_int", Type::Primitive(PrimitiveType::I64))
            .fptoint("a", PrimitiveType::F64, PrimitiveType::U16, var("x"))
            .fptoint("b", PrimitiveType::F64, PrimitiveType::I64, var("x"))
            .ret_void();
        let module = builder.build();
        let instructions = &module.functions["to_int"].basic_blocks["entry"].instructions;
        assert!(matches!(
            instructions[0],
            Instruction::FloatToInt { signed: false, .. }
        ));
        assert!(matches!(
            instructions[1],
            Instruction::FloatToInt { signed: true, .. }
        ));
    }

    #[test]
    #[should_panic(expected = "inttofp source must be an integer type, found f32")]
    fn test_inttofp_rejects_float_source() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::F64))
            .inttofp("r", PrimitiveType::F32, PrimitiveType::F64, var("v"));
    }

    #[test]
    #[should_panic(expected = "fptoint target must be an integer type, found f64")]
    fn test_fptoint_rejects_float_target() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::I32))
            .fptoint("r", PrimitiveType::F32, PrimitiveType::F64, var("v"));
    }

    #[test]
    #[should_panic(expected = "fpcast converts between f32 and f64, found i32 to f64")]
    fn test_fpcast_rejects_integer_source() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::F64))
            .fpcast("r", PrimitiveType::I32, PrimitiveType::F64, var("v"));
    }

    #[test]
    fn test_funcaddr_builder() {
        let mut builder = IRBuilder::new();
        builder
            .function("take_addr", Type::Primitive(PrimitiveType::Ptr))
            .funcaddr("p", "helper")
            .ret(Type::Primitive(PrimitiveType::Ptr), var("p"));
        let module = builder.build();
        let instructions = &module.functions["take_addr"].basic_blocks["entry"].instructions;
        assert_eq!(
            instructions[0],
            Instruction::FuncAddr {
                result: "p",
                func_name: "helper",
            }
        );
    }

    #[test]
    fn test_callptr_builder() {
        let i64_ty = Type::Primitive(PrimitiveType::I64);
        let f32_ty = Type::Primitive(PrimitiveType::F32);
        let mut builder = IRBuilder::new();
        builder
            .function("indirect", Type::Primitive(PrimitiveType::I64))
            .callptr(
                Some("r"),
                i64_ty.clone(),
                var("f"),
                vec![(f32_ty.clone(), f32(1.5)), (i64_ty.clone(), var("n"))],
                None,
            )
            .callptr(None, Type::Void, var("g"), vec![], None)
            .callptr(
                Some("v"),
                Type::Primitive(PrimitiveType::I32),
                var("h"),
                vec![
                    (Type::Primitive(PrimitiveType::Ptr), var("fmt")),
                    (Type::Primitive(PrimitiveType::F64), var("x")),
                ],
                Some(1),
            )
            .ret_void();
        let module = builder.build();
        let instructions = &module.functions["indirect"].basic_blocks["entry"].instructions;
        assert_eq!(
            instructions[0],
            Instruction::CallIndirect {
                result: Some("r"),
                target: var("f"),
                args: vec![(f32_ty, f32(1.5)), (i64_ty.clone(), var("n"))],
                variadic_fixed: None,
                return_type: i64_ty,
            }
        );
        assert_eq!(
            instructions[1],
            Instruction::CallIndirect {
                result: None,
                target: var("g"),
                args: vec![],
                variadic_fixed: None,
                return_type: Type::Void,
            }
        );
        assert_eq!(
            instructions[2],
            Instruction::CallIndirect {
                result: Some("v"),
                target: var("h"),
                args: vec![
                    (Type::Primitive(PrimitiveType::Ptr), var("fmt")),
                    (Type::Primitive(PrimitiveType::F64), var("x")),
                ],
                variadic_fixed: Some(1),
                return_type: Type::Primitive(PrimitiveType::I32),
            }
        );
    }

    #[test]
    #[should_panic(expected = "callptr variadic_fixed (2) exceeds the argument count (1)")]
    fn test_callptr_rejects_variadic_fixed_beyond_arguments() {
        IRBuilder::new()
            .function("bad", Type::Primitive(PrimitiveType::I32))
            .callptr(
                None,
                Type::Primitive(PrimitiveType::I32),
                var("f"),
                vec![(Type::Primitive(PrimitiveType::I32), i32(1))],
                Some(2),
            );
    }

    #[test]
    fn test_each_function_keeps_its_own_blocks() {
        let mut builder = IRBuilder::new();

        builder
            .function("one", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(1));
        builder
            .function("two", Type::Primitive(PrimitiveType::I32))
            .ret(Type::Primitive(PrimitiveType::I32), i32(2));

        let module = builder.build();

        let one = &module.functions["one"];
        let two = &module.functions["two"];
        assert_eq!(one.basic_blocks.len(), 1);
        assert_eq!(two.basic_blocks.len(), 1);
        assert_eq!(one.basic_blocks["entry"].instructions.len(), 1);
        assert_eq!(
            one.basic_blocks["entry"].instructions[0].to_string(),
            "ret.i32 1"
        );
        assert_eq!(
            two.basic_blocks["entry"].instructions[0].to_string(),
            "ret.i32 2"
        );
    }

    #[test]
    #[should_panic(expected = "block 'entry' is already defined in '@f'")]
    fn test_duplicate_block_label_panics() {
        let mut builder = IRBuilder::new();
        builder.function("f", Type::Void).block("entry");
    }

    #[test]
    #[should_panic(expected = "zext zero-extends an unsigned or bool source")]
    fn test_zext_rejects_signed_source() {
        let mut builder = IRBuilder::new();
        builder
            .function("f", Type::Primitive(PrimitiveType::I64))
            .zext("r", PrimitiveType::I32, PrimitiveType::I64, var("x"));
    }

    #[test]
    #[should_panic(expected = "sext sign-extends a signed source")]
    fn test_sext_rejects_unsigned_source() {
        let mut builder = IRBuilder::new();
        builder
            .function("f", Type::Primitive(PrimitiveType::I64))
            .sext("r", PrimitiveType::U8, PrimitiveType::I64, var("x"));
    }

    #[test]
    fn test_extend_picks_instruction_from_source_signedness() {
        let mut builder = IRBuilder::new();
        builder
            .function("f", Type::Primitive(PrimitiveType::I64))
            .zext("z", PrimitiveType::U8, PrimitiveType::I64, var("u"))
            .sext("s", PrimitiveType::I8, PrimitiveType::I64, var("i"))
            .ret(Type::Primitive(PrimitiveType::I64), var("s"));

        let module = builder.build();
        let instructions = &module.functions["f"].basic_blocks["entry"].instructions;
        assert!(matches!(instructions[0], Instruction::ZeroExtend { .. }));
        assert!(matches!(instructions[1], Instruction::SignExtend { .. }));
    }

    #[test]
    #[should_panic(expected = "copy moves one scalar value")]
    fn test_copy_rejects_aggregate() {
        let mut builder = IRBuilder::new();
        builder.function("f", Type::Void).copy(
            "r",
            Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
                size: 2,
            },
            var("x"),
        );
    }

    #[test]
    #[should_panic(expected = "bitcast reinterprets equal widths")]
    fn test_bitcast_rejects_width_change() {
        let mut builder = IRBuilder::new();
        builder.function("f", Type::Void).bitcast(
            "r",
            PrimitiveType::I32,
            PrimitiveType::F64,
            var("x"),
        );
    }

    #[test]
    #[should_panic(expected = "no open function")]
    fn test_instruction_without_function_panics() {
        let mut builder = IRBuilder::new();
        builder.ret_void();
    }

    #[test]
    #[should_panic(expected = "extern function '@printf' has no body")]
    fn test_extern_function_has_no_body() {
        let mut builder = IRBuilder::new();
        builder
            .external_function("printf", vec![], Type::Void)
            .ret_void();
    }

    #[test]
    #[should_panic(expected = "has no text form")]
    fn test_annotation_without_text_form_panics() {
        let mut builder = IRBuilder::new();
        builder
            .function("f", Type::Void)
            .annotate(FunctionAnnotation::Deprecated(None));
    }

    #[test]
    #[should_panic(expected = "cannot be used together")]
    fn test_conflicting_annotations_panic() {
        let mut builder = IRBuilder::new();
        builder
            .function("f", Type::Void)
            .inline()
            .annotate(FunctionAnnotation::NoInline);
    }

    #[test]
    fn test_declare_type_is_kept_in_module() {
        let mut builder = IRBuilder::new();
        builder
            .declare_type(
                "Pair",
                Type::Struct {
                    fields: vec![StructField {
                        name: "a",
                        ty: Type::Primitive(PrimitiveType::I32),
                    }],
                    packed: false,
                },
            )
            .function("f", Type::Void)
            .ret_void();

        let module = builder.build();
        assert!(module.type_declarations.contains_key("Pair"));
    }

    #[test]
    #[should_panic(expected = "already ends with a terminator")]
    fn test_instruction_after_terminator_panics() {
        let mut builder = IRBuilder::new();
        builder.function("f", Type::Void).ret_void().binary(
            BinaryOp::Add,
            "r",
            PrimitiveType::I32,
            i32(1),
            i32(2),
        );
    }

    #[test]
    #[should_panic(expected = "must end with a terminator")]
    fn test_block_without_terminator_panics_at_build() {
        let mut builder = IRBuilder::new();
        builder.function("f", Type::Void).binary(
            BinaryOp::Add,
            "r",
            PrimitiveType::I32,
            i32(1),
            i32(2),
        );
        builder.build();
    }
}
