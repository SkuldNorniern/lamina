#[cfg(test)]
mod tests {
    use super::super::super::function::FunctionAnnotation;
    use super::super::super::instruction::{AllocType, BinaryOp, CmpOp, Instruction};
    #[cfg(feature = "nightly")]
    use super::super::super::instruction::SimdOp;
    use super::super::super::module::Module;
    #[cfg(feature = "nightly")]
    use super::super::super::module::ModuleAnnotation;
    use super::super::super::types::{PrimitiveType, Type, Value};
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
}

