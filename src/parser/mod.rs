//! Lamina IR parser.
//!
//! This module provides parsing functionality for Lamina IR text into structured
//! Module representations.

mod functions;
mod globals;
mod instructions;
pub mod state;
mod types;
mod values;

use self::functions::parse_function_def;
use self::globals::parse_global_declaration;
use self::state::ParserState;
use self::types::parse_type_declaration;
use crate::{LaminaError, Module};

/// Calculates the Levenshtein edit distance between two strings.
///
/// This function computes the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to transform one string
/// into another. The comparison is case-insensitive for better typo detection.
///
/// # Arguments
///
/// * `s1` - First string to compare
/// * `s2` - Second string to compare
/// * `max_distance` - Maximum distance to consider (for early termination optimization)
///
/// # Returns
///
/// The edit distance between the two strings, or `max_distance + 1` if the
/// distance exceeds `max_distance` (for early termination).
///
/// # Examples
///
/// ```
/// # use lamina::parser::edit_distance;
/// assert_eq!(edit_distance("inline", "inlien", None), 2);
/// assert_eq!(edit_distance("export", "EXPORT", None), 0); // case-insensitive
/// assert_eq!(edit_distance("extern", "external", Some(2)), 3); // exceeds max
/// ```
pub fn edit_distance(s1: &str, s2: &str, max_distance: Option<usize>) -> usize {
    // Normalize to lowercase for case-insensitive comparison
    let s1_lower: Vec<char> = s1.to_lowercase().chars().collect();
    let s2_lower: Vec<char> = s2.to_lowercase().chars().collect();
    let m = s1_lower.len();
    let n = s2_lower.len();

    // Early exit for empty strings
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Early exit if length difference exceeds max_distance
    if let Some(max) = max_distance {
        let len_diff = m.abs_diff(n);
        if len_diff > max {
            return max + 1;
        }
    }

    // Use space-optimized DP: only store two rows at a time
    // This reduces space complexity from O(m*n) to O(min(m,n))
    let (shorter, longer) = if m <= n {
        (&s1_lower, &s2_lower)
    } else {
        (&s2_lower, &s1_lower)
    };
    let short_len = shorter.len();
    let long_len = longer.len();

    // Previous row (dp[i-1])
    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    // Current row (dp[i])
    let mut curr_row = vec![0; short_len + 1];

    for i in 1..=long_len {
        curr_row[0] = i;

        for j in 1..=short_len {
            // Cost is 0 if characters match, 1 otherwise
            let cost = if longer[i - 1] == shorter[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution

            // Early termination if we exceed max_distance
            if let Some(max) = max_distance
                && curr_row[j] > max
            {
                return max + 1;
            }
        }

        // Swap rows for next iteration
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[short_len]
}

/// Returns all valid primitive type names as strings.
///
/// This function delegates to the IR module, ensuring that parser error messages
/// always stay in sync with the actual type system.
pub fn get_primitive_type_names() -> &'static [&'static str] {
    crate::ir::PrimitiveType::all_names()
}

/// Returns all valid allocation type names as strings.
///
/// This function delegates to the IR module, ensuring that parser error messages
/// always stay in sync with the actual type system.
pub fn get_alloc_type_names() -> &'static [&'static str] {
    crate::ir::AllocType::all_names()
}

/// Returns all valid instruction opcodes that can appear after an assignment.
///
/// These are instructions that produce a result value (e.g., `%result = add.i32 ...`).
/// This function delegates to the IR module, ensuring that parser error messages
/// always stay in sync with the actual instruction set.
pub fn get_assignment_opcode_names() -> &'static [&'static str] {
    crate::ir::assignment_opcode_names()
}

/// Returns all valid instruction opcodes that don't require an assignment.
///
/// These are instructions that don't produce a result value (e.g., `ret.void`, `jmp label`).
/// This function delegates to the IR module, ensuring that parser error messages
/// always stay in sync with the actual instruction set.
pub fn get_non_assignment_opcode_names() -> &'static [&'static str] {
    crate::ir::non_assignment_opcode_names()
}

/// Parses a string containing Lamina IR text into a Module.
pub fn parse_module(input: &str) -> Result<Module<'_>, LaminaError> {
    let mut state = ParserState::new(input);
    let mut module = Module::new();

    let mut seen_names = std::collections::HashSet::new();

    loop {
        state.skip_whitespace_and_comments();
        if state.is_eof() {
            break;
        }

        let keyword_slice = state.peek_slice(6).unwrap_or("");

        if keyword_slice.starts_with("type") {
            let decl = parse_type_declaration(&mut state)?;
            let name = decl.name;
            if !seen_names.insert(name) {
                return Err(state.error(format!(
                    "Duplicate name '{}': a type, function, or global with this name already exists\n  Hint: Each name must be unique across types, functions, and globals",
                    name
                )));
            }
            if module.type_declarations.insert(name, decl).is_some() {
                return Err(state.error(format!(
                    "Duplicate type declaration: @{}\n  Hint: Each type can only be declared once",
                    name
                )));
            }
        } else if keyword_slice.starts_with("global") {
            let decl = parse_global_declaration(&mut state)?;
            let name = decl.name;
            if !seen_names.insert(name) {
                return Err(state.error(format!(
                    "Duplicate name '{}': a type, function, or global with this name already exists\n  Hint: Each name must be unique across types, functions, and globals",
                    name
                )));
            }
            if module.global_declarations.insert(name, decl).is_some() {
                return Err(state.error(format!(
                    "Duplicate global declaration: @{}\n  Hint: Each global can only be declared once",
                    name
                )));
            }
        } else if keyword_slice.starts_with("fn") || keyword_slice.starts_with('@') {
            let func = parse_function_def(&mut state)?;
            let name = func.name;
            if !seen_names.insert(name) {
                return Err(state.error(format!(
                    "Duplicate name '{}': a type, function, or global with this name already exists\n  Hint: Each name must be unique across types, functions, and globals",
                    name
                )));
            }
            if module.functions.insert(name, func).is_some() {
                return Err(state.error(format!(
                    "Duplicate function definition: @{}\n  Hint: Each function can only be defined once",
                    name
                )));
            }
        } else {
            let token = state.peek_slice(20).unwrap_or("");
            let suggestions = if token.starts_with("type") {
                "Did you mean 'type'? (Note: keywords are case-sensitive)"
            } else if token.starts_with("global") {
                "Did you mean 'global'?"
            } else if token.starts_with("fn") || token.starts_with('@') {
                "Did you mean 'fn' or '@' for a function?"
            } else {
                "Expected one of: 'type', 'global', 'fn', or '@' (for function)"
            };

            return Err(state.error(format!(
                "Unexpected token at top level: {:?}\n  Hint: {}",
                token, suggestions
            )));
        }
    }

    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::parse_module;
    use crate::{
        AllocType, BinaryOp, CmpOp, Instruction, LaminaError, Literal, Module, PrimitiveType, Type,
        Value,
    };
    use std::fs;

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
        let func = module
            .functions
            .get("add")
            .expect("Function @add not found");

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.params[0].name, "a");
        assert_eq!(
            func.signature.params[0].ty,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.signature.params[1].name, "b");
        assert_eq!(
            func.signature.params[1].ty,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(
            func.signature.return_type,
            Type::Primitive(PrimitiveType::I32)
        );
        assert_eq!(func.basic_blocks.len(), 1);
        assert_eq!(func.entry_block, "entry");

        let entry_block = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");
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
        if let Type::Array { element_type, size } = &matrix.ty {
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
        assert_eq!(
            count.initializer.as_ref(),
            Some(&Value::Constant(Literal::I64(10)))
        );

        let message = module.global_declarations.get("message").unwrap();
        assert_eq!(message.name, "message");
        if let Type::Array { element_type, size } = &message.ty {
            assert_eq!(*size, 12);
            assert_eq!(element_type.as_ref(), &Type::Primitive(PrimitiveType::Bool));
        } else {
            panic!("Expected array for message");
        }
        assert_eq!(
            message.initializer.as_ref(),
            Some(&Value::Constant(Literal::String("hello world")))
        );

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
        assert!(matches!(
            block.instructions[0],
            Instruction::Ret {
                ty: Type::Void,
                value: None
            }
        ));
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
            assert!(
                e.to_string()
                    .contains("must end with a terminator instruction")
            );
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
                ret.i32 -42
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module
            .functions
            .get("conditional")
            .expect("Function not found");

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
            }
            _ => panic!("Expected ret instruction"),
        }

        match &negative.instructions[0] {
            Instruction::Ret { ty, value } => {
                assert_eq!(*ty, Type::Primitive(PrimitiveType::I32));
                if let Some(Value::Constant(Literal::I32(val))) = value {
                    assert_eq!(*val, -42);
                } else {
                    panic!("Expected constant -42");
                }
            }
            _ => panic!("Expected ret instruction"),
        }
    }

    #[test]
    fn test_parse_array_and_struct_operations() {
        let input = r#"
            fn @test_composite() -> ptr {
              entry:
                %arr_ptr = alloc.stack [4 x i32]
                %elem_ptr = getelementptr %arr_ptr, 2, i32
                %elem = load.i32 %elem_ptr
                
                %struct_ptr = alloc.stack struct { x: i32, y: i32 }
                %y_ptr = getfieldptr %struct_ptr, 1
                %y_val = load.i32 %y_ptr
                
                ret.ptr %struct_ptr
            }
        "#;

        let module = parse_module(input).expect("Parsing failed");
        let func = module
            .functions
            .get("test_composite")
            .expect("Function not found");

        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");
        assert_eq!(entry.instructions.len(), 7); // alloc, getelementptr, load, alloc, getfieldptr, load, ret

        // Check first instruction is alloc
        match &entry.instructions[0] {
            Instruction::Alloc {
                result,
                allocated_ty,
                alloc_type,
            } => {
                assert_eq!(*result, "arr_ptr");
                if let Type::Array { element_type, size } = allocated_ty {
                    assert_eq!(**element_type, Type::Primitive(PrimitiveType::I32));
                    assert_eq!(*size, 4);
                } else {
                    panic!("Expected array type");
                }
                assert_eq!(*alloc_type, AllocType::Stack);
            }
            _ => panic!("Expected alloc instruction"),
        }

        // Check second instruction is getelementptr
        match &entry.instructions[1] {
            Instruction::GetElemPtr {
                result,
                array_ptr,
                index,
                element_type: _,
            } => {
                assert_eq!(*result, "elem_ptr");
                assert_eq!(*array_ptr, Value::Variable("arr_ptr"));
                assert_eq!(*index, Value::Constant(Literal::I32(2)));
            }
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
        let caller = module
            .functions
            .get("caller")
            .expect("Caller function not found");
        let entry = caller
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        match &entry.instructions[0] {
            Instruction::Call {
                result,
                func_name,
                args,
            } => {
                let result_str: &str = result.as_ref().unwrap();
                let func_str: &str = func_name;
                assert_eq!(result_str, "result");
                assert_eq!(func_str, "callee");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Value::Constant(Literal::I32(5)));
                assert_eq!(args[1], Value::Constant(Literal::I32(10)));
            }
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
        let func = module
            .functions
            .get("test_comparisons")
            .expect("Function not found");
        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        assert_eq!(entry.instructions.len(), 7); // 6 comparisons + ret

        match &entry.instructions[0] {
            Instruction::Cmp {
                op,
                result,
                ty,
                lhs,
                rhs,
            } => {
                assert_eq!(*op, CmpOp::Eq);
                assert_eq!(*result, "eq");
                assert_eq!(*ty, PrimitiveType::I32);
                assert_eq!(*lhs, Value::Variable("a"));
                assert_eq!(*rhs, Value::Variable("b"));
            }
            _ => panic!("Expected comparison instruction"),
        }

        match &entry.instructions[1] {
            Instruction::Cmp { op, result, .. } => {
                assert_eq!(*op, CmpOp::Ne);
                assert_eq!(*result, "ne");
            }
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
        let entry = func
            .basic_blocks
            .get("entry")
            .expect("Entry block not found");

        // Check the add operation uses I8 type
        match &entry.instructions[0] {
            Instruction::Binary { op, ty, .. } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*ty, PrimitiveType::I8);
            }
            _ => panic!("Expected Binary instruction"),
        }

        // Check the zext operation
        match &entry.instructions[1] {
            Instruction::ZeroExtend {
                source_type,
                target_type,
                ..
            } => {
                assert_eq!(*source_type, PrimitiveType::I8);
                assert_eq!(*target_type, PrimitiveType::I64);
            }
            _ => panic!("Expected ZeroExtend instruction"),
        }
    }

    #[test]
    fn test_parse_tensor_benchmark() -> Result<(), LaminaError> {
        // Load the benchmark Lamina code
        let source = fs::read_to_string("benchmarks/2Dmatmul/2Dmatmul.lamina").map_err(|e| {
            LaminaError::ParsingError(format!("Failed to read benchmark file: {}", e))
        })?;

        let module: Module = parse_module(&source)?;

        // Basic Assertions on the parsed module
        assert_eq!(
            module.type_declarations.len(),
            0,
            "Should have no type declarations"
        );
        assert_eq!(
            module.global_declarations.len(),
            0,
            "Should have no global declarations"
        );
        assert_eq!(module.functions.len(), 4, "Should have 4 functions");

        // Check @main function details
        let main_func = module
            .functions
            .get("main")
            .expect("Missing @main function");
        assert_eq!(main_func.name, "main");
        assert!(
            main_func.signature.params.is_empty(),
            "@main should have no parameters"
        );
        assert_eq!(
            main_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64),
            "@main should return i64"
        );
        assert_eq!(
            main_func.basic_blocks.len(),
            1,
            "@main should have 1 basic block"
        );
        let entry_block = main_func
            .basic_blocks
            .get("entry")
            .expect("@main missing entry block");
        assert!(
            entry_block.instructions.len() > 5,
            "@main entry block should have several instructions"
        );
        assert!(
            matches!(
                entry_block.instructions.last(),
                Some(Instruction::Ret { .. })
            ),
            "@main should end with ret"
        );

        // Check @matmul_2d_optimized function details (more complex)
        let matmul_func = module
            .functions
            .get("matmul_2d_optimized")
            .expect("Missing @matmul_2d_optimized function");
        assert_eq!(matmul_func.name, "matmul_2d_optimized");
        assert_eq!(
            matmul_func.signature.params.len(),
            3,
            "@matmul_2d_optimized should have 3 parameters"
        );
        assert_eq!(
            matmul_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64),
            "@matmul_2d_optimized should return i64"
        );

        // Check that it has several basic blocks (it's a complex function)
        assert!(
            matmul_func.basic_blocks.len() >= 7,
            "@matmul_2d_optimized should have at least 7 basic blocks"
        );
        assert!(matmul_func.basic_blocks.contains_key("entry"));

        // Check for get_matrix_a_element function
        let get_a_func = module
            .functions
            .get("get_matrix_a_element")
            .expect("Missing @get_matrix_a_element function");
        assert_eq!(get_a_func.signature.params.len(), 2);

        // Check for get_matrix_b_element function
        let get_b_func = module
            .functions
            .get("get_matrix_b_element")
            .expect("Missing @get_matrix_b_element function");
        assert_eq!(get_b_func.signature.params.len(), 2);
        assert_eq!(
            get_b_func.signature.return_type,
            Type::Primitive(PrimitiveType::I64)
        );
        assert!(get_b_func.basic_blocks.contains_key("entry"));

        Ok(())
    }

    #[test]
    fn test_parse_error_recovery() -> Result<(), LaminaError> {
        // Test that parser gives helpful error messages for common mistakes and handles valid edge cases

        // Missing function body should fail
        let source_missing_body = "fn @test() -> i64";
        assert!(parse_module(source_missing_body).is_err());

        // Malformed instruction (missing operand) should fail
        let source_malformed = r#"
fn @test() -> i64 {
  entry:
    %result = add.i64 42
    ret.i64 %result
}
"#;
        assert!(parse_module(source_malformed).is_err());

        // Undefined variable should parse successfully (variables don't need to be pre-declared)
        let source_undefined_var = r#"
fn @test() -> i64 {
  entry:
    %result = add.i64 %undefined, 42
    ret.i64 %result
}
"#;
        let parsed = parse_module(source_undefined_var)?;
        assert_eq!(parsed.functions.len(), 1);
        assert!(parsed.functions.contains_key("test"));

        Ok(())
    }

    #[test]
    fn test_parse_complex_expressions() -> Result<(), LaminaError> {
        let source = r#"
fn @complex_math(i64 %a, i64 %b, i64 %c) -> i64 {
  entry:
    %temp1 = mul.i64 %a, %b
    %temp2 = add.i64 %temp1, %c
    %temp3 = sub.i64 %temp2, 10
    %result = div.i64 %temp3, 2
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("complex_math").unwrap();
        assert_eq!(func.signature.params.len(), 3);
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 5);

        Ok(())
    }

    #[test]
    fn test_parse_nested_calls() -> Result<(), LaminaError> {
        let source = r#"
fn @outer() -> i64 {
  entry:
    %result = call @middle()
    ret.i64 %result
}

fn @middle() -> i64 {
  entry:
    %result = call @inner(42)
    ret.i64 %result
}

fn @inner(i64 %x) -> i64 {
  entry:
    %result = add.i64 %x, 1
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 3);

        // Check that all functions exist and have correct signatures
        assert!(module.functions.contains_key("outer"));
        assert!(module.functions.contains_key("middle"));
        assert!(module.functions.contains_key("inner"));

        let inner_func = module.functions.get("inner").unwrap();
        assert_eq!(inner_func.signature.params.len(), 1);

        Ok(())
    }

    #[test]
    fn test_parse_phi_complex() -> Result<(), LaminaError> {
        let source = r#"
fn @test_phi_complex(i64 %cond1, i64 %cond2) -> i64 {
  entry:
    %is_true1 = ne.i64 %cond1, 0
    br %is_true1, block_a, block_b

  block_a:
    %val_a = add.i64 %cond2, 10
    jmp merge1

  block_b:
    %val_b = mul.i64 %cond2, 2
    jmp merge1

  merge1:
    %intermediate = phi.i64 [%val_a, block_a], [%val_b, block_b]
    %is_true2 = gt.i64 %intermediate, 5
    br %is_true2, block_c, block_d

  block_c:
    %final_c = add.i64 %intermediate, 100
    jmp merge2

  block_d:
    %final_d = sub.i64 %intermediate, 50
    jmp merge2

  merge2:
    %result = phi.i64 [%final_c, block_c], [%final_d, block_d]
    ret.i64 %result
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_phi_complex").unwrap();

        // Should have 7 basic blocks
        assert_eq!(func.basic_blocks.len(), 7);

        // Check that all expected blocks exist
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("block_a"));
        assert!(func.basic_blocks.contains_key("block_b"));
        assert!(func.basic_blocks.contains_key("merge1"));
        assert!(func.basic_blocks.contains_key("block_c"));
        assert!(func.basic_blocks.contains_key("block_d"));
        assert!(func.basic_blocks.contains_key("merge2"));

        Ok(())
    }

    #[test]
    fn test_parse_memory_operations_comprehensive() -> Result<(), LaminaError> {
        let source = r#"
fn @test_memory_ops() -> i64 {
  entry:
    %ptr1 = alloc.heap i64
    %ptr2 = alloc.heap i64
    store.i64 %ptr1, 42
    store.i64 %ptr2, 24
    %val1 = load.i64 %ptr1
    %val2 = load.i64 %ptr2
    %sum = add.i64 %val1, %val2
    dealloc.heap %ptr1
    dealloc.heap %ptr2
    ret.i64 %sum
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_memory_ops").unwrap();

        assert_eq!(func.basic_blocks["entry"].instructions.len(), 10);

        Ok(())
    }

    #[test]
    fn test_parse_large_function() -> Result<(), LaminaError> {
        // Test that parser can handle functions with many instructions and blocks
        let mut source = String::from(
            r#"
fn @large_function(i64 %input) -> i64 {
  entry:
"#,
        );

        // Add many instructions
        for i in 0..50 {
            source.push_str(&format!("    %temp{} = add.i64 %input, {}\n", i, i));
        }

        source.push_str("    ret.i64 %temp49\n}");

        let module = parse_module(&source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("large_function").unwrap();

        // Should have 51 instructions (50 adds + 1 ret)
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 51);

        Ok(())
    }

    #[test]
    fn test_parse_empty_blocks() -> Result<(), LaminaError> {
        let source = r#"
fn @test_empty_blocks() -> i64 {
  entry:
    jmp block1

  block1:
    jmp block2

  block2:
    ret.i64 42
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_empty_blocks").unwrap();

        assert_eq!(func.basic_blocks.len(), 3);

        // Check that blocks exist even if they only contain jumps
        assert!(func.basic_blocks.contains_key("entry"));
        assert!(func.basic_blocks.contains_key("block1"));
        assert!(func.basic_blocks.contains_key("block2"));

        Ok(())
    }

    #[test]
    fn test_parse_instruction_variations() -> Result<(), LaminaError> {
        let source = r#"
fn @test_instructions() -> i64 {
  entry:
    %a = add.i64 1, 2
    %b = sub.i64 %a, 3
    %c = mul.i64 %b, 4
    %d = div.i64 %c, 2
    %e = rem.i64 %d, 3
    %f = eq.i64 %e, 0
    %g = ne.i64 %f, 0
    %h = lt.i64 %g, 1
    %i = le.i64 %h, 1
    %j = gt.i64 %i, 0
    %k = ge.i64 %j, 0
    print %k
    ret.i64 %k
}
"#;
        let module = parse_module(source)?;

        assert_eq!(module.functions.len(), 1);
        let func = module.functions.get("test_instructions").unwrap();

        // Should have 12 instructions (11 computations + 1 print + 1 ret)
        assert_eq!(func.basic_blocks["entry"].instructions.len(), 13);

        Ok(())
    }

    #[test]
    fn test_parse_minimal_programs() -> Result<(), LaminaError> {
        use crate::ir::IRBuilder;
        use crate::ir::builder::i32 as ir_i32;
        use crate::ir::types::{PrimitiveType, Type};

        // Test very simple programs that should parse correctly and produce correct IR

        // Single instruction function
        let source1 = r#"
fn @minimal() -> i64 {
  entry:
    ret.i64 0
}
"#;
        let mut builder1 = IRBuilder::new();
        builder1
            .function("minimal", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i32(0));
        let expected1 = builder1.build();
        let parsed1 = parse_module(source1)?;
        assert_eq!(parsed1, expected1);

        // Function with just print
        let source2 = r#"
fn @print_only() -> i64 {
  entry:
    print 42
    ret.i64 0
}
"#;
        let mut builder2 = IRBuilder::new();
        builder2
            .function("print_only", Type::Primitive(PrimitiveType::I64))
            .print(ir_i32(42))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i32(0));
        let expected2 = builder2.build();
        let parsed2 = parse_module(source2)?;
        assert_eq!(parsed2, expected2);

        // Empty function (just return)
        let source3 = r#"
fn @empty() -> i64 {
  entry:
    ret.i64 42
}
"#;
        let mut builder3 = IRBuilder::new();
        builder3
            .function("empty", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i32(42));
        let expected3 = builder3.build();
        let parsed3 = parse_module(source3)?;
        assert_eq!(parsed3, expected3);

        Ok(())
    }

    #[test]
    fn test_parse_whitespace_tolerance() -> Result<(), LaminaError> {
        use crate::ir::IRBuilder;
        use crate::ir::builder::i32 as ir_i32;
        use crate::ir::types::{PrimitiveType, Type};

        // Expected IR structure for all test cases
        let mut builder = IRBuilder::new();
        builder
            .function("test", Type::Primitive(PrimitiveType::I64))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i32(0));

        let expected_module = builder.build();

        // Test that parser handles various whitespace patterns and produces correct IR

        // Normal spacing should work and produce correct IR
        let source_normal = r#"
fn @test() -> i64 {
  entry:
    ret.i64 0
}
"#;
        let parsed_normal = parse_module(source_normal)?;
        assert_eq!(parsed_normal, expected_module);

        // Extra spaces should work and produce the same correct IR
        let source_extra_spaces = r#"
fn   @test(  )   ->   i64   {
    entry   :
        ret.i64   0
}
"#;
        let parsed_extra_spaces = parse_module(source_extra_spaces)?;
        assert_eq!(parsed_extra_spaces, expected_module);

        // Minimal spacing should work and produce the same correct IR
        let source_minimal = r#"fn @test() -> i64 {
entry:
ret.i64 0
}"#;
        let parsed_minimal = parse_module(source_minimal)?;
        assert_eq!(parsed_minimal, expected_module);

        Ok(())
    }
}
