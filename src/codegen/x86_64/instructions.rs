use crate::{
    Instruction, /*Value,*/ Type, PrimitiveType, AllocType, BinaryOp, CmpOp,
    LaminaError, Result, /*Identifier*/
    Identifier,
};
use super::state::{CodegenState, FunctionContext, ValueLocation, ARG_REGISTERS, RETURN_REGISTER};
use super::util::get_value_operand_asm;
use std::io::Write;

// Generate assembly for a single instruction (main translation logic)
pub fn generate_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
    _func_name: Identifier<'a>,
) -> Result<()> {
    writeln!(writer, "        # IR: {}", instr)?; // Comment with the original IR

    match instr {
        Instruction::Ret { ty:_, value } => {
            if let Some(val) = value {
                let src_loc = get_value_operand_asm(val, state, func_ctx)?;
                // Move value to %rax first, regardless of function
                // The caller (C runtime for main, or another func) will read the appropriate register (%eax or %rax).
                writeln!(writer, "        movq {}, %rax  # Potential Return value", src_loc)?;
            }
            // Jump to the unified epilogue instead of emitting leave/ret here
            writeln!(writer, "        jmp {} # Jump to epilogue", func_ctx.epilogue_label)?;
        }

        Instruction::Store { ty, ptr, value } => {
            let val_loc = get_value_operand_asm(value, state, func_ctx)?;
            let ptr_loc = get_value_operand_asm(ptr, state, func_ctx)?;

            // Need temporary registers
            // TODO: Handle immediate values more directly where possible
            writeln!(writer, "        movq {}, %r10 # Store value", val_loc)?;
            writeln!(writer, "        movq {}, %r11 # Store address", ptr_loc)?;
            match ty {
                 Type::Primitive(PrimitiveType::I32) => writeln!(writer, "        movl %r10d, (%r11)")?,
                 Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => 
                     writeln!(writer, "        movq %r10, (%r11)")?,
                 Type::Primitive(PrimitiveType::Bool) => writeln!(writer, "        movb %r10b, (%r11)")?,
                 _ => return Err(LaminaError::CodegenError(format!("Store for type '{}' not implemented yet", ty))),
            }
        }

        Instruction::Alloc { result, alloc_type, .. } => { 
            match alloc_type {
                 AllocType::Stack => {
                    let result_loc = func_ctx.get_value_location(result)?;
                    if let ValueLocation::StackOffset(offset) = result_loc {
                        writeln!(writer, "        leaq {}(%rbp), %rax  # Calculate address for alloc {}", offset, result)?; 
                        writeln!(writer, "        movq %rax, {}(%rbp) # Store alloc ptr result", offset)?; 
                    } else {
                         return Err(LaminaError::CodegenError("Stack allocation result location invalid".to_string()));
                    }
                 }
                 AllocType::Heap => {
                     return Err(LaminaError::CodegenError("Heap allocation requires runtime/libc (malloc)".to_string()));
                 }
             }
        }

        Instruction::Binary { op, result, ty, lhs, rhs } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            let (op_mnemonic, size_suffix, mov_instr, div_instr) = match ty {
                PrimitiveType::I32 => ("l", "d", "movl", "idivl"),
                PrimitiveType::I64 | PrimitiveType::Ptr => ("q", "", "movq", "idivq"), 
                 _ => return Err(LaminaError::CodegenError(format!("Binary op for type '{}' not supported yet", ty)))
            };
            
            // Use correct registers based on size suffix
            let lhs_reg = if size_suffix == "d" { "%eax" } else { "%rax" };
            let rhs_reg = if size_suffix == "d" { "%r10d" } else { "%r10" };

            // Load operands into registers
            writeln!(writer, "        {} {}, {} # Binary/Div LHS (Dividend)", mov_instr, lhs_op, lhs_reg)?; 
            writeln!(writer, "        {} {}, {} # Binary/Div RHS (Divisor)", mov_instr, rhs_op, rhs_reg)?; 

            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    let full_op = match op {
                        BinaryOp::Add => format!("add{}", op_mnemonic),
                        BinaryOp::Sub => format!("sub{}", op_mnemonic),
                        BinaryOp::Mul => format!("imul{}", op_mnemonic), 
                        _ => unreachable!(),
                    };
                    // Perform op: rhs into lhs register
                    writeln!(writer, "        {} {}, {} # Binary Op", full_op, rhs_reg, lhs_reg)?;
                }
                BinaryOp::Div => {
                    match ty {
                        PrimitiveType::I32 => writeln!(writer, "        cltd # Sign extend %eax to %edx")?,
                        PrimitiveType::I64 | PrimitiveType::Ptr => writeln!(writer, "        cqto # Sign extend %rax to %rdx")?,
                        _ => unreachable!(),
                    }
                    // Divide rdx:rax by rhs register
                    writeln!(writer, "        {} {} # Signed division", div_instr, rhs_reg)?; 
                    // Quotient is now in lhs_reg (%rax or %eax)
                }
            };
            // Store result from lhs_reg
            writeln!(writer, "        {} {}, {} # Store Binary/Div Result", mov_instr, lhs_reg, dest_asm)?;
        }
        Instruction::Load { result, ty, ptr } => {
            let ptr_op = get_value_operand_asm(ptr, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            // Select correct mov instruction and temporary register based on type
            let (mov_mnemonic, temp_reg) = match ty {
                 Type::Primitive(PrimitiveType::I8) => ("movsbq", "%r10"), // Sign-extend byte load into 64-bit register
                 Type::Primitive(PrimitiveType::I32) => ("movslq", "%r10"), // Sign-extend load into 64-bit register
                 Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => ("movq", "%r10"),
                 Type::Primitive(PrimitiveType::Bool) => ("movzbq", "%r10"), // Zero-extend byte to quad
                 _ => return Err(LaminaError::CodegenError(format!("Load for type '{}' not implemented yet", ty))),
             };

            // If ptr_op refers to a global via RIP-relative addressing...
            if ptr_op.contains("(%rip)") {
                // Load directly from global memory into the temporary register
                writeln!(writer, "        {} {}, {} # Load global directly", mov_mnemonic, ptr_op, temp_reg)?;
            } else {
                // Otherwise, assume ptr_op holds an address (e.g., stack slot containing a pointer)
                // Load the address itself into %r11 first
                writeln!(writer, "        movq {}, %r11 # Load address from operand", ptr_op)?; 
                // Then load the value from the address in %r11 into the temporary register
                writeln!(writer, "        {} (%r11), {} # Load value from address", mov_mnemonic, temp_reg)?; 
            }
            // Store the final value from the temporary register (%r10) to the destination
            writeln!(writer, "        movq {}, {} # Store loaded result", temp_reg, dest_asm)?; 
        }
        Instruction::Cmp { op, result, ty, lhs, rhs } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            let (cmp_mnemonic, mov_instr, reg_suffix) = match ty {
                PrimitiveType::I8 => ("cmpb", "movb", "b"),
                PrimitiveType::I32 => ("cmpl", "movl", "d"),
                PrimitiveType::I64 | PrimitiveType::Ptr => ("cmpq", "movq", ""),
                PrimitiveType::Bool => ("cmpb", "movb", "b"),
                 _ => return Err(LaminaError::CodegenError(format!("Cmp op for type '{}' not supported yet", ty)))
            };

            // Use %r10 for rhs, %r11 for lhs
            writeln!(writer, "        {} {}, %r10{} # Cmp RHS", mov_instr, rhs_op, reg_suffix)?; 
            writeln!(writer, "        {} {}, %r11{} # Cmp LHS", mov_instr, lhs_op, reg_suffix)?; 
            writeln!(writer, "        {} %r10{}, %r11{} # Cmp Op", cmp_mnemonic, reg_suffix, reg_suffix)?; 

            let set_instr = match op {
                CmpOp::Eq => "sete", CmpOp::Ne => "setne",
                CmpOp::Gt => "setg", CmpOp::Ge => "setge",
                CmpOp::Lt => "setl", CmpOp::Le => "setle",
            };

            writeln!(writer, "        {} %al # Set byte based on flags", set_instr)?;
            // Store the boolean result directly as a byte
            writeln!(writer, "        movb %al, {} # Store Cmp result (byte)", dest_asm)?;
        }
        Instruction::Br { condition, true_label, false_label } => {
             let cond_op = get_value_operand_asm(condition, state, func_ctx)?;
             let true_asm_label = func_ctx.get_block_label(true_label)?;
             let false_asm_label = func_ctx.get_block_label(false_label)?;

             // Load the boolean byte into AL and test it directly
             writeln!(writer, "        movb {}, %al # Load boolean byte", cond_op)?; 
             writeln!(writer, "        testb %al, %al # Test AL, sets ZF if AL is 0")?;
             writeln!(writer, "        jne {} # Jump if condition != 0 (ZF=0)", true_asm_label)?; 
             writeln!(writer, "        jmp {} # Jump if condition == 0 (ZF=1)", false_asm_label)?; 
        }

        Instruction::Call { result, func_name, args } => {
            // 1. Handle arguments (System V ABI)
            let num_args = args.len();
            let num_reg_args = std::cmp::min(num_args, ARG_REGISTERS.len());

            // Move register arguments
            writeln!(writer, "        # Setup register arguments for call @{}", func_name)?;
            for i in 0..num_reg_args {
                let arg_val = &args[i];
                let arg_loc = get_value_operand_asm(arg_val, state, func_ctx)?;
                let reg = ARG_REGISTERS[i];
                // TODO: Check arg type and use movl/movq etc.
                writeln!(writer, "        movq {}, {} # Arg {}", arg_loc, reg, i)?; 
            }

            // Push stack arguments (in reverse order)
            if num_args > num_reg_args {
                 writeln!(writer, "        # Setup stack arguments for call @{}", func_name)?;
                for i in (num_reg_args..num_args).rev() {
                    let arg_val = &args[i];
                    let arg_loc = get_value_operand_asm(arg_val, state, func_ctx)?;
                     // TODO: Check arg type and use movl/movq etc.
                     // Use a temporary register before pushing
                     writeln!(writer, "        movq {}, %r10", arg_loc)?; 
                     writeln!(writer, "        pushq %r10 # Arg {}", i)?; 
                }
            }

            // TODO: Handle vector arguments (XMM registers) - Set %al to 0 for varargs?
            // writeln!(writer, "        xor %al, %al")?;

            // 2. Perform the call
            // Assume func_name maps directly to an assembly label for now.
            // A more robust system might involve looking up the label.
            let call_target = format!("func_{}", func_name);
            writeln!(writer, "        call {}", call_target)?; 

            // 3. Clean up stack arguments (if any)
            let stack_arg_bytes = if num_args > num_reg_args {
                 (num_args - num_reg_args) * 8 // Assuming 8 bytes per stack arg for now
            } else { 0 };

            if stack_arg_bytes > 0 {
                writeln!(writer, "        addq ${}, %rsp # Cleanup stack args", stack_arg_bytes)?;
            }

            // 4. Store the result (if any)
            if let Some(res_name) = result {
                // Assume integer/pointer result in %rax
                let dest_op = func_ctx.get_value_location(res_name)?;
                let dest_asm = dest_op.to_operand_string();
                // TODO: Handle different result types (movl etc.)
                writeln!(writer, "        movq {}, {} # Store call result", RETURN_REGISTER, dest_asm)?; 
            }
        }
        Instruction::Jmp { target_label } => {
            let asm_label = func_ctx.get_block_label(target_label)?;
            writeln!(writer, "        jmp {}", asm_label)?;
        }
        Instruction::GetFieldPtr { result, struct_ptr:_, field_index:_ } => {
             // Need struct layout info (not implemented yet)
             writeln!(writer, "        # GetFieldPtr for {} not implemented (needs struct layout)", result)?;
             let dest_op = func_ctx.get_value_location(result)?;
             writeln!(writer, "        movq $0, {}", dest_op.to_operand_string())?;
        }
         Instruction::GetElemPtr { result, array_ptr, index } => { // Removed non-existent array_ty
            let array_ptr_op = get_value_operand_asm(array_ptr, state, func_ctx)?;
            let index_op = get_value_operand_asm(index, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            // TODO: Need way to determine element size from array_ptr type info
            let element_size: i64 = 8; // Placeholder: Assume 8-byte elements (i64/ptr)

            writeln!(writer, "        movq {}, %rax # GEP Base Ptr", array_ptr_op)?; 
            writeln!(writer, "        movq {}, %r10 # GEP Index", index_op)?; 
            // Ensure index is sign-extended if it's i32? Assume 64-bit for now.
            writeln!(writer, "        imulq ${}, %r10 # GEP Offset = Index * ElemSize", element_size)?; 
            writeln!(writer, "        addq %r10, %rax # GEP Result = Base + Offset")?;
            writeln!(writer, "        movq %rax, {} # Store GEP Result", dest_asm)?;
        }
        Instruction::Print { value } => {
            // Get label for format string "%lld\\n", adding it to state if needed.
            let format_label = state.add_rodata_string("%lld\\n");
            // Get the assembly operand for the value to print
            let val_op = get_value_operand_asm(value, state, func_ctx)?;

            // ABI requires stack to be 16-byte aligned before call
            // Since we are already in a function where alignment is maintained
            // before calls, we assume it's currently aligned here.
            // We need to preserve caller-saved registers if printf modifies them
            // (rax, rcx, rdx, rsi, rdi, r8-r11 are caller-saved)
            // printf uses rdi, rsi, rdx, rcx, r8, r9 for args, clobbers rax.
            // Let's save rax, r10, r11 which might be used by our code.
            writeln!(writer, "        pushq %rax")?;
            writeln!(writer, "        pushq %r10")?;
            writeln!(writer, "        pushq %r11")?;

            // Arguments for printf(format_string, value)
            writeln!(writer, "        leaq {}(%rip), %rdi # Arg 1: Format string address", format_label)?;
            writeln!(writer, "        movq {}, %rsi # Arg 2: Value to print", val_op)?;
            writeln!(writer, "        movl $0, %eax # Variadic call: AL=0 (no FP args)")?;
            writeln!(writer, "        call printf@PLT")?;

            // Restore caller-saved registers
            writeln!(writer, "        popq %r11")?;
            writeln!(writer, "        popq %r10")?;
            writeln!(writer, "        popq %rax")?;
        }
        Instruction::ZeroExtend { result, source_type, target_type, value } => {
            let value_op = get_value_operand_asm(value, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();
            
            match (source_type, target_type) {
                (PrimitiveType::I8, PrimitiveType::I32) => {
                    writeln!(writer, "        movzbl {}, %eax # Zero extend i8->i32", value_op)?;
                    writeln!(writer, "        movl %eax, {} # Store zero-extended result", dest_asm)?;
                },
                (PrimitiveType::I8, PrimitiveType::I64) => {
                    writeln!(writer, "        movzbl {}, %eax # Zero extend i8->i32", value_op)?;
                    writeln!(writer, "        movzlq %eax, %rax # Zero extend i32->i64")?;
                    writeln!(writer, "        movq %rax, {} # Store zero-extended result", dest_asm)?;
                },
                (PrimitiveType::I32, PrimitiveType::I64) => {
                    writeln!(writer, "        movl {}, %eax # Load source i32", value_op)?;
                    writeln!(writer, "        movslq %eax, %rax # Zero extend i32->i64")?;
                    writeln!(writer, "        movq %rax, {} # Store zero-extended result", dest_asm)?;
                },
                (PrimitiveType::Bool, PrimitiveType::I32) => {
                    writeln!(writer, "        movzbl {}, %eax # Zero extend bool->i32", value_op)?;
                    writeln!(writer, "        movl %eax, {} # Store zero-extended result", dest_asm)?;
                },
                (PrimitiveType::Bool, PrimitiveType::I64) => {
                    writeln!(writer, "        movzbl {}, %eax # Zero extend bool->i32", value_op)?;
                    writeln!(writer, "        movzlq %eax, %rax # Zero extend i32->i64")?;
                    writeln!(writer, "        movq %rax, {} # Store zero-extended result", dest_asm)?;
                },
                _ => return Err(LaminaError::CodegenError(format!("Unsupported zero extension: {} to {}", source_type, target_type)))
            }
        },
        _ => {
            writeln!(writer, "        # Instruction {} not implemented yet", instr.to_string().split(' ').next().unwrap_or("Unknown"))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::{Literal, PrimitiveType, Type, Value};
    use crate::ir::instruction::{Instruction, BinaryOp, CmpOp, AllocType};
    use crate::codegen::x86_64::state::{CodegenState, FunctionContext, ValueLocation};
    use std::io::Cursor;
    use indexmap::IndexMap;
    use std::collections::HashMap;

    // Helper to create a default FunctionContext for testing
    fn setup_test_context_basic<'a>() -> FunctionContext<'a> {
        let mut func_ctx = FunctionContext::new();
        func_ctx.epilogue_label = ".Lepilogue_test_func".to_string();
        func_ctx.value_locations.insert("ptr1", ValueLocation::StackOffset(-16));
        func_ctx.value_locations.insert("val1", ValueLocation::StackOffset(-24));
        func_ctx.value_locations.insert("val2", ValueLocation::StackOffset(-32));
        func_ctx.value_locations.insert("result", ValueLocation::StackOffset(-40));
        func_ctx.block_labels.insert("entry", ".Lblock_entry".to_string());
        func_ctx.block_labels.insert("true_block", ".Lblock_true".to_string());
        func_ctx.block_labels.insert("false_block", ".Lblock_false".to_string());
        func_ctx
    }

    // Helper function to create a CodegenState with some globals for testing
    fn setup_test_state_with_globals<'a>() -> CodegenState<'a> {
        let mut state = CodegenState::new();
        state.global_layout.insert("my_global_i32", "global_my_global_i32".to_string());
        state.global_layout.insert("my_global_i64", "global_my_global_i64".to_string());
        state
    }

    // Helper to generate assembly for a single instruction
    // Now accepts state explicitly
    fn generate_test_asm<'a>(instr: &Instruction<'a>, ctx: &FunctionContext<'a>, state: &mut CodegenState<'a>) -> String {
        let mut output = Cursor::new(Vec::new());
        generate_instruction(instr, &mut output, state, ctx, "test_func")
            .expect("Failed to generate instruction");

        String::from_utf8(output.into_inner())
            .expect("Failed to convert output to string")
    }

    // Helper to generate assembly and assert it contains specific snippets
    // Now accepts state explicitly
    fn assert_asm_contains<'a>(instr: &Instruction<'a>, ctx: &FunctionContext<'a>, state: &mut CodegenState<'a>, expected_snippets: &[&str]) {
        let asm = generate_test_asm(instr, ctx, state);
        println!("Generated ASM for {}:\n{}", instr, asm); // Print ASM for debugging
        for snippet in expected_snippets {
            assert!(asm.contains(snippet),
                "ASM output does not contain expected snippet: \nSnippet: \t{:?}\nFull ASM:\n{}", snippet, asm);
        }
    }

    // Helper to assert that generating an instruction results in a CodegenError
    // Now accepts state explicitly
    fn assert_codegen_error<'a>(instr: &Instruction<'a>, ctx: &FunctionContext<'a>, state: &mut CodegenState<'a>, expected_msg_part: &str) {
        let mut output = Cursor::new(Vec::new());
        let result = generate_instruction(instr, &mut output, state, ctx, "test_func");

        match result {
            Err(LaminaError::CodegenError(msg)) => {
                assert!(msg.contains(expected_msg_part),
                    "Expected error message containing \nExpected: \t{:?}\nActual: \t{:?}", expected_msg_part, msg);
            }
            Err(other_err) => {
                panic!("Expected CodegenError, but got different error: {:?}", other_err);
            }
            Ok(_) => {
                panic!("Expected CodegenError, but instruction generation succeeded.");
            }
        }
    }

    // --- Instruction Tests ---

    #[test]
    fn test_ret_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new(); // Globals not needed for ret
        // Test return with value
        let ret_val_instr = Instruction::Ret {
            ty: Type::Primitive(PrimitiveType::I64),
            value: Some(Value::Variable("result")),
        };
        assert_asm_contains(&ret_val_instr, &ctx, &mut state, &[
            "movq -40(%rbp), %rax",
            "jmp .Lepilogue_test_func",
        ]);

        // Test return void (no value)
        let ret_void_instr = Instruction::Ret {
            ty: Type::Void,
            value: None,
        };
        let asm_void = generate_test_asm(&ret_void_instr, &ctx, &mut state);
        assert!(!asm_void.contains("%rax"), "Void return should not touch %rax");
        assert!(asm_void.contains("jmp .Lepilogue_test_func"));
    }

    #[test]
    fn test_store_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = setup_test_state_with_globals();
        // Store i32 to stack var
        let store_i32 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::I32(42)),
        };
        assert_asm_contains(&store_i32, &ctx, &mut state, &[
            "movq $42, %r10",
            "movq -16(%rbp), %r11",
            "movl %r10d, (%r11)",
        ]);

        // Store i64 to stack var
        let store_i64 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("ptr1"),
            value: Value::Variable("val2"),
        };
        assert_asm_contains(&store_i64, &ctx, &mut state, &[
            "movq -32(%rbp), %r10",
            "movq -16(%rbp), %r11",
            "movq %r10, (%r11)",
        ]);

        // Store bool to stack var
        let store_bool = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::Bool(true)),
        };
        assert_asm_contains(&store_bool, &ctx, &mut state, &[
            "movq $1, %r10", // true is 1
            "movq -16(%rbp), %r11",
            "movb %r10b, (%r11)",
        ]);

        // Store Ptr to stack var
        let store_ptr = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Ptr),
            ptr: Value::Variable("ptr1"), // ptr to ptr
            value: Value::Variable("val2"), // address to store (e.g. result of GEP)
        };
        assert_asm_contains(&store_ptr, &ctx, &mut state, &[
            "movq -32(%rbp), %r10", // val2 (address)
            "movq -16(%rbp), %r11", // ptr1 (ptr to ptr)
            "movq %r10, (%r11)",    // Store address into ptr to ptr
        ]);

        // Store i64 to global
        let store_global_i64 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Global("my_global_i64"),
            value: Value::Variable("val1"),
        };
        assert_asm_contains(&store_global_i64, &ctx, &mut state, &[
            "movq -24(%rbp), %r10",       // Load value from val1
            "movq global_my_global_i64(%rip), %r11", // Load address of global into r11
            "movq %r10, (%r11)",          // Store value to global address (indirect)
        ]);

        // Store constant i32 to global
        let store_global_const_i32 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Global("my_global_i32"),
            value: Value::Constant(Literal::I32(99)),
        };
        assert_asm_contains(&store_global_const_i32, &ctx, &mut state, &[
            "movq $99, %r10",              // Load constant value
            "movq global_my_global_i32(%rip), %r11", // Load address of global into r11
            "movl %r10d, (%r11)",          // Store value to global address (indirect)
        ]);

        // Store unsupported type
        let store_err = Instruction::Store {
            ty: Type::Void, // Invalid type to store
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::I32(0)),
        };
        assert_codegen_error(&store_err, &ctx, &mut state, "Store for type");
    }

    #[test]
    fn test_alloc_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Stack allocation
        let alloc_stack = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_asm_contains(&alloc_stack, &ctx, &mut state, &[
            "leaq -40(%rbp), %rax", // Get address of stack slot
            "movq %rax, -40(%rbp)", // Store address in itself (alloc returns pointer)
        ]);

        // Heap allocation (should error)
        let alloc_heap = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Heap,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_codegen_error(&alloc_heap, &ctx, &mut state, "Heap allocation requires runtime");

        // Stack allocation with invalid location (should error)
        let mut ctx_invalid_loc = setup_test_context_basic();
        ctx_invalid_loc.value_locations.insert("result", ValueLocation::Register("%rax".to_string())); // Invalid location for alloc result
        let alloc_stack_invalid = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
         assert_codegen_error(&alloc_stack_invalid, &ctx_invalid_loc, &mut state, "Stack allocation result location invalid");
    }

    #[test]
    fn test_binary_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Add i32
        let add_i32 = Instruction::Binary {
            op: BinaryOp::Add,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(5)),
        };
        assert_asm_contains(&add_i32, &ctx, &mut state, &[
            "movl -24(%rbp), %eax",
            "movl $5, %r10d",
            "addl %r10d, %eax",
            "movl %eax, -40(%rbp)",
        ]);

        // Sub i64
        let sub_i64 = Instruction::Binary {
            op: BinaryOp::Sub,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(&sub_i64, &ctx, &mut state, &[
            "movq -24(%rbp), %rax",
            "movq -32(%rbp), %r10",
            "subq %r10, %rax",
            "movq %rax, -40(%rbp)",
        ]);

        // Mul ptr (treated as i64)
        let mul_ptr = Instruction::Binary {
            op: BinaryOp::Mul,
            result: "result",
            ty: PrimitiveType::Ptr,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I64(8)),
        };
        assert_asm_contains(&mul_ptr, &ctx, &mut state, &[
            "movq -24(%rbp), %rax",
            "movq $8, %r10",
            "imulq %r10, %rax",
            "movq %rax, -40(%rbp)",
        ]);

        // Div i32
        let div_i32 = Instruction::Binary {
            op: BinaryOp::Div,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(2)),
        };
        assert_asm_contains(&div_i32, &ctx, &mut state, &[
            "movl -24(%rbp), %eax", // Dividend
            "movl $2, %r10d",      // Divisor
            "cltd",                 // Sign extend eax to edx:eax
            "idivl %r10d",          // Divide edx:eax by r10d
            "movl %eax, -40(%rbp)", // Store quotient (in eax)
        ]);

        // Div i64
        let div_i64 = Instruction::Binary {
            op: BinaryOp::Div,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(&div_i64, &ctx, &mut state, &[
            "movq -24(%rbp), %rax",
            "movq -32(%rbp), %r10",
            "cqto",                // Sign extend rax to rdx:rax
            "idivq %r10",          // Divide rdx:rax by r10
            "movq %rax, -40(%rbp)", // Store quotient (in rax)
        ]);

        // Binary op unsupported type
        let bin_err = Instruction::Binary {
            op: BinaryOp::Add,
            result: "result",
            ty: PrimitiveType::Bool, // Invalid type for add
            lhs: Value::Constant(Literal::Bool(true)),
            rhs: Value::Constant(Literal::Bool(false)),
        };
        assert_codegen_error(&bin_err, &ctx, &mut state, "Binary op for type");
    }

    #[test]
    fn test_load_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = setup_test_state_with_globals();
        // Load i8 from stack
        let load_i8 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I8),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(&load_i8, &ctx, &mut state, &[
            "movq -16(%rbp), %r11", // Load address
            "movsbq (%r11), %r10",  // Load byte and sign-extend to quad
            "movq %r10, -40(%rbp)", // Store result
        ]);
        
        // Load i32 from stack
        let load_i32 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(&load_i32, &ctx, &mut state, &[
            "movq -16(%rbp), %r11",
            "movslq (%r11), %r10",  // Load long and sign-extend to quad
            "movq %r10, -40(%rbp)",
        ]);

        // Load i64 from stack
        let load_i64 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(&load_i64, &ctx, &mut state, &[
            "movq -16(%rbp), %r11",
            "movq (%r11), %r10",   // Load quad
            "movq %r10, -40(%rbp)",
        ]);

        // Load bool from stack
        let load_bool = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(&load_bool, &ctx, &mut state, &[
            "movq -16(%rbp), %r11",
            "movzbq (%r11), %r10", // Load byte and zero-extend to quad
            "movq %r10, -40(%rbp)",
        ]);

        // Load Ptr from stack
        let load_ptr = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::Ptr),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(&load_ptr, &ctx, &mut state, &[
            "movq -16(%rbp), %r11",
            "movq (%r11), %r10",   // Load quad
            "movq %r10, -40(%rbp)",
        ]);

        // Load i32 from global
        let load_global_i32 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Global("my_global_i32"),
        };
        assert_asm_contains(&load_global_i32, &ctx, &mut state, &[
            "movslq global_my_global_i32(%rip), %r10", // Load global i32 and sign-extend
            "movq %r10, -40(%rbp)",               // Store result
        ]);

        // Load i64 from global
        let load_global_i64 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Global("my_global_i64"),
        };
        assert_asm_contains(&load_global_i64, &ctx, &mut state, &[
            "movq global_my_global_i64(%rip), %r10", // Load global i64 directly
            "movq %r10, -40(%rbp)",               // Store result
        ]);

        // Load unsupported type
        let load_err = Instruction::Load {
            result: "result",
            ty: Type::Void, // Invalid type
            ptr: Value::Variable("ptr1"),
        };
        assert_codegen_error(&load_err, &ctx, &mut state, "Load for type");
    }

    #[test]
    fn test_cmp_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Cmp i8
        let cmp_i8 = Instruction::Cmp {
            op: CmpOp::Eq,
            result: "result",
            ty: PrimitiveType::I8,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I8(10)),
        };
        assert_asm_contains(&cmp_i8, &ctx, &mut state, &[
            "movb $10, %r10b",
            "movb -24(%rbp), %r11b",
            "cmpb %r10b, %r11b",
            "sete %al",
            "movb %al, -40(%rbp)",
        ]);
        
        // Cmp i32
        let cmp_i32 = Instruction::Cmp {
            op: CmpOp::Ne,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(10)),
        };
        assert_asm_contains(&cmp_i32, &ctx, &mut state, &[
            "movl $10, %r10d",
            "movl -24(%rbp), %r11d",
            "cmpl %r10d, %r11d",
            "setne %al",
            "movb %al, -40(%rbp)",
        ]);
        
        // Cmp i64
        let cmp_i64 = Instruction::Cmp {
            op: CmpOp::Gt,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(&cmp_i64, &ctx, &mut state, &[
            "movq -32(%rbp), %r10",
            "movq -24(%rbp), %r11",
            "cmpq %r10, %r11",
            "setg %al",
            "movb %al, -40(%rbp)",
        ]);

        // Cmp bool
        let cmp_bool = Instruction::Cmp {
            op: CmpOp::Eq,
            result: "result",
            ty: PrimitiveType::Bool,
            lhs: Value::Constant(Literal::Bool(true)),
            rhs: Value::Variable("val1"),
        };
        assert_asm_contains(&cmp_bool, &ctx, &mut state, &[
            "movb -24(%rbp), %r10b",
            "movb $1, %r11b",
            "cmpb %r10b, %r11b",
            "sete %al",
            "movb %al, -40(%rbp)",
        ]);

        // Cmp ptr
        let cmp_ptr = Instruction::Cmp {
            op: CmpOp::Le,
            result: "result",
            ty: PrimitiveType::Ptr,
            lhs: Value::Variable("ptr1"),
            rhs: Value::Variable("val1"), // Comparing address with integer value
        };
        assert_asm_contains(&cmp_ptr, &ctx, &mut state, &[
            "movq -24(%rbp), %r10",
            "movq -16(%rbp), %r11",
            "cmpq %r10, %r11",
            "setle %al",
            "movb %al, -40(%rbp)",
        ]);

        // Cmp unsupported type
        let cmp_err = Instruction::Cmp {
            op: CmpOp::Eq,
            result: "result",
            ty: PrimitiveType::F32, // Invalid type for this codegen
            lhs: Value::Constant(Literal::I32(0)), // Placeholder values
            rhs: Value::Constant(Literal::I32(0)),
        };
        assert_codegen_error(&cmp_err, &ctx, &mut state, "Cmp op for type");
    }

    #[test]
    fn test_br_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        let br_instr = Instruction::Br {
            condition: Value::Variable("val1"), // Assumed bool stored here
            true_label: "true_block",
            false_label: "false_block",
        };
        assert_asm_contains(&br_instr, &ctx, &mut state, &[
            "movb -24(%rbp), %al",
            "testb %al, %al",
            "jne .Lblock_true",
            "jmp .Lblock_false",
        ]);
    }

    #[test]
    fn test_jmp_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        let jmp_instr = Instruction::Jmp {
            target_label: "true_block",
        };
        assert_asm_contains(&jmp_instr, &ctx, &mut state, &["jmp .Lblock_true"]);
    }

    #[test]
    fn test_call_instruction() {
        let mut ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Add locations for arguments if needed by get_value_operand_asm
        ctx.value_locations.insert("arg1", ValueLocation::StackOffset(-48)); 
        ctx.value_locations.insert("arg2", ValueLocation::StackOffset(-56)); 
        ctx.value_locations.insert("arg7", ValueLocation::StackOffset(-64)); // For stack arg

        // Call with 2 register args and result
        let call_reg_args = Instruction::Call {
            result: Some("result"),
            func_name: "callee",
            args: vec![Value::Variable("arg1"), Value::Variable("arg2")],
        };
        assert_asm_contains(&call_reg_args, &ctx, &mut state, &[
            "# Setup register arguments for call @callee",
            "movq -48(%rbp), %rdi # Arg 0",
            "movq -56(%rbp), %rsi # Arg 1",
            "call func_callee",
            "movq %rax, -40(%rbp) # Store call result",
        ]);

        // Call with 7 args (6 reg, 1 stack) and no result
        let call_stack_args = Instruction::Call {
            result: None,
            func_name: "callee_many",
            args: vec![
                Value::Constant(Literal::I64(1)),
                Value::Constant(Literal::I64(2)),
                Value::Constant(Literal::I64(3)),
                Value::Constant(Literal::I64(4)),
                Value::Constant(Literal::I64(5)),
                Value::Constant(Literal::I64(6)),
                Value::Variable("arg7"), // 7th arg from stack
            ],
        };
        assert_asm_contains(&call_stack_args, &ctx, &mut state, &[
            "# Setup register arguments for call @callee_many",
            "movq $1, %rdi # Arg 0",
            "movq $2, %rsi # Arg 1",
            "movq $3, %rdx # Arg 2",
            "movq $4, %rcx # Arg 3",
            "movq $5, %r8 # Arg 4",
            "movq $6, %r9 # Arg 5",
            "# Setup stack arguments for call @callee_many",
            "movq -64(%rbp), %r10", // Load 7th arg into temp reg
            "pushq %r10 # Arg 6",     // Push 7th arg
            "call func_callee_many",
            "addq $8, %rsp # Cleanup stack args", // Clean up 1 stack arg (8 bytes)
        ]);
        // Check result storage is NOT present
        let asm_stack = generate_test_asm(&call_stack_args, &ctx, &mut state);
        assert!(!asm_stack.contains("Store call result"));
    }

    #[test]
    fn test_getfieldptr_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Currently outputs a comment and zero
        let getfield_instr = Instruction::GetFieldPtr {
            result: "result",
            struct_ptr: Value::Variable("ptr1"),
            field_index: 1,
        };
        assert_asm_contains(&getfield_instr, &ctx, &mut state, &[
            "# GetFieldPtr for result not implemented",
            "movq $0, -40(%rbp)", // Store 0 in result
        ]);
    }

    #[test]
    fn test_getelementptr_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // GEP with variable index (i64)
        let gep_var_idx = Instruction::GetElemPtr {
            result: "result",
            array_ptr: Value::Variable("ptr1"),
            index: Value::Variable("val1"), // Assume val1 holds i64 index
        };
        assert_asm_contains(&gep_var_idx, &ctx, &mut state, &[
            "movq -16(%rbp), %rax",  // Load array pointer (ptr1)
            "movq -24(%rbp), %r10",  // Load index (val1)
            "imulq $8, %r10",        // Multiply by element size (hardcoded 8 for now)
            "addq %r10, %rax",       // Calculate address
            "movq %rax, -40(%rbp)",  // Store result pointer (result)
        ]);
        
        // GEP with constant index (i32)
        let gep_const_idx = Instruction::GetElemPtr {
            result: "result",
            array_ptr: Value::Variable("ptr1"),
            index: Value::Constant(Literal::I32(3)),
        };
        assert_asm_contains(&gep_const_idx, &ctx, &mut state, &[
            "movq -16(%rbp), %rax",
            "movq $3, %r10",
            "imulq $8, %r10",
            "addq %r10, %rax",
            "movq %rax, -40(%rbp)",
        ]);
    }

    #[test]
    fn test_print_instruction() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        let print_instr = Instruction::Print {
            value: Value::Variable("val1"),
        };
        // Need to call it once to ensure the label exists in state
        let _ = generate_test_asm(&print_instr, &ctx, &mut state);
        // Now assert with the state possibly modified
        assert_asm_contains(&print_instr, &ctx, &mut state, &[
            "leaq .L.rodata_str_0(%rip), %rdi", // Assuming first rodata string
            "movq -24(%rbp), %rsi",
            "movl $0, %eax",
            "call printf@PLT",
            "pushq %rax", // Check registers are saved/restored
            "popq %rax",
            "pushq %r10",
            "popq %r10",
            "pushq %r11",
            "popq %r11",
        ]);
    }

    #[test]
    fn test_zeroextend_instructions() {
        let ctx = setup_test_context_basic();
        let mut state = CodegenState::new();
        // Zext i8 -> i32
        let zext_i8_i32 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I8,
            target_type: PrimitiveType::I32,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(&zext_i8_i32, &ctx, &mut state, &[
            "movzbl -24(%rbp), %eax",
            "movl %eax, -40(%rbp)",
        ]);
        
        // Zext i8 -> i64
        let zext_i8_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I8,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(&zext_i8_i64, &ctx, &mut state, &[
            "movzbl -24(%rbp), %eax",
            "movzlq %eax, %rax",
            "movq %rax, -40(%rbp)",
        ]);
        
        // Zext i32 -> i64
        let zext_i32_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I32,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(&zext_i32_i64, &ctx, &mut state, &[
            "movl -24(%rbp), %eax",
            "movslq %eax, %rax", 
            "movq %rax, -40(%rbp)",
        ]);

        // Zext bool -> i32
        let zext_bool_i32 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::Bool,
            target_type: PrimitiveType::I32,
            value: Value::Constant(Literal::Bool(true)),
        };
        assert_asm_contains(&zext_bool_i32, &ctx, &mut state, &[
            "movzbl $1, %eax", // Load constant bool and zero-extend
            "movl %eax, -40(%rbp)",
        ]);

        // Zext bool -> i64
        let zext_bool_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::Bool,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(&zext_bool_i64, &ctx, &mut state, &[
            "movzbl -24(%rbp), %eax",
            "movzlq %eax, %rax",
            "movq %rax, -40(%rbp)",
        ]);

        // Zext invalid types
        let zext_err = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I64,
            target_type: PrimitiveType::I32,
            value: Value::Variable("val1"),
        };
        assert_codegen_error(&zext_err, &ctx, &mut state, "Unsupported zero extension");
    }
}
