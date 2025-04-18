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
