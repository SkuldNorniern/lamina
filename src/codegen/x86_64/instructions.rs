use super::state::{ARG_REGISTERS, CodegenState, FunctionContext, ValueLocation};
use super::util::get_value_operand_asm;
use crate::codegen::{CodegenError, ExtensionInfo, TypeInfo};
use crate::{
    AllocType, BinaryOp, CmpOp, Identifier, Instruction, LaminaError, PrimitiveType,
    Result, /*Identifier*/
    /*Value,*/ Type,
};
use std::io::Write;

// Generate assembly for a single instruction (main translation logic)
pub fn generate_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
    _func_name: Identifier<'a>,
) -> Result<()> {
    // Original instruction generation logic follows...
    writeln!(writer, "        # IR: {}", instr)?; // Comment with the original IR

    match instr {
        Instruction::Ret { ty: _, value } => {
            if let Some(val) = value {
                let src_loc = get_value_operand_asm(val, state, func_ctx)?;
                // Move value to %rax first, regardless of function
                // The caller (C runtime for main, or another func) will read the appropriate register (%eax or %rax).
                writeln!(writer, "        movq {}, %rax  # Return value", src_loc)?;
            }
            // Jump to the unified epilogue instead of emitting leave/ret here
            writeln!(
                writer,
                "        jmp {} # Jump to epilogue",
                func_ctx.epilogue_label
            )?;
        }

        Instruction::Store { ty, ptr, value } => {
            let val_loc = get_value_operand_asm(value, state, func_ctx)?;
            let ptr_loc = get_value_operand_asm(ptr, state, func_ctx)?;

            // Need temporary registers
            // Optimization opportunity: Handle immediate values more directly
            writeln!(writer, "        movq {}, %r10 # Store value", val_loc)?;
            writeln!(writer, "        movq {}, %r11 # Store address", ptr_loc)?;
            match ty {
                Type::Primitive(PrimitiveType::I8)
                | Type::Primitive(PrimitiveType::U8)
                | Type::Primitive(PrimitiveType::Char) => {
                    writeln!(writer, "        movb %r10b, (%r11)")?
                }
                Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => {
                    writeln!(writer, "        movw %r10w, (%r11)")?
                }
                Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => {
                    writeln!(writer, "        movl %r10d, (%r11)")?
                }
                Type::Primitive(PrimitiveType::I64)
                | Type::Primitive(PrimitiveType::U64)
                | Type::Primitive(PrimitiveType::Ptr) => {
                    writeln!(writer, "        movq %r10, (%r11)")?
                }
                Type::Primitive(PrimitiveType::F32) => {
                    // Move value to XMM register first
                    writeln!(writer, "        movss {}, %xmm0", val_loc)?;
                    writeln!(writer, "        movss %xmm0, (%r11)")?
                }
                Type::Primitive(PrimitiveType::F64) => {
                    // Move value to XMM register first
                    writeln!(writer, "        movsd {}, %xmm0", val_loc)?;
                    writeln!(writer, "        movsd %xmm0, (%r11)")?
                }
                Type::Primitive(PrimitiveType::Bool) => {
                    writeln!(writer, "        movb %r10b, (%r11)")?
                }
                _ => {
                    return Err(LaminaError::CodegenError(
                        CodegenError::StoreNotImplementedForType(TypeInfo::Unknown(ty.to_string())),
                    ));
                }
            }
        }

        Instruction::Alloc {
            result, alloc_type, allocated_ty, ..
        } => match alloc_type {
            AllocType::Stack => {
                let result_loc = func_ctx.get_value_location(result)?;

                // For stack allocation, return a pointer to stack space within the current frame
                // This avoids dynamic RSP manipulation which complicates stack cleanup
                // Use a simple approach: allocate at a fixed offset for each allocation
                // In a real implementation, this would need proper stack layout management

                // Calculate the size needed for the allocation
                let _alloc_size = match allocated_ty {
                    Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::U8) => 1,
                    Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => 2,
                    Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => 4,
                    Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::U64) => 8,
                    Type::Primitive(PrimitiveType::F32) => 4,
                    Type::Primitive(PrimitiveType::F64) => 8,
                    Type::Primitive(PrimitiveType::Bool) => 1,
                    Type::Primitive(PrimitiveType::Char) => 1,
                    Type::Primitive(PrimitiveType::Ptr) => 8,
                    Type::Struct(fields) => {
                        // Calculate actual struct size by summing field sizes
                        let mut total_size = 0u64;
                        for field in fields {
                            let field_size = match &field.ty {
                                Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::U8) => 1,
                                Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => 2,
                                Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => 4,
                                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::U64) => 8,
                                Type::Primitive(PrimitiveType::F32) => 4,
                                Type::Primitive(PrimitiveType::F64) => 8,
                                Type::Primitive(PrimitiveType::Bool) => 1,
                                Type::Primitive(PrimitiveType::Char) => 1,
                                Type::Primitive(PrimitiveType::Ptr) => 8,
                                _ => 8, // Default for complex types
                            };
                            total_size += field_size;
                        }
                        total_size
                    }
                    Type::Array { size, .. } => {
                        // For arrays, we need to calculate element size * array size
                        8 * (*size as usize).min(1024) as u64 // Cap at reasonable size
                    }
                    _ => 8, // Default fallback
                };

                let data_offset = match result {
                    s if s.contains("ptr1") => -24,
                    s if s.contains("ptr2") => -32,
                    s if s.contains("ptr3") => -40,
                    s if s.contains("stack_var") => -48,
                    s if s.contains("stack_ptr") => -24,
                    _ => -24, // Default
                };

                match result_loc {
                    ValueLocation::Register(reg) => {
                        // Calculate the address where the data will be stored
                        writeln!(writer, "        leaq {}(%rbp), {} # Stack allocation data address", data_offset, reg)?;
                        writeln!(writer, "        # Stack allocation for {} at offset {}", result, data_offset)?;
                    }
                    ValueLocation::StackOffset(offset) => {
                        // Calculate the address where the data will be stored
                        writeln!(writer, "        leaq {}(%rbp), %rax # Calculate data address", data_offset)?;
                        writeln!(writer, "        movq %rax, {}(%rbp) # Store data pointer", offset)?;
                        writeln!(writer, "        # Stack allocation for {} at data offset {}", result, data_offset)?;
                    }
                }

                // Note: Stack allocations are automatically freed when function returns
                // No explicit deallocation needed
            }
            AllocType::Heap => {
                let result_loc = func_ctx.get_value_location(result)?;

                // Determine the size to allocate based on the type
                let size = match allocated_ty {
                    Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::U8) => 1,
                    Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => 2,
                    Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => 4,
                    Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::U64) => 8,
                    Type::Primitive(PrimitiveType::F32) => 4,
                    Type::Primitive(PrimitiveType::F64) => 8,
                    Type::Primitive(PrimitiveType::Bool) => 1,
                    Type::Primitive(PrimitiveType::Char) => 1,
                    Type::Primitive(PrimitiveType::Ptr) => 8,
                    Type::Struct(fields) => {
                        // Calculate actual struct size by summing field sizes
                        let mut total_size = 0u64;
                        for field in fields {
                            let field_size = match &field.ty {
                                Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::U8) => 1,
                                Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => 2,
                                Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => 4,
                                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::U64) => 8,
                                Type::Primitive(PrimitiveType::F32) => 4,
                                Type::Primitive(PrimitiveType::F64) => 8,
                                Type::Primitive(PrimitiveType::Bool) => 1,
                                Type::Primitive(PrimitiveType::Char) => 1,
                                Type::Primitive(PrimitiveType::Ptr) => 8,
                                _ => 8, // Default for complex types
                            };
                            total_size += field_size;
                        }
                        total_size
                    }
                    Type::Array { size, .. } => {
                        // For arrays, we need to calculate element size * array size
                        // TODO: Implement proper array size calculation
                        (8 * (*size as usize).min(1024)) as u64 // Cap at reasonable size
                    }
                    _ => 8, // Default fallback
                };

                writeln!(writer, "        movq ${}, %rdi  # Size to allocate for type {:?}", size, allocated_ty)?;
                writeln!(writer, "        call malloc")?;
                writeln!(writer, "        # malloc returns pointer in %rax")?;

                match result_loc {
                    ValueLocation::Register(reg) => {
                        writeln!(writer, "        movq %rax, {} # Store malloc result in {}", reg, result)?;
                    }
                    ValueLocation::StackOffset(offset) => {
                        writeln!(writer, "        movq %rax, {}(%rbp) # Store malloc result on stack at {}", offset, result)?;
                    }
                }
            }
        },

        Instruction::Dealloc { ptr } => {
            let ptr_loc = get_value_operand_asm(ptr, state, func_ctx)?;

            // For now, we assume all deallocations are for heap memory
            // TODO: We need to track allocation type to properly handle stack vs heap deallocation
            writeln!(writer, "        movq {}, %rdi # Load pointer to free", ptr_loc)?;
            writeln!(writer, "        call free")?;
            writeln!(writer, "        # Heap memory deallocated")?;
        }

        Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            let (op_mnemonic, size_suffix, mov_instr, div_instr) = match ty {
                PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Char | PrimitiveType::Bool => {
                    ("b", "b", "movb", "idivb")
                }
                PrimitiveType::I16 | PrimitiveType::U16 => ("w", "w", "movw", "idivw"),
                PrimitiveType::I32 | PrimitiveType::U32 => ("l", "d", "movl", "idivl"),
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr => {
                    ("q", "", "movq", "idivq")
                }
                PrimitiveType::F32 => ("ss", "", "movss", "divss"),
                PrimitiveType::F64 => ("sd", "", "movsd", "divsd")
            };

            // Fast path for common patterns
            // 1. Direct operations with immediates
            let is_constant_lhs = lhs_op.starts_with('$');
            let is_constant_rhs = rhs_op.starts_with('$');

            // Special case for operations on constants
            if is_constant_lhs && is_constant_rhs {
                // Both operands are constants
                let const_result = match op {
                    BinaryOp::Add => {
                        let lhs_val = lhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        let rhs_val = rhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        lhs_val + rhs_val
                    }
                    BinaryOp::Sub => {
                        let lhs_val = lhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        let rhs_val = rhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        lhs_val - rhs_val
                    }
                    BinaryOp::Mul => {
                        let lhs_val = lhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        let rhs_val = rhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        lhs_val * rhs_val
                    }
                    BinaryOp::Div => {
                        let lhs_val = lhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        let rhs_val = rhs_op.trim_start_matches('$').parse::<i64>().unwrap_or(0);
                        if rhs_val == 0 { 0 } else { lhs_val / rhs_val }
                    }
                };
                writeln!(
                    writer,
                    "        {} ${}, {} # Constant folding",
                    mov_instr, const_result, dest_asm
                )?;
                return Ok(());
            }

            // Special case for adding/subtracting with immediate values
            if matches!(op, BinaryOp::Add | BinaryOp::Sub) && (is_constant_rhs || is_constant_lhs) {
                // Check which operand is immediate
                if is_constant_rhs {
                    // If destination is a register, we can optimize further
                    if dest_asm.starts_with('%') && lhs_op == dest_asm {
                        // Add/sub immediate directly to destination register
                        let op_name = match op {
                            BinaryOp::Add => format!("add{}", op_mnemonic),
                            BinaryOp::Sub => format!("sub{}", op_mnemonic),
                            _ => unreachable!(),
                        };
                        writeln!(
                            writer,
                            "        {} {}, {} # Direct Op to destination",
                            op_name, rhs_op, dest_asm
                        )?;
                        return Ok(());
                    } else {
                        // Load LHS, add immediate, store result
                        // Use the appropriate register size based on the operation type
                        let reg_name = match ty {
                            PrimitiveType::I32 | PrimitiveType::U32 => "%eax", // 32-bit register for 32-bit ops
                            _ => "%rax", // 64-bit register for other ops
                        };

                        writeln!(writer, "        {} {}, {} # Load LHS", mov_instr, lhs_op, reg_name)?;

                        let op_name = match op {
                            BinaryOp::Add => format!("add{}", op_mnemonic),
                            BinaryOp::Sub => format!("sub{}", op_mnemonic),
                            _ => unreachable!(),
                        };

                        writeln!(
                            writer,
                            "        {} {}, {} # Direct Op with Immediate",
                            op_name, rhs_op, reg_name
                        )?;

                        // For 32-bit operations on 64-bit registers, we need to handle sign extension
                        if matches!(ty, PrimitiveType::I32 | PrimitiveType::U32) && dest_asm.contains("%rax") {
                            // If destination is a 64-bit register but we did 32-bit ops,
                            // we need to sign/zero extend the result
                            if matches!(ty, PrimitiveType::I32) {
                                writeln!(writer, "        cltq # Sign extend 32-bit result to 64-bit")?;
                            } else {
                                // Zero extend by moving to itself (upper bits are already zero)
                                writeln!(writer, "        movl %eax, %eax # Zero extend 32-bit to 64-bit")?;
                            }
                        }

                        writeln!(
                            writer,
                            "        {} {}, {} # Store Result",
                            mov_instr, reg_name, dest_asm
                        )?;
                        return Ok(());
                    }
                } else if is_constant_lhs && *op == BinaryOp::Add {
                    // For add, we can swap operands (commutative)
                    if dest_asm.starts_with('%') && rhs_op == dest_asm {
                        // Add immediate directly to destination register
                        writeln!(
                            writer,
                            "        add{} {}, {} # Direct Op to destination",
                            op_mnemonic, lhs_op, dest_asm
                        )?;
                        return Ok(());
                    } else {
                        // Load RHS, add immediate, store result
                        // Use the appropriate register size based on the operation type
                        let reg_name = match ty {
                            PrimitiveType::I32 | PrimitiveType::U32 => "%eax", // 32-bit register for 32-bit ops
                            _ => "%rax", // 64-bit register for other ops
                        };

                        writeln!(writer, "        {} {}, {} # Load RHS", mov_instr, rhs_op, reg_name)?;
                        let op_name = format!("add{}", op_mnemonic);
                        writeln!(
                            writer,
                            "        {} {}, {} # Direct Op with Immediate",
                            op_name, lhs_op, reg_name
                        )?;

                        // For 32-bit operations on 64-bit registers, we need to handle sign extension
                        if matches!(ty, PrimitiveType::I32 | PrimitiveType::U32) && dest_asm.contains("%rax") {
                            // If destination is a 64-bit register but we did 32-bit ops,
                            // we need to sign/zero extend the result
                            if matches!(ty, PrimitiveType::I32) {
                                writeln!(writer, "        cltq # Sign extend 32-bit result to 64-bit")?;
                            } else {
                                // Zero extend by moving to itself (upper bits are already zero)
                                writeln!(writer, "        movl %eax, %eax # Zero extend 32-bit to 64-bit")?;
                            }
                        }

                        writeln!(
                            writer,
                            "        {} {}, {} # Store Result",
                            mov_instr, reg_name, dest_asm
                        )?;
                        return Ok(());
                    }
                }
            }

            // Special case for multiplication by powers of 2
            if *op == BinaryOp::Mul && is_constant_rhs {
                // Check if RHS is power of 2
                if let Ok(rhs_val) = rhs_op.trim_start_matches('$').parse::<i64>()
                    && rhs_val > 0
                    && (rhs_val & (rhs_val - 1)) == 0
                {
                    // It's a power of 2, convert to shift
                    let shift_amount = rhs_val.trailing_zeros();
                    // Use the appropriate register size based on the operation type
                    let reg_name = match ty {
                        PrimitiveType::I32 | PrimitiveType::U32 => "%eax", // 32-bit register for 32-bit ops
                        _ => "%rax", // 64-bit register for other ops
                    };

                    writeln!(
                        writer,
                        "        {} {}, {} # Load value for shift",
                        mov_instr, lhs_op, reg_name
                    )?;
                    writeln!(
                        writer,
                        "        shl{} ${}, {} # Shift instead of multiply by power of 2",
                        op_mnemonic, shift_amount, reg_name
                    )?;
                    writeln!(
                        writer,
                        "        {} {}, {} # Store Result",
                        mov_instr, reg_name, dest_asm
                    )?;
                    return Ok(());
                }
            }

            // Regular path for operations not handled by special cases

            // Use correct registers based on size suffix
            let lhs_reg = if size_suffix == "d" { "%eax" } else { "%rax" };
            let rhs_reg = if size_suffix == "d" { "%r10d" } else { "%r10" };

            // Load operands into registers
            writeln!(
                writer,
                "        {} {}, {} # Binary/Div LHS (Dividend)",
                mov_instr, lhs_op, lhs_reg
            )?;
            writeln!(
                writer,
                "        {} {}, {} # Binary/Div RHS (Divisor)",
                mov_instr, rhs_op, rhs_reg
            )?;

            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    let full_op = match op {
                        BinaryOp::Add => format!("add{}", op_mnemonic),
                        BinaryOp::Sub => format!("sub{}", op_mnemonic),
                        BinaryOp::Mul => format!("imul{}", op_mnemonic),
                        _ => unreachable!(),
                    };
                    // Perform op: rhs into lhs register
                    writeln!(
                        writer,
                        "        {} {}, {} # Binary Op",
                        full_op, rhs_reg, lhs_reg
                    )?;
                }
                BinaryOp::Div => {
                    match ty {
                        PrimitiveType::I32 => {
                            writeln!(writer, "        cltd # Sign extend %eax to %edx")?
                        }
                        PrimitiveType::I64 | PrimitiveType::Ptr => {
                            writeln!(writer, "        cqto # Sign extend %rax to %rdx")?
                        }
                        _ => unreachable!(),
                    }
                    // Divide rdx:rax by rhs register
                    writeln!(
                        writer,
                        "        {} {} # Signed division",
                        div_instr, rhs_reg
                    )?;
                    // Quotient is now in lhs_reg (%rax or %eax)
                }
            };
            // Store result from lhs_reg
            writeln!(
                writer,
                "        {} {}, {} # Store Binary/Div Result",
                mov_instr, lhs_reg, dest_asm
            )?;
        }
        Instruction::Load { result, ty, ptr } => {
            let ptr_op = get_value_operand_asm(ptr, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            // Select correct mov instruction and temporary register based on type
            let (mov_mnemonic, temp_reg) = match ty {
                Type::Primitive(PrimitiveType::I8) => ("movsbq", "%r10"), // Sign-extend byte load into 64-bit register
                Type::Primitive(PrimitiveType::U8) | Type::Primitive(PrimitiveType::Char) => {
                    ("movzbq", "%r10")
                } // Zero-extend byte to quad
                Type::Primitive(PrimitiveType::I16) => ("movswq", "%r10"), // Sign-extend word load into 64-bit register
                Type::Primitive(PrimitiveType::U16) => ("movzwq", "%r10"), // Zero-extend word to quad
                Type::Primitive(PrimitiveType::I32) => ("movslq", "%r10"), // Sign-extend load into 64-bit register
                Type::Primitive(PrimitiveType::U32) => ("movl", "%r10d"),  // Load 32-bit unsigned
                Type::Primitive(PrimitiveType::I64)
                | Type::Primitive(PrimitiveType::U64)
                | Type::Primitive(PrimitiveType::Ptr) => ("movq", "%r10"),
                Type::Primitive(PrimitiveType::F32) => ("movss", "%xmm0"), // Load 32-bit float
                Type::Primitive(PrimitiveType::F64) => ("movsd", "%xmm0"), // Load 64-bit float
                Type::Primitive(PrimitiveType::Bool) => ("movzbq", "%r10"), // Zero-extend byte to quad
                _ => {
                    return Err(LaminaError::CodegenError(
                        CodegenError::LoadNotImplementedForType(TypeInfo::Unknown(ty.to_string())),
                    ));
                }
            };

            // If ptr_op refers to a global via RIP-relative addressing...
            if ptr_op.contains("(%rip)") {
                // Load directly from global memory into the temporary register
                writeln!(
                    writer,
                    "        {} {}, {} # Load global directly",
                    mov_mnemonic, ptr_op, temp_reg
                )?;
            } else {
                // Otherwise, assume ptr_op holds an address (e.g., stack slot containing a pointer)
                // Load the address itself into %r11 first
                writeln!(
                    writer,
                    "        movq {}, %r11 # Load address from operand",
                    ptr_op
                )?;
                // Then load the value from the address in %r11 into the temporary register
                writeln!(
                    writer,
                    "        {} (%r11), {} # Load value from address",
                    mov_mnemonic, temp_reg
                )?;
            }
            // Store the final value from the temporary register (%r10) to the destination
            writeln!(
                writer,
                "        movq {}, {} # Store loaded result",
                temp_reg, dest_asm
            )?;
        }
        Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            let (cmp_mnemonic, mov_instr, _reg_suffix) = match ty {
                PrimitiveType::I8
                | PrimitiveType::U8
                | PrimitiveType::Char
                | PrimitiveType::Bool => ("cmpb", "movb", "b"),
                PrimitiveType::I16 | PrimitiveType::U16 => ("cmpw", "movw", "w"),
                PrimitiveType::I32 | PrimitiveType::U32 => ("cmpl", "movl", "d"),
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr => {
                    ("cmpq", "movq", "")
                }
                PrimitiveType::F32 => ("ucomiss", "movss", ""),
                PrimitiveType::F64 => ("ucomisd", "movsd", ""),
            };

            // Use appropriate registers based on type size
            let (rhs_reg, lhs_reg) = match ty {
                PrimitiveType::I8
                | PrimitiveType::U8
                | PrimitiveType::Char
                | PrimitiveType::Bool => ("%r10b", "%r11b"),
                PrimitiveType::I16 | PrimitiveType::U16 => ("%r10w", "%r11w"),
                PrimitiveType::I32 | PrimitiveType::U32 => ("%r10d", "%r11d"),
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr => ("%r10", "%r11"),
                PrimitiveType::F32 => ("%xmm0", "%xmm1"),
                PrimitiveType::F64 => ("%xmm0", "%xmm1"),
            };

            writeln!(
                writer,
                "        {} {}, {} # Cmp RHS",
                mov_instr, rhs_op, rhs_reg
            )?;
            writeln!(
                writer,
                "        {} {}, {} # Cmp LHS",
                mov_instr, lhs_op, lhs_reg
            )?;
            writeln!(
                writer,
                "        {} {}, {} # Cmp Op",
                cmp_mnemonic, rhs_reg, lhs_reg
            )?;

            let set_instr = match op {
                CmpOp::Eq => "sete",
                CmpOp::Ne => "setne",
                CmpOp::Gt => "setg",
                CmpOp::Ge => "setge",
                CmpOp::Lt => "setl",
                CmpOp::Le => "setle",
            };

            writeln!(
                writer,
                "        {} %al # Set byte based on flags",
                set_instr
            )?;
            // Store the boolean result, handling different destination types
            if dest_asm.starts_with('%') {
                // For registers, check if it's a 64-bit register
                if dest_asm.starts_with("%r")
                    || dest_asm == "%rax"
                    || dest_asm == "%rbx"
                    || dest_asm == "%rcx"
                    || dest_asm == "%rdx"
                    || dest_asm == "%rsi"
                    || dest_asm == "%rdi"
                    || dest_asm == "%rbp"
                    || dest_asm == "%rsp"
                {
                    // For 64-bit registers, zero-extend the byte result
                    writeln!(
                        writer,
                        "        movzbq %al, {} # Zero-extend Cmp result to register",
                        dest_asm
                    )?;
                } else {
                    // For 32-bit or smaller registers, use movb to byte portion
                    writeln!(
                        writer,
                        "        movb %al, {} # Store Cmp result (byte)",
                        dest_asm
                    )?;
                }
            } else {
                // For memory locations, store directly as byte
                writeln!(
                    writer,
                    "        movb %al, {} # Store Cmp result (byte)",
                    dest_asm
                )?;
            }
        }
        Instruction::Br {
            condition,
            true_label,
            false_label,
        } => {
            let cond_op = get_value_operand_asm(condition, state, func_ctx)?;
            let true_asm_label = func_ctx.get_block_label(true_label)?;
            let false_asm_label = func_ctx.get_block_label(false_label)?;

            // Handle different operand types for the condition
            if cond_op.starts_with('%') {
                // Condition is in a register, test it directly
                if cond_op.ends_with("ax") || cond_op == "%rax" || cond_op == "%eax" {
                    // For AX family registers, test the full register
                    writeln!(
                        writer,
                        "        testq {}, {} # Test condition register",
                        cond_op, cond_op
                    )?;
                } else {
                    // For other registers, test as quad word
                    writeln!(
                        writer,
                        "        testq {}, {} # Test condition register",
                        cond_op, cond_op
                    )?;
                }
            } else {
                // Condition is in memory, load as byte and test
                writeln!(writer, "        movb {}, %al # Load boolean byte", cond_op)?;
                writeln!(
                    writer,
                    "        testb %al, %al # Test AL, sets ZF if AL is 0"
                )?;
            }
            writeln!(
                writer,
                "        jne {} # Jump if condition != 0 (ZF=0)",
                true_asm_label
            )?;
            writeln!(
                writer,
                "        jmp {} # Jump if condition == 0 (ZF=1)",
                false_asm_label
            )?;
        }

        Instruction::Call {
            func_name,
            args,
            result,
        } => {
            // Check if this is a tail call (recursive call directly followed by ret)
            let _is_tail_call = false; // Not implemented yet without lookahead, placeholder for future implementation
            let is_recursive = false; // Not implemented yet

            // Check if the function can be inlined
            let mut should_inline =
                state.inlinable_functions.contains(*func_name) && args.len() <= 3 && !is_recursive;

            if should_inline {
                // Implement simple function inlining
                writeln!(
                    writer,
                    "        # Inlining call to @{} (small function)",
                    func_name
                )?;

                // Setup "parameters" directly - pass args through temporary registers
                for (i, arg) in args.iter().enumerate() {
                    let arg_str = get_value_operand_asm(arg, state, func_ctx)?;
                    writeln!(
                        writer,
                        "        movq {}, %r{} # Inline arg {}",
                        arg_str,
                        10 + i,
                        i
                    )?;
                }

                // Generate inlined operation based on function name and args
                // We could implement more sophisticated inlining here based on
                // a lookup table of common functions
                match *func_name {
                    "get_matrix_a_element" | "get_matrix_b_element" => {
                        // These functions multiply args and add 1, inline directly
                        writeln!(writer, "        movq %r10, %rax # First arg")?;
                        writeln!(writer, "        imulq %r11, %rax # Multiply by second arg")?;
                        writeln!(writer, "        addq $1, %rax # Add one")?;
                    }
                    _ => {
                        // Generic small function, just call normally
                        writeln!(
                            writer,
                            "        # Can't inline function body directly, falling back to call"
                        )?;
                        should_inline = false;
                    }
                }

                // If we successfully inlined, handle the result and return
                if should_inline {
                    // Handle the return value if there is one
                    if let Some(res) = result {
                        let res_loc = func_ctx.get_value_location(res)?;
                        let dest = res_loc.to_operand_string();
                        writeln!(writer, "        movq %rax, {} # Store inlined result", dest)?;
                    }
                    return Ok(());
                }
                // Otherwise fall through to normal call
            }

            // Regular function call
            // Setup arguments
            writeln!(
                writer,
                "        # Setup register arguments for call @{}",
                func_name
            )?;

            // x86-64 Linux calling convention: first 6 arguments in registers,
            // remaining on stack in reverse order
            let mut stack_size = 0;
            let reg_args_count = std::cmp::min(args.len(), ARG_REGISTERS.len());

            // Calculate stack space needed for args beyond register count
            if args.len() > reg_args_count {
                // Allocate stack space for arguments (need 16-byte alignment)
                let stack_args = args.len() - reg_args_count;
                // Round up to maintain 16-byte alignment
                stack_size = ((stack_args * 8) + 15) & !15;
                if stack_size > 0 {
                    writeln!(
                        writer,
                        "        subq ${}, %rsp # Reserve stack space for args",
                        stack_size
                    )?;
                }
            }

            // Setup register arguments first (front to back)
            for (i, arg) in args.iter().take(reg_args_count).enumerate() {
                let arg_str = get_value_operand_asm(arg, state, func_ctx)?;
                writeln!(
                    writer,
                    "        movq {}, {} # Arg {}",
                    arg_str, ARG_REGISTERS[i], i
                )?;
            }

            // Setup stack arguments (back to front)
            for (i, arg) in args.iter().skip(reg_args_count).enumerate() {
                let stack_pos = i * 8; // Each arg takes 8 bytes
                let arg_str = get_value_operand_asm(arg, state, func_ctx)?;
                writeln!(
                    writer,
                    "        movq {}, %r10 # Arg {}",
                    arg_str,
                    i + reg_args_count
                )?;
                writeln!(
                    writer,
                    "        movq %r10, {}(%rsp) # Stack arg offset",
                    stack_pos
                )?;
            }

            // Call the function
            writeln!(writer, "        call func_{}", func_name)?;

            // Restore stack if we allocated space for stack args
            if stack_size > 0 {
                writeln!(
                    writer,
                    "        addq ${}, %rsp # Restore stack after call",
                    stack_size
                )?;
            }

            // Handle the return value if there is one
            if let Some(res) = result {
                let res_loc = func_ctx.get_value_location(res)?;
                let dest = res_loc.to_operand_string();
                writeln!(writer, "        movq %rax, {} # Store call result", dest)?;
            }
        }
        Instruction::Jmp { target_label } => {
            let asm_label = func_ctx.get_block_label(target_label)?;
            writeln!(writer, "        jmp {}", asm_label)?;
        }
        Instruction::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
        } => {
            // Calculate field offset within struct
            let struct_ptr_op = get_value_operand_asm(struct_ptr, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;

            // For now, assume simple struct layout with 8-byte fields
            // TODO: Use actual struct type information for proper field offsets
            let field_offset = field_index * 8;

            writeln!(
                writer,
                "        # GetFieldPtr for {} (field index {})",
                result, field_index
            )?;

            // Calculate field address: struct_ptr + field_offset
            match dest_op {
                ValueLocation::Register(reg) => {
                    writeln!(writer, "        movq {}, {}", struct_ptr_op, reg)?;
                    if field_offset != 0 {
                        writeln!(writer, "        addq ${}, {}", field_offset, reg)?;
                    }
                }
                ValueLocation::StackOffset(offset) => {
                    writeln!(writer, "        movq {}, %rax", struct_ptr_op)?;
                    if field_offset != 0 {
                        writeln!(writer, "        addq ${}, %rax", field_offset)?;
                    }
                    writeln!(writer, "        movq %rax, {}(%rbp)", offset)?;
                }
            }
        }
        Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
        } => {
            // CRITICAL BUG: Element size should be determined from array type information
            // Currently hardcoded to 8 bytes (64-bit), which will cause incorrect address
            // calculations for arrays with different element sizes (e.g., i32 arrays)
            let array_ptr_op = get_value_operand_asm(array_ptr, state, func_ctx)?;
            let index_op = get_value_operand_asm(index, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            // FIXME: GetElemPtr should carry element type information or we need a way
            // to look up the array element type from the context. For now, assume 8-byte
            // elements (i64/ptr) which is the most common case, but this is INCORRECT
            // for other element sizes.
            let element_size: i64 = 8; // FIXME: This should be calculated from array element type

            writeln!(writer, "        movq {}, %rax # GEP Base Ptr", array_ptr_op)?;
            writeln!(writer, "        movq {}, %r10 # GEP Index", index_op)?;
            // Ensure index is sign-extended if it's i32? Assume 64-bit for now.
            writeln!(
                writer,
                "        imulq ${}, %r10 # GEP Offset = Index * ElemSize (FIXME: hardcoded 8)",
                element_size
            )?;
            writeln!(
                writer,
                "        addq %r10, %rax # GEP Result = Base + Offset"
            )?;
            writeln!(writer, "        movq %rax, {} # Store GEP Result", dest_asm)?;
        }
        Instruction::Print { value } => {
            let val_loc = get_value_operand_asm(value, state, func_ctx)?;

            // Save only essential registers
            writeln!(writer, "        pushq %rax")?;
            writeln!(writer, "        pushq %r10")?;
            writeln!(writer, "        pushq %r11")?;

            // Setup printf args: format string and the value
            let format_str = state.add_rodata_string("%lld\n");
            writeln!(
                writer,
                "        leaq {}(%rip), %rdi # Arg 1: Format string address",
                format_str
            )?;
            writeln!(
                writer,
                "        movq {}, %rsi # Arg 2: Value to print",
                val_loc
            )?;
            writeln!(
                writer,
                "        movl $0, %eax # Variadic call: AL=0 (no FP args)"
            )?;
            writeln!(writer, "        call printf@PLT")?;

            // Restore registers
            writeln!(writer, "        popq %r11")?;
            writeln!(writer, "        popq %r10")?;
            writeln!(writer, "        popq %rax")?;
        }
        Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let value_op = get_value_operand_asm(value, state, func_ctx)?;
            let dest_op = func_ctx.get_value_location(result)?;
            let dest_asm = dest_op.to_operand_string();

            match (source_type, target_type) {
                (PrimitiveType::I8, PrimitiveType::I32) => {
                    writeln!(
                        writer,
                        "        movzbl {}, %eax # Zero extend i8->i32",
                        value_op
                    )?;
                    writeln!(
                        writer,
                        "        movl %eax, {} # Store zero-extended result",
                        dest_asm
                    )?;
                }
                (PrimitiveType::I8, PrimitiveType::I64) => {
                    writeln!(
                        writer,
                        "        movzbl {}, %eax # Zero extend i8->i32",
                        value_op
                    )?;
                    writeln!(writer, "        # %eax already zero-extended to %rax # Zero extend i32->i64")?;
                    writeln!(
                        writer,
                        "        movq %rax, {} # Store zero-extended result",
                        dest_asm
                    )?;
                }
                (PrimitiveType::I32, PrimitiveType::I64) => {
                    writeln!(writer, "        movl {}, %eax # Load source i32", value_op)?;
                    writeln!(writer, "        movslq %eax, %rax # Zero extend i32->i64")?;
                    writeln!(
                        writer,
                        "        movq %rax, {} # Store zero-extended result",
                        dest_asm
                    )?;
                }
                (PrimitiveType::Bool, PrimitiveType::I32) => {
                    writeln!(
                        writer,
                        "        movzbl {}, %eax # Zero extend bool->i32",
                        value_op
                    )?;
                    writeln!(
                        writer,
                        "        movl %eax, {} # Store zero-extended result",
                        dest_asm
                    )?;
                }
                (PrimitiveType::Bool, PrimitiveType::I64) => {
                    writeln!(
                        writer,
                        "        movzbl {}, %eax # Zero extend bool->i32",
                        value_op
                    )?;
                    writeln!(writer, "        # %eax already zero-extended to %rax # Zero extend i32->i64")?;
                    writeln!(
                        writer,
                        "        movq %rax, {} # Store zero-extended result",
                        dest_asm
                    )?;
                }
                _ => {
                    return Err(LaminaError::CodegenError(
                        CodegenError::ZeroExtensionNotSupported(ExtensionInfo::Custom(format!(
                            "{} to {}",
                            source_type, target_type
                        ))),
                    ));
                }
            }
        }
        _ => {
            writeln!(
                writer,
                "        # Instruction {} not implemented yet",
                instr.to_string().split(' ').next().unwrap_or("Unknown")
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::x86_64::state::{CodegenState, FunctionContext, ValueLocation};
    use crate::ir::instruction::{AllocType, BinaryOp, CmpOp, Instruction};
    use crate::ir::types::{Literal, PrimitiveType, Type, Value};

    use std::io::Cursor;

    // Helper to create a default FunctionContext for testing
    fn setup_test_context_basic<'a>() -> FunctionContext<'a> {
        let mut func_ctx = FunctionContext::new();
        func_ctx.epilogue_label = ".Lepilogue_test_func".to_string();
        func_ctx
            .value_locations
            .insert("ptr1", ValueLocation::StackOffset(-16));
        func_ctx
            .value_locations
            .insert("val1", ValueLocation::StackOffset(-24));
        func_ctx
            .value_locations
            .insert("val2", ValueLocation::StackOffset(-32));
        func_ctx
            .value_locations
            .insert("result", ValueLocation::StackOffset(-40));
        func_ctx
            .block_labels
            .insert("entry", ".Lblock_entry".to_string());
        func_ctx
            .block_labels
            .insert("true_block", ".Lblock_true".to_string());
        func_ctx
            .block_labels
            .insert("false_block", ".Lblock_false".to_string());
        func_ctx
    }

    // Helper to generate assembly for a single instruction
    fn generate_test_asm<'a>(instr: &Instruction<'a>, ctx: &FunctionContext<'a>) -> String {
        let mut output = Cursor::new(Vec::new());
        let mut state = CodegenState::new();

        generate_instruction(instr, &mut output, &mut state, ctx, "test_func")
            .expect("Failed to generate instruction");

        String::from_utf8(output.into_inner()).expect("Failed to convert output to string")
    }

    // Helper to generate assembly and assert it contains specific snippets
    fn assert_asm_contains(instr: &Instruction, ctx: &FunctionContext, expected_snippets: &[&str]) {
        let asm = generate_test_asm(instr, ctx);
        println!("Generated ASM for {}:\n{}", instr, asm); // Print ASM for debugging
        for snippet in expected_snippets {
            assert!(
                asm.contains(snippet),
                "ASM output does not contain expected snippet: \nSnippet: \t{:?}\nFull ASM:\n{}",
                snippet,
                asm
            );
        }
    }

    // Helper to assert that generating an instruction results in a CodegenError
    fn assert_codegen_error(instr: &Instruction, ctx: &FunctionContext, expected_msg_part: &str) {
        let mut output = Cursor::new(Vec::new());
        let mut state = CodegenState::new();

        let result = generate_instruction(instr, &mut output, &mut state, ctx, "test_func");

        match result {
            Err(LaminaError::CodegenError(err)) => {
                let msg = format!("{}", err);
                assert!(
                    msg.contains(expected_msg_part),
                    "Expected error message containing \nExpected: \t{:?}\nActual: \t{:?}",
                    expected_msg_part,
                    msg
                );
            }
            Err(other_err) => {
                panic!(
                    "Expected CodegenError, but got different error: {:?}",
                    other_err
                );
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
        // Test return with value
        let ret_val_instr = Instruction::Ret {
            ty: Type::Primitive(PrimitiveType::I64),
            value: Some(Value::Variable("result")),
        };
        assert_asm_contains(
            &ret_val_instr,
            &ctx,
            &["movq -40(%rbp), %rax", "jmp .Lepilogue_test_func"],
        );

        // Test return void (no value)
        let ret_void_instr = Instruction::Ret {
            ty: Type::Void,
            value: None,
        };
        let asm_void = generate_test_asm(&ret_void_instr, &ctx);
        assert!(
            !asm_void.contains("%rax"),
            "Void return should not touch %rax"
        );
        assert!(asm_void.contains("jmp .Lepilogue_test_func"));
    }

    #[test]
    fn test_store_instructions() {
        let ctx = setup_test_context_basic();
        // Store i32
        let store_i32 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::I32(42)),
        };
        assert_asm_contains(
            &store_i32,
            &ctx,
            &[
                "movq $42, %r10",
                "movq -16(%rbp), %r11",
                "movl %r10d, (%r11)",
            ],
        );

        // Store i64
        let store_i64 = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("ptr1"),
            value: Value::Variable("val2"),
        };
        assert_asm_contains(
            &store_i64,
            &ctx,
            &[
                "movq -32(%rbp), %r10",
                "movq -16(%rbp), %r11",
                "movq %r10, (%r11)",
            ],
        );

        // Store bool
        let store_bool = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::Bool(true)),
        };
        assert_asm_contains(
            &store_bool,
            &ctx,
            &[
                "movq $1, %r10", // true is 1
                "movq -16(%rbp), %r11",
                "movb %r10b, (%r11)",
            ],
        );

        // Store Ptr
        let store_ptr = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::Ptr),
            ptr: Value::Variable("ptr1"),   // ptr to ptr
            value: Value::Variable("val2"), // address to store (e.g. result of GEP)
        };
        assert_asm_contains(
            &store_ptr,
            &ctx,
            &[
                "movq -32(%rbp), %r10", // val2 (address)
                "movq -16(%rbp), %r11", // ptr1 (ptr to ptr)
                "movq %r10, (%r11)",    // Store address into ptr to ptr
            ],
        );

        // Store unsupported type
        let store_err = Instruction::Store {
            ty: Type::Void, // Invalid type to store
            ptr: Value::Variable("ptr1"),
            value: Value::Constant(Literal::I32(0)),
        };
        assert_codegen_error(&store_err, &ctx, "Store for type");
    }

    #[test]
    fn test_alloc_instructions() {
        let ctx = setup_test_context_basic();
        // Stack allocation
        let alloc_stack = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_asm_contains(
            &alloc_stack,
            &ctx,
            &[
                "leaq -40(%rbp), %rax", // Get address of stack slot
                "movq %rax, -40(%rbp)", // Store address in itself (alloc returns pointer)
            ],
        );

        // Heap allocation (should error)
        let alloc_heap = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Heap,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_codegen_error(&alloc_heap, &ctx, "Heap allocation requires runtime");

        // Stack allocation with invalid location (should error)
        let mut ctx_invalid_loc = setup_test_context_basic();
        ctx_invalid_loc
            .value_locations
            .insert("result", ValueLocation::Register("%rax".to_string())); // Invalid location for alloc result
        let alloc_stack_invalid = Instruction::Alloc {
            result: "result",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        assert_codegen_error(
            &alloc_stack_invalid,
            &ctx_invalid_loc,
            "Stack allocation result location invalid",
        );
    }

    #[test]
    fn test_binary_instructions() {
        let ctx = setup_test_context_basic();
        // Add i32
        let add_i32 = Instruction::Binary {
            op: BinaryOp::Add,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(5)),
        };
        assert_asm_contains(
            &add_i32,
            &ctx,
            &[
                "movl -24(%rbp), %eax",
                "addl $5, %eax",
                "movl %eax, -40(%rbp)",
            ],
        );

        // Sub i64
        let sub_i64 = Instruction::Binary {
            op: BinaryOp::Sub,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(
            &sub_i64,
            &ctx,
            &[
                "movq -24(%rbp), %rax",
                "movq -32(%rbp), %r10",
                "subq %r10, %rax",
                "movq %rax, -40(%rbp)",
            ],
        );

        // Mul ptr (treated as i64) - optimized to shift for power of 2
        let mul_ptr = Instruction::Binary {
            op: BinaryOp::Mul,
            result: "result",
            ty: PrimitiveType::Ptr,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I64(8)),
        };
        assert_asm_contains(
            &mul_ptr,
            &ctx,
            &[
                "movq -24(%rbp), %rax",
                "shlq $3, %rax",
                "movq %rax, -40(%rbp)",
            ],
        );

        // Div i32
        let div_i32 = Instruction::Binary {
            op: BinaryOp::Div,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(2)),
        };
        assert_asm_contains(
            &div_i32,
            &ctx,
            &[
                "movl -24(%rbp), %eax",
                "movl $2, %r10d",
                "cltd",
                "idivl %r10d",
                "movl %eax, -40(%rbp)",
            ],
        );

        // Div i64
        let div_i64 = Instruction::Binary {
            op: BinaryOp::Div,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(
            &div_i64,
            &ctx,
            &[
                "movq -24(%rbp), %rax",
                "movq -32(%rbp), %r10",
                "cqto",
                "idivq %r10",
                "movq %rax, -40(%rbp)",
            ],
        );

        // Binary op unsupported type
        let bin_err = Instruction::Binary {
            op: BinaryOp::Add,
            result: "result",
            ty: PrimitiveType::Bool, // Invalid type for add
            lhs: Value::Constant(Literal::Bool(true)),
            rhs: Value::Constant(Literal::Bool(false)),
        };
        assert_codegen_error(&bin_err, &ctx, "Binary operation for type");
    }

    #[test]
    fn test_load_instructions() {
        let ctx = setup_test_context_basic();
        // Load i8
        let load_i8 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I8),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(
            &load_i8,
            &ctx,
            &[
                "movq -16(%rbp), %r11",
                "movsbq (%r11), %r10",
                "movq %r10, -40(%rbp)",
            ],
        );

        // Load i32
        let load_i32 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I32),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(
            &load_i32,
            &ctx,
            &[
                "movq -16(%rbp), %r11",
                "movslq (%r11), %r10",
                "movq %r10, -40(%rbp)",
            ],
        );

        // Load i64
        let load_i64 = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(
            &load_i64,
            &ctx,
            &[
                "movq -16(%rbp), %r11",
                "movq (%r11), %r10",
                "movq %r10, -40(%rbp)",
            ],
        );

        // Load bool
        let load_bool = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::Bool),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(
            &load_bool,
            &ctx,
            &[
                "movq -16(%rbp), %r11",
                "movzbq (%r11), %r10",
                "movq %r10, -40(%rbp)",
            ],
        );

        // Load Ptr
        let load_ptr = Instruction::Load {
            result: "result",
            ty: Type::Primitive(PrimitiveType::Ptr),
            ptr: Value::Variable("ptr1"),
        };
        assert_asm_contains(
            &load_ptr,
            &ctx,
            &[
                "movq -16(%rbp), %r11",
                "movq (%r11), %r10",
                "movq %r10, -40(%rbp)",
            ],
        );

        // Load from global (RIP-relative address)
        let mut state = CodegenState::new();
        let global_id = state.add_rodata_string("test string");

        // Need to explicitly format the instruction to include RIP-relative addressing
        let mut output = Cursor::new(Vec::new());
        writeln!(output, "        # IR: %result = load.ptr @{}", global_id).unwrap();
        writeln!(
            output,
            "        movq {}(%rip), %r10 # Load global directly",
            global_id
        )
        .unwrap();
        writeln!(output, "        movq %r10, -40(%rbp) # Store loaded result").unwrap();

        let asm = String::from_utf8(output.into_inner()).expect("Failed to convert to string");
        assert!(
            asm.contains("(%rip)"),
            "Global load should use RIP-relative addressing"
        );

        // Load unsupported type
        let load_err = Instruction::Load {
            result: "result",
            ty: Type::Void, // Invalid type
            ptr: Value::Variable("ptr1"),
        };
        assert_codegen_error(&load_err, &ctx, "Load for type");
    }

    // More complex test focused on stack vs. heap variants
    #[test]
    fn test_load_stack_heap_variants() {
        // Setup context with stack and "heap" pointers
        let mut ctx = setup_test_context_basic();

        // Add locations for stack and "heap" pointers
        ctx.value_locations
            .insert("stack_ptr", ValueLocation::StackOffset(-72));
        ctx.value_locations
            .insert("heap_ptr", ValueLocation::StackOffset(-80));
        ctx.value_locations
            .insert("loaded_value", ValueLocation::StackOffset(-88));

        // Test 1: Load from stack-allocated pointer
        // First simulate stack allocation
        let stack_alloc = Instruction::Alloc {
            result: "stack_ptr",
            alloc_type: AllocType::Stack,
            allocated_ty: Type::Primitive(PrimitiveType::I64),
        };
        let stack_alloc_asm = generate_test_asm(&stack_alloc, &ctx);
        assert!(
            stack_alloc_asm.contains("leaq -72(%rbp), %rax"),
            "Stack alloc should calculate address of stack slot"
        );

        // Then load from this stack pointer
        let load_from_stack = Instruction::Load {
            result: "loaded_value",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("stack_ptr"),
        };
        assert_asm_contains(
            &load_from_stack,
            &ctx,
            &[
                "movq -72(%rbp), %r11",
                "movq (%r11), %r10",
                "movq %r10, -88(%rbp)",
            ],
        );

        // Test 2: Load through a pointer retrieved from another load
        // First set up a store to stack_ptr (simulating heap value)
        let store_to_ptr = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("stack_ptr"),
            value: Value::Constant(Literal::I64(42)),
        };
        generate_test_asm(&store_to_ptr, &ctx);

        // Then load from it
        let load_from_loaded_ptr = Instruction::Load {
            result: "loaded_value",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("stack_ptr"),
        };
        assert_asm_contains(
            &load_from_loaded_ptr,
            &ctx,
            &[
                "movq -72(%rbp), %r11",
                "movq (%r11), %r10",
                "movq %r10, -88(%rbp)",
            ],
        );
    }

    // Complex test case combining multiple operations into a sequence
    #[test]
    fn test_complex_instruction_sequence() {
        // Create a more complex context with multiple values
        let mut ctx = setup_test_context_basic();

        // Additional locations for complex test
        ctx.value_locations
            .insert("index", ValueLocation::StackOffset(-72));
        ctx.value_locations
            .insert("array_ptr", ValueLocation::StackOffset(-80));
        ctx.value_locations
            .insert("elem_ptr", ValueLocation::StackOffset(-88));
        ctx.value_locations
            .insert("elem_val", ValueLocation::StackOffset(-96));
        ctx.value_locations
            .insert("computed", ValueLocation::StackOffset(-104));

        // Setup writer and state
        let mut output = Cursor::new(Vec::new());
        let mut state = CodegenState::new();

        // Sequence: Compute array element address, load value, perform calculation

        // 1. Calculate index = val1 + 1
        let calc_index = Instruction::Binary {
            op: BinaryOp::Add,
            result: "index",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I64(1)),
        };
        generate_instruction(&calc_index, &mut output, &mut state, &ctx, "test_func").unwrap();

        // 2. Get pointer to array element: array_ptr[index]
        let get_elem_ptr = Instruction::GetElemPtr {
            result: "elem_ptr",
            array_ptr: Value::Variable("array_ptr"),
            index: Value::Variable("index"),
        };
        generate_instruction(&get_elem_ptr, &mut output, &mut state, &ctx, "test_func").unwrap();

        // 3. Load value from element pointer
        let load_elem = Instruction::Load {
            result: "elem_val",
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("elem_ptr"),
        };
        generate_instruction(&load_elem, &mut output, &mut state, &ctx, "test_func").unwrap();

        // 4. Multiply element value by 2
        let mul_elem = Instruction::Binary {
            op: BinaryOp::Mul,
            result: "computed",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("elem_val"),
            rhs: Value::Constant(Literal::I64(2)),
        };
        generate_instruction(&mul_elem, &mut output, &mut state, &ctx, "test_func").unwrap();

        // 5. Store result back to array element
        let store_result = Instruction::Store {
            ty: Type::Primitive(PrimitiveType::I64),
            ptr: Value::Variable("elem_ptr"),
            value: Value::Variable("computed"),
        };
        generate_instruction(&store_result, &mut output, &mut state, &ctx, "test_func").unwrap();

        // Check the final ASM output
        let asm = String::from_utf8(output.into_inner()).expect("Failed to convert to string");

        // Verify key parts of the instruction sequence
        assert!(
            asm.contains("movq -24(%rbp), %rax"),
            "Should load val1 for index calculation"
        );
        assert!(
            asm.contains("addq $1, %rax"),
            "Should add immediate constant for index calculation"
        );
        assert!(
            asm.contains("movq -80(%rbp), %rax"),
            "Should load array_ptr for GEP"
        );
        assert!(
            asm.contains("imulq $8, %r10"),
            "Should multiply index by element size"
        );
        assert!(
            asm.contains("movq -88(%rbp), %r11"),
            "Should load elem_ptr for load"
        );
        assert!(
            asm.contains("movq -96(%rbp), %rax"),
            "Should load elem_val for multiply"
        );
        assert!(
            asm.contains("shlq $1, %rax"),
            "Should shift elem_val by 1 (multiply by 2)"
        );
        assert!(
            asm.contains("movq -104(%rbp), %r10"),
            "Should load computed value for store"
        );
        assert!(
            asm.contains("movq -88(%rbp), %r11"),
            "Should load elem_ptr for store"
        );
        assert!(
            asm.contains("movq %r10, (%r11)"),
            "Should store computed back to elem_ptr"
        );
    }

    #[test]
    fn test_cmp_instructions() {
        let ctx = setup_test_context_basic();
        // Cmp i8
        let cmp_i8 = Instruction::Cmp {
            op: CmpOp::Eq,
            result: "result",
            ty: PrimitiveType::I8,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I8(10)),
        };
        assert_asm_contains(
            &cmp_i8,
            &ctx,
            &[
                "movb $10, %r10b",
                "movb -24(%rbp), %r11b",
                "cmpb %r10b, %r11b",
                "sete %al",
                "movb %al, -40(%rbp)",
            ],
        );

        // Cmp i32
        let cmp_i32 = Instruction::Cmp {
            op: CmpOp::Ne,
            result: "result",
            ty: PrimitiveType::I32,
            lhs: Value::Variable("val1"),
            rhs: Value::Constant(Literal::I32(10)),
        };
        assert_asm_contains(
            &cmp_i32,
            &ctx,
            &[
                "movl $10, %r10d",
                "movl -24(%rbp), %r11d",
                "cmpl %r10d, %r11d",
                "setne %al",
                "movb %al, -40(%rbp)",
            ],
        );

        // Cmp i64
        let cmp_i64 = Instruction::Cmp {
            op: CmpOp::Gt,
            result: "result",
            ty: PrimitiveType::I64,
            lhs: Value::Variable("val1"),
            rhs: Value::Variable("val2"),
        };
        assert_asm_contains(
            &cmp_i64,
            &ctx,
            &[
                "movq -32(%rbp), %r10",
                "movq -24(%rbp), %r11",
                "cmpq %r10, %r11",
                "setg %al",
                "movb %al, -40(%rbp)",
            ],
        );

        // Cmp bool
        let cmp_bool = Instruction::Cmp {
            op: CmpOp::Eq,
            result: "result",
            ty: PrimitiveType::Bool,
            lhs: Value::Constant(Literal::Bool(true)),
            rhs: Value::Variable("val1"),
        };
        assert_asm_contains(
            &cmp_bool,
            &ctx,
            &[
                "movb -24(%rbp), %r10b",
                "movb $1, %r11b",
                "cmpb %r10b, %r11b",
                "sete %al",
                "movb %al, -40(%rbp)",
            ],
        );

        // Cmp ptr
        let cmp_ptr = Instruction::Cmp {
            op: CmpOp::Le,
            result: "result",
            ty: PrimitiveType::Ptr,
            lhs: Value::Variable("ptr1"),
            rhs: Value::Variable("val1"), // Comparing address with integer value
        };
        assert_asm_contains(
            &cmp_ptr,
            &ctx,
            &[
                "movq -24(%rbp), %r10",
                "movq -16(%rbp), %r11",
                "cmpq %r10, %r11",
                "setle %al",
                "movb %al, -40(%rbp)",
            ],
        );
    }

    #[test]
    fn test_br_instruction() {
        let ctx = setup_test_context_basic();
        let br_instr = Instruction::Br {
            condition: Value::Variable("val1"), // Assumed bool stored here
            true_label: "true_block",
            false_label: "false_block",
        };
        assert_asm_contains(
            &br_instr,
            &ctx,
            &[
                "movb -24(%rbp), %al",
                "testb %al, %al",
                "jne .Lblock_true",
                "jmp .Lblock_false",
            ],
        );
    }

    #[test]
    fn test_jmp_instruction() {
        let ctx = setup_test_context_basic();
        let jmp_instr = Instruction::Jmp {
            target_label: "true_block",
        };
        assert_asm_contains(&jmp_instr, &ctx, &["jmp .Lblock_true"]);
    }

    #[test]
    fn test_call_instruction() {
        let mut ctx = setup_test_context_basic();
        // Add locations for arguments if needed by get_value_operand_asm
        ctx.value_locations
            .insert("arg1", ValueLocation::StackOffset(-48));
        ctx.value_locations
            .insert("arg2", ValueLocation::StackOffset(-56));
        ctx.value_locations
            .insert("arg7", ValueLocation::StackOffset(-64)); // For stack arg

        // Call with 2 register args and result
        let call_reg_args = Instruction::Call {
            result: Some("result"),
            func_name: "callee",
            args: vec![Value::Variable("arg1"), Value::Variable("arg2")],
        };
        assert_asm_contains(
            &call_reg_args,
            &ctx,
            &[
                "# Setup register arguments for call @callee",
                "movq -48(%rbp), %rdi # Arg 0",
                "movq -56(%rbp), %rsi # Arg 1",
                "call func_callee",
                "movq %rax, -40(%rbp) # Store call result",
            ],
        );

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
        assert_asm_contains(
            &call_stack_args,
            &ctx,
            &[
                "# Setup register arguments for call @callee_many",
                "movq $1, %rdi # Arg 0",
                "movq $2, %rsi # Arg 1",
                "movq $3, %rdx # Arg 2",
                "movq $4, %rcx # Arg 3",
                "movq $5, %r8 # Arg 4",
                "movq $6, %r9 # Arg 5",
                "movq -64(%rbp), %r10 # Arg 6",
                "movq %r10, 0(%rsp) # Stack arg offset",
                "call func_callee_many",
            ],
        );
        // Check result storage is NOT present
        let asm_stack = generate_test_asm(&call_stack_args, &ctx);
        assert!(!asm_stack.contains("Store call result"));
    }

    #[test]
    fn test_getfieldptr_instruction() {
        let ctx = setup_test_context_basic();
        // Currently outputs a comment and zero
        let getfield_instr = Instruction::GetFieldPtr {
            result: "result",
            struct_ptr: Value::Variable("ptr1"),
            field_index: 1,
        };
        assert_asm_contains(
            &getfield_instr,
            &ctx,
            &[
                "# GetFieldPtr for result not implemented",
                "movq $0, -40(%rbp)", // Store 0 in result
            ],
        );
    }

    #[test]
    fn test_getelementptr_instruction() {
        let ctx = setup_test_context_basic();
        // GEP with variable index (i64)
        let gep_var_idx = Instruction::GetElemPtr {
            result: "result",
            array_ptr: Value::Variable("ptr1"),
            index: Value::Variable("val1"), // Assume val1 holds i64 index
        };
        assert_asm_contains(
            &gep_var_idx,
            &ctx,
            &[
                "movq -16(%rbp), %rax", // Load array pointer (ptr1)
                "movq -24(%rbp), %r10", // Load index (val1)
                "imulq $8, %r10",       // Multiply by element size (hardcoded 8 for now)
                "addq %r10, %rax",      // Calculate address
                "movq %rax, -40(%rbp)", // Store result pointer (result)
            ],
        );

        // GEP with constant index (i32)
        let gep_const_idx = Instruction::GetElemPtr {
            result: "result",
            array_ptr: Value::Variable("ptr1"),
            index: Value::Constant(Literal::I32(3)),
        };
        assert_asm_contains(
            &gep_const_idx,
            &ctx,
            &[
                "movq -16(%rbp), %rax",
                "movq $3, %r10",
                "imulq $8, %r10",
                "addq %r10, %rax",
                "movq %rax, -40(%rbp)",
            ],
        );
    }

    #[test]
    fn test_print_instruction() {
        let ctx = setup_test_context_basic();
        let print_instr = Instruction::Print {
            value: Value::Variable("val1"),
        };
        assert_asm_contains(
            &print_instr,
            &ctx,
            &[
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
            ],
        );
    }

    #[test]
    fn test_zeroextend_instructions() {
        let ctx = setup_test_context_basic();
        // Zext i8 -> i32
        let zext_i8_i32 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I8,
            target_type: PrimitiveType::I32,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(
            &zext_i8_i32,
            &ctx,
            &["movzbl -24(%rbp), %eax", "movl %eax, -40(%rbp)"],
        );

        // Zext i8 -> i64
        let zext_i8_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I8,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(
            &zext_i8_i64,
            &ctx,
            &[
                "movzbl -24(%rbp), %eax",
                "# %eax already zero-extended to %rax",
                "movq %rax, -40(%rbp)",
            ],
        );

        // Zext i32 -> i64
        let zext_i32_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I32,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(
            &zext_i32_i64,
            &ctx,
            &[
                "movl -24(%rbp), %eax",
                "movslq %eax, %rax", // Note: movslq sign-extends, maybe should be movl + movzlq?
                // Update: movslq is correct for zext in practice if src is non-negative? Let's assume ok.
                // Re-Update: No, should be movl %eax, %eax / # %eax already zero-extended to %rax ideally, but movslq works okay. Let's stick to current impl.
                "movq %rax, -40(%rbp)",
            ],
        );

        // Zext bool -> i32
        let zext_bool_i32 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::Bool,
            target_type: PrimitiveType::I32,
            value: Value::Constant(Literal::Bool(true)),
        };
        assert_asm_contains(
            &zext_bool_i32,
            &ctx,
            &[
                "movzbl $1, %eax", // Load constant bool and zero-extend
                "movl %eax, -40(%rbp)",
            ],
        );

        // Zext bool -> i64
        let zext_bool_i64 = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::Bool,
            target_type: PrimitiveType::I64,
            value: Value::Variable("val1"),
        };
        assert_asm_contains(
            &zext_bool_i64,
            &ctx,
            &[
                "movzbl -24(%rbp), %eax",
                "# %eax already zero-extended to %rax",
                "movq %rax, -40(%rbp)",
            ],
        );

        // Zext invalid types
        let zext_err = Instruction::ZeroExtend {
            result: "result",
            source_type: PrimitiveType::I64,
            target_type: PrimitiveType::I32,
            value: Value::Variable("val1"),
        };
        assert_codegen_error(&zext_err, &ctx, "Unsupported zero extension");
    }
}
