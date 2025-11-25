use super::IsaWidth;
use super::state::{CodegenState, FunctionContext, RETURN_REGISTER};
use crate::codegen::{CodegenError, TypeInfo};
use crate::{BinaryOp, Identifier, Instruction, LaminaError, PrimitiveType, Type};
use std::io::Write;
use std::result::Result;

pub fn generate_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    _func_name: Identifier<'a>,
) -> Result<(), LaminaError> {
    writeln!(writer, "    # IR: {}", instr)?;

    match instr {
        Instruction::Alloc {
            result,
            alloc_type,
            allocated_ty: _,
        } => {
            // For stack allocations of aggregates, store the address of the slot into result
            match alloc_type {
                crate::AllocType::Stack => {
                    let dest = func_ctx.get_value_location(result)?.to_operand_string();
                    // Compute address of destination stack slot into t0 and store to dest
                    if dest.contains("(s0)") {
                        materialize_address(writer, &dest, "t0")?;
                        store_to_location(writer, "t0", &dest, state)?;
                    } else {
                        // Destination is a register
                        // Materialize its own stack slot address: find its location from context
                        let loc = func_ctx.get_value_location(result)?;
                        if let super::state::ValueLocation::StackOffset(off) = loc {
                            writeln!(writer, "    addi t0, s0, {}", off)?;
                            store_to_location(writer, "t0", &dest, state)?;
                        }
                    }
                }
                crate::AllocType::Heap => {
                    // Not implemented: requires runtime malloc; leave as no-op or error
                }
            }
        }
        Instruction::Ret { ty: _, value } => {
            if let Some(val) = value {
                let src = super::util::get_value_operand_asm(val, state, func_ctx)?;
                materialize_to_reg(writer, &src, RETURN_REGISTER, state)?;
            }
            writeln!(writer, "    j {}", func_ctx.epilogue_label)?;
        }

        Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let lhs_op = super::util::get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = super::util::get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();

            match ty {
                PrimitiveType::I32 | PrimitiveType::U32 => {
                    binary_i32(writer, op, &lhs_op, &rhs_op, &dest, state)?
                }
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr => {
                    binary_i64(writer, op, &lhs_op, &rhs_op, &dest, state)?
                }
                _ => {
                    return Err(LaminaError::CodegenError(
                        CodegenError::BinaryOpNotSupportedForType(TypeInfo::Primitive(*ty)),
                    ));
                }
            }
        }

        Instruction::Jmp { target_label } => {
            let label = func_ctx.get_block_label(target_label)?;
            writeln!(writer, "    j {}", label)?;
        }

        Instruction::Br {
            condition,
            true_label,
            false_label,
        } => {
            let cond = super::util::get_value_operand_asm(condition, state, func_ctx)?;
            materialize_to_reg(writer, &cond, "t0", state)?;
            let tlabel = func_ctx.get_block_label(true_label)?;
            let flabel = func_ctx.get_block_label(false_label)?;
            writeln!(writer, "    bnez t0, {}", tlabel)?;
            writeln!(writer, "    j {}", flabel)?;
        }

        Instruction::Cmp {
            op,
            result,
            ty: _,
            lhs,
            rhs,
        } => {
            // Very basic: compute lhs - rhs and set 1/0 in dest
            let lhs_op = super::util::get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = super::util::get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &lhs_op, "t0", state)?;
            materialize_to_reg(writer, &rhs_op, "t1", state)?;
            match op {
                crate::CmpOp::Eq => {
                    writeln!(writer, "    sub t2, t0, t1")?;
                    writeln!(writer, "    seqz t3, t2")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
                crate::CmpOp::Ne => {
                    writeln!(writer, "    sub t2, t0, t1")?;
                    writeln!(writer, "    snez t3, t2")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
                crate::CmpOp::Lt => {
                    writeln!(writer, "    slt t3, t0, t1")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
                crate::CmpOp::Le => {
                    writeln!(writer, "    sgt t2, t0, t1")?;
                    writeln!(writer, "    seqz t3, t2")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
                crate::CmpOp::Gt => {
                    writeln!(writer, "    sgt t3, t0, t1")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
                crate::CmpOp::Ge => {
                    writeln!(writer, "    slt t2, t0, t1")?;
                    writeln!(writer, "    seqz t3, t2")?;
                    store_to_location(writer, "t3", &dest, state)?;
                }
            }
        }

        Instruction::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        } => {
            // base + index * elem_size
            let base = super::util::get_value_operand_asm(array_ptr, state, func_ctx)?;
            let idx = super::util::get_value_operand_asm(index, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &base, "t0", state)?;
            materialize_to_reg(writer, &idx, "t1", state)?;
            let elem_size = match element_type {
                PrimitiveType::I8
                | PrimitiveType::U8
                | PrimitiveType::Bool
                | PrimitiveType::Char => 1,
                PrimitiveType::I16 | PrimitiveType::U16 => 2,
                PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => 4,
                PrimitiveType::I64
                | PrimitiveType::U64
                | PrimitiveType::F64
                | PrimitiveType::Ptr => 8,
            } as i64;
            if elem_size == 1 {
                // no scale
            } else if (elem_size & (elem_size - 1)) == 0 {
                // power of two -> shift
                let sh = (elem_size as u64).trailing_zeros();
                writeln!(writer, "    slli t1, t1, {}", sh)?;
            } else {
                writeln!(writer, "    li t2, {}", elem_size)?;
                writeln!(writer, "    mul t1, t1, t2")?;
            }
            writeln!(writer, "    add t0, t0, t1")?;
            store_to_location(writer, "t0", &dest, state)?;
        }

        Instruction::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
        } => {
            // Simplified: assume 8-byte fields
            let base = super::util::get_value_operand_asm(struct_ptr, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &base, "t0", state)?;
            let scale = if matches!(state.width(), IsaWidth::Rv32) {
                4
            } else {
                8
            };
            writeln!(writer, "    li t1, {}", field_index * scale)?;
            writeln!(writer, "    add t0, t0, t1")?;
            store_to_location(writer, "t0", &dest, state)?;
        }

        Instruction::PtrToInt {
            result, ptr_value, ..
        } => {
            let src = super::util::get_value_operand_asm(ptr_value, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &src, "t0", state)?;
            store_to_location(writer, "t0", &dest, state)?;
        }

        Instruction::IntToPtr {
            result, int_value, ..
        } => {
            let src = super::util::get_value_operand_asm(int_value, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &src, "t0", state)?;
            store_to_location(writer, "t0", &dest, state)?;
        }

        Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let src = super::util::get_value_operand_asm(value, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &src, "t0", state)?;
            let mask = match (source_type, target_type) {
                (
                    PrimitiveType::I8 | PrimitiveType::U8 | PrimitiveType::Bool,
                    PrimitiveType::I32
                    | PrimitiveType::U32
                    | PrimitiveType::I64
                    | PrimitiveType::U64,
                ) => Some(0xFFu64),
                (
                    PrimitiveType::I16 | PrimitiveType::U16,
                    PrimitiveType::I32
                    | PrimitiveType::U32
                    | PrimitiveType::I64
                    | PrimitiveType::U64,
                ) => Some(0xFFFFu64),
                (
                    PrimitiveType::I32 | PrimitiveType::U32,
                    PrimitiveType::I64 | PrimitiveType::U64,
                ) => None,
                _ => None,
            };
            if let Some(m) = mask {
                writeln!(writer, "    li t1, {}", m)?;
                writeln!(writer, "    and t0, t0, t1")?;
            }
            store_to_location(writer, "t0", &dest, state)?;
        }

        Instruction::Call {
            func_name,
            args,
            result,
        } => {
            // Pass up to 8 args in a0-a7
            for (i, arg) in args.iter().enumerate().take(8) {
                let op = super::util::get_value_operand_asm(arg, state, func_ctx)?;
                let reg = match i {
                    0 => "a0",
                    1 => "a1",
                    2 => "a2",
                    3 => "a3",
                    4 => "a4",
                    5 => "a5",
                    6 => "a6",
                    _ => "a7",
                };
                materialize_to_reg(writer, &op, reg, state)?;
            }
            // Call target (match function label convention)
            writeln!(writer, "    call func_{}", func_name)?;
            if let Some(res) = result {
                let dest = func_ctx.get_value_location(res)?.to_operand_string();
                store_to_location(writer, "a0", &dest, state)?;
            }
        }

        Instruction::WritePtr { ptr, result } => {
            // Expect ptr to be stack slot holding an address; write pointer-sized value to stdout
            let ptr_op = super::util::get_value_operand_asm(ptr, state, func_ctx)?;
            // Load target address into a1
            if ptr_op.contains("(s0)") {
                if matches!(state.width(), IsaWidth::Rv32) {
                    writeln!(writer, "    lw a1, {}", ptr_op)?;
                } else {
                    writeln!(writer, "    ld a1, {}", ptr_op)?;
                }
            } else {
                materialize_to_reg(writer, &ptr_op, "a1", state)?;
            }
            // a0=1 (stdout), a2=word size, a7=64 write
            writeln!(writer, "    li a0, 1")?;
            match state.width() {
                IsaWidth::Rv32 => writeln!(writer, "    li a2, 4")?,
                _ => writeln!(writer, "    li a2, 8")?,
            }
            writeln!(writer, "    li a7, 64")?;
            writeln!(writer, "    ecall")?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            store_to_location(writer, "a0", &dest, state)?;
        }

        Instruction::WriteByte { value, result } => {
            // Allocate small buffer on stack and write one byte via Linux write syscall
            let stack_alloc = if matches!(state.width(), IsaWidth::Rv32) {
                8
            } else {
                16
            };
            writeln!(writer, "    addi sp, sp, -{}", stack_alloc)?;

            // Store byte to [sp]
            let val_op = super::util::get_value_operand_asm(value, state, func_ctx)?;
            materialize_to_reg(writer, &val_op, "t0", state)?;
            writeln!(writer, "    sb t0, 0(sp)")?;

            // a0=1 (stdout), a1=sp, a2=1, a7=64 (sys_write), ecall
            writeln!(writer, "    li a0, 1")?;
            writeln!(writer, "    mv a1, sp")?;
            writeln!(writer, "    li a2, 1")?;
            writeln!(writer, "    li a7, 64")?;
            writeln!(writer, "    ecall")?;

            // Store syscall result (bytes written or -1)
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            store_to_location(writer, "a0", &dest, state)?;

            // Restore stack
            writeln!(writer, "    addi sp, sp, {}", stack_alloc)?;
        }

        Instruction::Write {
            buffer,
            size,
            result,
        } => {
            // Linux RISC-V write syscall: a7=64, a0=fd, a1=buf, a2=count
            writeln!(writer, "    li a0, 1")?; // stdout
            let buf_op = super::util::get_value_operand_asm(buffer, state, func_ctx)?;
            materialize_address(writer, &buf_op, "a1")?;
            let size_op = super::util::get_value_operand_asm(size, state, func_ctx)?;
            materialize_to_reg(writer, &size_op, "a2", state)?;
            writeln!(writer, "    li a7, 64")?;
            writeln!(writer, "    ecall")?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            store_to_location(writer, "a0", &dest, state)?;
        }

        Instruction::Read {
            buffer,
            size,
            result,
        } => {
            // Linux RISC-V read syscall: a7=63, a0=fd, a1=buf, a2=count
            writeln!(writer, "    li a0, 0")?; // stdin
            let buf_op = super::util::get_value_operand_asm(buffer, state, func_ctx)?;
            materialize_address(writer, &buf_op, "a1")?;
            let size_op = super::util::get_value_operand_asm(size, state, func_ctx)?;
            materialize_to_reg(writer, &size_op, "a2", state)?;
            writeln!(writer, "    li a7, 63")?;
            writeln!(writer, "    ecall")?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            store_to_location(writer, "a0", &dest, state)?;
        }

        Instruction::ReadByte { result } => {
            // Read one byte from stdin into stack buffer, return byte or error code
            let stack_alloc = if matches!(state.width(), IsaWidth::Rv32) {
                8
            } else {
                16
            };
            writeln!(writer, "    addi sp, sp, -{}", stack_alloc)?;
            writeln!(writer, "    li a0, 0")?; // stdin
            writeln!(writer, "    mv a1, sp")?; // buffer
            writeln!(writer, "    li a2, 1")?; // size
            writeln!(writer, "    li a7, 63")?; // read
            writeln!(writer, "    ecall")?;
            // a0 = bytes read or -1
            // If a0 == 1, load the byte and return it; else return a0
            writeln!(writer, "    li t1, 1")?;
            writeln!(writer, "    bne a0, t1, 1f")?;
            writeln!(writer, "    lbu t0, 0(sp)")?;
            writeln!(writer, "    mv a0, t0")?;
            writeln!(writer, "1:")?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            store_to_location(writer, "a0", &dest, state)?;
            writeln!(writer, "    addi sp, sp, {}", stack_alloc)?;
        }

        Instruction::Load { result, ty, ptr } => {
            // Only support stack offsets and simple registers/globals for now
            let ptr_op = super::util::get_value_operand_asm(ptr, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            if ptr_op.contains("(s0)") {
                // Direct stack slot load
                match ty {
                    Type::Primitive(PrimitiveType::I8)
                    | Type::Primitive(PrimitiveType::U8)
                    | Type::Primitive(PrimitiveType::Bool)
                    | Type::Primitive(PrimitiveType::Char) => {
                        writeln!(writer, "    lbu t0, {}", ptr_op)?;
                        store_to_location(writer, "t0", &dest, state)?;
                    }
                    Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => {
                        writeln!(writer, "    lhu t0, {}", ptr_op)?;
                        store_to_location(writer, "t0", &dest, state)?;
                    }
                    Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => {
                        writeln!(writer, "    lw t0, {}", ptr_op)?;
                        store_to_location(writer, "t0", &dest, state)?;
                    }
                    Type::Primitive(PrimitiveType::I64)
                    | Type::Primitive(PrimitiveType::U64)
                    | Type::Primitive(PrimitiveType::Ptr) => {
                        if matches!(state.width(), IsaWidth::Rv32) {
                            // Minimal RV32: load low word only
                            writeln!(writer, "    lw t0, {}", ptr_op)?;
                        } else {
                            writeln!(writer, "    ld t0, {}", ptr_op)?;
                        }
                        store_to_location(writer, "t0", &dest, state)?;
                    }
                    _ => {
                        return Err(LaminaError::CodegenError(
                            CodegenError::LoadNotImplementedForType(TypeInfo::Unknown(
                                ty.to_string(),
                            )),
                        ));
                    }
                }
            } else {
                // Treat ptr_op as an address in a register or label
                materialize_address(writer, &ptr_op, "t1")?;
                // Load 64-bit by default (or 32-bit on RV32)
                if matches!(state.width(), IsaWidth::Rv32) {
                    writeln!(writer, "    lw t0, 0(t1)")?;
                } else {
                    writeln!(writer, "    ld t0, 0(t1)")?;
                }
                store_to_location(writer, "t0", &dest, state)?;
            }
        }

        Instruction::Store { ty, ptr, value } => {
            let ptr_op = super::util::get_value_operand_asm(ptr, state, func_ctx)?;
            let val_op = super::util::get_value_operand_asm(value, state, func_ctx)?;
            materialize_to_reg(writer, &val_op, "t0", state)?;
            if ptr_op.contains("(s0)") {
                match ty {
                    Type::Primitive(PrimitiveType::I8)
                    | Type::Primitive(PrimitiveType::U8)
                    | Type::Primitive(PrimitiveType::Bool)
                    | Type::Primitive(PrimitiveType::Char) => {
                        writeln!(writer, "    sb t0, {}", ptr_op)?;
                    }
                    Type::Primitive(PrimitiveType::I16) | Type::Primitive(PrimitiveType::U16) => {
                        writeln!(writer, "    sh t0, {}", ptr_op)?;
                    }
                    Type::Primitive(PrimitiveType::I32) | Type::Primitive(PrimitiveType::U32) => {
                        writeln!(writer, "    sw t0, {}", ptr_op)?;
                    }
                    Type::Primitive(PrimitiveType::I64)
                    | Type::Primitive(PrimitiveType::U64)
                    | Type::Primitive(PrimitiveType::Ptr) => {
                        if matches!(state.width(), IsaWidth::Rv32) {
                            writeln!(writer, "    sw t0, {}", ptr_op)?;
                        } else {
                            writeln!(writer, "    sd t0, {}", ptr_op)?;
                        }
                    }
                    _ => {
                        return Err(LaminaError::CodegenError(
                            CodegenError::StoreNotImplementedForType(TypeInfo::Unknown(
                                ty.to_string(),
                            )),
                        ));
                    }
                }
            } else {
                materialize_address(writer, &ptr_op, "t1")?;
                if matches!(state.width(), IsaWidth::Rv32) {
                    writeln!(writer, "    sw t0, 0(t1)")?;
                } else {
                    writeln!(writer, "    sd t0, 0(t1)")?;
                }
            }
        }

        _ => {
            writeln!(writer, "    # Unimplemented on riscv: {}", instr)?;
        }
    }

    Ok(())
}

fn materialize_to_reg<W: Write>(
    writer: &mut W,
    op: &str,
    dest: &str,
    state: &CodegenState,
) -> Result<(), LaminaError> {
    if let Ok(imm) = op.parse::<i64>() {
        writeln!(writer, "    li {}, {}", dest, imm)?;
    } else if op.ends_with("(adrp+add)") || op.contains("(adrp+add)") {
        // simplistic label materialization for globals/rodata
        let label = op.trim_end_matches("(adrp+add)");
        writeln!(writer, "    la {}, {}", dest, label)?;
    } else if op.contains("(s0)") || op.contains("(fp)") || op.contains("(sp)") {
        // already an address expression; load from it
        if matches!(state.width(), IsaWidth::Rv32) {
            writeln!(writer, "    lw {}, {}", dest, op)?;
        } else {
            writeln!(writer, "    ld {}, {}", dest, op)?;
        }
    } else {
        // assume register name
        writeln!(writer, "    mv {}, {}", dest, op)?;
    }
    Ok(())
}

fn store_to_location<W: Write>(
    writer: &mut W,
    src_op: &str,
    dest: &str,
    state: &CodegenState,
) -> Result<(), LaminaError> {
    if dest.starts_with('a') || dest.starts_with('t') || dest.starts_with('s') {
        writeln!(writer, "    mv {}, {}", dest, src_op)?;
    } else if dest.contains("(s0)") {
        if matches!(state.width(), IsaWidth::Rv32) {
            writeln!(writer, "    sw {}, {}", src_op, dest)?;
        } else {
            writeln!(writer, "    sd {}, {}", src_op, dest)?;
        }
    } else {
        return Err(LaminaError::CodegenError(CodegenError::InternalError));
    }
    Ok(())
}

fn binary_i32<W: Write>(
    writer: &mut W,
    op: &BinaryOp,
    lhs: &str,
    rhs: &str,
    dest: &str,
    state: &CodegenState,
) -> Result<(), LaminaError> {
    materialize_to_reg(writer, lhs, "t0", state)?;
    materialize_to_reg(writer, rhs, "t1", state)?;
    match op {
        BinaryOp::Add => writeln!(writer, "    addw t2, t0, t1")?,
        BinaryOp::Sub => writeln!(writer, "    subw t2, t0, t1")?,
        BinaryOp::Mul => writeln!(writer, "    mulw t2, t0, t1")?,
        BinaryOp::Div => writeln!(writer, "    divw t2, t0, t1")?,
        BinaryOp::Rem => writeln!(writer, "    // TODO: implement remainder")?,
        BinaryOp::And => writeln!(writer, "    andw t2, t0, t1")?,
        BinaryOp::Or => writeln!(writer, "    orw  t2, t0, t1")?,
        BinaryOp::Xor => writeln!(writer, "    xorw t2, t0, t1")?,
        BinaryOp::Shl => writeln!(writer, "    sllw t2, t0, t1")?,
        BinaryOp::Shr => writeln!(writer, "    sraw t2, t0, t1")?,
    }
    store_to_location(writer, "t2", dest, state)
}

fn binary_i64<W: Write>(
    writer: &mut W,
    op: &BinaryOp,
    lhs: &str,
    rhs: &str,
    dest: &str,
    state: &CodegenState,
) -> Result<(), LaminaError> {
    materialize_to_reg(writer, lhs, "t0", state)?;
    materialize_to_reg(writer, rhs, "t1", state)?;
    match op {
        BinaryOp::Add => writeln!(writer, "    add t2, t0, t1")?,
        BinaryOp::Sub => writeln!(writer, "    sub t2, t0, t1")?,
        BinaryOp::Mul => writeln!(writer, "    mul t2, t0, t1")?,
        BinaryOp::Div => writeln!(writer, "    div t2, t0, t1")?,
        BinaryOp::Rem => writeln!(writer, "    // TODO: implement remainder")?,
        BinaryOp::And => writeln!(writer, "    and t2, t0, t1")?,
        BinaryOp::Or => writeln!(writer, "    or  t2, t0, t1")?,
        BinaryOp::Xor => writeln!(writer, "    xor t2, t0, t1")?,
        BinaryOp::Shl => writeln!(writer, "    sll t2, t0, t1")?,
        BinaryOp::Shr => writeln!(writer, "    sra t2, t0, t1")?,
    }
    store_to_location(writer, "t2", dest, state)
}

fn materialize_address<W: Write>(writer: &mut W, op: &str, dest: &str) -> Result<(), LaminaError> {
    if let Some(idx) = op.find('(') {
        // form: offset(s0)
        let off_str = &op[..idx];
        let offset: i64 = off_str.trim().parse().unwrap_or(0);
        if offset == 0 {
            writeln!(writer, "    mv {}, s0", dest)?;
        } else {
            writeln!(writer, "    addi {}, s0, {}", dest, offset)?;
        }
    } else if op.starts_with('a') || op.starts_with('t') || op.starts_with('s') {
        // register already holds an address
        writeln!(writer, "    mv {}, {}", dest, op)?;
    } else {
        // assume label
        writeln!(writer, "    la {}, {}", dest, op)?;
    }
    Ok(())
}
