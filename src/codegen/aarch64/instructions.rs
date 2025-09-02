use super::state::{ARG_REGISTERS, CodegenState, FunctionContext, ValueLocation, RETURN_REGISTER};
use super::util::{get_value_operand_asm, materialize_label_address};
use crate::{AllocType, BinaryOp, CmpOp, Identifier, Instruction, LaminaError, PrimitiveType, Result, Type};
use std::io::Write;

pub fn generate_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &FunctionContext<'a>,
    _func_name: Identifier<'a>,
) -> Result<()> {
    writeln!(writer, "        // IR: {}", instr)?;

    match instr {
        Instruction::Ret { ty: _, value } => {
            if let Some(val) = value {
                let src = get_value_operand_asm(val, state, func_ctx)?;
                match src.as_str() {
                    s if s.starts_with('x') => writeln!(writer, "        mov {}, {}", RETURN_REGISTER, s)?,
                    s if s.starts_with("[x29,") => {
                        materialize_address_operand(writer, s, "x9")?;
                        writeln!(writer, "        ldr {}, [x9]", RETURN_REGISTER)?;
                    }
                    _ => materialize_to_reg(writer, &src, RETURN_REGISTER)?,
                }
            }
            writeln!(writer, "        b {}", func_ctx.epilogue_label)?;
        }

        Instruction::Store { ty, ptr, value } => {
            let val = get_value_operand_asm(value, state, func_ctx)?;
            let ptr_op = get_value_operand_asm(ptr, state, func_ctx)?;
            materialize_address_operand(writer, &ptr_op, "x11")?;
            match ty {
                Type::Primitive(PrimitiveType::I32) => {
                    materialize_to_reg(writer, &val, "x10")?;
                    writeln!(writer, "        str w10, [x11]")?;
                }
                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => {
                    materialize_to_reg(writer, &val, "x10")?;
                    writeln!(writer, "        str x10, [x11]")?;
                }
                Type::Primitive(PrimitiveType::Bool) | Type::Primitive(PrimitiveType::I8) => {
                    materialize_to_reg(writer, &val, "x10")?;
                    writeln!(writer, "        strb w10, [x11]")?;
                }
                _ => return Err(LaminaError::CodegenError(format!("Store for type '{}' not implemented yet", ty))),
            }
        }

        Instruction::Alloc { result, alloc_type, .. } => match alloc_type {
            AllocType::Stack => {
                let result_loc = func_ctx.get_value_location(result)?;
                if let ValueLocation::StackOffset(offset) = result_loc {
                    materialize_address(writer, "x0", offset)?;
                    materialize_address(writer, "x9", offset)?;
                    writeln!(writer, "        str x0, [x9]")?;
                } else {
                    return Err(LaminaError::CodegenError("Stack allocation result location invalid".to_string()));
                }
            }
            AllocType::Heap => {
                return Err(LaminaError::CodegenError("Heap allocation requires runtime/libc (malloc)".to_string()));
            }
        },

        Instruction::Binary { op, result, ty, lhs, rhs } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            match ty {
                PrimitiveType::I32 => binary_i32(writer, op, &lhs_op, &rhs_op, &dest)?,
                PrimitiveType::I64 | PrimitiveType::Ptr => binary_i64(writer, op, &lhs_op, &rhs_op, &dest)?,
                _ => return Err(LaminaError::CodegenError(format!("Binary op for type '{}' not supported yet", ty))),
            }
        }

        Instruction::Load { result, ty, ptr } => {
            let ptr_op = get_value_operand_asm(ptr, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_address_operand(writer, &ptr_op, "x11")?;
            match ty {
                Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::Bool) => {
                    writeln!(writer, "        ldrb w10, [x11]")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                Type::Primitive(PrimitiveType::I32) => {
                    writeln!(writer, "        ldr w10, [x11]")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => {
                    writeln!(writer, "        ldr x10, [x11]")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                _ => return Err(LaminaError::CodegenError(format!("Load for type '{}' not implemented yet", ty))),
            }
        }

        Instruction::Cmp { op, result, ty: _, lhs, rhs } => {
            let lhs_op = get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &lhs_op, "x10")?;
            materialize_to_reg(writer, &rhs_op, "x11")?;
            writeln!(writer, "        cmp x10, x11")?;
            let set_instr = match op {
                CmpOp::Eq => ("cset", "eq"),
                CmpOp::Ne => ("cset", "ne"),
                CmpOp::Gt => ("cset", "gt"),
                CmpOp::Ge => ("cset", "ge"),
                CmpOp::Lt => ("cset", "lt"),
                CmpOp::Le => ("cset", "le"),
            };
            writeln!(writer, "        {} x12, {}", set_instr.0, set_instr.1)?;
            if dest.starts_with('x') {
                writeln!(writer, "        mov {}, x12", dest)?;
            } else {
                materialize_address_operand(writer, &dest, "x9")?;
                writeln!(writer, "        strb w12, [x9]")?;
            }
        }

        Instruction::Br { condition, true_label, false_label } => {
            let cond = get_value_operand_asm(condition, state, func_ctx)?;
            let tlabel = func_ctx.get_block_label(true_label)?;
            let flabel = func_ctx.get_block_label(false_label)?;
            materialize_to_reg(writer, &cond, "x10")?;
            writeln!(writer, "        cbnz x10, {}", tlabel)?;
            writeln!(writer, "        b {}", flabel)?;
        }

        Instruction::Call { func_name, args, result } => {
            for (i, arg) in args.iter().enumerate().take(ARG_REGISTERS.len()) {
                let op = get_value_operand_asm(arg, state, func_ctx)?;
                materialize_to_reg(writer, &op, ARG_REGISTERS[i])?;
            }
            writeln!(writer, "        bl func_{}", func_name)?;
            if let Some(res) = result {
                let dest = func_ctx.get_value_location(res)?.to_operand_string();
                store_to_location(writer, RETURN_REGISTER, &dest)?;
            }
        }

        Instruction::Jmp { target_label } => {
            let label = func_ctx.get_block_label(target_label)?;
            writeln!(writer, "        b {}", label)?;
        }

        Instruction::GetElemPtr { result, array_ptr, index } => {
            let base = get_value_operand_asm(array_ptr, state, func_ctx)?;
            let idx = get_value_operand_asm(index, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &base, "x0")?;
            materialize_to_reg(writer, &idx, "x1")?;
            writeln!(writer, "        lsl x1, x1, #3")?;
            writeln!(writer, "        add x0, x0, x1")?;
            store_to_location(writer, "x0", &dest)?;
        }

        Instruction::Print { value } => {
            let val = get_value_operand_asm(value, state, func_ctx)?;
            let fmt_label = state.add_rodata_string("%lld\n");
            for line in materialize_label_address("x0", &fmt_label) { writeln!(writer, "{}", line)?; }
            materialize_to_reg(writer, &val, "x1")?;
            writeln!(writer, "        bl _printf")?;
        }

        Instruction::ZeroExtend { result, source_type, target_type, value } => {
            let src = get_value_operand_asm(value, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            match (source_type, target_type) {
                (PrimitiveType::I8 | PrimitiveType::Bool, PrimitiveType::I32) => {
                    materialize_to_reg(writer, &src, "x10")?;
                    writeln!(writer, "        uxtb w10, w10")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                (PrimitiveType::I8 | PrimitiveType::Bool, PrimitiveType::I64) => {
                    materialize_to_reg(writer, &src, "x10")?;
                    writeln!(writer, "        uxtb x10, x10")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                (PrimitiveType::I32, PrimitiveType::I64) => {
                    materialize_to_reg(writer, &src, "x10")?;
                    writeln!(writer, "        mov w10, w10")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                (PrimitiveType::Bool, PrimitiveType::I32) => {
                    materialize_to_reg(writer, &src, "x10")?;
                    writeln!(writer, "        and w10, w10, #0xFF")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                (PrimitiveType::Bool, PrimitiveType::I64) => {
                    materialize_to_reg(writer, &src, "x10")?;
                    writeln!(writer, "        and x10, x10, #0xFF")?;
                    store_to_location(writer, "x10", &dest)?;
                }
                _ => return Err(LaminaError::CodegenError(format!("Unsupported zero extension: {} to {}", source_type, target_type))),
            }
        }

        _ => {
            writeln!(writer, "        // Unimplemented instruction on aarch64: {}", instr)?;
        }
    }

    Ok(())
}

fn materialize_to_reg<W: Write>(writer: &mut W, op: &str, dest: &str) -> Result<()> {
    if op.starts_with('x') {
        writeln!(writer, "        mov {}, {}", dest, op)?;
    } else if op.starts_with("[x29,") {
        materialize_address_operand(writer, op, "x9")?;
        writeln!(writer, "        ldr {}, [x9]", dest)?;
    } else if let Some(imm) = op.strip_prefix('#') {
        let value: u64 = imm.parse().map_err(|_| LaminaError::CodegenError("invalid immediate".into()))?;
        // Materialize 64-bit immediate with movz/movk sequence
        let mut first = true;
        for shift in [0u32, 16, 32, 48] {
            let part = ((value >> shift) & 0xFFFF) as u16;
            if part != 0 || first {
                if first {
                    writeln!(writer, "        movz {}, #{}, lsl #{}", dest, part, shift)?;
                    first = false;
                } else {
                    writeln!(writer, "        movk {}, #{}, lsl #{}", dest, part, shift)?;
                }
            }
        }
    } else if op.ends_with("(adrp+add)") {
        let label = op.trim_end_matches("(adrp+add)");
        for line in materialize_label_address(dest, label) { writeln!(writer, "{}", line)?; }
    } else {
        // Fallback try mov
        writeln!(writer, "        mov {}, {}", dest, op)?;
    }
    Ok(())
}

fn materialize_address_operand<W: Write>(writer: &mut W, op: &str, dest: &str) -> Result<()> {
    if op.starts_with("[x29,") {
        if let Some(off) = parse_fp_offset(op) {
            materialize_address(writer, dest, off)?;
            return Ok(());
        }
    }
    if op.ends_with("(adrp+add)") {
        let label = op.trim_end_matches("(adrp+add)");
        for line in materialize_label_address(dest, label) { writeln!(writer, "{}", line)?; }
        return Ok(());
    }
    Err(LaminaError::CodegenError(format!("Unsupported address operand: {}", op)))
}

fn materialize_address<W: Write>(writer: &mut W, dest: &str, offset: i64) -> Result<()> {
    if offset >= 0 {
        writeln!(writer, "        add {}, x29, #{}", dest, offset)?;
    } else {
        writeln!(writer, "        sub {}, x29, #{}", dest, (-offset))?;
    }
    Ok(())
}

fn parse_fp_offset(s: &str) -> Option<i64> {
    let start = s.find('#')? + 1;
    let end = s[start..].find(']')? + start;
    s[start..end].parse().ok()
}

fn store_to_location<W: Write>(writer: &mut W, src_reg: &str, dest: &str) -> Result<()> {
    if dest.starts_with('x') {
        writeln!(writer, "        mov {}, {}", dest, src_reg)?;
    } else if dest.starts_with("[x29,") {
        materialize_address_operand(writer, dest, "x9")?;
        writeln!(writer, "        str {}, [x9]", src_reg)?;
    } else {
        return Err(LaminaError::CodegenError(format!("Unsupported destination location: {}", dest)));
    }
    Ok(())
}

fn binary_i32<W: Write>(writer: &mut W, op: &BinaryOp, lhs: &str, rhs: &str, dest: &str) -> Result<()> {
    materialize_to_reg(writer, lhs, "x10")?;
    materialize_to_reg(writer, rhs, "x11")?;
    match op {
        BinaryOp::Add => writeln!(writer, "        add w12, w10, w11")?,
        BinaryOp::Sub => writeln!(writer, "        sub w12, w10, w11")?,
        BinaryOp::Mul => writeln!(writer, "        mul w12, w10, w11")?,
        BinaryOp::Div => writeln!(writer, "        sdiv w12, w10, w11")?,
    }
    if dest.starts_with("[x29,") {
        materialize_address_operand(writer, dest, "x9")?;
        writeln!(writer, "        str w12, [x9]")?;
    } else {
        writeln!(writer, "        mov {}, x12", dest)?;
    }
    Ok(())
}

fn binary_i64<W: Write>(writer: &mut W, op: &BinaryOp, lhs: &str, rhs: &str, dest: &str) -> Result<()> {
    materialize_to_reg(writer, lhs, "x10")?;
    materialize_to_reg(writer, rhs, "x11")?;
    match op {
        BinaryOp::Add => writeln!(writer, "        add x12, x10, x11")?,
        BinaryOp::Sub => writeln!(writer, "        sub x12, x10, x11")?,
        BinaryOp::Mul => writeln!(writer, "        mul x12, x10, x11")?,
        BinaryOp::Div => writeln!(writer, "        sdiv x12, x10, x11")?,
    }
    if dest.starts_with("[x29,") {
        materialize_address_operand(writer, dest, "x9")?;
        writeln!(writer, "        str x12, [x9]")?;
    } else {
        writeln!(writer, "        mov {}, x12", dest)?;
    }
    Ok(())
}


