use super::state::{CodegenState, FunctionContext, ValueLocation, RETURN_REGISTER};
use crate::codegen::{CodegenError, TypeInfo};
use crate::{BinaryOp, Identifier, Instruction, LaminaError, PrimitiveType, Result, Type, Value};
use std::io::Write;

pub fn generate_instruction<'a, W: Write>(
    instr: &Instruction<'a>,
    writer: &mut W,
    state: &mut CodegenState<'a>,
    func_ctx: &mut FunctionContext<'a>,
    _func_name: Identifier<'a>,
) -> Result<()> {
    writeln!(writer, "    # IR: {}", instr)?;

    match instr {
        Instruction::Ret { ty: _, value } => {
            if let Some(val) = value {
                let src = super::util::get_value_operand_asm(val, state, func_ctx)?;
                materialize_to_reg(writer, &src, RETURN_REGISTER)?;
            }
            writeln!(writer, "    j {}", func_ctx.epilogue_label)?;
        }

        Instruction::Binary { op, result, ty, lhs, rhs } => {
            let lhs_op = super::util::get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = super::util::get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();

            match ty {
                PrimitiveType::I32 | PrimitiveType::U32 => binary_i32(writer, op, &lhs_op, &rhs_op, &dest)?,
                PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::Ptr => binary_i64(writer, op, &lhs_op, &rhs_op, &dest)?,
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

        Instruction::Br { condition, true_label, false_label } => {
            let cond = super::util::get_value_operand_asm(condition, state, func_ctx)?;
            materialize_to_reg(writer, &cond, "t0")?;
            let tlabel = func_ctx.get_block_label(true_label)?;
            let flabel = func_ctx.get_block_label(false_label)?;
            writeln!(writer, "    bnez t0, {}", tlabel)?;
            writeln!(writer, "    j {}", flabel)?;
        }

        Instruction::Cmp { op, result, ty: _, lhs, rhs } => {
            // Very basic: compute lhs - rhs and set 1/0 in dest
            let lhs_op = super::util::get_value_operand_asm(lhs, state, func_ctx)?;
            let rhs_op = super::util::get_value_operand_asm(rhs, state, func_ctx)?;
            let dest = func_ctx.get_value_location(result)?.to_operand_string();
            materialize_to_reg(writer, &lhs_op, "t0")?;
            materialize_to_reg(writer, &rhs_op, "t1")?;
            match op {
                crate::CmpOp::Eq => {
                    writeln!(writer, "    sub t2, t0, t1")?;
                    store_to_location(writer, "seqz t3, t2", &dest, true)?;
                }
                crate::CmpOp::Ne => {
                    writeln!(writer, "    sub t2, t0, t1")?;
                    store_to_location(writer, "snez t3, t2", &dest, true)?;
                }
                crate::CmpOp::Lt => {
                    writeln!(writer, "    slt t3, t0, t1")?;
                    store_to_location(writer, "mv t3, t3", &dest, false)?;
                }
                crate::CmpOp::Le => {
                    writeln!(writer, "    sgt t2, t0, t1")?;
                    store_to_location(writer, "seqz t3, t2", &dest, true)?;
                }
                crate::CmpOp::Gt => {
                    writeln!(writer, "    sgt t3, t0, t1")?;
                    store_to_location(writer, "mv t3, t3", &dest, false)?;
                }
                crate::CmpOp::Ge => {
                    writeln!(writer, "    slt t2, t0, t1")?;
                    store_to_location(writer, "seqz t3, t2", &dest, true)?;
                }
            }
        }

        _ => {
            writeln!(writer, "    # Unimplemented on riscv: {}", instr)?;
        }
    }

    Ok(())
}

fn materialize_to_reg<W: Write>(writer: &mut W, op: &str, dest: &str) -> Result<()> {
    if let Ok(imm) = op.parse::<i64>() {
        writeln!(writer, "    li {}, {}", dest, imm)?;
    } else if op.ends_with("(adrp+add)") || op.contains("(adrp+add)") {
        // simplistic label materialization for globals/rodata
        let label = op.trim_end_matches("(adrp+add)");
        writeln!(writer, "    la {}, {}", dest, label)?;
    } else if op.contains("(s0)") || op.contains("(fp)") || op.contains("(sp)") {
        // already an address expression; load from it
        writeln!(writer, "    ld {}, {}", dest, op.replace("0(s0) # off ", ""))?;
    } else {
        // assume register name
        writeln!(writer, "    mv {}, {}", dest, op)?;
    }
    Ok(())
}

fn store_to_location<W: Write>(writer: &mut W, src_op: &str, dest: &str, is_pseudo: bool) -> Result<()> {
    if dest.starts_with('a') || dest.starts_with('t') || dest.starts_with('s') {
        if is_pseudo {
            writeln!(writer, "    {}", src_op)?;
            writeln!(writer, "    mv {}, t3", dest)?;
        } else {
            // src_op assumed to be register
            writeln!(writer, "    mv {}, {}", dest, src_op.split_whitespace().last().unwrap_or("t3"))?;
        }
    } else if dest.contains("(s0)") {
        if is_pseudo {
            writeln!(writer, "    {}", src_op)?;
            writeln!(writer, "    sd t3, {}", dest.replace("0(s0) # off ", ""))?;
        } else {
            writeln!(writer, "    sd {}, {}", src_op, dest.replace("0(s0) # off ", ""))?;
        }
    } else {
        return Err(LaminaError::CodegenError(CodegenError::InternalError));
    }
    Ok(())
}

fn binary_i32<W: Write>(writer: &mut W, op: &BinaryOp, lhs: &str, rhs: &str, dest: &str) -> Result<()> {
    materialize_to_reg(writer, lhs, "t0")?;
    materialize_to_reg(writer, rhs, "t1")?;
    match op {
        BinaryOp::Add => writeln!(writer, "    addw t2, t0, t1")?,
        BinaryOp::Sub => writeln!(writer, "    subw t2, t0, t1")?,
        BinaryOp::Mul => writeln!(writer, "    mulw t2, t0, t1")?,
        BinaryOp::Div => writeln!(writer, "    divw t2, t0, t1")?,
    }
    store_to_location(writer, "t2", dest, false)
}

fn binary_i64<W: Write>(writer: &mut W, op: &BinaryOp, lhs: &str, rhs: &str, dest: &str) -> Result<()> {
    materialize_to_reg(writer, lhs, "t0")?;
    materialize_to_reg(writer, rhs, "t1")?;
    match op {
        BinaryOp::Add => writeln!(writer, "    add t2, t0, t1")?,
        BinaryOp::Sub => writeln!(writer, "    sub t2, t0, t1")?,
        BinaryOp::Mul => writeln!(writer, "    mul t2, t0, t1")?,
        BinaryOp::Div => writeln!(writer, "    div t2, t0, t1")?,
    }
    store_to_location(writer, "t2", dest, false)
}


