use super::*;
use crate::mir::{IntRotOp, IntUnOp};

pub(super) fn emit_int_unary<W: Write>(
    writer: &mut W,
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let bits = ty.size_bytes() * 8;
    load_operand_to_register(src, writer, reg_alloc, stack_slots, "3")?;
    zero_extend(writer, bits)?;
    match op {
        IntUnOp::Popcount => writeln!(writer, "    popcntd 3, 3")?,
        IntUnOp::Ctlz => {
            writeln!(writer, "    cntlzd 3, 3")?;
            if bits < 64 {
                writeln!(writer, "    addi 3, 3, -{}", 64 - bits)?;
            }
        }
        IntUnOp::Cttz => {
            if bits < 64 {
                emit_load_imm64(writer, 4, 1i64 << bits)?;
                writeln!(writer, "    or 3, 3, 4")?;
            }
            writeln!(writer, "    neg 4, 3")?;
            writeln!(writer, "    and 3, 3, 4")?;
            writeln!(writer, "    addi 3, 3, -1")?;
            writeln!(writer, "    popcntd 3, 3")?;
        }
        IntUnOp::Bswap => {
            if bits > 8 {
                emit_bswap64(writer)?;
                if bits < 64 {
                    writeln!(writer, "    srdi 3, 3, {}", 64 - bits)?;
                }
            }
        }
    }
    extend_integer(writer, "3", ty, true)?;
    store_r3(writer, dst, reg_alloc, stack_slots)
}

pub(super) fn emit_int_rotate<W: Write>(
    writer: &mut W,
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    (lhs, rhs): (&Operand, &Operand),
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let bits = ty.size_bytes() * 8;
    load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
    zero_extend(writer, bits)?;
    match bits {
        64 => {
            if op == IntRotOp::Rotr {
                writeln!(writer, "    neg 4, 4")?;
            }
            writeln!(writer, "    rotld 3, 3, 4")?;
        }
        32 => {
            if op == IntRotOp::Rotr {
                writeln!(writer, "    neg 4, 4")?;
            }
            writeln!(writer, "    rlwnm 3, 3, 4, 0, 31")?;
        }
        _ => {
            let mask = bits - 1;
            writeln!(writer, "    andi. 4, 4, {mask}")?;
            writeln!(writer, "    neg 0, 4")?;
            writeln!(writer, "    andi. 0, 0, {mask}")?;
            match op {
                IntRotOp::Rotl => {
                    writeln!(writer, "    sld 5, 3, 4")?;
                    writeln!(writer, "    srd 0, 3, 0")?;
                }
                IntRotOp::Rotr => {
                    writeln!(writer, "    srd 5, 3, 4")?;
                    writeln!(writer, "    sld 0, 3, 0")?;
                }
            }
            writeln!(writer, "    or 3, 5, 0")?;
            zero_extend(writer, bits)?;
        }
    }
    extend_integer(writer, "3", ty, true)?;
    store_r3(writer, dst, reg_alloc, stack_slots)
}

fn zero_extend<W: Write>(writer: &mut W, bits: usize) -> Result<(), LaminaError> {
    if bits < 64 {
        writeln!(writer, "    clrldi 3, 3, {}", 64 - bits)?;
    }
    Ok(())
}

fn emit_bswap64<W: Write>(writer: &mut W) -> Result<(), LaminaError> {
    for (mask, shift) in [
        (0x00FF_00FF_00FF_00FFi64, 8),
        (0x0000_FFFF_0000_FFFFi64, 16),
    ] {
        emit_load_imm64(writer, 0, mask)?;
        writeln!(writer, "    srdi 4, 3, {shift}")?;
        writeln!(writer, "    and 4, 4, 0")?;
        writeln!(writer, "    and 3, 3, 0")?;
        writeln!(writer, "    sldi 3, 3, {shift}")?;
        writeln!(writer, "    or 3, 3, 4")?;
    }
    writeln!(writer, "    rotldi 3, 3, 32")?;
    Ok(())
}

fn store_r3<W: Write>(
    writer: &mut W,
    dst: &Register,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(vreg) => store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?,
        Register::Physical(p) => writeln!(writer, "    mr {}, 3", p.name)?,
    }
    Ok(())
}
