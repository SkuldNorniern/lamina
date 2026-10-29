use super::*;
use crate::mir::{IntRotOp, IntUnOp};

pub(super) fn emit_int_unary<W: Write>(
    writer: &mut W,
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let target = reg_alloc.target();
    require_rv64(target, "integer bit intrinsics")?;
    let bits = ty.size_bytes() * 8;
    load_operand_to_register(src, writer, reg_alloc, stack_slots, "a0")?;
    zero_extend(writer, "a0", bits)?;
    match op {
        IntUnOp::Popcount => emit_popcount(writer)?,
        IntUnOp::Ctlz => {
            emit_clz(writer)?;
            if bits < 64 {
                writeln!(writer, "    addi a0, a0, -{}", 64 - bits)?;
            }
        }
        IntUnOp::Cttz => {
            if bits < 64 {
                writeln!(writer, "    li a1, {}", 1u64 << bits)?;
                writeln!(writer, "    or a0, a0, a1")?;
            }
            writeln!(writer, "    neg a1, a0")?;
            writeln!(writer, "    and a0, a0, a1")?;
            writeln!(writer, "    addi a0, a0, -1")?;
            emit_popcount(writer)?;
        }
        IntUnOp::Bswap => {
            if bits > 8 {
                emit_bswap64(writer)?;
                if bits < 64 {
                    writeln!(writer, "    srli a0, a0, {}", 64 - bits)?;
                }
            }
        }
    }
    extend_integer(writer, "a0", ty, target, true)?;
    store_a0(writer, dst, reg_alloc, stack_slots)
}

pub(super) fn emit_int_rotate<W: Write>(
    writer: &mut W,
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    (lhs, rhs): (&Operand, &Operand),
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let target = reg_alloc.target();
    require_rv64(target, "integer rotates")?;
    let bits = ty.size_bytes() * 8;
    let mask = bits - 1;
    load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;
    zero_extend(writer, "a0", bits)?;
    writeln!(writer, "    li a3, {mask}")?;
    writeln!(writer, "    and a1, a1, a3")?;
    writeln!(writer, "    neg a2, a1")?;
    writeln!(writer, "    and a2, a2, a3")?;
    let (near, far) = match op {
        IntRotOp::Rotl => ("sll", "srl"),
        IntRotOp::Rotr => ("srl", "sll"),
    };
    writeln!(writer, "    {near} a4, a0, a1")?;
    writeln!(writer, "    {far} a5, a0, a2")?;
    writeln!(writer, "    or a0, a4, a5")?;
    zero_extend(writer, "a0", bits)?;
    extend_integer(writer, "a0", ty, target, true)?;
    store_a0(writer, dst, reg_alloc, stack_slots)
}

fn require_rv64(target: RiscVTarget, what: &str) -> Result<(), LaminaError> {
    if target.xlen == Xlen::Rv64 {
        Ok(())
    } else {
        Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!("RISC-V32 backend does not support {what} yet"),
        )))
    }
}

fn zero_extend<W: Write>(writer: &mut W, reg: &str, bits: usize) -> Result<(), LaminaError> {
    if bits < 64 {
        writeln!(writer, "    slli {reg}, {reg}, {}", 64 - bits)?;
        writeln!(writer, "    srli {reg}, {reg}, {}", 64 - bits)?;
    }
    Ok(())
}

fn emit_popcount<W: Write>(writer: &mut W) -> Result<(), LaminaError> {
    writeln!(writer, "    srli a1, a0, 1")?;
    writeln!(writer, "    li a2, 0x5555555555555555")?;
    writeln!(writer, "    and a1, a1, a2")?;
    writeln!(writer, "    sub a0, a0, a1")?;
    writeln!(writer, "    li a2, 0x3333333333333333")?;
    writeln!(writer, "    and a1, a0, a2")?;
    writeln!(writer, "    srli a0, a0, 2")?;
    writeln!(writer, "    and a0, a0, a2")?;
    writeln!(writer, "    add a0, a0, a1")?;
    writeln!(writer, "    srli a1, a0, 4")?;
    writeln!(writer, "    add a0, a0, a1")?;
    writeln!(writer, "    li a2, 0x0f0f0f0f0f0f0f0f")?;
    writeln!(writer, "    and a0, a0, a2")?;
    for shift in [8, 16, 32] {
        writeln!(writer, "    srli a1, a0, {shift}")?;
        writeln!(writer, "    add a0, a0, a1")?;
    }
    writeln!(writer, "    andi a0, a0, 0xff")?;
    Ok(())
}

fn emit_clz<W: Write>(writer: &mut W) -> Result<(), LaminaError> {
    writeln!(writer, "    li a1, 0")?;
    for step in [32u32, 16, 8, 4, 2, 1] {
        writeln!(writer, "    srli a2, a0, {}", 64 - step)?;
        writeln!(writer, "    sltiu a2, a2, 1")?;
        writeln!(writer, "    slli a2, a2, {}", step.trailing_zeros())?;
        writeln!(writer, "    add a1, a1, a2")?;
        writeln!(writer, "    sll a0, a0, a2")?;
    }
    writeln!(writer, "    sltiu a2, a0, 1")?;
    writeln!(writer, "    add a0, a1, a2")?;
    Ok(())
}

fn emit_bswap64<W: Write>(writer: &mut W) -> Result<(), LaminaError> {
    for (mask, shift) in [
        (0x00FF_00FF_00FF_00FFu64, 8),
        (0x0000_FFFF_0000_FFFFu64, 16),
    ] {
        writeln!(writer, "    li a2, {mask}")?;
        writeln!(writer, "    srli a1, a0, {shift}")?;
        writeln!(writer, "    and a1, a1, a2")?;
        writeln!(writer, "    and a0, a0, a2")?;
        writeln!(writer, "    slli a0, a0, {shift}")?;
        writeln!(writer, "    or a0, a0, a1")?;
    }
    writeln!(writer, "    srli a1, a0, 32")?;
    writeln!(writer, "    slli a0, a0, 32")?;
    writeln!(writer, "    or a0, a0, a1")?;
    Ok(())
}

fn store_a0<W: Write>(
    writer: &mut W,
    dst: &Register,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(vreg) => {
            store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?
        }
        Register::Physical(p) => writeln!(writer, "    mv {}, a0", p.name)?,
    }
    Ok(())
}
