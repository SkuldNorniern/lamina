use super::*;
use crate::mir::{AtomicBinOp, MemoryOrdering};
use lamina_platform::PointerWidth;

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let target = reg_alloc.target();
    match inst {
        MirInst::AtomicLoad {
            ty,
            dst,
            addr,
            ordering,
        } => {
            let size = access_size(ty, target, pointer_width)?;
            let load = load_op(size);
            address_into_t0(addr, writer, reg_alloc, stack_slots)?;
            match ordering {
                MemoryOrdering::Relaxed => writeln!(writer, "    {load} a0, 0(t0)")?,
                MemoryOrdering::Acquire => {
                    writeln!(writer, "    {load} a0, 0(t0)")?;
                    writeln!(writer, "    fence r, rw")?;
                }
                MemoryOrdering::SeqCst => {
                    writeln!(writer, "    fence rw, rw")?;
                    writeln!(writer, "    {load} a0, 0(t0)")?;
                    writeln!(writer, "    fence r, rw")?;
                }
                MemoryOrdering::Release | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic load", ordering));
                }
            }
            store_a0(writer, dst, reg_alloc, stack_slots)?;
        }
        MirInst::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            let size = access_size(ty, target, pointer_width)?;
            let store = store_op(size);
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "a1")?;
            address_into_t0(addr, writer, reg_alloc, stack_slots)?;
            match ordering {
                MemoryOrdering::Relaxed => {}
                MemoryOrdering::Release => writeln!(writer, "    fence rw, w")?,
                MemoryOrdering::SeqCst => writeln!(writer, "    fence rw, rw")?,
                MemoryOrdering::Acquire | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic store", ordering));
                }
            }
            writeln!(writer, "    {store} a1, 0(t0)")?;
        }
        MirInst::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ordering,
        } => {
            let size = access_size(ty, target, pointer_width)?;
            require_a(target)?;
            load_operand_to_register(value, writer, reg_alloc, stack_slots, "a1")?;
            address_into_t0(addr, writer, reg_alloc, stack_slots)?;
            if size >= 4 {
                if *op == AtomicBinOp::Sub {
                    writeln!(writer, "    neg a1, a1")?;
                }
                let name = match op {
                    AtomicBinOp::Exchange => "amoswap",
                    AtomicBinOp::Add | AtomicBinOp::Sub => "amoadd",
                    AtomicBinOp::And => "amoand",
                    AtomicBinOp::Or => "amoor",
                    AtomicBinOp::Xor => "amoxor",
                };
                writeln!(
                    writer,
                    "    {name}.{}{} a0, a1, (t0)",
                    width_suffix(size),
                    amo_suffix(ordering)
                )?;
            } else {
                writeln!(writer, "    li a3, {}", (1u64 << (size * 8)) - 1)?;
                writeln!(writer, "    and a1, a1, a3")?;
                sub_word_prologue(writer, size)?;
                let (lr, sc) = lr_sc_suffixes(ordering, ordering);
                writeln!(writer, "1:")?;
                writeln!(writer, "    lr.w{lr} a0, (t0)")?;
                writeln!(writer, "    and a4, a0, a7")?;
                writeln!(writer, "    srl a4, a4, a2")?;
                match op {
                    AtomicBinOp::Exchange => writeln!(writer, "    mv a5, a1")?,
                    AtomicBinOp::Add => writeln!(writer, "    add a5, a4, a1")?,
                    AtomicBinOp::Sub => writeln!(writer, "    sub a5, a4, a1")?,
                    AtomicBinOp::And => writeln!(writer, "    and a5, a4, a1")?,
                    AtomicBinOp::Or => writeln!(writer, "    or a5, a4, a1")?,
                    AtomicBinOp::Xor => writeln!(writer, "    xor a5, a4, a1")?,
                }
                writeln!(writer, "    and a5, a5, a3")?;
                writeln!(writer, "    sll a5, a5, a2")?;
                writeln!(writer, "    not a6, a7")?;
                writeln!(writer, "    and a0, a0, a6")?;
                writeln!(writer, "    or a0, a0, a5")?;
                writeln!(writer, "    sc.w{sc} a6, a0, (t0)")?;
                writeln!(writer, "    bnez a6, 1b")?;
                writeln!(writer, "    mv a0, a4")?;
            }
            extend_integer(writer, "a0", ty, target, true, pointer_width)?;
            store_a0(writer, dst, reg_alloc, stack_slots)?;
        }
        MirInst::AtomicCompareExchange {
            ty,
            dst,
            addr,
            expected,
            desired,
            success_ordering,
            failure_ordering,
        } => {
            let size = access_size(ty, target, pointer_width)?;
            require_a(target)?;
            load_operand_to_register(expected, writer, reg_alloc, stack_slots, "a1")?;
            load_operand_to_register(desired, writer, reg_alloc, stack_slots, "a3")?;
            address_into_t0(addr, writer, reg_alloc, stack_slots)?;
            let (lr, sc) = lr_sc_suffixes(success_ordering, failure_ordering);
            if size >= 4 {
                let w = width_suffix(size);
                writeln!(writer, "1:")?;
                writeln!(writer, "    lr.{w}{lr} a0, (t0)")?;
                writeln!(writer, "    bne a0, a1, 2f")?;
                writeln!(writer, "    sc.{w}{sc} a4, a3, (t0)")?;
                writeln!(writer, "    bnez a4, 1b")?;
                writeln!(writer, "    j 3f")?;
                writeln!(writer, "2:")?;
                writeln!(writer, "3:")?;
            } else {
                writeln!(writer, "    li a7, {}", (1u64 << (size * 8)) - 1)?;
                writeln!(writer, "    and a1, a1, a7")?;
                writeln!(writer, "    and a3, a3, a7")?;
                sub_word_prologue(writer, size)?;
                writeln!(writer, "    sll a5, a3, a2")?;
                writeln!(writer, "1:")?;
                writeln!(writer, "    lr.w{lr} a0, (t0)")?;
                writeln!(writer, "    and a4, a0, a7")?;
                writeln!(writer, "    srl a4, a4, a2")?;
                writeln!(writer, "    bne a4, a1, 2f")?;
                writeln!(writer, "    not a6, a7")?;
                writeln!(writer, "    and a0, a0, a6")?;
                writeln!(writer, "    or a0, a0, a5")?;
                writeln!(writer, "    sc.w{sc} a6, a0, (t0)")?;
                writeln!(writer, "    bnez a6, 1b")?;
                writeln!(writer, "    j 3f")?;
                writeln!(writer, "2:")?;
                writeln!(writer, "3:")?;
                writeln!(writer, "    mv a0, a4")?;
            }
            extend_integer(writer, "a0", ty, target, true, pointer_width)?;
            store_a0(writer, dst, reg_alloc, stack_slots)?;
        }
        MirInst::Fence { ordering } => match ordering {
            MemoryOrdering::Relaxed => {}
            MemoryOrdering::Acquire => writeln!(writer, "    fence r, rw")?,
            MemoryOrdering::Release => writeln!(writer, "    fence rw, w")?,
            MemoryOrdering::AcqRel | MemoryOrdering::SeqCst => {
                writeln!(writer, "    fence rw, rw")?;
            }
        },
        other => {
            return Err(LaminaError::ValidationError(format!(
                "RISC-V backend does not handle atomic instruction {other}"
            )));
        }
    }
    Ok(())
}

fn sub_word_prologue<W: Write>(writer: &mut W, size: usize) -> Result<(), LaminaError> {
    writeln!(writer, "    andi a2, t0, 3")?;
    writeln!(writer, "    slli a2, a2, 3")?;
    writeln!(writer, "    andi t0, t0, -4")?;
    writeln!(writer, "    li a7, {}", (1u64 << (size * 8)) - 1)?;
    writeln!(writer, "    sll a7, a7, a2")?;
    Ok(())
}

fn address_into_t0<W: Write>(
    addr: &AddressMode,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            match base {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "t0")?
                }
                Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
            }
            if *offset != 0 {
                writeln!(writer, "    li a4, {offset}")?;
                writeln!(writer, "    add t0, t0, a4")?;
            }
            Ok(())
        }
        _ => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            "RISC-V atomics support only base+offset addressing".to_string(),
        ))),
    }
}

fn lr_sc_suffixes(
    success: &MemoryOrdering,
    failure: &MemoryOrdering,
) -> (&'static str, &'static str) {
    let lr = if matches!(success, MemoryOrdering::SeqCst | MemoryOrdering::AcqRel)
        || matches!(failure, MemoryOrdering::SeqCst)
    {
        ".aqrl"
    } else if matches!(success, MemoryOrdering::Acquire)
        || matches!(failure, MemoryOrdering::Acquire)
    {
        ".aq"
    } else {
        ""
    };
    let sc = match success {
        MemoryOrdering::SeqCst | MemoryOrdering::AcqRel => ".aqrl",
        MemoryOrdering::Release => ".rl",
        _ => "",
    };
    (lr, sc)
}

fn amo_suffix(ordering: &MemoryOrdering) -> &'static str {
    match ordering {
        MemoryOrdering::Relaxed => "",
        MemoryOrdering::Acquire => ".aq",
        MemoryOrdering::Release => ".rl",
        MemoryOrdering::AcqRel | MemoryOrdering::SeqCst => ".aqrl",
    }
}

fn access_size(
    ty: &MirType,
    target: RiscVTarget,
    pointer_width: PointerWidth,
) -> Result<usize, LaminaError> {
    let size = ty.size_bytes(pointer_width);
    match size {
        1 | 2 | 4 => Ok(size),
        8 if target.xlen == Xlen::Rv64 => Ok(8),
        _ => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "RISC-V atomic operation on {size}-byte values is not supported on this target"
            ),
        ))),
    }
}

fn require_a(target: RiscVTarget) -> Result<(), LaminaError> {
    if target.a {
        Ok(())
    } else {
        Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "atomic read-modify-write needs the A extension, and the target is {}",
                target.isa_name()
            ),
        )))
    }
}

fn width_suffix(size: usize) -> &'static str {
    if size == 8 { "d" } else { "w" }
}

fn load_op(size: usize) -> &'static str {
    match size {
        1 => "lb",
        2 => "lh",
        4 => "lw",
        _ => "ld",
    }
}

fn store_op(size: usize) -> &'static str {
    match size {
        1 => "sb",
        2 => "sh",
        4 => "sw",
        _ => "sd",
    }
}

fn ordering_error(what: &str, ordering: &MemoryOrdering) -> LaminaError {
    LaminaError::ValidationError(format!(
        "RISC-V backend: {what} cannot use {ordering} ordering"
    ))
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
