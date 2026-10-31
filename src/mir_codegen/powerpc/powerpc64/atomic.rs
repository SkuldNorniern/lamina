use super::*;
use crate::mir::{AtomicBinOp, MemoryOrdering};
use lamina_platform::PointerWidth;

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::AtomicLoad {
            ty,
            dst,
            addr,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            address_into_r5(addr, writer, reg_alloc, stack_slots)?;
            match ordering {
                MemoryOrdering::Relaxed => {}
                MemoryOrdering::Acquire => {}
                MemoryOrdering::SeqCst => writeln!(writer, "    sync")?,
                MemoryOrdering::Release | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic load", ordering));
                }
            }
            writeln!(writer, "    {} 3, 0(5)", load_op(size))?;
            if matches!(ordering, MemoryOrdering::Acquire | MemoryOrdering::SeqCst) {
                writeln!(writer, "    lwsync")?;
            }
            extend_integer(writer, "3", ty, true, pointer_width)?;
            store_r3(writer, dst, reg_alloc, stack_slots)?;
        }
        MirInst::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            address_into_r5(addr, writer, reg_alloc, stack_slots)?;
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "4")?;
            match ordering {
                MemoryOrdering::Relaxed => {}
                MemoryOrdering::Release => writeln!(writer, "    lwsync")?,
                MemoryOrdering::SeqCst => writeln!(writer, "    sync")?,
                MemoryOrdering::Acquire | MemoryOrdering::AcqRel => {
                    return Err(ordering_error("atomic store", ordering));
                }
            }
            writeln!(writer, "    {} 4, 0(5)", store_op(size))?;
        }
        MirInst::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ordering,
        } => {
            let size = access_size(ty, pointer_width)?;
            address_into_r5(addr, writer, reg_alloc, stack_slots)?;
            load_operand_to_register(value, writer, reg_alloc, stack_slots, "4")?;
            leading_barrier(writer, ordering)?;
            let (lwarx, stwcx) = (load_reserve(size), store_conditional(size));
            writeln!(writer, "1:")?;
            writeln!(writer, "    {lwarx} 3, 0, 5")?;
            let new = match op {
                AtomicBinOp::Exchange => "4",
                AtomicBinOp::Add => {
                    writeln!(writer, "    add 0, 3, 4")?;
                    "0"
                }
                AtomicBinOp::Sub => {
                    writeln!(writer, "    subf 0, 4, 3")?;
                    "0"
                }
                AtomicBinOp::And => {
                    writeln!(writer, "    and 0, 3, 4")?;
                    "0"
                }
                AtomicBinOp::Or => {
                    writeln!(writer, "    or 0, 3, 4")?;
                    "0"
                }
                AtomicBinOp::Xor => {
                    writeln!(writer, "    xor 0, 3, 4")?;
                    "0"
                }
            };
            writeln!(writer, "    {stwcx} {new}, 0, 5")?;
            writeln!(writer, "    bne 1b")?;
            trailing_barrier(writer, ordering)?;
            extend_integer(writer, "3", ty, true, pointer_width)?;
            store_r3(writer, dst, reg_alloc, stack_slots)?;
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
            let size = access_size(ty, pointer_width)?;
            if matches!(
                failure_ordering,
                MemoryOrdering::Release | MemoryOrdering::AcqRel
            ) {
                return Err(ordering_error("cmpxchg failure", failure_ordering));
            }
            address_into_r5(addr, writer, reg_alloc, stack_slots)?;
            load_operand_to_register(expected, writer, reg_alloc, stack_slots, "4")?;
            if size < 8 {
                writeln!(writer, "    clrldi 4, 4, {}", 64 - size * 8)?;
            }
            load_operand_to_register(desired, writer, reg_alloc, stack_slots, "0")?;
            leading_barrier(writer, success_ordering)?;
            writeln!(writer, "1:")?;
            writeln!(writer, "    {} 3, 0, 5", load_reserve(size))?;
            writeln!(writer, "    cmpd 3, 4")?;
            writeln!(writer, "    bne 2f")?;
            writeln!(writer, "    {} 0, 0, 5", store_conditional(size))?;
            writeln!(writer, "    bne 1b")?;
            writeln!(writer, "2:")?;
            if is_acquire(success_ordering) || is_acquire(failure_ordering) {
                writeln!(writer, "    isync")?;
            }
            extend_integer(writer, "3", ty, true, pointer_width)?;
            store_r3(writer, dst, reg_alloc, stack_slots)?;
        }
        MirInst::Fence { ordering } => match ordering {
            MemoryOrdering::Relaxed => {}
            MemoryOrdering::Acquire | MemoryOrdering::Release | MemoryOrdering::AcqRel => {
                writeln!(writer, "    lwsync")?;
            }
            MemoryOrdering::SeqCst => writeln!(writer, "    sync")?,
        },
        other => {
            return Err(LaminaError::ValidationError(format!(
                "PowerPC64 backend does not handle atomic instruction {other}"
            )));
        }
    }
    Ok(())
}

fn leading_barrier<W: Write>(writer: &mut W, ordering: &MemoryOrdering) -> Result<(), LaminaError> {
    match ordering {
        MemoryOrdering::Release | MemoryOrdering::AcqRel => writeln!(writer, "    lwsync")?,
        MemoryOrdering::SeqCst => writeln!(writer, "    sync")?,
        MemoryOrdering::Relaxed | MemoryOrdering::Acquire => {}
    }
    Ok(())
}

fn trailing_barrier<W: Write>(
    writer: &mut W,
    ordering: &MemoryOrdering,
) -> Result<(), LaminaError> {
    if is_acquire(ordering) {
        writeln!(writer, "    isync")?;
    }
    Ok(())
}

fn is_acquire(ordering: &MemoryOrdering) -> bool {
    matches!(
        ordering,
        MemoryOrdering::Acquire | MemoryOrdering::AcqRel | MemoryOrdering::SeqCst
    )
}

fn address_into_r5<W: Write>(
    addr: &AddressMode,
    writer: &mut W,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            match base {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?
                }
                Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
            }
            if *offset != 0 {
                let offset = i64::from(*offset);
                if (-0x8000..0x8000).contains(&offset) {
                    writeln!(writer, "    addi 5, 5, {offset}")?;
                } else {
                    emit_load_imm64(writer, 0, offset)?;
                    writeln!(writer, "    add 5, 5, 0")?;
                }
            }
            Ok(())
        }
        _ => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            "PowerPC64 atomics support only base+offset addressing".to_string(),
        ))),
    }
}

fn access_size(ty: &MirType, pointer_width: PointerWidth) -> Result<usize, LaminaError> {
    match ty.size_bytes(pointer_width) {
        size @ (1 | 2 | 4 | 8) => Ok(size),
        _ => Err(LaminaError::ValidationError(
            "PowerPC64 atomic operation has an unsupported width".into(),
        )),
    }
}

fn load_op(size: usize) -> &'static str {
    match size {
        1 => "lbz",
        2 => "lhz",
        4 => "lwz",
        _ => "ld",
    }
}

fn store_op(size: usize) -> &'static str {
    match size {
        1 => "stb",
        2 => "sth",
        4 => "stw",
        _ => "std",
    }
}

fn load_reserve(size: usize) -> &'static str {
    match size {
        1 => "lbarx",
        2 => "lharx",
        4 => "lwarx",
        _ => "ldarx",
    }
}

fn store_conditional(size: usize) -> &'static str {
    match size {
        1 => "stbcx.",
        2 => "sthcx.",
        4 => "stwcx.",
        _ => "stdcx.",
    }
}

fn ordering_error(what: &str, ordering: &MemoryOrdering) -> LaminaError {
    LaminaError::ValidationError(format!(
        "PowerPC64 backend: {what} cannot use {ordering} ordering"
    ))
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
