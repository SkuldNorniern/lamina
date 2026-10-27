//! Atomic loads, stores, read-modify-write operations and fences for x86_64.

use super::*;
use crate::mir::{AtomicBinOp, MemoryOrdering};

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::AtomicLoad { ty, dst, addr, .. } => {
            let size = ty.size_bytes();
            load_base(address_base(addr)?, writer, reg_alloc, stack_slots, "rax")?;
            writeln!(writer, "    {}", sized_load("rax", size))?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::AtomicStore {
            ty,
            src,
            addr,
            ordering,
        } => {
            let size = ty.size_bytes();
            let base = address_base(addr)?;
            load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
            let ptr = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
            load_base(base, writer, reg_alloc, stack_slots, ptr)?;
            let rax = register_for_size("rax", size).ok_or_else(invalid_size)?;
            let suffix = size_suffix_for(size);
            if *ordering == MemoryOrdering::SeqCst {
                writeln!(writer, "    xchg{suffix} %{rax}, (%{ptr})")?;
            } else {
                writeln!(writer, "    mov{suffix} %{rax}, (%{ptr})")?;
            }
            reg_alloc.free_scratch(ptr);
        }
        MirInst::AtomicBinary {
            op,
            ty,
            dst,
            addr,
            value,
            ..
        } => {
            let size = ty.size_bytes();
            let suffix = size_suffix_for(size);
            let rax = register_for_size("rax", size).ok_or_else(invalid_size)?;
            let ptr = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
            match op {
                AtomicBinOp::Exchange => {
                    load_operand_to_rax(value, writer, reg_alloc, stack_slots)?;
                    load_base(address_base(addr)?, writer, reg_alloc, stack_slots, ptr)?;
                    writeln!(writer, "    xchg{suffix} %{rax}, (%{ptr})")?;
                }
                AtomicBinOp::Add | AtomicBinOp::Sub => {
                    load_operand_to_rax(value, writer, reg_alloc, stack_slots)?;
                    if *op == AtomicBinOp::Sub {
                        writeln!(writer, "    negq %rax")?;
                    }
                    load_base(address_base(addr)?, writer, reg_alloc, stack_slots, ptr)?;
                    writeln!(writer, "    lock xadd{suffix} %{rax}, (%{ptr})")?;
                }
                AtomicBinOp::And | AtomicBinOp::Or | AtomicBinOp::Xor => {
                    let new = match reg_alloc.alloc_scratch() {
                        Some(new) => new,
                        None => {
                            reg_alloc.free_scratch(ptr);
                            return Err(scratch_error());
                        }
                    };
                    let mnemonic = match op {
                        AtomicBinOp::And => "and",
                        AtomicBinOp::Or => "or",
                        _ => "xor",
                    };
                    let new_sized = register_for_size(new, size).ok_or_else(invalid_size)?;
                    load_base(address_base(addr)?, writer, reg_alloc, stack_slots, ptr)?;
                    writeln!(writer, "    {} (%{ptr}), %rax", sized_load_mnemonic(size))?;
                    writeln!(writer, "1:")?;
                    load_operand_to_register(value, writer, reg_alloc, stack_slots, new)?;
                    writeln!(writer, "    {mnemonic}q %rax, %{new}")?;
                    writeln!(writer, "    lock cmpxchg{suffix} %{new_sized}, (%{ptr})")?;
                    writeln!(writer, "    jne 1b")?;
                    reg_alloc.free_scratch(new);
                }
                AtomicBinOp::CompareExchange => {
                    reg_alloc.free_scratch(ptr);
                    return Err(LaminaError::ValidationError(
                        "x86_64 backend does not support atomic_cmpxchg yet".to_string(),
                    ));
                }
            }
            reg_alloc.free_scratch(ptr);
            extend_rax(size, writer)?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::AtomicCompareExchange { .. } => {
            return Err(LaminaError::ValidationError(
                "x86_64 backend does not support atomic_cmpxchg yet".to_string(),
            ));
        }
        MirInst::Fence { ordering } => {
            if *ordering == MemoryOrdering::SeqCst {
                writeln!(writer, "    mfence")?;
            }
        }
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support atomic instruction {other}"
            )));
        }
    }
    Ok(())
}

fn address_base(addr: &AddressMode) -> Result<&Register, LaminaError> {
    match addr {
        AddressMode::BaseOffset {
            base, offset: 0, ..
        } => Ok(base),
        _ => Err(LaminaError::ValidationError(format!(
            "x86_64 backend does not support atomic address mode {addr:?}"
        ))),
    }
}

fn load_base<W: Write>(
    base: &Register,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target: &str,
) -> Result<(), LaminaError> {
    match base {
        Register::Virtual(vreg) => {
            load_register_to_register(vreg, writer, reg_alloc, stack_slots, target)?
        }
        Register::Physical(phys) => writeln!(writer, "    movq %{}, %{target}", phys.name)?,
    }
    Ok(())
}

fn sized_load(reg: &str, size: usize) -> String {
    match size {
        1 => format!("movsbq (%{reg}), %rax"),
        2 => format!("movswq (%{reg}), %rax"),
        4 => format!("movslq (%{reg}), %rax"),
        _ => format!("movq (%{reg}), %rax"),
    }
}

fn sized_load_mnemonic(size: usize) -> &'static str {
    match size {
        1 => "movsbq",
        2 => "movswq",
        4 => "movslq",
        _ => "movq",
    }
}

fn extend_rax<W: Write>(size: usize, writer: &mut W) -> Result<(), LaminaError> {
    match size {
        1 => writeln!(writer, "    movsbq %al, %rax")?,
        2 => writeln!(writer, "    movswq %ax, %rax")?,
        4 => writeln!(writer, "    movslq %eax, %rax")?,
        _ => {}
    }
    Ok(())
}

fn size_suffix_for(size: usize) -> &'static str {
    match size {
        1 => "b",
        2 => "w",
        4 => "l",
        _ => "q",
    }
}

fn invalid_size() -> LaminaError {
    LaminaError::ValidationError("x86_64 atomic operation has an unsupported width".to_string())
}
