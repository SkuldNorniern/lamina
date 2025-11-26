use crate::mir::register::{Register, VirtualReg};
use crate::mir_codegen::regalloc::RegisterAllocator;

/// Load a virtual register into a destination register
pub fn load_register_to_register<W: std::io::Write>(
    src: &VirtualReg,
    writer: &mut W,
    reg_alloc: &crate::mir_codegen::riscv::regalloc::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(src) {
        writeln!(writer, "    mv {}, {}", dest_reg, phys)?;
    } else if let Some(offset) = stack_slots.get(src) {
        writeln!(writer, "    ld {}, {}(fp)", dest_reg, offset)?;
    } else {
        panic!("Virtual register {:?} has no mapping or stack slot", src);
    }
    Ok(())
}

/// Store a register to a virtual register
pub fn store_register_to_register<W: std::io::Write>(
    src_reg: &str,
    dst: &VirtualReg,
    writer: &mut W,
    reg_alloc: &crate::mir_codegen::riscv::regalloc::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
) -> Result<(), std::io::Error> {
    if let Some(phys) = reg_alloc.get_mapping(dst) {
        writeln!(writer, "    mv {}, {}", phys, src_reg)?;
    } else if let Some(offset) = stack_slots.get(dst) {
        writeln!(writer, "    sd {}, {}(fp)", src_reg, offset)?;
    } else {
        panic!("Virtual register {:?} has no mapping or stack slot", dst);
    }
    Ok(())
}

/// Load an operand into a destination register
pub fn load_operand_to_register<W: std::io::Write>(
    operand: &crate::mir::Operand,
    writer: &mut W,
    reg_alloc: &crate::mir_codegen::riscv::regalloc::RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<VirtualReg, i32>,
    dest_reg: &str,
) -> Result<(), std::io::Error> {
    match operand {
        crate::mir::Operand::Register(reg) => match reg {
            Register::Virtual(v) => {
                load_register_to_register(v, writer, reg_alloc, stack_slots, dest_reg)
            }
            Register::Physical(p) => {
                writeln!(writer, "    mv {}, {}", dest_reg, p.name)?;
                Ok(())
            }
        },
        crate::mir::Operand::Immediate(imm) => {
            match imm {
                crate::mir::instruction::Immediate::I8(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I16(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I32(v) => {
                    writeln!(writer, "    li {}, {}", dest_reg, *v as i64)?;
                }
                crate::mir::instruction::Immediate::I64(v) => {
                    if *v == 0 {
                        writeln!(writer, "    mv {}, zero", dest_reg)?;
                    } else {
                        writeln!(writer, "    li {}, {}", dest_reg, v)?;
                    }
                }
                crate::mir::instruction::Immediate::F32(_)
                | crate::mir::instruction::Immediate::F64(_) => {
                    writeln!(writer, "    # TODO: floating point immediates")?;
                    writeln!(writer, "    mv {}, zero", dest_reg)?;
                }
            }
            Ok(())
        }
    }
}

/// Emit RISC-V instruction for integer binary operations
pub fn emit_int_binary_op<W: std::io::Write>(
    op: &crate::mir::IntBinOp,
    writer: &mut W,
) -> Result<(), std::io::Error> {
    match op {
        crate::mir::IntBinOp::Add => writeln!(writer, "    add")?,
        crate::mir::IntBinOp::Sub => writeln!(writer, "    sub")?,
        crate::mir::IntBinOp::Mul => writeln!(writer, "    mul")?,
        crate::mir::IntBinOp::SDiv => writeln!(writer, "    div")?,
        crate::mir::IntBinOp::UDiv => writeln!(writer, "    divu")?,
        crate::mir::IntBinOp::SRem => writeln!(writer, "    rem")?,
        crate::mir::IntBinOp::URem => writeln!(writer, "    remu")?,
        crate::mir::IntBinOp::And => writeln!(writer, "    and")?,
        crate::mir::IntBinOp::Or => writeln!(writer, "    or")?,
        crate::mir::IntBinOp::Xor => writeln!(writer, "    xor")?,
        crate::mir::IntBinOp::Shl => writeln!(writer, "    sll")?,
        crate::mir::IntBinOp::AShr => writeln!(writer, "    sra")?,
        crate::mir::IntBinOp::LShr => writeln!(writer, "    srl")?,
    }
    Ok(())
}

/// Emit RISC-V instruction for integer comparison operations
pub fn emit_int_cmp_op<W: std::io::Write>(
    op: &crate::mir::IntCmpOp,
    writer: &mut W,
) -> Result<(), std::io::Error> {
    match op {
        crate::mir::IntCmpOp::Eq => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    seqz a0, a0")?; // Set if equal to zero
        }
        crate::mir::IntCmpOp::Ne => {
            writeln!(writer, "    xor a0, a0, a1")?;
            writeln!(writer, "    snez a0, a0")?; // Set if not equal to zero
        }
        crate::mir::IntCmpOp::SLt => writeln!(writer, "    slt a0, a0, a1")?,
        crate::mir::IntCmpOp::SLe => {
            writeln!(writer, "    sgt a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::SGt => writeln!(writer, "    sgt a0, a0, a1")?,
        crate::mir::IntCmpOp::SGe => {
            writeln!(writer, "    slt a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::ULt => writeln!(writer, "    sltu a0, a0, a1")?,
        crate::mir::IntCmpOp::ULe => {
            writeln!(writer, "    sgtu a0, a0, a1")?; // a <= b is !(a > b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
        crate::mir::IntCmpOp::UGt => writeln!(writer, "    sgtu a0, a0, a1")?,
        crate::mir::IntCmpOp::UGe => {
            writeln!(writer, "    sltu a0, a0, a1")?; // a >= b is !(a < b)
            writeln!(writer, "    xori a0, a0, 1")?; // Invert result
        }
    }
    Ok(())
}
