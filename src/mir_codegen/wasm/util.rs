use std::io::Write;
use std::result::Result;

use crate::error::LaminaError;
use crate::mir::{Immediate, Operand, Register, VirtualReg};

/// Load an operand into the WASM stack
pub fn load_operand_wasm<W: Write>(
    operand: &Operand,
    writer: &mut W,
    vreg_to_local: &std::collections::HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(reg) => {
            load_register_wasm(reg, writer, vreg_to_local)?;
        }
        Operand::Immediate(imm) => match imm {
            Immediate::I8(val) => {
                writeln!(writer, "      i64.const {}", *val as i64)?;
            }
            Immediate::I16(val) => {
                writeln!(writer, "      i64.const {}", *val as i64)?;
            }
            Immediate::I32(val) => {
                writeln!(writer, "      i64.const {}", *val as i64)?;
            }
            Immediate::I64(val) => {
                writeln!(writer, "      i64.const {}", val)?;
            }
            Immediate::F32(_) | Immediate::F64(_) => {
                writeln!(writer, "      ;; TODO: floating point immediates")?;
                writeln!(writer, "      i64.const 0")?;
            }
        },
        #[allow(unreachable_patterns)]
        _ => {
            return Err(LaminaError::ValidationError(
                "Unsupported operand type for WASM".to_string(),
            ));
        }
    }
    Ok(())
}

/// Load a register onto the WASM stack
pub fn load_register_wasm<W: Write>(
    reg: &Register,
    writer: &mut W,
    vreg_to_local: &std::collections::HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match reg {
        Register::Physical(p) => {
            writeln!(writer, "      ;; TODO: physical register {}", p.name)?;
            writeln!(writer, "      i64.const 0")?;
        }
        Register::Virtual(v) => {
            if let Some(local_idx) = vreg_to_local.get(v) {
                writeln!(writer, "      local.get $l{}", local_idx)?;
            } else {
                writeln!(writer, "      ;; ERROR: no mapping for virtual register")?;
                writeln!(writer, "      i64.const 0")?;
            }
        }
    }
    Ok(())
}

/// Store the top of the WASM stack to a register
pub fn store_to_register_wasm<W: Write>(
    reg: &Register,
    writer: &mut W,
    vreg_to_local: &std::collections::HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match reg {
        Register::Physical(p) => {
            writeln!(
                writer,
                "      ;; TODO: store to physical register {}",
                p.name
            )?;
        }
        Register::Virtual(v) => {
            if let Some(local_idx) = vreg_to_local.get(v) {
                writeln!(writer, "      local.set $l{}", local_idx)?;
            } else {
                writeln!(writer, "      ;; ERROR: no mapping for virtual register")?;
            }
        }
    }
    Ok(())
}

/// Emit WASM instruction for integer binary operations
pub fn emit_int_binary_op<W: Write>(
    op: &crate::mir::IntBinOp,
    writer: &mut W,
) -> Result<(), LaminaError> {
    match op {
        crate::mir::IntBinOp::Add => writeln!(writer, "      i64.add")?,
        crate::mir::IntBinOp::Sub => writeln!(writer, "      i64.sub")?,
        crate::mir::IntBinOp::Mul => writeln!(writer, "      i64.mul")?,
        crate::mir::IntBinOp::SDiv => writeln!(writer, "      i64.div_s")?,
        crate::mir::IntBinOp::UDiv => writeln!(writer, "      i64.div_u")?,
        crate::mir::IntBinOp::SRem => writeln!(writer, "      i64.rem_s")?,
        crate::mir::IntBinOp::URem => writeln!(writer, "      i64.rem_u")?,
        crate::mir::IntBinOp::And => writeln!(writer, "      i64.and")?,
        crate::mir::IntBinOp::Or => writeln!(writer, "      i64.or")?,
        crate::mir::IntBinOp::Xor => writeln!(writer, "      i64.xor")?,
        crate::mir::IntBinOp::Shl => writeln!(writer, "      i64.shl")?,
        crate::mir::IntBinOp::AShr => writeln!(writer, "      i64.shr_s")?,
        crate::mir::IntBinOp::LShr => writeln!(writer, "      i64.shr_u")?,
    }
    Ok(())
}

/// Emit WASM instruction for integer comparison operations
pub fn emit_int_cmp_op<W: Write>(
    op: &crate::mir::IntCmpOp,
    writer: &mut W,
) -> Result<(), LaminaError> {
    match op {
        crate::mir::IntCmpOp::Eq => writeln!(writer, "      i64.eq")?,
        crate::mir::IntCmpOp::Ne => writeln!(writer, "      i64.ne")?,
        crate::mir::IntCmpOp::SLt => writeln!(writer, "      i64.lt_s")?,
        crate::mir::IntCmpOp::SLe => writeln!(writer, "      i64.le_s")?,
        crate::mir::IntCmpOp::SGt => writeln!(writer, "      i64.gt_s")?,
        crate::mir::IntCmpOp::SGe => writeln!(writer, "      i64.ge_s")?,
        crate::mir::IntCmpOp::ULt => writeln!(writer, "      i64.lt_u")?,
        crate::mir::IntCmpOp::ULe => writeln!(writer, "      i64.le_u")?,
        crate::mir::IntCmpOp::UGt => writeln!(writer, "      i64.gt_u")?,
        crate::mir::IntCmpOp::UGe => writeln!(writer, "      i64.ge_u")?,
    }
    Ok(())
}
