use std::io::Write;
use std::result::Result;

use crate::error::LaminaError;
use crate::mir::{
    FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, IntBinOp, IntCmpOp, Operand, Register, VirtualReg,
};

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
            Immediate::F32(val) => {
                writeln!(writer, "      f32.const {}", val)?;
            }
            Immediate::F64(val) => {
                writeln!(writer, "      f64.const {}", val)?;
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
        Register::Physical(_) => {
            return Err(LaminaError::ValidationError(
                "WASM: Physical registers are not supported. WASM uses a stack-based execution model with local variables only.".to_string(),
            ));
        }
        Register::Virtual(v) => {
            if let Some(local_idx) = vreg_to_local.get(v) {
                writeln!(writer, "      local.get $l{}", local_idx)?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "WASM: No mapping found for virtual register {:?}. This indicates a register allocation error.",
                    v
                )));
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
        Register::Physical(_) => {
            return Err(LaminaError::ValidationError(
                "WASM: Physical registers are not supported. WASM uses a stack-based execution model with local variables only.".to_string(),
            ));
        }
        Register::Virtual(v) => {
            if let Some(local_idx) = vreg_to_local.get(v) {
                writeln!(writer, "      local.set $l{}", local_idx)?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "WASM: No mapping found for virtual register {:?}. This indicates a register allocation error.",
                    v
                )));
            }
        }
    }
    Ok(())
}

/// Emit WASM instruction for integer binary operations
pub fn emit_int_binary_op<W: Write>(op: &IntBinOp, writer: &mut W) -> Result<(), LaminaError> {
    match op {
        IntBinOp::Add => writeln!(writer, "      i64.add")?,
        IntBinOp::Sub => writeln!(writer, "      i64.sub")?,
        IntBinOp::Mul => writeln!(writer, "      i64.mul")?,
        IntBinOp::SDiv => writeln!(writer, "      i64.div_s")?,
        IntBinOp::UDiv => writeln!(writer, "      i64.div_u")?,
        IntBinOp::SRem => writeln!(writer, "      i64.rem_s")?,
        IntBinOp::URem => writeln!(writer, "      i64.rem_u")?,
        IntBinOp::And => writeln!(writer, "      i64.and")?,
        IntBinOp::Or => writeln!(writer, "      i64.or")?,
        IntBinOp::Xor => writeln!(writer, "      i64.xor")?,
        IntBinOp::Shl => writeln!(writer, "      i64.shl")?,
        IntBinOp::AShr => writeln!(writer, "      i64.shr_s")?,
        IntBinOp::LShr => writeln!(writer, "      i64.shr_u")?,
    }
    Ok(())
}

/// Emit WASM instruction for integer comparison operations
pub fn emit_int_cmp_op<W: Write>(op: &IntCmpOp, writer: &mut W) -> Result<(), LaminaError> {
    match op {
        IntCmpOp::Eq => writeln!(writer, "      i64.eq")?,
        IntCmpOp::Ne => writeln!(writer, "      i64.ne")?,
        IntCmpOp::SLt => writeln!(writer, "      i64.lt_s")?,
        IntCmpOp::SLe => writeln!(writer, "      i64.le_s")?,
        IntCmpOp::SGt => writeln!(writer, "      i64.gt_s")?,
        IntCmpOp::SGe => writeln!(writer, "      i64.ge_s")?,
        IntCmpOp::ULt => writeln!(writer, "      i64.lt_u")?,
        IntCmpOp::ULe => writeln!(writer, "      i64.le_u")?,
        IntCmpOp::UGt => writeln!(writer, "      i64.gt_u")?,
        IntCmpOp::UGe => writeln!(writer, "      i64.ge_u")?,
    }
    Ok(())
}

/// Load a floating-point operand onto the WASM stack
/// WASM stores all values as i64, so we need to reinterpret for FP operations
pub fn load_fp_operand_wasm<W: Write>(
    operand: &Operand,
    writer: &mut W,
    vreg_to_local: &std::collections::HashMap<VirtualReg, usize>,
    is_f32: bool,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(reg) => {
            // Load i64 value first
            load_register_wasm(reg, writer, vreg_to_local)?;
            // Reinterpret as float
            if is_f32 {
                writeln!(writer, "      i32.wrap_i64")?;
                writeln!(writer, "      f32.reinterpret_i32")?;
            } else {
                writeln!(writer, "      f64.reinterpret_i64")?;
            }
        }
        Operand::Immediate(imm) => match imm {
            Immediate::F32(val) => {
                writeln!(writer, "      f32.const {}", val)?;
            }
            Immediate::F64(val) => {
                writeln!(writer, "      f64.const {}", val)?;
            }
            _ => {
                return Err(LaminaError::ValidationError(
                    "Expected floating-point immediate for FP operation".to_string(),
                ));
            }
        },
        #[allow(unreachable_patterns)]
        _ => {
            return Err(LaminaError::ValidationError(
                "Unsupported operand type for WASM FP".to_string(),
            ));
        }
    }
    Ok(())
}

/// Store a floating-point value from the WASM stack to a register
/// WASM stores all values as i64, so we reinterpret from float
pub fn store_fp_to_register_wasm<W: Write>(
    writer: &mut W,
    vreg_to_local: &std::collections::HashMap<VirtualReg, usize>,
    vreg: &VirtualReg,
    is_f32: bool,
) -> Result<(), LaminaError> {
    // Reinterpret float as integer for storage
    if is_f32 {
        writeln!(writer, "      i32.reinterpret_f32")?;
        writeln!(writer, "      i64.extend_i32_u")?;
    } else {
        writeln!(writer, "      i64.reinterpret_f64")?;
    }

    if let Some(local_idx) = vreg_to_local.get(vreg) {
        writeln!(writer, "      local.set $l{}", local_idx)?;
    } else {
        return Err(LaminaError::ValidationError(format!(
            "WASM: No mapping found for virtual register {:?}",
            vreg
        )));
    }
    Ok(())
}

/// Emit WASM instruction for floating-point binary operations
pub fn emit_float_binary_op<W: Write>(
    op: &FloatBinOp,
    writer: &mut W,
    is_f32: bool,
) -> Result<(), LaminaError> {
    let suffix = if is_f32 { "f32" } else { "f64" };
    match op {
        FloatBinOp::FAdd => writeln!(writer, "      {}.add", suffix)?,
        FloatBinOp::FSub => writeln!(writer, "      {}.sub", suffix)?,
        FloatBinOp::FMul => writeln!(writer, "      {}.mul", suffix)?,
        FloatBinOp::FDiv => writeln!(writer, "      {}.div", suffix)?,
    }
    Ok(())
}

/// Emit WASM instruction for floating-point unary operations
pub fn emit_float_unary_op<W: Write>(
    op: &FloatUnOp,
    writer: &mut W,
    is_f32: bool,
) -> Result<(), LaminaError> {
    let suffix = if is_f32 { "f32" } else { "f64" };
    match op {
        FloatUnOp::FNeg => writeln!(writer, "      {}.neg", suffix)?,
        FloatUnOp::FSqrt => writeln!(writer, "      {}.sqrt", suffix)?,
    }
    Ok(())
}

/// Emit WASM instruction for floating-point comparison operations
pub fn emit_float_cmp_op<W: Write>(
    op: &FloatCmpOp,
    writer: &mut W,
    is_f32: bool,
) -> Result<(), LaminaError> {
    let suffix = if is_f32 { "f32" } else { "f64" };
    match op {
        FloatCmpOp::Eq => writeln!(writer, "      {}.eq", suffix)?,
        FloatCmpOp::Ne => writeln!(writer, "      {}.ne", suffix)?,
        FloatCmpOp::Lt => writeln!(writer, "      {}.lt", suffix)?,
        FloatCmpOp::Le => writeln!(writer, "      {}.le", suffix)?,
        FloatCmpOp::Gt => writeln!(writer, "      {}.gt", suffix)?,
        FloatCmpOp::Ge => writeln!(writer, "      {}.ge", suffix)?,
    }
    Ok(())
}
