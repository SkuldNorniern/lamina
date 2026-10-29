//! Instruction lowering for the armv7 backend.

use crate::mir_codegen::common::{unsupported_overflow_flag, unsupported_thread_local};
use std::io::Write;

use crate::{
    error::LaminaError,
    mir::{
        Aggregate, FloatBinOp, FloatUnOp, Instruction as MirInst, IntBinOp, IntRotOp, IntUnOp,
        MirType, Operand, Register, ScalarType,
    },
    mir_codegen::{
        MirCodegenSettings,
        arm::{float_condition, int_condition},
        common::{emit_source_loc, integer_conversion_with_width},
    },
};
use lamina_codegen::aarch64::FrameMap;
use lamina_codegen::abi::Abi;
use lamina_mir::SourceLoc;
use lamina_platform::PointerWidth;

use super::abi::{Armv7Abi, Signatures};
use super::call;
use super::operand::{
    ADDR, F32_SCRATCH, F64_SCRATCH, Scratch, add_offset, extend_integer, frame_address, load_reg,
    materialize, materialize_float, mov_imm32, store_result, unsupported,
};

pub(crate) const POINTER_WIDTH: PointerWidth = PointerWidth::Bits32;

pub(crate) struct Ctx<'a> {
    pub(crate) frame: &'a FrameMap,
    pub(crate) abi: &'a Armv7Abi,
    pub(crate) signatures: &'a Signatures,
    pub(crate) epilogue: &'a str,
    pub(crate) block_prefix: &'a str,
    pub(crate) ret_aggregate: Option<Aggregate>,
    pub(crate) ret_float: Option<MirType>,
    pub(crate) sret_slot: Option<i32>,
    pub(crate) settings: &'a MirCodegenSettings,
}

/// Width of a value the backend can hold in one core register.
pub(crate) fn is_wide_integer(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::I64))
}

/// The one message the backend gives for an operation the i64 legalization pass has
/// not split into register pairs yet. It names the operation so the failure points at
/// the program, never at a silently truncated answer.
pub(crate) fn reject_wide(operation: &str) -> LaminaError {
    unsupported(format!(
        "armv7 has no 64-bit integer registers, so `{operation}` on i64 cannot be lowered until \
         LUMIR legalizes 64-bit integers into 32-bit register pairs; narrow the value to i32"
    ))
}

fn check_scalar(operation: &str, ty: &MirType) -> Result<(), LaminaError> {
    if is_wide_integer(ty) {
        return Err(reject_wide(operation));
    }
    if ty.is_vector() {
        return Err(unsupported(format!(
            "armv7 backend has no vector unit: `{operation}` on {ty}"
        )));
    }
    Ok(())
}

fn float_scratch(double: bool, index: usize) -> &'static str {
    if double {
        F64_SCRATCH[index]
    } else {
        F32_SCRATCH[index]
    }
}

fn suffix(double: bool) -> &'static str {
    if double { "f64" } else { "f32" }
}

/// Materializes a boolean from the flags: zero, then one under `cond`.
fn set_condition<W: Write>(w: &mut W, dest: &str, cond: &str) -> Result<(), LaminaError> {
    writeln!(w, "    mov {dest}, #0")?;
    writeln!(w, "    mov{cond} {dest}, #1")?;
    Ok(())
}

/// Moves exactly `bytes` through `ip`, from the address in `source` to the address in
/// `destination`: whole words first, then the tail a byte at a time. A caller's buffer
/// for a result is the object itself and carries no padding, so nothing may be written
/// past `bytes`.
pub(crate) fn copy_through_pointer<W: Write>(
    w: &mut W,
    source: &str,
    destination: &str,
    bytes: usize,
) -> Result<(), LaminaError> {
    for word in 0..bytes / 4 {
        let at = word * 4;
        writeln!(w, "    ldr {ADDR}, [{source}, #{at}]")?;
        writeln!(w, "    str {ADDR}, [{destination}, #{at}]")?;
    }
    for at in bytes - bytes % 4..bytes {
        writeln!(w, "    ldrb {ADDR}, [{source}, #{at}]")?;
        writeln!(w, "    strb {ADDR}, [{destination}, #{at}]")?;
    }
    Ok(())
}

pub(crate) fn emit_instruction<W: Write>(
    w: &mut W,
    inst: &MirInst,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
    loc: Option<SourceLoc>,
) -> Result<(), LaminaError> {
    emit_source_loc(w, ctx.settings, loc)?;
    match inst {
        MirInst::Comment { text } => writeln!(w, "    @ {text}")?,
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {}
        MirInst::Unreachable => writeln!(w, "    udf #0")?,

        MirInst::ThreadLocalAddr { .. } => {
            return Err(unsupported_thread_local("armv7"));
        }
        MirInst::IntOverflow { .. } => {
            return Err(unsupported_overflow_flag("armv7"));
        }
        MirInst::SymbolAddress { dst, symbol } => {
            let label = ctx.abi.mangle_function_name(symbol);
            let value = scratch.alloc()?;
            writeln!(w, "    movw {value}, #:lower16:{label}")?;
            writeln!(w, "    movt {value}, #:upper16:{label}")?;
            store_result(w, dst, value, ctx.frame)?;
            scratch.free(value);
        }

        MirInst::IntConvert { .. } => {
            if let Some(operations) =
                integer_conversion_with_width(inst, MirType::Scalar(ScalarType::I32), POINTER_WIDTH)
            {
                for operation in &operations {
                    emit_instruction(w, operation, ctx, scratch, None)?;
                }
            }
        }

        MirInst::Copy { ty, dst, src } => emit_move(w, ty, dst, src, ctx, scratch, "copy")?,
        MirInst::Bitcast { to, dst, src, .. } => {
            emit_move(w, to, dst, src, ctx, scratch, "bitcast")?
        }

        MirInst::IntBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => emit_int_binary(w, *op, ty, dst, lhs, rhs, ctx, scratch)?,

        MirInst::IntUnary { op, ty, dst, src } => {
            emit_int_unary(w, *op, ty, dst, src, ctx, scratch)?
        }

        MirInst::IntRotate {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => emit_int_rotate(w, *op, ty, dst, lhs, rhs, ctx, scratch)?,

        MirInst::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let double = ty.size_bytes(POINTER_WIDTH) == 8;
            let a = float_scratch(double, 0);
            let b = float_scratch(double, 1);
            materialize_float(w, lhs, a, double, ctx.frame, scratch)?;
            materialize_float(w, rhs, b, double, ctx.frame, scratch)?;
            let mnemonic = match op {
                FloatBinOp::FAdd => "vadd",
                FloatBinOp::FSub => "vsub",
                FloatBinOp::FMul => "vmul",
                FloatBinOp::FDiv => "vdiv",
            };
            writeln!(w, "    {mnemonic}.{} {a}, {a}, {b}", suffix(double))?;
            store_float(w, dst, a, ctx)?;
        }

        MirInst::FloatUnary { op, ty, dst, src } => {
            let double = ty.size_bytes(POINTER_WIDTH) == 8;
            let a = float_scratch(double, 0);
            materialize_float(w, src, a, double, ctx.frame, scratch)?;
            let mnemonic = match op {
                FloatUnOp::FNeg => "vneg",
                FloatUnOp::FSqrt => "vsqrt",
            };
            writeln!(w, "    {mnemonic}.{} {a}, {a}", suffix(double))?;
            store_float(w, dst, a, ctx)?;
        }

        MirInst::IntCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            check_scalar("icmp", ty)?;
            let signed = !matches!(int_condition(*op), "lo" | "ls" | "hi" | "hs");
            let a = scratch.alloc()?;
            let b = scratch.alloc()?;
            materialize(w, lhs, a, ctx.frame)?;
            materialize(w, rhs, b, ctx.frame)?;
            extend_integer(w, a, ty, signed)?;
            extend_integer(w, b, ty, signed)?;
            writeln!(w, "    cmp {a}, {b}")?;
            set_condition(w, a, int_condition(*op))?;
            store_result(w, dst, a, ctx.frame)?;
            scratch.free(a);
            scratch.free(b);
        }

        MirInst::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let double = ty.size_bytes(POINTER_WIDTH) == 8;
            let a = float_scratch(double, 0);
            let b = float_scratch(double, 1);
            materialize_float(w, lhs, a, double, ctx.frame, scratch)?;
            materialize_float(w, rhs, b, double, ctx.frame, scratch)?;
            writeln!(w, "    vcmp.{} {a}, {b}", suffix(double))?;
            writeln!(w, "    vmrs APSR_nzcv, fpscr")?;
            let value = scratch.alloc()?;
            set_condition(w, value, float_condition(*op))?;
            store_result(w, dst, value, ctx.frame)?;
            scratch.free(value);
        }

        MirInst::Select {
            ty,
            dst,
            cond,
            true_val,
            false_val,
        } => {
            if ty.is_float() && ty.size_bytes(POINTER_WIDTH) == 8 {
                let a = float_scratch(true, 0);
                let b = float_scratch(true, 1);
                let flag = scratch.alloc()?;
                load_reg(w, cond, flag, ctx.frame)?;
                materialize_float(w, true_val, a, true, ctx.frame, scratch)?;
                materialize_float(w, false_val, b, true, ctx.frame, scratch)?;
                writeln!(w, "    cmp {flag}, #0")?;
                writeln!(w, "    vmoveq.f64 {a}, {b}")?;
                store_float(w, dst, a, ctx)?;
                scratch.free(flag);
            } else {
                check_scalar("select", ty)?;
                let flag = scratch.alloc()?;
                let a = scratch.alloc()?;
                let b = scratch.alloc()?;
                load_reg(w, cond, flag, ctx.frame)?;
                materialize(w, true_val, a, ctx.frame)?;
                materialize(w, false_val, b, ctx.frame)?;
                writeln!(w, "    cmp {flag}, #0")?;
                writeln!(w, "    moveq {a}, {b}")?;
                store_result(w, dst, a, ctx.frame)?;
                scratch.free(flag);
                scratch.free(a);
                scratch.free(b);
            }
        }

        MirInst::Load { ty, dst, addr, .. } => {
            check_scalar("load", ty)?;
            let address = scratch.alloc()?;
            super::operand::materialize_address(w, addr, address, ctx.frame, scratch)?;
            if ty.is_float() && ty.size_bytes(POINTER_WIDTH) == 8 {
                let value = F64_SCRATCH[0];
                writeln!(w, "    vldr {value}, [{address}]")?;
                store_float(w, dst, value, ctx)?;
            } else {
                let value = scratch.alloc()?;
                match ty.size_bytes(POINTER_WIDTH) {
                    1 => writeln!(w, "    ldrsb {value}, [{address}]")?,
                    2 => writeln!(w, "    ldrsh {value}, [{address}]")?,
                    4 => writeln!(w, "    ldr {value}, [{address}]")?,
                    other => return Err(unsupported(format!("armv7 load of {other} bytes"))),
                }
                store_result(w, dst, value, ctx.frame)?;
                scratch.free(value);
            }
            scratch.free(address);
        }

        MirInst::Store { ty, src, addr, .. } => {
            check_scalar("store", ty)?;
            let address = scratch.alloc()?;
            super::operand::materialize_address(w, addr, address, ctx.frame, scratch)?;
            if ty.is_float() && ty.size_bytes(POINTER_WIDTH) == 8 {
                let value = F64_SCRATCH[0];
                materialize_float(w, src, value, true, ctx.frame, scratch)?;
                writeln!(w, "    vstr {value}, [{address}]")?;
            } else {
                let value = scratch.alloc()?;
                materialize(w, src, value, ctx.frame)?;
                match ty.size_bytes(POINTER_WIDTH) {
                    1 => writeln!(w, "    strb {value}, [{address}]")?,
                    2 => writeln!(w, "    strh {value}, [{address}]")?,
                    4 => writeln!(w, "    str {value}, [{address}]")?,
                    other => return Err(unsupported(format!("armv7 store of {other} bytes"))),
                }
                scratch.free(value);
            }
            scratch.free(address);
        }

        MirInst::StackAddr { dst, slot } => {
            let offset = ctx
                .frame
                .slot_of(&Register::Virtual(*slot))
                .ok_or_else(|| unsupported("armv7 stack address has no frame slot".to_string()))?;
            let value = scratch.alloc()?;
            frame_address(w, value, offset)?;
            store_result(w, dst, value, ctx.frame)?;
            scratch.free(value);
        }

        MirInst::Lea { dst, base, offset } => {
            let value = scratch.alloc()?;
            load_reg(w, base, value, ctx.frame)?;
            add_offset(w, value, i64::from(*offset))?;
            store_result(w, dst, value, ctx.frame)?;
            scratch.free(value);
        }

        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            if is_wide_integer(from) {
                return Err(reject_wide("sitofp/uitofp"));
            }
            let value = scratch.alloc()?;
            materialize(w, src, value, ctx.frame)?;
            extend_integer(w, value, from, *signed)?;
            let double = to.size_bytes(POINTER_WIDTH) == 8;
            let holder = F32_SCRATCH[0];
            writeln!(w, "    vmov {holder}, {value}")?;
            let source = if *signed { "s32" } else { "u32" };
            if double {
                writeln!(w, "    vcvt.f64.{source} {}, {holder}", F64_SCRATCH[0])?;
                store_float(w, dst, F64_SCRATCH[0], ctx)?;
            } else {
                writeln!(w, "    vcvt.f32.{source} {holder}, {holder}")?;
                store_float(w, dst, holder, ctx)?;
            }
            scratch.free(value);
        }

        MirInst::FloatConvert { to, dst, src, .. } => {
            let widen = to.size_bytes(POINTER_WIDTH) == 8;
            if widen {
                materialize_float(w, src, F32_SCRATCH[0], false, ctx.frame, scratch)?;
                writeln!(w, "    vcvt.f64.f32 {}, {}", F64_SCRATCH[0], F32_SCRATCH[0])?;
                store_float(w, dst, F64_SCRATCH[0], ctx)?;
            } else {
                materialize_float(w, src, F64_SCRATCH[0], true, ctx.frame, scratch)?;
                writeln!(w, "    vcvt.f32.f64 {}, {}", F32_SCRATCH[0], F64_SCRATCH[0])?;
                store_float(w, dst, F32_SCRATCH[0], ctx)?;
            }
        }

        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            if is_wide_integer(to) {
                return Err(reject_wide("fptosi/fptoui"));
            }
            let double = from.size_bytes(POINTER_WIDTH) == 8;
            let holder = float_scratch(double, 0);
            materialize_float(w, src, holder, double, ctx.frame, scratch)?;
            let target = if *signed { "s32" } else { "u32" };
            writeln!(
                w,
                "    vcvt.{target}.{} {}, {holder}",
                suffix(double),
                F32_SCRATCH[0]
            )?;
            let value = scratch.alloc()?;
            writeln!(w, "    vmov {value}, {}", F32_SCRATCH[0])?;
            saturate_narrow(w, value, to, *signed, scratch)?;
            store_result(w, dst, value, ctx.frame)?;
            scratch.free(value);
        }

        MirInst::Jmp { target } => writeln!(w, "    b {}{target}", ctx.block_prefix)?,

        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            let flag = scratch.alloc()?;
            load_reg(w, cond, flag, ctx.frame)?;
            writeln!(w, "    cmp {flag}, #0")?;
            writeln!(w, "    bne {}{true_target}", ctx.block_prefix)?;
            writeln!(w, "    b {}{false_target}", ctx.block_prefix)?;
            scratch.free(flag);
        }

        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            let subject = scratch.alloc()?;
            load_reg(w, value, subject, ctx.frame)?;
            for (constant, target) in cases {
                let probe = scratch.alloc()?;
                mov_imm32(w, probe, *constant as u32)?;
                writeln!(w, "    cmp {subject}, {probe}")?;
                writeln!(w, "    beq {}{target}", ctx.block_prefix)?;
                scratch.free(probe);
            }
            writeln!(w, "    b {}{default}", ctx.block_prefix)?;
            scratch.free(subject);
        }

        MirInst::Call { name, args, ret } => {
            call::emit_call(w, name, args, ret.as_ref(), ctx, scratch)?;
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed,
            ret,
            ret_ty,
        } => call::emit_call_indirect(
            w,
            target,
            arg_types,
            args,
            *variadic_fixed,
            ret.as_ref(),
            *ret_ty,
            ctx,
            scratch,
        )?,
        MirInst::TailCall { name, args } => call::emit_tail_call(w, name, args, ctx, scratch)?,
        MirInst::Ret { value } => call::emit_return(w, value.as_ref(), ctx, scratch)?,

        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            return Err(unsupported(
                "armv7 backend does not implement atomic operations".to_string(),
            ));
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported(
                "armv7 backend does not implement vector operations".to_string(),
            ));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported(
                "armv7 backend does not implement inline assembly".to_string(),
            ));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(unsupported(
                "armv7 backend does not implement variadic argument reading".to_string(),
            ));
        }
    }
    Ok(())
}

fn store_float<W: Write>(
    w: &mut W,
    dst: &Register,
    source: &str,
    ctx: &Ctx<'_>,
) -> Result<(), LaminaError> {
    super::operand::store_float_result(w, dst, source, ctx.frame)
}

fn emit_move<W: Write>(
    w: &mut W,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
    operation: &str,
) -> Result<(), LaminaError> {
    if ty.is_vector() {
        return Err(unsupported(format!(
            "armv7 backend has no vector unit: `{operation}` of {ty}"
        )));
    }
    if ty.size_bytes(POINTER_WIDTH) == 8 {
        if !ty.is_float() {
            return Err(reject_wide(operation));
        }
        let value = F64_SCRATCH[0];
        materialize_float(w, src, value, true, ctx.frame, scratch)?;
        return store_float(w, dst, value, ctx);
    }
    let value = scratch.alloc()?;
    materialize(w, src, value, ctx.frame)?;
    store_result(w, dst, value, ctx.frame)?;
    scratch.free(value);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_binary<W: Write>(
    w: &mut W,
    op: IntBinOp,
    ty: &MirType,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    check_scalar(int_binary_name(op), ty)?;
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    let a = scratch.alloc()?;
    let b = scratch.alloc()?;
    let d = scratch.alloc()?;
    materialize(w, lhs, a, ctx.frame)?;
    materialize(w, rhs, b, ctx.frame)?;

    let unsigned = matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr);
    if unsigned {
        extend_integer(w, a, ty, false)?;
        extend_integer(w, b, ty, false)?;
    } else if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
        extend_integer(w, a, ty, true)?;
        if !matches!(op, IntBinOp::AShr) {
            extend_integer(w, b, ty, true)?;
        }
    }

    let (zero, overflow) = op.division_checks(
        *ty,
        lhs.integer_constant(),
        rhs.integer_constant(),
        POINTER_WIDTH,
    );
    if zero {
        writeln!(w, "    cmp {b}, #0")?;
        writeln!(w, "    bne 1f")?;
        writeln!(w, "    udf #0")?;
        writeln!(w, "1:")?;
    }
    if overflow {
        let minimum = 1u64 << (bits - 1);
        mov_imm32(
            w,
            d,
            ty.signed_integer(minimum as i64, POINTER_WIDTH) as u32,
        )?;
        writeln!(w, "    cmn {b}, #1")?;
        writeln!(w, "    bne 2f")?;
        writeln!(w, "    cmp {a}, {d}")?;
        writeln!(w, "    bne 2f")?;
        writeln!(w, "    udf #0")?;
        writeln!(w, "2:")?;
    }
    if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) {
        writeln!(w, "    and {b}, {b}, #{}", bits - 1)?;
    }

    match op {
        IntBinOp::Add => writeln!(w, "    add {d}, {a}, {b}")?,
        IntBinOp::Sub => writeln!(w, "    sub {d}, {a}, {b}")?,
        IntBinOp::Mul => writeln!(w, "    mul {d}, {a}, {b}")?,
        IntBinOp::UDiv => writeln!(w, "    udiv {d}, {a}, {b}")?,
        IntBinOp::SDiv => writeln!(w, "    sdiv {d}, {a}, {b}")?,
        IntBinOp::URem => {
            writeln!(w, "    udiv {d}, {a}, {b}")?;
            writeln!(w, "    mls {d}, {d}, {b}, {a}")?;
        }
        IntBinOp::SRem => {
            writeln!(w, "    sdiv {d}, {a}, {b}")?;
            writeln!(w, "    mls {d}, {d}, {b}, {a}")?;
        }
        IntBinOp::And => writeln!(w, "    and {d}, {a}, {b}")?,
        IntBinOp::Or => writeln!(w, "    orr {d}, {a}, {b}")?,
        IntBinOp::Xor => writeln!(w, "    eor {d}, {a}, {b}")?,
        IntBinOp::Shl => writeln!(w, "    lsl {d}, {a}, {b}")?,
        IntBinOp::LShr => writeln!(w, "    lsr {d}, {a}, {b}")?,
        IntBinOp::AShr => writeln!(w, "    asr {d}, {a}, {b}")?,
    }
    extend_integer(w, d, ty, true)?;
    store_result(w, dst, d, ctx.frame)?;
    scratch.free(a);
    scratch.free(b);
    scratch.free(d);
    Ok(())
}

fn int_binary_name(op: IntBinOp) -> &'static str {
    match op {
        IntBinOp::Add => "add",
        IntBinOp::Sub => "sub",
        IntBinOp::Mul => "mul",
        IntBinOp::UDiv => "udiv",
        IntBinOp::SDiv => "sdiv",
        IntBinOp::URem => "urem",
        IntBinOp::SRem => "srem",
        IntBinOp::And => "and",
        IntBinOp::Or => "or",
        IntBinOp::Xor => "xor",
        IntBinOp::Shl => "shl",
        IntBinOp::LShr => "lshr",
        IntBinOp::AShr => "ashr",
    }
}

fn emit_int_unary<W: Write>(
    w: &mut W,
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let name = match op {
        IntUnOp::Popcount => "popcount",
        IntUnOp::Ctlz => "ctlz",
        IntUnOp::Cttz => "cttz",
        IntUnOp::Bswap => "bswap",
    };
    check_scalar(name, ty)?;
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    let value = scratch.alloc()?;
    materialize(w, src, value, ctx.frame)?;
    extend_integer(w, value, ty, false)?;
    match op {
        IntUnOp::Bswap => {
            if bits == 16 {
                writeln!(w, "    rev16 {value}, {value}")?;
                writeln!(w, "    uxth {value}, {value}")?;
            } else if bits == 32 {
                writeln!(w, "    rev {value}, {value}")?;
            }
        }
        IntUnOp::Ctlz => {
            writeln!(w, "    clz {value}, {value}")?;
            if bits < 32 {
                writeln!(w, "    sub {value}, {value}, #{}", 32 - bits)?;
            }
        }
        IntUnOp::Cttz => {
            if bits < 32 {
                writeln!(w, "    orr {value}, {value}, #{}", 1u32 << bits)?;
            }
            writeln!(w, "    rbit {value}, {value}")?;
            writeln!(w, "    clz {value}, {value}")?;
        }
        IntUnOp::Popcount => {
            let temp = scratch.alloc()?;
            let mask = scratch.alloc()?;
            mov_imm32(w, mask, 0x5555_5555)?;
            writeln!(w, "    and {temp}, {mask}, {value}, lsr #1")?;
            writeln!(w, "    sub {value}, {value}, {temp}")?;
            mov_imm32(w, mask, 0x3333_3333)?;
            writeln!(w, "    and {temp}, {mask}, {value}, lsr #2")?;
            writeln!(w, "    and {value}, {value}, {mask}")?;
            writeln!(w, "    add {value}, {value}, {temp}")?;
            writeln!(w, "    add {value}, {value}, {value}, lsr #4")?;
            mov_imm32(w, mask, 0x0f0f_0f0f)?;
            writeln!(w, "    and {value}, {value}, {mask}")?;
            mov_imm32(w, mask, 0x0101_0101)?;
            writeln!(w, "    mul {value}, {value}, {mask}")?;
            writeln!(w, "    lsr {value}, {value}, #24")?;
            scratch.free(temp);
            scratch.free(mask);
        }
    }
    store_result(w, dst, value, ctx.frame)?;
    scratch.free(value);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_rotate<W: Write>(
    w: &mut W,
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let name = match op {
        IntRotOp::Rotl => "rotl",
        IntRotOp::Rotr => "rotr",
    };
    check_scalar(name, ty)?;
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    let value = scratch.alloc()?;
    let amount = scratch.alloc()?;
    materialize(w, lhs, value, ctx.frame)?;
    materialize(w, rhs, amount, ctx.frame)?;
    extend_integer(w, value, ty, false)?;
    writeln!(w, "    and {amount}, {amount}, #{}", bits - 1)?;
    if bits == 32 {
        if matches!(op, IntRotOp::Rotl) {
            writeln!(w, "    rsb {amount}, {amount}, #32")?;
        }
        writeln!(w, "    ror {value}, {value}, {amount}")?;
    } else {
        let complement = scratch.alloc()?;
        let upper = scratch.alloc()?;
        writeln!(w, "    rsb {complement}, {amount}, #{bits}")?;
        let (left, right) = match op {
            IntRotOp::Rotl => (amount, complement),
            IntRotOp::Rotr => (complement, amount),
        };
        writeln!(w, "    lsl {upper}, {value}, {left}")?;
        writeln!(w, "    lsr {value}, {value}, {right}")?;
        writeln!(w, "    orr {value}, {value}, {upper}")?;
        extend_integer(w, value, ty, false)?;
        scratch.free(complement);
        scratch.free(upper);
    }
    extend_integer(w, value, ty, true)?;
    store_result(w, dst, value, ctx.frame)?;
    scratch.free(value);
    scratch.free(amount);
    Ok(())
}

/// Clamps a converted value into the range of a type narrower than a word. VFP already
/// saturates to 32 bits and turns NaN into zero.
fn saturate_narrow<W: Write>(
    w: &mut W,
    value: &str,
    ty: &MirType,
    signed: bool,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    if bits >= 32 {
        return Ok(());
    }
    let bound = scratch.alloc()?;
    let (minimum, maximum) = if signed {
        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
    } else {
        (0, (1i64 << bits) - 1)
    };
    mov_imm32(w, bound, maximum as u32)?;
    writeln!(w, "    cmp {value}, {bound}")?;
    writeln!(
        w,
        "    mov{} {value}, {bound}",
        if signed { "gt" } else { "hi" }
    )?;
    mov_imm32(w, bound, minimum as u32)?;
    writeln!(w, "    cmp {value}, {bound}")?;
    writeln!(
        w,
        "    mov{} {value}, {bound}",
        if signed { "lt" } else { "lo" }
    )?;
    scratch.free(bound);
    Ok(())
}
