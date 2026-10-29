//! Instruction lowering for the Cortex-M backend.

use std::io::Write;

use crate::mir_codegen::common::{unsupported_overflow_flag, unsupported_thread_local};
use crate::{
    error::LaminaError,
    mir::{
        Aggregate, Instruction as MirInst, IntBinOp, IntRotOp, IntUnOp, MemoryAttrs, MirType,
        Operand, Register, ScalarType,
    },
    mir_codegen::{
        MirCodegenSettings,
        arm::int_condition,
        common::{emit_source_loc, integer_conversion_with_width},
    },
};
use lamina_mir::SourceLoc;
use lamina_platform::PointerWidth;

use super::abi::CortexMAbi;
use super::call;
use super::isa::{
    Alu, Isa, alu, and_imm, clear_before_compare, condition_move, condition_set, mov_imm,
    reverse_subtract,
};
use super::operand::{Access, Scratch, extend_integer, unsupported};
use crate::mir_codegen::arm::armv7::abi::Signatures;

pub(crate) const POINTER_WIDTH: PointerWidth = PointerWidth::Bits32;

pub(crate) struct Ctx<'a> {
    pub(crate) isa: Isa,
    pub(crate) access: Access<'a>,
    pub(crate) abi: &'a CortexMAbi,
    pub(crate) signatures: &'a Signatures,
    pub(crate) epilogue: &'a str,
    pub(crate) block_prefix: &'a str,
    pub(crate) ret_aggregate: Option<Aggregate>,
    pub(crate) sret_slot: Option<i32>,
    pub(crate) settings: &'a MirCodegenSettings,
}

pub(crate) fn is_wide_integer(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::I64))
}

/// The one message the backend gives for an operation the i64 legalization pass has
/// not split into register pairs. It names the operation, so a program that reaches
/// here fails rather than getting a truncated answer.
pub(crate) fn reject_wide(isa: Isa, operation: &str) -> LaminaError {
    unsupported(format!(
        "{} has no 64-bit integer registers, so `{operation}` on i64 cannot be lowered until \
         LUMIR legalizes 64-bit integers into 32-bit register pairs; narrow the value to i32",
        isa.target_name()
    ))
}

/// Cortex-M floating point: the M0 and M0+ have no FPU at all, and the single-precision
/// VFP an M4F or M7 may carry is not used by this backend. Refused by name rather than
/// lowered to soft-float helpers that nothing here ships.
pub(crate) fn reject_float(isa: Isa, operation: &str, ty: &MirType) -> LaminaError {
    let detail = if matches!(ty, MirType::Scalar(ScalarType::F64)) {
        "no Cortex-M FPU is double precision"
    } else {
        "this backend emits no VFP instructions and no soft-float library calls"
    };
    unsupported(format!(
        "{} does not implement floating point: `{operation}` on {ty} ({detail})",
        isa.target_name()
    ))
}

/// ARMv6-M has no unaligned load or store: an `ldr` or `ldrh` whose address is not a
/// multiple of its size takes a HardFault. ARMv7-M permits both. Refused by name
/// rather than lowered to a byte-at-a-time sequence, which this backend does not emit.
fn check_alignment(
    isa: Isa,
    operation: &str,
    ty: &MirType,
    attrs: &MemoryAttrs,
) -> Result<(), LaminaError> {
    let size = ty.size_bytes(POINTER_WIDTH);
    if isa == Isa::Thumb1 && usize::from(attrs.align) < size {
        return Err(unsupported(format!(
            "thumbv6m takes a HardFault on an unaligned access, and this `{operation}` of {ty} \
             declares {}-byte alignment for a {size}-byte value",
            attrs.align
        )));
    }
    Ok(())
}

fn check_scalar(isa: Isa, operation: &str, ty: &MirType) -> Result<(), LaminaError> {
    if is_wide_integer(ty) {
        return Err(reject_wide(isa, operation));
    }
    if ty.is_float() {
        return Err(reject_float(isa, operation, ty));
    }
    if ty.is_vector() {
        return Err(unsupported(format!(
            "{} has no vector unit: `{operation}` on {ty}",
            isa.target_name()
        )));
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
    let isa = ctx.isa;
    match inst {
        MirInst::Comment { text } => writeln!(w, "    @ {text}")?,
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {}
        MirInst::Unreachable => writeln!(w, "    udf #0")?,
        MirInst::ThreadLocalAddr { .. } => {
            return Err(unsupported_thread_local(isa.target_name()));
        }
        MirInst::IntOverflow { .. } => {
            return Err(unsupported_overflow_flag(isa.target_name()));
        }

        MirInst::SymbolAddress { dst, symbol } => {
            let label = ctx.abi.mangle(symbol);
            let value = scratch.alloc()?;
            match isa {
                Isa::Thumb2 => {
                    writeln!(w, "    movw {value}, #:lower16:{label}")?;
                    writeln!(w, "    movt {value}, #:upper16:{label}")?;
                }
                // Thumb-1 has no `movw`, so the address comes from a literal pool the
                // assembler places at the next `.ltorg`, which the block loop emits
                // after every block that ends in an unconditional branch.
                Isa::Thumb1 => writeln!(w, "    ldr {value}, ={label}")?,
            }
            ctx.access.store_result(w, dst, value, scratch)?;
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

        MirInst::FloatBinary { ty, .. }
        | MirInst::FloatUnary { ty, .. }
        | MirInst::FloatCmp { ty, .. } => return Err(reject_float(isa, "a float operation", ty)),
        MirInst::FloatConvert { to, .. } => return Err(reject_float(isa, "fptrunc/fpext", to)),
        MirInst::IntToFloat { to, .. } => return Err(reject_float(isa, "sitofp/uitofp", to)),
        MirInst::FloatToInt { from, .. } => return Err(reject_float(isa, "fptosi/fptoui", from)),

        MirInst::IntCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            check_scalar(isa, "icmp", ty)?;
            let cond = int_condition(*op);
            let signed = !matches!(cond, "lo" | "ls" | "hi" | "hs");
            let a = scratch.alloc()?;
            let b = scratch.alloc()?;
            let value = scratch.alloc()?;
            ctx.access.materialize(w, lhs, a)?;
            ctx.access.materialize(w, rhs, b)?;
            extend_integer(w, a, ty, signed)?;
            extend_integer(w, b, ty, signed)?;
            clear_before_compare(w, value)?;
            writeln!(w, "    cmp {a}, {b}")?;
            condition_set(w, isa, value, cond)?;
            ctx.access.store_result(w, dst, value, scratch)?;
            scratch.free(a);
            scratch.free(b);
            scratch.free(value);
        }

        MirInst::Select {
            ty,
            dst,
            cond,
            true_val,
            false_val,
        } => {
            check_scalar(isa, "select", ty)?;
            let flag = scratch.alloc()?;
            let a = scratch.alloc()?;
            let b = scratch.alloc()?;
            ctx.access.load_reg(w, cond, flag)?;
            ctx.access.materialize(w, true_val, a)?;
            ctx.access.materialize(w, false_val, b)?;
            writeln!(w, "    cmp {flag}, #0")?;
            condition_move(w, isa, a, b, "eq")?;
            ctx.access.store_result(w, dst, a, scratch)?;
            scratch.free(flag);
            scratch.free(a);
            scratch.free(b);
        }

        MirInst::Load {
            ty,
            dst,
            addr,
            attrs,
        } => {
            check_scalar(isa, "load", ty)?;
            check_alignment(isa, "load", ty, attrs)?;
            let address = scratch.alloc()?;
            ctx.access.materialize_address(w, addr, address, scratch)?;
            let value = scratch.alloc()?;
            // Thumb-1 encodes a signed load only in its register-offset form, so a
            // narrow load is widened afterwards on both encodings.
            match ty.size_bytes(POINTER_WIDTH) {
                1 => {
                    writeln!(w, "    ldrb {value}, [{address}]")?;
                    writeln!(w, "    sxtb {value}, {value}")?;
                }
                2 => {
                    writeln!(w, "    ldrh {value}, [{address}]")?;
                    writeln!(w, "    sxth {value}, {value}")?;
                }
                4 => writeln!(w, "    ldr {value}, [{address}]")?,
                other => {
                    return Err(unsupported(format!(
                        "{} load of {other} bytes",
                        isa.target_name()
                    )));
                }
            }
            ctx.access.store_result(w, dst, value, scratch)?;
            scratch.free(value);
            scratch.free(address);
        }

        MirInst::Store {
            ty,
            src,
            addr,
            attrs,
        } => {
            check_scalar(isa, "store", ty)?;
            check_alignment(isa, "store", ty, attrs)?;
            let address = scratch.alloc()?;
            ctx.access.materialize_address(w, addr, address, scratch)?;
            let value = scratch.alloc()?;
            ctx.access.materialize(w, src, value)?;
            match ty.size_bytes(POINTER_WIDTH) {
                1 => writeln!(w, "    strb {value}, [{address}]")?,
                2 => writeln!(w, "    strh {value}, [{address}]")?,
                4 => writeln!(w, "    str {value}, [{address}]")?,
                other => {
                    return Err(unsupported(format!(
                        "{} store of {other} bytes",
                        isa.target_name()
                    )));
                }
            }
            scratch.free(value);
            scratch.free(address);
        }

        MirInst::StackAddr { dst, slot } => {
            let offset = ctx
                .access
                .slot_of(&Register::Virtual(*slot))
                .ok_or_else(|| {
                    unsupported("cortex-m stack address has no frame slot".to_string())
                })?;
            let value = scratch.alloc()?;
            ctx.access.frame_address(w, value, offset)?;
            ctx.access.store_result(w, dst, value, scratch)?;
            scratch.free(value);
        }

        MirInst::Lea { dst, base, offset } => {
            let value = scratch.alloc()?;
            ctx.access.load_reg(w, base, value)?;
            ctx.access
                .add_offset(w, value, i64::from(*offset), scratch)?;
            ctx.access.store_result(w, dst, value, scratch)?;
            scratch.free(value);
        }

        MirInst::Jmp { target } => writeln!(w, "    b {}{target}", ctx.block_prefix)?,

        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            let flag = scratch.alloc()?;
            ctx.access.load_reg(w, cond, flag)?;
            writeln!(w, "    cmp {flag}, #0")?;
            // The conditional branch skips one unconditional branch, which keeps it
            // inside the +-256 byte reach Thumb-1 gives it whatever the block's size.
            writeln!(w, "    beq 8f")?;
            writeln!(w, "    b {}{true_target}", ctx.block_prefix)?;
            writeln!(w, "8:")?;
            writeln!(w, "    b {}{false_target}", ctx.block_prefix)?;
            scratch.free(flag);
        }

        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            let subject = scratch.alloc()?;
            ctx.access.load_reg(w, value, subject)?;
            for (constant, target) in cases {
                let probe = scratch.alloc()?;
                mov_imm(w, isa, probe, *constant as u32)?;
                writeln!(w, "    cmp {subject}, {probe}")?;
                writeln!(w, "    bne 8f")?;
                writeln!(w, "    b {}{target}", ctx.block_prefix)?;
                writeln!(w, "8:")?;
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
            ..
        } => call::emit_call_indirect(
            w,
            target,
            arg_types,
            args,
            *variadic_fixed,
            ret.as_ref(),
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
            return Err(unsupported(format!(
                "{} backend does not implement atomic operations",
                isa.target_name()
            )));
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported(format!(
                "{} backend does not implement vector operations",
                isa.target_name()
            )));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported(format!(
                "{} backend does not implement inline assembly",
                isa.target_name()
            )));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(unsupported(format!(
                "{} backend does not implement variadic argument reading",
                isa.target_name()
            )));
        }
    }
    Ok(())
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
    check_scalar(ctx.isa, operation, ty)?;
    if ty.size_bytes(POINTER_WIDTH) == 8 {
        return Err(reject_wide(ctx.isa, operation));
    }
    let value = scratch.alloc()?;
    ctx.access.materialize(w, src, value)?;
    ctx.access.store_result(w, dst, value, scratch)?;
    scratch.free(value);
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
    let isa = ctx.isa;
    check_scalar(isa, int_binary_name(op), ty)?;
    let division = matches!(
        op,
        IntBinOp::UDiv | IntBinOp::SDiv | IntBinOp::URem | IntBinOp::SRem
    );
    if division && !isa.has_hardware_divide() {
        return Err(unsupported(format!(
            "thumbv6m has no divide instruction, and this backend does not call the \
             `__aeabi_idiv` runtime it does not ship: `{}` is not supported",
            int_binary_name(op)
        )));
    }
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    let a = scratch.alloc()?;
    let b = scratch.alloc()?;
    let d = scratch.alloc()?;
    ctx.access.materialize(w, lhs, a)?;
    ctx.access.materialize(w, rhs, b)?;

    if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
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
        mov_imm(
            w,
            isa,
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
        and_imm(w, isa, b, (bits - 1) as u32, d)?;
    }

    match op {
        IntBinOp::Add => alu(w, isa, Alu::Add, d, a, b)?,
        IntBinOp::Sub => alu(w, isa, Alu::Sub, d, a, b)?,
        IntBinOp::Mul => alu(w, isa, Alu::Mul, d, a, b)?,
        IntBinOp::And => alu(w, isa, Alu::And, d, a, b)?,
        IntBinOp::Or => alu(w, isa, Alu::Or, d, a, b)?,
        IntBinOp::Xor => alu(w, isa, Alu::Xor, d, a, b)?,
        IntBinOp::Shl => alu(w, isa, Alu::Shl, d, a, b)?,
        IntBinOp::LShr => alu(w, isa, Alu::LShr, d, a, b)?,
        IntBinOp::AShr => alu(w, isa, Alu::AShr, d, a, b)?,
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
    }
    extend_integer(w, d, ty, true)?;
    ctx.access.store_result(w, dst, d, scratch)?;
    scratch.free(a);
    scratch.free(b);
    scratch.free(d);
    Ok(())
}

/// Population count of `value`, using `temp` and `mask` as working registers.
fn popcount<W: Write>(
    w: &mut W,
    isa: Isa,
    value: &str,
    temp: &str,
    mask: &str,
) -> Result<(), LaminaError> {
    mov_imm(w, isa, mask, 0x5555_5555)?;
    writeln!(w, "    lsrs {temp}, {value}, #1")?;
    writeln!(w, "    ands {temp}, {mask}")?;
    writeln!(w, "    subs {value}, {value}, {temp}")?;
    mov_imm(w, isa, mask, 0x3333_3333)?;
    writeln!(w, "    lsrs {temp}, {value}, #2")?;
    writeln!(w, "    ands {temp}, {mask}")?;
    writeln!(w, "    ands {value}, {mask}")?;
    writeln!(w, "    adds {value}, {value}, {temp}")?;
    writeln!(w, "    lsrs {temp}, {value}, #4")?;
    writeln!(w, "    adds {value}, {value}, {temp}")?;
    mov_imm(w, isa, mask, 0x0f0f_0f0f)?;
    writeln!(w, "    ands {value}, {mask}")?;
    mov_imm(w, isa, mask, 0x0101_0101)?;
    writeln!(w, "    muls {value}, {mask}, {value}")?;
    writeln!(w, "    lsrs {value}, {value}, #24")?;
    Ok(())
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
    let isa = ctx.isa;
    let name = match op {
        IntUnOp::Popcount => "popcount",
        IntUnOp::Ctlz => "ctlz",
        IntUnOp::Cttz => "cttz",
        IntUnOp::Bswap => "bswap",
    };
    check_scalar(isa, name, ty)?;
    let bits = ty.size_bytes(POINTER_WIDTH) * 8;
    let value = scratch.alloc()?;
    ctx.access.materialize(w, src, value)?;
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
        IntUnOp::Ctlz if isa == Isa::Thumb2 => {
            writeln!(w, "    clz {value}, {value}")?;
            if bits < 32 {
                writeln!(w, "    sub {value}, {value}, #{}", 32 - bits)?;
            }
        }
        IntUnOp::Cttz if isa == Isa::Thumb2 => {
            if bits < 32 {
                let bit = scratch.alloc()?;
                mov_imm(w, isa, bit, 1u32 << bits)?;
                alu(w, isa, Alu::Or, value, value, bit)?;
                scratch.free(bit);
            }
            writeln!(w, "    rbit {value}, {value}")?;
            writeln!(w, "    clz {value}, {value}")?;
        }
        // ARMv6-M has neither `clz` nor `rbit`. Smearing the value right turns the
        // leading-zero count into a population count, and isolating the lowest set bit
        // turns the trailing-zero count into one. A zero input smears to zero, whose
        // complement has 32 bits set, and isolates to zero, whose predecessor has 32
        // bits set, so both give the width without a special case.
        IntUnOp::Ctlz => {
            let temp = scratch.alloc()?;
            let mask = scratch.alloc()?;
            for shift in [1, 2, 4, 8, 16] {
                writeln!(w, "    lsrs {temp}, {value}, #{shift}")?;
                writeln!(w, "    orrs {value}, {temp}")?;
            }
            writeln!(w, "    mvns {value}, {value}")?;
            popcount(w, isa, value, temp, mask)?;
            if bits < 32 {
                writeln!(w, "    subs {value}, #{}", 32 - bits)?;
            }
            scratch.free(temp);
            scratch.free(mask);
        }
        IntUnOp::Cttz => {
            let temp = scratch.alloc()?;
            let mask = scratch.alloc()?;
            if bits < 32 {
                mov_imm(w, isa, temp, 1u32 << bits)?;
                writeln!(w, "    orrs {value}, {temp}")?;
            }
            writeln!(w, "    negs {temp}, {value}")?;
            writeln!(w, "    ands {value}, {temp}")?;
            writeln!(w, "    subs {value}, #1")?;
            popcount(w, isa, value, temp, mask)?;
            scratch.free(temp);
            scratch.free(mask);
        }
        IntUnOp::Popcount => {
            let temp = scratch.alloc()?;
            let mask = scratch.alloc()?;
            popcount(w, isa, value, temp, mask)?;
            scratch.free(temp);
            scratch.free(mask);
        }
    }
    ctx.access.store_result(w, dst, value, scratch)?;
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
    let isa = ctx.isa;
    let name = match op {
        IntRotOp::Rotl => "rotl",
        IntRotOp::Rotr => "rotr",
    };
    check_scalar(isa, name, ty)?;
    let bits = (ty.size_bytes(POINTER_WIDTH) * 8) as u32;
    let value = scratch.alloc()?;
    let amount = scratch.alloc()?;
    let other = scratch.alloc()?;
    ctx.access.materialize(w, lhs, value)?;
    ctx.access.materialize(w, rhs, amount)?;
    extend_integer(w, value, ty, false)?;
    and_imm(w, isa, amount, bits - 1, other)?;
    reverse_subtract(w, isa, other, amount, bits)?;
    let (left, right) = match op {
        IntRotOp::Rotl => (amount, other),
        IntRotOp::Rotr => (other, amount),
    };
    // A rotate by zero would shift the other half by the full width, which Thumb
    // defines as producing zero rather than the value back, so the halves are taken
    // apart with the shift amounts that cannot reach the width.
    let upper = scratch.alloc()?;
    writeln!(w, "    mov {upper}, {value}")?;
    writeln!(w, "    lsls {upper}, {left}")?;
    writeln!(w, "    lsrs {value}, {right}")?;
    writeln!(w, "    orrs {value}, {upper}")?;
    scratch.free(upper);
    if bits < 32 {
        extend_integer(w, value, ty, false)?;
    }
    extend_integer(w, value, ty, true)?;
    ctx.access.store_result(w, dst, value, scratch)?;
    scratch.free(value);
    scratch.free(amount);
    scratch.free(other);
    Ok(())
}
