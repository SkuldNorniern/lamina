//! Call sites, returns and the I/O builtins for the armv7 backend.

use std::io::Write;

use crate::{
    error::LaminaError,
    mir::{Aggregate, MirType, Operand, Register, ScalarType},
    mir_codegen::common::unsupported_io,
};
use lamina_codegen::abi::Abi;
use lamina_platform::TargetOperatingSystem;

use super::abi::{ArgKind, ArgPlace, classify, returns_in_memory, stack_bytes};
use super::emit::{Ctx, POINTER_WIDTH, copy_through_pointer, is_wide_integer, reject_wide};
use super::operand::{
    ADDR, F32_SCRATCH, F64_SCRATCH, Scratch, frame_address, frame_vstore, materialize,
    materialize_float, mov_imm32, sp_adjust, store_result, unsupported,
};

/// Where the words of an argument come from once it has to move into registers or onto
/// the stack.
struct Outgoing<'a> {
    operand: &'a Operand,
    kind: ArgKind,
    place: ArgPlace,
}

fn float_register(index: usize, double: bool) -> String {
    if double {
        format!("d{}", index / 2)
    } else {
        format!("s{index}")
    }
}

/// Loads the eight bytes of a double-sized argument into `low` and `high`.
fn load_double_words<W: Write>(
    w: &mut W,
    operand: &Operand,
    low: &str,
    high: &str,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    materialize_float(w, operand, F64_SCRATCH[0], true, ctx.frame, scratch)?;
    writeln!(w, "    vmov {low}, {high}, {}", F64_SCRATCH[0])?;
    Ok(())
}

/// Writes the stack half of every argument. `base` is the register the outgoing area
/// is addressed from, with `bias` added to each offset.
fn emit_stack_arguments<W: Write>(
    w: &mut W,
    outgoing: &[Outgoing<'_>],
    base: &str,
    bias: i32,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    for argument in outgoing {
        let Some(span) = argument.place.stack else {
            continue;
        };
        let at = bias + span.offset as i32;
        let kind = &argument.kind;
        if let Some(aggregate) = kind.aggregate {
            let source = scratch.alloc()?;
            let target = scratch.alloc()?;
            materialize(w, argument.operand, source, ctx.frame)?;
            let skipped = argument.place.core.map_or(0, |core| core.words * 4);
            writeln!(w, "    add {source}, {source}, #{skipped}")?;
            if (0..=4095).contains(&at) {
                writeln!(w, "    add {target}, {base}, #{at}")?;
            } else {
                mov_imm32(w, target, at as u32)?;
                writeln!(w, "    add {target}, {base}, {target}")?;
            }
            copy_through_pointer(w, source, target, aggregate.size - skipped)?;
            scratch.free(source);
            scratch.free(target);
        } else if kind.double {
            materialize_float(
                w,
                argument.operand,
                F64_SCRATCH[0],
                true,
                ctx.frame,
                scratch,
            )?;
            if (0..=1020).contains(&at) {
                writeln!(w, "    vstr {}, [{base}, #{at}]", F64_SCRATCH[0])?;
            } else {
                mov_imm32(w, ADDR, at as u32)?;
                writeln!(w, "    add {ADDR}, {base}, {ADDR}")?;
                writeln!(w, "    vstr {}, [{ADDR}]", F64_SCRATCH[0])?;
            }
        } else if kind.float {
            materialize_float(
                w,
                argument.operand,
                F32_SCRATCH[0],
                false,
                ctx.frame,
                scratch,
            )?;
            if (0..=1020).contains(&at) {
                writeln!(w, "    vstr {}, [{base}, #{at}]", F32_SCRATCH[0])?;
            } else {
                mov_imm32(w, ADDR, at as u32)?;
                writeln!(w, "    add {ADDR}, {base}, {ADDR}")?;
                writeln!(w, "    vstr {}, [{ADDR}]", F32_SCRATCH[0])?;
            }
        } else {
            let value = scratch.alloc()?;
            materialize(w, argument.operand, value, ctx.frame)?;
            if (0..=4095).contains(&at) {
                writeln!(w, "    str {value}, [{base}, #{at}]")?;
            } else {
                mov_imm32(w, ADDR, at as u32)?;
                writeln!(w, "    add {ADDR}, {base}, {ADDR}")?;
                writeln!(w, "    str {value}, [{ADDR}]")?;
            }
            scratch.free(value);
        }
    }
    Ok(())
}

/// Fills r0-r3 and the VFP bank. Runs after the stack half so nothing it writes is
/// clobbered.
fn emit_register_arguments<W: Write>(
    w: &mut W,
    outgoing: &[Outgoing<'_>],
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    for argument in outgoing {
        let kind = &argument.kind;
        if let Some(index) = argument.place.vfp {
            let register = float_register(index, kind.double);
            materialize_float(
                w,
                argument.operand,
                &register,
                kind.double,
                ctx.frame,
                scratch,
            )?;
            continue;
        }
        let Some(core) = argument.place.core else {
            continue;
        };
        if kind.aggregate.is_some() {
            let address = scratch.alloc()?;
            materialize(w, argument.operand, address, ctx.frame)?;
            for word in 0..core.words {
                writeln!(
                    w,
                    "    ldr r{}, [{address}, #{}]",
                    core.first + word,
                    word * 4
                )?;
            }
            scratch.free(address);
        } else if kind.double {
            if core.words < 2 {
                return Err(unsupported(
                    "armv7 cannot split a double across a register and the stack".to_string(),
                ));
            }
            let low = format!("r{}", core.first);
            let high = format!("r{}", core.first + 1);
            load_double_words(w, argument.operand, &low, &high, ctx, scratch)?;
        } else {
            materialize(w, argument.operand, &format!("r{}", core.first), ctx.frame)?;
        }
    }
    Ok(())
}

struct CallPlan<'a> {
    outgoing: Vec<Outgoing<'a>>,
    space: usize,
}

fn plan_call<'a>(
    args: &'a [Operand],
    mut kinds: Vec<ArgKind>,
    mut fixed: Option<usize>,
    hidden: bool,
    base_standard: bool,
) -> Result<CallPlan<'a>, LaminaError> {
    for kind in &kinds {
        if kind.aggregate.is_none() && is_wide_integer(&kind.ty) {
            return Err(reject_wide("call argument"));
        }
    }
    if hidden {
        kinds.insert(0, ArgKind::pointer());
        fixed = fixed.map(|fixed| fixed + 1);
    }
    let places = classify(&kinds, fixed, base_standard);
    let space = stack_bytes(&places);
    let skip = usize::from(hidden);
    let outgoing = args
        .iter()
        .zip(kinds.into_iter().skip(skip))
        .zip(places.into_iter().skip(skip))
        .map(|((operand, kind), place)| Outgoing {
            operand,
            kind,
            place,
        })
        .collect();
    Ok(CallPlan { outgoing, space })
}

#[allow(clippy::too_many_arguments)]
fn store_call_result<W: Write>(
    w: &mut W,
    ret: Option<&Register>,
    result: Option<Aggregate>,
    float: Option<MirType>,
    base_standard: bool,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let Some(dst) = ret else {
        return Ok(());
    };
    if let Some(ty) = float {
        let double = ty.size_bytes(POINTER_WIDTH) == 8;
        let offset = ctx
            .frame
            .slot_of(dst)
            .ok_or_else(|| unsupported(format!("armv7 result {dst} has no frame slot")))?;
        if base_standard {
            // A runtime helper hands a float back in r0, and a double in r0:r1.
            let address = scratch.alloc()?;
            frame_address(w, address, offset)?;
            writeln!(w, "    str r0, [{address}]")?;
            if double {
                writeln!(w, "    str r1, [{address}, #4]")?;
            }
            scratch.free(address);
            return Ok(());
        }
        let register = if double { "d0" } else { "s0" };
        return frame_vstore(w, register, offset);
    }
    match result {
        Some(aggregate) if !returns_in_memory(Some(aggregate)) => {
            let address = scratch.alloc()?;
            let offset = ctx.frame.slot_of(dst).ok_or_else(|| {
                unsupported("armv7 aggregate result has no frame slot".to_string())
            })?;
            frame_address(w, address, offset)?;
            writeln!(w, "    str r0, [{address}]")?;
            if aggregate.size > 4 {
                writeln!(w, "    str r1, [{address}, #4]")?;
            }
            scratch.free(address);
        }
        Some(_) => {}
        None => store_result(w, dst, "r0", ctx.frame)?,
    }
    Ok(())
}

pub(crate) fn emit_call<W: Write>(
    w: &mut W,
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    if name.starts_with("lamina.") {
        return emit_builtin(w, name, args, ret, ctx, scratch);
    }

    let result = ctx.signatures.result(name);
    let hidden = returns_in_memory(result);
    let plan = plan_call(
        args,
        ctx.signatures.arguments(name, args, ctx.func),
        ctx.signatures.fixed_arguments(name),
        hidden,
        ctx.signatures.uses_base_standard(name),
    )?;

    sp_adjust(w, "sub", plan.space)?;
    emit_stack_arguments(w, &plan.outgoing, "sp", 0, ctx, scratch)?;
    emit_register_arguments(w, &plan.outgoing, ctx, scratch)?;
    if hidden {
        let dst = ret.ok_or_else(|| {
            unsupported("armv7 memory return needs a destination register".to_string())
        })?;
        let offset = ctx.frame.slot_of(dst).ok_or_else(|| {
            unsupported("armv7 memory return destination has no frame slot".to_string())
        })?;
        frame_address(w, "r0", offset)?;
    }

    let target = ctx
        .abi
        .call_stub(name)
        .unwrap_or_else(|| ctx.abi.mangle_function_name(name));
    writeln!(w, "    bl {target}")?;
    sp_adjust(w, "add", plan.space)?;

    store_call_result(
        w,
        ret,
        result,
        ctx.signatures.float_result(name),
        ctx.signatures.uses_base_standard(name),
        ctx,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_call_indirect<W: Write>(
    w: &mut W,
    target: &Operand,
    arg_types: &[MirType],
    args: &[Operand],
    variadic_fixed: Option<usize>,
    ret: Option<&Register>,
    ret_ty: Option<MirType>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    // A legalized pair the called pointer's signature does not type reaches the
    // callee as the aggregate it was spilled to, not as its address.
    let kinds: Vec<ArgKind> = arg_types
        .iter()
        .zip(args)
        .map(|(ty, arg)| match ctx.func.variadic_wide_argument(arg) {
            Some(aggregate) => ArgKind::of(MirType::Scalar(ScalarType::Ptr), Some(aggregate)),
            None => ArgKind::of(*ty, None),
        })
        .collect();
    let plan = plan_call(args, kinds, variadic_fixed, false, false)?;

    let callee = scratch.alloc()?;
    materialize(w, target, callee, ctx.frame)?;
    sp_adjust(w, "sub", plan.space)?;
    emit_stack_arguments(w, &plan.outgoing, "sp", 0, ctx, scratch)?;
    emit_register_arguments(w, &plan.outgoing, ctx, scratch)?;
    writeln!(w, "    blx {callee}")?;
    sp_adjust(w, "add", plan.space)?;
    scratch.free(callee);

    let float = ret_ty.filter(|ty| ty.is_float());
    store_call_result(w, ret, None, float, false, ctx, scratch)
}

pub(crate) fn emit_tail_call<W: Write>(
    w: &mut W,
    name: &str,
    args: &[Operand],
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let result = ctx.signatures.result(name);
    let hidden = returns_in_memory(result);
    let plan = plan_call(
        args,
        ctx.signatures.arguments(name, args, ctx.func),
        ctx.signatures.fixed_arguments(name),
        hidden,
        ctx.signatures.uses_base_standard(name),
    )?;
    if plan.space > 0 || hidden {
        // The callee's outgoing area would overwrite this frame's incoming arguments,
        // so the call is made the ordinary way and its result returned.
        emit_call(w, name, args, None, ctx, scratch)?;
        writeln!(w, "    b {}", ctx.epilogue)?;
        return Ok(());
    }
    emit_register_arguments(w, &plan.outgoing, ctx, scratch)?;
    let target = ctx
        .abi
        .call_stub(name)
        .unwrap_or_else(|| ctx.abi.mangle_function_name(name));
    super::emit_restore(w, ctx.func.sig.variadic)?;
    writeln!(w, "    b {target}")?;
    Ok(())
}

pub(crate) fn emit_return<W: Write>(
    w: &mut W,
    value: Option<&Operand>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let Some(value) = value else {
        writeln!(w, "    b {}", ctx.epilogue)?;
        return Ok(());
    };

    if ctx.ret_aggregate.is_none()
        && let Some(ty) = ctx.ret_float
    {
        let double = ty.size_bytes(POINTER_WIDTH) == 8;
        let register = if double { "d0" } else { "s0" };
        materialize_float(w, value, register, double, ctx.frame, scratch)?;
        writeln!(w, "    b {}", ctx.epilogue)?;
        return Ok(());
    }

    match ctx.ret_aggregate {
        Some(aggregate) => {
            let address = scratch.alloc()?;
            materialize(w, value, address, ctx.frame)?;
            match ctx.sret_slot {
                Some(offset) => {
                    let buffer = scratch.alloc()?;
                    super::operand::frame_load(w, buffer, offset)?;
                    copy_through_pointer(w, address, buffer, aggregate.size)?;
                    writeln!(w, "    mov r0, {buffer}")?;
                    scratch.free(buffer);
                }
                None => {
                    if aggregate.size > 4 {
                        writeln!(w, "    ldr r1, [{address}, #4]")?;
                    }
                    writeln!(w, "    ldr r0, [{address}]")?;
                }
            }
            scratch.free(address);
        }
        None => materialize(w, value, "r0", ctx.frame)?,
    }
    writeln!(w, "    b {}", ctx.epilogue)?;
    Ok(())
}

/// Maps the raw result of a read or write to the I/O contract: a failure becomes -2,
/// and with `end_of_input` a zero count becomes -1.
fn emit_io_result<W: Write>(w: &mut W, end_of_input: bool) -> Result<(), LaminaError> {
    writeln!(w, "    cmp r0, #0")?;
    writeln!(w, "    mvnlt r0, #1")?;
    if end_of_input {
        writeln!(w, "    cmp r0, #0")?;
        writeln!(w, "    mvneq r0, #0")?;
    }
    Ok(())
}

fn emit_builtin<W: Write>(
    w: &mut W,
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    if ctx.abi.target_os() != TargetOperatingSystem::Linux {
        return Err(unsupported_io(name, ctx.abi.target_os()));
    }
    match (name, args.len()) {
        ("lamina.writebyte", 1) => {
            let value = scratch.alloc()?;
            materialize(w, &args[0], value, ctx.frame)?;
            writeln!(w, "    sub sp, sp, #8")?;
            writeln!(w, "    strb {value}, [sp]")?;
            writeln!(w, "    mov r0, #1")?;
            writeln!(w, "    mov r1, sp")?;
            writeln!(w, "    mov r2, #1")?;
            writeln!(w, "    bl write")?;
            emit_io_result(w, false)?;
            writeln!(w, "    add sp, sp, #8")?;
            scratch.free(value);
            if let Some(dst) = ret {
                store_result(w, dst, "r0", ctx.frame)?;
            }
        }
        ("lamina.readbyte", 0) => {
            writeln!(w, "    sub sp, sp, #8")?;
            writeln!(w, "    mov r0, #0")?;
            writeln!(w, "    mov r1, sp")?;
            writeln!(w, "    mov r2, #1")?;
            writeln!(w, "    bl read")?;
            if let Some(dst) = ret {
                let value = scratch.alloc()?;
                writeln!(w, "    ldrb {value}, [sp]")?;
                writeln!(w, "    subs r0, r0, #1")?;
                writeln!(w, "    moveq r0, {value}")?;
                store_result(w, dst, "r0", ctx.frame)?;
                scratch.free(value);
            }
            writeln!(w, "    add sp, sp, #8")?;
        }
        ("lamina.write", 2) => {
            materialize(w, &args[0], "r1", ctx.frame)?;
            materialize(w, &args[1], "r2", ctx.frame)?;
            writeln!(w, "    mov r0, #1")?;
            writeln!(w, "    bl write")?;
            emit_io_result(w, false)?;
            if let Some(dst) = ret {
                store_result(w, dst, "r0", ctx.frame)?;
            }
        }
        ("lamina.read", 2) => {
            materialize(w, &args[0], "r1", ctx.frame)?;
            materialize(w, &args[1], "r2", ctx.frame)?;
            writeln!(w, "    mov r0, #0")?;
            writeln!(w, "    cmp r2, #0")?;
            writeln!(w, "    beq 3f")?;
            writeln!(w, "    bl read")?;
            emit_io_result(w, true)?;
            writeln!(w, "3:")?;
            if let Some(dst) = ret {
                store_result(w, dst, "r0", ctx.frame)?;
            }
        }
        _ => {
            return Err(unsupported(format!(
                "armv7 backend does not implement the builtin `{name}`"
            )));
        }
    }
    Ok(())
}
