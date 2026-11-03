//! Call sites, returns and the I/O builtins for the Cortex-M backend.
//!
//! The outgoing argument area is reserved once by the prologue, so no call site moves
//! the stack pointer and every frame slot keeps one fixed offset from it.

use std::io::Write;

use crate::{
    error::LaminaError,
    mir::{Aggregate, Function, Instruction as MirInst, MirType, Operand, Register},
    mir_codegen::common::unsupported_io,
};
use lamina_codegen::abi::Abi;
use lamina_platform::TargetOperatingSystem;

use super::emit::{Ctx, is_wide_integer, reject_wide};
use super::isa::mov_imm;
use super::operand::{Scratch, copy_through_pointer, unsupported};
use crate::mir_codegen::arm::armv7::abi::{
    ArgKind, ArgPlace, Signatures, classify, returns_in_memory, stack_bytes,
};

/// Bytes the I/O builtins borrow at the bottom of the outgoing area for a one-byte
/// buffer, kept eight-byte aligned.
const IO_SCRATCH_BYTES: usize = 8;

struct Outgoing<'a> {
    operand: &'a Operand,
    kind: ArgKind,
    place: ArgPlace,
}

struct CallPlan<'a> {
    outgoing: Vec<Outgoing<'a>>,
    space: usize,
}

fn plan_call<'a>(
    isa: super::isa::Isa,
    args: &'a [Operand],
    mut kinds: Vec<ArgKind>,
    mut fixed: Option<usize>,
    hidden: bool,
) -> Result<CallPlan<'a>, LaminaError> {
    for kind in &kinds {
        if kind.aggregate.is_none() && is_wide_integer(&kind.ty) {
            return Err(reject_wide(isa, "call argument"));
        }
        if kind.float || kind.double {
            return Err(super::emit::reject_float(isa, "a call argument", &kind.ty));
        }
    }
    if hidden {
        kinds.insert(0, ArgKind::pointer());
        fixed = fixed.map(|fixed| fixed + 1);
    }
    // Cortex-M has no co-processor registers to place an argument in, so every call
    // takes the AAPCS base standard.
    let places = classify(&kinds, fixed, true);
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

/// Bytes the prologue reserves at `[sp, #0]`: the widest outgoing argument area any
/// call in the function needs, and the buffer the I/O builtins write a byte through.
pub(crate) fn outgoing_space(
    isa: super::isa::Isa,
    func: &Function,
    signatures: &Signatures,
) -> usize {
    let mut space = 0usize;
    for block in &func.blocks {
        for inst in &block.instructions {
            let needed = match &inst.instr {
                MirInst::Call { name, args, .. } | MirInst::TailCall { name, args } => {
                    if name.starts_with("lamina.") {
                        IO_SCRATCH_BYTES
                    } else {
                        let hidden = returns_in_memory(signatures.result(name));
                        plan_call(
                            isa,
                            args,
                            signatures.arguments(name, args, func),
                            signatures.fixed_arguments(name),
                            hidden,
                        )
                        .map_or(0, |plan| plan.space)
                    }
                }
                MirInst::CallIndirect {
                    arg_types,
                    args,
                    variadic_fixed,
                    ..
                } => {
                    let kinds = arg_types.iter().map(|ty| ArgKind::of(*ty, None)).collect();
                    plan_call(isa, args, kinds, *variadic_fixed, false).map_or(0, |plan| plan.space)
                }
                _ => 0,
            };
            space = space.max(needed);
        }
    }
    space
}

/// Writes the stack half of every argument into the reserved outgoing area.
fn emit_stack_arguments<W: Write>(
    w: &mut W,
    outgoing: &[Outgoing<'_>],
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    for argument in outgoing {
        let Some(span) = argument.place.stack else {
            continue;
        };
        let at = span.offset as i32;
        if let Some(aggregate) = argument.kind.aggregate {
            let source = scratch.alloc()?;
            let target = scratch.alloc()?;
            ctx.access.materialize(w, argument.operand, source)?;
            let skipped = argument.place.core.map_or(0, |core| core.words * 4);
            ctx.access.add_offset(w, source, skipped as i64, scratch)?;
            ctx.access.frame_address(w, target, at)?;
            copy_through_pointer(
                w,
                ctx.isa,
                source,
                target,
                aggregate.size - skipped,
                scratch,
            )?;
            scratch.free(source);
            scratch.free(target);
        } else {
            let value = scratch.alloc()?;
            ctx.access.materialize(w, argument.operand, value)?;
            super::operand::frame_store(w, ctx.isa, value, at, scratch)?;
            scratch.free(value);
        }
    }
    Ok(())
}

/// Fills r0-r3. Runs after the stack half, so nothing it writes is clobbered.
fn emit_register_arguments<W: Write>(
    w: &mut W,
    outgoing: &[Outgoing<'_>],
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    for argument in outgoing {
        let Some(core) = argument.place.core else {
            continue;
        };
        if argument.kind.aggregate.is_some() {
            let address = scratch.alloc()?;
            ctx.access.materialize(w, argument.operand, address)?;
            for word in 0..core.words {
                writeln!(
                    w,
                    "    ldr r{}, [{address}, #{}]",
                    core.first + word,
                    word * 4
                )?;
            }
            scratch.free(address);
        } else {
            ctx.access
                .materialize(w, argument.operand, &format!("r{}", core.first))?;
        }
    }
    Ok(())
}

fn store_call_result<W: Write>(
    w: &mut W,
    ret: Option<&Register>,
    result: Option<Aggregate>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let Some(dst) = ret else {
        return Ok(());
    };
    match result {
        Some(aggregate) if !returns_in_memory(Some(aggregate)) => {
            let address = scratch.alloc()?;
            let offset = ctx.access.slot_of(dst).ok_or_else(|| {
                unsupported("cortex-m aggregate result has no frame slot".to_string())
            })?;
            ctx.access.frame_address(w, address, offset)?;
            writeln!(w, "    str r0, [{address}]")?;
            if aggregate.size > 4 {
                writeln!(w, "    str r1, [{address}, #4]")?;
            }
            scratch.free(address);
        }
        Some(_) => {}
        None => ctx.access.store_result(w, dst, "r0", scratch)?,
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
        ctx.isa,
        args,
        ctx.signatures.arguments(name, args, ctx.func),
        ctx.signatures.fixed_arguments(name),
        hidden,
    )?;
    if plan.space as i32 > ctx.access.layout.outgoing {
        return Err(unsupported(format!(
            "cortex-m call to @{name} needs {} bytes of outgoing arguments and the frame \
             reserved {}",
            plan.space, ctx.access.layout.outgoing
        )));
    }

    emit_stack_arguments(w, &plan.outgoing, ctx, scratch)?;
    emit_register_arguments(w, &plan.outgoing, ctx, scratch)?;
    if hidden {
        let dst = ret.ok_or_else(|| {
            unsupported("cortex-m memory return needs a destination register".to_string())
        })?;
        let offset = ctx.access.slot_of(dst).ok_or_else(|| {
            unsupported("cortex-m memory return destination has no frame slot".to_string())
        })?;
        ctx.access.frame_address(w, "r0", offset)?;
    }

    let target = ctx.abi.call_target(name);
    writeln!(w, "    bl {target}")?;
    store_call_result(w, ret, result, ctx, scratch)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_call_indirect<W: Write>(
    w: &mut W,
    target: &Operand,
    arg_types: &[MirType],
    args: &[Operand],
    variadic_fixed: Option<usize>,
    ret: Option<&Register>,
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let kinds: Vec<ArgKind> = arg_types.iter().map(|ty| ArgKind::of(*ty, None)).collect();
    let plan = plan_call(ctx.isa, args, kinds, variadic_fixed, false)?;
    if plan.space as i32 > ctx.access.layout.outgoing {
        return Err(unsupported(
            "cortex-m indirect call needs more outgoing argument space than the frame reserved"
                .to_string(),
        ));
    }

    let callee = scratch.alloc()?;
    ctx.access.materialize(w, target, callee)?;
    emit_stack_arguments(w, &plan.outgoing, ctx, scratch)?;
    emit_register_arguments(w, &plan.outgoing, ctx, scratch)?;
    writeln!(w, "    blx {callee}")?;
    scratch.free(callee);

    store_call_result(w, ret, None, ctx, scratch)
}

/// A tail call is made as an ordinary call whose result is returned. Thumb-1 `pop`
/// cannot restore `lr`, only `pc`, so the frame cannot be torn down before a jump.
pub(crate) fn emit_tail_call<W: Write>(
    w: &mut W,
    name: &str,
    args: &[Operand],
    ctx: &Ctx<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    emit_call(w, name, args, None, ctx, scratch)?;
    writeln!(w, "    b {}", ctx.epilogue)?;
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

    match ctx.ret_aggregate {
        Some(aggregate) => {
            let address = scratch.alloc()?;
            ctx.access.materialize(w, value, address)?;
            match ctx.sret_slot {
                Some(offset) => {
                    let buffer = scratch.alloc()?;
                    super::operand::frame_load(w, ctx.isa, buffer, offset)?;
                    copy_through_pointer(w, ctx.isa, address, buffer, aggregate.size, scratch)?;
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
        None => ctx.access.materialize(w, value, "r0")?,
    }
    writeln!(w, "    b {}", ctx.epilogue)?;
    Ok(())
}

/// Maps the raw result of a read or write to the I/O contract: a failure becomes -2,
/// and with `end_of_input` a zero count becomes -1.
fn emit_io_result<W: Write>(
    w: &mut W,
    ctx: &Ctx<'_>,
    end_of_input: bool,
) -> Result<(), LaminaError> {
    let isa = ctx.isa;
    writeln!(w, "    cmp r0, #0")?;
    writeln!(w, "    bge 4f")?;
    mov_imm(w, isa, "r0", (-2i32) as u32)?;
    writeln!(w, "4:")?;
    if end_of_input {
        writeln!(w, "    cmp r0, #0")?;
        writeln!(w, "    bne 5f")?;
        mov_imm(w, isa, "r0", (-1i32) as u32)?;
        writeln!(w, "5:")?;
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
    if !matches!(
        ctx.abi.target_os(),
        TargetOperatingSystem::Linux | TargetOperatingSystem::Unknown
    ) {
        return Err(unsupported_io(name, ctx.abi.target_os()));
    }
    let isa = ctx.isa;
    match (name, args.len()) {
        // Thumb-1 has no byte load or store against the stack pointer, so the buffer
        // address goes through a callee-saved register that also survives the call.
        ("lamina.writebyte", 1) => {
            let buffer = scratch.alloc()?;
            let value = scratch.alloc()?;
            ctx.access.materialize(w, &args[0], value)?;
            writeln!(w, "    mov {buffer}, sp")?;
            writeln!(w, "    strb {value}, [{buffer}]")?;
            scratch.free(value);
            writeln!(w, "    movs r0, #1")?;
            writeln!(w, "    mov r1, {buffer}")?;
            writeln!(w, "    movs r2, #1")?;
            writeln!(w, "    bl write")?;
            scratch.free(buffer);
            emit_io_result(w, ctx, false)?;
            if let Some(dst) = ret {
                ctx.access.store_result(w, dst, "r0", scratch)?;
            }
        }
        ("lamina.readbyte", 0) => {
            let buffer = scratch.alloc()?;
            writeln!(w, "    mov {buffer}, sp")?;
            writeln!(w, "    movs r0, #0")?;
            writeln!(w, "    mov r1, {buffer}")?;
            writeln!(w, "    movs r2, #1")?;
            writeln!(w, "    bl read")?;
            if let Some(dst) = ret {
                let value = scratch.alloc()?;
                writeln!(w, "    ldrb {value}, [{buffer}]")?;
                writeln!(w, "    subs r0, r0, #1")?;
                writeln!(w, "    bne 6f")?;
                writeln!(w, "    mov r0, {value}")?;
                writeln!(w, "6:")?;
                ctx.access.store_result(w, dst, "r0", scratch)?;
                scratch.free(value);
            }
            scratch.free(buffer);
        }
        ("lamina.write", 2) => {
            ctx.access.materialize(w, &args[0], "r1")?;
            ctx.access.materialize(w, &args[1], "r2")?;
            writeln!(w, "    movs r0, #1")?;
            writeln!(w, "    bl write")?;
            emit_io_result(w, ctx, false)?;
            if let Some(dst) = ret {
                ctx.access.store_result(w, dst, "r0", scratch)?;
            }
        }
        ("lamina.read", 2) => {
            ctx.access.materialize(w, &args[0], "r1")?;
            ctx.access.materialize(w, &args[1], "r2")?;
            writeln!(w, "    movs r0, #0")?;
            writeln!(w, "    cmp r2, #0")?;
            writeln!(w, "    beq 3f")?;
            writeln!(w, "    bl read")?;
            emit_io_result(w, ctx, true)?;
            writeln!(w, "3:")?;
            if let Some(dst) = ret {
                ctx.access.store_result(w, dst, "r0", scratch)?;
            }
        }
        _ => {
            return Err(unsupported(format!(
                "{} backend does not implement the builtin `{name}`",
                isa.target_name()
            )));
        }
    }
    Ok(())
}
