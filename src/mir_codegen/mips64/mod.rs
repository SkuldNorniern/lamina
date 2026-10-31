//! MIR to MIPS64 assembly for Linux under the N64 ABI, in both byte orders.
//!
//! Every virtual register lives in a frame slot holding a full doubleword, written and
//! read at that width, so a narrow value never depends on which end of the slot the
//! target puts it. Values cross between the integer and floating-point files through
//! `mtc1`/`mfc1`, which move register to register and so never see a byte order
//! either.
//!
//! Every branch and jump is followed by a `nop`. The emitter runs under
//! `.set noreorder`, so the delay slot is this file's to fill and nothing is scheduled
//! into it.
//!
//! The emitter moves operands through `$2`, `$3`, `$12`, `$13`, `$14` and `$15` and
//! writes results back, so no register is allocated. `$30` is the frame pointer, `$29`
//! the stack pointer, `$31` the return address and `$25` holds a callee's address
//! across the call, which a position-independent callee needs.

mod abi;
mod operand;
#[cfg(test)]
mod tests;

use crate::mir_codegen::capability::{CapabilitySet, CodegenCapability};
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::{
    FunctionEmissionPlan, assign_stack_slots, compile_functions_parallel, emit_debug_file,
    emit_global_data, emit_global_directives, emit_gnu_stack_note, emit_symbol_visibility,
    external_names, integer_conversion_with_width, lamina_to_codegen_error, parallel_codegen_error,
    promoted_argument_type, reject_thread_local_globals, unsupported_inline_asm,
    unsupported_overflow_flag, unsupported_thread_local,
};
use crate::{
    error::LaminaError,
    mir::{
        Aggregate, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Instruction as MirInst, IntBinOp,
        IntCmpOp, MirType, Module as MirModule, Operand, Register, ScalarType,
        instruction::{AddressMode, MemoryAttrs},
    },
    mir_codegen::{
        CodegenError, MirCodegenSettings, RegallocStrategy, validate_module_call_parameters,
    },
};
use abi::{
    A0, A1, ARG_FPRS, ARG_GPRS, ArgKind, ArgPlace, FA0, FA1, FA2, FP, SP, T0, T1, T2, T3, T9, TRAP,
    WORD, ZERO,
};
use lamina_mir::SourceLoc;
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};
use operand::{
    Slots, fpr_to_int, int_to_fpr, load_float_operand, load_operand, phys, slot, store_float,
    store_register, unsupported,
};
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
};

use lamina_codegen::abi::common_call_stub;

/// The widest aggregate N64 returns in registers: `$2` and `$3`.
const MAX_REGISTER_RETURN: usize = 16;

/// Unaligned doubleword and word forms, which assemble to the left/right pairs. MIPS
/// traps on a misaligned access, and a pointer into a packed aggregate is misaligned.
const ULOAD: &str = "uld";
const USTORE: &str = "usd";

/// Whether an access has to take the unaligned form.
///
/// MIPS traps on a misaligned address where every other target this compiler emits for
/// reads it anyway, so the unaligned pair is the only safe lowering when the address is
/// short of the width being moved. `MemoryAttrs::align` states the alignment of the
/// address, so a field of a packed struct is covered whether or not the program spelled
/// the alignment out.
const fn unaligned(claimed: MemoryAttrs, size: usize) -> bool {
    size > 1 && (claimed.align as usize) < size
}

/// Signatures the caller side needs: parameter types, variadic arity and float returns.
struct Signatures {
    params: HashMap<String, Vec<ArgKind>>,
    variadic: HashMap<String, usize>,
    float_returns: HashMap<String, MirType>,
    aggregate_returns: HashMap<String, Aggregate>,
}

impl Signatures {
    fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| {
                    let kinds = func
                        .sig
                        .params
                        .iter()
                        .map(|p| ArgKind {
                            ty: p.ty,
                            aggregate: p.aggregate,
                        })
                        .collect();
                    (name.clone(), kinds)
                })
                .collect(),
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, func)| (name.clone(), func.sig.params.len()))
                .collect(),
            float_returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| {
                    let ty = func.sig.ret_ty?;
                    (func.sig.ret_aggregate.is_none() && ty.is_float()).then(|| (name.clone(), ty))
                })
                .collect(),
            aggregate_returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
                .collect(),
        }
    }

    fn result(&self, callee: &str) -> Option<Aggregate> {
        self.aggregate_returns.get(callee).copied()
    }

    /// Whether the callee takes a hidden pointer to the aggregate it returns.
    fn hidden_result(&self, callee: &str) -> bool {
        self.result(callee)
            .is_some_and(|a| a.size > MAX_REGISTER_RETURN)
    }

    /// Types of a call's arguments as the convention sees them. Past a variadic
    /// callee's fixed parameters N64 passes a float in a general register, so the
    /// promoted type decides nothing there and the scalar kind is enough.
    fn call_kinds(&self, callee: &str, args: &[Operand]) -> Vec<ArgKind> {
        let fixed = self.variadic.get(callee).copied();
        match self.params.get(callee) {
            Some(declared) if fixed.is_none() && declared.len() == args.len() => declared.clone(),
            declared => {
                let fixed = fixed.unwrap_or(0);
                args.iter()
                    .enumerate()
                    .map(|(index, arg)| match declared {
                        Some(kinds) if index < fixed && index < kinds.len() => kinds[index],
                        _ => ArgKind::scalar(promoted_argument_type(arg)),
                    })
                    .collect()
            }
        }
    }

    /// Whether the call reaches a variadic callee past its fixed parameters, where a
    /// float travels in a general register.
    fn variadic_from(&self, callee: &str) -> Option<usize> {
        self.variadic.get(callee).copied()
    }
}

/// Everything the instruction emitter needs beyond the instruction itself.
struct EmitCtx<'a> {
    func_name: &'a str,
    target_os: TargetOperatingSystem,
    settings: &'a MirCodegenSettings,
    loc: Option<SourceLoc>,
    /// Sequence for compiler-generated labels.
    label_seq: u32,
    /// Offset from the frame pointer of the doubleword the byte intrinsics buffer in.
    io_scratch: i32,
    signatures: &'a Signatures,
    /// Set when this function returns a float, which leaves in `$f0`.
    ret_float: Option<MirType>,
    /// Set when this function returns an aggregate.
    ret_aggregate: Option<Aggregate>,
    /// Frame offset of the hidden result pointer, when the aggregate is too wide to
    /// come back in registers.
    sret: Option<i32>,
}

impl EmitCtx<'_> {
    fn fresh_label(&mut self, tag: &str) -> String {
        let seq = self.label_seq;
        self.label_seq = self.label_seq.saturating_add(1);
        format!(".L_{}.{tag}.{seq}", self.func_name)
    }

    fn block_label(&self, block: &str) -> String {
        format!(".L_{}.{block}", self.func_name)
    }
}

/// Capabilities declared by this backend; see `CapabilitySet::for_architecture`.
pub(crate) fn capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::standard_native()
        .with(CodegenCapability::IndirectCalls)
        .with(CodegenCapability::AggregateArguments);
    caps.remove(&CodegenCapability::InlineAssembly);
    caps
}

pub fn generate_mir_mips64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
    arch: TargetArchitecture,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Linux {
        return Err(unsupported(format!(
            "mips64 code generation supports Linux only, not {target_os}"
        )));
    }
    if settings.regalloc != RegallocStrategy::Incremental {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for mips64".to_string(),
        ));
    }
    validate_module_call_parameters(module, arch)?;
    reject_thread_local_globals(module, "mips64")?;
    let signatures = Arc::new(Signatures::from_module(module));
    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_function(name, func, os, settings_arc.as_ref(), signatures.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    writeln!(writer, ".option pic0")?;
    writeln!(writer, ".set noreorder")?;
    writeln!(writer, ".data")?;
    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings)?;
    emit_global_directives(writer, module, str::to_string)?;
    emit_symbol_visibility(writer, module, target_os, str::to_string)?;
    for name in external_names(module) {
        writeln!(writer, ".extern {name}")?;
    }
    for result in results {
        writer.write_all(&result.assembly)?;
    }
    emit_global_data(writer, module, target_os, str::to_string)?;
    emit_gnu_stack_note(writer, target_os)?;
    Ok(())
}

/// Bytes of outgoing argument area the widest call in this function needs.
fn outgoing_area(func: &Function, signatures: &Signatures) -> usize {
    let mut widest = 0;
    for block in &func.blocks {
        for inst in &block.instructions {
            let (kinds, first_gpr) = match &inst.instr {
                MirInst::Call { name, args, .. } | MirInst::TailCall { name, args } => (
                    signatures.call_kinds(name, args),
                    usize::from(signatures.hidden_result(name)),
                ),
                MirInst::CallIndirect {
                    arg_types, args, ..
                } => {
                    let types: Vec<MirType> = if arg_types.len() == args.len() {
                        arg_types.clone()
                    } else {
                        args.iter().map(promoted_argument_type).collect()
                    };
                    (types.into_iter().map(ArgKind::scalar).collect(), 0)
                }
                _ => continue,
            };
            widest = widest.max(abi::call_area(&kinds, first_gpr));
        }
    }
    widest
}

fn compile_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let io = |e: io::Error| CodegenError::InvalidCodegenOptions(format!("IO error: {e}"));
    if let Some(align) = func.stack_alignments.values().copied().max()
        && align > 16
    {
        return Err(lamina_to_codegen(unsupported(format!(
            "mips64 stack slot wanting {align}-byte alignment; the frame pointer holds sixteen"
        ))));
    }
    writeln!(output, "{func_name}:").map_err(io)?;

    let (slots, _, _) = assign_stack_slots(func, abi::slot_offset);
    let locals = slots
        .values()
        .map(|offset| (-offset) as usize)
        .max()
        .unwrap_or(0);
    let outgoing = outgoing_area(func, signatures);
    let hidden = func
        .sig
        .ret_aggregate
        .is_some_and(|a| a.size > MAX_REGISTER_RETURN);
    let sret = hidden.then(|| -(locals as i32) - WORD);
    let locals = locals + usize::from(hidden) * WORD as usize;
    abi::prologue(&mut output, locals, outgoing).map_err(io)?;
    if let Some(offset) = sret {
        abi::fp_store(&mut output, "sd", ARG_GPRS[0], offset, T0).map_err(io)?;
    }
    store_incoming_arguments(&mut output, func, &slots, usize::from(hidden))
        .map_err(lamina_to_codegen)?;

    let mut ctx = EmitCtx {
        func_name,
        target_os,
        settings,
        loc: None,
        label_seq: 0,
        io_scratch: abi::io_scratch_offset(locals),
        signatures,
        ret_float: func
            .sig
            .ret_ty
            .filter(|ty| func.sig.ret_aggregate.is_none() && ty.is_float()),
        ret_aggregate: func.sig.ret_aggregate,
        sret,
    };
    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    j {}", ctx.block_label(&func.entry)).map_err(io)?;
        writeln!(output, "    nop").map_err(io)?;
    }
    for (position, &index) in plan.order.iter().enumerate() {
        let next = plan
            .order
            .get(position + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        emit_block(func, index, next, &plan, &mut output, &slots, &mut ctx)
            .map_err(lamina_to_codegen)?;
    }
    Ok(output)
}

fn lamina_to_codegen(error: LaminaError) -> CodegenError {
    lamina_to_codegen_error(error)
}

/// Stores each incoming argument into its frame slot.
fn store_incoming_arguments<W: Write>(
    writer: &mut W,
    func: &Function,
    slots: &Slots,
    first_gpr: usize,
) -> Result<(), LaminaError> {
    let kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind {
            ty: p.ty,
            aggregate: p.aggregate,
        })
        .collect();
    let places = abi::classify_kinds(&kinds, first_gpr);
    let banks = ARG_GPRS.len();
    for ((param, _), place) in func.sig.params.iter().zip(&kinds).zip(places) {
        let Register::Virtual(vreg) = &param.reg else {
            continue;
        };
        let Some(&offset) = slots.get(vreg) else {
            continue;
        };
        let single = param.ty.size_bytes(PointerWidth::Bits64) == 4;
        match place {
            ArgPlace::Gpr(bank) => {
                abi::fp_store(writer, "sd", ARG_GPRS[bank], offset, T0)?;
            }
            ArgPlace::Fpr(bank) => {
                store_float(ARG_FPRS[bank], vreg, writer, slots, single)?;
            }
            ArgPlace::Words { first, words } => {
                for index in 0..words {
                    let at = offset + index as i32 * WORD;
                    abi::fp_store(writer, "sd", ARG_GPRS[first + index], at, T0)?;
                }
            }
            ArgPlace::Split { first, words } => {
                for index in 0..words {
                    let at = offset + index as i32 * WORD;
                    if first + index < banks {
                        abi::fp_store(writer, "sd", ARG_GPRS[first + index], at, T0)?;
                    } else {
                        let from = abi::incoming_stack_offset(first + index - banks);
                        abi::fp_load(writer, "ld", T1, from, T2)?;
                        abi::fp_store(writer, "sd", T1, at, T2)?;
                    }
                }
            }
            ArgPlace::Stack(index) => {
                let from = abi::incoming_stack_offset(index);
                abi::fp_load(writer, "ld", T1, from, T2)?;
                abi::fp_store(writer, "sd", T1, offset, T2)?;
            }
            ArgPlace::StackWords { slot: first, words } => {
                for index in 0..words {
                    let from = abi::incoming_stack_offset(first + index);
                    abi::fp_load(writer, "ld", T1, from, T2)?;
                    abi::fp_store(writer, "sd", T1, offset + index as i32 * WORD, T2)?;
                }
            }
        }
    }
    Ok(())
}

fn emit_block<W: Write>(
    func: &Function,
    index: usize,
    next: &str,
    plan: &FunctionEmissionPlan,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    let block = &func.blocks[index];
    let func_name = ctx.func_name;
    if plan.aligned_headers.contains(&index) {
        writeln!(writer, "    .p2align 4")?;
    }
    writeln!(writer, ".L_{func_name}.{}:", block.label)?;
    let mut position = 0;
    while position < block.instructions.len() {
        let inst = &block.instructions[position];
        if plan.comparisons.contains(&(index, position)) {
            if let (
                MirInst::IntCmp {
                    op, lhs, rhs, ty, ..
                },
                MirInst::Br {
                    true_target,
                    false_target,
                    ..
                },
            ) = (&inst.instr, &block.instructions[position + 1].instr)
            {
                emit_source_loc(writer, ctx.settings, inst.loc)?;
                load_operand(lhs, writer, slots, A0)?;
                load_operand(rhs, writer, slots, A1)?;
                let unsigned = is_unsigned(*op);
                extend_integer(writer, A0, ty, !unsigned)?;
                extend_integer(writer, A1, ty, !unsigned)?;
                set_compare(writer, T0, A0, A1, *op)?;
                let (invert, taken, other) =
                    plan.branch(func, index, next, true_target, false_target);
                let taken = taken.to_string();
                branch_to_block(writer, ctx, invert, T0, ZERO, &taken)?;
                if let Some(other) = other {
                    writeln!(writer, "    j {}", ctx.block_label(other))?;
                    writeln!(writer, "    nop")?;
                }
            }
            position += 1;
        } else if matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
        } else {
            ctx.loc = inst.loc;
            emit_instruction(inst, writer, slots, ctx)?;
        }
        position += 1;
    }
    Ok(())
}

/// Branches to a block label. The conditional forms reach sixteen signed bits of
/// instructions, which an optimised function outruns, so the condition selects a jump
/// over the two instructions that skip it instead.
fn branch_to_block<W: Write>(
    writer: &mut W,
    ctx: &mut EmitCtx<'_>,
    on_equal: bool,
    lhs: &str,
    rhs: &str,
    label: &str,
) -> Result<(), LaminaError> {
    let skip = ctx.fresh_label("skip");
    let inverse = if on_equal { "bne" } else { "beq" };
    writeln!(writer, "    {inverse} {lhs}, {rhs}, {skip}")?;
    writeln!(writer, "    nop")?;
    writeln!(writer, "    j {}", ctx.block_label(label))?;
    writeln!(writer, "    nop")?;
    writeln!(writer, "{skip}:")?;
    Ok(())
}

fn is_unsigned(op: IntCmpOp) -> bool {
    matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    )
}

/// Leaves 1 or 0 in `dst` for the comparison of `lhs` against `rhs`. `dst` may be
/// either operand: every form reads both before writing.
fn set_compare<W: Write>(
    writer: &mut W,
    dst: &str,
    lhs: &str,
    rhs: &str,
    op: IntCmpOp,
) -> Result<(), LaminaError> {
    match op {
        IntCmpOp::Eq => {
            writeln!(writer, "    xor {dst}, {lhs}, {rhs}")?;
            writeln!(writer, "    sltiu {dst}, {dst}, 1")?;
        }
        IntCmpOp::Ne => {
            writeln!(writer, "    xor {dst}, {lhs}, {rhs}")?;
            writeln!(writer, "    sltu {dst}, {ZERO}, {dst}")?;
        }
        IntCmpOp::SLt => writeln!(writer, "    slt {dst}, {lhs}, {rhs}")?,
        IntCmpOp::ULt => writeln!(writer, "    sltu {dst}, {lhs}, {rhs}")?,
        IntCmpOp::SGt => writeln!(writer, "    slt {dst}, {rhs}, {lhs}")?,
        IntCmpOp::UGt => writeln!(writer, "    sltu {dst}, {rhs}, {lhs}")?,
        IntCmpOp::SLe => {
            writeln!(writer, "    slt {dst}, {rhs}, {lhs}")?;
            writeln!(writer, "    xori {dst}, {dst}, 1")?;
        }
        IntCmpOp::ULe => {
            writeln!(writer, "    sltu {dst}, {rhs}, {lhs}")?;
            writeln!(writer, "    xori {dst}, {dst}, 1")?;
        }
        IntCmpOp::SGe => {
            writeln!(writer, "    slt {dst}, {lhs}, {rhs}")?;
            writeln!(writer, "    xori {dst}, {dst}, 1")?;
        }
        IntCmpOp::UGe => {
            writeln!(writer, "    sltu {dst}, {lhs}, {rhs}")?;
            writeln!(writer, "    xori {dst}, {dst}, 1")?;
        }
    }
    Ok(())
}

/// The condition instruction for a float comparison, and whether the branch that reads
/// it is taken when the condition held. `Ne` is the one ordering that includes
/// unordered operands, so it tests equality and takes the other branch.
fn float_condition(op: FloatCmpOp, single: bool) -> (&'static str, bool, bool) {
    let suffix = if single { "s" } else { "d" };
    let _ = suffix;
    match op {
        FloatCmpOp::Eq => ("eq", false, true),
        FloatCmpOp::Ne => ("eq", false, false),
        FloatCmpOp::Lt => ("lt", false, true),
        FloatCmpOp::Le => ("le", false, true),
        FloatCmpOp::Gt => ("lt", true, true),
        FloatCmpOp::Ge => ("le", true, true),
    }
}

/// Sign- or zero-extends a narrow integer in `reg` to the 64-bit form values take.
fn extend_integer<W: Write>(
    writer: &mut W,
    reg: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    let bits = match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => 8,
        MirType::Scalar(ScalarType::I16) => 16,
        MirType::Scalar(ScalarType::I32) => 32,
        _ => return Ok(()),
    };
    if !signed && bits < 32 {
        let mask = (1u32 << bits) - 1;
        writeln!(writer, "    andi {reg}, {reg}, {mask}")?;
        return Ok(());
    }
    let shift = 64 - bits;
    let right = if signed { "dsra" } else { "dsrl" };
    writeln!(writer, "    dsll {reg}, {reg}, {shift}")?;
    writeln!(writer, "    {right} {reg}, {reg}, {shift}")?;
    Ok(())
}

/// Stores an integer register to a MIR destination.
fn store_result<W: Write>(
    writer: &mut W,
    src: &str,
    dst: &Register,
    slots: &Slots,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(vreg) => store_register(src, vreg, writer, slots),
        Register::Physical(physical) => {
            let name = phys(physical.name);
            if name != src {
                writeln!(writer, "    move {name}, {src}")?;
            }
            Ok(())
        }
    }
}

fn load_base<W: Write>(
    writer: &mut W,
    base: &Register,
    slots: &Slots,
    dst: &str,
) -> Result<(), LaminaError> {
    match base {
        Register::Virtual(vreg) => {
            let offset = slot(slots, vreg)?;
            abi::fp_load(writer, "ld", dst, offset, dst)?;
        }
        Register::Physical(physical) => {
            writeln!(writer, "    move {dst}, {}", phys(physical.name))?;
        }
    }
    Ok(())
}

fn call_target(name: &str, target_os: TargetOperatingSystem) -> String {
    common_call_stub(name, target_os).unwrap_or_else(|| name.to_string())
}

/// Calls `name`. The address goes in `$25` first: a position-independent callee, which
/// every function in a shared C library is, derives its own global pointer from it.
fn emit_direct_call<W: Write>(
    writer: &mut W,
    name: &str,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    abi::symbol_address(writer, T9, T0, &call_target(name, target_os))?;
    writeln!(writer, "    jalr {T9}")?;
    writeln!(writer, "    nop")?;
    Ok(())
}

/// Places the arguments of a call in the outgoing area the prologue already reserved.
/// The stack words go first, so the register loads that follow are not disturbed.
fn pass_arguments<W: Write>(
    writer: &mut W,
    args: &[Operand],
    kinds: &[ArgKind],
    slots: &Slots,
    hidden: Option<i32>,
    variadic_from: Option<usize>,
) -> Result<(), LaminaError> {
    let first_gpr = usize::from(hidden.is_some());
    let places = abi::classify_kinds(kinds, first_gpr);
    let banks = ARG_GPRS.len();
    for ((index, (arg, kind)), place) in args.iter().zip(kinds).enumerate().zip(&places) {
        match *place {
            ArgPlace::Stack(at) => {
                load_operand(arg, writer, slots, T1)?;
                abi::memory(writer, "sd", T1, SP, abi::outgoing_stack_offset(at), T2)?;
            }
            ArgPlace::StackWords { slot: at, words } => {
                load_operand(arg, writer, slots, T1)?;
                for word in 0..words {
                    let step = word as i64 * i64::from(WORD);
                    abi::memory(writer, ULOAD, T2, T1, step, T3)?;
                    abi::memory(
                        writer,
                        "sd",
                        T2,
                        SP,
                        abi::outgoing_stack_offset(at + word),
                        T3,
                    )?;
                }
            }
            ArgPlace::Split { first, words } => {
                load_operand(arg, writer, slots, T1)?;
                for word in first..first + words {
                    if word < banks {
                        continue;
                    }
                    let step = (word - first) as i64 * i64::from(WORD);
                    abi::memory(writer, ULOAD, T2, T1, step, T3)?;
                    abi::memory(
                        writer,
                        "sd",
                        T2,
                        SP,
                        abi::outgoing_stack_offset(word - banks),
                        T3,
                    )?;
                }
            }
            _ => {}
        }
        let _ = (index, kind);
    }
    for ((arg, kind), place) in args.iter().zip(kinds).zip(&places) {
        if let ArgPlace::Fpr(bank) = *place {
            load_float_operand(
                arg,
                writer,
                slots,
                ARG_FPRS[bank],
                kind.ty.size_bytes(PointerWidth::Bits64) == 4,
            )?;
        }
    }
    if let Some(offset) = hidden {
        abi::address(writer, ARG_GPRS[0], FP, i64::from(offset))?;
    }
    for ((index, (arg, kind)), place) in args.iter().zip(kinds).enumerate().zip(&places) {
        match *place {
            ArgPlace::Gpr(bank) => load_operand(arg, writer, slots, ARG_GPRS[bank])?,
            ArgPlace::Words { first, words } | ArgPlace::Split { first, words } => {
                load_operand(arg, writer, slots, T1)?;
                for word in 0..words {
                    if first + word >= banks {
                        break;
                    }
                    let step = word as i64 * i64::from(WORD);
                    abi::memory(writer, ULOAD, ARG_GPRS[first + word], T1, step, T2)?;
                }
            }
            ArgPlace::Fpr(bank) => {
                // A float past a variadic callee's fixed parameters travels in the
                // general register that matches its doubleword as well.
                if variadic_from.is_some_and(|fixed| index >= fixed) {
                    load_operand(arg, writer, slots, ARG_GPRS[bank])?;
                }
                let _ = kind;
            }
            ArgPlace::Stack(_) | ArgPlace::StackWords { .. } => {}
        }
    }
    Ok(())
}

/// Stores a call's result into its destination: `$2`, or `$f0` when the callee returns
/// a float.
fn store_call_result<W: Write>(
    writer: &mut W,
    ret: &Register,
    float: Option<MirType>,
    slots: &Slots,
) -> Result<(), LaminaError> {
    match (ret, float) {
        (Register::Virtual(vreg), Some(ty)) => store_float(
            FA0,
            vreg,
            writer,
            slots,
            ty.size_bytes(PointerWidth::Bits64) == 4,
        ),
        (dst, _) => store_result(writer, A0, dst, slots),
    }
}

fn emit_instruction<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    if let Some(operations) =
        integer_conversion_with_width(inst, MirType::Scalar(ScalarType::I64), PointerWidth::Bits64)
    {
        for operation in operations {
            emit_instruction(&operation, writer, slots, ctx)?;
        }
        return Ok(());
    }
    if ctx.settings.emit_asm_debug_lines
        && let Some(loc) = ctx.loc
    {
        writeln!(writer, "    .loc 1 {} {}", loc.line, loc.column)?;
    }
    match inst {
        MirInst::ThreadLocalAddr { .. } => {
            return Err(unsupported_thread_local("mips64"));
        }
        MirInst::IntOverflow { .. } => {
            return Err(unsupported_overflow_flag("mips64"));
        }
        MirInst::Copy { ty, dst, src } => {
            if ty.is_vector() {
                return Err(unsupported("mips64 vector Copy"));
            }
            copy_value(writer, slots, ty, ty, dst, src)?;
        }
        MirInst::Bitcast { from, to, dst, src } => {
            if from.is_vector() || to.is_vector() {
                return Err(unsupported("mips64 vector Bitcast"));
            }
            copy_value(writer, slots, from, to, dst, src)?;
        }
        MirInst::SymbolAddress { dst, symbol } => {
            abi::symbol_address(writer, A0, T0, symbol)?;
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::StackAddr { dst, slot: stack } => {
            let offset = slot(slots, stack)?;
            abi::address(writer, A0, FP, i64::from(offset))?;
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::Lea { dst, base, offset } => {
            let Register::Virtual(base_vreg) = base else {
                return Err(unsupported(format!(
                    "mips64 lea base {base:?} is not a virtual register"
                )));
            };
            let base_slot = slot(slots, base_vreg)?;
            abi::fp_load(writer, "ld", A0, base_slot, A0)?;
            if *offset != 0 {
                abi::address(writer, A0, A0, i64::from(*offset))?;
            }
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            emit_int_binary(writer, slots, ctx, op, dst, lhs, rhs, ty)?;
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            load_operand(lhs, writer, slots, A0)?;
            load_operand(rhs, writer, slots, A1)?;
            let unsigned = is_unsigned(*op);
            extend_integer(writer, A0, ty, !unsigned)?;
            extend_integer(writer, A1, ty, !unsigned)?;
            set_compare(writer, A0, A0, A1, *op)?;
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty,
        } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            let float = ty.is_float();
            if float {
                load_float_operand(false_val, writer, slots, FA1, single)?;
                load_float_operand(true_val, writer, slots, FA0, single)?;
            } else {
                load_operand(false_val, writer, slots, A1)?;
                load_operand(true_val, writer, slots, A0)?;
            }
            load_operand(&Operand::Register(cond.clone()), writer, slots, T0)?;
            if float {
                let move_fp = if single { "movz.s" } else { "movz.d" };
                writeln!(writer, "    {move_fp} {FA0}, {FA1}, {T0}")?;
            } else {
                writeln!(writer, "    movz {A0}, {A1}, {T0}")?;
            }
            match (dst, float) {
                (Register::Virtual(vreg), true) => store_float(FA0, vreg, writer, slots, single)?,
                (dst, _) => store_result(writer, A0, dst, slots)?,
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs,
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported(
                    "mips64 load supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let size = ty.size_bytes(PointerWidth::Bits64);
            let wide = unaligned(*attrs, size);
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("lb", false),
                MirType::Scalar(ScalarType::I16) if wide => ("ulh", false),
                MirType::Scalar(ScalarType::I16) => ("lh", false),
                MirType::Scalar(ScalarType::I32) if wide => ("ulw", false),
                MirType::Scalar(ScalarType::I32) => ("lw", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_))
                    if wide =>
                {
                    ("uld", false)
                }
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("ld", false)
                }
                MirType::Scalar(ScalarType::F32) if wide => ("ulw", false),
                MirType::Scalar(ScalarType::F64) if wide => ("uld", false),
                MirType::Scalar(ScalarType::F32) => ("lwc1", true),
                MirType::Scalar(ScalarType::F64) => ("ldc1", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "mips64 load of vector type {ty} is not implemented"
                    )));
                }
            };
            load_base(writer, base, slots, T0)?;
            let float = float || (wide && ty.is_float());
            let reg = if float && !wide { FA0 } else { A0 };
            abi::memory(writer, mnemonic, reg, T0, i64::from(*offset), T1)?;
            if wide && ty.is_float() {
                int_to_fpr(writer, FA0, A0, size == 4)?;
            }
            match (dst, float) {
                (Register::Virtual(vreg), true) => store_float(
                    FA0,
                    vreg,
                    writer,
                    slots,
                    matches!(ty, MirType::Scalar(ScalarType::F32)),
                )?,
                (dst, _) => store_result(writer, A0, dst, slots)?,
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs,
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported(
                    "mips64 store supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let size = ty.size_bytes(PointerWidth::Bits64);
            let wide = unaligned(*attrs, size);
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("sb", false),
                MirType::Scalar(ScalarType::I16) if wide => ("ush", false),
                MirType::Scalar(ScalarType::I16) => ("sh", false),
                MirType::Scalar(ScalarType::I32) if wide => ("usw", false),
                MirType::Scalar(ScalarType::I32) => ("sw", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_))
                    if wide =>
                {
                    ("usd", false)
                }
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("sd", false)
                }
                MirType::Scalar(ScalarType::F32) if wide => ("usw", false),
                MirType::Scalar(ScalarType::F64) if wide => ("usd", false),
                MirType::Scalar(ScalarType::F32) => ("swc1", true),
                MirType::Scalar(ScalarType::F64) => ("sdc1", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "mips64 store of vector type {ty} is not implemented"
                    )));
                }
            };
            let reg = if float { FA0 } else { A0 };
            if float {
                load_float_operand(
                    src,
                    writer,
                    slots,
                    reg,
                    matches!(ty, MirType::Scalar(ScalarType::F32)),
                )?;
            } else if wide && ty.is_float() {
                load_float_operand(src, writer, slots, FA0, size == 4)?;
                fpr_to_int(writer, A0, FA0, size == 4)?;
            } else {
                load_operand(src, writer, slots, reg)?;
            }
            load_base(writer, base, slots, T0)?;
            abi::memory(writer, mnemonic, reg, T0, i64::from(*offset), T1)?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match (ctx.ret_aggregate, ctx.sret) {
                    (Some(aggregate), Some(offset)) => {
                        load_operand(val, writer, slots, T0)?;
                        abi::fp_load(writer, "ld", A1, offset, A1)?;
                        for index in 0..aggregate.words() {
                            let at = index as i64 * i64::from(WORD);
                            abi::memory(writer, ULOAD, T1, T0, at, T2)?;
                            abi::memory(writer, USTORE, T1, A1, at, T2)?;
                        }
                        writeln!(writer, "    move {A0}, {A1}")?;
                    }
                    (Some(aggregate), None) => {
                        load_operand(val, writer, slots, T0)?;
                        if aggregate.words() > 1 {
                            abi::memory(writer, ULOAD, A1, T0, i64::from(WORD), T1)?;
                        }
                        abi::memory(writer, ULOAD, A0, T0, 0, T1)?;
                    }
                    (None, _) => match ctx.ret_float {
                        Some(ty) => load_float_operand(
                            val,
                            writer,
                            slots,
                            FA0,
                            ty.size_bytes(PointerWidth::Bits64) == 4,
                        )?,
                        None => load_operand(val, writer, slots, A0)?,
                    },
                }
            }
            abi::epilogue(writer, None)?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    j {}", ctx.block_label(target))?;
            writeln!(writer, "    nop")?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            load_operand(&Operand::Register(cond.clone()), writer, slots, T0)?;
            branch_to_block(writer, ctx, false, T0, ZERO, true_target)?;
            writeln!(writer, "    j {}", ctx.block_label(false_target))?;
            writeln!(writer, "    nop")?;
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            load_operand(&Operand::Register(value.clone()), writer, slots, A0)?;
            for (case_val, case_label) in cases {
                abi::load_imm(writer, T0, *case_val)?;
                branch_to_block(writer, ctx, true, A0, T0, case_label)?;
            }
            writeln!(writer, "    j {}", ctx.block_label(default))?;
            writeln!(writer, "    nop")?;
        }
        MirInst::Call { name, args, ret } => {
            emit_call(name, args, ret.as_ref(), writer, slots, ctx)?;
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed,
            ret,
            ret_ty,
        } => {
            let kinds: Vec<ArgKind> = arg_types.iter().copied().map(ArgKind::scalar).collect();
            pass_arguments(writer, args, &kinds, slots, None, *variadic_fixed)?;
            load_operand(target, writer, slots, T9)?;
            writeln!(writer, "    jalr {T9}")?;
            writeln!(writer, "    nop")?;
            if let Some(ret) = ret {
                store_call_result(writer, ret, ret_ty.filter(MirType::is_float), slots)?;
            }
        }
        // The epilogue restores the frame pointer the argument values are addressed
        // from, so the arguments cannot be placed after it. Lowered as an ordinary call
        // and a return instead, which costs a frame but never returns a wrong answer.
        MirInst::TailCall { name, args } => {
            if ctx.ret_aggregate.is_some()
                || ctx.signatures.result(name).is_some()
                || ctx
                    .signatures
                    .call_kinds(name, args)
                    .iter()
                    .any(|kind| kind.aggregate.is_some())
            {
                return Err(unsupported(
                    "mips64 tail call passing or returning an aggregate",
                ));
            }
            emit_call(name, args, None, writer, slots, ctx)?;
            abi::epilogue(writer, None)?;
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            load_float_operand(lhs, writer, slots, FA0, single)?;
            load_float_operand(rhs, writer, slots, FA1, single)?;
            let mnemonic = match op {
                FloatBinOp::FAdd => "add",
                FloatBinOp::FSub => "sub",
                FloatBinOp::FMul => "mul",
                FloatBinOp::FDiv => "div",
            };
            writeln!(writer, "    {mnemonic}.{suffix} {FA0}, {FA0}, {FA1}")?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            load_float_operand(src, writer, slots, FA0, single)?;
            let mnemonic = match op {
                FloatUnOp::FNeg => "neg",
                FloatUnOp::FSqrt => "sqrt",
            };
            writeln!(writer, "    {mnemonic}.{suffix} {FA0}, {FA0}")?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            load_float_operand(lhs, writer, slots, FA0, single)?;
            load_float_operand(rhs, writer, slots, FA1, single)?;
            let (condition, swap, when_true) = float_condition(*op, single);
            let (first, second) = if swap { (FA1, FA0) } else { (FA0, FA1) };
            writeln!(writer, "    c.{condition}.{suffix} {first}, {second}")?;
            let done = ctx.fresh_label("fcmp");
            writeln!(writer, "    li {A0}, 1")?;
            let branch = if when_true { "bc1t" } else { "bc1f" };
            writeln!(writer, "    {branch} {done}")?;
            writeln!(writer, "    nop")?;
            writeln!(writer, "    li {A0}, 0")?;
            writeln!(writer, "{done}:")?;
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            emit_float_to_int(writer, slots, ctx, from, to, *signed, dst, src)?;
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            emit_int_to_float(writer, slots, ctx, from, to, *signed, dst, src)?;
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let from_single = from.size_bytes(PointerWidth::Bits64) == 4;
            let to_single = to.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(src, writer, slots, FA0, from_single)?;
            match (from_single, to_single) {
                (true, false) => writeln!(writer, "    cvt.d.s {FA0}, {FA0}")?,
                (false, true) => writeln!(writer, "    cvt.s.d {FA0}, {FA0}")?,
                _ => {}
            }
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, to_single)?;
            }
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Unreachable => {
            writeln!(writer, "{TRAP}")?;
        }
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {}
        MirInst::IntConvert { .. } => {
            return Err(unsupported(
                "mips64 received an integer conversion that was not lowered",
            ));
        }
        MirInst::IntUnary { .. } => {
            return Err(unsupported(
                "mips64 integer bit intrinsics (popcount, ctlz, cttz, bswap)",
            ));
        }
        MirInst::IntRotate { .. } => {
            return Err(unsupported("mips64 integer rotates"));
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported("mips64 vector operations"));
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            return Err(unsupported("mips64 atomic operations and fences"));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported_inline_asm("mips64"));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(unsupported("mips64 variadic argument access"));
        }
    }
    Ok(())
}

/// Moves a value between frame slots, converting between the integer and the
/// floating-point placement when the register class changes.
fn copy_value<W: Write>(
    writer: &mut W,
    slots: &Slots,
    from: &MirType,
    to: &MirType,
    dst: &Register,
    src: &Operand,
) -> Result<(), LaminaError> {
    let from_single = from.size_bytes(PointerWidth::Bits64) == 4;
    let to_single = to.size_bytes(PointerWidth::Bits64) == 4;
    match (from.is_float(), to.is_float()) {
        (true, true) => {
            load_float_operand(src, writer, slots, FA0, from_single)?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, to_single)?;
            }
        }
        (false, false) => {
            load_operand(src, writer, slots, A0)?;
            extend_integer(writer, A0, to, true)?;
            store_result(writer, A0, dst, slots)?;
        }
        (false, true) => {
            load_operand(src, writer, slots, A0)?;
            int_to_fpr(writer, FA0, A0, to_single)?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, to_single)?;
            }
        }
        (true, false) => {
            load_float_operand(src, writer, slots, FA0, from_single)?;
            fpr_to_int(writer, A0, FA0, from_single)?;
            store_result(writer, A0, dst, slots)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_float_to_int<W: Write>(
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
) -> Result<(), LaminaError> {
    let single = from.size_bytes(PointerWidth::Bits64) == 4;
    let suffix = if single { "s" } else { "d" };
    let bits = to.size_bytes(PointerWidth::Bits64) * 8;
    load_float_operand(src, writer, slots, FA0, single)?;
    let nan = ctx.fresh_label("f2i_nan");
    let done = ctx.fresh_label("f2i");
    writeln!(writer, "    mov.{suffix} {FA1}, {FA0}")?;
    writeln!(writer, "    c.eq.{suffix} {FA0}, {FA0}")?;
    writeln!(writer, "    li {A0}, 0")?;
    writeln!(writer, "    bc1f {nan}")?;
    writeln!(writer, "    nop")?;
    if signed || bits < 64 {
        writeln!(writer, "    trunc.l.{suffix} {FA0}, {FA0}")?;
        writeln!(writer, "    dmfc1 {A0}, {FA0}")?;
        // A truncation that overflows gives the largest signed doubleword whichever
        // way the value ran, so a source below the range is put back at the smallest.
        let ordered = ctx.fresh_label("f2i_sat");
        abi::load_imm(writer, T1, i64::MAX)?;
        writeln!(writer, "    bne {A0}, {T1}, {ordered}")?;
        writeln!(writer, "    nop")?;
        negative_source(writer, suffix, single)?;
        writeln!(writer, "    bc1f {ordered}")?;
        writeln!(writer, "    nop")?;
        abi::load_imm(writer, A0, i64::MIN)?;
        writeln!(writer, "{ordered}:")?;
    } else {
        // Above two to the sixty-third a signed truncation saturates, so the value
        // comes down by that much first and the bit goes back afterwards.
        let small = ctx.fresh_label("f2u_small");
        let bias: i64 = if single {
            0x5f00_0000
        } else {
            0x43e0_0000_0000_0000
        };
        abi::load_imm(writer, T1, bias)?;
        int_to_fpr(writer, FA2, T1, single)?;
        writeln!(writer, "    c.lt.{suffix} {FA0}, {FA2}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    bc1t {small}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    sub.{suffix} {FA0}, {FA0}, {FA2}")?;
        writeln!(writer, "    trunc.l.{suffix} {FA0}, {FA0}")?;
        writeln!(writer, "    dmfc1 {A0}, {FA0}")?;
        abi::load_imm(writer, T1, i64::MIN)?;
        writeln!(writer, "    or {A0}, {A0}, {T1}")?;
        writeln!(writer, "    j {done}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{small}:")?;
        writeln!(writer, "    trunc.l.{suffix} {FA0}, {FA0}")?;
        writeln!(writer, "    dmfc1 {A0}, {FA0}")?;
    }
    writeln!(writer, "    j {done}")?;
    writeln!(writer, "    nop")?;
    writeln!(writer, "{nan}:")?;
    writeln!(writer, "    li {A0}, 0")?;
    writeln!(writer, "{done}:")?;
    if !signed {
        // Nothing below zero is in range, and the truncation keeps the sign.
        let positive = ctx.fresh_label("f2u_pos");
        negative_source(writer, suffix, single)?;
        writeln!(writer, "    bc1f {positive}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    li {A0}, 0")?;
        writeln!(writer, "{positive}:")?;
    }
    if bits < 64 {
        if signed {
            clamp(writer, ctx, (1i64 << (bits - 1)) - 1, -(1i64 << (bits - 1)))?;
        } else {
            clamp_unsigned(writer, ctx, (1i64 << bits) - 1)?;
        }
        extend_integer(writer, A0, to, signed)?;
    }
    store_result(writer, A0, dst, slots)
}

/// Sets the floating-point condition when the value kept in `$f2` is below zero.
fn negative_source<W: Write>(
    writer: &mut W,
    suffix: &str,
    single: bool,
) -> Result<(), LaminaError> {
    int_to_fpr(writer, FA2, ZERO, single)?;
    writeln!(writer, "    c.lt.{suffix} {FA1}, {FA2}")?;
    writeln!(writer, "    nop")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_to_float<W: Write>(
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
) -> Result<(), LaminaError> {
    let from_bits = from.size_bytes(PointerWidth::Bits64) * 8;
    let single = to.size_bytes(PointerWidth::Bits64) == 4;
    let suffix = if single { "s" } else { "d" };
    load_operand(src, writer, slots, A0)?;
    if from_bits < 64 {
        extend_integer(writer, A0, from, signed)?;
    }
    if signed || from_bits < 64 {
        writeln!(writer, "    dmtc1 {A0}, {FA0}")?;
        writeln!(writer, "    cvt.{suffix}.l {FA0}, {FA0}")?;
    } else {
        // A value with the top bit set reads as negative, so it is halved, converted
        // and doubled, with the bit that halving dropped folded back in first.
        let positive = ctx.fresh_label("u2f_pos");
        let done = ctx.fresh_label("u2f");
        writeln!(writer, "    bgez {A0}, {positive}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    dsrl {T1}, {A0}, 1")?;
        writeln!(writer, "    andi {T2}, {A0}, 1")?;
        writeln!(writer, "    or {T1}, {T1}, {T2}")?;
        writeln!(writer, "    dmtc1 {T1}, {FA0}")?;
        writeln!(writer, "    cvt.{suffix}.l {FA0}, {FA0}")?;
        writeln!(writer, "    add.{suffix} {FA0}, {FA0}, {FA0}")?;
        writeln!(writer, "    j {done}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{positive}:")?;
        writeln!(writer, "    dmtc1 {A0}, {FA0}")?;
        writeln!(writer, "    cvt.{suffix}.l {FA0}, {FA0}")?;
        writeln!(writer, "{done}:")?;
    }
    if let Register::Virtual(vreg) = dst {
        store_float(FA0, vreg, writer, slots, single)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_binary<W: Write>(
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
    op: &IntBinOp,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    ty: &MirType,
) -> Result<(), LaminaError> {
    load_operand(lhs, writer, slots, A0)?;
    load_operand(rhs, writer, slots, A1)?;
    if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
        extend_integer(writer, A0, ty, false)?;
        extend_integer(writer, A1, ty, false)?;
    }
    if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
        extend_integer(writer, A0, ty, true)?;
        if !matches!(op, IntBinOp::AShr) {
            extend_integer(writer, A1, ty, true)?;
        }
    }
    let (zero, overflow) = op.division_checks(
        *ty,
        lhs.integer_constant(),
        rhs.integer_constant(),
        PointerWidth::Bits64,
    );
    if zero {
        let ok = ctx.fresh_label("nonzero");
        writeln!(writer, "    bne {A1}, {ZERO}, {ok}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{TRAP}")?;
        writeln!(writer, "{ok}:")?;
    }
    if overflow {
        let ok = ctx.fresh_label("nooverflow");
        let minimum = ty.signed_integer(
            1i64 << (ty.size_bytes(PointerWidth::Bits64) * 8 - 1),
            PointerWidth::Bits64,
        );
        writeln!(writer, "    dli {T0}, -1")?;
        writeln!(writer, "    bne {A1}, {T0}, {ok}")?;
        writeln!(writer, "    nop")?;
        abi::load_imm(writer, T0, minimum)?;
        writeln!(writer, "    bne {A0}, {T0}, {ok}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{TRAP}")?;
        writeln!(writer, "{ok}:")?;
    }
    if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr)
        && ty.size_bytes(PointerWidth::Bits64) < 8
    {
        let mask = ty.size_bytes(PointerWidth::Bits64) * 8 - 1;
        writeln!(writer, "    andi {A1}, {A1}, {mask}")?;
    }
    match op {
        IntBinOp::SDiv | IntBinOp::SRem => {
            writeln!(writer, "    ddiv {ZERO}, {A0}, {A1}")?;
            let read = if matches!(op, IntBinOp::SDiv) {
                "mflo"
            } else {
                "mfhi"
            };
            writeln!(writer, "    {read} {A0}")?;
        }
        IntBinOp::UDiv | IntBinOp::URem => {
            writeln!(writer, "    ddivu {ZERO}, {A0}, {A1}")?;
            let read = if matches!(op, IntBinOp::UDiv) {
                "mflo"
            } else {
                "mfhi"
            };
            writeln!(writer, "    {read} {A0}")?;
        }
        IntBinOp::Mul => {
            writeln!(writer, "    dmult {A0}, {A1}")?;
            writeln!(writer, "    mflo {A0}")?;
        }
        IntBinOp::Shl => writeln!(writer, "    dsllv {A0}, {A0}, {A1}")?,
        IntBinOp::LShr => writeln!(writer, "    dsrlv {A0}, {A0}, {A1}")?,
        IntBinOp::AShr => writeln!(writer, "    dsrav {A0}, {A0}, {A1}")?,
        other => {
            let mnemonic = match other {
                IntBinOp::Add => "daddu",
                IntBinOp::Sub => "dsubu",
                IntBinOp::And => "and",
                IntBinOp::Or => "or",
                IntBinOp::Xor => "xor",
                _ => unreachable!("handled above"),
            };
            writeln!(writer, "    {mnemonic} {A0}, {A0}, {A1}")?;
        }
    }
    extend_integer(writer, A0, ty, true)?;
    store_result(writer, A0, dst, slots)
}

/// Holds `$2` within a signed range.
fn clamp<W: Write>(
    writer: &mut W,
    ctx: &mut EmitCtx<'_>,
    max: i64,
    min: i64,
) -> Result<(), LaminaError> {
    for (limit, above) in [(max, true), (min, false)] {
        let keep = ctx.fresh_label("clamp");
        abi::load_imm(writer, T2, limit)?;
        let (first, second) = if above { (T2, A0) } else { (A0, T2) };
        writeln!(writer, "    slt {T1}, {first}, {second}")?;
        writeln!(writer, "    beq {T1}, {ZERO}, {keep}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    move {A0}, {T2}")?;
        writeln!(writer, "{keep}:")?;
    }
    Ok(())
}

/// Holds `$2` at or below an unsigned ceiling. The value cannot be negative, the
/// conversion that produced it being unsigned.
fn clamp_unsigned<W: Write>(
    writer: &mut W,
    ctx: &mut EmitCtx<'_>,
    max: i64,
) -> Result<(), LaminaError> {
    let keep = ctx.fresh_label("clampu");
    abi::load_imm(writer, T2, max)?;
    writeln!(writer, "    sltu {T1}, {T2}, {A0}")?;
    writeln!(writer, "    beq {T1}, {ZERO}, {keep}")?;
    writeln!(writer, "    nop")?;
    writeln!(writer, "    move {A0}, {T2}")?;
    writeln!(writer, "{keep}:")?;
    Ok(())
}

/// The byte and buffer intrinsics leave their result in `$2` like any other call.
fn store_intrinsic_result<W: Write>(
    writer: &mut W,
    ret: Option<&Register>,
    slots: &Slots,
) -> Result<(), LaminaError> {
    match ret {
        Some(dst) => store_result(writer, A0, dst, slots),
        None => Ok(()),
    }
}

fn emit_call<W: Write>(
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    if (name == "lamina.write" || name == "lamina.read") && args.len() == 2 {
        load_operand(&args[0], writer, slots, "$5")?;
        load_operand(&args[1], writer, slots, "$6")?;
        let writing = name == "lamina.write";
        writeln!(writer, "    li $4, {}", i32::from(writing))?;
        let done = ctx.fresh_label("io");
        if !writing {
            writeln!(writer, "    beq $6, {ZERO}, {done}")?;
            writeln!(writer, "    nop")?;
        }
        let libc = if writing { "write" } else { "read" };
        emit_direct_call(writer, libc, ctx.target_os)?;
        if writing {
            writeln!(writer, "    bgez {A0}, {done}")?;
            writeln!(writer, "    nop")?;
            writeln!(writer, "    li {A0}, -2")?;
            writeln!(writer, "{done}:")?;
        } else {
            let positive = ctx.fresh_label("io_pos");
            writeln!(writer, "    bgez {A0}, {positive}")?;
            writeln!(writer, "    nop")?;
            writeln!(writer, "    li {A0}, -2")?;
            writeln!(writer, "    j {done}")?;
            writeln!(writer, "    nop")?;
            writeln!(writer, "{positive}:")?;
            writeln!(writer, "    bne {A0}, {ZERO}, {done}")?;
            writeln!(writer, "    nop")?;
            writeln!(writer, "    li {A0}, -1")?;
            writeln!(writer, "{done}:")?;
        }
        return store_intrinsic_result(writer, ret, slots);
    }
    if name == "lamina.writebyte" && args.len() == 1 {
        let buffer = ctx.io_scratch;
        load_operand(&args[0], writer, slots, A0)?;
        abi::fp_store(writer, "sb", A0, buffer, T0)?;
        writeln!(writer, "    li $4, 1")?;
        abi::address(writer, "$5", FP, i64::from(buffer))?;
        writeln!(writer, "    li $6, 1")?;
        emit_direct_call(writer, "write", ctx.target_os)?;
        let done = ctx.fresh_label("writebyte");
        writeln!(writer, "    bgez {A0}, {done}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    li {A0}, -2")?;
        writeln!(writer, "{done}:")?;
        return store_intrinsic_result(writer, ret, slots);
    }
    if name == "lamina.readbyte" && args.is_empty() {
        let buffer = ctx.io_scratch;
        writeln!(writer, "    li $4, 0")?;
        abi::address(writer, "$5", FP, i64::from(buffer))?;
        writeln!(writer, "    li $6, 1")?;
        emit_direct_call(writer, "read", ctx.target_os)?;
        let one = ctx.fresh_label("readbyte_one");
        let other = ctx.fresh_label("readbyte_other");
        let done = ctx.fresh_label("readbyte_done");
        writeln!(writer, "    li {T1}, 1")?;
        writeln!(writer, "    beq {A0}, {T1}, {one}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    bne {A0}, {ZERO}, {other}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "    li {A0}, -1")?;
        writeln!(writer, "    j {done}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{other}:")?;
        writeln!(writer, "    li {A0}, -2")?;
        writeln!(writer, "    j {done}")?;
        writeln!(writer, "    nop")?;
        writeln!(writer, "{one}:")?;
        abi::fp_load(writer, "lbu", A0, buffer, T0)?;
        writeln!(writer, "{done}:")?;
        return store_intrinsic_result(writer, ret, slots);
    }

    let kinds = ctx.signatures.call_kinds(name, args);
    let result = ctx.signatures.result(name);
    let result_slot = match ret {
        Some(Register::Virtual(vreg)) => slots.get(vreg).copied(),
        _ => None,
    };
    let hidden = match result {
        Some(aggregate) if aggregate.size > MAX_REGISTER_RETURN => Some(
            result_slot.ok_or_else(|| unsupported("mips64 aggregate result has no frame slot"))?,
        ),
        _ => None,
    };
    pass_arguments(
        writer,
        args,
        &kinds,
        slots,
        hidden,
        ctx.signatures.variadic_from(name),
    )?;
    emit_direct_call(writer, name, ctx.target_os)?;
    match (ret, result) {
        (_, Some(aggregate)) if aggregate.size > MAX_REGISTER_RETURN => {}
        (Some(_), Some(aggregate)) => {
            let offset = result_slot
                .ok_or_else(|| unsupported("mips64 aggregate result has no frame slot"))?;
            abi::fp_store(writer, "sd", A0, offset, T0)?;
            if aggregate.words() > 1 {
                abi::fp_store(writer, "sd", A1, offset + WORD, T0)?;
            }
        }
        (Some(dst), None) => {
            let float = ctx.signatures.float_returns.get(name).copied();
            store_call_result(writer, dst, float, slots)?;
        }
        (None, _) => {}
    }
    Ok(())
}
