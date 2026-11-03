//! MIR to s390x assembly for Linux under the z/Architecture ELF ABI.
//!
//! The first big-endian target. Every virtual register lives in a frame slot holding a
//! full doubleword, so an integer narrower than 64 bits is kept extended in the slot's
//! low-order half and a float narrower than 64 bits in its high-order half — the two
//! classes do not share a slot's bytes here the way they do on a little-endian target.
//!
//! The emitter moves operands through `%r2`, `%r3`, `%r1`, `%r10`, `%r12` and `%r13`
//! and writes results back, so no register is allocated. `%r11` is the frame pointer,
//! `%r15` the stack pointer, and `%r0` is never used: it reads as zero as a base or
//! index register.

mod abi;
mod atomic;
mod bit;
mod inline_asm;
mod operand;
mod overflow;
#[cfg(test)]
mod tests;
mod variadic;

use crate::mir_codegen::capability::{CapabilitySet, CodegenCapability};
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::{
    FunctionEmissionPlan, assign_stack_slots, compile_functions_parallel, emit_debug_file,
    emit_global_data, emit_global_directives, emit_gnu_stack_note, emit_symbol_visibility,
    external_names, integer_conversion_with_width, lamina_to_codegen_error, parallel_codegen_error,
    promoted_argument_type,
};
use crate::{
    error::LaminaError,
    mir::{
        Aggregate, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Instruction as MirInst, IntBinOp,
        IntCmpOp, MirType, Module as MirModule, Operand, Register, ScalarType,
        instruction::AddressMode,
    },
    mir_codegen::{
        CodegenError, MirCodegenSettings, RegallocStrategy, validate_module_call_parameters,
    },
};
use abi::{
    A0, A1, ARG_FPRS, ARG_GPRS, ArgKind, ArgPlace, DIV_HI, DIV_LO, FA0, FA1, FP,
    RELATIVE_LONG_ALIGN, SP, T0, T1, T2, TRAP, WORD,
};
use lamina_mir::{SourceLoc, TlsModel};
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};
use operand::{
    Slots, load_float_operand, load_operand, phys, slot, store_float, store_register, unsupported,
};
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
};

use lamina_codegen::abi::common_call_stub;

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
        self.result(callee).is_some_and(|a| a.size > WORD as usize)
    }

    /// Types of a call's arguments as the convention sees them. A matching prototype
    /// gives them; past a variadic callee's fixed parameters a float still travels in
    /// a floating-point register, which is what the z/Architecture ABI does.
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
    /// Whether the frame pointer was rounded down from the entry stack pointer.
    aligned: bool,
    signatures: &'a Signatures,
    /// Set when this function returns a float, which leaves in `%f0`.
    ret_float: Option<MirType>,
    /// Set when this function returns an aggregate.
    ret_aggregate: Option<Aggregate>,
    /// Frame offset of the hidden result pointer, when the aggregate is too wide to
    /// come back in a register.
    sret: Option<i32>,
    /// Set in a variadic function: where its `va_list` operations read from.
    va: Option<variadic::VaArea>,
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
    CapabilitySet::standard_native()
        .with(CodegenCapability::IndirectCalls)
        .with(CodegenCapability::AggregateArguments)
}

pub fn generate_mir_s390x_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Linux {
        return Err(unsupported(format!(
            "s390x code generation supports Linux only, not {target_os}"
        )));
    }
    if settings.regalloc != RegallocStrategy::Incremental {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for s390x".to_string(),
        ));
    }
    validate_module_call_parameters(module, TargetArchitecture::S390x)?;
    let signatures = Arc::new(Signatures::from_module(module));
    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_function(name, func, os, settings_arc.as_ref(), signatures.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    writeln!(writer, ".machinemode zarch")?;
    writeln!(writer, ".data")?;
    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings, module)?;
    emit_global_directives(writer, module, str::to_string)?;
    emit_symbol_visibility(writer, module, target_os, str::to_string)?;
    for name in external_names(module) {
        writeln!(writer, ".extern {name}")?;
    }
    for result in results {
        writer.write_all(&result.assembly)?;
    }
    emit_global_data(
        writer,
        &relative_long_module(module),
        target_os,
        str::to_string,
    )?;
    emit_gnu_stack_note(writer, target_os)?;
    Ok(())
}

/// `larl`, which every symbol reference here goes through, encodes a halfword
/// displacement and so can only name an even address. A byte-aligned global placed on
/// an odd one is loaded off by one, silently, so every global is given at least
/// halfword alignment. Only the globals are rewritten; the data emitter reads nothing
/// else from the module.
fn relative_long_module(module: &MirModule) -> MirModule {
    let mut copy = MirModule::new(module.name.clone());
    copy.relocation = module.relocation;
    copy.source_files = module.source_files.clone();
    for (name, global) in &module.globals {
        let mut global = global.clone();
        global.align = global.align.max(RELATIVE_LONG_ALIGN);
        copy.globals.insert(name.clone(), global);
    }
    copy
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
    writeln!(output, "{func_name}:").map_err(io)?;

    let (slots, _, _) = assign_stack_slots(func, abi::slot_offset);
    let locals = slots
        .values()
        .map(|offset| (-offset) as usize)
        .max()
        .unwrap_or(0);
    let outgoing = outgoing_area(func, signatures);
    let aligned = func
        .stack_alignments
        .values()
        .any(|align| *align > abi::WORD as usize);
    let hidden = func
        .sig
        .ret_aggregate
        .is_some_and(|a| a.size > abi::WORD as usize);
    let sret = hidden.then(|| -(locals as i32) - abi::WORD);
    let locals = locals + usize::from(hidden) * abi::WORD as usize;
    let va = func.sig.variadic.then(|| {
        let kinds: Vec<ArgKind> = func
            .sig
            .params
            .iter()
            .map(|p| ArgKind {
                ty: p.ty,
                aggregate: p.aggregate,
            })
            .collect();
        let first_gpr = usize::from(hidden);
        let places = abi::classify_kinds(&kinds, first_gpr);
        variadic::VaArea::new(-(locals as i32) - variadic::SAVE_BYTES, &places, first_gpr)
    });
    let locals = locals + usize::from(func.sig.variadic) * variadic::SAVE_BYTES as usize;
    let locals = if aligned {
        locals + abi::ALIGN_SLACK
    } else {
        locals
    };
    abi::prologue(&mut output, locals, outgoing, aligned).map_err(io)?;
    if let Some(offset) = sret {
        abi::fp_store(&mut output, "stg", ARG_GPRS[0], offset, T0).map_err(io)?;
    }
    if let Some(area) = &va {
        variadic::spill_save_area(&mut output, area).map_err(lamina_to_codegen)?;
    }
    store_incoming_arguments(&mut output, func, &slots, aligned, usize::from(hidden))
        .map_err(lamina_to_codegen)?;

    let mut ctx = EmitCtx {
        func_name,
        target_os,
        settings,
        loc: None,
        label_seq: 0,
        io_scratch: abi::io_scratch_offset(locals),
        aligned,
        signatures,
        ret_float: func
            .sig
            .ret_ty
            .filter(|ty| func.sig.ret_aggregate.is_none() && ty.is_float()),
        ret_aggregate: func.sig.ret_aggregate,
        sret,
        va,
    };
    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    jg {}", ctx.block_label(&func.entry)).map_err(io)?;
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
    aligned: bool,
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
    for ((param, kind), place) in func.sig.params.iter().zip(&kinds).zip(places) {
        let Register::Virtual(vreg) = &param.reg else {
            continue;
        };
        let Some(&offset) = slots.get(vreg) else {
            continue;
        };
        let single = param.ty.size_bytes(PointerWidth::Bits64) == 4;
        match place {
            ArgPlace::Gpr(bank) if kind.by_reference() => {
                let words = kind.aggregate.map_or(0, |a| a.words());
                copy_words(writer, ARG_GPRS[bank], words, offset)?;
            }
            ArgPlace::Gpr(bank) => {
                abi::fp_store(writer, "stg", ARG_GPRS[bank], offset, T0)?;
            }
            ArgPlace::Fpr(bank) => {
                store_float(ARG_FPRS[bank], vreg, writer, slots, single)?;
            }
            ArgPlace::Stack(index) => {
                let at = i64::from(abi::incoming_stack_offset(index));
                abi::memory(writer, "lg", T0, abi::entry_base(aligned), at, T0)?;
                if kind.by_reference() {
                    let words = kind.aggregate.map_or(0, |a| a.words());
                    copy_words(writer, T0, words, offset)?;
                } else {
                    abi::fp_store(writer, "stg", T0, offset, T1)?;
                }
            }
        }
    }
    Ok(())
}

/// Copies `words` doublewords from the address in `from` into the frame slot block
/// that starts at `offset`.
fn copy_words<W: Write>(
    writer: &mut W,
    from: &str,
    words: usize,
    offset: i32,
) -> Result<(), LaminaError> {
    for index in 0..words {
        let at = index as i64 * i64::from(WORD);
        abi::memory(writer, "lg", T1, from, at, T2)?;
        abi::fp_store(writer, "stg", T1, offset + index as i32 * WORD, T2)?;
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
                let (invert, taken, other) =
                    plan.branch(func, index, next, true_target, false_target);
                writeln!(writer, "    {} {A0}, {A1}", compare_mnemonic(unsigned))?;
                let mask = condition_mask(*op, invert);
                writeln!(writer, "    brcl {mask}, {}", ctx.block_label(taken))?;
                if let Some(other) = other {
                    writeln!(writer, "    jg {}", ctx.block_label(other))?;
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

fn is_unsigned(op: IntCmpOp) -> bool {
    matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    )
}

fn compare_mnemonic(unsigned: bool) -> &'static str {
    if unsigned { "clgr" } else { "cgr" }
}

/// Branch mask selecting the condition codes the comparison leaves: 8 for equal, 4 for
/// first operand low, 2 for high, 1 for unordered. Inverting takes the complement
/// within the three codes a fixed-point comparison can set.
fn condition_mask(op: IntCmpOp, invert: bool) -> u8 {
    let mask = match op {
        IntCmpOp::Eq => 8,
        IntCmpOp::Ne => 6,
        IntCmpOp::SLt | IntCmpOp::ULt => 4,
        IntCmpOp::SLe | IntCmpOp::ULe => 12,
        IntCmpOp::SGt | IntCmpOp::UGt => 2,
        IntCmpOp::SGe | IntCmpOp::UGe => 10,
    };
    if invert { 14 - mask } else { mask }
}

/// Branch mask for a floating-point comparison, where code 3 means unordered. An
/// ordering holds only when the operands compare, so unordered is left out of every
/// mask but the one for inequality.
fn float_mask(op: FloatCmpOp) -> u8 {
    match op {
        FloatCmpOp::Eq => 8,
        FloatCmpOp::Ne => 7,
        FloatCmpOp::Lt => 4,
        FloatCmpOp::Le => 12,
        FloatCmpOp::Gt => 2,
        FloatCmpOp::Ge => 10,
    }
}

/// Sign- or zero-extends a narrow integer in `reg` to the 64-bit form values take.
fn extend_integer<W: Write>(
    writer: &mut W,
    reg: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    let mnemonic = match (ty, signed) {
        (MirType::Scalar(ScalarType::I1 | ScalarType::I8), true) => "lgbr",
        (MirType::Scalar(ScalarType::I1 | ScalarType::I8), false) => "llgcr",
        (MirType::Scalar(ScalarType::I16), true) => "lghr",
        (MirType::Scalar(ScalarType::I16), false) => "llghr",
        (MirType::Scalar(ScalarType::I32), true) => "lgfr",
        (MirType::Scalar(ScalarType::I32), false) => "llgfr",
        _ => return Ok(()),
    };
    writeln!(writer, "    {mnemonic} {reg}, {reg}")?;
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
                writeln!(writer, "    lgr {name}, {src}")?;
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
            abi::fp_load(writer, "lg", dst, offset, dst)?;
        }
        Register::Physical(physical) => {
            writeln!(writer, "    lgr {dst}, {}", phys(physical.name))?;
        }
    }
    Ok(())
}

fn call_target(name: &str, target_os: TargetOperatingSystem) -> String {
    common_call_stub(name, target_os).unwrap_or_else(|| name.to_string())
}

/// Places the arguments of a call in the outgoing area the prologue already reserved.
/// The stack words go first, through `%r1`, so the register loads that follow can use
/// their own destinations as scratch.
fn pass_arguments<W: Write>(
    writer: &mut W,
    args: &[Operand],
    kinds: &[ArgKind],
    slots: &Slots,
    hidden: Option<i32>,
) -> Result<(), LaminaError> {
    let first_gpr = usize::from(hidden.is_some());
    let places = abi::classify_kinds(kinds, first_gpr);
    let mut copy_at = abi::copy_base(kinds, first_gpr);
    let mut copies = vec![None; kinds.len()];
    for (index, kind) in kinds.iter().enumerate() {
        if kind.by_reference() {
            copies[index] = Some(copy_at);
            copy_at += kind.aggregate.map_or(0, |a| a.words()) as i64 * i64::from(WORD);
        }
    }
    for ((arg, kind), at) in args.iter().zip(kinds).zip(&copies) {
        let Some(at) = *at else { continue };
        load_operand(arg, writer, slots, T0)?;
        for word in 0..kind.aggregate.map_or(0, |a| a.words()) {
            let step = word as i64 * i64::from(WORD);
            abi::memory(writer, "lg", T1, T0, step, T2)?;
            abi::memory(writer, "stg", T1, SP, at + step, T2)?;
        }
    }
    for ((index, arg), place) in args.iter().enumerate().zip(&places) {
        let ArgPlace::Stack(at) = *place else {
            continue;
        };
        match copies[index] {
            Some(copy) => abi::address(writer, T0, SP, copy)?,
            None => load_operand(arg, writer, slots, T0)?,
        }
        abi::memory(writer, "stg", T0, SP, abi::outgoing_stack_offset(at), T1)?;
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
        let ArgPlace::Gpr(bank) = *place else {
            continue;
        };
        match copies[index] {
            Some(copy) => abi::address(writer, ARG_GPRS[bank], SP, copy)?,
            None if kind.aggregate.is_some() => {
                load_operand(arg, writer, slots, ARG_GPRS[bank])?;
                abi::memory(writer, "lg", ARG_GPRS[bank], ARG_GPRS[bank], 0, T0)?;
            }
            None => load_operand(arg, writer, slots, ARG_GPRS[bank])?,
        }
    }
    Ok(())
}

/// Stores a call's result into its destination: `%r2`, or `%f0` when the callee returns
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

/// Moves a 64-bit integer register into a floating-point register. A single occupies
/// the left half of the register, so its bits shift up first.
fn int_to_fpr<W: Write>(writer: &mut W, dst: &str, src: &str, single: bool) -> io::Result<()> {
    if single {
        writeln!(writer, "    sllg {T0}, {src}, 32")?;
        writeln!(writer, "    ldgr {dst}, {T0}")
    } else {
        writeln!(writer, "    ldgr {dst}, {src}")
    }
}

/// The reverse: a single lands right-aligned and zero-extended, the way a 32-bit
/// integer is held.
fn fpr_to_int<W: Write>(writer: &mut W, dst: &str, src: &str, single: bool) -> io::Result<()> {
    writeln!(writer, "    lgdr {dst}, {src}")?;
    if single {
        writeln!(writer, "    srlg {dst}, {dst}, 32")?;
    }
    Ok(())
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
        writeln!(writer, "    .loc {} {} {}", loc.file, loc.line, loc.column)?;
    }
    match inst {
        MirInst::ThreadLocalAddr { dst, symbol, model } => {
            emit_thread_local(writer, slots, ctx, dst, symbol, *model)?;
        }
        MirInst::IntOverflow {
            op,
            ty,
            signed,
            dst,
            lhs,
            rhs,
        } => {
            overflow::emit(writer, slots, ctx, op, ty, *signed, dst, lhs, rhs)?;
        }
        MirInst::Copy { ty, dst, src } => {
            if ty.is_vector() {
                return Err(unsupported("s390x vector Copy"));
            }
            copy_value(writer, slots, ty, ty, dst, src)?;
        }
        MirInst::Bitcast { from, to, dst, src } => {
            if from.is_vector() || to.is_vector() {
                return Err(unsupported("s390x vector Bitcast"));
            }
            copy_value(writer, slots, from, to, dst, src)?;
        }
        MirInst::SymbolAddress { dst, symbol } => {
            writeln!(writer, "    larl {A0}, {symbol}")?;
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
                    "s390x lea base {base:?} is not a virtual register"
                )));
            };
            let base_slot = slot(slots, base_vreg)?;
            abi::fp_load(writer, "lg", A0, base_slot, A0)?;
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
            writeln!(writer, "    {} {A0}, {A1}", compare_mnemonic(unsigned))?;
            let done = ctx.fresh_label("cmp");
            writeln!(writer, "    lghi {A0}, 1")?;
            writeln!(writer, "    brcl {}, {done}", condition_mask(*op, false))?;
            writeln!(writer, "    lghi {A0}, 0")?;
            writeln!(writer, "{done}:")?;
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
            let done = ctx.fresh_label("select");
            writeln!(writer, "    ltgr {T0}, {T0}")?;
            writeln!(writer, "    brcl 6, {done}")?;
            if float {
                writeln!(writer, "    ldr {FA0}, {FA1}")?;
            } else {
                writeln!(writer, "    lgr {A0}, {A1}")?;
            }
            writeln!(writer, "{done}:")?;
            match (dst, float) {
                (Register::Virtual(vreg), true) => store_float(FA0, vreg, writer, slots, single)?,
                (dst, _) => store_result(writer, A0, dst, slots)?,
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported(
                    "s390x load supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("lgb", false),
                MirType::Scalar(ScalarType::I16) => ("lgh", false),
                MirType::Scalar(ScalarType::I32) => ("lgf", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("lg", false)
                }
                MirType::Scalar(ScalarType::F32) => ("ley", true),
                MirType::Scalar(ScalarType::F64) => ("ldy", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "s390x load of vector type {ty} is not implemented"
                    )));
                }
            };
            load_base(writer, base, slots, T0)?;
            let reg = if float { FA0 } else { A0 };
            abi::memory(writer, mnemonic, reg, T0, i64::from(*offset), T1)?;
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
            attrs: _,
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported(
                    "s390x store supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("stcy", false),
                MirType::Scalar(ScalarType::I16) => ("sthy", false),
                MirType::Scalar(ScalarType::I32) => ("sty", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("stg", false)
                }
                MirType::Scalar(ScalarType::F32) => ("stey", true),
                MirType::Scalar(ScalarType::F64) => ("stdy", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "s390x store of vector type {ty} is not implemented"
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
                        abi::fp_load(writer, "lg", A1, offset, A1)?;
                        for index in 0..aggregate.words() {
                            let at = index as i64 * i64::from(WORD);
                            abi::memory(writer, "lg", T1, T0, at, T2)?;
                            abi::memory(writer, "stg", T1, A1, at, T2)?;
                        }
                        writeln!(writer, "    lgr {A0}, {A1}")?;
                    }
                    (Some(_), None) => {
                        load_operand(val, writer, slots, T0)?;
                        abi::memory(writer, "lg", A0, T0, 0, T1)?;
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
            abi::epilogue(writer, ctx.aligned, None)?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    jg {}", ctx.block_label(target))?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            load_operand(&Operand::Register(cond.clone()), writer, slots, T0)?;
            writeln!(writer, "    ltgr {T0}, {T0}")?;
            writeln!(writer, "    brcl 6, {}", ctx.block_label(true_target))?;
            writeln!(writer, "    jg {}", ctx.block_label(false_target))?;
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            load_operand(&Operand::Register(value.clone()), writer, slots, A0)?;
            for (case_val, case_label) in cases {
                abi::load_imm(writer, T0, *case_val)?;
                writeln!(writer, "    cgr {A0}, {T0}")?;
                writeln!(writer, "    brcl 8, {}", ctx.block_label(case_label))?;
            }
            writeln!(writer, "    jg {}", ctx.block_label(default))?;
        }
        MirInst::Call { name, args, ret } => {
            emit_call(name, args, ret.as_ref(), writer, slots, ctx)?;
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed: _,
            ret,
            ret_ty,
        } => {
            let kinds: Vec<ArgKind> = arg_types.iter().copied().map(ArgKind::scalar).collect();
            pass_arguments(writer, args, &kinds, slots, None)?;
            load_operand(target, writer, slots, T1)?;
            writeln!(writer, "    basr %r14, {T1}")?;
            if let Some(ret) = ret {
                store_call_result(writer, ret, ret_ty.filter(MirType::is_float), slots)?;
            }
        }
        // The epilogue restores every callee-saved register, the frame pointer the
        // argument values are addressed from among them, so the arguments cannot be
        // placed after it. Lowered as an ordinary call and a return instead, which
        // costs a frame but never returns a wrong answer.
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
                    "s390x tail call passing or returning an aggregate",
                ));
            }
            emit_call(name, args, None, writer, slots, ctx)?;
            abi::epilogue(writer, ctx.aligned, None)?;
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(lhs, writer, slots, FA0, single)?;
            load_float_operand(rhs, writer, slots, FA1, single)?;
            let mnemonic = match (op, single) {
                (FloatBinOp::FAdd, false) => "adbr",
                (FloatBinOp::FSub, false) => "sdbr",
                (FloatBinOp::FMul, false) => "mdbr",
                (FloatBinOp::FDiv, false) => "ddbr",
                (FloatBinOp::FAdd, true) => "aebr",
                (FloatBinOp::FSub, true) => "sebr",
                (FloatBinOp::FMul, true) => "meebr",
                (FloatBinOp::FDiv, true) => "debr",
            };
            writeln!(writer, "    {mnemonic} {FA0}, {FA1}")?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(src, writer, slots, FA0, single)?;
            let mnemonic = match (op, single) {
                (FloatUnOp::FNeg, false) => "lcdbr",
                (FloatUnOp::FNeg, true) => "lcebr",
                (FloatUnOp::FSqrt, false) => "sqdbr",
                (FloatUnOp::FSqrt, true) => "sqebr",
            };
            writeln!(writer, "    {mnemonic} {FA0}, {FA0}")?;
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
            load_float_operand(lhs, writer, slots, FA0, single)?;
            load_float_operand(rhs, writer, slots, FA1, single)?;
            let compare = if single { "cebr" } else { "cdbr" };
            writeln!(writer, "    {compare} {FA0}, {FA1}")?;
            let done = ctx.fresh_label("fcmp");
            writeln!(writer, "    lghi {A0}, 1")?;
            writeln!(writer, "    brcl {}, {done}", float_mask(*op))?;
            writeln!(writer, "    lghi {A0}, 0")?;
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
            let single = from.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(src, writer, slots, FA0, single)?;
            let compare = if single { "cebr" } else { "cdbr" };
            let convert = match (single, signed) {
                (false, true) => format!("cgdbr {A0}, 5, {FA0}"),
                (true, true) => format!("cgebr {A0}, 5, {FA0}"),
                (false, false) => format!("clgdbr {A0}, 5, {FA0}, 0"),
                (true, false) => format!("clgebr {A0}, 5, {FA0}, 0"),
            };
            let nan = ctx.fresh_label("f2i_nan");
            let done = ctx.fresh_label("f2i");
            writeln!(writer, "    {compare} {FA0}, {FA0}")?;
            writeln!(writer, "    brcl 1, {nan}")?;
            writeln!(writer, "    {convert}")?;
            writeln!(writer, "    jg {done}")?;
            writeln!(writer, "{nan}:")?;
            writeln!(writer, "    lghi {A0}, 0")?;
            writeln!(writer, "{done}:")?;
            let bits = to.size_bytes(PointerWidth::Bits64) * 8;
            if bits < 64 {
                if *signed {
                    clamp(writer, ctx, (1i64 << (bits - 1)) - 1, -(1i64 << (bits - 1)))?;
                } else {
                    clamp_unsigned(writer, ctx, (1i64 << bits) - 1)?;
                }
                extend_integer(writer, A0, to, *signed)?;
            }
            store_result(writer, A0, dst, slots)?;
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let from_bits = from.size_bytes(PointerWidth::Bits64) * 8;
            load_operand(src, writer, slots, A0)?;
            if from_bits < 64 {
                extend_integer(writer, A0, from, *signed)?;
            }
            let single = to.size_bytes(PointerWidth::Bits64) == 4;
            let convert = match (single, signed) {
                (false, true) => format!("cdgbr {FA0}, {A0}"),
                (true, true) => format!("cegbr {FA0}, {A0}"),
                (false, false) => format!("cdlgbr {FA0}, 0, {A0}, 0"),
                (true, false) => format!("celgbr {FA0}, 0, {A0}, 0"),
            };
            writeln!(writer, "    {convert}")?;
            if let Register::Virtual(vreg) = dst {
                store_float(FA0, vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let from_single = from.size_bytes(PointerWidth::Bits64) == 4;
            let to_single = to.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(src, writer, slots, FA0, from_single)?;
            match (from_single, to_single) {
                (true, false) => writeln!(writer, "    ldebr {FA0}, {FA0}")?,
                (false, true) => writeln!(writer, "    ledbr {FA0}, {FA0}")?,
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
                "s390x received an integer conversion that was not lowered",
            ));
        }
        MirInst::IntUnary { op, ty, dst, src } => {
            bit::emit_int_unary(writer, slots, *op, ty, dst, src)?;
        }
        MirInst::IntRotate {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            bit::emit_int_rotate(writer, slots, *op, ty, dst, lhs, rhs)?;
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported("s390x vector operations"));
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            atomic::emit(inst, writer, slots, ctx)?;
        }
        MirInst::InlineAsm { .. } => {
            inline_asm::emit(inst, writer, slots)?;
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            variadic::emit(inst, writer, slots, ctx)?;
        }
    }
    Ok(())
}

/// The address of a thread-local symbol in the running thread's block.
///
/// The thread pointer is two halves read out of access registers `%a0` and `%a1`.
/// Under local exec the offset from it is a link-time constant, which only a data
/// relocation can hold, so it goes in a literal the load reaches relative to the
/// program counter; under initial exec that literal is the GOT entry the linker
/// builds instead.
fn emit_thread_local<W: Write>(
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
    dst: &Register,
    symbol: &str,
    model: TlsModel,
) -> Result<(), LaminaError> {
    writeln!(writer, "    ear {A0}, %a0")?;
    writeln!(writer, "    sllg {A0}, {A0}, 32")?;
    writeln!(writer, "    ear {A0}, %a1")?;
    match model {
        TlsModel::LocalExec => {
            let literal = ctx.fresh_label("tls");
            writeln!(writer, "    .section .rodata.cst8,\"aM\",@progbits,8")?;
            writeln!(writer, "    .align 8")?;
            writeln!(writer, "{literal}:")?;
            writeln!(writer, "    .quad {symbol}@NTPOFF")?;
            writeln!(writer, "    .text")?;
            writeln!(writer, "    lgrl {T0}, {literal}")?;
        }
        TlsModel::InitialExec => {
            writeln!(writer, "    larl {T0}, {symbol}@INDNTPOFF")?;
            writeln!(writer, "    lg {T0}, 0({T0})")?;
        }
    }
    writeln!(writer, "    agr {A0}, {T0}")?;
    store_result(writer, A0, dst, slots)
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
        writeln!(writer, "    ltgr {T0}, {A1}")?;
        writeln!(writer, "    brcl 6, {ok}")?;
        writeln!(writer, "{TRAP}")?;
        writeln!(writer, "{ok}:")?;
    }
    if overflow {
        let ok = ctx.fresh_label("nooverflow");
        let minimum = ty.signed_integer(
            1i64 << (ty.size_bytes(PointerWidth::Bits64) * 8 - 1),
            PointerWidth::Bits64,
        );
        writeln!(writer, "    lghi {T0}, -1")?;
        writeln!(writer, "    cgr {A1}, {T0}")?;
        writeln!(writer, "    brcl 6, {ok}")?;
        abi::load_imm(writer, T0, minimum)?;
        writeln!(writer, "    cgr {A0}, {T0}")?;
        writeln!(writer, "    brcl 6, {ok}")?;
        writeln!(writer, "{TRAP}")?;
        writeln!(writer, "{ok}:")?;
    }
    if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr)
        && ty.size_bytes(PointerWidth::Bits64) < 8
    {
        let mask = ty.size_bytes(PointerWidth::Bits64) * 8 - 1;
        writeln!(writer, "    lghi {T0}, {mask}")?;
        writeln!(writer, "    ngr {A1}, {T0}")?;
    }
    match op {
        IntBinOp::SDiv | IntBinOp::SRem => {
            writeln!(writer, "    lgr {DIV_LO}, {A0}")?;
            writeln!(writer, "    dsgr {DIV_HI}, {A1}")?;
            let result = if matches!(op, IntBinOp::SDiv) {
                DIV_LO
            } else {
                DIV_HI
            };
            writeln!(writer, "    lgr {A0}, {result}")?;
        }
        IntBinOp::UDiv | IntBinOp::URem => {
            writeln!(writer, "    lghi {DIV_HI}, 0")?;
            writeln!(writer, "    lgr {DIV_LO}, {A0}")?;
            writeln!(writer, "    dlgr {DIV_HI}, {A1}")?;
            let result = if matches!(op, IntBinOp::UDiv) {
                DIV_LO
            } else {
                DIV_HI
            };
            writeln!(writer, "    lgr {A0}, {result}")?;
        }
        IntBinOp::Shl => writeln!(writer, "    sllg {A0}, {A0}, 0({A1})")?,
        IntBinOp::LShr => writeln!(writer, "    srlg {A0}, {A0}, 0({A1})")?,
        IntBinOp::AShr => writeln!(writer, "    srag {A0}, {A0}, 0({A1})")?,
        other => {
            let mnemonic = match other {
                IntBinOp::Add => "agr",
                IntBinOp::Sub => "sgr",
                IntBinOp::Mul => "msgr",
                IntBinOp::And => "ngr",
                IntBinOp::Or => "ogr",
                IntBinOp::Xor => "xgr",
                _ => unreachable!("handled above"),
            };
            writeln!(writer, "    {mnemonic} {A0}, {A1}")?;
        }
    }
    extend_integer(writer, A0, ty, true)?;
    store_result(writer, A0, dst, slots)
}

/// Holds `%r2` within a signed range.
fn clamp<W: Write>(
    writer: &mut W,
    ctx: &mut EmitCtx<'_>,
    max: i64,
    min: i64,
) -> Result<(), LaminaError> {
    for (limit, mask) in [(max, 2u8), (min, 4u8)] {
        let keep = ctx.fresh_label("clamp");
        abi::load_imm(writer, T2, limit)?;
        writeln!(writer, "    cgr {A0}, {T2}")?;
        writeln!(writer, "    brcl {}, {keep}", 14 - mask)?;
        writeln!(writer, "    lgr {A0}, {T2}")?;
        writeln!(writer, "{keep}:")?;
    }
    Ok(())
}

/// Holds `%r2` at or below an unsigned ceiling. The value cannot be negative, the
/// conversion that produced it being unsigned.
fn clamp_unsigned<W: Write>(
    writer: &mut W,
    ctx: &mut EmitCtx<'_>,
    max: i64,
) -> Result<(), LaminaError> {
    let keep = ctx.fresh_label("clampu");
    abi::load_imm(writer, T2, max)?;
    writeln!(writer, "    clgr {A0}, {T2}")?;
    writeln!(writer, "    brcl 12, {keep}")?;
    writeln!(writer, "    lgr {A0}, {T2}")?;
    writeln!(writer, "{keep}:")?;
    Ok(())
}

/// The byte and buffer intrinsics leave their result in `%r2` like any other call.
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
        load_operand(&args[0], writer, slots, "%r3")?;
        load_operand(&args[1], writer, slots, "%r4")?;
        let writing = name == "lamina.write";
        writeln!(writer, "    lghi %r2, {}", i32::from(writing))?;
        let done = ctx.fresh_label("io");
        if !writing {
            writeln!(writer, "    ltgr {T0}, %r4")?;
            writeln!(writer, "    brcl 8, {done}")?;
        }
        let libc = if writing { "write" } else { "read" };
        writeln!(
            writer,
            "    brasl %r14, {}",
            call_target(libc, ctx.target_os)
        )?;
        writeln!(writer, "    ltgr {A0}, {A0}")?;
        if writing {
            writeln!(writer, "    brcl 10, {done}")?;
            writeln!(writer, "    lghi {A0}, -2")?;
            writeln!(writer, "{done}:")?;
        } else {
            let positive = ctx.fresh_label("io_pos");
            writeln!(writer, "    brcl 10, {positive}")?;
            writeln!(writer, "    lghi {A0}, -2")?;
            writeln!(writer, "    jg {done}")?;
            writeln!(writer, "{positive}:")?;
            writeln!(writer, "    brcl 6, {done}")?;
            writeln!(writer, "    lghi {A0}, -1")?;
            writeln!(writer, "{done}:")?;
        }
        return store_intrinsic_result(writer, ret, slots);
    }
    if name == "lamina.writebyte" && args.len() == 1 {
        let buffer = ctx.io_scratch;
        load_operand(&args[0], writer, slots, A0)?;
        abi::fp_store(writer, "stcy", A0, buffer, T0)?;
        writeln!(writer, "    lghi %r2, 1")?;
        abi::address(writer, "%r3", FP, i64::from(buffer))?;
        writeln!(writer, "    lghi %r4, 1")?;
        writeln!(
            writer,
            "    brasl %r14, {}",
            call_target("write", ctx.target_os)
        )?;
        let done = ctx.fresh_label("writebyte");
        writeln!(writer, "    ltgr {A0}, {A0}")?;
        writeln!(writer, "    brcl 10, {done}")?;
        writeln!(writer, "    lghi {A0}, -2")?;
        writeln!(writer, "{done}:")?;
        return store_intrinsic_result(writer, ret, slots);
    }
    if name == "lamina.readbyte" && args.is_empty() {
        let buffer = ctx.io_scratch;
        writeln!(writer, "    lghi %r2, 0")?;
        abi::address(writer, "%r3", FP, i64::from(buffer))?;
        writeln!(writer, "    lghi %r4, 1")?;
        writeln!(
            writer,
            "    brasl %r14, {}",
            call_target("read", ctx.target_os)
        )?;
        let one = ctx.fresh_label("readbyte_one");
        let other = ctx.fresh_label("readbyte_other");
        let done = ctx.fresh_label("readbyte_done");
        writeln!(writer, "    lghi {T1}, 1")?;
        writeln!(writer, "    cgr {A0}, {T1}")?;
        writeln!(writer, "    brcl 8, {one}")?;
        writeln!(writer, "    ltgr {A0}, {A0}")?;
        writeln!(writer, "    brcl 6, {other}")?;
        writeln!(writer, "    lghi {A0}, -1")?;
        writeln!(writer, "    jg {done}")?;
        writeln!(writer, "{other}:")?;
        writeln!(writer, "    lghi {A0}, -2")?;
        writeln!(writer, "    jg {done}")?;
        writeln!(writer, "{one}:")?;
        abi::fp_load(writer, "llgc", A0, buffer, T0)?;
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
        Some(aggregate) if aggregate.size > WORD as usize => Some(
            result_slot.ok_or_else(|| unsupported("s390x aggregate result has no frame slot"))?,
        ),
        _ => None,
    };
    pass_arguments(writer, args, &kinds, slots, hidden)?;
    writeln!(
        writer,
        "    brasl %r14, {}",
        call_target(name, ctx.target_os)
    )?;
    match (ret, result) {
        (_, Some(aggregate)) if aggregate.size > WORD as usize => {}
        (Some(dst), Some(_)) => {
            let offset = result_slot
                .ok_or_else(|| unsupported("s390x aggregate result has no frame slot"))?;
            let _ = dst;
            abi::fp_store(writer, "stg", A0, offset, T0)?;
        }
        (Some(dst), None) => {
            let float = ctx.signatures.float_returns.get(name).copied();
            store_call_result(writer, dst, float, slots)?;
        }
        (None, _) => {}
    }
    Ok(())
}
