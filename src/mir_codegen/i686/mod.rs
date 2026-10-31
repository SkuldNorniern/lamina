//! MIR to i686 assembly for Linux under the cdecl calling convention.
//!
//! Every virtual register lives in an eight-byte frame slot and the emitter moves
//! values through `%eax`, `%ecx` and `%edx`, so none of the four callee-saved
//! registers is ever touched and the prologue has nothing to save. With six general
//! registers and `%ebp` pinned to the frame, keeping values in memory is what the
//! target affords; global allocation is rejected by the dispatcher.
//!
//! cdecl passes every argument on the stack, lowest address first, and the caller
//! releases the area. The outgoing area is rounded to sixteen bytes so `%esp` is
//! aligned at each `call`, matching what gcc emits for `-m32` on Linux.
//!
//! 64-bit integers are refused rather than split: the LUMIR pass that legalizes them
//! into register pairs is what will make them work, and a quiet truncation here would
//! be worse than a failed compile.

mod abi;
mod operand;
#[cfg(test)]
mod tests;

use crate::mir_codegen::capability::{CapabilitySet, CodegenCapability};
use crate::mir_codegen::common::{
    FunctionEmissionPlan, assign_stack_slots_framed, compile_functions_parallel, emit_debug_file,
    emit_global_data, emit_global_directives, emit_gnu_stack_note, emit_source_loc,
    emit_symbol_visibility, external_names, integer_conversion_with_width, lamina_to_codegen_error,
    parallel_codegen_error,
};
use crate::mir_codegen::common::{
    reject_thread_local_globals, unsupported_inline_asm, unsupported_overflow_flag,
    unsupported_thread_local,
};
use crate::{
    error::LaminaError,
    mir::{
        Aggregate, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Instruction as MirInst, IntBinOp,
        IntCmpOp, IntRotOp, IntUnOp, MirType, Module as MirModule, Operand, Register,
        RelocationModel, ScalarType, instruction::AddressMode, instruction::Immediate,
        register::RegisterClass,
    },
    mir_codegen::{
        CodegenError, MirCodegenSettings, RegallocStrategy, validate_module_call_parameters,
    },
};
use abi::{INCOMING_BASE, PTR, WORD};
use lamina_codegen::abi::common_call_stub;
use lamina_mir::SourceLoc;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use operand::{
    EAX, ECX, EDX, Slots, byte_reg, is_single, load_operand, pop_float, push_float, slot,
    store_register, unsupported, word_reg,
};
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::Arc,
};

/// Capabilities declared by this backend; see `CapabilitySet::for_architecture`.
pub(crate) fn capabilities() -> CapabilitySet {
    let mut caps = CapabilitySet::standard_native();
    caps.remove(&CodegenCapability::InlineAssembly);
    caps.with(CodegenCapability::IndirectCalls)
        .with(CodegenCapability::AggregateArguments)
}

/// Refuses a type this backend cannot hold in one 32-bit register.
fn reject_wide(ty: &MirType, what: &str) -> Result<(), LaminaError> {
    if ty.is_vector() {
        return Err(unsupported(format!("i686 {what} on the vector type {ty}")));
    }
    if matches!(ty, MirType::Scalar(ScalarType::I64)) {
        return Err(unsupported(format!(
            "i686 {what} on i64: 64-bit integers are not yet legalized into 32-bit register pairs"
        )));
    }
    Ok(())
}

/// cdecl returns a struct through a hidden pointer and a legalized 64-bit integer in
/// `%edx:%eax`, so only the former takes the sret slot.
fn returns_through_memory(aggregate: Aggregate, func_name: &str) -> Result<bool, LaminaError> {
    if aggregate.is_wide_scalar() && aggregate.size > 2 * WORD {
        return Err(unsupported(format!(
            "i686 return of @{func_name}: a legalized integer of {} bytes does not fit in %edx:%eax",
            aggregate.size
        )));
    }
    Ok(aggregate.is_composite())
}

/// Type a variadic or unprototyped argument reaches the callee as: a float as a
/// double, everything else as a machine word.
fn promoted(arg: &Operand) -> MirType {
    let float = match arg {
        Operand::Register(Register::Virtual(vreg)) => vreg.class == RegisterClass::Fpr,
        Operand::Immediate(Immediate::F32(_) | Immediate::F64(_)) => true,
        _ => false,
    };
    if float {
        MirType::Scalar(ScalarType::F64)
    } else {
        MirType::Scalar(ScalarType::I32)
    }
}

/// What a call site needs to know about its callee.
struct CallAbi {
    types: Vec<MirType>,
    aggregates: Vec<Option<usize>>,
    ret_aggregate: Option<Aggregate>,
    ret_ty: Option<MirType>,
}

/// Prototypes the caller side reads: parameter types and aggregates, variadic arity
/// and the return shape.
struct Signatures {
    params: HashMap<String, Vec<MirType>>,
    param_aggregates: HashMap<String, Vec<Option<usize>>>,
    variadic: HashMap<String, usize>,
    returns: HashMap<String, MirType>,
    return_aggregates: HashMap<String, Aggregate>,
}

impl Signatures {
    fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| (name.clone(), func.sig.params.iter().map(|p| p.ty).collect()))
                .collect(),
            param_aggregates: module
                .functions
                .iter()
                .map(|(name, func)| {
                    (
                        name.clone(),
                        func.sig
                            .params
                            .iter()
                            .map(|p| p.aggregate.map(|a| a.size))
                            .collect(),
                    )
                })
                .collect(),
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, func)| (name.clone(), func.sig.params.len()))
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_ty?)))
                .collect(),
            return_aggregates: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
                .collect(),
        }
    }

    fn call_abi(&self, callee: &str, args: &[Operand]) -> CallAbi {
        let fixed = self.variadic.get(callee).copied();
        let declared = self.params.get(callee);
        let matches_prototype =
            fixed.is_none() && declared.is_some_and(|types| types.len() == args.len());
        let types: Vec<MirType> = if matches_prototype {
            declared.cloned().unwrap_or_default()
        } else {
            let fixed = fixed.unwrap_or(0);
            args.iter()
                .enumerate()
                .map(|(index, arg)| match declared {
                    Some(types) if index < fixed && index < types.len() => types[index],
                    _ => promoted(arg),
                })
                .collect()
        };
        let aggregates = if matches_prototype {
            self.param_aggregates
                .get(callee)
                .cloned()
                .unwrap_or_else(|| vec![None; args.len()])
        } else {
            vec![None; args.len()]
        };
        CallAbi {
            types,
            aggregates,
            ret_aggregate: self.return_aggregates.get(callee).copied(),
            ret_ty: self.returns.get(callee).copied(),
        }
    }
}

struct EmitCtx<'a> {
    func_name: &'a str,
    target_os: TargetOperatingSystem,
    settings: &'a MirCodegenSettings,
    label_seq: u32,
    signatures: &'a Signatures,
    ret_float: Option<MirType>,
    /// Frame offset of the hidden pointer a struct return is written through, with
    /// the size of that aggregate.
    sret: Option<(i32, usize)>,
    /// Whether the return value is a legalized 64-bit integer, which leaves in
    /// `%edx:%eax`.
    wide_return: bool,
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

pub fn generate_mir_i686_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    reject_thread_local_globals(module, "i686")?;
    if target_os != TargetOperatingSystem::Linux {
        return Err(unsupported(format!(
            "i686 code generation supports Linux only, not {target_os}"
        )));
    }
    if settings.regalloc != RegallocStrategy::Incremental {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for i686".to_string(),
        ));
    }
    if module.relocation != RelocationModel::Static {
        return Err(unsupported(format!(
            "i686 addresses symbols directly; the {:?} relocation model needs GOT-relative addressing, which is not implemented",
            module.relocation
        )));
    }
    validate_module_call_parameters(module, TargetArchitecture::I686)?;

    let signatures = Arc::new(Signatures::from_module(module));
    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_function(name, func, os, settings_arc.as_ref(), signatures.as_ref())
        })
        .map_err(parallel_codegen_error)?;

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

fn compile_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, CodegenError> {
    let io = |e: io::Error| CodegenError::InvalidCodegenOptions(format!("IO error: {e}"));
    if func.sig.naked {
        return Err(CodegenError::UnsupportedFeature(format!(
            "i686 does not emit naked function @{func_name}"
        )));
    }
    if func.sig.variadic {
        return Err(CodegenError::UnsupportedFeature(format!(
            "i686 does not define the variadic function @{func_name}; reading a va_list is not implemented"
        )));
    }
    let mut output = Vec::new();
    writeln!(output, "{func_name}:").map_err(io)?;

    let (slots, _, mut total) =
        assign_stack_slots_framed(func, abi::slot_offset, |_| true, abi::FRAME_RESIDUE);
    let wide_return = func
        .sig
        .ret_aggregate
        .is_some_and(|aggregate| aggregate.is_wide_scalar());
    let sret = match func.sig.ret_aggregate {
        Some(aggregate)
            if returns_through_memory(aggregate, func_name).map_err(lamina_to_codegen_error)? =>
        {
            let offset = abi::slot_offset(total);
            total += 1;
            Some((offset, aggregate.size))
        }
        _ => None,
    };
    let frame = abi::frame_bytes(total * abi::SLOT as usize);
    abi::prologue(&mut output, frame).map_err(io)?;
    store_incoming_arguments(&mut output, func, &slots, sret).map_err(lamina_to_codegen_error)?;

    let mut ctx = EmitCtx {
        func_name,
        target_os,
        settings,
        label_seq: 0,
        signatures,
        ret_float: func.sig.ret_ty.filter(MirType::is_float),
        sret,
        wide_return,
    };
    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    jmp {}", ctx.block_label(&func.entry)).map_err(io)?;
    }
    for (position, &index) in plan.order.iter().enumerate() {
        let next = plan
            .order
            .get(position + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        emit_block(func, index, next, &plan, &mut output, &slots, &mut ctx)
            .map_err(lamina_to_codegen_error)?;
    }
    Ok(output)
}

/// Copies `bytes` from `src_offset(src_base)` to `dst_offset(dst_base)` through
/// `scratch`, which must differ from both bases.
fn copy_bytes<W: Write>(
    writer: &mut W,
    (src_base, src_offset): (&str, i32),
    (dst_base, dst_offset): (&str, i32),
    bytes: usize,
    scratch: &str,
) -> Result<(), LaminaError> {
    let mut done = 0usize;
    while bytes - done >= 4 {
        let at = done as i32;
        writeln!(
            writer,
            "    movl {}({src_base}), {scratch}",
            src_offset + at
        )?;
        writeln!(
            writer,
            "    movl {scratch}, {}({dst_base})",
            dst_offset + at
        )?;
        done += 4;
    }
    if bytes - done >= 2 {
        let at = done as i32;
        let half = word_reg(scratch)?;
        writeln!(writer, "    movw {}({src_base}), {half}", src_offset + at)?;
        writeln!(writer, "    movw {half}, {}({dst_base})", dst_offset + at)?;
        done += 2;
    }
    if bytes - done == 1 {
        let at = done as i32;
        let low = byte_reg(scratch)?;
        writeln!(writer, "    movb {}({src_base}), {low}", src_offset + at)?;
        writeln!(writer, "    movb {low}, {}({dst_base})", dst_offset + at)?;
    }
    Ok(())
}

/// Copies each incoming argument from the caller's area into its frame slot.
fn store_incoming_arguments<W: Write>(
    writer: &mut W,
    func: &Function,
    slots: &Slots,
    sret: Option<(i32, usize)>,
) -> Result<(), LaminaError> {
    if let Some((offset, _)) = sret {
        writeln!(writer, "    movl {INCOMING_BASE}(%ebp), {EAX}")?;
        writeln!(writer, "    movl {EAX}, {offset}(%ebp)")?;
    }
    let types: Vec<MirType> = func.sig.params.iter().map(|p| p.ty).collect();
    let aggregates: Vec<Option<usize>> = func
        .sig
        .params
        .iter()
        .map(|p| p.aggregate.map(|a| a.size))
        .collect();
    let places = abi::classify(&types, &aggregates, sret.is_some());
    for (param, place) in func.sig.params.iter().zip(places) {
        let Register::Virtual(vreg) = &param.reg else {
            return Err(unsupported(
                "i686 does not take a parameter in a pinned physical register",
            ));
        };
        let Some(&dst) = slots.get(vreg) else {
            continue;
        };
        let from = INCOMING_BASE + place.offset as i32;
        match param.aggregate {
            Some(aggregate) => {
                copy_bytes(writer, ("%ebp", from), ("%ebp", dst), aggregate.size, EAX)?;
            }
            None if param.ty.is_float() => {
                copy_bytes(
                    writer,
                    ("%ebp", from),
                    ("%ebp", dst),
                    param.ty.size_bytes(PTR),
                    EAX,
                )?;
            }
            None => {
                reject_wide(&param.ty, &format!("parameter of @{}", func.sig.name))?;
                writeln!(writer, "    movl {from}(%ebp), {EAX}")?;
                extend_in_place(writer, EAX, &param.ty, true)?;
                writeln!(writer, "    movl {EAX}, {dst}(%ebp)")?;
            }
        }
    }
    Ok(())
}

/// Rewrites a narrow integer in `reg` to its canonical form: `i1` as 0 or 1, and a
/// narrower integer sign-extended whatever its declared signedness, which the
/// unsigned consumers undo before use.
fn extend_in_place<W: Write>(
    writer: &mut W,
    reg: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    match ty {
        MirType::Scalar(ScalarType::I1) => {
            writeln!(writer, "    andl $1, {reg}")?;
        }
        MirType::Scalar(ScalarType::I8) => {
            let low = byte_reg(reg)?;
            let op = if signed { "movsbl" } else { "movzbl" };
            writeln!(writer, "    {op} {low}, {reg}")?;
        }
        MirType::Scalar(ScalarType::I16) => {
            let half = word_reg(reg)?;
            let op = if signed { "movswl" } else { "movzwl" };
            writeln!(writer, "    {op} {half}, {reg}")?;
        }
        _ => {}
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
    if plan.aligned_headers.contains(&index) {
        writeln!(writer, "    .p2align 4")?;
    }
    writeln!(writer, "{}:", ctx.block_label(&block.label))?;
    let mut position = 0;
    while position < block.instructions.len() {
        let inst = &block.instructions[position];
        if plan.comparisons.contains(&(index, position))
            && let (
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
            emit_compare(op, ty, lhs, rhs, writer, slots)?;
            let (invert, taken, other) = plan.branch(func, index, next, true_target, false_target);
            let suffix = if invert {
                inverted_suffix(*op)
            } else {
                condition_suffix(*op)
            };
            writeln!(writer, "    j{suffix} {}", ctx.block_label(taken))?;
            if let Some(other) = other {
                writeln!(writer, "    jmp {}", ctx.block_label(other))?;
            }
            position += 2;
            continue;
        }
        if !matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
            emit_instruction(&inst.instr, inst.loc, writer, slots, ctx)?;
        }
        position += 1;
    }
    Ok(())
}

fn unsigned_compare(op: IntCmpOp) -> bool {
    matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    )
}

fn condition_suffix(op: IntCmpOp) -> &'static str {
    match op {
        IntCmpOp::Eq => "e",
        IntCmpOp::Ne => "ne",
        IntCmpOp::SLt => "l",
        IntCmpOp::SLe => "le",
        IntCmpOp::SGt => "g",
        IntCmpOp::SGe => "ge",
        IntCmpOp::ULt => "b",
        IntCmpOp::ULe => "be",
        IntCmpOp::UGt => "a",
        IntCmpOp::UGe => "ae",
    }
}

fn inverted_suffix(op: IntCmpOp) -> &'static str {
    match op {
        IntCmpOp::Eq => "ne",
        IntCmpOp::Ne => "e",
        IntCmpOp::SLt => "ge",
        IntCmpOp::SLe => "g",
        IntCmpOp::SGt => "le",
        IntCmpOp::SGe => "l",
        IntCmpOp::ULt => "ae",
        IntCmpOp::ULe => "a",
        IntCmpOp::UGt => "be",
        IntCmpOp::UGe => "b",
    }
}

/// Leaves the flags of `lhs <=> rhs` set, with both operands widened the way the
/// comparison reads them.
fn emit_compare<W: Write>(
    op: &IntCmpOp,
    ty: &MirType,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    reject_wide(ty, "comparison")?;
    let signed = !unsigned_compare(*op);
    load_operand(lhs, writer, slots, EAX)?;
    extend_in_place(writer, EAX, ty, signed)?;
    load_operand(rhs, writer, slots, ECX)?;
    extend_in_place(writer, ECX, ty, signed)?;
    writeln!(writer, "    cmpl {ECX}, {EAX}")?;
    Ok(())
}

/// The address an `AddressMode` names, with its base (and index) loaded into the
/// scratch registers it reserves.
fn emit_address<W: Write>(
    addr: &AddressMode,
    writer: &mut W,
    slots: &Slots,
) -> Result<String, LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            load_operand(&Operand::Register(base.clone()), writer, slots, ECX)?;
            Ok(format!("{offset}({ECX})"))
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => {
            load_operand(&Operand::Register(base.clone()), writer, slots, ECX)?;
            load_operand(&Operand::Register(index.clone()), writer, slots, EDX)?;
            Ok(format!("{offset}({ECX},{EDX},{scale})"))
        }
    }
}

fn call_target(name: &str, target_os: TargetOperatingSystem) -> String {
    common_call_stub(name, target_os).unwrap_or_else(|| name.to_string())
}

fn emit_instruction<W: Write>(
    inst: &MirInst,
    loc: Option<SourceLoc>,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    emit_source_loc(writer, ctx.settings, loc)?;
    if let MirInst::IntConvert { from, to, .. } = inst {
        reject_wide(from, "integer conversion")?;
        reject_wide(to, "integer conversion")?;
        if let Some(operations) =
            integer_conversion_with_width(inst, MirType::Scalar(ScalarType::I32), PTR)
        {
            for operation in operations {
                emit_instruction(&operation, None, writer, slots, ctx)?;
            }
            return Ok(());
        }
    }
    match inst {
        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => emit_copy(ty, dst, src, writer, slots)?,
        MirInst::ThreadLocalAddr { .. } => {
            return Err(unsupported_thread_local("i686"));
        }
        MirInst::IntOverflow { .. } => {
            return Err(unsupported_overflow_flag("i686"));
        }
        MirInst::SymbolAddress { dst, symbol } => {
            writeln!(writer, "    movl ${symbol}, {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::StackAddr { dst, slot: stack } => {
            let offset = slot(slots, stack)?;
            writeln!(writer, "    leal {offset}(%ebp), {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::Lea { dst, base, offset } => {
            load_operand(&Operand::Register(base.clone()), writer, slots, ECX)?;
            writeln!(writer, "    leal {offset}({ECX}), {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::IntBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => emit_int_binary(op, ty, dst, lhs, rhs, writer, slots)?,
        MirInst::IntCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            emit_compare(op, ty, lhs, rhs, writer, slots)?;
            let suffix = condition_suffix(*op);
            let low = byte_reg(EAX)?;
            writeln!(writer, "    set{suffix} {low}")?;
            writeln!(writer, "    movzbl {low}, {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::IntUnary { op, ty, dst, src } => emit_int_unary(op, ty, dst, src, writer, slots)?,
        MirInst::IntRotate {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => emit_int_rotate(op, ty, dst, lhs, rhs, writer, slots)?,
        MirInst::Select {
            ty,
            dst,
            cond,
            true_val,
            false_val,
        } => {
            reject_wide(ty, "select")?;
            if ty.is_float() {
                return Err(unsupported("i686 select on a floating-point value"));
            }
            load_operand(false_val, writer, slots, ECX)?;
            load_operand(true_val, writer, slots, EAX)?;
            load_operand(&Operand::Register(cond.clone()), writer, slots, EDX)?;
            writeln!(writer, "    testl {EDX}, {EDX}")?;
            writeln!(writer, "    cmovel {ECX}, {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::Load { ty, dst, addr, .. } => {
            reject_wide(ty, "load")?;
            let address = emit_address(addr, writer, slots)?;
            match ty {
                MirType::Scalar(ScalarType::F32) => {
                    writeln!(writer, "    flds {address}")?;
                    pop_float(dst, writer, slots, true)?;
                }
                MirType::Scalar(ScalarType::F64) => {
                    writeln!(writer, "    fldl {address}")?;
                    pop_float(dst, writer, slots, false)?;
                }
                MirType::Scalar(ScalarType::I1) => {
                    writeln!(writer, "    movzbl {address}, {EAX}")?;
                    writeln!(writer, "    andl $1, {EAX}")?;
                    store_register(EAX, dst, writer, slots)?;
                }
                MirType::Scalar(ScalarType::I8) => {
                    writeln!(writer, "    movsbl {address}, {EAX}")?;
                    store_register(EAX, dst, writer, slots)?;
                }
                MirType::Scalar(ScalarType::I16) => {
                    writeln!(writer, "    movswl {address}, {EAX}")?;
                    store_register(EAX, dst, writer, slots)?;
                }
                _ => {
                    writeln!(writer, "    movl {address}, {EAX}")?;
                    store_register(EAX, dst, writer, slots)?;
                }
            }
        }
        MirInst::Store { ty, src, addr, .. } => {
            reject_wide(ty, "store")?;
            if ty.is_float() {
                let single = is_single(ty);
                push_float(src, writer, slots, single)?;
                let address = emit_address(addr, writer, slots)?;
                let store = if single { "fstps" } else { "fstpl" };
                writeln!(writer, "    {store} {address}")?;
            } else {
                let address = emit_address(addr, writer, slots)?;
                load_operand(src, writer, slots, EAX)?;
                match ty.size_bytes(PTR) {
                    1 => writeln!(writer, "    movb {}, {address}", byte_reg(EAX)?)?,
                    2 => writeln!(writer, "    movw {}, {address}", word_reg(EAX)?)?,
                    _ => writeln!(writer, "    movl {EAX}, {address}")?,
                }
            }
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    jmp {}", ctx.block_label(target))?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            load_operand(&Operand::Register(cond.clone()), writer, slots, EAX)?;
            writeln!(writer, "    testl {EAX}, {EAX}")?;
            writeln!(writer, "    jne {}", ctx.block_label(true_target))?;
            writeln!(writer, "    jmp {}", ctx.block_label(false_target))?;
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            load_operand(&Operand::Register(value.clone()), writer, slots, EAX)?;
            for (case, label) in cases {
                let case = i32::try_from(*case).map_err(|_| {
                    unsupported(format!("i686 switch case {case} does not fit a word"))
                })?;
                writeln!(writer, "    cmpl ${case}, {EAX}")?;
                writeln!(writer, "    je {}", ctx.block_label(label))?;
            }
            writeln!(writer, "    jmp {}", ctx.block_label(default))?;
        }
        MirInst::Call { name, args, ret } => {
            if emit_builtin(name, args, ret.as_ref(), writer, slots, ctx)? {
                return Ok(());
            }
            let call = ctx.signatures.call_abi(name, args);
            let target = call_target(name, ctx.target_os);
            emit_call(
                &call,
                name,
                args,
                ret.as_ref(),
                Some(&target),
                writer,
                slots,
            )?;
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed,
            ret,
            ret_ty,
        } => {
            let mut types = arg_types.clone();
            if let Some(fixed) = variadic_fixed {
                for (index, ty) in types.iter_mut().enumerate() {
                    if index >= *fixed && is_single(ty) {
                        *ty = MirType::Scalar(ScalarType::F64);
                    }
                }
            }
            let call = CallAbi {
                aggregates: vec![None; args.len()],
                types,
                ret_aggregate: None,
                ret_ty: *ret_ty,
            };
            load_operand(target, writer, slots, EAX)?;
            // The pointer is parked in an area of its own, sized so `%esp` keeps its
            // alignment, and released with the argument area after the call.
            writeln!(writer, "    subl ${INDIRECT_AREA}, %esp")?;
            writeln!(writer, "    movl {EAX}, (%esp)")?;
            emit_call(
                &call,
                "the called pointer",
                args,
                ret.as_ref(),
                None,
                writer,
                slots,
            )?;
        }
        MirInst::TailCall { name, args } => {
            if ctx.sret.is_some() {
                return Err(unsupported(format!(
                    "i686 does not tail call @{name} from a function that returns an aggregate"
                )));
            }
            let call = ctx.signatures.call_abi(name, args);
            let target = call_target(name, ctx.target_os);
            emit_call(&call, name, args, None, Some(&target), writer, slots)?;
            abi::epilogue(writer, 0)?;
        }
        MirInst::Ret { value } => {
            if let Some(value) = value {
                match (ctx.sret, ctx.ret_float) {
                    _ if ctx.wide_return => {
                        load_operand(value, writer, slots, ECX)?;
                        writeln!(writer, "    movl 4({ECX}), {EDX}")?;
                        writeln!(writer, "    movl ({ECX}), {EAX}")?;
                    }
                    (Some((offset, size)), _) => {
                        load_operand(value, writer, slots, EAX)?;
                        writeln!(writer, "    movl {offset}(%ebp), {ECX}")?;
                        copy_bytes(writer, (EAX, 0), (ECX, 0), size, EDX)?;
                        writeln!(writer, "    movl {ECX}, {EAX}")?;
                    }
                    (None, Some(ty)) => push_float(value, writer, slots, is_single(&ty))?,
                    (None, None) => {
                        if let Some(ty) = func_return_type(ctx) {
                            reject_wide(&ty, "return value")?;
                        }
                        load_operand(value, writer, slots, EAX)?;
                    }
                }
            }
            abi::epilogue(writer, ctx.sret.map_or(0, |_| WORD))?;
        }
        MirInst::Unreachable => writeln!(writer, "    ud2")?,
        MirInst::Comment { text } => writeln!(writer, "    # {text}")?,
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {}
        MirInst::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let single = is_single(ty);
            // The right operand is parked in memory and the left stays on the x87
            // stack, so the operation is the `f<op> mem` form, which reads
            // `st(0) <op> mem` in either assembler syntax. The two-register pop
            // forms name the subtraction and the division the other way round in
            // AT&T syntax than in Intel's, and nothing here depends on that.
            writeln!(writer, "    subl $16, %esp")?;
            push_float(rhs, writer, slots, single)?;
            let (store, suffix) = if single {
                ("fstps", "s")
            } else {
                ("fstpl", "l")
            };
            writeln!(writer, "    {store} (%esp)")?;
            push_float(lhs, writer, slots, single)?;
            let mnemonic = match op {
                FloatBinOp::FAdd => "fadd",
                FloatBinOp::FSub => "fsub",
                FloatBinOp::FMul => "fmul",
                FloatBinOp::FDiv => "fdiv",
            };
            writeln!(writer, "    {mnemonic}{suffix} (%esp)")?;
            writeln!(writer, "    addl $16, %esp")?;
            pop_float(dst, writer, slots, single)?;
        }
        MirInst::FloatUnary { op, ty, dst, src } => {
            let single = is_single(ty);
            push_float(src, writer, slots, single)?;
            let mnemonic = match op {
                FloatUnOp::FNeg => "fchs",
                FloatUnOp::FSqrt => "fsqrt",
            };
            writeln!(writer, "    {mnemonic}")?;
            pop_float(dst, writer, slots, single)?;
        }
        MirInst::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let single = is_single(ty);
            // fucomip compares st(0) with st(1) and reports in the integer flags, so
            // the right operand goes on first and the comparison reads as unsigned.
            push_float(rhs, writer, slots, single)?;
            push_float(lhs, writer, slots, single)?;
            writeln!(writer, "    fucomip %st(1), %st")?;
            writeln!(writer, "    fstp %st(0)")?;
            let low = byte_reg(EAX)?;
            // An unordered compare sets ZF, PF and CF, so equality must also rule
            // parity out and inequality must let it through.
            match op {
                FloatCmpOp::Eq => {
                    writeln!(writer, "    sete {low}")?;
                    writeln!(writer, "    setnp {}", byte_reg(ECX)?)?;
                    writeln!(writer, "    andb {}, {low}", byte_reg(ECX)?)?;
                }
                FloatCmpOp::Ne => {
                    writeln!(writer, "    setne {low}")?;
                    writeln!(writer, "    setp {}", byte_reg(ECX)?)?;
                    writeln!(writer, "    orb {}, {low}", byte_reg(ECX)?)?;
                }
                FloatCmpOp::Gt => writeln!(writer, "    seta {low}")?,
                FloatCmpOp::Ge => writeln!(writer, "    setae {low}")?,
                FloatCmpOp::Lt => writeln!(writer, "    setb {low}")?,
                FloatCmpOp::Le => writeln!(writer, "    setbe {low}")?,
            }
            writeln!(writer, "    movzbl {low}, {EAX}")?;
            store_register(EAX, dst, writer, slots)?;
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            push_float(src, writer, slots, is_single(from))?;
            pop_float(dst, writer, slots, is_single(to))?;
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            reject_wide(from, "integer to float conversion")?;
            if !signed {
                return Err(unsupported(
                    "i686 unsigned integer to float conversion is not implemented",
                ));
            }
            load_operand(src, writer, slots, EAX)?;
            extend_in_place(writer, EAX, from, true)?;
            writeln!(writer, "    pushl {EAX}")?;
            writeln!(writer, "    fildl (%esp)")?;
            writeln!(writer, "    addl $4, %esp")?;
            pop_float(dst, writer, slots, is_single(to))?;
        }
        MirInst::FloatToInt { .. } => {
            return Err(unsupported(
                "i686 float to integer conversion is not implemented: x87 reports an out-of-range or NaN operand as the integer indefinite, not as the saturating result the contract names",
            ));
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported("i686 vector operations are not implemented"));
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            return Err(unsupported(
                "i686 atomic operations and fences are not implemented",
            ));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported_inline_asm("i686"));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(unsupported(
                "i686 variadic argument access is not implemented",
            ));
        }
        MirInst::IntConvert { .. } => {
            return Err(unsupported(
                "i686 received an integer conversion that was not lowered",
            ));
        }
    }
    Ok(())
}

fn func_return_type(ctx: &EmitCtx<'_>) -> Option<MirType> {
    ctx.signatures.returns.get(ctx.func_name).copied()
}

fn emit_copy<W: Write>(
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    if ty.is_vector() {
        return Err(unsupported(format!("i686 copy of the vector type {ty}")));
    }
    if matches!(ty, MirType::Scalar(ScalarType::F64)) {
        let Register::Virtual(target) = dst else {
            return Err(unsupported("i686 copies a double only into a frame slot"));
        };
        let to = slot(slots, target)?;
        match src {
            Operand::Immediate(Immediate::F64(value)) => {
                let bits = value.to_bits();
                writeln!(writer, "    movl ${}, {to}(%ebp)", bits as u32 as i32)?;
                writeln!(
                    writer,
                    "    movl ${}, {}(%ebp)",
                    (bits >> 32) as u32 as i32,
                    to + 4
                )?;
            }
            Operand::Register(Register::Virtual(source)) => {
                let from = slot(slots, source)?;
                copy_bytes(writer, ("%ebp", from), ("%ebp", to), 8, EAX)?;
            }
            other => {
                return Err(unsupported(format!(
                    "i686 cannot copy {other} into a double"
                )));
            }
        }
        return Ok(());
    }
    reject_wide(ty, "copy")?;
    load_operand(src, writer, slots, EAX)?;
    extend_in_place(writer, EAX, ty, true)?;
    store_register(EAX, dst, writer, slots)
}

fn emit_int_binary<W: Write>(
    op: &IntBinOp,
    ty: &MirType,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    reject_wide(ty, &format!("{op:?}"))?;
    let bits = ty.size_bytes(PTR) * 8;
    let result = match op {
        IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr => {
            load_operand(lhs, writer, slots, EAX)?;
            extend_in_place(writer, EAX, ty, matches!(op, IntBinOp::AShr))?;
            load_operand(rhs, writer, slots, ECX)?;
            if bits < 32 {
                writeln!(writer, "    andl ${}, {ECX}", bits - 1)?;
            }
            let mnemonic = match op {
                IntBinOp::Shl => "shll",
                IntBinOp::LShr => "shrl",
                _ => "sarl",
            };
            writeln!(writer, "    {mnemonic} %cl, {EAX}")?;
            EAX
        }
        IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::UDiv | IntBinOp::URem => {
            let signed = matches!(op, IntBinOp::SDiv | IntBinOp::SRem);
            load_operand(lhs, writer, slots, EAX)?;
            extend_in_place(writer, EAX, ty, signed)?;
            load_operand(rhs, writer, slots, ECX)?;
            extend_in_place(writer, ECX, ty, signed)?;
            if signed && bits < 32 {
                let (_, overflow) =
                    op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant(), PTR);
                if overflow {
                    let minimum = ty.signed_integer(1i64 << (bits - 1), PTR);
                    writeln!(
                        writer,
                        "    cmpl $-1, {ECX}\n    jne 2f\n    cmpl ${minimum}, {EAX}\n    jne 2f\n    ud2\n2:"
                    )?;
                }
            }
            if signed {
                writeln!(writer, "    cltd")?;
                writeln!(writer, "    idivl {ECX}")?;
            } else {
                writeln!(writer, "    xorl {EDX}, {EDX}")?;
                writeln!(writer, "    divl {ECX}")?;
            }
            if matches!(op, IntBinOp::SRem | IntBinOp::URem) {
                EDX
            } else {
                EAX
            }
        }
        _ => {
            load_operand(lhs, writer, slots, EAX)?;
            load_operand(rhs, writer, slots, ECX)?;
            let mnemonic = match op {
                IntBinOp::Add => "addl",
                IntBinOp::Sub => "subl",
                IntBinOp::Mul => "imull",
                IntBinOp::And => "andl",
                IntBinOp::Or => "orl",
                IntBinOp::Xor => "xorl",
                other => {
                    return Err(unsupported(format!("i686 integer operation {other:?}")));
                }
            };
            writeln!(writer, "    {mnemonic} {ECX}, {EAX}")?;
            EAX
        }
    };
    extend_in_place(writer, result, ty, true)?;
    store_register(result, dst, writer, slots)
}

fn emit_int_unary<W: Write>(
    op: &IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    reject_wide(ty, &format!("{op:?}"))?;
    let bits = ty.size_bytes(PTR) * 8;
    load_operand(src, writer, slots, EAX)?;
    extend_in_place(writer, EAX, ty, false)?;
    match op {
        IntUnOp::Bswap => match bits {
            32 => writeln!(writer, "    bswapl {EAX}")?,
            16 => writeln!(writer, "    rolw $8, {}", word_reg(EAX)?)?,
            _ => {}
        },
        IntUnOp::Popcount => {
            writeln!(writer, "    movl {EAX}, {ECX}")?;
            writeln!(writer, "    shrl $1, {ECX}")?;
            writeln!(writer, "    andl $1431655765, {ECX}")?;
            writeln!(writer, "    subl {ECX}, {EAX}")?;
            writeln!(writer, "    movl {EAX}, {ECX}")?;
            writeln!(writer, "    andl $858993459, {ECX}")?;
            writeln!(writer, "    shrl $2, {EAX}")?;
            writeln!(writer, "    andl $858993459, {EAX}")?;
            writeln!(writer, "    addl {ECX}, {EAX}")?;
            writeln!(writer, "    movl {EAX}, {ECX}")?;
            writeln!(writer, "    shrl $4, {ECX}")?;
            writeln!(writer, "    addl {ECX}, {EAX}")?;
            writeln!(writer, "    andl $252645135, {EAX}")?;
            writeln!(writer, "    imull $16843009, {EAX}, {EAX}")?;
            writeln!(writer, "    shrl $24, {EAX}")?;
        }
        IntUnOp::Ctlz => {
            // bsr gives the index of the highest set bit, counted from bit 0, so the
            // leading zeros within `bits` are `bits - 1 - index`. It leaves the
            // destination alone when the operand is zero, which the moved-in default
            // covers.
            writeln!(writer, "    movl $-1, {ECX}")?;
            writeln!(writer, "    bsrl {EAX}, {EDX}")?;
            writeln!(writer, "    cmovel {ECX}, {EDX}")?;
            writeln!(writer, "    movl ${}, {EAX}", bits - 1)?;
            writeln!(writer, "    subl {EDX}, {EAX}")?;
        }
        IntUnOp::Cttz => {
            writeln!(writer, "    movl ${bits}, {ECX}")?;
            writeln!(writer, "    bsfl {EAX}, {EDX}")?;
            writeln!(writer, "    cmovel {ECX}, {EDX}")?;
            writeln!(writer, "    movl {EDX}, {EAX}")?;
        }
    }
    extend_in_place(writer, EAX, ty, true)?;
    store_register(EAX, dst, writer, slots)
}

fn emit_int_rotate<W: Write>(
    op: &IntRotOp,
    ty: &MirType,
    dst: &Register,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    reject_wide(ty, &format!("{op:?}"))?;
    let bits = ty.size_bytes(PTR) * 8;
    load_operand(lhs, writer, slots, EAX)?;
    extend_in_place(writer, EAX, ty, false)?;
    load_operand(rhs, writer, slots, ECX)?;
    writeln!(writer, "    andl ${}, {ECX}", bits - 1)?;
    let (mnemonic, target) = match (op, bits) {
        (IntRotOp::Rotl, 32) => ("roll", EAX.to_string()),
        (IntRotOp::Rotr, 32) => ("rorl", EAX.to_string()),
        (IntRotOp::Rotl, 16) => ("rolw", word_reg(EAX)?.to_string()),
        (IntRotOp::Rotr, 16) => ("rorw", word_reg(EAX)?.to_string()),
        (IntRotOp::Rotl, 8) => ("rolb", byte_reg(EAX)?.to_string()),
        (IntRotOp::Rotr, 8) => ("rorb", byte_reg(EAX)?.to_string()),
        _ => return Err(unsupported(format!("i686 {op:?} on a {bits}-bit value"))),
    };
    writeln!(writer, "    {mnemonic} %cl, {target}")?;
    extend_in_place(writer, EAX, ty, true)?;
    store_register(EAX, dst, writer, slots)
}

/// Bytes reserved below the argument area for the target of an indirect call. Only
/// one word is needed; the rest is padding that keeps `%esp` aligned for the call.
const INDIRECT_AREA: usize = TargetArchitecture::I686.stack_alignment_at_call();

/// Places a call's arguments, calls `target` and takes the result. With no `target`
/// the address is already parked in the word below the argument area, which this
/// releases along with it.
fn emit_call<W: Write>(
    call: &CallAbi,
    callee: &str,
    args: &[Operand],
    ret: Option<&Register>,
    target: Option<&str>,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    let wide_result = match call.ret_aggregate {
        Some(aggregate) => !returns_through_memory(aggregate, callee)?,
        None => false,
    };
    let sret_buffer = match (call.ret_aggregate, ret) {
        (Some(_), _) if wide_result => None,
        (Some(_), Some(Register::Virtual(vreg))) => Some(slot(slots, vreg)?),
        (Some(_), _) => {
            return Err(unsupported(
                "i686 call returning an aggregate has no frame slot to write it into",
            ));
        }
        (None, _) => None,
    };
    let places = abi::classify(&call.types, &call.aggregates, sret_buffer.is_some());
    let space = abi::outgoing_bytes(&places, sret_buffer.is_some());
    if space > 0 {
        writeln!(writer, "    subl ${space}, %esp")?;
    }
    for (index, arg) in args.iter().enumerate() {
        let place = places[index];
        let at = place.offset as i32;
        match call.aggregates.get(index).copied().flatten() {
            Some(size) => {
                load_operand(arg, writer, slots, ECX)?;
                copy_bytes(writer, (ECX, 0), ("%esp", at), size, EAX)?;
            }
            None if call.types[index].is_float() => {
                let single = is_single(&call.types[index]);
                push_float(arg, writer, slots, single)?;
                let store = if single { "fstps" } else { "fstpl" };
                writeln!(writer, "    {store} {at}(%esp)")?;
            }
            None => {
                reject_wide(&call.types[index], "call argument")?;
                load_operand(arg, writer, slots, EAX)?;
                writeln!(writer, "    movl {EAX}, {at}(%esp)")?;
            }
        }
    }
    if let Some(buffer) = sret_buffer {
        writeln!(writer, "    leal {buffer}(%ebp), {EAX}")?;
        writeln!(writer, "    movl {EAX}, (%esp)")?;
    }
    // A struct-returning callee pops the hidden pointer itself, so the caller
    // releases everything but that word.
    let callee_pops = if sret_buffer.is_some() { WORD } else { 0 };
    let release = match target {
        Some(target) => {
            writeln!(writer, "    call {target}")?;
            space - callee_pops
        }
        None => {
            writeln!(writer, "    call *{space}(%esp)")?;
            space + INDIRECT_AREA - callee_pops
        }
    };
    if release > 0 {
        writeln!(writer, "    addl ${release}, %esp")?;
    }
    let Some(ret) = ret else {
        return Ok(());
    };
    if wide_result {
        let Register::Virtual(vreg) = ret else {
            return Err(unsupported(
                "i686 call returning a legalized 64-bit integer has no frame slot to write it into",
            ));
        };
        let buffer = slot(slots, vreg)?;
        writeln!(writer, "    movl {EAX}, {buffer}(%ebp)")?;
        writeln!(writer, "    movl {EDX}, {}(%ebp)", buffer + WORD as i32)?;
        return Ok(());
    }
    if call.ret_aggregate.is_some() {
        return Ok(());
    }
    match call.ret_ty.filter(MirType::is_float) {
        Some(ty) => pop_float(ret, writer, slots, is_single(&ty))?,
        None => {
            // A C callee leaves the bits above a narrow return unspecified, so the
            // value is brought back to its canonical form before it is stored.
            if let Some(ty) = call.ret_ty {
                reject_wide(&ty, "call result")?;
                extend_in_place(writer, EAX, &ty, true)?;
            }
            store_register(EAX, ret, writer, slots)?;
        }
    }
    Ok(())
}

/// Lowers the `lamina.*` I/O intrinsics, returning whether `name` was one of them.
///
/// The syscall is made directly through `int $0x80`, where `%ebx` carries the first
/// argument; it is callee-saved, so it is pushed and popped around each one.
fn emit_builtin<W: Write>(
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    writer: &mut W,
    slots: &Slots,
    ctx: &mut EmitCtx<'_>,
) -> Result<bool, LaminaError> {
    const SYS_READ: i32 = 3;
    const SYS_WRITE: i32 = 4;

    match (name, args.len()) {
        ("lamina.writebyte", 1) => {
            load_operand(&args[0], writer, slots, EAX)?;
            writeln!(writer, "    subl $16, %esp")?;
            writeln!(writer, "    movb {}, (%esp)", byte_reg(EAX)?)?;
            writeln!(writer, "    movl %esp, {ECX}")?;
            writeln!(writer, "    pushl %ebx")?;
            writeln!(writer, "    movl ${SYS_WRITE}, {EAX}")?;
            writeln!(writer, "    movl $1, %ebx")?;
            writeln!(writer, "    movl $1, {EDX}")?;
            writeln!(writer, "    int $0x80")?;
            writeln!(writer, "    popl %ebx")?;
            emit_io_result(writer, false)?;
            writeln!(writer, "    addl $16, %esp")?;
        }
        ("lamina.readbyte", 0) => {
            writeln!(writer, "    subl $16, %esp")?;
            writeln!(writer, "    movl %esp, {ECX}")?;
            writeln!(writer, "    pushl %ebx")?;
            writeln!(writer, "    movl ${SYS_READ}, {EAX}")?;
            writeln!(writer, "    xorl %ebx, %ebx")?;
            writeln!(writer, "    movl $1, {EDX}")?;
            writeln!(writer, "    int $0x80")?;
            writeln!(writer, "    popl %ebx")?;
            // The byte, 0..=255, after one read; -1 at end of input and -2 on any
            // failure, keyed on the count so a NUL byte is not read as the end.
            writeln!(writer, "    movzbl (%esp), {ECX}")?;
            writeln!(writer, "    movl $-1, {EDX}")?;
            writeln!(writer, "    testl {EAX}, {EAX}")?;
            writeln!(writer, "    cmovel {EDX}, {ECX}")?;
            writeln!(writer, "    movl $-2, {EDX}")?;
            writeln!(writer, "    cmpl $1, {EAX}")?;
            writeln!(writer, "    cmoval {EDX}, {ECX}")?;
            writeln!(writer, "    movl {ECX}, {EAX}")?;
            writeln!(writer, "    addl $16, %esp")?;
        }
        ("lamina.write", 2) | ("lamina.read", 2) => {
            let writing = name == "lamina.write";
            let done = ctx.fresh_label("io");
            load_operand(&args[0], writer, slots, ECX)?;
            load_operand(&args[1], writer, slots, EDX)?;
            if !writing {
                writeln!(writer, "    movl {EDX}, {EAX}")?;
                writeln!(writer, "    testl {EAX}, {EAX}")?;
                writeln!(writer, "    je {done}")?;
            }
            writeln!(writer, "    pushl %ebx")?;
            writeln!(
                writer,
                "    movl ${}, {EAX}",
                if writing { SYS_WRITE } else { SYS_READ }
            )?;
            if writing {
                writeln!(writer, "    movl $1, %ebx")?;
            } else {
                writeln!(writer, "    xorl %ebx, %ebx")?;
            }
            writeln!(writer, "    int $0x80")?;
            writeln!(writer, "    popl %ebx")?;
            emit_io_result(writer, !writing)?;
            writeln!(writer, "{done}:")?;
        }
        _ => return Ok(false),
    }
    if let Some(ret) = ret {
        store_register(EAX, ret, writer, slots)?;
    }
    Ok(true)
}

/// Turns a raw syscall result in `%eax` into the I/O contract: the count stays, a
/// failure becomes -2, and with `end_of_input` a zero count becomes -1.
fn emit_io_result<W: Write>(writer: &mut W, end_of_input: bool) -> Result<(), LaminaError> {
    writeln!(writer, "    testl {EAX}, {EAX}")?;
    writeln!(writer, "    movl $-2, {ECX}")?;
    writeln!(writer, "    cmovsl {ECX}, {EAX}")?;
    if end_of_input {
        writeln!(writer, "    testl {EAX}, {EAX}")?;
        writeln!(writer, "    movl $-1, {ECX}")?;
        writeln!(writer, "    cmovel {ECX}, {EAX}")?;
    }
    Ok(())
}
