//! MIR to LoongArch64 assembly for Linux under the LP64D calling convention.
//!
//! Every virtual register lives in a frame slot. The emitter moves operands through
//! `$a0`, `$a1`, `$a2`, `$t0` and `$t1` and writes results back, so no register is
//! allocated and no callee-saved register is used. Global allocation is rejected by the
//! dispatcher, and aggregates by the call-parameter validator.

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
    promoted_argument_type,
};
use crate::{
    error::LaminaError,
    mir::{
        FloatBinOp, FloatCmpOp, FloatUnOp, Function, Instruction as MirInst, IntBinOp, IntCmpOp,
        MirType, Module as MirModule, Operand, Register, ScalarType, instruction::AddressMode,
    },
    mir_codegen::{
        CodegenError, MirCodegenSettings, RegallocStrategy, validate_module_call_parameters,
    },
};
use abi::{ARG_FPRS, ARG_GPRS, ArgPlace, WORD};
use lamina_mir::SourceLoc;
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
    params: HashMap<String, Vec<MirType>>,
    variadic: HashMap<String, usize>,
    float_returns: HashMap<String, MirType>,
}

impl Signatures {
    fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| (name.clone(), func.sig.params.iter().map(|p| p.ty).collect()))
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
                    ty.is_float().then(|| (name.clone(), ty))
                })
                .collect(),
        }
    }

    /// Types of a call's arguments as the convention sees them. A matching prototype
    /// gives them; past a variadic callee's fixed parameters, a float travels as an
    /// integer, the way LP64D passes it through the integer bank.
    fn call_types(&self, callee: &str, args: &[Operand]) -> Vec<MirType> {
        let fixed = self.variadic.get(callee).copied();
        let mut types = match self.params.get(callee) {
            Some(declared) if fixed.is_none() && declared.len() == args.len() => declared.clone(),
            declared => {
                let fixed = fixed.unwrap_or(0);
                args.iter()
                    .enumerate()
                    .map(|(index, arg)| match declared {
                        Some(kinds) if index < fixed && index < kinds.len() => kinds[index],
                        _ => promoted_argument_type(arg),
                    })
                    .collect()
            }
        };
        if let Some(fixed) = fixed {
            integer_past_fixed(&mut types, fixed);
        }
        types
    }
}

fn integer_past_fixed(types: &mut [MirType], fixed: usize) {
    for ty in types.iter_mut().skip(fixed) {
        if ty.is_float() {
            *ty = MirType::Scalar(ScalarType::I64);
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
    stack_size: usize,
    signatures: &'a Signatures,
    /// Set when this function returns a float, which leaves in `$fa0`.
    ret_float: Option<MirType>,
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
    CapabilitySet::standard_native().with(CodegenCapability::IndirectCalls)
}

pub fn generate_mir_loongarch64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Linux {
        return Err(unsupported(format!(
            "LoongArch64 code generation supports Linux only, not {target_os}"
        )));
    }
    if settings.regalloc != RegallocStrategy::Incremental {
        return Err(LaminaError::ValidationError(
            "global register allocation (LinearScanGlobal / GraphColorGlobal) is not implemented for LoongArch64".to_string(),
        ));
    }
    validate_module_call_parameters(module, TargetArchitecture::LoongArch64)?;
    let signatures = Arc::new(Signatures::from_module(module));
    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_function(name, func, os, settings_arc.as_ref(), signatures.as_ref())
        })
        .map_err(parallel_codegen_error)?;

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
    let stack_size = slots
        .values()
        .map(|offset| (-offset - 16) as usize)
        .max()
        .unwrap_or(0);
    abi::prologue(&mut output, stack_size).map_err(io)?;
    store_incoming_arguments(&mut output, func, &slots).map_err(lamina_to_codegen)?;

    let mut ctx = EmitCtx {
        func_name,
        target_os,
        settings,
        loc: None,
        label_seq: 0,
        stack_size,
        signatures,
        ret_float: func.sig.ret_ty.filter(MirType::is_float),
    };
    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    b {}", ctx.block_label(&func.entry)).map_err(io)?;
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
) -> Result<(), LaminaError> {
    let types: Vec<MirType> = func.sig.params.iter().map(|p| p.ty).collect();
    let places = abi::classify(&types);
    for (param, place) in func.sig.params.iter().zip(places) {
        let Register::Virtual(vreg) = &param.reg else {
            continue;
        };
        let Some(&offset) = slots.get(vreg) else {
            continue;
        };
        let single = param.ty.size_bytes(PointerWidth::Bits64) == 4;
        match place {
            ArgPlace::Gpr(bank) => {
                abi::fp_store(writer, "st.d", ARG_GPRS[bank], offset, "$t0")?;
            }
            ArgPlace::Fpr(bank) => {
                store_float(ARG_FPRS[bank], vreg, writer, slots, single)?;
            }
            ArgPlace::Stack(index) => {
                abi::fp_load(writer, "ld.d", "$t0", index as i32 * WORD, "$t0")?;
                abi::fp_store(writer, "st.d", "$t0", offset, "$t0")?;
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
                load_operand(lhs, writer, slots, "$a0")?;
                load_operand(rhs, writer, slots, "$a1")?;
                let unsigned = matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                );
                extend_integer(writer, "$a0", ty, !unsigned)?;
                extend_integer(writer, "$a1", ty, !unsigned)?;
                let (invert, taken, other) =
                    plan.branch(func, index, next, true_target, false_target);
                let (mnemonic, a, b) = branch_for(*op, invert);
                writeln!(
                    writer,
                    "    {mnemonic} {a}, {b}, {}",
                    ctx.block_label(taken)
                )?;
                if let Some(other) = other {
                    writeln!(writer, "    b {}", ctx.block_label(other))?;
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

/// The branch taken when the comparison holds (or, when `invert`, when it does not),
/// with its operands in the order the two-register branch compares them.
fn branch_for(op: IntCmpOp, invert: bool) -> (&'static str, &'static str, &'static str) {
    match (op, invert) {
        (IntCmpOp::Eq, false) | (IntCmpOp::Ne, true) => ("beq", "$a0", "$a1"),
        (IntCmpOp::Ne, false) | (IntCmpOp::Eq, true) => ("bne", "$a0", "$a1"),
        (IntCmpOp::SLt, false) | (IntCmpOp::SGe, true) => ("blt", "$a0", "$a1"),
        (IntCmpOp::SLe, false) | (IntCmpOp::SGt, true) => ("bge", "$a1", "$a0"),
        (IntCmpOp::SGt, false) | (IntCmpOp::SLe, true) => ("blt", "$a1", "$a0"),
        (IntCmpOp::SGe, false) | (IntCmpOp::SLt, true) => ("bge", "$a0", "$a1"),
        (IntCmpOp::ULt, false) | (IntCmpOp::UGe, true) => ("bltu", "$a0", "$a1"),
        (IntCmpOp::ULe, false) | (IntCmpOp::UGt, true) => ("bgeu", "$a1", "$a0"),
        (IntCmpOp::UGt, false) | (IntCmpOp::ULe, true) => ("bltu", "$a1", "$a0"),
        (IntCmpOp::UGe, false) | (IntCmpOp::ULt, true) => ("bgeu", "$a0", "$a1"),
    }
}

/// Sign- or zero-extends a narrow integer in `reg` to the 64-bit form values take.
fn extend_integer<W: Write>(
    writer: &mut W,
    reg: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
    ) {
        return Ok(());
    }
    let shift = 64 - ty.size_bytes(PointerWidth::Bits64) * 8;
    writeln!(writer, "    slli.d {reg}, {reg}, {shift}")?;
    let right = if signed { "srai.d" } else { "srli.d" };
    writeln!(writer, "    {right} {reg}, {reg}, {shift}")?;
    Ok(())
}

/// Stores `$a0` (or any integer register `src`) to a MIR destination.
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

/// Reads `ty`-wide memory at `base + offset` into or from `reg`, building the address
/// in `scratch` when the offset does not fit 12 bits. `scratch` must differ from `base`
/// and `reg`.
fn memory_access<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    reg: &str,
    base: &str,
    offset: i64,
    scratch: &str,
) -> Result<(), LaminaError> {
    if abi::fits_si12(offset) {
        writeln!(writer, "    {mnemonic} {reg}, {base}, {offset}")?;
    } else {
        abi::load_imm(writer, scratch, offset)?;
        writeln!(writer, "    add.d {scratch}, {base}, {scratch}")?;
        writeln!(writer, "    {mnemonic} {reg}, {scratch}, 0")?;
    }
    Ok(())
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
            abi::fp_load(writer, "ld.d", dst, offset, dst)?;
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

/// Places the arguments of a call and returns how many bytes of outgoing stack area to
/// release afterwards. The stack words go first, through `$t0`, so the register loads
/// that follow can use their own destinations as scratch.
fn pass_arguments<W: Write>(
    writer: &mut W,
    args: &[Operand],
    types: &[MirType],
    slots: &Slots,
) -> Result<usize, LaminaError> {
    let places = abi::classify(types);
    let space = abi::outgoing_bytes(&places);
    if space > 0 {
        abi::adjust_sp(writer, -(space as i64))?;
    }
    for (arg, place) in args.iter().zip(&places) {
        let ArgPlace::Stack(index) = *place else {
            continue;
        };
        load_operand(arg, writer, slots, "$t0")?;
        abi::sp_address(writer, "$t1", index * WORD as usize)?;
        writeln!(writer, "    st.d $t0, $t1, 0")?;
    }
    for ((arg, ty), place) in args.iter().zip(types).zip(&places) {
        if let ArgPlace::Fpr(bank) = *place {
            load_float_operand(
                arg,
                writer,
                slots,
                ARG_FPRS[bank],
                ty.size_bytes(PointerWidth::Bits64) == 4,
            )?;
        }
    }
    for (arg, place) in args.iter().zip(&places) {
        if let ArgPlace::Gpr(bank) = *place {
            load_operand(arg, writer, slots, ARG_GPRS[bank])?;
        }
    }
    Ok(space)
}

/// Stores a call's result into its destination: `$a0`, or `$fa0` when the callee returns
/// a float.
fn store_call_result<W: Write>(
    writer: &mut W,
    ret: &Register,
    float: Option<MirType>,
    slots: &Slots,
) -> Result<(), LaminaError> {
    match (ret, float) {
        (Register::Virtual(vreg), Some(ty)) => store_float(
            "$fa0",
            vreg,
            writer,
            slots,
            ty.size_bytes(PointerWidth::Bits64) == 4,
        ),
        (dst, _) => store_result(writer, "$a0", dst, slots),
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
        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if ty.is_vector() {
                return Err(unsupported("LoongArch64 vector Copy"));
            }
            match src {
                Operand::Register(Register::Virtual(vreg)) => {
                    let offset = slot(slots, vreg)?;
                    abi::fp_load(writer, copy_load(ty), "$a0", offset, "$a0")?;
                }
                other => load_operand(other, writer, slots, "$a0")?,
            }
            extend_integer(writer, "$a0", ty, true)?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::SymbolAddress { dst, symbol } => {
            writeln!(writer, "    pcalau12i $a0, %pc_hi20({symbol})")?;
            writeln!(writer, "    addi.d $a0, $a0, %pc_lo12({symbol})")?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::StackAddr { dst, slot: stack } => {
            let offset = slot(slots, stack)?;
            abi::fp_address(writer, "$a0", offset)?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::Lea { dst, base, offset } => {
            let Register::Virtual(base_vreg) = base else {
                return Err(unsupported(format!(
                    "LoongArch64 lea base {base:?} is not a virtual register"
                )));
            };
            let base_slot = slot(slots, base_vreg)?;
            abi::fp_load(writer, "ld.d", "$a0", base_slot, "$a0")?;
            if abi::fits_si12(i64::from(*offset)) {
                if *offset != 0 {
                    writeln!(writer, "    addi.d $a0, $a0, {offset}")?;
                }
            } else {
                abi::load_imm(writer, "$t0", i64::from(*offset))?;
                writeln!(writer, "    add.d $a0, $a0, $t0")?;
            }
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            load_operand(lhs, writer, slots, "$a0")?;
            load_operand(rhs, writer, slots, "$a1")?;
            if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                extend_integer(writer, "$a0", ty, false)?;
                extend_integer(writer, "$a1", ty, false)?;
            }
            if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                extend_integer(writer, "$a0", ty, true)?;
                if !matches!(op, IntBinOp::AShr) {
                    extend_integer(writer, "$a1", ty, true)?;
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
                writeln!(writer, "    bnez $a1, {ok}\n    break 0\n{ok}:")?;
            }
            if overflow {
                let ok = ctx.fresh_label("nooverflow");
                let minimum = ty.signed_integer(
                    1i64 << (ty.size_bytes(PointerWidth::Bits64) * 8 - 1),
                    PointerWidth::Bits64,
                );
                writeln!(writer, "    addi.d $t0, $zero, -1")?;
                writeln!(writer, "    bne $a1, $t0, {ok}")?;
                abi::load_imm(writer, "$t0", minimum)?;
                writeln!(writer, "    bne $a0, $t0, {ok}\n    break 0\n{ok}:")?;
            }
            if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr)
                && ty.size_bytes(PointerWidth::Bits64) < 8
            {
                let mask = ty.size_bytes(PointerWidth::Bits64) * 8 - 1;
                writeln!(writer, "    andi $a1, $a1, {mask}")?;
            }
            let mnemonic = match op {
                IntBinOp::Add => "add.d",
                IntBinOp::Sub => "sub.d",
                IntBinOp::Mul => "mul.d",
                IntBinOp::SDiv => "div.d",
                IntBinOp::UDiv => "div.du",
                IntBinOp::SRem => "mod.d",
                IntBinOp::URem => "mod.du",
                IntBinOp::And => "and",
                IntBinOp::Or => "or",
                IntBinOp::Xor => "xor",
                IntBinOp::Shl => "sll.d",
                IntBinOp::AShr => "sra.d",
                IntBinOp::LShr => "srl.d",
            };
            writeln!(writer, "    {mnemonic} $a0, $a0, $a1")?;
            extend_integer(writer, "$a0", ty, true)?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            load_operand(lhs, writer, slots, "$a0")?;
            load_operand(rhs, writer, slots, "$a1")?;
            let unsigned = matches!(
                op,
                IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
            );
            extend_integer(writer, "$a0", ty, !unsigned)?;
            extend_integer(writer, "$a1", ty, !unsigned)?;
            emit_int_cmp(op, writer)?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            load_operand(false_val, writer, slots, "$a1")?;
            load_operand(true_val, writer, slots, "$a0")?;
            load_operand(&Operand::Register(cond.clone()), writer, slots, "$t0")?;
            let done = ctx.fresh_label("select");
            writeln!(writer, "    bnez $t0, {done}")?;
            writeln!(writer, "    move $a0, $a1")?;
            writeln!(writer, "{done}:")?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            let AddressMode::BaseOffset { base, offset, .. } = addr else {
                return Err(unsupported(
                    "LoongArch64 load supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("ld.b", false),
                MirType::Scalar(ScalarType::I16) => ("ld.h", false),
                MirType::Scalar(ScalarType::I32) => ("ld.w", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("ld.d", false)
                }
                MirType::Scalar(ScalarType::F32) => ("fld.s", true),
                MirType::Scalar(ScalarType::F64) => ("fld.d", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "LoongArch64 load of vector type {ty} is not implemented"
                    )));
                }
            };
            load_base(writer, base, slots, "$t0")?;
            let reg = if float { "$fa0" } else { "$a0" };
            memory_access(writer, mnemonic, reg, "$t0", i64::from(*offset), "$t1")?;
            match (dst, float) {
                (Register::Virtual(vreg), true) => store_float(
                    "$fa0",
                    vreg,
                    writer,
                    slots,
                    matches!(ty, MirType::Scalar(ScalarType::F32)),
                )?,
                (dst, _) => store_result(writer, "$a0", dst, slots)?,
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
                    "LoongArch64 store supports only base+offset addressing; BaseIndexScale is not implemented",
                ));
            };
            let (mnemonic, float) = match ty {
                MirType::Scalar(ScalarType::I1 | ScalarType::I8) => ("st.b", false),
                MirType::Scalar(ScalarType::I16) => ("st.h", false),
                MirType::Scalar(ScalarType::I32) => ("st.w", false),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    ("st.d", false)
                }
                MirType::Scalar(ScalarType::F32) => ("fst.s", true),
                MirType::Scalar(ScalarType::F64) => ("fst.d", true),
                MirType::Vector(_) => {
                    return Err(unsupported(format!(
                        "LoongArch64 store of vector type {ty} is not implemented"
                    )));
                }
            };
            let reg = if float { "$fa0" } else { "$a0" };
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
            load_base(writer, base, slots, "$t0")?;
            memory_access(writer, mnemonic, reg, "$t0", i64::from(*offset), "$t1")?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match ctx.ret_float {
                    Some(ty) => load_float_operand(
                        val,
                        writer,
                        slots,
                        "$fa0",
                        ty.size_bytes(PointerWidth::Bits64) == 4,
                    )?,
                    None => load_operand(val, writer, slots, "$a0")?,
                }
            }
            abi::epilogue(writer, ctx.stack_size, None)?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    b {}", ctx.block_label(target))?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            load_operand(&Operand::Register(cond.clone()), writer, slots, "$t0")?;
            writeln!(writer, "    bnez $t0, {}", ctx.block_label(true_target))?;
            writeln!(writer, "    b {}", ctx.block_label(false_target))?;
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            load_operand(&Operand::Register(value.clone()), writer, slots, "$a0")?;
            for (case_val, case_label) in cases {
                abi::load_imm(writer, "$t0", *case_val)?;
                writeln!(writer, "    beq $a0, $t0, {}", ctx.block_label(case_label))?;
            }
            writeln!(writer, "    b {}", ctx.block_label(default))?;
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
            let mut types = arg_types.clone();
            if let Some(fixed) = variadic_fixed {
                integer_past_fixed(&mut types, *fixed);
            }
            let space = pass_arguments(writer, args, &types, slots)?;
            load_operand(target, writer, slots, "$t1")?;
            writeln!(writer, "    jirl $ra, $t1, 0")?;
            if space > 0 {
                abi::adjust_sp(writer, space as i64)?;
            }
            if let Some(ret) = ret {
                store_call_result(writer, ret, ret_ty.filter(MirType::is_float), slots)?;
            }
        }
        MirInst::TailCall { name, args } => {
            let types = ctx.signatures.call_types(name, args);
            let places = abi::classify(&types);
            let own_stack = ctx
                .signatures
                .params
                .get(ctx.func_name)
                .map(|kinds| abi::outgoing_bytes(&abi::classify(kinds)))
                .unwrap_or(0);
            if abi::outgoing_bytes(&places) > own_stack {
                return Err(unsupported(format!(
                    "LoongArch64 tail call to @{name} passes more stack arguments than this function received"
                )));
            }
            for (arg, place) in args.iter().zip(&places) {
                if let ArgPlace::Stack(index) = *place {
                    load_operand(arg, writer, slots, "$t0")?;
                    abi::fp_store(writer, "st.d", "$t0", index as i32 * WORD, "$t0")?;
                }
            }
            for ((arg, ty), place) in args.iter().zip(&types).zip(&places) {
                if let ArgPlace::Fpr(bank) = *place {
                    load_float_operand(
                        arg,
                        writer,
                        slots,
                        ARG_FPRS[bank],
                        ty.size_bytes(PointerWidth::Bits64) == 4,
                    )?;
                }
            }
            for (arg, place) in args.iter().zip(&places) {
                if let ArgPlace::Gpr(bank) = *place {
                    load_operand(arg, writer, slots, ARG_GPRS[bank])?;
                }
            }
            let target = call_target(name, ctx.target_os);
            abi::epilogue(writer, ctx.stack_size, Some(&target))?;
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
            load_float_operand(lhs, writer, slots, "$fa0", single)?;
            load_float_operand(rhs, writer, slots, "$fa1", single)?;
            let mnemonic = match op {
                FloatBinOp::FAdd => "fadd",
                FloatBinOp::FSub => "fsub",
                FloatBinOp::FMul => "fmul",
                FloatBinOp::FDiv => "fdiv",
            };
            writeln!(writer, "    {mnemonic}.{suffix} $fa0, $fa0, $fa1")?;
            if let Register::Virtual(vreg) = dst {
                store_float("$fa0", vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let single = ty.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            load_float_operand(src, writer, slots, "$fa0", single)?;
            let mnemonic = match op {
                FloatUnOp::FNeg => "fneg",
                FloatUnOp::FSqrt => "fsqrt",
            };
            writeln!(writer, "    {mnemonic}.{suffix} $fa0, $fa0")?;
            if let Register::Virtual(vreg) = dst {
                store_float("$fa0", vreg, writer, slots, single)?;
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
            load_float_operand(lhs, writer, slots, "$fa0", single)?;
            load_float_operand(rhs, writer, slots, "$fa1", single)?;
            let (compare, a, b) = match op {
                FloatCmpOp::Eq => ("ceq", "$fa0", "$fa1"),
                FloatCmpOp::Ne => ("cune", "$fa0", "$fa1"),
                FloatCmpOp::Lt => ("clt", "$fa0", "$fa1"),
                FloatCmpOp::Le => ("cle", "$fa0", "$fa1"),
                FloatCmpOp::Gt => ("clt", "$fa1", "$fa0"),
                FloatCmpOp::Ge => ("cle", "$fa1", "$fa0"),
            };
            writeln!(writer, "    fcmp.{compare}.{suffix} $fcc0, {a}, {b}")?;
            writeln!(writer, "    movcf2gr $a0, $fcc0")?;
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            if !signed {
                return Err(unsupported(
                    "LoongArch64 unsigned float to integer conversion is not implemented",
                ));
            }
            let single = from.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            load_float_operand(src, writer, slots, "$fa0", single)?;
            writeln!(writer, "    fcmp.ceq.{suffix} $fcc0, $fa0, $fa0")?;
            writeln!(writer, "    movcf2gr $t0, $fcc0")?;
            writeln!(writer, "    sub.d $t0, $zero, $t0")?;
            writeln!(writer, "    ftintrz.l.{suffix} $fa0, $fa0")?;
            writeln!(writer, "    movfr2gr.d $a0, $fa0")?;
            writeln!(writer, "    and $a0, $a0, $t0")?;
            let bits = to.size_bytes(PointerWidth::Bits64) * 8;
            if bits < 64 {
                let max = (1i64 << (bits - 1)) - 1;
                clamp_above(writer, max, "slt")?;
                clamp_below(writer, -(1i64 << (bits - 1)))?;
                extend_integer(writer, "$a0", to, true)?;
            }
            store_result(writer, "$a0", dst, slots)?;
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let from_bits = from.size_bytes(PointerWidth::Bits64) * 8;
            if !signed && from_bits == 64 {
                return Err(unsupported(
                    "LoongArch64 unsigned 64-bit integer to float conversion is not implemented",
                ));
            }
            load_operand(src, writer, slots, "$a0")?;
            if from_bits < 64 {
                if *signed {
                    extend_integer(writer, "$a0", from, true)?;
                } else {
                    writeln!(writer, "    slli.d $a0, $a0, {}", 64 - from_bits)?;
                    writeln!(writer, "    srli.d $a0, $a0, {}", 64 - from_bits)?;
                }
            }
            let single = to.size_bytes(PointerWidth::Bits64) == 4;
            let suffix = if single { "s" } else { "d" };
            writeln!(writer, "    movgr2fr.d $fa0, $a0")?;
            writeln!(writer, "    ffint.{suffix}.l $fa0, $fa0")?;
            if let Register::Virtual(vreg) = dst {
                store_float("$fa0", vreg, writer, slots, single)?;
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let from_single = from.size_bytes(PointerWidth::Bits64) == 4;
            let to_single = to.size_bytes(PointerWidth::Bits64) == 4;
            load_float_operand(src, writer, slots, "$fa0", from_single)?;
            match (from_single, to_single) {
                (true, false) => writeln!(writer, "    fcvt.d.s $fa0, $fa0")?,
                (false, true) => writeln!(writer, "    fcvt.s.d $fa0, $fa0")?,
                (true, true) => writeln!(writer, "    fmov.s $fa0, $fa0")?,
                (false, false) => writeln!(writer, "    fmov.d $fa0, $fa0")?,
            }
            if let Register::Virtual(vreg) = dst {
                store_float("$fa0", vreg, writer, slots, to_single)?;
            }
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Unreachable => {
            writeln!(writer, "    break 0")?;
        }
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {}
        MirInst::IntConvert { .. } => {
            return Err(unsupported(
                "LoongArch64 received an integer conversion that was not lowered",
            ));
        }
        MirInst::IntUnary { .. } => {
            return Err(unsupported(
                "LoongArch64 integer bit intrinsics (popcount, ctlz, cttz, bswap)",
            ));
        }
        MirInst::IntRotate { .. } => {
            return Err(unsupported("LoongArch64 integer rotates"));
        }
        MirInst::VectorOp { .. } => {
            return Err(unsupported("LoongArch64 vector operations"));
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            return Err(unsupported("LoongArch64 atomic operations and fences"));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported(
                "inline assembly is not supported by the LoongArch64 backend",
            ));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(unsupported("LoongArch64 variadic argument access"));
        }
        #[cfg(feature = "nightly")]
        MirInst::SimdBinary { .. }
        | MirInst::SimdUnary { .. }
        | MirInst::SimdTernary { .. }
        | MirInst::SimdShuffle { .. }
        | MirInst::SimdExtract { .. }
        | MirInst::SimdInsert { .. }
        | MirInst::SimdLoad { .. }
        | MirInst::SimdStore { .. } => {
            return Err(unsupported("LoongArch64 SIMD operations"));
        }
    }
    Ok(())
}

fn copy_load(ty: &MirType) -> &'static str {
    match ty.size_bytes(PointerWidth::Bits64) {
        1 => "ld.b",
        2 => "ld.h",
        4 => "ld.w",
        _ => "ld.d",
    }
}

fn emit_int_cmp<W: Write>(op: &IntCmpOp, writer: &mut W) -> Result<(), LaminaError> {
    match op {
        IntCmpOp::Eq => {
            writeln!(writer, "    xor $a0, $a0, $a1")?;
            writeln!(writer, "    sltui $a0, $a0, 1")?;
        }
        IntCmpOp::Ne => {
            writeln!(writer, "    xor $a0, $a0, $a1")?;
            writeln!(writer, "    sltu $a0, $zero, $a0")?;
        }
        IntCmpOp::SLt => writeln!(writer, "    slt $a0, $a0, $a1")?,
        IntCmpOp::SGt => writeln!(writer, "    slt $a0, $a1, $a0")?,
        IntCmpOp::SLe => {
            writeln!(writer, "    slt $a0, $a1, $a0")?;
            writeln!(writer, "    xori $a0, $a0, 1")?;
        }
        IntCmpOp::SGe => {
            writeln!(writer, "    slt $a0, $a0, $a1")?;
            writeln!(writer, "    xori $a0, $a0, 1")?;
        }
        IntCmpOp::ULt => writeln!(writer, "    sltu $a0, $a0, $a1")?,
        IntCmpOp::UGt => writeln!(writer, "    sltu $a0, $a1, $a0")?,
        IntCmpOp::ULe => {
            writeln!(writer, "    sltu $a0, $a1, $a0")?;
            writeln!(writer, "    xori $a0, $a0, 1")?;
        }
        IntCmpOp::UGe => {
            writeln!(writer, "    sltu $a0, $a0, $a1")?;
            writeln!(writer, "    xori $a0, $a0, 1")?;
        }
    }
    Ok(())
}

/// Replaces `$a0` with `limit` when it is above `limit`, without branching. `compare` is
/// `slt` for a signed limit and `sltu` for an unsigned one.
fn clamp_above<W: Write>(writer: &mut W, limit: i64, compare: &str) -> Result<(), LaminaError> {
    abi::load_imm(writer, "$t1", limit)?;
    writeln!(writer, "    {compare} $t2, $t1, $a0")?;
    writeln!(writer, "    sub.d $t2, $zero, $t2")?;
    writeln!(writer, "    xor $t3, $a0, $t1")?;
    writeln!(writer, "    and $t3, $t3, $t2")?;
    writeln!(writer, "    xor $a0, $a0, $t3")?;
    Ok(())
}

/// Replaces `$a0` with `limit` when it is below `limit`, signed.
fn clamp_below<W: Write>(writer: &mut W, limit: i64) -> Result<(), LaminaError> {
    abi::load_imm(writer, "$t1", limit)?;
    writeln!(writer, "    slt $t2, $a0, $t1")?;
    writeln!(writer, "    sub.d $t2, $zero, $t2")?;
    writeln!(writer, "    xor $t3, $a0, $t1")?;
    writeln!(writer, "    and $t3, $t3, $t2")?;
    writeln!(writer, "    xor $a0, $a0, $t3")?;
    Ok(())
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
        load_operand(&args[0], writer, slots, "$a1")?;
        load_operand(&args[1], writer, slots, "$a2")?;
        let writing = name == "lamina.write";
        writeln!(writer, "    addi.d $a0, $zero, {}", i32::from(writing))?;
        let done = ctx.fresh_label("io");
        if !writing {
            writeln!(writer, "    beqz $a2, {done}")?;
        }
        let libc = if writing { "write" } else { "read" };
        writeln!(writer, "    bl {}", call_target(libc, ctx.target_os))?;
        if writing {
            writeln!(writer, "    bge $a0, $zero, {done}")?;
            writeln!(writer, "    addi.d $a0, $zero, -2")?;
            writeln!(writer, "{done}:")?;
        } else {
            let positive = ctx.fresh_label("io_pos");
            writeln!(writer, "    bge $a0, $zero, {positive}")?;
            writeln!(writer, "    addi.d $a0, $zero, -2")?;
            writeln!(writer, "    b {done}")?;
            writeln!(writer, "{positive}:")?;
            writeln!(writer, "    bnez $a0, {done}")?;
            writeln!(writer, "    addi.d $a0, $zero, -1")?;
            writeln!(writer, "{done}:")?;
        }
        if let Some(ret) = ret {
            store_call_result(writer, ret, None, slots)?;
        }
        return Ok(());
    }
    if name == "lamina.writebyte" && args.len() == 1 {
        load_operand(&args[0], writer, slots, "$a0")?;
        writeln!(writer, "    addi.d $sp, $sp, -16")?;
        writeln!(writer, "    st.b $a0, $sp, 0")?;
        writeln!(writer, "    addi.d $a0, $zero, 1")?;
        writeln!(writer, "    move $a1, $sp")?;
        writeln!(writer, "    addi.d $a2, $zero, 1")?;
        writeln!(writer, "    bl {}", call_target("write", ctx.target_os))?;
        let done = ctx.fresh_label("writebyte");
        writeln!(writer, "    bge $a0, $zero, {done}")?;
        writeln!(writer, "    addi.d $a0, $zero, -2")?;
        writeln!(writer, "{done}:")?;
        writeln!(writer, "    addi.d $sp, $sp, 16")?;
        if let Some(ret) = ret {
            store_call_result(writer, ret, None, slots)?;
        }
        return Ok(());
    }
    if name == "lamina.readbyte" && args.is_empty() {
        writeln!(writer, "    addi.d $sp, $sp, -16")?;
        writeln!(writer, "    addi.d $a0, $zero, 0")?;
        writeln!(writer, "    move $a1, $sp")?;
        writeln!(writer, "    addi.d $a2, $zero, 1")?;
        writeln!(writer, "    bl {}", call_target("read", ctx.target_os))?;
        let one = ctx.fresh_label("readbyte_one");
        let other = ctx.fresh_label("readbyte_other");
        let done = ctx.fresh_label("readbyte_done");
        writeln!(writer, "    addi.d $t1, $zero, 1")?;
        writeln!(writer, "    beq $a0, $t1, {one}")?;
        writeln!(writer, "    bnez $a0, {other}")?;
        writeln!(writer, "    addi.d $a0, $zero, -1")?;
        writeln!(writer, "    b {done}")?;
        writeln!(writer, "{other}:")?;
        writeln!(writer, "    addi.d $a0, $zero, -2")?;
        writeln!(writer, "    b {done}")?;
        writeln!(writer, "{one}:")?;
        writeln!(writer, "    ld.bu $a0, $sp, 0")?;
        writeln!(writer, "{done}:")?;
        writeln!(writer, "    addi.d $sp, $sp, 16")?;
        if let Some(ret) = ret {
            store_call_result(writer, ret, None, slots)?;
        }
        return Ok(());
    }

    let types = ctx.signatures.call_types(name, args);
    let space = pass_arguments(writer, args, &types, slots)?;
    writeln!(writer, "    bl {}", call_target(name, ctx.target_os))?;
    if space > 0 {
        abi::adjust_sp(writer, space as i64)?;
    }
    if let Some(ret) = ret {
        let float = ctx.signatures.float_returns.get(name).copied();
        store_call_result(writer, ret, float, slots)?;
    }
    Ok(())
}
