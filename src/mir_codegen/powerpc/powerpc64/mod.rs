//! PowerPC64 ELFv2 code generator for MIR.
//!
//! Targets little-endian PowerPC64 (ppc64le) running Linux, using the
//! ELFv2 ABI.  Assembly output is suitable for GNU `as` or LLVM `as`.
//!
//! ## Register conventions used internally
//!
//! - `r3`   — primary integer scratch / return value
//! - `r4`   — secondary integer scratch
//! - `r3–r10` — integer arguments (in order)
//! - `f1`   — primary float scratch / return value
//! - `f2`   — secondary float scratch
//! - Stack pointer: `r1`
//! - TOC: `r2` (never clobbered)
//! - Link register saved/restored in prologue/epilogue

mod util;

use crate::mir::register::PhysicalReg;
use crate::mir_codegen::common::integer_conversion;
use crate::mir_codegen::common::{
    CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, self_tail_arguments, self_tail_schedule,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note};

use std::{
    collections::{BTreeSet, HashMap},
    io::Write,
    sync::Arc,
};
use util::{
    load_operand_to_register, load_register_to_r3, load_register_to_register, store_r3_to_register,
};

use crate::{
    error::LaminaError,
    mir::{
        FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Instruction as MirInst, IntBinOp,
        IntCmpOp, MirType, Module as MirModule, Operand, Register, ScalarType, Signature,
        VirtualReg,
        instruction::{AddressMode, Immediate},
        register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::CapabilitySet,
        common::{CodegenBase, compile_functions_parallel, parallel_codegen_error},
        validate_module_call_parameters,
    },
};

use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator,
    powerpc::{
        Ppc64Abi, Ppc64Frame, Ppc64RegAlloc,
        frame::{LINKAGE_SIZE, load_imm, r1_address, r1_mem},
    },
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

pub struct Ppc64Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> Ppc64Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
        }
    }

    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), LaminaError> {
        generate_mir_ppc64_with_units_and_settings(
            module,
            writer,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for Ppc64Codegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn capabilities() -> CapabilitySet {
        CapabilitySet::standard_native()
    }

    fn prepare(
        &mut self,
        types: &HashMap<String, MirType>,
        globals: &HashMap<String, Global>,
        funcs: &HashMap<String, Signature>,
        codegen_units: usize,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError> {
        self.base.prepare_base(
            types,
            globals,
            funcs,
            codegen_units,
            verbose,
            options,
            input_name,
        )
    }

    fn compile(&mut self) -> Result<(), CodegenError> {
        self.base.compile_base()
    }

    fn finalize(&mut self) -> Result<(), CodegenError> {
        self.base.finalize_base()
    }

    fn emit_asm(&mut self) -> Result<(), CodegenError> {
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_ppc64_with_units_and_settings(
                    module,
                    writer,
                    target_os,
                    codegen_units,
                    &MirCodegenSettings::default(),
                )
            },
            "PowerPC64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not yet supported for PowerPC64".to_string(),
        ))
    }
}

/// Spill slots from the linear allocator are numbered from zero, so they start after the
/// slots that `assign_ppc64_slots` handed out and cannot share one with a reservation.
fn ppc64_stack_offset_for_linear_spill(off: i32, first_free_slot: usize) -> i32 {
    let k = ((-off) / 8) as usize;
    let slot_ix = first_free_slot + k.saturating_sub(1);
    Ppc64Frame::calculate_stack_offset(slot_ix)
}

/// Gives `vreg` the slot at the bottom of its block. Slots grow upward from `r1`, so the
/// extra slots of a reservation sit above the address the register holds.
fn claim_ppc64_slot(
    func: &Function,
    vreg: &VirtualReg,
    slots: &mut HashMap<VirtualReg, i32>,
    next: &mut usize,
) {
    let extra = func.stack_reservations.get(vreg).copied().unwrap_or(0);
    slots.insert(*vreg, Ppc64Frame::calculate_stack_offset(*next));
    *next += extra + 1;
}

/// Gives every register that needs a slot one, defined registers first, and returns the
/// number of slots used.
fn assign_ppc64_slots(func: &Function) -> (HashMap<VirtualReg, i32>, usize) {
    let mut defined = BTreeSet::new();
    let mut used = BTreeSet::new();
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        if let Some(Register::Virtual(vreg)) = inst.def_reg() {
            defined.insert(*vreg);
        }
        if let MirInst::StackAddr { slot, .. } = inst {
            used.insert(*slot);
        }
        for reg in inst.use_regs() {
            if let Register::Virtual(vreg) = reg {
                used.insert(*vreg);
            }
        }
    }
    let mut slots = HashMap::new();
    let mut next = 0usize;
    for vreg in &defined {
        claim_ppc64_slot(func, vreg, &mut slots, &mut next);
    }
    for vreg in &used {
        if !defined.contains(vreg) {
            claim_ppc64_slot(func, vreg, &mut slots, &mut next);
        }
    }
    (slots, next)
}

/// Bytes of frame locals, measured from the end of the linkage area to the highest byte
/// that any slot or reservation reaches.
fn ppc64_local_bytes(func: &Function, slots: &HashMap<VirtualReg, i32>) -> usize {
    let base = i64::from(Ppc64Frame::calculate_stack_offset(0));
    slots
        .iter()
        .map(|(vreg, off)| {
            let extra = func.stack_reservations.get(vreg).copied().unwrap_or(0) as i64;
            i64::from(*off) + 8 * (extra + 1)
        })
        .max()
        .map_or(0, |top| (top - base).max(0) as usize)
}

fn compile_single_function_ppc64(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &Ppc64Signatures,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let abi = Ppc64Abi::new(target_os);

    let label = abi.mangle_function_name(func_name);
    let block_prefix = format!(".L_{}.", label.trim_start_matches('.'));
    let entry = format!(".L_gep_{}", label.trim_start_matches('.'));
    let local = format!(".L_lep_{}", label.trim_start_matches('.'));
    writeln!(output, "{label}:")
        .and_then(|()| writeln!(output, "{entry}:"))
        .and_then(|()| writeln!(output, "    addis 2, 12, .TOC.-{entry}@ha"))
        .and_then(|()| writeln!(output, "    addi 2, 2, .TOC.-{entry}@l"))
        .and_then(|()| writeln!(output, "{local}:"))
        .and_then(|()| writeln!(output, "    .localentry {label}, {local}-{entry}"))
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("I/O error: {e}")))?;

    let mut reg_alloc = Ppc64RegAlloc::new(target_os);
    let mut stack_slots: HashMap<VirtualReg, i32> = HashMap::new();

    if settings.regalloc != RegallocStrategy::Incremental {
        let (assigned, first_free_slot) = assign_ppc64_slots(func);
        stack_slots = assigned;
        let pool = Ppc64RegAlloc::gpr_pool_for_global_allocation();
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| i.vreg.class == RegisterClass::Gpr)
            .collect();
        let mut plan = match settings.regalloc {
            RegallocStrategy::LinearScanGlobal => {
                LinearScanAllocator::allocate(&intervals, pool.as_slice())
            }
            RegallocStrategy::GraphColorGlobal => {
                GraphColorAllocator::allocate(&intervals, pool.as_slice())
            }
            RegallocStrategy::Incremental => {
                return Err(CodegenError::InvalidCodegenOptions(
                    "internal: incremental in global branch".to_string(),
                ));
            }
        };
        plan = coalesce_dying_inputs(func, &intervals, &plan, |_, _| true);
        reg_alloc = Ppc64RegAlloc::from_global_plan(target_os, &plan);
        for (v, a) in &plan {
            if let MirAllocation::Spill(off) = a {
                stack_slots.insert(
                    *v,
                    ppc64_stack_offset_for_linear_spill(*off, first_free_slot),
                );
            }
        }
    } else {
        let mut next_slot = 0usize;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                    && !stack_slots.contains_key(vreg)
                {
                    claim_ppc64_slot(func, vreg, &mut stack_slots, &mut next_slot);
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg)
                    {
                        claim_ppc64_slot(func, vreg, &mut stack_slots, &mut next_slot);
                    }
                }
                if let MirInst::StackAddr { slot, .. } = inst
                    && !stack_slots.contains_key(slot)
                {
                    claim_ppc64_slot(func, slot, &mut stack_slots, &mut next_slot);
                }
            }
        }
    }

    let local_bytes = ppc64_local_bytes(func, &stack_slots);
    Ppc64Frame::generate_prologue(&mut output, local_bytes)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    let param_types: Vec<MirType> = func.sig.params.iter().map(|p| p.ty).collect();
    let param_places = classify_ppc64_arguments(&param_types);
    for (index, param) in func.sig.params.iter().enumerate() {
        if let Register::Virtual(vreg) = &param.reg
            && let Some(offset) = stack_slots.get(vreg)
        {
            if let PpcArgPlace::Fpr(bank) = param_places[index] {
                emit_store_fp_result(
                    &bank.to_string(),
                    vreg,
                    &mut output,
                    &reg_alloc,
                    &stack_slots,
                    param.ty.size_bytes() == 4,
                )
                .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
            } else if let Some(src) = Ppc64Abi::ARG_REGISTERS.get(index) {
                extend_param(&mut output, src, &param.ty)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                r1_mem(&mut output, "std", src, *offset)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
            } else {
                let disp = Ppc64Frame::aligned_frame_size(local_bytes) as i32
                    + parameter_save_offset(index);
                r1_mem(&mut output, "ld", "0", disp)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                extend_param(&mut output, "0", &param.ty)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                r1_mem(&mut output, "std", "0", *offset)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
            }
        }
    }
    for param in &func.sig.params {
        if let Register::Virtual(vreg) = &param.reg
            && let Some(reg) = reg_alloc.get_mapping_for(vreg)
            && let Some(offset) = stack_slots.get(vreg)
        {
            r1_mem(&mut output, "ld", reg, *offset)
                .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
    }
    let plan = FunctionEmissionPlan::new(func);
    let emit = Ppc64Emit {
        target_os,
        block_prefix: &block_prefix,
        settings,
        local_bytes,
        signatures,
        ret_float: func.sig.ret_ty.filter(MirType::is_float),
    };
    let mut debug_line: u32 = 0;
    let mut cmp_seq: u32 = 0;
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    b {block_prefix}{}", func.entry)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    for (slot, &index) in plan.order.iter().enumerate() {
        let next = plan
            .order
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        emit_planned_block_ppc64(
            func,
            index,
            next,
            &plan,
            &mut output,
            &mut reg_alloc,
            &stack_slots,
            &emit,
            &mut debug_line,
            &mut cmp_seq,
        )
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    Ok(output)
}

pub fn generate_mir_ppc64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_ppc64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_ppc64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_ppc64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_ppc64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::PowerPC64)?;
    let abi = Ppc64Abi::new(target_os);

    writeln!(writer, ".abiversion 2")?;
    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_print_format())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    emit_debug_file(writer, settings)?;
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;

    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(Ppc64Signatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_ppc64(
                name,
                func,
                os,
                settings_arc.as_ref(),
                signatures.as_ref(),
            )
        })
        .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }
    emit_global_data(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;
    emit_gnu_stack_note(writer, target_os)?;

    Ok(())
}

#[allow(clippy::too_many_lines)]
/// Where an argument sits under elfv2: a general register, a float register, or the
/// parameter save area. A float takes the next f register while still consuming its
/// parameter slot, which leaves the matching general register unused.
#[derive(Clone, Copy)]
enum PpcArgPlace {
    Gpr(usize),
    Fpr(usize),
    Stack,
}

/// f1 to f13 carry floats.
const FLOAT_ARGUMENT_REGISTERS: usize = 13;

fn classify_ppc64_arguments(types: &[MirType]) -> Vec<PpcArgPlace> {
    let mut floats = 0;
    types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            if ty.is_float() && floats < FLOAT_ARGUMENT_REGISTERS {
                floats += 1;
                PpcArgPlace::Fpr(floats)
            } else if index < Ppc64Abi::ARG_REGISTERS.len() {
                PpcArgPlace::Gpr(index)
            } else {
                PpcArgPlace::Stack
            }
        })
        .collect()
}

/// Parameter and result types of each function, so a call site knows which bank an
/// argument belongs in.
#[derive(Default)]
struct Ppc64Signatures {
    params: HashMap<String, Vec<MirType>>,
    returns: HashMap<String, MirType>,
}

impl Ppc64Signatures {
    fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| (name.clone(), func.sig.params.iter().map(|p| p.ty).collect()))
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_ty?)))
                .collect(),
        }
    }

    /// An undeclared callee has no signature, so its arguments pass as integers.
    fn arguments(&self, callee: &str, count: usize) -> Vec<MirType> {
        match self.params.get(callee) {
            Some(types) if types.len() == count => types.clone(),
            _ => vec![MirType::i64(); count],
        }
    }

    fn float_result(&self, callee: &str) -> Option<MirType> {
        self.returns.get(callee).copied().filter(MirType::is_float)
    }
}

struct Ppc64Emit<'a> {
    target_os: TargetOperatingSystem,
    block_prefix: &'a str,
    settings: &'a MirCodegenSettings,
    local_bytes: usize,
    signatures: &'a Ppc64Signatures,
    /// Set when this function returns a float, which leaves in f1.
    ret_float: Option<MirType>,
}

#[allow(clippy::too_many_arguments)]
fn emit_planned_block_ppc64<W: Write>(
    func: &Function,
    index: usize,
    next: &str,
    plan: &FunctionEmissionPlan,
    writer: &mut W,
    reg_alloc: &mut Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    emit: &Ppc64Emit<'_>,
    debug_line: &mut u32,
    cmp_seq: &mut u32,
) -> Result<(), LaminaError> {
    let block = &func.blocks[index];
    let prefix = emit.block_prefix;
    if plan.aligned_headers.contains(&index) {
        writeln!(writer, "    .p2align 4")?;
    }
    writeln!(writer, "{prefix}{}:", block.label)?;
    let mut position = 0;
    while position < block.instructions.len() {
        let inst = &block.instructions[position];
        if let Some(pairs) = self_tail_arguments(func, inst) {
            let schedule = self_tail_schedule(&pairs, |reg| match reg {
                Register::Virtual(v) => reg_alloc
                    .get_mapping_for(v)
                    .map(|name| Register::Physical(PhysicalReg::new(name, RegisterClass::Gpr)))
                    .unwrap_or_else(|| reg.clone()),
                _ => reg.clone(),
            });
            for (src, dst) in schedule {
                let src = match src {
                    CopyLocation::Value(src) => src,
                    CopyLocation::Scratch => Operand::Register(Register::Physical(
                        PhysicalReg::new("0", RegisterClass::Gpr),
                    )),
                };
                load_operand_to_register(&src, writer, reg_alloc, stack_slots, "3")?;
                match dst {
                    CopyLocation::Value(Operand::Register(Register::Virtual(dst))) => {
                        store_r3_to_register(&dst, writer, reg_alloc, stack_slots)?
                    }
                    CopyLocation::Value(Operand::Register(Register::Physical(dst))) => {
                        writeln!(writer, "    mr {}, 3", dst.name)?
                    }
                    CopyLocation::Scratch => writeln!(writer, "    mr 0, 3")?,
                    _ => unreachable!(),
                }
            }
            if func.entry != next {
                writeln!(writer, "    b {prefix}{}", func.entry)?;
            }
        } else if plan.comparisons.contains(&(index, position)) {
            if let (
                MirInst::IntCmp {
                    op, lhs, rhs, ty, ..
                },
                MirInst::Br {
                    true_target,
                    false_target,
                    ..
                },
            ) = (inst, &block.instructions[position + 1])
            {
                load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
                load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
                let unsigned = matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                );
                extend_integer(writer, "3", ty, !unsigned)?;
                extend_integer(writer, "4", ty, !unsigned)?;
                let compare = if unsigned { "cmpld" } else { "cmpd" };
                writeln!(writer, "    {compare} 3, 4")?;
                let (invert, taken, other) =
                    plan.branch(func, index, next, true_target, false_target);
                let mnemonic = match (op, invert) {
                    (IntCmpOp::Eq, false) | (IntCmpOp::Ne, true) => "beq",
                    (IntCmpOp::Ne, false) | (IntCmpOp::Eq, true) => "bne",
                    (IntCmpOp::SLt | IntCmpOp::ULt, false)
                    | (IntCmpOp::SGe | IntCmpOp::UGe, true) => "blt",
                    (IntCmpOp::SLe | IntCmpOp::ULe, false)
                    | (IntCmpOp::SGt | IntCmpOp::UGt, true) => "ble",
                    (IntCmpOp::SGt | IntCmpOp::UGt, false)
                    | (IntCmpOp::SLe | IntCmpOp::ULe, true) => "bgt",
                    (IntCmpOp::SGe | IntCmpOp::UGe, false)
                    | (IntCmpOp::SLt | IntCmpOp::ULt, true) => "bge",
                };
                writeln!(writer, "    {mnemonic} {prefix}{taken}")?;
                if let Some(other) = other {
                    writeln!(writer, "    b {prefix}{other}")?;
                }
            }
            position += 1;
        } else if matches!(inst, MirInst::Jmp { target } if target == next) {
        } else {
            emit_instruction_ppc64(
                inst,
                writer,
                reg_alloc,
                stack_slots,
                emit,
                debug_line,
                cmp_seq,
            )?;
        }
        position += 1;
    }
    Ok(())
}

fn emit_instruction_ppc64<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &mut Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    emit: &Ppc64Emit<'_>,
    debug_line: &mut u32,
    cmp_seq: &mut u32,
) -> Result<(), LaminaError> {
    let Ppc64Emit {
        target_os,
        block_prefix,
        settings,
        local_bytes,
        signatures,
        ret_float,
    } = *emit;
    if settings.emit_asm_debug_lines {
        *debug_line = debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", *debug_line)?;
    }
    if let Some(operations) = integer_conversion(inst) {
        for operation in operations {
            emit_instruction_ppc64(
                &operation,
                writer,
                reg_alloc,
                stack_slots,
                emit,
                debug_line,
                cmp_seq,
            )?;
        }
        return Ok(());
    }
    match inst {
        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if matches!(ty, MirType::Vector(_)) {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("PowerPC64 backend does not support vector Copy of type {ty}"),
                )));
            }
            match src {
                Operand::Register(Register::Virtual(vreg)) => {
                    if let Some(physical) = reg_alloc.get_mapping_for(vreg) {
                        writeln!(writer, "    mr 3, {physical}")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        r1_mem(writer, copy_load_mnemonic(ty), "3", *offset)?;
                    } else {
                        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                            format!("PowerPC64 Copy source {vreg:?} has no register or stack slot"),
                        )));
                    }
                }
                Operand::Register(Register::Physical(physical)) => {
                    writeln!(writer, "    mr 3, {}", physical.name)?;
                }
                Operand::Immediate(_) => {
                    load_operand_to_register(src, writer, reg_alloc, stack_slots, "3")?;
                }
            }
            extend_integer(writer, "3", ty, true)?;
            match dst {
                Register::Virtual(vreg) => {
                    if let Some(physical) = reg_alloc.get_mapping_for(vreg) {
                        writeln!(writer, "    mr {physical}, 3")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        r1_mem(writer, "std", "3", *offset)?;
                    } else {
                        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                            format!(
                                "PowerPC64 Copy destination {vreg:?} has no register or stack slot"
                            ),
                        )));
                    }
                }
                Register::Physical(phys) if phys.name != "3" => {
                    writeln!(writer, "    mr {}, 3", phys.name)?;
                }
                Register::Physical(_) => {}
            }
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
            if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                extend_integer(writer, "3", ty, false)?;
                extend_integer(writer, "4", ty, false)?;
            }
            if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                extend_integer(writer, "3", ty, true)?;
                if !matches!(op, IntBinOp::AShr) {
                    extend_integer(writer, "4", ty, true)?;
                }
            }
            let (zero, overflow) =
                op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant());
            if zero {
                writeln!(writer, "    tdi 4, 4, 0")?;
            }
            if overflow {
                let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
                writeln!(writer, "    cmpdi 4, -1\n    bne 2f")?;
                emit_load_imm64(writer, 5, minimum)?;
                writeln!(writer, "    td 4, 3, 5\n2:")?;
            }
            if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) {
                let clear = 64 - (ty.size_bytes() * 8).trailing_zeros();
                writeln!(writer, "    clrldi 4, 4, {clear}")?;
            }
            match op {
                IntBinOp::Add => writeln!(writer, "    add 3, 3, 4")?,
                IntBinOp::Sub => writeln!(writer, "    sub 3, 3, 4")?,
                IntBinOp::Mul => writeln!(writer, "    mulld 3, 3, 4")?,
                IntBinOp::SDiv => writeln!(writer, "    divd 3, 3, 4")?,
                IntBinOp::UDiv => writeln!(writer, "    divdu 3, 3, 4")?,
                IntBinOp::SRem => {
                    // rem = lhs - (lhs/rhs)*rhs
                    writeln!(writer, "    divd 5, 3, 4")?;
                    writeln!(writer, "    mulld 5, 5, 4")?;
                    writeln!(writer, "    sub 3, 3, 5")?;
                }
                IntBinOp::URem => {
                    writeln!(writer, "    divdu 5, 3, 4")?;
                    writeln!(writer, "    mulld 5, 5, 4")?;
                    writeln!(writer, "    sub 3, 3, 5")?;
                }
                IntBinOp::And => writeln!(writer, "    and 3, 3, 4")?,
                IntBinOp::Or => writeln!(writer, "    or 3, 3, 4")?,
                IntBinOp::Xor => writeln!(writer, "    xor 3, 3, 4")?,
                IntBinOp::Shl => writeln!(writer, "    sld 3, 3, 4")?,
                IntBinOp::AShr => writeln!(writer, "    srad 3, 3, 4")?,
                IntBinOp::LShr => writeln!(writer, "    srd 3, 3, 4")?,
            }
            extend_integer(writer, "3", ty, true)?;
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "3")?;
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "4")?;
            if matches!(
                op,
                IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
            ) {
                extend_integer(writer, "3", ty, false)?;
                extend_integer(writer, "4", ty, false)?;
            }
            // cmpd sets CR0; use branch + li to materialise 0/1 into r3.
            writeln!(writer, "    cmpd 3, 4")?;
            let (set_true, set_false) = match op {
                IntCmpOp::Eq => ("beq", "bne"),
                IntCmpOp::Ne => ("bne", "beq"),
                IntCmpOp::SLt => ("blt", "bge"),
                IntCmpOp::SLe => ("ble", "bgt"),
                IntCmpOp::SGt => ("bgt", "ble"),
                IntCmpOp::SGe => ("bge", "blt"),
                // Unsigned: use cmplw/cmpld
                IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe => {
                    // Redo with unsigned compare
                    writeln!(writer, "    cmpld 3, 4")?;
                    match op {
                        IntCmpOp::ULt => ("blt", "bge"),
                        IntCmpOp::ULe => ("ble", "bgt"),
                        IntCmpOp::UGt => ("bgt", "ble"),
                        _ => ("bge", "blt"),
                    }
                }
            };
            let seq = *cmp_seq;
            *cmp_seq = cmp_seq.saturating_add(1);
            let prefix = emit.block_prefix;
            writeln!(writer, "    {set_true} {prefix}cmp{seq}.true")?;
            writeln!(writer, "    li 3, 0")?;
            writeln!(writer, "    b {prefix}cmp{seq}.end")?;
            writeln!(writer, "{prefix}cmp{seq}.true:")?;
            writeln!(writer, "    li 3, 1")?;
            writeln!(writer, "{prefix}cmp{seq}.end:")?;
            let _ = set_false; // used in branch selection above
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;
            // Double has no suffix on PowerPC; only single takes the s form.
            let suffix = if is_f32 { "s" } else { "" };
            // Load float operands from integer stack slots into FPRs via memory.
            emit_load_fp_operand(lhs, writer, reg_alloc, stack_slots, "1", is_f32)?;
            emit_load_fp_operand(rhs, writer, reg_alloc, stack_slots, "2", is_f32)?;
            match op {
                FloatBinOp::FAdd => writeln!(writer, "    fadd{suffix} 1, 1, 2")?,
                FloatBinOp::FSub => writeln!(writer, "    fsub{suffix} 1, 1, 2")?,
                FloatBinOp::FMul => writeln!(writer, "    fmul{suffix} 1, 1, 2")?,
                FloatBinOp::FDiv => writeln!(writer, "    fdiv{suffix} 1, 1, 2")?,
            }
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let is_f32 = from.size_bytes() == 4;
            emit_load_fp_operand(src, writer, reg_alloc, stack_slots, "1", is_f32)?;
            // The convert saturates but turns NaN into the most negative integer, so
            // the unordered bit of a self-comparison picks zero instead.
            writeln!(writer, "    fcmpu 0, 1, 1")?;
            let convert = if *signed { "fctidz" } else { "fctiduz" };
            writeln!(writer, "    {convert} 1, 1")?;
            writeln!(writer, "    stfd 1, -8(1)")?;
            writeln!(writer, "    ld 3, -8(1)")?;
            writeln!(writer, "    li 4, 0")?;
            writeln!(writer, "    isel 3, 4, 3, 3")?;
            let bits = to.size_bytes() * 8;
            if bits < 64 {
                let compare = if *signed { "cmpd" } else { "cmpld" };
                let max = if *signed {
                    (1i64 << (bits - 1)) - 1
                } else {
                    (1i64 << bits) - 1
                };
                emit_load_imm64(writer, 4, max)?;
                writeln!(writer, "    {compare} 0, 3, 4")?;
                writeln!(writer, "    isel 3, 4, 3, 1")?;
                if *signed {
                    emit_load_imm64(writer, 4, -(1i64 << (bits - 1)))?;
                    writeln!(writer, "    cmpd 0, 3, 4")?;
                    writeln!(writer, "    isel 3, 4, 3, 0")?;
                }
                extend_integer(writer, "3", to, true)?;
            }
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let is_f32 = to.size_bytes() == 4;
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "3")?;
            let bits = from.size_bytes() * 8;
            if bits < 64 {
                if *signed {
                    extend_integer(writer, "3", from, true)?;
                } else {
                    writeln!(writer, "    clrldi 3, 3, {}", 64 - bits)?;
                }
            }
            writeln!(writer, "    std 3, -8(1)")?;
            writeln!(writer, "    lfd 1, -8(1)")?;
            let convert = match (*signed, is_f32) {
                (true, false) => "fcfid",
                (false, false) => "fcfidu",
                (true, true) => "fcfids",
                (false, true) => "fcfidus",
            };
            writeln!(writer, "    {convert} 1, 1")?;
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let to_f32 = to.size_bytes() == 4;
            emit_load_fp_operand(
                src,
                writer,
                reg_alloc,
                stack_slots,
                "1",
                from.size_bytes() == 4,
            )?;
            if to_f32 {
                writeln!(writer, "    frsp 1, 1")?;
            }
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, to_f32)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let is_f32 = ty.size_bytes() == 4;
            let suffix = if is_f32 { "s" } else { "" };
            emit_load_fp_operand(src, writer, reg_alloc, stack_slots, "1", is_f32)?;
            match op {
                FloatUnOp::FNeg => writeln!(writer, "    fneg 1, 1")?,
                FloatUnOp::FSqrt => writeln!(writer, "    fsqrt{suffix} 1, 1")?,
            }
            if let Register::Virtual(vreg) = dst {
                emit_store_fp_result("1", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;
            emit_load_fp_operand(lhs, writer, reg_alloc, stack_slots, "1", is_f32)?;
            emit_load_fp_operand(rhs, writer, reg_alloc, stack_slots, "2", is_f32)?;
            writeln!(writer, "    fcmpu 0, 1, 2")?;
            let branch = match op {
                FloatCmpOp::Eq => "beq",
                FloatCmpOp::Ne => "bne",
                FloatCmpOp::Lt => "blt",
                FloatCmpOp::Le => "ble",
                FloatCmpOp::Gt => "bgt",
                FloatCmpOp::Ge => "bge",
            };
            let seq = *cmp_seq;
            *cmp_seq = cmp_seq.saturating_add(1);
            let prefix = emit.block_prefix;
            writeln!(writer, "    {branch} {prefix}fcmp{seq}.true")?;
            writeln!(writer, "    li 3, 0")?;
            writeln!(writer, "    b {prefix}fcmp{seq}.end")?;
            writeln!(writer, "{prefix}fcmp{seq}.true:")?;
            writeln!(writer, "    li 3, 1")?;
            writeln!(writer, "{prefix}fcmp{seq}.end:")?;
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // Materialise both values, then pick using isel or a branch.
            load_operand_to_register(false_val, writer, reg_alloc, stack_slots, "4")?;
            load_operand_to_register(true_val, writer, reg_alloc, stack_slots, "3")?;
            // Test condition in r5.
            match cond {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
            }
            writeln!(writer, "    cmpdi 5, 0")?;
            // isel: if CR0[EQ] is set (cond == 0), select false (r4); else true (r3)
            writeln!(writer, "    isel 3, 4, 3, 2")?; // CR0 bit 2 = EQ
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            match value {
                Register::Virtual(v) => {
                    load_register_to_r3(v, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            for (case_val, case_label) in cases {
                writeln!(writer, "    cmpdi 3, {case_val}")?;
                writeln!(writer, "    beq {block_prefix}{case_label}")?;
            }
            writeln!(writer, "    b {block_prefix}{default}")?;
        }
        MirInst::Call { name, args, ret } => {
            let abi = Ppc64Abi::new(target_os);
            let arg_regs = Ppc64Abi::ARG_REGISTERS;

            if name == "writeptr" && args.len() == 2 {
                // write(1, ptr, width): the pointee's bytes go out straight from memory.
                let Operand::Immediate(Immediate::I64(width)) = &args[1] else {
                    return Err(LaminaError::ValidationError(
                        "writeptr width must be an immediate".to_string(),
                    ));
                };
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "4")?;
                writeln!(writer, "    li 5, {width}")?;
                writeln!(writer, "    li 3, 1")?;
                let write = abi
                    .call_stub("write")
                    .unwrap_or_else(|| abi.mangle_function_name("write"));
                writeln!(writer, "    bl {write}")?;
                writeln!(writer, "    nop")?;
            } else if (name == "write" || name == "read") && args.len() == 2 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "4")?;
                load_operand_to_register(&args[1], writer, reg_alloc, stack_slots, "5")?;
                writeln!(writer, "    li 3, {}", i32::from(name == "write"))?;
                let callee = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                writeln!(writer, "    bl {callee}")?;
                writeln!(writer, "    nop")?;
            } else if name == "writebyte" && args.len() == 1 {
                // putchar reports the character written, or -1; writebyte reports the
                // byte count, so turn the result into 1 or -1.
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "3")?;
                writeln!(writer, "    clrldi 3, 3, 56")?;
                let putchar = abi
                    .call_stub("putchar")
                    .unwrap_or_else(|| abi.mangle_function_name("putchar"));
                writeln!(writer, "    bl {putchar}")?;
                writeln!(writer, "    nop")?;
                writeln!(writer, "    li 4, -1")?;
                writeln!(writer, "    cmpd 0, 3, 4")?;
                writeln!(writer, "    li 3, 1")?;
                writeln!(writer, "    isel 3, 4, 3, 2")?;
            } else if name == "readbyte" && args.is_empty() {
                // getchar reports the byte or -1 at end of input, where readbyte
                // reports zero.
                let getchar = abi
                    .call_stub("getchar")
                    .unwrap_or_else(|| abi.mangle_function_name("getchar"));
                writeln!(writer, "    bl {getchar}")?;
                writeln!(writer, "    nop")?;
                writeln!(writer, "    li 4, 0")?;
                writeln!(writer, "    cmpdi 3, 0")?;
                writeln!(writer, "    isel 3, 4, 3, 0")?;
            } else if name == "printf32" || name == "printf64" {
                if let Some(arg) = args.first() {
                    writeln!(writer, "    addis 3, 2, .L_mir_fmt_float@toc@ha")?;
                    writeln!(writer, "    addi 3, 3, .L_mir_fmt_float@toc@l")?;
                    emit_load_fp_operand(
                        arg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        "1",
                        name == "printf32",
                    )?;
                    writeln!(writer, "    stfd 1, -8(1)")?;
                    writeln!(writer, "    ld 4, -8(1)")?;
                    writeln!(writer, "    bl printf")?;
                    writeln!(writer, "    nop")?;
                }
            } else if name == "print" || name == "printu" {
                if let Some(arg) = args.first() {
                    let fmt = if name == "printu" {
                        ".L_mir_fmt_uint"
                    } else {
                        ".L_mir_fmt_int"
                    };
                    writeln!(writer, "    addis 3, 2, {fmt}@toc@ha")?;
                    writeln!(writer, "    addi 3, 3, {fmt}@toc@l")?;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "4")?;
                    writeln!(writer, "    bl printf")?;
                    writeln!(writer, "    nop")?;
                }
            } else {
                let types = signatures.arguments(name, args.len());
                let places = classify_ppc64_arguments(&types);
                for ((arg, ty), place) in args.iter().zip(&types).zip(&places) {
                    match *place {
                        PpcArgPlace::Gpr(bank) => {
                            load_operand_to_register(
                                arg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                arg_regs[bank],
                            )?;
                        }
                        PpcArgPlace::Fpr(bank) => {
                            emit_load_fp_operand(
                                arg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                &bank.to_string(),
                                ty.size_bytes() == 4,
                            )?;
                        }
                        PpcArgPlace::Stack => {}
                    }
                }

                let outgoing = outgoing_frame_size(args.len())?;
                if let Some(frame) = outgoing {
                    // The parameter save area starts above the linkage area and holds a
                    // slot for every parameter, register ones included. `stdu` moves r1
                    // and writes the back chain in one step, which the slot offsets then
                    // have to account for.
                    writeln!(writer, "    stdu 1, -{frame}(1)")?;
                    let moved = biased_slots(stack_slots, frame as i32);
                    for ((index, arg), place) in args.iter().enumerate().zip(&places) {
                        if !matches!(place, PpcArgPlace::Stack) {
                            continue;
                        }
                        load_operand_to_register(arg, writer, reg_alloc, &moved, "11")?;
                        r1_mem(writer, "std", "11", parameter_save_offset(index))?;
                    }
                }

                let target_sym = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                writeln!(writer, "    bl {target_sym}")?;
                writeln!(writer, "    nop")?;

                if let Some(frame) = outgoing {
                    writeln!(writer, "    addi 1, 1, {frame}")?;
                }
            }

            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                match signatures.float_result(name) {
                    Some(ty) => emit_store_fp_result(
                        "1",
                        vreg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        ty.size_bytes() == 4,
                    )?,
                    None => store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::TailCall { name, args } => {
            if name == "writeptr" {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "PowerPC64: writeptr is not supported".to_string(),
                )));
            }
            let abi = Ppc64Abi::new(target_os);
            let arg_regs = Ppc64Abi::ARG_REGISTERS;
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());
            for (i, arg) in args.iter().take(num_reg_args).enumerate() {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[i])?;
            }
            let target_sym = abi
                .call_stub(name)
                .unwrap_or_else(|| abi.mangle_function_name(name));
            if num_stack_args == 0 {
                Ppc64Frame::generate_tail_epilogue(writer, local_bytes).map_err(|e| {
                    LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(e.to_string()))
                })?;
                writeln!(writer, "    b {target_sym}")?;
            } else {
                // Stack arguments belong in this frame's own save area, which the
                // caller's frame has no room for, so the tail call becomes a call and
                // a return.
                let frame = outgoing_frame_size(args.len())?.unwrap_or(0);
                writeln!(writer, "    stdu 1, -{frame}(1)")?;
                let moved = biased_slots(stack_slots, frame as i32);
                for (index, arg) in args.iter().enumerate().skip(num_reg_args) {
                    load_operand_to_register(arg, writer, reg_alloc, &moved, "11")?;
                    r1_mem(writer, "std", "11", parameter_save_offset(index))?;
                }
                writeln!(writer, "    bl {target_sym}")?;
                writeln!(writer, "    nop")?;
                writeln!(writer, "    addi 1, 1, {frame}")?;
                Ppc64Frame::generate_epilogue(writer, local_bytes).map_err(|e| {
                    LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(e.to_string()))
                })?;
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            let load_op = match ty {
                MirType::Scalar(ScalarType::I1) | MirType::Scalar(ScalarType::I8) => "lbz",
                MirType::Scalar(ScalarType::I16) => "lhz",
                MirType::Scalar(ScalarType::I32) => "lwz",
                MirType::Scalar(ScalarType::I64)
                | MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => "ld",
                MirType::Scalar(ScalarType::F32) => "lfs",
                MirType::Scalar(ScalarType::F64) => "lfd",
                other => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!("PowerPC64 load: unsupported type {other:?}"),
                    )));
                }
            };
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                        }
                        Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
                    }
                    writeln!(writer, "    {load_op} 3, {offset}(5)")?;
                    extend_integer(writer, "3", ty, true)?;
                    if let Register::Virtual(vreg) = dst {
                        if matches!(ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64)) {
                            emit_store_fp_result(
                                "3",
                                vreg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                *ty == MirType::Scalar(ScalarType::F32),
                            )?;
                        } else {
                            store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
                        }
                    }
                }
                _ => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        "PowerPC64 load: only base+offset addressing supported".to_string(),
                    )));
                }
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs: _,
        } => {
            let store_op = match ty {
                MirType::Scalar(ScalarType::I1) | MirType::Scalar(ScalarType::I8) => "stb",
                MirType::Scalar(ScalarType::I16) => "sth",
                MirType::Scalar(ScalarType::I32) => "stw",
                MirType::Scalar(ScalarType::I64)
                | MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => "std",
                MirType::Scalar(ScalarType::F32) => "stfs",
                MirType::Scalar(ScalarType::F64) => "stfd",
                other => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!("PowerPC64 store: unsupported type {other:?}"),
                    )));
                }
            };
            if matches!(ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64)) {
                emit_load_fp_operand(
                    src,
                    writer,
                    reg_alloc,
                    stack_slots,
                    "3",
                    *ty == MirType::Scalar(ScalarType::F32),
                )?;
            } else {
                load_operand_to_register(src, writer, reg_alloc, stack_slots, "3")?;
            }
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "5")?;
                        }
                        Register::Physical(p) => writeln!(writer, "    mr 5, {}", p.name)?,
                    }
                    writeln!(writer, "    {store_op} 3, {offset}(5)")?;
                }
                _ => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        "PowerPC64 store: only base+offset addressing supported".to_string(),
                    )));
                }
            }
        }
        MirInst::SymbolAddress { dst, symbol } => {
            let label = Ppc64Abi::new(target_os).mangle_function_name(symbol);
            writeln!(writer, "    addis 3, 2, {label}@toc@ha")?;
            writeln!(writer, "    addi 3, 3, {label}@toc@l")?;
            match dst {
                Register::Virtual(vreg) => {
                    store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(phys) if phys.name != "3" => {
                    writeln!(writer, "    mr {}, 3", phys.name)?;
                }
                Register::Physical(_) => {}
            }
        }
        MirInst::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).ok_or_else(|| {
                LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
                    "PowerPC64 stack address {slot:?} has no frame slot"
                )))
            })?;
            r1_address(writer, "3", *offset)?;
            match dst {
                Register::Virtual(vreg) => {
                    store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(phys) if phys.name != "3" => {
                    writeln!(writer, "    mr {}, 3", phys.name)?;
                }
                Register::Physical(_) => {}
            }
        }
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "3")?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            if i16::try_from(*offset).is_ok() {
                if *offset != 0 {
                    writeln!(writer, "    addi 3, 3, {offset}")?;
                }
            } else {
                load_imm(writer, "0", i64::from(*offset))?;
                writeln!(writer, "    add 3, 3, 0")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    b {block_prefix}{target}")?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            match cond {
                Register::Virtual(v) => {
                    load_register_to_r3(v, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => writeln!(writer, "    mr 3, {}", p.name)?,
            }
            writeln!(writer, "    cmpdi 3, 0")?;
            writeln!(writer, "    bne {block_prefix}{true_target}")?;
            writeln!(writer, "    b {block_prefix}{false_target}")?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match ret_float {
                    Some(ty) => emit_load_fp_operand(
                        val,
                        writer,
                        reg_alloc,
                        stack_slots,
                        "1",
                        ty.size_bytes() == 4,
                    )?,
                    None => load_operand_to_register(val, writer, reg_alloc, stack_slots, "3")?,
                }
            }
            Ppc64Frame::generate_epilogue(writer, local_bytes).map_err(|e| {
                LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(e.to_string()))
            })?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        other => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("PowerPC64 backend: instruction not yet supported: {other:?}"),
            )));
        }
    }
    Ok(())
}

fn copy_load_mnemonic(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "lbz",
        2 => "lhz",
        4 => "lwz",
        _ => "ld",
    }
}

/// Load a float operand (stored as integer bits in a stack slot) into FPR `fN`.
///
/// Strategy: store the integer bits to a temp stack slot and then `lfs`/`lfd` them
/// into the FPR, which reinterprets the bit pattern as a float.
fn emit_load_fp_operand<W: Write>(
    operand: &Operand,
    writer: &mut W,
    reg_alloc: &mut Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    fpr: &str,
    is_f32: bool,
) -> Result<(), LaminaError> {
    let load_fp = if is_f32 { "lfs" } else { "lfd" };
    match operand {
        Operand::Register(reg) => {
            match reg {
                Register::Virtual(v) => {
                    if let Some(offset) = stack_slots.get(v) {
                        r1_mem(writer, load_fp, fpr, *offset)?;
                    } else if let Some(phys) = reg_alloc.get_mapping_for(v) {
                        // Store to temp stack location then load as float
                        writeln!(writer, "    std {phys}, -8(1)")?;
                        writeln!(writer, "    {load_fp} {fpr}, -8(1)")?;
                    }
                }
                Register::Physical(p) => {
                    writeln!(writer, "    std {}, -8(1)", p.name)?;
                    writeln!(writer, "    {load_fp} {fpr}, -8(1)")?;
                }
            }
        }
        Operand::Immediate(imm) => {
            let bits: i64 = match imm {
                Immediate::F32(v) => v.to_bits() as i64,
                Immediate::F64(v) => v.to_bits() as i64,
                Immediate::I32(v) => *v as i64,
                Immediate::I64(v) => *v,
                other => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!("PowerPC64: float immediate {other:?} not supported"),
                    )));
                }
            };
            emit_load_imm64(writer, 11, bits)?;
            writeln!(writer, "    std 11, -8(1)")?;
            writeln!(writer, "    {load_fp} {fpr}, -8(1)")?;
        }
    }
    Ok(())
}

/// Materialise a full 64-bit constant. `li` only carries a signed 16-bit field.
/// Offset of a parameter's slot in the caller's save area. Every parameter has one,
/// including those passed in registers, so the slots of the first eight are skipped
/// rather than reused.
fn parameter_save_offset(index: usize) -> i32 {
    (LINKAGE_SIZE + index * 8) as i32
}

/// Frame a call needs for its stack arguments, or `None` when every argument fits in
/// a register.
fn outgoing_frame_size(args: usize) -> Result<Option<usize>, LaminaError> {
    if args <= Ppc64Abi::ARG_REGISTERS.len() {
        return Ok(None);
    }
    let frame = (LINKAGE_SIZE + args * 8 + 15) & !15;
    if frame > 32767 {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "PowerPC64 call needs a {frame} byte parameter save area, past the 16-bit displacement range"
            ),
        )));
    }
    Ok(Some(frame))
}

/// Frame slot offsets seen from an r1 that has moved down by `frame`.
fn biased_slots(slots: &HashMap<VirtualReg, i32>, frame: i32) -> HashMap<VirtualReg, i32> {
    slots
        .iter()
        .map(|(vreg, offset)| (*vreg, offset + frame))
        .collect()
}

fn emit_load_imm64<W: Write>(writer: &mut W, reg: u8, value: i64) -> Result<(), LaminaError> {
    if (-0x8000..0x8000).contains(&value) {
        writeln!(writer, "    li {reg}, {value}")?;
        return Ok(());
    }
    let v = value as u64;
    let hi = ((v >> 48) & 0xffff) as u16;
    let hm = ((v >> 32) & 0xffff) as u16;
    let lm = ((v >> 16) & 0xffff) as u16;
    let lo = (v & 0xffff) as u16;
    writeln!(writer, "    lis {reg}, {}", hi as i16)?;
    writeln!(writer, "    ori {reg}, {reg}, {hm}")?;
    writeln!(writer, "    rldicr {reg}, {reg}, 32, 31")?;
    writeln!(writer, "    oris {reg}, {reg}, {lm}")?;
    writeln!(writer, "    ori {reg}, {reg}, {lo}")?;
    Ok(())
}

/// Store FPR result back to a virtual register stack slot as integer bits.
fn emit_store_fp_result<W: Write>(
    fpr: &str,
    vreg: &VirtualReg,
    writer: &mut W,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    is_f32: bool,
) -> Result<(), LaminaError> {
    let store_fp = if is_f32 { "stfs" } else { "stfd" };
    if let Some(offset) = stack_slots.get(vreg) {
        r1_mem(writer, store_fp, fpr, *offset)?;
    } else if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
        writeln!(writer, "    {store_fp} {fpr}, -8(1)")?;
        writeln!(writer, "    ld {phys}, -8(1)")?;
    }
    Ok(())
}

fn extend_param(w: &mut impl Write, register: &str, ty: &MirType) -> Result<(), LaminaError> {
    if matches!(ty, MirType::Scalar(ScalarType::I1)) {
        writeln!(w, "    clrldi {register}, {register}, 56")?;
        return Ok(());
    }
    extend_integer(w, register, ty, true)
}

fn extend_integer(
    w: &mut impl Write,
    register: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    let instruction = match ty {
        MirType::Scalar(ScalarType::I8) => "extsb",
        MirType::Scalar(ScalarType::I16) => "extsh",
        MirType::Scalar(ScalarType::I32) => "extsw",
        _ => return Ok(()),
    };
    if signed {
        writeln!(w, "    {instruction} {register}, {register}")?;
    } else {
        let clear = 64 - ty.size_bytes() * 8;
        writeln!(w, "    clrldi {register}, {register}, {clear}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder,
        instruction::{Instruction, IntBinOp, Operand},
    };
    use std::error::Error;

    fn make_module_with_empty_func() -> MirModule {
        let mut m = MirModule::new("test_ppc64");
        let f = FunctionBuilder::new("test_fn")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Ret { value: None })
            .build();
        m.add_function(f);
        m
    }

    /// Replays the constant loads that build the frame size and the first address built
    /// from `r1`. Returns the frame size and that address, both as offsets from `r1`.
    fn ppc64_frame_arithmetic(asm: &str) -> (i64, Option<i64>) {
        let mut regs: HashMap<&str, i64> = HashMap::new();
        let (mut frame, mut address) = (0i64, None);
        for line in asm.lines().map(str::trim) {
            let (op, args) = line.split_once(' ').unwrap_or((line, ""));
            let parts: Vec<&str> = args.split(", ").collect();
            match (op, parts.as_slice()) {
                ("li", [r, v]) => {
                    regs.insert(r, v.parse().unwrap_or(0));
                }
                ("lis", [r, v]) => {
                    regs.insert(r, v.parse::<i64>().unwrap_or(0) << 16);
                }
                ("ori", [r, _, v]) => {
                    *regs.entry(r).or_insert(0) |= v.parse::<i64>().unwrap_or(0);
                }
                ("addis", [r, base, v]) => {
                    let base = if *base == "0" {
                        0
                    } else {
                        regs.get(base).copied().unwrap_or(0)
                    };
                    regs.insert(r, base + (v.parse::<i64>().unwrap_or(0) << 16));
                }
                ("stdux", _) => frame = -regs.get("12").copied().unwrap_or(0),
                ("stdu", [_, v]) => {
                    frame = -v.trim_end_matches("(1)").parse::<i64>().unwrap_or(0);
                }
                ("addi", ["3", "1", v]) => {
                    let v = v.parse().unwrap_or(0);
                    regs.insert("3", v);
                    address.get_or_insert(v);
                }
                ("add", ["3", "1", "3"]) => {
                    address.get_or_insert(regs.get("3").copied().unwrap_or(0));
                }
                _ => {}
            }
        }
        (frame, address)
    }

    #[test]
    fn lea_offset_beyond_the_immediate_range_is_added_through_r0() -> Result<(), Box<dyn Error>> {
        let slot = VirtualReg::gpr(1);
        let base = Register::Virtual(VirtualReg::gpr(0));
        let mut function = FunctionBuilder::new("element")
            .returns(MirType::ptr())
            .block("entry")
            .instr(MirInst::StackAddr {
                dst: base.clone(),
                slot,
            })
            .instr(MirInst::Lea {
                dst: Register::Virtual(VirtualReg::gpr(2)),
                base,
                offset: 39999,
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(2)))),
            })
            .build();
        function.stack_reservations.insert(slot, 4999);
        let mut module = MirModule::new("element");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_ppc64_with_units_and_settings(
            &module,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )?;
        let assembly = String::from_utf8(output)?;
        assert!(!assembly.contains("addi 3, 3, 39999"), "{assembly}");
        assert!(assembly.contains("    add 3, 3, 0\n"), "{assembly}");
        Ok(())
    }

    #[test]
    fn stack_reservation_is_reserved_in_the_prologue() -> Result<(), Box<dyn Error>> {
        let slot = VirtualReg::gpr(1);
        let mut function = FunctionBuilder::new("reserved")
            .returns(MirType::ptr())
            .block("entry")
            .instr(MirInst::StackAddr {
                dst: Register::Virtual(VirtualReg::gpr(0)),
                slot,
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(Register::Virtual(VirtualReg::gpr(0)))),
            })
            .build();
        function.stack_reservations.insert(slot, 4999);
        let mut module = MirModule::new("reserved");
        module.add_function(function);
        for regalloc in [
            RegallocStrategy::Incremental,
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc,
                ..Default::default()
            };
            let mut output = Vec::new();
            generate_mir_ppc64_with_units_and_settings(
                &module,
                &mut output,
                TargetOperatingSystem::Linux,
                1,
                &settings,
            )?;
            let assembly = String::from_utf8(output)?;
            let (frame, address) = ppc64_frame_arithmetic(&assembly);
            let address = address.unwrap_or(i64::MIN);
            assert!(
                frame >= address + 40000,
                "{regalloc:?}: frame {frame} {assembly}"
            );
            assert!(address >= 32, "{regalloc:?}: address {address} {assembly}");
        }
        Ok(())
    }

    #[test]
    fn stack_address_reserves_slot_without_register_use() {
        let slot = VirtualReg::gpr(10);
        let address = Register::Virtual(VirtualReg::gpr(0));
        let stack_addr = MirInst::StackAddr {
            dst: address.clone(),
            slot,
        };
        assert!(stack_addr.use_regs().is_empty());
        let mut module = MirModule::new("stack_address");
        module.add_function(
            FunctionBuilder::new("address")
                .returns(MirType::ptr())
                .block("entry")
                .instr(stack_addr)
                .instr(MirInst::Ret {
                    value: Some(Operand::Register(address)),
                })
                .build(),
        );
        for regalloc in [
            RegallocStrategy::Incremental,
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc,
                ..Default::default()
            };
            let mut output = Vec::new();
            generate_mir_ppc64_with_units_and_settings(
                &module,
                &mut output,
                TargetOperatingSystem::Linux,
                1,
                &settings,
            )
            .unwrap();
            let assembly = String::from_utf8(output).unwrap();
            assert!(
                assembly.contains("    stdu 1, -48(1)"),
                "{regalloc:?}: {assembly}"
            );
            assert!(
                assembly.contains("    addi 3, 1, 40"),
                "{regalloc:?}: {assembly}"
            );
            assert!(
                !assembly.contains("    ld 3, 40(1)"),
                "{regalloc:?}: {assembly}"
            );
        }
    }

    #[test]
    fn narrow_parameters_are_sign_extended_on_entry() {
        for (ty, extend) in [
            (MirType::Scalar(ScalarType::I8), "extsb 3, 3"),
            (MirType::Scalar(ScalarType::I16), "extsh 3, 3"),
            (MirType::Scalar(ScalarType::I32), "extsw 3, 3"),
        ] {
            let reg = Register::Virtual(VirtualReg::gpr(0));
            let mut module = MirModule::new("narrow_entry");
            module.add_function(
                FunctionBuilder::new("narrow")
                    .param(reg.clone(), ty)
                    .returns(ty)
                    .block("entry")
                    .instr(MirInst::Ret {
                        value: Some(Operand::Register(reg)),
                    })
                    .build(),
            );
            let mut output = Vec::new();
            generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
            let s = String::from_utf8(output).unwrap();
            assert!(s.contains(&format!("    {extend}\n    std 3, ")), "{s}");
        }
    }

    #[test]
    fn bool_parameters_are_zero_extended_on_entry() {
        let reg = Register::Virtual(VirtualReg::gpr(0));
        let ty = MirType::Scalar(ScalarType::I1);
        let mut module = MirModule::new("flag_entry");
        module.add_function(
            FunctionBuilder::new("flag")
                .param(reg.clone(), ty)
                .returns(ty)
                .block("entry")
                .instr(MirInst::Ret {
                    value: Some(Operand::Register(reg)),
                })
                .build(),
        );
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("    clrldi 3, 3, 56\n    std 3, "), "{s}");
    }

    #[test]
    fn float_memory_transfers_use_fpr_bits() {
        for (ty, load, store) in [
            (MirType::Scalar(ScalarType::F32), "lfs", "stfs"),
            (MirType::Scalar(ScalarType::F64), "lfd", "stfd"),
        ] {
            let base = Register::Virtual(VirtualReg::gpr(0));
            let value = Register::Virtual(VirtualReg::fpr(1));
            let mut module = MirModule::new("float_memory");
            module.add_function(
                FunctionBuilder::new("transfer")
                    .param(base.clone(), MirType::ptr())
                    .block("entry")
                    .instr(MirInst::Load {
                        ty,
                        dst: value.clone(),
                        addr: AddressMode::BaseOffset {
                            base: base.clone(),
                            offset: 0,
                            address_space: Default::default(),
                        },
                        attrs: Default::default(),
                    })
                    .instr(MirInst::Store {
                        ty,
                        src: Operand::Register(value),
                        addr: AddressMode::BaseOffset {
                            base,
                            offset: 8,
                            address_space: Default::default(),
                        },
                        attrs: Default::default(),
                    })
                    .instr(MirInst::Ret { value: None })
                    .build(),
            );
            let mut output = Vec::new();
            generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
            let assembly = String::from_utf8(output).unwrap();
            let lines: Vec<_> = assembly.lines().collect();
            let start = lines
                .iter()
                .position(|line| line.contains(&format!("{load} 3, 0(5)")))
                .unwrap();
            assert!(
                lines[start + 1].contains(&format!("{store} 3,")),
                "{assembly}"
            );
            assert!(lines[start + 1].ends_with("(1)"), "{assembly}");
            let end = lines
                .iter()
                .position(|line| line.contains(&format!("{store} 3, 8(5)")))
                .unwrap();
            assert!(
                lines[start + 2..end]
                    .iter()
                    .any(|line| line.contains(&format!("{load} 3,")) && line.ends_with("(1)")),
                "{assembly}"
            );
        }
    }

    #[test]
    fn stack_arguments_sit_in_the_parameter_save_area() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @ten(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j) -> i64 {\nentry:\n  %s = add.i64 %i, %j\n  ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n  %r = call @ten(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n  ret.i64 %r\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "stack_args").expect("lower stack arguments");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        // The ninth and tenth parameters sit at 32 + index * 8 of the caller's frame,
        // which `stdu` claims while writing the back chain.
        let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
        assert!(caller.contains("    stdu 1, -112(1)\n"), "{assembly}");
        assert!(caller.contains("    std 11, 96(1)\n"), "{assembly}");
        assert!(caller.contains("    std 11, 104(1)\n"), "{assembly}");
        assert!(caller.contains("    addi 1, 1, 112\n"), "{assembly}");

        let callee = assembly.split("\nten:\n").nth(1).ok_or("missing ten")?;
        let frame = callee
            .lines()
            .find_map(|line| {
                let rest = line.trim().strip_prefix("stdu 1, -")?;
                rest.strip_suffix("(1)")?.parse::<i32>().ok()
            })
            .ok_or("missing frame")?;
        assert!(
            callee.contains(&format!("    ld 0, {}(1)\n", frame + 96)),
            "{assembly}"
        );
        assert!(
            callee.contains(&format!("    ld 0, {}(1)\n", frame + 104)),
            "{assembly}"
        );
        Ok(())
    }

    #[test]
    fn byte_io_goes_through_putchar_and_getchar() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir =
            "fn @main() -> i64 {\nentry:\n  %w = writebyte 72\n  %r = readbyte\n  ret.i64 %r\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "byte_io").expect("lower byte io");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("    bl putchar\n"), "{assembly}");
        assert!(assembly.contains("    bl getchar\n"), "{assembly}");
        assert!(!assembly.contains("bl writebyte"), "{assembly}");
        assert!(!assembly.contains("bl readbyte"), "{assembly}");
        Ok(())
    }

    #[test]
    fn writeptr_sends_the_pointee_bytes_through_write() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @main() -> i32 {\nentry:\n  %p = alloc.stack i32\n  store.i32 %p, 72\n  %n = writeptr %p\n  ret.i32 0\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "writeptr").expect("lower writeptr");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("    li 5, 4\n"), "{assembly}");
        assert!(assembly.contains("    li 3, 1\n"), "{assembly}");
        assert!(assembly.contains("    bl write\n"), "{assembly}");
        Ok(())
    }

    #[test]
    fn floats_cross_the_boundary_in_f_registers() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @add(i64 %n, f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  %w = sitofp.i64.f64 %n\n  %t = add.f64 %s, %w\n  ret.f64 %t\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(7, 1.5, 2.5)\n  %i = fptosi.f64.i64 %r\n  ret.i64 %i\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "floats").expect("lower floats");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        // The integer takes r3 by its parameter position, while the two floats take f1
        // and f2 and leave r4 and r5 unused.
        let callee = assembly.split("\nadd:\n").nth(1).ok_or("missing add")?;
        let prologue = callee
            .split(".L_add.entry:")
            .next()
            .ok_or("missing entry")?;
        assert!(prologue.contains("    std 3, "), "{assembly}");
        assert!(prologue.contains("    stfd 1, "), "{assembly}");
        assert!(prologue.contains("    stfd 2, "), "{assembly}");
        assert!(!prologue.contains("    std 4, "), "{assembly}");
        assert!(callee.contains("    lfd 1, "), "{assembly}");

        let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
        let call = caller.split("bl add").next().ok_or("missing call")?;
        assert!(call.contains("    lfd 1, "), "{assembly}");
        assert!(call.contains("    lfd 2, "), "{assembly}");
        Ok(())
    }

    #[test]
    fn conversions_saturate_clamp_and_zero_nan() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @main() -> i64 {\nentry:\n  %f = add.f64 1.5, 0.0\n  %wide = fptosi.f64.i64 %f\n  %narrow = fptosi.f64.i8 %f\n  %unsigned = fptoui.f64.u32 %f\n  %back = sitofp.i64.f64 %wide\n  %from_u = uitofp.u32.f64 %unsigned\n  ret.i64 %wide\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "conversions").expect("lower conversions");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        // The unordered bit of a self-comparison selects zero for NaN, and a narrow
        // destination is clamped to its own range before it is extended.
        assert!(assembly.contains("    fcmpu 0, 1, 1\n"), "{assembly}");
        assert!(assembly.contains("    fctidz 1, 1\n"), "{assembly}");
        assert!(assembly.contains("    fctiduz 1, 1\n"), "{assembly}");
        assert!(assembly.contains("    isel 3, 4, 3, 3\n"), "{assembly}");
        assert!(assembly.contains("    cmpld 0, 3, 4\n"), "{assembly}");
        assert!(assembly.contains("    extsb 3, 3\n"), "{assembly}");
        assert!(assembly.contains("    fcfid 1, 1\n"), "{assembly}");
        assert!(assembly.contains("    fcfidu 1, 1\n"), "{assembly}");
        Ok(())
    }

    #[test]
    fn module_globals_are_addressed_through_toc_relocations() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "global @counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 @counter\n  ret.i64 %a\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "globals").expect("lower global module");
        let mut output = Vec::new();
        generate_mir_ppc64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(
            assembly.contains("    addis 3, 2, counter@toc@ha\n"),
            "{assembly}"
        );
        assert!(
            assembly.contains("    addi 3, 3, counter@toc@l\n"),
            "{assembly}"
        );
        assert!(
            assembly.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
            "{assembly}"
        );
        Ok(())
    }

    #[test]
    fn test_generate_empty_function() {
        let module = make_module_with_empty_func();
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("test_fn:"), "Expected function label: {s}");
        assert!(s.contains("blr"), "Expected blr in output: {s}");
    }

    #[test]
    fn test_generate_integer_add() {
        let mut m = MirModule::new("test_ppc64_add");
        let v0 = VirtualReg::gpr(0);
        let v2 = VirtualReg::gpr(2);
        let f = FunctionBuilder::new("add_fn")
            .param(Register::Virtual(v0), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(v2),
                lhs: Operand::Register(Register::Virtual(v0)),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(Register::Virtual(v2))),
            })
            .build();
        m.add_function(f);
        let mut output = Vec::new();
        generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("add 3, 3, 4"), "Expected add instruction: {s}");
    }

    #[test]
    fn test_generate_jmp_br() {
        let mut m = MirModule::new("test_ppc64_cf");
        let cond = VirtualReg::gpr(0);
        let f = FunctionBuilder::new("cf_fn")
            .param(Register::Virtual(cond), MirType::Scalar(ScalarType::I1))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Br {
                cond: Register::Virtual(cond),
                true_target: "then".to_string(),
                false_target: "else".to_string(),
            })
            .block("then")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(1))),
            })
            .block("else")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();
        m.add_function(f);
        let mut output = Vec::new();
        generate_mir_ppc64(&m, &mut output, TargetOperatingSystem::Linux).unwrap();
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("bne"), "Expected bne in output: {s}");
        assert!(
            s.contains(".L_cf_fn.then:"),
            "Expected function-qualified .L_cf_fn.then in output: {s}"
        );
        assert!(
            !s.contains("\n.L_then:"),
            "block labels must be qualified by function: {s}"
        );
    }

    #[test]
    fn float_copy_emits_bit_move_not_add() {
        let ty = MirType::Scalar(ScalarType::F64);
        let src = Register::Virtual(VirtualReg::fpr(0));
        let dst = Register::Virtual(VirtualReg::fpr(1));
        let function = FunctionBuilder::new("copy_float")
            .param(src.clone(), ty)
            .returns(ty)
            .block("entry")
            .instr(Instruction::Copy {
                ty,
                dst: dst.clone(),
                src: Operand::Register(src),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();
        let mut module = MirModule::new("copy_test");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_ppc64(&module, &mut output, TargetOperatingSystem::Linux).unwrap();
        let assembly = String::from_utf8(output).unwrap();

        assert!(assembly.contains("ld 3"), "expected bit load: {assembly}");
        assert!(assembly.contains("std 3"), "expected bit store: {assembly}");
        assert!(
            !assembly.contains("fadd"),
            "unexpected float add: {assembly}"
        );
    }
}
