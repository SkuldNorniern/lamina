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
use crate::mir_codegen::common::external_names;
use crate::mir_codegen::common::integer_conversion;
use crate::mir_codegen::common::{
    CallSignatures, emit_debug_file, emit_gnu_stack_note, print_format_label,
};
use crate::mir_codegen::common::{
    ClobberIndex, CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, emit_symbol_visibility, self_tail_arguments, self_tail_schedule,
};

use std::{
    collections::{BTreeSet, HashMap},
    io::{self, Write},
    sync::Arc,
};
use util::{
    load_operand_to_register, load_register_to_r3, load_register_to_register, store_r3_to_register,
};

use crate::{
    error::LaminaError,
    mir::{
        Aggregate, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Instruction as MirInst,
        IntBinOp, IntCmpOp, MirType, Module as MirModule, Operand, Register, ScalarType, Signature,
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

use lamina_codegen::abi::{Abi, RuntimeSymbol};
use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator, LiveInterval,
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
    signatures: &CallSignatures,
    aggregates: &AggregateSignatures,
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
        // Every allocatable register is volatile in ELFv2, so a bl destroys whatever the
        // plan is holding across it.
        let mut calls = ClobberIndex::default();
        for (index, inst) in func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .enumerate()
        {
            if matches!(
                inst,
                MirInst::Call { .. } | MirInst::CallIndirect { .. } | MirInst::TailCall { .. }
            ) {
                calls.mark((), index);
            }
        }
        let allowed = |interval: &LiveInterval, _reg| calls.is_clear((), interval);
        let mut plan = match settings.regalloc {
            RegallocStrategy::LinearScanGlobal => {
                LinearScanAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::GraphColorGlobal => {
                GraphColorAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::Incremental => {
                return Err(CodegenError::InvalidCodegenOptions(
                    "internal: incremental in global branch".to_string(),
                ));
            }
        };
        plan = coalesce_dying_inputs(func, &intervals, &plan, allowed);
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

    let hidden_result_slot = func
        .sig
        .ret_aggregate
        .filter(|aggregate| aggregate.size as u64 > MAX_REGISTER_AGGREGATE)
        .map(|_| Ppc64Frame::aligned_frame_size(local_bytes) as i32 + parameter_save_offset(0));
    store_incoming_parameters(
        &mut output,
        func,
        local_bytes,
        hidden_result_slot,
        &reg_alloc,
        &stack_slots,
    )
    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
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
        aggregates,
        aggregate_flow: func.sig.ret_aggregate.is_some()
            || func.sig.params.iter().any(|p| p.aggregate.is_some()),
        ret_aggregate: func.sig.ret_aggregate,
        hidden_result_slot,
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
    emit_ppc64_module(module, writer, target_os, codegen_units, settings)
}

fn emit_ppc64_module<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    let abi = Ppc64Abi::new(target_os);

    writeln!(writer, ".abiversion 2")?;
    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_print_format())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    emit_debug_file(writer, settings)?;
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;

    for func_name in external_names(module) {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(CallSignatures::from_module(module));
    let aggregates = Arc::new(AggregateSignatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_ppc64(
                name,
                func,
                os,
                settings_arc.as_ref(),
                signatures.as_ref(),
                aggregates.as_ref(),
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
/// Where an argument sits under ELFv2. Every argument takes doublewords of the parameter
/// save area in order, and one passed in registers takes the general register that
/// matches its first doubleword. An aggregate that runs past r10 is split: the doublewords
/// that fit go in registers and the rest in the save area.
#[derive(Clone, Copy)]
enum PpcArgPlace {
    Gpr(usize),
    Fpr { bank: usize, gpr: usize },
    Stack(usize),
    Words { first: usize, words: usize },
    Split { first: usize, words: usize },
    StackWords { slot: usize, words: usize },
}

/// f1 to f13 carry floats.
const FLOAT_ARGUMENT_REGISTERS: usize = 13;

/// Flushes every stream after a print.
///
/// The print intrinsics go through buffered `printf` while the byte intrinsics write
/// straight out, so without this the two leave in a different order here than on the
/// other backends.
fn emit_flush<W: Write>(writer: &mut W, fflush: &str) -> Result<(), LaminaError> {
    writeln!(writer, "    li 3, 0")?;
    writeln!(writer, "    bl {fflush}")?;
    writeln!(writer, "    nop")?;
    Ok(())
}

fn classify_ppc64_arguments(kinds: &[ArgKind], shift: usize) -> Vec<PpcArgPlace> {
    let banks = Ppc64Abi::ARG_REGISTERS.len();
    let mut floats = 0;
    let mut dwords = shift;
    kinds
        .iter()
        .map(|kind| {
            let place = match kind.aggregate {
                Some(aggregate) => {
                    let words = aggregate.words();
                    if dwords + words <= banks {
                        PpcArgPlace::Words {
                            first: dwords,
                            words,
                        }
                    } else if dwords < banks {
                        PpcArgPlace::Split {
                            first: dwords,
                            words,
                        }
                    } else {
                        PpcArgPlace::StackWords {
                            slot: dwords,
                            words,
                        }
                    }
                }
                None if kind.ty.is_float() && floats < FLOAT_ARGUMENT_REGISTERS => {
                    floats += 1;
                    PpcArgPlace::Fpr {
                        bank: floats,
                        gpr: dwords,
                    }
                }
                None if dwords < banks => PpcArgPlace::Gpr(dwords),
                None => PpcArgPlace::Stack(dwords),
            };
            dwords += kind.aggregate.map_or(1, |aggregate| aggregate.words());
            place
        })
        .collect()
}

/// A parameter or argument: its scalar type, and the aggregate it stands for when it is one.
#[derive(Clone, Copy)]
struct ArgKind {
    ty: MirType,
    aggregate: Option<Aggregate>,
}

impl ArgKind {
    fn scalar(ty: MirType) -> Self {
        Self {
            ty,
            aggregate: None,
        }
    }
}

const MAX_REGISTER_AGGREGATE: u64 = 16;

struct AggregateSignatures {
    params: HashMap<String, Vec<Option<Aggregate>>>,
    returns: HashMap<String, Aggregate>,
}

impl AggregateSignatures {
    fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| {
                    (
                        name.clone(),
                        func.sig.params.iter().map(|p| p.aggregate).collect(),
                    )
                })
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
                .collect(),
        }
    }

    fn arguments(&self, callee: &str, types: &[MirType], fixed: Option<usize>) -> Vec<ArgKind> {
        let declared = self.params.get(callee);
        types
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                let aggregate = match declared {
                    Some(aggregates) if fixed.is_none() && aggregates.len() == types.len() => {
                        aggregates[index]
                    }
                    Some(aggregates) if index < fixed.unwrap_or(0) => {
                        aggregates.get(index).copied().flatten()
                    }
                    _ => None,
                };
                ArgKind { ty, aggregate }
            })
            .collect()
    }

    fn result(&self, callee: &str) -> Option<Aggregate> {
        self.returns.get(callee).copied()
    }

    fn takes_aggregate(&self, callee: &str) -> bool {
        self.params
            .get(callee)
            .is_some_and(|aggregates| aggregates.iter().any(Option::is_some))
            || self.returns.contains_key(callee)
    }
}

/// Copies doublewords `from_word` onward of an aggregate argument into the parameter save
/// area, starting at save slot `slot`.
fn copy_to_save_area<W: Write>(
    writer: &mut W,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    arg: &Operand,
    slot: usize,
    from_word: usize,
    words: usize,
) -> Result<(), LaminaError> {
    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "11")?;
    for word in from_word..words {
        writeln!(writer, "    ld 12, {}(11)", word * 8)?;
        r1_mem(writer, "std", "12", parameter_save_offset(slot + word))?;
    }
    Ok(())
}

fn copy_from_save_area<W: Write>(
    writer: &mut W,
    frame: i32,
    slot: usize,
    dest: i32,
) -> io::Result<()> {
    r1_mem(writer, "ld", "0", frame + parameter_save_offset(slot))?;
    r1_mem(writer, "std", "0", dest)
}

/// Stores each incoming parameter into its frame slot, and keeps the hidden result pointer
/// in the save slot it arrived with so the return can hand it back in r3.
fn store_incoming_parameters<W: Write>(
    writer: &mut W,
    func: &Function,
    local_bytes: usize,
    hidden_result_slot: Option<i32>,
    reg_alloc: &Ppc64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
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
    let places = classify_ppc64_arguments(&kinds, usize::from(hidden_result_slot.is_some()));
    let frame = Ppc64Frame::aligned_frame_size(local_bytes) as i32;
    let banks = Ppc64Abi::ARG_REGISTERS.len();
    if let Some(slot) = hidden_result_slot {
        r1_mem(writer, "std", "3", slot)?;
    }
    for (param, place) in func.sig.params.iter().zip(&places) {
        let Register::Virtual(vreg) = &param.reg else {
            continue;
        };
        let Some(&offset) = stack_slots.get(vreg) else {
            continue;
        };
        match *place {
            PpcArgPlace::Fpr { bank, .. } => emit_store_fp_result(
                &bank.to_string(),
                vreg,
                writer,
                reg_alloc,
                stack_slots,
                param.ty.size_bytes() == 4,
            )?,
            PpcArgPlace::Gpr(bank) => {
                let src = Ppc64Abi::ARG_REGISTERS[bank];
                extend_param(writer, src, &param.ty)?;
                r1_mem(writer, "std", src, offset)?;
            }
            PpcArgPlace::Stack(slot) => {
                r1_mem(writer, "ld", "0", frame + parameter_save_offset(slot))?;
                extend_param(writer, "0", &param.ty)?;
                r1_mem(writer, "std", "0", offset)?;
            }
            PpcArgPlace::Words { first, words } => {
                for word in 0..words {
                    r1_mem(
                        writer,
                        "std",
                        Ppc64Abi::ARG_REGISTERS[first + word],
                        offset + 8 * word as i32,
                    )?;
                }
            }
            PpcArgPlace::Split { first, words } => {
                for word in 0..words {
                    let at = offset + 8 * word as i32;
                    if first + word < banks {
                        r1_mem(writer, "std", Ppc64Abi::ARG_REGISTERS[first + word], at)?;
                    } else {
                        copy_from_save_area(writer, frame, first + word, at)?;
                    }
                }
            }
            PpcArgPlace::StackWords { slot, words } => {
                for word in 0..words {
                    copy_from_save_area(writer, frame, slot + word, offset + 8 * word as i32)?;
                }
            }
        }
    }
    Ok(())
}

/// Everything the instruction emitter needs beyond the instruction itself.
struct Ppc64Emit<'a> {
    target_os: TargetOperatingSystem,
    block_prefix: &'a str,
    settings: &'a MirCodegenSettings,
    local_bytes: usize,
    signatures: &'a CallSignatures,
    /// Set when this function returns a float, which leaves in f1.
    ret_float: Option<MirType>,
    aggregates: &'a AggregateSignatures,
    aggregate_flow: bool,
    ret_aggregate: Option<Aggregate>,
    /// Save slot holding the caller's hidden result pointer, when this function returns
    /// an aggregate through one.
    hidden_result_slot: Option<i32>,
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
        if let Some(pairs) = self_tail_arguments(func, inst).filter(|_| !emit.aggregate_flow) {
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
        aggregates,
        aggregate_flow,
        ret_aggregate,
        hidden_result_slot,
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
            if matches!(op, FloatCmpOp::Le | FloatCmpOp::Ge) {
                writeln!(writer, "    bun {prefix}fcmp{seq}.false")?;
            }
            writeln!(writer, "    {branch} {prefix}fcmp{seq}.true")?;
            writeln!(writer, "{prefix}fcmp{seq}.false:")?;
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
            let result = aggregates.result(name);
            let result_slot = match ret {
                Some(Register::Virtual(vreg)) => stack_slots.get(vreg).copied(),
                _ => None,
            };

            if (name == "lamina.write" || name == "lamina.read") && args.len() == 2 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "4")?;
                load_operand_to_register(&args[1], writer, reg_alloc, stack_slots, "5")?;
                writeln!(writer, "    li 3, {}", i32::from(name == "lamina.write"))?;
                let libc = if name == "lamina.write" {
                    "write"
                } else {
                    "read"
                };
                let callee = abi
                    .call_stub(libc)
                    .unwrap_or_else(|| abi.mangle_function_name(libc));
                writeln!(writer, "    bl {callee}")?;
                writeln!(writer, "    nop")?;
            } else if name == "lamina.writebyte" && args.len() == 1 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "3")?;
                writeln!(writer, "    addi 1, 1, -16")?;
                writeln!(writer, "    stb 3, 0(1)")?;
                writeln!(writer, "    li 3, 1")?;
                writeln!(writer, "    mr 4, 1")?;
                writeln!(writer, "    li 5, 1")?;
                let write = abi
                    .call_stub("write")
                    .unwrap_or_else(|| abi.mangle_function_name("write"));
                writeln!(writer, "    bl {write}")?;
                writeln!(writer, "    nop")?;
                writeln!(writer, "    addi 1, 1, 16")?;
            } else if name == "lamina.readbyte" && args.is_empty() {
                writeln!(writer, "    addi 1, 1, -16")?;
                writeln!(writer, "    li 3, 0")?;
                writeln!(writer, "    mr 4, 1")?;
                writeln!(writer, "    li 5, 1")?;
                let read = abi
                    .call_stub("read")
                    .unwrap_or_else(|| abi.mangle_function_name("read"));
                writeln!(writer, "    bl {read}")?;
                writeln!(writer, "    nop")?;
                writeln!(writer, "    cmpdi 0, 3, 1")?;
                writeln!(writer, "    lbz 4, 0(1)")?;
                writeln!(writer, "    addi 1, 1, 16")?;
                writeln!(writer, "    isel 3, 4, 3, 2")?;
            } else if name == "lamina.printf32" || name == "lamina.printf64" {
                let runtime = abi.builtin_library();
                let (printf, fflush) = (
                    runtime.c_name(RuntimeSymbol::Printf),
                    runtime.c_name(RuntimeSymbol::Fflush),
                );
                if let Some(arg) = args.first() {
                    writeln!(writer, "    addis 3, 2, .L_mir_fmt_float@toc@ha")?;
                    writeln!(writer, "    addi 3, 3, .L_mir_fmt_float@toc@l")?;
                    emit_load_fp_operand(
                        arg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        "1",
                        name == "lamina.printf32",
                    )?;
                    writeln!(writer, "    stfd 1, -8(1)")?;
                    writeln!(writer, "    ld 4, -8(1)")?;
                    writeln!(writer, "    bl {printf}")?;
                    writeln!(writer, "    nop")?;
                    emit_flush(writer, fflush)?;
                }
            } else if name == "lamina.print" || name == "lamina.printu" {
                let runtime = abi.builtin_library();
                let (printf, fflush) = (
                    runtime.c_name(RuntimeSymbol::Printf),
                    runtime.c_name(RuntimeSymbol::Fflush),
                );
                if let Some(arg) = args.first() {
                    let fmt = print_format_label(name);
                    writeln!(writer, "    addis 3, 2, {fmt}@toc@ha")?;
                    writeln!(writer, "    addi 3, 3, {fmt}@toc@l")?;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "4")?;
                    writeln!(writer, "    bl {printf}")?;
                    writeln!(writer, "    nop")?;
                    emit_flush(writer, fflush)?;
                }
            } else {
                let types = signatures.arguments(name, args);
                let kinds = aggregates.arguments(name, &types, signatures.fixed_arguments(name));
                let fixed = signatures.fixed_arguments(name).unwrap_or(
                    if signatures.may_be_variadic(name, args.len()) {
                        0
                    } else {
                        args.len()
                    },
                );
                let hidden = match result {
                    Some(aggregate) if aggregate.size as u64 > MAX_REGISTER_AGGREGATE => {
                        Some(result_slot.ok_or_else(|| {
                            LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                                "PowerPC64 aggregate result has no frame slot".to_string(),
                            ))
                        })?)
                    }
                    _ => None,
                };
                let shift = usize::from(hidden.is_some());
                let places = classify_ppc64_arguments(&kinds, shift);
                let dwords = shift
                    + kinds
                        .iter()
                        .map(|kind| kind.aggregate.map_or(1, |aggregate| aggregate.words()))
                        .sum::<usize>();
                for (index, ((arg, kind), place)) in
                    args.iter().zip(&kinds).zip(&places).enumerate()
                {
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
                        PpcArgPlace::Fpr { bank, gpr } => {
                            emit_load_fp_operand(
                                arg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                &bank.to_string(),
                                kind.ty.size_bytes() == 4,
                            )?;
                            // ELFv2 hands a variadic callee its floats in both banks, and
                            // the callee reads whichever one its va_arg takes.
                            if index >= fixed && gpr < arg_regs.len() {
                                writeln!(writer, "    stfd {bank}, -8(1)")?;
                                writeln!(writer, "    ld {}, -8(1)", arg_regs[gpr])?;
                            }
                        }
                        PpcArgPlace::Words { first, words } => {
                            load_operand_to_register(arg, writer, reg_alloc, stack_slots, "11")?;
                            for word in 0..words {
                                writeln!(
                                    writer,
                                    "    ld {}, {}(11)",
                                    arg_regs[first + word],
                                    word * 8
                                )?;
                            }
                        }
                        PpcArgPlace::Split { first, .. } => {
                            load_operand_to_register(arg, writer, reg_alloc, stack_slots, "11")?;
                            for word in 0..arg_regs.len() - first {
                                writeln!(
                                    writer,
                                    "    ld {}, {}(11)",
                                    arg_regs[first + word],
                                    word * 8
                                )?;
                            }
                        }
                        PpcArgPlace::Stack(_) | PpcArgPlace::StackWords { .. } => {}
                    }
                }

                let outgoing = outgoing_frame_size(dwords)?;
                let moved = outgoing.map(|frame| biased_slots(stack_slots, frame as i32));
                let slots = moved.as_ref().unwrap_or(stack_slots);
                if let Some(frame) = outgoing {
                    // The parameter save area starts above the linkage area and holds a
                    // slot for every parameter, register ones included. `stdu` moves r1
                    // and writes the back chain in one step, which the slot offsets then
                    // have to account for.
                    writeln!(writer, "    stdu 1, -{frame}(1)")?;
                }
                if let Some(slot) = hidden {
                    writeln!(
                        writer,
                        "    addi 3, 1, {}",
                        slot + outgoing.unwrap_or(0) as i32
                    )?;
                }
                for (arg, place) in args.iter().zip(&places) {
                    match *place {
                        PpcArgPlace::Stack(slot) => {
                            load_operand_to_register(arg, writer, reg_alloc, slots, "11")?;
                            r1_mem(writer, "std", "11", parameter_save_offset(slot))?;
                        }
                        PpcArgPlace::StackWords { slot, words } => {
                            copy_to_save_area(writer, reg_alloc, slots, arg, slot, 0, words)?;
                        }
                        PpcArgPlace::Split { first, words } => {
                            let from_word = arg_regs.len() - first;
                            copy_to_save_area(
                                writer, reg_alloc, slots, arg, first, from_word, words,
                            )?;
                        }
                        _ => {}
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
                match result {
                    Some(aggregate) if aggregate.size as u64 > MAX_REGISTER_AGGREGATE => {}
                    Some(aggregate) => {
                        let slot = result_slot.ok_or_else(|| {
                            LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                                "PowerPC64 aggregate result has no frame slot".to_string(),
                            ))
                        })?;
                        r1_mem(writer, "std", "3", slot)?;
                        if aggregate.words() > 1 {
                            r1_mem(writer, "std", "4", slot + 8)?;
                        }
                    }
                    None => match signatures.float_result(name) {
                        Some(ty) => emit_store_fp_result(
                            "1",
                            vreg,
                            writer,
                            reg_alloc,
                            stack_slots,
                            ty.size_bytes() == 4,
                        )?,
                        None => store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?,
                    },
                }
            }
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed,
            ret,
            ret_ty,
        } => {
            let arg_regs = Ppc64Abi::ARG_REGISTERS;
            let kinds: Vec<ArgKind> = arg_types.iter().map(|&ty| ArgKind::scalar(ty)).collect();
            let places = classify_ppc64_arguments(&kinds, 0);
            let fixed = variadic_fixed.unwrap_or(args.len());
            for (index, ((arg, ty), place)) in args.iter().zip(arg_types).zip(&places).enumerate() {
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
                    PpcArgPlace::Fpr { bank, gpr } => {
                        emit_load_fp_operand(
                            arg,
                            writer,
                            reg_alloc,
                            stack_slots,
                            &bank.to_string(),
                            ty.size_bytes() == 4,
                        )?;
                        if index >= fixed && gpr < arg_regs.len() {
                            writeln!(writer, "    stfd {bank}, -8(1)")?;
                            writeln!(writer, "    ld {}, -8(1)", arg_regs[gpr])?;
                        }
                    }
                    _ => {}
                }
            }
            // ELFv2 wants the entry point in r12, and the global entry sequence
            // rebuilds r2 from it.
            load_operand_to_register(target, writer, reg_alloc, stack_slots, "12")?;
            let outgoing = outgoing_frame_size(args.len())?;
            if let Some(frame) = outgoing {
                writeln!(writer, "    stdu 1, -{frame}(1)")?;
                let moved = biased_slots(stack_slots, frame as i32);
                for (arg, place) in args.iter().zip(&places) {
                    if let PpcArgPlace::Stack(slot) = *place {
                        load_operand_to_register(arg, writer, reg_alloc, &moved, "11")?;
                        r1_mem(writer, "std", "11", parameter_save_offset(slot))?;
                    }
                }
            }
            writeln!(writer, "    std 2, 24(1)")?;
            writeln!(writer, "    mtctr 12")?;
            writeln!(writer, "    bctrl")?;
            writeln!(writer, "    ld 2, 24(1)")?;
            if let Some(frame) = outgoing {
                writeln!(writer, "    addi 1, 1, {frame}")?;
            }
            if let Some(Register::Virtual(vreg)) = ret {
                match ret_ty {
                    Some(ty) if ty.is_float() => emit_store_fp_result(
                        "1",
                        vreg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        ty.size_bytes() == 4,
                    )?,
                    _ => store_r3_to_register(vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::TailCall { name, args } => {
            if aggregate_flow || aggregates.takes_aggregate(name) {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "PowerPC64: TailCall with an aggregate is not supported".to_string(),
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
                match (ret_aggregate, hidden_result_slot) {
                    (Some(aggregate), Some(slot)) => {
                        load_operand_to_register(val, writer, reg_alloc, stack_slots, "11")?;
                        r1_mem(writer, "ld", "4", slot)?;
                        for word in 0..aggregate.words() {
                            writeln!(writer, "    ld 12, {}(11)", word * 8)?;
                            writeln!(writer, "    std 12, {}(4)", word * 8)?;
                        }
                        writeln!(writer, "    mr 3, 4")?;
                    }
                    (Some(aggregate), None) => {
                        load_operand_to_register(val, writer, reg_alloc, stack_slots, "11")?;
                        for word in 0..aggregate.words() {
                            writeln!(
                                writer,
                                "    ld {}, {}(11)",
                                Ppc64Abi::ARG_REGISTERS[word],
                                word * 8
                            )?;
                        }
                    }
                    (None, _) => match ret_float {
                        Some(ty) => emit_load_fp_operand(
                            val,
                            writer,
                            reg_alloc,
                            stack_slots,
                            "1",
                            ty.size_bytes() == 4,
                        )?,
                        None => load_operand_to_register(val, writer, reg_alloc, stack_slots, "3")?,
                    },
                }
            }
            Ppc64Frame::generate_epilogue(writer, local_bytes).map_err(|e| {
                LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(e.to_string()))
            })?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Unreachable => {
            writeln!(writer, "    trap")?;
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!(
                    "atomic instruction `{inst}` is not yet supported by the PowerPC64 backend"
                ),
            )));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                "PowerPC64 backend does not support variadic arguments yet".to_string(),
            )));
        }
        MirInst::IntUnary { .. } => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                "PowerPC64 backend does not support integer bit intrinsics yet".to_string(),
            )));
        }
        MirInst::IntRotate { .. } => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                "PowerPC64 backend does not support integer rotates yet".to_string(),
            )));
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
fn outgoing_frame_size(dwords: usize) -> Result<Option<usize>, LaminaError> {
    if dwords <= Ppc64Abi::ARG_REGISTERS.len() {
        return Ok(None);
    }
    let frame = (LINKAGE_SIZE + dwords * 8 + 15) & !15;
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
mod tests;
