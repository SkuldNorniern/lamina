use crate::mir::register::PhysicalReg;
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::external_names;
use crate::mir_codegen::common::{
    CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, emit_symbol_visibility, self_tail_arguments, self_tail_schedule,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note};
use crate::mir_codegen::common::{
    reject_thread_local_globals, reject_underaligned_accesses, unsupported_inline_asm,
    unsupported_overflow_flag, unsupported_thread_local,
};
use lamina_mir::SourceLoc;
mod atomic;
mod bit;
mod simd;
use crate::mir_codegen::simd::reserve_vector_slots;
mod util;
mod variadic;

use crate::mir_codegen::common::{
    align_up, integer_conversion_with_width, io_codegen_error, promoted_argument_type,
};

use util::{
    clamp_above, clamp_below, emit_int_cmp_op, extend_to_xlen, load_fp_operand_to_register,
    load_operand_to_register, load_register_to_register, sign_extend_to_xlen,
    store_fp_register_to_register, store_register_to_register,
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
        common::{assign_stack_slots, compile_functions_parallel, parallel_codegen_error},
        validate_module_call_parameters,
    },
};

use lamina_codegen::{
    GraphColorAllocator, LinearScanAllocator, LiveInterval, LocalRegisterAllocator,
    riscv::{RiscVAbi, RiscVFrame, RiscVRegAlloc, RiscVTarget, Xlen},
};
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};
use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    sync::Arc,
};

use crate::mir_codegen::common::CodegenBase;

/// Trait-backed MIR ⇒ RISC-V code generator.
pub struct RiscVCodegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> RiscVCodegen<'a> {
    pub fn new(target_os: TargetOperatingSystem, pointer_width: PointerWidth) -> Self {
        Self {
            base: CodegenBase::new(target_os, pointer_width),
        }
    }

    /// Attach the MIR module that should be emitted in the next codegen pass.
    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    /// Drain the internal assembly buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    /// Emit assembly for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), LaminaError> {
        generate_mir_riscv_with_units_and_settings(
            module,
            writer,
            RiscVTarget::general(Xlen::Rv64),
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for RiscVCodegen<'a> {
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
                generate_mir_riscv_with_units_and_settings(
                    module,
                    writer,
                    RiscVTarget::general(Xlen::Rv64),
                    target_os,
                    codegen_units,
                    &MirCodegenSettings::default(),
                )
            },
            "RISC-V",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not supported".to_string(),
        ))
    }
}

fn compile_single_function_riscv(
    func_name: &str,
    func: &Function,
    target: RiscVTarget,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &AggregateSignatures,
) -> Result<Vec<u8>, CodegenError> {
    let reserved = reserve_vector_slots(func);
    let func = reserved.as_ref().unwrap_or(func);
    let mut output = Vec::new();
    let abi = RiscVAbi::new(target_os);

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    let slot_target = RiscVTarget::general(Xlen::Rv64);
    let wide_float_regs: HashSet<_> = func
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter(|i| instruction_type(i).is_some_and(|ty| *ty == MirType::Scalar(ScalarType::F64)))
        .filter_map(|i| {
            i.def_reg().and_then(|r| match r {
                Register::Virtual(v) => Some(*v),
                _ => None,
            })
        })
        .collect();
    let mut stack_slots: HashMap<VirtualReg, i32> = HashMap::new();
    let mut reg_alloc = RiscVRegAlloc::with_target(target_os, target);
    let mut live_across_calls = HashMap::new();

    if settings.regalloc != RegallocStrategy::Incremental {
        (stack_slots, _, _) =
            assign_stack_slots(func, |i| RiscVFrame::calculate_stack_offset(i, slot_target));
        let pool = RiscVRegAlloc::gpr_pool_for_global_allocation();
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| {
                i.vreg.class == RegisterClass::Gpr
                    && !func.stack_reservations.contains_key(&i.vreg)
                    && !(target.xlen == Xlen::Rv32 && wide_float_regs.contains(&i.vreg))
            })
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
        live_across_calls = live_vregs_across_calls(func, &intervals);
        reg_alloc = RiscVRegAlloc::from_global_plan(target_os, target, &plan, &stack_slots);
    } else if !func.stack_reservations.is_empty() {
        (stack_slots, _, _) =
            assign_stack_slots(func, |i| RiscVFrame::calculate_stack_offset(i, slot_target));
    } else {
        let mut next_slot = 0usize;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                    && !stack_slots.contains_key(vreg)
                {
                    stack_slots.insert(
                        *vreg,
                        RiscVFrame::calculate_stack_offset(next_slot, slot_target),
                    );
                    next_slot += 1;
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg)
                    {
                        stack_slots.insert(
                            *vreg,
                            RiscVFrame::calculate_stack_offset(next_slot, slot_target),
                        );
                        next_slot += 1;
                    }
                }
                // use_regs omits the slot a StackAddr names.
                if let MirInst::StackAddr { slot, .. } = &inst.instr
                    && !stack_slots.contains_key(slot)
                {
                    stack_slots.insert(
                        *slot,
                        RiscVFrame::calculate_stack_offset(next_slot, slot_target),
                    );
                    next_slot += 1;
                }
            }
        }
    }

    let word_bytes = target.word_bytes();
    let sret_offset = func
        .sig
        .ret_aggregate
        .filter(|aggregate| {
            aggregate.size as u64 > Aggregate::max_in_registers(target_pointer(target))
        })
        .map(|_| {
            stack_slots
                .values()
                .copied()
                .min()
                .unwrap_or(-2 * word_bytes)
                - word_bytes
        });
    let stack_size = stack_slots
        .values()
        .copied()
        .chain(sret_offset)
        .map(|offset| (-offset - 16) as usize)
        .max()
        .unwrap_or(0);
    let callee_saved_registers = reg_alloc.used_callee_saved_registers();
    let va_area = func.sig.variadic.then(|| {
        let kinds: Vec<ArgKind> = func
            .sig
            .params
            .iter()
            .map(|p| ArgKind {
                ty: p.ty,
                aggregate: p.aggregate,
            })
            .collect();
        let first_gpr = usize::from(sret_offset.is_some());
        let places = classify_riscv_arguments(
            &kinds,
            first_gpr,
            target_pointer(target),
            target_arch(target),
        );
        variadic::VaArea::new(&places, first_gpr, target)
    });
    // The save area sits between `fp` and the caller's `sp`, so `fp` starts that far
    // below the incoming stack arguments and every one of them shifts with it.
    let incoming_stack_base = va_area.map_or(0, |area| area.reserve());
    if incoming_stack_base > 0 {
        RiscVFrame::adjust_sp(&mut output, -i64::from(incoming_stack_base))
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    RiscVFrame::generate_prologue(&mut output, stack_size, &callee_saved_registers, target)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    if let Some(area) = &va_area {
        variadic::spill(&mut output, area, target).map_err(io_codegen_error)?;
    }

    if let Some(offset) = sret_offset {
        RiscVFrame::fp_store(&mut output, target.store_word(), "a0", offset, "t0")
            .map_err(io_codegen_error)?;
    }
    let first_gpr = usize::from(sret_offset.is_some());
    for through_stack_pointers in [false, true] {
        store_incoming_arguments(
            &mut output,
            func,
            first_gpr,
            &reg_alloc,
            &stack_slots,
            target,
            through_stack_pointers,
        )
        .map_err(io_codegen_error)?;
    }

    let mut ctx = EmitCtx {
        func_name,
        func,
        target,
        target_os,
        settings,
        loc: None,
        label_seq: 0,
        live_across_calls,
        stack_size,
        callee_saved_registers,
        va_area,
        signatures,
        ret_float: func.sig.ret_ty.filter(MirType::is_float),
        ret_aggregate: func.sig.ret_aggregate,
        sret_offset,
    };
    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    j .L_{func_name}.{}", func.entry)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    let mut starts = Vec::new();
    let mut start = 0;
    for block in &func.blocks {
        starts.push(start);
        start += block.instructions.len();
    }
    for (slot, &index) in plan.order.iter().enumerate() {
        let next = plan
            .order
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        emit_planned_block_riscv(
            func,
            index,
            starts[index],
            next,
            &plan,
            &mut output,
            &mut reg_alloc,
            &stack_slots,
            &mut ctx,
        )
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    Ok(output)
}

fn live_vregs_across_calls(
    function: &Function,
    intervals: &[LiveInterval],
) -> HashMap<usize, Vec<VirtualReg>> {
    let mut result = HashMap::new();
    let mut instruction_index = 0;
    for block in &function.blocks {
        for instruction in &block.instructions {
            if matches!(
                &instruction.instr,
                MirInst::Call { .. } | MirInst::CallIndirect { .. }
            ) {
                let mut live: Vec<_> = intervals
                    .iter()
                    .filter(|interval| {
                        interval.start < instruction_index && interval.end > instruction_index
                    })
                    .map(|interval| interval.vreg)
                    .collect();
                live.sort_unstable();
                result.insert(instruction_index, live);
            }
            instruction_index += 1;
        }
    }
    result
}

/// Rejects a value that still needs two rv32 registers at the point assembly is
/// emitted. Reaching here means the 64-bit legalization did not split it, and a
/// truncating answer would be worse than no answer, so the compile stops and names
/// what was left behind.
fn reject_unsplit_wide_values(module: &MirModule, target: RiscVTarget) -> Result<(), LaminaError> {
    if target.xlen == Xlen::Rv64 {
        return Ok(());
    }
    let refuse = |what: String| {
        Err(LaminaError::ValidationError(format!(
            "{what} needs two rv32 registers; the 64-bit legalization pass did not split it, \
             and {} cannot hold it in one",
            target.isa_name()
        )))
    };
    let i64_type = |ty: &MirType| matches!(ty, MirType::Scalar(ScalarType::I64));
    for (name, global) in &module.globals {
        if i64_type(&global.ty) {
            return refuse(format!("global @{name}"));
        }
    }
    for (name, function) in &module.functions {
        if function.sig.params.iter().any(|param| i64_type(&param.ty)) {
            return refuse(format!("a parameter of @{name}"));
        }
        if function.sig.ret_ty.as_ref().is_some_and(i64_type) {
            return refuse(format!("the result of @{name}"));
        }
        for block in &function.blocks {
            for instruction in &block.instructions {
                let instruction = &instruction.instr;
                if let MirInst::Select { ty, .. } = instruction
                    && ty.size_bytes(PointerWidth::Bits32) > 4
                {
                    return Err(LaminaError::ValidationError(format!(
                        "select of a {ty} in @{name} picks between two integer registers, \
                         which hold 32 bits on {}; it is refused rather than truncated",
                        target.isa_name()
                    )));
                }
                let named = instruction_type(instruction).is_some_and(i64_type)
                    || matches!(instruction, MirInst::IntConvert { from, to, .. }
                        if i64_type(from) || i64_type(to));
                if named {
                    return refuse(format!("`{instruction}` in @{name}"));
                }
            }
        }
    }
    Ok(())
}

fn instruction_type(instruction: &MirInst) -> Option<&MirType> {
    match instruction {
        MirInst::Copy { ty, .. }
        | MirInst::IntBinary { ty, .. }
        | MirInst::IntRotate { ty, .. }
        | MirInst::FloatBinary { ty, .. }
        | MirInst::FloatUnary { ty, .. }
        | MirInst::IntUnary { ty, .. }
        | MirInst::IntCmp { ty, .. }
        | MirInst::FloatCmp { ty, .. }
        | MirInst::Select { ty, .. }
        | MirInst::Load { ty, .. }
        | MirInst::Store { ty, .. }
        | MirInst::VectorOp { ty, .. } => Some(ty),
        _ => None,
    }
}

pub fn generate_mir_riscv<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_riscv_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_riscv_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_riscv_with_units_and_settings(
        module,
        writer,
        RiscVTarget::general(Xlen::Rv64),
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_riscv_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target: RiscVTarget,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, target_arch(target))?;
    emit_riscv_module(module, writer, target, target_os, codegen_units, settings)
}

/// Which `TargetArchitecture` an ISA selection is, for the facts that hang off it.
fn target_arch(target: RiscVTarget) -> TargetArchitecture {
    match target.xlen {
        Xlen::Rv32 => TargetArchitecture::Riscv32,
        Xlen::Rv64 => TargetArchitecture::Riscv64,
    }
}

fn emit_riscv_module<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target: RiscVTarget,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    reject_unsplit_wide_values(module, target)?;
    reject_thread_local_globals(module, "RISC-V")?;
    reject_underaligned_accesses(module, target_arch(target).pointer_width(), "RISC-V")?;
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(AggregateSignatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_riscv(
                name,
                func,
                target,
                os,
                settings_arc.as_ref(),
                signatures.as_ref(),
            )
        })
        .map_err(parallel_codegen_error)?;

    let abi = RiscVAbi::new(target_os);

    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    emit_debug_file(writer, settings, module)?;
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;

    for func_name in external_names(module) {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    for result in results {
        writer.write_all(&result.assembly)?;
    }
    emit_global_data(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;
    emit_gnu_stack_note(writer, target_os)?;

    Ok(())
}

/// Where an argument sits under lp64d: an integer register, a float register from a
/// bank of its own, an eight-byte stack slot, or the consecutive integer registers and
/// stack slots an aggregate of two words or fewer occupies.
#[derive(Clone, Copy)]
enum RiscVArgPlace {
    Gpr(usize),
    Fpr(usize),
    Stack(usize),
    Words { first: usize, words: usize },
    StackWords { slot: usize, words: usize },
    Split { reg: usize, slot: usize },
}

/// Puts each argument where the call expects it and reserves the outgoing stack area,
/// returning how many bytes of it to release after the call. An aggregate larger than
/// two words travels as the address of a copy made in that area, and a hidden result
/// pointer, when there is one, takes a0 ahead of the arguments.
#[allow(clippy::too_many_arguments)]
fn pass_riscv_arguments<W: Write>(
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    args: &[Operand],
    kinds: &[ArgKind],
    hidden: Option<i32>,
    target: RiscVTarget,
    pointer_width: PointerWidth,
) -> Result<usize, LaminaError> {
    let arg_regs = RiscVAbi::ARG_REGISTERS;
    let word = target.word_bytes() as usize;
    let pointer = target_pointer(target);
    let places = classify_riscv_arguments(
        kinds,
        usize::from(hidden.is_some()),
        pointer,
        target_arch(target),
    );
    let stack_words: usize = places
        .iter()
        .map(|place| match *place {
            RiscVArgPlace::Stack(_) => 1,
            RiscVArgPlace::StackWords { words, .. } => words,
            RiscVArgPlace::Split { .. } => 1,
            RiscVArgPlace::Gpr(_) | RiscVArgPlace::Fpr(_) | RiscVArgPlace::Words { .. } => 0,
        })
        .sum();
    let mut copies = vec![None; kinds.len()];
    let mut copy_bytes = stack_words * word;
    for (kind, copy) in kinds.iter().zip(copies.iter_mut()) {
        if kind.by_reference(pointer) {
            *copy = Some(copy_bytes);
            copy_bytes += kind.aggregate.map_or(0, |a| a.words_of(pointer)) * word;
        }
    }
    let stack_space = align_up(copy_bytes, target_arch(target).stack_alignment_at_call());
    if stack_space > 0 {
        RiscVFrame::adjust_sp(writer, -(stack_space as i64))?;
    }

    for ((arg, kind), copy) in args.iter().zip(kinds).zip(&copies) {
        let Some(at) = *copy else { continue };
        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "a7")?;
        for index in 0..kind.aggregate.map_or(0, |a| a.words_of(pointer)) {
            let offset = index * word;
            writeln!(writer, "    {} t0, {offset}(a7)", target.load_word())?;
            writeln!(
                writer,
                "    {} t0, {}(sp)",
                target.store_word(),
                at + offset
            )?;
        }
    }
    for (arg, place) in args.iter().zip(&places) {
        let (slot, first_word, words) = match *place {
            RiscVArgPlace::StackWords { slot, words } => (slot, 0, words),
            RiscVArgPlace::Split { slot, .. } => (slot, 1, 2),
            _ => continue,
        };
        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
        for index in first_word..words {
            let offset = index * word;
            writeln!(writer, "    {} a7, {offset}(t0)", target.load_word())?;
            writeln!(
                writer,
                "    {} a7, {}(sp)",
                target.store_word(),
                slot * word + (index - first_word) * word
            )?;
        }
    }

    if let Some(offset) = hidden {
        RiscVFrame::fp_address(writer, arg_regs[0], offset)?;
    }
    for (index, ((arg, kind), place)) in args.iter().zip(kinds).zip(&places).enumerate() {
        match *place {
            RiscVArgPlace::Gpr(bank) => match copies[index] {
                Some(at) => sp_address(writer, arg_regs[bank], at)?,
                None => {
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[bank])?
                }
            },
            RiscVArgPlace::Fpr(bank) => load_fp_operand_to_register(
                arg,
                writer,
                reg_alloc,
                stack_slots,
                &format!("fa{bank}"),
                kind.ty.size_bytes(pointer_width) == 4,
            )?,
            RiscVArgPlace::Words { first, words } => {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                for index in 0..words {
                    let offset = index * word;
                    writeln!(
                        writer,
                        "    {} {}, {offset}(t0)",
                        target.load_word(),
                        arg_regs[first + index]
                    )?;
                }
            }
            RiscVArgPlace::Split { reg, .. } => {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                writeln!(
                    writer,
                    "    {} {}, 0(t0)",
                    target.load_word(),
                    arg_regs[reg]
                )?;
            }
            RiscVArgPlace::Stack(_) | RiscVArgPlace::StackWords { .. } => {}
        }
    }

    for (index, (arg, place)) in args.iter().zip(&places).enumerate() {
        let RiscVArgPlace::Stack(slot) = *place else {
            continue;
        };
        match copies[index] {
            Some(at) => sp_address(writer, "t0", at)?,
            None => load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?,
        }
        writeln!(
            writer,
            "    {} t0, {}(sp)",
            target.store_word(),
            slot * word
        )?;
    }
    Ok(stack_space)
}

fn sp_address<W: Write>(writer: &mut W, dst: &str, offset: usize) -> io::Result<()> {
    if offset <= 2047 {
        writeln!(writer, "    addi {dst}, sp, {offset}")
    } else {
        RiscVFrame::load_imm(writer, dst, offset as i64)?;
        writeln!(writer, "    add {dst}, sp, {dst}")
    }
}

fn copy_from_pointer_to_frame<W: Write>(
    writer: &mut W,
    pointer: &str,
    words: usize,
    destination: i32,
    target: RiscVTarget,
) -> io::Result<()> {
    let word = target.word_bytes();
    for index in 0..words {
        let offset = index as i32 * word;
        writeln!(writer, "    {} t0, {offset}({pointer})", target.load_word())?;
        RiscVFrame::fp_store(
            writer,
            target.store_word(),
            "t0",
            destination + offset,
            "t0",
        )?;
    }
    Ok(())
}

/// Stores each incoming argument into its frame slot. Stack-passed aggregates held
/// behind a pointer load it through a7, so they run after every register argument has
/// left the argument registers.
fn store_incoming_arguments<W: Write>(
    writer: &mut W,
    func: &Function,
    first_gpr: usize,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
    through_stack_pointers: bool,
) -> io::Result<()> {
    let arg_regs = RiscVAbi::ARG_REGISTERS;
    let word = target.word_bytes();
    let kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind {
            ty: p.ty,
            aggregate: p.aggregate,
        })
        .collect();
    let pointer = target_pointer(target);
    let places = classify_riscv_arguments(&kinds, first_gpr, pointer, target_arch(target));
    // In a variadic function `fp` sits below the save area, so every incoming stack
    // argument is that much further up.
    let incoming_stack_base = if func.sig.variadic {
        variadic::VaArea::new(&places, first_gpr, target).reserve()
    } else {
        0
    };
    for ((param, kind), place) in func.sig.params.iter().zip(&kinds).zip(&places) {
        let via_stack_pointer =
            kind.by_reference(pointer) && matches!(place, RiscVArgPlace::Stack(_));
        if via_stack_pointer != through_stack_pointers {
            continue;
        }
        let Register::Virtual(vreg) = &param.reg else {
            continue;
        };
        let Some(&slot) = stack_slots.get(vreg) else {
            continue;
        };
        let words = kind.aggregate.map_or(0, |a| a.words_of(pointer));
        match *place {
            RiscVArgPlace::Fpr(bank) => store_fp_register_to_register(
                &format!("fa{bank}"),
                vreg,
                writer,
                reg_alloc,
                stack_slots,
                param.ty.size_bytes(func.pointer_width) == 4,
            )?,
            RiscVArgPlace::Gpr(bank) if kind.by_reference(pointer) => {
                copy_from_pointer_to_frame(writer, arg_regs[bank], words, slot, target)?
            }
            RiscVArgPlace::Gpr(bank) => match reg_alloc.get_mapping(vreg) {
                Some(physical) => writeln!(writer, "    mv {physical}, {}", arg_regs[bank])?,
                None => {
                    RiscVFrame::fp_store(writer, target.store_word(), arg_regs[bank], slot, "t0")?
                }
            },
            RiscVArgPlace::Stack(caller) if kind.by_reference(pointer) => {
                let from = incoming_stack_base + caller as i32 * word;
                RiscVFrame::fp_load(writer, target.load_word(), "a7", from, "a7")?;
                copy_from_pointer_to_frame(writer, "a7", words, slot, target)?;
            }
            RiscVArgPlace::Stack(caller) => {
                let from = incoming_stack_base + caller as i32 * word;
                RiscVFrame::fp_load(writer, target.load_word(), "t0", from, "t0")?;
                match reg_alloc.get_mapping(vreg) {
                    Some(physical) => writeln!(writer, "    mv {physical}, t0")?,
                    None => RiscVFrame::fp_store(writer, target.store_word(), "t0", slot, "t0")?,
                }
            }
            RiscVArgPlace::Words { first, words } => {
                for index in 0..words {
                    let at = slot + index as i32 * word;
                    RiscVFrame::fp_store(
                        writer,
                        target.store_word(),
                        arg_regs[first + index],
                        at,
                        "t0",
                    )?;
                }
            }
            RiscVArgPlace::StackWords {
                slot: caller,
                words,
            } => {
                for index in 0..words {
                    let from = incoming_stack_base + (caller + index) as i32 * word;
                    let at = slot + index as i32 * word;
                    RiscVFrame::fp_load(writer, target.load_word(), "t0", from, "t0")?;
                    RiscVFrame::fp_store(writer, target.store_word(), "t0", at, "t0")?;
                }
            }
            RiscVArgPlace::Split { reg, slot: caller } => {
                RiscVFrame::fp_store(writer, target.store_word(), arg_regs[reg], slot, "t0")?;
                let from = incoming_stack_base + caller as i32 * word;
                let at = slot + word;
                RiscVFrame::fp_load(writer, target.load_word(), "t0", from, "t0")?;
                RiscVFrame::fp_store(writer, target.store_word(), "t0", at, "t0")?;
            }
        }
    }
    Ok(())
}

/// The pointer width the target's registers hold, which decides how many registers an
/// aggregate spans and how large one may be before it travels by reference.
fn target_pointer(target: RiscVTarget) -> PointerWidth {
    match target.xlen {
        Xlen::Rv32 => PointerWidth::Bits32,
        _ => PointerWidth::Bits64,
    }
}

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

    fn float(&self) -> bool {
        self.aggregate.is_none() && self.ty.is_float()
    }

    fn by_reference(&self, pointer: PointerWidth) -> bool {
        self.aggregate
            .is_some_and(|a| a.size as u64 > Aggregate::max_in_registers(pointer))
    }
}

/// An aggregate of two words or fewer takes consecutive integer registers when they all
/// remain. With one register left, its first word goes there and the second on the stack.
/// With none left it goes to the stack whole. A larger one passes the address of a copy.
fn classify_riscv_arguments(
    kinds: &[ArgKind],
    first_gpr: usize,
    pointer: PointerWidth,
    arch: TargetArchitecture,
) -> Vec<RiscVArgPlace> {
    let banks = RiscVAbi::ARG_REGISTERS.len();
    // The psABI wants the greater of XLEN and the natural alignment, up to 16. This
    // backend counts plain words, which `honors_argument_alignment` states.
    let honors_alignment = arch.honors_argument_alignment();
    let mut gprs = first_gpr;
    let mut fprs = 0;
    let mut stack = 0usize;
    let mut places = Vec::with_capacity(kinds.len());
    for kind in kinds {
        if honors_alignment && kind.aggregate.is_some_and(|a| a.align >= 16) {
            gprs = gprs.next_multiple_of(2);
            stack = stack.next_multiple_of(2);
        }
        match kind.aggregate {
            Some(aggregate) if !kind.by_reference(pointer) => {
                let words = aggregate.words_of(pointer);
                if gprs + words <= banks {
                    places.push(RiscVArgPlace::Words { first: gprs, words });
                    gprs += words;
                } else if words == 2 && gprs < banks {
                    places.push(RiscVArgPlace::Split {
                        reg: gprs,
                        slot: stack,
                    });
                    gprs += 1;
                    stack += 1;
                } else {
                    places.push(RiscVArgPlace::StackWords { slot: stack, words });
                    stack += words;
                }
            }
            _ if kind.float() && fprs < banks => {
                places.push(RiscVArgPlace::Fpr(fprs));
                fprs += 1;
            }
            _ if gprs < banks => {
                places.push(RiscVArgPlace::Gpr(gprs));
                gprs += 1;
            }
            _ => {
                places.push(RiscVArgPlace::Stack(stack));
                stack += 1;
            }
        }
    }
    places
}

struct AggregateSignatures {
    params: HashMap<String, Vec<ArgKind>>,
    returns: HashMap<String, Aggregate>,
    float_returns: HashMap<String, MirType>,
    variadic: HashSet<String>,
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
                        func.sig
                            .params
                            .iter()
                            .map(|p| ArgKind {
                                ty: p.ty,
                                aggregate: p.aggregate,
                            })
                            .collect(),
                    )
                })
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
                .collect(),
            float_returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| {
                    let ty = func.sig.ret_ty?;
                    (func.sig.ret_aggregate.is_none() && ty.is_float()).then(|| (name.clone(), ty))
                })
                .collect(),
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, _)| name.clone())
                .collect(),
        }
    }

    fn fixed_arguments(&self, callee: &str) -> Option<usize> {
        self.variadic
            .contains(callee)
            .then(|| self.params.get(callee).map_or(0, Vec::len))
    }

    /// Arguments of a call as the convention sees them. A matching prototype gives the
    /// kinds; past a variadic callee's fixed parameters, and for a callee with no
    /// prototype, the operand does.
    fn arguments(&self, callee: &str, args: &[Operand], func: &Function) -> Vec<ArgKind> {
        let fixed = self.fixed_arguments(callee);
        if fixed.is_none()
            && let Some(kinds) = self.params.get(callee).filter(|k| k.len() == args.len())
        {
            return kinds.clone();
        }
        let declared = self.params.get(callee);
        let fixed = fixed.unwrap_or(0);
        args.iter()
            .enumerate()
            .map(|(index, arg)| match declared {
                Some(kinds) if index < fixed && index < kinds.len() => kinds[index],
                // A legalized pair the callee's declaration does not type reaches it
                // as the aggregate it was spilled to, not as the address of one.
                _ => match func.variadic_wide_argument(arg) {
                    Some(aggregate) => ArgKind {
                        ty: MirType::Scalar(ScalarType::Ptr),
                        aggregate: Some(aggregate),
                    },
                    None => ArgKind::scalar(promoted_argument_type(arg)),
                },
            })
            .collect()
    }

    fn result(&self, callee: &str) -> Option<Aggregate> {
        self.returns.get(callee).copied()
    }

    fn float_result(&self, callee: &str) -> Option<MirType> {
        self.float_returns.get(callee).copied()
    }
}

/// Everything the instruction emitter needs beyond the instruction itself.
struct EmitCtx<'a> {
    func_name: &'a str,
    func: &'a Function,
    target: RiscVTarget,
    target_os: TargetOperatingSystem,
    settings: &'a MirCodegenSettings,
    loc: Option<SourceLoc>,
    /// Sequence for compiler-generated labels, so they do not depend on any address.
    label_seq: u32,
    live_across_calls: HashMap<usize, Vec<VirtualReg>>,
    stack_size: usize,
    callee_saved_registers: Vec<&'static str>,
    /// Set in a variadic function, where it says where the unnamed arguments sit and
    /// how much `sp` the prologue took for them.
    va_area: Option<variadic::VaArea>,
    signatures: &'a AggregateSignatures,
    /// Set when this function returns a float, which leaves in fa0.
    ret_float: Option<MirType>,
    ret_aggregate: Option<Aggregate>,
    sret_offset: Option<i32>,
}

#[allow(clippy::too_many_arguments)]
fn emit_planned_block_riscv<W: Write>(
    func: &Function,
    index: usize,
    block_start: usize,
    next: &str,
    plan: &FunctionEmissionPlan,
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), LaminaError> {
    let block = &func.blocks[index];
    let func_name = ctx.func_name;
    let aggregate_flow =
        ctx.ret_aggregate.is_some() || func.sig.params.iter().any(|p| p.aggregate.is_some());
    if plan.aligned_headers.contains(&index) {
        writeln!(writer, "    .p2align 4")?;
    }
    writeln!(writer, ".L_{func_name}.{}:", block.label)?;
    let mut position = 0;
    while position < block.instructions.len() {
        let inst = &block.instructions[position];
        if let Some(pairs) = self_tail_arguments(func, inst).filter(|_| !aggregate_flow) {
            let schedule = self_tail_schedule(&pairs, |reg| match reg {
                Register::Virtual(v) => reg_alloc
                    .get_mapping(v)
                    .map(|name| Register::Physical(PhysicalReg::new(name, RegisterClass::Gpr)))
                    .unwrap_or_else(|| reg.clone()),
                _ => reg.clone(),
            });
            for (src, dst) in schedule {
                let src = match src {
                    CopyLocation::Value(src) => src,
                    CopyLocation::Scratch => Operand::Register(Register::Physical(
                        PhysicalReg::new("a2", RegisterClass::Gpr),
                    )),
                };
                load_operand_to_register(&src, writer, reg_alloc, stack_slots, "a0")?;
                match dst {
                    CopyLocation::Value(Operand::Register(Register::Virtual(dst))) => {
                        store_register_to_register("a0", &dst, writer, reg_alloc, stack_slots)?
                    }
                    CopyLocation::Value(Operand::Register(Register::Physical(dst))) => {
                        writeln!(writer, "    mv {}, a0", dst.name)?
                    }
                    CopyLocation::Scratch => writeln!(writer, "    mv a2, a0")?,
                    _ => unreachable!(),
                }
            }
            if func.entry != next {
                writeln!(writer, "    j .L_{func_name}.{}", func.entry)?;
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
            ) = (&inst.instr, &block.instructions[position + 1].instr)
            {
                emit_source_loc(writer, ctx.settings, inst.loc)?;
                load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;
                load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;
                let unsigned = matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                );
                extend_integer(writer, "a0", ty, ctx.target, !unsigned, func.pointer_width)?;
                extend_integer(writer, "a1", ty, ctx.target, !unsigned, func.pointer_width)?;
                let (invert, taken, other) =
                    plan.branch(func, index, next, true_target, false_target);
                let (mnemonic, lhs, rhs) = match (op, invert) {
                    (IntCmpOp::Eq, false) | (IntCmpOp::Ne, true) => ("beq", "a0", "a1"),
                    (IntCmpOp::Ne, false) | (IntCmpOp::Eq, true) => ("bne", "a0", "a1"),
                    (IntCmpOp::SLt, false) | (IntCmpOp::SGe, true) => ("blt", "a0", "a1"),
                    (IntCmpOp::SLe, false) | (IntCmpOp::SGt, true) => ("bge", "a1", "a0"),
                    (IntCmpOp::SGt, false) | (IntCmpOp::SLe, true) => ("blt", "a1", "a0"),
                    (IntCmpOp::SGe, false) | (IntCmpOp::SLt, true) => ("bge", "a0", "a1"),
                    (IntCmpOp::ULt, false) | (IntCmpOp::UGe, true) => ("bltu", "a0", "a1"),
                    (IntCmpOp::ULe, false) | (IntCmpOp::UGt, true) => ("bgeu", "a1", "a0"),
                    (IntCmpOp::UGt, false) | (IntCmpOp::ULe, true) => ("bltu", "a1", "a0"),
                    (IntCmpOp::UGe, false) | (IntCmpOp::ULt, true) => ("bgeu", "a0", "a1"),
                };
                writeln!(
                    writer,
                    "    {mnemonic} {lhs}, {rhs}, .L_{func_name}.{taken}"
                )?;
                if let Some(other) = other {
                    writeln!(writer, "    j .L_{func_name}.{other}")?;
                }
            }
            position += 1;
        } else if matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
        } else {
            ctx.loc = inst.loc;
            emit_instruction_riscv(
                inst,
                block_start + position,
                writer,
                reg_alloc,
                stack_slots,
                ctx,
                func.pointer_width,
            )?;
        }
        position += 1;
    }
    Ok(())
}

fn emit_instruction_riscv<W: Write>(
    inst: &MirInst,
    instruction_index: usize,
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    ctx: &mut EmitCtx<'_>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    if let Some(operations) = integer_conversion_with_width(
        inst,
        if ctx.target.xlen == Xlen::Rv32 {
            MirType::Scalar(ScalarType::I32)
        } else {
            MirType::Scalar(ScalarType::I64)
        },
        pointer_width,
    ) {
        for operation in operations {
            emit_instruction_riscv(
                &operation,
                instruction_index,
                writer,
                reg_alloc,
                stack_slots,
                ctx,
                pointer_width,
            )?;
        }
        return Ok(());
    }
    let target_os = ctx.target_os;
    let settings = ctx.settings;
    let func_name = ctx.func_name;
    if settings.emit_asm_debug_lines
        && let Some(loc) = ctx.loc
    {
        writeln!(writer, "    .loc {} {} {}", loc.file, loc.line, loc.column)?;
    }
    match inst {
        MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
            simd::emit(inst, writer, reg_alloc, stack_slots, ctx.target)?
        }

        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if ctx.target.xlen == Xlen::Rv32 && *ty == MirType::Scalar(ScalarType::F64) {
                load_fp_operand_to_register(src, writer, reg_alloc, stack_slots, "fa0", false)?;
                let Register::Virtual(dst) = dst else {
                    return Err(LaminaError::ValidationError(
                        "RV32 f64 requires a stack slot".into(),
                    ));
                };
                store_fp_register_to_register("fa0", dst, writer, reg_alloc, stack_slots, false)?;
                return Ok(());
            }
            if matches!(ty, MirType::Vector(_)) {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("RISC-V backend does not support vector Copy of type {ty}"),
                )));
            }
            match src {
                Operand::Register(Register::Virtual(vreg)) => {
                    if let Some(physical) = reg_alloc.get_mapping(vreg) {
                        writeln!(writer, "    mv a0, {physical}")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        RiscVFrame::fp_load(
                            writer,
                            copy_load_mnemonic(ty, ctx.target, pointer_width),
                            "a0",
                            *offset,
                            "a0",
                        )?;
                    } else {
                        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                            format!("RISC-V Copy source {vreg:?} has no register or stack slot"),
                        )));
                    }
                }
                Operand::Register(Register::Physical(physical)) => {
                    writeln!(writer, "    mv a0, {}", physical.name)?;
                }
                Operand::Immediate(immediate) => {
                    let bits = match immediate {
                        Immediate::I8(value) => *value as i64 as u64,
                        Immediate::I16(value) => *value as i64 as u64,
                        Immediate::I32(value) => *value as i64 as u64,
                        Immediate::I64(value) => *value as u64,
                        Immediate::F32(value) => u64::from(value.to_bits()),
                        Immediate::F64(value) => value.to_bits(),
                    };
                    writeln!(writer, "    li a0, {}", bits as i64)?;
                }
            }
            extend_integer(writer, "a0", ty, ctx.target, true, pointer_width)?;
            match dst {
                Register::Virtual(vreg) => {
                    if let Some(physical) = reg_alloc.get_mapping(vreg) {
                        writeln!(writer, "    mv {physical}, a0")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        RiscVFrame::fp_store(writer, ctx.target.store_word(), "a0", *offset, "t0")?;
                    } else {
                        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                            format!(
                                "RISC-V Copy destination {vreg:?} has no register or stack slot"
                            ),
                        )));
                    }
                }
                Register::Physical(phys) if phys.name != "a0" => {
                    writeln!(writer, "    mv {}, a0", phys.name)?;
                }
                Register::Physical(_) => {}
            }
        }
        MirInst::SymbolAddress { dst, symbol } => {
            let label = RiscVAbi::new(ctx.target_os).mangle_function_name(symbol);
            writeln!(writer, "    la a0, {label}")?;
            match dst {
                Register::Virtual(vreg) => {
                    store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(phys) if phys.name != "a0" => {
                    writeln!(writer, "    mv {}, a0", phys.name)?;
                }
                Register::Physical(_) => {}
            }
        }
        MirInst::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).ok_or_else(|| {
                LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
                    "RISC-V stack address {slot:?} has no frame slot"
                )))
            })?;
            RiscVFrame::fp_address(writer, "a0", *offset)?;
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Lea { dst, base, offset } => {
            let Register::Virtual(base_vreg) = base else {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("RISC-V lea base {base:?} is not a virtual register"),
                )));
            };
            if let Some(physical) = reg_alloc.get_mapping(base_vreg) {
                writeln!(writer, "    mv a0, {physical}")?;
            } else if let Some(slot) = stack_slots.get(base_vreg) {
                RiscVFrame::fp_load(writer, ctx.target.load_word(), "a0", *slot, "a0")?;
            } else {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("RISC-V lea base {base_vreg:?} has no register or stack slot"),
                )));
            }
            if (-2048..=2047).contains(&i64::from(*offset)) {
                if *offset != 0 {
                    writeln!(writer, "    addi a0, a0, {offset}")?;
                }
            } else {
                RiscVFrame::load_imm(writer, "t0", i64::from(*offset))?;
                writeln!(writer, "    add a0, a0, t0")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

            if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                extend_integer(writer, "a0", ty, ctx.target, false, pointer_width)?;
                extend_integer(writer, "a1", ty, ctx.target, false, pointer_width)?;
            }
            if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                extend_integer(writer, "a0", ty, ctx.target, true, pointer_width)?;
                if !matches!(op, IntBinOp::AShr) {
                    extend_integer(writer, "a1", ty, ctx.target, true, pointer_width)?;
                }
            }
            let (zero, overflow) = op.division_checks(
                *ty,
                lhs.integer_constant(),
                rhs.integer_constant(),
                pointer_width,
            );
            if zero {
                writeln!(writer, "    bnez a1, 1f\n    .word 0\n1:")?;
            }
            if overflow {
                let minimum = ty.signed_integer(
                    1i64 << (ty.size_bytes(pointer_width) * 8 - 1),
                    pointer_width,
                );
                writeln!(
                    writer,
                    "    li t0, -1\n    bne a1, t0, 2f\n    li t0, {minimum}\n    bne a0, t0, 2f\n    .word 0\n2:"
                )?;
            }
            if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr)
                && ty.size_bytes(pointer_width) < 8
            {
                let mask = ty.size_bytes(pointer_width) * 8 - 1;
                writeln!(writer, "    andi a1, a1, {mask}")?;
            }
            match op {
                IntBinOp::Add => writeln!(writer, "    add a0, a0, a1")?,
                IntBinOp::Sub => writeln!(writer, "    sub a0, a0, a1")?,
                IntBinOp::Mul
                | IntBinOp::SDiv
                | IntBinOp::UDiv
                | IntBinOp::SRem
                | IntBinOp::URem
                    if !ctx.target.m =>
                {
                    return Err(LaminaError::ValidationError(format!(
                        "{op:?} needs the M extension, but the target is {}. Multiply and \
                         divide have no base-ISA encoding; select an ISA with M or lower \
                         the operation before codegen.",
                        ctx.target.isa_name()
                    )));
                }
                IntBinOp::Mul => writeln!(writer, "    mul a0, a0, a1")?,
                IntBinOp::SDiv => writeln!(writer, "    div a0, a0, a1")?,
                IntBinOp::UDiv => writeln!(writer, "    divu a0, a0, a1")?,
                IntBinOp::SRem => writeln!(writer, "    rem a0, a0, a1")?,
                IntBinOp::URem => writeln!(writer, "    remu a0, a0, a1")?,
                IntBinOp::And => writeln!(writer, "    and a0, a0, a1")?,
                IntBinOp::Or => writeln!(writer, "    or a0, a0, a1")?,
                IntBinOp::Xor => writeln!(writer, "    xor a0, a0, a1")?,
                IntBinOp::Shl => writeln!(writer, "    sll a0, a0, a1")?,
                IntBinOp::AShr => writeln!(writer, "    sra a0, a0, a1")?,
                IntBinOp::LShr => writeln!(writer, "    srl a0, a0, a1")?,
            }

            extend_integer(writer, "a0", ty, ctx.target, true, pointer_width)?;
            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

            if matches!(
                op,
                IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
            ) {
                extend_integer(writer, "a0", ty, ctx.target, false, pointer_width)?;
                extend_integer(writer, "a1", ty, ctx.target, false, pointer_width)?;
            }
            // Perform comparison
            emit_int_cmp_op(op, writer)?;

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Call { name, args, ret } => {
            let abi = RiscVAbi::new(target_os);
            let live_vregs = ctx
                .live_across_calls
                .get(&instruction_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let caller_saved_mappings = reg_alloc.caller_saved_mappings(live_vregs);
            spill_caller_saved(writer, &caller_saved_mappings, stack_slots, ctx.target)?;

            // Handle write and read intrinsics
            if (name == "lamina.write" || name == "lamina.read") && args.len() == 2 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "a1")?;
                load_operand_to_register(&args[1], writer, reg_alloc, stack_slots, "a2")?;
                writeln!(writer, "    li a0, {}", i32::from(name == "lamina.write"))?;
                if name == "lamina.read" {
                    writeln!(writer, "    beqz a2, 1f")?;
                }
                let libc = if name == "lamina.write" {
                    "write"
                } else {
                    "read"
                };
                let callee = abi
                    .call_stub(libc)
                    .unwrap_or_else(|| abi.mangle_function_name(libc));
                writeln!(writer, "    call {callee}")?;
                if name == "lamina.write" {
                    writeln!(writer, "    bgez a0, 1f")?;
                    writeln!(writer, "    li a0, -2")?;
                    writeln!(writer, "1:")?;
                } else {
                    writeln!(writer, "    bgez a0, 2f")?;
                    writeln!(writer, "    li a0, -2")?;
                    writeln!(writer, "    j 1f")?;
                    writeln!(writer, "2:")?;
                    writeln!(writer, "    bnez a0, 1f")?;
                    writeln!(writer, "    li a0, -1")?;
                    writeln!(writer, "1:")?;
                }
            } else if name == "lamina.writebyte" && args.len() == 1 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "a0")?;
                writeln!(writer, "    addi sp, sp, -16")?;
                writeln!(writer, "    sb a0, 0(sp)")?;
                writeln!(writer, "    li a0, 1")?;
                writeln!(writer, "    mv a1, sp")?;
                writeln!(writer, "    li a2, 1")?;
                let write = abi
                    .call_stub("write")
                    .unwrap_or_else(|| abi.mangle_function_name("write"));
                writeln!(writer, "    call {write}")?;
                writeln!(writer, "    bgez a0, 1f")?;
                writeln!(writer, "    li a0, -2")?;
                writeln!(writer, "1:")?;
                writeln!(writer, "    addi sp, sp, 16")?;
            } else if name == "lamina.readbyte" && args.is_empty() {
                writeln!(writer, "    addi sp, sp, -16")?;
                writeln!(writer, "    li a0, 0")?;
                writeln!(writer, "    mv a1, sp")?;
                writeln!(writer, "    li a2, 1")?;
                let read = abi
                    .call_stub("read")
                    .unwrap_or_else(|| abi.mangle_function_name("read"));
                writeln!(writer, "    call {read}")?;
                writeln!(writer, "    li t1, 1")?;
                writeln!(writer, "    beq a0, t1, 1f")?;
                writeln!(writer, "    bnez a0, 2f")?;
                writeln!(writer, "    li a0, -1")?;
                writeln!(writer, "    j 3f")?;
                writeln!(writer, "2:")?;
                writeln!(writer, "    li a0, -2")?;
                writeln!(writer, "    j 3f")?;
                writeln!(writer, "1:")?;
                writeln!(writer, "    lbu a0, 0(sp)")?;
                writeln!(writer, "3:")?;
                writeln!(writer, "    addi sp, sp, 16")?;
            } else {
                // General function call implementation
                // RISC-V calling convention: first 8 args in a0-a7, remaining on stack

                let mut kinds = ctx.signatures.arguments(name, args, ctx.func);
                // LP64D sends a variadic argument through the integer bank, a float
                // included, so the callee's va_arg finds the bits where it looks.
                if let Some(fixed) = ctx.signatures.fixed_arguments(name) {
                    let pointer = target_pointer(ctx.target);
                    for kind in kinds.iter_mut().skip(fixed) {
                        if kind.ty.size_bytes(pointer) > pointer.bytes() {
                            return Err(LaminaError::CodegenError(
                                CodegenError::UnsupportedFeature(format!(
                                    "a variadic {} argument to @{name} spans two {} registers, \
                                     which this backend does not yet place",
                                    kind.ty,
                                    ctx.target.isa_name()
                                )),
                            ));
                        }
                        if kind.float() {
                            kind.ty = MirType::Scalar(ScalarType::I64);
                        }
                    }
                }
                let result_slot = match ret {
                    Some(Register::Virtual(vreg)) => stack_slots.get(vreg).copied(),
                    _ => None,
                };
                let hidden = match ctx.signatures.result(name) {
                    Some(aggregate)
                        if aggregate.size as u64
                            > Aggregate::max_in_registers(target_pointer(ctx.target)) =>
                    {
                        Some(result_slot.ok_or_else(|| {
                            LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                                "RISC-V aggregate result has no frame slot".to_string(),
                            ))
                        })?)
                    }
                    _ => None,
                };
                let stack_space = pass_riscv_arguments(
                    writer,
                    reg_alloc,
                    stack_slots,
                    args,
                    &kinds,
                    hidden,
                    ctx.target,
                    pointer_width,
                )?;

                // Resolve function name (check for intrinsic stubs first)
                let target_sym = if let Some(stub) = abi.call_stub(name) {
                    stub
                } else {
                    abi.mangle_function_name(name)
                };

                // Emit call instruction
                writeln!(writer, "    call {target_sym}")?;

                // Clean up stack arguments
                if stack_space > 0 {
                    RiscVFrame::adjust_sp(writer, stack_space as i64)?;
                }
            }

            reload_caller_saved(writer, &caller_saved_mappings, stack_slots, ctx.target)?;

            // Handle return value: a0, or fa0 when the callee returns a float.
            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                match ctx.signatures.result(name) {
                    Some(aggregate)
                        if aggregate.size as u64
                            > Aggregate::max_in_registers(target_pointer(ctx.target)) => {}
                    Some(aggregate) => {
                        let slot = stack_slots.get(vreg).copied().ok_or_else(|| {
                            LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                                "RISC-V aggregate result has no frame slot".to_string(),
                            ))
                        })?;
                        RiscVFrame::fp_store(writer, ctx.target.store_word(), "a0", slot, "t0")?;
                        if aggregate.words_of(target_pointer(ctx.target)) > 1 {
                            let at = slot + ctx.target.word_bytes();
                            RiscVFrame::fp_store(writer, ctx.target.store_word(), "a1", at, "t0")?;
                        }
                    }
                    None => match ctx.signatures.float_result(name) {
                        Some(ty) => store_fp_register_to_register(
                            "fa0",
                            vreg,
                            writer,
                            reg_alloc,
                            stack_slots,
                            ty.size_bytes(pointer_width) == 4,
                        )?,
                        None => {
                            store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?
                        }
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
            let live_vregs = ctx
                .live_across_calls
                .get(&instruction_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let caller_saved_mappings = reg_alloc.caller_saved_mappings(live_vregs);
            spill_caller_saved(writer, &caller_saved_mappings, stack_slots, ctx.target)?;

            // t1 is a temporary the argument registers never overlap, so the target
            // survives the argument setup.
            load_operand_to_register(target, writer, reg_alloc, stack_slots, "t1")?;
            let mut kinds: Vec<ArgKind> = arg_types
                .iter()
                .map(|ty| ArgKind {
                    ty: *ty,
                    aggregate: None,
                })
                .collect();
            if let Some(fixed) = variadic_fixed {
                for kind in kinds.iter_mut().skip(*fixed).filter(|kind| kind.float()) {
                    kind.ty = MirType::Scalar(ScalarType::I64);
                }
            }
            let stack_space = pass_riscv_arguments(
                writer,
                reg_alloc,
                stack_slots,
                args,
                &kinds,
                None,
                ctx.target,
                pointer_width,
            )?;
            writeln!(writer, "    jalr t1")?;
            if stack_space > 0 {
                RiscVFrame::adjust_sp(writer, stack_space as i64)?;
            }
            reload_caller_saved(writer, &caller_saved_mappings, stack_slots, ctx.target)?;
            if let Some(Register::Virtual(vreg)) = ret {
                match ret_ty {
                    Some(ty) if ty.is_float() => store_fp_register_to_register(
                        "fa0",
                        vreg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        ty.size_bytes(pointer_width) == 4,
                    )?,
                    _ => store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::TailCall { name, args } => {
            let abi = RiscVAbi::new(target_os);
            if ctx.ret_aggregate.is_some()
                || ctx.signatures.result(name).is_some()
                || ctx
                    .signatures
                    .arguments(name, args, ctx.func)
                    .iter()
                    .any(|kind| kind.aggregate.is_some())
            {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RISC-V: TailCall with an aggregate is not supported".to_string(),
                )));
            }
            let arg_regs = RiscVAbi::ARG_REGISTERS;
            let kinds = ctx.signatures.arguments(name, args, ctx.func);
            let places = classify_riscv_arguments(
                &kinds,
                0,
                target_pointer(ctx.target),
                target_arch(ctx.target),
            );
            for ((arg, kind), place) in args.iter().zip(&kinds).zip(&places) {
                match *place {
                    RiscVArgPlace::Gpr(bank) => load_operand_to_register(
                        arg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        arg_regs[bank],
                    )?,
                    RiscVArgPlace::Fpr(bank) => load_fp_operand_to_register(
                        arg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        &format!("fa{bank}"),
                        kind.ty.size_bytes(pointer_width) == 4,
                    )?,
                    _ => {}
                }
            }
            for (arg, place) in args.iter().zip(&places) {
                let RiscVArgPlace::Stack(slot) = *place else {
                    continue;
                };
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                // A tail call reuses the caller's stack argument area, which in a
                // variadic function begins above the save area rather than at `fp`.
                let at = ctx.va_area.map_or(0, |area| area.reserve())
                    + slot as i32 * ctx.target.word_bytes();
                RiscVFrame::fp_store(writer, ctx.target.store_word(), "t0", at, "t0")?;
            }
            let target_sym = if let Some(stub) = abi.call_stub(name) {
                stub
            } else {
                abi.mangle_function_name(name)
            };
            release_va_save_area(writer, ctx)?;
            RiscVFrame::generate_tail_epilogue(
                writer,
                ctx.stack_size,
                &ctx.callee_saved_registers,
                &target_sym,
                ctx.target,
            )
            .map_err(|e| {
                LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(e.to_string()))
            })?;
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            let load_op = match ty {
                MirType::Scalar(ScalarType::I1) | MirType::Scalar(ScalarType::I8) => "lb",
                MirType::Scalar(ScalarType::I16) => "lh",
                MirType::Scalar(ScalarType::I32) => "lw",
                MirType::Scalar(ScalarType::I64) => "ld",
                MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => ctx.target.load_word(),
                MirType::Scalar(ScalarType::F32) => "flw",
                MirType::Scalar(ScalarType::F64) => "fld",
                MirType::Vector(_) => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!(
                            "RISC-V load unsupported for type {ty:?}. \
                             Vector types are not yet implemented for RISC-V."
                        ),
                    )));
                }
            };

            let is_float = matches!(
                ty,
                MirType::Scalar(ScalarType::F32) | MirType::Scalar(ScalarType::F64)
            );

            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "t0")?
                        }
                        Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
                    }

                    if is_float {
                        // Load to floating-point register fa0
                        writeln!(writer, "    {load_op} fa0, {offset}(t0)")?;

                        if let Register::Virtual(vreg) = dst {
                            let is_f32 = matches!(ty, MirType::Scalar(ScalarType::F32));
                            store_fp_register_to_register(
                                "fa0",
                                vreg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                is_f32,
                            )?;
                        }
                    } else {
                        writeln!(writer, "    {load_op} a0, {offset}(t0)")?;

                        if let Register::Virtual(vreg) = dst {
                            store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
                        }
                    }
                }
                _ => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        "RISC-V load supports only base+offset addressing. \
                             Complex addressing modes (BaseIndexScale) are not yet implemented."
                            .to_string(),
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
                MirType::Scalar(ScalarType::I1) | MirType::Scalar(ScalarType::I8) => "sb",
                MirType::Scalar(ScalarType::I16) => "sh",
                MirType::Scalar(ScalarType::I32) => "sw",
                MirType::Scalar(ScalarType::I64) => "sd",
                MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => ctx.target.store_word(),
                MirType::Scalar(ScalarType::F32) => "fsw",
                MirType::Scalar(ScalarType::F64) => "fsd",
                MirType::Vector(_) => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!(
                            "RISC-V store unsupported for type {ty:?}. \
                             Vector types are not yet implemented for RISC-V."
                        ),
                    )));
                }
            };

            let is_float = matches!(
                ty,
                MirType::Scalar(ScalarType::F32) | MirType::Scalar(ScalarType::F64)
            );
            let is_f32 = matches!(ty, MirType::Scalar(ScalarType::F32));

            if is_float {
                load_fp_operand_to_register(src, writer, reg_alloc, stack_slots, "fa0", is_f32)?;
            } else {
                load_operand_to_register(src, writer, reg_alloc, stack_slots, "a0")?;
            }

            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "t0")?
                        }
                        Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
                    }

                    if is_float {
                        writeln!(writer, "    {store_op} fa0, {offset}(t0)")?;
                    } else {
                        writeln!(writer, "    {store_op} a0, {offset}(t0)")?;
                    }
                }
                _ => {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        "RISC-V store supports only base+offset addressing. \
                             Complex addressing modes (BaseIndexScale) are not yet implemented."
                            .to_string(),
                    )));
                }
            }
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match (ctx.ret_aggregate, ctx.sret_offset) {
                    (Some(aggregate), Some(offset)) => {
                        load_operand_to_register(val, writer, reg_alloc, stack_slots, "t0")?;
                        RiscVFrame::fp_load(writer, ctx.target.load_word(), "a1", offset, "a1")?;
                        for index in 0..aggregate.words_of(target_pointer(ctx.target)) {
                            let at = index as i32 * ctx.target.word_bytes();
                            writeln!(writer, "    {} a7, {at}(t0)", ctx.target.load_word())?;
                            writeln!(writer, "    {} a7, {at}(a1)", ctx.target.store_word())?;
                        }
                        writeln!(writer, "    mv a0, a1")?;
                    }
                    (Some(aggregate), None) => {
                        load_operand_to_register(val, writer, reg_alloc, stack_slots, "t0")?;
                        writeln!(writer, "    {} a0, 0(t0)", ctx.target.load_word())?;
                        if aggregate.words_of(target_pointer(ctx.target)) > 1 {
                            let at = ctx.target.word_bytes();
                            writeln!(writer, "    {} a1, {at}(t0)", ctx.target.load_word())?;
                        }
                    }
                    (None, _) => match ctx.ret_float {
                        Some(ty) => load_fp_operand_to_register(
                            val,
                            writer,
                            reg_alloc,
                            stack_slots,
                            "fa0",
                            ty.size_bytes(pointer_width) == 4,
                        )?,
                        None => {
                            load_operand_to_register(val, writer, reg_alloc, stack_slots, "a0")?
                        }
                    },
                }
            }
            // Epilogue
            release_va_save_area(writer, ctx)?;
            RiscVFrame::generate_epilogue(
                writer,
                ctx.stack_size,
                &ctx.callee_saved_registers,
                ctx.target,
            )?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    j .L_{func_name}.{target}")?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_register(vreg, writer, reg_alloc, stack_slots, "t0")?;
                writeln!(writer, "    bnez t0, .L_{func_name}.{true_target}")?;
                writeln!(writer, "    j .L_{func_name}.{false_target}")?;
            }
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes(pointer_width) == 4;
            ctx.target
                .require_float(is_f32)
                .map_err(|e| LaminaError::ValidationError(format!("{:?}: {e}", op)))?;
            let suffix = if is_f32 { "s" } else { "d" };

            // Load operands to floating-point registers
            load_fp_operand_to_register(lhs, writer, reg_alloc, stack_slots, "fa0", is_f32)?;
            load_fp_operand_to_register(rhs, writer, reg_alloc, stack_slots, "fa1", is_f32)?;

            // Perform floating-point operation
            match op {
                FloatBinOp::FAdd => writeln!(writer, "    fadd.{suffix} fa0, fa0, fa1")?,
                FloatBinOp::FSub => writeln!(writer, "    fsub.{suffix} fa0, fa0, fa1")?,
                FloatBinOp::FMul => writeln!(writer, "    fmul.{suffix} fa0, fa0, fa1")?,
                FloatBinOp::FDiv => writeln!(writer, "    fdiv.{suffix} fa0, fa0, fa1")?,
            }

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let is_f32 = from.size_bytes(pointer_width) == 4;
            ctx.target
                .require_float(is_f32)
                .map_err(|e| LaminaError::ValidationError(format!("floattoint: {e}")))?;
            if ctx.target.xlen == Xlen::Rv32 {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RV32 float to integer conversion is not supported".to_string(),
                )));
            }
            let suffix = if is_f32 { "s" } else { "d" };
            load_fp_operand_to_register(src, writer, reg_alloc, stack_slots, "fa0", is_f32)?;
            // fcvt saturates to the 64-bit range already; NaN yields the maximum, so
            // mask it to zero with the ordered comparison.
            let form = if *signed { "l" } else { "lu" };
            writeln!(writer, "    fcvt.{form}.{suffix} a0, fa0, rtz")?;
            writeln!(writer, "    feq.{suffix} t0, fa0, fa0")?;
            writeln!(writer, "    neg t0, t0")?;
            writeln!(writer, "    and a0, a0, t0")?;
            let bits = to.size_bytes(pointer_width) * 8;
            if bits < 64 {
                let compare = if *signed { "slt" } else { "sltu" };
                let max = if *signed {
                    (1i64 << (bits - 1)) - 1
                } else {
                    (1i64 << bits) - 1
                };
                clamp_above(writer, max, compare)?;
                if *signed {
                    clamp_below(writer, -(1i64 << (bits - 1)))?;
                }
                sign_extend_to_xlen(writer, bits)?;
            }
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let is_f32 = to.size_bytes(pointer_width) == 4;
            ctx.target
                .require_float(is_f32)
                .map_err(|e| LaminaError::ValidationError(format!("inttofloat: {e}")))?;
            let bits = from.size_bytes(pointer_width) * 8;
            let xlen_bits = ctx.target.xlen.bits() as usize;
            if bits > xlen_bits {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!(
                        "RV{xlen_bits} integer to float conversion from a {bits}-bit source is not supported: the source does not fit in one register"
                    ),
                )));
            }
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "a0")?;
            extend_to_xlen(writer, bits, xlen_bits, *signed)?;
            let suffix = if is_f32 { "s" } else { "d" };
            let form = match (ctx.target.xlen, *signed) {
                (Xlen::Rv32, true) => "w",
                (Xlen::Rv32, false) => "wu",
                (Xlen::Rv64, true) => "l",
                (Xlen::Rv64, false) => "lu",
            };
            writeln!(writer, "    fcvt.{suffix}.{form} fa0, a0")?;
            if let Register::Virtual(vreg) = dst {
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let to_f32 = to.size_bytes(pointer_width) == 4;
            ctx.target
                .require_float(true)
                .map_err(|e| LaminaError::ValidationError(format!("floatconvert: {e}")))?;
            ctx.target
                .require_float(false)
                .map_err(|e| LaminaError::ValidationError(format!("floatconvert: {e}")))?;
            let from_f32 = from.size_bytes(pointer_width) == 4;
            load_fp_operand_to_register(src, writer, reg_alloc, stack_slots, "fa0", from_f32)?;
            if to_f32 {
                writeln!(writer, "    fcvt.s.d fa0, fa0")?;
            } else {
                writeln!(writer, "    fcvt.d.s fa0, fa0")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, to_f32)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let is_f32 = ty.size_bytes(pointer_width) == 4;
            let suffix = if is_f32 { "s" } else { "d" };

            // Load operand to floating-point register
            load_fp_operand_to_register(src, writer, reg_alloc, stack_slots, "fa0", is_f32)?;

            // Perform floating-point unary operation
            match op {
                FloatUnOp::FNeg => writeln!(writer, "    fneg.{suffix} fa0, fa0")?,
                FloatUnOp::FSqrt => writeln!(writer, "    fsqrt.{suffix} fa0, fa0")?,
            }

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes(pointer_width) == 4;
            ctx.target
                .require_float(is_f32)
                .map_err(|e| LaminaError::ValidationError(format!("{op:?}: {e}")))?;
            let suffix = if is_f32 { "s" } else { "d" };

            // Load operands to floating-point registers
            load_fp_operand_to_register(lhs, writer, reg_alloc, stack_slots, "fa0", is_f32)?;
            load_fp_operand_to_register(rhs, writer, reg_alloc, stack_slots, "fa1", is_f32)?;

            // Perform floating-point comparison
            // Result goes into integer register a0
            match op {
                FloatCmpOp::Eq => writeln!(writer, "    feq.{suffix} a0, fa0, fa1")?,
                FloatCmpOp::Ne => {
                    // NE = !(a == b)
                    writeln!(writer, "    feq.{suffix} a0, fa0, fa1")?;
                    writeln!(writer, "    xori a0, a0, 1")?;
                }
                FloatCmpOp::Lt => writeln!(writer, "    flt.{suffix} a0, fa0, fa1")?,
                FloatCmpOp::Le => writeln!(writer, "    fle.{suffix} a0, fa0, fa1")?,
                FloatCmpOp::Gt => {
                    // GT = b < a
                    writeln!(writer, "    flt.{suffix} a0, fa1, fa0")?
                }
                FloatCmpOp::Ge => {
                    // GE = b <= a
                    writeln!(writer, "    fle.{suffix} a0, fa1, fa0")?
                }
            }

            // Store result to destination
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // Load false_val into a1, true_val into a0, then pick with branch.
            load_operand_to_register(false_val, writer, reg_alloc, stack_slots, "a1")?;
            load_operand_to_register(true_val, writer, reg_alloc, stack_slots, "a0")?;
            match cond {
                Register::Virtual(vreg) => {
                    load_register_to_register(vreg, writer, reg_alloc, stack_slots, "t0")?;
                }
                Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
            }
            // If condition is zero (false), replace a0 with a1.
            let sel = ctx.label_seq;
            ctx.label_seq = ctx.label_seq.saturating_add(1);
            writeln!(writer, "    bnez t0, .L_{func_name}.sel.{sel}")?;
            writeln!(writer, "    mv a0, a1")?;
            writeln!(writer, ".L_{func_name}.sel.{sel}:")?;
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            match value {
                Register::Virtual(v) => {
                    load_register_to_register(v, writer, reg_alloc, stack_slots, "a0")?;
                }
                Register::Physical(p) => writeln!(writer, "    mv a0, {}", p.name)?,
            }
            for (case_val, case_label) in cases {
                writeln!(writer, "    li t0, {case_val}")?;
                writeln!(writer, "    beq a0, t0, .L_{func_name}.{case_label}")?;
            }
            writeln!(writer, "    j .L_{func_name}.{default}")?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Unreachable => {
            // Emit an illegal instruction — RISC-V has no official trap mnemonic,
            // but encoding 0x00000000 is reserved and will cause an illegal-instruction exception.
            writeln!(writer, "    .word 0")?;
        }
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {
            // No-op in AOT path.
        }
        MirInst::VectorOp { .. } => simd::emit(inst, writer, reg_alloc, stack_slots, ctx.target)?,
        MirInst::IntUnary { op, ty, dst, src } => {
            bit::emit_int_unary(
                writer,
                *op,
                ty,
                dst,
                src,
                reg_alloc,
                stack_slots,
                pointer_width,
            )?;
        }
        MirInst::IntRotate {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            bit::emit_int_rotate(
                writer,
                *op,
                ty,
                dst,
                (lhs, rhs),
                reg_alloc,
                stack_slots,
                pointer_width,
            )?;
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            atomic::emit(inst, writer, reg_alloc, stack_slots, pointer_width)?
        }
        MirInst::ThreadLocalAddr { .. } => {
            return Err(unsupported_thread_local("RISC-V"));
        }
        MirInst::IntOverflow { .. } => {
            return Err(unsupported_overflow_flag("RISC-V"));
        }
        MirInst::InlineAsm { .. } => {
            return Err(unsupported_inline_asm("RISC-V"));
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            variadic::emit(
                inst,
                writer,
                reg_alloc,
                stack_slots,
                ctx.va_area,
                ctx.target,
            )?;
        }
        other => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("RISC-V backend: instruction not yet supported: {other}"),
            )));
        }
    }

    Ok(())
}

/// Gives back the bytes the prologue took below the incoming stack arguments, so the
/// epilogue's own unwinding lands on the caller's `sp`.
fn release_va_save_area<W: Write>(writer: &mut W, ctx: &EmitCtx<'_>) -> Result<(), LaminaError> {
    let reserve = ctx.va_area.map_or(0, |area| area.reserve());
    if reserve > 0 {
        RiscVFrame::adjust_sp(writer, i64::from(reserve))?;
    }
    Ok(())
}

fn spill_caller_saved<W: Write>(
    writer: &mut W,
    mappings: &[(VirtualReg, &'static str)],
    stack_slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
) -> Result<(), LaminaError> {
    for (vreg, physical) in mappings {
        let offset = stack_slots.get(vreg).ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
                "RISC-V caller-saved value {vreg:?} has no spill slot"
            )))
        })?;
        RiscVFrame::fp_store(writer, target.store_word(), physical, *offset, "t0")?;
    }
    Ok(())
}

fn reload_caller_saved<W: Write>(
    writer: &mut W,
    mappings: &[(VirtualReg, &'static str)],
    stack_slots: &HashMap<VirtualReg, i32>,
    target: RiscVTarget,
) -> Result<(), LaminaError> {
    for (vreg, physical) in mappings {
        let offset = stack_slots.get(vreg).ok_or_else(|| {
            LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
                "RISC-V caller-saved value {vreg:?} has no spill slot"
            )))
        })?;
        RiscVFrame::fp_load(writer, target.load_word(), physical, *offset, physical)?;
    }
    Ok(())
}

fn copy_load_mnemonic(
    ty: &MirType,
    target: RiscVTarget,
    pointer_width: PointerWidth,
) -> &'static str {
    match ty.size_bytes(pointer_width) {
        1 => "lb",
        2 => "lh",
        4 => "lw",
        _ => target.load_word(),
    }
}

fn extend_integer(
    w: &mut impl Write,
    register: &str,
    ty: &MirType,
    target: RiscVTarget,
    signed: bool,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
    ) {
        return Ok(());
    }
    let bits = ty.size_bytes(pointer_width) as u32 * 8;
    let shift = target.xlen.bits().saturating_sub(bits);
    if shift != 0 {
        writeln!(w, "    slli {register}, {register}, {shift}")?;
        let instruction = if signed { "srai" } else { "srli" };
        writeln!(w, "    {instruction} {register}, {register}, {shift}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
