use crate::mir::register::PhysicalReg;
use crate::mir_codegen::common::{
    CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, self_tail_arguments, self_tail_schedule,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note};
mod simd;
use crate::mir_codegen::simd::reserve_vector_slots;
mod util;

use crate::mir_codegen::common::{
    CallSignatures, align_up, integer_conversion_with_width, print_format_label, writeptr_width,
};

use util::{
    clamp_above, clamp_below, emit_int_cmp_op, load_fp_operand_to_register,
    load_operand_to_register, load_register_to_register, sign_extend_to_xlen,
    store_fp_register_to_register, store_register_to_register,
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
        common::{assign_stack_slots, compile_functions_parallel, parallel_codegen_error},
        validate_module_call_parameters,
    },
};

use lamina_codegen::{
    GraphColorAllocator, LinearScanAllocator, LiveInterval, LocalRegisterAllocator,
    riscv::{RiscVAbi, RiscVFrame, RiscVRegAlloc, RiscVTarget, Xlen},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    sync::Arc,
};

use crate::mir_codegen::common::CodegenBase;

/// Trait-backed MIR ⇒ RISC-V code generator.
pub struct RiscVCodegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> RiscVCodegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
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
    signatures: &CallSignatures,
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
        .filter(|i| instruction_type(i).is_some_and(|ty| *ty == MirType::f64()))
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
                if let MirInst::StackAddr { slot, .. } = inst
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

    let stack_size = stack_slots
        .values()
        .map(|offset| (-offset - 16) as usize)
        .max()
        .unwrap_or(0);
    let callee_saved_registers = reg_alloc.used_callee_saved_registers();
    RiscVFrame::generate_prologue(&mut output, stack_size, &callee_saved_registers, target)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    if !func.sig.params.is_empty() {
        let arg_regs = RiscVAbi::ARG_REGISTERS;
        let param_types: Vec<MirType> = func.sig.params.iter().map(|p| p.ty).collect();
        let places = classify_riscv_arguments(&param_types);
        for (param, place) in func.sig.params.iter().zip(&places) {
            let Register::Virtual(vreg) = &param.reg else {
                continue;
            };
            let Some(slot_off) = stack_slots.get(vreg) else {
                continue;
            };
            if let RiscVArgPlace::Fpr(bank) = *place {
                store_fp_register_to_register(
                    &format!("fa{bank}"),
                    vreg,
                    &mut output,
                    &reg_alloc,
                    &stack_slots,
                    param.ty.size_bytes() == 4,
                )
                .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                continue;
            }
            if let RiscVArgPlace::Gpr(bank) = *place {
                if let Some(physical) = reg_alloc.get_mapping(vreg) {
                    writeln!(output, "    mv {physical}, {}", arg_regs[bank]).map_err(|e| {
                        CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                    })?;
                } else {
                    RiscVFrame::fp_store(
                        &mut output,
                        target.store_word(),
                        arg_regs[bank],
                        *slot_off,
                        "t0",
                    )
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                }
            } else {
                // fp is the caller's sp, so stack args start there and go up.
                let RiscVArgPlace::Stack(slot) = *place else {
                    continue;
                };
                let caller_off = slot as i32 * target.word_bytes();
                RiscVFrame::fp_load(&mut output, target.load_word(), "t0", caller_off, "t0")
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                if let Some(physical) = reg_alloc.get_mapping(vreg) {
                    writeln!(output, "    mv {physical}, t0").map_err(|e| {
                        CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                    })?;
                } else {
                    RiscVFrame::fp_store(&mut output, target.store_word(), "t0", *slot_off, "t0").map_err(
                        |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                    )?;
                }
            }
        }
    }

    let mut ctx = EmitCtx {
        func_name,
        target,
        target_os,
        settings,
        debug_line: 0,
        label_seq: 0,
        live_across_calls,
        stack_size,
        callee_saved_registers,
        signatures,
        ret_float: func.sig.ret_ty.filter(MirType::is_float),
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
                instruction,
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

fn validate_rv32_i64(module: &MirModule, target: RiscVTarget) -> Result<(), LaminaError> {
    if target.xlen == Xlen::Rv64 {
        return Ok(());
    }
    let i64_type = |ty: &MirType| matches!(ty, MirType::Scalar(ScalarType::I64));
    let has_i64_global = module.globals.values().any(|global| i64_type(&global.ty));
    let has_i64_function = module.functions.values().any(|function| {
        function.sig.params.iter().any(|param| i64_type(&param.ty))
            || function.sig.ret_ty.as_ref().is_some_and(&i64_type)
            || function.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|i| instruction_type(i).is_some_and(&i64_type)
                        || matches!(i, MirInst::IntConvert { from, to, .. } if i64_type(from) || i64_type(to)))
            })
    });
    if has_i64_global || has_i64_function {
        target.require_i64().map_err(LaminaError::ValidationError)?;
    }
    Ok(())
}

fn instruction_type(instruction: &MirInst) -> Option<&MirType> {
    match instruction {
        MirInst::Copy { ty, .. }
        | MirInst::IntBinary { ty, .. }
        | MirInst::FloatBinary { ty, .. }
        | MirInst::FloatUnary { ty, .. }
        | MirInst::IntCmp { ty, .. }
        | MirInst::FloatCmp { ty, .. }
        | MirInst::Select { ty, .. }
        | MirInst::Load { ty, .. }
        | MirInst::Store { ty, .. }
        | MirInst::VectorOp { ty, .. } => Some(ty),
        #[cfg(feature = "nightly")]
        MirInst::SimdBinary { ty, .. }
        | MirInst::SimdUnary { ty, .. }
        | MirInst::SimdTernary { ty, .. }
        | MirInst::SimdShuffle { ty, .. }
        | MirInst::SimdExtract { ty, .. }
        | MirInst::SimdInsert { ty, .. }
        | MirInst::SimdLoad { ty, .. }
        | MirInst::SimdStore { ty, .. }
        | MirInst::AtomicLoad { ty, .. }
        | MirInst::AtomicStore { ty, .. }
        | MirInst::AtomicBinary { ty, .. }
        | MirInst::AtomicCompareExchange { ty, .. } => Some(ty),
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
    let arch = match target.xlen {
        Xlen::Rv32 => TargetArchitecture::Riscv32,
        Xlen::Rv64 => TargetArchitecture::Riscv64,
    };
    validate_module_call_parameters(module, arch)?;
    validate_rv32_i64(module, target)?;
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(CallSignatures::from_module(module));
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
    writeln!(writer, "{}", abi.get_print_format())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    emit_debug_file(writer, settings)?;
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;

    for func_name in &module.external_functions {
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

/// Per-function state shared by the block and instruction emitters.
/// Where an argument sits under lp64d: an integer register, a float register from a
/// bank of its own, or an eight-byte stack slot.
#[derive(Clone, Copy)]
enum RiscVArgPlace {
    Gpr(usize),
    Fpr(usize),
    Stack(usize),
}

/// Floats take the next fa register and integers the next a register, from banks that
/// run out separately. A float left without an fa register falls back to the integer
/// bank before the stack, carrying its bits.
/// Puts each argument where the call expects it and reserves the outgoing stack area,
/// returning how many bytes of it to release after the call.
fn pass_riscv_arguments<W: Write>(
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    args: &[Operand],
    types: &[MirType],
    target: RiscVTarget,
) -> Result<usize, LaminaError> {
    let arg_regs = RiscVAbi::ARG_REGISTERS;
    let places = classify_riscv_arguments(types);
    for ((arg, ty), place) in args.iter().zip(types).zip(&places) {
        match *place {
            RiscVArgPlace::Gpr(bank) => {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[bank])?;
            }
            RiscVArgPlace::Fpr(bank) => {
                load_fp_operand_to_register(
                    arg,
                    writer,
                    reg_alloc,
                    stack_slots,
                    &format!("fa{bank}"),
                    ty.size_bytes() == 4,
                )?;
            }
            RiscVArgPlace::Stack(_) => {}
        }
    }

    let stack_args = places
        .iter()
        .filter(|place| matches!(place, RiscVArgPlace::Stack(_)))
        .count();
    if stack_args == 0 {
        return Ok(0);
    }
    let stack_space = align_up(stack_args * target.word_bytes() as usize, 16);
    RiscVFrame::adjust_sp(writer, -(stack_space as i64))?;
    for (arg, place) in args.iter().zip(&places) {
        let RiscVArgPlace::Stack(slot) = *place else {
            continue;
        };
        let offset = slot * target.word_bytes() as usize;
        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
        writeln!(writer, "    {} t0, {offset}(sp)", target.store_word())?;
    }
    Ok(stack_space)
}

/// Format label for a print intrinsic. Mach-O has no local label prefix, so the same
/// strings are emitted with two leading underscores there.
fn fmt_label(call: &str, target_os: TargetOperatingSystem) -> String {
    let label = print_format_label(call);
    match target_os {
        TargetOperatingSystem::MacOS => format!("__{}", label.trim_start_matches(".L_")),
        _ => label.to_string(),
    }
}

/// Name of printf for this target, honouring any stub the ABI prefers.
fn printf_symbol(abi: &RiscVAbi, target_os: TargetOperatingSystem) -> String {
    abi.call_stub("print").unwrap_or_else(|| match target_os {
        TargetOperatingSystem::MacOS => "_printf".to_string(),
        _ => "printf".to_string(),
    })
}

fn classify_riscv_arguments(types: &[MirType]) -> Vec<RiscVArgPlace> {
    let banks = RiscVAbi::ARG_REGISTERS.len();
    let mut gprs = 0;
    let mut fprs = 0;
    let mut stack = 0;
    let mut places = Vec::with_capacity(types.len());
    for ty in types {
        if ty.is_float() && fprs < banks {
            places.push(RiscVArgPlace::Fpr(fprs));
            fprs += 1;
        } else if gprs < banks {
            places.push(RiscVArgPlace::Gpr(gprs));
            gprs += 1;
        } else {
            places.push(RiscVArgPlace::Stack(stack));
            stack += 1;
        }
    }
    places
}

/// Everything the instruction emitter needs beyond the instruction itself.
struct EmitCtx<'a> {
    func_name: &'a str,
    target: RiscVTarget,
    target_os: TargetOperatingSystem,
    settings: &'a MirCodegenSettings,
    debug_line: u32,
    /// Sequence for compiler-generated labels, so they do not depend on any address.
    label_seq: u32,
    live_across_calls: HashMap<usize, Vec<VirtualReg>>,
    stack_size: usize,
    callee_saved_registers: Vec<&'static str>,
    signatures: &'a CallSignatures,
    /// Set when this function returns a float, which leaves in fa0.
    ret_float: Option<MirType>,
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
    if plan.aligned_headers.contains(&index) {
        writeln!(writer, "    .p2align 4")?;
    }
    writeln!(writer, ".L_{func_name}.{}:", block.label)?;
    let mut position = 0;
    while position < block.instructions.len() {
        let inst = &block.instructions[position];
        if let Some(pairs) = self_tail_arguments(func, inst) {
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
            ) = (inst, &block.instructions[position + 1])
            {
                load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;
                load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;
                let unsigned = matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                );
                extend_integer(writer, "a0", ty, ctx.target, !unsigned)?;
                extend_integer(writer, "a1", ty, ctx.target, !unsigned)?;
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
        } else if matches!(inst, MirInst::Jmp { target } if target == next) {
        } else {
            emit_instruction_riscv(
                inst,
                block_start + position,
                writer,
                reg_alloc,
                stack_slots,
                ctx,
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
) -> Result<(), LaminaError> {
    if let Some(operations) = integer_conversion_with_width(
        inst,
        if ctx.target.xlen == Xlen::Rv32 {
            MirType::i32()
        } else {
            MirType::i64()
        },
    ) {
        for operation in operations {
            emit_instruction_riscv(
                &operation,
                instruction_index,
                writer,
                reg_alloc,
                stack_slots,
                ctx,
            )?;
        }
        return Ok(());
    }
    let target_os = ctx.target_os;
    let settings = ctx.settings;
    let func_name = ctx.func_name;
    if settings.emit_asm_debug_lines {
        ctx.debug_line = ctx.debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", ctx.debug_line)?;
    }
    match inst {
        MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
            simd::emit(inst, writer, reg_alloc, stack_slots, ctx.target)?
        }

        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if ctx.target.xlen == Xlen::Rv32 && *ty == MirType::f64() {
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
                            copy_load_mnemonic(ty, ctx.target),
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
            extend_integer(writer, "a0", ty, ctx.target, true)?;
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
                extend_integer(writer, "a0", ty, ctx.target, false)?;
                extend_integer(writer, "a1", ty, ctx.target, false)?;
            }
            if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                extend_integer(writer, "a0", ty, ctx.target, true)?;
                if !matches!(op, IntBinOp::AShr) {
                    extend_integer(writer, "a1", ty, ctx.target, true)?;
                }
            }
            let (zero, overflow) =
                op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant());
            if zero {
                writeln!(writer, "    bnez a1, 1f\n    .word 0\n1:")?;
            }
            if overflow {
                let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
                writeln!(
                    writer,
                    "    li t0, -1\n    bne a1, t0, 2f\n    li t0, {minimum}\n    bne a0, t0, 2f\n    .word 0\n2:"
                )?;
            }
            if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) && ty.size_bytes() < 8
            {
                let mask = ty.size_bytes() * 8 - 1;
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

            extend_integer(writer, "a0", ty, ctx.target, true)?;
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
                extend_integer(writer, "a0", ty, ctx.target, false)?;
                extend_integer(writer, "a1", ty, ctx.target, false)?;
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

            // Handle print intrinsic
            if name == "writeptr" && args.len() == 2 {
                // write(1, ptr, width): the pointee's bytes go out straight from memory.
                let width = writeptr_width(&args[1])?;
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "a1")?;
                writeln!(writer, "    li a2, {width}")?;
                writeln!(writer, "    li a0, 1")?;
                let write = abi
                    .call_stub("write")
                    .unwrap_or_else(|| "write".to_string());
                writeln!(writer, "    call {write}")?;
            } else if (name == "write" || name == "read") && args.len() == 2 {
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "a1")?;
                load_operand_to_register(&args[1], writer, reg_alloc, stack_slots, "a2")?;
                writeln!(writer, "    li a0, {}", i32::from(name == "write"))?;
                let callee = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                writeln!(writer, "    call {callee}")?;
            } else if name == "writebyte" && args.len() == 1 {
                // putchar reports the character written, or -1; writebyte reports the
                // byte count, so turn the result into 1 or -1.
                load_operand_to_register(&args[0], writer, reg_alloc, stack_slots, "a0")?;
                writeln!(writer, "    andi a0, a0, 255")?;
                let putchar = abi
                    .call_stub("putchar")
                    .unwrap_or_else(|| abi.mangle_function_name("putchar"));
                writeln!(writer, "    call {putchar}")?;
                writeln!(writer, "    li t1, -1")?;
                writeln!(writer, "    xor t2, a0, t1")?;
                writeln!(writer, "    seqz t2, t2")?;
                writeln!(writer, "    neg t2, t2")?;
                writeln!(writer, "    li a0, 1")?;
                writeln!(writer, "    or a0, a0, t2")?;
            } else if name == "readbyte" && args.is_empty() {
                // getchar reports the byte or -1 at end of input, where readbyte
                // reports zero.
                let getchar = abi
                    .call_stub("getchar")
                    .unwrap_or_else(|| abi.mangle_function_name("getchar"));
                writeln!(writer, "    call {getchar}")?;
                writeln!(writer, "    slti t1, a0, 0")?;
                writeln!(writer, "    addi t1, t1, -1")?;
                writeln!(writer, "    and a0, a0, t1")?;
            } else if name == "printf32" || name == "printf64" {
                if let Some(arg) = args.first() {
                    writeln!(writer, "    la a0, {}", fmt_label(name, target_os))?;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "a1")?;
                    if name == "printf32" {
                        writeln!(writer, "    fmv.w.x ft0, a1")?;
                        writeln!(writer, "    fcvt.d.s ft0, ft0")?;
                        writeln!(writer, "    fmv.x.d a1, ft0")?;
                    }
                    let printf_name = printf_symbol(&abi, target_os);
                    writeln!(writer, "    call {printf_name}")?;
                }
            } else if name == "print" || name == "printu" {
                if let Some(arg) = args.first() {
                    writeln!(writer, "    la a0, {}", fmt_label(name, target_os))?;
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "a1")?;
                    let printf_name = printf_symbol(&abi, target_os);
                    writeln!(writer, "    call {printf_name}")?;
                }
            } else {
                // General function call implementation
                // RISC-V calling convention: first 8 args in a0-a7, remaining on stack

                let types = ctx.signatures.arguments(name, args.len());
                let stack_space =
                    pass_riscv_arguments(writer, reg_alloc, stack_slots, args, &types, ctx.target)?;

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
                match ctx.signatures.float_result(name) {
                    Some(ty) => store_fp_register_to_register(
                        "fa0",
                        vreg,
                        writer,
                        reg_alloc,
                        stack_slots,
                        ty.size_bytes() == 4,
                    )?,
                    None => store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
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
            let stack_space =
                pass_riscv_arguments(writer, reg_alloc, stack_slots, args, arg_types, ctx.target)?;
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
                        ty.size_bytes() == 4,
                    )?,
                    _ => store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::TailCall { name, args } => {
            let abi = RiscVAbi::new(target_os);
            if name == "print" {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RISC-V: TailCall to print is not supported".to_string(),
                )));
            }
            if name == "writeptr" {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RISC-V: writeptr is not supported".to_string(),
                )));
            }
            let arg_regs = RiscVAbi::ARG_REGISTERS;
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());
            for i in 0..num_reg_args {
                let arg = &args[i];
                let dest_reg = arg_regs[i];
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
            }
            if num_stack_args > 0 {
                for (j, arg) in args.iter().skip(num_reg_args).enumerate() {
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                    RiscVFrame::fp_store(
                        writer,
                        ctx.target.store_word(),
                        "t0",
                        j as i32 * ctx.target.word_bytes(),
                        "t0",
                    )?;
                }
            }
            let target_sym = if let Some(stub) = abi.call_stub(name) {
                stub
            } else {
                abi.mangle_function_name(name)
            };
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
                match ctx.ret_float {
                    Some(ty) => load_fp_operand_to_register(
                        val,
                        writer,
                        reg_alloc,
                        stack_slots,
                        "fa0",
                        ty.size_bytes() == 4,
                    )?,
                    None => load_operand_to_register(val, writer, reg_alloc, stack_slots, "a0")?,
                }
            }
            // Epilogue
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
            let is_f32 = ty.size_bytes() == 4;
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
            let is_f32 = from.size_bytes() == 4;
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
            let bits = to.size_bytes() * 8;
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
            let is_f32 = to.size_bytes() == 4;
            ctx.target
                .require_float(is_f32)
                .map_err(|e| LaminaError::ValidationError(format!("inttofloat: {e}")))?;
            if ctx.target.xlen == Xlen::Rv32 {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RV32 integer to float conversion is not supported".to_string(),
                )));
            }
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "a0")?;
            let bits = from.size_bytes() * 8;
            if bits < 64 {
                if *signed {
                    sign_extend_to_xlen(writer, bits)?;
                } else {
                    writeln!(writer, "    slli a0, a0, {}", 64 - bits)?;
                    writeln!(writer, "    srli a0, a0, {}", 64 - bits)?;
                }
            }
            let suffix = if is_f32 { "s" } else { "d" };
            let form = if *signed { "l" } else { "lu" };
            writeln!(writer, "    fcvt.{suffix}.{form} fa0, a0")?;
            if let Register::Virtual(vreg) = dst {
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, is_f32)?;
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            let to_f32 = to.size_bytes() == 4;
            ctx.target
                .require_float(true)
                .map_err(|e| LaminaError::ValidationError(format!("floatconvert: {e}")))?;
            ctx.target
                .require_float(false)
                .map_err(|e| LaminaError::ValidationError(format!("floatconvert: {e}")))?;
            let from_f32 = from.size_bytes() == 4;
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
            let is_f32 = ty.size_bytes() == 4;
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
            let is_f32 = ty.size_bytes() == 4;
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
        other => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("RISC-V backend: instruction not yet supported: {other}"),
            )));
        }
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

fn copy_load_mnemonic(ty: &MirType, target: RiscVTarget) -> &'static str {
    match ty.size_bytes() {
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
) -> Result<(), LaminaError> {
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
    ) {
        return Ok(());
    }
    let bits = ty.size_bytes() as u32 * 8;
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
