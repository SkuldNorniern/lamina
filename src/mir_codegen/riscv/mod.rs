mod util;

use lamina_codegen::riscv::{RiscVAbi, RiscVFrame, RiscVRegAlloc, RiscVTarget, Xlen};
use std::io::Write;
use std::result::Result;
use util::{
    emit_int_cmp_op, load_fp_operand_to_register, load_operand_to_register,
    load_register_to_register, store_fp_register_to_register, store_register_to_register,
};

use crate::error::LaminaError;
use crate::mir::instruction::{AddressMode, Immediate};
use crate::mir::register::RegisterClass;
use crate::mir::{
    Block as MirBlock, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Instruction as MirInst,
    IntBinOp, MirType, Module as MirModule, Operand, Register, ScalarType, Signature, VirtualReg,
};
use crate::mir_codegen::common::{
    assign_stack_slots, compile_functions_parallel, parallel_codegen_error,
};
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
    capability::CapabilitySet, validate_module_call_parameters,
};

use lamina_codegen::{
    GraphColorAllocator, LinearScanAllocator, LiveInterval, LocalRegisterAllocator,
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::collections::HashMap;
use std::sync::Arc;

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
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let abi = RiscVAbi::new(target_os);

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    if settings.emit_asm_debug_lines {
        let tag = settings.debug_file_tag.replace('\"', "'");
        writeln!(output, "    .file 1 \"{tag}\"")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    let mut stack_slots: HashMap<VirtualReg, i32> = HashMap::new();
    let mut reg_alloc = RiscVRegAlloc::with_target(target_os, target);
    let mut live_across_calls = HashMap::new();

    if settings.regalloc != RegallocStrategy::Incremental {
        (stack_slots, _) =
            assign_stack_slots(func, |i| RiscVFrame::calculate_stack_offset(i, target));
        let pool = RiscVRegAlloc::gpr_pool_for_global_allocation();
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| i.vreg.class == RegisterClass::Gpr)
            .collect();
        let plan = match settings.regalloc {
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
        live_across_calls = live_vregs_across_calls(func, &intervals);
        reg_alloc = RiscVRegAlloc::from_global_plan(target_os, target, &plan, &stack_slots);
    } else {
        let mut next_slot = 0usize;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                    && !stack_slots.contains_key(vreg)
                {
                    stack_slots
                        .insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot, target));
                    next_slot += 1;
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg)
                    {
                        stack_slots
                            .insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot, target));
                        next_slot += 1;
                    }
                }
            }
        }
    }

    let stack_size = stack_slots.len() * target.word_bytes() as usize;
    let callee_saved_registers = reg_alloc.used_callee_saved_registers();
    RiscVFrame::generate_prologue(&mut output, stack_size, &callee_saved_registers, target)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    if !func.sig.params.is_empty() {
        let arg_regs = RiscVAbi::ARG_REGISTERS;
        for (index, param) in func.sig.params.iter().enumerate() {
            let Register::Virtual(vreg) = &param.reg else {
                continue;
            };
            let Some(slot_off) = stack_slots.get(vreg) else {
                continue;
            };
            if index < arg_regs.len() {
                if let Some(physical) = reg_alloc.get_mapping(vreg) {
                    writeln!(output, "    mv {physical}, {}", arg_regs[index]).map_err(|e| {
                        CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                    })?;
                } else {
                    writeln!(
                        output,
                        "    {} {}, {slot_off}(fp)",
                        target.store_word(),
                        arg_regs[index]
                    )
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                }
            } else {
                // fp is the caller's sp, so stack args start there and go up.
                let caller_off = (index - arg_regs.len()) as i32 * target.word_bytes();
                writeln!(output, "    {} t0, {caller_off}(fp)", target.load_word())
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                if let Some(physical) = reg_alloc.get_mapping(vreg) {
                    writeln!(output, "    mv {physical}, t0").map_err(|e| {
                        CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                    })?;
                } else {
                    writeln!(output, "    {} t0, {slot_off}(fp)", target.store_word()).map_err(
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
    };
    let entry_block = func.entry_block().ok_or_else(|| {
        CodegenError::InvalidFuncs(vec![format!(
            "{func_name}: entry block {:?} is not present in the function",
            func.entry
        )])
    })?;
    let mut block_start = 0;
    let mut block_starts = HashMap::new();
    for block in &func.blocks {
        block_starts.insert(block.label.as_str(), block_start);
        block_start += block.instructions.len();
    }
    let entry_start = block_starts
        .get(entry_block.label.as_str())
        .copied()
        .ok_or_else(|| {
            CodegenError::InvalidFuncs(vec![format!(
                "{func_name}: entry block position is unavailable"
            )])
        })?;
    emit_block_riscv(
        entry_block,
        entry_start,
        &mut output,
        &mut reg_alloc,
        &stack_slots,
        &mut ctx,
    )?;
    for block in &func.blocks {
        if block == entry_block {
            continue;
        }
        let start = block_starts
            .get(block.label.as_str())
            .copied()
            .ok_or_else(|| {
                CodegenError::InvalidFuncs(vec![format!(
                    "{func_name}: block '{}' position is unavailable",
                    block.label
                )])
            })?;
        emit_block_riscv(
            block,
            start,
            &mut output,
            &mut reg_alloc,
            &stack_slots,
            &mut ctx,
        )?;
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
            if matches!(instruction, MirInst::Call { .. }) {
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
                    .filter_map(instruction_type)
                    .any(&i64_type)
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
    let abi = RiscVAbi::new(target_os);

    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_print_format())?;
    writeln!(writer, "{}", abi.get_text_section())?;
    writeln!(writer, "{}", abi.get_main_global())?;

    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_riscv(name, func, target, os, settings_arc.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    Ok(())
}

/// Per-function state shared by the block and instruction emitters.
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
}

fn emit_block_riscv<W: Write>(
    block: &MirBlock,
    block_start: usize,
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    ctx: &mut EmitCtx<'_>,
) -> Result<(), CodegenError> {
    writeln!(writer, ".L_{}_{}:", ctx.func_name, block.label)
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    for (block_instruction_index, inst) in block.instructions.iter().enumerate() {
        let instruction_index = block_start + block_instruction_index;
        emit_instruction_riscv(inst, instruction_index, writer, reg_alloc, stack_slots, ctx)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
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
    let target_os = ctx.target_os;
    let settings = ctx.settings;
    let func_name = ctx.func_name;
    if settings.emit_asm_debug_lines {
        ctx.debug_line = ctx.debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", ctx.debug_line)?;
    }
    match inst {
        MirInst::Copy { ty, dst, src } => {
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
                        writeln!(writer, "    {} a0, {offset}(fp)", copy_load_mnemonic(ty))?;
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
                    writeln!(writer, "    li a0, {bits}")?;
                }
            }
            match dst {
                Register::Virtual(vreg) => {
                    if let Some(physical) = reg_alloc.get_mapping(vreg) {
                        writeln!(writer, "    mv {physical}, a0")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        writeln!(writer, "    {} a0, {offset}(fp)", copy_store_mnemonic(ty))?;
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
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

            // Perform operation
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
            ty: _,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

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
            if name == "print" {
                if let Some(arg) = args.first() {
                    // Load format string address to a0
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            writeln!(writer, "    la a0, __mir_fmt_int")?;
                        }
                        _ => {
                            writeln!(writer, "    la a0, .L_mir_fmt_int")?;
                        }
                    }
                    // Load value to print to a1
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "a1")?;
                    // Call printf
                    let printf_name = abi.call_stub("print").unwrap_or_else(|| match target_os {
                        TargetOperatingSystem::MacOS => "_printf".to_string(),
                        _ => "printf".to_string(),
                    });
                    writeln!(writer, "    call {printf_name}")?;
                }
            } else {
                // General function call implementation
                // RISC-V calling convention: first 8 args in a0-a7, remaining on stack

                let arg_regs = RiscVAbi::ARG_REGISTERS;
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                // Pass register arguments (a0-a7)
                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
                }

                // Pass stack arguments (16-byte aligned)
                let stack_space = if num_stack_args > 0 {
                    // Align to 16 bytes
                    ((num_stack_args * ctx.target.word_bytes() as usize) + 15) & !15
                } else {
                    0
                };

                if stack_space > 0 {
                    // Allocate stack space
                    writeln!(writer, "    addi sp, sp, -{stack_space}")?;

                    // Store arguments on stack (in order, starting at sp+0)
                    for (i, arg) in args.iter().skip(num_reg_args).enumerate() {
                        let offset = i * ctx.target.word_bytes() as usize;
                        // Load argument to a temporary register (use t0)
                        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                        // Store to stack
                        writeln!(writer, "    {} t0, {offset}(sp)", ctx.target.store_word())?;
                    }
                }

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
                    writeln!(writer, "    addi sp, sp, {stack_space}")?;
                }
            }

            reload_caller_saved(writer, &caller_saved_mappings, stack_slots, ctx.target)?;

            // Handle return value (always in a0)
            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::TailCall { name, args } => {
            let abi = RiscVAbi::new(target_os);
            if name == "print" {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "RISC-V: TailCall to print is not supported".to_string(),
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
                    writeln!(
                        writer,
                        "    {} t0, {}(fp)",
                        ctx.target.store_word(),
                        j as i32 * ctx.target.word_bytes()
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
                MirType::Scalar(ScalarType::I64) | MirType::Scalar(ScalarType::Ptr) => "ld",
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
                AddressMode::BaseOffset { base, offset } => {
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
                MirType::Scalar(ScalarType::I64) | MirType::Scalar(ScalarType::Ptr) => "sd",
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
                AddressMode::BaseOffset { base, offset } => {
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
                load_operand_to_register(val, writer, reg_alloc, stack_slots, "a0")?;
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
            writeln!(writer, "    j .L_{func_name}_{target}")?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_register(vreg, writer, reg_alloc, stack_slots, "t0")?;
                writeln!(writer, "    bnez t0, .L_{func_name}_{true_target}")?;
                writeln!(writer, "    j .L_{func_name}_{false_target}")?;
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
            writeln!(writer, "    bnez t0, .L_{func_name}_sel_{sel}")?;
            writeln!(writer, "    mv a0, a1")?;
            writeln!(writer, ".L_{func_name}_sel_{sel}:")?;
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
                writeln!(writer, "    beq a0, t0, .L_{func_name}_{case_label}")?;
            }
            writeln!(writer, "    j .L_{func_name}_{default}")?;
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
        MirInst::VectorOp { .. } => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                "VectorOp is not yet supported by the RISC-V backend".to_string(),
            )));
        }
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
        writeln!(
            writer,
            "    {} {physical}, {offset}(fp)",
            target.store_word()
        )?;
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
        writeln!(
            writer,
            "    {} {physical}, {offset}(fp)",
            target.load_word()
        )?;
    }
    Ok(())
}

fn copy_load_mnemonic(ty: &MirType) -> &'static str {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => "lbu",
        MirType::Scalar(ScalarType::I16) => "lhu",
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => "lwu",
        MirType::Scalar(ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr)
        | MirType::Vector(_) => "ld",
    }
}

fn copy_store_mnemonic(ty: &MirType) -> &'static str {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => "sb",
        MirType::Scalar(ScalarType::I16) => "sh",
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => "sw",
        MirType::Scalar(ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr)
        | MirType::Vector(_) => "sd",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::FunctionBuilder;
    use std::error::Error;

    #[test]
    fn rv32_uses_word_memory_ops_not_doubleword() {
        let ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let build = || {
            let mut module = MirModule::new("width_test");
            module.add_function(
                FunctionBuilder::new("addup")
                    .param(a.clone(), ty)
                    .returns(ty)
                    .block("entry")
                    .instr(MirInst::IntBinary {
                        op: IntBinOp::Add,
                        ty,
                        dst: b.clone(),
                        lhs: Operand::Register(a.clone()),
                        rhs: Operand::Immediate(Immediate::I32(1)),
                    })
                    .instr(MirInst::Ret {
                        value: Some(Operand::Register(b.clone())),
                    })
                    .build(),
            );
            module
        };

        let emit = |target| {
            let mut out = Vec::new();
            generate_mir_riscv_with_units_and_settings(
                &build(),
                &mut out,
                target,
                TargetOperatingSystem::Linux,
                1,
                &MirCodegenSettings::default(),
            )
            .expect("codegen");
            String::from_utf8(out).expect("utf8")
        };

        let rv32 = emit(RiscVTarget::general(Xlen::Rv32));
        assert!(!rv32.contains("    sd "), "rv32 emitted sd:\n{rv32}");
        assert!(!rv32.contains("    ld "), "rv32 emitted ld:\n{rv32}");
        assert!(rv32.contains("    sw "), "rv32 emitted no sw:\n{rv32}");

        let rv64 = emit(RiscVTarget::general(Xlen::Rv64));
        assert!(rv64.contains("    sd "), "rv64 lost sd:\n{rv64}");
    }

    #[test]
    fn rv32_rejects_i64_instead_of_truncating_it() {
        let ty = MirType::Scalar(ScalarType::I64);
        let dst = Register::Virtual(VirtualReg::gpr(0));
        let function = FunctionBuilder::new("wide")
            .returns(ty)
            .block("entry")
            .instr(MirInst::Copy {
                ty,
                dst: dst.clone(),
                src: Operand::Immediate(Immediate::I64(1_i64 << 40)),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();
        let mut module = MirModule::new("wide_test");
        module.add_function(function);
        let mut output = Vec::new();
        let result = generate_mir_riscv_with_units_and_settings(
            &module,
            &mut output,
            RiscVTarget::general(Xlen::Rv32),
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        );
        assert!(result.is_err(), "rv32 i64 must be rejected");
        let Err(error) = result else {
            return;
        };
        let message = error.to_string();
        assert!(
            message.contains("i64"),
            "message did not name type: {message}"
        );
        assert!(
            message.contains("rv32imafd"),
            "message did not name target: {message}"
        );
    }

    #[test]
    fn caller_saved_value_live_across_call_is_spilled_and_reloaded() -> Result<(), Box<dyn Error>> {
        let ty = MirType::Scalar(ScalarType::I64);
        let live = Register::Virtual(VirtualReg::gpr(0));
        let result = Register::Virtual(VirtualReg::gpr(1));
        let caller = FunctionBuilder::new("caller")
            .returns(ty)
            .block("entry")
            .instr(MirInst::Copy {
                ty,
                dst: live.clone(),
                src: Operand::Immediate(Immediate::I64(41)),
            })
            .instr(MirInst::Call {
                name: "callee".to_string(),
                args: Vec::new(),
                ret: None,
            })
            .instr(MirInst::IntBinary {
                op: IntBinOp::Add,
                ty,
                dst: result.clone(),
                lhs: Operand::Register(live),
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(result)),
            })
            .build();
        let callee = FunctionBuilder::new("callee")
            .block("entry")
            .instr(MirInst::Ret { value: None })
            .build();
        let mut module = MirModule::new("call_preservation");
        module.add_function(caller);
        module.add_function(callee);
        let settings = MirCodegenSettings {
            regalloc: RegallocStrategy::GraphColorGlobal,
            ..MirCodegenSettings::default()
        };
        let mut output = Vec::new();
        generate_mir_riscv_with_units_and_settings(
            &module,
            &mut output,
            RiscVTarget::general(Xlen::Rv64),
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )?;
        let assembly = String::from_utf8(output)?;
        let caller_body = assembly.split("caller:").nth(1);
        assert!(caller_body.is_some(), "missing caller body in:\n{assembly}");
        let Some(caller_body) = caller_body else {
            return Ok(());
        };
        let call_position = caller_body.find("call callee");
        assert!(
            call_position.is_some(),
            "missing call instruction in:\n{caller_body}"
        );
        let Some(call_position) = call_position else {
            return Ok(());
        };
        let before_call = &caller_body[..call_position];
        let after_call = &caller_body[call_position..];
        assert!(
            before_call.contains("sd t1,"),
            "missing spill:\n{caller_body}"
        );
        assert!(
            after_call.contains("ld t1,"),
            "missing reload:\n{caller_body}"
        );
        Ok(())
    }

    #[test]
    fn globally_allocated_callee_saved_register_is_saved_by_frame() -> Result<(), Box<dyn Error>> {
        let ty = MirType::Scalar(ScalarType::I64);
        let value = Register::Virtual(VirtualReg::gpr(0));
        let function = FunctionBuilder::new("uses_saved")
            .returns(ty)
            .block("entry")
            .instr(MirInst::Copy {
                ty,
                dst: value.clone(),
                src: Operand::Immediate(Immediate::I64(7)),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(value)),
            })
            .build();
        let mut module = MirModule::new("saved_register");
        module.add_function(function);
        let settings = MirCodegenSettings {
            regalloc: RegallocStrategy::LinearScanGlobal,
            ..MirCodegenSettings::default()
        };
        let mut output = Vec::new();
        generate_mir_riscv_with_units_and_settings(
            &module,
            &mut output,
            RiscVTarget::general(Xlen::Rv64),
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("sd s11,"), "missing save:\n{assembly}");
        assert!(assembly.contains("ld s11,"), "missing restore:\n{assembly}");
        Ok(())
    }

    #[test]
    fn base_isa_rejects_multiply_instead_of_emitting_it() {
        let ty = MirType::Scalar(ScalarType::I64);
        let dst = Register::Virtual(VirtualReg::gpr(0));
        let function = FunctionBuilder::new("times")
            .returns(ty)
            .block("entry")
            .instr(MirInst::IntBinary {
                op: IntBinOp::Mul,
                ty,
                dst: dst.clone(),
                lhs: Operand::Immediate(Immediate::I64(6)),
                rhs: Operand::Immediate(Immediate::I64(7)),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();
        let mut module = MirModule::new("ext_test");
        module.add_function(function);

        let mut out = Vec::new();
        let err = generate_mir_riscv_with_units_and_settings(
            &module,
            &mut out,
            RiscVTarget::base(Xlen::Rv64),
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )
        .expect_err("rv64i has no mul");
        let msg = err.to_string();
        assert!(msg.contains("M extension"), "unhelpful message: {msg}");
        assert!(msg.contains("rv64i"), "message should name the ISA: {msg}");

        let mut out = Vec::new();
        generate_mir_riscv_with_units_and_settings(
            &module,
            &mut out,
            RiscVTarget::general(Xlen::Rv64),
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )
        .expect("rv64g has mul");
        assert!(String::from_utf8_lossy(&out).contains("mul "));
    }

    #[test]
    fn block_labels_are_unique_across_functions() {
        // Both functions have a block called "entry".
        let ty = MirType::Scalar(ScalarType::I64);
        let mut module = MirModule::new("label_test");
        for name in ["first", "second"] {
            let dst = Register::Virtual(VirtualReg::gpr(0));
            module.add_function(
                FunctionBuilder::new(name)
                    .returns(ty)
                    .block("entry")
                    .instr(MirInst::Copy {
                        ty,
                        dst: dst.clone(),
                        src: Operand::Immediate(Immediate::I64(1)),
                    })
                    .instr(MirInst::Ret {
                        value: Some(Operand::Register(dst)),
                    })
                    .build(),
            );
        }
        let mut output = Vec::new();
        generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("RISC-V codegen should succeed");
        let asm = String::from_utf8(output).expect("assembly should be UTF-8");

        let mut labels: Vec<&str> = asm
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with(".L_") && l.ends_with(':'))
            .collect();
        let before = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(before, labels.len(), "duplicate labels in:\n{asm}");
    }

    #[test]
    fn incoming_arguments_are_stored_into_their_slots() {
        let ty = MirType::Scalar(ScalarType::I64);
        let p0 = Register::Virtual(VirtualReg::gpr(0));
        let p1 = Register::Virtual(VirtualReg::gpr(1));
        let sum = Register::Virtual(VirtualReg::gpr(2));
        let function = FunctionBuilder::new("add_two")
            .param(p0.clone(), ty)
            .param(p1.clone(), ty)
            .returns(ty)
            .block("entry")
            .instr(MirInst::IntBinary {
                op: IntBinOp::Add,
                ty,
                dst: sum.clone(),
                lhs: Operand::Register(p0),
                rhs: Operand::Register(p1),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(sum)),
            })
            .build();
        let mut module = MirModule::new("param_test");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("RISC-V codegen should succeed");
        let asm = String::from_utf8(output).expect("assembly should be UTF-8");

        let body = asm.split("add_two:").nth(1).expect("function label");
        let prologue_end = body.find(".L_").unwrap_or(body.len());
        let prologue = &body[..prologue_end];
        assert!(
            prologue.contains("sd a0,") && prologue.contains("sd a1,"),
            "arguments were never stored, prologue was:\n{prologue}"
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
            .instr(MirInst::Copy {
                ty,
                dst: dst.clone(),
                src: Operand::Register(src),
            })
            .instr(MirInst::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();
        let mut module = MirModule::new("copy_test");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_riscv(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("RISC-V codegen should succeed");
        let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

        assert!(assembly.contains("ld a0"), "expected bit load: {assembly}");
        assert!(assembly.contains("sd a0"), "expected bit store: {assembly}");
        assert!(
            !assembly.contains("fadd"),
            "unexpected float add: {assembly}"
        );
    }
}
