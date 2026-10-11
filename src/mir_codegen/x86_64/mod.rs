//! x86_64 code generation for MIR (Mid-level IR).
//!
//! Code generation from MIR to x86_64 assembly,
//! supporting System V AMD64 and Microsoft x64 calling conventions.

pub mod constants;
pub mod util;

use constants::{fd, linux, macos, stack, windows};
use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame, extend_integer, register_for_size};
use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    io::{self, Error, Write},
};
use util::*;

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Instruction as MirInst,
        IntBinOp, IntCmpOp, MirType, Module as MirModule, Operand, Register, Signature, VirtualReg,
        instruction::Immediate, register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::CapabilitySet, validate_module_call_parameters,
    },
};

use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator, LiveInterval,
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::{collections::hash_map::Entry, sync::Arc};

use crate::mir_codegen::common::{
    CodegenBase, assign_stack_slots, assign_stack_slots_where, compile_functions_parallel,
    emit_print_format_section, parallel_codegen_error,
};

/// MIR to x86_64 code generator implementing the `Codegen` trait.
pub struct X86Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> X86Codegen<'a> {
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
        generate_mir_x86_64_with_units(module, writer, self.base.target_os, codegen_units)
    }
}

impl<'a> Codegen for X86Codegen<'a> {
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
                generate_mir_x86_64_with_units(module, writer, target_os, codegen_units)
            },
            "x86_64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not supported".to_string(),
        ))
    }
}

pub struct RaxTrackingWriter<W> {
    inner: W,
    rax_value: Option<VirtualReg>,
}

impl<W: Write> Write for RaxTrackingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.rax_value = None;
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// True for `movq %r, %r`. Several lowering paths pick the same register for source and
/// destination, and the move that results does nothing.
fn is_self_move(line: &str) -> bool {
    let Some(operands) = line.trim_start().strip_prefix("movq ") else {
        return false;
    };
    matches!(operands.split_once(", "), Some((src, dst)) if src.starts_with('%') && src == dst)
}

/// Drop `movq %r, %r` from finished assembly.
fn strip_self_moves(asm: Vec<u8>) -> Vec<u8> {
    let Ok(text) = String::from_utf8(asm) else {
        return Vec::new();
    };
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !is_self_move(line.trim_end_matches('\n')) {
            out.push_str(line);
        }
    }
    out.into_bytes()
}

fn callee_saved(reg: &str, target_os: TargetOperatingSystem) -> bool {
    matches!(reg, "rbx" | "r12" | "r13" | "r14" | "r15")
        || (target_os == TargetOperatingSystem::Windows && matches!(reg, "rsi" | "rdi"))
}

fn instruction_clobbers(
    inst: &MirInst,
    leaf: bool,
    target_os: TargetOperatingSystem,
    func_name: &str,
) -> Vec<&'static str> {
    let scratch = if leaf {
        vec!["rcx", "rdx"]
    } else {
        vec!["r10", "r11"]
    };
    match inst {
        MirInst::TailCall { name, args } if name == func_name => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            arg_regs[..args.len().min(arg_regs.len())].to_vec()
        }
        MirInst::Call { .. } | MirInst::TailCall { .. } => {
            let mut regs = vec!["rcx", "rdx", "r8", "r9", "r10", "r11"];
            if target_os != TargetOperatingSystem::Windows {
                regs.extend(["rsi", "rdi"]);
            }
            regs
        }
        MirInst::IntBinary { op, rhs, .. } => match op {
            IntBinOp::SDiv | IntBinOp::UDiv | IntBinOp::SRem | IntBinOp::URem => {
                vec!["rcx", "rdx"]
            }
            IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr
                if matches!(rhs, Operand::Register(_)) =>
            {
                let mut regs = scratch;
                regs.push("rcx");
                regs
            }
            _ => scratch,
        },
        MirInst::IntCmp { .. }
        | MirInst::Store { .. }
        | MirInst::Lea { .. }
        | MirInst::Select { .. } => scratch,
        _ => Vec::new(),
    }
}

fn layout_global_stack_slots(
    func: &Function,
    plan: &HashMap<VirtualReg, MirAllocation<&'static str>>,
) -> (HashMap<VirtualReg, i32>, BTreeSet<VirtualReg>, usize) {
    let mut address_taken: BTreeSet<_> = func.stack_reservations.keys().copied().collect();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        if let MirInst::StackAddr { slot, .. } = inst {
            address_taken.insert(*slot);
        }
    }
    let (mut slots, defs, mut total) =
        assign_stack_slots_where(func, X86Frame::calculate_stack_offset, |reg| {
            address_taken.contains(reg)
                || !matches!(plan.get(reg), Some(MirAllocation::Register(_)))
        });
    for reg in address_taken {
        if let Entry::Vacant(entry) = slots.entry(reg) {
            let extra = func.stack_reservations.get(&reg).copied().unwrap_or(0);
            entry.insert(X86Frame::calculate_stack_offset(total + extra));
            total += extra + 1;
        }
    }
    (slots, defs, total)
}

fn compile_single_function_x86_64(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = RaxTrackingWriter {
        inner: Vec::new(),
        rax_value: None,
    };
    let abi = X86ABI::new(target_os);

    ensure_signature_support(&func.sig)
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    if settings.emit_asm_debug_lines {
        let tag = settings.debug_file_tag.replace('\"', "'");
        writeln!(output, "    .file 1 \"{tag}\"")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    // A self tail call lowers to a jump back past the prologue, so it neither needs a
    // frame nor clobbers anything a real call would.
    let is_leaf = !func.blocks.iter().any(|b| {
        b.instructions.iter().any(|i| match i {
            MirInst::Call { .. } => true,
            MirInst::TailCall { name, .. } => name != func_name,
            _ => false,
        })
    });

    let mut reg_alloc = if is_leaf {
        X64RegAlloc::new_leaf(target_os)
    } else {
        X64RegAlloc::new(target_os)
    };

    let (stack_slots, def_regs, total_slots);

    let mut saved_regs = Vec::new();
    if settings.regalloc != RegallocStrategy::Incremental {
        let pool = X64RegAlloc::gpr_pool_for_global_allocation(target_os, is_leaf);
        let clobbers: Vec<_> = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| {
                let mut regs = instruction_clobbers(inst, is_leaf, target_os, func_name);
                for reg in inst.use_regs().into_iter().chain(inst.def_reg()) {
                    if let Register::Physical(phys) = reg {
                        regs.push(phys.name);
                    }
                }
                regs
            })
            .collect();
        let allowed = |interval: &LiveInterval, reg| {
            clobbers[interval.start..=interval.end]
                .iter()
                .all(|regs| !regs.contains(&reg))
        };
        let mut address_taken: BTreeSet<_> = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match inst {
                MirInst::StackAddr { slot, .. } => Some(*slot),
                _ => None,
            })
            .collect();
        address_taken.extend(func.stack_reservations.keys());
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| i.vreg.class == RegisterClass::Gpr && !address_taken.contains(&i.vreg))
            .collect();
        let plan = match settings.regalloc {
            RegallocStrategy::LinearScanGlobal => {
                LinearScanAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::GraphColorGlobal => {
                GraphColorAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::Incremental => unreachable!(),
        };
        (stack_slots, def_regs, total_slots) = layout_global_stack_slots(func, &plan);
        reg_alloc = X64RegAlloc::from_global_plan(target_os, is_leaf, &plan);
        for allocation in plan.values() {
            if let MirAllocation::Register(reg) = allocation
                && callee_saved(reg, target_os)
            {
                saved_regs.push(*reg);
            }
        }
        saved_regs.sort_unstable();
        saved_regs.dedup();
    } else {
        (stack_slots, def_regs, total_slots) =
            assign_stack_slots(func, X86Frame::calculate_stack_offset);
    }

    let stack_size = (total_slots + saved_regs.len()) * 8;
    let has_frame = settings.regalloc == RegallocStrategy::Incremental
        || !is_leaf
        || stack_size != 0
        || func.sig.params.len() > abi.arg_registers().len();
    if has_frame {
        X86Frame::generate_prologue(&mut output, stack_size)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    for (index, reg) in saved_regs.iter().enumerate() {
        let offset = X86Frame::calculate_stack_offset(total_slots + index);
        writeln!(output, "    movq %{reg}, {offset}(%rbp)")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    // A self tail call re-enters here with its arguments already in the ABI argument
    // registers, so it reuses the parameter homing below and skips frame teardown and
    // setup entirely.
    writeln!(output, "{}:", self_tail_entry_label(func_name))
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    if !func.sig.params.is_empty() {
        let abi_for_func = X86ABI::new(target_os);
        let arg_regs = abi_for_func.arg_registers();

        if settings.regalloc == RegallocStrategy::Incremental {
            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        let phys_arg = arg_regs[index];
                        let destination = reg_alloc
                            .get_mapping_for(vreg)
                            .map(|reg| format!("%{reg}"))
                            .unwrap_or_else(|| format!("{slot_off}(%rbp)"));
                        let source = format!("%{phys_arg}");
                        if source == destination {
                            continue;
                        }
                        writeln!(output, "    movq {source}, {destination}").map_err(|e| {
                            CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                        })?;
                    } else {
                        let stack_index = index - arg_regs.len();
                        // Saved rbp and the return address sit at rbp+0 and rbp+8;
                        // on Windows the caller's home space follows before the
                        // first stack argument.
                        let home = if target_os == TargetOperatingSystem::Windows {
                            windows::SHADOW_SPACE_SIZE
                        } else {
                            0
                        };
                        let caller_off = 16 + home + (stack_index as i32) * 8;
                        writeln!(output, "    movq {caller_off}(%rbp), %rax").map_err(|e| {
                            CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                        })?;
                        store_rax_to_register(vreg, &mut output, &reg_alloc, &stack_slots)
                            .map_err(|e| {
                                CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                            })?;
                    }
                }
            }
        } else {
            let pairs: Vec<_> = func
                .sig
                .params
                .iter()
                .enumerate()
                .filter_map(|(index, param)| {
                    let Register::Virtual(vreg) = &param.reg else {
                        return None;
                    };
                    let destination = reg_alloc
                        .get_mapping_for(vreg)
                        .map(|reg| format!("%{reg}"))
                        .or_else(|| {
                            stack_slots
                                .get(vreg)
                                .map(|offset| format!("{offset}(%rbp)"))
                        })?;
                    let source = if let Some(reg) = arg_regs.get(index) {
                        format!("%{reg}")
                    } else {
                        let home = if target_os == TargetOperatingSystem::Windows {
                            windows::SHADOW_SPACE_SIZE
                        } else {
                            0
                        };
                        let offset = 16 + home + ((index - arg_regs.len()) as i32) * 8;
                        format!("{offset}(%rbp)")
                    };
                    Some((source, destination))
                })
                .collect();
            emit_parallel_copy(&pairs, &mut output)
                .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
    }

    // Where this function's own parameters live, so a self tail call can write straight
    // into them instead of staging through the ABI argument registers.
    let self_param_homes: Vec<Option<String>> = func
        .sig
        .params
        .iter()
        .map(|param| match &param.reg {
            Register::Virtual(vreg) => reg_alloc
                .get_mapping_for(vreg)
                .map(|reg| format!("%{reg}"))
                .or_else(|| stack_slots.get(vreg).map(|off| format!("{off}(%rbp)"))),
            Register::Physical(reg) => Some(format!("%{reg}")),
        })
        .collect();

    let layout = block_layout(func);
    if layout.first().map(|&i| func.blocks[i].label.as_str()) != Some(func.entry.as_str()) {
        writeln!(output, "    jmp .L_{}.{}", func_name, func.entry)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    // A comparison whose only consumer is the branch right after it needs no setCC and no
    // materialised 0/1: the flags from cmpq drive jCC directly.
    let mut use_counts: HashMap<VirtualReg, usize> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            for reg in inst.use_regs() {
                if let Register::Virtual(vreg) = reg {
                    *use_counts.entry(*vreg).or_insert(0) += 1;
                }
            }
        }
    }

    // A block reached from later in the layout is a loop header. Aligning one keeps a
    // large loop off a bad decode boundary, but a short loop already fits the fetch
    // window and only pays for the padding, so only loops past the threshold get it.
    const ALIGN_LOOP_MIN_INSTRUCTIONS: usize = 24;
    let loop_headers: HashSet<usize> = layout
        .iter()
        .enumerate()
        .filter(|&(slot, &index)| {
            let label = func.blocks[index].label.as_str();
            let Some(last) = layout[slot + 1..]
                .iter()
                .rposition(|&later| func.blocks[later].successors().contains(&label))
            else {
                return false;
            };
            let body: usize = layout[slot..=slot + 1 + last]
                .iter()
                .map(|&b| func.blocks[b].instructions.len())
                .sum();
            body >= ALIGN_LOOP_MIN_INSTRUCTIONS
        })
        .map(|(_, &index)| index)
        .collect();

    let mut debug_line: u32 = 0;
    for (slot, &block_index) in layout.iter().enumerate() {
        let block = &func.blocks[block_index];
        let next_label = layout
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        output.rax_value = None;
        if loop_headers.contains(&block_index) {
            writeln!(output, "    .p2align 4,,10")
                .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
        }
        writeln!(output, ".L_{}.{}:", func_name, block.label)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

        let mut skip_next = false;
        for (position, inst) in block.instructions.iter().enumerate() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if let MirInst::IntCmp {
                op,
                dst,
                lhs,
                rhs,
                ty,
            } = inst
                && let Register::Virtual(cmp_dst) = dst
                && let Some(MirInst::Br {
                    cond: Register::Virtual(cond),
                    true_target,
                    false_target,
                }) = block.instructions.get(position + 1)
                && cond == cmp_dst
                && use_counts.get(cmp_dst).copied() == Some(1)
            {
                let scratch =
                    emit_int_compare(op, ty, lhs, rhs, &mut output, &mut reg_alloc, &stack_slots)
                        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                let suffix = cmp_suffix(op)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                if true_target == next_label {
                    let inverted = inverted_cmp_suffix(op)
                        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                    writeln!(output, "    j{inverted} .L_{func_name}.{false_target}").map_err(
                        |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                    )?;
                } else {
                    writeln!(output, "    j{suffix} .L_{func_name}.{true_target}").map_err(
                        |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                    )?;
                    if false_target != next_label {
                        writeln!(output, "    jmp .L_{func_name}.{false_target}").map_err(|e| {
                            CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                        })?;
                    }
                }
                if !scratch.is_empty() && scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
                output.rax_value = None;
                skip_next = true;
                continue;
            }
            emit_instruction_x86_64(
                inst,
                &mut output,
                &mut reg_alloc,
                &stack_slots,
                stack_size,
                has_frame,
                &saved_regs,
                target_os,
                func_name,
                &def_regs,
                settings,
                &mut debug_line,
                next_label,
                &self_param_homes,
                &func.entry,
            )
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
        output.rax_value = None;
    }

    Ok(strip_self_moves(output.inner))
}

pub fn generate_mir_x86_64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_x86_64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_x86_64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_x86_64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_x86_64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::X86_64)?;
    let abi = X86ABI::new(target_os);

    emit_print_format_section(writer, target_os)?;

    // Emit external function declarations first
    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }

    writeln!(writer, ".text")?;
    writeln!(writer, "{}", abi.get_main_global())?;

    let settings_arc = Arc::new(settings.clone());
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_x86_64(name, func, os, settings_arc.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    Ok(())
}

fn ensure_signature_support(sig: &Signature) -> Result<(), LaminaError> {
    for (idx, param) in sig.params.iter().enumerate() {
        ensure_type_supported(&param.ty, &format!("parameter {} of '{}'", idx, sig.name))?;
    }

    if let Some(ret_ty) = &sig.ret_ty {
        ensure_type_supported(ret_ty, &format!("return type of '{}'", sig.name))?;
    }

    Ok(())
}

fn ensure_type_supported(ty: &MirType, context: &str) -> Result<(), LaminaError> {
    if ty.is_float() || ty.is_vector() {
        return Err(LaminaError::ValidationError(format!(
            "x86_64 backend does not support {context} (type {ty})"
        )));
    }
    Ok(())
}

fn restore_saved_registers<W: Write>(
    writer: &mut W,
    stack_size: usize,
    saved_regs: &[&str],
) -> Result<(), Error> {
    let first_slot = stack_size / 8 - saved_regs.len();
    for (index, reg) in saved_regs.iter().enumerate() {
        let offset = X86Frame::calculate_stack_offset(first_slot + index);
        writeln!(writer, "    movq {offset}(%rbp), %{reg}")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_instruction_x86_64(
    inst: &MirInst,
    writer: &mut RaxTrackingWriter<impl Write>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    stack_size: usize,
    has_frame: bool,
    saved_regs: &[&str],
    target_os: TargetOperatingSystem,
    func_name: &str,
    def_regs: &BTreeSet<VirtualReg>,
    settings: &MirCodegenSettings,
    debug_line: &mut u32,
    next_label: &str,
    self_param_homes: &[Option<String>],
    entry_label: &str,
) -> Result<(), LaminaError> {
    if settings.emit_asm_debug_lines {
        *debug_line = debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", *debug_line)?;
    }
    if let MirInst::IntConvert {
        from,
        to,
        signed,
        dst,
        src,
    } = inst
    {
        load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
        extend_integer(
            writer,
            "rax",
            from.size_bytes().min(to.size_bytes()),
            *signed,
        )?;
        if to.size_bytes() < 8 {
            extend_integer(writer, "rax", to.size_bytes(), true)?;
        }
        match dst {
            Register::Virtual(vreg) => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
            Register::Physical(physical) => writeln!(writer, "    movq %rax, %{}", physical.name)?,
        }
        return Ok(());
    }
    match inst {
        MirInst::Copy { ty, dst, src } => {
            if matches!(ty, MirType::Vector(_)) {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support vector Copy of type {ty}"
                )));
            }
            load_copy_operand_to_rax(ty, src, writer, reg_alloc, stack_slots)?;
            match dst {
                Register::Virtual(vreg) => {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(phys) if phys.name != "rax" => {
                    writeln!(writer, "    movq %rax, %{}", phys.name)?;
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
            // Get dst physical register if available
            let dst_phys = match dst {
                Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                Register::Physical(p) => Some(p.name),
            };

            if let Some(dp) = dst_phys {
                // Optimized path: emit directly into dp, avoiding rax bounce
                match op {
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr => {
                        // Load lhs into dp
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, dp)?;
                        if matches!(op, IntBinOp::LShr) {
                            extend_integer(writer, dp, ty.size_bytes(), false)?;
                        }
                        match rhs {
                            Operand::Immediate(imm) => {
                                let sv: u64 = match imm {
                                    Immediate::I8(v) => *v as u64,
                                    Immediate::I16(v) => *v as u64,
                                    Immediate::I32(v) => *v as u64,
                                    Immediate::I64(v) => *v as u64,
                                    _ => {
                                        return Err(LaminaError::ValidationError(
                                            "Shift count must be integer".to_string(),
                                        ));
                                    }
                                };
                                match op {
                                    IntBinOp::Shl => writeln!(writer, "    shlq ${sv}, %{dp}")?,
                                    IntBinOp::AShr => writeln!(writer, "    sarq ${sv}, %{dp}")?,
                                    IntBinOp::LShr => writeln!(writer, "    shrq ${sv}, %{dp}")?,
                                    _ => unreachable!(),
                                }
                            }
                            Operand::Register(_) => {
                                // Need shift count in %cl; load rhs into rcx
                                load_operand_to_register(
                                    rhs,
                                    writer,
                                    reg_alloc,
                                    stack_slots,
                                    "rcx",
                                )?;
                                match op {
                                    IntBinOp::Shl => writeln!(writer, "    shlq %cl, %{dp}")?,
                                    IntBinOp::AShr => writeln!(writer, "    sarq %cl, %{dp}")?,
                                    IntBinOp::LShr => writeln!(writer, "    shrq %cl, %{dp}")?,
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                    IntBinOp::SDiv | IntBinOp::UDiv | IntBinOp::SRem | IntBinOp::URem => {
                        // Division always goes through rax/rdx; use rcx for divisor to avoid conflicts
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "rax")?;
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
                            extend_integer(writer, "rax", ty.size_bytes(), false)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), false)?;
                        }
                        match op {
                            IntBinOp::SDiv => {
                                writeln!(writer, "    cqto")?;
                                writeln!(writer, "    idivq %rcx")?;
                            }
                            IntBinOp::UDiv => {
                                writeln!(writer, "    xorq %rdx, %rdx")?;
                                writeln!(writer, "    divq %rcx")?;
                            }
                            IntBinOp::SRem => {
                                writeln!(writer, "    cqto")?;
                                writeln!(writer, "    idivq %rcx")?;
                                writeln!(writer, "    movq %rdx, %rax")?;
                            }
                            IntBinOp::URem => {
                                writeln!(writer, "    xorq %rdx, %rdx")?;
                                writeln!(writer, "    divq %rcx")?;
                                writeln!(writer, "    movq %rdx, %rax")?;
                            }
                            _ => unreachable!(),
                        }
                        // Store result (rax) into dp
                        if dp != "rax" {
                            writeln!(writer, "    movq %rax, %{dp}")?;
                        }
                    }
                    _ => {
                        // Arithmetic/bitwise: load lhs into dp, then apply op with rhs
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, dp)?;

                        // Try to use rhs directly (register or small immediate)
                        match rhs {
                            Operand::Immediate(imm) => {
                                let v: i64 = match imm {
                                    Immediate::I8(x) => *x as i64,
                                    Immediate::I16(x) => *x as i64,
                                    Immediate::I32(x) => *x as i64,
                                    Immediate::I64(x) => *x,
                                    _ => {
                                        return Err(LaminaError::ValidationError(
                                            "Float immediate in integer op".to_string(),
                                        ));
                                    }
                                };
                                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                                    // Fits in 32-bit sign-extended immediate
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq ${v}, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq ${v}, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq ${v}, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  ${v}, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq ${v}, %{dp}")?,
                                        IntBinOp::Mul => {
                                            writeln!(writer, "    imulq ${v}, %{dp}, %{dp}")?
                                        }
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // Large immediate: load into rax as temp, then op
                                    writeln!(writer, "    movq ${v}, %rax")?;
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %rax, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %rax, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq %rax, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %rax, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %rax, %{dp}")?,
                                        IntBinOp::Mul => writeln!(writer, "    imulq %rax, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                }
                            }
                            Operand::Register(rhs_reg) => {
                                // Get rhs physical register if available for direct register op
                                let rhs_phys = match rhs_reg {
                                    Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                                    Register::Physical(p) => Some(p.name),
                                };
                                if let Some(rp) = rhs_phys {
                                    // Both in registers: direct op, no memory access
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %{rp}, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %{rp}, %{dp}")?,
                                        IntBinOp::Mul => {
                                            writeln!(writer, "    imulq %{rp}, %{dp}")?
                                        }
                                        IntBinOp::And => writeln!(writer, "    andq %{rp}, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %{rp}, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %{rp}, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // rhs in stack: load to rax as temp, then op
                                    load_operand_to_register(
                                        rhs,
                                        writer,
                                        reg_alloc,
                                        stack_slots,
                                        "rax",
                                    )?;
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %rax, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %rax, %{dp}")?,
                                        IntBinOp::Mul => writeln!(writer, "    imulq %rax, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq %rax, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %rax, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %rax, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        }
                        // No store needed — result already in dp!
                    }
                }
                extend_integer(writer, dp, ty.size_bytes(), true)?;
            } else {
                // No physical register for dst — fall back to rax accumulator path
                load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
                if matches!(op, IntBinOp::LShr) {
                    extend_integer(writer, "rax", ty.size_bytes(), false)?;
                }

                match op {
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr => match rhs {
                        Operand::Immediate(imm) => {
                            let shift_val = match imm {
                                Immediate::I8(v) => *v as u64,
                                Immediate::I16(v) => *v as u64,
                                Immediate::I32(v) => *v as u64,
                                Immediate::I64(v) => *v as u64,
                                _ => {
                                    return Err(LaminaError::ValidationError(
                                        "Shift count must be an integer immediate".to_string(),
                                    ));
                                }
                            };
                            match op {
                                IntBinOp::Shl => writeln!(writer, "    shlq ${shift_val}, %rax")?,
                                IntBinOp::AShr => writeln!(writer, "    sarq ${shift_val}, %rax")?,
                                IntBinOp::LShr => writeln!(writer, "    shrq ${shift_val}, %rax")?,
                                _ => unreachable!(),
                            }
                        }
                        Operand::Register(_) => {
                            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
                            writeln!(writer, "    movq %{scratch}, %rcx")?;
                            match op {
                                IntBinOp::Shl => writeln!(writer, "    shlq %cl, %rax")?,
                                IntBinOp::AShr => writeln!(writer, "    sarq %cl, %rax")?,
                                IntBinOp::LShr => writeln!(writer, "    shrq %cl, %rax")?,
                                _ => unreachable!(),
                            }
                            if scratch != "rbx" {
                                reg_alloc.free_scratch(scratch);
                            }
                        }
                    },
                    IntBinOp::SDiv | IntBinOp::UDiv | IntBinOp::SRem | IntBinOp::URem => {
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
                            extend_integer(writer, "rax", ty.size_bytes(), false)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), false)?;
                        }
                        match op {
                            IntBinOp::SDiv => {
                                writeln!(writer, "    cqto")?;
                                writeln!(writer, "    idivq %rcx")?;
                            }
                            IntBinOp::UDiv => {
                                writeln!(writer, "    xorq %rdx, %rdx")?;
                                writeln!(writer, "    divq %rcx")?;
                            }
                            IntBinOp::SRem => {
                                writeln!(writer, "    cqto")?;
                                writeln!(writer, "    idivq %rcx")?;
                                writeln!(writer, "    movq %rdx, %rax")?;
                            }
                            IntBinOp::URem => {
                                writeln!(writer, "    xorq %rdx, %rdx")?;
                                writeln!(writer, "    divq %rcx")?;
                                writeln!(writer, "    movq %rdx, %rax")?;
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;

                        match op {
                            IntBinOp::Add => writeln!(writer, "    addq %{scratch}, %rax")?,
                            IntBinOp::Sub => writeln!(writer, "    subq %{scratch}, %rax")?,
                            IntBinOp::Mul => writeln!(writer, "    imulq %{scratch}, %rax")?,
                            IntBinOp::And => writeln!(writer, "    andq %{scratch}, %rax")?,
                            IntBinOp::Or => writeln!(writer, "    orq  %{scratch}, %rax")?,
                            IntBinOp::Xor => writeln!(writer, "    xorq %{scratch}, %rax")?,
                            _ => unreachable!(),
                        }

                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    }
                }

                extend_integer(writer, "rax", ty.size_bytes(), true)?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let scratch = emit_int_compare(op, ty, lhs, rhs, writer, reg_alloc, stack_slots)?;
            let suffix = cmp_suffix(op)?;
            writeln!(writer, "    set{suffix} %al")?;
            writeln!(writer, "    movzbq %al, %rax")?;

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }

            if !scratch.is_empty() && scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "print" {
                if let Some(arg) = args.first() {
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            // al carries the vector-register count for variadic
                            // calls; printf reads it before saving xmm state.
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call _printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call _fflush")?;
                        }
                        TargetOperatingSystem::Windows => {
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rcx")?;
                            writeln!(writer, "    movq %rax, %rdx")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    movq $0, %rcx")?;
                            writeln!(writer, "    call fflush")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        }
                        _ => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call fflush")?;
                        }
                    }
                }
            } else if name == "writebyte" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writebyte requires one argument".to_string())
                })?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movb %al, (%rsp)")?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
                        // Windows syscall numbers are unstable and cannot be invoked
                        // directly like on Linux/macOS. Route the byte through the CRT
                        // (putchar) instead, matching how `print` uses `printf`.
                        writeln!(writer, "    movzbl (%rsp), %ecx")?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call putchar")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        // putchar yields the character written; the syscall
                        // paths yield the byte count, so report 1 or -1.
                        writeln!(writer, "    movl %eax, %ecx")?;
                        writeln!(writer, "    movl $1, %eax")?;
                        writeln!(writer, "    movl $-1, %edx")?;
                        writeln!(writer, "    cmpl $-1, %ecx")?;
                        writeln!(writer, "    cmovel %edx, %eax")?;
                        writeln!(writer, "    cltq")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;
            } else if name == "readbyte" && args.is_empty() {
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call getchar")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        // getchar reports EOF as -1; the syscall paths report 0.
                        writeln!(writer, "    movslq %eax, %rax")?;
                        writeln!(writer, "    xorl %ecx, %ecx")?;
                        writeln!(writer, "    cmpq $-1, %rax")?;
                        writeln!(writer, "    cmoveq %rcx, %rax")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                // getchar already returns the byte (or -1); the syscall returns a
                // count, so swap in the byte it stored only on a full read.
                if target_os != TargetOperatingSystem::Windows {
                    writeln!(writer, "    movzbl (%rsp), %ecx")?;
                    writeln!(writer, "    cmpq $1, %rax")?;
                    writeln!(writer, "    cmovel %ecx, %eax")?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else if name == "writeptr" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writeptr requires one argument".to_string())
                })?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movzbl (%rax), %ecx")?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                writeln!(writer, "    movb %cl, (%rsp)")?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
                        writeln!(writer, "    movzbl (%rsp), %ecx")?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call putchar")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        // putchar yields the character written; the syscall
                        // paths yield the byte count, so report 1 or -1.
                        writeln!(writer, "    movl %eax, %ecx")?;
                        writeln!(writer, "    movl $1, %eax")?;
                        writeln!(writer, "    movl $-1, %edx")?;
                        writeln!(writer, "    cmpl $-1, %ecx")?;
                        writeln!(writer, "    cmovel %edx, %eax")?;
                        writeln!(writer, "    cltq")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    // Rounded up so rsp stays 16-aligned at the call; the
                    // padding lands above the arguments, leaving their offsets
                    // from rsp unchanged.
                    let total_stack = (shadow_space + (num_stack_args * stack::SLOT_SIZE))
                        .div_ceil(stack::ALIGNMENT)
                        * stack::ALIGNMENT;
                    if total_stack > 0 {
                        writeln!(writer, "    subq ${total_stack}, %rsp")?;
                    }
                    for i in 0..num_stack_args {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        let stack_offset = shadow_space + (i * stack::SLOT_SIZE);
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                    }
                } else {
                    // An odd number of pushes would leave rsp 8 off at the call.
                    if num_stack_args % 2 == 1 {
                        writeln!(writer, "    subq $8, %rsp")?;
                    }
                    for i in (0..num_stack_args).rev() {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    pushq %rax")?;
                    }
                }

                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
                }

                let mangled_name = abi.mangle_function_name(name);
                writeln!(writer, "    call {mangled_name}")?;

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    let total_stack = (shadow_space + (num_stack_args * stack::SLOT_SIZE))
                        .div_ceil(stack::ALIGNMENT)
                        * stack::ALIGNMENT;
                    if total_stack > 0 {
                        writeln!(writer, "    addq ${total_stack}, %rsp")?;
                    }
                } else if num_stack_args > 0 {
                    let pushed = num_stack_args * stack::SLOT_SIZE;
                    let padded = pushed.div_ceil(stack::ALIGNMENT) * stack::ALIGNMENT;
                    writeln!(writer, "    addq ${padded}, %rsp")?;
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            if let AddressMode::BaseOffset {
                base, offset: 0, ..
            } = addr
            {
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    }
                }
                writeln!(writer, "    {}", load_insn_for(ty))?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support load address mode {addr:?}"
                )));
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs: _,
        } => {
            // Simple direct store for now
            load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
            if let AddressMode::BaseOffset {
                base, offset: 0, ..
            } = addr
            {
                let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_register(vreg, writer, reg_alloc, stack_slots, scratch)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %{}", phys.name, scratch)?;
                    }
                }
                writeln!(
                    writer,
                    "    mov{} %{}, (%{scratch})",
                    size_suffix(ty),
                    rax_for(ty)
                )?;
                if scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support store address mode {addr:?}"
                )));
            }
        }
        MirInst::TailCall { name, args } if name == func_name => {
            // Self tail call: move the arguments straight into the parameter homes as a
            // parallel copy and jump to the entry block, skipping the ABI registers, the
            // prologue and the epilogue entirely.
            let direct = args.len() == self_param_homes.len()
                && self_param_homes.iter().all(|home| home.is_some())
                && args.iter().all(|arg| match arg {
                    Operand::Register(Register::Virtual(vreg)) => {
                        reg_alloc.get_mapping_for(vreg).is_some() || stack_slots.contains_key(vreg)
                    }
                    Operand::Register(Register::Physical(_)) => true,
                    Operand::Immediate(Immediate::I8(_) | Immediate::I16(_))
                    | Operand::Immediate(Immediate::I32(_) | Immediate::I64(_)) => true,
                    _ => false,
                });
            if direct {
                let mut pairs = Vec::with_capacity(args.len());
                for (arg, home) in args.iter().zip(self_param_homes) {
                    let source = match arg {
                        Operand::Register(Register::Virtual(vreg)) => reg_alloc
                            .get_mapping_for(vreg)
                            .map(|reg| format!("%{reg}"))
                            .unwrap_or_else(|| format!("{}(%rbp)", stack_slots[vreg])),
                        Operand::Register(Register::Physical(reg)) => format!("%{reg}"),
                        Operand::Immediate(imm) => {
                            let value = match imm {
                                Immediate::I8(v) => i64::from(*v),
                                Immediate::I16(v) => i64::from(*v),
                                Immediate::I32(v) => i64::from(*v),
                                Immediate::I64(v) => *v,
                                other => {
                                    return Err(LaminaError::ValidationError(format!(
                                        "self tail call argument {other:?} is not an integer"
                                    )));
                                }
                            };
                            format!("${value}")
                        }
                    };
                    pairs.push((source, home.clone().unwrap_or_default()));
                }
                emit_parallel_copy(&pairs, writer)?;
                if entry_label != next_label {
                    writeln!(writer, "    jmp .L_{func_name}.{entry_label}")?;
                }
                return Ok(());
            }
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            for (index, arg) in args.iter().enumerate().take(arg_regs.len()) {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[index])?;
            }
            writeln!(writer, "    jmp {}", self_tail_entry_label(func_name))?;
        }
        MirInst::TailCall { name, args } => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());

            // 1. Handle Register Arguments
            for i in 0..num_reg_args {
                let arg = &args[i];
                let dest_reg = arg_regs[i];
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
            }

            // 2. Handle Stack Arguments (Overwrite incoming args)
            // Note: TailCallOptimization guarantees args.len() == current_func.args.len()
            // So we can strictly overwrite our own incoming stack slots.
            if num_stack_args > 0 {
                let shadow_space = if target_os == TargetOperatingSystem::Windows {
                    windows::SHADOW_SPACE_SIZE as usize
                } else {
                    0
                };

                for i in 0..num_stack_args {
                    let arg_idx = num_reg_args + i;
                    let arg = &args[arg_idx];

                    // Incoming args are at RBP + 16 + shadow + i*8
                    // (Return address is 8, saved RBP pushed, so RBP points to saved RBP)
                    // Wait:
                    // Standard Prologue: push rbp; mov rbp, rsp
                    // Stack:
                    //   [RBP + 16 + shadow + 8*i] -> Arg N (Stack Arg i)
                    //   [RBP + 8]  -> Return Address
                    //   [RBP + 0]  -> Saved RBP
                    //
                    // So first stack arg is at RBP + 16 (on Linux/Mac) or RBP + 16 + 32 (Windows? No shadow is allocated by caller?)
                    // Windows: Shadow space is 32 bytes allocated *by caller* right before return address?
                    // No, shadow space is allocated by caller *above* return address?
                    // Microsoft x64: "The caller allocates space for 4 register parameters..."
                    // Stack: [RetAddr] [Home P1] [Home P2] [Home P3] [Home P4] [Stack Arg 5] ...
                    // Wait, Home space is "Shadow Space". It's strictly for the first 4 args (which are in regs).
                    // Stack args start *after* shadow space.
                    // So RBP + 16 + 32 + i*8.

                    let offset = 16 + shadow_space + (i * stack::SLOT_SIZE);

                    // Load to RAX (scratch) first
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;

                    // Store to incoming slot
                    writeln!(writer, "    movq %rax, {offset}(%rbp)")?;
                }
            }

            restore_saved_registers(writer, stack_size, saved_regs)?;
            writeln!(writer, "    movq %rbp, %rsp")?;
            writeln!(writer, "    popq %rbp")?;
            let mangled_name = abi.mangle_function_name(name);
            writeln!(writer, "    jmp {mangled_name}")?;
        }
        MirInst::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).ok_or_else(|| {
                LaminaError::ValidationError("stack address has no frame slot".into())
            })?;
            writeln!(writer, "    leaq {offset}(%rbp), %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(vreg) => {
                    let is_placeholder = !def_regs.contains(vreg);

                    if is_placeholder {
                        if let Some(slot_offset) = stack_slots.get(vreg) {
                            let total_offset = *slot_offset as i64 + (*offset as i64);
                            if total_offset == 0 {
                                writeln!(writer, "    leaq (%rbp), %rax")?;
                            } else {
                                writeln!(writer, "    leaq {total_offset}(%rbp), %rax")?;
                            }
                        } else {
                            return Err(LaminaError::ValidationError(format!(
                                "LEA placeholder base vreg {vreg:?} has no stack slot"
                            )));
                        }
                    } else if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        if *offset == 0 {
                            writeln!(writer, "    movq %{phys}, %rax")?;
                        } else {
                            writeln!(writer, "    leaq {offset}(%{phys}), %rax")?;
                        }
                    } else if let Some(slot_offset) = stack_slots.get(vreg) {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                        writeln!(writer, "    movq {slot_offset}(%rbp), %{scratch}")?;
                        if *offset == 0 {
                            writeln!(writer, "    movq %{scratch}, %rax")?;
                        } else {
                            writeln!(writer, "    leaq {offset}(%{scratch}), %rax")?;
                        }
                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    } else {
                        return Err(LaminaError::ValidationError(format!(
                            "x86_64 backend cannot lower LEA for base vreg {vreg:?} (no mapping/slot)"
                        )));
                    }
                }
                Register::Physical(phys) => {
                    if *offset == 0 {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    } else {
                        writeln!(writer, "    leaq {}(%{}), %rax", offset, phys.name)?;
                    }
                }
            }

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let (add, sub, mul, div) = if is_f32(ty) {
                ("addss", "subss", "mulss", "divss")
            } else {
                ("addsd", "subsd", "mulsd", "divsd")
            };
            load_float_operand_to_xmm(lhs, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            load_float_operand_to_xmm(rhs, writer, reg_alloc, stack_slots, "xmm1", ty)?;
            let sse_op = match op {
                FloatBinOp::FAdd => add,
                FloatBinOp::FSub => sub,
                FloatBinOp::FMul => mul,
                FloatBinOp::FDiv => div,
            };
            writeln!(writer, "    {sse_op} %xmm1, %xmm0")?;
            if let Register::Virtual(vreg) = dst {
                store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, ty)?;
            }
        }
        MirInst::FloatUnary { op, ty, dst, src } => {
            load_float_operand_to_xmm(src, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            match op {
                FloatUnOp::FNeg => {
                    if is_f32(ty) {
                        // XOR with the sign bit of f32
                        writeln!(writer, "    movq $0x80000000, %rax")?;
                        writeln!(writer, "    movd %rax, %xmm1")?;
                        writeln!(writer, "    xorps %xmm1, %xmm0")?;
                    } else {
                        // XOR with the sign bit of f64
                        writeln!(writer, "    movq $0x8000000000000000, %rax")?;
                        writeln!(writer, "    movq %rax, %xmm1")?;
                        writeln!(writer, "    xorpd %xmm1, %xmm0")?;
                    }
                }
                FloatUnOp::FSqrt => {
                    let sqrt = if is_f32(ty) { "sqrtss" } else { "sqrtsd" };
                    writeln!(writer, "    {sqrt} %xmm0, %xmm0")?;
                }
            }
            if let Register::Virtual(vreg) = dst {
                store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, ty)?;
            }
        }
        MirInst::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            load_float_operand_to_xmm(lhs, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            load_float_operand_to_xmm(rhs, writer, reg_alloc, stack_slots, "xmm1", ty)?;
            let ucomi = if is_f32(ty) { "ucomiss" } else { "ucomisd" };
            writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
            match op {
                FloatCmpOp::Eq => writeln!(writer, "    sete %al")?,
                FloatCmpOp::Ne => writeln!(writer, "    setne %al")?,
                FloatCmpOp::Lt => writeln!(writer, "    setb %al")?,
                FloatCmpOp::Le => writeln!(writer, "    setbe %al")?,
                FloatCmpOp::Gt => writeln!(writer, "    seta %al")?,
                FloatCmpOp::Ge => writeln!(writer, "    setae %al")?,
            }
            writeln!(writer, "    movzbq %al, %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // Load false_val into a scratch register, true_val into rax.
            // Then test the condition and cmovnz true_val path.
            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
            load_operand_to_register(false_val, writer, reg_alloc, stack_slots, scratch)?;
            load_operand_to_rax(true_val, writer, reg_alloc, stack_slots)?;
            match cond {
                Register::Virtual(vreg) => {
                    if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        writeln!(writer, "    testq %{phys}, %{phys}")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        writeln!(writer, "    cmpq $0, {offset}(%rbp)")?;
                    }
                }
                Register::Physical(p) => {
                    writeln!(writer, "    testq %{}, %{}", p.name, p.name)?;
                }
            }
            // cmovz: if condition is zero (false), move scratch into rax
            writeln!(writer, "    cmovzq %{scratch}, %rax")?;
            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            // Load switch value into rax, then emit a linear cmp/je chain.
            match value {
                Register::Virtual(vreg) => {
                    load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => {
                    writeln!(writer, "    movq %{}, %rax", p.name)?;
                }
            }
            for (case_val, case_label) in cases {
                writeln!(writer, "    cmpq ${case_val}, %rax")?;
                writeln!(writer, "    je .L_{func_name}.{case_label}")?;
            }
            writeln!(writer, "    jmp .L_{func_name}.{default}")?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                load_operand_to_rax(val, writer, reg_alloc, stack_slots)?;
            }
            // Epilogue
            restore_saved_registers(writer, stack_size, saved_regs)?;
            if has_frame {
                X86Frame::generate_epilogue(writer, stack_size)?;
            } else {
                writeln!(writer, "    ret")?;
            }
        }
        MirInst::Jmp { target } => {
            if target != next_label {
                writeln!(writer, "    jmp .L_{func_name}.{target}")?;
            }
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
            }
            writeln!(writer, "    testq %rax, %rax")?;
            if true_target == next_label {
                writeln!(writer, "    jz .L_{func_name}.{false_target}")?;
            } else {
                writeln!(writer, "    jnz .L_{func_name}.{true_target}")?;
                if false_target != next_label {
                    writeln!(writer, "    jmp .L_{func_name}.{false_target}")?;
                }
            }
        }
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support MIR instruction {other:?}"
            )));
        }
    }

    Ok(())
}

/// Width suffix for a `mov` touching memory of this type.
fn size_suffix(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "b",
        2 => "w",
        4 => "l",
        _ => "q",
    }
}

/// The sub-register of `rax` holding exactly the bytes of this type.
fn rax_for(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "al",
        2 => "ax",
        4 => "eax",
        _ => "rax",
    }
}

/// Narrow integers in 64-bit registers are sign-extended; loads read only their type width.
/// Value for `cmpq $imm, %rax`, or None when the operand is not an immediate that fits
/// a sign-extended imm32. Unsigned compares see the operand zero-extended to the type
/// width, matching the `extend_integer` the register path performs.
fn cmp_immediate(rhs: &Operand, ty: &MirType, unsigned: bool) -> Option<i64> {
    let Operand::Immediate(imm) = rhs else {
        return None;
    };
    let raw = match imm {
        Immediate::I8(v) => i64::from(*v),
        Immediate::I16(v) => i64::from(*v),
        Immediate::I32(v) => i64::from(*v),
        Immediate::I64(v) => *v,
        _ => return None,
    };
    let value = if unsigned {
        match ty.size_bytes() {
            1 => raw & 0xff,
            2 => raw & 0xffff,
            4 => raw & 0xffff_ffff,
            _ => raw,
        }
    } else {
        raw
    };
    i32::try_from(value).ok().map(i64::from)
}

/// Condition-code suffix for the negation of a comparison.
fn inverted_cmp_suffix(op: &IntCmpOp) -> Result<&'static str, LaminaError> {
    Ok(match op {
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
        #[allow(unreachable_patterns)]
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support integer comparison {other:?}"
            )));
        }
    })
}

/// Condition-code suffix for an integer comparison, as used by `setCC` and `jCC`.
fn cmp_suffix(op: &IntCmpOp) -> Result<&'static str, LaminaError> {
    Ok(match op {
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
        #[allow(unreachable_patterns)]
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support integer comparison {other:?}"
            )));
        }
    })
}

/// Emit the `cmpq` for an integer comparison, leaving the flags set. Returns the scratch
/// register that must be released, or "" when the right operand folded into an immediate.
fn emit_int_compare(
    op: &IntCmpOp,
    ty: &MirType,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut RaxTrackingWriter<impl Write>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<&'static str, LaminaError> {
    let unsigned = matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    );
    // Both sides already in registers at full width: compare them where they sit.
    if !unsigned || ty.size_bytes() == 8 {
        let physical = |value: &Operand| match value {
            Operand::Register(Register::Virtual(vreg)) => reg_alloc.get_mapping_for(vreg),
            _ => None,
        };
        if let (Some(left), Some(right)) = (physical(lhs), physical(rhs)) {
            writeln!(writer, "    cmpq %{right}, %{left}")?;
            return Ok("");
        }
    }
    load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
    if let Some(v) = cmp_immediate(rhs, ty, unsigned) {
        if unsigned {
            extend_integer(writer, "rax", ty.size_bytes(), false)?;
        }
        writeln!(writer, "    cmpq ${v}, %rax")?;
        return Ok("");
    }
    let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
    if unsigned {
        extend_integer(writer, "rax", ty.size_bytes(), false)?;
        extend_integer(writer, scratch, ty.size_bytes(), false)?;
    }
    writeln!(writer, "    cmpq %{scratch}, %rax")?;
    Ok(scratch)
}

/// Emission order for a function's blocks: entry first, then each block's fall-through
/// successor next whenever it has not been placed yet. A branch whose target follows it
/// immediately needs no jump at all.
fn block_layout(func: &Function) -> Vec<usize> {
    let index_of: HashMap<&str, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();
    let mut placed = vec![false; func.blocks.len()];
    let mut order = Vec::with_capacity(func.blocks.len());
    let mut queue: VecDeque<usize> = VecDeque::new();
    if let Some(&entry) = index_of.get(func.entry.as_str()) {
        queue.push_back(entry);
    }
    while let Some(start) = queue.pop_front() {
        let mut current = start;
        loop {
            if placed[current] {
                break;
            }
            placed[current] = true;
            order.push(current);
            let successors = func.blocks[current].successors();
            // The last successor is the one the backend would fall out to: Jmp's target,
            // or Br's false target. Place it next and queue the rest.
            for label in successors.iter().rev().skip(1) {
                if let Some(&next) = index_of.get(label)
                    && !placed[next]
                {
                    queue.push_back(next);
                }
            }
            // Prefer the last successor (a Jmp's target, or a Br's false arm). When it is
            // already placed the edge runs backwards, so falling through to the other
            // successor lets the branch invert and the extra jump disappear.
            let fallthrough = successors
                .iter()
                .rev()
                .filter_map(|label| index_of.get(label).copied())
                .find(|&next| !placed[next]);
            match fallthrough {
                Some(next) => current = next,
                None => break,
            }
        }
    }
    for (index, done) in placed.iter().enumerate() {
        if !done {
            order.push(index);
        }
    }
    order
}

/// Emit a parallel copy into `dsts`, honouring the fact that a source may itself be one
/// of the destinations. Identity moves are dropped; a cycle is broken through rax.
fn emit_parallel_copy(
    pairs: &[(String, String)],
    writer: &mut RaxTrackingWriter<impl Write>,
) -> Result<(), LaminaError> {
    let mut pending: Vec<(String, String)> = pairs
        .iter()
        .filter(|(src, dst)| src != dst)
        .cloned()
        .collect();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|(_, dst)| !pending.iter().any(|(src, _)| src == dst));
        let index = match ready {
            Some(index) => index,
            None => {
                // Every remaining destination is still somebody's source: break the cycle
                // by parking one value in rax.
                let (src, dst) = pending[0].clone();
                writeln!(writer, "    movq {src}, %rax")?;
                for entry in pending.iter_mut() {
                    if entry.0 == src {
                        entry.0 = "%rax".to_string();
                    }
                }
                pending[0] = ("%rax".to_string(), dst);
                continue;
            }
        };
        let (src, dst) = pending.remove(index);
        if src.ends_with("(%rbp)") && dst.ends_with("(%rbp)") {
            writeln!(writer, "    movq {src}, %rax")?;
            writeln!(writer, "    movq %rax, {dst}")?;
        } else {
            writeln!(writer, "    movq {src}, {dst}")?;
        }
    }
    Ok(())
}

/// Label just past the prologue, where a self tail call re-enters the function.
fn self_tail_entry_label(func_name: &str) -> String {
    format!(".L_{func_name}.tailentry")
}

fn load_insn_for(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "movsbq (%rax), %rax",
        2 => "movswq (%rax), %rax",
        4 => "movslq (%rax), %rax",
        _ => "movq (%rax), %rax",
    }
}

fn load_copy_operand_to_rax(
    ty: &MirType,
    src: &Operand,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let size = ty.size_bytes();
    match src {
        Operand::Register(Register::Virtual(vreg)) => {
            if let Some(physical) = reg_alloc.get_mapping_for(vreg) {
                emit_copy_register_load(writer, physical, size)?;
            } else if let Some(offset) = stack_slots.get(vreg) {
                match size {
                    1 => writeln!(writer, "    movsbq {offset}(%rbp), %rax")?,
                    2 => writeln!(writer, "    movswq {offset}(%rbp), %rax")?,
                    4 => writeln!(writer, "    movslq {offset}(%rbp), %rax")?,
                    _ => writeln!(writer, "    movq {offset}(%rbp), %rax")?,
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "Virtual register {vreg:?} has no mapping or stack slot"
                )));
            }
        }
        Operand::Register(Register::Physical(physical)) => {
            emit_copy_register_load(writer, physical.name, size)?;
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
            let shift = 64 - size * 8;
            let value = ((bits << shift) as i64) >> shift;
            writeln!(writer, "    movq ${value}, %rax")?;
        }
    }
    Ok(())
}

fn emit_copy_register_load(
    writer: &mut impl Write,
    register: &str,
    size: usize,
) -> Result<(), LaminaError> {
    let narrowed = register_for_size(register, size).ok_or_else(|| {
        LaminaError::ValidationError(format!(
            "x86_64 backend cannot select a {size}-byte alias for register {register}"
        ))
    })?;
    match size {
        1 => writeln!(writer, "    movsbq %{narrowed}, %rax")?,
        2 => writeln!(writer, "    movswq %{narrowed}, %rax")?,
        4 => writeln!(writer, "    movslq %{narrowed}, %rax")?,
        _ => writeln!(writer, "    movq %{narrowed}, %rax")?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mir::{FunctionBuilder, ScalarType, codegen::from_ir},
        parser::parse_module,
    };

    #[test]
    fn global_frame_keeps_only_dense_storage() {
        let spilled = VirtualReg::gpr(0);
        let array = VirtualReg::gpr(1);
        let mapped = VirtualReg::gpr(2);
        let mut func = FunctionBuilder::new("dense")
            .block("entry")
            .instr(MirInst::Copy {
                ty: MirType::i64(),
                dst: Register::Virtual(spilled),
                src: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(MirInst::StackAddr {
                dst: Register::Virtual(mapped),
                slot: array,
            })
            .build();
        func.stack_reservations.insert(array, 3);
        let plan = HashMap::from([
            (spilled, MirAllocation::Spill(-80)),
            (mapped, MirAllocation::Register("r12")),
        ]);
        let (slots, _, total) = layout_global_stack_slots(&func, &plan);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[&spilled], -8);
        assert_eq!(slots[&array], -40);
        assert_eq!(total, 5);
    }

    #[test]
    fn global_leaf_omits_empty_frame_but_keeps_stack_argument_base() {
        for strategy in [
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc: strategy,
                ..Default::default()
            };
            for (source, frame) in [
                ("fn @f() -> i64 {\nentry:\nret.i64 42\n}", false),
                (
                    "fn @f(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g) -> i64 {\nentry:\nret.i64 %g\n}",
                    true,
                ),
                (
                    "fn @f() -> ptr {\nentry:\n%p = alloc.stack [24 x i8]\nret.ptr %p\n}",
                    true,
                ),
            ] {
                let ir = parse_module(source).expect("parse frame case");
                let mir = from_ir(&ir, "frame").expect("lower frame case");
                let function = mir.get_function("f").expect("frame function");
                let assembly = compile_single_function_x86_64(
                    "f",
                    function,
                    TargetOperatingSystem::Linux,
                    &settings,
                )
                .expect("emit frame case");
                let assembly = String::from_utf8(assembly).expect("assembly text");
                assert_eq!(assembly.contains("pushq %rbp"), frame, "{assembly}");
                assert_eq!(assembly.contains("popq %rbp"), frame, "{assembly}");
                if !frame {
                    assert!(
                        !assembly.contains("%rsp") && !assembly.contains("%rbp"),
                        "{assembly}"
                    );
                }
            }
        }
    }

    #[test]
    fn parameter_cycle_preserves_both_sources() {
        let mut writer = RaxTrackingWriter {
            inner: Vec::new(),
            rax_value: None,
        };
        emit_parallel_copy(
            &[
                ("%rdi".into(), "%rsi".into()),
                ("%rsi".into(), "%rdi".into()),
            ],
            &mut writer,
        )
        .expect("emit parameter swap");
        assert_eq!(
            String::from_utf8(writer.inner).expect("assembly text"),
            "    movq %rdi, %rax\n    movq %rsi, %rdi\n    movq %rax, %rsi\n"
        );
    }

    #[test]
    fn global_frame_keeps_unreferenced_reservations() {
        let mut func = Function::new(Signature::new("reserved"));
        let slot = VirtualReg::gpr(10);
        func.stack_reservations.insert(slot, 2);
        let plan = HashMap::from([(slot, MirAllocation::Register("r12"))]);
        let (slots, _, total) = layout_global_stack_slots(&func, &plan);
        assert_eq!(slots[&slot], -24);
        assert_eq!(total, 3);
    }

    #[test]
    fn rax_round_trip_requires_no_intervening_emission() {
        let reg = VirtualReg::gpr(0);
        let other = VirtualReg::gpr(1);
        let slots = HashMap::from([(reg, -8), (other, -16)]);
        let alloc = X64RegAlloc::new(TargetOperatingSystem::Linux);
        for emission in [
            "    addq $1, %rax\n",
            "    sete %al\n",
            "    call unknown\n",
            "    movq %rcx, (%rdx)\n",
            "    jmp .L_join\n",
            ".L_join:\n",
        ] {
            let mut writer = RaxTrackingWriter {
                inner: Vec::new(),
                rax_value: None,
            };
            store_rax_to_register(&reg, &mut writer, &alloc, &slots).unwrap();
            let stored_len = writer.inner.len();
            load_operand_to_rax(
                &Operand::Register(Register::Virtual(reg)),
                &mut writer,
                &alloc,
                &slots,
            )
            .unwrap();
            assert_eq!(writer.inner.len(), stored_len);
            writer.write_all(emission.as_bytes()).unwrap();
            assert_eq!(writer.rax_value, None);
            let invalidated_len = writer.inner.len();
            load_register_to_rax(&reg, &mut writer, &alloc, &slots).unwrap();
            assert!(writer.inner.len() > invalidated_len);

            store_rax_to_register(&reg, &mut writer, &alloc, &slots).unwrap();
            load_register_to_rax(&other, &mut writer, &alloc, &slots).unwrap();
            assert_eq!(writer.rax_value, None);
        }
    }

    #[test]
    fn float_copy_emits_bit_move_not_add() {
        let ty = MirType::Scalar(ScalarType::F64);
        let dst = Register::Virtual(VirtualReg::fpr(0));
        let function = FunctionBuilder::new("copy_float")
            .block("entry")
            .instr(MirInst::Copy {
                ty,
                dst: dst.clone(),
                src: Operand::Immediate(Immediate::F64(-0.0)),
            })
            .instr(MirInst::Ret { value: None })
            .build();
        let mut module = MirModule::new("copy_test");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_x86_64(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("x86_64 codegen should succeed");
        let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

        assert!(assembly.contains("movq"), "expected move: {assembly}");
        assert!(
            !assembly.contains("addsd"),
            "unexpected float add: {assembly}"
        );
    }
}
