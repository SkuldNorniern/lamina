use super::variadic::{SAVE_AREA_SLOTS, VaArea, spill_register_save_area};
use super::*;
use crate::mir::RelocationModel;
use crate::mir::codegen::MAX_REGISTER_AGGREGATE;
use crate::mir_codegen::common::argument_types;
use crate::mir_codegen::common::claim_stack_slot;
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::io_codegen_error;
use crate::mir_codegen::common::lamina_to_codegen_error;
use crate::mir_codegen::common::{ClobberIndex, FunctionEmissionPlan};

pub(super) fn callee_saved(reg: &str, target_os: TargetOperatingSystem) -> bool {
    matches!(reg, "rbx" | "r12" | "r13" | "r14" | "r15")
        || (target_os == TargetOperatingSystem::Windows && matches!(reg, "rsi" | "rdi"))
}

pub(super) fn instruction_clobbers(
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
        MirInst::Call { .. } | MirInst::CallIndirect { .. } | MirInst::TailCall { .. } => {
            let mut regs = vec!["rcx", "rdx", "r8", "r9", "r10", "r11"];
            if target_os != TargetOperatingSystem::Windows {
                regs.extend(["rsi", "rdi"]);
            }
            regs.extend(X64RegAlloc::xmm_pool_for_global_allocation(
                TargetOperatingSystem::Linux,
            ));
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
        MirInst::IntToFloat { .. } | MirInst::FloatToInt { .. } => vec!["rcx", "rdx"],
        MirInst::IntUnary { .. } => vec!["rcx"],
        MirInst::IntRotate { .. } => vec!["rcx", "rdx"],
        MirInst::IntCmp { .. }
        | MirInst::Store { .. }
        | MirInst::Lea { .. }
        | MirInst::Select { .. } => scratch,
        MirInst::VaStart { .. } | MirInst::VaArg { .. } | MirInst::VaCopy { .. } => {
            vec!["rcx", "rdx"]
        }
        MirInst::InlineAsm { .. } => super::inline_asm::touched_registers(inst),
        _ => Vec::new(),
    }
}

pub(super) fn layout_global_stack_slots(
    func: &Function,
    plan: &HashMap<VirtualReg, MirAllocation<&'static str>>,
) -> (HashMap<VirtualReg, i32>, BTreeSet<VirtualReg>, usize) {
    let mut address_taken: BTreeSet<_> = func
        .stack_reservations
        .keys()
        .copied()
        .filter(|reg| {
            reg.class != RegisterClass::Vec
                || !matches!(plan.get(reg), Some(MirAllocation::Register(_)))
        })
        .collect();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        if let MirInst::StackAddr { slot, .. } = &inst.instr {
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
            let align = func.stack_alignments.get(&reg).copied();
            let index =
                claim_stack_slot(&mut total, extra, align, &X86Frame::calculate_stack_offset);
            entry.insert(X86Frame::calculate_stack_offset(index));
        }
    }
    (slots, defs, total)
}

fn allocate_global(
    strategy: RegallocStrategy,
    intervals: &[LiveInterval],
    pool: &[&'static str],
    allowed: impl Fn(&LiveInterval, &'static str) -> bool,
) -> HashMap<VirtualReg, MirAllocation<&'static str>> {
    match strategy {
        RegallocStrategy::LinearScanGlobal => {
            LinearScanAllocator::allocate_with_constraints(intervals, pool, allowed)
        }
        RegallocStrategy::GraphColorGlobal => {
            GraphColorAllocator::allocate_with_constraints(intervals, pool, allowed)
        }
        RegallocStrategy::Incremental => unreachable!(),
    }
}

pub(super) fn compile_single_function_x86_64(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    abi_extensions: &AbiExtensions,
) -> Result<Vec<u8>, CodegenError> {
    let reserved = simd::reserve_vector_slots(func);
    let func = reserved.as_ref().unwrap_or(func);
    let naked = func.sig.naked;
    if naked {
        ensure_naked_body(func_name, func, target_os)?;
    }
    let mut output = RaxTrackingWriter {
        inner: Vec::new(),
        rax_value: None,
    };
    let abi = X86ABI::new(target_os);

    ensure_signature_support(&func.sig).map_err(lamina_to_codegen_error)?;

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{label}:").map_err(io_codegen_error)?;
    let unwind = os_uses_elf(target_os);
    if unwind {
        writeln!(output, "    .cfi_startproc").map_err(io_codegen_error)?;
    }

    // A self tail call lowers to a jump back past the prologue, so it neither needs a
    // frame nor clobbers anything a real call would.
    let is_leaf = !func.blocks.iter().any(|b| {
        b.instructions.iter().any(|i| match &i.instr {
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

    let (stack_slots, total_slots);

    let mut saved_regs = Vec::new();
    if settings.regalloc != RegallocStrategy::Incremental && !naked {
        let pool = X64RegAlloc::gpr_pool_for_global_allocation(target_os, is_leaf);
        let mut clobbers = ClobberIndex::default();
        let instructions = func.blocks.iter().flat_map(|block| &block.instructions);
        for (index, inst) in instructions.enumerate() {
            for name in instruction_clobbers(inst, is_leaf, target_os, func_name) {
                clobbers.mark(name, index);
            }
            for reg in inst.use_regs().into_iter().chain(inst.def_reg()) {
                if let Register::Physical(phys) = reg {
                    clobbers.mark(phys.name, index);
                }
            }
        }
        let allowed = |interval: &LiveInterval, reg| clobbers.is_clear(reg, interval);
        let mut address_taken: BTreeSet<_> = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match &inst.instr {
                MirInst::StackAddr { slot, .. } => Some(*slot),
                _ => None,
            })
            .collect();
        address_taken.extend(func.stack_reservations.keys());
        let all_intervals = LinearScanAllocator::compute_intervals(func);
        let intervals: Vec<_> = all_intervals
            .iter()
            .filter(|i| i.vreg.class == RegisterClass::Gpr && !address_taken.contains(&i.vreg))
            .cloned()
            .collect();
        let eligible_vectors = simd::register_eligible_vectors(func);
        let vector_intervals: Vec<_> = all_intervals
            .into_iter()
            .filter(|i| eligible_vectors.contains(&i.vreg))
            .collect();
        let xmm_pool = X64RegAlloc::xmm_pool_for_global_allocation(target_os);
        let mut plan = allocate_global(settings.regalloc, &intervals, pool.as_slice(), allowed);
        plan.extend(allocate_global(
            settings.regalloc,
            &vector_intervals,
            xmm_pool.as_slice(),
            allowed,
        ));
        plan = coalesce_dying_inputs(func, &intervals, &plan, allowed);
        (stack_slots, _, total_slots) = layout_global_stack_slots(func, &plan);
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
        (stack_slots, _, total_slots) = assign_stack_slots(func, X86Frame::calculate_stack_offset);
    }

    // An aggregate too large for the return registers travels through a hidden pointer
    // the caller passes in the first argument register. The callee keeps it in a slot
    // of its own, since the calls it makes would clobber that register.
    if !naked {
        for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
            for reg in super::inline_asm::touched_registers(inst) {
                if callee_saved(reg, target_os) {
                    saved_regs.push(reg);
                }
            }
        }
    }
    saved_regs.sort_unstable();
    saved_regs.dedup();

    let sret = func
        .sig
        .ret_aggregate
        .filter(|aggregate| returns_through_pointer(*aggregate, target_os));
    let sret_slot = sret.map(|_| X86Frame::calculate_stack_offset(total_slots + saved_regs.len()));
    let va_base = total_slots + saved_regs.len() + usize::from(sret.is_some());
    let va_area = if func.sig.variadic && target_os != TargetOperatingSystem::Windows {
        let descs: Vec<ArgDesc> = func
            .sig
            .params
            .iter()
            .map(|p| ArgDesc {
                ty: p.ty,
                aggregate: p.aggregate,
            })
            .collect();
        let locations = classify_argument_descs_with_sret(&descs, target_os, sret.is_some());
        let save = X86Frame::calculate_stack_offset(va_base + SAVE_AREA_SLOTS - 1);
        Some(VaArea::new(save, &locations))
    } else {
        None
    };
    let va_slots = if va_area.is_some() {
        SAVE_AREA_SLOTS
    } else {
        0
    };
    let stack_size = (va_base + va_slots) * 8;
    let has_frame = !naked
        && (settings.regalloc == RegallocStrategy::Incremental
            || !is_leaf
            || stack_size != 0
            || func.sig.params.len() > abi.arg_registers().len()
            || va_area.is_some());
    if has_frame {
        X86Frame::generate_prologue(&mut output, stack_size, unwind)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    } else if unwind {
        writeln!(output, "    .cfi_def_cfa %rsp, 8").map_err(io_codegen_error)?;
    }

    for (index, reg) in saved_regs.iter().enumerate() {
        let offset = X86Frame::calculate_stack_offset(total_slots + index);
        writeln!(output, "    movq %{reg}, {offset}(%rbp)")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        if unwind {
            writeln!(output, "    .cfi_offset %{reg}, {}", offset - 16)
                .map_err(io_codegen_error)?;
        }
    }

    if let Some(area) = &va_area {
        spill_register_save_area(&mut output, area, func_name).map_err(io_codegen_error)?;
    }

    // A self tail call re-enters here with its arguments already in the ABI argument
    // registers, so it reuses the parameter homing below and skips frame teardown and
    // setup entirely.
    writeln!(output, "{}:", self_tail_entry_label(func_name))
        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

    if let Some(offset) = sret_slot {
        let hidden = abi.arg_registers()[0];
        writeln!(output, "    movq %{hidden}, {offset}(%rbp)")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    if !func.sig.params.is_empty() {
        let abi_for_func = X86ABI::new(target_os);
        let arg_regs = abi_for_func.arg_registers();
        let descs: Vec<ArgDesc> = func
            .sig
            .params
            .iter()
            .map(|p| ArgDesc {
                ty: p.ty,
                aggregate: p.aggregate,
            })
            .collect();
        let locations = classify_argument_descs_with_sret(&descs, target_os, sret.is_some());
        home_aggregate_parameters(
            &mut output,
            &func.sig.params,
            &locations,
            arg_regs,
            &stack_slots,
            target_os,
        )?;

        if settings.regalloc == RegallocStrategy::Incremental {
            for (index, param) in func.sig.params.iter().enumerate() {
                if param.aggregate.is_some() {
                    continue;
                }
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    match locations[index] {
                        ArgLocation::GprWords { .. } | ArgLocation::StackWords { .. } => {}
                        ArgLocation::Gpr(gpr) => {
                            let phys_arg = arg_regs[gpr];
                            let destination = reg_alloc
                                .get_mapping_for(vreg)
                                .map(|reg| format!("%{reg}"))
                                .unwrap_or_else(|| format!("{slot_off}(%rbp)"));
                            let source = format!("%{phys_arg}");
                            if source == destination {
                                continue;
                            }
                            writeln!(output, "    movq {source}, {destination}")
                                .map_err(io_codegen_error)?;
                        }
                        ArgLocation::Xmm(xmm) => {
                            writeln!(output, "{}", move_xmm_to_rax(&param.ty, xmm))
                                .map_err(io_codegen_error)?;
                            store_rax_to_register(vreg, &mut output, &reg_alloc, &stack_slots)
                                .map_err(io_codegen_error)?;
                        }
                        ArgLocation::Stack(stack_index) => {
                            // Saved rbp and the return address sit at rbp+0 and rbp+8;
                            // on Windows the caller's home space follows before the
                            // first stack argument.
                            let home = if target_os == TargetOperatingSystem::Windows {
                                windows::SHADOW_SPACE_SIZE
                            } else {
                                0
                            };
                            let caller_off = 16 + home + (stack_index as i32) * 8;
                            writeln!(output, "    movq {caller_off}(%rbp), %rax")
                                .map_err(io_codegen_error)?;
                            store_rax_to_register(vreg, &mut output, &reg_alloc, &stack_slots)
                                .map_err(io_codegen_error)?;
                        }
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
                    if param.aggregate.is_some() {
                        return None;
                    }
                    let source = match locations[index] {
                        ArgLocation::GprWords { .. } | ArgLocation::StackWords { .. } => {
                            return None;
                        }
                        ArgLocation::Gpr(gpr) => format!("%{}", arg_regs[gpr]),
                        ArgLocation::Stack(stack_index) => {
                            let home = if target_os == TargetOperatingSystem::Windows {
                                windows::SHADOW_SPACE_SIZE
                            } else {
                                0
                            };
                            let offset = 16 + home + (stack_index as i32) * 8;
                            format!("{offset}(%rbp)")
                        }
                        ArgLocation::Xmm(_) => return None,
                    };
                    let destination = reg_alloc
                        .get_mapping_for(vreg)
                        .map(|reg| format!("%{reg}"))
                        .or_else(|| {
                            stack_slots
                                .get(vreg)
                                .map(|offset| format!("{offset}(%rbp)"))
                        })?;
                    Some((source, destination))
                })
                .collect();
            emit_parallel_copy(&pairs, &mut output).map_err(lamina_to_codegen_error)?;
            for (index, param) in func.sig.params.iter().enumerate() {
                let (ArgLocation::Xmm(xmm), Register::Virtual(vreg)) =
                    (locations[index], &param.reg)
                else {
                    continue;
                };
                if reg_alloc.get_mapping_for(vreg).is_none() && !stack_slots.contains_key(vreg) {
                    continue;
                }
                writeln!(output, "{}", move_xmm_to_rax(&param.ty, xmm))
                    .map_err(io_codegen_error)?;
                store_rax_to_register(vreg, &mut output, &reg_alloc, &stack_slots)
                    .map_err(io_codegen_error)?;
            }
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

    let param_extensions: Vec<Option<(usize, bool)>> = func
        .sig
        .params
        .iter()
        .map(|param| narrow_canonical_extension(&param.ty))
        .collect();
    emit_narrow_param_normalization(&mut output, &param_extensions, &self_param_homes)?;

    let plan = FunctionEmissionPlan::new(func);
    let layout = &plan.order;
    let rotations = &plan.rotations;
    if layout.first().map(|&i| func.blocks[i].label.as_str()) != Some(func.entry.as_str()) {
        writeln!(output, "    jmp .L_{}.{}", func_name, func.entry).map_err(io_codegen_error)?;
    }

    // A comparison whose only consumer is the branch right after it needs no setCC and no
    // materialised 0/1: the flags from cmpq drive jCC directly.
    let use_counts = use_counts(func);
    let flag_producers = &plan.comparisons;

    let loop_headers = &plan.aligned_headers;

    for (slot, &block_index) in layout.iter().enumerate() {
        let block = &func.blocks[block_index];
        let next_label = layout
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        output.rax_value = None;
        if loop_headers.contains(&block_index) {
            writeln!(output, "    .p2align 5,,31").map_err(io_codegen_error)?;
        }
        writeln!(output, ".L_{}.{}:", func_name, block.label).map_err(io_codegen_error)?;

        let mut skip = 0;
        for (position, inst) in block.instructions.iter().enumerate() {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            if let Some((length, address, dst)) = select_lea(
                &block.instructions[position..],
                &reg_alloc,
                &use_counts,
                |offset| flag_producers.contains(&(block_index, position + offset)),
                func.pointer_width,
            ) {
                emit_source_loc(&mut output, settings, inst.loc)
                    .map_err(lamina_to_codegen_error)?;
                writeln!(output, "    leaq {address}, %{dst}")
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                skip = length - 1;
                continue;
            }
            if let MirInst::IntCmp {
                op,
                dst,
                lhs,
                rhs,
                ty,
            } = &inst.instr
                && let Register::Virtual(cmp_dst) = dst
                && let Some(MirInst::Br {
                    cond: Register::Virtual(cond),
                    true_target,
                    false_target,
                }) = block.instructions.get(position + 1).map(|i| &i.instr)
                && cond == cmp_dst
                && flag_producers.contains(&(block_index, position))
            {
                emit_source_loc(&mut output, settings, inst.loc)
                    .map_err(lamina_to_codegen_error)?;
                let scratch = emit_int_compare(
                    op,
                    ty,
                    lhs,
                    rhs,
                    &mut output,
                    &mut reg_alloc,
                    &stack_slots,
                    func.pointer_width,
                )
                .map_err(lamina_to_codegen_error)?;
                let suffix = cmp_suffix(op).map_err(lamina_to_codegen_error)?;
                if true_target == next_label
                    || rotations
                        .get(&block_index)
                        .is_some_and(|&body| func.blocks[body].label == *false_target)
                {
                    let inverted = inverted_cmp_suffix(op).map_err(lamina_to_codegen_error)?;
                    writeln!(output, "    j{inverted} .L_{func_name}.{false_target}")
                        .map_err(io_codegen_error)?;
                    if true_target != next_label {
                        writeln!(output, "    jmp .L_{func_name}.{true_target}")
                            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                    }
                } else {
                    writeln!(output, "    j{suffix} .L_{func_name}.{true_target}")
                        .map_err(io_codegen_error)?;
                    if false_target != next_label {
                        writeln!(output, "    jmp .L_{func_name}.{false_target}")
                            .map_err(io_codegen_error)?;
                    }
                }
                if !scratch.is_empty() && scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
                output.rax_value = None;
                skip = 1;
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
                settings,
                inst.loc,
                abi_extensions,
                next_label,
                &self_param_homes,
                &param_extensions,
                &func.entry,
                sret_slot,
                va_area.as_ref(),
                func.pointer_width,
            )
            .map_err(|e| match e {
                LaminaError::CodegenError(refusal @ CodegenError::CpuLevelRequired { .. }) => {
                    refusal
                }
                other => CodegenError::InvalidCodegenOptions(other.to_string()),
            })?;
        }
        output.rax_value = None;
    }

    if unwind {
        writeln!(output, "    .cfi_endproc").map_err(io_codegen_error)?;
    }
    Ok(strip_self_moves(output.inner))
}

/// A `@naked` body is emitted verbatim, so it must be inline assembly with no operands and
/// nothing that expects a frame, parameters or a return value.
fn ensure_naked_body(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
) -> Result<(), CodegenError> {
    let refuse = |reason: &str| {
        CodegenError::InvalidMir(vec![format!("@naked function '{func_name}' {reason}")])
    };
    if target_os != TargetOperatingSystem::Linux {
        return Err(CodegenError::UnsupportedFeature(format!(
            "@naked function '{func_name}' is supported on x86_64 Linux only"
        )));
    }
    if !func.sig.params.is_empty()
        || func.sig.variadic
        || func.sig.ret_ty.is_some()
        || func.sig.ret_aggregate.is_some()
    {
        return Err(refuse("takes parameters or returns a value"));
    }
    let [block] = func.blocks.as_slice() else {
        return Err(refuse("must have exactly one block"));
    };
    let Some((last, body)) = block.instructions.split_last() else {
        return Err(refuse("has an empty body"));
    };
    if !matches!(&last.instr, MirInst::Unreachable) {
        return Err(refuse("must end in unreachable"));
    }
    for inst in body {
        match &inst.instr {
            MirInst::InlineAsm {
                operands,
                result: None,
                ..
            } if operands.is_empty() => {}
            _ => {
                return Err(refuse(
                    "contains an instruction other than operand-free inline asm",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn ensure_signature_support(sig: &Signature) -> Result<(), LaminaError> {
    for (idx, param) in sig.params.iter().enumerate() {
        ensure_type_supported(&param.ty, &format!("parameter {} of '{}'", idx, sig.name))?;
    }

    if let Some(ret_ty) = &sig.ret_ty {
        ensure_type_supported(ret_ty, &format!("return type of '{}'", sig.name))?;
    }

    Ok(())
}

pub(super) fn ensure_type_supported(ty: &MirType, context: &str) -> Result<(), LaminaError> {
    if ty.is_vector() {
        return Err(LaminaError::ValidationError(format!(
            "x86_64 backend does not support {context} (type {ty})"
        )));
    }
    Ok(())
}

/// Width and signedness of a narrow integer parameter. The ABI leaves the bits above
/// the declared width unspecified, so the callee rebuilds the canonical 64-bit form
/// that the rest of the function assumes: sign-extended for signed types, zero-extended
/// for `i1`, which only ever holds 0 or 1.
pub(super) type NarrowExtension = (usize, bool);

/// Where an argument or parameter is passed under the target convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArgLocation {
    Gpr(usize),
    Xmm(usize),
    Stack(usize),
    /// An integer aggregate in `words` consecutive general registers from `first`.
    GprWords {
        first: usize,
        words: usize,
    },
    /// An integer aggregate in `words` consecutive stack slots from `first`.
    StackWords {
        first: usize,
        words: usize,
    },
}

/// One argument as the convention sees it: its scalar type, or the aggregate it is a
/// pointer to.
#[derive(Clone, Copy, Debug)]
pub(super) struct ArgDesc {
    pub(super) ty: MirType,
    pub(super) aggregate: Option<Aggregate>,
}

impl ArgDesc {
    pub(super) fn scalar(ty: MirType) -> Self {
        Self {
            ty,
            aggregate: None,
        }
    }
}

/// Whether Windows passes this aggregate in a register. Only the sizes of a
/// register-width integer go by value; every other size goes by reference, to a copy
/// the caller owns.
pub(super) fn windows_aggregate_fits(aggregate: Aggregate) -> bool {
    matches!(aggregate.size, 1 | 2 | 4 | 8)
}

/// Whether an aggregate return travels through a hidden pointer rather than registers.
pub(super) fn returns_through_pointer(
    aggregate: Aggregate,
    target_os: TargetOperatingSystem,
) -> bool {
    if target_os == TargetOperatingSystem::Windows {
        !windows_aggregate_fits(aggregate)
    } else {
        aggregate.size as u64 > MAX_REGISTER_AGGREGATE
    }
}

/// Whether an aggregate argument travels as a pointer to a copy the caller owns.
pub(super) fn passed_by_reference(aggregate: Aggregate, target_os: TargetOperatingSystem) -> bool {
    target_os == TargetOperatingSystem::Windows && !windows_aggregate_fits(aggregate)
}

const SSE_ARGUMENT_COUNT: usize = 8;

/// System V draws integer and floating arguments from separate register banks, so a
/// float takes the next xmm register even when integer registers remain. Windows gives
/// each position one slot, which is an xmm register for floats and a general register
/// for integers, and spills positions past the fourth to the stack.
pub(super) fn classify_arguments(
    types: &[MirType],
    target_os: TargetOperatingSystem,
) -> Vec<ArgLocation> {
    let descs: Vec<ArgDesc> = types.iter().copied().map(ArgDesc::scalar).collect();
    classify_argument_descs(&descs, target_os)
}

pub(super) fn classify_argument_descs(
    descs: &[ArgDesc],
    target_os: TargetOperatingSystem,
) -> Vec<ArgLocation> {
    classify_argument_descs_with_sret(descs, target_os, false)
}

/// `sret` reserves the first general register for the hidden pointer a large
/// aggregate return is written through, which shifts every other argument along.
pub(super) fn classify_argument_descs_with_sret(
    descs: &[ArgDesc],
    target_os: TargetOperatingSystem,
    sret: bool,
) -> Vec<ArgLocation> {
    let gpr_count = X86ABI::new(target_os).arg_registers().len();
    let windows = target_os == TargetOperatingSystem::Windows;
    let mut gprs = usize::from(sret);
    let mut sses = 0;
    let mut stack = 0;
    let mut locations = Vec::with_capacity(descs.len());
    for (position, desc) in descs.iter().enumerate() {
        if let Some(aggregate) = desc.aggregate {
            let words = aggregate.words();
            let slot = position + usize::from(sret);
            let location = if windows {
                if slot < gpr_count {
                    ArgLocation::Gpr(slot)
                } else {
                    stack += 1;
                    ArgLocation::Stack(stack - 1)
                }
            } else if words * 8 <= MAX_REGISTER_AGGREGATE as usize && gprs + words <= gpr_count {
                gprs += words;
                ArgLocation::GprWords {
                    first: gprs - words,
                    words,
                }
            } else {
                // A 16-byte-aligned aggregate starts on an even stack slot, so an odd
                // slot before it is left as padding.
                if aggregate.align >= 16 && stack % 2 == 1 {
                    stack += 1;
                }
                stack += words;
                ArgLocation::StackWords {
                    first: stack - words,
                    words,
                }
            };
            locations.push(location);
            continue;
        }
        let ty = desc.ty;
        let float = ty.is_float();
        let slot = position + usize::from(sret);
        let location = if windows && slot < gpr_count {
            if float {
                ArgLocation::Xmm(slot)
            } else {
                ArgLocation::Gpr(slot)
            }
        } else if !windows && float && sses < SSE_ARGUMENT_COUNT {
            sses += 1;
            ArgLocation::Xmm(sses - 1)
        } else if !windows && !float && gprs < gpr_count {
            gprs += 1;
            ArgLocation::Gpr(gprs - 1)
        } else {
            stack += 1;
            ArgLocation::Stack(stack - 1)
        };
        locations.push(location);
    }
    locations
}

/// Writes the incoming words of each aggregate parameter into the frame slot the callee
/// addresses them through.
fn home_aggregate_parameters<W: Write>(
    output: &mut RaxTrackingWriter<W>,
    params: &[Parameter],
    locations: &[ArgLocation],
    arg_regs: &[&str],
    stack_slots: &HashMap<VirtualReg, i32>,
    target_os: TargetOperatingSystem,
) -> Result<(), CodegenError> {
    let io_error = io_codegen_error;
    let home = if target_os == TargetOperatingSystem::Windows {
        windows::SHADOW_SPACE_SIZE
    } else {
        0
    };
    for (index, param) in params.iter().enumerate() {
        if param.aggregate.is_none() {
            continue;
        }
        let Register::Virtual(slot) = &param.reg else {
            continue;
        };
        let Some(&base) = stack_slots.get(slot) else {
            return Err(CodegenError::InvalidCodegenOptions(
                "aggregate parameter has no frame slot".to_string(),
            ));
        };
        let Some(aggregate) = param.aggregate else {
            continue;
        };
        if passed_by_reference(aggregate, target_os) {
            let pointer = match locations[index] {
                ArgLocation::Gpr(gpr) => format!("%{}", arg_regs[gpr]),
                ArgLocation::Stack(slot_index) => {
                    let at = 16 + home + (slot_index as i32) * 8;
                    writeln!(output, "    movq {at}(%rbp), %r11").map_err(io_error)?;
                    "%r11".to_string()
                }
                _ => continue,
            };
            for word in 0..aggregate.words() {
                let at = (word as i32) * 8;
                writeln!(output, "    movq {at}({pointer}), %rax").map_err(io_error)?;
                writeln!(output, "    movq %rax, {}(%rbp)", base + at).map_err(io_error)?;
            }
            output.rax_value = None;
            continue;
        }
        let sources: Vec<(usize, Option<i32>, &str)> = match locations[index] {
            ArgLocation::Gpr(gpr) => vec![(0, None, arg_regs[gpr])],
            ArgLocation::GprWords { first, words } => (0..words)
                .map(|word| (word, None, arg_regs[first + word]))
                .collect(),
            ArgLocation::Stack(slot_index) => {
                vec![(0, Some(16 + home + (slot_index as i32) * 8), "rax")]
            }
            ArgLocation::StackWords { first, words } => (0..words)
                .map(|word| (word, Some(16 + home + ((first + word) as i32) * 8), "rax"))
                .collect(),
            ArgLocation::Xmm(_) => Vec::new(),
        };
        for (word, caller, source) in sources {
            if let Some(caller) = caller {
                writeln!(output, "    movq {caller}(%rbp), %rax").map_err(io_error)?;
            }
            writeln!(
                output,
                "    movq %{source}, {}(%rbp)",
                base + (word as i32) * 8
            )
            .map_err(io_error)?;
        }
        output.rax_value = None;
    }
    Ok(())
}

pub(super) fn stack_argument_count(locations: &[ArgLocation]) -> usize {
    locations
        .iter()
        .map(|location| match location {
            ArgLocation::Stack(slot) => slot + 1,
            ArgLocation::StackWords { first, words } => first + words,
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Moves the float bits of an xmm register into rax, the form every value takes.
pub(super) fn move_xmm_to_rax(ty: &MirType, xmm: usize) -> String {
    if is_f32(ty) {
        format!("    movd %xmm{xmm}, %eax")
    } else {
        format!("    movq %xmm{xmm}, %rax")
    }
}

/// Width-extension and float-passing facts about the module's functions. A returned
/// narrow value is normalised to its canonical form by the caller after the call;
/// arguments are widened by the caller only when the callee is external, since the C
/// ABI names the signedness and leaves the upper bits unspecified.
#[derive(Default)]
pub(super) struct AbiExtensions {
    returns: HashMap<String, NarrowExtension>,
    external_arguments: HashMap<String, Vec<Option<NarrowExtension>>>,
    parameter_types: HashMap<String, Vec<MirType>>,
    parameter_aggregates: HashMap<String, Vec<Option<Aggregate>>>,
    return_aggregates: HashMap<String, Aggregate>,
    return_types: HashMap<String, MirType>,
    variadic: HashSet<String>,
    /// Symbols whose address is loaded through the GOT under `@pic` or `@pie`.
    got_symbols: HashSet<String>,
}

impl AbiExtensions {
    pub(super) fn from_module(module: &MirModule) -> Self {
        let returns = module
            .functions
            .iter()
            .filter_map(|(name, func)| {
                let ty = func.sig.ret_ty.as_ref()?;
                Some((name.clone(), narrow_canonical_extension(ty)?))
            })
            .collect();
        let external_arguments = module
            .external_functions
            .iter()
            .filter_map(|name| {
                let sig = &module.functions.get(name)?.sig;
                let extensions: Vec<_> = sig
                    .params
                    .iter()
                    .map(|param| narrow_extension(&param.ty, param.signedness))
                    .collect();
                extensions
                    .iter()
                    .any(Option::is_some)
                    .then(|| (name.clone(), extensions))
            })
            .collect();
        let parameter_types = module
            .functions
            .iter()
            .map(|(name, func)| (name.clone(), func.sig.params.iter().map(|p| p.ty).collect()))
            .collect();
        let parameter_aggregates = module
            .functions
            .iter()
            .map(|(name, func)| {
                (
                    name.clone(),
                    func.sig.params.iter().map(|p| p.aggregate).collect(),
                )
            })
            .collect();
        let return_types = module
            .functions
            .iter()
            .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_ty?)))
            .collect();
        let return_aggregates = module
            .functions
            .iter()
            .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
            .collect();
        let got_symbols = match module.relocation {
            RelocationModel::Static => HashSet::new(),
            RelocationModel::Pie => module.external_functions.clone(),
            RelocationModel::Pic => module
                .external_functions
                .iter()
                .chain(&module.exported_functions)
                .chain(
                    module
                        .globals
                        .iter()
                        .filter(|(_, global)| global.exported)
                        .map(|(name, _)| name),
                )
                .cloned()
                .collect(),
        };
        Self {
            returns,
            external_arguments,
            parameter_types,
            parameter_aggregates,
            return_types,
            return_aggregates,
            got_symbols,
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, _)| name.clone())
                .collect(),
        }
    }

    pub(super) fn uses_got(&self, symbol: &str) -> bool {
        self.got_symbols.contains(symbol)
    }

    pub(super) fn return_extension(&self, callee: &str) -> Option<NarrowExtension> {
        self.returns.get(callee).copied()
    }

    pub(super) fn external_argument(&self, callee: &str, index: usize) -> Option<NarrowExtension> {
        self.external_arguments
            .get(callee)?
            .get(index)
            .copied()
            .flatten()
    }

    pub(super) fn parameter_types(&self, function: &str) -> Vec<MirType> {
        self.parameter_types
            .get(function)
            .cloned()
            .unwrap_or_default()
    }

    /// Types of a call's arguments.
    pub(super) fn argument_types(&self, callee: &str, args: &[Operand]) -> Vec<MirType> {
        argument_types(
            self.parameter_types.get(callee),
            self.fixed_arguments(callee),
            args,
        )
    }

    /// Arguments of a call as the convention sees them, scalar types paired with the
    /// aggregate markers of the callee's parameters.
    pub(super) fn argument_descs(&self, callee: &str, args: &[Operand]) -> Vec<ArgDesc> {
        let types = self.argument_types(callee, args);
        let aggregates = self.parameter_aggregates.get(callee);
        types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| ArgDesc {
                ty,
                aggregate: aggregates
                    .filter(|markers| markers.len() == args.len())
                    .and_then(|markers| markers[index]),
            })
            .collect()
    }

    /// Index of the first argument the convention treats as variadic, if any.
    pub(super) fn fixed_arguments(&self, callee: &str) -> Option<usize> {
        self.variadic
            .contains(callee)
            .then(|| self.parameter_types.get(callee).map_or(0, Vec::len))
    }

    /// Whether the callee reads its arguments the way a variadic C function does. A
    /// declaration ending in `...` says so, and a parameter list that does not match the
    /// call means the callee is undeclared and may be one.
    pub(super) fn may_be_variadic(&self, callee: &str, count: usize) -> bool {
        self.variadic.contains(callee)
            || !self
                .parameter_types
                .get(callee)
                .is_some_and(|types| types.len() == count)
    }

    pub(super) fn return_aggregate(&self, callee: &str) -> Option<Aggregate> {
        self.return_aggregates.get(callee).copied()
    }

    pub(super) fn float_return(&self, callee: &str) -> Option<MirType> {
        self.return_types
            .get(callee)
            .copied()
            .filter(MirType::is_float)
    }
}

/// Width and signedness of a narrow integer value crossing a boundary. `i1` holds only
/// 0 or 1, so it is always zero-extended.
pub(super) fn narrow_extension(ty: &MirType, signedness: Signedness) -> Option<NarrowExtension> {
    let signed = signedness == Signedness::Signed;
    match ty {
        MirType::Scalar(ScalarType::I1) => Some((1, false)),
        MirType::Scalar(ScalarType::I8) => Some((1, signed)),
        MirType::Scalar(ScalarType::I16) => Some((2, signed)),
        MirType::Scalar(ScalarType::I32) => Some((4, signed)),
        _ => None,
    }
}

/// The canonical in-register form of a narrow value. Inside a function every narrow
/// integer is held sign-extended to 64 bits whatever its declared signedness, and the
/// unsigned consumers (unsigned compare, unsigned division, logical shift right)
/// re-extend before use. `i1` is the exception: it carries only 0 or 1.
///
/// Every value entering that world — an incoming parameter, a value returned by a call,
/// a load — is normalised to this form. Only an argument handed to an external function
/// keeps the declared signedness, because the C ABI names it.
pub(super) fn narrow_canonical_extension(ty: &MirType) -> Option<NarrowExtension> {
    match ty {
        MirType::Scalar(ScalarType::I1) => narrow_extension(ty, Signedness::Unsigned),
        _ => narrow_extension(ty, Signedness::Signed),
    }
}

/// Rewrite each narrow parameter home in place to its canonical 64-bit form.
pub(super) fn emit_narrow_param_normalization<W: Write>(
    writer: &mut W,
    extensions: &[Option<(usize, bool)>],
    homes: &[Option<String>],
) -> Result<(), CodegenError> {
    let io_error = io_codegen_error;
    for (extension, home) in extensions.iter().zip(homes) {
        let (Some((width, signed)), Some(home)) = (extension, home) else {
            continue;
        };
        let insn = match (width, signed) {
            (1, true) => "movsbq",
            (2, true) => "movswq",
            (_, true) => "movslq",
            (1, false) => "movzbl",
            (2, false) => "movzwl",
            (_, false) => "movl",
        };
        if let Some(reg) = home.strip_prefix('%') {
            let source = register_for_size(reg, *width).ok_or_else(|| {
                CodegenError::InvalidCodegenOptions(format!("no alias for register {reg}"))
            })?;
            let dest = if *signed {
                reg.to_string()
            } else {
                register_for_size(reg, 4).unwrap_or_else(|| reg.to_string())
            };
            writeln!(writer, "    {insn} %{source}, %{dest}").map_err(io_error)?;
        } else {
            let dest = if *signed { "rax" } else { "eax" };
            writeln!(writer, "    {insn} {home}, %{dest}").map_err(io_error)?;
            writeln!(writer, "    movq %rax, {home}").map_err(io_error)?;
        }
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[allow(clippy::expect_used)]
mod global_regalloc_tests {
    use crate::mir_codegen::{
        simd_support::tools_available,
        test_support::{PRINT_I64_HELPERS, compile},
    };
    use std::{
        env, fs,
        path::Path,
        process::{self, Command},
    };

    #[test]
    fn global_allocators_preserve_program_outputs() {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return;
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let directory = env::temp_dir().join(format!("lamina-regalloc-{}", process::id()));
        fs::create_dir_all(&directory).expect("create executable directory");
        for allocator in ["linear", "graph"] {
            for level in ["0", "3"] {
                for case in [
                    "recursive_fibonacci",
                    "calling_convention_bug_test",
                    "loops",
                    "stack_frame_overflow_bug_test",
                    "float_phi_copy",
                    "gep_variable_indices",
                    "function_call_isolation",
                    "recursion_with_calculation",
                    "type_conversions",
                    "narrow_unsigned_canonical",
                    "array_operations_comprehensive",
                    "struct_operations_comprehensive",
                    "volatile_access",
                ] {
                    let executable = directory.join(format!("{case}-{allocator}-{level}"));
                    compile(
                        root.join(format!("testcases/{case}.lamina")),
                        &executable,
                        &["--opt-level", level, "--regalloc", allocator],
                    )
                    .expect("compile program");
                    let output = Command::new("timeout")
                        .arg("10")
                        .arg(&executable)
                        .output()
                        .expect("run executable");
                    assert!(
                        output.status.success(),
                        "{case}/{allocator}/O{level}: {output:?}"
                    );
                    let expected =
                        fs::read_to_string(root.join(format!("testcases/{case}.expected")))
                            .expect("read expected output");
                    assert_eq!(
                        String::from_utf8_lossy(&output.stdout).trim(),
                        expected.trim(),
                        "{case}/{allocator}/O{level}"
                    );
                }
            }
        }
        fs::remove_dir_all(directory).expect("remove executable directory");
    }

    #[test]
    fn global_allocators_preserve_argument_permutations_and_call_live_values() {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return;
        }
        let directory = env::temp_dir().join(format!("lamina-regalloc-args-{}", process::id()));
        fs::create_dir_all(&directory).expect("create executable directory");
        let source = directory.join("arguments.lamina");
        fs::write(
            &source,
            format!(
                "{}{}",
                r#"
@noinline
fn @digits(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) -> i64 {
entry:
    %b10 = mul.i64 %b, 10
    %c100 = mul.i64 %c, 100
    %d1000 = mul.i64 %d, 1000
    %e10000 = mul.i64 %e, 10000
    %f100000 = mul.i64 %f, 100000
    %g1000000 = mul.i64 %g, 1000000
    %h10000000 = mul.i64 %h, 10000000
    %ab = add.i64 %a, %b10
    %abc = add.i64 %ab, %c100
    %abcd = add.i64 %abc, %d1000
    %abcde = add.i64 %abcd, %e10000
    %abcdef = add.i64 %abcde, %f100000
    %abcdefg = add.i64 %abcdef, %g1000000
    %all = add.i64 %abcdefg, %h10000000
    ret.i64 %all
}
@noinline
fn @probe(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) -> i64 {
entry:
    %reversed = call @digits(%h, %g, %f, %e, %d, %c, %b, %a)
    %forward = call @digits(%a, %b, %c, %d, %e, %f, %g, %h)
    %sum = add.i64 %reversed, %forward
    ret.i64 %sum
}
fn @main() -> i64 {
entry:
    %result = call @probe(1, 2, 3, 4, 5, 6, 7, 8)
    call @print_i64(%result)
    ret.i64 0
}
"#,
                PRINT_I64_HELPERS
            ),
        )
        .expect("write source");
        for allocator in ["linear", "graph"] {
            for level in ["0", "1", "2", "3"] {
                let executable = directory.join(format!("arguments-{allocator}-{level}"));
                compile(
                    &source,
                    &executable,
                    &["--opt-level", level, "--regalloc", allocator],
                )
                .expect("compile program");
                let output = Command::new("timeout")
                    .arg("10")
                    .arg(&executable)
                    .output()
                    .expect("run executable");
                assert!(output.status.success(), "{allocator}/O{level}: {output:?}");
                assert_eq!(output.stdout, b"99999999\n", "{allocator}/O{level}");
            }
        }
        fs::remove_dir_all(directory).expect("remove executable directory");
    }

    #[test]
    fn optimized_defaults_match_linear_scan_and_preserve_explicit_overrides() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let directory = env::temp_dir().join(format!("lamina-regalloc-defaults-{}", process::id()));
        fs::create_dir_all(&directory).expect("create assembly directory");
        for level in ["0", "1", "2", "3"] {
            let default = if matches!(level, "1" | "2" | "3") {
                "linear"
            } else {
                "incremental"
            };
            let compile = |name: &str, args: &[&str]| {
                let assembly = directory.join(format!("{name}-{level}.s"));
                compile(
                    root.join("testcases/loops.lamina"),
                    &assembly,
                    &[args, &["--emit-asm"]].concat(),
                )
                .expect("compile assembly");
                fs::read(assembly).expect("read assembly")
            };
            assert_eq!(
                compile("default", &["--opt-level", level]),
                compile("explicit", &["--opt-level", level, "--regalloc", default])
            );
            for allocator in ["incremental", "linear", "graph"] {
                assert_eq!(
                    compile("before", &["--regalloc", allocator, "--opt-level", level]),
                    compile("after", &["--opt-level", level, "--regalloc", allocator])
                );
            }
        }
        fs::remove_dir_all(directory).expect("remove assembly directory");
    }

    #[test]
    fn u64_constant_above_i64_max_is_unsigned() {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return;
        }
        let directory = env::temp_dir().join(format!("lamina-u64-constant-{}", process::id()));
        fs::create_dir_all(&directory).expect("create executable directory");
        let source = directory.join("print_u64.lamina");
        fs::write(
            &source,
            format!(
                "{}{}",
                r#"
fn @main() -> i64 {
entry:
    call @print_u64(18446744073709551615)
    call @print_u64(9223372036854775808)
    call @print_u64(9223372036854775807)
    ret.i64 0
}
"#,
                PRINT_I64_HELPERS
            ),
        )
        .expect("write u64 source");
        let executable = directory.join("print_u64");
        compile(&source, &executable, &["--opt-level", "0"]).expect("compile program");
        let output = Command::new("timeout")
            .arg("10")
            .arg(&executable)
            .output()
            .expect("run u64 executable");
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            output.stdout,
            b"18446744073709551615\n9223372036854775808\n9223372036854775807\n"
        );
        fs::remove_dir_all(directory).expect("remove executable directory");
    }

    #[test]
    fn padded_struct_storage_preserves_adjacent_allocations_and_frames() {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return;
        }
        let directory = env::temp_dir().join(format!("lamina-padded-storage-{}", process::id()));
        fs::create_dir_all(&directory).expect("create executable directory");
        let source = directory.join("padded.lamina");
        fs::write(
            &source,
            format!(
                "{}{}",
                r#"
@extern
fn @malloc(i64 %n) -> ptr
@extern
fn @free(ptr %p) -> void
@noinline
fn @padded() -> i64 {
entry:
    %stack = stackalloc.struct { a: i8, b: i64, c: i8 }
    %other = stackalloc.[8 x i64]
    store.i64 %other, 4321
    %middle = getfieldptr.struct { a: i8, b: i64, c: i8 } %stack, 1
    %last = getfieldptr.struct { a: i8, b: i64, c: i8 } %stack, 2
    store.i64 %middle, 1234
    store.i8 %last, 99
    %heap = call @malloc(48)
    %second = getelem.struct { a: i8, b: i64, c: i8 } %heap, 1
    %heap_last = getfieldptr.struct { a: i8, b: i64, c: i8 } %second, 2
    store.i8 %heap_last, 77
    %x = load.i64 %middle
    %y = load.i8 %last
    %z = load.i8 %heap_last
    %guard = load.i64 %other
    call @free(%heap)
    %wide_y_bits = bitcast.i8.u8 %y
    %wide_y = extend.u8.i64 %wide_y_bits
    %wide_z_bits = bitcast.i8.u8 %z
    %wide_z = extend.u8.i64 %wide_z_bits
    %sum = add.i64 %x, %wide_y
    %sum2 = add.i64 %sum, %wide_z
    %result = add.i64 %sum2, %guard
    ret.i64 %result
}
fn @main() -> i64 {
entry:
    %result = call @padded()
    call @print_i64(%result)
    ret.i64 0
}
"#,
                PRINT_I64_HELPERS
            ),
        )
        .expect("write padded source");
        for allocator in ["incremental", "linear", "graph"] {
            for level in ["0", "1", "2", "3"] {
                let executable = directory.join(format!("padded-{allocator}-{level}"));
                compile(
                    &source,
                    &executable,
                    &["--opt-level", level, "--regalloc", allocator],
                )
                .expect("compile program");
                let output = Command::new("timeout")
                    .arg("10")
                    .arg(&executable)
                    .output()
                    .expect("run padded executable");
                assert!(output.status.success(), "{allocator}/O{level}: {output:?}");
                assert_eq!(output.stdout, b"5731\n", "{allocator}/O{level}");
            }
        }
        fs::remove_dir_all(directory).expect("remove executable directory");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod naked_tests {
    use super::*;
    use crate::{mir::codegen::from_ir, parser::parse_module};

    fn assembly_for(
        source: &str,
        naked: bool,
        os: TargetOperatingSystem,
    ) -> Result<String, String> {
        let ir = parse_module(source).expect("parse function");
        let mut mir = from_ir(&ir, "naked").expect("lower function");
        if naked {
            mir.functions.get_mut("f").expect("function f").sig.naked = true;
        }
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, os).map_err(|error| error.to_string())?;
        Ok(String::from_utf8(output).expect("utf8 assembly"))
    }

    #[test]
    fn naked_body_is_emitted_without_prologue_or_epilogue() {
        let asm = assembly_for(
            "fn @f() -> void {\nentry:\n    asm.void \"cli; hlt\"\n    unreachable\n}\n",
            true,
            TargetOperatingSystem::Linux,
        )
        .expect("naked function should emit");
        let body: Vec<&str> = asm
            .lines()
            .skip_while(|line| !line.starts_with("f:"))
            .skip(1)
            .filter(|line| !line.trim_start().starts_with('.'))
            .collect();
        assert_eq!(body, vec!["    cli; hlt", "    ud2"], "{asm}");
    }

    #[test]
    fn naked_body_with_an_ir_instruction_is_refused() {
        let error = assembly_for(
            "fn @f() -> void {\nentry:\n    %x = add.i64 1, 2\n    unreachable\n}\n",
            true,
            TargetOperatingSystem::Linux,
        )
        .expect_err("ret in a naked body must be refused");
        assert!(
            error.contains("@naked function 'f' contains an instruction other than"),
            "{error}"
        );
    }

    #[test]
    fn naked_function_is_refused_off_x86_64_linux() {
        let error = assembly_for(
            "fn @f() -> void {\nentry:\n    asm.void \"cli\"\n    unreachable\n}\n",
            true,
            TargetOperatingSystem::Windows,
        )
        .expect_err("naked must be refused off Linux");
        assert!(error.contains("x86_64 Linux only"), "{error}");
    }

    #[test]
    fn naked_function_with_a_parameter_is_refused() {
        let error = assembly_for(
            "fn @f(i64 %a) -> void {\nentry:\n    asm.void \"cli\"\n    unreachable\n}\n",
            true,
            TargetOperatingSystem::Linux,
        )
        .expect_err("naked with a parameter must be refused");
        assert!(
            error.contains("takes parameters or returns a value"),
            "{error}"
        );
    }

    #[test]
    fn frame_slots_are_addressed_from_rbp_never_below_rsp() {
        let asm = assembly_for(
            "fn @f() -> i64 {\nentry:\n    %p = stackalloc.[24 x i8]\n    ret.i64 0\n}\n",
            false,
            TargetOperatingSystem::Linux,
        )
        .expect("frame function should emit");
        assert!(asm.contains("(%rbp)"), "{asm}");
        assert!(!asm.contains("(%rsp)"), "{asm}");
    }
}
