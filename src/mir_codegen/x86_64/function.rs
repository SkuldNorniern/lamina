use super::*;
use crate::mir_codegen::common::FunctionEmissionPlan;

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

pub(super) fn layout_global_stack_slots(
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

pub(super) fn compile_single_function_x86_64(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    return_extensions: &ReturnExtensions,
) -> Result<Vec<u8>, CodegenError> {
    let reserved = simd::reserve_vector_slots(func);
    let func = reserved.as_ref().unwrap_or(func);
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
        let mut plan = match settings.regalloc {
            RegallocStrategy::LinearScanGlobal => {
                LinearScanAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::GraphColorGlobal => {
                GraphColorAllocator::allocate_with_constraints(&intervals, pool.as_slice(), allowed)
            }
            RegallocStrategy::Incremental => unreachable!(),
        };
        plan = coalesce_dying_inputs(func, &intervals, &plan, allowed);
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

    let param_extensions: Vec<Option<(usize, bool)>> = func
        .sig
        .params
        .iter()
        .map(|param| narrow_param_extension(&param.ty))
        .collect();
    emit_narrow_param_normalization(&mut output, &param_extensions, &self_param_homes)?;

    let plan = FunctionEmissionPlan::new(func);
    let layout = &plan.order;
    let rotations = &plan.rotations;
    if layout.first().map(|&i| func.blocks[i].label.as_str()) != Some(func.entry.as_str()) {
        writeln!(output, "    jmp .L_{}.{}", func_name, func.entry)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    // A comparison whose only consumer is the branch right after it needs no setCC and no
    // materialised 0/1: the flags from cmpq drive jCC directly.
    let use_counts = use_counts(func);
    let flag_producers = &plan.comparisons;

    let loop_headers = &plan.aligned_headers;

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
            ) {
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
            } = inst
                && let Register::Virtual(cmp_dst) = dst
                && let Some(MirInst::Br {
                    cond: Register::Virtual(cond),
                    true_target,
                    false_target,
                }) = block.instructions.get(position + 1)
                && cond == cmp_dst
                && flag_producers.contains(&(block_index, position))
            {
                let scratch =
                    emit_int_compare(op, ty, lhs, rhs, &mut output, &mut reg_alloc, &stack_slots)
                        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                let suffix = cmp_suffix(op)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                if true_target == next_label
                    || rotations
                        .get(&block_index)
                        .is_some_and(|&body| func.blocks[body].label == *false_target)
                {
                    let inverted = inverted_cmp_suffix(op)
                        .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                    writeln!(output, "    j{inverted} .L_{func_name}.{false_target}").map_err(
                        |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                    )?;
                    if true_target != next_label {
                        writeln!(output, "    jmp .L_{func_name}.{true_target}")
                            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
                    }
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
                &def_regs,
                settings,
                &mut debug_line,
                return_extensions,
                next_label,
                &self_param_homes,
                &param_extensions,
                &func.entry,
            )
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
        output.rax_value = None;
    }

    Ok(strip_self_moves(output.inner))
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
    if ty.is_float() || ty.is_vector() {
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

/// Declared return extension of every function in the module that returns a narrow integer.
pub(super) type ReturnExtensions = HashMap<String, NarrowExtension>;

pub(super) fn narrow_param_extension(ty: &MirType) -> Option<NarrowExtension> {
    match ty {
        MirType::Scalar(ScalarType::I1) => Some((1, false)),
        MirType::Scalar(ScalarType::I8) => Some((1, true)),
        MirType::Scalar(ScalarType::I16) => Some((2, true)),
        MirType::Scalar(ScalarType::I32) => Some((4, true)),
        _ => None,
    }
}

/// Rewrite each narrow parameter home in place to its canonical 64-bit form.
pub(super) fn emit_narrow_param_normalization<W: Write>(
    writer: &mut W,
    extensions: &[Option<(usize, bool)>],
    homes: &[Option<String>],
) -> Result<(), CodegenError> {
    let io_error = |e: Error| CodegenError::InvalidCodegenOptions(format!("IO error: {e}"));
    for (extension, home) in extensions.iter().zip(homes) {
        let (Some((width, signed)), Some(home)) = (extension, home) else {
            continue;
        };
        let insn = match (width, signed) {
            (1, false) => "movzbl",
            (1, true) => "movsbq",
            (2, _) => "movswq",
            _ => "movslq",
        };
        if let Some(reg) = home.strip_prefix('%') {
            let source = register_for_size(reg, *width).ok_or_else(|| {
                CodegenError::InvalidCodegenOptions(format!("no alias for register {reg}"))
            })?;
            let dest = if insn == "movzbl" {
                register_for_size(reg, 4).unwrap_or_else(|| reg.to_string())
            } else {
                reg.to_string()
            };
            writeln!(writer, "    {insn} %{source}, %{dest}").map_err(io_error)?;
        } else {
            let dest = if insn == "movzbl" { "eax" } else { "rax" };
            writeln!(writer, "    {insn} {home}, %{dest}").map_err(io_error)?;
            writeln!(writer, "    movq %rax, {home}").map_err(io_error)?;
        }
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[allow(clippy::expect_used)]
mod global_regalloc_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
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
                    "array_operations_comprehensive",
                    "struct_operations_comprehensive",
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
    print %result
    ret.i64 0
}
"#,
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
            let default = if matches!(level, "2" | "3") {
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
    fn padded_struct_storage_preserves_adjacent_allocations_and_frames() {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return;
        }
        let directory = env::temp_dir().join(format!("lamina-padded-storage-{}", process::id()));
        fs::create_dir_all(&directory).expect("create executable directory");
        let source = directory.join("padded.lamina");
        fs::write(
            &source,
            r#"
@noinline
fn @padded() -> i64 {
entry:
    %stack = alloc.stack struct { a: i8, b: i64, c: i8 }
    %other = alloc.stack [8 x i64]
    store.i64 %other, 4321
    %middle = getfieldptr %stack, 1
    %last = getfieldptr %stack, 2
    store.i64 %middle, 1234
    store.i8 %last, 99
    %heap = alloc.heap [2 x struct { a: i8, b: i64, c: i8 }]
    %second = getelementptr %heap, 1, struct { a: i8, b: i64, c: i8 }
    %heap_last = getfieldptr %second, 2
    store.i8 %heap_last, 77
    %x = load.i64 %middle
    %y = load.i8 %last
    %z = load.i8 %heap_last
    %guard = load.i64 %other
    dealloc.heap %heap
    %wide_y = zext.i8.i64 %y
    %wide_z = zext.i8.i64 %z
    %sum = add.i64 %x, %wide_y
    %sum2 = add.i64 %sum, %wide_z
    %result = add.i64 %sum2, %guard
    ret.i64 %result
}
fn @main() -> i64 {
entry:
    %result = call @padded()
    print %result
    ret.i64 0
}
"#,
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
