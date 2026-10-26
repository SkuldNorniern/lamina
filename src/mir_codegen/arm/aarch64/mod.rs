//! AArch64 code generation for MIR (Mid-level IR).
//!
//! Code generation from MIR to AArch64 assembly,
//! following the AAPCS64 calling convention.

mod simd;
mod util;

use crate::mir::register::PhysicalReg;
use crate::mir_codegen::common::{
    CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, integer_conversion, self_tail_arguments, self_tail_schedule,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note};
use crate::mir_codegen::simd::reserve_vector_slots;

use std::{collections::HashMap, io::Write, slice, str, sync::Arc};

use util::{emit_mov_imm64, imm_to_u64};

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, Aggregate, FloatBinOp, FloatCmpOp, FloatUnOp, Function, Global, Immediate,
        Instruction as MirInst, IntBinOp, IntCmpOp, MirType, Module as MirModule, Operand,
        Register, ScalarType, Signature, VirtualReg, register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::CapabilitySet,
        common::{
            CodegenBase, compile_functions_parallel, emit_print_format_section,
            parallel_codegen_error,
        },
        validate_module_call_parameters,
    },
};

use lamina_codegen::os_uses_macho;
use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator,
    aarch64::{A64RegAlloc, AArch64ABI, FrameMap},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Convert an x-register name to its w-register alias (lower 32 bits).
fn w_alias(xreg: &str) -> String {
    if let Some(rest) = xreg.strip_prefix('x') {
        format!("w{rest}")
    } else {
        xreg.to_string()
    }
}

/// Convert a w-register name to its x-register alias (full 64 bits).
fn x_alias(reg: &str) -> String {
    if let Some(rest) = reg.strip_prefix('w') {
        format!("x{rest}")
    } else {
        reg.to_string()
    }
}

/// Conditional branches reach +-1MB. A function whose encoding stays under this
/// bound keeps direct conditional branches; larger ones use inverted branches.
const DIRECT_BRANCH_BYTES: usize = 512 * 1024;

/// Where one argument sits under AAPCS64: `words` consecutive general registers from
/// `first`, or `words` consecutive eight-byte stack slots from `first`. A scalar is one
/// word; an integer aggregate is as many as its size needs.
#[derive(Clone, Copy)]
enum ArgPlace {
    Gpr {
        first: usize,
        words: usize,
    },
    /// A float in v`index`, which aapcs64 draws from a bank of its own.
    Fpr {
        index: usize,
    },
    Stack {
        first: usize,
        words: usize,
    },
}

const ARGUMENT_REGISTERS: usize = 8;

/// One argument as the convention sees it: a float, an integer aggregate, or a plain
/// integer word.
#[derive(Clone, Copy)]
struct ArgKind {
    float: bool,
    ty_is_f32: bool,
    aggregate: Option<Aggregate>,
}

impl ArgKind {
    fn of(ty: MirType, aggregate: Option<Aggregate>) -> Self {
        Self {
            float: aggregate.is_none() && ty.is_float(),
            ty_is_f32: ty.size_bytes() == 4,
            aggregate,
        }
    }
}

/// Floats take the next v register and integers the next x register, from banks that
/// run out separately. An aggregate that does not fit in the registers left goes to
/// the stack whole, and takes the remaining ones out of play.
fn classify_aarch64_arguments(kinds: &[ArgKind]) -> Vec<ArgPlace> {
    let mut gprs = 0;
    let mut fprs = 0;
    let mut stack = 0;
    let mut places = Vec::with_capacity(kinds.len());
    for kind in kinds {
        if kind.float {
            if fprs < ARGUMENT_REGISTERS {
                places.push(ArgPlace::Fpr { index: fprs });
                fprs += 1;
            } else {
                places.push(ArgPlace::Stack {
                    first: stack,
                    words: 1,
                });
                stack += 1;
            }
            continue;
        }
        let words = kind.aggregate.map_or(1, |a| a.words());
        if gprs + words <= ARGUMENT_REGISTERS {
            places.push(ArgPlace::Gpr { first: gprs, words });
            gprs += words;
        } else {
            gprs = ARGUMENT_REGISTERS;
            places.push(ArgPlace::Stack {
                first: stack,
                words,
            });
            stack += words;
        }
    }
    places
}

/// Aggregate markers of each function's boundary, so a call site knows how the callee
/// reads its arguments and hands back its result.
#[derive(Default)]
struct AggregateSignatures {
    params: HashMap<String, Vec<ArgKind>>,
    returns: HashMap<String, Aggregate>,
    float_returns: HashMap<String, MirType>,
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
                            .map(|p| ArgKind::of(p.ty, p.aggregate))
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
        }
    }

    /// An undeclared callee has no signature, so its arguments pass as integers.
    fn arguments(&self, callee: &str, count: usize) -> Vec<ArgKind> {
        match self.params.get(callee) {
            Some(kinds) if kinds.len() == count => kinds.clone(),
            _ => vec![ArgKind::of(MirType::i64(), None); count],
        }
    }

    fn result(&self, callee: &str) -> Option<Aggregate> {
        self.returns.get(callee).copied()
    }

    fn float_result(&self, callee: &str) -> Option<MirType> {
        self.float_returns.get(callee).copied()
    }
}

fn compile_single_function_aarch64(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &AggregateSignatures,
) -> Result<Vec<u8>, CodegenError> {
    let direct = compile_single_function_aarch64_with(
        func_name, func, target_os, settings, signatures, false,
    )?;
    let instructions = direct
        .split(|&b| b == b'\n')
        .filter(|line| {
            let line = str::from_utf8(line).unwrap_or("").trim_start();
            !line.is_empty()
                && !line.starts_with('.')
                && !line.starts_with("//")
                && !line.ends_with(':')
        })
        .count();
    if instructions * 4 < DIRECT_BRANCH_BYTES {
        return Ok(direct);
    }
    compile_single_function_aarch64_with(func_name, func, target_os, settings, signatures, true)
}

fn compile_single_function_aarch64_with(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &AggregateSignatures,
    long_branches: bool,
) -> Result<Vec<u8>, CodegenError> {
    let reserved = reserve_vector_slots(func);
    let func = reserved.as_ref().unwrap_or(func);
    let mut output = Vec::new();
    let abi = AArch64ABI::new(target_os);

    let label = abi.mangle_function_name(func_name);
    writeln!(output, "{label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    let mut ra_pro = A64RegAlloc::new();
    let s0 = ra_pro.alloc_scratch().unwrap_or("x19");
    let s1 = ra_pro.alloc_scratch().unwrap_or("x20");
    if s0 != "x29" || s1 != "x30" {
        writeln!(output, "    stp x29, x30, [sp, #-16]!")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    } else {
        writeln!(output, "    stp {s0}, {s1}, [sp, #-16]!")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }
    writeln!(output, "    mov x29, sp")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    let mut frame = FrameMap::from_function(func);

    let makes_calls = func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .any(|i| matches!(i, MirInst::Call { .. }));
    let global_plan: Option<HashMap<VirtualReg, MirAllocation<&'static str>>> =
        if settings.regalloc != RegallocStrategy::Incremental && !makes_calls {
            let pool = A64RegAlloc::gpr_pool_for_global_allocation();
            let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
                .into_iter()
                .filter(|i| {
                    i.vreg.class == RegisterClass::Gpr
                        && !func.stack_reservations.contains_key(&i.vreg)
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
            for (v, a) in &plan {
                if reserved.is_none()
                    && let MirAllocation::Spill(off) = a
                {
                    frame.slots.insert(Register::Virtual(*v), *off);
                }
            }
            frame.recompute_frame_size_from_slots();
            Some(plan)
        } else {
            None
        };

    let has_many_vars = func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i, MirInst::IntBinary { .. }))
        .count()
        > 100;
    let adjusted_frame_size = if has_many_vars {
        (frame.frame_size + 1024) & !15
    } else {
        frame.frame_size
    };

    frame.frame_size = adjusted_frame_size;
    if adjusted_frame_size > 0 {
        emit_sp_adjust(&mut output, "sub", adjusted_frame_size as u64)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    let param_kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind::of(p.ty, p.aggregate))
        .collect();
    let param_places = classify_aarch64_arguments(&param_kinds);
    for (p, place) in func.sig.params.iter().zip(&param_places) {
        if let Some(off) = frame.slot_of(&p.reg) {
            let addr = ra_pro.alloc_scratch().unwrap_or("x19");
            emit_frame_slot_address(&mut output, addr, off)
                .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;

            match *place {
                ArgPlace::Fpr { index } => {
                    // The slot holds the bits, so the float register is moved through a
                    // general one on its way in.
                    let value = ra_pro.alloc_scratch().unwrap_or("x20");
                    if p.ty.size_bytes() == 4 {
                        writeln!(&mut output, "    fmov {}, s{index}", w_alias(value))
                            .and_then(|()| {
                                writeln!(&mut output, "    str {}, [{addr}]", w_alias(value))
                            })
                            .map_err(|e| {
                                CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                            })?;
                    } else {
                        writeln!(&mut output, "    fmov {value}, d{index}")
                            .and_then(|()| writeln!(&mut output, "    str {value}, [{addr}]"))
                            .map_err(|e| {
                                CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                            })?;
                    }
                    ra_pro.free_scratch(value);
                }
                ArgPlace::Gpr { first, words } => {
                    for word in 0..words {
                        let source = format!("x{}", first + word);
                        if p.aggregate.is_none() {
                            emit_narrow_param_extension(&mut output, &source, &p.ty).map_err(
                                |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                            )?;
                        }
                        writeln!(output, "    str {source}, [{addr}, #{}]", word * 8).map_err(
                            |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                        )?;
                    }
                }
                ArgPlace::Stack { first, words } => {
                    let val_reg = ra_pro.alloc_scratch().unwrap_or("x20");
                    for word in 0..words {
                        let caller_offset = 16 + (first + word) * 8;
                        writeln!(output, "    ldr {val_reg}, [x29, #{caller_offset}]").map_err(
                            |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                        )?;
                        if p.aggregate.is_none() {
                            emit_narrow_param_extension(&mut output, val_reg, &p.ty).map_err(
                                |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                            )?;
                        }
                        writeln!(output, "    str {val_reg}, [{addr}, #{}]", word * 8).map_err(
                            |e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")),
                        )?;
                    }
                    ra_pro.free_scratch(val_reg);
                }
            }
            ra_pro.free_scratch(addr);
        }
    }

    let epilogue_label = format!(".Lret_{}", label.trim_start_matches('_'));
    let block_prefix = format!(".L_{}.", label.trim_start_matches('_'));

    let has_complex_function =
        func.blocks.len() > 50 || func.blocks.iter().any(|b| b.instructions.len() > 100);

    let mut ra = if let Some(ref plan) = global_plan {
        A64RegAlloc::from_global_plan(plan)
    } else {
        let mut r = A64RegAlloc::new();
        if has_complex_function {
            r.set_conservative_mode();
        }
        r
    };

    if let Some(global_plan) = &global_plan {
        for param in &func.sig.params {
            if let Register::Virtual(vreg) = &param.reg
                && let Some(MirAllocation::Register(reg)) = global_plan.get(vreg)
                && let Some(offset) = frame.slot_of(&param.reg)
            {
                let initialize = |output: &mut Vec<u8>| -> Result<(), LaminaError> {
                    if (-256..=255).contains(&offset) {
                        writeln!(output, "    ldur {reg}, [x29, #{offset}]")?;
                    } else {
                        emit_frame_slot_address(output, "x16", offset)?;
                        writeln!(output, "    ldr {reg}, [x16]")?;
                    }
                    Ok(())
                };
                initialize(&mut output)
                    .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
            }
        }
    }

    let plan = FunctionEmissionPlan::new(func);
    let mut debug_line: u32 = 0;
    if plan
        .order
        .first()
        .is_some_and(|&i| func.blocks[i].label != func.entry)
    {
        writeln!(output, "    b {block_prefix}{}", func.entry)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    for (slot, &index) in plan.order.iter().enumerate() {
        let block = &func.blocks[index];
        let next = plan
            .order
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        let emit = |output: &mut Vec<u8>,
                    ra: &mut A64RegAlloc,
                    debug_line: &mut u32|
         -> Result<(), LaminaError> {
            if plan.aligned_headers.contains(&index) {
                writeln!(output, "    .p2align 4,,12")?;
            }
            writeln!(output, "{block_prefix}{}:", block.label)?;
            let mut position = 0;
            while position < block.instructions.len() {
                let inst = &block.instructions[position];
                if let Some(pairs) = self_tail_arguments(func, inst) {
                    let schedule = self_tail_schedule(&pairs, |reg| match reg {
                        Register::Virtual(v) => ra
                            .get_mapping_for(v)
                            .map(|name| {
                                Register::Physical(PhysicalReg::new(name, RegisterClass::Gpr))
                            })
                            .unwrap_or_else(|| reg.clone()),
                        _ => reg.clone(),
                    });
                    for (src, dst) in schedule {
                        let src = match src {
                            CopyLocation::Value(src) => src,
                            CopyLocation::Scratch => Operand::Register(Register::Physical(
                                PhysicalReg::new("x17", RegisterClass::Gpr),
                            )),
                        };
                        let dst = match dst {
                            CopyLocation::Value(Operand::Register(dst)) => dst,
                            CopyLocation::Scratch => {
                                Register::Physical(PhysicalReg::new("x17", RegisterClass::Gpr))
                            }
                            _ => unreachable!(),
                        };
                        emit_materialize_operand(output, &src, "x16", &frame, ra)?;
                        store_result(output, &dst, "x16", &frame, ra)?;
                    }
                    if func.entry != next {
                        writeln!(output, "    b {block_prefix}{}", func.entry)?;
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
                        emit_materialize_operand(output, lhs, "x16", &frame, ra)?;
                        emit_materialize_operand(output, rhs, "x17", &frame, ra)?;
                        let unsigned = matches!(
                            op,
                            IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                        );
                        extend_integer(output, "x16", ty, !unsigned)?;
                        extend_integer(output, "x17", ty, !unsigned)?;
                        writeln!(output, "    cmp x16, x17")?;
                        let (invert, taken, other) =
                            plan.branch(func, index, next, true_target, false_target);
                        let cond = match (op, invert) {
                            (IntCmpOp::Eq, false) | (IntCmpOp::Ne, true) => "eq",
                            (IntCmpOp::Ne, false) | (IntCmpOp::Eq, true) => "ne",
                            (IntCmpOp::SLt, false) | (IntCmpOp::SGe, true) => "lt",
                            (IntCmpOp::SLe, false) | (IntCmpOp::SGt, true) => "le",
                            (IntCmpOp::SGt, false) | (IntCmpOp::SLe, true) => "gt",
                            (IntCmpOp::SGe, false) | (IntCmpOp::SLt, true) => "ge",
                            (IntCmpOp::ULt, false) | (IntCmpOp::UGe, true) => "lo",
                            (IntCmpOp::ULe, false) | (IntCmpOp::UGt, true) => "ls",
                            (IntCmpOp::UGt, false) | (IntCmpOp::ULe, true) => "hi",
                            (IntCmpOp::UGe, false) | (IntCmpOp::ULt, true) => "hs",
                        };
                        emit_branch_if(
                            output,
                            cond,
                            &format!("{block_prefix}{taken}"),
                            long_branches,
                        )?;
                        if let Some(other) = other {
                            writeln!(output, "    b {block_prefix}{other}")?;
                        }
                    }
                    position += 1;
                } else if matches!(inst, MirInst::Jmp { target } if target == next) {
                } else {
                    emit_block(
                        slice::from_ref(inst),
                        output,
                        &frame,
                        &abi,
                        ra,
                        &epilogue_label,
                        &block_prefix,
                        settings,
                        debug_line,
                        long_branches,
                        signatures,
                        func.sig.ret_aggregate,
                        func.sig.ret_ty.filter(|ty| ty.is_float()),
                    )?;
                }
                position += 1;
            }
            Ok(())
        };
        emit(&mut output, &mut ra, &mut debug_line)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }

    writeln!(output, "{epilogue_label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    if adjusted_frame_size > 0 {
        emit_sp_adjust(&mut output, "add", adjusted_frame_size as u64)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    writeln!(output, "    ldp x29, x30, [sp], #16")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    writeln!(output, "    ret")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    Ok(output)
}

/// Generate AArch64 assembly from a MIR module.
pub fn generate_mir_aarch64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_aarch64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_aarch64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    generate_mir_aarch64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_aarch64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::Aarch64)?;
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(AggregateSignatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function_aarch64(
                name,
                func,
                os,
                settings_arc.as_ref(),
                signatures.as_ref(),
            )
        })
        .map_err(parallel_codegen_error)?;

    emit_print_format_section(writer, target_os)?;
    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings)?;

    let abi = AArch64ABI::new(target_os);

    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }
    emit_global_data(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;
    emit_gnu_stack_note(writer, target_os)?;

    Ok(())
}

/// Emit assembly for a sequence of MIR instructions.
#[allow(clippy::too_many_arguments)]
fn emit_block<W: Write>(
    insts: &[MirInst],
    w: &mut W,
    frame: &FrameMap,
    abi: &AArch64ABI,
    ra: &mut A64RegAlloc,
    epilogue_label: &str,
    block_prefix: &str,
    settings: &MirCodegenSettings,
    debug_line: &mut u32,
    long_branches: bool,
    signatures: &AggregateSignatures,
    ret_aggregate: Option<Aggregate>,
    ret_float: Option<MirType>,
) -> Result<(), LaminaError> {
    for inst in insts {
        if settings.emit_asm_debug_lines {
            *debug_line = debug_line.saturating_add(1);
            writeln!(w, "    .loc 1 {} 0", *debug_line)?;
        }
        match inst {
            MirInst::SymbolAddress { dst, symbol } => {
                let label = abi.mangle_function_name(symbol);
                let scratch = ra.alloc_scratch().unwrap_or("x19");
                if os_uses_macho(abi.target_os()) {
                    writeln!(w, "    adrp {scratch}, {label}@PAGE")?;
                    writeln!(w, "    add {scratch}, {scratch}, {label}@PAGEOFF")?;
                } else {
                    writeln!(w, "    adrp {scratch}, {label}")?;
                    writeln!(w, "    add {scratch}, {scratch}, :lo12:{label}")?;
                }
                store_result(w, dst, &x_alias(scratch), frame, ra)?;
                ra.free_scratch(scratch);
            }
            MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
                simd::emit(inst, w, frame, ra)?
            }
            MirInst::IntConvert { .. } => {
                if let Some(operations) = integer_conversion(inst) {
                    emit_block(
                        &operations,
                        w,
                        frame,
                        abi,
                        ra,
                        epilogue_label,
                        block_prefix,
                        settings,
                        debug_line,
                        long_branches,
                        signatures,
                        ret_aggregate,
                        ret_float,
                    )?;
                }
            }

            MirInst::Copy { ty, dst, src }
            | MirInst::Bitcast {
                to: ty, dst, src, ..
            } => {
                if matches!(ty, MirType::Vector(_)) {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!("AArch64 backend does not support vector Copy of type {ty}"),
                    )));
                }
                let scratch = ra.alloc_scratch().unwrap_or("x19");
                let value = x_alias(scratch);
                match src {
                    Operand::Immediate(immediate) => {
                        emit_mov_imm64(w, &value, imm_to_u64(immediate))?
                    }
                    Operand::Register(register) => {
                        load_copy_reg_to(w, register, &value, ty.size_bytes(), frame, ra)?
                    }
                }
                extend_integer(w, &value, ty, true)?;
                store_result(w, dst, &value, frame, ra)?;
                ra.free_scratch(scratch);
            }
            MirInst::IntBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                    extend_integer(w, s_l, ty, false)?;
                    extend_integer(w, s_r, ty, false)?;
                }
                let dl = x_alias(s_d);
                let rl = x_alias(s_l);
                let rr = x_alias(s_r);
                let (zero, overflow) =
                    op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant());
                if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                    extend_integer(w, s_l, ty, true)?;
                    if !matches!(op, IntBinOp::AShr) {
                        extend_integer(w, s_r, ty, true)?;
                    }
                }
                if zero {
                    writeln!(w, "    cbnz {rr}, 1f\n    udf #0\n1:")?;
                }
                if overflow {
                    let minimum = 1u64 << (ty.size_bytes() * 8 - 1);
                    writeln!(w, "    cmn {rr}, #1\n    b.ne 2f")?;
                    emit_mov_imm64(w, &dl, ty.signed_integer(minimum as i64) as u64)?;
                    writeln!(w, "    cmp {rl}, {dl}\n    b.ne 2f\n    udf #0\n2:")?;
                }
                if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr)
                    && ty.size_bytes() < 8
                {
                    let mask = ty.size_bytes() * 8 - 1;
                    writeln!(w, "    and {rr}, {rr}, #{mask}")?;
                }
                match op {
                    IntBinOp::Add => {
                        writeln!(w, "    add {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::Sub => {
                        writeln!(w, "    sub {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::Mul => {
                        writeln!(w, "    mul {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::UDiv => {
                        writeln!(w, "    udiv {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::SDiv => {
                        writeln!(w, "    sdiv {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::URem => {
                        writeln!(w, "    udiv {dl}, {rl}, {rr}")?;
                        writeln!(w, "    msub {dl}, {dl}, {rr}, {rl}")?;
                    }
                    IntBinOp::SRem => {
                        writeln!(w, "    sdiv {dl}, {rl}, {rr}")?;
                        writeln!(w, "    msub {dl}, {dl}, {rr}, {rl}")?;
                    }
                    IntBinOp::And => {
                        writeln!(w, "    and {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::Or => {
                        writeln!(w, "    orr {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::Xor => {
                        writeln!(w, "    eor {dl}, {rl}, {rr}")?;
                    }
                    IntBinOp::Shl => match rhs {
                        Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= ty.size_bytes() as u32 * 8 - 1;
                            writeln!(w, "    lsl {dl}, {rl}, #{sh}")?;
                        }
                        _ => {
                            writeln!(w, "    lslv {dl}, {rl}, {rr}")?;
                        }
                    },
                    IntBinOp::LShr => match rhs {
                        Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= ty.size_bytes() as u32 * 8 - 1;
                            writeln!(w, "    lsr {dl}, {rl}, #{sh}")?;
                        }
                        _ => {
                            writeln!(w, "    lsrv {dl}, {rl}, {rr}")?;
                        }
                    },
                    IntBinOp::AShr => match rhs {
                        Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= ty.size_bytes() as u32 * 8 - 1;
                            writeln!(w, "    asr {dl}, {rl}, #{sh}")?;
                        }
                        _ => {
                            writeln!(w, "    asrv {dl}, {rl}, {rr}")?;
                        }
                    },
                }
                extend_integer(w, s_d, ty, true)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes() == 4;
                let suffix = if is32 { "s" } else { "d" };

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_l))?;
                    writeln!(w, "    fmov s1, {}", w_alias(s_r))?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_l))?;
                    writeln!(w, "    fmov d1, {}", x_alias(s_r))?;
                }

                match op {
                    FloatBinOp::FAdd => writeln!(w, "    fadd {suffix}0, {suffix}0, {suffix}1")?,
                    FloatBinOp::FSub => writeln!(w, "    fsub {suffix}0, {suffix}0, {suffix}1")?,
                    FloatBinOp::FMul => writeln!(w, "    fmul {suffix}0, {suffix}0, {suffix}1")?,
                    FloatBinOp::FDiv => writeln!(w, "    fdiv {suffix}0, {suffix}0, {suffix}1")?,
                }

                if is32 {
                    writeln!(w, "    fmov {}, s0", w_alias(s_d))?;
                } else {
                    writeln!(w, "    fmov {}, d0", x_alias(s_d))?;
                }

                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatUnary { op, src, dst, ty } => {
                let s_s = ra.alloc_scratch().unwrap_or("x19");
                let s_d = ra.alloc_scratch().unwrap_or("x20");
                emit_materialize_operand(w, src, s_s, frame, ra)?;

                let is32 = ty.size_bytes() == 4;
                let suffix = if is32 { "s" } else { "d" };

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_s))?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_s))?;
                }

                match op {
                    FloatUnOp::FNeg => writeln!(w, "    fneg {suffix}0, {suffix}0")?,
                    FloatUnOp::FSqrt => writeln!(w, "    fsqrt {suffix}0, {suffix}0")?,
                }

                if is32 {
                    writeln!(w, "    fmov {}, s0", w_alias(s_d))?;
                } else {
                    writeln!(w, "    fmov {}, d0", x_alias(s_d))?;
                }

                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_s);
                ra.free_scratch(s_d);
            }
            MirInst::IntCmp {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                if matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                ) {
                    extend_integer(w, s_l, ty, false)?;
                    extend_integer(w, s_r, ty, false)?;
                }
                writeln!(w, "    cmp {}, {}", x_alias(s_l), x_alias(s_r))?;
                let cond = match op {
                    IntCmpOp::Eq => "eq",
                    IntCmpOp::Ne => "ne",
                    IntCmpOp::SLt => "lt",
                    IntCmpOp::SLe => "le",
                    IntCmpOp::SGt => "gt",
                    IntCmpOp::SGe => "ge",
                    IntCmpOp::ULt => "lo",
                    IntCmpOp::ULe => "ls",
                    IntCmpOp::UGt => "hi",
                    IntCmpOp::UGe => "hs",
                };
                writeln!(w, "    cset {}, {}", x_alias(s_d), cond)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatCmp {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes() == 4;

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_l))?;
                    writeln!(w, "    fmov s1, {}", w_alias(s_r))?;
                    writeln!(w, "    fcmp s0, s1")?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_l))?;
                    writeln!(w, "    fmov d1, {}", x_alias(s_r))?;
                    writeln!(w, "    fcmp d0, d1")?;
                }

                let cond = match op {
                    FloatCmpOp::Eq => "eq",
                    FloatCmpOp::Ne => "ne",
                    FloatCmpOp::Lt => "mi",
                    FloatCmpOp::Le => "ls",
                    FloatCmpOp::Gt => "gt",
                    FloatCmpOp::Ge => "ge",
                };

                writeln!(w, "    cset {}, {}", x_alias(s_d), cond)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::Load { ty, dst, addr, .. } => {
                let a = ra.alloc_scratch().unwrap_or("x19");
                let t = ra.alloc_scratch().unwrap_or("x20");
                materialize_address(w, addr, a, frame, ra)?;
                match ty.size_bytes() {
                    1 => writeln!(w, "    ldrsb {}, [{}]", x_alias(t), a)?,
                    2 => writeln!(w, "    ldrsh {}, [{}]", x_alias(t), a)?,
                    4 => writeln!(w, "    ldrsw {}, [{}]", x_alias(t), a)?,
                    8 => writeln!(w, "    ldr {}, [{}]", x_alias(t), a)?,
                    _ => writeln!(w, "    // Unhandled load size: {}", ty.size_bytes())?,
                }
                store_result(w, dst, &x_alias(t), frame, ra)?;
                ra.free_scratch(a);
                ra.free_scratch(t);
            }
            MirInst::Select {
                ty,
                dst,
                cond,
                true_val,
                false_val,
            } => {
                let r_cond = ra.alloc_scratch().unwrap_or("x19");
                let r_t = ra.alloc_scratch().unwrap_or("x20");
                let r_f = ra.alloc_scratch().unwrap_or("x21");
                load_reg_to(w, cond, r_cond, frame, ra)?;
                emit_materialize_operand(w, true_val, r_t, frame, ra)?;
                emit_materialize_operand(w, false_val, r_f, frame, ra)?;
                if ty.size_bytes() == 4 {
                    writeln!(w, "    cmp {}, #0", w_alias(r_cond))?;
                    writeln!(
                        w,
                        "    csel {}, {}, {}, ne",
                        w_alias(r_t),
                        w_alias(r_t),
                        w_alias(r_f)
                    )?;
                    store_result(w, dst, &x_alias(r_t), frame, ra)?;
                } else {
                    writeln!(w, "    cmp {}, #0", x_alias(r_cond))?;
                    writeln!(
                        w,
                        "    csel {}, {}, {}, ne",
                        x_alias(r_t),
                        x_alias(r_t),
                        x_alias(r_f)
                    )?;
                    store_result(w, dst, &x_alias(r_t), frame, ra)?;
                }
                ra.free_scratch(r_cond);
                ra.free_scratch(r_t);
                ra.free_scratch(r_f);
            }
            MirInst::Store { ty, src, addr, .. } => {
                let a = ra.alloc_scratch().unwrap_or("x19");
                let t = ra.alloc_scratch().unwrap_or("x20");
                materialize_address(w, addr, a, frame, ra)?;
                emit_materialize_operand(w, src, t, frame, ra)?;
                match ty.size_bytes() {
                    1 => writeln!(w, "    strb {}, [{}]", w_alias(t), a)?,
                    2 => writeln!(w, "    strh {}, [{}]", w_alias(t), a)?,
                    4 => writeln!(w, "    str {}, [{}]", w_alias(t), a)?,
                    8 => writeln!(w, "    str {}, [{}]", x_alias(t), a)?,
                    _ => writeln!(w, "    // Unhandled store size: {}", ty.size_bytes())?,
                }
                ra.free_scratch(a);
                ra.free_scratch(t);
            }
            MirInst::StackAddr { dst, slot } => {
                let offset = frame.slot_of(&Register::Virtual(*slot)).ok_or_else(|| {
                    LaminaError::ValidationError("stack address has no frame slot".into())
                })?;
                let t = ra.alloc_scratch().unwrap_or("x19");
                emit_mov_imm64(w, t, i64::from(offset) as u64)?;
                writeln!(w, "    add {t}, x29, {t}")?;
                store_result(w, dst, t, frame, ra)?;
                ra.free_scratch(t);
            }
            MirInst::Lea { dst, base, offset } => {
                let t = ra.alloc_scratch().unwrap_or("x19");
                match base {
                    // A placeholder's slot is the storage itself, so its
                    // address is the value; any other register holds a pointer
                    // that has to be loaded first.
                    Register::Virtual(_) if !frame.holds_value(base) => {
                        if let Some(slot_off) = frame.slot_of(base) {
                            // Compute address: x29 + slot_off + offset
                            let total = slot_off as i64 + (*offset as i64);
                            if (0..=4095).contains(&total) {
                                writeln!(w, "    add {t}, x29, #{total}")?;
                            } else if total < 0 && -total <= 4095 {
                                writeln!(w, "    sub {}, x29, #{}", t, -total)?;
                            } else {
                                emit_mov_imm64(w, t, total as u64)?;
                                writeln!(w, "    add {t}, x29, {t}")?;
                            }
                        } else {
                            // No stack slot: load value from physical reg and add offset
                            load_reg_to(w, base, t, frame, ra)?;
                            if *offset != 0 {
                                let off = *offset as i64;
                                if (0..=4095).contains(&off) {
                                    writeln!(w, "    add {t}, {t}, #{off}")?;
                                } else if off < 0 && -off <= 4095 {
                                    writeln!(w, "    sub {}, {}, #{}", t, t, -off)?;
                                } else {
                                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                                    emit_mov_imm64(w, oreg, off as u64)?;
                                    writeln!(w, "    add {t}, {t}, {oreg}")?;
                                    ra.free_scratch(oreg);
                                }
                            }
                        }
                    }
                    _ => {
                        // Physical register: load value and add offset
                        load_reg_to(w, base, t, frame, ra)?;
                        if *offset != 0 {
                            let off = *offset as i64;
                            if (0..=4095).contains(&off) {
                                writeln!(w, "    add {t}, {t}, #{off}")?;
                            } else if off < 0 && -off <= 4095 {
                                writeln!(w, "    sub {}, {}, #{}", t, t, -off)?;
                            } else {
                                let oreg = ra.alloc_scratch().unwrap_or("x20");
                                emit_mov_imm64(w, oreg, off as u64)?;
                                writeln!(w, "    add {t}, {t}, {oreg}")?;
                                ra.free_scratch(oreg);
                            }
                        }
                    }
                }
                store_result(w, dst, t, frame, ra)?;
                ra.free_scratch(t);
            }
            MirInst::IntToFloat {
                from,
                to,
                signed,
                dst,
                src,
            } => emit_int_to_float(w, from, to, *signed, dst, src, frame, ra)?,
            MirInst::FloatToInt {
                from,
                to,
                signed,
                dst,
                src,
            } => emit_float_to_int(w, from, to, *signed, dst, src, frame, ra)?,
            MirInst::Call { name, args, ret } => {
                if name == "print" && args.len() == 1 {
                    // Special-case intrinsic: print(integer) via printf("%lld\n", value)
                    emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            // Darwin AArch64 variadic ABI requires arguments to be available in the stack home area
                            // Ensure 16-byte alignment and spill the first vararg to stack
                            writeln!(w, "    sub sp, sp, #32")?; // create home area
                            writeln!(w, "    adrp x0, .L_mir_fmt_int@PAGE")?;
                            writeln!(w, "    add x0, x0, .L_mir_fmt_int@PAGEOFF")?;
                            writeln!(w, "    str x1, [sp]")?; // spill the vararg as required by ABI
                            writeln!(w, "    bl _printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(w, "    mov x0, #0")?; // NULL flushes all streams
                            writeln!(w, "    bl _fflush")?;
                            writeln!(w, "    add sp, sp, #32")?; // restore stack
                        }
                        _ => {
                            writeln!(w, "    adrp x0, .L_mir_fmt_int")?;
                            writeln!(w, "    add x0, x0, :lo12:.L_mir_fmt_int")?;
                            writeln!(w, "    bl printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(w, "    mov x0, #0")?; // NULL flushes all streams
                            writeln!(w, "    bl fflush")?;
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else if (name == "printf32" || name == "printf64") && args.len() == 1 {
                    emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                    if name == "printf32" {
                        writeln!(w, "    fmov s0, w1")?;
                        writeln!(w, "    fcvt d0, s0")?;
                        writeln!(w, "    fmov x1, d0")?;
                    }
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            writeln!(w, "    sub sp, sp, #32")?;
                            writeln!(w, "    adrp x0, .L_mir_fmt_float@PAGE")?;
                            writeln!(w, "    add x0, x0, .L_mir_fmt_float@PAGEOFF")?;
                            writeln!(w, "    str x1, [sp]")?;
                            writeln!(w, "    bl _printf")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    bl _fflush")?;
                            writeln!(w, "    add sp, sp, #32")?;
                        }
                        _ => {
                            writeln!(w, "    fmov d0, x1")?;
                            writeln!(w, "    adrp x0, .L_mir_fmt_float")?;
                            writeln!(w, "    add x0, x0, :lo12:.L_mir_fmt_float")?;
                            writeln!(w, "    bl printf")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    bl fflush")?;
                        }
                    }
                } else if name == "writebyte" && args.len() == 1 {
                    // Write a single byte to stdout using macOS ARM64 syscall
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            writeln!(w, "    sub sp, sp, #16")?;
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl write")?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                    }
                } else if name == "readbyte" && args.is_empty() {
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            writeln!(w, "    ldrb {}, [sp]", w_alias("x9"))?;
                            if let Some(dst) = ret {
                                let dst_reg = ra.alloc_scratch().unwrap_or("x10");
                                writeln!(w, "    cmp x0, #1")?;
                                writeln!(w, "    uxtb {}, {}", x_alias("x9"), w_alias("x9"))?;
                                writeln!(
                                    w,
                                    "    csel {}, {}, x0, eq",
                                    x_alias(dst_reg),
                                    x_alias("x9")
                                )?;
                                store_result(w, dst, &x_alias(dst_reg), frame, ra)?;
                                ra.free_scratch(dst_reg);
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            // Fallback to libc read
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl read")?;
                            writeln!(w, "    ldrb {}, [sp]", w_alias("x9"))?;
                            if let Some(dst) = ret {
                                let dst_reg = ra.alloc_scratch().unwrap_or("x10");
                                writeln!(w, "    cmp x0, #1")?;
                                writeln!(w, "    uxtb {}, {}", x_alias("x9"), w_alias("x9"))?;
                                writeln!(
                                    w,
                                    "    csel {}, {}, x0, eq",
                                    x_alias(dst_reg),
                                    x_alias("x9")
                                )?;
                                store_result(w, dst, &x_alias(dst_reg), frame, ra)?;
                                ra.free_scratch(dst_reg);
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                    }
                } else if name == "writeptr" && args.len() == 2 {
                    // write(1, ptr, width): the bytes go out straight from memory.
                    let Operand::Immediate(Immediate::I64(width)) = &args[1] else {
                        return Err(LaminaError::ValidationError(
                            "writeptr width must be an immediate".to_string(),
                        ));
                    };
                    emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                    writeln!(w, "    mov x2, #{width}")?;
                    writeln!(w, "    mov x0, #1")?;
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                        }
                        _ => {
                            writeln!(w, "    bl write")?;
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else if name == "write" && args.len() == 2 {
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            // write(fd=1, buf=args[0], size=args[1])
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    bl write")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                    }
                } else if name == "read" && args.len() == 2 {
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            // read(fd=0, buf=args[0], size=args[1])
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    bl read")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                    }
                } else {
                    let kinds = signatures.arguments(name, args.len());
                    let places = classify_aarch64_arguments(&kinds);
                    for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                        match *place {
                            ArgPlace::Fpr { index } => {
                                // Values are held as bits in general registers, so a
                                // float argument moves across on the way out.
                                let bits = ra.alloc_scratch().unwrap_or("x9");
                                emit_materialize_operand(w, argument, bits, frame, ra)?;
                                if kind.ty_is_f32 {
                                    writeln!(w, "    fmov s{index}, {}", w_alias(bits))?;
                                } else {
                                    writeln!(w, "    fmov d{index}, {bits}")?;
                                }
                                ra.free_scratch(bits);
                            }
                            ArgPlace::Gpr { first, words } => {
                                if kind.aggregate.is_none() {
                                    emit_materialize_operand(
                                        w,
                                        argument,
                                        &format!("x{first}"),
                                        frame,
                                        ra,
                                    )?;
                                    continue;
                                }
                                let address = ra.alloc_scratch().unwrap_or("x9");
                                emit_materialize_operand(w, argument, address, frame, ra)?;
                                for word in 0..words {
                                    writeln!(
                                        w,
                                        "    ldr x{}, [{address}, #{}]",
                                        first + word,
                                        word * 8
                                    )?;
                                }
                                ra.free_scratch(address);
                            }
                            ArgPlace::Stack { .. } => {}
                        }
                    }

                    let stack_words: usize = places
                        .iter()
                        .map(|place| match place {
                            ArgPlace::Stack { words, .. } => *words,
                            ArgPlace::Gpr { .. } | ArgPlace::Fpr { .. } => 0,
                        })
                        .sum();
                    let stack_space = (stack_words * 8 + 15) & !15;

                    if stack_space > 0 {
                        emit_sp_adjust(w, "sub", stack_space as u64)?;
                        for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                            let ArgPlace::Stack { first, words } = *place else {
                                continue;
                            };
                            let scratch = ra.alloc_scratch().unwrap_or("x9");
                            if kind.aggregate.is_none() {
                                emit_materialize_operand(w, argument, scratch, frame, ra)?;
                                writeln!(w, "    str {scratch}, [sp, #{}]", first * 8)?;
                            } else {
                                let address = ra.alloc_scratch().unwrap_or("x10");
                                emit_materialize_operand(w, argument, address, frame, ra)?;
                                for word in 0..words {
                                    writeln!(w, "    ldr {scratch}, [{address}, #{}]", word * 8)?;
                                    writeln!(
                                        w,
                                        "    str {scratch}, [sp, #{}]",
                                        (first + word) * 8
                                    )?;
                                }
                                ra.free_scratch(address);
                            }
                            ra.free_scratch(scratch);
                        }
                    }

                    let target_sym: String = abi
                        .call_stub(name)
                        .unwrap_or_else(|| abi.mangle_function_name(name));
                    writeln!(w, "    mov x8, xzr")?;
                    writeln!(w, "    bl {target_sym}")?;

                    if stack_space > 0 {
                        emit_sp_adjust(w, "add", stack_space as u64)?;
                    }

                    if let Some(dst) = ret
                        && let Some(ty) = signatures.float_result(name)
                    {
                        // The result arrives in v0 and the body wants its bits.
                        if ty.size_bytes() == 4 {
                            writeln!(w, "    fmov w0, s0")?;
                        } else {
                            writeln!(w, "    fmov x0, d0")?;
                        }
                        store_result(w, dst, "x0", frame, ra)?;
                    } else if let Some(dst) = ret {
                        match signatures.result(name) {
                            // The result register names the slot the words land in; the
                            // body addresses it with a stack address of its own.
                            Some(aggregate) => {
                                let off = frame.slot_of(dst).ok_or_else(|| {
                                    LaminaError::ValidationError(
                                        "aggregate result has no frame slot".to_string(),
                                    )
                                })?;
                                let addr = ra.alloc_scratch().unwrap_or("x9");
                                emit_frame_slot_address(w, addr, off)?;
                                writeln!(w, "    str x0, [{addr}]")?;
                                if aggregate.words() > 1 {
                                    writeln!(w, "    str x1, [{addr}, #8]")?;
                                }
                                ra.free_scratch(addr);
                            }
                            None => store_result(w, dst, "x0", frame, ra)?,
                        }
                    }
                }
            }
            MirInst::TailCall { name, args } => {
                let kinds = signatures.arguments(name, args.len());
                let places = classify_aarch64_arguments(&kinds);
                for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                    match *place {
                        ArgPlace::Fpr { index } => {
                            let bits = ra.alloc_scratch().unwrap_or("x9");
                            emit_materialize_operand(w, argument, bits, frame, ra)?;
                            if kind.ty_is_f32 {
                                writeln!(w, "    fmov s{index}, {}", w_alias(bits))?;
                            } else {
                                writeln!(w, "    fmov d{index}, {bits}")?;
                            }
                            ra.free_scratch(bits);
                        }
                        ArgPlace::Gpr { first, .. } => {
                            emit_materialize_operand(w, argument, &format!("x{first}"), frame, ra)?;
                        }
                        ArgPlace::Stack { first, .. } => {
                            let scratch = ra.alloc_scratch().unwrap_or("x9");
                            emit_materialize_operand(w, argument, scratch, frame, ra)?;
                            writeln!(w, "    str {scratch}, [x29, #{}]", 16 + first * 8)?;
                            ra.free_scratch(scratch);
                        }
                    }
                }
                let target_sym: String = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                if frame.frame_size > 0 {
                    emit_sp_adjust(w, "add", frame.frame_size as u64)?;
                }
                writeln!(w, "    ldp x29, x30, [sp], #16")?;
                writeln!(w, "    b {target_sym}")?;
            }
            MirInst::Ret { value } => {
                if let Some(v) = value
                    && ret_aggregate.is_none()
                    && let Some(ty) = ret_float
                {
                    let bits = ra.alloc_scratch().unwrap_or("x9");
                    emit_materialize_operand(w, v, bits, frame, ra)?;
                    if ty.size_bytes() == 4 {
                        writeln!(w, "    fmov s0, {}", w_alias(bits))?;
                    } else {
                        writeln!(w, "    fmov d0, {bits}")?;
                    }
                    ra.free_scratch(bits);
                    writeln!(w, "    b {epilogue_label}")?;
                    continue;
                }
                if let Some(v) = value {
                    match ret_aggregate {
                        // The returned operand addresses the aggregate; the words
                        // themselves go back in x0 and x1.
                        Some(aggregate) => {
                            let addr = ra.alloc_scratch().unwrap_or("x9");
                            emit_materialize_operand(w, v, addr, frame, ra)?;
                            if aggregate.words() > 1 {
                                writeln!(w, "    ldr x1, [{addr}, #8]")?;
                            }
                            writeln!(w, "    ldr x0, [{addr}]")?;
                            ra.free_scratch(addr);
                        }
                        None => emit_materialize_operand(w, v, "x0", frame, ra)?,
                    }
                }
                writeln!(w, "    b {epilogue_label}")?;
            }
            MirInst::Jmp { target } => {
                writeln!(w, "    b {block_prefix}{target}")?;
            }
            MirInst::Br {
                cond,
                true_target,
                false_target,
            } => {
                let t = ra.alloc_scratch().unwrap_or("x19");
                load_reg_to(w, cond, t, frame, ra)?;
                writeln!(w, "    cmp {}, #0", x_alias(t))?;
                emit_branch_if(
                    w,
                    "ne",
                    &format!("{block_prefix}{true_target}"),
                    long_branches,
                )?;
                writeln!(w, "    b {block_prefix}{false_target}")?;
                ra.free_scratch(t);
            }
            MirInst::Switch {
                value,
                cases,
                default,
            } => {
                let t = ra.alloc_scratch().unwrap_or("x19");
                load_reg_to(w, value, t, frame, ra)?;
                for (case_val, case_label) in cases {
                    emit_mov_imm64(w, "x9", *case_val as u64)?;
                    writeln!(w, "    cmp {}, x9", x_alias(t))?;
                    emit_branch_if(
                        w,
                        "eq",
                        &format!("{block_prefix}{case_label}"),
                        long_branches,
                    )?;
                }
                writeln!(w, "    b {block_prefix}{default}")?;
                ra.free_scratch(t);
            }
            MirInst::Comment { text } => {
                writeln!(w, "    // {text}")?;
            }
            MirInst::Unreachable => {
                // Emit an undefined-instruction trap so the CPU halts immediately if reached.
                writeln!(w, "    udf #0")?;
            }
            MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {
                // No-op in AOT path — only meaningful for JIT/GC runtimes.
            }
            MirInst::VectorOp { .. } => simd::emit(inst, w, frame, ra)?,
            #[cfg(feature = "nightly")]
            MirInst::SimdBinary { .. }
            | MirInst::SimdUnary { .. }
            | MirInst::SimdTernary { .. }
            | MirInst::SimdShuffle { .. }
            | MirInst::SimdExtract { .. }
            | MirInst::SimdInsert { .. }
            | MirInst::SimdLoad { .. }
            | MirInst::SimdStore { .. }
            | MirInst::AtomicLoad { .. }
            | MirInst::AtomicStore { .. }
            | MirInst::AtomicBinary { .. }
            | MirInst::AtomicCompareExchange { .. }
            | MirInst::Fence { .. } => {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    "SIMD/Atomic instructions not yet supported by AArch64 backend".to_string(),
                )));
            }
        }
    }
    Ok(())
}

/// Materialize an operand into a register.
fn emit_materialize_operand<W: Write>(
    w: &mut W,
    op: &Operand,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    match op {
        Operand::Immediate(imm) => emit_mov_imm64(w, dest, imm_to_u64(imm))?,
        Operand::Register(r) => load_reg_to(w, r, dest, frame, ra)?,
    }
    Ok(())
}

/// Load a register value into a destination register.
fn load_reg_to<W: Write>(
    w: &mut W,
    r: &Register,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    match r {
        Register::Virtual(v) => {
            if let Some(preg) = ra.get_mapping_for(v) {
                let src = if dest.starts_with('w') {
                    w_alias(preg)
                } else {
                    x_alias(preg)
                };
                if src != dest {
                    writeln!(w, "    mov {dest}, {src}")?;
                }
                return Ok(());
            }
            if let Some(off) = frame.slot_of(r) {
                // Use ldur with signed 9-bit offset when possible, otherwise compute address
                if (-256..=255).contains(&off) {
                    writeln!(w, "    ldur {dest}, [x29, #{off}]")?;
                } else {
                    let mut addr = ra.alloc_scratch().unwrap_or("x12");
                    if addr == dest {
                        if let Some(other) = ra.alloc_scratch() {
                            addr = other;
                        } else {
                            addr = if dest != "x11" { "x11" } else { "x10" };
                        }
                    }
                    emit_frame_slot_address(w, addr, off)?;
                    writeln!(w, "    ldr {dest}, [{addr}]")?;
                    if matches!(addr, "x9" | "x10" | "x11" | "x12") {
                        ra.free_scratch(addr);
                    }
                }
            } else {
                writeln!(w, "    // no slot for {r}, leaving {dest} unchanged")?;
            }
        }
        Register::Physical(p) => {
            if p.name != dest {
                writeln!(w, "    mov {}, {}", dest, p.name)?;
            }
        }
    }
    Ok(())
}

/// Store a value from a source register to a destination register or stack slot.
fn store_result<W: Write>(
    w: &mut W,
    dst: &Register,
    src_reg: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(v) => {
            if let Some(preg) = ra.get_mapping_for(v) {
                let slot = if src_reg.starts_with('w') {
                    w_alias(preg)
                } else {
                    x_alias(preg)
                };
                writeln!(w, "    mov {slot}, {src_reg}")?;
                return Ok(());
            }
            if let Some(off) = frame.slot_of(dst) {
                if (-256..=255).contains(&off) {
                    writeln!(w, "    stur {src_reg}, [x29, #{off}]")?;
                } else {
                    let mut addr = ra.alloc_scratch().unwrap_or("x12");
                    if addr == src_reg {
                        if let Some(other) = ra.alloc_scratch() {
                            addr = other;
                        } else {
                            addr = if src_reg != "x11" { "x11" } else { "x10" };
                        }
                    }
                    emit_frame_slot_address(w, addr, off)?;
                    writeln!(w, "    str {src_reg}, [{addr}]")?;
                    if matches!(addr, "x9" | "x10" | "x11" | "x12") {
                        ra.free_scratch(addr);
                    }
                }
            } else {
                writeln!(w, "    // no slot for {dst}")?;
            }
        }
        Register::Physical(p) => {
            if p.name != src_reg {
                writeln!(w, "    mov {}, {}", p.name, src_reg)?;
            }
        }
    }
    Ok(())
}

fn load_copy_reg_to<W: Write>(
    w: &mut W,
    register: &Register,
    dest: &str,
    size: usize,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    match register {
        Register::Virtual(vreg) => {
            if let Some(physical) = ra.get_mapping_for(vreg) {
                let source = x_alias(physical);
                if source != dest {
                    writeln!(w, "    mov {dest}, {source}")?;
                }
            } else if let Some(offset) = frame.slot_of(register) {
                let mnemonic = match size {
                    1 => "ldursb",
                    2 => "ldursh",
                    4 => "ldursw",
                    _ => "ldur",
                };
                if (-256..=255).contains(&offset) {
                    writeln!(w, "    {mnemonic} {dest}, [x29, #{offset}]")?;
                } else {
                    let allocated = ra.alloc_scratch();
                    let address = allocated.unwrap_or("x12");
                    emit_frame_slot_address(w, address, offset)?;
                    let mnemonic = match size {
                        1 => "ldrsb",
                        2 => "ldrsh",
                        4 => "ldrsw",
                        _ => "ldr",
                    };
                    writeln!(w, "    {mnemonic} {dest}, [{address}]")?;
                    if allocated.is_some() {
                        ra.free_scratch(address);
                    }
                }
            } else {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("AArch64 Copy source {register} has no register or stack slot"),
                )));
            }
        }
        Register::Physical(physical) => {
            let source = x_alias(physical.name);
            if source != dest {
                writeln!(w, "    mov {dest}, {source}")?;
            }
        }
    }
    Ok(())
}

fn emit_branch_if<W: Write>(
    w: &mut W,
    cond: &str,
    target: &str,
    long: bool,
) -> Result<(), LaminaError> {
    if !long {
        writeln!(w, "    b.{cond} {target}")?;
        return Ok(());
    }
    let inverse = match cond {
        "eq" => "ne",
        "ne" => "eq",
        "lt" => "ge",
        "ge" => "lt",
        "le" => "gt",
        "gt" => "le",
        "lo" => "hs",
        "hs" => "lo",
        "ls" => "hi",
        "hi" => "ls",
        other => {
            return Err(LaminaError::CodegenError(
                CodegenError::InvalidCodegenOptions(format!("unknown AArch64 condition {other}")),
            ));
        }
    };
    writeln!(w, "    b.{inverse} 9f")?;
    writeln!(w, "    b {target}")?;
    writeln!(w, "9:")?;
    Ok(())
}

fn emit_sp_adjust<W: Write>(w: &mut W, op: &str, amount: u64) -> Result<(), LaminaError> {
    if amount <= 4095 {
        writeln!(w, "    {op} sp, sp, #{amount}")?;
    } else {
        emit_mov_imm64(w, "x16", amount)?;
        writeln!(w, "    {op} sp, sp, x16")?;
    }
    Ok(())
}

fn emit_frame_slot_address<W: Write>(
    w: &mut W,
    dest: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    let offset = i64::from(offset);
    if (0..=4095).contains(&offset) {
        writeln!(w, "    add {dest}, x29, #{offset}")?;
    } else if offset < 0 && -offset <= 4095 {
        writeln!(w, "    sub {dest}, x29, #{}", -offset)?;
    } else {
        emit_mov_imm64(w, dest, offset as u64)?;
        writeln!(w, "    add {dest}, x29, {dest}")?;
    }
    Ok(())
}

/// Materialize an address operand into a register.
fn materialize_address<W: Write>(
    w: &mut W,
    addr: &AddressMode,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            // Materialize base value (should be an address) into dest
            load_reg_to(w, base, dest, frame, ra)?;
            if *offset != 0 {
                let off_val = *offset as i64; // Sign-extend i16 to i64 for range checking
                // AArch64 add/sub with immediate supports 12-bit signed immediates (-2048 to 2047)
                if (0..=4095).contains(&off_val) {
                    writeln!(w, "    add {dest}, {dest}, #{off_val}")?;
                } else if off_val < 0 && -off_val <= 4095 {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -off_val)?;
                } else {
                    // Offset too large, materialize in register
                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                    if oreg == dest {
                        // Need a different register
                        if let Some(other) = ra.alloc_scratch() {
                            let temp = other;
                            emit_mov_imm64(w, temp, off_val as u64)?;
                            writeln!(w, "    add {dest}, {dest}, {temp}")?;
                            ra.free_scratch(temp);
                        } else {
                            // Fallback: use x20 if dest is not x20
                            let temp = if dest != "x20" { "x20" } else { "x19" };
                            emit_mov_imm64(w, temp, off_val as u64)?;
                            writeln!(w, "    add {dest}, {dest}, {temp}")?;
                        }
                    } else {
                        emit_mov_imm64(w, oreg, off_val as u64)?;
                        writeln!(w, "    add {dest}, {dest}, {oreg}")?;
                        ra.free_scratch(oreg);
                    }
                }
            }
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => {
            // dest = base (base holds an address)
            load_reg_to(w, base, dest, frame, ra)?;
            let mut idx = ra.alloc_scratch().unwrap_or("x19");
            if idx == dest {
                if let Some(other) = ra.alloc_scratch() {
                    idx = other;
                } else {
                    idx = "x20";
                }
            }
            load_reg_to(w, index, idx, frame, ra)?; // idx = index
            let sh = match *scale {
                2 => 1,
                4 => 2,
                8 => 3,
                _ => 0,
            };
            if sh > 0 {
                writeln!(w, "    lsl {idx}, {idx}, #{sh}")?;
            }
            writeln!(w, "    add {dest}, {dest}, {idx}")?;
            ra.free_scratch(idx);
            if *offset != 0 {
                let off_val = *offset as i64; // Sign-extend i8 to i64 for range checking
                // AArch64 add/sub with immediate supports 12-bit signed immediates (-2048 to 2047)
                // i8 is always within this range, but check for consistency
                if (0..=4095).contains(&off_val) {
                    writeln!(w, "    add {dest}, {dest}, #{off_val}")?;
                } else if off_val < 0 && -off_val <= 4095 {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -off_val)?;
                } else {
                    // Offset too large (shouldn't happen with i8, but handle for safety)
                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                    if oreg == dest {
                        let temp = if dest != "x20" { "x20" } else { "x19" };
                        emit_mov_imm64(w, temp, off_val as u64)?;
                        writeln!(w, "    add {dest}, {dest}, {temp}")?;
                    } else {
                        emit_mov_imm64(w, oreg, off_val as u64)?;
                        writeln!(w, "    add {dest}, {dest}, {oreg}")?;
                        ra.free_scratch(oreg);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Trait-backed MIR ⇒ AArch64 code generator.
pub struct AArch64Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> AArch64Codegen<'a> {
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
        generate_mir_aarch64_with_units_and_settings(
            module,
            writer,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for AArch64Codegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn capabilities() -> CapabilitySet {
        CapabilitySet::extended_native()
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
                generate_mir_aarch64_with_units(module, writer, target_os, codegen_units)
            },
            "AArch64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not implemented for AArch64 MIR backend".into(),
        ))
    }
}

fn emit_narrow_param_extension(
    w: &mut impl Write,
    xreg: &str,
    ty: &MirType,
) -> Result<(), LaminaError> {
    let instruction = match ty {
        MirType::Scalar(ScalarType::I8) => "sxtb",
        MirType::Scalar(ScalarType::I16) => "sxth",
        MirType::Scalar(ScalarType::I32) => "sxtw",
        MirType::Scalar(ScalarType::I1) => "uxtb",
        _ => return Ok(()),
    };
    writeln!(w, "    {instruction} {xreg}, {}", w_alias(xreg))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_to_float<W: Write>(
    w: &mut W,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    let s_s = ra.alloc_scratch().unwrap_or("x19");
    let s_d = ra.alloc_scratch().unwrap_or("x20");
    emit_materialize_operand(w, src, s_s, frame, ra)?;
    extend_integer(w, s_s, from, signed)?;
    let cvt = if signed { "scvtf" } else { "ucvtf" };
    if to.size_bytes() == 4 {
        writeln!(w, "    {cvt} s0, {s_s}")?;
        writeln!(w, "    fmov {}, s0", w_alias(s_d))?;
    } else {
        writeln!(w, "    {cvt} d0, {s_s}")?;
        writeln!(w, "    fmov {s_d}, d0")?;
    }
    store_result(w, dst, &x_alias(s_d), frame, ra)?;
    ra.free_scratch(s_s);
    ra.free_scratch(s_d);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_float_to_int<W: Write>(
    w: &mut W,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), LaminaError> {
    let s_s = ra.alloc_scratch().unwrap_or("x19");
    let s_d = ra.alloc_scratch().unwrap_or("x20");
    let s_t = ra.alloc_scratch().unwrap_or("x21");
    emit_materialize_operand(w, src, s_s, frame, ra)?;
    if from.size_bytes() == 4 {
        writeln!(w, "    fmov s0, {}", w_alias(s_s))?;
    } else {
        writeln!(w, "    fmov d0, {s_s}")?;
    }
    let bits = to.size_bytes() * 8;
    let d32 = w_alias(s_d);
    let t32 = w_alias(s_t);
    if bits == 64 {
        let cvt = if signed { "fcvtzs" } else { "fcvtzu" };
        let from_suffix = if from.size_bytes() == 4 { "s0" } else { "d0" };
        writeln!(w, "    {cvt} {s_d}, {from_suffix}")?;
    } else {
        let cvt = if signed { "fcvtzs" } else { "fcvtzu" };
        let from_suffix = if from.size_bytes() == 4 { "s0" } else { "d0" };
        writeln!(w, "    {cvt} {d32}, {from_suffix}")?;
        if bits < 32 {
            let (min, max) = if signed {
                (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
            } else {
                (0, (1i64 << bits) - 1)
            };
            writeln!(w, "    mov {t32}, #{max}")?;
            writeln!(w, "    cmp {d32}, {t32}")?;
            writeln!(
                w,
                "    csel {d32}, {d32}, {t32}, {}",
                if signed { "le" } else { "ls" }
            )?;
            if signed {
                writeln!(w, "    mov {t32}, #{min}")?;
                writeln!(w, "    cmp {d32}, {t32}")?;
                writeln!(w, "    csel {d32}, {d32}, {t32}, ge")?;
            }
        }
    }
    extend_integer(w, s_d, to, signed)?;
    store_result(w, dst, s_d, frame, ra)?;
    ra.free_scratch(s_s);
    ra.free_scratch(s_d);
    ra.free_scratch(s_t);
    Ok(())
}

fn extend_integer(
    w: &mut impl Write,
    register: &str,
    ty: &MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
    ) {
        return Ok(());
    }
    let bits = ty.size_bytes() * 8;
    let instruction = if signed { "sbfx" } else { "ubfx" };
    writeln!(w, "    {instruction} {register}, {register}, #0, #{bits}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, ScalarType};
    use std::error::Error;

    #[test]
    fn global_parameters_are_initialized_before_entry() -> Result<(), Box<dyn Error>> {
        use crate::mir::{Block, Parameter};
        let reg = Register::Virtual(VirtualReg::gpr(0));
        let mut func = Function::new(
            Signature::new("identity")
                .with_params(vec![Parameter::new(reg.clone(), MirType::i64())])
                .with_return(MirType::i64()),
        );
        let mut block = Block::new("entry");
        block.push(MirInst::Ret {
            value: Some(Operand::Register(reg)),
        });
        func.add_block(block);
        for regalloc in [
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc,
                ..Default::default()
            };
            let asm = String::from_utf8(
                compile_single_function_aarch64(
                    "identity",
                    &func,
                    TargetOperatingSystem::Linux,
                    &settings,
                    &AggregateSignatures::default(),
                )
                .map_err(|error| error.to_string())?,
            )?;
            let (prologue, body) = asm
                .split_once(".L_identity.entry:")
                .ok_or("missing entry")?;
            let reg = body
                .lines()
                .find_map(|line| line.strip_prefix("    mov x0, "))
                .ok_or("missing return move")?;
            assert!(
                prologue.contains(&format!("ldur {reg}, [x29, #-8]")),
                "{asm}"
            );
        }
        Ok(())
    }

    #[test]
    fn narrow_parameters_are_sign_extended_on_entry() -> Result<(), Box<dyn Error>> {
        use crate::mir::{Block, Parameter};
        for (ty, extend) in [
            (MirType::i8(), "sxtb x0, w0"),
            (MirType::i16(), "sxth x0, w0"),
            (MirType::i32(), "sxtw x0, w0"),
        ] {
            let reg = Register::Virtual(VirtualReg::gpr(0));
            let mut func = Function::new(
                Signature::new("narrow")
                    .with_params(vec![Parameter::new(reg.clone(), ty)])
                    .with_return(ty),
            );
            let mut block = Block::new("entry");
            block.push(MirInst::Ret {
                value: Some(Operand::Register(reg)),
            });
            func.add_block(block);
            let asm = String::from_utf8(
                compile_single_function_aarch64(
                    "narrow",
                    &func,
                    TargetOperatingSystem::Linux,
                    &MirCodegenSettings::default(),
                    &AggregateSignatures::default(),
                )
                .map_err(|error| error.to_string())?,
            )?;
            let (prologue, _) = asm.split_once(".L_narrow.entry:").ok_or("missing entry")?;
            assert!(prologue.contains(&format!("    {extend}\n")), "{asm}");
        }
        Ok(())
    }

    #[test]
    fn bool_parameters_are_zero_extended_on_entry() -> Result<(), Box<dyn Error>> {
        use crate::mir::{Block, Parameter};
        let ty = MirType::Scalar(ScalarType::I1);
        let reg = Register::Virtual(VirtualReg::gpr(0));
        let mut func = Function::new(
            Signature::new("flag")
                .with_params(vec![Parameter::new(reg.clone(), ty)])
                .with_return(ty),
        );
        let mut block = Block::new("entry");
        block.push(MirInst::Ret {
            value: Some(Operand::Register(reg)),
        });
        func.add_block(block);
        let asm = String::from_utf8(
            compile_single_function_aarch64(
                "flag",
                &func,
                TargetOperatingSystem::Linux,
                &MirCodegenSettings::default(),
                &AggregateSignatures::default(),
            )
            .map_err(|error| error.to_string())?,
        )?;
        let (prologue, _) = asm.split_once(".L_flag.entry:").ok_or("missing entry")?;
        assert!(prologue.contains("    uxtb x0, w0\n"), "{asm}");
        Ok(())
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
        generate_mir_aarch64(&module, &mut output, TargetOperatingSystem::Linux)
            .expect("AArch64 codegen should succeed");
        let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

        assert!(assembly.contains("mov "), "expected move: {assembly}");
        assert!(
            !assembly.contains("fadd"),
            "unexpected float add: {assembly}"
        );
    }

    #[test]
    fn mach_o_targets_call_underscored_libc_symbols() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir =
            "fn @main() -> i64 {\nentry:\n  %v = add.i64 1, 2\n  call @print(%v)\n  ret.i64 0\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "print_test").expect("lower print module");
        for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::IOS] {
            let mut output = Vec::new();
            generate_mir_aarch64(&mir, &mut output, os)?;
            let assembly = String::from_utf8(output)?;
            assert!(assembly.contains("bl _printf\n"), "{os:?}: {assembly}");
            assert!(assembly.contains("_main:\n"), "{os:?}: {assembly}");
            assert!(!assembly.contains("bl printf\n"), "{os:?}: {assembly}");
        }
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("bl printf\n"), "{assembly}");
        Ok(())
    }

    #[test]
    fn global_allocation_keeps_values_out_of_caller_saved_pool_across_calls()
    -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @id(i64 %x) -> i64 {\nentry:\n  %r = mul.i64 %x, 1\n  ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n  %a = add.i64 7, 0\n  %b = mul.i64 %a, 3\n  %c = call @id(5)\n  %d = add.i64 %b, %c\n  ret.i64 %d\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "call_crossing").expect("lower call module");
        for regalloc in [
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc,
                ..Default::default()
            };
            let mut output = Vec::new();
            generate_mir_aarch64_with_units_and_settings(
                &mir,
                &mut output,
                TargetOperatingSystem::Linux,
                1,
                &settings,
            )?;
            let assembly = String::from_utf8(output)?;
            let (_, main) = assembly.split_once("\nmain:\n").ok_or("missing main")?;
            let main = main.split_once(".Lret_main:").ok_or("missing epilogue")?.0;
            assert!(main.contains("bl id"), "{assembly}");
            for reg in ["x13", "x14", "x15"] {
                assert!(!main.contains(reg), "{regalloc:?} uses {reg}: {assembly}");
            }
        }
        Ok(())
    }

    #[test]
    fn tail_call_releases_the_frame_the_prologue_reserved() -> Result<(), Box<dyn Error>> {
        use crate::mir::{Block, Parameter};
        let x = Register::Virtual(VirtualReg::gpr(0));
        let mut func = Function::new(
            Signature::new("f")
                .with_params(vec![Parameter::new(x.clone(), MirType::i64())])
                .with_return(MirType::i64()),
        );
        let mut block = Block::new("entry");
        let mut prev = Operand::Register(x);
        for id in 1..=101 {
            let dst = Register::Virtual(VirtualReg::gpr(id));
            block.push(MirInst::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::i64(),
                dst: dst.clone(),
                lhs: prev,
                rhs: Operand::Immediate(Immediate::I64(1)),
            });
            prev = Operand::Register(dst);
        }
        block.push(MirInst::TailCall {
            name: "g".into(),
            args: vec![prev],
        });
        func.add_block(block);
        let asm = String::from_utf8(
            compile_single_function_aarch64(
                "f",
                &func,
                TargetOperatingSystem::Linux,
                &MirCodegenSettings::default(),
                &AggregateSignatures::default(),
            )
            .map_err(|error| error.to_string())?,
        )?;
        let reserved = asm
            .lines()
            .find_map(|line| line.strip_prefix("    sub sp, sp, #"))
            .ok_or("missing frame reservation")?;
        assert!(
            asm.contains(&format!(
                "    add sp, sp, #{reserved}\n    ldp x29, x30, [sp], #16\n    b g\n"
            )),
            "{asm}"
        );
        Ok(())
    }

    #[test]
    fn long_conditional_branches_are_inverted_over_unconditional_branches()
    -> Result<(), Box<dyn Error>> {
        let mut direct = Vec::new();
        emit_branch_if(&mut direct, "lt", ".L_f.far", false)?;
        assert_eq!(String::from_utf8(direct)?, "    b.lt .L_f.far\n");
        let mut long = Vec::new();
        emit_branch_if(&mut long, "lt", ".L_f.far", true)?;
        assert_eq!(
            String::from_utf8(long)?,
            "    b.ge 9f\n    b .L_f.far\n9:\n"
        );
        Ok(())
    }

    #[test]
    fn switch_case_constants_are_materialized_without_immediate_range_limits()
    -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @main() -> i64 {\nentry:\n  %x = add.i64 100000, 0\n  switch.i64 %x, default, [100000, hit], [-5, neg]\nhit:\n  ret.i64 1\nneg:\n  ret.i64 2\ndefault:\n  ret.i64 3\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "switch_range").expect("lower switch module");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(!assembly.contains("mov x9, #100000"), "{assembly}");
        assert!(assembly.contains("movz x9, #34464, lsl #0"), "{assembly}");
        Ok(())
    }

    #[test]
    fn integer_float_conversions_select_width_and_saturation_forms() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @main() -> i64 {\nentry:\n  %a = sitofp.i8.f32 -3\n  %b = fptosi.f64.i8 2.5\n  %c = fptoui.f64.u64 1.0\n  %d = uitofp.u64.f64 7\n  ret.i64 0\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "conversions").expect("lower conversion module");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        assert!(assembly.contains("sbfx x"), "{assembly}");
        assert!(assembly.contains("scvtf s0, x"), "{assembly}");
        assert!(assembly.contains("fcvtzs w"), "{assembly}");
        assert!(assembly.contains("csel w"), "{assembly}");
        assert!(assembly.contains("fcvtzu x"), "{assembly}");
        assert!(assembly.contains("ucvtf d0, x"), "{assembly}");
        Ok(())
    }

    #[test]
    fn module_globals_are_addressed_through_page_relocations() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "global @counter: i64 = 41\n\nfn @main() -> i64 {\nentry:\n  %a = load.i64 @counter\n  ret.i64 %a\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "globals").expect("lower global module");
        let mut linux = Vec::new();
        generate_mir_aarch64(&mir, &mut linux, TargetOperatingSystem::Linux)?;
        let linux = String::from_utf8(linux)?;
        assert!(linux.contains("adrp x9, counter\n"), "{linux}");
        assert!(linux.contains("add x9, x9, :lo12:counter\n"), "{linux}");
        assert!(
            linux.contains(".data\n.p2align 3\ncounter:\n    .byte 0x29, 0x00"),
            "{linux}"
        );
        let mut macho = Vec::new();
        generate_mir_aarch64(&mir, &mut macho, TargetOperatingSystem::MacOS)?;
        let macho = String::from_utf8(macho)?;
        assert!(macho.contains("adrp x9, _counter@PAGE\n"), "{macho}");
        assert!(macho.contains("add x9, x9, _counter@PAGEOFF\n"), "{macho}");
        Ok(())
    }

    #[test]
    fn frames_beyond_twelve_bit_immediates_are_addressed_through_a_register()
    -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @main() -> i64 {\nentry:\n  %arr = alloc.stack [5000 x i8]\n  %p = getelem.ptr %arr, 4999, i8\n  store.i8 %p, 42\n  %v = load.i8 %p\n  %w = zext.i8.i64 %v\n  ret.i64 %w\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "big_frame").expect("lower big frame module");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;
        for line in assembly.lines() {
            if line.contains("sp, sp, #") || line.contains("x29, #") || line.contains("x29, x29, #")
            {
                let imm: i64 = line
                    .rsplit_once('#')
                    .and_then(|(_, n)| n.trim().parse().ok())
                    .unwrap_or(0);
                assert!(
                    imm.abs() <= 4095,
                    "immediate out of range in: {line}\n{assembly}"
                );
            }
        }
        assert!(assembly.contains("sub sp, sp, x16"), "{assembly}");
        Ok(())
    }

    #[test]
    fn integer_aggregates_use_consecutive_x_registers() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @sum(struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  %ap = getfieldptr %p, 0\n  %a = load.i64 %ap\n  ret.i64 %a\n}\nfn @wide(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n  ret.i64 %a\n}\nfn @main() -> i64 {\nentry:\n  %q = alloc.stack struct { a: i64, b: i64 }\n  %r = call @sum(%q)\n  %s = call @wide(1, 2, 3, 4, 5, 6, 7, %q)\n  ret.i64 %r\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "aggregates").expect("lower aggregates");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        let prologue = assembly
            .split("\nsum:\n")
            .nth(1)
            .and_then(|body| body.split(".L_sum.entry:").next())
            .ok_or("missing sum prologue")?;
        let home = prologue
            .lines()
            .filter(|line| line.contains(", x29, #"))
            .find_map(|line| line.trim().strip_prefix("sub ")?.split(',').next())
            .ok_or("missing frame address")?
            .to_string();
        assert!(
            prologue.contains(&format!("str x0, [{home}, #0]")),
            "{assembly}"
        );
        assert!(
            prologue.contains(&format!("str x1, [{home}, #8]")),
            "{assembly}"
        );

        let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
        let call = caller.split("bl sum").next().ok_or("missing call")?;
        assert!(call.contains("ldr x0, ["), "{assembly}");
        assert!(call.contains("ldr x1, ["), "{assembly}");

        // Seven scalars leave one register, which is too few for two words, so the
        // aggregate goes to the stack whole and x7 stays unused.
        let wide = assembly
            .split("\nwide:\n")
            .nth(1)
            .and_then(|body| body.split(".L_wide.entry:").next())
            .ok_or("missing wide prologue")?;
        assert!(!wide.contains("str x7, "), "{assembly}");
        assert!(wide.contains("ldr x11, [x29, #16]"), "{assembly}");
        assert!(wide.contains("ldr x11, [x29, #24]"), "{assembly}");
        Ok(())
    }

    #[test]
    fn floats_cross_the_boundary_in_v_registers() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @add(f64 %a, f64 %b) -> f64 {\nentry:\n  %s = add.f64 %a, %b\n  ret.f64 %s\n}\nfn @addf(f32 %a, i64 %n) -> f32 {\nentry:\n  %s = add.f32 %a, 1.0\n  ret.f32 %s\n}\nfn @main() -> i64 {\nentry:\n  %r = call @add(1.5, 2.5)\n  %i = fptosi.f64.i64 %r\n  ret.i64 %i\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "floats").expect("lower floats");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        // Both arguments arrive in the float bank, and the result leaves in d0.
        let callee = assembly.split("\nadd:\n").nth(1).ok_or("missing add")?;
        let prologue = callee
            .split(".L_add.entry:")
            .next()
            .ok_or("missing entry")?;
        assert!(prologue.contains("fmov x"), "{assembly}");
        assert!(prologue.contains(", d0\n"), "{assembly}");
        assert!(prologue.contains(", d1\n"), "{assembly}");
        assert!(callee.contains("    fmov d0, x"), "{assembly}");

        // A float and an integer draw from separate banks, so the integer still takes x0.
        let mixed = assembly.split("\naddf:\n").nth(1).ok_or("missing addf")?;
        let mixed_prologue = mixed
            .split(".L_addf.entry:")
            .next()
            .ok_or("missing entry")?;
        assert!(mixed_prologue.contains(", s0\n"), "{assembly}");
        assert!(mixed_prologue.contains("    str x0, "), "{assembly}");

        let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
        let call = caller.split("bl add").next().ok_or("missing call")?;
        assert!(call.contains("    fmov d0, x"), "{assembly}");
        assert!(call.contains("    fmov d1, x"), "{assembly}");
        assert!(caller.contains("    fmov x0, d0\n"), "{assembly}");
        Ok(())
    }

    #[test]
    fn integer_aggregate_returns_come_back_in_x0_and_x1() -> Result<(), Box<dyn Error>> {
        use crate::{mir::codegen::from_ir, parser::parse_module};
        let ir = "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n  %p = alloc.stack struct { a: i64, b: i64 }\n  %pa = getfieldptr %p, 0\n  store.i64 %pa, 7\n  ret.ptr %p\n}\nfn @main() -> i64 {\nentry:\n  %r = call @make()\n  %ra = getfieldptr %r, 0\n  %a = load.i64 %ra\n  ret.i64 %a\n}\n";
        let module = parse_module(ir)?;
        let mir = from_ir(&module, "returns").expect("lower aggregate return");
        let mut output = Vec::new();
        generate_mir_aarch64(&mir, &mut output, TargetOperatingSystem::Linux)?;
        let assembly = String::from_utf8(output)?;

        let callee = assembly.split("\nmake:\n").nth(1).ok_or("missing make")?;
        let epilogue = callee.split(".Lret_make:").next().ok_or("missing return")?;
        let address = epilogue
            .lines()
            .rev()
            .find_map(|line| {
                let rest = line.trim().strip_prefix("ldr x0, [")?;
                rest.strip_suffix(']')
            })
            .ok_or("missing result load")?
            .to_string();
        assert!(
            epilogue.contains(&format!("ldr x1, [{address}, #8]")),
            "{assembly}"
        );

        let caller = assembly.split("\nmain:\n").nth(1).ok_or("missing main")?;
        let after = caller.split("bl make").nth(1).ok_or("missing call")?;
        let home = after
            .lines()
            .find_map(|line| line.trim().strip_prefix("sub ")?.split(',').next())
            .ok_or("missing result slot")?
            .to_string();
        assert!(after.contains(&format!("str x0, [{home}]")), "{assembly}");
        assert!(
            after.contains(&format!("str x1, [{home}, #8]")),
            "{assembly}"
        );
        Ok(())
    }
}
