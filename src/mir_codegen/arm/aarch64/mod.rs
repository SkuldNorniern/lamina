//! AArch64 code generation for MIR (Mid-level IR).
//!
//! Code generation from MIR to AArch64 assembly,
//! following the AAPCS64 calling convention.

mod atomic;
mod bit;
mod div;
mod imm;
mod inline_asm;
mod overflow;
mod simd;
mod util;
mod variadic;

use crate::mir::Block;
use crate::mir::codegen::MAX_REGISTER_AGGREGATE;
use crate::mir::register::PhysicalReg;
use crate::mir_codegen::arm::{float_condition, int_condition, invert_condition};
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::external_names;
use crate::mir_codegen::common::lamina_to_codegen_error;
use crate::mir_codegen::common::unsupported_io;
use crate::mir_codegen::common::{
    ClobberIndex, CopyLocation, FunctionEmissionPlan, coalesce_dying_inputs, emit_global_data,
    emit_global_directives, emit_macho_thread_local_globals, emit_static_global_data,
    emit_symbol_visibility, integer_conversion, self_tail_arguments, self_tail_schedule,
};
use crate::mir_codegen::common::{emit_debug_file, emit_gnu_stack_note, promoted_argument_type};
use crate::mir_codegen::common::{reject_thread_local_globals, unsupported_thread_local};
use crate::mir_codegen::macho_unwind;
use crate::mir_codegen::simd::reserve_vector_slots;
use lamina_mir::{SourceLoc, TlsModel};

use std::{
    collections::HashMap, collections::HashSet, io::Error as IoError, io::Write, slice, str,
    sync::Arc,
};

use imm::{arith_immediate, logical_immediate, negated_arith_immediate};
use util::{emit_mov_imm64, imm_to_u64};

use crate::{
    error::LaminaError,
    mir::{
        AddressMode, Aggregate, FloatBinOp, FloatMember, FloatUnOp, Function, Global,
        Instruction as MirInst, IntBinOp, IntCmpOp, MirType, Module as MirModule, Operand,
        Register, ScalarType, Signature, VirtualReg, index_scale_shift, register::RegisterClass,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings, RegallocStrategy,
        capability::{CapabilitySet, CodegenCapability},
        common::{CodegenBase, compile_functions_parallel, parallel_codegen_error},
        validate_module_call_parameters,
    },
};

use lamina_codegen::{
    Allocation as MirAllocation, GraphColorAllocator, LinearScanAllocator, LiveInterval,
    aarch64::{A64RegAlloc, AArch64ABI, FrameMap},
};
use lamina_codegen::{os_uses_coff, os_uses_elf, os_uses_macho};
use lamina_platform::{PointerWidth, StackProbe, TargetArchitecture, TargetOperatingSystem};

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
    Homogeneous {
        first: usize,
        count: usize,
        member: FloatMember,
    },
    /// `bytes` of outgoing stack space at `offset` from the start of the argument area.
    Stack {
        offset: usize,
        bytes: usize,
    },
}

const ARGUMENT_REGISTERS: usize = 8;

/// Saved frame pointer and link register, which the prologue pushes before the frame.
pub(super) const FRAME_RECORD_BYTES: i32 = 16;

/// Checks the `sret` and `byval` attributes this function states against where AAPCS64
/// actually places the values. Honouring a claim the placement contradicts would turn a
/// frontend's wrong classification into a wrong call.
fn verify_abi_claims(
    func: &Function,
    kinds: &[ArgKind],
    places: &[ArgPlace],
    target_os: TargetOperatingSystem,
) -> Result<(), CodegenError> {
    let name = &func.sig.name;
    if func.sig.sret_claimed && !func.sig.ret_aggregate.is_some_and(passed_in_memory) {
        return Err(CodegenError::InvalidCodegenOptions(format!(
            "@{name} states sret on its return, but aarch64_{target_os} returns that type in registers, not through a hidden pointer"
        )));
    }
    for (index, param) in func.sig.params.iter().enumerate() {
        if !param.byval_claimed {
            continue;
        }
        let param_name = param.name.clone().unwrap_or_else(|| index.to_string());
        let by_reference = kinds.get(index).is_some_and(|kind| kind.by_reference);
        let placed = match places.get(index) {
            Some(ArgPlace::Stack { .. }) if !by_reference => continue,
            Some(ArgPlace::Stack { .. }) => "a pointer to a copy the caller owns, on the stack",
            Some(_) if by_reference => "a pointer to a copy the caller owns, in a register",
            Some(ArgPlace::Homogeneous { .. }) => "floating-point registers",
            Some(_) => "registers",
            None => "nowhere",
        };
        return Err(CodegenError::InvalidCodegenOptions(format!(
            "%{param_name} of @{name} states byval, but aarch64_{target_os} passes it in {placed}"
        )));
    }
    Ok(())
}

fn passed_in_memory(aggregate: Aggregate) -> bool {
    aggregate.size as u64 > MAX_REGISTER_AGGREGATE && aggregate.homogeneous_members().is_none()
}

fn member_register(member: FloatMember, index: usize) -> String {
    match member {
        FloatMember::F32 => format!("s{index}"),
        FloatMember::F64 => format!("d{index}"),
    }
}

/// One argument as the convention sees it: a float, an integer aggregate, or a plain
/// integer word.
#[derive(Clone, Copy)]
struct ArgKind {
    float: bool,
    ty_is_f32: bool,
    /// Size the convention gives the value, which Darwin packs stack slots to.
    bytes: usize,
    aggregate: Option<Aggregate>,
    /// aapcs64 hands a composite over 16 bytes to the callee as a pointer to a copy
    /// the caller owns, so it takes one register however large it is.
    by_reference: bool,
}

impl ArgKind {
    fn of(ty: MirType, aggregate: Option<Aggregate>, pointer_width: PointerWidth) -> Self {
        let by_reference = aggregate.is_some_and(passed_in_memory);
        Self {
            float: aggregate.is_none() && ty.is_float(),
            ty_is_f32: ty.size_bytes(pointer_width) == 4,
            bytes: if by_reference {
                8
            } else {
                aggregate.map_or(ty.size_bytes(pointer_width), |a| a.size)
            },
            aggregate,
            by_reference,
        }
    }
}

/// Floats take the next v register and integers the next x register, from banks that
/// run out separately. An aggregate that does not fit in the registers left goes to
/// the stack whole, and takes the remaining ones out of play.
/// Copies `words` eight-byte words from the address in `pointer` to the address in
/// `destination`.
fn copy_through_pointer<W: Write>(
    w: &mut W,
    pointer: &str,
    destination: &str,
    words: usize,
    ra: &mut A64RegAlloc,
) -> Result<(), IoError> {
    let value = ra
        .alloc_scratch()
        .ok_or_else(|| IoError::other("aarch64 scratch registers exhausted"))?;
    for word in 0..words {
        let at = word * 8;
        writeln!(w, "    ldr {value}, [{pointer}, #{at}]")?;
        writeln!(w, "    str {value}, [{destination}, #{at}]")?;
    }
    ra.free_scratch(value);
    Ok(())
}

/// Places `kinds` by AAPCS64. `fixed` is the count of named arguments when the callee
/// is variadic, which Darwin needs because it puts the rest on the stack.
///
/// Writing ARM64 Code for Apple Platforms, "Pass arguments to functions correctly":
/// a stack argument may consume a slot that is not a multiple of eight bytes. The
/// divergence reaches scalars only -- a composite still takes whole slots at an
/// alignment of at least eight, as AAPCS64 stage C requires. The same section lets a
/// sixteen-byte-aligned argument start in an odd-numbered `x` register, where AAPCS64
/// stage C.8 rounds the next general register number up to even.
fn classify_aarch64_arguments(
    kinds: &[ArgKind],
    target_os: TargetOperatingSystem,
    fixed: Option<usize>,
) -> Vec<ArgPlace> {
    let darwin = os_uses_macho(target_os);
    let coff = os_uses_coff(target_os);
    let mut gprs = 0usize;
    let mut fprs = 0;
    let mut stack = 0;
    let mut places = Vec::with_capacity(kinds.len());
    // Linux gives every stack argument a whole eight-byte slot. Darwin packs a named
    // scalar at its natural size and alignment; a composite and a variadic argument
    // take whole slots, as AAPCS64 says.
    let honors_alignment = TargetArchitecture::Aarch64.honors_argument_alignment();
    let on_stack = |stack: &mut usize, kind: &ArgKind, packed: bool| {
        let bytes = if packed {
            kind.bytes.max(1)
        } else {
            kind.bytes.max(1).div_ceil(8) * 8
        };
        let align = if packed {
            bytes.min(8).next_power_of_two()
        } else if honors_alignment && !kind.by_reference {
            kind.aggregate.map_or(8, |aggregate| aggregate.align).max(8)
        } else {
            8
        };
        *stack = stack.div_ceil(align) * align;
        let offset = *stack;
        *stack += bytes;
        ArgPlace::Stack { offset, bytes }
    };
    for (index, kind) in kinds.iter().enumerate() {
        // Darwin hands a variadic callee its arguments on the stack, whichever
        // registers are still free.
        let unnamed = fixed.is_some_and(|fixed| index >= fixed);
        let variadic = darwin && unnamed;
        let packed = darwin && !variadic && kind.aggregate.is_none();
        if variadic {
            places.push(on_stack(&mut stack, kind, packed));
            continue;
        }
        // Windows hands a variadic callee its floats in general registers; it has no
        // vector save area for va_arg to read.
        let as_integer = coff && unnamed;
        if kind.float && !as_integer {
            if fprs < ARGUMENT_REGISTERS {
                places.push(ArgPlace::Fpr { index: fprs });
                fprs += 1;
            } else {
                places.push(on_stack(&mut stack, kind, packed));
            }
            continue;
        }
        if let Some((member, count)) = kind
            .aggregate
            .filter(|_| !as_integer)
            .and_then(|a| a.homogeneous_members())
        {
            if fprs + count <= ARGUMENT_REGISTERS {
                places.push(ArgPlace::Homogeneous {
                    first: fprs,
                    count,
                    member,
                });
                fprs += count;
            } else {
                fprs = ARGUMENT_REGISTERS;
                places.push(on_stack(&mut stack, kind, packed));
            }
            continue;
        }
        let words = if kind.by_reference {
            1
        } else {
            kind.aggregate.map_or(1, |a| a.words())
        };
        if honors_alignment
            && !darwin
            && !kind.by_reference
            && kind.aggregate.is_some_and(|a| a.align >= 16)
        {
            gprs = gprs.next_multiple_of(2);
        }
        if gprs + words <= ARGUMENT_REGISTERS {
            places.push(ArgPlace::Gpr { first: gprs, words });
            gprs += words;
        } else {
            gprs = ARGUMENT_REGISTERS;
            places.push(on_stack(&mut stack, kind, packed));
        }
    }
    places
}

/// Every caller frees the scratch it takes, so an empty pool means one lowering path
/// holds more at once than the pool has. Falling back to a callee-saved register would
/// hide that behind a register nothing saves.
fn unknown_condition(cond: &str) -> LaminaError {
    LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(format!(
        "unknown AArch64 condition {cond}"
    )))
}

fn libc_write(os: TargetOperatingSystem) -> &'static str {
    match os {
        TargetOperatingSystem::Windows => "_write",
        _ => "write",
    }
}

fn libc_read(os: TargetOperatingSystem) -> &'static str {
    match os {
        TargetOperatingSystem::Windows => "_read",
        _ => "read",
    }
}

fn scratch_error() -> CodegenError {
    CodegenError::UnsupportedFeature("aarch64 scratch registers exhausted".to_string())
}

/// Bytes of outgoing stack the places need, before the sixteen-byte round up.
fn aarch64_stack_bytes(places: &[ArgPlace]) -> usize {
    places
        .iter()
        .map(|place| match place {
            ArgPlace::Stack { offset, bytes } => offset + bytes,
            ArgPlace::Gpr { .. } | ArgPlace::Fpr { .. } | ArgPlace::Homogeneous { .. } => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Store of `bytes` from `source`, narrow forms included so a packed Darwin slot does
/// not overwrite its neighbour.
fn store_argument_bytes(
    w: &mut impl Write,
    source: &str,
    base: &str,
    offset: usize,
    bytes: usize,
) -> Result<(), IoError> {
    match bytes {
        1 => writeln!(w, "    strb {}, [{base}, #{offset}]", w_alias(source)),
        2 => writeln!(w, "    strh {}, [{base}, #{offset}]", w_alias(source)),
        4 => writeln!(w, "    str {}, [{base}, #{offset}]", w_alias(source)),
        _ => writeln!(w, "    str {source}, [{base}, #{offset}]"),
    }
}

/// Load of `bytes` into `destination`, matching `store_argument_bytes`.
fn load_argument_bytes(
    w: &mut impl Write,
    destination: &str,
    base: &str,
    offset: usize,
    bytes: usize,
) -> Result<(), IoError> {
    match bytes {
        1 => writeln!(w, "    ldrb {}, [{base}, #{offset}]", w_alias(destination)),
        2 => writeln!(w, "    ldrh {}, [{base}, #{offset}]", w_alias(destination)),
        4 => writeln!(w, "    ldr {}, [{base}, #{offset}]", w_alias(destination)),
        _ => writeln!(w, "    ldr {destination}, [{base}, #{offset}]"),
    }
}

/// Aggregate markers of each function's boundary, so a call site knows how the callee
/// reads its arguments and hands back its result.
struct AggregateSignatures {
    pointer_width: PointerWidth,
    params: HashMap<String, Vec<ArgKind>>,
    returns: HashMap<String, Aggregate>,
    float_returns: HashMap<String, MirType>,
    variadic: HashSet<String>,
    /// Functions the module declares but does not define.
    external: HashSet<String>,
}

impl AggregateSignatures {
    #[cfg(test)]
    fn empty(pointer_width: PointerWidth) -> Self {
        Self {
            pointer_width,
            params: HashMap::new(),
            returns: HashMap::new(),
            float_returns: HashMap::new(),
            variadic: HashSet::new(),
            external: HashSet::new(),
        }
    }

    fn from_module(module: &MirModule, pointer_width: PointerWidth) -> Self {
        Self {
            pointer_width,
            params: module
                .functions
                .iter()
                .map(|(name, func)| {
                    (
                        name.clone(),
                        func.sig
                            .params
                            .iter()
                            .map(|p| ArgKind::of(p.ty, p.aggregate, pointer_width))
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
            external: module.external_functions.clone(),
        }
    }

    /// Index of the first argument the convention treats as variadic, if any.
    fn fixed_arguments(&self, callee: &str) -> Option<usize> {
        self.variadic
            .contains(callee)
            .then(|| self.params.get(callee).map_or(0, Vec::len))
    }

    /// Arguments of a call as the convention sees them. A matching prototype gives the
    /// kinds; past a variadic callee's fixed parameters, and for a callee with no
    /// prototype, the operand does.
    fn arguments(&self, callee: &str, args: &[Operand]) -> Vec<ArgKind> {
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
                _ => ArgKind::of(promoted_argument_type(arg), None, self.pointer_width),
            })
            .collect()
    }

    fn result(&self, callee: &str) -> Option<Aggregate> {
        self.returns.get(callee).copied()
    }

    fn float_result(&self, callee: &str) -> Option<MirType> {
        self.float_returns.get(callee).copied()
    }

    fn is_external(&self, name: &str) -> bool {
        self.external.contains(name)
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

    let unwind = os_uses_elf(target_os);
    let io_error = |e: IoError| CodegenError::InvalidCodegenOptions(format!("IO error: {e}"));
    if unwind {
        writeln!(output, "    .cfi_startproc").map_err(io_error)?;
    }
    let mut frame = FrameMap::from_function(func);

    let makes_calls = func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                &i.instr,
                MirInst::Call { .. }
                    | MirInst::CallIndirect { .. }
                    | MirInst::ThreadLocalAddr { .. }
            )
        });
    // A tail call leaves through its own epilogue and inline asm picks physical
    // registers without consulting the allocator, so neither may hold a planned value in
    // a callee-saved register, and neither gets a plan at all once the function calls.
    let allows_callee_saved = !func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                &i.instr,
                MirInst::TailCall { .. } | MirInst::InlineAsm { .. }
            )
        });
    let global_plan: Option<HashMap<VirtualReg, MirAllocation<&'static str>>> = if settings.regalloc
        != RegallocStrategy::Incremental
        && (allows_callee_saved || !makes_calls)
    {
        // A call destroys every caller-saved register, so a value whose interval
        // covers one may only live in a callee-saved register; everything else keeps
        // the x13-x15 pool.
        let mut call_clobbers: ClobberIndex<&'static str> = ClobberIndex::default();
        for (index, inst) in func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .enumerate()
        {
            if matches!(
                &inst.instr,
                MirInst::Call { .. }
                    | MirInst::CallIndirect { .. }
                    | MirInst::ThreadLocalAddr { .. }
            ) {
                for reg in A64RegAlloc::caller_saved_in_global_pool() {
                    call_clobbers.mark(reg, index);
                }
            }
        }
        let allowed = |interval: &LiveInterval, reg| call_clobbers.is_clear(reg, interval);
        let pool = A64RegAlloc::gpr_pool_for_global_allocation(allows_callee_saved);
        let intervals: Vec<_> = LinearScanAllocator::compute_intervals(func)
            .into_iter()
            .filter(|i| {
                i.vreg.class == RegisterClass::Gpr && !func.stack_reservations.contains_key(&i.vreg)
            })
            .collect();
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
        frame.recompute_frame_size_from_slots();
        Some(plan)
    } else {
        None
    };

    let callee_saved: Vec<&'static str> = global_plan
        .as_ref()
        .map(A64RegAlloc::callee_saved_in_plan)
        .unwrap_or_default();

    let param_kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind::of(p.ty, p.aggregate, func.pointer_width))
        .collect();
    let param_places = classify_aarch64_arguments(&param_kinds, target_os, None);
    verify_abi_claims(func, &param_kinds, &param_places, target_os)?;
    // Windows keeps va_list a plain pointer, so the general register save area has to
    // sit directly under the incoming stack arguments. Growing the frame record is
    // what puts it there, above the frame and below the caller's stack arguments.
    let frame_record = if func.sig.variadic && os_uses_coff(target_os) {
        variadic::windows_frame_record_bytes(&param_places)
    } else {
        FRAME_RECORD_BYTES
    };

    let mut ra_pro = A64RegAlloc::new();
    writeln!(output, "    stp x29, x30, [sp, #-{frame_record}]!").map_err(io_error)?;
    if unwind {
        writeln!(output, "    .cfi_def_cfa_offset {frame_record}").map_err(io_error)?;
        writeln!(output, "    .cfi_offset 29, -16").map_err(io_error)?;
        writeln!(output, "    .cfi_offset 30, -8").map_err(io_error)?;
    }
    writeln!(output, "    mov x29, sp").map_err(io_error)?;
    if unwind {
        writeln!(output, "    .cfi_def_cfa_register 29").map_err(io_error)?;
    }

    let has_many_vars = func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(&i.instr, MirInst::IntBinary { .. }))
        .count()
        > 100;
    let adjusted_frame_size = if has_many_vars {
        (frame.frame_size + 1024) & !15
    } else {
        frame.frame_size
    };

    // aapcs64 hands a composite over 16 bytes back through the caller's buffer,
    // whose address arrives in x8. It has to outlive the calls the body makes.
    let sret_slot = func
        .sig
        .ret_aggregate
        .filter(|aggregate| passed_in_memory(*aggregate))
        .map(|_| -adjusted_frame_size - 8);
    let adjusted_frame_size = match sret_slot {
        Some(_) => adjusted_frame_size + 16,
        None => adjusted_frame_size,
    };
    let va_save = (func.sig.variadic && os_uses_elf(target_os))
        .then(|| -adjusted_frame_size - variadic::SAVE_AREA_BYTES);
    let adjusted_frame_size = match va_save {
        Some(_) => adjusted_frame_size + variadic::SAVE_AREA_BYTES,
        None => adjusted_frame_size,
    };
    let callee_saved_bytes = (callee_saved.len() * 8).next_multiple_of(16) as i32;
    let callee_saved_base =
        (!callee_saved.is_empty()).then(|| -adjusted_frame_size - callee_saved_bytes);
    let adjusted_frame_size = adjusted_frame_size + callee_saved_bytes;

    frame.frame_size = adjusted_frame_size;
    let page_size = TargetArchitecture::Aarch64.stack_page_size() as u64;
    if target_os.stack_probe() == Some(StackProbe::PageLoop)
        && adjusted_frame_size as u64 > page_size
    {
        emit_probed_sp_sub(&mut output, adjusted_frame_size as u64, page_size)
            .map_err(lamina_to_codegen_error)?;
    } else if adjusted_frame_size > 0 {
        emit_sp_adjust(&mut output, "sub", adjusted_frame_size as u64)
            .map_err(lamina_to_codegen_error)?;
    }
    if let Some(offset) = sret_slot {
        emit_frame_slot_store(&mut output, "x8", offset).map_err(lamina_to_codegen_error)?;
    }
    if let Some(base) = callee_saved_base {
        emit_callee_saved(&mut output, &callee_saved, base, true)
            .map_err(lamina_to_codegen_error)?;
        if unwind {
            for (index, reg) in callee_saved.iter().enumerate() {
                writeln!(
                    output,
                    "    .cfi_offset {}, {}",
                    reg.trim_start_matches('x'),
                    base + index as i32 * 8 - 16
                )
                .map_err(io_error)?;
            }
        }
    }

    let va_area = func.sig.variadic.then(|| {
        let named_stack = aarch64_stack_bytes(&param_places).next_multiple_of(8) as i32;
        match va_save {
            Some(save) => {
                variadic::VaArea::aapcs64(save, FRAME_RECORD_BYTES + named_stack, &param_places)
            }
            None if os_uses_coff(target_os) => {
                variadic::VaArea::windows(frame_record, named_stack, &param_places)
            }
            None => variadic::VaArea::darwin(FRAME_RECORD_BYTES + named_stack),
        }
    });
    if let Some(area) = &va_area {
        variadic::spill_save_area(&mut output, area).map_err(lamina_to_codegen_error)?;
    }
    for ((p, kind), place) in func.sig.params.iter().zip(&param_kinds).zip(&param_places) {
        if let Some(off) = frame.slot_of(&p.reg) {
            let addr = ra_pro.alloc_scratch().ok_or_else(scratch_error)?;
            emit_frame_slot_address(&mut output, addr, off).map_err(lamina_to_codegen_error)?;

            match *place {
                ArgPlace::Fpr { index } => {
                    // The slot holds the bits, so the float register is moved through a
                    // general one on its way in.
                    let value = ra_pro.alloc_scratch().ok_or_else(scratch_error)?;
                    if p.ty.size_bytes(func.pointer_width) == 4 {
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
                ArgPlace::Homogeneous {
                    first,
                    count,
                    member,
                } => {
                    for index in 0..count {
                        writeln!(
                            output,
                            "    str {}, [{addr}, #{}]",
                            member_register(member, first + index),
                            index * member.size_bytes()
                        )
                        .map_err(|e| {
                            CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                        })?;
                    }
                }
                ArgPlace::Gpr { first, words } if kind.by_reference => {
                    let _ = words;
                    copy_through_pointer(
                        &mut output,
                        &format!("x{first}"),
                        addr,
                        p.aggregate.map_or(0, |a| a.words()),
                        &mut ra_pro,
                    )
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                }
                ArgPlace::Stack { offset, .. } if kind.by_reference => {
                    let pointer = ra_pro.alloc_scratch().ok_or_else(scratch_error)?;
                    writeln!(
                        output,
                        "    ldr {pointer}, [x29, #{}]",
                        frame_record + offset as i32
                    )
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                    copy_through_pointer(
                        &mut output,
                        pointer,
                        addr,
                        p.aggregate.map_or(0, |a| a.words()),
                        &mut ra_pro,
                    )
                    .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
                    ra_pro.free_scratch(pointer);
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
                ArgPlace::Stack { offset, bytes } => {
                    let val_reg = ra_pro.alloc_scratch().ok_or_else(scratch_error)?;
                    let words = bytes.div_ceil(8).max(1);
                    for word in 0..words {
                        let caller_offset = frame_record as usize + offset + word * 8;
                        let width = (bytes - word * 8).min(8);
                        load_argument_bytes(&mut output, val_reg, "x29", caller_offset, width)
                            .map_err(|e| {
                                CodegenError::InvalidCodegenOptions(format!("IO error: {e}"))
                            })?;
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
                initialize(&mut output).map_err(lamina_to_codegen_error)?;
            }
        }
    }

    let plan = FunctionEmissionPlan::new(func);
    let multiply_adds = multiply_adds(func, func.pointer_width);
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
        let emit = |output: &mut Vec<u8>, ra: &mut A64RegAlloc| -> Result<(), LaminaError> {
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
                } else if multiply_adds.contains(&(index, position))
                    && multiply_add_operands_ready(block, position, ra)
                {
                    if let (
                        MirInst::IntBinary {
                            dst: product,
                            lhs: Operand::Register(a),
                            rhs: Operand::Register(b),
                            ..
                        },
                        MirInst::IntBinary { dst, lhs, rhs, .. },
                    ) = (&inst.instr, &block.instructions[position + 1].instr)
                    {
                        emit_source_loc(output, settings, inst.loc)?;
                        let addend = match (lhs, rhs) {
                            (Operand::Register(r), other) if r == product => other,
                            (other, _) => other,
                        };
                        let mut scratches = Vec::new();
                        let mut operands = Vec::with_capacity(3);
                        for operand in [
                            Operand::Register(a.clone()),
                            Operand::Register(b.clone()),
                            addend.clone(),
                        ] {
                            operands.push(match operand_in_register(&operand, ra) {
                                Some(preg) => x_alias(preg),
                                None => {
                                    let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                    scratches.push(s);
                                    emit_materialize_operand(output, &operand, s, &frame, ra)?;
                                    x_alias(s)
                                }
                            });
                        }
                        let held = dest_in_register(dst, ra);
                        let target = match held {
                            Some(preg) => preg,
                            None => {
                                let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                scratches.push(s);
                                s
                            }
                        };
                        writeln!(
                            output,
                            "    madd {}, {}, {}, {}",
                            x_alias(target),
                            operands[0],
                            operands[1],
                            operands[2]
                        )?;
                        if held.is_none() {
                            store_result(output, dst, &x_alias(target), &frame, ra)?;
                        }
                        for scratch in scratches {
                            ra.free_scratch(scratch);
                        }
                    }
                    position += 1;
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
                        emit_source_loc(output, settings, inst.loc)?;
                        let unsigned = matches!(
                            op,
                            IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                        );
                        if ty.size_bytes(func.pointer_width) == 8 {
                            // Nothing to extend, so read registers in place and fold the constant.
                            let left = match operand_in_register(lhs, ra) {
                                Some(preg) => x_alias(preg),
                                None => {
                                    emit_materialize_operand(output, lhs, "x16", &frame, ra)?;
                                    "x16".to_string()
                                }
                            };
                            let folded = match rhs {
                                Operand::Immediate(imm) => fold_int_cmp_immediate(imm_to_u64(imm)),
                                Operand::Register(_) => None,
                            };
                            if let Some((mnemonic, operand)) = folded {
                                writeln!(output, "    {mnemonic} {left}, {operand}")?;
                            } else if let Some(preg) = operand_in_register(rhs, ra) {
                                writeln!(output, "    cmp {left}, {}", x_alias(preg))?;
                            } else {
                                emit_materialize_operand(output, rhs, "x17", &frame, ra)?;
                                writeln!(output, "    cmp {left}, x17")?;
                            }
                        } else {
                            emit_materialize_operand(output, lhs, "x16", &frame, ra)?;
                            emit_materialize_operand(output, rhs, "x17", &frame, ra)?;
                            extend_integer(output, "x16", ty, !unsigned, func.pointer_width)?;
                            extend_integer(output, "x17", ty, !unsigned, func.pointer_width)?;
                            writeln!(output, "    cmp x16, x17")?;
                        }
                        let (invert, taken, other) =
                            plan.branch(func, index, next, true_target, false_target);
                        let direct = int_condition(*op);
                        let cond = if invert {
                            invert_condition(direct).ok_or_else(|| unknown_condition(direct))?
                        } else {
                            direct
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
                } else if matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
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
                        inst.loc,
                        long_branches,
                        signatures,
                        func.sig.ret_aggregate,
                        func.sig.ret_ty.filter(|ty| ty.is_float()),
                        sret_slot,
                        va_area,
                        func.pointer_width,
                    )?;
                }
                position += 1;
            }
            Ok(())
        };
        emit(&mut output, &mut ra).map_err(lamina_to_codegen_error)?;
    }

    writeln!(output, "{epilogue_label}:")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    if unwind {
        writeln!(output, "    .cfi_remember_state").map_err(io_error)?;
    }
    if let Some(base) = callee_saved_base {
        emit_callee_saved(&mut output, &callee_saved, base, false)
            .map_err(lamina_to_codegen_error)?;
        if unwind {
            for reg in &callee_saved {
                writeln!(output, "    .cfi_restore {}", reg.trim_start_matches('x'))
                    .map_err(io_error)?;
            }
        }
    }
    if adjusted_frame_size > 0 {
        emit_sp_adjust(&mut output, "add", adjusted_frame_size as u64)
            .map_err(lamina_to_codegen_error)?;
    }
    writeln!(output, "    ldp x29, x30, [sp], #{frame_record}").map_err(io_error)?;
    if unwind {
        writeln!(output, "    .cfi_def_cfa sp, 0").map_err(io_error)?;
        writeln!(output, "    .cfi_restore 29").map_err(io_error)?;
        writeln!(output, "    .cfi_restore 30").map_err(io_error)?;
    }
    writeln!(output, "    ret").map_err(io_error)?;
    if unwind {
        writeln!(output, "    .cfi_restore_state").map_err(io_error)?;
        writeln!(output, "    .cfi_endproc").map_err(io_error)?;
    }
    if let Some(encoding) = macho_unwind::plan_aarch64(target_os, frame_record, &callee_saved) {
        writeln!(output, "{}:", macho_unwind::function_end_label(&label)).map_err(io_error)?;
        macho_unwind::emit(&mut output, &label, encoding).map_err(io_error)?;
    }

    Ok(output)
}

/// Generate AArch64 assembly from a MIR module.
/// Maps the raw result of a read or write call to the I/O contract. A failure becomes
/// -2, which `carry` reads from the carry flag as Mach-O reports it and otherwise from
/// the sign of the result. With `end_of_input`, a zero count becomes -1.
fn emit_io_result<W: Write>(w: &mut W, carry: bool, end_of_input: bool) -> Result<(), LaminaError> {
    if carry {
        writeln!(w, "    mov x9, #-2")?;
        writeln!(w, "    csel x0, x9, x0, cs")?;
    } else {
        writeln!(w, "    cmp x0, #0")?;
        writeln!(w, "    mov x9, #-2")?;
        writeln!(w, "    csel x0, x9, x0, lt")?;
    }
    if end_of_input {
        writeln!(w, "    cmp x0, #0")?;
        writeln!(w, "    mov x9, #-1")?;
        writeln!(w, "    csel x0, x9, x0, eq")?;
    }
    Ok(())
}

pub fn generate_mir_aarch64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    generate_mir_aarch64_with_units(module, writer, target_os, 1, pointer_width)
}

pub fn generate_mir_aarch64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    generate_mir_aarch64_with_units_and_settings(
        module,
        writer,
        target_os,
        codegen_units,
        &MirCodegenSettings::default(),
        pointer_width,
    )
}

pub fn generate_mir_aarch64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    if os_uses_coff(target_os) {
        reject_thread_local_globals(module, "AArch64 (COFF)")?;
    }
    validate_module_call_parameters(module, TargetArchitecture::Aarch64)?;
    let abi = AArch64ABI::new(target_os);
    reject_register_symbols(module, &abi)?;
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(AggregateSignatures::from_module(module, pointer_width));
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

    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings, module)?;

    for func_name in external_names(module) {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {label}")?;
    }
    emit_global_directives(writer, module, |name| abi.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        abi.mangle_function_name(name)
    })?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }
    if os_uses_macho(target_os) {
        emit_static_global_data(writer, module, target_os, |name| {
            abi.mangle_function_name(name)
        })?;
        emit_macho_thread_local_globals(writer, module, |name| abi.mangle_function_name(name))?;
    } else {
        emit_global_data(writer, module, target_os, |name| {
            abi.mangle_function_name(name)
        })?;
    }
    emit_gnu_stack_note(writer, target_os)?;

    Ok(())
}

/// Whether the assembler reads `label` as a register name rather than a symbol.
fn is_aarch64_register_name(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "sp" | "wsp" | "xzr" | "wzr" | "lr" | "fp" | "ip0" | "ip1"
    ) {
        return true;
    }
    let mut chars = lower.chars();
    let (Some(prefix), digits) = (chars.next(), chars.as_str()) else {
        return false;
    };
    let last = match prefix {
        'x' | 'w' | 'b' | 'h' | 's' | 'd' | 'q' | 'v' | 'z' => 31,
        'p' => 15,
        _ => return false,
    };
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && digits.parse::<u8>().is_ok_and(|n| n <= last)
}

/// A function named like a register would be emitted as `bl <name>`, which the assembler
/// rejects. Symbols are not mangled to avoid that, since the name is the ABI symbol.
fn reject_register_symbols(module: &MirModule, abi: &AArch64ABI) -> Result<(), LaminaError> {
    let mut names: Vec<&str> = module.functions.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        if is_aarch64_register_name(&abi.mangle_function_name(name)) {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!(
                    "@{name} is an AArch64 register name, so the assembler cannot use it as a branch target; rename the function"
                ),
            )));
        }
    }
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
    loc: Option<SourceLoc>,
    long_branches: bool,
    signatures: &AggregateSignatures,
    ret_aggregate: Option<Aggregate>,
    ret_float: Option<MirType>,
    sret_slot: Option<i32>,
    va_area: Option<variadic::VaArea>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    for inst in insts {
        if settings.emit_asm_debug_lines
            && let Some(loc) = loc
        {
            writeln!(w, "    .loc {} {} {}", loc.file, loc.line, loc.column)?;
        }
        match inst {
            MirInst::ThreadLocalAddr { dst, symbol, model } => {
                let target_os = abi.target_os();
                if os_uses_coff(target_os) {
                    return Err(unsupported_thread_local("AArch64"));
                }
                if os_uses_macho(target_os) {
                    let descriptor = abi.mangle_function_name(symbol);
                    writeln!(w, "    adrp x0, {descriptor}@TLVPPAGE")?;
                    writeln!(w, "    ldr x0, [x0, {descriptor}@TLVPPAGEOFF]")?;
                    writeln!(w, "    ldr x8, [x0]")?;
                    writeln!(w, "    blr x8")?;
                    store_result(w, dst, "x0", frame, ra)?;
                } else {
                    let label = abi.mangle_function_name(symbol);
                    let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    let reg = x_alias(scratch);
                    match model {
                        TlsModel::LocalExec => {
                            writeln!(w, "    mrs {reg}, tpidr_el0")?;
                            writeln!(w, "    add {reg}, {reg}, :tprel_hi12:{label}, lsl #12")?;
                            writeln!(w, "    add {reg}, {reg}, :tprel_lo12_nc:{label}")?;
                        }
                        TlsModel::InitialExec => {
                            let thread_pointer = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            let tp = x_alias(thread_pointer);
                            writeln!(w, "    adrp {reg}, :gottprel:{label}")?;
                            writeln!(w, "    ldr {reg}, [{reg}, :gottprel_lo12:{label}]")?;
                            writeln!(w, "    mrs {tp}, tpidr_el0")?;
                            writeln!(w, "    add {reg}, {reg}, {tp}")?;
                            ra.free_scratch(thread_pointer);
                        }
                    }
                    store_result(w, dst, &reg, frame, ra)?;
                    ra.free_scratch(scratch);
                }
            }
            MirInst::IntOverflow {
                op,
                ty,
                signed,
                dst,
                lhs,
                rhs,
            } => overflow::emit(w, *op, ty, *signed, dst, lhs, rhs, frame, ra, pointer_width)?,
            MirInst::SymbolAddress { dst, symbol } => {
                let label = abi.mangle_function_name(symbol);
                let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                if os_uses_macho(abi.target_os()) {
                    writeln!(w, "    adrp {scratch}, {label}@PAGE")?;
                    writeln!(w, "    add {scratch}, {scratch}, {label}@PAGEOFF")?;
                } else if os_uses_elf(abi.target_os()) && signatures.is_external(symbol) {
                    // A PIE link refuses a PC-relative reference to a symbol that may
                    // come from a shared library.
                    writeln!(w, "    adrp {scratch}, :got:{label}")?;
                    writeln!(w, "    ldr {scratch}, [{scratch}, :got_lo12:{label}]")?;
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
                if let Some(operations) = integer_conversion(inst, pointer_width) {
                    emit_block(
                        &operations,
                        w,
                        frame,
                        abi,
                        ra,
                        epilogue_label,
                        block_prefix,
                        settings,
                        None,
                        long_branches,
                        signatures,
                        ret_aggregate,
                        ret_float,
                        sret_slot,
                        va_area,
                        pointer_width,
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
                // A whole word needs no extension, so it goes straight across.
                if ty.size_bytes(pointer_width) == 8
                    && let Some(to) = dest_in_register(dst, ra)
                {
                    let to = x_alias(to);
                    match src {
                        Operand::Immediate(immediate) => {
                            emit_mov_imm64(w, &to, imm_to_u64(immediate))?
                        }
                        Operand::Register(register) => {
                            load_copy_reg_to(w, register, &to, 8, frame, ra)?
                        }
                    }
                    continue;
                }
                let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let value = x_alias(scratch);
                match src {
                    Operand::Immediate(immediate) => {
                        emit_mov_imm64(w, &value, imm_to_u64(immediate))?
                    }
                    Operand::Register(register) => load_copy_reg_to(
                        w,
                        register,
                        &value,
                        ty.size_bytes(pointer_width),
                        frame,
                        ra,
                    )?,
                }
                extend_integer(w, &value, ty, true, pointer_width)?;
                store_result(w, dst, &value, frame, ra)?;
                ra.free_scratch(scratch);
            }
            MirInst::IntBinary {
                op: op @ (IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::UDiv | IntBinOp::URem),
                lhs: Operand::Register(numerator),
                rhs: Operand::Immediate(divisor),
                dst,
                ty,
            } if ty.size_bytes(pointer_width) == 8
                && has_reciprocal(*op, imm_to_u64(divisor) as i64) =>
            {
                let divisor = imm_to_u64(divisor) as i64;
                let mut n_scratch = None;
                let n = match numerator {
                    Register::Virtual(v) if ra.get_mapping_for(v).is_some() => {
                        ra.get_mapping_for(v).map(x_alias).unwrap_or_default()
                    }
                    _ => {
                        let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        n_scratch = Some(scratch);
                        load_reg_to(w, numerator, scratch, frame, ra)?;
                        x_alias(scratch)
                    }
                };
                let quotient = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let q = x_alias(quotient);
                let helper = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let m = x_alias(helper);
                if matches!(op, IntBinOp::SDiv | IntBinOp::SRem) {
                    let (multiplier, shift) = div::signed_reciprocal(divisor);
                    emit_mov_imm64(w, &m, multiplier as u64)?;
                    writeln!(w, "    smulh {q}, {n}, {m}")?;
                    if divisor > 0 && multiplier < 0 {
                        writeln!(w, "    add {q}, {q}, {n}")?;
                    } else if divisor < 0 && multiplier > 0 {
                        writeln!(w, "    sub {q}, {q}, {n}")?;
                    }
                    if shift > 0 {
                        writeln!(w, "    asr {q}, {q}, #{shift}")?;
                    }
                    writeln!(w, "    add {q}, {q}, {q}, lsr #63")?;
                } else {
                    let (multiplier, add, shift) = div::unsigned_reciprocal(divisor as u64);
                    emit_mov_imm64(w, &m, multiplier)?;
                    writeln!(w, "    umulh {q}, {n}, {m}")?;
                    if add {
                        writeln!(w, "    sub {m}, {n}, {q}")?;
                        writeln!(w, "    add {q}, {q}, {m}, lsr #1")?;
                        if shift > 1 {
                            writeln!(w, "    lsr {q}, {q}, #{}", shift - 1)?;
                        }
                    } else if shift > 0 {
                        writeln!(w, "    lsr {q}, {q}, #{shift}")?;
                    }
                }
                if matches!(op, IntBinOp::SRem | IntBinOp::URem) {
                    emit_mov_imm64(w, &m, divisor as u64)?;
                    writeln!(w, "    msub {q}, {q}, {m}, {n}")?;
                }
                store_result(w, dst, &q, frame, ra)?;
                ra.free_scratch(helper);
                ra.free_scratch(quotient);
                if let Some(scratch) = n_scratch {
                    ra.free_scratch(scratch);
                }
            }
            MirInst::IntBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let (zero, overflow) = op.division_checks(
                    *ty,
                    lhs.integer_constant(),
                    rhs.integer_constant(),
                    pointer_width,
                );
                let narrow = matches!(
                    ty,
                    MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
                );
                let is_shift = matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr);
                let masks_shift_amount = is_shift && ty.size_bytes(pointer_width) < 8;
                let rewrites_lhs = narrow
                    && matches!(
                        op,
                        IntBinOp::UDiv
                            | IntBinOp::URem
                            | IntBinOp::LShr
                            | IntBinOp::SDiv
                            | IntBinOp::SRem
                            | IntBinOp::AShr
                    );
                let rewrites_rhs = (narrow
                    && matches!(
                        op,
                        IntBinOp::UDiv
                            | IntBinOp::URem
                            | IntBinOp::LShr
                            | IntBinOp::SDiv
                            | IntBinOp::SRem
                    ))
                    || masks_shift_amount;
                let folded_rhs = match rhs {
                    Operand::Immediate(immediate) if !zero && !overflow && !rewrites_rhs => {
                        fold_int_binary_immediate(*op, imm_to_u64(immediate))
                    }
                    _ => None,
                };
                let rhs_unused = folded_rhs.is_some()
                    || (is_shift && matches!(rhs, Operand::Immediate(_)) && !zero && !overflow);
                let dst_written_once =
                    !zero && !overflow && !matches!(op, IntBinOp::URem | IntBinOp::SRem);

                let mut lhs_scratch = None;
                let rl = match operand_in_register(lhs, ra) {
                    Some(preg) if !rewrites_lhs => x_alias(preg),
                    _ => {
                        let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        lhs_scratch = Some(s);
                        x_alias(s)
                    }
                };
                let mut rhs_scratch = None;
                let rr = if rhs_unused {
                    "<unused>".to_string()
                } else {
                    match operand_in_register(rhs, ra) {
                        Some(preg) if !rewrites_rhs => x_alias(preg),
                        _ => {
                            let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            rhs_scratch = Some(s);
                            x_alias(s)
                        }
                    }
                };
                let mut dst_scratch = None;
                let dl = match dest_in_register(dst, ra) {
                    Some(preg) if dst_written_once => x_alias(preg),
                    _ => {
                        let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        dst_scratch = Some(s);
                        x_alias(s)
                    }
                };

                if let Some(s) = lhs_scratch {
                    emit_materialize_operand(w, lhs, s, frame, ra)?;
                }
                if let Some(s) = rhs_scratch {
                    emit_materialize_operand(w, rhs, s, frame, ra)?;
                }
                if matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr) {
                    extend_integer(w, &rl, ty, false, pointer_width)?;
                    if !rhs_unused {
                        extend_integer(w, &rr, ty, false, pointer_width)?;
                    }
                }
                if matches!(op, IntBinOp::SDiv | IntBinOp::SRem | IntBinOp::AShr) {
                    extend_integer(w, &rl, ty, true, pointer_width)?;
                    if !matches!(op, IntBinOp::AShr) && !rhs_unused {
                        extend_integer(w, &rr, ty, true, pointer_width)?;
                    }
                }
                if zero {
                    writeln!(w, "    cbnz {rr}, 1f\n    udf #0\n1:")?;
                }
                if overflow {
                    let minimum = 1u64 << (ty.size_bytes(pointer_width) * 8 - 1);
                    writeln!(w, "    cmn {rr}, #1\n    b.ne 2f")?;
                    emit_mov_imm64(
                        w,
                        &dl,
                        ty.signed_integer(minimum as i64, pointer_width) as u64,
                    )?;
                    writeln!(w, "    cmp {rl}, {dl}\n    b.ne 2f\n    udf #0\n2:")?;
                }
                if masks_shift_amount && !rhs_unused {
                    let mask = ty.size_bytes(pointer_width) * 8 - 1;
                    writeln!(w, "    and {rr}, {rr}, #{mask}")?;
                }
                if let Some((mnemonic, operand)) = &folded_rhs {
                    writeln!(w, "    {mnemonic} {dl}, {rl}, {operand}")?;
                } else {
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
                                sh &= ty.size_bytes(pointer_width) as u32 * 8 - 1;
                                writeln!(w, "    lsl {dl}, {rl}, #{sh}")?;
                            }
                            _ => {
                                writeln!(w, "    lslv {dl}, {rl}, {rr}")?;
                            }
                        },
                        IntBinOp::LShr => match rhs {
                            Operand::Immediate(imm) => {
                                let mut sh = imm_to_u64(imm) as u32;
                                sh &= ty.size_bytes(pointer_width) as u32 * 8 - 1;
                                writeln!(w, "    lsr {dl}, {rl}, #{sh}")?;
                            }
                            _ => {
                                writeln!(w, "    lsrv {dl}, {rl}, {rr}")?;
                            }
                        },
                        IntBinOp::AShr => match rhs {
                            Operand::Immediate(imm) => {
                                let mut sh = imm_to_u64(imm) as u32;
                                sh &= ty.size_bytes(pointer_width) as u32 * 8 - 1;
                                writeln!(w, "    asr {dl}, {rl}, #{sh}")?;
                            }
                            _ => {
                                writeln!(w, "    asrv {dl}, {rl}, {rr}")?;
                            }
                        },
                    }
                }
                extend_integer(w, &dl, ty, true, pointer_width)?;
                if let Some(s) = dst_scratch {
                    store_result(w, dst, &x_alias(s), frame, ra)?;
                }
                if let Some(s) = lhs_scratch {
                    ra.free_scratch(s);
                }
                if let Some(s) = rhs_scratch {
                    ra.free_scratch(s);
                }
                if let Some(s) = dst_scratch {
                    ra.free_scratch(s);
                }
            }
            MirInst::FloatBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let s_r = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let s_d = ra.alloc_scratch().ok_or_else(scratch_error)?;
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes(pointer_width) == 4;
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
            MirInst::IntUnary { op, ty, dst, src } => {
                bit::emit_int_unary(w, *op, ty, dst, src, frame, ra, pointer_width)?;
            }
            MirInst::IntRotate {
                op,
                ty,
                dst,
                lhs,
                rhs,
            } => {
                bit::emit_int_rotate(w, *op, ty, dst, (lhs, rhs), frame, ra, pointer_width)?;
            }
            MirInst::FloatUnary { op, src, dst, ty } => {
                let s_s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let s_d = ra.alloc_scratch().ok_or_else(scratch_error)?;
                emit_materialize_operand(w, src, s_s, frame, ra)?;

                let is32 = ty.size_bytes(pointer_width) == 4;
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
                let unsigned_extend = matches!(
                    ty,
                    MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
                ) && matches!(
                    op,
                    IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
                );
                let mut lhs_scratch = None;
                let rl = match operand_in_register(lhs, ra) {
                    Some(preg) if !unsigned_extend => x_alias(preg),
                    _ => {
                        let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        lhs_scratch = Some(s);
                        x_alias(s)
                    }
                };
                let folded_rhs = match rhs {
                    Operand::Immediate(immediate) => {
                        let value = imm_to_u64(immediate);
                        let value = if unsigned_extend {
                            let bits = ty.size_bytes(pointer_width) as u32 * 8;
                            value & ((1u64 << bits) - 1)
                        } else {
                            value
                        };
                        fold_int_cmp_immediate(value)
                    }
                    _ => None,
                };
                let mut rhs_scratch = None;
                let rr = if folded_rhs.is_some() {
                    "<unused>".to_string()
                } else {
                    match operand_in_register(rhs, ra) {
                        Some(preg) if !unsigned_extend => x_alias(preg),
                        _ => {
                            let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            rhs_scratch = Some(s);
                            x_alias(s)
                        }
                    }
                };
                let mut dst_scratch = None;
                let dl = match dest_in_register(dst, ra) {
                    Some(preg) => x_alias(preg),
                    None => {
                        let s = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        dst_scratch = Some(s);
                        x_alias(s)
                    }
                };
                if let Some(s) = lhs_scratch {
                    emit_materialize_operand(w, lhs, s, frame, ra)?;
                }
                if let Some(s) = rhs_scratch {
                    emit_materialize_operand(w, rhs, s, frame, ra)?;
                }
                if unsigned_extend {
                    extend_integer(w, &rl, ty, false, pointer_width)?;
                    if folded_rhs.is_none() {
                        extend_integer(w, &rr, ty, false, pointer_width)?;
                    }
                }
                match &folded_rhs {
                    Some((mnemonic, operand)) => writeln!(w, "    {mnemonic} {rl}, {operand}")?,
                    None => writeln!(w, "    cmp {rl}, {rr}")?,
                }
                let cond = int_condition(*op);
                writeln!(w, "    cset {dl}, {cond}")?;
                if let Some(s) = dst_scratch {
                    store_result(w, dst, &x_alias(s), frame, ra)?;
                }
                if let Some(s) = lhs_scratch {
                    ra.free_scratch(s);
                }
                if let Some(s) = rhs_scratch {
                    ra.free_scratch(s);
                }
                if let Some(s) = dst_scratch {
                    ra.free_scratch(s);
                }
            }
            MirInst::FloatCmp {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let s_r = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let s_d = ra.alloc_scratch().ok_or_else(scratch_error)?;
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes(pointer_width) == 4;

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_l))?;
                    writeln!(w, "    fmov s1, {}", w_alias(s_r))?;
                    writeln!(w, "    fcmp s0, s1")?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_l))?;
                    writeln!(w, "    fmov d1, {}", x_alias(s_r))?;
                    writeln!(w, "    fcmp d0, d1")?;
                }

                let cond = float_condition(*op);

                writeln!(w, "    cset {}, {}", x_alias(s_d), cond)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::Load { ty, dst, addr, .. } => {
                let size = ty.size_bytes(pointer_width);
                let held = dest_in_register(dst, ra);
                let t = match held {
                    Some(preg) => preg,
                    None => ra.alloc_scratch().ok_or_else(scratch_error)?,
                };
                let (place, scratches) = emit_access_address(w, addr, size, frame, ra)?;
                match size {
                    1 => writeln!(w, "    ldrsb {}, {place}", x_alias(t))?,
                    2 => writeln!(w, "    ldrsh {}, {place}", x_alias(t))?,
                    4 => writeln!(w, "    ldrsw {}, {place}", x_alias(t))?,
                    8 => writeln!(w, "    ldr {}, {place}", x_alias(t))?,
                    _ => writeln!(w, "    // Unhandled load size: {size}")?,
                }
                if held.is_none() {
                    store_result(w, dst, &x_alias(t), frame, ra)?;
                    ra.free_scratch(t);
                }
                for scratch in scratches {
                    ra.free_scratch(scratch);
                }
            }
            MirInst::Select {
                ty,
                dst,
                cond,
                true_val,
                false_val,
            } => {
                let r_cond = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let r_t = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let r_f = ra.alloc_scratch().ok_or_else(scratch_error)?;
                load_reg_to(w, cond, r_cond, frame, ra)?;
                emit_materialize_operand(w, true_val, r_t, frame, ra)?;
                emit_materialize_operand(w, false_val, r_f, frame, ra)?;
                if ty.size_bytes(pointer_width) == 4 {
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
                let size = ty.size_bytes(pointer_width);
                let held = operand_in_register(src, ra);
                let t = match held {
                    Some(preg) => preg,
                    None => ra.alloc_scratch().ok_or_else(scratch_error)?,
                };
                let (place, scratches) = emit_access_address(w, addr, size, frame, ra)?;
                if held.is_none() {
                    emit_materialize_operand(w, src, t, frame, ra)?;
                }
                match size {
                    1 => writeln!(w, "    strb {}, {place}", w_alias(t))?,
                    2 => writeln!(w, "    strh {}, {place}", w_alias(t))?,
                    4 => writeln!(w, "    str {}, {place}", w_alias(t))?,
                    8 => writeln!(w, "    str {}, {place}", x_alias(t))?,
                    _ => writeln!(w, "    // Unhandled store size: {size}")?,
                }
                if held.is_none() {
                    ra.free_scratch(t);
                }
                for scratch in scratches {
                    ra.free_scratch(scratch);
                }
            }
            MirInst::StackAddr { dst, slot } => {
                let offset = frame.slot_of(&Register::Virtual(*slot)).ok_or_else(|| {
                    LaminaError::ValidationError("stack address has no frame slot".into())
                })?;
                let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
                emit_mov_imm64(w, t, i64::from(offset) as u64)?;
                writeln!(w, "    add {t}, x29, {t}")?;
                store_result(w, dst, t, frame, ra)?;
                ra.free_scratch(t);
            }
            MirInst::Lea { dst, base, offset } => {
                let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
                // The base register holds the pointer value.
                load_reg_to(w, base, t, frame, ra)?;
                if *offset != 0 {
                    let off = *offset as i64;
                    if (0..=4095).contains(&off) {
                        writeln!(w, "    add {t}, {t}, #{off}")?;
                    } else if off < 0 && -off <= 4095 {
                        writeln!(w, "    sub {}, {}, #{}", t, t, -off)?;
                    } else {
                        let oreg = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        emit_mov_imm64(w, oreg, off as u64)?;
                        writeln!(w, "    add {t}, {t}, {oreg}")?;
                        ra.free_scratch(oreg);
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
            } => emit_int_to_float(w, from, to, *signed, dst, src, frame, ra, pointer_width)?,
            MirInst::FloatConvert { to, dst, src, .. } => {
                emit_float_convert(w, to, dst, src, frame, ra, pointer_width)?;
            }
            MirInst::FloatToInt {
                from,
                to,
                signed,
                dst,
                src,
            } => emit_float_to_int(w, from, to, *signed, dst, src, frame, ra, pointer_width)?,
            MirInst::CallIndirect {
                target,
                arg_types,
                args,
                variadic_fixed,
                ret,
                ret_ty,
            } => {
                let kinds: Vec<ArgKind> = arg_types
                    .iter()
                    .map(|ty| ArgKind::of(*ty, None, pointer_width))
                    .collect();
                let places = classify_aarch64_arguments(&kinds, abi.target_os(), *variadic_fixed);
                let stack_space = (aarch64_stack_bytes(&places) + 15) & !15;
                if stack_space > 0 {
                    emit_sp_adjust(w, "sub", stack_space as u64)?;
                    for (argument, place) in args.iter().zip(&places) {
                        let ArgPlace::Stack { offset, bytes } = *place else {
                            continue;
                        };
                        let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        emit_materialize_operand(w, argument, scratch, frame, ra)?;
                        store_argument_bytes(w, scratch, "sp", offset, bytes)?;
                        ra.free_scratch(scratch);
                    }
                }
                for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                    match *place {
                        ArgPlace::Fpr { index } => {
                            let bits = ra.alloc_scratch().ok_or_else(scratch_error)?;
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
                        ArgPlace::Stack { .. } => {}
                        ArgPlace::Homogeneous { .. } => {
                            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                                "aarch64 indirect call with a homogeneous float aggregate argument"
                                    .to_string(),
                            )));
                        }
                    }
                }
                // x16 is the intra-procedure scratch register, so it is free here and
                // is not one of the argument registers already filled.
                emit_materialize_operand(w, target, "x16", frame, ra)?;
                writeln!(w, "    blr x16")?;
                if stack_space > 0 {
                    emit_sp_adjust(w, "add", stack_space as u64)?;
                }
                if let Some(dst) = ret {
                    match ret_ty {
                        Some(ty) if ty.is_float() && ty.size_bytes(pointer_width) == 4 => {
                            writeln!(w, "    fmov w0, s0")?;
                            store_result(w, dst, "x0", frame, ra)?;
                        }
                        Some(ty) if ty.is_float() => {
                            writeln!(w, "    fmov x0, d0")?;
                            store_result(w, dst, "x0", frame, ra)?;
                        }
                        _ => store_result(w, dst, "x0", frame, ra)?,
                    }
                }
            }
            MirInst::Call { name, args, ret } => {
                if name == "lamina.writebyte" && args.len() == 1 {
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
                            emit_io_result(w, true, false)?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        os @ (TargetOperatingSystem::Linux | TargetOperatingSystem::Windows) => {
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl {}", libc_write(os))?;
                            emit_io_result(w, false, false)?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        other => {
                            return Err(unsupported_io(name, other));
                        }
                    }
                } else if name == "lamina.readbyte" && args.is_empty() {
                    writeln!(w, "    sub sp, sp, #16")?;
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            writeln!(w, "    csinv x0, x0, xzr, cc")?;
                        }
                        os @ (TargetOperatingSystem::Linux | TargetOperatingSystem::Windows) => {
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl {}", libc_read(os))?;
                        }
                        other => {
                            return Err(unsupported_io(name, other));
                        }
                    }
                    if let Some(dst) = ret {
                        let dst_reg = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        let d = x_alias(dst_reg);
                        writeln!(w, "    ldrb {}, [sp]", w_alias(dst_reg))?;
                        writeln!(w, "    subs x0, x0, #1")?;
                        writeln!(w, "    csel {d}, {d}, x0, eq")?;
                        store_result(w, dst, &d, frame, ra)?;
                        ra.free_scratch(dst_reg);
                    }
                    writeln!(w, "    add sp, sp, #16")?;
                } else if name == "lamina.write" && args.len() == 2 {
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            emit_io_result(w, true, false)?;
                        }
                        os @ (TargetOperatingSystem::Linux | TargetOperatingSystem::Windows) => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    bl {}", libc_write(os))?;
                            emit_io_result(w, false, false)?;
                        }
                        other => {
                            return Err(unsupported_io(name, other));
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else if name == "lamina.read" && args.len() == 2 {
                    match abi.target_os() {
                        os if os_uses_macho(os) => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    cbz x2, 2f")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            emit_io_result(w, true, true)?;
                            writeln!(w, "2:")?;
                        }
                        os @ (TargetOperatingSystem::Linux | TargetOperatingSystem::Windows) => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    cbz x2, 2f")?;
                            writeln!(w, "    bl {}", libc_read(os))?;
                            emit_io_result(w, false, true)?;
                            writeln!(w, "2:")?;
                        }
                        other => {
                            return Err(unsupported_io(name, other));
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else {
                    let kinds = signatures.arguments(name, args);
                    let places = classify_aarch64_arguments(
                        &kinds,
                        abi.target_os(),
                        signatures.fixed_arguments(name),
                    );
                    for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                        match *place {
                            ArgPlace::Fpr { index } => {
                                // Values are held as bits in general registers, so a
                                // float argument moves across on the way out.
                                let bits = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                emit_materialize_operand(w, argument, bits, frame, ra)?;
                                if kind.ty_is_f32 {
                                    writeln!(w, "    fmov s{index}, {}", w_alias(bits))?;
                                } else {
                                    writeln!(w, "    fmov d{index}, {bits}")?;
                                }
                                ra.free_scratch(bits);
                            }
                            ArgPlace::Homogeneous {
                                first,
                                count,
                                member,
                            } => {
                                let address = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                emit_materialize_operand(w, argument, address, frame, ra)?;
                                for index in 0..count {
                                    writeln!(
                                        w,
                                        "    ldr {}, [{address}, #{}]",
                                        member_register(member, first + index),
                                        index * member.size_bytes()
                                    )?;
                                }
                                ra.free_scratch(address);
                            }
                            ArgPlace::Gpr { .. } if kind.by_reference => {}
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
                                let address = ra.alloc_scratch().ok_or_else(scratch_error)?;
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

                    let stack_bytes = aarch64_stack_bytes(&places);
                    // Each by-reference aggregate needs a copy of its own, which the
                    // callee may write to. They sit above the stack arguments.
                    let mut copy_offsets = Vec::with_capacity(kinds.len());
                    let mut copy_bytes = 0;
                    for kind in &kinds {
                        if kind.by_reference {
                            copy_offsets.push(Some(((stack_bytes + 15) & !15) + copy_bytes));
                            copy_bytes += (kind.aggregate.map_or(0, |a| a.words()) * 8 + 15) & !15;
                        } else {
                            copy_offsets.push(None);
                        }
                    }
                    let stack_space = (((stack_bytes + 15) & !15) + copy_bytes + 15) & !15;

                    if stack_space > 0 {
                        emit_sp_adjust(w, "sub", stack_space as u64)?;
                        for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                            let ArgPlace::Stack { offset, bytes } = *place else {
                                continue;
                            };
                            if kind.by_reference {
                                continue;
                            }
                            let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            if kind.aggregate.is_none() {
                                emit_materialize_operand(w, argument, scratch, frame, ra)?;
                                store_argument_bytes(w, scratch, "sp", offset, bytes)?;
                            } else {
                                let address = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                emit_materialize_operand(w, argument, address, frame, ra)?;
                                for word in 0..bytes.div_ceil(8) {
                                    writeln!(w, "    ldr {scratch}, [{address}, #{}]", word * 8)?;
                                    writeln!(w, "    str {scratch}, [sp, #{}]", offset + word * 8)?;
                                }
                                ra.free_scratch(address);
                            }
                            ra.free_scratch(scratch);
                        }
                    }

                    for ((argument, kind), (place, copy_at)) in args
                        .iter()
                        .zip(&kinds)
                        .zip(places.iter().zip(&copy_offsets))
                    {
                        let Some(copy_at) = *copy_at else {
                            continue;
                        };
                        let words = kind.aggregate.map_or(0, |a| a.words());
                        let source = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        let target = ra.alloc_scratch().ok_or_else(scratch_error)?;
                        emit_materialize_operand(w, argument, source, frame, ra)?;
                        writeln!(w, "    add {target}, sp, #{copy_at}")?;
                        copy_through_pointer(w, source, target, words, ra)?;
                        match *place {
                            ArgPlace::Gpr { first, .. } => {
                                writeln!(w, "    mov x{first}, {target}")?;
                            }
                            ArgPlace::Stack { offset, .. } => {
                                writeln!(w, "    str {target}, [sp, #{offset}]")?;
                            }
                            ArgPlace::Fpr { .. } | ArgPlace::Homogeneous { .. } => {}
                        }
                        ra.free_scratch(source);
                        ra.free_scratch(target);
                    }

                    let target_sym: String = abi
                        .call_stub(name)
                        .unwrap_or_else(|| abi.mangle_function_name(name));
                    match signatures
                        .result(name)
                        .filter(|aggregate| passed_in_memory(*aggregate))
                        .and(ret.as_ref())
                        .and_then(|dst| frame.slot_of(dst))
                    {
                        Some(off) => emit_frame_slot_address(w, "x8", off)?,
                        None => writeln!(w, "    mov x8, xzr")?,
                    }
                    writeln!(w, "    bl {target_sym}")?;

                    if stack_space > 0 {
                        emit_sp_adjust(w, "add", stack_space as u64)?;
                    }

                    if let Some(dst) = ret
                        && let Some(ty) = signatures.float_result(name)
                    {
                        // The result arrives in v0 and the body wants its bits.
                        if ty.size_bytes(pointer_width) == 4 {
                            writeln!(w, "    fmov w0, s0")?;
                        } else {
                            writeln!(w, "    fmov x0, d0")?;
                        }
                        store_result(w, dst, "x0", frame, ra)?;
                    } else if let Some(dst) = ret {
                        match signatures.result(name) {
                            Some(aggregate) if aggregate.homogeneous_members().is_some() => {
                                let (member, count) =
                                    aggregate.homogeneous_members().ok_or_else(scratch_error)?;
                                let off = frame.slot_of(dst).ok_or_else(|| {
                                    LaminaError::ValidationError(
                                        "aggregate result has no frame slot".to_string(),
                                    )
                                })?;
                                let addr = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                emit_frame_slot_address(w, addr, off)?;
                                for index in 0..count {
                                    writeln!(
                                        w,
                                        "    str {}, [{addr}, #{}]",
                                        member_register(member, index),
                                        index * member.size_bytes()
                                    )?;
                                }
                                ra.free_scratch(addr);
                            }
                            // The result register names the slot the words land in; the
                            // body addresses it with a stack address of its own. A large
                            // one was written through x8 by the callee already.
                            Some(aggregate) if aggregate.size as u64 <= MAX_REGISTER_AGGREGATE => {
                                let off = frame.slot_of(dst).ok_or_else(|| {
                                    LaminaError::ValidationError(
                                        "aggregate result has no frame slot".to_string(),
                                    )
                                })?;
                                let addr = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                emit_frame_slot_address(w, addr, off)?;
                                writeln!(w, "    str x0, [{addr}]")?;
                                if aggregate.words() > 1 {
                                    writeln!(w, "    str x1, [{addr}, #8]")?;
                                }
                                ra.free_scratch(addr);
                            }
                            Some(_) => {}
                            None => store_result(w, dst, "x0", frame, ra)?,
                        }
                    }
                }
            }
            MirInst::TailCall { name, args } => {
                let frame_record =
                    va_area.map_or(FRAME_RECORD_BYTES, |area| area.frame_record_bytes());
                let kinds = signatures.arguments(name, args);
                let places = classify_aarch64_arguments(
                    &kinds,
                    abi.target_os(),
                    signatures.fixed_arguments(name),
                );
                for ((argument, kind), place) in args.iter().zip(&kinds).zip(&places) {
                    match *place {
                        ArgPlace::Fpr { index } => {
                            let bits = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            emit_materialize_operand(w, argument, bits, frame, ra)?;
                            if kind.ty_is_f32 {
                                writeln!(w, "    fmov s{index}, {}", w_alias(bits))?;
                            } else {
                                writeln!(w, "    fmov d{index}, {bits}")?;
                            }
                            ra.free_scratch(bits);
                        }
                        ArgPlace::Homogeneous {
                            first,
                            count,
                            member,
                        } => {
                            let address = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            emit_materialize_operand(w, argument, address, frame, ra)?;
                            for index in 0..count {
                                writeln!(
                                    w,
                                    "    ldr {}, [{address}, #{}]",
                                    member_register(member, first + index),
                                    index * member.size_bytes()
                                )?;
                            }
                            ra.free_scratch(address);
                        }
                        ArgPlace::Gpr { first, .. } => {
                            emit_materialize_operand(w, argument, &format!("x{first}"), frame, ra)?;
                        }
                        ArgPlace::Stack { offset, bytes } => {
                            let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            emit_materialize_operand(w, argument, scratch, frame, ra)?;
                            store_argument_bytes(
                                w,
                                scratch,
                                "x29",
                                frame_record as usize + offset,
                                bytes,
                            )?;
                            ra.free_scratch(scratch);
                        }
                    }
                }
                let target_sym: String = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                let unwind = os_uses_elf(abi.target_os());
                if unwind {
                    writeln!(w, "    .cfi_remember_state")?;
                }
                if frame.frame_size > 0 {
                    emit_sp_adjust(w, "add", frame.frame_size as u64)?;
                }
                writeln!(w, "    ldp x29, x30, [sp], #{frame_record}")?;
                if unwind {
                    writeln!(w, "    .cfi_def_cfa sp, 0")?;
                    writeln!(w, "    .cfi_restore 29")?;
                    writeln!(w, "    .cfi_restore 30")?;
                }
                writeln!(w, "    b {target_sym}")?;
                if unwind {
                    writeln!(w, "    .cfi_restore_state")?;
                }
            }
            MirInst::Ret { value } => {
                if let Some(v) = value
                    && ret_aggregate.is_none()
                    && let Some(ty) = ret_float
                {
                    let bits = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    emit_materialize_operand(w, v, bits, frame, ra)?;
                    if ty.size_bytes(pointer_width) == 4 {
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
                        // themselves go back in x0 and x1, or the members in v0 onward.
                        Some(aggregate) => {
                            let addr = ra.alloc_scratch().ok_or_else(scratch_error)?;
                            emit_materialize_operand(w, v, addr, frame, ra)?;
                            if let Some((member, count)) = aggregate.homogeneous_members() {
                                for index in 0..count {
                                    writeln!(
                                        w,
                                        "    ldr {}, [{addr}, #{}]",
                                        member_register(member, index),
                                        index * member.size_bytes()
                                    )?;
                                }
                                ra.free_scratch(addr);
                                writeln!(w, "    b {epilogue_label}")?;
                                continue;
                            }
                            match sret_slot {
                                Some(offset) => {
                                    let buffer = ra.alloc_scratch().ok_or_else(scratch_error)?;
                                    emit_frame_slot_load(w, buffer, offset)?;
                                    copy_through_pointer(w, addr, buffer, aggregate.words(), ra)?;
                                    writeln!(w, "    mov x0, {buffer}")?;
                                    ra.free_scratch(buffer);
                                }
                                None => {
                                    if aggregate.words() > 1 {
                                        writeln!(w, "    ldr x1, [{addr}, #8]")?;
                                    }
                                    writeln!(w, "    ldr x0, [{addr}]")?;
                                }
                            }
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
                let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
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
                let folded: Vec<_> = cases
                    .iter()
                    .map(|(case_val, _)| fold_int_cmp_immediate(*case_val as u64))
                    .collect();
                let t = ra.alloc_scratch().ok_or_else(scratch_error)?;
                let constant = if folded.iter().any(Option::is_none) {
                    Some(ra.alloc_scratch().ok_or_else(scratch_error)?)
                } else {
                    None
                };
                load_reg_to(w, value, t, frame, ra)?;
                for ((case_val, case_label), fold) in cases.iter().zip(&folded) {
                    match (fold, constant) {
                        (Some((mnemonic, operand)), _) => {
                            writeln!(w, "    {mnemonic} {}, {operand}", x_alias(t))?;
                        }
                        (None, Some(constant)) => {
                            emit_mov_imm64(w, constant, *case_val as u64)?;
                            writeln!(w, "    cmp {}, {}", x_alias(t), x_alias(constant))?;
                        }
                        (None, None) => unreachable!("a case without a fold reserved a register"),
                    }
                    emit_branch_if(
                        w,
                        "eq",
                        &format!("{block_prefix}{case_label}"),
                        long_branches,
                    )?;
                }
                writeln!(w, "    b {block_prefix}{default}")?;
                if let Some(constant) = constant {
                    ra.free_scratch(constant);
                }
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
            MirInst::VaStart { .. }
            | MirInst::VaArg { .. }
            | MirInst::VaEnd { .. }
            | MirInst::VaCopy { .. } => {
                variadic::emit(inst, w, frame, ra, abi.target_os(), va_area, block_prefix)?;
            }
            MirInst::VectorOp { .. } => simd::emit(inst, w, frame, ra)?,
            MirInst::AtomicLoad { .. }
            | MirInst::AtomicStore { .. }
            | MirInst::AtomicBinary { .. }
            | MirInst::AtomicCompareExchange { .. }
            | MirInst::Fence { .. } => atomic::emit(inst, w, frame, ra, pointer_width)?,
            MirInst::InlineAsm { .. } => {
                inline_asm::emit(inst, w, frame, ra, pointer_width)?;
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

/// The mnemonic and operand text that carry `value` inside an integer binary instruction,
/// or `None` when the constant has to be materialised into a register first.
fn fold_int_binary_immediate(op: IntBinOp, value: u64) -> Option<(&'static str, String)> {
    match op {
        IntBinOp::Add => arith_immediate(value)
            .map(|text| ("add", text))
            .or_else(|| negated_arith_immediate(value).map(|text| ("sub", text))),
        IntBinOp::Sub => arith_immediate(value)
            .map(|text| ("sub", text))
            .or_else(|| negated_arith_immediate(value).map(|text| ("add", text))),
        IntBinOp::And => logical_immediate(value).map(|text| ("and", text)),
        IntBinOp::Or => logical_immediate(value).map(|text| ("orr", text)),
        IntBinOp::Xor => logical_immediate(value).map(|text| ("eor", text)),
        _ => None,
    }
}

/// The comparison mnemonic and operand text that carry `value`, or `None` when it needs a
/// register. A negative constant compares through `cmn`, which sets the same flags.
fn fold_int_cmp_immediate(value: u64) -> Option<(&'static str, String)> {
    arith_immediate(value)
        .map(|text| ("cmp", text))
        .or_else(|| negated_arith_immediate(value).map(|text| ("cmn", text)))
}

/// The physical register a source operand already occupies, when reading it needs no
/// load. Only a virtual register with a global mapping qualifies; an immediate, a frame
/// slot or a physical operand still goes through scratch.
fn operand_in_register(op: &Operand, ra: &A64RegAlloc) -> Option<&'static str> {
    match op {
        Operand::Register(Register::Virtual(v)) => ra.get_mapping_for(v),
        _ => None,
    }
}

/// The physical register a destination already occupies, so a lowering can write it in
/// place instead of computing into scratch and copying back.
fn dest_in_register(dst: &Register, ra: &A64RegAlloc) -> Option<&'static str> {
    match dst {
        Register::Virtual(v) => ra.get_mapping_for(v),
        _ => None,
    }
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
                    let mut addr = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    if addr == dest {
                        if let Some(other) = ra.alloc_scratch() {
                            ra.free_scratch(addr);
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
                    let mut addr = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    if addr == src_reg {
                        if let Some(other) = ra.alloc_scratch() {
                            ra.free_scratch(addr);
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
    let inverse = invert_condition(cond).ok_or_else(|| unknown_condition(cond))?;
    writeln!(w, "    b.{inverse} 9f")?;
    writeln!(w, "    b {target}")?;
    writeln!(w, "9:")?;
    Ok(())
}

fn emit_probed_sp_sub<W: Write>(w: &mut W, amount: u64, page_size: u64) -> Result<(), LaminaError> {
    emit_mov_imm64(w, "x16", amount)?;
    writeln!(w, "    sub x16, sp, x16")?;
    let remainder = amount % page_size;
    if remainder != 0 {
        emit_sp_adjust(w, "sub", remainder)?;
        writeln!(w, "    str xzr, [sp]")?;
    }
    emit_mov_imm64(w, "x17", page_size)?;
    writeln!(w, "1:")?;
    writeln!(w, "    sub sp, sp, x17")?;
    writeln!(w, "    str xzr, [sp]")?;
    writeln!(w, "    cmp sp, x16")?;
    writeln!(w, "    b.hi 1b")?;
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

/// Stores a register into a frame slot, going through a scratch when the offset is
/// past the reach of `stur`.
/// Saves or restores the callee-saved registers a global plan handed out, in the area
/// reserved for them at `base` from the frame pointer. Adjacent pairs go through
/// `stp`/`ldp` pairs, anchored on x29 while the offsets stay in their reach and on a
/// scratch copy of the area address otherwise.
fn emit_callee_saved<W: Write>(
    w: &mut W,
    regs: &[&'static str],
    base: i32,
    save: bool,
) -> Result<(), LaminaError> {
    let pair = if save { "stp" } else { "ldp" };
    let last = base + (regs.len() as i32 - 1) * 8;
    let mut index = 0;
    if (-512..=504).contains(&base) && (-512..=504).contains(&last) {
        while index + 1 < regs.len() {
            let offset = base + index as i32 * 8;
            writeln!(
                w,
                "    {pair} {}, {}, [x29, #{offset}]",
                regs[index],
                regs[index + 1]
            )?;
            index += 2;
        }
        if index < regs.len() {
            let offset = base + index as i32 * 8;
            if save {
                emit_frame_slot_store(w, regs[index], offset)?;
            } else {
                emit_frame_slot_load(w, regs[index], offset)?;
            }
        }
        return Ok(());
    }

    emit_frame_slot_address(w, "x16", base)?;
    while index + 1 < regs.len() {
        writeln!(
            w,
            "    {pair} {}, {}, [x16, #{}]",
            regs[index],
            regs[index + 1],
            index * 8
        )?;
        index += 2;
    }
    if index < regs.len() {
        let single = if save { "str" } else { "ldr" };
        writeln!(w, "    {single} {}, [x16, #{}]", regs[index], index * 8)?;
    }
    Ok(())
}

fn emit_frame_slot_store<W: Write>(
    w: &mut W,
    source: &str,
    offset: i32,
) -> Result<(), LaminaError> {
    if (-256..=255).contains(&offset) {
        writeln!(w, "    stur {source}, [x29, #{offset}]")?;
    } else {
        emit_frame_slot_address(w, "x16", offset)?;
        writeln!(w, "    str {source}, [x16]")?;
    }
    Ok(())
}

/// Loads a frame slot into a register, going through a scratch when the offset is
/// past the reach of `ldur`.
fn emit_frame_slot_load<W: Write>(w: &mut W, dest: &str, offset: i32) -> Result<(), LaminaError> {
    if (-256..=255).contains(&offset) {
        writeln!(w, "    ldur {dest}, [x29, #{offset}]")?;
    } else {
        emit_frame_slot_address(w, dest, offset)?;
        writeln!(w, "    ldr {dest}, [{dest}]")?;
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
                    let oreg = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    if oreg == dest {
                        // Need a different register
                        if let Some(other) = ra.alloc_scratch() {
                            let temp = other;
                            emit_mov_imm64(w, temp, off_val as u64)?;
                            writeln!(w, "    add {dest}, {dest}, {temp}")?;
                            ra.free_scratch(temp);
                        } else {
                            return Err(scratch_error().into());
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
            let mut idx = ra.alloc_scratch().ok_or_else(scratch_error)?;
            if idx == dest {
                if let Some(other) = ra.alloc_scratch() {
                    idx = other;
                } else {
                    return Err(scratch_error().into());
                }
            }
            load_reg_to(w, index, idx, frame, ra)?; // idx = index
            let sh = index_scale_shift(*scale).ok_or_else(|| {
                LaminaError::ValidationError(format!("address scale {scale} is not 1, 2, 4 or 8"))
            })?;
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
                    let oreg = ra.alloc_scratch().ok_or_else(scratch_error)?;
                    if oreg == dest {
                        return Err(scratch_error().into());
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

/// Whether `madd` has its register inputs in hand. Without an allocation plan it would
/// need a scratch for each of its four registers and more to reach far slots.
fn multiply_add_operands_ready(block: &Block, position: usize, ra: &A64RegAlloc) -> bool {
    let inputs = block.instructions[position..=position + 1]
        .iter()
        .flat_map(|inst| inst.use_regs());
    let product = block.instructions[position].def_reg();
    inputs
        .filter(|register| Some(*register) != product)
        .all(|register| match register {
            Register::Virtual(v) => ra.get_mapping_for(v).is_some(),
            Register::Physical(_) => false,
        })
}

/// A whole-word multiply whose only use is the add right after it, by block and position.
/// `madd` does both, and the product never needs a register of its own.
fn multiply_adds(func: &Function, pointer_width: PointerWidth) -> HashSet<(usize, usize)> {
    let mut reads: HashMap<&Register, usize> = HashMap::new();
    let mut writes: HashMap<&Register, usize> = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        for register in inst.use_regs() {
            *reads.entry(register).or_default() += 1;
        }
        if let Some(register) = inst.def_reg() {
            *writes.entry(register).or_default() += 1;
        }
    }
    let mut found = HashSet::new();
    for (index, block) in func.blocks.iter().enumerate() {
        for (position, pair) in block.instructions.windows(2).enumerate() {
            let (
                MirInst::IntBinary {
                    op: IntBinOp::Mul,
                    ty,
                    dst: product @ Register::Virtual(_),
                    lhs: Operand::Register(_),
                    rhs: Operand::Register(_),
                },
                MirInst::IntBinary {
                    op: IntBinOp::Add,
                    ty: add_ty,
                    lhs,
                    rhs,
                    ..
                },
            ) = (&pair[0].instr, &pair[1].instr)
            else {
                continue;
            };
            let is_product =
                |operand: &Operand| matches!(operand, Operand::Register(r) if r == product);
            if ty == add_ty
                && ty.size_bytes(pointer_width) == 8
                && is_product(lhs) != is_product(rhs)
                && reads.get(product) == Some(&1)
                && writes.get(product) == Some(&1)
            {
                found.insert((index, position));
            }
        }
    }
    found
}

/// Whether this division has a reciprocal: not by -1, 0 or 1, which other paths handle,
/// and not by the values the constants cannot reach.
fn has_reciprocal(op: IntBinOp, divisor: i64) -> bool {
    match op {
        IntBinOp::SDiv | IntBinOp::SRem => !matches!(divisor, -1..=1 | i64::MIN),
        _ => divisor >= 2,
    }
}

/// The bracketed address a load or store of `size` bytes takes, using registers the
/// operands already hold. Returns the scratch registers the caller frees after the access.
fn emit_access_address<W: Write>(
    w: &mut W,
    addr: &AddressMode,
    size: usize,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(String, Vec<&'static str>), LaminaError> {
    let mut scratches = Vec::new();
    let mut in_register =
        |w: &mut W, register: &Register, ra: &mut A64RegAlloc| -> Result<String, LaminaError> {
            if let Register::Virtual(v) = register
                && let Some(preg) = ra.get_mapping_for(v)
            {
                return Ok(x_alias(preg));
            }
            let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
            scratches.push(scratch);
            load_reg_to(w, register, scratch, frame, ra)?;
            Ok(x_alias(scratch))
        };
    let place = match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            let offset = i64::from(*offset);
            let unit = size.max(1) as i64;
            let encodable = (-256..=255).contains(&offset)
                || (offset > 0 && offset % unit == 0 && offset / unit <= 4095);
            if !encodable {
                let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
                materialize_address(w, addr, scratch, frame, ra)?;
                return Ok((format!("[{scratch}]"), vec![scratch]));
            }
            let base = in_register(w, base, ra)?;
            if offset == 0 {
                format!("[{base}]")
            } else {
                format!("[{base}, #{offset}]")
            }
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } if *offset == 0 && (*scale == 1 || usize::from(*scale) == size) => {
            let base = in_register(w, base, ra)?;
            let index = in_register(w, index, ra)?;
            match index_scale_shift(*scale) {
                Some(shift) if shift > 0 => format!("[{base}, {index}, lsl #{shift}]"),
                _ => format!("[{base}, {index}]"),
            }
        }
        AddressMode::BaseIndexScale { .. } => {
            let scratch = ra.alloc_scratch().ok_or_else(scratch_error)?;
            materialize_address(w, addr, scratch, frame, ra)?;
            return Ok((format!("[{scratch}]"), vec![scratch]));
        }
    };
    Ok((place, scratches))
}

/// Trait-backed MIR ⇒ AArch64 code generator.
pub struct AArch64Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> AArch64Codegen<'a> {
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
        generate_mir_aarch64_with_units_and_settings(
            module,
            writer,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
            self.base.pointer_width,
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
            .with(CodegenCapability::AggregateArguments)
            .with(CodegenCapability::IndirectCalls)
            .with(CodegenCapability::GlobalRegisterAllocation)
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
        let pointer_width = self.base.pointer_width;
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_aarch64_with_units(
                    module,
                    writer,
                    target_os,
                    codegen_units,
                    pointer_width,
                )
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
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let s_s = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let s_d = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, src, s_s, frame, ra)?;
    extend_integer(w, s_s, from, signed, pointer_width)?;
    let cvt = if signed { "scvtf" } else { "ucvtf" };
    if to.size_bytes(pointer_width) == 4 {
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

fn emit_float_convert<W: Write>(
    w: &mut W,
    to: &MirType,
    dst: &Register,
    src: &Operand,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let s_s = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, src, s_s, frame, ra)?;
    if to.size_bytes(pointer_width) == 4 {
        writeln!(w, "    fmov d0, {s_s}")?;
        writeln!(w, "    fcvt s0, d0")?;
        writeln!(w, "    fmov {}, s0", w_alias(s_s))?;
    } else {
        writeln!(w, "    fmov s0, {}", w_alias(s_s))?;
        writeln!(w, "    fcvt d0, s0")?;
        writeln!(w, "    fmov {s_s}, d0")?;
    }
    store_result(w, dst, &x_alias(s_s), frame, ra)?;
    ra.free_scratch(s_s);
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
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let s_s = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let s_d = ra.alloc_scratch().ok_or_else(scratch_error)?;
    let s_t = ra.alloc_scratch().ok_or_else(scratch_error)?;
    emit_materialize_operand(w, src, s_s, frame, ra)?;
    if from.size_bytes(pointer_width) == 4 {
        writeln!(w, "    fmov s0, {}", w_alias(s_s))?;
    } else {
        writeln!(w, "    fmov d0, {s_s}")?;
    }
    let bits = to.size_bytes(pointer_width) * 8;
    let d32 = w_alias(s_d);
    let t32 = w_alias(s_t);
    if bits == 64 {
        let cvt = if signed { "fcvtzs" } else { "fcvtzu" };
        let from_suffix = if from.size_bytes(pointer_width) == 4 {
            "s0"
        } else {
            "d0"
        };
        writeln!(w, "    {cvt} {s_d}, {from_suffix}")?;
    } else {
        let cvt = if signed { "fcvtzs" } else { "fcvtzu" };
        let from_suffix = if from.size_bytes(pointer_width) == 4 {
            "s0"
        } else {
            "d0"
        };
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
    extend_integer(w, s_d, to, signed, pointer_width)?;
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
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    if !matches!(
        ty,
        MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I32)
    ) {
        return Ok(());
    }
    let bits = ty.size_bytes(pointer_width) * 8;
    let instruction = if signed { "sbfx" } else { "ubfx" };
    writeln!(w, "    {instruction} {register}, {register}, #0, #{bits}")?;
    Ok(())
}

#[cfg(test)]
mod stack_probe_tests;
#[cfg(test)]
mod tests;
