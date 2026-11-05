//! armv7 code generation for MIR, in ARM (A32) state.
//!
//! A32 rather than Thumb-2: every encoding is one word wide, every data-processing
//! instruction carries a condition so a boolean needs no IT block, and `b` reaches far
//! enough that no long-branch pass is needed. Returns still go through `bx lr` and
//! calls through `bl`/`blx`, so the linker can interwork with a Thumb C library.
//!
//! The calling convention is AAPCS in its VFP hard-float variant: see `abi`.

pub(crate) mod abi;
mod call;
mod emit;
mod inline_asm;
mod operand;
#[cfg(test)]
mod tests;
mod variadic;

use crate::mir_codegen::common::{reject_thread_local_globals, reject_underaligned_accesses};
use std::{collections::HashMap, io::Write, sync::Arc};

use crate::{
    error::LaminaError,
    mir::{Function, Global, Instruction as MirInst, MirType, Module as MirModule, Signature},
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, MirCodegenSettings,
        capability::{CapabilitySet, CodegenCapability},
        common::{
            CodegenBase, FunctionEmissionPlan, compile_functions_parallel, emit_debug_file,
            emit_global_data, emit_global_directives, emit_symbol_visibility, external_names,
            lamina_to_codegen_error, parallel_codegen_error,
        },
        validate_module_call_parameters,
    },
};
use lamina_codegen::{aarch64::FrameMap, abi::Abi};
use lamina_platform::{PointerWidth, StackProbe, TargetArchitecture, TargetOperatingSystem};

use abi::{ArgKind, Armv7Abi, Signatures, classify, returns_in_memory};
use emit::{Ctx, copy_through_pointer, is_wide_integer, reject_wide};
use operand::{
    ENTRY_SP, FP, SAVED_BYTES, Scratch, extend_integer, frame_address, frame_store, frame_vstore,
    is_modified_immediate, mov_imm32, sp_adjust, unsupported,
};

const SAVED_CORE: &str = "{r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}";
const SAVED_DOUBLES: &str = "{d8, d9, d10, d11, d12, d13, d14, d15}";

pub(crate) fn emit_restore<W: Write>(w: &mut W, variadic: bool) -> Result<(), LaminaError> {
    writeln!(w, "    mov sp, {ENTRY_SP}")?;
    writeln!(w, "    vpop {SAVED_DOUBLES}")?;
    writeln!(w, "    pop {SAVED_CORE}")?;
    if variadic {
        writeln!(w, "    add sp, sp, #{}", variadic::SAVE_AREA_BYTES)?;
    }
    Ok(())
}

/// Reports the first i64 the function asks the backend to hold in a register, naming
/// the operation. Eight-byte moves that only marshal an argument are not reported.
fn reject_wide_integers(name: &str, func: &Function) -> Result<(), LaminaError> {
    let wide = |ty: &MirType| is_wide_integer(ty);
    if func.sig.ret_ty.as_ref().is_some_and(wide) && func.sig.ret_aggregate.is_none() {
        return Err(reject_wide(&format!("the return type of @{name}")));
    }
    for param in &func.sig.params {
        if param.aggregate.is_none() && wide(&param.ty) {
            return Err(reject_wide(&format!("a parameter of @{name}")));
        }
    }
    for block in &func.blocks {
        for inst in &block.instructions {
            let reported = match &inst.instr {
                MirInst::IntBinary { ty, .. } if wide(ty) => Some("an integer arithmetic operand"),
                MirInst::IntCmp { ty, .. } if wide(ty) => Some("an integer comparison"),
                MirInst::IntUnary { ty, .. } if wide(ty) => Some("a bit-counting operation"),
                MirInst::IntRotate { ty, .. } if wide(ty) => Some("a rotate"),
                MirInst::Load { ty, .. } if wide(ty) => Some("a load"),
                MirInst::Store { ty, .. } if wide(ty) => Some("a store"),
                MirInst::Copy { ty, .. } if wide(ty) => Some("a copy"),
                MirInst::Select { ty, .. } if wide(ty) => Some("a select"),
                MirInst::Bitcast { to, .. } if wide(to) => Some("a bitcast"),
                _ => None,
            };
            if let Some(what) = reported {
                return Err(reject_wide(&format!("{what} in @{name}")));
            }
        }
    }
    Ok(())
}

/// Strongest alignment the frame has to stand on, which is also the alignment the
/// frame size is rounded to so the frame pointer keeps it.
fn frame_alignment(func: &Function) -> usize {
    let at_call = TargetArchitecture::Armv7.stack_alignment_at_call();
    func.stack_alignments
        .values()
        .copied()
        .max()
        .unwrap_or(at_call)
        .max(at_call)
        .next_power_of_two()
}

/// `dest = entry stack pointer + offset`, for reaching the caller's stack arguments.
fn entry_sp_address<W: Write>(w: &mut W, dest: &str, offset: i32) -> Result<(), LaminaError> {
    let magnitude = offset.unsigned_abs();
    if is_modified_immediate(magnitude) {
        let op = if offset < 0 { "sub" } else { "add" };
        writeln!(w, "    {op} {dest}, {ENTRY_SP}, #{magnitude}")?;
    } else {
        mov_imm32(w, dest, offset as u32)?;
        writeln!(w, "    add {dest}, {ENTRY_SP}, {dest}")?;
    }
    Ok(())
}

/// Spills one incoming parameter into its frame slot.
fn spill_parameter<W: Write>(
    w: &mut W,
    ty: MirType,
    kind: &ArgKind,
    place: &abi::ArgPlace,
    offset: i32,
    saved: i32,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    if let Some(index) = place.vfp {
        let register = if kind.double {
            format!("d{}", index / 2)
        } else {
            format!("s{index}")
        };
        return frame_vstore(w, &register, offset);
    }

    let mut written = 0usize;
    if let Some(core) = place.core {
        if kind.aggregate.is_some() {
            let address = scratch.alloc()?;
            frame_address(w, address, offset)?;
            for word in 0..core.words {
                writeln!(
                    w,
                    "    str r{}, [{address}, #{}]",
                    core.first + word,
                    word * 4
                )?;
            }
            scratch.free(address);
            written = core.words * 4;
        } else if kind.double {
            writeln!(
                w,
                "    vmov {}, r{}, r{}",
                operand::F64_SCRATCH[0],
                core.first,
                core.first + 1
            )?;
            frame_vstore(w, operand::F64_SCRATCH[0], offset)?;
            return Ok(());
        } else {
            let register = format!("r{}", core.first);
            extend_integer(w, &register, &ty, true)?;
            frame_store(w, &register, offset)?;
            return Ok(());
        }
    }

    let Some(span) = place.stack else {
        return Ok(());
    };
    let at = saved + span.offset as i32;
    let source = scratch.alloc()?;
    entry_sp_address(w, source, at)?;
    if let Some(aggregate) = kind.aggregate {
        let target = scratch.alloc()?;
        frame_address(w, target, offset + written as i32)?;
        copy_through_pointer(w, source, target, aggregate.size - written)?;
        scratch.free(target);
    } else if kind.double {
        writeln!(w, "    vldr {}, [{source}]", operand::F64_SCRATCH[0])?;
        frame_vstore(w, operand::F64_SCRATCH[0], offset)?;
    } else if kind.float {
        writeln!(w, "    vldr {}, [{source}]", operand::F32_SCRATCH[0])?;
        frame_vstore(w, operand::F32_SCRATCH[0], offset)?;
    } else {
        let value = scratch.alloc()?;
        writeln!(w, "    ldr {value}, [{source}]")?;
        extend_integer(w, value, &ty, true)?;
        frame_store(w, value, offset)?;
        scratch.free(value);
    }
    scratch.free(source);
    Ok(())
}

fn compile_single_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, CodegenError> {
    compile_function(func_name, func, target_os, settings, signatures)
        .map_err(lamina_to_codegen_error)
}

/// Lowers sp one page at a time and touches each page, so a frame larger than the
/// guard page cannot step over it. The prologue has already pushed r4, so it holds
/// the final sp; what the stores write does not matter.
fn probed_sp_sub<W: Write>(w: &mut W, amount: u32, page_size: u32) -> Result<(), LaminaError> {
    mov_imm32(w, "r4", amount)?;
    writeln!(w, "    sub r4, sp, r4")?;
    let remainder = amount % page_size;
    if remainder != 0 {
        sp_adjust(w, "sub", remainder as usize)?;
        writeln!(w, "    str r4, [sp]")?;
    }
    writeln!(w, "1:")?;
    writeln!(w, "    sub sp, sp, #{page_size}")?;
    writeln!(w, "    str r4, [sp]")?;
    writeln!(w, "    cmp sp, r4")?;
    writeln!(w, "    bhi 1b")?;
    Ok(())
}

fn compile_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, LaminaError> {
    reject_wide_integers(func_name, func)?;
    variadic::reject_co_processor_boundary(func_name, func)?;
    let armv7 = Armv7Abi::new(target_os);
    let label = armv7.mangle_function_name(func_name);
    let mut w: Vec<u8> = Vec::new();
    let mut scratch = Scratch::default();

    let saved = variadic::saved_bytes(func);
    writeln!(w, "{label}:")?;
    writeln!(w, "    .cfi_startproc")?;
    if func.sig.variadic {
        variadic::save_core_registers(&mut w)?;
    }
    writeln!(w, "    push {SAVED_CORE}")?;
    writeln!(w, "    .cfi_def_cfa_offset {}", saved - SAVED_BYTES + 40)?;
    for (index, number) in [4, 5, 6, 7, 8, 9, 10, 11, 12, 14].into_iter().enumerate() {
        writeln!(
            w,
            "    .cfi_offset {number}, {}",
            index as i32 * 4 - (saved - SAVED_BYTES + 40)
        )?;
    }
    writeln!(w, "    vpush {SAVED_DOUBLES}")?;
    writeln!(w, "    .cfi_def_cfa_offset {saved}")?;
    writeln!(w, "    mov {ENTRY_SP}, sp")?;
    writeln!(w, "    .cfi_def_cfa_register 10")?;

    let mut frame = FrameMap::from_function(func);
    let memory_return = returns_in_memory(func.sig.ret_aggregate);
    let sret_slot = memory_return.then(|| -frame.frame_size - 8);
    if sret_slot.is_some() {
        frame.frame_size += 16;
    }
    // The frame pointer carries the strongest alignment any slot asked for, which the
    // prologue's fixed pushes do not leave the stack pointer on.
    let alignment = frame_alignment(func);
    frame.frame_size =
        (frame.frame_size.max(0) as usize).div_ceil(alignment) as i32 * alignment as i32;
    let page_size = TargetArchitecture::Armv7.stack_page_size();
    match target_os.stack_probe() {
        Some(StackProbe::PageLoop) if frame.frame_size.max(0) as usize > page_size => {
            probed_sp_sub(&mut w, frame.frame_size as u32, page_size as u32)?
        }
        _ => sp_adjust(&mut w, "sub", frame.frame_size.max(0) as usize)?,
    }
    writeln!(w, "    bic sp, sp, #{}", alignment - 1)?;
    let size = frame.frame_size.max(0) as u32;
    if is_modified_immediate(size) {
        writeln!(w, "    add {FP}, sp, #{size}")?;
    } else {
        mov_imm32(&mut w, FP, size)?;
        writeln!(w, "    add {FP}, sp, {FP}")?;
    }
    if let Some(offset) = sret_slot {
        frame_store(&mut w, "r0", offset)?;
    }

    let mut kinds: Vec<ArgKind> = func
        .sig
        .params
        .iter()
        .map(|p| ArgKind::of(p.ty, p.aggregate))
        .collect();
    if memory_return {
        kinds.insert(0, ArgKind::pointer());
    }
    let places = classify(&kinds, None, false);
    let skip = usize::from(memory_return);
    for ((param, kind), place) in func
        .sig
        .params
        .iter()
        .zip(kinds.iter().skip(skip))
        .zip(places.iter().skip(skip))
    {
        if let Some(offset) = frame.slot_of(&param.reg) {
            spill_parameter(&mut w, param.ty, kind, place, offset, saved, &mut scratch)?;
        }
    }

    let epilogue = format!(".Lret_{label}");
    let block_prefix = format!(".L_{label}.");
    let ctx = Ctx {
        func,
        frame: &frame,
        abi: &armv7,
        signatures,
        epilogue: &epilogue,
        block_prefix: &block_prefix,
        ret_aggregate: func.sig.ret_aggregate,
        ret_float: func.sig.ret_ty.filter(|ty| ty.is_float()),
        sret_slot,
        settings,
        va_base: func
            .sig
            .variadic
            .then(|| variadic::area_base(func))
            .transpose()?,
    };

    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&index| func.blocks[index].label != func.entry)
    {
        writeln!(w, "    b {block_prefix}{}", func.entry)?;
    }
    for (slot, &index) in plan.order.iter().enumerate() {
        let block = &func.blocks[index];
        let next = plan
            .order
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        writeln!(w, "{block_prefix}{}:", block.label)?;
        for inst in &block.instructions {
            if matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
                continue;
            }
            emit::emit_instruction(&mut w, &inst.instr, &ctx, &mut scratch, inst.loc)?;
        }
    }

    writeln!(w, "{epilogue}:")?;
    writeln!(w, "    .cfi_remember_state")?;
    emit_restore(&mut w, func.sig.variadic)?;
    writeln!(w, "    .cfi_def_cfa 13, 0")?;
    writeln!(w, "    bx lr")?;
    writeln!(w, "    .cfi_restore_state")?;
    writeln!(w, "    .cfi_endproc")?;
    Ok(w)
}

/// `@` starts a comment in ARM assembly, so an ELF section type is spelled with `%`.
pub(crate) fn arm_section_types(assembly: Vec<u8>) -> Vec<u8> {
    let needle = b",\"ax\",@progbits";
    let mut out = assembly;
    let mut at = 0;
    while let Some(found) = out[at..]
        .windows(needle.len())
        .position(|window| window == needle)
    {
        let sigil = at + found + needle.len() - "@progbits".len();
        out[sigil] = b'%';
        at = sigil;
    }
    out
}

/// A function named like a register would come out as `bl r0`, which the assembler
/// reads as an instruction rather than a symbol.
fn reject_register_symbols(module: &MirModule) -> Result<(), LaminaError> {
    let mut names: Vec<&str> = module.functions.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        if is_armv7_register_name(name) {
            return Err(unsupported(format!(
                "@{name} is an armv7 register name, so the assembler cannot use it as a branch \
                 target; rename the function"
            )));
        }
    }
    Ok(())
}

fn is_armv7_register_name(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "sp" | "lr" | "pc" | "fp" | "ip" | "sl" | "sb"
    ) {
        return true;
    }
    let mut chars = lower.chars();
    let (Some(prefix), digits) = (chars.next(), chars.as_str()) else {
        return false;
    };
    let last = match prefix {
        'r' | 's' => 31,
        'd' => 15,
        'q' => 15,
        _ => return false,
    };
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && digits.parse::<u8>().is_ok_and(|n| n <= last)
}

pub fn generate_mir_armv7_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    reject_thread_local_globals(module, "armv7")?;
    reject_underaligned_accesses(module, PointerWidth::Bits32, "armv7")?;
    if !matches!(
        target_os,
        TargetOperatingSystem::Linux | TargetOperatingSystem::Unknown
    ) {
        return Err(unsupported(format!(
            "armv7 code generation targets ELF; {target_os:?} is not supported"
        )));
    }
    let variant = settings.arch_variant.armv7().unwrap_or_default();
    let environment = settings.environment(TargetArchitecture::Armv7, target_os);
    if !environment.hardware_float() {
        return Err(unsupported(format!(
            "armv7 code generation needs a VFP unit; the {environment} environment passes \
             floats in integer registers and this backend does not lower to soft-float \
             libcalls"
        )));
    }
    if !variant.hardware_divide {
        return Err(unsupported(format!(
            "armv7 code generation emits sdiv and udiv; {} without the idiv extension \
             needs a division libcall this backend does not emit",
            variant.profile.arch_name()
        )));
    }
    validate_module_call_parameters(module, TargetArchitecture::Armv7)?;
    reject_register_symbols(module)?;

    let armv7 = Armv7Abi::new(target_os);
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(Signatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function(name, func, os, settings_arc.as_ref(), signatures.as_ref())
        })
        .map_err(parallel_codegen_error)?;

    writeln!(writer, ".syntax unified")?;
    writeln!(writer, ".arch {}", variant.profile.arch_name())?;
    if variant.hardware_divide {
        writeln!(writer, ".arch_extension idiv")?;
    }
    if let Some(fpu) = variant.fpu.filter(|_| environment.hardware_float()) {
        writeln!(writer, ".fpu {fpu}")?;
    }
    writeln!(writer, ".eabi_attribute 28, 1")?;
    writeln!(writer, ".arm")?;
    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings, module)?;

    for func_name in external_names(module) {
        writeln!(writer, ".extern {func_name}")?;
    }
    emit_global_directives(writer, module, |name| armv7.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        armv7.mangle_function_name(name)
    })?;

    for result in results {
        writer.write_all(&arm_section_types(result.assembly))?;
    }
    emit_global_data(writer, module, target_os, |name| {
        armv7.mangle_function_name(name)
    })?;
    // `@` starts a comment in ARM assembly, so the note's type is spelled with `%`.
    writeln!(writer, ".section .note.GNU-stack,\"\",%progbits")?;
    Ok(())
}

pub fn generate_mir_armv7<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_armv7_with_units_and_settings(
        module,
        writer,
        target_os,
        1,
        &MirCodegenSettings::default(),
    )
}

pub fn capabilities() -> CapabilitySet {
    [
        CodegenCapability::IntegerArithmetic,
        CodegenCapability::FloatingPointArithmetic,
        CodegenCapability::ControlFlow,
        CodegenCapability::FunctionCalls,
        CodegenCapability::Recursion,
        CodegenCapability::StackAllocation,
        CodegenCapability::MemoryOperations,
        CodegenCapability::SystemCalls,
        CodegenCapability::InlineAssembly,
        CodegenCapability::ForeignFunctionInterface,
        CodegenCapability::AggregateArguments,
        CodegenCapability::IndirectCalls,
    ]
    .into_iter()
    .collect()
}

/// Trait-backed MIR to armv7 code generator.
pub struct Armv7Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> Armv7Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os, PointerWidth::Bits32),
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
        generate_mir_armv7_with_units_and_settings(
            module,
            writer,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for Armv7Codegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 32;

    fn capabilities() -> CapabilitySet {
        capabilities()
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
        let units = self.base.codegen_units;
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_armv7_with_units_and_settings(
                    module,
                    writer,
                    target_os,
                    codegen_units,
                    &MirCodegenSettings::default(),
                )
            },
            "armv7",
            units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "binary emission is not implemented for the armv7 MIR backend".into(),
        ))
    }
}
