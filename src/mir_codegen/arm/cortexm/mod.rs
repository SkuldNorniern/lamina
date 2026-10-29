//! ARM Cortex-M code generation for MIR: ARMv6-M and ARMv7-M, Thumb only.
//!
//! There is no ARM state on a Cortex-M, so the choice the armv7 backend made — A32, in
//! which every data-processing instruction is conditional and a boolean needs no `IT`
//! block — is not available. This backend faces that directly, and in the narrowest way
//! it can: the only conditional data-processing instructions it emits come from
//! [`isa::condition_set`] and [`isa::condition_move`], which on ARMv7-M write the `IT`
//! and the single instruction it guards together from one condition, so the block is
//! one instruction long and its pattern always matches. On ARMv6-M, which has no `IT`
//! at all, the same two helpers branch over one instruction instead. No other `IT` is
//! emitted anywhere, and no `IT` block ever holds more than one instruction.
//!
//! The frame is addressed from the stack pointer, which the prologue sets once and
//! nothing moves afterwards: the widest outgoing argument area any call needs is
//! reserved there too. That is what makes ARMv6-M workable, where a load against a
//! general register reaches 124 bytes but one against the stack pointer reaches 1020.
//!
//! Floating point is refused by name on both: the M0 has no FPU, the single-precision
//! VFP of an M4F or M7 is not used here, and no soft-float library is shipped. Integer
//! division is refused by name on ARMv6-M, which has no divide instruction, rather than
//! calling an `__aeabi_idiv` that nothing here provides.

mod abi;
mod call;
mod emit;
mod isa;
mod operand;
#[cfg(test)]
mod tests;

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
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};

use crate::mir_codegen::arm::armv7::abi::{
    ArgKind, ArgPlace, Signatures, classify, returns_in_memory,
};
use crate::mir_codegen::arm::armv7::arm_section_types;
use abi::CortexMAbi;
use emit::{Ctx, is_wide_integer, reject_float, reject_wide};
use isa::Isa;
use operand::{
    Access, Layout, SAVED_BYTES, SAVED_LIST, Scratch, check_frame_alignment, copy_through_pointer,
    extend_integer, unsupported,
};

fn isa_of(arch: TargetArchitecture) -> Option<Isa> {
    match arch {
        TargetArchitecture::Thumbv6m => Some(Isa::Thumb1),
        TargetArchitecture::Thumbv7m => Some(Isa::Thumb2),
        _ => None,
    }
}

/// Reports the first i64 or float the function asks the backend to hold in a register,
/// naming the operation.
fn reject_unlowered_values(isa: Isa, name: &str, func: &Function) -> Result<(), LaminaError> {
    let wide = |ty: &MirType| is_wide_integer(ty);
    if let Some(ty) = func.sig.ret_ty.filter(|ty| ty.is_float()) {
        return Err(reject_float(
            isa,
            &format!("the return type of @{name}"),
            &ty,
        ));
    }
    if func.sig.ret_ty.as_ref().is_some_and(wide) && func.sig.ret_aggregate.is_none() {
        return Err(reject_wide(isa, &format!("the return type of @{name}")));
    }
    for param in &func.sig.params {
        if param.aggregate.is_some() {
            continue;
        }
        if param.ty.is_float() {
            return Err(reject_float(
                isa,
                &format!("a parameter of @{name}"),
                &param.ty,
            ));
        }
        if wide(&param.ty) {
            return Err(reject_wide(isa, &format!("a parameter of @{name}")));
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
                return Err(reject_wide(isa, &format!("{what} in @{name}")));
            }
        }
    }
    Ok(())
}

/// Spills one incoming parameter into its frame slot.
fn spill_parameter<W: Write>(
    w: &mut W,
    ty: MirType,
    kind: &ArgKind,
    place: &ArgPlace,
    offset: i32,
    access: &Access<'_>,
    scratch: &mut Scratch,
) -> Result<(), LaminaError> {
    let mut written = 0usize;
    if let Some(core) = place.core {
        if let Some(aggregate) = kind.aggregate {
            let address = scratch.alloc()?;
            access.frame_address(w, address, offset)?;
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
            let _ = aggregate;
        } else {
            let register = format!("r{}", core.first);
            extend_integer(w, &register, &ty, true)?;
            operand::frame_store(w, access.isa, &register, offset, scratch)?;
            return Ok(());
        }
    }

    let Some(span) = place.stack else {
        return Ok(());
    };
    let at = access.layout.incoming() + span.offset as i32;
    let source = scratch.alloc()?;
    access.frame_address(w, source, at)?;
    if let Some(aggregate) = kind.aggregate {
        let target = scratch.alloc()?;
        access.frame_address(w, target, offset + written as i32)?;
        copy_through_pointer(
            w,
            access.isa,
            source,
            target,
            aggregate.size - written,
            scratch,
        )?;
        scratch.free(target);
    } else {
        let value = scratch.alloc()?;
        writeln!(w, "    ldr {value}, [{source}]")?;
        extend_integer(w, value, &ty, true)?;
        operand::frame_store(w, access.isa, value, offset, scratch)?;
        scratch.free(value);
    }
    scratch.free(source);
    Ok(())
}

/// Adds or subtracts a constant from the stack pointer.
fn sp_adjust<W: Write>(w: &mut W, isa: Isa, op: &str, bytes: i32) -> Result<(), LaminaError> {
    if bytes == 0 {
        return Ok(());
    }
    let limit = if isa == Isa::Thumb1 { 508 } else { 4095 };
    if bytes <= limit {
        writeln!(w, "    {op} sp, #{bytes}")?;
        return Ok(());
    }
    // `r4` is not live: the prologue has not filled a slot yet, and the epilogue has
    // already produced the result in r0/r1.
    isa::mov_imm(w, isa, "r4", bytes as u32)?;
    if op == "sub" {
        writeln!(w, "    negs r4, r4")?;
    }
    writeln!(w, "    add sp, r4")?;
    Ok(())
}

/// Bytes a Thumb-1 `ldr rd, =symbol` reaches forward to its literal, less a margin for
/// the instruction that jumps over the pool.
const POOL_REACH: usize = 900;

/// Tracks how far the assembler would have to look forward for a pending Thumb-1
/// literal, and drops a pool before that distance runs out. The pool is jumped over,
/// so it can sit in the middle of a block.
struct LiteralPool {
    prefix: String,
    pending: bool,
    bytes: usize,
    index: usize,
}

impl LiteralPool {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            pending: false,
            bytes: 0,
            index: 0,
        }
    }

    /// Every Thumb-1 instruction is two bytes except `bl` and a wide branch, which are
    /// four, so four bytes a line is an upper bound on what was just emitted.
    fn observe(&mut self, emitted: &[u8]) {
        for line in emitted.split(|byte| *byte == b'\n') {
            if line.starts_with(b"    ") {
                self.bytes += 4;
                if line.windows(3).any(|window| window == b", =") {
                    self.pending = true;
                }
            }
        }
    }

    fn drain_if_needed<W: Write>(&mut self, w: &mut W) -> Result<(), LaminaError> {
        if !self.pending || self.bytes < POOL_REACH {
            return Ok(());
        }
        self.index += 1;
        let label = format!("{}pool{}", self.prefix, self.index);
        writeln!(w, "    b {label}")?;
        writeln!(w, "    .ltorg")?;
        writeln!(w, "{label}:")?;
        self.clear();
        Ok(())
    }

    fn clear(&mut self) {
        self.pending = false;
        self.bytes = 0;
    }
}

/// Whether the block's last instruction leaves no path into the bytes that follow it,
/// so a Thumb-1 literal pool can be dropped there.
fn block_ends_unconditionally(inst: Option<&MirInst>, elided_jump: bool) -> bool {
    match inst {
        Some(MirInst::Jmp { .. }) => !elided_jump,
        Some(
            MirInst::Br { .. }
            | MirInst::Switch { .. }
            | MirInst::Ret { .. }
            | MirInst::TailCall { .. }
            | MirInst::Unreachable,
        ) => true,
        _ => false,
    }
}

fn compile_single_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    isa: Isa,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, CodegenError> {
    compile_function(func_name, func, target_os, isa, settings, signatures)
        .map_err(lamina_to_codegen_error)
}

fn compile_function(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    isa: Isa,
    settings: &MirCodegenSettings,
    signatures: &Signatures,
) -> Result<Vec<u8>, LaminaError> {
    reject_unlowered_values(isa, func_name, func)?;
    check_frame_alignment(func_name, func)?;
    let cortexm = CortexMAbi::new(target_os);
    let label = cortexm.mangle(func_name);
    let mut w: Vec<u8> = Vec::new();
    let mut scratch = Scratch::default();

    let mut frame = FrameMap::from_function(func);
    let memory_return = returns_in_memory(func.sig.ret_aggregate);
    let sret_raw = memory_return.then(|| -frame.frame_size - 8);
    if sret_raw.is_some() {
        frame.frame_size += 16;
    }
    let layout = Layout::new(
        call::outgoing_space(isa, func, signatures),
        frame.frame_size,
    );
    let access = Access {
        isa,
        frame: &frame,
        layout,
    };

    writeln!(w, "    .thumb_func")?;
    writeln!(w, "{label}:")?;
    writeln!(w, "    .cfi_startproc")?;
    writeln!(w, "    push {SAVED_LIST}")?;
    writeln!(w, "    .cfi_def_cfa_offset {SAVED_BYTES}")?;
    for (index, number) in [4, 5, 6, 7, 14].into_iter().enumerate() {
        writeln!(
            w,
            "    .cfi_offset {number}, {}",
            index as i32 * 4 - SAVED_BYTES
        )?;
    }
    sp_adjust(&mut w, isa, "sub", layout.total)?;
    writeln!(w, "    .cfi_def_cfa_offset {}", SAVED_BYTES + layout.total)?;

    if let Some(raw) = sret_raw {
        operand::frame_store(&mut w, isa, "r0", layout.slot(raw), &mut scratch)?;
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
    let places = classify(&kinds, None, true);
    let skip = usize::from(memory_return);
    for ((param, kind), place) in func
        .sig
        .params
        .iter()
        .zip(kinds.iter().skip(skip))
        .zip(places.iter().skip(skip))
    {
        if let Some(offset) = access.slot_of(&param.reg) {
            spill_parameter(&mut w, param.ty, kind, place, offset, &access, &mut scratch)?;
        }
    }

    let epilogue = format!(".Lret_{label}");
    let block_prefix = format!(".L_{label}.");
    let ctx = Ctx {
        isa,
        access,
        abi: &cortexm,
        signatures,
        epilogue: &epilogue,
        block_prefix: &block_prefix,
        ret_aggregate: func.sig.ret_aggregate,
        sret_slot: sret_raw.map(|raw| layout.slot(raw)),
        settings,
    };

    let plan = FunctionEmissionPlan::new(func);
    if plan
        .order
        .first()
        .is_some_and(|&index| func.blocks[index].label != func.entry)
    {
        writeln!(w, "    b {block_prefix}{}", func.entry)?;
    }
    let mut pool = LiteralPool::new(&block_prefix);
    for (slot, &index) in plan.order.iter().enumerate() {
        let block = &func.blocks[index];
        let next = plan
            .order
            .get(slot + 1)
            .map(|&i| func.blocks[i].label.as_str())
            .unwrap_or("");
        writeln!(w, "{block_prefix}{}:", block.label)?;
        let mut elided = false;
        for inst in &block.instructions {
            if matches!(&inst.instr, MirInst::Jmp { target } if target == next) {
                elided = true;
                continue;
            }
            let start = w.len();
            emit::emit_instruction(&mut w, &inst.instr, &ctx, &mut scratch, inst.loc)?;
            if isa == Isa::Thumb1 {
                pool.observe(&w[start..]);
                pool.drain_if_needed(&mut w)?;
            }
        }
        // ARMv6-M reaches a symbol through a literal pool, which `ldr rd, =sym` asks
        // the assembler to place at the next `.ltorg`. Dropping one after every block
        // that cannot fall through keeps every pool within the 1020 bytes a
        // PC-relative load reaches forward.
        if isa == Isa::Thumb1
            && block_ends_unconditionally(block.instructions.last().map(|i| &i.instr), elided)
        {
            writeln!(w, "    .ltorg")?;
            pool.clear();
        }
    }

    writeln!(w, "{epilogue}:")?;
    writeln!(w, "    .cfi_remember_state")?;
    sp_adjust(&mut w, isa, "add", layout.total)?;
    writeln!(w, "    .cfi_def_cfa_offset {SAVED_BYTES}")?;
    writeln!(w, "    pop {{r4, r5, r6, r7, pc}}")?;
    writeln!(w, "    .cfi_restore_state")?;
    writeln!(w, "    .cfi_endproc")?;
    if isa == Isa::Thumb1 {
        writeln!(w, "    .ltorg")?;
    }
    Ok(w)
}

/// A function named like a register would come out as `bl r0`, which the assembler
/// reads as an instruction rather than a symbol.
fn reject_register_symbols(module: &MirModule) -> Result<(), LaminaError> {
    let mut names: Vec<&str> = module.functions.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        if is_register_name(name) {
            return Err(unsupported(format!(
                "@{name} is a Thumb register name, so the assembler cannot use it as a branch \
                 target; rename the function"
            )));
        }
    }
    Ok(())
}

fn is_register_name(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "sp" | "lr" | "pc" | "fp" | "ip" | "sl" | "sb"
    ) {
        return true;
    }
    let mut chars = lower.chars();
    let (Some('r'), digits) = (chars.next(), chars.as_str()) else {
        return false;
    };
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && digits.parse::<u8>().is_ok_and(|n| n <= 15)
}

pub fn generate_mir_cortexm_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    let isa = isa_of(target_arch).ok_or_else(|| {
        unsupported(format!(
            "{target_arch:?} is not a Cortex-M target for this backend"
        ))
    })?;
    if !matches!(
        target_os,
        TargetOperatingSystem::Linux | TargetOperatingSystem::Unknown
    ) {
        return Err(unsupported(format!(
            "{} code generation targets bare-metal ELF; {target_os:?} is not supported",
            isa.target_name()
        )));
    }
    if let Some(variant) = settings.arch_variant.armv7()
        && variant.hard_float
    {
        return Err(unsupported(format!(
            "{} code generation emits no VFP instructions; a hard-float variant is not \
             supported",
            isa.target_name()
        )));
    }
    validate_module_call_parameters(module, target_arch)?;
    reject_register_symbols(module)?;

    let cortexm = CortexMAbi::new(target_os);
    let settings_arc = Arc::new(settings.clone());
    let signatures = Arc::new(Signatures::from_module(module));
    let results =
        compile_functions_parallel(module, target_os, codegen_units, move |name, func, os| {
            compile_single_function(
                name,
                func,
                os,
                isa,
                settings_arc.as_ref(),
                signatures.as_ref(),
            )
        })
        .map_err(parallel_codegen_error)?;

    writeln!(writer, ".syntax unified")?;
    writeln!(writer, ".arch {}", isa.arch_name())?;
    writeln!(writer, ".thumb")?;
    writeln!(writer, ".text")?;
    emit_debug_file(writer, settings)?;

    for func_name in external_names(module) {
        writeln!(writer, ".extern {func_name}")?;
    }
    emit_global_directives(writer, module, |name| cortexm.mangle_function_name(name))?;
    emit_symbol_visibility(writer, module, target_os, |name| {
        cortexm.mangle_function_name(name)
    })?;

    for result in results {
        writer.write_all(&arm_section_types(result.assembly))?;
    }
    emit_global_data(writer, module, target_os, |name| {
        cortexm.mangle_function_name(name)
    })?;
    // `@` starts a comment in ARM assembly, so the note's type is spelled with `%`.
    writeln!(writer, ".section .note.GNU-stack,\"\",%progbits")?;
    Ok(())
}

pub fn capabilities() -> CapabilitySet {
    [
        CodegenCapability::IntegerArithmetic,
        CodegenCapability::ControlFlow,
        CodegenCapability::FunctionCalls,
        CodegenCapability::Recursion,
        CodegenCapability::StackAllocation,
        CodegenCapability::MemoryOperations,
        CodegenCapability::ForeignFunctionInterface,
        CodegenCapability::AggregateArguments,
        CodegenCapability::IndirectCalls,
    ]
    .into_iter()
    .collect()
}

/// Trait-backed MIR to Cortex-M code generator.
pub struct CortexMCodegen<'a> {
    base: CodegenBase<'a>,
    arch: TargetArchitecture,
}

impl<'a> CortexMCodegen<'a> {
    pub fn new(arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os, PointerWidth::Bits32),
            arch,
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
        generate_mir_cortexm_with_units_and_settings(
            module,
            writer,
            self.arch,
            self.base.target_os,
            codegen_units,
            &MirCodegenSettings::default(),
        )
    }
}

impl<'a> Codegen for CortexMCodegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Unknown;
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
        let arch = self.arch;
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_cortexm_with_units_and_settings(
                    module,
                    writer,
                    arch,
                    target_os,
                    codegen_units,
                    &MirCodegenSettings::default(),
                )
            },
            "cortexm",
            units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "binary emission is not implemented for the Cortex-M MIR backend".into(),
        ))
    }
}
