//! Common code for MIR codegen backends.

use lamina_mir::{Instruction as MirInstruction, ScalarType, SourceLoc, instruction::AddressMode};
use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fmt::{Debug, Display},
    hash::Hash,
    io::Write,
    mem,
    sync::{Arc, mpsc},
    thread,
};

use crate::{
    error::LaminaError,
    mir::{
        CodePlacement, Function, Global, Immediate, Instruction, IntBinOp, MirType,
        Module as MirModule, Operand, Register, RegisterClass, Signature, SymbolVisibility,
        VirtualReg,
    },
    mir_codegen::{CodegenError, CodegenOptions, MirCodegenSettings},
};
use lamina_codegen::{Allocation, LiveInterval, os_uses_coff, os_uses_macho};
use lamina_platform::{MemoryOrdering, PointerWidth, TargetOperatingSystem};

pub fn unsupported_io(operation: &str, os: TargetOperatingSystem) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "{operation} has no implementation for {os:?}"
    )))
}

/// The refusal a backend gives for `<op>.ovf`, which it must name rather than lower
/// to the wrapping arithmetic alone.
pub fn unsupported_overflow_flag(backend: &str) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "checked arithmetic ('<op>.ovf') is not supported by the {backend} backend; it is implemented for x86_64 and s390x"
    )))
}

/// The refusal a backend gives for a thread-local address, whose relocations are
/// target-specific and implemented for x86_64 and s390x.
pub fn unsupported_thread_local(backend: &str) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "thread-local storage ('@thread_local') is not supported by the {backend} backend; it is implemented for x86_64 and s390x, and for AArch64 on macOS"
    )))
}

/// The refusal a backend gives for inline assembly, whose operand placement and
/// clobber names are target-specific and implemented for x86_64, aarch64 and s390x.
pub fn unsupported_inline_asm(backend: &str) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "inline assembly is not supported by the {backend} backend; it is implemented for x86_64, aarch64 and s390x"
    )))
}

/// Rejects modules that declare a thread-local global, for backends that have no
/// TLS relocations.
pub fn reject_thread_local_globals(module: &MirModule, backend: &str) -> Result<(), LaminaError> {
    match module
        .globals
        .iter()
        .find(|(_, global)| global.thread_local.is_some())
    {
        None => Ok(()),
        Some((name, _)) => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "thread-local storage is not supported by the {backend} backend; it is implemented for x86_64 and s390x, and for AArch64 on macOS (global '{name}')"
            ),
        ))),
    }
}

/// The refusal for a thread-local global on a target whose TLS layout the backend does not
/// emit.
pub fn unsupported_thread_local_on_os(name: &str, target_os: TargetOperatingSystem) -> LaminaError {
    let needs = if os_uses_macho(target_os) {
        "__DATA,__thread_vars and @TLVP relocations"
    } else {
        "a TEB TLS-array load (%gs:0x58) and @SECREL32 relocations"
    };
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "thread-local storage on {target_os:?} is not supported: '{name}' needs {needs}, which this backend does not emit"
    )))
}

/// Rejects the thread-local globals the x86_64 backend cannot place on a target, before any
/// code is written. Mach-O needs TLV descriptors it does not write. COFF takes `.tls$` data
/// aligned to at most 8 bytes, since the TLS directory weld builds has no room for more.
pub fn reject_x86_64_thread_local_globals(
    module: &MirModule,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    let mut globals: Vec<(&str, usize)> = module
        .globals
        .iter()
        .filter(|(_, global)| global.thread_local.is_some())
        .map(|(name, global)| (name.as_str(), global.align))
        .collect();
    globals.sort_unstable();
    if os_uses_macho(target_os)
        && let Some((name, _)) = globals.first()
    {
        return Err(unsupported_thread_local_on_os(name, target_os));
    }
    if os_uses_coff(target_os)
        && let Some((name, align)) = globals.iter().find(|(_, align)| *align > 8)
    {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!(
                "thread-local '{name}' is aligned to {align} bytes, but the .tls$ section takes at most 8-byte alignment on {target_os:?}"
            ),
        )));
    }
    Ok(())
}

/// Rejects a module with a thread-local global on a COFF or Mach-O target, before any
/// code is written.
pub fn reject_thread_local_globals_on_os(
    module: &MirModule,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    if !(os_uses_coff(target_os) || os_uses_macho(target_os)) {
        return Ok(());
    }
    let mut names: Vec<&str> = module
        .globals
        .iter()
        .filter(|(_, global)| global.thread_local.is_some())
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    match names.first() {
        None => Ok(()),
        Some(name) => Err(unsupported_thread_local_on_os(name, target_os)),
    }
}

/// Rejects a compare-exchange whose failure ordering is not one a failed exchange can
/// carry. The rule lives in [`MemoryOrdering::cmpxchg_failure_rejection`]; this walks a
/// module so every backend gets it from one place.
pub fn reject_invalid_cmpxchg_orderings(module: &MirModule) -> Result<(), LaminaError> {
    for (name, function) in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let MirInstruction::AtomicCompareExchange {
                    success_ordering,
                    failure_ordering,
                    ..
                } = &instruction.instr
                else {
                    continue;
                };
                if let Some(reason) =
                    MemoryOrdering::cmpxchg_failure_rejection(*success_ordering, *failure_ordering)
                {
                    return Err(LaminaError::ValidationError(format!(
                        "atomic_cmpxchg cannot use {failure_ordering} failure ordering with \
                         {success_ordering} success ordering: {reason} (function '{name}')"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Rejects an atomic access whose address cannot hold the natural alignment of its
/// type.
///
/// An atomic access is naturally aligned by rule: an LL/SC target faults outright on
/// any other address, and the lock-prefixed forms that do not fault carry a cost the
/// IR never asks for. No atomic instruction states an alignment, so the base register
/// is the program's responsibility, but a displacement the address mode adds to it is
/// this compiler's: an offset, or an index scale, that is not a multiple of the
/// accessed size moves a naturally aligned base off its boundary.
pub fn reject_underaligned_atomics(
    module: &MirModule,
    pointer: PointerWidth,
) -> Result<(), LaminaError> {
    for (name, function) in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let (ty, addr, what) = match &instruction.instr {
                    MirInstruction::AtomicLoad { ty, addr, .. } => (ty, addr, "atomic load"),
                    MirInstruction::AtomicStore { ty, addr, .. } => (ty, addr, "atomic store"),
                    MirInstruction::AtomicBinary { ty, addr, .. } => {
                        (ty, addr, "atomic read-modify-write")
                    }
                    MirInstruction::AtomicCompareExchange { ty, addr, .. } => {
                        (ty, addr, "atomic compare-exchange")
                    }
                    _ => continue,
                };
                let natural = ty.alignment(pointer) as i64;
                let (displacement, step) = match addr {
                    AddressMode::BaseOffset { offset, .. } => (i64::from(*offset), natural),
                    AddressMode::BaseIndexScale { offset, scale, .. } => {
                        (i64::from(*offset), i64::from(*scale))
                    }
                };
                if displacement % natural != 0 || step % natural != 0 {
                    return Err(LaminaError::ValidationError(format!(
                        "a {what} of {ty} must be naturally aligned ({natural} bytes), and the \
                         address {addr} cannot be (function '{name}')"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Rejects a load or store that states an alignment below the natural alignment of
/// its type, for backends whose memory instructions fault on an unaligned address.
pub fn reject_underaligned_accesses(
    module: &MirModule,
    pointer: PointerWidth,
    backend: &str,
) -> Result<(), LaminaError> {
    for (name, function) in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let (ty, attrs, what) = match &instruction.instr {
                    MirInstruction::Load { ty, attrs, .. } => (ty, attrs, "load"),
                    MirInstruction::Store { ty, attrs, .. } => (ty, attrs, "store"),
                    _ => continue,
                };
                let natural = ty.alignment(pointer);
                if attrs.stated_align && usize::from(attrs.align) < natural {
                    return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                        format!(
                            "{backend} cannot issue a {what} of {ty} at alignment {} (natural alignment is {natural}); its memory instructions fault on an unaligned address (function '{name}')",
                            attrs.align
                        ),
                    )));
                }
            }
        }
    }
    Ok(())
}

pub fn parallel_codegen_error(error: CodegenError) -> LaminaError {
    LaminaError::CodegenError(error)
}

/// Assign one stack slot per virtual register defined or used in `func`.
///
/// Defined registers are laid out first, then used-only registers. The byte
/// offset for each slot index is computed by the backend-specific `offset_for_slot`.
/// Returns the slot map together with the set of defined registers, which some
/// backends need when emitting instructions.
///
/// Registers are collected into `BTreeSet`s so slot indices — and therefore the
/// emitted stack offsets — are a deterministic function of the register ids,
/// keeping codegen reproducible across runs.
pub fn assign_stack_slots(
    func: &Function,
    offset_for_slot: impl Fn(usize) -> i32,
) -> (HashMap<VirtualReg, i32>, BTreeSet<VirtualReg>, usize) {
    assign_stack_slots_where(func, offset_for_slot, |_| true)
}

pub(super) fn assign_stack_slots_where(
    func: &Function,
    offset_for_slot: impl Fn(usize) -> i32,
    needs_slot: impl Fn(&VirtualReg) -> bool,
) -> (HashMap<VirtualReg, i32>, BTreeSet<VirtualReg>, usize) {
    assign_stack_slots_framed(func, offset_for_slot, needs_slot, 0)
}

/// `frame_residue` is the address of the frame base modulo the alignment that
/// `stackalloc` asks for, so an offset `o` is aligned when `frame_residue + o` is a
/// multiple of it.
pub(super) fn assign_stack_slots_framed(
    func: &Function,
    offset_for_slot: impl Fn(usize) -> i32,
    needs_slot: impl Fn(&VirtualReg) -> bool,
    frame_residue: i64,
) -> (HashMap<VirtualReg, i32>, BTreeSet<VirtualReg>, usize) {
    let mut def_regs: BTreeSet<VirtualReg> = BTreeSet::new();
    let mut used_regs: BTreeSet<VirtualReg> = BTreeSet::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(Register::Virtual(vreg)) = inst.def_reg() {
                def_regs.insert(*vreg);
            }
            if let Instruction::StackAddr { slot, .. } = &inst.instr {
                used_regs.insert(*slot);
            }
            for reg in inst.use_regs() {
                if let Register::Virtual(vreg) = reg {
                    used_regs.insert(*vreg);
                }
            }
        }
    }

    let mut stack_slots: HashMap<VirtualReg, i32> = HashMap::new();
    let mut next_slot = 0usize;
    let claim = |vreg: &VirtualReg, slots: &mut HashMap<VirtualReg, i32>, next: &mut usize| {
        let extra = func.stack_reservations.get(vreg).copied().unwrap_or(0);
        let align = func.stack_alignments.get(vreg).copied();
        let index = claim_stack_slot_framed(next, extra, align, &offset_for_slot, frame_residue);
        slots.insert(*vreg, offset_for_slot(index));
    };
    for vreg in &def_regs {
        if needs_slot(vreg) && !stack_slots.contains_key(vreg) {
            claim(vreg, &mut stack_slots, &mut next_slot);
        }
    }
    for vreg in &used_regs {
        if needs_slot(vreg) && !def_regs.contains(vreg) && !stack_slots.contains_key(vreg) {
            claim(vreg, &mut stack_slots, &mut next_slot);
        }
    }
    (stack_slots, def_regs, next_slot)
}

/// Claims the slot block of one register and returns the index its address uses.
///
/// The block is `extra` slots deep beyond `next`, and the register addresses its
/// deepest slot. When `align` is set, that index is stepped upward until its offset
/// is a multiple of `align`, and the skipped indices stay unused.
pub(super) fn claim_stack_slot(
    next: &mut usize,
    extra: usize,
    align: Option<usize>,
    offset_for_slot: &impl Fn(usize) -> i32,
) -> usize {
    claim_stack_slot_framed(next, extra, align, offset_for_slot, 0)
}

pub(super) fn claim_stack_slot_framed(
    next: &mut usize,
    extra: usize,
    align: Option<usize>,
    offset_for_slot: &impl Fn(usize) -> i32,
    frame_residue: i64,
) -> usize {
    let mut index = *next + extra;
    if let Some(align) = align {
        let limit = index + align;
        while (i64::from(offset_for_slot(index)) + frame_residue) % align as i64 != 0 {
            index += 1;
            assert!(
                index < limit,
                "no slot index within {align} steps lands on a {align}-byte boundary"
            );
        }
    }
    *next = index + 1;
    index
}

pub(super) fn integer_conversion(
    instruction: &Instruction,
    pointer_width: PointerWidth,
) -> Option<Vec<Instruction>> {
    integer_conversion_with_width(instruction, MirType::Scalar(ScalarType::I64), pointer_width)
}

pub(super) fn integer_conversion_with_width(
    instruction: &Instruction,
    word_ty: MirType,
    pointer_width: PointerWidth,
) -> Option<Vec<Instruction>> {
    let word_bits = word_ty.size_bytes(pointer_width) * 8;
    let Instruction::IntConvert {
        from,
        to,
        signed,
        dst,
        src,
    } = instruction
    else {
        return None;
    };
    let width = from
        .size_bytes(pointer_width)
        .min(to.size_bytes(pointer_width))
        * 8;
    let mut operations = if *signed {
        let shift = (word_bits - width) as i64;
        vec![
            Instruction::IntBinary {
                op: IntBinOp::Shl,
                ty: word_ty,
                dst: dst.clone(),
                lhs: src.clone(),
                rhs: Operand::Immediate(Immediate::I64(shift)),
            },
            Instruction::IntBinary {
                op: IntBinOp::AShr,
                ty: word_ty,
                dst: dst.clone(),
                lhs: Operand::Register(dst.clone()),
                rhs: Operand::Immediate(Immediate::I64(shift)),
            },
        ]
    } else if width == word_bits {
        vec![Instruction::Copy {
            ty: word_ty,
            dst: dst.clone(),
            src: src.clone(),
        }]
    } else {
        vec![Instruction::IntBinary {
            op: IntBinOp::And,
            ty: word_ty,
            dst: dst.clone(),
            lhs: src.clone(),
            rhs: Operand::Immediate(Immediate::I64(((1u64 << width) - 1) as i64)),
        }]
    };
    if to.size_bytes(pointer_width) < word_ty.size_bytes(pointer_width) {
        operations.push(Instruction::Copy {
            ty: *to,
            dst: dst.clone(),
            src: Operand::Register(dst.clone()),
        });
    }
    Some(operations)
}

/// Base structure for codegen backends with common fields.
pub struct CodegenBase<'a> {
    pub target_os: TargetOperatingSystem,
    pub pointer_width: PointerWidth,
    pub module: Option<&'a MirModule>,
    pub prepared: bool,
    pub verbose: bool,
    pub output: Vec<u8>,
    pub codegen_units: usize,
}

impl<'a> CodegenBase<'a> {
    pub fn new(target_os: TargetOperatingSystem, pointer_width: PointerWidth) -> Self {
        Self {
            target_os,
            pointer_width,
            module: None,
            prepared: false,
            verbose: false,
            output: Vec::new(),
            codegen_units: 1,
        }
    }

    pub fn set_module(&mut self, module: &'a MirModule) {
        self.module = Some(module);
    }

    pub fn drain_output(&mut self) -> Vec<u8> {
        mem::take(&mut self.output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_base(
        &mut self,
        _types: &HashMap<String, MirType>,
        _globals: &HashMap<String, Global>,
        _funcs: &HashMap<String, Signature>,
        codegen_units: usize,
        verbose: bool,
        _options: &[CodegenOptions],
        _input_name: &str,
    ) -> Result<(), CodegenError> {
        const MAX_CODEGEN_UNITS: usize = 16;
        if codegen_units == 0 {
            return Err(CodegenError::InvalidCodegenOptions(
                "codegen_units must be at least 1".to_string(),
            ));
        }
        if codegen_units > MAX_CODEGEN_UNITS {
            return Err(CodegenError::InvalidCodegenOptions(format!(
                "codegen_units exceeds maximum: {MAX_CODEGEN_UNITS}"
            )));
        }
        self.codegen_units = codegen_units;
        self.verbose = verbose;
        self.prepared = true;
        Ok(())
    }

    pub fn compile_base(&self) -> Result<(), CodegenError> {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(
                "Codegen not prepared".to_string(),
            ));
        }
        Ok(())
    }

    pub fn finalize_base(&mut self) -> Result<(), CodegenError> {
        self.module = None;
        self.prepared = false;
        Ok(())
    }

    pub fn emit_asm_base<F>(&mut self, emit_fn: F, backend_name: &str) -> Result<(), CodegenError>
    where
        F: FnOnce(&MirModule, &mut Vec<u8>, TargetOperatingSystem) -> Result<(), LaminaError>,
    {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(format!(
                "emit_asm called before prepare for {backend_name}"
            )));
        }
        let module = self.module.ok_or_else(|| {
            CodegenError::InvalidCodegenOptions(format!(
                "No module set for emission in {backend_name} backend"
            ))
        })?;
        self.output.clear();
        emit_fn(module, &mut self.output, self.target_os).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("{backend_name} emission failed: {e}"))
        })
    }

    pub fn emit_asm_base_with_units<F>(
        &mut self,
        emit_fn: F,
        backend_name: &str,
        codegen_units: usize,
    ) -> Result<(), CodegenError>
    where
        F: FnOnce(
            &MirModule,
            &mut Vec<u8>,
            TargetOperatingSystem,
            usize,
        ) -> Result<(), LaminaError>,
    {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(format!(
                "emit_asm called before prepare for {backend_name}"
            )));
        }
        let module = self.module.ok_or_else(|| {
            CodegenError::InvalidCodegenOptions(format!(
                "No module set for emission in {backend_name} backend"
            ))
        })?;
        self.output.clear();
        emit_fn(module, &mut self.output, self.target_os, codegen_units).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("{backend_name} emission failed: {e}"))
        })
    }
}

/// Compilation task for parallel execution.
pub struct CompilationTask {
    pub func_name: String,
    pub func: Function,
    pub func_index: usize,
}

/// Compilation result from a worker thread.
pub struct CompilationResult {
    pub func_name: String,
    pub func_index: usize,
    pub assembly: Vec<u8>,
}

/// Compile functions in parallel using channels (no Arc<Mutex>).
///
/// Distributes function compilation tasks across multiple threads
/// using message passing. Each thread compiles functions independently and sends
/// results back via channels. The main thread collects and merges results.
pub fn compile_functions_parallel<F>(
    module: &MirModule,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
    compile_func: F,
) -> Result<Vec<CompilationResult>, CodegenError>
where
    F: Fn(&str, &Function, TargetOperatingSystem) -> Result<Vec<u8>, CodegenError>
        + Send
        + Sync
        + 'static,
{
    if codegen_units == 1 {
        return compile_functions_sequential(module, target_os, compile_func);
    }

    let mut functions: Vec<(String, Function)> = module
        .functions
        .iter()
        .filter_map(|(name, func)| {
            if module.is_external(name) {
                None
            } else {
                Some((name.clone(), func.clone()))
            }
        })
        .collect();

    functions.sort_by(|a, b| a.0.cmp(&b.0));

    let functions: Vec<(String, Function, usize)> = functions
        .into_iter()
        .enumerate()
        .map(|(idx, (name, func))| (name, func, idx))
        .collect();

    if functions.is_empty() {
        return Ok(Vec::new());
    }

    let compile_func_arc = Arc::new(compile_func);

    let num_workers = codegen_units.min(functions.len());
    let mut handles = Vec::new();
    let mut task_senders = Vec::new();
    let (result_sender, result_receiver) = mpsc::channel::<CompilationResult>();

    for _worker_id in 0..num_workers {
        let (task_sender, task_receiver) = mpsc::channel::<CompilationTask>();
        task_senders.push(task_sender);
        let result_sender = result_sender.clone();
        let compile_func_arc = compile_func_arc.clone();
        let handle = thread::spawn(move || -> Result<(), CodegenError> {
            let mut error_occurred = None;
            while let Ok(task) = task_receiver.recv() {
                if error_occurred.is_some() {
                    continue;
                }
                match compile_func_arc(&task.func_name, &task.func, target_os) {
                    Ok(assembly) => {
                        let assembly = match place_function_assembly(
                            &task.func_name,
                            &task.func,
                            target_os,
                            assembly,
                        ) {
                            Ok(assembly) => assembly,
                            Err(e) => {
                                error_occurred = Some(e);
                                continue;
                            }
                        };
                        if result_sender
                            .send(CompilationResult {
                                func_name: task.func_name,
                                func_index: task.func_index,
                                assembly,
                            })
                            .is_err()
                        {
                            error_occurred = Some(CodegenError::InvalidCodegenOptions(
                                "Failed to send compilation result".to_string(),
                            ));
                        }
                    }
                    Err(e) => {
                        error_occurred = Some(e);
                    }
                }
            }
            error_occurred.map_or(Ok(()), Err)
        });
        handles.push(handle);
    }

    drop(result_sender);

    for (idx, (func_name, func, func_index)) in functions.iter().enumerate() {
        let worker_id = idx % num_workers;
        let _ = task_senders[worker_id].send(CompilationTask {
            func_name: func_name.clone(),
            func: func.clone(),
            func_index: *func_index,
        });
    }
    drop(task_senders);

    let mut results: Vec<CompilationResult> = Vec::new();
    let mut worker_errors = Vec::new();
    let mut lost_results = false;

    for _ in 0..functions.len() {
        match result_receiver.recv() {
            Ok(result) => results.push(result),
            Err(_) => {
                lost_results = true;
                break;
            }
        }
    }

    for (idx, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                worker_errors.push(e);
            }
            Err(e) => {
                worker_errors.push(CodegenError::InvalidCodegenOptions(format!(
                    "Worker thread {idx} panicked: {e:?}"
                )));
            }
        }
    }

    // A worker that fails to compile sends no result, so the receiver closes
    // early. Report what the worker actually hit, not the closed channel.
    if let Some(first_error) = worker_errors.into_iter().next() {
        return Err(first_error);
    }
    if lost_results {
        return Err(CodegenError::InvalidCodegenOptions(
            "Worker thread communication error".to_string(),
        ));
    }

    results.sort_by_key(|r| r.func_index);
    Ok(results)
}

/// Compile functions sequentially (fallback for single-threaded).
fn compile_functions_sequential<F>(
    module: &MirModule,
    target_os: TargetOperatingSystem,
    compile_func: F,
) -> Result<Vec<CompilationResult>, CodegenError>
where
    F: Fn(&str, &Function, TargetOperatingSystem) -> Result<Vec<u8>, CodegenError>,
{
    let mut functions: Vec<(&str, &Function)> = module
        .functions
        .iter()
        .filter_map(|(name, func)| {
            if module.is_external(name) {
                None
            } else {
                Some((name.as_str(), func))
            }
        })
        .collect();

    functions.sort_by(|a, b| a.0.cmp(b.0));

    let mut results = Vec::new();
    for (idx, (func_name, func)) in functions.into_iter().enumerate() {
        let assembly = place_function_assembly(
            func_name,
            func,
            target_os,
            compile_func(func_name, func, target_os)?,
        )?;
        results.push(CompilationResult {
            func_name: func_name.to_string(),
            func_index: idx,
            assembly,
        });
    }
    results.sort_by_key(|r| r.func_index);
    Ok(results)
}

/// Emit the single `.file 1` entry that every `.loc` in the translation unit refers to.
pub fn emit_debug_file<W: Write>(
    writer: &mut W,
    settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    if settings.emit_asm_debug_lines {
        let tag = settings.debug_file_tag.replace('"', "'");
        writeln!(writer, "    .file 1 \"{tag}\"")?;
    }
    Ok(())
}

/// Emit the `.loc` row for an instruction that came from `loc`. An instruction a
/// pass synthesized has no position and gets no row.
pub fn emit_source_loc<W: Write>(
    writer: &mut W,
    settings: &MirCodegenSettings,
    loc: Option<SourceLoc>,
) -> Result<(), LaminaError> {
    if settings.emit_asm_debug_lines
        && let Some(loc) = loc
    {
        writeln!(writer, "    .loc 1 {} {}", loc.line, loc.column)?;
    }
    Ok(())
}

/// Places a function's assembly: `@section` names its section, otherwise `@cold` and `@hot`
/// functions get their own ELF text section so the linker can group them apart from the
/// rest of the code. `@align` pads the entry to its boundary.
fn place_function_assembly(
    func_name: &str,
    func: &Function,
    target_os: TargetOperatingSystem,
    assembly: Vec<u8>,
) -> Result<Vec<u8>, CodegenError> {
    let elf = os_uses_elf_visibility(target_os);
    let section = match (&func.sig.section, func.sig.placement) {
        (Some(name), _) => Some(code_section_directive(func_name, name, target_os)?),
        (None, Some(CodePlacement::Cold)) if elf => Some(format!(
            "    .section .text.unlikely.{func_name},\"ax\",@progbits\n"
        )),
        (None, Some(CodePlacement::Hot)) if elf => Some(format!(
            "    .section .text.hot.{func_name},\"ax\",@progbits\n"
        )),
        (None, _) => None,
    };
    if section.is_none() && func.sig.align.is_none() {
        return Ok(assembly);
    }
    let mut placed = Vec::new();
    if let Some(section) = &section {
        placed.extend_from_slice(section.as_bytes());
    }
    if let Some(alignment) = func.sig.align {
        placed
            .extend_from_slice(format!("    .p2align {}\n", alignment.trailing_zeros()).as_bytes());
    }
    placed.extend_from_slice(&assembly);
    if section.is_some() {
        placed.extend_from_slice(b"    .text\n");
    }
    Ok(placed)
}

/// The directive that opens the section a `@section` name places code in. COFF and ELF
/// take any section name. Mach-O names a section by a `segment,section` pair of at most 16
/// bytes each, so the only names with a mapping are `.text`, which stays in `__TEXT,__text`,
/// and `.text.<name>`, which goes to `__TEXT,__<name>` when `<name>` is 1 to 14 bytes of
/// ASCII letters, digits and underscores. Any other name is refused.
fn code_section_directive(
    func_name: &str,
    name: &str,
    target_os: TargetOperatingSystem,
) -> Result<String, CodegenError> {
    if !os_uses_macho(target_os) || name == ".text" {
        return Ok(format!("    .section {name},\"ax\",@progbits\n"));
    }
    let suffix = name.strip_prefix(".text.").filter(|suffix| {
        (1..=14).contains(&suffix.len())
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    });
    match suffix {
        Some(suffix) => Ok(format!(
            "    .section __TEXT,__{suffix},regular,pure_instructions\n"
        )),
        None => Err(CodegenError::UnsupportedFeature(format!(
            "@section '{name}' on '{func_name}' has no section on {target_os:?}: Mach-O names a section by a segment and a section of at most 16 bytes, and only '.text' and '.text.<name>' (name of 1 to 14 letters, digits or underscores, placed in __TEXT,__<name>) are mapped"
        ))),
    }
}

/// ELF targets take `.internal`, `.hidden` and `.protected`. Mach-O has only
/// `.private_extern`, and COFF, wasm and GPU targets have no such symbol visibility.
fn os_uses_elf_visibility(target_os: TargetOperatingSystem) -> bool {
    matches!(
        target_os,
        TargetOperatingSystem::Linux
            | TargetOperatingSystem::Android
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
    )
}

fn visibility_directive(
    visibility: SymbolVisibility,
    target_os: TargetOperatingSystem,
) -> Option<&'static str> {
    if os_uses_macho(target_os) {
        return match visibility {
            SymbolVisibility::Protected => None,
            _ => Some(".private_extern"),
        };
    }
    if !os_uses_elf_visibility(target_os) {
        return None;
    }
    Some(match visibility {
        SymbolVisibility::Internal => ".internal",
        SymbolVisibility::Private => ".local",
        SymbolVisibility::Hidden => ".hidden",
        SymbolVisibility::Protected => ".protected",
    })
}

/// Emits the symbol-visibility and weak-linkage directives for every function annotated
/// with one.
pub fn emit_symbol_visibility<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    let mut names: Vec<(&str, SymbolVisibility)> = module
        .functions
        .iter()
        .filter(|(name, _)| !module.is_external(name))
        .filter_map(|(name, func)| func.sig.visibility.map(|v| (name.as_str(), v)))
        .collect();
    names.sort_unstable_by(|a, b| a.0.cmp(b.0));
    for (name, visibility) in names {
        if let Some(directive) = visibility_directive(visibility, target_os) {
            writeln!(writer, "{directive} {}", mangle(name))?;
        }
    }
    let mut weak: Vec<&str> = module
        .functions
        .iter()
        .filter(|(name, func)| !module.is_external(name) && func.sig.weak)
        .map(|(name, _)| name.as_str())
        .collect();
    weak.sort_unstable();
    let directive = if os_uses_macho(target_os) {
        ".weak_definition"
    } else {
        ".weak"
    };
    for name in weak {
        writeln!(writer, "{directive} {}", mangle(name))?;
    }
    Ok(())
}

/// Emit `.globl` for `main`, every exported function, and every function whose visibility
/// is `hidden` or `protected`, which stay linkable from other objects. Other functions stay
/// local.
pub fn emit_global_directives<W: Write>(
    writer: &mut W,
    module: &MirModule,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    let mut names: Vec<&str> = module
        .exported_functions
        .iter()
        .map(String::as_str)
        .chain(module.functions.contains_key("main").then_some("main"))
        .chain(
            module
                .functions
                .iter()
                .filter(|(name, func)| {
                    !module.is_external(name)
                        && matches!(
                            func.sig.visibility,
                            Some(SymbolVisibility::Hidden | SymbolVisibility::Protected)
                        )
                })
                .map(|(name, _)| name.as_str()),
        )
        .chain(
            module
                .globals
                .iter()
                .filter(|(_, global)| global.exported)
                .map(|(name, _)| name.as_str()),
        )
        .collect();
    names.sort_unstable();
    names.dedup();
    for name in names {
        writeln!(writer, ".globl {}", mangle(name))?;
    }
    Ok(())
}

/// Rejects modules that declare globals, for backends that cannot place them yet.
pub fn reject_module_globals(module: &MirModule, backend: &str) -> Result<(), LaminaError> {
    if module.globals.is_empty() {
        return Ok(());
    }
    Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
        format!("{backend} does not yet support global variables"),
    )))
}

/// Rejects `@weak`, `@section` and `@align` functions for backends that do not emit them.
pub fn reject_function_placement(module: &MirModule, backend: &str) -> Result<(), LaminaError> {
    let placed = module
        .functions
        .iter()
        .find(|(_, func)| func.sig.weak || func.sig.section.is_some() || func.sig.align.is_some());
    match placed {
        None => Ok(()),
        Some((name, _)) => Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            format!("{backend} does not yet support @weak, @section or @align (function '{name}')"),
        ))),
    }
}

/// Without this note the linker marks the stack executable, so every ELF target emits it
/// even though nothing in the generated code needs a trampoline.
pub fn emit_gnu_stack_note<W: Write>(
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    if os_uses_macho(target_os)
        || os_uses_coff(target_os)
        || matches!(
            target_os,
            TargetOperatingSystem::Cuda | TargetOperatingSystem::AmdHsa
        )
    {
        return Ok(());
    }
    writeln!(writer, ".section .note.GNU-stack,\"\",@progbits")?;
    Ok(())
}

/// Emits storage for every global. Globals whose initializer is all zero (or absent) go to
/// the bss section, the rest to data.
pub fn emit_global_data<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    reject_thread_local_globals_on_os(module, target_os)?;
    emit_global_storage(writer, module, target_os, mangle, |_| true)
}

/// Emits the globals of the x86_64 backend, whose thread-locals on COFF targets go to `.tls$`.
pub fn emit_x86_64_global_data<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    reject_x86_64_thread_local_globals(module, target_os)?;
    emit_global_storage(writer, module, target_os, mangle, |_| true)
}

/// Emits storage for the globals that are not thread-local.
pub fn emit_static_global_data<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    emit_global_storage(writer, module, target_os, mangle, |global| {
        global.thread_local.is_none()
    })
}

fn emit_global_storage<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
    select: impl Fn(&Global) -> bool,
) -> Result<(), LaminaError> {
    let (data_section, bss_section) = match target_os {
        os if os_uses_macho(os) => (".section __DATA,__data", ".section __DATA,__data"),
        _ => (".data", ".bss"),
    };
    let (tls_data_section, tls_bss_section) = if os_uses_coff(target_os) {
        (".section .tls$,\"aw\"", ".section .tls$,\"aw\"")
    } else {
        (
            ".section .tdata,\"awT\",@progbits",
            ".section .tbss,\"awT\",@nobits",
        )
    };
    let mut names: Vec<&str> = module
        .globals
        .iter()
        .filter(|(_, global)| select(global))
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    for name in names {
        let global = &module.globals[name];
        let (data_section, bss_section) = match global.thread_local {
            None => (data_section, bss_section),
            Some(_) => (tls_data_section, tls_bss_section),
        };
        write_global_storage(writer, &mangle(name), global, data_section, bss_section)?;
    }
    Ok(())
}

fn write_global_storage<W: Write>(
    writer: &mut W,
    label: &str,
    global: &Global,
    data_section: &str,
    bss_section: &str,
) -> Result<(), LaminaError> {
    let align_log2 = global.align.trailing_zeros();
    let size = global.size;
    match global.initializer.as_deref() {
        Some(bytes) if bytes.iter().any(|&b| b != 0) => {
            writeln!(writer, "{data_section}")?;
            writeln!(writer, ".p2align {align_log2}")?;
            writeln!(writer, "{label}:")?;
            for chunk in bytes.chunks(16) {
                let list: Vec<String> = chunk.iter().map(|b| format!("{b:#04x}")).collect();
                writeln!(writer, "    .byte {}", list.join(", "))?;
            }
            if let Some(tail) = size.checked_sub(bytes.len())
                && tail > 0
            {
                writeln!(writer, "    .zero {tail}")?;
            }
        }
        _ => {
            writeln!(writer, "{bss_section}")?;
            writeln!(writer, ".p2align {align_log2}")?;
            writeln!(writer, "{label}:")?;
            writeln!(writer, "    .zero {size}")?;
        }
    }
    Ok(())
}

/// The symbol of a thread-local's TLV descriptor. Source identifiers have no dots, so
/// the suffix cannot collide with a mangled global.
pub fn macho_tlv_descriptor(label: &str) -> String {
    format!("{label}.tlv")
}

/// Emits the thread-local globals of a Mach-O target. Each gets a 24-byte TLV descriptor
/// in `__thread_vars`: the `__tlv_bootstrap` thunk, a zero key that dyld fills, and the
/// storage address, which the linker turns into the offset from the template start. The
/// storage itself goes to `__thread_data` when it has a nonzero initializer, else to
/// `__thread_bss`.
pub fn emit_macho_thread_local_globals<W: Write>(
    writer: &mut W,
    module: &MirModule,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    let mut names: Vec<&str> = module
        .globals
        .iter()
        .filter(|(_, global)| global.thread_local.is_some())
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    if names.is_empty() {
        return Ok(());
    }
    writeln!(
        writer,
        ".section __DATA,__thread_vars,thread_local_variables"
    )?;
    for name in &names {
        let label = mangle(name);
        writeln!(writer, ".p2align 3")?;
        writeln!(writer, "{}:", macho_tlv_descriptor(&label))?;
        writeln!(writer, "    .quad __tlv_bootstrap")?;
        writeln!(writer, "    .quad 0")?;
        writeln!(writer, "    .quad {label}")?;
    }
    for name in names {
        let global = &module.globals[name];
        write_global_storage(
            writer,
            &mangle(name),
            global,
            ".section __DATA,__thread_data,thread_local_regular",
            ".section __DATA,__thread_bss,thread_local_zerofill",
        )?;
    }
    Ok(())
}

/// The scalar types each function takes and returns, so a call site can classify its
/// arguments without looking the callee up again.
pub struct CallSignatures {
    params: HashMap<String, Vec<MirType>>,
    returns: HashMap<String, MirType>,
    variadic: HashSet<String>,
}

impl CallSignatures {
    pub fn from_module(module: &MirModule) -> Self {
        Self {
            params: module
                .functions
                .iter()
                .map(|(name, func)| (name.clone(), func.sig.params.iter().map(|p| p.ty).collect()))
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_ty?)))
                .collect(),
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, _)| name.clone())
                .collect(),
        }
    }

    /// Whether the callee reads its arguments the way a variadic C function does. A
    /// declaration ending in `...` says so, and a parameter list that does not match the
    /// call means the callee is undeclared and may be one.
    pub fn may_be_variadic(&self, callee: &str, count: usize) -> bool {
        self.variadic.contains(callee)
            || !self
                .params
                .get(callee)
                .is_some_and(|types| types.len() == count)
    }

    /// Index of the first argument the convention treats as variadic, if any.
    pub fn fixed_arguments(&self, callee: &str) -> Option<usize> {
        self.variadic
            .contains(callee)
            .then(|| self.params.get(callee).map_or(0, Vec::len))
    }

    pub fn arguments(&self, callee: &str, args: &[Operand]) -> Vec<MirType> {
        argument_types(self.params.get(callee), self.fixed_arguments(callee), args)
    }

    pub fn float_result(&self, callee: &str) -> Option<MirType> {
        self.returns.get(callee).copied().filter(MirType::is_float)
    }
}

/// Types a call's arguments travel with. A matching prototype gives them. Past the
/// fixed parameters of a variadic callee, and for a callee with no prototype, the
/// operand's register class does, the way C promotes a variadic argument.
pub fn argument_types(
    declared: Option<&Vec<MirType>>,
    fixed: Option<usize>,
    args: &[Operand],
) -> Vec<MirType> {
    if fixed.is_none()
        && let Some(types) = declared.filter(|types| types.len() == args.len())
    {
        return types.clone();
    }
    let fixed = fixed.unwrap_or(0);
    args.iter()
        .enumerate()
        .map(|(index, arg)| match declared {
            Some(types) if index < fixed && index < types.len() => types[index],
            _ => promoted_argument_type(arg),
        })
        .collect()
}

/// Type a variadic argument reaches the callee as. A float travels as a double, the
/// way C promotes one, and everything else as a word.
pub fn promoted_argument_type(arg: &Operand) -> MirType {
    let float = match arg {
        Operand::Register(Register::Virtual(vreg)) => vreg.class == RegisterClass::Fpr,
        Operand::Immediate(Immediate::F32(_) | Immediate::F64(_)) => true,
        _ => false,
    };
    if float {
        MirType::Scalar(ScalarType::F64)
    } else {
        MirType::Scalar(ScalarType::I64)
    }
}

/// External names in a fixed order. The module holds them in a hash set, which would
/// otherwise put the declarations in a different order on every run.
pub fn external_names(module: &MirModule) -> Vec<&str> {
    let mut names: Vec<&str> = module
        .external_functions
        .iter()
        .map(String::as_str)
        .collect();
    names.sort_unstable();
    names
}

/// Wraps a failed write to the assembly buffer. Every backend writes through
/// `std::io::Write`, where the only realistic failure is an allocation one.
pub fn io_codegen_error(error: impl Display) -> CodegenError {
    CodegenError::InvalidCodegenOptions(format!("IO error: {error}"))
}

/// Rounds `value` up to the next multiple of `align`, which must be a power of two.
pub fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// Convert LaminaError to CodegenError with consistent error type.
pub fn lamina_to_codegen_error(err: LaminaError) -> CodegenError {
    match err {
        LaminaError::InternalError(msg) => {
            CodegenError::InvalidCodegenOptions(format!("Internal error: {msg}"))
        }
        LaminaError::CodegenError(inner) => inner,
        LaminaError::ParsingError(msg)
        | LaminaError::ValidationError(msg)
        | LaminaError::MirError(msg)
        | LaminaError::IoError(msg)
        | LaminaError::Utf8Error(msg)
        | LaminaError::RuntimeError(msg) => CodegenError::InvalidCodegenOptions(msg),
    }
}

/// Macro to implement common backend methods that delegate to CodegenBase.
///
/// This reduces duplication across backends. All backends have the same
/// `new()`, `set_module()`, and `drain_output()` implementations.
#[macro_export]
macro_rules! impl_backend_boilerplate {
    ($backend:ty) => {
        impl<'a> $backend {
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
        }
    };
}

/// Macro to implement common Codegen trait methods that delegate to CodegenBase.
///
/// This reduces duplication in `prepare()`, `compile()`, and `finalize()` methods
/// across all backends.
#[macro_export]
macro_rules! impl_codegen_trait_methods {
    ($backend:ty) => {
        fn prepare(
            &mut self,
            types: &std::collections::HashMap<String, $crate::mir::MirType>,
            globals: &std::collections::HashMap<String, $crate::mir::Global>,
            funcs: &std::collections::HashMap<String, $crate::mir::Signature>,
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
    };
}

#[cfg(test)]
mod tests {
    use super::assign_stack_slots;
    use lamina_mir::{
        block::Block,
        function::{Function, Signature},
        instruction::Instruction,
        register::{Register, RegisterClass, VirtualReg},
    };

    fn vreg(id: u32) -> VirtualReg {
        VirtualReg {
            id,
            class: RegisterClass::Gpr,
        }
    }

    fn frame_with_def_then_slot() -> Function {
        let mut func = Function::new(Signature::new("f"));
        let mut block = Block::new("entry");
        block.push(Instruction::StackAddr {
            dst: Register::Virtual(vreg(0)),
            slot: vreg(5),
        });
        func.add_block(block);
        func
    }

    #[test]
    fn aligned_slot_is_placed_on_its_alignment() {
        let mut func = frame_with_def_then_slot();
        func.stack_alignments.insert(vreg(0), 16);

        let (slots, _, total) = assign_stack_slots(&func, |i| -((i as i32) + 1) * 8);

        assert_eq!(slots[&vreg(0)], -16);
        assert_eq!(slots[&vreg(0)] % 16, 0);
        assert_eq!(slots[&vreg(5)], -24);
        assert_eq!(total, 3);
    }

    #[test]
    fn unaligned_slot_placement_is_unchanged() {
        let func = frame_with_def_then_slot();

        let (slots, _, total) = assign_stack_slots(&func, |i| -((i as i32) + 1) * 8);

        assert_eq!(slots[&vreg(0)], -8);
        assert_eq!(slots[&vreg(5)], -16);
        assert_eq!(total, 2);
    }

    /// A stack allocation wider than one slot must not share space with the
    /// registers assigned after it.
    #[test]
    fn reserved_slots_are_not_handed_to_other_registers() {
        let mut func = Function::new(Signature::new("f"));
        let mut block = Block::new("entry");
        block.push(Instruction::StackAddr {
            dst: Register::Virtual(vreg(1)),
            slot: vreg(0),
        });
        func.add_block(block);
        func.stack_reservations.insert(vreg(0), 4);

        let (slots, _, total) = assign_stack_slots(&func, |i| -((i as i32) + 1) * 8);

        assert_eq!(total, 6);
        let base = slots[&vreg(0)];
        let other = slots[&vreg(1)];
        // vreg 0 owns 5 slots ending at its own address; nothing else may land
        // inside base..base + 5 * 8.
        assert!(other <= base - 8 || other >= base + 40);
    }
}

fn dying_input(inst: &Instruction) -> Option<(VirtualReg, VirtualReg)> {
    match inst {
        Instruction::IntBinary {
            op: IntBinOp::Add,
            dst: Register::Virtual(dst),
            lhs: Operand::Register(Register::Virtual(src)),
            ..
        }
        | Instruction::Copy {
            dst: Register::Virtual(dst),
            src: Operand::Register(Register::Virtual(src)),
            ..
        } if dst != src => Some((*dst, *src)),
        Instruction::IntConvert {
            from,
            to,
            dst: Register::Virtual(dst),
            src: Operand::Register(Register::Virtual(src)),
            ..
        } if dst != src
            && from.size_bytes(PointerWidth::Bits64) == 8
            && to.size_bytes(PointerWidth::Bits64) == 8 =>
        {
            Some((*dst, *src))
        }
        _ => None,
    }
}

fn virtual_register(reg: &Register) -> Option<VirtualReg> {
    match reg {
        Register::Virtual(reg) => Some(*reg),
        _ => None,
    }
}

/// Instruction positions at which each key is clobbered, in ascending order.
#[derive(Default)]
pub(super) struct ClobberIndex<K> {
    positions: HashMap<K, Vec<usize>>,
}

impl<K: Copy + Eq + Hash> ClobberIndex<K> {
    /// Record that `key` is clobbered at `index`. Calls must come in ascending index order.
    pub(super) fn mark(&mut self, key: K, index: usize) {
        let list = self.positions.entry(key).or_default();
        if list.last() != Some(&index) {
            list.push(index);
        }
    }

    /// Whether `key` is clobbered nowhere in the closed range `[interval.start, interval.end]`.
    pub(super) fn is_clear(&self, key: K, interval: &LiveInterval) -> bool {
        self.positions.get(&key).is_none_or(|list| {
            let next = list.partition_point(|&index| index < interval.start);
            list.get(next).is_none_or(|&index| index > interval.end)
        })
    }
}

pub(super) fn block_live_out(func: &Function) -> Vec<HashSet<VirtualReg>> {
    let labels: HashMap<_, _> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.label.as_str(), index))
        .collect();
    let mut live_in = vec![HashSet::new(); func.blocks.len()];
    let mut live_out = live_in.clone();
    loop {
        let mut changed = false;
        for (index, block) in func.blocks.iter().enumerate().rev() {
            let out: HashSet<_> = block
                .successors()
                .iter()
                .filter_map(|label| labels.get(label))
                .flat_map(|&next| live_in[next].iter().copied())
                .collect();
            let mut input = out.clone();
            for inst in block.instructions.iter().rev() {
                if let Some(dst) = inst.def_reg().and_then(virtual_register) {
                    input.remove(&dst);
                }
                input.extend(inst.use_regs().into_iter().filter_map(virtual_register));
            }
            changed |= input != live_in[index] || out != live_out[index];
            live_in[index] = input;
            live_out[index] = out;
        }
        if !changed {
            break;
        }
    }
    live_out
}

pub(super) fn coalesce_dying_inputs<R: Copy + Eq>(
    func: &Function,
    intervals: &[LiveInterval],
    original_plan: &HashMap<VirtualReg, Allocation<R>>,
    allowed: impl Fn(&LiveInterval, R) -> bool,
) -> HashMap<VirtualReg, Allocation<R>> {
    let mut plan = original_plan.clone();
    let virtual_reg = |reg: &Register| match reg {
        Register::Virtual(v) => Some(*v),
        _ => None,
    };
    let mut affinities = HashSet::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        match &inst.instr {
            Instruction::Copy {
                dst,
                src: Operand::Register(src),
                ..
            } => {
                if let (Some(dst), Some(src)) = (virtual_reg(dst), virtual_reg(src)) {
                    affinities.insert((src, dst));
                }
            }
            Instruction::TailCall { name, args } if name == &func.sig.name => {
                for (arg, param) in args.iter().zip(&func.sig.params) {
                    if let Operand::Register(arg) = arg
                        && let (Some(src), Some(dst)) = (virtual_reg(arg), virtual_reg(&param.reg))
                    {
                        affinities.insert((src, dst));
                    }
                }
            }
            _ => {}
        }
    }
    let live_out = block_live_out(func);
    let mut definitions = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        if let Some(dst) = inst.def_reg().and_then(virtual_reg) {
            *definitions.entry(dst).or_insert(0) += 1;
        }
    }
    let mut candidates = Vec::new();
    let mut block_start = func
        .blocks
        .iter()
        .map(|b| b.instructions.len())
        .sum::<usize>();
    for (index, block) in func.blocks.iter().enumerate().rev() {
        block_start -= block.instructions.len();
        let mut live = live_out[index].clone();
        let mut block_candidates = Vec::new();
        for (offset, inst) in block.instructions.iter().enumerate().rev() {
            let dst = inst.def_reg().and_then(virtual_reg);
            if let Some((dst, src)) = dying_input(inst)
                && !(matches!(&inst.instr, Instruction::IntBinary { .. })
                    && !affinities.contains(&(dst, src)))
                && !live.contains(&src)
                && definitions.get(&dst) == Some(&1)
                && !func
                    .sig
                    .params
                    .iter()
                    .any(|param| virtual_reg(&param.reg) == Some(dst))
            {
                block_candidates.push((block_start + offset, dst, src));
            }
            if let Some(dst) = dst {
                live.remove(&dst);
            }
            live.extend(inst.use_regs().into_iter().filter_map(virtual_reg));
        }
        candidates.extend(block_candidates);
    }
    candidates.reverse();
    if candidates.is_empty() {
        return plan;
    }

    let mut points = Vec::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let mut live = live_out[index].clone();
        let mut block_points = Vec::new();
        for inst in block.instructions.iter().rev() {
            let after = live.clone();
            let dst = inst.def_reg().and_then(virtual_reg);
            if let Some(dst) = dst {
                live.remove(&dst);
            }
            live.extend(inst.use_regs().into_iter().filter_map(virtual_reg));
            block_points.push((inst, live.clone(), after, dst));
        }
        points.extend(block_points.into_iter().rev());
    }
    let by_reg: HashMap<_, _> = intervals
        .iter()
        .map(|interval| (interval.vreg, interval))
        .collect();
    let mut touching: HashMap<VirtualReg, Vec<usize>> = HashMap::new();
    for (index, (_, before, after, defined)) in points.iter().enumerate() {
        for vreg in before.iter().chain(after).chain(defined.iter()) {
            let list = touching.entry(*vreg).or_default();
            if list.last() != Some(&index) {
                list.push(index);
            }
        }
    }
    for (_, dst, src) in candidates {
        let (Some(output), Some(Allocation::Register(reg))) =
            (by_reg.get(&dst), plan.get(&src).copied())
        else {
            continue;
        };
        if !allowed(output, reg) {
            continue;
        }
        let held = |other: &VirtualReg| {
            *other != dst && plan.get(other) == Some(&Allocation::Register(reg))
        };
        let conflicts = touching.get(&dst).is_some_and(|indices| {
            indices.iter().any(|&index| {
                let (inst, before, after, defined) = &points[index];
                let dying = dying_input(inst);
                let dst_before = before.contains(&dst);
                (dst_before && before.iter().any(held))
                    || (after.contains(&dst) && after.iter().any(held))
                    || (*defined == Some(dst)
                        && before
                            .iter()
                            .any(|other| held(other) && dying != Some((dst, *other))))
                    || (dst_before
                        && defined.is_some_and(|other| held(&other) && dying != Some((other, dst))))
            })
        });
        if conflicts {
            continue;
        }
        plan.insert(dst, Allocation::Register(reg));
    }
    plan
}

pub(super) struct NaturalLoop {
    header: usize,
    latch: usize,
    body: HashSet<usize>,
}

fn loop_successors(func: &Function, index: usize) -> Vec<&str> {
    match func.blocks[index].terminator() {
        Some(Instruction::TailCall { name, .. }) if name == &func.sig.name => vec![&func.entry],
        _ => func.blocks[index].successors(),
    }
}

pub(super) fn natural_loops(func: &Function) -> Vec<NaturalLoop> {
    let index_of: HashMap<&str, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.label.as_str(), i))
        .collect();
    let Some(&entry) = index_of.get(func.entry.as_str()) else {
        return Vec::new();
    };

    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
    for index in 0..func.blocks.len() {
        for successor in loop_successors(func, index) {
            if let Some(&target) = index_of.get(successor) {
                predecessors[target].push(index);
            }
        }
    }

    let mut back_edges: Vec<(usize, usize)> = Vec::new();
    let mut on_path = vec![false; func.blocks.len()];
    let mut visited = vec![false; func.blocks.len()];
    let mut stack = vec![(entry, 0usize)];
    visited[entry] = true;
    on_path[entry] = true;
    while let Some(&mut (node, ref mut next)) = stack.last_mut() {
        let successors = loop_successors(func, node);
        if *next < successors.len() {
            let edge = successors[*next];
            *next += 1;
            if let Some(&target) = index_of.get(edge) {
                if on_path[target] {
                    back_edges.push((node, target));
                } else if !visited[target] {
                    visited[target] = true;
                    on_path[target] = true;
                    stack.push((target, 0));
                }
            }
        } else {
            on_path[node] = false;
            stack.pop();
        }
    }

    let mut loops = Vec::new();
    for (latch, header) in back_edges {
        let mut body = HashSet::from([header]);
        let mut work = vec![latch];
        while let Some(node) = work.pop() {
            if !body.insert(node) {
                continue;
            }
            work.extend(predecessors[node].iter().copied());
        }
        loops.push(NaturalLoop {
            header,
            latch,
            body,
        });
    }
    loops
}

pub(super) fn aligned_loop_headers(
    func: &Function,
    loops: &[NaturalLoop],
    min_instructions: usize,
) -> HashSet<usize> {
    loops
        .iter()
        .filter(|natural_loop| {
            natural_loop
                .body
                .iter()
                .map(|&b| func.blocks[b].instructions.len())
                .sum::<usize>()
                >= min_instructions
        })
        .map(|natural_loop| natural_loop.header)
        .collect()
}

pub(super) fn rotate_loops(
    func: &Function,
    loops: &[NaturalLoop],
    layout: &mut Vec<usize>,
) -> HashMap<usize, usize> {
    let mut rotations = HashMap::new();
    let mut ordered: Vec<_> = loops.iter().collect();
    ordered.sort_by_key(|natural_loop| natural_loop.body.len());
    for natural_loop in ordered {
        let NaturalLoop {
            header,
            latch,
            body,
        } = natural_loop;
        if rotations.contains_key(header) || header == latch {
            continue;
        }
        let instructions = &func.blocks[*header].instructions;
        let [cmp, br] = instructions.as_slice() else {
            continue;
        };
        let (
            Instruction::IntCmp { dst, .. },
            Instruction::Br {
                cond,
                true_target,
                false_target,
            },
        ) = (&cmp.instr, &br.instr)
        else {
            continue;
        };
        if dst != cond
            || func
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .flat_map(|i| i.use_regs())
                .filter(|reg| *reg == dst)
                .count()
                != 1
        {
            continue;
        }
        let inside: Vec<_> = [true_target, false_target]
            .into_iter()
            .filter_map(|label| {
                func.blocks
                    .iter()
                    .position(|block| block.label == *label)
                    .filter(|index| *index != *header && body.contains(index))
            })
            .collect();
        let [body_entry] = inside.as_slice() else {
            continue;
        };
        if !matches!(
            func.blocks[*latch].terminator(),
            Some(Instruction::Jmp { .. } | Instruction::TailCall { .. })
        ) {
            continue;
        }
        let Some(start) = layout.iter().position(|index| index == header) else {
            continue;
        };
        let Some(end) = layout.iter().position(|index| index == latch) else {
            continue;
        };
        if end <= start
            || layout.get(start + 1) != Some(body_entry)
            || layout[start + 1..=end]
                .iter()
                .any(|index| !body.contains(index))
        {
            continue;
        }
        layout.remove(start);
        layout.insert(end, *header);
        rotations.insert(*header, *body_entry);
    }
    rotations
}

pub(super) fn block_layout(func: &Function) -> Vec<usize> {
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
            for label in successors.iter().rev().skip(1) {
                if let Some(&next) = index_of.get(label)
                    && !placed[next]
                {
                    queue.push_back(next);
                }
            }
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

pub(super) fn self_tail_entry_label(func_name: &str) -> String {
    format!(".L_{func_name}.tailentry")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CopyLocation<T> {
    Value(T),
    Scratch,
}

pub(super) fn parallel_copy_schedule<T: Clone + PartialEq>(
    pairs: &[(T, T)],
) -> Vec<(CopyLocation<T>, CopyLocation<T>)> {
    use CopyLocation::{Scratch, Value};
    let mut pending: Vec<_> = pairs
        .iter()
        .filter(|(src, dst)| src != dst)
        .map(|(src, dst)| (Value(src.clone()), Value(dst.clone())))
        .collect();
    let mut moves = Vec::new();
    while !pending.is_empty() {
        if let Some(index) = pending
            .iter()
            .position(|(_, dst)| !pending.iter().any(|(src, _)| src == dst))
        {
            moves.push(pending.remove(index));
        } else {
            let src = pending[0].0.clone();
            moves.push((src.clone(), Scratch));
            for (source, _) in &mut pending {
                if *source == src {
                    *source = Scratch;
                }
            }
        }
    }
    moves
}

pub(super) fn use_counts(func: &Function) -> HashMap<VirtualReg, usize> {
    let mut counts = HashMap::new();
    for inst in func.blocks.iter().flat_map(|block| &block.instructions) {
        for reg in inst.use_regs() {
            if let Register::Virtual(reg) = reg {
                *counts.entry(*reg).or_insert(0) += 1;
            }
        }
    }
    counts
}

pub(super) fn branch_comparisons(
    func: &Function,
    counts: &HashMap<VirtualReg, usize>,
) -> HashSet<(usize, usize)> {
    func.blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .instructions
                .windows(2)
                .enumerate()
                .filter_map(move |(i, pair)| match (&pair[0].instr, &pair[1].instr) {
                    (
                        Instruction::IntCmp {
                            dst: Register::Virtual(dst),
                            ..
                        },
                        Instruction::Br {
                            cond: Register::Virtual(cond),
                            ..
                        },
                    ) if dst == cond && counts.get(dst) == Some(&1) => Some((b, i)),
                    _ => None,
                })
        })
        .collect()
}

pub(super) struct FunctionEmissionPlan {
    pub order: Vec<usize>,
    pub aligned_headers: HashSet<usize>,
    pub rotations: HashMap<usize, usize>,
    pub comparisons: HashSet<(usize, usize)>,
}

impl FunctionEmissionPlan {
    pub fn new(func: &Function) -> Self {
        let loops = natural_loops(func);
        let mut order = block_layout(func);
        let rotations = rotate_loops(func, &loops, &mut order);
        let aligned_headers = aligned_loop_headers(func, &loops, 2)
            .into_iter()
            .map(|header| rotations.get(&header).copied().unwrap_or(header))
            .collect();
        let comparisons = branch_comparisons(func, &use_counts(func));
        Self {
            order,
            aligned_headers,
            rotations,
            comparisons,
        }
    }

    pub fn branch<'a>(
        &self,
        func: &'a Function,
        block: usize,
        next: &str,
        true_target: &'a str,
        false_target: &'a str,
    ) -> (bool, &'a str, Option<&'a str>) {
        let invert = true_target == next
            || self
                .rotations
                .get(&block)
                .is_some_and(|&body| func.blocks[body].label == false_target);
        let (taken, fallthrough) = if invert {
            (false_target, true_target)
        } else {
            (true_target, false_target)
        };
        (invert, taken, (fallthrough != next).then_some(fallthrough))
    }
}

pub(super) fn self_tail_arguments(
    func: &Function,
    inst: &Instruction,
) -> Option<Vec<(Operand, Register)>> {
    match inst {
        Instruction::TailCall { name, args }
            if name == &func.sig.name && args.len() == func.sig.params.len() =>
        {
            Some(
                args.iter()
                    .cloned()
                    .zip(func.sig.params.iter().map(|p| p.reg.clone()))
                    .collect(),
            )
        }
        _ => None,
    }
}

pub(super) fn self_tail_schedule(
    pairs: &[(Operand, Register)],
    map: impl Fn(&Register) -> Register,
) -> Vec<(CopyLocation<Operand>, CopyLocation<Operand>)> {
    let pairs: Vec<_> = pairs
        .iter()
        .map(|(src, dst)| {
            let src = match src {
                Operand::Register(reg) => Operand::Register(map(reg)),
                _ => src.clone(),
            };
            (src, Operand::Register(map(dst)))
        })
        .collect();
    parallel_copy_schedule(&pairs)
}

#[cfg(test)]
mod copy_schedule_tests {
    use super::{CopyLocation, parallel_copy_schedule};

    #[test]
    fn schedules_preserve_simultaneous_values() {
        for encoded in 0..625 {
            let mut choices = encoded;
            let mut pairs = Vec::new();
            for dst in 0..4 {
                let src = choices % 5;
                choices /= 5;
                pairs.push((src, dst));
            }
            let mut values = [0, 1, 2, 3, 4];
            let mut scratch = None;
            for (src, dst) in parallel_copy_schedule(&pairs) {
                let value = match src {
                    CopyLocation::Value(src) => values[src],
                    CopyLocation::Scratch => scratch.unwrap(),
                };
                match dst {
                    CopyLocation::Value(dst) => values[dst] = value,
                    CopyLocation::Scratch => scratch = Some(value),
                }
            }
            for (src, dst) in pairs {
                assert_eq!(values[dst], src);
            }
        }
        assert!(parallel_copy_schedule(&[(1, 1), (2, 2)]).is_empty());
    }
}

#[cfg(test)]
mod analysis_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::mir::{FunctionBuilder, Instruction as MirInst, IntCmpOp};
    use lamina_codegen::{Allocation as MirAllocation, LinearScanAllocator};
    #[test]
    fn rotation_preserves_nested_loop_alignment() {
        let condition = Register::Virtual(VirtualReg::gpr(0));
        let comparison = |dst| MirInst::IntCmp {
            op: IntCmpOp::SGt,
            dst,
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(0)),
            ty: MirType::Scalar(ScalarType::I64),
        };
        let mut builder = FunctionBuilder::new("nested")
            .block("entry")
            .instr(MirInst::Jmp {
                target: "outer".into(),
            })
            .block("outer")
            .instr(comparison(condition.clone()))
            .instr(MirInst::Br {
                cond: condition.clone(),
                true_target: "exit".into(),
                false_target: "inner_entry".into(),
            })
            .block("inner_entry")
            .instr(MirInst::Jmp {
                target: "inner".into(),
            })
            .block("inner")
            .instr(comparison(VirtualReg::gpr(2).into()))
            .instr(MirInst::Br {
                cond: VirtualReg::gpr(2).into(),
                true_target: "outer_latch".into(),
                false_target: "body".into(),
            })
            .block("body");
        for _ in 0..24 {
            builder = builder.instr(MirInst::Copy {
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::Immediate(Immediate::I64(0)),
            });
        }
        let func = builder
            .instr(MirInst::Jmp {
                target: "inner".into(),
            })
            .block("outer_latch")
            .instr(MirInst::Jmp {
                target: "outer".into(),
            })
            .block("exit")
            .instr(MirInst::Ret { value: None })
            .build();
        let loops = natural_loops(&func);
        let aligned = aligned_loop_headers(&func, &loops, 24);
        let labels = |set: &HashSet<usize>| {
            set.iter()
                .map(|&index| func.blocks[index].label.as_str())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(labels(&aligned), BTreeSet::from(["inner", "outer"]));
        let mut layout = block_layout(&func);
        let rotations = rotate_loops(&func, &loops, &mut layout);
        assert_eq!(rotations.len(), 1);
        assert_eq!(
            layout
                .iter()
                .map(|&i| func.blocks[i].label.as_str())
                .collect::<Vec<_>>(),
            [
                "entry",
                "outer",
                "inner_entry",
                "body",
                "inner",
                "exit",
                "outer_latch"
            ]
        );
        assert_eq!(
            aligned_loop_headers(&func, &natural_loops(&func), 24),
            aligned
        );
        let aligned_after = aligned
            .iter()
            .map(|header| rotations.get(header).copied().unwrap_or(*header))
            .collect();
        assert_eq!(labels(&aligned_after), BTreeSet::from(["body", "outer"]));
    }

    #[test]
    fn dying_input_coalescing_checks_cfg_liveness_and_constraints() {
        let a = VirtualReg::gpr(0);
        let b = VirtualReg::gpr(1);
        let c = VirtualReg::gpr(2);
        for (live_source, conflict, permitted, expected) in [
            (false, false, true, "r8"),
            (true, false, true, "r9"),
            (false, true, true, "r9"),
            (false, false, false, "r9"),
        ] {
            let mut func = FunctionBuilder::new("increment")
                .param(a.into(), MirType::Scalar(ScalarType::I64))
                .block("body")
                .instr(MirInst::IntBinary {
                    op: IntBinOp::Add,
                    dst: b.into(),
                    lhs: Operand::Register(a.into()),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                    ty: MirType::Scalar(ScalarType::I64),
                })
                .instr(MirInst::TailCall {
                    name: "increment".into(),
                    args: [Some(b), live_source.then_some(a), conflict.then_some(c)]
                        .into_iter()
                        .flatten()
                        .map(|v| Operand::Register(v.into()))
                        .collect(),
                })
                .block("entry")
                .instr(MirInst::Copy {
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: c.into(),
                    src: Operand::Register(a.into()),
                })
                .instr(MirInst::Jmp {
                    target: "body".into(),
                })
                .build();
            func.entry = "entry".into();
            let intervals = LinearScanAllocator::compute_intervals(&func);
            let mut plan = HashMap::from([
                (a, MirAllocation::Register("r8")),
                (b, MirAllocation::Register("r9")),
                (
                    c,
                    MirAllocation::Register(if conflict { "r8" } else { "r10" }),
                ),
            ]);
            plan = coalesce_dying_inputs(&func, &intervals, &plan, |_, _| permitted);
            assert_eq!(plan[&b], MirAllocation::Register(expected));
        }
    }
}

#[cfg(test)]
mod backend_plan_tests {
    use super::*;
    use crate::mir::{FunctionBuilder, IntCmpOp};
    use crate::mir_codegen::{
        MirCodegenSettings, RegallocStrategy, generate_mir_to_target_with_settings,
    };
    use lamina_platform::{Target, TargetArchitecture};
    use std::error::Error;

    #[test]
    fn native_backends_rotate_self_tail_loops_and_schedule_swaps() -> Result<(), Box<dyn Error>> {
        let a = VirtualReg::gpr(0);
        let b = VirtualReg::gpr(1);
        let cond = VirtualReg::gpr(2);
        let func = FunctionBuilder::new("swap")
            .param(a.into(), MirType::Scalar(ScalarType::I64))
            .param(b.into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SGt,
                dst: cond.into(),
                lhs: Operand::Register(a.into()),
                rhs: Operand::Register(b.into()),
                ty: MirType::Scalar(ScalarType::I64),
            })
            .instr(Instruction::Br {
                cond: cond.into(),
                true_target: "exit".into(),
                false_target: "body".into(),
            })
            .block("body")
            .instr(Instruction::TailCall {
                name: "swap".into(),
                args: vec![Operand::Register(b.into()), Operand::Register(a.into())],
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(a.into())),
            })
            .build();
        let mut module = MirModule::new("swap_test");
        module.add_function(func);
        for (arch, save, restore, branch, materialize, external) in [
            (
                TargetArchitecture::Aarch64,
                "mov x17, x16",
                "mov x16, x17",
                "b.le .L_swap.body",
                "cset ",
                "b swap",
            ),
            (
                TargetArchitecture::Riscv64,
                "mv a2, a0",
                "mv a0, a2",
                "bge a1, a0, .L_swap.body",
                "slt ",
                "tail swap",
            ),
            (
                TargetArchitecture::PowerPC64,
                "mr 0, 3",
                "mr 3, 0",
                "b .L_swap.body",
                "cmp0.true",
                "b swap",
            ),
        ] {
            for regalloc in [
                RegallocStrategy::Incremental,
                RegallocStrategy::LinearScanGlobal,
                RegallocStrategy::GraphColorGlobal,
            ] {
                let settings = MirCodegenSettings {
                    regalloc,
                    ..Default::default()
                };
                let mut output = Vec::new();
                generate_mir_to_target_with_settings(
                    &module,
                    &mut output,
                    &Target::new(arch, TargetOperatingSystem::Linux),
                    1,
                    &settings,
                )?;
                let asm = String::from_utf8(output)?;
                assert!(
                    asm.contains(save) && asm.contains(restore),
                    "{arch:?} {regalloc:?}: {asm}"
                );
                assert!(asm.contains(branch), "{arch:?} {regalloc:?}: {asm}");
                assert!(
                    !asm.contains(materialize) && !asm.contains(external),
                    "{arch:?} {regalloc:?}: {asm}"
                );
                assert!(
                    asm.find(".L_swap.body:") < asm.find(".L_swap.entry:"),
                    "{asm}"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod export_linkage_tests {
    use super::{reject_thread_local_globals, reject_underaligned_accesses};
    #[cfg(feature = "nightly")]
    use crate::mir_codegen::arx64::generate_mir_arx64;
    use crate::mir_codegen::{MirCodegenSettings, generate_mir_to_target_with_settings};
    use crate::{
        error::LaminaError,
        mir::{Module as MirModule, codegen::from_ir_x86_64},
        mir_codegen::{
            arm::aarch64::generate_mir_aarch64, powerpc::powerpc64::generate_mir_ppc64,
            riscv::generate_mir_riscv, x86_64::generate_mir_x86_64,
        },
        parser::parse_module,
    };
    use lamina_platform::PointerWidth;
    use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};

    const SOURCE: &str = r#"
@export fn @answer() -> i64 {
entry:
 ret.i64 42
}
fn @helper() -> i64 {
entry:
 ret.i64 7
}
fn @main() -> i64 {
entry:
 %a = call @answer()
 %b = call @helper()
 %c = add.i64 %a, %b
 ret.i64 %c
}
"#;

    type Emitter = fn(&MirModule, &mut Vec<u8>, TargetOperatingSystem) -> Result<(), LaminaError>;

    #[test]
    fn only_exported_functions_and_main_are_global() {
        let ir = parse_module(SOURCE).expect("parse export module");
        let mir = from_ir_x86_64(&ir, "exports").expect("lower export module");
        #[cfg_attr(not(feature = "nightly"), allow(unused_mut))]
        let mut emitters: Vec<(&str, Emitter, TargetOperatingSystem)> = vec![
            (
                "x86_64",
                generate_mir_x86_64::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "aarch64",
                |module, writer, os| generate_mir_aarch64(module, writer, os, PointerWidth::Bits64),
                TargetOperatingSystem::Linux,
            ),
            (
                "riscv",
                generate_mir_riscv::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "ppc64le",
                generate_mir_ppc64::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
        ];
        #[cfg(feature = "nightly")]
        emitters.push((
            "arx64",
            generate_mir_arx64::<Vec<u8>>,
            TargetOperatingSystem::Unknown,
        ));
        for (name, emit, os) in emitters {
            let mut output = Vec::new();
            emit(&mir, &mut output, os).expect("emit module");
            let text = String::from_utf8(output).expect("utf8 assembly");
            assert!(text.contains(".globl answer\n"), "{name}: {text}");
            assert!(text.contains(".globl main\n"), "{name}: {text}");
            assert!(!text.contains(".globl helper\n"), "{name}: {text}");
        }
    }

    const VISIBILITY_SOURCE: &str = r#"
@hidden fn @hidden_fn() -> i64 {
entry:
 ret.i64 1
}
@protected fn @protected_fn() -> i64 {
entry:
 ret.i64 2
}
@internal fn @internal_fn() -> i64 {
entry:
 ret.i64 3
}
@private fn @private_fn() -> i64 {
entry:
 ret.i64 4
}
@cold fn @cold_fn() -> i64 {
entry:
 ret.i64 5
}
@hot fn @hot_fn() -> i64 {
entry:
 ret.i64 6
}
fn @main() -> i64 {
entry:
 %a = call @hidden_fn()
 %b = call @protected_fn()
 %c = call @internal_fn()
 %d = call @private_fn()
 %e = call @cold_fn()
 %f = call @hot_fn()
 ret.i64 %a
}
"#;

    fn emit_visibility_module(os: TargetOperatingSystem) -> String {
        let ir = parse_module(VISIBILITY_SOURCE).expect("parse visibility module");
        let mir = from_ir_x86_64(&ir, "visibility").expect("lower visibility module");
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, os).expect("emit visibility module");
        String::from_utf8(output).expect("utf8 assembly")
    }

    #[test]
    fn elf_visibility_annotations_emit_matching_directives() {
        let text = emit_visibility_module(TargetOperatingSystem::Linux);
        assert!(text.contains(".hidden hidden_fn\n"), "{text}");
        assert!(text.contains(".globl hidden_fn\n"), "{text}");
        assert!(text.contains(".protected protected_fn\n"), "{text}");
        assert!(text.contains(".globl protected_fn\n"), "{text}");
        assert!(text.contains(".internal internal_fn\n"), "{text}");
        assert!(!text.contains(".globl internal_fn\n"), "{text}");
        assert!(text.contains(".local private_fn\n"), "{text}");
        assert!(!text.contains(".globl private_fn\n"), "{text}");
        assert!(!text.contains(".hidden main"), "{text}");
    }

    #[test]
    fn macho_visibility_uses_private_extern_and_skips_protected() {
        let text = emit_visibility_module(TargetOperatingSystem::MacOS);
        assert!(text.contains(".private_extern _hidden_fn\n"), "{text}");
        assert!(text.contains(".private_extern _internal_fn\n"), "{text}");
        assert!(text.contains(".private_extern _private_fn\n"), "{text}");
        assert!(!text.contains(".protected"), "{text}");
        assert!(!text.contains(".hidden"), "{text}");
        assert!(!text.contains(".internal"), "{text}");
    }

    #[test]
    fn coff_targets_emit_no_elf_visibility_directives() {
        let text = emit_visibility_module(TargetOperatingSystem::Windows);
        for directive in [
            ".hidden",
            ".protected",
            ".internal",
            ".local",
            ".private_extern",
        ] {
            assert!(!text.contains(directive), "{directive}: {text}");
        }
    }

    #[test]
    fn cold_and_hot_functions_get_their_own_text_section() {
        let text = emit_visibility_module(TargetOperatingSystem::Linux);
        let cold_section = ".section .text.unlikely.cold_fn,\"ax\",@progbits\n";
        let cold_label = "cold_fn:\n";
        let start = text.find(cold_section).expect(&text);
        let label = text.find(cold_label).expect(&text);
        assert!(start < label, "{text}");
        let resumed = text[label..].find("    .text\n").expect(&text) + label;
        assert!(resumed > label, "{text}");
        assert!(
            text.contains(".section .text.hot.hot_fn,\"ax\",@progbits\n"),
            "{text}"
        );
        assert!(!text.contains(".section .text.unlikely.main"), "{text}");
    }

    #[test]
    fn cold_and_hot_sections_are_not_emitted_for_macho() {
        let text = emit_visibility_module(TargetOperatingSystem::MacOS);
        assert!(!text.contains(".text.unlikely"), "{text}");
        assert!(!text.contains(".text.hot"), "{text}");
    }

    #[test]
    fn backends_without_global_support_reject_modules_with_globals() {
        let ir = parse_module(
            "global $g: i64 = 1\nfn @main() -> i64 {\nentry:\n %x = load.i64 $g\n ret.i64 %x\n}\n",
        )
        .expect("parse global module");
        let mir = from_ir_x86_64(&ir, "globals").expect("lower global module");
        use crate::mir_codegen::{amdgpu, ptx};
        type GpuEmitter = fn(&MirModule, &mut Vec<u8>) -> Result<(), LaminaError>;
        let rejecting: [(&str, GpuEmitter); 2] = [
            ("amdgpu", amdgpu::emit::<Vec<u8>>),
            ("ptx", ptx::emit::<Vec<u8>>),
        ];
        for (name, emit) in rejecting {
            let mut output = Vec::new();
            let error = emit(&mir, &mut output).expect_err(name);
            assert!(
                error
                    .to_string()
                    .contains("does not yet support global variables"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn backends_without_tls_name_the_feature_they_refuse() {
        let ir = parse_module(
            "@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n %x = load.i64 $t\n ret.i64 %x\n}\n",
        )
        .expect("parse thread-local module");
        let mir = from_ir_x86_64(&ir, "tls").expect("lower thread-local module");
        for (name, backend) in [
            ("aarch64", "AArch64"),
            ("riscv64", "RISC-V"),
            ("powerpc64", "PowerPC64"),
        ] {
            let error = reject_thread_local_globals(&mir, backend).expect_err(name);
            let text = error.to_string();
            assert!(text.contains("thread-local storage"), "{name}: {text}");
            assert!(text.contains(backend), "{name}: {text}");
            assert!(text.contains("'t'"), "{name}: {text}");
        }
    }

    #[test]
    fn refusal_messages_name_every_backend_that_implements_the_feature() {
        use crate::mir_codegen::common::{
            unsupported_inline_asm, unsupported_overflow_flag, unsupported_thread_local,
        };
        for text in [
            unsupported_overflow_flag("AArch64").to_string(),
            unsupported_thread_local("AArch64").to_string(),
            unsupported_inline_asm("AArch64").to_string(),
        ] {
            assert!(text.contains("x86_64"), "{text}");
            assert!(text.contains("s390x"), "{text}");
            assert!(!text.contains("x86_64 only"), "{text}");
        }
    }

    #[test]
    fn every_backend_without_inline_asm_names_the_same_supported_set() {
        use crate::mir_codegen::common::unsupported_inline_asm;
        for backend in [
            "WASM",
            "armv7",
            "thumbv6m",
            "thumbv7m",
            "LoongArch64",
            "AArch64",
            "RISC-V",
            "mips64",
            "i686",
            "PowerPC64",
        ] {
            let text = unsupported_inline_asm(backend).to_string();
            assert!(text.contains(backend), "{text}");
            assert!(text.contains("x86_64, aarch64 and s390x"), "{text}");
            assert!(!text.contains("x86_64 Linux only"), "{text}");
        }
    }

    #[test]
    fn s390x_implements_what_the_refusal_messages_credit_it_with() {
        for source in [
            "fn @main() -> i64 {\nentry:\n %f = add.ovf.i64 1, 2\n %z = extend.bool.i64 %f\n ret.i64 %z\n}\n",
            "@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n %x = load.i64 $t\n ret.i64 %x\n}\n",
            "fn @main() -> i64 {\nentry:\n %r = asm.i64 \"lghi $0, 7\", out(r2)\n ret.i64 %r\n}\n",
        ] {
            let ir = parse_module(source).expect("parse s390x feature module");
            let mir = from_ir_x86_64(&ir, "s390x").expect("lower s390x feature module");
            let mut out = Vec::new();
            generate_mir_to_target_with_settings(
                &mir,
                &mut out,
                &Target::new(TargetArchitecture::S390x, TargetOperatingSystem::Linux),
                1,
                &MirCodegenSettings::default(),
            )
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        }
    }

    #[test]
    fn backends_needing_natural_alignment_name_the_access_they_refuse() {
        let ir = parse_module(
            "fn @main(ptr %p) -> i64 {\nentry:\n store.i32 %p, 1 align 1\n ret.i64 0\n}\n",
        )
        .expect("parse under-aligned module");
        let mir = from_ir_x86_64(&ir, "align").expect("lower under-aligned module");
        let error = reject_underaligned_accesses(&mir, PointerWidth::Bits64, "RISC-V")
            .expect_err("riscv faults on an unaligned store");
        let text = error.to_string();
        assert!(
            text.contains("RISC-V cannot issue a store of i32"),
            "{text}"
        );
        assert!(text.contains("alignment 1"), "{text}");

        let ir =
            parse_module("fn @main(ptr %p) -> i64 {\nentry:\n store.i32 %p, 1\n ret.i64 0\n}\n")
                .expect("parse natural module");
        let mir = from_ir_x86_64(&ir, "align").expect("lower natural module");
        reject_underaligned_accesses(&mir, PointerWidth::Bits64, "RISC-V")
            .expect("a derived alignment is never a refusal");
    }

    #[test]
    fn exported_globals_are_global_and_zero_storage_is_bss() {
        let ir = parse_module(
            r#"
@export global $shared: i64 = 3
global $hidden: i64
fn @main() -> i64 {
entry:
 %a = load.i64 $shared
 %b = load.i64 $hidden
 %c = add.i64 %a, %b
 ret.i64 %c
}
"#,
        )
        .expect("parse global module");
        let mir = from_ir_x86_64(&ir, "globals").expect("lower global module");
        let mut output = Vec::new();
        generate_mir_x86_64::<Vec<u8>>(&mir, &mut output, TargetOperatingSystem::Linux)
            .expect("emit global module");
        let text = String::from_utf8(output).expect("utf8 assembly");
        assert!(text.contains(".globl shared\n"), "{text}");
        assert!(!text.contains(".globl hidden\n"), "{text}");
        assert!(
            text.contains(".data\n.p2align 3\nshared:\n    .byte 0x03, 0x00"),
            "{text}"
        );
        assert!(
            text.contains(".bss\n.p2align 3\nhidden:\n    .zero 8\n"),
            "{text}"
        );
    }

    #[test]
    fn global_align_annotation_sets_the_emitted_boundary() {
        let ir = parse_module(
            "@align(64)\nglobal $arena: [256 x i8]\n\
             @align(16)\nglobal $tab: [2 x i64] = [1, 2]\n\
             fn @main() -> i64 {\nentry:\n ret.i64 0\n}\n",
        )
        .expect("parse aligned globals");
        let mir = from_ir_x86_64(&ir, "aligned").expect("lower aligned globals");
        let mut output = Vec::new();
        generate_mir_x86_64::<Vec<u8>>(&mir, &mut output, TargetOperatingSystem::Linux)
            .expect("emit aligned globals");
        let text = String::from_utf8(output).expect("utf8 assembly");
        assert!(
            text.contains(".bss\n.p2align 6\narena:\n    .zero 256"),
            "{text}"
        );
        assert!(text.contains(".data\n.p2align 4\ntab:"), "{text}");
    }

    #[test]
    fn unwritten_tail_of_initialized_global_is_zeroed() {
        let ir = parse_module(
            "global $g: [10000000000 x i8] = [1, 2]\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
        )
        .expect("parse sparse array global");
        let mir = from_ir_x86_64(&ir, "globals").expect("lower sparse array global");
        let mut output = Vec::new();
        generate_mir_x86_64::<Vec<u8>>(&mir, &mut output, TargetOperatingSystem::Linux)
            .expect("emit sparse array global");
        let text = String::from_utf8(output).expect("utf8 assembly");
        assert!(
            text.contains("g:\n    .byte 0x01, 0x02\n    .zero 9999999998\n"),
            "{text}"
        );
    }

    #[test]
    fn elf_targets_mark_stack_non_executable() {
        let ir = parse_module("fn @main() -> i64 {\nentry:\n ret.i64 0\n}\n").expect("parse");
        let mir = from_ir_x86_64(&ir, "stack_note").expect("lower");
        let note = ".section .note.GNU-stack,\"\",@progbits";
        for (os, expected) in [
            (TargetOperatingSystem::Linux, true),
            (TargetOperatingSystem::FreeBSD, true),
            (TargetOperatingSystem::MacOS, false),
            (TargetOperatingSystem::IOS, false),
            (TargetOperatingSystem::TvOS, false),
            (TargetOperatingSystem::WatchOS, false),
            (TargetOperatingSystem::VisionOS, false),
            (TargetOperatingSystem::Windows, false),
            (TargetOperatingSystem::Horizon, true),
            (TargetOperatingSystem::Orbis, true),
            (TargetOperatingSystem::Prospero, true),
        ] {
            let mut output = Vec::new();
            generate_mir_x86_64::<Vec<u8>>(&mir, &mut output, os).expect("emit");
            let text = String::from_utf8(output).expect("utf8 assembly");
            assert_eq!(text.contains(note), expected, "{os:?}: {text}");
        }
    }
}
