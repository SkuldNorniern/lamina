//! Common code for MIR codegen backends.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    fmt::Debug,
    io::Write,
    mem,
    sync::{Arc, mpsc},
    thread,
};

use crate::{
    error::LaminaError,
    mir::{
        Function, Global, Immediate, Instruction, IntBinOp, MirType, Module as MirModule, Operand,
        Register, Signature, VirtualReg,
    },
    mir_codegen::{CodegenError, CodegenOptions, MirCodegenSettings},
};
use lamina_codegen::{Allocation, LiveInterval};
use lamina_platform::TargetOperatingSystem;

pub fn parallel_codegen_error(error: impl Debug) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
        "Parallel compilation error: {error:?}"
    )))
}

/// Assign one stack slot per virtual register defined or used in `func`.
///
/// Defined registers are laid out first, then used-only registers, so the slot
/// indices match the order the per-backend code previously produced. The byte
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
    let mut def_regs: BTreeSet<VirtualReg> = BTreeSet::new();
    let mut used_regs: BTreeSet<VirtualReg> = BTreeSet::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(Register::Virtual(vreg)) = inst.def_reg() {
                def_regs.insert(*vreg);
            }
            if let Instruction::StackAddr { slot, .. } = inst {
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
    // A reserved register addresses the lowest byte of its block, which sits at
    // the deepest slot, so the extra slots are claimed ahead of it.
    let claim = |vreg: &VirtualReg, slots: &mut HashMap<VirtualReg, i32>, next: &mut usize| {
        let extra = func.stack_reservations.get(vreg).copied().unwrap_or(0);
        slots.insert(*vreg, offset_for_slot(*next + extra));
        *next += extra + 1;
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

pub(super) fn integer_conversion(instruction: &Instruction) -> Option<Vec<Instruction>> {
    integer_conversion_with_width(instruction, MirType::i64())
}

pub(super) fn integer_conversion_with_width(
    instruction: &Instruction,
    word_ty: MirType,
) -> Option<Vec<Instruction>> {
    let word_bits = word_ty.size_bytes() * 8;
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
    let width = from.size_bytes().min(to.size_bytes()) * 8;
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
    if to.size_bytes() < word_ty.size_bytes() {
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
    pub module: Option<&'a MirModule>,
    pub prepared: bool,
    pub verbose: bool,
    pub output: Vec<u8>,
    pub codegen_units: usize,
}

impl<'a> CodegenBase<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            target_os,
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
        let assembly = compile_func(func_name, func, target_os)?;
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

/// Emit `.globl` for `main` and every exported function, leaving other functions local.
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

/// Emits storage for every global. Globals whose initializer is all zero (or absent) go to
/// the bss section, the rest to data.
pub fn emit_global_data<W: Write>(
    writer: &mut W,
    module: &MirModule,
    target_os: TargetOperatingSystem,
    mangle: impl Fn(&str) -> String,
) -> Result<(), LaminaError> {
    let (data_section, bss_section) = match target_os {
        TargetOperatingSystem::MacOS => (".section __DATA,__data", ".section __DATA,__data"),
        _ => (".data", ".bss"),
    };
    let mut names: Vec<&str> = module.globals.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        let global = &module.globals[name];
        let label = mangle(name);
        let align_log2 = global.ty.alignment().trailing_zeros();
        let size = global.ty.size_bytes();
        match global.initializer.as_deref() {
            Some(bytes) if bytes.iter().any(|&b| b != 0) => {
                writeln!(writer, "{data_section}")?;
                writeln!(writer, ".p2align {align_log2}")?;
                writeln!(writer, "{label}:")?;
                for chunk in bytes.chunks(16) {
                    let list: Vec<String> = chunk.iter().map(|b| format!("{b:#04x}")).collect();
                    writeln!(writer, "    .byte {}", list.join(", "))?;
                }
            }
            _ => {
                writeln!(writer, "{bss_section}")?;
                writeln!(writer, ".p2align {align_log2}")?;
                writeln!(writer, "{label}:")?;
                writeln!(writer, "    .zero {size}")?;
            }
        }
    }
    Ok(())
}

/// Helper to emit rodata section with format string for print intrinsics.
pub fn emit_print_format_section<W: Write>(
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    match target_os {
        TargetOperatingSystem::MacOS => {
            writeln!(writer, ".section __TEXT,__cstring,cstring_literals")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        TargetOperatingSystem::Windows => {
            writeln!(writer, ".section .rdata,\"dr\"")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        TargetOperatingSystem::Linux => {
            writeln!(writer, ".section .rodata")?;
            writeln!(writer, ".L_mir_fmt_int: .string \"%lld\\n\"")?;
        }
        _ => {
            writeln!(writer, ".section .rodata")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
    }
    Ok(())
}

/// Convert LaminaError to CodegenError with consistent error type.
pub fn lamina_to_codegen_error(err: LaminaError) -> CodegenError {
    match err {
        LaminaError::InternalError(msg) => {
            CodegenError::InvalidCodegenOptions(format!("Internal error: {msg}"))
        }
        LaminaError::CodegenError(inner) => CodegenError::InvalidCodegenOptions(inner.to_string()),
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
        _ => None,
    }
}

fn virtual_register(reg: &Register) -> Option<VirtualReg> {
    match reg {
        Register::Virtual(reg) => Some(*reg),
        _ => None,
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
        match inst {
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
    let mut points = Vec::new();
    let mut definitions = HashMap::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let mut live = live_out[index].clone();
        let mut block_points = Vec::new();
        for inst in block.instructions.iter().rev() {
            let after = live.clone();
            let dst = inst.def_reg().and_then(virtual_reg);
            if let Some(dst) = dst {
                live.remove(&dst);
                *definitions.entry(dst).or_insert(0) += 1;
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
    for (inst, _, after, _) in &points {
        let Some((dst, src)) = dying_input(inst) else {
            continue;
        };
        if (matches!(inst, Instruction::IntBinary { .. }) && !affinities.contains(&(dst, src)))
            || after.contains(&src)
            || definitions.get(&dst) != Some(&1)
            || func
                .sig
                .params
                .iter()
                .any(|param| virtual_reg(&param.reg) == Some(dst))
        {
            continue;
        }
        let (Some(output), Some(Allocation::Register(reg))) =
            (by_reg.get(&dst), plan.get(&src).copied())
        else {
            continue;
        };
        if !allowed(output, reg) {
            continue;
        }
        let interferes = |other| {
            points.iter().any(|(inst, before, after, defined)| {
                (before.contains(&dst) && before.contains(&other))
                    || (after.contains(&dst) && after.contains(&other))
                    || (*defined == Some(dst)
                        && before.contains(&other)
                        && dying_input(inst) != Some((dst, other)))
                    || (*defined == Some(other)
                        && before.contains(&dst)
                        && dying_input(inst) != Some((other, dst)))
            })
        };
        if plan.iter().any(|(&other, &allocation)| {
            other != dst && allocation == Allocation::Register(reg) && interferes(other)
        }) {
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
        let [
            Instruction::IntCmp { dst, .. },
            Instruction::Br {
                cond,
                true_target,
                false_target,
            },
        ] = instructions.as_slice()
        else {
            continue;
        };
        if dst != cond
            || func
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .flat_map(Instruction::use_regs)
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
                .filter_map(move |(i, pair)| match (&pair[0], &pair[1]) {
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
        let aligned_headers = aligned_loop_headers(func, &loops, 24)
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
            ty: MirType::i64(),
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
                ty: MirType::i64(),
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
                .param(a.into(), MirType::i64())
                .block("body")
                .instr(MirInst::IntBinary {
                    op: IntBinOp::Add,
                    dst: b.into(),
                    lhs: Operand::Register(a.into()),
                    rhs: Operand::Immediate(Immediate::I64(1)),
                    ty: MirType::i64(),
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
                    ty: MirType::i64(),
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
    use lamina_platform::TargetArchitecture;
    use std::error::Error;

    #[test]
    fn native_backends_rotate_self_tail_loops_and_schedule_swaps() -> Result<(), Box<dyn Error>> {
        let a = VirtualReg::gpr(0);
        let b = VirtualReg::gpr(1);
        let cond = VirtualReg::gpr(2);
        let func = FunctionBuilder::new("swap")
            .param(a.into(), MirType::i64())
            .param(b.into(), MirType::i64())
            .returns(MirType::i64())
            .block("entry")
            .instr(Instruction::IntCmp {
                op: IntCmpOp::SGt,
                dst: cond.into(),
                lhs: Operand::Register(a.into()),
                rhs: Operand::Register(b.into()),
                ty: MirType::i64(),
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
                "ble .L_swap.body",
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
                    arch,
                    TargetOperatingSystem::Linux,
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
    use crate::{
        error::LaminaError,
        mir::{Module as MirModule, codegen::from_ir},
        mir_codegen::{
            arm::aarch64::generate_mir_aarch64, arx64::generate_mir_arx64,
            powerpc::powerpc64::generate_mir_ppc64, riscv::generate_mir_riscv,
            x86_64::generate_mir_x86_64,
        },
        parser::parse_module,
    };
    use lamina_platform::TargetOperatingSystem;

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
        let mir = from_ir(&ir, "exports").expect("lower export module");
        let emitters: [(&str, Emitter, TargetOperatingSystem); 5] = [
            (
                "x86_64",
                generate_mir_x86_64::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "aarch64",
                generate_mir_aarch64::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "riscv",
                generate_mir_riscv::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "ppc64",
                generate_mir_ppc64::<Vec<u8>>,
                TargetOperatingSystem::Linux,
            ),
            (
                "arx64",
                generate_mir_arx64::<Vec<u8>>,
                TargetOperatingSystem::Unknown,
            ),
        ];
        for (name, emit, os) in emitters {
            let mut output = Vec::new();
            emit(&mir, &mut output, os).expect("emit module");
            let text = String::from_utf8(output).expect("utf8 assembly");
            assert!(text.contains(".globl answer\n"), "{name}: {text}");
            assert!(text.contains(".globl main\n"), "{name}: {text}");
            assert!(!text.contains(".globl helper\n"), "{name}: {text}");
        }
    }

    #[test]
    fn exported_globals_are_global_and_zero_storage_is_bss() {
        let ir = parse_module(
            r#"
@export global @shared: i64 = 3
global @hidden: i64
fn @main() -> i64 {
entry:
 %a = load.i64 @shared
 %b = load.i64 @hidden
 %c = add.i64 %a, %b
 ret.i64 %c
}
"#,
        )
        .expect("parse global module");
        let mir = from_ir(&ir, "globals").expect("lower global module");
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
}
