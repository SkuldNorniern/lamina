//! Common code for MIR codegen backends.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{Arc, mpsc};
use std::thread;

use crate::error::LaminaError;
use crate::mir::{Function, Global, MirType, Module as MirModule, Register, Signature, VirtualReg};
use crate::mir_codegen::{CodegenError, CodegenOptions};
use lamina_platform::TargetOperatingSystem;

pub fn parallel_codegen_error(error: impl std::fmt::Debug) -> LaminaError {
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
pub fn assign_stack_slots(
    func: &Function,
    offset_for_slot: impl Fn(usize) -> i32,
) -> (HashMap<VirtualReg, i32>, HashSet<VirtualReg>) {
    let mut def_regs: HashSet<VirtualReg> = HashSet::new();
    let mut used_regs: HashSet<VirtualReg> = HashSet::new();
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(Register::Virtual(vreg)) = inst.def_reg() {
                def_regs.insert(*vreg);
            }
            for reg in inst.use_regs() {
                if let Register::Virtual(vreg) = reg {
                    used_regs.insert(*vreg);
                }
            }
        }
    }

    let mut stack_slots: HashMap<VirtualReg, i32> = HashMap::new();
    for vreg in &def_regs {
        if !stack_slots.contains_key(vreg) {
            let slot_index = stack_slots.len();
            stack_slots.insert(*vreg, offset_for_slot(slot_index));
        }
    }
    for vreg in &used_regs {
        if !def_regs.contains(vreg) && !stack_slots.contains_key(vreg) {
            let slot_index = stack_slots.len();
            stack_slots.insert(*vreg, offset_for_slot(slot_index));
        }
    }
    (stack_slots, def_regs)
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
        std::mem::take(&mut self.output)
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

    for _ in 0..functions.len() {
        match result_receiver.recv() {
            Ok(result) => results.push(result),
            Err(_) => {
                worker_errors.push(CodegenError::InvalidCodegenOptions(
                    "Worker thread communication error".to_string(),
                ));
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

    if let Some(first_error) = worker_errors.into_iter().next() {
        return Err(first_error);
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
