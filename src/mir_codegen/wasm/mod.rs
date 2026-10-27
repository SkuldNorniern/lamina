mod simd;
pub mod util;

use crate::mir_codegen::common::external_names;
use crate::mir_codegen::common::{integer_conversion, io_codegen_error};
use crate::mir_codegen::simd::vector_slots;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env::temp_dir,
    fs,
    sync::Arc,
};

use std::{io::Write, process::id, str};

use crate::error::LaminaError;

use crate::{
    mir::{
        AddressMode, Function, Global, Immediate, Instruction as MirInst, IntBinOp, IntCmpOp,
        MirType, Module as MirModule, Operand, Register, ScalarType, Signature, VirtualReg,
    },
    mir_codegen::{
        Codegen, CodegenError, CodegenOptions, assemble,
        capability::CapabilitySet,
        common::{CodegenBase, compile_functions_parallel, parallel_codegen_error},
        validate_module_call_parameters,
    },
};

use lamina_codegen::wasm::WasmABI;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use util::{
    emit_float_binary_op, emit_float_cmp_op, emit_float_unary_op, emit_int_binary_op,
    emit_int_cmp_op, load_fp_operand_wasm, load_operand_wasm, load_register_wasm,
    store_fp_to_register_wasm, store_to_register_wasm,
};

/// Index type of linear memory. MIR holds every pointer in an i64, so wasm32 narrows
/// one at each memory operation and wasm64 passes it through.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Memory {
    I32,
    I64,
}

impl Memory {
    fn of(arch: TargetArchitecture) -> Self {
        match arch {
            TargetArchitecture::Wasm64 => Self::I64,
            _ => Self::I32,
        }
    }

    fn index_type(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// Narrows the address on the stack to what the memory indexes with.
    fn narrow(self, writer: &mut impl Write, indent: &str) -> Result<(), LaminaError> {
        if self == Self::I32 {
            writeln!(writer, "{indent}i32.wrap_i64")?;
        }
        Ok(())
    }

    /// Widens an address that came out of the memory index type back to an i64.
    fn widen(self, writer: &mut impl Write, indent: &str) -> Result<(), LaminaError> {
        if self == Self::I32 {
            writeln!(writer, "{indent}i64.extend_i32_u")?;
        }
        Ok(())
    }
}

/// Trait-backed MIR ⇒ WebAssembly code generator.
pub struct WasmCodegen<'a> {
    base: CodegenBase<'a>,
    memory: Memory,
}

impl<'a> WasmCodegen<'a> {
    pub fn new(target_arch: TargetArchitecture, target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
            memory: Memory::of(target_arch),
        }
    }

    /// Attach the MIR module that should be emitted in the next codegen pass.
    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    /// Drain the internal WASM buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    /// Emit WASM for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), LaminaError> {
        generate_mir_wasm_with_units(
            module,
            writer,
            self.memory,
            self.base.target_os,
            codegen_units,
        )
    }
}

impl<'a> Codegen for WasmCodegen<'a> {
    const BIN_EXT: &'static str = "wasm";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = true;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn capabilities() -> CapabilitySet {
        CapabilitySet::wasm()
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
        let memory = self.memory;
        self.base.emit_asm_base(
            |module, writer, os| generate_mir_wasm_with_units(module, writer, memory, os, 1),
            "WASM",
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        // WASM codegen matches other targets: emit_asm generates WAT,
        // then assemble module handles wat2wasm conversion
        // This keeps the pipeline consistent: emit_asm -> assemble -> link

        // First generate WAT text format
        self.emit_asm()?;

        // Get the WAT content
        let wat_content = self.base.drain_output();

        // Write WAT to temporary file
        let temp_wat = temp_dir().join(format!("lamina_wasm_{}.wat", id()));
        fs::write(&temp_wat, &wat_content).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("Failed to write temporary WAT file: {e}"))
        })?;

        // Convert WAT to binary WASM using wat2wasm
        let temp_wasm = temp_dir().join(format!("lamina_wasm_{}.wasm", id()));
        let _assemble_result = assemble::assemble(
            &temp_wat,
            &temp_wasm,
            TargetArchitecture::Wasm32,
            self.base.target_os,
            Some(assemble::AssemblerBackend::Wat2Wasm),
            &[],
            self.base.verbose,
        )
        .map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("Failed to assemble WASM: {e}"))
        })?;

        // Read binary WASM back into output buffer
        let wasm_binary = fs::read(&temp_wasm).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("Failed to read WASM binary: {e}"))
        })?;

        self.base.output = wasm_binary;

        // Clean up temporary files
        let _ = fs::remove_file(&temp_wat);
        let _ = fs::remove_file(&temp_wasm);

        Ok(())
    }
}

fn compile_single_function_wasm(
    func_name: &str,
    func: &Function,
    memory: Memory,
    target_os: TargetOperatingSystem,
    addresses: &Addresses,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let abi = WasmABI::new(target_os);

    let mangled_name = abi.mangle_function_name(func_name);
    writeln!(output, "  (func ${mangled_name}").map_err(io_codegen_error)?;

    for (i, _param) in func.sig.params.iter().enumerate() {
        writeln!(output, "    (param $p{i} i64)").map_err(io_codegen_error)?;
    }

    if func.sig.ret_ty.is_some() {
        writeln!(output, "    (result i64)").map_err(io_codegen_error)?;
    }

    let vectors: HashSet<_> = vector_slots(func).into_iter().map(|(v, _)| v).collect();
    let mut local_vregs = BTreeSet::new();
    for param in &func.sig.params {
        if let Register::Virtual(v) = &param.reg {
            local_vregs.insert(v);
        }
    }
    for block in &func.blocks {
        for inst in &block.instructions {
            if let Some(dst) = inst.def_reg()
                && let Register::Virtual(vreg) = dst
            {
                local_vregs.insert(vreg);
            }
            for reg in inst.use_regs() {
                if let Register::Virtual(vreg) = reg {
                    local_vregs.insert(vreg);
                }
            }
        }
    }
    let frame = stack_frame_layout(func);

    let mut vreg_to_local: HashMap<VirtualReg, usize> = HashMap::new();
    for (local_idx, vreg) in local_vregs.into_iter().enumerate() {
        vreg_to_local.insert(*vreg, local_idx);
        let declaration = if vectors.contains(vreg) {
            format!("    (local $l{local_idx} v128)")
        } else {
            abi.generate_local_decl(local_idx)
        };
        writeln!(output, "{declaration}").map_err(io_codegen_error)?;
    }

    let mut block_labels: HashMap<&str, usize> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        block_labels.insert(&block.label, idx);
    }

    writeln!(output, "    (local $pc i64)").map_err(io_codegen_error)?;
    if frame.size > 0 {
        writeln!(output, "    (local $fp i64)").map_err(io_codegen_error)?;
    }
    for (i, param) in func.sig.params.iter().enumerate() {
        writeln!(output, "    local.get $p{i}")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        store_to_register_wasm(&param.reg, &mut output, &vreg_to_local)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    if frame.size > 0 {
        let size = frame.size;
        let index = memory.index_type();
        writeln!(
            output,
            "    global.get $sp\n    {index}.const {size}\n    {index}.lt_u\n    (if (then unreachable))\n    global.get $sp\n    {index}.const {size}\n    {index}.sub\n    global.set $sp\n    global.get $sp"
        )
        .map_err(io_codegen_error)?;
        memory
            .widen(&mut output, "    ")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        writeln!(output, "    local.set $fp").map_err(io_codegen_error)?;
    }
    let entry_idx = block_labels.get(func.entry.as_str()).copied().unwrap_or(0);
    writeln!(output, "    i64.const {entry_idx}").map_err(io_codegen_error)?;
    writeln!(output, "    local.set $pc").map_err(io_codegen_error)?;
    writeln!(output, "    (loop $dispatch_loop").map_err(io_codegen_error)?;

    let num_blocks = func.blocks.len();
    for i in (0..num_blocks).rev() {
        writeln!(output, "      (block $block_{i}").map_err(io_codegen_error)?;
    }

    writeln!(output, "        local.get $pc").map_err(io_codegen_error)?;
    writeln!(output, "        i32.wrap_i64").map_err(io_codegen_error)?;
    write!(output, "        br_table").map_err(io_codegen_error)?;
    for i in 0..num_blocks {
        write!(output, " {i}").map_err(io_codegen_error)?;
    }
    writeln!(output, " 0").map_err(io_codegen_error)?;

    for block in func.blocks.iter() {
        writeln!(output, "      )").map_err(io_codegen_error)?;
        writeln!(output, "      ;; Block: {}", block.label).map_err(io_codegen_error)?;
        for inst in &block.instructions {
            emit_instruction_wasm(
                inst,
                &mut output,
                memory,
                &vreg_to_local,
                &block_labels,
                &frame,
                addresses,
            )
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
        writeln!(output, "      br $dispatch_loop").map_err(io_codegen_error)?;
    }

    writeln!(output, "    )").map_err(io_codegen_error)?;

    if func.sig.ret_ty.is_some() {
        writeln!(output, "    i64.const 0").map_err(io_codegen_error)?;
    }

    writeln!(output, "  )").map_err(io_codegen_error)?;

    Ok(output)
}

pub fn generate_mir_wasm<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_wasm_with_units(module, writer, Memory::of(target_arch), target_os, 1)
}

pub fn generate_mir_wasm_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    memory: Memory,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    let arch = match memory {
        Memory::I32 => TargetArchitecture::Wasm32,
        Memory::I64 => TargetArchitecture::Wasm64,
    };
    validate_module_call_parameters(module, arch)?;
    let table = function_table(module);
    let addresses = Arc::new(Addresses {
        globals: global_addresses(module)?,
        functions: table
            .iter()
            .enumerate()
            .map(|(index, name)| ((*name).to_string(), index as u32))
            .collect(),
        imports: module
            .external_functions
            .iter()
            .filter_map(|name| {
                let sig = &module.functions.get(name)?.sig;
                Some((
                    name.clone(),
                    ImportShape {
                        params: sig.params.iter().map(|p| float_shape(&p.ty)).collect(),
                        result: sig.ret_ty.as_ref().and_then(float_shape),
                    },
                ))
            })
            .collect(),
    });
    let results = compile_functions_parallel(module, target_os, codegen_units, {
        let addresses = Arc::clone(&addresses);
        move |name, func, os| compile_single_function_wasm(name, func, memory, os, &addresses)
    })
    .map_err(parallel_codegen_error)?;

    let abi = WasmABI::new(target_os);

    // WASM module header
    writeln!(writer, "(module")?;
    let uses = |symbol: &str| results.iter().any(|result| calls(&result.assembly, symbol));
    let wasi = target_os == TargetOperatingSystem::Wasi;
    if wasi && memory == Memory::I64 {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            "WASI passes pointers as i32, so it has no wasm64 form".to_string(),
        )));
    }
    if wasi && uses("$logf") {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            "WASI prints no floats; there is no formatter for one".to_string(),
        )));
    }
    let wasi_printing = wasi && (uses("$log") || uses("$logu"));
    if wasi_printing {
        writeln!(writer, "  {WASI_FD_WRITE_IMPORT}")?;
    }
    for (symbol, decl) in abi.get_runtime_imports() {
        // WASI defines its own printing below instead of importing a console.
        if wasi && matches!(*symbol, "$log" | "$logu" | "$logf") {
            continue;
        }
        if uses(symbol) {
            writeln!(writer, "  {decl}")?;
        }
    }

    // Emit import declarations for external functions
    for func_name in external_names(module) {
        if let Some(func) = module.functions.get(func_name) {
            let mangled_name = abi.mangle_function_name(func_name);
            write!(
                writer,
                "  (import \"env\" \"{func_name}\" (func ${mangled_name}"
            )?;

            let shape = &addresses.imports[func_name];
            for param in &shape.params {
                write!(writer, " (param {})", wasm_type(param.as_ref()))?;
            }

            if func.sig.ret_ty.is_some() {
                write!(writer, " (result {})", wasm_type(shape.result.as_ref()))?;
            }

            writeln!(writer, "))")?;
        }
    }

    // A call through a pointer indexes the table, so a module that makes one needs
    // the table, the element segment naming every function, and a declared signature
    // for each shape it calls.
    let indirect: BTreeSet<(usize, bool)> = module
        .functions
        .values()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| match inst {
            MirInst::CallIndirect { args, ret_ty, .. } => Some((args.len(), ret_ty.is_some())),
            _ => None,
        })
        .collect();
    let addressed = module
        .functions
        .values()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            matches!(inst, MirInst::SymbolAddress { symbol, .. }
                if addresses.functions.contains_key(symbol))
        });
    for (arity, returns) in &indirect {
        let name = signature_name(*arity, *returns);
        let params = " (param i64)".repeat(*arity);
        let result = if *returns { " (result i64)" } else { "" };
        writeln!(writer, "  (type ${name} (func{params}{result}))")?;
    }
    if !indirect.is_empty() || addressed {
        writeln!(writer, "  (table {} funcref)", table.len())?;
        write!(writer, "  (elem (i32.const 0)")?;
        for name in &table {
            write!(writer, " ${}", abi.mangle_function_name(name))?;
        }
        writeln!(writer, ")")?;
    }

    // Memory must follow the imports; loads and stores need it declared.
    let index = memory.index_type();
    let wide = if memory == Memory::I64 { " i64" } else { "" };
    writeln!(writer, "  (memory (export \"memory\"){wide} 1)")?;
    writeln!(
        writer,
        "  (global $sp (mut {index}) ({index}.const {WASM_STACK_TOP}))"
    )?;
    for (name, global) in &module.globals {
        let Some(bytes) = global.initializer.as_deref() else {
            continue;
        };
        if bytes.iter().all(|&b| b == 0) {
            continue;
        }
        let address = addresses.globals[name];
        let escaped: String = bytes.iter().map(|b| format!("\\{b:02x}")).collect();
        writeln!(writer, "  (data ({index}.const {address}) \"{escaped}\")")?;
    }

    if wasi_printing {
        writeln!(writer, "{WASI_PRINT_HELPERS}")?;
    }

    for result in &results {
        writer.write_all(&result.assembly)?;
    }

    if let Some(main) = module.functions.get("main") {
        writeln!(writer, "  (export \"main\" (func $main))")?;
        // A WASI command starts at _start and returns nothing.
        if wasi {
            writeln!(writer, "  (func (export \"_start\")")?;
            writeln!(writer, "    call $main")?;
            if main.sig.ret_ty.is_some() {
                writeln!(writer, "    drop")?;
            }
            writeln!(writer, "  )")?;
        }
    }

    writeln!(writer, ")")?;

    Ok(())
}

/// Whether the emitted body calls `symbol`. The name has to end where the match does,
/// so `$log` does not answer for `$logu`.
fn calls(assembly: &[u8], symbol: &str) -> bool {
    let needle = format!("call {symbol}");
    let Ok(text) = str::from_utf8(assembly) else {
        return false;
    };
    text.match_indices(&needle).any(|(at, _)| {
        text[at + needle.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_')
    })
}

const WASM_STACK_TOP: u32 = 65536;

/// WASI has no console to import, so a module that prints writes its own digits and
/// hands them to `fd_write`. The iovec sits at 0, the byte count it writes back at 8,
/// and the digits end just below `WASM_GLOBAL_BASE`.
const WASI_FD_WRITE_IMPORT: &str = "(import \"wasi_snapshot_preview1\" \"fd_write\" (func $fd_write (param i32 i32 i32 i32) (result i32)))";

const WASI_PRINT_HELPERS: &str = r#"  (func $lamina_digits (param $v i64) (param $neg i32) (result i32)
    (local $pos i32)
    i32.const 40
    local.set $pos
    local.get $pos
    i32.const 10
    i32.store8
    (loop $next
      local.get $pos
      i32.const 1
      i32.sub
      local.set $pos
      local.get $pos
      local.get $v
      i64.const 10
      i64.rem_u
      i32.wrap_i64
      i32.const 48
      i32.add
      i32.store8
      local.get $v
      i64.const 10
      i64.div_u
      local.set $v
      local.get $v
      i64.const 0
      i64.ne
      br_if $next
    )
    (if (local.get $neg) (then
      local.get $pos
      i32.const 1
      i32.sub
      local.set $pos
      local.get $pos
      i32.const 45
      i32.store8
    ))
    local.get $pos
  )
  (func $lamina_write (param $pos i32)
    i32.const 0
    local.get $pos
    i32.store
    i32.const 4
    i32.const 41
    local.get $pos
    i32.sub
    i32.store
    i32.const 1
    i32.const 0
    i32.const 1
    i32.const 8
    call $fd_write
    drop
  )
  (func $logu (param $v i64)
    local.get $v
    i32.const 0
    call $lamina_digits
    call $lamina_write
  )
  (func $log (param $v i64)
    (local $neg i32)
    local.get $v
    i64.const 0
    i64.lt_s
    local.set $neg
    (if (local.get $neg) (then
      i64.const 0
      local.get $v
      i64.sub
      local.set $v
    ))
    local.get $v
    local.get $neg
    call $lamina_digits
    call $lamina_write
  )"#;

/// Globals occupy linear memory from this address upward, below the stack. What sits
/// below them is the scratch the WASI print helpers write into.
/// Zero-initialized globals need no data segment because memory starts zeroed.
const WASM_GLOBAL_BASE: u32 = 48;

/// Where a name resolves to. A global names a linear-memory address; a function names
/// its slot in the module's table, which is what a call through a pointer indexes.
pub struct Addresses {
    globals: BTreeMap<String, u32>,
    functions: BTreeMap<String, u32>,
    /// Scalar shape of each imported function. A wasm import has to declare the types
    /// the host wrote, so the bits the backend holds in an i64 are reinterpreted on the
    /// way out and back.
    imports: BTreeMap<String, ImportShape>,
}

/// Width of each parameter and of the result of an imported function, `None` where the
/// value is an integer and travels as an i64.
#[derive(Clone)]
pub struct ImportShape {
    params: Vec<Option<MirType>>,
    result: Option<MirType>,
}

fn float_shape(ty: &MirType) -> Option<MirType> {
    ty.is_float().then_some(*ty)
}

/// Declares `ty` the way a wasm signature spells it.
fn wasm_type(ty: Option<&MirType>) -> &'static str {
    match ty {
        Some(ty) if ty.size_bytes() == 4 => "f32",
        Some(_) => "f64",
        None => "i64",
    }
}

/// Turns the i64 bits on the stack into the float an import declares.
fn reinterpret_out(writer: &mut impl Write, ty: &MirType) -> Result<(), LaminaError> {
    if ty.size_bytes() == 4 {
        writeln!(writer, "      i32.wrap_i64")?;
        writeln!(writer, "      f32.reinterpret_i32")?;
    } else {
        writeln!(writer, "      f64.reinterpret_i64")?;
    }
    Ok(())
}

/// Turns a float an import returned back into the i64 bits the backend holds.
fn reinterpret_in(writer: &mut impl Write, ty: &MirType) -> Result<(), LaminaError> {
    if ty.size_bytes() == 4 {
        writeln!(writer, "      i32.reinterpret_f32")?;
        writeln!(writer, "      i64.extend_i32_u")?;
    } else {
        writeln!(writer, "      i64.reinterpret_f64")?;
    }
    Ok(())
}

/// Functions the module can take the address of, in table order. Imports and
/// definitions share one index space, and the table lists both.
fn function_table(module: &MirModule) -> Vec<&str> {
    let mut names: Vec<&str> = module
        .external_functions
        .iter()
        .map(String::as_str)
        .collect();
    names.extend(
        module
            .functions
            .keys()
            .map(String::as_str)
            .filter(|name| !module.external_functions.contains(*name)),
    );
    names.sort_unstable();
    names
}

/// Signature a `call_indirect` declares. Every wasm function the backend writes takes
/// and returns i64, so an argument count and a result are all that tell two apart.
fn signature_name(arity: usize, returns: bool) -> String {
    let result = if returns { "r" } else { "v" };
    format!("sig_{arity}_{result}")
}

fn global_addresses(module: &MirModule) -> Result<BTreeMap<String, u32>, LaminaError> {
    let mut names: Vec<&str> = module.globals.keys().map(String::as_str).collect();
    names.sort_unstable();
    let mut addresses = BTreeMap::new();
    let mut next = WASM_GLOBAL_BASE;
    for name in names {
        let global = &module.globals[name];
        let align = u32::try_from(global.align)
            .map_err(|_| LaminaError::ValidationError("global alignment overflow".into()))?;
        next = next.div_ceil(align) * align;
        addresses.insert(name.to_string(), next);
        let size = u32::try_from(global.size)
            .map_err(|_| LaminaError::ValidationError("global size overflow".into()))?;
        next = next
            .checked_add(size)
            .ok_or_else(|| LaminaError::ValidationError("global data overflows memory".into()))?;
    }
    if next > WASM_STACK_TOP {
        return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
            "WASM globals do not fit below the stack in one memory page".into(),
        )));
    }
    Ok(addresses)
}

struct StackFrame {
    offsets: HashMap<VirtualReg, i64>,
    size: i64,
}

fn stack_frame_layout(func: &Function) -> StackFrame {
    let slots: BTreeSet<&VirtualReg> = func
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| match inst {
            MirInst::StackAddr { slot, .. } => Some(slot),
            _ => None,
        })
        .collect();
    let mut offsets = HashMap::new();
    let mut size = 0i64;
    for slot in slots {
        let units = 1 + func.stack_reservations.get(slot).copied().unwrap_or(0);
        offsets.insert(*slot, size);
        size += 8 * units as i64;
    }
    StackFrame {
        offsets,
        size: (size + 15) / 16 * 16,
    }
}

fn release_stack_frame(
    writer: &mut impl Write,
    memory: Memory,
    frame: &StackFrame,
) -> Result<(), LaminaError> {
    if frame.size > 0 {
        let size = frame.size;
        let index = memory.index_type();
        writeln!(writer, "        local.get $fp")?;
        memory.narrow(writer, "        ")?;
        writeln!(
            writer,
            "        {index}.const {size}\n        {index}.add\n        global.set $sp"
        )?;
    }
    Ok(())
}

fn extend_integer_wasm(
    writer: &mut impl Write,
    ty: MirType,
    signed: bool,
) -> Result<(), LaminaError> {
    if ty.size_bytes() < 8 {
        let shift = 64 - ty.size_bytes() * 8;
        let right = if signed { "i64.shr_s" } else { "i64.shr_u" };
        writeln!(
            writer,
            "      i64.const {shift}\n      i64.shl\n      i64.const {shift}\n      {right}"
        )?;
    }
    Ok(())
}

fn emit_instruction_wasm(
    inst: &MirInst,
    writer: &mut impl Write,
    memory: Memory,
    vreg_to_local: &HashMap<VirtualReg, usize>,
    block_labels: &HashMap<&str, usize>,
    frame: &StackFrame,
    addresses: &Addresses,
) -> Result<(), LaminaError> {
    if let Some(operations) = integer_conversion(inst) {
        for operation in operations {
            emit_instruction_wasm(
                &operation,
                writer,
                memory,
                vreg_to_local,
                block_labels,
                frame,
                addresses,
            )?;
        }
        return Ok(());
    }
    match inst {
        MirInst::SymbolAddress { dst, symbol } => {
            let address = addresses
                .globals
                .get(symbol)
                .or_else(|| addresses.functions.get(symbol))
                .ok_or_else(|| {
                    LaminaError::ValidationError(format!("symbol address @{symbol} has no global"))
                })?;
            writeln!(writer, "        i64.const {address}")?;
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::CallIndirect {
            target,
            args,
            ret,
            ret_ty,
            ..
        } => {
            for arg in args {
                load_operand_wasm(arg, writer, vreg_to_local)?;
            }
            load_operand_wasm(target, writer, vreg_to_local)?;
            // A table index is an i32 whatever the memory indexes with.
            writeln!(writer, "      i32.wrap_i64")?;
            writeln!(
                writer,
                "      call_indirect (type ${})",
                signature_name(args.len(), ret_ty.is_some())
            )?;
            if let Some(dst) = ret {
                store_to_register_wasm(dst, writer, vreg_to_local)?;
            }
        }
        MirInst::VectorOp { .. } => simd::emit(inst, writer, vreg_to_local)?,
        MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
            simd::emit(inst, writer, vreg_to_local)?
        }

        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if matches!(ty, MirType::Vector(_)) {
                return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                    format!("WASM backend does not support vector Copy of type {ty}"),
                )));
            }
            match src {
                Operand::Immediate(Immediate::F32(value)) => {
                    writeln!(writer, "      i64.const {}", value.to_bits())?;
                }
                Operand::Immediate(Immediate::F64(value)) => {
                    writeln!(writer, "      i64.const {}", value.to_bits() as i64)?;
                }
                _ => load_operand_wasm(src, writer, vreg_to_local)?,
            }
            if !ty.is_float() {
                extend_integer_wasm(writer, *ty, true)?;
            }
            store_to_register_wasm(dst, writer, vreg_to_local)?;
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let (_, overflow) =
                op.division_checks(*ty, lhs.integer_constant(), rhs.integer_constant());
            if overflow {
                let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
                load_operand_wasm(lhs, writer, vreg_to_local)?;
                extend_integer_wasm(writer, *ty, true)?;
                writeln!(writer, "      i64.const {minimum}\n      i64.eq")?;
                load_operand_wasm(rhs, writer, vreg_to_local)?;
                extend_integer_wasm(writer, *ty, true)?;
                writeln!(
                    writer,
                    "      i64.const -1\n      i64.eq\n      i32.and\n      if\n        unreachable\n      end"
                )?;
            }
            let unsigned = matches!(op, IntBinOp::UDiv | IntBinOp::URem | IntBinOp::LShr);
            load_operand_wasm(lhs, writer, vreg_to_local)?;
            extend_integer_wasm(writer, *ty, !unsigned)?;
            load_operand_wasm(rhs, writer, vreg_to_local)?;
            if matches!(op, IntBinOp::Shl | IntBinOp::LShr | IntBinOp::AShr) {
                writeln!(
                    writer,
                    "      i64.const {}\n      i64.and",
                    ty.size_bytes() * 8 - 1
                )?;
            } else {
                extend_integer_wasm(writer, *ty, !unsigned)?;
            }
            emit_int_binary_op(op, writer)?;
            extend_integer_wasm(writer, *ty, true)?;
            store_to_register_wasm(dst, writer, vreg_to_local)?;
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let signed = matches!(
                op,
                IntCmpOp::SLt | IntCmpOp::SLe | IntCmpOp::SGt | IntCmpOp::SGe
            );
            load_operand_wasm(lhs, writer, vreg_to_local)?;
            extend_integer_wasm(writer, *ty, signed)?;
            load_operand_wasm(rhs, writer, vreg_to_local)?;
            extend_integer_wasm(writer, *ty, signed)?;

            emit_int_cmp_op(op, writer)?;

            // Convert i32 result to i64
            writeln!(writer, "      i64.extend_i32_u")?;

            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "print" || name == "printu" {
                // Handle print intrinsic
                if let Some(arg) = args.first() {
                    load_operand_wasm(arg, writer, vreg_to_local)?;
                    if name == "printu" {
                        writeln!(writer, "      call $logu")?;
                    } else {
                        writeln!(writer, "      call $log")?;
                    }
                }
            } else if name == "printf32" || name == "printf64" {
                if let Some(arg) = args.first() {
                    load_operand_wasm(arg, writer, vreg_to_local)?;
                    if name == "printf32" {
                        writeln!(writer, "      i32.wrap_i64")?;
                        writeln!(writer, "      f32.reinterpret_i32")?;
                        writeln!(writer, "      f64.promote_f32")?;
                    } else {
                        writeln!(writer, "      f64.reinterpret_i64")?;
                    }
                    writeln!(writer, "      call $logf")?;
                }
            } else {
                let shape = addresses.imports.get(name);
                for (index, arg) in args.iter().enumerate() {
                    load_operand_wasm(arg, writer, vreg_to_local)?;
                    if let Some(ty) = shape
                        .and_then(|shape| shape.params.get(index))
                        .and_then(Option::as_ref)
                    {
                        reinterpret_out(writer, ty)?;
                    }
                }

                writeln!(writer, "      call ${name}")?;

                if let Some(ty) = shape.and_then(|shape| shape.result.as_ref()) {
                    reinterpret_in(writer, ty)?;
                }
            }

            // Handle return value (already on stack if function returns)
            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            // Compute address: base + offset
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    // Load base address onto stack
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }
                    memory.narrow(writer, "      ")?;

                    // Emit load instruction based on type
                    match ty {
                        MirType::Scalar(ScalarType::I8) => {
                            writeln!(writer, "      i64.load8_u")?;
                        }
                        MirType::Scalar(ScalarType::I16) => {
                            writeln!(writer, "      i64.load16_u")?;
                        }
                        MirType::Scalar(ScalarType::I32) => {
                            writeln!(writer, "      i64.load32_u")?;
                        }
                        MirType::Scalar(ScalarType::I64)
                        | MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                            writeln!(writer, "      i64.load")?;
                        }
                        MirType::Scalar(ScalarType::F32) => {
                            writeln!(writer, "      f32.load")?;
                            // Convert to i64 for storage (WebAssembly uses separate stacks)
                            writeln!(writer, "      i32.reinterpret_f32")?;
                            writeln!(writer, "      i64.extend_i32_u")?;
                        }
                        MirType::Scalar(ScalarType::F64) => {
                            writeln!(writer, "      f64.load")?;
                            // Convert to i64 for storage
                            writeln!(writer, "      i64.reinterpret_f64")?;
                        }
                        _ => {
                            // Default to i64 for unknown types
                            writeln!(writer, "      i64.load")?;
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
                    // Load base address
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Load index
                    load_register_wasm(index, writer, vreg_to_local)?;

                    // Scale index
                    writeln!(writer, "      i64.const {}", *scale as i64)?;
                    writeln!(writer, "      i64.mul")?;

                    // Add base + scaled index
                    writeln!(writer, "      i64.add")?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }
                    memory.narrow(writer, "      ")?;

                    // Emit load instruction based on type (same as BaseOffset)
                    match ty {
                        MirType::Scalar(ScalarType::I8) => {
                            writeln!(writer, "      i64.load8_u")?;
                        }
                        MirType::Scalar(ScalarType::I16) => {
                            writeln!(writer, "      i64.load16_u")?;
                        }
                        MirType::Scalar(ScalarType::I32) => {
                            writeln!(writer, "      i64.load32_u")?;
                        }
                        MirType::Scalar(ScalarType::F32) => {
                            writeln!(writer, "      f32.load")?;
                            writeln!(writer, "      i32.reinterpret_f32")?;
                            writeln!(writer, "      i64.extend_i32_u")?;
                        }
                        MirType::Scalar(ScalarType::F64) => {
                            writeln!(writer, "      f64.load")?;
                            writeln!(writer, "      i64.reinterpret_f64")?;
                        }
                        _ => {
                            writeln!(writer, "      i64.load")?;
                        }
                    }
                }
            }

            // Store loaded value to destination register
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs: _,
        } => {
            // WebAssembly store expects: address on stack, then value on top
            // So we compute address first, then load value

            // Compute address: base + offset
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    // Load base address onto stack
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }
                    memory.narrow(writer, "      ")?;
                }
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                    ..
                } => {
                    // Load base address
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Load index
                    load_register_wasm(index, writer, vreg_to_local)?;

                    // Scale index
                    writeln!(writer, "      i64.const {}", *scale as i64)?;
                    writeln!(writer, "      i64.mul")?;

                    // Add base + scaled index
                    writeln!(writer, "      i64.add")?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }
                    memory.narrow(writer, "      ")?;
                }
            }

            // Now load value to store (goes on top of address)
            load_operand_wasm(src, writer, vreg_to_local)?;

            // Emit store instruction based on type
            // Stack now: address (bottom), value (top)
            match ty {
                MirType::Scalar(ScalarType::I8) => {
                    writeln!(writer, "      i64.store8")?;
                }
                MirType::Scalar(ScalarType::I16) => {
                    writeln!(writer, "      i64.store16")?;
                }
                MirType::Scalar(ScalarType::I32) => {
                    writeln!(writer, "      i64.store32")?;
                }
                MirType::Scalar(ScalarType::I64)
                | MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)) => {
                    writeln!(writer, "      i64.store")?;
                }
                MirType::Scalar(ScalarType::F32) => {
                    // Convert i64 to f32
                    writeln!(writer, "      i32.wrap_i64")?;
                    writeln!(writer, "      f32.reinterpret_i32")?;
                    writeln!(writer, "      f32.store")?;
                }
                MirType::Scalar(ScalarType::F64) => {
                    // Convert i64 to f64
                    writeln!(writer, "      f64.reinterpret_i64")?;
                    writeln!(writer, "      f64.store")?;
                }
                _ => {
                    // Default to i64 for unknown types
                    writeln!(writer, "      i64.store")?;
                }
            }
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                // Load the return value
                load_operand_wasm(val, writer, vreg_to_local)?;
            } else {
                // No return value - use 0
                writeln!(writer, "        i64.const 0")?;
            }
            release_stack_frame(writer, memory, frame)?;
            writeln!(writer, "        return")?;
        }
        MirInst::Jmp { target } => {
            // Set PC to target block index and continue dispatch loop
            if let Some(&target_idx) = block_labels.get(target.as_str()) {
                writeln!(writer, "        i64.const {target_idx}")?;
                writeln!(writer, "        local.set $pc")?;
                writeln!(writer, "        br $dispatch_loop")?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "Unknown block label: {target}"
                )));
            }
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            load_register_wasm(cond, writer, vreg_to_local)?;
            // Convert i64 condition to i32 for WASM if statement
            writeln!(writer, "        i32.wrap_i64")?;
            writeln!(writer, "        (if")?;
            writeln!(writer, "          (then")?;
            if let Some(&true_idx) = block_labels.get(true_target.as_str()) {
                writeln!(writer, "            i64.const {true_idx}")?;
                writeln!(writer, "            local.set $pc")?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "Unknown block label: {true_target}"
                )));
            }
            writeln!(writer, "          )")?;
            writeln!(writer, "          (else")?;
            if let Some(&false_idx) = block_labels.get(false_target.as_str()) {
                writeln!(writer, "            i64.const {false_idx}")?;
                writeln!(writer, "            local.set $pc")?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "Unknown block label: {false_target}"
                )));
            }
            writeln!(writer, "          )")?;
            writeln!(writer, "        )")?;
            // Continue dispatch loop
            writeln!(writer, "        br $dispatch_loop")?;
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let is_f32 = to.size_bytes() == 4;
            load_operand_wasm(src, writer, vreg_to_local)?;
            if from.size_bytes() < 8 {
                extend_integer_wasm(writer, *from, *signed)?;
            }
            let unsigned_word = !*signed && from.size_bytes() == 8;
            let convert = match (is_f32, unsigned_word) {
                (true, true) => "f32.convert_i64_u",
                (true, false) => "f32.convert_i64_s",
                (false, true) => "f64.convert_i64_u",
                (false, false) => "f64.convert_i64_s",
            };
            writeln!(writer, "      {convert}")?;
            if is_f32 {
                writeln!(writer, "      i32.reinterpret_f32")?;
                writeln!(writer, "      i64.extend_i32_u")?;
            } else {
                writeln!(writer, "      i64.reinterpret_f64")?;
            }
            store_to_register_wasm(dst, writer, vreg_to_local)?;
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            load_operand_wasm(src, writer, vreg_to_local)?;
            if from.size_bytes() == 4 {
                writeln!(writer, "      i32.wrap_i64")?;
                writeln!(writer, "      f32.reinterpret_i32")?;
            } else {
                writeln!(writer, "      f64.reinterpret_i64")?;
            }
            if to.size_bytes() == 4 {
                writeln!(writer, "      f32.demote_f64")?;
                writeln!(writer, "      i32.reinterpret_f32")?;
                writeln!(writer, "      i64.extend_i32_u")?;
            } else {
                writeln!(writer, "      f64.promote_f32")?;
                writeln!(writer, "      i64.reinterpret_f64")?;
            }
            store_to_register_wasm(dst, writer, vreg_to_local)?;
        }
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            // trunc_sat maps NaN to 0 and saturates to the 64-bit range. Narrower
            // targets are clamped with select, then sign-extended to match the x86
            // backend's canonical register form.
            let is_f32 = from.size_bytes() == 4;
            load_fp_operand_wasm(src, writer, vreg_to_local, is_f32)?;
            let trunc = match (is_f32, *signed) {
                (true, true) => "i64.trunc_sat_f32_s",
                (true, false) => "i64.trunc_sat_f32_u",
                (false, true) => "i64.trunc_sat_f64_s",
                (false, false) => "i64.trunc_sat_f64_u",
            };
            writeln!(writer, "      {trunc}")?;
            store_to_register_wasm(dst, writer, vreg_to_local)?;
            let bits = to.size_bytes() * 8;
            if bits < 64 {
                let clamps: Vec<(i128, &str)> = if *signed {
                    let max = (1i128 << (bits - 1)) - 1;
                    let min = -(1i128 << (bits - 1));
                    vec![(max, "i64.gt_s"), (min, "i64.lt_s")]
                } else {
                    vec![((1i128 << bits) - 1, "i64.gt_u")]
                };
                for (bound, compare) in clamps {
                    writeln!(writer, "      i64.const {bound}")?;
                    load_operand_wasm(&Operand::Register(dst.clone()), writer, vreg_to_local)?;
                    load_operand_wasm(&Operand::Register(dst.clone()), writer, vreg_to_local)?;
                    writeln!(writer, "      i64.const {bound}")?;
                    writeln!(writer, "      {compare}")?;
                    writeln!(writer, "      select")?;
                    store_to_register_wasm(dst, writer, vreg_to_local)?;
                }
                load_operand_wasm(&Operand::Register(dst.clone()), writer, vreg_to_local)?;
                extend_integer_wasm(writer, *to, true)?;
                store_to_register_wasm(dst, writer, vreg_to_local)?;
            }
        }
        MirInst::FloatBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;

            // Load operands as floats
            load_fp_operand_wasm(lhs, writer, vreg_to_local, is_f32)?;
            load_fp_operand_wasm(rhs, writer, vreg_to_local, is_f32)?;

            // Perform floating-point operation
            emit_float_binary_op(op, writer, is_f32)?;

            // Store result back to register
            if let Register::Virtual(vreg) = dst {
                store_fp_to_register_wasm(writer, vreg_to_local, vreg, is_f32)?;
            }
        }
        MirInst::FloatUnary { op, dst, src, ty } => {
            let is_f32 = ty.size_bytes() == 4;

            // Load operand as float
            load_fp_operand_wasm(src, writer, vreg_to_local, is_f32)?;

            // Perform floating-point unary operation
            emit_float_unary_op(op, writer, is_f32)?;

            // Store result back to register
            if let Register::Virtual(vreg) = dst {
                store_fp_to_register_wasm(writer, vreg_to_local, vreg, is_f32)?;
            }
        }
        MirInst::FloatCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let is_f32 = ty.size_bytes() == 4;

            // Load operands as floats
            load_fp_operand_wasm(lhs, writer, vreg_to_local, is_f32)?;
            load_fp_operand_wasm(rhs, writer, vreg_to_local, is_f32)?;

            // Perform floating-point comparison (result is i32)
            emit_float_cmp_op(op, writer, is_f32)?;

            // Extend i32 result to i64 for storage
            writeln!(writer, "      i64.extend_i32_u")?;

            // Store result to destination
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // WASM select: push true, false, then cond as i32.
            load_operand_wasm(true_val, writer, vreg_to_local)?;
            load_operand_wasm(false_val, writer, vreg_to_local)?;
            load_register_wasm(cond, writer, vreg_to_local)?;
            writeln!(writer, "        i32.wrap_i64")?;
            writeln!(writer, "        select")?;
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            // Emit a linear cmp chain using WASM block/br_if.
            load_register_wasm(value, writer, vreg_to_local)?;
            for (case_val, case_label) in cases {
                if let Some(&target_idx) = block_labels.get(case_label.as_str()) {
                    writeln!(writer, "        i64.const {case_val}")?;
                    // Load value again for comparison (stack-machine needs two copies).
                    load_register_wasm(value, writer, vreg_to_local)?;
                    writeln!(writer, "        i64.eq")?;
                    writeln!(writer, "        (if (then")?;
                    writeln!(writer, "          i64.const {target_idx}")?;
                    writeln!(writer, "          local.set $pc")?;
                    writeln!(writer, "          br $dispatch_loop")?;
                    writeln!(writer, "        ))")?;
                }
            }
            // Fall through to default.
            if let Some(&default_idx) = block_labels.get(default.as_str()) {
                writeln!(writer, "        i64.const {default_idx}")?;
                writeln!(writer, "        local.set $pc")?;
            }
            writeln!(writer, "        br $dispatch_loop")?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "        ;; {text}")?;
        }
        MirInst::StackAddr { dst, slot } => {
            writeln!(writer, "        local.get $fp")?;
            let offset = frame.offsets[slot];
            if offset != 0 {
                writeln!(writer, "        i64.const {offset}")?;
                writeln!(writer, "        i64.add")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Lea { dst, base, offset } => {
            // In WASM linear memory there are no general-purpose effective-address
            // computations on integers, so we materialise base + offset as i64.
            load_register_wasm(base, writer, vreg_to_local)?;
            if *offset != 0 {
                writeln!(writer, "        i64.const {offset}")?;
                writeln!(writer, "        i64.add")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::TailCall { name, args } => {
            for arg in args.iter() {
                load_operand_wasm(arg, writer, vreg_to_local)?;
            }
            if frame.size == 0 {
                writeln!(writer, "        return_call ${name}")?;
            } else {
                // Arguments may point into this frame, so it stays live until the callee returns.
                writeln!(writer, "        call ${name}")?;
                release_stack_frame(writer, memory, frame)?;
                writeln!(writer, "        return")?;
            }
        }
        MirInst::Unreachable => {
            writeln!(writer, "        unreachable")?;
        }
        MirInst::SafePoint | MirInst::StackMap { .. } | MirInst::PatchPoint { .. } => {
            // No-op in AOT/WASM path.
        }
        other => {
            return Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
                format!("WASM backend: instruction not yet supported: {other}"),
            )));
        }
    }

    Ok(())
}

// Utility functions are now in the util module

#[cfg(test)]
mod tests;
