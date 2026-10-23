mod simd;
pub mod util;

use crate::mir_codegen::common::integer_conversion;
use crate::mir_codegen::simd::vector_slots;

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    env::temp_dir,
    fs,
};

use std::{io::Write, process::id};

use crate::error::LaminaError;

use crate::{
    mir::{
        AddressMode, Function, Global, Immediate, Instruction as MirInst, IntBinOp, MirType,
        Module as MirModule, Operand, Register, ScalarType, Signature, VirtualReg,
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

/// Trait-backed MIR ⇒ WebAssembly code generator.
pub struct WasmCodegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> WasmCodegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
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
        generate_mir_wasm_with_units(module, writer, self.base.target_os, codegen_units)
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
        self.base.emit_asm_base(generate_mir_wasm, "WASM")
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
    target_os: TargetOperatingSystem,
) -> Result<Vec<u8>, CodegenError> {
    let mut output = Vec::new();
    let abi = WasmABI::new(target_os);

    let mangled_name = abi.mangle_function_name(func_name);
    writeln!(output, "  (func ${mangled_name}")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    for (i, _param) in func.sig.params.iter().enumerate() {
        writeln!(output, "    (param $p{i} i64)")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    if func.sig.ret_ty.is_some() {
        writeln!(output, "    (result i64)")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
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
            // use_regs omits the slot a StackAddr names, so claim it here.
            if let MirInst::StackAddr { slot, .. } = inst {
                local_vregs.insert(slot);
            }
        }
    }

    let mut vreg_to_local: HashMap<VirtualReg, usize> = HashMap::new();
    for (local_idx, vreg) in local_vregs.into_iter().enumerate() {
        vreg_to_local.insert(*vreg, local_idx);
        let declaration = if vectors.contains(vreg) {
            format!("    (local $l{local_idx} v128)")
        } else {
            abi.generate_local_decl(local_idx)
        };
        writeln!(output, "{declaration}")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    let mut block_labels: HashMap<&str, usize> = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        block_labels.insert(&block.label, idx);
    }

    writeln!(output, "    (local $pc i64)")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    for (i, param) in func.sig.params.iter().enumerate() {
        writeln!(output, "    local.get $p{i}")
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        store_to_register_wasm(&param.reg, &mut output, &vreg_to_local)
            .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
    }
    let entry_idx = block_labels.get(func.entry.as_str()).copied().unwrap_or(0);
    writeln!(output, "    i64.const {entry_idx}")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    writeln!(output, "    local.set $pc")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    writeln!(output, "    (loop $dispatch_loop")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    let num_blocks = func.blocks.len();
    for i in (0..num_blocks).rev() {
        writeln!(output, "      (block $block_{i}")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    writeln!(output, "        local.get $pc")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    writeln!(output, "        i32.wrap_i64")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    write!(output, "        br_table")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    for i in 0..num_blocks {
        write!(output, " {i}")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }
    writeln!(output, " 0")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    for block in func.blocks.iter() {
        writeln!(output, "      )")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
        writeln!(output, "      ;; Block: {}", block.label)
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
        for inst in &block.instructions {
            emit_instruction_wasm(inst, &mut output, &vreg_to_local, &block_labels)
                .map_err(|e| CodegenError::InvalidCodegenOptions(e.to_string()))?;
        }
        writeln!(output, "      br $dispatch_loop")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    writeln!(output, "    )")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    if func.sig.ret_ty.is_some() {
        writeln!(output, "    i64.const 0")
            .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;
    }

    writeln!(output, "  )")
        .map_err(|e| CodegenError::InvalidCodegenOptions(format!("IO error: {e}")))?;

    Ok(output)
}

pub fn generate_mir_wasm<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_wasm_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_wasm_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), LaminaError> {
    validate_module_call_parameters(module, TargetArchitecture::Wasm64)?;
    let results = compile_functions_parallel(
        module,
        target_os,
        codegen_units,
        compile_single_function_wasm,
    )
    .map_err(parallel_codegen_error)?;

    let abi = WasmABI::new(target_os);

    // WASM module header
    writeln!(writer, "(module")?;
    writeln!(writer, "  {}", abi.get_print_import())?;
    for decl in abi.get_runtime_imports() {
        writeln!(writer, "  {decl}")?;
    }

    // Emit import declarations for external functions
    for func_name in &module.external_functions {
        if let Some(func) = module.functions.get(func_name) {
            let mangled_name = abi.mangle_function_name(func_name);
            write!(
                writer,
                "  (import \"env\" \"{func_name}\" (func ${mangled_name}"
            )?;

            // Parameters
            for _param in &func.sig.params {
                write!(writer, " (param i64)")?;
            }

            // Return type
            if func.sig.ret_ty.is_some() {
                write!(writer, " (result i64)")?;
            }

            writeln!(writer, ")")?;
        }
    }

    // Global variables for virtual registers
    let mut global_count = 0;
    for func in module.functions.values() {
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(_) = dst
                {
                    global_count += 1;
                }
            }
        }
    }

    // Memory must follow the imports; loads and stores need it declared.
    writeln!(writer, "  (memory (export \"memory\") 1)")?;

    for i in 0..global_count {
        writeln!(writer, "{}", abi.generate_global_decl(i))?;
    }

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    for func_name in module.functions.keys() {
        if func_name == "main" {
            writeln!(writer, "  (export \"main\" (func $main))")?;
        }
    }

    writeln!(writer, ")")?;

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
    vreg_to_local: &HashMap<VirtualReg, usize>,
    block_labels: &HashMap<&str, usize>,
) -> Result<(), LaminaError> {
    if let Some(operations) = integer_conversion(inst) {
        for operation in operations {
            emit_instruction_wasm(&operation, writer, vreg_to_local, block_labels)?;
        }
        return Ok(());
    }
    match inst {
        MirInst::VectorOp { .. } => simd::emit(inst, writer, vreg_to_local)?,
        MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
            simd::emit(inst, writer, vreg_to_local)?
        }

        MirInst::Copy { ty, dst, src } => {
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
            ty: _,
        } => {
            load_operand_wasm(lhs, writer, vreg_to_local)?;
            load_operand_wasm(rhs, writer, vreg_to_local)?;

            emit_int_cmp_op(op, writer)?;

            // Convert i32 result to i64
            writeln!(writer, "      i64.extend_i32_u")?;

            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "print" {
                // Handle print intrinsic
                if let Some(arg) = args.first() {
                    load_operand_wasm(arg, writer, vreg_to_local)?;
                    writeln!(writer, "      call $log")?;
                }
            } else {
                // General function call implementation
                // WebAssembly passes all arguments on the stack in order
                for arg in args.iter() {
                    load_operand_wasm(arg, writer, vreg_to_local)?;
                }

                // Call the function
                writeln!(writer, "      call ${name}")?;

                // Note: WebAssembly functions return values on the stack
                // If there's a return value, it's already on the stack
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
                    // wasm32 memory takes i32 addresses.
                    writeln!(writer, "      i32.wrap_i64")?;

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
                    writeln!(writer, "      i32.wrap_i64")?;

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
                    writeln!(writer, "      i32.wrap_i64")?;
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
                    writeln!(writer, "      i32.wrap_i64")?;
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
            // Return exits the function
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
            // Locals stand in for frame slots here, so the slot's local carries
            // the address the rest of the lowering expects.
            load_register_wasm(&Register::Virtual(*slot), writer, vreg_to_local)?;
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
            // WASM does not have a guaranteed tail-call instruction in baseline WAT,
            // so we emit a normal return_call if the target is known, otherwise
            // a regular call followed by return.
            for (i, arg) in args.iter().enumerate() {
                let _ = i;
                load_operand_wasm(arg, writer, vreg_to_local)?;
            }
            writeln!(writer, "        call ${name}")?;
            writeln!(writer, "        return")?;
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
mod tests {
    use super::*;
    use crate::mir::FunctionBuilder;

    #[test]
    fn float_copy_emits_local_move_not_add() {
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
        generate_mir_wasm(&module, &mut output, TargetOperatingSystem::Unknown)
            .expect("WASM codegen should succeed");
        let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

        assert!(
            assembly.contains("local.get"),
            "expected local move: {assembly}"
        );
        assert!(
            assembly.contains("local.set"),
            "expected local move: {assembly}"
        );
        assert!(
            !assembly.contains("f64.add"),
            "unexpected float add: {assembly}"
        );
    }

    #[test]
    fn dispatcher_starts_at_non_leading_entry_block() {
        let mut function = FunctionBuilder::new("entry_second")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("first")
            .instr(MirInst::Ret {
                value: Some(Operand::Immediate(Immediate::I64(7))),
            })
            .block("second")
            .instr(MirInst::Ret {
                value: Some(Operand::Immediate(Immediate::I64(9))),
            })
            .build();
        function.entry = "second".to_string();
        let mut module = MirModule::new("entry_test");
        module.add_function(function);
        let mut output = Vec::new();
        generate_mir_wasm(&module, &mut output, TargetOperatingSystem::Unknown)
            .expect("WASM codegen should succeed");
        let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

        assert!(
            assembly.contains("i64.const 1\n    local.set $pc"),
            "dispatcher should start at the entry block index: {assembly}"
        );
    }
}
