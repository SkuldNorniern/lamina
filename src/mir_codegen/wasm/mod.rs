pub mod abi;
pub mod regalloc;
pub mod util;

use std::fs;
use std::io::Write;
use std::result::Result;

use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions, assemble,
    capability::{CapabilitySet, CodegenCapability},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use abi::WasmABI;
use util::{
    emit_int_binary_op, emit_int_cmp_op, load_operand_wasm, load_register_wasm,
    store_to_register_wasm,
};

use crate::mir_codegen::common::CodegenBase;

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
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_wasm(module, writer, self.base.target_os)
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
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            // Print requires runtime support (console.log import)
            CodegenCapability::Print,
            // Memory operations are supported but limited
            CodegenCapability::MemoryOperations,
            // Note: The following are NOT supported:
            // - HeapAllocation (requires WASM memory management)
            // - SystemCalls (WASM is sandboxed)
            // - InlineAssembly (not applicable to WASM)
            // - Threading (requires WASM threads proposal)
            // - AtomicOperations (requires WASM threads proposal)
        ]
        .into_iter()
        .collect()
    }

    fn prepare(
        &mut self,
        types: &std::collections::HashMap<String, crate::mir::MirType>,
        globals: &std::collections::HashMap<String, crate::mir::Global>,
        funcs: &std::collections::HashMap<String, crate::mir::Signature>,
        codegen_units: usize,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError> {
        self.base
            .prepare_base(types, globals, funcs, codegen_units, verbose, options, input_name)
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
        let temp_wat = std::env::temp_dir().join(format!("lamina_wasm_{}.wat", std::process::id()));
        fs::write(&temp_wat, &wat_content).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!(
                "Failed to write temporary WAT file: {}",
                e
            ))
        })?;

        // Convert WAT to binary WASM using wat2wasm
        let temp_wasm =
            std::env::temp_dir().join(format!("lamina_wasm_{}.wasm", std::process::id()));
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
            CodegenError::InvalidCodegenOptions(format!("Failed to assemble WASM: {}", e))
        })?;

        // Read binary WASM back into output buffer
        let wasm_binary = fs::read(&temp_wasm).map_err(|e| {
            CodegenError::InvalidCodegenOptions(format!("Failed to read WASM binary: {}", e))
        })?;

        self.base.output = wasm_binary;

        // Clean up temporary files
        let _ = fs::remove_file(&temp_wat);
        let _ = fs::remove_file(&temp_wasm);

        Ok(())
    }
}

use crate::mir_codegen::common::compile_functions_parallel;

fn compile_single_function_wasm(
    func_name: &str,
    func: &crate::mir::Function,
    target_os: TargetOperatingSystem,
) -> Result<Vec<u8>, crate::mir_codegen::CodegenError> {
    use std::io::Write;
    let mut output = Vec::new();
    let abi = WasmABI::new(target_os);
    
    let mangled_name = abi.mangle_function_name(func_name);
    writeln!(output, "  (func ${}", mangled_name).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

    for (i, _param) in func.sig.params.iter().enumerate() {
        writeln!(output, "    (param $p{} i64)", i).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }

    if func.sig.ret_ty.is_some() {
        writeln!(output, "    (result i64)").map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }

    let mut local_vregs = std::collections::HashSet::new();
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

    let mut vreg_to_local: std::collections::HashMap<crate::mir::VirtualReg, usize> =
        std::collections::HashMap::new();
    for (local_idx, vreg) in local_vregs.into_iter().enumerate() {
        vreg_to_local.insert(*vreg, local_idx);
        writeln!(output, "{}", abi.generate_local_decl(local_idx)).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }

    let mut block_labels: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        block_labels.insert(&block.label, idx);
    }

    writeln!(output, "    (local $pc i64)").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    writeln!(output, "    i64.const 0").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    writeln!(output, "    local.set $pc").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    writeln!(output, "    (loop $dispatch_loop").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

    let num_blocks = func.blocks.len();
    for i in (0..num_blocks).rev() {
        writeln!(output, "      (block $block_{}", i).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }

    writeln!(output, "        local.get $pc").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    writeln!(output, "        i32.wrap_i64").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    write!(output, "        br_table").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    for i in 0..num_blocks {
        write!(output, " {}", i).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }
    writeln!(output, " 0").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

    for block in func.blocks.iter() {
        writeln!(output, "      )").map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        writeln!(output, "      ;; Block: {}", block.label).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        for inst in &block.instructions {
            emit_instruction_wasm(inst, &mut output, &vreg_to_local, &block_labels).map_err(|e| {
                crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
            })?;
        }
        writeln!(output, "      br $dispatch_loop").map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }

    writeln!(output, "    )").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    
    if func.sig.ret_ty.is_some() {
        writeln!(output, "    i64.const 0").map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
    }
    
    writeln!(output, "  )").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

    Ok(output)
}

pub fn generate_mir_wasm<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    generate_mir_wasm_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_wasm_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), crate::error::LaminaError> {
    let abi = WasmABI::new(target_os);

    // WASM module header
    writeln!(writer, "(module")?;
    writeln!(writer, "  {}", abi.get_print_import())?;

    // Emit import declarations for external functions
    for func_name in &module.external_functions {
        if let Some(func) = module.functions.get(func_name) {
            let mangled_name = abi.mangle_function_name(func_name);
            write!(writer, "  (import \"env\" \"{}\" (func ${}", func_name, mangled_name)?;
            
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

    for i in 0..global_count {
        writeln!(writer, "{}", abi.generate_global_decl(i))?;
    }

    let results = compile_functions_parallel(
        module,
        target_os,
        codegen_units,
        compile_single_function_wasm,
    ).map_err(|e| {
        use crate::codegen::FeatureType;
        crate::error::LaminaError::CodegenError(
            crate::codegen::CodegenError::UnsupportedFeature(
                FeatureType::Custom(format!("Parallel compilation error: {:?}", e))
            )
        )
    })?;

    for result in results {
        writer.write_all(&result.assembly)?;
            }

    for (func_name, _func) in &module.functions {
        if func_name == "main" {
            writeln!(writer, "  (export \"main\" (func $main))")?;
        }
    }

    writeln!(writer, ")")?;

    Ok(())
}

fn emit_instruction_wasm(
    inst: &MirInst,
    writer: &mut impl Write,
    vreg_to_local: &std::collections::HashMap<crate::mir::VirtualReg, usize>,
    block_labels: &std::collections::HashMap<&str, usize>,
) -> Result<(), crate::error::LaminaError> {
    match inst {
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            load_operand_wasm(lhs, writer, vreg_to_local)?;
            load_operand_wasm(rhs, writer, vreg_to_local)?;

            emit_int_binary_op(op, writer)?;

            if let Register::Virtual(vreg) = dst {
                store_to_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
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
                writeln!(writer, "      call ${}", name)?;

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
                crate::mir::AddressMode::BaseOffset { base, offset } => {
                    // Load base address onto stack
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }

                    // Emit load instruction based on type
                    match ty {
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I8) => {
                            writeln!(writer, "      i64.load8_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I16) => {
                            writeln!(writer, "      i64.load16_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I32) => {
                            writeln!(writer, "      i64.load32_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)
                        | crate::mir::MirType::Scalar(crate::mir::ScalarType::Ptr) => {
                            writeln!(writer, "      i64.load")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::F32) => {
                            writeln!(writer, "      f32.load")?;
                            // Convert to i64 for storage (WebAssembly uses separate stacks)
                            writeln!(writer, "      i32.reinterpret_f32")?;
                            writeln!(writer, "      i64.extend_i32_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::F64) => {
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
                crate::mir::AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
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

                    // Emit load instruction based on type (same as BaseOffset)
                    match ty {
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I8) => {
                            writeln!(writer, "      i64.load8_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I16) => {
                            writeln!(writer, "      i64.load16_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I32) => {
                            writeln!(writer, "      i64.load32_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)
                        | crate::mir::MirType::Scalar(crate::mir::ScalarType::Ptr) => {
                            writeln!(writer, "      i64.load")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::F32) => {
                            writeln!(writer, "      f32.load")?;
                            writeln!(writer, "      i32.reinterpret_f32")?;
                            writeln!(writer, "      i64.extend_i32_u")?;
                        }
                        crate::mir::MirType::Scalar(crate::mir::ScalarType::F64) => {
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
                crate::mir::AddressMode::BaseOffset { base, offset } => {
                    // Load base address onto stack
                    load_register_wasm(base, writer, vreg_to_local)?;

                    // Add offset if non-zero
                    if *offset != 0 {
                        writeln!(writer, "      i64.const {}", *offset as i64)?;
                        writeln!(writer, "      i64.add")?;
                    }
                }
                crate::mir::AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
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
                }
            }

            // Now load value to store (goes on top of address)
            load_operand_wasm(src, writer, vreg_to_local)?;

            // Emit store instruction based on type
            // Stack now: address (bottom), value (top)
            match ty {
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I8) => {
                    writeln!(writer, "      i64.store8")?;
                }
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I16) => {
                    writeln!(writer, "      i64.store16")?;
                }
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I32) => {
                    writeln!(writer, "      i64.store32")?;
                }
                crate::mir::MirType::Scalar(crate::mir::ScalarType::I64)
                | crate::mir::MirType::Scalar(crate::mir::ScalarType::Ptr) => {
                    writeln!(writer, "      i64.store")?;
                }
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F32) => {
                    // Convert i64 to f32
                    writeln!(writer, "      i32.wrap_i64")?;
                    writeln!(writer, "      f32.reinterpret_i32")?;
                    writeln!(writer, "      f32.store")?;
                }
                crate::mir::MirType::Scalar(crate::mir::ScalarType::F64) => {
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
                writeln!(writer, "        i64.const {}", target_idx)?;
                writeln!(writer, "        local.set $pc")?;
                writeln!(writer, "        br $dispatch_loop")?;
            } else {
                return Err(crate::error::LaminaError::ValidationError(format!(
                    "Unknown block label: {}",
                    target
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
                writeln!(writer, "            i64.const {}", true_idx)?;
                writeln!(writer, "            local.set $pc")?;
            } else {
                return Err(crate::error::LaminaError::ValidationError(format!(
                    "Unknown block label: {}",
                    true_target
                )));
            }
            writeln!(writer, "          )")?;
            writeln!(writer, "          (else")?;
            if let Some(&false_idx) = block_labels.get(false_target.as_str()) {
                writeln!(writer, "            i64.const {}", false_idx)?;
                writeln!(writer, "            local.set $pc")?;
            } else {
                return Err(crate::error::LaminaError::ValidationError(format!(
                    "Unknown block label: {}",
                    false_target
                )));
            }
            writeln!(writer, "          )")?;
            writeln!(writer, "        )")?;
            // Continue dispatch loop
            writeln!(writer, "        br $dispatch_loop")?;
        }
        _ => {
            writeln!(writer, "      ;; TODO: unimplemented instruction")?;
        }
    }

    Ok(())
}

// Utility functions are now in the util module
