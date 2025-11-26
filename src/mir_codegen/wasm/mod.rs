pub mod abi;
pub mod regalloc;
pub mod util;

use std::io::Write;
use std::result::Result;

use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::{Codegen, CodegenError, CodegenOptions};
use crate::target::TargetOperatingSystem;
use abi::WasmABI;
use util::{
    emit_int_binary_op, emit_int_cmp_op, load_operand_wasm, load_register_wasm,
    store_to_register_wasm,
};

/// Trait-backed MIR ⇒ WebAssembly code generator.
pub struct WasmCodegen<'a> {
    target_os: TargetOperatingSystem,
    module: Option<&'a MirModule>,
    prepared: bool,
    verbose: bool,
    output: Vec<u8>,
}

impl<'a> WasmCodegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            target_os,
            module: None,
            prepared: false,
            verbose: false,
            output: Vec::new(),
        }
    }

    /// Attach the MIR module that should be emitted in the next codegen pass.
    pub fn set_module(&mut self, module: &'a MirModule) {
        self.module = Some(module);
    }

    /// Drain the internal WASM buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output)
    }

    /// Emit WASM for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_wasm(module, writer, self.target_os)
    }
}

impl<'a> Codegen for WasmCodegen<'a> {
    const BIN_EXT: &'static str = "wasm";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn prepare(
        &mut self,
        _types: &std::collections::HashMap<String, crate::mir::MirType>,
        _globals: &std::collections::HashMap<String, crate::mir::Global>,
        _funcs: &std::collections::HashMap<String, crate::mir::Signature>,
        verbose: bool,
        _options: &[CodegenOptions],
        _input_name: &str,
    ) -> Result<(), CodegenError> {
        self.verbose = verbose;
        self.prepared = true;
        Ok(())
    }

    fn compile(&mut self) -> Result<(), CodegenError> {
        if !self.prepared {
            return Err(CodegenError::InvalidCodegenOptions(
                "Codegen not prepared".to_string(),
            ));
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), CodegenError> {
        Ok(())
    }

    fn emit_asm(&mut self) -> Result<(), CodegenError> {
        if let Some(module) = self.module {
            generate_mir_wasm(module, &mut self.output, self.target_os).map_err(|e| {
                CodegenError::InvalidCodegenOptions(format!("WASM emission failed: {}", e))
            })?;
        } else {
            return Err(CodegenError::InvalidCodegenOptions(
                "No module set for emission".to_string(),
            ));
        }
        Ok(())
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary WASM emission not supported".to_string(),
        ))
    }
}

pub fn generate_mir_wasm<W: Write>(
    module: &MirModule,
    writer: &mut W,
    _target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    // WASM module header
    writeln!(writer, "(module")?;
    writeln!(writer, "  {}", WasmABI::get_print_import())?;

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
        writeln!(writer, "{}", WasmABI::generate_global_decl(i))?;
    }

    // Functions
    for (func_name, func) in &module.functions {
        let mangled_name = WasmABI::mangle_function_name(func_name);
        writeln!(writer, "  (func ${}", mangled_name)?;

        // Parameters
        for (i, _param) in func.sig.params.iter().enumerate() {
            writeln!(writer, "    (param $p{} i64)", i)?;
        }

        // Return type
        if func.sig.ret_ty.is_some() {
            writeln!(writer, "    (result i64)")?;
        }

        // Local variables for virtual registers
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

        // Map virtual registers to local indices
        let mut vreg_to_local: std::collections::HashMap<crate::mir::VirtualReg, usize> =
            std::collections::HashMap::new();
        for (local_idx, vreg) in local_vregs.into_iter().enumerate() {
            vreg_to_local.insert(*vreg, local_idx);
            writeln!(writer, "{}", WasmABI::generate_local_decl(local_idx))?;
        }

        // Function body
        for block in &func.blocks {
            writeln!(writer, "    ;; block {}", block.label)?;

            for inst in &block.instructions {
                emit_instruction_wasm(inst, writer, &vreg_to_local)?;
            }
        }

        writeln!(writer, "  )")?;

        // Export main function
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
                && let Register::Virtual(vreg) = ret_reg {
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
                load_operand_wasm(val, writer, vreg_to_local)?;
            } else {
                writeln!(writer, "      i64.const 0")?;
            }
            writeln!(writer, "      return")?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "      br $block_{}", target)?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_wasm(&Register::Virtual(*vreg), writer, vreg_to_local)?;
            }
            writeln!(writer, "      (if")?;
            writeln!(writer, "        (then")?;
            writeln!(writer, "          br $block_{}", true_target)?;
            writeln!(writer, "        )")?;
            writeln!(writer, "        (else")?;
            writeln!(writer, "          br $block_{}", false_target)?;
            writeln!(writer, "        )")?;
            writeln!(writer, "      )")?;
        }
        _ => {
            writeln!(writer, "      ;; TODO: unimplemented instruction")?;
        }
    }

    Ok(())
}

// Utility functions are now in the util module
