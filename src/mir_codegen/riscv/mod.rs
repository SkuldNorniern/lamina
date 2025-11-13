mod abi;
mod frame;
mod regalloc;
mod util;

use abi::RiscVAbi;
use frame::RiscVFrame;
use regalloc::RiscVRegAlloc;
use std::io::Write;
use std::result::Result;
use util::*;

use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::{Codegen, CodegenError, CodegenOptions};
use crate::target::TargetOperatingSystem;

/// Trait-backed MIR ⇒ RISC-V code generator.
pub struct RiscVCodegen<'a> {
    target_os: TargetOperatingSystem,
    module: Option<&'a MirModule>,
    prepared: bool,
    verbose: bool,
    output: Vec<u8>,
}

impl<'a> RiscVCodegen<'a> {
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

    /// Drain the internal assembly buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output)
    }

    /// Emit assembly for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_riscv(module, writer, self.target_os)
    }
}

impl<'a> Codegen for RiscVCodegen<'a> {
    const BIN_EXT: &'static str = "o";
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
            generate_mir_riscv(module, &mut self.output, self.target_os).map_err(|e| {
                CodegenError::InvalidCodegenOptions(format!("RISC-V emission failed: {}", e))
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
            "Binary emission not supported".to_string(),
        ))
    }
}

pub fn generate_mir_riscv<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    let abi = RiscVAbi::new(target_os);

    // Emit format strings for print intrinsics
    writeln!(writer, "{}", abi.get_data_section())?;
    writeln!(writer, "{}", abi.get_print_format())?;

    // Text section header
    writeln!(writer, "{}", abi.get_text_section())?;
    writeln!(writer, "{}", abi.get_main_global())?;

    for (func_name, func) in &module.functions {
        // Function label
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, "{}:", label)?;

        // Create register allocator for this function
        let mut reg_alloc = RiscVRegAlloc::new(target_os);

        // Allocate stack space for virtual registers
        let mut stack_slots: std::collections::HashMap<crate::mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        let mut next_slot = 0;

        // Assign stack slots to all virtual registers used in the function
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                        && !stack_slots.contains_key(vreg) {
                            stack_slots
                                .insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot));
                            next_slot += 1;
                        }
                // Also check for registers used in operands
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg) {
                            stack_slots
                                .insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot));
                            next_slot += 1;
                        }
                }
            }
        }

        // Generate function prologue
        let stack_size = stack_slots.len() * 8;
        RiscVFrame::generate_prologue(writer, stack_size)?;

        // Process each block
        for block in &func.blocks {
            writeln!(writer, ".L_{}:", block.label)?;

            for inst in &block.instructions {
                emit_instruction_riscv(inst, writer, &mut reg_alloc, &stack_slots)?;
            }
        }
    }

    Ok(())
}

fn emit_instruction_riscv<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &mut RiscVRegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
) -> Result<(), crate::error::LaminaError> {
    match inst {
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, "a1")?;

            // Perform operation
            match op {
                crate::mir::IntBinOp::Add => writeln!(writer, "    add a0, a0, a1")?,
                crate::mir::IntBinOp::Sub => writeln!(writer, "    sub a0, a0, a1")?,
                crate::mir::IntBinOp::Mul => writeln!(writer, "    mul a0, a0, a1")?,
                crate::mir::IntBinOp::SDiv => writeln!(writer, "    div a0, a0, a1")?,
                crate::mir::IntBinOp::UDiv => writeln!(writer, "    divu a0, a0, a1")?,
                crate::mir::IntBinOp::SRem => writeln!(writer, "    rem a0, a0, a1")?,
                crate::mir::IntBinOp::URem => writeln!(writer, "    remu a0, a0, a1")?,
                crate::mir::IntBinOp::And => writeln!(writer, "    and a0, a0, a1")?,
                crate::mir::IntBinOp::Or => writeln!(writer, "    or a0, a0, a1")?,
                crate::mir::IntBinOp::Xor => writeln!(writer, "    xor a0, a0, a1")?,
                crate::mir::IntBinOp::Shl => writeln!(writer, "    sll a0, a0, a1")?,
                crate::mir::IntBinOp::AShr => writeln!(writer, "    sra a0, a0, a1")?,
                crate::mir::IntBinOp::LShr => writeln!(writer, "    srl a0, a0, a1")?,
                _ => writeln!(writer, "    # TODO: unimplemented binary op")?,
            }

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc)?;
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, "a1")?;

            // Perform comparison
            emit_int_cmp_op(op, writer)?;

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc)?;
            }
        }
        MirInst::Call { name, args, ret } => {
            // Handle print intrinsic
            if name == "print" {
                if let Some(arg) = args.first() {
                    load_operand_to_register(arg, writer, reg_alloc, "a0")?;
                    // Print integer (simplified - would need proper printf setup)
                    writeln!(writer, "    # print intrinsic - would call printf")?;
                }
            } else {
                writeln!(writer, "    # TODO: function calls")?;
            }

            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg {
                    // Assume return value is in a0
                    store_register_to_register("a0", vreg, writer, reg_alloc)?;
                }
        }
        MirInst::Load {
            dst,
            addr,
            ty: _,
            attrs: _,
        } => {
            writeln!(writer, "    # TODO: load instruction")?;
            // For now, just push a dummy value
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("zero", vreg, writer, reg_alloc)?;
            }
        }
        MirInst::Store {
            addr: _,
            src: _,
            ty: _,
            attrs: _,
        } => {
            writeln!(writer, "    # TODO: store instruction")?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                load_operand_to_register(val, writer, reg_alloc, "a0")?;
            }
            // Epilogue
            let stack_size = stack_slots.len() * 8;
            RiscVFrame::generate_epilogue(writer, stack_size)?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    j .L_{}", target)?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_register(vreg, writer, reg_alloc, "t0")?;
                writeln!(writer, "    bnez t0, .L_{}", true_target)?;
                writeln!(writer, "    j .L_{}", false_target)?;
            }
        }
        _ => {
            writeln!(writer, "    # TODO: unimplemented instruction")?;
        }
    }

    Ok(())
}
