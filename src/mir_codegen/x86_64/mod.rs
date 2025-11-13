pub mod abi;
pub mod frame;
pub mod regalloc;
pub mod util;

use abi::X86ABI;
use frame::X86Frame;
use regalloc::X64RegAlloc;
use std::io::Write;
use std::result::Result;
use util::*;

use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::{Codegen, CodegenError, CodegenOptions};
use crate::target::TargetOperatingSystem;

/// Trait-backed MIR ⇒ x86_64 code generator.
pub struct X86Codegen<'a> {
    target_os: TargetOperatingSystem,
    module: Option<&'a MirModule>,
    prepared: bool,
    verbose: bool,
    output: Vec<u8>,
}

impl<'a> X86Codegen<'a> {
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
        generate_mir_x86_64(module, writer, self.target_os)
    }
}

impl<'a> Codegen for X86Codegen<'a> {
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
            generate_mir_x86_64(module, &mut self.output, self.target_os).map_err(|e| {
                CodegenError::InvalidCodegenOptions(format!("Assembly emission failed: {}", e))
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

pub fn generate_mir_x86_64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    let abi = X86ABI::new(target_os);

    // Emit format strings for print intrinsics
    match target_os {
        TargetOperatingSystem::MacOS => {
            writeln!(writer, ".section __TEXT,__cstring,cstring_literals")?;
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

    // Text section header
    writeln!(writer, ".text")?;
    writeln!(writer, "{}", abi.get_main_global())?;

    for (func_name, func) in &module.functions {
        // Function label
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, "{}:", label)?;

        // Create register allocator for this function
        let mut reg_alloc = X64RegAlloc::new(target_os);

        // Allocate stack space for virtual registers
        let mut stack_slots: std::collections::HashMap<crate::mir::VirtualReg, i32> =
            std::collections::HashMap::new();

        // Assign stack slots to all virtual registers used in the function
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                        && !stack_slots.contains_key(vreg) {
                            stack_slots
                                .insert(*vreg, X86Frame::calculate_stack_offset(stack_slots.len()));
                        }
                // Also check for registers used in operands
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg) {
                            stack_slots
                                .insert(*vreg, X86Frame::calculate_stack_offset(stack_slots.len()));
                        }
                }
            }
        }

        // Generate function prologue
        let stack_size = stack_slots.len() * 8;
        X86Frame::generate_prologue(writer, stack_size)?;

        // Process each block
        for block in &func.blocks {
            writeln!(writer, ".L_{}:", block.label)?;

            for inst in &block.instructions {
                emit_instruction_x86_64(
                    inst,
                    writer,
                    &mut reg_alloc,
                    &stack_slots,
                    stack_size,
                    target_os,
                )?;
            }
        }
    }

    Ok(())
}

fn emit_instruction_x86_64(
    inst: &MirInst,
    writer: &mut impl Write,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
    stack_size: usize,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    match inst {
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            // Load lhs to rax
            load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
            // Load rhs to scratch register
            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;

            match op {
                crate::mir::IntBinOp::Add => writeln!(writer, "    addq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::Sub => writeln!(writer, "    subq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::Mul => writeln!(writer, "    imulq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::SDiv => {
                    writeln!(writer, "    cqto")?;
                    writeln!(writer, "    idivq %{}", scratch)?;
                }
                crate::mir::IntBinOp::UDiv => {
                    writeln!(writer, "    xorq %rdx, %rdx")?;
                    writeln!(writer, "    divq %{}", scratch)?;
                }
                crate::mir::IntBinOp::SRem => {
                    writeln!(writer, "    cqto")?;
                    writeln!(writer, "    idivq %{}", scratch)?;
                    writeln!(writer, "    movq %rdx, %rax")?;
                }
                crate::mir::IntBinOp::URem => {
                    writeln!(writer, "    xorq %rdx, %rdx")?;
                    writeln!(writer, "    divq %{}", scratch)?;
                    writeln!(writer, "    movq %rdx, %rax")?;
                }
                crate::mir::IntBinOp::And => writeln!(writer, "    andq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::Or => writeln!(writer, "    orq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::Xor => writeln!(writer, "    xorq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::Shl => writeln!(writer, "    shlq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::AShr => writeln!(writer, "    sarq %{}, %rax", scratch)?,
                crate::mir::IntBinOp::LShr => writeln!(writer, "    shrq %{}, %rax", scratch)?,
                _ => writeln!(writer, "    # TODO: unimplemented binary op")?,
            }

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }

            // Free scratch register
            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            // Load lhs to rax
            load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
            // Load rhs to scratch register
            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;

            writeln!(writer, "    cmpq %{}, %rax", scratch)?;
            match op {
                crate::mir::IntCmpOp::Eq => writeln!(writer, "    sete %al")?,
                crate::mir::IntCmpOp::Ne => writeln!(writer, "    setne %al")?,
                crate::mir::IntCmpOp::SLt => writeln!(writer, "    setl %al")?,
                crate::mir::IntCmpOp::SLe => writeln!(writer, "    setle %al")?,
                crate::mir::IntCmpOp::SGt => writeln!(writer, "    setg %al")?,
                crate::mir::IntCmpOp::SGe => writeln!(writer, "    setge %al")?,
                crate::mir::IntCmpOp::ULt => writeln!(writer, "    setb %al")?,
                crate::mir::IntCmpOp::ULe => writeln!(writer, "    setbe %al")?,
                crate::mir::IntCmpOp::UGt => writeln!(writer, "    seta %al")?,
                crate::mir::IntCmpOp::UGe => writeln!(writer, "    setae %al")?,
                _ => writeln!(writer, "    # TODO: unimplemented compare op")?,
            }
            writeln!(writer, "    movzbq %al, %rax")?;

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }

            // Free scratch register
            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "print" {
                // Handle print intrinsic
                if let Some(arg) = args.first() {
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                    writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                    writeln!(writer, "    movq %rax, %rsi")?;
                    if target_os == TargetOperatingSystem::MacOS {
                        writeln!(writer, "    call _printf")?;
                    } else {
                        writeln!(writer, "    xorl %eax, %eax")?;
                        writeln!(writer, "    call printf")?;
                    }
                }
            } else {
                writeln!(writer, "    # TODO: function calls")?;
            }

            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg {
                    // For now, assume return value is in rax
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
        }
        MirInst::Load {
            dst,
            addr,
            ty: _,
            attrs: _,
        } => {
            // Simple direct load for now - assume addr is BaseOffset with offset 0
            if let crate::mir::AddressMode::BaseOffset { base, offset: 0 } = addr {
                if let Register::Virtual(vreg) = base {
                    load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                }
                writeln!(writer, "    movq (%rax), %rax")?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::Store {
            addr,
            src,
            ty: _,
            attrs: _,
        } => {
            // Simple direct store for now
            load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
            if let crate::mir::AddressMode::BaseOffset { base, offset: 0 } = addr {
                let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                if let Register::Virtual(vreg) = base {
                    load_register_to_register(vreg, writer, reg_alloc, stack_slots, scratch)?;
                }
                writeln!(writer, "    movq %rax, (%{})", scratch)?;
                if scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
            }
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                load_operand_to_rax(val, writer, reg_alloc, stack_slots)?;
            }
            // Epilogue
            X86Frame::generate_epilogue(writer, stack_size)?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    jmp .L_{}", target)?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            // Load condition to register
            if let Register::Virtual(vreg) = cond {
                load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
            }
            writeln!(writer, "    testq %rax, %rax")?;
            writeln!(writer, "    jnz .L_{}", true_target)?;
            writeln!(writer, "    jmp .L_{}", false_target)?;
        }
        _ => {
            writeln!(writer, "    # TODO: unimplemented instruction")?;
        }
    }

    Ok(())
}
