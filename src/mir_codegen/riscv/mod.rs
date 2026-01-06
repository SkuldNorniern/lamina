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
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions,
    capability::{CapabilitySet, CodegenCapability},
};
use lamina_platform::TargetOperatingSystem;

use crate::mir_codegen::common::CodegenBase;

/// Trait-backed MIR ⇒ RISC-V code generator.
pub struct RiscVCodegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> RiscVCodegen<'a> {
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

    /// Emit assembly for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_riscv(module, writer, self.base.target_os)
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

    fn capabilities() -> CapabilitySet {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            CodegenCapability::Print,
            CodegenCapability::StackAllocation,
            CodegenCapability::MemoryOperations,
            CodegenCapability::SystemCalls,
            CodegenCapability::InlineAssembly,
            CodegenCapability::ForeignFunctionInterface,
        ]
        .into_iter()
        .collect()
    }

    fn prepare(
        &mut self,
        types: &std::collections::HashMap<String, crate::mir::MirType>,
        globals: &std::collections::HashMap<String, crate::mir::Global>,
        funcs: &std::collections::HashMap<String, crate::mir::Signature>,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError> {
        self.base
            .prepare_base(types, globals, funcs, verbose, options, input_name)
    }

    fn compile(&mut self) -> Result<(), CodegenError> {
        self.base.compile_base()
    }

    fn finalize(&mut self) -> Result<(), CodegenError> {
        self.base.finalize_base()
    }

    fn emit_asm(&mut self) -> Result<(), CodegenError> {
        self.base.emit_asm_base(generate_mir_riscv, "RISC-V")
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

    // Emit external function declarations first
    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {}", label)?;
    }

    for (func_name, func) in &module.functions {
        // Skip external functions - they're already declared above
        if module.is_external(func_name) {
            continue;
        }
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
                    && !stack_slots.contains_key(vreg)
                {
                    stack_slots.insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot));
                    next_slot += 1;
                }
                // Also check for registers used in operands
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg
                        && !stack_slots.contains_key(vreg)
                    {
                        stack_slots.insert(*vreg, RiscVFrame::calculate_stack_offset(next_slot));
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
                emit_instruction_riscv(inst, writer, &mut reg_alloc, &stack_slots, target_os)?;
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
            // Load lhs to a0
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

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
            }

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
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
            load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "a0")?;

            // Load rhs to a1
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "a1")?;

            // Perform comparison
            emit_int_cmp_op(op, writer)?;

            // Store result
            if let Register::Virtual(vreg) = dst {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Call { name, args, ret } => {
            let abi = RiscVAbi::new(target_os);

            // Handle print intrinsic
            if name == "print" {
                if let Some(arg) = args.first() {
                    // Load format string address to a0
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            writeln!(writer, "    la a0, __mir_fmt_int")?;
                        }
                        _ => {
                            writeln!(writer, "    la a0, .L_mir_fmt_int")?;
                        }
                    }
                    // Load value to print to a1
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, "a1")?;
                    // Call printf
                    let printf_name = abi.call_stub("print").unwrap_or_else(|| match target_os {
                        TargetOperatingSystem::MacOS => "_printf".to_string(),
                        _ => "printf".to_string(),
                    });
                    writeln!(writer, "    call {}", printf_name)?;
                }
            } else {
                // General function call implementation
                // RISC-V calling convention: first 8 args in a0-a7, remaining on stack

                let arg_regs = RiscVAbi::ARG_REGISTERS;
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                // Pass register arguments (a0-a7)
                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
                }

                // Pass stack arguments (16-byte aligned)
                let stack_space = if num_stack_args > 0 {
                    // Align to 16 bytes
                    ((num_stack_args * 8) + 15) & !15
                } else {
                    0
                };

                if stack_space > 0 {
                    // Allocate stack space
                    writeln!(writer, "    addi sp, sp, -{}", stack_space)?;

                    // Store arguments on stack (in order, starting at sp+0)
                    for (i, arg) in args.iter().skip(num_reg_args).enumerate() {
                        let offset = i * 8;
                        // Load argument to a temporary register (use t0)
                        load_operand_to_register(arg, writer, reg_alloc, stack_slots, "t0")?;
                        // Store to stack
                        writeln!(writer, "    sd t0, {}(sp)", offset)?;
                    }
                }

                // Resolve function name (check for intrinsic stubs first)
                let target_sym = if let Some(stub) = abi.call_stub(name) {
                    stub
                } else {
                    abi.mangle_function_name(name)
                };

                // Emit call instruction
                writeln!(writer, "    call {}", target_sym)?;

                // Clean up stack arguments
                if stack_space > 0 {
                    writeln!(writer, "    addi sp, sp, {}", stack_space)?;
                }
            }

            // Handle return value (always in a0)
            if let Some(ret_reg) = ret
                && let Register::Virtual(vreg) = ret_reg
            {
                store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty: _,
            attrs: _,
        } => {
            match addr {
                crate::mir::instruction::AddressMode::BaseOffset { base, offset } => {
                    // Load base address into t0
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "t0")?
                        }
                        Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
                    }

                    // Load value from [t0 + offset] into a0
                    // TODO: Handle different types (lw vs ld) based on ty
                    writeln!(writer, "    ld a0, {}(t0)", offset)?;

                    // Store a0 into dst
                    if let Register::Virtual(vreg) = dst {
                        store_register_to_register("a0", vreg, writer, reg_alloc, stack_slots)?;
                    }
                }
                _ => writeln!(writer, "    # TODO: complex addressing modes for Load")?,
            }
        }
        MirInst::Store {
            addr,
            src,
            ty: _,
            attrs: _,
        } => {
            // Load value to store into a0
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "a0")?;

            match addr {
                crate::mir::instruction::AddressMode::BaseOffset { base, offset } => {
                    // Load base address into t0
                    match base {
                        Register::Virtual(v) => {
                            load_register_to_register(v, writer, reg_alloc, stack_slots, "t0")?
                        }
                        Register::Physical(p) => writeln!(writer, "    mv t0, {}", p.name)?,
                    }

                    // Store a0 into [t0 + offset]
                    // TODO: Handle different types (sw vs sd) based on ty
                    writeln!(writer, "    sd a0, {}(t0)", offset)?;
                }
                _ => writeln!(writer, "    # TODO: complex addressing modes for Store")?,
            }
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                load_operand_to_register(val, writer, reg_alloc, stack_slots, "a0")?;
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
                load_register_to_register(vreg, writer, reg_alloc, stack_slots, "t0")?;
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
