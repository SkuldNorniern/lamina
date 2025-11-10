mod regalloc;

use regalloc::X64RegAlloc;
use std::io::Write;
use std::result::Result;

use crate::mir_codegen::{Codegen, CodegenError, CodegenOptions, TargetOs};
use crate::mir::{Instruction as MirInst, Module as MirModule, Register};

pub fn generate_mir_x86_64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOs,
) -> Result<(), crate::error::LaminaError> {
    // Emit format strings for print intrinsics
    match target_os {
        TargetOs::MacOs => {
            writeln!(writer, ".section __TEXT,__cstring,cstring_literals")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        TargetOs::Linux => {
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
    writeln!(writer, ".globl main")?; // Will be adjusted per platform in function loop

    for (func_name, func) in &module.functions {
        // Function label
        let label = if target_os == TargetOs::MacOs && func_name == "main" {
            "_main".to_string()
        } else if target_os == TargetOs::MacOs {
            format!("_{}", func_name)
        } else {
            func_name.to_string()
        };

        writeln!(writer, "{}:", label)?;

        // Prologue: save callee-saved registers and set up frame
        writeln!(writer, "    pushq %rbp")?;
        writeln!(writer, "    movq %rsp, %rbp")?;

        // Create register allocator for this function
        let mut reg_alloc = X64RegAlloc::new();

        // Allocate stack space for virtual registers
        let mut stack_slots: std::collections::HashMap<crate::mir::VirtualReg, i32> = std::collections::HashMap::new();
        let mut stack_offset = -8i32;

        // Assign stack slots to all virtual registers used in the function
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg() {
                    if let Register::Virtual(vreg) = dst {
                        if !stack_slots.contains_key(&vreg) {
                            stack_slots.insert(*vreg, stack_offset);
                            stack_offset -= 8;
                        }
                    }
                }
                // Also check for registers used in operands
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        if !stack_slots.contains_key(&vreg) {
                            stack_slots.insert(*vreg, stack_offset);
                            stack_offset -= 8;
                        }
                    }
                }
            }
        }

        // Allocate stack space
        let stack_size = if stack_offset < -8 { (-stack_offset) as u32 } else { 0 };
        if stack_size > 0 {
            writeln!(writer, "    subq ${}, %rsp", stack_size)?;
        }

        // Process each block
        for block in &func.blocks {
            writeln!(writer, ".L_{}:", block.label)?;

            for inst in &block.instructions {
                emit_instruction_x86_64(inst, writer, &mut reg_alloc, &stack_slots, stack_size, target_os)?;
            }
        }
    }

    Ok(())
}

fn load_register_to_rax(
    reg: &Register,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
) -> Result<(), crate::error::LaminaError> {
    match reg {
        Register::Physical(p) => {
            writeln!(writer, "    movq %{}, %rax", p.name)?;
        }
        Register::Virtual(v) => {
            if let Some(phys) = reg_alloc.get_mapping_for(v) {
                writeln!(writer, "    movq %{}, %rax", phys)?;
            } else if let Some(offset) = stack_slots.get(v) {
                writeln!(writer, "    movq {}(%rbp), %rax", offset)?;
            } else {
                writeln!(writer, "    # ERROR: no mapping for virtual register")?;
            }
        }
    }
    Ok(())
}

fn store_rax_to_register(
    reg: &Register,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
) -> Result<(), crate::error::LaminaError> {
    match reg {
        Register::Physical(p) => {
            writeln!(writer, "    movq %rax, %{}", p.name)?;
        }
        Register::Virtual(v) => {
            if let Some(phys) = reg_alloc.get_mapping_for(v) {
                writeln!(writer, "    movq %rax, %{}", phys)?;
            } else if let Some(offset) = stack_slots.get(v) {
                writeln!(writer, "    movq %rax, {}(%rbp)", offset)?;
            } else {
                writeln!(writer, "    # ERROR: no mapping for virtual register")?;
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
    stack_size: u32,
    target_os: TargetOs,
) -> Result<(), crate::error::LaminaError> {
    match inst {
        MirInst::IntBinary { op, dst, lhs, rhs, ty: _ } => {
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
            store_rax_to_register(dst, writer, reg_alloc, stack_slots)?;

            // Free scratch register
            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::IntCmp { op, dst, lhs, rhs, ty: _ } => {
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
            store_rax_to_register(dst, writer, reg_alloc, stack_slots)?;

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
                    if target_os == TargetOs::MacOs {
                        writeln!(writer, "    call _printf")?;
                    } else {
                        writeln!(writer, "    xorl %eax, %eax")?;
                        writeln!(writer, "    call printf")?;
                    }
                }
            } else {
                writeln!(writer, "    # TODO: function calls")?;
            }

            if let Some(ret_reg) = ret {
                // For now, assume return value is in rax
                store_rax_to_register(ret_reg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Load { dst, addr, ty: _, attrs: _ } => {
            // Simple direct load for now - assume addr is BaseOffset with offset 0
            if let crate::mir::AddressMode::BaseOffset { base, offset: 0 } = addr {
                load_register_to_rax(base, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movq (%rax), %rax")?;
                store_rax_to_register(dst, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Store { addr, src, ty: _, attrs: _ } => {
            // Simple direct store for now
            load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
            if let crate::mir::AddressMode::BaseOffset { base, offset: 0 } = addr {
                let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                load_register_to_register(base, writer, reg_alloc, stack_slots, scratch)?;
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
            if stack_size > 0 {
                writeln!(writer, "    addq ${}, %rsp", stack_size)?;
            }
            writeln!(writer, "    popq %rbp")?;
            writeln!(writer, "    ret")?;
        }
        MirInst::Jmp { target } => {
            writeln!(writer, "    jmp .L_{}", target)?;
        }
        MirInst::Br { cond, true_target, false_target } => {
            // Load condition to register
            load_register_to_rax(cond, writer, reg_alloc, stack_slots)?;
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

fn load_operand_to_rax(
    operand: &crate::mir::Operand,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
) -> Result<(), crate::error::LaminaError> {
    match operand {
        crate::mir::Operand::Register(reg) => {
            load_register_to_rax(reg, writer, reg_alloc, stack_slots)?;
        }
        crate::mir::Operand::Immediate(imm) => {
            match imm {
                crate::mir::Immediate::I64(val) => {
                    writeln!(writer, "    movq ${}, %rax", val)?;
                }
                _ => writeln!(writer, "    # TODO: other immediate types")?,
            }
        }
        _ => writeln!(writer, "    # TODO: other operand types")?,
    }
    Ok(())
}

fn load_operand_to_register(
    operand: &crate::mir::Operand,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), crate::error::LaminaError> {
    match operand {
        crate::mir::Operand::Register(reg) => {
            load_register_to_register(reg, writer, reg_alloc, stack_slots, target_reg)?;
        }
        crate::mir::Operand::Immediate(imm) => {
            match imm {
                crate::mir::Immediate::I64(val) => {
                    writeln!(writer, "    movq ${}, %{}", val, target_reg)?;
                }
                _ => writeln!(writer, "    # TODO: other immediate types")?,
            }
        }
        _ => writeln!(writer, "    # TODO: other operand types")?,
    }
    Ok(())
}

fn load_register_to_register(
    reg: &Register,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &std::collections::HashMap<crate::mir::VirtualReg, i32>,
    target_reg: &str,
) -> Result<(), crate::error::LaminaError> {
    match reg {
        Register::Physical(p) => {
            writeln!(writer, "    movq %{}, %{}", p.name, target_reg)?;
        }
        Register::Virtual(v) => {
            if let Some(phys) = reg_alloc.get_mapping_for(v) {
                writeln!(writer, "    movq %{}, %{}", phys, target_reg)?;
            } else if let Some(offset) = stack_slots.get(v) {
                writeln!(writer, "    movq {}(%rbp), %{}", offset, target_reg)?;
            } else {
                writeln!(writer, "    # ERROR: no mapping for virtual register")?;
            }
        }
    }
    Ok(())
}