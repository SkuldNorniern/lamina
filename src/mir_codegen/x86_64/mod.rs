//! x86_64 code generation for MIR (Mid-level IR).
//!
//! This module provides code generation from MIR to x86_64 assembly,
//! supporting System V AMD64 and Microsoft x64 calling conventions.

pub mod abi;
pub mod constants;
pub mod frame;
pub mod regalloc;
pub mod util;

use abi::X86ABI;
use constants::{fd, linux, macos, stack, windows};
use frame::X86Frame;
use regalloc::X64RegAlloc;
use std::io::Write;
use std::result::Result;
use util::*;

use crate::error::LaminaError;
use crate::mir::{Instruction as MirInst, MirType, Module as MirModule, Register};
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions,
    capability::{CapabilitySet, CodegenCapability},
};
use lamina_platform::TargetOperatingSystem;

use crate::mir_codegen::common::CodegenBase;

/// MIR to x86_64 code generator implementing the `Codegen` trait.
pub struct X86Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> X86Codegen<'a> {
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
        codegen_units: usize,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_x86_64_with_units(module, writer, self.base.target_os, codegen_units)
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
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_x86_64_with_units(module, writer, target_os, codegen_units)
            },
            "x86_64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not supported".to_string(),
        ))
    }
}

use crate::mir_codegen::common::{compile_functions_parallel, emit_print_format_section};

fn compile_single_function_x86_64(
    func_name: &str,
    func: &crate::mir::Function,
    target_os: TargetOperatingSystem,
) -> Result<Vec<u8>, crate::mir_codegen::CodegenError> {
    use std::io::Write;
    let mut output = Vec::new();
    let abi = X86ABI::new(target_os);

    ensure_signature_support(&func.sig).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
    })?;
    
        let label = abi.mangle_function_name(func_name);
    writeln!(output, "{}:", label).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

        let mut reg_alloc = X64RegAlloc::new(target_os);

        let mut stack_slots: std::collections::HashMap<crate::mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        let mut def_regs: std::collections::HashSet<crate::mir::VirtualReg> =
            std::collections::HashSet::new();
        let mut used_regs: std::collections::HashSet<crate::mir::VirtualReg> =
            std::collections::HashSet::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                {
                    def_regs.insert(*vreg);
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        used_regs.insert(*vreg);
                    }
                }
            }
        }

        for vreg in &def_regs {
            if !stack_slots.contains_key(vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(*vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }
        for vreg in used_regs {
            if !def_regs.contains(&vreg) && !stack_slots.contains_key(&vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }

        let stack_size = stack_slots.len() * 8;
    X86Frame::generate_prologue(&mut output, stack_size).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
    })?;

        if !func.sig.params.is_empty() {
            let abi_for_func = X86ABI::new(target_os);
            let arg_regs = abi_for_func.arg_registers();

            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        let phys_arg = arg_regs[index];
                    writeln!(output, "    movq %{}, {}(%rbp)", phys_arg, slot_off).map_err(|e| {
                        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                    })?;
                    } else {
                        let stack_index = index - arg_regs.len();
                        let caller_off = 16 + (stack_index as i32) * 8;
                    writeln!(output, "    movq {}(%rbp), %rax", caller_off).map_err(|e| {
                        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                    })?;
                    writeln!(output, "    movq %rax, {}(%rbp)", slot_off).map_err(|e| {
                        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                    })?;
                    }
                }
            }
        }

    writeln!(output, "    jmp .L_{}_{}", func_name, func.entry).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

        for block in &func.blocks {
        writeln!(output, ".L_{}_{}:", func_name, block.label).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;

            for inst in &block.instructions {
                emit_instruction_x86_64(
                    inst,
                &mut output,
                    &mut reg_alloc,
                    &stack_slots,
                    stack_size,
                    target_os,
                    func_name,
                    &def_regs,
            ).map_err(|e| {
                crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
            })?;
            }
        }

    Ok(output)
}

pub fn generate_mir_x86_64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    generate_mir_x86_64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_x86_64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), crate::error::LaminaError> {
    let abi = X86ABI::new(target_os);

    emit_print_format_section(writer, target_os)?;

    // Emit external function declarations first
    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {}", label)?;
    }

    writeln!(writer, ".text")?;
    writeln!(writer, "{}", abi.get_main_global())?;

    let results = compile_functions_parallel(
        module,
        target_os,
        codegen_units,
        compile_single_function_x86_64,
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

    Ok(())
}

fn ensure_signature_support(sig: &crate::mir::Signature) -> Result<(), LaminaError> {
    for (idx, param) in sig.params.iter().enumerate() {
        ensure_type_supported(&param.ty, &format!("parameter {} of '{}'", idx, sig.name))?;
    }

    if let Some(ret_ty) = &sig.ret_ty {
        ensure_type_supported(ret_ty, &format!("return type of '{}'", sig.name))?;
    }

    Ok(())
}

fn ensure_type_supported(ty: &MirType, context: &str) -> Result<(), LaminaError> {
    if ty.is_float() || ty.is_vector() {
        return Err(LaminaError::ValidationError(format!(
            "x86_64 backend does not support {} (type {})",
            context, ty
        )));
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
    func_name: &str,
    def_regs: &std::collections::HashSet<crate::mir::VirtualReg>,
) -> Result<(), crate::error::LaminaError> {
    match inst {
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;

            match op {
                crate::mir::IntBinOp::Shl
                | crate::mir::IntBinOp::AShr
                | crate::mir::IntBinOp::LShr => match rhs {
                    crate::mir::Operand::Immediate(imm) => {
                        let shift_val = match imm {
                            crate::mir::instruction::Immediate::I8(v) => *v as u64,
                            crate::mir::instruction::Immediate::I16(v) => *v as u64,
                            crate::mir::instruction::Immediate::I32(v) => *v as u64,
                            crate::mir::instruction::Immediate::I64(v) => *v as u64,
                            _ => {
                                return Err(LaminaError::ValidationError(
                                    "Shift count must be an integer immediate".to_string(),
                                ));
                            }
                        };
                        match op {
                            crate::mir::IntBinOp::Shl => {
                                writeln!(writer, "    shlq ${}, %rax", shift_val)?
                            }
                            crate::mir::IntBinOp::AShr => {
                                writeln!(writer, "    sarq ${}, %rax", shift_val)?
                            }
                            crate::mir::IntBinOp::LShr => {
                                writeln!(writer, "    shrq ${}, %rax", shift_val)?
                            }
                            _ => unreachable!(),
                        }
                    }
                    crate::mir::Operand::Register(_) => {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
                        writeln!(writer, "    movq %{}, %rcx", scratch)?;
                        match op {
                            crate::mir::IntBinOp::Shl => writeln!(writer, "    shlq %cl, %rax")?,
                            crate::mir::IntBinOp::AShr => writeln!(writer, "    sarq %cl, %rax")?,
                            crate::mir::IntBinOp::LShr => writeln!(writer, "    shrq %cl, %rax")?,
                            _ => unreachable!(),
                        }
                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    }
                },
                _ => {
                    let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;

                    match op {
                        crate::mir::IntBinOp::Add => {
                            writeln!(writer, "    addq %{}, %rax", scratch)?
                        }
                        crate::mir::IntBinOp::Sub => {
                            writeln!(writer, "    subq %{}, %rax", scratch)?
                        }
                        crate::mir::IntBinOp::Mul => {
                            writeln!(writer, "    imulq %{}, %rax", scratch)?
                        }
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
                        crate::mir::IntBinOp::And => {
                            writeln!(writer, "    andq %{}, %rax", scratch)?
                        }
                        crate::mir::IntBinOp::Or => writeln!(writer, "    orq %{}, %rax", scratch)?,
                        crate::mir::IntBinOp::Xor => {
                            writeln!(writer, "    xorq %{}, %rax", scratch)?
                        }
                        _ => unreachable!(),
                    }

                    if scratch != "rbx" {
                        reg_alloc.free_scratch(scratch);
                    }
                }
            }

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty: _,
        } => {
            load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
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
                #[allow(unreachable_patterns)]
                other => {
                    return Err(LaminaError::ValidationError(format!(
                        "x86_64 backend does not support integer comparison {:?}",
                        other
                    )));
                }
            }
            writeln!(writer, "    movzbq %al, %rax")?;

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }

            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "print" {
                if let Some(arg) = args.first() {
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    call _printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call _fflush")?;
                        }
                        TargetOperatingSystem::Windows => {
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rcx")?;
                            writeln!(writer, "    movq %rax, %rdx")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    movq $0, %rcx")?;
                            writeln!(writer, "    call fflush")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        }
                        _ => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call fflush")?;
                        }
                    }
                }
            } else if name == "writebyte" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writebyte requires one argument".to_string())
                })?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movb %al, (%rsp)")?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;
            } else {
                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                if target_os == TargetOperatingSystem::Windows {
                    let shadow_space = if num_reg_args > 0 {
                        windows::SHADOW_SPACE_SIZE as usize
                    } else {
                        0
                    };
                    let total_stack = shadow_space + (num_stack_args * stack::SLOT_SIZE);
                    if total_stack > 0 {
                        writeln!(writer, "    subq ${}, %rsp", total_stack)?;
                    }
                    for i in 0..num_stack_args {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        let stack_offset = shadow_space + (i * stack::SLOT_SIZE);
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, {}(%rsp)", stack_offset)?;
                    }
                } else {
                    for i in (0..num_stack_args).rev() {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    pushq %rax")?;
                    }
                }

                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
                }

                let mangled_name = abi.mangle_function_name(name);
                writeln!(writer, "    call {}", mangled_name)?;

                if target_os == TargetOperatingSystem::Windows {
                    let shadow_space = if num_reg_args > 0 {
                        windows::SHADOW_SPACE_SIZE as usize
                    } else {
                        0
                    };
                    let total_stack = shadow_space + (num_stack_args * stack::SLOT_SIZE);
                    if total_stack > 0 {
                        writeln!(writer, "    addq ${}, %rsp", total_stack)?;
                    }
                } else if num_stack_args > 0 {
                    writeln!(
                        writer,
                        "    addq ${}, %rsp",
                        num_stack_args * stack::SLOT_SIZE
                    )?;
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::Load {
            dst,
            addr,
            ty: _,
            attrs: _,
        } => {
            if let crate::mir::AddressMode::BaseOffset { base, offset: 0 } = addr {
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    }
                }
                writeln!(writer, "    movq (%rax), %rax")?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support load address mode {:?}",
                    addr
                )));
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
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_register(vreg, writer, reg_alloc, stack_slots, scratch)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %{}", phys.name, scratch)?;
                    }
                }
                writeln!(writer, "    movq %rax, (%{})", scratch)?;
                if scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support store address mode {:?}",
                    addr
                )));
            }
        }
        MirInst::TailCall { name, args } => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());

            // 1. Handle Register Arguments
            for i in 0..num_reg_args {
                let arg = &args[i];
                let dest_reg = arg_regs[i];
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
            }

            // 2. Handle Stack Arguments (Overwrite incoming args)
            // Note: TailCallOptimization ensures args.len() == current_func.args.len()
            // So we can strictly overwrite our own incoming stack slots.
            if num_stack_args > 0 {
                let shadow_space = if target_os == TargetOperatingSystem::Windows {
                    windows::SHADOW_SPACE_SIZE as usize
                } else {
                    0
                };

                for i in 0..num_stack_args {
                    let arg_idx = num_reg_args + i;
                    let arg = &args[arg_idx];

                    // Incoming args are at RBP + 16 + shadow + i*8
                    // (Return address is 8, saved RBP pushed, so RBP points to saved RBP)
                    // Wait:
                    // Standard Prologue: push rbp; mov rbp, rsp
                    // Stack:
                    //   [RBP + 16 + shadow + 8*i] -> Arg N (Stack Arg i)
                    //   [RBP + 8]  -> Return Address
                    //   [RBP + 0]  -> Saved RBP
                    //
                    // So first stack arg is at RBP + 16 (on Linux/Mac) or RBP + 16 + 32 (Windows? No shadow is allocated by caller?)
                    // Windows: Shadow space is 32 bytes allocated *by caller* right before return address?
                    // No, shadow space is allocated by caller *above* return address?
                    // Microsoft x64: "The caller allocates space for 4 register parameters..."
                    // Stack: [RetAddr] [Home P1] [Home P2] [Home P3] [Home P4] [Stack Arg 5] ...
                    // Wait, Home space is "Shadow Space". It's strictly for the first 4 args (which are in regs).
                    // Stack args start *after* shadow space.
                    // So RBP + 16 + 32 + i*8.

                    let offset = 16 + shadow_space + (i * stack::SLOT_SIZE);

                    // Load to RAX (scratch) first
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;

                    // Store to incoming slot
                    writeln!(writer, "    movq %rax, {}(%rbp)", offset)?;
                }
            }

            // 3. Teardown Frame (Epilogue without ret)
            // Restore Stack Pointer
            writeln!(writer, "    movq %rbp, %rsp")?;
            // Restore Base Pointer
            writeln!(writer, "    popq %rbp")?;

            // 4. Jump to target
            let mangled_name = abi.mangle_function_name(name);
            writeln!(writer, "    jmp {}", mangled_name)?;
        }
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(vreg) => {
                    let is_placeholder = !def_regs.contains(vreg);

                    if is_placeholder {
                        if let Some(slot_offset) = stack_slots.get(vreg) {
                            let total_offset = *slot_offset as i64 + (*offset as i64);
                            if total_offset == 0 {
                                writeln!(writer, "    leaq (%rbp), %rax")?;
                            } else {
                                writeln!(writer, "    leaq {}(%rbp), %rax", total_offset)?;
                            }
                        } else {
                            return Err(LaminaError::ValidationError(format!(
                                "LEA placeholder base vreg {:?} has no stack slot",
                                vreg
                            )));
                        }
                    } else if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        if *offset == 0 {
                            writeln!(writer, "    movq %{}, %rax", phys)?;
                        } else {
                            writeln!(writer, "    leaq {}(%{}), %rax", offset, phys)?;
                        }
                    } else if let Some(slot_offset) = stack_slots.get(vreg) {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                        writeln!(writer, "    movq {}(%rbp), %{}", slot_offset, scratch)?;
                        if *offset == 0 {
                            writeln!(writer, "    movq %{}, %rax", scratch)?;
                        } else {
                            writeln!(writer, "    leaq {}(%{}), %rax", offset, scratch)?;
                        }
                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    } else {
                        return Err(LaminaError::ValidationError(format!(
                            "x86_64 backend cannot lower LEA for base vreg {:?} (no mapping/slot)",
                            vreg
                        )));
                    }
                }
                Register::Physical(phys) => {
                    if *offset == 0 {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    } else {
                        writeln!(writer, "    leaq {}(%{}), %rax", offset, phys.name)?;
                    }
                }
            }

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
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
            // Use function-scoped label to avoid collisions
            writeln!(writer, "    jmp .L_{}_{}", func_name, target)?;
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
            }
            writeln!(writer, "    testq %rax, %rax")?;
            // Use function-scoped labels to avoid collisions
            writeln!(writer, "    jnz .L_{}_{}", func_name, true_target)?;
            writeln!(writer, "    jmp .L_{}_{}", func_name, false_target)?;
        }
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support MIR instruction {:?}",
                other
            )));
        }
    }

    Ok(())
}
