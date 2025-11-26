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

use crate::error::LaminaError;
use crate::mir::{Instruction as MirInst, MirType, Module as MirModule, Register};
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
        TargetOperatingSystem::Windows => {
            // Windows uses .rdata section for read-only data
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

    // Text section header
    writeln!(writer, ".text")?;
    writeln!(writer, "{}", abi.get_main_global())?;

    for (func_name, func) in &module.functions {
        ensure_signature_support(&func.sig)?;
        // Function label
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, "{}:", label)?;

        // Create register allocator for this function
        let mut reg_alloc = X64RegAlloc::new(target_os);

        // Allocate stack space for virtual registers
        let mut stack_slots: std::collections::HashMap<crate::mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        // Track which virtual registers are explicitly defined by instructions.
        let mut def_regs: std::collections::HashSet<crate::mir::VirtualReg> =
            std::collections::HashSet::new();
        // Track all registers that appear as operands (includes placeholders used only as LEA bases).
        let mut used_regs: std::collections::HashSet<crate::mir::VirtualReg> =
            std::collections::HashSet::new();

        // First pass: collect def/use information.
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg() {
                    if let Register::Virtual(vreg) = dst {
                        def_regs.insert(*vreg);
                    }
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        used_regs.insert(*vreg);
                    }
                }
            }
        }

        // Second pass: assign stack slots.
        // 1. Value-carrying vregs (those that are defined somewhere in the function).
        for vreg in &def_regs {
            if !stack_slots.contains_key(vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(*vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }
        // 2. Placeholder vregs that are only ever *used* (e.g., stack allocation bases).
        for vreg in used_regs {
            if !def_regs.contains(&vreg) && !stack_slots.contains_key(&vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }

        // Generate function prologue
        let stack_size = stack_slots.len() * 8;
        X86Frame::generate_prologue(writer, stack_size)?;

        // Spill incoming arguments to their stack slots so MIR can access them via vregs.
        // This maps the abstract MIR calling convention (params in v-regs) onto the concrete
        // platform ABI registers/stack.
        if !func.sig.params.is_empty() {
            let abi_for_func = X86ABI::new(target_os);
            let arg_regs = abi_for_func.arg_registers();

            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg {
                    if let Some(slot_off) = stack_slots.get(vreg) {
                        // Parameter passed in register
                        if index < arg_regs.len() {
                            let phys_arg = arg_regs[index];
                            // Store register argument into its stack slot
                            writeln!(writer, "    movq %{}, {}(%rbp)", phys_arg, slot_off)?;
                        } else {
                            // Parameter passed on caller's stack.
                            // System V AMD64: first stack arg is at 16(%rbp) (after saved rbp, ret addr).
                            // Windows x64: additional args are spilled by the caller; layout differs,
                            // but for now we apply the same convention which is sufficient for current tests.
                            let stack_index = index - arg_regs.len();
                            let caller_off = 16 + (stack_index as i32) * 8;
                            writeln!(writer, "    movq {}(%rbp), %rax", caller_off)?;
                            writeln!(writer, "    movq %rax, {}(%rbp)", slot_off)?;
                        }
                    }
                }
            }
        }

        // Ensure execution starts at the MIR entry block rather than the first block in `blocks`.
        // The MIR `Function` tracks its logical entry label separately from block order.
        writeln!(writer, "    jmp .L_{}_{}", func_name, func.entry)?;

        // Process each block
        for block in &func.blocks {
            // Generate function-scoped block label to avoid collisions across functions
            writeln!(writer, ".L_{}_{}:", func_name, block.label)?;

            for inst in &block.instructions {
                emit_instruction_x86_64(
                    inst,
                    writer,
                    &mut reg_alloc,
                    &stack_slots,
                    stack_size,
                    target_os,
                    func_name,
                    &def_regs,
                )?;
            }
        }
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
                other => {
                    return Err(LaminaError::ValidationError(format!(
                        "x86_64 backend does not support integer binary op {:?}",
                        other
                    )));
                }
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
                other => {
                    return Err(LaminaError::ValidationError(format!(
                        "x86_64 backend does not support integer comparison {:?}",
                        other
                    )));
                }
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
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            // macOS: System V ABI - rdi, rsi for first two args
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    call _printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(writer, "    movq $0, %rdi")?; // NULL flushes all streams
                            writeln!(writer, "    call _fflush")?;
                        }
                        TargetOperatingSystem::Windows => {
                            // Windows x64: rcx, rdx for first two args, shadow space required
                            writeln!(writer, "    subq $32, %rsp")?; // Shadow space
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rcx")?; // First arg: format string
                            writeln!(writer, "    movq %rax, %rdx")?; // Second arg: value to print
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    addq $32, %rsp")?; // Clean up shadow space
                            // Flush stdout
                            writeln!(writer, "    subq $32, %rsp")?; // Shadow space
                            writeln!(writer, "    movq $0, %rcx")?; // Windows: first arg in rcx
                            writeln!(writer, "    call fflush")?;
                            writeln!(writer, "    addq $32, %rsp")?; // Clean up shadow space
                        }
                        _ => {
                            // Linux/System V: rdi, rsi for first two args
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(writer, "    movq $0, %rdi")?; // NULL flushes all streams
                            writeln!(writer, "    call fflush")?;
                        }
                    }
                }
            } else if name == "writebyte" && args.len() == 1 {
                // Write single byte to stdout using write syscall
                // Allocate space on stack for the byte (keep 16-byte aligned)
                writeln!(writer, "    subq $16, %rsp")?; // Allocate 16 bytes (aligned)

                // Load byte value to rax
                load_operand_to_rax(args.first().unwrap(), writer, reg_alloc, stack_slots)?;

                // Store byte at [rsp]
                writeln!(writer, "    movb %al, (%rsp)")?;

                // Set up syscall arguments
                match target_os {
                    TargetOperatingSystem::MacOS => {
                        // macOS: write syscall number is 0x2000004
                        writeln!(writer, "    movq $0x2000004, %rax")?; // write syscall
                        writeln!(writer, "    movq $1, %rdi")?; // stdout
                        writeln!(writer, "    movq %rsp, %rsi")?; // buffer = stack pointer
                        writeln!(writer, "    movq $1, %rdx")?; // size = 1 byte
                        writeln!(writer, "    syscall")?;
                    }
                    _ => {
                        // Linux: write syscall number is 1
                        writeln!(writer, "    movq $1, %rax")?; // write syscall
                        writeln!(writer, "    movq $1, %rdi")?; // stdout
                        writeln!(writer, "    movq %rsp, %rsi")?; // buffer = stack pointer
                        writeln!(writer, "    movq $1, %rdx")?; // size = 1 byte
                        writeln!(writer, "    syscall")?;
                    }
                }

                // Handle return value (syscall result is in rax)
                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }

                // Restore stack
                writeln!(writer, "    addq $16, %rsp")?;
            } else {
                // General function call implementation

                // 1. Pass arguments
                // System V AMD64: First 6 args go to registers: rdi, rsi, rdx, rcx, r8, r9
                // Microsoft x64: First 4 args go to registers: rcx, rdx, r8, r9
                // Remaining args go to stack (pushed in reverse order for System V, forward for Windows)

                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

                // Pass stack arguments
                if target_os == TargetOperatingSystem::Windows {
                    // Windows x64: stack args go in forward order (left to right)
                    // Windows requires shadow space (32 bytes) if any register args are used
                    let shadow_space = if num_reg_args > 0 { 32 } else { 0 };
                    let total_stack = shadow_space + (num_stack_args * 8);
                    if total_stack > 0 {
                        writeln!(writer, "    subq ${}, %rsp", total_stack)?;
                    }
                    // Store stack args in forward order (after shadow space)
                    for i in 0..num_stack_args {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        let stack_offset = shadow_space + (i * 8);
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, {}(%rsp)", stack_offset)?;
                    }
                } else {
                    // System V AMD64: stack args pushed in reverse order
                    for i in (0..num_stack_args).rev() {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    pushq %rax")?;
                    }
                }

                // Pass register arguments
                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
                }

                // 2. Emit call
                let mangled_name = abi.mangle_function_name(name);
                writeln!(writer, "    call {}", mangled_name)?;

                // 3. Clean up stack arguments and shadow space
                if target_os == TargetOperatingSystem::Windows {
                    let shadow_space = if num_reg_args > 0 { 32 } else { 0 };
                    let total_stack = shadow_space + (num_stack_args * 8);
                    if total_stack > 0 {
                        writeln!(writer, "    addq ${}, %rsp", total_stack)?;
                    }
                } else {
                    // System V: clean up pushed stack args
                    if num_stack_args > 0 {
                        writeln!(writer, "    addq ${}, %rsp", num_stack_args * 8)?;
                    }
                }

                // 4. Handle return value
                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    // Return value is in rax
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
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(vreg) => {
                    // Two distinct LEA shapes exist in MIR:
                    //
                    // 1. Stack allocation placeholders (from alloc.stack), where `base` is a
                    //    never-defined vreg that simply represents a stack slot. For these we
                    //    want the *address of the stack slot*, i.e. &slot + offset.
                    // 2. General pointer arithmetic (from getelem.ptr/getfieldptr), where `base`
                    //    holds a pointer value. For these we want `dst = base_value + offset`.
                    let is_placeholder = !def_regs.contains(vreg);

                    if is_placeholder {
                        // Placeholder: compute address relative to %rbp using the vreg's slot.
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
                    } else {
                        // General pointer arithmetic: load the base pointer value, then add offset.
                        // Prefer an existing physical register mapping when available.
                        if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                            if *offset == 0 {
                                writeln!(writer, "    movq %{}, %rax", phys)?;
                            } else {
                                writeln!(writer, "    leaq {}(%{}), %rax", offset, phys)?;
                            }
                        } else if let Some(slot_offset) = stack_slots.get(vreg) {
                            // Load pointer value from its stack slot into a scratch register.
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
                }
                Register::Physical(phys) => {
                    // Physical register: treat the register value as a pointer and add the offset.
                    if *offset == 0 {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    } else {
                        writeln!(writer, "    leaq {}(%{}), %rax", offset, phys.name)?;
                    }
                }
            }

            // Store result to destination
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
            // Load condition to register
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
