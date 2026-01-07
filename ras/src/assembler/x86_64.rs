//! x86_64 architecture-specific code generation
//!
//! This module handles compilation of MIR to x86_64 binary machine code.

use crate::assembler::core::RasAssembler;
use crate::error::RasError;

/// Compile MIR to binary for x86_64
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
pub fn compile_mir_x86_64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    function_name: Option<&str>,
) -> Result<Vec<u8>, RasError> {
    use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
    use lamina_mir::Register;

    let abi = X86ABI::new(assembler.target_os);
    let mut code = Vec::new();

    let functions_to_compile: Vec<_> = if let Some(name) = function_name {
        module.functions.iter()
            .filter(|(n, _)| *n == name)
            .collect()
    } else {
        module.functions.iter().collect()
    };

    for (func_name, func) in functions_to_compile {
        let mut reg_alloc = X64RegAlloc::new(assembler.target_os);
        let mut stack_slots: std::collections::HashMap<lamina_mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        let mut def_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();
        let mut used_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();

        // Collect register usage (reuse logic from mir_codegen)
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

        // Allocate stack slots (reuse logic from mir_codegen)
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

        // Generate function prologue (binary encoded)
        let prologue = encode_prologue_x86_64(stack_size as u32)?;
        code.extend_from_slice(&prologue);

        // Handle function parameters (reuse ABI logic)
        if !func.sig.params.is_empty() {
            let arg_regs = abi.arg_registers();
            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        // MOV from argument register to stack slot
                        let mov_bytes = encode_mov_reg_mem_x86_64(
                            arg_regs[index],
                            *slot_off,
                        )?;
                        code.extend_from_slice(&mov_bytes);
                    } else {
                        // Handle stack arguments
                        let caller_off = 16 + ((index - arg_regs.len()) as i32) * 8;
                        let mov1 = encode_mov_mem_reg_x86_64(caller_off, "rax")?;
                        code.extend_from_slice(&mov1);
                        let mov2 = encode_mov_reg_mem_x86_64("rax", *slot_off)?;
                        code.extend_from_slice(&mov2);
                    }
                }
            }
        }

        // Compile each block
        for block in &func.blocks {
            for inst in &block.instructions {
                let inst_bytes = encode_mir_instruction_x86_64(
                    inst,
                    &mut reg_alloc,
                    &stack_slots,
                    stack_size,
                    func_name,
                )?;
                code.extend_from_slice(&inst_bytes);
            }
        }

        // Generate function epilogue
        let epilogue = encode_epilogue_x86_64()?;
        code.extend_from_slice(&epilogue);

        // RET instruction
        code.push(0xC3);
    }

    Ok(code)
}

/// Encode x86_64 prologue
fn encode_prologue_x86_64(stack_size: u32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    // push rbp: 55
    code.push(0x55);
    // mov rbp, rsp: 48 89 E5
    code.extend_from_slice(&[0x48, 0x89, 0xE5]);
    // sub rsp, imm32: 48 83 EC id
    if stack_size > 0 {
        code.push(0x48);
        code.push(0x83);
        code.push(0xEC);
        code.extend_from_slice(&stack_size.to_le_bytes());
    }
    Ok(code)
}

/// Encode x86_64 epilogue
fn encode_epilogue_x86_64() -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    // mov rsp, rbp: 48 89 EC
    code.extend_from_slice(&[0x48, 0x89, 0xEC]);
    // pop rbp: 5D
    code.push(0x5D);
    Ok(code)
}

/// Encode MOV register to memory
fn encode_mov_reg_mem_x86_64(
    src_reg: &str,
    offset: i32,
) -> Result<Vec<u8>, RasError> {
    // MOV [rbp+offset], reg: REX.W + 89 /r
    let mut code = Vec::new();
    let reg_enc = parse_register_x86_64(src_reg)?;
    code.push(0x48); // REX.W
    code.push(0x89);
    // ModR/M: [rbp] + reg encoding
    // Mod=01 (disp8/32), R/M=101 (rbp), REG=reg_enc
    code.push(0x45 | (reg_enc << 3));
    // For 32-bit offset, we need 4 bytes
    if (-128..=127).contains(&offset) {
        code.push(offset as u8);
    } else {
        code.push(0x85 | (reg_enc << 3)); // Mod=10 (disp32), R/M=101
        code.extend_from_slice(&offset.to_le_bytes());
    }
    Ok(code)
}

/// Encode MOV memory to register
fn encode_mov_mem_reg_x86_64(
    offset: i32,
    dst_reg: &str,
) -> Result<Vec<u8>, RasError> {
    // MOV reg, [rbp+offset]: REX.W + 8B /r
    let mut code = Vec::new();
    let reg_enc = parse_register_x86_64(dst_reg)?;
    code.push(0x48); // REX.W
    code.push(0x8B);
    // ModR/M: [rbp] + reg encoding
    if (-128..=127).contains(&offset) {
        code.push(0x45 | (reg_enc << 3)); // Mod=01 (disp8)
        code.push(offset as u8);
    } else {
        code.push(0x85 | (reg_enc << 3)); // Mod=10 (disp32)
        code.extend_from_slice(&offset.to_le_bytes());
    }
    Ok(code)
}

/// Parse register name to encoding (x86_64)
fn parse_register_x86_64(reg: &str) -> Result<u8, RasError> {
    let reg = reg.trim_start_matches('%');
    match reg {
        "rax" => Ok(0),
        "rcx" => Ok(1),
        "rdx" => Ok(2),
        "rbx" => Ok(3),
        "rsp" => Ok(4),
        "rbp" => Ok(5),
        "rsi" => Ok(6),
        "rdi" => Ok(7),
        _ => Err(RasError::EncodingError(format!("Unknown register: {}", reg))),
    }
}

/// Encode MIR instruction to binary (x86_64)
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
fn encode_mir_instruction_x86_64(
    inst: &lamina_mir::Instruction,
    _reg_alloc: &mut lamina_codegen::x86_64::X64RegAlloc,
    _stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _stack_size: usize,
    _func_name: &str,
) -> Result<Vec<u8>, RasError> {
    // TODO: Implement full MIR instruction encoding
    // For now, return placeholder
    match inst {
        lamina_mir::Instruction::Ret { value: None } => {
            Ok(vec![0xC3]) // RET
        }
        _ => Err(RasError::EncodingError(
            "MIR instruction encoding not yet fully implemented".to_string(),
        )),
    }
}





