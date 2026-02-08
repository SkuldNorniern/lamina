//! x86_64 architecture-specific code generation
//!
//! This module handles compilation of MIR to x86_64 binary machine code.

use crate::assembler::core::RasAssembler;
use crate::error::RasError;
use std::collections::HashMap;

/// Compile MIR to binary for x86_64
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
pub fn compile_mir_x86_64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    function_name: Option<&str>,
) -> Result<(Vec<u8>, HashMap<String, usize>), RasError> {
    use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
    use lamina_mir::Register;

    let abi = X86ABI::new(assembler.target_os);
    let mut code = Vec::new();
    let mut function_offsets: HashMap<String, usize> = HashMap::new();

    let functions_to_compile: Vec<(String, &lamina_mir::Function)> = if let Some(name) = function_name {
        module
            .functions
            .get(name)
            .map(|f| vec![(name.to_string(), f)])
            .unwrap_or_default()
    } else {
        // Deterministic order for offset stability across runs.
        let mut names: Vec<String> = module.functions.keys().cloned().collect();
        names.sort();
        names.into_iter()
            .filter_map(|n| module.functions.get(&n).map(|f| (n, f)))
            .collect()
    };

    for (func_name, func) in functions_to_compile {
        // Current x86_64 binary encoder is intentionally minimal.
        // Refuse complex control flow for now so we don't produce nonsense binaries.
        if func.blocks.len() != 1 {
            return Err(RasError::EncodingError(format!(
                "x86_64 JIT encoder currently supports only single-block functions; '{}' has {} block(s)",
                func_name,
                func.blocks.len()
            )));
        }

        for inst in &func.blocks[0].instructions {
            match inst {
                lamina_mir::Instruction::Jmp { .. }
                | lamina_mir::Instruction::Br { .. }
                | lamina_mir::Instruction::Switch { .. }
                | lamina_mir::Instruction::Call { .. }
                | lamina_mir::Instruction::TailCall { .. } => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT encoder does not yet support control-flow/calls in '{}': {:?}",
                        func_name, inst
                    )));
                }
                _ => {}
            }
        }

        function_offsets.insert(func_name.clone(), code.len());

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
        let aligned_stack_size = (stack_size + 15) & !15;

        // Generate function prologue (binary encoded)
        let prologue = encode_prologue_x86_64(aligned_stack_size as u32)?;
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
                    aligned_stack_size,
                    &func_name,
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

    Ok((code, function_offsets))
}

/// Encode x86_64 prologue
fn encode_prologue_x86_64(stack_size: u32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    // push rbp: 55
    code.push(0x55);
    // mov rbp, rsp: 48 89 E5
    code.extend_from_slice(&[0x48, 0x89, 0xE5]);
    if stack_size > 0 {
        // Keep stack 16-byte aligned after `push rbp` (SysV/Microsoft x64 requirement).
        // Use imm8 when possible, else imm32.
        if stack_size <= 0x7f {
            // sub rsp, imm8 : 48 83 EC ib
            code.extend_from_slice(&[0x48, 0x83, 0xEC, stack_size as u8]);
        } else {
            // sub rsp, imm32 : 48 81 EC id
            code.extend_from_slice(&[0x48, 0x81, 0xEC]);
            code.extend_from_slice(&stack_size.to_le_bytes());
        }
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
    let mut code = Vec::new();
    let src = parse_register_x86_64(src_reg)?;
    // MOV [rbp+disp], r64: REX.W + 89 /r
    let rex = rex_w_r_b(true, src, 5);
    code.push(rex);
    code.push(0x89);

    if (-128..=127).contains(&offset) {
        // mod=01, r/m=101 (rbp)
        code.push(modrm(0b01, src, 0b101));
        code.push(offset as u8);
    } else {
        // mod=10, r/m=101 (rbp), disp32
        code.push(modrm(0b10, src, 0b101));
        code.extend_from_slice(&offset.to_le_bytes());
    }
    Ok(code)
}

/// Encode MOV memory to register
fn encode_mov_mem_reg_x86_64(
    offset: i32,
    dst_reg: &str,
) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let dst = parse_register_x86_64(dst_reg)?;
    // MOV r64, [rbp+disp]: REX.W + 8B /r
    let rex = rex_w_r_b(true, dst, 5);
    code.push(rex);
    code.push(0x8B);
    if (-128..=127).contains(&offset) {
        code.push(modrm(0b01, dst, 0b101));
        code.push(offset as u8);
    } else {
        code.push(modrm(0b10, dst, 0b101));
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
        "r8" => Ok(8),
        "r9" => Ok(9),
        "r10" => Ok(10),
        "r11" => Ok(11),
        "r12" => Ok(12),
        "r13" => Ok(13),
        "r14" => Ok(14),
        "r15" => Ok(15),
        _ => Err(RasError::EncodingError(format!("Unknown register: {}", reg))),
    }
}

#[inline]
fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0b11) << 6) | ((reg & 0b111) << 3) | (rm & 0b111)
}

#[inline]
fn rex_w_r_b(w: bool, reg: u8, rm: u8) -> u8 {
    0x40 | ((w as u8) << 3) | (((reg >> 3) & 1) << 2) | (((rm >> 3) & 1) << 0)
}

/// Encode MIR instruction to binary (x86_64)
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
fn encode_mir_instruction_x86_64(
    inst: &lamina_mir::Instruction,
    _reg_alloc: &mut lamina_codegen::x86_64::X64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _stack_size: usize,
    _func_name: &str,
) -> Result<Vec<u8>, RasError> {
    use lamina_mir::{Immediate, IntBinOp, IntCmpOp, Operand, Register};

    fn vreg_slot(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        reg: &lamina_mir::VirtualReg,
    ) -> Result<i32, RasError> {
        slots.get(reg).copied().ok_or_else(|| {
            RasError::EncodingError(format!("Missing stack slot for vreg {:?}", reg))
        })
    }

    fn mov_imm64(reg: u8, imm: i64) -> Vec<u8> {
        let mut code = Vec::new();
        let rex = 0x48 | (((reg >> 3) & 1) as u8);
        code.push(rex);
        code.push(0xB8 + (reg & 0b111));
        code.extend_from_slice(&(imm as i64).to_le_bytes());
        code
    }

    fn load_vreg_to_reg(slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>, v: &lamina_mir::VirtualReg, dst: &str) -> Result<Vec<u8>, RasError> {
        let off = vreg_slot(slots, v)?;
        encode_mov_mem_reg_x86_64(off, dst)
    }

    fn store_reg_to_vreg(slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>, v: &lamina_mir::VirtualReg, src: &str) -> Result<Vec<u8>, RasError> {
        let off = vreg_slot(slots, v)?;
        encode_mov_reg_mem_x86_64(src, off)
    }

    fn materialize_operand_to_reg(slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>, op: &Operand, dst_reg: &str) -> Result<Vec<u8>, RasError> {
        match op {
            Operand::Immediate(imm) => {
                let v = match imm {
                    Immediate::I8(v) => *v as i64,
                    Immediate::I16(v) => *v as i64,
                    Immediate::I32(v) => *v as i64,
                    Immediate::I64(v) => *v,
                    Immediate::F32(_) | Immediate::F64(_) => {
                        return Err(RasError::EncodingError("float immediates not supported in x86_64 JIT encoder".into()));
                    }
                };
                let reg = parse_register_x86_64(dst_reg)?;
                Ok(mov_imm64(reg, v))
            }
            Operand::Register(reg) => match reg {
                Register::Virtual(v) => load_vreg_to_reg(slots, v, dst_reg),
                Register::Physical(p) => {
                    // mov dst, phys
                    let dst = parse_register_x86_64(dst_reg)?;
                    let src = parse_register_x86_64(&p.name)?;
                    if dst == src {
                        Ok(Vec::new())
                    } else {
                        let rex = rex_w_r_b(true, src, dst);
                        Ok(vec![rex, 0x89, modrm(0b11, src, dst)])
                    }
                }
            },
        }
    }

    let mut code = Vec::new();

    match inst {
        lamina_mir::Instruction::Ret { value } => {
            if let Some(v) = value {
                // Move return value into rax.
                code.extend_from_slice(&materialize_operand_to_reg(stack_slots, v, "rax")?);
            }
            Ok(code)
        }
        lamina_mir::Instruction::IntBinary { op, dst, lhs, rhs, .. } => {
            // rax = lhs; rcx = rhs; rax = op(rax, rcx)
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, lhs, "rax")?);
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, rhs, "rcx")?);

            let rex = 0x48; // rax/rcx are low regs; REX.W only
            match op {
                IntBinOp::Add => code.extend_from_slice(&[rex, 0x01, 0xC8]), // add rax, rcx
                IntBinOp::Sub => code.extend_from_slice(&[rex, 0x29, 0xC8]), // sub rax, rcx
                IntBinOp::Mul => code.extend_from_slice(&[rex, 0x0F, 0xAF, 0xC1]), // imul rax, rcx
                IntBinOp::And => code.extend_from_slice(&[rex, 0x21, 0xC8]), // and rax, rcx
                IntBinOp::Or => code.extend_from_slice(&[rex, 0x09, 0xC8]),  // or rax, rcx
                IntBinOp::Xor => code.extend_from_slice(&[rex, 0x31, 0xC8]), // xor rax, rcx
                other => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT encoder: unsupported int binop {:?}",
                        other
                    )));
                }
            }

            if let Register::Virtual(v) = dst {
                code.extend_from_slice(&store_reg_to_vreg(stack_slots, v, "rax")?);
                Ok(code)
            } else {
                Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst registers not supported".into(),
                ))
            }
        }
        lamina_mir::Instruction::IntCmp { op, dst, lhs, rhs, .. } => {
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, lhs, "rax")?);
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, rhs, "rcx")?);

            // cmp rax, rcx
            code.extend_from_slice(&[0x48, 0x39, 0xC8]);

            let setcc = match op {
                IntCmpOp::Eq => 0x94,
                IntCmpOp::Ne => 0x95,
                IntCmpOp::SLt => 0x9C,
                IntCmpOp::SLe => 0x9E,
                IntCmpOp::SGt => 0x9F,
                IntCmpOp::SGe => 0x9D,
                IntCmpOp::ULt => 0x92, // setb
                IntCmpOp::ULe => 0x96, // setbe
                IntCmpOp::UGt => 0x97, // seta
                IntCmpOp::UGe => 0x93, // setae
            };
            // setcc al
            code.extend_from_slice(&[0x0F, setcc, 0xC0]);
            // movzx eax, al (zero-extends into rax)
            code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);

            if let Register::Virtual(v) = dst {
                code.extend_from_slice(&store_reg_to_vreg(stack_slots, v, "rax")?);
                Ok(code)
            } else {
                Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst registers not supported".into(),
                ))
            }
        }
        other => Err(RasError::EncodingError(format!(
            "x86_64 JIT encoder: unsupported instruction {:?}",
            other
        ))),
    }
}



