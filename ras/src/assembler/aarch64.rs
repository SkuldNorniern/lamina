//! AArch64 binary code generation
//!
//! This module handles compilation of MIR to AArch64 binary machine code.

use crate::assembler::core::RasAssembler;
use crate::error::RasError;

#[cfg(feature = "encoder")]
#[derive(Debug, Clone)]
struct BlFixup {
    /// Offset in the final code buffer where the BL instruction word begins.
    patch_location: usize,
    /// Target function name as referenced by MIR (may include or omit '@').
    target_name: String,
}

#[cfg(feature = "encoder")]
static PRINT_I64_FORMAT: [u8; 6] = *b"%lld\n\0";

/// Compile MIR to binary for AArch64
///
/// This reuses the instruction emission logic from mir_codegen/aarch64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
pub fn compile_mir_aarch64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    _function_name: Option<&str>,
) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
    use lamina_codegen::aarch64::{A64RegAlloc, AArch64ABI, FrameMap};
    use lamina_mir::Register;

    let _abi = AArch64ABI::new(assembler.target_os);
    let mut code = Vec::new();
    let mut function_offsets: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // Always compile all functions (needed for internal function calls)
    // Collect all function names first to ensure deterministic order
    let mut all_function_names: Vec<String> = module.functions.keys().cloned().collect();
    all_function_names.sort(); // Sort for deterministic order
    
    // Pre-calculate function offsets (estimate) to handle recursive calls
    // We'll use estimated offsets initially, then update with actual offsets as we compile
    let mut estimated_sizes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for func_name in &all_function_names {
    if let Some(func) = module.functions.get(func_name) {
            // Rough estimate: prologue + epilogue + RET + instructions
            let inst_count = func.blocks.iter().map(|b| b.instructions.len()).sum::<usize>();
            let estimated_size = 16 + (inst_count * 4) + 12 + 4; // prologue + instructions + epilogue + ret
            estimated_sizes.insert(func_name.clone(), estimated_size);
    }
    }
    
    // Pre-populate function_offsets with estimated offsets
    let mut current_estimate = 0;
    for func_name in &all_function_names {
    function_offsets.insert(func_name.clone(), current_estimate);
    current_estimate += estimated_sizes.get(func_name).copied().unwrap_or(100);
    }
    
    // Track internal direct-call fixups (BL) and patch them after final function offsets are known.
    let mut bl_fixups: Vec<BlFixup> = Vec::new();
    
    // Now compile all functions, updating offsets with actual values
    for func_name in &all_function_names {
    let func = module.functions.get(func_name)
            .ok_or_else(|| RasError::EncodingError(
                format!("Function '{}' not found in module", func_name)
            ))?;
        
        // Update function offset with actual value
    function_offsets.insert(func_name.clone(), code.len());
    let mut reg_alloc = A64RegAlloc::new();
    let frame = FrameMap::from_function(func);
    let mut stack_slots: std::collections::HashMap<lamina_mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        
        // Convert FrameMap slots to HashMap for easier lookup
    for (reg, offset) in &frame.slots {
            if let Register::Virtual(vreg) = reg {
                stack_slots.insert(*vreg, *offset);
            }
    }

    let stack_size = frame.frame_size as usize;
        
        // Ensure stack is 16-byte aligned (AAPCS64 requirement)
        // The prologue saves x29, x30 (16 bytes), so we need to ensure
        // the total stack frame is 16-byte aligned
    let aligned_stack_size = (stack_size + 15) & !15;

        // Generate function prologue (binary encoded)
    let prologue = encode_prologue_aarch64(aligned_stack_size)?;
    code.extend_from_slice(&prologue);

        // Handle function parameters: spill ABI arg registers to the FrameMap stack slots.
        //
        // Our JIT encoder currently materializes virtual registers by loading from their
        // stack slot, so parameters must be stored to their slots up-front.
    if !func.sig.params.is_empty() {
            let arg_regs = AArch64ABI::ARG_REGISTERS;
            
            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        // Store x0-x7 directly to the virtual register stack slot.
                        let str_bytes =
                            encode_str_aarch64(arg_regs[index], 29 /* x29 (FP) */, *slot_off)?;
                        code.extend_from_slice(&str_bytes);
                    } else {
                        // Handle stack arguments (AAPCS64: stack args start at caller's [sp, #0])
                        // After prologue: stp x29,x30,[sp,#-16]! then mov x29,sp
                        // Caller's stack args are now at [x29, #16] (16 bytes for saved fp/lr)
                        // First stack arg (arg8) is at [x29, #16], second at [x29, #24], etc.
                        let stack_arg_index = index - arg_regs.len();
                        let caller_off = (16 + stack_arg_index * 8) as i32; // 16 for saved fp/lr
                        let ldr1 = encode_ldr_aarch64("x10", 29, caller_off)?;
                        code.extend_from_slice(&ldr1);
                        let str1 = encode_str_aarch64("x10", 29, *slot_off)?;
                        code.extend_from_slice(&str1);
                    }
                }
            }
    }

        #[derive(Debug)]
        enum BranchFixupKind {
            B { target: String },
            Cbnz { rt: u8, target: String },
            BToEpilogue,
        }

        #[derive(Debug)]
        struct BranchFixup {
            patch_location: usize,
            kind: BranchFixupKind,
        }

        fn write_u32_le(buf: &mut [u8], at: usize, word: u32) -> Result<(), RasError> {
            if at + 4 > buf.len() {
                return Err(RasError::EncodingError(format!(
                    "Patch location out of bounds: {} (len={})",
                    at,
                    buf.len()
                )));
            }
            buf[at..at + 4].copy_from_slice(&word.to_le_bytes());
            Ok(())
        }

        fn encode_b(from_pc: usize, to_pc: usize) -> Result<u32, RasError> {
            let delta = to_pc as i64 - from_pc as i64;
            if delta % 4 != 0 {
                return Err(RasError::EncodingError(format!(
                    "Unaligned B target delta {} (from={}, to={})",
                    delta, from_pc, to_pc
                )));
            }
            let imm26 = delta / 4;
            if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26) {
                return Err(RasError::EncodingError(format!(
                    "B target out of range (delta={} bytes)",
                    delta
                )));
            }
            Ok(0x1400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF))
        }

        fn encode_cbnz(rt: u8, from_pc: usize, to_pc: usize) -> Result<u32, RasError> {
            let delta = to_pc as i64 - from_pc as i64;
            if delta % 4 != 0 {
                return Err(RasError::EncodingError(format!(
                    "Unaligned CBNZ target delta {} (from={}, to={})",
                    delta, from_pc, to_pc
                )));
            }
            let imm19 = delta / 4;
            if !(-(1i64 << 18)..(1i64 << 18)).contains(&imm19) {
                return Err(RasError::EncodingError(format!(
                    "CBNZ target out of range (delta={} bytes)",
                    delta
                )));
            }
            Ok(0xB500_0000u32 | (((imm19 as u32) & 0x7_FFFF) << 5) | (rt as u32))
        }

        let mut block_offsets: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut branch_fixups: Vec<BranchFixup> = Vec::new();

        // Compile blocks. Terminators are handled here (never silently dropped).
        for block in &func.blocks {
            block_offsets.insert(block.label.clone(), code.len());

            let term = block.terminator().ok_or_else(|| {
                RasError::EncodingError(format!(
                    "Block '{}' has no terminator (invalid MIR)",
                    block.label
                ))
            })?;

            for inst in block.body() {
                if inst.is_terminator() {
                    return Err(RasError::EncodingError(format!(
                        "Terminator found in block body '{}' (invalid MIR): {:?}",
                        block.label, inst
                    )));
                }
                let current_offset = code.len();
                let inst_bytes = encode_mir_instruction_aarch64_with_context(
                    assembler,
                    inst,
                    &mut reg_alloc,
                    &stack_slots,
                    aligned_stack_size,
                    func_name,
                    &function_offsets,
                    current_offset,
                    &mut bl_fixups,
                )?;
                code.extend_from_slice(&inst_bytes);
            }

            match term {
                lamina_mir::Instruction::Ret { value } => {
                    if let Some(v) = value {
                        materialize_operand_aarch64(
                            assembler,
                            v,
                            0, // x0
                            &stack_slots,
                            &mut reg_alloc,
                            &mut code,
                            aligned_stack_size,
                        )?;
                    }
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes()); // B <epilogue> (patched)
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::BToEpilogue,
                    });
                }
                lamina_mir::Instruction::Jmp { target } => {
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes()); // B <target> (patched)
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: target.clone(),
                        },
                    });
                }
                lamina_mir::Instruction::Br {
                    cond,
                    true_target,
                    false_target,
                } => {
                    // Load condition value and branch on non-zero.
                    let cond_reg_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let cond_reg = parse_register_aarch64(cond_reg_str)?;
                    materialize_operand_aarch64(
                        assembler,
                        &lamina_mir::Operand::Register(cond.clone()),
                        cond_reg,
                        &stack_slots,
                        &mut reg_alloc,
                        &mut code,
                        aligned_stack_size,
                    )?;
                    reg_alloc.free_scratch(cond_reg_str);

                    // CBNZ <cond>, <true>
                    let patch_location = code.len();
                    let placeholder = 0xB500_0000u32 | (cond_reg as u32);
                    code.extend_from_slice(&placeholder.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::Cbnz {
                            rt: cond_reg,
                            target: true_target.clone(),
                        },
                    });

                    // B <false>
                    let patch_location = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes());
                    branch_fixups.push(BranchFixup {
                        patch_location,
                        kind: BranchFixupKind::B {
                            target: false_target.clone(),
                        },
                    });
                }
                lamina_mir::Instruction::Switch { .. } => {
                    return Err(RasError::EncodingError(
                        "Switch terminator not yet supported by AArch64 JIT backend".to_string(),
                    ));
                }
                lamina_mir::Instruction::TailCall { .. } => {
                    return Err(RasError::EncodingError(
                        "TailCall terminator not yet supported by AArch64 JIT backend".to_string(),
                    ));
                }
                lamina_mir::Instruction::Unreachable => {
                    return Err(RasError::EncodingError(
                        "Unreachable terminator not yet supported by AArch64 JIT backend".to_string(),
                    ));
                }
                other => {
                    return Err(RasError::EncodingError(format!(
                        "Unexpected terminator in block '{}': {:?}",
                        block.label, other
                    )));
                }
            }
        }

        // Patch branches that target blocks or the epilogue.
        let epilogue_offset = code.len();
        for fix in &branch_fixups {
            let from_pc = fix.patch_location;
            let to_pc = match &fix.kind {
                BranchFixupKind::BToEpilogue => epilogue_offset,
                BranchFixupKind::B { target } => *block_offsets.get(target).ok_or_else(|| {
                    RasError::EncodingError(format!(
                        "Branch target block '{}' not found in function '{}'",
                        target, func_name
                    ))
                })?,
                BranchFixupKind::Cbnz { target, .. } => *block_offsets.get(target).ok_or_else(|| {
                    RasError::EncodingError(format!(
                        "Branch target block '{}' not found in function '{}'",
                        target, func_name
                    ))
                })?,
            };
            let patched = match &fix.kind {
                BranchFixupKind::BToEpilogue | BranchFixupKind::B { .. } => encode_b(from_pc, to_pc)?,
                BranchFixupKind::Cbnz { rt, .. } => encode_cbnz(*rt, from_pc, to_pc)?,
            };
            write_u32_le(&mut code, fix.patch_location, patched)?;
        }

        // Generate function epilogue (must pass aligned_stack_size to restore SP)
        let epilogue = encode_epilogue_aarch64(aligned_stack_size)?;
        code.extend_from_slice(&epilogue);

        // RET instruction (x30 is LR)
        code.extend_from_slice(&encode_ret_aarch64(30)?);
    }

    // Patch BL fixups now that all functions have their final offsets.
    fn lookup_function_offset(
        function_offsets: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Option<usize> {
        function_offsets
            .get::<str>(name)
            .copied()
            .or_else(|| {
                if name.starts_with('@') {
                    function_offsets.get(&name[1..]).copied()
                } else {
                    function_offsets.get(&format!("@{}", name)).copied()
                }
            })
    }

    for fixup in &bl_fixups {
        let target_offset =
            lookup_function_offset(&function_offsets, &fixup.target_name).ok_or_else(|| {
                RasError::EncodingError(format!(
                    "BL target function '{}' not found. Available: {:?}",
                    fixup.target_name,
                    function_offsets.keys().collect::<Vec<_>>()
                ))
            })?;

        let from_pc = fixup.patch_location;
        let delta = target_offset as i64 - from_pc as i64;
        if delta % 4 != 0 {
            return Err(RasError::EncodingError(format!(
                "Unaligned BL target delta {} (from={}, to={})",
                delta, from_pc, target_offset
            )));
        }
        let imm26 = delta / 4;
        if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26) {
            return Err(RasError::EncodingError(format!(
                "BL target out of range (delta={} bytes)",
                delta
            )));
        }
        let word = 0x9400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF);
        if fixup.patch_location + 4 > code.len() {
            return Err(RasError::EncodingError(format!(
                "BL patch location out of bounds: {} (len={})",
                fixup.patch_location,
                code.len()
            )));
        }
        code[fixup.patch_location..fixup.patch_location + 4].copy_from_slice(&word.to_le_bytes());
    }

    Ok((code, function_offsets))
}

// Encoding functions extracted from backup file

/// Encode STP 64-bit pre-index instruction
/// stp Xt, Xt2, [Xn|SP, #imm]!
fn enc_stp_pre_64(rt: u8, rt2: u8, rn: u8, imm_bytes: i32) -> Result<u32, RasError> {
    if imm_bytes % 8 != 0 {
        return Err(RasError::EncodingError("STP imm must be multiple of 8".into()));
    }
    let imm7 = imm_bytes / 8;
    if !(-64..=63).contains(&imm7) {
        return Err(RasError::EncodingError(format!("STP imm7 out of range: {}", imm7)));
    }
    let imm7_bits = (imm7 as u32) & 0x7F;

    // STP 64-bit, pre-index base opcode: 0xA980_0000
    // imm7 at [21:15], Rt2 at [14:10], Rn at [9:5], Rt at [4:0]
    Ok(0xA980_0000 | (imm7_bits << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32))
}

/// Encode LDP 64-bit post-index instruction
/// ldp Xt, Xt2, [Xn|SP], #imm
fn enc_ldp_post_64(rt: u8, rt2: u8, rn: u8, imm_bytes: i32) -> Result<u32, RasError> {
    if imm_bytes % 8 != 0 {
        return Err(RasError::EncodingError("LDP imm must be multiple of 8".into()));
    }
    let imm7 = imm_bytes / 8;
    if !(-64..=63).contains(&imm7) {
        return Err(RasError::EncodingError(format!("LDP imm7 out of range: {}", imm7)));
    }
    let imm7_bits = (imm7 as u32) & 0x7F;

    // LDP 64-bit, post-index base opcode: 0xA8C0_0000
    // imm7 at [21:15], Rt2 at [14:10], Rn at [9:5], Rt at [4:0]
    Ok(0xA8C0_0000 | (imm7_bits << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32))
}

/// Encode AArch64 prologue
fn encode_prologue_aarch64(stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    
    // stp x29, x30, [sp, #-16]!
    let stp = enc_stp_pre_64(29, 30, 31, -16)?;
    code.extend_from_slice(&stp.to_le_bytes());

    // add x29, sp, #0   (aka mov x29, sp)
    let mov_fp = 0x9100_03FDu32;
    code.extend_from_slice(&mov_fp.to_le_bytes());

    // SUB sp, sp, #<stack_size> (allocate stack frame)
    // Ensure stack is 16-byte aligned (AAPCS64 requirement)
    // The prologue already saved x29, x30 (16 bytes), maintaining alignment
    if stack_size > 0 {
        // Ensure stack_size is 16-byte aligned (AAPCS64 requirement)
    let aligned_size = (stack_size + 15) & !15;
        
    if aligned_size > 0xFFF {
            return Err(RasError::EncodingError(
                format!("Stack size {} (aligned: {}) too large for single SUB instruction", 
                        stack_size, aligned_size)
            ));
    }
        // sub sp, sp, #aligned
        // Encoding matches clang/llvm-mc:
        //   sub sp, sp, #imm12  => 0xD10003FF | (imm12 << 10)
        let sub_sp = 0xD100_03FFu32 | ((aligned_size as u32) << 10);
        code.extend_from_slice(&sub_sp.to_le_bytes());
    }

    Ok(code)
}

/// Encode AArch64 epilogue
/// Must restore SP before LDP (undo the SUB sp, sp, #aligned_size from prologue)
fn encode_epilogue_aarch64(aligned_stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    if aligned_stack_size > 0 {
        if aligned_stack_size > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "stack restore too large for single ADD: {}",
                aligned_stack_size
            )));
        }
        // add sp, sp, #aligned_stack_size
        // Encoding matches clang/llvm-mc:
        //   add sp, sp, #imm12  => 0x910003FF | (imm12 << 10)
        let add_sp = 0x9100_03FFu32 | ((aligned_stack_size as u32) << 10);
        code.extend_from_slice(&add_sp.to_le_bytes());
    }

    // ldp x29, x30, [sp], #16
    let ldp = enc_ldp_post_64(29, 30, 31, 16)?;
    code.extend_from_slice(&ldp.to_le_bytes());

    Ok(code)
}

/// Encode STR instruction (AArch64)
/// Handles both positive and negative offsets by using SUB for negative offsets
fn encode_str_aarch64(
    src_reg: &str,
    base_reg: u8,
    offset: i32,
) -> Result<Vec<u8>, RasError> {
    let src = parse_register_aarch64(src_reg)?;
    let mut code = Vec::new();

    // STUR Xt, [Xn, #imm9] (signed, unscaled) for offsets in [-256, 255].
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = 0xF800_0000u32
            | (imm9 << 12)
            | ((base_reg as u32) << 5)
            | (src as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }

    // STR Xt, [Xn, #imm12] (unsigned, scaled by 8 for 64-bit).
    if offset >= 0 && offset <= (0xFFF * 8) && (offset % 8 == 0) {
        let imm12 = (offset as u32) / 8;
        let inst = 0xF900_0000u32
            | (imm12 << 10)
            | ((base_reg as u32) << 5)
            | (src as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }

    // Fallback for larger negative offsets: sub x10, base, #abs; stur Xt, [x10]
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "STR offset {} out of range",
                offset
            )));
        }
        let sub_inst = 0xD100_0000u32
            | ((abs_offset & 0xFFF) << 10)
            | ((base_reg as u32) << 5)
            | 10u32;
        code.extend_from_slice(&sub_inst.to_le_bytes());

        let stur = 0xF800_0000u32 | (10u32 << 5) | (src as u32);
        code.extend_from_slice(&stur.to_le_bytes());
        return Ok(code);
    }

    Err(RasError::EncodingError(format!(
        "STR offset {} out of range",
        offset
    )))
}

/// Encode LDR instruction (AArch64)
/// Handles both positive and negative offsets by using SUB for negative offsets
fn encode_ldr_aarch64(
    dst_reg: &str,
    base_reg: u8,
    offset: i32,
) -> Result<Vec<u8>, RasError> {
    let dst = parse_register_aarch64(dst_reg)?;
    let mut code = Vec::new();

    // LDUR Xt, [Xn, #imm9] (signed, unscaled) for offsets in [-256, 255].
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = 0xF840_0000u32
            | (imm9 << 12)
            | ((base_reg as u32) << 5)
            | (dst as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }

    // LDR Xt, [Xn, #imm12] (unsigned, scaled by 8 for 64-bit).
    if offset >= 0 && offset <= (0xFFF * 8) && (offset % 8 == 0) {
        let imm12 = (offset as u32) / 8;
        let inst = 0xF940_0000u32
            | (imm12 << 10)
            | ((base_reg as u32) << 5)
            | (dst as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }

    // Fallback for larger negative offsets: sub x10, base, #abs; ldur Xt, [x10]
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "LDR offset {} out of range",
                offset
            )));
        }
        let sub_inst = 0xD100_0000u32
            | ((abs_offset & 0xFFF) << 10)
            | ((base_reg as u32) << 5)
            | 10u32;
        code.extend_from_slice(&sub_inst.to_le_bytes());

        let ldur = 0xF840_0000u32 | (10u32 << 5) | (dst as u32);
        code.extend_from_slice(&ldur.to_le_bytes());
        return Ok(code);
    }

    Err(RasError::EncodingError(format!(
        "LDR offset {} out of range",
        offset
    )))
}

/// Encode RET instruction (AArch64)
/// RET Xn = 0xD65F0000 | (n << 5)
fn encode_ret_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD65F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// Encode BR instruction (AArch64)
/// BR Xn = 0xD61F0000 | (n << 5)
fn encode_br_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD61F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// Encode BLR instruction (AArch64)
/// BLR Xn = 0xD63F0000 | (n << 5)
fn encode_blr_aarch64(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD63F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// Parse register name to encoding (AArch64)
fn parse_register_aarch64(reg: &str) -> Result<u8, RasError> {
    let reg = reg.trim_start_matches('%');
    match reg {
    "x0" | "w0" => Ok(0),
    "x1" | "w1" => Ok(1),
    "x2" | "w2" => Ok(2),
    "x3" | "w3" => Ok(3),
    "x4" | "w4" => Ok(4),
    "x5" | "w5" => Ok(5),
    "x6" | "w6" => Ok(6),
    "x7" | "w7" => Ok(7),
    "x8" | "w8" => Ok(8),
    "x9" | "w9" => Ok(9),
    "x10" | "w10" => Ok(10),
    "x11" | "w11" => Ok(11),
    "x12" | "w12" => Ok(12),
    "x13" | "w13" => Ok(13),
    "x14" | "w14" => Ok(14),
    "x15" | "w15" => Ok(15),
    "x16" | "w16" | "ip0" => Ok(16),
    "x17" | "w17" | "ip1" => Ok(17),
    "x18" | "w18" => Ok(18),
    "x19" | "w19" => Ok(19),
    "x20" | "w20" => Ok(20),
    "x21" | "w21" => Ok(21),
    "x22" | "w22" => Ok(22),
    "x23" | "w23" => Ok(23),
    "x24" | "w24" => Ok(24),
    "x25" | "w25" => Ok(25),
    "x26" | "w26" => Ok(26),
    "x27" | "w27" => Ok(27),
    "x28" | "w28" => Ok(28),
    "x29" | "w29" | "fp" => Ok(29),
    "x30" | "w30" | "lr" => Ok(30),
    "x31" | "w31" | "sp" | "xzr" | "wzr" => Ok(31),
    _ => Err(RasError::EncodingError(format!("Unknown register: {}", reg))),
    }
}

/// Encode MIR instruction to binary (AArch64)
///
/// This reuses the instruction emission logic from mir_codegen/aarch64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
fn encode_mir_instruction_aarch64(
    assembler: &mut RasAssembler,
    inst: &lamina_mir::Instruction,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    func_name: &str,
) -> Result<Vec<u8>, RasError> {
    let mut bl_fixups = Vec::<BlFixup>::new();
    encode_mir_instruction_aarch64_with_context(
        assembler,
        inst, reg_alloc, stack_slots, stack_size, func_name,
        &std::collections::HashMap::new(),
        0,
        &mut bl_fixups,
    )
}

fn encode_mir_instruction_aarch64_with_context(
    assembler: &mut RasAssembler,
    inst: &lamina_mir::Instruction,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    _func_name: &str,
    function_offsets: &std::collections::HashMap<String, usize>,
    current_offset: usize,
    bl_fixups: &mut Vec<BlFixup>,
) -> Result<Vec<u8>, RasError> {
    use lamina_mir::{IntBinOp, Register};
    let mut code = Vec::new();
    
    match inst {
            lamina_mir::Instruction::Ret { value } => {
                if let Some(v) = value {
                    // Load return value to x0 (return value register)
                    materialize_operand_aarch64(assembler, v, 0, stack_slots, reg_alloc, &mut code, stack_size)?;
                }
                // Don't emit RET here - it will be emitted after the epilogue at function end
                // Just fall through to the epilogue
            }
    lamina_mir::Instruction::IntBinary { op, dst, lhs, rhs, ty: _ } => {
            // Allocate scratch registers
            let lhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let rhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x11");
            let dst_reg_str = reg_alloc.alloc_scratch().unwrap_or("x12");
            let lhs_reg = parse_register_aarch64(lhs_reg_str)?;
            let rhs_reg = parse_register_aarch64(rhs_reg_str)?;
            let dst_reg = parse_register_aarch64(dst_reg_str)?;
            
                // Materialize operands
                materialize_operand_aarch64(assembler, lhs, lhs_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
                materialize_operand_aarch64(assembler, rhs, rhs_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
            
            // Encode binary operation (64-bit, no shift).
            let inst = match op {
                IntBinOp::Add => 0x8B00_0000u32, // add xd, xn, xm
                IntBinOp::Sub => 0xCB00_0000u32, // sub xd, xn, xm
                IntBinOp::Mul => 0x9B00_7C00u32, // mul xd, xn, xm (alias madd xd,xn,xm,xzr)
                IntBinOp::UDiv => 0x9AC0_0800u32, // udiv xd, xn, xm
                IntBinOp::SDiv => 0x9AC0_0C00u32, // sdiv xd, xn, xm
                IntBinOp::And => 0x8A00_0000u32, // and xd, xn, xm
                IntBinOp::Or => 0xAA00_0000u32,  // orr xd, xn, xm
                IntBinOp::Xor => 0xCA00_0000u32, // eor xd, xn, xm
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "Unsupported IntBinary operation: {:?}",
                        op
                    )));
                }
            } | ((rhs_reg as u32) << 16)
                | ((lhs_reg as u32) << 5)
                | (dst_reg as u32);
            code.extend_from_slice(&inst.to_le_bytes());
            
            // Store result to destination
            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg) {
                    code.extend_from_slice(&encode_str_aarch64(
                        dst_reg_str,
                        29, // x29 (FP)
                        *offset,
                    )?);
                }
            
            // Free scratch registers
            reg_alloc.free_scratch(lhs_reg_str);
            reg_alloc.free_scratch(rhs_reg_str);
            reg_alloc.free_scratch(dst_reg_str);
    }
    lamina_mir::Instruction::Load { dst, addr, ty: _, .. } => {
            use lamina_mir::AddressMode;
            // Load from memory address to destination
            let tmp_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let tmp_reg = parse_register_aarch64(tmp_reg_str)?;
            
            // Handle address mode
            match addr {
                AddressMode::BaseOffset { base, offset } => {
                    // Materialize base register
                    let base_reg_str = if let Register::Virtual(vreg) = base {
                        if let Some(base_offset) = stack_slots.get(vreg) {
                            let base_tmp = reg_alloc.alloc_scratch().unwrap_or("x11");
                            code.extend_from_slice(&encode_ldr_aarch64(
                                base_tmp,
                                29, // x29 (FP)
                                *base_offset,
                            )?);
                            base_tmp
                        } else {
                            return Err(RasError::EncodingError(
                                format!("No stack slot for base register: {:?}", base)
                            ));
                        }
                    } else {
                        // Physical register - use directly
                        return Err(RasError::EncodingError(
                            "Physical register base not yet supported".to_string()
                        ));
                    };
                    let base_reg = parse_register_aarch64(base_reg_str)?;
                    
                    // LDR tmp, [base_reg, #offset]
                    if *offset >= 0 && (*offset as u32) <= 0xFFF {
                        let imm9 = (*offset as u32) & 0x1FF;
                        let ldr_inst = ((((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                                      (0b111u32 << 27)) |         // [26] = 0
                                      (0b01u32 << 24)) |         // [21] = 0 (unscaled)
                                      (imm9 << 12)) |        // [11:10] = 00
                                      ((base_reg as u32) << 5) |  // [9:5] = Rn
                                      (tmp_reg as u32);        // [4:0] = Rt
                        code.extend_from_slice(&ldr_inst.to_le_bytes());
                    } else {
                        return Err(RasError::EncodingError(
                            format!("Load offset {} out of range", offset)
                        ));
                    }
                    
                    if base_reg_str != "x10" && base_reg_str != "x11" {
                        reg_alloc.free_scratch(base_reg_str);
                    }
                }
                AddressMode::BaseIndexScale { .. } => {
                    return Err(RasError::EncodingError(
                        "BaseIndexScale address mode not yet implemented".to_string()
                    ));
                }
            }
            
            // Store to destination
            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg) {
                    code.extend_from_slice(&encode_str_aarch64(
                        tmp_reg_str,
                        29, // x29 (FP)
                        *offset,
                    )?);
                }
            
            reg_alloc.free_scratch(tmp_reg_str);
    }
    lamina_mir::Instruction::Store { src, addr, ty: _, .. } => {
            use lamina_mir::AddressMode;
            // Store from source to memory address
            let src_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let src_reg = parse_register_aarch64(src_reg_str)?;
            
                // Materialize source
                materialize_operand_aarch64(assembler, src, src_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
            
            // Handle address mode
            match addr {
                AddressMode::BaseOffset { base, offset } => {
                    // Materialize base register
                    let base_reg_str = if let Register::Virtual(vreg) = base {
                        if let Some(base_offset) = stack_slots.get(vreg) {
                            let base_tmp = reg_alloc.alloc_scratch().unwrap_or("x11");
                            code.extend_from_slice(&encode_ldr_aarch64(
                                base_tmp,
                                29, // x29 (FP)
                                *base_offset,
                            )?);
                            base_tmp
                        } else {
                            return Err(RasError::EncodingError(
                                format!("No stack slot for base register: {:?}", base)
                            ));
                        }
                    } else {
                        return Err(RasError::EncodingError(
                            "Physical register base not yet supported".to_string()
                        ));
                    };
                    let base_reg = parse_register_aarch64(base_reg_str)?;
                    
                    // STR src_reg, [base_reg, #offset]
                    if *offset >= 0 && (*offset as u32) <= 0xFFF {
                        let imm9 = (*offset as u32) & 0x1FF;
                        let str_inst = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                                      (0b111u32 << 27)) |         // [21] = 0 (unscaled)
                                      (imm9 << 12)) |        // [11:10] = 00
                                      ((base_reg as u32) << 5) |  // [9:5] = Rn
                                      (src_reg as u32);        // [4:0] = Rt
                        code.extend_from_slice(&str_inst.to_le_bytes());
                    } else {
                        return Err(RasError::EncodingError(
                            format!("Store offset {} out of range", offset)
                        ));
                    }
                    
                    if base_reg_str != "x10" && base_reg_str != "x11" {
                        reg_alloc.free_scratch(base_reg_str);
                    }
                }
                AddressMode::BaseIndexScale { .. } => {
                    return Err(RasError::EncodingError(
                        "BaseIndexScale address mode not yet implemented".to_string()
                    ));
                }
            }
            
            reg_alloc.free_scratch(src_reg_str);
    }
    lamina_mir::Instruction::IntCmp { op, dst, lhs, rhs, ty } => {
            use lamina_mir::IntCmpOp;
            // Allocate scratch registers
            let lhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
            let rhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x11");
            let dst_reg_str = reg_alloc.alloc_scratch().unwrap_or("x12");
            let lhs_reg = parse_register_aarch64(lhs_reg_str)?;
            let rhs_reg = parse_register_aarch64(rhs_reg_str)?;
            let dst_reg = parse_register_aarch64(dst_reg_str)?;
            
                // Materialize operands
                materialize_operand_aarch64(assembler, lhs, lhs_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
                materialize_operand_aarch64(assembler, rhs, rhs_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
            
            if ty.size_bytes() != 8 {
                return Err(RasError::EncodingError(format!(
                    "IntCmp currently only supported for i64 in AArch64 JIT (got {:?})",
                    ty
                )));
            }

            // CMP lhs, rhs is alias for: SUBS xzr, lhs, rhs
            let cmp_inst = 0xEB00_001Fu32
                | ((rhs_reg as u32) << 16)
                | ((lhs_reg as u32) << 5);
            code.extend_from_slice(&cmp_inst.to_le_bytes());

            // CSET dst, cond is alias for: CSINC dst, xzr, xzr, inv(cond)
            let cond_code = match op {
                IntCmpOp::Eq => 0b0000u32,  // eq
                IntCmpOp::Ne => 0b0001u32,  // ne
                IntCmpOp::ULt => 0b0011u32, // lo
                IntCmpOp::ULe => 0b1001u32, // ls
                IntCmpOp::UGt => 0b1000u32, // hi
                IntCmpOp::UGe => 0b0010u32, // hs
                IntCmpOp::SLt => 0b1011u32, // lt
                IntCmpOp::SLe => 0b1101u32, // le
                IntCmpOp::SGt => 0b1100u32, // gt
                IntCmpOp::SGe => 0b1010u32, // ge
            };
            let inv_cond = cond_code ^ 1;
            let cset_inst = 0x9A9F_07E0u32 | (inv_cond << 12) | (dst_reg as u32);
            code.extend_from_slice(&cset_inst.to_le_bytes());
            
            // Store result to destination
            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg) {
                    code.extend_from_slice(&encode_str_aarch64(
                        dst_reg_str,
                        29, // x29 (FP)
                        *offset,
                    )?);
                }
            
            // Free scratch registers
            reg_alloc.free_scratch(lhs_reg_str);
            reg_alloc.free_scratch(rhs_reg_str);
            reg_alloc.free_scratch(dst_reg_str);
    }
    lamina_mir::Instruction::Call { name, args, ret } => {
            use lamina_codegen::aarch64::AArch64ABI;
            let _abi = AArch64ABI::new(assembler.target_os);

            // Materialize arguments into argument registers (x0-x7).
            let arg_regs = AArch64ABI::ARG_REGISTERS;
            for (i, arg) in args.iter().enumerate().take(8) {
                let arg_reg_str = arg_regs[i];
                let arg_reg = parse_register_aarch64(arg_reg_str)?;
                materialize_operand_aarch64(
                    assembler,
                    arg,
                    arg_reg,
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
            }

            // Handle stack arguments (args beyond 8).
            let stack_args = if args.len() > 8 { &args[8..] } else { &[] };
            let stack_space = (stack_args.len() * 8 + 15) & !15;
            if stack_space > 0 {
                if stack_space > 0xFFF {
                    return Err(RasError::EncodingError(format!(
                        "Stack space {} too large for single SUB",
                        stack_space
                    )));
                }
                let sub_inst = (((0b1u32 << 31) | (0b1u32 << 30)) | (0b100010u32 << 23))
                    | ((stack_space as u32 & 0xFFF) << 10)
                    | (31u32 << 5)
                    | 31u32;
                code.extend_from_slice(&sub_inst.to_le_bytes());

                for (i, arg) in stack_args.iter().enumerate() {
                    let offset = i * 8;
                    let scratch_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                    let scratch = parse_register_aarch64(scratch_str)?;
                    materialize_operand_aarch64(
                        assembler,
                        arg,
                        scratch,
                        stack_slots,
                        reg_alloc,
                        &mut code,
                        stack_size,
                    )?;

                    if offset <= 0xFFF {
                        let imm9 = (offset as u32) & 0x1FF;
                        let str_inst = (((0b11u32 << 30) | (0b111u32 << 27)) | (imm9 << 12))
                            | (31u32 << 5)
                            | (scratch as u32);
                        code.extend_from_slice(&str_inst.to_le_bytes());
                    } else {
                        return Err(RasError::EncodingError(format!(
                            "Stack argument offset {} too large",
                            offset
                        )));
                    }
                    reg_alloc.free_scratch(scratch_str);
                }
            }

            // External function calls need special handling for JIT.
            // Keep `print(x)` as a special-case intrinsic.
            if name == "print" && args.len() == 1 {
                if stack_space != 0 {
                    return Err(RasError::EncodingError(
                        "print() intrinsic does not support stack-passed args".to_string(),
                    ));
                }
                // (Existing printf-based implementation below.)
                // Resolve printf - try both "printf" and "_printf" on macOS
                // The actual symbol name may vary, but dlsym usually finds "printf"
                let printf_name = "printf";
                let printf_name_alt = "_printf";
                
                let printf_addr = if assembler.function_pointers.contains_key(printf_name) {
                    *assembler.function_pointers.get(printf_name).unwrap()
                } else if assembler.function_pointers.contains_key(printf_name_alt) {
                    *assembler.function_pointers.get(printf_name_alt).unwrap()
                } else {
                    // Try to resolve "printf" first
                    if assembler.register_function(printf_name).is_err() {
                        // Fallback to "_printf" on macOS
                        if assembler.target_os == lamina_platform::TargetOperatingSystem::MacOS {
                            if let Err(e) = assembler.register_function(printf_name_alt) {
                                return Err(RasError::EncodingError(format!(
                                    "Failed to resolve printf or _printf for print() intrinsic: {}. \
                                     Runtime function resolution may not be available on this system.",
                                    e
                                )));
                            }
                            *assembler.function_pointers.get(printf_name_alt)
                                .ok_or_else(|| RasError::EncodingError(
                                    format!("{} not resolved", printf_name_alt)
                                ))?
                        } else {
                            return Err(RasError::EncodingError(format!(
                                "Failed to resolve {} for print() intrinsic. \
                                 Runtime function resolution may not be available on this system.",
                                printf_name
                            )));
                        }
                    } else {
                        *assembler.function_pointers.get(printf_name)
                            .ok_or_else(|| RasError::EncodingError(
                                format!("{} not resolved", printf_name)
                            ))?
                    }
                };
                // Match clang's lowering for macOS AArch64 varargs:
                //   sub sp, sp, #32
                //   str x8, [sp]
                //   x0 = "%lld\\n"
                //   bl printf
                //   add sp, sp, #32
                let home_area_size = 32u32;

                // Allocate home area (keeps SP 16-byte aligned).
                let sub_sp = 0xD100_03FFu32 | ((home_area_size & 0xFFF) << 10);
                code.extend_from_slice(&sub_sp.to_le_bytes());

                // Spill the variadic integer argument to the home area at [sp].
                materialize_operand_aarch64(
                    assembler,
                    &args[0],
                    8, // x8
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_str_aarch64("x8", 31, 0)?);

                // Load format string pointer into x0.
                let fmt_ptr = PRINT_I64_FORMAT.as_ptr() as u64;
                materialize_operand_aarch64(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(fmt_ptr as i64)),
                    0, // x0
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;

                // Load printf address into x16 and call via BLR.
                materialize_operand_aarch64(
                    assembler,
                    &lamina_mir::Operand::Immediate(lamina_mir::Immediate::I64(printf_addr as i64)),
                    16, // x16
                    stack_slots,
                    reg_alloc,
                    &mut code,
                    stack_size,
                )?;
                code.extend_from_slice(&encode_blr_aarch64(16)?);

                // Restore SP.
                let add_sp = 0x9100_03FFu32 | ((home_area_size & 0xFFF) << 10);
                code.extend_from_slice(&add_sp.to_le_bytes());
            } else {
                // Internal direct call (BL). Emit placeholder and patch later.
                let is_internal = function_offsets.contains_key(name)
                    || (name.starts_with('@') && function_offsets.contains_key(&name[1..]))
                    || (!name.starts_with('@') && function_offsets.contains_key(&format!("@{}", name)));
                if !is_internal {
                    return Err(RasError::EncodingError(format!(
                        "External function call '{}' requires runtime resolution (not implemented for AArch64 JIT)",
                        name
                    )));
                }

                let bl_pc = current_offset + code.len();
                code.extend_from_slice(&0x9400_0000u32.to_le_bytes()); // BL <target> (patched)
                bl_fixups.push(BlFixup {
                    patch_location: bl_pc,
                    target_name: name.clone(),
                });

                if stack_space > 0 {
                    let add_inst = ((0b1u32 << 31) | (0b100010u32 << 23))
                        | ((stack_space as u32 & 0xFFF) << 10)
                        | (31u32 << 5)
                        | 31u32;
                    code.extend_from_slice(&add_inst.to_le_bytes());
                }

                if let Some(dst) = ret
                    && let Register::Virtual(vreg) = dst
                    && let Some(offset) = stack_slots.get(vreg)
                {
                    code.extend_from_slice(&encode_str_aarch64(
                        "x0",
                        29, // x29 (FP)
                        *offset,
                    )?);
                }
            }
    }
    lamina_mir::Instruction::Jmp { .. } => {
            return Err(RasError::EncodingError(
                "Jmp must be handled at block/terminator level (bug: reached instruction encoder)"
                    .to_string(),
            ));
    }
    lamina_mir::Instruction::Br { .. } => {
            return Err(RasError::EncodingError(
                "Br must be handled at block/terminator level (bug: reached instruction encoder)"
                    .to_string(),
            ));
    }
    _ => {
            return Err(RasError::EncodingError(
                format!("MIR instruction not yet implemented: {:?}", inst)
            ));
    }
    }
    
    Ok(code)
}

/// Materialize an operand into a register (AArch64)
/// Loads from stack slot if operand is a virtual register, or moves immediate
#[cfg(feature = "encoder")]
fn materialize_operand_aarch64(
    _assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    dst_reg: u8,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
) -> Result<(), RasError> {
    use lamina_mir::{Immediate, Operand, Register};
    
    match op {
    Operand::Immediate(imm) => {
            // MOVZ/MOVK sequence (64-bit).
            let imm_val: u64 = match imm {
                Immediate::I8(v) => *v as i64 as u64,
                Immediate::I16(v) => *v as i64 as u64,
                Immediate::I32(v) => *v as i64 as u64,
                Immediate::I64(v) => *v as u64,
                _ => {
                    return Err(RasError::EncodingError(
                        "Floating-point immediates not yet supported".to_string(),
                    ))
                }
            };

            // Fast path: single MOVZ.
            if imm_val <= 0xFFFF {
                let movz = 0xD280_0000u32 | ((imm_val as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movz.to_le_bytes());
                return Ok(());
            }

            // General path: MOVZ for low 16 bits + MOVK for remaining chunks.
            let chunk0 = (imm_val & 0xFFFF) as u16;
            let chunk1 = ((imm_val >> 16) & 0xFFFF) as u16;
            let chunk2 = ((imm_val >> 32) & 0xFFFF) as u16;
            let chunk3 = ((imm_val >> 48) & 0xFFFF) as u16;

            let movz = 0xD280_0000u32 | ((chunk0 as u32) << 5) | (dst_reg as u32);
            code.extend_from_slice(&movz.to_le_bytes());

            if chunk1 != 0 {
                let movk = 0xF280_0000u32
                    | (0b01u32 << 21)
                    | ((chunk1 as u32) << 5)
                    | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk2 != 0 {
                let movk = 0xF280_0000u32
                    | (0b10u32 << 21)
                    | ((chunk2 as u32) << 5)
                    | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk3 != 0 {
                let movk = 0xF280_0000u32
                    | (0b11u32 << 21)
                    | ((chunk3 as u32) << 5)
                    | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
    }
    Operand::Register(Register::Virtual(vreg)) => {
            // Load from stack slot
            // FrameMap stack slots are FP-relative offsets (negative for locals).
            if let Some(offset) = stack_slots.get(vreg) {
                let dst_reg_str = format!("x{}", dst_reg);
                code.extend_from_slice(&encode_ldr_aarch64(
                    &dst_reg_str,
                    29, // x29 (FP)
                    *offset,
                )?);
            } else {
                return Err(RasError::EncodingError(
                    format!("No stack slot for virtual register: {:?}", vreg)
                ));
            }
    }
    Operand::Register(Register::Physical(_)) => {
            // Physical registers are already in place, but we need to move to dst
            // For now, assume it's already correct (this is a simplification)
            return Err(RasError::EncodingError(
                "Physical register operands not yet fully supported".to_string()
            ));
    }
    }
    
    Ok(())
}
