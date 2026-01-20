//! AArch64 binary code generation
//!
//! This module handles compilation of MIR to AArch64 binary machine code.

use crate::assembler::core::RasAssembler;
use crate::error::RasError;

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
    
    // Track BL fixups: (patch_location, target_function_name, bl_pc_offset)
    #[derive(Debug)]
    struct BlFixup {
        patch_location: usize,  // Offset in code where BL instruction is
        target_name: String,    // Function name to call
        bl_pc_offset: usize,    // PC offset of the BL instruction (for PC+4 calculation)
    }
    let _bl_fixups: Vec<BlFixup> = Vec::new();
    
    // Now compile all functions, updating offsets with actual values
    for func_name in &all_function_names {
    let func = module.functions.get(func_name)
            .ok_or_else(|| RasError::EncodingError(
                format!("Function '{}' not found in module", func_name)
            ))?;
        
        // Update function offset with actual value
    let func_start = code.len();
    function_offsets.insert(func_name.clone(), func_start);
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

        // Handle function parameters - copy from ABI registers to allocator registers/stack slots
        // Since allocator uses x9-x15 (caller-saved), we copy x0-x7 to allocator registers first
        // Then store to stack slots if needed
    if !func.sig.params.is_empty() {
            let arg_regs = AArch64ABI::ARG_REGISTERS;
            // Allocator registers we can use (x9-x15, avoiding x8 which might be used by platform)
            let allocator_regs = ["x9", "x10", "x11", "x12", "x13", "x14", "x15"];
            
            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        // Copy from ABI register (x0-x7) to allocator register first
                        // This ensures the value is in an allocator register if the allocator wants to use it
                        if index < allocator_regs.len() {
                            // Copy x0-x7 to x9-x15
                            let mov_bytes = encode_mov_reg_aarch64(
                                parse_register_aarch64(allocator_regs[index])?,
                                parse_register_aarch64(arg_regs[index])?,
                            );
                            code.extend_from_slice(&mov_bytes);
                            
                            // Also store to stack slot for spilling
                            let adjusted_offset = *slot_off - (aligned_stack_size as i32);
                            let str_bytes = encode_str_aarch64(
                                allocator_regs[index],
                                29, // x29 (FP)
                                adjusted_offset,
                            )?;
                            code.extend_from_slice(&str_bytes);
                        } else {
                            // More than 7 args in registers - store directly to stack
                            let adjusted_offset = *slot_off - (aligned_stack_size as i32);
                            let str_bytes = encode_str_aarch64(
                                arg_regs[index],
                                29, // x29 (FP)
                                adjusted_offset,
                            )?;
                            code.extend_from_slice(&str_bytes);
                        }
                    } else {
                        // Handle stack arguments (AAPCS64: stack args start at caller's [sp, #0])
                        // After prologue: stp x29,x30,[sp,#-16]! then mov x29,sp
                        // Caller's stack args are now at [x29, #16] (16 bytes for saved fp/lr)
                        // First stack arg (arg8) is at [x29, #16], second at [x29, #24], etc.
                        let stack_arg_index = index - arg_regs.len();
                        let caller_off = (16 + stack_arg_index * 8) as i32; // 16 for saved fp/lr
                        let ldr1 = encode_ldr_aarch64("x10", 29, caller_off)?;
                        code.extend_from_slice(&ldr1);
                        let adjusted_offset = *slot_off - (aligned_stack_size as i32);
                        let str1 = encode_str_aarch64("x10", 29, adjusted_offset)?;
                        code.extend_from_slice(&str1);
                    }
                }
            }
    }

        // Compile each block
    for block in &func.blocks {
            for inst in &block.instructions {
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
                )?;
                code.extend_from_slice(&inst_bytes);
            }
    }

        // Generate function epilogue (must pass aligned_stack_size to restore SP)
    let epilogue = encode_epilogue_aarch64(aligned_stack_size)?;
    code.extend_from_slice(&epilogue);

        // RET instruction (x30 is LR)
    code.extend_from_slice(&encode_ret_aarch64(30)?);
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
        // SUB (immediate): [31]=sf(1), [30]=S(0), [29:24]=opcode(100010), [23:22]=shift(0), [21:10]=imm12, [9:5]=Rn, [4:0]=Rd
        let sub_sp = (0b1u32 << 31)           // [31] = sf (1 for 64-bit)
            | (0b0u32 << 30)                  // [30] = S (0, non-setting)
            | (0b100010u32 << 23)             // [29:24] = opcode (100010 for SUB immediate)
            | ((aligned_size as u32) << 10)   // [21:10] = imm12
            | (31u32 << 5)                     // [9:5] = Rn (sp = 31)
            | 31u32;                           // [4:0] = Rd (sp = 31)
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
        let add_sp = (0b1u32 << 31)
            | (0b100010u32 << 23)
            | ((aligned_stack_size as u32) << 10)
            | (31u32 << 5)
            | 31u32;
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
    
    if (0..=255).contains(&offset) {
        // Small positive offset: use unscaled STR with imm9 (range 0-255 bytes)
        // STR (immediate, unscaled): [31:30]=size(11), [29:27]=111, [26]=0, [25:24]=opc(00), [23:22]=00, [21]=0, [20:12]=imm9, [11:10]=00, [9:5]=Rn, [4:0]=Rt
        // For 64-bit STR unscaled: size=11, opc=00
        let imm9 = offset as u32 & 0x1FF;
        let inst = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                   (0b111u32 << 27)) |         // [21] = 0 (unscaled)
                   (imm9 << 12)) |        // [11:10] = 00
                   ((base_reg as u32) << 5) |  // [9:5] = Rn
                   (src as u32);            // [4:0] = Rt
        code.extend_from_slice(&inst.to_le_bytes());
    } else if offset > 255 && offset <= 0xFFF {
        // Larger positive offset: use scaled STR with imm12 (range 0-4095*8 bytes for 64-bit)
        // STR (immediate, scaled): [31:30]=size(11), [29:27]=111, [26]=0, [25:24]=opc(00), [23:22]=size(11), [21]=1, [20:12]=imm12[11:3], [11:10]=imm12[2:0], [9:5]=Rn, [4:0]=Rt
        // For 64-bit STR scaled: size=11, opc=00, scaled=1, imm12 = offset / 8
        if offset % 8 != 0 {
            return Err(RasError::EncodingError(
                format!("STR offset {} must be multiple of 8 for scaled format", offset)
            ));
        }
        let imm12 = (offset as u32) / 8;
        let inst = ((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                   (0b111u32 << 27)) |        // [25:24] = opc (00 = STR)
                   (0b11u32 << 22) |        // [23:22] = size (11 = 64-bit)
                   (0b1u32 << 21) |         // [21] = 1 (scaled)
                   ((imm12 >> 3) << 12) |   // [20:12] = imm12[11:3]
                   ((imm12 & 0x7) << 10) |  // [11:10] = imm12[2:0]
                   ((base_reg as u32) << 5) |  // [9:5] = Rn
                   (src as u32);            // [4:0] = Rt
        code.extend_from_slice(&inst.to_le_bytes());
    } else if (-256..0).contains(&offset) {
        // Negative offset: use SUB to adjust base, then STR with positive offset
        // SUB x10, <base>, #abs(offset)
    let abs_offset = (-offset) as u32;
    if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(
                format!("STR offset {} out of range", offset)
            ));
    }
    let sub_inst = (((0b1u32 << 31) |        // sf=1 (64-bit)
                      (0b1u32 << 30)) |        // S=0
                      (0b100010u32 << 23)) |        // shift=0
                      ((abs_offset & 0xFFF) << 10) | // imm12
                      ((base_reg as u32) << 5) | // Rn
                      10u32;           // Rd=x10 (scratch)
    code.extend_from_slice(&sub_inst.to_le_bytes());
        
        // STR <src>, [x10]
        let str_inst = ((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                      (0b111u32 << 27)) |        // [11:10] = 00
                      (10u32 << 5) |           // [9:5] = Rn = x10
                      (src as u32);            // [4:0] = Rt
        code.extend_from_slice(&str_inst.to_le_bytes());
    } else {
    return Err(RasError::EncodingError(
            format!("STR offset {} out of range (must be -4095 to 4095)", offset)
    ));
    }
    
    Ok(code)
}

/// Encode MOV (register) instruction: MOV <Xd>, <Xn>
/// MOV is an alias for ORR <Xd>, XZR, <Xn>
/// ORR (shifted register): sf=1, opc=01, shift=00, N=0, Rm=<src>, imm6=0, Rn=31 (XZR), Rd=<dst>
fn encode_mov_reg_aarch64(dst: u8, src: u8) -> Vec<u8> {
    // ORR <Xd>, XZR, <Xn>
    // Encoding: [31]=sf(1), [30:29]=opc(01), [28:24]=01010, [23:22]=shift(00), [21:16]=imm6(0), [15:10]=Rm, [9:5]=Rn(31=XZR), [4:0]=Rd
    let inst = (0b1u32 << 31) |           // [31] = sf (1 for 64-bit)
              (0b01u32 << 29) |           // [30:29] = opc (01 = ORR)
              (0b01010u32 << 24) |        // [28:24] = 01010 (ORR opcode)
              (0b0u32 << 22) |            // [23:22] = shift (00 = LSL)
              (0b000000u32 << 16) |       // [21:16] = imm6 (0 = no shift)
              ((src as u32) << 10) |      // [15:10] = Rm (source register)
              (31u32 << 5) |              // [9:5] = Rn (31 = XZR)
              (dst as u32);               // [4:0] = Rd (destination register)
    inst.to_le_bytes().to_vec()
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
    
    if (0..=255).contains(&offset) {
        // Small positive offset: use unscaled LDR with imm9 (range 0-255 bytes)
        // LDR (immediate, unscaled): [31:30]=size(11), [29:27]=111, [26]=0, [25:24]=opc(01), [23:22]=00, [21]=0, [20:12]=imm9, [11:10]=00, [9:5]=Rn, [4:0]=Rt
        // For 64-bit LDR unscaled: size=11, opc=01
        let imm9 = offset as u32 & 0x1FF;
        let inst = ((((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                   (0b111u32 << 27)) |         // [26] = 0
                   (0b01u32 << 24)) |         // [21] = 0 (unscaled)
                   (imm9 << 12)) |        // [11:10] = 00
                   ((base_reg as u32) << 5) |  // [9:5] = Rn
                   (dst as u32);            // [4:0] = Rt
        code.extend_from_slice(&inst.to_le_bytes());
    } else if offset > 255 && offset <= 0xFFF {
        // Larger positive offset: use scaled LDR with imm12 (range 0-4095*8 bytes for 64-bit)
        // LDR (immediate, scaled): [31:30]=size(11), [29:27]=111, [26]=0, [25:24]=opc(01), [23:22]=size(11), [21]=1, [20:12]=imm12[11:3], [11:10]=imm12[2:0], [9:5]=Rn, [4:0]=Rt
        // For 64-bit LDR scaled: size=11, opc=01, scaled=1, imm12 = offset / 8
        if offset % 8 != 0 {
            return Err(RasError::EncodingError(
                format!("LDR offset {} must be multiple of 8 for scaled format", offset)
            ));
        }
        let imm12 = (offset as u32) / 8;
        let inst = ((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                   (0b111u32 << 27)) |         // [26] = 0
                   (0b01u32 << 24) |        // [25:24] = opc (01 = LDR)
                   (0b11u32 << 22) |        // [23:22] = size (11 = 64-bit)
                   (0b1u32 << 21) |         // [21] = 1 (scaled)
                   ((imm12 >> 3) << 12) |   // [20:12] = imm12[11:3]
                   ((imm12 & 0x7) << 10) |  // [11:10] = imm12[2:0]
                   ((base_reg as u32) << 5) |  // [9:5] = Rn
                   (dst as u32);            // [4:0] = Rt
        code.extend_from_slice(&inst.to_le_bytes());
    } else if (-256..0).contains(&offset) {
        // Small negative offset: use unscaled LDR with imm9 (range -256 to -1 bytes)
        // LDR (immediate, unscaled): [31:30]=size(11), [29:27]=111, [26]=0, [25:24]=opc(01), [23:22]=00, [21]=0, [20:12]=imm9, [11:10]=00, [9:5]=Rn, [4:0]=Rt
        // For negative offsets, imm9 is signed: -256 to 255
        let imm9 = offset & 0x1FF; // Sign-extend to 9 bits
        let inst = ((((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                   (0b111u32 << 27)) |         // [26] = 0
                   (0b01u32 << 24)) |         // [21] = 0 (unscaled)
                   ((imm9 as u32) << 12)) |        // [11:10] = 00
                   ((base_reg as u32) << 5) |  // [9:5] = Rn
                   (dst as u32);            // [4:0] = Rt
        code.extend_from_slice(&inst.to_le_bytes());
    } else if (-0xFFF..-256).contains(&offset) {
        // Larger negative offset: use SUB to adjust base, then LDR with positive offset
        // SUB x10, <base>, #abs(offset)
    let abs_offset = (-offset) as u32;
    if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(
                format!("LDR offset {} out of range", offset)
            ));
    }
    let sub_inst = (((0b1u32 << 31) |        // sf=1 (64-bit)
                      (0b1u32 << 30)) |        // S=0
                      (0b100010u32 << 23)) |        // shift=0
                      ((abs_offset & 0xFFF) << 10) | // imm12
                      ((base_reg as u32) << 5) | // Rn
                      10u32;           // Rd=x10 (scratch)
    code.extend_from_slice(&sub_inst.to_le_bytes());
        
        // LDR <dst>, [x10]
        let ldr_inst = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                      (0b111u32 << 27)) |         // [26] = 0
                      (0b01u32 << 24)) |        // [11:10] = 00
                      (10u32 << 5) |           // [9:5] = Rn = x10
                      (dst as u32);            // [4:0] = Rt
        code.extend_from_slice(&ldr_inst.to_le_bytes());
    } else {
    return Err(RasError::EncodingError(
            format!("LDR offset {} out of range (must be -4095 to 4095)", offset)
    ));
    }
    
    Ok(code)
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
    encode_mir_instruction_aarch64_with_context(
        assembler,
        inst, reg_alloc, stack_slots, stack_size, func_name,
        &std::collections::HashMap::new(), 0
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
    _current_offset: usize,
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
            
            // Encode binary operation
            match op {
                IntBinOp::Add => {
                    // ADD dst, lhs, rhs
                    // ARM64 encoding: [31]=sf(1), [30]=S(0), [29:24]=opcode(010110), [23:22]=shift(00), [21:16]=imm6(0), [15:10]=Rm, [9:5]=Rn, [4:0]=Rd
                    let add_inst = (0b1u32 << 31) |           // [31] = sf (1 for 64-bit)
                                  (0b0u32 << 30) |            // [30] = S (0, non-setting)
                                  (0b010110u32 << 24) |      // [29:24] = opcode (010110 for ADD)
                                  (0b00u32 << 22) |           // [23:22] = shift type (00 for LSL)
                                  (0b000000u32 << 16) |      // [21:16] = imm6 (0 for no shift)
                                  ((rhs_reg as u32) << 10) | // [15:10] = Rm (second source)
                                  ((lhs_reg as u32) << 5) |  // [9:5] = Rn (first source)
                                  (dst_reg as u32);          // [4:0] = Rd (destination)
                    code.extend_from_slice(&add_inst.to_le_bytes());
                }
                IntBinOp::Sub => {
                    // SUB dst, lhs, rhs
                    // Encoding: sf=1, op=1, S=0, shift=00, Rm=<rhs>, imm6=0, Rn=<lhs>, Rd=<dst>
                    // Format: [31]=sf(1), [30]=S(0), [29:24]=opcode(010111), [23:22]=shift(00), [21:16]=imm6(0), [15:10]=Rm, [9:5]=Rn, [4:0]=Rd
                    let sub_inst = (0b1u32 << 31) |           // [31] = sf (1 for 64-bit)
                                  (0b0u32 << 30) |            // [30] = S (0, non-setting)
                                  (0b010111u32 << 24) |      // [29:24] = opcode (010111 for SUB)
                                  (0b00u32 << 22) |           // [23:22] = shift type (00 for LSL)
                                  (0b000000u32 << 16) |      // [21:16] = imm6 (0 for no shift)
                                  ((rhs_reg as u32) << 10) | // [15:10] = Rm (second source)
                                  ((lhs_reg as u32) << 5) |  // [9:5] = Rn (first source)
                                  (dst_reg as u32);          // [4:0] = Rd (destination)
                    code.extend_from_slice(&sub_inst.to_le_bytes());
                }
                IntBinOp::Mul => {
                    // MUL dst, lhs, rhs (MADD dst, lhs, rhs, xzr)
                    // MADD encoding: sf=1, op=11011, Ra=xzr(31), Rm=<rhs>, Rn=<lhs>, Rd=<dst>
                    // Format: [31]=sf(1), [30:29]=00, [28:24]=opcode(11011), [23:21]=Ra[2:0], [20:16]=Rm, [15:10]=Ra[4:3], [9:5]=Rn, [4:0]=Rd
                    // For MUL: Ra = xzr = 31
                    let mul_inst = (0b1u32 << 31) |           // [31] = sf (1 for 64-bit)
                                  (0b00u32 << 29) |          // [30:29] = 00
                                  (0b11011u32 << 24) |       // [28:24] = opcode (11011 for MADD)
                                  (0b111u32 << 21) |         // [23:21] = Ra[2:0] (31 & 0x7 = 7)
                                  ((rhs_reg as u32) << 16) | // [20:16] = Rm (second source)
                                  (0b11u32 << 10) |          // [15:10] = Ra[4:3] (31 >> 3 = 3)
                                  ((lhs_reg as u32) << 5) |  // [9:5] = Rn (first source)
                                  (dst_reg as u32);          // [4:0] = Rd (destination)
                    code.extend_from_slice(&mul_inst.to_le_bytes());
                }
                _ => {
                    return Err(RasError::EncodingError(
                        format!("Unsupported IntBinary operation: {:?}", op)
                    ));
                }
            }
            
            // Store result to destination
            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg) {
                    let adjusted_offset = *offset - (stack_size as i32);
                    code.extend_from_slice(&encode_str_aarch64(
                        dst_reg_str,
                        29, // x29 (FP)
                        adjusted_offset,
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
                            let adjusted_base_offset = *base_offset - (stack_size as i32);
                            code.extend_from_slice(&encode_ldr_aarch64(
                                base_tmp,
                                29, // x29 (FP)
                                adjusted_base_offset,
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
                    let adjusted_offset = *offset - (stack_size as i32);
                    code.extend_from_slice(&encode_str_aarch64(
                        tmp_reg_str,
                        29, // x29 (FP)
                        adjusted_offset,
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
                            let adjusted_base_offset = *base_offset - (stack_size as i32);
                            code.extend_from_slice(&encode_ldr_aarch64(
                                base_tmp,
                                29, // x29 (FP)
                                adjusted_base_offset,
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
            
            // CMP instruction: CMP lhs, rhs
            // Encoding: sf=1 (64-bit), op=1, S=1, shift=00, Rm=rhs, imm6=0, Rn=lhs, Rd=31 (xzr)
            let is_64bit = ty.size_bytes() == 8;
            let cmp_inst = ((((if is_64bit { 1u32 } else { 0u32 }) << 31) | // sf
                          (0b1u32 << 30) |        // op=1
                          (0b1u32 << 29) |        // S=1 (set flags)
                          (0b11010u32 << 24)) |         // shift=00
                          ((rhs_reg as u32) << 16)) |          // imm6=0
                          ((lhs_reg as u32) << 5) | // Rn
                          31u32;           // Rd=xzr
            code.extend_from_slice(&cmp_inst.to_le_bytes());
            
            // CSET instruction: CSET dst, cond
            // Encoding: sf=1, op=0, S=0, opcode=11010100, cond=<cond>, o2=0, Rn=31 (xzr), Rd=dst
            let cond_code = match op {
                IntCmpOp::Eq => 0b0000u32,  // eq
                IntCmpOp::Ne => 0b0001u32,  // ne
                IntCmpOp::SLt => 0b1011u32, // lt (signed)
                IntCmpOp::SLe => 0b1101u32, // le (signed)
                IntCmpOp::SGt => 0b1100u32, // gt (signed)
                IntCmpOp::SGe => 0b1010u32, // ge (signed)
                IntCmpOp::ULt => 0b0011u32, // lo (unsigned)
                IntCmpOp::ULe => 0b1001u32, // ls (unsigned)
                IntCmpOp::UGt => 0b1000u32, // hi (unsigned)
                IntCmpOp::UGe => 0b0010u32, // hs (unsigned)
            };
            let cset_inst = (((if is_64bit { 1u32 } else { 0u32 }) << 31) |        // S=0
                           (0b11010100u32 << 21) |  // opcode (CSET)
                           (cond_code << 12)) |        // o2=0
                           (31u32 << 5) |          // Rn=xzr
                           (dst_reg as u32);       // Rd
            code.extend_from_slice(&cset_inst.to_le_bytes());
            
            // Store result to destination
            if let Register::Virtual(vreg) = dst
                && let Some(offset) = stack_slots.get(vreg) {
                    let adjusted_offset = *offset - (stack_size as i32);
                    code.extend_from_slice(&encode_str_aarch64(
                        dst_reg_str,
                        29, // x29 (FP)
                        adjusted_offset,
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
            
            // Materialize arguments into argument registers (x0-x7)
            let arg_regs = AArch64ABI::ARG_REGISTERS;
                for (i, arg) in args.iter().enumerate().take(8) {
                    let arg_reg_str = arg_regs[i];
                    let arg_reg = parse_register_aarch64(arg_reg_str)?;
                    materialize_operand_aarch64(assembler, arg, arg_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
                }
            
            // Handle stack arguments (args beyond 8)
            let stack_args = if args.len() > 8 { &args[8..] } else { &[] };
            let stack_space = (stack_args.len() * 8 + 15) & !15;
            
            if stack_space > 0 {
                // SUB sp, sp, #stack_space
                if stack_space > 0xFFF {
                    return Err(RasError::EncodingError(
                        format!("Stack space {} too large for single SUB", stack_space)
                    ));
                }
                let sub_inst = (((0b1u32 << 31) |        // sf=1 (64-bit)
                              (0b1u32 << 30)) |        // S=0
                              (0b100010u32 << 23)) |        // shift=0
                              ((stack_space as u32 & 0xFFF) << 10) | // imm12
                              (31u32 << 5) |         // Rn=sp (x31)
                              31u32;          // Rd=sp (x31)
                code.extend_from_slice(&sub_inst.to_le_bytes());
                
                // Store stack arguments
                for (i, arg) in stack_args.iter().enumerate() {
                    let offset = i * 8;
                    let scratch_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                        let scratch = parse_register_aarch64(scratch_str)?;
                        materialize_operand_aarch64(assembler, arg, scratch, stack_slots, reg_alloc, &mut code, stack_size)?;
                    
                    // STR scratch, [sp, #offset]
                    if offset <= 0xFFF {
                        let imm9 = (offset as u32) & 0x1FF;
                        let str_inst = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                                      (0b111u32 << 27)) |         // [21] = 0 (unscaled)
                                      (imm9 << 12)) |        // [11:10] = 00
                                      (31u32 << 5) |           // [9:5] = Rn = sp
                                      (scratch as u32);        // [4:0] = Rt
                        code.extend_from_slice(&str_inst.to_le_bytes());
                    } else {
                        return Err(RasError::EncodingError(
                            format!("Stack argument offset {} too large", offset)
                        ));
                    }
                    
                    reg_alloc.free_scratch(scratch_str);
                }
            }
            
            // For JIT, external function calls need special handling
            // For "print" intrinsic, we'll use printf with proper macOS variadic ABI
            if name == "print" && args.len() == 1 {
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
                
                // Materialize the value to print into x1 (second argument for printf)
                materialize_operand_aarch64(assembler, &args[0], 1, stack_slots, reg_alloc, &mut code, stack_size)?;
                
                // macOS variadic ABI requires:
                // 1. Allocate 32 bytes for home area (16-byte aligned)
                // 2. Format string in x0
                // 3. Value in x1
                // 4. Store x1 to [sp] (spill variadic argument to stack home area)
                
                // Allocate 48 bytes for home area (matches AOT codegen: 0x30)
                // AOT uses 48 bytes: 32 for home area + 16 for local vars
                let home_area_size = 48u32;
                let sub_inst = (((0b1u32 << 31) |        // sf=1 (64-bit)
                             (0b1u32 << 30)) |         // [29] = 0
                             (0b100010u32 << 23)) |         // [22] = 0
                             ((home_area_size & 0xFFF) << 10) | // [21:10] = imm12
                             (31u32 << 5) |           // [9:5] = Rn = sp
                             (31u32);                 // [4:0] = Rd = sp
                code.extend_from_slice(&sub_inst.to_le_bytes());
                
                // Load format string "%lld\n\0" into scratch register first
                // Format string: "%lld\n\0" = [0x25, 0x6c, 0x6c, 0x64, 0x0a, 0x00]
                let scratch_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                let scratch = parse_register_aarch64(scratch_str)?;
                
                // Format string bytes: "%lld\n\0" = [0x25, 0x6c, 0x6c, 0x64, 0x0a, 0x00]
                let format_str = b"%lld\n\0";
                let format_bytes = u64::from_le_bytes([
                    format_str[0], format_str[1], format_str[2], format_str[3],
                    format_str[4], format_str[5], 0, 0
                ]);
                
                // Load format string bytes into scratch register
                let chunk0 = (format_bytes & 0xFFFF) as u16;
                let chunk1 = ((format_bytes >> 16) & 0xFFFF) as u16;
                let chunk2 = ((format_bytes >> 32) & 0xFFFF) as u16;
                let chunk3 = ((format_bytes >> 48) & 0xFFFF) as u16;
                
                if chunk0 != 0 || (chunk1 == 0 && chunk2 == 0 && chunk3 == 0) {
                    let movz = ((0b1u32 << 31) | (0b100101u32 << 25)) |
                              ((chunk0 as u32) << 5) | (scratch as u32);
                    code.extend_from_slice(&movz.to_le_bytes());
                }
                if chunk1 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b01u32 << 21) |
                              ((chunk1 as u32) << 5) | (scratch as u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                if chunk2 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b10u32 << 21) |
                              ((chunk2 as u32) << 5) | (scratch as u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                if chunk3 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b11u32 << 21) |
                              ((chunk3 as u32) << 5) | (scratch as u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                
                // Store x1 to [sp] FIRST (spill variadic argument - CRITICAL for macOS ABI, matches AOT codegen)
                // AOT code: str x8, [x9] where x9=sp, so value goes to [sp]
                let str_x1 = ((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                            (0b111u32 << 27)) |        // [11:10] = 00
                            (31u32 << 5) |           // [9:5] = Rn = sp
                            (1u32);                  // [4:0] = Rt = x1
                code.extend_from_slice(&str_x1.to_le_bytes());
                
                // Store format string at [sp+16] (after home area, matches AOT pattern of using higher offsets)
                let str_fmt = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                             (0b111u32 << 27)) |         // [21] = 0 (unscaled)
                             (16u32 << 12)) |        // [11:10] = 00
                             (31u32 << 5) |           // [9:5] = Rn = sp
                             (scratch as u32);        // [4:0] = Rt
                code.extend_from_slice(&str_fmt.to_le_bytes());
                
                // MOV x0, sp+16 (format string address - first argument)
                // ADD x0, sp, #16
                let add_x0 = ((0b1u32 << 31) |         // [29] = 0
                            (0b100010u32 << 23)) |         // [22] = 0
                            ((16u32 & 0xFFF) << 10) | // [21:10] = imm12 = 16
                            (31u32 << 5);                  // [4:0] = Rd = x0
                code.extend_from_slice(&add_x0.to_le_bytes());
                
                // Load printf function pointer into x16 using MOVZ+MOVK
                let addr_chunk0 = (printf_addr & 0xFFFF) as u16;
                let addr_chunk1 = ((printf_addr >> 16) & 0xFFFF) as u16;
                let addr_chunk2 = ((printf_addr >> 32) & 0xFFFF) as u16;
                let addr_chunk3 = ((printf_addr >> 48) & 0xFFFF) as u16;
                
                if addr_chunk0 != 0 || (addr_chunk1 == 0 && addr_chunk2 == 0 && addr_chunk3 == 0) {
                    let movz = ((0b1u32 << 31) | (0b100101u32 << 25)) |
                              ((addr_chunk0 as u32) << 5) | (16u32);
                    code.extend_from_slice(&movz.to_le_bytes());
                }
                if addr_chunk1 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b01u32 << 21) |
                              ((addr_chunk1 as u32) << 5) | (16u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                if addr_chunk2 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b10u32 << 21) |
                              ((addr_chunk2 as u32) << 5) | (16u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                if addr_chunk3 != 0 {
                    let movk = (0b1u32 << 31) | (0b100101u32 << 25) | (0b11u32 << 21) |
                              ((addr_chunk3 as u32) << 5) | (16u32);
                    code.extend_from_slice(&movk.to_le_bytes());
                }
                
                // BLR x16 (call printf indirectly)
                let blr_inst = (0b11010110u32 << 24) | (0b00001u32 << 21) | (16u32 << 5);
                code.extend_from_slice(&blr_inst.to_le_bytes());
                
                // Flush stdout (optional but good practice)
                // For now, skip fflush to keep it simple - can add later if needed
                
                // Restore stack (ADD sp, sp, #32)
                let add_inst = ((0b1u32 << 31) |         // [29] = 0
                             (0b100010u32 << 23)) |         // [22] = 0
                             ((home_area_size & 0xFFF) << 10) | // [21:10] = imm12 = 32
                             (31u32 << 5) |           // [9:5] = Rn = sp
                             (31u32);                 // [4:0] = Rd = sp
                code.extend_from_slice(&add_inst.to_le_bytes());
                
                reg_alloc.free_scratch(scratch_str);
            } else {
                // Check if this is an internal function call
                // Try exact match, then try with/without @ prefix
                let is_internal = if let Some(module_ptr) = assembler.current_module {
                    unsafe {
                        (*module_ptr).functions.contains_key(name) ||
                        (if name.starts_with('@') {
                            (*module_ptr).functions.contains_key(&name[1..])
                        } else {
                            (*module_ptr).functions.contains_key(&format!("@{}", name))
                        })
                    }
                } else {
                    false
                };
                
                if is_internal {
                    // Internal function call - use direct call (BL) with PC-relative addressing
                    // For JIT, we don't know the exact offset yet, so we'll use a placeholder
                    // In a full implementation, we'd do a two-pass compilation or use a fixup table
                    // For now, we'll use a relative offset of 0 (will be patched later)
                    
                    // Materialize arguments into argument registers (x0-x7)
                    let arg_regs = AArch64ABI::ARG_REGISTERS;
                for (i, arg) in args.iter().enumerate().take(8) {
                    let arg_reg_str = arg_regs[i];
                    let arg_reg = parse_register_aarch64(arg_reg_str)?;
                    materialize_operand_aarch64(assembler, arg, arg_reg, stack_slots, reg_alloc, &mut code, stack_size)?;
                }
                    
                    // Handle stack arguments (args beyond 8)
                    let stack_args = if args.len() > 8 { &args[8..] } else { &[] };
                    let stack_space = (stack_args.len() * 8 + 15) & !15;
                    
                    if stack_space > 0 {
                        // SUB sp, sp, #stack_space
                        if stack_space > 0xFFF {
                            return Err(RasError::EncodingError(
                                format!("Stack space {} too large for single SUB", stack_space)
                            ));
                        }
                        // SUB sp, sp, #stack_space (op=1 for SUB, not op=0 for ADD!)
                        let sub_inst = (((0b1u32 << 31) | (0b1u32 << 30)) |
                                      (0b100010u32 << 23)) |
                                      ((stack_space as u32 & 0xFFF) << 10) |
                                      (31u32 << 5) | (31u32);
                        code.extend_from_slice(&sub_inst.to_le_bytes());
                        
                        // Store stack arguments
                        for (i, arg) in stack_args.iter().enumerate() {
                            let offset = i * 8;
                            let scratch_str = reg_alloc.alloc_scratch().unwrap_or("x9");
                        let scratch = parse_register_aarch64(scratch_str)?;
                        materialize_operand_aarch64(assembler, arg, scratch, stack_slots, reg_alloc, &mut code, stack_size)?;
                            
                            // STR scratch, [sp, #offset]
                            if offset <= 0xFFF {
                                let imm9 = (offset as u32) & 0x1FF;
                                let str_inst = (((0b11u32 << 30) |        // [31:30] = size (11 = 64-bit)
                                              (0b111u32 << 27)) |         // [21] = 0 (unscaled)
                                              (imm9 << 12)) |        // [11:10] = 00
                                              (31u32 << 5) |           // [9:5] = Rn = sp
                                              (scratch as u32);        // [4:0] = Rt
                                code.extend_from_slice(&str_inst.to_le_bytes());
                            } else {
                                return Err(RasError::EncodingError(
                                    format!("Stack argument offset {} too large", offset)
                                ));
                            }
                            reg_alloc.free_scratch(scratch_str);
                        }
                    }
                    
                    // BL <label> - direct call with PC-relative offset
                    // Encoding: op=100101, imm26=<signed offset / 4>
                    // Use function_offsets which contains estimated offsets (updated as we compile)
                    let current_offset = code.len();
                    // Try multiple name variations (with and without @ prefix)
                    let target_offset = function_offsets.get(name)
                        .or_else(|| {
                            if name.starts_with('@') {
                                function_offsets.get(&name[1..])
                            } else {
                                function_offsets.get(&format!("@{}", name))
                            }
                        });
                    
                    if let Some(&target_offset) = target_offset {
                        // BL uses PC-relative addressing: offset = (target - (PC + 4)) / 4
                        // PC is the address of the BL instruction, but ARM uses PC+4
                        let pc_at_bl = current_offset as i64;
                        let offset = (target_offset as i64) - (pc_at_bl + 4);
                        if offset % 4 != 0 {
                            return Err(RasError::EncodingError(
                                format!("Function offset {} is not 4-byte aligned", offset)
                            ));
                        }
                        let imm26_signed = offset / 4;
                        // BL uses signed 26-bit immediate: range is -2^25 to 2^25-1 (±128MB)
                        if !(-(1i64 << 25)..(1i64 << 25)).contains(&imm26_signed) {
                            return Err(RasError::EncodingError(
                                format!("Function offset {} too large for BL instruction (max ±128MB)", offset)
                            ));
                        }
                        // Convert to unsigned 26-bit value (two's complement)
                        let imm26 = (imm26_signed as u32) & 0x3FFFFFF;
                        let bl_inst = (0b100101u32 << 26) | imm26;
                        code.extend_from_slice(&bl_inst.to_le_bytes());
                    } else {
                        return Err(RasError::EncodingError(
                            format!("Internal function '{}' not found in compiled functions. Available: {:?}", 
                                name, function_offsets.keys().collect::<Vec<_>>())
                        ));
                    }
                    
                    // Restore stack if needed
                    if stack_space > 0 {
                        let add_inst = ((0b1u32 << 31) |
                                      (0b100010u32 << 23)) |
                                      ((stack_space as u32 & 0xFFF) << 10) |
                                      (31u32 << 5) | (31u32);
                        code.extend_from_slice(&add_inst.to_le_bytes());
                    }
                    
                    // Store return value if needed
                    if let Some(_ret_ty) = ret {
                        // TODO: Handle return value storage
                    }
                } else {
                    // External function call - needs runtime resolution
                    return Err(RasError::EncodingError(
                        format!("External function call '{}' requires runtime function resolution (not yet implemented for JIT)", name)
                    ));
                }
            }
            
            // Restore stack if needed (unreachable for now, but kept for future implementation)
            if stack_space > 0 {
                // ADD sp, sp, #stack_space
                let add_inst = ((0b1u32 << 31) |        // S=0
                              (0b100010u32 << 23)) |        // shift=0
                              ((stack_space as u32 & 0xFFF) << 10) | // imm12
                              (31u32 << 5) |         // Rn=sp (x31)
                              31u32;          // Rd=sp (x31)
                code.extend_from_slice(&add_inst.to_le_bytes());
            }
            
            // Store return value if needed
            if let Some(dst) = ret
                && let Register::Virtual(vreg) = dst
                    && let Some(offset) = stack_slots.get(vreg) {
                        // STR x0, [x29, #offset]
                        let adjusted_offset = *offset - (stack_size as i32);
                        code.extend_from_slice(&encode_str_aarch64(
                            "x0",
                            29, // x29 (FP)
                            adjusted_offset,
                        )?);
                    }
    }
    lamina_mir::Instruction::Jmp { .. } => {
            // Jumps are handled at block level, not instruction level
            // For now, just return empty (will be handled by block structure)
    }
    lamina_mir::Instruction::Br { .. } => {
            // Branches are handled at block level
            // For now, just return empty
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
    stack_size: usize,
) -> Result<(), RasError> {
    use lamina_mir::{Immediate, Operand, Register};
    
    match op {
    Operand::Immediate(imm) => {
            // MOV dst, #imm
            let imm_val = match imm {
                Immediate::I8(v) => *v as u64,
                Immediate::I16(v) => *v as u64,
                Immediate::I32(v) => *v as u64,
                Immediate::I64(v) => *v as u64,
                _ => return Err(RasError::EncodingError(
                    "Floating-point immediates not yet supported".to_string()
                )),
            };
            
            // Handle sign extension for signed types
            let imm_val = match imm {
                Immediate::I8(v) => *v as u64,
                Immediate::I16(v) => *v as u64,
                Immediate::I32(v) => *v as u64,
                Immediate::I64(v) => *v as u64,
                _ => imm_val,
            };
            
            // Use MOVZ for small immediates (0-65535)
            if imm_val <= 0xFFFF {
                // MOVZ: [31:30]=sf(11), [29:27]=op(010), [28:23]=100101, [22:21]=hw(00), [20:5]=imm16, [4:0]=Rd
                let mov_inst = ((0b11u32 << 30) |       // [31:30] = sf (11 = 64-bit)
                              (0b010u32 << 27) |       // [29:27] = op (010)
                              (0b100101u32 << 23)) |        // [22:21] = hw (00 = bits 0-15)
                              ((imm_val as u32 & 0xFFFF) << 5) | // [20:5] = imm16
                              (dst_reg as u32);        // [4:0] = Rd
                code.extend_from_slice(&mov_inst.to_le_bytes());
            } else {
                // For larger immediates, use MOVZ + MOVK sequence
                // Break 64-bit value into 4 chunks of 16 bits each
                let chunk0 = (imm_val & 0xFFFF) as u16;
                let chunk1 = ((imm_val >> 16) & 0xFFFF) as u16;
                let chunk2 = ((imm_val >> 32) & 0xFFFF) as u16;
                let chunk3 = ((imm_val >> 48) & 0xFFFF) as u16;
                
                // Emit MOVZ for the first non-zero chunk (or chunk0 if all are zero)
                let first_chunk = if chunk0 != 0 {
                    (chunk0, 0b00u32) // hw=00
                } else if chunk1 != 0 {
                    (chunk1, 0b01u32) // hw=01
                } else if chunk2 != 0 {
                    (chunk2, 0b10u32) // hw=10
                } else {
                    (chunk3, 0b11u32) // hw=11
                };
                
                // MOVZ: [31:30]=sf(11), [29:27]=op(010), [28:23]=100101, [22:21]=hw, [20:5]=imm16, [4:0]=Rd
                let movz_inst = (0b11u32 << 30) |       // [31:30] = sf (11 = 64-bit)
                               (0b010u32 << 27) |       // [29:27] = op (010)
                               (0b100101u32 << 23) |    // [28:23] = opcode (100101 = MOVZ)
                               (first_chunk.1 << 21) |  // [22:21] = hw
                               ((first_chunk.0 as u32) << 5) | // [20:5] = imm16
                               (dst_reg as u32);        // [4:0] = Rd
                code.extend_from_slice(&movz_inst.to_le_bytes());
                
                // Emit MOVK for remaining non-zero chunks
                // MOVK: sf=1, op=100101, hw=<hw>, imm16=<chunk>, Rd=<dst>
                if chunk0 != 0 && first_chunk.1 != 0b00 {
                    let movk_inst = ((0b1u32 << 31) |        // sf=1 (64-bit)
                                   (0b100101u32 << 25)) |        // hw=00
                                   ((chunk0 as u32) << 5) | // imm16
                                   (dst_reg as u32);        // Rd
                    code.extend_from_slice(&movk_inst.to_le_bytes());
                }
                if chunk1 != 0 && first_chunk.1 != 0b01 {
                    let movk_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                   (0b100101u32 << 25) |    // opcode (MOVK)
                                   (0b01u32 << 21) |        // hw=01
                                   ((chunk1 as u32) << 5) | // imm16
                                   (dst_reg as u32);        // Rd
                    code.extend_from_slice(&movk_inst.to_le_bytes());
                }
                if chunk2 != 0 && first_chunk.1 != 0b10 {
                    let movk_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                   (0b100101u32 << 25) |    // opcode (MOVK)
                                   (0b10u32 << 21) |        // hw=10
                                   ((chunk2 as u32) << 5) | // imm16
                                   (dst_reg as u32);        // Rd
                    code.extend_from_slice(&movk_inst.to_le_bytes());
                }
                if chunk3 != 0 && first_chunk.1 != 0b11 {
                    let movk_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                   (0b100101u32 << 25) |    // opcode (MOVK)
                                   (0b11u32 << 21) |        // hw=11
                                   ((chunk3 as u32) << 5) | // imm16
                                   (dst_reg as u32);        // Rd
                    code.extend_from_slice(&movk_inst.to_le_bytes());
                }
            }
    }
    Operand::Register(Register::Virtual(vreg)) => {
            // Load from stack slot
            // Stack slots are at negative offsets relative to FP after stack allocation
            // But FP is set before stack allocation, so adjust offset by -stack_size
            if let Some(offset) = stack_slots.get(vreg) {
                let adjusted_offset = *offset - (stack_size as i32);
                let dst_reg_str = format!("x{}", dst_reg);
                code.extend_from_slice(&encode_ldr_aarch64(
                    &dst_reg_str,
                    29, // x29 (FP)
                    adjusted_offset,
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

