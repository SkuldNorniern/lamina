//! ras assembler implementation
//!
//! This module provides the main assembler functionality, reusing code from
//! lamina's mir_codegen where possible.

use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
use crate::object::ObjectWriter;
use crate::parser::AssemblyParser;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// ras assembler - converts assembly text to object files
pub struct RasAssembler {
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    encoder: Box<dyn InstructionEncoder>,
    object_writer: Box<dyn ObjectWriter>,
}

impl RasAssembler {
    /// Create a new ras assembler
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        // Create encoder based on target architecture
        let encoder: Box<dyn InstructionEncoder> = match target_arch {
            TargetArchitecture::X86_64 => {
                Box::new(crate::encoder::x86_64::X86_64Encoder::new())
            }
            TargetArchitecture::Aarch64 => {
                Box::new(crate::encoder::aarch64::AArch64Encoder::new())
            }
            TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
                return Err(RasError::UnsupportedTarget(
                    "RISC-V encoder not yet implemented".to_string(),
                ));
            }
            _ => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported architecture: {:?}",
                    target_arch
                )));
            }
        };

        // Create object writer based on target OS
        let object_writer: Box<dyn ObjectWriter> = match target_os {
            TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD => {
                Box::new(crate::object::ElfWriter::new())
            }
            TargetOperatingSystem::MacOS => {
                Box::new(crate::object::MachOWriter::new())
            }
            TargetOperatingSystem::Windows => {
                Box::new(crate::object::CoffWriter::new())
            }
            _ => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported operating system: {:?}",
                    target_os
                )));
            }
        };

        Ok(Self {
            target_arch,
            target_os,
            encoder,
            object_writer,
        })
    }

    /// Assemble assembly text to object file
    pub fn assemble_text_to_object(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        // 1. Parse assembly text
        let mut parser = AssemblyParser::new();
        let parsed = parser
            .parse(asm_text)
            .map_err(|e| RasError::ParseError(e.to_string()))?;

        // 2. Encode instructions to binary
        let mut code = Vec::new();
        for inst in &parsed.instructions {
            let bytes = self
                .encoder
                .encode_instruction(inst)
                .map_err(|e| RasError::EncodingError(e.to_string()))?;
            code.extend_from_slice(&bytes);
        }

        // 3. Generate object file
        self.object_writer
            .write_object_file(
                output_path,
                &code,
                &parsed.sections,
                &parsed.symbols,
                self.target_arch,
                self.target_os,
            )
            .map_err(|e| RasError::ObjectError(e.to_string()))?;

        Ok(())
    }

    /// Compile MIR module directly to binary (for runtime compilation)
    ///
    /// This method reuses code from lamina's mir_codegen but generates binary
    /// instead of assembly text. It's used for runtime compilation (JIT).
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        // Reuse register allocation and ABI from mir_codegen
        match self.target_arch {
            TargetArchitecture::X86_64 => {
                self.compile_mir_x86_64(module)
            }
            TargetArchitecture::Aarch64 => {
                self.compile_mir_aarch64(module)
            }
            _ => Err(RasError::UnsupportedTarget(format!(
                "MIR compilation not supported for architecture: {:?}",
                self.target_arch
            ))),
        }
    }

    /// Compile MIR to binary for x86_64
    ///
    /// This reuses the instruction emission logic from mir_codegen/x86_64
    /// but generates binary instead of assembly text.
    #[cfg(feature = "encoder")]
    fn compile_mir_x86_64(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
        use lamina_mir::{Instruction as MirInst, Register};

        let abi = X86ABI::new(self.target_os);
        let mut code = Vec::new();

        for (func_name, func) in &module.functions {
            let mut reg_alloc = X64RegAlloc::new(self.target_os);
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
            let prologue = self.encode_prologue_x86_64(stack_size as u32)?;
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
                            let mov_bytes = self.encode_mov_reg_mem_x86_64(
                                arg_regs[index],
                                *slot_off,
                            )?;
                            code.extend_from_slice(&mov_bytes);
                        } else {
                            // Handle stack arguments
                            let caller_off = 16 + ((index - arg_regs.len()) as i32) * 8;
                            let mov1 = self.encode_mov_mem_reg_x86_64(caller_off, "rax")?;
                            code.extend_from_slice(&mov1);
                            let mov2 = self.encode_mov_reg_mem_x86_64("rax", *slot_off)?;
                            code.extend_from_slice(&mov2);
                        }
                    }
                }
            }

            // Compile each block
            for block in &func.blocks {
                for inst in &block.instructions {
                    let inst_bytes = self.encode_mir_instruction_x86_64(
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
            let epilogue = self.encode_epilogue_x86_64()?;
            code.extend_from_slice(&epilogue);

            // RET instruction
            code.push(0xC3);
        }

        Ok(code)
    }

    /// Encode x86_64 prologue
    fn encode_prologue_x86_64(&mut self, stack_size: u32) -> Result<Vec<u8>, RasError> {
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
            code.extend_from_slice(&(stack_size as u32).to_le_bytes());
        }
        Ok(code)
    }

    /// Encode x86_64 epilogue
    fn encode_epilogue_x86_64(&mut self) -> Result<Vec<u8>, RasError> {
        let mut code = Vec::new();
        // mov rsp, rbp: 48 89 EC
        code.extend_from_slice(&[0x48, 0x89, 0xEC]);
        // pop rbp: 5D
        code.push(0x5D);
        Ok(code)
    }

    /// Encode MOV register to memory
    fn encode_mov_reg_mem_x86_64(
        &mut self,
        src_reg: &str,
        offset: i32,
    ) -> Result<Vec<u8>, RasError> {
        // MOV [rbp+offset], reg: REX.W + 89 /r
        let mut code = Vec::new();
        let reg_enc = self.parse_register_x86_64(src_reg)?;
        code.push(0x48); // REX.W
        code.push(0x89);
        // ModR/M: [rbp] + reg encoding
        // Mod=01 (disp8/32), R/M=101 (rbp), REG=reg_enc
        code.push(0x45 | (reg_enc << 3));
        // For 32-bit offset, we need 4 bytes
        if offset >= -128 && offset <= 127 {
            code.push(offset as u8);
        } else {
            code.push(0x85 | (reg_enc << 3)); // Mod=10 (disp32), R/M=101
            code.extend_from_slice(&offset.to_le_bytes());
        }
        Ok(code)
    }

    /// Encode MOV memory to register
    fn encode_mov_mem_reg_x86_64(
        &mut self,
        offset: i32,
        dst_reg: &str,
    ) -> Result<Vec<u8>, RasError> {
        // MOV reg, [rbp+offset]: REX.W + 8B /r
        let mut code = Vec::new();
        let reg_enc = self.parse_register_x86_64(dst_reg)?;
        code.push(0x48); // REX.W
        code.push(0x8B);
        // ModR/M: [rbp] + reg encoding
        if offset >= -128 && offset <= 127 {
            code.push(0x45 | (reg_enc << 3)); // Mod=01 (disp8)
            code.push(offset as u8);
        } else {
            code.push(0x85 | (reg_enc << 3)); // Mod=10 (disp32)
            code.extend_from_slice(&offset.to_le_bytes());
        }
        Ok(code)
    }

    /// Parse register name to encoding (x86_64)
    fn parse_register_x86_64(&self, reg: &str) -> Result<u8, RasError> {
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
        &mut self,
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

    /// Compile MIR to binary for AArch64
    ///
    /// This reuses the instruction emission logic from mir_codegen/aarch64
    /// but generates binary instead of assembly text.
    #[cfg(feature = "encoder")]
    fn compile_mir_aarch64(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        use lamina_codegen::aarch64::{A64RegAlloc, AArch64ABI, FrameMap};
        use lamina_mir::{Instruction as MirInst, Register};

        let abi = AArch64ABI::new(self.target_os);
        let mut code = Vec::new();

        for (func_name, func) in &module.functions {
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

            // Generate function prologue (binary encoded)
            let prologue = self.encode_prologue_aarch64(stack_size)?;
            code.extend_from_slice(&prologue);

            // Handle function parameters (reuse ABI logic)
            if !func.sig.params.is_empty() {
                let arg_regs = AArch64ABI::ARG_REGISTERS;
                for (index, param) in func.sig.params.iter().enumerate() {
                    if let Register::Virtual(vreg) = &param.reg
                        && let Some(slot_off) = stack_slots.get(vreg)
                    {
                        if index < arg_regs.len() {
                            // STR from argument register to stack slot
                            let str_bytes = self.encode_str_aarch64(
                                arg_regs[index],
                                29, // x29 (FP)
                                *slot_off,
                            )?;
                            code.extend_from_slice(&str_bytes);
                        } else {
                            // Handle stack arguments (AAPCS64: stack args start at [sp, #0])
                            let caller_off = ((index - arg_regs.len()) * 8) as i32;
                            let ldr1 = self.encode_ldr_aarch64("x10", 29, caller_off)?;
                            code.extend_from_slice(&ldr1);
                            let str1 = self.encode_str_aarch64("x10", 29, *slot_off)?;
                            code.extend_from_slice(&str1);
                        }
                    }
                }
            }

            // Compile each block
            for block in &func.blocks {
                for inst in &block.instructions {
                    let inst_bytes = self.encode_mir_instruction_aarch64(
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
            let epilogue = self.encode_epilogue_aarch64()?;
            code.extend_from_slice(&epilogue);

            // RET instruction (x30 is LR)
            code.extend_from_slice(&self.encode_ret_aarch64(30)?);
        }

        Ok(code)
    }

    /// Encode AArch64 prologue
    fn encode_prologue_aarch64(&mut self, stack_size: usize) -> Result<Vec<u8>, RasError> {
        let mut code = Vec::new();
        
        // STP x29, x30, [sp, #-16]! (store pair, pre-index)
        // Format: STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
        // Encoding bits: [31:30]=opc (00=64-bit), [29]=V (0), [28]=L (0=store), [27:23]=imm7, [22:16]=Rt2, [15:10]=Rn, [9:5]=Rt
        // imm7 = -16 / 8 = -2, encoded as signed 7-bit: 0b1111110 (bits 6:0 of imm7)
        // For pre-index: bit 11=1 (pre-index), imm7 bits [6:0] = signed offset/8
        let imm7 = 0b1111110u32; // -2 in 7-bit signed (0x7E)
        let stp_inst = (0b00u32 << 30) | // opc=00 (64-bit)
                       (0b0u32 << 29) |  // V=0
                       (0b0u32 << 28) |  // L=0 (store)
                       (0b1u32 << 27) |  // pre-index
                       ((imm7 & 0x7F) << 20) | // imm7[6:0] at bits [26:20]
                       (30u32 << 16) |   // Rt2=x30 at bits [22:16]
                       (31u32 << 10) |   // Rn=sp (x31) at bits [15:10]
                       (29u32 << 5);      // Rt=x29 at bits [9:5]
        code.extend_from_slice(&stp_inst.to_le_bytes());

        // MOV x29, sp (set frame pointer)
        // ORR x29, xzr, sp (MOV is alias for ORR with zero register)
        // ORR (shifted register): sf=1, opc=01, shift=00, N=0, Rm=31 (sp), imm6=0, Rn=31 (xzr), Rd=29
        let mov_inst = (0b1u32 << 31) |    // sf=1 (64-bit)
                       (0b01u32 << 29) |   // opc=01
                       (0b01010u32 << 24) | // opcode
                       (0b0u32 << 23) |    // N=0
                       (0b00u32 << 21) |   // shift=00
                       (31u32 << 16) |     // Rm=sp (x31)
                       (0u32 << 10) |      // imm6=0
                       (31u32 << 5) |      // Rn=xzr (x31)
                       (29u32 << 0);       // Rd=x29
        code.extend_from_slice(&mov_inst.to_le_bytes());

        // SUB sp, sp, #<stack_size> (allocate stack frame)
        if stack_size > 0 {
            if stack_size > 0xFFF {
                return Err(RasError::EncodingError(
                    "Stack size too large for single SUB instruction".to_string(),
                ));
            }
            // SUB (immediate): sf=1, op=0, S=0, shift=0, imm12=<size>, Rn=31 (SP), Rd=31 (SP)
            let sub_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                          (0b0u32 << 30) |        // op=0
                          (0b0u32 << 29) |        // S=0
                          (0b100010u32 << 23) |   // opcode
                          (0b0u32 << 22) |        // shift=0
                          ((stack_size as u32 & 0xFFF) << 10) | // imm12
                          (31u32 << 5) |         // Rn=sp (x31)
                          (31u32 << 0);          // Rd=sp (x31)
            code.extend_from_slice(&sub_inst.to_le_bytes());
        }

        Ok(code)
    }

    /// Encode AArch64 epilogue
    fn encode_epilogue_aarch64(&mut self) -> Result<Vec<u8>, RasError> {
        let mut code = Vec::new();
        
        // LDP x29, x30, [sp], #16 (load pair, post-index)
        // Format: LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
        // Encoding bits: [31:30]=opc (00=64-bit), [29]=V (0), [28]=L (1=load), [27]=0 (post-index), [26:20]=imm7, [22:16]=Rt2, [15:10]=Rn, [9:5]=Rt
        // imm7 = 16 / 8 = 2
        let imm7 = 0b0000010u32; // 2 in 7-bit
        let ldp_inst = (0b00u32 << 30) | // opc=00 (64-bit)
                      (0b0u32 << 29) |  // V=0
                      (0b1u32 << 28) |  // L=1 (load)
                      (0b0u32 << 27) |  // post-index
                      ((imm7 & 0x7F) << 20) | // imm7[6:0] at bits [26:20]
                      (30u32 << 16) |   // Rt2=x30 at bits [22:16]
                      (31u32 << 10) |   // Rn=sp (x31) at bits [15:10]
                      (29u32 << 5);     // Rt=x29 at bits [9:5]
        code.extend_from_slice(&ldp_inst.to_le_bytes());

        Ok(code)
    }

    /// Encode STR instruction (AArch64)
    /// Handles both positive and negative offsets by using SUB for negative offsets
    fn encode_str_aarch64(
        &mut self,
        src_reg: &str,
        base_reg: u8,
        offset: i32,
    ) -> Result<Vec<u8>, RasError> {
        let src = self.parse_register_aarch64(src_reg)?;
        let mut code = Vec::new();
        
        if offset >= 0 && offset <= 0xFFF {
            // Positive offset: direct STR
            // STR (immediate): size=11 (64-bit), opc=00, imm12=<offset>, Rn=<base>, Rt=<src>
            let inst = 0b11_111_0_00_000000000000_00000_00000u32;
            let inst = inst | ((src as u32) << 0);
            let inst = inst | ((base_reg as u32) << 5);
            let inst = inst | ((offset as u32 & 0xFFF) << 10);
            code.extend_from_slice(&inst.to_le_bytes());
        } else if offset < 0 && offset >= -0xFFF {
            // Negative offset: use SUB to adjust base, then STR with positive offset
            // SUB x10, <base>, #abs(offset)
            let abs_offset = (-offset) as u32;
            if abs_offset > 0xFFF {
                return Err(RasError::EncodingError(
                    format!("STR offset {} out of range", offset)
                ));
            }
            let sub_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                          (0b0u32 << 30) |        // op=0
                          (0b0u32 << 29) |        // S=0
                          (0b100010u32 << 23) |   // opcode
                          (0b0u32 << 22) |        // shift=0
                          ((abs_offset & 0xFFF) << 10) | // imm12
                          ((base_reg as u32) << 5) | // Rn
                          (10u32 << 0);           // Rd=x10 (scratch)
            code.extend_from_slice(&sub_inst.to_le_bytes());
            
            // STR <src>, [x10]
            let str_inst = 0b11_111_0_00_000000000000_01010_00000u32;
            let str_inst = str_inst | ((src as u32) << 0);
            code.extend_from_slice(&str_inst.to_le_bytes());
        } else {
            return Err(RasError::EncodingError(
                format!("STR offset {} out of range (must be -4095 to 4095)", offset)
            ));
        }
        
        Ok(code)
    }

    /// Encode LDR instruction (AArch64)
    /// Handles both positive and negative offsets by using SUB for negative offsets
    fn encode_ldr_aarch64(
        &mut self,
        dst_reg: &str,
        base_reg: u8,
        offset: i32,
    ) -> Result<Vec<u8>, RasError> {
        let dst = self.parse_register_aarch64(dst_reg)?;
        let mut code = Vec::new();
        
        if offset >= 0 && offset <= 0xFFF {
            // Positive offset: direct LDR
            // LDR (immediate): size=11 (64-bit), opc=01, imm12=<offset>, Rn=<base>, Rt=<dst>
            let inst = 0b11_111_0_01_000000000000_00000_00000u32;
            let inst = inst | ((dst as u32) << 0);
            let inst = inst | ((base_reg as u32) << 5);
            let inst = inst | ((offset as u32 & 0xFFF) << 10);
            code.extend_from_slice(&inst.to_le_bytes());
        } else if offset < 0 && offset >= -0xFFF {
            // Negative offset: use SUB to adjust base, then LDR with positive offset
            // SUB x10, <base>, #abs(offset)
            let abs_offset = (-offset) as u32;
            if abs_offset > 0xFFF {
                return Err(RasError::EncodingError(
                    format!("LDR offset {} out of range", offset)
                ));
            }
            let sub_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                          (0b0u32 << 30) |        // op=0
                          (0b0u32 << 29) |        // S=0
                          (0b100010u32 << 23) |   // opcode
                          (0b0u32 << 22) |        // shift=0
                          ((abs_offset & 0xFFF) << 10) | // imm12
                          ((base_reg as u32) << 5) | // Rn
                          (10u32 << 0);           // Rd=x10 (scratch)
            code.extend_from_slice(&sub_inst.to_le_bytes());
            
            // LDR <dst>, [x10]
            let ldr_inst = 0b11_111_0_01_000000000000_01010_00000u32;
            let ldr_inst = ldr_inst | ((dst as u32) << 0);
            code.extend_from_slice(&ldr_inst.to_le_bytes());
        } else {
            return Err(RasError::EncodingError(
                format!("LDR offset {} out of range (must be -4095 to 4095)", offset)
            ));
        }
        
        Ok(code)
    }

    /// Encode RET instruction (AArch64)
    fn encode_ret_aarch64(&mut self, reg: u8) -> Result<Vec<u8>, RasError> {
        // RET: op=1101011, op2=00000, Rn=<reg>, op3=00000
        let inst = 0b1101011_0_000_00000_00000_00000u32;
        let inst = inst | ((reg as u32) << 5);
        Ok(inst.to_le_bytes().to_vec())
    }

    /// Parse register name to encoding (AArch64)
    fn parse_register_aarch64(&self, reg: &str) -> Result<u8, RasError> {
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
        &mut self,
        inst: &lamina_mir::Instruction,
        reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
        stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        _stack_size: usize,
        _func_name: &str,
    ) -> Result<Vec<u8>, RasError> {
        use lamina_mir::{IntBinOp, Operand, Register};
        let mut code = Vec::new();
        
        match inst {
            lamina_mir::Instruction::Ret { value } => {
                if let Some(v) = value {
                    // Load return value to x0
                    self.materialize_operand_aarch64(v, 0, stack_slots, reg_alloc, &mut code)?;
                }
                code.extend_from_slice(&self.encode_ret_aarch64(30)?);
            }
            lamina_mir::Instruction::IntBinary { op, dst, lhs, rhs, ty: _ } => {
                // Allocate scratch registers
                let lhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
                let rhs_reg_str = reg_alloc.alloc_scratch().unwrap_or("x11");
                let dst_reg_str = reg_alloc.alloc_scratch().unwrap_or("x12");
                let lhs_reg = self.parse_register_aarch64(lhs_reg_str)?;
                let rhs_reg = self.parse_register_aarch64(rhs_reg_str)?;
                let dst_reg = self.parse_register_aarch64(dst_reg_str)?;
                
                // Materialize operands
                self.materialize_operand_aarch64(lhs, lhs_reg, stack_slots, reg_alloc, &mut code)?;
                self.materialize_operand_aarch64(rhs, rhs_reg, stack_slots, reg_alloc, &mut code)?;
                
                // Encode binary operation
                match op {
                    IntBinOp::Add => {
                        // ADD dst, lhs, rhs
                        let add_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                      (0b0u32 << 30) |        // op=0
                                      (0b0u32 << 29) |        // S=0
                                      (0b01011u32 << 24) |    // opcode
                                      (0b0u32 << 22) |        // shift=00
                                      ((rhs_reg as u32) << 16) | // Rm
                                      (0u32 << 10) |         // imm6=0
                                      ((lhs_reg as u32) << 5) | // Rn
                                      (dst_reg as u32);       // Rd
                        code.extend_from_slice(&add_inst.to_le_bytes());
                    }
                    IntBinOp::Sub => {
                        // SUB dst, lhs, rhs
                        let sub_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                      (0b0u32 << 30) |        // op=0
                                      (0b0u32 << 29) |        // S=0
                                      (0b11011u32 << 24) |    // opcode (SUB)
                                      (0b0u32 << 22) |        // shift=00
                                      ((rhs_reg as u32) << 16) | // Rm
                                      (0u32 << 10) |         // imm6=0
                                      ((lhs_reg as u32) << 5) | // Rn
                                      (dst_reg as u32);       // Rd
                        code.extend_from_slice(&sub_inst.to_le_bytes());
                    }
                    IntBinOp::Mul => {
                        // MUL dst, lhs, rhs (MADD dst, lhs, rhs, xzr)
                        let mul_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                      (0b00u32 << 29) |       // opcode
                                      (0b11011u32 << 24) |    // opcode (MADD)
                                      ((rhs_reg as u32) << 16) | // Rm
                                      (0u32 << 15) |         // o0=0
                                      ((lhs_reg as u32) << 10) | // Ra (xzr=31, but we use 0 for MUL)
                                      ((lhs_reg as u32) << 5) | // Rn
                                      (dst_reg as u32);       // Rd
                        // Actually, MUL is MADD with Ra=xzr (31)
                        let mul_inst = (0b1u32 << 31) |
                                      (0b00u32 << 29) |
                                      (0b11011u32 << 24) |
                                      ((rhs_reg as u32) << 16) |
                                      (31u32 << 10) |         // Ra=xzr
                                      ((lhs_reg as u32) << 5) |
                                      (dst_reg as u32);
                        code.extend_from_slice(&mul_inst.to_le_bytes());
                    }
                    _ => {
                        return Err(RasError::EncodingError(
                            format!("Unsupported IntBinary operation: {:?}", op)
                        ));
                    }
                }
                
                // Store result to destination
                if let Register::Virtual(vreg) = dst {
                    if let Some(offset) = stack_slots.get(vreg) {
                        code.extend_from_slice(&self.encode_str_aarch64(
                            dst_reg_str,
                            29, // x29 (FP)
                            *offset,
                        )?);
                    }
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
                let tmp_reg = self.parse_register_aarch64(tmp_reg_str)?;
                
                // Handle address mode
                match addr {
                    AddressMode::BaseOffset { base, offset } => {
                        // Materialize base register
                        let base_reg_str = if let Register::Virtual(vreg) = base {
                            if let Some(base_offset) = stack_slots.get(vreg) {
                                let base_tmp = reg_alloc.alloc_scratch().unwrap_or("x11");
                                code.extend_from_slice(&self.encode_ldr_aarch64(
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
                        let base_reg = self.parse_register_aarch64(base_reg_str)?;
                        
                        // LDR tmp, [base_reg, #offset]
                        if *offset >= 0 && (*offset as u32) <= 0xFFF {
                            let ldr_inst = 0b11_111_0_01_000000000000_00000_00000u32;
                            let ldr_inst = ldr_inst | ((tmp_reg as u32) << 0);
                            let ldr_inst = ldr_inst | ((base_reg as u32) << 5);
                            let ldr_inst = ldr_inst | (((*offset as u32) & 0xFFF) << 10);
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
                if let Register::Virtual(vreg) = dst {
                    if let Some(offset) = stack_slots.get(vreg) {
                        code.extend_from_slice(&self.encode_str_aarch64(
                            tmp_reg_str,
                            29, // x29 (FP)
                            *offset,
                        )?);
                    }
                }
                
                reg_alloc.free_scratch(tmp_reg_str);
            }
            lamina_mir::Instruction::Store { src, addr, ty: _, .. } => {
                use lamina_mir::AddressMode;
                // Store from source to memory address
                let src_reg_str = reg_alloc.alloc_scratch().unwrap_or("x10");
                let src_reg = self.parse_register_aarch64(src_reg_str)?;
                
                // Materialize source
                self.materialize_operand_aarch64(src, src_reg, stack_slots, reg_alloc, &mut code)?;
                
                // Handle address mode
                match addr {
                    AddressMode::BaseOffset { base, offset } => {
                        // Materialize base register
                        let base_reg_str = if let Register::Virtual(vreg) = base {
                            if let Some(base_offset) = stack_slots.get(vreg) {
                                let base_tmp = reg_alloc.alloc_scratch().unwrap_or("x11");
                                code.extend_from_slice(&self.encode_ldr_aarch64(
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
                        let base_reg = self.parse_register_aarch64(base_reg_str)?;
                        
                        // STR src_reg, [base_reg, #offset]
                        if *offset >= 0 && (*offset as u32) <= 0xFFF {
                            let str_inst = 0b11_111_0_00_000000000000_00000_00000u32;
                            let str_inst = str_inst | ((src_reg as u32) << 0);
                            let str_inst = str_inst | ((base_reg as u32) << 5);
                            let str_inst = str_inst | (((*offset as u32) & 0xFFF) << 10);
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
        &mut self,
        op: &lamina_mir::Operand,
        dst_reg: u8,
        stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
        code: &mut Vec<u8>,
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
                // Use MOVZ for small immediates
                if imm_val <= 0xFFFF {
                    let mov_inst = (0b1u32 << 31) |        // sf=1 (64-bit)
                                  (0b100101u32 << 25) |    // opcode (MOVZ)
                                  (0b10u32 << 23) |        // hw=00
                                  ((imm_val as u32 & 0xFFFF) << 5) | // imm16
                                  (dst_reg as u32);        // Rd
                    code.extend_from_slice(&mov_inst.to_le_bytes());
                } else {
                    return Err(RasError::EncodingError(
                        "Large immediates require MOVZ+MOVK sequence (not yet implemented)".to_string()
                    ));
                }
            }
            Operand::Register(Register::Virtual(vreg)) => {
                // Load from stack slot
                if let Some(offset) = stack_slots.get(vreg) {
                    let dst_reg_str = format!("x{}", dst_reg);
                    code.extend_from_slice(&self.encode_ldr_aarch64(
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
}

