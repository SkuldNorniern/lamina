//! x86-64 binary instruction encoder
//!
//! This encoder converts x86-64 instructions to binary machine code.
//! It reuses knowledge from lamina's mir_codegen but generates binary directly.

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

/// x86-64 instruction encoder
pub struct X86_64Encoder {
    position: usize,
}

impl Default for X86_64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Encoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }

    /// Encode REX prefix
    fn encode_rex(&self, w: bool, r: u8, x: u8, b: u8) -> u8 {
        let mut rex = 0x40;
        if w {
            rex |= 0x08;
        }
        if r > 0 {
            rex |= 0x04;
        }
        if x > 0 {
            rex |= 0x02;
        }
        if b > 0 {
            rex |= 0x01;
        }
        rex
    }

    /// Parse register name to encoding
    fn parse_register(&self, reg: &str) -> Result<u8, RasError> {
        // Remove % prefix if present
        let reg = reg.trim_start_matches('%');
        
        match reg {
            "rax" | "eax" | "ax" | "al" => Ok(0),
            "rcx" | "ecx" | "cx" | "cl" => Ok(1),
            "rdx" | "edx" | "dx" | "dl" => Ok(2),
            "rbx" | "ebx" | "bx" | "bl" => Ok(3),
            "rsp" | "esp" | "sp" | "ah" => Ok(4),
            "rbp" | "ebp" | "bp" | "ch" => Ok(5),
            "rsi" | "esi" | "si" | "dh" => Ok(6),
            "rdi" | "edi" | "di" | "bh" => Ok(7),
            "r8" => Ok(8),
            "r9" => Ok(9),
            "r10" => Ok(10),
            "r11" => Ok(11),
            "r12" => Ok(12),
            "r13" => Ok(13),
            "r14" => Ok(14),
            "r15" => Ok(15),
            _ => Err(RasError::EncodingError(format!(
                "Unknown register: {}",
                reg
            ))),
        }
    }
}

impl InstructionEncoder for X86_64Encoder {
    fn encode_instruction(
        &mut self,
        inst: &ParsedInstruction,
    ) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let mut code = Vec::new();

        match opcode.as_str() {
            "movq" | "mov" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "mov requires 2 operands".to_string(),
                    ));
                }

                let dst = &inst.operands[0];
                let src = &inst.operands[1];

                // Check if source is immediate
                if src.starts_with('$') {
                    // mov imm, reg
                    let imm_str = src.trim_start_matches('$');
                    let imm: i64 = imm_str
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;

                    let dst_reg = self.parse_register(dst)?;

                    // MOV r/m64, imm64: REX.W + B8+rd /0 io
                    code.push(self.encode_rex(true, 0, 0, 0)); // REX.W
                    code.push(0xB8 | dst_reg);
                    code.extend_from_slice(&imm.to_le_bytes());
                } else {
                    // mov reg, reg
                    let dst_reg = self.parse_register(dst)?;
                    let src_reg = self.parse_register(src)?;

                    // MOV r/m64, r64: REX.W + 89 /r
                    code.push(self.encode_rex(true, 0, 0, 0)); // REX.W
                    code.push(0x89);
                    code.push(0xC0 | (src_reg << 3) | dst_reg);
                }
            }
            "ret" => {
                // RET: C3
                code.push(0xC3);
            }
            "addq" | "add" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "add requires 2 operands".to_string(),
                    ));
                }

                let dst = &inst.operands[0];
                let src = &inst.operands[1];

                if src.starts_with('$') {
                    // add imm, reg
                    let imm_str = src.trim_start_matches('$');
                    let imm: i32 = imm_str
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;

                    let dst_reg = self.parse_register(dst)?;

                    // ADD r/m64, imm32: REX.W + 81 /0 id
                    code.push(self.encode_rex(true, 0, 0, 0));
                    code.push(0x81);
                    code.push(0xC0 | dst_reg);
                    code.extend_from_slice(&imm.to_le_bytes());
                } else {
                    // add reg, reg
                    let dst_reg = self.parse_register(dst)?;
                    let src_reg = self.parse_register(src)?;

                    // ADD r/m64, r64: REX.W + 01 /r
                    code.push(self.encode_rex(true, 0, 0, 0));
                    code.push(0x01);
                    code.push(0xC0 | (src_reg << 3) | dst_reg);
                }
            }
            _ => {
                return Err(RasError::EncodingError(format!(
                    "Unsupported instruction: {}",
                    opcode
                )));
            }
        }

        self.position += code.len();
        Ok(code)
    }

    fn current_position(&self) -> usize {
        self.position
    }
}

