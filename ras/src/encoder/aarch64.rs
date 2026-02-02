//! AArch64 binary instruction encoder
//!
//! This encoder converts AArch64 instructions to binary machine code.
//! AArch64 uses fixed 32-bit instruction encoding.

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

/// AArch64 instruction encoder
pub struct AArch64Encoder {
    position: usize,
}

impl Default for AArch64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Encoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }

    /// Parse register name to encoding (0-31)
    fn parse_register(&self, reg: &str) -> Result<u8, RasError> {
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
            _ => Err(RasError::EncodingError(format!(
                "Unknown register: {}",
                reg
            ))),
        }
    }

    /// Encode a 32-bit AArch64 instruction
    fn encode_u32(&self, inst: u32) -> Vec<u8> {
        inst.to_le_bytes().to_vec()
    }

    /// Encode MOV (register) instruction: MOV <Xd>, <Xn>
    /// Encoding: 0b10101010_000_<Xn>_000000_<Xd>
    fn encode_mov_reg(&self, dst: u8, src: u8) -> Vec<u8> {
        // ORR <Xd>, XZR, <Xn> (MOV is alias for ORR with zero register)
        // ORR (shifted register): sf=1, opc=01, shift=00, N=0, Rm=<src>, imm6=0, Rn=31 (XZR), Rd=<dst>
        let inst = 0b1_01_01010_0_0_000000_11111_00000_00000;
        let inst = inst | (dst as u32);
        let inst = inst | ((31u32) << 5); // XZR
        let inst = inst | ((src as u32) << 16);
        self.encode_u32(inst)
    }

    /// Encode MOV (immediate) instruction: MOV <Xd>, #<imm>
    /// For 64-bit: MOVZ or MOVN with appropriate encoding
    fn encode_mov_imm64(&self, dst: u8, imm: u64) -> Result<Vec<u8>, RasError> {
        // MOVZ: move with zero, can encode 16-bit immediate at positions 0, 16, 32, 48
        // MOVZ encoding: sf=1, op=100101, hw=00/01/10/11, imm16=<low16>, Rd=<dst>
        if imm <= 0xFFFF {
            // Can encode in single MOVZ
            let inst = 0b1_100101_10_0000000000000000_00000;
            let inst = inst | (dst as u32);
            let inst = inst | ((imm as u32 & 0xFFFF) << 5);
            Ok(self.encode_u32(inst))
        } else if (imm & 0xFFFF) == 0 && (imm >> 16) <= 0xFFFF {
            // Can encode in MOVZ with hw=1
            let inst = 0b1_100101_10_01_0000000000000000_00000;
            let inst = inst | (dst as u32);
            let inst = inst | (((imm >> 16) as u32 & 0xFFFF) << 5);
            Ok(self.encode_u32(inst))
        } else {
            // For larger immediates, use MOVZ + MOVK sequence
            // For now, return error - full implementation would emit multiple instructions
            Err(RasError::EncodingError(
                "Large immediate values require MOVZ+MOVK sequence (not yet implemented)".to_string(),
            ))
        }
    }

    /// Encode ADD (register) instruction: ADD <Xd>, <Xn>, <Xm>
    /// Encoding: sf=1, op=0, S=0, shift=00, Rm=<Xm>, imm6=0, Rn=<Xn>, Rd=<Xd>
    /// ARM64 format: [31]=sf(1), [30]=S(0), [29:24]=opcode(010110), [23:22]=shift(00), [21:16]=imm6(0), [15:10]=Rm, [9:5]=Rn, [4:0]=Rd
    fn encode_add_reg(&self, dst: u8, src1: u8, src2: u8) -> Vec<u8> {
        let inst = (0b1u32 << 31) |           // [31] = sf (1 for 64-bit)
                  (0b0u32 << 30) |            // [30] = S (0, non-setting)
                  (0b010110u32 << 24) |      // [29:24] = opcode (010110 for ADD)
                  (0b00u32 << 22) |           // [23:22] = shift type (00 for LSL)
                  (0b000000u32 << 16) |      // [21:16] = imm6 (0 for no shift)
                  ((src2 as u32) << 10) |    // [15:10] = Rm (second source)
                  ((src1 as u32) << 5) |     // [9:5] = Rn (first source)
                  (dst as u32);              // [4:0] = Rd (destination)
        self.encode_u32(inst)
    }

    /// Encode ADD (immediate) instruction: ADD <Xd>, <Xn>, #<imm>
    /// Encoding: sf=1, op=0, S=0, shift=0, imm12=<imm>, Rn=<Xn>, Rd=<Xd>
    fn encode_add_imm(&self, dst: u8, src: u8, imm: u32) -> Result<Vec<u8>, RasError> {
        if imm > 0xFFF {
            return Err(RasError::EncodingError(
                "ADD immediate must be 12 bits or less".to_string(),
            ));
        }
        let inst = 0b1_0_0_100010_0_000000000000_00000_00000;
        let inst = inst | (dst as u32);
        let inst = inst | ((src as u32) << 5);
        let inst = inst | ((imm & 0xFFF) << 10);
        Ok(self.encode_u32(inst))
    }

    /// Encode STR (store register) instruction: STR <Xt>, [<Xn|SP>, #<imm>]
    /// Encoding: size=11 (64-bit), opc=00, imm12=<imm>, Rn=<base>, Rt=<src>
    fn encode_str(&self, src: u8, base: u8, offset: i32) -> Result<Vec<u8>, RasError> {
        if !(0..=0xFFF).contains(&offset) {
            return Err(RasError::EncodingError(
                "STR offset must be 0-4095".to_string(),
            ));
        }
        let inst = 0b11_111_0_00_000000000000_00000_00000;
        let inst = inst | (src as u32);
        let inst = inst | ((base as u32) << 5);
        let inst = inst | ((offset as u32 & 0xFFF) << 10);
        Ok(self.encode_u32(inst))
    }

    /// Encode LDR (load register) instruction: LDR <Xt>, [<Xn|SP>, #<imm>]
    /// Encoding: size=11 (64-bit), opc=01, imm12=<imm>, Rn=<base>, Rt=<dst>
    fn encode_ldr(&self, dst: u8, base: u8, offset: i32) -> Result<Vec<u8>, RasError> {
        if !(0..=0xFFF).contains(&offset) {
            return Err(RasError::EncodingError(
                "LDR offset must be 0-4095".to_string(),
            ));
        }
        let inst = 0b11_111_0_01_000000000000_00000_00000;
        let inst = inst | (dst as u32);
        let inst = inst | ((base as u32) << 5);
        let inst = inst | ((offset as u32 & 0xFFF) << 10);
        Ok(self.encode_u32(inst))
    }

    /// Encode RET instruction: RET <Xn>
    /// RET Xn = 0xD65F0000 | (n << 5)
    fn encode_ret(&self, reg: u8) -> Vec<u8> {
        let instr: u32 = 0xD65F_0000 | ((reg as u32) << 5);
        self.encode_u32(instr)
    }

    /// Encode BR instruction: BR <Xn>
    /// BR Xn = 0xD61F0000 | (n << 5)
    fn encode_br(&self, reg: u8) -> Vec<u8> {
        let instr: u32 = 0xD61F_0000 | ((reg as u32) << 5);
        self.encode_u32(instr)
    }

    /// Encode BLR instruction: BLR <Xn>
    /// BLR Xn = 0xD63F0000 | (n << 5)
    fn encode_blr(&self, reg: u8) -> Vec<u8> {
        let instr: u32 = 0xD63F_0000 | ((reg as u32) << 5);
        self.encode_u32(instr)
    }
}

impl InstructionEncoder for AArch64Encoder {
    fn encode_instruction(
        &mut self,
        inst: &ParsedInstruction,
    ) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let mut code = Vec::new();

        match opcode.as_str() {
            "mov" | "movz" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "mov requires 2 operands".to_string(),
                    ));
                }

                let dst = &inst.operands[0];
                let src = &inst.operands[1];

                let dst_reg = self.parse_register(dst)?;

                if src.starts_with('#') {
                    // mov imm, reg
                    let imm_str = src.trim_start_matches('#');
                    let imm: u64 = imm_str
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    code.extend_from_slice(&self.encode_mov_imm64(dst_reg, imm)?);
                } else {
                    // mov reg, reg
                    let src_reg = self.parse_register(src)?;
                    code.extend_from_slice(&self.encode_mov_reg(dst_reg, src_reg));
                }
            }
            "ret" => {
                if inst.operands.is_empty() {
                    // RET (defaults to x30/LR)
                    code.extend_from_slice(&self.encode_ret(30));
                } else {
                    let reg = self.parse_register(&inst.operands[0])?;
                    code.extend_from_slice(&self.encode_ret(reg));
                }
            }
            "add" => {
                if inst.operands.len() != 3 {
                    return Err(RasError::EncodingError(
                        "add requires 3 operands".to_string(),
                    ));
                }

                let dst = &inst.operands[0];
                let src1 = &inst.operands[1];
                let src2 = &inst.operands[2];

                let dst_reg = self.parse_register(dst)?;
                let src1_reg = self.parse_register(src1)?;

                if src2.starts_with('#') {
                    // add reg, reg, #imm
                    let imm_str = src2.trim_start_matches('#');
                    let imm: u32 = imm_str
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    code.extend_from_slice(&self.encode_add_imm(dst_reg, src1_reg, imm)?);
                } else {
                    // add reg, reg, reg
                    let src2_reg = self.parse_register(src2)?;
                    code.extend_from_slice(&self.encode_add_reg(dst_reg, src1_reg, src2_reg));
                }
            }
            "str" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "str requires 2 operands".to_string(),
                    ));
                }

                let src = &inst.operands[0];
                let mem = &inst.operands[1];

                let src_reg = self.parse_register(src)?;

                // Parse memory operand: [reg, #offset] or [reg]
                if mem.starts_with('[') && mem.ends_with(']') {
                    let inner = &mem[1..mem.len() - 1];
                    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                    let base_reg = self.parse_register(parts[0])?;
                    let offset = if parts.len() > 1 {
                        let off_str = parts[1].trim_start_matches('#');
                        off_str
                            .parse::<i32>()
                            .map_err(|_| RasError::EncodingError("Invalid offset".to_string()))?
                    } else {
                        0
                    };
                    code.extend_from_slice(&self.encode_str(src_reg, base_reg, offset)?);
                } else {
                    return Err(RasError::EncodingError(
                        "str requires memory operand [reg, #offset]".to_string(),
                    ));
                }
            }
            "ldr" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "ldr requires 2 operands".to_string(),
                    ));
                }

                let dst = &inst.operands[0];
                let mem = &inst.operands[1];

                let dst_reg = self.parse_register(dst)?;

                // Parse memory operand: [reg, #offset] or [reg]
                if mem.starts_with('[') && mem.ends_with(']') {
                    let inner = &mem[1..mem.len() - 1];
                    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                    let base_reg = self.parse_register(parts[0])?;
                    let offset = if parts.len() > 1 {
                        let off_str = parts[1].trim_start_matches('#');
                        off_str
                            .parse::<i32>()
                            .map_err(|_| RasError::EncodingError("Invalid offset".to_string()))?
                    } else {
                        0
                    };
                    code.extend_from_slice(&self.encode_ldr(dst_reg, base_reg, offset)?);
                } else {
                    return Err(RasError::EncodingError(
                        "ldr requires memory operand [reg, #offset]".to_string(),
                    ));
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

