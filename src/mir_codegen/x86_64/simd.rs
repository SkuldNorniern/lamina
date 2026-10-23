use crate::mir::{Instruction, MirType, Operand, Register, VectorOp};

pub(super) use crate::mir_codegen::simd::reserve_vector_slots;

use super::{
    RaxTrackingWriter,
    util::{load_operand_to_rax, store_rax_to_register},
};
use crate::{
    error::LaminaError,
    mir::{AddressMode, VectorLane, VectorType, VirtualReg},
    mir_codegen::CodegenError,
};
use lamina_codegen::x86_64::X64RegAlloc;
use std::{collections::HashMap, io::Write};

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

fn lane(ty: MirType) -> Result<VectorLane, LaminaError> {
    match ty {
        MirType::Vector(VectorType::V128(lane)) => Ok(lane),
        MirType::Vector(VectorType::V256(_)) => Err(unsupported("x86_64 V256 requires AVX")),
        _ => Err(unsupported(format!("expected V128, got {ty:?}"))),
    }
}

fn slot(reg: &Register, slots: &HashMap<VirtualReg, i32>) -> Result<i32, LaminaError> {
    if let Register::Virtual(v) = reg
        && let Some(offset) = slots.get(v)
    {
        return Ok(*offset);
    }
    Err(unsupported(
        "vector value requires virtual register stack slot",
    ))
}

fn load_vector<W: Write>(
    operand: &Operand,
    xmm: &str,
    writer: &mut RaxTrackingWriter<W>,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    let offset = slot(reg, slots)?;
    writeln!(writer, "    movdqu {offset}(%rbp), %{xmm}")?;
    Ok(())
}

pub(super) fn memory<W: Write>(
    inst: &Instruction,
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let (ty, addr) = match inst {
        Instruction::Load { ty, addr, .. } | Instruction::Store { ty, addr, .. } => (*ty, addr),
        _ => unreachable!(),
    };
    lane(ty)?;
    let AddressMode::BaseOffset { base, offset, .. } = addr else {
        return Err(unsupported(format!("vector memory address {addr:?}")));
    };
    load_operand_to_rax(&Operand::Register(base.clone()), writer, ra, slots)?;
    match inst {
        Instruction::Load { dst, .. } => {
            let dst = slot(dst, slots)?;
            writeln!(writer, "    movdqu {offset}(%rax), %xmm0")?;
            writeln!(writer, "    movdqu %xmm0, {dst}(%rbp)")?;
        }
        Instruction::Store { src, .. } => {
            load_vector(src, "xmm0", writer, slots)?;
            writeln!(writer, "    movdqu %xmm0, {offset}(%rax)")?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn binary_mnemonic(op: VectorOp, lane: VectorLane) -> Result<&'static str, LaminaError> {
    use VectorLane::*;
    use VectorOp::*;
    let mnemonic = match (op, lane) {
        (VAdd, I8) => "paddb",
        (VAdd, I16) => "paddw",
        (VAdd, I32) => "paddd",
        (VAdd, I64) => "paddq",
        (VAdd, F32) => "addps",
        (VAdd, F64) => "addpd",
        (VSub, I8) => "psubb",
        (VSub, I16) => "psubw",
        (VSub, I32) => "psubd",
        (VSub, I64) => "psubq",
        (VSub, F32) => "subps",
        (VSub, F64) => "subpd",
        (VMul, I16) => "pmullw",
        (VMul, I32) => "pmulld",
        (VMul, F32) => "mulps",
        (VMul, F64) => "mulpd",
        (VAnd, F32) => "andps",
        (VAnd, F64) => "andpd",
        (VAnd, _) => "pand",
        (VOr, F32) => "orps",
        (VOr, F64) => "orpd",
        (VOr, _) => "por",
        (VXor, F32) => "xorps",
        (VXor, F64) => "xorpd",
        (VXor, _) => "pxor",
        (VShl, I16) => "psllw",
        (VShl, I32) => "pslld",
        (VShl, I64) => "psllq",
        (VLShr, I16) => "psrlw",
        (VLShr, I32) => "psrld",
        (VLShr, I64) => "psrlq",
        (VAShr, I16) => "psraw",
        (VAShr, I32) => "psrad",
        _ => {
            return Err(unsupported(format!(
                "x86_64 unsupported {op:?} for {lane:?}"
            )));
        }
    };
    Ok(mnemonic)
}

pub(super) fn operation<W: Write>(
    op: VectorOp,
    ty: MirType,
    dst: &Register,
    operands: &[Operand],
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let lane = lane(ty)?;
    if matches!(op, VectorOp::VExtractLane | VectorOp::VInsertLane) {
        return lane_access(op, lane, dst, operands, writer, ra, slots);
    }
    if op == VectorOp::VSplat {
        if operands.len() != 1 {
            return Err(unsupported("VSplat requires one operand"));
        }
        load_scalar(&operands[0], writer, ra, slots)?;
        match lane {
            VectorLane::I64 | VectorLane::F64 => {
                writeln!(writer, "    movq %rax, %xmm0")?;
                let mnemonic = if lane == VectorLane::I64 {
                    "punpcklqdq"
                } else {
                    "unpcklpd"
                };
                writeln!(writer, "    {mnemonic} %xmm0, %xmm0")?;
            }
            _ => {
                writeln!(writer, "    movd %eax, %xmm0")?;
                if lane == VectorLane::I8 {
                    writeln!(writer, "    punpcklbw %xmm0, %xmm0")?;
                }
                if matches!(lane, VectorLane::I8 | VectorLane::I16) {
                    writeln!(writer, "    punpcklwd %xmm0, %xmm0")?;
                }
                let shuffle = if lane == VectorLane::F32 {
                    "shufps $0, %xmm0"
                } else {
                    "pshufd $0, %xmm0"
                };
                writeln!(writer, "    {shuffle}, %xmm0")?;
            }
        }
    } else {
        if operands.len() != 2 {
            return Err(unsupported(format!("{op:?} requires two operands")));
        }
        let mnemonic = binary_mnemonic(op, lane)?;
        load_vector(&operands[0], "xmm0", writer, slots)?;
        load_vector(&operands[1], "xmm1", writer, slots)?;
        writeln!(writer, "    {mnemonic} %xmm1, %xmm0")?;
    }
    let offset = slot(dst, slots)?;
    writeln!(writer, "    movdqu %xmm0, {offset}(%rbp)")?;
    Ok(())
}

fn load_scalar<W: Write>(
    operand: &Operand,
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    use crate::mir::Immediate;
    let bits = match operand {
        Operand::Immediate(Immediate::F32(v)) => Some(i64::from(v.to_bits())),
        Operand::Immediate(Immediate::F64(v)) => Some(v.to_bits() as i64),
        _ => None,
    };
    if let Some(bits) = bits {
        load_operand_to_rax(&Operand::imm_i64(bits), writer, ra, slots)?;
    } else {
        load_operand_to_rax(operand, writer, ra, slots)?;
    }
    Ok(())
}

fn lane_access<W: Write>(
    op: VectorOp,
    lane: VectorLane,
    dst: &Register,
    operands: &[Operand],
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    use crate::mir::Immediate;
    let extract = op == VectorOp::VExtractLane;
    let count = if extract { 2 } else { 3 };
    if operands.len() != count {
        return Err(unsupported(format!("{op:?} requires {count} operands")));
    }
    let index = match &operands[count - 1] {
        Operand::Immediate(Immediate::I8(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I16(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I32(v)) => i64::from(*v),
        Operand::Immediate(Immediate::I64(v)) => *v,
        _ => {
            return Err(unsupported(format!(
                "{op:?} {lane:?} lane index must be integer immediate"
            )));
        }
    };
    let (suffix, lanes, gpr) = match lane {
        VectorLane::I8 => ("b", 16, "eax"),
        VectorLane::I16 => ("w", 8, "eax"),
        VectorLane::I32 | VectorLane::F32 => ("d", 4, "eax"),
        VectorLane::I64 | VectorLane::F64 => ("q", 2, "rax"),
    };
    if !(0..lanes).contains(&index) {
        return Err(unsupported(format!(
            "{op:?} {lane:?} lane index {index} out of range"
        )));
    }
    load_vector(&operands[0], "xmm0", writer, slots)?;
    if extract {
        writeln!(writer, "    pextr{suffix} ${index}, %xmm0, %{gpr}")?;
        match dst {
            Register::Virtual(v) => store_rax_to_register(v, writer, ra, slots)?,
            Register::Physical(p) => writeln!(writer, "    movq %rax, %{}", p.name)?,
        }
    } else {
        load_scalar(&operands[1], writer, ra, slots)?;
        writeln!(writer, "    pinsr{suffix} ${index}, %{gpr}, %xmm0")?;
        let offset = slot(dst, slots)?;
        writeln!(writer, "    movdqu %xmm0, {offset}(%rbp)")?;
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[allow(clippy::expect_used)]
mod tests {

    use crate::mir_codegen::simd_support::{run, tools_available};

    use crate::{
        mir::{
            AddressMode, AddressSpace, Block, Function, Instruction, IntBinOp, MemoryAttrs,
            MirType, Module, Operand, Parameter, Register, ScalarType, Signature, VectorLane,
            VectorOp, VectorType, VirtualReg,
        },
        mir_codegen::{MirCodegenSettings, RegallocStrategy, generate_mir_to_target_with_settings},
    };
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
    use std::{array, env, fmt::Write as _, fs, path::Path, process, process::Command};

    const LANES: [VectorLane; 6] = [
        VectorLane::I8,
        VectorLane::I16,
        VectorLane::I32,
        VectorLane::I64,
        VectorLane::F32,
        VectorLane::F64,
    ];
    const BINARY: [VectorOp; 9] = [
        VectorOp::VAdd,
        VectorOp::VSub,
        VectorOp::VMul,
        VectorOp::VAnd,
        VectorOp::VOr,
        VectorOp::VXor,
        VectorOp::VShl,
        VectorOp::VLShr,
        VectorOp::VAShr,
    ];

    fn scalar(lane: VectorLane) -> MirType {
        MirType::Scalar(match lane {
            VectorLane::I8 => ScalarType::I8,
            VectorLane::I16 => ScalarType::I16,
            VectorLane::I32 => ScalarType::I32,
            VectorLane::I64 => ScalarType::I64,
            VectorLane::F32 => ScalarType::F32,
            VectorLane::F64 => ScalarType::F64,
        })
    }

    fn supported(op: VectorOp, lane: VectorLane) -> bool {
        use VectorLane::*;
        use VectorOp::*;
        match op {
            VMul => !matches!(lane, I8 | I64),
            VShl | VLShr => matches!(lane, I16 | I32 | I64),
            VAShr => matches!(lane, I16 | I32),
            _ => true,
        }
    }

    fn reg(id: u32) -> Register {
        Register::Virtual(VirtualReg::gpr(id))
    }
    fn vector(id: u32) -> Register {
        Register::Virtual(VirtualReg::vec(id))
    }
    fn addr(id: u32) -> AddressMode {
        AddressMode::BaseOffset {
            address_space: AddressSpace::Generic,
            base: reg(id),
            offset: 1,
        }
    }

    fn function(name: &str, ty: MirType, op: VectorOp, index: i64) -> Function {
        let MirType::Vector(v) = ty else {
            panic!("vector type required")
        };
        let (VectorType::V128(lane) | VectorType::V256(lane)) = v;
        let st = scalar(lane);
        let mut f = Function::new(
            Signature::new(name).with_params(
                (0..3)
                    .map(|i| Parameter::new(reg(i), MirType::ptr()))
                    .collect(),
            ),
        );
        let vector = |id| {
            if name.ends_with("_gpr") {
                reg(id)
            } else {
                vector(id)
            }
        };
        let mut b = Block::new("entry");
        b.push(Instruction::Load {
            ty,
            dst: vector(3),
            addr: addr(0),
            attrs: MemoryAttrs::default(),
        });
        let single = matches!(op, VectorOp::VSplat | VectorOp::VInsertLane);
        if single {
            b.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::ptr(),
                dst: reg(6),
                lhs: Operand::reg(reg(1)),
                rhs: Operand::imm_i64(1),
            });
        }
        b.push(Instruction::Load {
            ty: if single { st } else { ty },
            dst: if single { reg(4) } else { vector(4) },
            addr: if single {
                AddressMode::BaseOffset {
                    address_space: AddressSpace::Generic,
                    base: reg(6),
                    offset: 0,
                }
            } else {
                addr(1)
            },
            attrs: MemoryAttrs::default(),
        });
        let operands = match op {
            VectorOp::VSplat => vec![Operand::reg(reg(4))],
            VectorOp::VExtractLane => vec![Operand::reg(vector(3)), Operand::imm_i64(index)],
            VectorOp::VInsertLane => vec![
                Operand::reg(vector(3)),
                Operand::reg(reg(4)),
                Operand::imm_i64(index),
            ],
            _ => vec![Operand::reg(vector(3)), Operand::reg(vector(4))],
        };
        let dst = if op == VectorOp::VExtractLane {
            reg(5)
        } else {
            vector(5)
        };
        b.push(Instruction::VectorOp {
            op,
            ty,
            dst: dst.clone(),
            operands,
        });
        if op == VectorOp::VExtractLane {
            b.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::ptr(),
                dst: reg(7),
                lhs: Operand::reg(reg(2)),
                rhs: Operand::imm_i64(1),
            });
        }
        b.push(Instruction::Store {
            ty: if op == VectorOp::VExtractLane { st } else { ty },
            src: Operand::reg(dst),
            addr: if op == VectorOp::VExtractLane {
                AddressMode::BaseOffset {
                    address_space: AddressSpace::Generic,
                    base: reg(7),
                    offset: 0,
                }
            } else {
                addr(2)
            },
            attrs: MemoryAttrs::default(),
        });
        b.push(Instruction::Ret { value: None });
        f.add_block(b);
        f
    }

    fn emit(module: &Module, strategy: RegallocStrategy) -> Result<Vec<u8>, String> {
        let mut asm = Vec::new();
        generate_mir_to_target_with_settings(
            module,
            &mut asm,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings {
                regalloc: strategy,
                ..Default::default()
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        Ok(asm)
    }

    fn bits(bytes: &[u8]) -> u64 {
        let mut buf = [0; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(buf)
    }

    fn expected(
        op: VectorOp,
        lane: VectorLane,
        a: &[u8; 16],
        b: &[u8; 16],
        index: usize,
    ) -> Vec<u8> {
        use VectorOp::*;
        let width = scalar(lane).size_bytes();
        match op {
            VSplat => return b[..width].repeat(16 / width),
            VExtractLane => return a[index * width..(index + 1) * width].to_vec(),
            VInsertLane => {
                let mut out = *a;
                out[index * width..(index + 1) * width].copy_from_slice(&b[..width]);
                return out.to_vec();
            }
            _ => (),
        }
        let shift = bits(&b[..8]);
        a.chunks_exact(width)
            .zip(b.chunks_exact(width))
            .flat_map(|(a, b)| {
                let x = bits(a);
                let y = bits(b);
                let n = width * 8;
                let value = match op {
                    VAnd => x & y,
                    VOr => x | y,
                    VXor => x ^ y,
                    VShl => {
                        if shift >= n as u64 {
                            0
                        } else {
                            x << shift
                        }
                    }
                    VLShr => {
                        if shift >= n as u64 {
                            0
                        } else {
                            x >> shift
                        }
                    }
                    VAShr => {
                        let signed = ((x << (64 - n)) as i64) >> (64 - n);
                        (signed >> shift.min(n as u64 - 1)) as u64
                    }
                    _ if lane == VectorLane::F32 => {
                        let x = f32::from_bits(x as u32);
                        let y = f32::from_bits(y as u32);
                        u64::from(
                            match op {
                                VAdd => x + y,
                                VSub => x - y,
                                VMul => x * y,
                                _ => unreachable!(),
                            }
                            .to_bits(),
                        )
                    }
                    _ if lane == VectorLane::F64 => {
                        let x = f64::from_bits(x);
                        let y = f64::from_bits(y);
                        match op {
                            VAdd => x + y,
                            VSub => x - y,
                            VMul => x * y,
                            _ => unreachable!(),
                        }
                        .to_bits()
                    }
                    VAdd => x.wrapping_add(y),
                    VSub => x.wrapping_sub(y),
                    VMul => x.wrapping_mul(y),
                    _ => unreachable!(),
                };
                value.to_le_bytes()[..width].to_vec()
            })
            .collect()
    }

    struct Case {
        op: VectorOp,
        lane: VectorLane,
        index: usize,
        a: [u8; 16],
        b: [u8; 16],
    }

    fn execute(cases: &[Case], strategy: RegallocStrategy, dir: &Path, immediate: bool) {
        let mut module = Module::new("simd");
        let mut harness = String::from("#include <stdio.h>\n#include <string.h>\n");
        for i in 0..cases.len() {
            writeln!(harness, "extern void simd_{i}(void*, void*, void*);").expect("format");
        }
        harness.push_str("int main(void) {\n");
        let mut want = Vec::new();
        for (i, c) in cases.iter().enumerate() {
            module.add_function(function(
                &format!("simd_{i}"),
                MirType::Vector(VectorType::V128(c.lane)),
                c.op,
                c.index as i64,
            ));
            let expected = expected(c.op, c.lane, &c.a, &c.b, c.index);
            let a = c.a.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            let b = c.b.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            writeln!(harness, "{{ unsigned char a[18]={{0,{a},0}}, b[18]={{0,{b},0}}, out[18]; memset(out, 165, 18); simd_{i}(a,b,out); fwrite(out,1,18,stdout); }}").expect("format");
            let mut guarded = vec![165; 18];
            guarded[1..1 + expected.len()].copy_from_slice(&expected);
            want.extend(guarded);
        }
        harness.push_str("return 0; }\n");
        let gpr_functions: Vec<_> = module
            .functions
            .keys()
            .filter(|name| {
                name.strip_prefix("simd_")
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|i| i % 2 == 1)
            })
            .cloned()
            .collect();
        for name in gpr_functions {
            let i = name
                .strip_prefix("simd_")
                .expect("prefix")
                .parse::<usize>()
                .expect("index");
            let c = &cases[i];
            let mut f = function(
                &format!("{name}_gpr"),
                MirType::Vector(VectorType::V128(c.lane)),
                c.op,
                c.index as i64,
            );
            f.sig.name = name.clone();
            module.functions.insert(name, f);
        }
        if immediate {
            for (i, c) in cases.iter().enumerate() {
                let f = module
                    .functions
                    .get_mut(&format!("simd_{i}"))
                    .expect("function");
                let value = match c.lane {
                    VectorLane::F32 => Operand::imm_f32(f32::from_bits(bits(&c.b[..4]) as u32)),
                    VectorLane::F64 => Operand::imm_f64(f64::from_bits(bits(&c.b[..8]))),
                    VectorLane::I8 => Operand::imm_i8(bits(&c.b[..1]) as i8),
                    VectorLane::I16 => {
                        Operand::Immediate(crate::mir::Immediate::I16(bits(&c.b[..2]) as i16))
                    }
                    VectorLane::I32 => Operand::imm_i32(bits(&c.b[..4]) as i32),
                    _ => Operand::imm_i64(bits(&c.b[..scalar(c.lane).size_bytes()]) as i64),
                };
                for inst in &mut f.blocks[0].instructions {
                    if let Instruction::VectorOp { op, operands, .. } = inst {
                        match op {
                            VectorOp::VSplat => operands[0] = value.clone(),
                            VectorOp::VInsertLane => operands[1] = value.clone(),
                            _ => (),
                        }
                    }
                }
            }
        }
        let mut asm = emit(&module, strategy).expect("emit SIMD assembly");
        for i in 0..cases.len() {
            asm.extend_from_slice(format!("\n.globl simd_{i}\n").as_bytes());
        }
        asm.extend_from_slice(b".section .note.GNU-stack,\"\",@progbits\n");
        fs::write(dir.join("simd.s"), asm).expect("write assembly");
        fs::write(dir.join("harness.c"), harness).expect("write harness");
        run(Command::new("as")
            .arg("--64")
            .arg(dir.join("simd.s"))
            .arg("-o")
            .arg(dir.join("simd.o")));
        run(Command::new("gcc")
            .arg(dir.join("harness.c"))
            .arg(dir.join("simd.o"))
            .arg("-o")
            .arg(dir.join("simd")));
        let actual = Command::new(dir.join("simd"))
            .output()
            .expect("execute SIMD");
        assert!(actual.status.success());
        assert_eq!(actual.stdout.len(), want.len());
        for (i, (got, want)) in actual
            .stdout
            .chunks_exact(18)
            .zip(want.chunks_exact(18))
            .enumerate()
        {
            let c = &cases[i];
            assert_eq!(
                got, want,
                "{strategy:?} {:?} {:?} lane {} case {i}",
                c.op, c.lane, c.index
            );
        }
    }

    #[test]
    fn simd_executes_every_supported_pair() {
        if !tools_available(&["as", "gcc"]) {
            return;
        }
        let mut cases = Vec::new();
        for lane in LANES {
            let width = scalar(lane).size_bytes();
            for pattern in 0..3 {
                let mut a =
                    array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(129 + pattern));
                let mut b =
                    array::from_fn(|i| (i as u8).wrapping_mul(19).wrapping_add(91 - pattern));
                if lane == VectorLane::F32 {
                    for i in 0..4 {
                        a[i * 4..i * 4 + 4].copy_from_slice(
                            &(-2.5_f32 + i as f32 + f32::from(pattern)).to_le_bytes(),
                        );
                        b[i * 4..i * 4 + 4].copy_from_slice(&(0.5_f32 + i as f32).to_le_bytes());
                    }
                } else if lane == VectorLane::F64 {
                    for i in 0..2 {
                        a[i * 8..i * 8 + 8].copy_from_slice(
                            &(-2.5_f64 + i as f64 + f64::from(pattern)).to_le_bytes(),
                        );
                        b[i * 8..i * 8 + 8].copy_from_slice(&(0.5_f64 + i as f64).to_le_bytes());
                    }
                }
                for op in BINARY {
                    if !supported(op, lane) {
                        continue;
                    }
                    if matches!(op, VectorOp::VShl | VectorOp::VLShr | VectorOp::VAShr) {
                        for count in [
                            0,
                            3,
                            width as u64 * 8 - 1,
                            width as u64 * 8,
                            width as u64 * 8 + 1,
                            64,
                            u64::MAX,
                        ] {
                            let mut counts = b;
                            counts[..8].copy_from_slice(&count.to_le_bytes());
                            cases.push(Case {
                                op,
                                lane,
                                index: 0,
                                a,
                                b: counts,
                            });
                        }
                    } else {
                        cases.push(Case {
                            op,
                            lane,
                            index: 0,
                            a,
                            b,
                        });
                    }
                }
                cases.push(Case {
                    op: VectorOp::VSplat,
                    lane,
                    index: 0,
                    a,
                    b,
                });
                for index in [0, 16 / width - 1] {
                    for op in [VectorOp::VExtractLane, VectorOp::VInsertLane] {
                        cases.push(Case {
                            op,
                            lane,
                            index,
                            a,
                            b,
                        });
                    }
                }
            }
        }
        let dir = env::temp_dir().join(format!("lamina-simd-{}", process::id()));
        fs::create_dir_all(&dir).expect("create temp directory");
        for strategy in [
            RegallocStrategy::Incremental,
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            for immediate in [false, true] {
                execute(&cases, strategy, &dir, immediate);
            }
        }
        fs::remove_dir_all(dir).expect("remove temp directory");
    }

    #[test]
    fn unsupported_simd_returns_errors() {
        for lane in LANES {
            for op in BINARY {
                if supported(op, lane) {
                    continue;
                }
                let mut m = Module::new("negative");
                m.add_function(function(
                    "bad",
                    MirType::Vector(VectorType::V128(lane)),
                    op,
                    0,
                ));
                let error =
                    emit(&m, RegallocStrategy::Incremental).expect_err("unsupported combination");
                assert!(
                    error.contains("UnsupportedFeature")
                        && error.contains(&format!("{op:?}"))
                        && error.contains(&format!("{lane:?}")),
                    "{error}"
                );
            }
            for op in BINARY.into_iter().chain([
                VectorOp::VSplat,
                VectorOp::VExtractLane,
                VectorOp::VInsertLane,
            ]) {
                let mut m = Module::new("negative");
                let mut f = function("bad", MirType::Vector(VectorType::V256(lane)), op, 0);
                f.blocks[0].instructions.retain(|i| {
                    matches!(i, Instruction::VectorOp { .. } | Instruction::Ret { .. })
                });
                m.add_function(f);
                assert!(
                    emit(&m, RegallocStrategy::Incremental)
                        .expect_err("V256 unsupported")
                        .contains("V256")
                );
            }
            for store in [false, true] {
                let mut m = Module::new("negative");
                let mut f = function(
                    "bad",
                    MirType::Vector(VectorType::V256(lane)),
                    VectorOp::VAdd,
                    0,
                );
                f.blocks[0].instructions.retain(|i| {
                    if store {
                        matches!(i, Instruction::Store { .. } | Instruction::Ret { .. })
                    } else {
                        matches!(i, Instruction::Load { .. } | Instruction::Ret { .. })
                    }
                });
                m.add_function(f);
                assert!(
                    emit(&m, RegallocStrategy::Incremental)
                        .expect_err("V256 memory unsupported")
                        .contains("V256")
                );
            }
            for op in [VectorOp::VExtractLane, VectorOp::VInsertLane] {
                for index in [
                    Operand::reg(reg(0)),
                    Operand::imm_i64(-1),
                    Operand::imm_i64((16 / scalar(lane).size_bytes()) as i64),
                ] {
                    let mut m = Module::new("negative");
                    let mut f = function("bad", MirType::Vector(VectorType::V128(lane)), op, 0);
                    for i in &mut f.blocks[0].instructions {
                        if let Instruction::VectorOp { operands, .. } = i {
                            *operands.last_mut().expect("index") = index.clone();
                        }
                    }
                    m.add_function(f);
                    assert!(
                        emit(&m, RegallocStrategy::Incremental)
                            .expect_err("invalid index")
                            .contains("lane index")
                    );
                }
            }
        }
    }
}
