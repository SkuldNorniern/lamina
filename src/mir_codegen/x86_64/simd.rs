use super::instruction::{indexed_memory, release_indexed_scratch};
use crate::mir::{AddressMode, Instruction, MirType, Operand, Register, VectorOp};

pub(super) use crate::mir_codegen::simd::reserve_vector_slots;

use super::{
    RaxTrackingWriter,
    util::{load_operand_to_rax, store_rax_to_register},
};
use crate::{
    error::LaminaError,
    mir::{Function, RegisterClass, VectorLane, VectorType, VirtualReg},
    mir_codegen::CodegenError,
};
use lamina_codegen::x86_64::X64RegAlloc;
use lamina_platform::X86CpuLevel;
use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

/// Emitter scratch vectors. Global allocation never hands these out, so any vector
/// operation may clobber them freely.
const SCRATCH_A: &str = "xmm0";
const SCRATCH_B: &str = "xmm1";
const SCRATCH_SRC0: &str = "xmm2";
const SCRATCH_SRC1: &str = "xmm3";

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

fn lane(ty: MirType) -> Result<VectorLane, LaminaError> {
    match ty {
        MirType::Vector(VectorType::V128(lane)) => Ok(lane),
        MirType::Vector(VectorType::V256(_)) => Err(unsupported(
            "x86_64 V256 is not implemented: the backend emits only 128-bit SSE forms, at every CPU level",
        )),
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

fn xmm_of(reg: &Register, ra: &X64RegAlloc) -> Option<&'static str> {
    match reg {
        Register::Virtual(v) => ra.xmm_for(v),
        Register::Physical(_) => None,
    }
}

/// Returns the vector register holding `operand`, loading it into `scratch` first when
/// it lives in a stack slot.
fn source<W: Write>(
    operand: &Operand,
    scratch: &'static str,
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<&'static str, LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    if let Some(xmm) = xmm_of(reg, ra) {
        return Ok(xmm);
    }
    let offset = slot(reg, slots)?;
    writeln!(writer, "    movdqu {offset}(%rbp), %{scratch}")?;
    Ok(scratch)
}

/// Writes the vector held in `from` to the destination register.
fn sink<W: Write>(
    dst: &Register,
    from: &'static str,
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    if let Some(xmm) = xmm_of(dst, ra) {
        if xmm != from {
            writeln!(writer, "    movdqa %{from}, %{xmm}")?;
        }
    } else {
        let offset = slot(dst, slots)?;
        writeln!(writer, "    movdqu %{from}, {offset}(%rbp)")?;
    }
    Ok(())
}

pub(super) fn register_eligible_vectors(func: &Function) -> HashSet<VirtualReg> {
    let mut eligible = HashSet::new();
    let mut rejected: HashSet<VirtualReg> = func
        .sig
        .params
        .iter()
        .filter_map(|p| match p.reg {
            Register::Virtual(v) => Some(v),
            _ => None,
        })
        .collect();
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        let mut values: Vec<&Register> = Vec::new();
        match &inst.instr {
            Instruction::Load { ty, dst, .. } if ty.is_vector() => values.push(dst),
            Instruction::Store {
                ty,
                src: Operand::Register(src),
                ..
            } if ty.is_vector() => values.push(src),
            Instruction::VectorOp {
                op,
                ty,
                dst,
                operands,
                ..
            } if ty.is_vector() => match op {
                VectorOp::VExtractLane => {
                    values.extend(operands.first().and_then(|o| match o {
                        Operand::Register(r) => Some(r),
                        _ => None,
                    }));
                }
                VectorOp::VSplat => values.push(dst),
                VectorOp::VInsertLane => {
                    values.push(dst);
                    values.extend(operands.first().and_then(|o| match o {
                        Operand::Register(r) => Some(r),
                        _ => None,
                    }));
                }
                _ => {
                    values.push(dst);
                    values.extend(operands.iter().take(2).filter_map(|o| match o {
                        Operand::Register(r) => Some(r),
                        _ => None,
                    }));
                }
            },
            _ => {}
        }
        let value_regs: HashSet<VirtualReg> = values
            .iter()
            .filter_map(|r| match r {
                Register::Virtual(v) => Some(*v),
                _ => None,
            })
            .collect();
        for reg in inst.use_regs().into_iter().chain(inst.def_reg()) {
            if let Register::Virtual(v) = reg
                && v.class == RegisterClass::Vec
            {
                if value_regs.contains(v) {
                    eligible.insert(*v);
                } else {
                    rejected.insert(*v);
                }
            }
        }
    }
    eligible.retain(|v| !rejected.contains(v));
    eligible
}

pub(super) fn memory<W: Write>(
    inst: &Instruction,
    writer: &mut RaxTrackingWriter<W>,
    ra: &mut X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let (ty, addr) = match inst {
        Instruction::Load { ty, addr, .. } | Instruction::Store { ty, addr, .. } => (*ty, addr),
        _ => unreachable!(),
    };
    lane(ty)?;
    let (mem, scratches) = match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            load_operand_to_rax(&Operand::Register(base.clone()), writer, &*ra, slots)?;
            (format!("{offset}(%rax)"), Vec::new())
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => indexed_memory(base, index, *scale, *offset, writer, ra, slots)?,
    };
    match inst {
        Instruction::Load { dst, .. } => {
            if let Some(xmm) = xmm_of(dst, &*ra) {
                writeln!(writer, "    movdqu {mem}, %{xmm}")?;
            } else {
                writeln!(writer, "    movdqu {mem}, %{SCRATCH_A}")?;
                sink(dst, SCRATCH_A, writer, &*ra, slots)?;
            }
        }
        Instruction::Store { src, .. } => {
            let value = source(src, SCRATCH_SRC0, writer, &*ra, slots)?;
            writeln!(writer, "    movdqu %{value}, {mem}")?;
        }
        _ => unreachable!(),
    }
    release_indexed_scratch(scratches, ra);
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
        (VShlLanes, I32) => "vpsllvd",
        (VShlLanes, I64) => "vpsllvq",
        (VLShrLanes, I32) => "vpsrlvd",
        (VLShrLanes, I64) => "vpsrlvq",
        (VAShrLanes, I32) => "vpsravd",
        _ => {
            return Err(unsupported(format!(
                "x86_64 unsupported {op:?} for {lane:?}"
            )));
        }
    };
    Ok(mnemonic)
}

/// 64-bit lane multiply from 32-bit products, since SSE2 has no packed qword multiply:
/// `a*b = aL*bL + ((aH*bL + aL*bH) << 32)` modulo 2^64. Leaves the result in `xmm1`.
fn emit_mul_i64<W: Write>(
    a: &str,
    b: &str,
    writer: &mut RaxTrackingWriter<W>,
) -> Result<(), LaminaError> {
    writeln!(writer, "    movdqa %{a}, %xmm0")?;
    writeln!(writer, "    psrlq $32, %xmm0")?;
    writeln!(writer, "    pmuludq %{b}, %xmm0")?;
    writeln!(writer, "    movdqa %{b}, %xmm1")?;
    writeln!(writer, "    psrlq $32, %xmm1")?;
    writeln!(writer, "    pmuludq %{a}, %xmm1")?;
    writeln!(writer, "    paddq %xmm1, %xmm0")?;
    writeln!(writer, "    psllq $32, %xmm0")?;
    writeln!(writer, "    movdqa %{a}, %xmm1")?;
    writeln!(writer, "    pmuludq %{b}, %xmm1")?;
    writeln!(writer, "    paddq %xmm0, %xmm1")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn operation<W: Write>(
    op: VectorOp,
    ty: MirType,
    dst: &Register,
    operands: &[Operand],
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
    cpu_level: X86CpuLevel,
) -> Result<(), LaminaError> {
    let lane = lane(ty)?;
    if matches!(op, VectorOp::VExtractLane | VectorOp::VInsertLane) {
        return lane_access(op, lane, dst, operands, writer, ra, slots, cpu_level);
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
        return sink(dst, SCRATCH_A, writer, ra, slots);
    }
    if operands.len() != 2 {
        return Err(unsupported(format!("{op:?} requires two operands")));
    }
    let a = source(&operands[0], SCRATCH_SRC0, writer, ra, slots)?;
    if matches!(op, VectorOp::VShl | VectorOp::VLShr | VectorOp::VAShr) {
        let mnemonic = binary_mnemonic(op, lane)?;
        let work = xmm_of(dst, ra).unwrap_or(SCRATCH_A);
        if a != work {
            writeln!(writer, "    movdqa %{a}, %{work}")?;
        }
        load_scalar(&operands[1], writer, ra, slots)?;
        writeln!(writer, "    movq %rax, %{SCRATCH_SRC1}")?;
        writeln!(writer, "    {mnemonic} %{SCRATCH_SRC1}, %{work}")?;
        return sink(dst, work, writer, ra, slots);
    }
    let b = source(&operands[1], SCRATCH_SRC1, writer, ra, slots)?;
    if matches!(
        op,
        VectorOp::VShlLanes | VectorOp::VLShrLanes | VectorOp::VAShrLanes
    ) {
        let mnemonic = binary_mnemonic(op, lane)?;
        require_avx2(op, lane, cpu_level)?;
        let work = xmm_of(dst, ra).unwrap_or(SCRATCH_A);
        writeln!(writer, "    {mnemonic} %{b}, %{a}, %{work}")?;
        return sink(dst, work, writer, ra, slots);
    }
    if op == VectorOp::VMul && lane == VectorLane::I64 {
        if cpu_level >= X86CpuLevel::V4 {
            writeln!(writer, "    vpmullq %{b}, %{a}, %xmm1")?;
        } else {
            emit_mul_i64(a, b, writer)?;
        }
        return sink(dst, SCRATCH_B, writer, ra, slots);
    }
    let mnemonic = binary_mnemonic(op, lane)?;
    if mnemonic == "pmulld" {
        require_sse41(mnemonic, cpu_level)?;
    }
    let work = match xmm_of(dst, ra) {
        Some(xmm) if xmm != b => xmm,
        _ => SCRATCH_A,
    };
    if a != work {
        writeln!(writer, "    movdqa %{a}, %{work}")?;
    }
    writeln!(writer, "    {mnemonic} %{b}, %{work}")?;
    sink(dst, work, writer, ra, slots)
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

/// Per-lane counts are AVX2; SSE2 shifts every lane by one count and nothing else.
fn require_avx2(op: VectorOp, lane: VectorLane, cpu_level: X86CpuLevel) -> Result<(), LaminaError> {
    if cpu_level >= X86CpuLevel::V3 {
        return Ok(());
    }
    Err(unsupported(format!(
        "x86_64 {op:?} for {lane:?} is a per-lane vector shift, which x86_64 has only with AVX2 (x86-64-v3); {cpu_level} has one count for all lanes and nothing else\n  Hint: splat the count and write the scalar form, 'shl.<N x T> %a, %n' with a scalar %n, or build for x86-64-v3"
    )))
}

fn require_sse41(
    instruction: impl Into<String>,
    cpu_level: X86CpuLevel,
) -> Result<(), LaminaError> {
    let required = X86CpuLevel::V2;
    if cpu_level >= required {
        return Ok(());
    }
    Err(LaminaError::CodegenError(CodegenError::CpuLevelRequired {
        instruction: instruction.into(),
        required,
        selected: cpu_level,
    }))
}

#[allow(clippy::too_many_arguments)]
fn lane_access<W: Write>(
    op: VectorOp,
    lane: VectorLane,
    dst: &Register,
    operands: &[Operand],
    writer: &mut RaxTrackingWriter<W>,
    ra: &X64RegAlloc,
    slots: &HashMap<VirtualReg, i32>,
    cpu_level: X86CpuLevel,
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
    if suffix != "w" {
        let direction = if extract { "pextr" } else { "pinsr" };
        require_sse41(format!("{direction}{suffix}"), cpu_level)?;
    }
    if extract {
        let src = source(&operands[0], SCRATCH_SRC0, writer, ra, slots)?;
        writeln!(writer, "    pextr{suffix} ${index}, %{src}, %{gpr}")?;
        match dst {
            Register::Virtual(v) => store_rax_to_register(v, writer, ra, slots)?,
            Register::Physical(p) => writeln!(writer, "    movq %rax, %{}", p.name)?,
        }
    } else {
        let src = source(&operands[0], SCRATCH_A, writer, ra, slots)?;
        if src != SCRATCH_A {
            writeln!(writer, "    movdqa %{src}, %{SCRATCH_A}")?;
        }
        load_scalar(&operands[1], writer, ra, slots)?;
        writeln!(writer, "    pinsr{suffix} ${index}, %{gpr}, %{SCRATCH_A}")?;
        sink(dst, SCRATCH_A, writer, ra, slots)?;
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
#[allow(clippy::expect_used)]
mod tests {
    use crate::test_support::TempDir;

    use crate::mir_codegen::simd_support::{
        BINARY, Case, LANES, bits, declare_uses_as_params, expected, function, per_lane_shift, reg,
        run, scalar, tools_available, uniform_shift,
    };

    use crate::{
        mir::{Immediate, Instruction, MirType, Module, Operand, VectorLane, VectorOp, VectorType},
        mir_codegen::{MirCodegenSettings, RegallocStrategy, generate_mir_to_target_with_settings},
    };
    use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem, X86CpuLevel};
    use std::{array, fmt::Write as _, fs, path::Path, process::Command};

    fn supported(op: VectorOp, lane: VectorLane) -> bool {
        use VectorLane::*;
        use VectorOp::*;
        match op {
            VMul => !matches!(lane, I8),
            VShl | VLShr => matches!(lane, I16 | I32 | I64),
            VAShr => matches!(lane, I16 | I32),
            VShlLanes | VLShrLanes => matches!(lane, I32 | I64),
            VAShrLanes => matches!(lane, I32),
            _ => true,
        }
    }

    fn emit(module: &Module, strategy: RegallocStrategy) -> Result<Vec<u8>, String> {
        emit_at(module, strategy, X86CpuLevel::default())
    }

    fn emit_at(
        module: &Module,
        strategy: RegallocStrategy,
        level: X86CpuLevel,
    ) -> Result<Vec<u8>, String> {
        let mut asm = Vec::new();
        generate_mir_to_target_with_settings(
            module,
            &mut asm,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings {
                regalloc: strategy,
                x86_cpu_level: level,
                ..Default::default()
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        Ok(asm)
    }

    fn execute(
        cases: &[Case],
        strategy: RegallocStrategy,
        dir: &Path,
        immediate: bool,
        pointer_width: PointerWidth,
        level: X86CpuLevel,
    ) {
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
            let expected = expected(c.op, c.lane, &c.a, &c.b, c.index, pointer_width);
            let a = c.a.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            let b = c.b.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            writeln!(harness, "{{ unsigned char a[18]={{0,{a},0}}, b[18]={{0,{b},0}}, out[18]; memset(out, 165, 18); simd_{i}(a,b,out); fwrite(out,1,18,stdout); }}").expect("format");
            let mut guarded = vec![165; 18];
            guarded[1..1 + expected.len()].copy_from_slice(&expected);
            want.extend(guarded);
        }
        harness.push_str("return 0; }\n");
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
                    VectorLane::I16 => Operand::Immediate(Immediate::I16(bits(&c.b[..2]) as i16)),
                    VectorLane::I32 => Operand::imm_i32(bits(&c.b[..4]) as i32),
                    _ => Operand::imm_i64(
                        bits(&c.b[..scalar(c.lane).size_bytes(pointer_width)]) as i64
                    ),
                };
                for inst in &mut f.blocks[0].instructions {
                    if let Instruction::VectorOp { op, operands, .. } = &mut inst.instr {
                        match op {
                            VectorOp::VSplat => operands[0] = value.clone(),
                            VectorOp::VInsertLane => operands[1] = value.clone(),
                            op if uniform_shift(*op) => {
                                operands[1] = Operand::imm_i64(bits(&c.b[..8]) as i64);
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
        let mut asm = emit_at(&module, strategy, level).expect("emit SIMD assembly");
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
        let mut per_lane = Vec::new();
        for lane in LANES {
            let width = scalar(lane).size_bytes(PointerWidth::Bits64);
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
                    if per_lane_shift(op) {
                        let n = width as u64 * 8;
                        let interesting = [0, 1, 3, 5, n - 1, n, n + 1, u64::MAX];
                        for rotation in 0..interesting.len() {
                            let mut counts = [0; 16];
                            for i in 0..16 / width {
                                let count = interesting[(i + rotation) % interesting.len()];
                                counts[i * width..(i + 1) * width]
                                    .copy_from_slice(&count.to_le_bytes()[..width]);
                            }
                            per_lane.push(Case {
                                op,
                                lane,
                                index: 0,
                                a,
                                b: counts,
                            });
                        }
                        continue;
                    }
                    if uniform_shift(op) {
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
        let guard = TempDir::new("lamina-simd");
        let dir = guard.path();
        for strategy in [
            RegallocStrategy::Incremental,
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            for immediate in [false, true] {
                execute(
                    &cases,
                    strategy,
                    dir,
                    immediate,
                    PointerWidth::Bits64,
                    X86CpuLevel::default(),
                );
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                execute(
                    &per_lane,
                    strategy,
                    dir,
                    false,
                    PointerWidth::Bits64,
                    X86CpuLevel::V3,
                );
            }
        }
    }

    #[test]
    fn vector_values_stay_in_registers_under_global_allocation() {
        let ty = MirType::Vector(VectorType::V128(VectorLane::I64));
        let mut m = Module::new("resident");
        m.add_function(function("resident", ty, VectorOp::VAdd, 0));
        let slot_moves = |asm: &[u8]| {
            String::from_utf8_lossy(asm)
                .lines()
                .filter(|line| line.contains("movdqu") && line.contains("(%rbp)"))
                .count()
        };
        let linear = emit(&m, RegallocStrategy::LinearScanGlobal).expect("linear");
        let graph = emit(&m, RegallocStrategy::GraphColorGlobal).expect("graph");
        let incremental = emit(&m, RegallocStrategy::Incremental).expect("incremental");
        assert_eq!(slot_moves(&linear), 0);
        assert_eq!(slot_moves(&graph), 0);
        assert!(slot_moves(&incremental) > 0);
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
                    error.contains("Unsupported feature")
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
                    matches!(
                        &i.instr,
                        Instruction::VectorOp { .. } | Instruction::Ret { .. }
                    )
                });
                declare_uses_as_params(&mut f, &MirType::Vector(VectorType::V256(lane)), lane);
                m.add_function(f);
                let error = emit(&m, RegallocStrategy::Incremental).expect_err("V256 unsupported");
                assert!(error.to_lowercase().contains("v256"), "{error}");
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
                        matches!(
                            &i.instr,
                            Instruction::Store { .. } | Instruction::Ret { .. }
                        )
                    } else {
                        matches!(&i.instr, Instruction::Load { .. } | Instruction::Ret { .. })
                    }
                });
                declare_uses_as_params(&mut f, &MirType::Vector(VectorType::V256(lane)), lane);
                m.add_function(f);
                let error =
                    emit(&m, RegallocStrategy::Incremental).expect_err("V256 memory unsupported");
                assert!(error.to_lowercase().contains("v256"), "{error}");
            }
            for op in [VectorOp::VExtractLane, VectorOp::VInsertLane] {
                for index in [
                    Operand::reg(reg(0)),
                    Operand::imm_i64(-1),
                    Operand::imm_i64((16 / scalar(lane).size_bytes(PointerWidth::Bits64)) as i64),
                ] {
                    let mut m = Module::new("negative");
                    let mut f = function("bad", MirType::Vector(VectorType::V128(lane)), op, 0);
                    for i in &mut f.blocks[0].instructions {
                        if let Instruction::VectorOp { operands, .. } = &mut i.instr {
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
