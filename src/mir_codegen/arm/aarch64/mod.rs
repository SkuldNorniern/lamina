//! AArch64 code generation for MIR (Mid-level IR).
//!
//! This module provides code generation from MIR to AArch64 assembly,
//! following the AAPCS64 calling convention.

mod abi;
mod frame;
mod regalloc;
mod util;

use abi::AArch64ABI;
use frame::FrameMap;
use regalloc::A64RegAlloc;
use std::io::Write;
use std::result::Result;
use util::{emit_mov_imm64, imm_to_u64};

use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use crate::mir_codegen::{
    Codegen, CodegenError, CodegenOptions,
    capability::{CapabilitySet, CodegenCapability},
};
use lamina_platform::TargetOperatingSystem;

/// Convert an x-register name to its w-register alias (lower 32 bits).
fn w_alias(xreg: &str) -> String {
    if let Some(rest) = xreg.strip_prefix('x') {
        format!("w{}", rest)
    } else {
        xreg.to_string()
    }
}

/// Convert a w-register name to its x-register alias (full 64 bits).
fn x_alias(reg: &str) -> String {
    if let Some(rest) = reg.strip_prefix('w') {
        format!("x{}", rest)
    } else {
        reg.to_string()
    }
}

fn compile_single_function_aarch64(
    func_name: &str,
    func: &crate::mir::Function,
    target_os: TargetOperatingSystem,
) -> Result<Vec<u8>, crate::mir_codegen::CodegenError> {
    use std::io::Write;
    let mut output = Vec::new();
    let abi = AArch64ABI::new(target_os);

        if let Some(globl) = abi.get_global_directive(func_name) {
        writeln!(output, "{}", globl).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        }
        let label = abi.mangle_function_name(func_name);
    writeln!(output, "{}:", label).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

        let mut ra_pro = A64RegAlloc::new();
        let s0 = ra_pro.alloc_scratch().unwrap_or("x19");
        let s1 = ra_pro.alloc_scratch().unwrap_or("x20");
        if s0 != "x29" || s1 != "x30" {
        writeln!(output, "    stp x29, x30, [sp, #-16]!").map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        } else {
        writeln!(output, "    stp {}, {}, [sp, #-16]!", s0, s1).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        }
    writeln!(output, "    mov x29, sp").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

        let frame = FrameMap::from_function(func);

        let has_many_vars = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter(|i| matches!(i, MirInst::IntBinary { .. }))
            .count()
            > 100;
        let adjusted_frame_size = if has_many_vars {
            (frame.frame_size + 1024) & !15
        } else {
            frame.frame_size
        };

        if adjusted_frame_size > 0 {
        writeln!(output, "    sub sp, sp, #{}", adjusted_frame_size).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        }

        for (i, p) in func.sig.params.iter().enumerate() {
            if let Some(off) = frame.slot_of(&p.reg) {
                let addr = ra_pro.alloc_scratch().unwrap_or("x19");
                if off >= 0 {
                writeln!(output, "    add {}, x29, #{}", addr, off).map_err(|e| {
                    crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                })?;
                } else {
                writeln!(output, "    sub {}, x29, #{}", addr, -off).map_err(|e| {
                    crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                })?;
                }

                if i < 8 {
                writeln!(output, "    str x{}, [{}]", i, addr).map_err(|e| {
                    crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                })?;
                } else {
                    let caller_offset = 16 + (i - 8) * 8;
                    let val_reg = ra_pro.alloc_scratch().unwrap_or("x20");
                writeln!(output, "    ldr {}, [x29, #{}]", val_reg, caller_offset).map_err(|e| {
                    crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                })?;
                writeln!(output, "    str {}, [{}]", val_reg, addr).map_err(|e| {
                    crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
                })?;
                    ra_pro.free_scratch(val_reg);
                }
                ra_pro.free_scratch(addr);
            }
        }

        let epilogue_label = format!(".Lret_{}", label.trim_start_matches('_'));

        let mut ra = A64RegAlloc::new();
    let has_complex_function = func.blocks.len() > 50 || func.blocks.iter().any(|b| b.instructions.len() > 100);
        if has_complex_function {
            ra.set_conservative_mode();
        }

        if let Some(entry) = func.get_block(&func.entry) {
            let mut ra_entry = A64RegAlloc::new();
            if has_complex_function {
                ra_entry.set_conservative_mode();
            }
            emit_block(
                entry.instructions.as_slice(),
            &mut output,
                &frame,
                &abi,
                &mut ra_entry,
                &epilogue_label,
        ).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
        })?;
        }
        for b in &func.blocks {
            if b.label != func.entry {
            writeln!(output, "    .align 2").map_err(|e| {
                crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
            })?;
            writeln!(output, "{}:", b.label).map_err(|e| {
                crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
            })?;
                let mut ra_block = A64RegAlloc::new();
                if has_complex_function {
                    ra_block.set_conservative_mode();
                }
                emit_block(
                    b.instructions.as_slice(),
                &mut output,
                    &frame,
                    &abi,
                    &mut ra_block,
                    &epilogue_label,
            ).map_err(|e| {
                crate::mir_codegen::CodegenError::InvalidCodegenOptions(e.to_string())
            })?;
            }
        }

    writeln!(output, "{}:", epilogue_label).map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
        if adjusted_frame_size > 0 {
        writeln!(output, "    add sp, sp, #{}", adjusted_frame_size).map_err(|e| {
            crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
        })?;
        }
    writeln!(output, "    ldp x29, x30, [sp], #16").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;
    writeln!(output, "    ret").map_err(|e| {
        crate::mir_codegen::CodegenError::InvalidCodegenOptions(format!("IO error: {}", e))
    })?;

    Ok(output)
}

/// Generate AArch64 assembly from a MIR module.
pub fn generate_mir_aarch64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), crate::error::LaminaError> {
    generate_mir_aarch64_with_units(module, writer, target_os, 1)
}

pub fn generate_mir_aarch64_with_units<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    codegen_units: usize,
) -> Result<(), crate::error::LaminaError> {
    emit_print_format_section(writer, target_os)?;
    writeln!(writer, ".text")?;

    let abi = AArch64ABI::new(target_os);

    for func_name in &module.external_functions {
        let label = abi.mangle_function_name(func_name);
        writeln!(writer, ".extern {}", label)?;
    }

    let results = compile_functions_parallel(
        module,
        target_os,
        codegen_units,
        compile_single_function_aarch64,
    ).map_err(|e| {
        use crate::codegen::FeatureType;
        crate::error::LaminaError::CodegenError(
            crate::codegen::CodegenError::UnsupportedFeature(
                FeatureType::Custom(format!("Parallel compilation error: {:?}", e))
            )
        )
    })?;

    for result in results {
        writer.write_all(&result.assembly)?;
    }

    Ok(())
}

/// Emit assembly for a sequence of MIR instructions.
fn emit_block<W: Write>(
    insts: &[MirInst],
    w: &mut W,
    frame: &FrameMap,
    abi: &AArch64ABI,
    ra: &mut A64RegAlloc,
    epilogue_label: &str,
) -> Result<(), crate::error::LaminaError> {
    for inst in insts {
        match inst {
            MirInst::IntBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                let is32 = ty.size_bytes() == 4;
                let dl = if is32 { w_alias(s_d) } else { x_alias(s_d) };
                let rl = if is32 { w_alias(s_l) } else { x_alias(s_l) };
                let _lhs_is_reg = matches!(lhs, crate::mir::Operand::Register(_));
                match op {
                    crate::mir::IntBinOp::Add => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    add {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::Sub => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    sub {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::Mul => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    mul {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::UDiv => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    udiv {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::SDiv => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    sdiv {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::URem => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    udiv {}, {}, {}", dl, rl, rr)?;
                        writeln!(w, "    msub {}, {}, {}, {}", dl, dl, rr, rl)?;
                    }
                    crate::mir::IntBinOp::SRem => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    sdiv {}, {}, {}", dl, rl, rr)?;
                        writeln!(w, "    msub {}, {}, {}, {}", dl, dl, rr, rl)?;
                    }
                    crate::mir::IntBinOp::And => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    and {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::Or => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    orr {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::Xor => {
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    eor {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::Shl => match rhs {
                        crate::mir::Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= if is32 { 31 } else { 63 };
                            writeln!(w, "    lsl {}, {}, #{}", dl, rl, sh)?;
                        }
                        _ => {
                            emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                            let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                            writeln!(w, "    lslv {}, {}, {}", dl, rl, rr)?;
                        }
                    },
                    crate::mir::IntBinOp::LShr => match rhs {
                        crate::mir::Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= if is32 { 31 } else { 63 };
                            writeln!(w, "    lsr {}, {}, #{}", dl, rl, sh)?;
                        }
                        _ => {
                            emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                            let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                            writeln!(w, "    lsrv {}, {}, {}", dl, rl, rr)?;
                        }
                    },
                    crate::mir::IntBinOp::AShr => match rhs {
                        crate::mir::Operand::Immediate(imm) => {
                            let mut sh = imm_to_u64(imm) as u32;
                            sh &= if is32 { 31 } else { 63 };
                            writeln!(w, "    asr {}, {}, #{}", dl, rl, sh)?;
                        }
                        _ => {
                            emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                            let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                            writeln!(w, "    asrv {}, {}, {}", dl, rl, rr)?;
                        }
                    },
                }
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatBinary {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes() == 4;
                let suffix = if is32 { "s" } else { "d" };

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_l))?;
                    writeln!(w, "    fmov s1, {}", w_alias(s_r))?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_l))?;
                    writeln!(w, "    fmov d1, {}", x_alias(s_r))?;
                }

                match op {
                    crate::mir::FloatBinOp::FAdd => {
                        writeln!(w, "    fadd {}0, {}0, {}1", suffix, suffix, suffix)?
                    }
                    crate::mir::FloatBinOp::FSub => {
                        writeln!(w, "    fsub {}0, {}0, {}1", suffix, suffix, suffix)?
                    }
                    crate::mir::FloatBinOp::FMul => {
                        writeln!(w, "    fmul {}0, {}0, {}1", suffix, suffix, suffix)?
                    }
                    crate::mir::FloatBinOp::FDiv => {
                        writeln!(w, "    fdiv {}0, {}0, {}1", suffix, suffix, suffix)?
                    }
                }

                if is32 {
                    writeln!(w, "    fmov {}, s0", w_alias(s_d))?;
                } else {
                    writeln!(w, "    fmov {}, d0", x_alias(s_d))?;
                }

                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatUnary { op, src, dst, ty } => {
                let s_s = ra.alloc_scratch().unwrap_or("x19");
                let s_d = ra.alloc_scratch().unwrap_or("x20");
                emit_materialize_operand(w, src, s_s, frame, ra)?;

                let is32 = ty.size_bytes() == 4;
                let suffix = if is32 { "s" } else { "d" };

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_s))?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_s))?;
                }

                match op {
                    crate::mir::FloatUnOp::FNeg => {
                        writeln!(w, "    fneg {}0, {}0", suffix, suffix)?
                    }
                    crate::mir::FloatUnOp::FSqrt => {
                        writeln!(w, "    fsqrt {}0, {}0", suffix, suffix)?
                    }
                }

                if is32 {
                    writeln!(w, "    fmov {}, s0", w_alias(s_d))?;
                } else {
                    writeln!(w, "    fmov {}, d0", x_alias(s_d))?;
                }

                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_s);
                ra.free_scratch(s_d);
            }
            MirInst::IntCmp {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                if ty.size_bytes() == 4 {
                    writeln!(w, "    cmp {}, {}", w_alias(s_l), w_alias(s_r))?;
                } else {
                    writeln!(w, "    cmp {}, {}", x_alias(s_l), x_alias(s_r))?;
                }
                let cond = match op {
                    crate::mir::IntCmpOp::Eq => "eq",
                    crate::mir::IntCmpOp::Ne => "ne",
                    crate::mir::IntCmpOp::SLt => "lt",
                    crate::mir::IntCmpOp::SLe => "le",
                    crate::mir::IntCmpOp::SGt => "gt",
                    crate::mir::IntCmpOp::SGe => "ge",
                    crate::mir::IntCmpOp::ULt => "lo",
                    crate::mir::IntCmpOp::ULe => "ls",
                    crate::mir::IntCmpOp::UGt => "hi",
                    crate::mir::IntCmpOp::UGe => "hs",
                };
                writeln!(w, "    cset {}, {}", x_alias(s_d), cond)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::FloatCmp {
                op,
                lhs,
                rhs,
                dst,
                ty,
            } => {
                let s_l = ra.alloc_scratch().unwrap_or("x19");
                let s_r = ra.alloc_scratch().unwrap_or("x20");
                let s_d = ra.alloc_scratch().unwrap_or("x21");
                emit_materialize_operand(w, lhs, s_l, frame, ra)?;
                emit_materialize_operand(w, rhs, s_r, frame, ra)?;

                let is32 = ty.size_bytes() == 4;

                if is32 {
                    writeln!(w, "    fmov s0, {}", w_alias(s_l))?;
                    writeln!(w, "    fmov s1, {}", w_alias(s_r))?;
                    writeln!(w, "    fcmp s0, s1")?;
                } else {
                    writeln!(w, "    fmov d0, {}", x_alias(s_l))?;
                    writeln!(w, "    fmov d1, {}", x_alias(s_r))?;
                    writeln!(w, "    fcmp d0, d1")?;
                }

                let cond = match op {
                    crate::mir::FloatCmpOp::Eq => "eq",
                    crate::mir::FloatCmpOp::Ne => "ne",
                    crate::mir::FloatCmpOp::Lt => "mi",
                    crate::mir::FloatCmpOp::Le => "ls",
                    crate::mir::FloatCmpOp::Gt => "gt",
                    crate::mir::FloatCmpOp::Ge => "ge",
                };

                writeln!(w, "    cset {}, {}", x_alias(s_d), cond)?;
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
                ra.free_scratch(s_d);
            }
            MirInst::Load { ty, dst, addr, .. } => {
                let a = ra.alloc_scratch().unwrap_or("x19");
                let t = ra.alloc_scratch().unwrap_or("x20");
                materialize_address(w, addr, a, frame, ra)?;
                match ty.size_bytes() {
                    1 => writeln!(w, "    ldrb {}, [{}]", w_alias(t), a)?,
                    2 => writeln!(w, "    ldrh {}, [{}]", w_alias(t), a)?,
                    4 => writeln!(w, "    ldr {}, [{}]", w_alias(t), a)?,
                    8 => writeln!(w, "    ldr {}, [{}]", x_alias(t), a)?,
                    _ => writeln!(w, "    // Unhandled load size: {}", ty.size_bytes())?,
                }
                store_result(w, dst, &x_alias(t), frame, ra)?;
                ra.free_scratch(a);
                ra.free_scratch(t);
            }
            MirInst::Select {
                ty,
                dst,
                cond,
                true_val,
                false_val,
            } => {
                let r_cond = ra.alloc_scratch().unwrap_or("x19");
                let r_t = ra.alloc_scratch().unwrap_or("x20");
                let r_f = ra.alloc_scratch().unwrap_or("x21");
                load_reg_to(w, cond, r_cond, frame, ra)?;
                emit_materialize_operand(w, true_val, r_t, frame, ra)?;
                emit_materialize_operand(w, false_val, r_f, frame, ra)?;
                if ty.size_bytes() == 4 {
                    writeln!(w, "    cmp {}, #0", w_alias(r_cond))?;
                    writeln!(
                        w,
                        "    csel {}, {}, {}, ne",
                        w_alias(r_t),
                        w_alias(r_t),
                        w_alias(r_f)
                    )?;
                    store_result(w, dst, &x_alias(r_t), frame, ra)?;
                } else {
                    writeln!(w, "    cmp {}, #0", x_alias(r_cond))?;
                    writeln!(
                        w,
                        "    csel {}, {}, {}, ne",
                        x_alias(r_t),
                        x_alias(r_t),
                        x_alias(r_f)
                    )?;
                    store_result(w, dst, &x_alias(r_t), frame, ra)?;
                }
                ra.free_scratch(r_cond);
                ra.free_scratch(r_t);
                ra.free_scratch(r_f);
            }
            MirInst::Store { ty, src, addr, .. } => {
                let a = ra.alloc_scratch().unwrap_or("x19");
                let t = ra.alloc_scratch().unwrap_or("x20");
                materialize_address(w, addr, a, frame, ra)?;
                emit_materialize_operand(w, src, t, frame, ra)?;
                match ty.size_bytes() {
                    1 => writeln!(w, "    strb {}, [{}]", w_alias(t), a)?,
                    2 => writeln!(w, "    strh {}, [{}]", w_alias(t), a)?,
                    4 => writeln!(w, "    str {}, [{}]", w_alias(t), a)?,
                    8 => writeln!(w, "    str {}, [{}]", x_alias(t), a)?,
                    _ => writeln!(w, "    // Unhandled store size: {}", ty.size_bytes())?,
                }
                ra.free_scratch(a);
                ra.free_scratch(t);
            }
            MirInst::Lea { dst, base, offset } => {
                let t = ra.alloc_scratch().unwrap_or("x19");
                // LEA computes address of base's stack slot + offset
                match base {
                    Register::Virtual(_) => {
                        if let Some(slot_off) = frame.slot_of(base) {
                            // Compute address: x29 + slot_off + offset
                            let total = slot_off as i64 + (*offset as i64);
                            if (0..=4095).contains(&total) {
                                writeln!(w, "    add {}, x29, #{}", t, total)?;
                            } else if total < 0 && -total <= 4095 {
                                writeln!(w, "    sub {}, x29, #{}", t, -total)?;
                            } else {
                                emit_mov_imm64(w, t, total as u64)?;
                                writeln!(w, "    add {}, x29, {}", t, t)?;
                            }
                        } else {
                            // No stack slot: load value from physical reg and add offset
                            load_reg_to(w, base, t, frame, ra)?;
                            if *offset != 0 {
                                let off = *offset as i64;
                                if (0..=4095).contains(&off) {
                                    writeln!(w, "    add {}, {}, #{}", t, t, off)?;
                                } else if off < 0 && -off <= 4095 {
                                    writeln!(w, "    sub {}, {}, #{}", t, t, -off)?;
                                } else {
                                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                                    emit_mov_imm64(w, oreg, off as u64)?;
                                    writeln!(w, "    add {}, {}, {}", t, t, oreg)?;
                                    ra.free_scratch(oreg);
                                }
                            }
                        }
                    }
                    _ => {
                        // Physical register: load value and add offset
                        load_reg_to(w, base, t, frame, ra)?;
                        if *offset != 0 {
                            let off = *offset as i64;
                            if (0..=4095).contains(&off) {
                                writeln!(w, "    add {}, {}, #{}", t, t, off)?;
                            } else if off < 0 && -off <= 4095 {
                                writeln!(w, "    sub {}, {}, #{}", t, t, -off)?;
                            } else {
                                let oreg = ra.alloc_scratch().unwrap_or("x20");
                                emit_mov_imm64(w, oreg, off as u64)?;
                                writeln!(w, "    add {}, {}, {}", t, t, oreg)?;
                                ra.free_scratch(oreg);
                            }
                        }
                    }
                }
                store_result(w, dst, t, frame, ra)?;
                ra.free_scratch(t);
            }
            MirInst::Call { name, args, ret } => {
                if name == "print" && args.len() == 1 {
                    // Special-case intrinsic: print(integer) via printf("%lld\n", value)
                    emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            // Darwin AArch64 variadic ABI requires arguments to be available in the stack home area
                            // Ensure 16-byte alignment and spill the first vararg to stack
                            writeln!(w, "    sub sp, sp, #32")?; // create home area
                            writeln!(w, "    adrp x0, .L_mir_fmt_int@PAGE")?;
                            writeln!(w, "    add x0, x0, .L_mir_fmt_int@PAGEOFF")?;
                            writeln!(w, "    str x1, [sp]")?; // spill the vararg as required by ABI
                            writeln!(w, "    bl _printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(w, "    mov x0, #0")?; // NULL flushes all streams
                            writeln!(w, "    bl _fflush")?;
                            writeln!(w, "    add sp, sp, #32")?; // restore stack
                        }
                        _ => {
                            writeln!(w, "    adrp x0, .L_mir_fmt_int")?;
                            writeln!(w, "    add x0, x0, :lo12:.L_mir_fmt_int")?;
                            writeln!(w, "    bl printf")?;
                            // Flush stdout to ensure output appears immediately when mixing with syscall I/O
                            writeln!(w, "    mov x0, #0")?; // NULL flushes all streams
                            writeln!(w, "    bl fflush")?;
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else if name == "writebyte" && args.len() == 1 {
                    // Write a single byte to stdout using macOS ARM64 syscall
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            writeln!(w, "    sub sp, sp, #16")?;
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl write")?;
                            writeln!(w, "    dmb sy")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                    }
                } else if name == "readbyte" && args.is_empty() {
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            writeln!(w, "    ldrb {}, [sp]", w_alias("x9"))?;
                            if let Some(dst) = ret {
                                let dst_reg = ra.alloc_scratch().unwrap_or("x10");
                                writeln!(w, "    cmp x0, #1")?;
                                writeln!(w, "    uxtb {}, {}", x_alias("x9"), w_alias("x9"))?;
                                writeln!(
                                    w,
                                    "    csel {}, {}, x0, eq",
                                    x_alias(dst_reg),
                                    x_alias("x9")
                                )?;
                                store_result(w, dst, &x_alias(dst_reg), frame, ra)?;
                                ra.free_scratch(dst_reg);
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            // Fallback to libc read
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl read")?;
                            writeln!(w, "    ldrb {}, [sp]", w_alias("x9"))?;
                            if let Some(dst) = ret {
                                let dst_reg = ra.alloc_scratch().unwrap_or("x10");
                                writeln!(w, "    cmp x0, #1")?;
                                writeln!(w, "    uxtb {}, {}", x_alias("x9"), w_alias("x9"))?;
                                writeln!(
                                    w,
                                    "    csel {}, {}, x0, eq",
                                    x_alias(dst_reg),
                                    x_alias("x9")
                                )?;
                                store_result(w, dst, &x_alias(dst_reg), frame, ra)?;
                                ra.free_scratch(dst_reg);
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                    }
                } else if name == "writeptr" && args.len() == 1 {
                    // Write the byte value at pointer to stdout
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            writeln!(w, "    ldrb {}, [x1]", w_alias("x9"))?;
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            writeln!(w, "    ldrb {}, [x1]", w_alias("x9"))?;
                            writeln!(w, "    sub sp, sp, #16")?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    bl write")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                    }
                } else if name == "write" && args.len() == 2 {
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            // write(fd=1, buf=args[0], size=args[1])
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x16, #4")?;
                            writeln!(w, "    svc #0")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    bl write")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                    }
                } else if name == "read" && args.len() == 2 {
                    match abi.target_os() {
                        TargetOperatingSystem::MacOS => {
                            // read(fd=0, buf=args[0], size=args[1])
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x16, #3")?;
                            writeln!(w, "    svc #0")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                        _ => {
                            emit_materialize_operand(w, &args[0], "x1", frame, ra)?;
                            emit_materialize_operand(w, &args[1], "x2", frame, ra)?;
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    bl read")?;
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                        }
                    }
                } else {
                    for (i, a) in args.iter().enumerate().take(8) {
                        emit_materialize_operand(w, a, &format!("x{}", i), frame, ra)?;
                    }

                    let stack_args = if args.len() > 8 { &args[8..] } else { &[] };
                    let stack_space = (stack_args.len() * 8 + 15) & !15;

                    if stack_space > 0 {
                        writeln!(w, "    sub sp, sp, #{}", stack_space)?;
                        for (i, a) in stack_args.iter().enumerate() {
                            let offset = i * 8;
                            let scratch = ra.alloc_scratch().unwrap_or("x9");
                            emit_materialize_operand(w, a, scratch, frame, ra)?;
                            writeln!(w, "    str {}, [sp, #{}]", scratch, offset)?;
                            ra.free_scratch(scratch);
                        }
                    }

                    let target_sym: String = abi
                        .call_stub(name)
                        .unwrap_or_else(|| abi.mangle_function_name(name));
                    writeln!(w, "    bl {}", target_sym)?;

                    if stack_space > 0 {
                        writeln!(w, "    add sp, sp, #{}", stack_space)?;
                    }

                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                }
            }
            MirInst::TailCall { name, args } => {
                for (i, a) in args.iter().enumerate().take(8) {
                    emit_materialize_operand(w, a, &format!("x{}", i), frame, ra)?;
                }
                let target_sym: String = abi
                    .call_stub(name)
                    .unwrap_or_else(|| abi.mangle_function_name(name));
                if frame.frame_size > 0 {
                    writeln!(w, "    add sp, sp, #{}", frame.frame_size)?;
                }
                writeln!(w, "    ldp x29, x30, [sp], #16")?;
                writeln!(w, "    b {}", target_sym)?;
            }
            MirInst::Ret { value } => {
                if let Some(v) = value {
                    emit_materialize_operand(w, v, "x0", frame, ra)?;
                }
                writeln!(w, "    b {}", epilogue_label)?;
            }
            MirInst::Jmp { target } => {
                writeln!(w, "    b {}", target)?;
            }
            MirInst::Br {
                cond,
                true_target,
                false_target,
            } => {
                let t = ra.alloc_scratch().unwrap_or("x19");
                load_reg_to(w, cond, t, frame, ra)?;
                writeln!(w, "    cmp {}, #0", x_alias(t))?;
                writeln!(w, "    b.ne {}", true_target)?;
                writeln!(w, "    b {}", false_target)?;
                ra.free_scratch(t);
            }
            _ => {
                writeln!(w, "    // Unhandled MIR: {}", inst)?;
            }
        }
    }
    Ok(())
}

/// Materialize an operand into a register.
fn emit_materialize_operand<W: Write>(
    w: &mut W,
    op: &crate::mir::Operand,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), crate::error::LaminaError> {
    match op {
        crate::mir::Operand::Immediate(imm) => emit_mov_imm64(w, dest, imm_to_u64(imm))?,
        crate::mir::Operand::Register(r) => load_reg_to(w, r, dest, frame, ra)?,
    }
    Ok(())
}

/// Load a register value into a destination register.
fn load_reg_to<W: Write>(
    w: &mut W,
    r: &Register,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), crate::error::LaminaError> {
    match r {
        Register::Virtual(_v) => {
            // Always load from stack slot for correctness across blocks
            if let Some(off) = frame.slot_of(r) {
                // Use ldur with signed 9-bit offset when possible, otherwise compute address
                if (-256..=255).contains(&off) {
                    writeln!(w, "    ldur {}, [x29, #{}]", dest, off)?;
                } else {
                    let mut addr = ra.alloc_scratch().unwrap_or("x12");
                    if addr == dest {
                        if let Some(other) = ra.alloc_scratch() {
                            addr = other;
                        } else {
                            addr = if dest != "x11" { "x11" } else { "x10" };
                        }
                    }
                    if off >= 0 {
                        writeln!(w, "    add {}, x29, #{}", addr, off)?;
                    } else {
                        writeln!(w, "    sub {}, x29, #{}", addr, -off)?;
                    }
                    writeln!(w, "    ldr {}, [{}]", dest, addr)?;
                    if matches!(addr, "x9" | "x10" | "x11" | "x12") {
                        ra.free_scratch(addr);
                    }
                }
            } else {
                writeln!(w, "    // no slot for {}, leaving {} unchanged", r, dest)?;
            }
        }
        Register::Physical(p) => {
            if p.name != dest {
                writeln!(w, "    mov {}, {}", dest, p.name)?;
            }
        }
    }
    Ok(())
}

/// Store a value from a source register to a destination register or stack slot.
fn store_result<W: Write>(
    w: &mut W,
    dst: &Register,
    src_reg: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), crate::error::LaminaError> {
    match dst {
        Register::Virtual(_v) => {
            if let Some(off) = frame.slot_of(dst) {
                if (-256..=255).contains(&off) {
                    writeln!(w, "    stur {}, [x29, #{}]", src_reg, off)?;
                } else {
                    let mut addr = ra.alloc_scratch().unwrap_or("x12");
                    if addr == src_reg {
                        if let Some(other) = ra.alloc_scratch() {
                            addr = other;
                        } else {
                            addr = if src_reg != "x11" { "x11" } else { "x10" };
                        }
                    }
                    if off >= 0 {
                        writeln!(w, "    add {}, x29, #{}", addr, off)?;
                    } else {
                        writeln!(w, "    sub {}, x29, #{}", addr, -off)?;
                    }
                    writeln!(w, "    str {}, [{}]", src_reg, addr)?;
                    if matches!(addr, "x9" | "x10" | "x11" | "x12") {
                        ra.free_scratch(addr);
                    }
                }
            } else {
                writeln!(w, "    // no slot for {}", dst)?;
            }
        }
        Register::Physical(p) => {
            if p.name != src_reg {
                writeln!(w, "    mov {}, {}", p.name, src_reg)?;
            }
        }
    }
    Ok(())
}

/// Materialize an address operand into a register.
fn materialize_address<W: Write>(
    w: &mut W,
    addr: &crate::mir::AddressMode,
    dest: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), crate::error::LaminaError> {
    match addr {
        crate::mir::AddressMode::BaseOffset { base, offset } => {
            // Materialize base value (should be an address) into dest
            load_reg_to(w, base, dest, frame, ra)?;
            if *offset != 0 {
                let off_val = *offset as i64; // Sign-extend i16 to i64 for range checking
                // AArch64 add/sub with immediate supports 12-bit signed immediates (-2048 to 2047)
                if (0..=4095).contains(&off_val) {
                    writeln!(w, "    add {}, {}, #{}", dest, dest, off_val)?;
                } else if off_val < 0 && -off_val <= 4095 {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -off_val)?;
                } else {
                    // Offset too large, materialize in register
                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                    if oreg == dest {
                        // Need a different register
                        if let Some(other) = ra.alloc_scratch() {
                            let temp = other;
                            emit_mov_imm64(w, temp, off_val as u64)?;
                            writeln!(w, "    add {}, {}, {}", dest, dest, temp)?;
                            ra.free_scratch(temp);
                        } else {
                            // Fallback: use x20 if dest is not x20
                            let temp = if dest != "x20" { "x20" } else { "x19" };
                            emit_mov_imm64(w, temp, off_val as u64)?;
                            writeln!(w, "    add {}, {}, {}", dest, dest, temp)?;
                        }
                    } else {
                        emit_mov_imm64(w, oreg, off_val as u64)?;
                        writeln!(w, "    add {}, {}, {}", dest, dest, oreg)?;
                        ra.free_scratch(oreg);
                    }
                }
            }
        }
        crate::mir::AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
        } => {
            // dest = base (base holds an address)
            load_reg_to(w, base, dest, frame, ra)?;
            let mut idx = ra.alloc_scratch().unwrap_or("x19");
            if idx == dest {
                if let Some(other) = ra.alloc_scratch() {
                    idx = other;
                } else {
                    idx = "x20";
                }
            }
            load_reg_to(w, index, idx, frame, ra)?; // idx = index
            let sh = match *scale {
                1 => 0,
                2 => 1,
                4 => 2,
                8 => 3,
                _ => 0,
            };
            if sh > 0 {
                writeln!(w, "    lsl {}, {}, #{}", idx, idx, sh)?;
            }
            writeln!(w, "    add {}, {}, {}", dest, dest, idx)?;
            ra.free_scratch(idx);
            if *offset != 0 {
                let off_val = *offset as i64; // Sign-extend i8 to i64 for range checking
                // AArch64 add/sub with immediate supports 12-bit signed immediates (-2048 to 2047)
                // i8 is always within this range, but check for consistency
                if (0..=4095).contains(&off_val) {
                    writeln!(w, "    add {}, {}, #{}", dest, dest, off_val)?;
                } else if off_val < 0 && -off_val <= 4095 {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -off_val)?;
                } else {
                    // Offset too large (shouldn't happen with i8, but handle for safety)
                    let oreg = ra.alloc_scratch().unwrap_or("x20");
                    if oreg == dest {
                        let temp = if dest != "x20" { "x20" } else { "x19" };
                        emit_mov_imm64(w, temp, off_val as u64)?;
                        writeln!(w, "    add {}, {}, {}", dest, dest, temp)?;
                    } else {
                        emit_mov_imm64(w, oreg, off_val as u64)?;
                        writeln!(w, "    add {}, {}, {}", dest, dest, oreg)?;
                        ra.free_scratch(oreg);
                    }
                }
            }
        }
    }
    Ok(())
}

use crate::mir_codegen::common::{compile_functions_parallel, CodegenBase, emit_print_format_section};

/// Trait-backed MIR ⇒ AArch64 code generator.
pub struct AArch64Codegen<'a> {
    base: CodegenBase<'a>,
}

impl<'a> AArch64Codegen<'a> {
    pub fn new(target_os: TargetOperatingSystem) -> Self {
        Self {
            base: CodegenBase::new(target_os),
        }
    }

    /// Attach the MIR module that should be emitted in the next codegen pass.
    pub fn set_module(&mut self, module: &'a MirModule) {
        self.base.set_module(module);
    }

    /// Drain the internal assembly buffer produced by `emit_asm`.
    pub fn drain_output(&mut self) -> Vec<u8> {
        self.base.drain_output()
    }

    /// Emit assembly for the provided module directly into the supplied writer.
    pub fn emit_into<W: Write>(
        &mut self,
        module: &'a MirModule,
        writer: &mut W,
        codegen_units: usize,
    ) -> Result<(), crate::error::LaminaError> {
        generate_mir_aarch64_with_units(module, writer, self.base.target_os, codegen_units)
    }
}

impl<'a> Codegen for AArch64Codegen<'a> {
    const BIN_EXT: &'static str = "o";
    const CAN_OUTPUT_ASM: bool = true;
    const CAN_OUTPUT_BIN: bool = false;
    const SUPPORTED_CODEGEN_OPTS: &'static [CodegenOptions] =
        &[CodegenOptions::Debug, CodegenOptions::Release];
    const TARGET_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;
    const MAX_BIT_WIDTH: u8 = 64;

    fn capabilities() -> CapabilitySet {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            CodegenCapability::Print,
            CodegenCapability::StackAllocation,
            CodegenCapability::MemoryOperations,
            CodegenCapability::SystemCalls,
            CodegenCapability::InlineAssembly,
            CodegenCapability::SimdOperations,
            CodegenCapability::ForeignFunctionInterface,
        ]
        .into_iter()
        .collect()
    }

    fn prepare(
        &mut self,
        types: &std::collections::HashMap<String, crate::mir::MirType>,
        globals: &std::collections::HashMap<String, crate::mir::Global>,
        funcs: &std::collections::HashMap<String, crate::mir::Signature>,
        codegen_units: usize,
        verbose: bool,
        options: &[CodegenOptions],
        input_name: &str,
    ) -> Result<(), CodegenError> {
        self.base
            .prepare_base(types, globals, funcs, codegen_units, verbose, options, input_name)
    }

    fn compile(&mut self) -> Result<(), CodegenError> {
        self.base.compile_base()
    }

    fn finalize(&mut self) -> Result<(), CodegenError> {
        self.base.finalize_base()
    }

    fn emit_asm(&mut self) -> Result<(), CodegenError> {
        self.base.emit_asm_base_with_units(
            |module, writer, target_os, codegen_units| {
                generate_mir_aarch64_with_units(module, writer, target_os, codegen_units)
            },
            "AArch64",
            self.base.codegen_units,
        )
    }

    fn emit_bin(&mut self) -> Result<(), CodegenError> {
        Err(CodegenError::UnsupportedFeature(
            "Binary emission not implemented for AArch64 MIR backend".into(),
        ))
    }
}
