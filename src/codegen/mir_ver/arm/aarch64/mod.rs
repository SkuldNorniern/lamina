use crate::codegen::mir_ver::TargetOs;
mod abi;
mod frame;
mod regalloc;
mod util;
use crate::mir::{Instruction as MirInst, Module as MirModule, Register};
use abi::{call_stub, public_symbol};
use frame::FrameMap;
use regalloc::A64RegAlloc;
use std::io::Write;
use std::result::Result;
use util::{emit_mov_imm64, imm_to_u64};

fn w_alias(xreg: &str) -> String {
    if let Some(rest) = xreg.strip_prefix('x') {
        format!("w{}", rest)
    } else {
        xreg.to_string()
    }
}

fn x_alias(reg: &str) -> String {
    if let Some(rest) = reg.strip_prefix('w') {
        format!("x{}", rest)
    } else {
        reg.to_string()
    }
}

pub fn generate_mir_aarch64<'a, W: Write>(
    module: &'a MirModule,
    writer: &mut W,
    target_os: TargetOs,
) -> Result<(), crate::error::LaminaError> {
    // Emit a shared format string for print intrinsics, then text section header
    match target_os {
        TargetOs::MacOs => {
            writeln!(writer, ".section __TEXT,__cstring,cstring_literals")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
        _ => {
            writeln!(writer, ".section .rodata")?;
            writeln!(writer, ".L_mir_fmt_int: .asciz \"%lld\\n\"")?;
        }
    }
    // Header
    writeln!(writer, ".text")?;

    for (func_name, func) in &module.functions {
        let (globl, label) = public_symbol(func_name, target_os);
        if let Some(g) = globl {
            writeln!(writer, "{}", g)?;
        }
        writeln!(writer, "{}:", label)?;

        // Prologue: save FP/LR and create a simple stack frame for virtual registers
        // Prologue: save fp/lr using scratch regs from RA to avoid hardcoding temps
        let mut ra_pro = A64RegAlloc::new();
        let s0 = ra_pro.alloc_scratch().unwrap_or("x19");
        let s1 = ra_pro.alloc_scratch().unwrap_or("x20");
        if s0 != "x29" || s1 != "x30" {
            // fallback to canonical pair; stp requires valid pair, keep canonical for correctness
            writeln!(writer, "    stp x29, x30, [sp, #-16]!")?;
        } else {
            writeln!(writer, "    stp {}, {}, [sp, #-16]!", s0, s1)?;
        }
        writeln!(writer, "    mov x29, sp")?;

        // Compute stack slots for all virtual registers in the function
        let frame = FrameMap::from_function(func);
        if frame.frame_size > 0 {
            writeln!(writer, "    sub sp, sp, #{}", frame.frame_size)?;
        }

        // Spill incoming arguments (x0..x7) to their slots
        for (i, p) in func.sig.params.iter().enumerate() {
            if i < 8 {
                if let Some(off) = frame.slot_of(&p.reg) {
                    // use scratch reg for address calculation
                    let addr = ra_pro.alloc_scratch().unwrap_or("x19");
                    if off >= 0 {
                        writeln!(writer, "    add {}, x29, #{}", addr, off)?;
                    } else {
                        writeln!(writer, "    sub {}, x29, #{}", addr, -off)?;
                    }
                    writeln!(writer, "    str x{}, [{}]", i, addr)?;
                    ra_pro.free_scratch(addr);
                }
            }
        }

        // Prepare epilogue label so `ret` can branch to it safely
        let epilogue_label = format!(".Lret_{}", label.trim_start_matches('_'));

        // Emit blocks
        if let Some(entry) = func.get_block(&func.entry) {
            let mut ra = A64RegAlloc::new();
            emit_block(
                entry.instructions.as_slice(),
                writer,
                &frame,
                target_os,
                &mut ra,
                &epilogue_label,
            )?;
        }
        for b in &func.blocks {
            if b.label != func.entry {
                writeln!(writer, "{}:", b.label)?;
                let mut ra = A64RegAlloc::new();
                emit_block(
                    b.instructions.as_slice(),
                    writer,
                    &frame,
                    target_os,
                    &mut ra,
                    &epilogue_label,
                )?;
            }
        }

        // Epilogue
        writeln!(writer, "{}:", epilogue_label)?;
        if frame.frame_size > 0 {
            writeln!(writer, "    add sp, sp, #{}", frame.frame_size)?;
        }
        writeln!(writer, "    ldp x29, x30, [sp], #16")?;
        writeln!(writer, "    ret")?;
    }
    Ok(())
}

fn emit_block<W: Write>(
    insts: &[MirInst],
    w: &mut W,
    frame: &FrameMap,
    os: TargetOs,
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
                // Delay materializing rhs when we can use an immediate form (for shifts)
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
                        // Unsigned division: udiv dst, lhs, rhs
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    udiv {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::SDiv => {
                        // Signed division: sdiv dst, lhs, rhs
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        writeln!(w, "    sdiv {}, {}, {}", dl, rl, rr)?;
                    }
                    crate::mir::IntBinOp::URem => {
                        // Unsigned remainder: r = lhs - (lhs / rhs) * rhs
                        emit_materialize_operand(w, rhs, s_r, frame, ra)?;
                        let rr = if is32 { w_alias(s_r) } else { x_alias(s_r) };
                        // Compute quotient in dl
                        writeln!(w, "    udiv {}, {}, {}", dl, rl, rr)?;
                        // dl = lhs - (dl * rhs)
                        // msub d, d, rr, rl  => d = rl - d*rr
                        writeln!(w, "    msub {}, {}, {}, {}", dl, dl, rr, rl)?;
                    }
                    crate::mir::IntBinOp::SRem => {
                        // Signed remainder: r = lhs - (lhs / rhs) * rhs
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
                    _ => writeln!(w, "    // TODO: unimplemented binop {}", op)?,
                }
                store_result(w, dst, &x_alias(s_d), frame, ra)?;
                ra.free_scratch(s_l);
                ra.free_scratch(s_r);
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
            MirInst::Load { ty, dst, addr, .. } => {
                // Materialize address to a scratch then load
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
                // dst = cond ? true_val : false_val
                let r_cond = ra.alloc_scratch().unwrap_or("x19");
                let r_t = ra.alloc_scratch().unwrap_or("x20");
                let r_f = ra.alloc_scratch().unwrap_or("x21");
                load_reg_to(w, cond, r_cond, frame, ra)?;
                emit_materialize_operand(w, true_val, r_t, frame, ra)?;
                emit_materialize_operand(w, false_val, r_f, frame, ra)?;
                // Compare cond against zero, then csel based on NE
                if ty.size_bytes() == 4 {
                    writeln!(w, "    cmp {}, #0", w_alias(r_cond))?;
                    writeln!(
                        w,
                        "    csel {}, {}, {}, ne",
                        w_alias(r_t),
                        w_alias(r_t),
                        w_alias(r_f)
                    )?;
                    // Store result from r_t (holds selected)
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
                            if total >= 0 && total <= 4095 {
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
                                if off >= 0 && off <= 4095 {
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
                            if off >= 0 && off <= 4095 {
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
                    match os {
                        TargetOs::MacOs => {
                            // Darwin AArch64 variadic ABI requires arguments to be available in the stack home area
                            // Ensure 16-byte alignment and spill the first vararg to stack
                            writeln!(w, "    sub sp, sp, #32")?; // create home area
                            writeln!(w, "    adrp x0, .L_mir_fmt_int@PAGE")?;
                            writeln!(w, "    add x0, x0, .L_mir_fmt_int@PAGEOFF")?;
                            writeln!(w, "    str x1, [sp]")?; // spill the vararg as required by ABI
                            writeln!(w, "    bl _printf")?;
                            writeln!(w, "    add sp, sp, #32")?; // restore stack
                        }
                        _ => {
                            writeln!(w, "    adrp x0, .L_mir_fmt_int")?;
                            writeln!(w, "    add x0, x0, :lo12:.L_mir_fmt_int")?;
                            writeln!(w, "    bl printf")?;
                        }
                    }
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                } else if name == "writebyte" && args.len() == 1 {
                    // Write a single byte to stdout using macOS ARM64 syscall
                    match os {
                        TargetOs::MacOs => {
                            // Reserve stack space to hold 1 byte buffer (keep 16B alignment)
                            writeln!(w, "    sub sp, sp, #16")?;
                            // Materialize byte value and store to [sp]
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
                            writeln!(w, "    strb {}, [sp]", w_alias("x9"))?;
                            // Setup write(fd=1, buf=sp, size=1)
                            writeln!(w, "    mov x0, #1")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #4")?; // write syscall
                            writeln!(w, "    svc #0")?;
                            // Return result in x0
                            if let Some(dst) = ret {
                                store_result(w, dst, "x0", frame, ra)?;
                            }
                            // Restore stack
                            writeln!(w, "    add sp, sp, #16")?;
                        }
                        _ => {
                            // Fallback: call C library write(int,void*,size_t)
                            emit_materialize_operand(w, &args[0], "x9", frame, ra)?;
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
                } else if name == "readbyte" && args.is_empty() {
                    match os {
                        TargetOs::MacOs => {
                            // Reserve stack for 1 byte buffer
                            writeln!(w, "    sub sp, sp, #16")?;
                            // Setup read(fd=0, buf=sp, size=1)
                            writeln!(w, "    mov x0, #0")?;
                            writeln!(w, "    mov x1, sp")?;
                            writeln!(w, "    mov x2, #1")?;
                            writeln!(w, "    mov x16, #3")?; // read syscall
                            writeln!(w, "    svc #0")?;
                            // Load byte and select return value: byte if 1 read, else error code in x0
                            writeln!(w, "    ldrb {}, [sp]", w_alias("x9"))?;
                            if let Some(dst) = ret {
                                let dst_reg = ra.alloc_scratch().unwrap_or("x10");
                                // Compare and select in 64-bit, then zero-extend if needed
                                writeln!(w, "    cmp x0, #1")?;
                                // Place read byte into x form register first
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
                            // Restore stack
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
                    match os {
                        TargetOs::MacOs => {
                            // Load pointer into x1 and byte into w9, write 1 byte
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
                    match os {
                        TargetOs::MacOs => {
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
                    match os {
                        TargetOs::MacOs => {
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
                    // Generic call: pass up to 8 args in x0..x7
                    for (i, a) in args.iter().enumerate().take(8) {
                        emit_materialize_operand(w, a, &format!("x{}", i), frame, ra)?;
                    }
                    // Resolve symbol: intrinsic stub or platform-mangled function name
                    let target_sym: String = match call_stub(name, os) {
                        Some(sym) => sym.to_string(),
                        None => match os {
                            TargetOs::MacOs => format!("_{}", name),
                            _ => name.to_string(),
                        },
                    };
                    writeln!(w, "    bl {}", target_sym)?;
                    if let Some(dst) = ret {
                        store_result(w, dst, "x0", frame, ra)?;
                    }
                }
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
                // Compare condition against zero explicitly for robustness
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
                if off >= -256 && off <= 255 {
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

fn store_result<W: Write>(
    w: &mut W,
    dst: &Register,
    src_reg: &str,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
) -> Result<(), crate::error::LaminaError> {
    match dst {
        Register::Virtual(_v) => {
            // Always store to stack slot for correctness across blocks
            if let Some(off) = frame.slot_of(dst) {
                // Use stur with signed 9-bit offset when possible, otherwise compute address
                if off >= -256 && off <= 255 {
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
                if *offset > 0 {
                    writeln!(w, "    add {}, {}, #{}", dest, dest, offset)?;
                } else {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -(*offset as i32))?;
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
                if *offset > 0 {
                    writeln!(w, "    add {}, {}, #{}", dest, dest, offset)?;
                } else {
                    writeln!(w, "    sub {}, {}, #{}", dest, dest, -(*offset as i32))?;
                }
            }
        }
    }
    Ok(())
}
