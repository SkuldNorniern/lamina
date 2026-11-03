use super::*;
use crate::mir::{Instruction, VectorLane, VectorOp};
use crate::mir_codegen::simd::{index, lane, unsupported, width};

fn address(
    w: &mut impl Write,
    addr: &AddressMode,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset, .. } => {
            load_register_wasm(base, w, locals)?;
            writeln!(w, "    i64.const {offset}\n    i64.add")?;
        }
        AddressMode::BaseIndexScale {
            base,
            index,
            scale,
            offset,
            ..
        } => {
            load_register_wasm(base, w, locals)?;
            load_register_wasm(index, w, locals)?;
            writeln!(
                w,
                "    i64.const {scale}\n    i64.mul\n    i64.add\n    i64.const {offset}\n    i64.add"
            )?;
        }
    }
    writeln!(w, "    i32.wrap_i64")?;
    Ok(())
}

fn scalar(
    w: &mut impl Write,
    operand: &Operand,
    lane: VectorLane,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    load_operand_wasm(operand, w, locals)?;
    if width(lane) < 64 {
        writeln!(w, "    i32.wrap_i64")?;
    }
    match lane {
        VectorLane::F32 => writeln!(w, "    f32.reinterpret_i32")?,
        VectorLane::F64 => writeln!(w, "    f64.reinterpret_i64")?,
        _ => (),
    }
    Ok(())
}

pub(super) fn emit(
    inst: &Instruction,
    w: &mut impl Write,
    locals: &HashMap<VirtualReg, usize>,
) -> Result<(), LaminaError> {
    match inst {
        Instruction::Load { ty, dst, addr, .. } => {
            lane(*ty)?;
            address(w, addr, locals)?;
            writeln!(w, "    v128.load align=1")?;
            store_to_register_wasm(dst, w, locals)?;
        }
        Instruction::Store { ty, src, addr, .. } => {
            lane(*ty)?;
            address(w, addr, locals)?;
            load_operand_wasm(src, w, locals)?;
            writeln!(w, "    v128.store align=1")?;
        }
        Instruction::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            use VectorOp::*;
            let lane = lane(*ty)?;
            let bits = width(lane);
            let prefix = match lane {
                VectorLane::I8 => "i8x16",
                VectorLane::I16 => "i16x8",
                VectorLane::I32 => "i32x4",
                VectorLane::I64 => "i64x2",
                VectorLane::F32 => "f32x4",
                VectorLane::F64 => "f64x2",
            };
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            if (*op == VMul && lane == VectorLane::I8)
                || (float
                    && matches!(
                        op,
                        VShl | VLShr | VAShr | VShlLanes | VLShrLanes | VAShrLanes
                    ))
            {
                return Err(unsupported(format!("WASM unsupported {op:?} for {lane:?}")));
            }
            if matches!(op, VShlLanes | VLShrLanes | VAShrLanes) {
                return Err(unsupported(format!(
                    "WASM unsupported {op:?} for {lane:?}: a per-lane vector shift has no WASM SIMD form, which has only {prefix}.shl, {prefix}.shr_u and {prefix}.shr_s, one count for all lanes\n  Hint: splat the count and write the scalar form, 'shl.<N x T> %a, %n' with a scalar %n"
                )));
            }
            if matches!(op, VExtractLane | VInsertLane) {
                let i = index(*op, lane, operands)?;
                load_operand_wasm(&operands[0], w, locals)?;
                if *op == VExtractLane {
                    let suffix = if bits < 32 { "_u" } else { "" };
                    writeln!(w, "    {prefix}.extract_lane{suffix} {i}")?;
                    match lane {
                        VectorLane::F32 => {
                            writeln!(w, "    i32.reinterpret_f32\n    i64.extend_i32_u")?
                        }
                        VectorLane::F64 => writeln!(w, "    i64.reinterpret_f64")?,
                        _ if bits < 64 => writeln!(w, "    i64.extend_i32_u")?,
                        _ => (),
                    }
                } else {
                    scalar(w, &operands[1], lane, locals)?;
                    writeln!(w, "    {prefix}.replace_lane {i}")?;
                }
            } else if *op == VSplat {
                if operands.len() != 1 {
                    return Err(unsupported("VSplat requires one operand"));
                }
                scalar(w, &operands[0], lane, locals)?;
                writeln!(w, "    {prefix}.splat")?;
            } else {
                if operands.len() != 2 {
                    return Err(unsupported(format!("{op:?} requires two operands")));
                }
                if matches!(op, VShl | VLShr | VAShr) {
                    load_operand_wasm(&operands[1], w, locals)?;
                    writeln!(
                        w,
                        "    i64.const {bits}\n    i64.ge_u\n    if (result v128)"
                    )?;
                    if *op == VAShr {
                        load_operand_wasm(&operands[0], w, locals)?;
                        writeln!(w, "    i32.const {}\n    {prefix}.shr_s", bits - 1)?;
                    } else {
                        writeln!(w, "    v128.const i32x4 0 0 0 0")?;
                    }
                    writeln!(w, "    else")?;
                    load_operand_wasm(&operands[0], w, locals)?;
                    load_operand_wasm(&operands[1], w, locals)?;
                    writeln!(w, "    i32.wrap_i64")?;
                    let mnemonic = match op {
                        VShl => "shl",
                        VLShr => "shr_u",
                        _ => "shr_s",
                    };
                    writeln!(w, "    {prefix}.{mnemonic}\n    end")?;
                } else {
                    load_operand_wasm(&operands[0], w, locals)?;
                    load_operand_wasm(&operands[1], w, locals)?;
                    let mnemonic = match op {
                        VAdd => format!("{prefix}.add"),
                        VSub => format!("{prefix}.sub"),
                        VMul => format!("{prefix}.mul"),
                        VAnd => "v128.and".into(),
                        VOr => "v128.or".into(),
                        VXor => "v128.xor".into(),
                        _ => {
                            return Err(unsupported(format!(
                                "WASM unsupported {op:?} for {lane:?}"
                            )));
                        }
                    };
                    writeln!(w, "    {mnemonic}")?;
                }
            }
            store_to_register_wasm(dst, w, locals)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use crate::test_support::TempDir;

    use crate::mir_codegen::simd_support::*;
    use crate::{
        mir::{MirType, Module, VectorLane, VectorOp, VectorType},
        mir_codegen::RegallocStrategy,
    };
    use lamina_platform::PointerWidth;
    use std::{array, fs, process::Command};

    const MISMATCH_FN: &str = "(func $mismatch (param $out i32) (param $expected i32) (result i32)\n  (local $k i32) (local $bad i32)\n  (loop $next\n    (if (i32.ne (i32.load8_u (i32.add (local.get $out) (local.get $k))) (i32.load8_u (i32.add (local.get $expected) (local.get $k))))\n      (then (local.set $bad (i32.add (local.get $bad) (i32.const 1)))))\n    (local.set $k (i32.add (local.get $k) (i32.const 1)))\n    (br_if $next (i32.lt_u (local.get $k) (i32.const 18))))\n  (local.get $bad))\n";

    #[test]
    fn wasm_executes_every_supported_pair() {
        if !tools_available(&["wasmtime"]) {
            return;
        }

        let mut cases = Vec::new();
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
                    if !supported(Backend::Wasm, op, lane) {
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

        let guard = TempDir::new("lamina-wasm-simd");

        let dir = guard.path();
        for imm in [false, true] {
            let mut module = Module::new("simd");
            let mut data = Vec::new();
            for (i, c) in cases.iter().enumerate() {
                let name = format!("simd_{i}");
                let mut f = function(
                    &name,
                    MirType::Vector(VectorType::V128(c.lane)),
                    c.op,
                    c.index as i64,
                );
                let mut b = c.b;
                if imm {
                    immediate(&mut f, c.lane);
                    let bytes = match c.lane {
                        VectorLane::F32 => (-2.5_f32).to_bits() as u64,
                        VectorLane::F64 => (-2.5_f64).to_bits(),
                        _ => (-123_i64) as u64,
                    }
                    .to_le_bytes();
                    let width = scalar(c.lane).size_bytes(PointerWidth::Bits64);
                    if matches!(c.op, VectorOp::VSplat | VectorOp::VInsertLane) {
                        b[..width].copy_from_slice(&bytes[..width]);
                    }
                }
                module.add_function(f);
                let result = expected(c.op, c.lane, &c.a, &b, c.index, PointerWidth::Bits64);
                let mut guarded = [165; 18];
                guarded[1..1 + result.len()].copy_from_slice(&result);
                let mut block = [0u8; 128];
                block[1..17].copy_from_slice(&c.a);
                block[33..49].copy_from_slice(&c.b);
                block[64..82].copy_from_slice(&[165; 18]);
                block[96..114].copy_from_slice(&guarded);
                data.extend(block);
            }
            let mut wat =
                emit(&module, Backend::Wasm, RegallocStrategy::Incremental, true).expect("emit");
            wat.pop();
            wat.pop();
            for name in module.functions.keys() {
                wat.push_str(&format!("(export \"{name}\" (func ${name}))\n"));
            }
            wat.push_str(MISMATCH_FN);
            wat.push_str("(func (export \"check\") (result i32)\n  (local $bad i32)\n");
            for i in 0..cases.len() {
                wat.push_str(&format!(
                    "  (memory.init $rows (i32.const 0) (i32.const {}) (i32.const 128))\n  (call $simd_{i} (i64.const 0) (i64.const 32) (i64.const 64))\n  (local.set $bad (i32.add (local.get $bad) (call $mismatch (i32.const 64) (i32.const 96))))\n",
                    i * 128
                ));
            }
            wat.push_str("  (local.get $bad))\n");
            wat.push_str("(data $rows \"");
            for byte in &data {
                wat.push_str(&format!("\\{byte:02x}"));
            }
            wat.push_str("\")\n)");
            fs::write(dir.join("simd.wat"), wat).expect("write wat");
            let output = Command::new("wasmtime")
                .args(["run", "--invoke", "check"])
                .arg(dir.join("simd.wat"))
                .output()
                .expect("run wasmtime");
            assert!(output.status.success(), "{output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "0\n",
                "imm {imm}: some lanes did not match the expected result or guard bytes"
            );
        }
    }

    #[test]
    fn wasm_rejects_unsupported_pairs() {
        negative(Backend::Wasm, PointerWidth::Bits64);
    }
}
