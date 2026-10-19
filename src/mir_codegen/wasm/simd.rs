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
                || (float && matches!(op, VShl | VLShr | VAShr))
            {
                return Err(unsupported(format!("WASM unsupported {op:?} for {lane:?}")));
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
                        "    i64x2.extract_lane 0\n    i64.const {bits}\n    i64.ge_u\n    if (result v128)"
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
                    writeln!(w, "    i64x2.extract_lane 0\n    i32.wrap_i64")?;
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
    use crate::mir_codegen::simd_support::*;
    use crate::{
        mir::{MirType, Module, VectorLane, VectorOp, VectorType},
        mir_codegen::RegallocStrategy,
    };
    use std::{
        array, env, fs,
        process::{self, Command},
    };

    #[test]
    fn wasm_executes_every_supported_pair() {
        if !tool_available("uv") {
            eprintln!("skipping wasm execution: uv not installed");
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

        let dir = env::temp_dir().join(format!("lamina-wasm-simd-{}", process::id()));
        fs::create_dir_all(&dir).expect("directory");
        for imm in [false, true] {
            let mut module = Module::new("simd");
            let mut data = Vec::new();
            for (i, c) in cases.iter().enumerate() {
                let name = format!("simd_{i}{}", if i % 2 == 1 { "_gpr" } else { "" });
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
                    let width = scalar(c.lane).size_bytes();
                    if matches!(c.op, VectorOp::VSplat | VectorOp::VInsertLane) {
                        b[..width].copy_from_slice(&bytes[..width]);
                    }
                }
                module.add_function(f);
                let result = expected(c.op, c.lane, &c.a, &b, c.index);
                data.extend(c.a);
                data.extend(c.b);
                let mut guarded = [165; 18];
                guarded[1..1 + result.len()].copy_from_slice(&result);
                data.extend(guarded);
            }
            let mut wat =
                emit(&module, Backend::Wasm, RegallocStrategy::Incremental, true).expect("emit");
            wat.pop();
            wat.pop();
            for name in module.functions.keys() {
                wat.push_str(&format!("(export \"{name}\" (func ${name}))\n"));
            }
            wat.push(')');
            fs::write(dir.join("simd.wat"), wat).expect("write wat");
            fs::write(dir.join("data"), data).expect("write data");
            fs::write(dir.join("run.py"), r#"
import pathlib, wasmtime
p = pathlib.Path(__file__).parent
engine = wasmtime.Engine()
wasm = wasmtime.wat2wasm((p/'simd.wat').read_text())
(p/'simd.wasm').write_bytes(wasm)
module = wasmtime.Module(engine, (p/'simd.wasm').read_bytes())
store = wasmtime.Store(engine)
imports = [wasmtime.Func(store, i.type, lambda *args: 0 if len(args) else 0) for i in module.imports]
instance = wasmtime.Instance(store, module, imports)
exports = instance.exports(store)
memory = exports['memory']
data = (p/'data').read_bytes()
for i in range(len(data)//50):
    row = data[i*50:(i+1)*50]
    memory.write(store, b'\x00'+row[:16]+b'\x00', 0)
    memory.write(store, b'\x00'+row[16:32]+b'\x00', 32)
    memory.write(store, bytes([165])*18, 64)
    name = f'simd_{i}' + ('_gpr' if i%2 else '')
    exports[name](store, 0, 32, 64)
    got = bytes(memory.read(store,64,82))
    assert got == row[32:], (i, name, got.hex(), row[32:].hex())
"#).expect("write runner");
            run(Command::new("uv")
                .args(["run", "--with", "wasmtime", "python"])
                .arg(dir.join("run.py")));
        }
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn wasm_rejects_unsupported_pairs() {
        negative(Backend::Wasm);
    }
}
