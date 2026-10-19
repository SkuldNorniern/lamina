#![allow(dead_code, clippy::expect_used)]
use crate::{
    mir::{
        AddressMode, AddressSpace, Block, Function, Instruction, IntBinOp, MemoryAttrs, MirType,
        Module, Operand, Parameter, Register, ScalarType, Signature, VectorLane, VectorOp,
        VectorType, VirtualReg,
    },
    mir_codegen::{MirCodegenSettings, RegallocStrategy, generate_mir_to_target_with_settings},
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::{env, fs, process, process::Command, process::Stdio};

pub const LANES: [VectorLane; 6] = [
    VectorLane::I8,
    VectorLane::I16,
    VectorLane::I32,
    VectorLane::I64,
    VectorLane::F32,
    VectorLane::F64,
];
pub const BINARY: [VectorOp; 9] = [
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

pub fn scalar(lane: VectorLane) -> MirType {
    MirType::Scalar(match lane {
        VectorLane::I8 => ScalarType::I8,
        VectorLane::I16 => ScalarType::I16,
        VectorLane::I32 => ScalarType::I32,
        VectorLane::I64 => ScalarType::I64,
        VectorLane::F32 => ScalarType::F32,
        VectorLane::F64 => ScalarType::F64,
    })
}

pub fn reg(id: u32) -> Register {
    Register::Virtual(VirtualReg::gpr(id))
}
pub fn vector(id: u32) -> Register {
    Register::Virtual(VirtualReg::vec(id))
}
pub fn addr(id: u32) -> AddressMode {
    AddressMode::BaseOffset {
        address_space: AddressSpace::Generic,
        base: reg(id),
        offset: 1,
    }
}

pub fn function(name: &str, ty: MirType, op: VectorOp, index: i64) -> Function {
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
            ty: MirType::i64(),
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
            ty: MirType::i64(),
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

pub fn bits(bytes: &[u8]) -> u64 {
    let mut buf = [0; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

pub fn expected(
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

pub struct Case {
    pub op: VectorOp,
    pub lane: VectorLane,
    pub index: usize,
    pub a: [u8; 16],
    pub b: [u8; 16],
}

pub fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

pub fn tools_available(names: &[&str]) -> bool {
    for name in names {
        if !tool_available(name) {
            eprintln!("skipping test: {name} unavailable");
            return false;
        }
    }
    true
}

pub fn run(command: &mut Command) {
    let output = command.output().expect("execute tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Neon,
    Rvv,
    Wasm,
}

pub fn supported(backend: Backend, op: VectorOp, lane: VectorLane) -> bool {
    use VectorLane::*;
    use VectorOp::*;
    if matches!(lane, F32 | F64) && matches!(op, VShl | VLShr | VAShr) {
        return false;
    }
    !matches!(
        (backend, op, lane),
        (Backend::Neon, VMul, I64) | (Backend::Wasm, VMul, I8)
    )
}

pub fn emit(
    module: &Module,
    backend: Backend,
    strategy: RegallocStrategy,
    v: bool,
) -> Result<String, String> {
    let mut asm = Vec::new();
    let settings = MirCodegenSettings {
        regalloc: strategy,
        ..Default::default()
    };
    let result = if matches!(backend, Backend::Rvv) {
        use crate::mir_codegen::riscv::generate_mir_riscv_with_units_and_settings;
        use lamina_codegen::riscv::{RiscVTarget, Xlen};
        generate_mir_riscv_with_units_and_settings(
            module,
            &mut asm,
            RiscVTarget {
                v,
                ..RiscVTarget::general(Xlen::Rv64)
            },
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )
    } else {
        generate_mir_to_target_with_settings(
            module,
            &mut asm,
            if matches!(backend, Backend::Neon) {
                TargetArchitecture::Aarch64
            } else {
                TargetArchitecture::Wasm32
            },
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )
    };
    if result.is_err() {
        assert!(asm.is_empty(), "failed lowering emitted partial assembly");
    }
    result.map_err(|e| format!("{e:?}"))?;
    Ok(String::from_utf8(asm).expect("UTF8"))
}

pub fn ops() -> impl Iterator<Item = VectorOp> {
    BINARY.into_iter().chain([
        VectorOp::VSplat,
        VectorOp::VExtractLane,
        VectorOp::VInsertLane,
    ])
}

pub fn immediate(f: &mut Function, lane: VectorLane) {
    let value = match lane {
        VectorLane::F32 => Operand::imm_f32(-2.5),
        VectorLane::F64 => Operand::imm_f64(-2.5),
        _ => Operand::imm_i64(-123),
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

pub fn negative(backend: Backend) {
    for lane in LANES {
        for op in ops() {
            let mut m = Module::new("negative");
            let f = function("bad", MirType::Vector(VectorType::V128(lane)), op, 0);
            m.add_function(f.clone());
            if !supported(backend, op, lane) {
                let error = emit(&m, backend, RegallocStrategy::Incremental, true)
                    .expect_err("unsupported");
                assert!(
                    error.contains("UnsupportedFeature")
                        && error.contains(&format!("{op:?}"))
                        && error.contains(&format!("{lane:?}")),
                    "{error}"
                );
            }
            if matches!(backend, Backend::Rvv) {
                let error = emit(&m, backend, RegallocStrategy::Incremental, false)
                    .expect_err("requires V");
                assert!(error.contains("V extension"), "{error}");
            }
            for part in 0..3 {
                let mut f = function(
                    "bad",
                    MirType::Vector(VectorType::V256(lane)),
                    if part < 2 { VectorOp::VAdd } else { op },
                    0,
                );
                f.blocks[0].instructions.retain(|i| match part {
                    0 => matches!(i, Instruction::Load { .. } | Instruction::Ret { .. }),
                    1 => matches!(i, Instruction::Store { .. } | Instruction::Ret { .. }),
                    _ => matches!(i, Instruction::VectorOp { .. } | Instruction::Ret { .. }),
                });
                m.functions.insert("bad".into(), f);
                let error =
                    emit(&m, backend, RegallocStrategy::Incremental, true).expect_err("V256");
                assert!(error.contains("V256"), "{error}");
            }
            if matches!(op, VectorOp::VExtractLane | VectorOp::VInsertLane) {
                for index in [
                    Operand::reg(reg(0)),
                    Operand::imm_i64(-1),
                    Operand::imm_i64((16 / scalar(lane).size_bytes()) as i64),
                ] {
                    let mut f = f.clone();
                    for inst in &mut f.blocks[0].instructions {
                        if let Instruction::VectorOp { operands, .. } = inst {
                            *operands.last_mut().expect("index") = index.clone();
                        }
                    }
                    m.functions.insert("bad".into(), f);
                    let error =
                        emit(&m, backend, RegallocStrategy::Incremental, true).expect_err("index");
                    assert!(error.contains("lane index"), "{error}");
                }
            }
        }
    }
}

pub fn native(backend: Backend, mnemonic: impl Fn(VectorOp, VectorLane, usize) -> String) {
    if !tools_available(&["clang", "llvm-objdump-18"]) {
        return;
    }
    let dir = env::temp_dir().join(format!("lamina-{backend:?}-{}", process::id()));
    fs::create_dir_all(&dir).expect("directory");
    for strategy in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let mut module = Module::new("simd");
        for lane in LANES {
            for op in ops().filter(|&op| supported(backend, op, lane)) {
                let count = 16 / scalar(lane).size_bytes();
                for index in 0..count {
                    for gpr in [false, true] {
                        for imm in [false, true] {
                            let name = format!(
                                "simd_{lane:?}_{op:?}_{index}_{imm}{}",
                                if gpr { "_gpr" } else { "" }
                            );
                            let mut f = function(
                                &name,
                                MirType::Vector(VectorType::V128(lane)),
                                op,
                                index as i64,
                            );
                            if imm {
                                immediate(&mut f, lane);
                            }
                            let mut single = Module::new("single");
                            single.add_function(f.clone());
                            let asm = emit(&single, backend, strategy, true).expect("emit");
                            if matches!(backend, Backend::Rvv) {
                                let bits = scalar(lane).size_bytes() * 8;
                                assert!(
                                    asm.contains(&format!(
                                        "vsetivli zero, {}, e{bits}, m1, tu, mu",
                                        128 / bits
                                    )),
                                    "{asm}"
                                );
                                assert!(
                                    asm.contains(&format!("vle{bits}.v"))
                                        && asm.contains(&format!("vse{bits}.v")),
                                    "{asm}"
                                );
                            }
                            if matches!(backend, Backend::Neon) && op == VectorOp::VExtractLane {
                                assert!(asm.contains("umov "), "{asm}");
                            }
                            let want = mnemonic(op, lane, index);
                            assert!(
                                asm.contains(&want),
                                "{backend:?} {strategy:?} {op:?} {lane:?}: expected {want}: {asm}"
                            );
                            assert!(asm.contains(if matches!(backend, Backend::Neon) {
                                "ldr q0"
                            } else {
                                "vle"
                            }));
                            module.add_function(f);
                        }
                    }
                }
            }
        }
        fs::write(
            dir.join("simd.s"),
            emit(&module, backend, strategy, true).expect("emit"),
        )
        .expect("write");
        let mut command = Command::new("clang");
        command.arg(if matches!(backend, Backend::Neon) {
            "--target=aarch64-linux-gnu"
        } else {
            "--target=riscv64-linux-gnu"
        });
        if matches!(backend, Backend::Rvv) {
            command.arg("-march=rv64gcv");
        }
        run(command
            .arg("-c")
            .arg(dir.join("simd.s"))
            .arg("-o")
            .arg(dir.join("simd.o")));
        let output = Command::new("llvm-objdump-18")
            .arg("-d")
            .arg(dir.join("simd.o"))
            .output()
            .expect("disassemble");
        assert!(output.status.success());
        let disasm = String::from_utf8(output.stdout).expect("UTF8");
        assert!(disasm.contains(if matches!(backend, Backend::Neon) {
            "v0.16b"
        } else {
            "vsetivli"
        }));
        assert!(!disasm.contains("<unknown>"));
    }
    fs::remove_dir_all(dir).expect("cleanup");
}
