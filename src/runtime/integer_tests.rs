use super::interpret_mir_function;
use crate::{
    mir::{
        Block, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Module, Operand,
        Register, Signature, VirtualReg,
    },
    mir_codegen::{
        MirCodegenSettings, RegallocStrategy, generate_mir_to_target_with_settings,
        wasm::generate_mir_wasm,
    },
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::{
    env::temp_dir,
    error::Error,
    fs,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{self, Command},
};

type TestResult = Result<(), Box<dyn Error>>;

fn reg(id: u32) -> Register {
    Register::Virtual(VirtualReg::gpr(id))
}
fn imm(value: i64) -> Operand {
    Operand::Immediate(Immediate::I64(value))
}

fn operation(
    block: &mut Block,
    id: &mut u32,
    op: IntBinOp,
    ty: MirType,
    lhs: i64,
    rhs: i64,
    constants: bool,
) -> Register {
    let left = reg(*id);
    let right = reg(*id + 1);
    let dst = reg(*id + 2);
    *id += 3;
    let (lhs, rhs) = if constants {
        (imm(lhs), imm(rhs))
    } else {
        block.push(Instruction::Copy {
            ty,
            dst: left.clone(),
            src: imm(lhs),
        });
        block.push(Instruction::Copy {
            ty,
            dst: right.clone(),
            src: imm(rhs),
        });
        (Operand::Register(left), Operand::Register(right))
    };
    block.push(Instruction::IntBinary {
        op,
        ty,
        dst: dst.clone(),
        lhs,
        rhs,
    });
    dst
}

fn module(block: Block) -> Module {
    let mut function = Function::new(Signature::new("main").with_return(MirType::i64()));
    function.blocks.push(block);
    let mut module = Module::new("integer_semantics");
    module.add_function(function);
    module
}

fn available(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

struct Native {
    arch: TargetArchitecture,
    triple: &'static str,
    emulator: Option<&'static str>,
    start: &'static str,
}

fn targets(include_unavailable: bool) -> Vec<Native> {
    let mut targets = vec![Native {
        arch: TargetArchitecture::X86_64,
        triple: "x86_64-linux-gnu",
        emulator: None,
        start: ".text\n.globl _start\n_start:\n call main\n mov %rax, %rdi\n mov $60, %rax\n syscall\n",
    }];
    for target in [
        Native {
            arch: TargetArchitecture::Aarch64,
            triple: "aarch64-linux-gnu",
            emulator: Some("qemu-aarch64-static"),
            start: ".text\n.globl _start\n_start:\n bl main\n mov x8, #93\n svc #0\n",
        },
        Native {
            arch: TargetArchitecture::Riscv64,
            triple: "riscv64-linux-gnu",
            emulator: Some("qemu-riscv64-static"),
            start: ".text\n.globl _start\n_start:\n call main\n li a7, 93\n ecall\n",
        },
        Native {
            arch: TargetArchitecture::PowerPC64,
            triple: "powerpc64le-linux-gnu",
            emulator: Some("qemu-ppc64le-static"),
            start: ".text\n.globl _start\n_start:\n bl main\n li 0, 1\n sc\n",
        },
    ] {
        if include_unavailable
            || (available("clang")
                && available("ld.lld-18")
                && target.emulator.is_some_and(available))
        {
            targets.push(target);
        }
    }
    targets
}

fn emit(
    module: &Module,
    target: &Native,
    allocator: RegallocStrategy,
) -> Result<String, Box<dyn Error>> {
    let mut bytes = Vec::new();
    generate_mir_to_target_with_settings(
        module,
        &mut bytes,
        target.arch,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings {
            regalloc: allocator,
            ..Default::default()
        },
    )?;
    Ok(String::from_utf8(bytes)?)
}

fn run(
    module: &Module,
    target: &Native,
    allocator: RegallocStrategy,
    directory: &Path,
) -> Result<process::Output, Box<dyn Error>> {
    let source = directory.join("test.s");
    let object = directory.join("test.o");
    let binary = directory.join("test");
    fs::write(
        &source,
        format!("{}\n{}", emit(module, target, allocator)?, target.start),
    )?;
    if target.emulator.is_none() {
        let output = Command::new("cc")
            .args(["-nostdlib", "-no-pie"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    } else {
        let output = Command::new("clang")
            .args(["-target", target.triple, "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .output()?;
        assert!(
            output.status.success(),
            "{}: {}",
            target.triple,
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new("ld.lld-18")
            .args(["-e", "_start", "-o"])
            .arg(&binary)
            .arg(&object)
            .output()?;
        assert!(
            output.status.success(),
            "{}: {}",
            target.triple,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let mut command = if available("prlimit") {
        let mut command = Command::new("prlimit");
        command.args(["--core=1:1", "--", "timeout", "10"]);
        command
    } else {
        let mut command = Command::new("timeout");
        command.arg("10");
        command
    };
    if let Some(emulator) = target.emulator {
        command.arg(emulator);
    }
    Ok(command.current_dir(directory).arg(binary).output()?)
}

fn check_wasm(module: &Module, directory: &Path, traps: bool) -> TestResult {
    if !available("wat2wasm") || !available("node") {
        return Ok(());
    }
    let mut bytes = Vec::new();
    generate_mir_wasm(module, &mut bytes, TargetOperatingSystem::Linux)?;
    let wat = directory.join("test.wat");
    let wasm = directory.join("test.wasm");
    fs::write(&wat, bytes)?;
    let output = Command::new("wat2wasm")
        .arg(wat)
        .arg("-o")
        .arg(&wasm)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new("node")
        .args([
            "-e",
            r#"
const fs = require('fs');
const module = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const imports = {};
for (const entry of WebAssembly.Module.imports(module)) {
    (imports[entry.module] ??= {})[entry.name] = () => 0n;
}
const instance = new WebAssembly.Instance(module, imports);
const traps = process.argv[2] === 'true';
try {
    const value = instance.exports.main();
    if (traps || value !== 0n) process.exit(1);
} catch (error) {
    if (!traps || !(error instanceof WebAssembly.RuntimeError)) throw error;
}
"#,
        ])
        .arg(&wasm)
        .arg(traps.to_string())
        .output()?;
    assert!(output.status.success(), "WASM traps={traps}: {output:?}");
    Ok(())
}

fn directory(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path = temp_dir().join(format!("lamina-integer-{name}-{}", process::id()));
    fs::create_dir_all(&path)?;
    Ok(path)
}

#[test]
fn wrapping_arithmetic_and_masked_shifts_match_native_execution() -> TestResult {
    for ty in [
        MirType::i8(),
        MirType::i16(),
        MirType::i32(),
        MirType::i64(),
    ] {
        let width = ty.size_bytes() * 8;
        let min = -(1i128 << (width - 1));
        let max = -min - 1;
        let mut cases = vec![
            (IntBinOp::Add, max as i64, 1, min as i64),
            (IntBinOp::Add, max as i64, 2, (min + 1) as i64),
            (IntBinOp::Add, -1, 1, 0),
            (IntBinOp::Sub, 0, 1, -1),
            (IntBinOp::Mul, -1, 2, -2),
            (IntBinOp::Mul, max as i64, 2, -2),
            (IntBinOp::Sub, min as i64, 1, max as i64),
            (IntBinOp::Sub, min as i64, 2, (max - 1) as i64),
            (IntBinOp::Mul, min as i64, 2, 0),
            (IntBinOp::Mul, min as i64, 3, min as i64),
            (IntBinOp::SDiv, min as i64, 1, min as i64),
            (IntBinOp::SRem, min as i64, 1, 0),
            (IntBinOp::SDiv, -17, -1, 17),
            (IntBinOp::SRem, -17, -1, 0),
            (IntBinOp::UDiv, -1, 2, max as i64),
            (IntBinOp::URem, -1, 2, 1),
        ];
        for count in [width as i64, width as i64 + 1, width as i64 * 1024 + 3, -1] {
            let shift = count as usize & (width - 1);
            let mask = (1u128 << width) - 1;
            let bits = ((min as u128) & mask) >> shift;
            cases.extend([
                (
                    IntBinOp::Shl,
                    1,
                    count,
                    ty.signed_integer((1u64 << shift) as i64),
                ),
                (
                    IntBinOp::LShr,
                    min as i64,
                    count,
                    ty.signed_integer(bits as i64),
                ),
                (IntBinOp::AShr, min as i64, count, (min >> shift) as i64),
            ]);
        }
        for constants in [false, true] {
            let mut block = Block::new("entry");
            let mut id = 0;
            let mut failures = imm(0);
            for &(op, lhs, rhs, expected) in &cases {
                let dst = operation(&mut block, &mut id, op, ty, lhs, rhs, constants);
                let wide = reg(id);
                let failed = reg(id + 1);
                let accumulated = reg(id + 2);
                id += 3;
                if ty == MirType::i64() {
                    block.push(Instruction::Copy {
                        ty,
                        dst: wide.clone(),
                        src: Operand::Register(dst),
                    });
                } else {
                    block.push(Instruction::IntConvert {
                        from: ty,
                        to: MirType::i64(),
                        signed: true,
                        dst: wide.clone(),
                        src: Operand::Register(dst),
                    });
                }

                block.push(Instruction::IntCmp {
                    op: IntCmpOp::Ne,
                    ty: MirType::i64(),
                    dst: failed.clone(),
                    lhs: Operand::Register(wide),
                    rhs: imm(expected),
                });
                block.push(Instruction::IntBinary {
                    op: IntBinOp::Or,
                    ty: MirType::i64(),
                    dst: accumulated.clone(),
                    lhs: failures,
                    rhs: Operand::Register(failed),
                });
                failures = Operand::Register(accumulated);
            }
            block.push(Instruction::Ret {
                value: Some(failures),
            });
            let module = module(block);
            assert_eq!(
                interpret_mir_function(&module.functions["main"], &[])?,
                Some(0)
            );
            if available("cc") {
                let directory = directory("values")?;
                for target in targets(false) {
                    for allocator in [
                        RegallocStrategy::Incremental,
                        RegallocStrategy::LinearScanGlobal,
                        RegallocStrategy::GraphColorGlobal,
                    ] {
                        let output = run(&module, &target, allocator, &directory)?;
                        assert_eq!(
                            output.status.code(),
                            Some(0),
                            "{} {allocator:?}: {output:?}",
                            target.triple
                        );
                    }
                }
                check_wasm(&module, &directory, false)?;
                fs::remove_dir_all(directory)?;
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_division_traps_in_interpreter_and_native_execution() -> TestResult {
    let native = available("cc");
    let directory = directory("traps")?;
    let targets = targets(false);
    for ty in [
        MirType::i8(),
        MirType::i16(),
        MirType::i32(),
        MirType::i64(),
    ] {
        let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
        for (op, lhs, rhs) in [
            (IntBinOp::SDiv, 1, 0),
            (IntBinOp::SRem, 1, 0),
            (IntBinOp::UDiv, 1, 0),
            (IntBinOp::URem, 1, 0),
            (IntBinOp::SDiv, minimum, -1),
            (IntBinOp::SRem, minimum, -1),
        ] {
            for constants in [false, true] {
                let mut block = Block::new("entry");
                operation(&mut block, &mut 0, op, ty, lhs, rhs, constants);
                block.push(Instruction::Ret {
                    value: Some(imm(0)),
                });
                let original = module(block);
                for optimize in [false, true] {
                    let mut module = original.clone();
                    if optimize {
                        use crate::mir::transform::{
                            ConstantFolding, DeadCodeElimination, Peephole, StrengthReduction,
                            Transform,
                        };
                        let function = module.functions.get_mut("main").ok_or("missing main")?;
                        ConstantFolding
                            .apply(function)
                            .map_err(|error| format!("{error:?}"))?;
                        Peephole
                            .apply(function)
                            .map_err(|error| format!("{error:?}"))?;
                        StrengthReduction
                            .apply(function)
                            .map_err(|error| format!("{error:?}"))?;
                        DeadCodeElimination
                            .apply(function)
                            .map_err(|error| format!("{error:?}"))?;
                    }
                    let result = interpret_mir_function(&module.functions["main"], &[]);
                    assert!(result.is_err(), "{ty} {op} {lhs} {rhs}");
                    check_wasm(&module, &directory, true)?;
                    if native {
                        for target in &targets {
                            let output =
                                run(&module, target, RegallocStrategy::Incremental, &directory)?;
                            let signal = output.status.signal();
                            assert!(
                                matches!(signal, Some(4 | 5 | 8)),
                                "{} {ty} {op} {lhs} {rhs} constants={constants} optimize={optimize}: {output:?}",
                                target.triple
                            );
                            if target.arch == TargetArchitecture::X86_64 && ty == MirType::i64() {
                                assert_eq!(signal, Some(8));
                            }
                        }
                    }
                }
            }
        }
    }
    fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn division_guards_skip_proven_safe_operands() -> TestResult {
    for target in targets(true) {
        if target.arch == TargetArchitecture::X86_64 {
            continue;
        }
        for (left, right, expected) in [
            (None, Some(2), 0),
            (Some(7), Some(-1), 0),
            (None, Some(-1), 1),
            (Some(7), None, 1),
            (None, None, 2),
        ] {
            let mut block = Block::new("entry");
            for (id, value) in [(0, 7), (1, 2)] {
                block.push(Instruction::Copy {
                    ty: MirType::i64(),
                    dst: reg(id),
                    src: imm(value),
                });
            }
            block.push(Instruction::IntBinary {
                op: IntBinOp::SDiv,
                ty: MirType::i64(),
                dst: reg(2),
                lhs: left.map_or_else(|| Operand::Register(reg(0)), imm),
                rhs: right.map_or_else(|| Operand::Register(reg(1)), imm),
            });
            block.push(Instruction::Ret {
                value: Some(Operand::Register(reg(2))),
            });
            let assembly = emit(&module(block), &target, RegallocStrategy::Incremental)?;
            let count = assembly
                .lines()
                .filter(|line| {
                    let line = line.trim();
                    line == "udf #0"
                        || line == ".word 0"
                        || line.starts_with("tdi ")
                        || line.starts_with("td ")
                })
                .count();
            assert_eq!(
                count, expected,
                "{} {left:?} {right:?}: {assembly}",
                target.triple
            );
        }
    }
    Ok(())
}

#[test]
fn division_traps_stay_inside_unentered_loops() -> TestResult {
    use crate::mir::transform::{LoopInvariantCodeMotion, Transform};
    let mut entry = Block::new("entry");
    entry.push(Instruction::Copy {
        ty: MirType::i64(),
        dst: reg(0),
        src: imm(0),
    });
    entry.push(Instruction::Jmp {
        target: "header".into(),
    });
    let mut header = Block::new("header");
    header.push(Instruction::Br {
        cond: reg(0),
        true_target: "body".into(),
        false_target: "exit".into(),
    });
    let mut body = Block::new("body");
    operation(
        &mut body,
        &mut 1,
        IntBinOp::SDiv,
        MirType::i64(),
        1,
        0,
        true,
    );
    body.push(Instruction::Jmp {
        target: "header".into(),
    });
    let mut exit = Block::new("exit");
    exit.push(Instruction::Ret {
        value: Some(imm(0)),
    });
    let mut function = Function::new(Signature::new("main").with_return(MirType::i64()));
    function.blocks = vec![entry, header, body, exit];
    assert_eq!(interpret_mir_function(&function, &[])?, Some(0));
    LoopInvariantCodeMotion
        .apply(&mut function)
        .map_err(|error| format!("{error:?}"))?;
    assert_eq!(interpret_mir_function(&function, &[])?, Some(0));
    Ok(())
}
