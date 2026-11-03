use super::interpret_mir_function;
use crate::mir_codegen::simd_support::{skip_test, tools_available};
use crate::test_support::TempDir;
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
use lamina_mir::ScalarType;
use lamina_platform::{PointerWidth, Target, TargetArchitecture, TargetOperatingSystem};
use std::{
    error::Error,
    fs,
    os::unix::process::ExitStatusExt,
    path::Path,
    process::{self, Command},
};

pub(super) type TestResult = Result<(), Box<dyn Error>>;

pub(super) fn reg(id: u32) -> Register {
    Register::Virtual(VirtualReg::gpr(id))
}
pub(super) fn imm(value: i64) -> Operand {
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

pub(super) fn module(block: Block) -> Module {
    module_returning(block, MirType::Scalar(ScalarType::I64))
}

pub(super) fn module_returning(mut block: Block, ret: MirType) -> Module {
    for instruction in &mut block.instructions {
        instruction.normalize_integer_immediates(Some(ret));
    }
    let mut function = Function::new(Signature::new("main").with_return(ret));
    function.blocks.push(block);
    let mut module = Module::new("integer_semantics");
    module.add_function(function);
    module
}

pub(super) fn available(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

pub(super) struct Native {
    pub(super) arch: TargetArchitecture,
    pub(super) triple: &'static str,
    pub(super) emulator: Option<&'static str>,
    pub(super) cc_flags: &'static [&'static str],
    pub(super) widest_integer_bytes: usize,
    pub(super) atomics: bool,
    pub(super) global_regalloc: bool,
    pub(super) start: &'static str,
}

impl Native {
    pub(super) fn lowers(&self, ty: MirType) -> bool {
        ty.size_bytes(PointerWidth::Bits64) <= self.widest_integer_bytes
    }

    pub(super) fn allocators(&self) -> &'static [RegallocStrategy] {
        if self.global_regalloc {
            &[
                RegallocStrategy::Incremental,
                RegallocStrategy::LinearScanGlobal,
                RegallocStrategy::GraphColorGlobal,
            ]
        } else {
            &[RegallocStrategy::Incremental]
        }
    }

    pub(super) fn accumulator(&self) -> MirType {
        match self.widest_integer_bytes {
            4 => MirType::Scalar(ScalarType::I32),
            _ => MirType::Scalar(ScalarType::I64),
        }
    }
}

/// The widths a value check accumulates its result in: one per distinct widest
/// integer among the native targets, skipping any narrower than the type tested.
fn accumulator_types(ty: MirType) -> Vec<MirType> {
    let mut widths: Vec<MirType> = Vec::new();
    for target in targets(false) {
        let accumulator = target.accumulator();
        if accumulator.size_bytes(PointerWidth::Bits64) >= ty.size_bytes(PointerWidth::Bits64)
            && !widths.contains(&accumulator)
        {
            widths.push(accumulator);
        }
    }
    widths
}

fn multilib_32_bit() -> bool {
    Command::new("cc")
        .args(["-m32", "-xc", "-c", "/dev/null", "-o", "/dev/null"])
        .output()
        .is_ok_and(|out| out.status.success())
}

pub(super) fn targets(include_unavailable: bool) -> Vec<Native> {
    let mut targets = vec![Native {
        arch: TargetArchitecture::X86_64,
        triple: "x86_64-linux-gnu",
        emulator: None,
        cc_flags: &[],
        widest_integer_bytes: 8,
        atomics: true,
        global_regalloc: true,
        start: ".text\n.globl _start\n_start:\n call main\n mov %rax, %rdi\n mov $60, %rax\n syscall\n",
    }];
    if !include_unavailable && !(available("cc") && multilib_32_bit()) {
        skip_test("i686 native execution needs cc with 32-bit multilib");
    }
    if include_unavailable || (available("cc") && multilib_32_bit()) {
        targets.push(Native {
            arch: TargetArchitecture::I686,
            triple: "i686-linux-gnu",
            emulator: None,
            cc_flags: &["-m32"],
            widest_integer_bytes: 4,
            atomics: false,
            global_regalloc: false,
            start: ".text\n.globl _start\n_start:\n call main\n movl %eax, %ebx\n movl $1, %eax\n int $0x80\n",
        });
    }
    for target in [
        Native {
            arch: TargetArchitecture::Aarch64,
            triple: "aarch64-linux-gnu",
            emulator: Some("qemu-aarch64-static"),
            cc_flags: &[],
            widest_integer_bytes: 8,
            atomics: true,
            global_regalloc: true,
            start: ".text\n.globl _start\n_start:\n bl main\n mov x8, #93\n svc #0\n",
        },
        Native {
            arch: TargetArchitecture::Riscv64,
            triple: "riscv64-linux-gnu",
            emulator: Some("qemu-riscv64-static"),
            cc_flags: &[],
            widest_integer_bytes: 8,
            atomics: true,
            global_regalloc: true,
            start: ".text\n.globl _start\n_start:\n call main\n li a7, 93\n ecall\n",
        },
        Native {
            arch: TargetArchitecture::PowerPC64,
            triple: "powerpc64le-linux-gnu",
            emulator: Some("qemu-ppc64le-static"),
            cc_flags: &[],
            widest_integer_bytes: 8,
            atomics: true,
            global_regalloc: true,
            start: ".text\n.globl _start\n_start:\n bl main\n li 0, 1\n sc\n",
        },
    ] {
        let usable =
            available("clang") && available("ld.lld-18") && target.emulator.is_some_and(available);
        if !include_unavailable && !usable {
            skip_test(&format!(
                "{} native execution needs clang, ld.lld-18 and {}",
                target.triple,
                target.emulator.unwrap_or("an emulator")
            ));
        }
        if include_unavailable || usable {
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
        &Target::new(target.arch, TargetOperatingSystem::Linux),
        1,
        &MirCodegenSettings {
            regalloc: allocator,
            ..Default::default()
        },
    )?;
    Ok(String::from_utf8(bytes)?)
}

pub(super) fn run(
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
            .args(target.cc_flags)
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
    if !tools_available(&["wasmtime"]) {
        return Ok(());
    }
    let mut wat = Vec::new();
    generate_mir_wasm(
        module,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )?;
    let wat_path = directory.join("test.wat");
    fs::write(&wat_path, wat)?;
    let output = Command::new("wasmtime")
        .args(["run", "--invoke", "main"])
        .arg(&wat_path)
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if traps {
        assert!(
            !output.status.success() && stderr.contains("wasm trap"),
            "WASM did not trap: {output:?}"
        );
    } else {
        assert!(output.status.success(), "WASM traps=false: {stderr}");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "0\n");
    }
    Ok(())
}

pub(super) fn directory(name: &str) -> TempDir {
    TempDir::new(&format!("lamina-integer-{name}"))
}

#[test]
fn wrapping_arithmetic_and_masked_shifts_match_native_execution() -> TestResult {
    for ty in [
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::I16),
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I64),
    ] {
        let width = ty.size_bytes(PointerWidth::Bits64) * 8;
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
                    ty.signed_integer((1u64 << shift) as i64, PointerWidth::Bits64),
                ),
                (
                    IntBinOp::LShr,
                    min as i64,
                    count,
                    ty.signed_integer(bits as i64, PointerWidth::Bits64),
                ),
                (IntBinOp::AShr, min as i64, count, (min >> shift) as i64),
            ]);
        }
        for constants in [false, true] {
            for accumulator in accumulator_types(ty) {
                let mut block = Block::new("entry");
                let mut id = 0;
                let mut failures = imm(0);
                for &(op, lhs, rhs, expected) in &cases {
                    let dst = operation(&mut block, &mut id, op, ty, lhs, rhs, constants);
                    let wide = reg(id);
                    let failed = reg(id + 1);
                    let accumulated = reg(id + 2);
                    id += 3;
                    if ty == accumulator {
                        block.push(Instruction::Copy {
                            ty,
                            dst: wide.clone(),
                            src: Operand::Register(dst),
                        });
                    } else {
                        block.push(Instruction::IntConvert {
                            from: ty,
                            to: accumulator,
                            signed: true,
                            dst: wide.clone(),
                            src: Operand::Register(dst),
                        });
                    }

                    block.push(Instruction::IntCmp {
                        op: IntCmpOp::Ne,
                        ty: accumulator,
                        dst: failed.clone(),
                        lhs: Operand::Register(wide),
                        rhs: imm(expected),
                    });
                    let flag = reg(id);
                    id += 1;
                    block.push(Instruction::IntConvert {
                        from: MirType::Scalar(ScalarType::I1),
                        to: accumulator,
                        signed: false,
                        dst: flag.clone(),
                        src: Operand::Register(failed),
                    });
                    block.push(Instruction::IntBinary {
                        op: IntBinOp::Or,
                        ty: accumulator,
                        dst: accumulated.clone(),
                        lhs: failures,
                        rhs: Operand::Register(flag),
                    });
                    failures = Operand::Register(accumulated);
                }
                block.push(Instruction::Ret {
                    value: Some(failures),
                });
                let module = module_returning(block, accumulator);
                assert_eq!(
                    interpret_mir_function(&module.functions["main"], &[])?,
                    Some(0)
                );
                if available("cc") {
                    let guard = directory("values");
                    let directory = guard.path();
                    for target in targets(false)
                        .iter()
                        .filter(|target| target.accumulator() == accumulator && target.lowers(ty))
                    {
                        for &allocator in target.allocators() {
                            let output = run(&module, target, allocator, directory)?;
                            assert_eq!(
                                output.status.code(),
                                Some(0),
                                "{} {allocator:?}: {output:?}",
                                target.triple
                            );
                        }
                    }
                    if accumulator == MirType::Scalar(ScalarType::I64) {
                        check_wasm(&module, directory, false)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_division_traps_in_interpreter_and_native_execution() -> TestResult {
    let native = available("cc");
    let guard = directory("traps");
    let directory = guard.path();
    let targets = targets(false);
    for ty in [
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::I16),
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I64),
    ] {
        let minimum = ty.signed_integer(
            1i64 << (ty.size_bytes(PointerWidth::Bits64) * 8 - 1),
            PointerWidth::Bits64,
        );
        for (op, lhs, rhs) in [
            (IntBinOp::SDiv, 1, 0),
            (IntBinOp::SRem, 1, 0),
            (IntBinOp::UDiv, 1, 0),
            (IntBinOp::URem, 1, 0),
            (IntBinOp::SDiv, minimum, -1),
            (IntBinOp::SRem, minimum, -1),
        ] {
            for constants in [false, true] {
                for accumulator in accumulator_types(ty) {
                    let mut block = Block::new("entry");
                    operation(&mut block, &mut 0, op, ty, lhs, rhs, constants);
                    block.push(Instruction::Ret {
                        value: Some(imm(0)),
                    });
                    let original = module_returning(block, accumulator);
                    for optimize in [false, true] {
                        let mut module = original.clone();
                        if optimize {
                            use crate::mir::transform::{
                                ConstantFolding, DeadCodeElimination, Peephole, StrengthReduction,
                                Transform,
                            };
                            let function =
                                module.functions.get_mut("main").ok_or("missing main")?;
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
                        if accumulator == MirType::Scalar(ScalarType::I64) {
                            check_wasm(&module, directory, true)?;
                        }
                        if native {
                            for target in targets.iter().filter(|target| {
                                target.accumulator() == accumulator && target.lowers(ty)
                            }) {
                                let output =
                                    run(&module, target, RegallocStrategy::Incremental, directory)?;
                                let signal = output.status.signal();
                                assert!(
                                    matches!(signal, Some(4 | 5 | 8)),
                                    "{} {ty} {op} {lhs} {rhs} constants={constants} optimize={optimize}: {output:?}",
                                    target.triple
                                );
                                if target.arch == TargetArchitecture::X86_64
                                    && ty == MirType::Scalar(ScalarType::I64)
                                {
                                    assert_eq!(signal, Some(8));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn division_guards_skip_proven_safe_operands() -> TestResult {
    for target in targets(true) {
        if matches!(
            target.arch,
            TargetArchitecture::X86_64 | TargetArchitecture::I686
        ) {
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
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: reg(id),
                    src: imm(value),
                });
            }
            block.push(Instruction::IntBinary {
                op: IntBinOp::SDiv,
                ty: MirType::Scalar(ScalarType::I64),
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
        ty: MirType::Scalar(ScalarType::I64),
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
        MirType::Scalar(ScalarType::I64),
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
    let mut function =
        Function::new(Signature::new("main").with_return(MirType::Scalar(ScalarType::I64)));
    function.blocks = vec![entry, header, body, exit];
    assert_eq!(interpret_mir_function(&function, &[])?, Some(0));
    LoopInvariantCodeMotion
        .apply(&mut function)
        .map_err(|error| format!("{error:?}"))?;
    assert_eq!(interpret_mir_function(&function, &[])?, Some(0));
    Ok(())
}

#[test]
fn narrow_comparisons_and_conversions_match_native_execution() -> TestResult {
    let guard = directory("comparisons");
    let directory = guard.path();
    for ty in [
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::I16),
        MirType::Scalar(ScalarType::I32),
    ] {
        let width = ty.size_bytes(PointerWidth::Bits64) * 8;
        let modulus = 1i64 << width;
        let sign = modulus / 2;
        let values = [0, 1, sign - 1, sign, modulus - 1, -1];
        for converted in [false, true] {
            let mut block = Block::new("entry");
            let mut id = 0;
            let mut failures = imm(0);
            for lhs in values {
                for rhs in values {
                    for op in [
                        IntCmpOp::Eq,
                        IntCmpOp::Ne,
                        IntCmpOp::SLt,
                        IntCmpOp::SLe,
                        IntCmpOp::SGt,
                        IntCmpOp::SGe,
                        IntCmpOp::ULt,
                        IntCmpOp::ULe,
                        IntCmpOp::UGt,
                        IntCmpOp::UGe,
                    ] {
                        let unsigned = |v: i64| v & (modulus - 1);
                        let signed = |v| {
                            let v = unsigned(v);
                            if v >= sign { v - modulus } else { v }
                        };
                        let expected = match op {
                            IntCmpOp::Eq => unsigned(lhs) == unsigned(rhs),
                            IntCmpOp::Ne => unsigned(lhs) != unsigned(rhs),
                            IntCmpOp::SLt => signed(lhs) < signed(rhs),
                            IntCmpOp::SLe => signed(lhs) <= signed(rhs),
                            IntCmpOp::SGt => signed(lhs) > signed(rhs),
                            IntCmpOp::SGe => signed(lhs) >= signed(rhs),
                            IntCmpOp::ULt => unsigned(lhs) < unsigned(rhs),
                            IntCmpOp::ULe => unsigned(lhs) <= unsigned(rhs),
                            IntCmpOp::UGt => unsigned(lhs) > unsigned(rhs),
                            IntCmpOp::UGe => unsigned(lhs) >= unsigned(rhs),
                        };
                        let narrow = |v| {
                            Operand::Immediate(match width {
                                8 => Immediate::I8(v as i8),
                                16 => Immediate::I16(v as i16),
                                _ => Immediate::I32(v as i32),
                            })
                        };
                        let mut operands = [narrow(lhs), narrow(rhs)];
                        if converted {
                            let dst = reg(id);
                            id += 1;
                            block.push(Instruction::IntConvert {
                                from: MirType::Scalar(ScalarType::I64),
                                to: ty,
                                signed: false,
                                dst: dst.clone(),
                                src: imm(lhs),
                            });
                            operands[0] = Operand::Register(dst);
                        }
                        let result = reg(id);
                        let failed = reg(id + 1);
                        let accumulated = reg(id + 2);
                        id += 3;
                        block.push(Instruction::IntCmp {
                            op,
                            ty,
                            dst: result.clone(),
                            lhs: operands[0].clone(),
                            rhs: operands[1].clone(),
                        });
                        block.push(Instruction::IntCmp {
                            op: IntCmpOp::Ne,
                            ty: MirType::Scalar(ScalarType::I1),
                            dst: failed.clone(),
                            lhs: Operand::Register(result),
                            rhs: imm(i64::from(expected)),
                        });
                        let flag = reg(id);
                        id += 1;
                        block.push(Instruction::IntConvert {
                            from: MirType::Scalar(ScalarType::I1),
                            to: MirType::Scalar(ScalarType::I64),
                            signed: false,
                            dst: flag.clone(),
                            src: Operand::Register(failed),
                        });
                        block.push(Instruction::IntBinary {
                            op: IntBinOp::Or,
                            ty: MirType::Scalar(ScalarType::I64),
                            dst: accumulated.clone(),
                            lhs: failures,
                            rhs: Operand::Register(flag),
                        });
                        failures = Operand::Register(accumulated);
                    }
                }
            }
            block.push(Instruction::Ret {
                value: Some(failures),
            });
            let module = module(block);
            assert_eq!(
                interpret_mir_function(&module.functions["main"], &[])?,
                Some(0),
                "{ty} converted={converted}"
            );
            for allocator in [
                RegallocStrategy::Incremental,
                RegallocStrategy::LinearScanGlobal,
            ] {
                let output = run(&module, &targets(false)[0], allocator, directory)?;
                assert_eq!(
                    output.status.code(),
                    Some(0),
                    "{ty} converted={converted}: {output:?}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn integer_conversions_preserve_source_bits_and_destination_width() -> TestResult {
    let guard = directory("conversions");
    let directory = guard.path();
    let mut block = Block::new("entry");
    let mut id = 0;
    let mut failures = imm(0);
    for from in [
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::I16),
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I64),
    ] {
        for to in [
            MirType::Scalar(ScalarType::I8),
            MirType::Scalar(ScalarType::I16),
            MirType::Scalar(ScalarType::I32),
            MirType::Scalar(ScalarType::I64),
        ] {
            if from == to {
                continue;
            }
            for signed in [false, true] {
                if signed
                    && from.size_bytes(PointerWidth::Bits64) > to.size_bytes(PointerWidth::Bits64)
                {
                    continue;
                }
                for value in [
                    0, 1, -1, 127, 128, 255, 32768, 65535, 2147483648, 4294967295,
                ] {
                    let dst = reg(id);
                    let wide = reg(id + 1);
                    let failed = reg(id + 2);
                    let accumulated = reg(id + 3);
                    id += 4;
                    let expected = to.signed_integer(
                        if signed {
                            from.signed_integer(value, PointerWidth::Bits64)
                        } else {
                            from.unsigned_integer(value, PointerWidth::Bits64) as i64
                        },
                        PointerWidth::Bits64,
                    );
                    let instruction = Instruction::IntConvert {
                        from,
                        to,
                        signed,
                        dst: dst.clone(),
                        src: imm(value),
                    };
                    let mut single = Block::new("entry");
                    single.push(instruction.clone());
                    single.push(Instruction::Ret {
                        value: Some(Operand::Register(dst.clone())),
                    });
                    assert_eq!(
                        interpret_mir_function(&module(single).functions["main"], &[])?,
                        Some(expected),
                        "{from} -> {to} signed={signed} value={value}"
                    );
                    block.push(instruction);
                    block.push(if to == MirType::Scalar(ScalarType::I64) {
                        Instruction::Copy {
                            ty: to,
                            dst: wide.clone(),
                            src: Operand::Register(dst),
                        }
                    } else {
                        Instruction::IntConvert {
                            from: to,
                            to: MirType::Scalar(ScalarType::I64),
                            signed: true,
                            dst: wide.clone(),
                            src: Operand::Register(dst),
                        }
                    });
                    block.push(Instruction::IntCmp {
                        op: IntCmpOp::Ne,
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: failed.clone(),
                        lhs: Operand::Register(wide),
                        rhs: imm(expected),
                    });
                    let flag = reg(id);
                    id += 1;
                    block.push(Instruction::IntConvert {
                        from: MirType::Scalar(ScalarType::I1),
                        to: MirType::Scalar(ScalarType::I64),
                        signed: false,
                        dst: flag.clone(),
                        src: Operand::Register(failed),
                    });
                    block.push(Instruction::IntBinary {
                        op: IntBinOp::Or,
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: accumulated.clone(),
                        lhs: failures,
                        rhs: Operand::Register(flag),
                    });
                    failures = Operand::Register(accumulated);
                }
            }
        }
    }
    block.push(Instruction::Ret {
        value: Some(failures),
    });
    let module = module(block);
    for allocator in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
    ] {
        let output = run(&module, &targets(false)[0], allocator, directory)?;
        assert_eq!(output.status.code(), Some(0), "{output:?}");
    }
    Ok(())
}
