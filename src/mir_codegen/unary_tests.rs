use super::{generate_mir_to_target, test_support::cross_assembler_ready};
use crate::mir::{
    Block, FloatBinOp, FloatUnOp, Function, Immediate, Instruction, MirType, Module, Operand,
    Parameter, Register, Signature, VirtualReg,
};
use crate::mir_codegen::simd_support::{objdump_tool, tools_available};
use crate::test_support::TempDir;
use lamina_mir::ScalarType;
use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};
use std::{error::Error, fs, process::Command};

fn unary_module(register_inputs: bool) -> Module {
    let mut module = Module::new("float_unary");
    let mut function = Function::new(Signature::new("main"));
    let mut block = Block::new("entry");
    let mut id = 0;
    for (ty, value) in [
        (MirType::Scalar(ScalarType::F32), Immediate::F32(4.0)),
        (MirType::Scalar(ScalarType::F64), Immediate::F64(4.0)),
    ] {
        let src = if register_inputs {
            let reg = Register::Virtual(VirtualReg::fpr(100 + id));
            function.sig.params.push(Parameter::new(reg.clone(), ty));
            Operand::Register(reg)
        } else {
            Operand::Immediate(value)
        };
        for op in [FloatUnOp::FNeg, FloatUnOp::FSqrt] {
            block.push(Instruction::FloatUnary {
                op,
                ty,
                dst: VirtualReg::fpr(id).into(),
                src: src.clone(),
            });
            id += 1;
        }
    }
    for (ty, value) in [
        (MirType::Scalar(ScalarType::F32), Immediate::F32(4.0)),
        (MirType::Scalar(ScalarType::F64), Immediate::F64(4.0)),
    ] {
        let src = if register_inputs {
            let reg = Register::Virtual(VirtualReg::fpr(100 + id));
            function.sig.params.push(Parameter::new(reg.clone(), ty));
            Operand::Register(reg)
        } else {
            Operand::Immediate(value)
        };
        for op in [
            FloatBinOp::FAdd,
            FloatBinOp::FSub,
            FloatBinOp::FMul,
            FloatBinOp::FDiv,
        ] {
            block.push(Instruction::FloatBinary {
                op,
                ty,
                dst: VirtualReg::fpr(id).into(),
                lhs: src.clone(),
                rhs: src.clone(),
            });
            id += 1;
        }
    }
    block.push(Instruction::Ret { value: None });
    function.blocks.push(block);
    module.add_function(function);
    module
}

#[test]
fn float_unary_cpu_output_assembles() -> Result<(), Box<dyn Error>> {
    let guard = TempDir::new("lamina-float-unary");
    let dir = guard.path();
    let module = unary_module(false);
    for (arch, triple) in [
        (TargetArchitecture::PowerPC64, "powerpc64-linux-gnu"),
        (TargetArchitecture::PowerPC64, "powerpc64le-linux-gnu"),
        (TargetArchitecture::X86_64, "x86_64-linux-gnu"),
        (TargetArchitecture::Aarch64, "aarch64-linux-gnu"),
        (TargetArchitecture::Riscv64, "riscv64-linux-gnu"),
        (TargetArchitecture::Riscv32, "riscv32-linux-gnu"),
    ] {
        let module = if arch == TargetArchitecture::Riscv32 {
            unary_module(true)
        } else {
            module.clone()
        };
        let target_flag = format!("--target={triple}");
        if !cross_assembler_ready(dir, &[target_flag.as_str()]) {
            continue;
        }
        let mut assembly = Vec::new();
        generate_mir_to_target(
            &module,
            &mut assembly,
            &Target::new(arch, TargetOperatingSystem::Linux),
            1,
        )?;
        let source = dir.join(format!("{triple}.s"));
        let object = dir.join(format!("{triple}.o"));
        fs::write(&source, assembly)?;
        let result = Command::new("clang")
            .arg(format!("--target={triple}"))
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .output()?;
        assert!(
            result.status.success(),
            "{triple}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let disassembly = Command::new(objdump_tool().expect("cross_assembler_ready found one"))
            .arg("-d")
            .arg(object)
            .output()?;
        assert!(
            disassembly.status.success(),
            "{triple}: {}",
            String::from_utf8_lossy(&disassembly.stderr)
        );
        let text = String::from_utf8(disassembly.stdout)?;
        assert!(!text.contains("<unknown>"), "{triple}: {text}");
        if arch == TargetArchitecture::PowerPC64 {
            assert_eq!(text.matches("\tfneg ").count(), 2, "{text}");
            assert!(text.contains("\tfsqrts "), "{text}");
            assert!(text.contains("\tfsqrt "), "{text}");
        }
    }
    Ok(())
}

#[test]
fn float_unary_wasm_output_validates() -> Result<(), Box<dyn Error>> {
    if !tools_available(&["wasmtime"]) {
        return Ok(());
    }
    let module = unary_module(false);
    let guard = TempDir::new("lamina-float-unary-wasm");
    let dir = guard.path();
    for arch in [TargetArchitecture::Wasm32, TargetArchitecture::Wasm64] {
        let mut assembly = Vec::new();
        generate_mir_to_target(
            &module,
            &mut assembly,
            &Target::new(arch, TargetOperatingSystem::Unknown),
            1,
        )?;
        let source = dir.join("unary.wat");
        fs::write(&source, assembly)?;
        let compiled = dir.join(format!("unary-{arch:?}.cwasm"));
        let result = Command::new("wasmtime")
            .arg("compile")
            .arg(&source)
            .arg("-o")
            .arg(&compiled)
            .output()?;
        assert!(
            result.status.success(),
            "{arch:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    Ok(())
}
