#![allow(clippy::expect_used)]

use super::*;
use crate::{
    mir::{Module, VectorLane, VectorOp, VectorType, codegen::from_ir},
    mir_codegen::{CodegenError, MirCodegenSettings, simd_support},
    parser::parse_module,
};
use lamina_platform::X86CpuLevel;

fn emit(module: &Module, level: X86CpuLevel) -> Result<String, LaminaError> {
    let settings = MirCodegenSettings {
        x86_cpu_level: level,
        ..MirCodegenSettings::default()
    };
    let mut out = Vec::new();
    generate_mir_x86_64_with_units_and_settings(
        module,
        &mut out,
        TargetOperatingSystem::Linux,
        1,
        &settings,
    )?;
    Ok(String::from_utf8(out).expect("utf8 assembly"))
}

fn popcount_module() -> Module {
    let ir = parse_module(
        "fn @f(i64 %x) -> i64 {\nentry:\n    %p = popcount.i64 %x\n    ret.i64 %p\n}\n",
    )
    .expect("parse popcount");
    from_ir(&ir, "popcount").expect("lower popcount")
}

fn lane_module(op: VectorOp, lane: VectorLane) -> Module {
    let mut module = Module::new("lane");
    module.add_function(simd_support::function(
        "f",
        MirType::Vector(VectorType::V128(lane)),
        op,
        1,
    ));
    module
}

fn refused(result: Result<String, LaminaError>) -> (String, X86CpuLevel, X86CpuLevel) {
    match result {
        Err(LaminaError::CodegenError(CodegenError::CpuLevelRequired {
            instruction,
            required,
            selected,
        })) => (instruction, required, selected),
        Err(other) => panic!("expected CpuLevelRequired, got {other:?}"),
        Ok(asm) => panic!("expected refusal, emitted:\n{asm}"),
    }
}

#[test]
fn popcount_emits_popcnt_at_v2_and_above() {
    for level in [X86CpuLevel::V2, X86CpuLevel::V3, X86CpuLevel::V4] {
        let asm = emit(&popcount_module(), level).expect("popcount emits");
        assert!(asm.contains("popcntq %rax, %rax"), "{level}: {asm}");
        assert!(
            !asm.contains("movabsq $0x5555555555555555"),
            "{level}: {asm}"
        );
    }
}

#[test]
fn popcount_falls_back_to_swar_at_v1_without_clobbering_rcx_or_rdx() {
    let asm = emit(&popcount_module(), X86CpuLevel::V1).expect("popcount emits at v1");
    assert!(!asm.contains("popcntq"), "{asm}");
    assert!(asm.contains("movabsq $0x5555555555555555, %rdx"), "{asm}");
    assert!(asm.contains("imulq %rdx, %rax"), "{asm}");
    let push_rdx = asm.find("pushq %rdx").expect("rdx saved");
    let pop_rdx = asm.find("popq %rdx").expect("rdx restored");
    assert!(push_rdx < pop_rdx, "{asm}");
}

#[test]
fn sse41_lane_extract_and_insert_are_refused_below_v2() {
    let lanes = [
        VectorLane::I8,
        VectorLane::I32,
        VectorLane::I64,
        VectorLane::F32,
        VectorLane::F64,
    ];
    for lane in lanes {
        let suffix = match lane {
            VectorLane::I8 => "b",
            VectorLane::I32 | VectorLane::F32 => "d",
            _ => "q",
        };
        for (op, prefix) in [
            (VectorOp::VExtractLane, "pextr"),
            (VectorOp::VInsertLane, "pinsr"),
        ] {
            let module = lane_module(op, lane);
            let (instruction, required, selected) = refused(emit(&module, X86CpuLevel::V1));
            assert_eq!(instruction, format!("{prefix}{suffix}"), "{lane:?}");
            assert_eq!(required, X86CpuLevel::V2);
            assert_eq!(selected, X86CpuLevel::V1);

            let asm = emit(&module, X86CpuLevel::V2).expect("lane access emits at v2");
            assert!(
                asm.contains(&format!("{prefix}{suffix} $1")),
                "{lane:?}: {asm}"
            );
        }
    }
}

#[test]
fn sse2_word_lane_access_is_allowed_at_v1() {
    for op in [VectorOp::VExtractLane, VectorOp::VInsertLane] {
        let asm = emit(&lane_module(op, VectorLane::I16), X86CpuLevel::V1)
            .expect("16-bit lane access is SSE2");
        let prefix = if op == VectorOp::VExtractLane {
            "pextrw"
        } else {
            "pinsrw"
        };
        assert!(asm.contains(prefix), "{asm}");
    }
}

#[test]
fn packed_dword_multiply_is_refused_below_v2() {
    let module = lane_module(VectorOp::VMul, VectorLane::I32);
    let (instruction, required, selected) = refused(emit(&module, X86CpuLevel::V1));
    assert_eq!(instruction, "pmulld");
    assert_eq!(required, X86CpuLevel::V2);
    assert_eq!(selected, X86CpuLevel::V1);
    let asm = emit(&module, X86CpuLevel::V2).expect("pmulld emits at v2");
    assert!(asm.contains("pmulld %xmm"), "{asm}");
}

#[test]
fn packed_word_multiply_stays_available_at_v1() {
    let asm = emit(
        &lane_module(VectorOp::VMul, VectorLane::I16),
        X86CpuLevel::V1,
    )
    .expect("pmullw is SSE2");
    assert!(asm.contains("pmullw"), "{asm}");
}

#[test]
fn refusal_names_the_instruction_and_both_levels() {
    let error = emit(
        &lane_module(VectorOp::VMul, VectorLane::I32),
        X86CpuLevel::V1,
    )
    .expect_err("pmulld must be refused at v1");
    let message = error.to_string();
    assert!(message.contains("`pmulld`"), "{message}");
    assert!(message.contains("x86-64-v2"), "{message}");
    assert!(message.contains("x86-64-v1"), "{message}");
}
