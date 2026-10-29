#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::{
    mir::{Module, codegen::from_ir},
    mir_codegen::MirCodegenSettings,
    parser::parse_module,
};

fn lower(source: &str) -> Module {
    from_ir(&parse_module(source).expect("parse"), "vector").expect("lower")
}

fn assembly(module: &Module) -> String {
    let mut out = Vec::new();
    generate_mir_x86_64_with_units_and_settings(
        module,
        &mut out,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )
    .expect("emit");
    String::from_utf8(out).expect("utf8")
}

#[cfg(feature = "nightly")]
fn mnemonics(asm: &str) -> Vec<&str> {
    asm.lines()
        .filter(|l| l.contains("xmm"))
        .filter_map(|l| l.split_whitespace().next())
        .collect()
}

/// A vector instruction with every register replaced by the order it first appears, so
/// two lowerings can be compared without their virtual register numbering.
#[cfg(feature = "nightly")]
fn shape(module: &Module) -> Vec<String> {
    use crate::mir::{Instruction as MirInstruction, Operand};
    let mut names: Vec<String> = Vec::new();
    let mut index = |reg: String| {
        let at = names.iter().position(|n| *n == reg).unwrap_or_else(|| {
            names.push(reg);
            names.len() - 1
        });
        format!("r{at}")
    };
    let mut out = Vec::new();
    for inst in module
        .functions
        .values()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
    {
        let text = match &inst.instr {
            MirInstruction::VectorOp {
                op,
                ty,
                dst,
                operands,
            } => {
                let operands: Vec<String> = operands
                    .iter()
                    .map(|o| match o {
                        Operand::Register(r) => index(r.to_string()),
                        other => other.to_string(),
                    })
                    .collect();
                format!(
                    "{op}.{ty} {} = {}",
                    index(dst.to_string()),
                    operands.join(", ")
                )
            }
            MirInstruction::Load { ty, dst, .. } if ty.is_vector() => {
                format!("load.{ty} {}", index(dst.to_string()))
            }
            MirInstruction::Store { ty, src, .. } if ty.is_vector() => {
                let src = match src {
                    Operand::Register(r) => index(r.to_string()),
                    other => other.to_string(),
                };
                format!("store.{ty} {src}")
            }
            _ => continue,
        };
        out.push(text);
    }
    out
}

#[test]
fn add_v4i32_emits_one_packed_add() {
    let module = lower(
        "fn @f(ptr %p) -> i64 {\n\
         entry:\n\
         \x20 %a = load.<4 x i32> %p\n\
         \x20 %b = load.<4 x i32> %p\n\
         \x20 %c = add.<4 x i32> %a, %b\n\
         \x20 store.<4 x i32> %p, %c\n\
         \x20 ret.i64 0\n\
         }\n",
    );
    let asm = assembly(&module);
    assert_eq!(
        asm.matches("paddd").count(),
        1,
        "expected exactly one paddd:\n{asm}"
    );
    assert_eq!(asm.matches("addl").count(), 0, "scalar adds leaked:\n{asm}");
}

#[test]
fn every_lane_width_reaches_its_own_packed_mnemonic() {
    for (ty, mnemonic) in [
        ("<16 x i8>", "paddb"),
        ("<8 x i16>", "paddw"),
        ("<4 x i32>", "paddd"),
        ("<2 x i64>", "paddq"),
        ("<4 x f32>", "addps"),
        ("<2 x f64>", "addpd"),
    ] {
        let module = lower(&format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n  %a = load.{ty} %p\n  %c = add.{ty} %a, %a\n  store.{ty} %p, %c\n  ret.i64 0\n}}\n"
        ));
        let asm = assembly(&module);
        assert!(asm.contains(mnemonic), "{ty} missed {mnemonic}:\n{asm}");
    }
}

#[test]
fn every_lane_width_shuffles() {
    for ty in [
        "<16 x i8>",
        "<8 x i16>",
        "<4 x i32>",
        "<2 x i64>",
        "<4 x f32>",
        "<2 x f64>",
    ] {
        let lanes: usize = ty[1..ty.find(' ').expect("lane count")]
            .parse()
            .expect("lane count");
        let mask: Vec<String> = (0..lanes)
            .map(|i| (2 * lanes - 1 - i).to_string())
            .collect();
        let module = lower(&format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n  %a = load.{ty} %p\n  %s = shuffle.{ty} %a, %a, [{}]\n  store.{ty} %p, %s\n  ret.i64 0\n}}\n",
            mask.join(", ")
        ));
        assembly(&module);
    }
}

#[test]
fn splat_extract_and_insert_reach_their_instructions() {
    let module = lower(
        "fn @f(ptr %p) -> i64 {\n\
         entry:\n\
         \x20 %v = splat.<4 x i32> 7\n\
         \x20 %w = insertlane.<4 x i32> %v, 9, 1\n\
         \x20 %x = extractlane.<4 x i32> %w, 2\n\
         \x20 %r = extend.i32.i64 %x\n\
         \x20 ret.i64 %r\n\
         }\n",
    );
    let asm = assembly(&module);
    for mnemonic in ["pshufd", "pinsrd", "pextrd"] {
        assert!(asm.contains(mnemonic), "missing {mnemonic}:\n{asm}");
    }
}

/// The auto-vectorizer builds `Instruction::VectorOp` directly; the surface lowers to
/// the same thing. If these two ever drift apart there are two vector representations
/// in the compiler, which is the failure this design exists to avoid.
#[cfg(feature = "nightly")]
#[test]
fn vectorizer_and_surface_paths_agree() {
    use crate::mir::transform::{TransformPipeline, VectorUnit};
    use lamina_platform::X86CpuLevel;

    let mut vectorized = lower(
        "fn @f() -> i64 {
entry:
  %a = stackalloc.[64 x i64]
  %j = add.i64 0, 0
  jmp head
head:
  %more = lt.i64 %j, 64
  br %more, body, done
body:
  %q = getelem.i64 %a, %j
  %x = load.i64 %q
  %z = add.i64 %x, 5
  store.i64 %q, %z
  %j = add.i64 %j, 1
  jmp head
done:
  ret.i64 0
}",
    );
    TransformPipeline::default_for_opt_level_on(3, Some(VectorUnit::X86Sse2(X86CpuLevel::V2)))
        .apply_to_module(&mut vectorized)
        .expect("vectorize");

    let surface = lower(
        "fn @f(ptr %q) -> i64 {\n\
         entry:\n\
         \x20 %s = splat.<2 x i64> 5\n\
         \x20 %x = load.<2 x i64> %q\n\
         \x20 %z = add.<2 x i64> %x, %s\n\
         \x20 store.<2 x i64> %q, %z\n\
         \x20 ret.i64 0\n\
         }\n",
    );

    assert!(shape(&surface).len() >= 4, "nothing to compare");
    assert_eq!(
        shape(&vectorized),
        shape(&surface),
        "the vectorizer and the surface built different vector MIR"
    );
    assert!(!mnemonics(&assembly(&surface)).is_empty(), "no vector code");
    assert_eq!(
        mnemonics(&assembly(&vectorized)),
        mnemonics(&assembly(&surface)),
        "the vectorizer and the surface emitted different vector code"
    );
}
