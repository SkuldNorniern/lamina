use super::abi::{
    ArgKind, ArgPlace, call_area, classify_kinds, frame_bytes, incoming_stack_offset,
};
use super::bit::{emit_int_rotate, emit_int_unary};
use super::variadic::{SAVE_BYTES, VaArea};
use crate::mir::codegen::from_ir_for_target;
use crate::mir::{
    Aggregate, AggregateClass, AggregateShape, IntRotOp, IntUnOp, MirType, Operand, Register,
    ScalarType, VirtualReg,
};
use crate::mir_codegen::settings::MirCodegenSettings;
use crate::parser::parse_module;
use lamina_ir::types::{Type as IRType, sizeof_type};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem, VaListAbi};
use std::collections::HashMap;

fn variadic_asm(source: &str) -> String {
    let ir = parse_module(source).expect("parse");
    let mir = from_ir_for_target(&ir, "module", TargetArchitecture::S390x.data_layout())
        .expect("lower to MIR");
    let mut out = Vec::new();
    super::generate_mir_s390x_with_units_and_settings(
        &mir,
        &mut out,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )
    .expect("emit");
    String::from_utf8(out).expect("utf8")
}

fn scalars(types: &[MirType]) -> Vec<ArgKind> {
    types.iter().copied().map(ArgKind::scalar).collect()
}

#[test]
fn floats_and_integers_take_separate_argument_banks() {
    let types = [
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F64),
        MirType::Scalar(ScalarType::I64),
        MirType::Scalar(ScalarType::F32),
    ];
    assert_eq!(
        classify_kinds(&scalars(&types), 0),
        vec![
            ArgPlace::Gpr(0),
            ArgPlace::Fpr(0),
            ArgPlace::Gpr(1),
            ArgPlace::Fpr(1)
        ]
    );
}

#[test]
fn the_sixth_integer_argument_goes_to_the_stack() {
    let types = vec![MirType::Scalar(ScalarType::I64); 7];
    let kinds = scalars(&types);
    let places = classify_kinds(&kinds, 0);
    assert_eq!(places[4], ArgPlace::Gpr(4));
    assert_eq!(places[5], ArgPlace::Stack(0));
    assert_eq!(call_area(&kinds, 0), 16);
}

#[test]
fn the_fifth_float_argument_goes_to_the_stack() {
    let types = vec![MirType::Scalar(ScalarType::F64); 5];
    assert_eq!(classify_kinds(&scalars(&types), 0)[4], ArgPlace::Stack(0));
}

#[test]
fn the_frame_holds_a_save_area_below_everything_else() {
    assert_eq!(frame_bytes(0, 0), 168);
    assert_eq!(frame_bytes(24, 16), 160 + 16 + 24 + 8);
    assert_eq!(incoming_stack_offset(0), 160);
    assert_eq!(incoming_stack_offset(2), 176);
}

#[test]
fn an_aggregate_past_a_doubleword_travels_by_reference() {
    let wide = ArgKind {
        ty: MirType::Scalar(ScalarType::Ptr),
        aggregate: Some(Aggregate {
            size: 16,
            align: 8,
            class: AggregateClass::Integer,
            shape: AggregateShape::Composite,
        }),
    };
    let narrow = ArgKind {
        ty: MirType::Scalar(ScalarType::Ptr),
        aggregate: Some(Aggregate {
            size: 8,
            align: 8,
            class: AggregateClass::Integer,
            shape: AggregateShape::Composite,
        }),
    };
    assert!(wide.by_reference());
    assert!(!narrow.by_reference());
    assert_eq!(
        classify_kinds(&[wide, narrow], 0),
        vec![ArgPlace::Gpr(0), ArgPlace::Gpr(1)]
    );
    assert_eq!(call_area(&[wide], 0), 16);
    assert_eq!(call_area(&[narrow], 0), 0);
}

#[test]
fn a_hidden_result_pointer_takes_the_first_register() {
    let kinds = scalars(&[MirType::Scalar(ScalarType::I64)]);
    assert_eq!(classify_kinds(&kinds, 1), vec![ArgPlace::Gpr(1)]);
}

/// Executes the `lghi`, `lgfi`, `llilf` and `iihf` lines `load_imm` writes, using the
/// architectural definition of each.
fn run_load_imm(value: i64) -> i64 {
    let mut out = Vec::new();
    super::abi::load_imm(&mut out, "%r1", value).expect("write to vec");
    let mut reg = 0i64;
    for line in String::from_utf8(out).expect("utf8").lines() {
        let line = line.trim();
        let (op, args) = line.split_once(' ').expect("mnemonic and operands");
        let args: Vec<&str> = args.split(", ").collect();
        let imm = |s: &str| s.parse::<i64>().expect("immediate");
        reg = match (op, args.as_slice()) {
            ("lghi" | "lgfi", ["%r1", v]) => imm(v),
            ("llilf", ["%r1", v]) => imm(v),
            ("iihf", ["%r1", v]) => (reg & 0xffff_ffff) | (imm(v) << 32),
            other => panic!("unexpected instruction {other:?}"),
        };
    }
    reg
}

#[test]
fn load_imm_reconstructs_every_boundary_value() {
    let values = [
        0,
        1,
        -1,
        32767,
        -32768,
        32768,
        -32769,
        i64::from(i32::MAX),
        i64::from(i32::MIN),
        i64::from(i32::MAX) + 1,
        i64::from(i32::MIN) - 1,
        0x1234_5678_9abc_def0,
        i64::MAX,
        i64::MIN,
        0x0000_0001_0000_0000,
        -0x0000_0001_0000_0000,
    ];
    for value in values {
        assert_eq!(run_load_imm(value), value, "load_imm({value:#x})");
    }
}

fn one_slot() -> (HashMap<VirtualReg, i32>, VirtualReg) {
    let vreg = VirtualReg::gpr(1);
    let mut slots = HashMap::new();
    slots.insert(vreg, -8);
    (slots, vreg)
}

fn unary_asm(op: IntUnOp, ty: MirType) -> String {
    let (slots, vreg) = one_slot();
    let mut out = Vec::new();
    emit_int_unary(
        &mut out,
        &slots,
        op,
        &ty,
        &Register::Virtual(vreg),
        &Operand::Register(Register::Virtual(vreg)),
    )
    .expect("emit");
    String::from_utf8(out).expect("utf8")
}

#[test]
fn bit_intrinsics_use_the_z_architecture_instructions() {
    let i64_ty = MirType::Scalar(ScalarType::I64);
    assert!(unary_asm(IntUnOp::Popcount, i64_ty).contains("popcnt"));
    assert!(unary_asm(IntUnOp::Ctlz, i64_ty).contains("flogr"));
    assert!(unary_asm(IntUnOp::Cttz, i64_ty).contains("popcnt"));
    assert!(unary_asm(IntUnOp::Bswap, i64_ty).contains("lrvgr"));
}

#[test]
fn a_byte_wide_ctlz_corrects_for_the_doubleword_flogr_counts_over() {
    let asm = unary_asm(IntUnOp::Ctlz, MirType::Scalar(ScalarType::I8));
    assert!(asm.contains("aghi %r2, -56"), "{asm}");
}

#[test]
fn a_byte_bswap_is_the_identity() {
    let asm = unary_asm(IntUnOp::Bswap, MirType::Scalar(ScalarType::I8));
    assert!(!asm.contains("lrvgr"), "{asm}");
}

#[test]
fn a_rotate_masks_its_count_to_the_operand_width() {
    let (slots, vreg) = one_slot();
    let mut out = Vec::new();
    emit_int_rotate(
        &mut out,
        &slots,
        IntRotOp::Rotl,
        &MirType::Scalar(ScalarType::I16),
        &Register::Virtual(vreg),
        &Operand::Register(Register::Virtual(vreg)),
        &Operand::Register(Register::Virtual(vreg)),
    )
    .expect("emit");
    let asm = String::from_utf8(out).expect("utf8");
    assert!(asm.contains("lghi %r1, 15"), "{asm}");
    assert!(asm.contains("sllg") && asm.contains("srlg"), "{asm}");
}

#[test]
fn the_va_list_counters_start_past_the_named_parameters() {
    let places = [
        ArgPlace::Gpr(0),
        ArgPlace::Fpr(0),
        ArgPlace::Gpr(1),
        ArgPlace::Stack(0),
    ];
    let area = VaArea::new(-SAVE_BYTES, &places, 0);
    assert_eq!(area.gpr, 2);
    assert_eq!(area.fpr, 1);
    assert_eq!(area.overflow, incoming_stack_offset(1));
    assert_eq!(area.save, -160);
}

#[test]
fn a_hidden_result_pointer_counts_against_the_va_list_registers() {
    let area = VaArea::new(-SAVE_BYTES, &[ArgPlace::Gpr(1)], 1);
    assert_eq!(area.gpr, 2);
}

#[test]
fn the_va_list_is_the_thirty_two_bytes_the_elf_abi_states() {
    assert_eq!(
        TargetArchitecture::S390x.va_list(),
        VaListAbi::Defined {
            size: 32,
            alignment: 8
        }
    );
    assert_eq!(
        sizeof_type(&IRType::VaList, TargetArchitecture::S390x.data_layout()),
        Some(32)
    );
}

#[test]
fn va_start_fills_the_four_elf_abi_va_list_fields() {
    let asm = variadic_asm(
        "fn @v(i64 %n, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  va_end %ap\n  ret.i64 %x\n}\n",
    );
    assert!(asm.contains("stg %r2, 0(%r12)"), "{asm}");
    assert!(asm.contains("stg %r2, 8(%r12)"), "{asm}");
    assert!(asm.contains("stg %r2, 16(%r12)"), "{asm}");
    assert!(asm.contains("stg %r2, 24(%r12)"), "{asm}");
}

#[test]
fn the_save_area_is_indexed_by_register_number() {
    let asm = variadic_asm(
        "fn @v(i64 %n, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  %y = va_arg.f64 %ap\n  va_end %ap\n  ret.i64 %x\n}\n",
    );
    assert!(asm.contains("lay %r2, 16(%r1,%r2)"), "{asm}");
    assert!(asm.contains("lay %r2, 128(%r1,%r2)"), "{asm}");
    assert!(asm.contains("clgfi %r3, 5"), "{asm}");
    assert!(asm.contains("clgfi %r3, 4"), "{asm}");
}

#[test]
fn every_global_is_at_least_halfword_aligned() {
    let asm = variadic_asm(
        r#"
global $odd: [3 x i8] = "Pq"
global $next: [5 x i8] = "Rstu"
global $empty: [1 x i8]

fn @main() -> i64 {
entry:
  %a = getelem.i8 $odd, 0
  %b = getelem.i8 $next, 0
  %c = getelem.i8 $empty, 0
  %x = load.i8 %a
  %y = load.i8 %b
  %z = load.i8 %c
  ret.i64 0
}
"#,
    );
    assert!(asm.contains("larl"), "{asm}");
    for line in asm.lines() {
        let Some(exponent) = line.trim().strip_prefix(".p2align ") else {
            continue;
        };
        assert!(
            exponent.trim().parse::<u32>().expect("alignment exponent") >= 1,
            "a symbol `larl` names cannot sit on an odd address: {asm}"
        );
    }
}

fn lowered_aggregate(source: &str, function: &str) -> Aggregate {
    let ir = parse_module(source).expect("parse");
    let mir = from_ir_for_target(&ir, "module", TargetArchitecture::S390x.data_layout())
        .expect("lower to MIR");
    mir.functions[function].sig.params[0]
        .aggregate
        .expect("aggregate marker")
}

#[test]
fn a_record_of_one_float_travels_as_that_float() {
    let single = lowered_aggregate(
        "fn @take(struct { a: f32 } %p) -> i64 {\nentry:\nret.i64 0\n}",
        "take",
    );
    let kind = ArgKind {
        ty: MirType::Scalar(ScalarType::Ptr),
        aggregate: Some(single),
    };
    assert_eq!(
        kind.aggregate_float().map(|field| field.bytes),
        Some(4),
        "a record of one float rides in a floating-point register"
    );
    assert_eq!(classify_kinds(&[kind], 0), vec![ArgPlace::Fpr(0)]);

    let nested = lowered_aggregate(
        "fn @take(struct { a: struct { b: f64 } } %p) -> i64 {\nentry:\nret.i64 0\n}",
        "take",
    );
    assert_eq!(
        ArgKind {
            ty: MirType::Scalar(ScalarType::Ptr),
            aggregate: Some(nested),
        }
        .aggregate_float()
        .map(|field| field.bytes),
        Some(8),
        "the one-member rule follows a chain of records down"
    );
}

#[test]
fn a_float_pair_takes_a_general_register_and_a_float_triple_a_reference() {
    let pair = lowered_aggregate(
        "fn @take(struct { a: f32, b: f32 } %p) -> i64 {\nentry:\nret.i64 0\n}",
        "take",
    );
    let pair = ArgKind {
        ty: MirType::Scalar(ScalarType::Ptr),
        aggregate: Some(pair),
    };
    assert_eq!(pair.aggregate_float(), None);
    assert!(!pair.by_reference());
    assert_eq!(classify_kinds(&[pair], 0), vec![ArgPlace::Gpr(0)]);

    let triple = lowered_aggregate(
        "fn @take(struct { a: f32, b: f32, c: f32 } %p) -> i64 {\nentry:\nret.i64 0\n}",
        "take",
    );
    assert!(
        ArgKind {
            ty: MirType::Scalar(ScalarType::Ptr),
            aggregate: Some(triple),
        }
        .by_reference(),
        "twelve bytes is not one, two, four or eight, so it travels by reference"
    );
}

#[test]
fn an_aggregate_of_an_odd_size_travels_by_reference() {
    for size in [3usize, 5, 6, 7] {
        let kind = ArgKind {
            ty: MirType::Scalar(ScalarType::Ptr),
            aggregate: Some(Aggregate {
                size,
                align: 1,
                class: AggregateClass::Integer,
                shape: AggregateShape::Composite,
            }),
        };
        assert!(kind.by_reference(), "{size} bytes travels by reference");
    }
}
