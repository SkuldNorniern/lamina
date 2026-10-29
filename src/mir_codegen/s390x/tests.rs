use super::abi::{
    ArgKind, ArgPlace, call_area, classify_kinds, frame_bytes, incoming_stack_offset,
};
use crate::mir::{Aggregate, AggregateClass, AggregateShape, MirType, ScalarType};

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
