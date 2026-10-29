use super::abi::{
    ArgKind, ArgPlace, call_area, classify_kinds, frame_bytes, incoming_stack_offset,
};
use crate::mir::{Aggregate, AggregateClass, AggregateShape, MirType, ScalarType};

fn scalars(types: &[MirType]) -> Vec<ArgKind> {
    types.iter().copied().map(ArgKind::scalar).collect()
}

#[test]
fn argument_registers_are_numbered_by_doubleword_across_both_files() {
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
            ArgPlace::Fpr(1),
            ArgPlace::Gpr(2),
            ArgPlace::Fpr(3)
        ]
    );
}

#[test]
fn the_ninth_argument_goes_to_the_stack() {
    let types = vec![MirType::Scalar(ScalarType::I64); 10];
    let kinds = scalars(&types);
    let places = classify_kinds(&kinds, 0);
    assert_eq!(places[7], ArgPlace::Gpr(7));
    assert_eq!(places[8], ArgPlace::Stack(0));
    assert_eq!(places[9], ArgPlace::Stack(1));
    assert_eq!(call_area(&kinds, 0), 16);
}

#[test]
fn an_aggregate_takes_consecutive_doublewords_and_splits_at_the_eighth() {
    let aggregate = Aggregate {
        size: 24,
        align: 8,
        class: AggregateClass::Integer,
        shape: AggregateShape::Composite,
    };
    let kinds = vec![
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind::scalar(MirType::Scalar(ScalarType::I64)),
        ArgKind {
            ty: MirType::Scalar(ScalarType::Ptr),
            aggregate: Some(aggregate),
        },
    ];
    assert_eq!(
        classify_kinds(&kinds, 0)[7],
        ArgPlace::Split { first: 7, words: 3 }
    );
    assert_eq!(call_area(&kinds, 0), 16);
}

#[test]
fn a_hidden_result_pointer_shifts_every_argument_by_one_doubleword() {
    let types = vec![MirType::Scalar(ScalarType::I64); 2];
    assert_eq!(
        classify_kinds(&scalars(&types), 1),
        vec![ArgPlace::Gpr(1), ArgPlace::Gpr(2)]
    );
}

#[test]
fn the_frame_keeps_the_sixteen_byte_stack_alignment() {
    for locals in 0..40 {
        for outgoing in 0..40 {
            assert_eq!(frame_bytes(locals, outgoing) % 16, 0, "{locals} {outgoing}");
        }
    }
}

#[test]
fn incoming_stack_arguments_start_above_the_saved_registers() {
    assert_eq!(incoming_stack_offset(0), 16);
    assert_eq!(incoming_stack_offset(1), 24);
}
