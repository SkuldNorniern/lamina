use super::abi::{ArgSlot, aggregate_bytes, classify, frame_bytes, outgoing_bytes, slot_offset};
use crate::mir::MirType;
use lamina_mir::ScalarType;

#[test]
fn every_argument_takes_a_word_on_the_stack() {
    let types = [
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::Ptr),
        MirType::Scalar(ScalarType::F64),
    ];
    let places = classify(&types, &[None; 4], false);
    assert_eq!(
        places,
        [
            ArgSlot {
                offset: 0,
                bytes: 4
            },
            ArgSlot {
                offset: 4,
                bytes: 4
            },
            ArgSlot {
                offset: 8,
                bytes: 4
            },
            ArgSlot {
                offset: 12,
                bytes: 8
            },
        ]
    );
    assert_eq!(outgoing_bytes(&places, false), 32);
}

#[test]
fn a_hidden_struct_pointer_takes_the_first_word() {
    let types = [MirType::Scalar(ScalarType::I32)];
    let places = classify(&types, &[None], true);
    assert_eq!(
        places,
        [ArgSlot {
            offset: 4,
            bytes: 4
        }]
    );
    assert_eq!(outgoing_bytes(&places, true), 16);
}

#[test]
fn an_aggregate_argument_is_copied_by_value_rounded_to_a_word() {
    let types = [
        MirType::Scalar(ScalarType::Ptr),
        MirType::Scalar(ScalarType::I32),
    ];
    let places = classify(&types, &[Some(13), None], false);
    assert_eq!(
        places,
        [
            ArgSlot {
                offset: 0,
                bytes: 16
            },
            ArgSlot {
                offset: 16,
                bytes: 4
            },
        ]
    );
    assert_eq!(aggregate_bytes(13), 16);
}

#[test]
fn no_arguments_need_no_outgoing_area() {
    assert_eq!(outgoing_bytes(&[], false), 0);
}

/// `call` leaves `%esp` four past a boundary and `pushl %ebp` another four, so the
/// frame must be `8 mod 16` for `%esp` to sit on one inside the body.
#[test]
fn the_frame_restores_the_sixteen_byte_alignment_a_call_needs() {
    for locals in [0, 8, 16, 24, 120, 4096] {
        let frame = frame_bytes(locals);
        assert!(frame >= locals, "frame {frame} must hold {locals} bytes");
        assert_eq!((4 + 4 + frame) % 16, 0, "frame {frame} for {locals} locals");
    }
}

#[test]
fn frame_slots_descend_from_the_frame_pointer_in_eight_byte_steps() {
    assert_eq!(slot_offset(0), -8);
    assert_eq!(slot_offset(1), -16);
    assert_eq!(slot_offset(7), -64);
}
