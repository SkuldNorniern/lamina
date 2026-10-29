//! Shape rules shared by the IR builders. Each check panics with the rule the text parser enforces.

use crate::types::PrimitiveType;

pub(crate) fn is_signed(ty: PrimitiveType) -> bool {
    matches!(
        ty,
        PrimitiveType::I8
            | PrimitiveType::I16
            | PrimitiveType::I32
            | PrimitiveType::I64
            | PrimitiveType::I128
    )
}

pub(crate) fn integer_bits(ty: PrimitiveType) -> Option<u32> {
    match ty {
        PrimitiveType::Bool => Some(1),
        PrimitiveType::I8 | PrimitiveType::U8 => Some(8),
        PrimitiveType::I16 | PrimitiveType::U16 => Some(16),
        PrimitiveType::I32 | PrimitiveType::U32 => Some(32),
        PrimitiveType::I64 | PrimitiveType::U64 => Some(64),
        PrimitiveType::I128 | PrimitiveType::U128 => Some(128),
        _ => None,
    }
}

pub(crate) fn scalar_bits(ty: PrimitiveType) -> Option<u32> {
    match ty {
        PrimitiveType::F32 => Some(32),
        PrimitiveType::F64 => Some(64),
        PrimitiveType::Bool => None,
        _ => integer_bits(ty),
    }
}

pub(crate) fn integer_signedness(ty: PrimitiveType) -> Option<bool> {
    match ty {
        PrimitiveType::I8
        | PrimitiveType::I16
        | PrimitiveType::I32
        | PrimitiveType::I64
        | PrimitiveType::I128 => Some(true),
        PrimitiveType::U8
        | PrimitiveType::U16
        | PrimitiveType::U32
        | PrimitiveType::U64
        | PrimitiveType::U128 => Some(false),
        _ => None,
    }
}

pub(crate) fn check_zext(source: PrimitiveType, target: PrimitiveType) {
    assert!(
        !is_signed(source),
        "zext zero-extends an unsigned or bool source, found {source}; use sext for a signed source"
    );
    check_extend(source, target);
}

pub(crate) fn check_sext(source: PrimitiveType, target: PrimitiveType) {
    assert!(
        is_signed(source),
        "sext sign-extends a signed source, found {source}; use zext for an unsigned or bool source"
    );
    check_extend(source, target);
}

fn check_extend(source: PrimitiveType, target: PrimitiveType) {
    let source_bits = integer_bits(source)
        .unwrap_or_else(|| panic!("extend source must be an integer or bool, found {source}"));
    let target_bits = match target {
        PrimitiveType::Bool => None,
        _ => integer_bits(target),
    }
    .unwrap_or_else(|| panic!("extend target must be an integer, found {target}"));
    assert!(
        target_bits > source_bits,
        "extend from {source} to {target} must widen; use trunc to narrow"
    );
}

pub(crate) fn check_trunc(source: PrimitiveType, target: PrimitiveType) {
    let source_bits = match source {
        PrimitiveType::Bool => None,
        _ => integer_bits(source),
    }
    .unwrap_or_else(|| panic!("trunc source must be an integer, found {source}"));
    let target_bits = match target {
        PrimitiveType::Bool => None,
        _ => integer_bits(target),
    }
    .unwrap_or_else(|| panic!("trunc target must be an integer, found {target}"));
    assert!(
        target_bits < source_bits,
        "trunc from {source} to {target} must narrow; use extend to widen"
    );
}

pub(crate) fn check_bitcast(source: PrimitiveType, target: PrimitiveType) {
    if let (Some(source_bits), Some(target_bits)) = (scalar_bits(source), scalar_bits(target)) {
        assert!(
            source_bits == target_bits,
            "bitcast reinterprets equal widths, found {source} ({source_bits} bits) to {target} ({target_bits} bits)"
        );
    }
}
