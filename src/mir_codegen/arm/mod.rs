pub mod aarch64;
pub mod armv7;

use crate::mir::{FloatCmpOp, IntCmpOp};

/// Condition code for an integer comparison. A32 and A64 spell these the same, so both
/// backends read them from here.
pub(crate) fn int_condition(op: IntCmpOp) -> &'static str {
    match op {
        IntCmpOp::Eq => "eq",
        IntCmpOp::Ne => "ne",
        IntCmpOp::SLt => "lt",
        IntCmpOp::SLe => "le",
        IntCmpOp::SGt => "gt",
        IntCmpOp::SGe => "ge",
        IntCmpOp::ULt => "lo",
        IntCmpOp::ULe => "ls",
        IntCmpOp::UGt => "hi",
        IntCmpOp::UGe => "hs",
    }
}

/// Condition code for a floating-point comparison, read from the flags a `fcmp` or a
/// `vcmp` plus `vmrs` leaves behind.
pub(crate) fn float_condition(op: FloatCmpOp) -> &'static str {
    match op {
        FloatCmpOp::Eq => "eq",
        FloatCmpOp::Ne => "ne",
        FloatCmpOp::Lt => "mi",
        FloatCmpOp::Le => "ls",
        FloatCmpOp::Gt => "gt",
        FloatCmpOp::Ge => "ge",
    }
}

pub(crate) fn invert_condition(cond: &str) -> Option<&'static str> {
    Some(match cond {
        "eq" => "ne",
        "ne" => "eq",
        "lt" => "ge",
        "ge" => "lt",
        "le" => "gt",
        "gt" => "le",
        "lo" => "hs",
        "hs" => "lo",
        "ls" => "hi",
        "hi" => "ls",
        "mi" => "pl",
        "pl" => "mi",
        _ => return None,
    })
}
