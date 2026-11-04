//! Checking of the `sret` and `byval(size, align)` attributes a frontend states.
//!
//! An attribute is a claim about the target's C ABI, not an instruction. Lowering
//! classifies the type itself and compares; a disagreement is the bug these
//! attributes exist to catch, so it is refused rather than resolved in favour of
//! either side.

use crate::{
    ir::{Type as IRType, function::VariableAnnotation},
    mir::{codegen::error::FromIRError, types::Aggregate},
};

/// The size and alignment a `byval` attribute on this parameter states, if any.
pub fn stated_byval(annotations: &[VariableAnnotation]) -> Option<(u64, u64)> {
    annotations.iter().find_map(|annotation| match annotation {
        VariableAnnotation::ByVal { size, align } => Some((*size, *align)),
        _ => None,
    })
}

/// Checks a stated `byval` against the classification lowering reached on its own.
/// Returns whether the parameter carries the claim.
pub fn verify_byval(
    annotations: &[VariableAnnotation],
    computed: Option<Aggregate>,
    ty: &IRType<'_>,
    what: &str,
) -> Result<bool, FromIRError> {
    let Some((size, align)) = stated_byval(annotations) else {
        return Ok(false);
    };
    let Some(computed) = computed else {
        return Err(FromIRError::AbiClaimMismatch(format!(
            "{what} states byval({size}, {align}), but Lamina classifies {ty} as a scalar, which is passed in a register and never as a by-value copy"
        )));
    };
    if computed.size as u64 != size || computed.align as u64 != align {
        return Err(FromIRError::AbiClaimMismatch(format!(
            "{what} states byval({size}, {align}), but Lamina classifies {ty} as byval({}, {})",
            computed.size, computed.align
        )));
    }
    Ok(true)
}

/// Checks a stated `sret` against the classification lowering reached on its own.
pub fn verify_sret(
    sret: bool,
    computed: Option<Aggregate>,
    ty: &IRType<'_>,
    name: &str,
) -> Result<(), FromIRError> {
    if sret && computed.is_none() {
        return Err(FromIRError::AbiClaimMismatch(format!(
            "return of @{name} states sret, but Lamina classifies {ty} as a scalar, which is returned in a register and not through a hidden pointer"
        )));
    }
    Ok(())
}
