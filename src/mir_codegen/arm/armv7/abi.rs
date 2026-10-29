//! AAPCS argument placement for armv7, in the VFP hard-float variant.
//!
//! Core arguments take r0-r3 and then the stack; floating point arguments take s0-s15
//! and d0-d7 from a bank of their own, back-filled, and never r0-r3. A variadic
//! argument is placed by the base standard instead, so a `double` handed to `printf`
//! travels in an even core pair or on the stack.

use std::collections::{HashMap, HashSet};

use crate::mir::{
    Aggregate, Immediate, MirType, Module as MirModule, Operand, Register, RegisterClass,
    ScalarType,
};
use lamina_codegen::abi::{Abi, common_call_stub};
use lamina_platform::TargetOperatingSystem;

/// Largest composite AAPCS returns in a register: one word in r0. Anything larger is
/// written to a buffer the caller owns, whose address it passes as a hidden first
/// argument in r0.
pub(crate) const MAX_REGISTER_RETURN_COMPOSITE: usize = 4;

/// Largest legalized integer AAPCS returns in registers: two words in r0:r1, the way
/// a `long long` comes back.
pub(crate) const MAX_REGISTER_RETURN_WIDE_SCALAR: usize = 8;

pub(crate) struct Armv7Abi {
    target_os: TargetOperatingSystem,
}

impl Armv7Abi {
    pub(crate) fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }
}

impl Abi for Armv7Abi {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        name.to_string()
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CoreSpan {
    pub(crate) first: usize,
    pub(crate) words: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct StackSpan {
    pub(crate) offset: usize,
    pub(crate) bytes: usize,
}

/// Where one argument sits. A composite split by AAPCS rule C.5 has both a core span
/// and a stack span; everything else has exactly one of the three.
#[derive(Clone, Copy, Default)]
pub(crate) struct ArgPlace {
    pub(crate) core: Option<CoreSpan>,
    /// Index into s0-s15. A double starts on an even index and covers two.
    pub(crate) vfp: Option<usize>,
    pub(crate) stack: Option<StackSpan>,
}

#[derive(Clone, Copy)]
pub(crate) struct ArgKind {
    pub(crate) float: bool,
    pub(crate) double: bool,
    pub(crate) bytes: usize,
    pub(crate) align: usize,
    pub(crate) aggregate: Option<Aggregate>,
    pub(crate) ty: MirType,
}

impl ArgKind {
    pub(crate) fn of(ty: MirType, aggregate: Option<Aggregate>) -> Self {
        match aggregate {
            Some(aggregate) => Self {
                float: false,
                double: false,
                bytes: aggregate.size,
                align: aggregate.align.max(4),
                aggregate: Some(aggregate),
                ty,
            },
            None => {
                let double = matches!(ty, MirType::Scalar(ScalarType::F64));
                let bytes = if double { 8 } else { 4 };
                Self {
                    float: ty.is_float(),
                    double,
                    bytes,
                    align: bytes,
                    aggregate: None,
                    ty,
                }
            }
        }
    }

    pub(crate) fn pointer() -> Self {
        Self::of(MirType::Scalar(ScalarType::Ptr), None)
    }

    pub(crate) fn words(&self) -> usize {
        self.bytes.div_ceil(4)
    }
}

const CORE_ARGUMENT_REGISTERS: usize = 4;
const VFP_SINGLES: usize = 16;

/// Places `kinds` by AAPCS. `fixed` is the number of named parameters when the callee
/// is variadic; from there on the base standard applies.
pub(crate) fn classify(
    kinds: &[ArgKind],
    fixed: Option<usize>,
    base_standard: bool,
) -> Vec<ArgPlace> {
    let mut ncrn = 0usize;
    let mut nsaa = 0usize;
    let mut vfp_used = [false; VFP_SINGLES];
    let mut vfp_closed = false;
    let mut places = Vec::with_capacity(kinds.len());

    for (index, kind) in kinds.iter().enumerate() {
        let variadic = base_standard || fixed.is_some_and(|fixed| index >= fixed);
        let mut place = ArgPlace::default();
        let co_processor = kind.float && !variadic;

        if co_processor && !vfp_closed {
            let need = if kind.double { 2 } else { 1 };
            let slot = (0..=VFP_SINGLES - need)
                .step_by(need)
                .find(|start| vfp_used[*start..start + need].iter().all(|used| !used));
            if let Some(start) = slot {
                for used in &mut vfp_used[start..start + need] {
                    *used = true;
                }
                place.vfp = Some(start);
                places.push(place);
                continue;
            }
            // Once one co-processor argument misses, the rest of the bank is closed.
            vfp_closed = true;
        }

        if kind.align >= 8 {
            ncrn = ncrn.div_ceil(2) * 2;
            nsaa = nsaa.div_ceil(8) * 8;
        }
        let words = kind.words();

        if co_processor {
            // A float that missed the bank goes to the stack, never to r0-r3.
            place.stack = Some(StackSpan {
                offset: nsaa,
                bytes: kind.bytes,
            });
            nsaa += words * 4;
        } else if ncrn + words <= CORE_ARGUMENT_REGISTERS {
            place.core = Some(CoreSpan { first: ncrn, words });
            ncrn += words;
        } else if ncrn < CORE_ARGUMENT_REGISTERS && nsaa == 0 {
            let in_registers = CORE_ARGUMENT_REGISTERS - ncrn;
            place.core = Some(CoreSpan {
                first: ncrn,
                words: in_registers,
            });
            place.stack = Some(StackSpan {
                offset: 0,
                bytes: (words - in_registers) * 4,
            });
            nsaa = (words - in_registers) * 4;
            ncrn = CORE_ARGUMENT_REGISTERS;
        } else {
            ncrn = CORE_ARGUMENT_REGISTERS;
            place.stack = Some(StackSpan {
                offset: nsaa,
                bytes: kind.bytes,
            });
            nsaa += words * 4;
        }
        places.push(place);
    }
    places
}

/// Outgoing stack bytes the places need, rounded to the eight-byte public interface.
pub(crate) fn stack_bytes(places: &[ArgPlace]) -> usize {
    let raw = places
        .iter()
        .filter_map(|place| place.stack.map(|span| span.offset + span.bytes.max(4)))
        .max()
        .unwrap_or(0);
    raw.div_ceil(8) * 8
}

pub(crate) fn returns_in_memory(aggregate: Option<Aggregate>) -> bool {
    aggregate.is_some_and(|aggregate| {
        let max = if aggregate.is_composite() {
            MAX_REGISTER_RETURN_COMPOSITE
        } else {
            MAX_REGISTER_RETURN_WIDE_SCALAR
        };
        aggregate.size > max
    })
}

/// Type a variadic argument reaches the callee as on a 32-bit target: a float travels
/// as a double the way C promotes one, everything else as a single word.
fn promoted(arg: &Operand) -> MirType {
    let float = match arg {
        Operand::Register(Register::Virtual(vreg)) => vreg.class == RegisterClass::Fpr,
        Operand::Immediate(Immediate::F32(_) | Immediate::F64(_)) => true,
        _ => false,
    };
    if float {
        MirType::Scalar(ScalarType::F64)
    } else {
        MirType::Scalar(ScalarType::I32)
    }
}

/// Each function's boundary as the convention sees it, so a call site knows how the
/// callee reads its arguments and hands its result back.
pub(crate) struct Signatures {
    defined: HashSet<String>,
    params: HashMap<String, Vec<ArgKind>>,
    returns: HashMap<String, Aggregate>,
    float_returns: HashMap<String, MirType>,
    variadic: HashSet<String>,
}

impl Signatures {
    pub(crate) fn from_module(module: &MirModule) -> Self {
        Self {
            defined: module
                .functions
                .iter()
                .filter(|(_, func)| !func.blocks.is_empty())
                .map(|(name, _)| name.clone())
                .collect(),
            params: module
                .functions
                .iter()
                .map(|(name, func)| {
                    (
                        name.clone(),
                        func.sig
                            .params
                            .iter()
                            .map(|p| ArgKind::of(p.ty, p.aggregate))
                            .collect(),
                    )
                })
                .collect(),
            returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| Some((name.clone(), func.sig.ret_aggregate?)))
                .collect(),
            float_returns: module
                .functions
                .iter()
                .filter_map(|(name, func)| {
                    let ty = func.sig.ret_ty?;
                    (func.sig.ret_aggregate.is_none() && ty.is_float()).then(|| (name.clone(), ty))
                })
                .collect(),
            variadic: module
                .functions
                .iter()
                .filter(|(_, func)| func.sig.variadic)
                .map(|(name, _)| name.clone())
                .collect(),
        }
    }

    /// The ARM run-time ABI hands the compiler runtime helpers their arguments and
    /// results in core registers whatever variant the rest of the program uses, so
    /// `__aeabi_d2lz` reads its double from r0:r1 and not from d0. Reserved `__` names
    /// the module does not define itself are those helpers.
    pub(crate) fn uses_base_standard(&self, callee: &str) -> bool {
        callee.starts_with("__") && !self.defined.contains(callee)
    }

    pub(crate) fn fixed_arguments(&self, callee: &str) -> Option<usize> {
        self.variadic
            .contains(callee)
            .then(|| self.params.get(callee).map_or(0, Vec::len))
    }

    pub(crate) fn arguments(&self, callee: &str, args: &[Operand]) -> Vec<ArgKind> {
        let fixed = self.fixed_arguments(callee);
        if fixed.is_none()
            && let Some(kinds) = self.params.get(callee).filter(|k| k.len() == args.len())
        {
            return kinds.clone();
        }
        let declared = self.params.get(callee);
        let fixed = fixed.unwrap_or(0);
        args.iter()
            .enumerate()
            .map(|(index, arg)| match declared {
                Some(kinds) if index < fixed && index < kinds.len() => kinds[index],
                _ => ArgKind::of(promoted(arg), None),
            })
            .collect()
    }

    pub(crate) fn result(&self, callee: &str) -> Option<Aggregate> {
        self.returns.get(callee).copied()
    }

    pub(crate) fn float_result(&self, callee: &str) -> Option<MirType> {
        self.float_returns.get(callee).copied()
    }
}
