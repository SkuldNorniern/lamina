//! Block-local alias queries over MIR loads and stores. Design: Plan/alias-analysis-design.md.
//! Tokens name values, not registers, and facts never cross a block boundary.

use crate::mir::{
    AddressMode, AddressSpace, Block, Instruction, MirType, Operand, Register, ScalarType,
    VectorType, VirtualReg,
};
use std::collections::HashMap;
use std::mem;

/// Largest displacement or width for which interval arithmetic on one value is trusted.
/// Keeps every sum far below 2^32, so an interval test also holds for 32-bit pointers.
const MAX_VALUE_SPAN: i64 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AliasResult {
    NoAlias,
    MayAlias,
    SameRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Base {
    Frame { slot: VirtualReg, reserved: i64 },
    Value(u32),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Address {
    pub base: Base,
    pub disp: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Access {
    pub base: Base,
    pub disp: i64,
    pub width: i64,
    pub space: AddressSpace,
}

impl Access {
    fn contained(&self) -> bool {
        match self.base {
            Base::Frame { reserved, .. } => {
                self.disp >= 0
                    && self
                        .disp
                        .checked_add(self.width)
                        .is_some_and(|end| end <= reserved)
            }
            _ => false,
        }
    }

    pub(super) fn base_known(&self) -> bool {
        self.base != Base::Unknown
    }

    fn within_span(&self) -> bool {
        self.disp.abs() <= MAX_VALUE_SPAN && self.width <= MAX_VALUE_SPAN
    }
}

fn disjoint(a: &Access, b: &Access) -> bool {
    a.disp + a.width <= b.disp || b.disp + b.width <= a.disp
}

pub(super) fn alias(a: &Access, b: &Access) -> AliasResult {
    if a.space != b.space || a.width <= 0 || b.width <= 0 {
        return AliasResult::MayAlias;
    }
    if a.base == Base::Unknown || b.base == Base::Unknown {
        return AliasResult::MayAlias;
    }
    if a.base == b.base && a.disp == b.disp && a.width == b.width {
        return AliasResult::SameRange;
    }
    match (a.base, b.base) {
        (Base::Frame { slot: left, .. }, Base::Frame { slot: right, .. }) => {
            if !a.contained() || !b.contained() {
                AliasResult::MayAlias
            } else if left != right || disjoint(a, b) {
                AliasResult::NoAlias
            } else {
                AliasResult::MayAlias
            }
        }
        (Base::Value(left), Base::Value(right)) if left == right => {
            if a.within_span() && b.within_span() && disjoint(a, b) {
                AliasResult::NoAlias
            } else {
                AliasResult::MayAlias
            }
        }
        _ => AliasResult::MayAlias,
    }
}

/// Slot bytes by `sroa`'s convention. Zero on overflow, so nothing is contained in it.
fn frame_extent(words: Option<&usize>) -> i64 {
    words
        .copied()
        .unwrap_or(0)
        .checked_add(1)
        .and_then(|units| i64::try_from(units).ok())
        .and_then(|units| units.checked_mul(8))
        .unwrap_or(0)
}

/// Bytes one access touches. Pointers count as eight bytes, the widest pointer any target
/// uses, so a pointer access is never narrower than it really is.
fn access_width(ty: MirType) -> i64 {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => 1,
        MirType::Scalar(ScalarType::I16) => 2,
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => 4,
        MirType::Scalar(
            ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr | ScalarType::PtrIn(_),
        ) => 8,
        MirType::Vector(VectorType::V128(_)) => 16,
        MirType::Vector(VectorType::V256(_)) => 32,
    }
}

fn pointer_typed(ty: MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::Ptr | ScalarType::PtrIn(_)))
}

struct Scan<'a> {
    regs: HashMap<Register, Address>,
    next_token: u32,
    reservations: &'a HashMap<VirtualReg, usize>,
}

impl Scan<'_> {
    fn fresh(&mut self) -> Address {
        let base = match self.next_token.checked_add(1) {
            Some(next) => Base::Value(mem::replace(&mut self.next_token, next)),
            None => Base::Unknown,
        };
        Address { base, disp: 0 }
    }

    fn value_of(&mut self, reg: &Register) -> Address {
        if let Some(address) = self.regs.get(reg) {
            return *address;
        }
        let address = self.fresh();
        self.regs.insert(reg.clone(), address);
        address
    }

    fn access(&mut self, addr: &AddressMode, ty: MirType) -> Access {
        let width = access_width(ty);
        match addr {
            AddressMode::BaseOffset {
                base,
                offset,
                address_space,
            } => {
                let held = self.value_of(base);
                let (base, disp) = match (held.base, held.disp.checked_add(i64::from(*offset))) {
                    (Base::Frame { .. }, _) if *address_space != AddressSpace::default() => {
                        (Base::Unknown, 0)
                    }
                    (base, Some(disp)) => (base, disp),
                    (_, None) => (Base::Unknown, 0),
                };
                Access {
                    base,
                    disp,
                    width,
                    space: *address_space,
                }
            }
            AddressMode::BaseIndexScale { address_space, .. } => Access {
                base: Base::Unknown,
                disp: 0,
                width,
                space: *address_space,
            },
        }
    }

    fn defined_address(&mut self, inst: &Instruction) -> Address {
        match inst {
            Instruction::StackAddr { slot, .. } => Address {
                base: Base::Frame {
                    slot: *slot,
                    reserved: frame_extent(self.reservations.get(slot)),
                },
                disp: 0,
            },
            Instruction::Lea { base, offset, .. } => {
                let held = self.value_of(base);
                match held.disp.checked_add(i64::from(*offset)) {
                    Some(disp) => Address { disp, ..held },
                    None => self.fresh(),
                }
            }
            Instruction::Copy {
                ty,
                src: Operand::Register(src),
                ..
            } if pointer_typed(*ty) => self.value_of(src),
            _ => self.fresh(),
        }
    }

    fn step(&mut self, inst: &Instruction) -> Option<Access> {
        let access = match inst {
            Instruction::Load { ty, addr, .. } | Instruction::Store { ty, addr, .. } => {
                Some(self.access(addr, *ty))
            }
            _ => None,
        };
        if let Some(dst) = inst.def_reg() {
            let address = self.defined_address(inst);
            self.regs.insert(dst.clone(), address);
        }
        access
    }
}

/// The access of each instruction at its own point. Uses are read before the
/// instruction's definition, so `p = lea p, 8` sees the old `p`.
pub(super) fn scan_block(
    block: &Block,
    reservations: &HashMap<VirtualReg, usize>,
) -> Vec<Option<Access>> {
    let mut scan = Scan {
        regs: HashMap::new(),
        next_token: 0,
        reservations,
    };
    block
        .instructions
        .iter()
        .map(|inst| scan.step(inst))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Block, Immediate, MemoryAttrs};
    use lamina_mir::Located;

    fn slot(id: u32) -> VirtualReg {
        VirtualReg::gpr(100 + id)
    }

    fn frame(id: u32, reserved: i64) -> Base {
        Base::Frame {
            slot: slot(id),
            reserved,
        }
    }

    fn access(base: Base, disp: i64, width: i64) -> Access {
        Access {
            base,
            disp,
            width,
            space: AddressSpace::default(),
        }
    }

    #[test]
    fn query_table_refuses_uncertain_pairs() {
        let a16 = frame(0, 16);
        let b16 = frame(1, 16);
        let cases = [
            (access(a16, 0, 8), access(a16, 8, 8), AliasResult::NoAlias),
            (access(a16, 0, 8), access(a16, 4, 8), AliasResult::MayAlias),
            (access(a16, 0, 8), access(a16, 0, 8), AliasResult::SameRange),
            (access(a16, 0, 8), access(a16, 0, 4), AliasResult::MayAlias),
            (access(a16, 0, 8), access(b16, 0, 8), AliasResult::NoAlias),
            (access(a16, 12, 8), access(b16, 0, 8), AliasResult::MayAlias),
            (access(a16, -8, 8), access(b16, 0, 8), AliasResult::MayAlias),
            (
                access(a16, 0, 8),
                access(Base::Value(0), 0, 8),
                AliasResult::MayAlias,
            ),
            (
                access(Base::Value(3), 0, 8),
                access(Base::Value(3), 8, 8),
                AliasResult::NoAlias,
            ),
            (
                access(Base::Value(3), 0, 8),
                access(Base::Value(3), 4, 8),
                AliasResult::MayAlias,
            ),
            (
                access(Base::Value(3), 0, 8),
                access(Base::Value(4), 8, 8),
                AliasResult::MayAlias,
            ),
            (
                access(Base::Value(3), 0, 8),
                access(Base::Value(3), 0, 8),
                AliasResult::SameRange,
            ),
            (
                access(Base::Value(3), 0, 8),
                access(Base::Unknown, 8, 8),
                AliasResult::MayAlias,
            ),
            (
                access(a16, 0, 8),
                access(Base::Unknown, 0, 8),
                AliasResult::MayAlias,
            ),
            (
                access(Base::Value(3), 1 << 30, 8),
                access(Base::Value(3), 0, 8),
                AliasResult::MayAlias,
            ),
            (
                Access {
                    space: AddressSpace::Global,
                    ..access(a16, 0, 8)
                },
                access(b16, 8, 8),
                AliasResult::MayAlias,
            ),
        ];
        for (left, right, expected) in cases {
            assert_eq!(alias(&left, &right), expected, "{left:?} vs {right:?}");
            assert_eq!(
                alias(&right, &left),
                expected,
                "symmetry: {right:?} vs {left:?}"
            );
        }
    }

    #[test]
    fn exclusive_results_never_both_hold() {
        let bases = [
            frame(0, 16),
            frame(1, 8),
            Base::Value(0),
            Base::Value(1),
            Base::Unknown,
        ];
        for left_base in bases {
            for right_base in bases {
                for disp in [-4, 0, 4, 8, 12, 16] {
                    for width in [1, 4, 8] {
                        let left = access(left_base, 0, 8);
                        let right = access(right_base, disp, width);
                        let result = alias(&left, &right);
                        assert_eq!(result, alias(&right, &left));
                        if result == AliasResult::NoAlias {
                            match (left_base, right_base) {
                                (Base::Frame { slot: x, .. }, Base::Frame { slot: y, .. })
                                    if x != y => {}
                                _ => assert!(left_base == right_base && disjoint(&left, &right)),
                            }
                        }
                    }
                }
            }
        }
    }

    fn load(dst: u32, base: u32, offset: i16) -> Instruction {
        Instruction::Load {
            ty: MirType::Scalar(ScalarType::I64),
            dst: VirtualReg::gpr(dst).into(),
            addr: AddressMode::BaseOffset {
                base: VirtualReg::gpr(base).into(),
                offset,
                address_space: AddressSpace::default(),
            },
            attrs: MemoryAttrs::default(),
        }
    }

    fn store(base: u32, offset: i16) -> Instruction {
        Instruction::Store {
            ty: MirType::Scalar(ScalarType::I64),
            src: Operand::Immediate(Immediate::I64(1)),
            addr: AddressMode::BaseOffset {
                base: VirtualReg::gpr(base).into(),
                offset,
                address_space: AddressSpace::default(),
            },
            attrs: MemoryAttrs::default(),
        }
    }

    fn scan(instructions: Vec<Instruction>, reservations: &[(u32, usize)]) -> Vec<Option<Access>> {
        let block = Block {
            label: "entry".to_owned(),
            instructions: instructions.into_iter().map(Located::from).collect(),
        };
        let reservations: HashMap<VirtualReg, usize> =
            reservations.iter().map(|&(id, w)| (slot(id), w)).collect();
        scan_block(&block, &reservations)
    }

    #[test]
    fn stack_slot_fields_are_disjoint_through_copies_and_lea() {
        let accesses = scan(
            vec![
                Instruction::StackAddr {
                    dst: VirtualReg::gpr(0).into(),
                    slot: slot(0),
                },
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::Ptr),
                    dst: VirtualReg::gpr(1).into(),
                    src: Operand::Register(VirtualReg::gpr(0).into()),
                },
                Instruction::Lea {
                    dst: VirtualReg::gpr(2).into(),
                    base: VirtualReg::gpr(1).into(),
                    offset: 8,
                },
                store(0, 0),
                store(2, 0),
                load(3, 1, 8),
            ],
            &[(0, 1)],
        );
        assert_eq!(accesses[3].map(|a| a.disp), Some(0));
        assert_eq!(accesses[4].map(|a| a.disp), Some(8));
        assert_eq!(accesses[5].map(|a| a.disp), Some(8));
        assert_eq!(
            alias(
                &accesses[3].unwrap_or(access(Base::Unknown, 0, 8)),
                &accesses[4].unwrap_or(access(Base::Unknown, 0, 8))
            ),
            AliasResult::NoAlias
        );
        assert_eq!(
            alias(
                &accesses[4].unwrap_or(access(Base::Unknown, 0, 8)),
                &accesses[5].unwrap_or(access(Base::Unknown, 0, 8))
            ),
            AliasResult::SameRange
        );
    }

    #[test]
    fn redefinition_changes_the_value_token() {
        let accesses = scan(vec![store(0, 0), load(0, 1, 0), store(0, 0)], &[]);
        let first = accesses[0].unwrap_or(access(Base::Unknown, 0, 8));
        let second = accesses[2].unwrap_or(access(Base::Unknown, 0, 8));
        assert_eq!(alias(&first, &second), AliasResult::MayAlias);
    }

    #[test]
    fn self_update_reads_the_old_value() {
        let accesses = scan(
            vec![
                Instruction::Lea {
                    dst: VirtualReg::gpr(0).into(),
                    base: VirtualReg::gpr(0).into(),
                    offset: 8,
                },
                store(0, 0),
            ],
            &[],
        );
        assert_eq!(accesses[1].map(|a| a.disp), Some(8));
        let before = scan(
            vec![
                store(0, 0),
                Instruction::Lea {
                    dst: VirtualReg::gpr(0).into(),
                    base: VirtualReg::gpr(0).into(),
                    offset: 8,
                },
                store(0, 0),
            ],
            &[],
        );
        let first = before[0].unwrap_or(access(Base::Unknown, 0, 8));
        let second = before[2].unwrap_or(access(Base::Unknown, 0, 8));
        assert_eq!(first.base, second.base, "lea keeps the value token");
        assert_eq!(alias(&first, &second), AliasResult::NoAlias);
    }

    #[test]
    fn integer_copy_does_not_share_an_address() {
        let accesses = scan(
            vec![
                store(0, 0),
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: VirtualReg::gpr(1).into(),
                    src: Operand::Register(VirtualReg::gpr(0).into()),
                },
                store(1, 0),
            ],
            &[],
        );
        let first = accesses[0].unwrap_or(access(Base::Unknown, 0, 8));
        let second = accesses[2].unwrap_or(access(Base::Unknown, 0, 8));
        assert_eq!(alias(&first, &second), AliasResult::MayAlias);
    }

    #[test]
    fn non_default_space_on_a_frame_slot_is_unknown() {
        let accesses = scan(
            vec![
                Instruction::StackAddr {
                    dst: VirtualReg::gpr(0).into(),
                    slot: slot(0),
                },
                Instruction::Store {
                    ty: MirType::Scalar(ScalarType::I64),
                    src: Operand::Immediate(Immediate::I64(1)),
                    addr: AddressMode::BaseOffset {
                        base: VirtualReg::gpr(0).into(),
                        offset: 0,
                        address_space: AddressSpace::Global,
                    },
                    attrs: MemoryAttrs::default(),
                },
            ],
            &[(0, 0)],
        );
        assert_eq!(accesses[1].map(|a| a.base), Some(Base::Unknown));
    }
}
