use super::*;

pub(super) fn mapped_register(reg: &Register, alloc: &X64RegAlloc) -> Option<&'static str> {
    match reg {
        Register::Virtual(v) => alloc.get_mapping_for(v),
        Register::Physical(p) => Some(p.name),
    }
}

pub(super) fn integer_immediate(operand: &Operand) -> Option<i64> {
    match operand {
        Operand::Immediate(Immediate::I8(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I16(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I32(v)) => Some(i64::from(*v)),
        Operand::Immediate(Immediate::I64(v)) => Some(*v),
        _ => None,
    }
}

pub(super) fn select_lea(
    instructions: &[MirInst],
    alloc: &X64RegAlloc,
    uses: &HashMap<VirtualReg, usize>,
    flags_live: impl Fn(usize) -> bool,
) -> Option<(usize, String, &'static str)> {
    for length in (1..=instructions.len().min(3)).rev() {
        let MirInst::IntBinary {
            op: IntBinOp::Add,
            dst,
            lhs,
            rhs,
            ty,
        } = &instructions[length - 1]
        else {
            continue;
        };
        let Some(dst) = mapped_register(dst, alloc) else {
            continue;
        };
        if (0..length).any(&flags_live) || (length > 1 && ty.size_bytes() != 8) {
            continue;
        }
        let mut terms = Vec::new();
        let mut displacement = 0i64;
        let mut expanded = HashSet::new();
        fn collect(
            value: &Operand,
            prefix: &[MirInst],
            alloc: &X64RegAlloc,
            uses: &HashMap<VirtualReg, usize>,
            terms: &mut Vec<(&'static str, i64)>,
            displacement: &mut i64,
            expanded: &mut HashSet<usize>,
        ) -> Option<()> {
            if let Some(imm) = integer_immediate(value) {
                *displacement = displacement.checked_add(imm)?;
                return Some(());
            }
            let Operand::Register(reg) = value else {
                return None;
            };
            if let Register::Virtual(v) = reg
                && uses.get(v) == Some(&1)
                && let Some((
                    index,
                    MirInst::IntBinary {
                        op, lhs, rhs, ty, ..
                    },
                )) = prefix
                    .iter()
                    .enumerate()
                    .find(|(_, inst)| inst.def_reg() == Some(reg))
                && ty.size_bytes() == 8
            {
                match op {
                    IntBinOp::Add => {
                        collect(
                            lhs,
                            &prefix[..index],
                            alloc,
                            uses,
                            terms,
                            displacement,
                            expanded,
                        )?;
                        collect(
                            rhs,
                            &prefix[..index],
                            alloc,
                            uses,
                            terms,
                            displacement,
                            expanded,
                        )?;
                    }
                    IntBinOp::Shl => {
                        let shift = integer_immediate(rhs)?;
                        if !(1..=3).contains(&shift) {
                            return None;
                        }
                        let Operand::Register(src) = lhs else {
                            return None;
                        };
                        terms.push((mapped_register(src, alloc)?, 1 << shift));
                    }
                    _ => return None,
                }
                expanded.insert(index);
                return Some(());
            }
            terms.push((mapped_register(reg, alloc)?, 1));
            Some(())
        }
        let mut collect_operand = |value| {
            collect(
                value,
                &instructions[..length - 1],
                alloc,
                uses,
                &mut terms,
                &mut displacement,
                &mut expanded,
            )
        };
        if collect_operand(lhs).is_none()
            || collect_operand(rhs).is_none()
            || expanded.len() != length - 1
            || i32::try_from(displacement).is_err()
        {
            continue;
        }
        if length == 1 && terms.iter().any(|&(reg, _)| reg == dst) {
            continue;
        }
        let address = match terms.as_slice() {
            [(base, 1)] => format!("{displacement}(%{base})"),
            [(a, sa), (b, sb)] => {
                let (base, index, scale) = if *sa == 1 && *b != "rsp" {
                    (a, b, sb)
                } else if *sb == 1 && *a != "rsp" {
                    (b, a, sa)
                } else {
                    continue;
                };
                format!("{displacement}(%{base},%{index},{scale})")
            }
            _ => continue,
        };
        return Some((length, address, dst));
    }
    None
}

/// Narrow integers in 64-bit registers are sign-extended; loads read only their type width.
/// Value for `cmpq $imm, %rax`, or None when the operand is not an immediate that fits
/// a sign-extended imm32. Unsigned compares see the operand zero-extended to the type
/// width, matching the `extend_integer` the register path performs.
pub(super) fn cmp_immediate(rhs: &Operand, ty: &MirType, unsigned: bool) -> Option<i64> {
    let Operand::Immediate(imm) = rhs else {
        return None;
    };
    let raw = match imm {
        Immediate::I8(v) => i64::from(*v),
        Immediate::I16(v) => i64::from(*v),
        Immediate::I32(v) => i64::from(*v),
        Immediate::I64(v) => *v,
        _ => return None,
    };
    let value = if unsigned {
        match ty.size_bytes() {
            1 => raw & 0xff,
            2 => raw & 0xffff,
            4 => raw & 0xffff_ffff,
            _ => raw,
        }
    } else {
        raw
    };
    i32::try_from(value).ok().map(i64::from)
}
