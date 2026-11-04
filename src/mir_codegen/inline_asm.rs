//! The machinery the `asm.T` emitters share: template expansion, operand binding, and
//! the canonical form of a result.
//!
//! A backend supplies its own facts — which spellings name which physical register,
//! which registers it refuses and why, the order a `reg` operand tries the pool, and how
//! a template spells a bound register — and this module runs the contract over them. The
//! facts stay in the backend so that reading one backend still says what that backend
//! does; only the mechanism lives here.

pub(crate) use crate::mir::asm::GPR_CLASS;
use crate::{
    error::LaminaError,
    mir::{AsmOperand, MirType, ScalarType},
};

/// The physical register bound to one operand, and how the template spells it.
pub(crate) struct Binding {
    pub(crate) phys: &'static str,
    pub(crate) spelled: String,
}

/// One backend's register facts, which is everything binding needs that is not mechanism.
pub(crate) struct RegisterFacts {
    /// The physical register an operand spelling names, or why the backend refuses it.
    pub(crate) operand: fn(&str) -> Result<&'static str, LaminaError>,
    /// The physical register a clobber occupies, or `None` for a clobber that occupies
    /// none an operand could be bound to.
    pub(crate) clobber: fn(&str) -> Result<Option<&'static str>, LaminaError>,
    /// How a template spells the register an operand was bound to.
    pub(crate) spell: fn(&str) -> String,
    /// The registers a `reg` operand may be given, in the order they are tried.
    pub(crate) pool: &'static [&'static str],
}

/// A register spelled with a leading `%`, as the AT&T and s390x syntaxes write it.
pub(crate) fn spell_percent(name: &str) -> String {
    format!("%{name}")
}

/// A register spelled bare, as the ARM syntaxes write it.
pub(crate) fn spell_bare(name: &str) -> String {
    name.to_owned()
}

/// The two clobbers that name no register.
fn is_pseudo_clobber(clobber: &str) -> bool {
    matches!(clobber, "memory" | "flags")
}

pub(crate) fn location(operand: &AsmOperand) -> &str {
    match operand {
        AsmOperand::Input { location, .. } | AsmOperand::Output { location } => location,
    }
}

impl RegisterFacts {
    /// Binds every operand to a physical register. Named operands keep their spelling;
    /// `reg` operands take the first pool register that no named operand or clobber uses.
    pub(crate) fn bind(
        &self,
        operands: &[AsmOperand],
        clobbers: &[String],
    ) -> Result<Vec<Binding>, LaminaError> {
        let mut taken: Vec<&'static str> = Vec::new();
        let mut bindings: Vec<Option<Binding>> = Vec::with_capacity(operands.len());
        for operand in operands {
            bindings.push(self.bind_named(location(operand), &mut taken)?);
        }
        for clobber in clobbers {
            if is_pseudo_clobber(clobber) {
                continue;
            }
            if let Some(phys) = (self.clobber)(clobber)?
                && !taken.contains(&phys)
            {
                taken.push(phys);
            }
        }
        self.fill_class_operands(bindings, taken)
    }

    /// `None` for a `reg`-class operand, which is bound once the clobbers are known.
    fn bind_named(
        &self,
        name: &str,
        taken: &mut Vec<&'static str>,
    ) -> Result<Option<Binding>, LaminaError> {
        if name == GPR_CLASS {
            return Ok(None);
        }
        let phys = (self.operand)(name)?;
        if taken.contains(&phys) {
            return Err(LaminaError::ValidationError(format!(
                "register '{name}' is used by two inline asm operands"
            )));
        }
        taken.push(phys);
        Ok(Some(Binding {
            phys,
            spelled: (self.spell)(name),
        }))
    }

    fn fill_class_operands(
        &self,
        bindings: Vec<Option<Binding>>,
        mut taken: Vec<&'static str>,
    ) -> Result<Vec<Binding>, LaminaError> {
        let mut resolved = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if let Some(binding) = binding {
                resolved.push(binding);
                continue;
            }
            let phys = self
                .pool
                .iter()
                .copied()
                .find(|candidate| !taken.contains(candidate))
                .ok_or_else(|| {
                    LaminaError::ValidationError(
                        "no free register for an inline asm 'reg' operand".to_owned(),
                    )
                })?;
            taken.push(phys);
            resolved.push(Binding {
                phys,
                spelled: (self.spell)(phys),
            });
        }
        Ok(resolved)
    }

    /// The physical registers a clobber list names, in the order given.
    pub(crate) fn clobbered_registers(
        &self,
        clobbers: &[String],
    ) -> Result<Vec<&'static str>, LaminaError> {
        let mut regs = Vec::new();
        for clobber in clobbers {
            if is_pseudo_clobber(clobber) {
                continue;
            }
            if let Some(phys) = (self.clobber)(clobber)?
                && !regs.contains(&phys)
            {
                regs.push(phys);
            }
        }
        Ok(regs)
    }
}

/// Replaces `$n` with the spelling of operand `n` and `$$` with `$`, and decodes the
/// escapes `\n`, `\t`, `\\` and `\"`. Everything else is passed through untouched.
pub(crate) fn expand_template(template: &str, bindings: &[Binding]) -> Result<String, LaminaError> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '$' => {
                if chars.peek() == Some(&'$') {
                    chars.next();
                    out.push('$');
                    continue;
                }
                let mut digits = String::new();
                while let Some(d) = chars.peek().copied().filter(char::is_ascii_digit) {
                    digits.push(d);
                    chars.next();
                }
                let index: usize = digits.parse().map_err(|_| {
                    LaminaError::ValidationError("malformed `$` in inline asm template".to_owned())
                })?;
                let binding = bindings.get(index).ok_or_else(|| {
                    LaminaError::ValidationError(format!(
                        "inline asm template uses ${index}, but the instruction has {} operand(s)",
                        bindings.len()
                    ))
                })?;
                out.push_str(&binding.spelled);
            }
            '\\' => push_escape(&mut out, chars.next()),
            other => out.push(other),
        }
    }
    Ok(out)
}

fn push_escape(out: &mut String, escaped: Option<char>) {
    match escaped {
        Some('n') => out.push('\n'),
        Some('t') => out.push('\t'),
        Some('\\') => out.push('\\'),
        Some('"') => out.push('"'),
        Some(other) => {
            out.push('\\');
            out.push(other);
        }
        None => out.push('\\'),
    }
}

/// The register an `asm.T` reads its result from: the one bound to the `out(...)` operand.
pub(crate) fn output_register(
    operands: &[AsmOperand],
    bindings: &[Binding],
) -> Result<&'static str, LaminaError> {
    operands
        .iter()
        .zip(bindings)
        .find(|(operand, _)| matches!(operand, AsmOperand::Output { .. }))
        .map(|(_, binding)| binding.phys)
        .ok_or_else(|| {
            LaminaError::ValidationError("asm.T instruction has no out(...) operand".to_owned())
        })
}

/// Whether a result is the one type that is widened unsigned. A narrow result is widened
/// into its canonical in-register form, which is sign-extended for every integer type
/// whatever its declared signedness; `i1` is the exception and carries only 0 or 1.
pub(crate) fn result_is_bool(ty: &MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::I1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Operand, Register, VirtualReg};

    fn named(name: &str) -> Result<&'static str, LaminaError> {
        ["a", "b", "c"]
            .into_iter()
            .find(|candidate| *candidate == name)
            .ok_or_else(|| LaminaError::ValidationError(format!("unknown register '{name}'")))
    }

    fn clobber(name: &str) -> Result<Option<&'static str>, LaminaError> {
        if name == "vector" {
            return Ok(None);
        }
        named(name).map(Some)
    }

    const FACTS: RegisterFacts = RegisterFacts {
        operand: named,
        clobber,
        spell: spell_percent,
        pool: &["a", "b", "c"],
    };

    fn out(location: &str) -> AsmOperand {
        AsmOperand::Output {
            location: location.to_owned(),
        }
    }

    fn input(location: &str) -> AsmOperand {
        AsmOperand::Input {
            location: location.to_owned(),
            value: Operand::Register(Register::Virtual(VirtualReg::gpr(0))),
        }
    }

    fn spellings(operands: &[AsmOperand], clobbers: &[String]) -> Vec<String> {
        FACTS
            .bind(operands, clobbers)
            .expect("bind")
            .into_iter()
            .map(|binding| binding.spelled)
            .collect()
    }

    #[test]
    fn a_class_operand_skips_named_operands_and_clobbers() {
        assert_eq!(
            spellings(&[input("a"), out(GPR_CLASS)], &["b".to_owned()]),
            vec!["%a".to_owned(), "%c".to_owned()]
        );
    }

    #[test]
    fn the_pseudo_clobbers_take_no_register() {
        assert_eq!(
            spellings(
                &[out(GPR_CLASS)],
                &["memory".to_owned(), "flags".to_owned()]
            ),
            vec!["%a".to_owned()]
        );
    }

    #[test]
    fn a_clobber_that_occupies_no_register_leaves_the_pool_alone() {
        assert_eq!(
            spellings(&[out(GPR_CLASS)], &["vector".to_owned()]),
            vec!["%a".to_owned()]
        );
    }

    #[test]
    fn two_operands_may_not_name_one_register() {
        assert!(FACTS.bind(&[input("a"), out("a")], &[]).is_err());
    }

    #[test]
    fn an_exhausted_pool_is_refused() {
        let clobbers = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert!(FACTS.bind(&[out(GPR_CLASS)], &clobbers).is_err());
    }

    #[test]
    fn an_unknown_operand_or_clobber_is_refused() {
        assert!(FACTS.bind(&[out("z")], &[]).is_err());
        assert!(FACTS.bind(&[], &["z".to_owned()]).is_err());
    }

    #[test]
    fn clobbered_registers_drops_the_pseudo_clobbers_and_duplicates() {
        let clobbers = vec![
            "b".to_owned(),
            "memory".to_owned(),
            "b".to_owned(),
            "vector".to_owned(),
            "a".to_owned(),
        ];
        assert_eq!(
            FACTS.clobbered_registers(&clobbers).expect("clobbers"),
            vec!["b", "a"]
        );
    }

    #[test]
    fn a_template_substitutes_operands_and_decodes_escapes() {
        let bindings = FACTS.bind(&[input("a"), out("b")], &[]).expect("bind");
        assert_eq!(
            expand_template("$0 $1\\t$$1\\n\\\\\\\"\\q", &bindings).expect("expand"),
            "%a %b\t$1\n\\\"\\q"
        );
    }

    #[test]
    fn a_trailing_backslash_is_kept() {
        assert_eq!(expand_template("x\\", &[]).expect("expand"), "x\\");
    }

    #[test]
    fn a_template_operand_past_the_end_is_refused() {
        let bindings = FACTS.bind(&[out("a")], &[]).expect("bind");
        assert!(expand_template("$1", &bindings).is_err());
        assert!(expand_template("$x", &bindings).is_err());
    }

    #[test]
    fn the_output_register_is_the_one_bound_to_the_out_operand() {
        let operands = [input("a"), out("b")];
        let bindings = FACTS.bind(&operands, &[]).expect("bind");
        assert_eq!(output_register(&operands, &bindings).expect("output"), "b");
        let inputs = [input("a")];
        let only_inputs = FACTS.bind(&inputs, &[]).expect("bind");
        assert!(output_register(&inputs, &only_inputs).is_err());
    }

    #[test]
    fn only_a_boolean_result_is_widened_unsigned() {
        assert!(result_is_bool(&MirType::Scalar(ScalarType::I1)));
        for ty in [
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
        ] {
            assert!(!result_is_bool(&MirType::Scalar(ty)));
        }
    }
}
