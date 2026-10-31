//! Inline assembly for s390x Linux.
//!
//! Operands are bound to physical registers around the template: inputs are loaded from
//! their frame slots into the bound register, the template is emitted with `$n` replaced
//! by the register of operand `n`, and the output register is written back to the result
//! slot. Nothing the emitter produces keeps a value in a register across an instruction,
//! so a clobber costs nothing and needs no reporting; the prologue already saves `%r6`
//! through `%r15`. The three registers that are live across the body — the frame pointer,
//! the stack pointer and the saved entry stack pointer — are refused instead.

use super::abi::A0;
use super::extend_integer;
use super::operand::{Slots, load_float_operand, load_operand, store_register, unsupported};
use crate::{
    error::LaminaError,
    mir::{AsmOperand, Instruction as MirInst, MirType, Register, ScalarType},
};
use std::io::Write;

/// The register class accepted in place of a named register.
const GPR_CLASS: &str = "reg";

const GPRS: [&str; 16] = [
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
    "r15",
];

const FPRS: [&str; 16] = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
    "f15",
];

/// Registers the emitted body depends on between instructions: the frame pointer, the
/// stack pointer, and the entry stack pointer an over-aligned frame keeps aside.
const RESERVED: [&str; 3] = ["r9", "r11", "r15"];

/// Registers a `reg` operand may be given, in the order they are tried.
const REG_CLASS_POOL: [&str; 8] = ["r2", "r3", "r4", "r5", "r1", "r10", "r12", "r13"];

/// The physical register bound to one operand, and how the template spells it.
pub(super) struct Binding {
    reg: &'static str,
    spelled: String,
}

fn is_fpr(reg: &str) -> bool {
    FPRS.contains(&reg)
}

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for s390x; valid names are: {GPR_CLASS}, \
         {}, {}",
        GPRS.join(", "),
        FPRS.join(", ")
    ))
}

/// Resolves a register name written in an operand or a clobber to its canonical spelling.
fn named_register(name: &str) -> Result<&'static str, LaminaError> {
    if RESERVED.contains(&name) {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' is reserved by the s390x backend and cannot be named by inline \
             asm; %r11 is the frame pointer, %r15 the stack pointer and %r9 the entry stack \
             pointer"
        )));
    }
    GPRS.iter()
        .chain(FPRS.iter())
        .copied()
        .find(|candidate| *candidate == name)
        .ok_or_else(|| unknown_register(name))
}

fn location(operand: &AsmOperand) -> &str {
    match operand {
        AsmOperand::Input { location, .. } | AsmOperand::Output { location } => location,
    }
}

/// Binds every operand to a physical register. Named operands keep their spelling; `reg`
/// operands take the first pool register that no named operand or clobber uses.
pub(super) fn bind(
    operands: &[AsmOperand],
    clobbers: &[String],
) -> Result<Vec<Binding>, LaminaError> {
    let mut taken: Vec<&'static str> = Vec::new();
    let mut bindings: Vec<Option<Binding>> = Vec::with_capacity(operands.len());
    for operand in operands {
        let name = location(operand);
        if name == GPR_CLASS {
            bindings.push(None);
            continue;
        }
        let reg = named_register(name)?;
        if taken.contains(&reg) {
            return Err(LaminaError::ValidationError(format!(
                "register '{name}' is used by two inline asm operands"
            )));
        }
        taken.push(reg);
        bindings.push(Some(Binding {
            reg,
            spelled: format!("%{reg}"),
        }));
    }
    for clobber in clobbers {
        if matches!(clobber.as_str(), "memory" | "flags") {
            continue;
        }
        let reg = named_register(clobber)?;
        if !taken.contains(&reg) {
            taken.push(reg);
        }
    }
    let mut resolved = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if let Some(binding) = binding {
            resolved.push(binding);
            continue;
        }
        let reg = REG_CLASS_POOL
            .iter()
            .copied()
            .find(|candidate| !taken.contains(candidate))
            .ok_or_else(|| {
                LaminaError::ValidationError(
                    "no free register for an inline asm 'reg' operand".to_owned(),
                )
            })?;
        taken.push(reg);
        resolved.push(Binding {
            reg,
            spelled: format!("%{reg}"),
        });
    }
    Ok(resolved)
}

/// Replaces `$n` with the spelling of operand `n` and `$$` with `$`, and decodes the
/// escapes `\n`, `\t`, `\\` and `\"`. Everything else is passed through untouched.
fn expand_template(template: &str, bindings: &[Binding]) -> Result<String, LaminaError> {
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
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            },
            other => out.push(other),
        }
    }
    Ok(out)
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    slots: &Slots,
) -> Result<(), LaminaError> {
    let MirInst::InlineAsm {
        template,
        operands,
        clobbers,
        result,
        ..
    } = inst
    else {
        return Err(unsupported(
            "s390x inline assembly emitter received another instruction",
        ));
    };
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;

    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            let dst = format!("%{}", binding.reg);
            if is_fpr(binding.reg) {
                load_float_operand(value, writer, slots, &dst, false)?;
            } else {
                load_operand(value, writer, slots, &dst)?;
            }
        }
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(writer, "    {line}")?;
        }
    }

    let Some((dst, ty)) = result else {
        return Ok(());
    };
    let output = operands
        .iter()
        .zip(&bindings)
        .find(|(operand, _)| matches!(operand, AsmOperand::Output { .. }))
        .map(|(_, binding)| binding.reg)
        .ok_or_else(|| {
            LaminaError::ValidationError("asm.T instruction has no out(...) operand".to_owned())
        })?;
    if is_fpr(output) {
        return Err(LaminaError::ValidationError(format!(
            "inline asm result of type {ty} is read from the floating-point register \
             '%{output}'; an s390x asm result must come from a general register"
        )));
    }
    let source = format!("%{output}");
    if source != A0 {
        writeln!(writer, "    lgr {A0}, {source}")?;
    }
    extend_integer(writer, A0, ty, signed_result(ty))?;
    match dst {
        Register::Virtual(vreg) => store_register(A0, vreg, writer, slots),
        Register::Physical(physical) => {
            let name = super::phys(physical.name);
            if name != A0 {
                writeln!(writer, "    lgr {name}, {A0}")?;
            }
            Ok(())
        }
    }
}

/// How a result narrower than a doubleword is widened into its slot, matching the
/// canonical form the rest of the backend keeps: `i1` zero-extended, the rest signed.
fn signed_result(ty: &MirType) -> bool {
    !matches!(ty, MirType::Scalar(ScalarType::I1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Operand, VirtualReg};
    use std::collections::HashMap;

    fn out(location: &str) -> AsmOperand {
        AsmOperand::Output {
            location: location.to_owned(),
        }
    }

    fn input(location: &str, vreg: u32) -> AsmOperand {
        AsmOperand::Input {
            location: location.to_owned(),
            value: Operand::Register(Register::Virtual(VirtualReg::gpr(vreg))),
        }
    }

    fn spellings(operands: &[AsmOperand], clobbers: &[String]) -> Vec<String> {
        bind(operands, clobbers)
            .expect("bind")
            .into_iter()
            .map(|binding| binding.spelled)
            .collect()
    }

    #[test]
    fn s390x_named_operands_keep_their_registers() {
        assert_eq!(
            spellings(&[input("r3", 0), out("r4")], &[]),
            vec!["%r3".to_owned(), "%r4".to_owned()]
        );
    }

    #[test]
    fn s390x_reg_class_operands_avoid_named_operands_and_clobbers() {
        let clobbers = vec!["r3".to_owned(), "memory".to_owned()];
        assert_eq!(
            spellings(&[input("r2", 0), out(GPR_CLASS)], &clobbers),
            vec!["%r2".to_owned(), "%r4".to_owned()]
        );
    }

    #[test]
    fn s390x_inline_asm_refuses_the_frame_and_stack_pointers() {
        for name in ["r9", "r11", "r15"] {
            assert!(bind(&[out(name)], &[]).is_err());
            assert!(bind(&[], &[name.to_owned()]).is_err());
        }
    }

    #[test]
    fn s390x_inline_asm_refuses_x86_register_names() {
        assert!(bind(&[out("rax")], &[]).is_err());
        assert!(bind(&[], &["rdx".to_owned()]).is_err());
    }

    #[test]
    fn s390x_inline_asm_refuses_one_register_for_two_operands() {
        assert!(bind(&[input("r2", 0), out("r2")], &[]).is_err());
    }

    #[test]
    fn s390x_inline_asm_accepts_floating_point_registers() {
        assert_eq!(spellings(&[input("f4", 0)], &[]), vec!["%f4".to_owned()]);
    }

    #[test]
    fn s390x_inline_asm_template_substitutes_operands_and_escapes() {
        let bindings = bind(&[input("r2", 0), out("r3")], &[]).expect("bind");
        assert_eq!(
            expand_template("agr $1, $0\\nlghi $0, $$8", &bindings).expect("expand"),
            "agr %r3, %r2\nlghi %r2, $8"
        );
    }

    #[test]
    fn s390x_inline_asm_template_rejects_an_operand_past_the_end() {
        let bindings = bind(&[out("r2")], &[]).expect("bind");
        assert!(expand_template("lghi $1, 0", &bindings).is_err());
    }

    fn emit_to_string(inst: &MirInst, slots: &Slots) -> Result<String, LaminaError> {
        let mut out = Vec::new();
        emit(inst, &mut out, slots)?;
        Ok(String::from_utf8(out).expect("utf8"))
    }

    #[test]
    fn s390x_inline_asm_loads_inputs_and_writes_the_result_back() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        slots.insert(VirtualReg::gpr(1), -16);
        let inst = MirInst::InlineAsm {
            template: "lgr $1, $0".to_owned(),
            operands: vec![input("r3", 0), out("r4")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(1)),
                MirType::Scalar(ScalarType::I64),
            )),
        };
        let text = emit_to_string(&inst, &slots).expect("emit");
        assert_eq!(
            text,
            "    lg %r3, -8(%r11)\n    lgr %r4, %r3\n    lgr %r2, %r4\n    stg %r2, -16(%r11)\n"
        );
    }

    #[test]
    fn s390x_inline_asm_sign_extends_a_narrow_result() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "lghi $0, -5".to_owned(),
            operands: vec![out("r2")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I32),
            )),
        };
        let text = emit_to_string(&inst, &slots).expect("emit");
        assert_eq!(
            text,
            "    lghi %r2, -5\n    lgfr %r2, %r2\n    stg %r2, -8(%r11)\n"
        );
    }

    #[test]
    fn s390x_inline_asm_zero_extends_a_boolean_result() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "lghi $0, 1".to_owned(),
            operands: vec![out("r2")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I1),
            )),
        };
        assert!(
            emit_to_string(&inst, &slots)
                .expect("emit")
                .contains("llgcr")
        );
    }

    #[test]
    fn s390x_inline_asm_without_a_result_emits_only_the_template() {
        let inst = MirInst::InlineAsm {
            template: "lghi %r2, 1".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["r2".to_owned(), "memory".to_owned()],
            clobber_memory: true,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &HashMap::new()).expect("emit"),
            "    lghi %r2, 1\n"
        );
    }

    #[test]
    fn s390x_inline_asm_refuses_a_result_read_from_a_float_register() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "ldgr $0, %r2".to_owned(),
            operands: vec![out("f0")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I64),
            )),
        };
        assert!(emit_to_string(&inst, &slots).is_err());
    }
}
