//! Inline assembly for i686 Linux.
//!
//! Operands are bound to physical registers around the template: inputs are loaded from
//! their frame slots into the bound register, the template is emitted with `$n` replaced
//! by the register of operand `n`, and the output register is read back into `%eax` and
//! written to the result slot. The emitter keeps every value in a frame slot and moves it
//! through `%eax`, `%ecx` and `%edx`, so no register holds a live value across an
//! instruction and a clobber of a caller-saved register costs nothing. The registers the
//! function body does depend on are handled by name: `%ebx`, `%esi` and `%edi` are
//! callee-saved and the prologue saves none of them, so the template's use of one is
//! bracketed with a push and a pop; `%esp` and `%ebp` are refused.

use super::operand::{EAX, Slots, load_operand, store_register, unsupported};
use crate::{
    error::LaminaError,
    mir::{AsmOperand, Instruction as MirInst, MirType, ScalarType},
};
use std::io::Write;

/// The register class accepted in place of a named register.
const GPR_CLASS: &str = "reg";

/// Every spelling an operand or clobber may use, with the 32-bit register it names and
/// the width in bytes the spelling accesses. `%esi` and `%edi` have no byte form in
/// 32-bit mode, so none is listed.
const NAMES: &[(&str, &str, u8)] = &[
    ("eax", "eax", 4),
    ("ax", "eax", 2),
    ("al", "eax", 1),
    ("ebx", "ebx", 4),
    ("bx", "ebx", 2),
    ("bl", "ebx", 1),
    ("ecx", "ecx", 4),
    ("cx", "ecx", 2),
    ("cl", "ecx", 1),
    ("edx", "edx", 4),
    ("dx", "edx", 2),
    ("dl", "edx", 1),
    ("esi", "esi", 4),
    ("si", "esi", 2),
    ("edi", "edi", 4),
    ("di", "edi", 2),
];

/// Registers the emitted body depends on between instructions.
const RESERVED: [&str; 4] = ["esp", "ebp", "sp", "bp"];

/// Registers the cdecl callee must preserve and this backend's prologue does not save.
const CALLEE_SAVED: [&str; 3] = ["ebx", "esi", "edi"];

/// Registers a `reg` operand may be given, in the order they are tried. The three the
/// emitter moves values through come first, since they need no save around the template.
const REG_CLASS_POOL: [&str; 6] = ["eax", "ecx", "edx", "ebx", "esi", "edi"];

/// The physical register bound to one operand, and how the template spells it.
pub(super) struct Binding {
    phys: &'static str,
    spelled: String,
}

fn unknown_register(name: &str) -> LaminaError {
    let names: Vec<&str> = NAMES.iter().map(|(spelling, _, _)| *spelling).collect();
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for i686; valid names are: {GPR_CLASS}, {}",
        names.join(", ")
    ))
}

/// Resolves a register written in an operand or a clobber to the 32-bit register it
/// names, refusing the registers the emitted code depends on.
fn named_register(name: &str) -> Result<&'static str, LaminaError> {
    if RESERVED.contains(&name) {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' is reserved by the i686 backend and cannot be named by inline \
             asm; %ebp is the frame pointer every value is addressed from and %esp the stack \
             pointer"
        )));
    }
    NAMES
        .iter()
        .find(|(spelling, _, _)| *spelling == name)
        .map(|&(_, canonical, _)| canonical)
        .ok_or_else(|| unknown_register(name))
}

fn location(operand: &AsmOperand) -> &str {
    match operand {
        AsmOperand::Input { location, .. } | AsmOperand::Output { location } => location,
    }
}

/// Binds every operand to a physical register. Named operands keep their spelling, so a
/// template asking for `al` writes one byte and a template asking for `eax` writes four;
/// `reg` operands take the first pool register that no named operand or clobber uses.
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
        let phys = named_register(name)?;
        if taken.contains(&phys) {
            return Err(LaminaError::ValidationError(format!(
                "register '{name}' is used by two inline asm operands"
            )));
        }
        taken.push(phys);
        bindings.push(Some(Binding {
            phys,
            spelled: format!("%{name}"),
        }));
    }
    for clobber in clobbers {
        if matches!(clobber.as_str(), "memory" | "flags") {
            continue;
        }
        let phys = named_register(clobber)?;
        if !taken.contains(&phys) {
            taken.push(phys);
        }
    }
    let mut resolved = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if let Some(binding) = binding {
            resolved.push(binding);
            continue;
        }
        let phys = REG_CLASS_POOL
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
            spelled: format!("%{phys}"),
        });
    }
    Ok(resolved)
}

/// The canonical names of the registers a clobber list names, in the order given.
fn clobbered_registers(clobbers: &[String]) -> Result<Vec<&'static str>, LaminaError> {
    let mut regs = Vec::new();
    for clobber in clobbers {
        if matches!(clobber.as_str(), "memory" | "flags") {
            continue;
        }
        let phys = named_register(clobber)?;
        if !regs.contains(&phys) {
            regs.push(phys);
        }
    }
    Ok(regs)
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

/// The callee-saved registers the template touches, in the order they are pushed.
fn saved_registers(touched: &[&'static str]) -> Vec<&'static str> {
    let mut saved = Vec::new();
    for &phys in touched {
        if CALLEE_SAVED.contains(&phys) && !saved.contains(&phys) {
            saved.push(phys);
        }
    }
    saved
}

/// How a result narrower than a word is widened into its slot, matching the canonical
/// form the rest of the backend keeps: `i1` zero-extended, the rest signed.
fn signed_result(ty: &MirType) -> bool {
    !matches!(ty, MirType::Scalar(ScalarType::I1))
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
            "i686 inline assembly emitter received another instruction",
        ));
    };
    if let Some((_, ty)) = result {
        super::reject_wide(ty, "inline assembly result")?;
    }
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;
    let mut touched: Vec<&'static str> = bindings.iter().map(|binding| binding.phys).collect();
    touched.extend(clobbered_registers(clobbers)?);
    let saved = saved_registers(&touched);

    for phys in &saved {
        writeln!(writer, "    pushl %{phys}")?;
    }
    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            load_operand(value, writer, slots, &format!("%{}", binding.phys))?;
        }
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(writer, "    {line}")?;
        }
    }

    let output = match result {
        Some(_) => Some(
            operands
                .iter()
                .zip(&bindings)
                .find(|(operand, _)| matches!(operand, AsmOperand::Output { .. }))
                .map(|(_, binding)| binding.phys)
                .ok_or_else(|| {
                    LaminaError::ValidationError(
                        "asm.T instruction has no out(...) operand".to_owned(),
                    )
                })?,
        ),
        None => None,
    };
    if let Some(output) = output
        && format!("%{output}") != EAX
    {
        writeln!(writer, "    movl %{output}, {EAX}")?;
    }
    for phys in saved.iter().rev() {
        writeln!(writer, "    popl %{phys}")?;
    }
    let Some((dst, ty)) = result else {
        return Ok(());
    };
    super::extend_in_place(writer, EAX, ty, signed_result(ty))?;
    store_register(EAX, dst, writer, slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Operand, Register, VirtualReg};
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

    fn emit_to_string(inst: &MirInst, slots: &Slots) -> Result<String, LaminaError> {
        let mut out = Vec::new();
        emit(inst, &mut out, slots)?;
        Ok(String::from_utf8(out).expect("utf8"))
    }

    #[test]
    fn i686_named_operands_keep_their_spelling() {
        assert_eq!(
            spellings(&[input("ecx", 0), out("al")], &[]),
            vec!["%ecx".to_owned(), "%al".to_owned()]
        );
    }

    #[test]
    fn i686_a_narrow_spelling_binds_the_whole_register() {
        assert!(bind(&[input("al", 0), out("eax")], &[]).is_err());
        assert!(bind(&[input("cx", 0)], &["ecx".to_owned()]).is_ok());
    }

    #[test]
    fn i686_reg_class_operands_avoid_named_operands_and_clobbers() {
        let clobbers = vec!["edx".to_owned(), "memory".to_owned()];
        assert_eq!(
            spellings(&[input("eax", 0), out(GPR_CLASS)], &clobbers),
            vec!["%eax".to_owned(), "%ecx".to_owned()]
        );
    }

    #[test]
    fn i686_inline_asm_refuses_the_frame_and_stack_pointers() {
        for name in ["esp", "ebp", "sp", "bp"] {
            assert!(bind(&[out(name)], &[]).is_err(), "{name} as operand");
            assert!(bind(&[], &[name.to_owned()]).is_err(), "{name} as clobber");
        }
    }

    #[test]
    fn i686_inline_asm_refuses_sixty_four_bit_and_aarch64_names() {
        assert!(bind(&[out("rax")], &[]).is_err());
        assert!(bind(&[], &["r11".to_owned()]).is_err());
        assert!(bind(&[], &["x0".to_owned()]).is_err());
        assert!(bind(&[], &["sil".to_owned()]).is_err());
    }

    #[test]
    fn i686_inline_asm_refuses_one_register_for_two_operands() {
        assert!(bind(&[input("ecx", 0), out("cl")], &[]).is_err());
    }

    #[test]
    fn i686_inline_asm_template_substitutes_operands_and_escapes() {
        let bindings = bind(&[input("ecx", 0), out("eax")], &[]).expect("bind");
        assert_eq!(
            expand_template("movl $0, $1\\naddl $$8, $1", &bindings).expect("expand"),
            "movl %ecx, %eax\naddl $8, %eax"
        );
    }

    #[test]
    fn i686_inline_asm_template_rejects_an_operand_past_the_end() {
        let bindings = bind(&[out("eax")], &[]).expect("bind");
        assert!(expand_template("movl $$0, $1", &bindings).is_err());
    }

    #[test]
    fn i686_inline_asm_loads_inputs_and_writes_the_result_back() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        slots.insert(VirtualReg::gpr(1), -16);
        let inst = MirInst::InlineAsm {
            template: "movl $0, $1".to_owned(),
            operands: vec![input("ecx", 0), out("edx")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(1)),
                MirType::Scalar(ScalarType::I32),
            )),
        };
        assert_eq!(
            emit_to_string(&inst, &slots).expect("emit"),
            "    movl -8(%ebp), %ecx\n    movl %ecx, %edx\n    movl %edx, %eax\n    \
             movl %eax, -16(%ebp)\n"
        );
    }

    #[test]
    fn i686_inline_asm_sign_extends_a_narrow_result() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "movw $$-5, $0".to_owned(),
            operands: vec![out("ax")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I16),
            )),
        };
        assert_eq!(
            emit_to_string(&inst, &slots).expect("emit"),
            "    movw $-5, %ax\n    movswl %ax, %eax\n    movl %eax, -8(%ebp)\n"
        );
    }

    #[test]
    fn i686_inline_asm_zero_extends_a_boolean_result() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "movb $$1, $0".to_owned(),
            operands: vec![out("al")],
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
                .contains("movzbl %al, %eax")
        );
    }

    #[test]
    fn i686_inline_asm_brackets_a_callee_saved_clobber_with_a_push_and_a_pop() {
        let inst = MirInst::InlineAsm {
            template: "movl $$99, %ebx".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["ebx".to_owned(), "memory".to_owned()],
            clobber_memory: true,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &HashMap::new()).expect("emit"),
            "    pushl %ebx\n    movl $99, %ebx\n    popl %ebx\n"
        );
    }

    #[test]
    fn i686_inline_asm_brackets_a_callee_saved_operand_and_reads_its_result_first() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "movl $$7, $0".to_owned(),
            operands: vec![out("esi")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I32),
            )),
        };
        assert_eq!(
            emit_to_string(&inst, &slots).expect("emit"),
            "    pushl %esi\n    movl $7, %esi\n    movl %esi, %eax\n    popl %esi\n    \
             movl %eax, -8(%ebp)\n"
        );
    }

    #[test]
    fn i686_inline_asm_leaves_a_caller_saved_clobber_alone() {
        let inst = MirInst::InlineAsm {
            template: "movl $$7, %ecx".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["ecx".to_owned(), "flags".to_owned()],
            clobber_memory: false,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &HashMap::new()).expect("emit"),
            "    movl $7, %ecx\n"
        );
    }

    #[test]
    fn i686_inline_asm_refuses_a_sixty_four_bit_result() {
        let mut slots = HashMap::new();
        slots.insert(VirtualReg::gpr(0), -8);
        let inst = MirInst::InlineAsm {
            template: "movl $$1, $0".to_owned(),
            operands: vec![out("eax")],
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
