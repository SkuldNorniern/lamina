//! Inline assembly for armv7 Linux.
//!
//! Operands are bound to core registers around the template: inputs are loaded from
//! their frame slots into the bound register, the template is emitted with `$n` replaced
//! by the register of operand `n`, and the output register is widened and written to the
//! result slot. The backend keeps every value in a frame slot and evaluates one
//! instruction at a time through `r4` to `r9`, so no register holds a live value across
//! an instruction and a clobber needs no save here: the prologue already pushes `r4`
//! through `r12`, `lr` and `d8` through `d15`, and the epilogue restores them. The registers
//! the body does depend on between instructions are refused by name.
//!
//! A core register pair is not offered. An `AsmOperand` names one location and carries
//! no tie between two, so a 64-bit value would need an even-aligned pair the template has
//! no way to spell; an `i64` result is refused rather than truncated into one register.

use super::operand::{ADDR, ENTRY_SP, FP, materialize, store_result, unsupported};
use crate::{
    error::LaminaError,
    mir::{AsmOperand, Instruction as MirInst, MirType, ScalarType},
};
use lamina_codegen::aarch64::FrameMap;
use std::io::Write;

/// The register class accepted in place of a named register.
const GPR_CLASS: &str = "reg";

/// Core registers an operand or a clobber may name.
const CORE: [&str; 10] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"];

/// The link register, which the prologue saves and the epilogue restores, so a template
/// that calls may name it. It holds no value between instructions and is never bound to
/// an operand.
const LINK: &str = "lr";

/// Registers the emitted body depends on between instructions, with why each is refused.
const RESERVED: [(&str, &str); 6] = [
    (
        ENTRY_SP,
        "the entry stack pointer the epilogue restores sp from",
    ),
    (FP, "the frame pointer every value is addressed from"),
    (
        ADDR,
        "the address scratch this backend loads and stores through",
    ),
    ("r13", "the stack pointer"),
    ("sp", "the stack pointer"),
    ("pc", "the program counter"),
];

/// Registers a `reg` operand may be given, in the order they are tried.
const REG_CLASS_POOL: [&str; 10] = CORE;

/// The physical register bound to one operand, and how the template spells it.
pub(super) struct Binding {
    phys: &'static str,
    spelled: String,
}

/// A VFP register name, accepted as a clobber. `d8` through `d15` are callee-saved and
/// the prologue pushes them; the rest hold no value between instructions.
fn vfp_register(name: &str) -> bool {
    let limit = match name.as_bytes().first() {
        Some(b's') => 32,
        Some(b'd') => 16,
        _ => return false,
    };
    let digits = &name[1..];
    matches!(digits.parse::<usize>(), Ok(index) if index < limit && index.to_string() == digits)
}

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for armv7; valid names are: {GPR_CLASS}, {}, \
         and, as clobbers only, {LINK}, s0 through s31 and d0 through d15",
        CORE.join(", ")
    ))
}

fn reserved(name: &str) -> Option<LaminaError> {
    RESERVED
        .iter()
        .find(|(reserved, _)| *reserved == name)
        .map(|(_, why)| {
            LaminaError::ValidationError(format!(
                "register '{name}' is reserved by the armv7 backend and cannot be named by \
                 inline asm; it is {why}"
            ))
        })
}

/// Resolves a register an operand names. Only a core register can hold an operand: a VFP
/// register would need a load the backend writes with a different instruction, and the
/// link register is restored from the stack rather than kept.
fn operand_register(name: &str) -> Result<&'static str, LaminaError> {
    if let Some(error) = reserved(name) {
        return Err(error);
    }
    if name == LINK || vfp_register(name) {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' may be named as an armv7 inline asm clobber but not as an \
             operand; operands are held in the core registers {}",
            CORE.join(", ")
        )));
    }
    CORE.iter()
        .copied()
        .find(|candidate| *candidate == name)
        .ok_or_else(|| unknown_register(name))
}

/// Resolves a register a clobber names, returning the core register it occupies or
/// `None` for a VFP register or the link register, neither of which can take an operand.
fn clobber_register(name: &str) -> Result<Option<&'static str>, LaminaError> {
    if let Some(error) = reserved(name) {
        return Err(error);
    }
    if name == LINK || vfp_register(name) {
        return Ok(None);
    }
    CORE.iter()
        .copied()
        .find(|candidate| *candidate == name)
        .map(Some)
        .ok_or_else(|| unknown_register(name))
}

fn location(operand: &AsmOperand) -> &str {
    match operand {
        AsmOperand::Input { location, .. } | AsmOperand::Output { location } => location,
    }
}

/// Binds every operand to a core register. Named operands keep their spelling; `reg`
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
        let phys = operand_register(name)?;
        if taken.contains(&phys) {
            return Err(LaminaError::ValidationError(format!(
                "register '{name}' is used by two inline asm operands"
            )));
        }
        taken.push(phys);
        bindings.push(Some(Binding {
            phys,
            spelled: phys.to_owned(),
        }));
    }
    for clobber in clobbers {
        if matches!(clobber.as_str(), "memory" | "flags") {
            continue;
        }
        if let Some(phys) = clobber_register(clobber)?
            && !taken.contains(&phys)
        {
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
            spelled: phys.to_owned(),
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

/// Refuses a result the backend cannot read out of one core register, naming the pair
/// rule rather than writing half an answer.
fn check_result_type(ty: &MirType) -> Result<(), LaminaError> {
    if matches!(ty, MirType::Scalar(ScalarType::I64)) {
        return Err(unsupported(
            "armv7 inline assembly cannot return an i64: the value needs an even-aligned core \
             register pair, and an asm operand names one register with no way to spell its \
             other half; narrow the result to i32"
                .to_string(),
        ));
    }
    if ty.is_vector() {
        return Err(unsupported(format!(
            "armv7 inline assembly cannot return the vector type {ty}"
        )));
    }
    if matches!(ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64)) {
        return Err(unsupported(format!(
            "armv7 inline assembly cannot return {ty}: an asm operand is held in a core \
             register, and a VFP register may be named only as a clobber"
        )));
    }
    Ok(())
}

pub(super) fn emit<W: Write>(
    w: &mut W,
    inst: &MirInst,
    frame: &FrameMap,
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
            "armv7 inline assembly emitter received another instruction".to_string(),
        ));
    };
    if let Some((_, ty)) = result {
        check_result_type(ty)?;
    }
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;

    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            materialize(w, value, binding.phys, frame)?;
        }
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(w, "    {line}")?;
        }
    }

    let Some((dst, ty)) = result else {
        return Ok(());
    };
    let output = operands
        .iter()
        .zip(&bindings)
        .find(|(operand, _)| matches!(operand, AsmOperand::Output { .. }))
        .map(|(_, binding)| binding.phys)
        .ok_or_else(|| {
            LaminaError::ValidationError("asm.T instruction has no out(...) operand".to_owned())
        })?;
    if matches!(ty, MirType::Scalar(ScalarType::I1)) {
        writeln!(w, "    and {output}, {output}, #1")?;
    } else {
        super::operand::extend_integer(w, output, ty, true)?;
    }
    store_result(w, dst, output, frame)
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

    fn empty_frame() -> FrameMap {
        FrameMap {
            slots: HashMap::new(),
            frame_size: 0,
        }
    }

    fn emit_to_string(inst: &MirInst, frame: &FrameMap) -> Result<String, LaminaError> {
        let mut out = Vec::new();
        emit(&mut out, inst, frame)?;
        Ok(String::from_utf8(out).expect("utf8"))
    }

    #[test]
    fn armv7_named_operands_keep_their_spelling() {
        assert_eq!(
            spellings(&[input("r3", 0), out("r4")], &[]),
            vec!["r3".to_owned(), "r4".to_owned()]
        );
    }

    #[test]
    fn armv7_reg_class_operands_avoid_named_operands_and_clobbers() {
        let clobbers = vec!["r0".to_owned(), "memory".to_owned()];
        assert_eq!(
            spellings(&[input("r1", 0), out(GPR_CLASS)], &clobbers),
            vec!["r1".to_owned(), "r2".to_owned()]
        );
    }

    #[test]
    fn armv7_inline_asm_refuses_the_registers_the_body_depends_on() {
        for name in ["r10", "r11", "r12", "r13", "sp", "pc"] {
            assert!(bind(&[out(name)], &[]).is_err(), "{name} as operand");
            assert!(bind(&[], &[name.to_owned()]).is_err(), "{name} as clobber");
        }
    }

    #[test]
    fn armv7_inline_asm_refuses_x86_and_aarch64_names() {
        assert!(bind(&[out("rax")], &[]).is_err());
        assert!(bind(&[], &["eax".to_owned()]).is_err());
        assert!(bind(&[], &["x0".to_owned()]).is_err());
        assert!(bind(&[], &["r16".to_owned()]).is_err());
        assert!(bind(&[], &["r01".to_owned()]).is_err());
        assert!(bind(&[], &["d16".to_owned()]).is_err());
    }

    #[test]
    fn armv7_inline_asm_takes_the_link_and_vfp_registers_as_clobbers_only() {
        assert!(bind(&[], &["lr".to_owned()]).is_ok());
        assert!(bind(&[], &["d8".to_owned(), "s31".to_owned()]).is_ok());
        assert!(bind(&[out("lr")], &[]).is_err());
        assert!(bind(&[out("d8")], &[]).is_err());
    }

    #[test]
    fn armv7_a_vfp_clobber_leaves_the_core_pool_alone() {
        assert_eq!(
            spellings(&[out(GPR_CLASS)], &["d0".to_owned(), "lr".to_owned()]),
            vec!["r0".to_owned()]
        );
    }

    #[test]
    fn armv7_inline_asm_refuses_one_register_for_two_operands() {
        assert!(bind(&[input("r2", 0), out("r2")], &[]).is_err());
    }

    #[test]
    fn armv7_inline_asm_template_substitutes_operands_and_escapes() {
        let bindings = bind(&[input("r3", 0), out("r4")], &[]).expect("bind");
        assert_eq!(
            expand_template("add $1, $0, $$8\\nmov r9, #1", &bindings).expect("expand"),
            "add r4, r3, $8\nmov r9, #1"
        );
    }

    #[test]
    fn armv7_inline_asm_template_rejects_an_operand_past_the_end() {
        let bindings = bind(&[out("r2")], &[]).expect("bind");
        assert!(expand_template("mov $1, #0", &bindings).is_err());
    }

    #[test]
    fn armv7_inline_asm_loads_inputs_and_writes_the_result_back() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(1)), -16);
        let inst = MirInst::InlineAsm {
            template: "mov $1, $0".to_owned(),
            operands: vec![input("r3", 0), out("r4")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(1)),
                MirType::Scalar(ScalarType::I32),
            )),
        };
        assert_eq!(
            emit_to_string(&inst, &frame).expect("emit"),
            "    ldr r3, [r11, #-8]\n    mov r4, r3\n    str r4, [r11, #-16]\n"
        );
    }

    #[test]
    fn armv7_inline_asm_sign_extends_a_narrow_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mvn $0, #4".to_owned(),
            operands: vec![out("r5")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I8),
            )),
        };
        assert_eq!(
            emit_to_string(&inst, &frame).expect("emit"),
            "    mvn r5, #4\n    sxtb r5, r5\n    str r5, [r11, #-8]\n"
        );
    }

    #[test]
    fn armv7_inline_asm_masks_a_boolean_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mov $0, #1".to_owned(),
            operands: vec![out("r5")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I1),
            )),
        };
        assert!(
            emit_to_string(&inst, &frame)
                .expect("emit")
                .contains("and r5, r5, #1")
        );
    }

    #[test]
    fn armv7_inline_asm_without_a_result_emits_only_the_template() {
        let inst = MirInst::InlineAsm {
            template: "mov r4, #99".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["r4".to_owned(), "memory".to_owned(), "flags".to_owned()],
            clobber_memory: true,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &empty_frame()).expect("emit"),
            "    mov r4, #99\n"
        );
    }

    #[test]
    fn armv7_inline_asm_refuses_a_sixty_four_bit_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mov $0, #1".to_owned(),
            operands: vec![out("r0")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I64),
            )),
        };
        assert!(emit_to_string(&inst, &frame).is_err());
    }

    #[test]
    fn armv7_inline_asm_refuses_a_floating_point_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mov $0, #0".to_owned(),
            operands: vec![out("r0")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::F32),
            )),
        };
        assert!(emit_to_string(&inst, &frame).is_err());
    }
}
