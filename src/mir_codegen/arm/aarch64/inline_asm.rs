//! Inline assembly for AArch64.
//!
//! Operands are bound to physical registers around the template: inputs are staged on the
//! stack and then loaded into their bound registers, the template is emitted with `$n`
//! replaced by the register of operand `n`, and the output register is copied to the
//! result. Every register the template touches that carries a value the emitter still
//! needs, or that the AAPCS64 callee must preserve, is pushed before the template and
//! popped after it, so the instruction changes no register except through its result.

use super::*;
use crate::mir::AsmOperand;
use crate::mir_codegen::inline_asm::{
    Binding, GPR_CLASS, RegisterFacts, expand_template, output_register, result_is_bool, spell_bare,
};
use lamina_platform::PointerWidth;

const GPRS: [&str; 31] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30",
];

const FPRS: [&str; 32] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    "d28", "d29", "d30", "d31",
];

/// Registers a `reg` operand may be given, in the order they are tried. The emitter's own
/// scratch registers, the platform register and the frame and link registers are absent.
const REG_CLASS_POOL: [&str; 26] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
];

const FACTS: RegisterFacts = RegisterFacts {
    operand: operand_register,
    clobber: clobber_register,
    spell: spell_bare,
    pool: &REG_CLASS_POOL,
};

/// Index of a general register (`x` or `w` spelling) or a vector register (`d` or `v`
/// spelling), and whether it is a vector register.
fn register_index(name: &str) -> Option<(bool, usize)> {
    let mut chars = name.chars();
    let fp = match chars.next()? {
        'x' | 'w' => false,
        'd' | 'v' => true,
        _ => return None,
    };
    let digits = chars.as_str();
    let index: usize = digits.parse().ok()?;
    if index.to_string() != digits {
        return None;
    }
    let limit = if fp { FPRS.len() } else { GPRS.len() };
    (index < limit).then_some((fp, index))
}

fn canonical(fp: bool, index: usize) -> &'static str {
    if fp { FPRS[index] } else { GPRS[index] }
}

fn is_reserved(fp: bool, index: usize) -> bool {
    !fp && matches!(index, 16 | 17 | 18 | 29 | 30)
}

fn is_callee_saved(phys: &str) -> bool {
    match register_index(phys) {
        Some((false, index)) => (19..=28).contains(&index),
        Some((true, index)) => (8..=15).contains(&index),
        None => false,
    }
}

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for AArch64; valid names are: {GPR_CLASS}, \
         x0 through x15 and x19 through x28 (or their w spellings), and d8 through d15 or \
         v8 through v15 as clobbers"
    ))
}

/// Resolves a register written in an operand or a clobber to its canonical 64-bit or
/// 128-bit name, refusing the registers the emitted code depends on.
fn named_register(name: &str) -> Result<(bool, &'static str), LaminaError> {
    let (fp, index) = register_index(name).ok_or_else(|| unknown_register(name))?;
    if is_reserved(fp, index) {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' is reserved by the AArch64 backend and cannot be named by inline \
             asm; x16 and x17 are its scratch registers, x18 the platform register, x29 the \
             frame pointer and x30 the link register"
        )));
    }
    Ok((fp, canonical(fp, index)))
}

/// Resolves a register an operand names. A vector register may be named only as a
/// clobber: the emitter stages an operand through a general register.
fn operand_register(name: &str) -> Result<&'static str, LaminaError> {
    let (fp, phys) = named_register(name)?;
    if fp {
        return Err(LaminaError::ValidationError(format!(
            "register '{name}' is a vector register; AArch64 inline asm operands must be \
             general registers"
        )));
    }
    Ok(phys)
}

fn clobber_register(name: &str) -> Result<Option<&'static str>, LaminaError> {
    named_register(name).map(|(_, phys)| Some(phys))
}

pub(super) fn bind(
    operands: &[AsmOperand],
    clobbers: &[String],
) -> Result<Vec<Binding>, LaminaError> {
    FACTS.bind(operands, clobbers)
}

/// The registers pushed around the template: every callee-saved register it touches, and
/// every caller-saved register the allocator holds a value in.
fn saved_registers(touched: &[&'static str], ra: &A64RegAlloc) -> Vec<&'static str> {
    let mut saved: Vec<&'static str> = Vec::new();
    for &phys in touched {
        if saved.contains(&phys) {
            continue;
        }
        if is_callee_saved(phys) || ra.is_occupied(phys) {
            saved.push(phys);
        }
    }
    saved
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let MirInst::InlineAsm {
        template,
        operands,
        clobbers,
        result,
        ..
    } = inst
    else {
        return Err(LaminaError::ValidationError(
            "inline assembly emitter received another instruction".to_owned(),
        ));
    };
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;
    let clobbered = FACTS.clobbered_registers(clobbers)?;

    let output = match result {
        Some(_) => Some(output_register(operands, &bindings)?),
        None => None,
    };

    let mut touched: Vec<&'static str> = bindings.iter().map(|b| b.phys).collect();
    touched.extend(clobbered);
    let saved = saved_registers(&touched, ra);

    for phys in &saved {
        writeln!(w, "    str {phys}, [sp, #-16]!")?;
    }
    let mut staged: Vec<&'static str> = Vec::new();
    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            emit_materialize_operand(w, value, "x16", frame, ra)?;
            writeln!(w, "    str x16, [sp, #-16]!")?;
            staged.push(binding.phys);
        }
    }
    for phys in staged.iter().rev() {
        writeln!(w, "    ldr {phys}, [sp], #16")?;
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(w, "    {line}")?;
        }
    }

    if let (Some(output), Some((_, ty))) = (output, result) {
        writeln!(w, "    mov x16, {output}")?;
        if result_is_bool(ty) {
            writeln!(w, "    and x16, x16, #1")?;
        } else {
            extend_integer(w, "x16", ty, true, pointer_width)?;
        }
    }
    for phys in saved.iter().rev() {
        writeln!(w, "    ldr {phys}, [sp], #16")?;
    }
    if let Some((dst, _)) = result {
        store_result(w, dst, "x16", frame, ra)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Operand, VirtualReg};

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

    fn emit_to_string(
        inst: &MirInst,
        frame: &FrameMap,
        ra: &mut A64RegAlloc,
    ) -> Result<String, LaminaError> {
        let mut out = Vec::new();
        emit(inst, &mut out, frame, ra, PointerWidth::Bits64)?;
        Ok(String::from_utf8(out).expect("utf8"))
    }

    #[test]
    fn aarch64_named_operands_keep_their_spelling() {
        assert_eq!(
            spellings(&[input("x3", 0), out("w4")], &[]),
            vec!["x3".to_owned(), "w4".to_owned()]
        );
    }

    #[test]
    fn aarch64_reg_class_operands_avoid_named_operands_and_clobbers() {
        let clobbers = vec!["x0".to_owned(), "memory".to_owned()];
        assert_eq!(
            spellings(&[input("x1", 0), out(GPR_CLASS)], &clobbers),
            vec!["x1".to_owned(), "x2".to_owned()]
        );
    }

    #[test]
    fn aarch64_inline_asm_refuses_the_reserved_registers() {
        for name in ["x16", "x17", "x18", "x29", "x30", "w16", "w30"] {
            assert!(bind(&[out(name)], &[]).is_err(), "{name} as operand");
            assert!(bind(&[], &[name.to_owned()]).is_err(), "{name} as clobber");
        }
    }

    #[test]
    fn aarch64_inline_asm_refuses_x86_and_stack_names() {
        assert!(bind(&[out("rax")], &[]).is_err());
        assert!(bind(&[], &["rdx".to_owned()]).is_err());
        assert!(bind(&[], &["sp".to_owned()]).is_err());
        assert!(bind(&[], &["x31".to_owned()]).is_err());
        assert!(bind(&[], &["x01".to_owned()]).is_err());
    }

    #[test]
    fn aarch64_inline_asm_refuses_one_register_for_two_operands() {
        assert!(bind(&[input("x2", 0), out("w2")], &[]).is_err());
    }

    #[test]
    fn aarch64_inline_asm_refuses_a_vector_register_operand() {
        assert!(bind(&[out("d8")], &[]).is_err());
    }

    #[test]
    fn aarch64_inline_asm_accepts_vector_register_clobbers() {
        assert!(
            FACTS
                .clobbered_registers(&["d8".to_owned(), "v15".to_owned()])
                .is_ok()
        );
    }

    #[test]
    fn aarch64_inline_asm_template_substitutes_operands_and_escapes() {
        let bindings = bind(&[input("x3", 0), out("w4")], &[]).expect("bind");
        assert_eq!(
            expand_template("add $1, $0, $$8\\nmov x9, #1", &bindings).expect("expand"),
            "add w4, x3, $8\nmov x9, #1"
        );
    }

    #[test]
    fn aarch64_inline_asm_template_rejects_an_operand_past_the_end() {
        let bindings = bind(&[out("x2")], &[]).expect("bind");
        assert!(expand_template("mov $1, #0", &bindings).is_err());
    }

    #[test]
    fn aarch64_inline_asm_loads_inputs_and_writes_the_result_back() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(1)), -16);
        let inst = MirInst::InlineAsm {
            template: "mov $1, $0".to_owned(),
            operands: vec![input("x3", 0), out("x4")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(1)),
                MirType::Scalar(ScalarType::I64),
            )),
        };
        let text = emit_to_string(&inst, &frame, &mut A64RegAlloc::new()).expect("emit");
        assert_eq!(
            text,
            "    ldur x16, [x29, #-8]\n    str x16, [sp, #-16]!\n    ldr x3, [sp], #16\n    \
             mov x4, x3\n    mov x16, x4\n    stur x16, [x29, #-16]\n"
        );
    }

    #[test]
    fn aarch64_inline_asm_sign_extends_a_narrow_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mov w0, #-5".to_owned(),
            operands: vec![out("w0")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I32),
            )),
        };
        let text = emit_to_string(&inst, &frame, &mut A64RegAlloc::new()).expect("emit");
        assert!(text.contains("    mov w0, #-5\n    mov x16, x0\n    sbfx x16, x16, #0, #32\n"));
    }

    #[test]
    fn aarch64_inline_asm_masks_a_boolean_result() {
        let mut frame = empty_frame();
        frame
            .slots
            .insert(Register::Virtual(VirtualReg::gpr(0)), -8);
        let inst = MirInst::InlineAsm {
            template: "mov x0, #1".to_owned(),
            operands: vec![out("x0")],
            clobbers: Vec::new(),
            clobber_memory: false,
            result: Some((
                Register::Virtual(VirtualReg::gpr(0)),
                MirType::Scalar(ScalarType::I1),
            )),
        };
        let text = emit_to_string(&inst, &frame, &mut A64RegAlloc::new()).expect("emit");
        assert!(text.contains("    and x16, x16, #1\n"));
    }

    #[test]
    fn aarch64_inline_asm_saves_a_live_caller_saved_clobber() {
        let mut ra = A64RegAlloc::new();
        ra.occupy("x13");
        let inst = MirInst::InlineAsm {
            template: "mov x13, #7".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["x13".to_owned(), "memory".to_owned()],
            clobber_memory: true,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &empty_frame(), &mut ra).expect("emit"),
            "    str x13, [sp, #-16]!\n    mov x13, #7\n    ldr x13, [sp], #16\n"
        );
    }

    #[test]
    fn aarch64_inline_asm_leaves_an_idle_caller_saved_clobber_alone() {
        let inst = MirInst::InlineAsm {
            template: "mov x5, #7".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["x5".to_owned()],
            clobber_memory: false,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &empty_frame(), &mut A64RegAlloc::new()).expect("emit"),
            "    mov x5, #7\n"
        );
    }

    #[test]
    fn aarch64_inline_asm_saves_callee_saved_registers_it_touches() {
        let inst = MirInst::InlineAsm {
            template: "mov x19, #99\nfmov d8, xzr".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["x19".to_owned(), "d8".to_owned()],
            clobber_memory: false,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &empty_frame(), &mut A64RegAlloc::new()).expect("emit"),
            "    str x19, [sp, #-16]!\n    str d8, [sp, #-16]!\n    mov x19, #99\n    \
             fmov d8, xzr\n    ldr d8, [sp], #16\n    ldr x19, [sp], #16\n"
        );
    }

    #[test]
    fn aarch64_inline_asm_without_a_result_emits_only_the_template() {
        let inst = MirInst::InlineAsm {
            template: "nop".to_owned(),
            operands: Vec::new(),
            clobbers: vec!["memory".to_owned(), "flags".to_owned()],
            clobber_memory: true,
            result: None,
        };
        assert_eq!(
            emit_to_string(&inst, &empty_frame(), &mut A64RegAlloc::new()).expect("emit"),
            "    nop\n"
        );
    }
}
