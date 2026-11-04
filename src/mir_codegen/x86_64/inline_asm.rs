//! Inline assembly for x86_64 Linux and Windows.
//!
//! Operands are bound to physical registers before the template runs: inputs are copied
//! into their registers, the template is emitted with `$n` replaced by the register of
//! operand `n`, and the output register is copied to the result. Every register the
//! instruction touches is reported by `touched_registers`, so the allocators never keep a
//! live value in one across the instruction and the prologue saves the callee-saved ones.

use super::*;
use crate::mir::asm::{x86_64_register, x86_64_register_names};
use crate::mir::{AsmOperand, MirType, Register};
use crate::mir_codegen::inline_asm::{
    Binding, RegisterFacts, expand_template, output_register, spell_percent,
};
use lamina_codegen::x86_64::extend_integer;
use lamina_platform::PointerWidth;

/// Registers a `reg` operand may be given, in the order they are tried.
const REG_CLASS_POOL: &[&str] = &["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"];

const FACTS: RegisterFacts = RegisterFacts {
    operand: named_register,
    clobber: clobber_register,
    spell: spell_percent,
    pool: REG_CLASS_POOL,
};

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for x86_64; valid names are: reg, {}",
        x86_64_register_names().join(", ")
    ))
}

/// Resolves a register written in an operand or a clobber to its canonical 64-bit name.
fn named_register(name: &str) -> Result<&'static str, LaminaError> {
    x86_64_register(name)
        .map(|(phys, _)| phys)
        .ok_or_else(|| unknown_register(name))
}

fn clobber_register(name: &str) -> Result<Option<&'static str>, LaminaError> {
    named_register(name).map(Some)
}

pub(super) fn bind(
    operands: &[AsmOperand],
    clobbers: &[String],
) -> Result<Vec<Binding>, LaminaError> {
    FACTS.bind(operands, clobbers)
}

/// Every physical register an inline assembly instruction destroys or writes, by canonical
/// 64-bit name. Empty for other instructions and for instructions that fail to bind; the
/// emitter reports those errors.
pub(super) fn touched_registers(inst: &MirInst) -> Vec<&'static str> {
    let MirInst::InlineAsm {
        operands, clobbers, ..
    } = inst
    else {
        return Vec::new();
    };
    let mut regs: Vec<&'static str> = match bind(operands, clobbers) {
        Ok(bindings) => bindings.iter().map(|binding| binding.phys).collect(),
        Err(_) => return Vec::new(),
    };
    let Ok(clobbered) = FACTS.clobbered_registers(clobbers) else {
        return Vec::new();
    };
    regs.extend(clobbered);
    regs.sort_unstable();
    regs.dedup();
    regs
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target_os: TargetOperatingSystem,
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
    if !matches!(
        target_os,
        TargetOperatingSystem::Linux | TargetOperatingSystem::Windows
    ) {
        return Err(LaminaError::ValidationError(
            "the x86_64 backend emits inline assembly for Linux and Windows only".to_owned(),
        ));
    }
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;

    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            load_copy_operand_into(
                &MirType::Scalar(ScalarType::I64),
                value,
                binding.phys,
                writer,
                reg_alloc,
                stack_slots,
                pointer_width,
            )?;
        }
    }

    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            writeln!(writer, "    {line}")?;
        }
    }

    if let Some((dst, ty)) = result {
        let output = output_register(operands, &bindings)?;
        if let Some((size, signed)) = narrow_canonical_extension(ty) {
            extend_integer(writer, output, size, signed)?;
        }
        if let Register::Virtual(vreg) = dst {
            if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                if phys != output {
                    writeln!(writer, "    movq %{output}, %{phys}")?;
                }
            } else if let Some(offset) = stack_slots.get(vreg) {
                writeln!(writer, "    movq %{output}, {offset}(%rbp)")?;
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "inline asm result {dst} has no register or stack slot"
                )));
            }
        }
    }
    Ok(())
}
