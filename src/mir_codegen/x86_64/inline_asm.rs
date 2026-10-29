//! Inline assembly for x86_64 Linux.
//!
//! Operands are bound to physical registers before the template runs: inputs are copied
//! into their registers, the template is emitted with `$n` replaced by the register of
//! operand `n`, and the output register is copied to the result. Every register the
//! instruction touches is reported by `touched_registers`, so the allocators never keep a
//! live value in one across the instruction and the prologue saves the callee-saved ones.

use super::*;
use crate::mir::asm::{GPR_CLASS, x86_64_register, x86_64_register_names};
use crate::mir::{AsmOperand, MirType, Register};
use lamina_codegen::x86_64::extend_integer;

/// Registers a `reg` operand may be given, in the order they are tried.
const REG_CLASS_POOL: &[&str] = &["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"];

/// The physical register bound to one operand, and how the template spells it.
pub(super) struct Binding {
    phys: &'static str,
    spelled: String,
}

fn unknown_register(name: &str) -> LaminaError {
    LaminaError::ValidationError(format!(
        "unknown register '{name}' in inline asm for x86_64; valid names are: reg, {}",
        x86_64_register_names().join(", ")
    ))
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
        let (phys, _) = x86_64_register(name).ok_or_else(|| unknown_register(name))?;
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
        let (phys, _) = x86_64_register(clobber).ok_or_else(|| unknown_register(clobber))?;
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
            .find(|reg| !taken.contains(reg))
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
    for clobber in clobbers {
        if let Some((phys, _)) = x86_64_register(clobber) {
            regs.push(phys);
        }
    }
    regs.sort_unstable();
    regs.dedup();
    regs
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
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    target_os: TargetOperatingSystem,
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
    if target_os != TargetOperatingSystem::Linux {
        return Err(LaminaError::ValidationError(
            "inline assembly is supported on x86_64 Linux only".to_owned(),
        ));
    }
    let bindings = bind(operands, clobbers)?;
    let text = expand_template(template, &bindings)?;

    for (operand, binding) in operands.iter().zip(&bindings) {
        if let AsmOperand::Input { value, .. } = operand {
            load_copy_operand_into(
                &MirType::i64(),
                value,
                binding.phys,
                writer,
                reg_alloc,
                stack_slots,
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
        let output = operands
            .iter()
            .zip(&bindings)
            .find(|(operand, _)| matches!(operand, AsmOperand::Output { .. }))
            .map(|(_, binding)| binding.phys)
            .ok_or_else(|| {
                LaminaError::ValidationError("asm.T instruction has no out(...) operand".to_owned())
            })?;
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
