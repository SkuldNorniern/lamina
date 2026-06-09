//! ARX64 MIR codegen.
//!
//! This is the first real ARX64 assembly backend. It intentionally supports a
//! small integer/bare-metal subset matching the current reference emulator:
//! scalar GPR operations, base+offset loads/stores, calls, branches, and returns.
//! Unsupported MIR ops fail loudly instead of being silently lowered wrong.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::LaminaError;
use crate::mir::{
    AddressMode, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType,
    Module as MirModule, Operand, Register, ScalarType,
};
use crate::mir_codegen::{CodegenError, MirCodegenSettings};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

const RA: &str = "r1";
const SP: &str = "r2";
const RET: &str = "r5";
const FP: &str = "r30";
const SCR0: &str = "r11";
const SCR1: &str = "r12";
const SCR2: &str = "r13";
const ARG_REGS: [&str; 6] = ["r5", "r6", "r7", "r8", "r9", "r10"];
const UART_BASE: i64 = 0x0010_0000;

pub fn generate_mir_arx64<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
) -> Result<(), LaminaError> {
    generate_mir_arx64_with_units_and_settings(
        module,
        writer,
        target_os,
        1,
        &MirCodegenSettings::default(),
    )
}

pub fn generate_mir_arx64_with_units_and_settings<W: Write>(
    module: &MirModule,
    writer: &mut W,
    target_os: TargetOperatingSystem,
    _codegen_units: usize,
    _settings: &MirCodegenSettings,
) -> Result<(), LaminaError> {
    if target_os != TargetOperatingSystem::Unknown {
        return Err(LaminaError::ValidationError(format!(
            "ARX64 currently supports only unknown/bare-metal target OS, not {target_os:?}"
        )));
    }

    crate::mir_codegen::validate_module_call_parameters(module, TargetArchitecture::Arx64)?;

    writeln!(writer, ".section .text").map_err(io_error)?;
    let mut functions: Vec<_> = module
        .functions
        .iter()
        .filter(|(name, _)| !module.is_external(name))
        .collect();
    functions.sort_by(|a, b| a.0.cmp(b.0));

    for (name, func) in functions {
        emit_function(name, func, writer)?;
    }

    Ok(())
}

fn emit_function<W: Write>(name: &str, func: &Function, writer: &mut W) -> Result<(), LaminaError> {
    let mut stack = StackLayout::new(func);
    let frame_size = align_to(stack.bytes + 16, 16);

    writeln!(writer, ".globl {name}").map_err(io_error)?;
    writeln!(writer, "{name}:").map_err(io_error)?;
    emit_addi(writer, SP, SP, -(frame_size as i64))?;
    emit_store(writer, "sd", RA, frame_size as i64 - 8, SP)?;
    emit_store(writer, "sd", FP, frame_size as i64 - 16, SP)?;
    emit_addi(writer, FP, SP, frame_size as i64)?;

    for (idx, param) in func.sig.params.iter().enumerate() {
        let Some(src) = ARG_REGS.get(idx) else {
            return unsupported(format!(
                "ARX64 supports up to {} integer parameters",
                ARG_REGS.len()
            ));
        };
        let offset = stack.slot_for(&param.reg)?;
        emit_store(writer, "sd", src, offset, FP)?;
    }

    let mut blocks: Vec<_> = func.blocks.iter().collect();
    blocks.sort_by(|a, b| match (a.label.as_str(), b.label.as_str()) {
        ("entry", "entry") => std::cmp::Ordering::Equal,
        ("entry", _) => std::cmp::Ordering::Less,
        (_, "entry") => std::cmp::Ordering::Greater,
        _ => a.label.cmp(&b.label),
    });

    for block in blocks {
        writeln!(writer, "{}:", block_label(name, &block.label)).map_err(io_error)?;
        for inst in &block.instructions {
            emit_instruction(name, inst, &mut stack, frame_size, writer)?;
        }
    }

    Ok(())
}

fn emit_instruction<W: Write>(
    func_name: &str,
    inst: &Instruction,
    stack: &mut StackLayout,
    frame_size: usize,
    writer: &mut W,
) -> Result<(), LaminaError> {
    match inst {
        Instruction::Comment { text } => writeln!(writer, "    # {text}").map_err(io_error),
        Instruction::IntBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            ensure_gpr_scalar(ty)?;
            emit_operand_to_reg(writer, lhs, SCR0, stack)?;
            emit_operand_to_reg(writer, rhs, SCR1, stack)?;
            let mnemonic = match op {
                IntBinOp::Add => "add",
                IntBinOp::Sub => "sub",
                IntBinOp::Mul => "mul",
                IntBinOp::UDiv => "divu",
                IntBinOp::SDiv => "div",
                IntBinOp::URem => "remu",
                IntBinOp::SRem => "rem",
                IntBinOp::And => "and",
                IntBinOp::Or => "or",
                IntBinOp::Xor => "xor",
                IntBinOp::Shl => "sll",
                IntBinOp::LShr => "srl",
                IntBinOp::AShr => "sra",
            };
            writeln!(writer, "    {mnemonic} {SCR2}, {SCR0}, {SCR1}").map_err(io_error)?;
            store_reg_to_dst(writer, SCR2, dst, stack)
        }
        Instruction::IntCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            ensure_gpr_scalar(ty)?;
            emit_operand_to_reg(writer, lhs, SCR0, stack)?;
            emit_operand_to_reg(writer, rhs, SCR1, stack)?;
            match op {
                IntCmpOp::SLt => {
                    writeln!(writer, "    slt {SCR2}, {SCR0}, {SCR1}").map_err(io_error)?
                }
                IntCmpOp::ULt => {
                    writeln!(writer, "    sltu {SCR2}, {SCR0}, {SCR1}").map_err(io_error)?
                }
                IntCmpOp::SGt => {
                    writeln!(writer, "    slt {SCR2}, {SCR1}, {SCR0}").map_err(io_error)?
                }
                IntCmpOp::UGt => {
                    writeln!(writer, "    sltu {SCR2}, {SCR1}, {SCR0}").map_err(io_error)?
                }
                IntCmpOp::Eq
                | IntCmpOp::Ne
                | IntCmpOp::SLe
                | IntCmpOp::ULe
                | IntCmpOp::SGe
                | IntCmpOp::UGe => {
                    let set = unique_local("cmp_set");
                    let done = unique_local("cmp_done");
                    writeln!(writer, "    addi {SCR2}, r0, 0").map_err(io_error)?;
                    let branch = match op {
                        IntCmpOp::Eq => "beq",
                        IntCmpOp::Ne => "bne",
                        IntCmpOp::SLe | IntCmpOp::SGe => "bge",
                        IntCmpOp::ULe | IntCmpOp::UGe => "bgeu",
                        _ => unreachable!(),
                    };
                    let (lhs_reg, rhs_reg) = match op {
                        IntCmpOp::SLe | IntCmpOp::ULe => (SCR1, SCR0),
                        _ => (SCR0, SCR1),
                    };
                    writeln!(writer, "    {branch} {lhs_reg}, {rhs_reg}, {set}")
                        .map_err(io_error)?;
                    writeln!(writer, "    jal r0, {done}").map_err(io_error)?;
                    writeln!(writer, "{set}:").map_err(io_error)?;
                    writeln!(writer, "    addi {SCR2}, r0, 1").map_err(io_error)?;
                    writeln!(writer, "{done}:").map_err(io_error)?;
                }
            }
            store_reg_to_dst(writer, SCR2, dst, stack)
        }
        Instruction::Lea { dst, base, offset } => {
            emit_reg_to_reg(writer, base, SCR0, stack)?;
            emit_addi_or_load_add(writer, SCR2, SCR0, i64::from(*offset))?;
            store_reg_to_dst(writer, SCR2, dst, stack)
        }
        Instruction::Load { ty, dst, addr, .. } => {
            ensure_gpr_scalar(ty)?;
            let (base, offset) = base_offset(addr)?;
            emit_reg_to_reg(writer, base, SCR0, stack)?;
            writeln!(
                writer,
                "    {} {}, {}({})",
                load_mnemonic(ty)?,
                SCR1,
                offset,
                SCR0
            )
            .map_err(io_error)?;
            store_reg_to_dst(writer, SCR1, dst, stack)
        }
        Instruction::Store { ty, src, addr, .. } => {
            ensure_gpr_scalar(ty)?;
            let (base, offset) = base_offset(addr)?;
            emit_reg_to_reg(writer, base, SCR0, stack)?;
            emit_operand_to_reg(writer, src, SCR1, stack)?;
            writeln!(
                writer,
                "    {} {}, {}({})",
                store_mnemonic(ty)?,
                SCR1,
                offset,
                SCR0
            )
            .map_err(io_error)
        }
        Instruction::Jmp { target } => {
            writeln!(writer, "    jal r0, {}", block_label(func_name, target)).map_err(io_error)
        }
        Instruction::Br {
            cond,
            true_target,
            false_target,
        } => {
            emit_reg_to_reg(writer, cond, SCR0, stack)?;
            writeln!(
                writer,
                "    bne {}, r0, {}",
                SCR0,
                block_label(func_name, true_target)
            )
            .map_err(io_error)?;
            writeln!(
                writer,
                "    jal r0, {}",
                block_label(func_name, false_target)
            )
            .map_err(io_error)
        }
        Instruction::Call { name, args, ret } => {
            if name == "writebyte" && args.len() == 1 {
                emit_operand_to_reg(writer, &args[0], SCR0, stack)?;
                emit_li(writer, SCR1, UART_BASE)?;
                writeln!(writer, "    sb {SCR0}, 0({SCR1})").map_err(io_error)?;
                if let Some(dst) = ret {
                    emit_li(writer, RET, 1)?;
                    store_reg_to_dst(writer, RET, dst, stack)?;
                }
                return Ok(());
            }
            if name == "arx64_halt" && args.is_empty() {
                writeln!(writer, "    ebreak").map_err(io_error)?;
                if let Some(dst) = ret {
                    emit_li(writer, RET, 0)?;
                    store_reg_to_dst(writer, RET, dst, stack)?;
                }
                return Ok(());
            }
            if name == "arx64_boot_info" && args.is_empty() {
                if let Some(dst) = ret {
                    store_reg_to_dst(writer, "r10", dst, stack)?;
                }
                return Ok(());
            }
            if args.len() > ARG_REGS.len() {
                return unsupported(format!(
                    "ARX64 calls support up to {} integer args",
                    ARG_REGS.len()
                ));
            }
            for (idx, arg) in args.iter().enumerate() {
                emit_operand_to_reg(writer, arg, ARG_REGS[idx], stack)?;
            }
            writeln!(writer, "    jal {RA}, {name}").map_err(io_error)?;
            if let Some(dst) = ret {
                store_reg_to_dst(writer, RET, dst, stack)?;
            }
            Ok(())
        }
        Instruction::Ret { value } => {
            if let Some(value) = value {
                emit_operand_to_reg(writer, value, RET, stack)?;
            }
            emit_load(writer, "ld", RA, frame_size as i64 - 8, SP)?;
            emit_load(writer, "ld", FP, frame_size as i64 - 16, SP)?;
            emit_addi(writer, SP, SP, frame_size as i64)?;
            writeln!(writer, "    jalr r0, 0({RA})").map_err(io_error)
        }
        Instruction::Unreachable => writeln!(writer, "    ebreak").map_err(io_error),
        Instruction::SafePoint | Instruction::StackMap { .. } | Instruction::PatchPoint { .. } => {
            Ok(())
        }
        other => unsupported(format!("ARX64 backend does not support MIR op {other:?}")),
    }
}

struct StackLayout {
    slots: HashMap<Register, i64>,
    bytes: usize,
}

impl StackLayout {
    fn new(func: &Function) -> Self {
        let mut regs = BTreeMap::<String, Register>::new();
        for param in &func.sig.params {
            regs.insert(format!("{:?}", param.reg), param.reg.clone());
        }
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(reg) = inst.def_reg() {
                    regs.insert(format!("{reg:?}"), reg.clone());
                }
                for reg in inst.use_regs() {
                    regs.insert(format!("{reg:?}"), reg.clone());
                }
            }
        }

        let mut slots = HashMap::new();
        let mut next = -8i64;
        for reg in regs.values() {
            if matches!(reg, Register::Virtual(_)) {
                slots.insert(reg.clone(), next);
                next -= 8;
            }
        }

        Self {
            slots,
            bytes: (-next - 8) as usize,
        }
    }

    fn slot_for(&mut self, reg: &Register) -> Result<i64, LaminaError> {
        match reg {
            Register::Virtual(_) => self.slots.get(reg).copied().ok_or_else(|| {
                LaminaError::CodegenError(CodegenError::UnsupportedFeature(format!(
                    "missing ARX64 stack slot for {reg:?}"
                )))
            }),
            Register::Physical(_) => unsupported("physical destination stack slot unsupported"),
        }
    }
}

fn emit_operand_to_reg<W: Write>(
    writer: &mut W,
    operand: &Operand,
    target: &str,
    stack: &mut StackLayout,
) -> Result<(), LaminaError> {
    match operand {
        Operand::Register(reg) => emit_reg_to_reg(writer, reg, target, stack),
        Operand::Immediate(imm) => emit_immediate(writer, imm, target),
    }
}

fn emit_reg_to_reg<W: Write>(
    writer: &mut W,
    reg: &Register,
    target: &str,
    stack: &mut StackLayout,
) -> Result<(), LaminaError> {
    match reg {
        Register::Virtual(_) => {
            let offset = stack.slot_for(reg)?;
            emit_load(writer, "ld", target, offset, FP)
        }
        Register::Physical(p) => {
            writeln!(writer, "    addi {}, {}, 0", target, p.name).map_err(io_error)
        }
    }
}

fn store_reg_to_dst<W: Write>(
    writer: &mut W,
    src: &str,
    dst: &Register,
    stack: &mut StackLayout,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(_) => {
            let offset = stack.slot_for(dst)?;
            emit_store(writer, "sd", src, offset, FP)
        }
        Register::Physical(p) => {
            writeln!(writer, "    addi {}, {}, 0", p.name, src).map_err(io_error)
        }
    }
}

fn emit_immediate<W: Write>(
    writer: &mut W,
    imm: &Immediate,
    target: &str,
) -> Result<(), LaminaError> {
    let value = match imm {
        Immediate::I8(v) => i64::from(*v),
        Immediate::I16(v) => i64::from(*v),
        Immediate::I32(v) => i64::from(*v),
        Immediate::I64(v) => *v,
        Immediate::F32(_) | Immediate::F64(_) => {
            return unsupported("ARX64 backend does not support floating immediates");
        }
    };
    emit_li(writer, target, value)
}

fn emit_li<W: Write>(writer: &mut W, target: &str, value: i64) -> Result<(), LaminaError> {
    if fits_i12(value) {
        return emit_addi(writer, target, "r0", value);
    }
    let hi = ((value + 0x800) >> 12) & 0x000f_ffff;
    let lo = value - (hi << 12);
    writeln!(writer, "    lui {target}, {hi}").map_err(io_error)?;
    emit_addi(writer, target, target, lo)
}

fn emit_addi_or_load_add<W: Write>(
    writer: &mut W,
    dst: &str,
    base: &str,
    imm: i64,
) -> Result<(), LaminaError> {
    if fits_i12(imm) {
        emit_addi(writer, dst, base, imm)
    } else {
        emit_li(writer, SCR1, imm)?;
        writeln!(writer, "    add {dst}, {base}, {SCR1}").map_err(io_error)
    }
}

fn emit_addi<W: Write>(writer: &mut W, dst: &str, src: &str, imm: i64) -> Result<(), LaminaError> {
    if !fits_i12(imm) {
        return unsupported(format!("ARX64 addi immediate {imm} outside 12-bit range"));
    }
    writeln!(writer, "    addi {dst}, {src}, {imm}").map_err(io_error)
}

fn emit_load<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    dst: &str,
    offset: i64,
    base: &str,
) -> Result<(), LaminaError> {
    if !fits_i12(offset) {
        return unsupported(format!("ARX64 load offset {offset} outside 12-bit range"));
    }
    writeln!(writer, "    {mnemonic} {dst}, {offset}({base})").map_err(io_error)
}

fn emit_store<W: Write>(
    writer: &mut W,
    mnemonic: &str,
    src: &str,
    offset: i64,
    base: &str,
) -> Result<(), LaminaError> {
    if !fits_i12(offset) {
        return unsupported(format!("ARX64 store offset {offset} outside 12-bit range"));
    }
    writeln!(writer, "    {mnemonic} {src}, {offset}({base})").map_err(io_error)
}

fn base_offset(addr: &AddressMode) -> Result<(&Register, i64), LaminaError> {
    match addr {
        AddressMode::BaseOffset { base, offset } => Ok((base, i64::from(*offset))),
        AddressMode::BaseIndexScale { .. } => {
            unsupported("ARX64 backend does not support indexed addressing yet")
        }
    }
}

fn load_mnemonic(ty: &MirType) -> Result<&'static str, LaminaError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok("lb"),
        MirType::Scalar(ScalarType::I16) => Ok("lh"),
        MirType::Scalar(ScalarType::I32) => Ok("lw"),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok("ld"),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) | MirType::Vector(_) => {
            unsupported(format!("ARX64 load unsupported type {ty}"))
        }
    }
}

fn store_mnemonic(ty: &MirType) -> Result<&'static str, LaminaError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok("sb"),
        MirType::Scalar(ScalarType::I16) => Ok("sh"),
        MirType::Scalar(ScalarType::I32) => Ok("sw"),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok("sd"),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) | MirType::Vector(_) => {
            unsupported(format!("ARX64 store unsupported type {ty}"))
        }
    }
}

fn ensure_gpr_scalar(ty: &MirType) -> Result<(), LaminaError> {
    match ty {
        MirType::Scalar(
            ScalarType::I1
            | ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::Ptr,
        ) => Ok(()),
        _ => unsupported(format!(
            "ARX64 backend supports integer scalar MIR only, got {ty}"
        )),
    }
}

fn block_label(func: &str, block: &str) -> String {
    format!(".L_{}_{}", sanitize(func), sanitize(block))
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn unique_local(prefix: &str) -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!(
        ".L_arx64_{}_{}",
        prefix,
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn align_to(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

fn fits_i12(value: i64) -> bool {
    (-2048..=2047).contains(&value)
}

fn io_error(error: std::io::Error) -> LaminaError {
    LaminaError::CodegenError(CodegenError::InvalidCodegenOptions(format!(
        "ARX64 asm write failed: {error}"
    )))
}

fn unsupported<T>(message: impl Into<String>) -> Result<T, LaminaError> {
    Err(LaminaError::CodegenError(CodegenError::UnsupportedFeature(
        message.into(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{Block, Function, Immediate, Module, Operand, Signature, VirtualReg};

    #[test]
    fn emits_minimal_integer_return_function() {
        let mut module = Module::new("arx64_test");
        let mut func = Function::new(Signature::new("main").with_return(MirType::i64()));
        let mut entry = Block::new("entry");
        entry.push(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(7))),
        });
        func.add_block(entry);
        module.add_function(func);

        let mut asm = Vec::new();
        generate_mir_arx64(&module, &mut asm, TargetOperatingSystem::Unknown)
            .expect("ARX64 asm generation");
        let asm = String::from_utf8(asm).expect("utf8 asm");

        assert!(asm.contains(".globl main"));
        assert!(asm.contains("addi r5, r0, 7"));
        assert!(asm.contains("jalr r0, 0(r1)"));
    }

    #[test]
    fn emits_arx64_writebyte_and_halt_builtins() {
        let mut module = Module::new("arx64_io_test");
        let mut func = Function::new(Signature::new("main").with_return(MirType::i64()));
        let mut entry = Block::new("entry");
        entry.push(Instruction::Call {
            name: "writebyte".to_string(),
            args: vec![Operand::Immediate(Immediate::I64(65))],
            ret: None,
        });
        entry.push(Instruction::Call {
            name: "arx64_halt".to_string(),
            args: vec![],
            ret: None,
        });
        entry.push(Instruction::Call {
            name: "arx64_boot_info".to_string(),
            args: vec![],
            ret: Some(Register::Virtual(VirtualReg::gpr(1))),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        });
        func.add_block(entry);
        module.add_function(func);

        let mut asm = Vec::new();
        generate_mir_arx64(&module, &mut asm, TargetOperatingSystem::Unknown)
            .expect("ARX64 asm generation");
        let asm = String::from_utf8(asm).expect("utf8 asm");

        assert!(asm.contains("sb r11, 0(r12)"));
        assert!(asm.contains("ebreak"));
        assert!(asm.contains("sd r10, "));
    }
}
