//! System V variadic functions: the register save area a variadic callee spills on entry,
//! and the `va_list` operations that read it.

use super::*;
use lamina_platform::{TargetArchitecture, VaListAbi};

/// Byte offsets of the four System V `va_list` fields, and the size the target table
/// states. The fields fill it exactly, so widening one without the other is a build
/// error rather than an overrun.
const GP_OFFSET_FIELD: u32 = 0;
const FP_OFFSET_FIELD: u32 = 4;
const OVERFLOW_FIELD: u32 = 8;
const SAVE_FIELD: u32 = 16;
const VA_LIST_BYTES: u32 = match TargetArchitecture::X86_64.va_list() {
    VaListAbi::Defined { size, .. } => size as u32,
    VaListAbi::Undefined => panic!("x86_64 lowers va_list but the target table defines none"),
};
const _: () = assert!(SAVE_FIELD + GP_SLOT_BYTES == VA_LIST_BYTES);

const GP_ARG_REGS: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];
const GP_SAVE_BYTES: u32 = 48;
const VECTOR_SAVE_BYTES: u32 = 128;
const VECTOR_SLOT_BYTES: u32 = 16;
const GP_SLOT_BYTES: u32 = 8;

/// Eight-byte slots in the register save area, which is 176 bytes.
pub(super) const SAVE_AREA_SLOTS: usize = ((GP_SAVE_BYTES + VECTOR_SAVE_BYTES) / 8) as usize;

/// Where a variadic callee's `va_list` finds its arguments, relative to rbp.
#[derive(Clone, Copy, Debug)]
pub(super) struct VaArea {
    save: i32,
    overflow: i32,
    gp_offset: u32,
    fp_offset: u32,
}

impl VaArea {
    /// `save` is the offset of the save area's first byte. `locations` classifies the
    /// named parameters, which the `va_list` starts past.
    pub(super) fn new(save: i32, locations: &[ArgLocation]) -> Self {
        let gp_words: usize = locations
            .iter()
            .map(|location| match location {
                ArgLocation::Gpr(_) => 1,
                ArgLocation::GprWords { words, .. } => *words,
                _ => 0,
            })
            .sum();
        let vector_params = locations
            .iter()
            .filter(|location| matches!(location, ArgLocation::Xmm(_)))
            .count();
        let stack_words = stack_argument_count(locations);
        Self {
            save,
            overflow: 16 + GP_SLOT_BYTES as i32 * stack_words as i32,
            gp_offset: GP_SLOT_BYTES * gp_words as u32,
            fp_offset: GP_SAVE_BYTES + VECTOR_SLOT_BYTES * vector_params as u32,
        }
    }
}

/// Spills the argument registers into the save area on entry. The vector registers are
/// only read when `al` is nonzero, which the caller sets to the number of vector
/// registers it passed, so a caller with no floating point arguments never needs SSE
/// state to be valid.
pub(super) fn spill_register_save_area<W: Write>(
    writer: &mut W,
    area: &VaArea,
    func_name: &str,
) -> io::Result<()> {
    for (index, reg) in GP_ARG_REGS.iter().enumerate() {
        writeln!(
            writer,
            "    movq %{reg}, {}(%rbp)",
            area.save + (GP_SLOT_BYTES as i32) * index as i32
        )?;
    }
    let skip = format!(".L_{func_name}.va_xmm_skip");
    writeln!(writer, "    testb %al, %al")?;
    writeln!(writer, "    je {skip}")?;
    for index in 0..8 {
        writeln!(
            writer,
            "    movups %xmm{index}, {}(%rbp)",
            area.save + GP_SAVE_BYTES as i32 + (VECTOR_SLOT_BYTES as i32) * index
        )?;
    }
    writeln!(writer, "{skip}:")
}

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    area: Option<&VaArea>,
    target_os: TargetOperatingSystem,
    func_name: &str,
) -> Result<(), LaminaError> {
    if target_os == TargetOperatingSystem::Windows {
        return Err(unsupported(
            "variadic arguments are not yet supported on Windows x64; only System V",
        ));
    }
    let Some(area) = area else {
        return Err(unsupported(
            "variadic arguments are only available inside a variadic function",
        ));
    };
    match inst {
        MirInst::VaStart { ap } => {
            load_operand_to_register(ap, writer, reg_alloc, stack_slots, "rcx")?;
            writeln!(writer, "    movl ${}, (%rcx)", area.gp_offset)?;
            writeln!(
                writer,
                "    movl ${}, {FP_OFFSET_FIELD}(%rcx)",
                area.fp_offset
            )?;
            writeln!(writer, "    leaq {}(%rbp), %rax", area.overflow)?;
            writeln!(writer, "    movq %rax, {OVERFLOW_FIELD}(%rcx)")?;
            writeln!(writer, "    leaq {}(%rbp), %rax", area.save)?;
            writeln!(writer, "    movq %rax, {SAVE_FIELD}(%rcx)")?;
        }
        MirInst::VaArg { ty, dst, ap } => {
            let is_float = match ty {
                MirType::Scalar(ScalarType::F64) => true,
                MirType::Scalar(
                    ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::Ptr,
                ) => false,
                _ => {
                    return Err(unsupported(format!(
                        "va_arg of {ty:?} is not supported; use an integer, pointer or f64"
                    )));
                }
            };
            let Register::Virtual(vreg) = dst else {
                return Err(unsupported("va_arg must define a virtual register"));
            };
            let stack = format!(".L_{func_name}.va_arg{}.stack", vreg.id);
            let done = format!(".L_{func_name}.va_arg{}.done", vreg.id);
            let (cursor, limit, step) = if is_float {
                (
                    FP_OFFSET_FIELD,
                    GP_SAVE_BYTES + VECTOR_SAVE_BYTES,
                    VECTOR_SLOT_BYTES,
                )
            } else {
                (GP_OFFSET_FIELD, GP_SAVE_BYTES, GP_SLOT_BYTES)
            };
            load_operand_to_register(ap, writer, reg_alloc, stack_slots, "rcx")?;
            writeln!(writer, "    movl {cursor}(%rcx), %eax")?;
            writeln!(writer, "    cmpl ${limit}, %eax")?;
            writeln!(writer, "    jae {stack}")?;
            writeln!(writer, "    movq {SAVE_FIELD}(%rcx), %rdx")?;
            writeln!(writer, "    addq %rax, %rdx")?;
            writeln!(writer, "    addl ${step}, {cursor}(%rcx)")?;
            writeln!(writer, "    jmp {done}")?;
            writeln!(writer, "{stack}:")?;
            writeln!(writer, "    movq {OVERFLOW_FIELD}(%rcx), %rdx")?;
            writeln!(writer, "    leaq {GP_SLOT_BYTES}(%rdx), %rax")?;
            writeln!(writer, "    movq %rax, {OVERFLOW_FIELD}(%rcx)")?;
            writeln!(writer, "{done}:")?;
            writeln!(writer, "    movq (%rdx), %rax")?;
            canonicalize_integer(writer, "rax", ty)?;
            store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "rcx")?;
            load_operand_to_register(dst, writer, reg_alloc, stack_slots, "rdx")?;
            for offset in (0..VA_LIST_BYTES).step_by(GP_SLOT_BYTES as usize) {
                writeln!(writer, "    movq {offset}(%rcx), %rax")?;
                writeln!(writer, "    movq %rax, {offset}(%rdx)")?;
            }
        }
        _ => {
            return Err(unsupported("not a variadic instruction"));
        }
    }
    Ok(())
}
