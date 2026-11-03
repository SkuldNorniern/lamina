use super::*;
use lamina_platform::{TargetArchitecture, VaListAbi};

const DARWIN_CURSOR_BYTES: i32 = 8;

const STACK_FIELD: i32 = 0;
const GR_TOP_FIELD: i32 = 8;
const VR_TOP_FIELD: i32 = 16;
const GR_OFFS_FIELD: i32 = 24;
const VR_OFFS_FIELD: i32 = 28;
const VA_LIST_BYTES: i32 = match TargetArchitecture::Aarch64.va_list() {
    VaListAbi::Defined { size, .. } => size as i32,
    VaListAbi::Undefined => panic!("AArch64 lowers va_list but the target table defines none"),
};
const _: () = assert!(VR_OFFS_FIELD + 4 == VA_LIST_BYTES);

const GP_REGISTERS: i32 = ARGUMENT_REGISTERS as i32;
const GR_SLOT_BYTES: i32 = 8;
const VR_SLOT_BYTES: i32 = 16;
const GR_SAVE_BYTES: i32 = GP_REGISTERS * GR_SLOT_BYTES;
const VR_SAVE_BYTES: i32 = GP_REGISTERS * VR_SLOT_BYTES;
pub(super) const SAVE_AREA_BYTES: i32 = GR_SAVE_BYTES + VR_SAVE_BYTES;

#[derive(Clone, Copy, Debug)]
pub(super) struct Aapcs64Area {
    stack: i32,
    save: i32,
    gr_offs: i32,
    vr_offs: i32,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum VaArea {
    Darwin { start: i32 },
    Aapcs64(Aapcs64Area),
}

impl VaArea {
    pub(super) fn darwin(start: i32) -> Self {
        Self::Darwin { start }
    }

    pub(super) fn aapcs64(save: i32, stack: i32, places: &[ArgPlace]) -> Self {
        let gp_words: usize = places
            .iter()
            .map(|place| match place {
                ArgPlace::Gpr { words, .. } => *words,
                ArgPlace::Fpr { .. } | ArgPlace::Homogeneous { .. } | ArgPlace::Stack { .. } => 0,
            })
            .sum();
        let fp_regs: usize = places
            .iter()
            .map(|place| match place {
                ArgPlace::Fpr { .. } => 1,
                ArgPlace::Homogeneous { count, .. } => *count,
                ArgPlace::Gpr { .. } | ArgPlace::Stack { .. } => 0,
            })
            .sum();
        Self::Aapcs64(Aapcs64Area {
            stack,
            save,
            gr_offs: -GR_SLOT_BYTES * ARGUMENT_REGISTERS.saturating_sub(gp_words) as i32,
            vr_offs: -VR_SLOT_BYTES * ARGUMENT_REGISTERS.saturating_sub(fp_regs) as i32,
        })
    }
}

pub(super) fn spill_save_area<W: Write>(w: &mut W, area: &VaArea) -> Result<(), LaminaError> {
    let VaArea::Aapcs64(area) = *area else {
        return Ok(());
    };
    emit_frame_slot_address(w, "x16", area.save)?;
    for index in 0..GP_REGISTERS {
        writeln!(w, "    str x{index}, [x16, #{}]", GR_SLOT_BYTES * index)?;
    }
    for index in 0..GP_REGISTERS {
        writeln!(
            w,
            "    str q{index}, [x16, #{}]",
            GR_SAVE_BYTES + VR_SLOT_BYTES * index
        )?;
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

fn va_arg_is_float(ty: &MirType) -> Result<bool, LaminaError> {
    match ty {
        MirType::Scalar(ScalarType::F64) => Ok(true),
        MirType::Scalar(
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 | ScalarType::Ptr,
        ) => Ok(false),
        _ => Err(unsupported(format!(
            "va_arg of {ty:?} is not supported; use an integer, pointer or f64"
        ))),
    }
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    target_os: TargetOperatingSystem,
    area: Option<VaArea>,
    label_prefix: &str,
) -> Result<(), LaminaError> {
    if !os_uses_macho(target_os) && !os_uses_elf(target_os) {
        return Err(unsupported(format!(
            "variadic argument `{inst}` is not yet supported by the AArch64 backend on {target_os:?}; it is implemented for Darwin and ELF targets"
        )));
    }
    let Some(area) = area else {
        return Err(unsupported(
            "variadic arguments are only available inside a variadic function",
        ));
    };
    match area {
        VaArea::Darwin { start } => emit_darwin(inst, w, frame, ra, start),
        VaArea::Aapcs64(area) => emit_aapcs64(inst, w, frame, ra, &area, label_prefix),
    }
}

fn emit_darwin<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    start: i32,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::VaStart { ap } => {
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            emit_frame_slot_address(w, cursor, start)?;
            writeln!(w, "    str {cursor}, [{list}]")?;
            ra.free_scratch(cursor);
            ra.free_scratch(list);
        }
        MirInst::VaArg { ty, dst, ap } => {
            va_arg_is_float(ty)?;
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let next = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            writeln!(w, "    ldr {cursor}, [{list}]")?;
            writeln!(w, "    add {next}, {cursor}, #{DARWIN_CURSOR_BYTES}")?;
            writeln!(w, "    str {next}, [{list}]")?;
            writeln!(w, "    ldr {cursor}, [{cursor}]")?;
            emit_narrow_param_extension(w, cursor, ty)?;
            store_result(w, dst, cursor, frame, ra)?;
            ra.free_scratch(next);
            ra.free_scratch(cursor);
            ra.free_scratch(list);
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            let from = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let to = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, src, from, frame, ra)?;
            emit_materialize_operand(w, dst, to, frame, ra)?;
            writeln!(w, "    ldr {cursor}, [{from}]")?;
            writeln!(w, "    str {cursor}, [{to}]")?;
            ra.free_scratch(cursor);
            ra.free_scratch(to);
            ra.free_scratch(from);
        }
        _ => return Err(unsupported("not a variadic instruction")),
    }
    Ok(())
}

fn emit_aapcs64<W: Write>(
    inst: &MirInst,
    w: &mut W,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    area: &Aapcs64Area,
    label_prefix: &str,
) -> Result<(), LaminaError> {
    match inst {
        MirInst::VaStart { ap } => {
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let cursor = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            emit_frame_slot_address(w, cursor, area.stack)?;
            writeln!(w, "    str {cursor}, [{list}, #{STACK_FIELD}]")?;
            emit_frame_slot_address(w, cursor, area.save + GR_SAVE_BYTES)?;
            writeln!(w, "    str {cursor}, [{list}, #{GR_TOP_FIELD}]")?;
            emit_frame_slot_address(w, cursor, area.save + SAVE_AREA_BYTES)?;
            writeln!(w, "    str {cursor}, [{list}, #{VR_TOP_FIELD}]")?;
            let offsets = u64::from(area.gr_offs as u32) | (u64::from(area.vr_offs as u32) << 32);
            emit_mov_imm64(w, cursor, offsets)?;
            writeln!(w, "    str {cursor}, [{list}, #{GR_OFFS_FIELD}]")?;
            ra.free_scratch(cursor);
            ra.free_scratch(list);
        }
        MirInst::VaArg { ty, dst, ap } => {
            let is_float = va_arg_is_float(ty)?;
            let Register::Virtual(vreg) = dst else {
                return Err(unsupported("va_arg must define a virtual register"));
            };
            let (top_field, offs_field, step) = if is_float {
                (VR_TOP_FIELD, VR_OFFS_FIELD, VR_SLOT_BYTES)
            } else {
                (GR_TOP_FIELD, GR_OFFS_FIELD, GR_SLOT_BYTES)
            };
            let stack = format!("{label_prefix}va_arg{}.stack", vreg.id);
            let done = format!("{label_prefix}va_arg{}.done", vreg.id);
            let list = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let offs = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let value = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, ap, list, frame, ra)?;
            writeln!(w, "    ldrsw {offs}, [{list}, #{offs_field}]")?;
            writeln!(w, "    cmp {offs}, #0")?;
            writeln!(w, "    b.ge {stack}")?;
            writeln!(w, "    ldr {value}, [{list}, #{top_field}]")?;
            writeln!(w, "    add {value}, {value}, {offs}")?;
            writeln!(w, "    add {offs}, {offs}, #{step}")?;
            writeln!(w, "    str {}, [{list}, #{offs_field}]", w_alias(offs))?;
            writeln!(w, "    b {done}")?;
            writeln!(w, "{stack}:")?;
            writeln!(w, "    ldr {value}, [{list}, #{STACK_FIELD}]")?;
            writeln!(w, "    add {offs}, {value}, #{GR_SLOT_BYTES}")?;
            writeln!(w, "    str {offs}, [{list}, #{STACK_FIELD}]")?;
            writeln!(w, "{done}:")?;
            writeln!(w, "    ldr {value}, [{value}]")?;
            emit_narrow_param_extension(w, value, ty)?;
            store_result(w, dst, value, frame, ra)?;
            ra.free_scratch(value);
            ra.free_scratch(offs);
            ra.free_scratch(list);
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            let from = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let to = ra.alloc_scratch().ok_or_else(scratch_error)?;
            let word = ra.alloc_scratch().ok_or_else(scratch_error)?;
            emit_materialize_operand(w, src, from, frame, ra)?;
            emit_materialize_operand(w, dst, to, frame, ra)?;
            for offset in (0..VA_LIST_BYTES).step_by(GR_SLOT_BYTES as usize) {
                writeln!(w, "    ldr {word}, [{from}, #{offset}]")?;
                writeln!(w, "    str {word}, [{to}, #{offset}]")?;
            }
            ra.free_scratch(word);
            ra.free_scratch(to);
            ra.free_scratch(from);
        }
        _ => return Err(unsupported("not a variadic instruction")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aapcs(places: &[ArgPlace]) -> Aapcs64Area {
        match VaArea::aapcs64(-400, 32, places) {
            VaArea::Aapcs64(area) => area,
            VaArea::Darwin { .. } => panic!("expected the AAPCS64 area"),
        }
    }

    #[test]
    fn named_registers_move_the_offsets_past_them() {
        let area = aapcs(&[
            ArgPlace::Gpr { first: 0, words: 1 },
            ArgPlace::Fpr { index: 0 },
        ]);
        assert_eq!((area.stack, area.save), (32, -400));
        assert_eq!((area.gr_offs, area.vr_offs), (-56, -112));
    }

    #[test]
    fn an_aggregate_in_two_registers_counts_two_words() {
        let area = aapcs(&[ArgPlace::Gpr { first: 0, words: 2 }]);
        assert_eq!((area.gr_offs, area.vr_offs), (-48, -128));
    }

    #[test]
    fn no_named_registers_leaves_every_register_to_va_arg() {
        let area = aapcs(&[]);
        assert_eq!((area.gr_offs, area.vr_offs), (-64, -128));
    }

    #[test]
    fn a_full_register_file_leaves_nothing_for_va_arg() {
        let gprs: Vec<ArgPlace> = (0..8)
            .map(|first| ArgPlace::Gpr { first, words: 1 })
            .collect();
        let area = aapcs(&gprs);
        assert_eq!((area.gr_offs, area.vr_offs), (0, -128));
    }

    #[test]
    fn save_area_covers_eight_general_and_eight_vector_registers() {
        assert_eq!(SAVE_AREA_BYTES, 192);
        assert_eq!(GR_SAVE_BYTES + VR_SAVE_BYTES, SAVE_AREA_BYTES);
    }

    #[test]
    fn spill_writes_every_argument_register_into_the_save_area() -> Result<(), LaminaError> {
        let area = VaArea::aapcs64(-400, 32, &[]);
        let mut output = Vec::new();
        spill_save_area(&mut output, &area)?;
        let text = String::from_utf8_lossy(&output);
        assert!(text.starts_with("    sub x16, x29, #400\n"), "{text}");
        assert_eq!(text.matches("    str x").count(), 8, "{text}");
        assert_eq!(text.matches("    str q").count(), 8, "{text}");
        assert!(text.contains("    str q7, [x16, #176]\n"), "{text}");
        Ok(())
    }

    #[test]
    fn darwin_spills_nothing() -> Result<(), LaminaError> {
        let mut output = Vec::new();
        spill_save_area(&mut output, &VaArea::darwin(16))?;
        assert!(output.is_empty());
        Ok(())
    }
}
