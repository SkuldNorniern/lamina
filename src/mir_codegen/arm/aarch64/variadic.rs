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
pub(super) struct WindowsArea {
    start: Option<i32>,
    save: i32,
    first: i32,
}

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
    Windows(WindowsArea),
    Aapcs64(Aapcs64Area),
}

/// General-purpose registers the named parameters in `places` consume.
fn named_gp_words(places: &[ArgPlace]) -> usize {
    places
        .iter()
        .map(|place| match place {
            ArgPlace::Gpr { words, .. } => *words,
            ArgPlace::Fpr { .. } | ArgPlace::Homogeneous { .. } | ArgPlace::Stack { .. } => 0,
        })
        .sum()
}

/// Bytes a Windows variadic callee needs for the general register save area: one slot
/// for every argument register the named parameters left unused.
pub(super) fn windows_save_bytes(places: &[ArgPlace]) -> i32 {
    GR_SLOT_BYTES * ARGUMENT_REGISTERS.saturating_sub(named_gp_words(places)) as i32
}

/// Bytes the frame record occupies so that the Windows save area ends exactly where
/// the incoming stack arguments begin, keeping the whole run contiguous.
pub(super) fn windows_frame_record_bytes(places: &[ArgPlace]) -> i32 {
    (FRAME_RECORD_BYTES + windows_save_bytes(places) + 15) & !15
}

impl VaArea {
    pub(super) fn darwin(start: i32) -> Self {
        Self::Darwin { start }
    }

    /// `record` is the frame record size, which is also the offset of the first
    /// incoming stack argument from the frame pointer; `named_stack` is the space the
    /// named parameters already took there.
    pub(super) fn windows(record: i32, named_stack: i32, places: &[ArgPlace]) -> Self {
        let save_bytes = windows_save_bytes(places);
        let save = record - save_bytes;
        let start = if named_stack > 0 {
            None
        } else if save_bytes > 0 {
            Some(save)
        } else {
            Some(record)
        };
        Self::Windows(WindowsArea {
            start,
            save,
            first: named_gp_words(places).min(ARGUMENT_REGISTERS) as i32,
        })
    }

    pub(super) fn frame_record_bytes(&self) -> i32 {
        match self {
            Self::Windows(area) => area.save + GR_SLOT_BYTES * (GP_REGISTERS - area.first),
            Self::Darwin { .. } | Self::Aapcs64(_) => FRAME_RECORD_BYTES,
        }
    }

    pub(super) fn aapcs64(save: i32, stack: i32, places: &[ArgPlace]) -> Self {
        let gp_words = named_gp_words(places);
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
    let area = match *area {
        VaArea::Aapcs64(area) => area,
        VaArea::Windows(area) => return spill_windows_save_area(w, &area),
        VaArea::Darwin { .. } => return Ok(()),
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

fn spill_windows_save_area<W: Write>(w: &mut W, area: &WindowsArea) -> Result<(), LaminaError> {
    if area.first >= GP_REGISTERS {
        return Ok(());
    }
    emit_frame_slot_address(w, "x16", area.save)?;
    for index in area.first..GP_REGISTERS {
        writeln!(
            w,
            "    str x{index}, [x16, #{}]",
            GR_SLOT_BYTES * (index - area.first)
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
    if !os_uses_macho(target_os) && !os_uses_elf(target_os) && !os_uses_coff(target_os) {
        return Err(unsupported(format!(
            "variadic argument `{inst}` is not yet supported by the AArch64 backend on {target_os:?}; it is implemented for Darwin, ELF and COFF targets"
        )));
    }
    let Some(area) = area else {
        return Err(unsupported(
            "variadic arguments are only available inside a variadic function",
        ));
    };
    match area {
        VaArea::Darwin { start } => emit_pointer_list(inst, w, frame, ra, start),
        VaArea::Windows(area) => {
            let Some(start) = area.start else {
                return Err(unsupported(format!(
                    "variadic argument `{inst}`: a Windows AArch64 variadic function whose named parameters reach the stack is not supported"
                )));
            };
            emit_pointer_list(inst, w, frame, ra, start)
        }
        VaArea::Aapcs64(area) => emit_aapcs64(inst, w, frame, ra, &area, label_prefix),
    }
}

/// Darwin and Windows both hand out a plain pointer that walks one eight-byte slot at
/// a time; they differ only in where that walk starts and what it crosses.
fn emit_pointer_list<W: Write>(
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
            VaArea::Darwin { .. } | VaArea::Windows(_) => panic!("expected the AAPCS64 area"),
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

    fn windows(record: i32, named_stack: i32, places: &[ArgPlace]) -> WindowsArea {
        match VaArea::windows(record, named_stack, places) {
            VaArea::Windows(area) => area,
            VaArea::Darwin { .. } | VaArea::Aapcs64(_) => panic!("expected the Windows area"),
        }
    }

    fn one_named_word() -> [ArgPlace; 1] {
        [ArgPlace::Gpr { first: 0, words: 1 }]
    }

    #[test]
    fn the_windows_save_area_ends_where_the_stack_arguments_begin() {
        let places = one_named_word();
        let record = windows_frame_record_bytes(&places);
        let area = windows(record, 0, &places);
        assert_eq!((record, windows_save_bytes(&places)), (80, 56));
        assert_eq!(area.save + windows_save_bytes(&places), record);
        assert_eq!(area.start, Some(24));
    }

    #[test]
    fn every_count_of_named_words_keeps_the_save_area_adjacent() {
        for named in 0..=ARGUMENT_REGISTERS {
            let places: Vec<ArgPlace> = (0..named)
                .map(|first| ArgPlace::Gpr { first, words: 1 })
                .collect();
            let record = windows_frame_record_bytes(&places);
            let area = windows(record, 0, &places);
            assert_eq!(record % 16, 0, "{named}");
            assert_eq!(area.save + windows_save_bytes(&places), record, "{named}");
            assert_eq!(
                VaArea::windows(record, 0, &places).frame_record_bytes(),
                record
            );
        }
    }

    #[test]
    fn a_fully_named_register_file_starts_the_walk_at_the_stack_arguments() {
        let places: Vec<ArgPlace> = (0..ARGUMENT_REGISTERS)
            .map(|first| ArgPlace::Gpr { first, words: 1 })
            .collect();
        let record = windows_frame_record_bytes(&places);
        assert_eq!((record, windows_save_bytes(&places)), (16, 0));
        assert_eq!(windows(record, 0, &places).start, Some(16));
    }

    #[test]
    fn named_parameters_on_the_stack_leave_the_walk_undefined() {
        let places = [ArgPlace::Stack {
            offset: 0,
            bytes: 8,
        }];
        assert_eq!(windows(80, 8, &places).start, None);
    }

    #[test]
    fn a_named_float_leaves_every_general_register_to_va_arg() {
        let places = [ArgPlace::Fpr { index: 0 }];
        assert_eq!(windows_save_bytes(&places), 64);
        assert_eq!(
            windows(windows_frame_record_bytes(&places), 0, &places).first,
            0
        );
    }

    #[test]
    fn windows_spills_only_the_unnamed_registers() -> Result<(), LaminaError> {
        let places = one_named_word();
        let area = VaArea::windows(windows_frame_record_bytes(&places), 0, &places);
        let mut output = Vec::new();
        spill_save_area(&mut output, &area)?;
        let text = String::from_utf8_lossy(&output);
        assert!(text.starts_with("    add x16, x29, #24\n"), "{text}");
        assert_eq!(text.matches("    str x").count(), 7, "{text}");
        assert!(text.contains("    str x1, [x16, #0]\n"), "{text}");
        assert!(text.contains("    str x7, [x16, #48]\n"), "{text}");
        assert!(!text.contains("    str q"), "{text}");
        Ok(())
    }

    #[test]
    fn only_the_types_the_pointer_walk_can_place_are_accepted() {
        for ty in [
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::Ptr,
        ] {
            assert_eq!(va_arg_is_float(&MirType::Scalar(ty)).ok(), Some(false));
        }
        assert_eq!(
            va_arg_is_float(&MirType::Scalar(ScalarType::F64)).ok(),
            Some(true)
        );
        assert!(va_arg_is_float(&MirType::Scalar(ScalarType::F32)).is_err());
    }

    #[test]
    fn darwin_spills_nothing() -> Result<(), LaminaError> {
        let mut output = Vec::new();
        spill_save_area(&mut output, &VaArea::darwin(16))?;
        assert!(output.is_empty());
        Ok(())
    }
}
