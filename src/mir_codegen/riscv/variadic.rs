//! Variadic functions under the RISC-V psABI. Every unnamed argument occupies one
//! XLEN-wide slot, a float included, so a `va_list` is a plain pointer that steps one
//! slot at a time. The callee spills the argument registers no named parameter claimed
//! into a save area the prologue reserves directly below the incoming stack arguments,
//! which leaves the register-passed and stack-passed slots in one run.

use super::*;
use lamina_platform::VaListAbi;

/// Integer argument registers, which is also the number of slots the save area may hold.
const BANKS: usize = RiscVAbi::ARG_REGISTERS.len();

/// Where a variadic callee's unnamed arguments live, in bytes from `fp`.
///
/// `fp` is the caller's `sp` less [`VaArea::reserve`], so the save area runs from
/// `start` to `reserve` and the incoming stack arguments carry on from `reserve`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VaArea {
    reserve: i32,
    start: i32,
    first: usize,
    word: i32,
}

/// Integer argument registers the named parameters consume, the hidden result pointer
/// included, since `places` is built with it already counted.
fn named_banks(places: &[RiscVArgPlace], first_gpr: usize) -> usize {
    places
        .iter()
        .map(|place| match *place {
            RiscVArgPlace::Gpr(bank) => bank + 1,
            RiscVArgPlace::Words { first, words } => first + words,
            RiscVArgPlace::Split { reg, .. } => reg + 1,
            RiscVArgPlace::Flat { first, second } => [Some(first), second]
                .into_iter()
                .flatten()
                .filter(|slot| slot.field.float.is_none())
                .map(|slot| slot.bank + 1)
                .max()
                .unwrap_or(0),
            RiscVArgPlace::Fpr(_) | RiscVArgPlace::Stack(_) | RiscVArgPlace::StackWords { .. } => 0,
        })
        .fold(first_gpr, usize::max)
        .min(BANKS)
}

/// Words of incoming stack space the named parameters already took.
fn named_stack_words(places: &[RiscVArgPlace]) -> usize {
    places
        .iter()
        .map(|place| match *place {
            RiscVArgPlace::Stack(slot) => slot + 1,
            RiscVArgPlace::StackWords { slot, words } => slot + words,
            RiscVArgPlace::Split { slot, .. } => slot + 1,
            RiscVArgPlace::Gpr(_)
            | RiscVArgPlace::Fpr(_)
            | RiscVArgPlace::Words { .. }
            | RiscVArgPlace::Flat { .. } => 0,
        })
        .max()
        .unwrap_or(0)
}

impl VaArea {
    pub(super) fn new(places: &[RiscVArgPlace], first_gpr: usize, target: RiscVTarget) -> Self {
        let word = target.word_bytes();
        let first = named_banks(places, first_gpr);
        let save = (BANKS - first) as i32 * word;
        // The ABI keeps `sp` 16-byte aligned, so an odd number of saved registers is
        // padded underneath; the save area itself stays flush with the stack arguments.
        let reserve = align_up(save as usize, 16) as i32;
        Self {
            reserve,
            start: reserve - save + named_stack_words(places) as i32 * word,
            first,
            word,
        }
    }

    /// Bytes the prologue takes below the incoming stack arguments, before it saves the
    /// return address and frame pointer.
    pub(super) fn reserve(&self) -> i32 {
        self.reserve
    }
}

/// Writes the unnamed argument registers into the save area, on entry and before
/// anything else reads an argument register.
pub(super) fn spill<W: Write>(
    writer: &mut W,
    area: &VaArea,
    target: RiscVTarget,
) -> io::Result<()> {
    for bank in area.first..BANKS {
        let offset = area.start + (bank - area.first) as i32 * area.word;
        RiscVFrame::fp_store(
            writer,
            target.store_word(),
            RiscVAbi::ARG_REGISTERS[bank],
            offset,
            "t0",
        )?;
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(message.into()))
}

/// Whether `va_arg` of `ty` leaves in a float register. Anything a single XLEN slot
/// cannot hold is refused by name rather than read from the wrong place.
fn va_arg_is_float(ty: &MirType, target: RiscVTarget) -> Result<bool, LaminaError> {
    let xlen_bytes = target.word_bytes() as usize;
    let float = match ty {
        MirType::Scalar(ScalarType::F64) => true,
        MirType::Scalar(
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 | ScalarType::Ptr,
        ) => false,
        _ => {
            return Err(unsupported(format!(
                "va_arg of {ty:?} is not supported on {}; use an integer, pointer or f64",
                target.isa_name()
            )));
        }
    };
    if ty.size_bytes(target_pointer(target)) > xlen_bytes {
        return Err(unsupported(format!(
            "va_arg of {ty:?} spans two {} registers, which this backend does not place",
            target.isa_name()
        )));
    }
    Ok(float)
}

pub(super) fn emit<W: Write>(
    inst: &MirInst,
    writer: &mut W,
    reg_alloc: &RiscVRegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    area: Option<VaArea>,
    target: RiscVTarget,
) -> Result<(), LaminaError> {
    let Some(area) = area else {
        return Err(unsupported(
            "variadic arguments are only available inside a variadic function",
        ));
    };
    let load = target.load_word();
    let store = target.store_word();
    let word = area.word;
    match inst {
        MirInst::VaStart { ap } => {
            load_operand_to_register(ap, writer, reg_alloc, stack_slots, "t0")?;
            RiscVFrame::fp_address(writer, "t1", area.start)?;
            writeln!(writer, "    {store} t1, 0(t0)")?;
        }
        MirInst::VaArg { ty, dst, ap } => {
            let float = va_arg_is_float(ty, target)?;
            let Register::Virtual(vreg) = dst else {
                return Err(unsupported("va_arg must define a virtual register"));
            };
            load_operand_to_register(ap, writer, reg_alloc, stack_slots, "t0")?;
            writeln!(writer, "    {load} t1, 0(t0)")?;
            writeln!(writer, "    addi t2, t1, {word}")?;
            writeln!(writer, "    {store} t2, 0(t0)")?;
            writeln!(writer, "    {load} t1, 0(t1)")?;
            if float {
                writeln!(writer, "    fmv.d.x fa0, t1")?;
                store_fp_register_to_register("fa0", vreg, writer, reg_alloc, stack_slots, false)?;
            } else {
                extend_integer(writer, "t1", ty, target, true, target_pointer(target))?;
                store_register_to_register("t1", vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::VaEnd { .. } => {}
        MirInst::VaCopy { dst, src } => {
            load_operand_to_register(src, writer, reg_alloc, stack_slots, "t0")?;
            load_operand_to_register(dst, writer, reg_alloc, stack_slots, "t1")?;
            writeln!(writer, "    {load} t2, 0(t0)")?;
            writeln!(writer, "    {store} t2, 0(t1)")?;
        }
        _ => return Err(unsupported("not a variadic instruction")),
    }
    Ok(())
}

/// The `va_list` this backend writes is one pointer, which is what the target table
/// states for both RISC-V widths.
const _: () = {
    let rv64 = match TargetArchitecture::Riscv64.va_list() {
        VaListAbi::Defined { size, .. } => size,
        VaListAbi::Undefined => panic!("RISC-V lowers va_list but the target table defines none"),
    };
    let rv32 = match TargetArchitecture::Riscv32.va_list() {
        VaListAbi::Defined { size, .. } => size,
        VaListAbi::Undefined => panic!("RISC-V lowers va_list but the target table defines none"),
    };
    assert!(rv64 == 8 && rv32 == 4);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn rv64() -> RiscVTarget {
        RiscVTarget::general(Xlen::Rv64)
    }

    fn gprs(count: usize) -> Vec<RiscVArgPlace> {
        (0..count).map(RiscVArgPlace::Gpr).collect()
    }

    #[test]
    fn the_save_area_ends_where_the_incoming_stack_arguments_begin() {
        for named in 0..=BANKS {
            let places = gprs(named);
            let area = VaArea::new(&places, 0, rv64());
            let save = (BANKS - named) as i32 * 8;
            assert_eq!(area.reserve % 16, 0, "{named}");
            assert_eq!(area.start + save, area.reserve, "{named}");
            assert_eq!(area.first, named, "{named}");
        }
    }

    #[test]
    fn a_named_float_leaves_every_integer_register_to_va_arg() {
        let area = VaArea::new(&[RiscVArgPlace::Fpr(0)], 0, rv64());
        assert_eq!((area.first, area.start, area.reserve), (0, 0, 64));
    }

    #[test]
    fn an_aggregate_in_two_registers_counts_two_banks() {
        let area = VaArea::new(&[RiscVArgPlace::Words { first: 0, words: 2 }], 0, rv64());
        assert_eq!(area.first, 2);
        assert_eq!(area.start + 48, area.reserve);
    }

    #[test]
    fn a_hidden_result_pointer_takes_the_first_bank() {
        let area = VaArea::new(&[], 1, rv64());
        assert_eq!(area.first, 1);
        assert_eq!(area.start + 56, area.reserve);
    }

    #[test]
    fn named_stack_arguments_only_appear_once_the_registers_are_full() {
        let mut places = gprs(BANKS);
        places.push(RiscVArgPlace::Stack(0));
        let area = VaArea::new(&places, 0, rv64());
        assert_eq!((area.first, area.reserve), (BANKS, 0));
        assert_eq!(area.start, 8);
    }

    #[test]
    fn a_split_aggregate_fills_the_registers_and_the_first_stack_slot() {
        let mut places = gprs(BANKS - 1);
        places.push(RiscVArgPlace::Split {
            reg: BANKS - 1,
            slot: 0,
        });
        let area = VaArea::new(&places, 0, rv64());
        assert_eq!((area.first, area.reserve, area.start), (BANKS, 0, 8));
    }

    #[test]
    fn spill_writes_every_unnamed_register_into_the_save_area() -> Result<(), LaminaError> {
        let places = gprs(1);
        let area = VaArea::new(&places, 0, rv64());
        let mut out = Vec::new();
        spill(&mut out, &area, rv64())?;
        let text = String::from_utf8_lossy(&out).to_string();
        assert_eq!(text.matches("    sd a").count(), 7, "{text}");
        assert!(text.contains("    sd a1, 8(fp)\n"), "{text}");
        assert!(text.contains("    sd a7, 56(fp)\n"), "{text}");
        assert_eq!(area.reserve, 64);
        Ok(())
    }

    #[test]
    fn a_full_register_file_spills_nothing() -> Result<(), LaminaError> {
        let area = VaArea::new(&gprs(BANKS), 0, rv64());
        let mut out = Vec::new();
        spill(&mut out, &area, rv64())?;
        assert!(out.is_empty());
        Ok(())
    }

    #[test]
    fn only_the_types_one_slot_holds_are_accepted() {
        for ty in [
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::Ptr,
        ] {
            assert_eq!(
                va_arg_is_float(&MirType::Scalar(ty), rv64()).ok(),
                Some(false)
            );
        }
        assert_eq!(
            va_arg_is_float(&MirType::Scalar(ScalarType::F64), rv64()).ok(),
            Some(true)
        );
        assert!(va_arg_is_float(&MirType::Scalar(ScalarType::F32), rv64()).is_err());
        assert!(va_arg_is_float(&MirType::Scalar(ScalarType::I1), rv64()).is_err());
    }

    #[test]
    fn rv32_refuses_a_value_wider_than_its_register() {
        let rv32 = RiscVTarget::general(Xlen::Rv32);
        assert!(va_arg_is_float(&MirType::Scalar(ScalarType::F64), rv32).is_err());
        assert!(va_arg_is_float(&MirType::Scalar(ScalarType::I64), rv32).is_err());
        assert_eq!(
            va_arg_is_float(&MirType::Scalar(ScalarType::I32), rv32).ok(),
            Some(false)
        );
    }

    #[test]
    fn rv32_packs_eight_four_byte_slots() {
        let area = VaArea::new(&[], 0, RiscVTarget::general(Xlen::Rv32));
        assert_eq!((area.word, area.reserve, area.start), (4, 32, 0));
    }
}
