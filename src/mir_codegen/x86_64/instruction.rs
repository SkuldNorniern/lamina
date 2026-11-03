use super::variadic::VaArea;
use super::*;
use crate::mir_codegen::common::emit_source_loc;
use crate::mir_codegen::common::{align_up, unsupported_io};
use lamina_mir::{EightbyteClass, IntOverflowOp, SourceLoc, TlsModel};
use lamina_platform::PointerWidth;

/// Frame slots are laid out as the stack slots, then the callee-saved registers, then
/// the hidden sret pointer, then the variadic register save area, so the saved registers
/// sit just above the stack slots whatever follows them.
fn restore_saved_registers<W: Write>(
    writer: &mut W,
    stack_size: usize,
    saved_regs: &[&str],
    sret: bool,
    va_area: Option<&VaArea>,
) -> Result<(), Error> {
    let trailing = usize::from(sret) + va_area.map_or(0, VaArea::save_slots);
    let first_slot = stack_size / 8 - saved_regs.len() - trailing;
    for (index, reg) in saved_regs.iter().enumerate() {
        let offset = X86Frame::calculate_stack_offset(first_slot + index);
        writeln!(writer, "    movq {offset}(%rbp), %{reg}")?;
    }
    Ok(())
}

fn widen_external_argument<W: Write>(
    writer: &mut W,
    abi_extensions: &AbiExtensions,
    callee: &str,
    index: usize,
    register: &str,
) -> Result<(), LaminaError> {
    if let Some((width, signed)) = abi_extensions.external_argument(callee, index) {
        extend_integer(writer, register, width, signed)?;
    }
    Ok(())
}

/// Win64 passes a float in its xmm register and in the general register of the same
/// position, so a variadic callee that reads it as an integer finds it there. Run after
/// every general register is loaded, since the copies write argument registers.
fn duplicate_win64_float_arguments(
    writer: &mut impl Write,
    locations: &[ArgLocation],
    arg_regs: &[&str],
) -> Result<(), LaminaError> {
    for location in locations {
        if let ArgLocation::Xmm(xmm) = location
            && let Some(gpr) = arg_regs.get(*xmm)
        {
            writeln!(writer, "    movq %xmm{xmm}, %{gpr}")?;
        }
    }
    Ok(())
}

fn copy_exact_bytes_to_stack(
    writer: &mut impl Write,
    size: usize,
    src: &str,
    dst: usize,
) -> Result<(), LaminaError> {
    let mut at = 0;
    while at < size {
        let (mov, register, width) = match size - at {
            8.. => ("movq", "%r10", 8),
            4..=7 => ("movl", "%r10d", 4),
            2..=3 => ("movw", "%r10w", 2),
            _ => ("movb", "%r10b", 1),
        };
        writeln!(writer, "    {mov} {at}(%{src}), {register}")?;
        writeln!(writer, "    {mov} {register}, {}(%rsp)", dst + at)?;
        at += width;
    }
    Ok(())
}

fn copy_exact_bytes(
    writer: &mut impl Write,
    size: usize,
    src: &str,
    dst: &str,
) -> Result<(), LaminaError> {
    let mut at = 0;
    while at < size {
        let (mov, register, width) = match size - at {
            8.. => ("movq", "%rdx", 8),
            4..=7 => ("movl", "%edx", 4),
            2..=3 => ("movw", "%dx", 2),
            _ => ("movb", "%dl", 1),
        };
        writeln!(writer, "    {mov} {at}(%{src}), {register}")?;
        writeln!(writer, "    {mov} {register}, {at}(%{dst})")?;
        at += width;
    }
    Ok(())
}

enum StagedWord<'a> {
    Gpr(&'a str),
    Xmm(usize),
}

fn load_register_arguments<W: Write, RA: LocalRegisterAllocator<PhysReg = &'static str>>(
    writer: &mut RaxTrackingWriter<W>,
    args: &[Operand],
    types: &[MirType],
    locations: &[ArgLocation],
    arg_regs: &[&str],
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    for (index, location) in locations.iter().enumerate() {
        if let ArgLocation::Xmm(xmm) = location {
            let register = format!("xmm{xmm}");
            load_float_operand_to_xmm(
                &args[index],
                writer,
                reg_alloc,
                stack_slots,
                &register,
                &types[index],
            )?;
        }
    }
    for (index, location) in locations.iter().enumerate() {
        if let ArgLocation::Gpr(gpr) = location {
            load_operand_to_register(&args[index], writer, reg_alloc, stack_slots, arg_regs[*gpr])?;
        }
    }
    Ok(())
}

/// Loads one eight-byte word of an aggregate into rax. The argument operand holds the
/// address of the caller's copy.
fn load_aggregate_word<W: Write, RA: LocalRegisterAllocator<PhysReg = &'static str>>(
    arg: &Operand,
    word: usize,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
    writeln!(writer, "    movq {}(%rax), %rax", word * 8)?;
    writer.rax_value = None;
    Ok(())
}

/// Rewrite `register` to the canonical in-register form of `ty`, the form every narrow
/// value is held in between instructions.
/// Turns the raw result of a read or write syscall into the I/O contract: the count
/// stays, a failure becomes -2, and with `end_of_input` a zero count (a read that found
/// nothing) becomes -1. `carry` reads the failure from CF, as Mach-O does; otherwise
/// the result is signed, as Linux and the CRT report it.
fn emit_io_result(
    writer: &mut impl Write,
    carry: bool,
    end_of_input: bool,
) -> Result<(), LaminaError> {
    if carry {
        writeln!(writer, "    movq $-2, %rcx")?;
        writeln!(writer, "    cmovcq %rcx, %rax")?;
    } else {
        writeln!(writer, "    testq %rax, %rax")?;
        writeln!(writer, "    movq $-2, %rcx")?;
        writeln!(writer, "    cmovsq %rcx, %rax")?;
    }
    if end_of_input {
        writeln!(writer, "    testq %rax, %rax")?;
        writeln!(writer, "    movq $-1, %rcx")?;
        writeln!(writer, "    cmoveq %rcx, %rax")?;
    }
    Ok(())
}

pub(super) fn canonicalize_integer(
    writer: &mut impl Write,
    register: &str,
    ty: &MirType,
) -> Result<(), LaminaError> {
    if let Some((width, signed)) = narrow_canonical_extension(ty) {
        extend_integer(writer, register, width, signed)?;
    }
    Ok(())
}

/// Population count, leading and trailing zero counts and byte swap. The operand is
/// zero-extended to its width first, so the bits above the type never count.
fn signed_magic(divisor: i64) -> (u64, u32) {
    let ad = divisor.unsigned_abs() as u128;
    let two63: u128 = 1 << 63;
    let t = two63 + ((divisor as u64) >> 63) as u128;
    let anc = t - 1 - t % ad;
    let mut p: u32 = 63;
    let mut q1 = two63 / anc;
    let mut r1 = two63 - q1 * anc;
    let mut q2 = two63 / ad;
    let mut r2 = two63 - q2 * ad;
    loop {
        p += 1;
        q1 *= 2;
        r1 *= 2;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 *= 2;
        r2 *= 2;
        if r2 >= ad {
            q2 += 1;
            r2 -= ad;
        }
        let delta = ad - r2;
        if !(q1 < delta || (q1 == delta && r1 == 0)) {
            break;
        }
    }
    let mut magic = ((q2 + 1) as u64) as i64;
    if divisor < 0 {
        magic = magic.wrapping_neg();
    }
    (magic as u64, p - 64)
}

fn unsigned_magic(divisor: u64) -> (u64, u32) {
    let l = 64 - (divisor - 1).leading_zeros();
    let d = divisor as u128;
    let magic = ((1u128 << 64) * ((1u128 << l) - d)) / d + 1;
    (magic as u64, l)
}

const CONSTANT_DIVISOR_REDUCTION: bool = false;

fn const_divide_applies(op: IntBinOp, divisor: i64) -> bool {
    let signed = matches!(op, IntBinOp::SDiv | IntBinOp::SRem);
    CONSTANT_DIVISOR_REDUCTION && divisor != 0 && !(signed && divisor == -1)
}

fn emit_times_const<W: Write>(
    writer: &mut RaxTrackingWriter<W>,
    divisor: i64,
) -> Result<(), LaminaError> {
    if divisor > 0 {
        let shift = (divisor as u64).trailing_zeros();
        let odd = (divisor as u64) >> shift;
        match odd {
            1 => {}
            3 | 5 | 9 => writeln!(writer, "    leaq (%rax,%rax,{}), %rax", odd - 1)?,
            _ if i32::try_from(odd).is_ok() => writeln!(writer, "    imulq ${odd}, %rax, %rax")?,
            _ => writeln!(writer, "    movabsq ${odd}, %rdx\n    imulq %rdx, %rax")?,
        }
        if shift > 0 {
            writeln!(writer, "    shlq ${shift}, %rax")?;
        }
        return Ok(());
    }
    if i32::try_from(divisor).is_ok() {
        writeln!(writer, "    imulq ${divisor}, %rax, %rax")?;
    } else {
        writeln!(writer, "    movabsq ${divisor}, %rdx\n    imulq %rdx, %rax")?;
    }
    Ok(())
}

fn emit_const_divide<W: Write>(
    writer: &mut RaxTrackingWriter<W>,
    op: IntBinOp,
    divisor: i64,
) -> Result<bool, LaminaError> {
    let signed = matches!(op, IntBinOp::SDiv | IntBinOp::SRem);
    let remainder = matches!(op, IntBinOp::SRem | IntBinOp::URem);
    if !const_divide_applies(op, divisor) {
        return Ok(false);
    }
    let unsigned_magnitude = if signed {
        divisor.unsigned_abs()
    } else {
        divisor as u64
    };
    let power_of_two = unsigned_magnitude.is_power_of_two();
    let quotient_only_in_rax = divisor == 1 || (!signed && power_of_two && !remainder);
    if !quotient_only_in_rax || remainder {
        writeln!(writer, "    movq %rax, %rcx")?;
    }
    if signed {
        if divisor == 1 {
            // quotient is the dividend, already in rax
        } else if power_of_two {
            let k = unsigned_magnitude.trailing_zeros();
            writeln!(
                writer,
                "    movq %rcx, %rdx\n    sarq $63, %rdx\n    shrq ${}, %rdx\n    addq %rcx, %rdx\n    sarq ${k}, %rdx",
                64 - k
            )?;
            if divisor < 0 {
                writeln!(writer, "    negq %rdx")?;
            }
            writeln!(writer, "    movq %rdx, %rax")?;
        } else {
            let (magic, shift) = signed_magic(divisor);
            writeln!(writer, "    movabsq ${}, %rax", magic as i64)?;
            writeln!(writer, "    imulq %rcx")?;
            if divisor > 0 && (magic as i64) < 0 {
                writeln!(writer, "    addq %rcx, %rdx")?;
            }
            if divisor < 0 && (magic as i64) > 0 {
                writeln!(writer, "    subq %rcx, %rdx")?;
            }
            if shift > 0 {
                writeln!(writer, "    sarq ${shift}, %rdx")?;
            }
            writeln!(
                writer,
                "    movq %rdx, %rax\n    shrq $63, %rax\n    addq %rdx, %rax"
            )?;
        }
    } else if divisor == 1 {
        // quotient is the dividend, already in rax
    } else if power_of_two {
        let k = unsigned_magnitude.trailing_zeros();
        writeln!(writer, "    shrq ${k}, %rax")?;
    } else {
        let (magic, log) = unsigned_magic(unsigned_magnitude);
        writeln!(
            writer,
            "    movabsq ${}, %rax\n    mulq %rcx\n    movq %rcx, %rax\n    subq %rdx, %rax\n    shrq $1, %rax\n    addq %rdx, %rax\n    shrq ${}, %rax",
            magic as i64,
            log - 1
        )?;
    }
    if remainder {
        if divisor == 1 {
            writeln!(writer, "    xorl %eax, %eax")?;
        } else if !signed && power_of_two {
            writeln!(
                writer,
                "    movabsq ${}, %rdx\n    andq %rdx, %rcx\n    movq %rcx, %rax",
                unsigned_magnitude - 1
            )?;
        } else {
            emit_times_const(writer, divisor)?;
            writeln!(writer, "    subq %rax, %rcx\n    movq %rcx, %rax")?;
        }
    }
    Ok(true)
}

fn emit_int_divide<W: Write>(
    writer: &mut RaxTrackingWriter<W>,
    op: IntBinOp,
    ty: &MirType,
    lhs: &Operand,
    rhs: &Operand,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    if ty.size_bytes(pointer_width) == 8
        && let Some(divisor) = rhs.integer_constant()
        && emit_const_divide(writer, op, divisor)?
    {
        return Ok(());
    }
    if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
        extend_integer(writer, "rax", ty.size_bytes(pointer_width), false)?;
        extend_integer(writer, "rcx", ty.size_bytes(pointer_width), false)?;
    }
    if matches!(op, IntBinOp::SDiv | IntBinOp::SRem) && ty.size_bytes(pointer_width) < 8 {
        extend_integer(writer, "rax", ty.size_bytes(pointer_width), true)?;
        extend_integer(writer, "rcx", ty.size_bytes(pointer_width), true)?;
        let (_, overflow) = op.division_checks(
            *ty,
            lhs.integer_constant(),
            rhs.integer_constant(),
            pointer_width,
        );
        if overflow {
            let minimum = ty.signed_integer(
                1i64 << (ty.size_bytes(pointer_width) * 8 - 1),
                pointer_width,
            );
            writeln!(
                writer,
                "    cmpq $-1, %rcx\n    jne 2f\n    cmpq ${minimum}, %rax\n    jne 2f\n    ud2\n2:"
            )?;
        }
    }
    let signed = matches!(op, IntBinOp::SDiv | IntBinOp::SRem);
    let remainder = matches!(op, IntBinOp::SRem | IntBinOp::URem);
    let narrow = ty.size_bytes(pointer_width) == 8 && rhs.integer_constant().is_none();
    if narrow && signed {
        writeln!(
            writer,
            "    movslq %eax, %rdx\n    cmpq %rdx, %rax\n    jne 3f\n    movslq %ecx, %rdx\n    cmpq %rdx, %rcx\n    jne 3f\n    cmpq $-1, %rcx\n    je 3f\n    cltd\n    idivl %ecx"
        )?;
        if remainder {
            writeln!(writer, "    movslq %edx, %rax")?;
        } else {
            writeln!(writer, "    cltq")?;
        }
        writeln!(writer, "    jmp 4f\n3:")?;
    } else if narrow {
        writeln!(
            writer,
            "    movq %rax, %rdx\n    orq %rcx, %rdx\n    shrq $32, %rdx\n    jne 3f\n    xorl %edx, %edx\n    divl %ecx"
        )?;
        if remainder {
            writeln!(writer, "    movl %edx, %eax")?;
        }
        writeln!(writer, "    jmp 4f\n3:")?;
    }
    match op {
        IntBinOp::SDiv => {
            writeln!(writer, "    cqto")?;
            writeln!(writer, "    idivq %rcx")?;
        }
        IntBinOp::UDiv => {
            writeln!(writer, "    xorq %rdx, %rdx")?;
            writeln!(writer, "    divq %rcx")?;
        }
        IntBinOp::SRem => {
            writeln!(writer, "    cqto")?;
            writeln!(writer, "    idivq %rcx")?;
            writeln!(writer, "    movq %rdx, %rax")?;
        }
        IntBinOp::URem => {
            writeln!(writer, "    xorq %rdx, %rdx")?;
            writeln!(writer, "    divq %rcx")?;
            writeln!(writer, "    movq %rdx, %rax")?;
        }
        _ => unreachable!(),
    }
    if narrow {
        writeln!(writer, "4:")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_int_unary<W: Write>(
    op: IntUnOp,
    ty: &MirType,
    dst: &Register,
    src: &Operand,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    cpu_level: X86CpuLevel,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let bytes = ty.size_bytes(pointer_width);
    let bits = bytes * 8;
    load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
    extend_integer(writer, "rax", bytes, false)?;
    match op {
        IntUnOp::Popcount if cpu_level >= X86CpuLevel::V2 => {
            writeln!(writer, "    popcntq %rax, %rax")?;
        }
        IntUnOp::Popcount => emit_popcount_swar(writer)?,
        IntUnOp::Ctlz if cpu_level >= X86CpuLevel::V3 => {
            writeln!(writer, "    lzcntq %rax, %rax")?;
            if bits < 64 {
                writeln!(writer, "    subq ${}, %rax", 64 - bits)?;
            }
        }
        IntUnOp::Cttz if cpu_level >= X86CpuLevel::V3 => {
            if bits < 64 {
                writeln!(writer, "    btsq ${bits}, %rax")?;
            }
            writeln!(writer, "    tzcntq %rax, %rax")?;
        }
        IntUnOp::Ctlz => {
            writeln!(writer, "    movq $-1, %rcx")?;
            writeln!(writer, "    bsrq %rax, %rax")?;
            writeln!(writer, "    cmovzq %rcx, %rax")?;
            writeln!(writer, "    negq %rax")?;
            writeln!(writer, "    addq ${}, %rax", bits - 1)?;
        }
        IntUnOp::Cttz => {
            writeln!(writer, "    movq ${bits}, %rcx")?;
            writeln!(writer, "    bsfq %rax, %rax")?;
            writeln!(writer, "    cmovzq %rcx, %rax")?;
        }
        IntUnOp::Bswap => match bytes {
            1 => {}
            2 => writeln!(writer, "    rolw $8, %ax")?,
            4 => writeln!(writer, "    bswapl %eax")?,
            _ => writeln!(writer, "    bswapq %rax")?,
        },
    }
    canonicalize_integer(writer, "rax", ty)?;
    store_integer_result(dst, writer, reg_alloc, stack_slots)
}

/// Population count of the zero-extended value in `%rax` without POPCNT (x86-64-v1).
/// The classic SWAR reduction. `%rcx` and `%rdx` may hold live values, so they are
/// saved on the stack around the sequence rather than treated as scratch.
fn emit_popcount_swar<W: Write>(writer: &mut RaxTrackingWriter<W>) -> Result<(), LaminaError> {
    writeln!(writer, "    pushq %rcx")?;
    writeln!(writer, "    pushq %rdx")?;
    writeln!(writer, "    movq %rax, %rcx")?;
    writeln!(writer, "    shrq $1, %rcx")?;
    writeln!(writer, "    movabsq $0x5555555555555555, %rdx")?;
    writeln!(writer, "    andq %rdx, %rcx")?;
    writeln!(writer, "    subq %rcx, %rax")?;
    writeln!(writer, "    movabsq $0x3333333333333333, %rdx")?;
    writeln!(writer, "    movq %rax, %rcx")?;
    writeln!(writer, "    andq %rdx, %rax")?;
    writeln!(writer, "    shrq $2, %rcx")?;
    writeln!(writer, "    andq %rdx, %rcx")?;
    writeln!(writer, "    addq %rcx, %rax")?;
    writeln!(writer, "    movq %rax, %rcx")?;
    writeln!(writer, "    shrq $4, %rcx")?;
    writeln!(writer, "    addq %rcx, %rax")?;
    writeln!(writer, "    movabsq $0x0f0f0f0f0f0f0f0f, %rdx")?;
    writeln!(writer, "    andq %rdx, %rax")?;
    writeln!(writer, "    movabsq $0x0101010101010101, %rdx")?;
    writeln!(writer, "    imulq %rdx, %rax")?;
    writeln!(writer, "    shrq $56, %rax")?;
    writeln!(writer, "    popq %rdx")?;
    writeln!(writer, "    popq %rcx")?;
    Ok(())
}

/// Rotation of `lhs` by `rhs` modulo the bit width. Bits of `rdx` above the width are
/// discarded by the final canonicalization, so only the low width bits must be right.
#[allow(clippy::too_many_arguments)]
fn emit_int_rotate<W: Write>(
    op: IntRotOp,
    ty: &MirType,
    dst: &Register,
    (lhs, rhs): (&Operand, &Operand),
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let bytes = ty.size_bytes(pointer_width);
    let bits = bytes * 8;
    load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
    extend_integer(writer, "rax", bytes, false)?;
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
    writeln!(writer, "    andl ${}, %ecx", bits - 1)?;
    writeln!(writer, "    movq %rax, %rdx")?;
    match op {
        IntRotOp::Rotl => {
            writeln!(writer, "    shlq %cl, %rdx")?;
            writeln!(writer, "    negl %ecx")?;
            writeln!(writer, "    addl ${bits}, %ecx")?;
            writeln!(writer, "    andl $63, %ecx")?;
            writeln!(writer, "    shrq %cl, %rax")?;
            writeln!(writer, "    orq %rdx, %rax")?;
        }
        IntRotOp::Rotr => {
            writeln!(writer, "    shrq %cl, %rax")?;
            writeln!(writer, "    negl %ecx")?;
            writeln!(writer, "    addl ${bits}, %ecx")?;
            writeln!(writer, "    andl $63, %ecx")?;
            writeln!(writer, "    shlq %cl, %rdx")?;
            writeln!(writer, "    orq %rdx, %rax")?;
        }
    }
    canonicalize_integer(writer, "rax", ty)?;
    store_integer_result(dst, writer, reg_alloc, stack_slots)
}

fn store_integer_result<W: Write>(
    dst: &Register,
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    match dst {
        Register::Virtual(vreg) => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
        Register::Physical(physical) if physical.name != "rax" => {
            writeln!(writer, "    movq %rax, %{}", physical.name)?;
        }
        Register::Physical(_) => {}
    }
    Ok(())
}

fn fixed_bits(value: f64, is_f32: bool) -> i64 {
    if is_f32 {
        i64::from((value as f32).to_bits())
    } else {
        value.to_bits() as i64
    }
}

/// Emit `IntToFloat`. Integers of 32 bits or fewer convert through the 64-bit
/// signed form after extension. Unsigned 64-bit inputs at or above 2^63 are
/// halved with the lost bit kept as a sticky bit, converted, and doubled. Both
/// candidates are computed and the sign of the source selects one with a mask,
/// so no branch is needed.
#[allow(clippy::too_many_arguments)]
fn emit_int_to_float<W: Write, RA: LocalRegisterAllocator<PhysReg = &'static str>>(
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let is_f32 = is_f32(to);
    let (cvt, add, mov_out) = if is_f32 {
        ("cvtsi2ssq", "addss", "movd")
    } else {
        ("cvtsi2sdq", "addsd", "movq")
    };
    let width = from.size_bytes(pointer_width);
    load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
    if signed || width < 8 {
        extend_integer(writer, "rax", width, signed)?;
        writeln!(writer, "    {cvt} %rax, %xmm0")?;
        writeln!(writer, "    {mov_out} %xmm0, %rax")?;
    } else {
        writeln!(writer, "    movq %rax, %rdx")?;
        writeln!(writer, "    shrq $1, %rdx")?;
        writeln!(writer, "    movq %rax, %rcx")?;
        writeln!(writer, "    andl $1, %ecx")?;
        writeln!(writer, "    orq %rcx, %rdx")?;
        writeln!(writer, "    {cvt} %rdx, %xmm0")?;
        writeln!(writer, "    {add} %xmm0, %xmm0")?;
        writeln!(writer, "    movq %rax, %rcx")?;
        writeln!(writer, "    sarq $63, %rcx")?;
        writeln!(writer, "    {cvt} %rax, %xmm1")?;
        writeln!(writer, "    {mov_out} %xmm0, %rdx")?;
        writeln!(writer, "    {mov_out} %xmm1, %rax")?;
        writeln!(writer, "    andq %rcx, %rdx")?;
        writeln!(writer, "    notq %rcx")?;
        writeln!(writer, "    andq %rcx, %rax")?;
        writeln!(writer, "    orq %rdx, %rax")?;
    }
    match dst {
        Register::Virtual(vreg) => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
        Register::Physical(physical) => writeln!(writer, "    movq %rax, %{}", physical.name)?,
    }
    Ok(())
}

/// Emit `FloatToInt`: truncate toward zero, saturate to the target range, NaN
/// yields 0. The hardware instruction returns the "integer indefinite" value
/// for NaN and out-of-range inputs, so the result is overridden by comparisons
/// against constants that are exact in the source format.
#[allow(clippy::too_many_arguments)]
fn emit_float_to_int<W: Write, RA: LocalRegisterAllocator<PhysReg = &'static str>>(
    writer: &mut RaxTrackingWriter<W>,
    reg_alloc: &RA,
    stack_slots: &HashMap<VirtualReg, i32>,
    from: &MirType,
    to: &MirType,
    signed: bool,
    dst: &Register,
    src: &Operand,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let source_f32 = is_f32(from);
    let (cvt, ucomi, mov_const) = if source_f32 {
        ("cvttss2siq", "ucomiss", "movd")
    } else {
        ("cvttsd2siq", "ucomisd", "movq")
    };
    let bits = to.size_bytes(pointer_width) * 8;
    let constant = |value: f64| fixed_bits(value, source_f32);
    load_float_operand_to_xmm(src, writer, reg_alloc, stack_slots, "xmm0", from)?;
    writeln!(writer, "    {cvt} %xmm0, %rax")?;
    if signed {
        let (max, min) = if bits >= 64 {
            (i64::MAX, i64::MIN)
        } else {
            ((1i64 << (bits - 1)) - 1, -(1i64 << (bits - 1)))
        };
        let two63 = 2f64.powi(63);
        writeln!(writer, "    movq ${}, %rcx", max)?;
        writeln!(writer, "    movq ${}, %rdx", constant(two63))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovae %rcx, %rax")?;
        writeln!(writer, "    movq ${}, %rcx", min)?;
        writeln!(writer, "    movq ${}, %rdx", constant(-two63))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovb %rcx, %rax")?;
        if bits < 64 {
            writeln!(writer, "    movq ${}, %rcx", max)?;
            writeln!(writer, "    cmpq %rcx, %rax")?;
            writeln!(writer, "    cmovg %rcx, %rax")?;
            writeln!(writer, "    movq ${}, %rcx", min)?;
            writeln!(writer, "    cmpq %rcx, %rax")?;
            writeln!(writer, "    cmovl %rcx, %rax")?;
        }
        writeln!(writer, "    xorl %ecx, %ecx")?;
        writeln!(writer, "    {ucomi} %xmm0, %xmm0")?;
        writeln!(writer, "    cmovp %rcx, %rax")?;
    } else if bits < 64 {
        let max = (1i128 << bits) - 1;
        writeln!(writer, "    movq ${}, %rcx", max)?;
        writeln!(
            writer,
            "    movq ${}, %rdx",
            constant(2f64.powi(bits as i32))
        )?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovae %rcx, %rax")?;
        writeln!(writer, "    xorl %ecx, %ecx")?;
        writeln!(writer, "    movq ${}, %rdx", constant(-1.0))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovbe %rcx, %rax")?;
    } else {
        // Values in [2^63, 2^64) are truncated after subtracting 2^63, then the
        // top bit is restored.
        writeln!(writer, "    movq ${}, %rdx", constant(2f64.powi(63)))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    movapd %xmm0, %xmm2")?;
        writeln!(
            writer,
            "    {}",
            if source_f32 {
                "subss %xmm1, %xmm2"
            } else {
                "subsd %xmm1, %xmm2"
            }
        )?;
        writeln!(writer, "    {cvt} %xmm2, %rcx")?;
        writeln!(writer, "    movq $-9223372036854775808, %rdx")?;
        writeln!(writer, "    xorq %rdx, %rcx")?;
        writeln!(writer, "    movq ${}, %rdx", constant(2f64.powi(63)))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovae %rcx, %rax")?;
        writeln!(writer, "    movq ${}, %rdx", constant(2f64.powi(64)))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    movq $-1, %rcx")?;
        writeln!(writer, "    cmovae %rcx, %rax")?;
        writeln!(writer, "    xorl %ecx, %ecx")?;
        writeln!(writer, "    movq ${}, %rdx", constant(-1.0))?;
        writeln!(writer, "    {mov_const} %rdx, %xmm1")?;
        writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
        writeln!(writer, "    cmovbe %rcx, %rax")?;
    }
    if bits < 64 {
        canonicalize_integer(writer, "rax", to)?;
    }
    match dst {
        Register::Virtual(vreg) => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
        Register::Physical(physical) => writeln!(writer, "    movq %rax, %{}", physical.name)?,
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Every caller frees the scratch it takes, so an empty pool means one lowering path
/// holds more at once than the pool has. Falling back to a callee-saved register would
/// hide that behind a register nothing saves.
/// Renders `[base + index*scale + offset]` as an x86 memory operand. Operands that live in
/// stack slots are loaded into scratch registers; the caller hands the returned list to
/// `release_indexed_scratch` once the access has been emitted.
pub(super) fn indexed_memory<W: Write>(
    base: &Register,
    index: &Register,
    scale: u8,
    offset: i8,
    writer: &mut W,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(String, Vec<&'static str>), LaminaError> {
    let mut scratches = Vec::new();
    let mut gprs = [base, index].map(|reg| mapped_register(reg, reg_alloc));
    for (slot, reg) in gprs.iter_mut().zip([base, index]) {
        if slot.is_some() {
            continue;
        }
        let Register::Virtual(vreg) = reg else {
            return Err(scratch_error());
        };
        let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
        scratches.push(scratch);
        load_register_to_register(vreg, writer, reg_alloc, stack_slots, scratch)?;
        *slot = Some(scratch);
    }
    let [Some(b), Some(i)] = gprs else {
        return Err(scratch_error());
    };
    Ok((format!("{offset}(%{b},%{i},{scale})"), scratches))
}

pub(super) fn release_indexed_scratch(scratches: Vec<&'static str>, reg_alloc: &mut X64RegAlloc) {
    for scratch in scratches {
        if scratch != "rbx" {
            reg_alloc.free_scratch(scratch);
        }
    }
}

pub(super) fn scratch_error() -> LaminaError {
    LaminaError::CodegenError(CodegenError::UnsupportedFeature(
        "x86_64 scratch registers exhausted".to_string(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_instruction_x86_64(
    inst: &MirInst,
    writer: &mut RaxTrackingWriter<impl Write>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    stack_size: usize,
    has_frame: bool,
    saved_regs: &[&str],
    target_os: TargetOperatingSystem,
    func_name: &str,
    settings: &MirCodegenSettings,
    loc: Option<SourceLoc>,
    abi_extensions: &AbiExtensions,
    next_label: &str,
    self_param_homes: &[Option<String>],
    param_extensions: &[Option<(usize, bool)>],
    entry_label: &str,
    sret_slot: Option<i32>,
    va_area: Option<&super::variadic::VaArea>,
    pointer_width: PointerWidth,
) -> Result<(), LaminaError> {
    let unwind = os_uses_elf(target_os);
    // An elided fall-through jump writes no bytes, so a row for it would land on the
    // next block's first instruction.
    let elided = matches!(inst, MirInst::Jmp { target } if target == next_label);
    if !elided {
        emit_source_loc(writer, settings, loc)?;
    }
    if let MirInst::IntConvert {
        from,
        to,
        signed,
        dst,
        src,
    } = inst
    {
        let dst_phys = match dst {
            Register::Virtual(v) => reg_alloc.get_mapping_for(v),
            Register::Physical(p) => Some(p.name),
        };
        if let Some(target) = dst_phys {
            load_operand_to_register(src, writer, reg_alloc, stack_slots, target)?;
            extend_integer(
                writer,
                target,
                from.size_bytes(pointer_width)
                    .min(to.size_bytes(pointer_width)),
                *signed,
            )?;
            canonicalize_integer(writer, target, to)?;
            return Ok(());
        }
        load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
        extend_integer(
            writer,
            "rax",
            from.size_bytes(pointer_width)
                .min(to.size_bytes(pointer_width)),
            *signed,
        )?;
        canonicalize_integer(writer, "rax", to)?;
        match dst {
            Register::Virtual(vreg) => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
            Register::Physical(physical) => writeln!(writer, "    movq %rax, %{}", physical.name)?,
        }
        return Ok(());
    }
    match inst {
        MirInst::Copy { ty, dst, src }
        | MirInst::Bitcast {
            to: ty, dst, src, ..
        } => {
            if matches!(ty, MirType::Vector(_)) {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support vector Copy of type {ty}"
                )));
            }
            let src_phys = match src {
                Operand::Register(Register::Virtual(v)) => reg_alloc.get_mapping_for(v),
                Operand::Register(Register::Physical(p)) => Some(p.name),
                Operand::Immediate(_) => None,
            };
            let dst_phys = match dst {
                Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                Register::Physical(p) => Some(p.name),
            };
            let in_place =
                ty.size_bytes(pointer_width) == 8 && src_phys.is_some() && src_phys == dst_phys;
            if !in_place {
                match (dst, dst_phys) {
                    (_, Some(target)) => {
                        load_copy_operand_into(
                            ty,
                            src,
                            target,
                            writer,
                            reg_alloc,
                            stack_slots,
                            pointer_width,
                        )?;
                    }
                    (Register::Virtual(vreg), None) => {
                        let slot = stack_slots.get(vreg).copied();
                        let direct = match (ty.size_bytes(pointer_width), slot) {
                            (8, Some(offset)) => match src {
                                Operand::Register(Register::Virtual(v)) => {
                                    reg_alloc.get_mapping_for(v).map(|p| format!("%{p}"))
                                }
                                Operand::Register(Register::Physical(p)) => {
                                    Some(format!("%{}", p.name))
                                }
                                Operand::Immediate(_) => integer_immediate(src)
                                    .filter(|v| i32::try_from(*v).is_ok())
                                    .map(|v| format!("${v}")),
                            }
                            .map(|source| (source, offset)),
                            _ => None,
                        };
                        match direct {
                            Some((source, offset)) => {
                                writeln!(writer, "    movq {source}, {offset}(%rbp)")?;
                            }
                            None => {
                                load_copy_operand_to_rax(
                                    ty,
                                    src,
                                    writer,
                                    reg_alloc,
                                    stack_slots,
                                    pointer_width,
                                )?;
                                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                            }
                        }
                    }
                    (Register::Physical(_), None) => {}
                }
            }
        }
        MirInst::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            // Get dst physical register if available
            let dst_phys = match dst {
                Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                Register::Physical(p) => Some(p.name),
            };

            if let Some(dp) = dst_phys {
                // Optimized path: emit directly into dp, avoiding rax bounce
                let rhs_phys = match rhs {
                    Operand::Register(Register::Virtual(v)) => reg_alloc.get_mapping_for(v),
                    Operand::Register(Register::Physical(p)) => Some(p.name),
                    Operand::Immediate(_) => None,
                };
                let rhs_in_dp = rhs_phys == Some(dp);
                if rhs_in_dp {
                    writeln!(writer, "    movq %{dp}, %rax")?;
                }
                match op {
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr => {
                        // Load lhs into dp
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, dp)?;
                        if matches!(op, IntBinOp::AShr) {
                            extend_integer(writer, dp, ty.size_bytes(pointer_width), true)?;
                        }
                        if matches!(op, IntBinOp::AShr) {
                            extend_integer(writer, "rax", ty.size_bytes(pointer_width), true)?;
                        }
                        if matches!(op, IntBinOp::LShr) {
                            extend_integer(writer, dp, ty.size_bytes(pointer_width), false)?;
                        }
                        match rhs {
                            Operand::Immediate(imm) => {
                                let sv: u64 = match imm {
                                    Immediate::I8(v) => *v as u64,
                                    Immediate::I16(v) => *v as u64,
                                    Immediate::I32(v) => *v as u64,
                                    Immediate::I64(v) => *v as u64,
                                    _ => {
                                        return Err(LaminaError::ValidationError(
                                            "Shift count must be integer".to_string(),
                                        ));
                                    }
                                };
                                let sv = sv & (ty.size_bytes(pointer_width) as u64 * 8 - 1);
                                match op {
                                    IntBinOp::Shl => writeln!(writer, "    shlq ${sv}, %{dp}")?,
                                    IntBinOp::AShr => writeln!(writer, "    sarq ${sv}, %{dp}")?,
                                    IntBinOp::LShr => writeln!(writer, "    shrq ${sv}, %{dp}")?,
                                    _ => unreachable!(),
                                }
                            }
                            Operand::Register(_) if settings.x86_cpu_level() >= X86CpuLevel::V3 => {
                                let count = if rhs_in_dp {
                                    "rax"
                                } else if let (8, Some(r)) =
                                    (ty.size_bytes(pointer_width), rhs_phys)
                                {
                                    r
                                } else {
                                    load_operand_to_register(
                                        rhs,
                                        writer,
                                        reg_alloc,
                                        stack_slots,
                                        "rcx",
                                    )?;
                                    "rcx"
                                };
                                if ty.size_bytes(pointer_width) < 8 {
                                    writeln!(
                                        writer,
                                        "    andq ${}, %{count}",
                                        ty.size_bytes(pointer_width) * 8 - 1
                                    )?;
                                }
                                match op {
                                    IntBinOp::Shl => {
                                        writeln!(writer, "    shlxq %{count}, %{dp}, %{dp}")?
                                    }
                                    IntBinOp::AShr => {
                                        writeln!(writer, "    sarxq %{count}, %{dp}, %{dp}")?
                                    }
                                    IntBinOp::LShr => {
                                        writeln!(writer, "    shrxq %{count}, %{dp}, %{dp}")?
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            Operand::Register(_) => {
                                // Need shift count in %cl; load rhs into rcx
                                if rhs_in_dp {
                                    writeln!(writer, "    movq %rax, %rcx")?;
                                } else {
                                    load_operand_to_register(
                                        rhs,
                                        writer,
                                        reg_alloc,
                                        stack_slots,
                                        "rcx",
                                    )?;
                                }
                                if ty.size_bytes(pointer_width) < 8 {
                                    writeln!(
                                        writer,
                                        "    andl ${}, %ecx",
                                        ty.size_bytes(pointer_width) * 8 - 1
                                    )?;
                                }
                                match op {
                                    IntBinOp::Shl => writeln!(writer, "    shlq %cl, %{dp}")?,
                                    IntBinOp::AShr => writeln!(writer, "    sarq %cl, %{dp}")?,
                                    IntBinOp::LShr => writeln!(writer, "    shrq %cl, %{dp}")?,
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                    IntBinOp::SDiv | IntBinOp::UDiv | IntBinOp::SRem | IntBinOp::URem => {
                        // Division always goes through rax/rdx; use rcx for divisor to avoid conflicts
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, "rax")?;
                        if !(ty.size_bytes(pointer_width) == 8
                            && rhs
                                .integer_constant()
                                .is_some_and(|d| const_divide_applies(*op, d)))
                        {
                            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        }
                        emit_int_divide(writer, *op, ty, lhs, rhs, pointer_width)?;
                        // Store result (rax) into dp
                        if dp != "rax" {
                            writeln!(writer, "    movq %rax, %{dp}")?;
                        }
                    }
                    _ => {
                        // Arithmetic/bitwise: load lhs into dp, then apply op with rhs
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, dp)?;

                        // Try to use rhs directly (register or small immediate)
                        match rhs {
                            Operand::Immediate(imm) => {
                                let v: i64 = match imm {
                                    Immediate::I8(x) => *x as i64,
                                    Immediate::I16(x) => *x as i64,
                                    Immediate::I32(x) => *x as i64,
                                    Immediate::I64(x) => *x,
                                    _ => {
                                        return Err(LaminaError::ValidationError(
                                            "Float immediate in integer op".to_string(),
                                        ));
                                    }
                                };
                                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                                    // Fits in 32-bit sign-extended immediate
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq ${v}, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq ${v}, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq ${v}, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  ${v}, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq ${v}, %{dp}")?,
                                        IntBinOp::Mul => {
                                            writeln!(writer, "    imulq ${v}, %{dp}, %{dp}")?
                                        }
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // Large immediate: load into rax as temp, then op
                                    writeln!(writer, "    movq ${v}, %rax")?;
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %rax, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %rax, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq %rax, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %rax, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %rax, %{dp}")?,
                                        IntBinOp::Mul => writeln!(writer, "    imulq %rax, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                }
                            }
                            Operand::Register(_) => {
                                // Get rhs physical register if available for direct register op
                                let rhs_src = if rhs_in_dp { Some("rax") } else { rhs_phys };
                                if let Some(rp) = rhs_src {
                                    // Both in registers: direct op, no memory access
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %{rp}, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %{rp}, %{dp}")?,
                                        IntBinOp::Mul => {
                                            writeln!(writer, "    imulq %{rp}, %{dp}")?
                                        }
                                        IntBinOp::And => writeln!(writer, "    andq %{rp}, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %{rp}, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %{rp}, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // rhs in stack: load to rax as temp, then op
                                    load_operand_to_register(
                                        rhs,
                                        writer,
                                        reg_alloc,
                                        stack_slots,
                                        "rax",
                                    )?;
                                    match op {
                                        IntBinOp::Add => writeln!(writer, "    addq %rax, %{dp}")?,
                                        IntBinOp::Sub => writeln!(writer, "    subq %rax, %{dp}")?,
                                        IntBinOp::Mul => writeln!(writer, "    imulq %rax, %{dp}")?,
                                        IntBinOp::And => writeln!(writer, "    andq %rax, %{dp}")?,
                                        IntBinOp::Or => writeln!(writer, "    orq  %rax, %{dp}")?,
                                        IntBinOp::Xor => writeln!(writer, "    xorq %rax, %{dp}")?,
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        }
                        // No store needed — result already in dp!
                    }
                }
                canonicalize_integer(writer, dp, ty)?;
            } else {
                // No physical register for dst — fall back to rax accumulator path
                load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
                if matches!(op, IntBinOp::LShr) {
                    extend_integer(writer, "rax", ty.size_bytes(pointer_width), false)?;
                }

                match op {
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr
                        if settings.x86_cpu_level() >= X86CpuLevel::V3
                            && matches!(rhs, Operand::Register(_)) =>
                    {
                        let rhs_phys = match rhs {
                            Operand::Register(Register::Virtual(v)) => reg_alloc.get_mapping_for(v),
                            Operand::Register(Register::Physical(p)) => Some(p.name),
                            Operand::Immediate(_) => None,
                        };
                        let scratch = match (ty.size_bytes(pointer_width), rhs_phys) {
                            (8, Some(_)) => None,
                            _ => {
                                let scratch =
                                    reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
                                load_operand_to_register(
                                    rhs,
                                    writer,
                                    reg_alloc,
                                    stack_slots,
                                    scratch,
                                )?;
                                Some(scratch)
                            }
                        };
                        let count = scratch.or(rhs_phys).unwrap_or("rcx");
                        if ty.size_bytes(pointer_width) < 8 {
                            writeln!(
                                writer,
                                "    andq ${}, %{count}",
                                ty.size_bytes(pointer_width) * 8 - 1
                            )?;
                        }
                        match op {
                            IntBinOp::Shl => writeln!(writer, "    shlxq %{count}, %rax, %rax")?,
                            IntBinOp::AShr => writeln!(writer, "    sarxq %{count}, %rax, %rax")?,
                            IntBinOp::LShr => writeln!(writer, "    shrxq %{count}, %rax, %rax")?,
                            _ => unreachable!(),
                        }
                        if let Some(scratch) = scratch
                            && scratch != "rbx"
                        {
                            reg_alloc.free_scratch(scratch);
                        }
                    }
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr => match rhs {
                        Operand::Immediate(imm) => {
                            let shift_val = match imm {
                                Immediate::I8(v) => *v as u64,
                                Immediate::I16(v) => *v as u64,
                                Immediate::I32(v) => *v as u64,
                                Immediate::I64(v) => *v as u64,
                                _ => {
                                    return Err(LaminaError::ValidationError(
                                        "Shift count must be an integer immediate".to_string(),
                                    ));
                                }
                            };
                            let shift_val =
                                shift_val & (ty.size_bytes(pointer_width) as u64 * 8 - 1);
                            match op {
                                IntBinOp::Shl => writeln!(writer, "    shlq ${shift_val}, %rax")?,
                                IntBinOp::AShr => writeln!(writer, "    sarq ${shift_val}, %rax")?,
                                IntBinOp::LShr => writeln!(writer, "    shrq ${shift_val}, %rax")?,
                                _ => unreachable!(),
                            }
                        }
                        Operand::Register(_) => {
                            let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
                            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
                            writeln!(writer, "    movq %{scratch}, %rcx")?;
                            if ty.size_bytes(pointer_width) < 8 {
                                writeln!(
                                    writer,
                                    "    andl ${}, %ecx",
                                    ty.size_bytes(pointer_width) * 8 - 1
                                )?;
                            }
                            match op {
                                IntBinOp::Shl => writeln!(writer, "    shlq %cl, %rax")?,
                                IntBinOp::AShr => writeln!(writer, "    sarq %cl, %rax")?,
                                IntBinOp::LShr => writeln!(writer, "    shrq %cl, %rax")?,
                                _ => unreachable!(),
                            }
                            if scratch != "rbx" {
                                reg_alloc.free_scratch(scratch);
                            }
                        }
                    },
                    IntBinOp::SDiv | IntBinOp::UDiv | IntBinOp::SRem | IntBinOp::URem => {
                        if !(ty.size_bytes(pointer_width) == 8
                            && rhs
                                .integer_constant()
                                .is_some_and(|d| const_divide_applies(*op, d)))
                        {
                            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        }
                        emit_int_divide(writer, *op, ty, lhs, rhs, pointer_width)?;
                    }
                    _ => {
                        let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;

                        match op {
                            IntBinOp::Add => writeln!(writer, "    addq %{scratch}, %rax")?,
                            IntBinOp::Sub => writeln!(writer, "    subq %{scratch}, %rax")?,
                            IntBinOp::Mul => writeln!(writer, "    imulq %{scratch}, %rax")?,
                            IntBinOp::And => writeln!(writer, "    andq %{scratch}, %rax")?,
                            IntBinOp::Or => writeln!(writer, "    orq  %{scratch}, %rax")?,
                            IntBinOp::Xor => writeln!(writer, "    xorq %{scratch}, %rax")?,
                            _ => unreachable!(),
                        }

                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    }
                }

                canonicalize_integer(writer, "rax", ty)?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let scratch = emit_int_compare(
                op,
                ty,
                lhs,
                rhs,
                writer,
                reg_alloc,
                stack_slots,
                pointer_width,
            )?;
            let suffix = cmp_suffix(op)?;
            writeln!(writer, "    set{suffix} %al")?;
            writeln!(writer, "    movzbq %al, %rax")?;

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }

            if !scratch.is_empty() && scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
        }
        MirInst::Call { name, args, ret } => {
            if name == "lamina.writebyte" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writebyte requires one argument".to_string())
                })?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movb %al, (%rsp)")?;

                match target_os {
                    os if os_uses_macho(os) => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, true, false)?;
                    }
                    TargetOperatingSystem::Windows => {
                        writeln!(writer, "    movq %rsp, %rdx")?;
                        writeln!(writer, "    movq $1, %r8")?;
                        writeln!(writer, "    movq ${}, %rcx", fd::STDOUT)?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call _write")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    movslq %eax, %rax")?;
                        emit_io_result(writer, false, false)?;
                    }
                    TargetOperatingSystem::Linux => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, false, false)?;
                    }
                    TargetOperatingSystem::FreeBSD => {
                        writeln!(writer, "    movq ${}, %rax", freebsd::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, true, false)?;
                    }
                    other => {
                        return Err(unsupported_io(name, other));
                    }
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;
            } else if name == "lamina.readbyte" && args.is_empty() {
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;

                match target_os {
                    os if os_uses_macho(os) => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                        writeln!(writer, "    movq $-2, %rdx")?;
                        writeln!(writer, "    cmovcq %rdx, %rax")?;
                    }
                    TargetOperatingSystem::Windows => {
                        writeln!(writer, "    movq %rsp, %rdx")?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    xorl %ecx, %ecx")?;
                        writeln!(writer, "    movl $1, %r8d")?;
                        writeln!(writer, "    call _read")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    movslq %eax, %rax")?;
                    }
                    TargetOperatingSystem::FreeBSD => {
                        writeln!(writer, "    movq ${}, %rax", freebsd::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                        writeln!(writer, "    movq $-2, %rdx")?;
                        writeln!(writer, "    cmovcq %rdx, %rax")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                // Result: the byte (0..=255) after one read, -1 at end of file,
                // -2 on any failure. Keyed on the count, never on the byte, so a
                // NUL byte is not confused with EOF.
                writeln!(writer, "    movzbl (%rsp), %ecx")?;
                writeln!(writer, "    movq $-1, %rdx")?;
                writeln!(writer, "    testq %rax, %rax")?;
                writeln!(writer, "    cmoveq %rdx, %rcx")?;
                writeln!(writer, "    movq $-2, %rdx")?;
                writeln!(writer, "    cmpq $1, %rax")?;
                writeln!(writer, "    cmovaq %rdx, %rcx")?;
                writeln!(writer, "    movq %rcx, %rax")?;

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else if (name == "lamina.write" || name == "lamina.read") && args.len() == 2 {
                let writing = name == "lamina.write";
                load_operand_to_rax(&args[0], writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movq %rax, %r10")?;
                load_operand_to_rax(&args[1], writer, reg_alloc, stack_slots)?;
                if !writing {
                    writeln!(writer, "    testq %rax, %rax")?;
                    writeln!(writer, "    je 1f")?;
                }
                match target_os {
                    TargetOperatingSystem::Windows => {
                        // Windows syscall numbers are unstable, so the CRT does the work.
                        writeln!(writer, "    movq %rax, %r8")?;
                        writeln!(writer, "    movq %r10, %rdx")?;
                        writeln!(
                            writer,
                            "    movq ${}, %rcx",
                            if writing { fd::STDOUT } else { fd::STDIN }
                        )?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(
                            writer,
                            "    call {}",
                            if writing { "_write" } else { "_read" }
                        )?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    movslq %eax, %rax")?;
                        emit_io_result(writer, false, !writing)?;
                    }
                    os if os_uses_macho(os) => {
                        writeln!(writer, "    movq %rax, %rdx")?;
                        writeln!(writer, "    movq %r10, %rsi")?;
                        writeln!(
                            writer,
                            "    movq ${}, %rdi",
                            if writing { fd::STDOUT } else { fd::STDIN }
                        )?;
                        let syscall = if writing {
                            macos::SYS_WRITE
                        } else {
                            macos::SYS_READ
                        };
                        writeln!(writer, "    movq ${syscall}, %rax")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, true, !writing)?;
                    }
                    TargetOperatingSystem::Linux => {
                        writeln!(writer, "    movq %rax, %rdx")?;
                        writeln!(writer, "    movq %r10, %rsi")?;
                        writeln!(
                            writer,
                            "    movq ${}, %rdi",
                            if writing { fd::STDOUT } else { fd::STDIN }
                        )?;
                        let syscall = if writing {
                            linux::SYS_WRITE
                        } else {
                            linux::SYS_READ
                        };
                        writeln!(writer, "    movq ${syscall}, %rax")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, false, !writing)?;
                    }
                    TargetOperatingSystem::FreeBSD => {
                        writeln!(writer, "    movq %rax, %rdx")?;
                        writeln!(writer, "    movq %r10, %rsi")?;
                        writeln!(
                            writer,
                            "    movq ${}, %rdi",
                            if writing { fd::STDOUT } else { fd::STDIN }
                        )?;
                        let syscall = if writing {
                            freebsd::SYS_WRITE
                        } else {
                            freebsd::SYS_READ
                        };
                        writeln!(writer, "    movq ${syscall}, %rax")?;
                        writeln!(writer, "    syscall")?;
                        emit_io_result(writer, true, !writing)?;
                    }
                    other => {
                        return Err(unsupported_io(name, other));
                    }
                }
                if !writing {
                    writeln!(writer, "1:")?;
                }
                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let descs = abi_extensions.argument_descs(name, args);
                let types: Vec<MirType> = descs.iter().map(|desc| desc.ty).collect();
                // A callee returning a large aggregate writes it through a hidden
                // pointer, which takes the first general register.
                let sret_result = abi_extensions
                    .return_aggregate(name)
                    .filter(|aggregate| returns_through_pointer(*aggregate, target_os))
                    .and_then(|_| match ret {
                        Some(Register::Virtual(vreg)) => stack_slots.get(vreg).copied(),
                        _ => None,
                    });
                let locations =
                    classify_argument_descs_with_sret(&descs, target_os, sret_result.is_some());
                // One entry per eight-byte slot, lowest slot first: an aggregate on the
                // stack occupies as many as it has words.
                // One entry per stack slot, lowest first. `None` is a padding slot that the
                // classifier left between a 16-byte-aligned aggregate and the one before.
                let mut stack_args: Vec<Option<(usize, usize)>> = Vec::new();
                for (index, location) in locations.iter().enumerate() {
                    match location {
                        ArgLocation::Stack(_) => stack_args.push(Some((index, 0))),
                        ArgLocation::StackWords { first, words } => {
                            while stack_args.len() < *first {
                                stack_args.push(None);
                            }
                            stack_args.extend((0..*words).map(|word| Some((index, word))));
                        }
                        ArgLocation::Gpr(_)
                        | ArgLocation::Xmm(_)
                        | ArgLocation::GprWords { .. }
                        | ArgLocation::EightbyteWords { .. } => {}
                    }
                }
                let num_stack_args = stack_args.len();

                // Windows takes an aggregate that is not register sized as a pointer to a
                // copy belonging to the caller, which goes above the outgoing arguments.
                let mut copy_offsets: Vec<Option<usize>> = vec![None; descs.len()];
                let mut copy_bytes = 0;
                if target_os == TargetOperatingSystem::Windows {
                    let base = align_up(
                        windows::SHADOW_SPACE_SIZE as usize + (num_stack_args * stack::SLOT_SIZE),
                        stack::ALIGNMENT,
                    );
                    for (index, desc) in descs.iter().enumerate() {
                        if let Some(aggregate) = desc.aggregate
                            && passed_by_reference(aggregate, target_os)
                        {
                            copy_offsets[index] = Some(base + copy_bytes);
                            copy_bytes +=
                                align_up(aggregate.words() * stack::SLOT_SIZE, stack::ALIGNMENT);
                        }
                    }
                }

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    // Rounded up so rsp stays 16-aligned at the call; the
                    // padding lands above the arguments, leaving their offsets
                    // from rsp unchanged.
                    let total_stack = align_up(
                        shadow_space + (num_stack_args * stack::SLOT_SIZE),
                        stack::ALIGNMENT,
                    ) + copy_bytes;
                    if total_stack > 0 {
                        writeln!(writer, "    subq ${total_stack}, %rsp")?;
                    }
                    for (index, at) in copy_offsets.iter().enumerate() {
                        let Some(at) = *at else {
                            continue;
                        };
                        let size = descs[index].aggregate.map_or(0, |a| a.size);
                        load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                        copy_exact_bytes_to_stack(writer, size, "rax", at)?;
                        writer.rax_value = None;
                    }
                    for (slot, entry) in stack_args.iter().enumerate() {
                        let stack_offset = shadow_space + (slot * stack::SLOT_SIZE);
                        // Windows never pads: its aggregates that need a stack slot are not
                        // 16-byte aligned by value.
                        let Some((index, word)) = *entry else {
                            return Err(LaminaError::ValidationError(
                                "Windows stack arguments cannot need alignment padding".to_string(),
                            ));
                        };
                        if let Some(at) = copy_offsets[index] {
                            writeln!(writer, "    leaq {at}(%rsp), %rax")?;
                            writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                            continue;
                        }
                        if descs[index].aggregate.is_some() {
                            load_aggregate_word(
                                &args[index],
                                word,
                                writer,
                                reg_alloc,
                                stack_slots,
                            )?;
                        } else {
                            load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                            widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                        }
                        writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                    }
                } else {
                    // An odd number of pushes would leave rsp 8 off at the call.
                    if num_stack_args % 2 == 1 {
                        writeln!(writer, "    subq $8, %rsp")?;
                    }
                    for entry in stack_args.iter().rev() {
                        let Some((index, word)) = *entry else {
                            writeln!(writer, "    pushq $0")?;
                            continue;
                        };
                        if descs[index].aggregate.is_some() {
                            load_aggregate_word(
                                &args[index],
                                word,
                                writer,
                                reg_alloc,
                                stack_slots,
                            )?;
                        } else {
                            load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                            widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                        }
                        writeln!(writer, "    pushq %rax")?;
                    }
                }

                // A by-reference aggregate's register holds the address of the copy, so
                // the scalar pass must leave it alone.
                let mut scalar_locations = locations.clone();
                for (index, at) in copy_offsets.iter().enumerate() {
                    if at.is_some() {
                        scalar_locations[index] = ArgLocation::Stack(usize::MAX);
                    }
                }

                // Aggregate words are staged on the stack so that loading the scalar
                // arguments cannot clobber an argument register already filled.
                let mut staged: Vec<StagedWord> = Vec::new();
                for (index, location) in locations.iter().enumerate() {
                    if copy_offsets[index].is_some() {
                        continue;
                    }
                    if let ArgLocation::EightbyteWords { places, words } = location {
                        for (word, place) in places[..*words].iter().enumerate() {
                            load_aggregate_word(
                                &args[index],
                                word,
                                writer,
                                reg_alloc,
                                stack_slots,
                            )?;
                            writeln!(writer, "    pushq %rax")?;
                            staged.push(match place {
                                WordPlace::Gpr(gpr) => StagedWord::Gpr(arg_regs[*gpr]),
                                WordPlace::Xmm(xmm) => StagedWord::Xmm(*xmm),
                            });
                        }
                        continue;
                    }
                    let (first, words) = match location {
                        ArgLocation::GprWords { first, words } => (*first, *words),
                        ArgLocation::Gpr(gpr) if descs[index].aggregate.is_some() => (*gpr, 1),
                        _ => continue,
                    };
                    for word in 0..words {
                        load_aggregate_word(&args[index], word, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    pushq %rax")?;
                        staged.push(StagedWord::Gpr(arg_regs[first + word]));
                    }
                }

                load_register_arguments(
                    writer,
                    args,
                    &types,
                    &scalar_locations,
                    arg_regs,
                    reg_alloc,
                    stack_slots,
                )?;

                for word in staged.iter().rev() {
                    match word {
                        StagedWord::Gpr(register) => writeln!(writer, "    popq %{register}")?,
                        StagedWord::Xmm(xmm) => {
                            writeln!(writer, "    popq %r11")?;
                            writeln!(writer, "    movq %r11, %xmm{xmm}")?;
                        }
                    }
                }
                for (index, location) in locations.iter().enumerate() {
                    if let ArgLocation::Gpr(gpr) = location
                        && descs[index].aggregate.is_none()
                    {
                        widen_external_argument(
                            writer,
                            abi_extensions,
                            name,
                            index,
                            arg_regs[*gpr],
                        )?;
                    }
                }
                if target_os == TargetOperatingSystem::Windows {
                    duplicate_win64_float_arguments(writer, &locations, arg_regs)?;
                }

                for (index, at) in copy_offsets.iter().enumerate() {
                    if let (Some(at), ArgLocation::Gpr(gpr)) = (*at, &locations[index]) {
                        writeln!(writer, "    leaq {at}(%rsp), %{}", arg_regs[*gpr])?;
                    }
                }
                if let Some(offset) = sret_result {
                    writeln!(writer, "    leaq {offset}(%rbp), %{}", arg_regs[0])?;
                }
                if target_os != TargetOperatingSystem::Windows
                    && abi_extensions.may_be_variadic(name, args.len())
                {
                    let xmm_count: usize = locations.iter().map(ArgLocation::xmm_words).sum();
                    writeln!(writer, "    movl ${xmm_count}, %eax")?;
                }
                let mangled_name = abi.mangle_function_name(name);
                writeln!(writer, "    call {mangled_name}")?;

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    let total_stack = align_up(
                        shadow_space + (num_stack_args * stack::SLOT_SIZE),
                        stack::ALIGNMENT,
                    ) + copy_bytes;
                    if total_stack > 0 {
                        writeln!(writer, "    addq ${total_stack}, %rsp")?;
                    }
                } else if num_stack_args > 0 {
                    let pushed = num_stack_args * stack::SLOT_SIZE;
                    let padded = align_up(pushed, stack::ALIGNMENT);
                    writeln!(writer, "    addq ${padded}, %rsp")?;
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    if sret_result.is_some() {
                        // The callee wrote the words into the slot already.
                        writer.rax_value = None;
                    } else if let Some(aggregate) = abi_extensions.return_aggregate(name) {
                        // The result register names the slot the words are written to;
                        // the caller addresses it with a stack address of its own.
                        let base = *stack_slots.get(vreg).ok_or_else(|| {
                            LaminaError::ValidationError(
                                "aggregate result has no frame slot".to_string(),
                            )
                        })?;
                        let mut integer_returns = ["rax", "rdx"].into_iter();
                        let mut sse_returns = ["xmm0", "xmm1"].into_iter();
                        for word in 0..aggregate.words() {
                            let source = match aggregate.eightbyte_class(word) {
                                EightbyteClass::Integer => integer_returns.next(),
                                EightbyteClass::Sse => sse_returns.next(),
                            }
                            .ok_or_else(|| {
                                LaminaError::ValidationError(
                                    "aggregate result needs more return registers than the ABI has"
                                        .to_string(),
                                )
                            })?;
                            writeln!(
                                writer,
                                "    movq %{source}, {}(%rbp)",
                                base + (word as i32) * 8
                            )?;
                        }
                        writer.rax_value = None;
                    } else {
                        if let Some(ty) = abi_extensions.float_return(name) {
                            writeln!(writer, "{}", move_xmm_to_rax(&ty, 0))?;
                        } else if let Some((width, signed)) = abi_extensions.return_extension(name)
                        {
                            extend_integer(writer, "rax", width, signed)?;
                        }
                        store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                    }
                }
            }
        }
        MirInst::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            simd::operation(
                *op,
                *ty,
                dst,
                operands,
                writer,
                reg_alloc,
                stack_slots,
                settings.x86_cpu_level(),
            )?;
        }
        MirInst::Load { ty, .. } | MirInst::Store { ty, .. } if ty.is_vector() => {
            simd::memory(inst, writer, reg_alloc, stack_slots)?;
        }
        MirInst::Load {
            dst,
            addr,
            ty,
            attrs: _,
        } => {
            if let AddressMode::BaseOffset {
                base, offset: 0, ..
            } = addr
            {
                let base_phys = mapped_register(base, reg_alloc);
                let dst_phys = match dst {
                    Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                    Register::Physical(p) => Some(p.name),
                };
                match (base, base_phys, dst_phys) {
                    (_, Some(bp), Some(dp)) => {
                        writeln!(
                            writer,
                            "    {} (%{bp}), %{dp}",
                            load_mnemonic_for(ty, pointer_width)
                        )?;
                    }
                    (Register::Virtual(bv), None, Some(dp)) if stack_slots.contains_key(bv) => {
                        load_register_to_register(bv, writer, reg_alloc, stack_slots, dp)?;
                        writeln!(
                            writer,
                            "    {} (%{dp}), %{dp}",
                            load_mnemonic_for(ty, pointer_width)
                        )?;
                    }
                    _ => {
                        match base {
                            Register::Virtual(vreg) => {
                                load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                            }
                            Register::Physical(phys) => {
                                writeln!(writer, "    movq %{}, %rax", phys.name)?;
                            }
                        }
                        writeln!(writer, "    {}", load_insn_for(ty, pointer_width))?;
                        if let Register::Virtual(vreg) = dst {
                            store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                        }
                    }
                }
            } else if let AddressMode::BaseIndexScale {
                base,
                index,
                scale,
                offset,
                ..
            } = addr
            {
                let (mem, scratches) =
                    indexed_memory(base, index, *scale, *offset, writer, reg_alloc, stack_slots)?;
                let dst_phys = match dst {
                    Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                    Register::Physical(p) => Some(p.name),
                };
                match dst_phys {
                    Some(dp) => writeln!(
                        writer,
                        "    {} {mem}, %{dp}",
                        load_mnemonic_for(ty, pointer_width)
                    )?,
                    None => {
                        writeln!(
                            writer,
                            "    {} {mem}, %rax",
                            load_mnemonic_for(ty, pointer_width)
                        )?;
                        if let Register::Virtual(vreg) = dst {
                            store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                        }
                    }
                }
                release_indexed_scratch(scratches, reg_alloc);
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support load address mode {addr:?}"
                )));
            }
        }
        MirInst::Store {
            addr,
            src,
            ty,
            attrs: _,
        } => {
            let src_phys = match src {
                Operand::Register(Register::Virtual(v)) => reg_alloc.get_mapping_for(v),
                Operand::Register(Register::Physical(p)) => Some(p.name),
                Operand::Immediate(_) => None,
            };
            let size = ty.size_bytes(pointer_width);
            let imm_value = integer_immediate(src).map(|raw| {
                let shift = 64 - size * 8;
                (raw << shift) >> shift
            });
            if let (
                Some(sp),
                AddressMode::BaseOffset {
                    base, offset: 0, ..
                },
            ) = (src_phys, addr)
                && let Some(bp) = mapped_register(base, reg_alloc)
            {
                let alias = register_for_size(sp, size).ok_or_else(scratch_error)?;
                writeln!(
                    writer,
                    "    mov{} %{alias}, (%{bp})",
                    size_suffix(ty, pointer_width)
                )?;
            } else if let (
                Some(value),
                AddressMode::BaseOffset {
                    base, offset: 0, ..
                },
            ) = (imm_value.filter(|v| i32::try_from(*v).is_ok()), addr)
                && let Some(bp) = mapped_register(base, reg_alloc)
            {
                writeln!(
                    writer,
                    "    mov{} ${value}, (%{bp})",
                    size_suffix(ty, pointer_width)
                )?;
            } else if let (
                Some(sp),
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                    ..
                },
            ) = (src_phys, addr)
            {
                let alias = register_for_size(sp, size).ok_or_else(scratch_error)?;
                let (mem, scratches) =
                    indexed_memory(base, index, *scale, *offset, writer, reg_alloc, stack_slots)?;
                writeln!(
                    writer,
                    "    mov{} %{alias}, {mem}",
                    size_suffix(ty, pointer_width)
                )?;
                release_indexed_scratch(scratches, reg_alloc);
            } else {
                load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
                if let AddressMode::BaseOffset {
                    base, offset: 0, ..
                } = addr
                {
                    let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
                    match base {
                        Register::Virtual(vreg) => {
                            load_register_to_register(
                                vreg,
                                writer,
                                reg_alloc,
                                stack_slots,
                                scratch,
                            )?;
                        }
                        Register::Physical(phys) => {
                            writeln!(writer, "    movq %{}, %{}", phys.name, scratch)?;
                        }
                    }
                    writeln!(
                        writer,
                        "    mov{} %{}, (%{scratch})",
                        size_suffix(ty, pointer_width),
                        rax_for(ty, pointer_width)
                    )?;
                    if scratch != "rbx" {
                        reg_alloc.free_scratch(scratch);
                    }
                } else if let AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                    ..
                } = addr
                {
                    let (mem, scratches) = indexed_memory(
                        base,
                        index,
                        *scale,
                        *offset,
                        writer,
                        reg_alloc,
                        stack_slots,
                    )?;
                    writeln!(
                        writer,
                        "    mov{} %{}, {mem}",
                        size_suffix(ty, pointer_width),
                        rax_for(ty, pointer_width)
                    )?;
                    release_indexed_scratch(scratches, reg_alloc);
                } else {
                    return Err(LaminaError::ValidationError(format!(
                        "x86_64 backend does not support store address mode {addr:?}"
                    )));
                }
            }
        }
        MirInst::AtomicLoad { .. }
        | MirInst::AtomicStore { .. }
        | MirInst::AtomicBinary { .. }
        | MirInst::AtomicCompareExchange { .. }
        | MirInst::Fence { .. } => {
            super::atomic::emit(inst, writer, reg_alloc, stack_slots, pointer_width)?;
        }
        MirInst::InlineAsm { .. } => {
            super::inline_asm::emit(
                inst,
                writer,
                reg_alloc,
                stack_slots,
                target_os,
                pointer_width,
            )?;
        }
        MirInst::VaStart { .. }
        | MirInst::VaArg { .. }
        | MirInst::VaEnd { .. }
        | MirInst::VaCopy { .. } => {
            super::variadic::emit(inst, writer, reg_alloc, stack_slots, va_area, func_name)?;
        }
        MirInst::TailCall { name, args } if name == func_name => {
            // Self tail call: move the arguments straight into the parameter homes as a
            // parallel copy and jump to the entry block, skipping the ABI registers, the
            // prologue and the epilogue entirely.
            let direct = args.len() == self_param_homes.len()
                && self_param_homes.iter().all(|home| home.is_some())
                && args.iter().all(|arg| match arg {
                    Operand::Register(Register::Virtual(vreg)) => {
                        reg_alloc.get_mapping_for(vreg).is_some() || stack_slots.contains_key(vreg)
                    }
                    Operand::Register(Register::Physical(_)) => true,
                    Operand::Immediate(Immediate::I8(_) | Immediate::I16(_))
                    | Operand::Immediate(Immediate::I32(_) | Immediate::I64(_)) => true,
                    _ => false,
                });
            if direct {
                let mut pairs = Vec::with_capacity(args.len());
                for (arg, home) in args.iter().zip(self_param_homes) {
                    let source = match arg {
                        Operand::Register(Register::Virtual(vreg)) => reg_alloc
                            .get_mapping_for(vreg)
                            .map(|reg| format!("%{reg}"))
                            .unwrap_or_else(|| format!("{}(%rbp)", stack_slots[vreg])),
                        Operand::Register(Register::Physical(reg)) => format!("%{reg}"),
                        Operand::Immediate(imm) => {
                            let value = match imm {
                                Immediate::I8(v) => i64::from(*v),
                                Immediate::I16(v) => i64::from(*v),
                                Immediate::I32(v) => i64::from(*v),
                                Immediate::I64(v) => *v,
                                other => {
                                    return Err(LaminaError::ValidationError(format!(
                                        "self tail call argument {other:?} is not an integer"
                                    )));
                                }
                            };
                            format!("${value}")
                        }
                    };
                    pairs.push((source, home.clone().unwrap_or_default()));
                }
                emit_parallel_copy(&pairs, writer)?;
                emit_narrow_param_normalization(writer, param_extensions, self_param_homes)?;
                if entry_label != next_label {
                    writeln!(writer, "    jmp .L_{func_name}.{entry_label}")?;
                }
                return Ok(());
            }
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let types = abi_extensions.argument_types(func_name, args);
            let locations = classify_arguments(&types, target_os);
            load_register_arguments(
                writer,
                args,
                &types,
                &locations,
                arg_regs,
                reg_alloc,
                stack_slots,
            )?;
            writeln!(writer, "    jmp {}", self_tail_entry_label(func_name))?;
        }
        MirInst::TailCall { name, args } => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let types = abi_extensions.argument_types(name, args);
            let locations = classify_arguments(&types, target_os);
            let own_stack_args = stack_argument_count(&classify_arguments(
                &abi_extensions.parameter_types(func_name),
                target_os,
            ));

            // The incoming argument area is the caller's, so a jump can only reuse it
            // when the callee's stack arguments fit there. Otherwise keep the frame and
            // return whatever the call returns.
            if stack_argument_count(&locations) > own_stack_args {
                emit_instruction_x86_64(
                    &MirInst::Call {
                        name: name.clone(),
                        args: args.clone(),
                        ret: None,
                    },
                    writer,
                    reg_alloc,
                    stack_slots,
                    stack_size,
                    has_frame,
                    saved_regs,
                    target_os,
                    func_name,
                    settings,
                    None,
                    abi_extensions,
                    next_label,
                    self_param_homes,
                    param_extensions,
                    entry_label,
                    sret_slot,
                    va_area,
                    pointer_width,
                )?;
                restore_saved_registers(
                    writer,
                    stack_size,
                    saved_regs,
                    sret_slot.is_some(),
                    va_area,
                )?;
                if has_frame {
                    X86Frame::generate_epilogue(writer, stack_size, unwind)?;
                } else {
                    writeln!(writer, "    ret")?;
                }
                return Ok(());
            }

            // Stack arguments go into the incoming slots before any register is
            // overwritten, since their sources may live in argument registers.
            let shadow_space = if target_os == TargetOperatingSystem::Windows {
                windows::SHADOW_SPACE_SIZE as usize
            } else {
                0
            };
            for (index, location) in locations.iter().enumerate() {
                if let ArgLocation::Stack(slot) = location {
                    // Incoming args are at RBP + 16 + shadow + 8*slot: saved RBP and
                    // the return address sit below them, and on Windows the caller's
                    // shadow space precedes the stack arguments.
                    let offset = 16 + shadow_space + (slot * stack::SLOT_SIZE);
                    load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                    widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                    writeln!(writer, "    movq %rax, {offset}(%rbp)")?;
                }
            }
            load_register_arguments(
                writer,
                args,
                &types,
                &locations,
                arg_regs,
                reg_alloc,
                stack_slots,
            )?;
            for (index, location) in locations.iter().enumerate() {
                if let ArgLocation::Gpr(gpr) = location {
                    widen_external_argument(writer, abi_extensions, name, index, arg_regs[*gpr])?;
                }
            }
            if target_os == TargetOperatingSystem::Windows {
                duplicate_win64_float_arguments(writer, &locations, arg_regs)?;
            }

            restore_saved_registers(writer, stack_size, saved_regs, sret_slot.is_some(), va_area)?;
            if unwind {
                writeln!(writer, "    .cfi_remember_state")?;
            }
            writeln!(writer, "    movq %rbp, %rsp")?;
            writeln!(writer, "    popq %rbp")?;
            if unwind {
                writeln!(writer, "    .cfi_def_cfa %rsp, 8")?;
            }
            let mangled_name = abi.mangle_function_name(name);
            writeln!(writer, "    jmp {mangled_name}")?;
            if unwind {
                writeln!(writer, "    .cfi_restore_state")?;
            }
        }
        MirInst::SymbolAddress { dst, symbol } => {
            let label = X86ABI::new(target_os).mangle_function_name(symbol);
            let (load, operand) = if abi_extensions.uses_got(symbol) {
                ("movq", format!("{label}@GOTPCREL(%rip)"))
            } else {
                ("leaq", format!("{label}(%rip)"))
            };
            if let Some(dp) = mapped_register(dst, reg_alloc) {
                writeln!(writer, "    {load} {operand}, %{dp}")?;
                return Ok(());
            }
            writeln!(writer, "    {load} {operand}, %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::ThreadLocalAddr { dst, symbol, model } => {
            let label = X86ABI::new(target_os).mangle_function_name(symbol);
            if os_uses_coff(target_os) {
                writeln!(writer, "    movq %gs:0x58, %rax")?;
                writeln!(writer, "    movl _tls_index(%rip), %ecx")?;
                writeln!(writer, "    movq (%rax,%rcx,8), %rax")?;
                writeln!(writer, "    leaq {label}@SECREL32(%rax), %rax")?;
            } else {
                match model {
                    TlsModel::LocalExec => {
                        writeln!(writer, "    movq %fs:0, %rax")?;
                        writeln!(writer, "    leaq {label}@tpoff(%rax), %rax")?;
                    }
                    TlsModel::InitialExec => {
                        writeln!(writer, "    movq {label}@gottpoff(%rip), %rax")?;
                        writeln!(writer, "    addq %fs:0, %rax")?;
                    }
                }
            }
            if let Some(dp) = mapped_register(dst, reg_alloc) {
                writeln!(writer, "    movq %rax, %{dp}")?;
                return Ok(());
            }
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::IntOverflow {
            op,
            ty,
            signed,
            dst,
            lhs,
            rhs,
        } => {
            let size = ty.size_bytes(pointer_width);
            let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
            load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
            if size < 8 {
                // Below 64 bits the exact result fits in a register, so the question is
                // whether re-reading it at the operand width gives the same number.
                extend_integer(writer, "rax", size, *signed)?;
                extend_integer(writer, scratch, size, *signed)?;
                match op {
                    IntOverflowOp::Add => writeln!(writer, "    addq %{scratch}, %rax")?,
                    IntOverflowOp::Sub => writeln!(writer, "    subq %{scratch}, %rax")?,
                    IntOverflowOp::Mul => writeln!(writer, "    imulq %{scratch}, %rax")?,
                }
                writeln!(writer, "    movq %rax, %{scratch}")?;
                extend_integer(writer, scratch, size, *signed)?;
                writeln!(writer, "    cmpq %{scratch}, %rax")?;
                writeln!(writer, "    setne %al")?;
            } else {
                match (op, signed) {
                    (IntOverflowOp::Add, true) => {
                        writeln!(writer, "    addq %{scratch}, %rax")?;
                        writeln!(writer, "    seto %al")?;
                    }
                    (IntOverflowOp::Add, false) => {
                        writeln!(writer, "    addq %{scratch}, %rax")?;
                        writeln!(writer, "    setc %al")?;
                    }
                    (IntOverflowOp::Sub, true) => {
                        writeln!(writer, "    subq %{scratch}, %rax")?;
                        writeln!(writer, "    seto %al")?;
                    }
                    (IntOverflowOp::Sub, false) => {
                        writeln!(writer, "    subq %{scratch}, %rax")?;
                        writeln!(writer, "    setc %al")?;
                    }
                    (IntOverflowOp::Mul, true) => {
                        writeln!(writer, "    imulq %{scratch}, %rax")?;
                        writeln!(writer, "    seto %al")?;
                    }
                    (IntOverflowOp::Mul, false) => {
                        // The unsigned product needs the high word, which only the
                        // one-operand form produces, and that form writes %rdx.
                        writeln!(writer, "    pushq %rdx")?;
                        writeln!(writer, "    mulq %{scratch}")?;
                        writeln!(writer, "    setc %al")?;
                        writeln!(writer, "    movzbq %al, %rax")?;
                        writeln!(writer, "    popq %rdx")?;
                        if let Some(dp) = mapped_register(dst, reg_alloc) {
                            writeln!(writer, "    movq %rax, %{dp}")?;
                        } else if let Register::Virtual(vreg) = dst {
                            store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                        }
                        reg_alloc.free_scratch(scratch);
                        return Ok(());
                    }
                }
            }
            writeln!(writer, "    movzbq %al, %rax")?;
            if let Some(dp) = mapped_register(dst, reg_alloc) {
                writeln!(writer, "    movq %rax, %{dp}")?;
            } else if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
            reg_alloc.free_scratch(scratch);
        }
        MirInst::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).ok_or_else(|| {
                LaminaError::ValidationError("stack address has no frame slot".into())
            })?;
            if let Some(dp) = mapped_register(dst, reg_alloc) {
                writeln!(writer, "    leaq {offset}(%rbp), %{dp}")?;
                return Ok(());
            }
            writeln!(writer, "    leaq {offset}(%rbp), %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Lea { dst, base, offset } => {
            if let (Some(dp), Some(bp)) = (
                mapped_register(dst, reg_alloc),
                mapped_register(base, reg_alloc),
            ) {
                if *offset == 0 {
                    if bp != dp {
                        writeln!(writer, "    movq %{bp}, %{dp}")?;
                    }
                } else {
                    writeln!(writer, "    leaq {offset}(%{bp}), %{dp}")?;
                }
                return Ok(());
            }
            match base {
                Register::Virtual(vreg) => {
                    if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        if *offset == 0 {
                            writeln!(writer, "    movq %{phys}, %rax")?;
                        } else {
                            writeln!(writer, "    leaq {offset}(%{phys}), %rax")?;
                        }
                    } else if let Some(slot_offset) = stack_slots.get(vreg) {
                        let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
                        writeln!(writer, "    movq {slot_offset}(%rbp), %{scratch}")?;
                        if *offset == 0 {
                            writeln!(writer, "    movq %{scratch}, %rax")?;
                        } else {
                            writeln!(writer, "    leaq {offset}(%{scratch}), %rax")?;
                        }
                        if scratch != "rbx" {
                            reg_alloc.free_scratch(scratch);
                        }
                    } else {
                        return Err(LaminaError::ValidationError(format!(
                            "x86_64 backend cannot lower LEA for base vreg {vreg:?} (no mapping/slot)"
                        )));
                    }
                }
                Register::Physical(phys) => {
                    if *offset == 0 {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    } else {
                        writeln!(writer, "    leaq {}(%{}), %rax", offset, phys.name)?;
                    }
                }
            }

            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let (add, sub, mul, div) = if is_f32(ty) {
                ("addss", "subss", "mulss", "divss")
            } else {
                ("addsd", "subsd", "mulsd", "divsd")
            };
            load_float_operand_to_xmm(lhs, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            load_float_operand_to_xmm(rhs, writer, reg_alloc, stack_slots, "xmm1", ty)?;
            let sse_op = match op {
                FloatBinOp::FAdd => add,
                FloatBinOp::FSub => sub,
                FloatBinOp::FMul => mul,
                FloatBinOp::FDiv => div,
            };
            writeln!(writer, "    {sse_op} %xmm1, %xmm0")?;
            if let Register::Virtual(vreg) = dst {
                store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, ty)?;
            }
        }
        MirInst::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => emit_int_to_float(
            writer,
            reg_alloc,
            stack_slots,
            from,
            to,
            *signed,
            dst,
            src,
            pointer_width,
        )?,
        MirInst::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => emit_float_to_int(
            writer,
            reg_alloc,
            stack_slots,
            from,
            to,
            *signed,
            dst,
            src,
            pointer_width,
        )?,
        MirInst::CallIndirect {
            target,
            arg_types,
            args,
            variadic_fixed: _,
            ret,
            ret_ty,
        } => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let descs: Vec<ArgDesc> = arg_types.iter().copied().map(ArgDesc::scalar).collect();
            let types: Vec<MirType> = arg_types.clone();
            let locations = classify_argument_descs(&descs, target_os);
            let stack_indices: Vec<usize> = locations
                .iter()
                .enumerate()
                .filter_map(|(index, location)| {
                    matches!(location, ArgLocation::Stack(_)).then_some(index)
                })
                .collect();
            let num_stack_args = stack_indices.len();

            if target_os == TargetOperatingSystem::Windows {
                let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                let total_stack = align_up(
                    shadow_space + (num_stack_args * stack::SLOT_SIZE),
                    stack::ALIGNMENT,
                );
                if total_stack > 0 {
                    writeln!(writer, "    subq ${total_stack}, %rsp")?;
                }
                for (slot, &index) in stack_indices.iter().enumerate() {
                    let stack_offset = shadow_space + (slot * stack::SLOT_SIZE);
                    load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                    writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                }
            } else {
                if num_stack_args % 2 == 1 {
                    writeln!(writer, "    subq $8, %rsp")?;
                }
                for &index in stack_indices.iter().rev() {
                    load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                    writeln!(writer, "    pushq %rax")?;
                }
            }

            // r11 is caller-saved and never an argument register, so the target
            // survives loading the arguments.
            load_operand_to_rax(target, writer, reg_alloc, stack_slots)?;
            writeln!(writer, "    movq %rax, %r11")?;

            load_register_arguments(
                writer,
                args,
                &types,
                &locations,
                arg_regs,
                reg_alloc,
                stack_slots,
            )?;
            if target_os == TargetOperatingSystem::Windows {
                duplicate_win64_float_arguments(writer, &locations, arg_regs)?;
            }

            if target_os != TargetOperatingSystem::Windows {
                // al carries the vector-register count for a variadic callee, and a
                // prototypeless target may be one.
                let xmm_count = locations
                    .iter()
                    .filter(|location| matches!(location, ArgLocation::Xmm(_)))
                    .count();
                writeln!(writer, "    movl ${xmm_count}, %eax")?;
            }
            writeln!(writer, "    call *%r11")?;

            if target_os == TargetOperatingSystem::Windows {
                let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                let total_stack = align_up(
                    shadow_space + (num_stack_args * stack::SLOT_SIZE),
                    stack::ALIGNMENT,
                );
                if total_stack > 0 {
                    writeln!(writer, "    addq ${total_stack}, %rsp")?;
                }
            } else if num_stack_args > 0 {
                let pushed = num_stack_args * stack::SLOT_SIZE;
                let padded = align_up(pushed, stack::ALIGNMENT);
                writeln!(writer, "    addq ${padded}, %rsp")?;
            }

            if let Some(Register::Virtual(vreg)) = ret {
                match ret_ty {
                    Some(ty) if ty.is_float() => {
                        store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, ty)?;
                    }
                    _ => store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?,
                }
            }
        }
        MirInst::FloatConvert { from, to, dst, src } => {
            load_float_operand_to_xmm(src, writer, reg_alloc, stack_slots, "xmm0", from)?;
            if is_f32(to) {
                writeln!(writer, "    cvtsd2ss %xmm0, %xmm0")?;
            } else {
                writeln!(writer, "    cvtss2sd %xmm0, %xmm0")?;
            }
            if let Register::Virtual(vreg) = dst {
                store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, to)?;
            }
        }
        MirInst::FloatUnary { op, ty, dst, src } => {
            load_float_operand_to_xmm(src, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            match op {
                FloatUnOp::FNeg => {
                    if is_f32(ty) {
                        // XOR with the sign bit of f32
                        writeln!(writer, "    movq $0x80000000, %rax")?;
                        writeln!(writer, "    movd %rax, %xmm1")?;
                        writeln!(writer, "    xorps %xmm1, %xmm0")?;
                    } else {
                        // XOR with the sign bit of f64
                        writeln!(writer, "    movq $0x8000000000000000, %rax")?;
                        writeln!(writer, "    movq %rax, %xmm1")?;
                        writeln!(writer, "    xorpd %xmm1, %xmm0")?;
                    }
                }
                FloatUnOp::FSqrt => {
                    let sqrt = if is_f32(ty) { "sqrtss" } else { "sqrtsd" };
                    writeln!(writer, "    {sqrt} %xmm0, %xmm0")?;
                }
            }
            if let Register::Virtual(vreg) = dst {
                store_xmm_to_register("xmm0", vreg, writer, reg_alloc, stack_slots, ty)?;
            }
        }
        MirInst::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            load_float_operand_to_xmm(lhs, writer, reg_alloc, stack_slots, "xmm0", ty)?;
            load_float_operand_to_xmm(rhs, writer, reg_alloc, stack_slots, "xmm1", ty)?;
            let ucomi = if is_f32(ty) { "ucomiss" } else { "ucomisd" };
            writeln!(writer, "    {ucomi} %xmm1, %xmm0")?;
            match op {
                FloatCmpOp::Eq => writeln!(writer, "    sete %al")?,
                FloatCmpOp::Ne => writeln!(writer, "    setne %al")?,
                FloatCmpOp::Lt => writeln!(writer, "    setb %al")?,
                FloatCmpOp::Le => writeln!(writer, "    setbe %al")?,
                FloatCmpOp::Gt => writeln!(writer, "    seta %al")?,
                FloatCmpOp::Ge => writeln!(writer, "    setae %al")?,
            }
            if matches!(op, FloatCmpOp::Ne) {
                writeln!(writer, "    setp %ah")?;
                writeln!(writer, "    orb %ah, %al")?;
            } else {
                writeln!(writer, "    setnp %ah")?;
                writeln!(writer, "    andb %ah, %al")?;
            }
            writeln!(writer, "    movzbq %al, %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Select {
            dst,
            cond,
            true_val,
            false_val,
            ty: _,
        } => {
            // Load false_val into a scratch register, true_val into rax.
            // Then test the condition and cmovnz true_val path.
            let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
            load_operand_to_register(false_val, writer, reg_alloc, stack_slots, scratch)?;
            load_operand_to_rax(true_val, writer, reg_alloc, stack_slots)?;
            match cond {
                Register::Virtual(vreg) => {
                    if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        writeln!(writer, "    testq %{phys}, %{phys}")?;
                    } else if let Some(offset) = stack_slots.get(vreg) {
                        writeln!(writer, "    cmpq $0, {offset}(%rbp)")?;
                    }
                }
                Register::Physical(p) => {
                    writeln!(writer, "    testq %{}, %{}", p.name, p.name)?;
                }
            }
            // cmovz: if condition is zero (false), move scratch into rax
            writeln!(writer, "    cmovzq %{scratch}, %rax")?;
            if scratch != "rbx" {
                reg_alloc.free_scratch(scratch);
            }
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Switch {
            value,
            cases,
            default,
        } => {
            // Load switch value into rax, then emit a linear cmp/je chain.
            match value {
                Register::Virtual(vreg) => {
                    load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(p) => {
                    writeln!(writer, "    movq %{}, %rax", p.name)?;
                }
            }
            for (case_val, case_label) in cases {
                writeln!(writer, "    cmpq ${case_val}, %rax")?;
                writeln!(writer, "    je .L_{func_name}.{case_label}")?;
            }
            writeln!(writer, "    jmp .L_{func_name}.{default}")?;
        }
        MirInst::Comment { text } => {
            writeln!(writer, "    # {text}")?;
        }
        MirInst::Unreachable => {
            writeln!(writer, "    ud2")?;
        }
        MirInst::IntUnary { op, ty, dst, src } => {
            emit_int_unary(
                *op,
                ty,
                dst,
                src,
                writer,
                reg_alloc,
                stack_slots,
                settings.x86_cpu_level(),
                pointer_width,
            )?;
        }
        MirInst::IntRotate {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            emit_int_rotate(
                *op,
                ty,
                dst,
                (lhs, rhs),
                writer,
                reg_alloc,
                stack_slots,
                pointer_width,
            )?;
        }
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match (
                    abi_extensions.return_aggregate(func_name),
                    abi_extensions.float_return(func_name),
                ) {
                    // The returned operand addresses the aggregate. Small ones go back
                    // in rax and rdx; a large one is copied to the caller's buffer and
                    // its address comes back in rax.
                    (Some(aggregate), _) => {
                        load_operand_to_rax(val, writer, reg_alloc, stack_slots)?;
                        if let Some(offset) = sret_slot {
                            writeln!(writer, "    movq {offset}(%rbp), %rcx")?;
                            copy_exact_bytes(writer, aggregate.size, "rax", "rcx")?;
                            writeln!(writer, "    movq %rcx, %rax")?;
                        } else if aggregate.class == AggregateClass::Integer {
                            if aggregate.words() > 1 {
                                writeln!(writer, "    movq 8(%rax), %rdx")?;
                            }
                            writeln!(writer, "    movq (%rax), %rax")?;
                        } else {
                            writeln!(writer, "    movq %rax, %r11")?;
                            let mut sse_returns = ["xmm0", "xmm1"].into_iter();
                            for word in 0..aggregate.words() {
                                if aggregate.eightbyte_class(word) == EightbyteClass::Sse {
                                    let xmm = sse_returns.next().ok_or_else(|| {
                                        LaminaError::ValidationError(
                                            "aggregate return needs more SSE registers than the ABI has"
                                                .to_string(),
                                        )
                                    })?;
                                    writeln!(writer, "    movq {}(%r11), %{xmm}", word * 8)?;
                                }
                            }
                            let mut integer_returns = ["rax", "rdx"].into_iter();
                            for word in 0..aggregate.words() {
                                if aggregate.eightbyte_class(word) == EightbyteClass::Integer {
                                    let gpr = integer_returns.next().ok_or_else(|| {
                                        LaminaError::ValidationError(
                                            "aggregate return needs more general registers than the ABI has"
                                                .to_string(),
                                        )
                                    })?;
                                    writeln!(writer, "    movq {}(%r11), %{gpr}", word * 8)?;
                                }
                            }
                        }
                        writer.rax_value = None;
                    }
                    (None, Some(ty)) => {
                        load_float_operand_to_xmm(val, writer, reg_alloc, stack_slots, "xmm0", &ty)?
                    }
                    (None, None) => load_operand_to_rax(val, writer, reg_alloc, stack_slots)?,
                }
            }
            // Epilogue
            restore_saved_registers(writer, stack_size, saved_regs, sret_slot.is_some(), va_area)?;
            if has_frame {
                X86Frame::generate_epilogue(writer, stack_size, unwind)?;
            } else {
                writeln!(writer, "    ret")?;
            }
        }
        MirInst::Jmp { target } => {
            if target != next_label {
                writeln!(writer, "    jmp .L_{func_name}.{target}")?;
            }
        }
        MirInst::Br {
            cond,
            true_target,
            false_target,
        } => {
            if let Register::Virtual(vreg) = cond {
                load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
            }
            writeln!(writer, "    testq %rax, %rax")?;
            if true_target == next_label {
                writeln!(writer, "    jz .L_{func_name}.{false_target}")?;
            } else {
                writeln!(writer, "    jnz .L_{func_name}.{true_target}")?;
                if false_target != next_label {
                    writeln!(writer, "    jmp .L_{func_name}.{false_target}")?;
                }
            }
        }
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support MIR instruction {other:?}"
            )));
        }
    }

    Ok(())
}

/// Sign- or zero-extending load mnemonic for a value of this type into a 64-bit register.
pub(super) fn load_mnemonic_for(ty: &MirType, pointer_width: PointerWidth) -> &'static str {
    match ty.size_bytes(pointer_width) {
        1 => "movsbq",
        2 => "movswq",
        4 => "movslq",
        _ => "movq",
    }
}

/// Width suffix for a `mov` touching memory of this type.
pub(super) fn size_suffix(ty: &MirType, pointer_width: PointerWidth) -> &'static str {
    match ty.size_bytes(pointer_width) {
        1 => "b",
        2 => "w",
        4 => "l",
        _ => "q",
    }
}

/// The sub-register of `rax` holding exactly the bytes of this type.
pub(super) fn rax_for(ty: &MirType, pointer_width: PointerWidth) -> &'static str {
    match ty.size_bytes(pointer_width) {
        1 => "al",
        2 => "ax",
        4 => "eax",
        _ => "rax",
    }
}

/// Condition-code suffix for the negation of a comparison.
pub(super) fn inverted_cmp_suffix(op: &IntCmpOp) -> Result<&'static str, LaminaError> {
    Ok(match op {
        IntCmpOp::Eq => "ne",
        IntCmpOp::Ne => "e",
        IntCmpOp::SLt => "ge",
        IntCmpOp::SLe => "g",
        IntCmpOp::SGt => "le",
        IntCmpOp::SGe => "l",
        IntCmpOp::ULt => "ae",
        IntCmpOp::ULe => "a",
        IntCmpOp::UGt => "be",
        IntCmpOp::UGe => "b",
        #[allow(unreachable_patterns)]
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support integer comparison {other:?}"
            )));
        }
    })
}

/// Condition-code suffix for an integer comparison, as used by `setCC` and `jCC`.
pub(super) fn cmp_suffix(op: &IntCmpOp) -> Result<&'static str, LaminaError> {
    Ok(match op {
        IntCmpOp::Eq => "e",
        IntCmpOp::Ne => "ne",
        IntCmpOp::SLt => "l",
        IntCmpOp::SLe => "le",
        IntCmpOp::SGt => "g",
        IntCmpOp::SGe => "ge",
        IntCmpOp::ULt => "b",
        IntCmpOp::ULe => "be",
        IntCmpOp::UGt => "a",
        IntCmpOp::UGe => "ae",
        #[allow(unreachable_patterns)]
        other => {
            return Err(LaminaError::ValidationError(format!(
                "x86_64 backend does not support integer comparison {other:?}"
            )));
        }
    })
}

/// Emit the `cmpq` for an integer comparison, leaving the flags set. Returns the scratch
/// register that must be released, or "" when the right operand folded into an immediate.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_int_compare(
    op: &IntCmpOp,
    ty: &MirType,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut RaxTrackingWriter<impl Write>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
    pointer_width: PointerWidth,
) -> Result<&'static str, LaminaError> {
    let unsigned = matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    );
    // Both sides already in registers at full width: compare them where they sit.
    if !unsigned || ty.size_bytes(pointer_width) == 8 {
        let physical = |value: &Operand| match value {
            Operand::Register(Register::Virtual(vreg)) => reg_alloc.get_mapping_for(vreg),
            _ => None,
        };
        if let Some(left) = physical(lhs)
            && let Some(v) = cmp_immediate(rhs, ty, unsigned, pointer_width)
        {
            writeln!(writer, "    cmpq ${v}, %{left}")?;
            return Ok("");
        }
        if let (Some(left), Some(right)) = (physical(lhs), physical(rhs)) {
            writeln!(writer, "    cmpq %{right}, %{left}")?;
            return Ok("");
        }
    }
    load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
    if let Some(v) = cmp_immediate(rhs, ty, unsigned, pointer_width) {
        if unsigned {
            extend_integer(writer, "rax", ty.size_bytes(pointer_width), false)?;
        }
        writeln!(writer, "    cmpq ${v}, %rax")?;
        return Ok("");
    }
    let scratch = reg_alloc.alloc_scratch().ok_or_else(scratch_error)?;
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
    if unsigned {
        extend_integer(writer, "rax", ty.size_bytes(pointer_width), false)?;
        extend_integer(writer, scratch, ty.size_bytes(pointer_width), false)?;
    }
    writeln!(writer, "    cmpq %{scratch}, %rax")?;
    Ok(scratch)
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod narrow_integer_tests {
    use crate::mir_codegen::{
        simd_support::tools_available,
        test_support::{PRINT_I64_HELPERS, compile},
    };
    use crate::test_support::TempDir;
    use std::{error::Error, fmt::Write as _, fs, process::Command};

    #[test]
    fn narrow_signed_operations_at_every_optimization_level() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return Ok(());
        }
        let guard = TempDir::new("lamina-narrow");
        let directory = guard.path();
        let mut functions = String::new();
        let mut body = String::new();
        let mut expected = String::new();
        let mut id = 0;
        for width in [8, 16, 32] {
            for (op, left, right, result) in [
                ("div", -20, 3, -6),
                ("rem", -20, 3, -2),
                ("shr", -20, 3, -3),
                ("div", 20, -3, -6),
                ("rem", 20, -3, 2),
                ("div", -20, -3, 6),
                ("lt", -20, 3, 1),
                ("le", -20, 3, 1),
                ("gt", -20, 3, 0),
                ("ge", -20, 3, 0),
                ("lt", 3, -20, 0),
                ("le", 3, -20, 0),
                ("gt", 3, -20, 1),
                ("ge", 3, -20, 1),
                ("lt", -20, -20, 0),
                ("le", -20, -20, 1),
                ("gt", -20, -20, 0),
                ("ge", -20, -20, 1),
            ] {
                let compare = matches!(op, "lt" | "le" | "gt" | "ge");
                let conversion = if compare {
                    "extend.bool.i64".to_string()
                } else {
                    format!("extend.i{width}.i64")
                };
                writeln!(
                    functions,
                    "fn @op{id}(i{width} %a, i{width} %b) -> i64 {{\nentry:\n%r = {op}.i{width} %a, %b\n%z = {conversion} %r\nret.i64 %z\n}}"
                )?;
                writeln!(
                    body,
                    "%a{id} = add.i{width} {left}, 0\n%b{id} = add.i{width} {right}, 0\n%r{id} = {op}.i{width} %a{id}, %b{id}\n%z{id} = {conversion} %r{id}\ncall @print_i64(%z{id})\n%c{id} = call @op{id}(%a{id}, %b{id})\ncall @print_i64(%c{id})"
                )?;
                writeln!(expected, "{result}\n{result}")?;
                id += 1;
            }
            let maximum = (1_i64 << (width - 1)) - 1;
            let minimum = -(1_i64 << (width - 1));
            writeln!(
                body,
                "%overflow{width} = add.i{width} {maximum}, 1\n%negative{width} = shr.i{width} %overflow{width}, 3\n%wide{width} = extend.i{width}.i64 %negative{width}\ncall @print_i64(%wide{width})\n%unsigned{width}_bits = bitcast.i{width}.u{width} %overflow{width}\n%unsigned{width} = extend.u{width}.i64 %unsigned{width}_bits\ncall @print_i64(%unsigned{width})"
            )?;
            writeln!(expected, "{}\n{}", minimum >> 3, 1_i64 << (width - 1))?;
            // Adjacent elements catch stores that overwrite beyond their type width.
            writeln!(
                body,
                "%array{width} = stackalloc.[2 x i{width}]\n%low{width} = getelem.i{width} %array{width}, 0\n%high{width} = getelem.i{width} %array{width}, 1\nstore.i{width} %high{width}, 99\nstore.i{width} %low{width}, -20\n%load{width} = load.i{width} %low{width}\n%div{width} = div.i{width} %load{width}, 3\n%loaded{width} = extend.i{width}.i64 %div{width}\ncall @print_i64(%loaded{width})\n%neighbor{width} = load.i{width} %high{width}\n%neighborwide{width} = extend.i{width}.i64 %neighbor{width}\ncall @print_i64(%neighborwide{width})"
            )?;
            writeln!(expected, "-6\n99")?;
        }
        let source = directory.join("narrow.lamina");
        fs::write(
            &source,
            format!(
                "{functions}\n@export\nfn @main() -> i64 {{\nentry:\n{body}\nret.i64 0\n}}\n{PRINT_I64_HELPERS}"
            ),
        )?;
        for level in ["0", "1", "2", "3"] {
            for allocator in ["incremental", "linear", "graph"] {
                let executable = directory.join(format!("narrow-{level}-{allocator}"));
                compile(
                    &source,
                    &executable,
                    &["--opt-level", level, "--regalloc", allocator],
                )?;
                let output = Command::new("timeout")
                    .arg("10")
                    .arg(&executable)
                    .output()?;
                assert!(output.status.success(), "O{level}/{allocator}: {output:?}");
                assert_eq!(
                    String::from_utf8_lossy(&output.stdout),
                    expected,
                    "O{level}/{allocator}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn unsigned_narrow_parameters_and_returns_hold_the_canonical_form() -> Result<(), Box<dyn Error>>
    {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return Ok(());
        }
        let guard = TempDir::new("lamina-narrow-unsigned");
        let directory = guard.path();
        let mut functions = String::new();
        let mut body = String::new();
        let mut expected = String::new();
        for width in [8, 16, 32] {
            let maximum = (1_i64 << width) - 1;
            writeln!(
                functions,
                "fn @ident{width}(u{width} %p) -> u{width} {{\nentry:\nret.u{width} %p\n}}"
            )?;
            writeln!(
                functions,
                "fn @same{width}(u{width} %p) -> i64 {{\nentry:\n%x = xor.u{width} %p, 0\n%c = eq.u{width} %p, %x\n%z = extend.bool.i64 %c\nret.i64 %z\n}}"
            )?;
            writeln!(
                body,
                "%p{width} = trunc.i64.u{width} {maximum}\n%r{width} = call @ident{width}(%p{width})\n%w{width} = extend.u{width}.i64 %r{width}\ncall @print_i64(%w{width})\n%e{width} = call @same{width}(%p{width})\ncall @print_i64(%e{width})\n%x{width} = xor.u{width} %r{width}, 0\n%q{width} = eq.u{width} %r{width}, %x{width}\n%qz{width} = extend.bool.i64 %q{width}\ncall @print_i64(%qz{width})\n%d{width} = div.u{width} %r{width}, 7\n%dz{width} = extend.u{width}.i64 %d{width}\ncall @print_i64(%dz{width})\n%g{width} = gt.u{width} %r{width}, 1\n%gz{width} = extend.bool.i64 %g{width}\ncall @print_i64(%gz{width})"
            )?;
            writeln!(expected, "{maximum}\n1\n1\n{}\n1", maximum / 7)?;
        }
        let source = directory.join("unsigned.lamina");
        fs::write(
            &source,
            format!(
                "{functions}\n@export\nfn @main() -> i64 {{\nentry:\n{body}\nret.i64 0\n}}\n{PRINT_I64_HELPERS}"
            ),
        )?;
        for level in ["0", "1", "2", "3"] {
            for allocator in ["incremental", "linear", "graph"] {
                let executable = directory.join(format!("unsigned-{level}-{allocator}"));
                compile(
                    &source,
                    &executable,
                    &["--opt-level", level, "--regalloc", allocator],
                )?;
                let output = Command::new("timeout")
                    .arg("10")
                    .arg(&executable)
                    .output()?;
                assert!(output.status.success(), "O{level}/{allocator}: {output:?}");
                assert_eq!(
                    String::from_utf8_lossy(&output.stdout),
                    expected,
                    "O{level}/{allocator}"
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod variadic_pointer_call_tests {
    use crate::mir_codegen::{
        simd_support::tools_available,
        test_support::{PRINT_I64_HELPERS, compile},
    };
    use crate::test_support::TempDir;
    use std::{error::Error, fs, process::Command};

    #[test]
    fn callptr_passes_float_varargs_like_a_direct_call() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return Ok(());
        }
        let guard = TempDir::new("lamina-callptr-variadic");
        let directory = guard.path();
        let source = directory.join("variadic.lamina");
        fs::write(
            &source,
            format!(
                r#"@extern
fn @snprintf(ptr %buf, i64 %n, ptr %fmt, ...) -> i32

global $fmt: [16 x i8] = "%.1f %.1f %.1f\n"

fn @sum(i64 %n, ...) -> i64 {{
entry:
  %ap = stackalloc.valist
  %dsum = stackalloc.f64
  %i = stackalloc.i64
  store.f64 %dsum, 0.0
  store.i64 %i, 0
  va_start %ap
  jmp loop

loop:
  %iv = load.i64 %i
  %c = lt.i64 %iv, %n
  br %c, body, done

body:
  %y = va_arg.f64 %ap
  %b = load.f64 %dsum
  %t = add.f64 %b, %y
  store.f64 %dsum, %t
  %iv2 = add.i64 %iv, 1
  store.i64 %i, %iv2
  jmp loop

done:
  va_end %ap
  %ds = load.f64 %dsum
  %r = fptoint.f64.i64 %ds
  ret.i64 %r
}}

@export
fn @main() -> i64 {{
entry:
  %buf = stackalloc.i64
  %a = add.f32 2.5, 0.0
  %d = call @sum(i64 4, f64 10.0, f64 20.0, f64 30.0, f32 %a)
  %p = funcaddr @sum
  %c = callptr.i64 %p(i64 4, ..., f64 10.0, f64 20.0, f64 30.0, f32 %a)
  %q = funcaddr @snprintf
  %w = callptr.i32 %q(ptr %buf, i64 8, ptr $fmt, ..., f64 1234.5, f32 %a, f64 -7.5)
  %n = extend.i32.i64 %w
  call @print_i64(%d)
  call @print_i64(%c)
  call @print_i64(%n)
  ret.i64 0
}}
{PRINT_I64_HELPERS}"#
            ),
        )?;
        for level in ["0", "2"] {
            let executable = directory.join(format!("variadic-{level}"));
            compile(&source, &executable, &["--opt-level", level])?;
            let output = Command::new("timeout")
                .arg("10")
                .arg(&executable)
                .output()?;
            assert!(output.status.success(), "O{level}: {output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "62\n62\n16\n",
                "O{level}"
            );
        }
        Ok(())
    }
}
