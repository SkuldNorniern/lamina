use super::*;

pub(super) fn restore_saved_registers<W: Write>(
    writer: &mut W,
    stack_size: usize,
    saved_regs: &[&str],
) -> Result<(), Error> {
    let first_slot = stack_size / 8 - saved_regs.len();
    for (index, reg) in saved_regs.iter().enumerate() {
        let offset = X86Frame::calculate_stack_offset(first_slot + index);
        writeln!(writer, "    movq {offset}(%rbp), %{reg}")?;
    }
    Ok(())
}

fn writeptr_width(operand: &Operand) -> Result<usize, LaminaError> {
    match operand {
        Operand::Immediate(Immediate::I64(width)) => usize::try_from(*width).map_err(|_| {
            LaminaError::ValidationError("writeptr width must be non-negative".to_string())
        }),
        _ => Err(LaminaError::ValidationError(
            "writeptr width must be an immediate".to_string(),
        )),
    }
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
    def_regs: &BTreeSet<VirtualReg>,
    settings: &MirCodegenSettings,
    debug_line: &mut u32,
    abi_extensions: &AbiExtensions,
    next_label: &str,
    self_param_homes: &[Option<String>],
    param_extensions: &[Option<(usize, bool)>],
    entry_label: &str,
) -> Result<(), LaminaError> {
    if settings.emit_asm_debug_lines {
        *debug_line = debug_line.saturating_add(1);
        writeln!(writer, "    .loc 1 {} 0", *debug_line)?;
    }
    if let MirInst::IntConvert {
        from,
        to,
        signed,
        dst,
        src,
    } = inst
    {
        load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
        extend_integer(
            writer,
            "rax",
            from.size_bytes().min(to.size_bytes()),
            *signed,
        )?;
        if to.size_bytes() < 8 {
            extend_integer(writer, "rax", to.size_bytes(), true)?;
        }
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
            load_copy_operand_to_rax(ty, src, writer, reg_alloc, stack_slots)?;
            match dst {
                Register::Virtual(vreg) => {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
                Register::Physical(phys) if phys.name != "rax" => {
                    writeln!(writer, "    movq %rax, %{}", phys.name)?;
                }
                Register::Physical(_) => {}
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
                match op {
                    IntBinOp::Shl | IntBinOp::AShr | IntBinOp::LShr => {
                        // Load lhs into dp
                        load_operand_to_register(lhs, writer, reg_alloc, stack_slots, dp)?;
                        if matches!(op, IntBinOp::AShr) {
                            extend_integer(writer, dp, ty.size_bytes(), true)?;
                        }
                        if matches!(op, IntBinOp::AShr) {
                            extend_integer(writer, "rax", ty.size_bytes(), true)?;
                        }
                        if matches!(op, IntBinOp::LShr) {
                            extend_integer(writer, dp, ty.size_bytes(), false)?;
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
                                let sv = sv & (ty.size_bytes() as u64 * 8 - 1);
                                match op {
                                    IntBinOp::Shl => writeln!(writer, "    shlq ${sv}, %{dp}")?,
                                    IntBinOp::AShr => writeln!(writer, "    sarq ${sv}, %{dp}")?,
                                    IntBinOp::LShr => writeln!(writer, "    shrq ${sv}, %{dp}")?,
                                    _ => unreachable!(),
                                }
                            }
                            Operand::Register(_) => {
                                // Need shift count in %cl; load rhs into rcx
                                load_operand_to_register(
                                    rhs,
                                    writer,
                                    reg_alloc,
                                    stack_slots,
                                    "rcx",
                                )?;
                                if ty.size_bytes() < 8 {
                                    writeln!(
                                        writer,
                                        "    andl ${}, %ecx",
                                        ty.size_bytes() * 8 - 1
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
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
                            extend_integer(writer, "rax", ty.size_bytes(), false)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), false)?;
                        }
                        if matches!(op, IntBinOp::SDiv | IntBinOp::SRem) && ty.size_bytes() < 8 {
                            extend_integer(writer, "rax", ty.size_bytes(), true)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), true)?;
                            let (_, overflow) = op.division_checks(
                                *ty,
                                lhs.integer_constant(),
                                rhs.integer_constant(),
                            );
                            if overflow {
                                let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
                                writeln!(
                                    writer,
                                    "    cmpq $-1, %rcx\n    jne 2f\n    cmpq ${minimum}, %rax\n    jne 2f\n    ud2\n2:"
                                )?;
                            }
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
                            Operand::Register(rhs_reg) => {
                                // Get rhs physical register if available for direct register op
                                let rhs_phys = match rhs_reg {
                                    Register::Virtual(v) => reg_alloc.get_mapping_for(v),
                                    Register::Physical(p) => Some(p.name),
                                };
                                if let Some(rp) = rhs_phys {
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
                extend_integer(writer, dp, ty.size_bytes(), true)?;
            } else {
                // No physical register for dst — fall back to rax accumulator path
                load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
                if matches!(op, IntBinOp::LShr) {
                    extend_integer(writer, "rax", ty.size_bytes(), false)?;
                }

                match op {
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
                            let shift_val = shift_val & (ty.size_bytes() as u64 * 8 - 1);
                            match op {
                                IntBinOp::Shl => writeln!(writer, "    shlq ${shift_val}, %rax")?,
                                IntBinOp::AShr => writeln!(writer, "    sarq ${shift_val}, %rax")?,
                                IntBinOp::LShr => writeln!(writer, "    shrq ${shift_val}, %rax")?,
                                _ => unreachable!(),
                            }
                        }
                        Operand::Register(_) => {
                            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                            load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
                            writeln!(writer, "    movq %{scratch}, %rcx")?;
                            if ty.size_bytes() < 8 {
                                writeln!(writer, "    andl ${}, %ecx", ty.size_bytes() * 8 - 1)?;
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
                        load_operand_to_register(rhs, writer, reg_alloc, stack_slots, "rcx")?;
                        if matches!(op, IntBinOp::UDiv | IntBinOp::URem) {
                            extend_integer(writer, "rax", ty.size_bytes(), false)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), false)?;
                        }
                        if matches!(op, IntBinOp::SDiv | IntBinOp::SRem) && ty.size_bytes() < 8 {
                            extend_integer(writer, "rax", ty.size_bytes(), true)?;
                            extend_integer(writer, "rcx", ty.size_bytes(), true)?;
                            let (_, overflow) = op.division_checks(
                                *ty,
                                lhs.integer_constant(),
                                rhs.integer_constant(),
                            );
                            if overflow {
                                let minimum = ty.signed_integer(1i64 << (ty.size_bytes() * 8 - 1));
                                writeln!(
                                    writer,
                                    "    cmpq $-1, %rcx\n    jne 2f\n    cmpq ${minimum}, %rax\n    jne 2f\n    ud2\n2:"
                                )?;
                            }
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
                    }
                    _ => {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
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

                extend_integer(writer, "rax", ty.size_bytes(), true)?;
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
            let scratch = emit_int_compare(op, ty, lhs, rhs, writer, reg_alloc, stack_slots)?;
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
            if name == "print" {
                if let Some(arg) = args.first() {
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                    match target_os {
                        TargetOperatingSystem::MacOS => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            // al carries the vector-register count for variadic
                            // calls; printf reads it before saving xmm state.
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call _printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call _fflush")?;
                        }
                        TargetOperatingSystem::Windows => {
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rcx")?;
                            writeln!(writer, "    movq %rax, %rdx")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    movq $0, %rcx")?;
                            writeln!(writer, "    call fflush")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        }
                        _ => {
                            writeln!(writer, "    leaq .L_mir_fmt_int(%rip), %rdi")?;
                            writeln!(writer, "    movq %rax, %rsi")?;
                            writeln!(writer, "    xorl %eax, %eax")?;
                            writeln!(writer, "    call printf")?;
                            writeln!(writer, "    movq $0, %rdi")?;
                            writeln!(writer, "    call fflush")?;
                        }
                    }
                }
            } else if name == "writebyte" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writebyte requires one argument".to_string())
                })?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movb %al, (%rsp)")?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
                        // Windows syscall numbers are unstable and cannot be invoked
                        // directly like on Linux/macOS. Route the byte through the CRT
                        // (putchar) instead, matching how `print` uses `printf`.
                        writeln!(writer, "    movzbl (%rsp), %ecx")?;
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call putchar")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        // putchar yields the character written; the syscall
                        // paths yield the byte count, so report 1 or -1.
                        writeln!(writer, "    movl %eax, %ecx")?;
                        writeln!(writer, "    movl $1, %eax")?;
                        writeln!(writer, "    movl $-1, %edx")?;
                        writeln!(writer, "    cmpl $-1, %ecx")?;
                        writeln!(writer, "    cmovel %edx, %eax")?;
                        writeln!(writer, "    cltq")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;
            } else if name == "readbyte" && args.is_empty() {
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
                        writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        writeln!(writer, "    call getchar")?;
                        writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        // getchar reports EOF as -1; the syscall paths report 0.
                        writeln!(writer, "    movslq %eax, %rax")?;
                        writeln!(writer, "    xorl %ecx, %ecx")?;
                        writeln!(writer, "    cmpq $-1, %rax")?;
                        writeln!(writer, "    cmoveq %rcx, %rax")?;
                    }
                    _ => {
                        writeln!(writer, "    movq ${}, %rax", linux::SYS_READ)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDIN)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                // getchar already returns the byte (or -1); the syscall returns a
                // count, so swap in the byte it stored only on a full read.
                if target_os != TargetOperatingSystem::Windows {
                    writeln!(writer, "    movzbl (%rsp), %ecx")?;
                    writeln!(writer, "    cmpq $1, %rax")?;
                    writeln!(writer, "    cmovel %ecx, %eax")?;
                }

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else if name == "writeptr" && args.len() == 2 {
                let width = writeptr_width(&args[1])?;
                match target_os {
                    TargetOperatingSystem::Windows => {
                        // Windows syscall numbers are unstable; emit each byte through putchar,
                        // matching writebyte. The pointer is kept in the stack slot across calls.
                        writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                        load_operand_to_rax(&args[0], writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, (%rsp)")?;
                        for offset in 0..width {
                            writeln!(writer, "    movq (%rsp), %rax")?;
                            writeln!(writer, "    movzbl {offset}(%rax), %ecx")?;
                            writeln!(writer, "    subq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                            writeln!(writer, "    call putchar")?;
                            writeln!(writer, "    addq ${}, %rsp", windows::SHADOW_SPACE_SIZE)?;
                        }
                        // putchar yields the character written; report the byte count or -1.
                        writeln!(writer, "    movl %eax, %ecx")?;
                        writeln!(writer, "    movl ${width}, %eax")?;
                        writeln!(writer, "    movl $-1, %edx")?;
                        writeln!(writer, "    cmpl $-1, %ecx")?;
                        writeln!(writer, "    cmovel %edx, %eax")?;
                        writeln!(writer, "    cltq")?;
                        writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;
                    }
                    _ => {
                        load_operand_to_rax(&args[0], writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, %rsi")?;
                        writeln!(writer, "    movq ${width}, %rdx")?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        let syscall = match target_os {
                            TargetOperatingSystem::MacOS => macos::SYS_WRITE,
                            _ => linux::SYS_WRITE,
                        };
                        writeln!(writer, "    movq ${syscall}, %rax")?;
                        writeln!(writer, "    syscall")?;
                    }
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let types = abi_extensions.argument_types(name, args.len());
                let locations = classify_arguments(&types, target_os);
                let stack_args: Vec<usize> = locations
                    .iter()
                    .enumerate()
                    .filter_map(|(index, location)| {
                        matches!(location, ArgLocation::Stack(_)).then_some(index)
                    })
                    .collect();
                let num_stack_args = stack_args.len();

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    // Rounded up so rsp stays 16-aligned at the call; the
                    // padding lands above the arguments, leaving their offsets
                    // from rsp unchanged.
                    let total_stack = (shadow_space + (num_stack_args * stack::SLOT_SIZE))
                        .div_ceil(stack::ALIGNMENT)
                        * stack::ALIGNMENT;
                    if total_stack > 0 {
                        writeln!(writer, "    subq ${total_stack}, %rsp")?;
                    }
                    for (slot, &index) in stack_args.iter().enumerate() {
                        let stack_offset = shadow_space + (slot * stack::SLOT_SIZE);
                        load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                        widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                        writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                    }
                } else {
                    // An odd number of pushes would leave rsp 8 off at the call.
                    if num_stack_args % 2 == 1 {
                        writeln!(writer, "    subq $8, %rsp")?;
                    }
                    for &index in stack_args.iter().rev() {
                        load_operand_to_rax(&args[index], writer, reg_alloc, stack_slots)?;
                        widen_external_argument(writer, abi_extensions, name, index, "rax")?;
                        writeln!(writer, "    pushq %rax")?;
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
                        widen_external_argument(
                            writer,
                            abi_extensions,
                            name,
                            index,
                            arg_regs[*gpr],
                        )?;
                    }
                }

                let mangled_name = abi.mangle_function_name(name);
                writeln!(writer, "    call {mangled_name}")?;

                if target_os == TargetOperatingSystem::Windows {
                    // Every Windows call gets home space, argument count aside:
                    // the callee may spill into it whether or not it was passed
                    // register arguments.
                    let shadow_space = windows::SHADOW_SPACE_SIZE as usize;
                    let total_stack = (shadow_space + (num_stack_args * stack::SLOT_SIZE))
                        .div_ceil(stack::ALIGNMENT)
                        * stack::ALIGNMENT;
                    if total_stack > 0 {
                        writeln!(writer, "    addq ${total_stack}, %rsp")?;
                    }
                } else if num_stack_args > 0 {
                    let pushed = num_stack_args * stack::SLOT_SIZE;
                    let padded = pushed.div_ceil(stack::ALIGNMENT) * stack::ALIGNMENT;
                    writeln!(writer, "    addq ${padded}, %rsp")?;
                }

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    if let Some(ty) = abi_extensions.float_return(name) {
                        writeln!(writer, "{}", move_xmm_to_rax(&ty, 0))?;
                    } else if let Some((width, signed)) = abi_extensions.return_extension(name) {
                        extend_integer(writer, "rax", width, signed)?;
                    }
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            }
        }
        MirInst::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            simd::operation(*op, *ty, dst, operands, writer, reg_alloc, stack_slots)?;
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
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_rax(vreg, writer, reg_alloc, stack_slots)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %rax", phys.name)?;
                    }
                }
                writeln!(writer, "    {}", load_insn_for(ty))?;
                if let Register::Virtual(vreg) = dst {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
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
            // Simple direct store for now
            load_operand_to_rax(src, writer, reg_alloc, stack_slots)?;
            if let AddressMode::BaseOffset {
                base, offset: 0, ..
            } = addr
            {
                let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
                match base {
                    Register::Virtual(vreg) => {
                        load_register_to_register(vreg, writer, reg_alloc, stack_slots, scratch)?;
                    }
                    Register::Physical(phys) => {
                        writeln!(writer, "    movq %{}, %{}", phys.name, scratch)?;
                    }
                }
                writeln!(
                    writer,
                    "    mov{} %{}, (%{scratch})",
                    size_suffix(ty),
                    rax_for(ty)
                )?;
                if scratch != "rbx" {
                    reg_alloc.free_scratch(scratch);
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "x86_64 backend does not support store address mode {addr:?}"
                )));
            }
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
            let types = abi_extensions.argument_types(func_name, args.len());
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
            let types = abi_extensions.argument_types(name, args.len());
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
                    def_regs,
                    settings,
                    debug_line,
                    abi_extensions,
                    next_label,
                    self_param_homes,
                    param_extensions,
                    entry_label,
                )?;
                restore_saved_registers(writer, stack_size, saved_regs)?;
                if has_frame {
                    X86Frame::generate_epilogue(writer, stack_size)?;
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

            restore_saved_registers(writer, stack_size, saved_regs)?;
            writeln!(writer, "    movq %rbp, %rsp")?;
            writeln!(writer, "    popq %rbp")?;
            let mangled_name = abi.mangle_function_name(name);
            writeln!(writer, "    jmp {mangled_name}")?;
        }
        MirInst::StackAddr { dst, slot } => {
            let offset = stack_slots.get(slot).ok_or_else(|| {
                LaminaError::ValidationError("stack address has no frame slot".into())
            })?;
            writeln!(writer, "    leaq {offset}(%rbp), %rax")?;
            if let Register::Virtual(vreg) = dst {
                store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
            }
        }
        MirInst::Lea { dst, base, offset } => {
            match base {
                Register::Virtual(vreg) => {
                    let is_placeholder = !def_regs.contains(vreg);

                    if is_placeholder {
                        if let Some(slot_offset) = stack_slots.get(vreg) {
                            let total_offset = *slot_offset as i64 + (*offset as i64);
                            if total_offset == 0 {
                                writeln!(writer, "    leaq (%rbp), %rax")?;
                            } else {
                                writeln!(writer, "    leaq {total_offset}(%rbp), %rax")?;
                            }
                        } else {
                            return Err(LaminaError::ValidationError(format!(
                                "LEA placeholder base vreg {vreg:?} has no stack slot"
                            )));
                        }
                    } else if let Some(phys) = reg_alloc.get_mapping_for(vreg) {
                        if *offset == 0 {
                            writeln!(writer, "    movq %{phys}, %rax")?;
                        } else {
                            writeln!(writer, "    leaq {offset}(%{phys}), %rax")?;
                        }
                    } else if let Some(slot_offset) = stack_slots.get(vreg) {
                        let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
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
            let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
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
        MirInst::Ret { value } => {
            if let Some(val) = value {
                match abi_extensions.float_return(func_name) {
                    Some(ty) => {
                        load_float_operand_to_xmm(val, writer, reg_alloc, stack_slots, "xmm0", &ty)?
                    }
                    None => load_operand_to_rax(val, writer, reg_alloc, stack_slots)?,
                }
            }
            // Epilogue
            restore_saved_registers(writer, stack_size, saved_regs)?;
            if has_frame {
                X86Frame::generate_epilogue(writer, stack_size)?;
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

/// Width suffix for a `mov` touching memory of this type.
pub(super) fn size_suffix(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "b",
        2 => "w",
        4 => "l",
        _ => "q",
    }
}

/// The sub-register of `rax` holding exactly the bytes of this type.
pub(super) fn rax_for(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
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
pub(super) fn emit_int_compare(
    op: &IntCmpOp,
    ty: &MirType,
    lhs: &Operand,
    rhs: &Operand,
    writer: &mut RaxTrackingWriter<impl Write>,
    reg_alloc: &mut X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<&'static str, LaminaError> {
    let unsigned = matches!(
        op,
        IntCmpOp::ULt | IntCmpOp::ULe | IntCmpOp::UGt | IntCmpOp::UGe
    );
    // Both sides already in registers at full width: compare them where they sit.
    if !unsigned || ty.size_bytes() == 8 {
        let physical = |value: &Operand| match value {
            Operand::Register(Register::Virtual(vreg)) => reg_alloc.get_mapping_for(vreg),
            _ => None,
        };
        if let (Some(left), Some(right)) = (physical(lhs), physical(rhs)) {
            writeln!(writer, "    cmpq %{right}, %{left}")?;
            return Ok("");
        }
    }
    load_operand_to_rax(lhs, writer, reg_alloc, stack_slots)?;
    if let Some(v) = cmp_immediate(rhs, ty, unsigned) {
        if unsigned {
            extend_integer(writer, "rax", ty.size_bytes(), false)?;
        }
        writeln!(writer, "    cmpq ${v}, %rax")?;
        return Ok("");
    }
    let scratch = reg_alloc.alloc_scratch().unwrap_or("rbx");
    load_operand_to_register(rhs, writer, reg_alloc, stack_slots, scratch)?;
    if unsigned {
        extend_integer(writer, "rax", ty.size_bytes(), false)?;
        extend_integer(writer, scratch, ty.size_bytes(), false)?;
    }
    writeln!(writer, "    cmpq %{scratch}, %rax")?;
    Ok(scratch)
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod narrow_integer_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{
        env,
        error::Error,
        fmt::Write as _,
        fs,
        process::{self, Command},
    };

    #[test]
    fn narrow_signed_operations_at_every_optimization_level() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "gcc", "timeout"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-narrow-{}", process::id()));
        fs::create_dir_all(&directory)?;
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
                    "zext.bool.i64".to_string()
                } else {
                    format!("sext.i{width}.i64")
                };
                writeln!(
                    functions,
                    "fn @op{id}(i{width} %a, i{width} %b) -> i64 {{\nentry:\n%r = {op}.i{width} %a, %b\n%z = {conversion} %r\nret.i64 %z\n}}"
                )?;
                writeln!(
                    body,
                    "%a{id} = add.i{width} {left}, 0\n%b{id} = add.i{width} {right}, 0\n%r{id} = {op}.i{width} %a{id}, %b{id}\n%z{id} = {conversion} %r{id}\nprint %z{id}\n%c{id} = call @op{id}(%a{id}, %b{id})\nprint %c{id}"
                )?;
                writeln!(expected, "{result}\n{result}")?;
                id += 1;
            }
            let maximum = (1_i64 << (width - 1)) - 1;
            let minimum = -(1_i64 << (width - 1));
            writeln!(
                body,
                "%overflow{width} = add.i{width} {maximum}, 1\n%negative{width} = shr.i{width} %overflow{width}, 3\n%wide{width} = sext.i{width}.i64 %negative{width}\nprint %wide{width}\n%unsigned{width} = zext.i{width}.i64 %overflow{width}\nprint %unsigned{width}"
            )?;
            writeln!(expected, "{}\n{}", minimum >> 3, 1_i64 << (width - 1))?;
            // Adjacent elements catch stores that overwrite beyond their type width.
            writeln!(
                body,
                "%array{width} = alloc.stack [2 x i{width}]\n%low{width} = getelem.ptr %array{width}, 0, i{width}\n%high{width} = getelem.ptr %array{width}, 1, i{width}\nstore.i{width} %high{width}, 99\nstore.i{width} %low{width}, -20\n%load{width} = load.i{width} %low{width}\n%div{width} = div.i{width} %load{width}, 3\n%loaded{width} = sext.i{width}.i64 %div{width}\nprint %loaded{width}\n%neighbor{width} = load.i{width} %high{width}\n%neighborwide{width} = sext.i{width}.i64 %neighbor{width}\nprint %neighborwide{width}"
            )?;
            writeln!(expected, "-6\n99")?;
        }
        let source = directory.join("narrow.lamina");
        fs::write(
            &source,
            format!("{functions}\n@export\nfn @main() -> i64 {{\nentry:\n{body}\nret.i64 0\n}}\n"),
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
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn narrow_unsigned_mir_operations_preserve_bit_patterns() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["cc", "timeout"]) {
            return Ok(());
        }
        use crate::mir::{
            FunctionBuilder, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Module, Operand,
            Register, VirtualReg,
        };
        use crate::mir_codegen::{
            MirCodegenSettings, RegallocStrategy,
            x86_64::generate_mir_x86_64_with_units_and_settings,
        };
        use lamina_platform::TargetOperatingSystem;

        let directory = env::temp_dir().join(format!("lamina-narrow-mir-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let mut main = FunctionBuilder::new("main")
            .returns(MirType::i64())
            .block("entry");
        let mut expected = String::new();
        let mut next = 0;
        let mut register = || {
            let reg = Register::Virtual(VirtualReg::gpr(next));
            next += 1;
            reg
        };
        for (width, ty, immediate) in [
            (8, MirType::i8(), Immediate::I8(-20)),
            (16, MirType::i16(), Immediate::I16(-20)),
            (32, MirType::i32(), Immediate::I32(-20)),
        ] {
            let unsigned = (1_i64 << width) - 20;
            for (binary, comparison, right, result) in [
                (Some(IntBinOp::UDiv), None, 3, unsigned / 3),
                (Some(IntBinOp::URem), None, 3, unsigned % 3),
                (Some(IntBinOp::LShr), None, 3, unsigned >> 3),
                (Some(IntBinOp::UDiv), None, -3, 0),
                (Some(IntBinOp::URem), None, -3, unsigned),
                (Some(IntBinOp::LShr), None, 0, unsigned),
                (None, Some(IntCmpOp::ULt), 3, 0),
                (None, Some(IntCmpOp::ULe), 3, 0),
                (None, Some(IntCmpOp::UGt), 3, 1),
                (None, Some(IntCmpOp::UGe), 3, 1),
            ] {
                let lhs = register();
                let rhs = register();
                let dst = register();
                let wide = register();
                let imm3 = match width {
                    8 => Immediate::I8(right as i8),
                    16 => Immediate::I16(right as i16),
                    _ => Immediate::I32(right),
                };
                main = main
                    .instr(Instruction::Copy {
                        ty,
                        dst: lhs.clone(),
                        src: Operand::Immediate(immediate),
                    })
                    .instr(Instruction::Copy {
                        ty,
                        dst: rhs.clone(),
                        src: Operand::Immediate(imm3),
                    });
                let operation = if let Some(op) = binary {
                    Instruction::IntBinary {
                        op,
                        ty,
                        dst: dst.clone(),
                        lhs: Operand::Register(lhs),
                        rhs: Operand::Register(rhs),
                    }
                } else {
                    Instruction::IntCmp {
                        op: comparison.ok_or("missing comparison")?,
                        ty,
                        dst: dst.clone(),
                        lhs: Operand::Register(lhs),
                        rhs: Operand::Register(rhs),
                    }
                };
                main = main
                    .instr(operation)
                    .instr(Instruction::IntConvert {
                        from: if binary.is_some() { ty } else { MirType::i1() },
                        to: MirType::i64(),
                        signed: false,
                        dst: wide.clone(),
                        src: Operand::Register(dst),
                    })
                    .instr(Instruction::Call {
                        name: "print".into(),
                        args: vec![Operand::Register(wide)],
                        ret: None,
                    });
                writeln!(expected, "{result}")?;
            }
        }
        main = main.instr(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        });
        let mut module = Module::new("unsigned");
        module.add_function(main.build());
        for regalloc in [
            RegallocStrategy::Incremental,
            RegallocStrategy::LinearScanGlobal,
            RegallocStrategy::GraphColorGlobal,
        ] {
            let settings = MirCodegenSettings {
                regalloc,
                ..MirCodegenSettings::default()
            };
            let mut assembly = Vec::new();
            generate_mir_x86_64_with_units_and_settings(
                &module,
                &mut assembly,
                TargetOperatingSystem::Linux,
                1,
                &settings,
            )?;
            let source = directory.join("unsigned.s");
            let executable = directory.join("unsigned");
            fs::write(&source, &assembly)?;
            let compiled = Command::new("cc")
                .arg(&source)
                .args(["-no-pie", "-o"])
                .arg(&executable)
                .output()?;
            assert!(compiled.status.success(), "{regalloc:?}: {compiled:?}");
            let output = Command::new("timeout")
                .arg("10")
                .arg(&executable)
                .output()?;
            assert!(output.status.success(), "{regalloc:?}: {output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                expected,
                "{regalloc:?}"
            );
        }
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
