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
    next_label: &str,
    self_param_homes: &[Option<String>],
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
        MirInst::Copy { ty, dst, src } => {
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
            } else if name == "writeptr" && args.len() == 1 {
                let arg = args.first().ok_or_else(|| {
                    LaminaError::ValidationError("writeptr requires one argument".to_string())
                })?;
                load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                writeln!(writer, "    movzbl (%rax), %ecx")?;
                writeln!(writer, "    subq ${}, %rsp", stack::ALIGNMENT)?;
                writeln!(writer, "    movb %cl, (%rsp)")?;

                match target_os {
                    TargetOperatingSystem::MacOS => {
                        writeln!(writer, "    movq ${}, %rax", macos::SYS_WRITE)?;
                        writeln!(writer, "    movq ${}, %rdi", fd::STDOUT)?;
                        writeln!(writer, "    movq %rsp, %rsi")?;
                        writeln!(writer, "    movq $1, %rdx")?;
                        writeln!(writer, "    syscall")?;
                    }
                    TargetOperatingSystem::Windows => {
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

                writeln!(writer, "    addq ${}, %rsp", stack::ALIGNMENT)?;

                if let Some(ret_reg) = ret
                    && let Register::Virtual(vreg) = ret_reg
                {
                    store_rax_to_register(vreg, writer, reg_alloc, stack_slots)?;
                }
            } else {
                let abi = X86ABI::new(target_os);
                let arg_regs = abi.arg_registers();
                let num_reg_args = args.len().min(arg_regs.len());
                let num_stack_args = args.len().saturating_sub(arg_regs.len());

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
                    for i in 0..num_stack_args {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        let stack_offset = shadow_space + (i * stack::SLOT_SIZE);
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    movq %rax, {stack_offset}(%rsp)")?;
                    }
                } else {
                    // An odd number of pushes would leave rsp 8 off at the call.
                    if num_stack_args % 2 == 1 {
                        writeln!(writer, "    subq $8, %rsp")?;
                    }
                    for i in (0..num_stack_args).rev() {
                        let arg_idx = num_reg_args + i;
                        let arg = &args[arg_idx];
                        load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;
                        writeln!(writer, "    pushq %rax")?;
                    }
                }

                for i in 0..num_reg_args {
                    let arg = &args[i];
                    let dest_reg = arg_regs[i];
                    load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
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
                if entry_label != next_label {
                    writeln!(writer, "    jmp .L_{func_name}.{entry_label}")?;
                }
                return Ok(());
            }
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            for (index, arg) in args.iter().enumerate().take(arg_regs.len()) {
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, arg_regs[index])?;
            }
            writeln!(writer, "    jmp {}", self_tail_entry_label(func_name))?;
        }
        MirInst::TailCall { name, args } => {
            let abi = X86ABI::new(target_os);
            let arg_regs = abi.arg_registers();
            let num_reg_args = args.len().min(arg_regs.len());
            let num_stack_args = args.len().saturating_sub(arg_regs.len());

            // 1. Handle Register Arguments
            for i in 0..num_reg_args {
                let arg = &args[i];
                let dest_reg = arg_regs[i];
                load_operand_to_register(arg, writer, reg_alloc, stack_slots, dest_reg)?;
            }

            // 2. Handle Stack Arguments (Overwrite incoming args)
            // Note: TailCallOptimization guarantees args.len() == current_func.args.len()
            // So we can strictly overwrite our own incoming stack slots.
            if num_stack_args > 0 {
                let shadow_space = if target_os == TargetOperatingSystem::Windows {
                    windows::SHADOW_SPACE_SIZE as usize
                } else {
                    0
                };

                for i in 0..num_stack_args {
                    let arg_idx = num_reg_args + i;
                    let arg = &args[arg_idx];

                    // Incoming args are at RBP + 16 + shadow + i*8
                    // (Return address is 8, saved RBP pushed, so RBP points to saved RBP)
                    // Wait:
                    // Standard Prologue: push rbp; mov rbp, rsp
                    // Stack:
                    //   [RBP + 16 + shadow + 8*i] -> Arg N (Stack Arg i)
                    //   [RBP + 8]  -> Return Address
                    //   [RBP + 0]  -> Saved RBP
                    //
                    // So first stack arg is at RBP + 16 (on Linux/Mac) or RBP + 16 + 32 (Windows? No shadow is allocated by caller?)
                    // Windows: Shadow space is 32 bytes allocated *by caller* right before return address?
                    // No, shadow space is allocated by caller *above* return address?
                    // Microsoft x64: "The caller allocates space for 4 register parameters..."
                    // Stack: [RetAddr] [Home P1] [Home P2] [Home P3] [Home P4] [Stack Arg 5] ...
                    // Wait, Home space is "Shadow Space". It's strictly for the first 4 args (which are in regs).
                    // Stack args start *after* shadow space.
                    // So RBP + 16 + 32 + i*8.

                    let offset = 16 + shadow_space + (i * stack::SLOT_SIZE);

                    // Load to RAX (scratch) first
                    load_operand_to_rax(arg, writer, reg_alloc, stack_slots)?;

                    // Store to incoming slot
                    writeln!(writer, "    movq %rax, {offset}(%rbp)")?;
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
                load_operand_to_rax(val, writer, reg_alloc, stack_slots)?;
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
