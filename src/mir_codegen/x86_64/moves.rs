use super::*;

pub struct RaxTrackingWriter<W> {
    pub(super) inner: W,
    pub(super) rax_value: Option<VirtualReg>,
}

impl<W: Write> Write for RaxTrackingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.rax_value = None;
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// True for `movq %r, %r`. Several lowering paths pick the same register for source and
/// destination, and the move that results does nothing.
pub(super) fn is_self_move(line: &str) -> bool {
    let Some(operands) = line.trim_start().strip_prefix("movq ") else {
        return false;
    };
    matches!(operands.split_once(", "), Some((src, dst)) if src.starts_with('%') && src == dst)
}

/// Drop `movq %r, %r` from finished assembly.
pub(super) fn strip_self_moves(asm: Vec<u8>) -> Vec<u8> {
    let Ok(text) = String::from_utf8(asm) else {
        return Vec::new();
    };
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !is_self_move(line.trim_end_matches('\n')) {
            out.push_str(line);
        }
    }
    out.into_bytes()
}

/// Emit a parallel copy into `dsts`, honouring the fact that a source may itself be one
/// of the destinations. Identity moves are dropped; a cycle is broken through rax.
pub(super) fn emit_parallel_copy(
    pairs: &[(String, String)],
    writer: &mut RaxTrackingWriter<impl Write>,
) -> Result<(), LaminaError> {
    for (src, dst) in parallel_copy_schedule(pairs) {
        let location = |location| match location {
            CopyLocation::Value(value) => value,
            CopyLocation::Scratch => "%rax".to_string(),
        };
        let (src, dst) = (location(src), location(dst));
        if src.ends_with("(%rbp)") && dst.ends_with("(%rbp)") {
            writeln!(writer, "    movq {src}, %rax")?;
            writeln!(writer, "    movq %rax, {dst}")?;
        } else {
            writeln!(writer, "    movq {src}, {dst}")?;
        }
    }
    Ok(())
}

pub(super) fn load_insn_for(ty: &MirType) -> &'static str {
    match ty.size_bytes() {
        1 => "movsbq (%rax), %rax",
        2 => "movswq (%rax), %rax",
        4 => "movslq (%rax), %rax",
        _ => "movq (%rax), %rax",
    }
}

pub(super) fn load_copy_operand_to_rax(
    ty: &MirType,
    src: &Operand,
    writer: &mut impl Write,
    reg_alloc: &X64RegAlloc,
    stack_slots: &HashMap<VirtualReg, i32>,
) -> Result<(), LaminaError> {
    let size = ty.size_bytes();
    match src {
        Operand::Register(Register::Virtual(vreg)) => {
            if let Some(physical) = reg_alloc.get_mapping_for(vreg) {
                emit_copy_register_load(writer, physical, size)?;
            } else if let Some(offset) = stack_slots.get(vreg) {
                match size {
                    1 => writeln!(writer, "    movsbq {offset}(%rbp), %rax")?,
                    2 => writeln!(writer, "    movswq {offset}(%rbp), %rax")?,
                    4 => writeln!(writer, "    movslq {offset}(%rbp), %rax")?,
                    _ => writeln!(writer, "    movq {offset}(%rbp), %rax")?,
                }
            } else {
                return Err(LaminaError::ValidationError(format!(
                    "Virtual register {vreg:?} has no mapping or stack slot"
                )));
            }
        }
        Operand::Register(Register::Physical(physical)) => {
            emit_copy_register_load(writer, physical.name, size)?;
        }
        Operand::Immediate(immediate) => {
            let bits = match immediate {
                Immediate::I8(value) => *value as i64 as u64,
                Immediate::I16(value) => *value as i64 as u64,
                Immediate::I32(value) => *value as i64 as u64,
                Immediate::I64(value) => *value as u64,
                Immediate::F32(value) => u64::from(value.to_bits()),
                Immediate::F64(value) => value.to_bits(),
            };
            let shift = 64 - size * 8;
            let value = ((bits << shift) as i64) >> shift;
            writeln!(writer, "    movq ${value}, %rax")?;
        }
    }
    Ok(())
}

pub(super) fn emit_copy_register_load(
    writer: &mut impl Write,
    register: &str,
    size: usize,
) -> Result<(), LaminaError> {
    let narrowed = register_for_size(register, size).ok_or_else(|| {
        LaminaError::ValidationError(format!(
            "x86_64 backend cannot select a {size}-byte alias for register {register}"
        ))
    })?;
    match size {
        1 => writeln!(writer, "    movsbq %{narrowed}, %rax")?,
        2 => writeln!(writer, "    movswq %{narrowed}, %rax")?,
        4 => writeln!(writer, "    movslq %{narrowed}, %rax")?,
        _ => writeln!(writer, "    movq %{narrowed}, %rax")?,
    }
    Ok(())
}
