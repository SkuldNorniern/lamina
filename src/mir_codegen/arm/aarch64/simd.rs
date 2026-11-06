use super::*;
use crate::mir::{Instruction, VectorLane, VectorOp};
use crate::mir_codegen::simd::{index, lane, scalar_bits, unsupported, vector_slots, width};
use lamina_platform::PointerWidth;

/// First and last of v16-v31, which no call preserves and nothing else here touches.
const FIRST_HOME: u8 = 16;
const LAST_HOME: u8 = 31;

/// Vector registers that can live in v16-v31 instead of their frame slots: every mention
/// sits in one block, and that block runs nothing that could reach a call. A block that
/// needs more than the sixteen homes leaves the rest in their slots.
pub(super) fn vector_homes(func: &Function) -> HashMap<Register, u8> {
    let vectors: HashSet<Register> = vector_slots(func)
        .into_iter()
        .map(|(vreg, _)| Register::Virtual(vreg))
        .collect();
    let mut owner: HashMap<&Register, Option<usize>> = HashMap::new();
    for (index, block) in func.blocks.iter().enumerate() {
        let quiet = block.instructions.iter().all(|i| stays_inline(&i.instr));
        for inst in &block.instructions {
            let vector_inst = matches!(&inst.instr, Instruction::VectorOp { .. })
                || matches!(&inst.instr, Instruction::Load { ty, .. } | Instruction::Store { ty, .. } if ty.is_vector());
            for reg in inst.def_reg().into_iter().chain(inst.use_regs()) {
                if vectors.contains(reg) {
                    let here = (quiet && vector_inst).then_some(index);
                    owner
                        .entry(reg)
                        .and_modify(|seen| {
                            if *seen != here {
                                *seen = None;
                            }
                        })
                        .or_insert(here);
                }
            }
        }
    }
    let mut homes = HashMap::new();
    let mut next = vec![FIRST_HOME; func.blocks.len()];
    let mut owned: Vec<_> = owner
        .into_iter()
        .filter_map(|(reg, block)| Some((block?, reg)))
        .collect();
    owned.sort_by_key(|(block, reg)| (*block, format!("{reg:?}")));
    for (block, reg) in owned {
        if next[block] <= LAST_HOME {
            homes.insert(reg.clone(), next[block]);
            next[block] += 1;
        }
    }
    homes
}

/// Whether this instruction always stays inline, so v16-v31 survive it.
fn stays_inline(inst: &Instruction) -> bool {
    match inst {
        Instruction::IntBinary { ty, .. } | Instruction::IntCmp { ty, .. } => {
            matches!(ty, MirType::Scalar(_)) && ty.size_bytes(PointerWidth::Bits64) <= 8
        }
        Instruction::Load { .. }
        | Instruction::Store { .. }
        | Instruction::VectorOp { .. }
        | Instruction::Lea { .. }
        | Instruction::Jmp { .. }
        | Instruction::Br { .. }
        | Instruction::Ret { .. } => true,
        _ => false,
    }
}

fn vector_slot(frame: &FrameMap, reg: &Register) -> Result<i32, LaminaError> {
    frame
        .slot_of(reg)
        .ok_or_else(|| unsupported("vector requires virtual register stack slot"))
}

fn load_vector(
    w: &mut impl Write,
    operand: &Operand,
    v: u8,
    frame: &FrameMap,
    scratch: &str,
    homes: &HashMap<Register, u8>,
) -> Result<(), LaminaError> {
    let Operand::Register(reg) = operand else {
        return Err(unsupported("vector operand requires register"));
    };
    if let Some(home) = homes.get(reg) {
        writeln!(w, "    mov v{v}.16b, v{home}.16b")?;
        return Ok(());
    }
    emit_frame_slot_address(w, scratch, vector_slot(frame, reg)?)?;
    writeln!(w, "    ldr q{v}, [{scratch}]")?;
    Ok(())
}

/// The register holding `operand`: its home, or `scratch` after loading its slot there.
fn vector_operand(
    w: &mut impl Write,
    operand: &Operand,
    scratch: u8,
    frame: &FrameMap,
    address: &str,
    homes: &HashMap<Register, u8>,
) -> Result<u8, LaminaError> {
    if let Operand::Register(reg) = operand
        && let Some(home) = homes.get(reg)
    {
        return Ok(*home);
    }
    load_vector(w, operand, scratch, frame, address, homes)?;
    Ok(scratch)
}

/// Stores v`v` to where `dst` lives, unless it is already in `dst`'s home.
fn store_vector(
    w: &mut impl Write,
    dst: &Register,
    v: u8,
    frame: &FrameMap,
    address: &str,
    homes: &HashMap<Register, u8>,
) -> Result<(), LaminaError> {
    if let Some(home) = homes.get(dst) {
        if *home != v {
            writeln!(w, "    mov v{home}.16b, v{v}.16b")?;
        }
        return Ok(());
    }
    emit_frame_slot_address(w, address, vector_slot(frame, dst)?)?;
    writeln!(w, "    str q{v}, [{address}]")?;
    Ok(())
}

pub(super) fn emit(
    inst: &Instruction,
    w: &mut impl Write,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    homes: &HashMap<Register, u8>,
) -> Result<(), LaminaError> {
    let a = ra
        .alloc_scratch()
        .ok_or_else(|| unsupported("NEON requires address scratch register"))?;
    let t = ra
        .alloc_scratch()
        .ok_or_else(|| unsupported("NEON requires scalar scratch register"))?;
    let result = emit_inner(inst, w, frame, ra, a, t, homes);
    ra.free_scratch(t);
    ra.free_scratch(a);
    result
}

fn emit_inner(
    inst: &Instruction,
    w: &mut impl Write,
    frame: &FrameMap,
    ra: &mut A64RegAlloc,
    a: &str,
    t: &str,
    homes: &HashMap<Register, u8>,
) -> Result<(), LaminaError> {
    match inst {
        Instruction::Load { ty, dst, addr, .. } => {
            lane(*ty)?;
            materialize_address(w, addr, a, frame, ra)?;
            let v = homes.get(dst).copied().unwrap_or(0);
            writeln!(w, "    ldr q{v}, [{a}]")?;
            store_vector(w, dst, v, frame, a, homes)?;
        }
        Instruction::Store { ty, src, addr, .. } => {
            lane(*ty)?;
            // The address scratch is free again before the address is built.
            let v = vector_operand(w, src, 0, frame, a, homes)?;
            materialize_address(w, addr, a, frame, ra)?;
            writeln!(w, "    str q{v}, [{a}]")?;
        }
        Instruction::VectorOp {
            op,
            ty,
            dst,
            operands,
        } => {
            use VectorOp::*;
            let lane = lane(*ty)?;
            let bits = width(lane);
            let (arr, element, scalar) = match bits {
                8 => ("16b", "b", w_alias(t)),
                16 => ("8h", "h", w_alias(t)),
                32 => ("4s", "s", w_alias(t)),
                _ => ("2d", "d", t.to_string()),
            };
            let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
            if (*op == VMul && lane == VectorLane::I64)
                || (float
                    && matches!(
                        op,
                        VShl | VLShr | VAShr | VShlLanes | VLShrLanes | VAShrLanes
                    ))
            {
                return Err(unsupported(format!(
                    "AArch64 unsupported {op:?} for {lane:?}"
                )));
            }
            if matches!(op, VExtractLane | VInsertLane) {
                let i = index(*op, lane, operands)?;
                load_vector(w, &operands[0], 0, frame, a, homes)?;
                if *op == VExtractLane {
                    writeln!(w, "    umov {scalar}, v0.{element}[{i}]")?;
                    store_result(w, dst, t, frame, ra)?;
                    return Ok(());
                }
                emit_materialize_operand(w, &scalar_bits(&operands[1]), t, frame, ra)?;
                writeln!(w, "    ins v0.{element}[{i}], {scalar}")?;
            } else if *op == VSplat {
                if operands.len() != 1 {
                    return Err(unsupported("VSplat requires one operand"));
                }
                emit_materialize_operand(w, &scalar_bits(&operands[0]), t, frame, ra)?;
                let v = homes.get(dst).copied().unwrap_or(0);
                writeln!(w, "    dup v{v}.{arr}, {scalar}")?;
                return store_vector(w, dst, v, frame, a, homes);
            } else {
                if operands.len() != 2 {
                    return Err(unsupported(format!("{op:?} requires two operands")));
                }
                // Everything but a shift only reads its inputs, so homes go in directly.
                if !matches!(
                    op,
                    VShl | VLShr | VAShr | VShlLanes | VLShrLanes | VAShrLanes
                ) {
                    let mnemonic = match (op, float) {
                        (VAnd, _) => "and",
                        (VOr, _) => "orr",
                        (VXor, _) => "eor",
                        (VAdd, false) => "add",
                        (VSub, false) => "sub",
                        (VMul, false) => "mul",
                        (VAdd, true) => "fadd",
                        (VSub, true) => "fsub",
                        (VMul, true) => "fmul",
                        _ => {
                            return Err(unsupported(format!(
                                "AArch64 unsupported {op:?} for {lane:?}"
                            )));
                        }
                    };
                    let layout = if matches!(op, VAnd | VOr | VXor) {
                        "16b"
                    } else {
                        arr
                    };
                    let l = vector_operand(w, &operands[0], 0, frame, a, homes)?;
                    let r = vector_operand(w, &operands[1], 1, frame, a, homes)?;
                    let v = homes.get(dst).copied().unwrap_or(0);
                    writeln!(
                        w,
                        "    {mnemonic} v{v}.{layout}, v{l}.{layout}, v{r}.{layout}"
                    )?;
                    return store_vector(w, dst, v, frame, a, homes);
                }
                load_vector(w, &operands[0], 0, frame, a, homes)?;
                if !matches!(op, VShl | VLShr | VAShr) {
                    load_vector(w, &operands[1], 1, frame, a, homes)?;
                }
                match op {
                    VShlLanes | VLShrLanes | VAShrLanes => {
                        writeln!(w, "    mov {t}, #{}", bits - 1)?;
                        writeln!(w, "    dup v2.{arr}, {scalar}")?;
                        writeln!(w, "    cmhs v3.{arr}, v2.{arr}, v1.{arr}")?;
                        if *op == VAShrLanes {
                            writeln!(w, "    bsl v3.16b, v1.16b, v2.16b")?;
                            writeln!(w, "    neg v3.{arr}, v3.{arr}")?;
                            writeln!(w, "    sshl v0.{arr}, v0.{arr}, v3.{arr}")?;
                        } else {
                            writeln!(w, "    and v1.16b, v1.16b, v3.16b")?;
                            if *op == VLShrLanes {
                                writeln!(w, "    neg v1.{arr}, v1.{arr}")?;
                            }
                            writeln!(w, "    ushl v0.{arr}, v0.{arr}, v1.{arr}")?;
                            writeln!(w, "    and v0.16b, v0.16b, v3.16b")?;
                        }
                    }
                    VShl | VLShr | VAShr => {
                        emit_materialize_operand(w, &scalar_bits(&operands[1]), t, frame, ra)?;
                        writeln!(w, "    mov {a}, #{bits}")?;
                        writeln!(w, "    cmp {t}, {a}")?;
                        writeln!(w, "    csel {t}, {t}, {a}, lo")?;
                        if *op != VShl {
                            writeln!(w, "    neg {t}, {t}")?;
                        }
                        writeln!(w, "    dup v1.{arr}, {scalar}")?;
                        let mnemonic = if *op == VAShr { "sshl" } else { "ushl" };
                        writeln!(w, "    {mnemonic} v0.{arr}, v0.{arr}, v1.{arr}")?;
                    }
                    _ => unreachable!("only shifts reach here"),
                }
            }
            store_vector(w, dst, 0, frame, a, homes)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use crate::mir::{VectorLane, VectorOp};
    use crate::mir_codegen::simd_support::*;
    use lamina_platform::PointerWidth;

    #[test]
    fn neon_assembles_every_supported_pair() {
        native(
            Backend::Neon,
            |op, lane, index| {
                use VectorOp::*;
                let bits = scalar(lane).size_bytes(PointerWidth::Bits64) * 8;
                let arr = match bits {
                    8 => "16b",
                    16 => "8h",
                    32 => "4s",
                    _ => "2d",
                };
                let elem = match bits {
                    8 => "b",
                    16 => "h",
                    32 => "s",
                    _ => "d",
                };
                let float = matches!(lane, VectorLane::F32 | VectorLane::F64);
                match op {
                    VAdd | VSub | VMul => format!(
                        "{}{} v0.{arr}, v0.{arr}, v1.{arr}",
                        if float { "f" } else { "" },
                        match op {
                            VAdd => "add",
                            VSub => "sub",
                            _ => "mul",
                        }
                    ),
                    VAnd | VOr | VXor => format!(
                        "{} v0.16b, v0.16b, v1.16b",
                        match op {
                            VAnd => "and",
                            VOr => "orr",
                            _ => "eor",
                        }
                    ),
                    VShl | VLShr | VAShr | VShlLanes | VLShrLanes => format!(
                        "{} v0.{arr}, v0.{arr}, v1.{arr}",
                        if op == VAShr { "sshl" } else { "ushl" }
                    ),
                    VAShrLanes => format!("sshl v0.{arr}, v0.{arr}, v3.{arr}"),
                    VSplat => format!("dup v0.{arr},"),
                    VExtractLane => format!("v0.{elem}[{index}]"),
                    VInsertLane => format!("ins v0.{elem}[{index}],"),
                }
            },
            PointerWidth::Bits64,
        );
    }

    fn compiled(source: &str) -> String {
        use crate::mir::codegen::from_ir_x86_64;
        use crate::mir_codegen::arm::aarch64::generate_mir_aarch64_with_units_and_settings;
        use crate::{MirCodegenSettings, parser::parse_module};
        use lamina_platform::TargetOperatingSystem;

        let mir = from_ir_x86_64(&parse_module(source).expect("parse"), "homes").expect("lower");
        let mut output = Vec::new();
        generate_mir_aarch64_with_units_and_settings(
            &mir,
            &mut output,
            TargetOperatingSystem::MacOS,
            1,
            &MirCodegenSettings::default(),
            PointerWidth::Bits64,
        )
        .expect("compile");
        String::from_utf8(output).expect("utf8")
    }

    const VECTOR_BODY: &str = "  %v = load.<4 x i32> %p
  %s = splat.<4 x i32> 3
  %z = add.<4 x i32> %v, %s
  store.<4 x i32> %p, %z
";

    #[test]
    fn a_vector_in_a_call_free_block_stays_in_a_register() {
        let asm = compiled(&format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n{VECTOR_BODY}  ret.i64 0\n}}\n"
        ));
        assert!(asm.contains("add v"), "{asm}");
        assert!(!asm.contains("add v0.4s"), "{asm}");
        assert!(!asm.contains("mov v"), "{asm}");
    }

    #[test]
    fn a_vector_beside_a_call_stays_in_its_slot() {
        let asm = compiled(&format!(
            "fn @g() -> void {{\nentry:\n  ret.void\n}}\n\
             fn @f(ptr %p) -> i64 {{\nentry:\n{VECTOR_BODY}  call @g()\n  ret.i64 0\n}}\n"
        ));
        for home in 16..=31 {
            assert!(!asm.contains(&format!("v{home}.")), "v{home}: {asm}");
            assert!(!asm.contains(&format!("q{home},")), "q{home}: {asm}");
        }
    }

    #[test]
    fn a_vector_used_in_two_blocks_stays_in_its_slot() {
        let asm = compiled(
            "fn @f(ptr %p) -> i64 {\nentry:\n  %v = load.<4 x i32> %p\n  jmp next\nnext:\n  store.<4 x i32> %p, %v\n  ret.i64 0\n}\n",
        );
        for home in 16..=31 {
            assert!(!asm.contains(&format!("q{home},")), "q{home}: {asm}");
        }
    }

    #[test]
    fn neon_rejects_unsupported_pairs() {
        negative(Backend::Neon, PointerWidth::Bits64);
    }
}
