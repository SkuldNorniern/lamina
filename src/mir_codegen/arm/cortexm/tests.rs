use super::*;
use crate::mir::codegen::from_ir_for_target;
use crate::parser::parse_module;
use isa::{Isa, mov_imm};
use lamina_mir::ScalarType;
use lamina_platform::Target;

fn try_compile(arch: TargetArchitecture, source: &str) -> Result<String, String> {
    let module = parse_module(source).expect("parse");
    let mir = from_ir_for_target(&module, "cortexm_test", arch.data_layout())
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    generate_mir_cortexm_with_units_and_settings(
        &mir,
        &mut out,
        arch,
        TargetOperatingSystem::Linux,
        1,
        &MirCodegenSettings::default(),
    )
    .map_err(|e| e.to_string())?;
    Ok(String::from_utf8(out).expect("utf8"))
}

fn compile(arch: TargetArchitecture, source: &str) -> String {
    try_compile(arch, source).expect("codegen")
}

const BOOLEAN: &str = "fn @main(i32 %a, i32 %b) -> i32 {\n\
                       entry:\n  %c = lt.i32 %a, %b\n  %z = extend.bool.i32 %c\n  ret.i32 %z\n}\n";

/// Every `it` is followed by exactly one instruction, and that instruction carries the
/// condition the `it` named. A block that holds a second instruction, or whose
/// condition pattern does not match what follows, assembles on some assemblers and
/// does the wrong thing at run time, so the shape is asserted rather than assumed.
fn check_it_blocks(assembly: &str) -> usize {
    let lines: Vec<&str> = assembly.lines().map(str::trim).collect();
    let mut blocks = 0;
    for (index, line) in lines.iter().enumerate() {
        if line.starts_with("itt") || line.starts_with("ite") || line.starts_with("itet") {
            panic!("only the one-instruction `it` form is emitted, found `{line}`");
        }
        let Some(cond) = line.strip_prefix("it ") else {
            continue;
        };
        blocks += 1;
        let guarded = lines.get(index + 1).expect("an instruction after `it`");
        let (mnemonic, _) = guarded.split_once(' ').expect("operands");
        assert!(
            mnemonic.ends_with(cond),
            "`{guarded}` does not carry the condition `{cond}` its `it` named"
        );
        let after = lines.get(index + 2).copied().unwrap_or("");
        assert!(
            !after.starts_with("it "),
            "two `it` blocks in a row: {after}"
        );
    }
    blocks
}

#[test]
fn the_module_header_names_the_thumb_encoding() {
    let v7 = compile(TargetArchitecture::Thumbv7m, BOOLEAN);
    assert!(v7.contains(".arch armv7-m"), "{v7}");
    assert!(v7.contains(".thumb\n"), "{v7}");
    assert!(v7.contains(".thumb_func"), "{v7}");
    assert!(!v7.contains("@progbits"), "{v7}");
    let v6 = compile(TargetArchitecture::Thumbv6m, BOOLEAN);
    assert!(v6.contains(".arch armv6-m"), "{v6}");
    assert!(!v6.contains("\n.arm\n"), "{v6}");
}

#[test]
fn a_boolean_on_thumb2_sits_in_a_one_instruction_it_block() {
    let assembly = compile(TargetArchitecture::Thumbv7m, BOOLEAN);
    assert!(check_it_blocks(&assembly) > 0, "{assembly}");
    assert!(assembly.contains("it lt"), "{assembly}");
    assert!(assembly.contains("movlt "), "{assembly}");
}

#[test]
fn thumb1_has_no_it_block_at_all() {
    for source in [
        BOOLEAN,
        "fn @main(i32 %c, i32 %a, i32 %b) -> i32 {\n\
         entry:\n  %p = ne.i32 %c, 0\n  %r = select.i32 %p, %a, %b\n  ret.i32 %r\n}\n",
    ] {
        let assembly = compile(TargetArchitecture::Thumbv6m, source);
        assert_eq!(check_it_blocks(&assembly), 0, "{assembly}");
        for line in assembly.lines().map(str::trim) {
            assert!(!line.starts_with("it"), "{line}");
        }
    }
}

#[test]
fn thumb1_emits_no_instruction_armv6m_does_not_have() {
    let assembly = compile(
        TargetArchitecture::Thumbv6m,
        "fn @main(i32 %a, i32 %b) -> i32 {\n\
         entry:\n  %m = mul.i32 %a, %b\n  %o = or.i32 %m, %b\n  %s = shl.i32 %o, %a\n\
         \x20 %r = add.i32 %s, %b\n  ret.i32 %r\n}\n",
    );
    for forbidden in [
        "movw ", "movt ", " clz ", " rbit ", " sdiv ", " udiv ", " mls ", " cmn ",
    ] {
        assert!(!assembly.contains(forbidden), "{forbidden} in {assembly}");
    }
}

#[test]
fn thumb2_uses_the_instructions_armv7m_adds() {
    let assembly = compile(
        TargetArchitecture::Thumbv7m,
        "fn @main(i32 %a, i32 %b) -> i32 {\n\
         entry:\n  %d = div.i32 %a, %b\n  %r = rem.i32 %a, %b\n  %s = add.i32 %r, %d\n\
         \x20 ret.i32 %s\n}\n",
    );
    assert!(assembly.contains("sdiv "), "{assembly}");
    assert!(assembly.contains("mls "), "{assembly}");
    assert!(assembly.contains("movw "), "{assembly}");
}

#[test]
fn thumb1_refuses_division_by_name_rather_than_calling_a_runtime_it_does_not_ship() {
    let error = try_compile(
        TargetArchitecture::Thumbv6m,
        "fn @main(i32 %a, i32 %b) -> i32 {\nentry:\n  %d = div.i32 %a, %b\n  ret.i32 %d\n}\n",
    )
    .expect_err("armv6-m has no divide instruction");
    assert!(error.contains("thumbv6m"), "{error}");
    assert!(error.contains("__aeabi_idiv"), "{error}");
    assert!(error.contains("sdiv"), "{error}");
}

#[test]
fn floating_point_is_refused_by_name_on_both() {
    for (arch, name) in [
        (TargetArchitecture::Thumbv6m, "thumbv6m"),
        (TargetArchitecture::Thumbv7m, "thumbv7m"),
    ] {
        let error = try_compile(
            arch,
            "fn @half(f64 %x) -> f64 {\nentry:\n  %h = mul.f64 %x, 0.5\n  ret.f64 %h\n}\n\
             fn @main() -> i32 {\nentry:\n  ret.i32 0\n}\n",
        )
        .expect_err("no Cortex-M double precision");
        assert!(error.contains(name), "{error}");
        assert!(error.contains("floating point"), "{error}");
        assert!(error.contains("f64"), "{error}");
    }
}

#[test]
fn a_sixty_four_bit_integer_in_a_register_is_refused_by_name() {
    use crate::mir::{Block, Immediate, IntBinOp, Operand, Register, VirtualReg};
    let dst = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(Signature::new("wide"));
    let mut block = Block::new("entry");
    block.push(MirInst::IntBinary {
        op: IntBinOp::Mul,
        ty: MirType::Scalar(ScalarType::I64),
        dst,
        lhs: Operand::Immediate(Immediate::I64(3)),
        rhs: Operand::Immediate(Immediate::I64(4)),
    });
    block.push(MirInst::Ret { value: None });
    func.add_block(block);
    let mut module = MirModule::new("wide");
    module.functions.insert("wide".to_string(), func);

    let mut out = Vec::new();
    let error = generate_mir_cortexm_with_units_and_settings(
        &module,
        &mut out,
        TargetArchitecture::Thumbv7m,
        TargetOperatingSystem::Unknown,
        1,
        &MirCodegenSettings::default(),
    )
    .expect_err("a 64-bit multiply has no Cortex-M lowering");
    let message = error.to_string();
    assert!(message.contains("i64"), "{message}");
    assert!(message.contains("@wide"), "{message}");
}

#[test]
fn thumb1_builds_a_word_from_shifts_and_adds() {
    let cases: [(u32, &[&str]); 4] = [
        (7, &["movs r4, #7"]),
        (0x1000, &["movs r4, #1", "lsls r4, r4, #12"]),
        (0x1234, &["movs r4, #18", "lsls r4, r4, #8", "adds r4, #52"]),
        (
            0xdead_beef,
            &[
                "movs r4, #222",
                "lsls r4, r4, #8",
                "adds r4, #173",
                "lsls r4, r4, #8",
                "adds r4, #190",
                "lsls r4, r4, #8",
                "adds r4, #239",
            ],
        ),
    ];
    for (value, expected) in cases {
        let mut out = Vec::new();
        mov_imm(&mut out, Isa::Thumb1, "r4", value).expect("immediate");
        let text = String::from_utf8(out).expect("utf8");
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        assert_eq!(lines, expected, "{value:#x}");
    }
}

#[test]
fn the_frame_puts_the_outgoing_area_at_the_stack_pointer() {
    let layout = Layout::new(12, 32);
    assert_eq!(layout.outgoing, 16);
    assert_eq!(layout.locals, 32);
    // The five-register push leaves the stack pointer four bytes off eight, which the
    // frame's own padding word puts back.
    assert_eq!((SAVED_BYTES + layout.total) % 8, 0);
    assert_eq!(layout.slot(-8), 16 + 32 - 8);
    assert_eq!(layout.incoming(), layout.total + SAVED_BYTES);
}

#[test]
fn nothing_in_the_body_moves_the_stack_pointer() {
    let assembly = compile(
        TargetArchitecture::Thumbv7m,
        "fn @take(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) -> i32 {\n\
         entry:\n  ret.i32 %f\n}\n\
         fn @main() -> i32 {\nentry:\n  %r = call @take(1, 2, 3, 4, 5, 6)\n  ret.i32 %r\n}\n",
    );
    let adjustments = assembly
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("sub sp") || line.starts_with("add sp"))
        .count();
    // One per function prologue and one per epilogue, and nothing at the call site.
    assert_eq!(adjustments, 4, "{assembly}");
    assert!(assembly.contains("str r4, [sp, #0]"), "{assembly}");
}

#[test]
fn both_targets_are_thirty_two_bit_with_no_sixty_four_bit_registers() {
    for name in ["thumbv6m_unknown", "thumbv7m_unknown"] {
        let target: Target = name.parse().expect(name);
        assert_eq!(target.to_string(), name);
        assert!(!target.has_64bit_int_registers(), "{name}");
        assert_eq!(target.architecture.max_int_register_bits(), 32, "{name}");
        assert_eq!(target.stack_alignment_at_call(), 8, "{name}");
    }
    let m: Target = "armv7m_unknown".parse().expect("armv7m");
    assert_eq!(m.architecture, TargetArchitecture::Thumbv7m);
    assert_eq!(
        m.variant.armv7().map(|v| v.hardware_divide),
        Some(true),
        "the M profile has sdiv and udiv"
    );
    assert!(!m.environment.hardware_float(), "the backend emits no VFP");
}

#[test]
fn an_underaligned_access_is_refused_by_name_on_both() {
    for (arch, name) in [
        (TargetArchitecture::Thumbv6m, "thumbv6m"),
        (TargetArchitecture::Thumbv7m, "thumbv7m"),
    ] {
        for (source, operation) in [
            (
                "fn @main() -> i32 {\nentry:\n  %b = stackalloc.[16 x i8]\n  \
                 %p = getelem.i8 %b, 1\n  store.i32 %p, 7 align 1\n  ret.i32 0\n}\n",
                "store",
            ),
            (
                "fn @main() -> i32 {\nentry:\n  %b = stackalloc.[16 x i8]\n  \
                 %p = getelem.i8 %b, 1\n  %v = load.i32 %p align 1\n  ret.i32 0\n}\n",
                "load",
            ),
        ] {
            let error = try_compile(arch, source).expect_err("no unaligned access on Cortex-M");
            assert!(error.contains(name), "{error}");
            assert!(error.contains(operation), "{error}");
            assert!(error.contains("1-byte alignment"), "{error}");
        }
    }
}
