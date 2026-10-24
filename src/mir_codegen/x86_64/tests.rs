use super::*;
use crate::{
    mir::{FunctionBuilder, ScalarType, codegen::from_ir},
    parser::parse_module,
};
use std::{error::Error as StdError, slice};

#[test]
fn rotated_loops_with_shared_exit_branch_back_conditionally() {
    let ir = parse_module(
        r#"
fn @two(i64 %i, i64 %n) -> i64 {
entry:
    %pick = eq.i64 %i, 0
    br %pick, a, b
a:
    %stop_a = ge.i64 %i, %n
    br %stop_a, exit, body_a
body_a:
    %i = add.i64 %i, 1
    jmp a
b:
    %stop_b = ge.i64 %i, %n
    br %stop_b, exit, body_b
body_b:
    %i = add.i64 %i, 1
    jmp b
exit:
    ret.i64 %i
}
"#,
    )
    .unwrap();
    let mir = from_ir(&ir, "shared_exit").unwrap();
    let settings = MirCodegenSettings {
        regalloc: RegallocStrategy::LinearScanGlobal,
        ..Default::default()
    };
    let asm = String::from_utf8(
        compile_single_function_x86_64(
            "two",
            mir.get_function("two").unwrap(),
            TargetOperatingSystem::Linux,
            &settings,
            &Default::default(),
        )
        .unwrap(),
    )
    .unwrap();
    for target in ["body_a", "body_b"] {
        assert!(asm.contains(&format!("jl .L_two.{target}")), "{asm}");
        assert!(!asm.contains(&format!("jmp .L_two.{target}")), "{asm}");
    }
}

#[test]
fn self_tail_loop_rotates_with_initial_condition_check() {
    let i = VirtualReg::gpr(0);
    let n = VirtualReg::gpr(1);
    let cond = VirtualReg::gpr(2);
    let next = VirtualReg::gpr(3);
    let func = FunctionBuilder::new("rotate")
        .param(i.into(), MirType::i64())
        .param(n.into(), MirType::i64())
        .returns(MirType::i64())
        .block("entry")
        .instr(MirInst::IntCmp {
            op: IntCmpOp::SGt,
            dst: cond.into(),
            lhs: Operand::Register(i.into()),
            rhs: Operand::Register(n.into()),
            ty: MirType::i64(),
        })
        .instr(MirInst::Br {
            cond: cond.into(),
            true_target: "exit".into(),
            false_target: "body".into(),
        })
        .block("body")
        .instr(MirInst::IntBinary {
            op: IntBinOp::Add,
            dst: next.into(),
            lhs: Operand::Register(i.into()),
            rhs: Operand::Immediate(Immediate::I64(1)),
            ty: MirType::i64(),
        })
        .instr(MirInst::TailCall {
            name: "rotate".into(),
            args: vec![Operand::Register(next.into()), Operand::Register(n.into())],
        })
        .block("exit")
        .instr(MirInst::Ret {
            value: Some(Operand::Register(i.into())),
        })
        .build();
    for strategy in [
        RegallocStrategy::Incremental,
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc: strategy,
            ..Default::default()
        };
        let asm = String::from_utf8(
            compile_single_function_x86_64(
                "rotate",
                &func,
                TargetOperatingSystem::Linux,
                &settings,
                &Default::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let body = asm.find(".L_rotate.body:").unwrap();
        let header = asm.find(".L_rotate.entry:").unwrap();
        assert!(body < header, "{asm}");
        assert_eq!(asm.matches("jmp .L_rotate.entry").count(), 1, "{asm}");
        assert!(asm[..body].contains("jmp .L_rotate.entry"), "{asm}");
        assert!(asm[header..].contains("jle .L_rotate.body"), "{asm}");
    }
}

#[test]
fn lea_selection_respects_flags_uses_and_encoding() {
    let regs: Vec<_> = (0..5).map(VirtualReg::gpr).collect();
    let plan = regs
        .iter()
        .zip(["r8", "r9", "r10", "r11", "r12"])
        .map(|(&v, r)| (v, MirAllocation::Register(r)))
        .collect();
    let alloc = X64RegAlloc::from_global_plan(TargetOperatingSystem::Linux, true, &plan);
    let operand = |i: usize| Operand::Register(Register::Virtual(regs[i]));
    let binary = |op, dst: usize, lhs, rhs| MirInst::IntBinary {
        op,
        dst: Register::Virtual(regs[dst]),
        lhs,
        rhs,
        ty: MirType::i64(),
    };
    let imm = |v| Operand::Immediate(Immediate::I64(v));
    let add = binary(IntBinOp::Add, 2, operand(0), operand(1));
    assert_eq!(
        select_lea(slice::from_ref(&add), &alloc, &HashMap::new(), |_| {
            false
        }),
        Some((1, "0(%r8,%r9,1)".into(), "r10"))
    );
    assert!(select_lea(&[add], &alloc, &HashMap::new(), |_| true).is_none());
    for displacement in [i64::from(i32::MIN), i64::from(i32::MAX)] {
        let add = binary(IntBinOp::Add, 2, operand(0), imm(displacement));
        assert!(select_lea(&[add], &alloc, &HashMap::new(), |_| false).is_some());
    }
    let add = binary(IntBinOp::Add, 2, operand(0), imm(i64::from(i32::MAX) + 1));
    assert!(select_lea(&[add], &alloc, &HashMap::new(), |_| false).is_none());
    let instructions = [
        binary(IntBinOp::Shl, 2, operand(1), imm(3)),
        binary(IntBinOp::Add, 3, operand(0), operand(2)),
        binary(IntBinOp::Add, 4, operand(3), imm(24)),
    ];
    let mut uses = HashMap::from([(regs[2], 1), (regs[3], 1)]);
    assert_eq!(
        select_lea(&instructions, &alloc, &uses, |_| false),
        Some((3, "24(%r8,%r9,8)".into(), "r12"))
    );
    uses.insert(regs[2], 2);
    assert!(select_lea(&instructions, &alloc, &uses, |_| false).is_none());
    let add = binary(IntBinOp::Add, 0, operand(0), imm(1));
    assert!(select_lea(&[add], &alloc, &uses, |_| false).is_none());
}

#[test]
fn global_frame_keeps_only_dense_storage() {
    let spilled = VirtualReg::gpr(0);
    let array = VirtualReg::gpr(1);
    let mapped = VirtualReg::gpr(2);
    let mut func = FunctionBuilder::new("dense")
        .block("entry")
        .instr(MirInst::Copy {
            ty: MirType::i64(),
            dst: Register::Virtual(spilled),
            src: Operand::Immediate(Immediate::I64(1)),
        })
        .instr(MirInst::StackAddr {
            dst: Register::Virtual(mapped),
            slot: array,
        })
        .build();
    func.stack_reservations.insert(array, 3);
    let plan = HashMap::from([
        (spilled, MirAllocation::Spill(-80)),
        (mapped, MirAllocation::Register("r12")),
    ]);
    let (slots, _, total) = layout_global_stack_slots(&func, &plan);
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[&spilled], -8);
    assert_eq!(slots[&array], -40);
    assert_eq!(total, 5);
}

#[test]
fn global_leaf_omits_empty_frame_but_keeps_stack_argument_base() {
    for strategy in [
        RegallocStrategy::LinearScanGlobal,
        RegallocStrategy::GraphColorGlobal,
    ] {
        let settings = MirCodegenSettings {
            regalloc: strategy,
            ..Default::default()
        };
        for (source, frame) in [
            ("fn @f() -> i64 {\nentry:\nret.i64 42\n}", false),
            (
                "fn @f(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g) -> i64 {\nentry:\nret.i64 %g\n}",
                true,
            ),
            (
                "fn @f() -> ptr {\nentry:\n%p = alloc.stack [24 x i8]\nret.ptr %p\n}",
                true,
            ),
        ] {
            let ir = parse_module(source).expect("parse frame case");
            let mir = from_ir(&ir, "frame").expect("lower frame case");
            let function = mir.get_function("f").expect("frame function");
            let assembly = compile_single_function_x86_64(
                "f",
                function,
                TargetOperatingSystem::Linux,
                &settings,
                &Default::default(),
            )
            .expect("emit frame case");
            let assembly = String::from_utf8(assembly).expect("assembly text");
            assert_eq!(assembly.contains("pushq %rbp"), frame, "{assembly}");
            assert_eq!(assembly.contains("popq %rbp"), frame, "{assembly}");
            if !frame {
                assert!(
                    !assembly.contains("%rsp") && !assembly.contains("%rbp"),
                    "{assembly}"
                );
            }
        }
    }
}

#[test]
fn parameter_cycle_preserves_both_sources() {
    let mut writer = RaxTrackingWriter {
        inner: Vec::new(),
        rax_value: None,
    };
    emit_parallel_copy(
        &[
            ("%rdi".into(), "%rsi".into()),
            ("%rsi".into(), "%rdi".into()),
        ],
        &mut writer,
    )
    .expect("emit parameter swap");
    assert_eq!(
        String::from_utf8(writer.inner).expect("assembly text"),
        "    movq %rdi, %rax\n    movq %rsi, %rdi\n    movq %rax, %rsi\n"
    );
}

#[test]
fn global_frame_keeps_unreferenced_reservations() {
    let mut func = Function::new(Signature::new("reserved"));
    let slot = VirtualReg::gpr(10);
    func.stack_reservations.insert(slot, 2);
    let plan = HashMap::from([(slot, MirAllocation::Register("r12"))]);
    let (slots, _, total) = layout_global_stack_slots(&func, &plan);
    assert_eq!(slots[&slot], -24);
    assert_eq!(total, 3);
}

#[test]
fn rax_round_trip_requires_no_intervening_emission() {
    let reg = VirtualReg::gpr(0);
    let other = VirtualReg::gpr(1);
    let slots = HashMap::from([(reg, -8), (other, -16)]);
    let alloc = X64RegAlloc::new(TargetOperatingSystem::Linux);
    for emission in [
        "    addq $1, %rax\n",
        "    sete %al\n",
        "    call unknown\n",
        "    movq %rcx, (%rdx)\n",
        "    jmp .L_join\n",
        ".L_join:\n",
    ] {
        let mut writer = RaxTrackingWriter {
            inner: Vec::new(),
            rax_value: None,
        };
        store_rax_to_register(&reg, &mut writer, &alloc, &slots).unwrap();
        let stored_len = writer.inner.len();
        load_operand_to_rax(
            &Operand::Register(Register::Virtual(reg)),
            &mut writer,
            &alloc,
            &slots,
        )
        .unwrap();
        assert_eq!(writer.inner.len(), stored_len);
        writer.write_all(emission.as_bytes()).unwrap();
        assert_eq!(writer.rax_value, None);
        let invalidated_len = writer.inner.len();
        load_register_to_rax(&reg, &mut writer, &alloc, &slots).unwrap();
        assert!(writer.inner.len() > invalidated_len);

        store_rax_to_register(&reg, &mut writer, &alloc, &slots).unwrap();
        load_register_to_rax(&other, &mut writer, &alloc, &slots).unwrap();
        assert_eq!(writer.rax_value, None);
    }
}

#[test]
fn float_copy_emits_bit_move_not_add() {
    let ty = MirType::Scalar(ScalarType::F64);
    let dst = Register::Virtual(VirtualReg::fpr(0));
    let function = FunctionBuilder::new("copy_float")
        .block("entry")
        .instr(MirInst::Copy {
            ty,
            dst: dst.clone(),
            src: Operand::Immediate(Immediate::F64(-0.0)),
        })
        .instr(MirInst::Ret { value: None })
        .build();
    let mut module = MirModule::new("copy_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_x86_64(&module, &mut output, TargetOperatingSystem::Linux)
        .expect("x86_64 codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(assembly.contains("movq"), "expected move: {assembly}");
    assert!(
        !assembly.contains("addsd"),
        "unexpected float add: {assembly}"
    );
}

#[test]
fn stack_copy_cycle_preserves_saved_value() -> Result<(), Box<dyn StdError>> {
    let pairs = [
        ("-8(%rbp)".to_string(), "-16(%rbp)".to_string()),
        ("-16(%rbp)".to_string(), "-24(%rbp)".to_string()),
        ("-24(%rbp)".to_string(), "-8(%rbp)".to_string()),
    ];
    let mut writer = RaxTrackingWriter {
        inner: Vec::new(),
        rax_value: None,
    };
    emit_parallel_copy(&pairs, &mut writer)?;
    let asm = String::from_utf8(writer.inner)?;
    let mut values = HashMap::from([
        ("-8(%rbp)", 11),
        ("-16(%rbp)", 22),
        ("-24(%rbp)", 33),
        ("%rax", 0),
    ]);
    let mut stack = Vec::new();
    for line in asm.lines() {
        let (op, operands) = line.trim().split_once(' ').ok_or("missing operands")?;
        match op {
            "movq" => {
                let (src, dst) = operands.split_once(", ").ok_or("missing destination")?;
                values.insert(dst, values[src]);
            }
            "pushq" => stack.push(values[operands]),
            "popq" => {
                values.insert(operands, stack.pop().ok_or("empty stack")?);
            }
            _ => panic!("unexpected instruction: {line}"),
        }
    }
    assert_eq!(
        (values["-8(%rbp)"], values["-16(%rbp)"], values["-24(%rbp)"]),
        (33, 11, 22),
        "{asm}"
    );
    assert!(stack.is_empty());
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod c_caller_abi_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{
        env,
        error::Error,
        fs,
        process::{self, Command},
    };

    const LAMINA: &str = r#"
@export fn @i8_is_negative(i8 %x) -> i64 {
entry:
 %c = lt.i8 %x, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @i16_is_negative(i16 %x) -> i64 {
entry:
 %c = lt.i16 %x, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @i32_is_negative(i32 %x) -> i64 {
entry:
 %c = lt.i32 %x, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @seventh_i8(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i8 %x) -> i64 {
entry:
 %c2 = lt.i8 %x, 0
 br %c2, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @u8_below_200(u8 %x) -> i64 {
entry:
 %c = lt.u8 %x, 200
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @i8_widen(i8 %x) -> i64 {
entry:
 %w = sext.i8.i64 %x
 ret.i64 %w
}
@export fn @i32_widen(i32 %x) -> i64 {
entry:
 %w = sext.i32.i64 %x
 ret.i64 %w
}
@extern fn @dirty_i8() -> i8 {
entry:
 ret.i8 0
}
@extern fn @dirty_u8() -> u8 {
entry:
 ret.u8 0
}
@extern fn @dirty_i16() -> i16 {
entry:
 ret.i16 0
}
@extern fn @dirty_i32() -> i32 {
entry:
 ret.i32 0
}
@export fn @call_dirty_i8() -> i64 {
entry:
 %r = call @dirty_i8()
 %c = lt.i8 %r, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @call_dirty_u8() -> i64 {
entry:
 %r = call @dirty_u8()
 %c = lt.u8 %r, 200
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @call_dirty_i16() -> i64 {
entry:
 %r = call @dirty_i16()
 %c = lt.i16 %r, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
@export fn @call_dirty_i32() -> i64 {
entry:
 %r = call @dirty_i32()
 %c = lt.i32 %r, 0
 br %c, yes, no
yes:
 ret.i64 1
no:
 ret.i64 0
}
"#;

    const CALLER: &str = r#"
#include <stdio.h>
__asm__(".text\n.globl dirty_i8\n.type dirty_i8,@function\ndirty_i8:\n movl $255, %eax\n ret\n");
__asm__(".text\n.globl dirty_u8\n.type dirty_u8,@function\ndirty_u8:\n movl $255, %eax\n ret\n");
__asm__(".text\n.globl dirty_i16\n.type dirty_i16,@function\ndirty_i16:\n movl $65535, %eax\n ret\n");
__asm__(".text\n.globl dirty_i32\n.type dirty_i32,@function\ndirty_i32:\n movl $4294967295, %eax\n ret\n");
extern long i8_is_negative(long);
extern long i16_is_negative(long);
extern long i32_is_negative(long);
extern long seventh_i8(long, long, long, long, long, long, long);
extern long u8_below_200(long);
extern long i8_widen(long);
extern long i32_widen(long);
extern long call_dirty_i8(void);
extern long call_dirty_u8(void);
extern long call_dirty_i16(void);
extern long call_dirty_i32(void);
int main(void) {
    printf("%ld %ld %ld\n", i8_is_negative(255), i8_is_negative(-1), i8_is_negative(256));
    printf("%ld %ld %ld\n", i16_is_negative(65535), i16_is_negative(32767), i16_is_negative(65536));
    printf("%ld %ld %ld\n", i32_is_negative(4294967295L), i32_is_negative(2147483647L), i32_is_negative(4294967296L));
    printf("%ld %ld %ld\n", u8_below_200(255), u8_below_200(199), u8_below_200(455));
    printf("%ld %ld\n", i8_widen(255), i32_widen(4294967295L));
    printf("%ld\n", seventh_i8(0, 0, 0, 0, 0, 0, 255));
    printf("%ld %ld %ld %ld\n", call_dirty_i8(), call_dirty_u8(), call_dirty_i16(), call_dirty_i32());
    return 0;
}
"#;

    const EXPECTED: &str = "1 1 0\n1 0 0\n1 0 0\n0 1 1\n-1 -1\n1\n1 0 1 1\n";

    /// Names every `@export` function so the C caller can link against them.
    const EXPORTS: &[&str] = &[
        "i8_is_negative",
        "i16_is_negative",
        "i32_is_negative",
        "seventh_i8",
        "u8_below_200",
        "i8_widen",
        "i32_widen",
        "call_dirty_i8",
        "call_dirty_u8",
        "call_dirty_i16",
        "call_dirty_i32",
    ];

    #[test]
    fn narrow_integers_cross_the_c_abi_in_both_directions() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["cc"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-c-abi-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("narrow.lamina");
        fs::write(&source, LAMINA)?;
        let base = directory.join("narrow");
        compile(&source, &base, &["--emit-asm"])?;
        let mut assembly = fs::read_to_string(directory.join("narrow.s"))?;
        for name in EXPORTS {
            assembly.push_str(&format!(".globl {name}\n"));
        }
        let assembly_path = directory.join("narrow-linked.s");
        fs::write(&assembly_path, assembly)?;
        let caller = directory.join("caller.c");
        fs::write(&caller, CALLER)?;
        let executable = directory.join("narrow-c");
        let linked = Command::new("cc")
            .arg("-no-pie")
            .arg(&caller)
            .arg(&assembly_path)
            .arg("-o")
            .arg(&executable)
            .output()?;
        assert!(linked.status.success(), "link failed: {linked:?}");
        let output = Command::new(&executable).output()?;
        assert_eq!(String::from_utf8_lossy(&output.stdout), EXPECTED);
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
