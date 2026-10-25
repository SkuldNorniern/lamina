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
fn @plain_helper() -> i64 {
entry:
 ret.i64 7
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
        let assembly_path = directory.join("narrow.s");
        let assembly = fs::read_to_string(&assembly_path)?;
        assert!(!assembly.contains(".globl plain_helper"), "{assembly}");
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

#[cfg(all(test, target_os = "linux"))]
mod debug_line_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{
        env,
        error::Error,
        fs,
        process::{self, Command},
    };

    const SOURCE: &str = r#"
fn @double(i64 %x) -> i64 {
entry:
 %r = mul.i64 %x, 2
 ret.i64 %r
}
fn @triple(i64 %x) -> i64 {
entry:
 %r = mul.i64 %x, 3
 ret.i64 %r
}
fn @main() -> i64 {
entry:
 %a = call @double(10)
 %b = call @triple(10)
 print %a
 print %b
 ret.i64 0
}
"#;

    #[test]
    fn debug_lines_assemble_across_functions() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "clang", "gcc", "timeout"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-debug-lines-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("two_functions.lamina");
        fs::write(&source, SOURCE)?;

        let assembly = directory.join("two_functions");
        compile(&source, &assembly, &["-g", "--emit-asm"])?;
        let text = fs::read_to_string(directory.join("two_functions.s"))?;
        let file_entries = text
            .lines()
            .filter(|line| line.trim_start().starts_with(".file 1 "))
            .count();
        assert_eq!(file_entries, 1, "{text}");
        assert!(
            text.contains(&format!(".file 1 \"{}\"", source.display())),
            "{text}"
        );

        for assembler in ["gas", "clang"] {
            let executable = directory.join(format!("two_functions-{assembler}"));
            compile(&source, &executable, &["-g", "--assembler", assembler])?;
            let output = Command::new("timeout")
                .arg("10")
                .arg(&executable)
                .output()?;
            assert!(output.status.success(), "{assembler}: {output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "20\n30\n",
                "{assembler}"
            );
        }

        let refused = directory.join("two_functions-ras");
        let error = compile(&source, &refused, &["-g", "--assembler", "ras"])
            .expect_err("ras must refuse .loc");
        assert!(error.to_string().contains(".loc"), "{error}");
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod c_abi_harness {
    use crate::mir_codegen::test_support::compile;
    use std::{error::Error, fs, path::Path, process::Command};

    /// Compiles `lamina`, links it with `c_source` through `cc`, and returns the
    /// executable's stdout.
    pub(super) fn build_and_run(
        directory: &Path,
        name: &str,
        lamina: &str,
        exports: &[&str],
        c_source: &str,
    ) -> Result<String, Box<dyn Error>> {
        let source = directory.join(format!("{name}.lamina"));
        fs::write(&source, lamina)?;
        let base = directory.join(name);
        compile(&source, &base, &["--emit-asm"])?;
        let mut assembly = fs::read_to_string(directory.join(format!("{name}.s")))?;
        for export in exports {
            assembly.push_str(&format!(".globl {export}\n"));
        }
        let assembly_path = directory.join(format!("{name}-linked.s"));
        fs::write(&assembly_path, assembly)?;
        let caller = directory.join(format!("{name}-caller.c"));
        fs::write(&caller, c_source)?;
        let executable = directory.join(format!("{name}-bin"));
        let linked = Command::new("cc")
            .arg("-no-pie")
            .arg(&caller)
            .arg(&assembly_path)
            .arg("-o")
            .arg(&executable)
            .output()?;
        assert!(linked.status.success(), "link failed: {linked:?}");
        let output = Command::new(&executable).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod narrow_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use std::{env, error::Error, fs, process};

    /// Each `raw_*` callee returns the full 64-bit register it received, so the caller
    /// sees exactly the upper bits lamina left behind.
    const RAW_CALLEES: &str = r#"
#include <stdio.h>
__asm__(".text\n.globl raw_u8\n.type raw_u8,@function\nraw_u8:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_i8\n.type raw_i8,@function\nraw_i8:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_u16\n.type raw_u16,@function\nraw_u16:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_i16\n.type raw_i16,@function\nraw_i16:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_u32\n.type raw_u32,@function\nraw_u32:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_i32\n.type raw_i32,@function\nraw_i32:\n movq %rdi, %rax\n ret\n");
__asm__(".text\n.globl raw_ret_u8\n.type raw_ret_u8,@function\nraw_ret_u8:\n movl $511, %eax\n ret\n");
__asm__(".text\n.globl raw_ret_i8\n.type raw_ret_i8,@function\nraw_ret_i8:\n movl $255, %eax\n ret\n");
__asm__(".text\n.globl raw_ret_u16\n.type raw_ret_u16,@function\nraw_ret_u16:\n movl $131071, %eax\n ret\n");
__asm__(".text\n.globl raw_ret_i16\n.type raw_ret_i16,@function\nraw_ret_i16:\n movl $32768, %eax\n ret\n");
__asm__(".text\n.globl raw_ret_u32\n.type raw_ret_u32,@function\nraw_ret_u32:\n movq $-1, %rax\n ret\n");
__asm__(".text\n.globl raw_ret_i32\n.type raw_ret_i32,@function\nraw_ret_i32:\n movq $4294967295, %rax\n ret\n");
extern long probe_u8(void);
extern long probe_i8(void);
extern long probe_u16(void);
extern long probe_i16(void);
extern long probe_u32(void);
extern long probe_i32(void);
extern long probe_ret_u8(void);
extern long probe_ret_i8(void);
extern long probe_ret_u16(void);
extern long probe_ret_i16(void);
extern long probe_ret_u32(void);
extern long probe_ret_i32(void);
int main(void) {
    printf("%ld %ld %ld %ld %ld %ld\n", probe_u8(), probe_i8(), probe_u16(), probe_i16(), probe_u32(), probe_i32());
    printf("%ld %ld %ld %ld %ld %ld\n", probe_ret_u8(), probe_ret_i8(), probe_ret_u16(), probe_ret_i16(), probe_ret_u32(), probe_ret_i32());
    return 0;
}
"#;

    /// Each `probe_*` materializes a narrow constant and passes it to a raw callee.
    const OUTGOING: &str = r#"
@extern fn @raw_u8(u8 %x) -> i64 {
entry:
 ret.i64 0
}
@extern fn @raw_i8(i8 %x) -> i64 {
entry:
 ret.i64 0
}
@extern fn @raw_u16(u16 %x) -> i64 {
entry:
 ret.i64 0
}
@extern fn @raw_i16(i16 %x) -> i64 {
entry:
 ret.i64 0
}
@extern fn @raw_u32(u32 %x) -> i64 {
entry:
 ret.i64 0
}
@extern fn @raw_i32(i32 %x) -> i64 {
entry:
 ret.i64 0
}
@export fn @probe_u8() -> i64 {
entry:
 %v = add.u8 255, 0
 %r = call @raw_u8(%v)
 ret.i64 %r
}
@export fn @probe_i8() -> i64 {
entry:
 %v = add.i8 -1, 0
 %r = call @raw_i8(%v)
 ret.i64 %r
}
@export fn @probe_u16() -> i64 {
entry:
 %v = add.u16 65535, 0
 %r = call @raw_u16(%v)
 ret.i64 %r
}
@export fn @probe_i16() -> i64 {
entry:
 %v = add.i16 -1, 0
 %r = call @raw_i16(%v)
 ret.i64 %r
}
@export fn @probe_u32() -> i64 {
entry:
 %v = add.u32 4294967295, 0
 %r = call @raw_u32(%v)
 ret.i64 %r
}
@export fn @probe_i32() -> i64 {
entry:
 %v = add.i32 -1, 0
 %r = call @raw_i32(%v)
 ret.i64 %r
}
@extern fn @raw_ret_u8() -> u8 {
entry:
 ret.u8 0
}
@extern fn @raw_ret_i8() -> i8 {
entry:
 ret.i8 0
}
@extern fn @raw_ret_u16() -> u16 {
entry:
 ret.u16 0
}
@extern fn @raw_ret_i16() -> i16 {
entry:
 ret.i16 0
}
@extern fn @raw_ret_u32() -> u32 {
entry:
 ret.u32 0
}
@extern fn @raw_ret_i32() -> i32 {
entry:
 ret.i32 0
}
@export fn @probe_ret_u8() -> i64 {
entry:
 %r = call @raw_ret_u8()
 %w = zext.u8.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i8() -> i64 {
entry:
 %r = call @raw_ret_i8()
 %w = sext.i8.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_u16() -> i64 {
entry:
 %r = call @raw_ret_u16()
 %w = zext.u16.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i16() -> i64 {
entry:
 %r = call @raw_ret_i16()
 %w = sext.i16.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_u32() -> i64 {
entry:
 %r = call @raw_ret_u32()
 %w = zext.u32.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i32() -> i64 {
entry:
 %r = call @raw_ret_i32()
 %w = sext.i32.i64 %r
 ret.i64 %w
}
"#;

    /// Each `echo_*` receives a narrow argument from a C caller that uses the real
    /// C prototype, and returns it widened by lamina.
    const INCOMING: &str = r#"
@export fn @echo_u8(u8 %x) -> i64 {
entry:
 %w = zext.u8.i64 %x
 ret.i64 %w
}
@export fn @echo_i8(i8 %x) -> i64 {
entry:
 %w = sext.i8.i64 %x
 ret.i64 %w
}
@export fn @echo_u16(u16 %x) -> i64 {
entry:
 %w = zext.u16.i64 %x
 ret.i64 %w
}
@export fn @echo_i16(i16 %x) -> i64 {
entry:
 %w = sext.i16.i64 %x
 ret.i64 %w
}
@export fn @echo_u32(u32 %x) -> i64 {
entry:
 %w = zext.u32.i64 %x
 ret.i64 %w
}
@export fn @echo_i32(i32 %x) -> i64 {
entry:
 %w = sext.i32.i64 %x
 ret.i64 %w
}
"#;

    const INCOMING_CALLER: &str = r#"
#include <stdio.h>
extern long echo_u8(unsigned char);
extern long echo_i8(signed char);
extern long echo_u16(unsigned short);
extern long echo_i16(short);
extern long echo_u32(unsigned int);
extern long echo_i32(int);
int main(void) {
    printf("%ld %ld\n", echo_u8(255), echo_i8(255));
    printf("%ld %ld\n", echo_u16(65535), echo_i16(65535));
    printf("%ld %ld\n", echo_u32(4294967295u), echo_i32(-1));
    return 0;
}
"#;

    const OUTGOING_EXPECTED: &str =
        "255 -1 65535 -1 4294967295 -1\n255 -1 65535 -32768 4294967295 -1\n";
    const INCOMING_EXPECTED: &str = "255 -1\n65535 -1\n4294967295 -1\n";

    #[test]
    fn narrow_arguments_and_results_cross_the_c_abi_widened() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["cc"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-narrow-abi-{}", process::id()));
        fs::create_dir_all(&directory)?;

        let outgoing = build_and_run(
            &directory,
            "outgoing",
            OUTGOING,
            &[
                "probe_u8",
                "probe_i8",
                "probe_u16",
                "probe_i16",
                "probe_u32",
                "probe_i32",
                "probe_ret_u8",
                "probe_ret_i8",
                "probe_ret_u16",
                "probe_ret_i16",
                "probe_ret_u32",
                "probe_ret_i32",
            ],
            RAW_CALLEES,
        )?;
        assert_eq!(outgoing, OUTGOING_EXPECTED);

        let incoming = build_and_run(
            &directory,
            "incoming",
            INCOMING,
            &[
                "echo_u8", "echo_i8", "echo_u16", "echo_i16", "echo_u32", "echo_i32",
            ],
            INCOMING_CALLER,
        )?;
        assert_eq!(incoming, INCOMING_EXPECTED);

        fs::remove_dir_all(directory)?;
        Ok(())
    }

    const PARAM_WIDTHS: &str = r#"
@export fn @p_u8(u8 %x) -> u8 { entry: ret.u8 %x }
@export fn @p_i8(i8 %x) -> i8 { entry: ret.i8 %x }
@export fn @p_char(char %x) -> char { entry: ret.char %x }
@export fn @p_u16(u16 %x) -> u16 { entry: ret.u16 %x }
@export fn @p_i16(i16 %x) -> i16 { entry: ret.i16 %x }
@export fn @p_u32(u32 %x) -> u32 { entry: ret.u32 %x }
@export fn @p_i32(i32 %x) -> i32 { entry: ret.i32 %x }
"#;

    #[test]
    fn narrow_parameter_prologue_extends_by_declared_signedness() -> Result<(), Box<dyn Error>> {
        let directory = env::temp_dir().join(format!("lamina-param-widths-{}", process::id()));
        fs::create_dir_all(&directory)?;
        let source = directory.join("widths.lamina");
        fs::write(&source, PARAM_WIDTHS)?;
        let base = directory.join("widths");
        compile(&source, &base, &["--emit-asm"])?;
        let assembly = fs::read_to_string(directory.join("widths.s"))?;

        for (function, instruction) in [
            ("p_u8", "movzbl"),
            ("p_i8", "movsbq"),
            ("p_char", "movsbq"),
            ("p_u16", "movzwl"),
            ("p_i16", "movswq"),
            ("p_u32", "movl"),
            ("p_i32", "movslq"),
        ] {
            let body = assembly
                .split(&format!("\n{function}:\n"))
                .nth(1)
                .unwrap_or_default();
            let prologue = body.split("    ret").next().unwrap_or_default();
            assert!(
                prologue
                    .lines()
                    .any(|line| line.trim_start().starts_with(instruction)),
                "{function} prologue missing {instruction}: {prologue}"
            );
        }

        fs::remove_dir_all(directory)?;
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod float_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::simd_support::tools_available;
    use std::{env, error::Error, fs, process};

    const OUTGOING: &str = r#"
@extern fn @c_mix(i64 %a, f64 %x, i64 %b, f32 %y, f64 %z) -> f64 {
entry:
 ret.f64 0.0
}
@extern fn @c_spill(i64 %i0, f64 %f0, i64 %i1, f64 %f1, i64 %i2, f64 %f2, i64 %i3, f32 %f3, i64 %i4, f64 %f4, i64 %i5, f64 %f5, i64 %i6, f64 %f6, f64 %f7, f64 %f8) -> f64 {
entry:
 ret.f64 0.0
}
@extern fn @c_f32(f32 %x, i64 %n) -> f32 {
entry:
 %z = add.f32 0.0, 0.0
 ret.f32 %z
}
@export fn @run_mix() -> f64 {
entry:
 %y = add.f32 2.25, 0.0
 %r = call @c_mix(3, 1.5, 4, %y, 0.5)
 ret.f64 %r
}
@export fn @run_spill() -> f64 {
entry:
 %f = add.f32 3.5, 0.0
 %r = call @c_spill(1, 0.5, 2, 1.5, 3, 2.5, 4, %f, 5, 4.5, 6, 5.5, 7, 6.5, 7.5, 8.5)
 ret.f64 %r
}
@export fn @run_f32() -> f32 {
entry:
 %x = add.f32 2.5, 0.0
 %r = call @c_f32(%x, 3)
 ret.f32 %r
}
"#;

    const OUTGOING_CALLER: &str = r#"
#include <stdio.h>
double c_mix(long a, double x, long b, float y, double z) { return a + x * b + y - z; }
double c_spill(long i0, double f0, long i1, double f1, long i2, double f2, long i3, float f3, long i4, double f4, long i5, double f5, long i6, double f6, double f7, double f8) {
    return i0 + f0 + 2*i1 + 2*f1 + 4*i2 + 4*f2 + 8*i3 + 8*f3 + 16*i4 + 16*f4 + 32*i5 + 32*f5 + 64*i6 + 64*f6 + 128*f7 + 256*f8;
}
float c_f32(float x, long n) { return x * n; }
extern double run_mix(void);
extern double run_spill(void);
extern float run_f32(void);
int main(void) {
    printf("%.6f %.6f\n", run_mix(), c_mix(3, 1.5, 4, 2.25, 0.5));
    printf("%.6f %.6f\n", run_spill(), c_spill(1, 0.5, 2, 1.5, 3, 2.5, 4, 3.5, 5, 4.5, 6, 5.5, 7, 6.5, 7.5, 8.5));
    printf("%.6f %.6f\n", (double)run_f32(), (double)c_f32(2.5f, 3));
    return 0;
}
"#;

    const INCOMING: &str = r#"
@export fn @l_mix(i64 %a, f64 %x, i64 %b, f32 %y, f64 %z) -> f64 {
entry:
 %p = mul.f64 %x, %z
 %c = lt.i64 %a, %b
 br %c, yes, no
yes:
 ret.f64 %p
no:
 ret.f64 %x
}
@export fn @l_f32(f32 %x, f32 %y) -> f32 {
entry:
 %s = add.f32 %x, %y
 ret.f32 %s
}
@export fn @l_spill(i64 %i0, f64 %f0, i64 %i1, f64 %f1, i64 %i2, f64 %f2, i64 %i3, f32 %f3, i64 %i4, f64 %f4, i64 %i5, f64 %f5, i64 %i6, f64 %f6, f64 %f7, f64 %f8) -> f64 {
entry:
 %s1 = mul.f64 %f7, %f8
 %s2 = add.f64 %s1, %f0
 %c = eq.i64 %i6, 7
 br %c, yes, no
yes:
 ret.f64 %s2
no:
 ret.f64 %f6
}
"#;

    const INCOMING_CALLER: &str = r#"
#include <stdio.h>
extern double l_mix(long a, double x, long b, float y, double z);
extern float l_f32(float x, float y);
extern double l_spill(long i0, double f0, long i1, double f1, long i2, double f2, long i3, float f3, long i4, double f4, long i5, double f5, long i6, double f6, double f7, double f8);
int main(void) {
    printf("%.6f %.6f\n", l_mix(1, 1.5, 2, 2.25f, 4.0), 1.5 * 4.0);
    printf("%.6f %.6f\n", l_mix(3, 1.5, 2, 2.25f, 4.0), 1.5);
    printf("%.6f %.6f\n", (double)l_f32(1.25f, 2.5f), 3.75);
    printf("%.6f %.6f\n", l_spill(1, 0.5, 2, 1.5, 3, 2.5, 4, 3.5f, 5, 4.5, 6, 5.5, 7, 6.5, 7.5, 8.5), 7.5 * 8.5 + 0.5);
    printf("%.6f %.6f\n", l_spill(1, 0.5, 2, 1.5, 3, 2.5, 4, 3.5f, 5, 4.5, 6, 5.5, 8, 6.5, 7.5, 8.5), 6.5);
    return 0;
}
"#;

    #[test]
    fn float_arguments_and_results_cross_the_c_abi_in_both_directions() -> Result<(), Box<dyn Error>>
    {
        if !tools_available(&["cc"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-float-abi-{}", process::id()));
        fs::create_dir_all(&directory)?;

        let outgoing = build_and_run(
            &directory,
            "outgoing",
            OUTGOING,
            &["run_mix", "run_spill", "run_f32"],
            OUTGOING_CALLER,
        )?;
        assert_eq!(
            outgoing,
            "10.750000 10.750000\n4610.500000 4610.500000\n7.500000 7.500000\n"
        );

        let incoming = build_and_run(
            &directory,
            "incoming",
            INCOMING,
            &["l_mix", "l_f32", "l_spill"],
            INCOMING_CALLER,
        )?;
        assert_eq!(
            incoming,
            "6.000000 6.000000\n1.500000 1.500000\n3.750000 3.750000\n64.250000 64.250000\n6.500000 6.500000\n"
        );

        fs::remove_dir_all(directory)?;
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod aggregate_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::simd_support::tools_available;
    use std::{env, error::Error, fs, process};

    /// Two words, one packed word, a padded twelve bytes, and an aggregate pushed past
    /// the last argument register.
    const INCOMING: &str = r#"
@export fn @take_pair(struct { a: i64, b: i64 } %p) -> i64 {
entry:
 %ap = getfieldptr %p, 0
 %bp = getfieldptr %p, 1
 %a = load.i64 %ap
 %b = load.i64 %bp
 %s = add.i64 %a, %b
 ret.i64 %s
}
@export fn @take_small(struct { a: i32, b: i32 } %p) -> i64 {
entry:
 %ap = getfieldptr %p, 0
 %bp = getfieldptr %p, 1
 %a = load.i32 %ap
 %b = load.i32 %bp
 %s = add.i32 %a, %b
 %w = sext.i32.i64 %s
 ret.i64 %w
}
@export fn @take_three(struct { a: i32, b: i32, c: i32 } %p) -> i64 {
entry:
 %ap = getfieldptr %p, 0
 %bp = getfieldptr %p, 1
 %cp = getfieldptr %p, 2
 %a = load.i32 %ap
 %b = load.i32 %bp
 %c = load.i32 %cp
 %ab = add.i32 %a, %b
 %s = add.i32 %ab, %c
 %w = sext.i32.i64 %s
 ret.i64 %w
}
@export fn @take_spilled(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, struct { a: i64, b: i64 } %p) -> i64 {
entry:
 %ab = add.i64 %a, %b
 %cd = add.i64 %c, %d
 %ef = add.i64 %e, %f
 %abcd = add.i64 %ab, %cd
 %scalars = add.i64 %abcd, %ef
 %ap = getfieldptr %p, 0
 %bp = getfieldptr %p, 1
 %x = load.i64 %ap
 %y = load.i64 %bp
 %xy = add.i64 %x, %y
 %total = add.i64 %scalars, %xy
 ret.i64 %total
}
"#;

    const INCOMING_CALLER: &str = r#"
#include <stdio.h>
struct Pair { long a, b; };
struct Small { int a, b; };
struct Three { int a, b, c; };
extern long take_pair(struct Pair);
extern long take_small(struct Small);
extern long take_three(struct Three);
extern long take_spilled(long, long, long, long, long, long, struct Pair);
int main(void) {
    struct Pair pair = {17, 25};
    struct Small small = {3, 4};
    struct Three three = {1, 2, 3};
    printf("%ld %ld %ld %ld\n", take_pair(pair), take_small(small), take_three(three),
           take_spilled(1, 2, 3, 4, 5, 6, pair));
    return 0;
}
"#;

    const OUTGOING: &str = r#"
@extern fn @c_pair(struct { a: i64, b: i64 } %p) -> i64 {
entry:
 ret.i64 0
}
@extern fn @c_spilled(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, struct { a: i64, b: i64 } %p) -> i64 {
entry:
 ret.i64 0
}
@export fn @send_pair() -> i64 {
entry:
 %q = alloc.stack struct { a: i64, b: i64 }
 %qa = getfieldptr %q, 0
 %qb = getfieldptr %q, 1
 store.i64 %qa, 100
 store.i64 %qb, 7
 %r = call @c_pair(%q)
 ret.i64 %r
}
@export fn @send_spilled() -> i64 {
entry:
 %q = alloc.stack struct { a: i64, b: i64 }
 %qa = getfieldptr %q, 0
 %qb = getfieldptr %q, 1
 store.i64 %qa, 100
 store.i64 %qb, 7
 %r = call @c_spilled(1, 2, 3, 4, 5, 6, %q)
 ret.i64 %r
}
"#;

    const OUTGOING_CALLEES: &str = r#"
#include <stdio.h>
struct Pair { long a, b; };
long c_pair(struct Pair p) { return p.a + p.b; }
long c_spilled(long a, long b, long c, long d, long e, long f, struct Pair p) {
    return a + b + c + d + e + f + p.a * 1000 + p.b;
}
extern long send_pair(void);
extern long send_spilled(void);
int main(void) {
    printf("%ld %ld\n", send_pair(), send_spilled());
    return 0;
}
"#;

    /// A lamina callee building a result for a C caller, and a lamina caller reading one
    /// back from C.
    const RETURNS: &str = r#"
@export fn @make_pair(i64 %x) -> struct { a: i64, b: i64 } {
entry:
 %p = alloc.stack struct { a: i64, b: i64 }
 %pa = getfieldptr %p, 0
 %pb = getfieldptr %p, 1
 store.i64 %pa, %x
 %y = add.i64 %x, 1
 store.i64 %pb, %y
 ret.ptr %p
}
@export fn @make_small(i32 %x) -> struct { a: i32, b: i32 } {
entry:
 %p = alloc.stack struct { a: i32, b: i32 }
 %pa = getfieldptr %p, 0
 %pb = getfieldptr %p, 1
 store.i32 %pa, %x
 store.i32 %pb, 9
 ret.ptr %p
}
@extern fn @c_make(i64 %x) -> struct { a: i64, b: i64 } {
entry:
 %p = alloc.stack struct { a: i64, b: i64 }
 ret.ptr %p
}
@export fn @take_c() -> i64 {
entry:
 %r = call @c_make(10)
 %ra = getfieldptr %r, 0
 %rb = getfieldptr %r, 1
 %a = load.i64 %ra
 %b = load.i64 %rb
 %s = add.i64 %a, %b
 ret.i64 %s
}
"#;

    const RETURNS_CALLER: &str = r#"
#include <stdio.h>
struct Pair { long a, b; };
struct Small { int a, b; };
extern struct Pair make_pair(long);
extern struct Small make_small(int);
extern long take_c(void);
struct Pair c_make(long x) { struct Pair p = {x * 3, x * 5}; return p; }
int main(void) {
    struct Pair pair = make_pair(41);
    struct Small small = make_small(5);
    printf("%ld %ld %d %d %ld\n", pair.a, pair.b, small.a, small.b, take_c());
    return 0;
}
"#;

    #[test]
    fn integer_aggregates_cross_the_c_abi_in_both_directions() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["cc"]) {
            return Ok(());
        }
        let directory = env::temp_dir().join(format!("lamina-aggregate-abi-{}", process::id()));
        fs::create_dir_all(&directory)?;

        let incoming = build_and_run(
            &directory,
            "incoming",
            INCOMING,
            &["take_pair", "take_small", "take_three", "take_spilled"],
            INCOMING_CALLER,
        )?;
        assert_eq!(incoming, "42 7 6 63\n");

        let outgoing = build_and_run(
            &directory,
            "outgoing",
            OUTGOING,
            &["send_pair", "send_spilled"],
            OUTGOING_CALLEES,
        )?;
        assert_eq!(outgoing, "107 100028\n");

        let returns = build_and_run(
            &directory,
            "returns",
            RETURNS,
            &["make_pair", "make_small", "take_c"],
            RETURNS_CALLER,
        )?;
        assert_eq!(returns, "41 42 5 9 80\n");

        fs::remove_dir_all(directory)?;
        Ok(())
    }
}
