use super::*;
use crate::{
    mir::{
        FunctionBuilder, RelocationModel, ScalarType,
        codegen::{from_ir_for_target, from_ir_x86_64},
    },
    parser::parse_module,
};
use lamina_mir::Located;
use lamina_platform::{PointerWidth, Target, TargetArchitecture};
use std::{error::Error as StdError, slice};

#[test]
fn position_independent_modules_load_external_addresses_through_the_got() {
    let ir = parse_module(
        "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nfn @twice(i64 %x) -> i64 {\nentry:\n    ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n    %p = funcaddr @printf\n    %q = funcaddr @twice\n    ret.i64 0\n}\n",
    )
    .unwrap();
    let mut mir = from_ir_x86_64(&ir, "pie").unwrap();
    let assemble = |module: &MirModule| {
        let mut output = Vec::new();
        generate_mir_x86_64(module, &mut output, TargetOperatingSystem::Linux).unwrap();
        String::from_utf8(output).unwrap()
    };
    let plain = assemble(&mir);
    assert!(plain.contains("    leaq printf(%rip), "), "{plain}");
    assert!(!plain.contains("GOTPCREL"), "{plain}");

    mir.relocation = RelocationModel::Pie;
    let pie = assemble(&mir);
    assert!(pie.contains("    movq printf@GOTPCREL(%rip), "), "{pie}");
    assert!(pie.contains("    leaq twice(%rip), "), "{pie}");
}

#[test]
fn macos_external_function_addresses_load_through_the_got() {
    let ir = parse_module(
        "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nfn @twice(i64 %x) -> i64 {\nentry:\n    ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n    %p = funcaddr @printf\n    %q = funcaddr @twice\n    ret.i64 0\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "static").unwrap();
    let assemble = |os: TargetOperatingSystem| {
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, os).unwrap();
        String::from_utf8(output).unwrap()
    };
    let macos = assemble(TargetOperatingSystem::MacOS);
    assert!(
        macos.contains("    movq _printf@GOTPCREL(%rip), "),
        "{macos}"
    );
    assert!(!macos.contains("leaq _printf(%rip)"), "{macos}");
    assert!(macos.contains("    leaq _twice(%rip), "), "{macos}");
    assert!(!macos.contains("_twice@GOTPCREL"), "{macos}");

    let linux = assemble(TargetOperatingSystem::Linux);
    assert!(linux.contains("    leaq printf(%rip), "), "{linux}");
    assert!(!linux.contains("GOTPCREL"), "{linux}");
}

#[test]
fn calls_that_may_be_variadic_report_vector_count_in_al() {
    let ir = parse_module(
        "@extern\nfn @printf(ptr %fmt, ...) -> i32\n@extern\nfn @unknown(i64 %a, ...) -> i64\n\nglobal $fmt: [5 x i8] = \"%g\\n\"\n\nfn @known(i64 %a) -> i64 {\nentry:\n    ret.i64 %a\n}\nfn @main() -> i64 {\nentry:\n    %k = call @known(1)\n    %u = call @unknown(1)\n    %f = getelem.i8 $fmt, 0\n    %p = call @printf(%f, 2.5)\n    ret.i64 %k\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "al").unwrap();
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
    let assembly = String::from_utf8(output).unwrap();

    let before = |call: &str| -> String {
        let head = assembly.split(call).next().unwrap();
        head.lines().rev().take(3).collect::<Vec<_>>().join("\n")
    };
    assert!(
        before("    call unknown").contains("movl $0, %eax"),
        "{assembly}"
    );
    assert!(
        before("    call printf").contains("movl $1, %eax"),
        "{assembly}"
    );
    assert!(!before("    call known").contains("%eax"), "{assembly}");
}

fn assemble_for_windows(source: &str) -> Result<String, String> {
    let ir = parse_module(source).unwrap();
    let mir = from_ir_for_target(
        &ir,
        "win64",
        Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Windows).data_layout(),
    )
    .unwrap();
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Windows)
        .map_err(|error| error.to_string())?;
    Ok(String::from_utf8(output).unwrap())
}

const WIN64_VARIADIC_SUM: &str = "fn @sum(i64 %n, ...) -> i64 {\nentry:\n    %ap = stackalloc.valist\n    va_start %ap\n    %x = va_arg.i64 %ap\n    %y = va_arg.f64 %ap\n    va_end %ap\n    %yi = fptoint.f64.i64 %y\n    %r = add.i64 %n, %x\n    %s = add.i64 %r, %yi\n    ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n    %a = call @sum(1, 2, 2.5)\n    ret.i64 %a\n}\n";

#[test]
fn win64_variadic_callee_spills_all_four_argument_registers_into_home_space() {
    let assembly = assemble_for_windows(WIN64_VARIADIC_SUM).unwrap();
    let sum = assembly.split("\nsum:\n").nth(1).unwrap();
    for (offset, reg) in [(16, "rcx"), (24, "rdx"), (32, "r8"), (40, "r9")] {
        assert!(
            sum.contains(&format!("    movq %{reg}, {offset}(%rbp)")),
            "{sum}"
        );
    }
    assert!(!sum.contains("%al"), "{sum}");
    assert!(!sum.contains("movups"), "{sum}");
}

#[test]
fn win64_va_list_is_a_pointer_to_the_first_variadic_slot() {
    let assembly = assemble_for_windows(WIN64_VARIADIC_SUM).unwrap();
    let sum = assembly.split("\nsum:\n").nth(1).unwrap();
    assert!(
        sum.contains("    leaq 24(%rbp), %rax\n    movq %rax, (%rcx)"),
        "{sum}"
    );
    assert_eq!(sum.matches("    addq $8, (%rcx)").count(), 2, "{sum}");
    assert!(sum.contains("    movq (%rdx), %rax"), "{sum}");
}

#[test]
fn win64_variadic_call_passes_each_float_in_its_xmm_and_general_register() {
    let assembly = assemble_for_windows(WIN64_VARIADIC_SUM).unwrap();
    let main = assembly.split("\nsum:\n").next().unwrap();
    assert!(main.contains("    movq %xmm2, %r8\n    call sum"), "{main}");
    assert!(
        main.contains("    movq $1, %rcx\n    movq $2, %rdx"),
        "{main}"
    );
}

#[test]
fn win64_variadic_stack_arguments_follow_the_home_space() {
    let assembly = assemble_for_windows(
        "fn @count(i64 %n, ...) -> i64 {\nentry:\n    %ap = stackalloc.valist\n    va_start %ap\n    %x = va_arg.i64 %ap\n    va_end %ap\n    ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n    %a = call @count(0, 1, 2, 3, 4, 5)\n    ret.i64 %a\n}\n",
    )
    .unwrap();
    let main = assembly.split("\nmain:\n").nth(1).unwrap();
    assert!(
        main.contains("    movq $4, %rax\n    movq %rax, 32(%rsp)"),
        "{main}"
    );
    assert!(
        main.contains("    movq $5, %rax\n    movq %rax, 40(%rsp)"),
        "{main}"
    );
}

#[test]
fn win64_thread_local_uses_the_tls_index_sequence_and_secrel() {
    let asm = assemble_for_windows(
        "@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    %v = load.i64 $t\n    ret.i64 %v\n}\n",
    )
    .unwrap();
    assert!(asm.contains("\n.section .tls$,\"aw\"\n"), "{asm}");
    assert!(asm.contains("\n    .zero 8\n"), "{asm}");
    assert!(asm.contains("    movq %gs:0x58, %rax\n"), "{asm}");
    assert!(asm.contains("    movl _tls_index(%rip), %ecx\n"), "{asm}");
    assert!(asm.contains("    movq (%rax,%rcx,8), %rax\n"), "{asm}");
    assert!(asm.contains("    leaq t@SECREL32(%rax), %rax\n"), "{asm}");
    assert!(!asm.contains("_tls_index:"), "{asm}");
    assert!(!asm.contains("_tls_used"), "{asm}");
}

#[test]
fn win64_thread_local_alignment_above_eight_is_refused_before_any_code() {
    let ir = parse_module(
        "@align(16)\n@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    %v = load.i64 $t\n    ret.i64 %v\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "tls_align").unwrap();
    let mut output = Vec::new();
    let error = generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Windows)
        .expect_err("a 16-byte thread-local is refused on Windows")
        .to_string();
    assert!(error.contains("'t'"), "{error}");
    assert!(error.contains("16 bytes"), "{error}");
    assert!(output.is_empty());
}

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
    let mir = from_ir_x86_64(&ir, "shared_exit").unwrap();
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
        .param(i.into(), MirType::Scalar(ScalarType::I64))
        .param(n.into(), MirType::Scalar(ScalarType::I64))
        .returns(MirType::Scalar(ScalarType::I64))
        .block("entry")
        .instr(MirInst::IntCmp {
            op: IntCmpOp::SGt,
            dst: cond.into(),
            lhs: Operand::Register(i.into()),
            rhs: Operand::Register(n.into()),
            ty: MirType::Scalar(ScalarType::I64),
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
            ty: MirType::Scalar(ScalarType::I64),
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
    let binary = |op, dst: usize, lhs, rhs| {
        Located::from(MirInst::IntBinary {
            op,
            dst: Register::Virtual(regs[dst]),
            lhs,
            rhs,
            ty: MirType::Scalar(ScalarType::I64),
        })
    };
    let imm = |v| Operand::Immediate(Immediate::I64(v));
    let add = binary(IntBinOp::Add, 2, operand(0), operand(1));
    assert_eq!(
        select_lea(
            slice::from_ref(&add),
            &alloc,
            &HashMap::new(),
            |_| { false },
            PointerWidth::Bits64
        ),
        Some((1, "0(%r8,%r9,1)".into(), "r10"))
    );
    assert!(
        select_lea(
            &[add],
            &alloc,
            &HashMap::new(),
            |_| true,
            PointerWidth::Bits64
        )
        .is_none()
    );
    for displacement in [i64::from(i32::MIN), i64::from(i32::MAX)] {
        let add = binary(IntBinOp::Add, 2, operand(0), imm(displacement));
        assert!(
            select_lea(
                &[add],
                &alloc,
                &HashMap::new(),
                |_| false,
                PointerWidth::Bits64
            )
            .is_some()
        );
    }
    let add = binary(IntBinOp::Add, 2, operand(0), imm(i64::from(i32::MAX) + 1));
    assert!(
        select_lea(
            &[add],
            &alloc,
            &HashMap::new(),
            |_| false,
            PointerWidth::Bits64
        )
        .is_none()
    );
    let instructions = [
        binary(IntBinOp::Shl, 2, operand(1), imm(3)),
        binary(IntBinOp::Add, 3, operand(0), operand(2)),
        binary(IntBinOp::Add, 4, operand(3), imm(24)),
    ];
    let mut uses = HashMap::from([(regs[2], 1), (regs[3], 1)]);
    assert_eq!(
        select_lea(
            &instructions,
            &alloc,
            &uses,
            |_| false,
            PointerWidth::Bits64
        ),
        Some((3, "24(%r8,%r9,8)".into(), "r12"))
    );
    uses.insert(regs[2], 2);
    assert!(
        select_lea(
            &instructions,
            &alloc,
            &uses,
            |_| false,
            PointerWidth::Bits64
        )
        .is_none()
    );
    let add = binary(IntBinOp::Add, 0, operand(0), imm(1));
    assert!(select_lea(&[add], &alloc, &uses, |_| false, PointerWidth::Bits64).is_none());
}

#[test]
fn global_frame_keeps_only_dense_storage() {
    let spilled = VirtualReg::gpr(0);
    let array = VirtualReg::gpr(1);
    let mapped = VirtualReg::gpr(2);
    let mut func = FunctionBuilder::new("dense")
        .block("entry")
        .instr(MirInst::Copy {
            ty: MirType::Scalar(ScalarType::I64),
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
                "fn @f() -> ptr {\nentry:\n%p = stackalloc.[24 x i8]\nret.ptr %p\n}",
                true,
            ),
        ] {
            let ir = parse_module(source).expect("parse frame case");
            let mir = from_ir_x86_64(&ir, "frame").expect("lower frame case");
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
            let code: String = assembly
                .lines()
                .filter(|line| !line.trim_start().starts_with(".cfi_"))
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(code.contains("pushq %rbp"), frame, "{assembly}");
            assert_eq!(code.contains("popq %rbp"), frame, "{assembly}");
            assert!(assembly.contains(".cfi_startproc"), "{assembly}");
            assert!(assembly.contains(".cfi_endproc"), "{assembly}");
            if frame {
                assert!(
                    assembly.contains(".cfi_def_cfa_register %rbp"),
                    "{assembly}"
                );
            } else {
                assert!(assembly.contains(".cfi_def_cfa %rsp, 8"), "{assembly}");
                assert!(
                    !code.contains("%rsp") && !code.contains("%rbp"),
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

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod c_caller_abi_tests {
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use crate::test_support::TempDir;
    use std::{error::Error, fs, process::Command};

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
 %w = extend.i8.i64 %x
 ret.i64 %w
}
@export fn @i32_widen(i32 %x) -> i64 {
entry:
 %w = extend.i32.i64 %x
 ret.i64 %w
}
fn @plain_helper() -> i64 {
entry:
 ret.i64 7
}
@extern fn @dirty_i8() -> i8
@extern fn @dirty_u8() -> u8
@extern fn @dirty_i16() -> i16
@extern fn @dirty_i32() -> i32
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
        let guard = TempDir::new("lamina-c-abi");
        let directory = guard.path();
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
        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod debug_line_tests {
    use crate::mir_codegen::{
        simd_support::tools_available,
        test_support::{PRINT_I64_HELPERS, compile},
    };
    use crate::test_support::TempDir;
    use std::{error::Error, fs, process::Command};

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
 call @print_i64(%a)
 call @print_i64(%b)
 ret.i64 0
}
"#;

    #[test]
    fn debug_lines_assemble_across_functions() -> Result<(), Box<dyn Error>> {
        if !tools_available(&["as", "clang", "gcc", "timeout"]) {
            return Ok(());
        }
        let guard = TempDir::new("lamina-debug-lines");
        let directory = guard.path();
        let source = directory.join("two_functions.lamina");
        fs::write(&source, format!("{SOURCE}{PRINT_I64_HELPERS}"))?;

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

        for assembler in ["gas", "clang", "ras"] {
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

        // `%r = mul...` and `ret` of @double sit on lines 4 and 5, @triple's on 9 and 10.
        let rows: Vec<&str> = text
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with(".loc "))
            .filter_map(|line| line.split_whitespace().nth(2))
            .collect();
        for expected in ["4", "5", "9", "10"] {
            assert!(
                rows.contains(&expected),
                "missing .loc for line {expected}: {text}"
            );
        }
        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
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

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod narrow_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::{simd_support::tools_available, test_support::compile};
    use crate::test_support::TempDir;
    use std::{error::Error, fs};

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
@extern fn @raw_u8(u8 %x) -> i64
@extern fn @raw_i8(i8 %x) -> i64
@extern fn @raw_u16(u16 %x) -> i64
@extern fn @raw_i16(i16 %x) -> i64
@extern fn @raw_u32(u32 %x) -> i64
@extern fn @raw_i32(i32 %x) -> i64
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
@extern fn @raw_ret_u8() -> u8
@extern fn @raw_ret_i8() -> i8
@extern fn @raw_ret_u16() -> u16
@extern fn @raw_ret_i16() -> i16
@extern fn @raw_ret_u32() -> u32
@extern fn @raw_ret_i32() -> i32
@export fn @probe_ret_u8() -> i64 {
entry:
 %r = call @raw_ret_u8()
 %w = extend.u8.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i8() -> i64 {
entry:
 %r = call @raw_ret_i8()
 %w = extend.i8.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_u16() -> i64 {
entry:
 %r = call @raw_ret_u16()
 %w = extend.u16.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i16() -> i64 {
entry:
 %r = call @raw_ret_i16()
 %w = extend.i16.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_u32() -> i64 {
entry:
 %r = call @raw_ret_u32()
 %w = extend.u32.i64 %r
 ret.i64 %w
}
@export fn @probe_ret_i32() -> i64 {
entry:
 %r = call @raw_ret_i32()
 %w = extend.i32.i64 %r
 ret.i64 %w
}
"#;

    /// Each `echo_*` receives a narrow argument from a C caller that uses the real
    /// C prototype, and returns it widened by lamina.
    const INCOMING: &str = r#"
@export fn @echo_u8(u8 %x) -> i64 {
entry:
 %w = extend.u8.i64 %x
 ret.i64 %w
}
@export fn @echo_i8(i8 %x) -> i64 {
entry:
 %w = extend.i8.i64 %x
 ret.i64 %w
}
@export fn @echo_u16(u16 %x) -> i64 {
entry:
 %w = extend.u16.i64 %x
 ret.i64 %w
}
@export fn @echo_i16(i16 %x) -> i64 {
entry:
 %w = extend.i16.i64 %x
 ret.i64 %w
}
@export fn @echo_u32(u32 %x) -> i64 {
entry:
 %w = extend.u32.i64 %x
 ret.i64 %w
}
@export fn @echo_i32(i32 %x) -> i64 {
entry:
 %w = extend.i32.i64 %x
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
        let guard = TempDir::new("lamina-narrow-abi");
        let directory = guard.path();

        let outgoing = build_and_run(
            directory,
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
            directory,
            "incoming",
            INCOMING,
            &[
                "echo_u8", "echo_i8", "echo_u16", "echo_i16", "echo_u32", "echo_i32",
            ],
            INCOMING_CALLER,
        )?;
        assert_eq!(incoming, INCOMING_EXPECTED);

        Ok(())
    }

    const PARAM_WIDTHS: &str = r#"
@export fn @p_u8(u8 %x) -> u8 { entry: ret.u8 %x }
@export fn @p_i8(i8 %x) -> i8 { entry: ret.i8 %x }
@export fn @p_u16(u16 %x) -> u16 { entry: ret.u16 %x }
@export fn @p_i16(i16 %x) -> i16 { entry: ret.i16 %x }
@export fn @p_u32(u32 %x) -> u32 { entry: ret.u32 %x }
@export fn @p_i32(i32 %x) -> i32 { entry: ret.i32 %x }
"#;

    /// Every narrow parameter is normalised to the canonical in-register form, which is
    /// sign-extended whatever the declared signedness; unsigned consumers re-extend.
    #[test]
    fn narrow_parameter_prologue_normalizes_to_the_canonical_form() -> Result<(), Box<dyn Error>> {
        let guard = TempDir::new("lamina-param-widths");
        let directory = guard.path();
        let source = directory.join("widths.lamina");
        fs::write(&source, PARAM_WIDTHS)?;
        let base = directory.join("widths");
        compile(&source, &base, &["--emit-asm"])?;
        let assembly = fs::read_to_string(directory.join("widths.s"))?;

        for (function, instruction) in [
            ("p_u8", "movsbq"),
            ("p_i8", "movsbq"),
            ("p_u16", "movswq"),
            ("p_i16", "movswq"),
            ("p_u32", "movslq"),
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

        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod float_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::simd_support::tools_available;
    use crate::test_support::TempDir;
    use std::error::Error;

    const OUTGOING: &str = r#"
@extern fn @c_mix(i64 %a, f64 %x, i64 %b, f32 %y, f64 %z) -> f64
@extern fn @c_spill(i64 %i0, f64 %f0, i64 %i1, f64 %f1, i64 %i2, f64 %f2, i64 %i3, f32 %f3, i64 %i4, f64 %f4, i64 %i5, f64 %f5, i64 %i6, f64 %f6, f64 %f7, f64 %f8) -> f64
@extern fn @c_f32(f32 %x, i64 %n) -> f32
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
        let guard = TempDir::new("lamina-float-abi");
        let directory = guard.path();

        let outgoing = build_and_run(
            directory,
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
            directory,
            "incoming",
            INCOMING,
            &["l_mix", "l_f32", "l_spill"],
            INCOMING_CALLER,
        )?;
        assert_eq!(
            incoming,
            "6.000000 6.000000\n1.500000 1.500000\n3.750000 3.750000\n64.250000 64.250000\n6.500000 6.500000\n"
        );

        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod aggregate_c_abi_tests {
    use super::c_abi_harness::build_and_run;
    use crate::mir_codegen::simd_support::tools_available;
    use crate::test_support::TempDir;
    use std::error::Error;

    /// Two words, one packed word, a padded twelve bytes, and an aggregate pushed past
    /// the last argument register.
    const INCOMING: &str = r#"
@export fn @take_pair(struct { a: i64, b: i64 } %p) -> i64 {
entry:
 %ap = getfieldptr.struct { a: i64, b: i64 } %p, 0
 %bp = getfieldptr.struct { a: i64, b: i64 } %p, 1
 %a = load.i64 %ap
 %b = load.i64 %bp
 %s = add.i64 %a, %b
 ret.i64 %s
}
@export fn @take_small(struct { a: i32, b: i32 } %p) -> i64 {
entry:
 %ap = getfieldptr.struct { a: i32, b: i32 } %p, 0
 %bp = getfieldptr.struct { a: i32, b: i32 } %p, 1
 %a = load.i32 %ap
 %b = load.i32 %bp
 %s = add.i32 %a, %b
 %w = extend.i32.i64 %s
 ret.i64 %w
}
@export fn @take_three(struct { a: i32, b: i32, c: i32 } %p) -> i64 {
entry:
 %ap = getfieldptr.struct { a: i32, b: i32, c: i32 } %p, 0
 %bp = getfieldptr.struct { a: i32, b: i32, c: i32 } %p, 1
 %cp = getfieldptr.struct { a: i32, b: i32, c: i32 } %p, 2
 %a = load.i32 %ap
 %b = load.i32 %bp
 %c = load.i32 %cp
 %ab = add.i32 %a, %b
 %s = add.i32 %ab, %c
 %w = extend.i32.i64 %s
 ret.i64 %w
}
@export fn @take_spilled(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, struct { a: i64, b: i64 } %p) -> i64 {
entry:
 %ab = add.i64 %a, %b
 %cd = add.i64 %c, %d
 %ef = add.i64 %e, %f
 %abcd = add.i64 %ab, %cd
 %scalars = add.i64 %abcd, %ef
 %ap = getfieldptr.struct { a: i64, b: i64 } %p, 0
 %bp = getfieldptr.struct { a: i64, b: i64 } %p, 1
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
@extern fn @c_pair(struct { a: i64, b: i64 } %p) -> i64
@extern fn @c_spilled(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, struct { a: i64, b: i64 } %p) -> i64
@export fn @send_pair() -> i64 {
entry:
 %q = stackalloc.struct { a: i64, b: i64 }
 %qa = getfieldptr.struct { a: i64, b: i64 } %q, 0
 %qb = getfieldptr.struct { a: i64, b: i64 } %q, 1
 store.i64 %qa, 100
 store.i64 %qb, 7
 %r = call @c_pair(struct { a: i64, b: i64 } %q)
 ret.i64 %r
}
@export fn @send_spilled() -> i64 {
entry:
 %q = stackalloc.struct { a: i64, b: i64 }
 %qa = getfieldptr.struct { a: i64, b: i64 } %q, 0
 %qb = getfieldptr.struct { a: i64, b: i64 } %q, 1
 store.i64 %qa, 100
 store.i64 %qb, 7
 %r = call @c_spilled(1, 2, 3, 4, 5, 6, struct { a: i64, b: i64 } %q)
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
 %p = stackalloc.struct { a: i64, b: i64 }
 %pa = getfieldptr.struct { a: i64, b: i64 } %p, 0
 %pb = getfieldptr.struct { a: i64, b: i64 } %p, 1
 store.i64 %pa, %x
 %y = add.i64 %x, 1
 store.i64 %pb, %y
 ret.struct { a: i64, b: i64 } %p
}
@export fn @make_small(i32 %x) -> struct { a: i32, b: i32 } {
entry:
 %p = stackalloc.struct { a: i32, b: i32 }
 %pa = getfieldptr.struct { a: i32, b: i32 } %p, 0
 %pb = getfieldptr.struct { a: i32, b: i32 } %p, 1
 store.i32 %pa, %x
 store.i32 %pb, 9
 ret.struct { a: i32, b: i32 } %p
}
@extern fn @c_make(i64 %x) -> struct { a: i64, b: i64 }
@export fn @take_c() -> i64 {
entry:
 %r = call @c_make(10)
 %ra = getfieldptr.struct { a: i64, b: i64 } %r, 0
 %rb = getfieldptr.struct { a: i64, b: i64 } %r, 1
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
        let guard = TempDir::new("lamina-aggregate-abi");
        let directory = guard.path();

        let incoming = build_and_run(
            directory,
            "incoming",
            INCOMING,
            &["take_pair", "take_small", "take_three", "take_spilled"],
            INCOMING_CALLER,
        )?;
        assert_eq!(incoming, "42 7 6 63\n");

        let outgoing = build_and_run(
            directory,
            "outgoing",
            OUTGOING,
            &["send_pair", "send_spilled"],
            OUTGOING_CALLEES,
        )?;
        assert_eq!(outgoing, "107 100028\n");

        let returns = build_and_run(
            directory,
            "returns",
            RETURNS,
            &["make_pair", "make_small", "take_c"],
            RETURNS_CALLER,
        )?;
        assert_eq!(returns, "41 42 5 9 80\n");

        Ok(())
    }
}

#[test]
fn external_declarations_come_out_in_one_order() {
    let ir = parse_module(
        "@extern fn @zeta(i64 %a) -> i64\n@extern fn @alpha(i64 %a) -> i64\n\
         @extern fn @mu(i64 %a) -> i64\nfn @main() -> i64 {\nentry:\n\
         %a = call @zeta(1)\n%b = call @alpha(2)\n%c = call @mu(3)\nret.i64 %a\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "externs").unwrap();

    let emit = || {
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .filter(|line| line.starts_with(".extern "))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };

    let first = emit();
    assert_eq!(first, [".extern alpha", ".extern mu", ".extern zeta"]);
    for _ in 0..8 {
        assert_eq!(emit(), first);
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod edge_move_tests {
    use crate::mir_codegen::test_support::{PRINT_I64_HELPERS, compile};
    use crate::test_support::TempDir;
    use std::{error::Error, fs, process::Command};

    fn run(name: &str, lamina: &str) -> Result<String, Box<dyn Error>> {
        let guard = TempDir::new(&format!("lamina-edge-{name}"));
        let directory = guard.path();
        let source = directory.join(format!("{name}.lamina"));
        fs::write(&source, format!("{lamina}{PRINT_I64_HELPERS}"))?;
        let executable = directory.join(name);
        compile(&source, &executable, &[])?;
        let output = Command::new(&executable).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    #[test]
    fn swapped_copies_exchange_their_values() -> Result<(), Box<dyn Error>> {
        // a and b trade places on each back edge, so the loop leaves a = 1 and b = 2.
        let printed = run(
            "swap",
            "fn @main() -> i64 {\nentry:\n  %a = copy.i64 1\n  %b = copy.i64 2\n  %c = copy.i64 0\n  jmp loop\nloop:\n  %cn = add.i64 %c, 1\n  %go = lt.i64 %cn, 3\n  br %go, latch, done\nlatch:\n  %t0 = copy.i64 %a\n  %a = copy.i64 %b\n  %b = copy.i64 %t0\n  %c = copy.i64 %cn\n  jmp loop\ndone:\n  %t = mul.i64 %a, 10\n  %r = add.i64 %t, %b\n  call @print_i64(%r)\n  ret.i64 0\n}\n",
        )?;
        assert_eq!(printed, "12\n");
        Ok(())
    }

    #[test]
    fn switch_runs_its_edge_moves_after_the_scrutinee() -> Result<(), Box<dyn Error>> {
        // i is read by the switch, so i <- n must not happen before the comparison.
        let printed = run(
            "switch",
            "fn @main() -> i64 {\nentry:\n  %i = copy.i64 0\n  jmp loop\nloop:\n  %n = add.i64 %i, 2\n  switch.i64 %i, latch, [4, done]\nlatch:\n  %i = copy.i64 %n\n  jmp loop\ndone:\n  call @print_i64(%n)\n  ret.i64 0\n}\n",
        )?;
        assert_eq!(printed, "6\n");
        Ok(())
    }
}

#[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
mod intrinsic_namespace_tests {
    use crate::mir_codegen::test_support::{PRINT_I64_HELPERS, compile};
    use crate::test_support::TempDir;
    use std::{error::Error, fs, process::Command};

    #[test]
    fn a_function_may_take_an_intrinsic_name() -> Result<(), Box<dyn Error>> {
        let guard = TempDir::new("lamina-intrinsic");
        let directory = guard.path();
        for name in ["write", "read", "writebyte", "readbyte"] {
            let lamina = format!(
                "fn @{name}(i64 %x) -> i64 {{\nentry:\n  %r = add.i64 %x, 100\n  ret.i64 %r\n}}\n\nfn @main() -> i64 {{\nentry:\n  %v = call @{name}(5)\n  call @print_i64(%v)\n  ret.i64 0\n}}\n{PRINT_I64_HELPERS}"
            );
            let source = directory.join(format!("{name}.lamina"));
            fs::write(&source, lamina)?;
            let executable = directory.join(name);
            compile(&source, &executable, &[])?;
            let output = Command::new(&executable).output()?;
            assert_eq!(String::from_utf8_lossy(&output.stdout), "105\n", "@{name}");
        }
        Ok(())
    }
}

fn x86_assembly_for(source: &str) -> String {
    let ir = parse_module(source).expect("parse atomics");
    let mir = from_ir_x86_64(&ir, "atomics").expect("lower atomics");
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).expect("emit atomics");
    String::from_utf8(output).expect("utf8 assembly")
}

#[test]
fn atomic_cmpxchg_emits_locked_cmpxchg_with_expected_in_rax() {
    let asm = x86_assembly_for(
        "fn @f(ptr %p) -> i64 {\nentry:\n %old = atomic_cmpxchg.i32 %p, 1, 2, seqcst, acquire\n %w = extend.i32.i64 %old\n ret.i64 %w\n}",
    );
    assert_eq!(asm.matches("lock cmpxchgl %").count(), 1, "{asm}");
    assert!(
        asm.contains("movq $1, %rax"),
        "expected loads into rax\n{asm}"
    );
    assert!(!asm.contains("does not support"), "{asm}");
}

#[test]
fn atomics_emit_locked_read_modify_write_xchg_and_mfence() {
    let asm = x86_assembly_for(
        "fn @f(ptr %p) -> i64 {\nentry:\n %a = atomic_load.i64 %p, seqcst\n atomic_store.i64 %p, 5, seqcst\n %b = atomic_add.i64 %p, 1, seqcst\n %c = atomic_sub.i64 %p, 1, acqrel\n %d = atomic_exchange.i64 %p, 2, seqcst\n %e = atomic_and.i64 %p, 3, relaxed\n %g = atomic_or.i64 %p, 4, seqcst\n %h = atomic_xor.i64 %p, 5, seqcst\n fence seqcst\n fence acquire\n ret.i64 %b\n}",
    );
    assert_eq!(asm.matches("lock xaddq").count(), 2, "{asm}");
    assert_eq!(
        asm.matches("xchgq %rax, (%").count(),
        2,
        "seqcst store and exchange\n{asm}"
    );
    assert_eq!(
        asm.matches("lock cmpxchgq").count(),
        3,
        "and, or, xor loops\n{asm}"
    );
    assert_eq!(asm.matches("jne 1b").count(), 3, "{asm}");
    assert_eq!(
        asm.matches("mfence").count(),
        1,
        "only the seqcst fence\n{asm}"
    );
    assert!(asm.contains("negq %rax"), "{asm}");
    assert!(
        !asm.contains("lock xchg"),
        "xchg is implicitly locked\n{asm}"
    );
}

#[test]
fn atomic_loads_and_relaxed_stores_are_plain_moves() {
    let asm = x86_assembly_for(
        "fn @f(ptr %p) -> i64 {\nentry:\n %a = atomic_load.i64 %p, seqcst\n atomic_store.i64 %p, 7, release\n ret.i64 %a\n}",
    );
    assert!(!asm.contains("xchg"), "{asm}");
    assert!(!asm.contains("mfence"), "{asm}");
    assert!(!asm.contains("lock"), "{asm}");
    assert!(asm.contains("movq (%rax), %rax"), "{asm}");
    assert!(asm.contains("movq %rax, (%"), "{asm}");
}

#[test]
fn each_memory_ordering_emits_the_instruction_x86_requires() {
    let body = |op: &str| {
        x86_assembly_for(&format!(
            "fn @f(ptr %p) -> i64 {{\nentry:\n {op}\n ret.i64 0\n}}"
        ))
    };
    let load = |o: &str| body(&format!("%a = atomic_load.i64 %p, {o}"));
    assert_eq!(load("acquire"), load("relaxed"));
    assert_eq!(load("seqcst"), load("relaxed"));

    let store = |o: &str| body(&format!("atomic_store.i64 %p, 7, {o}"));
    assert_eq!(store("release"), store("relaxed"));
    assert!(
        store("relaxed").contains("movq %rax, (%"),
        "{}",
        store("relaxed")
    );
    let seqcst = store("seqcst");
    assert!(seqcst.contains("xchgq %rax, (%"), "{seqcst}");
    assert!(!seqcst.contains("movq %rax, (%"), "{seqcst}");

    let rmw = |o: &str| body(&format!("%a = atomic_add.i64 %p, 1, {o}"));
    assert_eq!(rmw("relaxed"), rmw("seqcst"));
    assert_eq!(rmw("acquire"), rmw("seqcst"));
    assert_eq!(rmw("release"), rmw("seqcst"));
    assert!(rmw("relaxed").contains("lock xaddq"), "{}", rmw("relaxed"));

    let cas = |s: &str, f: &str| body(&format!("%a = atomic_cmpxchg.i64 %p, 1, 2, {s}, {f}"));
    assert_eq!(cas("relaxed", "relaxed"), cas("seqcst", "seqcst"));
    assert_eq!(cas("acqrel", "acquire"), cas("seqcst", "seqcst"));
    assert!(cas("relaxed", "relaxed").contains("lock cmpxchgq"));

    let plain = body("%a = atomic_load.i64 %p, relaxed");
    let fence = |o: &str| body(&format!("fence {o}"));
    for o in ["relaxed", "acquire", "release", "acqrel"] {
        assert!(!fence(o).contains("mfence"), "{}", fence(o));
    }
    assert_eq!(fence("acquire"), fence("relaxed"));
    assert!(fence("seqcst").contains("mfence"), "{}", fence("seqcst"));
    assert!(!plain.contains("mfence"), "{plain}");
}

#[test]
fn narrow_atomic_add_uses_sized_locked_xadd_and_sign_extends() {
    let asm = x86_assembly_for(
        "fn @f(ptr %p) -> i64 {\nentry:\n %a = atomic_add.i8 %p, 1, seqcst\n ret.i64 0\n}",
    );
    assert!(asm.contains("lock xaddb %al, (%"), "{asm}");
    assert!(asm.contains("movsbq %al, %rax"), "{asm}");
}

#[test]
fn mach_o_io_intrinsics_report_a_failed_syscall_as_minus_two() {
    let source = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %w = write %p, 4\n  %r = read %p, 4\n  %b = writebyte 65\n  ret.i64 %w\n}\n";
    let ir = parse_module(source).expect("parse io");
    let mir = from_ir_x86_64(&ir, "io").expect("lower io");
    let mut mach = Vec::new();
    generate_mir_x86_64(&mir, &mut mach, TargetOperatingSystem::MacOS).expect("emit macos io");
    let mach = String::from_utf8(mach).expect("utf8 assembly");
    assert_eq!(
        mach.matches("syscall\n    movq $-2, %rcx\n    cmovcq %rcx, %rax\n")
            .count(),
        3,
        "{mach}"
    );

    let mut linux = Vec::new();
    generate_mir_x86_64(&mir, &mut linux, TargetOperatingSystem::Linux).expect("emit linux io");
    let linux = String::from_utf8(linux).expect("utf8 assembly");
    assert!(!linux.contains("cmovcq"), "{linux}");
}

#[test]
fn freebsd_io_intrinsics_use_the_bsd_syscall_numbers() {
    let source = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %w = write %p, 4\n  %r = read %p, 4\n  %b = writebyte 65\n  %c = readbyte\n  ret.i64 %w\n}\n";
    let ir = parse_module(source).expect("parse io");
    let mir = from_ir_x86_64(&ir, "io").expect("lower io");
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::FreeBSD).expect("emit freebsd");
    let asm = String::from_utf8(output).expect("utf8 assembly");

    assert_eq!(constants::freebsd::SYS_WRITE, 4);
    assert_eq!(constants::freebsd::SYS_READ, 3);

    // Number in %rax, arguments in %rdi %rsi %rdx, as on Linux; only the numbers differ.
    let expected = [
        "    movq %r10, %rsi\n    movq $1, %rdi\n    movq $4, %rax\n    syscall\n",
        "    movq %r10, %rsi\n    movq $0, %rdi\n    movq $3, %rax\n    syscall\n",
        "    movq $4, %rax\n    movq $1, %rdi\n    movq %rsp, %rsi\n    movq $1, %rdx\n    syscall\n",
        "    movq $3, %rax\n    movq $0, %rdi\n    movq %rsp, %rsi\n    movq $1, %rdx\n    syscall\n",
    ];
    for sequence in expected {
        assert_eq!(asm.matches(sequence).count(), 1, "{sequence}\n{asm}");
    }

    // FreeBSD reports a failed syscall in the carry flag, never as a negative count.
    assert_eq!(
        asm.matches("    syscall\n    movq $-2, %rcx\n    cmovcq %rcx, %rax\n")
            .count(),
        3,
        "{asm}"
    );
    assert_eq!(
        asm.matches("    syscall\n    movq $-2, %rdx\n    cmovcq %rdx, %rax\n")
            .count(),
        1,
        "{asm}"
    );
    assert!(!asm.contains("cmovsq"), "{asm}");
}

#[test]
fn thread_local_globals_emit_the_relocations_their_model_names() {
    let assemble = |annotation: &str| {
        let source = format!(
            "{annotation} global $t: i64 = 0\nfn @main() -> i64 {{\nentry:\n    store.i64 $t, 7\n    %v = load.i64 $t\n    ret.i64 %v\n}}\n"
        );
        let ir = parse_module(&source).unwrap();
        let mir = from_ir_x86_64(&ir, "tls").unwrap();
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
        String::from_utf8(output).unwrap()
    };

    let local = assemble("@thread_local");
    assert!(local.contains("    movq %fs:0, %rax"), "{local}");
    assert!(local.contains("    leaq t@tpoff(%rax), %rax"), "{local}");
    assert!(!local.contains("gottpoff"), "{local}");
    assert!(local.contains(".section .tbss,\"awT\",@nobits"), "{local}");

    let initial = assemble("@thread_local(initial_exec)");
    assert!(
        initial.contains("    movq t@gottpoff(%rip), %rax"),
        "{initial}"
    );
    assert!(initial.contains("    addq %fs:0, %rax"), "{initial}");
    assert!(!initial.contains("@tpoff("), "{initial}");
}

#[test]
fn an_initialized_thread_local_goes_to_tdata() {
    let ir = parse_module(
        "@thread_local global $t: i64 = 9\nfn @main() -> i64 {\nentry:\n    %v = load.i64 $t\n    ret.i64 %v\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "tls").unwrap();
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
    let assembly = String::from_utf8(output).unwrap();
    assert!(
        assembly.contains(".section .tdata,\"awT\",@progbits"),
        "{assembly}"
    );
}

#[test]
fn thread_local_storage_is_refused_on_mach_o() {
    let ir = parse_module(
        "@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    %v = load.i64 $t\n    ret.i64 %v\n}\n",
    )
    .unwrap();
    let mir = from_ir_x86_64(&ir, "tls").unwrap();
    let mut output = Vec::new();
    let error = generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::MacOS)
        .expect_err("Mach-O thread locals are refused");
    assert!(error.to_string().contains("@TLVP"), "{error}");
}

#[test]
fn overflow_flags_read_the_condition_the_signedness_names() {
    let assemble = |line: &str| {
        let source = format!(
            "fn @main(i64 %a, i64 %b) -> i64 {{\nentry:\n    {line}\n    %r = extend.bool.i64 %f\n    ret.i64 %r\n}}\n"
        );
        let ir = parse_module(&source).unwrap();
        let mir = from_ir_x86_64(&ir, "ovf").unwrap();
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
        String::from_utf8(output).unwrap()
    };

    let signed_add = assemble("%f = add.ovf.i64 %a, %b");
    assert!(signed_add.contains("    seto %al"), "{signed_add}");

    let unsigned_add = assemble("%f = add.ovf.u64 %a, %b");
    assert!(unsigned_add.contains("    setc %al"), "{unsigned_add}");

    let signed_mul = assemble("%f = mul.ovf.i64 %a, %b");
    assert!(signed_mul.contains("imulq"), "{signed_mul}");
    assert!(signed_mul.contains("    seto %al"), "{signed_mul}");

    let unsigned_mul = assemble("%f = mul.ovf.u64 %a, %b");
    assert!(unsigned_mul.contains("    mulq "), "{unsigned_mul}");
    assert!(unsigned_mul.contains("    setc %al"), "{unsigned_mul}");

    let source = "fn @main(i32 %a, i32 %b) -> i64 {\nentry:\n    %f = add.ovf.i32 %a, %b\n    %r = extend.bool.i64 %f\n    ret.i64 %r\n}\n";
    let ir = parse_module(source).unwrap();
    let mir = from_ir_x86_64(&ir, "ovf").unwrap();
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
    let narrow = String::from_utf8(output).unwrap();
    assert!(narrow.contains("    setne %al"), "{narrow}");
}

#[test]
fn a_stated_alignment_does_not_change_x86_64_code() {
    let assemble = |line: &str| {
        let source = format!("fn @main(ptr %p) -> i64 {{\nentry:\n    {line}\n    ret.i64 0\n}}\n");
        let ir = parse_module(&source).unwrap();
        let mir = from_ir_x86_64(&ir, "align").unwrap();
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Linux).unwrap();
        String::from_utf8(output).unwrap()
    };
    assert_eq!(
        assemble("store.i32 %p, 1"),
        assemble("store.i32 %p, 1 align 1"),
    );
}

#[test]
fn coff_section_attribute_places_code_in_the_named_section() {
    let asm = assemble_for_windows(
        "@section(\".text.custom\")\nfn @placed(i64 %x) -> i64 {\nentry:\n    ret.i64 %x\n}\n",
    )
    .unwrap();
    assert!(asm.contains("    .section .text.custom,\"axr\"\n"), "{asm}");
}

#[test]
fn macho_section_attribute_maps_text_names_and_refuses_the_rest() {
    let mac_asm = |section: &str| {
        let source = format!(
            "@section(\"{section}\")\nfn @placed(i64 %x) -> i64 {{\nentry:\n    ret.i64 %x\n}}\n"
        );
        let ir = parse_module(&source).unwrap();
        let mir = from_ir_x86_64(&ir, "macho_section").unwrap();
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::MacOS)
            .map(|()| String::from_utf8(output).unwrap())
    };
    let asm = mac_asm(".text").unwrap();
    assert!(
        asm.contains("    .section .text,\"ax\",@progbits\n"),
        "{asm}"
    );
    let asm = mac_asm(".text.custom").unwrap();
    assert!(
        asm.contains("    .section __TEXT,__custom,regular,pure_instructions\n"),
        "{asm}"
    );
    for name in [".rodata", ".text.", ".text.sixteen_chars_xx"] {
        let error = mac_asm(name).unwrap_err().to_string();
        assert!(error.contains(&format!("'{name}'")), "{error}");
        assert!(error.contains("16 bytes"), "{error}");
    }
}

#[test]
fn thread_local_globals_are_refused_on_macho_before_any_code() {
    let source = "@thread_local global $t: i64 = 0\nfn @main() -> i64 {\nentry:\n    %x = load.i64 $t\n    ret.i64 %x\n}\n";
    let ir = parse_module(source).unwrap();
    let mir = from_ir_x86_64(&ir, "tls_refusal").unwrap();
    let mut output = Vec::new();
    let error = generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::MacOS)
        .expect_err("thread-local global must be refused on Mach-O")
        .to_string();
    assert!(error.contains("'t'"), "{error}");
    assert!(error.contains("@TLVP"), "{error}");
    assert!(error.contains("does not emit"), "{error}");
    assert!(output.is_empty(), "wrote code before refusing");
}

const WIN64_FLOAT_PAIR_TAKE: &str = "type !F2 = struct { a: f32, b: f32 }\nfn @take(!F2 %p) -> i64 {\nentry:\n    %ap = getfieldptr.!F2 %p, 0\n    %a = load.f32 %ap\n    %ai = fptoint.f32.i64 %a\n    ret.i64 %ai\n}\nfn @main() -> i64 {\nentry:\n    %p = stackalloc.!F2\n    %pa = getfieldptr.!F2 %p, 0\n    store.f32 %pa, 2.0\n    %r = call @take(!F2 %p)\n    ret.i64 %r\n}\n";

fn windows_function<'a>(assembly: &'a str, name: &str) -> &'a str {
    let start = assembly
        .find(&format!("\n{name}:\n"))
        .unwrap_or_else(|| panic!("no {name} in {assembly}"));
    let rest = &assembly[start..];
    let end = rest.find("\n    ret\n").map_or(rest.len(), |at| at + 9);
    &rest[..end]
}

#[test]
fn win64_float_pair_argument_arrives_in_rcx_and_is_not_loaded_from_xmm() {
    let assembly = assemble_for_windows(WIN64_FLOAT_PAIR_TAKE).unwrap();
    let take = windows_function(&assembly, "take");
    let prologue = take.split(".L_take.entry:").next().unwrap();
    assert!(prologue.contains("    movq %rcx, -"), "{take}");
    assert!(!prologue.contains("%xmm"), "{take}");
    let main = windows_function(&assembly, "main");
    let staged = main.split("    call take\n").next().unwrap();
    assert!(!staged.contains("%xmm"), "{main}");
    assert!(staged.contains("%rcx"), "{main}");
}

#[test]
fn win64_float_pair_return_comes_back_in_rax() {
    let assembly = assemble_for_windows(
        "type !F2 = struct { a: f32, b: f32 }\nfn @make(f32 %x) -> !F2 {\nentry:\n    %p = stackalloc.!F2\n    %pa = getfieldptr.!F2 %p, 0\n    store.f32 %pa, %x\n    ret.!F2 %p\n}\nfn @main() -> i64 {\nentry:\n    %q = call @make(2.0)\n    %qa = getfieldptr.!F2 %q, 0\n    %v = load.f32 %qa\n    %vi = fptoint.f32.i64 %v\n    ret.i64 %vi\n}\n",
    )
    .unwrap();
    let make = windows_function(&assembly, "make");
    assert!(make.contains("    movq (%rax), %rax\n    addq $"), "{make}");
    let main = windows_function(&assembly, "main");
    assert!(
        main.contains("    call make\n    addq $32, %rsp\n    movq %rax, "),
        "{main}"
    );
    let staged = main.split("    call make\n").next().unwrap();
    assert!(!staged.contains("leaq"), "{main}");
}

#[test]
fn win64_double_pair_is_passed_by_reference_through_rcx() {
    let assembly = assemble_for_windows(
        "type !D2 = struct { a: f64, b: f64 }\nfn @take(!D2 %p) -> i64 {\nentry:\n    %ap = getfieldptr.!D2 %p, 0\n    %a = load.f64 %ap\n    %ai = fptoint.f64.i64 %a\n    ret.i64 %ai\n}\nfn @main() -> i64 {\nentry:\n    %p = stackalloc.!D2\n    %pa = getfieldptr.!D2 %p, 0\n    store.f64 %pa, 2.0\n    %r = call @take(!D2 %p)\n    ret.i64 %r\n}\n",
    )
    .unwrap();
    let take = windows_function(&assembly, "take");
    assert!(take.contains("    movq 0(%rcx), %rax\n"), "{take}");
    let main = windows_function(&assembly, "main");
    let staged = main.split("    call take\n").next().unwrap();
    assert!(staged.contains("    leaq "), "{main}");
    assert!(staged.contains(", %rcx\n"), "{main}");
    assert!(!staged.contains("%xmm"), "{main}");
}

#[test]
fn win64_by_reference_copy_reads_no_byte_past_the_aggregate() {
    let assembly = assemble_for_windows(
        "type !F3 = struct { a: f32, b: f32, c: f32 }\nfn @take(!F3 %p) -> i64 {\nentry:\n    %ap = getfieldptr.!F3 %p, 0\n    %a = load.f32 %ap\n    %ai = fptoint.f32.i64 %a\n    ret.i64 %ai\n}\nfn @main() -> i64 {\nentry:\n    %p = stackalloc.!F3\n    %pa = getfieldptr.!F3 %p, 0\n    store.f32 %pa, 2.0\n    %r = call @take(!F3 %p)\n    ret.i64 %r\n}\n",
    )
    .unwrap();
    let main = windows_function(&assembly, "main");
    let staged = main.split("    call take\n").next().unwrap();
    assert!(staged.contains("    movq 0(%rax), %r10\n"), "{main}");
    assert!(staged.contains("    movl 8(%rax), %r10d\n"), "{main}");
    assert!(!staged.contains("    movq 8(%rax)"), "{main}");
}

#[test]
fn win64_by_reference_parameter_copy_reads_no_byte_past_the_aggregate() {
    let assembly = assemble_for_windows(
        "type !F3 = struct { a: f32, b: f32, c: f32 }\nfn @take(!F3 %p) -> i64 {\nentry:\n    %ap = getfieldptr.!F3 %p, 0\n    %a = load.f32 %ap\n    %ai = fptoint.f32.i64 %a\n    ret.i64 %ai\n}\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n",
    )
    .unwrap();
    let take = windows_function(&assembly, "take");
    assert!(take.contains("    movq 0(%rcx), %rax\n"), "{take}");
    assert!(take.contains("    movl 8(%rcx), %eax\n"), "{take}");
    assert!(!take.contains("    movq 8(%rcx)"), "{take}");
}

#[test]
fn win64_float_struct_fifth_argument_is_passed_on_the_stack() {
    let assembly = assemble_for_windows(
        "type !F2 = struct { a: f32, b: f32 }\nfn @fifth(i64 %a, i64 %b, i64 %c, i64 %d, !F2 %p) -> i64 {\nentry:\n    %ap = getfieldptr.!F2 %p, 0\n    %av = load.f32 %ap\n    %ai = fptoint.f32.i64 %av\n    %s = add.i64 %a, %ai\n    ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n    %p = stackalloc.!F2\n    %pa = getfieldptr.!F2 %p, 0\n    store.f32 %pa, 2.0\n    %r = call @fifth(1, 2, 3, 4, !F2 %p)\n    ret.i64 %r\n}\n",
    )
    .unwrap();
    let fifth = windows_function(&assembly, "fifth");
    assert!(fifth.contains("    movq 48(%rbp), %rax\n"), "{fifth}");
    let main = windows_function(&assembly, "main");
    assert!(
        main.contains("    movq %rax, 32(%rsp)\n    movq $1, %rcx\n    movq $2, %rdx\n    movq $3, %r8\n    movq $4, %r9\n    call fifth\n"),
        "{main}"
    );
}

#[test]
fn win64_variadic_call_with_a_float_struct_keeps_the_struct_out_of_xmm() {
    let assembly = assemble_for_windows(
        "type !F2 = struct { a: f32, b: f32 }\nfn @sum(i64 %n, ...) -> i64 {\nentry:\n    %ap = stackalloc.valist\n    va_start %ap\n    %x = va_arg.i64 %ap\n    va_end %ap\n    ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n    %p = stackalloc.!F2\n    %pa = getfieldptr.!F2 %p, 0\n    store.f32 %pa, 2.0\n    %r = call @sum(1, !F2 %p, 2.5)\n    ret.i64 %r\n}\n",
    )
    .unwrap();
    let main = windows_function(&assembly, "main");
    let staged = main.split("    call sum\n").next().unwrap();
    assert!(staged.contains("    movq %xmm2, %r8\n"), "{main}");
    assert!(!staged.contains("%xmm1"), "{main}");
}

#[test]
fn float_struct_laid_out_for_sysv_is_refused_by_name_on_windows() {
    let ir = parse_module(WIN64_FLOAT_PAIR_TAKE).unwrap();
    let mir = from_ir_x86_64(&ir, "win64").unwrap();
    let mut output = Vec::new();
    let error = generate_mir_x86_64(&mir, &mut output, TargetOperatingSystem::Windows)
        .expect_err("a System V classified aggregate is not a Win64 placement");
    assert!(error.to_string().contains("@take"), "{error}");
    assert!(output.is_empty(), "wrote code before refusing");
}

const GLOBAL_LINKAGE_ELF_SOURCE: &str = "@weak global $weak_g: i64 = 1\n@hidden global $hidden_g: i64 = 2\n@protected global $protected_g: i64 = 3\n@internal global $internal_g: i64 = 4\n@private global $private_g: i64 = 5\n@export @hidden global $exported_g: i64 = 6\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n";
const GLOBAL_LINKAGE_MACHO_SOURCE: &str = "@weak global $weak_g: i64 = 1\n@hidden global $hidden_g: i64 = 2\n@internal global $internal_g: i64 = 3\n@private global $private_g: i64 = 4\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n";

fn x86_64_global_linkage_for(source: &str, os: TargetOperatingSystem) -> Result<String, String> {
    let ir = parse_module(source).unwrap();
    let mir = from_ir_x86_64(&ir, "global_linkage").unwrap();
    let mut output = Vec::new();
    generate_mir_x86_64(&mir, &mut output, os).map_err(|error| error.to_string())?;
    Ok(String::from_utf8(output).unwrap())
}

#[test]
fn elf_global_visibility_and_weak_use_the_function_directives() {
    let text = x86_64_global_linkage_for(GLOBAL_LINKAGE_ELF_SOURCE, TargetOperatingSystem::Linux)
        .expect("emit elf globals");
    for line in [
        ".weak weak_g",
        ".hidden hidden_g",
        ".protected protected_g",
        ".internal internal_g",
        ".local private_g",
        ".hidden exported_g",
        ".globl weak_g",
        ".globl hidden_g",
        ".globl protected_g",
        ".globl exported_g",
    ] {
        assert!(text.lines().any(|l| l == line), "missing {line}: {text}");
    }
    for line in [".globl internal_g", ".globl private_g"] {
        assert!(
            !text.lines().any(|l| l == line),
            "unexpected {line}: {text}"
        );
    }
}

#[test]
fn macho_hidden_global_is_private_extern_and_weak_is_weak_definition() {
    let text = x86_64_global_linkage_for(GLOBAL_LINKAGE_MACHO_SOURCE, TargetOperatingSystem::MacOS)
        .expect("emit macho globals");
    for line in [
        ".weak_definition _weak_g",
        ".globl _weak_g",
        ".private_extern _hidden_g",
        ".globl _hidden_g",
        ".private_extern _internal_g",
        ".private_extern _private_g",
    ] {
        assert!(text.lines().any(|l| l == line), "missing {line}: {text}");
    }
    for line in [
        ".globl _internal_g",
        ".globl _private_g",
        ".hidden",
        ".protected",
    ] {
        assert!(!text.contains(line), "unexpected {line}: {text}");
    }
}

#[test]
fn macho_refuses_protected_global_by_name() {
    let error = x86_64_global_linkage_for(
        "@protected global $prot: i64 = 1\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n",
        TargetOperatingSystem::MacOS,
    )
    .expect_err("protected global on Mach-O");
    assert!(
        error.contains("@protected on global 'prot' is not supported for MacOS"),
        "{error}"
    );
}

#[test]
fn coff_refuses_protected_global_by_name() {
    let error = x86_64_global_linkage_for(
        "@protected global $prot: i64 = 1\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n",
        TargetOperatingSystem::Windows,
    )
    .expect_err("protected global on COFF");
    assert!(
        error.contains("@protected on global 'prot' is not supported for Windows"),
        "{error}"
    );
}

#[test]
fn coff_weak_and_hidden_globals_stay_external() {
    for (annotation, global, directive) in [("@weak", "w", ".weak w"), ("@hidden", "h", ".globl h")]
    {
        let source = format!(
            "{annotation} global ${global}: i64 = 1\nfn @main() -> i64 {{\nentry:\n    ret.i64 0\n}}\n"
        );
        let text = x86_64_global_linkage_for(&source, TargetOperatingSystem::Windows)
            .expect("accepted on COFF");
        assert!(text.lines().any(|l| l == directive), "{directive}: {text}");
        assert!(!text.contains(".local"), "{annotation}: {text}");
    }
}

#[test]
fn elf_data_symbols_carry_their_type_and_size() {
    let assemble = |source: &str, os: TargetOperatingSystem| {
        let ir = parse_module(source).unwrap();
        let mir = from_ir_x86_64(&ir, "symbols").unwrap();
        let mut output = Vec::new();
        generate_mir_x86_64(&mir, &mut output, os).unwrap();
        String::from_utf8(output).unwrap()
    };
    let main = "fn @main() -> i64 {\nentry:\n    %a = load.i64 $counter\n    ret.i64 %a\n}\n";
    let plain = format!("global $counter: i64 = 41\nglobal $zeros: i64 = 0\n\n{main}");
    let linux = assemble(
        &format!("@thread_local\nglobal $tl: i64 = 7\n{plain}"),
        TargetOperatingSystem::Linux,
    );
    for (name, kind) in [
        ("counter", "object"),
        ("tl", "tls_object"),
        ("zeros", "object"),
    ] {
        assert!(
            linux.contains(&format!(".type {name}, %{kind}\n")),
            "{linux}"
        );
        assert!(
            linux.contains(&format!(".size {name}, .-{name}\n")),
            "{linux}"
        );
    }
    for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::Windows] {
        let asm = assemble(&plain, os);
        assert!(!asm.contains(".type ") && !asm.contains(".size "), "{asm}");
    }
}
