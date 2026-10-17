use super::*;
use crate::{
    mir::{FunctionBuilder, ScalarType, codegen::from_ir},
    parser::parse_module,
};
use std::slice;

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
