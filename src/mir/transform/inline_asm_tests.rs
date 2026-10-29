#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::Transform;
use super::copy_propagation::CopyPropagation;
use super::deadcode::DeadCodeElimination;
use super::loop_opt::LoopInvariantCodeMotion;
use super::memory::MemoryOptimization;
use super::scheduling::InstructionScheduling;
use super::test_utils::apply_pass;
use crate::mir::{
    AddressMode, AsmOperand, FunctionBuilder, Immediate, Instruction, IntBinOp, MemoryAttrs,
    MirType, Operand, Register, ScalarType, VirtualReg,
};

fn i64_ty() -> MirType {
    MirType::Scalar(ScalarType::I64)
}

fn gpr(id: u32) -> Register {
    VirtualReg::gpr(id).into()
}

fn asm_read(dst: Register, input: Register) -> Instruction {
    Instruction::InlineAsm {
        template: "movq $0, $1".to_owned(),
        operands: vec![
            AsmOperand::Input {
                location: "reg".to_owned(),
                value: Operand::Register(input),
            },
            AsmOperand::Output {
                location: "reg".to_owned(),
            },
        ],
        clobbers: Vec::new(),
        clobber_memory: false,
        result: Some((dst, i64_ty())),
    }
}

fn slot_at(base: Register) -> AddressMode {
    AddressMode::BaseOffset {
        base,
        offset: 0,
        address_space: Default::default(),
    }
}

#[test]
fn load_is_not_forwarded_across_inline_asm() {
    let base = gpr(0);
    let mut func = FunctionBuilder::new("f")
        .param(base.clone(), i64_ty())
        .returns(i64_ty())
        .block("entry")
        .instr(Instruction::Store {
            ty: i64_ty(),
            src: Operand::Immediate(Immediate::I64(1)),
            addr: slot_at(base.clone()),
            attrs: MemoryAttrs::default(),
        })
        .instr(asm_read(gpr(9), base.clone()))
        .instr(Instruction::Load {
            ty: i64_ty(),
            dst: gpr(1),
            addr: slot_at(base),
            attrs: MemoryAttrs::default(),
        })
        .instr(Instruction::Ret {
            value: Some(Operand::Register(gpr(1))),
        })
        .build();

    apply_pass(&MemoryOptimization, &mut func);
    assert!(matches!(
        &func.blocks[0].instructions[2].instr,
        Instruction::Load { .. }
    ));
}

#[test]
fn asm_with_result_is_not_hoisted_out_of_a_loop() {
    let mut func = FunctionBuilder::new("hoist")
        .param(gpr(0), i64_ty())
        .returns(i64_ty())
        .block("entry")
        .instr(Instruction::Jmp {
            target: "loop".to_owned(),
        })
        .block("loop")
        .instr(asm_read(gpr(1), gpr(0)))
        .instr(Instruction::Br {
            cond: gpr(2),
            true_target: "loop".to_owned(),
            false_target: "exit".to_owned(),
        })
        .block("exit")
        .instr(Instruction::Ret { value: None })
        .build();

    let changed = LoopInvariantCodeMotion.apply(&mut func).expect("licm");
    assert!(!changed);
    assert!(matches!(
        func.get_block("loop").unwrap().instructions[0].instr,
        Instruction::InlineAsm { .. }
    ));
}

#[test]
fn asm_with_unused_result_is_not_deleted() {
    let mut func = FunctionBuilder::new("keep")
        .param(gpr(0), i64_ty())
        .returns(i64_ty())
        .block("entry")
        .instr(asm_read(gpr(1), gpr(0)))
        .instr(Instruction::Ret {
            value: Some(Operand::Immediate(Immediate::I64(0))),
        })
        .build();

    apply_pass(&DeadCodeElimination, &mut func);
    assert!(matches!(
        &func.blocks[0].instructions[0].instr,
        Instruction::InlineAsm { .. }
    ));
}

#[test]
fn asm_keeps_its_position_relative_to_independent_loads() {
    let base = gpr(0);
    let mut func = FunctionBuilder::new("order")
        .param(base.clone(), i64_ty())
        .returns(i64_ty())
        .block("entry")
        .instr(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty(),
            dst: gpr(5),
            lhs: Operand::Immediate(Immediate::I64(1)),
            rhs: Operand::Immediate(Immediate::I64(2)),
        })
        .instr(asm_read(gpr(6), base.clone()))
        .instr(Instruction::Load {
            ty: i64_ty(),
            dst: gpr(7),
            addr: slot_at(base.clone()),
            attrs: MemoryAttrs::default(),
        })
        .instr(Instruction::Load {
            ty: i64_ty(),
            dst: gpr(8),
            addr: slot_at(base),
            attrs: MemoryAttrs::default(),
        })
        .instr(Instruction::Ret {
            value: Some(Operand::Register(gpr(8))),
        })
        .build();

    apply_pass(&InstructionScheduling, &mut func);
    let instructions = &func.blocks[0].instructions;
    let asm_index = instructions
        .iter()
        .position(|inst| matches!(&inst.instr, Instruction::InlineAsm { .. }))
        .expect("asm stays in the block");
    assert_eq!(asm_index, 1);
}

#[test]
fn copy_propagation_rewrites_inline_asm_inputs() {
    let mut func = FunctionBuilder::new("copy")
        .param(gpr(0), i64_ty())
        .returns(i64_ty())
        .block("entry")
        .instr(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty(),
            dst: gpr(1),
            lhs: Operand::Register(gpr(0)),
            rhs: Operand::Immediate(Immediate::I64(0)),
        })
        .instr(asm_read(gpr(2), gpr(1)))
        .instr(Instruction::Ret {
            value: Some(Operand::Register(gpr(2))),
        })
        .build();

    assert!(apply_pass(&CopyPropagation, &mut func));
    match &func.blocks[0].instructions[1].instr {
        Instruction::InlineAsm { operands, .. } => {
            assert!(matches!(
                &operands[0],
                AsmOperand::Input {
                    value: Operand::Register(reg),
                    ..
                } if *reg == gpr(0)
            ));
        }
        other => panic!("expected inline asm, found {other}"),
    }
}
