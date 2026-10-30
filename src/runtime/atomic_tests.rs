use super::integer_tests::{TestResult, available, directory, imm, module, reg, run, targets};
use super::interpret_mir_function;
use crate::{
    mir::{
        AddressMode, AddressSpace, AtomicBinOp, Block, Instruction, MemoryAttrs, MemoryOrdering,
        MirType, Module, Operand, Register, ScalarType, VirtualReg,
    },
    mir_codegen::RegallocStrategy,
};
use lamina_platform::PointerWidth;
use std::{error::Error, path::Path};

const INIT: i64 = 100;
const OPERAND: i64 = 27;

fn int_types() -> [MirType; 4] {
    [
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::I16),
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I64),
    ]
}

fn narrow(ty: MirType, value: i64) -> i64 {
    ty.signed_integer(value, PointerWidth::Bits64)
}

fn largest(ty: MirType) -> i64 {
    narrow(ty, i64::MAX)
}

fn cell_address(base: &Register) -> AddressMode {
    AddressMode::BaseOffset {
        address_space: AddressSpace::default(),
        base: base.clone(),
        offset: 0,
    }
}

fn cell(block: &mut Block, ty: MirType, ptr: &Register, init: i64) {
    block.push(Instruction::StackAddr {
        dst: ptr.clone(),
        slot: VirtualReg::gpr(50),
    });
    block.push(Instruction::Store {
        ty,
        src: imm(init),
        addr: cell_address(ptr),
        attrs: MemoryAttrs::default(),
    });
}

fn returning(block: &mut Block, value: Register) {
    block.push(Instruction::Ret {
        value: Some(Operand::Register(value)),
    });
}

fn rmw_program(ty: MirType, op: AtomicBinOp, init: i64, operand: i64, want_old: bool) -> Module {
    let ptr = reg(1);
    let old = reg(2);
    let new = reg(3);
    let mut block = Block::new("entry");
    cell(&mut block, ty, &ptr, init);
    block.push(Instruction::AtomicBinary {
        op,
        ty,
        dst: old.clone(),
        addr: cell_address(&ptr),
        value: imm(operand),
        ordering: MemoryOrdering::SeqCst,
    });
    block.push(Instruction::AtomicLoad {
        ty,
        dst: new.clone(),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::SeqCst,
    });
    returning(&mut block, if want_old { old } else { new });
    module(block)
}

fn cas_program(ty: MirType, init: i64, expected: i64, desired: i64, want_old: bool) -> Module {
    let ptr = reg(1);
    let old = reg(2);
    let new = reg(3);
    let mut block = Block::new("entry");
    cell(&mut block, ty, &ptr, init);
    block.push(Instruction::AtomicCompareExchange {
        ty,
        dst: old.clone(),
        addr: cell_address(&ptr),
        expected: imm(expected),
        desired: imm(desired),
        success_ordering: MemoryOrdering::SeqCst,
        failure_ordering: MemoryOrdering::SeqCst,
    });
    block.push(Instruction::AtomicLoad {
        ty,
        dst: new.clone(),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::SeqCst,
    });
    returning(&mut block, if want_old { old } else { new });
    module(block)
}

fn expected_rmw(ty: MirType, op: AtomicBinOp, old: i64, operand: i64) -> i64 {
    let raw = match op {
        AtomicBinOp::Add => old.wrapping_add(operand),
        AtomicBinOp::Sub => old.wrapping_sub(operand),
        AtomicBinOp::And => old & operand,
        AtomicBinOp::Or => old | operand,
        AtomicBinOp::Xor => old ^ operand,
        AtomicBinOp::Exchange => operand,
        AtomicBinOp::CompareExchange => unreachable!("not a binary operation"),
    };
    narrow(ty, raw)
}

fn interpreted(module: &Module) -> Result<Option<i64>, Box<dyn Error>> {
    Ok(interpret_mir_function(&module.functions["main"], &[])?)
}

fn native_matches(module: &Module, directory: &Path, expected: i64) -> TestResult {
    for target in targets(false) {
        let output = run(module, &target, RegallocStrategy::Incremental, directory)?;
        assert_eq!(
            output.status.code(),
            Some((expected & 0xff) as i32),
            "{} expected {expected}: {output:?}",
            target.triple
        );
    }
    Ok(())
}

#[test]
fn atomic_rmw_returns_old_value_and_stores_new_value() -> TestResult {
    let native = available("cc");
    let guard = directory("atomic-rmw");
    let directory = guard.path();
    for ty in int_types() {
        for op in [
            AtomicBinOp::Add,
            AtomicBinOp::Sub,
            AtomicBinOp::And,
            AtomicBinOp::Or,
            AtomicBinOp::Xor,
            AtomicBinOp::Exchange,
        ] {
            for (init, operand) in [(INIT, OPERAND), (largest(ty), 1)] {
                let old_before = narrow(ty, init);
                let new_after = expected_rmw(ty, op, init, operand);
                let want_old = rmw_program(ty, op, init, operand, true);
                let want_new = rmw_program(ty, op, init, operand, false);
                assert_eq!(interpreted(&want_old)?, Some(old_before), "{ty:?} {op} old");
                assert_eq!(interpreted(&want_new)?, Some(new_after), "{ty:?} {op} new");
                if native {
                    native_matches(&want_old, directory, old_before)?;
                    native_matches(&want_new, directory, new_after)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn atomic_add_returns_old_not_new() -> TestResult {
    let module = rmw_program(
        MirType::Scalar(ScalarType::I64),
        AtomicBinOp::Add,
        INIT,
        OPERAND,
        true,
    );
    assert_eq!(interpreted(&module)?, Some(INIT));
    let module = rmw_program(
        MirType::Scalar(ScalarType::I64),
        AtomicBinOp::Add,
        INIT,
        OPERAND,
        false,
    );
    assert_eq!(interpreted(&module)?, Some(INIT + OPERAND));
    Ok(())
}

#[test]
fn atomic_cmpxchg_returns_old_value_and_swaps_only_on_match() -> TestResult {
    let native = available("cc");
    let guard = directory("atomic-cmpxchg");
    let directory = guard.path();
    for ty in int_types() {
        let old_before = narrow(ty, INIT);
        for (expected, desired, new_after) in
            [(INIT, 77, narrow(ty, 77)), (INIT + 1, 77, old_before)]
        {
            let want_old = cas_program(ty, INIT, expected, desired, true);
            let want_new = cas_program(ty, INIT, expected, desired, false);
            assert_eq!(interpreted(&want_old)?, Some(old_before), "{ty:?} old");
            assert_eq!(interpreted(&want_new)?, Some(new_after), "{ty:?} new");
            if native {
                native_matches(&want_old, directory, old_before)?;
                native_matches(&want_new, directory, new_after)?;
            }
        }
    }
    Ok(())
}

#[test]
fn atomic_store_load_and_fence_round_trip() -> TestResult {
    let ty = MirType::Scalar(ScalarType::I16);
    let ptr = reg(1);
    let loaded = reg(2);
    let mut block = Block::new("entry");
    cell(&mut block, ty, &ptr, 0);
    block.push(Instruction::AtomicStore {
        ty,
        src: imm(0x1234),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::Release,
    });
    block.push(Instruction::Fence {
        ordering: MemoryOrdering::SeqCst,
    });
    block.push(Instruction::AtomicLoad {
        ty,
        dst: loaded.clone(),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::Acquire,
    });
    returning(&mut block, loaded);
    let module = module(block);
    assert_eq!(interpreted(&module)?, Some(0x1234));
    Ok(())
}

#[test]
fn atomic_orderings_the_backends_refuse_are_refused() -> TestResult {
    let ty = MirType::Scalar(ScalarType::I64);
    let ptr = reg(1);
    let value = reg(2);
    let mut block = Block::new("entry");
    cell(&mut block, ty, &ptr, 0);
    block.push(Instruction::AtomicLoad {
        ty,
        dst: value.clone(),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::Release,
    });
    returning(&mut block, value);
    assert!(interpreted(&module(block)).is_err());

    let mut block = Block::new("entry");
    cell(&mut block, ty, &ptr, 0);
    block.push(Instruction::AtomicStore {
        ty,
        src: imm(1),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::AcqRel,
    });
    block.push(Instruction::Ret { value: None });
    assert!(interpreted(&module(block)).is_err());
    Ok(())
}

#[test]
fn atomic_access_through_null_traps() -> TestResult {
    let ty = MirType::Scalar(ScalarType::I64);
    let ptr = reg(1);
    let value = reg(2);
    let mut block = Block::new("entry");
    block.push(Instruction::Copy {
        ty,
        dst: ptr.clone(),
        src: imm(0),
    });
    block.push(Instruction::AtomicLoad {
        ty,
        dst: value.clone(),
        addr: cell_address(&ptr),
        ordering: MemoryOrdering::SeqCst,
    });
    returning(&mut block, value);
    assert!(interpreted(&module(block)).is_err());
    Ok(())
}
