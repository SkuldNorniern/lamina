use super::*;
use crate::mir::codegen::from_ir_for_target;
use crate::mir::{Aggregate, AggregateClass, MirType};
use crate::parser::parse_module;
use abi::{ArgKind, classify, stack_bytes};

fn integer() -> ArgKind {
    ArgKind::of(MirType::i32(), None)
}

fn single() -> ArgKind {
    ArgKind::of(MirType::f32(), None)
}

fn double() -> ArgKind {
    ArgKind::of(MirType::f64(), None)
}

fn aggregate(size: usize, align: usize) -> ArgKind {
    ArgKind::of(
        MirType::ptr(),
        Some(Aggregate {
            size,
            align,
            class: AggregateClass::Integer,
        }),
    )
}

#[test]
fn integers_take_the_core_registers_then_the_stack() {
    let kinds: Vec<ArgKind> = (0..6).map(|_| integer()).collect();
    let places = classify(&kinds, None, false);
    for (index, place) in places.iter().take(4).enumerate() {
        assert_eq!(place.core.expect("core").first, index);
        assert!(place.vfp.is_none());
    }
    assert_eq!(places[4].stack.expect("stack").offset, 0);
    assert_eq!(places[5].stack.expect("stack").offset, 4);
    assert_eq!(stack_bytes(&places), 8);
}

#[test]
fn floats_take_the_vfp_bank_and_leave_the_core_registers_alone() {
    let kinds = vec![integer(), single(), integer(), double(), single()];
    let places = classify(&kinds, None, false);
    assert_eq!(places[0].core.expect("r0").first, 0);
    assert_eq!(places[1].vfp, Some(0));
    assert_eq!(places[2].core.expect("r1").first, 1);
    // A double needs an even pair, so it skips s1 and takes s2:s3.
    assert_eq!(places[3].vfp, Some(2));
    // The single back-fills the slot the double skipped.
    assert_eq!(places[4].vfp, Some(1));
    assert!(places.iter().all(|place| place.stack.is_none()));
}

#[test]
fn the_vfp_bank_closes_once_an_argument_misses_it() {
    let mut kinds: Vec<ArgKind> = (0..8).map(|_| double()).collect();
    kinds.push(single());
    kinds.push(single());
    let places = classify(&kinds, None, false);
    for (index, place) in places.iter().take(8).enumerate() {
        assert_eq!(place.vfp, Some(index * 2));
    }
    // s0-s15 are all taken, so the singles go to the stack rather than to r0-r3.
    assert!(places[8].core.is_none());
    assert_eq!(places[8].stack.expect("stack").offset, 0);
    assert_eq!(places[9].stack.expect("stack").offset, 4);
}

#[test]
fn a_variadic_double_starts_on_an_even_core_register() {
    let kinds = vec![integer(), integer(), double()];
    let places = classify(&kinds, Some(1), false);
    assert_eq!(places[0].core.expect("r0").first, 0);
    assert_eq!(places[1].core.expect("r1").first, 1);
    // r1 is taken and the double needs an even pair, so r2:r3 carries it and r1 is
    // not followed by r1:r2.
    let core = places[2].core.expect("core pair");
    assert_eq!(core.first, 2);
    assert_eq!(core.words, 2);
    assert!(places[2].vfp.is_none());
}

#[test]
fn a_variadic_double_that_misses_the_core_registers_is_eight_byte_aligned_on_the_stack() {
    let kinds = vec![integer(), integer(), integer(), double()];
    let places = classify(&kinds, Some(1), false);
    assert!(places[3].core.is_none());
    assert_eq!(places[3].stack.expect("stack").offset, 0);
}

#[test]
fn an_eight_byte_aligned_aggregate_rounds_the_core_register_up_to_even() {
    let kinds = vec![integer(), aggregate(8, 8)];
    let places = classify(&kinds, None, false);
    let core = places[1].core.expect("core pair");
    assert_eq!(core.first, 2);
    assert_eq!(core.words, 2);
}

#[test]
fn a_composite_straddles_the_core_registers_and_the_stack() {
    let kinds = vec![integer(), integer(), aggregate(16, 4)];
    let places = classify(&kinds, None, false);
    let core = places[2].core.expect("core part");
    assert_eq!(core.first, 2);
    assert_eq!(core.words, 2);
    assert_eq!(places[2].stack.expect("stack part").bytes, 8);
}

fn compile(source: &str) -> String {
    let module = parse_module(source).expect("parse");
    let mir = from_ir_for_target(&module, "armv7_test", PointerWidth::Bits32).expect("lower");
    let mut out = Vec::new();
    generate_mir_armv7(&mir, &mut out, TargetOperatingSystem::Linux).expect("codegen");
    String::from_utf8(out).expect("utf8")
}

#[test]
fn the_module_header_selects_arm_state_and_the_vfp_argument_abi() {
    let assembly = compile("fn @main() -> i32 {\nentry:\n  ret.i32 0\n}\n");
    assert!(assembly.contains(".arm"), "{assembly}");
    assert!(assembly.contains(".fpu vfpv3-d16"), "{assembly}");
    assert!(assembly.contains(".eabi_attribute 28, 1"), "{assembly}");
    assert!(!assembly.contains("@progbits"), "{assembly}");
}

#[test]
fn a_float_argument_reaches_the_callee_in_a_vfp_register() {
    let assembly = compile(
        "fn @take(f32 %x) -> f32 {\nentry:\n  ret.f32 %x\n}\n\
         fn @main() -> i32 {\nentry:\n  %r = call @take(f32 2.5)\n  ret.i32 0\n}\n",
    );
    assert!(assembly.contains("vmov s0,"), "{assembly}");
    assert!(assembly.contains("vstr s0,"), "{assembly}");
}

#[test]
fn a_sixty_four_bit_integer_is_rejected_by_name() {
    use crate::mir::{Block, Immediate, IntBinOp, Operand, Register, VirtualReg};
    let dst = Register::Virtual(VirtualReg::gpr(0));
    let mut func = Function::new(Signature::new("wide"));
    let mut block = Block::new("entry");
    block.push(MirInst::IntBinary {
        op: IntBinOp::Mul,
        ty: MirType::i64(),
        dst,
        lhs: Operand::Immediate(Immediate::I64(3)),
        rhs: Operand::Immediate(Immediate::I64(4)),
    });
    block.push(MirInst::Ret { value: None });
    func.add_block(block);
    let mut module = MirModule::new("wide");
    module.functions.insert("wide".to_string(), func);

    let mut out = Vec::new();
    let error = generate_mir_armv7(&module, &mut out, TargetOperatingSystem::Linux)
        .expect_err("a 64-bit multiply has no armv7 lowering");
    let message = error.to_string();
    assert!(message.contains("i64"), "{message}");
    assert!(message.contains("@wide"), "{message}");
    assert!(message.contains("arithmetic"), "{message}");
}

#[test]
fn a_double_is_two_words_wide() {
    let kind = ArgKind::of(MirType::f64(), None);
    assert!(kind.double);
    assert_eq!(kind.words(), 2);
    assert_eq!(kind.align, 8);
}
