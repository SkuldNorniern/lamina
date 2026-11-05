use super::*;
use crate::mir::codegen::from_ir_for_target;
use crate::mir::{Aggregate, AggregateClass, AggregateShape, MirType};
use crate::parser::parse_module;
use abi::{ArgKind, classify, returns_in_memory, stack_bytes};
use lamina_mir::ScalarType;

fn integer() -> ArgKind {
    ArgKind::of(MirType::Scalar(ScalarType::I32), None)
}

fn single() -> ArgKind {
    ArgKind::of(MirType::Scalar(ScalarType::F32), None)
}

fn double() -> ArgKind {
    ArgKind::of(MirType::Scalar(ScalarType::F64), None)
}

fn composite(size: usize, align: usize) -> Aggregate {
    Aggregate {
        size,
        align,
        class: AggregateClass::Integer,
        shape: AggregateShape::Composite,
    }
}

fn wide_scalar() -> Aggregate {
    Aggregate {
        size: 8,
        align: 8,
        class: AggregateClass::Integer,
        shape: AggregateShape::WideScalar,
    }
}

fn aggregate(size: usize, align: usize) -> ArgKind {
    ArgKind::of(
        MirType::Scalar(ScalarType::Ptr),
        Some(composite(size, align)),
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
    let mir = from_ir_for_target(
        &module,
        "armv7_test",
        TargetArchitecture::Armv7.data_layout(),
    )
    .expect("lower");
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
    let error = generate_mir_armv7(&module, &mut out, TargetOperatingSystem::Linux)
        .expect_err("a 64-bit multiply has no armv7 lowering");
    let message = error.to_string();
    assert!(message.contains("i64"), "{message}");
    assert!(message.contains("@wide"), "{message}");
    assert!(message.contains("arithmetic"), "{message}");
}

#[test]
fn a_double_is_two_words_wide() {
    let kind = ArgKind::of(MirType::Scalar(ScalarType::F64), None);
    assert!(kind.double);
    assert_eq!(kind.words(), 2);
    assert_eq!(kind.align, 8);
}

#[test]
fn aapcs_returns_a_composite_over_a_word_in_memory_and_a_legalized_integer_in_registers() {
    assert!(!returns_in_memory(Some(composite(4, 4))));
    assert!(returns_in_memory(Some(composite(5, 1))));
    assert!(returns_in_memory(Some(composite(8, 4))));
    assert!(returns_in_memory(Some(composite(16, 8))));
    assert!(!returns_in_memory(Some(wide_scalar())));
    assert!(!returns_in_memory(None));
}

#[test]
fn a_composite_result_over_a_word_takes_a_hidden_pointer_in_r0() {
    let assembly = compile(
        "fn @make(i32 %x) -> struct { a: i32, b: i32 } {\nentry:\n  \
         %p = stackalloc.struct { a: i32, b: i32 }\n  \
         %pa = getfieldptr.struct { a: i32, b: i32 } %p, 0\n  \
         store.i32 %pa, %x\n  ret.struct { a: i32, b: i32 } %p\n}\n",
    );
    assert!(assembly.contains("str r1, ["), "{assembly}");
    assert!(!assembly.contains("ldr r1, ["), "{assembly}");
}

#[test]
fn a_composite_result_of_one_word_still_comes_back_in_r0() {
    let assembly = compile(
        "fn @make(i32 %x) -> struct { a: i32 } {\nentry:\n  \
         %p = stackalloc.struct { a: i32 }\n  \
         %pa = getfieldptr.struct { a: i32 } %p, 0\n  \
         store.i32 %pa, %x\n  ret.struct { a: i32 } %p\n}\n",
    );
    assert!(!assembly.contains("str r1, ["), "{assembly}");
}

#[test]
fn a_result_of_five_bytes_is_copied_a_byte_at_a_time_past_the_first_word() {
    let assembly = compile(
        "fn @make() -> [5 x i8] {\nentry:\n  %p = stackalloc.[5 x i8]\n  \
         %e = getelem.i8 %p, 0\n  store.i8 %e, 3\n  ret.[5 x i8] %p\n}\n",
    );
    assert_eq!(assembly.matches("ldr r12, [").count(), 1, "{assembly}");
    assert_eq!(assembly.matches("ldrb r12, [").count(), 1, "{assembly}");
}

/// Lamina compiles the compiler runtime helpers for this target itself, with this
/// module's own variant, so a call to `__fixdfdi` hands its double to d0. Only the
/// ARM run-time ABI's own `__aeabi_*` helpers are fixed to the base standard.
#[test]
fn a_compiler_runtime_helper_is_called_with_the_module_s_own_float_variant() {
    let assembly =
        compile("fn @f(f64 %x) -> i64 {\nentry:\n  %r = fptoint.f64.i64 %x\n  ret.i64 %r\n}\n");
    assert!(assembly.contains("bl __fixdfdi"), "{assembly}");
    let call = assembly
        .split("bl __fixdfdi")
        .next()
        .unwrap_or_default()
        .to_string();
    assert!(
        !call.contains("vmov r0, r1, d"),
        "the double was handed to __fixdfdi in core registers:\n{assembly}"
    );
}

#[test]
fn an_aeabi_helper_still_reads_the_base_standard() {
    let signatures = abi::Signatures::from_module(
        &from_ir_for_target(
            &parse_module("fn @main() -> i32 {\nentry:\n  ret.i32 0\n}\n").expect("parse"),
            "armv7_test",
            TargetArchitecture::Armv7.data_layout(),
        )
        .expect("lower"),
    );
    assert!(signatures.uses_base_standard("__aeabi_d2lz"));
    assert!(!signatures.uses_base_standard("__fixdfdi"));
}

const WIDE_IN_THE_VARIADIC_TAIL: &str = "@extern fn @printf(ptr %fmt, ...) -> i32\n\
     global $fmt: [6 x i8] = \"%lld\\n\"\n\
     fn @main() -> i32 {\nentry:\n  %x = add.i64 1, 2\n  \
     %r = call @printf(ptr $fmt, i64 %x)\n  ret.i32 %r\n}\n";

#[test]
fn a_legalized_pair_in_the_variadic_tail_travels_as_its_value_in_an_even_core_pair() {
    let assembly = compile(WIDE_IN_THE_VARIADIC_TAIL);
    assert!(assembly.contains("ldr r2, ["), "{assembly}");
    assert!(assembly.contains("ldr r3, ["), "{assembly}");
    assert!(!assembly.contains("ldr r1, ["), "{assembly}");
}

#[test]
fn a_named_wide_parameter_of_a_variadic_callee_travels_as_its_value() {
    let assembly = compile(
        "@extern fn @take(i64 %a, ...) -> i32\n\
         fn @main() -> i32 {\nentry:\n  %x = add.i64 1, 2\n  \
         %r = call @take(i64 %x, i32 5)\n  ret.i32 %r\n}\n",
    );
    assert!(assembly.contains("ldr r0, ["), "{assembly}");
    assert!(assembly.contains("ldr r1, ["), "{assembly}");
    assert!(
        assembly.contains("mov r2, #5") || assembly.contains("movw r2, #5"),
        "{assembly}"
    );
}

const WIDE_VA_ARG_AFTER_A_WORD: &str = "fn @read(i32 %n, ...) -> i64 {\nentry:\n  \
     %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  \
     va_end %ap\n  ret.i64 %x\n}\n";

#[test]
fn a_variadic_callee_saves_the_core_argument_registers_below_the_stack_arguments() {
    let assembly = compile(WIDE_VA_ARG_AFTER_A_WORD);
    assert!(
        assembly.contains("    push {r0, r1, r2, r3}\n"),
        "{assembly}"
    );
    assert!(
        assembly.contains("    .cfi_def_cfa_offset 120\n"),
        "{assembly}"
    );
}

#[test]
fn va_start_points_past_the_named_parameters_in_the_save_area() {
    let assembly = compile(WIDE_VA_ARG_AFTER_A_WORD);
    assert!(assembly.contains(", r10, #108\n"), "{assembly}");
}

#[test]
fn va_arg_of_a_pair_rounds_the_cursor_up_to_eight_and_steps_by_eight() {
    let assembly = compile(WIDE_VA_ARG_AFTER_A_WORD);
    let rounds = assembly
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("bic r") && line.ends_with("#7"))
        .count();
    assert_eq!(rounds, 1, "{assembly}");
    assert!(assembly.contains(", #8\n"), "{assembly}");
}

#[test]
fn va_arg_of_a_word_does_not_round_the_cursor() {
    let assembly = compile(
        "fn @read(i32 %n, ...) -> i32 {\nentry:\n  %ap = stackalloc.valist\n  \
         va_start %ap\n  %x = va_arg.i32 %ap\n  va_end %ap\n  ret.i32 %x\n}\n",
    );
    assert!(
        !assembly
            .lines()
            .map(str::trim)
            .any(|line| line.starts_with("bic r") && line.ends_with("#7")),
        "{assembly}"
    );
}

#[test]
fn a_non_variadic_function_saves_no_core_argument_registers() {
    let assembly = compile("fn @id(i32 %n) -> i32 {\nentry:\n  ret.i32 %n\n}\n");
    assert!(!assembly.contains("push {r0, r1, r2, r3}"), "{assembly}");
    assert!(
        assembly.contains("    .cfi_def_cfa_offset 104\n"),
        "{assembly}"
    );
}

#[test]
fn frame_larger_than_a_page_is_probed() {
    let asm =
        compile("fn @f() -> i64 {\nentry:\n    %p = stackalloc.[65536 x i8]\n    ret.i64 0\n}\n");
    let probe = "1:\n    sub sp, sp, #4096\n    str r4, [sp]\n    cmp sp, r4\n    bhi 1b\n";
    assert!(asm.contains("    sub r4, sp, r4\n"), "{asm}");
    assert!(asm.contains(probe), "{asm}");
    let probe_at = asm.find(probe).expect("probe");
    let bic_at = asm.find("    bic sp, sp,").expect("bic");
    assert!(probe_at < bic_at, "{asm}");
}

#[test]
fn small_frame_gets_no_probe() {
    let asm =
        compile("fn @f() -> i64 {\nentry:\n    %p = stackalloc.[24 x i8]\n    ret.i64 0\n}\n");
    assert!(!asm.contains("bhi 1b"), "{asm}");
}
