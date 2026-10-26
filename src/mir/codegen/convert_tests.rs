#![allow(clippy::expect_used)]

use super::*;

fn lowered_global_module(source: &str) -> Result<Module, FromIRError> {
    let ir = parse_module(source).expect("parse globals");
    from_ir(&ir, "globals")
}

#[test]
fn global_initializers_lower_to_little_endian_bytes() {
    let module = lowered_global_module(
        "global @a: i64 = 41\nglobal @b: f64 = 2.5\nglobal @c: i32\n\
         fn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("lower globals");
    assert_eq!(
        module.globals["a"].initializer.as_deref(),
        Some(41i64.to_le_bytes().as_slice())
    );
    assert_eq!(
        module.globals["b"].initializer.as_deref(),
        Some(2.5f64.to_le_bytes().as_slice())
    );
    assert_eq!(module.globals["c"].initializer, None);
    assert!(!module.globals["a"].exported);
}

#[test]
fn exported_global_is_marked_exported() {
    let module = lowered_global_module(
        "@export global @g: i64 = 1\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("lower exported global");
    assert!(module.globals["g"].exported);
}

#[test]
fn global_address_is_materialized_once_at_entry() {
    let module = lowered_global_module(
        "global @g: i64 = 7\nfn @main() -> i64 {\nentry:\n %x = load.i64 @g\n store.i64 @g, %x\n %y = load.i64 @g\n ret.i64 %y\n}",
    )
    .expect("lower global loads and stores");
    let entry = &module.functions["main"].blocks[0].instructions;
    let symbol_addresses = entry
        .iter()
        .filter(|i| matches!(i, Instruction::SymbolAddress { symbol, .. } if symbol == "g"))
        .count();
    assert_eq!(symbol_addresses, 1);
    assert!(matches!(entry[0], Instruction::SymbolAddress { .. }));
}

#[test]
fn unknown_global_reference_is_rejected() {
    let error = lowered_global_module(
        "fn @main() -> i64 {\nentry:\n %x = load.i64 @missing\n ret.i64 %x\n}",
    )
    .expect_err("undeclared global");
    assert!(matches!(error, FromIRError::UnknownVariable));
}

#[test]
fn string_global_keeps_its_bytes_and_layout() {
    let module = lowered_global_module(
        "global @s: [4 x i8] = \"abc\"\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("string global");
    let global = &module.globals["s"];
    assert_eq!(global.initializer.as_deref(), Some(&b"abc\0"[..]));
    assert_eq!(global.size, 4);
    assert_eq!(global.align, 1);

    let error = lowered_global_module(
        "global @s: [3 x i8] = \"abc\"\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect_err("string without room for its terminator");
    assert!(matches!(error, FromIRError::UnsupportedGlobal(_)));
}

use crate::mir::TransformPipeline;
use crate::{
    ir::{
        FunctionParameter, IRBuilder,
        builder::{i64 as ir_i64, string, var},
    },
    parser::parse_module,
};

#[test]
fn select_literals_match_declared_widths() {
    for (ty, left, right) in [("i64", "11", "22"), ("f32", "1.5", "2.5")] {
        let source = format!(
            "fn @f(bool %c) -> {ty} {{ entry: %s = select.{ty} %c, {left}, {right} ret.{ty} %s }}"
        );
        let ir = parse_module(&source).expect("parse select");
        let module = from_ir(&ir, "select").expect("lower typed literals");
        assert!(module.validate_in_pipeline().is_ok());
    }
}

#[test]
fn bitcasts_preserve_source_bindings_and_define_distinct_results() {
    let ir = parse_module("fn @f(f64 %x) -> f64 { entry: %bits = bitcast.f64.i64 %x %back = bitcast.i64.f64 %bits ret.f64 %back }").expect("parse bitcasts");
    let module = from_ir(&ir, "bitcasts").expect("lower bitcasts");
    let function = &module.functions["f"];
    let instructions = &function.blocks[0].instructions;
    let Instruction::Bitcast { from, to, dst, src } = &instructions[0] else {
        panic!("expected bitcast")
    };
    assert_eq!((*from, *to), (MirType::f64(), MirType::i64()));
    assert_eq!(src, &Operand::reg(function.sig.params[0].reg.clone()));
    assert_ne!(dst, &function.sig.params[0].reg);
    let Instruction::Bitcast { dst: back, .. } = &instructions[1] else {
        panic!("expected round trip")
    };
    assert_ne!(dst, back);
    assert!(module.validate_in_pipeline().is_ok());
}

#[test]
fn lowering_rejects_undefined_and_forward_uses_before_initialization() {
    for source in [
        "fn @f() -> i64 { entry: %x = add.i64 %missing, 1 ret.i64 %x }",
        "fn @f() -> i64 { entry: %x = add.i64 %later, 1 %later = add.i64 1, 2 ret.i64 %x }",
    ] {
        let rejected = match parse_module(source) {
            Err(_) => true,
            Ok(ir) => matches!(from_ir(&ir, "invalid"), Err(FromIRError::InvalidMir(_))),
        };
        assert!(rejected, "{source}");
    }
}

#[test]
fn composite_gep_scales_constant_and_variable_indices() {
    for (element_type, stride) in [("[3 x i32]", 12), ("struct { x: i64, y: i64 }", 16)] {
        let source = format!(
            "fn @gep(ptr %p, i64 %index) -> ptr {{
entry:
%constant = getelem.ptr %p, 2, {element_type}
%variable = getelem.ptr %constant, %index, {element_type}
ret.ptr %variable
}}"
        );
        let ir = parse_module(&source).expect("parse composite elements");
        let entry = &ir.functions["gep"].basic_blocks["entry"];
        assert_eq!(
            entry.instructions[0].to_string(),
            format!("%constant = getelem.ptr %p, 2, {element_type}")
        );
        let mir = from_ir(&ir, "gep").expect("lower composite elements");
        mir.validate().expect("composite addresses verify");
        let instructions = &mir.get_function("gep").expect("gep function").blocks[0].instructions;
        assert!(instructions.iter().any(|inst| matches!(
            inst,
            Instruction::Lea { offset, .. } if i64::from(*offset) == 2 * stride
        )));
        assert!(instructions.iter().any(|inst| matches!(
            inst,
            Instruction::IntBinary {
                op: IntBinOp::Mul,
                rhs: Operand::Immediate(Immediate::I64(size)),
                ..
            } if *size == stride
        )));
    }
}

#[test]
fn constant_gep_preserves_pointer_width_offsets() {
    for (index, expected) in [
        (4294967297_i64, 34359738376_i64),
        (536870912, 4294967296),
        (-536870913, -4294967304),
        (i64::MIN, 0),
    ] {
        let source = format!(
            "fn @gep(ptr %p) -> ptr {{\nentry:\n%q = getelem.ptr %p, {index}, i64\nret.ptr %q\n}}"
        );
        let ir = parse_module(&source).expect("parse wide index");
        let mir = from_ir(&ir, "gep").expect("lower wide index");
        mir.validate().expect("address arithmetic verifies");
        let function = mir.get_function("gep").expect("gep function");
        let offset = function.blocks[0]
            .instructions
            .iter()
            .find_map(|inst| match inst {
                Instruction::Lea { offset, .. } => Some(i64::from(*offset)),
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty,
                    rhs: Operand::Immediate(Immediate::I64(offset)),
                    ..
                } if *ty == MirType::i64() => Some(*offset),
                _ => None,
            });
        assert_eq!(offset, Some(expected), "index {index}");
    }
}

#[test]
fn arithmetic_optimizations_preserve_immediate_widths() {
    let source = "fn @optimized(i32 %x) -> i32 {\nentry:\n%c = add.i32 1, 2\n%m = mul.i32 %x, 4\n%r = rem.u32 %x, 8\n%a = add.i32 %m, %r\n%b = add.i32 %a, %c\nret.i32 %b\n}";
    let ir = parse_module(source).expect("parse arithmetic");
    for level in 0..=3 {
        let mut mir = from_ir(&ir, "optimized").expect("lower arithmetic");
        TransformPipeline::default_for_opt_level(level)
            .apply_to_module(&mut mir)
            .expect("optimize arithmetic");
        mir.validate().expect("optimized widths match consumers");
    }
}

#[test]
fn conversions_and_variable_addresses_verify() {
    let ir = parse_module(
        "fn @convert(ptr %p, i32 %index, i8 %byte) -> ptr {\nentry:\n%wide = zext.i8.i64 %byte\n%signed = sext.i8.i64 %byte\n%narrow = trunc.i64.i32 %wide\n%address = ptrtoint %p, .i64\n%back = inttoptr %address, .i64\n%element = getelem.ptr %back, %index, i32\nret.ptr %element\n}",
    ).expect("parse conversions");
    let mir = from_ir(&ir, "conversions").expect("lower conversions");
    mir.validate().expect("conversion types match definitions");
}

#[test]
fn straight_line_reassignment_reads_previous_definition() {
    let ir = parse_module(
        "fn @sum() -> i64 {\nentry:\n%total = add.i64 1, 2\n%total = add.i64 %total, 3\n%total = add.i64 %total, 4\nret.i64 %total\n}",
    ).expect("parse reassignments");
    let mir = from_ir(&ir, "sum").expect("lower reassignments");
    mir.validate()
        .expect("reassignments have unique definitions");
    let function = mir.get_function("sum").expect("sum function");
    let instructions = &function.blocks[0].instructions;
    for pair in instructions[..3].windows(2) {
        let Instruction::IntBinary {
            lhs: Operand::Register(source),
            dst,
            ..
        } = &pair[1]
        else {
            panic!("expected dependent add");
        };
        assert_eq!(Some(source), pair[0].def_reg());
        assert_ne!(source, dst);
    }
}

#[test]
fn integer_aggregate_parameter_arrives_as_pointer_to_its_own_slot() {
    let ir = parse_module(
        "fn @sum(struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n%ap = getfieldptr %p, 0\n%a = load.i64 %ap\nret.i64 %a\n}",
    )
    .expect("parse aggregate parameter");
    let mir = from_ir(&ir, "sum").expect("lower aggregate parameter");
    let function = mir.get_function("sum").expect("sum function");
    let parameter = &function.sig.params[0];
    assert_eq!(parameter.ty, MirType::ptr());
    let aggregate = parameter.aggregate.expect("aggregate marker");
    assert_eq!(aggregate.size, 16);
    assert_eq!(aggregate.words(), 2);
    let Register::Virtual(slot) = &parameter.reg else {
        panic!("expected virtual slot");
    };
    assert_eq!(function.stack_reservations[slot], 1);
    let Instruction::StackAddr {
        slot: addressed, ..
    } = &function.blocks[0].instructions[0]
    else {
        panic!("expected stack address prelude");
    };
    assert_eq!(addressed, slot);
}

#[test]
fn aggregate_of_any_size_carries_its_layout() {
    let wide = parse_module(
        "fn @make() -> struct { a: i64, b: i64, c: i64 } {\nentry:\n%p = alloc.stack struct { a: i64, b: i64, c: i64 }\nret.ptr %p\n}",
    )
    .expect("parse wide aggregate return");
    let module = from_ir(&wide, "make").expect("wide aggregate return");
    let aggregate = module.functions["make"]
        .sig
        .ret_aggregate
        .expect("return aggregate");
    assert_eq!(aggregate.size, 24);

    let taken = parse_module(
        "fn @take(struct { a: i64, b: i64, c: i64 } %p) -> i64 {\nentry:\nret.i64 0\n}",
    )
    .expect("parse wide aggregate");
    let module = from_ir(&taken, "take").expect("wide aggregate parameter");
    let aggregate = module.functions["take"].sig.params[0]
        .aggregate
        .expect("aggregate marker");
    assert_eq!(aggregate.size, 24);
    assert_eq!(aggregate.words(), 3);
}

#[test]
fn float_and_oversized_aggregates_are_rejected_by_name() {
    let float =
        parse_module("fn @take(struct { a: f64, b: f64 } %p) -> i64 {\nentry:\nret.i64 0\n}")
            .expect("parse float aggregate");
    let error = from_ir(&float, "take").expect_err("float aggregate rejected");
    assert!(
        error.to_string().contains("%p of @take has a float field"),
        "{error}"
    );

    let returned = parse_module(
        "fn @make() -> struct { a: f64, b: f64 } {\nentry:\n%p = alloc.stack struct { a: f64, b: f64 }\nret.ptr %p\n}",
    )
    .expect("parse float aggregate return");
    let error = from_ir(&returned, "make").expect_err("float aggregate return rejected");
    assert!(
        error
            .to_string()
            .contains("return of @make has a float field"),
        "{error}"
    );
}

#[test]
fn integer_aggregate_return_lands_in_slot_the_caller_addresses() {
    let ir = parse_module(
        "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n%p = alloc.stack struct { a: i64, b: i64 }\nret.ptr %p\n}\nfn @use() -> i64 {\nentry:\n%r = call @make()\n%ap = getfieldptr %r, 0\n%a = load.i64 %ap\nret.i64 %a\n}",
    )
    .expect("parse aggregate return");
    let mir = from_ir(&ir, "returns").expect("lower aggregate return");

    let make = mir.get_function("make").expect("make function");
    assert_eq!(make.sig.ret_ty, Some(MirType::ptr()));
    let aggregate = make.sig.ret_aggregate.expect("return marker");
    assert_eq!(aggregate.words(), 2);

    let caller = mir.get_function("use").expect("use function");
    let Instruction::Call {
        ret: Some(slot), ..
    } = &caller.blocks[0].instructions[0]
    else {
        panic!("expected call");
    };
    let Instruction::StackAddr {
        slot: addressed, ..
    } = &caller.blocks[0].instructions[1]
    else {
        panic!("expected stack address after call");
    };
    assert_eq!(&Register::Virtual(*addressed), slot);
    assert_eq!(caller.stack_reservations[addressed], 1);
}

#[test]
fn width_conversions_reject_equal_widths_and_wrong_directions() {
    for name in ["sext", "trunc", "zext"] {
        let source =
            format!("fn @same(i64 %x) -> i64 {{\nentry:\n%a = {name}.i64.i64 %x\nret.i64 %a\n}}");
        let error = parse_module(&source).expect_err("equal width conversion rejected");
        assert!(
            error
                .to_string()
                .contains("Source and target types must be different"),
            "{name}: {error}"
        );
    }

    // bool is one bit, so widening it to a byte is a real conversion.
    let boolean = parse_module(
        "fn @flag(bool %b) -> i64 {\nentry:\n%a = zext.bool.i8 %b\n%w = zext.i8.i64 %a\nret.i64 %w\n}",
    )
    .expect("parse bool widening");
    let mir = from_ir(&boolean, "flag").expect("lower bool widening");
    let function = mir.get_function("flag").expect("flag function");
    assert!(
        matches!(
            function.blocks[0].instructions[0],
            Instruction::IntConvert { .. }
        ),
        "{:?}",
        function.blocks[0].instructions[0]
    );

    let narrowing =
        parse_module("fn @narrow(i64 %x) -> i64 {\nentry:\n%a = zext.i64.i32 %x\nret.i64 0\n}")
            .expect("parse narrowing zext");
    let error = from_ir(&narrowing, "narrow").expect_err("narrowing zext rejected");
    assert!(
        error.to_string().contains("zext from i64 to i32 narrows"),
        "{error}"
    );

    let widening =
        parse_module("fn @widen(i32 %x) -> i64 {\nentry:\n%a = trunc.i32.i64 %x\nret.i64 0\n}")
            .expect("parse widening trunc");
    let error = from_ir(&widening, "widen").expect_err("widening trunc rejected");
    assert!(
        error.to_string().contains("trunc from i32 to i64 widens"),
        "{error}"
    );
}

#[test]
fn padded_struct_allocations_use_abi_size() {
    let ir = parse_module(
        "fn @sizes() -> ptr {\nentry:\n%stack = alloc.stack struct { a: i8, b: i64, c: i8 }\n%heap = alloc.heap [2 x struct { a: i8, b: i64, c: i8 }]\nret.ptr %heap\n}",
    ).expect("parse padded allocations");
    let mir = from_ir(&ir, "sizes").expect("lower padded allocations");
    let function = mir.get_function("sizes").expect("sizes function");
    let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[0] else {
        panic!("expected stack allocation");
    };
    assert_eq!(function.stack_reservations[slot], 2);
    let Instruction::Call { args, .. } = &function.blocks[0].instructions[1] else {
        panic!("expected heap allocation");
    };
    assert_eq!(args, &[Operand::Immediate(Immediate::I64(48))]);
}

#[test]
fn stack_storage_is_not_a_register_use() {
    let ir =
        parse_module("fn @stack() -> ptr {\nentry:\n%p = alloc.stack [24 x i8]\nret.ptr %p\n}")
            .expect("parse stack allocation");
    let mir = from_ir(&ir, "stack").expect("lower stack allocation");
    mir.validate().expect("stack allocation verifies");
    let function = mir.get_function("stack").expect("stack function");
    let instruction = &function.blocks[0].instructions[0];
    let Instruction::StackAddr { slot, .. } = instruction else {
        panic!("expected stack address");
    };
    assert!(instruction.use_regs().is_empty());
    assert_eq!(function.stack_reservations[slot], 2);
}

#[test]
fn integer_immediates_follow_consuming_type() {
    for ty in ["i8", "i16", "i32", "i64"] {
        let source = format!(
            "fn @check(ptr %p) -> {ty} {{\nentry:\n%x = add.{ty} 1, -1\n%c = eq.{ty} %x, 0\nstore.{ty} %p, 7\nret.{ty} 0\n}}"
        );
        let ir = parse_module(&source).expect("parse typed operands");
        let mir = from_ir(&ir, "widths").expect("lower typed operands");
        mir.validate().expect("immediates match consuming types");
    }
}

#[test]
fn test_from_ir_simple_add() {
    let mut builder = IRBuilder::new();
    builder
        .function_with_params(
            "add",
            vec![
                FunctionParameter {
                    name: "a",
                    ty: IRType::Primitive(IRPrim::I64),
                    annotations: vec![],
                },
                FunctionParameter {
                    name: "b",
                    ty: IRType::Primitive(IRPrim::I64),
                    annotations: vec![],
                },
            ],
            IRType::Primitive(IRPrim::I64),
        )
        .binary(IRBin::Add, "sum", IRPrim::I64, var("a"), var("b"))
        .ret(IRType::Primitive(IRPrim::I64), var("sum"));

    let ir_module = builder.build();
    let mir_module = from_ir(&ir_module, "test").expect("from_ir should succeed");
    let func = mir_module.get_function("add").expect("function exists");
    assert_eq!(func.sig.name, "add");
    assert_eq!(func.entry, "entry");
    let entry = func.get_block("entry").expect("entry block exists");
    assert_eq!(entry.instructions.len(), 2);
    match &entry.instructions[0] {
        Instruction::IntBinary {
            op,
            ty,
            dst: _,
            lhs,
            rhs,
        } => {
            assert!(matches!(op, IntBinOp::Add));
            assert!(matches!(ty, MirType::Scalar(ScalarType::I64)));
            assert!(matches!(lhs, Operand::Register(_)));
            assert!(matches!(rhs, Operand::Register(_)));
        }
        other => panic!("Unexpected first instruction: {:?}", other),
    }
    match &entry.instructions[1] {
        Instruction::Ret { value } => {
            assert!(matches!(value, Some(Operand::Register(_))));
        }
        other => panic!("Unexpected second instruction: {:?}", other),
    }
}

#[test]
fn f64_phi_immediate_survives_lowering_as_copy() {
    let negative_zero = IRVal::Constant(IRLit::F64(-0.0));
    let other = IRVal::Constant(IRLit::F64(3.5));
    let mut builder = IRBuilder::new();
    builder
        .function_with_params(
            "choose",
            vec![FunctionParameter {
                name: "condition",
                ty: IRType::Primitive(IRPrim::I8),
                annotations: vec![],
            }],
            IRType::Primitive(IRPrim::F64),
        )
        .branch(var("condition"), "left", "right")
        .block("left")
        .jump("merge")
        .block("right")
        .jump("merge")
        .block("merge")
        .phi(
            "value",
            IRType::Primitive(IRPrim::F64),
            vec![(negative_zero, "left"), (other, "right")],
        )
        .ret(IRType::Primitive(IRPrim::F64), var("value"));

    let mir_module = from_ir(&builder.build(), "test").expect("from_ir should succeed");
    let function = mir_module
        .get_function("choose")
        .expect("function should exist");
    assert!(
        function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|instruction| matches!(
                instruction,
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::F64),
                    src: Operand::Immediate(Immediate::F64(value)),
                    ..
                } if value.to_bits() == (-0.0f64).to_bits()
            ))
    );
}

#[test]
fn test_print_numeric_succeeds() {
    let mut builder = IRBuilder::new();
    builder
        .function("main", IRType::Primitive(IRPrim::I64))
        .print(ir_i64(42))
        .ret(IRType::Primitive(IRPrim::I64), ir_i64(0));

    let ir_module = builder.build();
    let result = from_ir(&ir_module, "test");
    assert!(
        result.is_ok(),
        "print with numeric literal should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_print_string_fails() {
    let mut builder = IRBuilder::new();
    builder
        .function("main", IRType::Primitive(IRPrim::I64))
        .print(string("hello"))
        .ret(IRType::Primitive(IRPrim::I64), ir_i64(0));

    let ir_module = builder.build();
    let result = from_ir(&ir_module, "test");
    assert!(
        matches!(result, Err(FromIRError::PrintStringLiteralUnsupported)),
        "print with string literal should fail with PrintStringLiteralUnsupported: {:?}",
        result
    );
}

#[test]
fn test_print_string_rejected_text_ir() {
    let ir_source = r#"
fn @main() -> i64 {
  entry:
print "x"
ret.i64 0
}
"#;
    let ir_module = parse_module(ir_source).expect("parse should succeed");
    let result = from_ir(&ir_module, "test");
    assert!(
        matches!(result, Err(FromIRError::PrintStringLiteralUnsupported)),
        "print \"x\" in text IR should fail: {:?}",
        result
    );
    if let Err(e) = &result {
        let msg = format!("{}", e);
        assert!(
            msg.contains("writebyte"),
            "error should mention writebyte: {}",
            msg
        );
    }
}

#[test]
fn unsupported_instruction_names_its_block_and_function() {
    let ir = parse_module("fn @main() -> i64 {\nentry:\n%v = load.i64 5\nret.i64 %v\n}")
        .expect("parse reversed load");
    let error = from_ir(&ir, "reversed load").expect_err("lowering rejects reversed load");
    let text = error.to_string();
    assert!(text.contains("load.i64 5"), "{text}");
    assert!(text.contains("entry"), "{text}");
    assert!(text.contains("main"), "{text}");
}
