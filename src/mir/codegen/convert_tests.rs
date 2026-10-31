#![allow(clippy::expect_used)]

use super::*;
use crate::mir::{VectorLane, VectorType};

fn lowered_global_module(source: &str) -> Result<Module, FromIRError> {
    let ir = parse_module(source).expect("parse globals");
    from_ir_x86_64(&ir, "globals")
}

#[test]
fn global_initializers_lower_to_little_endian_bytes() {
    let module = lowered_global_module(
        "global $a: i64 = 41\nglobal $b: f64 = 2.5\nglobal $c: i32\n\
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
fn array_global_keeps_only_its_filled_prefix() {
    let module = lowered_global_module(
        "global $g: [10000000000 x i8] = [1, 2]\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("lower sparse array global");
    assert_eq!(
        module.globals["g"].initializer.as_deref(),
        Some([1u8, 2].as_slice())
    );
    assert_eq!(module.globals["g"].size, 10_000_000_000);
}

#[test]
fn exported_global_is_marked_exported() {
    let module = lowered_global_module(
        "@export global $g: i64 = 1\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("lower exported global");
    assert!(module.globals["g"].exported);
}

#[test]
fn global_address_is_materialized_once_at_entry() {
    let module = lowered_global_module(
        "global $g: i64 = 7\nfn @main() -> i64 {\nentry:\n %x = load.i64 $g\n store.i64 $g, %x\n %y = load.i64 $g\n ret.i64 %y\n}",
    )
    .expect("lower global loads and stores");
    let entry = &module.functions["main"].blocks[0].instructions;
    let symbol_addresses = entry
        .iter()
        .filter(|i| matches!(&i.instr, Instruction::SymbolAddress { symbol, .. } if symbol == "g"))
        .count();
    assert_eq!(symbol_addresses, 1);
    assert!(matches!(&entry[0].instr, Instruction::SymbolAddress { .. }));
}

#[test]
fn unknown_global_reference_is_rejected() {
    let error = lowered_global_module(
        "fn @main() -> i64 {\nentry:\n %x = load.i64 $missing\n ret.i64 %x\n}",
    )
    .expect_err("undeclared global");
    assert!(matches!(error, FromIRError::UnknownVariable));
}

#[test]
fn string_global_keeps_its_bytes_and_layout() {
    let module = lowered_global_module(
        "global $s: [4 x i8] = \"abc\"\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("string global");
    let global = &module.globals["s"];
    assert_eq!(global.initializer.as_deref(), Some(&b"abc\0"[..]));
    assert_eq!(global.size, 4);
    assert_eq!(global.align, 1);

    let error = lowered_global_module(
        "global $s: [3 x i8] = \"abc\"\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect_err("string without room for its terminator");
    assert!(matches!(error, FromIRError::UnsupportedGlobal(_)));
}

use crate::mir::TransformPipeline;
use crate::{
    ir::{FunctionParameter, IRBuilder, builder::var},
    parser::parse_module,
};

#[test]
fn select_literals_match_declared_widths() {
    for (ty, left, right) in [("i64", "11", "22"), ("f32", "1.5", "2.5")] {
        let source = format!(
            "fn @f(bool %c) -> {ty} {{ entry: %s = select.{ty} %c, {left}, {right} ret.{ty} %s }}"
        );
        let ir = parse_module(&source).expect("parse select");
        let module = from_ir_x86_64(&ir, "select").expect("lower typed literals");
        assert!(module.validate_in_pipeline().is_ok());
    }
}

#[test]
fn bitcasts_preserve_source_bindings_and_define_distinct_results() {
    let ir = parse_module("fn @f(f64 %x) -> f64 { entry: %bits = bitcast.f64.i64 %x %back = bitcast.i64.f64 %bits ret.f64 %back }").expect("parse bitcasts");
    let module = from_ir_x86_64(&ir, "bitcasts").expect("lower bitcasts");
    let function = &module.functions["f"];
    let instructions = &function.blocks[0].instructions;
    let Instruction::Bitcast { from, to, dst, src } = &instructions[0].instr else {
        panic!("expected bitcast")
    };
    assert_eq!(
        (*from, *to),
        (
            MirType::Scalar(ScalarType::F64),
            MirType::Scalar(ScalarType::I64)
        )
    );
    assert_eq!(src, &Operand::reg(function.sig.params[0].reg.clone()));
    assert_ne!(dst, &function.sig.params[0].reg);
    let Instruction::Bitcast { dst: back, .. } = &instructions[1].instr else {
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
            Ok(ir) => matches!(
                from_ir_x86_64(&ir, "invalid"),
                Err(FromIRError::InvalidMir(_))
            ),
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
%constant = getelem.{element_type} %p, 2
%variable = getelem.{element_type} %constant, %index
ret.ptr %variable
}}"
        );
        let ir = parse_module(&source).expect("parse composite elements");
        let entry = &ir.functions["gep"].basic_blocks["entry"];
        assert_eq!(
            entry.instructions[0].to_string(),
            format!("%constant = getelem.{element_type} %p, 2")
        );
        let mir = from_ir_x86_64(&ir, "gep").expect("lower composite elements");
        mir.validate().expect("composite addresses verify");
        let instructions = &mir.get_function("gep").expect("gep function").blocks[0].instructions;
        assert!(instructions.iter().any(|inst| matches!(
            &inst.instr,
            Instruction::Lea { offset, .. } if i64::from(*offset) == 2 * stride
        )));
        assert!(instructions.iter().any(|inst| matches!(
            &inst.instr,
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
    ] {
        let source = format!(
            "fn @gep(ptr %p) -> ptr {{\nentry:\n%q = getelem.i64 %p, {index}\nret.ptr %q\n}}"
        );
        let ir = parse_module(&source).expect("parse wide index");
        let mir = from_ir_x86_64(&ir, "gep").expect("lower wide index");
        mir.validate().expect("address arithmetic verifies");
        let function = mir.get_function("gep").expect("gep function");
        let offset = function.blocks[0]
            .instructions
            .iter()
            .find_map(|inst| match &inst.instr {
                Instruction::Lea { offset, .. } => Some(i64::from(*offset)),
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty,
                    rhs: Operand::Immediate(Immediate::I64(offset)),
                    ..
                } if *ty == MirType::Scalar(ScalarType::I64) => Some(*offset),
                _ => None,
            });
        assert_eq!(offset, Some(expected), "index {index}");
    }
}

#[test]
fn offsets_past_i32_reach_their_address() {
    let source = "fn @field(ptr %p) -> ptr {\nentry:\n%q = getfieldptr.struct { pad: [2400000000 x i8], tail: i64 } %p, 1\nret.ptr %q\n}";
    let ir = parse_module(source).expect("parse wide field index");
    let mir = from_ir_x86_64(&ir, "field").expect("lower wide field index");
    mir.validate().expect("address arithmetic verifies");
    let offset = mir.get_function("field").expect("field function").blocks[0]
        .instructions
        .iter()
        .find_map(|inst| match &inst.instr {
            Instruction::Lea { offset, .. } => Some(i64::from(*offset)),
            Instruction::IntBinary {
                op: IntBinOp::Add,
                ty,
                rhs: Operand::Immediate(Immediate::I64(offset)),
                ..
            } if *ty == MirType::Scalar(ScalarType::I64) => Some(*offset),
            _ => None,
        });
    assert_eq!(offset, Some(2_400_000_000));
}

#[test]
fn element_offset_that_overflows_is_rejected() {
    let source = format!(
        "fn @gep(ptr %p) -> ptr {{\nentry:\n%q = getelem.i64 %p, {}\nret.ptr %q\n}}",
        i64::MIN
    );
    let ir = parse_module(&source).expect("parse overflowing index");
    from_ir_x86_64(&ir, "gep").expect_err("offset does not fit a pointer");
}

#[test]
fn arithmetic_optimizations_preserve_immediate_widths() {
    let source = "fn @optimized(i32 %x) -> i32 {\nentry:\n%c = add.i32 1, 2\n%m = mul.i32 %x, 4\n%r = rem.u32 %x, 8\n%a = add.i32 %m, %r\n%b = add.i32 %a, %c\nret.i32 %b\n}";
    let ir = parse_module(source).expect("parse arithmetic");
    for level in 0..=3 {
        let mut mir = from_ir_x86_64(&ir, "optimized").expect("lower arithmetic");
        TransformPipeline::default_for_opt_level(level)
            .apply_to_module(&mut mir)
            .expect("optimize arithmetic");
        mir.validate().expect("optimized widths match consumers");
    }
}

#[test]
fn conversions_and_variable_addresses_verify() {
    let ir = parse_module(
        "fn @convert(ptr %p, i32 %index, i8 %byte) -> ptr {\nentry:\n%ubyte = bitcast.i8.u8 %byte\n%wide = extend.u8.i64 %ubyte\n%signed = extend.i8.i64 %byte\n%narrow = trunc.i64.i32 %wide\n%address = ptrtoint.i64 %p\n%back = inttoptr.i64 %address\n%element = getelem.i32 %back, %index\nret.ptr %element\n}",
    ).expect("parse conversions");
    let mir = from_ir_x86_64(&ir, "conversions").expect("lower conversions");
    mir.validate().expect("conversion types match definitions");
}

#[test]
fn straight_line_reassignment_reads_previous_definition() {
    let ir = parse_module(
        "fn @sum() -> i64 {\nentry:\n%total = add.i64 1, 2\n%total = add.i64 %total, 3\n%total = add.i64 %total, 4\nret.i64 %total\n}",
    ).expect("parse reassignments");
    let mir = from_ir_x86_64(&ir, "sum").expect("lower reassignments");
    mir.validate()
        .expect("reassignments have unique definitions");
    let function = mir.get_function("sum").expect("sum function");
    let instructions = &function.blocks[0].instructions;
    for pair in instructions[..3].windows(2) {
        let Instruction::IntBinary {
            lhs: Operand::Register(source),
            dst,
            ..
        } = &pair[1].instr
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
        "fn @sum(struct { a: i64, b: i64 } %p) -> i64 {\nentry:\n%ap = getfieldptr.struct { a: i64, b: i64 } %p, 0\n%a = load.i64 %ap\nret.i64 %a\n}",
    )
    .expect("parse aggregate parameter");
    let mir = from_ir_x86_64(&ir, "sum").expect("lower aggregate parameter");
    let function = mir.get_function("sum").expect("sum function");
    let parameter = &function.sig.params[0];
    assert_eq!(parameter.ty, MirType::Scalar(ScalarType::Ptr));
    let aggregate = parameter.aggregate.expect("aggregate marker");
    assert_eq!(aggregate.size, 16);
    assert_eq!(aggregate.words(), 2);
    let Register::Virtual(slot) = &parameter.reg else {
        panic!("expected virtual slot");
    };
    assert_eq!(function.stack_reservations[slot], 1);
    let Instruction::StackAddr {
        slot: addressed, ..
    } = &function.blocks[0].instructions[0].instr
    else {
        panic!("expected stack address prelude");
    };
    assert_eq!(addressed, slot);
}

#[test]
fn aggregate_of_any_size_carries_its_layout() {
    let wide = parse_module(
        "fn @make() -> struct { a: i64, b: i64, c: i64 } {\nentry:\n%p = stackalloc.struct { a: i64, b: i64, c: i64 }\nret.struct { a: i64, b: i64, c: i64 } %p\n}",
    )
    .expect("parse wide aggregate return");
    let module = from_ir_x86_64(&wide, "make").expect("wide aggregate return");
    let aggregate = module.functions["make"]
        .sig
        .ret_aggregate
        .expect("return aggregate");
    assert_eq!(aggregate.size, 24);

    let taken = parse_module(
        "fn @take(struct { a: i64, b: i64, c: i64 } %p) -> i64 {\nentry:\nret.i64 0\n}",
    )
    .expect("parse wide aggregate");
    let module = from_ir_x86_64(&taken, "take").expect("wide aggregate parameter");
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
    let error = from_ir_x86_64(&float, "take").expect_err("float aggregate rejected");
    assert!(
        error.to_string().contains("%p of @take has a float field"),
        "{error}"
    );

    let returned = parse_module(
        "fn @make() -> struct { a: f64, b: f64 } {\nentry:\n%p = stackalloc.struct { a: f64, b: f64 }\nret.struct { a: f64, b: f64 } %p\n}",
    )
    .expect("parse float aggregate return");
    let error = from_ir_x86_64(&returned, "make").expect_err("float aggregate return rejected");
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
        "fn @make() -> struct { a: i64, b: i64 } {\nentry:\n%p = stackalloc.struct { a: i64, b: i64 }\nret.struct { a: i64, b: i64 } %p\n}\nfn @use() -> i64 {\nentry:\n%r = call @make()\n%ap = getfieldptr.struct { a: i64, b: i64 } %r, 0\n%a = load.i64 %ap\nret.i64 %a\n}",
    )
    .expect("parse aggregate return");
    let mir = from_ir_x86_64(&ir, "returns").expect("lower aggregate return");

    let make = mir.get_function("make").expect("make function");
    assert_eq!(make.sig.ret_ty, Some(MirType::Scalar(ScalarType::Ptr)));
    let aggregate = make.sig.ret_aggregate.expect("return marker");
    assert_eq!(aggregate.words(), 2);

    let caller = mir.get_function("use").expect("use function");
    let Instruction::Call {
        ret: Some(slot), ..
    } = &caller.blocks[0].instructions[0].instr
    else {
        panic!("expected call");
    };
    let Instruction::StackAddr {
        slot: addressed, ..
    } = &caller.blocks[0].instructions[1].instr
    else {
        panic!("expected stack address after call");
    };
    assert_eq!(&Register::Virtual(*addressed), slot);
    assert_eq!(caller.stack_reservations[addressed], 1);
}

#[test]
fn width_conversions_reject_equal_widths_and_wrong_directions() {
    for (name, expected) in [
        ("extend", "keeps the width"),
        ("trunc", "Source and target types must be different"),
    ] {
        let source =
            format!("fn @same(i64 %x) -> i64 {{\nentry:\n%a = {name}.i64.i64 %x\nret.i64 %a\n}}");
        let error = parse_module(&source).expect_err("equal width conversion rejected");
        assert!(error.to_string().contains(expected), "{name}: {error}");
    }

    // bool is one bit, so widening it to a byte is a real conversion.
    let boolean = parse_module(
        "fn @flag(bool %b) -> i64 {\nentry:\n%a = extend.bool.i8 %b\n%u = bitcast.i8.u8 %a\n%w = extend.u8.i64 %u\nret.i64 %w\n}",
    )
    .expect("parse bool widening");
    let mir = from_ir_x86_64(&boolean, "flag").expect("lower bool widening");
    let function = mir.get_function("flag").expect("flag function");
    assert!(
        matches!(
            &function.blocks[0].instructions[0].instr,
            Instruction::IntConvert { .. }
        ),
        "{:?}",
        function.blocks[0].instructions[0]
    );

    let error =
        parse_module("fn @narrow(i64 %x) -> i64 {\nentry:\n%a = extend.i64.i32 %x\nret.i64 0\n}")
            .expect_err("narrowing extend rejected");
    assert!(
        error.to_string().contains("extend from i64 to i32 narrows"),
        "{error}"
    );

    let widening =
        parse_module("fn @widen(i32 %x) -> i64 {\nentry:\n%a = trunc.i32.i64 %x\nret.i64 0\n}")
            .expect("parse widening trunc");
    let error = from_ir_x86_64(&widening, "widen").expect_err("widening trunc rejected");
    assert!(
        error.to_string().contains("trunc from i32 to i64 widens"),
        "{error}"
    );
}

#[test]
fn padded_struct_allocations_use_abi_size() {
    let ir = parse_module(
        "fn @sizes() -> ptr {\nentry:\n%stack = stackalloc.struct { a: i8, b: i64, c: i8 }\n%array = stackalloc.[2 x struct { a: i8, b: i64, c: i8 }]\nret.ptr %array\n}",
    ).expect("parse padded allocations");
    let mir = from_ir_x86_64(&ir, "sizes").expect("lower padded allocations");
    let function = mir.get_function("sizes").expect("sizes function");
    let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[0].instr else {
        panic!("expected stack allocation");
    };
    assert_eq!(function.stack_reservations[slot], 2);
    let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[1].instr else {
        panic!("expected array allocation");
    };
    assert_eq!(function.stack_reservations[slot], 5);
}

#[test]
fn stack_storage_is_not_a_register_use() {
    let ir = parse_module("fn @stack() -> ptr {\nentry:\n%p = stackalloc.[24 x i8]\nret.ptr %p\n}")
        .expect("parse stack allocation");
    let mir = from_ir_x86_64(&ir, "stack").expect("lower stack allocation");
    mir.validate().expect("stack allocation verifies");
    let function = mir.get_function("stack").expect("stack function");
    let instruction = &function.blocks[0].instructions[0];
    let Instruction::StackAddr { slot, .. } = &instruction.instr else {
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
        let mir = from_ir_x86_64(&ir, "widths").expect("lower typed operands");
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
    let mir_module = from_ir_x86_64(&ir_module, "test").expect("from_ir_x86_64 should succeed");
    let func = mir_module.get_function("add").expect("function exists");
    assert_eq!(func.sig.name, "add");
    assert_eq!(func.entry, "entry");
    let entry = func.get_block("entry").expect("entry block exists");
    assert_eq!(entry.instructions.len(), 2);
    match &entry.instructions[0].instr {
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
    match &entry.instructions[1].instr {
        Instruction::Ret { value } => {
            assert!(matches!(value, Some(Operand::Register(_))));
        }
        other => panic!("Unexpected second instruction: {:?}", other),
    }
}

#[test]
fn f64_merge_immediate_survives_lowering_as_copy() {
    let ir = parse_module(
        "fn @choose(bool %condition) -> f64 {\nentry:\n  br %condition, left, right\nleft:\n  %value = copy.f64 -0.0\n  jmp merge\nright:\n  %value = copy.f64 3.5\n  jmp merge\nmerge:\n  ret.f64 %value\n}\n",
    )
    .expect("parse f64 merge");
    let mir_module = from_ir_x86_64(&ir, "test").expect("from_ir_x86_64 should succeed");
    let function = mir_module
        .get_function("choose")
        .expect("function should exist");
    assert!(
        function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|instruction| matches!(
                &instruction.instr,
                Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::F64),
                    src: Operand::Immediate(Immediate::F64(value)),
                    ..
                } if value.to_bits() == (-0.0f64).to_bits()
            ))
    );
}

#[test]
fn unsupported_instruction_names_its_block_and_function() {
    let ir = parse_module("fn @main() -> i64 {\nentry:\n%v = load.i64 5\nret.i64 %v\n}")
        .expect("parse reversed load");
    let error = from_ir_x86_64(&ir, "reversed load").expect_err("lowering rejects reversed load");
    let text = error.to_string();
    assert!(text.contains("load.i64 5"), "{text}");
    assert!(text.contains("entry"), "{text}");
    assert!(text.contains("main"), "{text}");
}

#[test]
fn variadic_f32_argument_widens_to_f64() {
    let source = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nglobal $fmt: [5 x i8] = \"%g\\n\"\n\nfn @sink(f32 %x) -> i32 {\nentry:\n  ret.i32 0\n}\n\nfn @main() -> i64 {\nentry:\n  %p = getelem.i8 $fmt, 0\n  %x = add.f32 1.5, 2.25\n  %c = call @printf(%p, %x)\n  %d = call @sink(%x)\n  ret.i64 0\n}\n";
    let ir = parse_module(source).expect("parse variadic call");
    let module = from_ir_x86_64(&ir, "variadic_f32").expect("lower variadic call");
    let widened: Vec<_> = module.functions["main"].blocks[0]
        .instructions
        .iter()
        .filter(|inst| matches!(&inst.instr, Instruction::FloatConvert { .. }))
        .collect();
    assert_eq!(widened.len(), 1, "{widened:?}");
    assert!(
        matches!(
            &widened[0].instr,
            Instruction::FloatConvert { from, to, .. }
                if *from == MirType::Scalar(ScalarType::F32) && *to == MirType::Scalar(ScalarType::F64)
        ),
        "{widened:?}"
    );
}

#[test]
fn variadic_callptr_widens_only_past_its_marker() {
    let source = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nglobal $fmt: [5 x i8] = \"%g\\n\"\n\nfn @main() -> i64 {\nentry:\n  %p = getelem.i8 $fmt, 0\n  %f = funcaddr @printf\n  %x = add.f32 1.5, 2.25\n  %c = callptr.i32 %f(ptr %p, ..., f32 %x)\n  %d = callptr.i32 %f(ptr %p, f32 %x)\n  ret.i64 0\n}\n";
    let ir = parse_module(source).expect("parse variadic callptr");
    let module = from_ir_x86_64(&ir, "variadic_callptr").expect("lower variadic callptr");
    let widened: Vec<_> = module.functions["main"].blocks[0]
        .instructions
        .iter()
        .filter(|inst| matches!(&inst.instr, Instruction::FloatConvert { .. }))
        .collect();
    assert_eq!(widened.len(), 1, "{widened:?}");
    assert!(
        matches!(
            &widened[0].instr,
            Instruction::FloatConvert { from, to, .. }
                if *from == MirType::Scalar(ScalarType::F32) && *to == MirType::Scalar(ScalarType::F64)
        ),
        "{widened:?}"
    );
}

#[test]
fn string_global_resolves_escapes() {
    let module = lowered_global_module(
        "global $s: [8 x i8] = \"a\\nb\\t\\x41\\\\\"\n\
         fn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("lower string global");
    assert_eq!(
        module.globals["s"].initializer.as_deref(),
        Some(b"a\nb\tA\\\0\0".as_slice())
    );

    let bad = lowered_global_module(
        "global $s: [4 x i8] = \"a\\q\"\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    );
    assert!(format!("{bad:?}").contains("not an escape"), "{bad:?}");
}

#[test]
fn volatile_ir_accesses_set_memory_attrs_bit() {
    let module = lowered_global_module(
        "fn @f(ptr %p) -> i64 {\nentry:\n %a = volatile load.i64 %p\n volatile store.i64 %p, 5\n %b = load.i64 %p\n ret.i64 %b\n}",
    )
    .expect("lower volatile accesses");
    let volatile_bits: Vec<bool> = module.functions["f"].blocks[0]
        .instructions
        .iter()
        .filter_map(|inst| match &inst.instr {
            Instruction::Load { attrs, .. } | Instruction::Store { attrs, .. } => {
                Some(attrs.volatile)
            }
            _ => None,
        })
        .collect();
    assert_eq!(volatile_bits, [true, true, false]);
}

#[test]
fn atomic_ir_lowers_to_mir_atomics_with_their_ordering() {
    let module = lowered_global_module(
        "fn @f(ptr %p) -> i64 {\nentry:\n %a = atomic_load.i64 %p, acquire\n atomic_store.i64 %p, 5, release\n %b = atomic_add.i64 %p, 1, acqrel\n %c = atomic_exchange.i64 %p, 2, seqcst\n fence seqcst\n ret.i64 %b\n}",
    )
    .expect("lower atomics");
    let instructions = &module.functions["f"].blocks[0].instructions;
    assert!(matches!(
        &instructions[0].instr,
        Instruction::AtomicLoad {
            ordering: MemoryOrdering::Acquire,
            ..
        }
    ));
    assert!(matches!(
        &instructions[1].instr,
        Instruction::AtomicStore {
            ordering: MemoryOrdering::Release,
            ..
        }
    ));
    assert!(matches!(
        &instructions[2].instr,
        Instruction::AtomicBinary {
            op: AtomicBinOp::Add,
            ordering: MemoryOrdering::AcqRel,
            ..
        }
    ));
    assert!(matches!(
        &instructions[3].instr,
        Instruction::AtomicBinary {
            op: AtomicBinOp::Exchange,
            ordering: MemoryOrdering::SeqCst,
            ..
        }
    ));
    assert!(matches!(
        &instructions[4].instr,
        Instruction::Fence {
            ordering: MemoryOrdering::SeqCst
        }
    ));
}

#[test]
fn atomic_binary_result_is_the_old_value_register() {
    let module = lowered_global_module(
        "fn @f(ptr %p) -> i64 {\nentry:\n %b = atomic_add.i64 %p, 1, seqcst\n ret.i64 %b\n}",
    )
    .expect("lower atomic add");
    let instructions = &module.functions["f"].blocks[0].instructions;
    let Instruction::AtomicBinary { dst, .. } = &instructions[0].instr else {
        panic!("expected AtomicBinary, found {:?}", instructions[0]);
    };
    assert!(matches!(
        &instructions[1].instr,
        Instruction::Ret { value: Some(Operand::Register(reg)) } if reg == dst
    ));
}

#[test]
fn atomic_compare_exchange_lowers_to_mir() {
    let ir = parse_module("fn @f(ptr %p) -> i64 {\nentry:\n ret.i64 0\n}").expect("parse");
    let mut module = ir;
    let block = module
        .functions
        .get_mut("f")
        .and_then(|func| func.basic_blocks.get_mut("entry"))
        .expect("entry block");
    block.instructions.insert(
        0,
        IRInst::AtomicCompareExchange {
            result: "old",
            ty: IRType::Primitive(IRPrim::I64),
            ptr: IRVal::Variable("p"),
            expected: IRVal::Variable("p"),
            desired: IRVal::Variable("p"),
            success_ordering: IRMemOrd::SeqCst,
            failure_ordering: IRMemOrd::SeqCst,
        },
    );
    let lowered = from_ir_x86_64(&module, "cas").expect("cmpxchg lowers");
    let entry = &lowered.functions["f"].blocks[0].instructions;
    assert!(matches!(
        &entry[0].instr,
        Instruction::AtomicCompareExchange {
            success_ordering: MemoryOrdering::SeqCst,
            failure_ordering: MemoryOrdering::SeqCst,
            ..
        }
    ));
}

#[test]
fn literal_argument_to_a_wide_parameter_lowers_as_a_wide_value() {
    let source = "fn @low(i128 %a) -> i64 {\nentry:\n%t = trunc.i128.i64 %a\nret.i64 %t\n}\nfn @main() -> i64 {\nentry:\n%v = call @low(5718798622190258910)\nret.i64 %v\n}";
    let ir = parse_module(source).expect("parse wide literal argument");
    let module = from_ir_x86_64(&ir, "wide_literal").expect("lower wide literal argument");
    assert!(module.validate_in_pipeline().is_ok());
}

#[test]
fn bit_intrinsics_lower_on_wide_operands() {
    for op in ["popcount", "ctlz", "cttz", "bswap"] {
        let source =
            format!("fn @f(u128 %a) -> u128 {{\nentry:\n%r = {op}.u128 %a\nret.u128 %r\n}}");
        let ir = parse_module(&source).expect("parse bit intrinsic");
        let module = from_ir_x86_64(&ir, "bits").expect("lower wide bit intrinsic");
        assert!(module.validate_in_pipeline().is_ok(), "{op}");
    }
    for op in ["rotl", "rotr"] {
        let source = format!(
            "fn @f(i128 %a, i128 %n) -> i128 {{\nentry:\n%r = {op}.i128 %a, %n\nret.i128 %r\n}}"
        );
        let ir = parse_module(&source).expect("parse wide rotate");
        let module = from_ir_x86_64(&ir, "rotate").expect("lower wide rotate");
        assert!(module.validate_in_pipeline().is_ok(), "{op}");
    }
}

#[test]
fn callptr_with_a_wide_argument_is_rejected_by_name() {
    let source = "fn @main() -> i64 {\nentry:\n%f = funcaddr @main\n%r = callptr.i64 %f(i128 1)\nret.i64 %r\n}";
    let ir = parse_module(source).expect("parse indirect call");
    let err = from_ir_x86_64(&ir, "indirect").expect_err("indirect wide argument must not lower");
    assert!(err.to_string().contains("callptr"), "{err}");
}

#[test]
fn io_result_is_i64_in_every_typed_position() {
    for op in [
        "%r = readbyte",
        "%r = writebyte 65",
        "%r = write %p, 3",
        "%r = read %p, 3",
    ] {
        for body in [
            "%c = add.i8 %r, 1\n ret.i64 0",
            "store.i8 %p, %r\n ret.i64 0",
        ] {
            let narrow = format!("fn @f(ptr %p) -> i64 {{\nentry:\n {op}\n {body}\n}}");
            let err = lowered_global_module(&narrow)
                .err()
                .unwrap_or_else(|| panic!("{op} / {body} must not lower at i8"))
                .to_string();
            assert!(
                err.contains("to have type i8") && err.contains("defined as i64"),
                "{op} / {body}: {err}"
            );
        }
        let narrow_ret = format!("fn @f(ptr %p) -> i8 {{\nentry:\n {op}\n ret.i8 %r\n}}");
        let err = lowered_global_module(&narrow_ret)
            .err()
            .unwrap_or_else(|| panic!("{op} returned at i8 must not lower"))
            .to_string();
        assert!(err.contains("defined as i64"), "{op} ret: {err}");
        let wide =
            format!("fn @f(ptr %p) -> i64 {{\nentry:\n {op}\n %c = add.i64 %r, 1\n ret.i64 %c\n}}");
        lowered_global_module(&wide).expect("io result used at i64 must lower");
    }
}

mod pairs_on_a_32_bit_target {
    use super::*;

    fn lower32(source: &str) -> Result<Module, FromIRError> {
        let ir = parse_module(source).expect("parse");
        from_ir_for_target(&ir, "pairs", TargetArchitecture::I686.data_layout())
    }

    fn body(source: &str) -> String {
        let module = lower32(source).expect("lower for a 32-bit target");
        let mut text = String::new();
        for function in module.functions.values() {
            for block in &function.blocks {
                for instruction in &block.instructions {
                    text.push_str(&format!("{}\n", instruction.instr));
                }
            }
        }
        text
    }

    const ADD: &str = "fn @sum(i64 %x, i64 %y) -> i64 {\nentry:\n %a = add.i64 %x, %y\n ret.i64 %a\n}\n\
         fn @main() -> i64 {\nentry:\n ret.i64 0\n}";

    #[test]
    fn no_instruction_still_names_a_64_bit_type() {
        let text = body(
            "fn @main() -> i64 {\nentry:\n\
             %a = add.i64 1, 2\n %b = sub.i64 %a, 3\n %c = mul.i64 %b, %a\n\
             %d = and.i64 %c, %b\n %e = or.i64 %d, %a\n %f = xor.i64 %e, %b\n\
             %g = shl.i64 %f, 33\n %h = shr.i64 %g, 7\n %i = lt.i64 %h, %a\n\
             %j = select.i64 %i, %h, %a\n ret.i64 %j\n}",
        );
        for line in text.lines() {
            assert!(!line.contains("i64"), "a 64-bit operation survived: {line}");
        }
    }

    #[test]
    fn a_64_bit_add_computes_its_own_carry_without_a_flags_register() {
        let text = body(ADD);
        // The carry is `((a & b) | ((a | b) & !sum)) >> 31`, so the low word's add is
        // followed by a shift that lands the carry bit in a register of its own.
        assert!(
            text.contains("lshr") && text.contains("31"),
            "no carry extraction in:\n{text}"
        );
        assert!(
            text.matches("add").count() >= 3,
            "a pair add needs the low word, the high word and the carry:\n{text}"
        );
    }

    #[test]
    fn division_calls_the_64_bit_runtime_helpers_not_the_128_bit_ones() {
        for (op, ty, symbol) in [
            ("div", "i64", "__divdi3"),
            ("div", "u64", "__udivdi3"),
            ("rem", "i64", "__moddi3"),
            ("rem", "u64", "__umoddi3"),
        ] {
            let source =
                format!("fn @main() -> {ty} {{\nentry:\n %a = {op}.{ty} 7, 3\n ret.{ty} %a\n}}");
            let module = lower32(&source).expect("lower a division");
            assert!(
                module.functions.contains_key(symbol),
                "{op}.{ty} did not call {symbol}"
            );
            assert!(
                !module.functions.keys().any(|name| name.ends_with("ti3")),
                "{op}.{ty} reached for a 128-bit helper"
            );
        }
    }

    #[test]
    fn a_shift_count_is_taken_modulo_sixty_four() {
        // 2 * 32 - 1 masks the count to the pair's width before it is split.
        assert!(
            body(
                "fn @sh(i64 %x, i64 %n) -> i64 {\nentry:\n %a = shl.i64 %x, %n\n ret.i64 %a\n}\n\
                  fn @main() -> i64 {\nentry:\n ret.i64 0\n}"
            )
            .contains("63"),
            "the shift count was not masked to the pair's width"
        );
    }

    #[test]
    fn a_64_bit_value_crosses_a_boundary_as_two_words() {
        let module = lower32(
            "fn @id(i64 %x) -> i64 {\nentry:\n ret.i64 %x\n}\n\
             fn @main() -> i64 {\nentry:\n %a = call @id(5)\n ret.i64 %a\n}",
        )
        .expect("lower a call");
        let id = &module.functions["id"];
        let aggregate = id.sig.params[0].aggregate.expect("parameter aggregate");
        assert_eq!(aggregate.size, 8);
        assert_eq!(aggregate.words_of(PointerWidth::Bits32), 2);
        assert_eq!(aggregate.words_of(PointerWidth::Bits64), 1);
        assert_eq!(id.sig.ret_aggregate.expect("result aggregate").size, 8);
    }

    #[test]
    fn a_64_bit_global_becomes_eight_bytes_of_memory() {
        let module = lower32("global $cell: i64 = -2\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}")
            .expect("lower a global");
        let cell = &module.globals["cell"];
        assert_eq!(cell.size, 8);
        assert_eq!(
            cell.initializer.as_deref(),
            Some((-2i64).to_le_bytes().as_slice())
        );
    }

    #[test]
    fn a_128_bit_integer_is_refused_by_name_rather_than_truncated() {
        let error = lower32("fn @main() -> i64 {\nentry:\n %a = add.i128 1, 2\n ret.i64 0\n}")
            .expect_err("i128 needs four registers on a 32-bit target");
        let message = error.to_string();
        assert!(message.contains("128-bit"), "{message}");
        assert!(message.contains("add.i128"), "{message}");
    }

    #[test]
    fn a_64_bit_atomic_is_refused_by_name_rather_than_split_unsafely() {
        let error = lower32(
            "fn @main() -> i64 {\nentry:\n %p = stackalloc.i64\n\
             %a = atomic_add.i64 %p, 5, seqcst\n ret.i64 0\n}",
        )
        .expect_err("two words cannot be updated atomically");
        let message = error.to_string();
        assert!(message.contains("atomic"), "{message}");
        assert!(message.contains("64-bit"), "{message}");
    }

    #[test]
    fn a_64_bit_target_is_untouched_by_any_of_this() {
        let ir = parse_module(ADD).expect("parse");
        let module = from_ir_for_target(&ir, "pairs", TargetArchitecture::X86_64.data_layout())
            .expect("lower");
        let text: String = module.functions["sum"].blocks[0]
            .instructions
            .iter()
            .map(|instruction| format!("{}\n", instruction.instr))
            .collect();
        assert!(
            text.contains("i64"),
            "a 64-bit target still works in i64:\n{text}"
        );
    }
}

#[test]
fn a_big_endian_global_holds_its_literal_most_significant_byte_first() {
    let ir = parse_module(
        "global $a: i64 = 41\nglobal $b: f64 = 2.5\nfn @main() -> i64 {\nentry:\n ret.i64 0\n}",
    )
    .expect("parse globals");
    let module = from_ir_for_target(&ir, "globals", TargetArchitecture::S390x.data_layout())
        .expect("lower globals");
    assert_eq!(
        module.globals["a"].initializer.as_deref(),
        Some(41i64.to_be_bytes().as_slice())
    );
    assert_eq!(
        module.globals["b"].initializer.as_deref(),
        Some(2.5f64.to_be_bytes().as_slice())
    );
}

#[test]
fn a_big_endian_pair_puts_its_high_word_at_the_lower_address() {
    // The low word is 3 and the high word 0, so which address each lands at says which
    // half of the pair the target calls the first word in memory.
    let low_word_offset = |layout| {
        let ir = parse_module(
            "fn @main() -> i64 {\nentry:\n %p = stackalloc.i128\n \
             store.i128 %p, 3\n ret.i64 0\n}",
        )
        .expect("parse pair store");
        let module = from_ir_for_target(&ir, "pair", layout).expect("lower pair store");
        let instructions: Vec<&Instruction> = module.functions["main"].blocks[0]
            .instructions
            .iter()
            .map(|inst| &inst.instr)
            .collect();
        let base = match instructions.first() {
            Some(Instruction::StackAddr { dst, .. }) => dst.clone(),
            other => panic!("expected a stack address, found {other:?}"),
        };
        let mut displaced = HashMap::new();
        let mut at = None;
        for instruction in instructions {
            match instruction {
                Instruction::Lea { dst, offset, .. } => {
                    displaced.insert(dst.clone(), *offset);
                }
                Instruction::Store { src, addr, .. } if src.integer_constant() == Some(3) => {
                    let AddressMode::BaseOffset { base: on, .. } = addr else {
                        panic!("expected base+offset addressing");
                    };
                    at = Some(if *on == base { 0 } else { displaced[on] });
                }
                _ => {}
            }
        }
        at.expect("no store of the low word")
    };
    assert_eq!(low_word_offset(TargetArchitecture::X86_64.data_layout()), 0);
    assert_eq!(low_word_offset(TargetArchitecture::S390x.data_layout()), 8);
}

/// Alignment of every load and store in `@main`, in program order.
fn access_aligns(source: &str) -> Vec<(u8, bool)> {
    let ir = parse_module(source).expect("parse alignment probe");
    let module = from_ir_x86_64(&ir, "align").expect("lower alignment probe");
    module.functions["main"].blocks[0]
        .instructions
        .iter()
        .filter_map(|inst| match &inst.instr {
            Instruction::Load { attrs, .. } | Instruction::Store { attrs, .. } => {
                Some((attrs.align, attrs.stated_align))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn a_packed_field_access_is_aligned_like_its_address() {
    assert_eq!(
        access_aligns(
            "type !Bw = packed struct { a: i8, b: i32 }\n\
             fn @main() -> i32 {\nentry:\n\
             %p = stackalloc.!Bw\n\
             %b = getfieldptr.!Bw %p, 1\n\
             store.i32 %b, 7\n\
             %v = load.i32 %b\n\
             ret.i32 %v\n}"
        ),
        vec![(1, false), (1, false)]
    );
}

#[test]
fn a_packed_field_keeps_the_alignment_its_offset_leaves() {
    assert_eq!(
        access_aligns(
            "type !Sm = packed struct { x: i32, s: i16, y: i32 }\n\
             fn @main() -> i32 {\nentry:\n\
             %p = stackalloc.!Sm\n\
             %y = getfieldptr.!Sm %p, 2\n\
             %v = load.i32 %y\n\
             ret.i32 %v\n}"
        ),
        vec![(2, false)]
    );
}

#[test]
fn a_plain_struct_field_access_keeps_its_natural_alignment() {
    assert_eq!(
        access_aligns(
            "type !Ns = struct { a: i8, b: i32 }\n\
             fn @main() -> i32 {\nentry:\n\
             %p = stackalloc.!Ns\n\
             %b = getfieldptr.!Ns %p, 1\n\
             store.i32 %b, 7\n\
             %v = load.i32 %b\n\
             ret.i32 %v\n}"
        ),
        vec![(4, false), (4, false)]
    );
}

#[test]
fn a_packed_field_of_a_pointer_the_function_was_handed_is_still_under_aligned() {
    assert_eq!(
        access_aligns(
            "type !Bw = packed struct { a: i8, b: i32 }\n\
             fn @main(ptr %p) -> i32 {\nentry:\n\
             %b = getfieldptr.!Bw %p, 1\n\
             %v = load.i32 %b\n\
             ret.i32 %v\n}"
        ),
        vec![(1, false)]
    );
}

#[test]
fn a_field_of_a_packed_global_is_aligned_like_the_global() {
    assert_eq!(
        access_aligns(
            "type !Pair = packed struct { a: i32, b: i32 }\n\
             global $g: !Pair = {1, 2}\n\
             fn @main() -> i32 {\nentry:\n\
             %b = getfieldptr.!Pair $g, 1\n\
             %v = load.i32 %b\n\
             ret.i32 %v\n}"
        ),
        vec![(1, false)]
    );
}

#[test]
fn a_nested_packed_field_carries_the_alignment_of_the_whole_path() {
    assert_eq!(
        access_aligns(
            "type !Tail = packed struct { a: i64, b: i8 }\n\
             type !Outer = struct { c: i8, tail: !Tail }\n\
             fn @main() -> i64 {\nentry:\n\
             %p = stackalloc.!Outer\n\
             %t = getfieldptr.!Outer %p, 1\n\
             %a = getfieldptr.!Tail %t, 0\n\
             %v = load.i64 %a\n\
             ret.i64 %v\n}"
        ),
        vec![(1, false)]
    );
}

#[test]
fn an_array_element_keeps_the_alignment_of_its_stride() {
    assert_eq!(
        access_aligns(
            "fn @main() -> i32 {\nentry:\n\
             %p = stackalloc.[4 x i32]\n\
             %e = getelem.i32 %p, 3\n\
             %v = load.i32 %e\n\
             ret.i32 %v\n}"
        ),
        vec![(4, false)]
    );
}

#[test]
fn a_stated_alignment_still_overrides_what_the_address_shows() {
    assert_eq!(
        access_aligns(
            "type !Ns = struct { a: i8, b: i32 }\n\
             fn @main() -> i32 {\nentry:\n\
             %p = stackalloc.!Ns\n\
             %b = getfieldptr.!Ns %p, 1\n\
             %v = load.i32 %b align 1\n\
             ret.i32 %v\n}"
        ),
        vec![(1, true)]
    );
}

#[test]
fn a_variable_index_scaled_by_a_constant_keeps_what_the_scale_guarantees() {
    assert_eq!(
        access_aligns(
            "fn @main(ptr %out, i64 %i) -> i64 {\nentry:\n\
             %scaled = mul.i64 %i, 2\n\
             %e = getelem.i64 %out, %scaled\n\
             %v = load.i64 %e\n\
             ret.i64 %v\n}"
        ),
        vec![(8, false)]
    );
}

#[test]
fn an_element_of_a_packed_array_is_aligned_like_its_stride() {
    assert_eq!(
        access_aligns(
            "type !Odd = packed struct { a: i64, b: i8 }\n\
             fn @main(ptr %p, i64 %i) -> i64 {\nentry:\n\
             %e = getelem.!Odd %p, %i\n\
             %a = getfieldptr.!Odd %e, 0\n\
             %v = load.i64 %a\n\
             ret.i64 %v\n}"
        ),
        vec![(1, false)]
    );
}

/// Every integer `IntConvert` source and its signedness in `@gep`, in program order.
fn index_widening(source: &str, arch: TargetArchitecture) -> Vec<(MirType, bool)> {
    let ir = parse_module(source).expect("parse index probe");
    let module = from_ir_for_target(&ir, "gep", arch.data_layout()).expect("lower index probe");
    module.functions["gep"]
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .filter_map(|inst| match &inst.instr {
            Instruction::IntConvert { from, signed, .. }
                if *from != MirType::Scalar(ScalarType::Ptr) =>
            {
                Some((*from, *signed))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn inline_asm_result_index_keeps_its_width_and_sign() {
    let source = "fn @gep(ptr %p) -> ptr {\nentry:\n%n = asm.i16 \"nop\", out(ax)\n%q = getelem.i16 %p, %n\nret.ptr %q\n}";
    for arch in [
        TargetArchitecture::Riscv32,
        TargetArchitecture::I686,
        TargetArchitecture::Armv7,
        TargetArchitecture::X86_64,
    ] {
        let widening = index_widening(source, arch);
        assert!(
            widening.contains(&(MirType::Scalar(ScalarType::I16), true)),
            "{arch:?}: {widening:?}"
        );
    }
}

#[test]
fn atomic_cmpxchg_result_index_keeps_its_width_and_sign() {
    let source = "fn @gep(ptr %p) -> ptr {\nentry:\n%cell = stackalloc.i16\nstore.i16 %cell, 3\n%n = atomic_cmpxchg.i16 %cell, 3, 4, seqcst, seqcst\n%q = getelem.i16 %p, %n\nret.ptr %q\n}";
    for arch in [TargetArchitecture::Riscv32, TargetArchitecture::X86_64] {
        let widening = index_widening(source, arch);
        assert!(
            widening.contains(&(MirType::Scalar(ScalarType::I16), true)),
            "{arch:?}: {widening:?}"
        );
    }
}

#[test]
fn vector_lane_index_keeps_the_lane_width_and_sign() {
    let source = "fn @gep(ptr %p) -> ptr {\nentry:\n%v = splat.<8 x i16> 1\n%n = extractlane.<8 x i16> %v, 0\n%q = getelem.i16 %p, %n\nret.ptr %q\n}";
    for arch in [TargetArchitecture::Riscv32, TargetArchitecture::X86_64] {
        let widening = index_widening(source, arch);
        assert!(
            widening.contains(&(MirType::Scalar(ScalarType::I16), true)),
            "{arch:?}: {widening:?}"
        );
    }
}

#[test]
fn a_register_of_unrecorded_type_is_refused() {
    let register = Register::Virtual(VirtualReg {
        id: 7,
        class: RegisterClass::Gpr,
    });
    let error = operand_type(&Operand::Register(register), &HashMap::new())
        .expect_err("a register without a recorded type must not be guessed");
    assert!(matches!(error, FromIRError::UnknownVariable), "{error}");
}

#[test]
fn inttoptr_suffix_must_match_its_operand_width() {
    let source = "fn @cast(ptr %p) -> ptr {\nentry:\n%a = ptrtoint.i64 %p\n%q = inttoptr.i32 %a\nret.ptr %q\n}";
    let ir = parse_module(source).expect("parse inttoptr");
    let err = from_ir_x86_64(&ir, "cast").expect_err("mismatched suffix must be rejected");
    assert!(
        format!("{err}").contains("inttoptr.i32"),
        "unexpected error: {err}"
    );
}

#[test]
fn inttoptr_of_inline_asm_keeps_its_source_width() {
    let source = "fn @cast() -> ptr {\nentry:\n%n = asm.i16 \"nop\", out(ax)\n%p = inttoptr.i16 %n\nret.ptr %p\n}";
    for arch in [TargetArchitecture::Riscv32, TargetArchitecture::X86_64] {
        let ir = parse_module(source).expect("parse inttoptr");
        let module = from_ir_for_target(&ir, "cast", arch.data_layout()).expect("lower inttoptr");
        let widening: Vec<_> = module.functions["cast"].blocks[0]
            .instructions
            .iter()
            .filter_map(|inst| match &inst.instr {
                Instruction::IntConvert { from, signed, .. } => Some((*from, *signed)),
                _ => None,
            })
            .collect();
        assert!(
            widening.contains(&(MirType::Scalar(ScalarType::I16), false)),
            "{arch:?}: {widening:?}"
        );
    }
}

#[test]
fn unary_immediates_refuse_a_width_no_immediate_holds() {
    let pointer = PointerWidth::Bits64;
    for (ty, zero, ones) in [
        (
            MirType::Scalar(ScalarType::I8),
            Immediate::I8(0),
            Immediate::I8(-1),
        ),
        (
            MirType::Scalar(ScalarType::I16),
            Immediate::I16(0),
            Immediate::I16(-1),
        ),
        (
            MirType::Scalar(ScalarType::I32),
            Immediate::I32(0),
            Immediate::I32(-1),
        ),
        (
            MirType::Scalar(ScalarType::I64),
            Immediate::I64(0),
            Immediate::I64(-1),
        ),
    ] {
        assert_eq!(zero_immediate(ty, pointer).expect("zero"), zero);
        assert_eq!(all_ones_immediate(ty, pointer).expect("ones"), ones);
    }

    for wide in [
        MirType::Vector(VectorType::V128(VectorLane::I32)),
        MirType::Vector(VectorType::V256(VectorLane::I32)),
    ] {
        assert_eq!(wide.size_bytes(pointer) % 8, 0);
        assert!(
            zero_immediate(wide, pointer).is_err(),
            "{wide:?} has no zero immediate"
        );
        assert!(
            all_ones_immediate(wide, pointer).is_err(),
            "{wide:?} has no all-ones immediate"
        );
    }

    assert!(
        zero_immediate(MirType::Scalar(ScalarType::Ptr), PointerWidth::Bits128).is_err(),
        "a 16-byte pointer has no zero immediate"
    );
}

#[test]
fn from_ir_x86_64_lowers_against_the_x86_64_layout_only() {
    let source = "fn @gep(ptr %p, i32 %i) -> ptr {\nentry:\n%q = getelem.i64 %p, %i\nret.ptr %q\n}";
    let ir = parse_module(source).expect("parse gep");
    let named = from_ir_x86_64(&ir, "gep").expect("lower by name");
    let explicit = from_ir_for_target(&ir, "gep", TargetArchitecture::X86_64.data_layout())
        .expect("lower by layout");
    let narrow = from_ir_for_target(&ir, "gep", TargetArchitecture::Riscv32.data_layout())
        .expect("lower for riscv32");

    let word = |module: &Module| {
        module.get_function("gep").expect("gep function").blocks[0]
            .instructions
            .iter()
            .find_map(|inst| match &inst.instr {
                Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    ty,
                    ..
                } => Some(*ty),
                _ => None,
            })
    };
    assert_eq!(word(&named), Some(MirType::Scalar(ScalarType::I64)));
    assert_eq!(word(&named), word(&explicit));
    assert_eq!(word(&narrow), Some(MirType::Scalar(ScalarType::I32)));
}
