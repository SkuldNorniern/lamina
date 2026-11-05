use super::*;
use crate::mir::{FunctionBuilder, TransformPipeline, codegen::from_ir_x86_64};
use crate::mir_codegen::{simd_support::tools_available, test_support::PRINT_I64_HELPERS};
use crate::parser::parse_module;
use crate::test_support::TempDir;
use lamina_platform::PointerWidth;
use std::{
    collections::HashSet,
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn float_copy_emits_local_move_not_add() {
    let ty = MirType::Scalar(ScalarType::F64);
    let src = Register::Virtual(VirtualReg::fpr(0));
    let dst = Register::Virtual(VirtualReg::fpr(1));
    let function = FunctionBuilder::new("copy_float")
        .param(src.clone(), ty)
        .returns(ty)
        .block("entry")
        .instr(MirInst::Copy {
            ty,
            dst: dst.clone(),
            src: Operand::Register(src),
        })
        .instr(MirInst::Ret {
            value: Some(Operand::Register(dst)),
        })
        .build();
    let mut module = MirModule::new("copy_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_wasm(
        &module,
        &mut output,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Unknown,
        PointerWidth::Bits64,
    )
    .expect("WASM codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(
        assembly.contains("local.get"),
        "expected local move: {assembly}"
    );
    assert!(
        assembly.contains("local.set"),
        "expected local move: {assembly}"
    );
    assert!(
        !assembly.contains("f64.add"),
        "unexpected float add: {assembly}"
    );
}

#[test]
fn dispatcher_starts_at_non_leading_entry_block() {
    let mut function = FunctionBuilder::new("entry_second")
        .returns(MirType::Scalar(ScalarType::I64))
        .block("first")
        .instr(MirInst::Ret {
            value: Some(Operand::Immediate(Immediate::I64(7))),
        })
        .block("second")
        .instr(MirInst::Ret {
            value: Some(Operand::Immediate(Immediate::I64(9))),
        })
        .build();
    function.entry = "second".to_string();
    let mut module = MirModule::new("entry_test");
    module.add_function(function);
    let mut output = Vec::new();
    generate_mir_wasm(
        &module,
        &mut output,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Unknown,
        PointerWidth::Bits64,
    )
    .expect("WASM codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(
        assembly.contains("i64.const 1\n    local.set $pc"),
        "dispatcher should start at the entry block index: {assembly}"
    );
}

fn run_wasm_main(test_name: &str, source: &str, opt_level: u8) -> Option<String> {
    if !tools_available(&["wasmtime"]) {
        return None;
    }
    let source = format!("{source}\n{PRINT_I64_HELPERS}");
    let ir = parse_module(&source).expect("source should parse");
    let mut mir = from_ir_x86_64(&ir, "module").expect("source should lower");
    TransformPipeline::default_for_opt_level(opt_level)
        .apply_to_module(&mut mir)
        .expect("transforms should apply");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )
    .expect("WASM codegen should succeed");
    let guard = TempDir::new(&format!("lamina-wasm-{test_name}"));
    // The module runs in its binary form, so every execution test here also covers
    // the in-process writer rather than wasmtime's own text reader.
    let text = String::from_utf8(wat).expect("WAT should be UTF-8");
    let binary = super::binary::encode_module(&text).expect("the wasm writer should encode");
    let wasm_path = guard.path().join("module.wasm");
    fs::write(&wasm_path, binary).expect("write wasm");
    let output = Command::new("wasmtime")
        .args(["run", "--invoke", "main"])
        .arg(&wasm_path)
        .output()
        .expect("run wasmtime");
    assert!(
        output.status.success(),
        "wasm run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(
        String::from_utf8(output.stdout)
            .expect("stdout should be UTF-8")
            .replace("\r\n", "\n"),
    )
}

/// Operation, type, operands and whether it overflows, at the edges of each width:
/// i8, u8 and i32 are checked by narrowing the i64 result, i64 and u64 by the sign
/// rules and by dividing the product back.
const OVERFLOW_CASES: &[(&str, &str, &str, &str, u8)] = &[
    ("add", "i8", "127", "1", 1),
    ("add", "i8", "126", "1", 0),
    ("sub", "i8", "-128", "1", 1),
    ("mul", "i8", "-128", "-1", 1),
    ("mul", "i8", "16", "8", 1),
    ("mul", "i8", "15", "8", 0),
    ("add", "u8", "255", "1", 1),
    ("sub", "u8", "0", "1", 1),
    ("mul", "u8", "16", "16", 1),
    ("mul", "u8", "15", "17", 0),
    ("add", "i32", "2147483647", "1", 1),
    ("mul", "i32", "46341", "46341", 1),
    ("mul", "i32", "46340", "46340", 0),
    ("add", "i64", "9223372036854775807", "1", 1),
    ("add", "i64", "9223372036854775807", "-1", 0),
    ("sub", "i64", "-9223372036854775808", "1", 1),
    ("sub", "i64", "-1", "9223372036854775807", 0),
    ("sub", "i64", "-2", "9223372036854775807", 1),
    ("mul", "i64", "-9223372036854775808", "-1", 1),
    ("mul", "i64", "-1", "-9223372036854775808", 1),
    ("mul", "i64", "-9223372036854775808", "1", 0),
    ("mul", "i64", "0", "-9223372036854775808", 0),
    ("mul", "i64", "3037000500", "3037000500", 1),
    ("mul", "i64", "3037000499", "3037000499", 0),
    ("add", "u64", "18446744073709551615", "1", 1),
    ("sub", "u64", "0", "1", 1),
    ("mul", "u64", "4294967296", "4294967296", 1),
    ("mul", "u64", "4294967295", "4294967297", 0),
];

#[test]
fn checked_arithmetic_reports_overflow_at_every_width() {
    // One function per operation, so the operands reach it as locals.
    let mut source = String::new();
    let mut main = String::from("fn @main() -> i64 {\nentry:\n");
    let mut expected = String::new();
    let mut defined = HashSet::new();
    for (index, (op, ty, lhs, rhs, overflows)) in OVERFLOW_CASES.iter().enumerate() {
        if defined.insert((op, ty)) {
            source.push_str(&format!(
                "fn @{op}_{ty}({ty} %a, {ty} %b) -> bool {{\nentry:\n %r = {op}.ovf.{ty} %a, %b\n ret.bool %r\n}}\n"
            ));
        }
        main.push_str(&format!(
            " %c{index} = call @{op}_{ty}({lhs}, {rhs})\n %v{index} = extend.bool.i64 %c{index}\n call @print_i64(%v{index})\n"
        ));
        expected.push_str(&format!("{overflows}\n"));
    }
    source.push_str(&main);
    source.push_str(" ret.i64 0\n}");
    expected.push_str("0\n");
    for opt_level in [0, 3] {
        let Some(stdout) = run_wasm_main("ovf", &source, opt_level) else {
            return;
        };
        assert_eq!(stdout, expected, "O{opt_level}");
    }
}

#[test]
fn distinct_stack_slots_in_one_frame_do_not_alias() {
    let Some(stdout) = run_wasm_main(
        "slots",
        "fn @main() -> i64 {\nentry:\n %p = stackalloc.i64\n %q = stackalloc.i64\n store.i64 %p, 11\n store.i64 %q, 22\n %x = load.i64 %p\n call @print_i64(%x)\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "11\n0\n");
}

#[test]
fn recursive_invocations_keep_their_own_stack_slots() {
    let Some(stdout) = run_wasm_main(
        "recursive",
        "fn @rec(i64 %n) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, %n\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n ret.i64 0\nstep:\n %m = sub.i64 %n, 1\n %r = call @rec(%m)\n %x = load.i64 %p\n %s = add.i64 %x, %r\n ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n %r = call @rec(3)\n ret.i64 %r\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "6\n");
}

#[test]
fn stack_frames_are_released_on_every_return() {
    let Some(stdout) = run_wasm_main(
        "release",
        "fn @leaf(i64 %n) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, %n\n %x = load.i64 %p\n ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n %i = add.i64 0, 0\n jmp loop\nloop:\n %r = call @leaf(%i)\n %i = add.i64 %i, 1\n %c = lt.i64 %i, 6000\n br %c, loop, done\ndone:\n ret.i64 %i\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "6000\n");
}

#[test]
fn signed_byte_compares_on_loaded_operands_read_the_sign() {
    let Some(stdout) = run_wasm_main(
        "cmp8",
        "fn @main() -> i64 {\nentry:\n %p = stackalloc.i8\n store.i8 %p, -1\n %x = load.i8 %p\n %a = lt.i8 %x, 0\n %pv0 = extend.bool.i64 %a\n call @print_i64(%pv0)\n %b = eq.i8 %x, -1\n %pv1 = extend.bool.i64 %b\n call @print_i64(%pv1)\n %c = lt.u8 %x, 1\n %pv2 = extend.bool.i64 %c\n call @print_i64(%pv2)\n %d = gt.i8 %x, -2\n %pv3 = extend.bool.i64 %d\n call @print_i64(%pv3)\n %e = gt.u8 %x, 1\n %pv4 = extend.bool.i64 %e\n call @print_i64(%pv4)\n %f = add.i8 %x, 0\n %g = lt.i8 %f, 0\n %pv5 = extend.bool.i64 %g\n call @print_i64(%pv5)\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "1\n1\n0\n1\n1\n1\n0\n");
}

#[test]
fn narrow_compares_normalize_operands_at_their_width() {
    let Some(stdout) = run_wasm_main(
        "cmpwidths",
        "fn @main() -> i64 {\nentry:\n %p8 = stackalloc.i8\n %p16 = stackalloc.i16\n %p32 = stackalloc.i32\n store.i8 %p8, -128\n store.i16 %p16, -32768\n store.i32 %p32, -2147483648\n %a8 = load.i8 %p8\n %a16 = load.i16 %p16\n %a32 = load.i32 %p32\n %s8 = lt.i8 %a8, 0\n %pv0 = extend.bool.i64 %s8\n call @print_i64(%pv0)\n %s16 = lt.i16 %a16, 0\n %pv1 = extend.bool.i64 %s16\n call @print_i64(%pv1)\n %s32 = le.i32 %a32, -2147483648\n %pv2 = extend.bool.i64 %s32\n call @print_i64(%pv2)\n %u8 = lt.u8 %a8, 127\n %pv3 = extend.bool.i64 %u8\n call @print_i64(%pv3)\n %u16 = gt.u16 %a16, 32767\n %pv4 = extend.bool.i64 %u16\n call @print_i64(%pv4)\n %e8 = eq.i8 %a8, -128\n %pv5 = extend.bool.i64 %e8\n call @print_i64(%pv5)\n %c8 = add.i8 %a8, 0\n %g8 = gt.i8 %c8, -127\n %pv6 = extend.bool.i64 %g8\n call @print_i64(%pv6)\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "1\n1\n1\n0\n1\n1\n0\n0\n");
}

#[test]
fn narrow_arguments_keep_their_declared_signedness_across_calls() {
    let Some(stdout) = run_wasm_main(
        "boundary_signedness",
        "fn @mk() -> i8 {\nentry:\n %v = trunc.i64.i8 255\n ret.i8 %v\n}\nfn @take(u8 %u, i8 %c) -> i64 {\nentry:\n %a = lt.u8 %u, 1\n %pv0 = extend.bool.i64 %a\n call @print_i64(%pv0)\n %b = lt.i8 %c, 0\n %pv1 = extend.bool.i64 %b\n call @print_i64(%pv1)\n ret.i64 0\n}\nfn @main() -> i64 {\nentry:\n %c = call @mk()\n %u = trunc.i64.u8 255\n %r = call @take(%u, %c)\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "0\n1\n0\n");
}

#[test]
fn entry_block_runs_first_after_block_reordering() {
    for opt_level in [0, 1, 2] {
        let Some(stdout) = run_wasm_main(
            "entry_block",
            "fn @main() -> i64 {\nentry:\n %v = readbyte\n %c = eq.i64 %v, 1\n br %c, a_wrong, z_right\na_wrong:\n ret.i64 7\nz_right:\n ret.i64 9\n}",
            opt_level,
        ) else {
            return;
        };
        assert_eq!(stdout, "9\n", "opt level {opt_level}");
    }
}

#[test]
fn tail_call_with_caller_frame_argument_reads_caller_storage() {
    let Some(stdout) = run_wasm_main(
        "tail_frame_arg",
        "fn @g(ptr %p, i64 %n) -> i64 {\nentry:\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n %q = stackalloc.i64\n store.i64 %q, 99\n %x = load.i64 %p\n %y = load.i64 %q\n %z = add.i64 %x, %y\n ret.i64 %z\nstep:\n %m = sub.i64 %n, 1\n %r = call @g(%p, %m)\n ret.i64 %r\n}\nfn @top(i64 %n, i64 %d) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, 7\n %r = call @g(%p, %n)\n ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n %k = add.i64 0, 3\n %r = call @top(%k, %k)\n ret.i64 %r\n}",
        2,
    ) else {
        return;
    };
    assert_eq!(stdout, "106\n");
}

#[test]
fn optimized_tail_calls_run_in_constant_stack() {
    let Some(stdout) = run_wasm_main(
        "tail_calls",
        "fn @count(i64 %n) -> i64 {\nentry:\n %c = eq.i64 %n, 0\n br %c, done, step\ndone:\n ret.i64 42\nstep:\n %m = sub.i64 %n, 1\n %r = call @count(%m)\n ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n %r = call @count(1000000)\n ret.i64 %r\n}",
        2,
    ) else {
        return;
    };
    assert_eq!(stdout, "42\n");
}

#[test]
fn relocated_global_initializers_hold_real_addresses() {
    let Some(stdout) = run_wasm_main(
        "global_relocs",
        "global $s: [4 x i8] = \"xyz\"\nglobal $p: ptr = $s + 1\nglobal $fp: ptr = @g\n\nfn @f() -> i64 {\nentry:\n  ret.i64 65\n}\n\nfn @g() -> i64 {\nentry:\n  ret.i64 66\n}\n\nfn @main() -> i64 {\nentry:\n  %q = load.ptr $p\n  %c = load.i8 %q\n  %w1 = writebyte %c\n  %sp = writebyte 32\n  %h = load.ptr $fp\n  %r = callptr.i64 %h()\n  %w2 = writebyte %r\n  %nl = writebyte 10\n  ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "y B\n0\n");
}

#[test]
fn an_address_wasm_cannot_place_is_refused_by_name() {
    let mut module = MirModule::new("extern_reloc");
    let mut global = Global::new("p", MirType::Scalar(ScalarType::Ptr), PointerWidth::Bits32);
    global.initializer = Some(vec![0; 4]);
    global.relocations.push(GlobalRelocation {
        offset: 0,
        symbol: "elsewhere".to_string(),
        addend: 0,
        size: 4,
    });
    module.add_global(global);
    module.add_function(
        FunctionBuilder::new("main")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(MirInst::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build(),
    );
    let mut wat = Vec::new();
    let error = generate_mir_wasm(
        &module,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Unknown,
        PointerWidth::Bits32,
    )
    .expect_err("an unplaceable address must be refused, not zeroed");
    let message = error.to_string();
    assert!(message.contains("$p"), "{message}");
    assert!(message.contains("elsewhere"), "{message}");
}

#[test]
fn every_lane_width_the_backend_vectorizes_runs_under_wasmtime() {
    let Some(stdout) = run_wasm_main(
        "lane_widths",
        "fn @main() -> i64 {\n  entry:\n    %b8 = stackalloc.[16 x i8]\n    %v8 = splat.<16 x i8> 5\n    %w8 = splat.<16 x i8> 3\n    %a8 = add.<16 x i8> %v8, %w8\n    %s8 = sub.<16 x i8> %a8, %w8\n    %h8 = shl.<16 x i8> %s8, 2\n    %r8 = shr.<16 x i8> %h8, 1\n    %i8 = insertlane.<16 x i8> %r8, 9, 2\n    store.<16 x i8> %b8, %i8\n    %x8 = extractlane.<16 x i8> %i8, 3\n    %e8 = extend.i8.i64 %x8\n    call @print_i64(%e8)\n    %y8 = extractlane.<16 x i8> %i8, 2\n    %f8 = extend.i8.i64 %y8\n    call @print_i64(%f8)\n\n    %v16 = splat.<8 x i16> 7\n    %w16 = splat.<8 x i16> 6\n    %m16 = mul.<8 x i16> %v16, %w16\n    %h16 = shr.<8 x i16> %m16, 1\n    %d16 = sub.<8 x i16> %h16, %w16\n    %u16 = add.<8 x i16> %d16, %v16\n    %j16 = insertlane.<8 x i16> %u16, 11, 5\n    %x16 = extractlane.<8 x i16> %j16, 5\n    %e16 = extend.i16.i64 %x16\n    call @print_i64(%e16)\n    %z16 = extractlane.<8 x i16> %j16, 1\n    %g16 = extend.i16.i64 %z16\n    call @print_i64(%g16)\n\n    %v64 = splat.<2 x i64> 100\n    %w64 = splat.<2 x i64> 4\n    %m64 = mul.<2 x i64> %v64, %w64\n    %d64 = sub.<2 x i64> %m64, %w64\n    %a64 = add.<2 x i64> %d64, %w64\n    %h64 = shl.<2 x i64> %a64, 1\n    %j64 = insertlane.<2 x i64> %h64, 3, 0\n    %x64 = extractlane.<2 x i64> %j64, 1\n    call @print_i64(%x64)\n    %y64 = extractlane.<2 x i64> %j64, 0\n    call @print_i64(%y64)\n\n    %vf = splat.<2 x f64> 2.5\n    %wf = splat.<2 x f64> 4.0\n    %mf = mul.<2 x f64> %vf, %wf\n    %af = add.<2 x f64> %mf, %wf\n    %sf = sub.<2 x f64> %af, %vf\n    %qf = insertlane.<2 x f64> %sf, 6.5, 0\n    %xf = extractlane.<2 x f64> %qf, 1\n    %zf = extractlane.<2 x f64> %qf, 0\n    %bf = fptoint.f64.i64 %zf\n    call @print_i64(%bf)\n    %cf = fptoint.f64.i64 %xf\n    call @print_i64(%cf)\n\n    %v32 = splat.<4 x f32> 1.5\n    %w32 = splat.<4 x f32> 2.0\n    %a32 = add.<4 x f32> %v32, %w32\n    %s32 = sub.<4 x f32> %a32, %v32\n    %x32 = extractlane.<4 x f32> %s32, 2\n    %c32 = fptoint.f32.i64 %x32\n    call @print_i64(%c32)\n\n    %p32 = splat.<4 x i32> 9\n    %q32 = splat.<4 x i32> 2\n    %k32 = shl.<4 x i32> %p32, 2\n    %l32 = shr.<4 x i32> %k32, 1\n    %u8 = splat.<16 x u8> 200\n    %k8 = shr.<16 x u8> %u8, 2\n    %c8 = extractlane.<16 x u8> %k8, 0\n    %d8 = extend.u8.i64 %c8\n    call @print_i64(%d8)\n    %uu16 = splat.<8 x u16> 60000\n    %k16 = shr.<8 x u16> %uu16, 3\n    %c16 = extractlane.<8 x u16> %k16, 1\n    %d16b = extend.u16.i64 %c16\n    call @print_i64(%d16b)\n    %u64 = splat.<2 x u64> 64\n    %k64 = shr.<2 x u64> %u64, 2\n    %c64 = extractlane.<2 x u64> %k64, 0\n    %cc64 = bitcast.u64.i64 %c64\n    call @print_i64(%cc64)\n    %u32 = splat.<4 x u32> 80\n    %k32b = shr.<4 x u32> %u32, 3\n    %c32b = extractlane.<4 x u32> %k32b, 0\n    %d32b = extend.u32.i64 %c32b\n    call @print_i64(%d32b)\n    %t32 = splat.<4 x i32> 5\n    %r32 = or.<4 x i32> %l32, %t32\n    %n32 = extractlane.<4 x i32> %r32, 0\n    %o32 = extend.i32.i64 %n32\n    call @print_i64(%o32)\n    ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(
        stdout,
        "10\n9\n11\n22\n800\n3\n6\n11\n2\n50\n7500\n16\n10\n23\n0\n"
    );
}

#[test]
fn a_variadic_call_is_refused_by_name_rather_than_written_unloadable() {
    let source = "@extern\nfn @printf(ptr %fmt, ...) -> i32\n\nglobal $fmt: [3 x i8] = \"%d\"\n\nfn @main() -> i64 {\nentry:\n  %f = getelem.i8 $fmt, 0\n  %r = call @printf(%f, 42)\n  ret.i64 0\n}\n";
    let ir = parse_module(source).expect("source should parse");
    let mir = from_ir_x86_64(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    let error = generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits32,
    )
    .expect_err("a variadic call must be refused");
    let message = error.to_string();
    assert!(message.contains("@printf"), "{message}");
    assert!(message.contains("variadic"), "{message}");
}

fn emit_wat(source: &str, arch: TargetArchitecture) -> String {
    let ir = parse_module(source).expect("source should parse");
    let mir = from_ir_x86_64(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        arch,
        TargetOperatingSystem::Unknown,
        PointerWidth::Bits64,
    )
    .expect("WASM codegen should succeed");
    String::from_utf8(wat).expect("WAT should be UTF-8")
}

#[test]
fn wasm64_indexes_memory_with_i64() {
    let source = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.i64\n  store.i64 %p, 41\n  %v = load.i64 %p\n  ret.i64 %v\n}\n";

    // The address is the line before the memory operation, and only wasm32 narrows it.
    let narrowed_address = |wat: &str| -> bool {
        let lines: Vec<&str> = wat.lines().map(str::trim).collect();
        lines.windows(2).any(|pair| {
            pair[0] == "i32.wrap_i64" && (pair[1] == "i64.load" || pair[1] == "i64.store")
        })
    };

    let narrow = emit_wat(source, TargetArchitecture::Wasm32);
    assert!(
        narrow.contains("(memory (export \"memory\") 1)"),
        "{narrow}"
    );
    assert!(narrow.contains("(global $sp (mut i32)"), "{narrow}");
    assert!(narrowed_address(&narrow), "{narrow}");

    let wide = emit_wat(source, TargetArchitecture::Wasm64);
    assert!(
        wide.contains("(memory (export \"memory\") i64 1)"),
        "{wide}"
    );
    assert!(wide.contains("(global $sp (mut i64)"), "{wide}");
    assert!(!narrowed_address(&wide), "{wide}");
}

#[test]
fn module_imports_only_helpers_it_calls() {
    let quiet = emit_wat(
        "fn @main() -> i64 {\nentry:\n  ret.i64 7\n}\n",
        TargetArchitecture::Wasm32,
    );
    assert!(!quiet.contains("(import"), "{quiet}");

    let printing = emit_wat(
        "fn @main() -> i64 {\nentry:\n  %w = writebyte 55\n  ret.i64 0\n}\n",
        TargetArchitecture::Wasm32,
    );
    assert!(
        printing.contains("(import \"env\" \"writebyte\""),
        "{printing}"
    );
    assert!(!printing.contains("$malloc"), "{printing}");
}

#[test]
fn call_through_pointer_indexes_function_table() {
    let source = "fn @twice(i64 %x) -> i64 {\nentry:\n  %r = mul.i64 %x, 2\n  ret.i64 %r\n}\n\nfn @main() -> i64 {\nentry:\n  %f = funcaddr @twice\n  %r = callptr.i64 %f(i64 21)\n  ret.i64 %r\n}\n";
    let wat = emit_wat(source, TargetArchitecture::Wasm32);

    assert!(
        wat.contains("(type $sig_1_r (func (param i64) (result i64)))"),
        "{wat}"
    );
    assert!(wat.contains("(table 2 funcref)"), "{wat}");
    assert!(wat.contains("(elem (i32.const 0) $main $twice)"), "{wat}");
    assert!(wat.contains("call_indirect (type $sig_1_r)"), "{wat}");

    // @twice is the second element, so taking its address pushes 1.
    let address = wat
        .lines()
        .map(str::trim)
        .position(|line| line == "i64.const 1")
        .is_some();
    assert!(address, "{wat}");

    let quiet = emit_wat(
        "fn @main() -> i64 {\nentry:\n  ret.i64 0\n}\n",
        TargetArchitecture::Wasm32,
    );
    assert!(!quiet.contains("(table"), "{quiet}");
}

#[test]
fn wasi_writes_without_a_console_import() {
    let source = "fn @main() -> i64 {\nentry:\n  %a = writebyte 79\n  %b = writebyte 75\n  %c = writebyte 10\n  ret.i64 0\n}\n";
    let ir = parse_module(source).expect("source should parse");
    let mir = from_ir_x86_64(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )
    .expect("WASM codegen should succeed");
    let wat = String::from_utf8(wat).expect("WAT should be UTF-8");

    assert!(
        wat.contains("\"wasi_snapshot_preview1\" \"fd_write\""),
        "{wat}"
    );
    assert!(!wat.contains("\"console\""), "{wat}");
    assert!(wat.contains("(func (export \"_start\")"), "{wat}");

    if !tools_available(&["wasmtime"]) {
        return;
    }
    let guard = TempDir::new("lamina-wasi");
    let dir = guard.path();
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "OK\n");
}

#[test]
fn wasi_has_no_wasm64_form() {
    let ir = parse_module("fn @main() -> i64 {\nentry:\n  ret.i64 0\n}\n").expect("parse");
    let mir = from_ir_x86_64(&ir, "module").expect("lower");
    let mut wat = Vec::new();
    let error = generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm64,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )
    .expect_err("wasm64 WASI should be refused");
    assert!(format!("{error:?}").contains("no wasm64 form"), "{error:?}");
}

#[test]
fn imports_declare_the_types_the_host_wrote() {
    let source = "@extern\nfn @host_scale(f64 %x, i64 %n) -> f64\n\nfn @main() -> i64 {\nentry:\n  %r = call @host_scale(2.5, 3)\n  %b = fptoint.f64.i64 %r\n  ret.i64 %b\n}\n";
    let wat = emit_wat(source, TargetArchitecture::Wasm32);

    let import = wat
        .lines()
        .find(|line| line.contains("host_scale"))
        .expect("import should be emitted");
    assert!(
        import.contains("(param f64) (param i64) (result f64)"),
        "{wat}"
    );
    assert_eq!(
        import.matches('(').count(),
        import.matches(')').count(),
        "unbalanced import: {import}"
    );

    // The backend holds every value as i64 bits, so the boundary reinterprets both ways.
    assert!(wat.contains("f64.reinterpret_i64"), "{wat}");
    assert!(wat.contains("i64.reinterpret_f64"), "{wat}");
}

#[test]
fn narrow_loads_sign_extend_like_every_other_backend() {
    // x86, riscv, aarch64 and ppc all sign-extend a narrow load, so wasm must too or
    // the same program prints 255 on one target and -1 on the others.
    for (ty, signed) in [
        ("i8", "i64.load8_s"),
        ("i16", "i64.load16_s"),
        ("i32", "i64.load32_s"),
    ] {
        let source = format!(
            "fn @main() -> i64 {{\nentry:\n  %p = stackalloc.{ty}\n  store.{ty} %p, -1\n  %x = load.{ty} %p\n  %w = extend.{ty}.i64 %x\n  call @print_i64(%w)\n  ret.i64 0\n}}\n{PRINT_I64_HELPERS}"
        );
        let wat = emit_wat(&source, TargetArchitecture::Wasm32);
        assert!(wat.contains(signed), "{ty}: {wat}");
        assert!(!wat.contains(&signed.replace("_s", "_u")), "{ty}: {wat}");
    }

    if !tools_available(&["wasmtime"]) {
        return;
    }
    let source = format!(
        "fn @main() -> i64 {{\nentry:\n  %p = stackalloc.i8\n  store.i8 %p, -1\n  %x = load.i8 %p\n  %pv0 = extend.i8.i64 %x\n  call @print_i64(%pv0)\n  ret.i64 0\n}}\n{PRINT_I64_HELPERS}"
    );
    let ir = parse_module(&source).expect("parse");
    let mir = from_ir_x86_64(&ir, "module").expect("lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )
    .expect("codegen");
    let guard = TempDir::new("lamina-narrow");
    let dir = guard.path();
    let path = dir.join("module.wat");
    fs::write(&path, wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "-1\n");
}

#[test]
fn call_without_declaration_is_rejected() {
    let declared = b"(module\n  (func $helper (result i64)\n    i64.const 1\n  )\n  (func $main\n    call $helper\n    drop\n  )\n)\n";
    undeclared_call(declared).expect("every call is declared");

    let missing = b"(module\n  (func $main\n    call $memset\n    drop\n  )\n)\n";
    let error = undeclared_call(missing).expect_err("call with no declaration");
    assert!(error.to_string().contains("$memset"), "{error}");
}

#[test]
fn imported_function_counts_as_declared() {
    let imported = b"(module\n  (import \"env\" \"memset\" (func $memset (param i64 i64 i64)))\n  (func $main\n    call $memset\n  )\n)\n";
    undeclared_call(imported).expect("an import declares the name");
}

fn emit_wasi(source: &str) -> Result<String, LaminaError> {
    let ir = parse_module(source).expect("source should parse");
    let mir = from_ir_x86_64(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
        PointerWidth::Bits64,
    )?;
    Ok(String::from_utf8(wat).expect("WAT should be UTF-8"))
}

#[test]
fn wasi_buffered_io_declares_each_helper_it_calls() {
    let wat = emit_wasi(
        "fn @main() -> i64 {\nentry:\n  %p = stackalloc.i8\n  store.i8 %p, 72\n  %a = load.i8 %p\n  %b = writebyte 105\n  %c = readbyte\n  %d = write %p, 1\n  %e = read %p, 1\n  %f = writebyte %c\n  ret.i64 0\n}\n",
    )
    .expect("WASI codegen should succeed");

    assert!(!wat.contains("\"env\""), "{wat}");
    assert!(
        wat.contains("\"wasi_snapshot_preview1\" \"fd_write\""),
        "{wat}"
    );
    assert!(
        wat.contains("\"wasi_snapshot_preview1\" \"fd_read\""),
        "{wat}"
    );
    let declared = declared_names(&wat);
    for helper in ["writebyte", "readbyte", "write", "read"] {
        assert!(wat.contains(&format!("call ${helper}")), "{helper}: {wat}");
        assert!(
            declared.contains(helper),
            "{helper} is called but not declared: {wat}"
        );
    }
}

#[test]
fn wasi_rejects_heap_allocation_instead_of_importing_malloc() {
    let error = emit_wasi(
        "@extern\nfn @malloc(i64 %n) -> ptr\n@extern\nfn @free(ptr %p) -> void\nfn @main() -> i64 {\nentry:\n  %h = call @malloc(8)\n  call @free(%h)\n  ret.i64 0\n}\n",
    )
    .expect_err("heap allocation has no WASI form");
    let message = error.to_string();
    assert!(message.contains("call @malloc"), "{message}");
    assert!(message.contains("wasm32-wasi"), "{message}");
}

#[test]
fn wasm32_keeps_env_helpers() {
    let wat = emit_wat(
        "fn @main() -> i64 {\nentry:\n  %b = writebyte 105\n  ret.i64 0\n}\n",
        TargetArchitecture::Wasm32,
    );
    assert!(wat.contains("(import \"env\" \"writebyte\""), "{wat}");
}

#[test]
fn wasi_writebyte_reaches_stdout() {
    if !tools_available(&["wasmtime"]) {
        return;
    }
    let wat = emit_wasi(
        "fn @main() -> i64 {\nentry:\n  %p = stackalloc.i8\n  store.i8 %p, 72\n  %v = load.i8 %p\n  %a = writebyte %v\n  %b = writebyte 105\n  %c = writebyte 10\n  ret.i64 0\n}\n",
    )
    .expect("WASI codegen should succeed");
    let guard = TempDir::new("lamina-wasi-io");
    let dir = guard.path();
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "Hi\n");
}

#[test]
fn wasi_memcpy_and_memset_fill_the_module_memory() {
    let wat = emit_wasi(
        &[
        "@extern\nfn @memcpy(ptr %d, ptr %s, i64 %n) -> ptr\n@extern\nfn @memset(ptr %d, i32 %v, i64 %n) -> ptr\nfn @main() -> i64 {\nentry:\n  %src = stackalloc.i64\n  %dst = stackalloc.i64\n  store.i64 %src, 578437695752307201\n  call @memset(%dst, 165, 8)\n  call @memcpy(%dst, %src, 4)\n  %x = load.i64 %dst\n  call @print_i64(%x)\n  ret.i64 0\n}\n",
            PRINT_I64_HELPERS,
        ]
        .concat(),
    )
    .expect("WASI codegen should succeed");
    assert!(!wat.contains("\"env\""), "{wat}");
    assert!(wat.contains("memory.copy"), "{wat}");
    assert!(wat.contains("memory.fill"), "{wat}");
    assert!(declared_names(&wat).contains("memcpy"), "{wat}");
    assert!(declared_names(&wat).contains("memset"), "{wat}");

    if !tools_available(&["wasmtime"]) {
        return;
    }
    let guard = TempDir::new("lamina-wasi-mem");
    let dir = guard.path();
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    assert!(output.status.success(), "{output:?}");
    let expected = format!("{}\n", 0xA5A5_A5A5_0403_0201u64 as i64);
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
}

#[test]
fn wasi_io_outcomes_reach_stdout_as_their_low_byte() {
    if !tools_available(&["wasmtime"]) {
        return;
    }
    let reads = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 1\n  %w = writebyte %r\n  ret.i64 0\n}\n";
    let readbyte =
        "fn @main() -> i64 {\nentry:\n  %r = readbyte\n  %w = writebyte %r\n  ret.i64 0\n}\n";
    let zero_read = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 0\n  %w = writebyte %r\n  ret.i64 0\n}\n";
    let cases: [(&str, &str, &[u8], &[u8]); 7] = [
        ("readbyte-byte", readbyte, b"A", &[65]),
        ("readbyte-end-of-input", readbyte, b"", &[255]),
        ("readbyte-nul", readbyte, b"\0", &[0]),
        ("read-count", reads, b"x", &[1]),
        ("read-end-of-input", reads, b"", &[255]),
        ("read-zero-length", zero_read, b"", &[0]),
        (
            "writebyte-count",
            "fn @main() -> i64 {\nentry:\n  %r = writebyte 65\n  %w = writebyte %r\n  ret.i64 0\n}\n",
            b"",
            &[65, 1],
        ),
    ];
    for (tag, source, stdin, expected) in cases {
        let wat = emit_wasi(source).expect("WASI codegen should succeed");
        let guard = TempDir::new(&format!("lamina-wasi-io-{tag}"));
        let dir = guard.path();
        let path = dir.join("module.wat");
        fs::write(&path, &wat).expect("write wat");
        let mut child = Command::new("wasmtime")
            .arg("run")
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("run wasmtime");
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(stdin)
            .expect("write stdin");
        let output = child.wait_with_output().expect("wait for wasmtime");
        assert!(output.status.success(), "{tag}: {output:?}");
        assert_eq!(output.stdout, expected, "{tag}");
    }
}

#[test]
fn pointer_imports_use_the_memory_index_type() {
    let source = "@extern\nfn @takes(ptr %p) -> ptr\n\nfn @main() -> i64 {\nentry:\n  %p = getelem.i8 $text, 0\n  %r = call @takes(%p)\n  %v = load.i8 %r\n  %w = extend.i8.i64 %v\n  ret.i64 %w\n}\n\nglobal $text: [6 x i8] = \"abcde\"\n";

    let wat32 = emit_wat(source, TargetArchitecture::Wasm32);
    let import = wat32
        .lines()
        .find(|line| line.contains("takes"))
        .expect("import should be emitted");
    assert!(import.contains("(param i32) (result i32)"), "{wat32}");
    assert!(wat32.contains("i32.wrap_i64"), "{wat32}");
    assert!(wat32.contains("i64.extend_i32_u"), "{wat32}");

    let wat64 = emit_wat(source, TargetArchitecture::Wasm64);
    let import = wat64
        .lines()
        .find(|line| line.contains("takes"))
        .expect("import should be emitted");
    assert!(import.contains("(param i64) (result i64)"), "{wat64}");
}

#[test]
fn pointer_imports_called_indirectly_go_through_a_thunk() {
    let source = "@extern\nfn @takes(ptr %p) -> i64\n\nfn @main() -> i64 {\nentry:\n  %p = getelem.i8 $text, 0\n  %f = funcaddr @takes\n  %r = callptr.i64 %f(ptr %p)\n  ret.i64 %r\n}\n\nglobal $text: [6 x i8] = \"abcde\"\n";
    let wat = emit_wat(source, TargetArchitecture::Wasm32);
    assert!(
        wat.contains("(func $takes.thunk (param i64) (result i64)"),
        "{wat}"
    );
    assert!(
        wat.contains("(elem (i32.const 0) $main $takes.thunk)"),
        "{wat}"
    );
}
