use super::*;
use crate::mir::{FunctionBuilder, TransformPipeline, codegen::from_ir};
use crate::mir_codegen::simd_support::tool_available;
use crate::parser::parse_module;
use std::{env, fs, process::Command};

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
    generate_mir_wasm(&module, &mut output, TargetOperatingSystem::Unknown)
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
    generate_mir_wasm(&module, &mut output, TargetOperatingSystem::Unknown)
        .expect("WASM codegen should succeed");
    let assembly = String::from_utf8(output).expect("assembly should be UTF-8");

    assert!(
        assembly.contains("i64.const 1\n    local.set $pc"),
        "dispatcher should start at the entry block index: {assembly}"
    );
}

const WASM_RUN_PY: &str = r#"import sys, wasmtime
engine = wasmtime.Engine()
store = wasmtime.Store(engine)
module = wasmtime.Module.from_file(engine, sys.argv[1])
linker = wasmtime.Linker(engine)
for imp in module.imports:
    def callback(*args, name=imp.name):
        if name == 'log':
            print(*args)
            return None
        if name == 'logu':
            print(args[0] & 0xffffffffffffffff)
            return None
        if name == 'logf':
            print('%g' % args[0])
            return None
        return None if name == 'free' else 0
    linker.define(store, imp.module, imp.name,
                  wasmtime.Func(store, imp.type, callback))
instance = linker.instantiate(store, module)
print('main returned', instance.exports(store)['main'](store))
"#;

/// Runs `main` under wasmtime and returns what it printed, or `None` when the
/// runner is not available. uv fetches wasmtime on demand, which only the Linux
/// leg of CI is set up for.
fn run_wasm_main(test_name: &str, source: &str, opt_level: u8) -> Option<String> {
    if !cfg!(target_os = "linux") || !tool_available("uv") {
        return None;
    }
    let ir = parse_module(source).expect("source should parse");
    let mut mir = from_ir(&ir, "module").expect("source should lower");
    TransformPipeline::default_for_opt_level(opt_level)
        .apply_to_module(&mut mir)
        .expect("transforms should apply");
    let mut wat = Vec::new();
    generate_mir_wasm(&mir, &mut wat, TargetOperatingSystem::Unknown)
        .expect("WASM codegen should succeed");
    let dir = env::temp_dir().join(format!("lamina-wasm-{test_name}-{}", id()));
    fs::create_dir_all(&dir).expect("create test dir");
    let wat_path = dir.join("module.wat");
    let run_path = dir.join("run.py");
    fs::write(&wat_path, wat).expect("write wat");
    fs::write(&run_path, WASM_RUN_PY).expect("write runner");
    let output = Command::new("uv")
        .args(["run", "--with", "wasmtime", "python"])
        .arg(&run_path)
        .arg(&wat_path)
        .output()
        .expect("run uv");
    fs::remove_dir_all(&dir).expect("cleanup");
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

#[test]
fn distinct_stack_slots_in_one_frame_do_not_alias() {
    let Some(stdout) = run_wasm_main(
        "slots",
        "fn @main() -> i64 {\nentry:\n %p = alloc.stack i64\n %q = alloc.stack i64\n store.i64 %p, 11\n store.i64 %q, 22\n %x = load.i64 %p\n print %x\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "11\nmain returned 0\n");
}

#[test]
fn recursive_invocations_keep_their_own_stack_slots() {
    let Some(stdout) = run_wasm_main(
        "recursive",
        "fn @rec(i64 %n) -> i64 {\nentry:\n %p = alloc.stack i64\n store.i64 %p, %n\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n ret.i64 0\nstep:\n %m = sub.i64 %n, 1\n %r = call @rec(%m)\n %x = load.i64 %p\n %s = add.i64 %x, %r\n ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n %r = call @rec(3)\n ret.i64 %r\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "main returned 6\n");
}

#[test]
fn stack_frames_are_released_on_every_return() {
    let Some(stdout) = run_wasm_main(
        "release",
        "fn @leaf(i64 %n) -> i64 {\nentry:\n %p = alloc.stack i64\n store.i64 %p, %n\n %x = load.i64 %p\n ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n %i = add.i64 0, 0\n jmp loop\nloop:\n %r = call @leaf(%i)\n %i = add.i64 %i, 1\n %c = lt.i64 %i, 6000\n br %c, loop, done\ndone:\n ret.i64 %i\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "main returned 6000\n");
}

#[test]
fn signed_byte_compares_on_loaded_operands_read_the_sign() {
    let Some(stdout) = run_wasm_main(
        "cmp8",
        "fn @main() -> i64 {\nentry:\n %p = alloc.stack i8\n store.i8 %p, -1\n %x = load.i8 %p\n %a = lt.i8 %x, 0\n print %a\n %b = eq.i8 %x, -1\n print %b\n %c = lt.u8 %x, 1\n print %c\n %d = gt.i8 %x, -2\n print %d\n %e = gt.u8 %x, 1\n print %e\n %f = add.i8 %x, 0\n %g = lt.i8 %f, 0\n print %g\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "1\n1\n0\n1\n1\n1\nmain returned 0\n");
}

#[test]
fn narrow_compares_normalize_operands_at_their_width() {
    let Some(stdout) = run_wasm_main(
        "cmpwidths",
        "fn @main() -> i64 {\nentry:\n %p8 = alloc.stack i8\n %p16 = alloc.stack i16\n %p32 = alloc.stack i32\n store.i8 %p8, -128\n store.i16 %p16, -32768\n store.i32 %p32, -2147483648\n %a8 = load.i8 %p8\n %a16 = load.i16 %p16\n %a32 = load.i32 %p32\n %s8 = lt.i8 %a8, 0\n print %s8\n %s16 = lt.i16 %a16, 0\n print %s16\n %s32 = le.i32 %a32, -2147483648\n print %s32\n %u8 = lt.u8 %a8, 127\n print %u8\n %u16 = gt.u16 %a16, 32767\n print %u16\n %e8 = eq.i8 %a8, -128\n print %e8\n %c8 = add.i8 %a8, 0\n %g8 = gt.i8 %c8, -127\n print %g8\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "1\n1\n1\n0\n1\n1\n0\nmain returned 0\n");
}

#[test]
fn narrow_arguments_keep_their_declared_signedness_across_calls() {
    let Some(stdout) = run_wasm_main(
        "boundary_signedness",
        "fn @mk() -> char {\nentry:\n %v = trunc.i64.char 255\n ret.char %v\n}\nfn @take(u8 %u, char %c) -> i64 {\nentry:\n %a = lt.u8 %u, 1\n print %a\n %b = lt.char %c, 0\n print %b\n ret.i64 0\n}\nfn @main() -> i64 {\nentry:\n %c = call @mk()\n %u = trunc.i64.u8 255\n %r = call @take(%u, %c)\n ret.i64 0\n}",
        0,
    ) else {
        return;
    };
    assert_eq!(stdout, "0\n1\nmain returned 0\n");
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
        assert_eq!(stdout, "main returned 9\n", "opt level {opt_level}");
    }
}

#[test]
fn tail_call_with_caller_frame_argument_reads_caller_storage() {
    let Some(stdout) = run_wasm_main(
        "tail_frame_arg",
        "fn @g(ptr %p, i64 %n) -> i64 {\nentry:\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n %q = alloc.stack i64\n store.i64 %q, 99\n %x = load.i64 %p\n %y = load.i64 %q\n %z = add.i64 %x, %y\n ret.i64 %z\nstep:\n %m = sub.i64 %n, 1\n %r = call @g(%p, %m)\n ret.i64 %r\n}\nfn @top(i64 %n, i64 %d) -> i64 {\nentry:\n %p = alloc.stack i64\n store.i64 %p, 7\n %r = call @g(%p, %n)\n ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n %k = add.i64 0, 3\n %r = call @top(%k, %k)\n ret.i64 %r\n}",
        2,
    ) else {
        return;
    };
    assert_eq!(stdout, "main returned 106\n");
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
    assert_eq!(stdout, "main returned 42\n");
}
