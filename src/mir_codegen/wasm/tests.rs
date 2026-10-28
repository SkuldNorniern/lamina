use super::*;
use crate::mir::{FunctionBuilder, TransformPipeline, codegen::from_ir};
use crate::mir_codegen::{simd_support::tool_available, test_support::PRINT_I64_HELPERS};
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
    generate_mir_wasm(
        &module,
        &mut output,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Unknown,
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
    )
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
        if name == 'writebyte':
            sys.stdout.flush()
            sys.stdout.buffer.write(bytes([args[0] & 0xff]))
            sys.stdout.buffer.flush()
            return 1
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
    let source = format!("{source}\n{PRINT_I64_HELPERS}");
    let ir = parse_module(&source).expect("source should parse");
    let mut mir = from_ir(&ir, "module").expect("source should lower");
    TransformPipeline::default_for_opt_level(opt_level)
        .apply_to_module(&mut mir)
        .expect("transforms should apply");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Unknown,
    )
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
        "fn @main() -> i64 {\nentry:\n %p = stackalloc.i64\n %q = stackalloc.i64\n store.i64 %p, 11\n store.i64 %q, 22\n %x = load.i64 %p\n call @print_i64(%x)\n ret.i64 0\n}",
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
        "fn @rec(i64 %n) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, %n\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n ret.i64 0\nstep:\n %m = sub.i64 %n, 1\n %r = call @rec(%m)\n %x = load.i64 %p\n %s = add.i64 %x, %r\n ret.i64 %s\n}\nfn @main() -> i64 {\nentry:\n %r = call @rec(3)\n ret.i64 %r\n}",
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
        "fn @leaf(i64 %n) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, %n\n %x = load.i64 %p\n ret.i64 %x\n}\nfn @main() -> i64 {\nentry:\n %i = add.i64 0, 0\n jmp loop\nloop:\n %r = call @leaf(%i)\n %i = add.i64 %i, 1\n %c = lt.i64 %i, 6000\n br %c, loop, done\ndone:\n ret.i64 %i\n}",
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
        "fn @main() -> i64 {\nentry:\n %p = stackalloc.i8\n store.i8 %p, -1\n %x = load.i8 %p\n %a = lt.i8 %x, 0\n %pv0 = zext.bool.i64 %a\n call @print_i64(%pv0)\n %b = eq.i8 %x, -1\n %pv1 = zext.bool.i64 %b\n call @print_i64(%pv1)\n %c = lt.u8 %x, 1\n %pv2 = zext.bool.i64 %c\n call @print_i64(%pv2)\n %d = gt.i8 %x, -2\n %pv3 = zext.bool.i64 %d\n call @print_i64(%pv3)\n %e = gt.u8 %x, 1\n %pv4 = zext.bool.i64 %e\n call @print_i64(%pv4)\n %f = add.i8 %x, 0\n %g = lt.i8 %f, 0\n %pv5 = zext.bool.i64 %g\n call @print_i64(%pv5)\n ret.i64 0\n}",
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
        "fn @main() -> i64 {\nentry:\n %p8 = stackalloc.i8\n %p16 = stackalloc.i16\n %p32 = stackalloc.i32\n store.i8 %p8, -128\n store.i16 %p16, -32768\n store.i32 %p32, -2147483648\n %a8 = load.i8 %p8\n %a16 = load.i16 %p16\n %a32 = load.i32 %p32\n %s8 = lt.i8 %a8, 0\n %pv0 = zext.bool.i64 %s8\n call @print_i64(%pv0)\n %s16 = lt.i16 %a16, 0\n %pv1 = zext.bool.i64 %s16\n call @print_i64(%pv1)\n %s32 = le.i32 %a32, -2147483648\n %pv2 = zext.bool.i64 %s32\n call @print_i64(%pv2)\n %u8 = lt.u8 %a8, 127\n %pv3 = zext.bool.i64 %u8\n call @print_i64(%pv3)\n %u16 = gt.u16 %a16, 32767\n %pv4 = zext.bool.i64 %u16\n call @print_i64(%pv4)\n %e8 = eq.i8 %a8, -128\n %pv5 = zext.bool.i64 %e8\n call @print_i64(%pv5)\n %c8 = add.i8 %a8, 0\n %g8 = gt.i8 %c8, -127\n %pv6 = zext.bool.i64 %g8\n call @print_i64(%pv6)\n ret.i64 0\n}",
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
        "fn @mk() -> i8 {\nentry:\n %v = trunc.i64.i8 255\n ret.i8 %v\n}\nfn @take(u8 %u, i8 %c) -> i64 {\nentry:\n %a = lt.u8 %u, 1\n %pv0 = zext.bool.i64 %a\n call @print_i64(%pv0)\n %b = lt.i8 %c, 0\n %pv1 = zext.bool.i64 %b\n call @print_i64(%pv1)\n ret.i64 0\n}\nfn @main() -> i64 {\nentry:\n %c = call @mk()\n %u = trunc.i64.u8 255\n %r = call @take(%u, %c)\n ret.i64 0\n}",
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
        "fn @g(ptr %p, i64 %n) -> i64 {\nentry:\n %c = eq.i64 %n, 0\n br %c, base, step\nbase:\n %q = stackalloc.i64\n store.i64 %q, 99\n %x = load.i64 %p\n %y = load.i64 %q\n %z = add.i64 %x, %y\n ret.i64 %z\nstep:\n %m = sub.i64 %n, 1\n %r = call @g(%p, %m)\n ret.i64 %r\n}\nfn @top(i64 %n, i64 %d) -> i64 {\nentry:\n %p = stackalloc.i64\n store.i64 %p, 7\n %r = call @g(%p, %n)\n ret.i64 %r\n}\nfn @main() -> i64 {\nentry:\n %k = add.i64 0, 3\n %r = call @top(%k, %k)\n ret.i64 %r\n}",
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

fn emit_wat(source: &str, arch: TargetArchitecture) -> String {
    let ir = parse_module(source).expect("source should parse");
    let mir = from_ir(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(&mir, &mut wat, arch, TargetOperatingSystem::Unknown)
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
    let mir = from_ir(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
    )
    .expect("WASM codegen should succeed");
    let wat = String::from_utf8(wat).expect("WAT should be UTF-8");

    assert!(
        wat.contains("\"wasi_snapshot_preview1\" \"fd_write\""),
        "{wat}"
    );
    assert!(!wat.contains("\"console\""), "{wat}");
    assert!(wat.contains("(func (export \"_start\")"), "{wat}");

    if !tool_available("wasmtime") {
        return;
    }
    let dir = env::temp_dir().join(format!("lamina-wasi-{}", id()));
    fs::create_dir_all(&dir).expect("create test dir");
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    fs::remove_dir_all(&dir).expect("cleanup");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "OK\n");
}

#[test]
fn wasi_has_no_wasm64_form() {
    let ir = parse_module("fn @main() -> i64 {\nentry:\n  ret.i64 0\n}\n").expect("parse");
    let mir = from_ir(&ir, "module").expect("lower");
    let mut wat = Vec::new();
    let error = generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm64,
        TargetOperatingSystem::Wasi,
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
            "fn @main() -> i64 {{\nentry:\n  %p = stackalloc.{ty}\n  store.{ty} %p, -1\n  %x = load.{ty} %p\n  %w = sext.{ty}.i64 %x\n  call @print_i64(%w)\n  ret.i64 0\n}}\n{PRINT_I64_HELPERS}"
        );
        let wat = emit_wat(&source, TargetArchitecture::Wasm32);
        assert!(wat.contains(signed), "{ty}: {wat}");
        assert!(!wat.contains(&signed.replace("_s", "_u")), "{ty}: {wat}");
    }

    if !tool_available("wasmtime") {
        return;
    }
    let source = format!(
        "fn @main() -> i64 {{\nentry:\n  %p = stackalloc.i8\n  store.i8 %p, -1\n  %x = load.i8 %p\n  %pv0 = sext.i8.i64 %x\n  call @print_i64(%pv0)\n  ret.i64 0\n}}\n{PRINT_I64_HELPERS}"
    );
    let ir = parse_module(&source).expect("parse");
    let mir = from_ir(&ir, "module").expect("lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
    )
    .expect("codegen");
    let dir = env::temp_dir().join(format!("lamina-narrow-{}", id()));
    fs::create_dir_all(&dir).expect("create test dir");
    let path = dir.join("module.wat");
    fs::write(&path, wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    fs::remove_dir_all(&dir).expect("cleanup");
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
    let mir = from_ir(&ir, "module").expect("source should lower");
    let mut wat = Vec::new();
    generate_mir_wasm(
        &mir,
        &mut wat,
        TargetArchitecture::Wasm32,
        TargetOperatingSystem::Wasi,
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
        "fn @main() -> i64 {\nentry:\n  %h = heapalloc.i64\n  heapfree %h\n  ret.i64 0\n}\n",
    )
    .expect_err("heap allocation has no WASI form");
    let message = error.to_string();
    assert!(message.contains("heapalloc"), "{message}");
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
    if !tool_available("wasmtime") {
        return;
    }
    let wat = emit_wasi(
        "fn @main() -> i64 {\nentry:\n  %p = stackalloc.i8\n  store.i8 %p, 72\n  %v = load.i8 %p\n  %a = writebyte %v\n  %b = writebyte 105\n  %c = writebyte 10\n  ret.i64 0\n}\n",
    )
    .expect("WASI codegen should succeed");
    let dir = env::temp_dir().join(format!("lamina-wasi-io-{}", id()));
    fs::create_dir_all(&dir).expect("create test dir");
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    fs::remove_dir_all(&dir).expect("cleanup");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "Hi\n");
}

#[test]
fn wasi_memcpy_and_memset_fill_the_module_memory() {
    let wat = emit_wasi(
        &[
        "fn @main() -> i64 {\nentry:\n  %src = stackalloc.i64\n  %dst = stackalloc.i64\n  store.i64 %src, 578437695752307201\n  memset %dst, 165, 8\n  memcpy %dst, %src, 4\n  %x = load.i64 %dst\n  call @print_i64(%x)\n  ret.i64 0\n}\n",
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

    if !tool_available("wasmtime") {
        return;
    }
    let dir = env::temp_dir().join(format!("lamina-wasi-mem-{}", id()));
    fs::create_dir_all(&dir).expect("create test dir");
    let path = dir.join("module.wat");
    fs::write(&path, &wat).expect("write wat");
    let output = Command::new("wasmtime")
        .arg("run")
        .arg(&path)
        .output()
        .expect("run wasmtime");
    fs::remove_dir_all(&dir).expect("cleanup");
    assert!(output.status.success(), "{output:?}");
    let expected = format!("{}\n", 0xA5A5_A5A5_0403_0201u64 as i64);
    assert_eq!(String::from_utf8_lossy(&output.stdout), expected);
}
