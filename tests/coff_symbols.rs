#![allow(clippy::expect_used)]

use lamina::api::{Target, compile_module_to_object};
use lamina::parser::parse_module;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::{self, FromStr};
use std::{env, fs};

const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;
const IMAGE_SYM_CLASS_WEAK_EXTERNAL: u8 = 105;
const IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY: u32 = 1;

struct CoffSymbol {
    name: String,
    value: u32,
    section: i16,
    class: u8,
    aux: Vec<u8>,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// Every record of a COFF symbol table, each carrying the auxiliary records that
/// follow it. The index of a record is its own symbol-table index, so auxiliary
/// records count towards it exactly as `NumberOfSymbols` counts them.
fn coff_symbols(object: &[u8]) -> Vec<Option<CoffSymbol>> {
    let sym_off = u32_at(object, 8) as usize;
    let count = u32_at(object, 12) as usize;
    let str_off = sym_off + count * 18;
    let mut out: Vec<Option<CoffSymbol>> = Vec::new();
    let mut index = 0;
    while index < count {
        let entry = sym_off + index * 18;
        let raw = &object[entry..entry + 18];
        let name = if u32_at(raw, 0) == 0 {
            let at = str_off + u32_at(raw, 4) as usize;
            let end = object[at..]
                .iter()
                .position(|&b| b == 0)
                .expect("nul terminated");
            str::from_utf8(&object[at..at + end])
                .expect("utf-8 name")
                .to_owned()
        } else {
            let end = raw[..8].iter().position(|&b| b == 0).unwrap_or(8);
            str::from_utf8(&raw[..end]).expect("utf-8 name").to_owned()
        };
        let aux_count = raw[17] as usize;
        let aux = object[entry + 18..entry + 18 + aux_count * 18].to_vec();
        out.push(Some(CoffSymbol {
            name,
            value: u32_at(raw, 8),
            section: u16_at(raw, 12) as i16,
            class: raw[16],
            aux,
        }));
        out.extend((0..aux_count).map(|_| None));
        index += 1 + aux_count;
    }
    assert_eq!(out.len(), count, "NumberOfSymbols counts auxiliary records");
    out
}

/// The weak external's default definition. ras names it after the weak symbol
/// and the object's first strong external, which is its business to pin; what
/// matters here is that one exists and the auxiliary record points at it.
fn find_default<'a>(symbols: &'a [Option<CoffSymbol>], weak: &str) -> (usize, &'a CoffSymbol) {
    let prefix = format!(".weak.{weak}");
    symbols
        .iter()
        .enumerate()
        .find_map(|(i, s)| match s {
            Some(s) if s.name.starts_with(&prefix) => Some((i, s)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("default definition for weak {weak} in COFF symbol table"))
}

fn find<'a>(symbols: &'a [Option<CoffSymbol>], name: &str) -> (usize, &'a CoffSymbol) {
    symbols
        .iter()
        .enumerate()
        .find_map(|(i, s)| match s {
            Some(s) if s.name == name => Some((i, s)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("symbol {name} in COFF symbol table"))
}

fn windows_object(ir: &str) -> Vec<u8> {
    let target = Target::from_str("x86_64_windows").expect("target string parses");
    let module = parse_module(ir).expect("module parses");
    compile_module_to_object(&module, target).expect("COFF object")
}

fn windows_error(ir: &str) -> String {
    let target = Target::from_str("x86_64_windows").expect("target string parses");
    let module = parse_module(ir).expect("module parses");
    compile_module_to_object(&module, target)
        .expect_err("COFF object is refused")
        .to_string()
}

const WEAK_WHICH: &str = r#"
@weak
fn @which() -> i64 {
entry:
    ret.i64 1
}

fn @main() -> i64 {
entry:
    %v = call @which()
    ret.i64 %v
}
"#;

#[test]
fn weak_function_is_a_coff_weak_external_with_its_auxiliary_record() {
    let object = windows_object(WEAK_WHICH);
    let symbols = coff_symbols(&object);

    let (default_index, default) = find_default(&symbols, "which");
    assert_eq!(
        default.class, IMAGE_SYM_CLASS_EXTERNAL,
        "link.exe takes only an external default"
    );
    assert!(
        default.section > 0,
        "the default definition is defined here"
    );

    let (_, weak) = find(&symbols, "which");
    assert_eq!(weak.class, IMAGE_SYM_CLASS_WEAK_EXTERNAL);
    assert_eq!(weak.section, 0, "a weak external is undefined");
    assert_eq!(weak.value, 0);
    assert_eq!(weak.aux.len(), 18, "exactly one auxiliary record");
    assert_eq!(
        u32_at(&weak.aux, 0),
        default_index as u32,
        "the auxiliary record names the default definition"
    );
    assert_eq!(
        u32_at(&weak.aux, 4),
        IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY,
        "search characteristics"
    );
    assert!(
        weak.aux[8..].iter().all(|&b| b == 0),
        "the rest of the auxiliary record is unused"
    );
}

#[test]
fn visibility_that_is_not_exported_becomes_a_file_local_coff_symbol() {
    for annotation in ["@hidden", "@internal", "@private"] {
        let ir = format!(
            "{annotation}\nfn @quiet() -> i64 {{\nentry:\n    ret.i64 7\n}}\n\nfn @main() -> i64 {{\nentry:\n    %v = call @quiet()\n    ret.i64 %v\n}}\n"
        );
        let object = windows_object(&ir);
        let symbols = coff_symbols(&object);
        assert_eq!(
            find(&symbols, "quiet").1.class,
            IMAGE_SYM_CLASS_STATIC,
            "{annotation}"
        );
        assert_eq!(
            find(&symbols, "main").1.class,
            IMAGE_SYM_CLASS_EXTERNAL,
            "{annotation}"
        );
    }
}

#[test]
fn protected_visibility_is_refused_on_coff() {
    for subject in [
        "@protected\nfn @p() -> i64 {\nentry:\n    ret.i64 1\n}\n",
        "@protected\nglobal $g: i64 = 1\n",
    ] {
        let ir = format!("{subject}\nfn @main() -> i64 {{\nentry:\n    ret.i64 0\n}}\n");
        let message = windows_error(&ir);
        assert!(message.contains("@protected"), "{message}");
        assert!(message.contains("no protected visibility"), "{message}");
    }
}

#[test]
fn weak_together_with_an_unexported_visibility_is_refused_on_coff() {
    let ir = "@weak\n@hidden\nfn @w() -> i64 {\nentry:\n    ret.i64 1\n}\n\nfn @main() -> i64 {\nentry:\n    ret.i64 0\n}\n";
    let message = windows_error(ir);
    assert!(message.contains("@weak"), "{message}");
    assert!(message.contains("weak external"), "{message}");
}

fn tool(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

fn skip(reason: &str) -> bool {
    let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|v| v == "1");
    assert!(!required, "{reason} (LAMINA_REQUIRE_TOOLS=1)");
    eprintln!("skipped: {reason}");
    true
}

/// The mingw-w64 runtime directory that holds `crt2.o`, and the gcc directory
/// beside it, or `None` when mingw-w64 is not installed.
fn mingw_dirs() -> Option<(PathBuf, PathBuf)> {
    let gcc = fs::read_dir("/usr/lib/gcc/x86_64-w64-mingw32")
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join("crtbegin.o").exists())?;
    let lib = [
        "/usr/x86_64-w64-mingw32/lib",
        "/usr/lib/x86_64-w64-mingw32/lib",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|dir| dir.join("crt2.o").exists())?;
    Some((gcc, lib))
}

fn link_and_run(out: &Path, gcc: &Path, lib: &Path, objects: &[&Path]) -> i32 {
    let mut link = Command::new("ld.lld-18");
    link.args(["-m", "i386pep", "-Bdynamic", "-o"])
        .arg(out)
        .arg(lib.join("crt2.o"))
        .arg(gcc.join("crtbegin.o"))
        .arg(format!("-L{}", gcc.display()))
        .arg(format!("-L{}", lib.display()))
        .args(objects)
        .args([
            "-lmingw32",
            "-lgcc",
            "-lgcc_eh",
            "-lmoldname",
            "-lmingwex",
            "-lmsvcrt",
            "-lkernel32",
        ])
        .arg(gcc.join("crtend.o"));
    let linked = link.output().expect("run ld.lld-18");
    assert!(
        linked.status.success(),
        "link: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new("wine").arg(out).output().expect("run wine");
    run.status.code().expect("exit code")
}

#[test]
fn a_coff_weak_definition_loses_to_a_strong_one_at_run_time() {
    if !tool("ld.lld-18") && skip("ld.lld-18 not installed") {
        return;
    }
    if !tool("x86_64-w64-mingw32-gcc") && skip("x86_64-w64-mingw32-gcc not installed") {
        return;
    }
    if !tool("wine") && skip("wine not installed") {
        return;
    }
    let Some((gcc, lib)) = mingw_dirs() else {
        skip("mingw-w64 runtime objects not installed");
        return;
    };

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("coff-weak-run");
    fs::create_dir_all(&dir).expect("scratch directory");

    let weak = dir.join("weak.obj");
    fs::write(&weak, windows_object(WEAK_WHICH)).expect("write weak object");

    let source = dir.join("strong.c");
    fs::write(&source, "long long which(void){ return 2; }\n").expect("write strong source");
    let strong = dir.join("strong.obj");
    let compiled = Command::new("x86_64-w64-mingw32-gcc")
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&strong)
        .output()
        .expect("run mingw gcc");
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    assert_eq!(
        link_and_run(&dir.join("alone.exe"), &gcc, &lib, &[&weak]),
        1,
        "the weak definition is used when nothing else defines the symbol"
    );
    assert_eq!(
        link_and_run(&dir.join("overridden.exe"), &gcc, &lib, &[&weak, &strong]),
        2,
        "a strong definition in another object overrides the weak one"
    );
}
