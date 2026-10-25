use super::test_support::compile;
use crate::{
    mir::{Immediate, Instruction, Operand, codegen::from_ir},
    parser::parse_module,
};
use std::{
    env::temp_dir,
    error::Error,
    fs,
    process::{self, Command},
};

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn aggregate_stack_storage_preserves_live_neighbors() -> Result<(), Box<dyn Error>> {
    let dir = temp_dir().join(format!("lamina-aggregate-storage-{}", process::id()));
    fs::create_dir_all(&dir)?;
    let cases = [
        ("@Pair", "%last = getfieldptr %p, 1"),
        (
            "struct { a: i8, b: i64, c: i64 }",
            "%last = getfieldptr %p, 2",
        ),
        ("tuple i8, i64, i64", "%last = getfieldptr %p, 2"),
        (
            "struct { a: i8, inner: @Pair }",
            "%inner = getfieldptr %p, 1\n%last = getfieldptr %inner, 1",
        ),
        (
            "struct { inner: @Pair, tail: i64 }",
            "%last = getfieldptr %p, 1",
        ),
        (
            "[3 x @Pair]",
            "%element = getelem.ptr %p, 2, @Pair\n%last = getfieldptr %element, 1",
        ),
        (
            "[3 x struct { a: i64, b: i64 }]",
            "%element = getelem.ptr %p, 2, struct { a: i64, b: i64 }\n%last = getfieldptr %element, 1",
        ),
    ];
    for (case, (ty, address)) in cases.iter().enumerate() {
        let source = format!(
            "type @Pair = struct {{ a: i64, b: i64 }}\nfn @main() -> i64 {{\nentry:\n%p = alloc.stack {ty}\n{address}\n%sentinel = add.i64 55, 0\n%guard = alloc.stack i64\nstore.i64 %guard, 55\nstore.i64 %last, 99\nprint %sentinel\n%live = load.i64 %guard\nprint %live\n%stored = load.i64 %last\nprint %stored\nret.i64 0\n}}\n"
        );
        let ir = parse_module(&source)?;
        let mir = from_ir(&ir, "aggregate").map_err(|error| error.to_string())?;
        let function = &mir.functions["main"];
        let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[0] else {
            panic!("expected stack storage");
        };
        let size = [16, 24, 24, 24, 24, 48, 48][case];
        assert_eq!(
            function.stack_reservations.get(slot).copied().unwrap_or(0) + 1,
            size / 8
        );
        let heap_source = source.replace("%p = alloc.stack", "%p = alloc.heap");
        let heap_ir = parse_module(&heap_source)?;
        let heap_mir = from_ir(&heap_ir, "heap").map_err(|error| error.to_string())?;
        let Instruction::Call { name, args, .. } =
            &heap_mir.functions["main"].blocks[0].instructions[0]
        else {
            panic!("expected heap allocation");
        };
        assert_eq!(name, "malloc");
        assert_eq!(args, &[Operand::Immediate(Immediate::I64(size as i64))]);
        let input = dir.join(format!("case-{case}.lamina"));
        fs::write(&input, source)?;
        for level in ["0", "1", "2", "3"] {
            let executable = dir.join(format!("case-{case}-{level}"));
            compile(&input, &executable, &["--opt-level", level])?;
            let actual = Command::new(executable).output()?;
            assert!(actual.status.success(), "case {case}, O{level}");
            assert_eq!(
                String::from_utf8(actual.stdout)?,
                "55\n55\n99\n",
                "case {case}, O{level}"
            );
        }
        let minimal = format!(
            "type @Pair = struct {{ a: i64, b: i64 }}\nfn @main() -> i64 {{\nentry:\n%p = alloc.stack {ty}\n{address}\n%sentinel = add.i64 55, 0\nstore.i64 %last, 99\nprint %sentinel\nret.i64 0\n}}\n"
        );
        fs::write(&input, minimal)?;
        for level in ["0", "1", "2", "3"] {
            let executable = dir.join(format!("minimal-{case}-{level}"));
            compile(&input, &executable, &["--opt-level", level])?;
            let actual = Command::new(executable).output()?;
            assert!(actual.status.success());
            assert_eq!(
                String::from_utf8(actual.stdout)?,
                "55\n",
                "minimal case {case}, O{level}"
            );
        }
    }
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn unsized_allocations_are_rejected() -> Result<(), Box<dyn Error>> {
    for declarations in [
        "",
        "type @Bad = struct { next: @Bad }",
        "type @Bad = struct { next: @Other }\ntype @Other = [2 x @Bad]",
    ] {
        for allocation in ["stack", "heap"] {
            let source = format!(
                "{declarations}\nfn @main() -> i64 {{\nentry:\n%p = alloc.{allocation} @Bad\nret.i64 0\n}}\n"
            );
            let rejected = match parse_module(&source) {
                Err(_) => true,
                Ok(ir) => from_ir(&ir, "unsized").is_err(),
            };
            assert!(rejected, "{source}");
        }
    }
    for ty in ["void", "[9223372036854775807 x i64]"] {
        for allocation in ["stack", "heap"] {
            let source = format!(
                "fn @main() -> i64 {{\nentry:\n%p = alloc.{allocation} {ty}\nret.i64 0\n}}\n"
            );
            let rejected = match parse_module(&source) {
                Err(_) => true,
                Ok(ir) => from_ir(&ir, "unsized").is_err(),
            };
            assert!(rejected, "{source}");
        }
    }

    Ok(())
}
