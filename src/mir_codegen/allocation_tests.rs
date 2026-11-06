use crate::{mir::codegen::from_ir_x86_64, parser::parse_module};
use std::error::Error;

// Builds and runs a native program with the host toolchain.
#[cfg(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "macos")
))]
mod native {
    use super::*;
    use crate::mir_codegen::test_support::PRINT_I64_HELPERS;
    use crate::test_support::TempDir;
    use crate::{mir::Instruction, mir_codegen::test_support::compile};
    use std::{fs, process::Command};

    #[test]
    fn aggregate_stack_storage_preserves_live_neighbors() -> Result<(), Box<dyn Error>> {
        let guard = TempDir::new("lamina-aggregate-storage");
        let dir = guard.path();
        let cases = [
            ("!Pair", "%last = getfieldptr.!Pair %p, 1"),
            (
                "struct { a: i8, b: i64, c: i64 }",
                "%last = getfieldptr.struct { a: i8, b: i64, c: i64 } %p, 2",
            ),
            (
                "struct { x: i8, y: i64, z: i64 }",
                "%last = getfieldptr.struct { x: i8, y: i64, z: i64 } %p, 2",
            ),
            (
                "struct { a: i8, inner: !Pair }",
                "%inner = getfieldptr.struct { a: i8, inner: !Pair } %p, 1\n%last = getfieldptr.!Pair %inner, 1",
            ),
            (
                "struct { inner: !Pair, tail: i64 }",
                "%last = getfieldptr.struct { inner: !Pair, tail: i64 } %p, 1",
            ),
            (
                "[3 x !Pair]",
                "%element = getelem.!Pair %p, 2\n%last = getfieldptr.!Pair %element, 1",
            ),
            (
                "[3 x struct { a: i64, b: i64 }]",
                "%element = getelem.struct { a: i64, b: i64 } %p, 2\n%last = getfieldptr.struct { a: i64, b: i64 } %element, 1",
            ),
        ];
        for (case, (ty, address)) in cases.iter().enumerate() {
            let source = format!(
                "type !Pair = struct {{ a: i64, b: i64 }}\nfn @main() -> i64 {{\nentry:\n%p = stackalloc.{ty}\n{address}\n%sentinel = add.i64 55, 0\n%guard = stackalloc.i64\nstore.i64 %guard, 55\nstore.i64 %last, 99\ncall @print_i64(%sentinel)\n%live = load.i64 %guard\ncall @print_i64(%live)\n%stored = load.i64 %last\ncall @print_i64(%stored)\nret.i64 0\n}}\n{PRINT_I64_HELPERS}"
            );
            let ir = parse_module(&source)?;
            let mir = from_ir_x86_64(&ir, "aggregate").map_err(|error| error.to_string())?;
            let function = &mir.functions["main"];
            let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[0].instr
            else {
                panic!("expected stack storage");
            };
            let size = [16, 24, 24, 24, 24, 48, 48][case];
            assert_eq!(
                function.stack_reservations.get(slot).copied().unwrap_or(0) + 1,
                size / 8
            );
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
                "type !Pair = struct {{ a: i64, b: i64 }}\nfn @main() -> i64 {{\nentry:\n%p = stackalloc.{ty}\n{address}\n%sentinel = add.i64 55, 0\nstore.i64 %last, 99\ncall @print_i64(%sentinel)\nret.i64 0\n}}\n{PRINT_I64_HELPERS}"
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
        Ok(())
    }
}

#[test]
fn unsized_allocations_are_rejected() -> Result<(), Box<dyn Error>> {
    for declarations in [
        "",
        "type !Bad = struct { next: !Bad }",
        "type !Bad = struct { next: !Other }\ntype !Other = [2 x !Bad]",
    ] {
        for allocation in ["stack", "heap"] {
            let source = format!(
                "{declarations}\nfn @main() -> i64 {{\nentry:\n%p = alloc.{allocation} !Bad\nret.i64 0\n}}\n"
            );
            let rejected = match parse_module(&source) {
                Err(_) => true,
                Ok(ir) => from_ir_x86_64(&ir, "unsized").is_err(),
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
                Ok(ir) => from_ir_x86_64(&ir, "unsized").is_err(),
            };
            assert!(rejected, "{source}");
        }
    }

    Ok(())
}
