//! What gdb and lldb can answer about a `-g` build.
//!
//! The two disagree about malformed DWARF often enough that passing both is the bar.
//! Each assertion is a question a person debugging would ask, not a comparison with
//! another toolchain's output.
#![allow(clippy::expect_used)]

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod common;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::common::TempDir;
    use std::{
        env, fs,
        path::{Path, PathBuf},
        process::{self, Command},
    };

    const SOURCE: &str = r#"fn @triple(i64 %n) -> i64 {
  entry:
    %r = mul.i64 %n, 3
    ret.i64 %r
}

fn @main() -> i64 {
  entry:
    %v = call @triple(7)
    %d = call @emit(%v)
    ret.i64 0
}

fn @emit(i64 %value) -> i64 {
  entry:
    %c = add.i64 %value, 27
    %w = writebyte %c
    ret.i64 0
}
"#;

    /// Skips when the tool is missing, unless the suite was told to require it.
    /// The debugger to drive, accepting a version-suffixed spelling: Ubuntu ships
    /// `lldb-18` with no unsuffixed alias unless the `lldb` package is installed.
    fn tool(name: &str) -> Option<String> {
        let runnable = |candidate: &str| {
            Command::new(candidate)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
        };
        if runnable(name) {
            return Some(name.to_string());
        }
        for suffix in ["-18", "-19", "-17"] {
            let versioned = format!("{name}{suffix}");
            if runnable(&versioned) {
                return Some(versioned);
            }
        }
        let required = env::var_os("LAMINA_REQUIRE_TOOLS").is_some_and(|value| value == "1");
        assert!(
            !required,
            "{name} is not installed (LAMINA_REQUIRE_TOOLS=1)"
        );
        None
    }

    fn build() -> (TempDir, PathBuf) {
        let guard = TempDir::new("lamina-debug-info");
        let stem = format!("lamina-debug-info-{}", process::id());
        let input = guard.path().join(format!("{stem}.lamina"));
        let exe = guard.path().join(&stem);
        fs::write(&input, SOURCE).expect("write program");
        let run = Command::new(env!("CARGO_BIN_EXE_lamina"))
            .arg(&input)
            .arg("-g")
            .arg("--target")
            .arg("x86_64_linux")
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("run lamina");
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        (guard, exe)
    }

    fn run(tool: &str, arguments: &[&str], exe: &Path) -> String {
        let output = Command::new(tool)
            .args(arguments)
            .arg(exe)
            .output()
            .unwrap_or_else(|e| panic!("run {tool}: {e}"));
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        text
    }

    #[test]
    fn gdb_lists_the_functions_with_their_types_and_lines() {
        let Some(gdb) = tool("gdb") else { return };
        let (_guard, exe) = build();
        let text = run(&gdb, &["-batch", "-ex", "info functions"], &exe);

        assert!(text.contains("i64 main(void);"), "{text}");
        // `triple` is on line 1 and `emit` on line 14 of the source above.
        assert!(text.contains("1:\tstatic i64 triple(i64);"), "{text}");
        assert!(text.contains("14:\tstatic i64 emit(i64);"), "{text}");
    }

    #[test]
    fn gdb_names_every_frame_of_a_backtrace() {
        let Some(gdb) = tool("gdb") else { return };
        let (_guard, exe) = build();
        let text = run(
            &gdb,
            &["-batch", "-ex", "break emit", "-ex", "run", "-ex", "bt"],
            &exe,
        );

        assert!(text.contains("in emit ("), "{text}");
        assert!(text.contains("#1  "), "{text}");
        assert!(text.contains("in main ()"), "{text}");
    }

    /// `triple(7)` is 21, and `emit` returns 0 after writing 21 + 27 as a byte. gdb
    /// reads both through `DW_AT_type`, so a wrong base type shows up here.
    #[test]
    fn gdb_reads_a_return_value_through_its_declared_type() {
        let Some(gdb) = tool("gdb") else { return };
        let (_guard, exe) = build();
        let text = run(
            &gdb,
            &[
                "-batch",
                "-ex",
                "break triple",
                "-ex",
                "run",
                "-ex",
                "finish",
            ],
            &exe,
        );

        assert!(text.contains("Value returned is $1 = 21"), "{text}");
    }

    /// Nothing knows where the allocator put a parameter, so the answer must be that
    /// it is unavailable. A number here would be a guess a debugger reported as fact.
    #[test]
    fn a_parameter_reads_as_unavailable_rather_than_as_a_guess() {
        let Some(gdb) = tool("gdb") else { return };
        let (_guard, exe) = build();
        let text = run(
            &gdb,
            &[
                "-batch",
                "-ex",
                "break triple",
                "-ex",
                "run",
                "-ex",
                "info args",
            ],
            &exe,
        );

        assert!(text.contains("n = <optimized out>"), "{text}");
    }

    #[test]
    fn lldb_stops_at_a_function_and_names_the_caller() {
        let Some(lldb) = tool("lldb") else { return };
        let (_guard, exe) = build();
        let text = run(
            &lldb,
            &[
                "--batch",
                "-o",
                "br set -n emit",
                "-o",
                "run",
                "-o",
                "bt",
                "-o",
                "frame variable",
            ],
            &exe,
        );

        assert!(text.contains("frame #0:"), "{text}");
        assert!(text.contains("`emit"), "{text}");
        assert!(text.contains("`main"), "{text}");
        assert!(text.contains("value"), "{text}");
    }

    #[test]
    fn the_tree_passes_the_dwarf_verifier() {
        let Some(verifier) = ["llvm-dwarfdump", "llvm-dwarfdump-18", "llvm-dwarfdump-19"]
            .into_iter()
            .find(|name| {
                Command::new(name)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success())
            })
        else {
            return;
        };
        let (_guard, exe) = build();
        let text = run(verifier, &["--verify"], &exe);

        assert!(text.contains("No errors."), "{text}");
    }
}
