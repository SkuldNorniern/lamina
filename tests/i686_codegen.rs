#![allow(clippy::expect_used)]

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod common;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::common::TempDir;
    use std::{fs, process::Command};

    const ALIGNED_SLOTS: &str = "fn @main() -> i32 {\nentry:\n  %a = stackalloc.i32 align 16\n  %pad = stackalloc.i32\n  %b = stackalloc.i32 align 16\n  store.i32 %a, 5\n  store.i32 %pad, 1\n  store.i32 %b, 6\n  %ta = ptrtoint.i32 %a\n  %tb = ptrtoint.i32 %b\n  %la = and.i32 %ta, 15\n  %lb = and.i32 %tb, 15\n  %r = add.i32 %la, %lb\n  ret.i32 %r\n}\n";

    #[test]
    fn align_16_stackalloc_is_sixteen_byte_aligned_on_i686() {
        let guard = TempDir::new("lamina-i686-align");
        let input = guard.path().join("align.lamina");
        let exe = guard.path().join("align");
        fs::write(&input, ALIGNED_SLOTS).expect("write program");
        let build = Command::new(env!("CARGO_BIN_EXE_lamina"))
            .arg(&input)
            .args(["--target", "i686_linux", "-o"])
            .arg(&exe)
            .output()
            .expect("run lamina");
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&exe).output().expect("run i686 program");
        assert_eq!(run.status.code(), Some(0), "low address bits were not zero");
    }

    #[test]
    fn debug_lines_name_the_source_lines_on_i686() {
        let guard = TempDir::new("lamina-i686-debug");
        let input = guard.path().join("lines.lamina");
        let output = guard.path().join("lines.s");
        fs::write(
            &input,
            "# header\nfn @main() -> i32 {\n  entry:\n    %a = add.i32 10, 20\n    %b = mul.i32 %a, 3\n    jmp tail\n\n  tail:\n    ret.i32 %b\n}\n",
        )
        .expect("write program");
        let run = Command::new(env!("CARGO_BIN_EXE_lamina"))
            .arg(&input)
            .args(["-g", "--target", "i686_linux", "--emit-asm", "-o"])
            .arg(&output)
            .output()
            .expect("run lamina");
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let assembly = fs::read_to_string(&output).expect("read assembly");
        let rows: Vec<&str> = assembly.lines().filter(|l| l.contains(".loc 1 ")).collect();
        assert_eq!(
            rows,
            ["    .loc 1 4 5", "    .loc 1 5 5", "    .loc 1 9 5",]
        );
    }
}
