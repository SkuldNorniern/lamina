#![allow(clippy::expect_used)]

mod common;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::common::TempDir;
    use std::{
        env, fs,
        io::Write,
        path::{Path, PathBuf},
        process::{self, Command, Stdio},
    };

    enum Input {
        Empty,
        Closed,
        Pipe(&'static [u8]),
    }

    fn build(tag: &str, source: &str) -> (TempDir, PathBuf) {
        let guard = TempDir::new("lamina-io-contract");
        let stem = format!("lamina-io-contract-{}-{tag}", process::id());
        let input = guard.path().join(format!("{stem}.lamina"));
        let exe = guard.path().join(&stem);
        fs::write(&input, source).expect("write program");
        let run = Command::new(env!("CARGO_BIN_EXE_lamina"))
            .arg(&input)
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

    fn run(exe: &Path, input: Input, stdout_closed: bool) -> (i32, Vec<u8>) {
        let mut script = format!("exec '{}'", exe.display());
        match input {
            Input::Empty => script.push_str(" </dev/null"),
            Input::Closed => script.push_str(" <&-"),
            Input::Pipe(_) => {}
        }
        if stdout_closed {
            script.push_str(" >&-");
        }
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn program");
        if let Input::Pipe(bytes) = input {
            child
                .stdin
                .take()
                .expect("piped stdin")
                .write_all(bytes)
                .expect("write stdin");
        }
        let output = child.wait_with_output().expect("wait for program");
        (output.status.code().expect("normal exit"), output.stdout)
    }

    const WRITEBYTE: &str = "fn @main() -> i64 {\nentry:\n  %r = writebyte 65\n  ret.i64 %r\n}\n";

    #[test]
    fn writebyte_success_returns_one() {
        let (_dir, exe) = build("writebyte-ok", WRITEBYTE);
        assert_eq!(run(&exe, Input::Empty, false), (1, b"A".to_vec()));
    }

    #[test]
    fn writebyte_failure_returns_minus_two() {
        let (_dir, exe) = build("writebyte-fail", WRITEBYTE);
        assert_eq!(run(&exe, Input::Empty, true), (254, Vec::new()));
    }

    const WRITE: &str = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  store.i8 %p, 79\n  %r = write %p, 3\n  ret.i64 %r\n}\n";

    #[test]
    fn write_success_returns_count() {
        let (_dir, exe) = build("write-ok", WRITE);
        let (code, stdout) = run(&exe, Input::Empty, false);
        assert_eq!(code, 3);
        assert_eq!(stdout.len(), 3);
    }

    #[test]
    fn write_zero_length_returns_zero() {
        let (_dir, exe) = build(
            "write-zero",
            "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = write %p, 0\n  ret.i64 %r\n}\n",
        );
        assert_eq!(run(&exe, Input::Empty, false), (0, Vec::new()));
    }

    #[test]
    fn write_failure_returns_minus_two() {
        let (_dir, exe) = build("write-fail", WRITE);
        assert_eq!(run(&exe, Input::Empty, true), (254, Vec::new()));
    }

    const READ: &str = "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 1\n  ret.i64 %r\n}\n";

    #[test]
    fn read_success_returns_count() {
        let (_dir, exe) = build("read-ok", READ);
        assert_eq!(run(&exe, Input::Pipe(b"x"), false), (1, Vec::new()));
    }

    #[test]
    fn read_zero_length_returns_zero_without_touching_input() {
        let (_dir, exe) = build(
            "read-zero",
            "fn @main() -> i64 {\nentry:\n  %p = stackalloc.[4 x i8]\n  %r = read %p, 0\n  ret.i64 %r\n}\n",
        );
        assert_eq!(run(&exe, Input::Closed, false), (0, Vec::new()));
    }

    #[test]
    fn read_end_of_input_returns_minus_one() {
        let (_dir, exe) = build("read-eof", READ);
        assert_eq!(run(&exe, Input::Empty, false), (255, Vec::new()));
    }

    #[test]
    fn read_failure_returns_minus_two() {
        let (_dir, exe) = build("read-fail", READ);
        assert_eq!(run(&exe, Input::Closed, false), (254, Vec::new()));
    }

    const READBYTE: &str = "fn @main() -> i64 {\nentry:\n  %r = readbyte\n  ret.i64 %r\n}\n";

    #[test]
    fn readbyte_returns_the_byte() {
        let (_dir, exe) = build("readbyte-byte", READBYTE);
        assert_eq!(run(&exe, Input::Pipe(b"A"), false), (65, Vec::new()));
    }

    #[test]
    fn readbyte_nul_is_not_end_of_input() {
        let (_dir, exe) = build("readbyte-nul", READBYTE);
        assert_eq!(run(&exe, Input::Pipe(b"\0"), false), (0, Vec::new()));
    }

    #[test]
    fn readbyte_end_of_input_returns_minus_one() {
        let (_dir, exe) = build("readbyte-eof", READBYTE);
        assert_eq!(run(&exe, Input::Empty, false), (255, Vec::new()));
    }

    #[test]
    fn readbyte_failure_returns_minus_two() {
        let (_dir, exe) = build("readbyte-fail", READBYTE);
        assert_eq!(run(&exe, Input::Closed, false), (254, Vec::new()));
    }
}
