use super::test_support::compile;
use std::{
    env::temp_dir,
    error::Error,
    fs,
    process::{self, Command},
};

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn float_comparisons_match_gcc() -> Result<(), Box<dyn Error>> {
    let dir = temp_dir().join(format!("lamina-float-comparisons-{}", process::id()));
    fs::create_dir_all(&dir)?;
    let c = dir.join("oracle.c");
    fs::write(
        &c,
        r#"
#include <stdio.h>
#include <math.h>
#define CHECK(T) do { \
    volatile T values[] = {NAN, INFINITY, -INFINITY, 0.0, -0.0, 1.0, -1.0, 2.5, -2.5}; \
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) { \
        T a = values[i], b = values[j]; \
        printf("%d\n%d\n%d\n%d\n%d\n%d\n", a == b, a != b, a < b, a <= b, a > b, a >= b); \
    } \
} while (0)
int main(void) { CHECK(float); CHECK(double); return 0; }
"#,
    )?;
    let oracle = dir.join("oracle");
    let output = Command::new("gcc")
        .arg("-O2")
        .arg(&c)
        .arg("-o")
        .arg(&oracle)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = Command::new(&oracle).output()?;
    assert!(expected.status.success());
    let mut source = String::from("fn @main() -> i64 {\nentry:\n");
    for ty in ["f32", "f64"] {
        for (i, (a, b)) in [
            ("0.0", "0.0"),
            ("1.0", "0.0"),
            ("-1.0", "0.0"),
            ("0.0", "1.0"),
            ("-0.0", "1.0"),
            ("1.0", "1.0"),
            ("-1.0", "1.0"),
            ("2.5", "1.0"),
            ("-2.5", "1.0"),
        ]
        .iter()
        .enumerate()
        {
            source.push_str(&format!(" %v{ty}_{i} = div.{ty} {a}, {b}\n"));
        }
        for i in 0..9 {
            source.push_str(&format!(" jmp {ty}_{i}\n{ty}_{i}:\n"));
            for j in 0..9 {
                for op in ["eq", "ne", "lt", "le", "gt", "ge"] {
                    source.push_str(&format!(" %r{ty}_{i}_{j}_{op} = {op}.{ty} %v{ty}_{i}, %v{ty}_{j}\n print %r{ty}_{i}_{j}_{op}\n"));
                }
            }
        }
    }
    source.push_str(" ret.i64 0\n}\n");
    let input = dir.join("comparisons.lamina");
    fs::write(&input, source)?;
    for level in ["0", "1", "2", "3"] {
        let executable = dir.join(format!("comparisons-{level}"));
        compile(&input, &executable, &["--opt-level", level])?;
        let actual = Command::new(executable).output()?;
        assert!(actual.status.success());
        assert_eq!(
            String::from_utf8(actual.stdout)?,
            String::from_utf8(expected.stdout.clone())?,
            "O{level}"
        );
    }
    fs::remove_dir_all(dir)?;
    Ok(())
}
