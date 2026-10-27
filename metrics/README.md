# Codegen metrics

Static, deterministic measurements of the code Lamina emits for each file in
`testcases/`, at `--opt-level 0..3`, for x86_64. No timing is measured. The
results are checked in as `baseline-x86_64.txt`, so a codegen change shows up as
a diff in the pull request.

## What is measured

Each testcase is compiled with
`lamina <file> --emit-asm --target x86_64_linux --opt-level N -j 1`. The
`--target` is pinned so the output does not depend on the host.

Per testcase and level:

- `insns`: real instructions in the `.s` output. Labels, directives (lines whose
  first token starts with `.`), comments (`#` to end of line) and blank lines are
  not counted. A label followed by an instruction on the same line counts the
  instruction.
- `spills`: `mov*` instructions with a memory operand based on `%rbp` or `%rsp`,
  written `N(%rbp)`, `-N(%rsp)`, or bare `(%rbp)`. Each instruction counts once,
  whether the frame slot is the source or destination. `lea` is not counted, since
  it computes an address without touching memory. `push`/`pop` are not counted.
- `calls`: `call` instructions.
- `branches`: `j*` instructions (conditional and unconditional jumps).
- `frame`: for each function (a non-`.L` label), the immediate of the first
  `subq $N, %rsp` in its body. A function without one has frame 0.

Totals are sums over all testcases at one level.

Only x86_64 is measured. Adding another target means a `--target` value for it,
the operand syntax for its assembly (AArch64 and RISC-V use different spill and
call forms), and a parser for them in `codegen_metrics.py`. Those parts are not
written.

## Regenerating the baseline

After an intended change to codegen:

    cargo build --release
    python3 metrics/codegen_metrics.py --regenerate
    git diff metrics/baseline-x86_64.txt

Read the diff, then commit the baseline with the change.

To check the current output without writing anything:

    python3 metrics/codegen_metrics.py --check

`--check` exits 0 when the output matches the baseline. Otherwise it exits 1 and
prints the totals (`total insns O0: 48213 -> 47990, -223`) and a per-testcase
list of what changed. CI runs `--check` in the `codegen-metrics` job.

## Read the diff, not the totals

A smaller instruction count is not automatically better. An optimisation can
remove instructions and add spills, or move work into a call, or lengthen a
dependency chain. Check the per-testcase changes, especially spills, calls and
frame sizes, before you accept a regenerated baseline.
