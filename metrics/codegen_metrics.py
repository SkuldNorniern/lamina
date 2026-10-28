#!/usr/bin/env python3
"""
Static codegen metrics for Lamina testcases (x86_64 only).

Compiles every testcases/*.lamina at --opt-level 0..3 with --emit-asm and
records deterministic counts from the emitted AT&T assembly. No timing.

  python3 metrics/codegen_metrics.py --regenerate   rewrite metrics/baseline-x86_64.txt
  python3 metrics/codegen_metrics.py --check        compare against it, exit 1 on any difference
  python3 metrics/codegen_metrics.py --benchmarks   print the same counts for benchmarks/*/*.lamina (not checked in)

--compiler PATH measures another lamina binary, such as one built before a change.

See metrics/README.md for metric definitions.
"""

import argparse
import concurrent.futures
import difflib
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "metrics"
BASELINE = METRICS_DIR / "baseline-x86_64.txt"
TESTCASES = ROOT / "testcases"
COMPILER = ROOT / "target" / "release" / ("lamina.exe" if os.name == "nt" else "lamina")
TARGET = "x86_64_linux"
LEVELS = (0, 1, 2, 3)

LABEL_RE = re.compile(r"^\s*([A-Za-z_.$][\w.$@]*):")
FUNC_LABEL_RE = re.compile(r"^([A-Za-z_][\w.$@]*):\s*$")
MEM_FRAME_RE = re.compile(r"(?<![\w$])-?\d*\(%(?:rbp|rsp)\)")
SUB_RSP_RE = re.compile(r"^subq\s+\$(\d+),\s*%rsp$")
BRANCH_RE = re.compile(r"^j[a-z]+$")
MOV_RE = re.compile(r"^mov[a-z0-9]*$")


def strip_comment(line):
    idx = line.find("#")
    return line if idx < 0 else line[:idx]


def parse_asm(text):
    """Return (instructions, spills, calls, branches) for one .s file."""
    insns = spills = calls = branches = 0
    for raw in text.splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        # Strip leading labels (e.g. `.L_x: .string "..."`), then skip directives.
        while True:
            lm = LABEL_RE.match(line)
            if not lm:
                break
            line = line[lm.end():].strip()
        if not line or line.startswith("."):
            continue

        insns += 1
        mnemonic, _, operands = line.partition(" ")
        operands = operands.strip()
        if mnemonic == "call":
            calls += 1
        if BRANCH_RE.match(mnemonic):
            branches += 1
        if MOV_RE.match(mnemonic) and MEM_FRAME_RE.search(operands):
            spills += 1
    return insns, spills, calls, branches


def parse_asm_per_function(text):
    """Frame size per function: the first `subq $N, %rsp` in each function body, else 0."""
    frames = {}
    current = None
    seen = set()
    for raw in text.splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        m = FUNC_LABEL_RE.match(line)
        if m:
            current = m.group(1)
            frames.setdefault(current, 0)
            seen.discard(current)
            continue
        if current is None or current in seen:
            continue
        sm = SUB_RSP_RE.match(line)
        if sm:
            frames[current] = int(sm.group(1))
            seen.add(current)
    return frames


def measure(text):
    insns, spills, calls, branches = parse_asm(text)
    return {
        "instructions": insns,
        "spills": spills,
        "calls": calls,
        "branches": branches,
        "frames": parse_asm_per_function(text),
    }


def compile_one(source, level, tmpdir, compiler):
    stem = os.path.join(tmpdir, f"{source.parent.name}_{source.stem}_O{level}")
    cmd = [
        str(compiler), str(source),
        "--emit-asm", "--target", TARGET, "--opt-level", str(level),
        "-j", "1", "-o", stem,
    ]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, cwd=str(ROOT), timeout=120)
    if proc.returncode != 0:
        return {"error": proc.returncode}
    with open(stem + ".s", encoding="utf-8", errors="replace") as f:
        return measure(f.read())


def testcase_sources():
    return {p.stem: p for p in sorted(TESTCASES.glob("*.lamina"))}


def benchmark_sources():
    return {p.parent.name: p for p in sorted((ROOT / "benchmarks").glob("*/*.lamina"))}


def collect(sources=None, compiler=COMPILER):
    compiler = Path(compiler)
    if not compiler.exists():
        print(f"error: {compiler} not found. Run: cargo build --release", file=sys.stderr)
        sys.exit(2)
    if sources is None:
        sources = testcase_sources()
    names = sorted(sources)
    results = {}
    with tempfile.TemporaryDirectory(prefix="lamina-metrics-") as tmpdir:
        jobs = [(n, lv) for n in names for lv in LEVELS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
            futures = {pool.submit(compile_one, sources[n], lv, tmpdir, compiler): (n, lv) for n, lv in jobs}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
    return names, results


def render(names, results):
    out = [
        "# Lamina static codegen metrics, x86_64_linux, --opt-level 0..3.",
        "# Generated by metrics/codegen_metrics.py --regenerate. Do not edit by hand.",
        "# insns: emitted instructions. spills: mov* with an N(%rbp)/N(%rsp) operand.",
        "# calls: call instructions. branches: j* instructions. frame: subq $N,%rsp per function.",
        "",
    ]
    for name in names:
        for lv in LEVELS:
            r = results[(name, lv)]
            out.append(f"[{name} O{lv}]")
            if "error" in r:
                out.append("status compile-failed")
            else:
                out.append(f"insns {r['instructions']}")
                out.append(f"spills {r['spills']}")
                out.append(f"calls {r['calls']}")
                out.append(f"branches {r['branches']}")
                for fn in sorted(r["frames"]):
                    out.append(f"frame {fn} {r['frames'][fn]}")
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def parse_baseline(text):
    """Parse a rendered baseline into {(testcase, level): {key: value}}."""
    data = {}
    cur = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name, lv = line[1:-1].rsplit(" O", 1)
            cur = data.setdefault((name, int(lv)), {})
            continue
        if cur is None:
            continue
        key, _, val = line.partition(" ")
        if key == "frame":
            fn, _, size = val.rpartition(" ")
            cur.setdefault("frames", {})[fn] = int(size)
        elif key in ("insns", "spills", "calls", "branches"):
            cur[key] = int(val)
        else:
            cur[key] = val
    return data


TOTAL_KEYS = ("insns", "spills", "calls", "branches")


def totals(data):
    t = {}
    for lv in LEVELS:
        t[lv] = {k: 0 for k in TOTAL_KEYS}
        for (name, l), rec in data.items():
            if l != lv:
                continue
            for k in TOTAL_KEYS:
                t[lv][k] += rec.get(k, 0)
    return t


def print_totals(label, data):
    t = totals(data)
    for lv in LEVELS:
        print(f"{label} O{lv}: " + ", ".join(f"{k} {t[lv][k]}" for k in TOTAL_KEYS))


def describe_delta(old, new):
    o_t, n_t = totals(old), totals(new)
    lines = []
    for lv in LEVELS:
        for k in TOTAL_KEYS:
            a, b = o_t[lv][k], n_t[lv][k]
            if a != b:
                lines.append(f"  total {k} O{lv}: {a} -> {b}, {b - a:+d}")
    return lines


def per_testcase_diff(old_text, new_text, old, new):
    """Readable diff, grouped by testcase and level."""
    lines = []
    keys = sorted(set(old) | set(new), key=lambda k: (k[0], k[1]))
    for key in keys:
        o, n = old.get(key), new.get(key)
        if o == n:
            continue
        name, lv = key
        lines.append(f"--- {name} O{lv}")
        if o is None or n is None:
            lines.append("  (testcase or level added/removed)")
            continue
        for k in TOTAL_KEYS:
            if o.get(k) != n.get(k):
                lines.append(f"  {k}: {o.get(k)} -> {n.get(k)}, {n.get(k, 0) - o.get(k, 0):+d}")
        of, nf = o.get("frames", {}), n.get("frames", {})
        for fn in sorted(set(of) | set(nf)):
            if of.get(fn) != nf.get(fn):
                lines.append(f"  frame {fn}: {of.get(fn, 'absent')} -> {nf.get(fn, 'absent')}")
        if o.get("status") != n.get("status"):
            lines.append(f"  status: {o.get('status')} -> {n.get('status')}")
    if not lines:
        # Only formatting or comment lines differ.
        lines.append("  (text differs but no metric changed)")
    return lines


REGEN_HINT = (
    "To accept an intended change, regenerate and commit the baseline:\n"
    "    cargo build --release\n"
    "    python3 metrics/codegen_metrics.py --regenerate\n"
    "    git add metrics/baseline-x86_64.txt\n"
    "Then read the diff in the PR. A smaller instruction count is not automatically better."
)


def cmd_regenerate():
    names, results = collect()
    text = render(names, results)
    BASELINE.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {BASELINE.relative_to(ROOT)} ({len(names)} testcases x {len(LEVELS)} levels)")
    print_totals("total", parse_baseline(text))
    return 0


def cmd_check():
    if not BASELINE.exists():
        print(f"error: {BASELINE.relative_to(ROOT)} is missing.\n{REGEN_HINT}")
        return 1
    names, results = collect()
    new_text = render(names, results)
    old_text = BASELINE.read_text(encoding="utf-8")
    if new_text == old_text:
        print(f"codegen metrics match {BASELINE.relative_to(ROOT)}")
        print_totals("total", parse_baseline(new_text))
        return 0

    old, new = parse_baseline(old_text), parse_baseline(new_text)
    print("Codegen metrics differ from the committed baseline.\n")
    print("Totals (baseline -> current):")
    for line in describe_delta(old, new):
        print(line)
    if not describe_delta(old, new):
        print("  (no total changed; per-function or status differences only)")
    print("\nPer-testcase changes:")
    for line in per_testcase_diff(old_text, new_text, old, new):
        print(line)
    print("\n" + REGEN_HINT)
    return 1


def cmd_benchmarks(compiler):
    names, results = collect(benchmark_sources(), compiler)
    for name in names:
        for lv in LEVELS:
            r = results[(name, lv)]
            if "error" in r:
                print(f"{name} O{lv}: compile-failed")
                continue
            print(f"{name} O{lv}: " + format_counts(r))
    print()
    sums = {lv: {k: 0 for k in MEASURE_KEYS} for lv in LEVELS}
    for (name, lv), r in results.items():
        if "error" not in r:
            for k in MEASURE_KEYS:
                sums[lv][k] += r[k]
    for lv in LEVELS:
        print(f"total O{lv}: " + format_counts(sums[lv]))
    return 0


MEASURE_KEYS = ("instructions", "spills", "calls", "branches")


def format_counts(r):
    return ", ".join(f"{k} {r[k]}" for k in MEASURE_KEYS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--regenerate", action="store_true", help="rewrite the baseline file")
    mode.add_argument("--check", action="store_true", help="compare against the baseline; exit 1 on difference")
    mode.add_argument("--benchmarks", action="store_true", help="print counts for benchmarks/*/*.lamina")
    ap.add_argument("--compiler", default=str(COMPILER), help="lamina binary to measure")
    args = ap.parse_args()
    if args.benchmarks:
        return cmd_benchmarks(args.compiler)
    if args.compiler != str(COMPILER):
        ap.error("--compiler applies to --benchmarks only; the baseline is measured with the built binary")
    return cmd_regenerate() if args.regenerate else cmd_check()


if __name__ == "__main__":
    sys.exit(main())
