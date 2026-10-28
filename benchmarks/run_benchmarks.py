#!/usr/bin/env python3
"""
Lamina runtime benchmark runner (x86_64 Linux).

Compiles every benchmarks/*/*.lamina at --opt-level 0..3, checks each binary's
stdout against its expected_output.txt, and times the binaries.

  python3 benchmarks/run_benchmarks.py                    run all benchmarks
  python3 benchmarks/run_benchmarks.py --filter struct    run names containing "struct"
  python3 benchmarks/run_benchmarks.py --save FILE        record the run as JSON
  python3 benchmarks/run_benchmarks.py --compare FILE     run, then print deltas against a saved run

--compiler PATH times another lamina binary, such as one built before a change.
"""

import argparse
import fnmatch
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
COMPILER = ROOT / "target" / "release" / ("lamina.exe" if os.name == "nt" else "lamina")
TARGET = "x86_64_linux"
LEVELS = (0, 1, 2, 3)
DEFAULT_RUNS = 15
DEFAULT_WARMUP = 1
SHORT_MS = 20.0
MIN_EFFECT = 0.02


def benchmark_sources(pattern=None):
    found = {}
    for path in sorted(BENCH_DIR.glob("*/*.lamina")):
        name = path.parent.name
        if pattern and not fnmatch.fnmatch(name, f"*{pattern}*"):
            continue
        found[name] = path
    return found


def expected_lines(source):
    path = source.parent / "expected_output.txt"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def compile_level(source, level, outdir, compiler):
    exe = outdir / f"{source.parent.name}_O{level}"
    cmd = [
        str(compiler), str(source),
        "--target", TARGET, "--opt-level", str(level), "-o", str(exe),
    ]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, cwd=str(ROOT), timeout=300,
                          text=True, errors="replace")
    if proc.returncode != 0 or not exe.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return None, tail[-1] if tail else f"exit {proc.returncode}"
    return exe, None


def run_once(exe):
    start = time.perf_counter()
    proc = subprocess.run([str(exe)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, cwd=str(ROOT), timeout=600,
                          text=True, errors="replace")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc


def check_output(proc, expected):
    if proc.returncode != 0:
        return f"exit code {proc.returncode}"
    got = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if expected is not None and got != expected:
        return f"output mismatch: expected {expected}, got {got}"
    return None


def summarize(samples):
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def measure(sources, runs, warmup, compiler):
    results = {}
    with tempfile.TemporaryDirectory(prefix="lamina-bench-") as tmpdir:
        outdir = Path(tmpdir)
        for name, source in sources.items():
            entry = {"status": "ok", "error": None, "levels": {}}
            results[name] = entry
            exes = {}
            for lv in LEVELS:
                exe, err = compile_level(source, lv, outdir, compiler)
                if exe is None:
                    entry["status"] = "compile-failed"
                    entry["error"] = f"O{lv}: {err}"
                    break
                exes[lv] = exe
            if entry["status"] != "ok":
                continue

            expected = expected_lines(source)
            samples = {lv: [] for lv in LEVELS}
            for _ in range(warmup):
                for lv in LEVELS:
                    _, proc = run_once(exes[lv])
                    problem = check_output(proc, expected)
                    if problem:
                        entry["status"] = "wrong-output"
                        entry["error"] = f"O{lv}: {problem}"
            for _ in range(runs):
                for lv in LEVELS:
                    elapsed, proc = run_once(exes[lv])
                    problem = check_output(proc, expected)
                    if problem and entry["status"] == "ok":
                        entry["status"] = "wrong-output"
                        entry["error"] = f"O{lv}: {problem}"
                    samples[lv].append(elapsed)
            for lv in LEVELS:
                entry["levels"][str(lv)] = summarize(samples[lv])
            print(f"  measured {name}", file=sys.stderr)
    return results


def real_change(before, after):
    """True when the ranges are disjoint and the medians differ by MIN_EFFECT or more."""
    disjoint = after["max"] < before["min"] or after["min"] > before["max"]
    return disjoint and abs(after["median"] / before["median"] - 1) >= MIN_EFFECT


def speedup_cell(base, cur):
    if not real_change(base, cur):
        return "noise"
    ratio = base["median"] / cur["median"]
    return f"{ratio:.2f}x" if ratio > 1 else f"{ratio:.2f}x SLOWER"


def time_cell(stats):
    return f"{stats['median']:9.2f} [{stats['min']:.2f}-{stats['max']:.2f}]"


def print_header(compiler, runs, warmup):
    print("Lamina runtime benchmarks")
    print(f"date:      {datetime.now().isoformat(timespec='seconds')}")
    print(f"host:      {platform.platform()}, {os.cpu_count()} logical CPUs, python {platform.python_version()}")
    print(f"compiler:  {compiler} (target {TARGET})")
    print("methodology:")
    print(f"  - each opt level is a separate binary built with --opt-level N")
    print(f"  - {warmup} warmup run(s) per binary, discarded; then {runs} timed runs per binary")
    print(f"  - the four levels of a benchmark alternate within each round, so drift hits all alike")
    print(f"  - time is wall clock around the child process, so it includes exec and startup (~1-2 ms)")
    print(f"  - cell: median [min-max] in ms; speedup = O0 median / Ox median")
    print(f"  - a difference is shown only when the two min-max ranges are disjoint AND the medians differ by >= {MIN_EFFECT:.0%};")
    print(f"    anything else is 'noise', not a speedup or a slowdown")
    print(f"  - stdout is checked against expected_output.txt on every run; a mismatch voids the row")
    print()


def print_table(results):
    header = f"{'benchmark':<22}" + "".join(f"{'O' + str(lv) + ' ms [min-max]':<34}" for lv in LEVELS)
    header += "".join(f"{'O' + str(lv) + ' vs O0':<16}" for lv in LEVELS[1:])
    header += f"{'O3 vs O2':<16}"
    print(header)
    print("-" * len(header))
    warnings = []
    for name, entry in results.items():
        if entry["status"] != "ok":
            print(f"{name:<22}{entry['status'].upper()}: {entry['error']}")
            continue
        lv_stats = {lv: entry["levels"][str(lv)] for lv in LEVELS}
        row = f"{name:<22}" + "".join(f"{time_cell(lv_stats[lv]):<34}" for lv in LEVELS)
        row += "".join(f"{speedup_cell(lv_stats[0], lv_stats[lv]):<16}" for lv in LEVELS[1:])
        row += f"{speedup_cell(lv_stats[2], lv_stats[3]):<16}"
        print(row)
        if lv_stats[0]["median"] < SHORT_MS:
            warnings.append(f"{name}: O0 median {lv_stats[0]['median']:.2f} ms is under {SHORT_MS:g} ms; startup noise dominates")
    for w in warnings:
        print(f"warning: {w}")


def print_compare(results, baseline_path):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["results"]
    print()
    print(f"comparison against {baseline_path}")
    print(f"{'benchmark':<22}{'level':<7}{'before ms':<26}{'after ms':<26}{'change':<18}")
    print("-" * 99)
    for name, entry in results.items():
        base_entry = baseline.get(name)
        if entry["status"] != "ok" or base_entry is None or base_entry["status"] != "ok":
            print(f"{name:<22}skipped (status now {entry['status']}, before {base_entry['status'] if base_entry else 'missing'})")
            continue
        for lv in LEVELS:
            before = base_entry["levels"][str(lv)]
            after = entry["levels"][str(lv)]
            if real_change(before, after):
                change = f"{(after['median'] / before['median'] - 1) * 100:+.1f}%"
            else:
                change = "noise"
            print(f"{name:<22}O{lv:<6}{time_cell(before):<26}{time_cell(after):<26}{change:<18}")


def main():
    parser = argparse.ArgumentParser(description="Lamina runtime benchmark runner")
    parser.add_argument("--filter", help="only benchmarks whose name contains this text (glob ok)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"timed runs per binary (default {DEFAULT_RUNS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"discarded warmup runs (default {DEFAULT_WARMUP})")
    parser.add_argument("--compiler", type=Path, default=COMPILER, help="lamina binary to time")
    parser.add_argument("--save", type=Path, metavar="FILE", help="write the run as JSON")
    parser.add_argument("--compare", type=Path, metavar="BASELINE", help="print deltas against a saved run")
    args = parser.parse_args()

    if args.runs < 2:
        parser.error("--runs must be at least 2 so a spread can be reported")
    if not args.compiler.exists():
        print(f"error: {args.compiler} not found. Run: cargo build --release", file=sys.stderr)
        return 2
    if args.compare and not args.compare.exists():
        print(f"error: baseline {args.compare} not found", file=sys.stderr)
        return 2

    sources = benchmark_sources(args.filter)
    if not sources:
        print("error: no benchmarks matched", file=sys.stderr)
        return 2

    print_header(args.compiler, args.runs, args.warmup)
    results = measure(sources, args.runs, args.warmup, args.compiler)
    print_table(results)

    if args.save:
        record = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "host": platform.platform(),
            "compiler": str(args.compiler),
            "target": TARGET,
            "runs": args.runs,
            "warmup": args.warmup,
            "results": results,
        }
        args.save.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved to {args.save}")

    if args.compare:
        print_compare(results, args.compare)

    failed = [n for n, e in results.items() if e["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
