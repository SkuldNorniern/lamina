#!/usr/bin/env python3
"""
Lamina runtime benchmark runner (x86_64 Linux).

Compiles every benchmarks/*/*.lamina at --opt-level 0..3, checks each binary's
stdout against its expected_output.txt, and times the binaries.

  python3 benchmarks/run_benchmarks.py                    run all benchmarks
  python3 benchmarks/run_benchmarks.py --filter struct    run names containing "struct"
  python3 benchmarks/run_benchmarks.py --vs gcc,clang,rustc
                                                          also build and time the C/Rust ports
  python3 benchmarks/run_benchmarks.py --save FILE        record the run as JSON
  python3 benchmarks/run_benchmarks.py --compare FILE     run, then print deltas against a saved run

--compiler PATH times another lamina binary, such as one built before a change.
"""

import argparse
import fnmatch
import json
import os
import platform
import random
import shutil
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
DEFAULT_RUNS = 50
DEFAULT_WARMUP = 3
SHORT_MS = 20.0
MIN_EFFECT = 0.02
OUTLIER_K = 3.0
MIN_OUTLIER_N = 8
MAX_DROP = 0.20
BOOT_ROUNDS = 2000
BOOT_SEED = 1

# Lamina cannot read argv, so its sizes are literals in each .lamina file. The
# native ports read the same values from argv, so the compilers cannot fold the
# loops into constants. 2Dmatmul is the exception: every port hardcodes its
# sizes, and the hashed element generator keeps the k-loop from closing.
NATIVE_ARGS = {
    "2Dmatmul": [],
    "fibonacci": ["10", "20", "30", "35", "12000000"],
    "primegeneration": ["100", "1000", "10000", "2000000"],
    "int_arith": ["100000000", "12345"],
    "array_hot_loop": ["4096", "150000"],
    "struct_particles": ["200000000", "3", "5"],
    "factorial": ["10", "12", "15", "18", "50000000"],
}

# A toolchain that rewrites a loop into a closed form makes its timing measure
# arithmetic, not the benchmark. These combinations are not timed.
EXCLUDED_NATIVE = {}

NATIVE_TOOLS = {
    "gcc": {
        "ext": ".c", "levels": (2, 3), "version": ["gcc", "--version"],
        "cmd": lambda src, exe, lv: ["gcc", f"-O{lv}", str(src), "-o", str(exe)],
    },
    "clang": {
        "ext": ".c", "levels": (2, 3), "version": ["clang", "--version"],
        "cmd": lambda src, exe, lv: ["clang", f"-O{lv}", str(src), "-o", str(exe)],
    },
    "rustc": {
        "ext": ".rs", "levels": (3,), "version": ["rustc", "--version"],
        "cmd": lambda src, exe, lv: ["rustc", "-C", f"opt-level={lv}", str(src), "-o", str(exe)],
    },
    # go has no -O levels; its single column is None, keyed "go". Pure Go (CGO_ENABLED=0).
    "go": {
        "ext": ".go", "levels": (None,), "version": ["go", "version"],
        "cmd": lambda src, exe, lv: ["env", "CGO_ENABLED=0", "go", "build", "-o", str(exe), str(src)],
    },
    # zig's -O3 analogue is ReleaseFast; ReleaseSafe keeps bounds and overflow checks.
    "zig": {
        "ext": ".zig", "levels": (3,), "version": ["zig", "version"],
        "cmd": lambda src, exe, lv: ["zig", "build-exe", "-OReleaseFast", str(src), f"-femit-bin={exe}"],
    },
}


def benchmark_sources(pattern=None):
    found = {}
    for path in sorted(BENCH_DIR.glob("*/*.lamina")):
        name = path.parent.name
        if pattern and not fnmatch.fnmatch(name, f"*{pattern}*"):
            continue
        found[name] = path
    return found


def parse_vs(text):
    tools = [t.strip().lower() for t in text.split(",") if t.strip()]
    for tool in tools:
        if tool not in NATIVE_TOOLS:
            raise ValueError(f"unknown tool '{tool}' (choose from {', '.join(NATIVE_TOOLS)})")
    return tools


def native_variants(tools):
    return [(tool, lv) for tool in tools for lv in NATIVE_TOOLS[tool]["levels"]]


def native_key(tool, lv):
    return tool if lv is None else f"{tool}-O{lv}"


def native_source(name, tool):
    return BENCH_DIR / name / f"{name}{NATIVE_TOOLS[tool]['ext']}"


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


def compile_native(name, tool, lv, outdir):
    exe = outdir / f"{name}_{native_key(tool, lv)}"
    cmd = NATIVE_TOOLS[tool]["cmd"](native_source(name, tool), exe, lv)
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, cwd=str(ROOT), timeout=300,
                          text=True, errors="replace")
    if proc.returncode != 0 or not exe.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return None, tail[-1] if tail else f"exit {proc.returncode}"
    return exe, None


def stripped_size(exe, outdir):
    if shutil.which("strip") is None:
        return None
    out = outdir / f"{exe.name}.stripped"
    proc = subprocess.run(["strip", "-o", str(out), str(exe)], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    if proc.returncode != 0 or not out.exists():
        return None
    return out.stat().st_size


def run_once(exe, args=()):
    start = time.perf_counter()
    proc = subprocess.run([str(exe), *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
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


def quartiles(xs):
    if len(xs) < 2:
        return xs[0], xs[0], xs[0]
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return q1, statistics.median(xs), q3


def fence(samples):
    q1, _, q3 = quartiles(samples)
    return q1 - OUTLIER_K * (q3 - q1), q3 + OUTLIER_K * (q3 - q1)


def keep_mask(samples):
    if len(samples) < MIN_OUTLIER_N:
        return [True] * len(samples)
    lo, hi = fence(samples)
    return [lo <= s <= hi for s in samples]


def summarize(samples):
    kept = [s for s, k in zip(samples, keep_mask(samples)) if k]
    kq1, kmed, kq3 = quartiles(kept)
    return {
        "median": kmed,
        "q1": kq1,
        "q3": kq3,
        "iqr": kq3 - kq1,
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(kept) if len(kept) > 1 else 0.0,
        "n": len(samples),
        "kept": len(kept),
        "dropped": len(samples) - len(kept),
        "fence": list(fence(samples)) if len(samples) >= MIN_OUTLIER_N else None,
        "samples": samples,
        "kept_samples": kept,
    }


def measure(sources, runs, warmup, compiler, vs_tools=()):
    results = {}
    with tempfile.TemporaryDirectory(prefix="lamina-bench-") as tmpdir:
        outdir = Path(tmpdir)
        for name, source in sources.items():
            entry = {"status": "ok", "error": None, "levels": {}, "natives": {}, "sizes": {}, "stripped": {}}
            results[name] = entry
            variants = {}
            for lv in LEVELS:
                exe, err = compile_level(source, lv, outdir, compiler)
                if exe is None:
                    entry["status"] = "compile-failed"
                    entry["error"] = f"O{lv}: {err}"
                    break
                variants[f"O{lv}"] = (exe, ())
            entry["excluded"] = {tool: why for (bench, tool), why in EXCLUDED_NATIVE.items() if bench == name}
            if entry["status"] == "ok":
                for tool, lv in native_variants(vs_tools):
                    if tool in entry["excluded"]:
                        continue
                    exe, err = compile_native(name, tool, lv, outdir)
                    if exe is None:
                        entry["status"] = "compile-failed"
                        entry["error"] = f"{tool} -O{lv}: {err}"
                        break
                    variants[native_key(tool, lv)] = (exe, tuple(NATIVE_ARGS[name]))
            if entry["status"] != "ok":
                continue

            for key, (exe, _) in variants.items():
                entry["sizes"][key] = exe.stat().st_size
                entry["stripped"][key] = stripped_size(exe, outdir)

            expected = expected_lines(source)
            samples = {key: [] for key in variants}
            for _ in range(warmup):
                for key, (exe, args) in variants.items():
                    _, proc = run_once(exe, args)
                    problem = check_output(proc, expected)
                    if problem:
                        entry["status"] = "wrong-output"
                        entry["error"] = f"{key}: {problem}"
            for _ in range(runs):
                for key, (exe, args) in variants.items():
                    elapsed, proc = run_once(exe, args)
                    problem = check_output(proc, expected)
                    if problem and entry["status"] == "ok":
                        entry["status"] = "wrong-output"
                        entry["error"] = f"{key}: {problem}"
                    samples[key].append(elapsed)
            for lv in LEVELS:
                entry["levels"][str(lv)] = summarize(samples[f"O{lv}"])
            for key in variants:
                if key not in {f"O{lv}" for lv in LEVELS}:
                    entry["natives"][key] = summarize(samples[key])
            print(f"  measured {name}", file=sys.stderr)
    return results


def unreliable(stats):
    return stats["dropped"] > MAX_DROP * stats["n"]


def bootstrap_ci(values):
    rng = random.Random(BOOT_SEED)
    meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(BOOT_ROUNDS))
    return meds[int(0.025 * BOOT_ROUNDS)], meds[int(0.975 * BOOT_ROUNDS) - 1]


def compare(base, cur, paired=True):
    """Returns (cur/base change factor, bootstrap 95% CI or None, state).

    paired: per round, excluding a round when either binary was an outlier in it; used within one run,
    where every binary of a benchmark runs in the same rounds. Unpaired: cell medians, for --compare.
    state is 'real' only when the CI excludes 1.00 and the change is MIN_EFFECT or more.
    """
    if paired:
        keep_b, keep_c = keep_mask(base["samples"]), keep_mask(cur["samples"])
        ratios = [c / b for b, c, kb, kc in zip(base["samples"], cur["samples"], keep_b, keep_c) if kb and kc]
        factor = statistics.median(ratios) if ratios else 1.0
        if not ratios or unreliable(base) or unreliable(cur):
            return factor, None, "unreliable"
        lo, hi = bootstrap_ci(ratios)
    else:
        factor = cur["median"] / base["median"]
        if unreliable(base) or unreliable(cur):
            return factor, None, "unreliable"
        lo, hi = bootstrap_ci_cells(base, cur)
    if (lo > 1 or hi < 1) and abs(factor - 1) >= MIN_EFFECT:
        return factor, (lo, hi), "real"
    return factor, (lo, hi), "noise"


def bootstrap_ci_cells(base, cur):
    rng = random.Random(BOOT_SEED)
    b, c = base["kept_samples"], cur["kept_samples"]
    ratios = sorted(
        statistics.median(rng.choices(c, k=len(c))) / statistics.median(rng.choices(b, k=len(b)))
        for _ in range(BOOT_ROUNDS)
    )
    return ratios[int(0.025 * BOOT_ROUNDS)], ratios[int(0.975 * BOOT_ROUNDS) - 1]


def speedup_cell(base, cur):
    factor, _, state = compare(base, cur)
    if state != "real":
        return state
    speed = 1 / factor
    return f"{speed:.2f}x" if speed > 1 else f"{speed:.2f}x SLOWER"


def time_cell(stats):
    drop = f" -{stats['dropped']}" if stats["dropped"] else ""
    flag = "!" if unreliable(stats) else ""
    return f"{stats['median']:8.2f} [{stats['q1']:.2f}-{stats['q3']:.2f}]{drop}{flag}"


def short_cell(stats):
    drop = f" -{stats['dropped']}" if stats["dropped"] else ""
    return f"{stats['median']:.2f}{drop}{'!' if unreliable(stats) else ''}"


def print_outlier_summary(results):
    cells = []
    for name, entry in results.items():
        if entry["status"] != "ok":
            continue
        for lv, stats in entry["levels"].items():
            cells.append((f"{name} O{lv}", stats))
        for key, stats in entry["natives"].items():
            cells.append((f"{name} {key}", stats))
    if not cells:
        return
    dropped = sum(s["dropped"] for _, s in cells)
    total = sum(s["n"] for _, s in cells)
    print(f"outliers: dropped {dropped} of {total} timed samples ({dropped / total:.1%}) across {len(cells)} cells")
    for label, s in cells:
        if unreliable(s):
            print(f"unreliable: {label} dropped {s['dropped']} of {s['n']} samples ({s['dropped'] / s['n']:.0%}); no verdict")


def tool_version(tool):
    cmd = NATIVE_TOOLS[tool]["version"]
    if shutil.which(cmd[0]) is None:
        return "not found"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = proc.stdout.strip().splitlines()
    return lines[0] if lines else "unknown"


def print_header(compiler, runs, warmup, vs_tools):
    print("Lamina runtime benchmarks")
    print(f"date:      {datetime.now().isoformat(timespec='seconds')}")
    print(f"host:      {platform.platform()}, {os.cpu_count()} logical CPUs, python {platform.python_version()}")
    print(f"compiler:  {compiler} (target {TARGET})")
    for tool in vs_tools:
        print(f"native:    {tool_version(tool)}")
    print("methodology:")
    print("  - each opt level is a separate binary built with --opt-level N")
    print(f"  - {warmup} warmup run(s) per binary, discarded; then {runs} timed runs per binary")
    print("  - all binaries of a benchmark alternate within each round, so drift hits all alike")
    print("  - time is wall clock around the child process, so it includes exec and startup (~1-2 ms)")
    print(f"  - outliers: Tukey far-out fence, a sample is dropped when outside Q1-{OUTLIER_K:g}*IQR .. Q3+{OUTLIER_K:g}*IQR")
    print(f"    (applied with >= {MIN_OUTLIER_N} samples); a cell with more than {MAX_DROP:.0%} dropped is '!' and unreliable")
    print("  - cell: median of kept samples [Q1-Q3] in ms; '-N' = N samples dropped")
    print("  - speedup = median over rounds of O0/Ox, paired within each round (a round is excluded when")
    print("    either binary was an outlier in it); rounds are shared by all binaries of a benchmark")
    print(f"  - significance: bootstrap 95% CI over rounds ({BOOT_ROUNDS} resamples, seed {BOOT_SEED});")
    print(f"    a speedup is shown only when the CI excludes 1.00 AND the medians differ by >= {MIN_EFFECT:.0%};")
    print("    anything else is 'noise', not a speedup or a slowdown; unreliable cells get no verdict")
    print("  - raw min/max and all samples, with kept/dropped counts, are in the --save JSON")
    print("  - stdout is checked against expected_output.txt on every run; a mismatch voids the row")
    if "go" in vs_tools:
        print("  - go has no -O levels: go build optimizes by default, so its column is 'go', not an O3")
        print("    its binary links the Go runtime and GC, so it is far larger than the C binaries")
    if "zig" in vs_tools:
        print("  - zig-O3 is zig build-exe -OReleaseFast (ReleaseSafe keeps bounds and overflow checks)")
    if vs_tools:
        print(f"  - native ports (C/Rust/Go/Zig) read their problem sizes from argv: {NATIVE_ARGS}")
        print("    Lamina has no argv, so its sizes are literals in the .lamina source")
        print(f"  - ratio = median over rounds of Lamina O3 / fastest native; above 1.00 means Lamina is slower")
        print(f"    '~' marks a ratio whose bootstrap CI includes 1.00, so the gap is not clearly separated")
    print()


def print_table(results):
    header = f"{'benchmark':<22}" + "".join(f"{'O' + str(lv) + ' ms [Q1-Q3]':<34}" for lv in LEVELS)
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


def print_native_table(results, vs_tools):
    keys = [native_key(tool, lv) for tool, lv in native_variants(vs_tools)]
    cols = ["L-O2", "L-O3"] + keys
    excluded_notes = []
    width = 16
    header = f"{'benchmark':<20}" + "".join(f"{c:<{width}}" for c in cols)
    header += f"{'L-O3 / best':<14}{'best native':<14}"
    print(header)
    print("-" * len(header))
    sizes_rows = []
    for name, entry in results.items():
        if entry["status"] != "ok":
            print(f"{name:<20}{entry['status'].upper()}: {entry['error']}")
            continue
        stats = {"L-O2": entry["levels"]["2"], "L-O3": entry["levels"]["3"]}
        stats.update(entry["natives"])
        row = f"{name:<20}" + "".join(f"{short_cell(stats[c]):<{width}}" if c in stats else f"{'excluded':<{width}}" for c in cols)
        for tool, why in entry.get("excluded", {}).items():
            excluded_notes.append(f"excluded {name} {tool}: {why}")
        best_key = min((k for k in keys if k in stats), key=lambda k: stats[k]["median"])
        best = stats[best_key]
        ratio, ci, state = compare(best, stats["L-O3"])
        mark = "!" if state == "unreliable" else ("~" if ci[0] <= 1 <= ci[1] else "")
        cell = f"{ratio:.2f}x{mark}"
        row += f"{cell:<14}{best_key:<14}"
        print(row)
        sizes_rows.append((name, entry["sizes"], entry["stripped"]))
    for note in excluded_notes:
        print(note)
    print()
    print_size_table(sizes_rows, keys, vs_tools)


def human_size(n):
    if n is None:
        return "-"
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024 ** 2:.2f} MiB"


def raw_cell(n):
    return "-" if n is None else f"{n:,}"


def ratio_cell(size, base):
    if size is None or not base:
        return "-"
    return f"{size / base:.2f}x"


def print_size_table(size_rows, keys, vs_tools):
    print("binary size (bytes, as linked, not stripped; stripped = strip run on a copy)")
    print("vs L-O3 = binary size / Lamina O3 size for the same benchmark, so 1.00x is Lamina O3")
    labels = [("L-O2", "O2"), ("L-O3", "O3")] + [(k, k) for k in keys]
    head = (f"  {'binary':<10}{'bytes':>12}  {'size':>10}  {'vs L-O3':>8}"
            f"   {'stripped':>12}  {'size':>10}  {'vs L-O3':>8}")
    for name, sizes, stripped in size_rows:
        print(name)
        print(head)
        print("  " + "-" * (len(head) - 2))
        base, stripped_base = sizes.get("O3"), stripped.get("O3")
        for label, key in labels:
            raw, stripped_raw = sizes.get(key), stripped.get(key)
            print(f"  {label:<10}{raw_cell(raw):>12}  {human_size(raw):>10}  {ratio_cell(raw, base):>8}"
                  f"   {raw_cell(stripped_raw):>12}  {human_size(stripped_raw):>10}"
                  f"  {ratio_cell(stripped_raw, stripped_base):>8}")
        print()
    if "go" in vs_tools:
        print("go: the binary links the Go runtime, scheduler and GC; most of its size is that runtime, not the program")
    if "rustc" in vs_tools:
        print("rustc: the binary links std (I/O, panic and backtrace support, allocator); most of its size is that runtime")
    if "zig" in vs_tools:
        print("zig: the binary links the Zig std startup and panic runtime; most of its size is that runtime")


def print_compare(results, baseline_path):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["results"]
    print()
    print(f"comparison against {baseline_path}")
    print("  unpaired: rounds of two separate runs do not line up, so cell medians are compared")
    print(f"{'benchmark':<22}{'level':<7}{'before ms':<34}{'after ms':<34}{'change':<18}")
    print("-" * 115)
    for name, entry in results.items():
        base_entry = baseline.get(name)
        if entry["status"] != "ok" or base_entry is None or base_entry["status"] != "ok":
            print(f"{name:<22}skipped (status now {entry['status']}, before {base_entry['status'] if base_entry else 'missing'})")
            continue
        for lv in LEVELS:
            before = summarize(base_entry["levels"][str(lv)]["samples"])
            after = entry["levels"][str(lv)]
            factor, _, state = compare(before, after, paired=False)
            change = f"{(factor - 1) * 100:+.1f}%" if state == "real" else state
            print(f"{name:<22}O{lv:<6}{time_cell(before):<34}{time_cell(after):<34}{change:<18}")


def main():
    parser = argparse.ArgumentParser(description="Lamina runtime benchmark runner")
    parser.add_argument("--filter", help="only benchmarks whose name contains this text (glob ok)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"timed runs per binary (default {DEFAULT_RUNS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"discarded warmup runs (default {DEFAULT_WARMUP})")
    parser.add_argument("--compiler", type=Path, default=COMPILER, help="lamina binary to time")
    parser.add_argument("--vs", metavar="TOOLS", help="also build and time native ports: gcc,clang,rustc,go,zig")
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

    vs_tools = []
    if args.vs:
        try:
            vs_tools = parse_vs(args.vs)
        except ValueError as exc:
            parser.error(str(exc))
        missing = [t for t in vs_tools if shutil.which(t) is None]
        if missing:
            print(f"error: not on PATH: {', '.join(missing)}", file=sys.stderr)
            return 2

    sources = benchmark_sources(args.filter)
    if vs_tools:
        ported = {}
        for name, source in sources.items():
            if name not in NATIVE_ARGS:
                print(f"skipped {name}: no native args in NATIVE_ARGS", file=sys.stderr)
                continue
            missing_src = [t for t in vs_tools if not native_source(name, t).exists()]
            if missing_src:
                print(f"skipped {name}: no port for {', '.join(missing_src)}", file=sys.stderr)
                continue
            ported[name] = source
        sources = ported
    if not sources:
        print("error: no benchmarks matched", file=sys.stderr)
        return 2

    print_header(args.compiler, args.runs, args.warmup, vs_tools)
    results = measure(sources, args.runs, args.warmup, args.compiler, vs_tools)
    if vs_tools:
        print_native_table(results, vs_tools)
    else:
        print_table(results)
    print_outlier_summary(results)

    if args.save:
        record = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "host": platform.platform(),
            "compiler": str(args.compiler),
            "target": TARGET,
            "vs": vs_tools,
            "runs": args.runs,
            "warmup": args.warmup,
            "methodology": {
                "outlier": f"tukey far-out fence k={OUTLIER_K:g}",
                "min_outlier_n": MIN_OUTLIER_N,
                "max_drop": MAX_DROP,
                "min_effect": MIN_EFFECT,
                "ci": 0.95,
                "bootstrap_rounds": BOOT_ROUNDS,
                "bootstrap_seed": BOOT_SEED,
            },
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
