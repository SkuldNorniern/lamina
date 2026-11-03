#!/usr/bin/env python3
"""
Lamina Test Runner
Runs test cases and verifies their expected outputs.
"""

import os
import platform
import shutil
import subprocess
import sys
import glob
import argparse
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the emoji below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Testcases that cannot run on s390x, with the reason. Any other testcase must compile,
# run and match its expected output, or have a s390x variant (see CROSS_TARGETS).
S390X_UNSUPPORTED = {
    'inline_asm_rdtsc': 'rdtsc reads the x86 timestamp counter; no s390x equivalent',
    'vector_operations': 'the z vector facility (z13 and later) is not implemented',
}


# Cross targets run through `--target <name>`: each program is compiled for the
# target, then executed under its user-mode emulator. Expected files are chosen
# by the target's byte order, not the host's. A testcase `<name>.<target>.lamina`
# is compiled in place of `<name>.lamina` for that target and shares its expected
# output; variant files are never run as testcases of their own.
CROSS_TARGETS = {
    's390x_linux': {
        'byteorder': 'big',
        'compiler': 's390x-linux-gnu-gcc',
        'emulator': 'qemu-s390x-static',
        'sysroot': '/usr/s390x-linux-gnu',
        'unsupported': S390X_UNSUPPORTED,
    },
    'aarch64_linux': {
        'byteorder': 'little',
        'compiler': 'aarch64-linux-gnu-gcc',
        'emulator': 'qemu-aarch64-static',
        'sysroot': '/usr/aarch64-linux-gnu',
        'unsupported': {},
    },
}

# A native run compiles for the host with no --target, so it takes the variant for
# the host's own target. Keyed by (platform.system(), platform.machine()).
HOST_VARIANTS = {
    ('Darwin', 'arm64'): 'aarch64_macos',
    ('Linux', 'aarch64'): 'aarch64_linux',
    ('Windows', 'AMD64'): 'x86_64_windows',
}

VARIANT_TARGETS = set(CROSS_TARGETS) | set(HOST_VARIANTS.values())

def host_variant():
    """The variant a native run on this machine selects, or None."""
    return HOST_VARIANTS.get((platform.system(), platform.machine()))

def variant_target(test_path):
    """The target that `<name>.<target>.lamina` is a variant for, or None for a base testcase."""
    stem = Path(test_path).stem
    _, dot, suffix = stem.rpartition('.')
    return suffix if dot and suffix in VARIANT_TARGETS else None

def testcase_source(test_path, target=None):
    """The file to compile for `target`: its variant when one exists, else the testcase."""
    test_path = Path(test_path)
    if target:
        variant = test_path.with_name(f"{test_path.stem}.{target}.lamina")
        if variant.exists():
            return str(variant)
    return str(test_path)

def discover(pattern):
    """Base testcases matching `pattern`; variant files are not testcases of their own."""
    return [p for p in sorted(glob.glob(pattern)) if variant_target(p) is None]

def cross_toolchain_problem(target):
    """Return why `target` cannot be built and run here, or None when it can."""
    spec = CROSS_TARGETS[target]
    missing = []
    for tool in (spec['compiler'], spec['emulator']):
        if shutil.which(tool) is None:
            missing.append(tool)
    if not (Path(spec['sysroot']) / 'lib' / 'crt1.o').is_file():
        missing.append(f"{spec['sysroot']}/lib/crt1.o")
    if missing:
        return "missing " + ", ".join(missing)
    return None


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color):
    print(f"{color}{message}{Colors.END}")

def run_command(cmd, cwd=None):
    """Run a command and return (success, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            # Tests that read stdin without declared input must see EOF rather
            # than inherit a terminal and block until the timeout.
            stdin=subprocess.DEVNULL,
            timeout=60,
            errors='replace'
        )
        return (
            result.returncode == 0,
            result.stdout.strip(),
            result.stderr.strip(),
            result.returncode,
        )
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out", None
    except Exception as e:
        return False, "", str(e), None

def describe_exit(code):
    """Human-readable exit status. Windows reports crashes as large NTSTATUS values."""
    if code is None:
        return "no exit status"
    if code < 0:
        return f"killed by signal {-code}"
    known = {
        0xC0000005: "access violation",
        0xC000001D: "illegal instruction",
        0xC00000FD: "stack overflow",
        0xC0000094: "integer divide by zero",
        0xC0000374: "heap corruption",
    }
    unsigned = code & 0xFFFFFFFF
    name = known.get(unsigned)
    detail = f" ({name})" if name else ""
    return f"exit code {code} [0x{unsigned:08X}]{detail}"

def is_crash(code):
    """True when the process died rather than chose its exit status.

    Testcases may return any value from main (stdin.lamina returns the first
    byte it read), so a plain non-zero exit is not a failure on its own.
    """
    if code is None:
        return True
    if code < 0:
        return True
    return (code & 0xFFFFFFFF) >= 0xC0000000

def record_crash(executable_path, code):
    """Append a crashed binary to target/crashes.txt so CI can debug it.

    A crash leaves no stdout and often no stderr, so the exit status alone says
    nothing about where it died. The workflow re-runs these under a debugger.
    """
    try:
        path = Path(__file__).parent / 'target' / 'crashes.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{executable_path}\t{describe_exit(code)}\n")
    except OSError:
        pass

def execution_failure(stdout, stderr, code):
    parts = [describe_exit(code)]
    if stderr:
        parts.append(f"stderr: {stderr}")
    if stdout:
        parts.append(f"stdout: {stdout}")
    return "Execution failed: " + "; ".join(parts)

def load_expected_output(test_path, byteorder=sys.byteorder):
    """Load expected output from .expected or expected_output.txt file"""
    test_path = Path(test_path)

    # A <stem>.<byteorder>.expected file takes precedence for programs of that byte order.
    expected_path = test_path.with_suffix('.expected')
    byte_order_path = test_path.with_name(f"{test_path.stem}.{byteorder}.expected")
    if byte_order_path.exists():
        expected_path = byte_order_path
    
    # If not found, try expected_output.txt (standard for benchmarks)
    if not expected_path.exists():
        expected_path = test_path.parent / "expected_output.txt"
        
    if not expected_path.exists():
        return None

    if test_path.name == 'io_types.lamina':
        with open(expected_path, 'rb') as f:
            return f.read().replace(b"\r\n", b"\n")

    with open(expected_path, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')

    expected_values = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            expected_values.append(line)

    return expected_values

def execution_argv(executable_path, target=None):
    """Command line that runs a built program on the host or under a cross emulator."""
    if target is None:
        return [executable_path]
    spec = CROSS_TARGETS[target]
    return [spec['emulator'], '-L', spec['sysroot'], executable_path]

def run_program(argv, cwd, stdin_input=None):
    """Run a program and return (success, stdout, stderr, exit_code)."""
    stdin_arg = {'input': stdin_input} if stdin_input is not None else {'stdin': subprocess.DEVNULL}
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60,
            errors='replace',
            **stdin_arg,
        )
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out", None
    except Exception as e:
        return False, "", str(e), None
    return (
        result.returncode == 0,
        result.stdout.strip(),
        result.stderr.strip(),
        result.returncode,
    )

def run_raw_output(argv, project_root):
    """Run a binary whose stdout is compared byte for byte (write sends raw memory)."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
    except Exception as e:
        return False, f"Execution failed: {str(e)}"
    if is_crash(result.returncode):
        return False, execution_failure(
            result.stdout.decode('utf-8', errors='replace'),
            result.stderr.decode('utf-8', errors='replace'),
            result.returncode,
        )
    return True, result.stdout.replace(b"\r\n", b"\n")

def compile_and_run_test(test_path, use_mir=False, target=None):
    """Compile and run a test case, return the output lines"""
    project_root = Path(__file__).parent
    test_path = Path(test_path)
    executable_name = test_path.stem
    if os.name == 'nt':
        executable_name += '.exe'
    # Built programs go under target/, which is ignored, so a run never leaves
    # binaries beside the sources or overwrites a tracked file.
    output_dir = Path(project_root) / 'target' / 'testcases'
    output_dir.mkdir(parents=True, exist_ok=True)
    executable_path = str((output_dir / executable_name).resolve())
    
    # Define stdin input for interactive tests
    stdin_inputs = {
        'stdin.lamina': 'A\nBUFFER_TEST\nB',
        'io_buffer.lamina': 'Buf',
        'readbyte_nul.lamina': 'a\x00b',
    }

    # Compile the test
    cmd_flags = "--emit-asm" if use_mir else ""
    target_flag = f"--target {target}" if target else ""
    source_path = testcase_source(test_path, target or host_variant())
    compile_cmd = (
        f'cargo run --release --quiet -- {source_path} {cmd_flags} {target_flag} -o "{executable_path}"'
    )
    success, stdout, stderr, _ = run_command(compile_cmd, cwd=project_root)

    if not success:
        return False, f"Compilation failed: {stderr}"

    argv = execution_argv(executable_path, target)

    if test_path.name == 'io_types.lamina':
        return run_raw_output(argv, project_root)

    stdin_input = stdin_inputs.get(test_path.name)
    _, stdout, stderr, exit_code = run_program(argv, project_root, stdin_input)

    if is_crash(exit_code):
        record_crash(executable_path, exit_code)
        return False, execution_failure(stdout, stderr, exit_code)

    # Return output lines (filter out empty lines and debug text)
    output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]

    return True, output_lines

def run_tests(use_mir=False, target=None, report_only=False):
    crashes = Path(__file__).parent / 'target' / 'crashes.txt'
    try:
        crashes.parent.mkdir(parents=True, exist_ok=True)
        crashes.write_text('')
    except OSError:
        pass
    """Run all test cases and report results"""
    mode = "MIR assembly" if use_mir else "default"
    if target:
        problem = cross_toolchain_problem(target)
        if problem:
            print_colored(f"⏭️  Skipping {target} run: {problem}", Colors.YELLOW)
            print_colored("   Install the cross gcc and qemu-user, or drop --target.", Colors.YELLOW)
            return
        mode = f"{mode}, target {target}"
    byteorder = CROSS_TARGETS[target]['byteorder'] if target else sys.byteorder

    print_colored(f"🧪 Running Lamina Test Suite ({mode})", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)

    passed = 0
    failed = 0
    skipped = 0
    failures = []

    # Discover tests
    test_files = discover("testcases/*.lamina")
    benchmark_files = discover("benchmarks/*/*.lamina")
    all_tests = test_files + benchmark_files

    for test_path in all_tests:
        test_name = Path(test_path).name
        print(f"\n📝 Testing {test_name}...", end="", flush=True)

        unsupported = CROSS_TARGETS[target]['unsupported'] if target else {}
        if Path(test_path).stem in unsupported:
            reason = unsupported[Path(test_path).stem]
            print_colored(f"\r📝 Testing {test_name}... ⏭️  SKIPPED (unsupported on {target}: {reason})", Colors.YELLOW)
            skipped += 1
            continue

        expected_output = load_expected_output(test_path, byteorder)
        if expected_output is None:
            print_colored(f"\n❌ FAILED {test_name}: No expected output found", Colors.RED)
            failures.append((test_name, "No expected output found"))
            failed += 1
            continue

        success, result = compile_and_run_test(test_path, use_mir, target)
        
        if not success:
            print_colored(f"\n❌ FAILED {test_name}: {result}", Colors.RED)
            failures.append((test_name, result))
            failed += 1
            continue
        
        actual_output = result
        
        if actual_output == expected_output:
            print_colored(f"\r📝 Testing {test_name}... ✅ PASSED", Colors.GREEN)
            # print(f"   Output: {actual_output}")
            passed += 1
        else:
            print_colored(f"\n❌ FAILED {test_name}: Output mismatch", Colors.RED)
            print(f"   Expected: {expected_output}")
            print(f"   Actual:   {actual_output}")
            failures.append(
                (test_name, f"Output mismatch; expected {expected_output}, got {actual_output}")
            )
            failed += 1
    
    # Summary
    print_colored("\n" + "=" * 50, Colors.BLUE)
    total = passed + failed
    if skipped:
        print_colored(f"⏭️  {skipped} testcases skipped as unsupported on {target}", Colors.YELLOW)
    if failed == 0:
        print_colored(f"🎉 All {total} tests PASSED!", Colors.GREEN + Colors.BOLD)
    else:
        print_colored(f"📊 Results: {passed}/{total} passed, {failed} failed", Colors.YELLOW)
        # Repeated at the end so CI logs that elide the middle still show them.
        print_colored("Failed tests:", Colors.RED + Colors.BOLD)
        for name, reason in failures:
            print_colored(f"  {name}: {reason}", Colors.RED)
        if failed > 0 and not report_only:
            sys.exit(1)

def list_tests():
    """List available test cases"""
    print_colored("📋 Available Test Cases:", Colors.BOLD + Colors.BLUE)
    
    test_files = discover("testcases/*.lamina")
    benchmark_files = discover("benchmarks/*/*.lamina")
    
    print_colored("\nStandard Tests:", Colors.BLUE)
    for test_path in test_files:
        print(f"  {Path(test_path).name}")
        
    print_colored("\nBenchmarks:", Colors.BLUE)
    for test_path in benchmark_files:
        print(f"  {Path(test_path).name} ({test_path})")

def run_single_test(test_name, use_mir=False, target=None):
    """Run a single test case"""
    # Try to find the test file
    if os.path.exists(test_name):
        test_path = test_name
    elif os.path.exists(f"testcases/{test_name}"):
        test_path = f"testcases/{test_name}"
    else:
        # Search in benchmarks
        matches = glob.glob(f"benchmarks/*/{test_name}")
        if matches:
            test_path = matches[0]
        else:
            print_colored(f"❌ Test '{test_name}' not found", Colors.RED)
            list_tests()
            return
    
    if target:
        problem = cross_toolchain_problem(target)
        if problem:
            print_colored(f"⏭️  Skipping {target} run: {problem}", Colors.YELLOW)
            return

    print_colored(f"🧪 Running single test: {test_path}", Colors.BOLD + Colors.BLUE)
    
    byteorder = CROSS_TARGETS[target]['byteorder'] if target else sys.byteorder
    expected_output = load_expected_output(test_path, byteorder)
    if expected_output is None:
        print_colored(f"⚠️  No expected output found for {test_path}", Colors.YELLOW)
        return

    unsupported = CROSS_TARGETS[target]['unsupported'] if target else {}
    if Path(test_path).stem in unsupported:
        print_colored(f"⏭️  SKIPPED (unsupported on {target}: {unsupported[Path(test_path).stem]})", Colors.YELLOW)
        return

    success, result = compile_and_run_test(test_path, use_mir, target)
    
    if not success:
        print_colored(f"❌ FAILED: {result}", Colors.RED)
        return
    
    actual_output = result
    
    if actual_output == expected_output:
        print_colored(f"✅ PASSED", Colors.GREEN)
        print(f"   Output: {actual_output}")
    else:
        print_colored(f"❌ FAILED: Output mismatch", Colors.RED)
        print(f"   Expected: {expected_output}")
        print(f"   Actual:   {actual_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lamina Test Runner')
    parser.add_argument('test', nargs='?', help='Specific test case to run')
    parser.add_argument('--list', action='store_true', help='List available tests')
    parser.add_argument('--mir', action='store_true', help='Use MIR codegen backend')
    parser.add_argument('--target', choices=sorted(CROSS_TARGETS),
                        help='Compile for a cross target and run under its emulator; '
                             'skipped with a message when the toolchain is absent')
    parser.add_argument('--report-only', action='store_true',
                        help='Print the counts and exit 0 even when testcases fail, '
                             'for platforms whose backend is still incomplete')
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
    elif args.test:
        run_single_test(args.test, args.mir, args.target)
    else:
        run_tests(args.mir, args.target, args.report_only)
