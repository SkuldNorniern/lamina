#!/usr/bin/env python3
"""
Lamina Test Runner
Runs test cases and verifies their expected outputs.
"""

import os
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

def execution_failure(stdout, stderr, code):
    parts = [describe_exit(code)]
    if stderr:
        parts.append(f"stderr: {stderr}")
    if stdout:
        parts.append(f"stdout: {stdout}")
    return "Execution failed: " + "; ".join(parts)

def load_expected_output(test_path):
    """Load expected output from .expected or expected_output.txt file"""
    test_path = Path(test_path)
    
    # Try .expected first (standard for testcases)
    expected_path = test_path.with_suffix('.expected')
    
    # If not found, try expected_output.txt (standard for benchmarks)
    if not expected_path.exists():
        expected_path = test_path.parent / "expected_output.txt"
        
    if not expected_path.exists():
        return None

    with open(expected_path, 'r') as f:
        lines = f.readlines()

    expected_values = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            expected_values.append(line)
            
    return expected_values

def compile_and_run_test(test_path, use_mir=False):
    """Compile and run a test case, return the output lines"""
    project_root = Path(__file__).parent
    test_path = Path(test_path)
    executable_name = test_path.stem
    if os.name == 'nt':
        executable_name += '.exe'
    # Resolve against the project root rather than using "./name", which is not a
    # runnable form on Windows.
    executable_path = str((Path(project_root) / executable_name).resolve())
    
    # Define stdin input for interactive tests
    stdin_inputs = {
        'stdin.lamina': 'A\nBUFFER_TEST\nB',
        'io_buffer.lamina': 'Buf',
    }

    # Compile the test
    cmd_flags = "--emit-mir-asm" if use_mir else ""
    compile_cmd = f"cargo run --release --quiet {test_path} {cmd_flags}"
    success, stdout, stderr, _ = run_command(compile_cmd, cwd=project_root)

    if not success:
        return False, f"Compilation failed: {stderr}"

    stdin_input = stdin_inputs.get(test_path.name)

    if stdin_input:
        try:
            result = subprocess.run(
                [executable_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                input=stdin_input,
                timeout=60,
                errors='replace'
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            exit_code = result.returncode
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
    else:
        run_cmd = f'"{executable_path}"'
        _, stdout, stderr, exit_code = run_command(run_cmd, cwd=project_root)

    if is_crash(exit_code):
        return False, execution_failure(stdout, stderr, exit_code)

    # Return output lines (filter out empty lines and debug text)
    output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]

    # Test-specific output filtering
    if test_path.name == 'stdin.lamina' and 'Results:' in output_lines:
        results_start = output_lines.index('Results:') + 1
        output_lines = output_lines[results_start:]
    elif test_path.name == 'io_buffer.lamina' and len(output_lines) >= 3:
        output_lines = output_lines[-3:]
    elif test_path.name == 'io_types.lamina':
        filtered_lines = []
        for line in output_lines:
            printable_line = ''.join(c for c in line if ord(c) >= 32 and ord(c) <= 126)
            if printable_line.strip():
                filtered_lines.append(printable_line.strip())
        output_lines = filtered_lines

    return True, output_lines

def run_tests(use_mir=False):
    """Run all test cases and report results"""
    mode = "MIR Codegen" if use_mir else "Legacy Codegen"
    print_colored(f"🧪 Running Lamina Test Suite ({mode})", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    
    passed = 0
    failed = 0
    failures = []
    
    # Discover tests
    test_files = sorted(glob.glob("testcases/*.lamina"))
    benchmark_files = sorted(glob.glob("benchmarks/*/*.lamina"))
    all_tests = test_files + benchmark_files
    
    for test_path in all_tests:
        test_name = Path(test_path).name
        print(f"\n📝 Testing {test_name}...", end="", flush=True)
        
        expected_output = load_expected_output(test_path)
        if expected_output is None:
             print_colored(f"\n⚠️  Skipping {test_name} (No expected output found)", Colors.YELLOW)
             continue

        success, result = compile_and_run_test(test_path, use_mir)
        
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
    if failed == 0:
        print_colored(f"🎉 All {total} tests PASSED!", Colors.GREEN + Colors.BOLD)
    else:
        print_colored(f"📊 Results: {passed}/{total} passed, {failed} failed", Colors.YELLOW)
        # Repeated at the end so CI logs that elide the middle still show them.
        print_colored("Failed tests:", Colors.RED + Colors.BOLD)
        for name, reason in failures:
            print_colored(f"  {name}: {reason}", Colors.RED)
        if failed > 0:
            sys.exit(1)

def list_tests():
    """List available test cases"""
    print_colored("📋 Available Test Cases:", Colors.BOLD + Colors.BLUE)
    
    test_files = sorted(glob.glob("testcases/*.lamina"))
    benchmark_files = sorted(glob.glob("benchmarks/*/*.lamina"))
    
    print_colored("\nStandard Tests:", Colors.BLUE)
    for test_path in test_files:
        print(f"  {Path(test_path).name}")
        
    print_colored("\nBenchmarks:", Colors.BLUE)
    for test_path in benchmark_files:
        print(f"  {Path(test_path).name} ({test_path})")

def run_single_test(test_name, use_mir=False):
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
    
    print_colored(f"🧪 Running single test: {test_path}", Colors.BOLD + Colors.BLUE)
    
    expected_output = load_expected_output(test_path)
    if expected_output is None:
        print_colored(f"⚠️  No expected output found for {test_path}", Colors.YELLOW)
        return

    success, result = compile_and_run_test(test_path, use_mir)
    
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
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
    elif args.test:
        run_single_test(args.test, args.mir)
    else:
        run_tests(args.mir)
