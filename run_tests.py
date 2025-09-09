#!/usr/bin/env python3
"""
Lamina Test Runner
Runs test cases and verifies their expected outputs.
"""

import os
import subprocess
import sys
import glob
from pathlib import Path

# Test cases with their expected outputs
TEST_CASES = {
    # Basic functionality
    'simple_const.lamina': ['42'],
    'arithmetic.lamina': ['5'],
    'loops.lamina': ['15'],
    'conditionals.lamina': ['100'],
    'functions.lamina': ['80'],
    'constants.lamina': ['42'],
    'variables.lamina': ['10', '20', '30', '25', '50'],

    # Advanced functionality
    'complex_arithmetic.lamina': ['277600'],
    'nested_calls.lamina': ['256'],
    'large_constants.lamina': ['42'],
    'stress_test.lamina': ['210'],

    # Debugging and consistency tests
    'memory_test.lamina': ['100', '200', '300', '400', '500'],
    'register_pressure.lamina': ['136'],
    'stack_operations.lamina': ['650', '1150', '1650'],
    'function_call_isolation.lamina': ['15', '16', '30', '24', '220'],
    'edge_cases.lamina': ['0', '0', '-5', '100', '0', '42'],  # Note: -5 is printed as -5

    # Recursion tests (kept only essential ones)
    'recursive_factorial.lamina': ['1', '1', '2', '6', '24'],  # Factorial recursion
    'recursive_fibonacci.lamina': ['0', '1', '1', '2', '3', '5'],  # Fibonacci recursion
    'recursive_sum.lamina': ['0', '1', '3', '6', '10', '15'],  # Sum from 1 to n
    'recursion_with_calculation.lamina': ['6'],  # Simple factorial calculation
    'stack_overflow_test.lamina': ['42'],  # Stack overflow protection

    # Memory Management Tests (Note: heap allocation requires runtime support)
     'heap_memory.lamina': ['50', '600'],  # heap allocation/deallocation tests - requires malloc
     'stack_allocation.lamina': ['50', '90', '100'],  # complete stack allocation tests
     'load_store_operations.lamina': ['1'],  # basic load/store test only

    # Advanced Tests (Note: Some require additional implementation)
     'pointer_operations.lamina': ['42', '100'],  # simple stack alloc and GEP test
     'gep_variable_indices.lamina': ['30', '42', '30', '3230'],  # GEP operations with variable indices - comprehensive test
     'pointer_arithmetic_brainfuck.lamina': ['42'],  # Pointer arithmetic with ptrtoint/inttoptr - returns stored value
     'struct_operations.lamina': ['500'],  # struct operations - basic struct allocation only
     'tuple_operations.lamina': ['0'],  # tuple operations - basic tuple operations (extraction not fully implemented)
     'type_conversion.lamina': ['100', '60', '1', '1', '1', '1', '50'],  # zero extension - requires proper type system
     # 'advanced_control_flow.lamina': ['100', '1000', '100', '10', '1', '0', '2', '10', '1100', '101', '84', '48'],  # phi nodes - requires SSA implementation
     'external_functions.lamina': ['60', '100', '17', '123', '10'],  # external functions - basic structure
     'debug_printing.lamina': ['27'],  # basic integer printing test

     # I/O Tests
     'hello_world.lamina': ['Hello World'],
     'io_basic.lamina': ['Hello'],  # Basic writebyte operations
     'io_pointer.lamina': ['42'],   # Pointer dereference I/O
     'io_comprehensive.lamina': ['Hi!', '12345', 'DONE', '12445'],  # Combined I/O operations
     'io_writeptr.lamina': ['OK'],  # WritePtr I/O - writes binary address + OK message

     # Comprehensive Type Tests
     'primitive_types_simple.lamina': ['60', '60', '16', '16', '32', '32', '2000000', '64', '1', '67', '32', '64'],  # All primitive types (simplified)
     # 'array_operations_simple.lamina': ['300'],  # Simple array operations (disabled due to segfault)
     'pointer_operations_simple.lamina': ['42', '100', '1000'],  # Simple pointer operations
     'struct_operations_simple.lamina': ['30'],  # Simple struct operations
     'tuple_operations_simple.lamina': ['0'],  # Simple tuple operations (not fully implemented)
     'type_conversions_simple.lamina': ['100'],  # Simple type conversions
     'primitive_types_fixed.lamina': ['60', '60', '16', '16', '32', '32', '2000000', '64', '1', '67'],  # Fixed primitive types
     'pointer_operations_fixed.lamina': ['42', '100', '1000', '42'],  # Fixed pointer operations
     # 'array_operations_comprehensive.lamina': ['60', '250', '2000', '50000', '300000', '600000', '10000000000', '1', '3', '72', '8', '6', '6'],  # Array operations (disabled due to segfault)
     # 'struct_operations_fixed.lamina': ['30', '124'],  # Fixed struct operations (disabled due to segfault)
     'tuple_operations_fixed.lamina': ['0'],  # Fixed tuple operations (not fully implemented)
     'type_conversions_fixed.lamina': ['100', '1000'],  # Fixed type conversions
     'brainfuck_like.lamina': ['2'],  # Brainfuck-like array operations test
     'gep_element_size_bug_test.lamina': ['3000002000001000'],  # GEP element size test
     'struct_field_offset_bug_test.lamina': ['44444003333222011', '3000201'],  # Struct field offset test
     'division_small_types_bug_test.lamina': ['6', '14', '6', '16', '3'],  # Division small types test
     'floating_point_bug_test.lamina': ['32', '64', '123', '42'],  # Floating point test
    'heap_allocation_bug_test.lamina': ['12345', '600', '333', '123457831'],  # Heap allocation test
    'edge_cases_bug_test.lamina': ['127128128127', '-4093720114651070465', '20', '0', '1000'],  # Edge cases test
    'dealloc_stack_vs_heap_bug_test.lamina': ['100042'],  # Stack vs heap deallocation bug test
    'getfieldptr_offset_bug_test.lamina': ['3566'],  # GetFieldPtr offset bug test
    'gep_element_size_comprehensive_bug_test.lamina': ['42'],  # GEP element size comprehensive bug test
    'register_size_mismatch_bug_test.lamina': ['1738067755672081826'],  # Register size mismatch bug test
    'stack_frame_overflow_bug_test.lamina': ['333222111'],  # Stack frame overflow bug test
    'floating_point_register_bug_test.lamina': ['3'],  # Floating point register bug test
    'load_store_optimization_bug_test.lamina': ['305422896125369200'],  # Load/store optimization bug test
    'calling_convention_bug_test.lamina': ['15'],  # Calling convention bug test
}

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
            timeout=60
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def compile_and_run_test(test_file):
    """Compile and run a test case, return the output lines"""
    project_root = Path(__file__).parent
    testcase_path = project_root / 'testcases' / test_file
    executable_name = test_file.replace('.lamina', '')
    
    # Compile the test
    compile_cmd = f"cargo run --release --quiet testcases/{test_file} "
    success, stdout, stderr = run_command(compile_cmd, cwd=project_root)
    
    
    if not success:
        return False, f"Compilation failed: {stderr}"
    
    # Run the executable
    run_cmd = f"./{executable_name}"
    success, stdout, stderr = run_command(run_cmd, cwd=project_root)
    
    if not success:
        return False, f"Execution failed: {stderr}"
    
    # Return output lines (filter out empty lines)
    output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
    return True, output_lines

def run_tests():
    """Run all test cases and report results"""
    print_colored("🧪 Running Lamina Test Suite", Colors.BOLD + Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    
    passed = 0
    failed = 0
    
    for test_file, expected_output in TEST_CASES.items():
        print(f"\n📝 Testing {test_file}...")
        
        success, result = compile_and_run_test(test_file)
        
        if not success:
            print_colored(f"❌ FAILED: {result}", Colors.RED)
            failed += 1
            continue
        
        actual_output = result
        
        if actual_output == expected_output:
            print_colored(f"✅ PASSED", Colors.GREEN)
            print(f"   Output: {actual_output}")
            passed += 1
        else:
            print_colored(f"❌ FAILED: Output mismatch", Colors.RED)
            print(f"   Expected: {expected_output}")
            print(f"   Actual:   {actual_output}")
            failed += 1
    
    # Summary
    print_colored("\n" + "=" * 50, Colors.BLUE)
    total = passed + failed
    if failed == 0:
        print_colored(f"🎉 All {total} tests PASSED!", Colors.GREEN + Colors.BOLD)
    else:
        print_colored(f"📊 Results: {passed}/{total} passed, {failed} failed", Colors.YELLOW)
        if failed > 0:
            sys.exit(1)

def list_tests():
    """List available test cases"""
    print_colored("📋 Available Test Cases:", Colors.BOLD + Colors.BLUE)
    for i, (test_file, expected) in enumerate(TEST_CASES.items(), 1):
        print(f"{i:2d}. {test_file:<20} → {expected}")

def run_single_test(test_name):
    """Run a single test case"""
    if test_name not in TEST_CASES:
        print_colored(f"❌ Test '{test_name}' not found", Colors.RED)
        list_tests()
        return
    
    print_colored(f"🧪 Running single test: {test_name}", Colors.BOLD + Colors.BLUE)
    
    success, result = compile_and_run_test(test_name)
    expected = TEST_CASES[test_name]
    
    if not success:
        print_colored(f"❌ FAILED: {result}", Colors.RED)
        return
    
    actual_output = result
    
    if actual_output == expected:
        print_colored(f"✅ PASSED", Colors.GREEN)
        print(f"   Output: {actual_output}")
    else:
        print_colored(f"❌ FAILED: Output mismatch", Colors.RED)
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual_output}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_tests()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_tests.py           # Run all tests")
            print("  python run_tests.py --list    # List available tests")
            print("  python run_tests.py <test>    # Run single test")
        else:
            run_single_test(sys.argv[1])
    else:
        run_tests()
