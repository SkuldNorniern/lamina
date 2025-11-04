#!/usr/bin/env python3
"""
Comprehensive recursive function testing script
Tests ALL recursive functions 10 times each to detect patterns in the bug
"""

import subprocess
import sys
from pathlib import Path
from collections import Counter

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

def test_recursive_function(test_name, expected_output, num_runs=10):
    """Test a recursive function multiple times and analyze results"""
    project_root = Path(__file__).parent
    results = []
    
    print(f"\n🔄 Testing {test_name}")
    print(f"   Expected: {expected_output}")
    print("=" * 80)
    
    for i in range(num_runs):
        # Compile and run the test
        compile_cmd = f"cargo run --quiet -- testcases/{test_name} --target aarch64_macos"
        success, stdout, stderr = run_command(compile_cmd, cwd=project_root)
        
        if not success:
            print(f"❌ Compilation failed on run {i+1}: {stderr}")
            return False, [], "COMPILATION_ERROR"
        
        # Run the executable
        executable_name = test_name.replace('.lamina', '')
        run_cmd = f"./{executable_name}"
        success, stdout, stderr = run_command(run_cmd, cwd=project_root)
        
        if not success:
            print(f"❌ Execution failed on run {i+1}: {stderr}")
            return False, [], "EXECUTION_ERROR"
        
        # Parse output
        output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
        results.append(output_lines)
        print(f"  Run {i+1:2d}: {output_lines}")
    
    # Analyze results
    print("\n📊 Analysis:")
    print("-" * 50)
    
    # Count unique results
    result_tuples = [tuple(result) for result in results]
    result_counts = Counter(result_tuples)
    
    # Check if results match expected
    actual_result = list(result_counts.most_common(1)[0][0])
    matches_expected = actual_result == expected_output
    
    if len(result_counts) == 1:
        if matches_expected:
            print(f"✅ CORRECT: All {num_runs} runs produced the expected result")
            return True, results, "CORRECT"
        else:
            print(f"❌ CONSISTENTLY WRONG: All {num_runs} runs produced wrong result")
            print(f"   Expected: {expected_output}")
            print(f"   Actual:   {actual_result}")
            return False, results, "CONSISTENTLY_WRONG"
    else:
        print(f"❌ NON-DETERMINISTIC: {len(result_counts)} different results detected!")
        print()
        for i, (result, count) in enumerate(result_counts.most_common(), 1):
            percentage = (count / num_runs) * 100
            print(f"   Result {i} (appeared {count}/{num_runs} times, {percentage:.1f}%): {list(result)}")
        
        return False, results, "NON_DETERMINISTIC"

def main():
    """Test all recursive functions with expected outputs"""
    
    # Define all recursive tests with their expected outputs
    recursive_tests = [
        # Basic recursive functions
        ('recursive_factorial.lamina', ['1', '1', '2', '6', '24']),
        ('recursive_fibonacci.lamina', ['0', '1', '1', '2', '3', '5']),
        ('recursive_sum.lamina', ['0', '1', '3', '6', '10', '15']),
        ('recursive_power.lamina', ['1', '2', '4', '8', '16', '9']),
        ('recursive_countdown.lamina', ['0', '0', '0', '0', '0', '0']),
        ('simple_recursive_debug.lamina', ['0', '1', '2', '3']),
        ('recursion_with_calculation.lamina', ['6']),
        ('recursion_depth_test.lamina', ['0', '0', '0', '0']),
        
        # Advanced recursive patterns
        ('recursive_parameter_passing.lamina', ['0', '1', '2', '3', '7']),
        ('recursive_mutual.lamina', ['1', '0', '1', '0', '0', '1']),
        ('recursive_tree_traversal.lamina', ['1', '3', '7', '15']),
        ('recursive_accumulator.lamina', ['1', '1', '2', '6', '24']),
        ('recursive_nested_calls.lamina', ['1', '1', '2', '6']),
        ('recursive_deep_call.lamina', ['42', '43', '44', '45', '47', '52']),
        ('recursive_conditional.lamina', ['0', '1', '3', '6', '10']),
        ('recursive_multiple_returns.lamina', ['0', '1', '2', '3', '5', '8']),
    ]
    
    print("🧪 COMPREHENSIVE RECURSIVE FUNCTION TESTING")
    print("=" * 80)
    print("Testing ALL recursive functions 10 times each to detect patterns in the bug")
    print(f"Total tests: {len(recursive_tests)}")
    
    results_summary = {
        'CORRECT': [],
        'CONSISTENTLY_WRONG': [],
        'NON_DETERMINISTIC': [],
        'COMPILATION_ERROR': [],
        'EXECUTION_ERROR': []
    }
    
    for test_name, expected_output in recursive_tests:
        try:
            is_correct, results, status = test_recursive_function(test_name, expected_output, 10)
            results_summary[status].append(test_name)
        except Exception as e:
            print(f"❌ Error testing {test_name}: {e}")
            results_summary['EXECUTION_ERROR'].append(test_name)
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    total_tests = len(recursive_tests)
    correct_tests = len(results_summary['CORRECT'])
    wrong_tests = len(results_summary['CONSISTENTLY_WRONG'])
    non_det_tests = len(results_summary['NON_DETERMINISTIC'])
    error_tests = len(results_summary['COMPILATION_ERROR']) + len(results_summary['EXECUTION_ERROR'])
    
    print(f"\n📊 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Correct: {correct_tests}")
    print(f"   ❌ Consistently Wrong: {wrong_tests}")
    print(f"   🔀 Non-Deterministic: {non_det_tests}")
    print(f"   💥 Errors: {error_tests}")
    
    print(f"\n✅ CORRECT TESTS ({correct_tests}):")
    for test in results_summary['CORRECT']:
        print(f"   - {test}")
    
    print(f"\n❌ CONSISTENTLY WRONG TESTS ({wrong_tests}):")
    for test in results_summary['CONSISTENTLY_WRONG']:
        print(f"   - {test}")
    
    print(f"\n🔀 NON-DETERMINISTIC TESTS ({non_det_tests}):")
    for test in results_summary['NON_DETERMINISTIC']:
        print(f"   - {test}")
    
    if results_summary['COMPILATION_ERROR']:
        print(f"\n💥 COMPILATION ERRORS ({len(results_summary['COMPILATION_ERROR'])}):")
        for test in results_summary['COMPILATION_ERROR']:
            print(f"   - {test}")
    
    if results_summary['EXECUTION_ERROR']:
        print(f"\n💥 EXECUTION ERRORS ({len(results_summary['EXECUTION_ERROR'])}):")
        for test in results_summary['EXECUTION_ERROR']:
            print(f"   - {test}")
    
    # Analysis
    print(f"\n🔍 BUG ANALYSIS:")
    if correct_tests == 0:
        print("   🚨 CRITICAL: NO recursive functions work correctly!")
        print("   📍 All recursive functions are systematically broken")
    elif wrong_tests > 0:
        print(f"   ⚠️  {wrong_tests} recursive functions are consistently wrong")
        print("   📍 Suggests systematic issue in recursive code generation")
    elif non_det_tests > 0:
        print(f"   🔀 {non_det_tests} recursive functions show non-deterministic behavior")
        print("   📍 Suggests memory corruption or race conditions")
    
    if wrong_tests > 0 or non_det_tests > 0:
        print(f"\n🎯 RECOMMENDED FOCUS:")
        print("   1. Fix stack frame management in recursive calls")
        print("   2. Fix register preservation across recursive calls")
        print("   3. Fix return value handling in recursive functions")
        print("   4. Add memory corruption detection")
        return 1
    else:
        print(f"\n🎉 All recursive functions are working correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())










