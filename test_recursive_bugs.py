#!/usr/bin/env python3
"""
Comprehensive recursive function testing script
Tests recursive functions 10 times each to detect non-deterministic behavior
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

def test_recursive_function(test_name, num_runs=10):
    """Test a recursive function multiple times and analyze results"""
    project_root = Path(__file__).parent
    results = []
    
    print(f"\n🔄 Testing {test_name} {num_runs} times...")
    print("=" * 60)
    
    for i in range(num_runs):
        # Compile and run the test
        compile_cmd = f"cargo run --quiet -- testcases/{test_name} --target aarch64_macos"
        success, stdout, stderr = run_command(compile_cmd, cwd=project_root)
        
        if not success:
            print(f"❌ Compilation failed on run {i+1}: {stderr}")
            return False, []
        
        # Run the executable
        executable_name = test_name.replace('.lamina', '')
        run_cmd = f"./{executable_name}"
        success, stdout, stderr = run_command(run_cmd, cwd=project_root)
        
        if not success:
            print(f"❌ Execution failed on run {i+1}: {stderr}")
            return False, []
        
        # Parse output
        output_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
        results.append(output_lines)
        print(f"  Run {i+1:2d}: {output_lines}")
    
    # Analyze results
    print("\n📊 Analysis:")
    print("-" * 40)
    
    # Count unique results
    result_tuples = [tuple(result) for result in results]
    result_counts = Counter(result_tuples)
    
    if len(result_counts) == 1:
        print(f"✅ CONSISTENT: All {num_runs} runs produced the same result")
        print(f"   Result: {list(result_counts.most_common(1)[0][0])}")
        return True, results
    else:
        print(f"❌ NON-DETERMINISTIC: {len(result_counts)} different results detected!")
        print()
        for i, (result, count) in enumerate(result_counts.most_common(), 1):
            percentage = (count / num_runs) * 100
            print(f"   Result {i} (appeared {count}/{num_runs} times, {percentage:.1f}%): {list(result)}")
        
        # Show which runs produced which results
        print("\n🔍 Run-by-run breakdown:")
        for i, result in enumerate(results, 1):
            result_idx = list(result_counts.keys()).index(tuple(result)) + 1
            print(f"   Run {i:2d}: Result {result_idx}")
        
        return False, results

def main():
    """Test all recursive functions"""
    recursive_tests = [
        'recursive_factorial.lamina',
        'recursive_fibonacci.lamina', 
        'recursive_sum.lamina',
        'recursive_power.lamina',
        'recursive_countdown.lamina',
        'recursion_with_calculation.lamina',
        'recursion_depth_test.lamina',
    ]
    
    print("🧪 COMPREHENSIVE RECURSIVE FUNCTION TESTING")
    print("=" * 60)
    print("Testing each recursive function 10 times to detect non-deterministic behavior")
    
    consistent_tests = []
    inconsistent_tests = []
    
    for test_name in recursive_tests:
        try:
            is_consistent, results = test_recursive_function(test_name, 10)
            if is_consistent:
                consistent_tests.append(test_name)
            else:
                inconsistent_tests.append(test_name)
        except Exception as e:
            print(f"❌ Error testing {test_name}: {e}")
            inconsistent_tests.append(test_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ CONSISTENT TESTS ({len(consistent_tests)}):")
    for test in consistent_tests:
        print(f"   - {test}")
    
    print(f"\n❌ NON-DETERMINISTIC TESTS ({len(inconsistent_tests)}):")
    for test in inconsistent_tests:
        print(f"   - {test}")
    
    print(f"\n📊 Overall: {len(consistent_tests)}/{len(recursive_tests)} tests are consistent")
    
    if inconsistent_tests:
        print(f"\n🚨 CRITICAL: {len(inconsistent_tests)} recursive functions have non-deterministic behavior!")
        print("   This indicates a serious bug in the recursive code generation.")
        return 1
    else:
        print(f"\n🎉 All recursive functions are working correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())










