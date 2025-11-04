# Lamina AArch64 Codegen Bug Report

## Summary

During comprehensive testing of the Lamina AArch64 codegen, we discovered **non-deterministic behavior** in recursive function calls that perform calculations. This report documents the findings and provides test cases for reproduction.

## 🐛 **Critical Bug: Non-Deterministic Recursive Function Calls**

### **Issue Description**
Recursive functions that perform calculations produce **inconsistent results** across multiple runs of the same program. The same input can produce different outputs on different executions.

### **Affected Patterns**
- ✅ **Simple recursion** (just returning values) - **WORKS CORRECTLY**
- ❌ **Recursive calculations** (factorial, power, etc.) - **NON-DETERMINISTIC**
- ✅ **Non-recursive functions** - **WORKS CORRECTLY**

### **Test Cases Demonstrating the Bug**

#### **1. Factorial Function (Non-Deterministic)**
```lamina
fn @factorial_simple(i64 %n) -> i64 {
  entry:
    %is_zero = eq.i64 %n, 0
    br %is_zero, base_case, recursive_case

  base_case:
    ret.i64 1

  recursive_case:
    %n_minus_1 = sub.i64 %n, 1
    %factorial_n_minus_1 = call @factorial_simple(%n_minus_1)
    %result = mul.i64 %n, %factorial_n_minus_1
    ret.i64 %result
}
```

**Expected**: `factorial(3) = 6`  
**Actual**: Alternates between `1` and `6` across runs

#### **2. Power Function (Non-Deterministic)**
```lamina
fn @power_of_two(i64 %exp) -> i64 {
  entry:
    %is_zero = eq.i64 %exp, 0
    br %is_zero, return_one, calculate_power

  return_one:
    ret.i64 1

  calculate_power:
    %exp_minus_1 = sub.i64 %exp, 1
    %prev_power = call @power_of_two(%exp_minus_1)
    %result = mul.i64 2, %prev_power
    ret.i64 %result
}
```

**Expected**: `power_of_two(4) = 16`  
**Actual**: Alternates between `1` and `16` across runs

### **Consistency Test Results**

Using `test_consistency.py` to run tests multiple times:

```bash
# Non-deterministic (recursive with calculations)
python3 test_consistency.py recursion_with_calculation.lamina 5
# Result: ❌ Inconsistent results detected!
# Result 1 (appeared 2 times): ['1']
# Result 2 (appeared 3 times): ['6']

# Deterministic (simple recursion)
python3 test_consistency.py recursion_depth_test.lamina 5
# Result: ✅ All 5 runs produced consistent output: ['0', '0', '0', '0']

# Deterministic (non-recursive)
python3 test_consistency.py function_call_isolation.lamina 5
# Result: ✅ All 5 runs produced consistent output: ['15', '16', '30', '24', '220']
```

## 🔍 **Root Cause Analysis**

The non-deterministic behavior suggests issues in:

1. **Stack Frame Management**: Incorrect stack pointer handling during recursive calls
2. **Register Preservation**: Caller-saved registers not properly saved/restored
3. **Return Value Handling**: Return values corrupted during recursive call chains
4. **Memory Corruption**: Uninitialized stack slots or buffer overflows

## 📊 **Comprehensive Test Suite**

### **Working Test Cases (23 total)**
- ✅ **Basic Operations**: Constants, arithmetic, variables, conditionals
- ✅ **Function Calls**: Non-recursive functions work perfectly
- ✅ **Complex Operations**: Large constants, nested calls, stress tests
- ✅ **Memory Operations**: Register pressure, stack operations, printf stress
- ✅ **Edge Cases**: Zero values, negative numbers, repeated operations

### **Known Problematic Cases**
- ❌ **Recursive calculations**: Any recursive function that performs arithmetic
- ❌ **Complex recursive patterns**: Multiple recursive functions calling each other

## 🛠️ **Debugging Tools Created**

### **1. Consistency Test Script**
```bash
python3 test_consistency.py <test_name> <num_runs>
```
Runs a test multiple times to detect non-deterministic behavior.

### **2. Comprehensive Test Runner**
```bash
python3 run_tests.py  # Run all 23 tests
python3 run_tests.py <test_name>  # Run specific test
```

### **3. Cargo Integration**
```bash
cargo test --test integration_tests  # Run integration tests
make test  # Run via Makefile
```

## 🎯 **Recommended Fixes**

### **Priority 1: Stack Frame Management**
- Ensure proper stack pointer alignment in recursive calls
- Verify stack frame size calculations
- Check for stack overflow conditions

### **Priority 2: Register Preservation**
- Review caller-saved register handling in recursive functions
- Ensure proper save/restore of registers across calls
- Check for register allocation conflicts

### **Priority 3: Return Value Handling**
- Verify return value storage and retrieval
- Check for memory corruption in return value paths
- Ensure proper cleanup of temporary values

## 📁 **Test Files for Reproduction**

### **Non-Deterministic Tests**
- `testcases/recursion_with_calculation.lamina` - Factorial function
- `testcases/multiple_functions.lamina` - Power function (removed from main suite)

### **Deterministic Tests**
- `testcases/recursion_depth_test.lamina` - Simple recursion
- `testcases/function_call_isolation.lamina` - Non-recursive functions
- `testcases/multiple_functions_simple.lamina` - Non-recursive version

### **Debugging Tools**
- `test_consistency.py` - Consistency testing script
- `run_tests.py` - Comprehensive test runner
- `tests/integration_tests.rs` - Cargo integration tests

## 🚀 **Next Steps**

1. **Isolate the bug** using the provided test cases
2. **Debug stack frame management** in recursive calls
3. **Fix register preservation** issues
4. **Add regression tests** to prevent future issues
5. **Verify fix** using the consistency test script

The comprehensive test suite now provides a solid foundation for debugging and preventing regressions in the AArch64 codegen.










