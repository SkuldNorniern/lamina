# Critical Bug: Recursive Function Calls Not Working

## 🚨 **CRITICAL ISSUE IDENTIFIED**

**All recursive functions in the Lamina AArch64 codegen are completely broken.** They consistently return only the base case values instead of performing the recursive calculations.

## 📊 **Test Results Summary**

### **Comprehensive Testing (10 runs each)**

| Test Case | Expected Output | Actual Output | Status |
|-----------|----------------|---------------|---------|
| `recursive_factorial.lamina` | `[1, 1, 2, 6, 24]` | `[1, 1, 1, 1, 1]` | ❌ **BROKEN** |
| `recursive_fibonacci.lamina` | `[0, 1, 1, 2, 3, 5]` | `[0, 0, 0, 0, 0, 0]` | ❌ **BROKEN** |
| `recursive_sum.lamina` | `[0, 1, 3, 6, 10, 15]` | `[0, 0, 0, 0, 0, 0]` | ❌ **BROKEN** |
| `recursive_power.lamina` | `[1, 2, 4, 8, 16, 9]` | `[1, 1, 1, 1, 1, 1]` | ❌ **BROKEN** |
| `recursive_countdown.lamina` | `[0, 0, 0, 0, 0, 0]` | `[0, 0, 0, 0, 0, 0]` | ✅ **WORKS** (only base case) |
| `recursion_depth_test.lamina` | `[0, 0, 0, 0]` | `[0, 0, 0, 0]` | ✅ **WORKS** (only base case) |
| `simple_recursive_debug.lamina` | `[0, 1, 2, 3]` | `[0, 0, 0, 0]` | ❌ **BROKEN** |

### **Special Case: Non-Deterministic Behavior**
- `recursion_with_calculation.lamina`: Alternates between `[1]` and `[6]` (70% vs 30%)
  - This is the **only** recursive function showing any calculation
  - Suggests it's partially working but with memory corruption

## 🔍 **Root Cause Analysis**

### **Assembly Code Analysis**
The generated assembly code **appears correct**:
- ✅ Recursive calls are present (`bl func_factorial`)
- ✅ Stack frame setup looks proper
- ✅ Register preservation seems correct
- ✅ Branch logic is implemented

### **Likely Issues**

#### **1. Stack Frame Corruption**
- **Problem**: Stack frames may be overwriting each other during recursion
- **Evidence**: All functions return base case values consistently
- **Impact**: Return values from recursive calls are lost

#### **2. Register Preservation Issues**
- **Problem**: Caller-saved registers not properly preserved across recursive calls
- **Evidence**: Consistent wrong results suggest systematic corruption
- **Impact**: Function arguments or return values corrupted

#### **3. Return Value Handling**
- **Problem**: Return values from recursive calls not properly stored/retrieved
- **Evidence**: Assembly shows correct `bl` calls but wrong results
- **Impact**: Recursive calculations lost

#### **4. Stack Pointer Management**
- **Problem**: Stack pointer not properly managed during recursive calls
- **Evidence**: Non-deterministic behavior in one test case
- **Impact**: Memory corruption and inconsistent results

## 🧪 **Test Cases for Debugging**

### **Minimal Reproduction Cases**

#### **1. Simple Recursive Counter**
```lamina
fn @count_up(i64 %n) -> i64 {
  entry:
    %is_zero = eq.i64 %n, 0
    br %is_zero, base_case, recursive_case

  base_case:
    ret.i64 0

  recursive_case:
    %n_minus_1 = sub.i64 %n, 1
    %result = call @count_up(%n_minus_1)
    %final = add.i64 %result, 1
    ret.i64 %final
}
```
**Expected**: `count_up(3) = 3`  
**Actual**: `count_up(3) = 0`

#### **2. Recursive Factorial**
```lamina
fn @factorial(i64 %n) -> i64 {
  entry:
    %is_zero = eq.i64 %n, 0
    br %is_zero, base_case, recursive_case

  base_case:
    ret.i64 1

  recursive_case:
    %n_minus_1 = sub.i64 %n, 1
    %factorial_n_minus_1 = call @factorial(%n_minus_1)
    %result = mul.i64 %n, %factorial_n_minus_1
    ret.i64 %result
}
```
**Expected**: `factorial(3) = 6`  
**Actual**: `factorial(3) = 1`

## 🛠️ **Debugging Strategy**

### **1. Stack Frame Analysis**
- Add debug prints to track stack pointer values
- Verify stack frame sizes are correct
- Check for stack overflow conditions

### **2. Register Preservation Verification**
- Verify all caller-saved registers are properly saved/restored
- Check for register allocation conflicts
- Ensure return values are not corrupted

### **3. Memory Corruption Detection**
- Add memory guards around stack frames
- Use debug tools to detect buffer overflows
- Verify no memory is being overwritten

### **4. Step-by-Step Debugging**
- Add debug prints in recursive functions
- Track function entry/exit and parameter values
- Verify recursive calls are actually being made

## 📁 **Files for Investigation**

### **Generated Assembly Files**
- `recursive_factorial.s` - Shows correct assembly but wrong results
- `recursion_with_calculation.s` - Only partially working case

### **Test Cases**
- `testcases/recursive_factorial.lamina` - Standard factorial
- `testcases/simple_recursive_debug.lamina` - Minimal reproduction
- `testcases/recursion_with_calculation.lamina` - Non-deterministic case

### **Testing Tools**
- `test_recursive_bugs.py` - Comprehensive recursive testing
- `test_consistency.py` - Consistency testing script

## 🎯 **Priority Fixes**

### **High Priority**
1. **Stack Frame Management**: Fix stack pointer handling in recursive calls
2. **Register Preservation**: Ensure proper save/restore of caller-saved registers
3. **Return Value Handling**: Fix return value storage and retrieval

### **Medium Priority**
4. **Memory Corruption**: Add bounds checking and memory guards
5. **Debugging Tools**: Add debug prints and stack tracing

## 🚀 **Next Steps**

1. **Isolate the bug** using the minimal reproduction cases
2. **Debug stack frame management** in recursive function calls
3. **Fix register preservation** issues
4. **Add regression tests** to prevent future issues
5. **Verify fix** using the comprehensive test suite

## 📊 **Impact Assessment**

- **Severity**: **CRITICAL** - All recursive functions are broken
- **Scope**: **WIDE** - Affects all recursive algorithms
- **Consistency**: **SYSTEMATIC** - All recursive functions fail the same way
- **Workaround**: **NONE** - Recursive functions cannot be used

This bug makes the Lamina compiler **unusable for any algorithm requiring recursion**, which includes many fundamental computer science algorithms like tree traversal, dynamic programming, and divide-and-conquer approaches.










