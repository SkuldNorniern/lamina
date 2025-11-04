# Fibonacci Benchmark

A simple benchmark that computes Fibonacci numbers using iterative algorithms.

## Overview

This benchmark computes the nth Fibonacci number using an iterative approach with loops, demonstrating basic arithmetic operations and control flow.

## Expected Output

The current implementation demonstrates AArch64 codegen working correctly by:

1. **Function Calls**: Successfully calls `fibonacci_iterative(10)`
2. **Arithmetic Operations**: Computes fibonacci values using iterative algorithm
3. **Loops and Control Flow**: Implements proper loop structures with conditional branching
4. **Return Values**: Returns computed results correctly

**Current Status**: The implementation compiles and runs correctly, demonstrating that the AArch64 codegen handles:
- ✅ Function definitions and calls
- ✅ Iterative algorithms with loops
- ✅ Conditional branching and control flow
- ✅ Arithmetic operations and register handling
- ✅ Memory management and stack operations

**Known Issue**: The AArch64 codegen has a printf corruption issue in complex programs with multiple print statements. This affects the fibonacci benchmark output.

**Working Correctly**:
- ✅ Core AArch64 codegen (function calls, arithmetic, registers)
- ✅ Simple programs (see simple_test.lamina)
- ✅ Fibonacci algorithm implementation
- ✅ Assembly generation and compilation

**Limitation**:
- ❌ Printf output corruption in complex programs with multiple print statements
- This is a known issue in the AArch64 backend that needs further investigation

The fibonacci computation itself works correctly - the issue is specifically with printf output formatting in complex scenarios.

## Language Implementations

- **Lamina**: Functional IR with SSA form
- **Python**: Reference implementation
- **C**: Low-level implementation
- **JavaScript**: Node.js implementation
- **Rust**: Systems programming implementation
- **Go**: Concurrent programming implementation

## Running the Benchmark

### Lamina
```bash
cd /path/to/lamina
cargo run -- benchmarks/fibonacci/fibonacci.lamina --target aarch64_macos
./fibonacci
```

### Python
```bash
cd benchmarks/fibonacci
python3 fibonacci.py
```

### C
```bash
cd benchmarks/fibonacci
gcc fibonacci.c -o fibonacci
./fibonacci
```

### JavaScript
```bash
cd benchmarks/fibonacci
node fibonacci.js
```

### Rust
```bash
cd benchmarks/fibonacci
rustc fibonacci.rs -o fibonacci
./fibonacci
```

### Go
```bash
cd benchmarks/fibonacci
go run fibonacci.go
```

## Algorithm

The benchmark uses an iterative Fibonacci algorithm:

```python
def fibonacci_iterative(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
```

This approach:
- Handles base cases (n=0, n=1)
- Uses iteration instead of recursion for better performance
- Demonstrates loop constructs and variable updates
- Shows arithmetic operations on integers
