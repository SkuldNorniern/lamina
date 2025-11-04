# Lamina Test Cases

This directory contains test cases for the Lamina compiler, specifically designed to verify the correctness of the AArch64 codegen.

## Test Categories

### Basic Functionality
- **`simple_const.lamina`** - Simple constant printing
- **`simple_print.lamina`** - Multiple print statements
- **`simple_plus.lamina`** - Basic arithmetic with variables

### Language Features
- **`arithmetic.lamina`** - Arithmetic operations (add, sub, mul, div)
- **`variables.lamina`** - Variable operations and SSA form
- **`conditionals.lamina`** - Conditional branching (`br`, `gt`, `le`)
- **`functions.lamina`** - Function definitions and calls
- **`constants.lamina`** - Various constant sizes and types

### Advanced Features
- **`loops.lamina`** - Simple arithmetic (renamed from loops)
- **`complex_arithmetic.lamina`** - Complex mathematical operations and nested calculations
- **`nested_calls.lamina`** - Nested function calls and complex control flow
- **`multiple_functions.lamina`** - Multiple function interaction and sum calculations  
- **`large_constants.lamina`** - Large constant handling (2^30 to 2^34)
- **`stress_test.lamina`** - Stress testing with many operations and function calls

## Running Tests

### Run All Tests
```bash
# Using the test runner
python3 run_tests.py

# Using Make
make test

# Using Cargo (integration tests)
cargo test --test integration_tests
```

### Run Single Test
```bash
# Using the test runner
python3 run_tests.py arithmetic.lamina

# Using Make
make test-single TEST=arithmetic

# Using Cargo (specific integration test)
cargo test test_arithmetic
```

### List Available Tests
```bash
python3 run_tests.py --list
```

## Test Output Format

Each test case has an expected output defined in `run_tests.py`. The test runner:

1. **Compiles** the `.lamina` file to assembly
2. **Assembles and links** with clang
3. **Executes** the resulting binary
4. **Compares** actual output with expected output
5. **Reports** pass/fail status

## Expected Outputs

| Test Case | Expected Output |
|-----------|----------------|
| `simple_const.lamina` | `['42']` |
| `simple_print.lamina` | `['100', '200', '300']` |
| `simple_plus.lamina` | `['45']` |
| `arithmetic.lamina` | `['5']` |
| `loops.lamina` | `['15']` |
| `conditionals.lamina` | `['100']` |
| `functions.lamina` | `['80']` |
| `constants.lamina` | `['42', '65536', '1000000', '123456789']` |
| `variables.lamina` | `['10', '20', '30', '25', '50']` |
| `complex_arithmetic.lamina` | `['277600']` |
| `nested_calls.lamina` | `['256']` |
| `multiple_functions.lamina` | `['56']` |
| `large_constants.lamina` | `['4294967296', '1073741824', '8589934592', '17179869184']` |
| `stress_test.lamina` | `['210']` |

## Test Structure

Each test case follows this structure:

```lamina
# Comment describing the test
fn @main() -> i64 {
  entry:
    # Test logic here
    print <value>  # Expected output
    ret.i64 0
}
```

## Adding New Tests

To add a new test case:

1. **Create** a new `.lamina` file in this directory
2. **Add** the test case and expected output to `run_tests.py`
3. **Verify** the test passes: `python3 run_tests.py <test_name>.lamina`
4. **Update** this README if needed

## Notes

- Tests use **Lamina IR syntax** with SSA form
- All tests target **AArch64 macOS** by default
- Print statements use **printf with correct Apple ARM64 ABI**
- Tests verify **core codegen functionality** including:
  - Constant materialization
  - Register allocation
  - Stack management
  - Function calls
  - Conditional branching
  - Arithmetic operations
