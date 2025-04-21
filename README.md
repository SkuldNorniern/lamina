# Lamina

Lamina is a compiler infrastructure for the Lamina Intermediate Representation (IR), a statically-typed, SSA-based language designed to bridge the gap between high-level languages and low-level backends like LLVM and Cranelift.

## Overview

Lamina IR serves as a mid-level intermediate representation with the following design principles:
- Human-readable and structurally clean
- Block-based control flow with explicit typing
- SSA (Static Single Assignment) form for all variables
- Strong type system with both primitive and composite types
- Explicit memory model distinguishing between stack and heap
- Designed for easy lowering to machine code

## Features

- **Type System**: Supports primitive types (i8, i32, i64, f32, f64, bool) and composite types (structs, arrays)
- **Memory Model**: Explicit stack and heap allocations with optional manual memory management
- **Control Flow**: Basic blocks with branching and jumping between blocks
- **SSA Representation**: Each variable is assigned exactly once, simplifying analysis and optimization
- **Function System**: Parameter passing with proper typing and return values
- **Annotations**: Optional function-level annotations for optimization hints
- **Backend Support**: Currently targets x86_64 assembly

## Project Structure

- `src/ir/`: Core IR data structures and type definitions
- `src/parser/`: Parser implementation for the Lamina IR text format
- `src/codegen/`: Code generation for target architectures
  - `x86_64/`: x86_64 assembly code generation
- `src/error/`: Error handling types and utilities
- `examples/`: Example Lamina IR programs

## Getting Started

### Prerequisites

- Rust and Cargo (latest stable version)

### Building

```bash
cargo build --release
```

### Testing

```bash
cargo test
```

### Running Examples

```bash
cargo run --example simple
```

## Lamina IR Syntax

Lamina IR files consist of type declarations, global definitions, and function declarations:

```
# Type declaration
type @Vec2 = struct { x: f32, y: f32 }

# Global value
global @message: [5 x i8] = "hello"

# Function with annotations
@export
fn @add(i32 %a, i32 %b) -> i32 {
  entry:
    %sum = add.i32 %a, %b
    ret.i32 %sum
}

# Function with control flow
fn @conditional(i32 %x) -> i32 {
  entry:
    %is_pos = gt.i32 %x, 0
    br %is_pos, positive, negative
  
  positive:
    ret.i32 1
    
  negative:
    ret.i32 -1
}
```

### Memory Operations

```
# Stack allocation
%ptr = alloc.ptr.stack i32

# Heap allocation
%hptr = alloc.ptr.heap i32

# Load and store
store.i32 %ptr, %val
%loaded = load.i32 %ptr

# Optional heap deallocation
dealloc.heap %hptr
```

### Composite Type Operations

```
# Struct field access
%field_ptr = getfield.ptr %struct, 0
%value = load.f32 %field_ptr

# Array element access
%elem_ptr = getelem.ptr %array, %index
%value = load.i32 %elem_ptr
```

## Advanced Example: Matrix Multiplication

```
# Matrix multiplication simulation
fn @compute_cell(i64 %row, i64 %col, i64 %size, i64 %matrix_value) -> i64 {
  entry:
    %row_plus_1 = add.i64 %row, 1
    %col_plus_1 = add.i64 %col, 1
    %factor1 = mul.i64 %matrix_value, %size
    %factor2 = mul.i64 %row_plus_1, %col_plus_1
    %result = mul.i64 %factor1, %factor2
    
    ret.i64 %result
}

@export
fn @main() -> i64 {
  entry:
    %matrix_size = add.i64 262144, 0
    %matrix_value = add.i64 2, 0
    
    print %matrix_size
    %_ = call @simulate_matmul(%matrix_size, %matrix_value)
    
    ret.i64 0
}
```


## Contributing

Contributions to Lamina are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.