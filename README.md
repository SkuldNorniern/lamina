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
- **Backend Support**: Directly generates x86_64 assembly without dependency on Cranelift or LLVM
- **IRBuilder API**: Programmatically construct IR modules for building compilers

## Direct Code Generation

Unlike many IR systems that rely on external backends like LLVM or Cranelift, Lamina directly generates machine code for the target architecture. This approach offers:

- Complete control over code generation and optimization
- Reduced dependencies and compilation overhead
- Customized optimizations specific to the IR semantics
- Faster compilation times for certain workloads

## Building Compilers with Lamina

Lamina provides a powerful IRBuilder API that makes it straightforward to build compilers for other languages. A typical compiler using Lamina would:

1. Parse source language into an AST
2. Use the IRBuilder to transform the AST into Lamina IR
3. Let Lamina handle optimization and code generation

The IRBuilder API provides methods for:
- Creating modules, functions, and basic blocks
- Inserting instructions with proper SSA form
- Defining and manipulating types
- Managing memory allocations
- Setting up control flow

Here's a simplified example of how to build a C compiler frontend using the Lamina IRBuilder:

```rust
fn compile_c_to_lamina(source: &str) -> Result<Module, Error> {
    // 1. Parse C code into an AST (not shown)
    let ast = parse_c_code(source)?;
    
    // 2. Create a module using the IRBuilder
    let mut builder = IRBuilder::new();
    let module = builder.create_module("c_program");
    
    // 3. Process global declarations
    for decl in ast.global_declarations() {
        process_global_declaration(&mut builder, decl)?;
    }
    
    // 4. Process function definitions
    for func in ast.function_definitions() {
        translate_c_function(&mut builder, func)?;
    }
    
    // 5. Return the completed module
    Ok(module)
}

fn translate_c_function(builder: &mut IRBuilder, func: &CFunctionDef) -> Result<(), Error> {
    // Create function signature
    let return_type = convert_c_type_to_lamina(func.return_type);
    let mut params = Vec::new();
    for param in &func.parameters {
        params.push(FunctionParameter {
            ty: convert_c_type_to_lamina(param.ty),
            name: param.name.clone(),
        });
    }
    
    // Create the function in the module
    let function = builder.create_function(&func.name, params, return_type);
    
    // Create entry block
    let entry_block = builder.create_block("entry");
    builder.set_current_block(entry_block);
    
    // Translate function body (recursively handle statements)
    translate_c_statements(builder, &func.body)?;
    
    // Add a return instruction if needed
    if !builder.current_block_terminates() {
        if return_type == Type::Void {
            builder.add_return_void();
        } else {
            let zero = builder.add_constant(return_type, 0);
            builder.add_return(zero);
        }
    }
    
    Ok(())
}
```

## Benchmarks

Lamina demonstrates competitive performance in computational tasks. Below are the results from our 2D matrix multiplication benchmark:

| Language/Framework | Time (seconds) | Performance Ratio |
|-------------------|----------------|-------------------|
| Lamina (Base)     | 0.2270         | 1.00x             |
| Zig               | 0.0032         | 0.01x             |
| C                 | 0.0141         | 0.06x             |
| C++               | 0.0146         | 0.06x             |
| Rust              | 0.0262         | 0.12x             |
| PHP               | 0.0313         | 0.14x             |
| Java              | 0.0771         | 0.34x             |
| Ruby              | 0.0909         | 0.40x             |
| Go                | 0.1430         | 0.63x             |
| C#                | 5.3603         | 23.61x            |
| Python            | 6.1892         | 27.26x            |
| JavaScript        | 7.9553         | 35.04x            |

*Note: Higher ratio indicates better performance relative to Lamina base implementation.*

## Project Structure

- `src/ir/`: Core IR data structures and type definitions
- `src/parser/`: Parser implementation for the Lamina IR text format
- `src/codegen/`: Code generation for target architectures
  - `x86_64/`: x86_64 assembly code generation
- `src/error/`: Error handling types and utilities
- `examples/`: Example Lamina IR programs
  - `arithmetic.lamina`: Basic arithmetic operations
  - `control_flow.lamina`: Conditional branching examples
  - `matmul.lamina`: 2D matrix multiplication simulation
  - `matmul3dim.lamina`: 3D matrix multiplication simulation
  - `tensor_benchmark.lamina`: Tensor operations benchmark

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
cargo run --example arithmetic
cargo run --example matmul
```

## Example: C to Lamina IR Compiler

The repository includes a proof-of-concept C to Lamina IR compiler built using the IRBuilder API. This example demonstrates how to parse C code and generate equivalent Lamina IR:

```
# examples/c_compiler.rs

// Input C code:
/*
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}

int main() {
    int result = factorial(5);
    return result;
}
*/

// Generated Lamina IR:
fn @factorial(i32 %n) -> i32 {
entry:
  %cond = le.i32 %n, 1
  br %cond, base_case, recursive_case

base_case:
  ret.i32 1

recursive_case:
  %n_minus_1 = sub.i32 %n, 1
  %rec_result = call @factorial(%n_minus_1)
  %result = mul.i32 %n, %rec_result
  ret.i32 %result
}

@export
fn @main() -> i32 {
entry:
  %result = call @factorial(5)
  ret.i32 %result
}
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

## Future Directions

- Expanding target architectures (ARM, RISC-V)
- Advanced optimizations including loop analysis and auto-vectorization
- Integration with higher-level languages including C, Rust, and custom languages
- JIT compilation support for dynamic code execution
- Enhanced debugging information and tooling
- Improved IRBuilder API for easier compiler development

## Contributing

Contributions to Lamina are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.