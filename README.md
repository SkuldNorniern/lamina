<div align="center">
  <img src="assets/logo.png" alt="Lamina Logo" width="240"/>
  <br/>
  <h1 style="font-size: 3.5em; margin: 0.2em 0 0 0; color: #2c3e50;">Lamina</h1>
  <p style="font-size: 1.3em; margin: 0.5em 0 1.5em 0; color: #7f8c8d; font-weight: 300;">
    <em>High-Performance Compiler Backend</em>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/language-Rust-orange.svg" alt="Language: Rust"/>
    <img src="https://img.shields.io/badge/status-Active-brightgreen.svg" alt="Status: Active"/>
  </p>
</div>

<br/>

Lamina is a compiler backend for the Lamina Intermediate Representation (IR): a small, statically-typed SSA IR.

## Why I made this

I built Lamina for a few reasons:

- **Education**: learn compiler internals by implementing the pieces end-to-end (IR, lowering, optimizations, codegen).
- **Optimization test playground**: a place to prototype and validate optimization passes without the overhead of a huge existing compiler.
- **A modern compiler ecosystem**: experiment with a clean IR/MIR split, a small toolchain, and a structure that can grow into more serious tooling over time.

## Overview

Lamina IR serves as a mid-level intermediate representation.

**Design Principles:**
- Human-readable and structurally clean
- Block-based control flow with explicit typing
- SSA (Static Single Assignment) form for all variables
- Strong type system with both primitive and composite types
- Explicit memory model distinguishing between stack and heap
- Designed for easy lowering to machine code

## Features

### Core Features
- **Type System**: Primitive and composite types
- **Memory Model**: Stack and heap allocations
- **Control Flow**: Basic blocks and branching
- **SSA Representation**: Single assignment form
- **Function System**: Typed parameters and returns

### Capabilities
- **Targets**: x86_64, AArch64, RISC-V, WebAssembly
- **MIR optimization pipeline**: O0–O3 with a transform pipeline
- **Native codegen**: emits assembly without LLVM/Cranelift
- **Tests**: an automated test suite
- **IRBuilder API**: programmatic IR construction
- **Benchmarks**: included (see below)

## Architecture & Code Generation

Lamina uses a two-stage compilation pipeline:

```
Lamina IR → MIR (Machine IR) → Optimizations → Native Assembly
```

### MIR Optimization Pipeline

Lamina includes an optimization pipeline with configurable optimization levels:

- **-O0**: No optimizations (fastest compilation)
- **-O1**: Stable, conservative optimizations (CFG simplification, jump threading)
- **-O2**: Additional optimizations (constant folding, memory optimization, addressing canonicalization)
- **-O3**: Aggressive optimizations (function inlining, strength reduction)

The pipeline includes transforms for:
- Control flow optimization (CFG simplification, jump threading)
- Constant folding and propagation
- Memory access optimization
- Function inlining
- Strength reduction
- And more (see `src/mir/transform/` for full list)

### Direct Code Generation

Lamina directly generates machine code for multiple target architectures without relying on external backends.

**Notes:**
- **Targets**: x86_64, AArch64, RISC-V, WebAssembly
- **No external backend**: no LLVM/Cranelift
- **Cross-platform**: macOS, Linux, Windows

## Current Status

### Working Features
- **Basic Arithmetic**: All arithmetic operations (add, sub, mul, div, rem, bitwise ops)
- **Control Flow**: Conditionals, loops, branching, and phi nodes for SSA
- **Function Calls**: Recursive and non-recursive function calls with proper ABI
- **Memory Operations**: Stack and heap allocations, load/store operations
- **Type System**: Primitives, arrays, structs, tuples, and pointers
- **I/O Operations**: Print statements with printf integration for all supported architectures
- **Performance**: Competitive with systems languages in benchmarks
- **Optimization Pipeline**: Configurable optimization levels with multiple transform passes


## Building Compilers with Lamina

The IRBuilder API makes it straightforward to build compilers for other languages. A typical compiler using Lamina would:

1. Parse source language into an AST
2. Use the IRBuilder to transform the AST into Lamina IR
3. Let Lamina handle optimization and code generation

The IRBuilder API provides methods for:
- Creating modules, functions, and basic blocks
- Inserting instructions with proper SSA form
- Defining and manipulating types
- Managing memory allocations
- Setting up control flow

Here's a simple example of creating a basic arithmetic function using the Lamina IRBuilder:

```rust
use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
use lamina::ir::builder::{var, i32};

// Create function: fn @add(i32 %a, i32 %b) -> i32
let mut builder = IRBuilder::new();
builder
    .function_with_params("add", vec![
        lamina::FunctionParameter {
            name: "a",
            ty: Type::Primitive(PrimitiveType::I32),
            annotations: vec![],
        },
        lamina::FunctionParameter {
            name: "b",
            ty: Type::Primitive(PrimitiveType::I32),
            annotations: vec![],
        },
    ], Type::Primitive(PrimitiveType::I32))
    // Add the two parameters
    .binary(BinaryOp::Add, "sum", PrimitiveType::I32, var("a"), var("b"))
    // Return the result
    .ret(Type::Primitive(PrimitiveType::I32), var("sum"));

let module = builder.build();

// Compile to assembly
use std::io::Write;
let mut assembly = Vec::new();
lamina::compile_lamina_ir_to_target_assembly(
    &format!("{}", module), // Convert module to IR text
    &mut assembly,
    "x86_64_linux"
)?;

println!("Generated assembly:\n{}", String::from_utf8(assembly)?);
```

This generates the equivalent Lamina IR:

```lamina
fn @add(i32 %a, i32 %b) -> i32 {
  entry:
    %result = add.i32 %a, %b
    ret.i32 %result
}
```

## Performance Benchmarks

The following results are from our 256×256 2D matrix multiplication benchmark (500 runs):

### Benchmark Results (AMD Ryzen 9 9900X)

| Language   | Time (s) | Performance Ratio | Memory (MB) | Memory Ratio |
|------------|----------|-------------------|-------------|-------------|
| Lamina     | 0.0372   | 1.00x (baseline)  | 1.38        | 1.00x       |
| Zig        | 0.0021   | 0.06x             | 0.50        | 0.36x       |
| C          | 0.0098   | 0.26x             | 1.50        | 1.09x       |
| C++        | 0.0101   | 0.27x             | 3.49        | 2.54x       |
| Go         | 0.0134   | 0.36x             | 1.60        | 1.16x       |
| Nim        | 0.0134   | 0.36x             | 1.50        | 1.09x       |
| Rust       | 0.0176   | 0.47x             | 1.91        | 1.39x       |
| C#         | 0.0333   | 0.90x             | 30.39       | 22.10x      |
| Java       | 0.0431   | 1.16x             | 42.93       | 31.22x      |
| PHP        | 0.5720   | 15.37x            | 20.50       | 14.91x      |
| Ruby       | 1.4744   | 39.63x            | 23.25       | 16.91x      |
| Python     | 2.2585   | 60.70x            | 12.38       | 9.00x       |
| JavaScript | 2.7995   | 75.24x            | 53.20       | 38.69x      |

**Notes:**
- Lower ratios indicate better performance relative to Lamina
- Statistical outliers were removed using IQR method
- Memory measurements include peak RSS during execution
- Results show median values from 500 runs for compiled languages

## Getting Started

### Prerequisites
- Rust 1.89+ (2024 edition)
- Clang/GCC for linking generated assembly
- macOS, Linux, or Windows


### Available Targets

**x86_64 (Intel/AMD 64-bit)**
- `x86_64_linux`: Linux x86_64
- `x86_64_macos`: macOS x86_64 (Intel Macs)
- `x86_64_windows`: Windows x86_64
- `x86_64_unknown`: Generic x86_64 (ELF conventions)

**AArch64 (ARM 64-bit)**
- `aarch64_macos`: macOS ARM64 (Apple Silicon)
- `aarch64_linux`: Linux ARM64
- `aarch64_windows`: Windows ARM64
- `aarch64_unknown`: Generic AArch64 (ELF conventions)

**RISC-V**
- `riscv32_unknown`: RISC-V 32-bit
- `riscv64_unknown`: RISC-V 64-bit
- `riscv128_unknown`: RISC-V 128-bit (nightly feature)

**WebAssembly**
- `wasm32_unknown`: WebAssembly 32-bit
- `wasm64_unknown`: WebAssembly 64-bit


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


## Future Directions

- **Enhanced Optimizations**: Complete optimization pipeline, loop analysis, auto-vectorization
- **Language Integration**: C and Rust frontends for compiling to Lamina IR
- **JIT Compilation**: Dynamic code execution engine
- **Debugging Tools**: Enhanced debugging information, DWARF support, interactive debugger
- **GPU Acceleration**: CUDA / Vulkan compute shader support
- **SIMD Support**: Auto-vectorization and explicit SIMD types




Please feel free to submit pull requests or open issues for bugs and feature requests.

---

<div align="center">
  <p>
    <strong>Lamina</strong> - A Modern Compiler Backend
  </p>
  <p>
    Built with Rust | Designed for Performance | Open Source
  </p>
</div>
