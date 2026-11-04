<div align="center">
  <img src="assets/logo.png" alt="Lamina Logo" width="240"/>
  <br/>
  <h1 style="font-size: 3.5em; margin: 0.2em 0 0 0; color: #2c3e50;">Lamina</h1>
  <p style="font-size: 1.3em; margin: 0.5em 0 1.5em 0; color: #7f8c8d; font-weight: 300;">
    <em>A compiler backend for the Lamina IR</em>
  </p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/language-Rust-orange.svg" alt="Language: Rust"/>
    <img src="https://img.shields.io/badge/status-Active-brightgreen.svg" alt="Status: Active"/>
  </p>
</div>

<br/>

Lamina is a compiler backend for the Lamina Intermediate Representation (IR): a small, statically-typed, block-based IR.

## Why I made this

I built Lamina for a few reasons:

- **Education**: learn compiler internals by implementing the pieces end-to-end (IR, lowering, optimizations, codegen).
- **Optimization test playground**: a place to prototype and validate optimization passes without the overhead of a huge existing compiler.
- **A compiler ecosystem**: experiment with a clean IR/MIR split, a small toolchain, and a structure that can grow into more serious tooling over time.

## Overview

Lamina IR serves as a mid-level intermediate representation.

**Design Principles:**
- Human-readable and structurally clean
- Block-based control flow with explicit typing
- Named, typed values: a name may be reassigned, but not to a different type
- Strong type system with both primitive and composite types
- Explicit stack allocation; heap memory comes from the C allocator through `@extern`
- Designed for easy lowering to machine code

## Features

### Core Features
- **Type System**: Primitive and composite types
- **Memory Model**: Stack slots and raw load/store; there is no heap instruction
- **Control Flow**: Basic blocks and branching
- **Reassignable Values**: A name may be assigned again, keeping its type
- **Function System**: Typed parameters and returns

### Capabilities
- **Targets**: x86_64 Linux is the reference target. The other targets emit assembly; see the [targets page](https://lamina.sh/docs/targets) for what each has been run against
- **MIR optimization pipeline**: `--opt-level 0` to `--opt-level 3` with a transform pipeline
- **Native codegen**: emits assembly without LLVM/Cranelift
- **JIT**: `--jit` compiles and runs the program in-process; `--sandbox` requires `--jit`
- **Tests**: an automated test suite
- **IRBuilder API**: programmatic IR construction
- **Benchmarks**: programs in `benchmarks/`

## Architecture & Code Generation

Lamina uses a two-stage compilation pipeline:

```
Lamina IR → MIR (Machine IR) → Optimizations → Native Assembly
```

### MIR Optimization Pipeline

Lamina includes an optimization pipeline with configurable optimization levels. The
level is selected with `--opt-level <0|1|2|3>`. There is no `-O0` spelling; `-O0` is
rejected as an unknown argument.

- **`--opt-level 0`**: No optimizations (fastest compilation)
- **`--opt-level 1`** (default): CFG simplification, jump threading, branch optimization, dead code elimination
- **`--opt-level 2`**: adds stack slot splitting and promotion to registers, constant folding, copy propagation, memory optimization, tail-call optimization, peephole, module inlining and bounded loop unrolling
- **`--opt-level 3`**: adds loop-invariant code motion, induction and strength reduction, CSE, instruction scheduling and loop rotation

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
- **Targets**: see [Available Targets](#available-targets). `run_tests.py` runs testcases under qemu-user for s390x, AArch64, RISC-V 64 and ARMv7 when their cross toolchains are installed, and runs i686 natively
- **No external backend**: no LLVM/Cranelift
- **Hosts**: Linux is the host the compiler is developed and run on. macOS and Windows objects are written cross from Linux and are not executed here

## Current Status

### Working Features
- **Basic Arithmetic**: All arithmetic operations (add, sub, mul, div, rem, bitwise ops)
- **Control Flow**: Conditionals, loops, branching, and `copy` assignments in each predecessor for merging values
- **Function Calls**: Recursive and non-recursive function calls with proper ABI
- **Memory Operations**: Stack allocation, load and store; heap memory through a `call @malloc`
- **Type System**: Primitives, arrays, structs, and pointers
- **I/O Operations**: Raw byte I/O (`write`, `read`, `writebyte`, `readbyte`), plus libc calls such as `printf` through `@extern` declarations. The `x86_64_unknown` and `aarch64_unknown` targets reject `writebyte` at code generation
- **Optimization Pipeline**: Configurable optimization levels with multiple transform passes


## Building Compilers with Lamina

The IRBuilder API makes it straightforward to build compilers for other languages. A typical compiler using Lamina would:

1. Parse source language into an AST
2. Use the IRBuilder to transform the AST into Lamina IR
3. Let Lamina handle optimization and code generation

The IRBuilder API provides methods for:
- Creating modules, functions, and basic blocks
- Inserting instructions with correctly typed values
- Defining and manipulating types
- Managing memory allocations
- Setting up control flow

Here's a simple example of creating a basic arithmetic function using the Lamina IRBuilder:

```rust
use lamina::ir::{IRBuilder, Type, PrimitiveType, BinaryOp};
use lamina::ir::builder::var;

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
    "x86_64_linux",
    1, // codegen units
)?;

println!("Generated assembly:\n{}", String::from_utf8(assembly)?);
```

The module prints as:

```lamina
fn @add(i32 %a, i32 %b) -> i32 {
entry:
  %sum = add.i32 %a, %b
  ret.i32 %sum
}
```

## Getting Started

### Prerequisites
- The Rust version named by `rust-version` in `Cargo.toml` (1.88), on edition 2024
- Linux is the host the compiler is developed and run on
- Clang/GCC only if you ask for one: `-c gnu` or `-c clang`. The default toolchain
  assembles with `ras`, which is linked in as a library, and links with `weld`.

### Installation

```bash
cargo install lamina
```

That installs two binaries: `lamina` and `weld`, the linker lamina uses by default.
`weld` doubles as a standalone `ld` replacement.

`lamina` looks for `weld` beside itself first and only then on `PATH`. Set `LAMINA_WELD`
to a path to link with a different one. A weld older than 0.2.3 is refused with a
`[WARN]` that names both versions, and lamina links with GNU ld instead. If no weld is
found, lamina also falls back to GNU ld with a `[WARN]`.


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
- `riscv64_linux`: RISC-V 64-bit Linux (`riscv64_unknown` and `riscv64gc_linux` are accepted aliases)

**WebAssembly**
- `wasm32_unknown`: WebAssembly 32-bit
- `wasm64_unknown`: WebAssembly 64-bit


### x86-64 CPU Levels

`--cpu-level <v1|v2|v3|v4>` sets the highest x86-64 microarchitecture level the generated code may use. The levels are the psABI ones (`x86-64-v1` to `x86-64-v4`; the `x86-64-` prefix is optional). The default is **v2**, which is what the x86-64 backend already emitted before the flag existed, so the default output is unchanged.

| Level | Features | Gated instructions the backend emits at this level |
|---|---|---|
| v1 | SSE2 (x86-64 baseline) | none |
| v2 | SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B | `popcntq`; `pinsrb/d/q` and `pextrb/d/q`; `pmulld` |
| v3 | v2 + AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE | none yet, so output matches v2 |
| v4 | v3 + AVX-512F, BW, CD, DQ, VL | none yet, so output matches v2 |

At v1:
- `popcount` is lowered to the classic SWAR sequence instead of `popcntq`. This is a deliberate fallback. The sequence saves and restores `%rcx` and `%rdx`.
- The SSE4.1 lane forms and `pmulld` are refused with a named error, for example ``instruction `pmulld` requires x86-64-v2, but the selected CPU level is x86-64-v1 (raise it with --cpu-level)``. These come from the vector path.
- 16-bit lane access and `pmullw` are SSE2 and are still emitted.

`--cpu-level` applies only to x86_64 targets and is rejected for others. `--jit` runs the code on the host, so pick a level the host supports.

## Lamina IR Syntax

Lamina IR files consist of type declarations, global definitions, and function declarations:

```
# Type declaration
type !Vec2 = struct { x: f32, y: f32 }

# Global value
global $message: [6 x i8] = "hello"

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

Heap memory comes from the C allocator, declared like any other extern:

```
@extern
fn @malloc(i64 %n) -> ptr
@extern
fn @free(ptr %p) -> void

# Stack allocation
%ptr = stackalloc.i32

# Heap allocation (size in bytes)
%hptr = call @malloc(4)

# Load and store
store.i32 %ptr, %val
%loaded = load.i32 %ptr

# Optional heap deallocation
call @free(%hptr)
```

### Composite Type Operations

```
# Struct field access
%field_ptr = getfieldptr.!Point %struct, 0
%value = load.f32 %field_ptr

# Array element access
%elem_ptr = getelem.i32 %array, %index
%value = load.i32 %elem_ptr
```

### Conversions

```
# Between integer widths
%wide = extend.i32.i64 %narrow
%narrow = trunc.i64.i32 %wide

# Between integers and floats
%f = inttofp.i64.f64 %i
%i = fptoint.f64.i64 %f

# Between float widths
%d = fpcast.f32.f64 %s
%s = fpcast.f64.f32 %d
```

### Unary Operations

```
%negated = neg.i64 %x      # also f32 and f64
%flipped = not.u32 %mask
%root = sqrt.f64 %x
```

### Globals

```
global $counter: i64 = 41
global $squares: [6 x i64] = [0, 1, 4, 9, 16, 25]
global $greeting: [6 x i8] = "hello"
@export global $shared: u32 = 7
```

### External Functions

```
@extern
fn @strlen(ptr %s) -> i64

%n = call @strlen(%p)
```

### Function Pointers

```
%f = funcaddr @twice
%r = callptr.i64 %f(i64 %v)
```

A `callptr` names the type of every argument, as a function signature does:
`callptr[.<ret-ty>] <ptr>(<ty> <arg>, <ty> <arg>, ...)`. The declared type is what a
literal operand is materialised as and what the ABI classifies the argument by, so
`callptr.i64 %f(f32 1.5)` passes a single-precision `1.5`.

### I/O Operations

`write`, `read`, `writebyte` and `readbyte` return the same values on every backend that implements them.

| Operation | Success | End of input | Failure |
|-----------|---------|--------------|---------|
| `write %buf, n` | bytes written (0 when `n` is 0) | not produced | -2 |
| `writebyte b` | 1 | not produced | -2 |
| `read %buf, n` | bytes read (0 when `n` is 0) | -1 | -2 |
| `readbyte` | the byte, 0 to 255 | -1 | -2 |

-1 is only ever end of input. A request for zero bytes is a success and returns 0 without consulting the input, so it never reports end of input. A NUL byte reads as 0, not as end of input.

On WebAssembly, the non-WASI target imports these operations from the host under the same names, and the host must follow this table. A target that cannot implement an operation rejects it at code generation.

## Future Directions

- **Enhanced Optimizations**: Complete optimization pipeline, loop analysis, auto-vectorization
- **Language Integration**: C and Rust frontends for compiling to Lamina IR
- **Debugging**: `-g` emits `.file`/`.loc` directives. With ras, ELF and Mach-O objects also get minimal DWARF. `ldb` (its own repo) reads it. Parameter locations (`DW_AT_location`) are not emitted yet
- **GPU Acceleration**: CUDA / Vulkan compute shader support
- **SIMD Support**: auto-vectorization. Explicit vectors have landed: `<N x T>` as a type suffix, with `splat`, `extractlane`, `insertlane` and `shuffle`




Please feel free to submit pull requests or open issues for bugs and feature requests.

---

<div align="center">
  <p>
    <strong>Lamina</strong>
  </p>
  <p>
    Built with Rust | Open Source
  </p>
</div>
