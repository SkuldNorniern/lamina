# LUMIR — Lamina Unified Machine Intermediate Representation

A low-level, machine-friendly intermediate representation for the Lamina compiler.

## Architecture

```text
Parser → IR → LUMIR → [Optimizations] → Code Generator → Assembly
                ↑                            ↓
                └───── Transform Passes ──────┘
```

LUMIR sits between high-level IR and target-specific assembly, providing:
- Machine-friendly representation (virtual registers, explicit addressing)
- Architecture-agnostic optimization layer
- Clear path to register allocation and code emission

## Module Structure

```
src/mir/
├── mod.rs           - Documentation & module exports
├── types.rs         - Type system (scalars, vectors)
├── register.rs      - Virtual/physical register management
├── instruction.rs   - Complete instruction set
├── block.rs         - Basic block representation
├── function.rs      - Function & signature representation
├── module.rs        - Module & global variable management
└── transform/       - Optimization pass infrastructure
    └── mod.rs       - Transform trait & categories
```

## Core Concepts

### Types (`types.rs`)

LUMIR supports:
- **Scalars**: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `ptr`, `i1`
- **Vectors**: `v128<lane>`, `v256<lane>` where lane ∈ {i8, i16, i32, i64, f32, f64}

```rust
use lamina::mir::{MirType, ScalarType, VectorType, VectorLane};

let i64_ty = MirType::Scalar(ScalarType::I64);
let vec_ty = MirType::Vector(VectorType::V128(VectorLane::F32));

assert_eq!(i64_ty.size_bytes(), 8);
assert_eq!(vec_ty.size_bytes(), 16);
```

### Registers (`register.rs`)

Three register classes:
- **GPR** (General Purpose): integers, pointers
- **FPR** (Floating Point): scalar floats
- **VEC** (Vector): SIMD operations

Virtual registers are unlimited and assigned sequentially:

```rust
use lamina::mir::{VirtualRegAllocator, RegisterClass};

let mut allocator = VirtualRegAllocator::new();
let v0 = allocator.allocate_gpr(); // v0
let v1 = allocator.allocate_fpr(); // v1
let v2 = allocator.allocate_vec(); // v2
```

Physical registers appear only after register allocation:

```rust
use lamina::mir::{PhysicalReg, RegisterClass};

let rax = PhysicalReg::new("rax", RegisterClass::Gpr);
let xmm0 = PhysicalReg::new("xmm0", RegisterClass::Fpr);
```

### Instructions (`instruction.rs`)

Comprehensive instruction set:

**Integer Arithmetic:**
```rust
Instruction::IntBinary {
    op: IntBinOp::Add,
    ty: MirType::Scalar(ScalarType::I64),
    dst: v2,
    lhs: Operand::Register(v0),
    rhs: Operand::Register(v1),
}
```

**Floating Point:**
```rust
Instruction::FloatBinary {
    op: FloatBinOp::FAdd,
    ty: MirType::Scalar(ScalarType::F64),
    dst: v3,
    lhs: Operand::Register(v1),
    rhs: Operand::Immediate(Immediate::F64(3.14)),
}
```

**Memory Operations:**
```rust
Instruction::Load {
    ty: MirType::Scalar(ScalarType::I32),
    dst: v0,
    addr: AddressMode::BaseOffset {
        base: v1,
        offset: 16,
    },
    attrs: MemoryAttrs { align: 4, volatile: false },
}
```

**Control Flow:**
```rust
Instruction::Br {
    cond: v0,
    true_target: "then_block".to_string(),
    false_target: "else_block".to_string(),
}
```

### Basic Blocks (`block.rs`)

A basic block is a sequence of instructions with single entry/exit:

```rust
use lamina::mir::BasicBlock;

let mut bb = BasicBlock::new("entry");
bb.push(/* instruction */);
bb.push(/* instruction */);
bb.push(Instruction::Ret { value: None });

assert!(bb.has_terminator());
```

### Functions (`function.rs`)

Functions contain signatures and basic blocks:

```rust
use lamina::mir::FunctionBuilder;

let func = FunctionBuilder::new("add")
    .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
    .param(VirtualReg::gpr(1).into(), MirType::Scalar(ScalarType::I64))
    .returns(MirType::Scalar(ScalarType::I64))
    .block("entry")
    .instr(Instruction::IntBinary {
        op: IntBinOp::Add,
        ty: MirType::Scalar(ScalarType::I64),
        dst: VirtualReg::gpr(2).into(),
        lhs: Operand::Register(VirtualReg::gpr(0).into()),
        rhs: Operand::Register(VirtualReg::gpr(1).into()),
    })
    .instr(Instruction::Ret {
        value: Some(Operand::Register(VirtualReg::gpr(2).into())),
    })
    .build();

// Validate structure
func.validate().unwrap();
```

### Modules (`module.rs`)

Modules contain functions and globals:

```rust
use lamina::mir::{ModuleBuilder, Global};

let module = ModuleBuilder::new("my_module")
    .function(/* function */)
    .global(Global::new("counter", MirType::Scalar(ScalarType::I64)))
    .build();

module.validate().unwrap();
```

## Transform System (`transform/`)

Optimization passes operate on LUMIR:

```rust
pub trait Transform: Default {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn category(&self) -> TransformCategory;
    fn level(&self) -> TransformLevel;
    // Future: fn apply(&mut self, func: &mut Function) -> Result<TransformStats>;
}
```

**Transform Categories:**
- DeadCodeElimination
- Inlining
- ConstantFolding
- CopyPropagation
- InstructionSelection
- ControlFlowOptimization
- ArithmeticOptimization
- MemoryOptimization

**Stability Levels:**
- `Stable`: Production-ready, always enabled
- `Experimental`: Testing phase, enabled at `-O2+`
- `Deprecated`: Being phased out, only at `-O3`

## Instruction Helpers

Instructions provide useful analysis methods:

```rust
let instr = Instruction::IntBinary { /* ... */ };

// Get destination register
if let Some(dst) = instr.def_reg() {
    println!("Defines: {}", dst);
}

// Get used registers
for reg in instr.use_regs() {
    println!("Uses: {}", reg);
}

// Check if terminator
if instr.is_terminator() {
    println!("Ends basic block");
}
```

## Addressing Modes

Two addressing modes supported:

**Simple Offset:**
```rust
AddressMode::BaseOffset {
    base: v0,      // Base register
    offset: 16,    // 12-bit signed immediate
}
// Generates: [v0 + 16]
```

**Indexed with Scale:**
```rust
AddressMode::BaseIndexScale {
    base: v0,      // Base register
    index: v1,     // Index register
    scale: 4,      // 1, 2, 4, or 8
    offset: 8,     // 4-bit signed immediate
}
// Generates: [v0 + v1*4 + 8]
```

## Calling Convention

**Abstract convention:**
- Arguments: `v0..v7` (8 abstract argument registers)
- Return: `v0` (abstract return register)

**Mapped to real ABI during emission:**
- **x86_64 System V**: `rdi, rsi, rdx, rcx, r8, r9, [stack]`
- **AArch64 AAPCS**: `x0-x7, [stack]`
- **WASM**: Stack-based

## Testing

All modules include comprehensive unit tests:

```bash
cargo test --lib mir
```

Currently: **19 tests, 100% passing**

## Future Work

### IR → LUMIR Lowering
Convert high-level IR to LUMIR representation.

### Optimization Passes
Implement transforms:
- Constant folding
- Dead code elimination
- Copy propagation
- Common subexpression elimination
- Loop optimizations

### Register Allocation
Assign physical registers to virtual registers:
- Linear scan allocation
- Graph coloring
- Live range splitting

### Code Emission
Generate assembly from LUMIR:
- x86_64 instruction selection
- AArch64 instruction selection
- Instruction scheduling
- Peephole optimizations

## Design Principles

1. **No Clones**: All data structures use references where possible
2. **Zero External Dependencies**: Pure Rust implementation
3. **Rust 2024**: Uses modern Rust patterns and features
4. **Well-Tested**: Comprehensive unit test coverage
5. **Documented**: Clear documentation and examples
6. **Extensible**: Easy to add new instructions and transforms

## Status

**Current:** Complete skeleton implementation (~1,500 LOC)
- ✅ Type system
- ✅ Register management
- ✅ Complete instruction set
- ✅ Basic blocks
- ✅ Functions & modules
- ✅ Transform infrastructure
- ✅ Unit tests

**Next Steps:**
- IR → LUMIR lowering pass
- First optimization transforms
- Register allocator integration
- x86_64/AArch64 code emission



