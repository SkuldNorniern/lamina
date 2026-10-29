//! Random straight-line blocks run on a concrete byte-addressed machine. Every analysis
//! answer is checked against the addresses actually touched, and the memory pass must
//! leave the defined values and memory unchanged.

use super::alias::{AliasResult, alias, scan_block};
use super::memory::MemoryOptimization;
use super::test_utils::apply_pass;
use crate::mir::{
    AddressMode, AddressSpace, Function, FunctionBuilder, Immediate, Instruction, IntBinOp,
    MemoryAttrs, MirType, Operand, Register, ScalarType, VirtualReg,
};
use std::collections::BTreeMap;

const POOL: u32 = 6;
const SLOTS: u32 = 3;
const SEEDS: u64 = 4000;
const HEAP_BASE: u64 = 0x1000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

fn pool_reg(id: u32) -> Register {
    VirtualReg::gpr(id).into()
}

/// Half the time a base is one of the first two pool registers, so accesses repeat.
fn base_reg(rng: &mut Rng) -> Register {
    if rng.below(2) == 0 {
        pool_reg(rng.below(2) as u32)
    } else {
        pool_reg(rng.below(POOL as u64) as u32)
    }
}

fn slot_reg(id: u32) -> VirtualReg {
    VirtualReg::gpr(100 + id)
}

fn ptr() -> MirType {
    MirType::Scalar(ScalarType::Ptr)
}

fn i64_ty() -> MirType {
    MirType::Scalar(ScalarType::I64)
}

fn attrs() -> MemoryAttrs {
    MemoryAttrs {
        volatile: false,
        ..MemoryAttrs::default()
    }
}

/// Slot placement: slot i starts where slot i-1 ends, so an access that leaves its frame
/// object lands in the neighbour.
struct Layout {
    bases: Vec<u64>,
}

fn layout(words: &[usize]) -> Layout {
    let mut bases = Vec::new();
    let mut next = HEAP_BASE;
    for &w in words {
        let bytes = 8 * (1 + w as u64);
        bases.push(next);
        next += bytes;
    }
    Layout { bases }
}

fn random_program(rng: &mut Rng) -> (Function, Layout, Vec<u64>) {
    let words: Vec<usize> = (0..SLOTS).map(|_| rng.below(3) as usize).collect();
    let layout = layout(&words);
    let offsets = [-8, -4, 0, 0, 8, 8, 12, 16, 20, 24, 32, 40];
    let widths = [
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::I64,
        ScalarType::I64,
        ScalarType::Ptr,
    ];

    let mut builder = FunctionBuilder::new("f");
    for id in 0..POOL {
        builder = builder.param(pool_reg(id), ptr());
    }
    builder = builder.returns(i64_ty()).block("entry");

    let len = 8 + rng.below(20) as usize;
    let sparse = rng.below(2) == 0;
    let mut instructions = Vec::new();
    for _ in 0..len {
        let dst = pool_reg(rng.below(POOL as u64) as u32);
        let src = pool_reg(rng.below(POOL as u64) as u32);
        let mut kind = rng.below(16);
        if sparse && kind < 6 && rng.below(4) != 0 {
            kind = 6 + kind % 10;
        }
        let inst = match kind {
            0 | 1 => Instruction::StackAddr {
                dst,
                slot: slot_reg(rng.below(SLOTS as u64) as u32),
            },
            2 => Instruction::Lea {
                dst,
                base: src,
                offset: rng.pick(&[-16, -8, -4, 0, 4, 8, 12, 16, 24, 40, i32::MAX, i32::MIN]),
            },
            3 => Instruction::Copy {
                ty: ptr(),
                dst,
                src: Operand::Register(src),
            },
            4 => Instruction::Copy {
                ty: i64_ty(),
                dst,
                src: Operand::Immediate(Immediate::I64(rng.pick(&[0, 7, 9, -1]))),
            },
            5 => Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64_ty(),
                dst,
                lhs: Operand::Register(src),
                rhs: Operand::Immediate(Immediate::I64(rng.pick(&[1, 8, -8]))),
            },
            6..=9 => {
                let width = rng.pick(&widths);
                let ty = MirType::Scalar(width);
                Instruction::Store {
                    ty,
                    src: Operand::Register(src),
                    addr: AddressMode::BaseOffset {
                        base: base_reg(rng),
                        offset: rng.pick(&offsets) as i16,
                        address_space: AddressSpace::default(),
                    },
                    attrs: attrs(),
                }
            }
            10..=13 => {
                let ty = MirType::Scalar(rng.pick(&widths));
                Instruction::Load {
                    ty,
                    dst,
                    addr: AddressMode::BaseOffset {
                        base: base_reg(rng),
                        offset: rng.pick(&offsets) as i16,
                        address_space: AddressSpace::default(),
                    },
                    attrs: attrs(),
                }
            }
            _ => Instruction::Call {
                name: "opaque".to_owned(),
                args: Vec::new(),
                ret: None,
            },
        };
        instructions.push(inst);
    }
    instructions.push(Instruction::Ret { value: None });
    for inst in instructions {
        builder = builder.instr(inst);
    }
    let mut func = builder.build();
    for (id, &w) in words.iter().enumerate() {
        if w > 0 {
            func.stack_reservations.insert(slot_reg(id as u32), w);
        }
    }

    let mut initial = Vec::new();
    for _ in 0..POOL {
        let value = match rng.below(5) {
            0 => {
                let slot = rng.below(SLOTS as u64) as usize;
                layout.bases[slot].wrapping_add(rng.pick(&[0, 4, 8, 16]))
            }
            1 => rng.pick(&[0u64, 1, u64::MAX - 7, 0x1008, 0x1010]),
            2 => layout.bases[0].wrapping_add(rng.below(40)).wrapping_sub(16),
            _ => rng.next(),
        };
        initial.push(value);
    }
    (func, layout, initial)
}

#[derive(Clone, Default)]
struct Machine {
    slot_bases: BTreeMap<u32, u64>,
    memory: BTreeMap<u64, u8>,
    regs: BTreeMap<u32, u64>,
    /// Concrete (address, width) of the access each instruction made, if any.
    touched: Vec<Option<(u64, u64)>>,
    /// Value each instruction defined, in execution order.
    defined: Vec<(usize, u64)>,
}

fn width_bytes(ty: MirType) -> u64 {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => 1,
        MirType::Scalar(ScalarType::I16) => 2,
        MirType::Scalar(ScalarType::I32 | ScalarType::F32) => 4,
        _ => 8,
    }
}

fn truncate(value: u64, ty: MirType) -> u64 {
    match width_bytes(ty) {
        8 => value,
        n => value & ((1u64 << (8 * n)) - 1),
    }
}

impl Machine {
    fn eval(&self, operand: &Operand) -> u64 {
        match operand {
            Operand::Register(Register::Virtual(reg)) => {
                self.regs.get(&reg.id).copied().unwrap_or(0)
            }
            Operand::Register(_) => 0,
            Operand::Immediate(imm) => match imm {
                Immediate::I8(v) => *v as i64 as u64,
                Immediate::I16(v) => *v as i64 as u64,
                Immediate::I32(v) => *v as i64 as u64,
                Immediate::I64(v) => *v as u64,
                _ => 0,
            },
        }
    }

    fn address(&self, addr: &AddressMode) -> u64 {
        match addr {
            AddressMode::BaseOffset { base, offset, .. } => {
                let held = self.eval(&Operand::Register(base.clone()));
                held.wrapping_add(i64::from(*offset) as u64)
            }
            AddressMode::BaseIndexScale { .. } => 0,
        }
    }

    fn write(&mut self, at: u64, value: u64, bytes: u64) {
        for i in 0..bytes {
            self.memory
                .insert(at.wrapping_add(i), (value >> (8 * i)) as u8);
        }
    }

    fn read(&self, at: u64, bytes: u64) -> u64 {
        (0..bytes).fold(0, |acc, i| {
            let byte = self.memory.get(&at.wrapping_add(i)).copied().unwrap_or(0);
            acc | (u64::from(byte) << (8 * i))
        })
    }

    fn define(&mut self, index: usize, dst: &Register, value: u64) {
        if let Register::Virtual(reg) = dst {
            self.regs.insert(reg.id, value);
            self.defined.push((index, value));
        }
    }

    fn run(&mut self, func: &Function) {
        for (index, inst) in func.blocks[0].instructions.iter().enumerate() {
            match &inst.instr {
                Instruction::StackAddr { dst, slot } => {
                    let base = self.slot_bases.get(&slot.id).copied().unwrap_or(0);
                    self.define(index, dst, base);
                }
                Instruction::Lea { dst, base, offset } => {
                    let value = self
                        .eval(&Operand::Register(base.clone()))
                        .wrapping_add(i64::from(*offset) as u64);
                    self.define(index, dst, value);
                }
                Instruction::Copy { ty, dst, src } => {
                    let value = truncate(self.eval(src), *ty);
                    self.define(index, dst, value);
                }
                Instruction::IntBinary {
                    ty, dst, lhs, rhs, ..
                } => {
                    let value = truncate(self.eval(lhs).wrapping_add(self.eval(rhs)), *ty);
                    self.define(index, dst, value);
                }
                Instruction::Load { ty, dst, addr, .. } => {
                    let at = self.address(addr);
                    let bytes = width_bytes(*ty);
                    self.touched[index] = Some((at, bytes));
                    let value = self.read(at, bytes);
                    self.define(index, dst, value);
                }
                Instruction::Store { ty, src, addr, .. } => {
                    let at = self.address(addr);
                    let bytes = width_bytes(*ty);
                    self.touched[index] = Some((at, bytes));
                    let value = truncate(self.eval(src), *ty);
                    self.write(at, value, bytes);
                }
                _ => {}
            }
        }
    }
}

fn run_machine(func: &Function, layout: &Layout, initial: &[u64]) -> Machine {
    let mut machine = Machine {
        touched: vec![None; func.blocks[0].instructions.len()],
        ..Machine::default()
    };
    for (id, value) in initial.iter().enumerate() {
        machine.regs.insert(id as u32, *value);
    }
    for (id, base) in layout.bases.iter().enumerate() {
        machine.slot_bases.insert(100 + id as u32, *base);
    }
    for offset in 0..0x200u64 {
        let byte = (offset as u8).wrapping_mul(37).wrapping_add(11);
        machine.memory.insert(HEAP_BASE + offset, byte);
    }
    machine.run(func);
    machine
}

fn spans_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.0.wrapping_add(b.1) && b.0 < a.0.wrapping_add(a.1)
}

/// Checks every answer the analysis gives for pairs of accesses in the block against the
/// addresses the machine touched. Returns how many pairs were answered `NoAlias` and how
/// many `SameRange`.
fn check_queries(func: &Function, machine: &Machine, seed: u64) -> (usize, usize) {
    let accesses = scan_block(&func.blocks[0], &func.stack_reservations);
    let mut no_alias = 0;
    let mut same_range = 0;
    for i in 0..accesses.len() {
        for j in (i + 1)..accesses.len() {
            let (Some(a), Some(b)) = (accesses[i], accesses[j]) else {
                continue;
            };
            let forward = alias(&a, &b);
            let backward = alias(&b, &a);
            assert_eq!(
                forward, backward,
                "seed {seed}: asymmetric answer for {i} and {j}"
            );
            let (Some(x), Some(y)) = (machine.touched[i], machine.touched[j]) else {
                continue;
            };
            match forward {
                AliasResult::NoAlias => {
                    no_alias += 1;
                    assert!(
                        !spans_overlap(x, y),
                        "seed {seed}: NoAlias for {i} {x:?} and {j} {y:?} which overlap in {func:?}"
                    );
                }
                AliasResult::SameRange => {
                    same_range += 1;
                    assert_eq!(
                        x, y,
                        "seed {seed}: SameRange for {i} and {j} with different spans"
                    );
                }
                AliasResult::MayAlias => {}
            }
        }
    }
    (no_alias, same_range)
}

#[test]
fn analysis_answers_agree_with_concrete_addresses() {
    let mut totals = (0usize, 0usize);
    for seed in 1..=SEEDS {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let (func, layout, initial) = random_program(&mut rng);
        let machine = run_machine(&func, &layout, &initial);
        let (no_alias, same_range) = check_queries(&func, &machine, seed);
        totals.0 += no_alias;
        totals.1 += same_range;
    }
    eprintln!(
        "alias answers over {SEEDS} blocks: NoAlias {} SameRange {}",
        totals.0, totals.1
    );
}

#[test]
fn memory_pass_preserves_every_defined_value_and_memory() {
    let mut forwarded = 0usize;
    for seed in 1..=SEEDS {
        let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
        let (func, layout, initial) = random_program(&mut rng);
        let before = run_machine(&func, &layout, &initial);

        let mut rewritten = func.clone();
        if apply_pass(&MemoryOptimization, &mut rewritten) {
            forwarded += 1;
        }
        let after = run_machine(&rewritten, &layout, &initial);

        assert_eq!(
            before.defined, after.defined,
            "seed {seed}: rewritten block defines different values\nbefore {func:?}\nafter {rewritten:?}"
        );
        assert_eq!(
            before.memory, after.memory,
            "seed {seed}: rewritten block leaves different memory"
        );
    }
    eprintln!("memory pass changed {forwarded} of {SEEDS} generated blocks");
}
