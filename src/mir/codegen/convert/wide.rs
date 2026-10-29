//! Lowering of `i128` and `u128` during IR-to-MIR conversion.
//!
//! No target has 128-bit integer registers, so a 128-bit value is two 64-bit words
//! everywhere. A wide value lives in a pair of 64-bit virtual registers, and every
//! operation becomes the sequence that computes both words. The backends and the
//! register allocators only ever see 64-bit values.
//!
//! Values cross a function boundary in memory, as 16-byte integer aggregates. That is
//! the convention the backends already implement, and System V classifies an
//! `__int128` the same way: two INTEGER words, register pair or stack, never split.
//! Division, remainder and float conversion call compiler-rt, through the
//! [`RuntimeSymbol`] seam, with the same aggregate convention.

use super::*;
use crate::ir::instruction::{AsmOperand as IRAsmOperand, UnaryOp};
use crate::ir::types::Literal;
use crate::mir::codegen::mapping::wide_aggregate;
use lamina_codegen::abi::BuiltinLibrary;
use lamina_platform::{Endianness, PointerWidth, TargetOperatingSystem};

/// The integer that takes two registers on a target with this register width: `i128`
/// where a register holds 64 bits, `i64` where it holds 32.
pub(super) fn is_wide_prim(ty: IRPrim, int64_regs: bool) -> bool {
    if int64_regs {
        matches!(ty, IRPrim::I128 | IRPrim::U128)
    } else {
        matches!(ty, IRPrim::I64 | IRPrim::U64)
    }
}

/// A value wider than a register is moved a word at a time, so the access the
/// instruction asks for is not the access the hardware performs. Rather than let a
/// stated alignment mean nothing, refuse it.
fn reject_wide_alignment(
    align: Option<u32>,
    ty: &IRType<'_>,
    what: &str,
) -> Result<(), FromIRError> {
    match align {
        None => Ok(()),
        Some(align) => Err(FromIRError::UnsupportedAlignment(format!(
            "{what}.{ty} align {align}: a value wider than a register is moved one word at a time, so the stated alignment would not describe the access performed"
        ))),
    }
}

pub(super) fn is_wide_type(ty: &IRType<'_>, int64_regs: bool) -> bool {
    matches!(ty, IRType::Primitive(p) if is_wide_prim(*p, int64_regs))
}

/// A 128-bit integer needs four registers on a 32-bit target, which this lowering does
/// not build, so it is refused by name rather than truncated.
pub(super) fn is_quad_prim(ty: IRPrim, int64_regs: bool) -> bool {
    !int64_regs && matches!(ty, IRPrim::I128 | IRPrim::U128)
}

pub(super) fn is_quad_type(ty: &IRType<'_>, int64_regs: bool) -> bool {
    matches!(ty, IRType::Primitive(p) if is_quad_prim(*p, int64_regs))
}

pub(super) fn word_ty(int64_regs: bool) -> MirType {
    if int64_regs {
        MirType::Scalar(ScalarType::I64)
    } else {
        MirType::Scalar(ScalarType::I32)
    }
}

const fn word_bits(int64_regs: bool) -> i64 {
    if int64_regs { 64 } else { 32 }
}

const fn word_bytes(int64_regs: bool) -> i32 {
    if int64_regs { 8 } else { 4 }
}

/// Byte offsets of the low and high word of a pair held in memory. The high word comes
/// first in big-endian order, so a pair written out is not two words at fixed places.
const fn word_offsets(int64_regs: bool, endianness: Endianness) -> (i32, i32) {
    let width = word_bytes(int64_regs);
    match endianness {
        Endianness::Little => (0, width),
        Endianness::Big => (width, 0),
    }
}

fn word_imm(int64_regs: bool, value: i64) -> Operand {
    if int64_regs {
        Operand::Immediate(Immediate::I64(value))
    } else {
        Operand::Immediate(Immediate::I32(value as i32))
    }
}

#[derive(Clone)]
pub(super) struct Words {
    lo: Operand,
    hi: Operand,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Abi {
    Wide,
    Scalar(MirType),
}

#[derive(Clone, Debug)]
pub(super) struct LibCall {
    name: String,
    params: Vec<Abi>,
    ret: Option<Abi>,
}

impl LibCall {
    pub(super) fn declaration(&self, int64_regs: bool) -> Function {
        let mut sig = Signature::new(self.name.clone());
        for (index, param) in self.params.iter().enumerate() {
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            let parameter = match param {
                Abi::Wide => Parameter::new(
                    Register::Virtual(VirtualReg::gpr(slot)),
                    MirType::Scalar(ScalarType::Ptr),
                )
                .with_aggregate(wide_aggregate(int64_regs)),
                Abi::Scalar(ty) => {
                    let class = if ty.is_float() {
                        RegisterClass::Fpr
                    } else {
                        RegisterClass::Gpr
                    };
                    Parameter::new(Register::Virtual(VirtualReg::new(slot, class)), *ty)
                }
            };
            sig.params.push(parameter);
        }
        match self.ret {
            None => {}
            Some(Abi::Wide) => {
                sig = sig
                    .with_return(MirType::Scalar(ScalarType::Ptr))
                    .with_return_aggregate(wide_aggregate(int64_regs));
            }
            Some(Abi::Scalar(ty)) => sig = sig.with_return(ty),
        }
        Function::new(sig)
    }
}

pub(super) struct Cx<'c, 'a> {
    pub(super) vregs: &'c mut VirtualRegAllocator,
    pub(super) vars: &'c mut VarBindings<'a>,
    pub(super) reservations: &'c mut HashMap<VirtualReg, usize>,
    pub(super) globals: &'c mut GlobalAddresses<'a>,
    pub(super) module: &'c IRModule<'a>,
    pub(super) pointer: PointerWidth,
    pub(super) int64_regs: bool,
    pub(super) endianness: Endianness,
    pub(super) out: Vec<Instruction>,
}

impl<'c, 'a> Cx<'c, 'a> {
    fn word(&self) -> MirType {
        word_ty(self.int64_regs)
    }

    fn bits(&self) -> i64 {
        word_bits(self.int64_regs)
    }

    fn imm(&self, value: i64) -> Operand {
        word_imm(self.int64_regs, value)
    }

    fn temp(&mut self) -> Register {
        Register::Virtual(self.vregs.allocate_gpr())
    }

    fn bin(&mut self, op: IntBinOp, lhs: Operand, rhs: Operand) -> Operand {
        if let (Some(a), Some(b)) = (lhs.integer_constant(), rhs.integer_constant())
            && matches!(lhs, Operand::Immediate(_))
            && matches!(rhs, Operand::Immediate(_))
            && let Some(value) = fold(op, a, b, self.int64_regs)
        {
            return self.imm(value);
        }
        let dst = self.temp();
        let ty = self.word();
        self.out.push(Instruction::IntBinary {
            op,
            ty,
            dst: dst.clone(),
            lhs,
            rhs,
        });
        Operand::Register(dst)
    }

    fn not(&mut self, value: Operand) -> Operand {
        let ones = self.imm(-1);
        self.bin(IntBinOp::Xor, value, ones)
    }

    fn nonzero(&mut self, value: Operand) -> Operand {
        let zero = self.imm(0);
        let negated = self.bin(IntBinOp::Sub, zero, value.clone());
        let any = self.bin(IntBinOp::Or, value, negated);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, any, top)
    }

    fn is_zero(&mut self, value: Operand) -> Operand {
        let nonzero = self.nonzero(value);
        let one = self.imm(1);
        self.bin(IntBinOp::Xor, nonzero, one)
    }

    fn carry_out(&mut self, x: Operand, y: Operand, sum: Operand) -> Operand {
        let both = self.bin(IntBinOp::And, x.clone(), y.clone());
        let either = self.bin(IntBinOp::Or, x, y);
        let not_sum = self.not(sum);
        let propagate = self.bin(IntBinOp::And, either, not_sum);
        let joined = self.bin(IntBinOp::Or, both, propagate);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, joined, top)
    }

    fn borrow_out(&mut self, x: Operand, y: Operand, diff: Operand) -> Operand {
        let not_x = self.not(x.clone());
        let lent = self.bin(IntBinOp::And, not_x, y.clone());
        let differ = self.bin(IntBinOp::Xor, x, y);
        let same = self.not(differ);
        let propagate = self.bin(IntBinOp::And, same, diff);
        let joined = self.bin(IntBinOp::Or, lent, propagate);
        let top = self.imm(self.bits() - 1);
        self.bin(IntBinOp::LShr, joined, top)
    }

    fn add(&mut self, a: Words, b: Words) -> Words {
        let lo = self.bin(IntBinOp::Add, a.lo.clone(), b.lo.clone());
        let carry = self.carry_out(a.lo, b.lo, lo.clone());
        let hi_sum = self.bin(IntBinOp::Add, a.hi, b.hi);
        let hi = self.bin(IntBinOp::Add, hi_sum, carry);
        Words { lo, hi }
    }

    fn sub(&mut self, a: Words, b: Words) -> Words {
        let lo = self.bin(IntBinOp::Sub, a.lo.clone(), b.lo.clone());
        let borrow = self.borrow_out(a.lo, b.lo, lo.clone());
        let hi_diff = self.bin(IntBinOp::Sub, a.hi, b.hi);
        let hi = self.bin(IntBinOp::Sub, hi_diff, borrow);
        Words { lo, hi }
    }

    fn mul_high_unsigned(&mut self, a: Operand, b: Operand) -> Operand {
        let half = self.bits() / 2;
        let mask = self.imm((1i64 << half) - 1);
        let half = self.imm(half);
        let a0 = self.bin(IntBinOp::And, a.clone(), mask.clone());
        let a1 = self.bin(IntBinOp::LShr, a, half.clone());
        let b0 = self.bin(IntBinOp::And, b.clone(), mask.clone());
        let b1 = self.bin(IntBinOp::LShr, b, half.clone());
        let w0 = self.bin(IntBinOp::Mul, a0.clone(), b0.clone());
        let w0_high = self.bin(IntBinOp::LShr, w0, half.clone());
        let a1b0 = self.bin(IntBinOp::Mul, a1.clone(), b0);
        let t = self.bin(IntBinOp::Add, a1b0, w0_high);
        let t_low = self.bin(IntBinOp::And, t.clone(), mask);
        let t_high = self.bin(IntBinOp::LShr, t, half.clone());
        let a0b1 = self.bin(IntBinOp::Mul, a0, b1.clone());
        let w1 = self.bin(IntBinOp::Add, t_low, a0b1);
        let w1_high = self.bin(IntBinOp::LShr, w1, half);
        let a1b1 = self.bin(IntBinOp::Mul, a1, b1);
        let partial = self.bin(IntBinOp::Add, a1b1, t_high);
        self.bin(IntBinOp::Add, partial, w1_high)
    }

    fn mul(&mut self, a: Words, b: Words) -> Words {
        let lo = self.bin(IntBinOp::Mul, a.lo.clone(), b.lo.clone());
        let high = self.mul_high_unsigned(a.lo.clone(), b.lo.clone());
        let cross_b = self.bin(IntBinOp::Mul, a.lo, b.hi);
        let cross_a = self.bin(IntBinOp::Mul, a.hi, b.lo);
        let high_cross = self.bin(IntBinOp::Add, cross_a, cross_b);
        let hi = self.bin(IntBinOp::Add, high, high_cross);
        Words { lo, hi }
    }

    /// Splits the shift count into its masked low six bits, and a mask that is all ones
    /// when the count is 64 or more.
    fn shift_parts(&mut self, count: Operand) -> (Operand, Operand) {
        let both = self.imm(2 * self.bits() - 1);
        let count = self.bin(IntBinOp::And, count, both);
        let within = self.imm(self.bits() - 1);
        let small = self.bin(IntBinOp::And, count.clone(), within);
        let log2 = self.imm(self.bits().trailing_zeros().into());
        let big = self.bin(IntBinOp::LShr, count, log2);
        let zero = self.imm(0);
        let mask = self.bin(IntBinOp::Sub, zero, big);
        (small, mask)
    }

    /// Bits of the low word that move up into the high word for a left shift below 64.
    /// Written as two shifts, so a count of zero contributes nothing and no shift
    /// reaches 64.
    fn cross_shift_left(&mut self, low: Operand, count: Operand) -> Operand {
        let one = self.imm(1);
        let halved = self.bin(IntBinOp::LShr, low, one);
        let top = self.imm(self.bits() - 1);
        let rest = self.bin(IntBinOp::Xor, count, top);
        self.bin(IntBinOp::LShr, halved, rest)
    }

    /// Bits of the high word that move down into the low word for a right shift below
    /// 64, with the same two-shift form.
    fn cross_shift_right(&mut self, high: Operand, count: Operand) -> Operand {
        let one = self.imm(1);
        let doubled = self.bin(IntBinOp::Shl, high, one);
        let top = self.imm(self.bits() - 1);
        let rest = self.bin(IntBinOp::Xor, count, top);
        self.bin(IntBinOp::Shl, doubled, rest)
    }

    fn select_mask(&mut self, mask: Operand, when_set: Operand, when_clear: Operand) -> Operand {
        let not_mask = self.not(mask.clone());
        let set = self.bin(IntBinOp::And, when_set, mask);
        let clear = self.bin(IntBinOp::And, when_clear, not_mask);
        self.bin(IntBinOp::Or, set, clear)
    }

    fn shl(&mut self, a: Words, count: Operand) -> Words {
        let (small, mask) = self.shift_parts(count);
        let not_mask = self.not(mask.clone());
        let low = self.bin(IntBinOp::Shl, a.lo.clone(), small.clone());
        let high_shifted = self.bin(IntBinOp::Shl, a.hi.clone(), small.clone());
        let crossed = self.cross_shift_left(a.lo, small);
        let high_small = self.bin(IntBinOp::Or, high_shifted, crossed);
        let lo = self.bin(IntBinOp::And, low.clone(), not_mask);
        let hi = self.select_mask(mask, low, high_small);
        Words { lo, hi }
    }

    fn right_shift(&mut self, a: Words, count: Operand, arithmetic: bool) -> Words {
        let (small, mask) = self.shift_parts(count);
        let not_mask = self.not(mask.clone());
        let shifted_low = self.bin(IntBinOp::LShr, a.lo.clone(), small.clone());
        let crossed = self.cross_shift_right(a.hi.clone(), small.clone());
        let low_small = self.bin(IntBinOp::Or, shifted_low, crossed);
        let high_op = if arithmetic {
            IntBinOp::AShr
        } else {
            IntBinOp::LShr
        };
        let high_shifted = self.bin(high_op, a.hi.clone(), small);
        let fill = if arithmetic {
            let top = self.imm(self.bits() - 1);
            self.bin(IntBinOp::AShr, a.hi, top)
        } else {
            self.imm(0)
        };
        let lo = self.select_mask(mask.clone(), high_shifted.clone(), low_small);
        let hi_small = self.bin(IntBinOp::And, high_shifted, not_mask);
        let hi_fill = self.bin(IntBinOp::And, fill, mask);
        let hi = self.bin(IntBinOp::Or, hi_small, hi_fill);
        Words { lo, hi }
    }

    fn unary_word(&mut self, op: IntUnOp, src: Operand) -> Operand {
        let dst = self.temp();
        let ty = self.word();
        self.out.push(Instruction::IntUnary {
            op,
            ty,
            dst: dst.clone(),
            src,
        });
        Operand::Register(dst)
    }

    fn popcount(&mut self, a: Words) -> Words {
        let lo = self.unary_word(IntUnOp::Popcount, a.lo);
        let hi = self.unary_word(IntUnOp::Popcount, a.hi);
        Words {
            lo: self.bin(IntBinOp::Add, lo, hi),
            hi: self.imm(0),
        }
    }

    /// The high word's count, or 64 plus the low word's when the high word is zero.
    fn leading_zeros(&mut self, a: Words) -> Words {
        let high = self.unary_word(IntUnOp::Ctlz, a.hi.clone());
        let low = self.unary_word(IntUnOp::Ctlz, a.lo.clone());
        let mask = self.is_zero(a.hi);
        let zero = self.imm(0);
        let mask = self.bin(IntBinOp::Sub, zero, mask);
        let low = self.bin(IntBinOp::And, low, mask);
        Words {
            lo: self.bin(IntBinOp::Add, high, low),
            hi: self.imm(0),
        }
    }

    /// The low word's count, or 64 plus the high word's when the low word is zero.
    fn trailing_zeros(&mut self, a: Words) -> Words {
        let low = self.unary_word(IntUnOp::Cttz, a.lo.clone());
        let high = self.unary_word(IntUnOp::Cttz, a.hi.clone());
        let mask = self.is_zero(a.lo);
        let zero = self.imm(0);
        let mask = self.bin(IntBinOp::Sub, zero, mask);
        let high = self.bin(IntBinOp::And, high, mask);
        Words {
            lo: self.bin(IntBinOp::Add, low, high),
            hi: self.imm(0),
        }
    }

    /// Rotation by `count` modulo 128. The shifts mask their count, so the opposite
    /// shift takes `-count`, and a zero count leaves the value unchanged.
    fn rotate(&mut self, a: Words, count: Operand, left: bool) -> Words {
        let zero = self.imm(0);
        let opposite = self.bin(IntBinOp::Sub, zero, count.clone());
        let (toward, away) = if left {
            (count, opposite)
        } else {
            (opposite, count)
        };
        let up = self.shl(a.clone(), toward);
        let down = self.right_shift(a, away, false);
        Words {
            lo: self.bin(IntBinOp::Or, up.lo, down.lo),
            hi: self.bin(IntBinOp::Or, up.hi, down.hi),
        }
    }

    fn unsigned_less(&mut self, a: Words, b: Words) -> Operand {
        let lo = self.bin(IntBinOp::Sub, a.lo.clone(), b.lo.clone());
        let borrow = self.borrow_out(a.lo, b.lo, lo);
        let hi_diff = self.bin(IntBinOp::Sub, a.hi.clone(), b.hi.clone());
        let hi = self.bin(IntBinOp::Sub, hi_diff, borrow);
        self.borrow_out(a.hi, b.hi, hi)
    }

    fn flip_sign(&mut self, a: Words) -> Words {
        let sign = self.imm(sign_bit(self.int64_regs));
        Words {
            lo: a.lo,
            hi: self.bin(IntBinOp::Xor, a.hi, sign),
        }
    }

    /// The address `offset` bytes past `base`, as a register, so the access itself
    /// uses offset zero, the only offset every backend accepts.
    fn word_address(&mut self, base: &Register, offset: i32) -> Register {
        let dst = self.temp();
        self.out.push(Instruction::Lea {
            dst: dst.clone(),
            base: base.clone(),
            offset,
        });
        dst
    }

    fn store_words(&mut self, base: Register, words: Words) {
        let attrs = MemoryAttrs {
            align: word_bytes(self.int64_regs) as u8,
            volatile: false,
            stated_align: false,
        };
        let ty = self.word();
        let (lo_at, hi_at) = word_offsets(self.int64_regs, self.endianness);
        for (src, offset) in [(words.lo, lo_at), (words.hi, hi_at)] {
            let base = if offset == 0 {
                base.clone()
            } else {
                self.word_address(&base, offset)
            };
            self.out.push(Instruction::Store {
                ty,
                src,
                addr: AddressMode::BaseOffset {
                    base,
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs,
            });
        }
    }

    fn load_words(&mut self, base: Register) -> Words {
        let attrs = MemoryAttrs {
            align: word_bytes(self.int64_regs) as u8,
            volatile: false,
            stated_align: false,
        };
        let ty = self.word();
        let (lo_at, hi_at) = word_offsets(self.int64_regs, self.endianness);
        let word = |cx: &mut Self, offset: i32| {
            let base = if offset == 0 {
                base.clone()
            } else {
                cx.word_address(&base, offset)
            };
            let dst = cx.temp();
            cx.out.push(Instruction::Load {
                ty,
                dst: dst.clone(),
                addr: AddressMode::BaseOffset {
                    base,
                    offset: 0,
                    address_space: Default::default(),
                },
                attrs,
            });
            Operand::Register(dst)
        };
        let lo = word(self, lo_at);
        let hi = word(self, hi_at);
        Words { lo, hi }
    }

    fn spill(&mut self, words: Words) -> Operand {
        let address = self.pair_slot();
        self.store_words(address.clone(), words);
        Operand::Register(address)
    }

    fn libcall_name(
        &mut self,
        symbol: RuntimeSymbol,
        params: Vec<Abi>,
        ret: Option<Abi>,
    ) -> String {
        let name = BuiltinLibrary::from_env(TargetOperatingSystem::Unknown)
            .c_name(symbol)
            .to_string();
        self.vars
            .libcalls
            .entry(name.clone())
            .or_insert_with(|| LibCall {
                name: name.clone(),
                params,
                ret,
            });
        name
    }

    fn call_wide_result(&mut self, name: String, args: Vec<Operand>) -> Words {
        let slot = self.vregs.allocate_gpr();
        self.reservations.insert(slot, 1);
        self.out.push(Instruction::Call {
            name,
            args,
            ret: Some(Register::Virtual(slot)),
        });
        let address = self.temp();
        self.out.push(Instruction::StackAddr {
            dst: address.clone(),
            slot,
        });
        self.load_words(address)
    }

    fn word_of(
        &mut self,
        ty: IRPrim,
        operand: Operand,
        signed: bool,
    ) -> Result<Operand, FromIRError> {
        let from = map_ir_prim(ty)?;
        let to = self.word();
        if from == to {
            return Ok(operand);
        }
        let dst = self.temp();
        self.out.push(Instruction::IntConvert {
            from,
            to,
            signed,
            dst: dst.clone(),
            src: operand,
        });
        Ok(Operand::Register(dst))
    }

    /// The operand a non-pair instruction reads from this value: its low word when the
    /// value is a pair, otherwise the value itself.
    fn narrow(&mut self, value: &IRVal<'a>) -> Result<Operand, FromIRError> {
        if is_wide_value(value, self.vars, self.int64_regs) {
            return Ok(self.operand_words(value)?.lo);
        }
        resolve_operand(value, self.vregs, self.vars, self.globals)
    }

    /// A stack slot two words deep, as a register holding its address.
    fn pair_slot(&mut self) -> Register {
        let slot = self.vregs.allocate_gpr();
        self.reservations.insert(slot, 1);
        let address = self.temp();
        self.out.push(Instruction::StackAddr {
            dst: address.clone(),
            slot,
        });
        address
    }

    fn operand_words(&mut self, value: &IRVal<'a>) -> Result<Words, FromIRError> {
        match value {
            IRVal::Constant(literal) => {
                let (lo, hi) = literal_words(literal, self.int64_regs)?;
                Ok(Words {
                    lo: self.imm(lo),
                    hi: self.imm(hi),
                })
            }
            IRVal::Variable(name) => {
                let name: &'a str = name;
                if let Some([lo, hi]) = self.vars.wide.get(name).cloned() {
                    return Ok(Words {
                        lo: Operand::Register(lo),
                        hi: Operand::Register(hi),
                    });
                }
                if self.vars.contains_key(name) {
                    return Err(FromIRError::UnsupportedType);
                }
                let lo = Register::Virtual(self.vregs.allocate_gpr());
                let hi = Register::Virtual(self.vregs.allocate_gpr());
                self.vars.wide.insert(name, [lo.clone(), hi.clone()]);
                Ok(Words {
                    lo: Operand::Register(lo),
                    hi: Operand::Register(hi),
                })
            }
            IRVal::Global(_) => Err(FromIRError::UnsupportedType),
        }
    }

    fn result_pair(&mut self, name: &'a str) -> [Register; 2] {
        if let Some(pair) = self.vars.wide.get(name).cloned() {
            return pair;
        }
        let pair = [self.temp(), self.temp()];
        self.vars.wide.insert(name, pair.clone());
        pair
    }

    fn write_pair(&mut self, pair: [Register; 2], words: Words) {
        let ty = self.word();
        self.out.push(Instruction::Copy {
            ty,
            dst: pair[0].clone(),
            src: words.lo,
        });
        self.out.push(Instruction::Copy {
            ty,
            dst: pair[1].clone(),
            src: words.hi,
        });
    }

    fn division(&mut self, signed: bool, remainder: bool, a: Words, b: Words) -> Words {
        let joined = self.bin(IntBinOp::Or, b.lo.clone(), b.hi.clone());
        let nonzero_divisor = self.nonzero(joined);
        let divisor = if signed {
            let sign = self.imm(sign_bit(self.int64_regs));
            let sign_flipped = self.bin(IntBinOp::Xor, a.hi.clone(), sign);
            let a_min_hi = self.is_zero(sign_flipped);
            let a_min_lo = self.is_zero(a.lo.clone());
            let ones = self.imm(-1);
            let b_lo_neg = self.bin(IntBinOp::Xor, b.lo.clone(), ones.clone());
            let b_lo_is = self.is_zero(b_lo_neg);
            let b_hi_neg = self.bin(IntBinOp::Xor, b.hi.clone(), ones);
            let b_hi_is = self.is_zero(b_hi_neg);
            let left = self.bin(IntBinOp::And, a_min_hi, a_min_lo);
            let right = self.bin(IntBinOp::And, b_lo_is, b_hi_is);
            let overflow = self.bin(IntBinOp::And, left, right);
            let one = self.imm(1);
            let fine = self.bin(IntBinOp::Xor, overflow, one);
            self.bin(IntBinOp::And, nonzero_divisor, fine)
        } else {
            nonzero_divisor
        };
        let trap_op = if signed {
            IntBinOp::SDiv
        } else {
            IntBinOp::UDiv
        };
        let trap_dst = self.temp();
        let ty = self.word();
        let one = self.imm(1);
        self.out.push(Instruction::IntBinary {
            op: trap_op,
            ty,
            dst: trap_dst,
            lhs: one,
            rhs: divisor,
        });

        let symbol = match (self.int64_regs, signed, remainder) {
            (true, true, false) => RuntimeSymbol::Divti3,
            (true, false, false) => RuntimeSymbol::Udivti3,
            (true, true, true) => RuntimeSymbol::Modti3,
            (true, false, true) => RuntimeSymbol::Umodti3,
            (false, true, false) => RuntimeSymbol::Divdi3,
            (false, false, false) => RuntimeSymbol::Udivdi3,
            (false, true, true) => RuntimeSymbol::Moddi3,
            (false, false, true) => RuntimeSymbol::Umoddi3,
        };
        let name = self.libcall_name(symbol, vec![Abi::Wide, Abi::Wide], Some(Abi::Wide));
        let pa = self.spill(a);
        let pb = self.spill(b);
        self.call_wide_result(name, vec![pa, pb])
    }
}

/// The value of the word's sign bit, as the word-width immediate that carries it.
const fn sign_bit(int64_regs: bool) -> i64 {
    if int64_regs {
        i64::MIN
    } else {
        i32::MIN as i64
    }
}

/// Folds an operation on two literal words at the word's width, with the same wrapping
/// and masking the backends use. Operations that can trap are left to run.
fn fold(op: IntBinOp, a: i64, b: i64, int64_regs: bool) -> Option<i64> {
    if !int64_regs {
        let (a, b) = (a as i32, b as i32);
        let count = b as u32 & 31;
        return Some(i64::from(match op {
            IntBinOp::Add => a.wrapping_add(b),
            IntBinOp::Sub => a.wrapping_sub(b),
            IntBinOp::Mul => a.wrapping_mul(b),
            IntBinOp::And => a & b,
            IntBinOp::Or => a | b,
            IntBinOp::Xor => a ^ b,
            IntBinOp::Shl => ((a as u32) << count) as i32,
            IntBinOp::LShr => ((a as u32) >> count) as i32,
            IntBinOp::AShr => a >> count,
            IntBinOp::UDiv | IntBinOp::SDiv | IntBinOp::URem | IntBinOp::SRem => return None,
        }));
    }
    let count = (b as u64 & 63) as u32;
    Some(match op {
        IntBinOp::Add => a.wrapping_add(b),
        IntBinOp::Sub => a.wrapping_sub(b),
        IntBinOp::Mul => a.wrapping_mul(b),
        IntBinOp::And => a & b,
        IntBinOp::Or => a | b,
        IntBinOp::Xor => a ^ b,
        IntBinOp::Shl => ((a as u64) << count) as i64,
        IntBinOp::LShr => ((a as u64) >> count) as i64,
        IntBinOp::AShr => a >> count,
        IntBinOp::UDiv | IntBinOp::SDiv | IntBinOp::URem | IntBinOp::SRem => return None,
    })
}

fn literal_words(literal: &Literal<'_>, int64_regs: bool) -> Result<(i64, i64), FromIRError> {
    if !int64_regs {
        let value: i64 = match literal {
            Literal::I8(v) => i64::from(*v),
            Literal::I16(v) => i64::from(*v),
            Literal::I32(v) => i64::from(*v),
            Literal::I64(v) => *v,
            Literal::U8(v) => i64::from(*v),
            Literal::U16(v) => i64::from(*v),
            Literal::U32(v) => i64::from(*v),
            Literal::U64(v) => *v as i64,
            Literal::Bool(v) => i64::from(*v),
            Literal::I128(_)
            | Literal::U128(_)
            | Literal::F32(_)
            | Literal::F64(_)
            | Literal::String(_) => return Err(FromIRError::UnsupportedType),
        };
        let value = value as u64;
        return Ok((
            i64::from(value as u32 as i32),
            i64::from((value >> 32) as u32 as i32),
        ));
    }
    let wide: i128 = match literal {
        Literal::I128(value) => return Ok(split(*value as u128)),
        Literal::U128(value) => return Ok(split(*value)),
        Literal::I8(v) => i128::from(*v),
        Literal::I16(v) => i128::from(*v),
        Literal::I32(v) => i128::from(*v),
        Literal::I64(v) => i128::from(*v),
        Literal::U8(v) => i128::from(*v),
        Literal::U16(v) => i128::from(*v),
        Literal::U32(v) => i128::from(*v),
        Literal::U64(v) => i128::from(*v),
        Literal::F32(_) | Literal::F64(_) | Literal::Bool(_) | Literal::String(_) => {
            return Err(FromIRError::UnsupportedType);
        }
    };
    Ok(split(wide as u128))
}

fn split(value: u128) -> (i64, i64) {
    (value as u64 as i64, (value >> 64) as u64 as i64)
}

/// Whether a value already stands for a pair. A literal is only self-evidently wide at
/// 128 bits; at 64 bits on a 32-bit target the literal's own type says nothing, so the
/// callee signature decides and a bare literal stays scalar here.
fn is_wide_value(value: &IRVal<'_>, vars: &VarBindings<'_>, int64_regs: bool) -> bool {
    match value {
        IRVal::Variable(name) => vars.wide.contains_key(name),
        IRVal::Constant(literal) => {
            int64_regs && matches!(literal, Literal::I128(_) | Literal::U128(_))
        }
        IRVal::Global(_) => false,
    }
}

/// Whether the instruction names a 128-bit type, or reads a 128-bit value. Every variant
/// is listed, so a new IR instruction forces a decision here.
fn touches_wide(
    instr: &IRInst<'_>,
    vars: &VarBindings<'_>,
    module: &IRModule<'_>,
    int64_regs: bool,
) -> bool {
    let typed = match instr {
        IRInst::Binary { ty, .. }
        | IRInst::OverflowFlag { ty, .. }
        | IRInst::Unary { ty, .. }
        | IRInst::Cmp { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::ZeroExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::SignExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::Trunc {
            source_type,
            target_type,
            ..
        }
        | IRInst::Bitcast {
            source_type,
            target_type,
            ..
        }
        | IRInst::FloatConvert {
            source_type,
            target_type,
            ..
        } => is_wide_prim(*source_type, int64_regs) || is_wide_prim(*target_type, int64_regs),
        IRInst::IntToFloat { source_type, .. } => is_wide_prim(*source_type, int64_regs),
        IRInst::FloatToInt { target_type, .. }
        | IRInst::PtrToInt { target_type, .. }
        | IRInst::IntToPtr { target_type, .. } => is_wide_prim(*target_type, int64_regs),
        IRInst::Switch { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::Select { ty, .. }
        | IRInst::Ret { ty, .. }
        | IRInst::Copy { ty, .. }
        | IRInst::Load { ty, .. }
        | IRInst::Store { ty, .. }
        | IRInst::AtomicLoad { ty, .. }
        | IRInst::AtomicStore { ty, .. }
        | IRInst::AtomicBinary { ty, .. }
        | IRInst::AtomicCompareExchange { ty, .. } => is_wide_type(ty, int64_regs),
        IRInst::VaArg { ty, .. } => is_wide_prim(*ty, int64_regs),
        IRInst::InlineAsm {
            result, operands, ..
        } => {
            result.as_ref().is_some_and(|(_, ty)| is_wide_prim(*ty, int64_regs))
                || operands.iter().any(|operand| {
                    matches!(operand, IRAsmOperand::In { value, .. } if is_wide_value(value, vars, int64_regs))
                })
        }
        IRInst::CallIndirect {
            args, return_type, ..
        } => is_wide_type(return_type, int64_regs) || args.iter().any(|(ty, _)| is_wide_type(ty, int64_regs)),
        IRInst::Call {
            func_name, args, ..
        } => {
            module.functions.get(func_name).is_some_and(|callee| {
                is_wide_type(&callee.signature.return_type, int64_regs)
                    || callee.signature.params.iter().any(|p| is_wide_type(&p.ty, int64_regs))
            }) || args.iter().any(|(_, arg)| is_wide_value(arg, vars, int64_regs))
        }
        IRInst::Write { .. }
        | IRInst::Read { .. }
        | IRInst::WriteByte { .. }
        | IRInst::ReadByte { .. } => !int64_regs,
        IRInst::Alloc { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::Fence { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::GetElemPtr { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. } => false,
        #[cfg(feature = "nightly")]
        IRInst::SimdExtract { scalar_type, .. } => is_wide_prim(*scalar_type, int64_regs),
        #[cfg(feature = "nightly")]
        IRInst::SimdBinary { .. }
        | IRInst::SimdUnary { .. }
        | IRInst::SimdTernary { .. }
        | IRInst::SimdShuffle { .. }
        | IRInst::SimdInsert { .. }
        | IRInst::SimdLoad { .. }
        | IRInst::SimdStore { .. } => false,
    };
    typed
        || instr
            .used_vars()
            .iter()
            .any(|name| vars.wide.contains_key(name))
}

/// Whether the instruction names a 128-bit type on a target whose registers are 32 bits
/// wide. Four-word integers are not lowered, so the compile stops here by name.
fn quad_mention(instr: &IRInst<'_>, module: &IRModule<'_>, int64_regs: bool) -> bool {
    if int64_regs {
        return false;
    }
    match instr {
        IRInst::Binary { ty, .. }
        | IRInst::Unary { ty, .. }
        | IRInst::Cmp { ty, .. }
        | IRInst::Switch { ty, .. }
        | IRInst::VaArg { ty, .. } => is_quad_prim(*ty, int64_regs),
        IRInst::ZeroExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::SignExtend {
            source_type,
            target_type,
            ..
        }
        | IRInst::Trunc {
            source_type,
            target_type,
            ..
        }
        | IRInst::Bitcast {
            source_type,
            target_type,
            ..
        }
        | IRInst::FloatConvert {
            source_type,
            target_type,
            ..
        }
        | IRInst::IntToFloat {
            source_type,
            target_type,
            ..
        }
        | IRInst::FloatToInt {
            source_type,
            target_type,
            ..
        } => is_quad_prim(*source_type, int64_regs) || is_quad_prim(*target_type, int64_regs),
        IRInst::PtrToInt { target_type, .. } | IRInst::IntToPtr { target_type, .. } => {
            is_quad_prim(*target_type, int64_regs)
        }
        IRInst::Select { ty, .. }
        | IRInst::Ret { ty, .. }
        | IRInst::Copy { ty, .. }
        | IRInst::Load { ty, .. }
        | IRInst::Store { ty, .. }
        | IRInst::Alloc {
            allocated_ty: ty, ..
        } => is_quad_type(ty, int64_regs),
        IRInst::Call { func_name, .. } => module.functions.get(func_name).is_some_and(|callee| {
            is_quad_type(&callee.signature.return_type, int64_regs)
                || callee
                    .signature
                    .params
                    .iter()
                    .any(|p| is_quad_type(&p.ty, int64_regs))
        }),
        IRInst::CallIndirect {
            args, return_type, ..
        } => {
            is_quad_type(return_type, int64_regs)
                || args.iter().any(|(ty, _)| is_quad_type(ty, int64_regs))
        }
        _ => false,
    }
}

pub(super) fn lower_instruction<'a>(
    instr: &IRInst<'a>,
    cx: &mut Cx<'_, 'a>,
) -> Result<bool, FromIRError> {
    if quad_mention(instr, cx.module, cx.int64_regs) {
        return Err(FromIRError::NotLegalizable(format!(
            "`{instr}` names a 128-bit integer on a 32-bit target: that takes four registers, and only pairs are split"
        )));
    }
    if !touches_wide(instr, cx.vars, cx.module, cx.int64_regs) {
        return Ok(false);
    }
    match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let signed = matches!(*ty, IRPrim::I128 | IRPrim::I64);
            let a = cx.operand_words(lhs)?;
            let b = cx.operand_words(rhs)?;
            let value = match op {
                IRBin::Add => cx.add(a, b),
                IRBin::Sub => cx.sub(a, b),
                IRBin::Mul => cx.mul(a, b),
                IRBin::And => Words {
                    lo: cx.bin(IntBinOp::And, a.lo, b.lo),
                    hi: cx.bin(IntBinOp::And, a.hi, b.hi),
                },
                IRBin::Or => Words {
                    lo: cx.bin(IntBinOp::Or, a.lo, b.lo),
                    hi: cx.bin(IntBinOp::Or, a.hi, b.hi),
                },
                IRBin::Xor => Words {
                    lo: cx.bin(IntBinOp::Xor, a.lo, b.lo),
                    hi: cx.bin(IntBinOp::Xor, a.hi, b.hi),
                },
                IRBin::Shl => cx.shl(a, b.lo),
                IRBin::Shr => cx.right_shift(a, b.lo, signed),
                IRBin::Div => cx.division(signed, false, a, b),
                IRBin::Rem => cx.division(signed, true, a, b),
                IRBin::Rotl => cx.rotate(a, b.lo, true),
                IRBin::Rotr => cx.rotate(a, b.lo, false),
            };
            let pair = cx.result_pair(result);
            cx.write_pair(pair, value);
        }
        IRInst::Unary {
            op,
            result,
            ty,
            value,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let a = cx.operand_words(value)?;
            let value = match op {
                UnaryOp::Neg => {
                    let zero = Words {
                        lo: cx.imm(0),
                        hi: cx.imm(0),
                    };
                    cx.sub(zero, a)
                }
                UnaryOp::Not => Words {
                    lo: cx.not(a.lo),
                    hi: cx.not(a.hi),
                },
                UnaryOp::Popcount => cx.popcount(a),
                UnaryOp::Ctlz => cx.leading_zeros(a),
                UnaryOp::Cttz => cx.trailing_zeros(a),
                UnaryOp::Bswap => Words {
                    lo: cx.unary_word(IntUnOp::Bswap, a.hi),
                    hi: cx.unary_word(IntUnOp::Bswap, a.lo),
                },
                UnaryOp::Sqrt => return Err(FromIRError::UnsupportedInstruction),
            };
            let pair = cx.result_pair(result);
            cx.write_pair(pair, value);
        }
        IRInst::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } if is_wide_prim(*ty, cx.int64_regs) => {
            let signed = matches!(*ty, IRPrim::I128 | IRPrim::I64);
            let a = cx.operand_words(lhs)?;
            let b = cx.operand_words(rhs)?;
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            let flag = match op {
                IRCmp::Eq | IRCmp::Ne => {
                    let lo = cx.bin(IntBinOp::Xor, a.lo, b.lo);
                    let hi = cx.bin(IntBinOp::Xor, a.hi, b.hi);
                    let differ = cx.bin(IntBinOp::Or, lo, hi);
                    let op = if matches!(op, IRCmp::Eq) {
                        IntCmpOp::Eq
                    } else {
                        IntCmpOp::Ne
                    };
                    let ty = cx.word();
                    let zero = cx.imm(0);
                    cx.out.push(Instruction::IntCmp {
                        op,
                        ty,
                        dst,
                        lhs: differ,
                        rhs: zero,
                    });
                    return Ok(true);
                }
                IRCmp::Lt | IRCmp::Le | IRCmp::Gt | IRCmp::Ge => {
                    let (x, y) = if signed {
                        (cx.flip_sign(a), cx.flip_sign(b))
                    } else {
                        (a, b)
                    };
                    match op {
                        IRCmp::Lt => cx.unsigned_less(x, y),
                        IRCmp::Gt => cx.unsigned_less(y, x),
                        IRCmp::Le => {
                            let greater = cx.unsigned_less(y, x);
                            let one = cx.imm(1);
                            cx.bin(IntBinOp::Xor, greater, one)
                        }
                        IRCmp::Ge => {
                            let less = cx.unsigned_less(x, y);
                            let one = cx.imm(1);
                            cx.bin(IntBinOp::Xor, less, one)
                        }
                        IRCmp::Eq | IRCmp::Ne => return Err(FromIRError::UnsupportedInstruction),
                    }
                }
            };
            let ty = cx.word();
            let zero = cx.imm(0);
            cx.out.push(Instruction::IntCmp {
                op: IntCmpOp::Ne,
                ty,
                dst,
                lhs: flag,
                rhs: zero,
            });
        }
        IRInst::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs)
            && !is_wide_prim(*source_type, cx.int64_regs) =>
        {
            extend_to_wide(cx, result, *source_type, value, false)?;
        }
        IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs)
            && !is_wide_prim(*source_type, cx.int64_regs) =>
        {
            extend_to_wide(cx, result, *source_type, value, true)?;
        }
        IRInst::Trunc {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*source_type, cx.int64_regs)
            && !is_wide_prim(*target_type, cx.int64_regs) =>
        {
            let a = cx.operand_words(value)?;
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            let word = cx.word();
            let to = map_ir_prim(*target_type)?;
            if to == word {
                cx.out.push(Instruction::Copy {
                    ty: word,
                    dst,
                    src: a.lo,
                });
            } else {
                cx.out.push(Instruction::IntConvert {
                    from: word,
                    to,
                    signed: false,
                    dst,
                    src: a.lo,
                });
            }
        }
        IRInst::Select {
            result,
            cond,
            true_val,
            false_val,
            ..
        } => {
            let cond_reg = match cond {
                IRVal::Variable(id) => cx
                    .vars
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                IRVal::Constant(_) | IRVal::Global(_) => {
                    return Err(FromIRError::UnsupportedInstruction);
                }
            };
            let t = cx.operand_words(true_val)?;
            let f = cx.operand_words(false_val)?;
            let pair = cx.result_pair(result);
            let ty = cx.word();
            cx.out.push(Instruction::Select {
                ty,
                dst: pair[0].clone(),
                cond: cond_reg.clone(),
                true_val: t.lo,
                false_val: f.lo,
            });
            cx.out.push(Instruction::Select {
                ty,
                dst: pair[1].clone(),
                cond: cond_reg,
                true_val: t.hi,
                false_val: f.hi,
            });
        }
        IRInst::Load {
            result,
            ty,
            ptr,
            volatile: _,
            align,
        } if is_wide_type(ty, cx.int64_regs) => {
            reject_wide_alignment(*align, ty, "load")?;
            let base = address_base(ptr, cx)?;
            let loaded = cx.load_words(base);
            let pair = cx.result_pair(result);
            cx.write_pair(pair, loaded);
        }
        IRInst::Store {
            ty,
            ptr,
            value,
            volatile: _,
            align,
        } if is_wide_type(ty, cx.int64_regs) => {
            reject_wide_alignment(*align, ty, "store")?;
            let base = address_base(ptr, cx)?;
            let words = cx.operand_words(value)?;
            cx.store_words(base, words);
        }
        IRInst::Copy { result, ty, value } if is_wide_type(ty, cx.int64_regs) => {
            let words = cx.operand_words(value)?;
            let pair = cx.result_pair(result);
            cx.write_pair(pair, words);
        }
        IRInst::Ret { ty, value } if is_wide_type(ty, cx.int64_regs) => {
            let Some(value) = value else {
                return Err(FromIRError::InvalidIR);
            };
            let words = cx.operand_words(value)?;
            let address = cx.spill(words);
            cx.out.push(Instruction::Ret {
                value: Some(address),
            });
        }
        IRInst::Call {
            result,
            func_name,
            args,
        } => {
            let callee = cx.module.functions.get(func_name);
            let mut mir_args = Vec::with_capacity(args.len());
            for (index, (_, arg)) in args.iter().enumerate() {
                let param_wide = callee
                    .and_then(|f| f.signature.params.get(index))
                    .is_some_and(|p| is_wide_type(&p.ty, cx.int64_regs));
                if param_wide || is_wide_value(arg, cx.vars, cx.int64_regs) {
                    let words = cx.operand_words(arg)?;
                    mir_args.push(cx.spill(words));
                } else {
                    mir_args.push(resolve_operand(arg, cx.vregs, cx.vars, cx.globals)?);
                }
            }
            let ret_wide =
                callee.is_some_and(|f| is_wide_type(&f.signature.return_type, cx.int64_regs));
            if ret_wide {
                let placeholder = cx.temp();
                cx.out.push(Instruction::Call {
                    name: (*func_name).to_string(),
                    args: mir_args,
                    ret: Some(placeholder.clone()),
                });
                if let Some(result) = result {
                    let loaded = cx.load_words(placeholder);
                    let pair = cx.result_pair(result);
                    cx.write_pair(pair, loaded);
                }
            } else {
                if callee.is_none() {
                    return Err(FromIRError::UnsupportedInstruction);
                }
                let ret = result.map(|r| resolve_or_alloc(r, cx.vregs, cx.vars));
                cx.out.push(Instruction::Call {
                    name: (*func_name).to_string(),
                    args: mir_args,
                    ret,
                });
            }
        }
        IRInst::CallIndirect { .. } => {
            return Err(FromIRError::UnsupportedAggregate(
                "a 128-bit argument or result of callptr: an indirect call carries no callee signature to pass a 16-byte aggregate by, so call the function directly".to_string(),
            ));
        }
        IRInst::IntToFloat {
            result,
            signed,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*source_type, cx.int64_regs) => {
            let words = cx.operand_words(value)?;
            let arg = cx.spill(words);
            let wide = cx.int64_regs;
            let (symbol, ret_ty) = match target_type {
                IRPrim::F64 if *signed && wide => {
                    (RuntimeSymbol::Floattidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 if wide => {
                    (RuntimeSymbol::Floatuntidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F32 if *signed && wide => {
                    (RuntimeSymbol::Floattisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 if wide => {
                    (RuntimeSymbol::Floatuntisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F64 if *signed => {
                    (RuntimeSymbol::Floatdidf, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 => (RuntimeSymbol::Floatundidf, MirType::Scalar(ScalarType::F64)),
                IRPrim::F32 if *signed => {
                    (RuntimeSymbol::Floatdisf, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 => (RuntimeSymbol::Floatundisf, MirType::Scalar(ScalarType::F32)),
                IRPrim::I8
                | IRPrim::I16
                | IRPrim::I32
                | IRPrim::I64
                | IRPrim::U8
                | IRPrim::U16
                | IRPrim::U32
                | IRPrim::U64
                | IRPrim::I128
                | IRPrim::U128
                | IRPrim::Bool
                | IRPrim::Ptr => return Err(FromIRError::UnsupportedInstruction),
            };
            let name = cx.libcall_name(symbol, vec![Abi::Wide], Some(Abi::Scalar(ret_ty)));
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::Call {
                name,
                args: vec![arg],
                ret: Some(dst),
            });
        }
        IRInst::FloatToInt {
            result,
            signed,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let wide = cx.int64_regs;
            let (symbol, param_ty) = match source_type {
                IRPrim::F64 if *signed && wide => {
                    (RuntimeSymbol::Fixdfti, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 if wide => {
                    (RuntimeSymbol::Fixunsdfti, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F32 if *signed && wide => {
                    (RuntimeSymbol::Fixsfti, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 if wide => {
                    (RuntimeSymbol::Fixunssfti, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F64 if *signed => {
                    (RuntimeSymbol::Fixdfdi, MirType::Scalar(ScalarType::F64))
                }
                IRPrim::F64 => (RuntimeSymbol::Fixunsdfdi, MirType::Scalar(ScalarType::F64)),
                IRPrim::F32 if *signed => {
                    (RuntimeSymbol::Fixsfdi, MirType::Scalar(ScalarType::F32))
                }
                IRPrim::F32 => (RuntimeSymbol::Fixunssfdi, MirType::Scalar(ScalarType::F32)),
                IRPrim::I8
                | IRPrim::I16
                | IRPrim::I32
                | IRPrim::I64
                | IRPrim::U8
                | IRPrim::U16
                | IRPrim::U32
                | IRPrim::U64
                | IRPrim::I128
                | IRPrim::U128
                | IRPrim::Bool
                | IRPrim::Ptr => return Err(FromIRError::UnsupportedInstruction),
            };
            let arg = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
            let name = cx.libcall_name(symbol, vec![Abi::Scalar(param_ty)], Some(Abi::Wide));
            let words = cx.call_wide_result(name, vec![arg]);
            let pair = cx.result_pair(result);
            cx.write_pair(pair, words);
        }
        IRInst::Write {
            buffer,
            size,
            result,
        } => {
            let buffer = cx.narrow(buffer)?;
            let size = cx.narrow(size)?;
            io_call(cx, "lamina.write", vec![buffer, size], result);
        }
        IRInst::WriteByte { value, result } => {
            let value = cx.narrow(value)?;
            io_call(cx, "lamina.writebyte", vec![value], result);
        }
        IRInst::Read {
            buffer,
            size,
            result,
        } => {
            let buffer = cx.narrow(buffer)?;
            let size = cx.narrow(size)?;
            io_call(cx, "lamina.read", vec![buffer, size], result);
        }
        IRInst::ReadByte { result } => {
            io_call(cx, "lamina.readbyte", Vec::new(), result);
        }
        IRInst::Bitcast {
            result,
            source_type,
            target_type,
            value,
        } => {
            let from_pair = is_wide_prim(*source_type, cx.int64_regs);
            let to_pair = is_wide_prim(*target_type, cx.int64_regs);
            match (from_pair, to_pair) {
                (true, true) => {
                    let words = cx.operand_words(value)?;
                    let pair = cx.result_pair(result);
                    cx.write_pair(pair, words);
                }
                (false, true) => {
                    if let IRVal::Constant(literal) = value {
                        let (lo, hi) = float_literal_words(literal, cx.int64_regs)?;
                        let words = Words {
                            lo: cx.imm(lo),
                            hi: cx.imm(hi),
                        };
                        let pair = cx.result_pair(result);
                        cx.write_pair(pair, words);
                    } else {
                        let src = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
                        let ty = map_ir_prim(*source_type)?;
                        let address = cx.pair_slot();
                        cx.out.push(Instruction::Store {
                            ty,
                            src,
                            addr: AddressMode::BaseOffset {
                                base: address.clone(),
                                offset: 0,
                                address_space: Default::default(),
                            },
                            attrs: MemoryAttrs {
                                align: word_bytes(cx.int64_regs) as u8,
                                volatile: false,
                                stated_align: false,
                            },
                        });
                        let words = cx.load_words(address);
                        let pair = cx.result_pair(result);
                        cx.write_pair(pair, words);
                    }
                }
                (true, false) => {
                    let words = cx.operand_words(value)?;
                    let address = cx.pair_slot();
                    cx.store_words(address.clone(), words);
                    let ty = map_ir_prim(*target_type)?;
                    let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
                    cx.out.push(Instruction::Load {
                        ty,
                        dst,
                        addr: AddressMode::BaseOffset {
                            base: address,
                            offset: 0,
                            address_space: Default::default(),
                        },
                        attrs: MemoryAttrs {
                            align: word_bytes(cx.int64_regs) as u8,
                            volatile: false,
                            stated_align: false,
                        },
                    });
                }
                (false, false) => return Err(FromIRError::UnsupportedInstruction),
            }
        }
        IRInst::PtrToInt {
            result,
            ptr_value,
            target_type,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let src = resolve_operand(ptr_value, cx.vregs, cx.vars, cx.globals)?;
            let lo = cx.temp();
            let to = cx.word();
            cx.out.push(Instruction::IntConvert {
                from: MirType::Scalar(ScalarType::Ptr),
                to,
                signed: false,
                dst: lo.clone(),
                src,
            });
            let pair = cx.result_pair(result);
            let words = Words {
                lo: Operand::Register(lo),
                hi: cx.imm(0),
            };
            cx.write_pair(pair, words);
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type,
        } if is_wide_prim(*target_type, cx.int64_regs) => {
            let src = cx.narrow(int_value)?;
            let from = cx.word();
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::IntConvert {
                from,
                to: MirType::Scalar(ScalarType::Ptr),
                signed: false,
                dst,
                src,
            });
        }
        IRInst::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        } => {
            let base = address_base(array_ptr, cx)?;
            let size = sizeof_type(element_type, cx.pointer)
                .and_then(|size| i64::try_from(size).ok())
                .ok_or(FromIRError::UnsupportedInstruction)?;
            let index = cx.narrow(index)?;
            let scale = cx.imm(size);
            let scaled = cx.bin(IntBinOp::Mul, index, scale);
            let address = cx.temp();
            let word = cx.word();
            cx.out.push(Instruction::IntConvert {
                from: MirType::Scalar(ScalarType::Ptr),
                to: word,
                signed: false,
                dst: address.clone(),
                src: Operand::Register(base),
            });
            let sum = cx.bin(IntBinOp::Add, Operand::Register(address), scaled);
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            cx.out.push(Instruction::IntConvert {
                from: word,
                to: MirType::Scalar(ScalarType::Ptr),
                signed: false,
                dst,
                src: sum,
            });
        }
        IRInst::Binary { .. }
        | IRInst::Unary { .. }
        | IRInst::Cmp { .. }
        | IRInst::ZeroExtend { .. }
        | IRInst::SignExtend { .. }
        | IRInst::Trunc { .. }
        | IRInst::Load { .. }
        | IRInst::Store { .. }
        | IRInst::Copy { .. }
        | IRInst::Ret { .. }
        | IRInst::IntToFloat { .. }
        | IRInst::FloatToInt { .. }
        | IRInst::FloatConvert { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Switch { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::Alloc { .. }
        | IRInst::Fence { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::PtrToInt { .. }
        | IRInst::IntToPtr { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. }
        | IRInst::InlineAsm { .. } => return Err(FromIRError::UnsupportedInstruction),
        IRInst::OverflowFlag { op, ty, .. } => {
            return Err(FromIRError::NotLegalizable(format!(
                "{op}.ovf.{ty} on a 32-bit target: the value is held in a register pair and no backend reports the overflow of the pair as a whole"
            )));
        }
        IRInst::VaArg { .. } => {
            return Err(FromIRError::NotLegalizable(
                "va_arg of a 64-bit integer on a 32-bit target: a variadic slot holds one word, and reading the pair is not implemented".to_string(),
            ));
        }
        IRInst::AtomicLoad { .. }
        | IRInst::AtomicStore { .. }
        | IRInst::AtomicBinary { .. }
        | IRInst::AtomicCompareExchange { .. } => {
            return Err(FromIRError::NotLegalizable(
                "an atomic operation on a 64-bit integer on a 32-bit target: two words cannot be read or written atomically without a double-word atomic".to_string(),
            ));
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdBinary { .. }
        | IRInst::SimdUnary { .. }
        | IRInst::SimdTernary { .. }
        | IRInst::SimdShuffle { .. }
        | IRInst::SimdExtract { .. }
        | IRInst::SimdInsert { .. }
        | IRInst::SimdLoad { .. }
        | IRInst::SimdStore { .. } => return Err(FromIRError::UnsupportedInstruction),
    }
    Ok(true)
}

/// An io intrinsic returns a signed count typed `i64`, so on a 32-bit target it defines
/// a pair: the call's own word, and that word's sign.
fn io_call<'a>(cx: &mut Cx<'_, 'a>, name: &str, args: Vec<Operand>, result: &'a str) {
    let pair = cx.result_pair(result);
    cx.out.push(Instruction::Call {
        name: name.to_string(),
        args,
        ret: Some(pair[0].clone()),
    });
    let top = cx.imm(cx.bits() - 1);
    let hi = cx.bin(IntBinOp::AShr, Operand::Register(pair[0].clone()), top);
    let ty = cx.word();
    cx.out.push(Instruction::Copy {
        ty,
        dst: pair[1].clone(),
        src: hi,
    });
}

/// The words of a float literal reinterpreted as an integer of the same width.
fn float_literal_words(literal: &Literal<'_>, int64_regs: bool) -> Result<(i64, i64), FromIRError> {
    let bits = match literal {
        Literal::F64(value) => value.to_bits() as i64,
        Literal::F32(value) => i64::from(value.to_bits()),
        other => return literal_words(other, int64_regs),
    };
    literal_words(&Literal::I64(bits), int64_regs)
}

fn extend_to_wide<'a>(
    cx: &mut Cx<'_, 'a>,
    result: &'a str,
    source: IRPrim,
    value: &IRVal<'a>,
    signed: bool,
) -> Result<(), FromIRError> {
    let src = resolve_operand(value, cx.vregs, cx.vars, cx.globals)?;
    let word = cx.word_of(source, src, signed)?;
    let hi = if signed {
        let top = cx.imm(cx.bits() - 1);
        cx.bin(IntBinOp::AShr, word.clone(), top)
    } else {
        cx.imm(0)
    };
    let pair = cx.result_pair(result);
    cx.write_pair(pair, Words { lo: word, hi });
    Ok(())
}

fn address_base<'a>(ptr: &IRVal<'a>, cx: &mut Cx<'_, 'a>) -> Result<Register, FromIRError> {
    match ptr {
        IRVal::Variable(id) => cx.vars.get(id).cloned().ok_or(FromIRError::UnknownVariable),
        IRVal::Global(name) => cx.globals.register(name, cx.vregs),
        IRVal::Constant(_) => Err(FromIRError::UnsupportedInstruction),
    }
}

pub(super) fn load_wide_parameter(
    address: Register,
    vregs: &mut VirtualRegAllocator,
    prelude: &mut Vec<Instruction>,
    int64_regs: bool,
    endianness: Endianness,
) -> [Register; 2] {
    let word = word_ty(int64_regs);
    let attrs = MemoryAttrs {
        align: word_bytes(int64_regs) as u8,
        volatile: false,
        stated_align: false,
    };
    let (lo_at, hi_at) = word_offsets(int64_regs, endianness);
    let mut read = |offset: i32| {
        let dst = Register::Virtual(vregs.allocate_gpr());
        let base = if offset == 0 {
            address.clone()
        } else {
            let stepped = Register::Virtual(vregs.allocate_gpr());
            prelude.push(Instruction::Lea {
                dst: stepped.clone(),
                base: address.clone(),
                offset,
            });
            stepped
        };
        prelude.push(Instruction::Load {
            ty: word,
            dst: dst.clone(),
            addr: AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
        dst
    };
    let lo = read(lo_at);
    let hi = read(hi_at);
    [lo, hi]
}
