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
use crate::ir::instruction::UnaryOp;
use crate::ir::types::Literal;
use crate::mir::codegen::mapping::wide_aggregate;
use lamina_codegen::abi::BuiltinLibrary;
use lamina_platform::TargetOperatingSystem;

pub(super) fn is_wide_prim(ty: IRPrim) -> bool {
    matches!(ty, IRPrim::I128 | IRPrim::U128)
}

pub(super) fn is_wide_type(ty: &IRType<'_>) -> bool {
    matches!(ty, IRType::Primitive(p) if is_wide_prim(*p))
}

fn imm(value: i64) -> Operand {
    Operand::Immediate(Immediate::I64(value))
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
    pub(super) fn declaration(&self) -> Function {
        let mut sig = Signature::new(self.name.clone());
        for (index, param) in self.params.iter().enumerate() {
            let slot = u32::try_from(index).unwrap_or(u32::MAX);
            let parameter = match param {
                Abi::Wide => {
                    Parameter::new(Register::Virtual(VirtualReg::gpr(slot)), MirType::ptr())
                        .with_aggregate(wide_aggregate())
                }
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
                    .with_return(MirType::ptr())
                    .with_return_aggregate(wide_aggregate());
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
    pub(super) out: Vec<Instruction>,
}

impl<'c, 'a> Cx<'c, 'a> {
    fn temp(&mut self) -> Register {
        Register::Virtual(self.vregs.allocate_gpr())
    }

    fn bin(&mut self, op: IntBinOp, lhs: Operand, rhs: Operand) -> Operand {
        if let (Operand::Immediate(Immediate::I64(a)), Operand::Immediate(Immediate::I64(b))) =
            (&lhs, &rhs)
            && let Some(value) = fold(op, *a, *b)
        {
            return imm(value);
        }
        let dst = self.temp();
        self.out.push(Instruction::IntBinary {
            op,
            ty: MirType::i64(),
            dst: dst.clone(),
            lhs,
            rhs,
        });
        Operand::Register(dst)
    }

    fn not(&mut self, value: Operand) -> Operand {
        self.bin(IntBinOp::Xor, value, imm(-1))
    }

    fn nonzero(&mut self, value: Operand) -> Operand {
        let negated = self.bin(IntBinOp::Sub, imm(0), value.clone());
        let any = self.bin(IntBinOp::Or, value, negated);
        self.bin(IntBinOp::LShr, any, imm(63))
    }

    fn is_zero(&mut self, value: Operand) -> Operand {
        let nonzero = self.nonzero(value);
        self.bin(IntBinOp::Xor, nonzero, imm(1))
    }

    fn carry_out(&mut self, x: Operand, y: Operand, sum: Operand) -> Operand {
        let both = self.bin(IntBinOp::And, x.clone(), y.clone());
        let either = self.bin(IntBinOp::Or, x, y);
        let not_sum = self.not(sum);
        let propagate = self.bin(IntBinOp::And, either, not_sum);
        let joined = self.bin(IntBinOp::Or, both, propagate);
        self.bin(IntBinOp::LShr, joined, imm(63))
    }

    fn borrow_out(&mut self, x: Operand, y: Operand, diff: Operand) -> Operand {
        let not_x = self.not(x.clone());
        let lent = self.bin(IntBinOp::And, not_x, y.clone());
        let differ = self.bin(IntBinOp::Xor, x, y);
        let same = self.not(differ);
        let propagate = self.bin(IntBinOp::And, same, diff);
        let joined = self.bin(IntBinOp::Or, lent, propagate);
        self.bin(IntBinOp::LShr, joined, imm(63))
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
        let mask = imm(0xffff_ffff);
        let a0 = self.bin(IntBinOp::And, a.clone(), mask.clone());
        let a1 = self.bin(IntBinOp::LShr, a, imm(32));
        let b0 = self.bin(IntBinOp::And, b.clone(), mask.clone());
        let b1 = self.bin(IntBinOp::LShr, b, imm(32));
        let w0 = self.bin(IntBinOp::Mul, a0.clone(), b0.clone());
        let w0_high = self.bin(IntBinOp::LShr, w0, imm(32));
        let a1b0 = self.bin(IntBinOp::Mul, a1.clone(), b0);
        let t = self.bin(IntBinOp::Add, a1b0, w0_high);
        let t_low = self.bin(IntBinOp::And, t.clone(), mask);
        let t_high = self.bin(IntBinOp::LShr, t, imm(32));
        let a0b1 = self.bin(IntBinOp::Mul, a0, b1.clone());
        let w1 = self.bin(IntBinOp::Add, t_low, a0b1);
        let w1_high = self.bin(IntBinOp::LShr, w1, imm(32));
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
        let count = self.bin(IntBinOp::And, count, imm(127));
        let small = self.bin(IntBinOp::And, count.clone(), imm(63));
        let big = self.bin(IntBinOp::LShr, count, imm(6));
        let mask = self.bin(IntBinOp::Sub, imm(0), big);
        (small, mask)
    }

    /// Bits of the low word that move up into the high word for a left shift below 64.
    /// Written as two shifts, so a count of zero contributes nothing and no shift
    /// reaches 64.
    fn cross_shift_left(&mut self, low: Operand, count: Operand) -> Operand {
        let halved = self.bin(IntBinOp::LShr, low, imm(1));
        let rest = self.bin(IntBinOp::Xor, count, imm(63));
        self.bin(IntBinOp::LShr, halved, rest)
    }

    /// Bits of the high word that move down into the low word for a right shift below
    /// 64, with the same two-shift form.
    fn cross_shift_right(&mut self, high: Operand, count: Operand) -> Operand {
        let doubled = self.bin(IntBinOp::Shl, high, imm(1));
        let rest = self.bin(IntBinOp::Xor, count, imm(63));
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
            self.bin(IntBinOp::AShr, a.hi, imm(63))
        } else {
            imm(0)
        };
        let lo = self.select_mask(mask.clone(), high_shifted.clone(), low_small);
        let hi_small = self.bin(IntBinOp::And, high_shifted, not_mask);
        let hi_fill = self.bin(IntBinOp::And, fill, mask);
        let hi = self.bin(IntBinOp::Or, hi_small, hi_fill);
        Words { lo, hi }
    }

    fn unary_word(&mut self, op: IntUnOp, src: Operand) -> Operand {
        let dst = self.temp();
        self.out.push(Instruction::IntUnary {
            op,
            ty: MirType::i64(),
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
            hi: imm(0),
        }
    }

    /// The high word's count, or 64 plus the low word's when the high word is zero.
    fn leading_zeros(&mut self, a: Words) -> Words {
        let high = self.unary_word(IntUnOp::Ctlz, a.hi.clone());
        let low = self.unary_word(IntUnOp::Ctlz, a.lo.clone());
        let mask = self.is_zero(a.hi);
        let mask = self.bin(IntBinOp::Sub, imm(0), mask);
        let low = self.bin(IntBinOp::And, low, mask);
        Words {
            lo: self.bin(IntBinOp::Add, high, low),
            hi: imm(0),
        }
    }

    /// The low word's count, or 64 plus the high word's when the low word is zero.
    fn trailing_zeros(&mut self, a: Words) -> Words {
        let low = self.unary_word(IntUnOp::Cttz, a.lo.clone());
        let high = self.unary_word(IntUnOp::Cttz, a.hi.clone());
        let mask = self.is_zero(a.lo);
        let mask = self.bin(IntBinOp::Sub, imm(0), mask);
        let high = self.bin(IntBinOp::And, high, mask);
        Words {
            lo: self.bin(IntBinOp::Add, low, high),
            hi: imm(0),
        }
    }

    /// Rotation by `count` modulo 128. The shifts mask their count, so the opposite
    /// shift takes `-count`, and a zero count leaves the value unchanged.
    fn rotate(&mut self, a: Words, count: Operand, left: bool) -> Words {
        let opposite = self.bin(IntBinOp::Sub, imm(0), count.clone());
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
        Words {
            lo: a.lo,
            hi: self.bin(IntBinOp::Xor, a.hi, imm(i64::MIN)),
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
            align: 8,
            volatile: false,
        };
        self.out.push(Instruction::Store {
            ty: MirType::i64(),
            src: words.lo,
            addr: AddressMode::BaseOffset {
                base: base.clone(),
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
        let high = self.word_address(&base, 8);
        self.out.push(Instruction::Store {
            ty: MirType::i64(),
            src: words.hi,
            addr: AddressMode::BaseOffset {
                base: high,
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
    }

    fn load_words(&mut self, base: Register) -> Words {
        let attrs = MemoryAttrs {
            align: 8,
            volatile: false,
        };
        let lo = self.temp();
        self.out.push(Instruction::Load {
            ty: MirType::i64(),
            dst: lo.clone(),
            addr: AddressMode::BaseOffset {
                base: base.clone(),
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
        let high = self.word_address(&base, 8);
        let hi = self.temp();
        self.out.push(Instruction::Load {
            ty: MirType::i64(),
            dst: hi.clone(),
            addr: AddressMode::BaseOffset {
                base: high,
                offset: 0,
                address_space: Default::default(),
            },
            attrs,
        });
        Words {
            lo: Operand::Register(lo),
            hi: Operand::Register(hi),
        }
    }

    fn spill(&mut self, words: Words) -> Operand {
        let slot = self.vregs.allocate_gpr();
        self.reservations.insert(slot, 1);
        let address = self.temp();
        self.out.push(Instruction::StackAddr {
            dst: address.clone(),
            slot,
        });
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
        if from == MirType::i64() {
            return Ok(operand);
        }
        let dst = self.temp();
        self.out.push(Instruction::IntConvert {
            from,
            to: MirType::i64(),
            signed,
            dst: dst.clone(),
            src: operand,
        });
        Ok(Operand::Register(dst))
    }

    fn operand_words(&mut self, value: &IRVal<'a>) -> Result<Words, FromIRError> {
        match value {
            IRVal::Constant(literal) => {
                let (lo, hi) = literal_words(literal)?;
                Ok(Words {
                    lo: imm(lo),
                    hi: imm(hi),
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
        self.out.push(Instruction::Copy {
            ty: MirType::i64(),
            dst: pair[0].clone(),
            src: words.lo,
        });
        self.out.push(Instruction::Copy {
            ty: MirType::i64(),
            dst: pair[1].clone(),
            src: words.hi,
        });
    }

    fn division(&mut self, signed: bool, remainder: bool, a: Words, b: Words) -> Words {
        let joined = self.bin(IntBinOp::Or, b.lo.clone(), b.hi.clone());
        let nonzero_divisor = self.nonzero(joined);
        let divisor = if signed {
            let sign_flipped = self.bin(IntBinOp::Xor, a.hi.clone(), imm(i64::MIN));
            let a_min_hi = self.is_zero(sign_flipped);
            let a_min_lo = self.is_zero(a.lo.clone());
            let b_lo_neg = self.bin(IntBinOp::Xor, b.lo.clone(), imm(-1));
            let b_lo_is = self.is_zero(b_lo_neg);
            let b_hi_neg = self.bin(IntBinOp::Xor, b.hi.clone(), imm(-1));
            let b_hi_is = self.is_zero(b_hi_neg);
            let left = self.bin(IntBinOp::And, a_min_hi, a_min_lo);
            let right = self.bin(IntBinOp::And, b_lo_is, b_hi_is);
            let overflow = self.bin(IntBinOp::And, left, right);
            let fine = self.bin(IntBinOp::Xor, overflow, imm(1));
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
        self.out.push(Instruction::IntBinary {
            op: trap_op,
            ty: MirType::i64(),
            dst: trap_dst,
            lhs: imm(1),
            rhs: divisor,
        });

        let symbol = match (signed, remainder) {
            (true, false) => RuntimeSymbol::Divti3,
            (false, false) => RuntimeSymbol::Udivti3,
            (true, true) => RuntimeSymbol::Modti3,
            (false, true) => RuntimeSymbol::Umodti3,
        };
        let name = self.libcall_name(symbol, vec![Abi::Wide, Abi::Wide], Some(Abi::Wide));
        let pa = self.spill(a);
        let pb = self.spill(b);
        self.call_wide_result(name, vec![pa, pb])
    }
}

/// Folds a 64-bit operation on two literal words, with the same wrapping and masking
/// the backends use. Operations that can trap are left to run.
fn fold(op: IntBinOp, a: i64, b: i64) -> Option<i64> {
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

fn literal_words(literal: &Literal<'_>) -> Result<(i64, i64), FromIRError> {
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

fn is_wide_value(value: &IRVal<'_>, vars: &VarBindings<'_>) -> bool {
    match value {
        IRVal::Variable(name) => vars.wide.contains_key(name),
        IRVal::Constant(literal) => matches!(literal, Literal::I128(_) | Literal::U128(_)),
        IRVal::Global(_) => false,
    }
}

/// Whether the instruction names a 128-bit type, or reads a 128-bit value. Every variant
/// is listed, so a new IR instruction forces a decision here.
fn touches_wide(instr: &IRInst<'_>, vars: &VarBindings<'_>, module: &IRModule<'_>) -> bool {
    let typed = match instr {
        IRInst::Binary { ty, .. } | IRInst::Unary { ty, .. } | IRInst::Cmp { ty, .. } => {
            is_wide_prim(*ty)
        }
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
        } => is_wide_prim(*source_type) || is_wide_prim(*target_type),
        IRInst::IntToFloat { source_type, .. } => is_wide_prim(*source_type),
        IRInst::FloatToInt { target_type, .. }
        | IRInst::PtrToInt { target_type, .. }
        | IRInst::IntToPtr { target_type, .. } => is_wide_prim(*target_type),
        IRInst::Switch { ty, .. } => is_wide_prim(*ty),
        IRInst::Select { ty, .. }
        | IRInst::Ret { ty, .. }
        | IRInst::Phi { ty, .. }
        | IRInst::Load { ty, .. }
        | IRInst::Store { ty, .. }
        | IRInst::AtomicLoad { ty, .. }
        | IRInst::AtomicStore { ty, .. }
        | IRInst::AtomicBinary { ty, .. }
        | IRInst::AtomicCompareExchange { ty, .. } => is_wide_type(ty),
        IRInst::VaArg { ty, .. } => is_wide_prim(*ty),
        IRInst::CallIndirect {
            args, return_type, ..
        } => is_wide_type(return_type) || args.iter().any(|(ty, _)| is_wide_type(ty)),
        IRInst::Call {
            func_name, args, ..
        } => {
            module.functions.get(func_name).is_some_and(|callee| {
                is_wide_type(&callee.signature.return_type)
                    || callee.signature.params.iter().any(|p| is_wide_type(&p.ty))
            }) || args.iter().any(|arg| is_wide_value(arg, vars))
        }
        IRInst::Alloc { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::MemCpy { .. }
        | IRInst::MemMove { .. }
        | IRInst::MemSet { .. }
        | IRInst::Dealloc { .. }
        | IRInst::Fence { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::GetElemPtr { .. }
        | IRInst::Write { .. }
        | IRInst::Read { .. }
        | IRInst::WriteByte { .. }
        | IRInst::ReadByte { .. }
        | IRInst::Print { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. } => false,
        #[cfg(feature = "nightly")]
        IRInst::SimdExtract { scalar_type, .. } => is_wide_prim(*scalar_type),
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

pub(super) fn lower_instruction<'a>(
    instr: &IRInst<'a>,
    cx: &mut Cx<'_, 'a>,
) -> Result<bool, FromIRError> {
    if !touches_wide(instr, cx.vars, cx.module) {
        return Ok(false);
    }
    match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } if is_wide_prim(*ty) => {
            let signed = *ty == IRPrim::I128;
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
        } if is_wide_prim(*ty) => {
            let a = cx.operand_words(value)?;
            let value = match op {
                UnaryOp::Neg => {
                    let zero = Words {
                        lo: imm(0),
                        hi: imm(0),
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
        } if is_wide_prim(*ty) => {
            let signed = *ty == IRPrim::I128;
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
                    cx.out.push(Instruction::IntCmp {
                        op,
                        ty: MirType::i64(),
                        dst,
                        lhs: differ,
                        rhs: imm(0),
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
                            cx.bin(IntBinOp::Xor, greater, imm(1))
                        }
                        IRCmp::Ge => {
                            let less = cx.unsigned_less(x, y);
                            cx.bin(IntBinOp::Xor, less, imm(1))
                        }
                        IRCmp::Eq | IRCmp::Ne => return Err(FromIRError::UnsupportedInstruction),
                    }
                }
            };
            cx.out.push(Instruction::IntCmp {
                op: IntCmpOp::Ne,
                ty: MirType::i64(),
                dst,
                lhs: flag,
                rhs: imm(0),
            });
        }
        IRInst::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type) && !is_wide_prim(*source_type) => {
            extend_to_wide(cx, result, *source_type, value, false)?;
        }
        IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*target_type) && !is_wide_prim(*source_type) => {
            extend_to_wide(cx, result, *source_type, value, true)?;
        }
        IRInst::Trunc {
            result,
            source_type,
            target_type,
            value,
        } if is_wide_prim(*source_type) && !is_wide_prim(*target_type) => {
            let a = cx.operand_words(value)?;
            let dst = resolve_or_alloc(result, cx.vregs, cx.vars);
            if matches!(target_type, IRPrim::I64 | IRPrim::U64) {
                cx.out.push(Instruction::Copy {
                    ty: MirType::i64(),
                    dst,
                    src: a.lo,
                });
            } else {
                cx.out.push(Instruction::IntConvert {
                    from: MirType::i64(),
                    to: map_ir_prim(*target_type)?,
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
            cx.out.push(Instruction::Select {
                ty: MirType::i64(),
                dst: pair[0].clone(),
                cond: cond_reg.clone(),
                true_val: t.lo,
                false_val: f.lo,
            });
            cx.out.push(Instruction::Select {
                ty: MirType::i64(),
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
            volatile: false,
        } if is_wide_type(ty) => {
            let base = address_base(ptr, cx)?;
            let loaded = cx.load_words(base);
            let pair = cx.result_pair(result);
            cx.write_pair(pair, loaded);
        }
        IRInst::Store {
            ty,
            ptr,
            value,
            volatile: false,
        } if is_wide_type(ty) => {
            let base = address_base(ptr, cx)?;
            let words = cx.operand_words(value)?;
            cx.store_words(base, words);
        }
        IRInst::Phi { result, ty, .. } if is_wide_type(ty) => {
            cx.result_pair(result);
        }
        IRInst::Ret { ty, value } if is_wide_type(ty) => {
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
            for (index, arg) in args.iter().enumerate() {
                let param_wide = callee
                    .and_then(|f| f.signature.params.get(index))
                    .is_some_and(|p| is_wide_type(&p.ty));
                if param_wide || is_wide_value(arg, cx.vars) {
                    let words = cx.operand_words(arg)?;
                    mir_args.push(cx.spill(words));
                } else {
                    mir_args.push(resolve_operand(arg, cx.vregs, cx.vars, cx.globals)?);
                }
            }
            let ret_wide = callee.is_some_and(|f| is_wide_type(&f.signature.return_type));
            if ret_wide {
                let Some(result) = result else {
                    return Err(FromIRError::InvalidIR);
                };
                let placeholder = cx.temp();
                cx.out.push(Instruction::Call {
                    name: (*func_name).to_string(),
                    args: mir_args,
                    ret: Some(placeholder.clone()),
                });
                let loaded = cx.load_words(placeholder);
                let pair = cx.result_pair(result);
                cx.write_pair(pair, loaded);
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
        } if is_wide_prim(*source_type) => {
            let words = cx.operand_words(value)?;
            let arg = cx.spill(words);
            let (symbol, ret_ty) = match target_type {
                IRPrim::F64 if *signed => (RuntimeSymbol::Floattidf, MirType::f64()),
                IRPrim::F64 => (RuntimeSymbol::Floatuntidf, MirType::f64()),
                IRPrim::F32 if *signed => (RuntimeSymbol::Floattisf, MirType::f32()),
                IRPrim::F32 => (RuntimeSymbol::Floatuntisf, MirType::f32()),
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
        } if is_wide_prim(*target_type) => {
            let (symbol, param_ty) = match source_type {
                IRPrim::F64 if *signed => (RuntimeSymbol::Fixdfti, MirType::f64()),
                IRPrim::F64 => (RuntimeSymbol::Fixunsdfti, MirType::f64()),
                IRPrim::F32 if *signed => (RuntimeSymbol::Fixsfti, MirType::f32()),
                IRPrim::F32 => (RuntimeSymbol::Fixunssfti, MirType::f32()),
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
        IRInst::Binary { .. }
        | IRInst::Unary { .. }
        | IRInst::Cmp { .. }
        | IRInst::ZeroExtend { .. }
        | IRInst::SignExtend { .. }
        | IRInst::Trunc { .. }
        | IRInst::Load { .. }
        | IRInst::Store { .. }
        | IRInst::Phi { .. }
        | IRInst::Ret { .. }
        | IRInst::IntToFloat { .. }
        | IRInst::FloatToInt { .. }
        | IRInst::Bitcast { .. }
        | IRInst::FloatConvert { .. }
        | IRInst::FuncAddr { .. }
        | IRInst::Switch { .. }
        | IRInst::Br { .. }
        | IRInst::Jmp { .. }
        | IRInst::Unreachable
        | IRInst::Alloc { .. }
        | IRInst::AtomicLoad { .. }
        | IRInst::AtomicStore { .. }
        | IRInst::AtomicBinary { .. }
        | IRInst::AtomicCompareExchange { .. }
        | IRInst::Fence { .. }
        | IRInst::MemCpy { .. }
        | IRInst::MemMove { .. }
        | IRInst::MemSet { .. }
        | IRInst::Dealloc { .. }
        | IRInst::GetFieldPtr { .. }
        | IRInst::GetElemPtr { .. }
        | IRInst::PtrToInt { .. }
        | IRInst::IntToPtr { .. }
        | IRInst::Write { .. }
        | IRInst::Read { .. }
        | IRInst::WriteByte { .. }
        | IRInst::ReadByte { .. }
        | IRInst::Print { .. }
        | IRInst::VaStart { .. }
        | IRInst::VaArg { .. }
        | IRInst::VaEnd { .. }
        | IRInst::VaCopy { .. } => return Err(FromIRError::UnsupportedInstruction),
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
        cx.bin(IntBinOp::AShr, word.clone(), imm(63))
    } else {
        imm(0)
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

pub(super) fn phi_words<'a>(
    value: &IRVal<'a>,
    vregs: &mut VirtualRegAllocator,
    vars: &mut VarBindings<'a>,
    globals: &mut GlobalAddresses<'a>,
    reservations: &mut HashMap<VirtualReg, usize>,
    module: &IRModule<'a>,
) -> Result<(Operand, Operand), FromIRError> {
    let mut cx = Cx {
        vregs,
        vars,
        reservations,
        globals,
        module,
        out: Vec::new(),
    };
    let words = cx.operand_words(value)?;
    Ok((words.lo, words.hi))
}

pub(super) fn load_wide_parameter(
    address: Register,
    vregs: &mut VirtualRegAllocator,
    prelude: &mut Vec<Instruction>,
) -> [Register; 2] {
    let lo = Register::Virtual(vregs.allocate_gpr());
    let hi = Register::Virtual(vregs.allocate_gpr());
    let attrs = MemoryAttrs {
        align: 8,
        volatile: false,
    };
    prelude.push(Instruction::Load {
        ty: MirType::i64(),
        dst: lo.clone(),
        addr: AddressMode::BaseOffset {
            base: address.clone(),
            offset: 0,
            address_space: Default::default(),
        },
        attrs,
    });
    let high = Register::Virtual(vregs.allocate_gpr());
    prelude.push(Instruction::Lea {
        dst: high.clone(),
        base: address,
        offset: 8,
    });
    prelude.push(Instruction::Load {
        ty: MirType::i64(),
        dst: hi.clone(),
        addr: AddressMode::BaseOffset {
            base: high,
            offset: 0,
            address_space: Default::default(),
        },
        attrs,
    });
    [lo, hi]
}
