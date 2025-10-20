pub mod block;
pub mod function;
pub mod instruction;
pub mod module;
pub mod register;
/// LUMIR — Lamina Unified Machine Intermediate Representation
///
/// LUMIR is a **low-level, machine-friendly layer** produced after IR Processing
/// (and used before/after register allocation). It is assembly-like,
/// easy to apply optimizations, and straightforward to lower into target assembly.
///
/// # Architecture Overview
///
/// ```text
/// Parser → IR → LUMIR → [Optimizations] → Code Generator → Assembly
/// ```
///
/// ## Types
///
/// - **Scalars**: `i8 | i16 | i32 | i64 | f32 | f64 | ptr | i1`
/// - **Vectors**: `v128<lane> | v256<lane>`
///   - where `lane` ∈ `{ i8, i16, i32, i64, f32, f64 }`
///
/// ## Registers & Classes
///
/// - **Virtual regs**: `v0, v1, v2, ...` (unlimited)
/// - **Register classes**:
///   - `gpr` — General Purpose (integers, pointers)
///   - `fpr` — Floating Point (scalar float ops)
///   - `vec` — SIMD Vector registers
/// - **Note**: Physical registers (like `%rax`, `x0`) appear only **post-RA** (after register allocation)
///
/// ## Addressing Modes
///
/// - **Simple**: `[base + imm12]` — base register + 12-bit immediate
/// - **Indexed**: `[base + idx<<scale + imm4]` — with scale ∈ `{1, 2, 4, 8}`
/// - **LEA**: `lea dst, base, offset` — computes addresses; backends may fold into `ld`/`st`
///
/// ## Core Operations
///
/// ### Integer Arithmetic
/// `add, sub, mul, udiv, sdiv, urem, srem, and, or, xor, not, shl, lshr, ashr`
///
/// ### Floating Point
/// `fadd, fsub, fmul, fdiv, fneg, fsqrt, (fma?)`
///
/// ### Comparisons
/// - **Integer**: `cmp.{eq,ne,ult,ule,ugt,uge,slt,sle,sgt,sge}.i* → i1`
/// - **Float**: `fcmp.{eq,ne,lt,le,gt,ge}.f* → i1`
///
/// ### Select (Conditional Move)
/// `select <ty> dst, i1 cond, r_true, r_false` — works on scalars & vectors
///
/// ### Memory Operations
/// - **Load**: `ld.<ty> dst, [addr] {align=A, volatile?}`
/// - **Store**: `st.<ty> src, [addr] {align=A, volatile?}`
/// - **Vector**: `ld.v128<lane>`, `st.v128<lane>` (and `v256<lane>`)
///
/// ### Vector Operations
/// `vadd, vsub, vmul, vand, vor, vxor, vshl, vlshr, vashr`  
/// `vsplat, vextractlane, vinsertlane, vshuffle(mask)`
///
/// ### Control Flow
/// `jmp, br, switch, call, ret`
///
/// ### Meta Operations
/// `unreachable, safepoint, stackmap <id>, patchpoint <id>` (for GC and profiling)
///
/// ## Calling Convention
///
/// - **Arguments**: `v0..v7` (abstract; 8 argument registers)
/// - **Return**: `v0` (abstract return register)
/// - The abstract calling convention is mapped to the real ABI during code emission:
///   - **x86_64**: System V ABI (`rdi, rsi, rdx, rcx, r8, r9`)
///   - **AArch64**: AAPCS (`x0-x7`)
///   - **WASM**: WASM ABI (`i32, i64, f32, f64`)
///
/// ## Core ops
/// ```text
///   int   : add/sub/mul/udiv/sdiv/urem/srem/and/or/xor/not/shl/lshr/ashr/…
///   fp    : fadd/fsub/fmul/fdiv/fneg/fsqrt/(fma?)
///   cmp   : cmp.{eq,ne,ult,ule,ugt,uge,slt,sle,sgt,sge}.i* → i1
///           fcmp.{eq,ne,lt,le,gt,ge}.f*                   → i1
///   select: select <ty> dst, i1 cond, r_true, r_false      (scalar or vector)
///   mem   : ld.<ty> dst, [addr] {align=A, volatile?}
///           st.<ty> src, [addr] {align=A, volatile?}
///           ld.v128<lane> / st.v128<lane> (and v256<lane>)
///   vec   : vadd/vsub/vmul/vand/vor/vxor/vshl/vlshr/vashr
///           vsplat/vextractlane/vinsertlane/vshuffle(mask)
///   ctrl  : jmp, br, switch, call, ret
///   meta  : unrechable, safepoint, stackmap <id>, patchpoint <id> (for GC and profiling)
///
/// ## Example (Minimal Function)
/// ```asm
/// .func add_store
/// bb0:
///   v2:i64 = add.i64 v0, v1    ; v2 = v0 + v1
///   lea     vA, v3, 0          ; vA = v3 + 0 (address calculation)
///   st.i64  v2, [vA] {align=8} ; *vA = v2
///   ret                        ; return (implicitly v0)
/// .endfunc
/// ```
///
/// ## Transform Passes
///
/// Optimizations operate on LUMIR before final code generation. See `transform` module.
// Core modules
pub mod types;

// Transform system
pub mod transform;

// Re-exports for convenience
pub use block::Block;
pub use function::{Function, FunctionBuilder, Parameter, Signature};
pub use instruction::{
    AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction, IntBinOp, IntCmpOp,
    MemoryAttrs, Operand, VectorOp,
};
pub use module::{Global, Module, ModuleBuilder};
pub use register::{PhysicalReg, Register, RegisterClass, VirtualReg, VirtualRegAllocator};
pub use types::{MirType, ScalarType, VectorLane, VectorType};

#[derive(Debug)]
pub enum FromIRError {
    InvalidIR,
    UnsupportedType,
    UnsupportedInstruction,
    MissingEntryBlock,
    UnknownVariable,
}
impl std::fmt::Display for FromIRError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self{
            FromIRError::InvalidIR => write!(f,"InvalidIR"),
            FromIRError::UnsupportedType => write!(f,"Unsupported Type"),
            FromIRError::UnsupportedInstruction => write!(f,"Unsupported Inst"),
            FromIRError::MissingEntryBlock => write!(f,"Missing Entry"),
            FromIRError::UnknownVariable => write!(f,"Variable Unknown"),
        }
    }
}

// From IR
pub fn from_ir(ir: &crate::ir::Module<'_>, name: &str) -> Result<Module, FromIRError> {
    // Create an empty MIR module
    let mut mir_module = Module::new(name);

    // Convert each IR function
    for (func_name, ir_func) in &ir.functions {
        let mir_func = convert_function(func_name, ir_func)?;
        mir_module.add_function(mir_func);
    }

    Ok(mir_module)
}

fn convert_function<'a>(
    name: &'a str,
    f: &crate::ir::function::Function<'a>,
) -> Result<Function, FromIRError> {
    use crate::ir::types::Type as IRType;
    // Build MIR function signature
    let mut vreg_alloc = VirtualRegAllocator::new();

    let mut mir_sig = Signature::new(name);

    // Map return type
    match &f.signature.return_type {
        IRType::Void => {
            // None indicates void
        }
        other => {
            let ty = map_ir_type(other)?;
            mir_sig = mir_sig.with_return(ty);
        }
    }

    // Map parameters to virtual registers v0.. and add to signature
    let mut var_to_reg: std::collections::HashMap<&'a str, Register> =
        std::collections::HashMap::new();

    for param in &f.signature.params {
        let class = reg_class_for_type(&param.ty);
        let v = match class {
            RegisterClass::Fpr => Register::Virtual(vreg_alloc.allocate_fpr()),
            RegisterClass::Vec => Register::Virtual(vreg_alloc.allocate_vec()),
            _ => Register::Virtual(vreg_alloc.allocate_gpr()),
        };
        let mir_ty = map_ir_type(&param.ty)?;
        mir_sig.params.push(Parameter::new(v.clone(), mir_ty));
        var_to_reg.insert(param.name, v);
    }

    let mut mir_func = Function::new(mir_sig).with_entry(f.entry_block);

    // Emit entry block first, then the rest sorted by label for determinism
    let mut labels: Vec<&str> = f.basic_blocks.keys().copied().collect();
    labels.sort_unstable();

    // Ensure entry is first
    if let Some(pos) = labels.iter().position(|l| *l == f.entry_block) {
        labels.swap(0, pos);
    } else {
        return Err(FromIRError::MissingEntryBlock);
    }

    for label in labels {
        let ir_block = &f.basic_blocks[label];
        let mut mir_block = Block::new(label);

        for instr in &ir_block.instructions {
            // Best-effort conversion; unsupported instructions return an error
            match convert_instruction(instr, &mut vreg_alloc, &mut var_to_reg) {
                Ok(Some(mir_instr)) => mir_block.push(mir_instr),
                Ok(None) => {
                    // Instruction lowered to nothing (e.g., alloc placeholder)
                }
                Err(e) => return Err(e),
            }
        }

        mir_func.add_block(mir_block);
    }

    Ok(mir_func)
}

fn convert_instruction<'a>(
    instr: &crate::ir::instruction::Instruction<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut std::collections::HashMap<&'a str, Register>,
) -> Result<Option<Instruction>, FromIRError> {
    use crate::ir::instruction::{BinaryOp as IRBin, CmpOp as IRCmp};
    use crate::ir::types::{PrimitiveType as IRPrim, Type as IRType, Value as IRVal};

    // Helper to map IR values to MIR operands without borrowing var_to_reg immutably for long
    fn ir_value_to_operand<'a>(
        map: &std::collections::HashMap<&'a str, Register>,
        v: &IRVal<'a>,
    ) -> Result<Operand, FromIRError> {
        match v {
            IRVal::Variable(id) => map
                .get(id)
                .cloned()
                .map(Operand::Register)
                .ok_or(FromIRError::UnknownVariable),
            IRVal::Constant(lit) => literal_to_immediate(lit).map(Operand::Immediate),
            IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
        }
    }

    match instr {
        crate::ir::instruction::Instruction::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            // Assign destination register
            let dst = match *ty {
                IRPrim::F32 | IRPrim::F64 => Register::Virtual(vreg_alloc.allocate_fpr()),
                _ => Register::Virtual(vreg_alloc.allocate_gpr()),
            };
            var_to_reg.insert(*result, dst.clone());

            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = ir_value_to_operand(var_to_reg, lhs)?;
            let rhs_op = ir_value_to_operand(var_to_reg, rhs)?;

            let mir = match (op, *ty) {
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FAdd,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FSub,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FMul,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FDiv,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Add, _) => Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, _) => Instruction::IntBinary {
                    op: IntBinOp::Sub,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, _) => Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, _) => Instruction::IntBinary {
                    op: IntBinOp::SDiv,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
            };
            Ok(Some(mir))
        }
        crate::ir::instruction::Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr()); // i1 in gpr class
            var_to_reg.insert(*result, dst.clone());
            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = ir_value_to_operand(var_to_reg, lhs)?;
            let rhs_op = ir_value_to_operand(var_to_reg, rhs)?;
            let mir = match (*ty, op) {
                (IRPrim::F32 | IRPrim::F64, IRCmp::Eq) => Instruction::FloatCmp {
                    op: FloatCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ne) => Instruction::FloatCmp {
                    op: FloatCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Lt) => Instruction::FloatCmp {
                    op: FloatCmpOp::Lt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Le) => Instruction::FloatCmp {
                    op: FloatCmpOp::Le,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Gt) => Instruction::FloatCmp {
                    op: FloatCmpOp::Gt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ge) => Instruction::FloatCmp {
                    op: FloatCmpOp::Ge,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Eq) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::Eq,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ne) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::Ne,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Lt) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::ULt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Le) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::ULe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Gt) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::UGt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ge) => {
                    Instruction::IntCmp {
                        op: IntCmpOp::UGe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (_, IRCmp::Eq) => Instruction::IntCmp {
                    op: IntCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ne) => Instruction::IntCmp {
                    op: IntCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Lt) => Instruction::IntCmp {
                    op: IntCmpOp::SLt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Le) => Instruction::IntCmp {
                    op: IntCmpOp::SLe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Gt) => Instruction::IntCmp {
                    op: IntCmpOp::SGt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ge) => Instruction::IntCmp {
                    op: IntCmpOp::SGe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
            };
            Ok(Some(mir))
        }
        crate::ir::instruction::Instruction::Br {
            condition,
            true_label,
            false_label,
        } => {
            // Only variable conditions supported
            let cond_reg = match condition {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(Some(Instruction::Br {
                cond: match cond_reg {
                    Register::Virtual(v) => Register::Virtual(v),
                    Register::Physical(p) => Register::Physical(p),
                },
                true_target: (*true_label).to_string(),
                false_target: (*false_label).to_string(),
            }))
        }
        crate::ir::instruction::Instruction::Jmp { target_label } => Ok(Some(Instruction::Jmp {
            target: (*target_label).to_string(),
        })),
        crate::ir::instruction::Instruction::Ret { ty, value } => {
            if matches!(ty, IRType::Void) {
                return Ok(Some(Instruction::Ret { value: None }));
            }
            let op = match value {
                Some(v) => Some(ir_value_to_operand(var_to_reg, v)?),
                None => None,
            };
            Ok(Some(Instruction::Ret { value: op }))
        }
        crate::ir::instruction::Instruction::Load { result, ty, ptr } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(Some(Instruction::Load {
                ty: mir_ty,
                dst,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: false,
                },
            }))
        }
        crate::ir::instruction::Instruction::Store { ty, ptr, value } => {
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let src = ir_value_to_operand(var_to_reg, value)?;
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(Some(Instruction::Store {
                ty: mir_ty,
                src,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: false,
                },
            }))
        }
        crate::ir::instruction::Instruction::Call {
            result,
            func_name,
            args,
        } => {
            let mut mir_args = Vec::new();
            for a in args {
                mir_args.push(ir_value_to_operand(var_to_reg, a)?);
            }
            let ret = if let Some(res) = result {
                let dst = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*res, dst.clone());
                Some(dst)
            } else {
                None
            };
            Ok(Some(Instruction::Call {
                name: (*func_name).to_string(),
                args: mir_args,
                ret,
            }))
        }
        // Allocation: model as no-op (value becomes a pointer register)
        crate::ir::instruction::Instruction::Alloc {
            result,
            alloc_type: _,
            allocated_ty: _,
        } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst);
            Ok(None)
        }
        // Unhandled/unsupported instructions for now
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn map_ir_prim(p: crate::ir::types::PrimitiveType) -> Result<MirType, FromIRError> {
    use crate::ir::types::PrimitiveType as IRPrim;
    let scalar = match p {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Char => ScalarType::I8,
        IRPrim::I16 | IRPrim::U16 => ScalarType::I16,
        IRPrim::I32 | IRPrim::U32 => ScalarType::I32,
        IRPrim::I64 | IRPrim::U64 => ScalarType::I64,
        IRPrim::F32 => ScalarType::F32,
        IRPrim::F64 => ScalarType::F64,
        IRPrim::Bool => ScalarType::I1,
        IRPrim::Ptr => ScalarType::Ptr,
    };
    Ok(MirType::Scalar(scalar))
}

fn map_ir_type(ty: &crate::ir::types::Type<'_>) -> Result<MirType, FromIRError> {
    use crate::ir::types::Type as IRType;
    match ty {
        IRType::Primitive(p) => map_ir_prim(*p),
        _ => Err(FromIRError::UnsupportedType),
    }
}

fn literal_to_immediate(l: &crate::ir::types::Literal<'_>) -> Result<Immediate, FromIRError> {
    use crate::ir::types::Literal as IRLit;
    let imm = match l {
        IRLit::I8(v) => Immediate::I8(*v),
        IRLit::I16(v) => Immediate::I16(*v),
        IRLit::I32(v) => Immediate::I32(*v),
        IRLit::I64(v) => Immediate::I64(*v),
        IRLit::U8(v) => Immediate::I8(*v as i8),
        IRLit::U16(v) => Immediate::I16(*v as i16),
        IRLit::U32(v) => Immediate::I32(*v as i32),
        IRLit::U64(v) => Immediate::I64(*v as i64),
        IRLit::F32(v) => Immediate::F32(*v),
        IRLit::F64(v) => Immediate::F64(*v),
        IRLit::Bool(b) => Immediate::I8(if *b { 1 } else { 0 }),
        IRLit::Char(c) => Immediate::I8(*c as i8),
        IRLit::String(_) => return Err(FromIRError::UnsupportedInstruction),
    };
    Ok(imm)
}

fn reg_class_for_type(ty: &crate::ir::types::Type<'_>) -> RegisterClass {
    use crate::ir::types::{PrimitiveType as IRPrim, Type as IRType};
    match ty {
        IRType::Primitive(IRPrim::F32) | IRType::Primitive(IRPrim::F64) => RegisterClass::Fpr,
        _ => RegisterClass::Gpr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{i64 as ir_i64, var};
    use crate::ir::instruction::BinaryOp;
    use crate::ir::types::{PrimitiveType, Type};
    use crate::ir::{FunctionParameter, IRBuilder};

    #[test]
    fn test_from_ir_simple_add() {
        // Build IR: fn @add(i64 %a, i64 %b) -> i64 { %sum = add.i64 %a, %b; ret.i64 %sum }
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "add",
                vec![
                FunctionParameter {
                    name: "a",
                    ty: Type::Primitive(PrimitiveType::I64),
                },
                FunctionParameter {
                    name: "b",
                    ty: Type::Primitive(PrimitiveType::I64),
                },
                ],
                Type::Primitive(PrimitiveType::I64),
            )
            .binary(BinaryOp::Add, "sum", PrimitiveType::I64, var("a"), var("b"))
            .ret(Type::Primitive(PrimitiveType::I64), var("sum"));

        let ir_module = builder.build();

        // Convert IR → MIR
        let mir_module = from_ir(&ir_module, "test").expect("from_ir should succeed");

        // Validate MIR module
        let func = mir_module.get_function("add").expect("function exists");
        assert_eq!(func.sig.name, "add");
        assert_eq!(func.entry, "entry");

        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2);

        // Check first instruction is an integer add producing i64
        match &entry.instructions[0] {
            Instruction::IntBinary {
                op,
                ty,
                dst: _,
                lhs,
                rhs,
            } => {
                assert!(matches!(op, IntBinOp::Add));
                assert!(matches!(ty, MirType::Scalar(ScalarType::I64)));
                assert!(matches!(lhs, Operand::Register(_)));
                assert!(matches!(rhs, Operand::Register(_)));
            }
            other => panic!("Unexpected first instruction: {:?}", other),
        }

        // Check second instruction is a return with the computed value
        match &entry.instructions[1] {
            Instruction::Ret { value } => {
                assert!(matches!(value, Some(Operand::Register(_))));
            }
            other => panic!("Unexpected second instruction: {:?}", other),
        }
    }

    #[test]
    fn test_from_ir_load_store_and_call() {
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "mem",
                vec![FunctionParameter {
                    name: "p",
                    ty: Type::Primitive(PrimitiveType::Ptr),
                }],
                Type::Void,
            )
            .store(Type::Primitive(PrimitiveType::I64), var("p"), ir_i64(42))
            .load("v", Type::Primitive(PrimitiveType::I64), var("p"))
            .call(None, "puts", vec![var("p")])
            .ret_void();

        let ir_module = builder.build();
        let mir_module = from_ir(&ir_module, "test").expect("from_ir should succeed");

        let func = mir_module.get_function("mem").expect("function exists");
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 4);
        assert!(matches!(&entry.instructions[0], Instruction::Store { .. }));
        assert!(matches!(&entry.instructions[1], Instruction::Load { .. }));
        assert!(matches!(&entry.instructions[2], Instruction::Call { .. }));
        assert!(matches!(&entry.instructions[3], Instruction::Ret { .. }));
    }
}
