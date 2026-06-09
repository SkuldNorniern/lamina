//! IR-to-MIR conversion implementation.
//!
//! Main conversion logic that transforms high-level
//! Lamina IR into low-level LUMIR. The conversion process includes:
//!
//! - Function signature conversion
//! - Instruction lowering
//! - Variable binding and register assignment
//! - Control flow graph construction
//! - Type mapping and validation

use std::collections::{HashMap, HashSet};

use super::error::FromIRError;
use super::mapping::{map_ir_prim, map_ir_type};
use crate::ir::instruction::{AllocType, BinaryOp as IRBin, CmpOp as IRCmp, Instruction as IRInst};
use crate::ir::types::{Literal as IRLit, PrimitiveType as IRPrim, Type as IRType, Value as IRVal};
use crate::mir::{
    AddressMode, Block, FloatBinOp, FloatCmpOp, Function, Immediate, Instruction,
    Instruction as MirInst, IntBinOp, IntCmpOp, MemoryAttrs, MirType, Module, Operand, Parameter,
    Register, RegisterClass, ScalarType, Signature, VirtualReg, VirtualRegAllocator,
};

#[cfg(feature = "nightly")]
use crate::mir::{MemoryOrdering, SimdOp};

/// Resolve an IR value to a MIR operand, allocating a fresh GPR binding on first encounter.
fn resolve_operand<'a>(
    v: &IRVal<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut HashMap<&'a str, Register>,
) -> Result<Operand, FromIRError> {
    match v {
        IRVal::Variable(id) => {
            if let Some(r) = var_to_reg.get(id) {
                Ok(Operand::Register(r.clone()))
            } else {
                let r = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*id, r.clone());
                Ok(Operand::Register(r))
            }
        }
        IRVal::Constant(lit) => literal_to_immediate(lit).map(Operand::Immediate),
        IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
    }
}

/// Look up `result` in the register map; if absent, allocate a fresh GPR and bind it.
fn resolve_or_alloc_gpr<'a>(
    result: &'a str,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut HashMap<&'a str, Register>,
) -> Register {
    if let Some(existing) = var_to_reg.get(result) {
        existing.clone()
    } else {
        let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
        var_to_reg.insert(result, fresh.clone());
        fresh
    }
}

/// Compute the byte size of an IR type; returns `None` for unsized/void types.
fn sizeof_ir_type(ty: &IRType<'_>) -> Option<u64> {
    match ty {
        IRType::Primitive(p) => Some(match p {
            IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 1,
            IRPrim::I16 | IRPrim::U16 => 2,
            IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 4,
            IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 8,
        }),
        IRType::Array { element_type, size } => {
            sizeof_ir_type(element_type).map(|es| es.saturating_mul(*size))
        }
        IRType::Struct(fields) => {
            let mut total = 0u64;
            for f in fields {
                total = total.saturating_add(sizeof_ir_type(&f.ty)?);
            }
            Some(total)
        }
        _ => None,
    }
}

/// Bit-width for sign-extension: returns 0 for float types (unsupported).
fn int_bits_for_sext(ty: &IRPrim) -> u32 {
    match ty {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 8,
        IRPrim::I16 | IRPrim::U16 => 16,
        IRPrim::I32 | IRPrim::U32 => 32,
        IRPrim::I64 | IRPrim::U64 | IRPrim::Ptr => 64,
        IRPrim::F32 | IRPrim::F64 => 0,
    }
}

/// Bit-width for bitcast: F32 = 32 bits, F64 = 64 bits (valid bitcast widths).
fn int_bits_for_bitcast(ty: &IRPrim) -> u32 {
    match ty {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 8,
        IRPrim::I16 | IRPrim::U16 => 16,
        IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 32,
        IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 64,
    }
}

pub fn from_ir(ir: &crate::ir::Module<'_>, name: &str) -> Result<Module, FromIRError> {
    let mut mir_module = Module::new(name);

    for (func_name, ir_func) in &ir.functions {
        // Check if function is external before converting
        if ir_func
            .annotations
            .contains(&crate::ir::function::FunctionAnnotation::Extern)
        {
            mir_module.mark_external(*func_name);
        }

        let mir_func = convert_function(func_name, ir_func)?;
        mir_module.add_function(mir_func);
    }

    Ok(mir_module)
}

fn convert_function<'a>(
    name: &'a str,
    f: &crate::ir::function::Function<'a>,
) -> Result<Function, FromIRError> {
    let mut vreg_alloc = VirtualRegAllocator::new();
    let mut mir_sig = Signature::new(name);

    match &f.signature.return_type {
        IRType::Void => {}
        other => {
            let ty = map_ir_type(other)?;
            mir_sig = mir_sig.with_return(ty);
        }
    }

    let mut var_to_reg: HashMap<&'a str, Register> = HashMap::new();

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

    // BFS traversal starting from entry block, guard with visited to avoid cycles
    let mut visited: HashSet<&'a str> = HashSet::new();
    let mut worklist: std::collections::VecDeque<&'a str> = std::collections::VecDeque::new();

    // Ensure entry exists
    if !f.basic_blocks.contains_key(f.entry_block) {
        return Err(FromIRError::MissingEntryBlock);
    }
    worklist.push_back(f.entry_block);
    visited.insert(f.entry_block);

    while let Some(label) = worklist.pop_front() {
        let ir_block = &f.basic_blocks[label];
        let mut mir_block = Block::new(label);

        for instr in &ir_block.instructions {
            match convert_instruction(instr, &mut vreg_alloc, &mut var_to_reg) {
                Ok(mir_instrs) => {
                    for mir_instr in mir_instrs {
                        mir_block.push(mir_instr);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // After converting this block, enqueue IR successors (prevents borrow/move issues)
        if let Some(term) = ir_block.instructions.last() {
            match term {
                IRInst::Br {
                    true_label,
                    false_label,
                    ..
                } => {
                    if !visited.contains(true_label) && f.basic_blocks.contains_key(true_label) {
                        visited.insert(true_label);
                        worklist.push_back(true_label);
                    }
                    if !visited.contains(false_label) && f.basic_blocks.contains_key(false_label) {
                        visited.insert(false_label);
                        worklist.push_back(false_label);
                    }
                }
                IRInst::Jmp { target_label } => {
                    if !visited.contains(target_label) && f.basic_blocks.contains_key(target_label)
                    {
                        visited.insert(target_label);
                        worklist.push_back(target_label);
                    }
                }
                IRInst::Switch { default, cases, .. } => {
                    if !visited.contains(default) && f.basic_blocks.contains_key(default) {
                        visited.insert(default);
                        worklist.push_back(default);
                    }
                    for (_, case_label) in cases {
                        let case_str: &str = case_label;
                        if !visited.contains(case_str) && f.basic_blocks.contains_key(case_str) {
                            visited.insert(case_str);
                            worklist.push_back(case_str);
                        }
                    }
                }
                _ => {}
            }
        }

        mir_func.add_block(mir_block);
    }

    {
        let mut edge_moves: HashMap<(String, String), Vec<Instruction>> = HashMap::new();

        for (succ_label, ir_block) in &f.basic_blocks {
            for instr in &ir_block.instructions {
                if let IRInst::Phi {
                    result,
                    ty,
                    incoming,
                } = instr
                {
                    if let Some(dst_reg) = var_to_reg.get(result).cloned() {
                        let mir_ty = map_ir_type(ty)?;
                        for (val, pred_label) in incoming {
                            let src_op = resolve_operand(val, &mut vreg_alloc, &mut var_to_reg)?;
                            // Encode a move using add with zero; the backend already handles it.
                            let mov_like = Instruction::IntBinary {
                                op: IntBinOp::Add,
                                ty: mir_ty,
                                dst: dst_reg.clone(),
                                lhs: src_op,
                                rhs: Operand::Immediate(Immediate::I64(0)),
                            };
                            edge_moves
                                .entry(((*pred_label).to_string(), (*succ_label).to_string()))
                                .or_default()
                                .push(mov_like);
                        }
                    } else {
                        // Bind destination conservatively to avoid cascading unknowns
                        let new_dst = Register::Virtual(vreg_alloc.allocate_gpr());
                        var_to_reg.insert(*result, new_dst);
                    }
                }
            }
        }

        // Apply edge moves with edge-splitting when necessary.
        let mut trampoline_counter: u64 = 0;
        for ((pred, succ), moves) in edge_moves {
            // Snapshot the current terminator without holding a mutable borrow
            let term = mir_func
                .get_block(&pred)
                .and_then(|bb| bb.terminator().cloned());

            let mut handled = false;
            if let Some(Instruction::Jmp { target }) = term.as_ref()
                && target == &succ
                && let Some(pred_bb) = mir_func.get_block_mut(&pred)
                && !pred_bb.instructions.is_empty()
            {
                let insert_pos = pred_bb.instructions.len().saturating_sub(1);
                pred_bb
                    .instructions
                    .splice(insert_pos..insert_pos, moves.clone());
                handled = true;
            }
            if !handled
                && let Some(Instruction::Br {
                    cond: _,
                    true_target,
                    false_target,
                }) = term.as_ref()
            {
                let (arm_is_true, matches) = if true_target == &succ {
                    (true, true)
                } else if false_target == &succ {
                    (false, true)
                } else {
                    (true, false)
                };
                if matches {
                    trampoline_counter = trampoline_counter.saturating_add(1);
                    let tramp_label = format!(
                        "{}_to_{}_phi{}",
                        pred.replace('%', "_"),
                        succ.replace('%', "_"),
                        trampoline_counter
                    );
                    // Create trampoline and add it to the function
                    let mut tramp = Block::new(tramp_label.clone());
                    for m in &moves {
                        tramp.push(m.clone());
                    }
                    tramp.push(Instruction::Jmp {
                        target: succ.clone(),
                    });
                    mir_func.add_block(tramp);
                    if let Some(pred_mut) = mir_func.get_block_mut(&pred)
                        && let Some(last) = pred_mut.instructions.last_mut()
                        && let Instruction::Br {
                            cond: _,
                            true_target: t,
                            false_target: f,
                        } = last
                    {
                        if arm_is_true {
                            *t = tramp_label.clone();
                        } else {
                            *f = tramp_label.clone();
                        }
                        handled = true;
                    }
                }
            }
            if !handled && let Some(pred_bb) = mir_func.get_block_mut(&pred) {
                if !pred_bb.instructions.is_empty() {
                    let insert_pos = pred_bb.instructions.len().saturating_sub(1);
                    pred_bb
                        .instructions
                        .splice(insert_pos..insert_pos, moves.clone());
                } else {
                    for m in &moves {
                        pred_bb.push(m.clone());
                    }
                }
            }
        }
    }

    add_missing_initializations(&mut mir_func);

    Ok(mir_func)
}

fn add_missing_initializations(func: &mut Function) {
    // Collect all defined and used virtual registers
    let mut defined_regs = HashSet::new();
    let mut used_regs = HashSet::new();

    // Collect parameter registers as defined
    for param in &func.sig.params {
        if let Register::Virtual(vreg) = &param.reg {
            defined_regs.insert(*vreg);
        }
    }

    for block in &func.blocks {
        for instr in &block.instructions {
            // Collect defined registers
            if let Some(def_reg) = instr.def_reg()
                && let Register::Virtual(vreg) = def_reg
            {
                defined_regs.insert(*vreg);
            }

            // Collect used registers from operands
            collect_regs_from_instruction(instr, &mut used_regs);
        }
    }

    // Find registers that are used but not defined
    let undefined_regs: Vec<_> = used_regs.difference(&defined_regs).cloned().collect();

    if !undefined_regs.is_empty()
        && let Some(entry_block) = func.blocks.iter_mut().find(|b| b.label == func.entry)
    {
        // Insert initializations at the beginning of the entry block
        let mut init_instrs = Vec::new();

        for vreg in undefined_regs {
            // Initialize undefined registers to 0
            let init_instr = MirInst::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: Register::Virtual(vreg),
                lhs: Operand::Immediate(Immediate::I64(0)),
                rhs: Operand::Immediate(Immediate::I64(0)),
            };
            init_instrs.push(init_instr);
        }

        // Insert at the beginning - use splice to insert all instructions at once
        entry_block.instructions.splice(0..0, init_instrs);
    }
}

fn collect_regs_from_instruction(instr: &Instruction, used_regs: &mut HashSet<VirtualReg>) {
    match instr {
        Instruction::IntBinary { lhs, rhs, .. }
        | Instruction::FloatBinary { lhs, rhs, .. }
        | Instruction::IntCmp { lhs, rhs, .. }
        | Instruction::FloatCmp { lhs, rhs, .. } => {
            collect_regs_from_operand(lhs, used_regs);
            collect_regs_from_operand(rhs, used_regs);
        }
        Instruction::Load { addr, .. } => {
            collect_regs_from_address_mode(addr, used_regs);
        }
        Instruction::Store { addr, src, .. } => {
            collect_regs_from_address_mode(addr, used_regs);
            collect_regs_from_operand(src, used_regs);
        }
        Instruction::Call { args, .. } => {
            for arg in args {
                collect_regs_from_operand(arg, used_regs);
            }
        }
        Instruction::Ret { value: Some(val) } => {
            collect_regs_from_operand(val, used_regs);
        }
        _ => {}
    }
}

fn collect_regs_from_operand(operand: &Operand, used_regs: &mut HashSet<VirtualReg>) {
    match operand {
        Operand::Register(reg) => {
            if let Register::Virtual(vreg) = reg {
                used_regs.insert(*vreg);
            }
        }
        Operand::Immediate(_) => {}
    }
}

fn collect_regs_from_address_mode(addr: &AddressMode, used_regs: &mut HashSet<VirtualReg>) {
    match addr {
        AddressMode::BaseOffset { base, .. } => {
            if let Register::Virtual(vreg) = base {
                used_regs.insert(*vreg);
            }
        }
        AddressMode::BaseIndexScale { base, index, .. } => {
            if let Register::Virtual(vreg) = base {
                used_regs.insert(*vreg);
            }
            if let Register::Virtual(vreg) = index {
                used_regs.insert(*vreg);
            }
        }
    }
}

fn convert_instruction<'a>(
    instr: &IRInst<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut HashMap<&'a str, Register>,
) -> Result<Vec<Instruction>, FromIRError> {
    #[cfg(feature = "nightly")]
    fn ir_address_mode_to_mir<'b>(
        ptr: &IRVal<'b>,
        vreg_alloc: &mut VirtualRegAllocator,
        var_to_reg: &mut std::collections::HashMap<&'b str, Register>,
    ) -> Result<AddressMode, FromIRError> {
        match ptr {
            IRVal::Variable(id) => {
                let base = if let Some(r) = var_to_reg.get(id) {
                    r.clone()
                } else {
                    let r = Register::Virtual(vreg_alloc.allocate_gpr());
                    var_to_reg.insert(*id, r.clone());
                    r
                };
                Ok(AddressMode::BaseOffset { base, offset: 0 })
            }
            IRVal::Constant(_) | IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
        }
    }

    match instr {
        IRInst::Binary {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            // Reuse existing binding for result if it exists; otherwise allocate a new one
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = match *ty {
                    IRPrim::F32 | IRPrim::F64 => Register::Virtual(vreg_alloc.allocate_fpr()),
                    _ => Register::Virtual(vreg_alloc.allocate_gpr()),
                };
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            let mir_ty = map_ir_prim(*ty)?;
            // Read operands using the PREVIOUS bindings to avoid using the new dest mapping
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg)?;

            let mir = match (op, *ty) {
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FAdd,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FSub,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FMul,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: FloatBinOp::FDiv,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Add, _) => Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, _) => Instruction::IntBinary {
                    op: IntBinOp::Sub,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, _) => Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, _) => Instruction::IntBinary {
                    op: IntBinOp::SDiv,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Rem, _) => Instruction::IntBinary {
                    op: IntBinOp::SRem,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::And, _) => Instruction::IntBinary {
                    op: IntBinOp::And,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Or, _) => Instruction::IntBinary {
                    op: IntBinOp::Or,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Xor, _) => Instruction::IntBinary {
                    op: IntBinOp::Xor,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Shl, _) => Instruction::IntBinary {
                    op: IntBinOp::Shl,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Shr, _) => Instruction::IntBinary {
                    op: IntBinOp::AShr,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
            };
            Ok(vec![mir])
        }
        // SSA merge: create a binding for the phi result so subsequent uses resolve.
        // Semantics are not materialized here; a later SSA elimination pass should lower this.
        IRInst::Phi {
            result,
            ty,
            incoming: _,
        } => {
            // Choose register class based on type
            let class = reg_class_for_type(ty);
            let dst = match class {
                RegisterClass::Fpr => Register::Virtual(vreg_alloc.allocate_fpr()),
                RegisterClass::Vec => Register::Virtual(vreg_alloc.allocate_vec()),
                _ => Register::Virtual(vreg_alloc.allocate_gpr()),
            };
            var_to_reg.insert(*result, dst);
            Ok(vec![])
        }
        IRInst::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg)?;
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
            Ok(vec![mir])
        }
        IRInst::Br {
            condition,
            true_label,
            false_label,
        } => {
            let cond_reg = match condition {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::Br {
                cond: cond_reg,
                true_target: (*true_label).to_string(),
                false_target: (*false_label).to_string(),
            }])
        }
        IRInst::Switch {
            ty: _,
            value,
            default,
            cases,
        } => {
            // Lower IR switch to MIR switch using a dedicated MIR instruction.
            // The MIR lowering and backends treat the value as an integer in a register
            // and cases as i64 immediates.
            let value_reg = match value {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };

            let mut mir_cases = Vec::with_capacity(cases.len());

            for (lit, label) in cases {
                let key: i64 = match lit {
                    IRLit::I8(v) => *v as i64,
                    IRLit::I16(v) => *v as i64,
                    IRLit::I32(v) => *v as i64,
                    IRLit::I64(v) => *v,
                    IRLit::U8(v) => *v as i64,
                    IRLit::U16(v) => *v as i64,
                    IRLit::U32(v) => *v as i64,
                    IRLit::U64(v) => *v as i64,
                    IRLit::Bool(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
                    IRLit::Char(c) => *c as i64,
                    IRLit::F32(_) | IRLit::F64(_) | IRLit::String(_) => {
                        return Err(FromIRError::UnsupportedType);
                    }
                };
                mir_cases.push((key, (*label).to_string()));
            }

            Ok(vec![Instruction::Switch {
                value: value_reg,
                cases: mir_cases,
                default: (*default).to_string(),
            }])
        }
        IRInst::Jmp { target_label } => Ok(vec![Instruction::Jmp {
            target: (*target_label).to_string(),
        }]),
        IRInst::Ret { ty, value } => {
            if matches!(ty, IRType::Void) {
                return Ok(vec![Instruction::Ret { value: None }]);
            }
            let op = match value {
                Some(v) => Some(resolve_operand(v, vreg_alloc, var_to_reg)?),
                None => None,
            };
            Ok(vec![Instruction::Ret { value: op }])
        }
        IRInst::Load { result, ty, ptr } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(vec![Instruction::Load {
                ty: mir_ty,
                dst,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: false,
                },
            }])
        }
        IRInst::Store { ty, ptr, value } => {
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let src = resolve_operand(value, vreg_alloc, var_to_reg)?;
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(vec![Instruction::Store {
                ty: mir_ty,
                src,
                addr,
                attrs: MemoryAttrs {
                    align: mir_ty.alignment() as u8,
                    volatile: false,
                },
            }])
        }
        IRInst::Call {
            result,
            func_name,
            args,
        } => {
            let mut mir_args = Vec::new();
            for a in args {
                mir_args.push(resolve_operand(a, vreg_alloc, var_to_reg)?);
            }
            let ret = result.map(|res| resolve_or_alloc_gpr(res, vreg_alloc, var_to_reg));
            Ok(vec![Instruction::Call {
                name: (*func_name).to_string(),
                args: mir_args,
                ret,
            }])
        }
        IRInst::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            // Compute mask based on source type bit-width
            let (mask, mir_ty) = match (source_type, target_type) {
                (IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char, _) => {
                    (0xFFu64, map_ir_prim(*target_type)?)
                }
                (IRPrim::I16 | IRPrim::U16, _) => (0xFFFFu64, map_ir_prim(*target_type)?),
                (IRPrim::I32 | IRPrim::U32, _) => (0xFFFF_FFFFu64, map_ir_prim(*target_type)?),
                _ => (0xFFFF_FFFF_FFFF_FFFFu64, map_ir_prim(*target_type)?),
            };
            let val_op = resolve_operand(value, vreg_alloc, var_to_reg)?;
            let mask_op = Operand::Immediate(Immediate::I64(mask as i64));
            Ok(vec![Instruction::IntBinary {
                op: IntBinOp::And,
                ty: mir_ty,
                dst,
                lhs: val_op,
                rhs: mask_op,
            }])
        }
        IRInst::Trunc {
            result,
            source_type: _,
            target_type,
            value,
        } => {
            // Lower truncation as an AND with a mask of the target width.
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let (mask, mir_ty) = match target_type {
                IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => {
                    (0xFFu64, map_ir_prim(*target_type)?)
                }
                IRPrim::I16 | IRPrim::U16 => (0xFFFFu64, map_ir_prim(*target_type)?),
                IRPrim::I32 | IRPrim::U32 => (0xFFFF_FFFFu64, map_ir_prim(*target_type)?),
                IRPrim::I64 | IRPrim::U64 | IRPrim::Ptr => {
                    (0xFFFF_FFFF_FFFF_FFFFu64, map_ir_prim(*target_type)?)
                }
                _ => (0xFFFF_FFFF_FFFF_FFFFu64, map_ir_prim(*target_type)?),
            };
            let val_op = resolve_operand(value, vreg_alloc, var_to_reg)?;
            let mask_op = Operand::Immediate(Immediate::I64(mask as i64));
            Ok(vec![Instruction::IntBinary {
                op: IntBinOp::And,
                ty: mir_ty,
                dst,
                lhs: val_op,
                rhs: mask_op,
            }])
        }
        IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);

            let src_bits = int_bits_for_sext(source_type);
            let dst_bits = int_bits_for_sext(target_type);

            if src_bits == 0 || dst_bits == 0 || src_bits >= dst_bits {
                return Err(FromIRError::UnsupportedType);
            }

            let mir_ty = map_ir_prim(*target_type)?;
            let shift = (dst_bits - src_bits) as i64;

            let val_op = resolve_operand(value, vreg_alloc, var_to_reg)?;

            // temp = (val << shift)
            let temp_reg = Register::Virtual(vreg_alloc.allocate_gpr());
            let shift_imm = Operand::Immediate(Immediate::I64(shift));

            let instrs = vec![
                Instruction::IntBinary {
                    op: IntBinOp::Shl,
                    ty: mir_ty,
                    dst: temp_reg.clone(),
                    lhs: val_op,
                    rhs: shift_imm,
                },
                // dst = temp >> shift (arithmetic shift right propagates sign)
                Instruction::IntBinary {
                    op: IntBinOp::AShr,
                    ty: mir_ty,
                    dst,
                    lhs: Operand::Register(temp_reg),
                    rhs: Operand::Immediate(Immediate::I64(shift)),
                },
            ];

            Ok(instrs)
        }
        IRInst::Bitcast {
            result,
            source_type,
            target_type,
            value,
        } => {
            if int_bits_for_bitcast(source_type) != int_bits_for_bitcast(target_type) {
                return Err(FromIRError::UnsupportedType);
            }

            match value {
                IRVal::Variable(id) => {
                    // Reuse the same MIR register for the result; MIR is untyped and the
                    // backend will see the result type through later uses.
                    let src_reg = resolve_or_alloc_gpr(id, vreg_alloc, var_to_reg);
                    var_to_reg.insert(*result, src_reg);
                    Ok(vec![])
                }
                IRVal::Constant(lit) => {
                    let mir_ty = map_ir_prim(*target_type)?;
                    let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);

                    let imm = match (source_type, target_type, lit) {
                        (IRPrim::F32, IRPrim::I32, IRLit::F32(v)) => {
                            Immediate::I32(v.to_bits() as i32)
                        }
                        (IRPrim::F64, IRPrim::I64, IRLit::F64(v)) => {
                            Immediate::I64(v.to_bits() as i64)
                        }
                        (IRPrim::I32, IRPrim::F32, IRLit::I32(v)) => Immediate::I32(*v),
                        (IRPrim::I64, IRPrim::F64, IRLit::I64(v)) => Immediate::I64(*v),
                        _ => return Err(FromIRError::UnsupportedInstruction),
                    };

                    Ok(vec![Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: mir_ty,
                        dst,
                        lhs: Operand::Immediate(imm),
                        rhs: Operand::Immediate(Immediate::I64(0)),
                    }])
                }
                IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
            }
        }
        IRInst::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);

            let mir_ty = map_ir_type(ty)?;

            // Condition must be a variable bound to a MIR register.
            let cond_reg = match cond {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };

            let tv = resolve_operand(true_val, vreg_alloc, var_to_reg)?;
            let fv = resolve_operand(false_val, vreg_alloc, var_to_reg)?;

            Ok(vec![Instruction::Select {
                dst,
                ty: mir_ty,
                cond: cond_reg,
                true_val: tv,
                false_val: fv,
            }])
        }
        IRInst::PtrToInt {
            result,
            ptr_value,
            target_type: _,
        } => {
            // On 64-bit, pointers are integers; lower to add with 0 to move value
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(ptr_value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst,
                lhs: val_op,
                rhs: Operand::Immediate(Immediate::I64(0)),
            }])
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type: _,
        } => {
            // Treat as identity move via add with 0 (pointer-sized)
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(int_value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst,
                lhs: val_op,
                rhs: Operand::Immediate(Immediate::I64(0)),
            }])
        }
        IRInst::MemCpy { dst, src, size } => {
            let args = vec![
                resolve_operand(dst, vreg_alloc, var_to_reg)?,
                resolve_operand(src, vreg_alloc, var_to_reg)?,
                resolve_operand(size, vreg_alloc, var_to_reg)?,
            ];
            Ok(vec![Instruction::Call {
                name: "memcpy".to_string(),
                args,
                ret: None,
            }])
        }
        IRInst::MemMove { dst, src, size } => {
            let args = vec![
                resolve_operand(dst, vreg_alloc, var_to_reg)?,
                resolve_operand(src, vreg_alloc, var_to_reg)?,
                resolve_operand(size, vreg_alloc, var_to_reg)?,
            ];
            Ok(vec![Instruction::Call {
                name: "memmove".to_string(),
                args,
                ret: None,
            }])
        }
        IRInst::MemSet { dst, value, size } => {
            let args = vec![
                resolve_operand(dst, vreg_alloc, var_to_reg)?,
                resolve_operand(value, vreg_alloc, var_to_reg)?,
                resolve_operand(size, vreg_alloc, var_to_reg)?,
            ];
            Ok(vec![Instruction::Call {
                name: "memset".to_string(),
                args,
                ret: None,
            }])
        }
        IRInst::GetFieldPtr {
            result,
            struct_ptr,
            field_index,
        } => {
            // Simplified layout: 8 bytes per field
            let base = match struct_ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let offset = (*field_index as i32) * 8;
            Ok(vec![Instruction::Lea { dst, base, offset }])
        }
        IRInst::GetElemPtr {
            result,
            array_ptr,
            index,
            element_type,
        } => {
            // Constant-index GEP -> single LEA; variable index currently unsupported in MIR single-op model
            let base = match array_ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let elem_size: i32 = match element_type {
                IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 1,
                IRPrim::I16 | IRPrim::U16 => 2,
                IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 4,
                IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 8,
            };
            let idx_const = match index {
                IRVal::Constant(lit) => match lit {
                    IRLit::I64(v) => Some(*v),
                    IRLit::I32(v) => Some(*v as i64),
                    IRLit::U64(v) => Some(*v as i64),
                    IRLit::U32(v) => Some(*v as i64),
                    _ => None,
                },
                _ => None,
            };
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);

            if let Some(iv) = idx_const {
                // Constant index: use simple LEA with offset
                let offset = (iv as i32).saturating_mul(elem_size);
                Ok(vec![Instruction::Lea { dst, base, offset }])
            } else {
                // Variable index: emit two instructions
                // 1. temp = index * elem_size
                // 2. result = base + temp

                // Get the index operand
                let index_op = resolve_operand(index, vreg_alloc, var_to_reg)?;

                // Allocate a temp register for the scaled index
                let temp_scaled = Register::Virtual(vreg_alloc.allocate_gpr());

                // Create instructions
                let instrs = vec![
                    // Instruction 1: temp_scaled = index * elem_size
                    Instruction::IntBinary {
                        op: IntBinOp::Mul,
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: temp_scaled.clone(),
                        lhs: index_op,
                        rhs: Operand::Immediate(Immediate::I64(elem_size as i64)),
                    },
                    // Instruction 2: result = base + temp_scaled
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: dst.clone(),
                        lhs: Operand::Register(base),
                        rhs: Operand::Register(temp_scaled),
                    },
                ];

                Ok(instrs)
            }
        }
        // Debug and I/O operations: lower to calls with conventional names
        IRInst::Print { value } => {
            if matches!(value, IRVal::Constant(IRLit::String(_))) {
                return Err(FromIRError::PrintStringLiteralUnsupported);
            }
            let arg = resolve_operand(value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::Call {
                name: "print".to_string(),
                args: vec![arg],
                ret: None,
            }])
        }
        IRInst::Write {
            buffer,
            size,
            result,
        } => {
            let buf = resolve_operand(buffer, vreg_alloc, var_to_reg)?;
            let sz = resolve_operand(size, vreg_alloc, var_to_reg)?;
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "write".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        IRInst::WriteByte { value, result } => {
            let v = resolve_operand(value, vreg_alloc, var_to_reg)?;
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "writebyte".to_string(),
                args: vec![v],
                ret: Some(dst),
            }])
        }
        IRInst::ReadByte { result } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "readbyte".to_string(),
                args: vec![],
                ret: Some(dst),
            }])
        }
        IRInst::WritePtr { ptr, result } => {
            let p = resolve_operand(ptr, vreg_alloc, var_to_reg)?;
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "writeptr".to_string(),
                args: vec![p],
                ret: Some(dst),
            }])
        }
        IRInst::Read {
            buffer,
            size,
            result,
        } => {
            let buf = resolve_operand(buffer, vreg_alloc, var_to_reg)?;
            let sz = resolve_operand(size, vreg_alloc, var_to_reg)?;
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            Ok(vec![Instruction::Call {
                name: "read".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        IRInst::Alloc {
            result,
            alloc_type,
            allocated_ty,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);

            match alloc_type {
                AllocType::Stack => {
                    let storage = Register::Virtual(vreg_alloc.allocate_gpr());
                    if let Some(sz) = sizeof_ir_type(allocated_ty) {
                        let slots_needed = sz.div_ceil(8) as usize;
                        for _ in 1..slots_needed {
                            vreg_alloc.allocate_gpr();
                        }
                    }
                    // Lea computes address of storage's stack slot into dst
                    Ok(vec![Instruction::Lea {
                        dst: dst.clone(),
                        base: storage,
                        offset: 0,
                    }])
                }
                AllocType::Heap => {
                    if let Some(sz) = sizeof_ir_type(allocated_ty) {
                        Ok(vec![Instruction::Call {
                            name: "malloc".to_string(),
                            args: vec![Operand::Immediate(Immediate::I64(sz as i64))],
                            ret: Some(dst),
                        }])
                    } else {
                        Err(FromIRError::UnsupportedType)
                    }
                }
            }
        }
        IRInst::Dealloc { ptr } => {
            // Lower heap dealloc to a conventional call that backends can map
            let p = resolve_operand(ptr, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::Call {
                name: "dealloc".to_string(),
                args: vec![p],
                ret: None,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdBinary {
            op,
            result,
            vector_type,
            lhs,
            rhs,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_vec());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_type(vector_type)?;
            let lhs_op = resolve_operand(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = resolve_operand(rhs, vreg_alloc, var_to_reg)?;
            let mir_op = match op {
                crate::ir::instruction::SimdOp::Add => SimdOp::Add,
                crate::ir::instruction::SimdOp::Sub => SimdOp::Sub,
                crate::ir::instruction::SimdOp::Mul => SimdOp::Mul,
                crate::ir::instruction::SimdOp::Div => SimdOp::Div,
                crate::ir::instruction::SimdOp::Min => SimdOp::Min,
                crate::ir::instruction::SimdOp::Max => SimdOp::Max,
                crate::ir::instruction::SimdOp::Abs => SimdOp::Abs,
                crate::ir::instruction::SimdOp::Neg => SimdOp::Neg,
                crate::ir::instruction::SimdOp::Sqrt => SimdOp::Sqrt,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::SimdBinary {
                op: mir_op,
                ty: mir_ty,
                dst,
                lhs: lhs_op,
                rhs: rhs_op,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::SimdUnary {
            op,
            result,
            vector_type,
            operand,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_vec());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_type(vector_type)?;
            let src_op = resolve_operand(operand, vreg_alloc, var_to_reg)?;
            let mir_op = match op {
                crate::ir::instruction::SimdOp::Abs => SimdOp::Abs,
                crate::ir::instruction::SimdOp::Neg => SimdOp::Neg,
                crate::ir::instruction::SimdOp::Sqrt => SimdOp::Sqrt,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(vec![Instruction::SimdUnary {
                op: mir_op,
                ty: mir_ty,
                dst,
                src: src_op,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::AtomicLoad {
            result,
            ty,
            ptr,
            ordering,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg)?;
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => MemoryOrdering::Relaxed,
                crate::ir::instruction::MemoryOrdering::Acquire => MemoryOrdering::Acquire,
                crate::ir::instruction::MemoryOrdering::Release => MemoryOrdering::Release,
                crate::ir::instruction::MemoryOrdering::AcqRel => MemoryOrdering::AcqRel,
                crate::ir::instruction::MemoryOrdering::SeqCst => MemoryOrdering::SeqCst,
            };
            Ok(vec![Instruction::AtomicLoad {
                ty: mir_ty,
                dst,
                addr,
                ordering: mir_ordering,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        } => {
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg)?;
            let val_op = resolve_operand(value, vreg_alloc, var_to_reg)?;
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => MemoryOrdering::Relaxed,
                crate::ir::instruction::MemoryOrdering::Acquire => MemoryOrdering::Acquire,
                crate::ir::instruction::MemoryOrdering::Release => MemoryOrdering::Release,
                crate::ir::instruction::MemoryOrdering::AcqRel => MemoryOrdering::AcqRel,
                crate::ir::instruction::MemoryOrdering::SeqCst => MemoryOrdering::SeqCst,
            };
            Ok(vec![Instruction::AtomicStore {
                ty: mir_ty,
                src: val_op,
                addr,
                ordering: mir_ordering,
            }])
        }
        #[cfg(feature = "nightly")]
        IRInst::Fence { ordering } => {
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => MemoryOrdering::Relaxed,
                crate::ir::instruction::MemoryOrdering::Acquire => MemoryOrdering::Acquire,
                crate::ir::instruction::MemoryOrdering::Release => MemoryOrdering::Release,
                crate::ir::instruction::MemoryOrdering::AcqRel => MemoryOrdering::AcqRel,
                crate::ir::instruction::MemoryOrdering::SeqCst => MemoryOrdering::SeqCst,
            };
            Ok(vec![Instruction::Fence {
                ordering: mir_ordering,
            }])
        }
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn literal_to_immediate(l: &IRLit<'_>) -> Result<Immediate, FromIRError> {
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

fn reg_class_for_type(ty: &IRType<'_>) -> RegisterClass {
    match ty {
        IRType::Primitive(IRPrim::F32) | IRType::Primitive(IRPrim::F64) => RegisterClass::Fpr,
        _ => RegisterClass::Gpr,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::ir::builder::{i64 as ir_i64, string, var};
    use crate::ir::instruction::BinaryOp;
    use crate::ir::types::{PrimitiveType, Type};
    use crate::ir::{FunctionParameter, IRBuilder};

    #[test]
    fn test_from_ir_simple_add() {
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "add",
                vec![
                    FunctionParameter {
                        name: "a",
                        ty: Type::Primitive(PrimitiveType::I64),
                        annotations: vec![],
                    },
                    FunctionParameter {
                        name: "b",
                        ty: Type::Primitive(PrimitiveType::I64),
                        annotations: vec![],
                    },
                ],
                Type::Primitive(PrimitiveType::I64),
            )
            .binary(BinaryOp::Add, "sum", PrimitiveType::I64, var("a"), var("b"))
            .ret(Type::Primitive(PrimitiveType::I64), var("sum"));

        let ir_module = builder.build();
        let mir_module = from_ir(&ir_module, "test").expect("from_ir should succeed");
        let func = mir_module.get_function("add").expect("function exists");
        assert_eq!(func.sig.name, "add");
        assert_eq!(func.entry, "entry");
        let entry = func.get_block("entry").expect("entry block exists");
        assert_eq!(entry.instructions.len(), 2);
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
        match &entry.instructions[1] {
            Instruction::Ret { value } => {
                assert!(matches!(value, Some(Operand::Register(_))));
            }
            other => panic!("Unexpected second instruction: {:?}", other),
        }
    }

    #[test]
    fn test_print_numeric_succeeds() {
        let mut builder = IRBuilder::new();
        builder
            .function("main", Type::Primitive(PrimitiveType::I64))
            .print(ir_i64(42))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(0));

        let ir_module = builder.build();
        let result = from_ir(&ir_module, "test");
        assert!(
            result.is_ok(),
            "print with numeric literal should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_print_string_fails() {
        let mut builder = IRBuilder::new();
        builder
            .function("main", Type::Primitive(PrimitiveType::I64))
            .print(string("hello"))
            .ret(Type::Primitive(PrimitiveType::I64), ir_i64(0));

        let ir_module = builder.build();
        let result = from_ir(&ir_module, "test");
        assert!(
            matches!(result, Err(FromIRError::PrintStringLiteralUnsupported)),
            "print with string literal should fail with PrintStringLiteralUnsupported: {:?}",
            result
        );
    }

    #[test]
    fn test_print_string_rejected_text_ir() {
        let ir_source = r#"
fn @main() -> i64 {
  entry:
    print "x"
    ret.i64 0
}
"#;
        let ir_module = crate::parser::parse_module(ir_source).expect("parse should succeed");
        let result = from_ir(&ir_module, "test");
        assert!(
            matches!(result, Err(FromIRError::PrintStringLiteralUnsupported)),
            "print \"x\" in text IR should fail: {:?}",
            result
        );
        if let Err(e) = &result {
            let msg = format!("{}", e);
            assert!(
                msg.contains("writebyte"),
                "error should mention writebyte: {}",
                msg
            );
        }
    }
}
