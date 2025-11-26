use super::error::FromIRError;
use super::mapping::{map_ir_prim, map_ir_type};
use crate::mir::{
    AddressMode, Block, Instruction, MemoryAttrs, Module, Operand, Parameter, Register,
    RegisterClass, Signature, VirtualReg, VirtualRegAllocator,
};
use std::collections::HashSet;

pub fn from_ir(ir: &crate::ir::Module<'_>, name: &str) -> Result<Module, FromIRError> {
    let mut mir_module = Module::new(name);

    for (func_name, ir_func) in &ir.functions {
        let mir_func = convert_function(func_name, ir_func)?;
        mir_module.add_function(mir_func);
    }

    Ok(mir_module)
}

// Helper to convert an IR value into a MIR operand using the current bindings.
// If a variable binding is missing, allocate a conservative GPR to keep lowering progressing.
fn ir_value_to_mir_operand<'a>(
    v: &crate::ir::types::Value<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut std::collections::HashMap<&'a str, Register>,
) -> Result<Operand, FromIRError> {
    match v {
        crate::ir::types::Value::Variable(id) => {
            if let Some(r) = var_to_reg.get(id) {
                Ok(Operand::Register(r.clone()))
            } else {
                let r = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*id, r.clone());
                Ok(Operand::Register(r))
            }
        }
        crate::ir::types::Value::Constant(lit) => literal_to_immediate(lit).map(Operand::Immediate),
        crate::ir::types::Value::Global(_) => Err(FromIRError::UnsupportedInstruction),
    }
}

fn convert_function<'a>(
    name: &'a str,
    f: &crate::ir::function::Function<'a>,
) -> Result<crate::mir::Function, FromIRError> {
    use crate::ir::types::Type as IRType;

    let mut vreg_alloc = VirtualRegAllocator::new();
    let mut mir_sig = Signature::new(name);

    match &f.signature.return_type {
        IRType::Void => {}
        other => {
            let ty = map_ir_type(other)?;
            mir_sig = mir_sig.with_return(ty);
        }
    }

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

    let mut mir_func = crate::mir::Function::new(mir_sig).with_entry(f.entry_block);

    // BFS traversal starting from entry block, guard with visited to avoid cycles
    let mut visited: std::collections::HashSet<&'a str> = std::collections::HashSet::new();
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
            use crate::ir::instruction::Instruction as IRInst;
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
                _ => {}
            }
        }

        mir_func.add_block(mir_block);
    }

    // Lower SSA phi nodes by inserting edge-specific copy moves.
    // For conditional branches, split the edge with a trampoline block to preserve semantics.
    {
        let mut edge_moves: std::collections::HashMap<(String, String), Vec<Instruction>> =
            std::collections::HashMap::new();

        for (succ_label, ir_block) in &f.basic_blocks {
            for instr in &ir_block.instructions {
                if let crate::ir::instruction::Instruction::Phi {
                    result,
                    ty,
                    incoming,
                } = instr
                {
                    if let Some(dst_reg) = var_to_reg.get(result).cloned() {
                        let mir_ty = map_ir_type(ty)?;
                        for (val, pred_label) in incoming {
                            let src_op =
                                ir_value_to_mir_operand(val, &mut vreg_alloc, &mut var_to_reg)?;
                            // Encode a move using add with zero; the backend already handles it.
                            let mov_like = Instruction::IntBinary {
                                op: crate::mir::IntBinOp::Add,
                                ty: mir_ty,
                                dst: dst_reg.clone(),
                                lhs: src_op,
                                rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
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
                    // Now update the predecessor's branch target
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

    // Workaround for parser bug: add initializations for variables that are used but not defined
    // This happens when the Lamina parser fails to parse assignment instructions in complex functions
    add_missing_initializations(&mut mir_func);

    Ok(mir_func)
}

fn add_missing_initializations(func: &mut crate::mir::Function) {
    use crate::mir::{Immediate, Instruction as MirInst, MirType, Operand, Register, ScalarType};
    use std::collections::HashSet;

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
                op: crate::mir::IntBinOp::Add,
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

fn collect_regs_from_instruction(
    instr: &crate::mir::Instruction,
    used_regs: &mut HashSet<VirtualReg>,
) {
    match instr {
        crate::mir::Instruction::IntBinary { lhs, rhs, .. } => {
            collect_regs_from_operand(lhs, used_regs);
            collect_regs_from_operand(rhs, used_regs);
        }
        crate::mir::Instruction::FloatBinary { lhs, rhs, .. } => {
            collect_regs_from_operand(lhs, used_regs);
            collect_regs_from_operand(rhs, used_regs);
        }
        crate::mir::Instruction::IntCmp { lhs, rhs, .. } => {
            collect_regs_from_operand(lhs, used_regs);
            collect_regs_from_operand(rhs, used_regs);
        }
        crate::mir::Instruction::FloatCmp { lhs, rhs, .. } => {
            collect_regs_from_operand(lhs, used_regs);
            collect_regs_from_operand(rhs, used_regs);
        }
        crate::mir::Instruction::Load { addr, .. } => {
            collect_regs_from_address_mode(addr, used_regs);
        }
        crate::mir::Instruction::Store { addr, src, .. } => {
            collect_regs_from_address_mode(addr, used_regs);
            collect_regs_from_operand(src, used_regs);
        }
        crate::mir::Instruction::Call { args, .. } => {
            for arg in args {
                collect_regs_from_operand(arg, used_regs);
            }
        }
        crate::mir::Instruction::Ret { value: Some(val) } => {
            collect_regs_from_operand(val, used_regs);
        }
        crate::mir::Instruction::Jmp { .. } | crate::mir::Instruction::Br { .. } => {}
        _ => {} // Handle other instruction types we don't care about
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
    instr: &crate::ir::instruction::Instruction<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut std::collections::HashMap<&'a str, Register>,
) -> Result<Vec<Instruction>, FromIRError> {
    use crate::ir::instruction::{BinaryOp as IRBin, CmpOp as IRCmp};
    use crate::ir::types::{PrimitiveType as IRPrim, Type as IRType, Value as IRVal};

    fn get_operand_permissive<'b>(
        v: &IRVal<'b>,
        vreg_alloc: &mut VirtualRegAllocator,
        var_to_reg: &mut std::collections::HashMap<&'b str, Register>,
    ) -> Result<Operand, FromIRError> {
        match v {
            IRVal::Variable(id) => {
                if let Some(r) = var_to_reg.get(id) {
                    Ok(Operand::Register(r.clone()))
                } else {
                    // Fallback: allocate a GPR binding so lowering can proceed.
                    let r = Register::Virtual(vreg_alloc.allocate_gpr());
                    var_to_reg.insert(*id, r.clone());
                    Ok(Operand::Register(r))
                }
            }
            IRVal::Constant(lit) => literal_to_immediate(lit).map(Operand::Immediate),
            IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
        }
    }

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
        crate::ir::instruction::Instruction::Binary {
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
            let lhs_op = get_operand_permissive(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = get_operand_permissive(rhs, vreg_alloc, var_to_reg)?;

            let mir = match (op, *ty) {
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: crate::mir::FloatBinOp::FAdd,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: crate::mir::FloatBinOp::FSub,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: crate::mir::FloatBinOp::FMul,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary {
                    op: crate::mir::FloatBinOp::FDiv,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Add, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Add,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Sub, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Sub,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Mul, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Mul,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Div, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::SDiv,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Rem, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::SRem,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::And, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::And,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Or, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Or,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Xor, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Xor,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Shl, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Shl,
                    ty: mir_ty,
                    dst: dst.clone(),
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRBin::Shr, _) => Instruction::IntBinary {
                    op: crate::mir::IntBinOp::AShr,
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
        crate::ir::instruction::Instruction::Phi {
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
        crate::ir::instruction::Instruction::Cmp {
            op,
            result,
            ty,
            lhs,
            rhs,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = get_operand_permissive(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = get_operand_permissive(rhs, vreg_alloc, var_to_reg)?;
            let mir = match (*ty, op) {
                (IRPrim::F32 | IRPrim::F64, IRCmp::Eq) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ne) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Lt) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Lt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Le) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Le,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Gt) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Gt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ge) => Instruction::FloatCmp {
                    op: crate::mir::FloatCmpOp::Ge,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Eq) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::Eq,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ne) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::Ne,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Lt) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::ULt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Le) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::ULe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Gt) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::UGt,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ge) => {
                    Instruction::IntCmp {
                        op: crate::mir::IntCmpOp::UGe,
                        ty: mir_ty,
                        dst,
                        lhs: lhs_op,
                        rhs: rhs_op,
                    }
                }
                (_, IRCmp::Eq) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::Eq,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ne) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::Ne,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Lt) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::SLt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Le) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::SLe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Gt) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::SGt,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
                (_, IRCmp::Ge) => Instruction::IntCmp {
                    op: crate::mir::IntCmpOp::SGe,
                    ty: mir_ty,
                    dst,
                    lhs: lhs_op,
                    rhs: rhs_op,
                },
            };
            Ok(vec![mir])
        }
        crate::ir::instruction::Instruction::Br {
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
        crate::ir::instruction::Instruction::Switch {
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

            use crate::ir::types::Literal as IRLit;
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
                        return Err(FromIRError::UnsupportedType)
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
        crate::ir::instruction::Instruction::Jmp { target_label } => Ok(vec![Instruction::Jmp {
            target: (*target_label).to_string(),
        }]),
        crate::ir::instruction::Instruction::Ret { ty, value } => {
            if matches!(ty, IRType::Void) {
                return Ok(vec![Instruction::Ret { value: None }]);
            }
            let op = match value {
                Some(v) => Some(get_operand_permissive(v, vreg_alloc, var_to_reg)?),
                None => None,
            };
            Ok(vec![Instruction::Ret { value: op }])
        }
        crate::ir::instruction::Instruction::Load { result, ty, ptr } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
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
        crate::ir::instruction::Instruction::Store { ty, ptr, value } => {
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let src = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
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
        crate::ir::instruction::Instruction::Call {
            result,
            func_name,
            args,
        } => {
            let mut mir_args = Vec::new();
            for a in args {
                mir_args.push(get_operand_permissive(a, vreg_alloc, var_to_reg)?);
            }
            let ret = if let Some(res) = result {
                let dst = if let Some(existing) = var_to_reg.get(res) {
                    existing.clone()
                } else {
                    let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                    var_to_reg.insert(*res, fresh.clone());
                    fresh
                };
                Some(dst)
            } else {
                None
            };
            Ok(vec![Instruction::Call {
                name: (*func_name).to_string(),
                args: mir_args,
                ret,
            }])
        }
        crate::ir::instruction::Instruction::ZeroExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            // Lower zext as an AND with a mask of the source width.
            // This preserves semantics across sizes and maps cleanly to MIR.
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            // Compute mask based on source type bit-width
            use crate::ir::types::PrimitiveType as IRPrim;
            let (mask, mir_ty) = match (source_type, target_type) {
                (IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char, _) => {
                    (0xFFu64, map_ir_prim(*target_type)?)
                }
                (IRPrim::I16 | IRPrim::U16, _) => (0xFFFFu64, map_ir_prim(*target_type)?),
                (IRPrim::I32 | IRPrim::U32, _) => (0xFFFF_FFFFu64, map_ir_prim(*target_type)?),
                (
                    IRPrim::I64 | IRPrim::U64 | IRPrim::Ptr,
                    IRPrim::I64 | IRPrim::U64 | IRPrim::Ptr,
                ) => (0xFFFF_FFFF_FFFF_FFFFu64, map_ir_prim(*target_type)?),
                _ => (0xFFFF_FFFF_FFFF_FFFFu64, map_ir_prim(*target_type)?),
            };
            let val_op = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let mask_op = Operand::Immediate(crate::mir::Immediate::I64(mask as i64));
            Ok(vec![Instruction::IntBinary {
                op: crate::mir::IntBinOp::And,
                ty: mir_ty,
                dst,
                lhs: val_op,
                rhs: mask_op,
            }])
        }
        crate::ir::instruction::Instruction::Trunc {
            result,
            source_type: _,
            target_type,
            value,
        } => {
            // Lower truncation as an AND with a mask of the target width.
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            use crate::ir::types::PrimitiveType as IRPrim;
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
            let val_op = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let mask_op = Operand::Immediate(crate::mir::Immediate::I64(mask as i64));
            Ok(vec![Instruction::IntBinary {
                op: crate::mir::IntBinOp::And,
                ty: mir_ty,
                dst,
                lhs: val_op,
                rhs: mask_op,
            }])
        }
        crate::ir::instruction::Instruction::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            // Lower sext using a mask and shifts to propagate the sign bit.
            use crate::ir::types::PrimitiveType as IRPrim;

            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            // Determine source and target bit-widths
            fn bits_for(ty: &IRPrim) -> u32 {
                match ty {
                    IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 8,
                    IRPrim::I16 | IRPrim::U16 => 16,
                    IRPrim::I32 | IRPrim::U32 => 32,
                    IRPrim::I64 | IRPrim::U64 | IRPrim::Ptr => 64,
                    IRPrim::F32 | IRPrim::F64 => 0,
                }
            }

            let src_bits = bits_for(&source_type);
            let dst_bits = bits_for(&target_type);

            if src_bits == 0 || dst_bits == 0 || src_bits >= dst_bits {
                return Err(FromIRError::UnsupportedType);
            }

            let mir_ty = map_ir_prim(*target_type)?;
            let shift = (dst_bits - src_bits) as i64;

            let val_op = get_operand_permissive(value, vreg_alloc, var_to_reg)?;

            // temp = (val << shift)
            let temp_reg = Register::Virtual(vreg_alloc.allocate_gpr());
            let shift_imm = Operand::Immediate(crate::mir::Immediate::I64(shift));

            let mut instrs = Vec::new();

            instrs.push(Instruction::IntBinary {
                op: crate::mir::IntBinOp::Shl,
                ty: mir_ty.clone(),
                dst: temp_reg.clone(),
                lhs: val_op,
                rhs: shift_imm,
            });

            // dst = temp >> shift (arithmetic shift right propagates sign)
            instrs.push(Instruction::IntBinary {
                op: crate::mir::IntBinOp::AShr,
                ty: mir_ty,
                dst,
                lhs: Operand::Register(temp_reg),
                rhs: Operand::Immediate(crate::mir::Immediate::I64(shift)),
            });

            Ok(instrs)
        }
        crate::ir::instruction::Instruction::Bitcast {
            result,
            source_type,
            target_type,
            value,
        } => {
            use crate::ir::types::PrimitiveType as IRPrim;

            // Only support bitcasts between primitive types of equal size.
            fn bits_for(ty: &IRPrim) -> u32 {
                match ty {
                    IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 8,
                    IRPrim::I16 | IRPrim::U16 => 16,
                    IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 32,
                    IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 64,
                }
            }

            if bits_for(&source_type) != bits_for(&target_type) {
                return Err(FromIRError::UnsupportedType);
            }

            match value {
                IRVal::Variable(id) => {
                    // Reuse the same MIR register for the result; MIR is untyped and the
                    // backend will see the result type through later uses.
                    let src_reg = if let Some(existing) = var_to_reg.get(id) {
                        existing.clone()
                    } else {
                        let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                        var_to_reg.insert(*id, fresh.clone());
                        fresh
                    };
                    var_to_reg.insert(*result, src_reg);
                    Ok(vec![])
                }
                IRVal::Constant(lit) => {
                    // For now, restrict constant bitcasts to a few common cases.
                    let mir_ty = map_ir_prim(*target_type)?;
                    let dst = if let Some(existing) = var_to_reg.get(result) {
                        existing.clone()
                    } else {
                        let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                        var_to_reg.insert(*result, fresh.clone());
                        fresh
                    };

                    let imm = match (source_type, target_type, lit) {
                        (IRPrim::F32, IRPrim::I32, crate::ir::types::Literal::F32(v)) => {
                            crate::mir::Immediate::I32(v.to_bits() as i32)
                        }
                        (IRPrim::F64, IRPrim::I64, crate::ir::types::Literal::F64(v)) => {
                            crate::mir::Immediate::I64(v.to_bits() as i64)
                        }
                        (IRPrim::I32, IRPrim::F32, crate::ir::types::Literal::I32(v)) => {
                            crate::mir::Immediate::I32(*v)
                        }
                        (IRPrim::I64, IRPrim::F64, crate::ir::types::Literal::I64(v)) => {
                            crate::mir::Immediate::I64(*v)
                        }
                        _ => return Err(FromIRError::UnsupportedInstruction),
                    };

                    Ok(vec![Instruction::IntBinary {
                        op: crate::mir::IntBinOp::Add,
                        ty: mir_ty,
                        dst,
                        lhs: Operand::Immediate(imm),
                        rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
                    }])
                }
                IRVal::Global(_) => Err(FromIRError::UnsupportedInstruction),
            }
        }
        crate::ir::instruction::Instruction::Select {
            result,
            ty,
            cond,
            true_val,
            false_val,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            let mir_ty = map_ir_type(ty)?;

            // Condition must be a variable bound to a MIR register.
            let cond_reg = match cond {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };

            let tv = get_operand_permissive(true_val, vreg_alloc, var_to_reg)?;
            let fv = get_operand_permissive(false_val, vreg_alloc, var_to_reg)?;

            Ok(vec![Instruction::Select {
                dst,
                ty: mir_ty,
                cond: cond_reg,
                true_val: tv,
                false_val: fv,
            }])
        }
        crate::ir::instruction::Instruction::PtrToInt {
            result,
            ptr_value,
            target_type: _,
        } => {
            // On 64-bit, pointers are integers; lower to add with 0 to move value
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let val_op = get_operand_permissive(ptr_value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                ty: crate::mir::types::MirType::Scalar(crate::mir::types::ScalarType::I64),
                dst,
                lhs: val_op,
                rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
            }])
        }
        crate::ir::instruction::Instruction::IntToPtr {
            result,
            int_value,
            target_type: _,
        } => {
            // Treat as identity move via add with 0 (pointer-sized)
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let val_op = get_operand_permissive(int_value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntBinary {
                op: crate::mir::IntBinOp::Add,
                ty: crate::mir::types::MirType::Scalar(crate::mir::types::ScalarType::I64),
                dst,
                lhs: val_op,
                rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
            }])
        }
        crate::ir::instruction::Instruction::MemCpy { dst, src, size } => {
            let mut args = Vec::new();
            args.push(get_operand_permissive(dst, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(src, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(size, vreg_alloc, var_to_reg)?);
            Ok(vec![Instruction::Call {
                name: "memcpy".to_string(),
                args,
                ret: None,
            }])
        }
        crate::ir::instruction::Instruction::MemMove { dst, src, size } => {
            let mut args = Vec::new();
            args.push(get_operand_permissive(dst, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(src, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(size, vreg_alloc, var_to_reg)?);
            Ok(vec![Instruction::Call {
                name: "memmove".to_string(),
                args,
                ret: None,
            }])
        }
        crate::ir::instruction::Instruction::MemSet { dst, value, size } => {
            let mut args = Vec::new();
            args.push(get_operand_permissive(dst, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(value, vreg_alloc, var_to_reg)?);
            args.push(get_operand_permissive(size, vreg_alloc, var_to_reg)?);
            Ok(vec![Instruction::Call {
                name: "memset".to_string(),
                args,
                ret: None,
            }])
        }
        crate::ir::instruction::Instruction::GetFieldPtr {
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
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let offset = (*field_index as i32) * 8;
            Ok(vec![Instruction::Lea { dst, base, offset }])
        }
        crate::ir::instruction::Instruction::GetElemPtr {
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
                    crate::ir::types::Literal::I64(v) => Some(*v),
                    crate::ir::types::Literal::I32(v) => Some(*v as i64),
                    crate::ir::types::Literal::U64(v) => Some(*v as i64),
                    crate::ir::types::Literal::U32(v) => Some(*v as i64),
                    _ => None,
                },
                _ => None,
            };
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            if let Some(iv) = idx_const {
                // Constant index: use simple LEA with offset
                let offset = (iv as i32).saturating_mul(elem_size);
                Ok(vec![Instruction::Lea { dst, base, offset }])
            } else {
                // Variable index: emit two instructions
                // 1. temp = index * elem_size
                // 2. result = base + temp
                
                // Get the index operand
                let index_op = get_operand_permissive(index, vreg_alloc, var_to_reg)?;
                
                // Allocate a temp register for the scaled index
                let temp_scaled = Register::Virtual(vreg_alloc.allocate_gpr());
                
                // Create instructions
                let mut instrs = Vec::new();
                
                // Instruction 1: temp_scaled = index * elem_size
                instrs.push(Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Mul,
                    ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                    dst: temp_scaled.clone(),
                    lhs: index_op,
                    rhs: Operand::Immediate(crate::mir::Immediate::I64(elem_size as i64)),
                });
                
                // Instruction 2: result = base + temp_scaled
                instrs.push(Instruction::IntBinary {
                    op: crate::mir::IntBinOp::Add,
                    ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                    dst: dst.clone(),
                    lhs: Operand::Register(base),
                    rhs: Operand::Register(temp_scaled),
                });
                
                Ok(instrs)
            }
        }
        // Debug and I/O operations: lower to calls with conventional names
        crate::ir::instruction::Instruction::Print { value } => {
            let arg = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::Call {
                name: "print".to_string(),
                args: vec![arg],
                ret: None,
            }])
        }
        crate::ir::instruction::Instruction::Write {
            buffer,
            size,
            result,
        } => {
            let buf = get_operand_permissive(buffer, vreg_alloc, var_to_reg)?;
            let sz = get_operand_permissive(size, vreg_alloc, var_to_reg)?;
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            Ok(vec![Instruction::Call {
                name: "write".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        crate::ir::instruction::Instruction::WriteByte { value, result } => {
            let v = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            Ok(vec![Instruction::Call {
                name: "writebyte".to_string(),
                args: vec![v],
                ret: Some(dst),
            }])
        }
        crate::ir::instruction::Instruction::ReadByte { result } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            Ok(vec![Instruction::Call {
                name: "readbyte".to_string(),
                args: vec![],
                ret: Some(dst),
            }])
        }
        crate::ir::instruction::Instruction::WritePtr { ptr, result } => {
            let p = get_operand_permissive(ptr, vreg_alloc, var_to_reg)?;
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            Ok(vec![Instruction::Call {
                name: "writeptr".to_string(),
                args: vec![p],
                ret: Some(dst),
            }])
        }
        crate::ir::instruction::Instruction::Read {
            buffer,
            size,
            result,
        } => {
            let buf = get_operand_permissive(buffer, vreg_alloc, var_to_reg)?;
            let sz = get_operand_permissive(size, vreg_alloc, var_to_reg)?;
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            Ok(vec![Instruction::Call {
                name: "read".to_string(),
                args: vec![buf, sz],
                ret: Some(dst),
            }])
        }
        crate::ir::instruction::Instruction::Alloc {
            result,
            alloc_type,
            allocated_ty,
        } => {
            use crate::ir::types::PrimitiveType as IRPrim;
            // Allocate a destination vreg representing the pointer
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };

            match alloc_type {
                crate::ir::instruction::AllocType::Stack => {
                    // For stack allocation: create storage vreg(s) based on type size
                    // For arrays/structs, we need multiple vregs to reserve enough stack space
                    fn sizeof_ty(ty: &crate::ir::types::Type<'_>) -> Option<u64> {
                        use crate::ir::types::Type as IRType;
                        match ty {
                            IRType::Primitive(p) => Some(match p {
                                IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 1,
                                IRPrim::I16 | IRPrim::U16 => 2,
                                IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 4,
                                IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 8,
                            }),
                            IRType::Array { element_type, size } => {
                                sizeof_ty(element_type).map(|es| es.saturating_mul(*size))
                            }
                            IRType::Struct(fields) => {
                                let mut total = 0u64;
                                for f in fields {
                                    if let Some(sz) = sizeof_ty(&f.ty) {
                                        total = total.saturating_add(sz);
                                    } else {
                                        return None;
                                    }
                                }
                                Some(total)
                            }
                            _ => None,
                        }
                    }
                    // Allocate storage vreg; for larger types, we still use one vreg
                    // The frame allocator will give it enough stack space
                    let storage = Register::Virtual(vreg_alloc.allocate_gpr());
                    // Compute size to ensure frame map accounts for it (store as metadata if needed)
                    // For now, each vreg gets 8 bytes; arrays/structs need contiguous slots
                    // Simple approach: allocate (size/8) vregs to reserve space
                    if let Some(sz) = sizeof_ty(allocated_ty) {
                        let slots_needed = sz.div_ceil(8) as usize;
                        // Allocate additional dummy vregs to reserve stack space
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
                crate::ir::instruction::AllocType::Heap => {
                    // Compute size in bytes for the allocated type and call malloc(size)
                    fn sizeof_ty(ty: &crate::ir::types::Type<'_>) -> Option<u64> {
                        use crate::ir::types::Type as IRType;
                        match ty {
                            IRType::Primitive(p) => Some(match p {
                                IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 1,
                                IRPrim::I16 | IRPrim::U16 => 2,
                                IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 4,
                                IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 8,
                            }),
                            IRType::Array { element_type, size } => {
                                sizeof_ty(element_type).map(|es| es.saturating_mul(*size))
                            }
                            IRType::Struct(fields) => {
                                let mut total = 0u64;
                                for f in fields {
                                    if let Some(sz) = sizeof_ty(&f.ty) {
                                        total = total.saturating_add(sz);
                                    } else {
                                        return None;
                                    }
                                }
                                Some(total)
                            }
                            _ => None,
                        }
                    }
                    if let Some(sz) = sizeof_ty(allocated_ty) {
                        Ok(vec![Instruction::Call {
                            name: "malloc".to_string(),
                            args: vec![Operand::Immediate(crate::mir::Immediate::I64(sz as i64))],
                            ret: Some(dst),
                        }])
                    } else {
                        Err(FromIRError::UnsupportedType)
                    }
                }
            }
        }
        crate::ir::instruction::Instruction::Dealloc { ptr } => {
            // Lower heap dealloc to a conventional call that backends can map
            let p = get_operand_permissive(ptr, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::Call {
                name: "dealloc".to_string(),
                args: vec![p],
                ret: None,
            }])
        }
        #[cfg(feature = "nightly")]
        crate::ir::instruction::Instruction::SimdBinary {
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
            let lhs_op = get_operand_permissive(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = get_operand_permissive(rhs, vreg_alloc, var_to_reg)?;
            let mir_op = match op {
                crate::ir::instruction::SimdOp::Add => crate::mir::SimdOp::Add,
                crate::ir::instruction::SimdOp::Sub => crate::mir::SimdOp::Sub,
                crate::ir::instruction::SimdOp::Mul => crate::mir::SimdOp::Mul,
                crate::ir::instruction::SimdOp::Div => crate::mir::SimdOp::Div,
                crate::ir::instruction::SimdOp::Min => crate::mir::SimdOp::Min,
                crate::ir::instruction::SimdOp::Max => crate::mir::SimdOp::Max,
                crate::ir::instruction::SimdOp::Abs => crate::mir::SimdOp::Abs,
                crate::ir::instruction::SimdOp::Neg => crate::mir::SimdOp::Neg,
                crate::ir::instruction::SimdOp::Sqrt => crate::mir::SimdOp::Sqrt,
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
        crate::ir::instruction::Instruction::SimdUnary {
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
            let src_op = get_operand_permissive(operand, vreg_alloc, var_to_reg)?;
            let mir_op = match op {
                crate::ir::instruction::SimdOp::Abs => crate::mir::SimdOp::Abs,
                crate::ir::instruction::SimdOp::Neg => crate::mir::SimdOp::Neg,
                crate::ir::instruction::SimdOp::Sqrt => crate::mir::SimdOp::Sqrt,
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
        crate::ir::instruction::Instruction::AtomicLoad {
            result,
            ty,
            ptr,
            ordering,
        } => {
            let dst = if let Some(existing) = var_to_reg.get(result) {
                existing.clone()
            } else {
                let fresh = Register::Virtual(vreg_alloc.allocate_gpr());
                var_to_reg.insert(*result, fresh.clone());
                fresh
            };
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg)?;
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => {
                    crate::mir::MemoryOrdering::Relaxed
                }
                crate::ir::instruction::MemoryOrdering::Acquire => {
                    crate::mir::MemoryOrdering::Acquire
                }
                crate::ir::instruction::MemoryOrdering::Release => {
                    crate::mir::MemoryOrdering::Release
                }
                crate::ir::instruction::MemoryOrdering::AcqRel => {
                    crate::mir::MemoryOrdering::AcqRel
                }
                crate::ir::instruction::MemoryOrdering::SeqCst => {
                    crate::mir::MemoryOrdering::SeqCst
                }
            };
            Ok(vec![Instruction::AtomicLoad {
                ty: mir_ty,
                dst,
                addr,
                ordering: mir_ordering,
            }])
        }
        #[cfg(feature = "nightly")]
        crate::ir::instruction::Instruction::AtomicStore {
            ty,
            ptr,
            value,
            ordering,
        } => {
            let mir_ty = map_ir_type(ty)?;
            let addr = ir_address_mode_to_mir(ptr, vreg_alloc, var_to_reg)?;
            let val_op = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => {
                    crate::mir::MemoryOrdering::Relaxed
                }
                crate::ir::instruction::MemoryOrdering::Acquire => {
                    crate::mir::MemoryOrdering::Acquire
                }
                crate::ir::instruction::MemoryOrdering::Release => {
                    crate::mir::MemoryOrdering::Release
                }
                crate::ir::instruction::MemoryOrdering::AcqRel => {
                    crate::mir::MemoryOrdering::AcqRel
                }
                crate::ir::instruction::MemoryOrdering::SeqCst => {
                    crate::mir::MemoryOrdering::SeqCst
                }
            };
            Ok(vec![Instruction::AtomicStore {
                ty: mir_ty,
                src: val_op,
                addr,
                ordering: mir_ordering,
            }])
        }
        #[cfg(feature = "nightly")]
        crate::ir::instruction::Instruction::Fence { ordering } => {
            let mir_ordering = match ordering {
                crate::ir::instruction::MemoryOrdering::Relaxed => {
                    crate::mir::MemoryOrdering::Relaxed
                }
                crate::ir::instruction::MemoryOrdering::Acquire => {
                    crate::mir::MemoryOrdering::Acquire
                }
                crate::ir::instruction::MemoryOrdering::Release => {
                    crate::mir::MemoryOrdering::Release
                }
                crate::ir::instruction::MemoryOrdering::AcqRel => {
                    crate::mir::MemoryOrdering::AcqRel
                }
                crate::ir::instruction::MemoryOrdering::SeqCst => {
                    crate::mir::MemoryOrdering::SeqCst
                }
            };
            Ok(vec![Instruction::Fence {
                ordering: mir_ordering,
            }])
        }
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn literal_to_immediate(
    l: &crate::ir::types::Literal<'_>,
) -> Result<crate::mir::Immediate, FromIRError> {
    use crate::ir::types::Literal as IRLit;
    let imm = match l {
        IRLit::I8(v) => crate::mir::Immediate::I8(*v),
        IRLit::I16(v) => crate::mir::Immediate::I16(*v),
        IRLit::I32(v) => crate::mir::Immediate::I32(*v),
        IRLit::I64(v) => crate::mir::Immediate::I64(*v),
        IRLit::U8(v) => crate::mir::Immediate::I8(*v as i8),
        IRLit::U16(v) => crate::mir::Immediate::I16(*v as i16),
        IRLit::U32(v) => crate::mir::Immediate::I32(*v as i32),
        IRLit::U64(v) => crate::mir::Immediate::I64(*v as i64),
        IRLit::F32(v) => crate::mir::Immediate::F32(*v),
        IRLit::F64(v) => crate::mir::Immediate::F64(*v),
        IRLit::Bool(b) => crate::mir::Immediate::I8(if *b { 1 } else { 0 }),
        IRLit::Char(c) => crate::mir::Immediate::I8(*c as i8),
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
    use crate::mir::MirType;

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
                assert!(matches!(op, crate::mir::IntBinOp::Add));
                assert!(matches!(
                    ty,
                    MirType::Scalar(crate::mir::types::ScalarType::I64)
                ));
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
}
