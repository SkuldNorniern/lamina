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

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    ir::{
        Module as IRModule,
        function::{Function as IRFunction, FunctionAnnotation},
        instruction::{AllocType, BinaryOp as IRBin, CmpOp as IRCmp, Instruction as IRInst},
        types::{
            Literal as IRLit, PrimitiveType as IRPrim, Type as IRType, Value as IRVal,
            resolve_type_names, sizeof_type,
        },
    },
    mir::{
        AddressMode, Block, FloatBinOp, FloatCmpOp, Function, Immediate, Instruction, IntBinOp,
        IntCmpOp, MemoryAttrs, MirType, Module, Operand, Parameter, Register, RegisterClass,
        ScalarType, Signature, VirtualReg, VirtualRegAllocator,
        codegen::{
            error::FromIRError,
            mapping::{map_ir_prim, map_ir_type},
        },
        verify::{PhiVerification, verify_before_phi_lowering},
    },
};

#[cfg(feature = "nightly")]
use crate::ir::instruction::{MemoryOrdering as IRMemOrd, SimdOp as IRSimdOp};
#[cfg(feature = "nightly")]
use crate::mir::{MemoryOrdering, SimdOp};

fn float_binary(
    op: FloatBinOp,
    ty: MirType,
    dst: Register,
    lhs: Operand,
    rhs: Operand,
) -> Instruction {
    Instruction::FloatBinary {
        op,
        ty,
        dst,
        lhs,
        rhs,
    }
}

/// Map an IR binary op to its MIR opcode, choosing signed vs. unsigned variants
/// based on the operand type. Float-applicable ops on float types are handled
/// separately by the caller before this function is reached.
fn lower_int_binop(op: &IRBin, ty: IRPrim) -> IntBinOp {
    let unsigned = matches!(ty, IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64);
    match op {
        IRBin::Add => IntBinOp::Add,
        IRBin::Sub => IntBinOp::Sub,
        IRBin::Mul => IntBinOp::Mul,
        IRBin::Div => {
            if unsigned {
                IntBinOp::UDiv
            } else {
                IntBinOp::SDiv
            }
        }
        IRBin::Rem => {
            if unsigned {
                IntBinOp::URem
            } else {
                IntBinOp::SRem
            }
        }
        IRBin::And => IntBinOp::And,
        IRBin::Or => IntBinOp::Or,
        IRBin::Xor => IntBinOp::Xor,
        IRBin::Shl => IntBinOp::Shl,
        // Arithmetic shift for signed types, logical shift for unsigned types.
        IRBin::Shr => {
            if unsigned {
                IntBinOp::LShr
            } else {
                IntBinOp::AShr
            }
        }
    }
}

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

/// Bit-width for bitcast: F32 = 32 bits, F64 = 64 bits (valid bitcast widths).
fn int_bits_for_bitcast(ty: &IRPrim) -> u32 {
    match ty {
        IRPrim::I8 | IRPrim::U8 | IRPrim::Bool | IRPrim::Char => 8,
        IRPrim::I16 | IRPrim::U16 => 16,
        IRPrim::I32 | IRPrim::U32 | IRPrim::F32 => 32,
        IRPrim::I64 | IRPrim::U64 | IRPrim::F64 | IRPrim::Ptr => 64,
    }
}

fn normalize_lowered_literal(operand: &mut Operand, ty: MirType) {
    operand.normalize_integer_immediate(ty);
    if ty == MirType::f32()
        && let Operand::Immediate(Immediate::F64(value)) = operand
    {
        *operand = Operand::imm_f32(*value as f32);
    }
}

pub fn from_ir(ir: &IRModule<'_>, name: &str) -> Result<Module, FromIRError> {
    let mut mir_module = Module::new(name);

    for (func_name, ir_func) in &ir.functions {
        // Check if function is external before converting
        if ir_func.annotations.contains(&FunctionAnnotation::Extern) {
            mir_module.mark_external(*func_name);
        }
        if ir_func.annotations.contains(&FunctionAnnotation::Export) {
            mir_module.mark_exported(*func_name);
        }

        let mir_func = convert_function(func_name, ir_func, ir)?;
        mir_module.add_function(mir_func);
    }

    mir_module
        .validate_in_pipeline()
        .map_err(FromIRError::InvalidMir)?;
    Ok(mir_module)
}

fn convert_function<'a>(
    name: &'a str,
    f: &IRFunction<'a>,
    module: &IRModule<'a>,
) -> Result<Function, FromIRError> {
    let mut vreg_alloc = VirtualRegAllocator::new();
    let mut mir_sig = Signature::new(name);
    if f.annotations.contains(&FunctionAnnotation::Kernel) {
        mir_sig.kernel = Some(Default::default());
    }

    match &f.signature.return_type {
        IRType::Void => {}
        other => {
            let ty = map_ir_type(other)?;
            mir_sig = mir_sig.with_return(ty);
        }
    }

    let mut var_to_reg: HashMap<&'a str, Register> = HashMap::new();
    let mut register_types = HashMap::new();
    let mut stack_allocs: HashSet<&'a str> = HashSet::new();
    let mut reservations: HashMap<VirtualReg, usize> = HashMap::new();

    for param in &f.signature.params {
        let class = reg_class_for_type(&param.ty);
        let v = match class {
            RegisterClass::Fpr => Register::Virtual(vreg_alloc.allocate_fpr()),
            RegisterClass::Vec => Register::Virtual(vreg_alloc.allocate_vec()),
            _ => Register::Virtual(vreg_alloc.allocate_gpr()),
        };
        let mir_ty = map_ir_type(&param.ty)?;
        register_types.insert(v.clone(), mir_ty);
        mir_sig.params.push(Parameter::new(v.clone(), mir_ty));
        var_to_reg.insert(param.name, v);
    }

    let mut mir_func = Function::new(mir_sig).with_entry(f.entry_block);

    let mut names: HashSet<&str> = f.signature.params.iter().map(|p| p.name).collect();
    let mut reassigned_names = HashSet::new();
    for block in f.basic_blocks.values() {
        for instruction in &block.instructions {
            if let Some(name) = instruction.defined_var()
                && !names.insert(name)
            {
                reassigned_names.insert(name);
            }
        }
    }

    // BFS traversal starting from entry block, guard with visited to avoid cycles
    let mut visited: HashSet<&'a str> = HashSet::new();
    let mut worklist: VecDeque<&'a str> = VecDeque::new();

    // Ensure entry exists
    if !f.basic_blocks.contains_key(f.entry_block) {
        return Err(FromIRError::MissingEntryBlock);
    }
    worklist.push_back(f.entry_block);
    visited.insert(f.entry_block);

    while let Some(label) = worklist.pop_front() {
        let ir_block = &f.basic_blocks[label];
        let mut mir_block = Block::new(label);

        // Registers already defined in this block. A scan of the emitted instructions per
        // instruction made conversion quadratic in the size of a single-block function.
        let mut defined_in_block: HashSet<Register> = HashSet::new();
        for instr in &ir_block.instructions {
            let mut resolved;
            let instr = if matches!(instr, IRInst::Alloc { .. } | IRInst::GetElemPtr { .. }) {
                resolved = instr.clone();
                let ty = match &mut resolved {
                    IRInst::Alloc { allocated_ty, .. } => allocated_ty,
                    IRInst::GetElemPtr { element_type, .. } => element_type,
                    _ => unreachable!(),
                };
                *ty = resolve_type_names(ty, &module.type_declarations)
                    .ok_or(FromIRError::UnsupportedType)?;
                &resolved
            } else {
                instr
            };
            let mut mir_instrs = convert_instruction(
                instr,
                &mut vreg_alloc,
                &mut var_to_reg,
                &mut stack_allocs,
                &mut reservations,
                &register_types,
            )?;
            if f.basic_blocks.len() == 1
                && let IRInst::Binary { result, .. } | IRInst::WriteByte { result, .. } = instr
                && let Some(last) = mir_instrs.last_mut()
                && let Some(old_dst) = last.def_reg().cloned()
                && defined_in_block.contains(&old_dst)
            {
                let fresh = Register::Virtual(match &old_dst {
                    Register::Virtual(vreg) if vreg.class == RegisterClass::Fpr => {
                        vreg_alloc.allocate_fpr()
                    }
                    _ => vreg_alloc.allocate_gpr(),
                });
                match last {
                    Instruction::IntBinary { dst, .. } | Instruction::FloatBinary { dst, .. } => {
                        *dst = fresh.clone()
                    }
                    Instruction::Call { ret, .. } => *ret = Some(fresh.clone()),
                    _ => return Err(FromIRError::UnsupportedInstruction),
                }
                var_to_reg.insert(result, fresh);
            }
            for mut mir_instr in mir_instrs {
                if let Instruction::Call { name, args, .. } = &mut mir_instr
                    && let Some(callee) = module.functions.get(name.as_str())
                {
                    for (argument, parameter) in args.iter_mut().zip(&callee.signature.params) {
                        let ty = map_ir_type(&parameter.ty)?;
                        normalize_lowered_literal(argument, ty);
                    }
                }
                mir_instr.normalize_integer_immediates(mir_func.sig.ret_ty);
                if let Some(dst) = mir_instr.def_reg() {
                    let ty = match &mir_instr {
                        Instruction::IntConvert { to, .. } | Instruction::Bitcast { to, .. } => {
                            Some(*to)
                        }
                        Instruction::Copy { ty, .. }
                        | Instruction::IntBinary { ty, .. }
                        | Instruction::FloatBinary { ty, .. }
                        | Instruction::FloatUnary { ty, .. }
                        | Instruction::Select { ty, .. }
                        | Instruction::Load { ty, .. } => Some(*ty),
                        Instruction::IntCmp { .. } | Instruction::FloatCmp { .. } => {
                            Some(MirType::i1())
                        }
                        Instruction::Call { name, .. } => module
                            .functions
                            .get(name.as_str())
                            .and_then(|callee| map_ir_type(&callee.signature.return_type).ok()),
                        Instruction::Lea { .. } | Instruction::StackAddr { .. } => {
                            Some(MirType::ptr())
                        }
                        _ => None,
                    };
                    if let Some(ty) = ty {
                        register_types.insert(dst.clone(), ty);
                    }
                }
                if let Some(dst) = mir_instr.def_reg() {
                    defined_in_block.insert(dst.clone());
                }
                mir_block.push(mir_instr);
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
        let mut phis = Vec::new();
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
                        let mut phi = PhiVerification {
                            block: (*succ_label).to_string(),
                            dst: dst_reg.clone(),
                            ty: mir_ty,
                            incoming: Vec::new(),
                        };
                        for (val, pred_label) in incoming {
                            if !visited.contains(pred_label) {
                                continue;
                            }
                            let src_op = resolve_operand(val, &mut vreg_alloc, &mut var_to_reg)?;
                            let mut mov_like = Instruction::Copy {
                                ty: mir_ty,
                                dst: dst_reg.clone(),
                                src: src_op,
                            };
                            mov_like.normalize_integer_immediates(None);
                            if let Instruction::Copy { src, .. } = &mov_like {
                                phi.incoming.push(((*pred_label).to_string(), src.clone()));
                            }
                            edge_moves
                                .entry(((*pred_label).to_string(), (*succ_label).to_string()))
                                .or_default()
                                .push(mov_like);
                        }
                        phis.push(phi);
                    } else {
                        // Bind destination conservatively to avoid cascading unknowns
                        let new_dst = Register::Virtual(vreg_alloc.allocate_gpr());
                        var_to_reg.insert(*result, new_dst);
                    }
                }
            }
        }

        verify_before_phi_lowering(
            &mir_func,
            &phis,
            &reassigned_names
                .iter()
                .filter_map(|name| var_to_reg.get(name).cloned())
                .collect(),
        )
        .map_err(FromIRError::InvalidMir)?;

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

    mir_func.stack_reservations = reservations;

    Ok(mir_func)
}

fn convert_instruction<'a>(
    instr: &IRInst<'a>,
    vreg_alloc: &mut VirtualRegAllocator,
    var_to_reg: &mut HashMap<&'a str, Register>,
    stack_allocs: &mut HashSet<&'a str>,
    reservations: &mut HashMap<VirtualReg, usize>,
    register_types: &HashMap<Register, MirType>,
) -> Result<Vec<Instruction>, FromIRError> {
    #[cfg(feature = "nightly")]
    fn ir_address_mode_to_mir<'b>(
        ptr: &IRVal<'b>,
        vreg_alloc: &mut VirtualRegAllocator,
        var_to_reg: &mut HashMap<&'b str, Register>,
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
                Ok(AddressMode::BaseOffset {
                    base,
                    offset: 0,
                    address_space: Default::default(),
                })
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
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FAdd, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FSub, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FMul, mir_ty, dst, lhs_op, rhs_op)
                }
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => {
                    float_binary(FloatBinOp::FDiv, mir_ty, dst, lhs_op, rhs_op)
                }
                (op, prim_ty) => Instruction::IntBinary {
                    op: lower_int_binop(op, prim_ty),
                    ty: mir_ty,
                    dst,
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
            let addr = AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            };
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
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg)?;
            normalize_lowered_literal(&mut src, mir_ty);
            let addr = AddressMode::BaseOffset {
                base,
                offset: 0,
                address_space: Default::default(),
            };
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
        }
        | IRInst::Trunc {
            result,
            source_type,
            target_type,
            value,
        }
        | IRInst::SignExtend {
            result,
            source_type,
            target_type,
            value,
        } => {
            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let signed = matches!(instr, IRInst::SignExtend { .. });
            if from.is_float() || to.is_float() || (signed && from.size_bytes() >= to.size_bytes())
            {
                return Err(FromIRError::UnsupportedType);
            }
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let src = resolve_operand(value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntConvert {
                from,
                to,
                signed,
                dst,
                src,
            }])
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

            let from = map_ir_prim(*source_type)?;
            let to = map_ir_prim(*target_type)?;
            let mut src = resolve_operand(value, vreg_alloc, var_to_reg)?;
            normalize_lowered_literal(&mut src, from);
            let dst =
                Register::Virtual(match reg_class_for_type(&IRType::Primitive(*target_type)) {
                    RegisterClass::Fpr => vreg_alloc.allocate_fpr(),
                    _ => vreg_alloc.allocate_gpr(),
                });
            var_to_reg.insert(result, dst.clone());
            Ok(vec![Instruction::Bitcast { from, to, dst, src }])
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

            let mut tv = resolve_operand(true_val, vreg_alloc, var_to_reg)?;
            let mut fv = resolve_operand(false_val, vreg_alloc, var_to_reg)?;
            normalize_lowered_literal(&mut tv, mir_ty);
            normalize_lowered_literal(&mut fv, mir_ty);

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
            target_type,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(ptr_value, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::IntConvert {
                from: MirType::ptr(),
                to: map_ir_prim(*target_type)?,
                signed: false,
                dst,
                src: val_op,
            }])
        }
        IRInst::IntToPtr {
            result,
            int_value,
            target_type: _,
        } => {
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            let val_op = resolve_operand(int_value, vreg_alloc, var_to_reg)?;
            let from = operand_type(&val_op, register_types);
            Ok(vec![Instruction::IntConvert {
                from,
                to: MirType::ptr(),
                signed: false,
                dst,
                src: val_op,
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
            field_byte_offset,
        } => {
            let base = match struct_ptr {
                IRVal::Variable(id) => var_to_reg
                    .get(id)
                    .cloned()
                    .ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let dst = resolve_or_alloc_gpr(result, vreg_alloc, var_to_reg);
            // Use the pre-computed ABI-correct offset when available; otherwise
            // fall back to a fixed 8-byte-per-field stride (preserves behaviour
            // for IR parsed from text without type information).
            let offset = field_byte_offset
                .map(|b| b as i32)
                .unwrap_or((*field_index as i32) * 8);
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
            let elem_size = sizeof_type(element_type)
                .and_then(|size| i64::try_from(size).ok())
                .ok_or(FromIRError::UnsupportedInstruction)?;
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
                let offset = iv.wrapping_mul(elem_size);
                if let Ok(offset) = i32::try_from(offset) {
                    return Ok(vec![Instruction::Lea { dst, base, offset }]);
                }
                let address = Register::Virtual(vreg_alloc.allocate_gpr());
                let sum = Register::Virtual(vreg_alloc.allocate_gpr());
                Ok(vec![
                    Instruction::IntConvert {
                        from: MirType::ptr(),
                        to: MirType::i64(),
                        signed: false,
                        dst: address.clone(),
                        src: Operand::Register(base),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: MirType::i64(),
                        dst: sum.clone(),
                        lhs: Operand::Register(address),
                        rhs: Operand::Immediate(Immediate::I64(offset)),
                    },
                    Instruction::IntConvert {
                        from: MirType::i64(),
                        to: MirType::ptr(),
                        signed: false,
                        dst,
                        src: Operand::Register(sum),
                    },
                ])
            } else {
                let index_op = resolve_operand(index, vreg_alloc, var_to_reg)?;
                let from = operand_type(&index_op, register_types);
                let wide_index = Register::Virtual(vreg_alloc.allocate_gpr());
                let scaled = Register::Virtual(vreg_alloc.allocate_gpr());
                let address = Register::Virtual(vreg_alloc.allocate_gpr());
                let sum = Register::Virtual(vreg_alloc.allocate_gpr());
                Ok(vec![
                    Instruction::IntConvert {
                        from,
                        to: MirType::i64(),
                        signed: from.size_bytes() < 8,
                        dst: wide_index.clone(),
                        src: index_op,
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Mul,
                        ty: MirType::i64(),
                        dst: scaled.clone(),
                        lhs: Operand::Register(wide_index),
                        rhs: Operand::Immediate(Immediate::I64(elem_size)),
                    },
                    Instruction::IntConvert {
                        from: MirType::ptr(),
                        to: MirType::i64(),
                        signed: false,
                        dst: address.clone(),
                        src: Operand::Register(base),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: MirType::i64(),
                        dst: sum.clone(),
                        lhs: Operand::Register(address),
                        rhs: Operand::Register(scaled),
                    },
                    Instruction::IntConvert {
                        from: MirType::i64(),
                        to: MirType::ptr(),
                        signed: false,
                        dst,
                        src: Operand::Register(sum),
                    },
                ])
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

            let size = sizeof_type(allocated_ty).ok_or(FromIRError::UnsupportedType)?;
            let size = i64::try_from(size).map_err(|_| FromIRError::UnsupportedType)?;
            match alloc_type {
                AllocType::Stack => {
                    stack_allocs.insert(result);
                    let storage_vreg = vreg_alloc.allocate_gpr();
                    let slots_needed = usize::try_from((size as u64).div_ceil(8))
                        .map_err(|_| FromIRError::UnsupportedType)?;
                    if slots_needed > 1 {
                        reservations.insert(storage_vreg, slots_needed - 1);
                    }
                    Ok(vec![Instruction::StackAddr {
                        dst,
                        slot: storage_vreg,
                    }])
                }
                AllocType::Heap => Ok(vec![Instruction::Call {
                    name: "malloc".to_string(),
                    args: vec![Operand::Immediate(Immediate::I64(size))],
                    ret: Some(dst),
                }]),
            }
        }
        IRInst::Dealloc { ptr } => {
            // dealloc.heap on a stack allocation is a no-op; passing a stack
            // address to free() aborts.
            if let IRVal::Variable(name) = ptr
                && stack_allocs.contains(name)
            {
                return Ok(Vec::new());
            }
            let p = resolve_operand(ptr, vreg_alloc, var_to_reg)?;
            Ok(vec![Instruction::Call {
                name: "free".to_string(),
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
                IRSimdOp::Add => SimdOp::Add,
                IRSimdOp::Sub => SimdOp::Sub,
                IRSimdOp::Mul => SimdOp::Mul,
                IRSimdOp::Div => SimdOp::Div,
                IRSimdOp::Min => SimdOp::Min,
                IRSimdOp::Max => SimdOp::Max,
                IRSimdOp::Abs => SimdOp::Abs,
                IRSimdOp::Neg => SimdOp::Neg,
                IRSimdOp::Sqrt => SimdOp::Sqrt,
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
                IRSimdOp::Abs => SimdOp::Abs,
                IRSimdOp::Neg => SimdOp::Neg,
                IRSimdOp::Sqrt => SimdOp::Sqrt,
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
                IRMemOrd::Relaxed => MemoryOrdering::Relaxed,
                IRMemOrd::Acquire => MemoryOrdering::Acquire,
                IRMemOrd::Release => MemoryOrdering::Release,
                IRMemOrd::AcqRel => MemoryOrdering::AcqRel,
                IRMemOrd::SeqCst => MemoryOrdering::SeqCst,
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
                IRMemOrd::Relaxed => MemoryOrdering::Relaxed,
                IRMemOrd::Acquire => MemoryOrdering::Acquire,
                IRMemOrd::Release => MemoryOrdering::Release,
                IRMemOrd::AcqRel => MemoryOrdering::AcqRel,
                IRMemOrd::SeqCst => MemoryOrdering::SeqCst,
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
                IRMemOrd::Relaxed => MemoryOrdering::Relaxed,
                IRMemOrd::Acquire => MemoryOrdering::Acquire,
                IRMemOrd::Release => MemoryOrdering::Release,
                IRMemOrd::AcqRel => MemoryOrdering::AcqRel,
                IRMemOrd::SeqCst => MemoryOrdering::SeqCst,
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

fn operand_type(operand: &Operand, types: &HashMap<Register, MirType>) -> MirType {
    match operand {
        Operand::Register(register) => types.get(register).copied().unwrap_or(MirType::i64()),
        Operand::Immediate(Immediate::I8(_)) => MirType::Scalar(ScalarType::I8),
        Operand::Immediate(Immediate::I16(_)) => MirType::Scalar(ScalarType::I16),
        Operand::Immediate(Immediate::I32(_)) => MirType::i32(),
        _ => MirType::i64(),
    }
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
    use crate::mir::TransformPipeline;
    use crate::{
        ir::{
            FunctionParameter, IRBuilder,
            builder::{i64 as ir_i64, string, var},
        },
        parser::parse_module,
    };

    #[test]
    fn select_literals_match_declared_widths() {
        for (ty, left, right) in [("i64", "11", "22"), ("f32", "1.5", "2.5")] {
            let source = format!(
                "fn @f(bool %c) -> {ty} {{ entry: %s = select.{ty} %c, {left}, {right} ret.{ty} %s }}"
            );
            let ir = parse_module(&source).expect("parse select");
            let module = from_ir(&ir, "select").expect("lower typed literals");
            assert!(module.validate_in_pipeline().is_ok());
        }
    }

    #[test]
    fn bitcasts_preserve_source_bindings_and_define_distinct_results() {
        let ir = parse_module("fn @f(f64 %x) -> f64 { entry: %bits = bitcast.f64.i64 %x %back = bitcast.i64.f64 %bits ret.f64 %back }").expect("parse bitcasts");
        let module = from_ir(&ir, "bitcasts").expect("lower bitcasts");
        let function = &module.functions["f"];
        let instructions = &function.blocks[0].instructions;
        let Instruction::Bitcast { from, to, dst, src } = &instructions[0] else {
            panic!("expected bitcast")
        };
        assert_eq!((*from, *to), (MirType::f64(), MirType::i64()));
        assert_eq!(src, &Operand::reg(function.sig.params[0].reg.clone()));
        assert_ne!(dst, &function.sig.params[0].reg);
        let Instruction::Bitcast { dst: back, .. } = &instructions[1] else {
            panic!("expected round trip")
        };
        assert_ne!(dst, back);
        assert!(module.validate_in_pipeline().is_ok());
    }

    #[test]
    fn lowering_rejects_undefined_and_forward_uses_before_initialization() {
        for source in [
            "fn @f() -> i64 { entry: %x = add.i64 %missing, 1 ret.i64 %x }",
            "fn @f() -> i64 { entry: %x = add.i64 %later, 1 %later = add.i64 1, 2 ret.i64 %x }",
        ] {
            let ir = parse_module(source).expect("parse invalid definitions");
            assert!(matches!(
                from_ir(&ir, "invalid"),
                Err(FromIRError::InvalidMir(_))
            ));
        }
    }

    #[test]
    fn composite_gep_scales_constant_and_variable_indices() {
        for (element_type, stride) in [("[3 x i32]", 12), ("struct { x: i64, y: i64 }", 16)] {
            let source = format!(
                "fn @gep(ptr %p, i64 %index) -> ptr {{
entry:
%constant = getelem.ptr %p, 2, {element_type}
%variable = getelem.ptr %constant, %index, {element_type}
ret.ptr %variable
}}"
            );
            let ir = parse_module(&source).expect("parse composite elements");
            let entry = &ir.functions["gep"].basic_blocks["entry"];
            assert_eq!(
                entry.instructions[0].to_string(),
                format!("%constant = getelem.ptr %p, 2, {element_type}")
            );
            let mir = from_ir(&ir, "gep").expect("lower composite elements");
            mir.validate().expect("composite addresses verify");
            let instructions =
                &mir.get_function("gep").expect("gep function").blocks[0].instructions;
            assert!(instructions.iter().any(|inst| matches!(
                inst,
                Instruction::Lea { offset, .. } if i64::from(*offset) == 2 * stride
            )));
            assert!(instructions.iter().any(|inst| matches!(
                inst,
                Instruction::IntBinary {
                    op: IntBinOp::Mul,
                    rhs: Operand::Immediate(Immediate::I64(size)),
                    ..
                } if *size == stride
            )));
        }
    }

    #[test]
    fn constant_gep_preserves_pointer_width_offsets() {
        for (index, expected) in [
            (4294967297_i64, 34359738376_i64),
            (536870912, 4294967296),
            (-536870913, -4294967304),
            (i64::MIN, 0),
        ] {
            let source = format!(
                "fn @gep(ptr %p) -> ptr {{\nentry:\n%q = getelem.ptr %p, {index}, i64\nret.ptr %q\n}}"
            );
            let ir = parse_module(&source).expect("parse wide index");
            let mir = from_ir(&ir, "gep").expect("lower wide index");
            mir.validate().expect("address arithmetic verifies");
            let function = mir.get_function("gep").expect("gep function");
            let offset = function.blocks[0]
                .instructions
                .iter()
                .find_map(|inst| match inst {
                    Instruction::Lea { offset, .. } => Some(i64::from(*offset)),
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty,
                        rhs: Operand::Immediate(Immediate::I64(offset)),
                        ..
                    } if *ty == MirType::i64() => Some(*offset),
                    _ => None,
                });
            assert_eq!(offset, Some(expected), "index {index}");
        }
    }

    #[test]
    fn arithmetic_optimizations_preserve_immediate_widths() {
        let source = "fn @optimized(i32 %x) -> i32 {\nentry:\n%c = add.i32 1, 2\n%m = mul.i32 %x, 4\n%r = rem.u32 %x, 8\n%a = add.i32 %m, %r\n%b = add.i32 %a, %c\nret.i32 %b\n}";
        let ir = parse_module(source).expect("parse arithmetic");
        for level in 0..=3 {
            let mut mir = from_ir(&ir, "optimized").expect("lower arithmetic");
            TransformPipeline::default_for_opt_level(level)
                .apply_to_module(&mut mir)
                .expect("optimize arithmetic");
            mir.validate().expect("optimized widths match consumers");
        }
    }

    #[test]
    fn conversions_and_variable_addresses_verify() {
        let ir = parse_module(
            "fn @convert(ptr %p, i32 %index, i8 %byte) -> ptr {\nentry:\n%wide = zext.i8.i64 %byte\n%signed = sext.i8.i64 %byte\n%narrow = trunc.i64.i32 %wide\n%address = ptrtoint %p, .i64\n%back = inttoptr %address, .i64\n%element = getelem.ptr %back, %index, i32\nret.ptr %element\n}",
        ).expect("parse conversions");
        let mir = from_ir(&ir, "conversions").expect("lower conversions");
        mir.validate().expect("conversion types match definitions");
    }

    #[test]
    fn straight_line_reassignment_reads_previous_definition() {
        let ir = parse_module(
            "fn @sum() -> i64 {\nentry:\n%total = add.i64 1, 2\n%total = add.i64 %total, 3\n%total = add.i64 %total, 4\nret.i64 %total\n}",
        ).expect("parse reassignments");
        let mir = from_ir(&ir, "sum").expect("lower reassignments");
        mir.validate()
            .expect("reassignments have unique definitions");
        let function = mir.get_function("sum").expect("sum function");
        let instructions = &function.blocks[0].instructions;
        for pair in instructions[..3].windows(2) {
            let Instruction::IntBinary {
                lhs: Operand::Register(source),
                dst,
                ..
            } = &pair[1]
            else {
                panic!("expected dependent add");
            };
            assert_eq!(Some(source), pair[0].def_reg());
            assert_ne!(source, dst);
        }
    }

    #[test]
    fn padded_struct_allocations_use_abi_size() {
        let ir = parse_module(
            "fn @sizes() -> ptr {\nentry:\n%stack = alloc.stack struct { a: i8, b: i64, c: i8 }\n%heap = alloc.heap [2 x struct { a: i8, b: i64, c: i8 }]\nret.ptr %heap\n}",
        ).expect("parse padded allocations");
        let mir = from_ir(&ir, "sizes").expect("lower padded allocations");
        let function = mir.get_function("sizes").expect("sizes function");
        let Instruction::StackAddr { slot, .. } = &function.blocks[0].instructions[0] else {
            panic!("expected stack allocation");
        };
        assert_eq!(function.stack_reservations[slot], 2);
        let Instruction::Call { args, .. } = &function.blocks[0].instructions[1] else {
            panic!("expected heap allocation");
        };
        assert_eq!(args, &[Operand::Immediate(Immediate::I64(48))]);
    }

    #[test]
    fn stack_storage_is_not_a_register_use() {
        let ir =
            parse_module("fn @stack() -> ptr {\nentry:\n%p = alloc.stack [24 x i8]\nret.ptr %p\n}")
                .expect("parse stack allocation");
        let mir = from_ir(&ir, "stack").expect("lower stack allocation");
        mir.validate().expect("stack allocation verifies");
        let function = mir.get_function("stack").expect("stack function");
        let instruction = &function.blocks[0].instructions[0];
        let Instruction::StackAddr { slot, .. } = instruction else {
            panic!("expected stack address");
        };
        assert!(instruction.use_regs().is_empty());
        assert_eq!(function.stack_reservations[slot], 2);
    }

    #[test]
    fn integer_immediates_follow_consuming_type() {
        for ty in ["i8", "i16", "i32", "i64"] {
            let source = format!(
                "fn @check(ptr %p) -> {ty} {{\nentry:\n%x = add.{ty} 1, -1\n%c = eq.{ty} %x, 0\nstore.{ty} %p, 7\nret.{ty} 0\n}}"
            );
            let ir = parse_module(&source).expect("parse typed operands");
            let mir = from_ir(&ir, "widths").expect("lower typed operands");
            mir.validate().expect("immediates match consuming types");
        }
    }

    #[test]
    fn test_from_ir_simple_add() {
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "add",
                vec![
                    FunctionParameter {
                        name: "a",
                        ty: IRType::Primitive(IRPrim::I64),
                        annotations: vec![],
                    },
                    FunctionParameter {
                        name: "b",
                        ty: IRType::Primitive(IRPrim::I64),
                        annotations: vec![],
                    },
                ],
                IRType::Primitive(IRPrim::I64),
            )
            .binary(IRBin::Add, "sum", IRPrim::I64, var("a"), var("b"))
            .ret(IRType::Primitive(IRPrim::I64), var("sum"));

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
    fn f64_phi_immediate_survives_lowering_as_copy() {
        let negative_zero = IRVal::Constant(IRLit::F64(-0.0));
        let other = IRVal::Constant(IRLit::F64(3.5));
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "choose",
                vec![FunctionParameter {
                    name: "condition",
                    ty: IRType::Primitive(IRPrim::I8),
                    annotations: vec![],
                }],
                IRType::Primitive(IRPrim::F64),
            )
            .branch(var("condition"), "left", "right")
            .block("left")
            .jump("merge")
            .block("right")
            .jump("merge")
            .block("merge")
            .phi(
                "value",
                IRType::Primitive(IRPrim::F64),
                vec![(negative_zero, "left"), (other, "right")],
            )
            .ret(IRType::Primitive(IRPrim::F64), var("value"));

        let mir_module = from_ir(&builder.build(), "test").expect("from_ir should succeed");
        let function = mir_module
            .get_function("choose")
            .expect("function should exist");
        assert!(
            function
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .any(|instruction| matches!(
                    instruction,
                    Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::F64),
                        src: Operand::Immediate(Immediate::F64(value)),
                        ..
                    } if value.to_bits() == (-0.0f64).to_bits()
                ))
        );
    }

    #[test]
    fn test_print_numeric_succeeds() {
        let mut builder = IRBuilder::new();
        builder
            .function("main", IRType::Primitive(IRPrim::I64))
            .print(ir_i64(42))
            .ret(IRType::Primitive(IRPrim::I64), ir_i64(0));

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
            .function("main", IRType::Primitive(IRPrim::I64))
            .print(string("hello"))
            .ret(IRType::Primitive(IRPrim::I64), ir_i64(0));

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
        let ir_module = parse_module(ir_source).expect("parse should succeed");
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
