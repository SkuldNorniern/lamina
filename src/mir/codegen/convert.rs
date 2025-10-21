use super::error::FromIRError;
use super::mapping::{map_ir_prim, map_ir_type};
use crate::mir::{
    AddressMode, Block, Instruction, MemoryAttrs, MirType, Operand, Parameter, Register,
    RegisterClass, Signature, VirtualRegAllocator, Module,
};

pub fn from_ir(ir: &crate::ir::Module<'_>, name: &str) -> Result<Module, FromIRError> {
    let mut mir_module = Module::new(name);

    for (func_name, ir_func) in &ir.functions {
        let mir_func = convert_function(func_name, ir_func)?;
        mir_module.add_function(mir_func);
    }

    Ok(mir_module)
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
                Ok(Some(mir_instr)) => mir_block.push(mir_instr),
                Ok(None) => {}
                Err(e) => return Err(e),
            }
        }

        // After converting this block, enqueue IR successors (prevents borrow/move issues)
        if let Some(term) = ir_block.instructions.last() {
            use crate::ir::instruction::Instruction as IRInst;
            match term {
                IRInst::Br { true_label, false_label, .. } => {
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
                    if !visited.contains(target_label) && f.basic_blocks.contains_key(target_label) {
                        visited.insert(target_label);
                        worklist.push_back(target_label);
                    }
                }
                _ => {}
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

    fn get_operand_permissive<'a>(
        v: &IRVal<'a>,
        vreg_alloc: &mut VirtualRegAllocator,
        var_to_reg: &mut std::collections::HashMap<&'a str, Register>,
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

    match instr {
        crate::ir::instruction::Instruction::Binary { op, result, ty, lhs, rhs } => {
            let dst = match *ty {
                IRPrim::F32 | IRPrim::F64 => Register::Virtual(vreg_alloc.allocate_fpr()),
                _ => Register::Virtual(vreg_alloc.allocate_gpr()),
            };
            var_to_reg.insert(*result, dst.clone());

            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = get_operand_permissive(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = get_operand_permissive(rhs, vreg_alloc, var_to_reg)?;

            let mir = match (op, *ty) {
                (IRBin::Add, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary { op: crate::mir::FloatBinOp::FAdd, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Sub, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary { op: crate::mir::FloatBinOp::FSub, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Mul, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary { op: crate::mir::FloatBinOp::FMul, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Div, IRPrim::F32 | IRPrim::F64) => Instruction::FloatBinary { op: crate::mir::FloatBinOp::FDiv, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Add, _) => Instruction::IntBinary { op: crate::mir::IntBinOp::Add, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Sub, _) => Instruction::IntBinary { op: crate::mir::IntBinOp::Sub, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Mul, _) => Instruction::IntBinary { op: crate::mir::IntBinOp::Mul, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRBin::Div, _) => Instruction::IntBinary { op: crate::mir::IntBinOp::SDiv, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
            };
            Ok(Some(mir))
        }
        // SSA merge: create a binding for the phi result so subsequent uses resolve.
        // Semantics are not materialized here; a later SSA elimination pass should lower this.
        crate::ir::instruction::Instruction::Phi { result, ty, incoming: _ } => {
            // Choose register class based on type
            let class = reg_class_for_type(ty);
            let dst = match class {
                RegisterClass::Fpr => Register::Virtual(vreg_alloc.allocate_fpr()),
                RegisterClass::Vec => Register::Virtual(vreg_alloc.allocate_vec()),
                _ => Register::Virtual(vreg_alloc.allocate_gpr()),
            };
            var_to_reg.insert(*result, dst);
            Ok(None)
        }
        crate::ir::instruction::Instruction::Cmp { op, result, ty, lhs, rhs } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            let mir_ty = map_ir_prim(*ty)?;
            let lhs_op = get_operand_permissive(lhs, vreg_alloc, var_to_reg)?;
            let rhs_op = get_operand_permissive(rhs, vreg_alloc, var_to_reg)?;
            let mir = match (*ty, op) {
                (IRPrim::F32 | IRPrim::F64, IRCmp::Eq) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Eq, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ne) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Ne, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Lt) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Lt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Le) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Le, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Gt) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Gt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::F32 | IRPrim::F64, IRCmp::Ge) => Instruction::FloatCmp { op: crate::mir::FloatCmpOp::Ge, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Eq) => Instruction::IntCmp { op: crate::mir::IntCmpOp::Eq, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ne) => Instruction::IntCmp { op: crate::mir::IntCmpOp::Ne, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Lt) => Instruction::IntCmp { op: crate::mir::IntCmpOp::ULt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Le) => Instruction::IntCmp { op: crate::mir::IntCmpOp::ULe, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Gt) => Instruction::IntCmp { op: crate::mir::IntCmpOp::UGt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (IRPrim::U8 | IRPrim::U16 | IRPrim::U32 | IRPrim::U64, IRCmp::Ge) => Instruction::IntCmp { op: crate::mir::IntCmpOp::UGe, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Eq) => Instruction::IntCmp { op: crate::mir::IntCmpOp::Eq, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Ne) => Instruction::IntCmp { op: crate::mir::IntCmpOp::Ne, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Lt) => Instruction::IntCmp { op: crate::mir::IntCmpOp::SLt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Le) => Instruction::IntCmp { op: crate::mir::IntCmpOp::SLe, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Gt) => Instruction::IntCmp { op: crate::mir::IntCmpOp::SGt, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
                (_, IRCmp::Ge) => Instruction::IntCmp { op: crate::mir::IntCmpOp::SGe, ty: mir_ty, dst, lhs: lhs_op, rhs: rhs_op },
            };
            Ok(Some(mir))
        }
        crate::ir::instruction::Instruction::Br { condition, true_label, false_label } => {
            let cond_reg = match condition {
                IRVal::Variable(id) => var_to_reg.get(id).cloned().ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            Ok(Some(Instruction::Br {
                cond: cond_reg,
                true_target: (*true_label).to_string(),
                false_target: (*false_label).to_string(),
            }))
        }
        crate::ir::instruction::Instruction::Jmp { target_label } => {
            Ok(Some(Instruction::Jmp { target: (*target_label).to_string() }))
        }
        crate::ir::instruction::Instruction::Ret { ty, value } => {
            if matches!(ty, IRType::Void) { return Ok(Some(Instruction::Ret { value: None })); }
            let op = match value { Some(v) => Some(get_operand_permissive(v, vreg_alloc, var_to_reg)?), None => None };
            Ok(Some(Instruction::Ret { value: op }))
        }
        crate::ir::instruction::Instruction::Load { result, ty, ptr } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg.get(id).cloned().ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(Some(Instruction::Load { ty: mir_ty, dst, addr, attrs: MemoryAttrs { align: mir_ty.alignment() as u8, volatile: false } }))
        }
        crate::ir::instruction::Instruction::Store { ty, ptr, value } => {
            let mir_ty = map_ir_type(ty)?;
            let base = match ptr {
                IRVal::Variable(id) => var_to_reg.get(id).cloned().ok_or(FromIRError::UnknownVariable)?,
                _ => return Err(FromIRError::UnsupportedInstruction),
            };
            let src = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let addr = AddressMode::BaseOffset { base, offset: 0 };
            Ok(Some(Instruction::Store { ty: mir_ty, src, addr, attrs: MemoryAttrs { align: mir_ty.alignment() as u8, volatile: false } }))
        }
        crate::ir::instruction::Instruction::Call { result, func_name, args } => {
            let mut mir_args = Vec::new();
            for a in args { mir_args.push(get_operand_permissive(a, vreg_alloc, var_to_reg)?); }
            let ret = if let Some(res) = result { let dst = Register::Virtual(vreg_alloc.allocate_gpr()); var_to_reg.insert(*res, dst.clone()); Some(dst) } else { None };
            Ok(Some(Instruction::Call { name: (*func_name).to_string(), args: mir_args, ret }))
        }
        // Debug and I/O operations: lower to calls with conventional names
        crate::ir::instruction::Instruction::Print { value } => {
            let arg = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            Ok(Some(Instruction::Call { name: "print".to_string(), args: vec![arg], ret: None }))
        }
        crate::ir::instruction::Instruction::Write { buffer, size, result } => {
            let buf = get_operand_permissive(buffer, vreg_alloc, var_to_reg)?;
            let sz = get_operand_permissive(size, vreg_alloc, var_to_reg)?;
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            Ok(Some(Instruction::Call { name: "write".to_string(), args: vec![buf, sz], ret: Some(dst) }))
        }
        crate::ir::instruction::Instruction::WriteByte { value, result } => {
            let v = get_operand_permissive(value, vreg_alloc, var_to_reg)?;
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            Ok(Some(Instruction::Call { name: "writebyte".to_string(), args: vec![v], ret: Some(dst) }))
        }
        crate::ir::instruction::Instruction::ReadByte { result } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            Ok(Some(Instruction::Call { name: "readbyte".to_string(), args: vec![], ret: Some(dst) }))
        }
        crate::ir::instruction::Instruction::WritePtr { ptr, result } => {
            let p = get_operand_permissive(ptr, vreg_alloc, var_to_reg)?;
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            Ok(Some(Instruction::Call { name: "writeptr".to_string(), args: vec![p], ret: Some(dst) }))
        }
        crate::ir::instruction::Instruction::Read { buffer, size, result } => {
            let buf = get_operand_permissive(buffer, vreg_alloc, var_to_reg)?;
            let sz = get_operand_permissive(size, vreg_alloc, var_to_reg)?;
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst.clone());
            Ok(Some(Instruction::Call { name: "read".to_string(), args: vec![buf, sz], ret: Some(dst) }))
        }
        crate::ir::instruction::Instruction::Alloc { result, .. } => {
            let dst = Register::Virtual(vreg_alloc.allocate_gpr());
            var_to_reg.insert(*result, dst);
            Ok(None)
        }
        _ => Err(FromIRError::UnsupportedInstruction),
    }
}

fn literal_to_immediate(l: &crate::ir::types::Literal<'_>) -> Result<crate::mir::Immediate, FromIRError> {
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

    #[test]
    fn test_from_ir_simple_add() {
        let mut builder = IRBuilder::new();
        builder
            .function_with_params(
                "add",
                vec![
                    FunctionParameter { name: "a", ty: Type::Primitive(PrimitiveType::I64) },
                    FunctionParameter { name: "b", ty: Type::Primitive(PrimitiveType::I64) },
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
            Instruction::IntBinary { op, ty, dst: _, lhs, rhs } => {
                assert!(matches!(op, crate::mir::IntBinOp::Add));
                assert!(matches!(ty, MirType::Scalar(crate::mir::types::ScalarType::I64)));
                assert!(matches!(lhs, Operand::Register(_)));
                assert!(matches!(rhs, Operand::Register(_)));
            }
            other => panic!("Unexpected first instruction: {:?}", other),
        }
        match &entry.instructions[1] { Instruction::Ret { value } => { assert!(matches!(value, Some(Operand::Register(_)))); } other => panic!("Unexpected second instruction: {:?}", other) }
    }
}


