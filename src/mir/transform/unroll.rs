use std::collections::{HashMap, HashSet};

use super::loop_opt::{LoopInfo, LoopInvariantCodeMotion};
use crate::mir::{
    Block, Function, Immediate, Instruction, IntBinOp, IntCmpOp, MirType, Operand, Register,
    ScalarType, VirtualReg,
    transform::{
        CfgSimplify, DeadCodeElimination, Peephole, Transform, TransformCategory, TransformError,
        TransformLevel,
    },
};

#[derive(Default)]
pub struct LoopUnrolling;

impl Transform for LoopUnrolling {
    fn name(&self) -> &'static str {
        "loop_unrolling"
    }
    fn description(&self) -> &'static str {
        "Unroll bounded counting loops with value renaming"
    }
    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }
    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }
    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        if !func.blocks.iter().any(|b| {
            matches!(
                b.instructions.last(),
                Some(
                    Instruction::Jmp { .. } | Instruction::Br { .. } | Instruction::TailCall { .. }
                )
            )
        }) {
            return Ok(false);
        }
        let initial_size: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
        if func.blocks.len() > 256 || initial_size > 8192 {
            return Ok(false);
        }
        let materialized = if func.blocks.iter().any(|b| matches!(b.instructions.last(), Some(Instruction::TailCall { name, .. }) if name == &func.sig.name)) {
            let saved = func.clone();
            if !materialize_tail_loop(func) { return Ok(false); }
            if !LoopInvariantCodeMotion.find_loops(func).iter().any(|info| analyze(func, info).is_some()) {
                *func = saved;
                return Ok(false);
            }
            true
        } else { false };
        let mut changed = false;
        for _ in 0..16 {
            let size: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
            if size >= initial_size + 1024 {
                break;
            }
            let candidate = LoopInvariantCodeMotion
                .find_loops(func)
                .iter()
                .find_map(|info| {
                    let shape = analyze(func, info)?;
                    let plan = choose_plan(func, &shape)?;
                    Some((shape, plan))
                });
            let Some((shape, (count, factor))) = candidate else {
                let loops = LoopInvariantCodeMotion.find_loops(func);
                let guarded = loops
                    .iter()
                    .find(|info| guarded_candidate(func, info, &loops));
                let Some(info) = guarded else {
                    break;
                };
                let Some(mut next) = register_base(func) else {
                    break;
                };
                let before = func.clone();
                clone_guarded(func, info, guarded_factor(func, info), &mut next);
                if func
                    .blocks
                    .iter()
                    .map(|b| b.instructions.len())
                    .sum::<usize>()
                    > initial_size + 1024
                {
                    *func = before;
                    break;
                }
                changed = true;
                continue;
            };
            let before = func.clone();
            if let Some(count) = count {
                let Some(mut next) = register_base(func) else {
                    break;
                };
                full_unroll(func, &shape, count, &mut next);
            } else {
                let Some(mut next) = register_base(func) else {
                    break;
                };
                partial_unroll(func, &shape, factor, &mut next);
            }
            if func
                .blocks
                .iter()
                .map(|b| b.instructions.len())
                .sum::<usize>()
                > initial_size + 1024
            {
                *func = before;
                break;
            }
            changed = true;
        }
        if changed || materialized {
            DeadCodeElimination.apply(func)?;
            Peephole.apply(func)?;
            CfgSimplify.apply(func)?;
        }
        Ok(changed || materialized)
    }
}

#[derive(Clone)]
struct Shape {
    header: String,
    body: Vec<Instruction>,
    labels: HashSet<String>,
    iv: Register,
    end: Operand,
    inclusive: bool,
    entry: String,
    live_out: HashSet<Register>,
}

fn analyze(func: &Function, info: &LoopInfo) -> Option<Shape> {
    if info.header.contains("__unroll_") {
        return None;
    }
    if info.blocks.len() < 2 || info.blocks.len() > 8 {
        return None;
    }
    let header = func.get_block(&info.header)?;
    let [
        Instruction::IntCmp {
            op,
            ty: MirType::Scalar(ScalarType::I64),
            dst,
            lhs: Operand::Register(iv),
            rhs: end,
        },
        Instruction::Br {
            cond,
            true_target,
            false_target,
        },
    ] = header.instructions.as_slice()
    else {
        return None;
    };
    if cond != dst
        || iv == dst
        || end
            .as_register()
            .is_some_and(|r| r == dst || !r.is_virtual())
        || func.blocks.iter().any(|b| {
            b.label != info.header && b.instructions.iter().any(|i| i.use_regs().contains(&dst))
        })
    {
        return None;
    }
    let (first, inclusive) = match op {
        IntCmpOp::SLt if !info.blocks.contains(false_target) => (true_target, false),
        IntCmpOp::SLe if !info.blocks.contains(false_target) => (true_target, true),
        IntCmpOp::SGe if !info.blocks.contains(true_target) => (false_target, false),
        IntCmpOp::SGt if !info.blocks.contains(true_target) => (false_target, true),
        _ => return None,
    };
    let outside: Vec<_> = func
        .blocks
        .iter()
        .filter(|b| {
            !info.blocks.contains(&b.label) && b.successors().contains(&info.header.as_str())
        })
        .collect();
    let [entry] = outside.as_slice() else {
        return None;
    };
    if entry.label.contains("__unroll_") {
        return None;
    }
    let mut labels = HashSet::from([info.header.clone()]);
    let mut body = Vec::new();
    let mut current = first.clone();
    while current != info.header {
        if !info.blocks.contains(&current) || !labels.insert(current.clone()) {
            return None;
        }
        let block = func.get_block(&current)?;
        let (Instruction::Jmp { target }, instructions) = block.instructions.split_last()? else {
            return None;
        };
        for i in instructions {
            if !matches!(
                i,
                Instruction::IntBinary {
                    ty: MirType::Scalar(ScalarType::I64),
                    ..
                } | Instruction::Copy {
                    ty: MirType::Scalar(ScalarType::I64),
                    ..
                }
            ) {
                return None;
            }
            if i.def_reg()
                .into_iter()
                .chain(i.use_regs())
                .any(|r| !r.is_virtual())
            {
                return None;
            }
        }
        body.extend_from_slice(instructions);
        current = target.clone();
    }
    if labels != info.blocks || body.len() > 48 {
        return None;
    }
    for b in &func.blocks {
        if !labels.contains(&b.label)
            && b.successors()
                .iter()
                .any(|s| labels.contains(*s) && *s != info.header)
        {
            return None;
        }
    }
    let mut offsets = HashMap::from([(iv.clone(), 0_i64)]);
    for i in &body {
        let offset = match i {
            Instruction::Copy {
                src: Operand::Register(src),
                ..
            } => offsets.get(src).copied(),
            Instruction::IntBinary {
                op: IntBinOp::Add,
                lhs: Operand::Register(src),
                rhs,
                ..
            } => offsets
                .get(src)
                .copied()
                .zip(rhs.integer_constant())
                .and_then(|(x, y)| x.checked_add(y)),
            _ => None,
        };
        if let Some(dst) = i.def_reg() {
            offsets.remove(dst);
            if let Some(offset) = offset {
                offsets.insert(dst.clone(), offset);
            }
        }
    }
    if offsets.get(iv) != Some(&1) {
        return None;
    }
    if body.iter().any(|i| {
        i.def_reg() == Some(dst) || end.as_register().is_some_and(|r| i.def_reg() == Some(r))
    }) {
        return None;
    }
    let mut live_out: HashSet<Register> = func
        .blocks
        .iter()
        .filter(|b| !labels.contains(&b.label) || b.label == info.header)
        .flat_map(|b| &b.instructions)
        .flat_map(Instruction::use_regs)
        .cloned()
        .collect();
    let mut defined = HashSet::new();
    for i in &body {
        for r in i.use_regs() {
            if !defined.contains(r) {
                live_out.insert(r.clone());
            }
        }
        if let Some(r) = i.def_reg() {
            defined.insert(r.clone());
        }
    }
    live_out.retain(|r| defined.contains(r));
    live_out.extend(
        body.iter()
            .flat_map(Instruction::use_regs)
            .filter(|r| !defined.contains(*r))
            .cloned(),
    );
    Some(Shape {
        header: info.header.clone(),
        body,
        labels,
        iv: iv.clone(),
        end: end.clone(),
        inclusive,
        entry: entry.label.clone(),
        live_out,
    })
}

fn choose_plan(func: &Function, shape: &Shape) -> Option<(Option<u64>, usize)> {
    let count = constant_trip_count(func, shape);
    if let Some(n) =
        count.filter(|n| *n <= 32 && shape.body.len().saturating_mul(*n as usize) <= 192)
    {
        return Some((Some(n), 0));
    }
    if shape.body.iter().any(|i| matches!(i, Instruction::IntBinary { op: IntBinOp::Mul, dst, lhs, rhs, .. } if lhs.as_register() == Some(dst) || rhs.as_register() == Some(dst))) {
        return None;
    }
    let work = shape.body.iter().filter(|i| !matches!(i, Instruction::Copy { .. }) && !matches!(i, Instruction::IntBinary { op: IntBinOp::Add, rhs, .. } if rhs.integer_constant() == Some(0))).count();
    let mut factor = match work {
        0..=4 => 8,
        5..=8 => 4,
        9..=16 => 2,
        _ => return None,
    };
    while factor > 1
        && (shape.body.len() * factor > 192 || count.is_some_and(|n| n < factor as u64))
    {
        factor /= 2;
    }
    (factor >= 2).then_some((None, factor))
}

fn guarded_factor(func: &Function, info: &LoopInfo) -> usize {
    let size: usize = func
        .blocks
        .iter()
        .filter(|b| info.blocks.contains(&b.label))
        .map(|b| b.instructions.len())
        .sum();
    if size <= 12 { 4 } else { 2 }
}

fn guarded_candidate(func: &Function, info: &LoopInfo, loops: &[LoopInfo]) -> bool {
    if info.header.contains("__unroll_")
        || info.blocks.len() > 8
        || loops
            .iter()
            .any(|other| other.header != info.header && info.blocks.contains(&other.header))
    {
        return false;
    }
    let mut size = 0;
    let mut exits = 0;
    for b in &func.blocks {
        if info.blocks.contains(&b.label) {
            size += b.instructions.len();
            for i in &b.instructions {
                if !matches!(
                    i,
                    Instruction::IntBinary {
                        ty: MirType::Scalar(ScalarType::I64),
                        ..
                    } | Instruction::IntCmp {
                        ty: MirType::Scalar(ScalarType::I64),
                        ..
                    } | Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::I64),
                        ..
                    } | Instruction::Jmp { .. }
                        | Instruction::Br { .. }
                ) {
                    return false;
                }
                if i.def_reg()
                    .into_iter()
                    .chain(i.use_regs())
                    .any(|r| !r.is_virtual())
                {
                    return false;
                }
            }
            exits += b
                .successors()
                .iter()
                .filter(|target| !info.blocks.contains(**target))
                .count();
        } else if (b.label.contains("__unroll_") && b.successors().contains(&info.header.as_str()))
            || b.successors()
                .iter()
                .any(|target| **target != info.header && info.blocks.contains(*target))
        {
            return false;
        }
    }
    exits >= 2
        && size <= 24
        && !func
            .blocks
            .iter()
            .filter(|b| info.blocks.contains(&b.label))
            .flat_map(|b| &b.instructions)
            .any(Instruction::may_trap_integer)
}

fn clone_guarded(func: &mut Function, info: &LoopInfo, factor: usize, next: &mut u32) {
    let mut prefix = format!("__unroll_guard_{}", info.header);
    while func.blocks.iter().any(|b| b.label.starts_with(&prefix)) {
        prefix.push('_');
    }
    let label = |name: &str, copy: usize| format!("{prefix}_{copy}_{name}");
    let mut local = HashSet::new();
    for b in func
        .blocks
        .iter()
        .filter(|b| info.blocks.contains(&b.label))
    {
        for (index, i) in b.instructions.iter().enumerate() {
            if let Some(dst) = i.def_reg() {
                if func.sig.params.iter().any(|p| &p.reg == dst) {
                    continue;
                }
                let unique = func
                    .blocks
                    .iter()
                    .flat_map(|b| &b.instructions)
                    .filter(|i| i.def_reg() == Some(dst))
                    .count()
                    == 1;
                let local_uses = func.blocks.iter().all(|other| {
                    other.instructions.iter().enumerate().all(|(at, i)| {
                        !i.use_regs().contains(&dst) || (other.label == b.label && at > index)
                    })
                });
                if unique && local_uses {
                    local.insert(dst.clone());
                }
            }
        }
    }
    let originals: Vec<_> = func
        .blocks
        .iter()
        .filter(|b| info.blocks.contains(&b.label))
        .cloned()
        .collect();
    let mut copies = Vec::new();
    for copy in 0..factor {
        let mut renamed = HashMap::new();
        for b in &originals {
            for dst in b.instructions.iter().filter_map(Instruction::def_reg) {
                if local.contains(dst) {
                    renamed.insert(dst.clone(), fresh(next));
                }
            }
        }
        for b in &originals {
            let mut block = b.clone();
            block.label = label(&b.label, copy);
            for i in &mut block.instructions {
                super::inline::ModuleInlining::map_registers(i, &mut |r| {
                    renamed.get(r).cloned().unwrap_or_else(|| r.clone())
                });
                let target = |name: &mut String| {
                    if *name == info.header {
                        *name = label(name, (copy + 1) % factor);
                    } else if info.blocks.contains(name) {
                        *name = label(name, copy);
                    }
                };
                match i {
                    Instruction::Jmp { target: name } => target(name),
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        target(true_target);
                        target(false_target);
                    }
                    _ => {}
                }
            }
            copies.push(block);
        }
    }
    for b in &mut func.blocks {
        if info.blocks.contains(&b.label) {
            continue;
        }
        for i in &mut b.instructions {
            match i {
                Instruction::Jmp { target } if *target == info.header => {
                    *target = label(&info.header, 0)
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if *true_target == info.header {
                        *true_target = label(&info.header, 0);
                    }
                    if *false_target == info.header {
                        *false_target = label(&info.header, 0);
                    }
                }
                _ => {}
            }
        }
    }
    if func.entry == info.header {
        func.entry = label(&info.header, 0);
    }
    func.blocks.retain(|b| !info.blocks.contains(&b.label));
    func.blocks.extend(copies);
}

fn materialize_tail_loop(func: &mut Function) -> bool {
    if func.sig.params.len() > 32 {
        return false;
    }
    if func
        .sig
        .params
        .iter()
        .any(|p| p.ty != MirType::Scalar(ScalarType::I64))
    {
        return false;
    }
    let Some(mut next) = register_base(func) else {
        return false;
    };
    let entry = func.entry.clone();
    let mut pre = format!("{}_unroll_entry", entry);
    while func.blocks.iter().any(|b| b.label == pre) {
        pre.push('_');
    }
    for block in &mut func.blocks {
        let Some(Instruction::TailCall { name, args }) = block.instructions.last() else {
            continue;
        };
        if name != &func.sig.name || args.len() != func.sig.params.len() {
            continue;
        }
        let mut snapshots = Vec::new();
        let mut assignments = Vec::new();
        for (param, arg) in func.sig.params.iter().zip(args) {
            if arg.as_register() == Some(&param.reg) {
                continue;
            }
            let saved = fresh(&mut next);
            snapshots.push(Instruction::Copy {
                ty: param.ty,
                dst: saved.clone(),
                src: arg.clone(),
            });
            assignments.push(Instruction::Copy {
                ty: param.ty,
                dst: param.reg.clone(),
                src: saved.into(),
            });
        }
        block.instructions.pop();
        block.instructions.extend(snapshots);
        block.instructions.extend(assignments);
        block.instructions.push(jump(&entry));
    }
    func.blocks.insert(
        0,
        Block {
            label: pre.clone(),
            instructions: vec![jump(&entry)],
        },
    );
    func.entry = pre;
    true
}

fn register_base(func: &Function) -> Option<u32> {
    let next = func
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .flat_map(|i| i.def_reg().into_iter().chain(i.use_regs()))
        .chain(func.sig.params.iter().map(|p| &p.reg))
        .filter_map(|r| match r {
            Register::Virtual(v) => Some(v.id),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .checked_add(1)?;
    next.checked_add(100_000)?;
    Some(next)
}

fn fresh(next: &mut u32) -> Register {
    let reg = VirtualReg::gpr(*next).into();
    *next += 1;
    reg
}
fn imm(n: i64) -> Operand {
    Operand::Immediate(Immediate::I64(n))
}
fn binary(op: IntBinOp, dst: &Register, lhs: Operand, rhs: Operand) -> Instruction {
    Instruction::IntBinary {
        op,
        ty: MirType::Scalar(ScalarType::I64),
        dst: dst.clone(),
        lhs,
        rhs,
    }
}
fn compare(op: IntCmpOp, dst: &Register, lhs: Operand, rhs: Operand) -> Instruction {
    Instruction::IntCmp {
        op,
        ty: MirType::Scalar(ScalarType::I64),
        dst: dst.clone(),
        lhs,
        rhs,
    }
}
fn jump(target: &str) -> Instruction {
    Instruction::Jmp {
        target: target.into(),
    }
}
fn branch(cond: &Register, yes: &str, no: &str) -> Instruction {
    Instruction::Br {
        cond: cond.clone(),
        true_target: yes.into(),
        false_target: no.into(),
    }
}

fn counter_only(shape: &Shape) -> bool {
    let mut dependent = HashSet::from([shape.iv.clone()]);
    for i in &shape.body {
        let uses_counter = i.use_regs().iter().any(|r| dependent.contains(*r));
        if uses_counter && i.may_trap_integer() {
            return false;
        }
        if let Some(dst) = i.def_reg() {
            dependent.remove(dst);
            if uses_counter {
                dependent.insert(dst.clone());
            }
        }
    }
    shape
        .live_out
        .iter()
        .all(|r| *r == shape.iv || !dependent.contains(r))
}

fn partial_unroll(func: &mut Function, shape: &Shape, factor: usize, next: &mut u32) {
    let mut work = shape.clone();
    let counter_only = counter_only(shape);
    if counter_only {
        work.live_out.remove(&shape.iv);
    }
    let mut prefix = format!("__unroll_{}", shape.header);
    while func.blocks.iter().any(|b| b.label.starts_with(&prefix)) {
        prefix.push('_');
    }
    let label = |suffix: &str| format!("{prefix}_{suffix}");
    let dispatch = label("dispatch");
    let overflow = label("overflow");
    let setup = label("setup");
    let procheck = label("procheck");
    let probody = label("probody");
    let maincheck = label("maincheck");
    let mainbody = label("mainbody");
    let oddcheck = label("oddcheck");
    let oddbody = label("oddbody");
    let trip = fresh(next);
    let remainder = fresh(next);
    let flag = fresh(next);
    let guard_flag = fresh(next);
    let pro_flag = fresh(next);
    let main_flag = fresh(next);
    let odd_flag = fresh(next);
    let pro_back = fresh(next);
    let main_back = fresh(next);
    let mut guard = func
        .get_block(&shape.header)
        .map(|b| b.instructions.clone())
        .unwrap_or_default();
    if let Some(Instruction::IntCmp { dst, .. }) = guard.first_mut() {
        *dst = guard_flag.clone();
    }
    if let Some(Instruction::Br { cond, .. }) = guard.last_mut() {
        *cond = guard_flag;
    }
    if let Some(Instruction::Br {
        true_target,
        false_target,
        ..
    }) = guard.last_mut()
    {
        let target = if shape.labels.contains(true_target) {
            true_target
        } else {
            false_target
        };
        *target = if shape.inclusive {
            overflow.clone()
        } else {
            setup.clone()
        };
    }
    for b in &mut func.blocks {
        if shape.labels.contains(&b.label) {
            continue;
        }
        for i in &mut b.instructions {
            match i {
                Instruction::Jmp { target } if *target == shape.header => {
                    *target = dispatch.clone()
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if *true_target == shape.header {
                        *true_target = dispatch.clone();
                    }
                    if *false_target == shape.header {
                        *false_target = dispatch.clone();
                    }
                }
                _ => {}
            }
        }
    }
    let mut add_block = |name: String, instructions: Vec<Instruction>| {
        func.blocks.push(Block {
            label: name,
            instructions,
        })
    };
    add_block(dispatch, guard);
    if shape.inclusive {
        add_block(
            overflow,
            vec![
                compare(IntCmpOp::Eq, &flag, shape.end.clone(), imm(i64::MAX)),
                branch(&flag, &shape.header, &setup),
            ],
        );
    }
    let mut setup_code = vec![binary(
        IntBinOp::Sub,
        &trip,
        shape.end.clone(),
        Operand::Register(shape.iv.clone()),
    )];
    if shape.inclusive {
        setup_code.push(binary(IntBinOp::Add, &trip, trip.clone().into(), imm(1)));
    }
    setup_code.extend([
        binary(
            IntBinOp::And,
            &remainder,
            trip.clone().into(),
            imm(factor as i64 - 1),
        ),
        binary(
            IntBinOp::Sub,
            &trip,
            trip.clone().into(),
            remainder.clone().into(),
        ),
        jump(&procheck),
    ]);
    if counter_only {
        setup_code.pop();
        setup_code.push(binary(
            IntBinOp::Add,
            &shape.iv,
            shape.end.clone(),
            imm(i64::from(shape.inclusive)),
        ));
        setup_code.push(jump(&procheck));
    }
    add_block(setup, setup_code);
    add_block(
        procheck.clone(),
        vec![
            compare(IntCmpOp::UGe, &pro_flag, remainder.clone().into(), imm(2)),
            branch(&pro_flag, &probody, &oddcheck),
        ],
    );
    let mut code = renamed_body(&work, 2, next);
    code.extend([
        binary(IntBinOp::Sub, &remainder, remainder.clone().into(), imm(2)),
        compare(IntCmpOp::UGe, &pro_back, remainder.clone().into(), imm(2)),
        branch(&pro_back, &probody, &oddcheck),
    ]);
    add_block(probody, code);
    add_block(
        oddcheck,
        vec![
            compare(IntCmpOp::Ne, &odd_flag, remainder.clone().into(), imm(0)),
            branch(&odd_flag, &oddbody, &maincheck),
        ],
    );
    let mut odd = renamed_body(&work, 1, next);
    odd.push(jump(&maincheck));
    add_block(oddbody, odd);
    add_block(
        maincheck.clone(),
        vec![
            compare(IntCmpOp::Ne, &main_flag, trip.clone().into(), imm(0)),
            branch(&main_flag, &mainbody, &shape.header),
        ],
    );
    let mut code = renamed_body(&work, factor, next);
    code.extend([
        binary(
            IntBinOp::Sub,
            &trip,
            trip.clone().into(),
            imm(factor as i64),
        ),
        compare(IntCmpOp::Ne, &main_back, trip.clone().into(), imm(0)),
        branch(&main_back, &mainbody, &shape.header),
    ]);
    add_block(mainbody, code);
}

fn renamed_body(shape: &Shape, count: usize, next: &mut u32) -> Vec<Instruction> {
    let first_fresh = *next;
    let mut values: HashMap<Register, Operand> = HashMap::new();
    let mut affine: HashMap<Register, (Operand, i64)> = HashMap::new();
    let mut definitions = Vec::new();
    let mut code = Vec::new();
    let only_increment = shape
        .body
        .iter()
        .all(|i| i.def_reg() == Some(&shape.iv) || !i.use_regs().contains(&&shape.iv));
    for _ in 0..count {
        for i in &shape.body {
            let Some(dst) = i.def_reg() else {
                continue;
            };
            if !definitions.contains(dst) {
                definitions.push(dst.clone());
            }
            if only_increment && dst == &shape.iv {
                continue;
            }
            let resolve = |o: &Operand| {
                o.as_register()
                    .and_then(|r| values.get(r))
                    .cloned()
                    .unwrap_or_else(|| o.clone())
            };
            let (op, mut lhs, mut rhs) = match i {
                Instruction::IntBinary { op, lhs, rhs, .. } => (*op, resolve(lhs), resolve(rhs)),
                Instruction::Copy { src, .. } => (IntBinOp::Add, resolve(src), imm(0)),
                _ => continue,
            };
            if op == IntBinOp::Add {
                if let Some((base, offset)) = lhs.as_register().and_then(|r| affine.get(r)) {
                    if let Some(n) = rhs.integer_constant() {
                        rhs = imm(n.wrapping_add(*offset));
                        lhs = base.clone();
                    }
                }
            }
            if matches!(
                op,
                IntBinOp::Add | IntBinOp::Sub | IntBinOp::Or | IntBinOp::Xor
            ) && rhs.integer_constant() == Some(0)
            {
                values.insert(dst.clone(), lhs);
            } else {
                let result = fresh(next);
                if op == IntBinOp::Add {
                    if let Some(n) = rhs.integer_constant() {
                        affine.insert(result.clone(), (lhs.clone(), n));
                    }
                }
                code.push(binary(op, &result, lhs, rhs));
                values.insert(dst.clone(), result.into());
            }
        }
    }
    if only_increment && count > 0 {
        let result = fresh(next);
        code.push(binary(
            IntBinOp::Add,
            &result,
            shape.iv.clone().into(),
            imm(count as i64),
        ));
        values.insert(shape.iv.clone(), result.into());
    }
    let mut copies = Vec::new();
    for dst in definitions {
        if !shape.live_out.contains(&dst) {
            continue;
        }
        if let Some(mut src) = values.remove(&dst) {
            if let Operand::Register(r) = &src {
                if !matches!(r, Register::Virtual(v) if v.id >= first_fresh) {
                    let saved = fresh(next);
                    code.push(Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: saved.clone(),
                        src,
                    });
                    src = saved.into();
                }
            }
            copies.push(Instruction::Copy {
                ty: MirType::Scalar(ScalarType::I64),
                dst,
                src,
            });
        }
    }
    code.extend(copies);
    let mut live = shape.live_out.clone();
    code = code
        .into_iter()
        .rev()
        .filter(|i| {
            let needed = i.may_trap_integer() || i.def_reg().is_none_or(|r| live.contains(r));
            if needed {
                if let Some(r) = i.def_reg() {
                    live.remove(r);
                }
                live.extend(i.use_regs().into_iter().cloned());
            }
            needed
        })
        .collect();
    code.reverse();
    coalesce_region(&mut code, &shape.live_out, first_fresh);
    code
}

fn coalesce_region(code: &mut Vec<Instruction>, live_out: &HashSet<Register>, first_fresh: u32) {
    let mut live = live_out.clone();
    let mut edges: HashSet<(Register, Register)> = HashSet::new();
    let mut candidates = Vec::new();
    let mut registers = HashSet::new();
    for i in code.iter().rev() {
        if let Some(dst) = i.def_reg() {
            registers.insert(dst.clone());
            let copy = match i {
                Instruction::Copy {
                    src: Operand::Register(src),
                    ..
                } => Some(src),
                _ => None,
            };
            for r in &live {
                if r != dst && Some(r) != copy {
                    edges.insert((dst.clone(), r.clone()));
                    edges.insert((r.clone(), dst.clone()));
                }
            }
            live.remove(dst);
            if let Some(src) = copy {
                candidates.push((dst.clone(), src.clone()));
            }
        }
        for r in i.use_regs() {
            registers.insert(r.clone());
            live.insert(r.clone());
        }
    }
    for i in code.iter() {
        if let Instruction::IntBinary {
            dst,
            lhs: Operand::Register(src),
            ..
        } = i
        {
            candidates.push((src.clone(), dst.clone()));
        }
    }
    let mut groups: HashMap<Register, Register> =
        registers.into_iter().map(|r| (r.clone(), r)).collect();
    for (a, b) in candidates {
        let (Some(a), Some(b)) = (groups.get(&a).cloned(), groups.get(&b).cloned()) else {
            continue;
        };
        if a == b
            || edges
                .iter()
                .any(|(x, y)| groups.get(x) == Some(&a) && groups.get(y) == Some(&b))
        {
            continue;
        }
        let canonical = |r: &Register| matches!(r, Register::Virtual(v) if v.id < first_fresh);
        if canonical(&a) && canonical(&b) {
            continue;
        }
        let (keep, remove) = if canonical(&b) { (&b, &a) } else { (&a, &b) };
        for group in groups.values_mut() {
            if group == remove {
                *group = keep.clone();
            }
        }
    }
    for i in code.iter_mut() {
        super::inline::ModuleInlining::map_registers(i, &mut |r| {
            groups.get(r).cloned().unwrap_or_else(|| r.clone())
        });
    }
    code.retain(
        |i| !matches!(i, Instruction::Copy { dst, src: Operand::Register(src), .. } if dst == src),
    );
}

fn full_unroll(func: &mut Function, shape: &Shape, count: u64, next: &mut u32) {
    let mut instructions = renamed_body(shape, count as usize, next);
    if let Some(header) = func.get_block_mut(&shape.header) {
        instructions.extend_from_slice(&header.instructions);
        if let Some(Instruction::Br {
            true_target,
            false_target,
            ..
        }) = instructions.last()
        {
            let exit = if shape.labels.contains(true_target) {
                false_target
            } else {
                true_target
            }
            .clone();
            instructions.pop();
            instructions.push(Instruction::Jmp { target: exit });
        }
        header.instructions = instructions;
    }
    func.blocks
        .retain(|b| b.label == shape.header || !shape.labels.contains(&b.label));
}

fn constant_trip_count(func: &Function, shape: &Shape) -> Option<u64> {
    let entry = func.get_block(&shape.entry)?;
    let mut values: HashMap<Register, i64> = HashMap::new();
    for i in &entry.instructions {
        let value = |o: &Operand| {
            o.integer_constant()
                .or_else(|| o.as_register().and_then(|r| values.get(r).copied()))
        };
        let result = match i {
            Instruction::Copy { src, .. } => value(src),
            Instruction::IntBinary {
                op, ty, lhs, rhs, ..
            } => value(lhs)
                .zip(value(rhs))
                .and_then(|(a, b)| op.evaluate(*ty, a, b)),
            _ => None,
        };
        if let Some(dst) = i.def_reg() {
            values.remove(dst);
            if let Some(v) = result {
                values.insert(dst.clone(), v);
            }
        }
    }
    let start = *values.get(&shape.iv)?;
    let end = shape
        .end
        .integer_constant()
        .or_else(|| values.get(shape.end.as_register()?).copied())?;
    if shape.inclusive && end == i64::MAX {
        return None;
    }
    let count = i128::from(end) - i128::from(start) + i128::from(shape.inclusive);
    u64::try_from(count.max(0)).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{
        mir::{codegen::from_ir, verify::verify_function_in_pipeline},
        parser::parse_module,
        runtime::executor::interpret_mir_function,
    };

    fn counting(start: i64, end: &str, comparison: &str) -> Function {
        let source = format!(
            "fn @f(i64 %n) -> i64 {{
entry:
%i = add.i64 {start}, 0
%a = add.i64 0, 0
%b = add.i64 1, 0
jmp test
test:
%c = {comparison}.i64 %i, {end}
br %c, body, exit
body:
%t = add.i64 %a, %b
%a = add.i64 %b, 0
%b = add.i64 %t, 0
%i = add.i64 %i, 1
jmp test
exit:
ret.i64 %b
}}"
        );
        from_ir(&parse_module(&source).unwrap(), "unroll")
            .unwrap()
            .functions
            .remove("f")
            .unwrap()
    }

    #[test]
    fn coalescing_preserves_invariant_operands_across_iterations() {
        let mut original = counting(0, "%n", "lt");
        let invariant = original.sig.params[0].reg.clone();
        if let Instruction::IntBinary { lhs, .. } =
            &mut original.get_block_mut("body").unwrap().instructions[0]
        {
            *lhs = invariant.into();
        }
        let mut func = original.clone();
        assert!(LoopUnrolling.apply(&mut func).unwrap());
        for n in 0..34 {
            assert_eq!(
                interpret_mir_function(&func, &[n]).unwrap(),
                interpret_mir_function(&original, &[n]).unwrap(),
                "{n}"
            );
        }
    }

    #[test]
    fn every_remainder_preserves_values_near_signed_limits() {
        for factor in [2, 4, 8, 16] {
            for start in [i64::MIN, -17, i64::MAX - 40] {
                for comparison in ["lt", "le"] {
                    let original = counting(start, "%n", comparison);
                    let shape =
                        analyze(&original, &LoopInvariantCodeMotion.find_loops(&original)[0])
                            .unwrap();
                    let mut func = original.clone();
                    let mut next = register_base(&func).unwrap();
                    partial_unroll(&mut func, &shape, factor, &mut next);
                    verify_function_in_pipeline(&func).unwrap();
                    for count in 0..=factor * 2 + 1 {
                        let end = start + count as i64;
                        assert_eq!(
                            interpret_mir_function(&func, &[end]).unwrap(),
                            interpret_mir_function(&original, &[end]).unwrap(),
                            "{factor} {start} {comparison} {end}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn growth_budget_limits_large_bodies_and_function_sizes() {
        let mut func = counting(0, "%n", "lt");
        let body = func.get_block_mut("body").unwrap();
        let instruction = body.instructions[0].clone();
        for _ in 0..100 {
            body.instructions.insert(0, instruction.clone());
        }
        assert!(!LoopUnrolling.apply(&mut func).unwrap());
        let mut func = counting(0, "%n", "lt");
        let body = func.get_block_mut("body").unwrap();
        let instruction = body.instructions[0].clone();
        for _ in 0..8192 {
            body.instructions.insert(0, instruction.clone());
        }
        assert!(!LoopUnrolling.apply(&mut func).unwrap());
    }

    #[test]
    fn guarded_copies_preserve_early_exits_in_every_position() {
        let source = "fn @f(i64 %n) -> i64 {
entry:
%d = add.i64 3, 0
jmp test
test:
%sq = mul.i64 %d, %d
%done = gt.i64 %sq, %n
br %done, prime, body
body:
%r = rem.i64 %n, %d
%div = eq.i64 %r, 0
br %div, composite, next
next:
%d = add.i64 %d, 2
jmp test
prime:
ret.i64 %d
composite:
ret.i64 0
}";
        let original = from_ir(&parse_module(source).unwrap(), "guarded")
            .unwrap()
            .functions
            .remove("f")
            .unwrap();
        let mut func = original.clone();
        let info = LoopInvariantCodeMotion.find_loops(&func).remove(0);
        let mut next = register_base(&func).unwrap();
        clone_guarded(&mut func, &info, 4, &mut next);
        assert!(!LoopUnrolling.apply(&mut func).unwrap());
        verify_function_in_pipeline(&func).unwrap();
        for n in -4..240 {
            assert_eq!(
                interpret_mir_function(&func, &[n]).unwrap(),
                interpret_mir_function(&original, &[n]).unwrap(),
                "{n}"
            );
        }
    }

    #[test]
    fn tail_recursion_preserves_parallel_argument_rotation() {
        let source = "fn @f(i64 %n, i64 %a, i64 %b, i64 %i) -> i64 {
entry:
%c = gt.i64 %i, %n
br %c, exit, body
body:
%t = add.i64 %a, %b
%j = add.i64 %i, 1
%r = call @f(%n, %b, %t, %j)
ret.i64 %r
exit:
ret.i64 %b
}";
        let mut func = from_ir(&parse_module(source).unwrap(), "tail")
            .unwrap()
            .functions
            .remove("f")
            .unwrap();
        super::super::TailCallOptimization.apply(&mut func).unwrap();
        assert!(LoopUnrolling.apply(&mut func).unwrap());
        verify_function_in_pipeline(&func).unwrap();
        for trips in 0..=33 {
            let (mut a, mut b) = (7_i64, 11_i64);
            for _ in 0..trips {
                (a, b) = (b, a.wrapping_add(b));
            }
            assert_eq!(
                interpret_mir_function(&func, &[trips - 1, 7, 11, 0]).unwrap(),
                Some(b)
            );
        }
    }

    #[test]
    fn renaming_removes_rotation_copies_between_iterations() {
        let func = counting(0, "%n", "lt");
        let shape = analyze(&func, &LoopInvariantCodeMotion.find_loops(&func)[0]).unwrap();
        let mut next = register_base(&func).unwrap();
        let body = renamed_body(&shape, 8, &mut next);
        let arithmetic: Vec<_> = body
            .iter()
            .filter(|i| matches!(i, Instruction::IntBinary { .. }))
            .collect();
        assert_eq!(arithmetic.len(), 9);
        assert!(body[..8].iter().all(|i| matches!(
            i,
            Instruction::IntBinary {
                op: IntBinOp::Add,
                ..
            }
        )));
        let mut values = HashSet::new();
        for i in &body[..8] {
            values.insert(i.def_reg().unwrap());
        }
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn partial_unrolling_preserves_remainders_and_signed_spans() {
        for cmp in ["lt", "le"] {
            let original = counting(-3, "%n", cmp);
            let mut func = original.clone();
            assert!(LoopUnrolling.apply(&mut func).unwrap());
            assert!(!LoopUnrolling.apply(&mut func).unwrap());
            verify_function_in_pipeline(&func).unwrap();
            for n in [-9, -3, -2, 3, 4, 5, 6, 13, 29, 30] {
                assert_eq!(
                    interpret_mir_function(&func, &[n]).unwrap(),
                    interpret_mir_function(&original, &[n]).unwrap(),
                    "{cmp} {n}"
                );
            }
        }
    }

    #[test]
    fn full_unrolling_preserves_zero_and_nonzero_trips() {
        for count in [0, 1, 7, 8, 9, 16, 32] {
            let mut func = counting(3, &(3 + count).to_string(), "lt");
            let before = interpret_mir_function(&func, &[0]).unwrap();
            assert!(LoopUnrolling.apply(&mut func).unwrap());
            assert!(LoopInvariantCodeMotion.find_loops(&func).is_empty());
            assert_eq!(interpret_mir_function(&func, &[0]).unwrap(), before);
        }
    }

    #[test]
    fn trip_counts_respect_start_polarity_and_empty_ranges() {
        for (start, end, cmp, expected) in [
            (3, "7", "lt", 4),
            (3, "7", "le", 5),
            (7, "3", "lt", 0),
            (7, "7", "lt", 0),
            (7, "7", "le", 1),
            (-5, "2", "lt", 7),
        ] {
            let func = counting(start, end, cmp);
            let loops = LoopInvariantCodeMotion.find_loops(&func);
            let shape = analyze(&func, &loops[0]).unwrap();
            assert_eq!(constant_trip_count(&func, &shape), Some(expected));
        }
        let func = counting(0, "9223372036854775807", "le");
        let shape = analyze(&func, &LoopInvariantCodeMotion.find_loops(&func)[0]).unwrap();
        assert_eq!(constant_trip_count(&func, &shape), None);
    }
}
