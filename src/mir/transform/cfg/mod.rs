//! Control flow graph (CFG) simplification transforms.
//!
//! Transforms that simplify the control flow graph:
//!
//! - **CfgSimplify**: Simplifies trivial branches and selects
//! - **JumpThreading**: Bypasses trivial jump-only blocks

use crate::mir::{
    Function, Instruction,
    transform::{Transform, TransformCategory, TransformError, TransformLevel},
};
use std::collections::{HashMap, HashSet};

/// Identify loop headers via back-edge detection (target block index ≤ source block index).
pub(crate) fn compute_back_edge_headers(func: &Function) -> HashSet<String> {
    let mut label_index: HashMap<&str, usize> = HashMap::new();
    for (i, b) in func.blocks.iter().enumerate() {
        label_index.insert(&b.label, i);
    }
    let mut headers: HashSet<String> = HashSet::new();
    for (i, b) in func.blocks.iter().enumerate() {
        for succ in b.successors() {
            if let Some(&tidx) = label_index.get(succ)
                && tidx <= i
            {
                headers.insert(succ.to_owned());
            }
        }
    }
    headers
}

/// Dominance relation over the blocks of one function.
///
/// Built from the immediate-dominator tree (Cooper, Harvey and Kennedy), so a query
/// costs constant time and the relation is never materialised as per-block sets.
/// Blocks unreachable from the entry are dominated by every block, which is what the
/// iterative dataflow formulation yields for them.
pub(crate) struct Dominators {
    index: HashMap<String, usize>,
    /// Pre- and post-order numbers in the dominator tree; `None` when unreachable.
    interval: Vec<Option<(usize, usize)>>,
}

impl Dominators {
    /// Whether block `a` dominates block `b` (a block dominates itself).
    pub(crate) fn dominates(&self, a: &str, b: &str) -> bool {
        let (Some(&ai), Some(&bi)) = (self.index.get(a), self.index.get(b)) else {
            return false;
        };
        match (self.interval[ai], self.interval[bi]) {
            (_, None) => true,
            (None, Some(_)) => false,
            (Some((a_in, a_out)), Some((b_in, _))) => a_in <= b_in && b_in < a_out,
        }
    }
}

/// Compute the dominance relation of `func`.
pub(crate) fn calculate_dominators(func: &Function) -> Dominators {
    let n = func.blocks.len();
    let mut index: HashMap<String, usize> = HashMap::with_capacity(n);
    for (i, block) in func.blocks.iter().enumerate() {
        index.insert(block.label.clone(), i);
    }

    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, block) in func.blocks.iter().enumerate() {
        for target in block.successors() {
            if let Some(&t) = index.get(target) {
                succs[i].push(t);
                preds[t].push(i);
            }
        }
    }

    let Some(&entry) = index.get(func.entry.as_str()) else {
        return Dominators {
            index,
            interval: vec![None; n],
        };
    };

    // Reverse post-order of the blocks reachable from the entry.
    let mut postorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack = vec![(entry, 0usize)];
    visited[entry] = true;
    while let Some((node, next)) = stack.last_mut() {
        let node = *node;
        if let Some(&succ) = succs[node].get(*next) {
            *next += 1;
            if !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(node);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = postorder.iter().rev().copied().collect();
    let mut rpo_number = vec![usize::MAX; n];
    for (number, &node) in rpo.iter().enumerate() {
        rpo_number[node] = number;
    }

    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[entry] = Some(entry);
    let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while rpo_number[a] > rpo_number[b] {
                a = idom[a].unwrap_or(entry);
            }
            while rpo_number[b] > rpo_number[a] {
                b = idom[b].unwrap_or(entry);
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &node in rpo.iter().skip(1) {
            let mut new_idom: Option<usize> = None;
            for &pred in &preds[node] {
                if idom[pred].is_none() {
                    continue;
                }
                new_idom = Some(match new_idom {
                    None => pred,
                    Some(current) => intersect(&idom, pred, current),
                });
            }
            if new_idom.is_some() && idom[node] != new_idom {
                idom[node] = new_idom;
                changed = true;
            }
        }
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &node in &rpo {
        if node != entry
            && let Some(parent) = idom[node]
        {
            children[parent].push(node);
        }
    }
    let mut interval = vec![None; n];
    let mut clock = 0usize;
    let mut stack = vec![(entry, 0usize)];
    interval[entry] = Some((clock, 0));
    clock += 1;
    while let Some((node, next)) = stack.last_mut() {
        let node = *node;
        if let Some(&child) = children[node].get(*next) {
            *next += 1;
            interval[child] = Some((clock, 0));
            clock += 1;
            stack.push((child, 0));
        } else {
            if let Some((_, out)) = interval[node].as_mut() {
                *out = clock;
            }
            clock += 1;
            stack.pop();
        }
    }

    Dominators { index, interval }
}

/// Simple CFG simplifications:
/// - br with identical true/false targets -> jmp
/// - select with identical true/false values -> move
#[derive(Default)]
pub struct CfgSimplify;

impl Transform for CfgSimplify {
    fn name(&self) -> &'static str {
        "cfg_simplify"
    }

    fn description(&self) -> &'static str {
        "Simplifies trivial branches and selects"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        // Safe local rewrites; treat as stable for -O1
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl CfgSimplify {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut changed = false;

        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                match &instr.instr {
                    // br cond, A, A -> jmp A
                    Instruction::Br {
                        cond: _,
                        true_target,
                        false_target,
                    } if true_target == false_target => {
                        let target = true_target.clone();
                        instr.instr = Instruction::Jmp { target };
                        changed = true;
                    }
                    // select cond, x, x -> copy x
                    Instruction::Select {
                        dst,
                        ty,
                        cond: _,
                        true_val,
                        false_val,
                    } if true_val == false_val => {
                        let replacement = Instruction::Copy {
                            ty: *ty,
                            dst: dst.clone(),
                            src: true_val.clone(),
                        };
                        instr.instr = replacement;
                        changed = true;
                    }
                    _ => {}
                }
            }
        }

        let mut preds: HashMap<String, Vec<String>> = HashMap::new();
        for block in &func.blocks {
            if let Some(term) = block.instructions.last()
                && term.is_terminator()
            {
                match &term.instr {
                    Instruction::Jmp { target } => {
                        preds
                            .entry(target.clone())
                            .or_default()
                            .push(block.label.clone());
                    }
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        preds
                            .entry(true_target.clone())
                            .or_default()
                            .push(block.label.clone());
                        preds
                            .entry(false_target.clone())
                            .or_default()
                            .push(block.label.clone());
                    }
                    Instruction::Switch { cases, default, .. } => {
                        preds
                            .entry(default.clone())
                            .or_default()
                            .push(block.label.clone());
                        for (_, case_target) in cases {
                            preds
                                .entry(case_target.clone())
                                .or_default()
                                .push(block.label.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut merges = Vec::new();
        for block in &func.blocks {
            if block.instructions.len() == 1
                && let Some(Instruction::Jmp { target }) =
                    block.instructions.last().map(|i| &i.instr)
                && let Some(preds_list) = preds.get(&block.label)
                && preds_list.len() == 1
            {
                let pred_label = preds_list[0].clone();
                merges.push((pred_label, target.clone(), block.label.clone()));
            }
        }

        let mut to_remove = Vec::new();
        let mut touched = HashSet::new();
        for (pred_label, new_target, trivial_label) in merges {
            if trivial_label == func.entry
                || trivial_label == new_target
                || [&pred_label, &new_target, &trivial_label]
                    .iter()
                    .any(|label| touched.contains(*label))
            {
                continue;
            }
            if let Some(pred_block) = func.blocks.iter_mut().find(|b| b.label == pred_label)
                && let Some(pred_term) = pred_block.instructions.last_mut()
                && let Instruction::Jmp { target } = &mut pred_term.instr
                && *target == trivial_label
            {
                touched.extend([pred_label, new_target.clone(), trivial_label.clone()]);
                *target = new_target;
                changed = true;
                to_remove.push(trivial_label);
            }
        }

        func.blocks.retain(|b| !to_remove.contains(&b.label));
        func.blocks.sort_by_key(|b| b.label.clone());

        Ok(changed)
    }
}

/// CFG Jump Threading / Target Bypass
/// Rewrites branches/jumps that target trivial jump-only blocks to jump directly
/// to the final destination. This reduces unnecessary hops and makes other passes more effective.
#[derive(Default)]
pub struct JumpThreading;

impl Transform for JumpThreading {
    fn name(&self) -> &'static str {
        "jump_threading"
    }

    fn description(&self) -> &'static str {
        "Bypass trivial jump-only blocks in branch/jump targets"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ControlFlowOptimization
    }

    fn level(&self) -> TransformLevel {
        // Safe as it preserves semantics of jumps/branches; treat as stable
        TransformLevel::Stable
    }

    fn apply(&self, func: &mut Function) -> Result<bool, TransformError> {
        self.apply_internal(func)
    }
}

impl JumpThreading {
    pub(crate) fn apply_internal(&self, func: &mut Function) -> Result<bool, TransformError> {
        let mut simple_jumps: HashMap<String, String> = HashMap::new();

        for block in &func.blocks {
            if block.instructions.len() == 1
                && let Instruction::Jmp { target } = &block.instructions[0].instr
            {
                simple_jumps.insert(block.label.clone(), target.clone());
            }
        }

        fn resolve_target<'m>(map: &'m HashMap<String, String>, tgt: &'m str) -> &'m str {
            let mut current = tgt;
            let mut seen = HashSet::new();
            const MAX_CHAIN: usize = 100;
            let mut i = 0;
            while let Some(next) = map.get(current) {
                if i >= MAX_CHAIN || !seen.insert(current) {
                    break;
                }
                current = next;
                i += 1;
            }
            current
        }

        // Map each trivial block to its non-trivial terminal, skipping cycles.
        let mut resolved_targets: HashMap<String, String> = HashMap::new();
        for k in simple_jumps.keys() {
            let resolved = resolve_target(&simple_jumps, k);
            if resolved != k.as_str() && !simple_jumps.contains_key(resolved) {
                resolved_targets.insert(k.clone(), resolved.to_owned());
            }
        }

        let mut changed = false;
        for block in &mut func.blocks {
            for instr in &mut block.instructions {
                match &mut instr.instr {
                    Instruction::Jmp { target } => {
                        if let Some(new_tgt) = resolved_targets.get(target)
                            && new_tgt != target
                        {
                            *target = new_tgt.clone();
                            changed = true;
                        }
                    }
                    Instruction::Br {
                        cond: _,
                        true_target,
                        false_target,
                    } => {
                        if let Some(new_tgt) = resolved_targets.get(true_target)
                            && new_tgt != true_target
                        {
                            *true_target = new_tgt.clone();
                            changed = true;
                        }
                        if let Some(new_tgt) = resolved_targets.get(false_target)
                            && new_tgt != false_target
                        {
                            *false_target = new_tgt.clone();
                            changed = true;
                        }
                    }
                    Instruction::Switch {
                        value: _, cases, ..
                    } => {
                        let mut local_change = false;
                        for (_val, tgt) in cases.iter_mut() {
                            if let Some(new_tgt) = resolved_targets.get(tgt)
                                && new_tgt != tgt
                            {
                                *tgt = new_tgt.clone();
                                local_change = true;
                            }
                        }
                        if local_change {
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, MirType, Operand, Register, ScalarType, VirtualReg,
        transform::test_utils::apply_pass,
    };

    fn i64() -> MirType {
        MirType::Scalar(ScalarType::I64)
    }

    fn cond_reg() -> Register {
        VirtualReg::gpr(0).into()
    }

    fn simple_linear_func() -> Function {
        FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build()
    }

    fn loop_func() -> Function {
        // entry -> loop_header -> loop_header (back edge)
        FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: "loop_header".to_owned(),
            })
            .block("loop_header")
            .instr(Instruction::Br {
                cond: cond_reg(),
                true_target: "loop_header".to_owned(),
                false_target: "exit".to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build()
    }

    #[test]
    fn adjacent_trivial_merges_preserve_branch_targets() {
        for reverse in [false, true] {
            let mut func = FunctionBuilder::new("chain")
                .returns(i64())
                .block("entry")
                .instr(Instruction::Jmp {
                    target: "first".into(),
                })
                .block("first")
                .instr(Instruction::Jmp {
                    target: "second".into(),
                })
                .block("second")
                .instr(Instruction::Jmp {
                    target: "exit".into(),
                })
                .block("exit")
                .instr(Instruction::Ret {
                    value: Some(Operand::imm_i64(42)),
                })
                .build();
            if reverse {
                func.blocks.swap(1, 2);
            }
            for _ in 0..4 {
                apply_pass(&CfgSimplify, &mut func);
                assert!(func.get_block(&func.entry).is_some());
                for block in &func.blocks {
                    for target in block.successors() {
                        assert!(func.get_block(target).is_some(), "{target}");
                    }
                }
            }
        }
    }

    #[test]
    fn entry_self_loops_survive_trivial_merging() {
        let mut func = FunctionBuilder::new("cycle")
            .block("entry")
            .instr(Instruction::Jmp {
                target: "entry".into(),
            })
            .build();
        apply_pass(&CfgSimplify, &mut func);
        assert_eq!(func.blocks.len(), 1);
        assert!(func.get_block(&func.entry).is_some());
    }

    #[test]
    fn no_back_edges_in_linear_func() {
        let func = simple_linear_func();
        assert!(compute_back_edge_headers(&func).is_empty());
    }

    #[test]
    fn self_loop_detected_as_back_edge() {
        let func = loop_func();
        let headers = compute_back_edge_headers(&func);
        assert!(headers.contains("loop_header"));
    }

    #[test]
    fn entry_dominates_all_blocks() {
        let func = loop_func();
        let doms = calculate_dominators(&func);
        for block in &func.blocks {
            assert!(
                doms.dominates("entry", &block.label),
                "entry should dominate '{}'",
                block.label
            );
        }
    }

    #[test]
    fn block_dominates_itself() {
        let func = simple_linear_func();
        let doms = calculate_dominators(&func);
        assert!(doms.dominates("entry", "entry"));
    }

    #[test]
    fn cfg_simplify_collapses_identical_br_targets() {
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Br {
                cond: cond_reg(),
                true_target: "exit".to_owned(),
                false_target: "exit".to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let changed = apply_pass(&CfgSimplify, &mut func);
        assert!(changed);
        let entry = func.get_block("entry").unwrap();
        assert!(
            matches!(&entry.instructions[0].instr, Instruction::Jmp { target } if target == "exit")
        );
    }

    #[test]
    fn cfg_simplify_collapses_identical_select_values() {
        let dst: Register = VirtualReg::gpr(1).into();
        let val = Operand::Immediate(Immediate::I64(5));
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Select {
                dst: dst.clone(),
                ty: i64(),
                cond: cond_reg(),
                true_val: val.clone(),
                false_val: val,
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        let changed = apply_pass(&CfgSimplify, &mut func);
        assert!(changed);
        let entry = func.get_block("entry").unwrap();
        assert!(matches!(
            &entry.instructions[0].instr,
            Instruction::Copy { .. }
        ));
    }

    #[test]
    fn cfg_simplify_leaves_float_select_alone() {
        let dst: Register = VirtualReg::gpr(1).into();
        let f64_ty = MirType::Scalar(ScalarType::F64);
        let val = Operand::Immediate(Immediate::F64(2.5));
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), f64_ty)
            .returns(f64_ty)
            .block("entry")
            .instr(Instruction::Select {
                dst: dst.clone(),
                ty: f64_ty,
                cond: cond_reg(),
                true_val: val.clone(),
                false_val: val,
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(dst)),
            })
            .build();

        let changed = apply_pass(&CfgSimplify, &mut func);
        assert!(changed);
        let entry = func.get_block("entry").unwrap();
        assert!(
            matches!(
                &entry.instructions[0].instr,
                Instruction::Copy {
                    ty,
                    src: Operand::Immediate(Immediate::F64(value)),
                    ..
                } if *ty == f64_ty && *value == 2.5
            ),
            "float select must become Copy, got {:?}",
            entry.instructions[0]
        );
    }

    #[test]
    fn cfg_simplify_no_change_when_already_simple() {
        let mut func = simple_linear_func();
        let changed = apply_pass(&CfgSimplify, &mut func);
        assert!(!changed);
    }

    #[test]
    fn jump_threading_bypasses_trivial_block() {
        let mut func = FunctionBuilder::new("f")
            .returns(i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: "trampoline".to_owned(),
            })
            .block("trampoline")
            .instr(Instruction::Jmp {
                target: "exit".to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let changed = apply_pass(&JumpThreading, &mut func);
        assert!(changed);
        let entry = func.get_block("entry").unwrap();
        assert!(
            matches!(&entry.instructions[0].instr, Instruction::Jmp { target } if target == "exit")
        );
    }

    #[test]
    fn jump_threading_no_change_without_trivial_blocks() {
        let mut func = simple_linear_func();
        let changed = apply_pass(&JumpThreading, &mut func);
        assert!(!changed);
    }

    #[test]
    fn jump_threading_handles_br_to_trivial_block() {
        let mut func = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), i64())
            .returns(i64())
            .block("entry")
            .instr(Instruction::Br {
                cond: cond_reg(),
                true_target: "trampoline".to_owned(),
                false_target: "exit".to_owned(),
            })
            .block("trampoline")
            .instr(Instruction::Jmp {
                target: "exit".to_owned(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Immediate(Immediate::I64(0))),
            })
            .build();

        let changed = apply_pass(&JumpThreading, &mut func);
        assert!(changed);
        let entry = func.get_block("entry").unwrap();
        if let Instruction::Br { true_target, .. } = &entry.instructions[0].instr {
            assert_eq!(true_target, "exit");
        } else {
            panic!("expected Br in entry block");
        }
    }
}
