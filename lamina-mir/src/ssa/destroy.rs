use std::fmt;

use super::SsaFunction;
use crate::{
    block::Block,
    function::Function,
    instruction::{Instruction, Operand},
    parallel::insert_edge_moves,
    register::{Register, VirtualRegAllocator},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestroyError {
    pub function: String,
    pub message: String,
}

impl fmt::Display for DestroyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot destroy SSA of '@{}': {}",
            self.function, self.message
        )
    }
}

/// Lowers a graph back to MIR. Each block parameter becomes its register, and each edge
/// copies its arguments into the target's parameter registers as one parallel copy. A value
/// keeps the register it was built with, so construction emits no copies and the
/// instructions come back verbatim. Copies owed to the synthetic entry go into a new entry
/// block that jumps to the real one.
pub fn destroy(graph: &SsaFunction) -> Result<Function, DestroyError> {
    let fail = |message: String| DestroyError {
        function: graph.name.clone(),
        message,
    };

    let mut placed: Vec<((usize, i64), Block)> = graph
        .unreachable
        .iter()
        .map(|(index, block)| ((*index, 0), block.clone()))
        .collect();
    let mut edges: Vec<((String, String), Vec<Instruction>)> = Vec::new();
    let mut entry_moves: Option<(String, Vec<Instruction>)> = None;

    for (index, block) in graph.blocks.iter().enumerate() {
        if index != 0 {
            placed.push((
                block.order,
                Block {
                    label: block.label.clone(),
                    instructions: block
                        .insts
                        .iter()
                        .map(|inst| inst.instruction.clone())
                        .collect(),
                },
            ));
        }

        for edge in &block.edges {
            let target = graph.blocks.get(edge.target.0 as usize).ok_or_else(|| {
                fail(format!("edge from '{}' names a missing block", block.label))
            })?;
            let mut moves = Vec::new();
            for (&param, &arg) in target.params.iter().zip(&edge.args) {
                let param = &graph.values[param.0 as usize];
                let arg = &graph.values[arg.0 as usize];
                if param.register == arg.register {
                    continue;
                }
                let ty = param
                    .ty
                    .or(arg.ty)
                    .ok_or_else(|| fail(format!("copy into '{}' has no type", param.register)))?;
                moves.push(Instruction::Copy {
                    ty,
                    dst: param.register.clone(),
                    src: Operand::Register(arg.register.clone()),
                });
            }
            if moves.is_empty() {
                continue;
            }
            if index == 0 {
                entry_moves = Some((target.label.clone(), moves));
                continue;
            }
            let pred = block.label.clone();
            let succ = target.label.clone();
            if edges.iter().any(|((p, s), _)| *p == pred && *s == succ) {
                continue;
            }
            edges.push(((pred, succ), moves));
        }
    }

    placed.sort_by_key(|(rank, _)| *rank);
    let mut func = graph.shell.clone();
    func.blocks = placed.into_iter().map(|(_, block)| block).collect();

    if let Some((target, moves)) = entry_moves {
        let mut label = format!("{target}_entry");
        while func.blocks.iter().any(|block| block.label == label) {
            label.push('_');
        }
        let mut entry = Block::new(label.clone());
        entry.push(Instruction::Jmp {
            target: target.clone(),
        });
        func.blocks.insert(0, entry);
        func.entry = label.clone();
        edges.push(((label, target), moves));
    }

    if !edges.is_empty() {
        let mut vreg_alloc = VirtualRegAllocator::new();
        let highest = graph
            .values
            .iter()
            .filter_map(|value| match &value.register {
                Register::Virtual(reg) => Some(reg.id),
                Register::Physical(_) => None,
            })
            .chain(func.stack_reservations.keys().map(|reg| reg.id))
            .max();
        if let Some(highest) = highest {
            for _ in 0..=highest {
                vreg_alloc.allocate_gpr();
            }
        }
        insert_edge_moves(&mut func, edges, &mut vreg_alloc);
    }

    Ok(func)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        function::FunctionBuilder,
        instruction::Operand,
        register::VirtualReg,
        ssa::{BlockId, SsaFunction, ValueId},
        types::MirType,
        verify::verify_function_in_pipeline,
    };

    fn gpr(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }

    fn imm(dst: u32, value: i64) -> Instruction {
        Instruction::Copy {
            ty: MirType::i64(),
            dst: gpr(dst),
            src: Operand::imm_i64(value),
        }
    }

    fn mov(dst: u32, src: u32) -> Instruction {
        Instruction::Copy {
            ty: MirType::i64(),
            dst: gpr(dst),
            src: Operand::v_gpr(src),
        }
    }

    fn ret(src: u32) -> Instruction {
        Instruction::Ret {
            value: Some(Operand::v_gpr(src)),
        }
    }

    fn branch(cond: u32, t: &str, f: &str) -> Instruction {
        Instruction::Br {
            cond: gpr(cond),
            true_target: t.to_string(),
            false_target: f.to_string(),
        }
    }

    fn jump(target: &str) -> Instruction {
        Instruction::Jmp {
            target: target.to_string(),
        }
    }

    fn switch(value: u32, default: &str, cases: &[(i64, &str)]) -> Instruction {
        Instruction::Switch {
            value: gpr(value),
            cases: cases
                .iter()
                .map(|(key, target)| (*key, (*target).to_string()))
                .collect(),
            default: default.to_string(),
        }
    }

    fn graph_of(function: &Function) -> SsaFunction {
        SsaFunction::build(function, &|_| None).expect("build")
    }

    fn round_trip(function: &Function) -> Function {
        let graph = graph_of(function);
        graph.verify().expect("verify");
        destroy(&graph).expect("destroy")
    }

    fn blocks_text(function: &Function) -> String {
        format!("{:?}", function.blocks)
    }

    fn assert_round_trips(function: &Function) {
        let out = round_trip(function);
        assert_eq!(blocks_text(&out), blocks_text(function));
        assert_eq!(out.sig, function.sig);
        assert_eq!(out.entry, function.entry);
    }

    fn diamond() -> Function {
        FunctionBuilder::new("diamond")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(branch(0, "left", "right"))
            .block("left")
            .instr(imm(1, 1))
            .instr(jump("join"))
            .block("right")
            .instr(imm(1, 2))
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build()
    }

    #[test]
    fn diamond_round_trips_to_the_same_blocks() {
        assert_round_trips(&diamond());
    }

    #[test]
    fn loop_carried_value_round_trips_and_keeps_its_header_parameter() {
        let f = FunctionBuilder::new("carried")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 0))
            .instr(jump("head"))
            .block("head")
            .instr(mov(2, 1))
            .instr(branch(0, "body", "exit"))
            .block("body")
            .instr(mov(1, 2))
            .instr(jump("head"))
            .block("exit")
            .instr(ret(1))
            .build();
        let graph = graph_of(&f);
        assert_eq!(graph.blocks[2].params.len(), 1);
        assert_round_trips(&f);
    }

    #[test]
    fn self_referential_parameter_is_pruned_and_round_trips() {
        let f = FunctionBuilder::new("spin")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 5))
            .instr(jump("head"))
            .block("head")
            .instr(branch(0, "body", "exit"))
            .block("body")
            .instr(mov(2, 1))
            .instr(jump("head"))
            .block("exit")
            .instr(ret(1))
            .build();
        let graph = graph_of(&f);
        assert!(graph.blocks[2].params.is_empty());
        assert_round_trips(&f);
    }

    #[test]
    fn critical_edge_round_trips_without_a_trampoline() {
        let f = FunctionBuilder::new("critical")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(branch(0, "join", "body"))
            .block("body")
            .instr(imm(1, 2))
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build();
        assert_round_trips(&f);
    }

    #[test]
    fn critical_edge_with_a_renamed_argument_gets_a_trampoline() {
        let f = FunctionBuilder::new("critical_renamed")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(imm(5, 7))
            .instr(branch(0, "join", "body"))
            .block("body")
            .instr(imm(1, 2))
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build();
        let mut graph = graph_of(&f);
        let seven = ValueId(
            graph
                .values
                .iter()
                .position(|v| v.register == gpr(5))
                .expect("v5") as u32,
        );
        let entry = graph
            .blocks
            .iter()
            .position(|b| b.label == "entry")
            .expect("entry");
        let join = graph
            .blocks
            .iter()
            .position(|b| b.label == "join")
            .expect("join");
        let edge = graph.blocks[entry]
            .edges
            .iter_mut()
            .find(|e| e.target == BlockId(join as u32))
            .expect("edge");
        edge.args[0] = seven;
        let out = destroy(&graph).expect("destroy");
        let tramp = out
            .blocks
            .iter()
            .find(|b| b.label == "entry_to_join_phi1")
            .expect("trampoline");
        assert_eq!(
            tramp.instructions,
            vec![
                Instruction::Copy {
                    ty: MirType::i64(),
                    dst: gpr(1),
                    src: Operand::v_gpr(5),
                },
                jump("join"),
            ]
        );
        let entry_block = out
            .blocks
            .iter()
            .find(|b| b.label == "entry")
            .expect("entry");
        assert_eq!(
            entry_block.terminator(),
            Some(&branch(0, "entry_to_join_phi1", "body"))
        );
    }

    #[test]
    fn parallel_copy_cycle_is_sequenced_with_a_saved_register() {
        let f = FunctionBuilder::new("swap")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(imm(2, 2))
            .instr(jump("head"))
            .block("head")
            .instr(branch(0, "body", "exit"))
            .block("body")
            .instr(mov(3, 1))
            .instr(mov(1, 2))
            .instr(mov(2, 3))
            .instr(jump("head"))
            .block("exit")
            .instr(ret(1))
            .build();
        let mut graph = graph_of(&f);
        let head = graph
            .blocks
            .iter()
            .position(|b| b.label == "head")
            .expect("head");
        let body = graph
            .blocks
            .iter()
            .position(|b| b.label == "body")
            .expect("body");
        let reg_of = |graph: &SsaFunction, p: ValueId| graph.values[p.0 as usize].register.clone();
        let params = graph.blocks[head].params.clone();
        let i1 = params
            .iter()
            .position(|&p| reg_of(&graph, p) == gpr(1))
            .expect("p1");
        let i2 = params
            .iter()
            .position(|&p| reg_of(&graph, p) == gpr(2))
            .expect("p2");
        let edge = graph.blocks[body]
            .edges
            .iter_mut()
            .find(|e| e.target == BlockId(head as u32))
            .expect("back edge");
        edge.args.swap(i1, i2);

        let out = destroy(&graph).expect("destroy");
        let body_block = out.blocks.iter().find(|b| b.label == "body").expect("body");
        let moves = &body_block.instructions[3..body_block.instructions.len() - 1];
        assert_eq!(moves.len(), 3);

        let mut regs: HashMap<Register, i64> = [(gpr(1), 10), (gpr(2), 20), (gpr(3), 0)]
            .into_iter()
            .collect();
        for instruction in moves {
            if let Instruction::Copy {
                dst,
                src: Operand::Register(src),
                ..
            } = &instruction.instr
            {
                let value = regs[&src.clone()];
                regs.insert(dst.clone(), value);
            } else {
                panic!("unexpected move {instruction:?}");
            }
        }
        assert_eq!(regs[&gpr(1)], 20);
        assert_eq!(regs[&gpr(2)], 10);
    }

    #[test]
    fn identical_branch_arms_needing_copies_share_one_trampoline() {
        let f = FunctionBuilder::new("twin")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(imm(5, 7))
            .instr(branch(0, "twin", "other"))
            .block("twin")
            .instr(branch(0, "join", "join"))
            .block("other")
            .instr(imm(1, 2))
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build();
        let mut graph = graph_of(&f);
        let seven = ValueId(
            graph
                .values
                .iter()
                .position(|v| v.register == gpr(5))
                .expect("v5") as u32,
        );
        let twin = graph
            .blocks
            .iter()
            .position(|b| b.label == "twin")
            .expect("twin");
        for edge in graph.blocks[twin].edges.iter_mut() {
            edge.args[0] = seven;
        }
        let out = destroy(&graph).expect("destroy");
        let tramp = out
            .blocks
            .iter()
            .find(|b| b.label == "twin_to_join_phi1")
            .expect("trampoline");
        assert_eq!(
            tramp.instructions,
            vec![
                Instruction::Copy {
                    ty: MirType::i64(),
                    dst: gpr(1),
                    src: Operand::v_gpr(5),
                },
                jump("join"),
            ]
        );
        let twin_block = out.blocks.iter().find(|b| b.label == "twin").expect("twin");
        assert_eq!(
            twin_block.terminator(),
            Some(&branch(0, "twin_to_join_phi1", "twin_to_join_phi1"))
        );
        verify_function_in_pipeline(&out).expect("destroyed MIR verifies");
    }

    #[test]
    fn switch_edge_moves_run_on_their_own_edge() {
        let f = FunctionBuilder::new("dispatch")
            .param(gpr(0), MirType::i64())
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(imm(5, 7))
            .instr(switch(0, "other", &[(4, "join")]))
            .block("other")
            .instr(imm(1, 2))
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build();
        let mut graph = graph_of(&f);
        let seven = ValueId(
            graph
                .values
                .iter()
                .position(|v| v.register == gpr(5))
                .expect("v5") as u32,
        );
        let entry = graph
            .blocks
            .iter()
            .position(|b| b.label == "entry")
            .expect("entry");
        let join = graph
            .blocks
            .iter()
            .position(|b| b.label == "join")
            .expect("join");
        let edge = graph.blocks[entry]
            .edges
            .iter_mut()
            .find(|e| e.target == BlockId(join as u32))
            .expect("edge");
        edge.args[0] = seven;
        let out = destroy(&graph).expect("destroy");

        let tramp = out
            .blocks
            .iter()
            .find(|b| b.label == "entry_to_join_phi1")
            .expect("trampoline");
        assert_eq!(
            tramp.instructions,
            vec![
                Instruction::Copy {
                    ty: MirType::i64(),
                    dst: gpr(1),
                    src: Operand::v_gpr(5),
                },
                jump("join"),
            ]
        );
        let entry_block = out
            .blocks
            .iter()
            .find(|b| b.label == "entry")
            .expect("entry");
        assert_eq!(
            entry_block.instructions,
            vec![
                imm(1, 1),
                imm(5, 7),
                switch(0, "other", &[(4, "entry_to_join_phi1")]),
            ]
        );
        verify_function_in_pipeline(&out).expect("destroyed MIR verifies");
    }

    #[test]
    fn copies_owed_to_the_synthetic_entry_go_into_a_new_entry_block() {
        let f = FunctionBuilder::new("loop_entry")
            .param(gpr(0), MirType::i1())
            .param(gpr(1), MirType::i64())
            .param(gpr(2), MirType::i64())
            .returns(MirType::i64())
            .block("entry")
            .instr(mov(3, 1))
            .instr(branch(0, "body", "exit"))
            .block("body")
            .instr(imm(1, 5))
            .instr(jump("entry"))
            .block("exit")
            .instr(ret(3))
            .build();
        let mut graph = graph_of(&f);
        let entry = graph
            .blocks
            .iter()
            .position(|b| b.label == "entry")
            .expect("entry");
        let slot = graph.blocks[entry]
            .params
            .iter()
            .position(|&p| graph.values[p.0 as usize].register == gpr(1))
            .expect("entry parameter v1");
        let other = ValueId(
            graph
                .values
                .iter()
                .position(|v| v.register == gpr(2))
                .expect("v2") as u32,
        );
        graph.blocks[0].edges[0].args[slot] = other;

        let out = destroy(&graph).expect("destroy");
        assert_eq!(out.entry, "entry_entry");
        assert_eq!(out.blocks[0].label, "entry_entry");
        assert_eq!(
            out.blocks[0].instructions,
            vec![
                Instruction::Copy {
                    ty: MirType::i64(),
                    dst: gpr(1),
                    src: Operand::v_gpr(2),
                },
                jump("entry"),
            ]
        );
        verify_function_in_pipeline(&out).expect("destroyed MIR verifies");
    }
}
