#![allow(clippy::expect_used)]

use lamina_mir::Located;
use std::{fs, path::Path};

use lamina_mir::{Module, ssa::SsaFunction};

use crate::{
    mir::{TransformPipeline, codegen::from_ir},
    parser::parse_module,
};

fn verify_module(module: &Module, label: &str) {
    let callee_return = |name: &str| module.get_function(name).and_then(|f| f.sig.ret_ty);
    for (name, function) in &module.functions {
        if function.blocks.is_empty() {
            continue;
        }
        let graph = SsaFunction::build(function, &callee_return)
            .unwrap_or_else(|e| panic!("{label} @{name}: {e}"));
        if let Err(violations) = graph.verify() {
            let lines: Vec<String> = violations.iter().map(|v| v.message.clone()).collect();
            panic!("{label} @{name}: {}", lines.join("; "));
        }
    }
}

#[test]
fn every_testcase_builds_and_verifies_ssa_at_each_opt_level() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testcases");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .expect("read testcases")
        .map(|entry| entry.expect("testcase entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "lamina"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());

    for path in &paths {
        let source = fs::read_to_string(path).expect("read testcase");
        let name = path.display().to_string();
        let ir = parse_module(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        for level in 0..=3u8 {
            let mut mir = from_ir(&ir, &name).unwrap_or_else(|e| panic!("{name}: {e}"));
            if level > 0 {
                TransformPipeline::default_for_opt_level(level)
                    .apply_to_module(&mut mir)
                    .unwrap_or_else(|e| panic!("{name} O{level}: {e}"));
            }
            verify_module(&mir, &format!("{name} O{level}"));
        }
    }
}

#[test]
fn destroy_copies_on_a_real_loop_edge_round_trip() {
    use lamina_mir::{
        Instruction, MirType, Operand,
        register::{Register, VirtualReg},
        ssa::{BlockId, SsaInst, SsaValue, ValueDef, ValueId, destroy},
        verify::verify_function_in_pipeline,
    };

    let source = "fn @count(i64 %n) -> i64 {\nentry:\n    %i = add.i64 0, 0\n    jmp head\nhead:\n    %c = lt.i64 %i, %n\n    br %c, body, done\nbody:\n    %i = add.i64 %i, 1\n    jmp head\ndone:\n    ret.i64 %i\n}\n";
    let ir = parse_module(source).expect("parse loop");
    let module = from_ir(&ir, "loop").expect("lower loop");
    let function = &module.functions["count"];
    let mut graph = SsaFunction::build(function, &|_| None).expect("build");
    graph.verify().expect("built graph verifies");

    let body = graph
        .blocks
        .iter()
        .position(|b| b.label == "body")
        .expect("body");
    let head = graph
        .blocks
        .iter()
        .position(|b| b.label == "head")
        .expect("head");
    let edge = graph.blocks[body]
        .edges
        .iter()
        .position(|e| e.target == BlockId(head as u32))
        .expect("back edge");
    let first_register = graph
        .values
        .iter()
        .filter_map(|value| match &value.register {
            Register::Virtual(reg) => Some(reg.id + 1),
            Register::Physical(_) => None,
        })
        .max()
        .unwrap_or(0);
    let args = graph.blocks[body].edges[edge].args.clone();
    for (position, arg) in args.into_iter().enumerate() {
        let source_value = graph.values[arg.0 as usize].clone();
        let copy_register: Register = VirtualReg::gpr(first_register + position as u32).into();
        let copy = ValueId(graph.values.len() as u32);
        let inst_index = graph.blocks[body].insts.len() - 1;
        graph.values.push(SsaValue {
            def: ValueDef::Inst {
                block: BlockId(body as u32),
                inst: inst_index as u32,
            },
            ty: source_value.ty,
            register: copy_register.clone(),
        });
        graph.blocks[body].insts.insert(
            inst_index,
            SsaInst {
                source: inst_index,
                defines: Some(copy),
                uses: vec![arg],
                instruction: Located::from(Instruction::Copy {
                    ty: source_value.ty.unwrap_or_else(MirType::i64),
                    dst: copy_register,
                    src: Operand::Register(source_value.register.clone()),
                }),
            },
        );
        graph.blocks[body].edges[edge].args[position] = copy;
    }
    graph.verify().expect("edited graph verifies");

    let destroyed = destroy(&graph).expect("destroy");
    let body_block = destroyed
        .blocks
        .iter()
        .find(|b| b.label == "body")
        .expect("body block");
    let head_register = graph.values[graph.blocks[head].params[0].0 as usize]
        .register
        .clone();
    assert!(
        body_block.instructions.iter().any(|inst| matches!(
            &inst.instr,
            Instruction::Copy { dst, .. } if *dst == head_register
        )),
        "{:?}",
        body_block.instructions
    );
    verify_function_in_pipeline(function).expect("source MIR verifies in the pipeline");
    verify_function_in_pipeline(&destroyed).expect("destroyed MIR verifies");

    let rebuilt = SsaFunction::build(&destroyed, &|_| None).expect("rebuild");
    rebuilt.verify().expect("rebuilt graph verifies");
    let again = destroy(&rebuilt).expect("destroy again");
    assert_eq!(
        format!("{:?}", again.blocks),
        format!("{:?}", destroyed.blocks)
    );
}
