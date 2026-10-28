#![allow(clippy::expect_used)]

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
