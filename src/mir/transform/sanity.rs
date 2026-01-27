//! Sanity checks and validation for MIR transforms.
//!
//! Validation functions to check that MIR structures are
//! well-formed after transformations. These checks catch bugs in
//! transform implementations.

use crate::mir::{Function, Instruction};

/// Validate that all branch and jump targets reference existing blocks.
pub fn validate_cfg(func: &Function) -> Result<(), String> {
    let labels: std::collections::HashSet<_> =
        func.blocks.iter().map(|b| b.label.clone()).collect();
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Instruction::Jmp { target } => {
                    if !labels.contains(target) {
                        return Err(format!(
                            "Invalid CFG: block '{}' jumps to missing target '{}'",
                            block.label, target
                        ));
                    }
                }
                Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } => {
                    if !labels.contains(true_target) {
                        return Err(format!(
                            "Invalid CFG: block '{}' branches to missing true_target '{}'",
                            block.label, true_target
                        ));
                    }
                    if !labels.contains(false_target) {
                        return Err(format!(
                            "Invalid CFG: block '{}' branches to missing false_target '{}'",
                            block.label, false_target
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
