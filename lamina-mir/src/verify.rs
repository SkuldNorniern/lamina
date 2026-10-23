//! Structural, SSA, dominance, and type verification for MIR.

use crate::{
    Function, Immediate, Instruction, MirType, Module, Operand, Register, RegisterClass,
    ScalarType, VectorLane, VectorOp, VectorType,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct Definition {
    block_index: usize,
    instruction_index: Option<usize>,
    ty: Option<MirType>,
}

#[derive(Clone, Copy)]
struct VerificationConfig {
    verify_terminator_placement: bool,
    verify_ssa: bool,
    verify_dominance: bool,
    verify_register_types: bool,
    verify_register_classes: bool,
    verify_immediate_widths: bool,
}

const STRICT_CONFIG: VerificationConfig = VerificationConfig {
    verify_terminator_placement: true,
    verify_ssa: true,
    verify_dominance: true,
    verify_register_types: true,
    verify_register_classes: true,
    verify_immediate_widths: true,
};
const PHI_LOWERED_CONFIG: VerificationConfig = VerificationConfig {
    verify_terminator_placement: true,
    verify_ssa: false,
    verify_dominance: false,
    verify_register_types: true,
    verify_register_classes: false,
    verify_immediate_widths: true,
};

const PRE_PHI_CONFIG: VerificationConfig = VerificationConfig {
    verify_ssa: true,
    verify_dominance: true,
    ..PHI_LOWERED_CONFIG
};

/// Phi definitions and edge uses retained until edge-copy insertion.
pub struct PhiVerification {
    pub block: String,
    pub dst: Register,
    pub ty: MirType,
    pub incoming: Vec<(String, Operand)>,
}

/// Verify typed SSA before edge copies, exempting source-declared reassignments.
/// Reassignment exemptions must come from source definitions, never verification failures.
pub fn verify_before_phi_lowering(
    function: &Function,
    phis: &[PhiVerification],
    reassignments: &HashSet<Register>,
) -> Result<(), Vec<String>> {
    let config = PRE_PHI_CONFIG;
    let mut errors = Vec::new();
    let block_indices = verify_structure(function, config, &mut errors);
    let mut definitions = collect_definitions(function, PHI_LOWERED_CONFIG, None, &mut errors);
    for phi in phis {
        if let Some(&block_index) = block_indices.get(phi.block.as_str()) {
            definitions
                .entry(phi.dst.clone())
                .or_default()
                .push(Definition {
                    block_index,
                    instruction_index: None,
                    ty: Some(phi.ty),
                });
        } else {
            errors.push(format!("expected phi block '{}' to exist", phi.block));
        }
    }
    verify_unique_definitions(function, &definitions, config, reassignments, &mut errors);
    let dominators = calculate_dominators(function, &block_indices);
    verify_dominance(function, &definitions, &dominators, &mut errors);
    verify_instruction_types(function, &definitions, config, None, &mut errors);
    let mut predecessors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for block in &function.blocks {
        if let Some(terminator) = block.instructions.last() {
            for target in instruction_targets(terminator) {
                predecessors.entry(target).or_default().insert(&block.label);
            }
        }
    }
    for phi in phis {
        let mut incoming = HashSet::new();
        for (predecessor, operand) in &phi.incoming {
            if !incoming.insert(predecessor.as_str()) {
                errors.push(location(
                    function,
                    &phi.block,
                    format!("expected unique phi predecessor, found duplicate '{predecessor}'"),
                ));
            }
            let Some(&index) = block_indices.get(predecessor.as_str()) else {
                errors.push(format!("expected phi predecessor '{predecessor}' to exist"));
                continue;
            };
            let block = &function.blocks[index];
            if !block
                .instructions
                .last()
                .is_some_and(|term| instruction_targets(term).contains(&phi.block.as_str()))
            {
                errors.push(location(
                    function,
                    predecessor,
                    format!("expected phi predecessor to branch to '{}'", phi.block),
                ));
            }
            verify_operand(
                operand,
                phi.ty,
                "phi incoming",
                &definitions,
                config,
                &mut errors,
            );
            if let Operand::Register(register) = operand {
                let dominates = definitions.get(register).is_some_and(|defs| {
                    reassignments.contains(register)
                        || (defs.len() == 1
                            && (defs[0].block_index == index
                                || dominators[index]
                                    .as_ref()
                                    .is_some_and(|set| set.contains(&defs[0].block_index))))
                });
                if !dominates {
                    errors.push(location(function, predecessor, format!("expected phi incoming register {register} to have a dominating definition")));
                }
            }
        }
        if predecessors
            .get(phi.block.as_str())
            .is_some_and(|expected| *expected != incoming)
        {
            errors.push(location(
                function,
                &phi.block,
                "expected phi inputs to cover every predecessor".into(),
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify one MIR function and return every problem found.
pub fn verify_function(function: &Function) -> Result<(), Vec<String>> {
    let errors = function_errors_with_config(function, STRICT_CONFIG, None);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a complete MIR module and return every problem found.
pub fn verify_module(module: &Module) -> Result<(), Vec<String>> {
    module.validate()
}

/// Verify phi-lowered MIR at optimizer and backend input boundaries.
#[doc(hidden)]
pub fn verify_function_in_pipeline(function: &Function) -> Result<(), Vec<String>> {
    let errors = function_errors_with_config(function, PHI_LOWERED_CONFIG, None);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify phi-lowered modules at optimizer and backend input boundaries.
#[doc(hidden)]
pub fn verify_module_in_pipeline(module: &Module) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for function in module.functions.values() {
        errors.extend(function_errors_with_config(
            function,
            PHI_LOWERED_CONFIG,
            Some(module),
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub(crate) fn function_errors(function: &Function, module: &Module) -> Vec<String> {
    function_errors_with_config(function, STRICT_CONFIG, Some(module))
}

fn function_errors_with_config(
    function: &Function,
    config: VerificationConfig,
    module: Option<&Module>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let block_indices = verify_structure(function, config, &mut errors);
    let definitions = collect_definitions(function, config, module, &mut errors);
    if config.verify_dominance {
        let dominators = calculate_dominators(function, &block_indices);
        verify_dominance(function, &definitions, &dominators, &mut errors);
    }
    verify_instruction_types(function, &definitions, config, module, &mut errors);
    errors
}

fn verify_structure<'a>(
    function: &'a Function,
    config: VerificationConfig,
    errors: &mut Vec<String>,
) -> HashMap<&'a str, usize> {
    let mut block_indices = HashMap::new();
    for (block_index, block) in function.blocks.iter().enumerate() {
        if block_indices
            .insert(block.label.as_str(), block_index)
            .is_some()
        {
            errors.push(location(
                function,
                &block.label,
                format!(
                    "expected a unique block label, found duplicate '{}'",
                    block.label
                ),
            ));
        }

        let terminator_indices: Vec<usize> = block
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(index, instruction)| instruction.is_terminator().then_some(index))
            .collect();
        if terminator_indices.is_empty()
            || (config.verify_terminator_placement && terminator_indices.len() != 1)
        {
            errors.push(location(
                function,
                &block.label,
                format!(
                    "expected exactly one terminator, found {}",
                    terminator_indices.len()
                ),
            ));
        }
        if config.verify_terminator_placement
            && let Some(terminator_index) = terminator_indices
                .iter()
                .find(|index| **index + 1 != block.instructions.len())
        {
            errors.push(instruction_location(
                function,
                &block.label,
                *terminator_index,
                format!(
                    "expected terminator to be the last instruction, found {} instruction(s) after it",
                    block.instructions.len() - terminator_index - 1
                ),
            ));
        }
    }

    if !block_indices.contains_key(function.entry.as_str()) {
        errors.push(location(
            function,
            &function.entry,
            format!(
                "expected entry block '{}' to exist, found no such block",
                function.entry
            ),
        ));
    }

    for block in &function.blocks {
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            for target in instruction_targets(instruction) {
                if !block_indices.contains_key(target) {
                    errors.push(instruction_location(
                        function,
                        &block.label,
                        instruction_index,
                        format!("expected branch target '{target}' to exist, found no such block"),
                    ));
                }
            }
        }
    }

    block_indices
}

fn collect_definitions(
    function: &Function,
    config: VerificationConfig,
    module: Option<&Module>,
    errors: &mut Vec<String>,
) -> HashMap<Register, Vec<Definition>> {
    let mut definitions: HashMap<Register, Vec<Definition>> = HashMap::new();
    let entry_index = function
        .blocks
        .iter()
        .position(|block| block.label == function.entry);

    if let Some(entry_index) = entry_index {
        for parameter in &function.sig.params {
            if parameter.reg.is_virtual() {
                definitions
                    .entry(parameter.reg.clone())
                    .or_default()
                    .push(Definition {
                        block_index: entry_index,
                        instruction_index: None,
                        ty: Some(parameter.ty),
                    });
            }
        }
    }
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            if let Some(register) = instruction.def_reg()
                && register.is_virtual()
            {
                definitions
                    .entry(register.clone())
                    .or_default()
                    .push(Definition {
                        block_index,
                        instruction_index: Some(instruction_index),
                        ty: match instruction {
                            Instruction::Call { name, .. } => module
                                .and_then(|m| m.functions.get(name))
                                .and_then(|callee| callee.sig.ret_ty),
                            _ => instruction_definition_type(instruction),
                        },
                    });
            }
        }
    }

    if config.verify_register_types {
        for (register, defs) in &definitions {
            let mut types = defs.iter().filter_map(|definition| definition.ty);
            if let Some(first) = types.next()
                && types.any(|ty| ty != first)
            {
                errors.push(location(
                    function,
                    &function.entry,
                    format!("expected register {register} definitions to have consistent types"),
                ));
            }
        }
    }
    verify_unique_definitions(function, &definitions, config, &HashSet::new(), errors);
    definitions
}

fn verify_unique_definitions(
    function: &Function,
    definitions: &HashMap<Register, Vec<Definition>>,
    config: VerificationConfig,
    reassignments: &HashSet<Register>,
    errors: &mut Vec<String>,
) {
    for (register, register_definitions) in definitions {
        if config.verify_ssa && register_definitions.len() > 1 && !reassignments.contains(register)
        {
            for definition in register_definitions.iter().skip(1) {
                let block_label = function
                    .blocks
                    .get(definition.block_index)
                    .map_or(function.entry.as_str(), |block| block.label.as_str());
                let message = format!(
                    "expected virtual register {register} to be defined once, found {} definitions",
                    register_definitions.len()
                );
                errors.push(match definition.instruction_index {
                    Some(index) => instruction_location(function, block_label, index, message),
                    None => location(function, block_label, message),
                });
            }
        }
    }
}

/// Compute dominator sets by block index; unreachable blocks have no set.
pub fn calculate_dominators(
    function: &Function,
    block_indices: &HashMap<&str, usize>,
) -> Vec<Option<HashSet<usize>>> {
    let block_count = function.blocks.len();
    let mut predecessors = vec![Vec::new(); block_count];
    for (block_index, block) in function.blocks.iter().enumerate() {
        if let Some(terminator) = block.instructions.last() {
            for target in instruction_targets(terminator) {
                if let Some(target_index) = block_indices.get(target) {
                    predecessors[*target_index].push(block_index);
                }
            }
        }
    }

    let Some(entry_index) = block_indices.get(function.entry.as_str()).copied() else {
        return vec![None; block_count];
    };
    let mut reachable = HashSet::new();
    let mut worklist = vec![entry_index];
    while let Some(block_index) = worklist.pop() {
        if !reachable.insert(block_index) {
            continue;
        }
        if let Some(terminator) = function.blocks[block_index].instructions.last() {
            for target in instruction_targets(terminator) {
                if let Some(target_index) = block_indices.get(target) {
                    worklist.push(*target_index);
                }
            }
        }
    }

    let mut dominators = vec![None; block_count];
    for block_index in &reachable {
        dominators[*block_index] = if *block_index == entry_index {
            Some(HashSet::from([entry_index]))
        } else {
            Some(reachable.clone())
        };
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block_index in &reachable {
            if *block_index == entry_index {
                continue;
            }
            let reachable_predecessors: Vec<usize> = predecessors[*block_index]
                .iter()
                .copied()
                .filter(|predecessor| reachable.contains(predecessor))
                .collect();
            let mut new_dominators = HashSet::new();
            if let Some(first_predecessor) = reachable_predecessors.first()
                && let Some(predecessor_dominators) = &dominators[*first_predecessor]
            {
                for dominator in predecessor_dominators {
                    new_dominators.insert(*dominator);
                }
            }
            for predecessor in reachable_predecessors.iter().skip(1) {
                if let Some(predecessor_dominators) = &dominators[*predecessor] {
                    new_dominators.retain(|candidate| predecessor_dominators.contains(candidate));
                }
            }
            new_dominators.insert(*block_index);
            if dominators[*block_index].as_ref() != Some(&new_dominators) {
                dominators[*block_index] = Some(new_dominators);
                changed = true;
            }
        }
    }

    dominators
}

fn verify_dominance(
    function: &Function,
    definitions: &HashMap<Register, Vec<Definition>>,
    dominators: &[Option<HashSet<usize>>],
    errors: &mut Vec<String>,
) {
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            for register in instruction.use_regs() {
                if !register.is_virtual() {
                    continue;
                }
                let Some(register_definitions) = definitions.get(register) else {
                    errors.push(instruction_location(
                        function,
                        &block.label,
                        instruction_index,
                        format!(
                            "expected use of virtual register {register} to have a dominating definition, found no definition"
                        ),
                    ));
                    continue;
                };
                if register_definitions.len() != 1 {
                    continue;
                }
                let definition = register_definitions[0];
                let dominates = if definition.block_index == block_index {
                    definition
                        .instruction_index
                        .is_none_or(|definition_index| definition_index < instruction_index)
                } else {
                    dominators[block_index]
                        .as_ref()
                        .is_none_or(|block_dominators| {
                            block_dominators.contains(&definition.block_index)
                        })
                };
                if !dominates {
                    let definition_block = &function.blocks[definition.block_index].label;
                    errors.push(instruction_location(
                        function,
                        &block.label,
                        instruction_index,
                        format!(
                            "expected definition of virtual register {register} in block '{definition_block}' to dominate this use, found a non-dominating definition"
                        ),
                    ));
                }
            }
        }
    }
}

fn verify_instruction_types(
    function: &Function,
    definitions: &HashMap<Register, Vec<Definition>>,
    config: VerificationConfig,
    module: Option<&Module>,
    errors: &mut Vec<String>,
) {
    for block in &function.blocks {
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            let mut instruction_errors = Vec::new();
            match instruction {
                Instruction::Call { name, args, .. } | Instruction::TailCall { name, args } => {
                    if let Some(callee) = module.and_then(|m| m.functions.get(name)) {
                        let op = if matches!(instruction, Instruction::TailCall { .. }) {
                            "tailcall"
                        } else {
                            "call"
                        };
                        if args.len() != callee.sig.params.len() {
                            instruction_errors.push(format!(
                                "expected {op} '{name}' to have {} arguments, found {}",
                                callee.sig.params.len(),
                                args.len()
                            ));
                        }
                        for (index, (argument, parameter)) in
                            args.iter().zip(&callee.sig.params).enumerate()
                        {
                            verify_operand(
                                argument,
                                parameter.ty,
                                &format!("{op} '{name}' argument {index}"),
                                definitions,
                                config,
                                &mut instruction_errors,
                            );
                        }
                        match instruction {
                            Instruction::Call { ret: Some(dst), .. } => match callee.sig.ret_ty {
                                Some(ty) => verify_destination(dst, ty, "call result", config, &mut instruction_errors),
                                None => instruction_errors.push(format!("expected void call '{name}' to have no result")),
                            },
                            Instruction::TailCall { .. } if function.sig.ret_ty != callee.sig.ret_ty => instruction_errors.push(format!("expected tailcall '{name}' result {:?} to match caller result {:?}", callee.sig.ret_ty, function.sig.ret_ty)),
                            _ => {},
                        }
                    }
                }
                Instruction::Copy { ty, dst, src } => {
                    verify_destination(dst, *ty, "Copy", config, &mut instruction_errors);
                    verify_operand(
                        src,
                        *ty,
                        "Copy source",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::Bitcast { from, to, dst, src } => {
                    if from.size_bytes() != to.size_bytes() || from.is_vector() || to.is_vector() {
                        instruction_errors.push("expected equal-width scalar bitcast".into());
                    }
                    verify_destination(dst, *to, "Bitcast", config, &mut instruction_errors);
                    verify_operand(
                        src,
                        *from,
                        "Bitcast source",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::IntConvert {
                    from,
                    to,
                    signed,
                    dst,
                    src,
                } => {
                    if !is_integer_or_pointer(*from)
                        || !is_integer_or_pointer(*to)
                        || (*signed && from.size_bytes() >= to.size_bytes())
                    {
                        instruction_errors.push("expected a valid integer conversion".into());
                    }
                    verify_destination(dst, *to, "IntConvert", config, &mut instruction_errors);
                    verify_operand(
                        src,
                        *from,
                        "IntConvert source",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::IntBinary {
                    ty, dst, lhs, rhs, ..
                } => {
                    if !is_integer_or_pointer(*ty) {
                        instruction_errors.push(format!(
                            "expected IntBinary type to be an integer or pointer scalar, found {ty}"
                        ));
                    }
                    verify_destination(dst, *ty, "IntBinary", config, &mut instruction_errors);
                    verify_operand(
                        lhs,
                        *ty,
                        "IntBinary lhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        rhs,
                        *ty,
                        "IntBinary rhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::FloatBinary {
                    ty, dst, lhs, rhs, ..
                } => {
                    if !is_float_scalar(*ty) {
                        instruction_errors.push(format!(
                            "expected FloatBinary type to be a float scalar, found {ty}"
                        ));
                    }
                    verify_destination(dst, *ty, "FloatBinary", config, &mut instruction_errors);
                    verify_operand(
                        lhs,
                        *ty,
                        "FloatBinary lhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        rhs,
                        *ty,
                        "FloatBinary rhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::FloatUnary { ty, dst, src, .. } => {
                    verify_destination(dst, *ty, "FloatUnary", config, &mut instruction_errors);
                    verify_operand(
                        src,
                        *ty,
                        "FloatUnary source",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::IntCmp {
                    ty, dst, lhs, rhs, ..
                } => {
                    if !is_integer_or_pointer(*ty) {
                        instruction_errors.push(format!(
                            "expected IntCmp operand type to be an integer or pointer scalar, found {ty}"
                        ));
                    }
                    verify_destination(
                        dst,
                        MirType::i1(),
                        "IntCmp",
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        lhs,
                        *ty,
                        "IntCmp lhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        rhs,
                        *ty,
                        "IntCmp rhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::FloatCmp {
                    ty, dst, lhs, rhs, ..
                } => {
                    if !is_float_scalar(*ty) {
                        instruction_errors.push(format!(
                            "expected FloatCmp operand type to be a float scalar, found {ty}"
                        ));
                    }
                    verify_destination(
                        dst,
                        MirType::i1(),
                        "FloatCmp",
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        lhs,
                        *ty,
                        "FloatCmp lhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        rhs,
                        *ty,
                        "FloatCmp rhs",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::Select {
                    ty,
                    dst,
                    cond,
                    true_val,
                    false_val,
                } => {
                    verify_destination(dst, *ty, "Select", config, &mut instruction_errors);
                    verify_register(
                        cond,
                        MirType::i1(),
                        "Select condition",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        true_val,
                        *ty,
                        "Select true arm",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                    verify_operand(
                        false_val,
                        *ty,
                        "Select false arm",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::Store { ty, src, .. } => {
                    verify_operand(
                        src,
                        *ty,
                        "Store source",
                        definitions,
                        config,
                        &mut instruction_errors,
                    );
                }
                Instruction::VectorOp {
                    op,
                    ty,
                    dst,
                    operands,
                } => {
                    let scalar = vector_scalar(*ty);
                    let dst_ty = if *op == VectorOp::VExtractLane {
                        scalar
                    } else {
                        *ty
                    };
                    verify_destination(dst, dst_ty, "VectorOp", config, &mut instruction_errors);
                    for (index, operand) in operands.iter().enumerate() {
                        let operand_ty = match (op, index) {
                            (VectorOp::VSplat, 0) | (VectorOp::VInsertLane, 1) => scalar,
                            (VectorOp::VExtractLane, 1) | (VectorOp::VInsertLane, 2) => {
                                MirType::i64()
                            }
                            _ => *ty,
                        };
                        verify_operand(
                            operand,
                            operand_ty,
                            if matches!(
                                (op, index),
                                (VectorOp::VExtractLane, 1) | (VectorOp::VInsertLane, 2)
                            ) {
                                "VectorOp lane index"
                            } else {
                                "VectorOp operand"
                            },
                            definitions,
                            config,
                            &mut instruction_errors,
                        );
                    }
                }
                Instruction::Ret { value: Some(value) } => {
                    if let Some(return_type) = function.sig.ret_ty {
                        verify_operand(
                            value,
                            return_type,
                            "return value",
                            definitions,
                            config,
                            &mut instruction_errors,
                        );
                    }
                }
                _ => {}
            }
            errors.extend(instruction_errors.into_iter().map(|message| {
                instruction_location(function, &block.label, instruction_index, message)
            }));
        }
    }
}

fn verify_destination(
    register: &Register,
    expected_type: MirType,
    instruction_name: &str,
    config: VerificationConfig,
    errors: &mut Vec<String>,
) {
    if !config.verify_register_types {
        return;
    }
    let expected_class = register_class_for_type(expected_type);
    if config.verify_register_classes && register.class() != expected_class {
        errors.push(format!(
            "expected {instruction_name} destination to have type {expected_type} ({expected_class:?} register), found {register} with {:?} register class",
            register.class()
        ));
    }
}

fn verify_operand(
    operand: &Operand,
    expected_type: MirType,
    operand_name: &str,
    definitions: &HashMap<Register, Vec<Definition>>,
    config: VerificationConfig,
    errors: &mut Vec<String>,
) {
    match operand {
        Operand::Register(register) => {
            verify_register(
                register,
                expected_type,
                operand_name,
                definitions,
                config,
                errors,
            );
        }
        Operand::Immediate(immediate) => {
            if config.verify_immediate_widths && !immediate_matches_type(*immediate, expected_type)
            {
                errors.push(format!(
                    "expected {operand_name} immediate to match type {expected_type}, found {} immediate",
                    immediate_type_name(*immediate)
                ));
            }
        }
    }
}

fn verify_register(
    register: &Register,
    expected_type: MirType,
    operand_name: &str,
    definitions: &HashMap<Register, Vec<Definition>>,
    config: VerificationConfig,
    errors: &mut Vec<String>,
) {
    if !config.verify_register_types {
        return;
    }
    let expected_class = register_class_for_type(expected_type);
    if config.verify_register_classes && register.class() != expected_class {
        errors.push(format!(
            "expected {operand_name} to have type {expected_type} ({expected_class:?} register), found {register} with {:?} register class",
            register.class()
        ));
        return;
    }
    let Some(register_definitions) = definitions.get(register) else {
        return;
    };
    for definition in register_definitions {
        if let Some(actual_type) = definition.ty
            && actual_type != expected_type
        {
            errors.push(format!(
            "expected {operand_name} to have type {expected_type}, found register {register} defined as {actual_type}"
        ));
        }
    }
}

fn instruction_definition_type(instruction: &Instruction) -> Option<MirType> {
    match instruction {
        Instruction::Copy { ty, .. }
        | Instruction::IntBinary { ty, .. }
        | Instruction::FloatBinary { ty, .. }
        | Instruction::FloatUnary { ty, .. }
        | Instruction::Select { ty, .. }
        | Instruction::Load { ty, .. } => Some(*ty),
        Instruction::VectorOp {
            op: VectorOp::VExtractLane,
            ty,
            ..
        } => Some(vector_scalar(*ty)),
        Instruction::VectorOp { ty, .. } => Some(*ty),
        Instruction::IntCmp { .. } | Instruction::FloatCmp { .. } => Some(MirType::i1()),
        Instruction::IntConvert { to, .. } | Instruction::Bitcast { to, .. } => Some(*to),
        Instruction::Lea { .. } | Instruction::StackAddr { .. } => Some(MirType::ptr()),
        _ => None,
    }
}

fn instruction_targets(instruction: &Instruction) -> Vec<&str> {
    match instruction {
        Instruction::Jmp { target } => vec![target],
        Instruction::Br {
            true_target,
            false_target,
            ..
        } => vec![true_target, false_target],
        Instruction::Switch { cases, default, .. } => {
            let mut targets = Vec::with_capacity(cases.len() + 1);
            targets.push(default.as_str());
            targets.extend(cases.iter().map(|(_, target)| target.as_str()));
            targets
        }
        _ => Vec::new(),
    }
}

fn vector_scalar(ty: MirType) -> MirType {
    let MirType::Vector(VectorType::V128(lane) | VectorType::V256(lane)) = ty else {
        return ty;
    };
    MirType::Scalar(match lane {
        VectorLane::I8 => ScalarType::I8,
        VectorLane::I16 => ScalarType::I16,
        VectorLane::I32 => ScalarType::I32,
        VectorLane::I64 => ScalarType::I64,
        VectorLane::F32 => ScalarType::F32,
        VectorLane::F64 => ScalarType::F64,
    })
}

fn is_integer_or_pointer(ty: MirType) -> bool {
    matches!(
        ty,
        MirType::Scalar(
            ScalarType::I1
                | ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::Ptr
                | ScalarType::PtrIn(_)
        )
    )
}

fn is_float_scalar(ty: MirType) -> bool {
    matches!(ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64))
}

fn register_class_for_type(ty: MirType) -> RegisterClass {
    match ty {
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => RegisterClass::Fpr,
        MirType::Vector(VectorType::V128(_) | VectorType::V256(_)) => RegisterClass::Vec,
        MirType::Scalar(_) => RegisterClass::Gpr,
    }
}

fn immediate_matches_type(immediate: Immediate, ty: MirType) -> bool {
    matches!(
        (immediate, ty),
        (
            Immediate::I8(_),
            MirType::Scalar(ScalarType::I1 | ScalarType::I8)
        ) | (Immediate::I16(_), MirType::Scalar(ScalarType::I16))
            | (Immediate::I32(_), MirType::Scalar(ScalarType::I32))
            | (
                Immediate::I64(_),
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr | ScalarType::PtrIn(_))
            )
            | (Immediate::F32(_), MirType::Scalar(ScalarType::F32))
            | (Immediate::F64(_), MirType::Scalar(ScalarType::F64))
    )
}

fn immediate_type_name(immediate: Immediate) -> &'static str {
    match immediate {
        Immediate::I8(_) => "i8",
        Immediate::I16(_) => "i16",
        Immediate::I32(_) => "i32",
        Immediate::I64(_) => "i64",
        Immediate::F32(_) => "f32",
        Immediate::F64(_) => "f64",
    }
}

fn location(function: &Function, block: &str, message: String) -> String {
    format!(
        "function '{}', block '{}': {message}",
        function.sig.name, block
    )
}

fn instruction_location(
    function: &Function,
    block: &str,
    instruction_index: usize,
    message: String,
) -> String {
    location(
        function,
        block,
        format!("instruction {instruction_index}: {message}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Block, FloatBinOp, FloatCmpOp, FunctionBuilder, IntBinOp, IntCmpOp, Parameter, Signature,
        VirtualReg,
    };

    fn verification_errors(function: &Function) -> Vec<String> {
        match verify_function(function) {
            Ok(()) => vec!["function unexpectedly passed verification".to_string()],
            Err(errors) => errors,
        }
    }

    fn pre_phi_errors(function: Function) -> Vec<String> {
        let mut module = Module::new("pre_phi");
        module.add_function(function);
        verify_before_phi_lowering(&module.functions["f"], &[], &HashSet::new())
            .err()
            .unwrap_or_default()
    }

    #[test]
    fn pre_phi_rejects_duplicate_definitions_and_non_dominating_uses() {
        let duplicate = FunctionBuilder::new("f")
            .param(VirtualReg::gpr(0).into(), MirType::i64())
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i64(1),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        assert!(
            pre_phi_errors(duplicate)
                .iter()
                .any(|e| e.contains("defined once"))
        );
        let forward = FunctionBuilder::new("f")
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::v_gpr(0),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i64(1),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        assert!(
            pre_phi_errors(forward)
                .iter()
                .any(|e| e.contains("non-dominating"))
        );
    }

    #[test]
    fn pre_phi_reassignment_exemptions_do_not_cover_temporaries() {
        let function = FunctionBuilder::new("f")
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i64(0),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i64(1),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::imm_i64(2),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::imm_i64(3),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        let reassignments = HashSet::from([VirtualReg::gpr(0).into()]);
        let errors = verify_before_phi_lowering(&function, &[], &reassignments)
            .err()
            .unwrap_or_default();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("v1 to be defined once"), "{errors:?}");
    }

    #[test]
    fn phi_inputs_are_checked_on_predecessor_edges() {
        let function = FunctionBuilder::new("f")
            .returns(MirType::i64())
            .block("entry")
            .instr(Instruction::Jmp {
                target: "merge".into(),
            })
            .block("merge")
            .instr(Instruction::Ret {
                value: Some(Operand::v_gpr(0)),
            })
            .build();
        let mut phi = PhiVerification {
            block: "merge".into(),
            dst: VirtualReg::gpr(0).into(),
            ty: MirType::i64(),
            incoming: vec![("entry".into(), Operand::imm_i64(1))],
        };
        assert!(verify_before_phi_lowering(&function, &[phi], &HashSet::new()).is_ok());
        phi = PhiVerification {
            block: "merge".into(),
            dst: VirtualReg::gpr(0).into(),
            ty: MirType::i64(),
            incoming: vec![("entry".into(), Operand::v_gpr(0))],
        };
        let errors = verify_before_phi_lowering(&function, &[phi], &HashSet::new())
            .err()
            .unwrap_or_default();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("phi incoming register") && e.contains("dominating")),
            "{errors:?}"
        );
    }

    #[test]
    fn pipeline_rejects_integer_immediate_width_mismatch() {
        let mut module = Module::new("width");
        module.add_function(
            FunctionBuilder::new("f")
                .block("entry")
                .instr(Instruction::Copy {
                    ty: MirType::i8(),
                    dst: VirtualReg::gpr(0).into(),
                    src: Operand::imm_i64(1),
                })
                .instr(Instruction::Ret { value: None })
                .build(),
        );
        let errors = module.validate_in_pipeline().err().unwrap_or_default();
        assert!(
            errors.iter().any(|e| e.contains("match type i8")),
            "{errors:?}"
        );
    }

    #[test]
    fn signatures_check_call_arguments_results_and_tail_results() {
        for tail in [false, true] {
            let mut module = Module::new("calls");
            module.add_function(
                FunctionBuilder::new("callee")
                    .param(VirtualReg::gpr(0).into(), MirType::i32())
                    .returns(MirType::i32())
                    .block("entry")
                    .instr(Instruction::Ret {
                        value: Some(Operand::v_gpr(0)),
                    })
                    .build(),
            );
            let call = if tail {
                Instruction::TailCall {
                    name: "callee".into(),
                    args: vec![Operand::v_gpr(0)],
                }
            } else {
                Instruction::Call {
                    name: "callee".into(),
                    args: vec![Operand::v_gpr(0)],
                    ret: Some(VirtualReg::gpr(1).into()),
                }
            };
            let mut caller = FunctionBuilder::new("caller")
                .param(VirtualReg::gpr(0).into(), MirType::i64())
                .returns(MirType::i64())
                .block("entry")
                .instr(call);
            if !tail {
                caller = caller.instr(Instruction::Ret {
                    value: Some(Operand::v_gpr(1)),
                });
            }
            module.add_function(caller.build());
            for errors in [
                module.validate().err().unwrap_or_default(),
                module.validate_in_pipeline().err().unwrap_or_default(),
            ] {
                assert!(
                    errors
                        .iter()
                        .any(|e| e.contains("argument 0") && e.contains("defined as i64")),
                    "{errors:?}"
                );
                assert!(
                    errors.iter().any(|e| if tail {
                        e.contains("match caller result")
                    } else {
                        e.contains("return value") && e.contains("defined as i32")
                    }),
                    "{errors:?}"
                );
            }
        }
    }

    #[test]
    fn signatures_reject_void_call_destinations() {
        let mut module = Module::new("void");
        module.add_function(
            FunctionBuilder::new("callee")
                .block("entry")
                .instr(Instruction::Ret { value: None })
                .build(),
        );
        module.add_function(
            FunctionBuilder::new("caller")
                .block("entry")
                .instr(Instruction::Call {
                    name: "callee".into(),
                    args: vec![],
                    ret: Some(VirtualReg::gpr(0).into()),
                })
                .instr(Instruction::Ret { value: None })
                .build(),
        );
        assert!(
            module
                .validate_in_pipeline()
                .err()
                .unwrap_or_default()
                .iter()
                .any(|e| e.contains("void call"))
        );
    }

    #[test]
    fn pipeline_rejects_register_width_disagreements_with_multiple_definitions() {
        let function = FunctionBuilder::new("bad_widths")
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i32(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i32(1),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_i64(2),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::v_gpr(0),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        let mut module = Module::new("bad_widths");
        module.add_function(function);
        let errors = module.validate_in_pipeline().err().unwrap_or_default();
        assert!(
            errors.iter().any(|e| e.contains("defined as i32")),
            "{errors:?}"
        );
    }

    #[test]
    fn pipeline_accepts_float_bits_in_general_register_storage() {
        let function = FunctionBuilder::new("float_storage")
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::f32(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::imm_f32(1.0),
            })
            .instr(Instruction::Bitcast {
                from: MirType::f32(),
                to: MirType::i32(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::v_gpr(0),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        let mut module = Module::new("float_storage");
        module.add_function(function);
        assert!(module.validate_in_pipeline().is_ok());
    }

    #[test]
    fn rejects_instruction_after_terminator() {
        let function = FunctionBuilder::new("bad_terminator")
            .block("entry")
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret { value: None })
            .block("exit")
            .instr(Instruction::Ret { value: None })
            .build();

        let mut module = Module::new("bad_terminator");
        module.add_function(function.clone());
        assert!(module.validate_in_pipeline().is_err());
        let errors = verification_errors(&function);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("exactly one terminator"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("last instruction"))
        );
    }

    #[test]
    fn accepts_one_final_terminator() {
        let function = FunctionBuilder::new("good_terminator")
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        assert!(verify_function(&function).is_ok());
    }

    #[test]
    fn rejects_multiple_virtual_register_definitions() {
        let function = FunctionBuilder::new("bad_ssa")
            .param(VirtualReg::gpr(0).into(), MirType::i64())
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        let errors = verification_errors(&function);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("found 2 definitions"))
        );
    }

    #[test]
    fn accepts_single_virtual_register_definitions() {
        let function = FunctionBuilder::new("good_ssa")
            .param(VirtualReg::gpr(0).into(), MirType::i64())
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                src: Operand::Register(VirtualReg::gpr(0).into()),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        assert!(verify_function(&function).is_ok());
    }

    #[test]
    fn rejects_definition_that_does_not_dominate_use() {
        let mut function = Function::new(
            Signature::new("bad_dominance")
                .with_params(vec![Parameter::new(
                    VirtualReg::gpr(0).into(),
                    MirType::i1(),
                )])
                .with_return(MirType::i64()),
        );
        let mut entry = Block::new("entry");
        entry.push(Instruction::Br {
            cond: VirtualReg::gpr(0).into(),
            true_target: "left".to_string(),
            false_target: "right".to_string(),
        });
        let mut left = Block::new("left");
        left.push(Instruction::Copy {
            ty: MirType::i64(),
            dst: VirtualReg::gpr(1).into(),
            src: Operand::Immediate(Immediate::I64(1)),
        });
        left.push(Instruction::Jmp {
            target: "merge".to_string(),
        });
        let mut right = Block::new("right");
        right.push(Instruction::Jmp {
            target: "merge".to_string(),
        });
        let mut merge = Block::new("merge");
        merge.push(Instruction::Ret {
            value: Some(Operand::Register(VirtualReg::gpr(1).into())),
        });
        function.add_block(entry);
        function.add_block(left);
        function.add_block(right);
        function.add_block(merge);

        let errors = verification_errors(&function);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("non-dominating definition"))
        );
    }

    #[test]
    fn accepts_definition_that_dominates_use() {
        let function = FunctionBuilder::new("good_dominance")
            .returns(MirType::i64())
            .block("entry")
            .instr(Instruction::Copy {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                src: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();

        assert!(verify_function(&function).is_ok());
    }

    #[test]
    fn rejects_instruction_type_disagreements() {
        let function = FunctionBuilder::new("bad_types")
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::f64(),
                dst: VirtualReg::fpr(0).into(),
                lhs: Operand::Immediate(Immediate::F64(1.0)),
                rhs: Operand::Immediate(Immediate::F64(2.0)),
            })
            .instr(Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: MirType::i64(),
                dst: VirtualReg::gpr(1).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::Eq,
                ty: MirType::i64(),
                dst: VirtualReg::fpr(2).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::FloatCmp {
                op: FloatCmpOp::Eq,
                ty: MirType::f64(),
                dst: VirtualReg::fpr(3).into(),
                lhs: Operand::Immediate(Immediate::F64(1.0)),
                rhs: Operand::Immediate(Immediate::F64(2.0)),
            })
            .instr(Instruction::Copy {
                ty: MirType::f64(),
                dst: VirtualReg::fpr(4).into(),
                src: Operand::Register(VirtualReg::gpr(1).into()),
            })
            .instr(Instruction::Select {
                ty: MirType::f64(),
                dst: VirtualReg::fpr(5).into(),
                cond: VirtualReg::fpr(2).into(),
                true_val: Operand::Register(VirtualReg::gpr(1).into()),
                false_val: Operand::Immediate(Immediate::F64(0.0)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        let errors = verification_errors(&function);
        assert!(errors.iter().any(|error| error.contains("IntBinary type")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("FloatBinary type"))
        );
        assert!(
            errors
                .iter()
                .filter(|error| error.contains("destination to have type i1"))
                .count()
                >= 2
        );
        assert!(errors.iter().any(|error| error.contains("Copy source")));
        assert!(errors.iter().any(|error| error.contains("Select true arm")));
    }

    #[test]
    fn accepts_instruction_type_agreement() {
        let function = FunctionBuilder::new("good_types")
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::i64(),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Immediate(Immediate::I64(1)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: MirType::f64(),
                dst: VirtualReg::fpr(1).into(),
                lhs: Operand::Immediate(Immediate::F64(1.0)),
                rhs: Operand::Immediate(Immediate::F64(2.0)),
            })
            .instr(Instruction::IntCmp {
                op: IntCmpOp::Eq,
                ty: MirType::i64(),
                dst: VirtualReg::gpr(2).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()),
                rhs: Operand::Immediate(Immediate::I64(3)),
            })
            .instr(Instruction::FloatCmp {
                op: FloatCmpOp::Eq,
                ty: MirType::f64(),
                dst: VirtualReg::gpr(3).into(),
                lhs: Operand::Register(VirtualReg::fpr(1).into()),
                rhs: Operand::Immediate(Immediate::F64(3.0)),
            })
            .instr(Instruction::Copy {
                ty: MirType::i1(),
                dst: VirtualReg::gpr(4).into(),
                src: Operand::Register(VirtualReg::gpr(3).into()),
            })
            .instr(Instruction::Select {
                ty: MirType::i64(),
                dst: VirtualReg::gpr(5).into(),
                cond: VirtualReg::gpr(2).into(),
                true_val: Operand::Register(VirtualReg::gpr(0).into()),
                false_val: Operand::Immediate(Immediate::I64(0)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        assert!(verify_function(&function).is_ok());
    }

    #[test]
    fn rejects_immediate_with_wrong_width() {
        let function = FunctionBuilder::new("bad_immediate")
            .block("entry")
            .instr(Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: MirType::f64(),
                dst: VirtualReg::fpr(0).into(),
                lhs: Operand::Immediate(Immediate::F64(1.0)),
                rhs: Operand::Immediate(Immediate::I64(2)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        let mut module = Module::new("bad_width");
        module.add_function(function.clone());
        assert!(module.validate_in_pipeline().is_err());
        let errors = verification_errors(&function);
        assert!(errors.iter().any(|error| {
            error.contains("expected FloatBinary rhs immediate to match type f64")
                && error.contains("found i64 immediate")
        }));
    }

    #[test]
    fn accepts_immediate_with_matching_width() {
        let function = FunctionBuilder::new("good_immediate")
            .block("entry")
            .instr(Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: MirType::f64(),
                dst: VirtualReg::fpr(0).into(),
                lhs: Operand::Immediate(Immediate::F64(1.0)),
                rhs: Operand::Immediate(Immediate::F64(2.0)),
            })
            .instr(Instruction::Ret { value: None })
            .build();

        assert!(verify_function(&function).is_ok());
    }
}
