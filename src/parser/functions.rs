//! Function parsing for Lamina IR.

use super::instructions::parse_instruction;
use super::state::ParserState;
use super::types::parse_type;
use crate::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature, Label,
    LaminaError,
};
use std::collections::{HashMap, HashSet};

/// Levenshtein edit distance for typo detection in annotation names.
/// Uses space-optimized dynamic programming with early termination.
fn edit_distance(s1: &str, s2: &str, max_distance: Option<usize>) -> usize {
    let s1_lower: Vec<char> = s1.to_lowercase().chars().collect();
    let s2_lower: Vec<char> = s2.to_lowercase().chars().collect();
    let m = s1_lower.len();
    let n = s2_lower.len();
    
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    
    if let Some(max) = max_distance {
        let len_diff = if m > n { m - n } else { n - m };
        if len_diff > max {
            return max + 1;
        }
    }
    
    let (shorter, longer) = if m <= n {
        (&s1_lower, &s2_lower)
    } else {
        (&s2_lower, &s1_lower)
    };
    let short_len = shorter.len();
    let long_len = longer.len();
    
    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    let mut curr_row = vec![0; short_len + 1];
    
    for i in 1..=long_len {
        curr_row[0] = i;
        
        for j in 1..=short_len {
            let cost = if longer[i - 1] == shorter[j - 1] { 0 } else { 1 };
            
            curr_row[j] = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
            
            if let Some(max) = max_distance {
                if curr_row[j] > max {
                    return max + 1;
                }
            }
        }
        
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    prev_row[short_len]
}

/// Parses function annotations.
pub fn parse_annotations(
    state: &mut ParserState<'_>,
) -> Result<Vec<FunctionAnnotation>, LaminaError> {
    let mut annotations = Vec::new();
    let mut seen = HashSet::new();
    
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('@') {
            state.advance();
            let name = state.parse_identifier_str()?;
            let annotation = match name {
                "inline" => FunctionAnnotation::Inline,
                "export" => FunctionAnnotation::Export,
                "extern" => FunctionAnnotation::Extern,
                "noreturn" => FunctionAnnotation::NoReturn,
                "noinline" => FunctionAnnotation::NoInline,
                "cold" => FunctionAnnotation::Cold,
                _ => {
                    let valid_annotations = ["inline", "export", "extern", "noreturn", "noinline", "cold"];
                    let mut suggestions = Vec::new();
                    const MAX_TYPO_DISTANCE: usize = 2;
                    
                    for valid in &valid_annotations {
                        let distance = edit_distance(name, valid, Some(MAX_TYPO_DISTANCE));
                        if distance <= MAX_TYPO_DISTANCE {
                            suggestions.push(*valid);
                        }
                    }
                    
                    suggestions.sort_by_key(|&s| edit_distance(name, s, None));
                    
                    let hint = if !suggestions.is_empty() {
                        if suggestions.len() == 1 {
                            format!("Did you mean @{}?", suggestions[0])
                        } else {
                            format!("Did you mean @{}?", suggestions.join(" or @"))
                        }
                    } else {
                        format!("Valid annotations are: {}", valid_annotations.join(", "))
                    };
                    
                    return Err(state.error(format!(
                        "Unknown function annotation: @{}\n  Hint: {}",
                        name, hint
                    )));
                }
            };
            
            if !seen.insert(annotation.clone()) {
                return Err(state.error(format!(
                    "Duplicate annotation: @{}",
                    name
                )));
            }
            
            if annotation == FunctionAnnotation::Inline && seen.contains(&FunctionAnnotation::NoInline) {
                return Err(state.error(
                    "Conflicting annotations: @inline and @noinline cannot be used together".to_string()
                ));
            }
            if annotation == FunctionAnnotation::NoInline && seen.contains(&FunctionAnnotation::Inline) {
                return Err(state.error(
                    "Conflicting annotations: @noinline and @inline cannot be used together".to_string()
                ));
            }
            if annotation == FunctionAnnotation::Extern && seen.contains(&FunctionAnnotation::Export) {
                return Err(state.error(
                    "Conflicting annotations: @extern and @export cannot be used together".to_string()
                ));
            }
            if annotation == FunctionAnnotation::Export && seen.contains(&FunctionAnnotation::Extern) {
                return Err(state.error(
                    "Conflicting annotations: @export and @extern cannot be used together".to_string()
                ));
            }
            
            annotations.push(annotation);
        } else {
            break;
        }
    }
    Ok(annotations)
}

/// Parses a function definition.
pub fn parse_function_def<'a>(state: &mut ParserState<'a>) -> Result<Function<'a>, LaminaError> {
    let annotations = parse_annotations(state)?;
    state.consume_keyword("fn")?;
    let name = state.parse_type_identifier()?;
    let signature = parse_fn_signature(state)?;
    state.expect_char('{')?;

    let mut basic_blocks = HashMap::new();
    let mut entry_block_label: Option<Label<'a>> = None;

    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('}') {
            state.advance();
            break;
        }

        let (label, block) = parse_basic_block(state)?;

        if entry_block_label.is_none() {
            entry_block_label = Some(label);
        }

        if basic_blocks.insert(label, block).is_some() {
            return Err(state.error(format!(
                "Redefinition of basic block label: '{}'",
                label
            )));
        }
    }

    let entry_block = entry_block_label
        .ok_or_else(|| {
            state.error("Function must have at least one basic block".to_string())
        })?;

    Ok(Function {
        name,
        signature,
        annotations,
        basic_blocks,
        entry_block,
    })
}

/// Parses a function signature.
pub fn parse_fn_signature<'a>(
    state: &mut ParserState<'a>,
) -> Result<FunctionSignature<'a>, LaminaError> {
    state.expect_char('(')?;
    let params = parse_param_list(state)?;
    state.expect_char(')')?;
    state.consume_keyword("->")?;
    let return_type = parse_type(state)?;
    Ok(FunctionSignature {
        params,
        return_type,
    })
}

/// Parses a parameter list.
pub fn parse_param_list<'a>(
    state: &mut ParserState<'a>,
) -> Result<Vec<FunctionParameter<'a>>, LaminaError> {
    let mut params = Vec::new();
    let mut param_names = std::collections::HashSet::new();
    
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }

        let param_ty = parse_type(state)?;
        let param_name = state.parse_value_identifier()?;
        
        if !param_names.insert(param_name) {
            return Err(state.error(format!(
                "Duplicate parameter name: %{}",
                param_name
            )));
        }
        
        params.push(FunctionParameter {
            name: param_name,
            ty: param_ty,
            annotations: vec![],
        });

        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }
        state.expect_char(',')?;
    }
    Ok(params)
}

/// Parses a basic block.
pub fn parse_basic_block<'a>(
    state: &mut ParserState<'a>,
) -> Result<(Label<'a>, BasicBlock<'a>), LaminaError> {
    let label = state.parse_label_identifier()?;
    state.expect_char(':')?;

    let mut instructions = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        let pos = state.position();
        if state.parse_label_identifier().is_ok() && state.current_char() == Some(':') {
            state.set_position(pos);
            break;
        }
        state.set_position(pos);

        if state.current_char() == Some('}') {
            break;
        }

        let instruction = parse_instruction(state)?;
        let is_terminator = instruction.is_terminator();
        instructions.push(instruction);

        if is_terminator {
            break;
        }
    }

    if instructions.is_empty() || instructions.last().is_none_or(|last| !last.is_terminator()) {
        return Err(state.error(format!(
            "Basic block '{}' must end with a terminator instruction (ret, jmp, br)",
            label
        )));
    }

    Ok((label, BasicBlock { instructions }))
}
