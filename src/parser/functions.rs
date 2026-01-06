//! Function parsing for Lamina IR.

use super::instructions::parse_instruction;
use super::state::ParserState;
use super::types::parse_type;
use crate::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature, Label,
    LaminaError,
};
use std::collections::{HashMap, HashSet};

/// Simple edit distance calculation for typo suggestions
fn simple_edit_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();
    
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    
    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    
    dp[m][n]
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
                    // Suggest similar annotation names for typos
                    let valid_annotations = ["inline", "export", "extern", "noreturn", "noinline", "cold"];
                    let mut suggestions = Vec::new();
                    
                    // Simple edit distance check (Levenshtein-like)
                    for valid in &valid_annotations {
                        if simple_edit_distance(name, valid) <= 2 {
                            suggestions.push(*valid);
                        }
                    }
                    
                    let hint = if !suggestions.is_empty() {
                        format!("Did you mean @{}?", suggestions.join(" or @"))
                    } else {
                        format!("Valid annotations are: {}", valid_annotations.join(", "))
                    };
                    
                    return Err(state.error(format!(
                        "Unknown function annotation: @{}\n  Hint: {}",
                        name, hint
                    )));
                }
            };
            
            // Check for duplicate annotations
            if !seen.insert(annotation.clone()) {
                return Err(state.error(format!(
                    "Duplicate annotation: @{}\n  Hint: Each annotation can only appear once per function",
                    name
                )));
            }
            
            // Check for conflicting annotations
            if annotation == FunctionAnnotation::Inline && seen.contains(&FunctionAnnotation::NoInline) {
                return Err(state.error(
                    "Conflicting annotations: @inline and @noinline cannot be used together\n  Hint: Remove one of these conflicting annotations".to_string()
                ));
            }
            if annotation == FunctionAnnotation::NoInline && seen.contains(&FunctionAnnotation::Inline) {
                return Err(state.error(
                    "Conflicting annotations: @noinline and @inline cannot be used together\n  Hint: Remove one of these conflicting annotations".to_string()
                ));
            }
            if annotation == FunctionAnnotation::Extern && seen.contains(&FunctionAnnotation::Export) {
                return Err(state.error(
                    "Conflicting annotations: @extern and @export cannot be used together\n  Hint: @extern is for imported functions, @export is for exported functions".to_string()
                ));
            }
            if annotation == FunctionAnnotation::Export && seen.contains(&FunctionAnnotation::Extern) {
                return Err(state.error(
                    "Conflicting annotations: @export and @extern cannot be used together\n  Hint: @export is for exported functions, @extern is for imported functions".to_string()
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
                "Redefinition of basic block label: '{}'\n  Hint: Each basic block label must be unique within a function",
                label
            )));
        }
    }

    let entry_block = entry_block_label
        .ok_or_else(|| {
            // Provide annotation-specific error messages
            if annotations.contains(&FunctionAnnotation::Extern) {
                state.error("External functions must have at least one basic block (e.g., 'entry: ret.void')\n  Hint: Even external function declarations need a basic block structure for compatibility".to_string())
            } else if annotations.contains(&FunctionAnnotation::Export) {
                state.error("Exported functions must have at least one basic block\n  Hint: Exported functions require a full implementation with at least one basic block (e.g., 'entry:')".to_string())
            } else if annotations.contains(&FunctionAnnotation::Inline) {
                state.error("Inline functions must have at least one basic block\n  Hint: Inline functions require a full implementation with at least one basic block (e.g., 'entry:')".to_string())
            } else if annotations.contains(&FunctionAnnotation::NoReturn) {
                state.error("NoReturn functions must have at least one basic block\n  Hint: NoReturn functions require a full implementation with at least one basic block (e.g., 'entry:')".to_string())
            } else if annotations.contains(&FunctionAnnotation::NoInline) {
                state.error("NoInline functions must have at least one basic block\n  Hint: NoInline functions require a full implementation with at least one basic block (e.g., 'entry:')".to_string())
            } else if annotations.contains(&FunctionAnnotation::Cold) {
                state.error("Cold functions must have at least one basic block\n  Hint: Cold functions require a full implementation with at least one basic block (e.g., 'entry:')".to_string())
            } else {
                state.error("Function must have at least one basic block\n  Hint: Functions require at least one basic block (e.g., 'entry:')".to_string())
            }
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
        
        // Check for duplicate parameter names
        if !param_names.insert(param_name) {
            return Err(state.error(format!(
                "Duplicate parameter name: %{}\n  Hint: Each parameter must have a unique name",
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
        let _current_pos = state.position();
        if state.parse_label_identifier().is_ok() && state.current_char() == Some(':') {
            state.set_position(_current_pos);
            break;
        }
        state.set_position(_current_pos);

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
