//! Function parsing for Lamina IR.

use super::instructions::parse_instruction;
use super::state::ParserState;
use super::types::parse_type;
use crate::{
    BasicBlock, Function, FunctionAnnotation, FunctionParameter, FunctionSignature, Label,
    LaminaError,
};
use std::collections::{HashMap, HashSet};

/// Calculates the Levenshtein edit distance between two strings.
///
/// This function computes the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to transform one string
/// into another. The comparison is case-insensitive for better typo detection.
///
/// # Arguments
///
/// * `s1` - First string to compare
/// * `s2` - Second string to compare
/// * `max_distance` - Maximum distance to consider (for early termination optimization)
///
/// # Returns
///
/// The edit distance between the two strings, or `max_distance + 1` if the
/// distance exceeds `max_distance` (for early termination).
///
/// # Examples
///
/// ```
/// # use crate::parser::functions::edit_distance;
/// assert_eq!(edit_distance("inline", "inlien", None), 2);
/// assert_eq!(edit_distance("export", "EXPORT", None), 0); // case-insensitive
/// assert_eq!(edit_distance("extern", "external", Some(2)), 3); // exceeds max
/// ```
fn edit_distance(s1: &str, s2: &str, max_distance: Option<usize>) -> usize {
    // Normalize to lowercase for case-insensitive comparison
    let s1_lower: Vec<char> = s1.to_lowercase().chars().collect();
    let s2_lower: Vec<char> = s2.to_lowercase().chars().collect();
    let m = s1_lower.len();
    let n = s2_lower.len();
    
    // Early exit for empty strings
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    
    // Early exit if length difference exceeds max_distance
    if let Some(max) = max_distance {
        let len_diff = if m > n { m - n } else { n - m };
        if len_diff > max {
            return max + 1;
        }
    }
    
    // Use space-optimized DP: only store two rows at a time
    // This reduces space complexity from O(m*n) to O(min(m,n))
    let (shorter, longer) = if m <= n {
        (&s1_lower, &s2_lower)
    } else {
        (&s2_lower, &s1_lower)
    };
    let short_len = shorter.len();
    let long_len = longer.len();
    
    // Previous row (dp[i-1])
    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    // Current row (dp[i])
    let mut curr_row = vec![0; short_len + 1];
    
    for i in 1..=long_len {
        curr_row[0] = i;
        
        for j in 1..=short_len {
            // Cost is 0 if characters match, 1 otherwise
            let cost = if longer[i - 1] == shorter[j - 1] { 0 } else { 1 };
            
            curr_row[j] = (prev_row[j] + 1)           // deletion
                .min(curr_row[j - 1] + 1)             // insertion
                .min(prev_row[j - 1] + cost);         // substitution
            
            // Early termination if we exceed max_distance
            if let Some(max) = max_distance {
                if curr_row[j] > max {
                    return max + 1;
                }
            }
        }
        
        // Swap rows for next iteration
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
                    // Suggest similar annotation names for typos
                    let valid_annotations = ["inline", "export", "extern", "noreturn", "noinline", "cold"];
                    let mut suggestions = Vec::new();
                    const MAX_TYPO_DISTANCE: usize = 2;
                    
                    // Find annotations within edit distance threshold
                    for valid in &valid_annotations {
                        let distance = edit_distance(name, valid, Some(MAX_TYPO_DISTANCE));
                        if distance <= MAX_TYPO_DISTANCE {
                            suggestions.push(*valid);
                        }
                    }
                    
                    // Sort suggestions by edit distance for better ordering
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
