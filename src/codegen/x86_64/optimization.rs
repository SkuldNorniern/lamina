use crate::{Function, Instruction, Result};
use std::collections::HashMap;

/// Different optimization levels for x86_64 codegen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,       // No optimizations
    Basic,      // Basic optimizations (constant folding, simple CSE)
    Moderate,   // Include more advanced optimizations
    Aggressive, // All optimizations enabled
}

/// Optimization configuration for x86_64 backend
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub level: OptimizationLevel,
    pub enable_constant_folding: bool,
    pub enable_dead_code_elimination: bool,
    pub enable_common_subexpression_elimination: bool,
    pub enable_register_coalescing: bool,
    pub enable_instruction_combining: bool,
    pub enable_peephole_optimizations: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Basic,
            enable_constant_folding: true,
            enable_dead_code_elimination: true,
            enable_common_subexpression_elimination: false,
            enable_register_coalescing: false,
            enable_instruction_combining: false,
            enable_peephole_optimizations: false,
        }
    }
}

impl OptimizationConfig {
    pub fn new(level: OptimizationLevel) -> Self {
        let mut config = Self {
            level,
            ..Default::default()
        };

        match level {
            OptimizationLevel::None => {
                config.enable_constant_folding = false;
                config.enable_dead_code_elimination = false;
                config.enable_common_subexpression_elimination = false;
                config.enable_register_coalescing = false;
                config.enable_instruction_combining = false;
                config.enable_peephole_optimizations = false;
            }
            OptimizationLevel::Basic => {
                // Default values are already set for basic
            }
            OptimizationLevel::Moderate => {
                config.enable_common_subexpression_elimination = true;
                config.enable_peephole_optimizations = true;
            }
            OptimizationLevel::Aggressive => {
                config.enable_common_subexpression_elimination = true;
                config.enable_register_coalescing = true;
                config.enable_instruction_combining = true;
                config.enable_peephole_optimizations = true;
            }
        }

        config
    }
}

/// Simple peephole optimizations on assembly output
pub fn apply_peephole_optimizations(assembly_lines: &mut Vec<String>) -> Result<bool> {
    let mut changed = false;
    let mut i = 0;

    while i < assembly_lines.len().saturating_sub(1) {
        let current_line = assembly_lines[i].trim();

        // Pattern: movq %reg, %reg -> remove (redundant move)
        if current_line.starts_with("movq") {
            let parts: Vec<&str> = current_line.split_whitespace().collect();
            if parts.len() >= 3 {
                let src = parts[1].trim_end_matches(',');
                let dst = parts[2];
                if src == dst {
                    assembly_lines.remove(i);
                    changed = true;
                    continue;
                }
            }
        }

        // Pattern: addq $0, %reg -> remove (add zero)
        if current_line.starts_with("addq $0,") || current_line.starts_with("subq $0,") {
            assembly_lines.remove(i);
            changed = true;
            continue;
        }

        // Pattern: imulq $1, %reg -> remove (multiply by one)
        if current_line.starts_with("imulq $1,") {
            assembly_lines.remove(i);
            changed = true;
            continue;
        }

        i += 1;
    }

    Ok(changed)
}

/// Analyze function complexity to determine appropriate optimization level
pub fn analyze_function_complexity<'a>(function: &Function<'a>) -> OptimizationLevel {
    let total_instructions: usize = function
        .basic_blocks
        .values()
        .map(|block| block.instructions.len())
        .sum();

    let num_blocks = function.basic_blocks.len();
    let num_calls = function
        .basic_blocks
        .values()
        .flat_map(|block| &block.instructions)
        .filter(|instr| matches!(instr, Instruction::Call { .. }))
        .count();

    // Simple heuristic for optimization level
    if total_instructions < 10 && num_blocks <= 2 && num_calls == 0 {
        OptimizationLevel::Aggressive
    } else if total_instructions < 50 && num_blocks <= 5 {
        OptimizationLevel::Moderate
    } else if total_instructions < 200 {
        OptimizationLevel::Basic
    } else {
        OptimizationLevel::None // Very large functions - avoid expensive optimizations
    }
}

/// Simple register usage analysis for better register allocation
/// Placeholder function for future implementation
pub fn analyze_register_usage<'a>(function: &Function<'a>) -> HashMap<String, u32> {
    let mut _usage_count = HashMap::new();

    // This is a simplified analysis - in practice, we'd need more sophisticated
    // data flow analysis to track register pressure accurately
    for block in function.basic_blocks.values() {
        for _instr in &block.instructions {
            // Count how many times each variable is used
            // This would require more complex pattern matching, so keeping it simple for now
        }
    }

    _usage_count
}

/// Estimate the benefit of inlining a function
pub fn estimate_inlining_benefit<'a>(function: &Function<'a>, call_sites: usize) -> f64 {
    let instruction_count = function
        .basic_blocks
        .values()
        .map(|block| block.instructions.len())
        .sum::<usize>();

    let complexity_penalty = match instruction_count {
        0..=5 => 1.0,
        6..=15 => 0.8,
        16..=30 => 0.5,
        31..=50 => 0.2,
        _ => 0.1,
    };

    let call_benefit = (call_sites as f64).log2().max(1.0);

    complexity_penalty * call_benefit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config() {
        let none_config = OptimizationConfig::new(OptimizationLevel::None);
        assert!(!none_config.enable_constant_folding);

        let basic_config = OptimizationConfig::new(OptimizationLevel::Basic);
        assert!(basic_config.enable_constant_folding);
        assert!(!basic_config.enable_register_coalescing);

        let aggressive_config = OptimizationConfig::new(OptimizationLevel::Aggressive);
        assert!(aggressive_config.enable_register_coalescing);
        assert!(aggressive_config.enable_instruction_combining);
    }

    #[test]
    fn test_apply_peephole_optimizations() {
        let mut lines = vec![
            "movq %rax, %rax".to_string(),
            "addq $0, %rbx".to_string(),
            "imulq $1, %rcx".to_string(),
            "movq %rdx, %rsi".to_string(),
        ];

        let changed = apply_peephole_optimizations(&mut lines).unwrap();
        assert!(changed);

        // Should remove the first three lines
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "movq %rdx, %rsi");
    }

    #[test]
    fn test_analyze_function_complexity() {
        // This would require setting up a proper Function structure
        // For now, just test that the function compiles
        // In a real test, we'd create mock functions with different characteristics
    }

    #[test]
    fn test_estimate_inlining_benefit() {
        // Test with mock data
        let benefit = estimate_inlining_benefit(&create_mock_function(5), 3);
        assert!(benefit > 0.0);
        assert!(benefit <= 3.0); // Should be reasonable
    }

    fn create_mock_function(_instruction_count: usize) -> Function<'static> {
        // This is a placeholder - in a real test we'd create a proper mock function
        use crate::{Function, FunctionSignature, PrimitiveType, Type};
        use std::collections::HashMap;

        Function {
            name: "test_func",
            signature: FunctionSignature {
                params: vec![],
                return_type: Type::Primitive(PrimitiveType::I64),
            },
            basic_blocks: HashMap::new(),
            entry_block: "entry",
            annotations: vec![],
        }
    }
}
