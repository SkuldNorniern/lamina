use super::types::ValueLocation;
use crate::{Instruction, PrimitiveType, Result, Type, Value};

/// Common instruction selection patterns that can be shared across architectures
pub trait InstructionSelector<'a> {
    /// Select the best instruction pattern for a given IR instruction
    fn select_instruction(&self, instr: &Instruction<'a>) -> Result<Vec<String>>;

    /// Get the optimal operand representation for a value
    fn select_operand(&self, value: &Value<'a>, location: &ValueLocation) -> String;

    /// Select the best addressing mode for memory operations
    fn select_addressing_mode(&self, base: &Value<'a>, offset: Option<i64>) -> String;
}

/// Common instruction patterns
pub enum InstructionPattern {
    /// Simple register-to-register operation
    RegisterToRegister {
        opcode: String,
        src: String,
        dst: String,
    },
    /// Register to memory operation
    RegisterToMemory {
        opcode: String,
        src: String,
        dst: String,
    },
    /// Memory to register operation
    MemoryToRegister {
        opcode: String,
        src: String,
        dst: String,
    },
    /// Immediate to register operation
    ImmediateToRegister {
        opcode: String,
        immediate: i64,
        dst: String,
    },
    /// Complex instruction with multiple operands
    Complex {
        template: String,
        operands: Vec<String>,
    },
}

/// Common addressing modes
#[derive(Debug, Clone)]
pub enum AddressingMode {
    /// Direct register
    Register(String),
    /// Immediate value
    Immediate(i64),
    /// Memory reference with base register
    Memory { base: String, offset: i64 },
    /// Memory reference with base + index
    IndexedMemory {
        base: String,
        index: String,
        scale: u8,
        offset: i64,
    },
    /// Global symbol reference
    Global(String),
}

impl AddressingMode {
    /// Convert addressing mode to assembly string (architecture-specific)
    pub fn to_asm_string(&self, syntax: ArchSyntax) -> String {
        match (self, syntax) {
            (AddressingMode::Register(reg), _) => reg.clone(),
            (AddressingMode::Immediate(val), ArchSyntax::ATT) => format!("${}", val),
            (AddressingMode::Immediate(val), ArchSyntax::Intel) => val.to_string(),
            (AddressingMode::Memory { base, offset }, ArchSyntax::ATT) => {
                if *offset == 0 {
                    format!("({})", base)
                } else {
                    format!("{}({})", offset, base)
                }
            }
            (AddressingMode::Memory { base, offset }, ArchSyntax::Intel) => {
                if *offset == 0 {
                    format!("[{}]", base)
                } else {
                    format!("[{} + {}]", base, offset)
                }
            }
            (
                AddressingMode::IndexedMemory {
                    base,
                    index,
                    scale,
                    offset,
                },
                ArchSyntax::ATT,
            ) => {
                format!("{}({},{},{})", offset, base, index, scale)
            }
            (
                AddressingMode::IndexedMemory {
                    base,
                    index,
                    scale,
                    offset,
                },
                ArchSyntax::Intel,
            ) => {
                format!("[{} + {} * {} + {}]", base, index, scale, offset)
            }
            (AddressingMode::Global(symbol), ArchSyntax::ATT) => format!("{}(%rip)", symbol),
            (AddressingMode::Global(symbol), ArchSyntax::Intel) => format!("[{}]", symbol),
        }
    }
}

/// Assembly syntax styles
#[derive(Debug, Clone, Copy)]
pub enum ArchSyntax {
    ATT,   // AT&T syntax (used by GAS)
    Intel, // Intel syntax
}

/// Instruction cost model for optimization
pub struct InstructionCost {
    /// Latency in cycles
    pub latency: u32,
    /// Throughput (instructions per cycle)
    pub throughput: f32,
    /// Code size in bytes
    pub size_bytes: u32,
}

/// Common instruction cost database
pub struct CostModel {
    costs: std::collections::HashMap<String, InstructionCost>,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    pub fn new() -> Self {
        Self {
            costs: std::collections::HashMap::new(),
        }
    }

    /// Add cost information for an instruction
    pub fn add_cost(&mut self, instruction: String, cost: InstructionCost) {
        self.costs.insert(instruction, cost);
    }

    /// Get cost for an instruction
    pub fn get_cost(&self, instruction: &str) -> Option<&InstructionCost> {
        self.costs.get(instruction)
    }

    /// Calculate total cost for a sequence of instructions
    pub fn calculate_sequence_cost(&self, instructions: &[String]) -> InstructionCost {
        let mut total_latency = 0;
        let mut total_size = 0;
        let mut min_throughput = f32::INFINITY;

        for instr in instructions {
            if let Some(cost) = self.get_cost(instr) {
                total_latency += cost.latency;
                total_size += cost.size_bytes;
                min_throughput = min_throughput.min(cost.throughput);
            }
        }

        InstructionCost {
            latency: total_latency,
            throughput: if min_throughput == f32::INFINITY {
                1.0
            } else {
                min_throughput
            },
            size_bytes: total_size,
        }
    }
}

/// Common instruction selection utilities
pub struct InstructionUtils;

impl InstructionUtils {
    /// Check if a value fits in an immediate field
    pub fn fits_in_immediate(value: i64, bits: u32) -> bool {
        let min_val = -(1i64 << (bits - 1));
        let max_val = (1i64 << (bits - 1)) - 1;
        value >= min_val && value <= max_val
    }

    /// Get register size suffix for a type
    pub fn get_size_suffix(ty: &Type, arch: &str) -> &'static str {
        match arch {
            "x86_64" => match ty {
                Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::Bool) => "b",
                Type::Primitive(PrimitiveType::I32) => "l",
                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => "q",
                Type::Primitive(PrimitiveType::F32) => "ss",
                _ => "q",
            },
            "aarch64" => match ty {
                Type::Primitive(PrimitiveType::I8) | Type::Primitive(PrimitiveType::Bool) => "b",
                Type::Primitive(PrimitiveType::I32) => "w",
                Type::Primitive(PrimitiveType::I64) | Type::Primitive(PrimitiveType::Ptr) => "x",
                Type::Primitive(PrimitiveType::F32) => "s",
                _ => "x",
            },
            _ => "",
        }
    }

    /// Check if an instruction can be folded into another
    pub fn can_fold_instruction(producer: &Instruction, consumer: &Instruction) -> bool {
        match (producer, consumer) {
            // Fold load into binary operation
            (Instruction::Load { result, .. }, Instruction::Binary { lhs, rhs, .. }) => {
                Value::Variable(result) == *lhs || Value::Variable(result) == *rhs
            }
            // Fold binary operation into store
            (Instruction::Binary { result, .. }, Instruction::Store { value, .. }) => {
                Value::Variable(result) == *value
            }
            _ => false,
        }
    }

    /// Select the best instruction variant based on operand types
    pub fn select_instruction_variant(
        base_opcode: &str,
        operand_types: &[ValueLocation],
        arch: &str,
    ) -> String {
        match arch {
            "x86_64" => {
                // x86_64 has many addressing modes and instruction variants
                match operand_types.len() {
                    2 => match (&operand_types[0], &operand_types[1]) {
                        (ValueLocation::Register(_), ValueLocation::Register(_)) => {
                            format!("{} %reg, %reg", base_opcode)
                        }
                        (ValueLocation::Immediate(_), ValueLocation::Register(_)) => {
                            format!("{} $imm, %reg", base_opcode)
                        }
                        (ValueLocation::StackOffset(_), ValueLocation::Register(_)) => {
                            format!("{} offset(%rbp), %reg", base_opcode)
                        }
                        _ => base_opcode.to_string(),
                    },
                    _ => base_opcode.to_string(),
                }
            }
            "aarch64" => {
                // AArch64 has more regular instruction encoding
                match operand_types.len() {
                    2 => format!("{} reg1, reg2", base_opcode),
                    3 => format!("{} reg1, reg2, reg3", base_opcode),
                    _ => base_opcode.to_string(),
                }
            }
            _ => base_opcode.to_string(),
        }
    }
}

/// Peephole optimization patterns
pub struct PeepholeOptimizer {
    patterns: Vec<OptimizationPattern>,
}

#[derive(Clone)]
pub struct OptimizationPattern {
    /// Pattern to match (sequence of instruction templates)
    pub pattern: Vec<String>,
    /// Replacement sequence
    pub replacement: Vec<String>,
    /// Description of the optimization
    pub description: String,
    /// Expected performance improvement
    pub improvement: f32,
}

impl PeepholeOptimizer {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add an optimization pattern
    pub fn add_pattern(&mut self, pattern: OptimizationPattern) {
        self.patterns.push(pattern);
    }

    /// Apply peephole optimizations to a sequence of instructions
    pub fn optimize(&self, instructions: &mut Vec<String>) -> usize {
        let mut optimizations_applied = 0;
        let mut i = 0;

        while i < instructions.len() {
            let mut matched = false;

            for pattern in &self.patterns {
                if i + pattern.pattern.len() <= instructions.len() {
                    let window = &instructions[i..i + pattern.pattern.len()];
                    if self.matches_pattern(window, &pattern.pattern) {
                        // Replace the matched sequence
                        instructions
                            .splice(i..i + pattern.pattern.len(), pattern.replacement.clone());
                        optimizations_applied += 1;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                i += 1;
            }
        }

        optimizations_applied
    }

    /// Check if a sequence matches a pattern
    fn matches_pattern(&self, sequence: &[String], pattern: &[String]) -> bool {
        if sequence.len() != pattern.len() {
            return false;
        }

        for (seq_instr, pat_instr) in sequence.iter().zip(pattern.iter()) {
            if !self.instruction_matches(seq_instr, pat_instr) {
                return false;
            }
        }

        true
    }

    /// Check if an instruction matches a pattern (with wildcards)
    fn instruction_matches(&self, instruction: &str, pattern: &str) -> bool {
        // Simple pattern matching - could be extended with regex
        if pattern == "*" {
            return true;
        }

        // Extract opcode from both
        let instr_opcode = instruction.split_whitespace().next().unwrap_or("");
        let pattern_opcode = pattern.split_whitespace().next().unwrap_or("");

        instr_opcode == pattern_opcode
    }
}

impl Default for PeepholeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addressing_mode_att_syntax() {
        let mode = AddressingMode::Memory {
            base: "%rbp".to_string(),
            offset: -8,
        };
        assert_eq!(mode.to_asm_string(ArchSyntax::ATT), "-8(%rbp)");

        let mode = AddressingMode::Immediate(42);
        assert_eq!(mode.to_asm_string(ArchSyntax::ATT), "$42");
    }

    #[test]
    fn test_addressing_mode_intel_syntax() {
        let mode = AddressingMode::Memory {
            base: "rbp".to_string(),
            offset: -8,
        };
        assert_eq!(mode.to_asm_string(ArchSyntax::Intel), "[rbp + -8]");
    }

    #[test]
    fn test_fits_in_immediate() {
        assert!(InstructionUtils::fits_in_immediate(127, 8));
        assert!(InstructionUtils::fits_in_immediate(-128, 8));
        assert!(!InstructionUtils::fits_in_immediate(128, 8));
        assert!(!InstructionUtils::fits_in_immediate(-129, 8));
    }

    #[test]
    fn test_cost_model() {
        let mut model = CostModel::new();
        model.add_cost(
            "add".to_string(),
            InstructionCost {
                latency: 1,
                throughput: 4.0,
                size_bytes: 3,
            },
        );

        let cost = model.get_cost("add").unwrap();
        assert_eq!(cost.latency, 1);
        assert_eq!(cost.throughput, 4.0);
        assert_eq!(cost.size_bytes, 3);
    }

    #[test]
    fn test_peephole_optimizer() {
        let mut optimizer = PeepholeOptimizer::new();
        optimizer.add_pattern(OptimizationPattern {
            pattern: vec!["mov %rax, %rbx".to_string(), "mov %rbx, %rax".to_string()],
            replacement: vec![], // Remove redundant moves
            description: "Remove redundant move pair".to_string(),
            improvement: 1.0,
        });

        let mut instructions = vec![
            "mov %rax, %rbx".to_string(),
            "mov %rbx, %rax".to_string(),
            "add %rcx, %rax".to_string(),
        ];

        let optimizations = optimizer.optimize(&mut instructions);
        assert_eq!(optimizations, 1);
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0], "add %rcx, %rax");
    }
}
