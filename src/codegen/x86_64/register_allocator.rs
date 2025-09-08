use super::register_info::CALLEE_SAVED_REGISTERS;
use crate::{Function, Instruction, Result, Value};
use std::collections::{HashMap, HashSet};

/// Available x86_64 general-purpose registers for allocation
const ALLOCATABLE_REGISTERS: &[&str] = &[
    "%rax", "%rcx", "%rdx", "%rbx", "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13",
    "%r14", "%r15",
];

/// Live intervals for variables in the IR
#[derive(Debug, Clone)]
pub struct LiveInterval {
    pub var: String,
    pub start: usize,     // Instruction index where variable is defined
    pub end: usize,       // Last instruction index where variable is used
    pub uses: Vec<usize>, // All instruction indices where variable is used
}

/// Interference node in the graph
#[derive(Debug, Clone)]
pub struct InterferenceNode {
    pub var: String,
    pub neighbors: HashSet<String>,
    pub degree: usize,
    pub spill_cost: f64,
    pub preferred_registers: Vec<String>,
}

/// Result of register allocation
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub assignments: HashMap<String, String>, // Variable -> Register
    pub spilled_vars: HashSet<String>,        // Variables that need to be spilled
    pub callee_saved_used: HashSet<String>,   // Callee-saved registers actually used
}

/// Simplified graph coloring register allocator
#[derive(Debug)]
pub struct GraphColoringAllocator {
    live_intervals: HashMap<String, LiveInterval>,
    instruction_count: usize,
}

impl Default for GraphColoringAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphColoringAllocator {
    pub fn new() -> Self {
        Self {
            live_intervals: HashMap::new(),
            instruction_count: 0,
        }
    }

    /// Main entry point for register allocation
    pub fn allocate_registers<'a>(
        &mut self,
        function: &'a Function<'a>,
    ) -> Result<AllocationResult> {
        // Build live intervals for all variables
        self.build_live_intervals(function)?;

        // Apply simplified graph coloring
        self.simple_coloring()
    }

    /// Build live intervals for all variables in the function
    fn build_live_intervals<'a>(&mut self, function: &'a Function<'a>) -> Result<()> {
        self.instruction_count = 0;
        let mut var_first_def = HashMap::new();
        let mut var_last_use = HashMap::new();
        let mut var_uses = HashMap::new();

        // Process each basic block in sorted order for determinism
        let mut sorted_blocks: Vec<_> = function.basic_blocks.keys().collect();
        sorted_blocks.sort();
        for block_label in sorted_blocks {
            let block = &function.basic_blocks[block_label];
            for (idx, instruction) in block.instructions.iter().enumerate() {
                let abs_idx = self.instruction_count + idx;

                // Track variable definitions
                if let Some(def_var) = self.get_defined_variable(instruction) {
                    var_first_def.entry(def_var.clone()).or_insert(abs_idx);
                    var_last_use.insert(def_var.clone(), abs_idx);
                    var_uses.entry(def_var).or_insert(Vec::new()).push(abs_idx);
                }

                // Track variable uses
                for used_var in self.get_used_variables(instruction) {
                    var_last_use.insert(used_var.clone(), abs_idx);
                    var_uses.entry(used_var).or_insert(Vec::new()).push(abs_idx);
                }
            }
            self.instruction_count += block.instructions.len();
        }

        // Create live intervals
        for (var, &first_def) in &var_first_def {
            if let Some(&last_use) = var_last_use.get(var) {
                let uses = var_uses.get(var).cloned().unwrap_or_else(Vec::new);
                // Ensure start <= end for valid live intervals
                let (start, end) = if first_def <= last_use {
                    (first_def, last_use)
                } else {
                    // This can happen with phi nodes or malformed IR
                    // In such cases, extend the interval to cover both points
                    (first_def.min(last_use), first_def.max(last_use))
                };
                self.live_intervals.insert(
                    var.clone(),
                    LiveInterval {
                        var: var.clone(),
                        start,
                        end,
                        uses,
                    },
                );
            }
        }

        Ok(())
    }

    /// Simplified graph coloring algorithm
    fn simple_coloring(&self) -> Result<AllocationResult> {
        let mut result = AllocationResult {
            assignments: HashMap::new(),
            spilled_vars: HashSet::new(),
            callee_saved_used: HashSet::new(),
        };

        // Build interference lists (sorted for determinism)
        let mut interference_lists: HashMap<String, HashSet<String>> = HashMap::new();

        let mut sorted_vars: Vec<_> = self.live_intervals.keys().collect();
        sorted_vars.sort();

        for var1 in &sorted_vars {
            let interval1 = &self.live_intervals[*var1];
            let var1_key = (*var1).to_string();
            interference_lists.insert(var1_key.clone(), HashSet::new());
            for var2 in &sorted_vars {
                let interval2 = &self.live_intervals[*var2];
                if var1 != var2 && self.intervals_interfere(interval1, interval2) {
                    interference_lists
                        .get_mut(&var1_key)
                        .unwrap()
                        .insert((*var2).to_string());
                }
            }
        }

        // Simple greedy coloring (sorted for determinism)
        let mut used_registers = HashMap::new();
        let mut sorted_vars_for_coloring: Vec<_> = interference_lists.keys().collect();
        sorted_vars_for_coloring.sort();

        for var in sorted_vars_for_coloring {
            let interferences = &interference_lists[var];
            let mut forbidden_regs = HashSet::new();

            // Check what registers are used by interfering variables
            for interfering_var in interferences {
                if let Some(reg) = result.assignments.get(interfering_var) {
                    forbidden_regs.insert(reg.clone());
                }
            }

            // Find first available register
            let mut assigned = false;
            for &register in ALLOCATABLE_REGISTERS {
                if !forbidden_regs.contains(register) {
                    result
                        .assignments
                        .insert(var.to_string(), register.to_string());
                    used_registers.insert(register.to_string(), var.to_string());

                    // Track callee-saved usage
                    if CALLEE_SAVED_REGISTERS.contains(&register) {
                        result.callee_saved_used.insert(register.to_string());
                    }
                    assigned = true;
                    break;
                }
            }

            // If no register available, mark for spilling
            if !assigned {
                result.spilled_vars.insert(var.to_string());
            }
        }

        Ok(result)
    }

    /// Check if two live intervals interfere
    fn intervals_interfere(&self, interval1: &LiveInterval, interval2: &LiveInterval) -> bool {
        !(interval1.end < interval2.start || interval2.end < interval1.start)
    }

    /// Get variable defined by an instruction
    fn get_defined_variable(&self, instruction: &Instruction) -> Option<String> {
        match instruction {
            Instruction::Binary { result, .. } => Some(result.to_string()),
            Instruction::Cmp { result, .. } => Some(result.to_string()),
            Instruction::Load { result, .. } => Some(result.to_string()),
            Instruction::Alloc { result, .. } => Some(result.to_string()),
            Instruction::Call {
                result: Some(result),
                ..
            } => Some(result.to_string()),
            Instruction::ZeroExtend { result, .. } => Some(result.to_string()),
            Instruction::GetFieldPtr { result, .. } => Some(result.to_string()),
            Instruction::GetElemPtr { result, .. } => Some(result.to_string()),
            Instruction::Tuple { result, .. } => Some(result.to_string()),
            Instruction::ExtractTuple { result, .. } => Some(result.to_string()),
            Instruction::Phi { result, .. } => Some(result.to_string()),
            Instruction::Write { result, .. } => Some(result.to_string()),
            Instruction::Read { result, .. } => Some(result.to_string()),
            Instruction::WriteByte { result, .. } => Some(result.to_string()),
            Instruction::ReadByte { result, .. } => Some(result.to_string()),
            Instruction::WritePtr { result, .. } => Some(result.to_string()),
            Instruction::ReadPtr { result, .. } => Some(result.to_string()),
            _ => None,
        }
    }

    /// Get variables used by an instruction
    fn get_used_variables(&self, instruction: &Instruction) -> Vec<String> {
        let mut used_vars = Vec::new();

        match instruction {
            Instruction::Binary { lhs, rhs, .. } => {
                if let Value::Variable(var) = lhs {
                    used_vars.push(var.to_string());
                }
                if let Value::Variable(var) = rhs {
                    used_vars.push(var.to_string());
                }
            }
            Instruction::Cmp { lhs, rhs, .. } => {
                if let Value::Variable(var) = lhs {
                    used_vars.push(var.to_string());
                }
                if let Value::Variable(var) = rhs {
                    used_vars.push(var.to_string());
                }
            }
            Instruction::Load {
                ptr: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Store {
                ptr: Value::Variable(ptr_var),
                value: Value::Variable(value_var),
                ..
            } => {
                used_vars.push(ptr_var.to_string());
                used_vars.push(value_var.to_string());
            }
            Instruction::Store {
                ptr: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Store {
                value: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Ret {
                value: Some(Value::Variable(var)),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Br {
                condition: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    if let Value::Variable(var) = arg {
                        used_vars.push(var.to_string());
                    }
                }
            }
            Instruction::ZeroExtend {
                value: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Write {
                buffer: Value::Variable(buf_var),
                size: Value::Variable(size_var),
                ..
            } => {
                used_vars.push(buf_var.to_string());
                used_vars.push(size_var.to_string());
            }
            Instruction::Write {
                buffer: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Write {
                size: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Read {
                buffer: Value::Variable(buf_var),
                size: Value::Variable(size_var),
                ..
            } => {
                used_vars.push(buf_var.to_string());
                used_vars.push(size_var.to_string());
            }
            Instruction::Read {
                buffer: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::Read {
                size: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::WriteByte {
                value: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            Instruction::WritePtr {
                ptr: Value::Variable(var),
                ..
            } => {
                used_vars.push(var.to_string());
            }
            _ => {}
        }

        used_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_interval_creation() {
        let interval = LiveInterval {
            var: "test_var".to_string(),
            start: 0,
            end: 5,
            uses: vec![1, 3, 5],
        };

        assert_eq!(interval.var, "test_var");
        assert_eq!(interval.start, 0);
        assert_eq!(interval.end, 5);
    }

    #[test]
    fn test_interference_detection() {
        let allocator = GraphColoringAllocator::new();

        // Create two overlapping intervals
        let interval1 = LiveInterval {
            var: "var1".to_string(),
            start: 0,
            end: 3,
            uses: vec![0, 2],
        };

        let interval2 = LiveInterval {
            var: "var2".to_string(),
            start: 2,
            end: 5,
            uses: vec![2, 4],
        };

        assert!(allocator.intervals_interfere(&interval1, &interval2));

        // Non-overlapping intervals
        let interval3 = LiveInterval {
            var: "var3".to_string(),
            start: 6,
            end: 8,
            uses: vec![6, 7],
        };

        assert!(!allocator.intervals_interfere(&interval1, &interval3));
    }

    #[test]
    fn test_register_allocation_result() {
        let result = AllocationResult {
            assignments: HashMap::new(),
            spilled_vars: HashSet::new(),
            callee_saved_used: HashSet::new(),
        };

        assert!(result.assignments.is_empty());
        assert!(result.spilled_vars.is_empty());
        assert!(result.callee_saved_used.is_empty());
    }
}
