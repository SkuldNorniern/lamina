use super::types::{AllocationResult, InterferenceNode, LiveRange};
use crate::{Function, Instruction, Result, Value};
use std::collections::{HashMap, HashSet};

/// Common interface for register allocators
pub trait RegisterAllocator<'a> {
    /// Allocate registers for a function
    fn allocate_registers(&mut self, func: &'a Function<'a>) -> Result<AllocationResult>;

    /// Get available registers for allocation
    fn available_registers(&self) -> &[String];

    /// Get callee-saved registers that are used
    fn get_callee_saved_used(&self) -> &HashSet<String>;
}

/// Simple linear scan register allocator
pub struct LinearScanAllocator {
    /// Available general-purpose registers
    available_registers: Vec<String>,
    /// Callee-saved registers that were used
    callee_saved_used: HashSet<String>,
}

impl LinearScanAllocator {
    pub fn new(registers: Vec<String>) -> Self {
        Self {
            available_registers: registers,
            callee_saved_used: HashSet::new(),
        }
    }
}

impl<'a> RegisterAllocator<'a> for LinearScanAllocator {
    fn allocate_registers(&mut self, func: &'a Function<'a>) -> Result<AllocationResult> {
        let live_ranges = compute_live_ranges(func)?;

        // Sort intervals by start point
        let mut sorted_intervals: Vec<(String, LiveRange)> = live_ranges.into_iter().collect();
        sorted_intervals.sort_by_key(|(_, range)| range.start);

        let mut assignments: HashMap<String, String> = HashMap::new();
        let mut spilled_vars = HashSet::new();
        let mut active_intervals: Vec<(String, LiveRange)> = Vec::new();

        for (var, interval) in sorted_intervals {
            // Remove expired intervals
            active_intervals.retain(|(_, active_interval)| active_interval.end > interval.start);

            // Try to find a free register
            let mut used_registers = HashSet::new();
            for (active_var, _) in &active_intervals {
                if let Some(reg) = assignments.get(active_var) {
                    used_registers.insert(reg.clone());
                }
            }

            let free_reg = self
                .available_registers
                .iter()
                .find(|reg| !used_registers.contains(*reg))
                .cloned();

            if let Some(reg) = free_reg {
                assignments.insert(var.clone(), reg.clone());
                active_intervals.push((var.clone(), interval));

                // Track callee-saved register usage
                if self.is_callee_saved(&reg) {
                    self.callee_saved_used.insert(reg);
                }
            } else {
                // Spill the variable
                spilled_vars.insert(var);
            }
        }

        Ok(AllocationResult {
            assignments,
            spilled_vars,
            callee_saved_used: self.callee_saved_used.clone(),
        })
    }

    fn available_registers(&self) -> &[String] {
        &self.available_registers
    }

    fn get_callee_saved_used(&self) -> &HashSet<String> {
        &self.callee_saved_used
    }
}

impl LinearScanAllocator {
    /// Check if a register is callee-saved (architecture-specific)
    fn is_callee_saved(&self, _reg: &str) -> bool {
        // This should be implemented per architecture
        false
    }
}

/// Graph coloring register allocator
pub struct GraphColoringAllocator {
    /// Available general-purpose registers
    available_registers: Vec<String>,
    /// Callee-saved registers that were used
    callee_saved_used: HashSet<String>,
}

impl GraphColoringAllocator {
    pub fn new(registers: Vec<String>) -> Self {
        Self {
            available_registers: registers,
            callee_saved_used: HashSet::new(),
        }
    }
}

impl<'a> RegisterAllocator<'a> for GraphColoringAllocator {
    fn allocate_registers(&mut self, func: &'a Function<'a>) -> Result<AllocationResult> {
        let live_ranges = compute_live_ranges(func)?;
        let interference_graph = build_interference_graph(&live_ranges);

        // Attempt graph coloring
        let coloring = color_graph(&interference_graph, &self.available_registers);

        let mut assignments = HashMap::new();
        let mut spilled_vars = HashSet::new();

        for (var, color) in coloring {
            match color {
                Some(reg) => {
                    assignments.insert(var.clone(), reg.clone());
                    if self.is_callee_saved(&reg) {
                        self.callee_saved_used.insert(reg);
                    }
                }
                None => {
                    spilled_vars.insert(var);
                }
            }
        }

        Ok(AllocationResult {
            assignments,
            spilled_vars,
            callee_saved_used: self.callee_saved_used.clone(),
        })
    }

    fn available_registers(&self) -> &[String] {
        &self.available_registers
    }

    fn get_callee_saved_used(&self) -> &HashSet<String> {
        &self.callee_saved_used
    }
}

impl GraphColoringAllocator {
    /// Check if a register is callee-saved (architecture-specific)
    fn is_callee_saved(&self, _reg: &str) -> bool {
        // This should be implemented per architecture
        false
    }
}

/// Compute live ranges for all variables in a function
pub fn compute_live_ranges<'a>(func: &'a Function<'a>) -> Result<HashMap<String, LiveRange>> {
    let mut live_ranges: HashMap<String, LiveRange> = HashMap::new();
    let mut instruction_index = 0;

    // First pass: find definitions and uses
    for block in func.basic_blocks.values() {
        for instr in &block.instructions {
            // Handle variable definitions
            if let Some(def_var) = get_defined_variable(instr) {
                live_ranges
                    .entry(def_var.to_string())
                    .or_insert_with(|| LiveRange {
                        var: def_var.to_string(),
                        start: instruction_index,
                        end: instruction_index,
                        uses: Vec::new(),
                        spill_cost: 1.0,
                    })
                    .start = instruction_index;
            }

            // Handle variable uses
            for use_var in get_used_variables(instr) {
                if let Some(range) = live_ranges.get_mut(&use_var) {
                    range.end = instruction_index;
                    range.uses.push(instruction_index);
                } else {
                    // Variable used before definition (probably a parameter)
                    live_ranges.insert(
                        use_var.clone(),
                        LiveRange {
                            var: use_var,
                            start: 0,
                            end: instruction_index,
                            uses: vec![instruction_index],
                            spill_cost: 1.0,
                        },
                    );
                }
            }

            instruction_index += 1;
        }
    }

    // Second pass: calculate spill costs based on usage frequency
    for range in live_ranges.values_mut() {
        range.spill_cost = calculate_spill_cost(range);
    }

    Ok(live_ranges)
}

/// Build interference graph from live ranges
pub fn build_interference_graph(
    live_ranges: &HashMap<String, LiveRange>,
) -> HashMap<String, InterferenceNode> {
    let mut graph = HashMap::new();

    // Initialize nodes
    for (var, range) in live_ranges {
        graph.insert(
            var.clone(),
            InterferenceNode {
                var: var.clone(),
                neighbors: HashSet::new(),
                degree: 0,
                spill_cost: range.spill_cost,
                preferred_registers: Vec::new(),
            },
        );
    }

    // Add edges for interfering ranges
    let vars: Vec<_> = live_ranges.keys().collect();
    for i in 0..vars.len() {
        for j in i + 1..vars.len() {
            let var1 = vars[i];
            let var2 = vars[j];
            let range1 = &live_ranges[var1];
            let range2 = &live_ranges[var2];

            if ranges_interfere(range1, range2) {
                if let Some(node1) = graph.get_mut(var1) {
                    node1.neighbors.insert(var2.clone());
                    node1.degree += 1;
                }
                if let Some(node2) = graph.get_mut(var2) {
                    node2.neighbors.insert(var1.clone());
                    node2.degree += 1;
                }
            }
        }
    }

    graph
}

/// Simple graph coloring algorithm
pub fn color_graph(
    graph: &HashMap<String, InterferenceNode>,
    available_colors: &[String],
) -> HashMap<String, Option<String>> {
    let mut coloring: HashMap<String, Option<String>> = HashMap::new();
    let mut remaining_nodes: Vec<_> = graph.keys().cloned().collect();

    // Sort by degree (lowest first for better chance of coloring)
    remaining_nodes.sort_by_key(|var| graph[var].degree);

    for var in remaining_nodes {
        let node = &graph[&var];

        // Find colors used by neighbors
        let mut used_colors = HashSet::new();
        for neighbor in &node.neighbors {
            if let Some(Some(color)) = coloring.get(neighbor) {
                used_colors.insert(color.clone());
            }
        }

        // Find first available color
        let assigned_color = available_colors
            .iter()
            .find(|color| !used_colors.contains(*color))
            .cloned();

        coloring.insert(var, assigned_color);
    }

    coloring
}

/// Check if two live ranges interfere
fn ranges_interfere(range1: &LiveRange, range2: &LiveRange) -> bool {
    !(range1.end < range2.start || range2.end < range1.start)
}

/// Calculate spill cost for a live range
fn calculate_spill_cost(range: &LiveRange) -> f64 {
    // Simple heuristic: cost increases with number of uses
    let use_count = range.uses.len() as f64;
    let range_length = (range.end - range.start) as f64;

    if range_length > 0.0 {
        use_count / range_length
    } else {
        use_count
    }
}

/// Extract the variable name defined by an instruction
fn get_defined_variable<'a>(instr: &Instruction<'a>) -> Option<&'a str> {
    match instr {
        Instruction::Binary { result, .. } => Some(result),
        Instruction::Cmp { result, .. } => Some(result),
        Instruction::Load { result, .. } => Some(result),
        Instruction::Alloc { result, .. } => Some(result),
        Instruction::Call {
            result: Some(result),
            ..
        } => Some(result),
        Instruction::ZeroExtend { result, .. } => Some(result),
        Instruction::Tuple { result, .. } => Some(result),
        Instruction::ExtractTuple { result, .. } => Some(result),
        Instruction::Phi { result, .. } => Some(result),
        _ => None,
    }
}

/// Extract variable names used by an instruction
fn get_used_variables<'a>(instr: &Instruction<'a>) -> Vec<String> {
    let mut vars = Vec::new();

    match instr {
        Instruction::Binary { lhs, rhs, .. } => {
            if let Value::Variable(var) = lhs {
                vars.push(var.to_string());
            }
            if let Value::Variable(var) = rhs {
                vars.push(var.to_string());
            }
        }
        Instruction::Cmp { lhs, rhs, .. } => {
            if let Value::Variable(var) = lhs {
                vars.push(var.to_string());
            }
            if let Value::Variable(var) = rhs {
                vars.push(var.to_string());
            }
        }
        Instruction::Load { ptr, .. } => {
            if let Value::Variable(var) = ptr {
                vars.push(var.to_string());
            }
        }
        Instruction::Store { ptr, value, .. } => {
            if let Value::Variable(var) = ptr {
                vars.push(var.to_string());
            }
            if let Value::Variable(var) = value {
                vars.push(var.to_string());
            }
        }
        Instruction::Ret {
            value: Some(value), ..
        } => {
            if let Value::Variable(var) = value {
                vars.push(var.to_string());
            }
        }
        Instruction::Br { condition, .. } => {
            if let Value::Variable(var) = condition {
                vars.push(var.to_string());
            }
        }
        Instruction::Call { args, .. } => {
            for arg in args {
                if let Value::Variable(var) = arg {
                    vars.push(var.to_string());
                }
            }
        }
        Instruction::Print { value } => {
            if let Value::Variable(var) = value {
                vars.push(var.to_string());
            }
        }
        Instruction::ZeroExtend { value, .. } => {
            if let Value::Variable(var) = value {
                vars.push(var.to_string());
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (value, _) in incoming {
                if let Value::Variable(var) = value {
                    vars.push(var.to_string());
                }
            }
        }
        _ => {}
    }

    vars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranges_interfere() {
        let range1 = LiveRange {
            var: "a".to_string(),
            start: 0,
            end: 5,
            uses: vec![1, 3, 5],
            spill_cost: 1.0,
        };

        let range2 = LiveRange {
            var: "b".to_string(),
            start: 3,
            end: 8,
            uses: vec![4, 6, 8],
            spill_cost: 1.0,
        };

        let range3 = LiveRange {
            var: "c".to_string(),
            start: 6,
            end: 10,
            uses: vec![7, 9],
            spill_cost: 1.0,
        };

        assert!(ranges_interfere(&range1, &range2)); // Overlap at 3-5
        assert!(!ranges_interfere(&range1, &range3)); // No overlap
        assert!(ranges_interfere(&range2, &range3)); // Overlap at 6-8
    }

    #[test]
    fn test_calculate_spill_cost() {
        let range = LiveRange {
            var: "test".to_string(),
            start: 0,
            end: 10,
            uses: vec![2, 4, 6, 8],
            spill_cost: 0.0,
        };

        let cost = calculate_spill_cost(&range);
        assert_eq!(cost, 0.4); // 4 uses / 10 range length
    }

    #[test]
    fn test_linear_scan_allocator() {
        let registers = vec!["%rax".to_string(), "%rbx".to_string()];
        let allocator = LinearScanAllocator::new(registers);

        assert_eq!(allocator.available_registers().len(), 2);
        assert!(allocator.get_callee_saved_used().is_empty());
    }
}
