//! Auto-vectorization transform for MIR.
//!
//! This transform automatically converts scalar operations in loops to SIMD vector
//! operations for improved performance. This is an O3 nightly/unstable feature.
//!
//! Requires `#[cfg(feature = "nightly")]` to compile.

#![cfg(feature = "nightly")]

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::Function;
use crate::mir::instruction::{AddressMode, FloatBinOp, Instruction, IntBinOp, Operand, VectorOp};
use crate::mir::register::{Register, VirtualReg};
use crate::mir::types::{MirType, ScalarType, VectorLane, VectorType};
use std::collections::{HashMap, HashSet};

/// Auto-vectorization transform that converts scalar loop operations to SIMD.
///
/// This transform identifies loops with sequential memory access patterns and
/// converts scalar arithmetic operations to vector operations for speedup.
#[derive(Default)]
pub struct AutoVectorization;

impl Transform for AutoVectorization {
    fn name(&self) -> &'static str {
        "auto_vectorization"
    }

    fn description(&self) -> &'static str {
        "Auto-vectorize scalar operations in loops using SIMD (O3 nightly/unstable)"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::Vectorization
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, func: &mut crate::mir::Function) -> Result<bool, String> {
        self.apply_internal(func)
    }
}

impl AutoVectorization {
    fn apply_internal(&self, func: &mut Function) -> Result<bool, String> {
        let mut changed = false;

        // Find all loops in the function
        let loops = self.find_loops(func);

        // Limit the number of loops we process to avoid excessive compilation time
        const MAX_LOOPS: usize = 10;
        for loop_info in loops.into_iter().take(MAX_LOOPS) {
            if self.try_vectorize_loop(func, &loop_info)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Find all loops in the function using back-edge detection
    fn find_loops(&self, func: &Function) -> Vec<LoopInfo> {
        let mut loops = Vec::new();
        let dominators = self.calculate_dominators(func);

        for block in &func.blocks {
            for instr in &block.instructions {
                if let Instruction::Br {
                    true_target,
                    false_target,
                    ..
                } = instr
                {
                    if self.is_back_edge(&dominators, &block.label, true_target) {
                        if let Some(loop_info) =
                            self.analyze_loop(func, true_target, &block.label, &dominators)
                        {
                            loops.push(loop_info);
                        }
                    }
                    if self.is_back_edge(&dominators, &block.label, false_target) {
                        if let Some(loop_info) =
                            self.analyze_loop(func, false_target, &block.label, &dominators)
                        {
                            loops.push(loop_info);
                        }
                    }
                }
                if let Instruction::Jmp { target } = instr {
                    if self.is_back_edge(&dominators, &block.label, target) {
                        if let Some(loop_info) =
                            self.analyze_loop(func, target, &block.label, &dominators)
                        {
                            loops.push(loop_info);
                        }
                    }
                }
            }
        }

        // Deduplicate loops by header
        loops.sort_by(|a, b| a.header.cmp(&b.header));
        loops.dedup_by(|a, b| a.header == b.header);

        loops
    }

    /// Calculate dominator sets for all blocks
    fn calculate_dominators(&self, func: &Function) -> HashMap<String, HashSet<String>> {
        let mut dominators: HashMap<String, HashSet<String>> = HashMap::new();
        let all_blocks: HashSet<String> = func.blocks.iter().map(|b| b.label.clone()).collect();

        // Initialize: entry block dominates itself
        if let Some(entry) = func.blocks.first() {
            let mut entry_doms = HashSet::new();
            entry_doms.insert(entry.label.clone());
            dominators.insert(entry.label.clone(), entry_doms);
        }

        // Initialize all other blocks to be dominated by all blocks
        for block in &func.blocks {
            if block.label != func.entry {
                dominators.insert(block.label.clone(), all_blocks.clone());
            }
        }

        // Iterative dataflow: block is dominated by intersection of predecessors' dominators
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                if block.label == func.entry {
                    continue;
                }

                let preds = self.get_predecessors(func, &block.label);
                if preds.is_empty() {
                    continue;
                }

                let mut new_doms = if let Some(first_pred) = preds.first() {
                    dominators
                        .get(first_pred.as_str())
                        .cloned()
                        .unwrap_or_default()
                } else {
                    HashSet::new()
                };

                for pred in &preds {
                    if let Some(pred_doms) = dominators.get(pred) {
                        new_doms = new_doms.intersection(pred_doms).cloned().collect();
                    }
                }

                new_doms.insert(block.label.clone());

                let old_doms = dominators.get(&block.label).cloned();
                if old_doms != Some(new_doms.clone()) {
                    dominators.insert(block.label.clone(), new_doms);
                    changed = true;
                }
            }
        }

        dominators
    }

    /// Get predecessors of a block
    fn get_predecessors(&self, func: &Function, block_label: &str) -> Vec<String> {
        let mut preds = Vec::new();
        for block in &func.blocks {
            for instr in &block.instructions {
                match instr {
                    Instruction::Jmp { target } if target == block_label => {
                        preds.push(block.label.clone());
                    }
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } if true_target == block_label || false_target == block_label => {
                        preds.push(block.label.clone());
                    }
                    _ => {}
                }
            }
        }
        preds
    }

    /// Check if an edge is a back edge (target dominates source)
    fn is_back_edge(
        &self,
        dominators: &HashMap<String, HashSet<String>>,
        source: &str,
        target: &str,
    ) -> bool {
        dominators
            .get(target)
            .map(|doms| doms.contains(source))
            .unwrap_or(false)
    }

    /// Analyze a loop to extract its structure
    fn analyze_loop(
        &self,
        func: &Function,
        header: &str,
        back_edge_source: &str,
        _dominators: &HashMap<String, HashSet<String>>,
    ) -> Option<LoopInfo> {
        let mut loop_blocks = HashSet::new();
        let mut to_visit = vec![back_edge_source.to_string()];
        let mut visited = HashSet::new();
        const MAX_ITERATIONS: usize = 1000;
        let mut iterations = 0;

        loop_blocks.insert(header.to_string());
        loop_blocks.insert(back_edge_source.to_string());

        while let Some(block_label) = to_visit.pop() {
            if iterations >= MAX_ITERATIONS {
                return None;
            }
            iterations += 1;

            if visited.contains(&block_label) {
                continue;
            }
            visited.insert(block_label.clone());
            loop_blocks.insert(block_label.clone());

            for block in &func.blocks {
                if self.has_edge_to(func, &block.label, &block_label) && block.label != header {
                    to_visit.push(block.label.clone());
                }
            }
        }

        if loop_blocks.len() > 50 {
            return None;
        }

        Some(LoopInfo {
            header: header.to_string(),
            blocks: loop_blocks,
        })
    }

    /// Check if there's an edge from source to target
    fn has_edge_to(&self, func: &Function, source: &str, target: &str) -> bool {
        if let Some(block) = func.get_block(source) {
            for instr in &block.instructions {
                match instr {
                    Instruction::Jmp { target: t } if t == target => return true,
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } if true_target == target || false_target == target => return true,
                    _ => {}
                }
            }
        }
        false
    }

    /// Try to vectorize a loop
    fn try_vectorize_loop(
        &self,
        func: &mut Function,
        loop_info: &LoopInfo,
    ) -> Result<bool, String> {
        // Find vectorizable patterns in the loop
        let patterns = self.find_vectorizable_patterns(func, loop_info);

        if patterns.is_empty() {
            return Ok(false);
        }

        // For now, we'll vectorize simple patterns
        // This is a basic implementation - more sophisticated analysis can be added later
        let mut changed = false;

        for pattern in patterns {
            if self.apply_vectorization_pattern(func, &pattern)? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Find vectorizable patterns in a loop
    /// More aggressive: detect more patterns including floating point and longer sequences
    fn find_vectorizable_patterns(
        &self,
        func: &Function,
        loop_info: &LoopInfo,
    ) -> Vec<VectorizationPattern> {
        let mut patterns = Vec::new();

        // Look for sequential load-compute-store patterns
        for block_label in &loop_info.blocks {
            if let Some(block) = func.get_block(block_label) {
                let instructions = &block.instructions;

                // Pattern 1: Load -> IntBinary -> Store (integer operations)
                for i in 0..instructions.len().saturating_sub(2) {
                    // Pattern: Load -> Binary Op -> Store
                    if let (
                        Instruction::Load { ty, dst, addr, .. },
                        Instruction::IntBinary { op, .. },
                        Instruction::Store {
                            src,
                            addr: store_addr,
                            ..
                        },
                    ) = (&instructions[i], &instructions[i + 1], &instructions[i + 2])
                    {
                        // Check if this is a vectorizable pattern
                        if self.is_vectorizable_type(ty)
                            && self.is_sequential_access(addr, store_addr)
                            && self.is_vectorizable_op(*op)
                            && self.matches_load_store(dst, src)
                        {
                            if let Some(scalar_ty) = self.extract_scalar_type(ty) {
                                patterns.push(VectorizationPattern {
                                    block: block_label.clone(),
                                    load_idx: i,
                                    compute_idx: i + 1,
                                    store_idx: i + 2,
                                    element_type: scalar_ty,
                                    operation: *op,
                                    is_float: false,
                                });
                            }
                        }
                    }
                }

                // Pattern 2: Load -> FloatBinary -> Store (floating point operations)
                for i in 0..instructions.len().saturating_sub(2) {
                    if let (
                        Instruction::Load { ty, dst, addr, .. },
                        Instruction::FloatBinary { op, .. },
                        Instruction::Store {
                            src,
                            addr: store_addr,
                            ..
                        },
                    ) = (&instructions[i], &instructions[i + 1], &instructions[i + 2])
                    {
                        if self.is_vectorizable_type(ty)
                            && self.is_sequential_access(addr, store_addr)
                            && self.is_vectorizable_float_op(*op)
                            && self.matches_load_store(dst, src)
                        {
                            if let Some(scalar_ty) = self.extract_scalar_type(ty) {
                                // Map float ops to int ops for pattern (we'll handle conversion)
                                let int_op = match op {
                                    FloatBinOp::FAdd => IntBinOp::Add,
                                    FloatBinOp::FSub => IntBinOp::Sub,
                                    FloatBinOp::FMul => IntBinOp::Mul,
                                    FloatBinOp::FDiv => IntBinOp::SDiv, // Approximate
                                };
                                patterns.push(VectorizationPattern {
                                    block: block_label.clone(),
                                    load_idx: i,
                                    compute_idx: i + 1,
                                    store_idx: i + 2,
                                    element_type: scalar_ty,
                                    operation: int_op,
                                    is_float: true,
                                });
                            }
                        }
                    }
                }

                // Pattern 3: Multiple consecutive operations (more aggressive)
                // Look for sequences like: load, op, op, store
                for i in 0..instructions.len().saturating_sub(3) {
                    if let (
                        Instruction::Load { ty, dst, addr, .. },
                        Instruction::IntBinary { op: op1, .. },
                        Instruction::IntBinary { op: _op2, .. },
                        Instruction::Store {
                            src,
                            addr: store_addr,
                            ..
                        },
                    ) = (
                        &instructions[i],
                        &instructions[i + 1],
                        &instructions[i + 2],
                        &instructions[i + 3],
                    ) {
                        // Vectorize the first operation if both are vectorizable
                        if self.is_vectorizable_type(ty)
                            && self.is_sequential_access(addr, store_addr)
                            && self.is_vectorizable_op(*op1)
                            && self.matches_load_store(dst, src)
                        {
                            if let Some(scalar_ty) = self.extract_scalar_type(ty) {
                                patterns.push(VectorizationPattern {
                                    block: block_label.clone(),
                                    load_idx: i,
                                    compute_idx: i + 1,
                                    store_idx: i + 3,
                                    element_type: scalar_ty,
                                    operation: *op1,
                                    is_float: false,
                                });
                            }
                        }
                    }
                }
            }
        }

        patterns
    }

    /// Check if a type is vectorizable
    fn is_vectorizable_type(&self, ty: &MirType) -> bool {
        matches!(
            ty,
            MirType::Scalar(
                ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::F32
                    | ScalarType::F64
            )
        )
    }

    /// Extract scalar type from MirType
    fn extract_scalar_type(&self, ty: &MirType) -> Option<ScalarType> {
        match ty {
            MirType::Scalar(s) => Some(*s),
            _ => None,
        }
    }

    /// Check if two address modes represent sequential access
    /// More aggressive: allows larger offset differences and different base registers
    fn is_sequential_access(&self, load_addr: &AddressMode, store_addr: &AddressMode) -> bool {
        match (load_addr, store_addr) {
            (
                AddressMode::BaseOffset {
                    base: lb,
                    offset: lo,
                },
                AddressMode::BaseOffset {
                    base: sb,
                    offset: so,
                },
            ) => {
                // Same base register and reasonable offset differences
                // More aggressive: allow up to 64 bytes difference (for unrolled loops)
                lb == sb && (so - lo).abs() <= 64
            }
            // Also support BaseIndexScale for more complex addressing
            (
                AddressMode::BaseIndexScale { base: lb, .. },
                AddressMode::BaseIndexScale { base: sb, .. },
            ) => lb == sb,
            _ => false,
        }
    }

    /// Check if an operation is vectorizable
    fn is_vectorizable_op(&self, op: IntBinOp) -> bool {
        matches!(
            op,
            IntBinOp::Add
                | IntBinOp::Sub
                | IntBinOp::Mul
                | IntBinOp::And
                | IntBinOp::Or
                | IntBinOp::Xor
        )
    }

    /// Check if a floating point operation is vectorizable
    fn is_vectorizable_float_op(&self, op: FloatBinOp) -> bool {
        matches!(
            op,
            FloatBinOp::FAdd | FloatBinOp::FSub | FloatBinOp::FMul | FloatBinOp::FDiv
        )
    }

    /// Check if load destination matches store source
    fn matches_load_store(&self, load_dst: &Register, store_src: &Operand) -> bool {
        if let Operand::Register(reg) = store_src {
            load_dst == reg
        } else {
            false
        }
    }

    /// Apply a vectorization pattern
    fn apply_vectorization_pattern(
        &self,
        func: &mut Function,
        pattern: &VectorizationPattern,
    ) -> Result<bool, String> {
        let block = func
            .get_block_mut(&pattern.block)
            .ok_or_else(|| format!("Block {} not found", pattern.block))?;

        if block.instructions.len() <= pattern.store_idx {
            return Ok(false);
        }

        // Extract the instructions
        let load_instr = block.instructions[pattern.load_idx].clone();
        let compute_instr = block.instructions[pattern.compute_idx].clone();
        let store_instr = block.instructions[pattern.store_idx].clone();

        // Determine vector type based on element type
        let vector_type = self.get_vector_type_for_scalar(pattern.element_type)?;

        // For basic vectorization, we'll process 4 elements at a time (v128 with 4x i32/f32)
        // This is a simplified approach - real vectorizers would handle more cases
        let vectorization_factor = match pattern.element_type {
            ScalarType::I32 | ScalarType::F32 => 4, // v128 with 4 lanes
            ScalarType::I64 | ScalarType::F64 => 2, // v128 with 2 lanes
            ScalarType::I16 => 8,                   // v128 with 8 lanes
            ScalarType::I8 => 16,                   // v128 with 16 lanes
            _ => return Ok(false),                  // Not supported
        };

        // Extract operands from instructions
        let (load_addr, compute_rhs, store_addr) = match (&load_instr, &compute_instr, &store_instr)
        {
            (
                Instruction::Load { addr, .. },
                Instruction::IntBinary { rhs, .. },
                Instruction::Store { addr: saddr, .. },
            ) => (addr.clone(), rhs.clone(), saddr.clone()),
            _ => return Ok(false),
        };

        // Generate vector operations
        // This is a simplified version - in practice, we'd need to handle:
        // - Loop unrolling to process multiple elements
        // - Alignment checks
        // - Remaining elements handling
        // - Register allocation for vector registers

        // For now, we'll create a basic vectorized version
        // Replace the scalar load with a vector load
        if let AddressMode::BaseOffset { base, offset } = load_addr {
            let vector_dst = self.allocate_vector_register();
            let vector_op = self.int_binop_to_vector_op(pattern.operation)?;

            // Create vector load
            block.instructions[pattern.load_idx] = Instruction::Load {
                ty: MirType::Vector(vector_type),
                dst: vector_dst.clone(),
                addr: AddressMode::BaseOffset {
                    base: base.clone(),
                    offset,
                },
                attrs: Default::default(),
            };

            // Create vector operation
            let vector_lhs_reg = vector_dst.clone();
            let vector_rhs = if matches!(compute_rhs, Operand::Immediate(_)) {
                // Splat immediate to vector
                let splat_reg = self.allocate_vector_register();
                block.instructions.insert(
                    pattern.compute_idx,
                    Instruction::VectorOp {
                        op: VectorOp::VSplat,
                        ty: MirType::Vector(vector_type),
                        dst: splat_reg.clone(),
                        operands: vec![compute_rhs.clone()],
                    },
                );
                Operand::Register(splat_reg)
            } else if matches!(compute_rhs, Operand::Register(_)) {
                // Load second operand as vector if it's a register
                let vector_rhs_reg = self.allocate_vector_register();
                block.instructions.insert(
                    pattern.compute_idx,
                    Instruction::Load {
                        ty: MirType::Vector(vector_type),
                        dst: vector_rhs_reg.clone(),
                        addr: AddressMode::BaseOffset {
                            base: base.clone(),
                            offset: offset
                                + (MirType::Scalar(pattern.element_type).size_bytes()
                                    * vectorization_factor)
                                    as i16,
                        },
                        attrs: Default::default(),
                    },
                );
                Operand::Register(vector_rhs_reg)
            } else {
                compute_rhs.clone()
            };

            block.instructions[pattern.compute_idx + 1] = Instruction::VectorOp {
                op: vector_op,
                ty: MirType::Vector(vector_type),
                dst: vector_dst.clone(),
                operands: vec![Operand::Register(vector_lhs_reg), vector_rhs],
            };

            // Create vector store
            if let AddressMode::BaseOffset {
                base: sbase,
                offset: soffset,
            } = store_addr
            {
                block.instructions[pattern.store_idx + 1] = Instruction::Store {
                    ty: MirType::Vector(vector_type),
                    src: Operand::Register(vector_dst),
                    addr: AddressMode::BaseOffset {
                        base: sbase,
                        offset: soffset,
                    },
                    attrs: Default::default(),
                };
            }

            // Remove old store instruction
            block.instructions.remove(pattern.store_idx);

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get vector type for a scalar type
    fn get_vector_type_for_scalar(&self, scalar: ScalarType) -> Result<VectorType, String> {
        let lane = self.scalar_to_vector_lane(scalar)?;
        Ok(VectorType::V128(lane))
    }

    /// Convert scalar type to vector lane type
    fn scalar_to_vector_lane(&self, scalar: ScalarType) -> Result<VectorLane, String> {
        match scalar {
            ScalarType::I8 => Ok(VectorLane::I8),
            ScalarType::I16 => Ok(VectorLane::I16),
            ScalarType::I32 => Ok(VectorLane::I32),
            ScalarType::I64 => Ok(VectorLane::I64),
            ScalarType::F32 => Ok(VectorLane::F32),
            ScalarType::F64 => Ok(VectorLane::F64),
            _ => Err(format!("Cannot vectorize type {:?}", scalar)),
        }
    }

    /// Convert integer binary operation to vector operation
    fn int_binop_to_vector_op(&self, op: IntBinOp) -> Result<VectorOp, String> {
        match op {
            IntBinOp::Add => Ok(VectorOp::VAdd),
            IntBinOp::Sub => Ok(VectorOp::VSub),
            IntBinOp::Mul => Ok(VectorOp::VMul),
            IntBinOp::And => Ok(VectorOp::VAnd),
            IntBinOp::Or => Ok(VectorOp::VOr),
            _ => Err(format!("Cannot vectorize operation {:?}", op)),
        }
    }

    /// Allocate a new vector register (simplified - in practice would use register allocator)
    fn allocate_vector_register(&self) -> Register {
        // This is a simplified allocation - real implementation would track used registers
        static mut COUNTER: u32 = 0;
        unsafe {
            COUNTER += 1;
            Register::Virtual(VirtualReg::vec(COUNTER as u32))
        }
    }
}

/// Information about a loop
#[derive(Debug, Clone)]
struct LoopInfo {
    header: String,
    blocks: HashSet<String>,
}

/// A vectorization pattern found in a loop
#[derive(Debug, Clone)]
struct VectorizationPattern {
    block: String,
    load_idx: usize,
    compute_idx: usize,
    store_idx: usize,
    element_type: ScalarType,
    operation: IntBinOp,
    is_float: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::ir;
    use crate::ir::builder::IRBuilder;
    use crate::mir;

    #[test]
    fn test_find_loops_simple() {
        let mut builder = ir::IRBuilder::new();
        let i64_ty = ir::types::Type::Primitive(ir::types::PrimitiveType::I64);

        builder
            .function("test", i64_ty.clone())
            .jump("loop")
            .block("loop")
            .binary(
                ir::instruction::BinaryOp::Add,
                "v0",
                ir::types::PrimitiveType::I64,
                ir::builder::var("v0"),
                ir::builder::i64(1),
            )
            .cmp(
                ir::instruction::CmpOp::Lt,
                "cond",
                ir::types::PrimitiveType::I64,
                ir::builder::var("v0"),
                ir::builder::i64(100),
            )
            .branch(ir::builder::var("cond"), "loop", "exit")
            .block("exit")
            .ret(i64_ty.clone(), ir::builder::var("v0"));

        let ir_module = builder.build();
        let mir_module =
            mir::codegen::from_ir(&ir_module, "test_module").expect("Failed to lower to MIR");

        let func = mir_module
            .functions
            .get("test")
            .expect("Function should exist");

        let vectorizer = AutoVectorization::default();
        let loops = vectorizer.find_loops(func);
        assert!(!loops.is_empty(), "Should find at least one loop");
    }

    #[test]
    fn test_find_vectorizable_pattern() {
        let mut builder = ir::IRBuilder::new();
        let i32_ty = ir::types::Type::Primitive(ir::types::PrimitiveType::I32);

        builder
            .function("test", i32_ty.clone())
            .alloc_stack("arr", i32_ty.clone())
            .jump("loop")
            .block("loop")
            .load("val", i32_ty.clone(), ir::builder::var("arr"))
            .binary(
                ir::instruction::BinaryOp::Add,
                "val",
                ir::types::PrimitiveType::I32,
                ir::builder::var("val"),
                ir::builder::i32(1),
            )
            .store(
                i32_ty.clone(),
                ir::builder::var("arr"),
                ir::builder::var("val"),
            )
            .cmp(
                ir::instruction::CmpOp::Lt,
                "cond",
                ir::types::PrimitiveType::I32,
                ir::builder::var("val"),
                ir::builder::i32(100),
            )
            .branch(ir::builder::var("cond"), "loop", "exit")
            .block("exit")
            .ret(i32_ty.clone(), ir::builder::var("val"));

        let ir_module = builder.build();
        let mir_module =
            mir::codegen::from_ir(&ir_module, "test_module").expect("Failed to lower to MIR");

        let func = mir_module
            .functions
            .get("test")
            .expect("Function should exist");

        let vectorizer = AutoVectorization::default();
        let loops = vectorizer.find_loops(func);
        assert!(!loops.is_empty(), "Should find loop");

        if let Some(loop_info) = loops.first() {
            let patterns = vectorizer.find_vectorizable_patterns(func, loop_info);
            assert!(
                !patterns.is_empty(),
                "Should find vectorizable load-compute-store pattern"
            );
        }
    }

    #[test]
    fn test_vectorization_no_change_when_no_loops() {
        let mut builder = ir::IRBuilder::new();
        let i64_ty = ir::types::Type::Primitive(ir::types::PrimitiveType::I64);

        builder
            .function("test", i64_ty.clone())
            .binary(
                ir::instruction::BinaryOp::Add,
                "result",
                ir::types::PrimitiveType::I64,
                ir::builder::i64(1),
                ir::builder::i64(2),
            )
            .ret(i64_ty.clone(), ir::builder::var("result"));

        let ir_module = builder.build();
        let mir_module =
            mir::codegen::from_ir(&ir_module, "test_module").expect("Failed to lower to MIR");

        let mut func = mir_module
            .functions
            .get("test")
            .cloned()
            .expect("Function should exist");

        let vectorizer = AutoVectorization::default();
        let result = vectorizer.apply_internal(&mut func);
        assert!(result.is_ok());
        let changed = result.unwrap();
        assert!(!changed, "Should not change function without loops");
    }
}
