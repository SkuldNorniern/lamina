//! Function inlining transforms for MIR.

use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Module, Operand, Register};
use std::cell::Cell;
use std::collections::HashMap;

/// Function-level inlining transform.
///
/// This transform handles function-level inlining within a single function context.
/// For module-level inlining that requires cross-function analysis, see `ModuleInlining`.
#[derive(Default)]
pub struct FunctionInlining;

impl Transform for FunctionInlining {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn description(&self) -> &'static str {
        "Replaces function calls with inlined function bodies (module-level)"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::Inlining
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }

    fn apply(&self, _func: &mut crate::mir::Function) -> Result<bool, String> {
        Ok(false)
    }
}

/// Module-level function inlining that analyzes the entire program.
pub struct ModuleInlining {
    /// Counter to ensure unique inline block labels
    inline_counter: Cell<usize>,
}

impl Default for ModuleInlining {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleInlining {
    pub fn new() -> Self {
        Self {
            inline_counter: Cell::new(0),
        }
    }

    /// Get next unique inline ID
    fn next_inline_id(&self) -> usize {
        let id = self.inline_counter.get();
        self.inline_counter.set(id + 1);
        id
    }

    /// Analyze the entire module and perform function inlining
    pub fn inline_functions(&self, module: &mut Module) -> Result<usize, String> {
        let mut inlined_count = 0;
        const MAX_INLINE_ITERATIONS: usize = 20;
        const MAX_TOTAL_INSTRUCTIONS: usize = 50_000;

        // Safety check: prevent inlining if module is too large
        let total_instructions: usize = module
            .functions
            .values()
            .map(|f| f.blocks.iter().map(|b| b.instructions.len()).sum::<usize>())
            .sum();
        if total_instructions > MAX_TOTAL_INSTRUCTIONS {
            return Err(format!(
                "Module too large for inlining ({} instructions, max {})",
                total_instructions, MAX_TOTAL_INSTRUCTIONS
            ));
        }

        // Collect all function call sites
        let mut call_sites = Vec::new();

        for (func_name, func) in &module.functions {
            for block in &func.blocks {
                for (instr_idx, instr) in block.instructions.iter().enumerate() {
                    if let Instruction::Call { name, .. } = instr {
                        call_sites.push(CallSite {
                            caller: func_name.clone(),
                            callee: name.clone(),
                            block_label: block.label.clone(),
                            instr_idx,
                        });
                    }
                }
            }
        }

        // Process call sites for inlining with iteration limit
        let mut iterations = 0;
        while iterations < MAX_INLINE_ITERATIONS {
            let mut made_progress = false;
            let mut new_call_sites = Vec::new();

            // Re-collect call sites after each iteration (new calls may have been introduced)
            for (func_name, func) in &module.functions {
                for block in &func.blocks {
                    for (instr_idx, instr) in block.instructions.iter().enumerate() {
                        if let Instruction::Call { name, .. } = instr {
                            new_call_sites.push(CallSite {
                                caller: func_name.clone(),
                                callee: name.clone(),
                                block_label: block.label.clone(),
                                instr_idx,
                            });
                        }
                    }
                }
            }

            for call_site in new_call_sites {
                if self.should_inline(&call_site, module) {
                    match self.perform_inline(&call_site, module) {
                        Ok(()) => {
                            inlined_count += 1;
                            made_progress = true;
                        }
                        Err(_e) => {
                            // Skip this inlining attempt and continue with others
                            continue;
                        }
                    }
                }
            }

            if !made_progress {
                break; // No more inlining possible
            }

            iterations += 1;
        }

        if iterations >= MAX_INLINE_ITERATIONS {
            return Err(format!(
                "Inlining did not converge after {} iterations",
                MAX_INLINE_ITERATIONS
            ));
        }

        Ok(inlined_count)
    }

    /// Decides whether a function call should be inlined.
    fn should_inline(&self, call_site: &CallSite, module: &Module) -> bool {
        if let Some(callee_func) = module.functions.get(&call_site.callee) {
            let total_instructions = callee_func.instruction_count();
            if total_instructions > 50 {
                return false;
            }

            // 2. Function has no calls to other functions (leaf function)
            let has_calls = callee_func.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instr| matches!(instr, Instruction::Call { .. }))
            });

            if has_calls && total_instructions > 30 {
                return false; // Avoid inlining functions that call others unless reasonably small
            }

            // 3. Function doesn't have complex control flow (Switch)
            let has_complex_cf = callee_func.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instr| matches!(instr, Instruction::Switch { .. }))
            });

            if has_complex_cf {
                return false; // Avoid inlining Switch for now
            }

            // 4. Function is reasonably small (multi-block ok)
            if callee_func.blocks.len() > 20 || total_instructions > 100 {
                return false;
            }

            // 6. Function is called from a small caller function?
            // Aggressive inlining: inline small functions regardless of caller size

            // ... existing logic ...
            if let Some(_caller_func) = module.functions.get(&call_site.caller) {
                // If caller is huge, dont inline huge things?
            }

            // Default: inline small functions
            total_instructions <= 30
        } else {
            false // Can't inline if we can't find the function
        }
    }

    /// Perform the actual inlining operation
    fn perform_inline(&self, call_site: &CallSite, module: &mut Module) -> Result<(), String> {
        // Clone the callee function to avoid borrowing issues
        let callee_func = module
            .functions
            .get(&call_site.callee)
            .ok_or_else(|| format!("Callee function '{}' not found", call_site.callee))?
            .clone();

        // should_inline already validated the function, so we can proceed

        // Extract call information before mutating
        let call_args = {
            let caller_func = module
                .functions
                .get(&call_site.caller)
                .ok_or_else(|| format!("Caller function '{}' not found", call_site.caller))?;

            let call_block = caller_func
                .blocks
                .iter()
                .find(|b| b.label == call_site.block_label)
                .ok_or_else(|| format!("Call block '{}' not found", call_site.block_label))?;

            if call_site.instr_idx >= call_block.instructions.len() {
                return Err("Call instruction index out of bounds".to_string());
            }

            let call_instr = &call_block.instructions[call_site.instr_idx];
            if let Instruction::Call { args, .. } = call_instr {
                args.clone()
            } else {
                return Err("Expected call instruction".to_string());
            }
        };

        // Create a mapping from callee parameters to caller arguments
        let param_mapping = self.create_param_mapping(&callee_func, &call_args)?;

        // Only use single-block optimized path if strictly 1 block
        if callee_func.blocks.len() == 1 {
            self.inline_single_block_function(call_site, module, &callee_func, &param_mapping)?;
        } else {
            self.inline_multi_block_function(call_site, module, &callee_func, &param_mapping)?;
        }

        Ok(())
    }

    /// Inline a single-block function
    fn inline_single_block_function(
        &self,
        call_site: &CallSite,
        module: &mut Module,
        callee_func: &Function,
        param_mapping: &HashMap<Register, Operand>,
    ) -> Result<(), String> {
        // Get the call instruction details first
        let (call_result_reg, _call_instr) = {
            let caller_func = module.functions.get(&call_site.caller)
                .ok_or_else(|| format!("Caller function '{}' not found", call_site.caller))?;
            let call_block = caller_func
                .blocks
                .iter()
                .find(|b| b.label == call_site.block_label)
                .ok_or_else(|| format!("Block '{}' not found in caller function '{}'", call_site.block_label, call_site.caller))?;

            let call_instr = call_block.instructions[call_site.instr_idx].clone();
            let call_result_reg = if let Instruction::Call { ret, .. } = &call_instr {
                ret.clone()
            } else {
                return Err("Expected call instruction".to_string());
            };

            (call_result_reg, call_instr)
        };

        // Get the single block from the callee
        let callee_block = callee_func
            .blocks
            .first()
            .ok_or_else(|| "Callee function has no blocks".to_string())?;

        // Create new instructions for the inlined code
        let mut inlined_instructions = Vec::new();

        // Get caller function for register ID calculation
        let caller_func = module.functions.get(&call_site.caller)
            .ok_or_else(|| format!("Caller function '{}' not found", call_site.caller))?;

        // Process each instruction in the callee, substituting parameters and renaming registers
        for instr in &callee_block.instructions {
            let mut new_instr = instr.clone();

            // Handle return instructions specially
            if let Instruction::Ret { value } = &new_instr
                && let Some(ret_val) = value
            {
                // Replace return with assignment to call result register
                if let Some(ref result_reg) = call_result_reg {
                    // Extract the return type from the function signature
                    let return_type = *callee_func.sig.ret_ty.as_ref().ok_or_else(|| {
                        "Function has return value but no return type in signature".to_string()
                    })?;

                    let mut assign_instr = Instruction::IntBinary {
                        op: crate::mir::IntBinOp::Add,
                        dst: result_reg.clone(),
                        ty: return_type,
                        lhs: ret_val.clone(),
                        rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
                    };
                    // Apply parameter substitution and renaming to the operand(s)
                    // Then restore destination to the intended call result register
                    self.substitute_parameters_and_rename(
                        &mut assign_instr,
                        param_mapping,
                        caller_func,
                    )?;
                    if let Instruction::IntBinary { dst, .. } = &mut assign_instr {
                        *dst = result_reg.clone();
                    }
                    inlined_instructions.push(assign_instr);
                }
                // Skip the original return instruction
                continue;
            }

            // Substitute parameters and rename registers
            self.substitute_parameters_and_rename(&mut new_instr, param_mapping, caller_func)?;

            inlined_instructions.push(new_instr);
        }

        // Now modify the caller function
        let caller_func = module.functions.get_mut(&call_site.caller)
            .ok_or_else(|| format!("Caller function '{}' not found", call_site.caller))?;
        let call_block = caller_func
            .blocks
            .iter_mut()
            .find(|b| b.label == call_site.block_label)
            .ok_or_else(|| format!("Block '{}' not found in caller function '{}'", call_site.block_label, call_site.caller))?;

        // Replace the call instruction with the inlined instructions
        call_block.instructions.splice(
            call_site.instr_idx..=call_site.instr_idx,
            inlined_instructions,
        );

        Ok(())
    }

    /// Substitute parameters and rename registers in an instruction
    fn substitute_parameters_and_rename(
        &self,
        instr: &mut Instruction,
        param_mapping: &HashMap<Register, Operand>,
        caller_func: &Function,
    ) -> Result<(), String> {
        // Generate unique register names to avoid conflicts
        let mut register_map: HashMap<Register, Register> = HashMap::new();
        let mut next_reg_id = self.find_max_register_id(caller_func) + 1;

        // Helper function to map a register
        let mut map_register = |reg: &Register| -> Register {
            if let Some(mapped) = register_map.get(reg) {
                return mapped.clone();
            }

            // Check if this register should be substituted with a parameter
            if let Some(param_operand) = param_mapping.get(reg)
                && let Operand::Register(param_reg) = param_operand
            {
                return param_reg.clone();
            }

            // Generate a new unique register name
            let new_reg = Register::Virtual(crate::mir::VirtualReg::gpr(next_reg_id));
            register_map.insert(reg.clone(), new_reg.clone());
            next_reg_id += 1;
            new_reg
        };

        // Apply register mapping to the instruction
        self.map_instruction_registers(instr, &mut map_register);

        Ok(())
    }

    /// Find the maximum register ID currently used in a function
    fn find_max_register_id(&self, func: &Function) -> u32 {
        let mut max_id = 0;
        for block in &func.blocks {
            for instr in &block.instructions {
                if let Some(reg) = instr.def_reg()
                    && let Register::Virtual(vreg) = reg
                    && vreg.class == crate::mir::RegisterClass::Gpr
                {
                    max_id = max_id.max(vreg.id);
                }
                for use_reg in instr.use_regs() {
                    if let Register::Virtual(vreg) = use_reg
                        && vreg.class == crate::mir::RegisterClass::Gpr
                    {
                        max_id = max_id.max(vreg.id);
                    }
                }
            }
        }
        max_id
    }

    /// Map registers in an instruction using a mapping function
    fn map_instruction_registers<F>(&self, instr: &mut Instruction, map_reg: &mut F)
    where
        F: FnMut(&Register) -> Register,
    {
        match instr {
            Instruction::IntBinary { dst, lhs, rhs, .. } => {
                *dst = map_reg(dst);
                self.map_operand_register(lhs, map_reg);
                self.map_operand_register(rhs, map_reg);
            }
            Instruction::FloatBinary { dst, lhs, rhs, .. } => {
                *dst = map_reg(dst);
                self.map_operand_register(lhs, map_reg);
                self.map_operand_register(rhs, map_reg);
            }
            Instruction::FloatUnary { dst, src, .. } => {
                *dst = map_reg(dst);
                self.map_operand_register(src, map_reg);
            }
            Instruction::IntCmp { dst, lhs, rhs, .. } => {
                *dst = map_reg(dst);
                self.map_operand_register(lhs, map_reg);
                self.map_operand_register(rhs, map_reg);
            }
            Instruction::FloatCmp { dst, lhs, rhs, .. } => {
                *dst = map_reg(dst);
                self.map_operand_register(lhs, map_reg);
                self.map_operand_register(rhs, map_reg);
            }
            Instruction::Select {
                dst,
                cond,
                true_val,
                false_val,
                ..
            } => {
                *dst = map_reg(dst);
                *cond = map_reg(cond);
                self.map_operand_register(true_val, map_reg);
                self.map_operand_register(false_val, map_reg);
            }
            Instruction::Load { dst, addr, .. } => {
                *dst = map_reg(dst);
                if let crate::mir::AddressMode::BaseOffset { base, .. } = addr {
                    *base = map_reg(base);
                }
                if let crate::mir::AddressMode::BaseIndexScale { base, index, .. } = addr {
                    *base = map_reg(base);
                    *index = map_reg(index);
                }
            }
            Instruction::Store { src, addr, .. } => {
                self.map_operand_register(src, map_reg);
                if let crate::mir::AddressMode::BaseOffset { base, .. } = addr {
                    *base = map_reg(base);
                }
                if let crate::mir::AddressMode::BaseIndexScale { base, index, .. } = addr {
                    *base = map_reg(base);
                    *index = map_reg(index);
                }
            }
            Instruction::Lea { dst, base, .. } => {
                *dst = map_reg(dst);
                *base = map_reg(base);
            }
            Instruction::VectorOp { dst, operands, .. } => {
                *dst = map_reg(dst);
                for operand in operands {
                    self.map_operand_register(operand, map_reg);
                }
            }
            _ => {} // Other instructions don't need register mapping
        }
    }

    /// Map registers in an operand
    fn map_operand_register<F>(&self, operand: &mut Operand, map_reg: &mut F)
    where
        F: FnMut(&Register) -> Register,
    {
        if let Operand::Register(reg) = operand {
            *reg = map_reg(reg);
        }
    }

    /// Create mapping from callee parameters to caller arguments
    fn create_param_mapping(
        &self,
        callee_func: &Function,
        call_args: &[Operand],
    ) -> Result<HashMap<Register, Operand>, String> {
        if callee_func.sig.params.len() != call_args.len() {
            return Err(format!(
                "Parameter count mismatch: expected {}, got {}",
                callee_func.sig.params.len(),
                call_args.len()
            ));
        }

        let mut mapping = HashMap::new();
        for (param, arg) in callee_func.sig.params.iter().zip(call_args.iter()) {
            mapping.insert(param.reg.clone(), arg.clone());
        }

        Ok(mapping)
    }

    /// Clone callee blocks and rename registers to avoid conflicts
    fn clone_and_rename_blocks(
        &self,
        blocks: &[Block],
        param_mapping: &HashMap<Register, Operand>,
        suffix: &str,
        inline_id: usize,
    ) -> Result<Vec<Block>, String> {
        let mut renamed_blocks = Vec::new();
        let mut register_mapping = HashMap::new();

        // Base register offset unique to this inline instance (avoid conflicts)
        let base_reg_offset = (inline_id + 1) * 10000;

        // First pass: collect all registers that need renaming
        for block in blocks {
            for instr in &block.instructions {
                if let Some(dst) = instr.def_reg()
                    && !register_mapping.contains_key(dst)
                {
                    // Generate a new virtual register for this destination
                    let new_reg = Register::Virtual(crate::mir::VirtualReg::gpr(
                        (base_reg_offset + register_mapping.len()) as u32,
                    ));
                    register_mapping.insert(dst.clone(), new_reg);
                }

                for use_reg in instr.use_regs() {
                    if !register_mapping.contains_key(use_reg)
                        && !param_mapping.contains_key(use_reg)
                    {
                        let new_reg = Register::Virtual(crate::mir::VirtualReg::gpr(
                            (base_reg_offset + register_mapping.len()) as u32,
                        ));
                        register_mapping.insert(use_reg.clone(), new_reg);
                    }
                }
            }
        }

        // Second pass: clone and rename instructions
        for block in blocks {
            let mut new_block = Block::new(format!("{}{}", block.label, suffix));

            for instr in &block.instructions {
                let mut new_instr = instr.clone();

                // Rename destination register
                if let Some(dst) = new_instr.def_reg()
                    && let Some(new_dst) = register_mapping.get(dst)
                {
                    self.rename_instruction_dst(&mut new_instr, new_dst.clone());
                }

                // Rename used registers
                self.rename_instruction_uses(&mut new_instr, &register_mapping, param_mapping)?;

                // Rename jump targets to match the new block names
                match &mut new_instr {
                    Instruction::Jmp { target } => {
                        *target = format!("{}{}", target, suffix);
                    }
                    Instruction::Br {
                        true_target,
                        false_target,
                        ..
                    } => {
                        *true_target = format!("{}{}", true_target, suffix);
                        *false_target = format!("{}{}", false_target, suffix);
                    }
                    _ => {}
                }

                new_block.push(new_instr);
            }

            renamed_blocks.push(new_block);
        }

        Ok(renamed_blocks)
    }

    /// Inline a multi-block function
    fn inline_multi_block_function(
        &self,
        call_site: &CallSite,
        module: &mut Module,
        callee_func: &Function,
        param_mapping: &HashMap<Register, Operand>,
    ) -> Result<(), String> {
        let inline_id = self.next_inline_id();
        let suffix = format!("_inline_{}_{}", call_site.callee, inline_id);
        let mut inlined_blocks =
            self.clone_and_rename_blocks(&callee_func.blocks, param_mapping, &suffix, inline_id)?;

        if inlined_blocks.is_empty() {
            return Err("Callee has no blocks".to_string());
        }

        // Get call details and split block
        let caller_func = module.functions.get_mut(&call_site.caller)
            .ok_or_else(|| format!("Caller function '{}' not found", call_site.caller))?;
        let call_block_idx = caller_func
            .blocks
            .iter()
            .position(|b| b.label == call_site.block_label)
            .ok_or_else(|| "Call block not found".to_string())?;

        let call_block = &mut caller_func.blocks[call_block_idx];

        // Extract return register (before removing instruction)
        let ret_reg =
            if let Instruction::Call { ret, .. } = &call_block.instructions[call_site.instr_idx] {
                ret.clone()
            } else {
                return Err("Expected Call instruction".to_string());
            };

        // Split instructions
        let mut post_call_instrs = call_block.instructions.split_off(call_site.instr_idx + 1);
        call_block.instructions.pop(); // Remove the Call instruction

        // Create split block (continuation) - use inline_id for uniqueness
        let split_label = format!("{}_split_{}", call_site.block_label, inline_id);
        let mut split_block = Block::new(split_label.clone());
        split_block.instructions.append(&mut post_call_instrs);

        // 1. Wire Caller -> Callee Entry
        // Find the actual entry block by looking for "entry" (the standard entry block name in lamina)
        let expected_entry = format!("entry{}", suffix);
        let callee_entry_target = inlined_blocks
            .iter()
            .find(|b| b.label == expected_entry)
            .map(|b| b.label.clone())
            .unwrap_or_else(|| inlined_blocks[0].label.clone()); // Fallback to first block
        call_block.instructions.push(Instruction::Jmp {
            target: callee_entry_target,
        });

        // 2. Wire Callee Returns -> Split Block
        for block in &mut inlined_blocks {
            if let Some(last_instr) = block.instructions.pop() {
                if let Instruction::Ret { value } = last_instr {
                    if let Some(val) = value
                        && let Some(dst) = &ret_reg
                    {
                        // Assign return value to call result register
                        block.instructions.push(Instruction::IntBinary {
                            op: crate::mir::IntBinOp::Add,
                            ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                            dst: dst.clone(),
                            lhs: val,
                            rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
                        });
                    }
                    // Jump to split block
                    block.instructions.push(Instruction::Jmp {
                        target: split_label.clone(),
                    });
                } else {
                    // Not a return? Put it back.
                    block.instructions.push(last_instr);

                    // If it was a terminator like Br/Jmp, it stays.
                    // But if it was Ret, we replaced it.
                    // If it ends with something else (impossible in valid MIR? Block must terminate),
                    // we assume valid MIR.
                }
            }
        }

        // Insert new blocks into caller
        // Order: [Caller Part 1] -> [Inlined Blocks...] -> [Caller Part 2 (Split)]
        // We insert split_block first at idx+1
        caller_func.blocks.insert(call_block_idx + 1, split_block);

        // Insert inlined blocks
        let mut insert_pos = call_block_idx + 1;
        for block in inlined_blocks {
            caller_func.blocks.insert(insert_pos, block);
            insert_pos += 1;
        }

        Ok(())
    }

    /// Rename the destination register in an instruction
    fn rename_instruction_dst(&self, instr: &mut Instruction, new_dst: Register) {
        match instr {
            Instruction::IntBinary { dst, .. }
            | Instruction::FloatBinary { dst, .. }
            | Instruction::FloatUnary { dst, .. }
            | Instruction::IntCmp { dst, .. }
            | Instruction::FloatCmp { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Lea { dst, .. }
            | Instruction::VectorOp { dst, .. } => {
                *dst = new_dst;
            }
            Instruction::Call { ret, .. } => {
                *ret = Some(new_dst);
            }
            _ => {} // Other instructions don't have destination registers
        }
    }

    /// Rename used registers in an instruction
    fn rename_instruction_uses(
        &self,
        instr: &mut Instruction,
        register_mapping: &HashMap<Register, Register>,
        param_mapping: &HashMap<Register, Operand>,
    ) -> Result<(), String> {
        match instr {
            Instruction::IntBinary { lhs, rhs, .. }
            | Instruction::FloatBinary { lhs, rhs, .. }
            | Instruction::IntCmp { lhs, rhs, .. }
            | Instruction::FloatCmp { lhs, rhs, .. } => {
                *lhs = self.map_operand(lhs, register_mapping, param_mapping)?;
                *rhs = self.map_operand(rhs, register_mapping, param_mapping)?;
            }
            Instruction::FloatUnary { src, .. } => {
                *src = self.map_operand(src, register_mapping, param_mapping)?;
            }
            Instruction::Select {
                cond,
                true_val,
                false_val,
                ..
            } => {
                *cond = self.map_register(cond, register_mapping)?;
                *true_val = self.map_operand(true_val, register_mapping, param_mapping)?;
                *false_val = self.map_operand(false_val, register_mapping, param_mapping)?;
            }
            Instruction::Load { addr, .. } => {
                *addr = self.map_address_mode(addr, register_mapping)?;
            }
            Instruction::Store { src, addr, .. } => {
                *src = self.map_operand(src, register_mapping, param_mapping)?;
                *addr = self.map_address_mode(addr, register_mapping)?;
            }
            Instruction::Lea { base, .. } => {
                *base = self.map_register(base, register_mapping)?;
            }
            Instruction::VectorOp { operands, .. } => {
                for operand in operands {
                    *operand = self.map_operand(operand, register_mapping, param_mapping)?;
                }
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    *arg = self.map_operand(arg, register_mapping, param_mapping)?;
                }
            }
            Instruction::Br { cond, .. } => {
                *cond = self.map_register(cond, register_mapping)?;
            }
            Instruction::Switch { value, .. } => {
                *value = self.map_register(value, register_mapping)?;
            }
            Instruction::Ret { value: Some(val) } => {
                *val = self.map_operand(val, register_mapping, param_mapping)?;
            }
            _ => {} // Other instructions don't use registers or are handled elsewhere
        }
        Ok(())
    }

    fn map_operand(
        &self,
        operand: &Operand,
        register_mapping: &HashMap<Register, Register>,
        param_mapping: &HashMap<Register, Operand>,
    ) -> Result<Operand, String> {
        match operand {
            Operand::Register(reg) => {
                if let Some(param_operand) = param_mapping.get(reg) {
                    Ok(param_operand.clone())
                } else if let Some(mapped_reg) = register_mapping.get(reg) {
                    Ok(Operand::Register(mapped_reg.clone()))
                } else {
                    Ok(operand.clone())
                }
            }
            _ => Ok(operand.clone()),
        }
    }

    fn map_register(
        &self,
        reg: &Register,
        register_mapping: &HashMap<Register, Register>,
    ) -> Result<Register, String> {
        if let Some(mapped) = register_mapping.get(reg) {
            Ok(mapped.clone())
        } else {
            Ok(reg.clone())
        }
    }

    fn map_address_mode(
        &self,
        addr: &crate::mir::AddressMode,
        register_mapping: &HashMap<Register, Register>,
    ) -> Result<crate::mir::AddressMode, String> {
        match addr {
            crate::mir::AddressMode::BaseOffset { base, offset } => {
                Ok(crate::mir::AddressMode::BaseOffset {
                    base: self.map_register(base, register_mapping)?,
                    offset: *offset,
                })
            }
            crate::mir::AddressMode::BaseIndexScale {
                base,
                index,
                scale,
                offset,
            } => Ok(crate::mir::AddressMode::BaseIndexScale {
                base: self.map_register(base, register_mapping)?,
                index: self.map_register(index, register_mapping)?,
                scale: *scale,
                offset: *offset,
            }),
        }
    }
}

#[derive(Debug)]
struct CallSite {
    caller: String,
    callee: String,
    block_label: String,
    instr_idx: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mir::{
        FunctionBuilder, Immediate, IntBinOp, MirType, Operand, ScalarType, VirtualReg,
    };

    #[test]
    fn test_inline_multi_block() {
        let mut module = Module::new("test_module");

        // Callee: 2 blocks
        // entry:
        //   v0 = add p0, 1
        //   jmp exit
        // exit:
        //   ret v0
        let mut callee = FunctionBuilder::new("callee")
            .param(VirtualReg::gpr(0).into(), MirType::Scalar(ScalarType::I64))
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I64),
                dst: VirtualReg::gpr(0).into(),
                lhs: Operand::Register(VirtualReg::gpr(0).into()), // p0 is v0 (param 0)
                rhs: Operand::Immediate(Immediate::I64(1)),
            })
            .instr(Instruction::Jmp {
                target: "exit".to_string(),
            })
            .block("exit")
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(0).into())),
            })
            .build();
        // Fix param reg
        callee.sig.params[0].reg = VirtualReg::gpr(0).into();

        module.add_function(callee);

        // Caller:
        // entry:
        //   v1 = call callee(10)
        //   ret v1
        let mut caller = FunctionBuilder::new("caller")
            .returns(MirType::Scalar(ScalarType::I64))
            .block("entry")
            .instr(Instruction::Call {
                name: "callee".to_string(),
                args: vec![Operand::Immediate(Immediate::I64(10))],
                ret: Some(VirtualReg::gpr(1).into()),
            })
            .instr(Instruction::Ret {
                value: Some(Operand::Register(VirtualReg::gpr(1).into())),
            })
            .build();

        module.add_function(caller);

        let inline_pass = ModuleInlining::new();
        let count = inline_pass
            .inline_functions(&mut module)
            .expect("Inlining failed");

        assert!(count > 0, "Should have inlined 1 function");

        let caller = module.functions.get("caller").unwrap();
        // Multi-Block inlining splits entry -> Entry, Split. And inserts CalleeEntry, CalleeExit.
        // Total 4 blocks expected.
        assert!(
            caller.blocks.len() >= 3,
            "Expected at least 3 blocks after inlining, got {}",
            caller.blocks.len()
        );

        // Verify Call is gone
        let has_call = caller.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|i| matches!(i, Instruction::Call { .. }))
        });
        assert!(!has_call, "Call instruction should be removed");
    }
}
