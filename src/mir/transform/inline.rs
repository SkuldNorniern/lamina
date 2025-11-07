use super::{Transform, TransformCategory, TransformLevel};
use crate::mir::{Block, Function, Instruction, Module, Operand, Register};
use std::collections::HashMap;

/// Function Inlining Transform for the pipeline system
/// Note: This is a placeholder for function-level inlining.
/// Real inlining requires module-level analysis and is handled by ModuleInlining.
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
        // Function inlining requires module-level analysis to see other functions
        // This transform is handled separately at the module level by ModuleInlining
        Ok(false)
    }
}

/// Module-level function inlining that can analyze the entire program
pub struct ModuleInlining;

impl Default for ModuleInlining {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleInlining {
    pub fn new() -> Self {
        Self
    }

    /// Analyze the entire module and perform function inlining
    pub fn inline_functions(&self, module: &mut Module) -> Result<usize, String> {
        let mut inlined_count = 0;

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

        // Process call sites for inlining
        for call_site in call_sites {
            if self.should_inline(&call_site, module) {
                match self.perform_inline(&call_site, module) {
                    Ok(()) => {
                        inlined_count += 1;
                    }
                    Err(e) => {
                        // Skip this inlining attempt and continue with others
                        // This allows the compiler to continue even if some functions can't be inlined
                        continue;
                    }
                }
            }
        }

        Ok(inlined_count)
    }

    /// Decide whether a function call should be inlined
    fn should_inline(&self, call_site: &CallSite, module: &Module) -> bool {
        // Get the callee function
        if let Some(callee_func) = module.functions.get(&call_site.callee) {
            // Conservative heuristics for inlining decisions:

            // 1. Function is very small (few instructions)
            let total_instructions = callee_func.instruction_count();
            if total_instructions > 20 {
                return false; // Too large to inline
            }

            // 2. Function has no calls to other functions (leaf function)
            let has_calls = callee_func.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instr| matches!(instr, Instruction::Call { .. }))
            });

            if has_calls && total_instructions > 10 {
                return false; // Avoid inlining functions that call others unless very small
            }

            // 3. Function doesn't have complex control flow
            let has_complex_cf = callee_func.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instr| matches!(instr, Instruction::Switch { .. }))
            });

            if has_complex_cf && total_instructions > 15 {
                return false; // Avoid inlining complex control flow
            }

            // 4. Function is single-block (no control flow)
            if callee_func.blocks.len() != 1 {
                return false; // Can't handle multi-block functions yet
            }

            // 5. Function has no branches or jumps (strict control flow check)
            let has_control_flow = callee_func.blocks.iter().any(|block| {
                block.instructions.iter().any(|instr| {
                    matches!(
                        instr,
                        Instruction::Jmp { .. }
                            | Instruction::Br { .. }
                            | Instruction::Switch { .. }
                    )
                })
            });

            if has_control_flow {
                return false; // Can't handle control flow yet
            }

            // 6. Function is called from a small caller function
            if let Some(caller_func) = module.functions.get(&call_site.caller) {
                let caller_size = caller_func.instruction_count();
                if caller_size < 50 || total_instructions < caller_size / 10 {
                    return true; // Good candidate for inlining
                }
            }

            // Default: inline very small functions
            total_instructions <= 5
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

        // For now, only inline single-block functions without control flow
        if callee_func.blocks.len() == 1 {
            self.inline_single_block_function(call_site, module, &callee_func, &param_mapping)?;
        } else {
            return Err("Multi-block function inlining not yet implemented".to_string());
        }

        Ok(())
    }

    /// Validate that a function is suitable for inlining
    fn validate_for_inlining(&self, func: &Function) -> Result<(), String> {
        // Check for complex control flow that we can't handle yet
        for block in &func.blocks {
            for instr in &block.instructions {
                match instr {
                    Instruction::Jmp { .. }
                    | Instruction::Br { .. }
                    | Instruction::Switch { .. } => {
                        return Err(
                            "Function contains control flow - not suitable for simple inlining"
                                .to_string(),
                        );
                    }
                    Instruction::Call { .. } => {
                        return Err(
                            "Function contains nested calls - not suitable for simple inlining"
                                .to_string(),
                        );
                    }
                    _ => {} // Other instructions are OK
                }
            }
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
        let (call_result_reg, call_instr) = {
            let caller_func = module.functions.get(&call_site.caller).unwrap();
            let call_block = caller_func
                .blocks
                .iter()
                .find(|b| b.label == call_site.block_label)
                .unwrap();

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
        let caller_func = module.functions.get(&call_site.caller).unwrap();

        // Process each instruction in the callee, substituting parameters and renaming registers
        for instr in &callee_block.instructions {
            let mut new_instr = instr.clone();

            // Handle return instructions specially
            if let Instruction::Ret { value } = &new_instr
                && let Some(ret_val) = value {
                    // Replace return with assignment to call result register
                    if let Some(ref result_reg) = call_result_reg {
                        let assign_instr = Instruction::IntBinary {
                            op: crate::mir::IntBinOp::Add,
                            dst: result_reg.clone(),
                            ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                            lhs: ret_val.clone(),
                            rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
                        };
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
        let caller_func = module.functions.get_mut(&call_site.caller).unwrap();
        let call_block = caller_func
            .blocks
            .iter_mut()
            .find(|b| b.label == call_site.block_label)
            .unwrap();

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
                && let Operand::Register(param_reg) = param_operand {
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
                        && vreg.class == crate::mir::RegisterClass::Gpr {
                            max_id = max_id.max(vreg.id);
                        }
                for use_reg in instr.use_regs() {
                    if let Register::Virtual(vreg) = use_reg
                        && vreg.class == crate::mir::RegisterClass::Gpr {
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
    ) -> Result<Vec<Block>, String> {
        let mut renamed_blocks = Vec::new();
        let mut register_mapping = HashMap::new();

        // First pass: collect all registers that need renaming
        for block in blocks {
            for instr in &block.instructions {
                if let Some(dst) = instr.def_reg()
                    && !register_mapping.contains_key(dst) {
                        // Generate a new virtual register for this destination
                        // In a real implementation, we'd get this from a register allocator
                        let new_reg = Register::Virtual(crate::mir::VirtualReg::gpr(
                            register_mapping.len() as u32 + 1000, // Offset to avoid conflicts
                        ));
                        register_mapping.insert(dst.clone(), new_reg);
                    }

                for use_reg in instr.use_regs() {
                    if !register_mapping.contains_key(use_reg)
                        && !param_mapping.contains_key(use_reg)
                    {
                        let new_reg = Register::Virtual(crate::mir::VirtualReg::gpr(
                            register_mapping.len() as u32 + 1000,
                        ));
                        register_mapping.insert(use_reg.clone(), new_reg);
                    }
                }
            }
        }

        // Second pass: clone and rename instructions
        for block in blocks {
            let mut new_block = Block::new(format!("{}_inline", block.label));

            for instr in &block.instructions {
                let mut new_instr = instr.clone();

                // Rename destination register
                if let Some(dst) = new_instr.def_reg()
                    && let Some(new_dst) = register_mapping.get(dst) {
                        self.rename_instruction_dst(&mut new_instr, new_dst.clone());
                    }

                // Rename used registers
                self.rename_instruction_uses(&mut new_instr, &register_mapping, param_mapping)?;

                new_block.push(new_instr);
            }

            renamed_blocks.push(new_block);
        }

        Ok(renamed_blocks)
    }

    /// Replace the call instruction with inlined code
    fn replace_call_with_inline(
        &self,
        call_block: &mut Block,
        call_idx: usize,
        inlined_blocks: &[Block],
        call_instr: &Instruction,
    ) -> Result<(), String> {
        let mut new_instructions = Vec::new();

        // Add instructions before the call
        for instr in &call_block.instructions[..call_idx] {
            new_instructions.push(instr.clone());
        }

        // Replace call with inlined blocks
        // For simplicity, inline all blocks sequentially
        // A real implementation would need proper control flow handling
        for block in inlined_blocks {
            for instr in &block.instructions {
                // Skip return instructions for now (simplified)
                if !matches!(instr, Instruction::Ret { .. }) {
                    new_instructions.push(instr.clone());
                } else if let Instruction::Ret { value } = instr {
                    // Replace return with assignment to call destination
                    if let Instruction::Call {
                        ret: Some(ret_reg), ..
                    } = call_instr
                        && let Some(return_val) = value {
                            // Create assignment: ret_reg = return_val
                            let assign_instr = Instruction::IntBinary {
                                op: crate::mir::IntBinOp::Add,
                                ty: crate::mir::MirType::Scalar(crate::mir::ScalarType::I64),
                                dst: ret_reg.clone(),
                                lhs: return_val.clone(),
                                rhs: Operand::Immediate(crate::mir::Immediate::I64(0)),
                            };
                            new_instructions.push(assign_instr);
                        }
                }
            }
        }

        // Add instructions after the call
        for instr in &call_block.instructions[call_idx + 1..] {
            new_instructions.push(instr.clone());
        }

        call_block.instructions = new_instructions;
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
            Instruction::Ret { value } => {
                if let Some(val) = value {
                    *val = self.map_operand(val, register_mapping, param_mapping)?;
                }
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
