use super::types::{StackFrame, ValueLocation, Parameter};
use super::utils::{get_type_size_bytes, get_type_alignment, align_to, align_to_signed};
use crate::{Function, Instruction, Type, Result};
use std::collections::HashMap;

/// Common stack layout computation that works for most architectures
pub struct StandardStackLayout {
    /// Stack grows downward if true
    pub stack_grows_down: bool,
    /// Stack alignment requirement (typically 16 bytes)
    pub stack_alignment: u64,
    /// Frame pointer offset from stack pointer
    pub frame_pointer_offset: i64,
}

impl StandardStackLayout {
    pub fn new(stack_grows_down: bool, stack_alignment: u64) -> Self {
        Self {
            stack_grows_down,
            stack_alignment,
            frame_pointer_offset: 0,
        }
    }

    /// Compute stack layout for a function with given calling convention
    pub fn compute_layout<'a>(
        &self,
        func: &'a Function<'a>,
        arg_registers: &[&str],
        value_locations: &mut HashMap<&'a str, ValueLocation>,
    ) -> Result<StackFrame> {
        let mut frame = StackFrame {
            total_size: 0,
            locals_size: 0,
            spills_size: 0,
            outgoing_args_size: 0,
            alignment: self.stack_alignment,
        };

        // Step 1: Assign parameter locations
        let mut stack_arg_offset = 16i64; // Start after saved frame pointer + return address
        for (i, param) in func.signature.params.iter().enumerate() {
            let location = if i < arg_registers.len() {
                // Parameter passed in register - will be spilled to stack
                ValueLocation::Register(arg_registers[i].to_string())
            } else {
                // Parameter passed on stack
                let loc = ValueLocation::StackOffset(stack_arg_offset);
                let param_size = get_type_size_bytes(&param.ty)?;
                stack_arg_offset += align_to(param_size, 8) as i64;
                loc
            };
            value_locations.insert(param.name, location);
        }

        // Step 2: Calculate space needed for local variables
        let mut local_vars = Vec::new();
        for block in func.basic_blocks.values() {
            for instr in &block.instructions {
                if let Some((var_name, var_type)) = self.extract_variable_info(instr) {
                    let size = get_type_size_bytes(&var_type)?;
                    let alignment = get_type_alignment(&var_type)?;
                    local_vars.push((var_name, size, alignment));
                }
            }
        }

        // Step 3: Assign stack offsets for local variables
        let mut current_offset = if self.stack_grows_down { -8i64 } else { 8i64 };
        
        for (var_name, size, alignment) in local_vars {
            // Align the offset for this variable
            if self.stack_grows_down {
                current_offset = align_to_signed(current_offset - size as i64, -(alignment as i64));
            } else {
                current_offset = align_to_signed(current_offset, alignment as i64);
            }
            
            value_locations.insert(var_name, ValueLocation::StackOffset(current_offset));
            frame.locals_size += size;
            
            if !self.stack_grows_down {
                current_offset += size as i64;
            }
        }

        // Step 4: Assign spill slots for register parameters
        for (i, param) in func.signature.params.iter().enumerate() {
            if i < arg_registers.len() {
                // This parameter was passed in a register, allocate spill slot
                let param_size = get_type_size_bytes(&param.ty)?;
                let param_align = get_type_alignment(&param.ty)?;
                
                if self.stack_grows_down {
                    current_offset = align_to_signed(current_offset - param_size as i64, -(param_align as i64));
                } else {
                    current_offset = align_to_signed(current_offset, param_align as i64);
                }
                
                // Update location to spill slot
                value_locations.insert(param.name, ValueLocation::StackOffset(current_offset));
                frame.spills_size += param_size;
                
                if !self.stack_grows_down {
                    current_offset += param_size as i64;
                }
            }
        }

        // Step 5: Calculate total frame size with alignment
        let total_used = frame.locals_size + frame.spills_size;
        frame.total_size = align_to(total_used, frame.alignment);

        Ok(frame)
    }

    /// Extract variable name and type from instruction that defines a variable
    fn extract_variable_info<'a>(&self, instr: &Instruction<'a>) -> Option<(&'a str, Type<'a>)> {
        match instr {
            Instruction::Alloc { result, allocated_ty, .. } => {
                Some((result, allocated_ty.clone()))
            }
            Instruction::Binary { result, ty, .. } => {
                Some((result, Type::Primitive(*ty)))
            }
            Instruction::Cmp { result, ty, .. } => {
                Some((result, Type::Primitive(*ty)))
            }
            Instruction::Load { result, ty, .. } => {
                Some((result, ty.clone()))
            }
            Instruction::Call { result: Some(result), .. } => {
                // Assume calls return pointer-sized values
                Some((result, Type::Primitive(crate::PrimitiveType::Ptr)))
            }
            Instruction::ZeroExtend { result, target_type, .. } => {
                Some((result, Type::Primitive(*target_type)))
            }
            Instruction::GetFieldPtr { result, .. } |
            Instruction::GetElemPtr { result, .. } => {
                Some((result, Type::Primitive(crate::PrimitiveType::Ptr)))
            }
            Instruction::Tuple { result, .. } |
            Instruction::ExtractTuple { result, .. } => {
                Some((result, Type::Primitive(crate::PrimitiveType::Ptr)))
            }
            Instruction::Phi { result, ty, .. } => {
                Some((result, ty.clone()))
            }
            _ => None,
        }
    }
}

/// Architecture-specific stack layout for x86_64
pub struct X86_64StackLayout;

impl X86_64StackLayout {
    pub fn new() -> StandardStackLayout {
        StandardStackLayout::new(true, 16) // Stack grows down, 16-byte aligned
    }
}

/// Architecture-specific stack layout for AArch64
pub struct AArch64StackLayout;

impl AArch64StackLayout {
    pub fn new() -> StandardStackLayout {
        StandardStackLayout::new(true, 16) // Stack grows down, 16-byte aligned
    }
}

/// Calculate maximum outgoing argument space needed by function
pub fn calculate_outgoing_args_size<'a>(func: &'a Function<'a>, arg_registers: &[&str]) -> Result<u64> {
    let mut max_stack_args = 0;
    
    for block in func.basic_blocks.values() {
        for instr in &block.instructions {
            if let Instruction::Call { args, .. } = instr {
                let stack_args = if args.len() > arg_registers.len() {
                    args.len() - arg_registers.len()
                } else {
                    0
                };
                max_stack_args = max_stack_args.max(stack_args);
            }
        }
    }
    
    // Each stack argument is typically 8 bytes (pointer-sized)
    Ok((max_stack_args * 8) as u64)
}

/// Optimize stack layout by reordering variables to minimize padding
pub fn optimize_stack_layout(variables: &mut Vec<(&str, u64, u64)>) {
    // Sort by alignment (descending) then by size (descending)
    // This minimizes padding between variables
    variables.sort_by(|a, b| {
        b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::PrimitiveType;

    #[test]
    fn test_standard_stack_layout() {
        let layout = StandardStackLayout::new(true, 16);
        assert!(layout.stack_grows_down);
        assert_eq!(layout.stack_alignment, 16);
    }

    #[test]
    fn test_calculate_outgoing_args_size() {
        // This would need a proper Function to test, but shows the concept
        let arg_registers = &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
        // Test would create a function with various call instructions
        // and verify the calculated outgoing args size
    }

    #[test]
    fn test_optimize_stack_layout() {
        let mut vars = vec![
            ("small", 1, 1),      // 1 byte, 1-byte aligned
            ("medium", 4, 4),     // 4 bytes, 4-byte aligned  
            ("large", 8, 8),      // 8 bytes, 8-byte aligned
            ("another_small", 2, 2), // 2 bytes, 2-byte aligned
        ];
        
        optimize_stack_layout(&mut vars);
        
        // Should be sorted by alignment desc, then size desc
        assert_eq!(vars[0].0, "large");     // 8-byte aligned
        assert_eq!(vars[1].0, "medium");    // 4-byte aligned
        assert_eq!(vars[2].0, "another_small"); // 2-byte aligned
        assert_eq!(vars[3].0, "small");     // 1-byte aligned
    }
} 