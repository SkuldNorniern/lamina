//! Runtime compilation and execution
//!
//! This module provides runtime code generation (JIT compilation) using ras,
//! with optional sandboxing for secure execution.

pub mod sandbox;
pub mod compiler;

pub use compiler::RuntimeCompiler;
pub use sandbox::{Sandbox, SandboxConfig};

use crate::error::LaminaError;
use crate::mir::Module as MirModule;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Runtime compilation result
pub struct RuntimeResult {
    /// Executable memory containing compiled code
    pub memory: ras::ExecutableMemory,
    /// Function pointer (unsafe - caller must ensure signature matches)
    pub function_ptr: *const u8,
}

// Re-export ras types for convenience
pub use ras::{ExecutableMemory, RasRuntime};

/// Compile MIR module to executable memory using runtime compilation
pub fn compile_to_runtime(
    module: &MirModule,
    target_arch: TargetArchitecture,
    target_os: TargetOperatingSystem,
    function_name: Option<&str>,
) -> Result<RuntimeResult, LaminaError> {
    #[cfg(feature = "encoder")]
    {
        use ras::RasRuntime;
        use ras::assembler::RasAssembler;
        
        let mut assembler = RasAssembler::new(target_arch, target_os)
            .map_err(|e| LaminaError::ValidationError(format!("Failed to create assembler: {}", e)))?;
        
        // Always compile all functions (needed for internal function calls)
        let (code, function_offsets) = assembler.compile_mir_to_binary_function(module, None)
            .map_err(|e| LaminaError::ValidationError(format!("Runtime compilation failed: {}", e)))?;
        
        // Find the function offset for the requested function
        let function_offset = if let Some(name) = function_name {
            let offset = function_offsets.get(name)
                .or_else(|| {
                    if name.starts_with('@') {
                        function_offsets.get(&name[1..])
                    } else {
                        function_offsets.get(&format!("@{}", name))
                    }
                })
                .copied();
            eprintln!("[DEBUG] Function '{}' offset: {:?}, available: {:?}", 
                     name, offset, function_offsets.keys().collect::<Vec<_>>());
            offset
        } else {
            // If no function name specified, use the first function (offset 0)
            Some(0)
        };
        
        // Allocate writable memory
        let mut memory = ras::ExecutableMemory::allocate_writable(code.len())
            .map_err(|e| LaminaError::ValidationError(format!("Memory allocation failed: {}", e)))?;
        
        // Write code
        memory.write_code(&code)
            .map_err(|e| LaminaError::ValidationError(format!("Failed to write code: {}", e)))?;
        
        // Make executable
        memory.make_executable()
            .map_err(|e| LaminaError::ValidationError(format!("Failed to make memory executable: {}", e)))?;
        
        // Get function pointer, adjusting for function offset if specified
        let base_ptr = unsafe { memory.as_function_ptr::<u8>() };
        let function_ptr = if let Some(offset) = function_offset {
            let adjusted = unsafe { (base_ptr as usize + offset) as *const u8 };
            eprintln!("[DEBUG] Function pointer: base={:p}, offset={}, adjusted={:p}", base_ptr, offset, adjusted);
            
            // Debug: Print first few instruction bytes at the function pointer
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    let remaining = code.len().saturating_sub(offset);
                    let bytes = std::slice::from_raw_parts(adjusted, std::cmp::min(128, remaining));
                    eprintln!("[DEBUG] First 32 bytes at function: {:02x?}", bytes.iter().take(32).collect::<Vec<_>>());
                    
                    // Check alignment
                    if (adjusted as usize) % 4 != 0 {
                        eprintln!("[WARNING] Function pointer is not 4-byte aligned! Address: {:p}, alignment: {} bytes", adjusted, (adjusted as usize) % 4);
                    } else {
                        eprintln!("[DEBUG] Function pointer is 4-byte aligned");
                    }
                    
                    // Decode first few instructions
                    eprintln!("[DEBUG] Decoding first 16 instructions (64 bytes):");
                    for i in 0..std::cmp::min(16, bytes.len() / 4) {
                        let inst_bytes = &bytes[i*4..(i+1)*4];
                        let inst = u32::from_le_bytes([inst_bytes[0], inst_bytes[1], inst_bytes[2], inst_bytes[3]]);
                        // Decode instruction type
                        let opcode = (inst >> 26) & 0x3F;
                        let opcode_top = (inst >> 28) & 0xF;
                        let inst_type = if opcode == 0b100101 {
                            "BL"
                        } else if opcode_top == 0b00 && (inst >> 27) & 0x1 == 1 {
                            "STP/LDP"
                        } else if (inst >> 31) & 0x1 == 1 && (inst >> 29) & 0x3 == 0b10 {
                            "ADD/SUB"
                        } else if (inst >> 31) & 0x1 == 1 && (inst >> 29) & 0x3 == 0b00 {
                            "STR/LDR"
                        } else if (inst >> 31) & 0x1 == 1 && (inst >> 29) & 0x3 == 0b01 {
                            "MOV/ORR"
                        } else if (inst >> 25) & 0x7F == 0b11010110 {
                            "RET/BR"
                        } else {
                            "UNKNOWN"
                        };
                        eprintln!("  [{}] 0x{:08x} ({})", i, inst, inst_type);
                    }
                }
            }
            
            adjusted
        } else {
            eprintln!("[DEBUG] Function pointer: base={:p}, no offset", base_ptr);
            base_ptr
        };
        
        Ok(RuntimeResult {
            memory,
            function_ptr,
        })
    }
    #[cfg(not(feature = "encoder"))]
    {
        Err(LaminaError::ValidationError(
            "Runtime compilation requires the 'encoder' feature to be enabled".to_string(),
        ))
    }
}

