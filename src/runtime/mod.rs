//! Runtime compilation and execution
//!
//! Runtime code generation (JIT compilation) using ras,
//! with optional sandboxing for secure execution.

pub mod compiler;
pub mod executor;
#[cfg(feature = "encoder")]
pub mod macro_helpers;
pub mod sandbox;

pub use compiler::RuntimeCompiler;
pub use executor::execute_jit_function;
#[cfg(feature = "encoder")]
pub use macro_helpers::compile_lir_internal;
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
    _module: &MirModule,
    _target_arch: TargetArchitecture,
    _target_os: TargetOperatingSystem,
    _function_name: Option<&str>,
) -> Result<RuntimeResult, LaminaError> {
    #[cfg(feature = "encoder")]
    {
        use ras::assembler::RasAssembler;

        // Keep JIT output quiet by default (release-friendly). Enable with `LAMINA_JIT_DEBUG=1`.
        let jit_debug = std::env::var_os("LAMINA_JIT_DEBUG").is_some();

        let mut assembler = RasAssembler::new(_target_arch, _target_os).map_err(|e| {
            LaminaError::ValidationError(format!("Failed to create assembler: {}", e))
        })?;

        // Always compile all functions (needed for internal function calls)
        let (code, function_offsets) = assembler
            .compile_mir_to_binary_function(_module, None)
            .map_err(|e| {
                LaminaError::ValidationError(format!("Runtime compilation failed: {}", e))
            })?;

        // Find the function offset for the requested function
        let function_offset = if let Some(name) = _function_name {
            let offset = function_offsets
                .get::<str>(name)
                .or_else(|| {
                    if name.starts_with('@') {
                        function_offsets.get(&name[1..])
                    } else {
                        function_offsets.get(&format!("@{}", name))
                    }
                })
                .copied();
            if jit_debug {
                eprintln!(
                    "[JIT-DEBUG] Function '{}' offset: {:?}, available: {:?}",
                    name,
                    offset,
                    function_offsets.keys().collect::<Vec<_>>()
                );
            }
            offset
        } else {
            // If no function name specified, use the first function (offset 0)
            Some(0)
        };

        // Allocate writable memory
        let mut memory = ras::ExecutableMemory::allocate_writable(code.len()).map_err(|e| {
            LaminaError::ValidationError(format!("Memory allocation failed: {}", e))
        })?;

        // Write code
        memory
            .write_code(&code)
            .map_err(|e| LaminaError::ValidationError(format!("Failed to write code: {}", e)))?;

        // Make executable
        memory.make_executable().map_err(|e| {
            LaminaError::ValidationError(format!("Failed to make memory executable: {}", e))
        })?;

        // Get function pointer, adjusting for function offset if specified
        let base_ptr = unsafe { memory.as_function_ptr::<u8>() };
        let function_ptr = if let Some(offset) = function_offset {
            // Ensure offset is 4-byte aligned for AArch64
            #[cfg(target_arch = "aarch64")]
            {
                if offset % 4 != 0 {
                    return Err(LaminaError::ValidationError(format!(
                        "Function offset {} is not 4-byte aligned (required for AArch64)",
                        offset
                    )));
                }
            }
            let adjusted = (base_ptr as usize + offset) as *const u8;
            if jit_debug {
                eprintln!(
                    "[JIT-DEBUG] Function pointer: base={:p}, offset={}, adjusted={:p}",
                    base_ptr, offset, adjusted
                );
            }

            // Debug: Print first few instruction bytes at the function pointer
            #[cfg(target_arch = "aarch64")]
            {
                if jit_debug {
                    unsafe {
                        let remaining = code.len().saturating_sub(offset);
                        let bytes =
                            std::slice::from_raw_parts(adjusted, std::cmp::min(128, remaining));
                        eprintln!(
                            "[JIT-DEBUG] First 32 bytes at function: {:02x?}",
                            bytes.iter().take(32).collect::<Vec<_>>()
                        );

                        // Check alignment
                        if (adjusted as usize) % 4 != 0 {
                            eprintln!(
                                "[JIT-DEBUG][WARNING] Function pointer is not 4-byte aligned! Address: {:p}, alignment: {} bytes",
                                adjusted,
                                (adjusted as usize) % 4
                            );
                        } else {
                            eprintln!("[JIT-DEBUG] Function pointer is 4-byte aligned");
                        }

                        // Check last 4 bytes for RET instruction (should be c0 03 5f d6 for ret x30)
                        if bytes.len() >= 4 {
                            let last_4 = &bytes[bytes.len().saturating_sub(4)..];
                            eprintln!(
                                "[JIT-DEBUG] Last 4 bytes (RET instruction): {:02x?}",
                                last_4
                            );
                            let ret_inst =
                                u32::from_le_bytes([last_4[0], last_4[1], last_4[2], last_4[3]]);
                            eprintln!(
                                "[JIT-DEBUG] RET instruction value: 0x{:08x} (expected: 0xd65f03c0 for ret x30)",
                                ret_inst
                            );
                        }

                        // Decode first few instructions
                        eprintln!("[JIT-DEBUG] Decoding first 16 instructions (64 bytes):");
                        for i in 0..std::cmp::min(16, bytes.len() / 4) {
                            let inst_bytes = &bytes[i * 4..(i + 1) * 4];
                            let inst = u32::from_le_bytes([
                                inst_bytes[0],
                                inst_bytes[1],
                                inst_bytes[2],
                                inst_bytes[3],
                            ]);
                            // Decode instruction type
                            let opcode = (inst >> 26) & 0x3F;
                            let opcode_top = (inst >> 28) & 0xF;
                            let bits_29_27 = (inst >> 27) & 0x7;
                            let bits_25_24 = (inst >> 24) & 0x3;
                            let inst_type = if (inst >> 25) & 0x7F == 0b1101011 {
                                // RET/BR: [31:25]=1101011, [24]=0, [23:21]=010
                                "RET/BR"
                            } else if opcode == 0b100101 {
                                "BL"
                            } else if opcode_top == 0b00 && (inst >> 27) & 0x1 == 1 {
                                // STP/LDP: [31:30]=00, [27]=1
                                if (inst >> 28) & 0x1 == 0 {
                                    "STP"
                                } else {
                                    "LDP"
                                }
                            } else if opcode_top == 0b00 && (inst >> 27) & 0x1 == 0 {
                                // LDP (post-index): [31:30]=00, [28]=1, [27]=0
                                if (inst >> 28) & 0x1 == 1 {
                                    "LDP"
                                } else {
                                    "STP"
                                }
                            } else if bits_29_27 == 0b010
                                && (inst >> 23) & 0x3F == 0b100010
                                && opcode_top != 0b00
                            {
                                // ADD/SUB (immediate): [29:27]=010, [28:23]=100010, but not LDP/STP
                                if (inst >> 30) & 0x1 == 0 {
                                    "ADD(imm)"
                                } else {
                                    "SUB(imm)"
                                }
                            } else if bits_29_27 == 0b001
                                && (inst >> 23) & 0x3F == 0b010110
                                && (inst >> 30) & 0x3 == 0b10
                            {
                                // ADD (register, shifted register): [31:30]=10, [29:27]=001, [28:24]=01011, [23:22]=00
                                // This is ADD with no shift
                                "ADD(reg)"
                            } else if bits_29_27 == 0b010
                                && (inst >> 23) & 0x3F == 0b100010
                                && (inst >> 30) & 0x3 == 0b10
                            {
                                // ADD/SUB (register, extended): [31:30]=10, [29:27]=010, [28:23]=100010
                                if (inst >> 30) & 0x1 == 0 {
                                    "ADD(reg,ext)"
                                } else {
                                    "SUB(reg,ext)"
                                }
                            } else if bits_29_27 == 0b010 && (inst >> 23) & 0x3F == 0b100101 {
                                // MOVZ/MOVK: [29:27]=010, [28:23]=100101
                                if (inst >> 21) & 0x3 == 0b00 {
                                    "MOVZ"
                                } else {
                                    "MOVK"
                                }
                            } else if bits_29_27 == 0b111 && (inst >> 26) & 0x1 == 0 {
                                // STR/LDR unscaled immediate: [29:27]=111, [26]=0
                                if bits_25_24 == 0b00 {
                                    "STR(unscaled)"
                                } else if bits_25_24 == 0b01 {
                                    "LDR(unscaled)"
                                } else {
                                    "STR/LDR(scaled)"
                                }
                            } else if bits_29_27 == 0b001
                                && (inst >> 23) & 0x3F == 0b010100
                                && (inst >> 30) & 0x3 == 0b11
                            {
                                // EOR/XOR (register): [31:30]=11, [29:27]=001, [28:23]=010100
                                "EOR/XOR"
                            } else if (inst >> 31) & 0x1 == 1 && (inst >> 29) & 0x3 == 0b01 {
                                "MOV/ORR"
                            } else {
                                "UNKNOWN"
                            };
                            eprintln!("  [{}] 0x{:08x} ({})", i, inst, inst_type);
                        }
                    }
                }
            }

            adjusted
        } else {
            if jit_debug {
                eprintln!(
                    "[JIT-DEBUG] Function pointer: base={:p}, no offset",
                    base_ptr
                );
            }
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
