//! Core assembler implementation
//!
//! This module contains the main RasAssembler struct and basic operations
//! that are architecture-independent.

use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
use crate::object::ObjectWriter;
use crate::parser::AssemblyParser;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// ras assembler - converts assembly text to object files
pub struct RasAssembler {
    pub(crate) target_arch: TargetArchitecture,
    pub(crate) target_os: TargetOperatingSystem,
    encoder: Box<dyn InstructionEncoder>,
    object_writer: Box<dyn ObjectWriter>,
    pub(crate) function_pointers: std::collections::HashMap<String, u64>, // Function name -> address
    #[cfg(feature = "encoder")]
    pub(crate) current_module: Option<*const lamina_mir::Module>, // Current module being compiled (for internal call detection)
}

impl RasAssembler {
    /// Create a new ras assembler
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        // Create encoder based on target architecture
        let encoder: Box<dyn InstructionEncoder> = match target_arch {
            TargetArchitecture::X86_64 => {
                Box::new(crate::encoder::x86_64::X86_64Encoder::new())
            }
            TargetArchitecture::Aarch64 => {
                Box::new(crate::encoder::aarch64::AArch64Encoder::new())
            }
            TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
                return Err(RasError::UnsupportedTarget(
                    "RISC-V encoder not yet implemented".to_string(),
                ));
            }
            _ => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported architecture: {:?}",
                    target_arch
                )));
            }
        };

        // Create object writer based on target OS
        let object_writer: Box<dyn ObjectWriter> = match target_os {
            TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD => {
                Box::new(crate::object::ElfWriter::new())
            }
            TargetOperatingSystem::MacOS => {
                Box::new(crate::object::MachOWriter::new())
            }
            TargetOperatingSystem::Windows => {
                Box::new(crate::object::CoffWriter::new())
            }
            _ => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported operating system: {:?}",
                    target_os
                )));
            }
        };

        Ok(Self {
            target_arch,
            target_os,
            encoder,
            object_writer,
            function_pointers: std::collections::HashMap::new(),
            #[cfg(feature = "encoder")]
            current_module: None,
        })
    }

    /// Assemble assembly text to object file
    pub fn assemble_text_to_object(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        // 1. Parse assembly text
        let mut parser = AssemblyParser::new();
        let parsed = parser
            .parse(asm_text)
            .map_err(|e| RasError::ParseError(e.to_string()))?;

        // 2. Encode instructions to binary
        let mut code = Vec::new();
        for inst in &parsed.instructions {
            let bytes = self
                .encoder
                .encode_instruction(inst)
                .map_err(|e| RasError::EncodingError(e.to_string()))?;
            code.extend_from_slice(&bytes);
        }

        // 3. Generate object file
        self.object_writer
            .write_object_file(
                output_path,
                &code,
                &parsed.sections,
                &parsed.symbols,
                self.target_arch,
                self.target_os,
            )
            .map_err(|e| RasError::ObjectError(e.to_string()))?;

        Ok(())
    }

    /// Register a function pointer for runtime calls
    /// This resolves the function using dlsym (on Unix) or GetProcAddress (on Windows)
    pub fn register_function(&mut self, name: &str) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use std::ffi::CString;
            
            let symbol = CString::new(name).map_err(|e| {
                RasError::EncodingError(format!("Invalid function name: {}", e))
            })?;
            
            // Try to resolve using RTLD_DEFAULT first (searches already loaded libraries)
            // This is safer and doesn't require opening/closing handles
            let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };
            
            if ptr.is_null() {
                // Fallback: try opening libc explicitly
                // Clear any previous error
                unsafe { libc::dlerror(); }
                
                let handle = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_LAZY) };
                if handle.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "unknown error (dlerror returned null)"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    return Err(RasError::EncodingError(
                        format!("Failed to open libc: {}", err_msg)
                    ));
                }
                
                // Clear error before dlsym
                unsafe { libc::dlerror(); }
                
                let ptr2 = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
                if ptr2.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "symbol not found"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    unsafe { libc::dlclose(handle) };
                    return Err(RasError::EncodingError(
                        format!("Failed to resolve symbol {}: {}", name, err_msg)
                    ));
                }
                
                self.function_pointers.insert(name.to_string(), ptr2 as u64);
                unsafe { libc::dlclose(handle) };
            } else {
                self.function_pointers.insert(name.to_string(), ptr as u64);
            }
            
            Ok(())
        }
        
        #[cfg(windows)]
        {
            use winapi::um::libloaderapi::{GetModuleHandleA, GetProcAddress};
            use std::ffi::CString;
            
            let module = unsafe { GetModuleHandleA(b"msvcrt.dll\0".as_ptr() as *const i8) };
            if module.is_null() {
                return Err(RasError::EncodingError("Failed to get msvcrt.dll handle".to_string()));
            }
            
            let symbol = CString::new(name).map_err(|e| {
                RasError::EncodingError(format!("Invalid function name: {}", e))
            })?;
            
            let ptr = unsafe { GetProcAddress(module, symbol.as_ptr()) };
            if ptr.is_null() {
                return Err(RasError::EncodingError(
                    format!("Failed to resolve symbol {}", name)
                ));
            }
            
            self.function_pointers.insert(name.to_string(), ptr as u64);
            Ok(())
        }
        
        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::EncodingError(
                "Runtime function resolution not supported on this platform".to_string()
            ))
        }
    }

    /// Compile MIR module directly to binary (for runtime compilation)
    ///
    /// This method reuses code from lamina's mir_codegen but generates binary
    /// instead of assembly text. It's used for runtime compilation (JIT).
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        let (code, _) = self.compile_mir_to_binary_function(module, None)?;
        Ok(code)
    }

    /// Compile a specific function from MIR module to binary
    /// If function_name is None, compiles all functions
    /// Returns (binary code, function_offsets map)
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary_function(
        &mut self,
        module: &lamina_mir::Module,
        function_name: Option<&str>,
    ) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
        // Store module reference for checking internal vs external calls
        self.current_module = Some(module);
        // Reuse register allocation and ABI from mir_codegen
        match self.target_arch {
            TargetArchitecture::X86_64 => {
                crate::assembler::x86_64::compile_mir_x86_64_function(self, module, function_name)
            }
            TargetArchitecture::Aarch64 => {
                crate::assembler::aarch64::compile_mir_aarch64_function(self, module, function_name)
            }
            _ => Err(RasError::UnsupportedTarget(format!(
                "MIR compilation not supported for architecture: {:?}",
                self.target_arch
            ))),
        }
    }
}





