//! ras - Raw Assembler
//!
//! ras is a cross-platform assembler that converts assembly text to object files.
//! It can be used as a standalone tool (like `as` or `gas`) or as a library.
//!
//! # Example
//!
//! ```rust
//! use ras::RasAssembler;
//! use lamina::target::{TargetArchitecture, TargetOperatingSystem};
//!
//! let mut assembler = RasAssembler::new(
//!     TargetArchitecture::X86_64,
//!     TargetOperatingSystem::Linux,
//! )?;
//!
//! let asm_text = r#"
//!     .text
//!     .global _start
//! _start:
//!     movq $42, %rax
//!     ret
//! "#;
//!
//! assembler.assemble_text_to_object(asm_text, "output.o")?;
//! # Ok::<(), ras::RasError>(())
//! ```

pub mod assembler;
pub mod encoder;
pub mod error;
pub mod object;
pub mod parser;
pub mod runtime;

pub use assembler::RasAssembler;
pub use error::RasError;
pub use runtime::{ExecutableMemory, RasRuntime};

use lamina::target::{TargetArchitecture, TargetOperatingSystem};

/// Main assembler interface
pub struct Ras {
    assembler: RasAssembler,
}

impl Ras {
    /// Create a new ras assembler instance
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        Ok(Self {
            assembler: RasAssembler::new(target_arch, target_os)?,
        })
    }

    /// Assemble assembly text to object file
    pub fn assemble(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        self.assembler.assemble_text_to_object(asm_text, output_path)
    }

    /// Assemble from file to file
    pub fn assemble_file(
        &mut self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        let asm_text = std::fs::read_to_string(input_path)
            .map_err(|e| RasError::IoError(format!("Failed to read input file: {}", e)))?;
        self.assemble(&asm_text, output_path)
    }

    /// Compile MIR module to binary (for runtime compilation)
    ///
    /// This generates binary machine code directly from MIR, bypassing
    /// assembly text generation. Used for runtime compilation (JIT).
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "mir")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina::mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        self.assembler.compile_mir_to_binary(module)
    }
}

