//! SIMD (Single Instruction, Multiple Data) capabilities detection.
//!
//! This module provides comprehensive SIMD capability detection for various
//! instruction set architectures, including x86_64 (SSE, AVX, AVX-512),
//! AArch64 (NEON, SVE), RISC-V (RVV), and WebAssembly SIMD.

use crate::target::{Target, TargetArchitecture};

/// SIMD instruction set extensions for x86_64.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86SimdExtension {
    /// MMX (MultiMedia eXtensions) - 64-bit integer SIMD
    Mmx,
    /// SSE (Streaming SIMD Extensions) - 128-bit floating-point
    Sse,
    /// SSE2 - 128-bit integer and double-precision floating-point
    Sse2,
    /// SSE3 - Additional instructions for complex arithmetic
    Sse3,
    /// SSSE3 (Supplemental SSE3) - Horizontal operations and shuffling
    Ssse3,
    /// SSE4.1 - Additional multimedia instructions
    Sse41,
    /// SSE4.2 - String processing and CRC32
    Sse42,
    /// AVX (Advanced Vector Extensions) - 256-bit floating-point
    Avx,
    /// AVX2 - 256-bit integer operations and FMA
    Avx2,
    /// FMA3 - Fused multiply-add (3-operand)
    Fma3,
    /// FMA4 - Fused multiply-add (4-operand, AMD-specific)
    Fma4,
    /// F16C - Half-precision floating-point conversion
    F16c,
    /// AVX-512 Foundation - 512-bit vectors, mask registers
    Avx512f,
    /// AVX-512 Vector Length Extensions - 128/256-bit AVX-512
    Avx512vl,
    /// AVX-512 Byte and Word Instructions
    Avx512bw,
    /// AVX-512 Doubleword and Quadword Instructions
    Avx512dq,
    /// AVX-512 Integer Fused Multiply-Add
    Avx512ifma,
    /// AVX-512 Vector Byte Manipulation Instructions
    Avx512vbmi,
    /// AVX-512 Vector Byte Manipulation Instructions 2
    Avx512vbmi2,
    /// AVX-512 Vector Neural Network Instructions
    Avx512vnni,
    /// AVX-512 BFloat16 Instructions
    Avx512bf16,
    /// AVX-512 FP16 Instructions
    Avx512fp16,
}

/// SIMD instruction set extensions for AArch64/ARM.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmSimdExtension {
    /// NEON - 128-bit SIMD (baseline for AArch64)
    Neon,
    /// SVE (Scalable Vector Extension) - Variable width vectors
    Sve,
    /// SVE2 - Enhanced SVE with additional instructions
    Sve2,
    /// SME (Scalable Matrix Extension) - Matrix operations
    Sme,
}

/// SIMD instruction set extensions for RISC-V.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiscvSimdExtension {
    /// RVV (RISC-V Vector Extension) - Variable width vectors
    Rvv,
    /// RVV 0.7.1 specification
    Rvv071,
    /// RVV 1.0 specification
    Rvv10,
}

/// Comprehensive SIMD capabilities for a platform.
#[cfg(feature = "nightly")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// Whether SIMD operations are supported at all
    pub supported: bool,
    /// Maximum vector width in bits (128, 256, 512, etc.)
    pub max_vector_width: u32,
    /// Whether 128-bit vectors (v128) are supported
    pub v128_supported: bool,
    /// Whether 256-bit vectors (v256) are supported
    pub v256_supported: bool,
    /// Whether 512-bit vectors (v512) are supported
    pub v512_supported: bool,
    /// Whether floating-point SIMD operations are supported
    pub float_simd_supported: bool,
    /// Whether integer SIMD operations are supported
    pub integer_simd_supported: bool,
    /// Whether fused multiply-add (FMA) is supported
    pub fma_supported: bool,
    /// Whether half-precision (FP16) is supported
    pub fp16_supported: bool,
    /// Whether bfloat16 is supported
    pub bf16_supported: bool,
    /// x86_64 specific extensions (empty for non-x86)
    #[cfg(feature = "nightly")]
    pub x86_extensions: Vec<X86SimdExtension>,
    /// ARM specific extensions (empty for non-ARM)
    #[cfg(feature = "nightly")]
    pub arm_extensions: Vec<ArmSimdExtension>,
    /// RISC-V specific extensions (empty for non-RISC-V)
    #[cfg(feature = "nightly")]
    pub riscv_extensions: Vec<RiscvSimdExtension>,
}

#[cfg(feature = "nightly")]
impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            supported: false,
            max_vector_width: 0,
            v128_supported: false,
            v256_supported: false,
            v512_supported: false,
            float_simd_supported: false,
            integer_simd_supported: false,
            fma_supported: false,
            fp16_supported: false,
            bf16_supported: false,
            x86_extensions: Vec::new(),
            arm_extensions: Vec::new(),
            riscv_extensions: Vec::new(),
        }
    }
}

#[cfg(feature = "nightly")]
impl SimdCapabilities {
    /// Detect SIMD capabilities for the given target.
    ///
    /// This provides a comprehensive detection of all available SIMD features
    /// based on the target architecture. For runtime detection, additional
    /// CPUID checks would be needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
    /// let caps = SimdCapabilities::detect(&target);
    /// assert!(caps.supported);
    /// assert!(caps.v128_supported);
    /// assert!(caps.v256_supported);
    /// # }
    /// ```
    pub fn detect(target: &Target) -> Self {
        match target.architecture {
            TargetArchitecture::X86_64 => Self::detect_x86_64(),
            TargetArchitecture::Aarch64 => Self::detect_aarch64(),
            TargetArchitecture::Arm32 => Self::detect_arm32(),
            TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => Self::detect_wasm(),
            TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => Self::detect_riscv(),
            #[cfg(feature = "nightly")]
            TargetArchitecture::Riscv128 => Self::detect_riscv128(),
            #[cfg(feature = "nightly")]
            TargetArchitecture::Lisa => Self::detect_lisa(),
            TargetArchitecture::Unknown => Self::default(),
        }
    }

    /// Detect x86_64 SIMD capabilities.
    ///
    /// x86_64 baseline includes SSE2 (128-bit). Modern CPUs typically support
    /// AVX (256-bit) and AVX2. AVX-512 (512-bit) is available on newer CPUs
    /// but may be disabled by some vendors.
    fn detect_x86_64() -> Self {
        Self {
            supported: true,
            max_vector_width: 512, // AVX-512 capable, but may not be available
            v128_supported: true,  // SSE2 is baseline for x86_64
            v256_supported: true,  // AVX is common on modern x86_64
            v512_supported: true,  // AVX-512 exists but may need runtime check
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: true, // FMA3 is common on modern x86_64
            fp16_supported: true, // F16C and AVX-512-FP16
            bf16_supported: true, // AVX-512-BF16
            x86_extensions: vec![
                X86SimdExtension::Sse2,  // Baseline
                X86SimdExtension::Sse3,
                X86SimdExtension::Ssse3,
                X86SimdExtension::Sse41,
                X86SimdExtension::Sse42,
                X86SimdExtension::Avx,   // Common
                X86SimdExtension::Avx2,  // Common
                X86SimdExtension::Fma3,   // Common
                X86SimdExtension::F16c,  // Common
                // AVX-512 extensions - may need runtime detection
                X86SimdExtension::Avx512f,
                X86SimdExtension::Avx512vl,
                X86SimdExtension::Avx512bw,
                X86SimdExtension::Avx512dq,
            ],
            arm_extensions: Vec::new(),
            riscv_extensions: Vec::new(),
        }
    }

    /// Detect AArch64 SIMD capabilities.
    ///
    /// AArch64 baseline includes NEON (128-bit). SVE (Scalable Vector Extension)
    /// provides variable-width vectors but is optional and requires runtime detection.
    fn detect_aarch64() -> Self {
        Self {
            supported: true,
            max_vector_width: 2048, // SVE can go up to 2048 bits, but 128 is baseline
            v128_supported: true,  // NEON is baseline for AArch64
            v256_supported: false,  // NEON doesn't support 256-bit, SVE would but it's optional
            v512_supported: false,  // SVE can support this but requires runtime detection
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: true, // NEON supports FMA
            fp16_supported: true, // NEON supports FP16
            bf16_supported: false, // BF16 requires SVE2 or newer
            x86_extensions: Vec::new(),
            arm_extensions: vec![
                ArmSimdExtension::Neon, // Baseline
                // SVE and SVE2 are optional and would need runtime detection
            ],
            riscv_extensions: Vec::new(),
        }
    }

    /// Detect ARM32 SIMD capabilities.
    ///
    /// ARM32 may support NEON, but it's optional. For simplicity, we assume
    /// NEON is available on modern ARM32 systems.
    fn detect_arm32() -> Self {
        Self {
            supported: true,
            max_vector_width: 128,
            v128_supported: true,
            v256_supported: false,
            v512_supported: false,
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: true,
            fp16_supported: true,
            bf16_supported: false,
            x86_extensions: Vec::new(),
            arm_extensions: vec![ArmSimdExtension::Neon],
            riscv_extensions: Vec::new(),
        }
    }

    /// Detect WebAssembly SIMD capabilities.
    ///
    /// WASM SIMD supports 128-bit vectors with a fixed instruction set.
    fn detect_wasm() -> Self {
        Self {
            supported: true,
            max_vector_width: 128,
            v128_supported: true,
            v256_supported: false,
            v512_supported: false,
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: false, // WASM SIMD doesn't include FMA
            fp16_supported: false,
            bf16_supported: false,
            x86_extensions: Vec::new(),
            arm_extensions: Vec::new(),
            riscv_extensions: Vec::new(),
        }
    }

    /// Detect RISC-V SIMD capabilities.
    ///
    /// RISC-V Vector Extension (RVV) supports variable-width vectors.
    /// The actual width is implementation-defined and requires runtime detection.
    fn detect_riscv() -> Self {
        Self {
            supported: true,
            max_vector_width: 512, // RVV can support various widths
            v128_supported: true,
            v256_supported: true, // RVV can support this
            v512_supported: true, // RVV can support this
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: false, // RVV may support FMA but it's not guaranteed
            fp16_supported: false,
            bf16_supported: false,
            x86_extensions: Vec::new(),
            arm_extensions: Vec::new(),
            riscv_extensions: vec![RiscvSimdExtension::Rvv],
        }
    }

    /// Detect RISC-V 128-bit architecture SIMD capabilities.
    #[cfg(feature = "nightly")]
    fn detect_riscv128() -> Self {
        Self {
            supported: true,
            max_vector_width: 512,
            v128_supported: true,
            v256_supported: true,
            v512_supported: true,
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: true,
            fp16_supported: true,
            bf16_supported: true,
            x86_extensions: Vec::new(),
            arm_extensions: Vec::new(),
            riscv_extensions: vec![RiscvSimdExtension::Rvv, RiscvSimdExtension::Rvv10],
        }
    }

    /// Detect Lisa architecture SIMD capabilities.
    #[cfg(feature = "nightly")]
    fn detect_lisa() -> Self {
        Self {
            supported: true,
            max_vector_width: 512,
            v128_supported: true,
            v256_supported: true,
            v512_supported: true,
            float_simd_supported: true,
            integer_simd_supported: true,
            fma_supported: true,
            fp16_supported: true,
            bf16_supported: true,
            x86_extensions: Vec::new(),
            arm_extensions: Vec::new(),
            riscv_extensions: Vec::new(),
        }
    }

    /// Check if a specific vector width is supported.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
    /// let caps = SimdCapabilities::detect(&target);
    /// assert!(caps.supports_vector_width(128));
    /// assert!(caps.supports_vector_width(256));
    /// assert!(caps.supports_vector_width(512));
    /// # }
    /// ```
    pub fn supports_vector_width(&self, width: u32) -> bool {
        if !self.supported {
            return false;
        }
        match width {
            128 => self.v128_supported,
            256 => self.v256_supported,
            512 => self.v512_supported,
            _ => width <= self.max_vector_width,
        }
    }

    /// Check if a specific x86 SIMD extension is supported.
    #[cfg(feature = "nightly")]
    pub fn has_x86_extension(&self, ext: X86SimdExtension) -> bool {
        self.x86_extensions.contains(&ext)
    }

    /// Check if a specific ARM SIMD extension is supported.
    #[cfg(feature = "nightly")]
    pub fn has_arm_extension(&self, ext: ArmSimdExtension) -> bool {
        self.arm_extensions.contains(&ext)
    }

    /// Check if a specific RISC-V SIMD extension is supported.
    #[cfg(feature = "nightly")]
    pub fn has_riscv_extension(&self, ext: RiscvSimdExtension) -> bool {
        self.riscv_extensions.contains(&ext)
    }
}

#[cfg(feature = "nightly")]
impl Target {
    /// Get SIMD capabilities for this target.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina_platform::Target;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let target = Target::detect_host();
    /// let caps = target.simd_capabilities();
    /// println!("SIMD supported: {}", caps.supported);
    /// println!("Max vector width: {} bits", caps.max_vector_width);
    /// # }
    /// ```
    pub fn simd_capabilities(&self) -> SimdCapabilities {
        SimdCapabilities::detect(self)
    }
}

#[cfg(test)]
#[cfg(feature = "nightly")]
mod tests {
    use super::*;
    use crate::target::{Target, TargetArchitecture, TargetOperatingSystem};

    #[test]
    fn test_simd_capabilities_x86_64() {
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
        let caps = SimdCapabilities::detect(&target);
        assert!(caps.supported);
        assert!(caps.v128_supported);
        assert!(caps.v256_supported);
        assert!(caps.v512_supported);
        assert!(caps.float_simd_supported);
        assert!(caps.integer_simd_supported);
        assert!(caps.fma_supported);
        assert!(caps.fp16_supported);
        assert!(caps.supports_vector_width(128));
        assert!(caps.supports_vector_width(256));
        assert!(caps.supports_vector_width(512));
        assert!(caps.has_x86_extension(X86SimdExtension::Sse2));
        assert!(caps.has_x86_extension(X86SimdExtension::Avx));
        assert!(caps.has_x86_extension(X86SimdExtension::Avx2));
    }

    #[test]
    fn test_simd_capabilities_aarch64() {
        let target = Target::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS);
        let caps = SimdCapabilities::detect(&target);
        assert!(caps.supported);
        assert!(caps.v128_supported);
        assert!(!caps.v256_supported); // NEON is 128-bit only
        assert!(caps.float_simd_supported);
        assert!(caps.integer_simd_supported);
        assert!(caps.fma_supported);
        assert!(caps.fp16_supported);
        assert!(caps.supports_vector_width(128));
        assert!(!caps.supports_vector_width(256));
        assert!(caps.has_arm_extension(ArmSimdExtension::Neon));
    }

    #[test]
    fn test_simd_capabilities_wasm() {
        let target = Target::new(TargetArchitecture::Wasm32, TargetOperatingSystem::Unknown);
        let caps = SimdCapabilities::detect(&target);
        assert!(caps.supported);
        assert!(caps.v128_supported);
        assert!(!caps.v256_supported);
        assert!(caps.float_simd_supported);
        assert!(caps.integer_simd_supported);
        assert!(!caps.fma_supported); // WASM SIMD doesn't include FMA
        assert!(!caps.fp16_supported);
    }

    #[test]
    fn test_simd_capabilities_riscv() {
        let target = Target::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux);
        let caps = SimdCapabilities::detect(&target);
        assert!(caps.supported);
        assert!(caps.v128_supported);
        assert!(caps.v256_supported);
        assert!(caps.v512_supported);
        assert!(caps.has_riscv_extension(RiscvSimdExtension::Rvv));
    }

    #[test]
    fn test_simd_capabilities_unknown() {
        let target = Target::new(TargetArchitecture::Unknown, TargetOperatingSystem::Unknown);
        let caps = SimdCapabilities::detect(&target);
        assert!(!caps.supported);
        assert!(!caps.v128_supported);
        assert!(!caps.v256_supported);
        assert!(!caps.v512_supported);
    }
}

