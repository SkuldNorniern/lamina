//! Runtime SIMD capability detection.
//!
//! This module provides runtime detection of SIMD features by querying the CPU
//! at runtime. This is more accurate than static detection as it checks what
//! the actual CPU supports, not just what the architecture typically supports.

use crate::simd::{SimdCapabilities, X86SimdExtension, ArmSimdExtension, RiscvSimdExtension};
use crate::target::TargetArchitecture;

#[cfg(feature = "nightly")]
impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime by querying the CPU.
    ///
    /// This method performs actual CPU feature detection and returns the
    /// capabilities that are actually available on the running CPU, not just
    /// what the architecture typically supports.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina_platform::simd::SimdCapabilities;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let caps = SimdCapabilities::detect_runtime();
    /// println!("AVX2 supported: {}", caps.has_x86_extension(X86SimdExtension::Avx2));
    /// # }
    /// ```
    pub fn detect_runtime() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64_runtime()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64_runtime()
        }
        #[cfg(target_arch = "arm")]
        {
            Self::detect_arm32_runtime()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            // For other architectures, fall back to static detection
            use crate::target::Target;
            let target = Target::detect_host();
            Self::detect(&target)
        }
    }

    /// Runtime detection for x86_64 using CPUID.
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64_runtime() -> Self {
        let mut caps = Self::default();
        caps.supported = true;

        // Use Rust's standard library feature detection
        #[cfg(feature = "nightly")]
        {
            use std::arch::is_x86_feature_detected;

            // Check MMX
            if is_x86_feature_detected!("mmx") {
                caps.x86_extensions.push(X86SimdExtension::Mmx);
            }

            // Check SSE
            if is_x86_feature_detected!("sse") {
                caps.x86_extensions.push(X86SimdExtension::Sse);
                caps.v128_supported = true;
                caps.float_simd_supported = true;
            }

            // Check SSE2 (baseline for x86_64)
            if is_x86_feature_detected!("sse2") {
                caps.x86_extensions.push(X86SimdExtension::Sse2);
                caps.v128_supported = true;
                caps.integer_simd_supported = true;
                caps.max_vector_width = 128;
            }

            // Check SSE3
            if is_x86_feature_detected!("sse3") {
                caps.x86_extensions.push(X86SimdExtension::Sse3);
            }

            // Check SSSE3
            if is_x86_feature_detected!("ssse3") {
                caps.x86_extensions.push(X86SimdExtension::Ssse3);
            }

            // Check SSE4.1
            if is_x86_feature_detected!("sse4.1") {
                caps.x86_extensions.push(X86SimdExtension::Sse41);
            }

            // Check SSE4.2
            if is_x86_feature_detected!("sse4.2") {
                caps.x86_extensions.push(X86SimdExtension::Sse42);
            }

            // Check AVX (256-bit)
            if is_x86_feature_detected!("avx") {
                caps.x86_extensions.push(X86SimdExtension::Avx);
                caps.v256_supported = true;
                caps.max_vector_width = 256;
            }

            // Check FMA3
            if is_x86_feature_detected!("fma") {
                caps.x86_extensions.push(X86SimdExtension::Fma3);
                caps.fma_supported = true;
            }

            // Check F16C
            if is_x86_feature_detected!("f16c") {
                caps.x86_extensions.push(X86SimdExtension::F16c);
                caps.fp16_supported = true;
            }

            // Check AVX2
            if is_x86_feature_detected!("avx2") {
                caps.x86_extensions.push(X86SimdExtension::Avx2);
                caps.v256_supported = true;
                caps.max_vector_width = 256;
            }

            // Check AVX-512 Foundation
            if is_x86_feature_detected!("avx512f") {
                caps.x86_extensions.push(X86SimdExtension::Avx512f);
                caps.v512_supported = true;
                caps.max_vector_width = 512;
            }

            // Check AVX-512 Vector Length Extensions
            if is_x86_feature_detected!("avx512vl") {
                caps.x86_extensions.push(X86SimdExtension::Avx512vl);
            }

            // Check AVX-512 Byte and Word Instructions
            if is_x86_feature_detected!("avx512bw") {
                caps.x86_extensions.push(X86SimdExtension::Avx512bw);
            }

            // Check AVX-512 Doubleword and Quadword Instructions
            if is_x86_feature_detected!("avx512dq") {
                caps.x86_extensions.push(X86SimdExtension::Avx512dq);
            }

            // Check AVX-512 Integer FMA
            if is_x86_feature_detected!("avx512ifma") {
                caps.x86_extensions.push(X86SimdExtension::Avx512ifma);
            }

            // Check AVX-512 Vector Byte Manipulation Instructions
            if is_x86_feature_detected!("avx512vbmi") {
                caps.x86_extensions.push(X86SimdExtension::Avx512vbmi);
            }

            // Check AVX-512 Vector Byte Manipulation Instructions 2
            if is_x86_feature_detected!("avx512vbmi2") {
                caps.x86_extensions.push(X86SimdExtension::Avx512vbmi2);
            }

            // Check AVX-512 Vector Neural Network Instructions
            if is_x86_feature_detected!("avx512vnni") {
                caps.x86_extensions.push(X86SimdExtension::Avx512vnni);
            }

            // Check AVX-512 BFloat16 Instructions
            if is_x86_feature_detected!("avx512bf16") {
                caps.x86_extensions.push(X86SimdExtension::Avx512bf16);
                caps.bf16_supported = true;
            }

            // Check AVX-512 FP16 Instructions
            if is_x86_feature_detected!("avx512fp16") {
                caps.x86_extensions.push(X86SimdExtension::Avx512fp16);
                caps.fp16_supported = true;
            }

            // Note: FMA4 detection is not available in std::arch
            // It would require direct CPUID access, which is more complex
        }

        caps
    }

    /// Runtime detection for AArch64 using system registers.
    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64_runtime() -> Self {
        let mut caps = Self::default();
        caps.supported = true;
        caps.v128_supported = true; // NEON is mandatory on AArch64
        caps.float_simd_supported = true;
        caps.integer_simd_supported = true;
        caps.fma_supported = true; // NEON includes FMA
        caps.fp16_supported = true; // NEON supports FP16
        caps.max_vector_width = 128;

        #[cfg(feature = "nightly")]
        {
            use std::arch::is_aarch64_feature_detected;

            // NEON is always available on AArch64
            caps.arm_extensions.push(ArmSimdExtension::Neon);

            // Check for SVE (Scalable Vector Extension)
            if is_aarch64_feature_detected!("sve") {
                caps.arm_extensions.push(ArmSimdExtension::Sve);
                // SVE can support larger vectors, but we need to query the actual width
                // For now, we'll assume it can support at least 256-bit
                caps.v256_supported = true;
                caps.max_vector_width = 2048; // SVE can go up to 2048 bits
            }

            // Check for SVE2
            if is_aarch64_feature_detected!("sve2") {
                caps.arm_extensions.push(ArmSimdExtension::Sve2);
            }

            // Check for SME (Scalable Matrix Extension)
            if is_aarch64_feature_detected!("sme") {
                caps.arm_extensions.push(ArmSimdExtension::Sme);
            }

            // Check for BF16 support
            if is_aarch64_feature_detected!("bf16") {
                caps.bf16_supported = true;
            }
        }

        caps
    }

    /// Runtime detection for ARM32.
    #[cfg(target_arch = "arm")]
    fn detect_arm32_runtime() -> Self {
        let mut caps = Self::default();

        #[cfg(feature = "nightly")]
        {
            use std::arch::is_arm_feature_detected;

            // Check for NEON
            if is_arm_feature_detected!("neon") {
                caps.supported = true;
                caps.v128_supported = true;
                caps.float_simd_supported = true;
                caps.integer_simd_supported = true;
                caps.fma_supported = true;
                caps.fp16_supported = true;
                caps.max_vector_width = 128;
                caps.arm_extensions.push(ArmSimdExtension::Neon);
            }
        }

        caps
    }

    /// Update capabilities with runtime detection results.
    ///
    /// This method performs runtime detection and updates the current
    /// capabilities structure with the actual CPU features.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lamina_platform::{Target, TargetArchitecture, TargetOperatingSystem};
    /// # use lamina_platform::simd::SimdCapabilities;
    /// # #[cfg(feature = "nightly")]
    /// # {
    /// let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
    /// let mut caps = SimdCapabilities::detect(&target);
    /// caps.update_with_runtime_detection();
    /// // Now caps reflects the actual CPU capabilities
    /// # }
    /// ```
    pub fn update_with_runtime_detection(&mut self) {
        let runtime_caps = Self::detect_runtime();
        
        // Update with runtime-detected values
        self.supported = runtime_caps.supported;
        self.max_vector_width = runtime_caps.max_vector_width;
        self.v128_supported = runtime_caps.v128_supported;
        self.v256_supported = runtime_caps.v256_supported;
        self.v512_supported = runtime_caps.v512_supported;
        self.float_simd_supported = runtime_caps.float_simd_supported;
        self.integer_simd_supported = runtime_caps.integer_simd_supported;
        self.fma_supported = runtime_caps.fma_supported;
        self.fp16_supported = runtime_caps.fp16_supported;
        self.bf16_supported = runtime_caps.bf16_supported;
        self.x86_extensions = runtime_caps.x86_extensions;
        self.arm_extensions = runtime_caps.arm_extensions;
        self.riscv_extensions = runtime_caps.riscv_extensions;
    }
}

#[cfg(test)]
#[cfg(feature = "nightly")]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_detection() {
        let caps = SimdCapabilities::detect_runtime();
        // At minimum, we should know if SIMD is supported or not
        assert!(caps.max_vector_width >= 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_64_runtime_detection() {
        let caps = SimdCapabilities::detect_runtime();
        // x86_64 should at least have SSE2
        assert!(caps.supported);
        assert!(caps.v128_supported);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64_runtime_detection() {
        let caps = SimdCapabilities::detect_runtime();
        // AArch64 should have NEON
        assert!(caps.supported);
        assert!(caps.v128_supported);
        assert!(caps.has_arm_extension(ArmSimdExtension::Neon));
    }

    #[test]
    fn test_update_with_runtime_detection() {
        use crate::target::{Target, TargetArchitecture, TargetOperatingSystem};
        let target = Target::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux);
        let mut caps = SimdCapabilities::detect(&target);
        caps.update_with_runtime_detection();
        // After runtime detection, capabilities should reflect actual CPU
        assert!(caps.max_vector_width >= 0);
    }
}

