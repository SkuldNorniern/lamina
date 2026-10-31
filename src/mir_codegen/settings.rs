//! Cross-backend MIR codegen settings (register allocation policy, basic debug emission).

use lamina_platform::{
    ArchVariant, TargetArchitecture, TargetEnvironment, TargetOperatingSystem, X86CpuLevel,
    default_environment,
};
#[cfg(feature = "nightly")]
use lamina_platform::{SimdCapabilities, Target};

/// How virtual GPRs are assigned before instruction emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegallocStrategy {
    /// Per-instruction `ensure_mapping` (legacy behavior).
    #[default]
    Incremental,
    /// One global pass using [`lamina_codegen::LinearScanAllocator`].
    LinearScanGlobal,
    /// One global pass using [`lamina_codegen::GraphColorAllocator`].
    GraphColorGlobal,
}

impl RegallocStrategy {
    pub fn for_opt_level(opt_level: u8) -> Self {
        if opt_level >= 1 {
            Self::LinearScanGlobal
        } else {
            Self::Incremental
        }
    }
}

/// Tunables for `generate_mir_to_target_with_settings`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirCodegenSettings {
    pub amdgpu_target: super::amdgpu::AmdGpuTarget,
    pub regalloc: RegallocStrategy,
    /// Sub-architecture the target selected, for every family that has an axis: the
    /// RISC-V ISA, the armv7 variant, the aarch64 level, the POWER level and the x86-64
    /// CPU level all arrive here. [`ArchVariant::None`] leaves each backend on its
    /// architecture's default.
    pub arch_variant: ArchVariant,
    /// C runtime and ABI the target declared. `None` means the codegen entry point
    /// should resolve it from the architecture and operating system.
    pub environment: Option<TargetEnvironment>,
    /// When true, emit GNU assembler `.file` / `.loc` directives (requires
    /// [`crate::mir_codegen::CodegenCapability::DebugInfo`] on the target).
    pub emit_asm_debug_lines: bool,
    /// String stored in `.file 1 "..."` when debug lines are enabled.
    pub debug_file_tag: String,
    /// SIMD capabilities for the target; when `Some`, SIMD-capable backends may
    /// advertise and use SIMD operations.
    #[cfg(feature = "nightly")]
    pub simd: Option<SimdCapabilities>,
}

impl Default for MirCodegenSettings {
    fn default() -> Self {
        Self {
            amdgpu_target: Default::default(),
            regalloc: RegallocStrategy::Incremental,
            arch_variant: ArchVariant::None,
            environment: None,
            emit_asm_debug_lines: false,
            debug_file_tag: "lamina".to_string(),
            #[cfg(feature = "nightly")]
            simd: None,
        }
    }
}

impl MirCodegenSettings {
    /// Select the sub-architecture a parsed target named.
    pub fn with_arch_variant(mut self, variant: ArchVariant) -> Self {
        self.arch_variant = variant;
        self
    }

    /// Apply the sub-architecture a parsed target named, leaving an axis value the
    /// command line already chose in place.
    pub fn with_target_variant(self, variant: ArchVariant) -> Self {
        if self.arch_variant == ArchVariant::None {
            self.with_arch_variant(variant)
        } else {
            self
        }
    }

    /// Select the environment a parsed target named.
    pub fn with_environment(mut self, environment: TargetEnvironment) -> Self {
        self.environment = Some(environment);
        self
    }

    /// Highest x86-64 level the emitted code may use. Instructions above it are replaced
    /// by a documented fallback or refused with
    /// [`crate::mir_codegen::CodegenError::CpuLevelRequired`]. It is the x86 arm of the
    /// variant axis, so it is read from [`MirCodegenSettings::arch_variant`].
    pub fn x86_cpu_level(&self) -> X86CpuLevel {
        self.arch_variant.x86().unwrap_or_default()
    }

    /// The environment, or the one the architecture and operating system declare.
    pub fn environment(
        &self,
        architecture: TargetArchitecture,
        operating_system: TargetOperatingSystem,
    ) -> TargetEnvironment {
        self.environment
            .unwrap_or_else(|| default_environment(architecture, operating_system))
    }
}

#[cfg(feature = "nightly")]
impl MirCodegenSettings {
    /// Populate [`MirCodegenSettings::simd`] by detecting capabilities from `target`.
    pub fn with_simd_detection(mut self, target: &Target) -> Self {
        self.simd = Some(target.simd_capabilities());
        self
    }
}
