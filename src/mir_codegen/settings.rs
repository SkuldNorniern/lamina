//! Cross-backend MIR codegen settings (register allocation policy, basic debug emission).

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

/// Tunables for `generate_mir_to_target_with_settings`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirCodegenSettings {
    pub regalloc: RegallocStrategy,
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
            regalloc: RegallocStrategy::Incremental,
            emit_asm_debug_lines: false,
            debug_file_tag: "lamina".to_string(),
            #[cfg(feature = "nightly")]
            simd: None,
        }
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
