//! Cross-backend MIR codegen settings (register allocation policy, basic debug emission).

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
    /// The compiler driver may also enable minimal ELF `.debug_line` / `.debug_info` output when
    /// the assembler backend is ras on ELF-class hosts.
    pub emit_asm_debug_lines: bool,
    /// String stored in `.file 1 "..."` when debug lines are enabled.
    pub debug_file_tag: String,
}

impl Default for MirCodegenSettings {
    fn default() -> Self {
        Self {
            regalloc: RegallocStrategy::Incremental,
            emit_asm_debug_lines: false,
            debug_file_tag: "lamina".to_string(),
        }
    }
}
