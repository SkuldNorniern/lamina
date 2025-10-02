/// Each transform operates on the MIR representation to optimize or
/// prepare code for code generation. Transforms are composable and
/// can be arranged in pipelines.
pub trait Transform: Default {
    /// Unique name for this transform
    fn name(&self) -> &'static str;

    /// Description of what this transform does
    fn description(&self) -> &'static str;

    /// Category of this transform
    fn category(&self) -> TransformCategory;

    /// Level of this transform
    fn level(&self) -> TransformLevel;

    // Future: fn apply
    // Future: fn should_run(
}

/// Stability level of a transform
///
/// This indicates the maturity and reliability of the transform,
/// allowing users to choose appropriate optimization levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransformLevel {
    /// Transform is deprecated and will be removed
    Deprecated,
    /// Transform is experimental and may change or have bugs
    Experimental,
    /// Transform is well-tested and production-ready
    Stable,
}

impl TransformLevel {
    /// Check if this level should be included at the given optimization level
    ///
    /// - `-O0`: No transforms
    /// - `-O1`: `Stable` transforms  
    /// - `-O2`: `Stable` + `Experimental` transforms
    /// - `-O3`: All transforms (including `Deprecated` for compatibility)
    pub fn is_enabled_at_opt_level(self, opt_level: u8) -> bool {
        match (self, opt_level) {
            (TransformLevel::Deprecated, 3..) => true, // Only at -O3+
            (TransformLevel::Experimental, 2..) => true, // At -O2+
            (TransformLevel::Stable, _) => true,       // Always enabled
            _ => false,
        }
    }
}

/// Categories of transformations
///
/// These categorize transforms by their primary purpose, helping with
/// organizing optimization passes and understanding what each does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformCategory {
    /// Removes code that doesn't affect program output
    DeadCodeElimination,
    /// Expands function calls inline for performance
    Inlining,
    /// Evaluates constant expressions at compile time
    ConstantFolding,
    /// Replaces variable copies with their source values
    CopyPropagation,
    /// Chooses optimal machine instructions
    InstructionSelection,
    /// Optimizes control flow patterns
    ControlFlowOptimization,
    /// Optimizes arithmetic operations
    ArithmeticOptimization,
    /// Optimizes memory access patterns
    MemoryOptimization,
}

// Example placeholder transform implementation
// This shows the pattern for implementing transforms
#[derive(Default)]
#[allow(dead_code)]
struct ExampleTransform;

impl Transform for ExampleTransform {
    fn name(&self) -> &'static str {
        "example_transform"
    }

    fn description(&self) -> &'static str {
        "Example placeholder transform"
    }

    fn category(&self) -> TransformCategory {
        TransformCategory::ConstantFolding
    }

    fn level(&self) -> TransformLevel {
        TransformLevel::Experimental
    }
}
