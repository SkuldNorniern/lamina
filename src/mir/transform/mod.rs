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

pub enum TransformLevel {
    Deprecated,
    Stable,
    Experimental
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
}