//! Backend capabilities for MIR codegen
//!
//! Capabilities that different codegen backends may or may not support.
//! This lets the compiler handle unsupported features and return clear
//! error messages.
//!
//! # Capability System
//!
//! Each backend declares which capabilities it has. When codegen encounters an operation
//! that requires an unsupported capability, it returns an `UnsupportedFeature` error with
//! a clear message indicating what's missing.
//!
//! ## Checking Capabilities
//!
//! ```rust
//! use lamina::mir_codegen::capability::{CodegenCapability, CapabilitySet};
//!
//! let caps = X86Codegen::capabilities();
//! if !caps.supports(&CodegenCapability::FloatingPointArithmetic) {
//!     return Err("Floating point not supported on this backend");
//! }
//! ```
//!
//! ## Backend Capability Matrix
//!
//! | Capability | x86_64 | AArch64 | RISC-V | WASM |
//! |------------|--------|---------|--------|------|
//! | IntegerArithmetic | ✅ | ✅ | ✅ | ✅ |
//! | FloatingPointArithmetic | ✅ | ✅ | ⚠️ | ⚠️ |
//! | ControlFlow | ✅ | ✅ | ✅ | ✅ |
//! | FunctionCalls | ✅ | ✅ | ✅ | ✅ |
//! | MemoryOperations | ✅ | ✅ | ⚠️ | ⚠️ |
//! | SimdOperations | ✅ | ✅ | ❌ | ❌ |
//!
//! ⚠️ = Partial support (some types/operations may not work)
//! ❌ = Not supported

use std::fmt;

/// Capabilities that a codegen backend may support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodegenCapability {
    /// Basic integer arithmetic operations
    IntegerArithmetic,

    /// Floating-point arithmetic operations
    FloatingPointArithmetic,

    /// Control flow (branches, jumps, loops)
    ControlFlow,

    /// Function calls
    FunctionCalls,

    /// Recursive function calls
    Recursion,

    /// Print/log output (may require runtime support)
    Print,

    /// Memory allocation on the heap
    HeapAllocation,

    /// Stack allocation
    StackAllocation,

    /// Memory load/store operations
    MemoryOperations,

    /// SIMD/Vector operations
    SimdOperations,

    /// Atomic operations
    AtomicOperations,

    /// System calls
    SystemCalls,

    /// Inline assembly
    InlineAssembly,

    /// Exception handling
    ExceptionHandling,

    /// Threads/parallelism
    Threading,

    /// Garbage collection integration
    GarbageCollection,

    /// FFI/external function calls
    ForeignFunctionInterface,

    /// Debug information generation
    DebugInfo,

    /// Tail call optimization
    TailCallOptimization,
}

impl CodegenCapability {
    /// Returns a human-readable description of the capability
    pub fn description(&self) -> &'static str {
        match self {
            Self::IntegerArithmetic => "Integer arithmetic operations",
            Self::FloatingPointArithmetic => "Floating-point arithmetic operations",
            Self::ControlFlow => "Control flow (branches, jumps, loops)",
            Self::FunctionCalls => "Function calls",
            Self::Recursion => "Recursive function calls",
            Self::Print => "Print/log output",
            Self::HeapAllocation => "Heap memory allocation",
            Self::StackAllocation => "Stack allocation",
            Self::MemoryOperations => "Memory load/store operations",
            Self::SimdOperations => "SIMD/Vector operations",
            Self::AtomicOperations => "Atomic operations",
            Self::SystemCalls => "System calls",
            Self::InlineAssembly => "Inline assembly",
            Self::ExceptionHandling => "Exception handling",
            Self::Threading => "Threads/parallelism",
            Self::GarbageCollection => "Garbage collection integration",
            Self::ForeignFunctionInterface => "Foreign function interface (FFI)",
            Self::DebugInfo => "Debug information generation",
            Self::TailCallOptimization => "Tail call optimization",
        }
    }

    /// Returns whether this capability is required for basic functionality
    pub fn is_core(&self) -> bool {
        matches!(
            self,
            Self::IntegerArithmetic | Self::ControlFlow | Self::FunctionCalls
        )
    }
}

impl fmt::Display for CodegenCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A set of capabilities supported by a backend
#[derive(Debug, Clone, Default)]
pub struct CapabilitySet {
    capabilities: std::collections::HashSet<CodegenCapability>,
}

impl CapabilitySet {
    /// Create an empty capability set
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a capability set with all core capabilities
    pub fn core() -> Self {
        let mut set = Self::new();
        set.add(CodegenCapability::IntegerArithmetic);
        set.add(CodegenCapability::ControlFlow);
        set.add(CodegenCapability::FunctionCalls);
        set
    }

    /// Add a capability to the set
    pub fn add(&mut self, cap: CodegenCapability) {
        self.capabilities.insert(cap);
    }

    /// Remove a capability from the set
    pub fn remove(&mut self, cap: &CodegenCapability) {
        self.capabilities.remove(cap);
    }

    /// Check if a capability is supported
    pub fn supports(&self, cap: &CodegenCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Get all supported capabilities
    pub fn all(&self) -> impl Iterator<Item = &CodegenCapability> {
        self.capabilities.iter()
    }

    /// Get all unsupported capabilities from a list
    pub fn unsupported<'a>(&self, required: &'a [CodegenCapability]) -> Vec<&'a CodegenCapability> {
        required.iter().filter(|cap| !self.supports(cap)).collect()
    }
}

impl FromIterator<CodegenCapability> for CapabilitySet {
    fn from_iter<T: IntoIterator<Item = CodegenCapability>>(iter: T) -> Self {
        Self {
            capabilities: iter.into_iter().collect(),
        }
    }
}

/// Capability set builders for common backend configurations.
///
/// These builders reduce duplication when declaring backend capabilities.
impl CapabilitySet {
    /// Standard native backend capabilities (x86_64, AArch64, RISC-V).
    ///
    /// Includes: integer/float arithmetic, control flow, function calls,
    /// recursion, print, stack/memory operations, system calls, inline assembly, FFI.
    pub fn standard_native() -> Self {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            CodegenCapability::Print,
            CodegenCapability::StackAllocation,
            CodegenCapability::MemoryOperations,
            CodegenCapability::SystemCalls,
            CodegenCapability::InlineAssembly,
            CodegenCapability::ForeignFunctionInterface,
        ]
        .into_iter()
        .collect()
    }

    /// Extended native backend capabilities (includes SIMD).
    ///
    /// Same as `standard_native()` plus SIMD operations.
    pub fn extended_native() -> Self {
        let mut set = Self::standard_native();
        set.capabilities.insert(CodegenCapability::SimdOperations);
        set
    }

    /// WASM backend capabilities.
    ///
    /// Includes: integer/float arithmetic, control flow, function calls,
    /// recursion, print, memory operations. Excludes: heap allocation,
    /// system calls, inline assembly (WASM is sandboxed).
    pub fn wasm() -> Self {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            CodegenCapability::Print,
            CodegenCapability::MemoryOperations,
        ]
        .into_iter()
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_set() {
        let mut caps = CapabilitySet::new();
        caps.add(CodegenCapability::IntegerArithmetic);
        caps.add(CodegenCapability::Print);

        assert!(caps.supports(&CodegenCapability::IntegerArithmetic));
        assert!(caps.supports(&CodegenCapability::Print));
        assert!(!caps.supports(&CodegenCapability::HeapAllocation));
    }

    #[test]
    fn test_core_capabilities() {
        let caps = CapabilitySet::core();
        assert!(caps.supports(&CodegenCapability::IntegerArithmetic));
        assert!(caps.supports(&CodegenCapability::ControlFlow));
        assert!(caps.supports(&CodegenCapability::FunctionCalls));
    }
}
