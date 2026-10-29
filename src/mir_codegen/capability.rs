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
//! let native = CapabilitySet::standard_native();
//! assert!(native.supports(&CodegenCapability::FloatingPointArithmetic));
//!
//! let wasm = CapabilitySet::wasm();
//! assert!(!wasm.supports(&CodegenCapability::SystemCalls));
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
//! | DebugInfo | ✅ | ✅ | ✅ | ❌ |
//!
//! ⚠️ = Partial support (some types/operations may not work)
//! ❌ = Not supported

use std::{collections::HashSet, fmt};

use super::{
    Codegen, arm, arm::aarch64::AArch64Codegen, arx64, i686, loongarch64, mips64,
    powerpc::powerpc64::Ppc64Codegen, riscv::RiscVCodegen, s390x, wasm::WasmCodegen,
    x86_64::X86Codegen,
};
#[cfg(feature = "nightly")]
use lamina_platform::SimdCapabilities;
use lamina_platform::TargetArchitecture;

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

    /// Aggregates passed or returned by value, classified into the platform ABI
    AggregateArguments,

    /// Calls through a function pointer
    IndirectCalls,

    /// Global register allocation (linear scan or graph coloring across blocks)
    GlobalRegisterAllocation,
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
            Self::AggregateArguments => "Aggregates passed or returned by value",
            Self::IndirectCalls => "Calls through function pointers",
            Self::GlobalRegisterAllocation => "Global register allocation",
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
    capabilities: HashSet<CodegenCapability>,
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

    /// Builder form of [`add`](Self::add), for declaring a set in one expression
    pub fn with(mut self, cap: CodegenCapability) -> Self {
        self.add(cap);
        self
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
    /// recursion, stack/memory operations, system calls, inline assembly, FFI.
    pub fn standard_native() -> Self {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
            CodegenCapability::StackAllocation,
            CodegenCapability::MemoryOperations,
            CodegenCapability::SystemCalls,
            CodegenCapability::InlineAssembly,
            CodegenCapability::ForeignFunctionInterface,
            CodegenCapability::DebugInfo,
        ]
        .into_iter()
        .collect()
    }

    /// Like [`for_architecture`] but includes [`CodegenCapability::SimdOperations`] when
    /// `simd.supported` is true, overriding the static per-arch default.
    #[cfg(feature = "nightly")]
    pub fn for_architecture_with_simd(
        arch: TargetArchitecture,
        simd: Option<&SimdCapabilities>,
    ) -> Self {
        let mut caps = Self::for_architecture(arch);
        if let Some(s) = simd {
            if s.supported {
                caps.add(CodegenCapability::SimdOperations);
            } else {
                caps.remove(&CodegenCapability::SimdOperations);
            }
        }
        caps
    }

    /// Capability set a backend declares for a target architecture.
    ///
    /// The match has no wildcard arm, so adding a `TargetArchitecture` fails to compile
    /// until it has an explicit declaration here. Backends with a `Codegen` impl declare
    /// their set in `Codegen::capabilities`; the rest declare it in their module.
    pub fn for_architecture(arch: TargetArchitecture) -> Self {
        match arch {
            TargetArchitecture::X86_64 => <X86Codegen<'static> as Codegen>::capabilities(),
            TargetArchitecture::Aarch64 => <AArch64Codegen<'static> as Codegen>::capabilities(),
            TargetArchitecture::Arx64 => arx64::capabilities(),
            TargetArchitecture::LoongArch64 => loongarch64::capabilities(),
            TargetArchitecture::S390x => s390x::capabilities(),
            TargetArchitecture::Mips64 | TargetArchitecture::Mips64El => mips64::capabilities(),
            TargetArchitecture::PowerPC64 | TargetArchitecture::PowerPC64Be => {
                <Ppc64Codegen<'static> as Codegen>::capabilities()
            }
            TargetArchitecture::Wasm32 | TargetArchitecture::Wasm64 => {
                <WasmCodegen<'static> as Codegen>::capabilities()
            }
            TargetArchitecture::Riscv32 => <RiscVCodegen<'static> as Codegen>::capabilities()
                .with(CodegenCapability::AggregateArguments)
                .with(CodegenCapability::GlobalRegisterAllocation),
            TargetArchitecture::Riscv64 => <RiscVCodegen<'static> as Codegen>::capabilities()
                .with(CodegenCapability::AggregateArguments)
                .with(CodegenCapability::IndirectCalls)
                .with(CodegenCapability::GlobalRegisterAllocation),
            #[cfg(feature = "nightly")]
            TargetArchitecture::Riscv128 => <RiscVCodegen<'static> as Codegen>::capabilities()
                .with(CodegenCapability::GlobalRegisterAllocation),
            TargetArchitecture::Nvptx64 => Self::gpu(),
            TargetArchitecture::Amdgcn => Self::gpu()
                .with(CodegenCapability::IntegerArithmetic)
                .with(CodegenCapability::MemoryOperations),
            TargetArchitecture::I686 => i686::capabilities(),
            TargetArchitecture::Armv7 => arm::armv7::capabilities(),
            TargetArchitecture::Thumbv6m | TargetArchitecture::Thumbv7m => {
                arm::cortexm::capabilities()
            }
            TargetArchitecture::Arm32 | TargetArchitecture::Unknown => Self::new(),
            #[cfg(feature = "nightly")]
            TargetArchitecture::Lisa => Self::new(),
        }
    }

    pub fn gpu() -> Self {
        [CodegenCapability::ControlFlow, CodegenCapability::Threading]
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
    /// recursion, memory operations. Excludes: heap allocation,
    /// system calls, inline assembly (WASM is sandboxed).
    pub fn wasm() -> Self {
        [
            CodegenCapability::IntegerArithmetic,
            CodegenCapability::FloatingPointArithmetic,
            CodegenCapability::ControlFlow,
            CodegenCapability::FunctionCalls,
            CodegenCapability::Recursion,
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

        assert!(caps.supports(&CodegenCapability::IntegerArithmetic));
        assert!(!caps.supports(&CodegenCapability::HeapAllocation));
    }

    #[test]
    fn gpu_capabilities_exclude_host_runtime() {
        for arch in [TargetArchitecture::Nvptx64, TargetArchitecture::Amdgcn] {
            let caps = CapabilitySet::for_architecture(arch);
            for cap in [
                CodegenCapability::HeapAllocation,
                CodegenCapability::SystemCalls,
                CodegenCapability::Recursion,
            ] {
                assert!(!caps.supports(&cap));
            }
            assert!(caps.supports(&CodegenCapability::Threading));
        }
    }

    #[test]
    fn test_core_capabilities() {
        let caps = CapabilitySet::core();
        assert!(caps.supports(&CodegenCapability::IntegerArithmetic));
        assert!(caps.supports(&CodegenCapability::ControlFlow));
        assert!(caps.supports(&CodegenCapability::FunctionCalls));
    }
}
