//! Function and module annotations for IR builder.
//!
//! This module provides methods for adding annotations to functions and modules.
//! Annotations provide metadata that affects optimization, linking, and code generation.

use super::IRBuilder;
use crate::ir::function::FunctionAnnotation;
#[cfg(feature = "nightly")]
use crate::ir::module::ModuleAnnotation;

impl<'a> IRBuilder<'a> {
    /// Adds an annotation to the current function
    pub fn annotate(&mut self, annotation: FunctionAnnotation) -> &mut Self {
        if let Some(func_name) = self.current_function
            && let Some(annotations) = self.function_annotations.get_mut(func_name)
        {
            annotations.push(annotation);
        }
        self
    }

    /// Marks the current function as inline
    pub fn inline(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Inline)
    }

    /// Marks the current function as exported
    pub fn export(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Export)
    }

    /// Marks the current function as external (imported)
    pub fn external(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Extern)
    }

    /// Marks the current function as having no return
    pub fn no_return(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::NoReturn)
    }

    /// Marks the current function as cold (rarely executed)
    pub fn cold(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Cold)
    }

    /// Marks the current function as hot (frequently executed)
    pub fn hot(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hot)
    }

    /// Marks the current function as pure (no side effects)
    pub fn pure(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Pure)
    }

    /// Marks the current function as const (compile-time evaluable)
    pub fn const_fn(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Const)
    }

    /// Marks the current function as internal (private to module)
    pub fn internal(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Internal)
    }

    /// Marks the current function as having private linkage (ELF-specific)
    pub fn private(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Private)
    }

    /// Marks the current function as having hidden visibility (ELF-specific)
    pub fn hidden(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hidden)
    }

    /// Marks the current function as having protected visibility (ELF-specific)
    pub fn protected(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Protected)
    }

    /// Marks the current function as unsafe
    pub fn unsafe_fn(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Unsafe)
    }

    /// Sets the C calling convention for the current function (system default).
    pub fn cc_c(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCc)
    }

    /// Sets the fastcall calling convention for the current function.
    pub fn cc_fast(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCfast)
    }

    /// Sets the cold calling convention for the current function.
    pub fn cc_cold(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCcold)
    }

    /// Sets the preserve_most calling convention for the current function.
    pub fn cc_preserve_most(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCpreserveMost)
    }

    /// Sets the preserve_all calling convention for the current function.
    pub fn cc_preserve_all(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCpreserveAll)
    }

    /// Sets the swift calling convention for the current function.
    pub fn cc_swift(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCswift)
    }

    /// Sets the tail calling convention for the current function.
    pub fn cc_tail(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::CCtail)
    }

    /// Sets a custom calling convention for the current function.
    pub fn calling_convention(&mut self, cc: &str) -> &mut Self {
        self.annotate(FunctionAnnotation::CallingConvention(cc.to_string()))
    }

    /// Sets the section for the current function
    pub fn section(&mut self, section: &str) -> &mut Self {
        self.annotate(FunctionAnnotation::Section(section.to_string()))
    }

    /// Sets the alignment for the current function
    pub fn align(&mut self, alignment: u32) -> &mut Self {
        self.annotate(FunctionAnnotation::Align(alignment))
    }

    /// Annotates the module with a global attribute.
    #[cfg(feature = "nightly")]
    pub fn annotate_module(&mut self, annotation: ModuleAnnotation) -> &mut Self {
        self.module.annotations.push(annotation);
        self
    }

    /// Enables position-independent code generation for this module.
    #[cfg(feature = "nightly")]
    pub fn pic(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentCode)
    }

    /// Enables position-independent executable generation for this module.
    #[cfg(feature = "nightly")]
    pub fn pie(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentExecutable)
    }

    /// Optimizes this module for execution speed.
    #[cfg(feature = "nightly")]
    pub fn optimize_for_speed(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::OptimizeForSpeed)
    }

    /// Optimizes this module for code size.
    #[cfg(feature = "nightly")]
    pub fn optimize_for_size(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::OptimizeForSize)
    }

    /// Includes debug information in the compiled output.
    #[cfg(feature = "nightly")]
    pub fn include_debug_info(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::IncludeDebugInfo)
    }

    /// Strips debug information and symbols from the compiled output.
    #[cfg(feature = "nightly")]
    pub fn strip_symbols(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::StripSymbols)
    }

    /// Specifies the target triple for this module.
    #[cfg(feature = "nightly")]
    pub fn target_triple(&mut self, triple: &str) -> &mut Self {
        self.annotate_module(ModuleAnnotation::TargetTriple(triple.to_string()))
    }
}

