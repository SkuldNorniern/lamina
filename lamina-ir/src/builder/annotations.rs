//! Function and module annotations for IR builder.
//!
//! This module provides methods for adding annotations to functions and modules.
//! Annotations provide metadata that affects optimization, linking, and code generation.

#[cfg(feature = "nightly")]
use crate::module::ModuleAnnotation;
use crate::{builder::IRBuilder, function::FunctionAnnotation};

impl<'a> IRBuilder<'a> {
    /// Adds a function annotation to the current function.
    ///
    /// Panics when the annotation has no text form, is already applied, or conflicts with
    /// an annotation already applied.
    pub fn annotate(&mut self, annotation: FunctionAnnotation) -> &mut Self {
        assert!(
            has_text_form(&annotation),
            "{annotation} has no text form in Lamina IR; use one of the annotations the text IR lists"
        );
        let function = self.open_function();
        assert!(
            !function.annotations.contains(&annotation),
            "{annotation} is applied twice to '@{}'",
            function.name
        );
        if let Some(other) = function
            .annotations
            .iter()
            .find(|other| annotations_conflict(other, &annotation))
        {
            panic!(
                "{annotation} and {other} cannot be used together on '@{}'",
                function.name
            );
        }
        function.annotations.push(annotation);
        self
    }

    /// Adds the `@inline` annotation.
    pub fn inline(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Inline)
    }

    /// Adds the `@export` annotation.
    pub fn export(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Export)
    }

    /// Adds the `@extern` annotation.
    pub fn external(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Extern)
    }

    /// Adds the `@noreturn` annotation.
    pub fn no_return(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::NoReturn)
    }

    /// Adds the `@cold` annotation.
    pub fn cold(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Cold)
    }

    /// Adds the `@hot` annotation.
    pub fn hot(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hot)
    }

    /// Adds the `@pure` annotation.
    pub fn pure(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Pure)
    }

    /// Adds the `@internal` annotation.
    pub fn internal(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Internal)
    }

    /// Adds the `@private` annotation.
    pub fn private(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Private)
    }

    /// Adds the `@hidden` annotation.
    pub fn hidden(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Hidden)
    }

    /// Adds the `@protected` annotation.
    pub fn protected(&mut self) -> &mut Self {
        self.annotate(FunctionAnnotation::Protected)
    }

    /// Adds the `@section("name")` annotation.
    pub fn section(&mut self, section: &str) -> &mut Self {
        self.annotate(FunctionAnnotation::Section(section.to_string()))
    }

    /// Adds the `@align(bytes)` annotation. `alignment` must be a power of two.
    pub fn align(&mut self, alignment: u32) -> &mut Self {
        self.annotate(FunctionAnnotation::Align(alignment))
    }

    #[cfg(feature = "nightly")]
    pub fn annotate_module(&mut self, annotation: ModuleAnnotation) -> &mut Self {
        self.module.annotations.push(annotation);
        self
    }

    #[cfg(feature = "nightly")]
    pub fn pic(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentCode)
    }

    #[cfg(feature = "nightly")]
    pub fn pie(&mut self) -> &mut Self {
        self.annotate_module(ModuleAnnotation::PositionIndependentExecutable)
    }
}

fn has_text_form(annotation: &FunctionAnnotation) -> bool {
    match annotation {
        FunctionAnnotation::Section(name) => !name.is_empty() && !name.contains(['\\', '"']),
        FunctionAnnotation::Align(bytes) => bytes.is_power_of_two(),
        FunctionAnnotation::Kernel
        | FunctionAnnotation::Inline
        | FunctionAnnotation::Export
        | FunctionAnnotation::Extern
        | FunctionAnnotation::NoReturn
        | FunctionAnnotation::NoInline
        | FunctionAnnotation::Cold
        | FunctionAnnotation::Hot
        | FunctionAnnotation::Pure
        | FunctionAnnotation::Internal
        | FunctionAnnotation::Private
        | FunctionAnnotation::Hidden
        | FunctionAnnotation::Protected
        | FunctionAnnotation::Weak
        | FunctionAnnotation::Naked
        | FunctionAnnotation::NoRedZone => true,
        _ => false,
    }
}

fn annotations_conflict(a: &FunctionAnnotation, b: &FunctionAnnotation) -> bool {
    use FunctionAnnotation::{Export, Extern, Hot, Naked, NoInline};
    let visibility = |x: &FunctionAnnotation| {
        matches!(
            x,
            FunctionAnnotation::Internal
                | FunctionAnnotation::Private
                | FunctionAnnotation::Hidden
                | FunctionAnnotation::Protected
        )
    };
    let local = |x: &FunctionAnnotation| {
        matches!(
            x,
            FunctionAnnotation::Internal | FunctionAnnotation::Private
        )
    };
    let pair = |x: &FunctionAnnotation,
                y: &FunctionAnnotation,
                p: &FunctionAnnotation,
                q: &FunctionAnnotation| { (x == p && y == q) || (x == q && y == p) };
    (visibility(a) && visibility(b))
        || pair(a, b, &FunctionAnnotation::Cold, &Hot)
        || matches!(
            (a, b),
            (
                FunctionAnnotation::Section(_),
                FunctionAnnotation::Section(_)
            ) | (FunctionAnnotation::Align(_), FunctionAnnotation::Align(_))
        )
        || (local(a) && *b == Export)
        || (*a == Export && local(b))
        || pair(a, b, &FunctionAnnotation::Inline, &NoInline)
        || pair(a, b, &Extern, &Export)
        || pair(a, b, &Naked, &FunctionAnnotation::Inline)
        || pair(a, b, &Naked, &FunctionAnnotation::Pure)
}
