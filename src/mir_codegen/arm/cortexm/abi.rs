//! The Cortex-M calling convention: AAPCS in its base standard.
//!
//! Argument placement, aggregate returns and the per-module signature table are shared
//! with the armv7 backend, which already implements the base standard for the variadic
//! and runtime-helper cases. The only part that differs is that there is no
//! co-processor bank here at all, so every call is classified as base standard and a
//! floating point argument never arises: the backend refuses floats before planning.

use lamina_codegen::abi::{Abi, common_call_stub};
use lamina_platform::TargetOperatingSystem;

pub(crate) struct CortexMAbi {
    target_os: TargetOperatingSystem,
}

impl CortexMAbi {
    pub(crate) fn new(target_os: TargetOperatingSystem) -> Self {
        Self { target_os }
    }

    pub(crate) fn mangle(&self, name: &str) -> String {
        self.mangle_function_name(name)
    }

    pub(crate) fn call_target(&self, name: &str) -> String {
        self.call_stub(name).unwrap_or_else(|| self.mangle(name))
    }
}

impl Abi for CortexMAbi {
    fn target_os(&self) -> TargetOperatingSystem {
        self.target_os
    }

    fn mangle_function_name(&self, name: &str) -> String {
        name.to_string()
    }

    fn call_stub(&self, name: &str) -> Option<String> {
        common_call_stub(name, self.target_os)
    }
}
