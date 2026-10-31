//! The `lamina` binary's own surface: argument parsing and the compile driver.
//!
//! Public only because the binary is a separate crate from the library. Nothing here
//! is part of the library's API and all of it may change without notice.

pub mod compile;
pub mod options;

pub(crate) mod host_cpu;
pub(crate) mod jit;
