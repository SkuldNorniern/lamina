//! Types the IR and the MIR both name, defined once so the two layers agree.
//!
//! Each of these was once declared in `lamina-ir` and again in `lamina-mir`, which made
//! `lamina::TlsModel` and `lamina::mir::TlsModel` distinct types with one name, bridged
//! by hand at the lowering boundary.

use std::{fmt, num::NonZeroU32};

/// How a thread-local symbol's address is computed.
///
/// Only the two models that need no runtime support are here. General-dynamic and
/// local-dynamic both call `__tls_get_addr`, which a freestanding target does not have,
/// so the parser names them and refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TlsModel {
    /// The offset from the thread pointer is a link-time constant. Only valid in an
    /// executable that defines the symbol itself.
    LocalExec,
    /// The offset from the thread pointer is loaded from the GOT, so the symbol may be
    /// defined by a shared object present at program start.
    InitialExec,
}

impl TlsModel {
    pub const fn as_str(&self) -> &'static str {
        match self {
            TlsModel::LocalExec => "local_exec",
            TlsModel::InitialExec => "initial_exec",
        }
    }
}

impl fmt::Display for TlsModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Ordering guarantees an atomic operation carries, relative to other memory
/// operations in the same thread and across threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrdering {
    /// No synchronization or ordering constraints.
    Relaxed,
    /// Prevents reordering of subsequent loads.
    Acquire,
    /// Prevents reordering of previous stores.
    Release,
    /// Both acquire and release semantics.
    AcqRel,
    /// Sequentially consistent: the strongest guarantee.
    SeqCst,
}

impl fmt::Display for MemoryOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MemoryOrdering::Relaxed => "relaxed",
            MemoryOrdering::Acquire => "acquire",
            MemoryOrdering::Release => "release",
            MemoryOrdering::AcqRel => "acqrel",
            MemoryOrdering::SeqCst => "seqcst",
        };
        write!(f, "{s}")
    }
}

/// A position in the `.lamina` source an instruction was parsed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceLoc {
    pub line: NonZeroU32,
    pub column: u32,
}

impl SourceLoc {
    /// A position, or `None` when `line` is zero: lines are numbered from one.
    pub fn new(line: u32, column: u32) -> Option<Self> {
        NonZeroU32::new(line).map(|line| Self { line, column })
    }

    pub fn line(self) -> u32 {
        self.line.get()
    }
}

impl fmt::Display for SourceLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_loc_rejects_line_zero() {
        assert!(SourceLoc::new(0, 4).is_none());
        let loc = SourceLoc::new(7, 4).expect("line 7 is a line");
        assert_eq!(loc.line(), 7);
        assert_eq!(loc.to_string(), "7:4");
    }

    #[test]
    fn tls_and_ordering_names_round_trip_to_text() {
        assert_eq!(TlsModel::LocalExec.as_str(), "local_exec");
        assert_eq!(TlsModel::InitialExec.to_string(), "initial_exec");
        assert_eq!(MemoryOrdering::AcqRel.to_string(), "acqrel");
        assert_eq!(MemoryOrdering::SeqCst.to_string(), "seqcst");
    }
}
