//! Source locations carried from the `.lamina` input down to the backends.

use crate::instruction::Instruction;
use std::fmt;
use std::num::NonZeroU32;
use std::ops::{Deref, DerefMut};

/// A position in the original `.lamina` source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceLoc {
    pub line: NonZeroU32,
    pub column: u32,
}

impl SourceLoc {
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

/// An instruction together with the `.lamina` line it came from, if any.
///
/// A synthesized instruction carries `loc: None` and must not borrow a line from
/// its neighbours.
#[derive(Debug, Clone)]
pub struct Located {
    pub instr: Instruction,
    pub loc: Option<SourceLoc>,
}

/// The position is metadata: two `Located` holding the same instruction are equal.
impl PartialEq for Located {
    fn eq(&self, other: &Self) -> bool {
        self.instr == other.instr
    }
}

impl Located {
    pub fn new(instr: Instruction) -> Self {
        Self { instr, loc: None }
    }

    pub fn at(instr: Instruction, loc: Option<SourceLoc>) -> Self {
        Self { instr, loc }
    }

    pub fn with_loc(mut self, loc: Option<SourceLoc>) -> Self {
        self.loc = loc;
        self
    }

    pub fn into_inner(self) -> Instruction {
        self.instr
    }
}

impl Deref for Located {
    type Target = Instruction;
    fn deref(&self) -> &Self::Target {
        &self.instr
    }
}

impl DerefMut for Located {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.instr
    }
}

impl From<Instruction> for Located {
    fn from(instr: Instruction) -> Self {
        Self::new(instr)
    }
}

impl AsRef<Instruction> for Located {
    fn as_ref(&self) -> &Instruction {
        &self.instr
    }
}

impl fmt::Display for Located {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.instr.fmt(f)
    }
}

impl PartialEq<Instruction> for Located {
    fn eq(&self, other: &Instruction) -> bool {
        &self.instr == other
    }
}

impl PartialEq<Located> for Instruction {
    fn eq(&self, other: &Located) -> bool {
        self == &other.instr
    }
}
