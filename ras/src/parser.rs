//! Assembly text parser
//!
//! Parses assembly text into structured representation.

use crate::encoder::traits::ParsedInstruction;
use crate::error::RasError;

/// Parsed assembly representation
pub struct ParsedAssembly {
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub instructions: Vec<ParsedInstruction>,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub flags: SectionFlags,
}

#[derive(Debug, Clone)]
pub struct SectionFlags {
    pub alloc: bool,
    pub exec: bool,
    pub write: bool,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub global: bool,
    pub section: String,
}

/// Assembly parser
pub struct AssemblyParser {
    current_section: String,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    instructions: Vec<ParsedInstruction>,
}

impl Default for AssemblyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AssemblyParser {
    pub fn new() -> Self {
        Self {
            current_section: ".text".to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            instructions: Vec::new(),
        }
    }

    pub fn parse(&mut self, text: &str) -> Result<ParsedAssembly, RasError> {
        for line in text.lines() {
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            // Parse directive
            if line.starts_with('.') {
                self.parse_directive(line)?;
            }
            // Parse label
            else if line.ends_with(':') {
                let label = line.trim_end_matches(':').trim();
                self.symbols.push(Symbol {
                    name: label.to_string(),
                    global: false,
                    section: self.current_section.clone(),
                });
            }
            // Parse instruction
            else {
                self.parse_instruction(line)?;
            }
        }

        Ok(ParsedAssembly {
            sections: self.sections.clone(),
            symbols: self.symbols.clone(),
            instructions: self.instructions.clone(),
        })
    }

    fn parse_directive(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            ".text" => {
                self.current_section = ".text".to_string();
                self.sections.push(Section {
                    name: ".text".to_string(),
                    flags: SectionFlags {
                        alloc: true,
                        exec: true,
                        write: false,
                    },
                });
            }
            ".data" => {
                self.current_section = ".data".to_string();
                self.sections.push(Section {
                    name: ".data".to_string(),
                    flags: SectionFlags {
                        alloc: true,
                        exec: false,
                        write: true,
                    },
                });
            }
            ".global" | ".globl" => {
                if parts.len() < 2 {
                    return Err(RasError::ParseError(
                        ".global requires a symbol name".to_string(),
                    ));
                }
                // Mark symbol as global (will be updated when symbol is defined)
                // For now, just note it
            }
            _ => {
                // Ignore unknown directives for now
            }
        }

        Ok(())
    }

    fn parse_instruction(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let opcode = parts[0].to_string();
        let operands: Vec<String> = if parts.len() > 1 {
            parts[1..]
                .join(" ")
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        } else {
            Vec::new()
        };

        self.instructions.push(ParsedInstruction { opcode, operands });

        Ok(())
    }
}

