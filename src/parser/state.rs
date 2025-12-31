//! Parser state management for Lamina IR parsing.

use crate::LaminaError;
use std::result::Result;

/// Parser state tracking position and input.
#[derive(Debug)]
pub struct ParserState<'a> {
    input: &'a str,
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ParserState<'a> {
    /// Creates a new parser state for the given input.
    pub fn new(input: &'a str) -> Self {
        ParserState {
            input,
            bytes: input.as_bytes(),
            position: 0,
        }
    }

    /// Returns the current character.
    pub fn current_char(&self) -> Option<char> {
        self.peek_char(0)
    }

    pub fn peek_char(&self, offset: usize) -> Option<char> {
        self.input[self.position..].chars().nth(offset)
    }

    pub fn peek_slice(&self, len: usize) -> Option<&'a str> {
        if self.position + len <= self.input.len() {
            Some(&self.input[self.position..self.position + len])
        } else {
            None
        }
    }

    /// Advances the parser position by one character.
    pub fn advance(&mut self) {
        if let Some(c) = self.current_char() {
            self.position += c.len_utf8();
        }
    }

    /// Advances the parser position by the specified number of characters.
    pub fn advance_by(&mut self, count: usize) {
        for _ in 0..count {
            self.advance();
        }
    }

    /// Checks if the parser has reached end of input.
    pub fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Skips whitespace and comments ('#' to end of line).
    pub fn skip_whitespace_and_comments(&mut self) {
        while !self.is_eof() {
            let byte = self.bytes[self.position];
            if byte.is_ascii_whitespace() {
                self.advance();
            } else if byte == b'#' {
                while !self.is_eof() && self.bytes[self.position] != b'\n' {
                    self.advance();
                }
                if !self.is_eof() {
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    /// Expects and consumes a specific character.
    pub fn expect_char(&mut self, expected: char) -> Result<(), LaminaError> {
        self.skip_whitespace_and_comments();
        if self.current_char() == Some(expected) {
            self.advance();
            Ok(())
        } else {
            let found = self.current_char()
                .map(|c| format!("'{}'", c))
                .unwrap_or_else(|| "end of input".to_string());
            
            let hint = match expected {
                '(' => "Did you forget an opening parenthesis?",
                ')' => "Did you forget a closing parenthesis?",
                '{' => "Did you forget an opening brace?",
                '}' => "Did you forget a closing brace?",
                '[' => "Did you forget an opening bracket?",
                ']' => "Did you forget a closing bracket?",
                ':' => "Did you forget a colon after an identifier?",
                ',' => "Did you forget a comma to separate items?",
                '=' => "Did you forget an equals sign?",
                _ => "",
            };
            
            let msg = if hint.is_empty() {
                format!("Expected character '{}', but found {}", expected, found)
            } else {
                format!("Expected character '{}', but found {}\n  Hint: {}", expected, found, hint)
            };
            
            Err(self.error(msg))
        }
    }

    /// Consumes a keyword if present.
    pub fn consume_keyword(&mut self, keyword: &str) -> Result<(), LaminaError> {
        self.skip_whitespace_and_comments();
        if self.input[self.position..].starts_with(keyword) {
            let next_char_pos = self.position + keyword.len();
            if next_char_pos >= self.input.len()
                || !self.input.as_bytes()[next_char_pos].is_ascii_alphanumeric()
            {
                self.position += keyword.len();
                Ok(())
            } else {
                let found = self.peek_slice(keyword.len() + 10).unwrap_or("");
                Err(self.error(format!(
                    "Expected keyword '{}', but found longer identifier '{}'\n  Hint: Keywords must be followed by whitespace or punctuation, not alphanumeric characters",
                    keyword, found
                )))
            }
        } else {
            let found = self.peek_slice(keyword.len().max(20)).unwrap_or("");
            let suggestion = if found.starts_with(&keyword[..keyword.len().min(found.len())]) {
                format!("Did you mean '{}'? (check spelling)", keyword)
            } else {
                format!("Expected keyword '{}'", keyword)
            };
            
            Err(self.error(format!(
                "{}, but found '{}'",
                suggestion, found
            )))
        }
    }

    /// Parses an identifier string.
    pub fn parse_identifier_str(&mut self) -> Result<&'a str, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        let first_byte = *self
            .bytes
            .get(start)
            .ok_or_else(|| self.error("Unexpected end of input while parsing identifier\n  Hint: Identifiers must start with a letter (a-z, A-Z) or underscore (_)".to_string()))?;
        if !(first_byte.is_ascii_alphabetic() || first_byte == b'_') {
            let found_char = first_byte as char;
            let hint = if found_char.is_ascii_digit() {
                "Identifiers cannot start with a digit. Did you mean to use a numeric literal instead?"
            } else {
                "Identifiers must start with a letter (a-z, A-Z) or underscore (_)"
            };
            return Err(self.error(format!(
                "Invalid identifier start: found '{}'\n  Hint: {}",
                found_char, hint
            )));
        }
        self.advance();

        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];
            if byte.is_ascii_alphanumeric() || byte == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(&self.input[start..self.position])
    }

    /// Parses a type identifier (starts with '@').
    pub fn parse_type_identifier(&mut self) -> Result<crate::Identifier<'a>, LaminaError> {
        self.expect_char('@')?;
        self.parse_identifier_str()
    }

    /// Parses a value identifier (starts with '%').
    pub fn parse_value_identifier(&mut self) -> Result<crate::Identifier<'a>, LaminaError> {
        self.expect_char('%')?;
        self.parse_identifier_str()
    }

    /// Parses a label identifier.
    pub fn parse_label_identifier(&mut self) -> Result<crate::Label<'a>, LaminaError> {
        self.parse_identifier_str()
    }

    /// Parses an integer literal.
    pub fn parse_integer(&mut self) -> Result<i64, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        let mut negative = false;
        if !self.is_eof() && self.bytes[self.position] == b'-' {
            negative = true;
            self.advance();
        }

        while !self.is_eof() && self.bytes[self.position].is_ascii_digit() {
            self.advance();
        }

        if start == self.position || (negative && start + 1 == self.position) {
            Err(self.error("Expected an integer literal\n  Hint: Integer literals can be positive (e.g., 42) or negative (e.g., -42)".to_string()))
        } else {
            let digits = &self.input[if negative { start + 1 } else { start }..self.position];

            if negative && digits == "9223372036854775808" {
                return Ok(i64::MIN);
            }

            if negative {
                match digits.parse::<i64>() {
                    Ok(val) => Ok(-val),
                    Err(e) => Err(self.error(format!("Failed to parse integer: {}", e))),
                }
            } else {
                digits
                    .parse::<i64>()
                    .map_err(|e| self.error(format!("Failed to parse integer: {}", e)))
            }
        }
    }

    /// Parses a float literal.
    pub fn parse_float(&mut self) -> Result<f32, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        if !self.is_eof() && self.bytes[self.position] == b'-' {
            self.advance();
        }

        let mut has_digit = false;

        while !self.is_eof() && self.bytes[self.position].is_ascii_digit() {
            has_digit = true;
            self.advance();
        }

        if !self.is_eof() && self.bytes[self.position] == b'.' {
            self.advance();

            while !self.is_eof() && self.bytes[self.position].is_ascii_digit() {
                has_digit = true;
                self.advance();
            }
        }

        if !has_digit {
            return Err(self.error("Expected a floating-point literal\n  Hint: Float literals must contain at least one digit (e.g., 3.14, -0.5, 42.0)".to_string()));
        }

        let value_str = &self.input[start..self.position];
        value_str
            .parse::<f32>()
            .map_err(|e| self.error(format!("Failed to parse float: {}", e)))
    }

    /// Parses a string literal.
    pub fn parse_string_literal(&mut self) -> Result<&'a str, LaminaError> {
        self.expect_char('"')?;
        let start = self.position;
        while !self.is_eof() && self.bytes[self.position] != b'"' {
            self.advance();
        }
        let end = self.position;
        self.expect_char('"')?;
        Ok(&self.input[start..end])
    }

    /// Creates a parsing error with the given message, including line and column information.
    pub fn error(&self, message: String) -> LaminaError {
        let (line, column) = self.get_line_column();
        let context = self.get_error_context();
        
        let error_msg = if context.is_empty() {
            format!("{} at line {}, column {}", message, line, column)
        } else {
            format!(
                "{}\n  at line {}, column {}\n  {}\n  {}^",
                message, line, column, context, " ".repeat(column.saturating_sub(1))
            )
        };
        
        LaminaError::ParsingError(error_msg)
    }
    
    /// Gets the line and column number for the current position.
    fn get_line_column(&self) -> (usize, usize) {
        let mut line = 1;
        let mut column = 1;
        let mut pos = 0;
        
        for (i, ch) in self.input.char_indices() {
            if i >= self.position {
                break;
            }
            if ch == '\n' {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
            pos = i;
        }
        
        (line, column)
    }
    
    /// Gets a context string showing the line where the error occurred.
    fn get_error_context(&self) -> String {
        let (line_num, _) = self.get_line_column();
        let lines: Vec<&str> = self.input.lines().collect();
        
        if line_num > 0 && line_num <= lines.len() {
            let line = lines[line_num - 1];
            // Truncate very long lines for readability
            if line.len() > 80 {
                format!("{}...", &line[..77])
            } else {
                line.to_string()
            }
        } else {
            String::new()
        }
    }

    /// Returns the current parser position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Sets the parser position.
    pub fn set_position(&mut self, pos: usize) {
        self.position = pos;
    }

    /// Peeks at a byte at the given offset from current position.
    pub fn peek_byte(&self, offset: usize) -> Option<u8> {
        let pos = self.position + offset;
        if pos < self.bytes.len() {
            Some(self.bytes[pos])
        } else {
            None
        }
    }

    /// Returns the bytes slice for peeking.
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }
}
