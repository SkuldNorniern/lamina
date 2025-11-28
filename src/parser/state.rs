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
            Err(self.error(format!(
                "Expected character '{}', found {:?}",
                expected,
                self.current_char()
            )))
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
                Err(self.error(format!(
                    "Expected keyword '{}', but found longer identifier starting with it",
                    keyword
                )))
            }
        } else {
            Err(self.error(format!(
                "Expected keyword '{}', found {:?}",
                keyword,
                self.peek_slice(keyword.len())
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
            .ok_or_else(|| self.error("Unexpected EOF parsing identifier".to_string()))?;
        if !(first_byte.is_ascii_alphabetic() || first_byte == b'_') {
            return Err(self.error(format!(
                "Identifier must start with a letter or underscore, found '{}'",
                first_byte as char
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
            Err(self.error("Expected an integer".to_string()))
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
            return Err(self.error("Expected a float".to_string()));
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

    /// Creates a parsing error with the given message.
    pub fn error(&self, message: String) -> LaminaError {
        LaminaError::ParsingError(format!("{} at position {}", message, self.position))
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
