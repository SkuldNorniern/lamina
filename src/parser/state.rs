use crate::LaminaError;
use std::result::Result;

#[derive(Debug)]
pub struct ParserState<'a> {
    input: &'a str,
    bytes: &'a [u8], // For efficient char checking
    position: usize, // Current byte position
}

impl<'a> ParserState<'a> {
    pub fn new(input: &'a str) -> Self {
        ParserState {
            input,
            bytes: input.as_bytes(),
            position: 0,
        }
    }

    // --- Basic Utils ---

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

    pub fn advance(&mut self) {
        if let Some(c) = self.current_char() {
            self.position += c.len_utf8();
        } else {
            // Cannot advance past end
        }
    }

    pub fn advance_by(&mut self, count: usize) {
        for _ in 0..count {
            self.advance();
        }
    }

    pub fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }

    pub fn skip_whitespace_and_comments(&mut self) {
        while !self.is_eof() {
            let byte = self.bytes[self.position];
            if byte.is_ascii_whitespace() {
                self.advance();
            } else if byte == b'#' {
                // Treat '#' as line comment start
                while !self.is_eof() && self.bytes[self.position] != b'\n' {
                    self.advance();
                }
                // Skip the newline itself if not EOF
                if !self.is_eof() {
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    // --- Token/Keyword Parsing ---

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

    pub fn consume_keyword(&mut self, keyword: &str) -> Result<(), LaminaError> {
        self.skip_whitespace_and_comments();
        if self.input[self.position..].starts_with(keyword) {
            // Check if keyword is followed by non-alphanumeric (or EOF)
            let next_char_pos = self.position + keyword.len();
            if next_char_pos >= self.input.len()
                || !self.input.as_bytes()[next_char_pos].is_ascii_alphanumeric()
            {
                self.position += keyword.len(); // Advance by byte length
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

    pub fn parse_identifier_str(&mut self) -> Result<&'a str, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        // Check the first character specifically
        // Basic identifiers (like types, keywords parsed as identifiers initially)
        // must start with a letter or underscore.
        let first_byte = *self
            .bytes
            .get(start)
            .ok_or_else(|| self.error("Unexpected EOF parsing identifier".to_string()))?;
        if !(first_byte.is_ascii_alphabetic() || first_byte == b'_') {
            // Identifiers used for values (%), types/globals (@), or labels (raw)
            // have their prefixes handled by their specific parse functions.
            // This base function expects a standard identifier start.
            return Err(self.error(format!(
                "Identifier must start with a letter or underscore, found '{}'",
                first_byte as char
            )));
        }
        self.advance(); // Consume first char

        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];
            if byte.is_ascii_alphanumeric() || byte == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        // No need to check start == self.position, as the first char check ensures we advance at least once.
        Ok(&self.input[start..self.position])
    }

    pub fn parse_type_identifier(&mut self) -> Result<crate::Identifier<'a>, LaminaError> {
        self.expect_char('@')?;
        self.parse_identifier_str()
    }

    pub fn parse_value_identifier(&mut self) -> Result<crate::Identifier<'a>, LaminaError> {
        self.expect_char('%')?;
        self.parse_identifier_str()
    }

    pub fn parse_label_identifier(&mut self) -> Result<crate::Label<'a>, LaminaError> {
        self.parse_identifier_str()
    }

    // Basic integer parsing (replace with more robust later)
    pub fn parse_integer(&mut self) -> Result<i64, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        // Handle negative sign for integer literals
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

            // Special case for i64::MIN
            if negative && digits == "9223372036854775808" {
                return Ok(i64::MIN);
            }

            // Parse as i64 directly to avoid overflow when handling large negative values
            if negative {
                // Handle negative numbers (avoid overflow by parsing as positive and then negating)
                match digits.parse::<i64>() {
                    Ok(val) => {
                        // No special check needed here anymore since we have a special case for MIN above
                        Ok(-val)
                    }
                    Err(e) => Err(self.error(format!("Failed to parse integer: {}", e))),
                }
            } else {
                // Parse positive numbers directly
                digits
                    .parse::<i64>()
                    .map_err(|e| self.error(format!("Failed to parse integer: {}", e)))
            }
        }
    }

    // Basic float parsing (improved to handle negative values)
    pub fn parse_float(&mut self) -> Result<f32, LaminaError> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        // Handle negative sign
        if !self.is_eof() && self.bytes[self.position] == b'-' {
            self.advance();
        }

        let mut has_digit = false;

        // Parse digits before decimal point
        while !self.is_eof() && self.bytes[self.position].is_ascii_digit() {
            has_digit = true;
            self.advance();
        }

        // Parse decimal point and digits after it
        if !self.is_eof() && self.bytes[self.position] == b'.' {
            self.advance();

            // Parse digits after decimal point
            while !self.is_eof() && self.bytes[self.position].is_ascii_digit() {
                has_digit = true;
                self.advance();
            }
        }

        // We must have at least one digit, and either a dot or at least one digit before the dot
        if !has_digit {
            return Err(self.error("Expected a float".to_string()));
        }

        // Parse the float
        let value_str = &self.input[start..self.position];
        value_str
            .parse::<f32>()
            .map_err(|e| self.error(format!("Failed to parse float: {}", e)))
    }

    // String literal parsing (basic)
    pub fn parse_string_literal(&mut self) -> Result<&'a str, LaminaError> {
        self.expect_char('"')?;
        let start = self.position;
        while !self.is_eof() && self.bytes[self.position] != b'"' {
            // Note: Escape sequences not yet supported in string literals
            self.advance();
        }
        let end = self.position;
        self.expect_char('"')?;
        Ok(&self.input[start..end])
    }

    // --- Error Helper ---
    pub fn error(&self, message: String) -> LaminaError {
        // Note: Line/column info not currently tracked
        LaminaError::ParsingError(format!("{} at position {}", message, self.position))
    }

    // Expose position for backtracking
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, pos: usize) {
        self.position = pos;
    }

    // Peek at a byte at a given offset from current position
    pub fn peek_byte(&self, offset: usize) -> Option<u8> {
        let pos = self.position + offset;
        if pos < self.bytes.len() {
            Some(self.bytes[pos])
        } else {
            None
        }
    }

    // Get the bytes slice for peeking
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }
}
