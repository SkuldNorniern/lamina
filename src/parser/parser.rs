use crate::{
    /*error::{*/LaminaError, Result/*}*/,
    // ir::*,
    Module, Identifier, Label, Type, Value, Literal, PrimitiveType, StructField, 
    Instruction, BinaryOp, CmpOp, AllocType, Function, BasicBlock, FunctionSignature, 
    FunctionParameter, FunctionAnnotation, TypeDeclaration, GlobalDeclaration,
};
use std::collections::HashMap;

// --- Parser State --- 

#[derive(Debug)]
pub struct ParserState<'a> {
    input: &'a str, 
    bytes: &'a [u8], // For efficient char checking
    position: usize, // Current byte position
}

impl<'a> ParserState<'a> {
    pub fn new(input: &'a str) -> Self {
        ParserState { input, bytes: input.as_bytes(), position: 0 }
    }

    // --- Basic Utils ---

    fn current_char(&self) -> Option<char> {
        self.peek_char(0)
    }

    fn peek_char(&self, offset: usize) -> Option<char> {
        self.input[self.position..].chars().nth(offset)
    }

    fn peek_slice(&self, len: usize) -> Option<&'a str> {
        if self.position + len <= self.input.len() {
            Some(&self.input[self.position .. self.position + len])
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if let Some(c) = self.current_char() {
            self.position += c.len_utf8();
        } else {
            // Cannot advance past end
        }
    }
    
    fn advance_by(&mut self, count: usize) {
        for _ in 0..count {
            self.advance();
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }

    fn skip_whitespace_and_comments(&mut self) {
        while !self.is_eof() {
            let byte = self.bytes[self.position];
            if byte.is_ascii_whitespace() {
                self.advance();
            } else if byte == b'#' { // Treat '#' as line comment start
                while !self.is_eof() && self.bytes[self.position] != b'\n' {
                    self.advance();
                }
                // Skip the newline itself if not EOF
                if !self.is_eof() { self.advance(); }
            } else {
                break;
            }
        }
    }

    // --- Token/Keyword Parsing ---

    fn expect_char(&mut self, expected: char) -> Result<()> {
        self.skip_whitespace_and_comments();
        if self.current_char() == Some(expected) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("Expected character '{}', found {:?}", expected, self.current_char())))
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> Result<()> {
        self.skip_whitespace_and_comments();
        if self.input[self.position..].starts_with(keyword) {
             // Check if keyword is followed by non-alphanumeric (or EOF)
             let next_char_pos = self.position + keyword.len();
             if next_char_pos >= self.input.len() || 
                !self.input.as_bytes()[next_char_pos].is_ascii_alphanumeric() 
             {
                 self.position += keyword.len(); // Advance by byte length
                 Ok(())
             } else {
                Err(self.error(format!("Expected keyword '{}', but found longer identifier starting with it", keyword)))
            }
        } else {
            Err(self.error(format!("Expected keyword '{}', found {:?}", keyword, self.peek_slice(keyword.len()))))
        }
    }

    fn parse_identifier_str(&mut self) -> Result<&'a str> {
        self.skip_whitespace_and_comments();
        let start = self.position;

        // Check the first character specifically
        // Basic identifiers (like types, keywords parsed as identifiers initially)
        // must start with a letter or underscore.
        let first_byte = *self.bytes.get(start).ok_or_else(|| self.error("Unexpected EOF parsing identifier".to_string()))?;
        if !(first_byte.is_ascii_alphabetic() || first_byte == b'_') {
             // Identifiers used for values (%), types/globals (@), or labels (raw)
             // have their prefixes handled by their specific parse functions.
             // This base function expects a standard identifier start.
             return Err(self.error(format!("Identifier must start with a letter or underscore, found '{}'", first_byte as char)));
        }
        self.advance(); // Consume first char

        // Consume subsequent alphanumeric or underscore characters
        while let Some(byte) = self.bytes.get(self.position) {
            if byte.is_ascii_alphanumeric() || *byte == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        // No need to check start == self.position, as the first char check ensures we advance at least once.
        Ok(&self.input[start..self.position])
    }

    fn parse_type_identifier(&mut self) -> Result<Identifier<'a>> {
         self.expect_char('@')?;
         self.parse_identifier_str()
    }
    
    fn parse_value_identifier(&mut self) -> Result<Identifier<'a>> {
         self.expect_char('%')?;
         self.parse_identifier_str()
    }

    fn parse_label_identifier(&mut self) -> Result<Label<'a>> {
        self.parse_identifier_str()
    }

    // Basic integer parsing (replace with more robust later)
    fn parse_integer(&mut self) -> Result<u64> {
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
            let parsed = digits.parse::<u64>().map_err(|e|
                self.error(format!("Failed to parse integer: {}", e))
            )?;
            
            // Apply negative sign if needed
            Ok(if negative {
                // Use wrapping conversion for negative values
                (-(parsed as i64)) as u64
            } else {
                parsed
            })
        }
    }
    
     // Basic float parsing (very basic)
    fn parse_float(&mut self) -> Result<f32> {
        self.skip_whitespace_and_comments();
        let start = self.position;
        let mut has_dot = false;
         while !self.is_eof() {
             let byte = self.bytes[self.position];
             if byte.is_ascii_digit() {
                 self.advance();
             } else if byte == b'.' && !has_dot {
                 has_dot = true;
                 self.advance();
             } else {
                 break;
             }
         }
         if start == self.position {
            Err(self.error("Expected a float".to_string()))
        } else {
             self.input[start..self.position].parse::<f32>().map_err(|e|
                 self.error(format!("Failed to parse float: {}", e))
             )
        }
    }

    // String literal parsing (basic)
    fn parse_string_literal(&mut self) -> Result<&'a str> {
        self.expect_char('"')?;
        let start = self.position;
        while !self.is_eof() && self.bytes[self.position] != b'"' {
            // TODO: Handle escape sequences
            self.advance();
        }
        let end = self.position;
        self.expect_char('"')?;
        Ok(&self.input[start..end])
    }

    // --- Error Helper ---
    fn error(&self, message: String) -> LaminaError {
        // TODO: Add line/column info
        LaminaError::ParsingError(format!("{} at position {}", message, self.position))
    }
}

// --- Main Parsing Function --- 

/// Parses a string containing Lamina IR text into a Module.
/// The lifetime 'a is tied to the input string slice.
pub fn parse_module<'a>(input: &'a str) -> Result<Module<'a>> {
    let mut state = ParserState::new(input);
    let mut module = Module::new();

    loop {
        state.skip_whitespace_and_comments();
        if state.is_eof() {
            break;
        }

        let keyword_slice = state.peek_slice(6).unwrap_or(""); // Peek enough for keywords

        if keyword_slice.starts_with("type") {
            let decl = parse_type_declaration(&mut state)?;
            module.type_declarations.insert(decl.name, decl);
        } else if keyword_slice.starts_with("global") {
            let decl = parse_global_declaration(&mut state)?;
            module.global_declarations.insert(decl.name, decl);
        } else if keyword_slice.starts_with("fn") || keyword_slice.starts_with('@') {
            let func = parse_function_def(&mut state)?;
            module.functions.insert(func.name, func);
        } else {
             return Err(state.error(format!("Unexpected token at top level: {:?}", state.peek_slice(10))));
        }
    }

    Ok(module)
}

// --- Type Parsing ---

fn parse_type_declaration<'a>(state: &mut ParserState<'a>) -> Result<TypeDeclaration<'a>> {
    state.consume_keyword("type")?;
    let name = state.parse_type_identifier()?;
    state.expect_char('=')?;
    let ty = parse_composite_type(state)?;
    Ok(TypeDeclaration { name, ty })
}

fn parse_composite_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>> {
    state.skip_whitespace_and_comments();
    let keyword_slice = state.peek_slice(6).unwrap_or(""); // Peek "struct"
    if keyword_slice.starts_with("struct") {
        // Struct: struct { name: type, ... }
        state.consume_keyword("struct")?;
        state.expect_char('{')?;
        let mut fields = Vec::new();
        loop {
            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') { 
                state.advance();
                break; 
            }
            let field_name = state.parse_identifier_str()?;
            state.expect_char(':')?;
            let field_ty = parse_type(state)?;
            fields.push(StructField { name: field_name, ty: field_ty });

            state.skip_whitespace_and_comments();
            if state.current_char() == Some('}') { 
                 state.advance();
                 break;
             }
            state.expect_char(',')?; // Expect comma or closing brace
        }
        Ok(Type::Struct(fields))
    } else if state.current_char() == Some('[') {
        // Array: [ size x type ]
        state.expect_char('[')?;
        let size = state.parse_integer()?;
        state.consume_keyword("x")?;
        let element_type = parse_type(state)?;
        state.expect_char(']')?;
        Ok(Type::Array { element_type: Box::new(element_type), size })
    } else {
        Err(state.error("Expected 'struct' or '[' for composite type".to_string()))
    }
}

fn parse_type<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>> {
    state.skip_whitespace_and_comments();
    match state.current_char() {
        Some('@') => {
            // Named type: @Name
            Ok(Type::Named(state.parse_type_identifier()?))
        }
        Some('[') => {
            // Inline Array Type
            parse_composite_type(state) // Delegate to handle arrays
        }
        Some('s') if state.peek_slice(6) == Some("struct") => {
             // Inline Struct Type
             parse_composite_type(state) // Delegate to handle structs
        }
        _ => {
            // Try matching primitive types
            let potential_primitive = state.parse_identifier_str()?;
            match potential_primitive {
                "i8" => Ok(Type::Primitive(PrimitiveType::I8)),
                "i32" => Ok(Type::Primitive(PrimitiveType::I32)),
                "i64" => Ok(Type::Primitive(PrimitiveType::I64)),
                "f32" => Ok(Type::Primitive(PrimitiveType::F32)),
                "bool" => Ok(Type::Primitive(PrimitiveType::Bool)),
                "ptr" => Ok(Type::Primitive(PrimitiveType::Ptr)),
                "void" => Ok(Type::Void),
                _ => Err(state.error(format!("Unknown or unexpected type identifier: {}", potential_primitive)))
            }
        }
    }
}

// --- Global Parsing ---

fn parse_global_declaration<'a>(state: &mut ParserState<'a>) -> Result<GlobalDeclaration<'a>> {
    state.consume_keyword("global")?;
    let name = state.parse_type_identifier()?; // Globals start with @
    state.expect_char(':')?;
    let ty = parse_type(state)?;
    
    state.skip_whitespace_and_comments();
    let initializer = if state.current_char() == Some('=') {
        state.advance(); // Consume '='
        // For global initializer, use the appropriate literal type based on declared type
        let value = parse_value_with_type_hint(state, &ty)?;
        Some(value) 
    } else {
        None
    };

    Ok(GlobalDeclaration { name, ty, initializer })
}

// Parse a value potentially using type hints for integer literals
fn parse_value_with_type_hint<'a>(state: &mut ParserState<'a>, type_hint: &Type<'a>) -> Result<Value<'a>> {
    state.skip_whitespace_and_comments();
    match state.current_char() {
        Some('%') => Ok(Value::Variable(state.parse_value_identifier()?)),
        Some('@') => Ok(Value::Global(state.parse_type_identifier()?)),
        Some('"') => Ok(Value::Constant(Literal::String(state.parse_string_literal()?))),
        Some('t') if state.peek_slice(4) == Some("true") => {
            state.consume_keyword("true")?;
            Ok(Value::Constant(Literal::Bool(true)))
        }
        Some('f') if state.peek_slice(5) == Some("false") => {
            state.consume_keyword("false")?;
            Ok(Value::Constant(Literal::Bool(false)))
        }
        Some(c) if c.is_ascii_digit() || c == '-' => {
            // Determine integer type based on type hint
            let use_i64 = match type_hint {
                Type::Primitive(PrimitiveType::I64) => true,
                _ => false,
            };
            
            // Could be integer or float
            let _current_pos = state.position; // Prefix unused variable
            if let Ok(f_val) = state.parse_float() {
                if state.input[_current_pos..state.position].contains('.') { 
                    Ok(Value::Constant(Literal::F32(f_val)))
                } else {
                    // It parsed as float but looked like an integer
                    state.position = _current_pos; // Backtrack
                    let int_val = state.parse_integer()? as i64;
                    if use_i64 {
                        Ok(Value::Constant(Literal::I64(int_val)))
                    } else {
                        Ok(Value::Constant(Literal::I32(int_val as i32)))
                    }
                }
            } else {
                // Failed float parse, try integer
                state.position = _current_pos; // Backtrack
                let int_val = state.parse_integer()? as i64;
                if use_i64 {
                    Ok(Value::Constant(Literal::I64(int_val)))
                } else {
                    Ok(Value::Constant(Literal::I32(int_val as i32)))
                }
            }
        }
        _ => Err(state.error("Expected value (%, @, literal)".to_string()))
    }
}

// Parses a value: literal or %variable or @global
// This version defaults to using I32 for integer literals
fn parse_value<'a>(state: &mut ParserState<'a>) -> Result<Value<'a>> {
    // Use I32 as the default type for integer literals
    let default_type = Type::Primitive(PrimitiveType::I32);
    parse_value_with_type_hint(state, &default_type)
}

// --- Function Parsing ---

fn parse_annotations<'a>(state: &mut ParserState<'a>) -> Result<Vec<FunctionAnnotation>> {
    let mut annotations = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('@') {
            state.advance();
            let name = state.parse_identifier_str()?;
            let annotation = match name {
                "inline" => FunctionAnnotation::Inline,
                "export" => FunctionAnnotation::Export,
                "noreturn" => FunctionAnnotation::NoReturn,
                "noinline" => FunctionAnnotation::NoInline,
                "cold" => FunctionAnnotation::Cold,
                _ => return Err(state.error(format!("Unknown function annotation: @{}", name))),
            };
            annotations.push(annotation);
        } else {
            break; // No more annotations
        }
    }
    Ok(annotations)
}

fn parse_function_def<'a>(state: &mut ParserState<'a>) -> Result<Function<'a>> {
    let annotations = parse_annotations(state)?;
    state.consume_keyword("fn")?;
    let name = state.parse_type_identifier()?; // Functions also start with @ in decl
    let signature = parse_fn_signature(state)?;
    state.expect_char('{')?;

    let mut basic_blocks = HashMap::new();
    let mut entry_block_label: Option<Label<'a>> = None;

    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some('}') {
            state.advance(); // Consume closing brace
            break;
        }
        
        // Parse Label: instruction instructions...
        let (label, block) = parse_basic_block(state)?;

        if entry_block_label.is_none() {
            entry_block_label = Some(label);
        }
        
        if basic_blocks.insert(label, block).is_some() {
            return Err(state.error(format!("Redefinition of basic block label: {}", label)));
        }
    }

    let entry_block = entry_block_label.ok_or_else(|| state.error("Function must have at least one basic block".to_string()))?;

    Ok(Function {
        name,
        signature,
        annotations,
        basic_blocks,
        entry_block,
    })
}

fn parse_fn_signature<'a>(state: &mut ParserState<'a>) -> Result<FunctionSignature<'a>> {
    state.expect_char('(')?;
    let params = parse_param_list(state)?;
    state.expect_char(')')?;
    state.consume_keyword("->")?;
    let return_type = parse_type(state)?;
    Ok(FunctionSignature { params, return_type })
}

fn parse_param_list<'a>(state: &mut ParserState<'a>) -> Result<Vec<FunctionParameter<'a>>> {
    let mut params = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
             break; // End of parameters
        }
        
        let param_ty = parse_type(state)?;
        let param_name = state.parse_value_identifier()?; // Params start with %
        params.push(FunctionParameter { name: param_name, ty: param_ty });

        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            break;
        }
        state.expect_char(',')?; // Expect comma or closing paren
    }
    Ok(params)
}

// --- Block and Instruction Parsing (Placeholders) ---

fn parse_basic_block<'a>(state: &mut ParserState<'a>) -> Result<(Label<'a>, BasicBlock<'a>)> {
     // Example: entry: instruction 
     let label = state.parse_label_identifier()?;
     state.expect_char(':')?;
     
     let mut instructions = Vec::new();
     loop {
         state.skip_whitespace_and_comments();
         // Check if we are at the start of the next block label or end of function
         let _current_pos = state.position; // Prefix unused variable
         if let Ok(_) = state.parse_label_identifier() {
             // Check if followed by a colon
             if state.current_char() == Some(':') {
                 state.position = _current_pos; // Backtrack, it's the next label
                 break;
             }
         }
         state.position = _current_pos; // Backtrack if it wasn't label:
         
         // Check for end of function
         if state.current_char() == Some('}') {
             break;
         }
         
         // If not label or end, parse an instruction
         let instruction = parse_instruction(state)?;
         let is_terminator = instruction.is_terminator(); // Need to add this method
         instructions.push(instruction);
         
         if is_terminator {
             break; // Block finished
         }
     }
     
     if instructions.is_empty() || !instructions.last().unwrap().is_terminator() {
         return Err(state.error(format!("Basic block '{}' must end with a terminator instruction (ret, jmp, br)", label)));
     }

     Ok((label, BasicBlock { instructions }))
}

fn parse_instruction<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    state.skip_whitespace_and_comments();
    let _start_pos = state.position; // Prefixed with _
    
    // Peek to see if it starts with % (result assignment) or an identifier (opcode)
    if state.current_char() == Some('%') {
        // --- Assignment form: %result = opcode ... ---
        let result = state.parse_value_identifier()?;
        state.expect_char('=')?;
        state.skip_whitespace_and_comments();
        let opcode_str = state.parse_identifier_str()?;
        
        // Parse based on opcode
        match opcode_str {
            "add" | "sub" | "mul" | "div" => parse_binary_op(state, result, opcode_str),
            "eq" | "ne" | "gt" | "ge" | "lt" | "le" => parse_cmp_op(state, result, opcode_str),
            "zext" => parse_zext(state, result),
            "alloc" => parse_alloc(state, result),
            "load" => parse_load(state, result),
            "getfield" => parse_getfield(state, result),
            "getfieldptr" => parse_getfield(state, result), // Alias for test compatibility
            "getelem" => parse_getelem(state, result),
            "getelementptr" => parse_getelem(state, result), // Alias for test compatibility
            "tuple" => parse_tuple(state, result),
            "extract" => parse_extract_tuple(state, result),
            "call" => parse_call(state, Some(result)), // Call with result
            "phi" => parse_phi(state, result),
             _ => Err(state.error(format!("Unknown opcode after assignment: {}", opcode_str)))
        }
    } else {
        // --- Non-assignment form: opcode ... ---
        let opcode_str = state.parse_identifier_str()?;
        match opcode_str {
            "store" => parse_store(state),
            "br" => parse_br(state),
            "jmp" => parse_jmp(state),
            "ret" => parse_ret(state),
            "dealloc" => parse_dealloc(state),
            "call" => parse_call(state, None), // Call without result (void)
            "print" => parse_print(state), // Add print case
            _ => Err(state.error(format!("Unknown instruction opcode: {}", opcode_str)))
        }
    }
}

// --- Specific Instruction Parsers --- 

fn parse_primitive_type_suffix<'a>(state: &mut ParserState<'a>) -> Result<PrimitiveType> {
     state.expect_char('.')?;
     let type_str = state.parse_identifier_str()?;
     match type_str {
            "i8" => Ok(PrimitiveType::I8),
            "i32" => Ok(PrimitiveType::I32),
            "i64" => Ok(PrimitiveType::I64),
            "f32" => Ok(PrimitiveType::F32),
            "bool" => Ok(PrimitiveType::Bool),
            "ptr" => Ok(PrimitiveType::Ptr),
            _ => Err(state.error(format!("Expected primitive type suffix, found '.{}'", type_str)))
     }
}

fn parse_type_suffix<'a>(state: &mut ParserState<'a>) -> Result<Type<'a>> {
     state.expect_char('.')?;
     parse_type(state) // Can be any type after the dot
}

fn parse_binary_op<'a>(state: &mut ParserState<'a>, result: Identifier<'a>, op_str: &str) -> Result<Instruction<'a>> {
    let op = match op_str {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        "div" => BinaryOp::Div,
        _ => unreachable!() // Should be checked before calling
    };
    let ty = parse_primitive_type_suffix(state)?;
    let lhs = parse_value(state)?;
    state.expect_char(',')?;
    let rhs = parse_value(state)?;
    Ok(Instruction::Binary { op, result, ty, lhs, rhs })
}

fn parse_cmp_op<'a>(state: &mut ParserState<'a>, result: Identifier<'a>, op_str: &str) -> Result<Instruction<'a>> {
     let op = match op_str {
        "eq" => CmpOp::Eq, "ne" => CmpOp::Ne,
        "gt" => CmpOp::Gt, "ge" => CmpOp::Ge,
        "lt" => CmpOp::Lt, "le" => CmpOp::Le,
        _ => unreachable!()
    };
    let ty = parse_primitive_type_suffix(state)?;
    let lhs = parse_value(state)?;
    state.expect_char(',')?;
    let rhs = parse_value(state)?;
    Ok(Instruction::Cmp { op, result, ty, lhs, rhs })
}

fn parse_alloc<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
     // Format: alloc.ptr.stack T or alloc.ptr.heap T
     // Also support: alloc.stack T or alloc.heap T directly
     state.expect_char('.')?;
     
     // Try to match allocation type directly
     let peek_str = state.peek_slice(5).unwrap_or("");
     if peek_str.starts_with("stack") {
         state.consume_keyword("stack")?;
         let allocated_ty = parse_type(state)?;
         return Ok(Instruction::Alloc { result, alloc_type: AllocType::Stack, allocated_ty });
     } else if peek_str.starts_with("heap") {
         state.consume_keyword("heap")?;
         let allocated_ty = parse_type(state)?;
         return Ok(Instruction::Alloc { result, alloc_type: AllocType::Heap, allocated_ty });
     }
     
     // Original format: alloc.ptr.{stack|heap}
     state.consume_keyword("ptr")?;
     state.expect_char('.')?;
     let alloc_type_str = state.parse_identifier_str()?;
     let alloc_type = match alloc_type_str {
         "stack" => AllocType::Stack,
         "heap" => AllocType::Heap,
         _ => return Err(state.error(format!("Invalid allocation type: {}", alloc_type_str)))
     };
     let allocated_ty = parse_type(state)?;
     state.skip_whitespace_and_comments(); // Consume trailing whitespace/newline
     Ok(Instruction::Alloc { result, alloc_type, allocated_ty })
}

fn parse_load<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
     // load.T ptr
     let ty = parse_type_suffix(state)?;
     let ptr = parse_value(state)?;
     Ok(Instruction::Load { result, ty, ptr })
}

fn parse_store<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
     // store.T ptr, val
     let ty = parse_type_suffix(state)?;
     let ptr = parse_value(state)?;
     state.expect_char(',')?;
     let value = parse_value(state)?;
     Ok(Instruction::Store { ty, ptr, value })
}

fn parse_getfield<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // Format should be: getfield.ptr struct_ptr, index
    // Support both formats:
    // 1. getfield.ptr struct_ptr, index
    // 2. getfieldptr struct_ptr, index (without the dot)
    
    // Check if we have a dot
    let current_pos = state.position;
    let has_dot = state.current_char() == Some('.');
    
    if has_dot {
        state.expect_char('.')?;
        state.consume_keyword("ptr")?;
    }
    
    let struct_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let field_index = state.parse_integer()? as usize;
    Ok(Instruction::GetFieldPtr { result, struct_ptr, field_index })
}

fn parse_getelem<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // Format should be: getelem.ptr array_ptr, index
    // Support both formats:
    // 1. getelem.ptr array_ptr, index
    // 2. getelementptr array_ptr, index (without the dot)
    
    // Check if we have a dot
    let current_pos = state.position;
    let has_dot = state.current_char() == Some('.');
    
    if has_dot {
        state.expect_char('.')?;
        state.consume_keyword("ptr")?;
    }
    
    let array_ptr = parse_value(state)?;
    state.expect_char(',')?;
    let index = parse_value(state)?; // Index can be variable or constant
    Ok(Instruction::GetElemPtr { result, array_ptr, index })
}

fn parse_tuple<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // tuple.T element1, element2, ... (Simplified: just parse elements)
    let mut elements = Vec::new();
    loop {
         state.skip_whitespace_and_comments();
         // Check if next token could be an operand or if we reached end of line/block
         let current_pos = state.position;
         if parse_value(state).is_ok() { 
             state.position = current_pos; // Backtrack
             let elem = parse_value(state)?;
             elements.push(elem);
             state.skip_whitespace_and_comments();
             if state.current_char() != Some(',') {
                 break; // No more elements
             }
             state.expect_char(',')?;
         } else {
             state.position = current_pos; // Backtrack
             break; // No more operands
         }
    }
    if elements.is_empty() {
        return Err(state.error("tuple instruction requires at least one element".to_string()));
    }
    Ok(Instruction::Tuple { result, elements })
}

fn parse_extract_tuple<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // extract.tuple tuple_val, index
    state.expect_char('.')?;
    state.consume_keyword("tuple")?;
    let tuple_val = parse_value(state)?;
    state.expect_char(',')?;
    let index = state.parse_integer()? as usize;
    Ok(Instruction::ExtractTuple { result, tuple_val, index })
}

fn parse_call<'a>(state: &mut ParserState<'a>, result: Option<Identifier<'a>>) -> Result<Instruction<'a>> {
    // call @func(arg1, arg2, ...)
    let func_name = state.parse_type_identifier()?; // @func
    state.expect_char('(')?;
    let mut args = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance(); // Consume ')'
            break;
        }
        args.push(parse_value(state)?);
        state.skip_whitespace_and_comments();
        if state.current_char() == Some(')') {
            state.advance(); // Consume ')'
            break;
        }
        state.expect_char(',')?;
    }
    Ok(Instruction::Call { result, func_name, args })
}

fn parse_phi<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // phi.T [val1, label1], [val2, label2], ...
    let ty = parse_type_suffix(state)?;
    let mut incoming = Vec::new();
    loop {
        state.skip_whitespace_and_comments();
        if state.current_char() != Some('[') {
            break; // End of incoming values
        }
        state.expect_char('[')?;
        let value = parse_value(state)?;
        state.expect_char(',')?;
        let label = state.parse_label_identifier()?;
        state.expect_char(']')?;
        incoming.push((value, label));

        state.skip_whitespace_and_comments();
        if state.current_char() != Some(',') {
            break; // No more pairs
        }
        state.expect_char(',')?;
    }
    if incoming.is_empty() {
        return Err(state.error("phi instruction requires at least one incoming value".to_string()));
    }
    Ok(Instruction::Phi { result, ty, incoming })
}

fn parse_br<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    // br cond, label1, label2
    let condition = parse_value(state)?;
    state.expect_char(',')?;
    let true_label = state.parse_label_identifier()?;
    state.expect_char(',')?;
    let false_label = state.parse_label_identifier()?;
    Ok(Instruction::Br { condition, true_label, false_label })
}

fn parse_jmp<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    // jmp label
    let target_label = state.parse_label_identifier()?;
    Ok(Instruction::Jmp { target_label })
}

fn parse_ret<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    // ret.T val  OR ret.void
    state.expect_char('.')?;
    state.skip_whitespace_and_comments();
    // Peek ahead to check for void
    if state.peek_slice(4) == Some("void") {
        state.consume_keyword("void")?;
        Ok(Instruction::Ret { ty: Type::Void, value: None })
    } else {
        let ty = parse_type(state)?;
        let value = parse_value(state)?;
        Ok(Instruction::Ret { ty, value: Some(value) })
    }
}

fn parse_dealloc<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    // dealloc.heap ptr
     state.expect_char('.')?;
     state.consume_keyword("heap")?;
     let ptr = parse_value(state)?;
     Ok(Instruction::Dealloc{ ptr })
}

// Add parser function for print
fn parse_print<'a>(state: &mut ParserState<'a>) -> Result<Instruction<'a>> {
    // Expects: print <value>
    let value = parse_value(state)?;
    Ok(Instruction::Print { value })
}

// Add parse function for zero extend
fn parse_zext<'a>(state: &mut ParserState<'a>, result: Identifier<'a>) -> Result<Instruction<'a>> {
    // Format: zext.i32.i64 %value
    state.expect_char('.')?;
    let source_type_str = state.parse_identifier_str()?;
    let source_type = match source_type_str {
        "i8" => PrimitiveType::I8,
        "i32" => PrimitiveType::I32,
        "i64" => PrimitiveType::I64,
        "bool" => PrimitiveType::Bool,
        _ => return Err(state.error(format!("Invalid source type for zext: {}", source_type_str)))
    };
    
    state.expect_char('.')?;
    let target_type_str = state.parse_identifier_str()?;
    let target_type = match target_type_str {
        "i32" => PrimitiveType::I32,
        "i64" => PrimitiveType::I64,
        _ => return Err(state.error(format!("Invalid target type for zext: {}", target_type_str)))
    };
    
    // Source type must be smaller than target type
    match (source_type, target_type) {
        (PrimitiveType::I8, PrimitiveType::I32) => {},
        (PrimitiveType::I8, PrimitiveType::I64) => {},
        (PrimitiveType::I32, PrimitiveType::I64) => {},
        (PrimitiveType::Bool, PrimitiveType::I32) => {},
        (PrimitiveType::Bool, PrimitiveType::I64) => {},
        _ => return Err(state.error(format!("Invalid type conversion: {} to {}", source_type_str, target_type_str)))
    }
    
    let value = parse_value(state)?;
    Ok(Instruction::ZeroExtend { result, source_type, target_type, value })
}

// Helper to check if an instruction is a terminator
impl<'a> Instruction<'a> {
    fn is_terminator(&self) -> bool {
        matches!(self, Instruction::Ret { .. } | Instruction::Jmp { .. } | Instruction::Br { .. })
    }
} 