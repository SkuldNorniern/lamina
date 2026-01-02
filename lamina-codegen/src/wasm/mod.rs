//! WebAssembly codegen utilities

pub mod abi;
pub mod regalloc;

pub use abi::WasmABI;
pub use regalloc::WasmRegAlloc;

