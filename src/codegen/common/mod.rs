pub mod globals;
pub mod instruction_sel;
pub mod register_alloc;
pub mod stack_layout;
pub mod traits;
pub mod types;
pub mod utils;

// Re-export commonly used items
pub use traits::*;
pub use types::*;
pub use utils::*;
