mod error;
mod mapped_mem;

pub use self::error::{Error, Result};
pub use self::mapped_mem::{FutureMemMap, MemMap};