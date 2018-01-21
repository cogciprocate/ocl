use std::str::SplitWhitespace;

/// Extensions of a platform.
#[derive(Debug, Clone)]
pub struct Extensions {
    pub(crate) inner: String,
}

impl Extensions {
    /// Iterate over every extensions,
    /// represented by an `str` that doesn't contains space.
    pub fn iter(&self) -> SplitWhitespace {
        self.inner.split_whitespace()
    }
}
