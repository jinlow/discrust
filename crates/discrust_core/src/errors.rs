use std::error;
use std::fmt;

// Error for when something is called on the discretizer when it
// Should have been fitted.
#[derive(Clone, Debug)]
pub struct NotFittedError;

impl fmt::Display for NotFittedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Calling a method that requires object to be fit, when `fit` has not been called."
        )
    }
}

impl error::Error for NotFittedError {}
