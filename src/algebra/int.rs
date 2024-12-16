use crate::algebra::{Semiring, Ring, Field};

impl Semiring for i32 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    const IDEMPOTENT: bool = false;

    fn has_inverse(&self) -> bool {
        *self != 0
    }
}

impl Ring for i32 {}


impl Field for i32 {}
