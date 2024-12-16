use crate::algebra::{Semiring, Ring, Field};

impl Semiring for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    const IDEMPOTENT: bool = false;

    fn has_inverse(&self) -> bool {
        *self != 0.0
    }
}

impl Ring for f64 {}

impl Field for f64 {}


impl Semiring for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    const IDEMPOTENT: bool = false;
}

impl Ring for f32 {}

impl Field for f32 {}
