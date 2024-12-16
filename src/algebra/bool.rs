use std::ops::{Add, Mul, Sub, Div};
use std::iter::{Sum, Product};
use crate::algebra::{Semiring, Ring, Field, BoolSemiring};
use crate::algebra::utils::ln_add_exp;


// SEMIRING OPERATIONS

impl<'a> Add<&'a Self> for BoolSemiring {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}


impl<'a> Sum<&'a Self> for BoolSemiring {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).any(|x| x))
    }
}

impl<'a> Mul<&'a Self> for BoolSemiring {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<'a> Product<&'a Self> for BoolSemiring {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).all(|x| x))
    }
}

impl Semiring for BoolSemiring {
    fn zero() -> Self {
        Self(false)
    }

    fn one() -> Self {
        Self(true)
    }

    const IDEMPOTENT: bool = false;

    fn has_inverse(&self) -> bool {
        self.0 == true
    }
}

// RING OPERATIONS

impl Ring for BoolSemiring {
    fn negate(&self) -> Self {
        Self(!self.0)
    }
}

impl<'a> Sub<&'a Self> for BoolSemiring {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        panic!("Subtraction not defined for BoolSemiring")
    }
}



// FIELD OPERATIONS

impl Field for BoolSemiring {}

impl<'a> Div<&'a Self> for BoolSemiring {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        if self.0 && rhs.0 {
            Self(true)
        } else {
            panic!("Division not defined for BoolSemiring")
        }
    }
}

// UTILITY

impl From<bool> for BoolSemiring {
    fn from(x: bool) -> Self {
        Self(x)
    }
}

impl From<BoolSemiring> for bool {
    fn from(x: BoolSemiring) -> bool {
        x.0
    }
}
