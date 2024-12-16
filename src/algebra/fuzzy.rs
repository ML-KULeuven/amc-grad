use std::ops::{Add, Mul, Sub, Div};
use std::iter::{Sum, Product};
use crate::algebra::{Semiring, Ring, Field, FuzzySemiring};


// SEMIRING OPERATIONS

impl<'a> Add<&'a Self> for FuzzySemiring {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(self.0.max(rhs.0))
    }
}


impl<'a> Sum<&'a Self> for FuzzySemiring {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc + x)
    }
}

impl<'a> Mul<&'a Self> for FuzzySemiring {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl<'a> Product<&'a Self> for FuzzySemiring {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Semiring for FuzzySemiring {
    fn zero() -> Self {
        Self(0.)
    }

    fn one() -> Self {
        Self(1.)
    }

    const IDEMPOTENT: bool = true;

    fn has_inverse(&self) -> bool {
        false
    }
}

// RING OPERATIONS

impl Ring for FuzzySemiring {
    fn negate(&self) -> Self {
        Self(1.-self.0)
    }
}

impl<'a> Sub<&'a Self> for FuzzySemiring {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        panic!("Fuzzy sub");
    }
}

impl Field for FuzzySemiring {}

impl<'a> Div<&'a Self> for FuzzySemiring {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        panic!("Fuzzy div");
    }
}


// UTILITY

impl From<f32> for FuzzySemiring {
    fn from(x: f32) -> Self {
        Self(x)
    }
}

impl From<FuzzySemiring> for f32 {
    fn from(x: FuzzySemiring) -> Self {
        x.0
    }
}
