use std::ops::{Add, Mul, Sub, Div};
use std::iter::{Sum, Product};
use crate::algebra::{Semiring, Ring, Field, LogSemiring};
use crate::algebra::utils::ln_add_exp;


// SEMIRING OPERATIONS

impl<'a> Add<&'a Self> for LogSemiring {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self(ln_add_exp(self.0, rhs.0))
    }
}


impl<'a> Sum<&'a Self> for LogSemiring {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Mul<&'a Self> for LogSemiring {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a> Product<&'a Self> for LogSemiring {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Semiring for LogSemiring {
    fn zero() -> Self {
        Self(f32::NEG_INFINITY)
    }

    fn one() -> Self {
        Self(0.)
    }

    const IDEMPOTENT: bool = false;

    fn has_inverse(&self) -> bool {
        self.0 != f32::NEG_INFINITY
    }
}

// RING OPERATIONS

impl Ring for LogSemiring {
    fn negate(&self) -> Self {
        let result = (-self.0.exp()).ln_1p();
        if result.is_nan() {
            panic!("ln_neg({}) = NaN", self.0);
        }
        Self(result)
    }
}

impl<'a> Sub<&'a Self> for LogSemiring {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        let result = (self.0.exp() - rhs.0.exp()).ln();
        if result.is_nan() {
            panic!("ln_sub_exp({}, {}) = NaN", self.0, rhs.0);
        }
        Self(result) // TODO?
    }
}



// FIELD OPERATIONS

impl Field for LogSemiring {}

impl<'a> Div<&'a Self> for LogSemiring {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        if (self.0 - rhs.0).is_nan() {
            panic!("ln_div({}, {}) = NaN", self.0, rhs.0);
        }
        Self(self.0 - rhs.0)
    }
}

// UTILITY

impl From<f32> for LogSemiring {
    fn from(x: f32) -> Self {
        Self(x)
    }
}

impl From<LogSemiring> for f32 {
    fn from(x: LogSemiring) -> Self {
        x.0
    }
}
