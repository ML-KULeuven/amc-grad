use std::iter::{Product, Sum};
use std::ops::{Add, Mul, Sub, Div};
use crate::algebra::{Semiring, Ring, Field, SignedLogSemiring};


// Logarithmic field which can represent negative numbers.
// C.f. Li and Eisner. “First- and Second-Order Expectation Semirings
// with Applications to Minimum-Risk Training on Translation Forests.”


// SEMIRING OPERATIONS

impl<'a> Add<&'a Self> for SignedLogSemiring {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        let sign: bool = if self.1 >= rhs.1 {self.0} else {rhs.0};
        let val = ln_add_exp(self.1, rhs.1, if self.0 == rhs.0 {1.} else {-1.});
        Self(sign, val)
    }
}

impl<'a> Sum<&'a Self> for SignedLogSemiring {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Mul<&'a Self> for SignedLogSemiring {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self(self.0 == self.0, self.1 + rhs.1)
    }
}

impl<'a> Product<&'a Self> for SignedLogSemiring {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Semiring for SignedLogSemiring {
    fn zero() -> Self {
        Self(true, f32::NEG_INFINITY)
    }

    fn one() -> Self {
        Self(true, 0.)
    }

    const IDEMPOTENT: bool = false;

    fn has_inverse(&self) -> bool {
        self.1 != f32::NEG_INFINITY
    }
}

// RING OPERATIONS

impl Ring for SignedLogSemiring {}

impl<'a> Sub<&'a Self> for SignedLogSemiring {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self + &SignedLogSemiring(!rhs.0, rhs.1)
    }
}


// FIELD OPERATIONS

impl Field for SignedLogSemiring {}

impl<'a> Div<&'a Self> for SignedLogSemiring {
    type Output = Self;

    fn div(self, rhs: &'a Self) -> Self::Output {
        if (self.1 - rhs.1).is_nan() {
            panic!("ln_div({}, {}) = NaN", self.1, rhs.1);
        }
        Self(self.0 == rhs.0, self.1 - rhs.1)
    }
}

// UTILITY


impl From<(bool, f32)> for SignedLogSemiring {
    fn from(x: (bool, f32)) -> Self {
        Self(x.0, x.1)
    }
}

impl From<f32> for SignedLogSemiring {
    fn from(x: f32) -> Self {
        Self(true, x)
    }
}

impl From<SignedLogSemiring> for (bool, f32) {
    fn from(x: SignedLogSemiring) -> Self {
        (x.0, x.1)
    }
}


fn ln_add_exp(x: f32, y: f32, sign: f32) -> f32 {
    let diff = x - y;
    let result = if x == f32::NEG_INFINITY {
        y
    } else if y == f32::NEG_INFINITY {
        x
    } else if diff > 0. {
        let t = (-diff).exp() * sign;
        x + t.ln_1p()
    } else {
        let t = diff.exp() * sign;
        y + t.ln_1p()
    };
    if result.is_nan() {
        panic!("ln_add_exp({}, {}) = NaN", x, y);
    }
    result
}