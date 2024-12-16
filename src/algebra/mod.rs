mod float;
mod log;
mod int;
mod circuit;
mod signed_log;
mod utils;
mod bool;
mod fuzzy;

use std::cell::RefCell;
use std::ops::{Add, Mul, Sub, Div};
use std::iter::{Sum, Product};
use std::fmt::Debug;
use std::rc::Rc;
use numpy::ndarray::Array1;
use crate::circuit::circuit::Node;
use crate::circuit::rcircuit::RNode;

pub trait Semiring:
    for<'a> Add<&'a Self, Output = Self> +
    for <'a> Sum<&'a Self> +           // Addition traits
    for <'a> Mul<&'a Self, Output = Self> +
    for<'a> Product<&'a Self> +        // Multiplication traits
    Sized + Clone + PartialEq + Debug  // Utility traits
{
    fn zero() -> Self;

    fn one() -> Self;

    // Whether the sum semiring operation is idempotent (a + a = a).
    const IDEMPOTENT: bool;

    fn has_inverse(&self) -> bool {
        false
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

pub trait Ring: Semiring + for <'a> Sub<&'a Self, Output = Self> {
    fn negate(&self) -> Self {
        Self::one() - &self
    }
}

pub trait Field: Ring + for <'a> Div<&'a Self, Output = Self> {}

#[derive(Debug, Clone, PartialEq)]
pub struct BoolSemiring (pub bool);

#[derive(Debug, Clone, PartialEq)]
pub struct LogSemiring (pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct FuzzySemiring(pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct SignedLogSemiring (pub bool, pub f32);


#[derive(Debug, Clone, PartialEq)]
pub struct LogNdarraySemiring(pub Array1<f32>);

// Higher-order semirings
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSemiring(pub Rc<RNode>);

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitSemiring {
    pub ix: usize,
    pub context: Option<Rc<RefCell<Vec<Node>>>>, // not proud of this one
}