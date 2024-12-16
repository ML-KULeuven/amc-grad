use pyo3::pyfunction;
use crate::circuit::rcircuit::{RCircuit, RNode};

mod amc;
pub mod circuit;
mod compile;
pub mod parser;
pub mod rcircuit;
mod weights;
mod tseitin;


#[pyfunction]
pub fn lit(value: i32) -> RCircuit {
    RNode::Val(value).into()
}

#[pyfunction]
pub fn zero() -> RCircuit {
    RNode::Zero.into()
}

#[pyfunction]
pub fn one() -> RCircuit {
    RNode::One.into()
}