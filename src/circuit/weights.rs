use std::borrow::Cow;
use crate::algebra::{Semiring, Ring};

// Algebraic Weights (or Labels) on the literals of a propositional formula.
pub trait Weights<T: Semiring> {
    fn val(&self, lit: i32) -> Option<Cow<T>>;
    fn len(&self) -> usize;

    fn add(&mut self, lit: i32, val: &T);
}

pub struct PosNegWeights<T: Semiring> {
    pos_weights: Vec<T>,
    neg_weights: Vec<T>,
}


impl<T: Semiring> PosNegWeights<T> {
    pub fn new(len: usize) -> Self {
        PosNegWeights {pos_weights: vec![T::zero(); len], neg_weights: vec![T::zero(); len]}
    }

    pub fn from_vecs<G: Into<T>>(pos_weights: Vec<G>, neg_weights: Vec<G>) -> Self {
        PosNegWeights {
            pos_weights: pos_weights.into_iter().map(|x| x.into()).collect(),
            neg_weights: neg_weights.into_iter().map(|x| x.into()).collect()
            }
    }

    pub fn into_vecs<G: From<T>>(self) -> (Vec<G>, Vec<G>) {
        (self.pos_weights.into_iter().map(|x| x.into()).collect(),
        self.neg_weights.into_iter().map(|x| x.into()).collect())
    }


}

impl<T: Semiring> Weights<T> for PosNegWeights<T>
{
    fn val(&self, lit: i32) -> Option<Cow<T>> {
        let lit_ix = (lit.abs() - 1) as usize;
        let weights = if lit > 0 {&self.pos_weights} else {&self.neg_weights};
        weights.get(lit_ix).map(|x| Cow::Borrowed(x))
    }

    fn len(&self) -> usize {
        self.pos_weights.len()
    }

    fn add(&mut self, lit: i32, val: &T) {
        let lit_ix = (lit.abs() - 1) as usize;
        if lit > 0 {
            self.pos_weights[lit_ix] = self.pos_weights[lit_ix].clone() + val;
        } else {
            self.neg_weights[lit_ix] = self.pos_weights[lit_ix].clone() + val;
        }
    }
}

pub struct PosOnlyWeights<T: Ring> {
    weights: Vec<T>,
}

impl <T: Ring> PosOnlyWeights<T> {
    pub fn new(len: usize) -> Self {
        PosOnlyWeights {weights: vec![T::zero(); len]}
    }

    pub fn into_vec<G: From<T>>(self) -> Vec<G> {
        self.weights.into_iter().map(|x| x.into()).collect()
    }

    pub fn from_vec<G: Into<T>>(weights: Vec<G>) -> Self {
        PosOnlyWeights { weights: weights.into_iter().map(|x| x.into()).collect() }
    }
}

impl<T: Ring> Weights<T> for PosOnlyWeights<T>
{
    fn val(&self, lit: i32) -> Option<Cow<T>> {
        let lit_ix = (lit.abs() - 1) as usize;
        let val = self.weights.get(lit_ix);
        if lit > 0 {
            val.map(|x| Cow::Borrowed(x))
        } else {
            val.map(|x| Cow::Owned(x.negate()))
        }
    }

    fn len(&self) -> usize {
        self.weights.len()
    }

    fn add(&mut self, lit: i32, val: &T) {
        let lit_ix = (lit.abs() - 1) as usize;
        if lit > 0 {
            self.weights[lit_ix] = self.weights[lit_ix].clone() + val;
        } else {
            self.weights[lit_ix] = self.weights[lit_ix].clone() - val;
        }
    }
}