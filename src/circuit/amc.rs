use std::borrow::Cow;
use std::borrow::Borrow;
use std::time::Instant;
use crate::algebra::{Semiring, Field};
use crate::circuit::weights::Weights;
use crate::circuit::circuit::{Circuit, Node};

/// Algebraic Model Counting
pub(crate) fn amc<T: Semiring>(circuit: &Circuit, weights: &impl Weights<T>) -> Vec<T>
{
    let mut buf: Vec<Cow<T>> = vec![Cow::Owned(T::zero()); circuit.nb_nodes()];
    for (i, node) in circuit.nodes.iter().enumerate() {
        buf[i] = match node {
            Node::Or(children) => Cow::Owned(children.iter().map(|&j| buf[j].borrow()).sum()),
            Node::And(children) => Cow::Owned(children.iter().map(|&j| buf[j].borrow()).product()),
            Node::Leaf(lit) => weights.val(*lit).unwrap_or_else(|| Cow::Owned(T::one())),
        };
    }
    buf.into_iter().map(|x| x.into_owned()).collect()
}

pub(crate) fn amc1<T: Semiring>(circuit: &Circuit, weights: &impl Weights<T>) -> T
{
    let mut buf: Vec<Cow<T>> = vec![Cow::Owned(T::zero()); circuit.nb_nodes()];
    for (i, node) in circuit.nodes.iter().enumerate() {
        buf[i] = match node {
            Node::Or(children) => Cow::Owned(children.iter().map(|&j| buf[j].borrow()).sum()),
            Node::And(children) => Cow::Owned(children.iter().map(|&j| buf[j].borrow()).product()),
            Node::Leaf(lit) => weights.val(*lit).unwrap_or(Cow::Owned(T::one())),
        };
    }
    buf.last().unwrap().clone().into_owned()
}


/// Computes the gradient of the
/// algebraic model count using backpropagation.
pub(crate) fn amc_backprop_alg1<T: Semiring>(circuit: &Circuit, weights: &impl Weights<T>, grad_weights: &mut impl Weights<T>) -> (T, f64)
{
    // First compute the forward pass
    let buf: Vec<T> = amc::<T>(circuit, weights);
    let t1 = Instant::now();
    // Buffer to store gradients of intermediate nodes
    let mut grad_buf: Vec<T> = vec![T::zero(); circuit.nb_nodes()];
    // Initialize the gradient of the output node
    let buf_size = grad_buf.len();
    grad_buf[buf_size-1] = T::one();

    // Backpropagation
    for (i, node) in circuit.nodes.iter().enumerate().rev() {
        match node {
            Node::Or(children) => {
                for &j in children {
                    grad_buf[j] = grad_buf[j].clone() + &grad_buf[i];
                }
            },
            Node::And(children) => {
                let mut r: Vec<T> = vec![T::one(); children.len()];
                let mut t = T::one();
                for (child_ix, &j) in children.iter().enumerate() {
                    r[child_ix] = t.clone();
                    t = t * &buf[j];
                }
                t = T::one();
                for (child_ix, &j) in children.iter().enumerate().rev() {
                    grad_buf[j] = (r[child_ix].clone() * &t * &grad_buf[i]) + &grad_buf[j];
                    t = t * &buf[j];
                }
            },
            Node::Leaf(lit) => {
                let lit_ix = (lit.abs() - 1) as usize;
                if lit_ix < grad_weights.len() {
                    grad_weights.add(*lit, &grad_buf[i]);
                }
            },
        }
    }
    let duration = t1.elapsed().as_secs_f64();
    (buf.last().unwrap().clone(), duration)
}

pub(crate) fn amc_backprop_naive<T: Semiring>(circuit: &Circuit, weights: &impl Weights<T>, grad_weights: &mut impl Weights<T>) -> (T, f64)
{
    // First compute the forward pass
    let buf: Vec<T> = amc::<T>(circuit, weights);
    let t1 = Instant::now();
    // Buffer to store gradients of intermediate nodes
    let mut grad_buf: Vec<T> = vec![T::zero(); circuit.nb_nodes()];
    // Initialize the gradient of the output node
    let buf_size = grad_buf.len();
    grad_buf[buf_size-1] = T::one();

    // Backpropagation
    for (i, node) in circuit.nodes.iter().enumerate().rev() {
        match node {
            Node::Or(children) => {
                for &j in children {
                    grad_buf[j] = grad_buf[j].clone() + &grad_buf[i];
                }
            },
            Node::And(children) => {
                for &j in children.iter() {
                    let r: T = children.iter().flat_map(|&k| if k != j { Some(&buf[k]) } else { None }).product();
                    grad_buf[j] = (r * &grad_buf[i]) + &grad_buf[j];
                }
            },
            Node::Leaf(lit) => {
                let lit_ix = (lit.abs() - 1) as usize;
                if lit_ix < grad_weights.len() {
                    grad_weights.add(*lit, &grad_buf[i]);
                }
            },
        }
    }
    let duration = t1.elapsed().as_secs_f64();
    (buf.last().unwrap().clone(), duration)
}


pub(crate) fn amc_backprop_cancel_order<T: Field>(circuit: &Circuit, weights: &impl Weights<T>, grad_weights: &mut impl Weights<T>) -> (T, f64)
{
    // First compute the forward pass
    let buf: Vec<T> = amc::<T>(circuit, weights);
    let t1 = Instant::now();
    // Buffer to store gradients of intermediate nodes
    let mut grad_buf: Vec<T> = vec![T::zero(); circuit.nb_nodes()];
    // Initialize the gradient of the output node
    let buf_size = grad_buf.len();
    grad_buf[buf_size-1] = T::one();

    // Backpropagation
    for (i, node) in circuit.nodes.iter().enumerate().rev() {
        match node {
            Node::Or(children) => {
                for &j in children {
                    grad_buf[j] = grad_buf[j].clone() + &grad_buf[i];
                }
            },
            Node::And(children) => {
                if buf[i].has_inverse() {
                    for &j in children {
                        grad_buf[j] = (buf[i].clone() / &buf[j]) * &grad_buf[i] + &grad_buf[j];
                    }
                } else {
                    let mut nb_zeros = 0;
                    let mut non_zero_prod = T::one();
                    let mut last_zero: usize = 0;
                    for &j in children {
                        if T::is_zero(&buf[j]) {
                            nb_zeros += 1;
                            last_zero = j;
                            if nb_zeros > 1 {
                                break;
                            }
                        } else {
                            non_zero_prod = non_zero_prod * &buf[j];
                        }
                    }
                    if nb_zeros == 1 {
                        grad_buf[last_zero] = non_zero_prod * &grad_buf[i] + &grad_buf[last_zero];
                    }
                }
            },
            Node::Leaf(lit) => {
                let lit_ix = (lit.abs() - 1) as usize;
                if lit_ix < grad_weights.len() {
                    grad_weights.add(*lit, &grad_buf[i]);
                }
            },
        }
    }
    let duration = t1.elapsed().as_secs_f64();
    (buf.last().unwrap().clone(), duration)
}

pub(crate) fn amc_backprop_order<T: Semiring>(circuit: &Circuit, weights: &impl Weights<T>, grad_weights: &mut impl Weights<T>) -> (T, f64)
{
    // First compute the forward pass
    let buf: Vec<T> = amc::<T>(circuit, weights);
    let t1: Instant = Instant::now();
    // Buffer to store gradients of intermediate nodes
    let mut grad_buf: Vec<T> = vec![T::zero(); circuit.nb_nodes()];
    // Initialize the gradient of the output node
    let buf_size = grad_buf.len();
    grad_buf[buf_size-1] = T::one();

    // Backpropagation
    for (i, node) in circuit.nodes.iter().enumerate().rev() {
        match node {
            Node::Or(children) => {
                for &j in children {
                    grad_buf[j] = grad_buf[j].clone() + &grad_buf[i];
                }
            },
            Node::And(children) => {
                let mut nb_max = 0;
                let mut non_max_prod = T::one();
                let mut last_max: usize = 0;
                for &j in children {
                    if buf[j] == buf[i] {
                        nb_max += 1;
                        last_max = j;
                        if nb_max > 1 {
                            non_max_prod = buf[i].clone();
                            break;
                        }
                    } else {
                        non_max_prod = non_max_prod * &buf[j];
                    }
                }
                grad_buf[last_max] = non_max_prod * &grad_buf[i] + &grad_buf[last_max];
                for &j in children {
                    if j != last_max {
                        grad_buf[j] = buf[i].clone() * &grad_buf[i] + &grad_buf[j];
                    }
                }
            },
            Node::Leaf(lit) => {
                let lit_ix = (lit.abs() - 1) as usize;
                if lit_ix < grad_weights.len() {
                    grad_weights.add(*lit, &grad_buf[i]);
                }
            },
        }
    }
    let duration = t1.elapsed().as_secs_f64();
    (buf.last().unwrap().clone(), duration)
}

pub(crate) fn amc_backprop_cancel<T: Field>(circuit: &Circuit, weights: &impl Weights<T>, grad_weights: &mut impl Weights<T>) -> (T, f64)
{
    // First compute the forward pass
    let buf: Vec<T> = amc::<T>(circuit, weights);
    let t1 = Instant::now();
    // Buffer to store gradients of intermediate nodes
    let mut grad_buf: Vec<T> = vec![T::zero(); circuit.nb_nodes()];
    // Initialize the gradient of the output node
    let buf_size = grad_buf.len();
    grad_buf[buf_size-1] = T::one();

    // Backpropagation
    for (i, node) in circuit.nodes.iter().enumerate().rev() {
        match node {
            Node::Or(children) => {
                for &j in children {
                    grad_buf[j] = grad_buf[j].clone() + &grad_buf[i];
                }
            },
            Node::And(children) => {
                for &j in children {
                    let residual = if buf[i].has_inverse() {
                        buf[i].clone() / &buf[j]
                    } else {
                        children.iter().flat_map(|&k| if k != j { Some(&buf[k]) } else { None }).product()
                    };
                    grad_buf[j] = (residual * &grad_buf[i]) + &grad_buf[j];
                }
            },
            Node::Leaf(lit) => {
                let lit_ix = (lit.abs() - 1) as usize;
                if lit_ix < grad_weights.len() {
                    grad_weights.add(*lit, &grad_buf[i]);
                }
            },
        }
    }
    let duration = t1.elapsed().as_secs_f64();
    (buf.last().unwrap().clone(), duration)
}

