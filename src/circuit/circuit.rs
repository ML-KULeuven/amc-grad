use std::cell::RefCell;
use std::rc::Rc;
use num_traits::Float;
use pyo3::{pyclass, pymethods, Python};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArrayDyn, ToPyArray};
use numpy::ndarray::{Array1, Array2, Axis, stack};
use crate::algebra::*;
use crate::circuit::amc::{amc1, amc_backprop_alg1, amc_backprop_cancel, amc_backprop_cancel_order, amc_backprop_naive, amc_backprop_order};
use crate::circuit::weights::{PosNegWeights, PosOnlyWeights};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Node {
    Or(Vec<usize>),
    And(Vec<usize>),
    Leaf(i32),
}


impl Node {
    pub fn add_child(&mut self, child: usize) {
        match self {
            Node::Or(children) => children.push(child),
            Node::And(children) => children.push(child),
            _ => panic!("Can only add child to Or/And node"),
        }
    }
}


#[pyclass]
pub struct Circuit {
    pub nodes: Vec<Node>, // Assumes nodes are topologically sorted
}

impl Circuit {
    fn _amc<W, T>(&self, weights: Vec<W>) -> T
        where T: Ring + From<W>
    {
        let weights: PosOnlyWeights<T> = PosOnlyWeights::from_vec(weights);
        amc1::<T>(&self, &weights)
    }

    fn _amc2<W, T>(&self, pos_weights: Vec<W>, neg_weights: Vec<W>) -> T
        where T: Ring + From<W>
    {
        let weights: PosNegWeights<T> = PosNegWeights::from_vecs(pos_weights, neg_weights);
        amc1::<T>(&self, &weights)
    }

    fn _amc_grad<W, T>(&self, weights: Vec<W>) -> (W, Vec<W>, f64)
        where W: From<T>, T: Field + From<W>
    {
        let mut grad_weights = PosOnlyWeights::new(weights.len());
        let weights = PosOnlyWeights::from_vec(weights);
        let (result, duration) = amc_backprop_cancel_order::<T>(&self, &weights, &mut grad_weights);
        (result.into(), grad_weights.into_vec(), duration)
    }

    fn _amc_grad2<W, T>(&self, pos_weights: Vec<W>, neg_weights: Vec<W>) -> (W, Vec<W>, Vec<W>, f64)
        where W: From<T>, T: Field + From<W>
    {
        let mut grad_weights = PosNegWeights::new(pos_weights.len());
        let weights = PosNegWeights::from_vecs(pos_weights, neg_weights);
        let (result, duration) = amc_backprop_cancel_order::<T>(&self, &weights, &mut grad_weights);
        let (pos_grad_weights, neg_grad_weights) = grad_weights.into_vecs();
        (result.into(), pos_grad_weights, neg_grad_weights, duration)
    }

}


#[pymethods]
impl Circuit {
    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn nb_vars(&self) -> u32 {
        self.nodes.iter().flat_map(|node| match node {
            Node::Leaf(v) => Some(v.abs() as u32),
            _ => None,
        }).max().unwrap_or(0)
    }

    fn _bool_grad(&self, pos_weights: Vec<bool>, neg_weights: Vec<bool>) -> (bool, Vec<bool>, Vec<bool>, f64) {
        self._amc_grad2::<bool, BoolSemiring>(pos_weights, neg_weights)
    }


    fn mc(&self) -> i32 {
        self._amc::<i32, i32>(vec![])
    }

    fn wmc(&self, weights: Vec<f64>) -> f64 {
        self._amc::<f64, f64>(weights)
    }

    fn _wmc_grad(&self, weights: Vec<f64>) -> (f64, Vec<f64>, f64) {
        self._amc_grad::<f64, f64>(weights)
    }

    fn _log_grad(&self, pos_weights: Vec<f32>, neg_weights: Vec<f32>) -> (f32, Vec<f32>, Vec<f32>, f64) {
        self._amc_grad2::<f32, LogSemiring>(pos_weights, neg_weights)
    }

    fn _fuzzy_grad(&self, pos_weights: Vec<f32>, neg_weights: Vec<f32>) -> (f32, Vec<f32>, Vec<f32>, f64) {
        self._amc_grad2::<f32, FuzzySemiring>(pos_weights, neg_weights)
    }

    fn circuit_transform(&self) -> Circuit {
        let nb_vars = self.nb_vars() + 1;
        let mut weights: Vec<Node> = vec![Node::Or(vec![]), Node::And(vec![])];
        for i in 1..nb_vars {
            weights.push(Node::Leaf(i as i32));
            weights.push(Node::Leaf(-(i as i32)));
        }
        let weights: Rc<RefCell<Vec<Node>>> = Rc::new(RefCell::new(weights));
        let pos_lit_map: Vec<CircuitSemiring> = (1..nb_vars).map(|i| CircuitSemiring {ix: 2*i as usize, context: Some(weights.clone())}).collect();
        let neg_lit_map: Vec<CircuitSemiring> = (1..nb_vars).map(|i| CircuitSemiring {ix: (2*i + 1) as usize, context: Some(weights.clone())}).collect();
        let amc_weights: PosNegWeights<CircuitSemiring> = PosNegWeights::from_vecs(pos_lit_map, neg_lit_map);
        let amc_result = amc1::<CircuitSemiring>(&self, &amc_weights);
        let result = amc_result.context.clone().unwrap().replace(vec![]);
        Circuit {nodes: result}
    }

    fn log_wmc(&self, weights: Vec<f32>) -> f32 {
        self._amc::<f32, LogSemiring>(weights).0
    }

    fn depth_width(&self) -> (usize, Vec<usize>) {
        let mut depths: Vec<usize> = vec![0; self.nodes.len()];
        let mut widths: Vec<usize> = vec![];
        for (i, node) in self.nodes.iter().enumerate() {
            depths[i] = match node {
                Node::Or(children) => {
                    children.iter().map(|&j| depths[j]).max().unwrap_or(0) + 1
                },
                Node::And(children) => {
                    children.iter().map(|&j| depths[j]).max().unwrap_or(0) + 1
                },
                _ => 1,
            };
            if depths[i] >= widths.len() {
                widths.resize(depths[i] + 1, 0);
            }
            widths[depths[i]] += 1;
        }
        let max_depth = *depths.iter().max().unwrap();
        (max_depth, widths)
    }

    fn node_width(&self) -> usize {
        self.nodes.iter().map(|node| match node {
            Node::Or(children) => children.len(),
            Node::And(children) => children.len(),
            _ => 0,
        }).max().unwrap()
    }
}

fn from_pyarray<T>(v: PyReadonlyArrayDyn<T>) -> Vec<Array1<T>>
where T: Float + Element
{
    v.as_array()
        .rows().into_iter().map(|x| x.to_owned())
        .collect()
}

fn to_pyarray<T>(py: Python, v: Vec<Array1<T>>) -> &PyArray2<T>
where T: Float + Element
{
    let size: usize = v.iter().map(|x| x.len()).max().unwrap();
    let v: Vec<Array1<T>> = v.into_iter().map(|x| {
        if x.len() < size {
            // Zero/one semiring elements need to be padded
            if T::is_zero(&x[0]) {Array1::zeros(size) } else {Array1::ones(size)}
        } else {
            x
        }
    }).collect();
    let v = v.iter().map(|arr| arr.view()).collect::<Vec<_>>();
    let v: Array2<T> = stack(Axis(0), &v).unwrap();
    v.to_pyarray(py)
}