use std::rc::Rc;
use pyo3::{pyclass, pymethods, PyResult};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyRuntimeError;
use crate::circuit::circuit::Circuit;
use crate::circuit::compile::{compile, to_dimacs};
use crate::circuit::tseitin::tseitin_transform;


#[derive(Debug, Hash, PartialEq, Eq)]
pub enum RNode {
    Prod(Vec<Rc<RNode>>),
    Sum(Vec<Rc<RNode>>),
    Neg(Rc<RNode>),
    Val(i32),
    One,
    Zero,
}

impl RNode {
    pub fn is_leaf(&self) -> bool {
        match self {
            RNode::Val(_) => true,
            RNode::Zero => true,
            RNode::One => true,
            _ => false,
        }
    }

    pub fn is_true(&self) -> bool {
        match self {
            RNode::One => true,
            _ => false,
        }
    }

    pub fn is_false(&self) -> bool {
        match self {
            RNode::Zero => true,
            _ => false,
        }
    }

    pub fn is_clause(&self) -> bool {
        match self {
            RNode::Sum(vs) => vs.iter().all(|v| v.is_leaf()),
            _ => false,
        }
    }

    pub fn is_term(&self) -> bool {
        match self {
            RNode::Prod(vs) => vs.iter().all(|v| v.is_leaf()),
            _ => false,
        }
    }

    pub fn is_cnf(&self) -> bool {
        match self {
            RNode::Prod(vs) => vs.iter().all(|v| v.is_clause()),
            _ => false,
        }
    }

    pub fn is_dnf(&self) -> bool {
        match self {
            RNode::Sum(vs) => vs.iter().all(|v| v.is_term()),
            _ => false,
        }
    }

    pub fn nb_vars(&self) -> u32 {
        match self {
            RNode::Val(v) => v.abs() as u32,
            RNode::Sum(vs) | RNode::Prod(vs) => vs.iter()
                .map(|v| v.nb_vars()).max().unwrap_or(0),
            RNode::Neg(v) => v.nb_vars(),
            _ => 0,
        }
    }

    pub fn children(&self) -> Vec<Rc<RNode>> {
        match self {
            RNode::Sum(vs) => vs.iter().map(|v| Rc::clone(v)).collect(),
            RNode::Prod(vs) => vs.iter().map(|v| Rc::clone(v)).collect(),
            RNode::Neg(v) => vec![Rc::clone(v)],
            _ => vec![],
        }
    }

    pub fn value(&self) -> i32 {
        match self {
            RNode::Val(v) => *v,
            _ => panic!("Not a value"),
        }
    }

    pub fn condition(self: Rc<Self>, lits: &Vec<i32>) -> Rc<RNode> {
        condition(self, lits)
    }

    pub fn simplify(self: Rc<Self>) -> Rc<RNode> {
        // Simplify the circuit by removing redundant nodes.
        match *self {
            RNode::Sum(ref vs) => simplify_or(&vs),
            RNode::Prod(ref vs) => simplify_and(&vs),
            RNode::Neg(ref v) => RNode::Neg(v.clone().simplify()).into(),
            _ => self,
        }
    }

    pub fn negate(self: Rc<Self>) -> Rc<RNode> {
        match *self {
            RNode::Val(ref v) => RNode::Val(-v).into(),
            RNode::Neg(ref v) => Rc::clone(&v),
            RNode::One => RNode::Zero.into(),
            RNode::Zero => RNode::One.into(),
            _ => RNode::Neg(self.clone()).into(),
        }
    }
}


fn simplify_or(children: &Vec<Rc<RNode>>) -> Rc<RNode> {
    let children: Vec<Rc<RNode>> = children.iter()
        .map(|c| c.clone().simplify())
        .filter(|c| !c.is_false())
        .collect();
    if children.iter().any(|c| (**c).is_true()) {
        return RNode::One.into();
    }
    match children.len() {
        0 => RNode::Zero.into(),
        1 => Rc::clone(&children[0]),
        _ => RNode::Sum(children).into(),
    }
}


fn simplify_and(children: &Vec<Rc<RNode>>) -> Rc<RNode> {
    let children: Vec<Rc<RNode>> = children.iter()
        .map(|c| c.clone().simplify())
        .filter(|c| !c.is_true())
        .collect();
    if children.iter().any(|c| c.is_false()) {
        return RNode::Zero.into();
    }

    match children.len() {
        0 => RNode::One.into(),
        1 => Rc::clone(&children[0]),
        _ => RNode::Prod(children).into(),
    }
}

/// Condition the circuit on a literal.
/// Returns a new circuit and a boolean indicating if the circuit was modified.
fn condition(node: Rc<RNode>, lits: &Vec<i32>) -> Rc<RNode> {
    match *node {
        RNode::Val(ref v) if lits.contains(v)  => RNode::One.into(),
        RNode::Val(ref v) if lits.contains(&-*v) => RNode::Zero.into(),
        RNode::Sum(ref vs) =>
            RNode::Sum(vs.iter().map(|v| v.clone().condition(lits)).collect()).into(),
        RNode::Prod(ref vs) =>
            RNode::Prod(vs.iter().map(|v| v.clone().condition(lits)).collect()).into(),
        RNode::Neg(ref v) =>
            RNode::Neg(v.clone().condition(lits)).into(),
        _ => node,
    }
}


#[pyclass(unsendable)]
pub struct RCircuit {
    node: Rc<RNode>,
}

impl From<RNode> for RCircuit {
    fn from(node: RNode) -> Self {
        RCircuit { node: Rc::new(node) }
    }
}


#[pymethods]
impl RCircuit {
    fn __str__(&self) -> String {
        format!("{:?}", self.node)
    }

    fn nb_vars(&self) -> u32 {
        self.node.nb_vars()
    }

    fn __and__(&self, other: &RCircuit) -> RCircuit {
        let children = vec![self.node.clone(), other.node.clone()];
        RNode::Prod(children).into()
    }

    fn __or__(&self, other: &RCircuit) -> RCircuit {
        let children = vec![self.node.clone(), other.node.clone()];
        RNode::Sum(children).into()
    }

    fn __invert__(&self) -> RCircuit {
        RCircuit{node: self.node.clone().negate()}
    }

    fn __int__(&self) -> i32 {
        self.node.value()
    }

    fn is_cnf(&self) -> bool {
        self.node.is_cnf()
    }

    fn is_dnf(&self) -> bool {
        self.node.is_dnf()
    }

    fn __richcmp__(&self, other: &RCircuit, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.node == other.node),
            CompareOp::Ne => Ok(self.node != other.node),
            // other options are not implemented
            _ => Err(PyRuntimeError::new_err("Can't compare circuits")),
        }
    }

    fn simplify(&self) -> RCircuit {
        RCircuit { node: self.node.clone().simplify() }
    }

    fn condition(&self, lits: Vec<i32>) -> RCircuit {
        RCircuit{node: self.node.clone().condition(&lits).into()}
    }

    fn tseitin(&self, nb_vars: u32) -> RCircuit {
        RCircuit{node: tseitin_transform(self.node.clone(), nb_vars)}
    }

    fn compile(&self, nb_vars: u32) -> PyResult<Circuit> {
        Ok(compile(self.node.clone(), nb_vars, "d4".to_string())?)
    }

    fn clauses(&self) -> Vec<Vec<i32>> {
        let nb_vars = self.node.nb_vars();
        self.tseitin(nb_vars).node.children().iter().map(|c| {
            c.children().iter().map(|l| {l.value()}).collect()
        }).collect()
    }

    fn to_dimacs(&self) -> PyResult<String> {
        if self.node.is_cnf() {
            Ok(to_dimacs(self.node.clone()))
        } else {
            Err(PyRuntimeError::new_err("Not a CNF"))
        }
    }
}
