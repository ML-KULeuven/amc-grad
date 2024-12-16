use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::iter::{Sum, Product};
use std::rc::Rc;
use crate::algebra::{Semiring, CircuitSemiring};
use crate::circuit::circuit::Node;


// SEMIRING OPERATIONS

fn get_context(lhs: Option<Rc<RefCell<Vec<Node>>>>, rhs: &Option<Rc<RefCell<Vec<Node>>>>) -> Rc<RefCell<Vec<Node>>> {
    match (lhs, rhs) {
        (Some(lhs), None) => lhs,
        (None, Some(rhs)) => rhs.clone(),
        (Some(lhs), Some(rhs)) => {
            if lhs.as_ptr() == rhs.as_ptr() {
                lhs
            } else {
                panic!("Context mismatch")
            }
        }
        _ => panic!("No context")
    }
}

impl<'a> Add<&'a Self> for CircuitSemiring {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        let context = get_context(self.context, &rhs.context);
        let new_node = Node::Or(vec![self.ix, rhs.ix]);
        let ix: usize = context.borrow().len();
        context.borrow_mut().push(new_node);
        Self {ix, context: Some(context)}
    }
}

impl<'a> Sum<&'a Self> for CircuitSemiring {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let (context, children) = iter.fold((None, Vec::new()), |(acc_context, mut acc_children), x| {
            let context = get_context(acc_context, &x.context);
            acc_children.push(x.ix);
            (Some(context), acc_children)
        });
        if children.len() == 0 {
            return Self::zero();
        }
        let new_node = Node::Or(children);
        let ix: usize = context.as_ref().unwrap().borrow().len();
        context.as_ref().unwrap().borrow_mut().push(new_node);
        Self {ix, context}
    }
}

impl<'a> Mul<&'a Self> for CircuitSemiring {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        let context = get_context(self.context, &rhs.context);
        let new_node = Node::And(vec![self.ix, rhs.ix]);
        let new_ix: usize = context.borrow().len();
        context.borrow_mut().push(new_node);
        Self {ix: new_ix, context: Some(context)}
    }
}

impl<'a> Product<&'a Self> for CircuitSemiring {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let (context, children) = iter.fold((None, Vec::new()), |(acc_context, mut acc_children), x| {
            let context = get_context(acc_context, &x.context);
            acc_children.push(x.ix);
            (Some(context), acc_children)
        });
        if children.len() == 0 {
            return Self::one();
        }
        let new_node = Node::And(children);
        let ix: usize = context.as_ref().unwrap().borrow().len();
        context.as_ref().unwrap().borrow_mut().push(new_node);
        Self {ix, context}

    }
}

impl Semiring for CircuitSemiring {
    fn zero() -> Self {
        Self {ix: 0, context: None}
    }

    fn one() -> Self {
        Self {ix: 1, context: None}
    }

    const IDEMPOTENT: bool = false;
}

