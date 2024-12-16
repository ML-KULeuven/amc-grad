use std::rc::Rc;
use std::iter::once;
use crate::circuit::rcircuit::RNode;


type Clause = Vec<i32>;

/// Tseitin transformation
/// See https://en.wikipedia.org/wiki/Tseytin_transformation
fn _цейтин(node: &RNode, next_var: &mut i32, clauses: &mut Vec<Clause>) -> i32 {
    match node {
        RNode::Sum(children) => {
            let vars: Vec<i32> = children.iter()
                .map(|child| _цейтин(child, next_var, clauses)).collect();
            *next_var += 1;
            let new_var = *next_var;
            for var in &vars {
                clauses.push(vec![new_var, -*var]);
            }
            let long_clause = vars.into_iter()
                .chain(once(-new_var)).collect();
            clauses.push(long_clause);
            new_var
        }
        RNode::Prod(children) => {
            let vars: Vec<i32> = children.iter()
                .map(|child| _цейтин(child, next_var, clauses)).collect();
            *next_var += 1;
            let new_var = *next_var;
            for var in &vars {
                clauses.push(vec![-new_var, *var]);
            }
            let long_clause: Clause = vars.iter()
                .map(|v| -v).chain(once(new_var)).collect();
            clauses.push(long_clause);
            new_var
        }
        RNode::Neg(child) => {
            let child_var: i32 = _цейтин(child, next_var, clauses);
            *next_var += 1;
            let new_var = *next_var;
            clauses.push(vec![new_var, child_var]);
            clauses.push(vec![-new_var, -child_var]);
            new_var
        }
        RNode::Val(value) => *value,
        _ => panic!("Unexpected node"),
    }
}


pub fn tseitin_transform(node: Rc<RNode>, nb_vars: u32) -> Rc<RNode> {
    if node.is_cnf() {
        return node;
    }

    let mut next_var = nb_vars as i32;
    let mut clauses = vec![];
    let new_var = _цейтин(&node, &mut next_var, &mut clauses);
    clauses.push(vec![new_var]);
    let clauses = clauses.iter()
        .map(|c| RNode::Sum(c.iter().map(|v| RNode::Val(*v).into()).collect()).into())
        .collect();
    RNode::Prod(clauses).into()
}
