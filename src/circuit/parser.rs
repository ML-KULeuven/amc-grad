use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use pyo3::pyfunction;
use crate::circuit::circuit::{Circuit, Node};
use crate::circuit::rcircuit::{RCircuit, RNode};

/// Parse a compiled d4 circuit from a file.
#[pyfunction]
pub fn load_d4(filename: &str) -> io::Result<Circuit> {

    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut nodes: Vec<Node> = Vec::new();
    let mut line = String::new();
    let mut node_map: Vec<Option<usize>> = vec![None]; // d4 node index -> circuit index
    let mut lits_map: Vec<Option<usize>> = Vec::new(); // literal -> circuit index
    let mut cnode_cache: HashMap<usize, Node> = HashMap::new(); // d4 node index -> node

    fn to_node_ix(node_map: &mut Vec<Option<usize>>, nodes: &mut Vec<Node>, cnode_cache: &mut HashMap<usize, Node>, d4_ix: usize) -> usize {
        match node_map[d4_ix] {
            Some(ix) => ix,
            None => {
                // Child node must be ready, as it's getting used.
                node_map[d4_ix] = Some(nodes.len());
                nodes.push(cnode_cache.remove(&d4_ix).unwrap());
                nodes.len()-1
            }
        }
    }

    fn to_lit_ix(lits_map: &mut Vec<Option<usize>>, nodes: &mut Vec<Node>, lit: i32) -> usize {
        let lit_ix = (2 * lit.abs() + (if lit > 0 {-2} else {-1})) as usize;
        if lit_ix >= lits_map.len() {
            lits_map.resize(lit_ix+1, None);
        }
        match lits_map[lit_ix] {
            Some(ix) => ix,
            None => {
                nodes.push(Node::Leaf(lit));
                lits_map[lit_ix] = Some(nodes.len()-1);
                nodes.len()-1
            }
        }
    }

    while reader.read_line(&mut line)? != 0 {
        if line.is_empty() {
            line.clear();
            continue;
        }
        let first_char = line.chars().nth(0).unwrap();

        if first_char == 'o' || first_char == 'a' || first_char == 'f' || first_char == 't' {
            // Introduction of a new node: add it to the cache.
            let node_index = line.split_ascii_whitespace()
                .nth(1).unwrap().parse().unwrap();
            let node = match first_char {
                'a' => Node::And(vec![]),
                't' => Node::And(vec![]),
                _ => Node::Or(vec![]),
            };
            cnode_cache.insert(node_index, node);
            node_map.push(None);
        } else {
            let mut iter = line.split_ascii_whitespace();
            let parent_ix = iter.next().unwrap().parse::<usize>().unwrap();
            let child_ix = iter.next().unwrap().parse::<usize>().unwrap();
            let lits: Vec<i32> = iter.take_while(|x| *x != "0")
                .map(|x| x.parse::<i32>().unwrap()).collect();

            let child_ix = to_node_ix(&mut node_map, &mut nodes, &mut cnode_cache, child_ix);
            let source_ix = if lits.is_empty() {
                child_ix
            } else {
                let mut lits_ix: Vec<usize> = lits.iter()
                    .map(|lit| to_lit_ix(&mut lits_map, &mut nodes, *lit)).collect();
                lits_ix.push(child_ix);
                nodes.push(Node::And(lits_ix));
                nodes.len()-1
            };
            cnode_cache.get_mut(&parent_ix).unwrap().add_child(source_ix);
        }
        line.clear();
    }

    // Finalize root node
    nodes.push(cnode_cache.remove(&1).unwrap());
    if !cnode_cache.is_empty() {
        panic!("Dangling nodes detected!");
    }
    Ok(Circuit { nodes })
}



#[pyfunction]
pub fn load_dimacs(filename: &str) -> io::Result<RCircuit> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut clauses = Vec::new();
    let lines = reader.lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.starts_with("c") && !l.starts_with("p"));

    for line in lines {
        let clause = line.split_whitespace()
            .map(|lit| lit.parse().unwrap())
            .take_while(|lit| *lit != 0)
            .map(|lit| RNode::Val(lit).into())
            .collect();
        clauses.push(RNode::Sum(clause).into());
    }

    Ok(RNode::Prod(clauses).into())
}
