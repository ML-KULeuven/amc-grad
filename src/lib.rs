mod circuit;
mod algebra;

use pyo3::prelude::*;
use circuit::circuit::Circuit;
use circuit::rcircuit::RCircuit;
use circuit::parser::{load_d4, load_dimacs};


#[pymodule]
fn kompyle(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_d4, m)?)?;
    m.add_function(wrap_pyfunction!(circuit::zero, m)?)?;
    m.add_function(wrap_pyfunction!(circuit::one, m)?)?;
    m.add_function(wrap_pyfunction!(circuit::lit, m)?)?;
    m.add_function(wrap_pyfunction!(load_dimacs, m)?)?;

    m.add_class::<Circuit>()?;
    m.add_class::<RCircuit>()?;
    Ok(())
}
