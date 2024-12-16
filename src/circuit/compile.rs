use std::rc::Rc;
use std::process::Command;
use tempfile::NamedTempFile;
use pyo3::Python;
use std::io::Write;
use std::ops::Add;
use std::path::Path;
use crate::circuit::circuit::Circuit;
use crate::circuit::parser::load_d4;
use crate::circuit::rcircuit::RNode;


use crate::circuit::tseitin::tseitin_transform;


const PLATFORM: &str =
if cfg!(target_os = "linux") & cfg!(target_arch = "x86_64") {
    "Linux"
} else if cfg!(target_os = "macos") & cfg!(target_arch = "x86_64") {
    "Darwin"
} else if cfg!(target_os = "macos") & cfg!(target_arch = "aarch64") {
    "Darwin-arm"
} else {
    panic!("Unsupported platform")
};


fn get_lib_path() -> String {
    Python::with_gil(|py| {
        let importlib = py.import("importlib.resources").unwrap();
        let path = importlib.getattr("files").unwrap()
            .call1(("kompyle",)).unwrap();
        path.to_string()
    })
}


fn set_executable(path: &str) {
    Command::new("chmod")
        .arg("+x")
        .arg(path)
        .output()
        .expect("Failed to set executable");
}


pub fn to_dimacs(node: Rc<RNode>) -> String {
    // Assumes that the circuit is in CNF
    match *node {
        RNode::Sum(ref nodes) => {
            nodes.iter()
                .map(|n| to_dimacs(n.clone()))
                .collect::<Vec<String>>()
                .join(" ").add(" 0")
        }
        RNode::Prod(ref nodes) => {
            let nb_vars = node.nb_vars();
            let header: String = format!("p cnf {} {}\n", nb_vars, nodes.len());
            let body = nodes.iter()
                .map(|n| to_dimacs(n.clone()))
                .collect::<Vec<String>>()
                .join("\n");
            header + &body
        }
        RNode::Val(value) => value.to_string(),
        _ => panic!("Unexpected true/false node"),
    }
}

pub fn compile(node: Rc<RNode>, nb_vars: u32, solver_name: String) -> Result<Circuit, std::io::Error> {
    let node = tseitin_transform(node.simplify(), nb_vars);
    let dimacs = to_dimacs(node);
    let mut file_dimacs = NamedTempFile::new()?;
    file_dimacs.write_all(dimacs.as_bytes())?;
    let file_nnf = NamedTempFile::new()?;

    if solver_name == "d4" {
        _compile_d4(file_dimacs.path(), file_nnf.path())
    } else if solver_name == "sharpsat" {
        _compile_sharpsat(file_dimacs.path(), file_nnf.path())
    } else {
        panic!("Unknown solver")
    }
}

fn _compile_d4(dimacs_file: &Path, out_file: &Path) -> Result<Circuit, std::io::Error> {
    // run d4 solver
    let d4_path = format!("{}/lib/{}/d4", get_lib_path(), PLATFORM);
    set_executable(&d4_path);
    let output = Command::new(d4_path)
        .arg("-dDNNF")
        .arg(dimacs_file)
        .arg(format!("-out={}", out_file.display()))
        .output()?;
    assert!(output.status.success());
    Ok(load_d4(out_file.to_str().unwrap())?)
}


fn _compile_sharpsat(dimacs_file: &Path, out_file: &Path) -> Result<Circuit, std::io::Error> {
    let solver_folder = format!("{}/lib/{}", get_lib_path(), PLATFORM);
    let solver_path = format!("{}/sharpSAT", solver_folder);
    let tmp_path = format!("{}/tmp", solver_folder);

    println!("Solver path: {}", solver_path);
    println!("TMP path: {}", tmp_path);
    set_executable(&solver_path);
    if !Path::new(tmp_path.as_str()).exists() {
        std::fs::create_dir(tmp_path.clone())?;
    }
    println!("RUNNING sharpSAT");
    println!("Current directory: {}", std::env::current_dir().unwrap().display());

    let output = Command::new(solver_path)
        //.current_dir(solver_folder)
        //.arg("-dDNNF")
        //.arg("-decot 1")
        //.arg("-decow 100")
        //.arg("-tmpdir ./tmp")
        //.arg("-cs 3500")
        //.arg(dimacs_file)
        // .arg(format!("-dDNNF_out {}", out_file.display()))
        .output()?;
    // print output
    println!("OUTPUT:");
    println!("{}", String::from_utf8_lossy(&output.stdout));
    println!("{}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
    todo!()
}