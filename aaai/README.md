# Replicating the Experiments

To replicate the AAAI experiments, install the following additional dependencies.
Experiments where ran on a MacBook Pro M2 (2022), using python 3.12.3 and rustc 1.85.0-nightly.

```bash
pip install numpy==1.26.4 torch==2.3.1 jax==0.4.31 tqdm==4.66.2
```

To replicate the kompyle results of Table 2 in the paper, run `python aaai/benchmark_kompyle.py`. 
For ablations, the $\nabla AMC$ algorithm can be changed in the file `src/circuit/circuit.rs` on lines 55 and 64. 