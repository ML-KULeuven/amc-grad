# Replicating the Experiments

To replicate the AAAI experiments, follow these steps.
Experiments where ran on a MacBook Pro M2 (2022), using python 3.12.3 and rustc 1.85.0-nightly.

1. Install kompyle.
```bash
pip install .
```
2. Install the following additional dependencies.
```bash 
pip install numpy==1.26.4 torch==2.3.1 jax==0.4.31 tqdm==4.66.2
```
3. [Download the circuits](https://doi.org/10.48804/MQLU85) and place the nnf files in the folder `amc-grad/mcc2021`.
4. To replicate the kompyle results of Table 2 in the paper, run `python aaai/benchmark_kompyle.py`. 
For ablations, the $\nabla \text{AMC}$ algorithm can be changed in the file `src/circuit/circuit.rs` on lines 55 and 64. 