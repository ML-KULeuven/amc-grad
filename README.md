# The Gradient of Algebraic Model Counting

### Installation

First, make sure you have [Python](https://www.python.org/) and [Rust](https://rustup.rs/) installed. Next, you can install `kompyle` using

```bash
pip install .
```


### Replicating the Experiments

To replicate the AAAI paper, make sure to also install the following dependencies. Experiments where ran on a MacBook Pro M2 (2022), using python 3.12.3 and rustc 1.85.0-nightly.

```bash
pip install numpy==1.26.4 torch==2.3.1 jax==0.4.31 tqdm==4.66.2
```

The scripts the replicate the experiments is stored in the `aaai` folder.