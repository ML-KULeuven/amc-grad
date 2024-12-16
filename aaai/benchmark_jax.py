import random

import math
import os
import time

import kompyle
import jax.numpy as jnp
from jax import grad, jit, vmap, random, config
import numpy as np


config.update("jax_enable_x64", True)


def wmc_python(nnf_string, weights):
    ONE = jnp.array(1., dtype=jnp.float64)
    ZERO = jnp.array(0., dtype=jnp.float64)

    weights = jnp.stack([1 - weights, weights], axis=1)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in lines:
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([ZERO, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([ONE, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[ix1, ix2]
                lits_val = nodes[target][0] * lit_weights.prod()

            if nodes[source][1] == 'o':
                nodes[source][0] = nodes[source][0] + lits_val
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] * lits_val

    return nodes[1][0]


def circuit_iterator():
    for i in range(200):
        path = f"mc2021/{i:03}.nnf"
        if not os.path.exists(path):
            continue
        size = os.stat(path).st_size
        if size != 0:
            print("Loading", path, f"({size / 1024 / 1024:.2f} MB)")
            yield path, kompyle.load_d4(path)


def get_weights(nb_vars: int):
    key = random.PRNGKey(26541)
    weights = random.uniform(key, (nb_vars,), minval=0., maxval=1., dtype=jnp.float64)
    weights = jnp.where(weights < 0.01, 0, weights)
    return weights


def bench_jax_grad(n=3):
    timings = []
    for path, c in circuit_iterator():
        with open(path) as f:
            nnf = f.read()
        weights = get_weights(c.nb_vars())
        wmc_grad = grad(wmc_python, argnums=1) #, static_argnums=0)

        for i in range(n):
            t1 = time.time()
            result = wmc_grad(nnf, weights)
            delta = time.time() - t1
            print("GRAD", f" - {delta:.4f}s")


if __name__ == "__main__":
    bench_jax_grad()
