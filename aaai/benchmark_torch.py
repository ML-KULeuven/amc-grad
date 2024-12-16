import time
import os

from tqdm import tqdm
import torch
import numpy as np
import kompyle


def circuit_iterator():
    for i in tqdm(range(200)):
        path = f"mc2021/{i:03}.nnf"
        if not os.path.exists(path):
            continue
        size = os.stat(path).st_size
        if size != 0:
            # print("Loading", path, f"({size / 1024 / 1024:.2f} MB)")
            yield path, kompyle.load_d4(path)


def print_timings(timings):
    timings = [np.mean(ts) * 1000 for ts in timings]
    print(timings)
    print(f"{np.mean(timings[2:]):.1f} ± {np.std(timings[2:]):.1f}")


def get_weights(nb_vars: int):
    torch.manual_seed(26541)
    weights = torch.ones(nb_vars, dtype=torch.float64)
    weights.uniform_(0., 1)
    weights[weights < 0.01] = 0
    weights.requires_grad = True
    return weights


def wmc_python(nnf_string, weights) -> torch.Tensor:
    ONE = torch.tensor(1., dtype=torch.float64)
    ZERO = torch.tensor(0., dtype=torch.float64)

    weights = torch.stack([1 - weights, weights], dim=1)

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


def log_python(nnf_string, weights) -> torch.Tensor:
    ONE = torch.tensor(0., dtype=torch.float64)
    ZERO = torch.tensor(-float('inf'), dtype=torch.float64)

    weights = torch.stack([1 - weights, weights], dim=1)
    weights = weights.log()

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
                lits_val = nodes[target][0] + lit_weights.sum()

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.logaddexp(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]


def bench_pytorch_grad(n=3, log=False):
    timings = [[] for _ in range(n)]
    for path, c in circuit_iterator():
        with open(path) as f:
            nnf = f.read()

        for i in range(n):
            weights = get_weights(c.nb_vars())
            if log:
                result = log_python(nnf, weights)
            else:
                result = wmc_python(nnf, weights)
            t1 = time.time()
            result.backward()
            delta = time.time() - t1
            print(delta)
            timings[i].append(delta)

    print_timings(timings)


if __name__ == "__main__":
    bench_pytorch_grad(12, False)
