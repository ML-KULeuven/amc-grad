import os
import time

from tqdm import tqdm

import kompyle
import torch
import numpy as np


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


def bench_wmc_grad(n=12):
    print("WMC semiring")
    timings = [[] for _ in range(n)]
    for _, c in circuit_iterator():
        for i in range(n):
            weights = get_weights(c.nb_vars()).tolist()
            result, grad, t = c._wmc_grad(weights)
            timings[i].append(t)
    print_timings(timings)


def bench_log_grad(n=3):
    print("Log semiring")
    timings = [[] for _ in range(n)]
    for _, c in circuit_iterator():
        for i in range(n):
            pos_weights = get_weights(c.nb_vars())
            neg_weights = 1 - pos_weights
            pos_weights = pos_weights.log().tolist()
            neg_weights = neg_weights.log().tolist()

            result, pos_grad, neg_grad, t = c._log_grad(pos_weights, neg_weights)
            timings[i].append(t)
    print_timings(timings)


def bench_fuzzy_grad(n=3):
    print("Fuzzy semiring")
    timings = [[] for _ in range(n)]
    for _, c in circuit_iterator():
        for i in range(n):
            pos_weights = get_weights(c.nb_vars())
            neg_weights = 1 - pos_weights
            pos_weights = pos_weights.tolist()
            neg_weights = neg_weights.tolist()

            result, pos_grad, neg_grad, t = c._fuzzy_grad(pos_weights, neg_weights)
            timings[i].append(t)
    print_timings(timings)


def bench_bool_grad(n=3):
    print("Bool semiring")
    timings = [[] for _ in range(n)]
    for _, c in circuit_iterator():
        weights = get_weights(c.nb_vars()).to(bool)
        pos_weights, neg_weights = weights.tolist(), (~weights).tolist()

        for i in range(n):
            t = c._bool_grad(pos_weights, neg_weights)[-1]
            timings[i].append(t)
    print_timings(timings)


def get_weights(nb_vars: int):
    torch.manual_seed(26541)
    weights = torch.ones(nb_vars, dtype=torch.float64)
    weights.uniform_(0., 1)
    weights[weights < 0.01] = 0
    weights.requires_grad = True
    return weights


if __name__ == "__main__":
    nb_repeats = 12
    bench_wmc_grad(nb_repeats)
    bench_bool_grad(nb_repeats)
    bench_log_grad(nb_repeats)
    bench_fuzzy_grad(nb_repeats)
