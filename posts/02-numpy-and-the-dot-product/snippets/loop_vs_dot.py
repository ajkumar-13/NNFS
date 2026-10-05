"""Time one batched layer call two ways: nested Python loops and np.dot.

Run from the series root:  python posts/02-numpy-and-the-dot-product/snippets/loop_vs_dot.py

The two results are compared before any time is reported. The times depend on the machine and
change from run to run; what the post quotes is the size of the gap.
"""
import time

import numpy as np


def layer_loop(samples, neurons):
    """One dot product per (sample, neuron) pair, with plain Python lists and loops."""
    outputs = []
    for sample in samples:
        row = []
        for neuron_weights in neurons:
            total = 0.0
            for x_i, w_i in zip(sample, neuron_weights):
                total += x_i * w_i
            row.append(total)
        outputs.append(row)
    return outputs


def best_of(repeats, function, *args):
    """Smallest wall-clock time of several calls, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def compare(n_samples, n_features, n_neurons, loop_repeats, dot_repeats):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features))    # one row per sample
    W = rng.standard_normal((n_neurons, n_features))    # one row per neuron
    X_list, W_list = X.tolist(), W.tolist()

    same = np.allclose(layer_loop(X_list, W_list), np.dot(X, W.T))
    loop_seconds = best_of(loop_repeats, layer_loop, X_list, W_list)
    dot_seconds = best_of(dot_repeats, np.dot, X, W.T)

    print(f"({n_samples}, {n_features}) by ({n_features}, {n_neurons}): "
          f"{n_samples * n_features * n_neurons:,d} multiply-adds, same numbers: {same}")
    print(f"  nested loops {loop_seconds * 1e3:12.4f} ms")
    print(f"  np.dot       {dot_seconds * 1e3:12.4f} ms")
    print(f"  the loops take {loop_seconds / dot_seconds:,.0f} times as long")


compare(3, 4, 3, loop_repeats=2000, dot_repeats=2000)       # the toy layer of this post
compare(1000, 100, 64, loop_repeats=3, dot_repeats=20)      # a realistic batch
