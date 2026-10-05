"""Post 30, sections 3 and 9: what each penalty does to one weight when no data gradient acts on it.

Run from the series root:
    python posts/30-l1-and-l2-regularisation/snippets/single_weight.py

A Layer_Dense(1, 1) is given an input of zero, so its data gradient is zero and backward stores
the penalty term alone. Part 1 applies plain gradient steps, weights -= learning_rate * dweights.
Part 2 hands the same layer to Optimizer_Adam. float64: this file does not call nnfs.init().

Needs NumPy (network.py imports the nnfs package without calling it). Takes about a second.
"""
import numpy as np

from network import Layer_Dense, Optimizer_Adam

ZERO_INPUT = np.zeros((1, 1))
ZERO_DVALUES = np.zeros((1, 1))


def one_weight(start, l1=0.0, l2=0.0):
    layer = Layer_Dense(1, 1, weight_regularizer_l1=l1, weight_regularizer_l2=l2)
    layer.weights = np.array([[start]])
    return layer


def plain_steps(layer, learning_rate, steps):
    """The weight after every plain gradient step on the penalty alone."""
    path = []
    for _ in range(steps):
        layer.forward(ZERO_INPUT)
        layer.backward(ZERO_DVALUES)
        layer.weights -= learning_rate * layer.dweights
        path.append(float(layer.weights[0, 0]))
    return np.array(path)


def adam_steps(layer, optimizer, steps):
    path = []
    for _ in range(steps):
        layer.forward(ZERO_INPUT)
        layer.backward(ZERO_DVALUES)
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()
        path.append(float(layer.weights[0, 0]))
    return np.array(path)


if __name__ == "__main__":
    np.random.seed(0)
    start, lam, learning_rate = 0.0105, 0.01, 0.1

    print(f"== Part 1: plain steps, learning rate {learning_rate}, lambda {lam}, start w = {start}")
    l1_path = plain_steps(one_weight(start, l1=lam), learning_rate, 1000)
    l2_path = plain_steps(one_weight(start, l2=lam), learning_rate, 1000)
    print("step      L1 weight    L2 weight")
    for step in (1, 10, 11, 12, 100, 1000):
        print(f"{step:4d}   {l1_path[step - 1]:12.6f}   {l2_path[step - 1]:10.6f}")
    print(f"L1 after step 10: |w| stays at {np.abs(l1_path[10:]).min():.6f} to {np.abs(l1_path[10:]).max():.6f}; "
          f"steps with w == 0: {int(np.sum(l1_path == 0))} of 1000")
    factor = 1 - 2 * learning_rate * lam
    print(f"L2 factor per step: 1 - 2 * learning rate * lambda = {factor:g}; "
          f"start * factor^1000 = {start * factor ** 1000:.6f}")

    zero_path = plain_steps(one_weight(0.0, l1=lam), learning_rate, 4)
    print("L1 from a weight of exactly 0:", "  ".join(f"{w:g}" for w in zero_path))

    print()
    print("== Part 2: Optimizer_Adam(learning_rate=0.001), start w = 1, L2 penalty alone, 100 steps")
    print("lambda    Adam, L2 in the gradient    w * (1 - 2 * 0.001 * lambda)^100")
    for lam in (1e-4, 1e-2, 1.0):
        adam_path = adam_steps(one_weight(1.0, l2=lam), Optimizer_Adam(learning_rate=0.001), 100)
        print(f"{lam:<8g}  {adam_path[-1]:<26.6f}  {(1 - 2 * 0.001 * lam) ** 100:.6f}")

    print()
    print("== Part 2: Optimizer_Adam(learning_rate=0.05), start w = 1, L1 = 5e-4 alone, 2,000 steps")
    adam_path = adam_steps(one_weight(1.0, l1=5e-4), Optimizer_Adam(learning_rate=0.05), 2000)
    tail = np.abs(adam_path[1000:])
    print(f"first step with w < 0: {int(np.argmax(adam_path < 0)) + 1}; "
          f"|w| over steps 1,001 to 2,000: {tail.min():.4f} to {tail.max():.4f}; "
          f"steps with |w| < 0.001: {int(np.sum(tail < 1e-3))}; with w == 0: {int(np.sum(adam_path == 0))}")

    print()
    print("== Part 2: Optimizer_Adam(learning_rate=0.05, decay=1e-5), start w = 1, one penalty alone, 10,001 steps")
    for label, strengths in (("L1 = 5e-4", dict(l1=5e-4)), ("L2 = 5e-4", dict(l2=5e-4))):
        adam_path = adam_steps(one_weight(1.0, **strengths), Optimizer_Adam(learning_rate=0.05, decay=1e-5), 10001)
        tail = np.abs(adam_path[-1000:])
        print(f"{label}: steps with |w| < 0.001 among the last 1,000: {int(np.sum(tail < 1e-3))}; "
              f"largest |w| among them: {tail.max():.4f}; steps with w == 0: {int(np.sum(adam_path == 0))}")
