"""Post 24, section 10: four mistakes with a velocity buffer, measured on the ravine of ravine.py.

Run from the series root:
    python posts/24-momentum/snippets/what_can_go_wrong.py

Each mistake is a subclass of Optimizer_SGD or a misuse of it, run from (10, 1) with a learning
rate of 0.019 and a momentum coefficient of 0.9. Nothing here is random; float64 throughout.

Needs NumPy (momentum_sgd.py also imports the nnfs package). Runs in about a second.
"""
import numpy as np

from momentum_sgd import Optimizer_SGD
from ravine import CURVATURE, Ravine, loss, pair, run

ALPHA = 0.019


class Buffer_Recreated(Optimizer_SGD):
    """Mistake 1: the buffers are zeroed on every call instead of on the first one."""

    def update_params(self, layer):
        layer.weight_momentums = np.zeros_like(layer.weights)
        layer.bias_momentums = np.zeros_like(layer.biases)
        super().update_params(layer)


class Other_Velocity_Line(Optimizer_SGD):
    """Mistake 2: the velocity line of the other convention, the += of this one."""

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
        layer.weight_momentums = self.momentum * layer.weight_momentums + layer.dweights
        layer.weights += layer.weight_momentums


if __name__ == "__main__":
    correct = run(Optimizer_SGD(learning_rate=ALPHA, momentum=0.9), 100)
    plain = run(Optimizer_SGD(learning_rate=ALPHA), 100)

    print("== 1. The buffer recreated on every step")
    recreated = run(Buffer_Recreated(learning_rate=ALPHA, momentum=0.9), 100)
    print(f"loss after 100 steps: correct {loss(correct[-1]):.4f}   recreated {loss(recreated[-1]):.4f}   "
          f"no momentum {loss(plain[-1]):.4f}")
    print(f"largest gap between the recreated path and the path without momentum: "
          f"{np.abs(recreated - plain).max():.1e}")

    print()
    print("== 2. v = beta * v + g, then weights += v")
    uphill = run(Other_Velocity_Line(learning_rate=ALPHA, momentum=0.9), 3)
    print("loss at the start and after 1, 2 and 3 steps: " + "  ".join(f"{loss(p):,.0f}" for p in uphill))

    print()
    print("== 3. A layer that still carries the buffer of an earlier run")
    layer = Ravine()
    first = Optimizer_SGD(learning_rate=ALPHA, momentum=0.9)
    for _ in range(3):
        layer.backward()
        first.update_params(layer)
    print(f"velocity left on the layer after 3 steps: {pair(layer.weight_momentums[0])}")
    layer.weights = np.array([[10.0, 1.0]])               # new starting weights, same layer object
    layer.backward()
    Optimizer_SGD(learning_rate=ALPHA, momentum=0.9).update_params(layer)
    fresh = Ravine()
    fresh.backward()
    Optimizer_SGD(learning_rate=ALPHA, momentum=0.9).update_params(fresh)
    print(f"first step of the second run: {pair(layer.weights[0] - np.array([10.0, 1.0]))}   "
          f"on a fresh layer: {pair(fresh.weights[0] - np.array([10.0, 1.0]))}")
    print(f"loss after that step: {loss(layer.weights):.2f}   on a fresh layer: {loss(fresh.weights):.2f}")

    print()
    print("== 4. momentum = 1.0: nothing is ever forgotten")
    for beta in (0.9, 1.0):
        path = run(Optimizer_SGD(learning_rate=ALPHA, momentum=beta), 2000)
        losses = np.array([loss(p) for p in path])
        print(f"momentum {beta}: loss after 100 steps {losses[100]:9.4f}   "
              f"lowest and highest loss over steps 1,000 to 2,000: {losses[1000:].min():.2e} and {losses[1000:].max():.2e}")
