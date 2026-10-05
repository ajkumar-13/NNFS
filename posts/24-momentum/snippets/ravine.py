"""Post 24, sections 1 to 3 and 8: momentum on a ravine with two parameters.

Run from the series root:
    python posts/24-momentum/snippets/ravine.py

The loss is L(x, y) = (x**2 + 100 * y**2) / 2: a gentle slope along x, the floor of the ravine,
and steep walls along y. Its gradient is (x, 100 * y) and its minimum is at (0, 0). Every path
starts at (10, 1) and is produced by the Optimizer_SGD class of momentum_sgd.py, driven through
a stand-in for a layer. Nothing here is random, and nnfs.init() is not called: float64 throughout.

Needs NumPy (momentum_sgd.py also imports the nnfs package). Runs in about a second.
"""
import numpy as np

from momentum_sgd import Optimizer_SGD

CURVATURE = np.array([[1.0, 100.0]])        # gradient = CURVATURE * position
START = np.array([[10.0, 1.0]])


class Ravine:
    """What an optimiser needs of a layer: weights, biases and their gradients."""

    def __init__(self):
        self.weights = START.copy()
        self.biases = np.zeros((1, 1))
        self.dbiases = np.zeros((1, 1))

    def backward(self):
        self.dweights = CURVATURE * self.weights


def loss(position):
    return float(np.sum(CURVATURE * position ** 2) / 2)


def pair(vector):
    """Two numbers to four decimals, with a rounding residue such as -1e-16 shown as 0."""
    x, y = np.round(vector, 4) + 0.0
    return f"({x:7.4f}, {y:7.4f})"


def run(optimizer, steps):
    """The positions visited, the start included: an array of shape (steps + 1, 2)."""
    layer = Ravine()
    path = [layer.weights.copy()]
    for _ in range(steps):
        layer.backward()
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()
        path.append(layer.weights.copy())
    return np.concatenate(path)


if __name__ == "__main__":
    alpha = 0.019

    print("== Sections 1 and 2: the first steps, without and with momentum")
    plain = run(Optimizer_SGD(learning_rate=alpha), 100)
    heavy = run(Optimizer_SGD(learning_rate=alpha, momentum=0.9), 100)
    print(f"start {pair(START[0])}  loss {loss(START):.1f}  learning rate {alpha}")
    print("step   no momentum: step, position             momentum 0.9: step, position")
    for t in range(1, 5):
        print(f"{t:4d}   {pair(plain[t] - plain[t - 1])} {pair(plain[t])}   "
              f"{pair(heavy[t] - heavy[t - 1])} {pair(heavy[t])}")
    print(f"steps 1 and 2 without momentum, added: {pair(plain[2] - plain[0])}")
    gradient_2 = CURVATURE[0] * heavy[1]
    print(f"step 2 with momentum = 0.9 * step 1 - alpha * gradient 2 = "
          f"{pair(0.9 * (heavy[1] - heavy[0]))} + {pair(-alpha * gradient_2)} = {pair(heavy[2] - heavy[1])}")

    print()
    print("== Sections 1 and 8: 100 steps for four momentum coefficients")
    print("beta    x      |y|        loss      travel across  travel along  across steps that reverse")
    for beta in (0.0, 0.5, 0.9, 0.99):
        path = run(Optimizer_SGD(learning_rate=alpha, momentum=beta), 100)
        steps = np.diff(path, axis=0)
        across = steps[np.abs(steps[:, 1]) > 1e-9, 1]       # leaves out steps that are rounding residue
        reversals = int(np.sum(across[1:] * across[:-1] < 0))
        print(f"{beta:4.2f}  {path[-1, 0]:7.4f}  {abs(path[-1, 1]):.2e}  {loss(path[-1]):9.4f}"
              f"  {np.abs(steps[:, 1]).sum():13.3f}  {np.abs(steps[:, 0]).sum():12.3f}  {reversals:25d}")

    print()
    print("== Section 3: the class against the formula")
    for beta in (0.0, 0.9):
        theta, v = START.copy(), np.zeros_like(START)
        by_formula = [theta.copy()]
        for _ in range(100):
            g = CURVATURE * theta
            v = beta * v - alpha * g                    # v_t = beta * v_(t-1) - alpha * g
            theta = theta + v                           # theta_t = theta_(t-1) + v_t
            by_formula.append(theta.copy())
        by_class = run(Optimizer_SGD(learning_rate=alpha, momentum=beta), 100)
        print(f"beta {beta}: largest gap between class and formula over 100 steps "
              f"{np.abs(by_class - np.concatenate(by_formula)).max():.1e}")
    theta = START.copy()
    without = [theta.copy()]
    for _ in range(100):
        theta = theta - alpha * (CURVATURE * theta)     # the update of post 22
        without.append(theta.copy())
    print(f"momentum 0.0 against theta -= alpha * g: largest gap "
          f"{np.abs(run(Optimizer_SGD(learning_rate=alpha), 100) - np.concatenate(without)).max():.1e}")

    print()
    print("== Section 3: a gradient that never changes (g = 1, alpha = 1)")
    for beta in (0.5, 0.9, 0.99):
        v, sizes = 0.0, {}
        for t in range(1, 1001):
            v = beta * v - 1.0
            sizes[t] = -v
        shown = "  ".join(f"t={t}: {sizes[t]:.3f}" for t in (1, 2, 3, 10, 50, 1000))
        print(f"beta {beta:4.2f}  {shown}   1 / (1 - beta) = {1 / (1 - beta):.0f}")
    print(f"share of the weight on the latest 10 gradients, beta 0.9: {1 - 0.9 ** 10:.3f}")

    print()
    print("== Section 3.1: the two conventions")
    for decay in (0.0, 0.01):
        beta = 0.9
        series = run(Optimizer_SGD(learning_rate=alpha, decay=decay, momentum=beta), 100)
        theta, u = START.copy(), np.zeros_like(START)
        other = [theta.copy()]
        for t in range(100):
            rate = alpha / (1.0 + decay * t)
            g = CURVATURE * theta
            u = beta * u + g                            # the gradient goes in unscaled
            theta = theta - rate * u                    # the learning rate enters at the step
            other.append(theta.copy())
        other = np.concatenate(other)
        print(f"decay {decay}: largest gap between the two paths {np.abs(series - other).max():.1e}   "
              f"loss after 100 steps {loss(series[-1]):.6f} against {loss(other[-1]):.6f}")
    layer = Ravine()
    optimizer = Optimizer_SGD(learning_rate=alpha, momentum=0.9)
    u = np.zeros_like(START)
    for _ in range(3):
        layer.backward()
        u = 0.9 * u + layer.dweights
        optimizer.update_params(layer)
    print(f"after 3 steps at a constant rate: v = {pair(layer.weight_momentums[0])}   "
          f"-alpha * u = {pair(-alpha * u[0])}")
