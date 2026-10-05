"""Post 25, section 10: four ways to get AdaGrad wrong, each measured.

Run from the series root:
    python posts/25-adagrad/snippets/what_can_go_wrong.py

Contents: epsilon moved inside the square root; epsilon set to zero; the cache zeroed on every
call; and the square root left out. Each mistake is a subclass that changes only update_params.

Needs NumPy and the nnfs package. The classes and the training loop are imported from adagrad.py
in the same directory. Seeded; about 10 seconds.
"""
import sys
import warnings

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

import numpy as np

from adagrad import Optimizer_Adagrad, toy_layer, train


class Adagrad_EpsilonInside(Optimizer_Adagrad):
    """epsilon under the square root instead of beside it."""

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         np.sqrt(layer.weight_cache + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         np.sqrt(layer.bias_cache + self.epsilon)


class Adagrad_CacheReset(Optimizer_Adagrad):
    """The cache is created afresh on every call, so it holds one squared gradient."""

    def update_params(self, layer):
        layer.weight_cache = np.zeros_like(layer.weights)
        layer.bias_cache   = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache)   + self.epsilon)


class Adagrad_NoRoot(Optimizer_Adagrad):
    """The step is divided by the cache itself."""

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache   += layer.dbiases ** 2
        layer.weights -= self.current_learning_rate * layer.dweights / \
                         (layer.weight_cache + self.epsilon)
        layer.biases  -= self.current_learning_rate * layer.dbiases / \
                         (layer.bias_cache   + self.epsilon)


def one_update(optimizer, gradient):
    """The step a fresh optimiser gives a weight of 0 whose first gradient is the given one."""
    layer = toy_layer([0.0] * len(gradient))
    layer.dweights = np.array([gradient])
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    return -layer.weights[0]


def losses_at(optimizer, epochs, wanted):
    """Train on seed 0 and return the loss, accuracy, dead neurons and largest weight at some epochs."""
    seen = {}

    def watch(epoch, loss, accuracy, dead, layers, _):
        if epoch in wanted:
            largest = max(float(np.max(np.abs(layer.weights))) for layer in layers)
            seen[epoch] = (float(loss), float(accuracy), dead, largest)

    train(optimizer, seed=0, epochs=epochs, watch=watch)
    return seen


print("== epsilon inside the root: the first step of a parameter, alpha = 1, epsilon = 1e-7")
gradients = [1e-1, 1e-3, 1e-4, 1e-5]
outside = one_update(Optimizer_Adagrad(), gradients)
inside = one_update(Adagrad_EpsilonInside(), gradients)
print("first gradient      ", "  ".join(f"{g:8.0e}" for g in gradients))
print("step, epsilon outside", "  ".join(f"{s:8.4f}" for s in outside))
print("step, epsilon inside ", "  ".join(f"{s:8.4f}" for s in inside))
print(f"the two agree while |g| is well above sqrt(epsilon) = {np.sqrt(1e-7):.2e}")
for name, optimizer in (("outside", Optimizer_Adagrad(learning_rate=1.0, decay=1e-4)),
                        ("inside ", Adagrad_EpsilonInside(learning_rate=1.0, decay=1e-4))):
    seen = losses_at(optimizer, 2, (0, 1))
    print(f"spiral run, seed 0, epsilon {name}: loss {seen[0][0]:.4f} before the first update, "
          f"{seen[1][0]:.4f} after it")

print()
print("== epsilon = 0: a parameter whose gradient is exactly zero")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    steps = one_update(Optimizer_Adagrad(epsilon=0.0), [0.5, 0.0])
print("steps for the gradients 0.5 and 0.0:", steps, "  warning:", caught[0].message)
print("with epsilon = 1e-7:               ", one_update(Optimizer_Adagrad(), [0.5, 0.0]))

print()
print("== the cache zeroed on every call (seed 0, 10,001 epochs)")
seen = losses_at(Adagrad_CacheReset(learning_rate=1.0, decay=1e-4), 10001, (1, 10, 100, 10000))
for epoch, (loss, accuracy, dead, largest) in seen.items():
    print(f"epoch {epoch:5d}: loss {loss:.4f}, accuracy {accuracy:.4f}, dead neurons {dead:2d} of 64")
print("every step has the size of the learning rate:",
      one_update(Adagrad_CacheReset(), [1e-1, 1e-3, -1e-5]))

print()
print("== the square root left out (seed 0, 101 epochs)")
print("first step for the gradients 0.1, 0.001 and 0.00001:",
      " ".join(f"{s:.4g}" for s in one_update(Adagrad_NoRoot(), [1e-1, 1e-3, 1e-5])))
seen = losses_at(Adagrad_NoRoot(learning_rate=1.0, decay=1e-4), 101, (0, 1, 100))
for epoch, (loss, accuracy, dead, largest) in seen.items():
    print(f"epoch {epoch:5d}: loss {loss:.4f}, accuracy {accuracy:.4f}, "
          f"largest |weight| after its update {largest:.3g}")
