"""Post 26, section 11: four ways to misuse Optimizer_RMSprop, measured.

Run from the series root:
    python posts/26-rmsprop/snippets/what_can_go_wrong.py

Contents: the learning rate of plain gradient descent given to RMSProp, on five seeds; rho = 1;
epsilon = 0 on a parameter whose gradient is zero; and a cache left on a layer by another
optimiser. The late spikes of the documented run are printed by rmsprop.py.

Needs NumPy and the nnfs package. The classes and train() are imported from rmsprop.py in the same
directory. Ten runs of 1,001 epochs and one of 11: about 10 seconds.
"""
import sys
import warnings

sys.dont_write_bytecode = True                  # keep the snippets directory free of __pycache__

import numpy as np

from rmsprop import Layer_Dense, Optimizer_RMSprop, train


def one_parameter_layer(weight, gradient):
    """A Layer_Dense with a single weight and bias, set by hand, in float64."""
    layer = Layer_Dense(1, 1)
    layer.weights = np.array([[weight]], dtype=np.float64)
    layer.biases = np.array([[0.0]], dtype=np.float64)
    layer.dweights = np.array([[gradient]], dtype=np.float64)
    layer.dbiases = np.array([[0.0]], dtype=np.float64)
    return layer


print("== 1. learning_rate=1.0, the default of Optimizer_SGD, against 0.02 (decay=1e-5, rho=0.999, 1,001 epochs)")
print("seed   rate 1.0: loss at epoch 1   loss at 1,000   accuracy at 1,000   |   rate 0.02: the same three")
for seed in (0, 1, 2, 3, 4):
    cells = []
    for rate in (1.0, 0.02):
        losses, accuracies = train(Optimizer_RMSprop(learning_rate=rate, decay=1e-5, rho=0.999),
                                   seed=seed, epochs=1001)
        cells.append(f"{losses[1]:8.4f}   {losses[1000]:.4f}   {accuracies[1000]:.4f}")
    print(f"{seed:4d}   " + "   |   ".join(cells), flush=True)

print()
print("== 2. rho=1.0: the cache never leaves zero (seed 0, 11 epochs)")
largest = []
losses, accuracies = train(Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=1.0), epochs=11,
                           watch=lambda epoch, dense1, dense2: largest.append(
                               max(dense1.weight_cache.max(), dense2.weight_cache.max())))
print("loss at epochs 0 to 5:", " ".join(f"{loss:.4f}" for loss in losses[:6]))
print(f"loss at epoch 10: {losses[10]:.4f}   accuracy: {accuracies[10]:.4f}   largest cache entry over the run: {max(largest)}")
print(f"learning_rate / epsilon = {0.02 / 1e-7:.0f}; two thirds of -ln(1e-7) = {-np.log(1e-7) * 2 / 3:.4f}")

print()
print("== 3. epsilon=0 and a gradient that is exactly zero")
for epsilon in (1e-7, 0.0):
    layer = one_parameter_layer(weight=0.5, gradient=0.0)
    optimizer = Optimizer_RMSprop(learning_rate=0.02, rho=0.999, epsilon=epsilon)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.update_params(layer)
    message = str(caught[0].message) if caught else "none"
    print(f"epsilon {epsilon}: weight after one update {layer.weights[0, 0]}, cache {layer.weight_cache[0, 0]}, warning: {message}")

print()
print("== 4. a cache of 122.9 left on the layer, gradient 0.1, a new Optimizer_RMSprop(rho=0.999)")
for cache in (None, 122.9):
    layer = one_parameter_layer(weight=0.5, gradient=0.1)
    if cache is not None:
        layer.weight_cache = np.array([[cache]])
        layer.bias_cache = np.array([[0.0]])
    optimizer = Optimizer_RMSprop(learning_rate=0.02, rho=0.999)
    steps, first = 0, None
    while steps == 0 or layer.weight_cache[0, 0] > 1.01 * 0.1 ** 2:
        before = layer.weights[0, 0]
        optimizer.update_params(layer)
        steps += 1
        if first is None:
            first = before - layer.weights[0, 0]
        if cache is None:
            break
    start = "no cache on the layer" if cache is None else f"cache {cache} on the layer"
    tail = "" if cache is None else f"; cache within 1 percent of g^2 = 0.01 after {steps:,} updates"
    print(f"{start}: first step {first:.6f}{tail}")
