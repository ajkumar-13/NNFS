"""Post 33, sections 2 to 5: ten stacked 64-neuron layers under the three schemes, forward and backward.

Run from the series root:
    python posts/33-weight-initialisation/snippets/ten_layers.py

The stack: spiral_data(samples=100, classes=3), ten Layer_Dense layers of 64 neurons, each followed
by an activation, then Layer_Dense(64, 3) and the combined softmax and loss. After np.random.seed(s)
the draws are the data and then the weights, layer by layer, so the three schemes see the same
standard normal draws and differ only in the scale.

It differs from init_scale.py of post 04 in three ways: an activation follows every layer, there
are ten layers instead of six, and it runs in float64 without nnfs.init(). Part 1 repeats post 04's
six linear layers to show that the numbers agree.

Needs NumPy and the nnfs package (spiral_data). Takes about ten seconds.
"""
import numpy as np
from nnfs.datasets import spiral_data

from network import (Layer_Dense, Activation_ReLU, Activation_Tanh,
                     Activation_Softmax_Loss_CategoricalCrossentropy,
                     build_stack, forward_stack, backward_stack)

SIZES = [2] + [64] * 10 + [3]
INITS = ("small", "xavier", "he")
N_SEEDS = 100


def measure(seed, init, activation):
    """One forward and one backward pass at initialisation; returns per-layer figures."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    dense, activations = build_stack(SIZES, init, activation)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = loss_activation.forward(forward_stack(X, dense, activations), y)
    loss_activation.backward(loss_activation.output, y)
    backward_stack(loss_activation.dinputs, dense, activations)

    square = [np.mean(X ** 2)] + [np.mean(a.output ** 2) for a in activations]
    gradient = [np.mean(dense[i + 1].dinputs ** 2) for i in range(10)]      # with respect to a_1 .. a_10
    return dict(
        loss=float(loss),
        std=[float(a.output.std()) for a in activations],
        zeros=[float(np.mean(a.output == 0)) for a in activations],
        forward=[square[i + 1] / square[i] for i in range(10)],             # layer i+1 over layer i
        backward=[gradient[i - 1] / gradient[i] for i in range(1, 10)],     # through dense layers 2 .. 10
        dweights=[float(np.sqrt(np.mean(layer.dweights ** 2))) for layer in dense],
    )


print("== 1. Post 04's stack again: six linear 64-neuron layers at scale 0.01, seed 0")
np.random.seed(0)
values, _ = spiral_data(samples=100, classes=3)
spreads = []
for n_inputs in (2, 64, 64, 64, 64, 64):
    layer = Layer_Dense(n_inputs, 64, init="small")
    layer.forward(values)
    values = layer.output
    spreads.append(values.std())
print(f"standard deviation of the outputs: layer 1 {spreads[0]:.4f}, layer 6 {spreads[5]:.1e}; "
      f"per layer x {(spreads[5] / spreads[0]) ** 0.2:.3f}   (0.01 * sqrt(64) = 0.08)")

print()
print("== 2. Ten ReLU layers, seed 0: standard deviation of each layer's activations")
runs = {init: measure(0, init, Activation_ReLU) for init in INITS}
print("layer   small       xavier      he          fraction of zeros (the same for all three)")
for i in range(10):
    print(f"{i + 1:<6}  " + "  ".join(f"{runs[init]['std'][i]:<10.3e}" for init in INITS) +
          f"  {runs['he']['zeros'][i]:.3f}")
print("loss at initialisation: " + ", ".join(f"{init} {runs[init]['loss']:.7f}" for init in INITS) +
      f"   (ln 3 = {np.log(3):.7f})")

print()
print(f"== 3. Ten ReLU layers: the factor per layer, mean over seeds 0 to {N_SEEDS - 1}")
print("init     forward, layer 1      forward, layers 2 to 10        backward, dense layers 2 to 10")
print("         predicted  measured   predicted  measured            predicted  measured")
spread = {init: [measure(seed, init, Activation_ReLU) for seed in range(N_SEEDS)] for init in INITS}
for init in INITS:
    scale = {"small": (0.01, 0.01), "xavier": (np.sqrt(2 / 66), np.sqrt(2 / 128)),
             "he": (1.0, np.sqrt(2 / 64))}[init]
    forward = np.mean([r["forward"] for r in spread[init]], axis=0)
    backward = np.mean([r["backward"] for r in spread[init]], axis=0)
    print(f"{init:<7}  {2 * scale[0] ** 2 / 2:<9.4g}  {forward[0]:<9.4g}  {64 * scale[1] ** 2 / 2:<9.4g}  "
          f"{forward[1:].min():.4f} to {forward[1:].max():.4f}    {64 * scale[1] ** 2 / 2:<9.4g}  "
          f"{backward.min():.4f} to {backward.max():.4f}")
single = runs["he"]["forward"][1:]
print(f"one seed scatters: he, seed 0, forward factors of layers 2 to 10 from {min(single):.2f} to {max(single):.2f}")
zeros = np.mean([r["zeros"] for r in spread["he"]], axis=0)
print(f"fraction of zeros per layer, mean over the seeds: {zeros.min():.3f} to {zeros.max():.3f}; "
      f"seed 0 alone: {min(runs['he']['zeros']):.3f} to {max(runs['he']['zeros']):.3f}")

print()
print("== 4. Ten ReLU layers, seed 0: root mean square of dweights, by dense layer")
print("init     layer 1     layers 2 to 10            layer 11 (output)")
for init in INITS:
    d = runs[init]["dweights"]
    print(f"{init:<7}  {d[0]:<10.2e}  {min(d[1:10]):.2e} to {max(d[1:10]):.2e}      {d[10]:.2e}")

print()
print(f"== 5. Ten tanh layers: the factor per layer, mean over seeds 0 to {N_SEEDS - 1}, and seed 0's spread")
print("init     forward, layers 2 to 10   backward, dense layers 2 to 10   std layer 1   std layer 10   "
      "rms dweights layer 1   layer 10")
for init in INITS:
    rows = [measure(seed, init, Activation_Tanh) for seed in range(N_SEEDS)]
    forward = np.mean([r["forward"] for r in rows], axis=0)
    backward = np.mean([r["backward"] for r in rows], axis=0)
    first = rows[0]
    print(f"{init:<7}  {forward[1:].min():.4f} to {forward[1:].max():.4f}          "
          f"{backward.min():.4f} to {backward.max():.4f}                 "
          f"{first['std'][0]:<11.3e}  {first['std'][9]:<12.3e}  {first['dweights'][0]:<20.2e}  {first['dweights'][9]:.2e}")
