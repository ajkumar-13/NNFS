"""Post 31, section 2: every subnetwork of a small network, enumerated, against the full network.

Run from the series root:
    python posts/31-dropout/snippets/ensemble.py

A 2-8-3 network with a dropout layer after the ReLU has 2^8 = 256 masks. For each one the
script computes the logits and the class probabilities of that thinned network (kept neurons
scaled by 1 / (1 - p)), weights it by the probability of its mask, and compares three averages
with the output of the full network, mask off. This file does not call nnfs.init(): float64.

Needs NumPy and the nnfs package (spiral_data only). Takes about a second.
"""
from itertools import product

import numpy as np
from nnfs.datasets import spiral_data

from network import Activation_ReLU, Activation_Softmax, Layer_Dense

N_NEURONS = 8
p = 0.5

np.random.seed(0)
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, N_NEURONS)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(N_NEURONS, 3)
softmax = Activation_Softmax()
dense1.weights = np.random.randn(2, N_NEURONS)      # weights and biases of ordinary size
dense1.biases = np.random.randn(1, N_NEURONS)
dense2.weights = np.random.randn(N_NEURONS, 3)
dense2.biases = np.random.randn(1, 3)

dense1.forward(X)
activation1.forward(dense1.output)

# The full network, mask off.
dense2.forward(activation1.output)
softmax.forward(dense2.output)
full_logits = dense2.output.copy()
full_probabilities = softmax.output.copy()

# Every mask, weighted by its probability.
mean_logits = np.zeros_like(full_logits)
mean_probabilities = np.zeros_like(full_probabilities)
mean_log_probabilities = np.zeros_like(full_probabilities)
total_weight = 0.0
disagree = []
n_kept = []
for kept in product((0, 1), repeat=N_NEURONS):
    kept = np.array(kept)
    weight = (1 - p) ** kept.sum() * p ** (N_NEURONS - kept.sum())
    dense2.forward(activation1.output * kept / (1 - p))
    softmax.forward(dense2.output)
    total_weight += weight
    mean_logits += weight * dense2.output
    mean_probabilities += weight * softmax.output
    mean_log_probabilities += weight * np.log(softmax.output)
    disagree.append(np.mean(np.argmax(softmax.output, axis=1) != np.argmax(full_probabilities, axis=1)))
    n_kept.append(kept.sum())

geometric = np.exp(mean_log_probabilities)
geometric /= geometric.sum(axis=1, keepdims=True)

print(f"masks: {2 ** N_NEURONS}, their probabilities sum to {total_weight:.6f}; points: {len(X)}; p = {p}")
print(f"mean of the logits over the masks, against the full network:        largest gap "
      f"{np.max(np.abs(mean_logits - full_logits)):.1e}")
print(f"renormalised geometric mean of the probabilities, against the full: largest gap "
      f"{np.max(np.abs(geometric - full_probabilities)):.1e}")
print(f"arithmetic mean of the probabilities, against the full:             largest gap "
      f"{np.max(np.abs(mean_probabilities - full_probabilities)):.4f}")
same = np.mean(np.argmax(mean_probabilities, axis=1) == np.argmax(full_probabilities, axis=1))
print(f"points where the arithmetic mean and the full network pick the same class: {int(round(same * len(X)))} of {len(X)}")
print(f"one thinned network against the full network: it picks another class on "
      f"{100 * min(disagree):.1f} to {100 * max(disagree):.1f} percent of the points "
      f"(mean over the masks {100 * np.mean(disagree):.1f})")
print("the same by the number of neurons a mask keeps, in percent of the points:")
print("kept   masks   lowest    mean   highest")
disagree, n_kept = np.array(disagree), np.array(n_kept)
for k in range(N_NEURONS + 1):
    d = 100 * disagree[n_kept == k]
    print(f"{k:4d}   {len(d):5d}   {d.min():6.1f}  {d.mean():6.1f}   {d.max():7.1f}")
