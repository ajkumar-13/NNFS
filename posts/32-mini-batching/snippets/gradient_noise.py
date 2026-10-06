"""Post 32, section 2: what a mini-batch gradient is, measured on one fixed network.

Run from the series root:
    python posts/32-mini-batching/snippets/gradient_noise.py

The network is not trained here. Its parameters are set once and stay fixed, so the only thing
that varies is which rows form the batch. Three checks: the batch gradients of one epoch,
weighted by batch size, average to the full-batch gradient; the size of the noise against the
batch size, compared with the closed form for sampling without replacement; and the noise left
after the moving average of Optimizer_Adam's first moment, for batches drawn independently and
for the disjoint batches of shuffled epochs.

Runs in float64 and does not call nnfs.init(). Needs NumPy and the nnfs package (spiral_data).
Takes about 5 seconds.
"""
import numpy as np
from nnfs.datasets import spiral_data

from network import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dense

np.random.seed(0)
X, y = spiral_data(samples=100, classes=3)
N = len(X)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
dense1.weights = np.random.randn(2, 64)             # unit scale, so that the gradients are not tiny
dense2.weights = 0.1 * np.random.randn(64, 3)


def gradient(rows):
    """The gradient of the mean loss of the given rows, all four arrays flattened into one vector."""
    dense1.forward(X[rows])
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.forward(dense2.output, y[rows])
    loss_activation.backward(loss_activation.output, y[rows])
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    return np.concatenate([a.ravel() for a in (dense1.dweights, dense1.dbiases, dense2.dweights, dense2.dbiases)])


full = gradient(np.arange(N))
print(f"float64, seed 0, {N} rows, {full.size} parameters, fixed weights")
print(f"length of the full-batch gradient: {np.linalg.norm(full):.4f}")

print()
print("== One epoch of batches of 32: the weighted mean of the batch gradients")
idx = np.random.permutation(N)
total = np.zeros_like(full)
plain = np.zeros_like(full)
n_batches = 0
for start in range(0, N, 32):
    rows = idx[start:start + 32]
    g = gradient(rows)
    total += len(rows) * g
    plain += g
    n_batches += 1
print(f"largest difference from the full-batch gradient, weighted by batch size: {np.abs(total / N - full).max():.1e}")
print(f"largest difference from the full-batch gradient, plain mean of {n_batches}:    {np.abs(plain / n_batches - full).max():.1e}")

print()
print("== Noise against batch size: root-mean-square length of (batch gradient - full gradient)")
per_sample = np.stack([gradient(np.array([i])) for i in range(N)])
sigma = np.sqrt(np.sum(np.var(per_sample, axis=0)))          # spread of the single-row gradients
print(f"spread of the {N} single-row gradients, sigma: {sigma:.4f}")
print("   B  measured  sigma/sqrt(B)  with the factor sqrt((N-B)/(N-1))  measured / length of full gradient")
DRAWS = 2000
for batch_size in (1, 8, 32, 100, 128, 300):
    squared = 0.0
    for _ in range(DRAWS):
        rows = np.random.permutation(N)[:batch_size]
        squared += np.sum((per_sample[rows].mean(axis=0) - full) ** 2)
    measured = np.sqrt(squared / DRAWS)
    simple = sigma / np.sqrt(batch_size)
    exact = simple * np.sqrt((N - batch_size) / (N - 1))
    print(f"{batch_size:4d}  {measured:8.4f}  {simple:13.4f}  {exact:35.4f}  {measured / np.linalg.norm(full):10.2f}")

print()
print("== The moving average of Adam's first moment, beta_1 = 0.9, batches of 32, fixed weights")
beta_1 = 0.9
STEPS = 20000
momentum = np.zeros_like(full)
squared_raw, squared_avg, counted = 0.0, 0.0, 0
for step in range(1, STEPS + 1):
    rows = np.random.permutation(N)[:32]
    g = per_sample[rows].mean(axis=0)
    momentum = beta_1 * momentum + (1 - beta_1) * g
    if step > 200:                                           # the start-up of the average is left out
        squared_raw += np.sum((g - full) ** 2)
        squared_avg += np.sum((momentum - full) ** 2)
        counted += 1
ratio = np.sqrt(squared_avg / squared_raw)
print(f"noise of the averaged gradient / noise of one batch gradient: {ratio:.4f}")
print(f"closed form for independent batches, sqrt((1 - beta_1) / (1 + beta_1)): {np.sqrt((1 - beta_1) / (1 + beta_1)):.4f}")

print()
print("== The same average over shuffled epochs of batches of 32, cut as the training loop cuts them")
momentum = np.zeros_like(full)
squared_raw, squared_avg, step = 0.0, 0.0, 0
while step < STEPS:
    idx = np.random.permutation(N)                           # one epoch: ten disjoint batches, the last of 12 rows
    for start in range(0, N, 32):
        g = per_sample[idx[start:start + 32]].mean(axis=0)
        momentum = beta_1 * momentum + (1 - beta_1) * g
        step += 1
        if step > 200:
            squared_raw += np.sum((g - full) ** 2)
            squared_avg += np.sum((momentum - full) ** 2)
print(f"noise of the averaged gradient / noise of one batch gradient: {np.sqrt(squared_avg / squared_raw):.4f}")
