"""Post 32, sections 2 and 7: the wall-clock time of one epoch against the batch size.

Run from the series root:
    python posts/32-mini-batching/snippets/timing.py

Random data in the shape of a small image problem: 6,400 rows of 784 float32 values, ten
classes, Layer_Dense(784, 128), ReLU, Layer_Dense(128, 10), Optimizer_Adam. Nothing is learned
(the labels are random); only the clock is read. Each batch size runs three epochs and the
fastest is printed, except batch size 1, which runs one. The times depend on the machine and
on what else it is doing, so the figures differ from run to run; the pattern is what the post
uses.

Does not call nnfs.init(). Needs NumPy only. Takes about 20 seconds.
"""
import time

import numpy as np

from network import (Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Layer_Dense,
                     Optimizer_Adam, train)

N_ROWS = 6400
np.random.seed(0)
X = np.random.randn(N_ROWS, 784).astype(np.float32)
y = np.random.randint(0, 10, size=N_ROWS)

print(f"{N_ROWS:,} rows of 784 float32 values, Layer_Dense(784, 128), ReLU, Layer_Dense(128, 10), Adam; fastest of 3 epochs, 1 epoch at batch size 1")
print("batch size  updates per epoch  seconds per epoch  microseconds per row  milliseconds per update")
for batch_size in (1, 8, 32, 64, 100, 128, 200, 256, 6400):
    np.random.seed(0)
    dense1, activation1 = Layer_Dense(784, 128), Activation_ReLU()
    dense2, loss_activation = Layer_Dense(128, 10), Activation_Softmax_Loss_CategoricalCrossentropy()
    dense1.weights = dense1.weights.astype(np.float32)
    dense1.biases = dense1.biases.astype(np.float32)
    dense2.weights = dense2.weights.astype(np.float32)
    dense2.biases = dense2.biases.astype(np.float32)
    optimizer = Optimizer_Adam(learning_rate=0.001)
    best = np.inf
    for _ in range(1 if batch_size == 1 else 3):
        started = time.perf_counter()
        train(X, y, dense1, activation1, dense2, loss_activation, optimizer, epochs=1, batch_size=batch_size)
        best = min(best, time.perf_counter() - started)
    updates = -(-N_ROWS // batch_size)
    print(f"{batch_size:10,d}  {updates:17,d}  {best:17.3f}  {1e6 * best / N_ROWS:20.1f}  {1e3 * best / updates:23.3f}", flush=True)
