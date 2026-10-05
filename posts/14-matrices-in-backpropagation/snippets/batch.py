"""Post 14, section 6: the same two lines on a batch of three samples.

The matrix product sums the per-sample outer products, so the gradient keeps the shape of
the weights whatever the batch size. Needs NumPy only.
"""
import numpy as np

X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])     # shape (3, 4): three samples, four inputs

dL_dZ = np.array([[1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [3.0, 3.0, 3.0]])          # shape (3, 3): one row per sample, one column per neuron

dL_dW = dL_dZ.T @ X                # (3, 3) @ (3, 4) = (3, 4)
dL_db = np.sum(dL_dZ, axis=0)      # shape (3,)

print("Weight gradients:")
print(dL_dW)
print("Bias gradients:", dL_db)

# The same result built one sample at a time: an outer product per sample, then a sum.
total = np.zeros((3, 4))
for i in range(len(X)):
    one_sample = np.outer(dL_dZ[i], X[i])    # what section 4 computed for a single sample
    print(f"sample {i + 1}, row of neuron 1:", one_sample[0])
    total += one_sample
print("sum of the three, row of neuron 1:", total[0])
print("largest difference from the matrix product:", np.max(np.abs(total - dL_dW)))

# One entry by hand: neuron 1, input 2.
by_hand = 1.0 * 2.0 + 2.0 * 5.0 + 3.0 * 2.7
print("dL_dW[0, 1] by hand:", round(by_hand, 10))
