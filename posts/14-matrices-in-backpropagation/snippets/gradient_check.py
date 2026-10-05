"""Post 14, section 9: the two-line backward pass checked against central differences.

The layer of post 13 on the batch of section 6.1, with a loss that is the mean over the batch
of the squared output sum. Float64 and h = 1e-5, as post 10 recommends. Needs NumPy only.
"""
import numpy as np

X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])     # shape (3, 4)
weights = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]])   # shape (3, 4): one row per neuron
biases = np.array([0.1, 0.2, 0.3])           # shape (3,)
N = len(X)


def loss(weights, biases):
    Z = X @ weights.T + biases               # shape (N, 3)
    A = np.maximum(0, Z)
    Y = np.sum(A, axis=1)                    # one output per sample
    return np.mean(Y ** 2)


# Forward pass and the upstream gradient, one row per sample.
Z = X @ weights.T + biases
A = np.maximum(0, Z)
Y = np.sum(A, axis=1, keepdims=True)         # shape (N, 1)
dL_dZ = (2 * Y / N) * np.where(Z > 0, 1.0, 0.0)   # shape (N, 3)

# The two-line backward pass.
dL_dW = dL_dZ.T @ X              # weight gradient: (m, n)
dL_db = np.sum(dL_dZ, axis=0)    # bias gradient:   (m,)

# Central differences, one parameter at a time.
h = 1e-5
num_dW = np.zeros_like(weights)
for k in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        plus, minus = weights.copy(), weights.copy()
        plus[k, j] += h
        minus[k, j] -= h
        num_dW[k, j] = (loss(plus, biases) - loss(minus, biases)) / (2 * h)
num_dB = np.zeros_like(biases)
for k in range(len(biases)):
    plus, minus = biases.copy(), biases.copy()
    plus[k] += h
    minus[k] -= h
    num_dB[k] = (loss(weights, plus) - loss(weights, minus)) / (2 * h)


def relative_error(analytic, numeric):
    return np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), np.abs(numeric)))


print("loss:", round(loss(weights, biases), 6))
print("dL_dZ:")
print(dL_dZ)
print("dL_dW, matrix form:")
print(dL_dW)
print("dL_dW, central differences:")
print(num_dW)
print("dL_db, matrix form:        ", dL_db)
print("dL_db, central differences:", num_dB)
print(f"largest relative error, weights: {relative_error(dL_dW, num_dW):.1e}")
print(f"largest relative error, biases:  {relative_error(dL_db, num_dB):.1e}")
