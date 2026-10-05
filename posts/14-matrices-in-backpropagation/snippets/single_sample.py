"""Post 14, sections 2 to 5: one sample through the layer of post 13.

The twelve weight gradients of post 13 come out of one matrix product, and the three bias
gradients are the upstream gradient itself. Needs NumPy only.
"""
import numpy as np

# The layer of post 13: three neurons over four inputs, one row of weights per neuron.
X = np.array([[1.0, 2.0, 3.0, 4.0]])               # shape (1, 4): one sample, four inputs
weights = np.array([[0.1, 0.2, 0.3, 0.4],          # neuron 1
                    [0.5, 0.6, 0.7, 0.8],          # neuron 2
                    [0.9, 1.0, 1.1, 1.2]])         # neuron 3, shape (3, 4)
biases = np.array([0.1, 0.2, 0.3])                 # shape (3,)

# Forward pass: the three outputs are summed and the sum is squared.
Z = X @ weights.T + biases                         # shape (1, 3)
A = np.maximum(0, Z)                               # ReLU
Y = np.sum(A)
L = Y ** 2

# The upstream gradient: three chain-rule factors per neuron.
dL_dY = 2 * Y                                      # scalar
dY_dA = np.ones_like(A)                            # the sum passes 1 to each neuron
dA_dZ = np.where(Z > 0, 1.0, 0.0)                  # the ReLU gate
dL_dZ = dL_dY * dY_dA * dA_dZ                      # shape (1, 3)

print("Z:", Z, " Y:", round(Y, 6), " L:", round(L, 6))
print("dL_dZ:", dL_dZ, dL_dZ.shape)

dL_dW = dL_dZ.T @ X                                # (3, 1) @ (1, 4) = (3, 4)
print(dL_dW)

# The same twelve numbers, one chain-rule expression per weight, as post 13 wrote them.
per_weight = np.zeros((3, 4))
for k in range(3):                                 # neuron
    for j in range(4):                             # input
        per_weight[k, j] = dL_dZ[0, k] * X[0, j]
print("shape:", dL_dW.shape, "equals weights.shape:", dL_dW.shape == weights.shape)
print("identical to the twelve per-weight products:", np.array_equal(dL_dW, per_weight))
print("identical to post 13's reshape and broadcast:",
      np.array_equal(dL_dW, dL_dZ.reshape(-1, 1) * X[0]))

# Outer product against dot product, on two vectors of the same length.
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print("dot product:", np.dot(a, b))
print("outer product:")
print(np.outer(a, b))
print("np.outer(dL_dZ, X) gives the same matrix:", np.array_equal(np.outer(dL_dZ, X), dL_dW))

# Bias gradients: the input of a bias is the constant 1.
dL_db = dL_dZ.flatten()                            # shape (3,), the shape of biases
print("dL_db:", dL_db, dL_db.shape)
print("np.sum(dL_dZ, axis=0):", np.sum(dL_dZ, axis=0))
