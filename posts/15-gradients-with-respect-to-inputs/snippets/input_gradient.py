"""Post 15, sections 2 to 4 and 7.1: the input gradient of the layer of posts 13 and 14.

Run from the series root:
    python posts/15-gradients-with-respect-to-inputs/snippets/input_gradient.py

Contents: the forward pass and the upstream gradient of the three-neuron layer; the input
gradient as a sum over paths, one path per neuron, written as two loops; the same gradient as
one matrix product, in the one-row-per-neuron layout of posts 13 and 14 and in the layout
Layer_Dense stores; a central difference on every input as the check; and the change in the
loss when one input is nudged.

Needs only NumPy. Nothing here is random, and every array is float64.
"""
import numpy as np

# The layer of posts 13 and 14: four inputs, three neurons, one row of weights per neuron.
X = np.array([[1.0, 2.0, 3.0, 4.0]])            # (1, 4): one sample
weights = np.array([[0.1, 0.2, 0.3, 0.4],       # neuron 1
                    [0.5, 0.6, 0.7, 0.8],       # neuron 2
                    [0.9, 1.0, 1.1, 1.2]])      # neuron 3; shape (3, 4)
biases = np.array([[0.1, 0.2, 0.3]])            # (1, 3)


def forward(X):
    """Z = X W + b with ReLU, then Y = the sum of the activations and L = Y squared."""
    Z = X @ weights.T + biases                  # (1, 3): one pre-activation per neuron
    A = np.maximum(0, Z)                        # ReLU
    Y = np.sum(A)
    return Z, Y, Y ** 2


Z, Y, L = forward(X)
dL_dZ = 2 * Y * (Z > 0)                         # (1, 3): 2Y through each ReLU gate

# Section 2: one input at a time, one path per neuron, summed.
n_neurons, n_inputs = weights.shape
by_paths = np.zeros((1, n_inputs))
for j in range(n_inputs):
    for k in range(n_neurons):
        by_paths[0, j] += dL_dZ[0, k] * weights[k, j]

# Section 3: the same sums as one matrix product.
dL_dX = dL_dZ @ weights                         # (1, 3) @ (3, 4) = (1, 4)

# Section 3.1: Layer_Dense stores the transposed array, one column per neuron.
W = weights.T                                   # (4, 3) = (n_inputs, n_neurons)
dinputs = np.dot(dL_dZ, W.T)                    # (1, 3) . (3, 4) = (1, 4)


def numerical_input_gradient(X, h=1e-5):
    """Central difference of the loss along each input in turn (post 10, section 2.5)."""
    grad = np.zeros_like(X)
    for index in np.ndindex(*X.shape):
        step = np.zeros_like(X)
        step[index] = h                         # move one input, hold the others
        grad[index] = (forward(X + step)[2] - forward(X - step)[2]) / (2 * h)
    return grad


numeric = numerical_input_gradient(X)


def row(values, fmt="{:.2f}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


print("== Forward pass of the layer of posts 13 and 14")
print(f"Z = {row(Z[0], '{:.1f}')}   Y = {Y:.1f}   L = {L:.2f}")
print(f"upstream gradient dL/dZ = {row(dL_dZ[0], '{:.1f}')}   shape {dL_dZ.shape}")

print()
print("== Section 2: the input gradient as a sum over paths, one path per neuron")
for j in range(n_inputs):
    terms = " + ".join(f"{dL_dZ[0, k]:.1f} * {weights[k, j]:.1f}" for k in range(n_neurons))
    print(f"dL/dx{j + 1} = {terms} = {by_paths[0, j]:.2f}")
print(f"for contrast, a weight has one path: dL/dw11 = dL/dz1 * x1 = {dL_dZ[0, 0]:.1f} * {X[0, 0]:.0f} = {dL_dZ[0, 0] * X[0, 0]:.1f}")

print()
print("== Section 3: the same four sums as one matrix product")
print(f"dL_dZ @ weights          = {row(dL_dX[0])}   shape {dL_dZ.shape} @ {weights.shape} = {dL_dX.shape}")
print(f"np.dot(dL_dZ, W.T)       = {row(dinputs[0])}   with W = weights.T, shape {W.shape}")
print(f"largest gap, loops against matrix product: {np.max(np.abs(by_paths - dL_dX)):.1e}")
print(f"largest gap between the two layouts:       {np.max(np.abs(dinputs - dL_dX)):.1e}")

print()
print("== Section 7.1: central difference on each input, h = 1e-5")
for j in range(n_inputs):
    print(f"dL/dx{j + 1}   matrix product {dL_dX[0, j]:10.6f}   central difference {numeric[0, j]:10.6f}")
print(f"largest gap: {np.max(np.abs(numeric - dL_dX)):.1e}")

print()
print("== Reading one component: nudge x1 by +0.01, the other inputs held fixed")
nudged = X.copy()
nudged[0, 0] += 0.01
print(f"predicted change in L: dL/dx1 * 0.01 = {dL_dX[0, 0] * 0.01:.4f}")
print(f"actual change in L:    {forward(nudged)[2] - L:.4f}")
