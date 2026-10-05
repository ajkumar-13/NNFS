"""Post 15, sections 5, 6, 7.2 and 7.3: the input gradient of a batch, and the hand-off.

Run from the series root:
    python posts/15-gradients-with-respect-to-inputs/snippets/batch_and_handoff.py

Contents: the input gradient of the layer of posts 13 and 14 for the three-sample batch of
post 14, one row per sample, checked against a central difference on all twelve inputs; the
three lines of a dense layer's backward pass in the layout Layer_Dense stores, each checked
against a central difference; and a second layer placed after the first, whose input
gradient is what the first layer needs before it can compute its own weight gradient.

Needs only NumPy. Nothing here is random, and every array is float64.
"""
import numpy as np

# The three-sample batch of post 14 and the layer of posts 13 and 14.
X = np.array([[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])          # (3, 4): N = 3 samples
weights = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]])      # (3, 4): one row per neuron
W = weights.T                                   # (4, 3): the layout Layer_Dense stores
b = np.array([[0.1, 0.2, 0.3]])                 # (1, 3)
N = len(X)


def central_difference(loss, array, h=1e-5):
    """Central difference of loss() along every entry of array, changed in place and restored."""
    grad = np.zeros_like(array)
    for index in np.ndindex(*array.shape):
        saved = array[index]
        array[index] = saved + h
        plus = loss()
        array[index] = saved - h
        minus = loss()
        array[index] = saved
        grad[index] = (plus - minus) / (2 * h)
    return grad


def gap(a, b):
    return np.max(np.abs(a - b))


def one_layer_loss():
    """Each sample's Y is the sum of its ReLU outputs; the loss is the mean of Y squared."""
    Z = np.dot(X, W) + b
    Y = np.sum(np.maximum(0, Z), axis=1, keepdims=True)
    return np.mean(Y ** 2)


# Forward pass, then the upstream gradient: one row per sample, one column per neuron.
Z = np.dot(X, W) + b                            # (3, 3)
Y = np.sum(np.maximum(0, Z), axis=1, keepdims=True)     # (3, 1)
dL_dZ = 2 * Y / N * (Z > 0)                     # (3, 3): the 1/N of the mean is in here

# Section 5: the input gradient of the whole batch is one product, in either layout.
dL_dX = dL_dZ @ weights                         # (3, 3) @ (3, 4) = (3, 4)

# Section 6: the three lines of the backward pass, in the layout Layer_Dense stores.
inputs, dvalues = X, dL_dZ
dweights = np.dot(inputs.T, dvalues)                    # (4, 3), the shape of W
dbiases = np.sum(dvalues, axis=0, keepdims=True)        # (1, 3), the shape of b
dinputs = np.dot(dvalues, W.T)                          # (3, 4), the shape of inputs

print("== Section 5: a batch of three samples through the layer of posts 13 and 14")
print("Y per sample:", ", ".join(f"{y:.2f}" for y in Y[:, 0]), f"  L = mean of Y^2 = {one_layer_loss():.4f}")
print("upstream gradient dL/dZ, shape", dL_dZ.shape)
print(np.round(dL_dZ, 4))
print("input gradient dL_dZ @ weights, shape", dL_dX.shape)
print(np.round(dL_dX, 4))
print(f"largest gap to a central difference on all {X.size} inputs: {gap(dL_dX, central_difference(one_layer_loss, X)):.1e}")
alone = np.vstack([dL_dZ[i:i + 1] @ weights for i in range(N)])
print(f"largest gap to the three rows computed one sample at a time: {gap(dL_dX, alone):.1e}")

print()
print("== Section 6: the three lines, in the layout Layer_Dense stores (W has shape (4, 3))")
print(f"dweights = np.dot(inputs.T, dvalues)                 shape {dweights.shape}   gap to central difference {gap(dweights, central_difference(one_layer_loss, W)):.1e}")
print(f"dbiases  = np.sum(dvalues, axis=0, keepdims=True)    shape {dbiases.shape}   gap to central difference {gap(dbiases, central_difference(one_layer_loss, b)):.1e}")
print(f"dinputs  = np.dot(dvalues, W.T)                      shape {dinputs.shape}   gap to central difference {gap(dinputs, central_difference(one_layer_loss, X)):.1e}")
print(f"dinputs against dL_dZ @ weights: gap {gap(dinputs, dL_dX):.1e}")
print(f"dweights against the one-row-per-neuron form (dL_dZ.T @ X).T: gap {gap(dweights, (dL_dZ.T @ X).T):.1e}")

# Section 7.3: a second dense layer after the first one's ReLU. Its two neurons are summed
# into Y, and the loss is again the mean of Y squared.
W2 = np.array([[0.2, -0.4],
               [-0.5, 0.1],
               [0.3, 0.6]])                     # (3, 2) = (n_inputs, n_neurons)
b2 = np.array([[0.05, -0.05]])                  # (1, 2)


def two_layer_loss():
    A1 = np.maximum(0, np.dot(X, W) + b)        # layer 1 and its ReLU
    Z2 = np.dot(A1, W2) + b2                    # layer 2, no activation
    return np.mean(np.sum(Z2, axis=1) ** 2)


Z1 = np.dot(X, W) + b
A1 = np.maximum(0, Z1)
Z2 = np.dot(A1, W2) + b2
Y2 = np.sum(Z2, axis=1, keepdims=True)

dvalues2 = 2 * Y2 / N * np.ones_like(Z2)        # (3, 2): what the loss hands to layer 2
dweights2 = np.dot(A1.T, dvalues2)              # (3, 2): stays in layer 2
dinputs2 = np.dot(dvalues2, W2.T)               # (3, 3): leaves layer 2

dvalues1 = dinputs2 * (Z1 > 0)                  # through layer 1's ReLU gate
dweights1 = np.dot(X.T, dvalues1)               # (4, 3): layer 1 can now learn
dinputs1 = np.dot(dvalues1, W.T)                # (3, 4): the gradient at the data

print()
print("== Section 7.3: two layers, and what travels between them")
print(f"layer 2 receives dvalues of shape {dvalues2.shape} and passes back dinputs of shape {dinputs2.shape}")
print(f"layer 2 weight gradient: gap to central difference {gap(dweights2, central_difference(two_layer_loss, W2)):.1e}")
print(f"layer 1 weight gradient, built from layer 2's dinputs: gap to central difference {gap(dweights1, central_difference(two_layer_loss, W)):.1e}")
print(f"layer 1 input gradient, shape {dinputs1.shape}: gap to central difference {gap(dinputs1, central_difference(two_layer_loss, X)):.1e}")
print(f"smallest |Z1| in the batch: {np.min(np.abs(Z1)):.2f}, so no finite-difference step crosses a ReLU corner")
