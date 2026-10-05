"""Post 15, section 8: five ways in which an input gradient goes wrong, or is checked wrongly.

Run from the series root:
    python posts/15-gradients-with-respect-to-inputs/snippets/what_can_go_wrong.py

Contents: a closed ReLU gate that removes one path from every sum, and a layer with every
gate closed; the transpose placed wrongly, which raises an error in a layer with different
numbers of inputs and neurons and returns wrong numbers silently in a square one; the input
gradient summed over the batch; the division by N applied a second time; and the central
difference check repeated in float32.

Needs only NumPy. Nothing here is random.
"""
import numpy as np

X = np.array([[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])          # the three-sample batch of post 14
weights = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2]])      # (3, 4): one row per neuron
b = np.array([[0.1, 0.2, 0.3]])


def loss(X, W, b):
    """Layer_Dense layout: W is (n_inputs, n_neurons). Mean over the batch of Y squared."""
    Y = np.sum(np.maximum(0, np.dot(X, W) + b), axis=1, keepdims=True)
    return np.mean(Y ** 2)


def backward_inputs(X, W, b):
    """The upstream gradient dL/dZ and the input gradient dL/dX of the loss above."""
    Z = np.dot(X, W) + b
    Y = np.sum(np.maximum(0, Z), axis=1, keepdims=True)
    dvalues = 2 * Y / len(X) * (Z > 0)
    return dvalues, np.dot(dvalues, W.T)


def numerical_inputs(X, W, b, h=1e-5):
    """Central difference of the loss along every input, in the dtype of X."""
    grad = np.zeros_like(X)
    h = X.dtype.type(h)
    for index in np.ndindex(*X.shape):
        step = np.zeros_like(X)
        step[index] = h
        grad[index] = (loss(X + step, W, b) - loss(X - step, W, b)) / (2 * h)
    return grad


def gap(a, b):
    return np.max(np.abs(a - b))


W = weights.T                                   # (4, 3)
one = X[:1]                                     # the first sample alone, shape (1, 4)

print("== A closed ReLU gate removes a path: neuron 2's weights negated, first sample only")
W_closed = W.copy()
W_closed[:, 1] = -W_closed[:, 1]
dvalues, dinputs = backward_inputs(one, W_closed, b)
print("pre-activations Z:", np.round(np.dot(one, W_closed) + b, 2)[0])
print("upstream gradient:", np.round(dvalues, 2)[0])
print("input gradient:   ", np.round(dinputs, 2)[0], f"  gap to central difference {gap(dinputs, numerical_inputs(one, W_closed, b)):.1e}")
dvalues_open, dinputs_open = backward_inputs(one, W, b)
print("with the gate open:", np.round(dinputs_open, 2)[0])
dvalues_dead, dinputs_dead = backward_inputs(one, -W, -b)
print("every gate closed (all weights and biases negated): input gradient", dinputs_dead[0])

print()
print("== The transpose in the wrong place")
dvalues, dinputs = backward_inputs(X, W, b)
try:
    np.dot(dvalues, W)
except ValueError as error:
    print("4 inputs, 3 neurons, np.dot(dvalues, W):", error)
W_square = weights[:, :3].T.copy()              # a layer with 3 inputs and 3 neurons
X_square = X[:, :3].copy()
dvalues_sq, dinputs_sq = backward_inputs(X_square, W_square, b)
wrong = np.dot(dvalues_sq, W_square)
numeric_sq = numerical_inputs(X_square, W_square, b)
print(f"3 inputs, 3 neurons, np.dot(dvalues, W.T): shape {dinputs_sq.shape}, gap to central difference {gap(dinputs_sq, numeric_sq):.1e}")
print(f"3 inputs, 3 neurons, np.dot(dvalues, W):   shape {wrong.shape}, gap to central difference {gap(wrong, numeric_sq):.1e}")
print("first row, correct:", np.round(dinputs_sq[0], 2), "  first row, wrong:", np.round(wrong[0], 2))

print()
print("== The input gradient summed over the batch")
summed = np.sum(dinputs, axis=0)
print(f"per sample: shape {dinputs.shape}; after np.sum(dinputs, axis=0): shape {summed.shape}")
print("rows:  ", ", ".join(str(np.round(r, 2)) for r in dinputs))
print("summed:", np.round(summed, 2))

print()
print("== The division by N applied a second time")
twice = dinputs / len(X)
numeric = numerical_inputs(X, W, b)
print(f"correct:        gap to central difference {gap(dinputs, numeric):.1e}")
print(f"divided again:  gap to central difference {gap(twice, numeric):.1e}; every entry is 1/{numeric[0, 0] / twice[0, 0]:.0f} of the true one")

print()
print("== The same check in float32, h = 1e-5 and h = 1e-2")
X32, W32, b32 = X.astype(np.float32), W.astype(np.float32), b.astype(np.float32)
analytic32 = backward_inputs(X32, W32, b32)[1]
for h in (1e-5, 1e-2):
    print(f"float32, h = {h:.0e}: gap to central difference {gap(analytic32, numerical_inputs(X32, W32, b32, h)):.1e}")
print(f"float64, h = 1e-05: gap to central difference {gap(dinputs, numeric):.1e}")
print(f"largest entry of the gradient: {np.max(np.abs(dinputs)):.2f}")
