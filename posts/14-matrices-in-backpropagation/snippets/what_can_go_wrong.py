"""Post 14, section 10: six ways the two-line backward pass goes wrong.

Each case prints what NumPy does, including the cases in which it raises nothing.
Needs NumPy only.
"""
import numpy as np

X1 = np.array([[1.0, 2.0, 3.0, 4.0]])              # one sample, shape (1, 4)
dZ1 = np.array([[43.2, 43.2, 43.2]])               # its upstream gradient, shape (1, 3)

X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])           # a batch, shape (3, 4)
dL_dZ = np.array([[1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [3.0, 3.0, 3.0]])                # its upstream gradient, shape (3, 3)
correct = dL_dZ.T @ X


def attempt(label, compute):
    try:
        result = compute()
        print(f"{label}: no error, shape {np.shape(result)}")
        return result
    except ValueError:
        print(f"{label}: ValueError")
        return None


print("== 1. The transpose is left out")
attempt("one sample, (1, 3) @ (1, 4)", lambda: dZ1 @ X1)
wrong = attempt("batch with N = m = 3, (3, 3) @ (3, 4)", lambda: dL_dZ @ X)
print("correct row of neuron 1:", correct[0])
print("wrong   row 1:          ", wrong[0])

print()
print("== 2. The upstream gradient is one-dimensional")
flat = dZ1.flatten()                               # shape (3,)
print("flat.T has shape", flat.T.shape, "so the transpose does nothing")
attempt("one sample, (3,) @ (1, 4)", lambda: flat.T @ X1)
attempt("batch with N = 3, (3,) @ (3, 4)", lambda: flat.T @ X)

print()
print("== 3. np.outer on a batch")
print("one sample: np.outer gives shape", np.outer(dZ1, X1).shape)
print("batch:      np.outer gives shape", np.outer(dL_dZ, X).shape, "and the product gives", correct.shape)

print()
print("== 4. The axis is left out of the bias sum")
print("np.sum(dL_dZ, axis=0):", np.sum(dL_dZ, axis=0))
print("np.sum(dL_dZ):        ", np.sum(dL_dZ))
print("np.sum(dL_dZ, axis=1):", np.sum(dL_dZ, axis=1), "(one number per sample, not per neuron)")

print()
print("== 5. The batch average is applied twice")
N = len(X)
mean_upstream = dL_dZ / N                          # the upstream of a loss that is a batch mean
print("row of neuron 1, mean loss:      ", (mean_upstream.T @ X)[0])
print("row of neuron 1, divided again:  ", ((mean_upstream.T @ X) / N)[0])

print()
print("== 6. The two weight layouts are mixed")
attempt("(3, 4) weights minus a (4, 3) gradient", lambda: np.ones((3, 4)) - 0.01 * (X.T @ dL_dZ))
X_square = X[:, :3]                                # three inputs and three neurons
rows_layout = dL_dZ.T @ X_square
cols_layout = X_square.T @ dL_dZ
print("square layer: both gradients have shape", rows_layout.shape, cols_layout.shape)
print("equal:", np.array_equal(rows_layout, cols_layout),
      " transposes of each other:", np.array_equal(rows_layout, cols_layout.T))
