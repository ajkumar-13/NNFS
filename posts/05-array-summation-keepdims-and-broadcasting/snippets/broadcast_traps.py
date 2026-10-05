"""Post 05, section 8: the ways a reduction or a broadcast goes wrong, each one run.

Run from the series root:
    python posts/05-array-summation-keepdims-and-broadcasting/snippets/broadcast_traps.py

Needs only NumPy. Nothing here is random, so the output is the same on every run.
"""
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("== A column meets a 1-D array: both stretch, and the result is a square")
y_pred = np.array([[0.9], [0.2], [0.8], [0.4]])    # shape (4, 1): one output per sample
y_true = np.array([1.0, 0.0, 1.0, 0.0])            # shape (4,):   one label per sample
diff = y_pred - y_true                             # (4, 1) with (4,) gives (4, 4)
print(diff.shape, f"{np.mean(diff ** 2):.4f}")
diff = y_pred - y_true.reshape(-1, 1)              # (4, 1) with (4, 1) gives (4, 1)
print(diff.shape, f"{np.mean(diff ** 2):.4f}")

print("== np.add with out= still broadcasts; out only has to hold the result")
row = np.array([10, 20, 30])                       # shape (3,)
out = np.empty((3, 3))
np.add(a, row, out=out)
print(out)

print("== The guard that works: assert the shape that was meant")
max_vals = np.max(a, axis=1)
try:
    assert max_vals.shape == (3, 1), f"expected a column of shape (3, 1), got {max_vals.shape}"
except AssertionError as err:
    print("AssertionError:", err)

print("== Operand order does not change the result shape, but in-place needs room on the left")
print((a + row).shape, (row + a).shape, np.array_equal(a + row, row + a))
try:
    row += a                                       # row would have to grow from (3,) to (3, 3)
except ValueError as err:
    print("ValueError:", err)

print("== A (3,) bias cannot take a (1, 3) gradient in place; a (1, 3) bias can")
dvalues = np.ones((300, 3))                              # a stand-in for the gradient that reaches a layer
dbiases = np.sum(dvalues, axis=0, keepdims=True)         # shape (1, 3), as post 16 computes it
biases = np.zeros(3)                                     # shape (3,)
try:
    biases -= 0.001 * dbiases
except ValueError as err:
    print("ValueError:", err)
biases = np.zeros((1, 3))                                # shape (1, 3), as Layer_Dense stores it
biases -= 0.001 * dbiases
print(biases, biases.shape)

print("== The star operator is element-wise; np.dot is the matrix product")
print(a * a)
print(np.dot(a, a))

print("== What print shows and what .shape shows")
print(np.sum(a, axis=0), np.sum(a, axis=0).shape)
print(np.sum(a, axis=0, keepdims=True), np.sum(a, axis=0, keepdims=True).shape)

print("== keepdims adds an axis, not data; mean, max and std take the same two arguments")
print(np.sum(a, axis=1).size, np.sum(a, axis=1, keepdims=True).size)
print(np.mean(a, axis=1).shape, np.max(a, axis=1).shape, np.std(a, axis=1).shape)
print(np.mean(a, axis=1, keepdims=True).shape, np.max(a, axis=1, keepdims=True).shape, np.std(a, axis=1, keepdims=True).shape)

print("== An axis the array does not have is an error, not a silent result")
try:
    np.sum(a, axis=2)
except ValueError as err:
    print(type(err).__name__ + ":", err)
