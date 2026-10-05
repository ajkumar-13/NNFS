"""Post 05, sections 2 and 3: reductions along an axis, keepdims, and the per-row max bug.

Run from the series root:
    python posts/05-array-summation-keepdims-and-broadcasting/snippets/axis_keepdims.py

Needs only NumPy. Nothing here is random, so the output is the same on every run.
"""
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("== Section 2: np.sum with no axis, axis=0 and axis=1")
print(np.sum(a))             # 45
print(np.sum(a, axis=None))  # 45 (same thing)
print(np.sum(a, axis=0))     # [12 15 18]
print(np.sum(a, axis=1))     # [ 6 15 24]
print(np.sum(a, axis=0).shape, np.sum(a, axis=1).shape)     # (3,) (3,)

print("== Section 2.4: the named axis disappears, whatever the number of dimensions")
b = np.ones((4, 2, 3))               # shape (4, 2, 3)
print(np.sum(b, axis=0).shape)       # (2, 3): axis 0 is gone
print(np.sum(b, axis=1).shape)       # (4, 3): axis 1 is gone
print(np.sum(b, axis=2).shape)       # (4, 2): axis 2 is gone
print(np.sum(b, axis=-1).shape)      # (4, 2): axis -1 is the last axis

print("== Section 3: keepdims=True keeps the reduced axis with size 1")
print(np.sum(a, axis=0, keepdims=True))     # one row:    shape (1, 3)
print(np.sum(a, axis=1, keepdims=True))     # one column: shape (3, 1)
print(np.sum(a, axis=0, keepdims=True).shape, np.sum(a, axis=1, keepdims=True).shape)

print("== Section 3.1: subtract the per-row max, without keepdims (wrong, and no error)")
max_vals = np.max(a, axis=1)        # [3 6 9], shape (3,)
print(a - max_vals)

print("== Section 3.1: subtract the per-row max, with keepdims=True (correct)")
max_vals = np.max(a, axis=1, keepdims=True)  # a column, shape (3, 1)
print(a - max_vals)

print("== Section 3.2: the same mistake on an array that is not square raises an error")
wide = np.arange(12).reshape(4, 3)           # 4 rows, 3 columns
try:
    wide - np.max(wide, axis=1)              # shapes (4, 3) and (4,)
except ValueError as err:
    print("ValueError:", str(err).strip())
print(wide - np.max(wide, axis=1, keepdims=True))

print("== Section 3.2: the same line on every shape from (1, 1) to (4, 4)")
print("        1 col   2 cols  3 cols  4 cols")
for rows in (1, 2, 3, 4):
    cells = []
    for cols in (1, 2, 3, 4):
        x = np.arange(rows * cols).reshape(rows, cols) ** 2      # every row has a different max
        correct = x - np.max(x, axis=1, keepdims=True)
        try:
            result = x - np.max(x, axis=1)
        except ValueError:
            cells.append("error")
            continue
        same = result.shape == correct.shape and np.array_equal(result, correct)
        cells.append("right" if same else "wrong")
    print(f"{rows} row{'s' if rows > 1 else ' '}  " + "   ".join(cells))

print("== Section 3.2: dividing each row by its sum, checked by the row sums")
wrong = a / np.sum(a, axis=1)                    # divides column j by the sum of row j
right = a / np.sum(a, axis=1, keepdims=True)     # divides row i by the sum of row i
print(np.round(np.sum(wrong, axis=1), 3))        # [0.425 1.25  2.075]
print(np.round(np.sum(right, axis=1), 3))        # [1. 1. 1.]
