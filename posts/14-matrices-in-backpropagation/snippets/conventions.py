"""Post 14, section 8: the two weight layouts give the same gradient, transposed.

Posts 01 to 03, 13 and 14 store one row of weights per neuron; `Layer_Dense` (post 04
onward) stores one column per neuron. Needs NumPy only.
"""
import numpy as np

X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])          # shape (3, 4)
dL_dZ = np.array([[1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [3.0, 3.0, 3.0]])               # shape (3, 3)

W_rows = np.array([[0.1, 0.2, 0.3, 0.4],
                   [0.5, 0.6, 0.7, 0.8],
                   [0.9, 1.0, 1.1, 1.2]])         # shape (3, 4): one row per neuron
W_cols = W_rows.T.copy()                          # shape (4, 3): one column per neuron
b_flat = np.array([0.1, 0.2, 0.3])                # shape (3,)
b_row = b_flat.reshape(1, -1)                     # shape (1, 3), as Layer_Dense stores it

# Forward pass in each layout.
Z_rows = X @ W_rows.T + b_flat
Z_cols = X @ W_cols + b_row
print("forward passes agree:", np.array_equal(Z_rows, Z_cols), Z_rows.shape)

# Weight gradient in each layout.
dW_rows = dL_dZ.T @ X                             # (3, 4), the shape of W_rows
dW_cols = X.T @ dL_dZ                             # (4, 3), the shape of W_cols
print("dW_rows", dW_rows.shape, "matches W_rows", W_rows.shape)
print("dW_cols", dW_cols.shape, "matches W_cols", W_cols.shape)
print(dW_cols)
print("dW_cols is the transpose of dW_rows:", np.array_equal(dW_cols, dW_rows.T))

# Bias gradient in each layout.
db_flat = np.sum(dL_dZ, axis=0)                   # (3,)
db_row = np.sum(dL_dZ, axis=0, keepdims=True)     # (1, 3)
print("db_flat", db_flat, db_flat.shape)
print("db_row ", db_row, db_row.shape)

# One update step in each layout leaves the two layers identical.
lr = 0.01
W_rows_new = W_rows - lr * dW_rows
W_cols_new = W_cols - lr * dW_cols
print("updated layers agree:", np.array_equal(W_cols_new, W_rows_new.T))
