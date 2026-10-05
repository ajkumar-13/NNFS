import numpy as np

W = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(W.shape, W.T.shape)
print(W.T)
print(np.shares_memory(W, W.T))     # True: .T is a view of the same numbers
print(np.array([1, 2, 3]).T.shape)  # a 1-D array has no second axis to swap

rows = [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]            # a plain Python list of lists

try:
    rows.T
except AttributeError as error:
    print("AttributeError:", error)
print(np.array(rows).T.shape)       # an array has .T

print([1, 2] + [3, 4])              # lists concatenate
print([1, 2] * 3)                   # lists repeat
print(np.array([1, 2]) + np.array([3, 4]))
print(np.array([1, 2]) * 3)
