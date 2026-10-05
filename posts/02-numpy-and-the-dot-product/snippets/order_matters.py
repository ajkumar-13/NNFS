import numpy as np

a = np.array([1, 2, 3])

B = np.array([[ 4,  5,  6],
              [ 7,  8,  9],
              [10, 11, 12]])

print(np.dot(a, B))   # vector first: one dot product per column of B
print(np.dot(B, a))   # matrix first: one dot product per row of B

C = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3): no longer square

print(np.dot(C, a))            # (2, 3) with (3,): the 3s meet, result (2,)
try:
    np.dot(a, C)               # (3,) with (2, 3): 3 meets 2
except ValueError as error:
    print("ValueError:", error)
