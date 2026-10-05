import numpy as np

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])    # shape (3, 4)

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])       # shape (4, 3)

print(np.dot(A, B))                 # (3, 4) with (4, 3): inner 4s meet, result (3, 3)
print(np.dot(B, A).shape)           # (4, 3) with (3, 4): inner 3s meet, result (4, 4)
