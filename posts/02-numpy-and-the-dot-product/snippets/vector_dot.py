import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]

print(np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32
print(np.dot(b, a))   # 4*1 + 5*2 + 6*3 = 32
print(np.inner(a, b), np.array(a) @ np.array(b))   # two other spellings of the same sum
