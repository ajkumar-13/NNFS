"""Post 01, section 10: the shape diary, checked on the arrays of this post.

Run from the series root:  python posts/01-neurons-and-layers/snippets/shape_diary.py
"""
import numpy as np

x = np.array([1.0, 2.0, 3.0, 2.5])              # one sample: 4 inputs
X = np.array([[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])          # a batch: 3 samples, one per row
w = np.array([0.2, 0.8, -0.5, 1.0])             # one neuron: 4 weights
W = np.array([[0.2, 0.8, -0.5, 1.0],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]])      # a layer: 3 neurons, one row of weights each
b = 2.0                                         # one neuron: 1 bias
biases = np.array([2.0, 3.0, 0.5])              # a layer: 1 bias per neuron

print("setting                    inputs   weights  biases   output")
rows = [
    ("one neuron, one sample", x, w, b, np.dot(w, x) + b),
    ("layer of 3, one sample", x, W, biases, np.dot(W, x) + biases),
    ("layer of 3, batch of 3", X, W, biases, np.dot(X, W.T) + biases),
    ("layer of 3, batch of 2", X[:2], W, biases, np.dot(X[:2], W.T) + biases),
]
for setting, inputs, weights, bias, output in rows:
    shapes = [str(np.shape(array)) for array in (inputs, weights, bias, output)]
    print(f"{setting:26s} {shapes[0]:8s} {shapes[1]:8s} {shapes[2]:8s} {shapes[3]}")

print()
print("a batch of 2 through the layer of 3, one step at a time")
batch = X[:2]
steps = [
    ("batch", batch),
    ("W", W),
    ("W.T", W.T),
    ("np.dot(batch, W.T)", np.dot(batch, W.T)),
    ("biases", biases),
    ("np.dot(batch, W.T) + biases", np.dot(batch, W.T) + biases),
]
for name, array in steps:
    print(f"  {name:28s} {np.shape(array)}")

print()
print(f"parameters in the layer: {W.size} weights + {biases.size} biases = {W.size + biases.size}")
