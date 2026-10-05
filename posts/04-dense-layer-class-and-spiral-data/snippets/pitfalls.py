"""Post 04, section 8: each failure the post warns about, triggered on purpose.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/pitfalls.py

Every block prints the error message or the wrong result that the section quotes.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # Small random weights; zero biases.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Before nnfs.init(): the same seed, plain NumPy. Kept for block 6.
np.random.seed(0)
X_plain, y_plain = spiral_data(samples=100, classes=3)
plain = Layer_Dense(2, 3)
plain.forward(X_plain)

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense1.forward(X)

print("1. n_inputs set to the batch size (300) instead of the feature count (2)")
try:
    wrong = Layer_Dense(300, 3)
    wrong.forward(X)
except ValueError as error:
    print("   ValueError:", error)

print("2. weights left in the old (n_neurons, n_inputs) layout")
old_layout = dense1.weights.T.copy()            # (3, 2): one row per neuron
try:
    np.dot(X, old_layout)
except ValueError as error:
    print("   3 neurons over 2 inputs, ValueError:", error)
square = Layer_Dense(3, 3)                      # 3 inputs, 3 neurons: both layouts are (3, 3)
right = np.dot(dense1.output, square.weights)
wrong = np.dot(dense1.output, square.weights.T)
print("   square layer, no error; same shape:", right.shape == wrong.shape,
      "| same numbers:", np.allclose(right, wrong))

print("3. reading the return value of forward")
result = dense1.forward(X)
print("   forward returned:", result)
try:
    result[:5]
except TypeError as error:
    print("   TypeError:", error)

print("4. the weights depend on what was drawn before them")
np.random.seed(0)
spiral_data(samples=100, classes=3)             # the data uses 300 random numbers first
after_data = Layer_Dense(2, 3)
np.random.seed(0)
before_data = Layer_Dense(2, 3)                 # the layer takes the first six numbers
print("   layer created after the data, first weight row: ", after_data.weights[0])
print("   layer created before the data, first weight row:", before_data.weights[0])

print("5. two instances differ; the same seed replays the same weights")
np.random.seed(0)
first = Layer_Dense(2, 64)
second = Layer_Dense(2, 64)
np.random.seed(0)
replay = Layer_Dense(2, 64)
print("   weights", first.weights.shape, "biases", first.biases.shape)
print("   two instances in one run equal:", np.array_equal(first.weights, second.weights))
print("   same seed again equal:", np.array_equal(first.weights, replay.weights))

print("6. what nnfs.init() changes")
print("   without it:", X_plain.dtype, "data,", plain.output.dtype, "output, second row", plain.output[1])
print("   with it:   ", X.dtype, "data,", dense1.output.dtype, "output, second row", dense1.output[1])
print(f"   largest difference between the two outputs: {np.max(np.abs(plain.output - dense1.output)):.1e}")
print("   np.zeros:", np.zeros(2).dtype, "| np.random.randn:", np.random.randn(2).dtype,
      "| np.ones:", np.ones(2).dtype, "| np.array:", np.array([1.0, 2.0]).dtype)
a = np.array([[1.0, 2.0]])
b = np.array([[3.0], [4.0]])
print("   np.dot of two float64 arrays returns:", np.dot(a, b).dtype)
try:
    np.dot([1.0, 2.0], [3.0, 4.0])
except AttributeError as error:
    print("   np.dot on plain lists, AttributeError:", error)
