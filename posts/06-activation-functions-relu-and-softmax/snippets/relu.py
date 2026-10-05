"""Post 06, section 2: the ReLU activation, its class, and the slopes of ReLU, sigmoid and tanh.

Run from the series root:
    python posts/06-activation-functions-relu-and-softmax/snippets/relu.py

Needs NumPy only.
"""
import numpy as np


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


inputs = np.array([1, -2, 3, -0.5, 0])
print(np.maximum(0, inputs))   # [1. 0. 3. 0. 0.]

# np.max is a different function: it reduces the whole array to its largest entry.
print("np.max(inputs):", np.max(inputs))

# The class on a batch: 2 samples, 3 neurons. The shape does not change.
batch = np.array([[ 1.5, -0.3,  0.0],
                  [-2.0,  4.0, -0.1]])
activation = Activation_ReLU()
activation.forward(batch)
print("batch shape:", batch.shape, "-> output shape:", activation.output.shape)
print(activation.output)


# Slopes (derivatives) of the three hidden-layer activations at a few positive inputs.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_slope(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh_slope(z):
    return 1 - np.tanh(z) ** 2


print()
print("largest sigmoid slope, at z = 0:", sigmoid_slope(0.0))
print("largest tanh slope, at z = 0:   ", tanh_slope(0.0))
print("    z   ReLU slope   sigmoid slope   tanh slope")
for z in [0.5, 2.5, 5.0, 10.0]:
    relu_slope = 1.0 if z > 0 else 0.0
    print(f"{z:5.1f}   {relu_slope:10.0f}   {sigmoid_slope(z):13.2e}   {tanh_slope(z):10.2e}")

# A signal that passes back through ten sigmoids is multiplied by at most 0.25 each time.
print("0.25 ** 10 =", 0.25 ** 10)

# ReLU outputs are never negative, so their average is above zero.
np.random.seed(0)
sample = np.random.randn(100000)                 # 100,000 standard-normal inputs
print()
print(f"mean of the inputs: {sample.mean():.3f}")
print(f"mean after ReLU:    {np.maximum(0, sample).mean():.3f}   (1 / sqrt(2 pi) = {1 / np.sqrt(2 * np.pi):.3f})")
print(f"share of the inputs that ReLU sets to zero: {np.mean(sample <= 0):.3f}")
