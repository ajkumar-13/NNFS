"""Post 34, section 7: a two-output softmax head and a one-output sigmoid head are the same model.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/softmax_equivalence.py

The softmax classes are those of post 19, imported from network.py. Float64, no nnfs.init().
Needs only NumPy. Takes about a second.
"""
import numpy as np

from binary_classes import Activation_Sigmoid_Loss_BinaryCrossentropy
from network import Activation_Softmax_Loss_CategoricalCrossentropy

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

rng = np.random.default_rng(34)
two_logits = rng.normal(scale=2.0, size=(6, 2))            # columns z_1, z_2: class 0, class 1
y = rng.integers(0, 2, size=6)
one_logit = two_logits[:, 1:2] - two_logits[:, 0:1]        # z = z_2 - z_1, shape (6, 1)

softmax_head = Activation_Softmax_Loss_CategoricalCrossentropy()
sigmoid_head = Activation_Sigmoid_Loss_BinaryCrossentropy()
softmax_loss = softmax_head.forward(two_logits, y)
sigmoid_loss = sigmoid_head.forward(one_logit, y)
softmax_head.backward(softmax_head.output, y)
sigmoid_head.backward(sigmoid_head.output, y)

print("== Six samples: softmax over (z_1, z_2) against sigmoid of z_2 - z_1")
print("y                      ", y)
print("z_1                    ", two_logits[:, 0])
print("z_2                    ", two_logits[:, 1])
print("softmax, column class 1", softmax_head.output[:, 1])
print("sigmoid(z_2 - z_1)     ", sigmoid_head.output[:, 0])
print(f"largest |softmax class 1 - sigmoid(z_2 - z_1)|: {np.abs(softmax_head.output[:, 1] - sigmoid_head.output[:, 0]).max():.1e}")
print(f"largest |softmax class 0 - sigmoid(z_1 - z_2)|: "
      f"{np.abs(softmax_head.output[:, 0] - (1 - sigmoid_head.output[:, 0])).max():.1e}")
print(f"categorical cross-entropy {softmax_loss:.6f}   binary cross-entropy {sigmoid_loss:.6f}   "
      f"gap {abs(softmax_loss - sigmoid_loss):.1e}")

print()
print("== The gradients")
print("softmax dinputs, column z_1", softmax_head.dinputs[:, 0])
print("softmax dinputs, column z_2", softmax_head.dinputs[:, 1])
print("sigmoid dinputs            ", sigmoid_head.dinputs[:, 0])
print(f"largest |column z_2 - sigmoid dinputs|: {np.abs(softmax_head.dinputs[:, 1] - sigmoid_head.dinputs[:, 0]).max():.1e}")
print(f"largest |column z_1 + column z_2|:      {np.abs(softmax_head.dinputs.sum(axis=1)).max():.1e}")

print()
print("== A shift of both logits changes nothing")
unshifted = softmax_head.output.copy()
softmax_head.forward(two_logits + 100.0, y)
print(f"largest change in the softmax output after adding 100 to both logits: "
      f"{np.abs(softmax_head.output - unshifted).max():.1e}")

print()
print("== Parameters of the last layer after 16 hidden neurons")
for outputs in (1, 2):
    print(f"Layer_Dense(16, {outputs}): {16 * outputs} weights + {outputs} biases = {16 * outputs + outputs}")
