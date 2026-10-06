"""Post 34, section 9: mistakes around the sigmoid head, each run and measured.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/what_can_go_wrong.py

Float64, no nnfs.init(). Needs only NumPy. Takes about a second.
"""
import numpy as np

from binary_classes import Activation_Sigmoid_Loss_BinaryCrossentropy
from gradient_check import numerical_gradient, relative_error
from network import Loss_CategoricalCrossentropy

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()

print("== 1. Accuracy from a column of predictions and a flat array of labels")
y = np.array([1, 0, 0, 1, 0, 0, 1, 0])
logits = 4.0 * (2 * y - 1).reshape(-1, 1)                  # every sample on the right side
loss_activation.forward(logits, y)
predictions = (loss_activation.output >= 0.5).astype(int)  # shape (8, 1)
wrong = predictions == y
right = predictions.ravel() == y
print(f"predictions {predictions.shape} == y {y.shape}: shape {wrong.shape}, mean {np.mean(wrong):.4f}")
print(f"predictions.ravel() == y: shape {right.shape}, mean {np.mean(right):.4f}")

print()
print("== 2. backward on the worked batch, with one thing wrong")
logits = np.array([[2.0], [-1.0], [0.5], [-3.0]])
y = np.array([1, 0, 0, 1])
measured = numerical_gradient(lambda: loss_activation.forward(logits, y), logits)
loss_activation.forward(logits, y)
loss_activation.backward(loss_activation.output, y)
correct = loss_activation.dinputs.copy()
loss_activation.backward(logits, y)                        # the logits, not the output
from_logits = loss_activation.dinputs.copy()
no_division = correct * len(logits)                        # the line without / samples
print(f"{'central difference':26}", measured[:, 0])
for name, values in (("correct", correct), ("logits handed to backward", from_logits), ("no division by samples", no_division)):
    print(f"{name:26}", values[:, 0], f" relative error {relative_error(values, measured):.1e}")

print()
print("== 3. Labels -1 and +1 in place of 0 and 1")
y_signed = np.array([1, -1, -1, 1])
print(f"worked batch: loss {loss_activation.forward(logits, y_signed):.4f} (with labels 0 and 1: {loss_activation.forward(logits, y):.4f})")
for z in (-20.0,):
    one = np.array([[z]])
    value = loss_activation.forward(one, np.array([-1]))
    loss_activation.backward(loss_activation.output, np.array([-1]))
    print(f"one sample, z = {z:g}, y = -1, a correct prediction: loss {value:.4f}, dinputs {loss_activation.dinputs[0, 0]:+.4f}")

print()
print("== 4. Two output neurons handed to the sigmoid head")
two = np.array([[2.0, -2.0], [-1.0, 1.0], [0.5, -0.5], [-3.0, 3.0]])
value = loss_activation.forward(two, y)
loss_activation.backward(loss_activation.output, y)
print(f"logits (4, 2), labels (4,): loss {value:.4f}, dinputs shape {loss_activation.dinputs.shape}, no error")
try:
    loss_activation.forward(two, np.eye(2)[y])
except ValueError as error:
    print("logits (4, 2), one-hot labels (4, 2): ValueError:", error)

print()
print("== 5. One sigmoid output handed to the categorical cross-entropy of post 19")
loss_activation.forward(logits, y)
categorical = Loss_CategoricalCrossentropy()
try:
    categorical.calculate(loss_activation.output, y)
except IndexError as error:
    print("integer labels: IndexError:", error)
value = categorical.calculate(loss_activation.output, np.eye(2)[y])
print(f"one-hot labels: loss {value:.4f}, no error; mean of -log(y_hat) over all four samples: "
      f"{np.mean(-np.log(loss_activation.output)):.4f}")
