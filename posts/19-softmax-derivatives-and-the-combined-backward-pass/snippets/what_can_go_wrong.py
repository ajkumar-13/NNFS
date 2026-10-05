"""Post 19, section 8: seven ways to get the combined backward wrong, each measured.

Run from the series root:
    python posts/19-softmax-derivatives-and-the-combined-backward-pass/snippets/what_can_go_wrong.py

Needs only NumPy. The classes are imported from combined_class.py in the same directory.
"""
import warnings

import numpy as np

from combined_class import (Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossentropy,
                            Loss_CategoricalCrossentropy, central_difference)

np.set_printoptions(precision=4, suppress=True)


def combined(logits, y):
    """Forward and backward through the combined class; returns it."""
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.forward(logits, y)
    softmax_loss.backward(softmax_loss.output, y)
    return softmax_loss


def jacobian_route(logits, y):
    """The loss backward of post 18, then the softmax backward."""
    activation, loss_fn = Activation_Softmax(), Loss_CategoricalCrossentropy()
    activation.forward(logits)
    loss_fn.backward(activation.output, y)
    activation.backward(loss_fn.dinputs)
    return activation.dinputs


rng = np.random.default_rng(4)
logits = rng.normal(size=(5, 4))
y = rng.integers(0, 4, size=5)
reference = combined(logits, y)
measured = central_difference(lambda z: reference.forward(z, y), logits)

print("1. the division by N is left out, or done twice (N = 5)")
no_division = reference.dinputs * len(logits)
twice = reference.dinputs / len(logits)
print("   left out:   gradient / central difference =", np.unique((no_division / measured).round(4)))
print("   done twice: gradient / central difference =", np.unique((twice / measured).round(4)))

print("2. the standalone softmax backward is called after the combined one")
activation = Activation_Softmax()
activation.forward(logits)
activation.backward(reference.dinputs)
print("   row 0, combined:      ", reference.dinputs[0])
print("   row 0, after the call:", activation.dinputs[0])
print("   largest gap to the central difference:", f"{np.abs(activation.dinputs - measured).max():.3f}")

print("3. a soft label, [0.9, 0.05, 0.05], for the prediction [0.7, 0.1, 0.2]")
soft = np.array([[0.9, 0.05, 0.05]])
soft_logits = np.log(np.array([[0.7, 0.1, 0.2]]))


def cross_entropy(z):
    shifted = np.exp(z - z.max(axis=1, keepdims=True))
    return float(np.mean(-np.sum(soft * np.log(shifted / shifted.sum(axis=1, keepdims=True)), axis=1)))


print("   combined class:      ", combined(soft_logits, soft).dinputs)
print("   Jacobian route:      ", jacobian_route(soft_logits, soft))
print("   central difference:  ", central_difference(cross_entropy, soft_logits))
print("   (y_hat - y) / N:     ", (np.array([[0.7, 0.1, 0.2]]) - soft) / 1)

print("4. integer labels as a column of shape (N, 1)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")             # the forward pass takes log(0) here
    column = combined(logits, y.reshape(-1, 1))
print("   labels:", y, " the 1 is subtracted in columns:", np.argmin(column.dinputs, axis=1))
print("   largest gap to the central difference:", f"{np.abs(column.dinputs - measured).max():.3f}")

print("5. a true-class probability of exactly 0: logits [0, -800, 0], true class 1")
extreme = np.array([[0.0, -800.0, 0.0]])
label = np.array([1])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    separate = jacobian_route(extreme, label)
print("   softmax output:", combined(extreme, label).output)
print("   Jacobian route:", separate)
for warning in caught:
    print("   raised:        ", type(warning.message).__name__ + ":", warning.message)
print("   combined class:", combined(extreme, label).dinputs)

print("6. a true-class probability below the clip: logits [-20, 0, 0], true class 0")
low = np.array([[-20.0, 0.0, 0.0]])
label = np.array([0])
low_pass = combined(low, label)
print(f"   probability {low_pass.output[0, 0]:.3e}  reported loss {low_pass.forward(low, label):.4f}"
      f"  loss without the clip {-np.log(low_pass.output[0, 0]):.4f}")
print("   slopes of the reported loss:", central_difference(lambda z: low_pass.forward(z, label), low) + 0.0)
print("   combined class:             ", combined(low, label).dinputs)

print("7. y_hat - y used for a loss that is not cross-entropy: softmax, then mean squared error")
one_hot = np.eye(4)[y]


def mse(z):
    activation.forward(z)
    return float(np.mean(np.sum((activation.output - one_hot) ** 2, axis=1)))


mse_slopes = central_difference(mse, logits)
print("   row 0, slopes of the squared error:", mse_slopes[0])
print("   row 0, (y_hat - y) / N:            ", reference.dinputs[0])
