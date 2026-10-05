"""Post 28, sections 3 and 4: the documented run of post 27, then a forward-only pass on test data.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/test_pass.py

Trains the network of post 27 on seed 0, draws 300 new spiral points after training, and
measures loss and accuracy on both sets with the same weights. Then it checks what the test
pass changed, splits the test loss by sample, and repeats the test on twenty more draws.

Needs NumPy and the nnfs package. Takes about 10 seconds.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from network import SETUP, build, evaluate, train

nnfs.init()
print(SETUP)
print("seed: 0, the seed nnfs.init() sets")

X, y, dense1, activation1, dense2, loss_activation, optimizer = build(seed=0)
last_loss, last_accuracy = train(X, y, dense1, activation1, dense2, loss_activation, optimizer)
print(f"last epoch of the loop, before its update: loss {last_loss:.4f}  acc {last_accuracy:.4f}")

# What a test pass must leave alone: the parameters, the gradients and the optimiser.
parameters_before = [a.copy() for a in (dense1.weights, dense1.biases, dense2.weights, dense2.biases)]
gradients_before = [a.copy() for a in (dense1.dweights, dense1.dbiases, dense2.dweights, dense2.dbiases)]
iterations_before = optimizer.iterations

train_loss, train_accuracy = evaluate(X, y, dense1, activation1, dense2, loss_activation)

# Fresh test data: the same generator, new points. Drawn after training and without a new
# seed, so these are the next 300 normal draws of the stream that np.random.seed(0) started.
X_test, y_test = spiral_data(samples=100, classes=3)

# Forward pass only. No backward call, no optimiser call.
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == y_test)

print()
print("== Section 3: the same weights on two sets of 300 points")
print(f"training data  loss {train_loss:.4f}  acc {train_accuracy:.4f}  ({int(round(train_accuracy * 300))} of 300)")
print(f"test data      loss {loss:.4f}  acc {accuracy:.4f}  ({int(np.sum(predictions == y_test))} of 300)")
print(f"gap            loss {loss - train_loss:+.4f}  acc {train_accuracy - accuracy:.4f}  "
      f"({100 * (train_accuracy - accuracy):.2f} percentage points)")
print(f"ln 3 = {np.log(3):.4f}, the loss of a uniform guess over three classes")
print(f"distinct points the two sets share: {len(set(map(tuple, X)) & set(map(tuple, X_test)))} "
      "(the centre (0, 0), where every arm starts)")

print()
print("== Section 3.1: what the test pass changed")
parameters_after = (dense1.weights, dense1.biases, dense2.weights, dense2.biases)
gradients_after = (dense1.dweights, dense1.dbiases, dense2.dweights, dense2.dbiases)
changed = sum(int(np.sum(before != after)) for before, after in zip(parameters_before, parameters_after))
print(f"parameters changed: {changed} of {sum(a.size for a in parameters_after)}")
print(f"gradient arrays changed: {sum(not np.array_equal(b, a) for b, a in zip(gradients_before, gradients_after))} of 4")
print(f"optimizer.iterations: {iterations_before} before, {optimizer.iterations} after")
print(f"dense1.inputs is X_test: {dense1.inputs is X_test}")

print()
print("== Section 4: where the test loss comes from")
sample_losses = loss_activation.loss.forward(loss_activation.output, y_test)    # one loss per test point
wrong = predictions != y_test
print(f"misclassified test points: {int(wrong.sum())} of 300, mean loss {sample_losses[wrong].mean():.4f}, "
      f"share of the summed loss {sample_losses[wrong].sum() / sample_losses.sum():.4f}")
print(f"correct test points: {int((~wrong).sum())} of 300, mean loss {sample_losses[~wrong].mean():.4f}")
evaluate(X, y, dense1, activation1, dense2, loss_activation)
train_losses = loss_activation.loss.forward(loss_activation.output, y)
train_wrong = np.argmax(loss_activation.output, axis=1) != y
print(f"misclassified training points: {int(train_wrong.sum())} of 300, mean loss {train_losses[train_wrong].mean():.4f}")

print()
print("== Section 4: twenty more test sets of 300 points, drawn one after the other")
accuracies = []
for _ in range(20):
    X_more, y_more = spiral_data(samples=100, classes=3)
    accuracies.append(evaluate(X_more, y_more, dense1, activation1, dense2, loss_activation)[1])
print(f"test accuracy: lowest {min(accuracies):.4f}, highest {max(accuracies):.4f}, mean {np.mean(accuracies):.4f}")
print(f"sets that reach the training accuracy of {train_accuracy:.4f}: {sum(a >= train_accuracy for a in accuracies)} of 20")
p = float(np.mean(accuracies))
print(f"binomial standard error at p = {p:.3f}: {np.sqrt(p * (1 - p) / 300):.4f} for 300 points, "
      f"{np.sqrt(p * (1 - p) / 10000):.4f} for 10,000")
