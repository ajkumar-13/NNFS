"""Post 28, section 8: three ways to get a wrong test figure, measured on the documented run.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/what_can_go_wrong.py

1. The seed set again before the test data is drawn. 2. The test data drawn before the layers
are created. 3. A backward pass and an optimiser step inside the test pass.

Needs NumPy and the nnfs package. Takes about 20 seconds (two training runs).
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from network import (SETUP, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy,
                     Layer_Dense, Optimizer_Adam, build, evaluate, train)

nnfs.init()
print(SETUP)
print("seed: 0")

X, y, dense1, activation1, dense2, loss_activation, optimizer = build(seed=0)
train(X, y, dense1, activation1, dense2, loss_activation, optimizer)
train_loss, train_accuracy = evaluate(X, y, dense1, activation1, dense2, loss_activation)
X_test, y_test = spiral_data(samples=100, classes=3)
test_loss, test_accuracy = evaluate(X_test, y_test, dense1, activation1, dense2, loss_activation)
print(f"correct: training acc {train_accuracy:.4f}, test acc {test_accuracy:.4f}, "
      f"gap {100 * (train_accuracy - test_accuracy):.2f} points")

print()
print("== 1. np.random.seed(0) again before the test data is drawn")
np.random.seed(0)
X_again, y_again = spiral_data(samples=100, classes=3)
again_loss, again_accuracy = evaluate(X_again, y_again, dense1, activation1, dense2, loss_activation)
print(f"identical to the training data: {np.array_equal(X_again, X) and np.array_equal(y_again, y)}")
print(f"'test' loss {again_loss:.4f}  acc {again_accuracy:.4f}  gap {100 * (train_accuracy - again_accuracy):.2f} points")

print()
print("== 2. the test data drawn before the layers are created")
np.random.seed(0)
X_b, y_b = spiral_data(samples=100, classes=3)
X_b_test, y_b_test = spiral_data(samples=100, classes=3)    # moved up: draws 301 to 600 of the stream
dense1_b = Layer_Dense(2, 64)                               # the weights now come from later draws
activation1_b = Activation_ReLU()
dense2_b = Layer_Dense(64, 3)
loss_activation_b = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer_b = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
train(X_b, y_b, dense1_b, activation1_b, dense2_b, loss_activation_b, optimizer_b)
train_b = evaluate(X_b, y_b, dense1_b, activation1_b, dense2_b, loss_activation_b)
test_b = evaluate(X_b_test, y_b_test, dense1_b, activation1_b, dense2_b, loss_activation_b)
print(f"same training data: {np.array_equal(X_b, X)}; same test data: {np.array_equal(X_b_test, X_test)}")
print(f"training loss {train_b[0]:.4f}  acc {train_b[1]:.4f};  test loss {test_b[0]:.4f}  acc {test_b[1]:.4f};  "
      f"gap {100 * (train_b[1] - test_b[1]):.2f} points")

print()
print("== 3. a backward pass and an optimiser step inside every test pass (first network)")
for passes in range(1, 101):
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test)
    accuracy = np.mean(np.argmax(loss_activation.output, axis=1) == y_test)

    loss_activation.backward(loss_activation.output, y_test)    # the mistake starts here
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    if passes in (1, 2, 10, 100):
        now_accuracy = evaluate(X, y, dense1, activation1, dense2, loss_activation)[1]
        print(f"test pass {passes:3d} reports loss {loss:.4f}  acc {accuracy:.4f}; "
              f"training acc after its update {now_accuracy:.4f}")
print(f"optimizer.iterations: {optimizer.iterations}")
