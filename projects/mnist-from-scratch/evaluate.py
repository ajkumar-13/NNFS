"""
Load a trained MNIST checkpoint and report test-set behaviour.

Outputs:
    * Test loss and accuracy
    * Confusion matrix (10 x 10) printed as a table
    * Per-class accuracy
    * Indices of misclassified test samples (so you can inspect them)

Usage:
    python evaluate.py                              # uses mnist_weights.pkl
    python evaluate.py --checkpoint other.pkl
    python evaluate.py --backend manual
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import load_mnist
from nn import (
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Layer_Dropout,
)


def restore_model(checkpoint_path):
    """Re-create the network and load weights from a pickle file."""
    weights = pickle.loads(Path(checkpoint_path).read_bytes())

    dense1 = Layer_Dense(784, 128)
    dense2 = Layer_Dense(128, 128)
    dense3 = Layer_Dense(128, 10)

    dense1.weights, dense1.biases = weights["dense1"]
    dense2.weights, dense2.biases = weights["dense2"]
    dense3.weights, dense3.biases = weights["dense3"]

    activation1 = Activation_ReLU()
    activation2 = Activation_ReLU()
    dropout1    = Layer_Dropout(0.1)
    dropout2    = Layer_Dropout(0.1)
    loss_act    = Activation_Softmax_Loss_CategoricalCrossentropy()
    return dense1, activation1, dropout1, dense2, activation2, dropout2, dense3, loss_act


def forward_eval(X, layers):
    """Forward pass with dropout disabled. Returns the softmax + loss object's output."""
    (dense1, activation1, dropout1,
     dense2, activation2, dropout2,
     dense3, loss_act) = layers

    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=False)

    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output, training=False)

    dense3.forward(dropout2.output)
    return dense3.output


def print_confusion_matrix(confusion):
    print("\nConfusion matrix (rows = true class, cols = predicted class):")
    header = "         " + "  ".join(f"{j:>5d}" for j in range(10))
    print(header)
    for i in range(10):
        row = "  ".join(f"{confusion[i][j]:>5d}" for j in range(10))
        print(f"  true {i}: {row}")


def evaluate(checkpoint="mnist_weights.pkl", backend="sklearn"):
    print(f"loading MNIST via backend={backend!r}...")
    _, _, X_test, y_test = load_mnist(backend=backend)

    print(f"restoring weights from {checkpoint}")
    layers = restore_model(checkpoint)
    loss_act = layers[-1]

    logits      = forward_eval(X_test, layers)
    test_loss   = loss_act.forward(logits, y_test)
    predictions = np.argmax(loss_act.output, axis=1)
    test_acc    = float(np.mean(predictions == y_test))

    n_wrong = int(np.sum(predictions != y_test))
    print(f"\n  loss     {test_loss:.4f}")
    print(f"  accuracy {test_acc:.4f}  ({len(y_test) - n_wrong}/{len(y_test)})")

    # Per-class accuracy.
    print("\nPer-class accuracy:")
    for c in range(10):
        mask = (y_test == c)
        if mask.sum() == 0:
            continue
        acc_c = float(np.mean(predictions[mask] == c))
        print(f"  digit {c}: {acc_c:.4f}  ({int(mask.sum())} samples)")

    # Confusion matrix.
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, predictions):
        confusion[int(true)][int(pred)] += 1
    print_confusion_matrix(confusion)

    # Misclassified indices (first 20).
    wrong_idx = np.where(predictions != y_test)[0]
    print(f"\n{n_wrong} misclassified samples. First 20 indices: "
          f"{wrong_idx[:20].tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="mnist_weights.pkl")
    p.add_argument("--backend", choices=["sklearn", "manual"], default="sklearn")
    args = p.parse_args()

    evaluate(checkpoint=args.checkpoint, backend=args.backend)
