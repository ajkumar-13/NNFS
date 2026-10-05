"""
Load a trained Fashion-MNIST checkpoint and report test-set behaviour.

Outputs:
    * Test loss and accuracy
    * Per-class accuracy (named — T-shirt, Trouser, ..., Ankle boot)
    * Confusion matrix as a 10 x 10 grid with class names
    * Most-confused class pairs (sorted by off-diagonal cell count)

Usage:
    python evaluate.py
    python evaluate.py --checkpoint other.pkl
    python evaluate.py --backend manual
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import CLASS_NAMES, load_fashion_mnist
from nn import (
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Layer_Dropout,
)


def restore_model(checkpoint_path):
    weights = pickle.loads(Path(checkpoint_path).read_bytes())
    dense1 = Layer_Dense(784, 128)
    dense2 = Layer_Dense(128, 128)
    dense3 = Layer_Dense(128, 10)
    dense1.weights, dense1.biases = weights["dense1"]
    dense2.weights, dense2.biases = weights["dense2"]
    dense3.weights, dense3.biases = weights["dense3"]
    return (dense1, Activation_ReLU(), Layer_Dropout(0.1),
            dense2, Activation_ReLU(), Layer_Dropout(0.1),
            dense3, Activation_Softmax_Loss_CategoricalCrossentropy())


def forward_eval(X, layers):
    (dense1, activation1, dropout1,
     dense2, activation2, dropout2,
     dense3, _) = layers

    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output, training=False)

    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output, training=False)

    dense3.forward(dropout2.output)
    return dense3.output


def print_confusion_with_names(confusion):
    """Print a 10x10 confusion matrix with class names down the side."""
    # Header row.
    header = "                  " + "  ".join(f"{i:>4d}" for i in range(10))
    print(header)
    print("                  " + "  ".join("----" for _ in range(10)))
    for i in range(10):
        name = f"{CLASS_NAMES[i]:>14s} ({i})"
        row  = "  ".join(f"{confusion[i][j]:>4d}" for j in range(10))
        print(f"  {name} | {row}")


def most_confused_pairs(confusion, k=5):
    """Return the top-k (true, pred) pairs with the largest off-diagonal count."""
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:
                pairs.append((confusion[i][j], i, j))
    pairs.sort(reverse=True)
    return pairs[:k]


def evaluate(checkpoint="fashion_mnist_weights.pkl", backend="sklearn"):
    print(f"loading Fashion-MNIST via backend={backend!r}...")
    _, _, X_test, y_test = load_fashion_mnist(backend=backend)

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

    print("\nPer-class accuracy:")
    for c in range(10):
        mask = (y_test == c)
        if mask.sum() == 0:
            continue
        acc_c = float(np.mean(predictions[mask] == c))
        print(f"  {CLASS_NAMES[c]:>14s} ({c}): {acc_c:.4f}  ({int(mask.sum())} samples)")

    # Confusion matrix.
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, predictions):
        confusion[int(true)][int(pred)] += 1
    print("\nConfusion matrix (rows = true class, cols = predicted class):")
    print_confusion_with_names(confusion)

    # Most-confused class pairs.
    print("\nTop 5 most-confused class pairs (true → predicted):")
    for cnt, t, p in most_confused_pairs(confusion, k=5):
        print(f"  {CLASS_NAMES[t]:>14s} ({t}) → {CLASS_NAMES[p]:<14s} ({p}):  {cnt:>4d} times")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="fashion_mnist_weights.pkl")
    p.add_argument("--backend", choices=["sklearn", "manual"], default="sklearn")
    args = p.parse_args()
    evaluate(checkpoint=args.checkpoint, backend=args.backend)
