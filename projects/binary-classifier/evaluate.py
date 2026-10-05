"""
Load a trained moons classifier and report behaviour:

  * Train and test accuracy + binary cross-entropy loss
  * 2x2 confusion matrix
  * Decision-boundary grid sampled at 200 x 200 points so a downstream
    plotter (matplotlib or otherwise) can render the boundary directly.

Usage:
    python evaluate.py                    # uses moons_weights.pkl
    python evaluate.py --checkpoint x.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from nn import (
    Activation_ReLU,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
    Layer_Dense,
)


def restore(checkpoint_path):
    payload = pickle.loads(Path(checkpoint_path).read_bytes())

    dense1 = Layer_Dense(2, 16)
    dense2 = Layer_Dense(16, 16)
    dense3 = Layer_Dense(16, 1)

    dense1.weights, dense1.biases = payload["dense1"]
    dense2.weights, dense2.biases = payload["dense2"]
    dense3.weights, dense3.biases = payload["dense3"]

    activation1 = Activation_ReLU()
    activation2 = Activation_ReLU()
    loss_act    = Activation_Sigmoid_Loss_BinaryCrossentropy()
    return dense1, activation1, dense2, activation2, dense3, loss_act, payload


def forward(X, layers):
    dense1, activation1, dense2, activation2, dense3, loss_act = layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    return dense3.output                          # raw logits


def metrics(name, X, y, layers):
    dense1, activation1, dense2, activation2, dense3, loss_act = layers
    logits = forward(X, layers)
    loss   = loss_act.forward(logits, y)
    preds  = (loss_act.output >= 0.5).astype(np.int64).ravel()
    acc    = float(np.mean(preds == y))

    confusion = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, preds):
        confusion[int(t)][int(p)] += 1

    print(f"\n[{name}] loss={loss:.4f}  accuracy={acc:.4f}  ({int(acc * len(y))}/{len(y)})")
    print(f"  confusion (rows=true, cols=pred):")
    print(f"               pred=0     pred=1")
    print(f"    true=0   {confusion[0][0]:>7d}    {confusion[0][1]:>7d}")
    print(f"    true=1   {confusion[1][0]:>7d}    {confusion[1][1]:>7d}")
    return preds, confusion


def decision_grid(layers, x_range=(-1.5, 2.5), y_range=(-1.0, 1.5), n=200):
    """Sample the model's predicted probability on an n x n grid.

    Returns:
        XX, YY  meshgrid of shape (n, n)
        ZZ      sigmoid probabilities of class 1, shape (n, n)
    """
    loss_act = layers[5]
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)

    logits = forward(grid, layers[:6])
    loss_act._sigmoid.forward(logits)
    probs = loss_act._sigmoid.output.reshape(n, n)
    return XX, YY, probs


def evaluate(checkpoint="moons_weights.pkl"):
    print(f"restoring weights from {checkpoint}")
    *layers, payload = restore(checkpoint)
    layers = tuple(layers)

    metrics("train", payload["X_train"], payload["y_train"], layers)
    metrics("test",  payload["X_test"],  payload["y_test"],  layers)

    # Sample the decision boundary for downstream plotting.
    XX, YY, probs = decision_grid(layers)
    print(f"\ndecision grid sampled at {probs.shape[0]} x {probs.shape[1]} points")
    print(f"  prob range:  [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  midline frac: {float(np.mean((probs > 0.45) & (probs < 0.55))):.3f}  (cells where 0.45 < p < 0.55)")

    # Optional: dump grid to .npz so a notebook can plot it without retraining.
    out_path = Path(checkpoint).with_name("decision_grid.npz")
    np.savez(out_path, XX=XX, YY=YY, probs=probs,
             X_train=payload["X_train"], y_train=payload["y_train"],
             X_test=payload["X_test"],   y_test=payload["y_test"])
    print(f"\nwrote decision-grid arrays to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="moons_weights.pkl")
    args = p.parse_args()
    evaluate(checkpoint=args.checkpoint)
