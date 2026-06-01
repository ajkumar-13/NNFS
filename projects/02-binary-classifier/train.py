"""
Train a 2-hidden-layer binary classifier on the two-moons dataset.

Architecture: 2 -> 16 -> ReLU -> 16 -> ReLU -> 1 -> Sigmoid + BCE
Optimiser:    Adam, lr=0.01, no decay
Regularisation: L2 on Dense weights (lambda = 1e-4)
Schedule:     2000 epochs, full-batch (n=800)

Usage:
    python train.py                 # default
    python train.py --epochs 5000   # longer run
    python train.py --noise 0.30    # harder dataset
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import make_moons, train_test_split
from nn import (
    Activation_ReLU,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
    Layer_Dense,
    Optimizer_Adam,
    regularization_loss,
)


def build_model():
    dense1      = Layer_Dense(2, 16, weight_regularizer_l2=1e-4)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(16, 16, weight_regularizer_l2=1e-4)
    activation2 = Activation_ReLU()
    dense3      = Layer_Dense(16, 1)
    loss_act    = Activation_Sigmoid_Loss_BinaryCrossentropy()
    return dense1, activation1, dense2, activation2, dense3, loss_act


def train(epochs=2000, noise=0.20, n_samples=1000,
          checkpoint="moons_weights.pkl", seed=0, log_every=200):
    np.random.seed(seed)

    X, y = make_moons(n_samples=n_samples, noise=noise, seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, seed=seed)
    print(f"train: {X_train.shape}    test: {X_test.shape}")

    (dense1, activation1, dense2, activation2, dense3, loss_act) = build_model()
    optimizer = Optimizer_Adam(learning_rate=0.01)

    for epoch in range(1, epochs + 1):
        # Forward.
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        data_loss = loss_act.forward(dense3.output, y_train)

        reg_loss = (regularization_loss(dense1) +
                    regularization_loss(dense2) +
                    regularization_loss(dense3))
        loss = data_loss + reg_loss

        predictions = (loss_act.output >= 0.5).astype(np.int64).ravel()
        train_acc   = float(np.mean(predictions == y_train))

        # Backward.
        loss_act.backward(loss_act.output, y_train)
        dense3.backward(loss_act.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update.
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:5d} | loss {loss:.4f} | train_acc {train_acc:.4f} | lr {optimizer.current_learning_rate:.4f}")

    # Save weights and the train/test split for evaluate.py.
    payload = {
        "dense1": (dense1.weights, dense1.biases),
        "dense2": (dense2.weights, dense2.biases),
        "dense3": (dense3.weights, dense3.biases),
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "config":  {"noise": noise, "n_samples": n_samples, "seed": seed},
    }
    Path(checkpoint).write_bytes(pickle.dumps(payload))
    print(f"\nwrote checkpoint to {checkpoint}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.20)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--checkpoint", default="moons_weights.pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    train(epochs=args.epochs, noise=args.noise, n_samples=args.n_samples,
          checkpoint=args.checkpoint, seed=args.seed, log_every=args.log_every)
