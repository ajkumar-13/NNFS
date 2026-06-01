"""
Train a 2-hidden-layer regression network on California housing prices.

Architecture: 8 -> 64 -> ReLU -> 64 -> ReLU -> 1 (linear output)
Loss:         mean squared error (MSE), in standardised target units
Optimiser:    Adam, lr=0.01, decay=1e-4
Regularisation: L2 weight decay (lambda=1e-4) on every Dense layer
Schedule:     200 epochs, mini-batch size 256

The output is a single linear neuron (no softmax, no sigmoid). The
target is standardised to zero mean and unit variance before training
so the loss is on the same scale as the lecture examples; predictions
are de-standardised at eval time so the RMSE is reported in dollars.

Usage:
    python train.py
    python train.py --epochs 500
    python train.py --backend manual
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import load_california_housing
from nn import (
    Activation_ReLU,
    Layer_Dense,
    Loss_MSE,
    Optimizer_Adam,
    regularization_loss,
)


def build_model():
    dense1      = Layer_Dense(8, 64, weight_regularizer_l2=1e-4)
    activation1 = Activation_ReLU()
    dense2      = Layer_Dense(64, 64, weight_regularizer_l2=1e-4)
    activation2 = Activation_ReLU()
    dense3      = Layer_Dense(64, 1)
    loss_fn     = Loss_MSE()
    return dense1, activation1, dense2, activation2, dense3, loss_fn


def train(epochs=200, batch_size=256, backend="sklearn",
          checkpoint="cal_housing_weights.pkl", seed=0, log_every=20):
    np.random.seed(seed)

    print(f"loading California housing via backend={backend!r}...")
    X_train, y_train, X_test, y_test, stats = load_california_housing(backend=backend, seed=seed)
    print(f"  train: {X_train.shape}    test: {X_test.shape}")
    print(f"  y_mean (raw): ${stats['y_mean'] * 100_000:>10,.0f}")
    print(f"  y_std  (raw): ${stats['y_std']  * 100_000:>10,.0f}\n")

    (dense1, activation1, dense2, activation2, dense3, loss_fn) = build_model()
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)

    n_samples = len(X_train)
    n_steps   = (n_samples + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_samples)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_samples, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # Forward.
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)
            data_loss = loss_fn.forward(dense3.output, y_batch)

            reg_loss = (regularization_loss(dense1) +
                        regularization_loss(dense2) +
                        regularization_loss(dense3))
            loss = data_loss + reg_loss

            # Backward.
            loss_fn.backward(dense3.output)
            dense3.backward(loss_fn.dinputs)
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

            epoch_loss += float(loss)
            n_batches += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            train_rmse = (epoch_loss / n_batches) ** 0.5 * stats['y_std'] * 100_000
            print(f"epoch {epoch:4d} | "
                  f"train_mse {epoch_loss / n_batches:.4f} (std units) | "
                  f"train_rmse ${train_rmse:>10,.0f} | "
                  f"lr {optimizer.current_learning_rate:.5f}")

    payload = {
        "dense1": (dense1.weights, dense1.biases),
        "dense2": (dense2.weights, dense2.biases),
        "dense3": (dense3.weights, dense3.biases),
        "stats":  stats,
        "X_test": X_test, "y_test": y_test,
        "X_train": X_train, "y_train": y_train,
    }
    Path(checkpoint).write_bytes(pickle.dumps(payload))
    print(f"\nwrote checkpoint to {checkpoint}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--backend", choices=["sklearn", "manual"], default="sklearn")
    p.add_argument("--checkpoint", default="cal_housing_weights.pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, backend=args.backend,
          checkpoint=args.checkpoint, seed=args.seed, log_every=args.log_every)
