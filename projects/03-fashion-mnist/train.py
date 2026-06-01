"""
Train the same 2-hidden-layer architecture from project 01 on Fashion-MNIST.

Architecture: 784 → 128 → ReLU → Dropout(0.1) → 128 → ReLU → Dropout(0.1) → 10
Optimiser:    Adam, lr=0.001, decay=1e-4
Regularisation: L2 weight decay (lambda=5e-4) on every Dense layer
Schedule:     20 epochs, mini-batch size 128

Test accuracy with these defaults sits around 88-89% — about 8-9 points
lower than the same architecture hits on MNIST, which is the standard
Fashion-MNIST-vs-MNIST gap and the reason Fashion-MNIST is recommended
as a "drop-in but harder" benchmark.

Usage:
    python train.py                       # uses sklearn backend
    python train.py --backend manual      # uses Zalando S3 mirror
    python train.py --epochs 5            # short smoke run
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import load_fashion_mnist
from nn import (
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Layer_Dropout,
    Optimizer_Adam,
    regularization_loss,
)


def build_model():
    dense1      = Layer_Dense(784, 128, weight_regularizer_l2=5e-4)
    activation1 = Activation_ReLU()
    dropout1    = Layer_Dropout(0.1)

    dense2      = Layer_Dense(128, 128, weight_regularizer_l2=5e-4)
    activation2 = Activation_ReLU()
    dropout2    = Layer_Dropout(0.1)

    dense3      = Layer_Dense(128, 10)
    loss_act    = Activation_Softmax_Loss_CategoricalCrossentropy()

    return dense1, activation1, dropout1, dense2, activation2, dropout2, dense3, loss_act


def train(epochs=20, batch_size=128, backend="sklearn",
          checkpoint="fashion_mnist_weights.pkl", seed=0):
    np.random.seed(seed)

    print(f"loading Fashion-MNIST via backend={backend!r}...")
    X_train, y_train, X_test, y_test = load_fashion_mnist(backend=backend)
    print(f"  train: {X_train.shape}    test: {X_test.shape}")

    (dense1, activation1, dropout1,
     dense2, activation2, dropout2,
     dense3, loss_act) = build_model()

    optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-4)

    n_samples = len(X_train)
    n_steps   = (n_samples + batch_size - 1) // batch_size
    print(f"training for {epochs} epochs, batch_size={batch_size} ({n_steps} steps/epoch)\n")

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n_samples)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # Forward.
            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dropout1.forward(activation1.output, training=True)

            dense2.forward(dropout1.output)
            activation2.forward(dense2.output)
            dropout2.forward(activation2.output, training=True)

            dense3.forward(dropout2.output)
            data_loss = loss_act.forward(dense3.output, y_batch)

            reg_loss = (regularization_loss(dense1) +
                        regularization_loss(dense2) +
                        regularization_loss(dense3))
            loss = data_loss + reg_loss

            predictions = np.argmax(loss_act.output, axis=1)
            epoch_correct += int(np.sum(predictions == y_batch))

            # Backward.
            loss_act.backward(loss_act.output, y_batch)
            dense3.backward(loss_act.dinputs)
            dropout2.backward(dense3.dinputs)
            activation2.backward(dropout2.dinputs)
            dense2.backward(activation2.dinputs)
            dropout1.backward(dense2.dinputs)
            activation1.backward(dropout1.dinputs)
            dense1.backward(activation1.dinputs)

            # Update.
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
            optimizer.post_update_params()

            epoch_loss += float(loss)
            n_batches += 1

        avg_loss  = epoch_loss / n_batches
        train_acc = epoch_correct / n_samples
        print(f"epoch {epoch:3d} | loss {avg_loss:.4f} | "
              f"train_acc {train_acc:.4f} | "
              f"lr {optimizer.current_learning_rate:.6f}")

    weights = {
        "dense1": (dense1.weights, dense1.biases),
        "dense2": (dense2.weights, dense2.biases),
        "dense3": (dense3.weights, dense3.biases),
    }
    Path(checkpoint).write_bytes(pickle.dumps(weights))
    print(f"\nwrote weights to {checkpoint}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--backend", choices=["sklearn", "manual"], default="sklearn")
    p.add_argument("--checkpoint", default="fashion_mnist_weights.pkl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train(epochs=args.epochs,
          batch_size=args.batch_size,
          backend=args.backend,
          checkpoint=args.checkpoint,
          seed=args.seed)
