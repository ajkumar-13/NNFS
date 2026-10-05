"""Post 22, sections 4 and 6: the documented run repeated from other seeds, and run for longer.

Run from the series root:
    python posts/22-gradient-descent-optimiser/snippets/seed_spread.py          seeds 0 to 4
    python posts/22-gradient-descent-optimiser/snippets/seed_spread.py 10       seeds 0 to 9
    python posts/22-gradient-descent-optimiser/snippets/seed_spread.py long     50,001 epochs, seeds 0 to 4

Each run calls np.random.seed(seed) after nnfs.init() and then draws the data and the weights, so
seed 0 is the run of train_sgd.py. Everything else is that run: the 2 -> 64 -> 3 network, float32,
Optimizer_SGD(learning_rate=1.0), full-batch epochs.

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from optimizer_sgd import (Layer_Dense, Activation_ReLU,
                           Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD)


def update_both(optimizer, dense1, dense2):
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


def train(seed, learning_rate=1.0, epochs=10001, update=update_both):
    """The loop of train_sgd.py from one seed. Returns the loss and the accuracy of every epoch,
    and the number of hidden neurons that output zero for all 300 samples in the last forward pass."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=learning_rate)

    losses, accuracies = np.empty(epochs), np.empty(epochs)
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        losses[epoch] = loss_activation.forward(dense2.output, y)
        accuracies[epoch] = np.mean(np.argmax(loss_activation.output, axis=1) == y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        update(optimizer, dense1, dense2)

    dead = int(np.sum(np.all(activation1.output == 0, axis=0)))
    return losses, accuracies, dead


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "5"
    nnfs.init()

    if mode == "long":
        marks = (10000, 20000, 30000, 40000, 50000)
        print("learning rate 1.0, 50,001 epochs: loss and accuracy at five epochs")
        print("seed  " + "  ".join(f"{mark:>13,}" for mark in marks) + "   mean accuracy, last 1,000")
        finals, means = [], []
        for seed in range(5):
            losses, accuracies, dead = train(seed, epochs=50001)
            finals.append(accuracies[-1])
            means.append(accuracies[-1000:].mean())
            print(f"{seed:<4}  " + "  ".join(f"{losses[mark]:.4f} {accuracies[mark]:.4f}" for mark in marks)
                  + f"   {means[-1]:.4f}", flush=True)
        print(f"accuracy at epoch 50,000: {min(finals):.4f} to {max(finals):.4f};"
              f" mean of the last 1,000 epochs: {min(means):.4f} to {max(means):.4f}")
    else:
        seeds = range(int(mode))
        print("learning rate 1.0, 10,001 epochs; the last four columns are over epochs 9,001 to 10,000")
        print("seed  final loss  final acc  loss rose on  dead   lowest loss  highest loss  lowest acc  highest acc")
        rows = []
        for seed in seeds:
            losses, accuracies, dead = train(seed)
            rises = int(np.sum(np.diff(losses) > 0))
            rows.append((losses[-1], accuracies[-1], rises, dead, accuracies[-1000:].mean(),
                         losses[-1000:].mean()))
            print(f"{seed:<4}  {losses[-1]:<10.4f}  {accuracies[-1]:<9.4f}  {rises:<12,}  {dead:<5}  "
                  f"{losses[-1000:].min():<11.4f}  {losses[-1000:].max():<12.4f}  "
                  f"{accuracies[-1000:].min():<10.4f}  {accuracies[-1000:].max():.4f}", flush=True)
        final_losses, final_accuracies, rises, dead, means, mean_losses = zip(*rows)
        print(f"over the {len(rows)} seeds: final loss {min(final_losses):.4f} to {max(final_losses):.4f},"
              f" final accuracy {min(final_accuracies):.4f} to {max(final_accuracies):.4f}")
        print(f"loss rose on {min(rises):,} to {max(rises):,} of 10,000 updates;"
              f" dead hidden neurons {min(dead)} to {max(dead)} of 64")
        print(f"mean accuracy over epochs 9,001 to 10,000: {min(means):.4f} to {max(means):.4f}")
        print("mean over epochs 9,001 to 10,000, seed by seed")
        print("seed  mean loss  mean acc")
        for seed, mean_loss, mean in zip(seeds, mean_losses, means):
            print(f"{seed:<4}  {mean_loss:<9.4f}  {mean:.4f}")
        print(f"mean loss over epochs 9,001 to 10,000: {min(mean_losses):.4f} to {max(mean_losses):.4f}")
