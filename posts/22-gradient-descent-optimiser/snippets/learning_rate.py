"""Post 22, section 5: six learning rates on the spiral network, five seeds each.

Run from the series root:
    python posts/22-gradient-descent-optimiser/snippets/learning_rate.py          2,001 epochs
    python posts/22-gradient-descent-optimiser/snippets/learning_rate.py full     10,001 epochs

The setup is that of train_sgd.py except for the learning rate; the seeds are those of
seed_spread.py. For every rate the table gives the smallest and the largest value over seeds 0 to 4.
The second block follows the largest rate through its first epochs.

Needs NumPy and the nnfs helper package (pip install nnfs).
"""
import sys

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from optimizer_sgd import (Layer_Dense, Activation_ReLU,
                           Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD)
from seed_spread import train

RATES = (0.001, 0.01, 0.1, 1.0, 3.0, 10.0)
SEEDS = range(5)


def span(values, form):
    return f"{format(min(values), form)} to {format(max(values), form)}"


def first_epoch_all_dead(seed, learning_rate, epochs=200):
    """The first epoch whose forward pass leaves every hidden neuron at zero for all 300 samples."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)
    dense1, activation1 = Layer_Dense(2, 64), Activation_ReLU()
    dense2, loss_activation = Layer_Dense(64, 3), Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=learning_rate)
    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss_activation.forward(dense2.output, y)
        if np.all(activation1.output == 0):
            return epoch
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
    return None


if __name__ == "__main__":
    epochs = 10001 if len(sys.argv) > 1 and sys.argv[1] == "full" else 2001
    nnfs.init()

    print(f"{epochs:,} epochs, seeds 0 to 4: smallest and largest value over the five seeds")
    print("rate    final loss        final accuracy    loss rose on      largest rise    highest loss      dead of 64")
    for rate in RATES:
        finals, accuracies_, rises, jumps, peaks, deads = [], [], [], [], [], []
        for seed in SEEDS:
            losses, accuracies, dead = train(seed, learning_rate=rate, epochs=epochs)
            steps = np.diff(losses)
            finals.append(losses[-1])
            accuracies_.append(accuracies[-1])
            rises.append(int(np.sum(steps > 0)))
            jumps.append(steps.max())
            peaks.append(losses.max())
            deads.append(dead)
        largest = f"{max(jumps):.1e}" if max(jumps) > 0 else "none"
        print(f"{rate:<6}  {span(finals, '.4f'):<16}  {span(accuracies_, '.4f'):<16}  "
              f"{span(rises, ','):<16}  {largest:<14}  {span(peaks, '.4f'):<16}  {span(deads, 'd')}",
              flush=True)

    print("learning rate 10.0: the first epoch at which all 64 hidden neurons are dead")
    print("seed 0 to 4:", [first_epoch_all_dead(seed, 10.0) for seed in SEEDS])
