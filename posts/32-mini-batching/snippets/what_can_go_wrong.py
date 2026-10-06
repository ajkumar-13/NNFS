"""Post 32, section 10: two mistakes in the batch loop, each run.

Run from the series root:
    python posts/32-mini-batching/snippets/what_can_go_wrong.py

Contents: the optimiser created inside the batch loop, and the rows shuffled while the labels
are not. The loop below is train() of network.py written out again with one switch for each
mistake. The correct runs to compare the first mistake with are the table for batch size 32
and 1,000 epochs that head_to_head.py prints; the third mistake of the section, the last
batch's loss read as the epoch's, is printed by network.py.

Needs NumPy and the nnfs package. Takes about 15 seconds.
"""
import numpy as np
import nnfs

from network import SEEDS, SETUP, Optimizer_Adam, build, evaluate

BATCH_SIZE = 32


def loop(seed, epochs, optimizer_in_loop=False, shuffle_labels=True):
    """The loop of section 3 with two deliberate mistakes that can be switched on."""
    X, y, X_test, y_test, dense1, activation1, dense2, loss_activation, optimizer = build(seed)
    n_samples = len(X)
    epoch_losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        X_shuf = X[idx]
        y_shuf = y[idx] if shuffle_labels else y            # mistake 2: the labels keep their stored order

        batch_losses = []
        for start in range(0, n_samples, BATCH_SIZE):
            X_batch = X_shuf[start:start + BATCH_SIZE]
            y_batch = y_shuf[start:start + BATCH_SIZE]

            if optimizer_in_loop:                           # mistake 1: a new optimiser for every batch
                optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

            dense1.forward(X_batch)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, y_batch)

            loss_activation.backward(loss_activation.output, y_batch)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()

            batch_losses.append(float(loss))

        epoch_losses.append(float(np.mean(batch_losses)))

    network = (dense1, activation1, dense2, loss_activation)
    train_loss, train_accuracy = evaluate(X, y, *network)
    return dict(epoch_losses=epoch_losses, updates=optimizer.iterations, rate=optimizer.current_learning_rate,
                train_loss=train_loss, train_accuracy=train_accuracy)


nnfs.init()
print(SETUP)
print(f"batch size {BATCH_SIZE} throughout")

print()
print("== 1. Optimizer_Adam created inside the batch loop, 1,000 epochs")
print("seed  counter at the end  last rate  train loss  train acc")
for seed in SEEDS:
    wrong = loop(seed, 1000, optimizer_in_loop=True)
    print(f"{seed:4d}  {wrong['updates']:18d}  {wrong['rate']:9.6f}  {wrong['train_loss']:10.4f}  "
          f"{wrong['train_accuracy']:9.4f}", flush=True)
t = 1
print(f"the corrections with the counter stuck at 0: first moment / {1 - 0.9 ** t:.1f}, "
      f"second moment / {1 - 0.999 ** t:.3f}, step x {(1 / (1 - 0.9 ** t)) / np.sqrt(1 / (1 - 0.999 ** t)):.3f}")

print()
print("== 2. X shuffled, y left in its stored order, 200 epochs")
print(f"ln 3 = {np.log(3):.4f}")
print("seed  first epoch loss  last epoch loss  train loss  train acc")
for seed in SEEDS:
    r = loop(seed, 200, shuffle_labels=False)
    print(f"{seed:4d}  {r['epoch_losses'][0]:16.4f}  {r['epoch_losses'][-1]:15.4f}  "
          f"{r['train_loss']:10.4f}  {r['train_accuracy']:9.4f}", flush=True)
