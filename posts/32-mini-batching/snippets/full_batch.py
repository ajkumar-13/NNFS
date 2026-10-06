"""Post 32, section 4: with the batch size equal to N the two loops are the full-batch loop.

Run from the series root:
    python posts/32-mini-batching/snippets/full_batch.py

Seed 0 for 10,001 epochs, two networks side by side: one sees the rows in their stored order
in every epoch, which is the documented run of post 28, and the other sees them reshuffled
every epoch. Each has one batch per epoch, and the two batches hold the same 300 rows; only
the order of the rows inside the batch differs.

Needs NumPy and the nnfs package. Takes about 20 seconds.
"""
import numpy as np
import nnfs

from network import SETUP, build, evaluate, train

nnfs.init()
print(SETUP)
EPOCHS = 10001

stored = build(seed=0)
shuffled = build(seed=0)            # the same data and weights; its shuffles follow the test data in the stream
X, y, X_test, y_test = stored[:4]

first_difference = None
for epoch in range(EPOCHS):
    stored_loss, _ = train(X, y, *stored[4:], epochs=1, batch_size=len(X), order=np.arange)
    shuffled_loss, _ = train(X, y, *shuffled[4:], epochs=1, batch_size=len(X))
    if first_difference is None and not np.array_equal(stored[4].weights, shuffled[4].weights):
        first_difference = epoch + 1
        gap = np.abs(stored[4].weights - shuffled[4].weights).max()
        print(f"the weights of dense1 are equal for {epoch} updates; after update {first_difference} "
              f"the largest difference is {gap:.1e} ({stored[4].weights.dtype})")
        print(f"epoch loss of that epoch: stored order {stored_loss[0]:.7f}, shuffled {shuffled_loss[0]:.7f}")

print()
print(f"== Seed 0, batch size {len(X)}, {EPOCHS:,} epochs, forward-only after the last update")
print("rows in          updates  train loss  train acc  test loss  test acc")
for name, (_, _, _, _, dense1, activation1, dense2, loss_activation, optimizer) in (("stored order", stored),
                                                                                   ("shuffled order", shuffled)):
    network = (dense1, activation1, dense2, loss_activation)
    train_loss, train_accuracy = evaluate(X, y, *network)
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    print(f"{name:15s}  {optimizer.iterations:7d}  {train_loss:10.4f}  {train_accuracy:9.4f}  "
          f"{test_loss:9.4f}  {test_accuracy:8.4f}")
print(f"largest difference between the two sets of dense1 weights: {np.abs(stored[4].weights - shuffled[4].weights).max():.2f}")
