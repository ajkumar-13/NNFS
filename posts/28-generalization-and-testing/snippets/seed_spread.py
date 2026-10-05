"""Post 28, sections 3 to 5: the gap over five seeds, and the test loss along the run.

Run from the series root:
    python posts/28-generalization-and-testing/snippets/seed_spread.py

For each seed the data and the weights are drawn again, the network of post 27 is trained for
10,001 epochs, and loss and accuracy are measured forward-only on the training data and on 300
test points. The test points are drawn before the loop here, so that they can be read during
training; the loop draws no random numbers, so they are the points test_pass.py draws after it.

Needs NumPy and the nnfs package. Takes about 45 seconds.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from network import SETUP, boundary_length, build, evaluate, train, weight_norm

SEEDS = (0, 1, 2, 3, 4)


def run(seed, n_neurons=64, follow=True):
    """One trained network: its final figures and, if follow is set, its check of lowest test loss."""
    X, y, dense1, activation1, dense2, loss_activation, optimizer = build(seed, n_neurons)
    X_test, y_test = spiral_data(samples=100, classes=3)
    network = (dense1, activation1, dense2, loss_activation)
    lowest = {"test_loss": np.inf, "curve": []}

    def on_check(epoch):
        test_loss, test_accuracy = evaluate(X_test, y_test, *network)
        train_loss, train_accuracy = evaluate(X, y, *network)
        lowest["curve"].append((epoch, train_loss, train_accuracy, test_loss, test_accuracy))
        if test_loss < lowest["test_loss"]:
            lowest.update(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                          test_loss=test_loss, test_accuracy=test_accuracy,
                          boundary=boundary_length(dense1, activation1, dense2),
                          weight_norm=weight_norm(dense1, dense2))

    last_loss, last_accuracy = train(X, y, *network, optimizer, on_check=on_check if follow else None)
    train_loss, train_accuracy = evaluate(X, y, *network)
    dead = int(np.sum(np.all(activation1.output == 0, axis=0)))    # hidden neurons at zero on all 300 training points
    test_loss, test_accuracy = evaluate(X_test, y_test, *network)
    final = dict(last_accuracy=last_accuracy, dead=dead, train_loss=train_loss, train_accuracy=train_accuracy,
                 test_loss=test_loss, test_accuracy=test_accuracy,
                 boundary=boundary_length(dense1, activation1, dense2),
                 weight_norm=weight_norm(dense1, dense2))
    return final, lowest


def final_table(n_neurons, follow=True):
    """Print one row per seed and the ranges; return the list of (final, lowest) pairs."""
    print("seed  train loss  train acc  test loss  test acc  gap in points")
    results = []
    for seed in SEEDS:
        final, lowest = run(seed, n_neurons, follow)
        results.append((final, lowest))
        print(f"{seed:4d}  {final['train_loss']:10.4f}  {final['train_accuracy']:9.4f}  {final['test_loss']:9.4f}  "
              f"{final['test_accuracy']:8.4f}  {100 * (final['train_accuracy'] - final['test_accuracy']):13.2f}", flush=True)
    train = [100 * f["train_accuracy"] for f, _ in results]
    test = [100 * f["test_accuracy"] for f, _ in results]
    gaps = [a - b for a, b in zip(train, test)]
    print(f"train accuracy {min(train):.2f} to {max(train):.2f} percent, test accuracy {min(test):.2f} to {max(test):.2f} percent, "
          f"gap {min(gaps):.2f} to {max(gaps):.2f} points")
    print(f"runs with a higher test loss than training loss: {sum(f['test_loss'] > f['train_loss'] for f, _ in results)} of {len(SEEDS)}")
    return results


if __name__ == "__main__":
    nnfs.init()
    print(SETUP)
    print(f"seeds: {SEEDS}, np.random.seed(s) before the training data, the weights and then the test data are drawn")

    print()
    print("== Section 3: 64 neurons, after the last update")
    results = final_table(64)
    print("loop's last accuracy, before its update: "
          + "  ".join(f"{f['last_accuracy']:.4f}" for f, _ in results))

    print()
    print("== Section 5: seed 0 along the run, read forward-only at the start of the epoch")
    print("epoch  train loss  train acc  test loss  test acc")
    for epoch, train_loss, train_accuracy, test_loss, test_accuracy in results[0][1]["curve"]:
        if epoch in (0, 100, 300, 700, 1000, 2000, 5000, 10000):
            print(f"{epoch:5d}  {train_loss:10.4f}  {train_accuracy:9.4f}  {test_loss:9.4f}  {test_accuracy:8.4f}")

    print()
    print("== Section 5: the check with the lowest test loss (checks every 100 epochs) against the end")
    print("seed  epoch  test loss -> end     test acc -> end     gap in points -> end  "
          "boundary -> end  weight norm -> end")
    lower = 0
    for seed, (final, lowest) in zip(SEEDS, results):
        gap_lowest = 100 * (lowest["train_accuracy"] - lowest["test_accuracy"])
        gap_final = 100 * (final["train_accuracy"] - final["test_accuracy"])
        lower += lowest["test_accuracy"] < final["test_accuracy"]
        print(f"{seed:4d}  {lowest['epoch']:5d}  {lowest['test_loss']:.4f} -> {final['test_loss']:.4f}  "
              f"   {lowest['test_accuracy']:.4f} -> {final['test_accuracy']:.4f}  "
              f"   {gap_lowest:5.2f} -> {gap_final:5.2f}        "
              f"   {lowest['boundary']:5d} -> {final['boundary']:5d}  "
              f"   {lowest['weight_norm']:6.2f} -> {final['weight_norm']:6.2f}")
    print(f"runs whose test accuracy is lower at the check of lowest test loss than at the end: {lower} of {len(SEEDS)}")
    ratios = [final["boundary"] / lowest["boundary"] for final, lowest in results]
    norms = [final["weight_norm"] / lowest["weight_norm"] for final, lowest in results]
    print(f"boundary length, end over check: {min(ratios):.2f} to {max(ratios):.2f}; "
          f"weight norm, end over check: {min(norms):.2f} to {max(norms):.2f}")
