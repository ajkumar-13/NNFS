"""Post 08, section 8.2: two batches with the same accuracy and different losses.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/loss_vs_accuracy.py

Needs only NumPy.
"""
import numpy as np

y_true = np.array([0, 1, 1])

barely_right = np.array([[0.7,  0.1,  0.2 ],      # the worked batch of section 4
                         [0.1,  0.5,  0.4 ],
                         [0.02, 0.9,  0.08]])
confidently_right = np.array([[0.95, 0.03, 0.02],
                              [0.02, 0.96, 0.02],
                              [0.01, 0.97, 0.02]])

results = {}
for name, probabilities in [("barely right", barely_right), ("confidently right", confidently_right)]:
    predictions   = np.argmax(probabilities, axis=1)
    accuracy      = np.mean(predictions == y_true)
    sample_losses = -np.log(probabilities[range(len(probabilities)), y_true])
    results[name] = np.mean(sample_losses)
    print(name)
    print("   predictions:", predictions, " true classes:", y_true)
    print(f"   accuracy: {accuracy:.3f} ({np.sum(predictions == y_true)} of {len(y_true)})")
    print("   per-sample losses:", np.round(sample_losses, 3))
    print(f"   mean loss: {results[name]:.3f}")
print(f"ratio of the two mean losses: {results['barely right'] / results['confidently right']:.1f}")

print("one two-class sample, true class 0, predicted two ways")
for row in ([0.51, 0.49], [0.99, 0.01]):
    print(f"   {row}: predicted class {np.argmax(row)}, loss {-np.log(row[0]):.3f}")
