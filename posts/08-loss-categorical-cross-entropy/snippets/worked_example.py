"""Post 08, sections 4 and 5: the worked batch, read with integer labels and with one-hot labels.

Run from the series root:
    python posts/08-loss-categorical-cross-entropy/snippets/worked_example.py

Needs only NumPy.
"""
import numpy as np

softmax_outputs = np.array([[0.7,  0.1, 0.2 ],
                            [0.1,  0.5, 0.4 ],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),
    class_targets
]
print(correct_confidences)

class_targets_onehot = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]])

correct_confidences = np.sum(
    class_targets_onehot * softmax_outputs,
    axis=1
)
print(correct_confidences)

sample_losses = -np.log(correct_confidences)
print(sample_losses)
print(np.mean(sample_losses))

# The table of section 4, to three decimals, and two facts section 5 and section 8.1 quote.
print()
for i, (target, p, loss) in enumerate(zip(class_targets, correct_confidences, sample_losses), start=1):
    print(f"sample {i}: true class {target}, probability on it {p:.2f}, loss {loss:.3f}")
print(f"batch loss, the mean of the three: {np.mean(sample_losses):.3f}")
print("np.eye(3)[class_targets] equals the one-hot rows:",
      np.array_equal(np.eye(3)[class_targets], class_targets_onehot))
print(f"exp(-batch loss) = {np.exp(-np.mean(sample_losses)):.3f},",
      f"the geometric mean of 0.7, 0.5 and 0.9 = {np.prod(correct_confidences) ** (1 / 3):.3f}")
