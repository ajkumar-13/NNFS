"""Post 34, sections 2, 4 and 5: the stable sigmoid, the combined class, and one worked batch.

Run from the series root:
    python posts/34-sigmoid-and-binary-cross-entropy/snippets/binary_classes.py

Activation_Sigmoid is the class of post 17 with a new forward; its backward is unchanged.
Activation_Sigmoid_Loss_BinaryCrossentropy is new. The other scripts of this post import both.
Float64, no nnfs.init(). Needs only NumPy. Takes about a second.
"""
import numpy as np


class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        # Changed in post 34: two forms, chosen by sign, so that np.exp
        # never receives a positive argument.
        out = np.empty_like(inputs, dtype=np.float64)
        pos = inputs >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-inputs[pos]))     # exp of a value <= 0
        neg = ~pos
        ex = np.exp(inputs[neg])                           # exp of a value < 0
        out[neg] = ex / (1.0 + ex)
        self.output = out

    def backward(self, dvalues):
        # f'(z) = sigma(z) * (1 - sigma(z)), read from the cached output
        self.dinputs = dvalues * self.output * (1 - self.output)


class Activation_Sigmoid_Loss_BinaryCrossentropy:

    def __init__(self):
        self.activation = Activation_Sigmoid()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output               # (N, 1) probabilities

        # Labels of shape (N,) or (N, 1) become a float column.
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(np.mean(sample_losses))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)

        # Two steps: subtract the label, normalise.
        self.dinputs = (dvalues - y_true) / samples


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

    logits = np.array([[2.0], [-1.0], [0.5], [-3.0]])      # (4, 1), float64
    y = np.array([1, 0, 0, 1])                             # (4,)

    loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()
    loss = loss_activation.forward(logits, y)
    loss_activation.backward(loss_activation.output, y)

    y_hat = loss_activation.output[:, 0]
    sample_losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    print("== The worked batch of sections 3 to 5")
    print("z             ", logits[:, 0])
    print("y             ", y)
    print("y_hat         ", y_hat)
    print("sample losses ", sample_losses)
    print(f"mean loss      {loss:.4f}")
    print("y_hat - y     ", y_hat - y)
    print("dinputs       ", loss_activation.dinputs[:, 0], " shape", loss_activation.dinputs.shape)
    print("predictions   ", (loss_activation.output >= 0.5).astype(int)[:, 0],
          f" accuracy {np.mean((loss_activation.output >= 0.5).astype(int)[:, 0] == y):.2f}")

    print()
    print("== Section 2: three properties of the sigmoid")
    sigmoid = Activation_Sigmoid()
    z = np.array([[-5.0, -2.0, 0.0, 2.0, 5.0]])
    sigmoid.forward(z)
    sigmoid.backward(np.ones_like(z))
    print("z                   ", z[0])
    print("sigma(z)            ", sigmoid.output[0])
    print("sigma(z) + sigma(-z)", sigmoid.output[0] + sigmoid.output[0][::-1])
    print("slope               ", sigmoid.dinputs[0])

    print()
    print("== Section 5: label shapes and output dtype")
    for name, labels in (("(4,) int", y), ("(4, 1) int", y.reshape(-1, 1)), ("(4, 1) float", y.reshape(-1, 1) * 1.0)):
        value = loss_activation.forward(logits, labels)
        loss_activation.backward(loss_activation.output, labels)
        print(f"labels {name:13} loss {value:.4f}  dinputs shape {loss_activation.dinputs.shape}")
    out32 = Activation_Sigmoid()
    out32.forward(logits.astype(np.float32))
    print("float32 logits give an output of dtype", out32.output.dtype)
