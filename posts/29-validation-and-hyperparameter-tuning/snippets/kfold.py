"""Post 29, sections 4 and 10: k_fold_split, the training and scoring helpers, and checks of the split.

Run from the series root:
    python posts/29-validation-and-hyperparameter-tuning/snippets/kfold.py

Contents: the network classes of posts 16 and 19 and Optimizer_Adam of post 27, unchanged;
k_fold_indices and k_fold_split; train(), accuracy(), k_fold_accuracies() and draw_data(), which the
other snippets of this post import; checks that the folds partition the data; and two faulty splits, measured.

Needs NumPy and the nnfs package (spiral_data and nnfs.init). Takes about 10 seconds.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)          # shape of weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   # shape of biases
        self.dinputs = np.dot(dvalues, self.weights.T)          # shape of inputs


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()                           # the caller's array stays intact
        self.dinputs[self.inputs <= 0] = 0                      # closed gates pass nothing back


class Activation_Softmax:
    def forward(self, inputs):
        shifted       = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values    = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output   = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # One Jacobian and one product for every sample of the batch.
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            jacobian = np.diagflat(single_output) - np.outer(single_output, single_output)
            self.dinputs[index] = single_dvalues @ jacobian


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples        = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Integer labels:
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # One-hot labels:
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels  = len(dvalues[0])

        # Integer labels become one-hot rows.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # The gradient of each sample's loss, then the 1/N of the batch mean.
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot, convert to indices.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Three lines: copy, subtract 1 at the true class, normalise.
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0.0,
                 epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.epsilon               = epsilon
        self.beta_1                = beta_1
        self.beta_2                = beta_2
        self.iterations            = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update_params(self, layer):
        # Lazy buffer creation: one momentum and one cache per parameter array.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        # 1) First moment: moving average of the gradient.
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums   = self.beta_1 * layer.bias_momentums + \
                                 (1 - self.beta_1) * layer.dbiases

        # 2) Bias correction of the first moment. The step counter is
        #    incremented in post_update_params, after this call, hence the + 1.
        t = self.iterations + 1
        weight_m_hat = layer.weight_momentums / (1 - self.beta_1 ** t)
        bias_m_hat   = layer.bias_momentums   / (1 - self.beta_1 ** t)

        # 3) Second moment: moving average of the squared gradient.
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache   = self.beta_2 * layer.bias_cache + \
                             (1 - self.beta_2) * layer.dbiases ** 2

        # 4) Bias correction of the second moment.
        weight_v_hat = layer.weight_cache / (1 - self.beta_2 ** t)
        bias_v_hat   = layer.bias_cache   / (1 - self.beta_2 ** t)

        # 5) Parameter update.
        layer.weights -= self.current_learning_rate * weight_m_hat / \
                         (np.sqrt(weight_v_hat) + self.epsilon)
        layer.biases  -= self.current_learning_rate * bias_m_hat / \
                         (np.sqrt(bias_v_hat)   + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


def k_fold_indices(n, k=5, shuffle=True, seed=0):
    """Yield (train_idx, val_idx) for each of the k folds of n samples."""
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)    # a generator of its own, shuffled once

    folds = np.array_split(idx, k)                  # sizes differ by at most one
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def k_fold_split(X, y, k=5, shuffle=True, seed=0):
    """Yield (train_X, train_y, val_X, val_y) for each fold."""
    for train_idx, val_idx in k_fold_indices(len(X), k, shuffle, seed):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train(X, y, learning_rate, n_neurons=64, epochs=1000, seed=0):
    """A fresh 2 -> n_neurons -> 3 network, trained full-batch with Adam. Returns the four objects."""
    np.random.seed(seed)                          # the initial weights depend on seed alone
    dense1 = Layer_Dense(2, n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss_activation.forward(dense2.output, y)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return dense1, activation1, dense2, loss_activation


def accuracy(model, X, y):
    """Forward only (post 28): the share of samples whose largest logit is the label."""
    dense1, activation1, dense2, loss_activation = model
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    return float(np.mean(np.argmax(dense2.output, axis=1) == y))


def k_fold_accuracies(X, y, learning_rate, n_neurons=64, k=5, seed=0):
    """The k validation accuracies of one candidate, on folds and initial weights fixed by seed."""
    fold_accs = []
    for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=k, seed=seed):
        model = train(tr_X, tr_y, learning_rate, n_neurons, seed=100 + seed)   # a new network every fold
        fold_accs.append(accuracy(model, va_X, va_y))
    return np.array(fold_accs)


def draw_data(seed):
    """The 300 points every search works on, and a test set drawn next from the same stream."""
    np.random.seed(seed)
    X, y = spiral_data(samples=100, classes=3)              # (300, 2), (300,), sorted by class
    X_test, y_test = spiral_data(samples=100, classes=3)    # 300 further points, kept for the end
    return X, y, X_test, y_test


SETUP = ("setup: nnfs.init() (float32, float32 np.dot), spiral_data(samples=100, classes=3) after "
         "np.random.seed(s), Layer_Dense(2, 64), ReLU, Layer_Dense(64, 3), softmax and cross-entropy, "
         "0.01 * randn weights after np.random.seed(100 + s), Optimizer_Adam(learning_rate) with no decay, "
         "1,000 full-batch epochs, folds from np.random.default_rng(s)")


def floor_split_indices(n, k):
    """The faulty split of section 10: a fold size of n // k, the remainder forgotten."""
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    fold_size = n // k
    for i in range(k):
        val_idx = idx[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        yield train_idx, val_idx


if __name__ == "__main__":
    nnfs.init()
    X, y, X_test, y_test = draw_data(0)           # the test set is not used in this script
    print("X", X.shape, X.dtype, "  y", y.shape, "  first and last ten labels", y[:10], y[-10:])
    print()

    print("== Section 4: the folds partition the data")
    for n, k in ((300, 5), (300, 7), (10, 3)):
        folds = list(k_fold_indices(n, k))
        sizes = [len(val_idx) for train_idx, val_idx in folds]
        times_validated = np.bincount(np.concatenate([val_idx for train_idx, val_idx in folds]), minlength=n)
        overlap = max(len(np.intersect1d(train_idx, val_idx)) for train_idx, val_idx in folds)
        complete = all(len(train_idx) + len(val_idx) == n for train_idx, val_idx in folds)
        print(f"n={n:3d} k={k}  validation sizes {sizes}  every sample validated once: "
              f"{bool(np.all(times_validated == 1))}  largest train/validation overlap: {overlap}  "
              f"train + validation = n: {complete}")
    first = [val_idx for train_idx, val_idx in k_fold_indices(300, 5, seed=0)]
    again = [val_idx for train_idx, val_idx in k_fold_indices(300, 5, seed=0)]
    other = [val_idx for train_idx, val_idx in k_fold_indices(300, 5, seed=1)]
    print("same seed, same folds:", all(np.array_equal(a, b) for a, b in zip(first, again)),
          "  other seed, same folds:", all(np.array_equal(a, b) for a, b in zip(first, other)))
    state = np.random.get_state()[1].copy()
    list(k_fold_split(X, y))
    print("global random stream untouched by the split:", bool(np.array_equal(state, np.random.get_state()[1])))
    print("classes per validation fold, shuffled:",
          [np.bincount(val_y, minlength=3).tolist() for _, _, val_X, val_y in k_fold_split(X, y)])
    print()

    print("== Section 10: a fold size of n // k")
    for n, k in ((300, 5), (300, 7)):
        validated = np.concatenate([val_idx for train_idx, val_idx in floor_split_indices(n, k)])
        print(f"n={n} k={k}  fold size {n // k}  samples validated {len(np.unique(validated))}  "
              f"never validated {n - len(np.unique(validated))}")
    print()

    print("== Section 10: no shuffle on data sorted by class (learning rate 0.05)")
    for shuffle in (False, True):
        fold_accs = []
        counts = []
        for tr_X, tr_y, va_X, va_y in k_fold_split(X, y, k=5, shuffle=shuffle):
            counts.append(np.bincount(va_y, minlength=3).tolist())
            fold_accs.append(accuracy(train(tr_X, tr_y, 0.05, seed=100), va_X, va_y))
        print(f"shuffle={shuffle!s:5}  classes per validation fold {counts}")
        print(f"               fold accuracies {np.round(fold_accs, 3)}  mean {np.mean(fold_accs):.3f}")
