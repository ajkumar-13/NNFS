# Capstone Project — MNIST Handwritten Digit Classification

*Apply everything you've built from scratch to a real dataset. No frameworks — just your own Dense layers, activations, loss, optimizers, and regularization.*

---

## The Challenge

Classify 28×28 grayscale images of handwritten digits (0–9) using the neural network you built across 31 lectures. Target: **95%+ test accuracy** using only NumPy.

---

## Why MNIST?

| Property | Value |
|---|---|
| Input size | 784 features (28 × 28 pixels, flattened) |
| Classes | 10 (digits 0–9) |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Difficulty | Moderate — achievable with a 2-layer network |

This is the natural next step after spiral data: real images, more features, more classes, more data.

---

## Part 1: Loading the Data

### Option A: Using `sklearn` (easiest)

```python
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Train: (60000, 784), Test: (10000, 784)
```

### Option B: Manual download (no sklearn)

```python
import urllib.request
import gzip
import struct
import numpy as np

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float64) / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Download files from the MNIST mirror
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

for name, fname in files.items():
    urllib.request.urlretrieve(base_url + fname, fname)

X_train = load_mnist_images("train-images-idx3-ubyte.gz")
y_train = load_mnist_labels("train-labels-idx1-ubyte.gz")
X_test  = load_mnist_images("t10k-images-idx3-ubyte.gz")
y_test  = load_mnist_labels("t10k-labels-idx1-ubyte.gz")
```

---

## Part 2: Visualize a Few Samples

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## Part 3: Paste Your From-Scratch Classes

Copy all the classes you built throughout the series:

```python
# Dense layer (Part 4, extended in Part 30 for regularization)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)
        # Regularization gradients
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU Activation (Part 6)
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# Softmax + Cross-Entropy combined (Parts 18–19)
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        # Softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Cross-entropy loss
        y_pred_clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_true)), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return np.mean(-np.log(correct_confidences))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


# Dropout (Part 31)
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Adam Optimizer (Part 27)
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1. + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights -= self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
```

---

## Part 4: Build the MNIST Network

```python
# Architecture: 784 → 128 → 128 → 10
dense1 = Layer_Dense(784, 128, weight_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)

dense2 = Layer_Dense(128, 128, weight_regularizer_l2=5e-4)
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(0.1)

dense3 = Layer_Dense(128, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-4)
```

**Why this architecture?**
- 128 hidden neurons: enough capacity for 10 classes without being massive
- 2 hidden layers: allows learning intermediate feature representations
- L2 regularization + dropout: prevents overfitting on 60K samples
- Adam: fast, reliable convergence

---

## Part 5: Train with Mini-Batches

Full-batch gradient descent on 60K samples is slow. Use **mini-batches**:

```python
EPOCHS = 20
BATCH_SIZE = 128

for epoch in range(EPOCHS):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    epoch_correct = 0
    n_batches = 0

    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward pass
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output, training=True)

        dense2.forward(dropout1.output)
        activation2.forward(dense2.output)
        dropout2.forward(activation2.output, training=True)

        dense3.forward(dropout2.output)
        data_loss = loss_activation.forward(dense3.output, y_batch)

        # Regularization loss
        reg_loss = 0
        for layer in [dense1, dense2, dense3]:
            if layer.weight_regularizer_l2 > 0:
                reg_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
        loss = data_loss + reg_loss

        # Accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        epoch_correct += np.sum(predictions == y_batch)

        # Backward pass
        loss_activation.backward(loss_activation.output, y_batch)
        dense3.backward(loss_activation.dinputs)
        dropout2.backward(dense3.dinputs)
        activation2.backward(dropout2.dinputs)
        dense2.backward(activation2.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dropout1.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

        epoch_loss += loss
        n_batches += 1

    # Epoch summary
    avg_loss = epoch_loss / n_batches
    train_acc = epoch_correct / len(X_train)
    print(f"Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | train_acc: {train_acc:.4f} | lr: {optimizer.current_learning_rate:.6f}")
```

---

## Part 6: Evaluate on Test Set

```python
# Forward pass with dropout DISABLED
dense1.forward(X_test)
activation1.forward(dense1.output)
dropout1.forward(activation1.output, training=False)  # ← disabled!

dense2.forward(dropout1.output)
activation2.forward(dense2.output)
dropout2.forward(activation2.output, training=False)  # ← disabled!

dense3.forward(dropout2.output)
test_loss = loss_activation.forward(dense3.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
test_acc = np.mean(predictions == y_test)

print(f"\nTest Results:")
print(f"  Loss:     {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f} ({int(test_acc * 10000)}/10000)")
```

**Expected:** ~96-97% test accuracy after 20 epochs.

---

## Part 7: Analyze Mistakes

```python
# Find misclassified samples
wrong = np.where(predictions != y_test)[0]
print(f"\nMisclassified: {len(wrong)} / {len(y_test)}")

# Show some mistakes
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = wrong[i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {y_test[idx]}, Pred: {predictions[idx]}")
    ax.axis('off')
plt.suptitle("Misclassified Digits")
plt.tight_layout()
plt.show()
```

---

## Part 8: Confusion Matrix

```python
confusion = np.zeros((10, 10), dtype=int)
for true, pred in zip(y_test, predictions):
    confusion[true][pred] += 1

print("\nConfusion Matrix:")
print("     ", "  ".join(f"{i}" for i in range(10)))
for i in range(10):
    row = "  ".join(f"{confusion[i][j]:4d}" for j in range(10))
    print(f"  {i}: {row}")
```

---

## Stretch Goals

| Goal | Difficulty | Hint |
|---|---|---|
| Reach 97%+ accuracy | Medium | Add a 3rd hidden layer (784→256→128→64→10) |
| Reach 98%+ accuracy | Hard | Larger layers + data augmentation (random shifts) |
| Implement mini-batch SGD with momentum | Medium | Use your `Optimizer_SGD` with momentum |
| Add learning rate warmup | Medium | Linearly increase LR for the first N iterations |
| Implement early stopping | Medium | Track validation loss, stop when it increases for 5 epochs |
| Per-class accuracy | Easy | Compute accuracy for each digit separately |
| Confusion analysis | Easy | Which digit pairs are most often confused? |
| Compare optimizers on MNIST | Hard | Run all 6 optimizers from the dashboard, compare convergence |

---

## What You've Demonstrated

By completing this capstone, you've shown mastery of:

- [x] **Dense layers** — Building `Layer_Dense` from scratch (Parts 1-4)
- [x] **Activations** — ReLU hidden activations + Softmax output (Part 6)
- [x] **Loss functions** — Categorical cross-entropy (Part 8)
- [x] **Backpropagation** — Full chain rule through every layer (Parts 10-21)
- [x] **Optimizers** — Adam with bias correction and decay (Parts 22-27)
- [x] **Regularization** — L2 weight penalty + dropout (Parts 28-31)
- [x] **Mini-batch training** — Processing data in chunks for efficiency
- [x] **Model evaluation** — Train/test split, accuracy, confusion analysis

**You built a neural network from zero and trained it on real data. No PyTorch, no TensorFlow — just NumPy and understanding.**

---

*See also: [Optimizer Dashboard](dashboards/Optimizer_Comparison.md) | [Regularization Dashboard](dashboards/Regularization_Comparison.md) | [Back to Index](INDEX.md)*
