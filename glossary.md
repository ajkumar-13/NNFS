# Neural Networks from Scratch — Glossary

*A plain-English dictionary for every term you'll encounter in this series. Bookmark this page.*

---

## A

### Accuracy
The percentage of predictions the network gets right. Unlike loss, accuracy is a human-readable score: 0.85 means 85% correct. See [Part 8](posts/08-loss-categorical-cross-entropy/index.md).

### Activation Function
A function applied **after** the weighted sum + bias in a neuron. Without it, stacking layers is useless (the network stays linear). The two we build: **ReLU** (hidden layers) and **Softmax** (output layer). See [Part 6](posts/06-activation-functions-relu-and-softmax/index.md).

### Adam (Adaptive Moment Estimation)
The most popular optimizer. Combines **momentum** (smoothed gradient direction) with **RMSProp** (per-parameter step sizes) plus **bias correction**. See [Part 27](posts/27-adam-optimizer/index.md).

### AdaGrad (Adaptive Gradient)
An optimizer that automatically shrinks the learning rate for frequently-updated parameters. Problem: the learning rate can shrink to near-zero. See [Part 25](posts/25-adagrad/index.md).

---

## B

### Backpropagation (Backprop)
The algorithm to compute **how much each weight contributed to the error**. It applies the chain rule layer-by-layer from output back to input. See [Parts 12–21](posts/12-backprop-through-a-single-neuron/index.md).

### Batch
A group of training samples processed together. Instead of feeding 1 sample at a time, we feed a **matrix** of samples. This is faster and produces more stable gradients. Shape: `(batch_size, features)`.

### Bias
A constant added to the weighted sum in a neuron. One bias per neuron. It allows the neuron to shift its activation threshold. Formula: $z = Xw + b$. See [Part 1](posts/01-neurons-and-layers/index.md).

### Bias Correction
In Adam, the momentum and cache estimates start at zero and are biased toward zero early in training. Bias correction divides by $(1 - \beta^t)$ to compensate. See [Part 27](posts/27-adam-optimizer/index.md).

### Broadcasting
NumPy's automatic stretching of smaller arrays to match larger ones during arithmetic. Critical for adding biases (shape `(1, n)`) to batch outputs (shape `(batch, n)`). See [Part 5](posts/05-array-summation-keepdims-and-broadcasting/index.md).

---

## C

### Cache (Optimizer)
A running sum (AdaGrad) or moving average (RMSProp, Adam) of squared gradients. Used to adapt the learning rate per parameter. Large cache → smaller effective learning rate.

### Categorical Cross-Entropy
The standard loss function for multiclass classification. Measures how far predicted probabilities are from the true class. Formula: $L = -\log(\hat{y}_{\text{correct class}})$. See [Part 8](posts/08-loss-categorical-cross-entropy/index.md).

### Chain Rule
The calculus rule that lets us differentiate **composed functions**: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$. It's the mathematical foundation of backpropagation. See [Part 11](posts/11-the-chain-rule/index.md).

### Clipping
Clamping predicted probabilities to a tiny range like `[1e-7, 1-1e-7]` to prevent `log(0)` (which is negative infinity). See [Part 8](posts/08-loss-categorical-cross-entropy/index.md).

### Co-Adaptation
When neurons learn to depend on specific other neurons instead of learning useful features independently. Dropout prevents this. See [Part 31](posts/31-dropout/index.md).

---

## D

### Data Leakage
When information from the test set accidentally influences training — for example, normalizing using test-set statistics. This inflates performance metrics. See [Part 29](posts/29-validation-and-hyperparameter-tuning/index.md).

### Dead Neuron
A neuron whose ReLU input is always ≤ 0 across all training samples, so it never activates and its weights never update. Can be caused by a learning rate that's too high.

### Dense Layer (Fully Connected Layer)
A layer where **every** input is connected to **every** neuron. Computes $\text{output} = X \cdot W + b$. See [Part 4](posts/04-dense-layer-class-and-spiral-data/index.md).

### Derivative
The rate of change of a function. For $f(x) = x^2$, the derivative is $f'(x) = 2x$. Tells us which direction to adjust a parameter to reduce loss. See [Part 10](posts/10-derivatives-partial-derivatives-and-gradients/index.md).

### `dinputs`
The gradient with respect to a layer's **inputs** — the output of a layer's `backward()` method. It's the "glue" that chains backward passes together. See [Part 15](posts/15-gradients-with-respect-to-inputs/index.md).

### Dot Product
Multiply corresponding elements and sum: $\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + ...$. This is the core computation of a neuron. `np.dot()` in NumPy. See [Part 2](posts/02-numpy-and-the-dot-product/index.md).

### Dropout
A regularization technique that randomly sets a fraction of neuron outputs to **zero** during training. Forces the network to learn redundant representations. See [Part 31](posts/31-dropout/index.md).

---

## E

### Epoch
One complete pass through the entire training dataset. If you have 300 samples and train for 100 epochs, the network sees every sample 100 times.

### Epsilon ($\epsilon$)
A tiny constant (typically $10^{-7}$) added to denominators in optimizers to prevent division by zero.

---

## F

### Feature
One dimension (column) of the input data. For spiral data with shape `(300, 2)`, there are 2 features: X₁ and X₂.

### Forward Pass
Computing outputs layer-by-layer from input to the final prediction. Each layer applies: $F_n = f(F_{n-1} \cdot W_n + b_n)$. See [Part 7](posts/07-coding-the-complete-forward-pass/index.md).

---

## G

### Generalization
The ability of a trained model to perform well on **unseen** data — not just the training set. Good generalization is the ultimate goal. See [Part 28](posts/28-generalization-and-testing/index.md).

### Gradient
A vector of all partial derivatives — tells us the direction of steepest ascent for each parameter. We move **opposite** to the gradient to minimize loss. See [Part 10](posts/10-derivatives-partial-derivatives-and-gradients/index.md).

### Gradient Descent
The optimization algorithm: repeatedly update parameters by subtracting `learning_rate × gradient`. See [Part 22](posts/22-gradient-descent-optimiser/index.md).

---

## H

### Hidden Layer
Any layer between the input and output layers. Called "hidden" because we don't directly observe its values — they're internal to the network.

### Hyperparameter
A setting chosen **before** training, not learned from data. Examples: learning rate, number of neurons, number of layers, dropout rate. See [Part 29](posts/29-validation-and-hyperparameter-tuning/index.md).

---

## I

### Initialization
How weights and biases are set before training starts. Weights: small random values (`0.01 * np.random.randn(...)`). Biases: zeros. See [Part 4](posts/04-dense-layer-class-and-spiral-data/index.md).

---

## J

### Jacobian
A matrix of all partial derivatives of a vector-valued function. For element-wise functions (ReLU), it's diagonal. For coupled functions (softmax), it's a full matrix. See [Part 17](posts/17-backpropagation-through-activation-functions/index.md).

---

## K

### K-Fold Cross-Validation
A technique to evaluate model performance when data is limited. Split data into K folds, train K times (each time using a different fold as validation). Average the results. See [Part 29](posts/29-validation-and-hyperparameter-tuning/index.md).

### keepdims
A NumPy parameter that preserves array dimensions after operations like `sum()` or `max()`. Critical for correct broadcasting. `axis=1, keepdims=True` gives a column vector that broadcasts across columns. See [Part 5](posts/05-array-summation-keepdims-and-broadcasting/index.md).

---

## L

### L1 Regularization (Lasso)
A penalty term that adds the **absolute values** of weights to the loss: $\lambda \sum |w_i|$. Tends to push small weights to exactly zero (sparsity). See [Part 30](posts/30-l1-and-l2-regularisation/index.md).

### L2 Regularization (Ridge / Weight Decay)
A penalty that adds the **squared** weights to the loss: $\lambda \sum w_i^2$. Penalizes large weights more heavily than small ones. See [Part 30](posts/30-l1-and-l2-regularisation/index.md).

### Layer
A group of neurons that all receive the same inputs. Produces one output per neuron. A layer with 3 neurons and 4 inputs has a weight matrix of shape `(4, 3)` (our convention). See [Part 1](posts/01-neurons-and-layers/index.md).

### Learning Rate ($\alpha$)
The step size for gradient descent. Too high → oscillates/diverges. Too low → trains too slowly. Typical values: 0.001 to 1.0. See [Part 22](posts/22-gradient-descent-optimiser/index.md).

### Learning Rate Decay
Gradually reducing the learning rate during training. Starts coarse (explore) and ends fine (exploit). See [Part 23](posts/23-learning-rate-decay/index.md).

### Logits
The raw, unnormalized scores produced by the network's final layer **before** softmax. Softmax converts logits into probabilities.

### Loss
A single number measuring how wrong the network's predictions are. Lower is better. We use **categorical cross-entropy** for classification. See [Part 8](posts/08-loss-categorical-cross-entropy/index.md).

---

## M

### Matrix Multiplication
The operation that makes layers work: each entry `result[i][j]` = dot product of row `i` from matrix A with column `j` from matrix B. Inner dimensions must match: $(m \times n) \cdot (n \times p) = (m \times p)$. See [Part 2](posts/02-numpy-and-the-dot-product/index.md).

### Momentum
An optimizer enhancement that accumulates a running average of past gradients. Helps escape local minima and dampens oscillations. Formula: $v = \beta v + (1-\beta) \nabla$. See [Part 24](posts/24-momentum/index.md).

---

## N

### Neuron
The fundamental unit. Takes inputs, multiplies by weights, adds bias, optionally applies activation. $\text{output} = f(w \cdot x + b)$. See [Part 1](posts/01-neurons-and-layers/index.md).

### Non-linearity
Any function that isn't a straight line. Without non-linear activations, multiple layers collapse to a single linear operation — making depth useless. See [Part 6](posts/06-activation-functions-relu-and-softmax/index.md).

---

## O

### One-Hot Encoding
Representing class labels as vectors with a 1 at the correct class index and 0 everywhere else. Class 2 out of 3 → `[0, 0, 1]`. See [Part 8](posts/08-loss-categorical-cross-entropy/index.md).

### Optimizer
The algorithm that updates weights using gradients. Progression: SGD → SGD+Decay → Momentum → AdaGrad → RMSProp → Adam. See [Parts 22–27](posts/22-gradient-descent-optimiser/index.md).

### Overfitting
When the network memorizes training data but fails on new data. Diagnostic: training accuracy high, test accuracy low. See [Part 28](posts/28-generalization-and-testing/index.md).

---

## P

### Partial Derivative
The derivative of a multi-variable function with respect to **one** variable, treating others as constants. $\frac{\partial f}{\partial w_1}$ tells us how changing $w_1$ alone affects $f$. See [Part 10](posts/10-derivatives-partial-derivatives-and-gradients/index.md).

### Probability
A Softmax output — values between 0 and 1 that sum to 1 across all classes. Represents the network's confidence for each class.

---

## R

### ReLU (Rectified Linear Unit)
The most common hidden-layer activation: $\text{ReLU}(z) = \max(0, z)$. Passes positive values through; kills negative values. See [Part 6](posts/06-activation-functions-relu-and-softmax/index.md).

### RMSProp (Root Mean Square Propagation)
An optimizer that fixes AdaGrad's shrinking learning rate by using an exponential moving average of squared gradients instead of a cumulative sum. See [Part 26](posts/26-rmsprop/index.md).

### Regularization
Any technique that reduces overfitting by constraining the model. We cover **L1**, **L2**, and **Dropout**. See [Parts 30–31](posts/30-l1-and-l2-regularisation/index.md).

---

## S

### Sample
One row of the input data — a single example the network processes. With shape `(300, 2)`, there are 300 samples.

### Softmax
The activation function for the output layer in classification. Converts logits into a probability distribution: $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$. See [Part 6](posts/06-activation-functions-relu-and-softmax/index.md).

### Spiral Data
The non-linear classification dataset used throughout this series. Three intertwined spirals in 2D space (3 classes, 2 features). Not linearly separable. See [Part 4](posts/04-dense-layer-class-and-spiral-data/index.md).

---

## T

### Training Loop
The repeated cycle: forward pass → compute loss → backward pass → update weights. One iteration = one loop. One epoch = one pass through all data.

### Transpose
Flipping a matrix's rows and columns: shape `(m, n)` becomes `(n, m)`. Written as $W^T$ or `W.T` in NumPy. Essential for making dot product shapes match. See [Part 2](posts/02-numpy-and-the-dot-product/index.md).

---

## V

### Validation Set
A held-out portion of data used to tune hyperparameters **without** contaminating the test set. See [Part 29](posts/29-validation-and-hyperparameter-tuning/index.md).

---

## W

### Weight
A parameter that determines how much a specific input contributes to a neuron's output. Learned during training via gradient descent. See [Part 1](posts/01-neurons-and-layers/index.md).

### Weighted Sum
The core computation before activation: $z = x_1 w_1 + x_2 w_2 + ... + x_n w_n + b$. This is what `np.dot()` + bias computes.

---

## Symbols Quick Reference

| Symbol | Meaning | First Appears |
|--------|---------|:---:|
| $X$ | Input data / features | Part 1 |
| $W$ | Weight matrix | Part 1 |
| $b$ | Bias vector | Part 1 |
| $z$ | Pre-activation (weighted sum) | Part 6 |
| $a$ or $\hat{y}$ | Post-activation / prediction | Part 6 |
| $y$ | True label | Part 8 |
| $L$ | Loss value | Part 8 |
| $\alpha$ | Learning rate | Part 22 |
| $\beta$ | Momentum coefficient | Part 24 |
| $\rho$ | RMSProp decay rate | Part 26 |
| $\lambda$ | Regularization strength | Part 30 |
| $\epsilon$ | Small constant to prevent ÷0 | Part 25 |
| $\frac{\partial L}{\partial W}$ | Gradient of loss w.r.t. weights | Part 12 |

---

*For the full notation system used in this series, see the [Notation Guide](notation_guide.md).*
