# Conceptual Quizzes — Parts 1–31

*3–5 multiple-choice questions per lecture. Each targets a common misconception or tests a key concept. Answers with explanations are in the collapsible section after each quiz. (Quizzes for Parts 32–35 are forthcoming.)*

---

## Part 1 — Neurons and Layers

**Q1.** What does a single neuron compute?

- (A) A matrix multiplication
- (B) A weighted sum of inputs plus a bias
- (C) A non-linear activation function
- (D) A loss function

**Q2.** If a neuron has 4 inputs, how many weights does it have?

- (A) 1
- (B) 3
- (C) 4
- (D) 5

**Q3.** What is the purpose of the bias term?

- (A) To normalize the output
- (B) To shift the activation function left or right
- (C) To prevent overfitting
- (D) To reduce the number of parameters

<details>
<summary>Answers</summary>

**Q1: (B)** A neuron computes $z = \sum w_i x_i + b$. The activation function is a separate step applied *after* the neuron.

**Q2: (C)** One weight per input. The bias is an additional parameter, giving 5 total parameters, but only 4 *weights*.

**Q3: (B)** The bias allows the neuron to produce non-zero output even when all inputs are zero. It shifts the decision boundary.

</details>

---

## Part 2 — NumPy and the Dot Product

**Q1.** What does `np.dot(X, W)` compute when `X` is `(5, 3)` and `W` is `(3, 4)`?

- (A) Element-wise multiplication → shape `(5, 3)`
- (B) Matrix multiplication → shape `(5, 4)`
- (C) Matrix multiplication → shape `(3, 3)`
- (D) It raises a shape error

**Q2.** Why do we use `np.dot` instead of a Python loop?

- (A) It gives different results
- (B) It uses less memory
- (C) It's optimized with C/BLAS, so it's much faster
- (D) Python loops can't multiply arrays

**Q3.** What is the dot product of `[1, 2, 3]` and `[4, 5, 6]`?

- (A) `[4, 10, 18]`
- (B) 32
- (C) 15
- (D) `[5, 7, 9]`

<details>
<summary>Answers</summary>

**Q1: (B)** Matrix multiply: inner dimensions match (3 = 3), output shape is outer dimensions (5, 4).

**Q2: (C)** NumPy delegates to compiled C/Fortran (BLAS), which can be 100–1000× faster.

**Q3: (B)** $1 \times 4 + 2 \times 5 + 3 \times 6 = 4 + 10 + 18 = 32$.

</details>

---

## Part 3 — Stacking Layers

**Q1.** In a 2-layer network, what must be true about Layer 2's input size?

- (A) It must equal the batch size
- (B) It must equal Layer 1's number of neurons
- (C) It must equal Layer 1's input size
- (D) It can be anything

**Q2.** What changes when you add more layers to a network?

- (A) Only the training speed changes
- (B) The network can learn more complex patterns
- (C) The number of inputs changes
- (D) The loss function must change

**Q3.** If Layer 1 maps `(batch, 4) → (batch, 8)` and Layer 2 maps `(batch, 8) → (batch, 3)`, what is the total number of weight parameters?

- (A) 32
- (B) 24
- (C) 56
- (D) 59

<details>
<summary>Answers</summary>

**Q1: (B)** Each layer's input size must match the previous layer's output size (number of neurons).

**Q2: (B)** More layers → more compositions of functions → capacity to represent more complex decision boundaries.

**Q3: (C)** Layer 1: $4 \times 8 = 32$, Layer 2: $8 \times 3 = 24$. Total weights: $32 + 24 = 56$. (Note: 59 would include 8 + 3 = 11 biases.)

</details>

---

## Part 4 — Dense Layer Class & Spiral Data

**Q1.** Why do we multiply random initial weights by 0.01?

- (A) To make training faster
- (B) To prevent large outputs that destabilize softmax/loss
- (C) To satisfy a NumPy requirement
- (D) To ensure weights are positive

**Q2.** The spiral dataset with `samples=100, classes=3` produces how many data points?

- (A) 100
- (B) 200
- (C) 300
- (D) 303

**Q3.** Why do we initialize biases to zero instead of random values?

- (A) Random biases cause errors
- (B) Zero biases are a convention; the weights already break symmetry
- (C) Biases are not used during training
- (D) It prevents overfitting

<details>
<summary>Answers</summary>

**Q1: (B)** Large initial weights → large layer outputs → softmax becomes near-one-hot → high cross-entropy loss → unstable gradients.

**Q2: (C)** 100 samples per class × 3 classes = 300 total points.

**Q3: (B)** Biases at zero is fine because the random weights already ensure different neurons compute different functions.

</details>

---

## Part 5 — Summation & Broadcasting

**Q1.** What is `np.sum([[1,2],[3,4]], axis=0)`?

- (A) `[3, 7]`
- (B) `[4, 6]`
- (C) `10`
- (D) `[[1,2],[3,4]]`

**Q2.** What does `keepdims=True` do?

- (A) Keeps the array in memory
- (B) Preserves the reduced axis as size 1 instead of removing it
- (C) Prevents the sum from being computed
- (D) Makes the output a Python list

**Q3.** Can you add a `(5, 3)` array and a `(5,)` array?

- (A) Yes, broadcasting works
- (B) No, shapes are incompatible
- (C) Yes, but only with `keepdims=True`
- (D) Only with explicit reshaping

<details>
<summary>Answers</summary>

**Q1: (B)** `axis=0` sums along rows (vertically): `[1+3, 2+4] = [4, 6]`.

**Q2: (B)** Without it, `np.sum(X, axis=1)` has shape `(n,)`. With `keepdims=True`, shape is `(n, 1)`, which is crucial for broadcasting.

**Q3: (B)** `(5, 3)` + `(5,)` → NumPy aligns from the right: 3 ≠ 5, so it fails. You'd need `(3,)` or `(5, 1)`.

</details>

---

## Part 6 — Activation Functions (ReLU & Softmax)

**Q1.** What is `ReLU(-3)`?

- (A) -3
- (B) 3
- (C) 0
- (D) 1

**Q2.** Why must we subtract `max(z)` before computing softmax?

- (A) To make the output probabilities larger
- (B) To prevent numerical overflow in `exp()`
- (C) To ensure the output sums to 1
- (D) It's optional — just a style choice

**Q3.** Softmax outputs always:

- (A) Sum to 0
- (B) Sum to 1 and are all non-negative
- (C) Are between -1 and 1
- (D) Are integers

**Q4.** If all softmax inputs are equal, what is the output?

- (A) All zeros
- (B) All ones
- (C) Uniform distribution (all equal, summing to 1)
- (D) Error

<details>
<summary>Answers</summary>

**Q1: (C)** ReLU = max(0, x). For negative inputs, output is 0.

**Q2: (B)** `exp(1000)` overflows to infinity. Subtracting the max ensures the largest exponent is `exp(0) = 1`.

**Q3: (B)** Each element is $e^{z_i} / \sum e^{z_j}$ which is positive, and the sum over $i$ is 1.

**Q4: (C)** If all $z_i$ are equal, all $e^{z_i}$ are equal, so each softmax output is $1/K$ where $K$ is the number of classes.

</details>

---

## Part 7 — Complete Forward Pass

**Q1.** In our 2-layer classification network, what is the activation function of each layer?

- (A) ReLU for both layers
- (B) Softmax for both layers
- (C) ReLU for the hidden layer, Softmax for the output layer
- (D) Sigmoid for the hidden layer, Softmax for the output layer

**Q2.** If the softmax output for a sample is `[0.1, 0.7, 0.2]`, what is the predicted class?

- (A) Class 0
- (B) Class 1
- (C) Class 2
- (D) Cannot determine

**Q3.** With 3 classes and random weights, what accuracy do you expect?

- (A) 0%
- (B) ~33%
- (C) 50%
- (D) ~100%

<details>
<summary>Answers</summary>

**Q1: (C)** Hidden layer uses ReLU (non-linear, unbounded). Output layer uses Softmax (produces probabilities).

**Q2: (B)** `np.argmax([0.1, 0.7, 0.2])` = 1. The model predicts class 1 with 70% confidence.

**Q3: (B)** Random guessing among 3 classes gives ~1/3 = 33% accuracy.

</details>

---

## Part 8 — Loss Functions (Categorical Cross-Entropy)

**Q1.** What is the cross-entropy loss when the model predicts 100% confidence on the correct class?

- (A) 1.0
- (B) 0.0
- (C) Infinity
- (D) -1.0

**Q2.** Why do we clip predictions before computing `log()`?

- (A) To improve accuracy
- (B) To prevent `log(0)` which is negative infinity
- (C) To normalize the predictions
- (D) Clipping is optional

**Q3.** Cross-entropy loss is ___ when the model is confident and wrong.

- (A) Very low (near 0)
- (B) Exactly 1.0
- (C) Very high (large positive number)
- (D) Negative

<details>
<summary>Answers</summary>

**Q1: (B)** $-\ln(1.0) = 0$. Perfect confidence on the correct class = zero loss.

**Q2: (B)** If the model predicts exactly 0 for the true class, $\ln(0) = -\infty$. Clipping to $10^{-7}$ prevents this.

**Q3: (C)** If the model predicts near-0 for the true class, $-\ln(\epsilon)$ is very large. High confidence + wrong = high punishment.

</details>

---

## Part 9 — Introduction to Optimization

**Q1.** What is the main problem with random search for optimization?

- (A) It always finds the global minimum
- (B) It doesn't scale — exponentially many random guesses needed for high-dim problems
- (C) It requires gradients
- (D) It only works for convex functions

**Q2.** Gradient descent updates weights in which direction?

- (A) The direction of the gradient (steepest ascent)
- (B) The opposite direction of the gradient (steepest descent)
- (C) A random direction
- (D) Perpendicular to the gradient

**Q3.** If the learning rate is too large, what happens?

- (A) Training is slow but stable
- (B) The loss oscillates or diverges
- (C) The model underfits
- (D) Nothing — larger is always better

<details>
<summary>Answers</summary>

**Q1: (B)** With thousands of parameters, the probability of randomly finding good weights is vanishingly small.

**Q2: (B)** The gradient points uphill. We want to go downhill, so we subtract: $w \leftarrow w - \alpha \nabla L$.

**Q3: (B)** The updates overshoot the minimum, causing the loss to bounce around or increase.

</details>

---

## Part 10 — Derivatives, Gradients & Partial Derivatives

**Q1.** The derivative of $f(x) = x^3$ at $x = 2$ is:

- (A) 8
- (B) 6
- (C) 12
- (D) 4

**Q2.** A partial derivative $\partial f / \partial x$ treats which variables as constants?

- (A) $x$
- (B) All variables except $x$
- (C) No variables
- (D) All variables

**Q3.** The gradient vector points in the direction of:

- (A) Steepest decrease
- (B) Steepest increase
- (C) Zero change
- (D) The nearest minimum

<details>
<summary>Answers</summary>

**Q1: (C)** $f'(x) = 3x^2$. At $x = 2$: $3(4) = 12$.

**Q2: (B)** A partial derivative differentiates with respect to one variable, treating all others as constants.

**Q3: (B)** The gradient points toward steepest ascent. We negate it for gradient *descent*.

</details>

---

## Part 11 — The Chain Rule

**Q1.** The chain rule states that $\frac{d}{dx} f(g(x))$ equals:

- (A) $f'(x) \cdot g'(x)$
- (B) $f'(g(x)) \cdot g'(x)$
- (C) $f(g'(x))$
- (D) $f'(g(x)) + g'(x)$

**Q2.** In neural networks, the chain rule enables:

- (A) Forward propagation
- (B) Backward propagation — gradients flow from loss back through layers
- (C) Data preprocessing
- (D) Weight initialization

**Q3.** If $y = (2x + 1)^3$, what is $dy/dx$ at $x = 1$?

- (A) 6
- (B) 18
- (C) 27
- (D) 54

<details>
<summary>Answers</summary>

**Q1: (B)** Derivative of outer function evaluated at inner × derivative of inner. This is the chain rule.

**Q2: (B)** Backprop is just the chain rule applied repeatedly through the computation graph.

**Q3: (D)** Let $u = 2x + 1$, so $y = u^3$. $dy/dx = 3u^2 \cdot 2 = 6u^2$. At $x=1$: $u = 3$, so $6(9) = 54$.

</details>

---

## Part 12 — Backprop Through a Single Neuron

**Q1.** For $z = wx + b$, what is $\partial z / \partial w$?

- (A) $w$
- (B) $x$
- (C) $b$
- (D) $1$

**Q2.** The "upstream gradient" in backpropagation is:

- (A) The gradient of the loss with respect to the current layer's output
- (B) The gradient of the loss with respect to the inputs
- (C) Always equal to 1
- (D) The learning rate

**Q3.** Why is $\partial z / \partial b = 1$ always?

- (A) Because $b$ is always 1
- (B) Because $b$ appears as a simple addition — the derivative of $x + c$ with respect to $c$ is 1
- (C) It's a convention
- (D) Because we initialize $b = 1$

<details>
<summary>Answers</summary>

**Q1: (B)** $z = wx + b$. Taking the partial w.r.t. $w$: $\partial z / \partial w = x$.

**Q2: (A)** The upstream gradient $\partial L / \partial z$ is what the next layer (closer to the loss) sends back.

**Q3: (B)** $z = wx + b$. $\partial z / \partial b = 1$ because $b$ is added directly.

</details>

---

## Part 13 — Backprop Through a Layer

**Q1.** The gradient of loss w.r.t. weights is computed as:

- (A) `dweights = np.dot(inputs, dvalues)`
- (B) `dweights = np.dot(inputs.T, dvalues)`
- (C) `dweights = np.dot(dvalues, inputs)`
- (D) `dweights = inputs * dvalues`

**Q2.** Why do we sum `dvalues` along `axis=0` to get `dbiases`?

- (A) Because biases are shared across the batch
- (B) Because biases are always zero
- (C) To normalize by batch size
- (D) It's an approximation

**Q3.** `dweights` has the same shape as:

- (A) The input
- (B) The output
- (C) The weight matrix
- (D) The bias vector

<details>
<summary>Answers</summary>

**Q1: (B)** $\partial L / \partial W = X^T \cdot \text{dvalues}$. The transpose is necessary to get the right shape.

**Q2: (A)** Each sample contributes a gradient for the shared bias; we sum them because the bias is the same for all samples.

**Q3: (C)** Gradients always have the same shape as the parameter they correspond to.

</details>

---

## Part 14 — Matrices in Backpropagation

**Q1.** In the backward pass, why do we use the transpose of the weight matrix?

- (A) To randomize the gradients
- (B) To reverse the dimension mapping from the forward pass
- (C) Transposing is optional
- (D) To make the matrix square

**Q2.** If the forward pass computes `output = np.dot(X, W)`, the backward pass for `dinputs` is:

- (A) `np.dot(W, dvalues)`
- (B) `np.dot(dvalues, W.T)`
- (C) `np.dot(W.T, dvalues)`
- (D) `np.dot(dvalues, W)`

<details>
<summary>Answers</summary>

**Q1: (B)** Forward: `(batch, n_in) × (n_in, n_out) → (batch, n_out)`. Backward: `(batch, n_out) × (n_out, n_in) → (batch, n_in)`. The transpose reverses the mapping.

**Q2: (B)** `dinputs = np.dot(dvalues, W.T)`. This maps the gradient from output space back to input space.

</details>

---

## Part 15 — Gradients w.r.t. Inputs

**Q1.** Why do we compute gradients with respect to *inputs* if inputs are not trainable?

- (A) To compute the loss
- (B) To pass gradients to the previous layer (which needs them for *its* weight gradients)
- (C) Inputs are trainable
- (D) For data augmentation

**Q2.** The shape of `dinputs` matches:

- (A) The output shape
- (B) The weight shape
- (C) The input shape
- (D) A scalar

<details>
<summary>Answers</summary>

**Q1: (B)** The "inputs" of layer $k$ are the "outputs" of layer $k-1$. Layer $k-1$ needs $\partial L / \partial \text{output}_{k-1}$ to compute its own weight gradients.

**Q2: (C)** Always. `dinputs.shape == inputs.shape`.

</details>

---

## Part 16 — Coding Backpropagation

**Q1.** After computing gradients, a simple weight update is:

- (A) `weights += lr * dweights`
- (B) `weights -= lr * dweights`
- (C) `weights = dweights`
- (D) `weights -= dweights`

**Q2.** What happens if you forget to reset gradients between training iterations?

- (A) Nothing, gradients are overwritten
- (B) Gradients accumulate across iterations, causing incorrect updates
- (C) The model converges faster
- (D) An error is raised

<details>
<summary>Answers</summary>

**Q1: (B)** We subtract to move in the direction of decreasing loss. The learning rate `lr` controls step size.

**Q2: (B)** If gradients aren't reset, they accumulate and the update direction becomes a mix of current and past gradients — a subtle bug.

</details>

---

## Part 17 — Backprop Through Activation Functions

**Q1.** For ReLU backward, if the forward input was negative, the gradient is:

- (A) The input value
- (B) 1
- (C) 0
- (D) -1

**Q2.** A "dead neuron" is one that:

- (A) Has very large weights
- (B) Always outputs 0 because its pre-ReLU input is always negative
- (C) Has been removed from the network
- (D) Has zero bias

**Q3.** Leaky ReLU addresses dead neurons by:

- (A) Removing the activation function
- (B) Allowing a small non-zero gradient for negative inputs
- (C) Increasing the learning rate
- (D) Using softmax instead

<details>
<summary>Answers</summary>

**Q1: (C)** ReLU derivative is 0 for negative inputs, 1 for positive. The gradient gets zeroed out.

**Q2: (B)** If a neuron's input is always negative, ReLU always outputs 0, and its gradient is always 0 — it can never recover.

**Q3: (B)** Leaky ReLU uses a small slope (e.g., 0.01) for negative inputs, keeping a small gradient alive.

</details>

---

## Part 18 — Backprop Through the Loss Function

**Q1.** The gradient of cross-entropy loss w.r.t. the predicted probability for the true class is:

- (A) $\hat{y}_k$
- (B) $-1/\hat{y}_k$
- (C) $1 - \hat{y}_k$
- (D) $\hat{y}_k - 1$

**Q2.** Why is the combined softmax + cross-entropy backward pass preferred over computing them separately?

- (A) It uses less code
- (B) It's numerically more stable and computationally simpler
- (C) The separate approach gives wrong answers
- (D) There's no difference

<details>
<summary>Answers</summary>

**Q1: (B)** $\partial L / \partial \hat{y}_k = -1 / \hat{y}_k$ for the true class. This is why clipping is important.

**Q2: (B)** Computing the Jacobian of softmax separately involves large intermediate values. The combined shortcut $\hat{y} - y$ avoids this entirely.

</details>

---

## Part 19 — Softmax Derivatives

**Q1.** The Jacobian of softmax is a:

- (A) Scalar
- (B) Vector
- (C) Matrix (because each output depends on *all* inputs)
- (D) 3D tensor

**Q2.** For the diagonal elements of the softmax Jacobian, $\partial S_i / \partial z_i$ equals:

- (A) $S_i$
- (B) $S_i(1 - S_i)$
- (C) $-S_i S_j$
- (D) $1 - S_i$

**Q3.** The combined softmax + cross-entropy gradient simplifies to:

- (A) $\hat{y}^2 - y$
- (B) $\hat{y} - y$ (predictions minus targets)
- (C) $y - \hat{y}$
- (D) $-\ln(\hat{y})$

<details>
<summary>Answers</summary>

**Q1: (C)** Since $S_i = e^{z_i} / \sum e^{z_j}$, each output depends on all inputs. The Jacobian is $K \times K$.

**Q2: (B)** Diagonal: $S_i(1 - S_i)$. Off-diagonal ($i \neq j$): $-S_i S_j$.

**Q3: (B)** The beautifully simple result: just subtract the one-hot true label from the softmax prediction.

</details>

---

## Part 20 — Assembling Full Backpropagation

**Q1.** The correct order for the backward pass is:

- (A) Input → Dense 1 → ReLU → Dense 2 → Softmax → Loss
- (B) Loss → Softmax+CCE → Dense 2 → ReLU → Dense 1
- (C) Dense 1 → Dense 2 → Loss
- (D) Any order works

**Q2.** What "seeds" the backward pass?

- (A) The initial weights
- (B) A gradient of 1.0 from the loss function
- (C) The input data
- (D) The learning rate

<details>
<summary>Answers</summary>

**Q1: (B)** Backward pass reverses the forward pass. Start at the loss and propagate back.

**Q2: (B)** The loss is the starting point. $\partial L / \partial L = 1$. This seeds the chain rule.

</details>

---

## Part 21 — Coding Complete Backpropagation

**Q1.** After one complete backward pass, which quantities have been computed?

- (A) Only the loss
- (B) `dweights`, `dbiases`, and `dinputs` for every layer; `dinputs` for every activation
- (C) Only the final layer's gradients
- (D) The updated weights

**Q2.** If the initial loss is ~1.1 for 3-class classification, this suggests:

- (A) The model is well-trained
- (B) The model is predicting near-random (since $-\ln(1/3) \approx 1.099$)
- (C) The model has converged
- (D) The loss function has a bug

**Q3.** After manually updating weights with `weights -= 0.01 * dweights`, the loss should:

- (A) Always increase
- (B) Decrease (if the learning rate is appropriate)
- (C) Stay exactly the same
- (D) Become zero

<details>
<summary>Answers</summary>

**Q1: (B)** A complete backward pass computes all gradients for all layers and activations.

**Q2: (B)** Loss of 1.099 = random performance for 3 classes. This is the expected starting point before training.

**Q3: (B)** Gradient descent moves in the direction that decreases loss. One update should reduce it (assuming the LR isn't too high).

</details>

---

## Part 22 — Gradient Descent

**Q1.** What is the learning rate?

- (A) How fast the computer runs
- (B) A multiplier that controls the step size of each weight update
- (C) The number of training epochs
- (D) The accuracy of the model

**Q2.** A learning rate that is too small will:

- (A) Cause the loss to diverge
- (B) Lead to very slow convergence
- (C) Cause overfitting
- (D) Produce perfect accuracy quickly

**Q3.** Gradient descent is guaranteed to find the global minimum when:

- (A) Always
- (B) The learning rate is small enough
- (C) The loss function is convex
- (D) The network is deep enough

<details>
<summary>Answers</summary>

**Q1: (B)** $w \leftarrow w - \alpha \nabla L$. $\alpha$ is the learning rate — it scales the gradient.

**Q2: (B)** Small steps → slow progress. It will eventually converge, but may take impractically many epochs.

**Q3: (C)** Only for convex functions. Neural network loss surfaces are non-convex — GD finds *a* local minimum (or saddle point), not necessarily the global one.

</details>

---

## Part 23 — Learning Rate Decay

**Q1.** Learning rate decay changes the LR:

- (A) Once at the beginning
- (B) Gradually over the course of training
- (C) Only when loss increases
- (D) Randomly each epoch

**Q2.** With $\alpha_t = \alpha_0 / (1 + d \cdot t)$ and $d = 0.001$, the LR is halved at approximately which epoch?

- (A) 100
- (B) 500
- (C) 1,000
- (D) 10,000

**Q3.** Why use decay instead of just picking a smaller LR?

- (A) Decay is always faster
- (B) A high initial LR allows fast early progress, then decay reduces oscillation near the minimum
- (C) Small LRs don't work
- (D) Decay prevents overfitting

<details>
<summary>Answers</summary>

**Q1: (B)** The LR decreases every iteration (or every epoch, depending on implementation).

**Q2: (C)** $\alpha_0 / (1 + 0.001 \times t) = \alpha_0 / 2$ when $t = 1000$.

**Q3: (B)** Fast early exploration + gentle fine-tuning near convergence. Best of both worlds.

</details>

---

## Part 24 — Momentum

**Q1.** Momentum adds what to gradient descent?

- (A) A random perturbation
- (B) An exponential moving average of past gradients (velocity)
- (C) A second-order derivative (Hessian)
- (D) Regularization

**Q2.** With $\beta = 0.9$, the velocity is approximately an average of the last ___ gradients:

- (A) 2
- (B) 5
- (C) 10
- (D) 100

**Q3.** Momentum helps most when the loss landscape has:

- (A) A single global minimum
- (B) Narrow ravines where plain SGD oscillates
- (C) No local minima
- (D) Uniform curvature

<details>
<summary>Answers</summary>

**Q1: (B)** Momentum maintains a running average of gradients, smoothing out noise and oscillations.

**Q2: (C)** The effective window is approximately $1/(1 - \beta) = 1/(1 - 0.9) = 10$.

**Q3: (B)** In narrow ravines, SGD bounces side-to-side. Momentum cancels the oscillation and accelerates along the ravine.

</details>

---

## Part 25 — AdaGrad

**Q1.** AdaGrad adapts the learning rate per:

- (A) Epoch
- (B) Layer
- (C) Individual parameter
- (D) Batch

**Q2.** AdaGrad's main limitation is:

- (A) It requires too much memory
- (B) The accumulated cache grows forever, causing the effective LR to approach zero
- (C) It doesn't support biases
- (D) It's slower than plain SGD

**Q3.** The $\epsilon$ in AdaGrad's update rule prevents:

- (A) Overfitting
- (B) Division by zero
- (C) Large gradients
- (D) Negative learning rates

<details>
<summary>Answers</summary>

**Q1: (C)** Each weight gets its own effective LR based on the sum of its squared gradients.

**Q2: (B)** $G_t = \sum_{i=1}^{t} g_i^2$ only grows → $\alpha / \sqrt{G + \epsilon}$ → 0. Learning eventually stops.

**Q3: (B)** When $G$ is 0 (early training or for some parameters), dividing by $\sqrt{0}$ would crash. $\epsilon \approx 10^{-7}$ prevents this.

</details>

---

## Part 26 — RMSProp

**Q1.** RMSProp fixes AdaGrad by:

- (A) Using a larger starting learning rate
- (B) Replacing the sum with an exponential moving average of squared gradients
- (C) Adding momentum
- (D) Using a fixed cache size

**Q2.** The $\rho$ parameter in RMSProp controls:

- (A) The learning rate
- (B) How quickly old gradient information is forgotten
- (C) The batch size
- (D) Number of epochs

**Q3.** With $\rho = 0.9$, the effective window of past gradients is approximately:

- (A) 5
- (B) 10
- (C) 90
- (D) 100

<details>
<summary>Answers</summary>

**Q1: (B)** EMA: $G_t = \rho \cdot G_{t-1} + (1 - \rho) \cdot g_t^2$. Old information is gradually forgotten.

**Q2: (B)** High $\rho$ → more memory (slow decay). Low $\rho$ → fast forgetting (responsive).

**Q3: (B)** Effective window ≈ $1 / (1 - \rho) = 1 / 0.1 = 10$.

</details>

---

## Part 27 — Adam Optimizer

**Q1.** Adam combines which two techniques?

- (A) SGD + L2 regularization
- (B) Momentum + RMSProp (adaptive moment estimation)
- (C) AdaGrad + Dropout
- (D) Learning rate decay + random search

**Q2.** Bias correction in Adam is most important during:

- (A) The last few epochs
- (B) The first few iterations (when moments are near zero due to initialization)
- (C) Every iteration equally
- (D) Only when using decay

**Q3.** Common Adam defaults are:

- (A) $\alpha = 0.1$, $\beta_1 = 0.5$, $\beta_2 = 0.5$
- (B) $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$
- (C) $\alpha = 1.0$, $\beta_1 = 0.99$, $\beta_2 = 0.99$
- (D) $\alpha = 0.01$, $\beta_1 = 0.1$, $\beta_2 = 0.1$

<details>
<summary>Answers</summary>

**Q1: (B)** Adam = **Ada**ptive **M**oment estimation. First moment (mean) ≈ momentum, second moment (variance) ≈ RMSProp.

**Q2: (B)** Moments initialized to 0 → biased low at the start. Dividing by $(1 - \beta^t)$ corrects this, and the correction fades as $t$ grows.

**Q3: (B)** These are the defaults from the original Adam paper (Kingma & Ba, 2014).

</details>

---

## Part 28 — Generalization & Testing

**Q1.** A model that achieves 99% train accuracy but 70% test accuracy is:

- (A) Underfitting
- (B) Overfitting
- (C) Perfectly trained
- (D) Broken

**Q2.** The test set should be used:

- (A) During training to improve the model
- (B) Only once, at the very end, to estimate real-world performance
- (C) To tune hyperparameters
- (D) To initialize weights

**Q3.** Which is NOT a sign of overfitting?

- (A) Training loss continues to decrease while test loss increases
- (B) Training and test accuracy are both ~85%
- (C) The model memorizes noise in the training data
- (D) Large gap between train and test performance

<details>
<summary>Answers</summary>

**Q1: (B)** A 29% gap between train and test = classic overfitting. The model memorized the training data.

**Q2: (B)** Using test data during training or tuning leaks information, making the test score overly optimistic.

**Q3: (B)** If train and test are similar (~85%), the model generalizes well — no overfitting.

</details>

---

## Part 29 — Validation & Hyperparameter Tuning

**Q1.** The validation set is used to:

- (A) Train the model
- (B) Select hyperparameters (LR, architecture, regularization)
- (C) Replace the test set
- (D) Initialize the network

**Q2.** In k-fold cross-validation, each sample appears in the validation set:

- (A) Never
- (B) Exactly once
- (C) k times
- (D) Randomly

**Q3.** "Data leakage" occurs when:

- (A) Data files are corrupted
- (B) Information from the test/validation set influences training decisions
- (C) The dataset is too small
- (D) The learning rate is too high

<details>
<summary>Answers</summary>

**Q1: (B)** Validation = used during development to pick the best model configuration. Test = final evaluation only.

**Q2: (B)** Each fold serves as validation exactly once, so every sample is validated exactly once across all $k$ folds.

**Q3: (B)** Any path by which test-time information reaches training — even indirectly (e.g., tuning on test accuracy) — is leakage.

</details>

---

## Part 30 — L1 & L2 Regularization

**Q1.** L2 regularization adds which term to the loss?

- (A) $\lambda \sum |w|$
- (B) $\lambda \sum w^2$
- (C) $\lambda \sum w$
- (D) $\lambda \sum \sqrt{w}$

**Q2.** L1 regularization tends to:

- (A) Make all weights equal
- (B) Push some weights to exactly zero (sparsity)
- (C) Increase all weights
- (D) Have no effect on weights

**Q3.** The term "weight decay" is most closely associated with:

- (A) L1 regularization
- (B) L2 regularization (because the gradient $2\lambda w$ shrinks weights proportionally)
- (C) Dropout
- (D) Learning rate decay

**Q4.** If $\lambda$ is too large, the model will:

- (A) Overfit
- (B) Underfit (weights shrink too much, model is too constrained)
- (C) Train faster
- (D) Have more parameters

<details>
<summary>Answers</summary>

**Q1: (B)** L2 adds $\lambda \sum w_i^2$. The squared term penalizes large weights quadratically.

**Q2: (B)** L1's gradient is $\pm \lambda$ (constant magnitude), which pushes small weights all the way to zero.

**Q3: (B)** L2 regularization = weight decay. The update becomes $w \leftarrow w(1 - 2\alpha\lambda) - \alpha \nabla L$, which decays the weight at each step.

**Q4: (B)** Excessive regularization constrains the model too much — it can't learn the underlying pattern.

</details>

---

## Part 31 — Dropout

**Q1.** Dropout randomly zeroes out neurons during:

- (A) Training only
- (B) Testing only
- (C) Both training and testing
- (D) Initialization only

**Q2.** "Inverted dropout" scales activations by $1/(1-p)$ during training so that:

- (A) Outputs are larger during training
- (B) No scaling is needed at test time — expected values match
- (C) The model trains faster
- (D) Dropout has no effect

**Q3.** A dropout rate of 0.5 means:

- (A) 50% of neurons are always active
- (B) Each neuron has a 50% chance of being zeroed on each forward pass
- (C) The model uses half the layers
- (D) Training takes half as long

**Q4.** Dropout prevents overfitting by:

- (A) Reducing the number of parameters
- (B) Forcing neurons to be independently useful (preventing co-adaptation)
- (C) Adding noise to the loss function
- (D) Increasing the learning rate

<details>
<summary>Answers</summary>

**Q1: (A)** At test time, all neurons are active. Dropout is a *training-time* regularizer.

**Q2: (B)** Scaling up during training ensures the expected activation magnitude is the same as at test time, so no adjustment is needed at inference.

**Q3: (B)** Each neuron independently has a 50% chance of being dropped on each forward pass. Different samples see different subsets.

**Q4: (B)** No neuron can rely on specific other neurons being present → each must learn independently useful features.

</details>

---

## Scoring Guide

| Score | Level |
|---|---|
| 90–100% | Mastery — you have a strong conceptual foundation |
| 70–89% | Solid — review the flagged topics |
| 50–69% | Developing — re-read the corresponding blog posts |
| Below 50% | Restart from the section that challenges you most |

---

*See also: [Exercises](exercises.md) | [Cheat Sheets](cheatsheets/) | [Back to Index](INDEX.md)*
