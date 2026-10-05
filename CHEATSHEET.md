# Cheatsheet

The formulas, shapes, and code of Neural Networks from Scratch on one page. A batch is $N$ rows; symbols are those of the [notation guide](notation_guide.md), and terms are defined in the [glossary](GLOSSARY.md).

## Shapes

| Object | Shape | Code |
|---|---|---|
| input batch $\mathbf{X}$ | $(N, n_\text{inputs})$ | `X` |
| weights $\mathbf{W}$ (from Post 04) | $(n_\text{inputs}, n_\text{neurons})$ | `0.01 * np.random.randn(n_inputs, n_neurons)` |
| biases $\mathbf{b}$ | $(1, n_\text{neurons})$ | `np.zeros((1, n_neurons))` |
| layer output $\mathbf{Z}$ | $(N, n_\text{neurons})$ | `np.dot(inputs, weights) + biases` |
| labels $\mathbf{y}$ | $(N,)$ indices or $(N, K)$ one-hot | `y` |

- Matrix product: $(m, n) \cdot (n, p) \rightarrow (m, p)$; the inner sizes must match. A `Layer_Dense(n_in, n_out)` turns $(N, n_\text{in})$ into $(N, n_\text{out})$; activations keep the shape.
- Posts 01 to 03 store weights the other way round, $(n_\text{neurons}, n_\text{inputs})$, and compute `np.dot(inputs, weights.T)`.
- A dense layer has $n_\text{in} \cdot n_\text{out} + n_\text{out}$ parameters; a network has the sum over its layers. A $4 \to 8 \to 3$ network has $40 + 27 = 67$.
- Reductions: on a $(3, 4)$ array, `axis=0` gives $(4,)$, `axis=1` gives $(3,)$, no axis gives a scalar; `keepdims=True` gives $(1, 4)$, $(3, 1)$, $(1, 1)$.
- Broadcasting compares shapes from the last axis; sizes are compatible when equal or when one is 1. $(5, 3) + (3,)$ and $(5, 3) + (5, 1)$ work; $(5, 3) + (5,)$ fails, because $(5,)$ is aligned with the last axis, of size 3.

## Forward pass

| Step | Formula | Code |
|---|---|---|
| neuron | $z = \sum_i w_i x_i + b$ | `np.dot(weights, inputs) + bias` |
| dense layer | $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ | `np.dot(inputs, self.weights) + self.biases` |
| ReLU | $\max(0, z)$ | `np.maximum(0, inputs)` |
| softmax | $\hat{y}_k = e^{z_k} / \sum_j e^{z_j}$ per row | `e = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))`, then `e / np.sum(e, axis=1, keepdims=True)` |
| sigmoid | $\sigma(z) = 1/(1 + e^{-z})$ | `1 / (1 + np.exp(-inputs))` |

Subtracting the row maximum before `np.exp` changes nothing mathematically and prevents overflow. Equal logits give a uniform softmax, $1/K$ for every class.

## Loss and accuracy

| Quantity | Formula | Note |
|---|---|---|
| categorical cross-entropy | $L = \frac{1}{N} \sum_i -\log \hat{y}_{i, c_i}$ | $c_i$ is sample $i$'s true class; with one-hot rows, `np.sum(y_pred * y_true, axis=1)` picks $\hat{y}_{i, c_i}$ |
| binary cross-entropy | $L = -\frac{1}{N} \sum_i [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]$ | sigmoid output, labels in $\{0, 1\}$ |
| clipping | `np.clip(y_pred, 1e-7, 1 - 1e-7)` | before every `log`, so a zero probability cannot give an infinite loss |
| accuracy | `np.mean(np.argmax(probs, axis=1) == y)` | for one-hot labels, take `np.argmax(y, axis=1)` first |

Reference values: a uniform guess over $K$ classes has loss $\ln K$ ($\ln 3 \approx 1.099$ on the spiral) and accuracy about $1/K$; a probability of 0.5 on the true class costs $\ln 2 \approx 0.693$.

## Derivatives

| Rule | Formula |
|---|---|
| definition | $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$ |
| power, sum, product | $(x^n)' = n x^{n-1}$, $(f + g)' = f' + g'$, $(fg)' = f'g + fg'$ |
| partial derivative | differentiate in one variable, hold the others constant |
| gradient | $\nabla f = (\partial f / \partial x_1, \dots, \partial f / \partial x_n)$, the direction of steepest increase |
| chain rule | $\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$, multiplied along every link of a longer chain |
| one neuron | for $z = w x + b$: $\partial z / \partial w = x$, $\partial z / \partial x = w$, $\partial z / \partial b = 1$ |

## Backward pass

Each component takes the gradient of its output, `dvalues`, and stores the gradient of its input, `dinputs`, which the previous component receives.

```python
# Layer_Dense.backward
self.dweights = np.dot(self.inputs.T, dvalues)           # (n_inputs, n_neurons)
self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)    # (1, n_neurons)
self.dinputs  = np.dot(dvalues, self.weights.T)           # (N, n_inputs)

# Activation_ReLU.backward
self.dinputs = dvalues.copy()          # copy, so the caller's array is not changed
self.dinputs[self.inputs <= 0] = 0

# Activation_Softmax_Loss_CategoricalCrossentropy.backward: (y_hat - y) / N
samples = len(dvalues)
if len(y_true.shape) == 2:
    y_true = np.argmax(y_true, axis=1)
self.dinputs = dvalues.copy()
self.dinputs[range(samples), y_true] -= 1
self.dinputs = self.dinputs / samples

# Activation_Sigmoid_Loss_BinaryCrossentropy.backward: (y_hat - y) / N
self.dinputs = (dvalues - y_true.reshape(-1, 1)) / len(dvalues)
```

- Softmax alone has a full Jacobian, $\partial \hat{y}_i / \partial z_j = \hat{y}_i(\delta_{ij} - \hat{y}_j)$; combined with cross-entropy it collapses to $(\hat{\mathbf{y}} - \mathbf{y})/N$, so the series always backpropagates through the combined class.
- Backward order for $2 \to 64 \to 3$: loss and softmax $(N, 3)$, `dense2` (`dweights` $(64, 3)$, `dinputs` $(N, 64)$), ReLU $(N, 64)$, `dense1` (`dweights` $(2, 64)$, `dinputs` $(N, 2)$).
- Gradient check: perturb one parameter by $\pm h$ and compare $\frac{L(\theta + h) - L(\theta - h)}{2h}$ with the analytic value. The central difference has error of order $h^2$; with $h = 10^{-5}$ in float64 a correct backward pass gives a relative error near $10^{-7}$ or below, and above $10^{-3}$ it is almost certainly a bug. Two cases inflate the error without a bug: gradients that are themselves tiny (with the `0.01` initialisation some bias gradients are near $10^{-8}$, where rounding dominates; check with a larger initial scale or an absolute tolerance), and float32 arrays, which `nnfs.init()` produces.

## Optimisers

All share one contract: `pre_update_params()` once per step, `update_params(layer)` per layer, `post_update_params()` to count the step. Here $g$ is the gradient and $t$ the update count.

| Optimiser | Update | Defaults in the series' class |
|---|---|---|
| gradient descent | $\theta \leftarrow \theta - \alpha g$ | `Optimizer_SGD(learning_rate=1.0)` |
| decay (any optimiser) | $\alpha_t = \alpha_0 / (1 + d \cdot t)$ | `decay=0.0`; the rate halves when $d \cdot t = 1$ |
| momentum | $v \leftarrow \beta v - \alpha g$, then $\theta \leftarrow \theta + v$ | `momentum=0.0`; 0.9 is the usual choice |
| AdaGrad | $G \leftarrow G + g^2$, $\theta \leftarrow \theta - \alpha g / (\sqrt{G} + \epsilon)$ | `learning_rate=1.0, epsilon=1e-7` |
| RMSProp | $G \leftarrow \rho G + (1 - \rho) g^2$, $\theta \leftarrow \theta - \alpha g / (\sqrt{G} + \epsilon)$ | `learning_rate=0.02, epsilon=1e-7, rho=0.9` |
| Adam | $m \leftarrow \beta_1 m + (1 - \beta_1) g$, $v \leftarrow \beta_2 v + (1 - \beta_2) g^2$, $\hat{m} = m / (1 - \beta_1^t)$, $\hat{v} = v / (1 - \beta_2^t)$, $\theta \leftarrow \theta - \alpha \hat{m} / (\sqrt{\hat{v}} + \epsilon)$ | `learning_rate=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999` |

- Classical momentum has no $(1 - \beta)$ factor; that factor belongs to Adam's moving averages.
- $\epsilon$ sits outside the square root, as in the code.
- An exponential moving average with decay $\rho$ remembers roughly the last $1/(1 - \rho)$ steps: 10 for 0.9, 1000 for 0.999.
- Adam's $t$ counts from 1 (the code uses `iterations + 1`), so the first corrected moment equals the first gradient.

## Generalisation and regularisation

- Splits: training 60 to 80 percent (learn the parameters), validation 10 to 20 percent (choose hyperparameters), test 10 to 20 percent (report once, after every decision is frozen). With few samples, use k-fold cross-validation on the training data. Fit any preprocessing statistics on the training data only.
- The test pass is forward only: no backward pass, no update, dropout off.

| Technique | Added to the loss | Added to `dweights` |
|---|---|---|
| L1 | $\lambda \sum \lvert w \rvert$ | $\lambda \, \text{sign}(w)$: `dL1 = np.ones_like(w); dL1[w < 0] = -1` |
| L2 | $\lambda \sum w^2$ | $2 \lambda w$ |

The penalty is computed per layer (`weight_regularizer_l1`, `weight_regularizer_l2`, and the bias versions), added to the data loss when the loss is reported, and its gradient is added in `Layer_Dense.backward`. Choose $\lambda$ on the validation set, never on the test set.

```python
# Layer_Dropout: rate p is the drop probability; the class keeps 1 - p
self.rate = 1 - rate                                   # in __init__
self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
self.output = inputs * self.binary_mask                # training only
self.dinputs = dvalues * self.binary_mask              # backward uses the same mask
```

Dividing by the keep rate $1 - p$ (inverted dropout) keeps the expected activation the same in training and evaluation, so evaluation simply skips the layer. A new mask is drawn on every forward pass.

## Practical training

```python
for epoch in range(EPOCHS):
    idx = np.random.permutation(len(X_train))          # reshuffle once per epoch
    X_shuf, y_shuf = X_train[idx], y_train[idx]
    for start in range(0, len(X_train), BATCH_SIZE):   # ceil(N / B) steps per epoch
        X_batch = X_shuf[start:start + BATCH_SIZE]
        y_batch = y_shuf[start:start + BATCH_SIZE]
        # forward, loss, backward, then
        # optimizer.pre_update_params(); optimizer.update_params(layer) for each layer;
        # optimizer.post_update_params()
```

Create the optimiser once, before the loops, so its counters and caches persist. A batch size $B$ of 32 to 512 is typical, 128 for MNIST on a CPU.

| Initialisation | Weight standard deviation | Use with |
|---|---|---|
| fixed scale (Posts 04 to 32) | 0.01 | shallow networks only |
| Glorot (Xavier) | $\sqrt{2 / (n_\text{in} + n_\text{out})}$ | tanh, sigmoid, linear |
| He | $\sqrt{2 / n_\text{in}}$ | ReLU |

Each layer multiplies the activation variance by about $n_\text{in} \cdot \text{Var}(W)$; Glorot and He choose $\text{Var}(W)$ so that the factor stays near 1. Biases start at zero.

| Task | Output layer | Loss | Gradient into the logits |
|---|---|---|---|
| $K$ classes | softmax | categorical cross-entropy | $(\hat{\mathbf{y}} - \mathbf{y})/N$ |
| two classes | sigmoid, one unit | binary cross-entropy | $(\hat{y} - y)/N$; labels $(N,)$, reshaped to $(N, 1)$ inside the class |

## When training goes wrong

1. Print the shape of every array once; most bugs are a shape that broadcasts silently, such as an $(N,)$ vector where $(N, 1)$ was meant, or a reduction without `keepdims=True`.
2. Check the first loss: before training it should be close to $\ln K$.
3. Gradient-check the backward pass on a tiny network before training a large one.
4. If the loss rises or oscillates, lower the learning rate; if it barely moves, raise it or check the sign of the update.
5. Check that dropout is off and no update runs during evaluation.
6. Shrink the problem: one layer, a handful of samples, two features. A correct network can drive the training loss of a few samples close to zero.
