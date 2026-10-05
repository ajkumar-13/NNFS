# Notation guide

The symbols, shapes, and code names used across Neural Networks from Scratch. Where a post's maths and its code name the same quantity differently, the maths uses the symbol below and the code uses the identifier in the right-hand column. Bold capitals are batch matrices, bold lowercase letters are vectors, and italic letters are scalars, including single elements such as $z$, $w$, $b$, and $a$ for one neuron.

## Forward pass

| Symbol | Meaning | Shape | Code |
|---|---|---|---|
| $\mathbf{X}$ | input batch, one sample per row | $(N, n_\text{inputs})$ | `inputs`, `X` |
| $\mathbf{W}$ | weights of a dense layer | $(n_\text{inputs}, n_\text{neurons})$ | `weights` |
| $\mathbf{b}$ | biases of a dense layer | $(1, n_\text{neurons})$ | `biases` |
| $\mathbf{Z}$ | weighted sums (pre-activation) | $(N, n_\text{neurons})$ | the dense layer's `output` |
| $\mathbf{A}$ | activation output | $(N, n_\text{neurons})$ | the activation's `output` |
| $\hat{\mathbf{y}}$ | predicted probabilities | $(N, K)$ | `y_pred`, the softmax `output` |
| $\mathbf{y}$ | true labels | $(N,)$ class indices or $(N, K)$ one-hot | `y_true`, `y` |
| $L$ | loss, the mean over the batch | scalar | `loss` |
| $N$ | batch size (number of samples) | scalar | `samples`, `len(X)` |
| $K$ | number of classes | scalar | `classes` |

A dense layer computes

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

and an activation follows, $\mathbf{A} = f(\mathbf{Z})$. The bias row is broadcast to every row of $\mathbf{X}\mathbf{W}$.

**The weight convention changes once.** Posts 01 to 03 store one row of weights per neuron, shape $(n_\text{neurons}, n_\text{inputs})$, and compute `np.dot(inputs, weights.T)`. From Post 04 on, `Layer_Dense` stores $(n_\text{inputs}, n_\text{neurons})$ so that the forward pass is `np.dot(inputs, self.weights)` with no transpose. Both are correct; a formula that looks transposed against an earlier post is usually this change. Post 14, section 8, reconciles the two.

## Backward pass

The gradient of the loss with respect to a quantity is written $\partial L / \partial (\cdot)$ in maths and with a leading `d` in code. Every gradient has the shape of the quantity it differentiates.

| Maths | Meaning | Code |
|---|---|---|
| $\partial L / \partial \mathbf{Z}$ | gradient arriving at a dense layer from the component after it | `dvalues` |
| $\partial L / \partial \mathbf{W}$ | weight gradient | `dweights` |
| $\partial L / \partial \mathbf{b}$ | bias gradient | `dbiases` |
| $\partial L / \partial \mathbf{X}$ | gradient passed back to the previous component | `dinputs` |

`dvalues` is the gradient coming into a component; `dinputs` is the gradient going out of it, and it becomes the previous component's `dvalues`. For a dense layer:

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Z}}, \qquad \frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{rows}} \frac{\partial L}{\partial \mathbf{Z}}, \qquad \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \, \mathbf{W}^\top$$

In code these are `np.dot(self.inputs.T, dvalues)`, `np.sum(dvalues, axis=0, keepdims=True)`, and `np.dot(dvalues, self.weights.T)`.

ReLU passes the gradient where its input was positive and zeroes it elsewhere, because

$$\frac{d \, \text{ReLU}(z)}{dz} = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$

The combined softmax and categorical cross-entropy backward, with $\mathbf{y}$ one-hot and $L$ the batch mean, is

$$\frac{\partial L}{\partial \mathbf{Z}} = \frac{\hat{\mathbf{y}} - \mathbf{y}}{N}$$

and sigmoid with binary cross-entropy gives the same form, $(\hat{y} - y)/N$, for each output.

## Shapes

The matrix product rule is $(m, n) \cdot (n, p) \rightarrow (m, p)$: the inner sizes must match and disappear.

| Product | Where | Result |
|---|---|---|
| $(N, n_\text{inputs}) \cdot (n_\text{inputs}, n_\text{neurons})$ | forward, `np.dot(X, W)` | $(N, n_\text{neurons})$ |
| $(n_\text{inputs}, N) \cdot (N, n_\text{neurons})$ | weight gradient, `np.dot(X.T, dvalues)` | $(n_\text{inputs}, n_\text{neurons})$ |
| $(N, n_\text{neurons}) \cdot (n_\text{neurons}, n_\text{inputs})$ | input gradient, `np.dot(dvalues, W.T)` | $(N, n_\text{inputs})$ |

Biases are stored as $(1, n_\text{neurons})$ so that $(N, n_\text{neurons}) + (1, n_\text{neurons}) \rightarrow (N, n_\text{neurons})$ broadcasts across the batch.

`axis` names the axis that disappears in a reduction. For a $(3, 4)$ array, `np.sum(a, axis=0)` has shape $(4,)$, `np.sum(a, axis=1)` has shape $(3,)$, and `keepdims=True` keeps the reduced axis as size 1: $(1, 4)$ and $(3, 1)$.

Labels come in two forms: class indices such as `[0, 2, 1]`, shape $(N,)$, or one-hot rows such as `[[1, 0, 0], [0, 0, 1], [0, 1, 0]]`, shape $(N, K)$.

## Optimisers

| Symbol | Meaning | Code |
|---|---|---|
| $\theta$ | any parameter (a weight or a bias) | `layer.weights`, `layer.biases` |
| $g$ | its gradient, $\partial L / \partial \theta$ | `layer.dweights`, `layer.dbiases` |
| $t$ | update counter | `iterations` |
| $\alpha$ | learning rate; $\alpha_0$ for the initial value under decay | `learning_rate`, `current_learning_rate` |
| $d$ | learning-rate decay | `decay` |
| $\beta$ | momentum coefficient | `momentum` |
| $v$ | momentum velocity | `weight_momentums`, `bias_momentums` |
| $G$ | cache of squared gradients (AdaGrad, RMSProp) | `weight_cache`, `bias_cache` |
| $\rho$ | RMSProp cache decay | `rho` |
| $m$, $v$ | Adam's first and second moment estimates | `weight_momentums`, `weight_cache` (and the bias versions) |
| $\beta_1$, $\beta_2$ | Adam's decay rates for $m$ and $v$ | `beta_1`, `beta_2` |
| $\hat{m}$, $\hat{v}$ | bias-corrected moments | `weight_m_hat`, `weight_v_hat` (and the bias versions) |
| $\epsilon$ | small constant that keeps a denominator above zero | `epsilon` |

In Adam, $v$ is the second moment, not the momentum velocity of Post 24; the code keeps the two apart by name.

## Regularisation, initialisation, and outputs

| Symbol | Meaning | Code |
|---|---|---|
| $\lambda$ | regularisation strength | `weight_regularizer_l1`, `weight_regularizer_l2`, `bias_regularizer_l1`, `bias_regularizer_l2` |
| $p$ | dropout rate, the probability of dropping a neuron | the `rate` argument of `Layer_Dropout`; the class stores the keep rate `1 - rate` |
| $n_\text{in}$, $n_\text{out}$ | fan-in and fan-out of a layer | `n_inputs`, `n_neurons` |
| $\sigma(z)$ | sigmoid, $1/(1 + e^{-z})$ | `Activation_Sigmoid`; `Activation_Sigmoid_Loss_BinaryCrossentropy` pairs it with the loss |

## Conventions

- Inline maths is written `$...$` and display maths `$$...$$` on its own line.
- Prose uses British spelling (optimiser, regularisation, initialisation); code uses the American spelling of its class and argument names (`Optimizer_Adam`, `weight_regularizer_l2`), as the reference implementation does.
- Indices start at 0, as in Python. Axis 0 is the batch axis.
