# 13 - Backpropagation through a layer of neurons

> **TL;DR.** The single-neuron recipe of post 12 does not change when three neurons share the same four inputs: the loss derivative is computed once and shared by all fifteen parameters, each neuron applies its own ReLU gate, and each weight multiplies the result by its own input. On the worked layer the shared value is $2\hat{y} = 43.2$ and every row of weight gradients is 43.2, 86.4, 129.6, and 172.8, confirmed by central differences to $6.4 \times 10^{-9}$. A 200-iteration loop takes the loss from 466.56 to about $6 \times 10^{-10}$ by switching two of the three neurons off on the way.
>
> **Prerequisites:** [Post 12](../12-backprop-through-a-single-neuron/index.md).
> **Safe to skip?** Skip it if the reader can already write down $\partial L / \partial w_{kj}$ for a layer of ReLU neurons whose outputs are summed and squared, say which factors of that product are shared by which parameters, and explain why the twelve weight gradients form an outer product.
>
> **After reading, you will be able to:**
>
> - Apply the chain rule across all fifteen parameters of a three-neuron layer with one shared upstream gradient.
> - Write the per-neuron backward pass as a Python loop and identify where it becomes a matrix multiplication.
> - Predict the gradient of any weight or bias from the input vector, the upstream gradient, and the neuron's ReLU gate.

![Four inputs 1, 2, 3 and 4 feed three ReLU neurons with pre-activations 3.1, 7.2 and 11.3, all three gates open; the activations are added to a layer sum of 21.6, which is squared to a loss of 466.56. The derivative of the loss, 2 times 21.6 = 43.2, flows back to all three neurons, and a table of weight gradients reads 43.2, 86.4, 129.6 and 172.8 in every row, beside three bias gradients of 43.2.](diagrams/01-layer-backprop.svg)

*One upstream value, three ReLU gates, twelve weight gradients: every column of the table is the upstream value times one input.*

---

## 1. The question: does the recipe change when neurons share inputs?

[Post 12](../12-backprop-through-a-single-neuron/index.md) derived the backward pass for a single neuron: a product of chain-rule factors from the loss, through the ReLU, to one weight. Each of its three weight gradients was the upstream gradient times the input attached to that weight.

A layer is several neurons reading the same inputs, and the question of this post is whether that sharing changes anything. The layer used here has **three neurons** over the same four inputs, each with its own four weights and its own bias, for 15 learnable parameters against the 4 of post 12. The three outputs are added, and the sum is squared to give a scalar loss.

The recipe does not change. The single-neuron chain rule runs once per neuron, and every run starts from the same number computed at the loss. The post writes that **same pattern, three times** as a loop and shows the line of code in which the loop becomes the matrix product of post 14.

---

## 2. The architecture

| Component | Detail |
|---|---|
| Inputs | $x_1 = 1,\ x_2 = 2,\ x_3 = 3,\ x_4 = 4$ |
| Neurons | 3, each with 4 weights and 1 bias |
| Activations | ReLU on each neuron's weighted sum |
| Layer sum | $\hat{y} = a_1 + a_2 + a_3$ |
| Loss | $L = \hat{y}^2$ (target 0) |

The forward pass for neuron $k$, with $k = 1, 2, 3$ counting neurons and $j = 1, 2, 3, 4$ counting inputs, is

$$z_k = \sum_{j=1}^{4} w_{kj} x_j + b_k, \qquad a_k = \text{ReLU}(z_k) = \max(0, z_k).$$

The layer sum and the loss are

$$\hat{y} = a_1 + a_2 + a_3, \qquad L = (\hat{y} - y)^2 = \hat{y}^2 \quad \text{with target } y = 0.$$

The target of zero is the simplification post 12 used, and $\hat{y}$ is, as there, the single number that feeds the loss: there one neuron's output, here the sum of three.

Three conventions need stating before any numbers appear.

- **Indices.** The weight $w_{kj}$ connects input $j$ to neuron $k$: neuron first, input second. The maths counts from 1 and the code from 0, so $w_{kj}$ is `weights[k-1][j-1]`; post 12 counted from 0 in both.
- **Weight layout.** The weights are stored one row per neuron, shape $(n_\text{neurons}, n_\text{inputs}) = (3, 4)$, the layout of posts 01 to 03. The `Layer_Dense` class of [post 04](../04-dense-layer-class-and-spiral-data/index.md) stores the transpose, and post 14 reconciles the two.
- **Code names.** The code calls the pre-activations `Z`, the activations `A`, the layer sum `Y`, and the loss `L`, and names a gradient by its fraction: `dL_dY` is $\partial L / \partial \hat{y}$ and `dL_dW` is the table of all $\partial L / \partial w_{kj}$.

---

## 3. The chain rule for one weight

Take $w_{11}$, the first weight of the first neuron. Changing it changes $z_1$, then $a_1$, then $\hat{y}$, then $L$, and nothing else: $w_{11}$ does not appear in $z_2$ or $z_3$. There is exactly one path from $w_{11}$ to the loss, and the chain rule of post 11 multiplies the local derivatives along it:

$$\frac{\partial L}{\partial w_{11}} = \underbrace{\frac{\partial L}{\partial \hat{y}}}_{2\hat{y}} \cdot \underbrace{\frac{\partial \hat{y}}{\partial a_1}}_{1} \cdot \underbrace{\frac{\partial a_1}{\partial z_1}}_{\mathbb{1}[z_1 > 0]} \cdot \underbrace{\frac{\partial z_1}{\partial w_{11}}}_{x_1}.$$

The four factors:

- $\partial L / \partial \hat{y} = 2\hat{y}$: the derivative of the squared-error loss, by the power rule.
- $\partial \hat{y} / \partial a_1 = 1$: the derivative of a sum with respect to one of its terms. This factor is new; a single neuron has no layer sum.
- $\partial a_1 / \partial z_1 = \mathbb{1}[z_1 > 0]$: the ReLU derivative, 1 when $z_1$ is positive and 0 otherwise. It acts as a gate that either passes the gradient or blocks it.
- $\partial z_1 / \partial w_{11} = x_1$: in the weighted sum, $w_{11}$ multiplies $x_1$ and every other term is a constant. Post 12 wrote this step as two factors, 1 for the sum and $x$ for the product.

### 3.1. Generalising to every weight in the layer

Nothing in the argument depended on the indices being 1 and 1. For neuron $k$ and input $j$:

$$\frac{\partial L}{\partial w_{kj}} = 2\hat{y} \cdot 1 \cdot \mathbb{1}[z_k > 0] \cdot x_j.$$

Three things can be read off the formula, and read in this order they are already the algorithm.

- **The upstream gradient $2\hat{y}$ is the same number in all twelve weight gradients.** Computing it once at the loss is enough. "Upstream" names the part of the product that arrives from the loss side at the point being discussed. Post 12 used the word for $\partial L / \partial z$, the gradient that arrives at the weights with the ReLU factor already in it; this post uses it for $2\hat{y}$, the gradient that arrives at the three ReLUs, and gives $\partial L / \partial z_k$ its own name below.
- **The ReLU gate depends only on the neuron index $k$, not on $j$.** The four weights of neuron $k$, and its bias, share one gate.
- **Only $x_j$ varies across the weights of a single neuron.** The input vector is the per-weight scaling factor.

The first three factors do not involve $j$, so their product is given a name of its own. It is the derivative of the loss with respect to the neuron's pre-activation:

$$\frac{\partial L}{\partial z_k} = 2\hat{y} \cdot 1 \cdot \mathbb{1}[z_k > 0], \qquad \frac{\partial L}{\partial w_{kj}} = \frac{\partial L}{\partial z_k} \cdot x_j.$$

The code stores the three of them as `dL_dZ`. For the bias of neuron $k$ the last factor is $\partial z_k / \partial b_k = 1$, because the bias is added without a multiplier:

$$\frac{\partial L}{\partial b_k} = \frac{\partial L}{\partial z_k} \cdot 1 = 2\hat{y} \cdot \mathbb{1}[z_k > 0].$$

The bias behaves like a weight whose input is the constant 1. All fifteen gradients are now written down, and only one chain was derived.

---

## 4. The forward pass, with numbers

The layer's inputs, weights, and biases, and the forward pass written one neuron at a time in plain Python:

```python
inputs = [1.0, 2.0, 3.0, 4.0]

weights = [[0.1, 0.2, 0.3, 0.4],     # neuron 1
           [0.5, 0.6, 0.7, 0.8],     # neuron 2
           [0.9, 1.0, 1.1, 1.2]]     # neuron 3

biases = [0.1, 0.2, 0.3]


def forward(weights, biases, inputs):
    """Weighted sum and ReLU for each neuron, then the layer sum Y and the loss L = Y ** 2."""
    Z = []
    for k in range(len(weights)):                # one neuron at a time
        z = biases[k]
        for j in range(len(inputs)):
            z += weights[k][j] * inputs[j]
        Z.append(z)
    A = [max(0.0, z) for z in Z]
    Y = sum(A)
    L = Y ** 2
    return Z, A, Y, L
```

By hand, for the first neuron,

$$z_1 = (0.1)(1) + (0.2)(2) + (0.3)(3) + (0.4)(4) + 0.1 = 0.1 + 0.4 + 0.9 + 1.6 + 0.1 = 3.1,$$

and in the same way $z_2 = 7.2$ and $z_3 = 11.3$. All three are positive, so ReLU passes each one unchanged, $\hat{y} = 3.1 + 7.2 + 11.3 = 21.6$, and $L = 21.6^2 = \mathbf{466.56}$.

Every ReLU gate is 1. That is a property of these particular numbers: section 5.3 closes a gate on purpose, and section 8.1 shows two of them closing during training.

---

## 5. The backward pass

With every gate equal to 1, the formula of section 3.1 collapses to

$$\frac{\partial L}{\partial w_{kj}} = 2\hat{y} \cdot x_j = 43.2 \cdot x_j.$$

The upstream gradient is $2\hat{y} = 2 \times 21.6 = 43.2$, shared by all twelve weights and all three biases, and $\partial L / \partial z_k = 43.2$ for each of the three neurons. The figure at the top of the post draws this backward pass on the layer, with the forward values and the gradients side by side.

| Weights | $x_j$ | Gradient |
|---|:---:|---:|
| $w_{11}, w_{21}, w_{31}$ | $x_1 = 1$ | $43.2$ |
| $w_{12}, w_{22}, w_{32}$ | $x_2 = 2$ | $86.4$ |
| $w_{13}, w_{23}, w_{33}$ | $x_3 = 3$ | $129.6$ |
| $w_{14}, w_{24}, w_{34}$ | $x_4 = 4$ | $172.8$ |

Each of the three bias gradients is $2\hat{y} \cdot 1 = 43.2$.

Two observations explain the structure of the table. **Every weight that multiplies the same input has the same gradient**, because here the rest of the chain is identical for the three neurons; it holds only while all the gates agree. **The gradient grows in proportion to the input**, which is why $x_4 = 4$ produces the largest gradient and $x_1 = 1$ the smallest. With inputs on very different scales, the weights attached to them receive gradients of very different sizes and learn at different rates; LeCun et al. (1998) recommend bringing the input features to a common scale so that those rates are balanced.

### 5.1. The backward pass as a loop

The formula of section 3.1 is a loop over neurons with a loop over inputs inside it. The function below takes the pre-activations and the layer sum from the forward pass and returns all fifteen gradients:

```python
def backward(Z, Y, inputs):
    """The single-neuron recipe of post 12, run once per neuron with a shared upstream value."""
    dL_dY = 2 * Y                                # computed once, at the loss
    dL_dZ = []
    dL_dW = []
    dL_db = []
    for k in range(len(Z)):                      # one neuron at a time
        gate = 1.0 if Z[k] > 0 else 0.0          # this neuron's own ReLU gate
        dL_dZ_k = dL_dY * 1.0 * gate             # loss factor, sum factor, gate
        dL_dZ.append(dL_dZ_k)
        dL_dW.append([dL_dZ_k * x for x in inputs])   # one gradient per weight: times its input
        dL_db.append(dL_dZ_k)                    # the bias multiplies a constant 1
    return dL_dZ, dL_dW, dL_db
```

`dL_dY` sits outside the loop because it is the same for every neuron, `gate` sits inside it because there is one per neuron, and the list comprehension is the inner loop over inputs. The loop needs `Z` and `inputs`, which is why a backward pass always follows a forward pass on the same data.

Run on the layer of section 4, `snippets/layer_backward.py` prints the table above, row by row:

```text
== Sections 4 and 5: forward pass, then the fifteen gradients from the loop
Z = [   3.1    7.2   11.3]   A = [   3.1    7.2   11.3]   Y = 21.6   L = 466.56
upstream dL/dY = 2Y = 43.2   gates = [1, 1, 1]   dL/dZ = [  43.2   43.2   43.2]
neuron 1   dL/dW = [  43.2   86.4  129.6  172.8]   dL/db =  43.2
neuron 2   dL/dW = [  43.2   86.4  129.6  172.8]   dL/db =  43.2
neuron 3   dL/dW = [  43.2   86.4  129.6  172.8]   dL/db =  43.2
largest gap to a central difference (h = 1e-5) over 15 parameters: 6.4e-09
```

### 5.2. Checking all fifteen against a central difference

A derivation and a loop that agree could still share a mistake, so the script also measures each gradient with the central difference of post 10, moving one parameter at a time by $\pm h$ with $h = 10^{-5}$. The last line of the output reports the result: the largest difference over the fifteen parameters is $6.4 \times 10^{-9}$, on gradients between 43.2 and 172.8, which is rounding error.

### 5.3. A neuron whose gate is closed

All three gates were open above, which hides the one factor that distinguishes neurons. The script therefore negates the four weights of neuron 2, so that $z_2 = -0.5 - 1.2 - 2.1 - 3.2 + 0.2 = -6.8$, and changes nothing else:

```text
== Section 5.3: neuron 2 switched off (its four weights negated)
Z = [   3.1   -6.8   11.3]   A = [   3.1    0.0   11.3]   Y = 14.4   L = 207.36
upstream dL/dY = 2Y = 28.8   gates = [1, 0, 1]   dL/dZ = [  28.8    0.0   28.8]
neuron 1   dL/dW = [  28.8   57.6   86.4  115.2]   dL/db =  28.8
neuron 2   dL/dW = [   0.0    0.0    0.0    0.0]   dL/db =   0.0
neuron 3   dL/dW = [  28.8   57.6   86.4  115.2]   dL/db =  28.8
largest gap to a central difference (h = 1e-5) over 15 parameters: 2.1e-09
```

The figure below sets the two printouts one above the other. Two things changed, and they are different in kind.

![Two tables of the layer's gradients, one above the other. With all three gates open, every neuron has a gradient of 43.2 at its pre-activation and weight gradients 43.2, 86.4, 129.6 and 172.8. With neuron 2 switched off, its pre-activation is minus 6.8, its gate 0 and its five gradients 0, while neurons 1 and 3 get 28.8, 57.6, 86.4 and 115.2, because the layer sum fell from 21.6 to 14.4.](diagrams/02-closed-gate.svg)

*Closing one gate zeroes that neuron's five gradients and, through the smaller layer sum, shrinks the other ten.*

- **Neuron 2's five gradients are all zero.** Its gate is 0, and the gate multiplies every one of them. A small change to any of its weights leaves $a_2$ at zero, so the neuron receives no update from this sample.
- **Neurons 1 and 3 have smaller gradients, 28.8 per unit of input instead of 43.2**, although nothing about them was touched. What changed is the upstream value: $\hat{y}$ fell from 21.6 to $3.1 + 0 + 11.3 = 14.4$, so $2\hat{y}$ fell to 28.8. The neurons are coupled through the sum that feeds the loss: no neuron's gradient contains another's weights or gate, but all contain $2\hat{y}$, which is built from all three activations. The bias gradients are now 28.8, 0, and 28.8.

Predicting any gradient in the layer therefore takes three questions: what is the upstream value $2\hat{y}$, is this neuron's $z_k$ positive, and which input does this weight multiply (1 for a bias)?

---

## 6. One gradient-descent step

The update rule of post 09 subtracts $\alpha$ times its gradient from each of the fifteen parameters at once, with learning rate $\alpha = 0.001$. The script prints neuron 1 after the step and the forward pass that follows:

```text
== Section 6: one gradient-descent step with learning rate 0.001
neuron 1 weights after the step: [0.0568 0.1136 0.1704 0.2272]   bias 0.0568
Z before = [   3.1    7.2   11.3]   Z after = [1.7608 5.8608 9.9608]
Y  21.6000 -> 17.5824
L  466.5600 -> 309.1408   ratio 0.6626
```

The loss drops by about one third, and the size of the drop can be predicted. Substituting the update into the weighted sum, as post 12 did for one neuron, shows that every open neuron's pre-activation falls by the same amount, $2\alpha\hat{y}\,(\sum_j x_j^2 + 1) = 2\alpha\hat{y} \cdot 31$, which is $1.3392$ here. With $m$ gates open, and as long as none of them closes during the step, the layer sum falls by $m$ such amounts:

$$\hat{y}^{\text{new}} = \hat{y}\,\bigl(1 - 2\alpha m \cdot 31\bigr) = \hat{y}\,(1 - 0.062\,m).$$

With $m = 3$ the factor is $0.814$, so $\hat{y}$ goes from 21.6 to $17.5824$ and the loss is multiplied by $0.814^2 = 0.6626$. Because all open neurons move down by equal amounts, the one with the smallest pre-activation will be the first to reach zero.

---

## 7. What this post adds beyond post 12

| Property | Post 12 (one neuron) | Post 13 (one layer) |
|---|:---:|:---:|
| Pre-activation | scalar `z` | vector `Z`, shape `(3,)` |
| Loss derivative | scalar $2\hat{y}$, with $\hat{y}$ the neuron's output | scalar $2\hat{y}$, with $\hat{y}$ the layer sum, shared |
| Activation derivative | one gate | vector of gates, one per neuron |
| Weight gradient | upstream scalar times the input vector | every entry of `dL_dZ` times every input (section 8.2) |
| Bias gradient | scalar | vector, equal to `dL_dZ` |

The recipe is unchanged and each quantity gains one dimension. Still missing are the batch of samples (post 14), the gradient with respect to the inputs, which reach the loss by three paths and not one (post 15), and the class that stores what the backward pass needs (post 16).

---

## 8. Make it run: the loop, 200 times

Three scripts under `snippets/` produce every number in this post. Each runs from the series root in under a second, needs only NumPy, and uses no random numbers:

- `python posts/13-backprop-through-a-layer/snippets/layer_backward.py` runs sections 4, 5, 6, and 8.2.
- `python posts/13-backprop-through-a-layer/snippets/training_loop.py` runs section 8.1.
- `python posts/13-backprop-through-a-layer/snippets/what_can_go_wrong.py` runs section 9.

### 8.1. The training loop

The training script holds the same layer as NumPy arrays, defines `relu` and `relu_deriv`, and repeats forward pass, backward pass, and update 200 times. Array operations replace the loops of section 5.1: `weights @ inputs` is the three weighted sums at once, and `dL_dZ` is a length-3 array.

```python
previous_gates = None
for i in range(200):
    # Forward.
    Z = weights @ inputs + biases
    A = relu(Z)
    Y = np.sum(A)
    L = Y ** 2

    # Backward: one upstream value, shared by all three neurons.
    dL_dY = 2 * Y                               # scalar
    dY_dA = np.ones_like(A)                     # all 1s
    dA_dZ = relu_deriv(Z)                       # one gate per neuron
    dL_dZ = dL_dY * dY_dA * dA_dZ               # shape (3,)

    # One gradient per weight: every dL_dZ entry times every input.
    dL_dW = dL_dZ.reshape(-1, 1) * inputs       # shape (3, 4)
    dL_db = dL_dZ                               # shape (3,)

    gates = dA_dZ.astype(int).tolist()
    if i % 40 == 0 or i == 199 or gates != previous_gates:
        print(f"iter {i:3d}  loss = {L:.6f}  Y = {Y:.6f}  gates = {gates}")
    previous_gates = gates

    # Update, only after every gradient has been computed.
    weights -= lr * dL_dW
    biases -= lr * dL_db
```

It prints a line every 40 iterations and at every iteration where a gate changes:

```text
iter   0  loss = 466.560000  Y = 21.600000  gates = [1, 1, 1]
iter   3  loss = 140.818219  Y = 11.866685  gates = [0, 1, 1]
iter  12  loss = 14.840513  Y = 3.852339  gates = [0, 0, 1]
iter  40  loss = 0.411915  Y = 0.641806  gates = [0, 0, 1]
iter  80  loss = 0.002461  Y = 0.049604  gates = [0, 0, 1]
iter 120  loss = 0.000015  Y = 0.003834  gates = [0, 0, 1]
iter 160  loss = 0.000000  Y = 0.000296  gates = [0, 0, 1]
iter 199  loss = 0.000000  Y = 0.000024  gates = [0, 0, 1]

loss at iteration 199, in full: 5.961e-10
final Z      : [-0.216657 -0.247661  0.000023]
```

The loss falls from 466.56 to $5.96 \times 10^{-10}$, but the gates do not stay open, as the figure below shows for the first 24 iterations. All three pre-activations fall by equal amounts, as section 6 predicted, and the smallest runs out first: at iteration 3 $z_1$ is $-0.2167$ and neuron 1's gate is closed. Neurons 2 and 3 carry on falling together until, at iteration 12, $z_2$ is $-0.2477$ and neuron 2's gate closes as well. From then on only neuron 3 learns. With $m = 1$ the formula of section 6 gives $\hat{y}^{\text{new}} = 0.938\,\hat{y}$, so $z_3 = \hat{y}$ shrinks towards zero and stays positive, and the loss approaches zero without reaching it.

![A line chart of the three pre-activations over the first 24 iterations. They start at 3.1, 7.2 and 11.3 and fall by equal amounts, staying 4.1 apart; the first crosses zero at iteration 3 and stays at minus 0.2167, the second crosses at iteration 12 and stays at minus 0.2477, and the third, the only open gate, keeps falling towards zero.](diagrams/04-gates-closing.svg)

*Open neurons fall by equal amounts per step, so the smallest pre-activation is the first to reach zero.*

Neurons 1 and 2 never reopen, because a closed gate passes no gradient to the weights that could move them: on this single input they are **dead neurons** in the sense of post 06. That outcome belongs to this example, with one input and one target. With many samples a neuron closed for one of them usually still receives gradient from others; what carries over is that the gates are recomputed at every forward pass, and a closed gate removes its neuron's parameters from that update.

### 8.2. Where the matrix product is hiding

The line that computes `dL_dW` is the structural centre of the post:

```python
dL_dW = dL_dZ.reshape(-1, 1) * inputs       # shape (3, 4)
```

![A column of three upstream gradients, each 43.2, sits left of a 3 by 4 matrix and the row of inputs 1, 2, 3 and 4 sits above it, so each cell is one entry of the column times one input. Every row reads 43.2, 86.4, 129.6 and 172.8, and the cell 172.8 is outlined with the 43.2 and the 4 that produce it.](diagrams/03-outer-product.svg)

*Every cell of the weight-gradient matrix is one entry of `dL_dZ` times one input.*

`dL_dZ` has shape `(3,)`. The reshape turns it into a column of shape `(3, 1)`, and NumPy broadcasts that column against the input vector of shape `(4,)` by the rule of post 05, producing a `(3, 4)` array whose entry in row $k$ and column $j$ is $\partial L / \partial z_k$ times $x_j$. That is the double loop of section 5.1 with both loops removed, and the figure above lays it out: row $k$ of the column meets column $j$ of the row at entry $(k, j)$.

This table of all pairwise products of two vectors is called their **outer product**. A column times a row is also an ordinary matrix product, $(3, 1) \cdot (1, 4) \rightarrow (3, 4)$ by the shape rule of post 02, and that is where the matrix multiplication of the backward pass comes from. The script confirms that the broadcast, `np.outer(dL_dZ, inputs)`, and the `(3, 1)` by `(1, 4)` product written with `@` all give the same array as the loop:

```text
== Section 8.2: the loop against its one-line NumPy forms
shape of the broadcast result: (3, 4)
loop equals broadcast: True
broadcast equals np.outer: True
broadcast equals the matrix product: True
```

A gradient always has the shape of the parameter it belongs to, here `(3, 4)`, so that the update subtracts element from matching element. Post 14 extends the product from one input vector to a whole batch and, in its section 8, rewrites it for the $(n_\text{inputs}, n_\text{neurons})$ layout of `Layer_Dense`.

---

## 9. What can go wrong?

`snippets/what_can_go_wrong.py` reproduces three mistakes on the layer of section 4.

**A learning rate that closes every gate.** With $\alpha = 0.01$ one step lowers each pre-activation by $2\alpha\hat{y} \cdot 31 = 13.392$, more than the largest of them:

```text
== 1. A learning rate of 0.01 instead of 0.001
iter 0  Z = [ 3.1  7.2 11.3]  loss = 466.56  largest |gradient| = 172.8
iter 1  Z = [-10.292  -6.192  -2.092]  loss = 0.00  largest |gradient| = 0.0
iter 2  Z = [-10.292  -6.192  -2.092]  loss = 0.00  largest |gradient| = 0.0
```

After one update every gate is closed and every gradient is exactly zero. The loss reads 0.00 only because the target is zero: the layer would output 0 for this input whatever the target was, and no further step can change that. The loss does not oscillate around the target; the ReLU clips the overshoot and the layer stops.

**Reshaping the wrong array.** Leaving out the reshape, or reshaping `inputs` instead of `dL_dZ`, fails in different ways:

```text
== 2. The reshape, done wrong
dL_dZ * inputs               -> operands could not be broadcast together with shapes (3,) (4,)
inputs.reshape(-1, 1) * dL_dZ -> shape (4, 3)  equal to the transpose of the right answer: True
weights - lr * flipped        -> operands could not be broadcast together with shapes (3,4) (4,3)
a square layer, 3 inputs and 3 neurons, raises nothing:
right, dL_dZ.reshape(-1, 1) * inputs:
[[10. 20. 30.]
 [ 0.  0.  0.]
 [30. 60. 90.]]
wrong, inputs.reshape(-1, 1) * dL_dZ:
[[10.  0. 30.]
 [20.  0. 60.]
 [30.  0. 90.]]
```

With 4 inputs and 3 neurons NumPy refuses both mistakes, the second only at the update. With as many inputs as neurons nothing is raised: the flipped product has the right shape and the wrong contents, a zero *column* where the closed gate of neuron 2 should give a zero *row*. A gradient check as in section 5.2 catches this where a shape check cannot.

**One gate for the whole layer.** Taking the ReLU derivative as a single number, for instance from the sign of the layer sum, gives a switched-off neuron the gradients of an open one:

```text
== 3. One gate for the whole layer (neuron 2 switched off)
Z = [ 3.1 -6.8 11.3]
neuron 2 row with one gate per neuron : [0. 0. 0. 0.]
neuron 2 row with one gate for all    : [ 28.8  57.6  86.4 115.2]
neuron 2 row by central difference    : [0. 0. 0. 0.]
largest gap, one gate per neuron: 2.1e-09
largest gap, one gate for all   : 115.2
```

The mistake is invisible while all three gates are 1, and in the training loop of section 8.1 that stops being true at iteration 3.

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| Same chain rule, more neurons | Backpropagation through a layer is post 12's recipe applied once per neuron |
| Shared factors | $2\hat{y}$ for all 15 parameters, one ReLU gate per neuron, one input $x_j$ per weight |
| Bias gradient | Equals $\partial L / \partial z_k$ (`dL_dZ`); its "input" is the constant 1 |
| Outer product | The weight-gradient matrix is the column `dL_dZ` times the row `inputs` |
| Check | A central difference with $h = 10^{-5}$ agrees with all fifteen gradients to $6.4 \times 10^{-9}$ |

---

## Common pitfalls

1. **Forgetting that each neuron has its own ReLU gate.** The gates are a vector with one entry per neuron, not a single number (section 9).
2. **Reshaping `inputs` instead of `dL_dZ`.** That transposes the gradient matrix, which raises an error at the update for a non-square layer and silently corrupts the update for a square one.
3. **Mixing the two weight layouts.** This post stores `weights` as $(n_\text{neurons}, n_\text{inputs})$ and `Layer_Dense` stores the transpose; a gradient must have the shape of the weights it updates.
4. **Assuming all gates are 1.** They are in the opening example. They are not three updates later (section 8.1), and in a network on real data each sample switches some of the neurons off.
5. **Updating `weights` before computing every gradient.** All fifteen gradients of one step come from the same forward pass. Compute them all, then update them all.
6. **Treating the neurons as independent.** Switching one neuron off changed the gradients of the other two from 43.2 to 28.8 per unit of input, through the shared $2\hat{y}$ (section 5.3).

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- LeCun, Y., Bottou, L., Orr, G. B., and Müller, K.-R., *"Efficient BackProp"*, in *Neural Networks: Tricks of the Trade* (Springer, 1998), for the advice to bring the inputs to a common scale so that the weights attached to them learn at comparable rates.
- Nielsen, M., *Neural Networks and Deep Learning*, chapter 2 (online, 2015).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 14 - Matrices in backpropagation](../14-matrices-in-backpropagation/index.md):** the outer product of section 8.2 written as one matrix product that handles a whole batch.
- **[Post 16 - Coding backpropagation](../16-coding-backpropagation/index.md):** the `backward` method of `Layer_Dense`, the class version of the loop written here by hand.
