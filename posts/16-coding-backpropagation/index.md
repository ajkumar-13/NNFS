# 16 - Coding backpropagation

> **TL;DR.** The three gradient formulas of posts 13 to 15 become a `backward` method on `Layer_Dense`: one NumPy line each for `dweights`, `dbiases`, and `dinputs`, fed by the inputs that `forward` now caches. `Activation_ReLU` gets a `backward` that is a **masked copy**: the upstream gradient, with a zero wherever the cached input was not positive. On the three-sample batch of post 14 the class returns that post's weight gradient with rows and columns exchanged, and a central-difference check of a Dense, ReLU, Dense chain built from the two classes agrees with all five of its gradients to a relative error of $1.2 \times 10^{-10}$ or less on the seed shown, while a ReLU without its mask fails the same check by more than 1 on each of ten seeds.
>
> **Prerequisites:** [Post 14](../14-matrices-in-backpropagation/index.md), [Post 15](../15-gradients-with-respect-to-inputs/index.md).
> **Safe to skip?** Skip it if the reader can already write both `backward` methods from memory, say which array each `forward` must keep for them, and explain why the ReLU version starts from `dvalues.copy()`.
>
> **After reading, you will be able to:**
>
> - Implement Layer_Dense.backward(dvalues) in three NumPy lines and explain what each line computes.
> - Implement Activation_ReLU.backward(dvalues) as a masked copy of the upstream gradient.
> - Name what each layer caches in forward and say why its backward method needs it.

![A card with the forward and backward methods of Layer_Dense, for a batch of N samples, n inputs and m neurons. forward stores self.inputs, tagged as cached, and computes self.output; backward takes dvalues and computes self.dweights, self.dbiases and self.dinputs. Grey arrows run left to right: inputs of shape (N, n) in, output of shape (N, m) out. Purple arrows run right to left: dvalues of shape (N, m) in from the next layer, dinputs of shape (N, n) out to the previous layer as its dvalues. dweights, shape (n, m), and dbiases, shape (1, m), stay on the layer for the optimiser.](diagrams/01-dense-backward-class.svg)

*One forward call, one backward call, three gradients out: `dweights` and `dbiases` stay on the layer, and `dinputs` goes back to the layer before.*

---

## 1. The question: what must a layer offer so that gradients can flow through it?

[Post 04](../04-dense-layer-class-and-spiral-data/index.md) introduced `Layer_Dense` with only a `forward` method. Posts 13 to 15 then derived, on loose arrays, the three gradients a dense layer needs. This post puts them where the forward pass already lives: in the class, as a mirror method named `backward`. It accepts the gradient of the loss with respect to the layer's output, named `dvalues` throughout the series, and it leaves three arrays on the layer.

| Gradient stored on the layer | Meaning | Read by |
|---|---|---|
| `self.dweights` | $\partial L / \partial \mathbf{W}$ | the optimiser, to update the weights |
| `self.dbiases` | $\partial L / \partial \mathbf{b}$ | the optimiser, to update the biases |
| `self.dinputs` | $\partial L / \partial \mathbf{X}$ | the previous component, as *its* `dvalues` |

This is the whole contract: a component that offers `forward`, `backward`, and `dinputs` can sit anywhere in a stack, and a component with parameters adds one gradient per parameter array. `dvalues` always has the shape of the component's `output`, because it is the gradient with respect to that output; an array of any other shape is a bug in whatever produced it. The figure at the top of the post draws this contract for `Layer_Dense`, with the shape of every array that enters or leaves it.

The three matrix expressions were derived in posts 14 and 15, and the notation guide lists them in this layout:

| Gradient | Formula | NumPy |
|---|:---:|---|
| `dweights` | $\mathbf{X}^{\top} \, (\partial L / \partial \mathbf{Z})$ | `np.dot(self.inputs.T, dvalues)` |
| `dbiases` | $\sum_{\text{rows}} \partial L / \partial \mathbf{Z}$ | `np.sum(dvalues, axis=0, keepdims=True)` |
| `dinputs` | $(\partial L / \partial \mathbf{Z}) \, \mathbf{W}^{\top}$ | `np.dot(dvalues, self.weights.T)` |

In words: `dweights` pairs each input with each neuron's upstream gradient and sums over the batch; `dbiases` sums the upstream gradient over the batch, because a bias enters every sample's pre-activation with the constant factor 1; `dinputs` sends each sample's upstream row back through the weights. The method needs these three lines and one more in `forward`, which keeps the inputs.

---

## 2. `Layer_Dense` with a backward method

```python
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
```

The constructor and the computation in `forward` are those of post 04. Three things are new.

**`forward` caches `self.inputs`.** The weight gradient multiplies the upstream gradient by the inputs the layer saw, and by the time `backward` runs, `forward` has long returned. Without the cached array the first line of `backward` has nothing to multiply. The assignment stores a reference to an array that already exists, so it copies nothing.

**The bias gradient uses `keepdims=True`.** The sum over axis 0 then keeps the shape `(1, n_neurons)` of `self.biases` instead of `(n_neurons,)` (post 05). The updated biases are the same either way (section 8); the flag is kept for the rule that **a gradient has the shape of the array it updates**.

**`self.dinputs` is the gradient with respect to the layer's inputs, not its weights.** The layer itself never uses it. The component before it reads it as its own `dvalues`, and that hand-off is what carries the gradient through a stack of layers.

None of the three lines accumulates. Each call to `backward` assigns fresh arrays, so the gradients of one batch never mix with those of the next and there is nothing to reset between iterations.

### 2.1. The weight layout and where the transposes come from

The class stores one column of weights per neuron, shape `(n_inputs, n_neurons)`, the layout of post 04, and the transposes in `backward` belong to that layout: $(n, N) \cdot (N, m) \rightarrow (n, m)$ for the weight gradient and $(N, m) \cdot (m, n) \rightarrow (N, n)$ for the input gradient. Posts 13 and 14 derived the weight gradient with one *row* per neuron, where it reads `dL_dZ.T @ X` and has shape $(m, n)$. The two results are transposes of each other, as the two stored weight arrays are (post 14, section 8).

---

## 3. A numerical check against posts 13 to 15

`snippets/backward_classes.py` holds the two classes of this post and runs them on numbers the reader has already seen. The first part takes the three-sample batch of post 14, section 6.1, and that post's made-up upstream gradient, in which every neuron receives 1 from the first sample, 2 from the second, and 3 from the third. The weights are those of the layer of post 13, stored one column per neuron.

```python
X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])        # (3, 4): three samples, four inputs

dvalues = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])           # (3, 3): one row per sample, one column per neuron

layer = Layer_Dense(4, 3)
layer.weights = np.array([[0.1, 0.5, 0.9],
                          [0.2, 0.6, 1.0],
                          [0.3, 0.7, 1.1],
                          [0.4, 0.8, 1.2]])     # (4, 3): the layer of post 13, one column per neuron
layer.biases = np.array([[0.1, 0.2, 0.3]])

layer.forward(X)
layer.backward(dvalues)
```

```text
== Section 3: the batch of post 14 through Layer_Dense.backward
dweights (4, 3)
[[ 0.5  0.5  0.5]
 [20.1 20.1 20.1]
 [10.9 10.9 10.9]
 [ 4.1  4.1  4.1]]
dbiases (1, 3)
[[6. 6. 6.]]
dinputs (3, 4)
[[1.5 1.8 2.1 2.4]
 [3.  3.6 4.2 4.8]
 [4.5 5.4 6.3 7.2]]
dweights is the transpose of post 14's batch result: True
dinputs rows are 1, 2 and 3 times post 15's [1.5 1.8 2.1 2.4]: True
dweights has the shape of weights: True
dbiases has the shape of biases: True
dinputs has the shape of inputs: True
```

Every number can be followed by hand. The first entry of `dweights` pairs the first input of each sample with its upstream value: $1.0 \cdot 1 + 2.0 \cdot 2 + (-1.5) \cdot 3 = 0.5$. Post 14 printed the same twelve numbers as three rows of $[0.5, 20.1, 10.9, 4.1]$, one row per neuron; here they stand in columns, one column per neuron. Each bias gradient is $1 + 2 + 3 = 6$. The first row of `dinputs` is the sum of each weight row, $0.1 + 0.5 + 0.9 = 1.5$ and so on, which is the vector $[1.5, 1.8, 2.1, 2.4]$ of post 15, and the second and third rows are 2 and 3 times it, because their upstream rows are.

Agreement with earlier posts shows the class is a faithful copy of the formulas. It does not test the formulas a second time, since both sides rest on the same derivation. Section 7 makes the independent test.

---

## 4. ReLU with a backward method

ReLU's local derivative is 1 where its input was positive and 0 elsewhere, including the corner $z = 0$, where the series takes 0 (post 10). For one element, with $a = \max(0, z)$:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \mathbb{1}[z > 0].$$

The indicator $\mathbb{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. The gradient either passes unchanged or is replaced by zero; nothing is scaled. In code:

```python
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()                           # the caller's array stays intact
        self.dinputs[self.inputs <= 0] = 0                      # closed gates pass nothing back
```

**`forward` caches `self.inputs`** for the same reason `Layer_Dense` does: `backward` must know which entries were positive, and the upstream gradient does not say. For ReLU the cached output would serve as well, since the output is positive exactly where the input is. Post 17 meets activations whose derivative is most easily written from the output.

**`backward` is a masked copy.** It starts from a copy of `dvalues` and writes 0 at every position where the cached input was not positive. The class has no weights and no biases, so it stores no `dweights` or `dbiases`: its only gradient is `dinputs`.

The script runs the method on three elements:

```python
relu = Activation_ReLU()
relu.forward(np.array([[1.0, -2.0, 3.0]]))
upstream = np.array([[5.0, 6.0, 7.0]])
relu.backward(upstream)
```

```text
== Section 4: the masked copy
inputs   [[ 1. -2.  3.]]
dvalues  [[5. 6. 7.]]
dinputs  [[5. 0. 7.]]
dinputs is a separate array: True
```

The middle input was $-2$, so its gate is closed and the 6 that arrived there is dropped. The other two inputs were positive and their gradients pass through unchanged.

### 4.1. Why a copy and not an assignment

![At the top, self.inputs reads 1, minus 2, 3, with the gate at minus 2 closed, and the caller hands in dvalues 5, 6, 7. Two panels run the two lines of backward. In the left panel, headed Without .copy(), dvalues and self.dinputs are two names for one array, which reads 5, 0, 7 after the masking line; its 0 is outlined in red, because the caller's 6 is now 0. In the right panel, headed With .copy(), they are two arrays: dvalues still reads 5, 6, 7, labelled unchanged, and self.dinputs reads 5, 0, 7, labelled masked.](diagrams/02-copy-not-alias.svg)

*Both versions leave the same numbers in `dinputs`. Only the right-hand one leaves the array it was handed as it found it.*

`self.dinputs = dvalues`, without `.copy()`, does not create an array. It gives a second name to the caller's array, and the masking line then writes zeros into the array the caller still holds, as the left panel of the figure shows. In the chain of section 6 that array is `dense2.dinputs`, and anything that reads it later would read a gradient the layer never computed. The copy costs one allocation and buys a simple guarantee: `backward` reads its argument and never writes to it. Section 8 runs both versions.

---

## 5. Caching: what each class stores

Every `backward` in the series follows the same rule: `forward` keeps what `backward` will need, and nothing else.

| Class | Cached in `forward` | Why `backward` needs it |
|---|---|---|
| `Layer_Dense` | `self.inputs` | the weight gradient is $\mathbf{X}^{\top}$ times the upstream gradient |
| `Activation_ReLU` | `self.inputs` | the mask is zero where the input was not positive |

`Layer_Dense` needs one more array in `backward`, its own weights, for the input gradient. Those are already stored on the layer, so only the inputs have to be added. The bias gradient needs neither: it is a sum of the upstream gradient alone.

The later components keep to the rule. The softmax of post 19 builds its derivative from its own output, which `forward` has already stored, and the loss of post 18 is handed the predictions and the labels as arguments of its `backward`, so it caches nothing.

A layer holds only the inputs of its most recent `forward` call, so every `backward` belongs to the `forward` immediately before it; section 8 shows what happens when another batch slips in between.

---

## 6. The backward chain, previewed

With a `backward` on both classes, a stack can be walked in both directions. The softmax and the cross-entropy loss do not have their backward methods yet (posts 18 and 19), so the script stands in for them with the loss of posts 14 and 15: each sample's outputs are summed to $\hat{y}_i$ (the column `Y` in the code), and $L$ is the mean of $\hat{y}_i^2$ over the batch. Its gradient with respect to the last layer's output is $2 \hat{y}_i / N$ in every column of row $i$.

```python
dense1 = Layer_Dense(4, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 2)

# Forward pass, left to right.
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

# A stand-in for the softmax and loss of posts 18 and 19: L = mean over the batch of
# (the sum of each sample's outputs) squared, the loss of posts 14 and 15.
Y = np.sum(dense2.output, axis=1, keepdims=True)                # (3, 1)
loss = np.mean(Y ** 2)
dloss = 2 * Y / len(X) * np.ones_like(dense2.output)            # (3, 2): dL/d(dense2.output)

# Backward pass, right to left: each dinputs is the next call's dvalues.
dense2.backward(dloss)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
```

The backward pass calls the same three objects in the opposite order, and every call is handed the `dinputs` of the call before it. The script prints what each object holds afterwards:

```text
== Section 6: a Dense, ReLU, Dense chain, forward and then backward
loss 1.029e-07
array                 shape     gradient              shape
dense1.weights        (4, 3)    dense1.dweights       (4, 3)
dense1.biases         (1, 3)    dense1.dbiases        (1, 3)
dense2.weights        (3, 2)    dense2.dweights       (3, 2)
dense2.biases         (1, 2)    dense2.dbiases        (1, 2)
dloss                 (3, 2)    -> dense2.backward
dense2.dinputs        (3, 3)    -> activation1.backward
activation1.dinputs   (3, 3)    -> dense1.backward
dense1.dinputs        (3, 4)    the gradient at the data; nothing reads it
closed ReLU gates: 2 of 9; zeros in activation1.dinputs: 2
```

All four parameter gradients have the shapes of their parameters. The array that travels changes width as it goes: `dense2` receives one column per neuron of its own and returns one column per input of its own, which is one column per neuron of `dense1`. Two of the nine ReLU gates were closed on this batch, and `activation1.dinputs` has zeros in exactly those two places. The loss is tiny because the weights start at a scale of 0.01 (post 04); the shapes do not depend on that.

Post 20 assembles the same chain with the real softmax and loss at its right-hand end, and post 21 runs it on the spiral data.

The arrangement is a choice. A `backward` could return its gradients instead of storing them, or take the cached arrays as arguments. The series stores them on `self`, as `forward` stores `self.output`, so that the optimiser and the next component find them under fixed names.

---

## 7. Make it run: a gradient check on the classes

Three scripts under `snippets/` produce every number in this post. Each needs only NumPy, sets `np.random.seed(0)`, and runs from the series root in under a second:

- `python posts/16-coding-backpropagation/snippets/backward_classes.py` runs sections 3, 4, and 6.
- `python posts/16-coding-backpropagation/snippets/gradient_check.py` runs this section.
- `python posts/16-coding-backpropagation/snippets/what_can_go_wrong.py` runs section 8.

Section 3 compared the class with the formulas it was copied from. A **gradient check** is independent of them: it measures each gradient with the central difference of post 10, which uses only the forward pass. One entry of an array is moved by $+h$ and by $-h$, the loss is recomputed both times, and the slope between the two is the measured partial derivative.

```python
def numerical_gradient(loss_fn, array, h=1e-5):
    """Central difference on every entry of array, which is changed in place and restored."""
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        saved = array[index]
        array[index] = saved + h
        plus = loss_fn()
        array[index] = saved - h
        minus = loss_fn()
        array[index] = saved
        grad[index] = (plus - minus) / (2 * h)
    return grad
```

`loss_fn` takes no arguments and reruns the whole forward pass, so the function can be pointed at any array the forward pass reads: `dense1.weights`, `dense2.biases`, or the data `X`. The step is $h = 10^{-5}$ and the arrays are float64, as post 10 recommends. The measured and the analytic gradient are compared entry by entry through the relative error $|a - n| / \max(|a|, |n|)$, with an entry where both are exactly 0 counted as agreeing, and the largest value over the array is reported. As in post 14, a largest relative error below $10^{-7}$ counts as a pass.

The script builds the chain of section 6 on five random samples. It replaces the 0.01-scale weights and the zero biases by standard-normal draws, so that the gradients are not all tiny, as they were in section 6, and the biases take part.

```text
== Activation_ReLU as written, seed 0
loss 0.6880; closed ReLU gates 13 of 15; smallest |Z1| 0.091
dense2.dweights  (3, 2)  largest relative error 7.2e-12  pass
dense2.dbiases   (1, 2)  largest relative error 6.5e-12  pass
dense1.dweights  (4, 3)  largest relative error 4.0e-11  pass
dense1.dbiases   (1, 3)  largest relative error 2.4e-12  pass
dense1.dinputs   (5, 4)  largest relative error 1.2e-10  pass
loss evaluations for the numerical side: 86; backward calls: 3
```

All five gradients agree with the measurement to about ten significant digits. `dense1.dweights` is the demanding line: it is correct only if `dense2` returned the right `dinputs`, the ReLU masked it at the right places, and `dense1` multiplied it by the inputs it had cached. The smallest pre-activation of the first layer is 0.091 in size, far more than $h$, so no step crosses a ReLU corner, where a finite difference and the convention of section 4 would disagree (post 10). The numerical side costs 86 forward passes, two for each of the 43 numbers checked, against three `backward` calls for the analytic side. Seed 0 is a lopsided test, though: 13 of its 15 gates are closed, so the mask is exercised at thirteen places and the open path at only two, and an entry that is exactly zero on both sides agrees without testing any arithmetic. That is one reason for the ten seeds below.

The check is also run on a ReLU whose `backward` returns `dvalues.copy()` and forgets the mask. Every shape is still right and no error is raised, but `dense1.dweights`, `dense1.dbiases`, and `dense1.dinputs` now fail with relative errors of 1.8, 1.0, and 1.3: they are wrong by as much as their own size. The two gradients of `dense2` still pass, because they are computed before the ReLU's `backward` is called. A failed check therefore also narrows the search: the bug lies after the last gradient that passes and before the first that fails, here in the `dinputs` line of `dense2` or in the ReLU.

One seed is one network. The script repeats both checks for seeds 0 to 9:

```text
== Seeds 0 to 9: largest relative error over the five gradients, and the entry it occurs at
seed   without the mask   as written   at an entry of     analytic value   absolute gap
   0            1.8e+00      1.2e-10   dense1.dinputs           1.67e-02        2.0e-12
   1            1.8e+00      5.3e-09   dense1.dinputs           7.26e-04        3.9e-12
   2            1.6e+00      4.8e-06   dense1.dinputs          -2.58e-06        1.2e-11
   3            1.7e+00      5.9e-10   dense1.dweights         -2.94e-01        1.7e-10
   4            1.7e+00      4.1e-08   dense1.dinputs          -1.58e-03        6.5e-11
   5            1.1e+00      1.8e-09   dense1.dinputs          -1.98e-02        3.5e-11
   6            1.1e+00      7.3e-09   dense1.dweights         -5.19e-03        3.8e-11
   7            1.4e+00      5.0e-10   dense1.dinputs          -4.37e-02        2.2e-11
   8            1.6e+00      1.2e-08   dense1.dinputs          -7.48e-03        9.2e-11
   9            1.9e+00      1.7e-09   dense1.dweights          4.36e-03        7.3e-12
```

![A chart of the largest relative error over the five gradients, on a log scale from 10 to the minus 12 up to 10, for seeds 0 to 9. Red triangles, the ReLU backward without the mask, lie between 1.1 and 1.9 on every seed. Blue circles, the classes as written, lie in the shaded region below the pass line at 10 to the minus 7 on nine seeds, from 1.2 times 10 to the minus 10 upwards; seed 2 sits above it at 4.8 times 10 to the minus 6, at an entry of minus 2.58 times 10 to the minus 6 that is off by only 1.2 times 10 to the minus 11.](diagrams/03-gradient-check.svg)

*The two error columns of the table on one log scale: the broken ReLU sits near 1 on every seed, the classes as written below the pass line on nine of ten.*

The figure plots the two error columns of the table. The broken ReLU fails on all ten seeds, with errors between 1.1 and 1.9. The correct classes stay below $10^{-7}$ on nine seeds and reach $4.8 \times 10^{-6}$ on seed 2, where the code is no different. The last three columns explain it. The entry concerned is a gradient of $-2.58 \times 10^{-6}$, and the measurement differs from it by $1.2 \times 10^{-11}$, an absolute gap of the same order as on the other seeds. Dividing an ordinary rounding error by a very small gradient gives a large relative error. This is the caveat the glossary attaches to the $10^{-7}$ threshold: it holds for gradients that are not themselves tiny. The absolute gap separates the two cases: a relative error of $10^{-6}$ on one entry near zero, with a gap no larger than on the seeds that pass, is a rounding effect, and relative errors near 1 are a bug. Post 21 returns to the point for a whole network.

---

## 8. What can go wrong?

`snippets/what_can_go_wrong.py` runs five mistakes on the two classes and prints what Python and NumPy do with each. Four of the five raise no error.

**`backward` before `forward`.** The cache does not exist until `forward` has run once, and the first line of `backward` fails on it:

```text
== 1. backward before forward
AttributeError: 'Layer_Dense' object has no attribute 'inputs'
```

This is the one mistake of the five that announces itself. Leaving the line `self.inputs = inputs` out of `forward` gives the same message.

**An alias instead of a copy.** The two ReLU versions of section 4.1, on the three-element example:

```text
== 2. alias or copy in the ReLU backward
Activation_ReLU  dinputs [[5. 0. 7.]]   caller's array afterwards [[5. 6. 7.]]   same array: False
ReLU_Alias       dinputs [[5. 0. 7.]]   caller's array afterwards [[5. 0. 7.]]   same array: True
```

Both return the right `dinputs`, so a test of the ReLU alone passes either way. Only the caller's array shows the difference.

**The wrong reduction for the biases.** Four ways to sum a $(2, 3)$ upstream gradient with rows $[1, 2, 3]$ and $[4, 5, 6]$, each followed by the update `biases - 0.1 * dbiases` on the biases $[0.1, 0.2, 0.3]$:

```text
== 3. four ways to write the bias sum
np.sum(d, axis=0, keepdims=True)   shape (1, 3)  biases - 0.1 * dbiases has shape (1, 3)  [[-0.4, -0.5, -0.6]]
np.sum(d, axis=0)                  shape (3,)    biases - 0.1 * dbiases has shape (1, 3)  [[-0.4, -0.5, -0.6]]
np.sum(d)                          shape ()      biases - 0.1 * dbiases has shape (1, 3)  [[-2.0, -1.9, -1.8]]
np.sum(d, axis=1, keepdims=True)   shape (2, 1)  biases - 0.1 * dbiases has shape (2, 3)  [[-0.5, -0.4, -0.3], [-1.4, -1.3, -1.2]]
```

The first line is the class. The second, without `keepdims`, gives the same biases, because a `(3,)` array broadcasts as a row; only the shape of the gradient differs from the shape of its parameter. The third forgets the axis: the sum is the single number 21, broadcasting hands it to every bias, and the biases keep their shape while each is moved by the total over every neuron and every sample. The fourth sums over the neurons instead of the samples and turns the bias row into a $(2, 3)$ array, one row per sample, which raises an error only at a later forward pass on a batch of three or more samples. Comparing `dbiases.shape` with `biases.shape` after the first backward pass catches all three.

**A second `forward` before `backward`.** The layer of section 3 is run forward on its batch, then forward on a second batch of the same shape, and then backward with the gradient meant for the first:

```text
== 4. a second forward call overwrites the cache
dweights, first column, from the cached training batch:         [ 0.5 20.1 10.9  4.1]
dweights, first column, after another batch went through forward: [11.  37.4 20.6 21.4]
same shape: True   same numbers: False
```

The second result is a perfectly formed weight gradient for a pairing of inputs and upstream gradient that never occurred. It happens in practice when a validation batch is evaluated between the forward and the backward pass of a training step. Changing the input array in place between the two calls has the same effect, since the cache is a reference to that array and not a copy of it.

**An input of exactly zero.** The mask is `self.inputs <= 0`, so an input of exactly 0 passes no gradient:

```text
== 5. an input of exactly zero
inputs  [[-1.  0.  1.]]   dinputs [[0. 0. 7.]]
```

ReLU has no derivative at 0, and the series takes 0 there, so `<= 0` is the mask and `< 0` would be a different convention. A gradient check must stay away from such points, as section 7 did.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Layer interface | `forward`, `backward`, and `dinputs`; a layer with parameters also stores `dweights` and `dbiases` |
| Three-line backward | `np.dot(self.inputs.T, dvalues)`, `np.sum(dvalues, axis=0, keepdims=True)`, `np.dot(dvalues, self.weights.T)` |
| Shapes | every gradient has the shape of the array it differentiates |
| ReLU backward | a masked copy: zero wherever the cached input was not positive, unchanged elsewhere |
| Caching | `forward` keeps the inputs; `backward` belongs to the `forward` call just before it |
| Chain | each component's `dinputs` is the `dvalues` of the component before it |
| Gradient check | central differences at $h = 10^{-5}$ agree to $1.2 \times 10^{-10}$ on seed 0; a missing mask gives errors above 1 |
| Still missing | the backward of softmax and of the loss (posts 18 and 19) |

---

## Common pitfalls

1. **Leaving `self.inputs = inputs` out of `forward`.** The first backward call raises `AttributeError`, as does calling `backward` before any `forward` (section 8).
2. **Assigning `dvalues` instead of copying it.** The `dinputs` returned is right either way, so only the caller's array shows the bug (section 4.1).
3. **Summing the bias gradient over the wrong axis, or over none.** `np.sum(dvalues)` returns one number that broadcasts to every bias without complaint; `dbiases.shape` must equal `biases.shape` (section 8).
4. **Mixing the two weight layouts.** The transposes in `backward` belong to the `(n_inputs, n_neurons)` layout of the class and are wrong for the row-per-neuron layout of posts 13 and 14 (section 2.1).
5. **Running another `forward` between a `forward` and its `backward`.** The cache holds the latest inputs only, and the gradient that comes out has the right shape and the wrong numbers (section 8).
6. **Reading `dweights` as the new weights.** It is a gradient; the optimiser subtracts it, scaled by the learning rate (post 22).

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms) for the general algorithm, and section 11.5 (Debugging Strategies) for comparing a backward pass with finite differences (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (Backpropagation), the source of the class design and the names `dvalues` and `dinputs` (2020).
- Nielsen, M., *Neural Networks and Deep Learning*, chapter 2 (How the backpropagation algorithm works) (online, 2015).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 17 - Backpropagation through activation functions](../17-backpropagation-through-activation-functions/index.md):** the ReLU backward in more detail, the same one-line pattern for sigmoid and tanh, and why softmax does not fit it.
- **[Post 20 - Assembling full backpropagation](../20-assembling-full-backpropagation/index.md):** the chain of section 6 with the real softmax and loss at its end, and the `dinputs` to `dvalues` hand-off traced through every component.
