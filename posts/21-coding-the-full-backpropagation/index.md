# 21 - Coding the full backpropagation

> **TL;DR.** With the classes of posts 16 to 19, one forward and one backward pass of a two-layer classifier on the 300 spiral points is a script of fifteen lines: three comments and twelve statements. It prints a first loss of 1.0986104, beside $\ln 3 = 1.0986123$, and leaves four gradient arrays, 21 numbers, each with the shape of its parameter. Shapes cannot show that the values are right, so a gradient check moves each of the 21 parameters by $\pm 10^{-5}$ in float64 and compares the measured slope with the stored gradient: over ten seeds the largest relative error is $4.5 \times 10^{-8}$. Run on the script's own 0.01 initialisation, or in the float32 that `nnfs.init()` sets, the same check fails although the code is correct.
>
> **Prerequisites:** [Post 20](../20-assembling-full-backpropagation/index.md).
> **Safe to skip?** Skip it if the reader can already write the script from memory, give the shape of each of its four gradient arrays, and run a central-difference check of a whole network that passes for the right reasons.
>
> **After reading, you will be able to:**
>
> - Read a fifteen-line script that runs a full forward and backward pass on the spiral dataset.
> - Inspect the four gradient arrays the script produces.
> - Sanity-check a gradient implementation by matching gradient shapes to parameter shapes.
> - Check the backward pass of a whole network against central differences in float64.

![The script of section 8 as a numbered listing of fifteen lines in three groups opened by the comments Network, Forward and Backward. Beside each line is what it leaves behind: dense1 and dense2 create weights (2, 3) and biases (1, 3), and weights (3, 3) and biases (1, 3), in orange; the forward calls give Z1, A1 and Z2, each (300, 3), and the loss 1.0986104 beside ln 3 = 1.0986123; the backward calls store dinputs (300, 3), then dweights (3, 3) and dbiases (1, 3) on dense2, dinputs (300, 3) with 456 zeros on the ReLU, and dweights (2, 3) and dbiases (1, 3) on dense1, in purple. A key counts 6 + 3 + 9 + 3 = 21 parameters and as many gradient entries.](diagrams/01-forward-backward-script.svg)

*The script is short because the classes do the work, and each gradient array it leaves has the shape of its parameter.*

---

## 1. The question: what does the whole pass look like as one script, and is its output right?

[Post 20](../20-assembling-full-backpropagation/index.md) wired the classes into four forward calls and four backward calls on a batch of four hand-written samples. This post runs the same calls on the spiral data, as one script, and looks at what they leave behind: four gradient arrays. It stops there. Nothing is updated and nothing is trained; the update and the training loop are post 22's.

The second half of the question is the harder one. A backward pass that is wrong still runs and still returns arrays. Three tests of rising strength are applied to the script: the first loss (section 5), the shapes of the gradients (section 7), and a **gradient check** of every one of the 21 numbers (section 9), which posts 10 and 16 announced for this post.

---

## 2. The dataset

The data is the spiral of [post 04](../04-dense-layer-class-and-spiral-data/index.md): three interleaved arms of 100 points each, 300 samples with two features, not separable by straight lines.

```python
nnfs.init()                                     # seed 0, float32 arrays, a patched np.dot

X, y = spiral_data(samples=100, classes=3)      # X: (300, 2), y: (300,)
```

```text
X (300, 2) float32   y (300,) uint8   samples per class [100 100 100]
points at the origin: 3   largest distance from it: 1.0
```

`X` holds the inputs and `y` one integer label per sample. Two facts of this data return later. The classes are balanced, 100 samples each, which decides the size of one gradient in section 7. And each arm starts at the origin, so three of the 300 points are exactly $(0, 0)$, which the check in section 11 meets again.

---

## 3. The three classes in one place

The script uses three classes: `Layer_Dense` and `Activation_ReLU` as post 16 left them, and the combined softmax and loss class of post 19. They are unchanged.

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


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                                    # cached for backward
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()                           # the caller's array stays intact
        self.dinputs[self.inputs <= 0] = 0                      # closed gates pass nothing back


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
```

The combined class uses `Activation_Softmax` and `Loss_CategoricalCrossentropy` for its forward pass; both stand above these three in `snippets/forward_backward.py`, as posts 18 and 19 left them. Its backward pass uses neither: it is the three-line shortcut of post 19.

---

## 4. Instantiating the network

```python
# Network.
dense1 = Layer_Dense(2, 3)                            # 2 inputs, 3 hidden neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)                            # 3 hidden neurons, 3 classes
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
```

Four objects, of which only the two dense layers have parameters:

| Array | Shape | Numbers |
|---|:---:|:---:|
| `dense1.weights` | `(2, 3)` | 6 |
| `dense1.biases` | `(1, 3)` | 3 |
| `dense2.weights` | `(3, 3)` | 9 |
| `dense2.biases` | `(1, 3)` | 3 |

That is $6 + 3 + 9 + 3 = 21$ parameters, the network of post 20. The weights are drawn at a scale of 0.01 and the biases start at zero.

---

## 5. The forward pass

```python
# Forward.
dense1.forward(X)                                     # (300, 2) -> (300, 3)
activation1.forward(dense1.output)                    # (300, 3), shape unchanged
dense2.forward(activation1.output)                    # (300, 3) -> (300, 3)
loss = loss_activation.forward(dense2.output, y)      # logits and labels -> one number
```

Each call reads the `output` of the call before it. With three hidden neurons and three classes, every array between `X` and the loss has shape `(300, 3)`: `dense1.output` ($\mathbf{Z}_1$), `activation1.output` ($\mathbf{A}_1$), `dense2.output` ($\mathbf{Z}_2$, the logits) and `loss_activation.output` ($\hat{\mathbf{y}}$, the predictions). The loss is one number, the mean cross-entropy of the batch.

```text
loss 1.0986104   ln 3 = 1.0986123
farthest any of the 900 probabilities is from 1/3: 6.8e-05
```

This is the first test. With weights of size 0.01 the logits are close to zero, every prediction is close to $1/3$, and the loss must be close to $\ln 3$, the loss of a uniform guess over three classes (post 08, section 8.1). The value 1.0986104 is the one post 08 printed for this network. A first loss far from $\ln 3$ would mean a broken forward pass or loss. A first loss that matches says nothing yet about the backward pass.

---

## 6. The backward pass

```python
# Backward.
loss_activation.backward(loss_activation.output, y)   # predictions and labels -> (300, 3)
dense2.backward(loss_activation.dinputs)              # stores (3, 3), (1, 3), (300, 3)
activation1.backward(dense2.dinputs)                  # stores (300, 3), masked
dense1.backward(activation1.dinputs)                  # stores (2, 3), (1, 3), (300, 2)
```

The same four objects in the opposite order. The first call is handed the predictions and the labels, and every later call the `dinputs` of the call before it; post 20, section 3, explains both rules.

```text
gradient                 shape      read by
loss_activation.dinputs  (300, 3)   dense2.backward
dense2.dinputs           (300, 3)   activation1.backward
activation1.dinputs      (300, 3)   dense1.backward
dense1.dinputs           (300, 2)   nothing
closed ReLU gates: 456 of 900; zeros in activation1.dinputs: 456
```

Each `dinputs` has the shape of the input of the object that stores it, one row per sample. The ReLU mask is visible in the last line: 456 of the 900 pre-activations in $\mathbf{Z}_1$ are not positive, and `activation1.dinputs` is zero in exactly those 456 places. Beside these four arrays, the two dense layers now hold `dweights` and `dbiases`.

---

## 7. Inspecting the gradients

```text
dense1.dweights:
[[ 1.577e-04  7.837e-05  4.732e-05]
 [ 1.816e-04  1.105e-05 -3.310e-05]]
dense1.dbiases: [[-3.606e-04  9.661e-05 -1.037e-04]]
dense2.dweights:
[[ 5.441e-05  1.074e-04 -1.618e-04]
 [-4.079e-05 -7.168e-05  1.125e-04]
 [-5.301e-05  8.582e-05 -3.281e-05]]
dense2.dbiases: [[-1.073e-05 -9.461e-06  2.003e-05]]
```

**The second test: every gradient array has the shape of its parameter.** Entry $(j, k)$ of `dense1.dweights` is the partial derivative of the loss with respect to entry $(j, k)$ of `dense1.weights`, so the two arrays must have the same layout.

```text
parameter        shape    gradient          shape    same   largest |entry|   exact zeros
dense1.weights   (2, 3)   dense1.dweights   (2, 3)   True   1.8e-04           0
dense1.biases    (1, 3)   dense1.dbiases    (1, 3)   True   3.6e-04           0
dense2.weights   (3, 3)   dense2.dweights   (3, 3)   True   1.6e-04           0
dense2.biases    (1, 3)   dense2.dbiases    (1, 3)   True   2.0e-05           0
gradient entries in all: 21
```

Twenty-one gradients for 21 parameters, shape for shape, and none of them exactly zero. A mismatch is always a bug, usually a wrong argument: handing the scalar loss to `dense2.backward` gave `dweights` the shape `(3, 4)` in post 20, section 7.

**The sizes can be explained.** A row of `loss_activation.dinputs` is $(\hat{\mathbf{y}} - \mathbf{y})/N$. With every prediction near $1/3$ and $N = 300$, it holds about $-2/900$ at the true class and $1/900$ at the other two; the script prints a largest entry of $2.22 \times 10^{-3}$. `dense2.dbiases` is the sum of that array down each column, and each class is the label of 100 samples and not the label of 200:

$$100 \cdot \left(-\frac{2}{900}\right) + 200 \cdot \frac{1}{900} = 0.$$

What survives, $2.0 \times 10^{-5}$ at most, comes from the distance of the predictions from $1/3$. Post 20, section 5, met the same cancellation on its balanced batch of six. The other three arrays each carry a factor from the small weights: `dense2.dweights` multiplies by the activations $\mathbf{A}_1$, which the 0.01 weights keep below 0.018, and both gradients of `dense1` have passed back through `dense2.weights`. Small first gradients are what a balanced data set and this initialisation produce; they are not a fault. What the scale of the initial weights does to a network is the subject of post 33.

---

## 8. The full fifteen-line script

```python
# Network.
dense1 = Layer_Dense(2, 3)                            # 2 inputs, 3 hidden neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)                            # 3 hidden neurons, 3 classes
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Forward.
dense1.forward(X)                                     # (300, 2) -> (300, 3)
activation1.forward(dense1.output)                    # (300, 3), shape unchanged
dense2.forward(activation1.output)                    # (300, 3) -> (300, 3)
loss = loss_activation.forward(dense2.output, y)      # logits and labels -> one number

# Backward.
loss_activation.backward(loss_activation.output, y)   # predictions and labels -> (300, 3)
dense2.backward(loss_activation.dinputs)              # stores (3, 3), (1, 3), (300, 3)
activation1.backward(dense2.dinputs)                  # stores (300, 3), masked
dense1.backward(activation1.dinputs)                  # stores (2, 3), (1, 3), (300, 2)
```

Fifteen lines: three comments and twelve statements, four per group. The figure at the top of the post sets each line beside what it leaves behind. After the last one the gradients are ready in `dense1.dweights`, `dense1.dbiases`, `dense2.dweights` and `dense2.dbiases`. The script does not grow in kind with the network. A further hidden layer adds one `Layer_Dense` and one `Activation_ReLU`, which is two objects, two forward calls and two backward calls, six lines (post 20, section 4).

The script is one pass over all 300 samples as a single batch. Wrapped in a loop, with an update of the four parameter arrays after the backward calls, it becomes a training run, and [post 22](../22-gradient-descent-optimiser/index.md) builds that.

---

## 9. The gradient check on the whole network

A matching shape says where a number is stored, not whether it is right. Dropping the last line of the combined class, or the mask of the ReLU, changes no shape. The **gradient check** tests the values. It needs only the forward pass: one parameter is moved by $+h$ and by $-h$, the loss is recomputed both times, and the central difference of post 10, section 2.5,

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h},$$

is compared with the entry the backward pass stored. Post 16, section 7, did this for the two classes with a stand-in loss. Here the loss is the real one, wrapped in a function that takes no arguments and reruns the four forward calls:

```python
    def loss_fn():
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        return loss_activation.forward(dense2.output, y)
```

`numerical_gradient(loss_fn, dense1.weights)` then measures all six entries of that array, and likewise for the other three. The function and the comparison are those of post 16, unchanged: $h = 10^{-5}$, the relative error $|a - n| / \max(|a|, |n|)$ between the analytic entry $a$ and the measured one $n$, the largest value over each array, and a pass below $10^{-7}$.

`snippets/gradient_check.py` differs from the script of section 8 in two ways, and both are needed.

- **It runs in float64.** It does not call `nnfs.init()`; `np.random.seed(0)` draws the same data without the conversion to float32.
- **It redraws the parameters at scale 1.** After the network is built, all four parameter arrays, biases included, are replaced by standard-normal draws, as in post 16. The classes, the data and the eight calls are those of the script. What the check then tests is the code of the backward pass, not the 21 numbers of section 7; section 11 measures those as well, as far as a finite difference can.

Section 11 shows what happens without either.

```text
== Seed 0, float64, h = 1e-05, weights and biases redrawn at scale 1
X float64, dense1.weights float64, loss float64
loss 1.9448419; closed ReLU gates 600 of 900; entries of Z1 exactly 0: 0; within h of 0: 0
gradient         shape    largest |entry|   largest relative error   largest absolute gap
dense2.dweights  (3, 3)   1.2e-01           3.5e-10                  1.4e-11   pass
dense2.dbiases   (1, 3)   4.6e-01           6.1e-11                  1.6e-11   pass
dense1.dweights  (2, 3)   5.7e-02           3.3e-10                  1.9e-11   pass
dense1.dbiases   (1, 3)   3.3e-01           1.4e-11                  4.7e-12   pass
forward passes for the numerical side: 42; backward calls: 4
```

All 21 gradients agree with the measurement to nine digits or more. The two lines of `dense1` are the demanding ones: they are right only if the combined class, `dense2.dinputs` and the ReLU mask before them are right. No entry of $\mathbf{Z}_1$ lies within $h$ of zero, so no step crosses the corner of a ReLU, where a finite difference measures neither of the two slopes (post 10).

The measurement costs two forward passes per parameter, 42 here against four backward calls. For the network of Part VI, with 64 hidden neurons and $2 \cdot 64 + 64 + 64 \cdot 3 + 3 = 387$ parameters, it would be 774 forward passes for one gradient. That is why the check is a test, run once on a small case, and backpropagation is what training uses.

One seed is one network, and seed 0 leaves 600 of its 900 gates closed. Ten seeds:

```text
seed   closed gates   smallest |Z1|   largest relative error   in                largest absolute gap
   0   600            1.3e-01         3.5e-10                  dense2.dweights   1.9e-11
   1   653            1.7e-03         3.1e-09                  dense2.dweights   2.0e-11
   2   414            3.9e-03         5.1e-10                  dense1.dweights   5.9e-11
   3   191            3.6e-04         1.0e-08                  dense1.dweights   4.5e-11
   4   58             1.4e-04         7.3e-09                  dense1.dweights   3.5e-11
   5   422            4.3e-03         5.9e-09                  dense1.dweights   3.0e-11
   6   73             1.8e-04         2.2e-09                  dense1.dweights   6.0e-11
   7   626            1.1e-03         2.3e-10                  dense2.dweights   1.3e-11
   8   600            2.6e-01         2.3e-10                  dense2.dbiases    1.2e-11
   9   531            4.1e-04         4.5e-08                  dense2.dweights   4.1e-11
```

The check passes on all ten, with between 58 and 653 gates closed. The largest relative error is $4.5 \times 10^{-8}$ and no absolute gap exceeds $6.0 \times 10^{-11}$. Ten seeds are not a guarantee: on seeds 10 to 49 the same code is above the pass mark six times, by most on seed 17 with $3.8 \times 10^{-4}$, and no absolute gap exceeds $5.6 \times 10^{-11}$, which is what post 16 found on one of its seeds. The glossary's reading of the number applies: near $10^{-7}$ or below the backward pass is right, above about $10^{-3}$ it is almost certainly wrong, and in between the absolute gap decides, as section 11 shows.

---

## 10. Make it run: the script and its check

Three scripts hold every code block and every printed number of this post. Each runs from the series root in under a second, is seeded, and needs NumPy and the `nnfs` package (`pip install nnfs`), which supplies the spiral data:

```text
python posts/21-coding-the-full-backpropagation/snippets/forward_backward.py
python posts/21-coding-the-full-backpropagation/snippets/gradient_check.py
python posts/21-coding-the-full-backpropagation/snippets/what_can_go_wrong.py
```

`forward_backward.py` is sections 2 to 8: the classes, the script, and the prints. It calls `nnfs.init()`, so its arrays are float32. `gradient_check.py` is section 9, in float64. `what_can_go_wrong.py` imports the check and runs section 11.

---

## 11. What can go wrong?

**The check is run on the script's own starting point.** The natural first attempt keeps the 0.01 weights and the zero biases. In float64, at seed 0, with code that is correct:

```text
loss 1.0986105; closed ReLU gates 456 of 900; entries of Z1 exactly 0: 9; within h of 0: 16
gradient         shape    largest |entry|   largest relative error   largest absolute gap
dense2.dweights  (3, 3)   1.6e-04           4.0e-07                  1.3e-11   FAIL
dense2.dbiases   (1, 3)   2.0e-05           6.1e-07                  6.5e-12   FAIL
dense1.dweights  (2, 3)   1.8e-04           1.4e-03                  6.8e-08   FAIL
dense1.dbiases   (1, 3)   3.6e-04           1.0e-01                  1.1e-05   FAIL
largest |Z1| 1.7e-02; samples with an entry of Z1 within h of 0: 10, of which at the origin: 3
```

All four lines fail, for two different reasons. The lines of `dense2` have absolute gaps near $10^{-11}$, the rounding level of the passing check of section 9. Their gradients are about 770 and 23,000 times smaller than there, so the same gap is a larger fraction: this is the tiny-gradient caveat of post 16. The lines of `dense1` have gaps of $6.8 \times 10^{-8}$ and $1.1 \times 10^{-5}$, far above rounding, and those are ReLU corners. The 0.01 weights keep every entry of $\mathbf{Z}_1$ below 0.018, and ten samples have an entry within $h$ of zero. A step of $h$ on a bias carries such an entry across the corner, and the measured slope is then neither the closed gate's nor the open one's. Three of the ten are the points at the origin, where the zero biases leave $\mathbf{Z}_1$ at exactly zero. They do no harm here: the three carry three different labels and the same prediction, so their errors cancel. The gaps come from the other seven, which lie between 0.02 and 0.71 from the origin, each close to a line through the origin on which one neuron's pre-activation changes sign. The figure below draws the three lines and rings the ten samples.

![Two scatter plots of the 300 spiral points of seed 0, class 0 as blue circles, class 1 as orange squares and class 2 as green triangles: the whole plane on the left, the square from minus 0.2 to 0.2 around the origin enlarged on the right. Three grey lines through the origin mark where each hidden neuron's pre-activation is 0 under the 0.01 weights and zero biases. Red rings mark the ten samples with an entry of Z1 within h of 0: three at the origin, one of each class, and seven on the lines, between 0.02 and 0.71 from the origin.](diagrams/02-relu-corners.svg)

*With zero biases every hidden neuron switches on a line through the origin, and the spiral arms start there.*

With only those seven left out the largest gap is $1.9 \times 10^{-11}$. With all ten left out:

```text
dense2.dweights  (3, 3)   1.8e-04           3.1e-07                  1.2e-11   FAIL
dense2.dbiases   (1, 3)   5.8e-03           8.9e-09                  1.0e-11   pass
dense1.dweights  (2, 3)   1.8e-04           1.5e-06                  9.6e-12   FAIL
dense1.dbiases   (1, 3)   3.0e-04           6.3e-08                  1.2e-11   pass
seeds 0 to 9 on this initialisation: the check fails on 10 of 10
```

Every absolute gap is back at $10^{-11}$; what still fails does so on small gradients alone. The last line counts the unchanged initialisation over ten seeds: it fails on every one. Nothing in the backward pass was wrong at any point. A check needs parameters of ordinary size and biases that are not zero, which is why section 9 redraws them, and a failed line is read together with its absolute gap.

**The division by the number of samples is left out.** A combined class whose `backward` lacks the line `self.dinputs /= samples`, on the redrawn network of section 9:

```text
gradient shapes equal parameter shapes: True; largest |row sum| of loss_activation.dinputs: 2.2e-16
dense2.dweights  (3, 3)   3.7e+01           1.0e+00                  3.7e+01   FAIL
relative error of dense2.dweights to four decimals: 0.9967; 1 - 1/N = 0.9967
```

The loss is unchanged, every shape is right, and the rows of `loss_activation.dinputs` still sum to zero, so neither the first loss, nor the shape test, nor the row-sum test of post 20 notices anything. The gradient check fails on all four arrays with the same relative error, $1 - 1/300$: every gradient is 300 times too large. An error of exactly this form points at a missing or doubled mean. A ReLU without its mask fails only from `dense1` on; post 16, section 7, measured it. The figure below sets the checks of sections 9 and 11 side by side, each gradient array as one point at its relative error and its absolute gap.

![A scatter plot on two logarithmic axes, the largest relative error of a gradient array across and its largest absolute gap up, four points per check. The passing check of section 9, blue circles, lies left of the pass mark of 10 to the minus 7 with gaps near 10 to the minus 11. The check on the 0.01 weights, orange triangles, has its two dense2 points just right of the mark at the same gaps, and its two dense1 points far higher, at gaps of 6.8 times 10 to the minus 8 and 1.1 times 10 to the minus 5; dashed arrows take them down to the rounding level once the ten samples are left out, hollow squares. A dotted line marks 6.0 times 10 to the minus 11, the largest gap at scale 1 over 50 seeds. The missing division, red diamonds, sits at a relative error of 1 with gaps from 17 to 140.](diagrams/03-error-and-gap.svg)

*Only the missing division is a bug; the dense1 points of the 0.01 check are ReLU corners.*

**The backward calls are reordered.** On fresh objects, `dense2.backward(loss_activation.dinputs)` before `loss_activation.backward` raises at once:

```text
AttributeError: 'Activation_Softmax_Loss_CategoricalCrossentropy' object has no attribute 'dinputs'
```

Each backward call reads an attribute that only the call before it creates.

**The check is run after `nnfs.init()`.** The same check, same seed, same redrawn parameters:

```text
X float32, dense1.weights float32, loss float32
float32, h = 1e-05:              largest relative error 1.9e-01 in dense1.dweights
float32, h = 1e-02:              largest relative error 1.5e-04 in dense1.dweights
arrays cast to float64, h = 1e-05: largest relative error 2.0e-03 in dense1.dweights
X float64, dense1.weights float64, but np.dot(X, dense1.weights) float32
```

A float32 loss keeps about seven digits, and two losses a step of $10^{-5}$ apart differ only around the last of them, so their difference is mostly rounding (post 10, section 7; post 15 met it on one layer). A step of $10^{-2}$ helps and still misses the pass mark by three orders of magnitude. Casting the data and the parameters to float64 afterwards is not enough either: `nnfs.init()` has replaced `np.dot` by a version that returns float32 whatever it is given (post 04), and the relative error stays at $2.0 \times 10^{-3}$. A gradient check belongs in a process that has not called `nnfs.init()`.

---

## 12. Summary

| Concept | Takeaway |
|---|---|
| The script | fifteen lines: four objects, four forward calls, four backward calls, three comments |
| First loss | 1.0986104 beside $\ln 3 = 1.0986123$; tests the forward pass and the loss only |
| Four gradient arrays | `dweights` and `dbiases` on both dense layers, 21 numbers |
| Shape test | each gradient has the shape of its parameter; necessary, and blind to wrong values |
| First gradients | at most $3.6 \times 10^{-4}$ here: balanced classes and 0.01 weights |
| Gradient check | central difference, $h = 10^{-5}$, float64, parameters of ordinary size; pass below $10^{-7}$ |
| Not here | the update and the training loop (post 22) |

---

## Common pitfalls

1. **Taking matching shapes for correct gradients.** A backward pass without the division by $N$ has every shape right and every value 300 times too large.
2. **Checking gradients on the 0.01 initialisation.** Tiny gradients and pre-activations within $h$ of zero make a correct backward pass fail; redraw weights and biases at scale 1 first.
3. **Checking gradients after `nnfs.init()`.** Its float32 arrays and its `np.dot` spoil the measurement even when the arrays are cast to float64.
4. **Reading a relative error without the absolute gap.** A gap near $10^{-11}$ on a gradient of $10^{-5}$ is rounding; a relative error near 1 is a bug.
5. **Expecting large first gradients.** On balanced data with 0.01 weights the largest of the 21 first gradients is $3.6 \times 10^{-4}$, and that is correct.
6. **Taking the script for a training run.** It is one forward and one backward pass; no parameter has changed when it ends.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms), and section 11.5 (Debugging Strategies) for comparing a backward pass with finite differences (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition*, course notes, "Putting it together: Minimal Neural Network Case Study": a two-layer classifier on the same spiral data, written with loose arrays.

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 22 - Gradient-descent optimiser](../22-gradient-descent-optimiser/index.md):** the update that turns the four gradient arrays into new parameters, and the loop that repeats this script.
- **[Post 33 - Weight initialisation](../33-weight-initialisation/index.md):** what the scale of the initial weights, 0.01 here, does to the signals and gradients of a network.
