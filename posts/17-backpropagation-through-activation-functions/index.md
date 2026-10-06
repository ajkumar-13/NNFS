# 17 - Backpropagation through activation functions

> **TL;DR.** An activation's backward step multiplies the incoming gradient by the activation's Jacobian, and the form of that Jacobian decides the code. For an **element-wise** activation (ReLU, sigmoid, tanh) the Jacobian is diagonal, so the step is the single line `dinputs = dvalues * f_prime(Z)`; for ReLU the slope is 0 or 1 and the line becomes a mask, which turns the upstream gradient $[5, 6, 7]$ at inputs $[1, -2, 3]$ into $[5, 0, 7]$. Softmax is **coupled**: all 9 entries of its Jacobian at the same three inputs are non-zero, and applying the element-wise line to it misses the measured gradient by 0.731.
>
> **Prerequisites:** [Post 16](../16-coding-backpropagation/index.md).
> **Safe to skip?** Skip it if the reader can already write the backward of ReLU, sigmoid, and tanh from their derivatives, and say what the off-diagonal entries of the softmax Jacobian do to that recipe.
>
> **After reading, you will be able to:**
>
> - Apply the element-wise activation backward in one line for ReLU, sigmoid, or tanh.
> - Say why ReLU's backward needs less arithmetic than sigmoid's although the code shape is the same.
> - Explain why softmax cannot be backpropagated element-wise.

![Two rows, each the row dvalues = 5, 6, 7 times a 3 by 3 Jacobian at z = 1, minus 2, 3, rows a1 to a3, columns z1 to z3, with the diagonal outlined. Top, sigmoid: only the diagonal is non-zero, 0.196612, 0.104994 and 0.045177, and the product 0.98306, 0.62996, 0.31624 equals dvalues * f_prime(Z), one multiply per entry. Bottom, softmax: all nine entries are non-zero, and the product minus 0.208216, minus 0.004467, 0.212683 matches the central difference, which the element-wise line misses by 0.731.](diagrams/01-elementwise-vs-coupled.svg)

*Top, a diagonal Jacobian: one slope per neuron and one multiply per element. Bottom, a full Jacobian: every output depends on every input, and the backward step is a matrix product for each sample.*

---

## 1. The question: what does an activation do to the gradient on its way back?

A typical block of the forward pass is a dense layer followed by an activation:

$$\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b}, \qquad \mathbf{A} = f(\mathbf{Z}).$$

Post 16 gave `Layer_Dense` and `Activation_ReLU` a `backward` method each. During the backward pass the activation is reached first. It receives the gradient of the loss at its *output*, $\partial L / \partial \mathbf{A}$ (named `dvalues` in code), and must produce the gradient at its *input*, $\partial L / \partial \mathbf{Z}$ (stored as `dinputs`), which the dense layer before it then receives as its own `dvalues`.

Take one sample, with pre-activations $z_1, \dots, z_m$ and outputs $a_1, \dots, a_m$. The loss depends on $z_j$ through every output that $z_j$ influences, so the chain rule sums over the outputs, as it summed over the neurons of a layer in post 15:

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^{m} \frac{\partial L}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_j}.$$

The factors $\partial a_k / \partial z_j$ are the local derivatives of the activation. Arranged in an $m \times m$ table with row $k$ and column $j$, they form its **Jacobian**. How many of them are non-zero depends on the kind of activation, and two cases matter:

- **Element-wise activation** (ReLU, sigmoid, tanh, and the variants of section 3.3). $a_k$ depends on $z_k$ alone. Every term of the sum with $k \ne j$ is zero, and one term is left.
- **Coupled activation** (softmax). $a_k$ depends on *every* $z_j$, because the softmax denominator sums over all the inputs. No term drops out.

Sections 2 and 3 work out the first case, where the sum collapses to one multiplication per element. Section 4 shows on three numbers why the second case does not collapse. The figure at the top of the post puts the two cases side by side on those three numbers: the sigmoid Jacobian of section 3.2 above the softmax Jacobian of section 4.

---

## 2. ReLU's backward, in detail

ReLU is $f(z) = \max(0, z)$, and post 10 found its derivative, a step function:

$$\frac{d \, \text{ReLU}(z)}{dz} = \begin{cases} 1 & z > 0 \\ 0 & z \le 0. \end{cases}$$

At $z = 0$ the two sides disagree and the derivative is strictly undefined; the series' code uses 0 there, as post 10 stated. With only the $k = j$ term left in the sum of section 1, the chain rule for neuron $k$ is

$$\frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial a_k} \cdot \frac{d a_k}{d z_k} = \begin{cases} \dfrac{\partial L}{\partial a_k} & z_k > 0 \\ 0 & z_k \le 0. \end{cases}$$

In words: **the gradient passes through unchanged where the input was positive and is set to zero where the input was not positive**. That sentence is the whole ReLU backward.

### 2.1. Worked example

| Neuron | $z_k$ (forward input) | $\partial L / \partial a_k$ (`dvalues`) | $\partial L / \partial z_k$ (`dinputs`) |
|:---:|:---:|:---:|:---:|
| 1 | $1.0$ (positive) | $5$ | $\mathbf{5}$ |
| 2 | $-2.0$ (not positive) | $6$ | $\mathbf{0}$ |
| 3 | $3.0$ (positive) | $7$ | $\mathbf{7}$ |

Neuron 2's gradient is zeroed because its input fell in ReLU's flat region: a small change in $z_2$ leaves $a_2$ at 0, so it cannot change the loss, whatever the 6 arriving from above says. For this one sample neuron 2 passes back nothing. If the same happens for every sample, the neuron is the **dead neuron** of post 06, and section 7 shows what that does to the weights behind it.

On the positive side nothing is lost. The slope is exactly 1, so the 5 and the 7 arrive at the dense layer at full size.

### 2.2. The code

```python
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs                    # cached: backward needs the sign of every input
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()           # a copy, so the caller's array is left alone
        self.dinputs[self.inputs <= 0] = 0      # no gradient where the input was not positive
```

This is the class of post 16, unchanged. `forward` caches its input because `backward` runs later and needs to know which inputs were positive. `backward` starts from a copy of `dvalues` and overwrites with 0 every position where the cached input was not positive. The `.copy()` matters: without it `self.dinputs` would be a second name for the caller's array, and the masking line would overwrite it (section 7). The method stores its result in `self.dinputs` and returns nothing, like `Layer_Dense.backward`, because the next component reads the attribute.

The figure below runs the worked example through the class: the forward method turns the input on the left into the output on the right, and the backward method turns the gradient arriving on the right into the masked gradient on the left.

![A card for the class Activation_ReLU, its forward method above its backward method, with the line self.inputs = inputs tagged cached. Forward, left to right: the inputs 1, minus 2, 3 go in and the output 1, 0, 3 comes out. Backward, right to left in purple: dvalues 5, 6, 7 goes in and dinputs 5, 0, 7 comes out, its 0 outlined under the input minus 2. Every array has shape 1 by 3, and the caller's dvalues is unchanged after the call.](diagrams/02-relu-backward.svg)

*The backward method reads the input that forward cached and zeroes the gradient where that input was not positive.*

`snippets/relu_backward.py` runs the class on the table above:

```text
== Section 2.1: the worked example
inputs  Z       : [[ 1. -2.  3.]]
output  A       : [[1. 0. 3.]]
dvalues dL/dA   : [[5. 6. 7.]]
dinputs dL/dZ   : [[5. 0. 7.]]
dvalues after the call, unchanged: [[5. 6. 7.]]
```

### 2.3. Testing the input or testing the output

The mask is written as `self.inputs <= 0`. For plain ReLU the test `self.output <= 0` selects exactly the same positions. The output is $\max(0, z)$, which is never negative and is 0 exactly when $z \le 0$, so "the output is not positive" and "the input is not positive" are the same statement, including at $z = 0$. The script counts both on a $4 \times 5$ batch in which two inputs sit exactly on the corner:

```text
== Section 2.3: testing the input or testing the output
entries: 20  exactly zero: 2  negative: 5
entries zeroed when the input is tested : 7
entries zeroed when the output is tested: 7
positions where the two masks differ    : 0
```

The series tests the input because that is the form in which the derivative is written, for ReLU and for every variant of it. The choice between `<=` and `<` only decides what happens at exactly $z = 0$, where either answer is a convention; `<=` gives the 0 that the rest of the series uses.

---

## 3. The general element-wise pattern

ReLU is one instance of a broader pattern. For any element-wise activation $a_k = f(z_k)$, the sum of section 1 keeps one term:

$$\frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial a_k} \cdot f'(z_k).$$

For a whole batch this is one **element-wise multiplication** in NumPy, `dinputs = dvalues * f_prime(Z)`, with no matrix product, no reshape, and no transpose. `dvalues`, `Z`, and `dinputs` all have the shape $(N, n_\text{neurons})$. ReLU's mask is this line with a slope of 0 or 1: multiplying by 1 keeps an entry and multiplying by 0 clears it. The figure below draws the three activations of the table that follows with their slopes, and sends the upstream gradient of section 2.1 through each.

![Three charts for z from minus 4 to 4, each with an activation in green and its slope in purple: ReLU, whose slope steps from 0 to 1 at z = 0 and is 0 at z = 0 itself; sigmoid, whose slope a(1 - a) peaks at 0.25; and tanh, whose slope 1 - a squared peaks at 1. Dots mark the slope at z = 1, minus 2, 3. The tables under the charts give the slopes and dinputs for dvalues = 5, 6, 7: 1, 0, 1 and 5, 0, 7 for ReLU; 0.197, 0.105, 0.045 and 0.983, 0.630, 0.316 for sigmoid; 0.420, 0.071, 0.010 and 2.100, 0.424, 0.069 for tanh. A line at the bottom gives the code all three share, dinputs = dvalues * f_prime(Z).](diagrams/03-elementwise-family.svg)

*The backward line is the same for all three. Only what `f_prime` returns changes, so a new element-wise activation costs one derivative.*

| Activation | $f(z)$ | $f'(z)$ | Backward |
|---|---|---|---|
| **ReLU** | $\max(0, z)$ | $1$ if $z > 0$, else $0$ | mask: zero where $z \le 0$ |
| **Sigmoid** | $\sigma(z) = 1 / (1 + e^{-z})$ | $\sigma(z) \, (1 - \sigma(z))$ | multiply by $a \, (1 - a)$ |
| **Tanh** | $(e^z - e^{-z}) / (e^z + e^{-z})$ | $1 - \tanh^2(z)$ | multiply by $1 - a^2$ |

### 3.1. The slopes of sigmoid and tanh

Both slopes follow from the chain rule of post 11 and two facts that post 10 did not derive. The first is that $e^z$ is its own derivative, which post 10 mentioned in its section 2.4. The second is that the power rule, stated there for whole-number powers, also holds for the exponent $-1$: the derivative of $u^{-1}$ is $-u^{-2}$.

**Sigmoid.** Write $\sigma(z) = u^{-1}$ with $u = 1 + e^{-z}$. The power rule gives $d\sigma/du = -u^{-2}$, and $du/dz = -e^{-z}$, so

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \, \bigl(1 - \sigma(z)\bigr).$$

The last step uses $1 - \sigma(z) = e^{-z} / (1 + e^{-z})$.

**Tanh.** Multiplying the numerator and the denominator of $\tanh(z)$ by $e^{-z}$ gives $(1 - e^{-2z}) / (1 + e^{-2z})$, which equals $2\sigma(2z) - 1$. Differentiating with the sigmoid result, and with the factor 2 that the inner $2z$ contributes,

$$\tanh'(z) = 4 \, \sigma(2z) \, \bigl(1 - \sigma(2z)\bigr) = \bigl(1 + \tanh(z)\bigr)\bigl(1 - \tanh(z)\bigr) = 1 - \tanh^2(z),$$

because $\sigma(2z) = (1 + \tanh(z))/2$ and $1 - \sigma(2z) = (1 - \tanh(z))/2$.

Each slope is written in terms of the function's own value. The backward step therefore needs no new exponential: it reads the output that `forward` already cached.

```python
class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # f'(z) = sigma(z) * (1 - sigma(z)), read from the cached output
        self.dinputs = dvalues * self.output * (1 - self.output)


class Activation_Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # f'(z) = 1 - tanh(z)^2, read from the cached output
        self.dinputs = dvalues * (1 - self.output ** 2)
```

`Activation_Sigmoid` returns in post 34 as the output activation for binary classification. `Activation_Tanh` is written here for the pattern and is not used again in the series. `snippets/elementwise_backward.py` sends the upstream gradient of section 2.1 through all three:

```text
== Section 3: one upstream gradient through three activations
Z       = [ 1. -2.  3.]
dvalues = [5. 6. 7.]
                  local slope f_prime(Z)    dinputs = dvalues * f_prime(Z)
ReLU        [1.000000 0.000000 1.000000]         [5.00000 0.00000 7.00000]
sigmoid     [0.196612 0.104994 0.045177]         [0.98306 0.62996 0.31624]
tanh        [0.419974 0.070651 0.009866]         [2.09987 0.42390 0.06906]
```

ReLU hands back 5 and 7 untouched. Sigmoid hands back less than a fifth of each entry, and tanh less than half. Both slopes are largest at $z = 0$ and fall away quickly on either side:

```text
== Section 3.1: the two slopes at their peak and far from it
z                         0           2           5          10
sigmoid slope      2.50e-01    1.05e-01    6.65e-03    4.54e-05
tanh slope         1.00e+00    7.07e-02    1.82e-04    8.24e-09
```

This is the saturation of post 06, which listed the same slopes before the series had a backward pass to use them in. In the backward line the slope is the factor itself: a sigmoid never passes more than a quarter of the gradient it receives, and at $z = 10$ it passes less than a twenty-thousandth.

### 3.2. The Jacobian view

For one sample with $m$ neurons, the Jacobian $\partial \mathbf{a} / \partial \mathbf{z}$ is the $m \times m$ matrix whose $(k, j)$ entry is $\partial a_k / \partial z_j$. For an element-wise activation the entry is zero unless $k = j$, because $a_k$ does not depend on $z_j$ when $k \ne j$. The Jacobian is therefore **diagonal**, with $f'(z_k)$ on the diagonal.

With the gradient written as a row vector, the sum of section 1 is the product of that row with the Jacobian, `dvalues @ J`. Multiplying a row by a diagonal matrix multiplies each entry by the matching diagonal value, which is exactly the element-wise line. The script builds the sigmoid Jacobian for the sample above and computes both:

```text
== Section 3.2: the Jacobian of one sample is diagonal
sigmoid Jacobian at Z:
[[0.196612 0.000000 0.000000]
 [0.000000 0.104994 0.000000]
 [0.000000 0.000000 0.045177]]
non-zero entries: 3 of 9
dvalues @ jacobian      = [0.98306 0.62996 0.31624]
dvalues * f_prime(Z)    = [0.98306 0.62996 0.31624]
```

The matrix form is correct and wasteful: it stores $m^2$ numbers of which at most $m$ are non-zero.

### 3.3. Why ReLU's backward needs the least arithmetic

The three classes have the same shape: cache in `forward`, one line of real work in `backward`. What differs is the arithmetic inside that line.

- **ReLU** needs one comparison per element, $z_k \le 0$, and no arithmetic on the gradient at all. Because the slope is 0 or 1, an entry of `dvalues` is either kept as it is or replaced by 0. The forward pass is one `np.maximum`.
- **Sigmoid** needs one subtraction and two multiplications per element in `backward`, and its forward pass needs a negation, an exponential, an addition, and a division per element. **Tanh** is similar: a multiplication, a subtraction, and a multiplication going back, and exponentials going forward.

This is a count of arithmetic operations per element, not a timing. `Activation_ReLU.backward` as written also copies `dvalues` and writes its zeros through a boolean mask, and NumPy does neither for free, so the count does not say which of the `backward` calls returns first on a given machine.

The second difference is in what arrives, not in what it costs to compute. On its active side ReLU multiplies the gradient by exactly 1, so the gradient crosses any number of active ReLU layers without shrinking. Sigmoid multiplies by at most 0.25 even at its best point, and tanh by less than 1 everywhere except at $z = 0$. Post 06 worked out the consequence: ten sigmoid layers scale a gradient by at most $0.25^{10} \approx 9.5 \times 10^{-7}$ through the activations alone. This is why ReLU is the default for hidden layers, and why sigmoid and tanh are listed here for the pattern and not as a recommendation.

ReLU's own weakness is the zero on the other side, and several variants relax it. Leaky ReLU replaces the flat region with a small slope, commonly 0.01, so its derivative is 1 for $z > 0$ and 0.01 otherwise, and its backward is `dvalues * np.where(self.inputs > 0, 1.0, 0.01)`. Parametric ReLU (He and colleagues, 2015) learns that slope, and GELU (Hendrycks and Gimpel, 2016) replaces the corner with a smooth curve. All of them are element-wise, so each backward is still one line with a different derivative.

---

## 4. Why softmax is different

Softmax is **not** element-wise. For one sample, each output is

$$a_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}},$$

so $a_k$ depends on every $z_j$ through the denominator. Raising one input raises its own output and, because the outputs must still sum to 1, lowers all the others. `snippets/softmax_is_coupled.py` shows it on the inputs of section 2.1:

```text
== Section 4: one sample through softmax
z = [ 1.000000 -2.000000  3.000000]
a = [0.118500 0.005900 0.875601]  sum = 1.000000

== Raising z_1 alone by 0.1 moves every output
a after  = [0.129350 0.005827 0.864823]
change   = [ 0.010851 -0.000073 -0.010778]
```

Only $z_1$ moved, and all three outputs changed. The Jacobian therefore has entries off the diagonal:

$$\frac{\partial a_k}{\partial z_j} = \begin{cases} a_k (1 - a_k) & k = j \\ -a_k a_j & k \ne j. \end{cases}$$

Post 19 derives these entries; here they are only evaluated.

```text
== The Jacobian, entry (k, j) = d a_k / d z_j
[[ 0.104457 -0.000699 -0.103758]
 [-0.000699  0.005865 -0.005166]
 [-0.103758 -0.005166  0.108924]]
non-zero entries: 9 of 9
every column sums to zero: True
```

All 9 entries are non-zero, and none of the terms in the sum of section 1 can be dropped. The diagonal entries $a_k(1 - a_k)$ have the same form as the sigmoid slope, which makes the element-wise line tempting. The script computes both candidates and measures the true gradient with the central difference of post 10:

```python
full = dvalues @ jacobian                       # sum over k of dL/da_k * d a_k / d z_j
diagonal_only = dvalues * a * (1 - a)           # the element-wise line, which keeps k = j only
```

```text
== Two candidate backward passes for dvalues = [5.000000 6.000000 7.000000]
full Jacobian product : [-0.208216 -0.004467  0.212683]
central difference    : [-0.208216 -0.004467  0.212683]
element-wise line     : [0.522287 0.035190 0.762469]
largest gap, full Jacobian product against the central difference: 2.9e-11
largest gap, element-wise line against the central difference    : 0.731
```

The full product agrees with the measurement. The element-wise line is wrong in size and, for the first two inputs, in sign, and it runs without any error, because its result has the right shape.

The cost of doing it properly is a matrix product for every sample: softmax over $K$ classes has a $K \times K$ Jacobian per row of the batch. [Post 19](../19-softmax-derivatives-and-the-combined-backward-pass/index.md) derives the Jacobian and shows that pairing softmax with categorical cross-entropy cancels almost all of it, so that the full matrix never has to be built.

---

## 5. Toolkit status

After posts 16 and 17 the classes of the series stand as follows.

| Class | `forward` | `backward` |
|---|:---:|:---:|
| `Layer_Dense` | post 04 | post 16 |
| `Activation_ReLU` | post 06 | post 16, explained in detail here |
| `Activation_Softmax` | post 06 | **pending** (post 19) |
| `Loss_CategoricalCrossentropy` | post 08 | **pending** (post 18) |

Two backward methods are left: the cross-entropy backward of post 18, and the softmax backward that post 19 combines with it.

---

## 6. Make it run: every backward against a finite difference

Four scripts under `snippets/` produce every number in this post: `relu_backward.py` (section 2), `elementwise_backward.py` (section 3), `softmax_is_coupled.py` (section 4), and `what_can_go_wrong.py` (section 7). Each needs only NumPy, prints the same output on every run, and runs from the series root in about a second, for example `python posts/17-backpropagation-through-activation-functions/snippets/relu_backward.py`.

A backward method is a claim about a derivative, and post 10 gave the tool for testing one: a central difference with $h = 10^{-5}$ in float64. The test needs a scalar loss whose gradient at the activation's output is known. The scripts use $L = \sum \text{dvalues} \cdot f(\mathbf{Z})$, summed over every entry, with `dvalues` a fixed array of random numbers: its gradient with respect to $\mathbf{A} = f(\mathbf{Z})$ is exactly `dvalues`. Each entry of $\mathbf{Z}$ is then nudged by $\pm h$ in turn, and the measured slope is compared with what `backward` stored in `dinputs`. On a seeded $3 \times 4$ batch (`np.random.seed(0)`):

```text
== Section 6: each backward against a central difference, h = 1e-5, float64
ReLU     largest gap between dinputs and the central difference: 6.7e-11
sigmoid  largest gap between dinputs and the central difference: 2.2e-11
tanh     largest gap between dinputs and the central difference: 8.3e-11
```

All three agree with the measurement to better than $10^{-10}$. For ReLU the check is only meaningful away from the corner: a central difference taken at $z = 0$ returns 0.5, which is neither slope (post 10, section 7), and one taken within $h$ of the corner returns some other value between 0 and 1. The smallest $|z|$ in this batch is 0.1032, far more than $h$ from the corner.

---

## 7. What can go wrong?

`snippets/what_can_go_wrong.py` reproduces three failures, and section 4 has already shown a fourth.

```text
== 1. The copy is left out
with .copy()   : dinputs = [5. 0. 7.], the caller's dvalues afterwards = [5. 6. 7.]
without .copy(): dinputs = [5. 0. 7.], the caller's dvalues afterwards = [5. 0. 7.]

== 2. The sigmoid slope is computed from the input instead of the cached output
largest gap to the central difference, slope from the output: 2.2e-11
largest gap to the central difference, slope from the input : 2.59
shapes of the two results: (3, 4) (3, 4)

== 3. A dead neuron: one column of Z is negative for every sample
dinputs of the activation:
[[ 0.3  0.   0.1]
 [ 0.   0.   0.6]
 [ 0.7  0.   0. ]
 [-0.4  0.   0.2]]
dweights of the dense layer before it:
[[-0.080  0.000  1.170]
 [ 0.620  0.000  0.340]]
dbiases: [[0.600 0.000 0.900]]
```

- **The copy is left out.** `self.dinputs = dvalues` gives the caller's array a second name, and the masking line then writes zeros into it. The activation's own result is still right, $[5, 0, 7]$, which is why the bug is hard to see; the damage is to the caller's $[5, 6, 7]$, which now also reads $[5, 0, 7]$. Any code that uses that array afterwards computes with the wrong numbers.
- **The slope is evaluated on the wrong array.** The formula $\sigma (1 - \sigma)$ takes the sigmoid's *output*. Writing `dvalues * Z * (1 - Z)` applies it to the input. The result has the right shape and raises no error, and it is off by as much as 2.59 on a batch where the correct line is within $2.2 \times 10^{-11}$ of the measurement. The same slip with tanh is `1 - Z ** 2`. The check of section 6 catches both at once.
- **A neuron is dead.** In the third block the second column of $\mathbf{Z}$ is negative for all four samples. The mask zeroes that whole column of `dinputs`, and the dense layer before the activation then computes a zero column of `dweights` and a zero in `dbiases` for that neuron. Its parameters receive no update from this batch. If the same holds for every batch, the neuron stays as it is for good. The backward pass describes the problem and does not repair it; a smaller learning rate and a leaky ReLU are the usual remedies.
- **The element-wise line is applied to softmax.** Section 4 measured it: a gap of 0.731 and two wrong signs, with no error raised.

---

## 8. Summary

| Concept | Takeaway |
|---|---|
| The backward step of an activation | $\partial L / \partial z_j = \sum_k (\partial L / \partial a_k)(\partial a_k / \partial z_j)$: the incoming gradient times the Jacobian |
| Element-wise activation | Diagonal Jacobian; the sum keeps one term; `dinputs = dvalues * f_prime(Z)` |
| ReLU backward | A masked copy: keep the gradient where the input was positive, zero it elsewhere |
| Sigmoid and tanh backward | Multiply by $a(1 - a)$ and by $1 - a^2$, read from the cached output |
| Why ReLU needs least arithmetic | A comparison and no arithmetic; a slope of exactly 1 on the active side |
| Softmax backward | Full Jacobian; a matrix product per sample; derived in post 19 |
| Practical default | ReLU on every hidden layer; softmax on the output for classification |

---

## Common pitfalls

1. **Forgetting `.copy()` in the masked backward.** Without it the mask is written into the caller's `dvalues`, and the bug shows up somewhere else (section 7).
2. **Forgetting to cache in `forward`.** ReLU's mask needs `self.inputs`, and the sigmoid and tanh lines need `self.output`. An activation whose `forward` does not store them has nothing to compute its slope from.
3. **Computing the sigmoid or tanh slope from the input.** $\sigma(1 - \sigma)$ and $1 - \tanh^2$ take the output of the activation. Applied to `Z` they return an array of the right shape and the wrong values.
4. **Backpropagating softmax element-wise.** Softmax is the one activation of the series whose Jacobian is full. Multiplying by $a_k(1 - a_k)$ keeps the diagonal and silently drops six of the nine terms in the three-class example.
5. **Returning `dinputs` instead of storing it.** The next component reads `activation.dinputs`, as it reads `dense.dinputs`. A `backward` that only returns the array leaves the attribute missing or stale.
6. **Using sigmoid for hidden layers because its backward is just as short.** The line is as short, but the slope is at most 0.25 and near zero where the unit saturates, so the gradient shrinks at every layer it crosses.

---

## Further reading

- Glorot, X., Bordes, A., and Bengio, Y., *"Deep Sparse Rectifier Neural Networks"* (AISTATS, 2011).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.3 (Hidden Units) (MIT Press, 2016).
- He, K., Zhang, X., Ren, S., and Sun, J., *"Delving Deep into Rectifiers"* (ICCV, 2015), for the parametric ReLU.
- Hendrycks, D. and Gimpel, K., *"Gaussian Error Linear Units (GELUs)"* (arXiv:1606.08415, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 18 - Backpropagation through the loss function](../18-backpropagation-through-the-loss-function/index.md):** where the first `dvalues` of the backward pass comes from.
- **[Post 19 - Softmax derivatives and the combined backward pass](../19-softmax-derivatives-and-the-combined-backward-pass/index.md):** the full softmax Jacobian of section 4, and the cancellation that avoids building it.
