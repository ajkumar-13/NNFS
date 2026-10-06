# 12 - Backpropagation through a single neuron

> **TL;DR.** Backpropagation is the chain rule of post 11 walked from the loss back to the parameters, and the smallest network on which that chain has more than one factor is a single neuron with three inputs, a ReLU, and a squared-error loss. For one weight the chain has four local derivatives: three are shared by every parameter and multiply into one **upstream gradient**, here 12, and the fourth is the input attached to that weight, so the four gradients are 12, $-24$, 36, and 12. One gradient-descent step with a learning rate of 0.01 takes the loss from 36 to 17.64, and every further step multiplies it by 0.49. A central difference confirms the four gradients to within $10^{-9}$, and a 200-iteration loop confirms the decay.
>
> **Prerequisites:** [Post 10](../10-derivatives-partial-derivatives-and-gradients/index.md), [Post 11](../11-the-chain-rule/index.md).
> **Safe to skip?** Skip it if the reader can already write $\partial L / \partial w_i$ for a ReLU neuron with a squared-error loss as a product of local derivatives, say which of those factors every parameter shares, and predict the loss after one gradient-descent step.
>
> **After reading, you will be able to:**
>
> - Identify the four chain-rule factors in the backward pass of a single neuron.
> - Compute by hand the gradient of a squared-error loss with respect to each weight and the bias of a three-input ReLU neuron.
> - Run a 200-iteration gradient-descent loop on one neuron and predict how fast its loss falls towards zero.

![A two-row diagram of one neuron. The forward row runs left to right: the inputs 1, -2, 3 and the bias input 1 times the weights -3, -1, 2 and the bias 1 give the products -3, 2, 6 and 1, their sum is z = 6, ReLU gives y hat = 6, and the squared error against the target 0 gives L = 36. The backward row runs right to left: factor 1, 2 times (y hat minus y), is 12, factor 2, the ReLU derivative at z > 0, is 1, and factor 3, the sum, is 1, so an upstream gradient of 12 reaches the parameters, where factor 4, the input, gives the gradients 12 times 1 = 12, 12 times -2 = -24, 12 times 3 = 36 and 12 times 1 = 12 for the bias.](diagrams/01-single-neuron-backprop.svg)

*One neuron, one chain. Every parameter receives the same upstream gradient; only the last factor differs.*

---

## 1. The question: what does the chain rule look like on the smallest network?

[Post 11](../11-the-chain-rule/index.md) stated the chain rule: the derivative of a composition is the product of the local derivatives along the chain. Post 09 stated what the result is for: every parameter moves a small step against its own slope of the loss. What is still missing is one complete, worked case in which a loss is traced back to actual weights, with numbers at every step.

This post takes the smallest network on which the chain rule still has more than one factor. It has one neuron, three inputs, three weights, one bias, a ReLU (rectified linear unit) activation, and a squared-error loss against a target. The question is: **which local derivatives multiply together to give the gradient of the loss with respect to each of the four parameters, and what do they come to?**

The point is *not* to train a useful model. It is to derive every step of the backward pass by hand, check the result against a measurement, and spend it in a 200-iteration training loop. The later posts of Part V extend this same recipe, starting with [post 13](../13-backprop-through-a-layer/index.md). The algorithm itself is old: the papers of Linnainmaa (1970), Werbos (1974), and Rumelhart, Hinton, and Williams (1986) are listed under Further reading.

---

## 2. The setup

A single neuron with three inputs:

| Quantity | Symbol | Value |
|---|---|---|
| Inputs | $x_1, x_2, x_3$ | $1,\ -2,\ 3$ |
| Weights | $w_1, w_2, w_3$ | $-3,\ -1,\ 2$ |
| Bias | $b$ | $1$ |
| Target output | $y$ | $0$ |

As in post 01, the mathematics counts from 1 and the code from 0, so `inputs[0]` holds $x_1$. The forward computation is three functions, one after another:

$$z = x_1 w_1 + x_2 w_2 + x_3 w_3 + b,$$

$$\hat{y} = \text{ReLU}(z) = \max(0, z),$$

$$L = (\hat{y} - y)^2 = \hat{y}^2 \quad (\text{since } y = 0).$$

Here $z$ is the pre-activation of post 01 (the weighted sum plus the bias) and $\hat{y}$ is the output of the neuron. In posts 06 to 08, $\hat{y}$ was a predicted probability; here it is a prediction of a real number, and the target $y$ is a real number as well.

The inputs and the target are data. They do not change during training and are treated as constants throughout the backward pass. The three weights and the bias are the parameters, the four numbers gradient descent is allowed to move.

**Goal of this post:** compute $\partial L / \partial w_1$, $\partial L / \partial w_2$, $\partial L / \partial w_3$, and $\partial L / \partial b$. Together those four numbers are the gradient $\nabla L$ of post 10, section 4, and they are everything the update rule needs.

### 2.1. Why squared error here and not cross-entropy

Squared error is the loss for a real-valued target (post 08, section 2.2), and its derivative is one line: $\partial L / \partial \hat{y} = 2(\hat{y} - y)$. Averaged over a batch it is the mean squared error of the regression project `nn-p04`. Cross-entropy returns in [post 18](../18-backpropagation-through-the-loss-function/index.md).

---

## 3. The forward pass, by hand

Plug the numbers in:

$$z = (1)(-3) + (-2)(-1) + (3)(2) + 1 = -3 + 2 + 6 + 1 = 6.$$

$$\hat{y} = \text{ReLU}(6) = 6.$$

$$L = (6 - 0)^2 = 36.$$

The loss is exactly 36: nothing in the example is random, so every run gives the same value. The forward pass also leaves behind two intermediate values, $z = 6$ and $\hat{y} = 6$. The backward pass needs both of them, so a forward pass always comes first.

---

## 4. The backward pass

The composition for $L$, read inside-out, is:

$$L = \Bigl(\text{ReLU}\bigl(\underbrace{x_1 w_1 + x_2 w_2 + x_3 w_3 + b}_{z}\bigr) - y\Bigr)^2.$$

Four functions are nested in it: a product ($x_1 w_1$), a sum (which gives $z$), the ReLU (which gives $\hat{y}$), and the squared error (which gives $L$). The chain rule for $\partial L / \partial w_1$ therefore produces four factors, one per function, written here from the loss end to the weight end:

$$\frac{\partial L}{\partial w_1} = \underbrace{\frac{\partial L}{\partial \hat{y}}}_{\text{factor 1}} \cdot \underbrace{\frac{\partial \hat{y}}{\partial z}}_{\text{factor 2}} \cdot \underbrace{\frac{\partial z}{\partial (x_1 w_1)}}_{\text{factor 3}} \cdot \underbrace{\frac{\partial (x_1 w_1)}{\partial w_1}}_{\text{factor 4}}.$$

Each of the four factors is a **local derivative**: the derivative of one function with respect to its own input, computed without looking at the rest of the chain. Computing them one at a time:

**Factor 1** is the derivative of the squared-error loss with respect to the output of the neuron:

$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2 \cdot (6 - 0) = 12.$$

**Factor 2** is the derivative of ReLU at $z = 6$. Since $z > 0$, the derivative is $1$ (post 10, section 3.1):

$$\frac{\partial \hat{y}}{\partial z} = 1.$$

**Factor 3** is the derivative of the sum with respect to one of its terms. The other terms are held fixed, so $z$ changes by exactly as much as the term does, and the derivative is $1$:

$$\frac{\partial z}{\partial (x_1 w_1)} = 1.$$

**Factor 4** is the derivative of the product $x_1 w_1$ with respect to $w_1$. With $x_1$ treated as a constant, it is $x_1$:

$$\frac{\partial (x_1 w_1)}{\partial w_1} = x_1 = 1.$$

Multiplying all four:

$$\frac{\partial L}{\partial w_1} = 12 \cdot 1 \cdot 1 \cdot 1 = 12.$$

That is the gradient of the loss with respect to the first weight. It is a rate (post 10, section 2.4): nudging $w_1$ up by $0.001$ raises the loss by about $0.012$. Three more parameters remain.

### 4.1. The pattern is clearer than the formula

A close look at the four factors shows that **only factor 4 depends on the parameter being differentiated**. Factors 1, 2, and 3 are the same whether the gradient being computed is for $w_1$, $w_2$, $w_3$, or $b$. Their product, $12 \cdot 1 \cdot 1 = 12$, is the **upstream gradient**: the gradient that arrives at the parameters from the part of the chain nearer the loss. Because factor 3 is 1, it equals $\partial L / \partial z$, the sensitivity of the loss to the pre-activation. From there:

- For $w_i$, factor 4 is the corresponding input $x_i$.
- For $b$, the bias is added to $z$ directly and is not multiplied by anything, so $\partial z / \partial b = 1$ and its gradient is the upstream gradient itself. Equivalently, factor 4 for the bias is $1$.

In one line each:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i, \qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}.$$

This is the part that scales unchanged to bigger networks: every weight's gradient is **the upstream gradient times the input connected to that weight**. That one sentence is most of backpropagation. Here every factor is a plain scalar; in the next posts the same sentence holds with the upstream gradient and the inputs promoted to vectors and matrices, and from post 16 on the upstream gradient is the array called `dvalues`.

### 4.2. When the ReLU is off

Factor 2 is 1 only because $z = 6$ is positive. If the pre-activation were negative, the ReLU would output 0, its derivative would be 0, and the upstream gradient would be factor 1 times 0, which is 0 whatever the loss is. All four gradients would then be zero and no parameter would move. This is the dead neuron of post 06, section 2.2, on the smallest possible scale: with a single sample, one negative pre-activation is enough. At exactly $z = 0$ the code uses a derivative of 0, the convention of post 10, section 3.1.

---

## 5. All four gradients

Applying the rule of section 4.1:

| Parameter | Upstream gradient (factors 1 to 3) | Factor 4 | Gradient |
|:---:|:---:|:---:|:---:|
| $w_1$ | $12$ | $x_1 = 1$ | $\mathbf{12}$ |
| $w_2$ | $12$ | $x_2 = -2$ | $\mathbf{-24}$ |
| $w_3$ | $12$ | $x_3 = 3$ | $\mathbf{36}$ |
| $b$ | $12$ | $1$ | $\mathbf{12}$ |

As a vector, $\nabla L = (12, -24, 36, 12)$.

The signs make sense. The output, 6, is above the target, 0, so the loss falls when $z$ falls. The weight $w_2$ is multiplied by a negative input, so increasing $w_2$ lowers $z$: its gradient is negative, and the update will *increase* it. The other three gradients are positive, and the update will decrease those parameters.

The sizes follow the inputs: $w_3$ has the largest input and the largest gradient, and a weight whose input is 0 would have a gradient of 0 however large the upstream gradient, because with that input the weight has no effect on the output. Section 9.1 measures the four numbers with the central difference of post 10 and finds the same values.

---

## 6. One gradient-descent step

With a learning rate $\alpha = 0.01$ and the update rule of post 09, $w_{\text{new}} = w_{\text{old}} - \alpha \cdot (\partial L / \partial w)$, applied to each parameter:

| Parameter | Old value | Gradient | $-\alpha \cdot \text{gradient}$ | New value |
|:---:|:---:|:---:|:---:|:---:|
| $w_1$ | $-3$ | $12$ | $-0.12$ | $-3.12$ |
| $w_2$ | $-1$ | $-24$ | $+0.24$ | $-0.76$ |
| $w_3$ | $2$ | $36$ | $-0.36$ | $1.64$ |
| $b$ | $1$ | $12$ | $-0.12$ | $0.88$ |

Running the new parameters through the forward pass:

$$z_{\text{new}} = (1)(-3.12) + (-2)(-0.76) + (3)(1.64) + 0.88 = -3.12 + 1.52 + 4.92 + 0.88 = 4.20.$$

$$\hat{y}_{\text{new}} = \text{ReLU}(4.20) = 4.20.$$

$$L_{\text{new}} = 4.20^2 = 17.64.$$

The figure below takes the step row by row and follows each of the four terms of $z$ through it.

![On the left, the update new = old minus 0.01 times the gradient, row by row: w1 from -3 with gradient 12 to -3.12, w2 from -1 with gradient -24 to -0.76, w3 from 2 with gradient 36 to 1.64, and the bias from 1 with gradient 12 to 0.88, with a note that the negative gradient of w2 raises it. On the right, a table of the terms of z before and after the step: -3 to -3.12, 2 to 1.52, 6 to 4.92 and 1 to 0.88, changes -0.12, -0.48, -1.08 and -0.12. Their sum z goes from 6 to 4.20, 0.7 times 6, and the loss L = z squared from 36 to 17.64, 0.49 times 36.](diagrams/02-one-step.svg)

*Each term of $z$ falls by $0.01 \cdot 12 \cdot x_i^2 = 0.12\,x_i^2$, and the bias term by $0.12$: 1.80 in all, so $z$ keeps 0.7 of its value.*

The loss dropped from $36$ to $17.64$ in a single step, to 49 percent of its value. That ratio is not a coincidence. Write $g$ for the upstream gradient. The update subtracts $\alpha g x_i$ from each weight and $\alpha g$ from the bias, so the next pre-activation is

$$z_{\text{new}} = \sum_i x_i (w_i - \alpha g x_i) + (b - \alpha g) = z - \alpha g \bigl(x_1^2 + x_2^2 + x_3^2 + 1\bigr).$$

The bracket is $1 + 4 + 9 + 1 = 15$: the sum of the squared inputs, plus 1 for the bias. While $z$ is positive and the target is 0, the upstream gradient is $g = 2z$, and so

$$z_{\text{new}} = z \,(1 - 2 \alpha \cdot 15) = z\,(1 - 0.3) = 0.7\,z.$$

Each step multiplies the pre-activation by $0.7$, and the loss, which is $z^2$, by $0.7^2 = 0.49$. After $t$ steps,

$$z_t = 6 \cdot 0.7^{\,t}, \qquad L_t = 36 \cdot 0.49^{\,t}.$$

The loss therefore decays geometrically: 36, 17.64, 8.6436, 4.2354, and so on. The factor depends on the learning rate and on the inputs, not on the weights, and section 10 shows what happens when $1 - 30\alpha$ drops below zero.

### 6.1. Why the loss approaches zero but never exactly reaches it

The formula $L_t = 36 \cdot 0.49^{\,t}$ is positive for every $t$. In exact arithmetic the loss never arrives at zero, because the gradient shrinks in step with $z$: the nearer the output is to the target, the smaller the upstream gradient $2z$, and the smaller the next step. Two things on a computer look like exceptions.

The first is display rounding. Printed to four decimal places, the loss reads `0.0000` from iteration 19 onward, but at iteration 20 its value is $36 \cdot 0.49^{20} \approx 2.3 \times 10^{-5}$: a small positive number, with a non-zero gradient.

The second is real. In the float64 run of section 9.2 the loss does become exactly `0.0`, at iteration 113. The pre-activation is a sum of four terms of ordinary size, and such a sum cannot resolve a result much smaller than $10^{-16}$ (the rounding error of post 10, section 6.1). At iteration 100 the computed $z$ is $1.8 \times 10^{-15}$, and at iteration 113 the four terms cancel exactly; from then on every gradient is zero. Whether the loss lands on exactly `0.0`, and at which iteration, is a property of the rounding: it can differ between machines and NumPy builds, and regrouping the update as `(lr * upstream) * inputs` leaves this run stalled at $z = 2.2 \times 10^{-16}$ with no exact zero at all. The geometric decay before that point does not change.

The figure below draws both on a log axis, where a constant factor per step is a straight line.

![A chart of the loss of the 200-iteration loop against the iteration, 0 to 120, on a log axis. The run lies on the dashed straight line 36 times 0.49 to the t from 36 at iteration 0, crosses the dotted line at 5 times 10 to the -5 below which the loss prints as 0.0000 at iteration 19, leaves the straight line in steps near 10 to the -30 from about iteration 100, and is exactly 0.0 from iteration 113, marked by a dotted vertical line. A table lists what the loop prints at iterations 0, 16, 20, 100 and 120: 36.0000, 0.0004, 0.0000, 0.0000 and 0.0000, against the values 36, 3.976 times 10 to the -4, 2.292 times 10 to the -5, 3.155 times 10 to the -30 and 0.](diagrams/03-loss-decay.svg)

*The printed 0.0000 is rounding; the exact zero from iteration 113 comes from the rounding of the float64 sum.*

Where the parameters stop is predictable. Every gradient is a multiple of $(x_1, x_2, x_3, 1) = (1, -2, 3, 1)$, so the parameters move along that one direction. Bringing $z$ from 6 to 0 takes a displacement $c$ with $15c = 6$, so $c = 0.4$, and they end at $(-3, -1, 2, 1) - 0.4 \cdot (1, -2, 3, 1) = (-3.4, -0.2, 0.8, 0.6)$, as section 9.2 prints.

One caution: with a target of 0 the loss is exactly zero for every $z \le 0$, since a ReLU outputs 0 there. The descent above stops at the edge of that region; a step that is too long jumps into it (section 10).

---

## 7. Why "back" propagation

The backward pass starts at the loss (the right end of the forward chain) and walks left, multiplying a running gradient by the local derivative of each function as it passes that function. By the time the walk reaches the weights at the left end, the running product is the gradient of the loss with respect to those weights.

```text
Loss  <-  ReLU  <-  Sum  <-  Multiply  <-  Weights
 12       x 1       x 1      x x_i
```

The running gradient is 12 on leaving the loss, still 12 after the ReLU and after the sum, and becomes $12 x_i$ at weight $i$.

The factors are plain numbers, so the direction of the walk does not change the product. It changes the cost. From the loss, the shared factors 1 to 3 are computed once and reused: four gradients cost one upstream gradient and four multiplications. From a parameter, the whole chain has to be walked again for every parameter. A network has many parameters and one loss, so the walk starts at the loss; that is the "back" in backpropagation, and the reason a complete gradient costs about one more pass through the network (post 10, section 4.3).

---

## 8. What this version of backpropagation is *not*

- **It is not vectorised.** Section 9 treats one neuron and one sample with scalar arithmetic. Posts 13 and 14 compute the same thing for a layer and a batch.
- **It is not the gradient with respect to the inputs.** This post computes $\partial L / \partial w$, not $\partial L / \partial x$, which is what a layer passes back to the layer before it (post 15).
- **It is not a numerical gradient.** The central difference appears in section 9.1 only as a check. Checking a complete network that way is the subject of post 21.

---

## 9. Make it run: the hand calculation and the 200-iteration loop

Three scripts under `snippets/` produce every number the post quotes. Each runs from the series root in about a second, uses no random numbers, and needs only NumPy:

- `python posts/12-backprop-through-a-single-neuron/snippets/by_hand.py` runs sections 3 to 6 and the gradient check of section 9.1.
- `python posts/12-backprop-through-a-single-neuron/snippets/training_loop.py` runs the loop of section 9.2.
- `python posts/12-backprop-through-a-single-neuron/snippets/what_can_go_wrong.py` runs section 10.

### 9.1. The four gradients, computed and measured

The backward pass of sections 4 and 5 is one short function. It takes the two values the forward pass left behind, $z$ and $\hat{y}$, and returns the upstream gradient and the four gradients:

```python
def backward(z, yhat, inputs, target):
    """The chain rule, right to left. Returns (upstream, dweights, dbias)."""
    dloss_dyhat = 2.0 * (yhat - target)       # factor 1: squared error
    dyhat_dz = 1.0 if z > 0 else 0.0          # factor 2: ReLU
    dz_dterm = 1.0                            # factor 3: a sum passes the gradient to each term
    upstream = dloss_dyhat * dyhat_dz * dz_dterm
    dweights = upstream * inputs              # factor 4 for weight i is the input x_i
    dbias = upstream * 1.0                    # factor 4 for the bias is 1
    return upstream, dweights, dbias
```

A leading `d` marks the gradient of the loss with respect to the named quantity (`notation_guide.md`), and `upstream * inputs` multiplies one number by an array of three, giving all three weight gradients at once. The script prints the factors and gradients of sections 4 and 5.

A derivation can be wrong in ways that still produce plausible numbers, so the script also *measures* the four gradients with the central difference of post 10 and the step that post settled on, $h = 10^{-5}$: each parameter in turn is moved to $+h$ and to $-h$ with the other three held fixed, and the difference of the two losses is divided by $2h$. The measurement, `numerical_gradients` in the script, never looks at the chain rule:

```text
== The same four numbers measured: central difference, h = 1e-5
dL/dw_1  chain rule   12.0   central difference   12.000000000
dL/dw_2  chain rule  -24.0   central difference  -24.000000000
dL/dw_3  chain rule   36.0   central difference   36.000000001
dL/db    chain rule   12.0   central difference   12.000000000
largest gap between the two columns: 7.7e-10
```

The two columns agree to better than $10^{-9}$. The measurement needed eight forward passes for four parameters; the chain rule needed one backward walk. The script goes on to take the step of section 6 and prints the numbers of that section's table, with measured shrink factors of 0.70 for $z$ and 0.49 for the loss.

### 9.2. The training loop

`snippets/training_loop.py` repeats the three stages (forward pass, backward pass, update) 200 times. After its docstring and the NumPy import, the file is:

```python
inputs = np.array([1.0, -2.0, 3.0])
weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
target = 0.0
lr = 0.01


def relu(z):
    return max(0.0, z)


def relu_deriv(z):
    return 1.0 if z > 0 else 0.0


first_zero = None
for i in range(200):
    # Forward pass.
    z = float(np.dot(inputs, weights)) + bias
    yhat = relu(z)
    loss = (yhat - target) ** 2

    # Backward pass: the local derivatives, multiplied right to left.
    dloss_dyhat = 2.0 * (yhat - target)     # factor 1
    dyhat_dz = relu_deriv(z)                # factor 2
    upstream = dloss_dyhat * dyhat_dz       # factor 3 is 1 for a sum
    dweights = upstream * inputs            # factor 4 for weight i: the input x_i
    dbias = upstream * 1.0                  # factor 4 for the bias: 1

    # Update.
    weights -= lr * dweights
    bias -= lr * dbias

    if first_zero is None and loss == 0.0:
        first_zero = i
    if i <= 20 and i % 4 == 0:
        print(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})")
    elif (i % 20 == 0 and i <= 120) or i == 199:
        print(f"iter {i:3d}  loss = {loss:.4f}   (exactly {loss:.3e})   z = {z:.3e}")

print(f"first iteration with a loss of exactly 0.0: {first_zero}")
print(f"final weights = {weights}")
print(f"final bias    = {bias:.6f}")
print(f"final z       = {float(np.dot(inputs, weights)) + bias:.3e}")
```

The loss printed at iteration $i$ is the one computed before that iteration's update, so iteration 0 shows the 36 of section 3. Each line prints the loss to four decimal places and in scientific notation.

```text
iter   0  loss = 36.0000   (exactly 3.600e+01)
iter   4  loss = 2.0753   (exactly 2.075e+00)
iter   8  loss = 0.1196   (exactly 1.196e-01)
iter  12  loss = 0.0069   (exactly 6.897e-03)
iter  16  loss = 0.0004   (exactly 3.976e-04)
iter  20  loss = 0.0000   (exactly 2.292e-05)
iter  40  loss = 0.0000   (exactly 1.459e-11)   z = 3.820e-06
iter  60  loss = 0.0000   (exactly 9.291e-18)   z = 3.048e-09
iter  80  loss = 0.0000   (exactly 5.916e-24)   z = 2.432e-12
iter 100  loss = 0.0000   (exactly 3.155e-30)   z = 1.776e-15
iter 120  loss = 0.0000   (exactly 0.000e+00)   z = 0.000e+00
iter 199  loss = 0.0000   (exactly 0.000e+00)   z = 0.000e+00
first iteration with a loss of exactly 0.0: 113
final weights = [-3.4 -0.2  0.8]
final bias    = 0.600000
final z       = 0.000e+00
```

Every printed loss up to iteration 60 is $36 \cdot 0.49^{\,i}$ to the digits shown. After that rounding shows (at iteration 80 the formula gives $5.915 \times 10^{-24}$), and it ends as section 6.1 described: an exact zero from iteration 113, with the parameters at $(-3.4, -0.2, 0.8)$ and $0.6$.

Forward to compute, backward to differentiate, update to descend: every training loop later in the series has the same three stages in the same order.

---

## 10. What can go wrong?

`snippets/what_can_go_wrong.py` reproduces the failures below. The first is the learning rate. With a general target $y$ the algebra of section 6 gives $z_{\text{new}} - y = (z - y)(1 - 30\alpha)$ for as long as $z$ stays positive: the gap between output and target shrinks by the factor. The script sweeps the learning rate for the target 0 of this post and for a target of 1:

```text
== 1. The learning rate: z after the first update, and the loss after 200 updates
each update multiplies (z - y) by 1 - 30 * lr while z stays positive
target y = 0
  lr = 0.01   factor =   0.70   z after 1 update =    4.20   loss after 200 = 0.000e+00
  lr = 0.035  factor =  -0.05   z after 1 update =   -0.30   loss after 200 = 0.000e+00
  lr = 0.05   factor =  -0.50   z after 1 update =   -3.00   loss after 200 = 0.000e+00
  lr = 0.1    factor =  -2.00   z after 1 update =  -12.00   loss after 200 = 0.000e+00
  lr = 1.0    factor = -29.00   z after 1 update = -174.00   loss after 200 = 0.000e+00
target y = 1
  lr = 0.01   factor =   0.70   z after 1 update =    4.50   loss after 200 = 1.972e-31
  lr = 0.035  factor =  -0.05   z after 1 update =    0.75   loss after 200 = 0.000e+00
  lr = 0.05   factor =  -0.50   z after 1 update =   -1.50   loss after 200 = 1.000e+00
  lr = 0.1    factor =  -2.00   z after 1 update =   -9.00   loss after 200 = 1.000e+00
  lr = 1.0    factor = -29.00   z after 1 update = -144.00   loss after 200 = 1.000e+00
```

The script also multiplies the bias gradient by an input and compares the result with a central difference:

```text
== 3. A bias gradient multiplied by an input, against a central difference (h = 1e-5)
dbias = upstream * 1    (correct) =   12.0   central difference 12.000000   relative error 3.8e-11
dbias = upstream * x_2  (wrong)   =  -24.0   central difference 12.000000   relative error 1.5e+00
dbias = upstream * x_3  (wrong)   =   36.0   central difference 12.000000   relative error 6.7e-01
```

- **The learning rate is too large, and one step kills the neuron.** Once $30\alpha$ exceeds 1 the factor is negative and the first update carries $z$ past the target. With $\alpha = 0.035$ and a target of 1 the overshoot is small, $z$ lands at 0.75, still positive, and the run recovers. With $\alpha = 0.05$ it lands at $-1.5$: the ReLU outputs 0, factor 2 is 0, every gradient is 0, and the loss stays at $(0 - 1)^2 = 1$ for all 200 updates.
- **With the target of this post, the same overshoot looks like a success.** For $y = 0$ every $\alpha$ from 0.035 upward ends with a loss of exactly zero after a single update, including $\alpha = 1$, which throws $z$ to $-174$. A dead neuron outputs 0, and 0 happens to be the target: the run switched the neuron off, and the loss cannot tell the difference. Hence the second sweep, with a target of 1.
- **The neuron starts dead.** With weights $(-3, -1, -2)$ the pre-activation is $-6$ and, as section 4.2 said, all four gradients are zero: the script's loss against a target of 1 is still 1.0 after 200 updates. No learning rate helps.
- **The bias gradient is multiplied by an input.** The bias gradient is the upstream gradient alone. The update still runs with a wrong one, so the error is easy to miss by eye, but the central difference above catches it at once. The relative error is the absolute difference of the two values divided by the larger of their absolute values. Multiplying by $x_1 = 1$ would have gone unnoticed, which is a reason not to test with inputs equal to 1.
- **The gradient check lands on the corner.** At a pre-activation of exactly 0 the script's backward pass returns a bias gradient of 0 and the central difference about $-1$; the comparison is invalid there, not the backward pass (post 10, section 7).

---

## 11. Summary

| Concept | Takeaway |
|---|---|
| Backpropagation | The chain rule, walked right to left through the forward computation |
| Four factors | $2(\hat{y} - y) = 12$ for the loss; $1$ for the ReLU at $z > 0$; $1$ for the sum; the input $x_i$ for the product |
| Upstream gradient | The product of factors 1 to 3, $\partial L / \partial z = 12$, shared by every parameter; the bias takes it unchanged |
| The four gradients | $(12, -24, 36, 12)$, confirmed by a central difference to within $10^{-9}$ |
| One step | With $\alpha = 0.01$ the loss goes from 36 to 17.64; each step multiplies $z$ by $1 - 30\alpha = 0.7$ and the loss by $0.49$ |
| ReLU off | $z \le 0$ makes factor 2 zero, and every gradient with it |

---

## Common pitfalls

1. **Giving the bias an input.** The bias is added to $z$, so $\partial z / \partial b = 1$ and its gradient is the upstream gradient itself.
2. **Confusing the upstream gradient with the loss.** The loss is a value, 36 here; the upstream gradient is a derivative, $\partial L / \partial z = 12$ here, and it is the one multiplied by the inputs.
3. **Changing the ReLU convention at $z = 0$ between places.** The series uses a derivative of 0 at the corner (`z > 0` in code), everywhere.
4. **Differentiating at stale values.** The ReLU derivative needs the $z$, and the loss derivative the $\hat{y}$, of the forward pass just run with the current parameters. The forward pass comes first and its intermediate values are kept.
5. **Choosing a learning rate that overshoots.** Above $\alpha = 1/30$ on this neuron the step passes the target, and a long enough step lands on the flat side of the ReLU, where training stops.
6. **Confusing $\partial L / \partial w$ with $\partial L / \partial x$.** The first updates a weight; the second is passed back to an earlier layer and updates nothing by itself.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms) (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 9 (2020).
- Linnainmaa, S., *"The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors"* (Master's thesis, University of Helsinki, 1970).
- Nielsen, M., *Neural Networks and Deep Learning*, chapter 2 (online, 2015).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).
- Werbos, P. J., *"Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences"* (PhD thesis, Harvard University, 1974).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 13 - Backpropagation through a layer of neurons](../13-backprop-through-a-layer/index.md):** the same recipe for several neurons that share one set of inputs.
- **[Post 16 - Coding backpropagation](../16-coding-backpropagation/index.md):** where the upstream gradient becomes `dvalues` and the recipe becomes a `backward` method on `Layer_Dense`.
