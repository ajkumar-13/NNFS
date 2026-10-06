# 10 - Derivatives, partial derivatives, and gradients

> **TL;DR.** Gradient descent (post 09) needs the gradient of the loss, and a gradient is a vector of partial derivatives, each of which is an ordinary derivative taken with every other variable held fixed. This post builds the three ideas in order: the **derivative** (the slope of a single-variable function), the **partial derivative** (the same slope along one axis of a multi-variable function), and the **gradient** (all the partial derivatives packed into one vector, which points in the direction of steepest ascent). It then measures them with **finite differences**: for $x^3$ at $x = 2$ a forward difference with $h = 10^{-3}$ gives 12.006001 and a central difference 12.000001 against the exact 12, and shrinking $h$ helps the central difference only down to about $10^{-5}$, below which rounding error takes over.
>
> **Prerequisites:** [Post 09](../09-introduction-to-optimisation/index.md).
> **Safe to skip?** Skip it if the reader can already differentiate $5x^3 + 2x$, find both partial derivatives of $x^2 y + 3y$, say what one component of a loss gradient means for one weight, and explain why a finite difference gets worse when its step is made too small.
>
> **After reading, you will be able to:**
>
> - Differentiate any polynomial using the power and sum rules.
> - Compute partial derivatives of a multi-variable function by holding the other variables constant.
> - Read a gradient vector component by component as the sensitivity of the loss to one weight.
> - Check an analytic derivative against a forward or a central finite difference at a sensibly chosen step size.

![The parabola f(x) = x squared for x from -3 to 3, with its tangent line drawn at three points: slope -3 at x = -1.5, slope +1 at x = 0.5, and slope +4 at x = 2. A table beside it lists the rule f'(x) = 2x at the three points, -3, 1 and 4, next to the central differences with h = 10 to the -5, -3.000000, 1.000000 and 4.000000.](diagrams/01-derivative-as-slope.svg)

*Three points on the same curve, three tangent lines, three slopes: $-3$, $+1$, and $+4$. The derivative is the function that returns the slope at any point, and a central difference measures the same three slopes.*

---

## 1. The question: what is a gradient, and why this calculus specifically?

Post 09 ended with the gradient-descent update rule: every parameter moves a small step against its own slope of the loss. With $\theta$ standing for the parameters, the rule reads

$$\theta \leftarrow \theta - \alpha \, \nabla L.$$

Three symbols carry weight in that line. The learning rate $\alpha$ is a positive scalar, covered from post 22 onward. The parameters $\theta$ are the weights and biases held by the `Layer_Dense` class. The third symbol, $\nabla L$, is the **gradient** of the loss with respect to the parameters. Computing $\nabla L$ is what backpropagation does (posts 12 to 21), but understanding what it *is* requires three building blocks of calculus:

| Building block | Question it answers | Where it appears next |
|---|---|---|
| **Derivative** | How does $f(x)$ change with $x$, for a function of a single variable? | Activation derivatives (post 17), loss derivatives (post 18) |
| **Partial derivative** | How does $f(w_1, w_2, \dots)$ change with one of its variables, the others held fixed? | Layer derivatives (posts 13 to 15), every weight update |
| **Gradient** | What is the vector of all the partial derivatives, taken together? | The right-hand side of every optimiser update (post 22 onward) |

Differential calculus is far older than neural networks. Newton and Leibniz developed it independently, Newton in the mid-1660s and Leibniz in the mid-1670s. Leibniz published first, in 1684, and Newton's own treatise on the method, written in 1671, was printed only in 1736; the artificial neuron of post 01 dates from 1943. The neural-network use of the subject is narrow: only the derivatives of polynomials, of ReLU, of the exponential inside softmax and sigmoid, and of the logarithm inside cross-entropy will appear in this series. This post covers polynomials and ReLU, and section 7 names the post in which each of the others is derived.

A fourth idea rides along with the three. A derivative can also be *measured*, by evaluating the function at two nearby points and taking the slope between them. That finite-difference estimate (section 2.5) is how every rule in this post is checked in section 6, and it returns in post 21 as the test of a complete backward pass.

---

## 2. Derivatives, formally

For a function of a single real variable $f(x)$, the **derivative** $f'(x)$ at a point is the slope of the tangent line to the graph at that point. Equivalently, it is the limit of the average rate of change over a step $h$ as the step shrinks to zero:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.$$

Intuitively, $f'(x)$ answers the question: "if $x$ is nudged by a very small amount, by how much does $f$ change, per unit of nudge?" The answer is a number: the local slope.

The limit can be carried out by hand for a simple function, and doing it once shows where the rules below come from. For $f(x) = x^2$ the quotient is

$$\frac{(x + h)^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h,$$

and as $h$ shrinks to zero the leftover $h$ vanishes, leaving $f'(x) = 2x$. The figure at the top of the post draws this result: on the curve $x^2$ the tangent at $x = -1.5$ has slope $-3$, the one at $x = 0.5$ has slope $1$, and the one at $x = 2$ has slope $4$.

### 2.1. The power rule

The same expansion works for any whole-number power, and the result is the **power rule**. For any monomial $a x^n$:

$$\frac{d}{dx}\bigl[a x^n\bigr] = n \cdot a \cdot x^{n-1}.$$

Together with the sum rule of section 2.2, this one rule differentiates every polynomial.

| $f(x)$ | $f'(x)$ | Note |
|:---:|:---:|:---|
| $c$ (constant) | $0$ | Flat function; slope is zero everywhere |
| $x$ | $1$ | Constant slope of one |
| $x^2$ | $2x$ | Slope grows linearly with $x$ |
| $2x^2$ | $4x$ | Same shape, scaled twice as steep |
| $x^3$ | $3x^2$ | Slope grows quadratically |
| $5x^4$ | $20x^3$ | Multiply by the exponent, drop it by one |

### 2.2. The sum rule

The derivative distributes over addition:

$$\frac{d}{dx}\bigl[f(x) + g(x)\bigr] = f'(x) + g'(x).$$

So differentiating a polynomial means differentiating each term and summing the results.

**Example.** $f(x) = x^3 + 2x^2 + 6$:

$$f'(x) = 3x^2 + 4x + 0 = 3x^2 + 4x.$$

The constant $6$ vanishes because its derivative is zero; everything else uses the power rule term by term.

### 2.3. Intuition: slope as sensitivity

The numerical value of $f'(x)$ at a point has three readings, all useful for neural networks:

- **Large positive slope.** A small increase in $x$ causes a large increase in $f$. Sensitivity is high; the input matters.
- **Near-zero slope.** A small change in $x$ causes almost no change in $f$. Sensitivity is low; the input barely matters near this point.
- **Negative slope.** A small increase in $x$ causes a *decrease* in $f$. The input still matters, but the direction is reversed.

The figure below applies the three readings to a single weight, with the loss $L = (w - 1)^2$ of section 4.3 and the update rule of section 1.

![The loss L(w) = (w - 1) squared plotted against one weight w, with its tangent at w = 2, w = 1 and w = 0, of slope +2, 0 and -2. Arrows at w = 2 and w = 0 point towards w = 1, where the loss is lowest. Three cards apply the update w gets w minus alpha times dL/dw: at w = 2 it gives 2 minus 2 alpha and w falls, at w = 1 it gives 1 and w stays, and at w = 0 it gives 2 alpha and w rises.](diagrams/02-slope-to-update.svg)

*One update rule at three points. The sign of the derivative decides which way the weight moves, and its size decides how far.*

When $f$ is the loss and $x$ is a weight, all three readings turn into update decisions: increase a weight whose $\frac{\partial L}{\partial w}$ is large and negative (the loss drops), decrease one whose derivative is large and positive, and leave a weight whose derivative is near zero almost where it is for now.

### 2.4. What a derivative is *not*

A boundary section, because the concept gets misused often.

- **A derivative is not a delta.** It is a *rate*, not a difference. The actual change in $f$ for a finite step $h$ is approximately $f'(x) \cdot h$, not $f'(x)$ itself, and the approximation improves as the step shrinks. For $f(x) = x^3$ at $x = 2$, where the slope is 12, a step of 0.1 changes $f$ by 1.261 against a predicted 1.2, and a step of 0.01 changes it by 0.120601 against a predicted 0.12.
- **A derivative is not always defined.** Functions with corners (like ReLU at $z = 0$) are non-differentiable at the corner. Practical implementations pick one side and move on (section 7).
- **A derivative is not a function value.** $f(x)$ is what the function equals at $x$; $f'(x)$ is how steeply it changes there. The two coincide for the special case $f(x) = e^x$.
- **A derivative is not a finite difference.** A finite difference stops the limit at a small but non-zero $h$, so it is an estimate that carries an error. Section 2.5 defines it. It is the right tool for checking a derivative and the wrong one for computing gradients during training (section 7).

### 2.5. Measuring a derivative: finite differences

The limit that defines $f'(x)$ cannot be run on a computer, but stopping it early can. Pick a small step $h$, evaluate $f$ at two nearby points, and take the slope of the straight line through them. There are two standard ways to choose the points.

The **forward difference** uses $x$ and $x + h$, which is exactly the quotient inside the limit:

$$D_{\text{fwd}}(h) = \frac{f(x + h) - f(x)}{h}.$$

The **central difference** uses one point on each side of $x$, at $x - h$ and $x + h$, which are $2h$ apart:

$$D_{\text{cen}}(h) = \frac{f(x + h) - f(x - h)}{2h}.$$

Both tend to $f'(x)$ as $h$ shrinks, but not equally fast. For $f(x) = x^3$ the two can be worked out exactly, using $(x \pm h)^3 = x^3 \pm 3x^2 h + 3x h^2 \pm h^3$:

$$D_{\text{fwd}}(h) = 3x^2 + 3xh + h^2, \qquad D_{\text{cen}}(h) = 3x^2 + h^2.$$

The exact derivative is $3x^2$, so everything after it is error. The forward difference is off by $3xh + h^2$, which shrinks in proportion to $h$: a step ten times smaller gives an error about ten times smaller. In the central difference the terms $x^3$ and $3xh^2$ are the same on both sides and cancel in the subtraction, the numerator is $6x^2 h + 2h^3$, and the error is $h^2$: a step ten times smaller gives an error a hundred times smaller. At $x = 2$ with $h = 10^{-3}$ the exact derivative is 12, the forward difference is $12 + 0.006 + 0.000001 = 12.006001$, and the central difference is $12.000001$.

The error that comes from stopping the limit early is called the **truncation error**. It is of order $h$ for the forward difference and of order $h^2$ for the central difference, and that holds for any smooth function, not only for $x^3$. The central difference is therefore the default choice, at the price of one extra function evaluation when $f(x)$ is already known. The algebra also suggests that a smaller step is always better. Section 6.1 measures both claims, and the second turns out to be false on a computer.

---

## 3. Partial derivatives

A real neural network has many parameters. The loss is a function of all of them at once; for the 21-parameter network of post 09,

$$L = f(w_1, w_2, \dots, w_{21}).$$

A **partial derivative** measures the sensitivity of $f$ to one of its inputs while every other input is held fixed. The notation uses $\partial$ instead of $d$ to flag this:

$$\frac{\partial f}{\partial w_k} = \text{slope of } f \text{ along the } w_k \text{ axis, with all other } w_j \text{ frozen.}$$

As a limit, it is the definition of section 2 with the step $h$ added to one variable only:

$$\frac{\partial f}{\partial w_k} = \lim_{h \to 0} \frac{f(w_1, \dots, w_k + h, \dots, w_{21}) - f(w_1, \dots, w_k, \dots, w_{21})}{h}.$$

The mechanical rule for computing one follows directly. To compute $\frac{\partial f}{\partial x}$ of a function $f(x, y, z)$, **treat $y$ and $z$ as constants** and differentiate with respect to $x$ by the rules of section 2.

### 3.1. Worked examples

**Example 1.** $f(x, y) = 2x + 3y^2$.

$$\frac{\partial f}{\partial x} = 2, \qquad \frac{\partial f}{\partial y} = 6y.$$

When $x$ is the variable, the $3y^2$ term is a constant and contributes zero. When $y$ is the variable, the $2x$ term is a constant and contributes zero. A frozen variable standing in a term of its own behaves like the constant $6$ of section 2.2: its derivative is zero.

**Example 2.** $f(x, y, z) = 3x^3 z - y^2 + 5z + 2yz$.

$$\frac{\partial f}{\partial x} = 9x^2 z.$$

Only the $3x^3 z$ term involves $x$; everything else is a constant. Inside that term the frozen $z$ is a constant multiplier, like the 3 in front of it, so it stays.

$$\frac{\partial f}{\partial y} = -2y + 2z.$$

The $-y^2$ term gives $-2y$; the $2yz$ term gives $2z$ (since $z$ is held constant); the other terms have no $y$ and contribute zero.

$$\frac{\partial f}{\partial z} = 3x^3 + 5 + 2y.$$

Three terms involve $z$; differentiate each with respect to $z$ and sum.

**Example 3: the ReLU derivative.** ReLU is a function of one variable, $\text{ReLU}(z) = \max(0, z)$, applied separately to every output of a layer, so each of those outputs has its own slope:

$$\frac{d}{dz} \text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

For $z > 0$ the function is $z$ itself, with slope 1 by the power rule; for $z < 0$ it is the constant 0, with slope 0. This is the exact formula backpropagation will use for the ReLU layer in [post 17](../17-backpropagation-through-activation-functions/index.md). At $z = 0$ the two sides disagree and the derivative is undefined; by convention the series' code, like the major frameworks, uses 0 there (section 7). The zero slope over the whole negative side is what makes the dead neuron of post 06 possible: a neuron whose input is negative for every sample passes back no gradient, so its weights stop updating.

### 3.2. Why the variable-freezing trick works

The trick has a geometric reading. A function of two variables defines a surface over the plane of its inputs, and with more variables the picture is the same in more dimensions. Walking along the $x$ axis means moving while every other coordinate stays fixed. Along that walk the other variables never change, so they behave exactly like constants, and the slope met on the walk is an ordinary one-variable derivative.

Frozen does not mean irrelevant. The other variables are held at their current values, and those values still appear in the answer. In example 2, $\partial f / \partial x = 9x^2 z$ at $x = 1$ is 27 when $z = 3$, zero when $z = 0$, and $-27$ when $z = -3$: starting the walk from a different point of the surface changes the slope met along the $x$ direction.

This is also why neural-network calculus is friendly. Each of the network's 21 parameters poses its own one-variable question, with the other 20 frozen at their current values, and each question is answered with the rules of section 2. Backpropagation (posts 12 to 21) answers all 21 in a single backward pass by sharing the work the questions have in common.

---

## 4. The gradient

The **gradient** of $f$ at a point is the vector of all its partial derivatives:

$$\nabla f = \left( \frac{\partial f}{\partial w_1},\ \frac{\partial f}{\partial w_2},\ \dots,\ \frac{\partial f}{\partial w_n} \right).$$

For a function of $n$ variables, the gradient is a vector of $n$ numbers, one per variable. It is the natural object to manipulate in code: NumPy, PyTorch, and JAX all store a gradient as an array of the same shape as the parameters it belongs to.

For example 2 of section 3.1 the gradient is $\nabla f = (9x^2 z,\ -2y + 2z,\ 3x^3 + 5 + 2y)$, and at the point $(x, y, z) = (1, 2, 3)$ it is the three numbers $(27, 2, 12)$. The figure below finds each component as the slope of $f$ along one axis through that point and assembles the three.

![Three panels slice f(x, y, z) = 3 x cubed z minus y squared plus 5z plus 2yz through the point (1, 2, 3), where f = 32, each along one axis with the other two variables frozen. Along x the slice is 9 x cubed plus 23, with slope 9 x squared z = 27 at x = 1; along y it is minus y squared plus 6y plus 24, slope minus 2y plus 2z = 2; along z it is 12z minus 4, slope 3 x cubed plus 5 plus 2y = 12. Arrows carry the three slopes into the gradient (27, 2, 12), whose central differences are 27.00000000, 2.00000000 and 12.00000000 and whose length is the square root of 877, about 29.61.](diagrams/03-partials-to-gradient.svg)

*One partial derivative per variable, each the slope of $f$ along one axis with the other two frozen. The gradient bundles the three into a single vector that points uphill.*

### 4.1. Two properties that matter for training

**The gradient points in the direction of steepest ascent.** Moving in parameter space (the space whose axes are the parameters $w_1, \dots, w_n$) along $\nabla f$ increases $f$ as fast as possible per unit step. The reason is that the rate of change of $f$ per unit distance in the direction of a unit vector $\mathbf{u}$ is the dot product $\nabla f \cdot \mathbf{u}$, called the directional derivative. By the geometric form of the dot product (post 02, section 2) that product equals $\|\nabla f\| \cos\phi$, where $\|\nabla f\|$ is the length of the gradient and $\phi$ is the angle between the gradient and $\mathbf{u}$. The cosine is largest, 1, when the step lines up with $\nabla f$; it is zero at right angles to the gradient and $-1$ in the opposite direction. Therefore moving along $-\nabla f$ decreases $f$ as fast as possible per unit step, which is the geometric reason gradient descent uses the *negative* gradient.

Section 6.3 measures this for example 2 at $(1, 2, 3)$. The gradient $(27, 2, 12)$ has length $\sqrt{877} \approx 29.61$; $f$ rises at 29.61 per unit distance along it, falls at 29.61 against it, does not change at right angles to it, and not one of 10,000 random directions climbs faster.

**The magnitude of the gradient encodes the local steepness.** $\|\nabla f\|$ is small where $f$ is nearly flat and large where it is steep. When the magnitude becomes very small, the optimiser is near a point where the surface is level (a minimum, a maximum, or a saddle point: a spot that slopes up in some directions and down in others) and the steps it takes are small.

For neural networks, this is everything. In the update of section 1,

$$\theta \leftarrow \theta - \alpha \, \nabla L,$$

$\theta$ is the vector of all the parameters and $\nabla L$ is the vector, of the same shape, of all the partial derivatives of the loss.

### 4.2. Reading a gradient component-by-component

For the 21-parameter network of post 09 (6 weights and 3 biases in the first layer, 9 weights and 3 biases in the second), the loss gradient is a vector of 21 numbers. Component $k$ is the partial derivative of the loss with respect to one specific weight or bias:

$$\nabla L = \left( \frac{\partial L}{\partial w_1},\ \frac{\partial L}{\partial w_2},\ \dots,\ \frac{\partial L}{\partial w_{21}} \right).$$

Each component answers a single, local question:

- **Large positive component** ($\partial L / \partial w \gg 0$). Increasing this weight would increase the loss. Therefore *decrease* it. The optimiser does this automatically via the minus sign.
- **Large negative component** ($\partial L / \partial w \ll 0$). Increasing this weight would decrease the loss. The optimiser, via the minus sign, increases it.
- **Near-zero component** ($\partial L / \partial w \approx 0$). This weight barely affects the loss right now. The optimiser leaves it nearly unchanged.

The size of a component is a sensitivity, and example 2 shows it in numbers. At $(1, 2, 3)$ a nudge of $+0.01$ in $x$ alone, whose component is 27, changes $f$ by 0.2727. The same nudge in $y$, whose component is 2, changes $f$ by 0.0199, and in $z$, whose component is 12, by 0.1200. Each change is close to the component times the nudge (0.27, 0.02, and 0.12), which is the rate-not-delta reading of section 2.4 applied to one axis at a time.

The whole training loop is the repeated application of those three rules across every parameter at once.

### 4.3. What the gradient is *not*

A boundary section.

- **It is not the function value.** Knowing $\nabla L$ does not reveal what $L$ is, only how it would change for small parameter moves.
- **It is not unique to a parameterisation.** Writing the same loss in terms of a new parameter $u$, with $w = 2u$, makes the derivative with respect to $u$ twice the derivative with respect to $w$: for $L = (w - 1)^2$ at $w = 2$ the first is 4 and the second is 2. The chain rule of post 11 explains the factor. The optimiser's behaviour follows whichever parameterisation is used.
- **It is not the direction to the global minimum.** It is the direction of *local* steepest descent, and it need not even point at the nearest minimum. The bowl $f(x, y) = x^2 + 10y^2$ has its only minimum at the origin, yet at $(1, 1)$ the negative gradient is $(-2, -20)$, while the origin lies in the direction $(-1, -1)$. Global minimisation is a separate problem; gradient descent walks downhill into whichever minimum its path leads to.
- **It is not free in cost.** Computing $\nabla L$ for a deep network takes a backward pass that costs about as much as a forward pass, up to a small constant factor. Posts 12 to 21 derive that efficient algorithm, backpropagation. The naive alternative, a finite difference for every parameter, has a cost that scales with the number of parameters (section 7).

---

## 5. Connecting calculus to the network

The pieces line up cleanly:

| Calculus object | What it computes for the network | Where it lives in code |
|---|---|---|
| Derivative $f'(x)$ | Per-element activation slope (for ReLU, 0 or 1) | `Activation_ReLU.backward` (post 17) |
| Partial derivative $\partial L / \partial w$ | Sensitivity of the loss to one weight | one entry of a layer's `dweights` array (from post 16 on) |
| Gradient $\nabla L$ | The full vector of partial derivatives | every layer's `dweights` and `dbiases`, taken together |

The chain rule, introduced in [post 11](../11-the-chain-rule/index.md), is what lets the partial derivatives of layered functions be computed by composing the partial derivatives of the pieces. Without the chain rule, computing the partial derivative of the loss with respect to a single first-layer weight would require expanding the entire forward pass into one giant expression and differentiating it. With the chain rule, the same number can be computed by multiplying together small per-layer derivatives, which is exactly what backpropagation does.

Working every partial derivative out by hand is useful for a single neuron (post 12) and a single layer (post 13). For a whole stack it is the labour that backpropagation automates, and the algorithm, not the hand derivation, is what the later posts rely on.

---

## 6. Make it run: finite differences against the rules

Nothing later in the series imports the code of this post; it exists to check the calculus above. Three scripts under `snippets/` produce every number the post quotes. Each runs from the series root in under a second and prints the same output on every run:

- `python posts/10-derivatives-partial-derivatives-and-gradients/snippets/finite_differences.py` runs sections 2.4, 2.5, 6.1, and 6.2. It needs only the standard library.
- `python posts/10-derivatives-partial-derivatives-and-gradients/snippets/partials_and_gradient.py` runs sections 3, 4, and 6.3. It needs NumPy and seeds its random directions with `np.random.seed(0)`.
- `python posts/10-derivatives-partial-derivatives-and-gradients/snippets/what_can_go_wrong.py` runs section 7. It needs NumPy.

The two estimates of section 2.5 are one line each:

```python
def forward_difference(f, x, h):
    """Slope of the straight line through (x, f(x)) and (x + h, f(x + h))."""
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
    """Slope of the straight line through (x - h, f(x - h)) and (x + h, f(x + h))."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

### 6.1. Forward against central, and the sweep over the step size

For $f(x) = x^3$ at $x = 2$ with $h = 0.001$ the first script prints both estimates next to the algebra of section 2.5:

```text
== Section 2.5: forward against central at h = 0.001
forward difference  12.006001   error 6.001e-03   algebra: 3xh + h^2 = 6.001e-03
central difference  12.000001   error 1.000e-06   algebra: h^2       = 1.000e-06
```

With the same step, the central difference is about 6,000 times closer to the exact 12. If a smaller step always helped, the cure for any remaining error would be to keep shrinking $h$. The script tries it, from $h = 10^{-1}$ down to $h = 10^{-16}$:

```text
== Section 6.1: the sweep over h in float64 (error = distance from the exact 12)
    h            forward     error            central     error
1e-01    12.610000000000   6.1e-01    12.010000000000   1.0e-02
1e-02    12.060100000000   6.0e-02    12.000100000000   1.0e-04
1e-03    12.006001000000   6.0e-03    12.000000999999   1.0e-06
1e-04    12.000600010023   6.0e-04    12.000000010013   1.0e-08
1e-05    12.000060000261   6.0e-05    12.000000000212   2.1e-10
1e-06    12.000006002211   6.0e-06    12.000000000345   3.5e-10
1e-07    12.000000584322   5.8e-07    11.999999993684   6.3e-09
1e-08    11.999999927070   7.3e-08    11.999999927070   7.3e-08
1e-09    12.000000992884   9.9e-07    12.000000992884   9.9e-07
1e-10    12.000000992884   9.9e-07    12.000000992884   9.9e-07
1e-11    12.000000992884   9.9e-07    12.000000992884   9.9e-07
1e-12    12.001066806988   1.1e-03    12.001066806988   1.1e-03
1e-13    11.990408665952   9.6e-03    11.990408665952   9.6e-03
1e-14    12.256862191862   2.6e-01    12.123635428907   1.2e-01
1e-15    10.658141036402   1.3e+00    11.990408665952   9.6e-03
1e-16     0.000000000000   1.2e+01     0.000000000000   1.2e+01
smallest forward error on this grid: 7.3e-08 at h = 1e-08
smallest central error on this grid: 2.1e-10 at h = 1e-05
the smallest forward error is 344 times the smallest central error
```

Read the two error columns from the top. Down to about $h = 10^{-7}$ the forward error is $6h$, as the algebra says for $x = 2$, and down to $h = 10^{-4}$ the central error is $h^2$. Then both columns turn round and grow. From $h = 10^{-8}$ to $h = 10^{-13}$ the two estimates print the same digits, so the central difference has lost its advantage, and at $h = 10^{-16}$ both report a derivative of exactly 0. The figure below draws both error columns on logarithmic axes, with $h$ growing from left to right.

![A log-log chart of the error of the forward and the central difference of x cubed at x = 2 against the step h, from 10 to the -16 up to 10 to the -1, in float64. From the right, the forward error falls as 6h and the central error as h squared; then both turn and rise along a dotted line, the rounding estimate 10 to the -15 over h. The best central error is 2.1 times 10 to the -10, at h = 10 to the -5, and the best forward error 7.3 times 10 to the -8, at h = 10 to the -8. At h = 10 to the -16 both estimates return 0, an error of 12.](diagrams/04-step-size-sweep.svg)

*Right of each minimum the error is truncation, $6h$ or $h^2$; left of it, rounding, about $10^{-15}/h$. The central difference bottoms out at $2.1 \times 10^{-10}$ at $h = 10^{-5}$.*

The cause is **rounding error**. A float64 number carries about 16 significant decimal digits; its machine epsilon, the relative gap between neighbouring numbers, is $2.2 \times 10^{-16}$. Near $f(2) = 8$ neighbouring float64 numbers are $1.8 \times 10^{-15}$ apart, so each computed value of $f$ can be wrong by about $10^{-15}$. The numerator of either estimate subtracts two almost equal values of $f$. Their leading digits cancel, and that $10^{-15}$ uncertainty is what remains to be divided by a tiny $h$. The rounding error of the estimate is therefore about $10^{-15} / h$, and it *grows* as $h$ shrinks: about $10^{-6}$ at $h = 10^{-9}$ and $10^{-3}$ at $h = 10^{-12}$, as the table shows. At $h = 10^{-16}$ the step is smaller than the gap between 2 and its float64 neighbours, so `2.0 + 1e-16 == 2.0` is `True`, the two function values are identical, and their difference is zero.

The total error therefore has two parts that pull in opposite directions: truncation error, which falls as $h$ shrinks, and rounding error, which rises. The best step is where they meet. For the central difference, $h^2 = 10^{-15} / h$ gives $h = 10^{-5}$; for the forward difference, $6h = 10^{-15} / h$ gives $h \approx 1.3 \times 10^{-8}$. The measured minima sit at those powers of ten: $2.1 \times 10^{-10}$ at $h = 10^{-5}$ for the central difference and $7.3 \times 10^{-8}$ at $h = 10^{-8}$ for the forward one. The central difference is more accurate at a given large step, and its best achievable error is 344 times smaller as well.

The working rule for float64 arithmetic, with inputs and outputs of ordinary size, is a central difference with $h$ near $10^{-5}$. That is the step the gradient check of post 21 uses.

### 6.2. The rules, checked

The same central difference, with $h = 10^{-5}$, checks every row of the power-rule table of section 2.1, the sum-rule example of section 2.2, and three more functions:

```text
every row at x = 1.5
f(x)               rule          central       gap
7 (a constant)     0.00      0.000000000   0.0e+00
x                  1.00      1.000000000   6.6e-12
x^2                3.00      3.000000000   2.0e-11
2x^2               6.00      6.000000000   3.9e-11
x^3                6.75      6.750000000   1.4e-10
5x^4              67.50     67.500000003   3.3e-09
x^3 + 2x^2 + 6    12.75     12.750000000   2.3e-10
5x^3 + 2x         35.75     35.750000001   5.1e-10
(2x + 1)^2        16.00     16.000000000   1.0e-10
largest gap between a rule and its central difference: 3.3e-09
```

Every rule agrees with its measurement to better than $10^{-8}$. The last row is the one a hand calculation most often gets wrong. Expanded, $(2x + 1)^2 = 4x^2 + 4x + 1$ and its derivative is $8x + 4$, which is 16 at $x = 1.5$. Applying the power rule to the bracket as if it were a bare $x$ gives $2(2x + 1) = 8$, half the measured slope; the missing factor of 2 is what the chain rule of post 11 supplies.

### 6.3. Partial derivatives and the gradient, checked

A partial derivative is measured the same way, with the step applied to one coordinate while the others stay where they are. That is the freezing rule of section 3 written as code:

```python
def numerical_gradient(f, point, h=1e-5):
    """Central difference along each axis in turn, the other coordinates held fixed."""
    point = np.asarray(point, dtype=np.float64)
    grad = np.zeros_like(point)
    for k in range(point.size):
        step = np.zeros_like(point)
        step[k] = h                    # move along axis k only
        grad[k] = (f(point + step) - f(point - step)) / (2 * h)
    return grad
```

For example 2 of section 3.1 the three partial derivatives found by freezing agree with the three measured ones:

```text
== Section 3.1, example 2: f(x, y, z) = 3x^3 z - y^2 + 5z + 2yz at (x, y, z) = (1, 2, 3)
f(1, 2, 3) = 32
df/dx = 9x^2 z         freezing rule  27.0   central difference  27.00000000
df/dy = -2y + 2z       freezing rule   2.0   central difference   2.00000000
df/dz = 3x^3 + 5 + 2y  freezing rule  12.0   central difference  12.00000000
largest gap between the two columns: 1.1e-09
```

The script then measures the rate of change of $f$ per unit distance from $(1, 2, 3)$ in several directions, which is the first property of section 4.1:

```text
== Section 4.1: the gradient at (1, 2, 3), and the rate of change of f in several directions
gradient = [27.  2. 12.], length = sqrt(27^2 + 2^2 + 12^2) = sqrt(877) = 29.614186
along the gradient      29.614186
against the gradient   -29.614186
across the gradient      0.000000
along the x axis        27.000000
10,000 random unit directions: steepest 29.613710, most downhill -29.608534
random directions steeper than the gradient direction: 0
random directions that go uphill: 4973 of 10000
```

No direction beats the gradient, and the steepest of the 10,000 random directions only approaches it. About half of the random directions go uphill, 4,973 of 10,000 here. That is the direction-blindness that defeated random perturbation in post 09: a random step is as likely to raise the loss as to lower it, while the gradient names the best direction without trying any.

Finally, one step of size $\alpha = 0.001$ against the gradient takes $f$ from 32 down to 31.1454, and the same step along the gradient takes it up to 32.8999. The first-order prediction for both, from the rate-not-delta reading of section 2.4, is a change of $\alpha \|\nabla f\|^2 = 0.877$.

---

## 7. What can go wrong?

`snippets/what_can_go_wrong.py` reproduces four ways in which a derivative, or the estimate of one, misleads. It first repeats the sweep of section 6.1 in float32, the precision that `nnfs.init()` gives the series' arrays (post 04, section 2):

```text
== The sweep over h in float32: f(x) = x^3 at x = 2, exact derivative 12
    h        forward     error        central     error
1e-01     12.6099873   6.1e-01     12.0099945   1.0e-02
1e-02     12.0600700   6.0e-02     12.0000839   8.4e-05
1e-03     12.0048513   4.9e-03     11.9998446   1.6e-04
1e-04     11.9876862   1.2e-02     11.9948387   5.2e-03
1e-05     12.0162964   1.6e-02     12.0162964   1.6e-02
1e-06     11.4440918   5.6e-01     11.4440918   5.6e-01
1e-07      0.0000000   1.2e+01      7.1525574   4.8e+00
1e-08      0.0000000   1.2e+01      0.0000000   1.2e+01
smallest forward error on this grid: 4.9e-03 at h = 1e-03
smallest central error on this grid: 8.4e-05 at h = 1e-02
```

It then differentiates ReLU at its corner, compares one loss under two parameterisations, and counts the function evaluations of a numerical gradient:

```text
== ReLU at its corner z = 0, h = 1e-5
forward difference   1.0   (the slope on the right)
backward difference  0.0   (the slope on the left)
central difference   0.5   (neither)
away from the corner: central difference at z = 0.3 is 1.0, at z = -0.3 is 0.0

== One loss, two parameterisations: L = (w - 1)^2, and the same loss written with w = 2u
dL/dw at w = 2: 2.000000
dL/du at u = 1: 4.000000   (the same point, since w = 2u = 2)

== The cost of a numerical gradient: function evaluations for n parameters
central-difference gradient of a 21-parameter function: 42 evaluations of the function
```

- **The step is too small, or the arithmetic is float32.** Section 6.1 showed the float64 case: below $h = 10^{-5}$ a smaller step makes the central difference worse. A float32 number keeps only about 7 significant digits (machine epsilon $1.2 \times 10^{-7}$), and the same sweep breaks down far sooner. The step $h = 10^{-5}$ that left an error of $2.1 \times 10^{-10}$ in float64 leaves $1.6 \times 10^{-2}$ here, the best central error on the grid is $8.4 \times 10^{-5}$ at $h = 10^{-2}$, and at $h = 10^{-7}$ the forward difference returns exactly 0 because `2.0 + 1e-7 == 2.0` in float32. A numerical check of a derivative is therefore run in float64, in a process that has not called `nnfs.init()`: after that call `np.dot` returns float32 even for float64 copies of the arrays (post 21, section 11).
- **The function has a corner.** At $z = 0$ the slope of ReLU is 1 on the right and 0 on the left, and strictly the derivative does not exist. Code has to return something, and `Activation_ReLU.backward` (post 17) returns 0 at exactly 0. A central difference that straddles the corner returns 0.5, which is neither slope, so a numerical check that lands within $h$ of a corner disagrees with a correct backward pass. The absolute value in the L1 penalty of post 30 has the same kind of corner at $w = 0$. Subgradient methods, which extend gradient descent to functions with many corners, are out of scope for this series.
- **The gradient is compared across parameterisations.** The same loss at the same point has slope 2 with respect to $w$ and slope 4 with respect to $u = w/2$. A gradient is a statement about one particular set of variables, and rescaling a parameter rescales its component (section 4.3).
- **A numerical gradient is used for training.** It needs two evaluations of the loss for every parameter: 42 forward passes for the 21 parameters of the post 09 network, and 2,000,000 for a network with a million parameters, all for a single update. Backpropagation delivers every partial derivative for about the price of one more pass. Finite differences are for checking a backward pass, not for replacing it.
- **A rule this post does not cover is needed.** The power and sum rules handle polynomials and nothing else. Every further derivative the series needs is derived in the post that uses it: the chain rule for nested functions in post 11, the slopes of sigmoid and tanh in post 17, the derivative of $-\log x$ (which is $-1/x$) in post 18, and the exponentials inside softmax in post 19. None of them has to be memorised in advance.
- **A derivative is confused with a differential.** The two are closely related. The derivative $f'(x)$ is the slope; the differential $dy = f'(x)\,dx$ uses that slope to estimate the change in $y$ for a small change $dx$ in $x$. This is the rate-not-delta distinction of section 2.4 under another name.
- **The gradient is expected to be a row, or a column.** Textbooks differ on whether a gradient is written as a row vector or as a column vector, and the choice is pure convention. The series sidesteps it: every gradient array has the shape of the parameter array it belongs to (`dweights` has the shape of `weights`), and the update needs only an element-wise multiplication and a subtraction.

---

## 8. Summary

| Concept | Takeaway |
|---|---|
| Derivative | Local slope of a single-variable function; the rate at which $f$ changes per unit change in $x$ |
| Power rule | $\frac{d}{dx} a x^n = n a x^{n-1}$ |
| Sum rule | The derivative of a sum is the sum of derivatives |
| Finite difference | Forward, $(f(x + h) - f(x))/h$, error of order $h$; central, $(f(x + h) - f(x - h))/(2h)$, error of order $h^2$ |
| Step size | In float64 the central difference is best near $h = 10^{-5}$; smaller steps lose digits to rounding |
| Partial derivative | Slope along one axis of a multi-variable function; freeze the others, then differentiate |
| ReLU derivative | $1$ if the input is positive, $0$ otherwise |
| Gradient $\nabla f$ | Vector of all partial derivatives; points uphill; its length says how steep |
| Why this matters | Gradient descent uses $-\nabla L$ as the per-step direction in parameter space |

---

## Common pitfalls

1. **Forgetting the chain rule when a variable is buried inside.** $\frac{d}{dx}(2x + 1)^2$ is not $2 \cdot (2x + 1)$; it is $8x + 4$, twice as large (section 6.2). Post 11 covers this. Until then, the power rule applies only when the thing raised to a power is the variable itself.
2. **Confusing $f(x)$ with $f'(x)$.** They take the same input and answer different questions: $f(x)$ is the height of the curve at $x$, and $f'(x)$ is its slope there. A loss can be large where its slope is zero and small where its slope is steep.
3. **Mishandling a frozen variable in a partial derivative.** A term with no trace of the live variable contributes zero: $3y^2$ adds nothing to $\partial f / \partial x$. A frozen variable that *multiplies* the live one stays as a constant factor: $\partial (2yz) / \partial y$ is $2z$, not 0 and not 2.
4. **Treating the gradient as a scalar.** It is a vector with one component per parameter. The learning rate is a single number that multiplies every component, and the subtraction is then component by component, so each weight moves by its own amount.
5. **Mixing up gradient ascent and descent.** $+\nabla L$ goes uphill (increases the loss); $-\nabla L$ goes downhill (decreases the loss). The optimiser always subtracts. A loss that rises on every iteration is the usual symptom of a flipped sign.
6. **Making $h$ as small as possible in a finite difference.** Below about $10^{-5}$ in float64 a central difference gets worse, not better, and at $h = 10^{-16}$ it returns 0 (section 6.1). In float32 the trouble starts near $10^{-2}$ (section 7).

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 4.3 (Gradient-Based Optimization) for the derivative, the partial derivative, and the gradient as the direction of steepest ascent, and section 11.5 (Debugging Strategies) for comparing a backward pass with finite differences (MIT Press, 2016).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapters 7 and 8 (2020).
- Leibniz, G. W., *"Nova Methodus pro Maximis et Minimis"* (Acta Eruditorum, 1684).
- Newton, I., *Method of Fluxions* (1671; published 1736).
- Strang, G., *Calculus*, chapters 1 to 4 for derivatives and chapter 13 for partial derivatives and the gradient (Wellesley-Cambridge, 2010).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 11 - The chain rule](../11-the-chain-rule/index.md):** the rule that composes derivatives through a stack of functions, and the engine behind backpropagation.
- **[Post 21 - Coding the full backpropagation](../21-coding-the-full-backpropagation/index.md):** where the central difference of section 2.5 returns as the gradient check that tests a complete backward pass.
