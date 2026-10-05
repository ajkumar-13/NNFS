# 11 - The chain rule

> **TL;DR.** A network's loss is a composition of functions, so the derivative of the loss with respect to a first-layer weight is buried several functions deep. The **chain rule** turns that derivative into a product of **local derivatives**, one per function, each evaluated at the value its function received: for $y = f(g(x))$, $dy/dx = f'(g(x)) \cdot g'(x)$. In the two-layer classifier the gradient has four factors for a first-layer weight and two for a second-layer weight. Measured with central differences on a small network, the product of the four factors agrees with the slope taken through the whole network to within $6.0 \times 10^{-11}$.
>
> **Prerequisites:** [Post 10](../10-derivatives-partial-derivatives-and-gradients/index.md).
> **Safe to skip?** Skip it if the reader can already differentiate $(3x + 2)^4$ without expanding it, say at which value each factor is evaluated, and write the gradient of a two-layer network's loss with respect to a first-layer weight as a product of one derivative per function.
>
> **After reading, you will be able to:**
>
> - State the chain rule in one sentence, including the point at which each local derivative is evaluated.
> - Apply the chain rule to a polynomial composition without expanding it.
> - Identify the local derivative of each function in a small neural-network composition.
> - Predict how many local-derivative factors appear in the gradient for a weight in any given layer.
> - Check a chain-rule result against a central difference taken through the whole composition.

![Two rows. The forward row runs left to right: x enters the inner function g, giving z = g(x), which enters the outer function f, giving y = f(g(x)). The backward row runs right to left with the slopes dy/dz and dz/dx. A formula reads dy/dx = dy/dz times dz/dx = f'(z) times g'(x).](diagrams/01-chain-rule.svg)

*Values are computed left to right, slopes right to left. Each box contributes one local derivative, and the slope of the whole chain is their product.*

---

## 1. The question: how is a derivative buried several functions deep computed?

The forward pass of a neural network is a **composition of functions**: the output of one function is the input of the next. In the classifier of [post 07](../07-coding-the-complete-forward-pass/index.md) one sample $\mathbf{x}$ is multiplied by $\mathbf{W}_1$, has a bias added, passes through ReLU, is multiplied by $\mathbf{W}_2$, has another bias added, passes through softmax, and finally enters the cross-entropy of post 08 with the true label $y$ to produce the loss. Written as one line, with CE for cross-entropy:

$$L = \text{CE}\Bigl(\text{softmax}\bigl(\text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2\bigr),\ y\Bigr).$$

To update $\mathbf{W}_1$, gradient descent needs $\partial L / \partial \mathbf{W}_1$: how the loss at the outside of that expression changes when an entry of $\mathbf{W}_1$, at the very inside, is nudged. The power and sum rules of post 10 differentiate a polynomial in the variable itself and do not reach inside a nested expression. The missing piece is a rule for functions inside functions.

That rule is the **chain rule**. Each function in the composition is differentiated on its own, as if nothing else existed, and the results are multiplied. The derivative of one function with respect to its own input is called a **local derivative**, and the gradient of the loss with respect to any parameter is a product of local derivatives, one per function between the parameter and the loss.

The rule is as old as calculus; Leibniz was using it in the 1670s. What neural networks needed was an economical order in which to apply it, the reverse order described by Linnainmaa (1970) and Werbos (1974), which Rumelhart, Hinton, and Williams (1986) popularised as the way to train layered networks under the name **backpropagation**. Posts 12 to 21 derive backpropagation step by step; this post explains the one rule it relies on.

---

## 2. The rule, formally

If $y$ depends on an intermediate quantity $z$, and $z$ depends on $x$, then $y$ depends on $x$ through $z$:

$$y = f(g(x)) \quad \text{where} \quad z = g(x).$$

Here $g$ is the **inner function**, applied first, and $f$ is the **outer function**, applied to what $g$ produced. The chain rule says the derivative of $y$ with respect to $x$ is the product of two local derivatives:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx} = f'(z) \cdot g'(x) = f'(g(x)) \cdot g'(x).$$

In one sentence: **the derivative of a composition is the derivative of the outer function, evaluated at the value the inner function produced, times the derivative of the inner function.** The point of evaluation is part of the rule: $g'$ is taken at the original $x$, and $f'$ at $z = g(x)$, the value $f$ actually received.

The reason is the reading of a derivative as a rate from post 10 (section 2.4). A small nudge $\Delta x$ changes $z$ by about $g'(x)\,\Delta x$, and that change is itself a small nudge to the input of $f$:

$$\Delta y \approx f'(z)\,\Delta z \approx f'(z)\,g'(x)\,\Delta x.$$

Dividing by $\Delta x$ gives the rate $f'(z)\,g'(x)$, and both approximations become exact as the nudge shrinks to zero: two rescalings in a row multiply. In Leibniz's notation the rule looks like the cancellation of $dz$ between two fractions. That is a useful way to remember it and to check units, but the argument above is what makes it true; a complete proof also has to treat the case in which the change in $z$ is exactly zero, a point a calculus text raises (Strang, 2010).

### 2.1. A first example, and two debts from post 10

Take $y = (2x + 1)^2$. The inner function is $g(x) = 2x + 1$ and the outer one is $f(z) = z^2$. Their local derivatives are $g'(x) = 2$ and $f'(z) = 2z$, so

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x) = 2(2x + 1) \cdot 2 = 8x + 4.$$

At $x = 1.5$ the inner value is $z = 4$, the two factors are $8$ and $2$, and the slope is $16$, the value post 10 measured with a central difference (section 6.2 there). The tempting answer $2(2x + 1) = 8$, which post 10 found to be half the measured slope, is the outer factor alone: the inner factor $g'(x) = 2$ was left out.

Post 10 (section 4.3) also noted that the loss $L = (w - 1)^2$ has slope 2 with respect to $w$ at $w = 2$, and slope 4 with respect to a new parameter $u$ defined by $w = 2u$. That is a chain with $w = g(u) = 2u$ inside:

$$\frac{dL}{du} = \frac{dL}{dw} \cdot \frac{dw}{du} = 2(w - 1) \cdot 2 = 2 \cdot 2 = 4 \quad \text{at } u = 1.$$

Rescaling a parameter inserts one more function into the chain, and with it one more factor.

### 2.2. What the chain rule is *not*

- **It is not the product rule.** The chain rule is for *composition*, $f$ applied to $g(x)$. A *product* of two functions of the same variable, $f(x) \cdot g(x)$, is differentiated with the product rule, $f'(x)\,g(x) + f(x)\,g'(x)$, which is a sum of two terms and not a product of two slopes. $(2x + 1)^2$ is a composition; $x^2 \cdot (2x + 1)$ is a product.
- **It is not a single-variable theorem.** The same rule holds for the partial derivatives of functions of several variables (section 5.1), and backpropagation uses that form everywhere.

---

## 3. A non-network example: distance, time, and fuel

A driver leaves New York for California at a constant speed and with a constant fuel economy.

| Quantity | Symbol | Relationship |
|---|:---:|---|
| Time (hours) | $x$ | independent variable |
| Distance (km) | $z = 60x$ | the car covers 60 km per hour |
| Fuel (litres) | $y = z / 30$ | the car uses 1 litre per 30 km |

How fast is fuel being used per hour, $dy/dx$? The formula for $y$ does not mention $x$ at all; time acts on fuel only through distance. The chain rule links the two:

$$\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx} = \frac{1}{30} \cdot 60 = 2 \quad \text{litres per hour}.$$

After 3 hours the car has driven 180 km and used 6 litres. The units behave like the symbols, litres per km times km per hour giving litres per hour. Neither local derivative needed to know anything about the other function: the fuel economy is a fact about the engine and the speed is a fact about the trip.

---

## 4. A polynomial example

Compute $\frac{d}{dx}\bigl[3(2x^2)^5\bigr]$, a power applied to a polynomial, in three steps.

**Step 1. Identify inner and outer.** Let $g(x) = 2x^2$ (the inner function) and $f(z) = 3z^5$ (the outer function, applied to $z = g(x)$).

**Step 2. Local derivatives.** Each is a single use of the power rule of post 10:

$$g'(x) = 4x, \qquad f'(z) = 15 z^4.$$

**Step 3. Multiply, with the outer derivative evaluated at the inner value.**

$$\frac{d}{dx}\bigl[3(2x^2)^5\bigr] = f'(g(x)) \cdot g'(x) = 15 (2x^2)^4 \cdot 4x = 15 \cdot 16 x^8 \cdot 4x = 960 x^9.$$

This example is small enough to check by expanding first: $3(2x^2)^5 = 96x^{10}$, whose derivative by the power rule is $960x^9$. At $x = 1$ the inner value is $z = 2$, the factors are $f'(2) = 15 \cdot 16 = 240$ and $g'(1) = 4$, and the slope is 960. Taking the outer derivative at $x$ in place of $z$ would give $f'(1) \cdot g'(1) = 15 \cdot 4 = 60$, wrong by a factor of 16.

---

## 5. The extended chain rule

For three composed functions, $y = f(g(h(x)))$, the rule extends by one more local derivative:

$$\frac{dy}{dx} = f'\bigl(g(h(x))\bigr) \cdot g'\bigl(h(x)\bigr) \cdot h'(x).$$

Every factor is evaluated at the value its own function received on the way in. As an example take $h(x) = 3x + 1$, $g(u) = u^2$, and $f(v) = 2v^3$. At $x = 1$ the values along the chain are $u = 4$ and $v = 16$, and the local derivatives are $h'(1) = 3$, $g'(4) = 8$, and $f'(16) = 6 \cdot 16^2 = 1{,}536$. The slope is their product, $1{,}536 \cdot 8 \cdot 3 = 36{,}864$. Expanding first gives the same number: $y = 2(3x + 1)^6$ and $dy/dx = 36(3x + 1)^5 = 36{,}864$.

For $n$ composed functions there are $n$ factors. Each layer, activation, and loss of a network is one such function, so the depth of the network sets the length of the chain.

### 5.1. The same rule with partial derivatives

The network in the next section uses partial derivatives in place of plain $d$, so one small case first. Suppose a scalar loss $L = u + v$ depends on $x$ through two intermediates, $u = 2x$ and $v = x^2$. Each path from $x$ to $L$ is a chain of its own, read off with $\partial$ replacing $d$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial x} = 1 \cdot 2 + 1 \cdot 2x = 2 + 2x.$$

At $x = 3$ the two paths contribute 2 and 6, and the slope is 8, the same as differentiating $L = 2x + x^2$ directly. The one new ingredient is that a variable which reaches the output along two paths *sums* the contributions of the paths; along each single path the product of local derivatives is unchanged. The sum over paths returns in post 15, where one input feeds several neurons.

---

## 6. The chain rule, applied to a neural network

Take the two-layer classifier of section 1 and one sample $\mathbf{x}$, a row of features. Read from the inside out, four functions stand between the input and the loss:

| Position | Function | Symbol |
|:---:|---|---|
| innermost | first dense layer | $\mathbf{z}_1 = \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1$ |
| next | ReLU activation | $\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$ |
| next | second dense layer | $\mathbf{z}_2 = \mathbf{a}_1\mathbf{W}_2 + \mathbf{b}_2$ |
| outermost | softmax and cross-entropy | $L = \text{CE}(\text{softmax}(\mathbf{z}_2),\ y)$ |

Softmax and cross-entropy are counted as one function from the logits $\mathbf{z}_2$ to the loss, because the series differentiates them together (post 19). The lowercase bold letters are the single-sample rows of the batch matrices $\mathbf{Z}_1$, $\mathbf{A}_1$, and $\mathbf{Z}_2$ of the [notation guide](../../notation_guide.md), and $L$ is a single number.

For $\partial L / \partial \mathbf{W}_1$ the chain rule produces one factor per function in the chain:

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{a}_1} \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{z}_1} \cdot \frac{\partial \mathbf{z}_1}{\partial \mathbf{W}_1}.$$

![Two rows. The forward row runs left to right: x, Dense 1, z1, ReLU, a1, Dense 2, z2, the loss L. The backward row holds one local derivative per function: L in z2 for Softmax + CE, z2 in a1 for Dense 2, a1 in z1 for ReLU, z1 in W1 for Dense 1. A band multiplies the four.](diagrams/02-chain-in-a-network.svg)

*Read the lower row from right to left: the combined softmax and cross-entropy, the second dense layer, ReLU, the first dense layer. Four functions, four factors.*

Because the quantities are rows of numbers and not single numbers, each factor is a table of partial derivatives with one row per output of its function and one column per input (such a table is called a Jacobian, a name post 17 returns to), and the dots are matrix products (post 02). For the network of post 09, with 2 inputs, 3 hidden neurons, and 3 classes, the four factors are:

| Factor | Function it belongs to | Shape | What it holds |
|---|---|:---:|---|
| $\partial L / \partial \mathbf{z}_2$ | softmax and cross-entropy | $(1, 3)$ | how the loss responds to each logit |
| $\partial \mathbf{z}_2 / \partial \mathbf{a}_1$ | second dense layer | $(3, 3)$ | how each logit responds to each hidden activation: the entries of $\mathbf{W}_2$ |
| $\partial \mathbf{a}_1 / \partial \mathbf{z}_1$ | ReLU | $(3, 3)$ | the ReLU slope of each hidden neuron, 1 or 0, on the diagonal |
| $\partial \mathbf{z}_1 / \partial \mathbf{W}_1$ | first dense layer | $(3, 6)$ | how each weighted sum responds to each of the six weights: the inputs $\mathbf{x}$ |

The product has shape $(1, 6)$: one number per entry of $\mathbf{W}_1$, which is what a gradient is (post 10, section 4). Section 8.2 measures all four tables and multiplies them.

Each factor is something one class can supply from its own inputs and outputs: the combined softmax and cross-entropy class, `Layer_Dense`, and `Activation_ReLU` each get a `backward` method for it from post 16 on. A backward pass walks the chain from the loss towards the input, multiplying a running gradient by the local derivative of each function it passes. That walk is **backpropagation**. What this post does not supply is the formula for each local derivative, which posts 12 to 19 derive, one function at a time.

### 6.1. Counting the factors

The number of factors in a gradient is fixed by where the parameter enters the forward pass: **one factor for the function the parameter sits in, and one for every function after it, up to and including the loss.**

| Parameter | Functions from the parameter to the loss | Factors |
|---|---|:---:|
| $\mathbf{W}_1$ or $\mathbf{b}_1$ | dense 1, ReLU, dense 2, softmax with cross-entropy | 4 |
| $\mathbf{W}_2$ or $\mathbf{b}_2$ | dense 2, softmax with cross-entropy | 2 |

The second row written out is

$$\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{W}_2},$$

and a bias has the same count as the weights of its layer, with the last factor taken with respect to the bias. Functions *before* the parameter contribute no factor; they only set the values at which the factors are evaluated.

The count generalises. In a network of $K$ dense layers with an activation after each of the first $K - 1$ and the combined softmax and cross-entropy at the end, a weight in layer $k$ is followed by its own dense layer, then an activation and a dense layer for each of the $K - k$ later layers, then the loss:

$$\text{factors for a weight in layer } k = 2(K - k) + 2.$$

For $K = 2$ this gives 4 and 2, as in the table; for a three-layer network it gives 6, 4, and 2. Counting softmax and cross-entropy as two separate functions, as some texts do, adds one to every count.

### 6.2. Why the product is taken from the loss end

The gradients for $\mathbf{W}_1$ and $\mathbf{W}_2$ begin with the same factor, $\partial L / \partial \mathbf{z}_2$, and so do those for $\mathbf{b}_1$ and $\mathbf{b}_2$. A product that is built up starting from the loss computes that shared part once and hands it on: the running product at any point of the walk is the gradient of the loss with respect to the value at that point, and every parameter further in reuses it. This reuse is what the [glossary](../../GLOSSARY.md) means when it calls backpropagation the chain rule applied in an order that reuses every intermediate product.

The order also decides the size of the intermediate results: starting from the loss the running product is always a single row, because $L$ is one number, while starting from $\mathbf{W}_1$ it is a table with one column per weight. This is why the pass runs backward, and why post 10 could say that backpropagation delivers every partial derivative for about the price of one more pass through the network.

The chain rule therefore does two things for a neural network. It makes the gradient *computable*, although the forward expression is deeply nested, and it makes the gradient *modular*: each class needs to know only its own local derivative, and combining them is the chain rule's job.

---

## 7. The pattern, in three steps

Reading off the chain rule for any composition is mechanical:

![Three cards work the derivative of 3 times (2x squared) to the fifth. Split it: g(x) = 2x squared, f(z) = 3z to the fifth. Differentiate each: g'(x) = 4x, f'(z) = 15z to the fourth. Multiply: 960x to the ninth. A band lists the same steps for a network: read the architecture, derive each local derivative, multiply.](diagrams/03-three-step-pattern.svg)

*The three steps that produce $960x^9$ here also produce the four-factor gradient of the network. Only the middle step requires new work for each kind of layer.*

Step 1, identifying the chain of functions, is reading the architecture. Step 3, multiplying, is matrix arithmetic, whose shapes and order posts 13 and 14 work through. The substance of the posts that follow is **step 2**: deriving the local derivatives of `Layer_Dense`, `Activation_ReLU`, and the combined softmax and cross-entropy in posts 12 to 19, before posts 20 and 21 assemble them into one backward pass. Step 2 is also why a backward pass needs the forward pass first: every local derivative is evaluated at a value the forward pass produced, so those values must have been computed, and kept.

---

## 8. Make it run: the chain rule against finite differences

Nothing later in the series imports the code of this post; it checks the calculus above with the central difference of post 10, at the step $h = 10^{-5}$ recommended there. Two scripts under `snippets/` produce every number the post quotes. Each runs from the series root in about a second, uses no random numbers, and prints the same output on every run:

- `python posts/11-the-chain-rule/snippets/chain_rule_checks.py` runs the scalar examples of sections 2 to 5 and the corner case of section 9. It needs only the standard library.
- `python posts/11-the-chain-rule/snippets/network_chain.py` runs sections 6 and 8.2 and the dropped factor of section 9. It needs NumPy.

### 8.1. Scalar chains

The three steps of section 7 fit in one function. It walks the chain forward, recording each local derivative at the value its function receives, and then multiplies the recorded factors starting from the output end:

```python
def chain_derivative(functions, derivatives, x):
    """Derivative at x of the composition that applies functions[0] first.

    Returns the derivative and the list of local derivatives, innermost first.
    """
    value = x
    factors = []
    for function, derivative in zip(functions, derivatives):
        factors.append(derivative(value))    # local slope, at the value this function receives
        value = function(value)              # then move one function along the chain
    product = 1.0
    for factor in reversed(factors):         # start at the output and multiply back to the input
        product *= factor
    return product, factors
```

For the polynomial of section 4 the script prints the factors, their product, and a central difference of the whole composition:

```text
== Section 4: y = 3(2x^2)^5 at x = 1
g(x) = 2x^2, f(z) = 3z^5
  local derivatives, outermost first: 240 * 4   (2 factors)
  chain rule 960.000000   central difference 960.000001   relative gap 1.2e-09
```

The central difference never sees the inner and outer functions separately. It evaluates the whole expression at two nearby points, and it lands on the product of the local derivatives, which is what makes it an independent check. The script prints the same comparison for every other scalar example of sections 2 to 5.

### 8.2. The four factors of the network, measured

The second script sends one sample, $\mathbf{x} = (1, -2)$ with true class 0, through the 2-3-3 network of section 6 with fixed weights. It uses no formula for any local derivative. Each of the four tables is *measured* on its own function, with one central difference per input of that function, at the value the forward pass delivered there:

```python
def local_derivatives(function, point, h=1e-5):
    """Table of d output_i / d input_j at point, one central difference per input (post 10)."""
    point = np.asarray(point, dtype=np.float64)
    columns = []
    for j in range(point.size):
        step = np.zeros_like(point)
        step[j] = h                              # move input j only
        columns.append((function(point + step) - function(point - step)) / (2 * h))
    return np.stack(columns, axis=1)             # shape (n_outputs, n_inputs)
```

The forward pass and the four measured tables:

```text
== Forward pass, one sample
z1 = x W1 + b1  = [ 1.2000 -1.6000  1.7000]
a1 = ReLU(z1)   = [1.2000 0.0000 1.7000]
z2 = a1 W2 + b2 = [-0.6600  1.3900  0.8400]
softmax(z2)     = [0.0755 0.5863 0.3383]
L               = 2.583967   (-log of the probability of class 0)

== The four local derivatives, each measured on its own function
dL/dz2   shape (1, 3)
[[-0.9245  0.5863  0.3383]]
dz2/da1  shape (3, 3)
[[ 0.3000  0.7000 -0.6000]
 [-0.2000  0.1000  0.9000]
 [ 0.5000 -0.4000  0.2000]]
da1/dz1  shape (3, 3)
[[1.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000]
 [0.0000 0.0000 1.0000]]
dz1/dW1  shape (3, 6)
[[ 1.0000  0.0000  0.0000 -2.0000  0.0000  0.0000]
 [ 0.0000  1.0000  0.0000  0.0000 -2.0000  0.0000]
 [ 0.0000  0.0000  1.0000  0.0000  0.0000 -2.0000]]
```

Each table can be read against section 6. The ReLU table is diagonal, with a 0 for the middle neuron, whose weighted sum is $-1.6$. The second dense layer's table holds the entries of $\mathbf{W}_2$, and the first dense layer's table holds only the inputs 1 and $-2$. The loss factor equals the softmax output with 1 subtracted at the true class, $0.0755 - 1 = -0.9245$, a pattern post 19 derives.

The script then multiplies the four tables in the order of the formula and compares the product with a central difference taken through the whole network, in which one entry of $\mathbf{W}_1$ at a time is moved and the loss is recomputed from scratch:

```text
== Section 8.2: the product of the four against one measurement through the whole network
product of the four factors, reshaped to the shape of W1:
[[-0.2255  0.0000  1.1500]
 [ 0.4510  0.0000 -2.3000]]
central difference through the whole network:
[[-0.2255  0.0000  1.1500]
 [ 0.4510  0.0000 -2.3000]]
largest gap between the two: 6.0e-11
```

The two agree to within $6.0 \times 10^{-11}$, which is the size of the error the central differences themselves carry at this step. The middle column is zero in both: the second hidden neuron is switched off for this sample, its ReLU factor is 0, and no change to its two weights can move the loss. The same check passes for $\mathbf{W}_2$ and $\mathbf{b}_1$, with the factor counts of section 6.1.

Measuring every table by finite differences is far too slow for training (post 10, section 7). It serves here to show that the product of local derivatives is the derivative of the whole, with no formula taken on trust. Posts 13 to 15 replace the measured tables by the three matrix expressions of the notation guide and never build them in full.

---

## 9. What can go wrong?

The scripts also reproduce ways of misapplying the rule. The first leaves the ReLU factor out of the network's product:

```text
== Section 9: one factor dropped (the ReLU factor)
product without da1/dz1, reshaped to the shape of W1:
[[-0.2255 -0.7238  1.1500]
 [ 0.4510  1.4477 -2.3000]]
largest gap to the measured gradient: 1.4477
```

- **A factor is dropped.** In a scalar chain the result is wrong by exactly the missing factor, as in section 2.1, so the symptom is a numerical check that disagrees by a constant factor, often 2 or the value of an input. In the network the effect is less uniform: above, four of the six entries are still right, because the two active neurons have a ReLU factor of 1, and only the entries of the switched-off neuron are wrong. A dropped factor can therefore survive a check on a few entries; the comparison has to cover every entry.
- **A local derivative does not exist.** $\text{ReLU}(2x - 1)$ has its corner where the inner function is zero, at $x = 0.5$. Away from it the chain rule and the measurement agree: slope 0 where the inner value is negative and 2 where it is positive. On the corner the convention of post 10 (section 3.1) gives $0 \cdot 2 = 0$, while a central difference straddles the corner and returns 1, the average of the one-sided slopes 0 and 2. A check that lands on a corner disagrees with a correct backward pass.
- **A variable reaches the loss along several paths and only one is counted.** The contributions of the paths add (section 5.1). In a network every hidden activation feeds several neurons of the next layer, and post 15 sums over them.
- **The matrix factors are multiplied in another order.** Scalar factors commute; tables of partial derivatives do not. With the shapes of section 6 the product is defined in the order written, and $(3, 6)$ times $(3, 3)$ is not defined at all (the inner-dimension rule of post 02).

---

## 10. Summary

| Concept | Takeaway |
|---|---|
| The rule | $\frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx} = f'(g(x)) \cdot g'(x)$ for $y = f(g(x))$ |
| Point of evaluation | Each local derivative is taken at the value its function received in the forward pass |
| Extended rule | $n$ composed functions give $n$ local derivatives, multiplied together |
| Several paths | A variable that reaches the output along several paths sums the products of the paths |
| Network reading | One factor per function from the parameter to the loss: 4 for $\mathbf{W}_1$ and 2 for $\mathbf{W}_2$ in the two-layer classifier |
| Backpropagation | The chain rule evaluated from the loss end, so that the running product is reused by every parameter further in |
| The check | The product of the factors equals a central difference through the whole composition, here to within $6.0 \times 10^{-11}$ |

---

## Common pitfalls

1. **Confusing the chain rule with the product rule.** $f(g(x))$ gives a product of slopes; $f(x) \cdot g(x)$ gives a sum of two terms.
2. **Evaluating the outer derivative at the wrong point.** $f'(g(x))$ is $f'$ with $g(x)$ substituted. Using $f'(x)$ gives 60 in place of 960 in section 4.
3. **Dropping a factor in a long chain.** Writing the chain as an explicit product, one factor per function, before simplifying anything is the remedy.
4. **Reading a zero ReLU factor as a bug.** A switched-off neuron contributes a factor of 0, so its weights receive no gradient from that sample. Those zeros are correct; if they occur for every sample, the neuron is the dead neuron of post 06.
5. **Expanding the whole forward expression before differentiating.** That worked for $3(2x^2)^5$ and is hopeless for a network. Each function is differentiated on its own, and only the results are combined.

---

## Further reading

- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 6.5 (Back-Propagation and Other Differentiation Algorithms), in particular section 6.5.2 on the chain rule of calculus (MIT Press, 2016).
- Griewank, A. and Walther, A., *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, chapter 3, for the forward and the reverse order of evaluating a chain and their costs (SIAM, 2008).
- Kinsley, H. and Kukieła, D., *Neural Networks from Scratch in Python*, chapter 8 (2020).
- Linnainmaa, S., *"The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors"* (Master's thesis, University of Helsinki, 1970).
- Rumelhart, D., Hinton, G., and Williams, R., *"Learning Representations by Back-Propagating Errors"* (Nature, 1986).
- Strang, G., *Calculus*, chapter 4 (Derivatives by the Chain Rule), for the rule and its proof (Wellesley-Cambridge, 2010).
- Werbos, P. J., *"Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences"* (PhD thesis, Harvard University, 1974).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 12 - Backpropagation through a single neuron](../12-backprop-through-a-single-neuron/index.md):** the first concrete application of the chain rule, on the smallest network whose chain has more than one factor.
- **[Post 19 - Softmax derivatives and the combined backward pass](../19-softmax-derivatives-and-the-combined-backward-pass/index.md):** where the loss factor $\partial L / \partial \mathbf{z}_2$ measured in section 8.2 gets its formula.
