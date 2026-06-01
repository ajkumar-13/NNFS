---
slug: 33-weight-initialisation
title: "Part 33 · Weight initialisation"
date: 2026-05-30
tags: [neural-networks, from-scratch, training, initialisation, he-init, xavier-init, glorot]
hero: diagrams/01-activation-variance-by-depth.svg
reading_time: 11
part: "Part VIII — Practical training and extensions"
---

# Part 33 · Weight initialisation

> **TL;DR.** Every `Layer_Dense` in the lectures initialised its weights with `0.01 * np.random.randn(...)`. That worked fine for two-hidden-layer networks but **fails silently** in deeper ones: the activations either shrink to zero through the layers (vanishing) or blow up (exploding), and training stalls before the first useful gradient appears. **Glorot/Xavier** initialisation (Glorot & Bengio, 2010) scales the random weights by $\sqrt{1/n_\text{in}}$ to preserve activation variance across layers — designed for symmetric activations like tanh. **He initialisation** (He et al., 2015) scales by $\sqrt{2/n_\text{in}}$ — designed for ReLU, which zeros half the activations and so needs the variance doubled to compensate. For a from-scratch series, He init is the right default for any network using ReLU; a one-line change to `Layer_Dense.__init__` is enough.
>
> **Reading time:** ~11 minutes.
>
> **After reading this you will be able to:**
> - Derive the variance-preservation argument that motivates Glorot and He initialisation.
> - Implement Xavier and He init in `Layer_Dense` and explain which one to use for which activation.
> - Recognise the symptoms of bad initialisation (zero gradients, NaN losses on step 1, plateaued training) and the fixes.

![Activation variance by layer depth under three initialisation schemes: the lectures' `0.01 * randn` shrinks to zero by layer 5, Xavier preserves variance through a 10-layer tanh stack, He preserves variance through a 10-layer ReLU stack.](diagrams/01-activation-variance-by-depth.svg)
*Same network, same input, three weight-scale choices. The wrong scale silently kills training before the first gradient is ever computed.*

---

## 1. The line the lectures glossed over

Every dense layer in the series was constructed with:

```python
self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
self.biases  = np.zeros((1, n_neurons))
```

`np.random.randn` draws from a standard normal (mean 0, variance 1). Multiplying by `0.01` makes the weights small — values in the rough range `[-0.03, +0.03]`. Biases start at zero.

For two-hidden-layer networks on small datasets (spiral, moons), this worked. For anything deeper, it fails in a specific predictable way that this lecture is about. The fix is one line; understanding *why* the fix is needed takes ten minutes.

The structural problem is that initialisation is the only thing standing between the optimiser and a wall of dead neurons. If the activations vanish in the first forward pass, the gradients vanish in the first backward pass, and the optimiser gets a zero update — every parameter stays at its initial value, training does nothing, and the loss curve is flat.

---

## 2. Variance through one linear layer

Consider one layer's pre-activation $z = \mathbf{W} \mathbf{x}$ (ignoring the bias for the moment). Each component of $z$ is a sum:

$$z_j = \sum_{i=1}^{n_\text{in}} W_{ij} \, x_i$$

If $W_{ij}$ and $x_i$ are independent, both have zero mean, and they are mutually independent across $i$, the variance of the sum is:

$$\text{Var}(z_j) = \sum_{i=1}^{n_\text{in}} \text{Var}(W_{ij}) \cdot \text{Var}(x_i) = n_\text{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

assuming all weights and inputs share the same per-element variance. So one forward pass through a linear layer **multiplies the variance by $n_\text{in} \cdot \text{Var}(W)$**.

The same argument applies layer by layer. After $L$ identical layers:

$$\text{Var}(z^{(L)}) = \big(n_\text{in} \cdot \text{Var}(W)\big)^L \cdot \text{Var}(\mathbf{x}^{(0)})$$

If $n_\text{in} \cdot \text{Var}(W) > 1$, the variance grows exponentially with depth (**exploding activations**). If $n_\text{in} \cdot \text{Var}(W) < 1$, it shrinks exponentially (**vanishing activations**). Only at *exactly* $n_\text{in} \cdot \text{Var}(W) = 1$ does it stay roughly stable.

That last equation is the entire initialisation story:

$$\boxed{\;\text{Var}(W) = \frac{1}{n_\text{in}}\;}$$

Pick the weight variance so that the product $n_\text{in} \cdot \text{Var}(W) = 1$, and activation variance is preserved across layers.

---

## 3. Glorot (Xavier) initialisation

The lectures' `0.01 * randn` sets $\text{Var}(W) = 0.01^2 = 10^{-4}$. For a layer with $n_\text{in} = 64$, that gives $n_\text{in} \cdot \text{Var}(W) = 0.0064$ — variance shrinks by 150× per layer. After 5 layers the activations are around $10^{-12}$ of their input scale and the gradients are effectively zero.

The fix from §2 is to set $\text{Var}(W) = 1 / n_\text{in}$, which means drawing from a normal of standard deviation $\sigma = 1 / \sqrt{n_\text{in}}$:

```python
self.weights = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs)
```

Glorot and Bengio (2010) refined this slightly. The same argument from §2 applies to the *backward* pass too — the gradient variance is preserved if $\text{Var}(W) = 1/n_\text{out}$. To keep both forward and backward variances stable simultaneously, they took the **average** of the two:

$$\text{Var}(W) = \frac{2}{n_\text{in} + n_\text{out}}$$

This is **Glorot initialisation** (also called **Xavier**, after the author's first name). In code:

```python
self.weights = np.random.randn(n_inputs, n_neurons) * \
    np.sqrt(2.0 / (n_inputs + n_neurons))
```

The Glorot-uniform variant draws from $\mathcal{U}(-a, a)$ with $a = \sqrt{6 / (n_\text{in} + n_\text{out})}$, which has the same variance as the normal version. The two are interchangeable in practice.

**When to use Glorot.** When the activation function is roughly symmetric around zero with bounded slope — tanh, sigmoid, softsign. These activations preserve approximately the same variance as their input.

**When NOT to use Glorot.** When the activation function is **ReLU**. That is the subject of the next section.

---

## 4. He initialisation: the ReLU fix

Glorot's variance-preservation argument assumes the activation function is approximately linear around zero. Tanh and sigmoid satisfy this (their derivative at zero is 1 and 1/4 respectively).

ReLU does not. It zeros out the entire negative half of its input:

$$\text{ReLU}(z) = \max(0, z)$$

If $z$ is drawn from a zero-mean distribution, half its values are negative and become zero after ReLU. The post-activation variance is **half** the pre-activation variance, not equal to it.

He et al. (2015) re-derived the variance-preservation argument with this in mind. To compensate for the lost half, they doubled the weight variance:

$$\text{Var}(W) = \frac{2}{n_\text{in}}$$

The code:

```python
self.weights = np.random.randn(n_inputs, n_neurons) * \
    np.sqrt(2.0 / n_inputs)
```

This is **He initialisation** (also called **Kaiming initialisation**, after the author's first name). It uses only $n_\text{in}$, not the harmonic mean, because the backward-pass argument was less critical in the original paper; in practice, the simpler form has stuck.

**When to use He.** Default for any ReLU-based network. That covers everything in this series past post 6.

A small table summarising the choice:

| Activation | Recommended init | Multiplier |
|---|---|---|
| tanh | Glorot / Xavier | $\sqrt{2 / (n_\text{in} + n_\text{out})}$ |
| sigmoid | Glorot / Xavier | $\sqrt{2 / (n_\text{in} + n_\text{out})}$ |
| **ReLU** | **He / Kaiming** | $\sqrt{2 / n_\text{in}}$ |
| Leaky ReLU | He (with leak factor in the formula) | $\sqrt{2 / ((1 + a^2) \cdot n_\text{in})}$ |
| Linear | Glorot | $\sqrt{2 / (n_\text{in} + n_\text{out})}$ |

For this series, since every hidden activation is ReLU, **the default should be He init**.

---

## 5. Updating `Layer_Dense`

A one-argument addition to the constructor lets the user choose:

```python
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, init="he",
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0,   bias_regularizer_l2=0.0):
        if init == "he":
            scale = np.sqrt(2.0 / n_inputs)
        elif init == "xavier" or init == "glorot":
            scale = np.sqrt(2.0 / (n_inputs + n_neurons))
        elif init == "small":  # the lectures' default, for backward compatibility
            scale = 0.01
        else:
            raise ValueError(f"unknown init: {init!r}")

        self.weights = scale * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

        # (regularisation attributes unchanged from post 30)
        ...
```

Three notes.

**Biases stay at zero.** Initialising biases at zero is universally fine. There is no variance argument for biases the way there is for weights; small uniform asymmetry comes from the weights alone.

**Backward compatibility.** Keeping `init="small"` as an opt-in option preserves the lectures' behaviour for the spiral / moons examples that worked fine without principled init. New code should pass `init="he"` explicitly.

**No optimiser changes.** All five optimisers from posts 22–27 work unchanged with any initialisation. He init makes them faster and more reliable; it does not change their interfaces.

---

## 6. Symptoms of bad initialisation

Three failure modes worth recognising at sight.

**Loss is exactly $\ln(C)$ on step 1 and never moves.** For a $C$-class classifier with cross-entropy loss, a uniform-prediction model has loss $\ln(C)$ — about 2.30 for 10 classes, 1.10 for 3 classes, 0.69 for 2 classes. If your loss starts at exactly this value and stays flat for many epochs, the activations have vanished and the softmax output is uniform; no gradient flows.

**Loss is `inf` or `NaN` immediately.** Activations have exploded — the variance grew large enough that `np.exp(z)` in the softmax overflowed. Symptom: print the first few activations; if any are `> 1e30`, you have exploding init.

**Loss decreases on the first few epochs and then stalls.** Often a sign that *some* layers are healthy and others are dead. Possible if you mixed init strategies, or if the input data isn't standardised (a feature with range $\pm 1000$ behaves like an exploding init even if the weights are sane).

**Diagnostic recipe.**

```python
dense1.forward(X)
print("dense1 output stats:")
print(f"  mean={dense1.output.mean():.4f}  std={dense1.output.std():.4f}")
print(f"  fraction zero (post-ReLU expect ~50%): not measured here")
activation1.forward(dense1.output)
print("activation1 output stats:")
print(f"  mean={activation1.output.mean():.4f}  std={activation1.output.std():.4f}")
print(f"  fraction zero: {(activation1.output == 0).mean():.4f}")
# Repeat for each layer; healthy stats: std around 1 (or close to input std),
# zero-fraction roughly 0.5 after ReLU.
```

If the std collapses to $10^{-5}$ or grows to $10^5$ by layer 3, the init is wrong.

---

## 7. Initialisation interacts with normalisation and depth

A few practical points where init effects compose with other design choices.

**Batch normalisation absorbs init.** Networks with batch normalisation (or layer norm) are far less sensitive to init choice — the normalisation step rescales activations to roughly unit variance regardless of what the weights produced. This is part of why batchnorm became so popular: it made deep networks trainable even with sloppy init.

**Residual connections need a smaller init for very deep nets.** The standard ResNet initialisation scales the last layer in each residual block by $1/\sqrt{2L}$ (where $L$ is the number of blocks) to keep the variance constant despite the residual addition. For non-residual networks, He init alone is enough.

**Transformers use slightly different defaults.** GPT-style transformer blocks initialise output projections to a smaller scale than He, on the grounds that residual additions accumulate variance and the output projections shouldn't amplify them. Specific value: `std = 0.02 / sqrt(2 * n_layers)`.

For the from-scratch series — which has no residuals, no batchnorm, and at most three hidden layers — **He init is the right and sufficient default**. The exotic variants matter when you're building specific deep architectures, not when you're learning the basics.

---

## 8. Anticipated questions

- **Why did the lectures' `0.01 * randn` work for the spiral classifier?** Because the network was shallow. Two hidden layers means the activation variance only shrinks twice; the first layer's output is still well within float32 precision and the gradients still propagate. With five layers it would have failed.
- **Should I re-initialise weights between runs?** Always. The whole point of random init is the symmetry-breaking it provides; reusing the same weights gives the same optimisation path.
- **Does He init help when the dataset is already standardised?** Yes — they solve different problems. Standardising the *input* makes the first layer behave well; He init makes every layer behave well. Use both.
- **What if my activation is a custom one — say Swish or GELU?** Use He init as a reasonable default. The exact correction factor for these activations exists in the literature (e.g. SELU has its own bespoke init in Klambauer et al., 2017) but the gain from getting it exactly right is small compared to getting it approximately right with He.
- **Is there a "best" init in absolute terms?** No — best init depends on architecture, activation, depth, and presence of normalisation. He for ReLU + dense, Glorot for tanh + dense, normal-with-truncation for transformers, and so on.
- **Why is the bias initialised to zero and not something nonzero?** Biases break symmetry only weakly (every neuron in a layer with zero biases has different weights, which is enough). Some recipes add tiny positive bias for ReLU (e.g., 0.01) to push more neurons into the active region from the start; the benefit is marginal.

---

## 9. Summary

| Concept | Takeaway |
|---|---|
| Variance preservation | $\text{Var}(W) = 1 / n_\text{in}$ keeps activations from vanishing or exploding |
| Glorot / Xavier | $\sigma = \sqrt{2 / (n_\text{in} + n_\text{out})}$; for tanh, sigmoid, linear |
| **He / Kaiming** | $\sigma = \sqrt{2 / n_\text{in}}$; **the right default for ReLU** |
| Bias init | always zero; symmetry breaks via the weights |
| The lectures' `0.01 * randn` | OK for shallow networks; fails silently at depth |
| Symptoms of bad init | loss stuck at $\ln(C)$, or `NaN` on step 1, or activations $\to 0$ after 3 layers |
| One-line fix in `Layer_Dense` | replace `0.01` with `np.sqrt(2.0 / n_inputs)` |

---

## Common pitfalls

- **Using `0.01 * randn` for a 6-layer ReLU network.** Activations shrink by ~32× per layer, so by layer 6 they are $\sim 10^{-9}$ of input scale. Training appears to do nothing.
- **Using He init for a tanh network.** Activations grow because tanh doesn't kill half the values. Use Glorot for tanh.
- **Initialising biases to nonzero random values.** Adds noise without breaking any symmetry (the weights already broke it). Stay at zero.
- **Forgetting to re-seed the RNG between experiments.** Reproducibility breaks; comparing two init schemes becomes confounded by the random draw.
- **Using a single init scheme for every layer when activations differ.** A network with a tanh hidden layer followed by a ReLU should use Glorot for the tanh's *input* layer and He for the ReLU's. In practice, just use He everywhere for ReLU-only networks.
- **Forgetting to standardise the input.** Even perfect init does not save you if the input has a feature with variance $10^6$. Standardise inputs first.

---

## Further reading

- Glorot, X. and Bengio, Y., *"Understanding the difficulty of training deep feedforward neural networks"* (AISTATS, 2010) — the original Xavier paper.
- He, K. et al., *"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"* (ICCV, 2015) — the He / Kaiming init paper.
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning* — chapter 8.4 (Parameter Initialisation Strategies) (MIT Press, 2016).
- Klambauer, G. et al., *"Self-Normalizing Neural Networks"* (NeurIPS, 2017) — SELU and its bespoke init.
- Saxe, A. M., McClelland, J. L., and Ganguli, S., *"Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"* (ICLR, 2014) — orthogonal initialisation, a related family worth knowing.

Full citations in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Part 34 — Sigmoid and binary cross-entropy](../34-sigmoid-and-binary-cross-entropy/index.md)** — the binary counterpart to softmax + categorical cross-entropy, with the same combined-derivative trick.
- **[Part 35 — What to read after this series](../35-whats-next/index.md)** — pointers to convolution, recurrence, attention, batchnorm, and the rest of the modern deep-learning toolkit.

---

> **Try it yourself:** Re-train [Project 03 — Fashion-MNIST](../../projects/03-fashion-mnist/README.md) with `init="he"` and compare to the default. The accuracy gap is small (the network is only 2 hidden layers), but the loss in the first 5 epochs drops noticeably faster.
