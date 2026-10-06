# 33 - Weight initialisation

> **TL;DR.** A dense layer multiplies the mean square of its input by $n_\text{in} \cdot \text{Var}(W)$, and a ReLU after it keeps half of that. With the fixed scale 0.01 of posts 04 to 32 and 64 neurons per layer, the factor is 0.0032 per layer, forward and backward: ten such layers start at a loss of exactly $\ln 3$, and neither plain gradient descent nor Adam moved them in 200 epochs. Glorot initialisation sets the factor to 1 for an activation that is linear near zero, He initialisation sets it to 1 for ReLU, and `Layer_Dense` gains an `init` argument that chooses between them. On the series' own one-hidden-layer spiral network both raised training accuracy over the 0.01 scale, on 19 and on 20 of 20 seeds, and He was not ahead of Glorot there.
>
> **Prerequisites:** [Post 04](../04-dense-layer-class-and-spiral-data/index.md), [Post 06](../06-activation-functions-relu-and-softmax/index.md).
> **Safe to skip?** Skip it if the reader can already derive the weight variances $2/(n_\text{in} + n_\text{out})$ and $2/n_\text{in}$ from the factor by which one layer scales its signal, and can tell a vanished forward pass from an exploded one by the first two losses of a run.
>
> **After reading, you will be able to:**
>
> - Derive the variance-preservation argument behind Glorot and He initialisation.
> - Implement both in Layer_Dense through an init argument and say which one fits which activation.
> - Recognise the symptoms of bad initialisation: gradients near zero, a NaN loss in the first steps, a plateau from the start.

![A log-scale chart of the standard deviation of each layer's activations in ten 64-neuron ReLU layers on seed 0. At scale 0.01 it falls from 0.0036 at layer 1 to 1.3 times 10 to the minus 14 at layer 10, with Glorot from 0.062 to 0.0017, and with He it stays between 0.22 and 0.36. A table below gives the factor per layer on the mean square, mean over seeds 0 to 99: predicted 0.0032, 0.5 and 1, measured within 4 percent of each, forward and backward.](diagrams/01-activation-spread-by-depth.svg)

*One set of weight draws at three scales through the same ten layers. Each layer multiplies the mean square by the same factor, so ten layers apply it ten times.*

---

## 1. The question: why does a fixed scale work in a shallow network and fail in a deep one?

Every dense layer of posts 04 to 32 drew its weights as `0.01 * np.random.randn(n_inputs, n_neurons)`, mean 0 and standard deviation 0.01, and set its biases to zero. [Post 04](../04-dense-layer-class-and-spiral-data/index.md), section 5.1, measured six stacked 64-neuron layers with nothing between them: at 0.01 the outputs shrink about 12 times per layer, to a standard deviation of $2.2 \times 10^{-8}$; at 1.0 they grow 8 times per layer; at $1/\sqrt{64} = 0.125$ they keep their size.

The question here is: **which weight scale lets a signal, and the gradient that returns through the same weights, keep its size through any number of layers, and what does a network look like when the scale is wrong?** The answer is a derivation with four assumptions and a factor of one half for ReLU, each factor measured on ten stacked layers, and one new argument of `Layer_Dense`.

---

## 2. One linear layer: the factor $n_\text{in} \cdot \text{Var}(W)$

Take one neuron of a layer with $n_\text{in}$ inputs and zero bias, as at initialisation. Its pre-activation is a sum over the inputs:

$$z_k = \sum_{j=1}^{n_\text{in}} w_{kj} \, x_j$$

Two assumptions are needed, and both hold exactly for a freshly constructed layer:

1. the weights $w_{kj}$ are drawn independently of each other from one distribution that is symmetric about 0, hence of mean 0, with variance $\text{Var}(W)$;
2. the weights are independent of the layer's inputs $x_j$.

Hold the inputs fixed and average over the draw of the weights. The mean of $z_k$ is 0, because every weight has mean 0. Its mean square is

$$\mathbb{E}\left[ z_k^2 \right] = \sum_{j} \sum_{j'} \mathbb{E}\left[ w_{kj} \, w_{kj'} \right] x_j \, x_{j'} = \text{Var}(W) \sum_{j=1}^{n_\text{in}} x_j^2$$

The cross terms with $j \ne j'$ vanish: two independent weights of mean 0 have $\mathbb{E}[w_{kj} w_{kj'}] = 0$. The terms with $j = j'$ contribute $\text{Var}(W) \, x_j^2$ each. Since $z_k$ has mean 0, its mean square is its variance, and writing the sum as $n_\text{in}$ times the mean of the $x_j^2$ gives

$$\text{Var}(z_k) = n_\text{in} \cdot \text{Var}(W) \cdot \overline{x^2}, \qquad \overline{x^2} = \frac{1}{n_\text{in}} \sum_{j=1}^{n_\text{in}} x_j^2$$

**One linear layer multiplies the mean square of its input by $n_\text{in} \cdot \text{Var}(W)$.** The quantity on the right is the mean square of the inputs, not their variance. The two agree only when the inputs have mean 0, a third assumption that textbook versions of the argument add and that the outputs of a ReLU do not satisfy. `snippets/variance_factors.py` feeds one input row of 64 values with mean 1 to 100,000 neurons:

```text
input: mean 1.0008, mean square 1.0790, variance 0.0775
init     n_in * Var(W)   mean square of z / mean square of x   variance of z / variance of x
small    0.0064          0.0064                                0.0890
xavier   1.0000          0.9982                                13.9031
he       2.0000          1.9963                                27.8063
```

The mean-square column reproduces the factor to the third decimal. The variance column is off by a factor of 14, the ratio of the input's mean square to its variance.

After $L$ linear layers of the same width the factor is applied $L$ times, $(n_\text{in} \cdot \text{Var}(W))^L$. Above 1 the signal grows exponentially with depth, **exploding activations**; below 1 it shrinks exponentially, **vanishing activations**. With the scale 0.01 and 64 inputs the factor is $64 \cdot 0.01^2 = 0.0064$, whose square root 0.08 is the shrinkage per layer of post 04. The factor is 1 when

$$\text{Var}(W) = \frac{1}{n_\text{in}}$$

that is, a standard deviation of $1/\sqrt{n_\text{in}}$, the scale LeCun, Bottou, Orr and Müller (1998) recommend and the 0.125 of post 04's third column.

---

## 3. Glorot initialisation: the backward pass has a factor too

The gradient travels through the same weights in the other direction. `Layer_Dense.backward` computes `dinputs = np.dot(dvalues, self.weights.T)` (post 16), which for one input is a sum over the layer's $n_\text{out}$ neurons:

$$\frac{\partial L}{\partial x_j} = \sum_{k=1}^{n_\text{out}} w_{kj} \, \frac{\partial L}{\partial z_k}$$

This is the sum of section 2 with $n_\text{out}$ terms in place of $n_\text{in}$. Under a further assumption,

3. the weights are independent of the gradient $\partial L / \partial z_k$ that arrives from above,

the same steps give a factor of $n_\text{out} \cdot \text{Var}(W)$ on the mean square of the gradient. Assumption 3 is only approximately true: the arriving gradient was computed from a forward pass through these very weights. The script checks the arithmetic on a fixed gradient row, where the assumption holds, and prints 0.0064, 1.0035 and 2.0070 against factors of 0.0064, 1 and 2; section 5 checks the assumption itself on a real backward pass.

So there are two conditions, $n_\text{in} \cdot \text{Var}(W) = 1$ for the signal and $n_\text{out} \cdot \text{Var}(W) = 1$ for the gradient, and a layer whose two sizes differ cannot meet both. Glorot and Bengio (2010) proposed the compromise that uses the average of the two sizes:

$$\text{Var}(W) = \frac{2}{n_\text{in} + n_\text{out}}$$

This is **Glorot initialisation**, also called **Xavier initialisation** after its first author's given name. The paper proposes it as a uniform distribution on $(-a, a)$ with $a = \sqrt{6/(n_\text{in} + n_\text{out})}$. A uniform distribution on $(-a, a)$ has variance $a^2/3$, which is the value above. This series draws from a normal distribution with that variance instead, as its released projects do; only the variance enters the derivation, and `snippets/network.py` prints 0.015625 for both forms at $n_\text{in} = n_\text{out} = 64$.

The derivation treats the activation as the identity, a fair description of one with slope 1 at zero and outputs centred on zero. Tanh is the standard example: it keeps 0.9806 of the mean square of inputs with standard deviation 0.1, and 0.3942 at standard deviation 1. A sigmoid, with slope $1/4$ at zero and outputs centred on $1/2$, meets neither condition. ReLU fails the description in a way that can be computed.

---

## 4. He initialisation: ReLU keeps half of the mean square

ReLU replaces every negative pre-activation by zero, $a_k = \max(0, z_k)$. Under assumptions 1 and 2 the pre-activation $z_k$ is symmetric about zero, because the weights are: $z_k$ and $-z_k$ are equally likely. Half of the values are therefore set to zero, and the positive half carries half of the total of $z_k^2$:

$$\mathbb{E}\left[ a_k^2 \right] = \frac{1}{2} \, \mathbb{E}\left[ z_k^2 \right]$$

The one half is a factor of the mean square. It is not a factor of the variance: the outputs of a ReLU have a positive mean, and for a normal $z$ with standard deviation $s$ the mean of $a$ is $s/\sqrt{2\pi}$ and its variance is $(1/2 - 1/(2\pi)) \, s^2$. The script prints all three for a million draws:

```text
ReLU: fraction of zeros 0.4995
ReLU: mean square of a / mean square of z = 0.5003   (closed form 1/2)
ReLU: variance of a / variance of z       = 0.3408   (closed form 1/2 - 1/(2 pi) = 0.3408)
```

The figure below draws the two halves of $z$ and sets the three ratios beside their closed forms.

![Left, the density of a zero-centred normal pre-activation z, its negative half shaded red and labelled set to 0, its positive half blue and labelled kept. Right, the ReLU output a: the same blue half, and a spike at a = 0 that holds half of the draws. A table gives the fraction of outputs at 0 as 1/2, measured 0.4995, the ratio of mean squares as 1/2, measured 0.5003, and the ratio of variances as 1/2 minus 1/(2π) = 0.3408, measured 0.3408.](diagrams/02-the-factor-of-two.svg)

*What ReLU does to a zero-centred pre-activation: the negative half moves to zero, half of the mean square survives, and 0.34 of the variance.*

The mean square is the quantity that matters, because section 2 showed that the next layer scales the mean square of what it receives. A dense layer followed by a ReLU therefore multiplies the mean square by

$$\frac{n_\text{in} \cdot \text{Var}(W)}{2}$$

He, Zhang, Ren and Sun (2015) set this factor to 1:

$$\text{Var}(W) = \frac{2}{n_\text{in}}$$

This is **He initialisation**, also called **Kaiming initialisation** after its first author's given name. The 2 undoes the halving and is not a tuned constant.

The backward pass has its own half. `Activation_ReLU.backward` zeroes the gradient where the pre-activation was not positive, which is half of the entries. Under a fourth assumption,

4. whether a gate is open is independent of the gradient that arrives at it,

the gate keeps half of the mean square of the gradient, and the backward factor of a dense layer with its ReLU is $n_\text{out} \cdot \text{Var}(W) / 2$. With $\text{Var}(W) = 2/n_\text{in}$ that is $n_\text{out}/n_\text{in}$ per layer. Over a whole network these ratios multiply to the number of outputs of the last layer over the number of inputs of the first, a fixed number that does not shrink or grow with depth, so the input size alone is enough and no average is needed; He et al. derive both versions and note that either one suffices.

| Hidden activation | Scheme | Standard deviation of the weights | Factor per layer at equal widths |
|---|---|---|---|
| tanh | Glorot | $\sqrt{2/(n_\text{in} + n_\text{out})}$ | 1 while the inputs stay small |
| ReLU | He | $\sqrt{2/n_\text{in}}$ | 1 |
| ReLU | Glorot | $\sqrt{2/(n_\text{in} + n_\text{out})}$ | $1/2$ |
| ReLU | fixed 0.01, 64 inputs | 0.01 | $64 \cdot 0.01^2 / 2 = 0.0032$ |

---

## 5. Ten layers, measured forward and backward

`snippets/ten_layers.py` builds the stack of the first figure on the spiral data: ten `Layer_Dense` layers of 64 neurons, each followed by `Activation_ReLU`, then `Layer_Dense(64, 3)` and the combined softmax and loss. Unlike post 04's `init_scale.py` it has an activation after every layer, ten layers, and float64 without `nnfs.init()`; it first repeats post 04's six linear layers and prints the same $2.2 \times 10^{-8}$. The three schemes get the same standard normal draws, so only the scale differs. On seed 0:

```text
layer   small       xavier      he          fraction of zeros (the same for all three)
1       3.565e-03   6.205e-02   3.565e-01   0.508
2       1.705e-04   3.710e-02   3.014e-01   0.543
3       9.798e-06   2.665e-02   3.062e-01   0.522
4       4.836e-07   1.644e-02   2.672e-01   0.538
5       3.096e-08   1.316e-02   3.024e-01   0.414
6       1.674e-09   8.890e-03   2.889e-01   0.539
7       1.049e-10   6.963e-03   3.200e-01   0.419
8       5.888e-12   4.887e-03   3.176e-01   0.604
9       2.875e-13   2.983e-03   2.742e-01   0.490
10      1.337e-14   1.734e-03   2.254e-01   0.550
loss at initialisation: small 1.0986123, xavier 1.0986607, he 1.1187940   (ln 3 = 1.0986123)
```

The columns are the standard deviation of each layer's activations. With 0.01 it falls by eleven orders of magnitude between layers 1 and 10, with Glorot by a factor of 36, and with He it stays between 0.22 and 0.36. The fraction of zeros is the same in all three columns: a positive factor on the weights changes the size of a pre-activation and not its sign. With 0.01 the ten-layer network outputs three equal probabilities and its loss equals $\ln 3$ to seven decimals.

A single draw scatters around the derived factor, from 0.66 to 1.39 for He on seed 0 in layers 2 to 10. The derivation is about the average over draws, so the script averages each factor over seeds 0 to 99:

```text
init     forward, layer 1      forward, layers 2 to 10        backward, dense layers 2 to 10
         predicted  measured   predicted  measured            predicted  measured
small    0.0001     0.0001006  0.0032     0.0031 to 0.0032    0.0032     0.0032 to 0.0032
xavier   0.0303     0.0305     0.5        0.4846 to 0.5077    0.5        0.4922 to 0.5024
he       1          1.006      1          0.9692 to 1.0153    1          0.9896 to 1.0091
```

**Forward, every mean is within 4 percent of the predicted factor.** Layer 1 has two inputs, so its factor is $2 \cdot \text{Var}(W)/2$: 1 for He, $2/66 = 0.0303$ for Glorot, whose scale also looks at the 64 outputs, and $10^{-4}$ for 0.01. Layers 2 to 10 give 0.0032, 0.5 and 1.

**Backward, the factors are the same three numbers,** measured on the mean square of `dinputs` from one dense layer to the one before it. Assumptions 3 and 4 are not exact, and on this stack the means are still within 2 percent of the prediction.

**The weight gradients are small in every layer at once.** `dweights` of a layer is the product of what arrives from below and what arrives from above (post 16), so a layer near the input has a full-size signal and a shrunken gradient, and a layer near the output the reverse. On seed 0 the root mean square of `dweights` under 0.01 is between $1.0 \times 10^{-16}$ and $7.8 \times 10^{-16}$ in all eleven layers; under Glorot it is $1.8 \times 10^{-5}$ to $2.5 \times 10^{-5}$ in layers 2 to 10; under He it is between 0.0023 and 0.016 in all eleven. A vanished network has no layer that still learns.

**Tanh is the mirror case, and a milder one.** The same stack with `Activation_Tanh` of post 17 gives a forward factor of 0.94 to 0.99 with Glorot. He's variance is twice too large for tanh: saturation holds the forward factor to 1.02 to 1.10, and the backward factor is 1.16 to 1.83 per layer.

---

## 6. The `init` argument of `Layer_Dense`

The class is that of [post 30](../30-l1-and-l2-regularisation/index.md) with one new argument and the lines that turn it into a scale. `forward`, `backward` and the four regulariser attributes are unchanged. The constructor now begins:

```python
    def __init__(self, n_inputs, n_neurons, init="he",
                 weight_regularizer_l1=0.0, weight_regularizer_l2=0.0,
                 bias_regularizer_l1=0.0, bias_regularizer_l2=0.0):
        # Added in post 33: the standard deviation of the initial weights.
        if init == "he":
            scale = np.sqrt(2.0 / n_inputs)
        elif init == "xavier" or init == "glorot":
            scale = np.sqrt(2.0 / (n_inputs + n_neurons))
        elif init == "small":
            scale = 0.01                        # the fixed scale of posts 04 to 32
        else:
            raise ValueError(f"unknown init: {init!r}")

        self.weights = scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
```

This is the spelling of the released project `nn-p02`, which needs the argument: the same four names, the same default, the same error. The projects `nn-p01`, `nn-p03` and `nn-p04` keep the fixed 0.01 and have no such argument.

**The default is `"he"`, and that changes what an unchanged script computes.** A script of posts 04 to 32 that calls `Layer_Dense(2, 64)` with this class draws weights 100 times larger in its first layer and gets other results. Every number of those posts was produced with 0.01, and reproducing one requires `init="small"`. `snippets/network.py` shows the three scales on the same draws, and the first line is a row that post 04 prints:

```text
init='small'   [0.01764052 0.00400157 0.00978738]
init='xavier'  [1.11568467 0.25308164 0.61900825]
init='he'      [1.76405235 0.40015721 0.97873798]
```

---

## 7. Symptoms of a wrong scale

The training runs of this section and the next share one setup, printed by the scripts: `nnfs.init()` once (seed 0, float32, and its own `np.dot`); per run `np.random.seed(s)`, `spiral_data(samples=100, classes=3)`, a number of hidden `Layer_Dense` layers of 64 neurons with ReLU, `Layer_Dense(64, 3)`, the combined softmax and loss, and full-batch epochs.

**Too small: gradients near zero and a plateau from the first epoch.** `snippets/symptoms.py` gives ten hidden layers the scale 0.01:

```text
seed 0, first pass: loss 1.0986123 (ln 3 = 1.0986123), largest |dweights| over the 11 layers 2.3e-15
one step of Optimizer_SGD(learning_rate=1.0): 0 of 37,184 weights changed; spacing of float32 at 0.01: 9.3e-10
```

The first loss is $\ln C$ for $C$ classes, the loss of answering $1/C$ for every sample: 1.0986 for three classes, 0.6931 for two, 2.3026 for ten. The largest gradient is $2.3 \times 10^{-15}$, and a step of that size is below the distance between neighbouring float32 numbers near 0.01, so the subtraction returns the old weight: not one of the 37,184 weights changes. After 200 updates the loss is still 1.0986 and the accuracy 0.3333 on each of seeds 0 to 4, under `Optimizer_SGD(learning_rate=1.0)` and under `Optimizer_Adam(learning_rate=0.02, decay=1e-5)` alike. No error is raised at any point.

**How deep is too deep depends on the optimiser.** Each extra hidden layer divides the first gradient by about 18, the square root of $1/0.0032$. Plain gradient descent takes steps proportional to the gradient and feels every one of those divisions; `snippets/depth_sgd.py` runs it for 501 epochs on seeds 0 to 4:

| Hidden layers | Loss at epoch 500, 0.01 | Loss at epoch 500, He |
|:---:|:---:|:---:|
| 1 | 1.0431 to 1.0713 | 1.0237 to 1.0604 |
| 2 | 1.0745 to 1.0946 | 0.4205 to 0.5755 |
| 3 | 1.0986 on all five seeds | 0.3769 to 0.5887 |

With one hidden layer the two scales are close after 501 epochs. With two, He is below a loss of 1.0 after 77 to 133 epochs on every seed and 0.01 has barely left $\ln 3$; with three, 0.01 has not moved in the fourth decimal.

Adam divides each gradient by a running size of that gradient (post 27), so the scale of the gradient cancels until it falls to the size of Adam's $\epsilon = 10^{-7}$. `snippets/depth_small.py` and `snippets/depth_he.py`, 501 epochs, seeds 0 to 4:

| Hidden layers | Largest first gradient, 0.01 | Runs with 0.01 that never go below 1.0 | Loss at epoch 500, 0.01 | Loss at epoch 500, He |
|:---:|:---:|:---:|:---:|:---:|
| 2 | $2.5 \times 10^{-5}$ to $3.8 \times 10^{-5}$ | 0 of 5 | 0.1189 to 0.2032 | 0.0251 to 0.0704 |
| 4 | $8.0 \times 10^{-8}$ to $1.1 \times 10^{-7}$ | 0 of 5 | 0.0513 to 0.3098 | 0.0144 to 0.0525 |
| 5 | $4.3 \times 10^{-9}$ to $6.8 \times 10^{-9}$ | 1 of 5 | 0.0956 to 1.0986 | 0.0121 to 0.0470 |
| 6 | $2.3 \times 10^{-10}$ to $4.6 \times 10^{-10}$ | 3 of 5 | 0.1120 to 1.0986 | 0.0111 to 0.0978 |

With Adam the 0.01 scale trains through four hidden layers on all five seeds, loses one seed at five and three at six. The two seeds that escape at six hidden layers need 186 and 191 epochs to pass a loss of 1.0, against 15 to 27 for He. The failure line sits where the first gradient drops well below $\epsilon$, for this width and this number of epochs. With He every run at every depth is below 1.0 within 37 epochs. The figure below draws every run of the two sweeps, one tick per seed.

![Two panels of bands of the loss at epoch 500, each band spanning five seeds with one tick per seed, 0.01 in grey and He in blue, with a dashed line at ln 3. Plain gradient descent: with one hidden layer both scales end between 1.02 and 1.07; with two and three He ends between 0.38 and 0.59 while 0.01 stays between 1.07 and 1.10. Adam: 0.01 trains at two and four hidden layers, leaves one run at ln 3 at five and three at six, and He ends below 0.1 at every depth. Two columns give the runs below a loss of 1.0 and the first epoch below 1.0.](diagrams/03-depth-and-optimiser.svg)

*Under plain gradient descent the 0.01 scale stays at ln 3 from three hidden layers; under Adam it first loses runs at five.*

The released project `nn-p02` documents the failure on two 16-neuron hidden layers under Adam. Its published evaluation reports, for two moons at noise 0.1, 200 of 200 held-out points with He against 174 of 200, 87.0 percent, with the 0.01 scale on seed 0, and 197 to 200 against at most 181 over seeds 0 to 9.

**Too large: a finite first loss, then NaN.** With every weight drawn at scale 1.0, a plain `randn`, the factor per ReLU layer is $64/2 = 32$. The same script, ten hidden layers:

```text
seed  loss, first pass  largest |output|  largest |dweights|  first NaN loss, SGD at 1.0  loss at epoch 3 and at epoch 200, Adam at 0.02
   0           10.7564           4.5e+07             2.9e+06  epoch 1                      10.6379  2.2565
   1           10.7027           6.7e+07             3.4e+06  epoch 1                      10.6917  0.5373
```

The first loss is not NaN, on any of the ten seeds the script runs. The softmax of post 06 subtracts the row maximum before exponentiating, so logits of $10^7$ produce a one-hot row without overflow, and the loss of post 08 clips at $10^{-7}$, so a confidently wrong sample costs $-\ln 10^{-7} = 16.118$. About two thirds of the samples are wrong, and the loss is near $16.118 \cdot 2/3 = 10.745$: between 10.38 and 11.19 on the ten seeds. The NaN arrives one update later. Gradients of $10^6$ under a learning rate of 1.0 move the weights to $10^6$, the next forward pass overflows float32, and the loss of epoch 1, the second epoch, is NaN on all ten seeds, with the warnings `overflow encountered in cast` and `invalid value encountered in dot`. Adam, whose steps are bounded by the learning rate, produces no NaN. Its loss after three updates is still between 9.51 and 11.23, and after 200 it is between 0.16 and 2.26.

| First two losses | Largest gradient | Reading |
|---|---|---|
| $\ln C$, then $\ln C$ | many orders below the weights | vanished: scale too small for the depth |
| about $16 \cdot (1 - 1/C)$, then NaN, or a slow fall under Adam | many orders above the weights | exploded: scale too large |
| a little above $\ln C$, then falling | within a few orders of the weights | healthy |

One forward pass and the table of section 5 separate the cases before any training.

---

## 8. What changes for the series' own network

The network of Part VI has one hidden layer, so the factor of section 4 is applied once and no signal vanishes. The scale still matters. `snippets/spiral_wins.py` repeats the documented run of [post 28](../28-generalization-and-testing/index.md), `Layer_Dense(2, 64)`, ReLU, `Layer_Dense(64, 3)`, `Optimizer_Adam(learning_rate=0.02, decay=1e-5)`, 10,001 epochs, with nothing changed but `init` on both layers. Accuracies are measured forward-only after the last update, and the test set is 100 points per class drawn after training. Run without arguments it trains seed 0, and its `small` columns are the figures of post 28:

```text
seed  train acc: small     he  xavier   test acc: small     he  xavier
   0             96.33  91.33  97.67             82.33  72.00  79.33
```

Given the seeds 0 to 19 as arguments, the script ends with the ranges and with the number of seeds on which the first of a pair has the higher accuracy, ties in brackets:

```text
range and mean over the 20 seeds, in percent
init     training accuracy         test accuracy
small    78.00 to 96.67 (88.97)    61.33 to 83.00 (76.65)
he       91.33 to 99.33 (97.13)    72.00 to 84.33 (78.72)
xavier   95.67 to 99.33 (97.45)    76.67 to 86.00 (81.48)
seeds on which the first has the higher accuracy than the second, of 20; in brackets, seeds on which the two are equal
pair               training   test
he over small      19 (0)     12 (1)
xavier over small  20 (0)     15 (0)
xavier over he     11 (1)     17 (0)
```

**Both derived scales fit the training set better than 0.01.** He has the higher training accuracy on 19 of the 20 seeds and Glorot on all 20. The exception is seed 0, the documented seed of Part VI, where 0.01 reaches 96.33 percent and He 91.33.

**Test accuracy follows for Glorot and not for He.** Glorot is ahead of 0.01 on 15 of the 20 seeds. He is ahead on 12 and level on one: on 8 of the first ten seeds and on 4 of the next ten.

**He is not ahead of Glorot on this ReLU network.** Glorot has the higher test accuracy on 17 of the 20 seeds, and in training accuracy the two are level, 11 seeds to 8. This does not contradict section 5. With one hidden layer there is no depth over which Glorot's half per layer could compound, and the two schemes differ mainly in the first layer, where two inputs give He a standard deviation of 1 and Glorot one of $\sqrt{2/66} = 0.17$. The evaluation of `nn-p02` reports the same for its two hidden layers: Glorot at 197 to 200 of 200, as He. The case for He over Glorot on ReLU is the factor of one half per layer, and it is a case about depth.

---

## 9. Normalisation and depth

An initialisation sets the scale once, before the first update, and nothing keeps the factors near 1 afterwards. Much deeper architectures therefore add parts that act at every step: normalisation layers, which rescale activations during training, and skip connections, which carry the signal around a block of layers. Both are topics of the series `cnn-from-scratch`, which starts from the He scale of this post.

---

## 10. Make it run: the scripts

Every code block and every number of this post comes from a script in `snippets/`, run from the series root, for example:

```text
python posts/33-weight-initialisation/snippets/ten_layers.py
```

| Script | Contents | Time |
|---|---|---|
| `network.py` | `Layer_Dense` with `init`, the other classes unchanged, the shared training loop; prints section 6 | 1 s |
| `variance_factors.py` | the factors of sections 2 to 4 on random numbers | 1 s |
| `ten_layers.py` | ten layers under three schemes, forward and backward, ReLU and tanh (section 5) | 10 s |
| `symptoms.py` | ten hidden layers at scale 0.01 and at scale 1.0 (section 7) | 30 to 40 s |
| `depth_sgd.py`, `depth_small.py`, `depth_he.py` | depth sweeps over five seeds (section 7) | 25 to 40 s each |
| `spiral_wins.py` | the run of post 28 under each scheme, seed by seed, with ranges and counts (section 8) | 25 s for seed 0; 7 min for seeds 0 to 19 |

The first three run in float64 without `nnfs.init()`. The training scripts call it once and reseed for each run; the depth scripts and `spiral_wins.py` take seeds as arguments. All need NumPy and the `nnfs` package.

---

## 11. What can go wrong?

**A misspelt scheme.** The names are lower case and the constructor checks them. `Layer_Dense(2, 3, init="He")` prints:

```text
ValueError: unknown init: 'He'
```

**An earlier script rerun with the new default.** Nothing is raised: the first layer's weights are 100 times larger and every figure of the run differs from the post it came from. `init="small"` is the argument that restores a result of posts 04 to 32.

---

## 12. Summary

| Concept | Takeaway |
|---|---|
| One linear layer | mean square of the signal times $n_\text{in} \cdot \text{Var}(W)$, of the gradient times $n_\text{out} \cdot \text{Var}(W)$ |
| ReLU | keeps half of the mean square and 0.34 of the variance |
| Glorot | $\text{Var}(W) = 2/(n_\text{in} + n_\text{out})$; factor 1 for tanh on small inputs, $1/2$ for ReLU |
| He | $\text{Var}(W) = 2/n_\text{in}$; factor 1 for ReLU, where the fixed 0.01 gives 0.0032 at 64 neurons |
| Symptoms | $\ln C$ and no movement; or a loss near $16 \cdot (1 - 1/C)$ and NaN after the first plain gradient step |
| On the spiral | 0.01 fails from three hidden layers under plain gradient descent and from five or six under Adam; with one, Glorot is not behind He |

---

## Common pitfalls

1. **Carrying the fixed 0.01 into a deeper network.** Each 64-neuron ReLU layer keeps 0.0032 of the signal and of the gradient. The run starts at $\ln C$ and stays there without an error.
2. **Using the Glorot scale for a deep ReLU stack.** It was derived for an activation that is linear near zero, and ReLU halves the mean square it lets through.
3. **Expecting NaN on the first loss of an exploded network.** The stable softmax and the clipped loss keep the first loss finite, near 10.7 for three classes. The NaN comes with the first large update.
4. **Rerunning an earlier post's script with the new default.** The default is `"he"`; the results of posts 04 to 32 belong to `init="small"`.
5. **Comparing two schemes on one seed.** On seed 0 the 0.01 scale had the higher training accuracy than He on the spiral, and on 19 other seeds it did not.

---

## Further reading

- Glorot, X. and Bengio, Y., *"Understanding the Difficulty of Training Deep Feedforward Neural Networks"* (AISTATS, 2010). The forward and backward conditions and the compromise between them.
- He, K., Zhang, X., Ren, S., and Sun, J., *"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"* (ICCV, 2015). The derivation for rectifiers.
- LeCun, Y., Bottou, L., Orr, G. B., and Müller, K.-R., *"Efficient BackProp"* (Neural Networks: Tricks of the Trade, 1998).
- Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, section 8.4 (MIT Press, 2016).

Full citations are in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

- **[Post 34 - Sigmoid and binary cross-entropy](../34-sigmoid-and-binary-cross-entropy/index.md):** the output layer and the loss for two classes, used by the project whose initialisation failure section 7 quotes.
- **[Post 27 - Adam optimiser](../27-adam-optimiser/index.md):** the normalisation by gradient size that lets a small initial scale survive a few more layers.
