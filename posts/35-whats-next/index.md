# Part 35 · What to read after this series

> **TL;DR.** Everything past this series (convolutional layers, recurrent layers, transformers, batch normalisation, residual connections, modern training tricks) is *additional layer types* and *additional training tricks* bolted onto the same forward-pass / backward-pass / optimiser-step skeleton posts 1–34 already established. This post is a structured reading list that names each topic, explains what it adds, and points to the canonical paper plus a from-scratch tutorial.
>
> **After reading this you will be able to:**
> - Place each major architectural family (CNN, RNN, transformer) on a "what does it add to a vanilla MLP" map.
> - Name the canonical paper and an implementation-from-scratch tutorial for each.
> - Decide what to study next based on the kind of problem you want to work on (vision, sequence, language, etc.).

![A map of the modern deep-learning toolkit organised by what it adds to the from-scratch MLP built in this series: new layer types (conv, RNN, attention), new training tricks (batch norm, residual, schedulers), new problem framings (self-supervised, transfer, RL).](diagrams/01-whats-next-map.svg)
*Three columns: new layers, new training infrastructure, new problem framings. Everything fits on top of the forward + backward + optimiser stack from posts 1–34.*

---

## 1. What you already have

Posts 1–34 leave a fully working multi-layer perceptron (MLP, a stack of dense layers) library plus four projects' worth of evidence that it works. The aim of this reading list is to make the post-series learning path explicit, rather than leaving each reader to figure out what comes next alone. A short stocktake of the toolkit:

- **Layer types:** dense (`Layer_Dense`), ReLU, sigmoid, softmax, dropout.
- **Losses:** categorical cross-entropy, binary cross-entropy, mean squared error.
- **Optimisers:** vanilla SGD, decay, momentum, AdaGrad, RMSProp, Adam.
- **Regularisation:** L1, L2, dropout.
- **Training infrastructure:** mini-batching, train/test split, k-fold CV, He init.
- **Worked projects:** MNIST, two-moons binary, Fashion-MNIST, California housing regression.

The forward-pass / backward-pass / optimiser-step pattern from post 21 generalises: every new layer type, no matter how exotic, has a `.forward(...)` and a `.backward(...)` that fit into the same training loop. Every new optimiser has the same `pre_update_params → update_params → post_update_params` contract from post 23. The skeleton is fixed.

What changes past this series is the *content* slotted into that skeleton. The next sections are organised by what the addition is.

---

## 2. New layer types

### 2.1. Convolutional layers (for images, audio, spatial data)

A dense layer treats inputs as a flat vector. Pixels at positions (5, 5) and (5, 6) are *independent* features as far as `Layer_Dense` is concerned: there is no built-in notion that they sit next to each other.

A **convolutional layer** fixes this. It applies the same small set of weights (a *filter*) at every position of the input in a sliding-window pattern. The output of one filter at one position is a single weighted sum of a small patch of the input. The same filter is reused across the whole image, drastically reducing parameter count and exploiting the fact that the same pattern (an edge, a corner, a texture) can appear anywhere.

What to read:

- **LeCun, Y. et al.**, *"Gradient-Based Learning Applied to Document Recognition"* (Proceedings of the IEEE, 1998). The LeNet paper. Still the canonical introduction.
- **Stanford CS231n notes**, ["Convolutional Neural Networks"](http://cs231n.github.io/convolutional-networks/). The best free explainer that goes from "what is convolution" to "modern architectures" without skipping steps.
- **From-scratch tutorial:** Andrej Karpathy's [`micrograd`](https://github.com/karpathy/micrograd) and his "neural networks: zero to hero" YouTube series both build convolutions on top of a from-scratch autograd (an engine that records operations and computes their gradients automatically), which is the natural next step after this series.

### 2.2. Recurrent layers (for sequences, time series, text)

A dense layer assumes inputs are independent samples. For a sequence (say, the words of a sentence) that's wrong: word $t$ depends on what came before.

A **recurrent layer** maintains an internal hidden state that gets updated at each timestep, blending the previous hidden state with the current input. The same weights are reused at every timestep, so a 100-word sentence and a 5-word sentence use the same parameters. **LSTM** and **GRU** are gated variants of the vanilla RNN that solve the vanishing-gradient problem for long sequences.

What to read:

- **Hochreiter, S. and Schmidhuber, J.**, *"Long Short-Term Memory"* (Neural Computation, 1997). The LSTM paper. Notoriously dense; pair it with the Olah blog post below.
- **Christopher Olah**, ["Understanding LSTM Networks"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). The universally-recommended intuitive explainer.
- **From-scratch tutorial:** Karpathy's [char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), a 110-line numpy implementation of a character-level RNN that learns to generate Shakespeare. The same skeleton you have, just with a hidden state passed forward through time.

### 2.3. Attention and transformers (the modern default)

Recurrent networks process sequences one timestep at a time, which is hard to parallelise. **Attention** is a different mechanism: at every output position, the network looks at *all* input positions weighted by learned similarity scores. No recurrence, fully parallel, scales to long sequences.

The **transformer** architecture (Vaswani et al., 2017) builds the entire model out of attention plus dense layers. It has replaced RNNs almost completely for language modelling and is the basis of every modern LLM (GPT, Claude, Llama, Gemini).

What to read:

- **Vaswani, A. et al.**, *"Attention Is All You Need"* (NeurIPS, 2017). The original transformer paper. Read it after the explainers below.
- **Jay Alammar**, ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/). The standard visual introduction.
- **Andrej Karpathy**, ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY). A 2-hour video building a working GPT in ~300 lines of PyTorch. The from-scratch version of "what does a transformer actually do".

---

## 3. New training infrastructure

### 3.1. Batch normalisation

Post 33 showed that activation variance through a deep network is fragile: bad initialisation kills training. **Batch normalisation** (Ioffe & Szegedy, 2015) sidesteps the problem by re-normalising activations to zero mean and unit variance *during* the forward pass, at every layer. With batch norm, networks become much less sensitive to initialisation and learning rate.

What to read:

- **Ioffe, S. and Szegedy, C.**, *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"* (ICML, 2015). "Internal covariate shift" is the paper's name for the way each layer's input distribution keeps shifting as the layers below it update.
- **Santurkar, S. et al.**, *"How Does Batch Normalization Help Optimization?"* (NeurIPS, 2018). The follow-up that argues the original "covariate shift" explanation was wrong and batch norm helps for a different reason. Required for understanding the modern view.

**Layer normalisation** is the variant transformers use; it normalises over the feature dimension rather than the batch dimension. Same idea, slightly different reduction axis.

### 3.2. Residual connections

Past a certain depth (~20 layers), even well-initialised networks stop training. **Residual connections** (He et al., 2015) add a "skip" path that bypasses each layer: $\text{output} = \text{layer}(x) + x$. The skip preserves the gradient signal regardless of layer depth, enabling 50- and 100-layer networks that simply did not train before ResNet.

What to read:

- **He, K. et al.**, *"Deep Residual Learning for Image Recognition"* (CVPR, 2016). The ResNet paper.

Residuals are now standard in every modern architecture, from CNNs to transformers.

### 3.3. Learning-rate schedulers beyond $1/(1+dt)$

Post 23 introduced one decay schedule. Production training uses fancier ones:

- **Cosine decay:** smooth half-cosine from $\alpha_0$ to a small floor.
- **Cosine with warm restarts:** cosine that periodically resets to $\alpha_0$ (Loshchilov & Hutter, 2017).
- **One-cycle policy:** ramp up to a peak, then ramp down (Smith, 2018). Used in fast.ai's training recipes.
- **Linear warmup:** start near zero, ramp up over the first few thousand steps. Standard for transformers.

Pick one for the problem; the differences are real but second-order compared to the choice of optimiser.

### 3.4. Mixed-precision training

Modern GPUs run float16 / bfloat16 math 4-8× faster than float32. **Mixed precision** runs the forward and backward in low precision while keeping the optimiser state in float32 (so accumulation doesn't lose precision). PyTorch's `torch.cuda.amp` and JAX's `jax.lax.precision` are the standard interfaces. This is the single biggest GPU-utilisation win in modern training.

### 3.5. Distributed training

For models that don't fit on one GPU or datasets that take days on a single device, distributed training spreads the work across many devices. The two main paradigms:

- **Data parallel:** every device has the same model, processes different batches, gradients synchronised at the end of each step. The easy case; works for anything that fits on one device.
- **Model parallel:** split the model itself across devices. Required for very large models (GPT-scale).

Production frameworks (PyTorch's `DistributedDataParallel`, JAX's `jax.pmap`) handle most of the bookkeeping.

---

## 4. New problem framings

### 4.1. Self-supervised learning

Supervised learning (this series) needs labelled data. **Self-supervised** learning constructs its own labels from the input itself: predict the next word, predict the masked pixels, predict whether two image crops come from the same image. The benefit: unlimited data (every wiki article, every photo on the internet) at the cost of zero labels.

What to read:

- **Devlin, J. et al.**, *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (NAACL, 2019). Masked-language-modelling.
- **Radford, A. et al.**, *"Language Models are Unsupervised Multitask Learners"* (GPT-2, 2019). Next-token prediction at scale.
- **He, K. et al.**, *"Masked Autoencoders Are Scalable Vision Learners"* (CVPR, 2022). The vision analogue.

Self-supervised pretraining followed by supervised fine-tuning is now the dominant paradigm for both vision and language.

### 4.2. Transfer learning and fine-tuning

Most production models are not trained from scratch. The recipe is: take a pretrained model (often self-supervised, often huge), then **fine-tune** it on your specific task with a much smaller labelled dataset. The pretrained model handles general features; fine-tuning specialises.

For language: **LoRA** and **parameter-efficient fine-tuning** (PEFT) techniques let you fine-tune a multi-billion-parameter model by training only a few million parameters' worth of "adapters". Standard production technique.

What to read:

- **Hu, E. J. et al.**, *"LoRA: Low-Rank Adaptation of Large Language Models"* (ICLR, 2022).
- **Hugging Face PEFT library docs:** the practical entry point.

### 4.3. Reinforcement learning

So far the series has covered **supervised** learning: each input has a known correct output. **Reinforcement learning** is different: an agent takes actions in an environment, occasionally receives a reward, and learns a policy that maximises future reward. The math is different (no per-sample target, no cross-entropy); the gradient comes from a *policy gradient* derivation, which differentiates the expected reward with respect to the action probabilities rather than against a fixed label.

What to read:

- **Sutton, R. S. and Barto, A. G.**, *Reinforcement Learning: An Introduction*, 2nd edition (MIT Press, 2018). The canonical textbook. Free PDF at [incompleteideas.net](http://incompleteideas.net/book/the-book.html).
- **Mnih, V. et al.**, *"Playing Atari with Deep Reinforcement Learning"* (NeurIPS Workshop, 2013). The paper that connected RL to deep networks.
- **OpenAI Spinning Up** ([spinningup.openai.com](https://spinningup.openai.com/)): a structured curriculum for going from "what is RL" to "implementing PPO from scratch".

### 4.4. Diffusion models (for generation)

For image / audio / video *generation*, the dominant modern technique is **diffusion**: train a model to reverse a noise-corruption process. Stable Diffusion, Midjourney, OpenAI's DALL-E 3, and Sora are all diffusion-based.

What to read:

- **Ho, J., Jain, A., and Abbeel, P.**, *"Denoising Diffusion Probabilistic Models"* (NeurIPS, 2020). The modern DDPM paper.
- **Lilian Weng's blog**, ["What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). The standard expository explainer.

---

## 5. Three practical books

If you want a single book to sit alongside this series, pick one of:

- **Goodfellow, Bengio, Courville**, *Deep Learning* (MIT Press, 2016). The canonical textbook. Free at [deeplearningbook.org](https://www.deeplearningbook.org/). Read the second half (chapters 6 onward); the first half overlaps with this series.
- **Howard, J. and Gugger, S.**, *Deep Learning for Coders with fastai and PyTorch* (O'Reilly, 2020). The most-practical-modern alternative. PyTorch-first; the from-scratch ethic of this series ported to a real production framework.
- **Bishop, C. M.**, *Deep Learning: Foundations and Concepts* (Springer, 2024). The modern successor to Bishop's classic *PRML*. The most mathematically rigorous of the three.

---

## 6. Three frameworks to learn next

After implementing the basics from scratch, the production frameworks become much easier to read because the reader already knows what each method is doing under the hood:

- **PyTorch:** the dominant research framework. Most papers ship PyTorch code; most modern tutorials assume PyTorch. Start here.
- **JAX:** Google's research framework. Functional, very fast on TPUs, becoming popular for RL and large-scale training. Worth learning second.
- **TensorFlow / Keras:** still the dominant production framework in many industries. Worth knowing for work in a TF shop.

The from-scratch series prepares the reader for all three. Knowing what a `Layer_Dense` does and what `backward` should return means the framework just spares the writing of it.

---

## 7. Reading checklist

A condensed list, ranked by "biggest payoff per hour" for someone who just finished the series:

| Priority | Topic | Best source |
|:---:|---|---|
| 1 | Read Karpathy's "Zero to Hero" videos | [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) |
| 2 | Learn PyTorch basics; reimplement [project 01](../../projects/01-mnist-from-scratch/README.md) in PyTorch | [pytorch.org tutorials](https://pytorch.org/tutorials/) |
| 3 | Read the CS231n CNN notes | [cs231n.github.io](http://cs231n.github.io/convolutional-networks/) |
| 4 | Build a transformer from scratch following Karpathy's GPT video | [Karpathy "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 5 | Read the BatchNorm paper + Santurkar follow-up | (see §3.1) |
| 6 | Read Sutton & Barto chapters 1-6 for RL fundamentals | [incompleteideas.net/book](http://incompleteideas.net/book/the-book.html) |
| 7 | Pick one of: image generation (diffusion), language fine-tuning (LoRA), or RL (PPO) and build something | various |

The from-scratch foundation built across posts 1–34 makes every item on this list easier. None of them are conceptually harder than backprop through a softmax + cross-entropy layer; they are just more layers, more tricks, and more compute.

---

## 8. Closing note

This is the last lecture in the series.

Across 35 posts the series went from "a neuron is a dot product plus a bias" (post 1) to "Adam is momentum's EMA over $g$ plus RMSProp's EMA over $g^2$ plus a bias-correction term" (post 27) to "here's what to read next" (this post). The four projects supplied working code at production scale. The series tries to leave nothing important hidden: every line in the final `Optimizer_Adam` class was derived from the chain rule the lectures derived from first principles.

If you build something with what you've learned, the project layout in [projects/README.md](../../projects/README.md) is happy to host a project 05.

---

## Common pitfalls

A few traps that catch people working through a list like this:

- **Reading the paper before the explainer.** The LSTM and transformer papers are notoriously dense; the Olah and Alammar explainers exist precisely to be read first.
- **Jumping straight to transformers.** Attention is far easier to follow after understanding why recurrence struggles with long sequences, so resist skipping §2.2.
- **Reaching for distributed training prematurely.** Multi-GPU and model-parallel setups add a lot of bookkeeping; a single GPU (or even CPU) is enough for everything on the reading checklist.
- **Collecting frameworks instead of building.** Learning PyTorch, JAX, and TensorFlow at once teaches none of them; pick PyTorch, build one project, then branch out.
- **Treating self-supervised, transfer, and reinforcement learning as interchangeable.** They solve different problems; the right one depends on whether labels exist and whether the task is interactive.
- **Skipping the from-scratch reimplementation.** Re-coding a project in PyTorch is what makes the framework's abstractions click; reading tutorials alone rarely does.

## Further reading

The links in §§2–4 above are the primary sources. A few extra organisational resources:

- **Distill.pub** ([distill.pub](https://distill.pub/)): beautiful interactive explainers for selected papers. The "Visualizing Neural Networks" and "Building Blocks of Interpretability" articles are particularly good.
- **Papers With Code** ([paperswithcode.com](https://paperswithcode.com/)): every paper with its open-source implementation linked. The standard place to find recent SOTA.
- **arXiv-sanity-lite** ([arxiv-sanity-lite.com](https://arxiv-sanity-lite.com/)): Andrej Karpathy's filtered arXiv reader. Personalised paper recommendations based on what you read.
- **The Stanford / MIT / Berkeley public course archives:** CS231n (Stanford, vision), CS224n (Stanford, NLP), 6.S191 (MIT, deep learning) are all freely available with lecture videos and assignments.

Full citations for everything in this post live in [REFERENCES.md](../../REFERENCES.md).

---

## What to read next

You've finished the series. The next move is to *build*: pick one of the projects in [projects/](../../projects/) and extend it with something you learned from §§2–4. Or start a new project on a problem you actually care about.

If you want a structured next step rather than a free build:

1. Re-implement one of the four projects in PyTorch. See what the framework gives you for free.
2. Build a small CNN (one conv layer + dense head) on MNIST or Fashion-MNIST. Compare to project 01 / 03.
3. Build a character-level RNN on a text corpus you care about. Compare to Karpathy's char-rnn.
4. Pick a recent paper you find interesting from Papers With Code, read it, and re-implement the headline result.

> *Last lecture of 35. Thanks for reading.*
