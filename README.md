# Neural Networks from Scratch

A series of 35 posts that builds a neural network by hand in Python and NumPy, with no framework and no automatic differentiation. Every piece of a multi-layer perceptron is written from first principles: the neuron and the dense layer, the forward pass, ReLU and softmax, categorical cross-entropy, the calculus that backpropagation needs, the backward pass of every component, six optimisers from plain gradient descent to Adam, generalisation and validation, L1 and L2 penalties, dropout, mini-batching, weight initialisation, and the sigmoid with binary cross-entropy. Four projects then put the same classes to work on real data.

It is the entry point of the family's neural-network series: Convolutional Neural Networks from Scratch and Recurrent Neural Networks from Scratch both start where this one ends.

## What you will build

- `Layer_Dense`, `Activation_ReLU`, and `Activation_Softmax`, each with a forward and a backward pass, and the loss classes that score them.
- `Activation_Softmax_Loss_CategoricalCrossentropy`, whose backward pass reduces to $(\hat{\mathbf{y}} - \mathbf{y})/N$, derived step by step from the chain rule.
- Six optimisers on one shared contract: gradient descent, learning-rate decay, momentum, AdaGrad, RMSProp, and Adam.
- L1 and L2 regularisation and `Layer_Dropout`, with the switch between training and evaluation in place.
- A mini-batch training loop, He and Glorot initialisation, and a sigmoid output with binary cross-entropy for two-class problems.

## The posts

### Part I - Foundations

After this part you can code a neuron and a layer, predict the shape of any `np.dot`, and apply NumPy's reduction and broadcasting rules without running the code.

1. [Neurons and layers](posts/01-neurons-and-layers/index.md)
2. [NumPy and the dot product](posts/02-numpy-and-the-dot-product/index.md)
3. [Stacking layers and the forward pass](posts/03-stacking-layers-and-the-forward-pass/index.md)
4. [The Dense layer class and spiral data](posts/04-dense-layer-class-and-spiral-data/index.md)
5. [Array summation, keepdims, and broadcasting](posts/05-array-summation-keepdims-and-broadcasting/index.md)

### Part II - Activations and forward pass

After this part you have a complete forward pass from inputs to class probabilities.

6. [Activation functions: ReLU and Softmax](posts/06-activation-functions-relu-and-softmax/index.md)
7. [Coding the complete forward pass](posts/07-coding-the-complete-forward-pass/index.md)

### Part III - Loss and optimisation

After this part you can score a batch with cross-entropy and accuracy, and say why random search cannot train a network.

8. [Loss: categorical cross-entropy](posts/08-loss-categorical-cross-entropy/index.md)
9. [Introduction to optimisation](posts/09-introduction-to-optimisation/index.md)

### Part IV - Calculus for backpropagation

After this part you can differentiate the functions a network uses, read a gradient component by component, and apply the chain rule to a composition of any depth. No calculus is assumed before it.

10. [Derivatives, partial derivatives, and gradients](posts/10-derivatives-partial-derivatives-and-gradients/index.md)
11. [The chain rule](posts/11-the-chain-rule/index.md)

### Part V - Backpropagation

After this part every component has a backward pass, and you can derive the three dense-layer gradients and the combined softmax and cross-entropy gradient yourself.

12. [Backpropagation through a single neuron](posts/12-backprop-through-a-single-neuron/index.md)
13. [Backpropagation through a layer of neurons](posts/13-backprop-through-a-layer/index.md)
14. [Matrices in backpropagation](posts/14-matrices-in-backpropagation/index.md)
15. [Gradients with respect to inputs](posts/15-gradients-with-respect-to-inputs/index.md)
16. [Coding backpropagation](posts/16-coding-backpropagation/index.md)
17. [Backpropagation through activation functions](posts/17-backpropagation-through-activation-functions/index.md)
18. [Backpropagation through the loss function](posts/18-backpropagation-through-the-loss-function/index.md)
19. [Softmax derivatives and the combined backward pass](posts/19-softmax-derivatives-and-the-combined-backward-pass/index.md)
20. [Assembling full backpropagation](posts/20-assembling-full-backpropagation/index.md)
21. [Coding the full backpropagation](posts/21-coding-the-full-backpropagation/index.md)

### Part VI - Optimisers

After this part you have six optimisers and can say what each one fixes in the one before it.

22. [Gradient-descent optimiser](posts/22-gradient-descent-optimiser/index.md)
23. [Learning-rate decay](posts/23-learning-rate-decay/index.md)
24. [Momentum](posts/24-momentum/index.md)
25. [AdaGrad](posts/25-adagrad/index.md)
26. [RMSProp](posts/26-rmsprop/index.md)
27. [Adam](posts/27-adam-optimiser/index.md)

### Part VII - Generalisation and regularisation

After this part you can measure the gap between training and test performance, choose hyperparameters without touching the test set, and add L1, L2, or dropout to close the gap.

28. [Generalization and testing](posts/28-generalization-and-testing/index.md)
29. [Validation and hyperparameter tuning](posts/29-validation-and-hyperparameter-tuning/index.md)
30. [L1 and L2 regularisation](posts/30-l1-and-l2-regularisation/index.md)
31. [Dropout](posts/31-dropout/index.md)

### Part VIII - Practical training and extensions

After this part you train with mini-batches, initialise for the activation you use, solve two-class problems, and know what convolution, recurrence, attention, and normalisation each add to this stack.

32. [Mini-batching](posts/32-mini-batching/index.md)
33. [Weight initialisation](posts/33-weight-initialisation/index.md)
34. [Sigmoid and binary cross-entropy](posts/34-sigmoid-and-binary-cross-entropy/index.md)
35. [What to read after this series](posts/35-whats-next/index.md)

## The projects

- [MNIST from scratch](projects/mnist-from-scratch/README.md) (`nn-p01`): handwritten digits with a two-hidden-layer network, Adam, L2, dropout, and mini-batches; from seed 0 it classifies 9,800 of the 10,000 test images correctly, 98.00 percent.
- [Binary classifier on two moons](projects/binary-classifier/README.md) (`nn-p02`): sigmoid and binary cross-entropy on a two-dimensional problem whose decision boundary can be drawn; from seed 0 it classifies all 200 held-out points correctly at noise 0.1.
- [Fashion-MNIST with the same network](projects/fashion-mnist/README.md) (`nn-p03`): the MNIST network, unchanged, on clothing images, and where it fails; from seed 0 it reaches 87.12 percent, 8,712 of the 10,000 test images.
- [California housing regression](projects/california-housing-regression/README.md) (`nn-p04`): the first regression, with mean squared error and a standardised target; from seed 0 it reaches $R^2 = 0.8231$ on the test fold, with an RMSE of 49,222 dollars.

## Before you start

You need to be able to write a Python list, call a function, and use a `for` loop. Nothing else is assumed: NumPy is introduced from its first function, and the calculus (derivatives, partial derivatives, the chain rule) is taught in Part IV from zero. Readers who already use a framework can start at Part II, but should not skip Part V. Everything runs on a laptop CPU.

## How a post is organised

Each post opens with a short summary, the prerequisites, and what you will be able to do afterwards, then works through numbered sections from the question to the code, and closes with what can go wrong, the common pitfalls, further reading, and what to read next. The posts trace the shape of each array before writing the code that produces it. The series website shows each post's questions, with hints and worked explanations, beside the post.

## What is measured

- From post 04 to post 31 every network is run on the same data: the spiral dataset, three classes of 100 points in two dimensions, from the `nnfs` package with its fixed seed. Post 09 adds the package's vertical dataset once, as an easy contrast.
- Part VI trains all six optimisers on one setup (the spiral data, a $2 \to 64 \to 3$ network, 10,001 epochs, the same initial weights), so the rows of its comparison differ only in the optimiser. On the documented seed plain gradient descent ends at 64.7 percent training accuracy, momentum at 95.7, AdaGrad at 84.0, RMSProp at 90.0, and Adam at 96.3; other seeds order the optimisers differently, and over ten seeds no ranking of the last four holds (post 27).
- Part VII measures each regulariser by the gap between training accuracy and accuracy on a fresh draw from the same spiral generator.
- Each project's README gives the commands that train and evaluate it and reports the test result those commands print.

## The code

Pure Python and NumPy, plus the `nnfs` package, which supplies `spiral_data` and a fixed seed. There is no library to install: the classes are written in the posts, and each project carries its own copy in `src/<package>/nn.py`.

```text
posts/NN-slug/   the post (index.md), its figures, and the scripts it runs (snippets/)
projects/        four applied projects, each a package under src/ with tests, docs, and a locked environment
poster/          the whole series on one page, light and dark
```

To run a post's script and a project:

```bash
python -m pip install numpy nnfs
python posts/04-dense-layer-class-and-spiral-data/snippets/dense_layer.py
cd projects/mnist-from-scratch
uv sync --frozen
uv run python scripts/download_mnist.py
uv run python -m mnist_from_scratch.train
uv run python -m mnist_from_scratch.evaluate
```

The projects need [uv](https://docs.astral.sh/uv/), which fetches Python 3.13 and the locked packages; NumPy, pinned at 2.3.5, is the only third-party package a project imports. Nothing is downloaded implicitly: the MNIST, Fashion-MNIST, and California housing projects each fetch their data with one script, which keeps a file only if its SHA-256 is the expected one, and the two-moons data are generated. No project uses scikit-learn. The [projects index](projects/README.md) lists the four with their results.

## Reference pages

- [Glossary](GLOSSARY.md): the terms the series defines.
- [Cheatsheet](CHEATSHEET.md): the formulas, shapes, and code on one page.
- [Notation guide](notation_guide.md): the symbols in the maths and the names in the code.
- [References](REFERENCES.md): the books, papers, and documentation the posts cite.
- [Poster](poster/one-page-of-neural-networks-from-scratch.svg): the series in eight panels on one A2 sheet.

## Corrections and feedback

This repository is the published source of the series and is maintained by its author. It does not take pull requests or issues. If you find a mistake or have a suggestion, send it through the series website.

## Licence

Prose, questions, and figures are licensed under CC BY 4.0. Code is licensed under MIT. See [LICENSE](LICENSE).
