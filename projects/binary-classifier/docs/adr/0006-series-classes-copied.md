# 0006. The series' classes are copied into the project under their own names

- Status: accepted
- Date: 2026-10-05

## Context

The series has no library: its classes are written in the posts, and each project carries the ones it uses. A project must also be self-contained, with its own dependency file and nothing imported from a sibling directory, so that it can be split into a repository of its own without changing a line.

The classes are named as the posts name them: `Layer_Dense`, `Activation_ReLU`, `Optimizer_Adam`, with attributes such as `dweights` set by one method and read by another, and optimiser state attached to the layer from outside. Python's naming conventions and a type checker would each want that code changed, and a formatter lays it out differently from the page of the post.

## Decision

`src/binary_classifier/nn.py` holds the project's own copy of the classes, with the names, the signatures, and the arithmetic of the posts. The post `nn-034` shows the code of `Activation_Sigmoid_Loss_BinaryCrossentropy` as it stands in this file, so the two are kept the same, token for token: a change to the class is a change to the post.

Everything the posts do not define lives outside that file: `Network` and the weights file in `model.py`, the generator in `data.py`, the commands in `train.py` and `evaluate.py`. Input checking belongs there, at the command line and at the file, and not inside the copied classes.

The checks follow from this:

- Lint is `ruff check` with the rule sets the series' projects share (E, F, W, I, B, UP, SIM, C4, PT, RUF). The naming rules are left out because the class names are the posts'.
- The formatter is enforced on every file, `nn.py` included, as in the other projects. `ruff format` changes line breaks, spacing and quote marks and nothing else, so the copied classes keep the posts' tokens in a different layout; one layout for all the code was preferred to an exemption for one file.
- No type checker is run. The classes create their attributes outside `__init__`, and annotating them would mean rewriting them.
- The tests carry the weight a type checker would: every class is checked against hand-computed values and every backward pass against numerical gradients, and the test command fails below 95 percent coverage. Coverage of `src/` was 100 percent, lines and branches, at 1.0.0.

## Consequences

- A reader can hold the post beside `nn.py` and find the same statements in the same order, with some lines broken differently.
- A fix to a class has to be made in the post and in each project that copies it. The series accepts that cost in place of a shared library.
- `Layer_Dense` seeds nothing itself and draws from NumPy's global generator, as in the posts. The training command seeds it; code that builds a `Network` directly must call `numpy.random.seed` first if it wants a repeatable model.
