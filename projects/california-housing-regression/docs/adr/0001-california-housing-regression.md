# 0001. Assemble the series' classes as they are, with a linear output, and freeze the baseline

- Status: accepted
- Date: 2026-10-05

## Context

Every post in the series classifies. The project exists to answer one question the posts leave open: does the forward and backward machinery survive a change of loss? That question is only answered if the classes here are the series' classes, changed in the two places a regression needs and nowhere else.

The series derives no mean squared error. `nn-008` names this project as the place where the reader meets it, so the loss class has to be written here, in the style of the classes around it.

Four things in the code look like mistakes to a maintainer who does not know this, and each is deliberate: the class names break Python naming conventions (`Layer_Dense`), the weights start at `0.01 * randn` although `nn-033` shows a better rule, the output layer carries no L2 penalty although the two hidden layers do, and the network has no activation after its last layer.

## Decision

1. `nn.py` holds `Layer_Dense`, `Activation_ReLU`, `Optimizer_Adam` and `regularization_loss` under the series' names with the series' arithmetic. The only additions are argument checks. Improvements to a class belong in the post that derives it first.
2. `Loss_MSE` is the one class with no post behind it. It follows the interface of the series' losses: `forward` returns the mean loss, `backward` leaves the gradient in `dinputs`. It averages over samples and outputs, so its gradient is $2(\hat{y} - y)/(NK)$, and it reshapes a vector of targets to the column of predictions, because without the reshape NumPy would broadcast $N$ predictions against $N$ targets into $N^2$ differences.
3. The output layer is `Layer_Dense(64, 1)` with nothing after it. A sigmoid or a softmax would bound the prediction; a regression target is not bounded.
4. The model and its training configuration are constants in `model.py`, `train.py` and `data.py`, not options: layer sizes 8, 64, 64, 1; L2 strength 0.0001 on the two hidden layers' weights and none elsewhere; Adam with learning rate 0.01 and decay 0.0001; 200 epochs; mini-batches of 256; a test fraction of 0.2; seed 0. The command line exposes the epoch count, the batch size and the seed for experiments, and no flag for the rest.
5. Any change that alters the weights of the seed 0 run is a breaking change. It needs a new major version and a new evaluation. A restructuring that claims to change nothing must show the same weight fingerprint for the documented run before it is merged.

## Consequences

- The naming rules of the linter are not enabled, and the classes are not rewritten in a more idiomatic style.
- The network keeps an initialisation and a penalty layout that the later posts of the series would not choose. The README says so under its limits.
- A reader who wants to try another width, rate or optimiser edits a constant and accepts that the result is no longer the documented one. This is a small cost, and it keeps every published figure tied to one configuration.
- `tests/test_model.py` pins the architecture, the parameter count, the penalty layout, the linear output and the order of the initial draws; `tests/test_train.py` pins a short run recorded from the scripts as they stood before version 1.0.0. A change that breaks the baseline fails a test before it reaches the evaluation.
