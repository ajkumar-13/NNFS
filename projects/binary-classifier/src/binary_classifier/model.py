"""The 2-16-16-1 network and its weights file.

Network wires the classes of binary_classifier.nn into the one network this
project trains. save_weights and load_weights write and read it as a
NumPy .npz archive of plain numeric arrays, which is loaded with pickling
switched off, so reading a weights file never runs code from the file.
"""

import zipfile
import zlib
from pathlib import Path

import numpy as np

from binary_classifier.nn import (
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
    Layer_Dense,
    regularization_loss,
)

LAYER_SIZES = (2, 16, 16, 1)
L2_LAMBDA = 1e-4
INIT_CHOICES = ("he", "xavier", "small")
FORMAT_VERSION = 1
DEFAULT_WEIGHTS = "moons_weights.npz"

_WEIGHT_SHAPES = {
    "dense1_weights": (2, 16),
    "dense1_biases": (1, 16),
    "dense2_weights": (16, 16),
    "dense2_biases": (1, 16),
    "dense3_weights": (16, 1),
    "dense3_biases": (1, 1),
}
_SPLIT_KEYS = ("X_train", "y_train", "X_test", "y_test")
_CONFIG_KEYS = ("format_version", "init", "noise", "n_samples", "seed", "epochs")
_TRAIN_HINT = "Train one with: uv run python -m binary_classifier.train"
# What NumPy and zipfile raise on a file that is not a readable archive of numeric arrays.
_UNREADABLE = (
    OSError,
    ValueError,
    EOFError,
    MemoryError,
    NotImplementedError,
    RuntimeError,
    zipfile.BadZipFile,
    zlib.error,
)


class WeightsError(Exception):
    """A weights file is missing, unreadable, or not what this project wrote."""


class Network:
    """Dense(2, 16), ReLU, Dense(16, 16), ReLU, Dense(16, 1), sigmoid with binary cross-entropy.

    The two hidden layers carry an L2 penalty of L2_LAMBDA on their weights.
    The three dense layers draw their weights from NumPy's global generator in
    the order dense1, dense2, dense3, so np.random.seed fixes the model.
    """

    def __init__(self, init="he"):
        if init not in INIT_CHOICES:
            raise ValueError(f"init must be one of {', '.join(INIT_CHOICES)}; got {init!r}")
        self.init = init
        self.dense1 = Layer_Dense(2, 16, init=init, weight_regularizer_l2=L2_LAMBDA)
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(16, 16, init=init, weight_regularizer_l2=L2_LAMBDA)
        self.activation2 = Activation_ReLU()
        self.dense3 = Layer_Dense(16, 1, init=init)
        self.loss_activation = Activation_Sigmoid_Loss_BinaryCrossentropy()

    @property
    def dense_layers(self):
        return (self.dense1, self.dense2, self.dense3)

    def parameter_count(self):
        return sum(layer.weights.size + layer.biases.size for layer in self.dense_layers)

    def forward(self, X):
        """Run the network on X of shape (N, 2); return the logits, shape (N, 1)."""
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        return self.dense3.output

    def data_loss(self, X, y):
        """Forward pass and mean binary cross-entropy; leaves the probabilities cached."""
        return self.loss_activation.forward(self.forward(X), y)

    def regularization_loss(self):
        return (
            regularization_loss(self.dense1)
            + regularization_loss(self.dense2)
            + regularization_loss(self.dense3)
        )

    def backward(self, y):
        """Backward pass from the probabilities cached by data_loss to every gradient."""
        self.loss_activation.backward(self.loss_activation.output, y)
        self.dense3.backward(self.loss_activation.dinputs)
        self.activation2.backward(self.dense3.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def predict_proba(self, X):
        """Probability of class 1 for each row of X, shape (N,)."""
        sigmoid = Activation_Sigmoid()
        sigmoid.forward(self.forward(X))
        return sigmoid.output.ravel()

    def predict(self, X):
        """Predicted class for each row of X: 1 where the probability is at least 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(np.int64)


def npz_path(path, what="weights"):
    """Return path as a Path if its name can be a .npz file of this project; else raise."""
    path = Path(path)
    if path.suffix == ".pkl":
        raise WeightsError(
            f"{path} is a pickle file, the weights format this project used before version "
            "1.0.0. Pickle files are no longer read or written, because loading one can run "
            "code. Use a name ending in .npz, and replace an old file by training again: "
            "uv run python -m binary_classifier.train"
        )
    if path.suffix != ".npz":
        raise WeightsError(f"the {what} name must end in .npz, got {path}")
    return path


def output_path(path, what="weights"):
    """Return path as a Path if a .npz file can be written there; else raise WeightsError."""
    path = npz_path(path, what)
    if not path.parent.is_dir():
        raise WeightsError(
            f"the directory {path.parent} does not exist; create it or choose another path "
            f"for the {what}"
        )
    return path


def save_weights(path, model, split, config):
    """Write the weights, the train and test split, and the run's configuration to path.

    split is (X_train, y_train, X_test, y_test). config is a mapping with the
    keys noise, n_samples, seed, and epochs.
    """
    path = output_path(path)
    X_train, y_train, X_test, y_test = split
    arrays = {
        "dense1_weights": model.dense1.weights,
        "dense1_biases": model.dense1.biases,
        "dense2_weights": model.dense2.weights,
        "dense2_biases": model.dense2.biases,
        "dense3_weights": model.dense3.weights,
        "dense3_biases": model.dense3.biases,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "format_version": np.array(FORMAT_VERSION, dtype=np.int64),
        "init": np.array(model.init),
        "noise": np.array(config["noise"], dtype=np.float64),
        "n_samples": np.array(config["n_samples"], dtype=np.int64),
        "seed": np.array(config["seed"], dtype=np.int64),
        "epochs": np.array(config["epochs"], dtype=np.int64),
    }
    try:
        np.savez(path, **arrays)
    except OSError as err:
        raise WeightsError(f"could not write the weights file {path}: {err}") from err
    return path


def _read_arrays(path):
    # allow_pickle=False is the point of the format: NumPy then refuses object
    # arrays, the only part of an .npz archive that is unpickled on load.
    wanted = (*_WEIGHT_SHAPES, *_SPLIT_KEYS, *_CONFIG_KEYS)
    try:
        # The file is opened here, not by np.load, so that it is closed even when
        # NumPy gives up half-way through a damaged archive.
        with open(path, "rb") as handle:
            archive = np.load(handle, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise WeightsError(
                    f"{path} holds a single array, not a weights archive. {_TRAIN_HINT}"
                )
            with archive:
                missing = [key for key in wanted if key not in archive.files]
                if missing:
                    raise WeightsError(
                        f"{path} is not a weights file of this project: it lacks "
                        f"{', '.join(missing)}. {_TRAIN_HINT}"
                    )
                return {key: archive[key] for key in wanted}
    except _UNREADABLE as err:
        raise WeightsError(
            f"{path} cannot be read as a .npz weights file ({type(err).__name__}: {err}). "
            f"{_TRAIN_HINT}"
        ) from err


def _check_points(path, name, X, y):
    if X.ndim != 2 or X.shape[1] != 2 or X.dtype.kind != "f" or not np.isfinite(X).all():
        raise WeightsError(
            f"{path}: X_{name} must be a finite float array of shape (N, 2); "
            f"found shape {X.shape}, dtype {X.dtype}"
        )
    if y.shape != (len(X),) or y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all():
        raise WeightsError(
            f"{path}: y_{name} must hold one label, 0 or 1, for each of the {len(X)} rows "
            f"of X_{name}; found shape {y.shape}, dtype {y.dtype}"
        )
    if len(X) == 0:
        raise WeightsError(f"{path}: the {name} set is empty")


def load_weights(path):
    """Read a weights file written by save_weights.

    Returns (model, split, config): the network with the stored weights, the
    tuple (X_train, y_train, X_test, y_test), and a dict with the keys init,
    noise, n_samples, seed, and epochs. Raises WeightsError, with a message
    that says what to do, when the file is missing, is not a .npz archive of
    numeric arrays, or does not hold what this project writes.
    """
    path = npz_path(path)
    if not path.is_file():
        raise WeightsError(f"no weights file at {path}. {_TRAIN_HINT}")
    arrays = _read_arrays(path)

    version = arrays["format_version"]
    if version.shape != () or version.dtype.kind not in "iu" or int(version) != FORMAT_VERSION:
        raise WeightsError(
            f"{path} has weights format {version.tolist()!r}; this version reads "
            f"format {FORMAT_VERSION}. {_TRAIN_HINT}"
        )
    for key, shape in _WEIGHT_SHAPES.items():
        value = arrays[key]
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise WeightsError(
                f"{path}: {key} must be a finite float array of shape {shape}; "
                f"found shape {value.shape}, dtype {value.dtype}"
            )
    _check_points(path, "train", arrays["X_train"], arrays["y_train"])
    _check_points(path, "test", arrays["X_test"], arrays["y_test"])

    init = arrays["init"]
    if init.shape != () or init.dtype.kind != "U" or str(init) not in INIT_CHOICES:
        raise WeightsError(
            f"{path}: init must be one of {', '.join(INIT_CHOICES)}; found {init.tolist()!r}"
        )
    kinds = {"noise": "f", "n_samples": "iu", "seed": "iu", "epochs": "iu"}
    for key, kind in kinds.items():
        if arrays[key].shape != () or arrays[key].dtype.kind not in kind:
            raise WeightsError(f"{path}: {key} must be a single number")
    config = {
        "init": str(init),
        "noise": float(arrays["noise"]),
        "n_samples": int(arrays["n_samples"]),
        "seed": int(arrays["seed"]),
        "epochs": int(arrays["epochs"]),
    }

    model = Network(init=config["init"])
    for index, layer in enumerate(model.dense_layers, start=1):
        layer.weights = arrays[f"dense{index}_weights"].astype(np.float64)
        layer.biases = arrays[f"dense{index}_biases"].astype(np.float64)
    split = (arrays["X_train"], arrays["y_train"], arrays["X_test"], arrays["y_test"])
    return model, split, config
