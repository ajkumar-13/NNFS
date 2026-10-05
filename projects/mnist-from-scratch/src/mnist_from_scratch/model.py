"""The network: three dense layers assembled from the series' classes, and its weights file.

    input (784) -> Dense(784, 128) -> ReLU -> Dropout(0.1)
                -> Dense(128, 128) -> ReLU -> Dropout(0.1)
                -> Dense(128, 10)  -> softmax and categorical cross-entropy

The two hidden layers carry an L2 penalty of 5e-4 on their weights; the output layer and all biases
carry none. That is the documented baseline, and the defaults of :class:`Network` are that
baseline. The constructor takes other sizes only so the tests can build a small network.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
import zlib
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from mnist_from_scratch.nn import (
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Layer_Dropout,
    regularization_loss,
)

N_INPUTS = 784
N_HIDDEN = 128
N_CLASSES = 10
L2_STRENGTH = 5e-4
DROPOUT_RATE = 0.1

WEIGHT_KEYS = (
    "dense1_weights",
    "dense1_biases",
    "dense2_weights",
    "dense2_biases",
    "dense3_weights",
    "dense3_biases",
)


class WeightsError(Exception):
    """The weights file is missing, unreadable, or does not fit this network."""


class Network:
    """The 784-128-128-10 classifier.

    Layers are built in the order dense1, dense2, dense3, and each draws its initial weights from
    NumPy's global generator, so the order is part of a seeded run.
    """

    def __init__(
        self,
        n_inputs: int = N_INPUTS,
        n_hidden: int = N_HIDDEN,
        n_classes: int = N_CLASSES,
        l2: float = L2_STRENGTH,
        dropout_rate: float = DROPOUT_RATE,
    ):
        self.dense1 = Layer_Dense(n_inputs, n_hidden, weight_regularizer_l2=l2)
        self.activation1 = Activation_ReLU()
        self.dropout1 = Layer_Dropout(dropout_rate)

        self.dense2 = Layer_Dense(n_hidden, n_hidden, weight_regularizer_l2=l2)
        self.activation2 = Activation_ReLU()
        self.dropout2 = Layer_Dropout(dropout_rate)

        self.dense3 = Layer_Dense(n_hidden, n_classes)
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    @property
    def dense_layers(self) -> tuple[Layer_Dense, Layer_Dense, Layer_Dense]:
        """The layers that hold parameters, in the order the optimiser updates them."""
        return (self.dense1, self.dense2, self.dense3)

    @property
    def n_inputs(self) -> int:
        return self.dense1.weights.shape[0]

    @property
    def n_classes(self) -> int:
        return self.dense3.weights.shape[1]

    def parameter_count(self) -> int:
        return sum(layer.weights.size + layer.biases.size for layer in self.dense_layers)

    def forward(self, X: np.ndarray, y: np.ndarray, training: bool) -> float:
        """Run the forward pass and return the data loss (mean cross-entropy, no penalty).

        The class probabilities are left in ``self.loss_activation.output``. Dropout is active only
        when ``training`` is true.
        """
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dropout1.forward(self.activation1.output, training=training)

        self.dense2.forward(self.dropout1.output)
        self.activation2.forward(self.dense2.output)
        self.dropout2.forward(self.activation2.output, training=training)

        self.dense3.forward(self.dropout2.output)
        return self.loss_activation.forward(self.dense3.output, y)

    def regularization_loss(self) -> float:
        """The L2 penalty summed over the dense layers."""
        return (
            regularization_loss(self.dense1)
            + regularization_loss(self.dense2)
            + regularization_loss(self.dense3)
        )

    def backward(self, y: np.ndarray) -> None:
        """Backpropagate from the probabilities of the last training forward pass."""
        self.loss_activation.backward(self.loss_activation.output, y)
        self.dense3.backward(self.loss_activation.dinputs)
        self.dropout2.backward(self.dense3.dinputs)
        self.activation2.backward(self.dropout2.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.dropout1.backward(self.dense2.dinputs)
        self.activation1.backward(self.dropout1.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def predictions(self) -> np.ndarray:
        """The predicted class of every row of the last forward pass."""
        return np.argmax(self.loss_activation.output, axis=1)

    def weights(self) -> dict[str, np.ndarray]:
        """The six parameter arrays under the names used in the weights file."""
        arrays = {}
        for index, layer in enumerate(self.dense_layers, start=1):
            arrays[f"dense{index}_weights"] = layer.weights
            arrays[f"dense{index}_biases"] = layer.biases
        return arrays

    def fingerprint(self) -> str:
        """SHA-256 of the six parameter arrays as 64-bit floats, in the order of the weights file.

        Two networks have the same fingerprint exactly when their parameters are equal bit for bit,
        which the weights file itself cannot show because the archive stores a timestamp.
        """
        digest = hashlib.sha256()
        for array in self.weights().values():
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Write the six arrays to an ``.npz`` archive, replacing ``path`` only once it is whole."""
        path = Path(path)
        partial = path.with_name(path.name + ".part")
        with open(partial, "wb") as handle:
            np.savez(handle, **self.weights())
        os.replace(partial, path)

    def load(self, path: str | Path) -> None:
        """Replace the parameters with those in ``path`` after checking that they fit.

        The archive is read with ``allow_pickle=False``, so loading never executes code from the
        file. Every array must have the shape of the layer it goes into, a floating-point type, and
        finite values; the arrays are converted to 64-bit floats, the type the network trains in.
        A failed check raises :class:`WeightsError` and leaves the network as it was.
        """
        path = Path(path)
        if not path.is_file():
            raise WeightsError(
                f"the weights file {path} does not exist. Train first:\n"
                "    uv run python -m mnist_from_scratch.train"
            )
        if path.suffix != ".npz":
            raise WeightsError(
                f"{path} is not an .npz file. Pickle checkpoints written before version 1.0.0 are "
                "not loaded, because unpickling can run code from the file. Train again:\n"
                "    uv run python -m mnist_from_scratch.train"
            )
        expected = self.weights()
        try:
            with zipfile.ZipFile(path) as archive:
                found = sorted(archive.namelist())
                if found != sorted(f"{key}.npy" for key in WEIGHT_KEYS):
                    raise WeightsError(
                        f"{path} does not hold exactly the arrays {', '.join(WEIGHT_KEYS)}; "
                        f"it holds {', '.join(found) or 'nothing'}"
                    )
                # Shapes and types come from the headers, before any array data is read, so a
                # file cannot make the loader allocate more than the network's own size.
                for key in WEIGHT_KEYS:
                    shape, dtype = _array_header(archive, f"{key}.npy")
                    if shape != expected[key].shape:
                        raise WeightsError(
                            f"{path}: {key} has shape {shape}, but this network needs "
                            f"{expected[key].shape}"
                        )
                    if dtype.kind != "f":
                        raise WeightsError(
                            f"{path}: {key} has type {dtype}, expected a floating-point type"
                        )
            with np.load(path, allow_pickle=False) as archive:
                loaded = {key: archive[key].astype(np.float64) for key in WEIGHT_KEYS}
        except (zipfile.BadZipFile, zlib.error, ValueError, OSError, EOFError) as error:
            raise WeightsError(f"{path} is not a readable weights file: {error}") from error

        for key in WEIGHT_KEYS:
            if not np.all(np.isfinite(loaded[key])):
                raise WeightsError(f"{path}: {key} contains values that are not finite")

        for index, layer in enumerate(self.dense_layers, start=1):
            layer.weights = loaded[f"dense{index}_weights"]
            layer.biases = loaded[f"dense{index}_biases"]


def _array_header(archive: zipfile.ZipFile, member: str) -> tuple[tuple[int, ...], np.dtype]:
    """The shape and type that one ``.npy`` member of an archive declares, without its data."""
    with archive.open(member) as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"{member} uses the unsupported .npy format version {version}")
    return shape, dtype
