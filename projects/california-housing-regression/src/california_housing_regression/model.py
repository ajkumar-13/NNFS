"""The network: three dense layers assembled from the series' classes, and its weights file.

    input (8) -> Dense(8, 64)  -> ReLU
              -> Dense(64, 64) -> ReLU
              -> Dense(64, 1)  -> the prediction, with mean squared error as the loss

The output is linear: no softmax, no sigmoid, the last dense layer's one number is the prediction.
The two hidden layers carry an L2 penalty of 1e-4 on their weights; the output layer and all biases
carry none. That is the documented baseline, and the defaults of :class:`Network` are that
baseline. The constructor takes other sizes only so the tests can build a small network.

The weights file holds the six parameter arrays and, beside them, the four scaling statistics of
the training fold, because a prediction means nothing without the numbers that scaled the inputs
and that turn the output back into dollars.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
import zlib
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from california_housing_regression.data import Scaler
from california_housing_regression.nn import (
    Activation_ReLU,
    Layer_Dense,
    Loss_MSE,
    regularization_loss,
)

N_INPUTS = 8
N_HIDDEN = 64
N_OUTPUTS = 1
L2_STRENGTH = 1e-4

WEIGHT_KEYS = (
    "dense1_weights",
    "dense1_biases",
    "dense2_weights",
    "dense2_biases",
    "dense3_weights",
    "dense3_biases",
)
FEATURE_KEYS = ("x_mean", "x_std")
SCALER_KEYS = (*FEATURE_KEYS, "y_mean", "y_std")
FILE_KEYS = WEIGHT_KEYS + SCALER_KEYS

TRAIN_COMMAND = "uv run python -m california_housing_regression.train"


class WeightsError(Exception):
    """The weights file is missing, unreadable, or does not fit this network."""


class Network:
    """The 8-64-64-1 regressor.

    Layers are built in the order dense1, dense2, dense3, and each draws its initial weights from
    NumPy's global generator, so the order is part of a seeded run.
    """

    def __init__(
        self,
        n_inputs: int = N_INPUTS,
        n_hidden: int = N_HIDDEN,
        n_outputs: int = N_OUTPUTS,
        l2: float = L2_STRENGTH,
    ):
        self.dense1 = Layer_Dense(n_inputs, n_hidden, weight_regularizer_l2=l2)
        self.activation1 = Activation_ReLU()

        self.dense2 = Layer_Dense(n_hidden, n_hidden, weight_regularizer_l2=l2)
        self.activation2 = Activation_ReLU()

        self.dense3 = Layer_Dense(n_hidden, n_outputs)
        self.loss = Loss_MSE()

    @property
    def dense_layers(self) -> tuple[Layer_Dense, Layer_Dense, Layer_Dense]:
        """The layers that hold parameters, in the order the optimiser updates them."""
        return (self.dense1, self.dense2, self.dense3)

    @property
    def n_inputs(self) -> int:
        return self.dense1.weights.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.dense3.weights.shape[1]

    def parameter_count(self) -> int:
        return sum(layer.weights.size + layer.biases.size for layer in self.dense_layers)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the forward pass and return the output layer's values, shape ``(N, n_outputs)``."""
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)

        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)

        self.dense3.forward(self.activation2.output)
        return self.dense3.output

    def forward(self, X: np.ndarray, y: np.ndarray) -> float:
        """Run the forward pass and return the data loss (mean squared error, no penalty).

        The predictions are left in ``self.dense3.output``. The loss is in the units of ``y``,
        which for this project are standardised units.
        """
        return self.loss.forward(self.predict(X), y)

    def regularization_loss(self) -> float:
        """The L2 penalty summed over the dense layers."""
        return (
            regularization_loss(self.dense1)
            + regularization_loss(self.dense2)
            + regularization_loss(self.dense3)
        )

    def backward(self) -> None:
        """Backpropagate from the predictions and targets of the last ``forward``."""
        self.loss.backward(self.dense3.output)
        self.dense3.backward(self.loss.dinputs)
        self.activation2.backward(self.dense3.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

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
        which the weights file itself cannot show because the archive stores a timestamp. The
        scaling statistics are not part of it: they follow from the data and the seed.
        """
        digest = hashlib.sha256()
        for array in self.weights().values():
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path, scaler: Scaler) -> None:
        """Write the parameters and the scaler to an ``.npz`` archive.

        ``path`` is replaced only once the new file is whole.
        """
        path = Path(path)
        partial = path.with_name(path.name + ".part")
        arrays = self.weights() | {
            "x_mean": np.asarray(scaler.x_mean, dtype=np.float32),
            "x_std": np.asarray(scaler.x_std, dtype=np.float32),
            "y_mean": np.float64(scaler.y_mean),
            "y_std": np.float64(scaler.y_std),
        }
        with open(partial, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(partial, path)

    def load(self, path: str | Path) -> Scaler:
        """Replace the parameters with those in ``path`` and return the scaler stored with them.

        The archive is read with ``allow_pickle=False``, so loading never executes code from the
        file. Every array must have the shape this network needs, a floating-point type, and finite
        values, and the two standard deviations must be positive. The parameters are converted to
        64-bit floats, the type the network trains in. A failed check raises
        :class:`WeightsError` and leaves the network as it was.
        """
        path = Path(path)
        if not path.is_file():
            raise WeightsError(
                f"the weights file {path} does not exist. Train first:\n    {TRAIN_COMMAND}"
            )
        if path.suffix != ".npz":
            raise WeightsError(
                f"{path} is not an .npz file. Pickle checkpoints written before version 1.0.0 are "
                f"not loaded, because unpickling can run code from the file. Train again:\n"
                f"    {TRAIN_COMMAND}"
            )
        shapes = {key: array.shape for key, array in self.weights().items()} | {
            "x_mean": (self.n_inputs,),
            "x_std": (self.n_inputs,),
            "y_mean": (),
            "y_std": (),
        }
        try:
            with zipfile.ZipFile(path) as archive:
                found = sorted(archive.namelist())
                if found != sorted(f"{key}.npy" for key in FILE_KEYS):
                    raise WeightsError(
                        f"{path} does not hold exactly the arrays {', '.join(FILE_KEYS)}; "
                        f"it holds {', '.join(found) or 'nothing'}"
                    )
                # Shapes and types come from the headers, before any array data is read, so a
                # file cannot make the loader allocate more than the network's own size.
                for key in FILE_KEYS:
                    shape, dtype = _array_header(archive, f"{key}.npy")
                    if shape != shapes[key]:
                        raise WeightsError(
                            f"{path}: {key} has shape {shape}, but this network needs {shapes[key]}"
                        )
                    if dtype.kind != "f":
                        raise WeightsError(
                            f"{path}: {key} has type {dtype}, expected a floating-point type"
                        )
            # The feature statistics go back to float32, the type they scale the data in; a value
            # too large for it becomes infinite here and is refused below.
            with np.load(path, allow_pickle=False) as archive, np.errstate(over="ignore"):
                loaded = {
                    key: archive[key].astype(np.float32 if key in FEATURE_KEYS else np.float64)
                    for key in FILE_KEYS
                }
        except (zipfile.BadZipFile, zlib.error, ValueError, OSError, EOFError) as error:
            raise WeightsError(f"{path} is not a readable weights file: {error}") from error

        for key in FILE_KEYS:
            if not np.all(np.isfinite(loaded[key])):
                raise WeightsError(f"{path}: {key} contains values that are not finite")
        for key in ("x_std", "y_std"):
            if np.any(loaded[key] <= 0):
                raise WeightsError(
                    f"{path}: {key} must be positive, because the data is divided by it"
                )

        for index, layer in enumerate(self.dense_layers, start=1):
            layer.weights = loaded[f"dense{index}_weights"]
            layer.biases = loaded[f"dense{index}_biases"]
        return Scaler(
            x_mean=loaded["x_mean"],
            x_std=loaded["x_std"],
            y_mean=float(loaded["y_mean"]),
            y_std=float(loaded["y_std"]),
        )


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
