"""Shared helpers: a central-difference gradient, a network with hand-set weights, a tiny run."""

import numpy as np
import pytest

from binary_classifier.model import Network
from binary_classifier.train import train


def _numerical_gradient(loss, array, step=1e-6):
    """Central-difference gradient of the scalar loss() with respect to array, in place."""
    gradient = np.zeros_like(array, dtype=np.float64)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        plus = loss()
        array[index] = original - step
        minus = loss()
        array[index] = original
        gradient[index] = (plus - minus) / (2 * step)
    return gradient


@pytest.fixture
def numerical_gradient():
    return _numerical_gradient


@pytest.fixture
def ramp_model():
    """A network whose logit is exactly 10 times the first input coordinate.

    Hidden unit 0 carries relu(x0) and hidden unit 1 carries relu(-x0) through
    both hidden layers; the output layer takes 10 times their difference. Every
    other weight is zero, so the probability of class 1 is sigmoid(10 * x0).
    """
    model = Network()
    for layer in model.dense_layers:
        layer.weights = np.zeros_like(layer.weights)
        layer.biases = np.zeros_like(layer.biases)
    model.dense1.weights[0, 0] = 1.0
    model.dense1.weights[0, 1] = -1.0
    model.dense2.weights[0, 0] = 1.0
    model.dense2.weights[1, 1] = 1.0
    model.dense3.weights[0, 0] = 10.0
    model.dense3.weights[1, 0] = -10.0
    return model


@pytest.fixture
def tiny_run(tmp_path, capsys):
    """A 200-epoch run on 40 points, with its weights file written under tmp_path."""
    result = train(
        epochs=200,
        noise=0.05,
        n_samples=40,
        seed=0,
        log_every=100,
        weights_path=tmp_path / "tiny.npz",
    )
    capsys.readouterr()
    return result
