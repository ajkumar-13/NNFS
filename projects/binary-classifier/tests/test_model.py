"""The assembled network: its size, its seeding, its forward and backward passes."""

import math
from itertools import pairwise

import numpy as np
import pytest

from binary_classifier.model import INIT_CHOICES, L2_LAMBDA, LAYER_SIZES, Network
from binary_classifier.nn import Layer_Dense


def test_parameter_count_is_337():
    model = Network()
    # (2 * 16 + 16) + (16 * 16 + 16) + (16 * 1 + 1) = 48 + 272 + 17.
    assert model.parameter_count() == 337
    closed_form = sum(n_in * n_out + n_out for n_in, n_out in pairwise(LAYER_SIZES))
    assert model.parameter_count() == closed_form


def test_layer_shapes_and_where_the_l2_penalty_sits():
    model = Network()
    assert [layer.weights.shape for layer in model.dense_layers] == [(2, 16), (16, 16), (16, 1)]
    assert [layer.biases.shape for layer in model.dense_layers] == [(1, 16), (1, 16), (1, 1)]
    assert model.dense1.weight_regularizer_l2 == L2_LAMBDA == 1e-4
    assert model.dense2.weight_regularizer_l2 == L2_LAMBDA
    assert model.dense3.weight_regularizer_l2 == 0.0


def test_seed_fixes_the_weights_in_the_order_dense1_dense2_dense3():
    np.random.seed(0)
    model = Network()
    reference = np.random.RandomState(0)
    for layer, (n_in, n_out) in zip(model.dense_layers, pairwise(LAYER_SIZES), strict=True):
        expected = math.sqrt(2.0 / n_in) * reference.randn(n_in, n_out)
        np.testing.assert_allclose(layer.weights, expected, rtol=1e-15)


@pytest.mark.parametrize("init", INIT_CHOICES)
def test_every_offered_initialisation_builds(init):
    np.random.seed(1)
    model = Network(init=init)
    np.random.seed(1)
    np.testing.assert_array_equal(model.dense1.weights, Layer_Dense(2, 16, init=init).weights)
    assert model.init == init


def test_unknown_initialisation_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match=r"init must be one of he, xavier, small; got 'lecun'"):
        Network(init="lecun")


def test_forward_equals_the_written_out_network():
    np.random.seed(2)
    model = Network()
    for layer in model.dense_layers:
        layer.biases = np.random.randn(*layer.biases.shape)
    X = np.random.randn(6, 2).astype(np.float32)

    hidden1 = np.maximum(0, X @ model.dense1.weights + model.dense1.biases)
    hidden2 = np.maximum(0, hidden1 @ model.dense2.weights + model.dense2.biases)
    expected = hidden2 @ model.dense3.weights + model.dense3.biases

    logits = model.forward(X)
    assert logits.shape == (6, 1)
    np.testing.assert_allclose(logits, expected, rtol=1e-12)


def test_ramp_model_logit_is_ten_times_the_first_coordinate(ramp_model):
    X = np.array([[0.3, 9.0], [-0.7, -4.0], [0.0, 1.0]])
    np.testing.assert_allclose(ramp_model.forward(X), [[3.0], [-7.0], [0.0]], atol=1e-15)


def test_backward_matches_numerical_gradient_for_every_parameter(numerical_gradient):
    np.random.seed(5)
    model = Network()
    for layer in model.dense_layers:
        layer.biases = 0.1 * np.random.randn(*layer.biases.shape)
    X = np.random.randn(7, 2)
    y = np.array([0, 1, 1, 0, 1, 0, 0])

    def loss():
        return model.data_loss(X, y) + model.regularization_loss()

    loss()
    model.backward(y)
    for layer in model.dense_layers:
        np.testing.assert_allclose(
            layer.dweights, numerical_gradient(loss, layer.weights), rtol=1e-5, atol=1e-9
        )
        np.testing.assert_allclose(
            layer.dbiases, numerical_gradient(loss, layer.biases), rtol=1e-5, atol=1e-9
        )


def test_regularization_loss_sums_the_two_hidden_layers(ramp_model):
    # dense1 holds weights 1 and -1, dense2 holds 1 and 1; the output layer carries no penalty.
    assert ramp_model.regularization_loss() == pytest.approx(L2_LAMBDA * (2.0 + 2.0))


def test_predict_proba_and_the_threshold(ramp_model):
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    proba = ramp_model.predict_proba(X)
    assert proba.shape == (3,)
    sigmoid_10 = 1.0 / (1.0 + math.exp(-10.0))
    np.testing.assert_allclose(proba, [sigmoid_10, 1.0 - sigmoid_10, 0.5], rtol=1e-12)
    # A probability of exactly 0.5 is class 1.
    np.testing.assert_array_equal(ramp_model.predict(X), [1, 0, 1])
    assert ramp_model.predict(X).dtype == np.int64
