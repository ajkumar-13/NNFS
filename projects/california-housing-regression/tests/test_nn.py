"""The series' classes: hand-computed forward values, backward passes against finite differences."""

import numpy as np
import pytest

from california_housing_regression.nn import (
    Activation_ReLU,
    Layer_Dense,
    Loss_MSE,
    Optimizer_Adam,
    regularization_loss,
)
from helpers import generator_state, numerical_gradient

# ---------------------------------------------------------------------------------- Layer_Dense


def test_dense_initialisation_draws_small_weights_and_zero_biases():
    np.random.seed(0)
    layer = Layer_Dense(4, 3)
    expected = 0.01 * np.random.RandomState(0).randn(4, 3)
    assert layer.weights.shape == (4, 3)
    assert np.array_equal(layer.weights, expected)
    assert layer.biases.shape == (1, 3)
    assert np.array_equal(layer.biases, np.zeros((1, 3)))


def test_dense_forward_on_a_hand_computed_case():
    layer = Layer_Dense(2, 3)
    layer.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.biases = np.array([[0.5, -0.5, 1.0]])
    layer.forward(np.array([[1.0, 2.0], [0.0, -1.0]]))
    assert np.array_equal(layer.output, np.array([[9.5, 11.5, 16.0], [-3.5, -5.5, -5.0]]))


def test_dense_backward_on_a_hand_computed_case():
    layer = Layer_Dense(2, 2)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.forward(np.array([[1.0, 2.0], [3.0, 4.0]]))
    layer.backward(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.array_equal(layer.dweights, np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert np.array_equal(layer.dbiases, np.array([[1.0, 1.0]]))
    assert np.array_equal(layer.dinputs, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_dense_backward_matches_finite_differences():
    layer = Layer_Dense(4, 3)
    layer.weights = np.random.randn(4, 3)
    layer.biases = np.random.randn(1, 3)
    inputs = np.random.randn(5, 4)
    upstream = np.random.randn(5, 3)

    def scalar():
        layer.forward(inputs)
        return float(np.sum(layer.output * upstream))

    layer.forward(inputs)
    layer.backward(upstream)
    assert layer.dweights.shape == (4, 3)
    assert layer.dbiases.shape == (1, 3)
    assert layer.dinputs.shape == (5, 4)
    assert np.allclose(layer.dweights, numerical_gradient(scalar, layer.weights), atol=1e-7)
    assert np.allclose(layer.dbiases, numerical_gradient(scalar, layer.biases), atol=1e-7)
    assert np.allclose(layer.dinputs, numerical_gradient(scalar, inputs), atol=1e-7)


def test_dense_accepts_float32_inputs_and_computes_in_float64():
    layer = Layer_Dense(3, 2)
    layer.forward(np.ones((4, 3), dtype=np.float32))
    assert layer.output.dtype == np.float64


@pytest.mark.parametrize(
    "strengths",
    [
        {"weight_regularizer_l2": 0.3},
        {"weight_regularizer_l1": 0.2},
        {"bias_regularizer_l2": 0.4},
        {"bias_regularizer_l1": 0.1},
        {
            "weight_regularizer_l1": 0.05,
            "weight_regularizer_l2": 0.3,
            "bias_regularizer_l1": 0.02,
            "bias_regularizer_l2": 0.4,
        },
    ],
)
def test_dense_penalty_gradients_match_finite_differences(strengths):
    layer = Layer_Dense(3, 2, **strengths)
    layer.weights = np.random.randn(3, 2)
    layer.biases = np.random.randn(1, 2)
    inputs = np.random.randn(4, 3)
    upstream = np.random.randn(4, 2)

    def scalar():
        layer.forward(inputs)
        return float(np.sum(layer.output * upstream)) + float(regularization_loss(layer))

    layer.forward(inputs)
    layer.backward(upstream)
    assert np.allclose(layer.dweights, numerical_gradient(scalar, layer.weights), atol=1e-6)
    assert np.allclose(layer.dbiases, numerical_gradient(scalar, layer.biases), atol=1e-6)


def test_dense_without_penalty_adds_nothing_to_the_gradient():
    plain = Layer_Dense(3, 2)
    penalised = Layer_Dense(3, 2, weight_regularizer_l2=0.5)
    penalised.weights = plain.weights.copy()
    inputs = np.random.randn(4, 3)
    upstream = np.random.randn(4, 2)
    for layer in (plain, penalised):
        layer.forward(inputs)
        layer.backward(upstream)
    assert np.allclose(penalised.dweights - plain.dweights, 2 * 0.5 * plain.weights)
    assert np.array_equal(penalised.dbiases, plain.dbiases)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"n_inputs": 0, "n_neurons": 3}, "at least one input and one neuron"),
        ({"n_inputs": 3, "n_neurons": 0}, "at least one input and one neuron"),
        ({"n_inputs": 3, "n_neurons": 2, "weight_regularizer_l2": -0.1}, "must not be negative"),
        ({"n_inputs": 3, "n_neurons": 2, "bias_regularizer_l1": -1}, "must not be negative"),
    ],
)
def test_dense_rejects_impossible_arguments(arguments, message):
    state = generator_state()
    with pytest.raises(ValueError, match=message):
        Layer_Dense(**arguments)
    # A rejected layer draws nothing, so it cannot shift a seeded run.
    assert generator_state() == state


# ------------------------------------------------------------------------------ Activation_ReLU


def test_relu_forward_and_backward_on_a_hand_computed_case():
    relu = Activation_ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0], [1.5, -0.5, 0.0]])
    relu.forward(inputs)
    assert np.array_equal(relu.output, np.array([[0.0, 0.0, 3.0], [1.5, 0.0, 0.0]]))

    upstream = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    relu.backward(upstream)
    # The gradient passes only where the input was strictly positive; at exactly zero it is zero.
    assert np.array_equal(relu.dinputs, np.array([[0.0, 0.0, 30.0], [40.0, 0.0, 0.0]]))
    assert np.array_equal(upstream, np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))


def test_relu_backward_matches_finite_differences_away_from_zero():
    relu = Activation_ReLU()
    inputs = np.random.randn(6, 4)
    inputs[np.abs(inputs) < 0.05] = 0.5
    upstream = np.random.randn(6, 4)

    def scalar():
        relu.forward(inputs)
        return float(np.sum(relu.output * upstream))

    relu.forward(inputs)
    relu.backward(upstream)
    assert np.allclose(relu.dinputs, numerical_gradient(scalar, inputs), atol=1e-7)


# ------------------------------------------------------------------------------------- Loss_MSE


def test_mse_forward_on_a_hand_computed_case():
    loss = Loss_MSE()
    y_pred = np.array([[1.0], [2.0], [4.0]])
    y_true = np.array([0.0, 2.0, 1.0])
    # Squared errors 1, 0 and 9; their mean is 10 / 3.
    assert loss.forward(y_pred, y_true) == pytest.approx(10 / 3)
    assert isinstance(loss.forward(y_pred, y_true), float)


def test_mse_is_zero_for_a_perfect_prediction_and_positive_otherwise():
    loss = Loss_MSE()
    y = np.array([[0.5], [-1.5], [2.0]])
    assert loss.forward(y, y.ravel()) == 0.0
    assert loss.forward(y + 0.1, y.ravel()) == pytest.approx(0.01)
    # The loss is symmetric: over-prediction and under-prediction by the same amount cost the same.
    assert loss.forward(y - 0.1, y.ravel()) == pytest.approx(0.01)


def test_mse_backward_on_a_hand_computed_case():
    loss = Loss_MSE()
    y_pred = np.array([[1.0], [2.0], [4.0]])
    loss.forward(y_pred, np.array([0.0, 2.0, 1.0]))
    loss.backward(y_pred)
    # 2 (prediction - target) / N with N = 3.
    assert loss.dinputs.shape == (3, 1)
    assert np.allclose(loss.dinputs, np.array([[2 / 3], [0.0], [2.0]]))


def test_mse_backward_matches_finite_differences():
    loss = Loss_MSE()
    y_pred = np.random.randn(7, 1)
    y_true = np.random.randn(7)
    loss.forward(y_pred, y_true)
    loss.backward(y_pred)
    numerical = numerical_gradient(lambda: Loss_MSE().forward(y_pred, y_true), y_pred)
    assert np.allclose(loss.dinputs, numerical, atol=1e-8)


def test_mse_averages_over_samples_and_outputs():
    loss = Loss_MSE()
    y_pred = np.array([[1.0, 2.0], [3.0, 5.0]])
    y_true = np.array([[0.0, 2.0], [3.0, 1.0]])
    # Squared errors 1, 0, 0 and 16 over N K = 4 entries.
    assert loss.forward(y_pred, y_true) == pytest.approx(17 / 4)
    loss.backward(y_pred)
    assert np.allclose(loss.dinputs, np.array([[0.5, 0.0], [0.0, 2.0]]))
    numerical = numerical_gradient(lambda: Loss_MSE().forward(y_pred, y_true), y_pred)
    assert np.allclose(loss.dinputs, numerical, atol=1e-8)


def test_mse_takes_targets_as_a_vector_or_a_column_and_holds_them_as_float64():
    y_pred = np.array([[1.0], [2.0], [4.0]])
    as_vector, as_column = Loss_MSE(), Loss_MSE()
    targets = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    assert as_vector.forward(y_pred, targets) == as_column.forward(y_pred, targets.reshape(3, 1))
    assert as_vector.y_true.shape == (3, 1)
    assert as_vector.y_true.dtype == np.float64


def test_mse_does_not_broadcast_a_vector_of_targets_against_a_column_of_predictions():
    # Without the reshape, (N, 1) - (N,) would broadcast to (N, N) and average N * N differences.
    y_pred = np.array([[1.0], [2.0], [3.0]])
    y_true = np.array([1.0, 2.0, 3.0])
    assert Loss_MSE().forward(y_pred, y_true) == 0.0


def test_mse_backward_uses_explicit_targets_when_given():
    loss = Loss_MSE()
    y_pred = np.array([[1.0], [3.0]])
    loss.forward(y_pred, np.array([1.0, 3.0]))
    loss.backward(y_pred)
    assert np.array_equal(loss.dinputs, np.zeros((2, 1)))
    loss.backward(y_pred, np.array([0.0, 0.0]))
    assert np.allclose(loss.dinputs, np.array([[1.0], [3.0]]))


def test_mse_backward_on_a_vector_of_predictions():
    loss = Loss_MSE()
    y_pred = np.array([1.0, 3.0])
    assert loss.forward(y_pred, np.array([0.0, 1.0])) == pytest.approx(2.5)
    loss.backward(y_pred)
    assert np.allclose(loss.dinputs, np.array([1.0, 2.0]))


def test_mse_rejects_a_wrong_number_of_targets():
    loss = Loss_MSE()
    with pytest.raises(ValueError, match="one target per prediction, got 2 targets"):
        loss.forward(np.zeros((3, 1)), np.zeros(2))
    loss.forward(np.zeros((3, 1)), np.zeros(3))
    with pytest.raises(ValueError, match="one target per prediction, got 4 targets"):
        loss.backward(np.zeros((3, 1)), np.zeros(4))


# ------------------------------------------------------------------------------- Optimizer_Adam


def layer_with_gradient(weights, dweights, biases, dbiases):
    layer = Layer_Dense(weights.shape[0], weights.shape[1])
    layer.weights = weights.astype(float)
    layer.biases = biases.astype(float)
    layer.dweights = dweights.astype(float)
    layer.dbiases = dbiases.astype(float)
    return layer


def test_adam_first_step_moves_every_parameter_by_the_learning_rate_against_its_gradient():
    # With bias correction the first step is lr * g / (|g| + epsilon): the sign of the gradient.
    optimizer = Optimizer_Adam(learning_rate=0.1)
    layer = layer_with_gradient(
        np.array([[1.0, -2.0]]),
        np.array([[0.5, -3.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[2.0, -0.1]]),
    )
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()
    assert np.allclose(layer.weights, np.array([[0.9, -1.9]]), atol=1e-6)
    assert np.allclose(layer.biases, np.array([[-0.1, 0.1]]), atol=1e-6)
    assert optimizer.iterations == 1


def test_adam_two_steps_on_a_hand_computed_case():
    optimizer = Optimizer_Adam(learning_rate=0.1, epsilon=1e-7, beta_1=0.9, beta_2=0.999)
    layer = layer_with_gradient(
        np.array([[1.0]]), np.array([[2.0]]), np.array([[0.0]]), np.array([[0.0]])
    )
    for gradient in (2.0, 1.0):
        layer.dweights = np.array([[gradient]])
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()

    # Step 1: m = 0.2, v = 0.004; corrected 2 and 4; update 0.1 * 2 / (2 + 1e-7).
    after_one = 1.0 - 0.1 * 2.0 / (2.0 + 1e-7)
    # Step 2: m = 0.9 * 0.2 + 0.1 = 0.28, v = 0.999 * 0.004 + 0.001 = 0.004996.
    m_hat = 0.28 / (1 - 0.9**2)
    v_hat = 0.004996 / (1 - 0.999**2)
    after_two = after_one - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-7)
    assert layer.weights[0, 0] == pytest.approx(after_two, rel=1e-12)
    assert layer.weight_momentums[0, 0] == pytest.approx(0.28)
    assert layer.weight_cache[0, 0] == pytest.approx(0.004996)
    # A zero gradient leaves the bias where it was.
    assert layer.biases[0, 0] == 0.0


def test_adam_decay_follows_the_inverse_time_schedule():
    optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-4)
    rates = []
    for _ in range(3):
        optimizer.pre_update_params()
        rates.append(optimizer.current_learning_rate)
        optimizer.post_update_params()
    assert rates == pytest.approx([0.01, 0.01 / 1.0001, 0.01 / 1.0002])


def test_adam_without_decay_keeps_the_learning_rate():
    optimizer = Optimizer_Adam(learning_rate=0.01)
    for _ in range(5):
        optimizer.pre_update_params()
        optimizer.post_update_params()
    assert optimizer.current_learning_rate == 0.01
    assert optimizer.iterations == 5


def test_adam_keeps_separate_state_for_each_layer():
    optimizer = Optimizer_Adam(learning_rate=0.1)
    first = layer_with_gradient(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    second = layer_with_gradient(
        np.ones((1, 1)), -np.ones((1, 1)), np.zeros((1, 1)), -np.ones((1, 1))
    )
    optimizer.pre_update_params()
    optimizer.update_params(first)
    optimizer.update_params(second)
    optimizer.post_update_params()
    assert first.weights[0, 0] == pytest.approx(0.9, abs=1e-6)
    assert second.weights[0, 0] == pytest.approx(1.1, abs=1e-6)
    assert first.weight_momentums[0, 0] == pytest.approx(0.1)
    assert second.weight_momentums[0, 0] == pytest.approx(-0.1)


def test_adam_minimises_a_quadratic():
    optimizer = Optimizer_Adam(learning_rate=0.05)
    layer = layer_with_gradient(
        np.array([[3.0, -4.0]]), np.zeros((1, 2)), np.array([[2.0, 2.0]]), np.zeros((1, 2))
    )
    for _ in range(600):
        layer.dweights = 2 * layer.weights
        layer.dbiases = 2 * layer.biases
        optimizer.pre_update_params()
        optimizer.update_params(layer)
        optimizer.post_update_params()
    assert np.all(np.abs(layer.weights) < 1e-3)
    assert np.all(np.abs(layer.biases) < 1e-3)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"learning_rate": 0}, "learning rate must be positive"),
        ({"learning_rate": -0.1}, "learning rate must be positive"),
        ({"decay": -1e-4}, "decay must not be negative"),
        ({"epsilon": 0}, "epsilon must be positive"),
        ({"beta_1": 1.0}, "beta_1 and beta_2 must be in"),
        ({"beta_2": -0.1}, "beta_1 and beta_2 must be in"),
    ],
)
def test_adam_rejects_impossible_settings(arguments, message):
    with pytest.raises(ValueError, match=message):
        Optimizer_Adam(**arguments)


# -------------------------------------------------------------------------- regularization_loss


def test_regularization_loss_on_a_hand_computed_case():
    layer = Layer_Dense(
        2,
        2,
        weight_regularizer_l1=0.1,
        weight_regularizer_l2=0.01,
        bias_regularizer_l1=0.2,
        bias_regularizer_l2=0.02,
    )
    layer.weights = np.array([[1.0, -2.0], [3.0, -4.0]])
    layer.biases = np.array([[0.5, -1.5]])
    # 0.1 * 10 + 0.01 * 30 + 0.2 * 2 + 0.02 * 2.5
    assert regularization_loss(layer) == pytest.approx(1.0 + 0.3 + 0.4 + 0.05)


def test_regularization_loss_is_zero_without_a_strength():
    layer = Layer_Dense(3, 3)
    layer.weights = np.random.randn(3, 3)
    assert regularization_loss(layer) == 0.0


def test_regularization_loss_with_l2_only_is_the_strength_times_the_sum_of_squares():
    layer = Layer_Dense(2, 2, weight_regularizer_l2=1e-4)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.biases = np.array([[9.0, 9.0]])
    assert regularization_loss(layer) == pytest.approx(1e-4 * 30)
