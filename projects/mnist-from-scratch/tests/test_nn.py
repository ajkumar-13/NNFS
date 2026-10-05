"""The series' classes: hand-computed forward values, backward passes against finite differences."""

import numpy as np
import pytest

from helpers import generator_state, numerical_gradient
from mnist_from_scratch.nn import (
    Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Layer_Dense,
    Layer_Dropout,
    Optimizer_Adam,
    regularization_loss,
)

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


def test_dense_penalty_gradients_match_finite_differences():
    layer = Layer_Dense(
        3,
        2,
        weight_regularizer_l1=0.03,
        weight_regularizer_l2=0.05,
        bias_regularizer_l1=0.07,
        bias_regularizer_l2=0.11,
    )
    layer.weights = np.random.randn(3, 2)
    layer.biases = np.random.randn(1, 2)
    inputs = np.random.randn(4, 3)

    def penalty():
        return float(regularization_loss(layer))

    layer.forward(inputs)
    layer.backward(np.zeros((4, 2)))
    assert np.allclose(layer.dweights, numerical_gradient(penalty, layer.weights), atol=1e-7)
    assert np.allclose(layer.dbiases, numerical_gradient(penalty, layer.biases), atol=1e-7)


def test_regularization_loss_on_a_hand_computed_case():
    layer = Layer_Dense(
        2,
        2,
        weight_regularizer_l1=0.1,
        weight_regularizer_l2=0.01,
        bias_regularizer_l1=0.2,
        bias_regularizer_l2=0.5,
    )
    layer.weights = np.array([[1.0, -2.0], [3.0, -4.0]])
    layer.biases = np.array([[0.5, -1.5]])
    # 0.1 * 10 + 0.01 * 30 + 0.2 * 2 + 0.5 * 2.5
    assert regularization_loss(layer) == pytest.approx(2.95)


def test_regularization_loss_is_zero_without_strengths():
    assert regularization_loss(Layer_Dense(3, 2)) == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"n_inputs": 0, "n_neurons": 3},
        {"n_inputs": 3, "n_neurons": 0},
        {"n_inputs": 3, "n_neurons": 2, "weight_regularizer_l2": -0.1},
        {"n_inputs": 3, "n_neurons": 2, "bias_regularizer_l1": -1e-9},
    ],
)
def test_dense_rejects_impossible_arguments(arguments):
    with pytest.raises(ValueError, match=r"at least one|must not be negative"):
        Layer_Dense(**arguments)


# ------------------------------------------------------------------------------ Activation_ReLU


def test_relu_forward_and_backward():
    relu = Activation_ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0], [1.5, -0.5, 0.0]])
    relu.forward(inputs)
    assert np.array_equal(relu.output, np.array([[0.0, 0.0, 3.0], [1.5, 0.0, 0.0]]))

    upstream = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    relu.backward(upstream)
    assert np.array_equal(relu.dinputs, np.array([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))
    assert np.array_equal(upstream, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


# ------------------------------------------- Activation_Softmax_Loss_CategoricalCrossentropy


def test_softmax_loss_forward_on_hand_computed_cases():
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    logits = np.array([[0.0, 0.0], [np.log(1.0), np.log(3.0)]])
    loss = loss_activation.forward(logits, np.array([0, 1]))
    assert np.allclose(loss_activation.output, np.array([[0.5, 0.5], [0.25, 0.75]]))
    assert loss == pytest.approx((np.log(2.0) - np.log(0.75)) / 2)


def test_softmax_rows_sum_to_one_and_survive_large_logits():
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = loss_activation.forward(np.array([[1000.0, 1001.0, 999.0]]), np.array([1]))
    assert np.all(np.isfinite(loss_activation.output))
    assert loss_activation.output.sum() == pytest.approx(1.0)
    assert np.isfinite(loss)


def test_softmax_loss_accepts_one_hot_labels():
    logits = np.random.randn(6, 4)
    labels = np.array([0, 3, 1, 2, 2, 0])
    one_hot = np.eye(4)[labels]

    sparse = Activation_Softmax_Loss_CategoricalCrossentropy()
    dense = Activation_Softmax_Loss_CategoricalCrossentropy()
    assert sparse.forward(logits, labels) == pytest.approx(dense.forward(logits, one_hot))
    sparse.backward(sparse.output, labels)
    dense.backward(dense.output, one_hot)
    assert np.array_equal(sparse.dinputs, dense.dinputs)


def test_softmax_loss_is_clipped_for_a_confident_wrong_answer():
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = loss_activation.forward(np.array([[100.0, 0.0]]), np.array([1]))
    assert loss == pytest.approx(-np.log(1e-7))


def test_softmax_loss_backward_is_probabilities_minus_one_hot_over_n():
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    logits = np.array([[0.0, 0.0], [np.log(1.0), np.log(3.0)]])
    labels = np.array([0, 1])
    loss_activation.forward(logits, labels)
    loss_activation.backward(loss_activation.output, labels)
    assert np.allclose(loss_activation.dinputs, np.array([[-0.25, 0.25], [0.125, -0.125]]))


def test_softmax_loss_backward_matches_finite_differences():
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    logits = np.random.randn(5, 4)
    labels = np.array([0, 3, 1, 2, 2])

    def scalar():
        return Activation_Softmax_Loss_CategoricalCrossentropy().forward(logits, labels)

    loss_activation.forward(logits, labels)
    loss_activation.backward(loss_activation.output, labels)
    assert loss_activation.dinputs.shape == (5, 4)
    assert np.allclose(loss_activation.dinputs, numerical_gradient(scalar, logits), atol=1e-7)


# -------------------------------------------------------------------------------- Layer_Dropout


def test_dropout_in_training_zeroes_some_units_and_rescales_the_rest():
    dropout = Layer_Dropout(0.25)
    inputs = np.ones((200, 50))
    dropout.forward(inputs, training=True)
    assert set(np.unique(dropout.output)) == {0.0, 1 / 0.75}
    assert np.array_equal(dropout.output, inputs * dropout.binary_mask)
    # 10,000 Bernoulli draws at keep probability 0.75: the kept share has standard error 0.0043.
    assert np.mean(dropout.output > 0) == pytest.approx(0.75, abs=0.02)
    assert dropout.output.mean() == pytest.approx(1.0, abs=0.03)


def test_dropout_backward_uses_the_mask_of_the_forward_pass():
    dropout = Layer_Dropout(0.5)
    dropout.forward(np.random.randn(8, 6), training=True)
    upstream = np.random.randn(8, 6)
    dropout.backward(upstream)
    assert np.array_equal(dropout.dinputs, upstream * dropout.binary_mask)
    assert np.all(dropout.dinputs[dropout.binary_mask == 0] == 0)


def test_dropout_in_evaluation_is_the_identity_and_draws_nothing():
    dropout = Layer_Dropout(0.5)
    inputs = np.random.randn(8, 6)
    before = generator_state()
    dropout.forward(inputs, training=False)
    assert np.array_equal(dropout.output, inputs)
    assert dropout.output is not inputs
    assert generator_state() == before


def test_dropout_with_rate_zero_keeps_everything():
    dropout = Layer_Dropout(0.0)
    inputs = np.random.randn(4, 3)
    dropout.forward(inputs, training=True)
    assert np.array_equal(dropout.output, inputs)


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_dropout_rejects_rates_outside_the_unit_interval(rate):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Layer_Dropout(rate)


# ------------------------------------------------------------------------------- Optimizer_Adam


def make_layer(weights, biases, dweights, dbiases):
    layer = Layer_Dense(*np.shape(weights))
    layer.weights = np.array(weights, dtype=float)
    layer.biases = np.array(biases, dtype=float)
    layer.dweights = np.array(dweights, dtype=float)
    layer.dbiases = np.array(dbiases, dtype=float)
    return layer


def step(optimizer, layer):
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()


def test_adam_first_step_moves_each_parameter_by_the_learning_rate():
    # After bias correction the first step is lr * g / (|g| + epsilon): the sign of the gradient.
    layer = make_layer([[1.0, -2.0]], [[0.5, 0.5]], [[0.3, -4.0]], [[0.0, 2.0]])
    optimizer = Optimizer_Adam(learning_rate=0.1)
    step(optimizer, layer)
    assert np.allclose(layer.weights, np.array([[0.9, -1.9]]), atol=1e-6)
    assert np.allclose(layer.biases, np.array([[0.5, 0.4]]), atol=1e-6)
    assert optimizer.iterations == 1


def test_adam_second_step_on_a_hand_computed_case():
    # One parameter, gradients 1 then 3, beta_1 = 0.9, beta_2 = 0.999, lr = 0.1.
    # Step 2: m = 0.1 * 0.9 + 0.1 * 3 = 0.39, m_hat = 0.39 / 0.19;
    #         v = 0.001 * 0.999 + 0.001 * 9 = 0.009999, v_hat = 0.009999 / 0.001999.
    layer = make_layer([[1.0]], [[0.0]], [[1.0]], [[0.0]])
    optimizer = Optimizer_Adam(learning_rate=0.1, epsilon=1e-7)
    step(optimizer, layer)
    after_first = 1.0 - 0.1 * 1.0 / (1.0 + 1e-7)
    assert layer.weights[0, 0] == pytest.approx(after_first, rel=1e-12)

    layer.dweights = np.array([[3.0]])
    step(optimizer, layer)
    m_hat = 0.39 / 0.19
    v_hat = 0.009999 / 0.001999
    expected = after_first - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-7)
    assert layer.weights[0, 0] == pytest.approx(expected, rel=1e-12)
    assert optimizer.iterations == 2


def test_adam_decay_shrinks_the_learning_rate_with_the_step_count():
    layer = make_layer([[1.0]], [[0.0]], [[1.0]], [[0.0]])
    optimizer = Optimizer_Adam(learning_rate=0.5, decay=0.1)
    rates = []
    for _ in range(3):
        step(optimizer, layer)
        rates.append(optimizer.current_learning_rate)
    assert rates == pytest.approx([0.5, 0.5 / 1.1, 0.5 / 1.2])


def test_adam_without_decay_keeps_the_learning_rate():
    layer = make_layer([[1.0]], [[0.0]], [[1.0]], [[0.0]])
    optimizer = Optimizer_Adam(learning_rate=0.5)
    for _ in range(3):
        step(optimizer, layer)
    assert optimizer.current_learning_rate == 0.5


def test_adam_keeps_separate_buffers_for_each_layer():
    first = make_layer([[1.0]], [[0.0]], [[1.0]], [[1.0]])
    second = make_layer([[1.0, 1.0]], [[0.0, 0.0]], [[-1.0, 2.0]], [[0.0, 0.0]])
    optimizer = Optimizer_Adam(learning_rate=0.1)
    optimizer.pre_update_params()
    optimizer.update_params(first)
    optimizer.update_params(second)
    optimizer.post_update_params()
    assert first.weight_momentums.shape == (1, 1)
    assert second.weight_cache.shape == (1, 2)
    assert np.allclose(second.weights, np.array([[1.1, 0.9]]), atol=1e-6)


@pytest.mark.parametrize(
    "arguments",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -0.1},
        {"decay": -1e-4},
        {"epsilon": 0.0},
        {"beta_1": 1.0},
        {"beta_2": -0.1},
    ],
)
def test_adam_rejects_impossible_arguments(arguments):
    with pytest.raises(ValueError, match=r"must be|must not be"):
        Optimizer_Adam(**arguments)
