"""The classes of binary_classifier.nn: hand-computed values and numerical gradient checks."""

import math

import numpy as np
import pytest

from binary_classifier.nn import (
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
    Layer_Dense,
    Optimizer_Adam,
    regularization_loss,
)

LN3 = math.log(3.0)


def hand_layer(**regularizers):
    """A 2-input, 3-neuron layer with weights and biases that are easy to multiply by hand."""
    layer = Layer_Dense(2, 3, **regularizers)
    layer.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.biases = np.array([[0.5, -1.0, 2.0]])
    return layer


# --- Layer_Dense ---------------------------------------------------------------------------


def test_dense_forward_hand_computed():
    layer = hand_layer()
    layer.forward(np.array([[1.0, 2.0], [0.0, -1.0]]))
    # Row 1: [1*1 + 2*4 + 0.5, 1*2 + 2*5 - 1, 1*3 + 2*6 + 2]; row 2: [-4 + 0.5, -5 - 1, -6 + 2].
    np.testing.assert_array_equal(layer.output, [[9.5, 11.0, 17.0], [-3.5, -6.0, -4.0]])
    assert layer.output.shape == (2, 3)


def test_dense_backward_hand_computed():
    layer = hand_layer()
    layer.forward(np.array([[1.0, 2.0], [3.0, 4.0]]))
    layer.backward(np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]))
    # dweights = inputs.T @ dvalues, dbiases = column sums, dinputs = dvalues @ weights.T.
    np.testing.assert_array_equal(layer.dweights, [[7.0, 3.0, -1.0], [10.0, 4.0, -2.0]])
    np.testing.assert_array_equal(layer.dbiases, [[3.0, 1.0, -1.0]])
    np.testing.assert_array_equal(layer.dinputs, [[-2.0, -2.0], [4.0, 13.0]])


@pytest.mark.parametrize(
    ("init", "scale"),
    [
        ("he", math.sqrt(2.0 / 2)),
        ("xavier", math.sqrt(2.0 / (2 + 16))),
        ("glorot", math.sqrt(2.0 / (2 + 16))),
        ("small", 0.01),
    ],
)
def test_dense_initialisation_scales_the_standard_normal_draw(init, scale):
    np.random.seed(7)
    layer = Layer_Dense(2, 16, init=init)
    expected = scale * np.random.RandomState(7).randn(2, 16)
    np.testing.assert_allclose(layer.weights, expected, rtol=1e-15)
    np.testing.assert_array_equal(layer.biases, np.zeros((1, 16)))


def test_dense_default_initialisation_is_he():
    np.random.seed(3)
    default = Layer_Dense(16, 16).weights
    np.random.seed(3)
    he = Layer_Dense(16, 16, init="he").weights
    np.testing.assert_array_equal(default, he)
    # 256 draws scaled by sqrt(2 / 16) = 0.354: the sample deviation sits near it.
    assert 0.30 < default.std() < 0.41


def test_dense_rejects_an_unknown_initialisation():
    with pytest.raises(ValueError, match=r"unknown init: 'lecun'"):
        Layer_Dense(2, 3, init="lecun")


def test_dense_backward_matches_numerical_gradient_with_every_regulariser(numerical_gradient):
    rng = np.random.default_rng(0)
    layer = Layer_Dense(
        4,
        3,
        weight_regularizer_l1=0.03,
        weight_regularizer_l2=0.05,
        bias_regularizer_l1=0.02,
        bias_regularizer_l2=0.04,
    )
    layer.weights = rng.normal(size=(4, 3))
    layer.biases = rng.normal(size=(1, 3))
    inputs = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 3))

    def loss():
        layer.forward(inputs)
        return float(np.sum(layer.output * upstream)) + regularization_loss(layer)

    loss()
    layer.backward(upstream)
    np.testing.assert_allclose(
        layer.dweights, numerical_gradient(loss, layer.weights), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        layer.dbiases, numerical_gradient(loss, layer.biases), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        layer.dinputs, numerical_gradient(loss, inputs), rtol=1e-6, atol=1e-8
    )


def test_dense_l1_gradient_is_the_sign_of_each_weight():
    layer = hand_layer(weight_regularizer_l1=0.5, bias_regularizer_l1=0.25)
    layer.weights[0, 0] = -1.0
    layer.forward(np.zeros((1, 2)))
    layer.backward(np.zeros((1, 3)))
    # With zero inputs and zero upstream gradient only the penalty terms remain.
    np.testing.assert_array_equal(layer.dweights, [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(layer.dbiases, [[0.25, -0.25, 0.25]])


# --- Activation_ReLU -----------------------------------------------------------------------


def test_relu_forward_and_backward_hand_computed():
    relu = Activation_ReLU()
    relu.forward(np.array([[-2.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(relu.output, [[0.0, 0.0, 3.0]])

    upstream = np.array([[5.0, 7.0, 11.0]])
    relu.backward(upstream)
    # The gradient passes only where the input was positive; an input of exactly 0 blocks it.
    np.testing.assert_array_equal(relu.dinputs, [[0.0, 0.0, 11.0]])
    np.testing.assert_array_equal(upstream, [[5.0, 7.0, 11.0]])


# --- Activation_Sigmoid --------------------------------------------------------------------


def test_sigmoid_forward_hand_computed():
    sigmoid = Activation_Sigmoid()
    sigmoid.forward(np.array([[0.0, LN3, -LN3]]))
    # sigmoid(ln 3) = 3 / 4 and sigmoid(-ln 3) = 1 / 4.
    np.testing.assert_allclose(sigmoid.output, [[0.5, 0.75, 0.25]], rtol=1e-15)


def test_sigmoid_is_stable_at_both_extremes():
    sigmoid = Activation_Sigmoid()
    # The suite turns warnings into errors, so an overflow in exp would fail this test.
    sigmoid.forward(np.array([[-1000.0, -40.0, 40.0, 1000.0]]))
    assert sigmoid.output[0, 0] == 0.0
    assert sigmoid.output[0, 3] == 1.0
    assert 0.0 < sigmoid.output[0, 1] < 1e-17
    assert np.isfinite(sigmoid.output).all()


def test_sigmoid_symmetry():
    z = np.linspace(-6, 6, 25).reshape(-1, 1)
    plus, minus = Activation_Sigmoid(), Activation_Sigmoid()
    plus.forward(z)
    minus.forward(-z)
    # sigmoid(-z) = 1 - sigmoid(z).
    np.testing.assert_allclose(plus.output + minus.output, 1.0, rtol=1e-12)
    assert plus.output.shape == (25, 1)


def test_sigmoid_of_float32_input_is_float64_at_float32_precision():
    sigmoid = Activation_Sigmoid()
    sigmoid.forward(np.array([[0.0, LN3, -LN3]], dtype=np.float32))
    assert sigmoid.output.dtype == np.float64
    # exp is taken in the precision of the input, so about seven digits survive.
    np.testing.assert_allclose(sigmoid.output, [[0.5, 0.75, 0.25]], rtol=1e-6)


def test_sigmoid_backward_matches_numerical_gradient(numerical_gradient):
    sigmoid = Activation_Sigmoid()
    z = np.array([[-2.0, -0.3, 0.0, 0.7, 3.0]])
    upstream = np.array([[1.0, -2.0, 0.5, 3.0, -1.0]])

    def loss():
        sigmoid.forward(z)
        return float(np.sum(sigmoid.output * upstream))

    loss()
    sigmoid.backward(upstream)
    np.testing.assert_allclose(sigmoid.dinputs, numerical_gradient(loss, z), rtol=1e-7)
    # At z = 0 the derivative is sigmoid * (1 - sigmoid) = 1 / 4.
    assert sigmoid.dinputs[0, 2] == pytest.approx(0.5 * 0.25)


# --- Activation_Sigmoid_Loss_BinaryCrossentropy --------------------------------------------


def test_combined_loss_hand_computed():
    combined = Activation_Sigmoid_Loss_BinaryCrossentropy()
    logits = np.array([[0.0], [LN3], [-LN3]])
    loss = combined.forward(logits, np.array([1, 1, 0]))
    # Probabilities 1/2, 3/4, 1/4 against targets 1, 1, 0: the three terms are
    # -ln(1/2), -ln(3/4), -ln(3/4).
    assert loss == pytest.approx((math.log(2.0) + 2 * math.log(4.0 / 3.0)) / 3, rel=1e-14)
    np.testing.assert_allclose(combined.output, [[0.5], [0.75], [0.25]], rtol=1e-15)
    assert isinstance(loss, float)


def test_combined_backward_is_prediction_minus_target_over_n():
    combined = Activation_Sigmoid_Loss_BinaryCrossentropy()
    y = np.array([1, 1, 0])
    combined.forward(np.array([[0.0], [LN3], [-LN3]]), y)
    combined.backward(combined.output, y)
    np.testing.assert_allclose(combined.dinputs, [[-0.5 / 3], [-0.25 / 3], [0.25 / 3]], rtol=1e-14)
    assert combined.dinputs.shape == (3, 1)


def test_combined_backward_matches_numerical_gradient(numerical_gradient):
    combined = Activation_Sigmoid_Loss_BinaryCrossentropy()
    logits = np.array([[-1.5], [0.2], [2.0], [-0.4]])
    y = np.array([0, 1, 1, 0])

    def loss():
        return combined.forward(logits, y)

    loss()
    combined.backward(combined.output, y)
    np.testing.assert_allclose(combined.dinputs, numerical_gradient(loss, logits), rtol=1e-7)


def test_combined_accepts_labels_as_a_vector_or_a_column():
    logits = np.array([[0.3], [-1.2], [2.5]])
    as_vector, as_column = (Activation_Sigmoid_Loss_BinaryCrossentropy() for _ in range(2))
    y = np.array([1, 0, 1])
    assert as_vector.forward(logits, y) == as_column.forward(logits, y.reshape(-1, 1))
    as_vector.backward(as_vector.output, y)
    as_column.backward(as_column.output, y.reshape(-1, 1))
    np.testing.assert_array_equal(as_vector.dinputs, as_column.dinputs)


def test_combined_loss_is_clipped_when_confidently_wrong():
    combined = Activation_Sigmoid_Loss_BinaryCrossentropy()
    loss = combined.forward(np.array([[1000.0], [-1000.0]]), np.array([0, 1]))
    # Both probabilities are clipped to within 1e-7 of the wrong end: each term is -ln(1e-7).
    assert loss == pytest.approx(-math.log(1e-7), rel=1e-9)


# --- Optimizer_Adam ------------------------------------------------------------------------


def scalar_layer(weight, bias=0.0):
    layer = Layer_Dense(1, 1)
    layer.weights = np.array([[weight]])
    layer.biases = np.array([[bias]])
    return layer


def adam_step(optimizer, layer, dweight, dbias=0.0):
    layer.dweights = np.array([[dweight]])
    layer.dbiases = np.array([[dbias]])
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()


def test_adam_first_two_steps_hand_computed():
    optimizer = Optimizer_Adam(learning_rate=0.1)
    layer = scalar_layer(1.0, bias=2.0)

    # Step 1, gradient 1: m = 0.1, m_hat = 1; v = 0.001, v_hat = 1; step = 0.1 / (1 + 1e-7).
    adam_step(optimizer, layer, dweight=1.0, dbias=-1.0)
    after_one = 1.0 - 0.1 * 1.0 / (1.0 + 1e-7)
    assert layer.weights[0, 0] == pytest.approx(after_one, rel=1e-12)
    assert layer.biases[0, 0] == pytest.approx(2.0 + 0.1 * 1.0 / (1.0 + 1e-7), rel=1e-12)
    assert optimizer.iterations == 1

    # Step 2, gradient 3: m = 0.9 * 0.1 + 0.1 * 3 = 0.39, m_hat = 0.39 / (1 - 0.81);
    # v = 0.999 * 0.001 + 0.001 * 9 = 0.009999, v_hat = 0.009999 / (1 - 0.998001).
    adam_step(optimizer, layer, dweight=3.0)
    m_hat = 0.39 / 0.19
    v_hat = 0.009999 / 0.001999
    after_two = after_one - 0.1 * m_hat / (math.sqrt(v_hat) + 1e-7)
    assert layer.weights[0, 0] == pytest.approx(after_two, rel=1e-12)
    assert optimizer.iterations == 2


def test_adam_leaves_a_parameter_with_zero_gradient_alone():
    optimizer = Optimizer_Adam(learning_rate=0.1)
    layer = scalar_layer(1.0, bias=2.0)
    adam_step(optimizer, layer, dweight=0.0, dbias=0.0)
    assert layer.weights[0, 0] == 1.0
    assert layer.biases[0, 0] == 2.0


def test_adam_learning_rate_decay():
    optimizer = Optimizer_Adam(learning_rate=1.0, decay=0.1)
    layer = scalar_layer(0.0)
    rates = []
    for _ in range(3):
        adam_step(optimizer, layer, dweight=1.0)
        rates.append(optimizer.current_learning_rate)
    # learning_rate / (1 + decay * iterations) with iterations 0, 1, 2.
    assert rates == pytest.approx([1.0, 1.0 / 1.1, 1.0 / 1.2])


def test_adam_without_decay_keeps_the_learning_rate():
    optimizer = Optimizer_Adam(learning_rate=0.01)
    layer = scalar_layer(0.0)
    for _ in range(3):
        adam_step(optimizer, layer, dweight=1.0)
    assert optimizer.current_learning_rate == 0.01


def test_adam_minimises_a_quadratic():
    optimizer = Optimizer_Adam(learning_rate=0.1)
    layer = scalar_layer(5.0)
    for _ in range(500):
        # Gradient of (w - 2)^2.
        adam_step(optimizer, layer, dweight=2.0 * (layer.weights[0, 0] - 2.0))
    assert layer.weights[0, 0] == pytest.approx(2.0, abs=1e-3)


# --- regularization_loss -------------------------------------------------------------------


def test_regularization_loss_hand_computed():
    layer = Layer_Dense(
        2,
        2,
        weight_regularizer_l1=0.1,
        weight_regularizer_l2=0.01,
        bias_regularizer_l1=0.2,
        bias_regularizer_l2=0.5,
    )
    layer.weights = np.array([[1.0, -2.0], [3.0, -4.0]])
    layer.biases = np.array([[0.5, -0.5]])
    # 0.1 * 10 + 0.01 * 30 + 0.2 * 1 + 0.5 * 0.5 = 1.0 + 0.3 + 0.2 + 0.25.
    assert regularization_loss(layer) == pytest.approx(1.75, rel=1e-14)


def test_regularization_loss_is_zero_without_regularisers():
    layer = hand_layer()
    assert regularization_loss(layer) == 0.0
