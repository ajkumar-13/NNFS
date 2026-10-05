"""The assembled network: shapes, the parameter count, gradients, both modes, the weights file."""

import hashlib
import pickle
import zipfile

import numpy as np
import pytest
from numpy.lib import format as npy_format

from fashion_mnist.model import WEIGHT_KEYS, Network, WeightsError
from helpers import generator_state, numerical_gradient


def small_network(dropout_rate=0.0, l2=0.01):
    """A 6-5-5-3 network with weights large enough for finite differences to resolve."""
    network = Network(n_inputs=6, n_hidden=5, n_classes=3, l2=l2, dropout_rate=dropout_rate)
    for layer in network.dense_layers:
        layer.weights = np.random.randn(*layer.weights.shape) * 0.7
        layer.biases = np.random.randn(*layer.biases.shape) * 0.1
    return network


# ------------------------------------------------------------------------------- the architecture


def test_default_network_is_784_128_128_10_with_118282_parameters():
    network = Network()
    assert [layer.weights.shape for layer in network.dense_layers] == [
        (784, 128),
        (128, 128),
        (128, 10),
    ]
    assert [layer.biases.shape for layer in network.dense_layers] == [(1, 128), (1, 128), (1, 10)]
    assert network.n_inputs == 784
    assert network.n_classes == 10
    # (784 * 128 + 128) + (128 * 128 + 128) + (128 * 10 + 10) = 100,480 + 16,512 + 1,290
    assert network.parameter_count() == 118_282


def test_default_network_penalises_the_hidden_weights_only():
    network = Network()
    assert network.dense1.weight_regularizer_l2 == 5e-4
    assert network.dense2.weight_regularizer_l2 == 5e-4
    assert network.dense3.weight_regularizer_l2 == 0
    for layer in network.dense_layers:
        assert layer.weight_regularizer_l1 == 0
        assert layer.bias_regularizer_l1 == 0
        assert layer.bias_regularizer_l2 == 0
    assert network.dropout1.rate == 0.9
    assert network.dropout2.rate == 0.9


def test_layers_draw_their_weights_in_the_order_dense1_dense2_dense3():
    np.random.seed(0)
    network = Network()
    reference = np.random.RandomState(0)
    assert np.array_equal(network.dense1.weights, 0.01 * reference.randn(784, 128))
    assert np.array_equal(network.dense2.weights, 0.01 * reference.randn(128, 128))
    assert np.array_equal(network.dense3.weights, 0.01 * reference.randn(128, 10))


def test_the_same_seed_builds_the_same_network():
    np.random.seed(7)
    first = Network().weights()
    np.random.seed(7)
    second = Network().weights()
    assert list(first) == list(WEIGHT_KEYS)
    for key in WEIGHT_KEYS:
        assert np.array_equal(first[key], second[key])


# ------------------------------------------------------------------- forward and backward passes


def test_forward_returns_the_mean_cross_entropy_and_leaves_probabilities():
    network = small_network()
    X = np.random.randn(4, 6)
    y = np.array([0, 2, 1, 1])
    loss = network.forward(X, y, training=False)
    probabilities = network.loss_activation.output
    assert probabilities.shape == (4, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert loss == pytest.approx(float(np.mean(-np.log(probabilities[range(4), y]))))
    assert np.array_equal(network.predictions(), np.argmax(probabilities, axis=1))


def test_untrained_default_network_starts_at_the_loss_of_a_uniform_guess():
    network = Network()
    loss = network.forward(np.random.rand(16, 784), np.random.randint(0, 10, size=16), False)
    assert loss == pytest.approx(np.log(10), abs=0.01)


def test_backward_matches_finite_differences_of_loss_plus_penalty():
    network = small_network(dropout_rate=0.0, l2=0.01)
    X = np.random.randn(7, 6)
    y = np.array([0, 2, 1, 1, 0, 2, 2])

    def total_loss():
        return network.forward(X, y, training=False) + network.regularization_loss()

    network.forward(X, y, training=True)
    network.backward(y)
    for layer in network.dense_layers:
        assert np.allclose(layer.dweights, numerical_gradient(total_loss, layer.weights), atol=1e-6)
        assert np.allclose(layer.dbiases, numerical_gradient(total_loss, layer.biases), atol=1e-6)


def test_regularization_loss_sums_the_hidden_layers():
    network = small_network(l2=0.01)
    expected = 0.01 * np.sum(network.dense1.weights**2) + 0.01 * np.sum(network.dense2.weights**2)
    assert network.regularization_loss() == pytest.approx(expected)


def test_training_mode_applies_dropout_and_evaluation_mode_does_not():
    network = small_network(dropout_rate=0.5)
    X = np.random.randn(64, 6)
    y = np.random.randint(0, 3, size=64)

    network.forward(X, y, training=True)
    first = network.loss_activation.output.copy()
    assert np.any(network.dropout1.output == 0)
    network.forward(X, y, training=True)
    assert not np.array_equal(first, network.loss_activation.output)

    before = generator_state()
    network.forward(X, y, training=False)
    evaluated = network.loss_activation.output.copy()
    network.forward(X, y, training=False)
    assert np.array_equal(evaluated, network.loss_activation.output)
    assert generator_state() == before
    assert np.array_equal(network.dropout1.output, network.activation1.output)


# -------------------------------------------------------------------------------- the weights file


def test_weights_survive_a_round_trip_through_the_file(tmp_path):
    network = Network()
    path = tmp_path / "weights.npz"
    network.save(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]

    restored = Network()
    restored.load(path)
    for key in WEIGHT_KEYS:
        assert np.array_equal(restored.weights()[key], network.weights()[key])
        assert restored.weights()[key].dtype == np.float64

    X = np.random.rand(5, 784)
    y = np.array([3, 1, 4, 1, 5])
    assert restored.forward(X, y, training=False) == network.forward(X, y, training=False)


def test_fingerprint_is_equal_exactly_when_the_weights_are(tmp_path):
    network = Network()
    path = tmp_path / "weights.npz"
    network.save(path)
    restored = Network()
    assert restored.fingerprint() != network.fingerprint()
    restored.load(path)
    assert restored.fingerprint() == network.fingerprint()
    assert len(network.fingerprint()) == 64

    # The smallest possible change to one weight changes the fingerprint.
    restored.dense2.weights[5, 7] = np.nextafter(restored.dense2.weights[5, 7], np.inf)
    assert restored.fingerprint() != network.fingerprint()


def test_fingerprint_of_a_known_network():
    network = Network(n_inputs=2, n_hidden=1, n_classes=1)
    for layer in network.dense_layers:
        layer.weights = np.zeros_like(layer.weights)
    # 4 weights and 3 biases, all zero: the SHA-256 of 56 zero bytes.
    assert network.fingerprint() == hashlib.sha256(bytes(56)).hexdigest()


def test_the_weights_file_holds_six_plain_arrays_and_no_pickle(tmp_path):
    path = tmp_path / "weights.npz"
    Network().save(path)
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(f"{key}.npy" for key in WEIGHT_KEYS)
    with np.load(path, allow_pickle=False) as archive:
        assert sum(archive[key].size for key in WEIGHT_KEYS) == 118_282


def test_load_reports_a_missing_file_with_the_command_that_creates_it(tmp_path):
    with pytest.raises(WeightsError, match=r"does not exist(.|\n)*fashion_mnist\.train"):
        Network().load(tmp_path / "absent.npz")


def test_load_refuses_a_pickle_checkpoint_without_reading_it(tmp_path):
    path = tmp_path / "fashion_mnist_weights.pkl"
    path.write_bytes(pickle.dumps({"dense1": "anything"}))
    with pytest.raises(WeightsError, match=r"not an \.npz file(.|\n)*unpickling can run code"):
        Network().load(path)


def test_load_refuses_a_file_that_is_not_an_archive(tmp_path):
    path = tmp_path / "weights.npz"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(WeightsError, match="not a readable weights file"):
        Network().load(path)


def test_load_refuses_an_archive_with_other_arrays(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, something_else=np.zeros(3))
    with pytest.raises(WeightsError, match="does not hold exactly the arrays"):
        Network().load(path)


def test_load_refuses_weights_of_another_architecture(tmp_path):
    path = tmp_path / "weights.npz"
    Network(n_hidden=64).save(path)
    with pytest.raises(WeightsError, match=r"dense1_weights has shape \(784, 64\)"):
        Network().load(path)


def test_load_reads_the_declared_shape_before_any_data(tmp_path):
    # A header that claims a million billion values, with no data behind it. A loader that
    # trusted the header would try to allocate the array; this one stops at the shape.
    path = tmp_path / "weights.npz"
    arrays = Network(n_inputs=6, n_hidden=5, n_classes=3).weights()
    with zipfile.ZipFile(path, "w") as archive:
        for key, array in arrays.items():
            with archive.open(f"{key}.npy", "w") as member:
                if key == "dense3_biases":
                    header = {"descr": "<f8", "fortran_order": False, "shape": (10**9, 10**6)}
                    npy_format.write_array_header_1_0(member, header)
                else:
                    npy_format.write_array(member, array)
    with pytest.raises(WeightsError, match=r"dense3_biases has shape \(1000000000, 1000000\)"):
        Network(n_inputs=6, n_hidden=5, n_classes=3).load(path)


def test_load_refuses_an_archive_with_a_repeated_array(tmp_path):
    path = tmp_path / "weights.npz"
    arrays = Network(n_inputs=6, n_hidden=5, n_classes=3).weights()
    np.savez(path, **arrays)
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("dense1_biases.npy", b"anything")
    with pytest.raises(WeightsError, match="does not hold exactly the arrays"):
        Network(n_inputs=6, n_hidden=5, n_classes=3).load(path)


def test_load_refuses_a_member_that_is_not_an_array(tmp_path):
    path = tmp_path / "weights.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key in WEIGHT_KEYS:
            archive.writestr(f"{key}.npy", b"not an array at all")
    with pytest.raises(WeightsError, match="not a readable weights file"):
        Network().load(path)


def test_load_refuses_an_unknown_array_format_version(tmp_path):
    path = tmp_path / "weights.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key in WEIGHT_KEYS:
            archive.writestr(f"{key}.npy", npy_format.magic(3, 0) + b"\x00" * 16)
    with pytest.raises(WeightsError, match=r"unsupported \.npy format version \(3, 0\)"):
        Network().load(path)


def test_load_accepts_32_bit_floats_and_computes_in_64_bit(tmp_path):
    path = tmp_path / "weights.npz"
    network = Network(n_inputs=6, n_hidden=5, n_classes=3)
    np.savez(path, **{key: array.astype(np.float32) for key, array in network.weights().items()})
    restored = Network(n_inputs=6, n_hidden=5, n_classes=3)
    restored.load(path)
    for key, array in restored.weights().items():
        assert array.dtype == np.float64
        assert np.array_equal(array, network.weights()[key].astype(np.float32))


def test_load_reads_the_long_header_format(tmp_path):
    path = tmp_path / "weights.npz"
    network = Network(n_inputs=6, n_hidden=5, n_classes=3)
    with zipfile.ZipFile(path, "w") as archive:
        for key, array in network.weights().items():
            with archive.open(f"{key}.npy", "w") as member:
                npy_format.write_array(member, array, version=(2, 0))
    restored = Network(n_inputs=6, n_hidden=5, n_classes=3)
    restored.load(path)
    assert restored.fingerprint() == network.fingerprint()


def test_load_refuses_arrays_that_are_not_floating_point(tmp_path):
    path = tmp_path / "weights.npz"
    arrays = Network(n_inputs=6, n_hidden=5, n_classes=3).weights()
    arrays["dense2_biases"] = np.zeros((1, 5), dtype=np.int64)
    np.savez(path, **arrays)
    with pytest.raises(WeightsError, match="dense2_biases has type int64"):
        Network(n_inputs=6, n_hidden=5, n_classes=3).load(path)


def test_load_refuses_values_that_are_not_finite(tmp_path):
    path = tmp_path / "weights.npz"
    arrays = Network(n_inputs=6, n_hidden=5, n_classes=3).weights()
    arrays["dense1_weights"] = arrays["dense1_weights"].copy()
    arrays["dense1_weights"][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(WeightsError, match="dense1_weights contains values that are not finite"):
        Network(n_inputs=6, n_hidden=5, n_classes=3).load(path)


def test_load_refuses_object_arrays_instead_of_unpickling_them(tmp_path):
    path = tmp_path / "weights.npz"
    arrays = Network(n_inputs=6, n_hidden=5, n_classes=3).weights()
    arrays["dense3_biases"] = np.array([[{"a": 1}, None, "text"]], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(WeightsError, match="dense3_biases has type object"):
        Network(n_inputs=6, n_hidden=5, n_classes=3).load(path)


def test_a_failed_load_leaves_the_network_unchanged(tmp_path):
    path = tmp_path / "weights.npz"
    Network(n_hidden=64).save(path)
    network = Network()
    before = {key: array.copy() for key, array in network.weights().items()}
    with pytest.raises(WeightsError):
        network.load(path)
    for key in WEIGHT_KEYS:
        assert np.array_equal(network.weights()[key], before[key])
