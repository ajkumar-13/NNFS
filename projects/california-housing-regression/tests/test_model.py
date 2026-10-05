"""The assembled network: shapes, the parameter count, gradients, the linear output, the file."""

import hashlib
import pickle
import zipfile

import numpy as np
import pytest
from numpy.lib import format as npy_format

from california_housing_regression.data import Scaler
from california_housing_regression.model import (
    FILE_KEYS,
    SCALER_KEYS,
    WEIGHT_KEYS,
    Network,
    WeightsError,
)
from helpers import generator_state, numerical_gradient


def small_network(l2=0.01):
    """A 6-5-5-1 network with weights large enough for finite differences to resolve."""
    network = Network(n_inputs=6, n_hidden=5, n_outputs=1, l2=l2)
    for layer in network.dense_layers:
        layer.weights = np.random.randn(*layer.weights.shape) * 0.7
        layer.biases = np.random.randn(*layer.biases.shape) * 0.1
    return network


def scaler_for(network, y_mean=2.0, y_std=1.25):
    features = network.n_inputs
    return Scaler(
        x_mean=np.linspace(-1, 1, features).astype(np.float32),
        x_std=np.linspace(0.5, 2, features).astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
    )


def file_arrays(network, scaler):
    """The ten arrays of a weights file, as plain arrays a test can alter before saving."""
    return {key: array.copy() for key, array in network.weights().items()} | {
        "x_mean": scaler.x_mean.copy(),
        "x_std": scaler.x_std.copy(),
        "y_mean": np.float64(scaler.y_mean),
        "y_std": np.float64(scaler.y_std),
    }


# ------------------------------------------------------------------------------- the architecture


def test_default_network_is_8_64_64_1_with_4801_parameters():
    network = Network()
    assert [layer.weights.shape for layer in network.dense_layers] == [(8, 64), (64, 64), (64, 1)]
    assert [layer.biases.shape for layer in network.dense_layers] == [(1, 64), (1, 64), (1, 1)]
    assert network.n_inputs == 8
    assert network.n_outputs == 1
    # (8 * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1) = 576 + 4,160 + 65
    assert network.parameter_count() == 4_801


def test_default_network_penalises_the_hidden_weights_only():
    network = Network()
    assert network.dense1.weight_regularizer_l2 == 1e-4
    assert network.dense2.weight_regularizer_l2 == 1e-4
    assert network.dense3.weight_regularizer_l2 == 0
    for layer in network.dense_layers:
        assert layer.weight_regularizer_l1 == 0
        assert layer.bias_regularizer_l1 == 0
        assert layer.bias_regularizer_l2 == 0


def test_layers_draw_their_weights_in_the_order_dense1_dense2_dense3():
    np.random.seed(0)
    network = Network()
    reference = np.random.RandomState(0)
    assert np.array_equal(network.dense1.weights, 0.01 * reference.randn(8, 64))
    assert np.array_equal(network.dense2.weights, 0.01 * reference.randn(64, 64))
    assert np.array_equal(network.dense3.weights, 0.01 * reference.randn(64, 1))


def test_the_same_seed_builds_the_same_network():
    np.random.seed(7)
    first = Network().weights()
    np.random.seed(7)
    second = Network().weights()
    assert list(first) == list(WEIGHT_KEYS)
    for key in WEIGHT_KEYS:
        assert np.array_equal(first[key], second[key])


# ------------------------------------------------------------------- forward and backward passes


def test_predict_on_a_hand_computed_case():
    network = Network(n_inputs=2, n_hidden=2, n_outputs=1)
    network.dense1.weights = np.array([[1.0, -1.0], [2.0, 1.0]])
    network.dense1.biases = np.array([[0.0, 1.0]])
    network.dense2.weights = np.array([[1.0, 0.0], [1.0, -2.0]])
    network.dense2.biases = np.array([[-1.0, 0.5]])
    network.dense3.weights = np.array([[3.0], [10.0]])
    network.dense3.biases = np.array([[-20.0]])

    # Row (1, 1): dense1 gives (3, 1), ReLU keeps both; dense2 gives (3, -1.5), ReLU (3, 0);
    # the output is 3 * 3 + 10 * 0 - 20 = -11.
    # Row (1, -2): dense1 gives (-3, -2), ReLU (0, 0); dense2 gives (-1, 0.5), ReLU (0, 0.5);
    # the output is 10 * 0.5 - 20 = -15.
    output = network.predict(np.array([[1.0, 1.0], [1.0, -2.0]]))
    assert output.shape == (2, 1)
    assert output.tolist() == [[-11.0], [-15.0]]


def test_the_output_is_linear_and_unbounded():
    network = small_network()
    X = np.random.randn(50, 6)
    before = network.predict(X).copy()
    # No activation follows the last dense layer: doubling its weights and bias doubles the output,
    # shifting its bias shifts the output, and nothing squashes values into a fixed range.
    network.dense3.weights *= 2
    network.dense3.biases *= 2
    assert np.allclose(network.predict(X), 2 * before)
    network.dense3.biases += 1e6
    shifted = network.predict(X)
    assert np.allclose(shifted, 2 * before + 1e6)
    assert shifted.min() > 1
    assert before.min() < 0 < before.max()


def test_forward_returns_the_mean_squared_error_and_leaves_the_predictions():
    network = small_network()
    X = np.random.randn(4, 6)
    y = np.array([0.5, -1.0, 2.0, 0.0])
    loss = network.forward(X, y)
    predictions = network.dense3.output
    assert predictions.shape == (4, 1)
    assert loss == pytest.approx(float(np.mean((predictions[:, 0] - y) ** 2)))
    assert np.array_equal(network.predict(X), predictions)


def test_forward_takes_float32_data_and_a_single_row():
    network = small_network()
    X = np.random.randn(1, 6).astype(np.float32)
    y = np.array([0.25], dtype=np.float32)
    loss = network.forward(X, y)
    assert isinstance(loss, float)
    assert network.dense3.output.dtype == np.float64


def test_backward_matches_finite_differences_of_the_penalised_loss():
    network = small_network(l2=0.01)
    X = np.random.randn(7, 6)
    y = np.random.randn(7)

    def total_loss():
        return network.forward(X, y) + float(network.regularization_loss())

    network.forward(X, y)
    network.backward()
    for layer in network.dense_layers:
        analytic_weights, analytic_biases = layer.dweights.copy(), layer.dbiases.copy()
        assert np.allclose(
            analytic_weights, numerical_gradient(total_loss, layer.weights), atol=1e-6
        )
        assert np.allclose(analytic_biases, numerical_gradient(total_loss, layer.biases), atol=1e-6)


def test_backward_reaches_the_inputs():
    network = small_network(l2=0.0)
    X = np.random.randn(5, 6)
    y = np.random.randn(5)
    network.forward(X, y)
    network.backward()
    numerical = numerical_gradient(lambda: network.forward(X, y), X)
    assert np.allclose(network.dense1.dinputs, numerical, atol=1e-6)


def test_regularization_loss_sums_the_two_hidden_penalties():
    network = small_network(l2=0.01)
    expected = 0.01 * (np.sum(network.dense1.weights**2) + np.sum(network.dense2.weights**2))
    assert network.regularization_loss() == pytest.approx(expected)
    assert small_network(l2=0.0).regularization_loss() == 0.0


def test_the_network_draws_no_random_numbers_after_it_is_built():
    network = small_network()
    X, y = np.random.randn(6, 6), np.random.randn(6)
    state = generator_state()
    network.forward(X, y)
    network.backward()
    network.predict(X)
    assert generator_state() == state


# ---------------------------------------------------------------------------------- fingerprint


def test_fingerprint_is_the_sha256_of_the_six_parameter_arrays():
    network = small_network()
    digest = hashlib.sha256()
    for key in WEIGHT_KEYS:
        digest.update(network.weights()[key].astype("<f8").tobytes())
    assert network.fingerprint() == digest.hexdigest()
    assert len(network.fingerprint()) == 64


def test_fingerprint_changes_with_any_parameter():
    network = small_network()
    before = network.fingerprint()
    network.dense3.biases[0, 0] = np.nextafter(network.dense3.biases[0, 0], np.inf)
    assert network.fingerprint() != before


# ------------------------------------------------------------------------------- the weights file


def test_save_and_load_round_trip_the_parameters_and_the_scaler(tmp_path):
    network = small_network()
    scaler = scaler_for(network)
    path = tmp_path / "weights.npz"
    network.save(path, scaler)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]

    restored = Network(n_inputs=6, n_hidden=5, n_outputs=1)
    assert restored.fingerprint() != network.fingerprint()
    loaded = restored.load(path)

    assert restored.fingerprint() == network.fingerprint()
    for key in WEIGHT_KEYS:
        assert np.array_equal(restored.weights()[key], network.weights()[key])
        assert restored.weights()[key].dtype == np.float64
    assert np.array_equal(loaded.x_mean, scaler.x_mean)
    assert np.array_equal(loaded.x_std, scaler.x_std)
    assert loaded.x_mean.dtype == np.float32
    assert loaded.y_mean == scaler.y_mean
    assert loaded.y_std == scaler.y_std
    assert isinstance(loaded.y_mean, float)
    assert loaded.matches(scaler)

    X = np.random.randn(3, 6)
    assert np.array_equal(restored.predict(X), network.predict(X))


def test_the_file_holds_ten_plain_arrays_and_no_pickle(tmp_path):
    network = small_network()
    path = tmp_path / "weights.npz"
    network.save(path, scaler_for(network))
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(f"{key}.npy" for key in FILE_KEYS)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind == "f" for key in FILE_KEYS)
        assert archive["y_mean"].shape == ()
        assert archive["x_std"].shape == (6,)
    assert FILE_KEYS == WEIGHT_KEYS + SCALER_KEYS


def test_save_replaces_an_existing_file_whole(tmp_path):
    path = tmp_path / "weights.npz"
    first, second = small_network(), small_network()
    first.save(path, scaler_for(first))
    second.save(path, scaler_for(second, y_mean=9.0))
    restored = Network(n_inputs=6, n_hidden=5, n_outputs=1)
    assert restored.load(path).y_mean == 9.0
    assert restored.fingerprint() == second.fingerprint()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["weights.npz"]


def test_load_names_the_training_command_when_the_file_is_missing(tmp_path):
    with pytest.raises(WeightsError, match=r"does not exist. Train first:\n.*regression\.train"):
        Network().load(tmp_path / "absent.npz")


def test_load_refuses_a_pickle_checkpoint_without_opening_it(tmp_path):
    class Canary:
        def __reduce__(self):
            return (pytest.fail, ("the pickle was opened",))

    path = tmp_path / "cal_housing_weights.pkl"
    path.write_bytes(pickle.dumps(Canary()))
    with pytest.raises(WeightsError, match=r"is not an \.npz file. Pickle checkpoints"):
        Network().load(path)


def test_load_refuses_a_pickle_renamed_to_npz(tmp_path):
    path = tmp_path / "weights.npz"
    path.write_bytes(pickle.dumps({"dense1": (np.zeros((8, 64)), np.zeros((1, 64)))}))
    with pytest.raises(WeightsError, match="is not a readable weights file"):
        Network().load(path)


def failing_load(tmp_path, arrays, **kwargs):
    """Save ``arrays`` as given and return the error a matching network raises on loading them."""
    path = tmp_path / "weights.npz"
    np.savez(path, **arrays, **kwargs)
    network = small_network()
    before = network.fingerprint()
    with pytest.raises(WeightsError) as error:
        network.load(path)
    # A refused file leaves the network as it was.
    assert network.fingerprint() == before
    return str(error.value)


def test_load_refuses_a_file_with_a_missing_or_an_extra_array(tmp_path):
    network = small_network()
    arrays = file_arrays(network, scaler_for(network))
    without = {key: value for key, value in arrays.items() if key != "y_std"}
    assert "does not hold exactly the arrays" in failing_load(tmp_path, without)
    assert "extra.npy" in failing_load(tmp_path, arrays, extra=np.zeros(3))
    assert "it holds nothing" in failing_load(tmp_path, {})


def test_load_refuses_weights_of_another_architecture(tmp_path):
    network = small_network()
    path = tmp_path / "weights.npz"
    network.save(path, scaler_for(network))
    with pytest.raises(WeightsError, match=r"dense1_weights has shape \(6, 5\), but this network"):
        Network().load(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("dense2_biases", np.zeros((1, 4)), "dense2_biases has shape (1, 4)"),
        ("x_mean", np.zeros(5, np.float32), "x_mean has shape (5,)"),
        ("y_mean", np.zeros(1), "y_mean has shape (1,)"),
        ("dense3_weights", np.zeros((5, 1), dtype=np.int64), "expected a floating-point type"),
        ("y_std", np.int64(1), "y_std has type int64"),
        ("dense1_weights", np.full((6, 5), np.nan), "dense1_weights contains values that are not"),
        ("dense3_biases", np.full((1, 1), np.inf), "dense3_biases contains values that are not"),
        ("x_mean", np.full(6, np.inf, np.float32), "x_mean contains values that are not finite"),
        ("x_mean", np.full(6, 1e300), "x_mean contains values that are not finite"),
        ("y_mean", np.float64(np.nan), "y_mean contains values that are not finite"),
        ("x_std", np.zeros(6, np.float32), "x_std must be positive"),
        ("y_std", np.float64(-1.0), "y_std must be positive"),
    ],
)
def test_load_refuses_arrays_that_do_not_fit(tmp_path, key, value, message):
    network = small_network()
    arrays = file_arrays(network, scaler_for(network))
    arrays[key] = value
    assert message in failing_load(tmp_path, arrays)


def test_load_refuses_an_object_array_instead_of_unpickling_it(tmp_path):
    network = small_network()
    arrays = file_arrays(network, scaler_for(network))
    arrays["dense1_weights"] = np.empty((6, 5), dtype=object)
    path = tmp_path / "weights.npz"
    np.savez(path, **arrays)
    with pytest.raises(WeightsError, match="expected a floating-point type"):
        network.load(path)


@pytest.mark.parametrize("content", [b"", b"not a zip archive at all", b"PK\x03\x04 truncated"])
def test_load_refuses_a_file_that_is_not_an_archive(tmp_path, content):
    path = tmp_path / "weights.npz"
    path.write_bytes(content)
    with pytest.raises(WeightsError, match="is not a readable weights file"):
        Network().load(path)


def test_load_refuses_a_member_that_is_not_an_npy_file(tmp_path):
    path = tmp_path / "weights.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for key in FILE_KEYS:
            archive.writestr(f"{key}.npy", b"these bytes are not an array")
    with pytest.raises(WeightsError, match="is not a readable weights file"):
        Network().load(path)


def test_load_refuses_an_unsupported_npy_version(tmp_path, monkeypatch):
    network = small_network()
    path = tmp_path / "weights.npz"
    network.save(path, scaler_for(network))
    monkeypatch.setattr(npy_format, "read_magic", lambda handle: (3, 0))
    with pytest.raises(WeightsError, match=r"unsupported \.npy format version \(3, 0\)"):
        network.load(path)


def test_load_reads_version_two_headers(tmp_path, monkeypatch):
    network = small_network()
    path = tmp_path / "weights.npz"
    scaler = scaler_for(network)
    with zipfile.ZipFile(path, "w") as archive:
        for key, array in file_arrays(network, scaler).items():
            with archive.open(f"{key}.npy", "w") as handle:
                npy_format.write_array(handle, np.asarray(array), version=(2, 0))
    restored = Network(n_inputs=6, n_hidden=5, n_outputs=1)
    assert restored.load(path).matches(scaler)
    assert restored.fingerprint() == network.fingerprint()


def test_load_converts_float32_parameters_to_float64(tmp_path):
    network = small_network()
    arrays = file_arrays(network, scaler_for(network))
    arrays["dense1_weights"] = arrays["dense1_weights"].astype(np.float32)
    path = tmp_path / "weights.npz"
    np.savez(path, **arrays)
    network.load(path)
    assert network.dense1.weights.dtype == np.float64
    assert np.array_equal(network.dense1.weights, arrays["dense1_weights"].astype(np.float64))
