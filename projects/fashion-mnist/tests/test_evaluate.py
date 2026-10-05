"""The evaluation: metrics on hand-made cases, the printed report, and the command's exit codes."""

import pickle

import numpy as np
import pytest

from fashion_mnist import data
from fashion_mnist import evaluate as evaluate_module
from fashion_mnist.data import DataError
from fashion_mnist.evaluate import (
    Report,
    accuracy,
    confusion_matrix,
    evaluate,
    format_report,
    main,
    most_confused_pairs,
    per_class_accuracy,
    score,
)
from fashion_mnist.model import Network
from helpers import write_dataset

Y_TRUE = np.array([0, 0, 1, 1, 2, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0, 2])


def identity_network():
    """A 2-2-2-2 network whose logits equal its (non-negative) inputs."""
    network = Network(n_inputs=2, n_hidden=2, n_classes=2, l2=0.0, dropout_rate=0.5)
    for layer in network.dense_layers:
        layer.weights = np.eye(2)
        layer.biases = np.zeros((1, 2))
    return network


def saved_untrained_network(tmp_path):
    np.random.seed(0)
    path = tmp_path / "weights.npz"
    Network().save(path)
    return path


# ----------------------------------------------------------------------------------- the metrics


def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    confusion = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert confusion.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 2]]
    assert confusion.dtype == np.int64


def test_accuracy_is_the_diagonal_share():
    assert accuracy(confusion_matrix(Y_TRUE, Y_PRED, 3)) == 5 / 7


def test_per_class_accuracy_reports_each_class_with_its_sample_count():
    rows = per_class_accuracy(confusion_matrix(Y_TRUE, Y_PRED, 3))
    assert rows == [(0, 0.5, 2), (1, 1.0, 2), (2, 2 / 3, 3)]


def test_per_class_accuracy_skips_a_class_without_samples():
    rows = per_class_accuracy(confusion_matrix(Y_TRUE, Y_PRED, 4))
    assert [label for label, _, _ in rows] == [0, 1, 2]


def test_metrics_reject_input_they_cannot_score():
    with pytest.raises(ValueError, match="two label vectors of one length"):
        confusion_matrix(np.array([0, 1]), np.array([0]), 2)
    with pytest.raises(ValueError, match="true labels must lie between 0 and 1"):
        confusion_matrix(np.array([0, 2]), np.array([0, 1]), 2)
    with pytest.raises(ValueError, match="predicted labels must lie between 0 and 1"):
        confusion_matrix(np.array([0, 1]), np.array([0, -1]), 2)
    with pytest.raises(ValueError, match="undefined for an empty set"):
        accuracy(np.zeros((3, 3), dtype=np.int64))


def test_most_confused_pairs_are_the_largest_entries_off_the_diagonal():
    confusion = np.array([[50, 7, 0], [3, 40, 9], [9, 1, 60]])
    # The two nines tie: the pair with the smaller true class comes first.
    assert most_confused_pairs(confusion, k=4) == [(1, 2, 9), (2, 0, 9), (0, 1, 7), (1, 0, 3)]
    assert most_confused_pairs(confusion, k=0) == []


def test_most_confused_pairs_ties_fall_back_to_the_predicted_class():
    confusion = np.array([[5, 2, 2], [0, 5, 0], [0, 0, 5]])
    assert most_confused_pairs(confusion) == [(0, 1, 2), (0, 2, 2)]


def test_most_confused_pairs_leave_out_pairs_that_never_occur():
    assert most_confused_pairs(confusion_matrix(Y_TRUE, Y_PRED, 3)) == [(0, 1, 1), (2, 0, 1)]
    assert most_confused_pairs(np.diag([4, 5, 6])) == []
    with pytest.raises(ValueError, match="must not be negative"):
        most_confused_pairs(np.diag([4, 5, 6]), k=-1)


# -------------------------------------------------------------------------------------- scoring


def test_score_on_a_network_with_hand_set_weights():
    network = identity_network()
    X = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0], [0.0, 0.5]])
    y = np.array([0, 1, 1, 1])
    report = score(network, X, y)

    # Logits equal the inputs, so the predictions are 0, 1, 0, 1 and only sample 2 is wrong.
    expected_loss = np.mean(
        [
            np.log1p(np.exp(-2.0)),
            np.log1p(np.exp(-3.0)),
            np.log1p(np.exp(1.0)),
            np.log1p(np.exp(-0.5)),
        ]
    )
    assert report.loss == pytest.approx(expected_loss)
    assert report.accuracy == 0.75
    assert report.n_samples == 4
    assert report.n_wrong == 1
    assert report.wrong_indices.tolist() == [2]
    assert report.confusion.tolist() == [[1, 0], [1, 2]]
    assert report.parameters == 18
    assert report.fingerprint == network.fingerprint()


def test_score_runs_with_dropout_off():
    network = identity_network()
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)
    assert score(network, X, y).loss == score(network, X, y).loss
    assert np.array_equal(network.dropout1.output, network.activation1.output)


def test_format_report_prints_every_figure():
    report = Report(
        parameters=118_282,
        fingerprint="ab" * 32,
        loss=0.06284,
        accuracy=0.75,
        n_samples=4,
        n_wrong=1,
        confusion=np.array([[1, 0], [1, 2]]),
        wrong_indices=np.array([2]),
    )
    assert format_report(report, class_names=("Shirt", "Coat")).splitlines() == [
        "  parameters 118,282",
        "  weights    sha256 " + "ab" * 32,
        "",
        "  loss     0.0628",
        "  accuracy 0.7500  (3/4)",
        "",
        "Per-class accuracy:",
        "  Shirt (0): 1.0000  (1 samples)",
        "   Coat (1): 0.6667  (3 samples)",
        "",
        "Confusion matrix (rows = true class, cols = predicted class):",
        "                 0      1",
        "  Shirt (0):     1      0",
        "   Coat (1):     1      2",
        "",
        "Top 1 most-confused class pairs (true -> predicted):",
        "   Coat (1) -> Shirt (0):     1 times",
        "",
        "1 misclassified samples. First 20 indices: [2]",
    ]


def test_format_report_names_the_ten_classes_by_default_in_plain_ascii():
    confusion = np.diag([1000] * 10)
    confusion[0, 0], confusion[0, 6] = 815, 185
    confusion[6, 6], confusion[6, 0] = 920, 80
    report = Report(118_282, "ab" * 32, 0.3419, 0.9735, 10_000, 265, confusion, np.arange(265))
    text = format_report(report)
    lines = text.splitlines()
    assert text.isascii()
    assert "  T-shirt/top (0): 0.8150  (1000 samples)" in lines
    assert "   Ankle boot (9): 1.0000  (1000 samples)" in lines
    assert lines[19].split() == [str(label) for label in range(10)]
    assert lines[20].startswith("  T-shirt/top (0):   815      0")
    assert "Top 2 most-confused class pairs (true -> predicted):" in lines
    assert "  T-shirt/top (0) ->       Shirt (6):   185 times" in lines
    assert "        Shirt (6) -> T-shirt/top (0):    80 times" in lines


def test_format_report_needs_one_name_for_every_class():
    report = Report(18, "ab" * 32, 0.7, 0.5, 4, 2, np.array([[1, 1], [1, 1]]), np.array([1, 2]))
    with pytest.raises(ValueError, match="one name for each of the 2 classes, got 10"):
        format_report(report)


def test_format_report_lists_at_most_twenty_errors():
    report = Report(
        parameters=18,
        fingerprint="cd" * 32,
        loss=1.0,
        accuracy=0.0,
        n_samples=30,
        n_wrong=30,
        confusion=np.array([[0, 30], [0, 0]]),
        wrong_indices=np.arange(30),
    )
    last_line = format_report(report, class_names=("Shirt", "Coat")).splitlines()[-1]
    assert last_line == f"30 misclassified samples. First 20 indices: {list(range(20))}"


# ------------------------------------------------------------------------------------- evaluate


def test_evaluate_scores_the_fixture_test_set(as_fashion_mnist, tmp_path):
    path = saved_untrained_network(tmp_path)
    lines = []
    report = evaluate(as_fashion_mnist, path, log=lines.append)

    assert report.parameters == 118_282
    assert report.n_samples == 6
    restored = Network()
    restored.load(path)
    assert report.fingerprint == restored.fingerprint()
    assert int(report.confusion.sum()) == 6
    assert report.accuracy == (6 - report.n_wrong) / 6
    assert report.loss == pytest.approx(np.log(10), abs=0.02)
    assert lines[0] == f"restoring weights from {path}"
    assert lines[1] == f"loading Fashion-MNIST from {as_fashion_mnist}"
    assert lines[2] == format_report(report)


def test_evaluate_rejects_images_of_another_size(tmp_path, monkeypatch):
    files = write_dataset(tmp_path, np.zeros((3, 4, 4)), [0, 1, 2], np.zeros((2, 4, 4)), [0, 1])
    monkeypatch.setattr(data, "FASHION_MNIST_FILES", files)
    path = saved_untrained_network(tmp_path)
    with pytest.raises(DataError, match="takes 784 pixels per image, but the test images have 16"):
        evaluate(tmp_path, path, log=lambda _: None)


# ------------------------------------------------------------------------------ the command line


def test_main_prints_the_report_and_returns_zero(as_fashion_mnist, tmp_path, capsys):
    path = saved_untrained_network(tmp_path)
    assert main(["--data-dir", str(as_fashion_mnist), "--weights", str(path)]) == 0
    output = capsys.readouterr().out
    assert "  parameters 118,282" in output
    assert "Per-class accuracy:" in output
    assert "Confusion matrix (rows = true class, cols = predicted class):" in output
    assert "most-confused class pairs (true -> predicted):" in output
    assert "   Ankle boot (9): " in output
    assert "misclassified samples. First 20 indices:" in output


def half_right(**_):
    return Report(18, "ef" * 32, 0.7, 0.5, 4, 2, np.array([[1, 1], [1, 1]]), np.array([1, 2]))


@pytest.mark.parametrize(("threshold", "status"), [("0.5", 0), ("0.4999", 0), ("0.5001", 1)])
def test_min_accuracy_turns_the_report_into_a_check(threshold, status, monkeypatch, capsys):
    monkeypatch.setattr(evaluate_module, "evaluate", half_right)
    assert main(["--min-accuracy", threshold]) == status
    error = capsys.readouterr().err
    if status:
        assert error == "error: test accuracy 0.5000 is below the required 0.5001\n"
    else:
        assert error == ""


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--min-accuracy", "1.5"], "must be a fraction between 0 and 1"),
        (["--min-accuracy", "-0.1"], "must be a fraction between 0 and 1"),
        (["--min-accuracy", "high"], "'high' is not a number"),
        (["--checkpoint", "old.pkl"], "unrecognized arguments"),
    ],
)
def test_main_rejects_bad_arguments(arguments, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_main_reports_missing_weights_with_the_command_that_creates_them(tmp_path, capsys):
    status = main(["--weights", str(tmp_path / "absent.npz"), "--data-dir", str(tmp_path)])
    assert status == 1
    error = capsys.readouterr().err
    assert error.startswith("error: the weights file")
    assert "uv run python -m fashion_mnist.train" in error


def test_main_refuses_an_old_pickle_checkpoint(tmp_path, capsys):
    path = tmp_path / "fashion_mnist_weights.pkl"
    path.write_bytes(pickle.dumps({"dense1": "anything"}))
    assert main(["--weights", str(path), "--data-dir", str(tmp_path)]) == 1
    assert (
        "Pickle checkpoints written before version 1.0.0 are not loaded" in capsys.readouterr().err
    )


def test_main_reports_exhausted_memory_with_status_one(monkeypatch, capsys):
    def exhausted(**_):
        raise MemoryError

    monkeypatch.setattr(evaluate_module, "evaluate", exhausted)
    assert main([]) == 1
    assert "not enough free memory" in capsys.readouterr().err


def test_main_reports_missing_data_with_the_download_command(tmp_path, capsys):
    path = saved_untrained_network(tmp_path)
    assert main(["--weights", str(path), "--data-dir", str(tmp_path / "no-data")]) == 1
    assert "uv run python scripts/download_fashion_mnist.py" in capsys.readouterr().err
