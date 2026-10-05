"""The two commands: exit codes, messages, and one run of each as a real process."""

import subprocess
import sys

import pytest

from binary_classifier import evaluate as evaluate_module
from binary_classifier import train as train_module

SMALL = ["--epochs", "5", "--n-samples", "30", "--log-every", "5"]


def test_train_command_writes_a_weights_file(tmp_path, capsys):
    target = tmp_path / "w.npz"
    assert train_module.main([*SMALL, "--weights", str(target)]) == 0
    assert target.is_file()
    assert f"wrote weights to {target}" in capsys.readouterr().out


def test_train_command_passes_every_option_through(tmp_path, capsys):
    target = tmp_path / "w.npz"
    train_module.main(
        [
            "--epochs",
            "3",
            "--n-samples",
            "50",
            "--noise",
            "0.2",
            "--seed",
            "9",
            "--init",
            "small",
            "--log-every",
            "1",
            "--weights",
            str(target),
        ]
    )
    out = capsys.readouterr().out
    assert "two moons: 50 points, noise 0.2, seed 9 | train (40, 2), test (10, 2)" in out
    assert "init small" in out
    assert out.count("epoch ") == 3


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--epochs", "0"], "epochs must be a whole number of at least 1, got 0"),
        (["--epochs", "many"], "invalid int value: 'many'"),
        (["--noise", "-0.5"], "noise must be a finite number that is not negative, got -0.5"),
        (["--noise", "nan"], "noise must be a finite number"),
        (["--n-samples", "2"], "both sets need at least one row"),
        (["--n-samples", "1"], "n_samples must be at least 2"),
        (["--seed", "-1"], "seed must be between 0 and 4294967295, got -1"),
        (["--log-every", "0"], "log_every must be a whole number of at least 1"),
        (["--init", "lecun"], "invalid choice: 'lecun'"),
        (["--weights", "old.pkl"], "is a pickle file"),
        (["--weights", "weights"], "the weights name must end in .npz, got weights"),
    ],
)
def test_train_command_rejects_bad_arguments(tmp_path, monkeypatch, capsys, arguments, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        train_module.main(["--epochs", "1", "--n-samples", "20", *arguments])
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "usage: python -m binary_classifier.train" in captured.err
    assert list(tmp_path.iterdir()) == []


def test_evaluate_command_scores_a_weights_file(tiny_run, tmp_path, capsys):
    grid = tmp_path / "g.npz"
    code = evaluate_module.main(["--weights", str(tiny_run.weights_path), "--grid", str(grid)])
    assert code == 0
    assert grid.is_file()
    assert "[test] loss " in capsys.readouterr().out


def test_evaluate_command_without_a_weights_file_says_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert evaluate_module.main([]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: no weights file at moons_weights.npz")
    assert "Train one with: uv run python -m binary_classifier.train" in captured.err
    assert captured.out == ""


def test_evaluate_command_refuses_a_pickle_file(tmp_path, capsys):
    old = tmp_path / "moons_weights.pkl"
    old.write_bytes(b"\x80\x04N.")
    assert evaluate_module.main(["--weights", str(old)]) == 1
    assert "Pickle files are no longer read or written" in capsys.readouterr().err


def test_evaluate_command_checks_the_minimum_accuracy(tiny_run, tmp_path, capsys):
    arguments = ["--weights", str(tiny_run.weights_path), "--grid", str(tmp_path / "g.npz")]
    # Any classifier meets a floor of 0; the full report is printed either way.
    assert evaluate_module.main([*arguments, "--min-accuracy", "0"]) == 0
    assert capsys.readouterr().err == ""

    # Eight test points cannot score above 1, so a floor of 1 passes only on a perfect score.
    accuracy = evaluate_module.evaluate(tiny_run.weights_path, grid=tmp_path / "g.npz")[
        "test"
    ].accuracy
    capsys.readouterr()
    code = evaluate_module.main([*arguments, "--min-accuracy", "1.0"])
    captured = capsys.readouterr()
    assert "[test] loss " in captured.out
    if accuracy < 1.0:
        assert code == 1
        assert f"error: test accuracy {accuracy:.4f} is below the required 1.0000" in captured.err
    else:
        assert code == 0


def test_evaluate_command_fails_below_the_minimum_accuracy(tmp_path, capsys):
    # One epoch from the small initialisation answers the same class for every point.
    target = tmp_path / "untrained.npz"
    train_module.main(
        ["--epochs", "1", "--n-samples", "40", "--init", "small", "--weights", str(target)]
    )
    capsys.readouterr()
    assert evaluate_module.main(["--weights", str(target), "--min-accuracy", "0.99"]) == 1
    assert "is below the required 0.9900" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1.5", "must be a fraction between 0 and 1, got 1.5"),
        ("-0.1", "must be a fraction between 0 and 1"),
        ("high", "'high' is not a number"),
    ],
)
def test_evaluate_command_rejects_a_bad_minimum_accuracy(capsys, value, message):
    with pytest.raises(SystemExit) as exit_info:
        evaluate_module.main(["--min-accuracy", value])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_both_commands_run_as_modules(tmp_path):
    target = tmp_path / "w.npz"
    trained = subprocess.run(
        [sys.executable, "-m", "binary_classifier.train", *SMALL, "--weights", str(target)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert trained.returncode == 0, trained.stderr
    assert "trained 5 epochs in " in trained.stdout

    evaluated = subprocess.run(
        [sys.executable, "-m", "binary_classifier.evaluate", "--weights", str(target)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert "parameters: 337" in evaluated.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["decision_grid.npz", "w.npz"]
