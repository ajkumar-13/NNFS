"""
Load a trained California-housing regressor and report behaviour:

  * Train and test MSE (in standardised units)
  * Train and test RMSE in dollars (de-standardised)
  * Mean absolute error (MAE) in dollars
  * R^2 (coefficient of determination)
  * Predicted-vs-actual arrays dumped to .npz so a notebook can plot
    a scatter / residual chart without retraining.

Usage:
    python evaluate.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from data import destandardise_y
from nn import (
    Activation_ReLU,
    Layer_Dense,
    Loss_MSE,
)


def restore_model(checkpoint_path):
    payload = pickle.loads(Path(checkpoint_path).read_bytes())
    dense1 = Layer_Dense(8, 64)
    dense2 = Layer_Dense(64, 64)
    dense3 = Layer_Dense(64, 1)
    dense1.weights, dense1.biases = payload["dense1"]
    dense2.weights, dense2.biases = payload["dense2"]
    dense3.weights, dense3.biases = payload["dense3"]
    activation1 = Activation_ReLU()
    activation2 = Activation_ReLU()
    loss_fn     = Loss_MSE()
    return (dense1, activation1, dense2, activation2, dense3, loss_fn), payload


def forward(X, layers):
    dense1, activation1, dense2, activation2, dense3, _ = layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    return dense3.output


def metrics(name, X, y_std, layers, stats):
    loss_fn = layers[-1]
    y_pred_std = forward(X, layers)
    mse_std = loss_fn.forward(y_pred_std, y_std)

    # De-standardise back to dollars (× $100k).
    y_pred = destandardise_y(y_pred_std.ravel(), stats) * 100_000
    y_true = destandardise_y(np.asarray(y_std).ravel(), stats) * 100_000

    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot

    print(f"\n[{name}]  n={len(y_true)}")
    print(f"  MSE (std units): {mse_std:.4f}")
    print(f"  RMSE (dollars):  ${rmse:>10,.0f}")
    print(f"  MAE  (dollars):  ${mae:>10,.0f}")
    print(f"  R^2:             {r2:.4f}")
    return y_pred, y_true


def evaluate(checkpoint="cal_housing_weights.pkl"):
    print(f"restoring weights from {checkpoint}")
    layers, payload = restore_model(checkpoint)
    stats = payload["stats"]

    train_pred, train_true = metrics("train", payload["X_train"], payload["y_train"], layers, stats)
    test_pred,  test_true  = metrics("test",  payload["X_test"],  payload["y_test"],  layers, stats)

    # Persist predicted/actual arrays for downstream plotting.
    out_path = Path(checkpoint).with_name("predictions.npz")
    np.savez(out_path,
             train_pred=train_pred, train_true=train_true,
             test_pred=test_pred,   test_true=test_true)
    print(f"\nwrote predicted-vs-actual arrays to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="cal_housing_weights.pkl")
    args = p.parse_args()
    evaluate(checkpoint=args.checkpoint)
