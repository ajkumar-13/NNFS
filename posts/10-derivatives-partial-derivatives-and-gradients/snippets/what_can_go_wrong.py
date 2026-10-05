"""Post 10, section 7: the places where a derivative or its numerical estimate misleads.

Run from the series root:
    python posts/10-derivatives-partial-derivatives-and-gradients/snippets/what_can_go_wrong.py

Contents: the sweep over h repeated in float32, the precision nnfs.init() switches to; ReLU at
its corner, where the forward, backward and central differences disagree; the gradient of the
same loss under two parameterisations; and the number of function evaluations a numerical
gradient costs.

Needs only NumPy. Nothing here is random.
"""
import numpy as np


def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def cube(x):
    return x * x * x


def relu(z):
    return max(0.0, z)


def main():
    print("== The sweep over h in float32: f(x) = x^3 at x = 2, exact derivative 12")
    print(f"{'h':>5}   {'forward':>12}   {'error':>7}   {'central':>12}   {'error':>7}")
    x32, two = np.float32(2.0), np.float32(2.0)
    rows = []
    for step in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        h = np.float32(step)
        fwd = (cube(x32 + h) - cube(x32)) / h                  # every operand is a float32
        cen = (cube(x32 + h) - cube(x32 - h)) / (two * h)
        assert fwd.dtype == np.float32 and cen.dtype == np.float32
        rows.append((step, abs(float(fwd) - 12.0), abs(float(cen) - 12.0)))
        print(f"{step:.0e}   {float(fwd):12.7f}   {abs(float(fwd) - 12.0):.1e}   {float(cen):12.7f}   {abs(float(cen) - 12.0):.1e}")
    best_fwd = min(rows, key=lambda r: r[1])
    best_cen = min(rows, key=lambda r: r[2])
    print(f"smallest forward error on this grid: {best_fwd[1]:.1e} at h = {best_fwd[0]:.0e}")
    print(f"smallest central error on this grid: {best_cen[2]:.1e} at h = {best_cen[0]:.0e}")
    print(f"float32 machine epsilon: {float(np.finfo(np.float32).eps):.3e}")
    print(f"float32: 2.0 + 1e-7 == 2.0 is {bool(x32 + np.float32(1e-7) == x32)}")

    print()
    print("== ReLU at its corner z = 0, h = 1e-5")
    h = 1e-5
    print(f"forward difference   {forward_difference(relu, 0.0, h):.1f}   (the slope on the right)")
    print(f"backward difference  {backward_difference(relu, 0.0, h):.1f}   (the slope on the left)")
    print(f"central difference   {central_difference(relu, 0.0, h):.1f}   (neither)")
    print(f"away from the corner: central difference at z = 0.3 is {central_difference(relu, 0.3, h):.1f}, at z = -0.3 is {central_difference(relu, -0.3, h):.1f}")

    print()
    print("== One loss, two parameterisations: L = (w - 1)^2, and the same loss written with w = 2u")
    loss_w = lambda w: (w - 1.0) * (w - 1.0)
    loss_u = lambda u: (2.0 * u - 1.0) * (2.0 * u - 1.0)
    print(f"dL/dw at w = 2: {central_difference(loss_w, 2.0, h):.6f}")
    print(f"dL/du at u = 1: {central_difference(loss_u, 1.0, h):.6f}   (the same point, since w = 2u = 2)")

    print()
    print("== The cost of a numerical gradient: function evaluations for n parameters")
    calls = 0

    def bowl(theta):
        nonlocal calls
        calls += 1
        return float(np.sum(theta * theta))

    theta = np.arange(21, dtype=np.float64) / 10.0      # 21 parameters, like the 2-3-3 network
    grad = np.zeros_like(theta)
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = h
        grad[k] = (bowl(theta + step) - bowl(theta - step)) / (2 * h)
    print(f"central-difference gradient of a 21-parameter function: {calls} evaluations of the function")
    print(f"largest gap to the exact gradient 2 * theta: {np.max(np.abs(grad - 2 * theta)):.1e}")


if __name__ == "__main__":
    main()
