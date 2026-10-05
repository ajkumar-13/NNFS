"""Post 10: partial derivatives and the gradient, each checked against a central difference.

Run from the series root:
    python posts/10-derivatives-partial-derivatives-and-gradients/snippets/partials_and_gradient.py

Contents: the partial derivatives of examples 1 and 2 of section 3.1 at one point, by the
freezing rule and by a central difference along each axis (section 6.3); the same partial at a
different frozen value (section 3.2); the gradient, its length, and the rate of change of f
along the gradient, against it, across it, and along 10,000 random directions (section 4.1);
and the change in f when one coordinate at a time is nudged (section 4.2).

Needs only NumPy. The random directions are drawn after np.random.seed(0).
"""
import numpy as np


def f(p):
    """Example 2 of section 3.1: f(x, y, z) = 3x^3 z - y^2 + 5z + 2yz."""
    x, y, z = p
    return 3 * x * x * x * z - y * y + 5 * z + 2 * y * z


def grad_f(p):
    """The three partial derivatives of f, each found by freezing the other two variables."""
    x, y, z = p
    return np.array([9 * x * x * z, -2 * y + 2 * z, 3 * x * x * x + 5 + 2 * y])


def numerical_gradient(f, point, h=1e-5):
    """Central difference along each axis in turn, the other coordinates held fixed."""
    point = np.asarray(point, dtype=np.float64)
    grad = np.zeros_like(point)
    for k in range(point.size):
        step = np.zeros_like(point)
        step[k] = h                    # move along axis k only
        grad[k] = (f(point + step) - f(point - step)) / (2 * h)
    return grad


def rate_along(f, point, direction, h=1e-5):
    """Rate of change of f per unit distance travelled from point along a unit vector."""
    return (f(point + h * direction) - f(point - h * direction)) / (2 * h)


def main():
    print("== Section 3.1, example 1: f(x, y) = 2x + 3y^2 at (x, y) = (1, 2)")
    g1 = numerical_gradient(lambda p: 2 * p[0] + 3 * p[1] * p[1], [1.0, 2.0])
    print(f"by the freezing rule:    df/dx = 2, df/dy = 6y = {6 * 2.0:.0f}")
    print(f"by central differences:  df/dx = {g1[0]:.6f}, df/dy = {g1[1]:.6f}")

    print()
    print("== Section 3.1, example 2: f(x, y, z) = 3x^3 z - y^2 + 5z + 2yz at (x, y, z) = (1, 2, 3)")
    p = np.array([1.0, 2.0, 3.0])
    analytic = grad_f(p)
    numeric = numerical_gradient(f, p)
    print(f"f(1, 2, 3) = {f(p):.0f}")
    for name, formula, a, n in zip(("df/dx", "df/dy", "df/dz"),
                                   ("9x^2 z", "-2y + 2z", "3x^3 + 5 + 2y"),
                                   analytic, numeric):
        print(f"{name} = {formula:<14} freezing rule {a:5.1f}   central difference {n:12.8f}")
    print(f"largest gap between the two columns: {np.max(np.abs(analytic - numeric)):.1e}")

    print()
    print("== Section 3.2: the frozen variables still set the value. df/dx = 9x^2 z at x = 1")
    for z in (3.0, 0.0, -3.0):
        q = np.array([1.0, 2.0, z])
        print(f"z = {z:4.1f}   9x^2 z = {grad_f(q)[0]:5.1f}   central difference {numerical_gradient(f, q)[0]:12.8f}")

    print()
    print("== Section 4.1: the gradient at (1, 2, 3), and the rate of change of f in several directions")
    length = np.sqrt(np.sum(analytic * analytic))
    uphill = analytic / length
    print(f"gradient = {analytic}, length = sqrt(27^2 + 2^2 + 12^2) = sqrt({np.sum(analytic * analytic):.0f}) = {length:.6f}")
    across = np.array([0.0, 12.0, -2.0]) / np.sqrt(148.0)   # at right angles to the gradient
    print(f"along the gradient    {rate_along(f, p, uphill):11.6f}")
    print(f"against the gradient  {rate_along(f, p, -uphill):11.6f}")
    print(f"across the gradient   {abs(rate_along(f, p, across)):11.6f}")
    print(f"along the x axis      {rate_along(f, p, np.array([1.0, 0.0, 0.0])):11.6f}")
    np.random.seed(0)
    directions = np.random.randn(10000, 3)
    directions = directions / np.sqrt(np.sum(directions * directions, axis=1, keepdims=True))
    rates = np.array([rate_along(f, p, d) for d in directions])
    print(f"10,000 random unit directions: steepest {rates.max():.6f}, most downhill {rates.min():.6f}")
    print(f"random directions steeper than the gradient direction: {int(np.sum(rates > length))}")
    print(f"random directions that go uphill: {int(np.sum(rates > 0))} of {len(rates)}")

    print()
    print("== Section 4.2: nudge one coordinate by +0.01, the others held fixed")
    for k, name in enumerate(("x", "y", "z")):
        step = np.zeros(3)
        step[k] = 0.01
        print(f"{name}: component {analytic[k]:5.1f}   predicted change {analytic[k] * 0.01:.4f}   actual change in f {f(p + step) - f(p):.4f}")

    print()
    print("== Section 4.1: one step of size 0.001 along and against the gradient")
    alpha = 0.001
    print(f"predicted change: alpha * length^2 = {alpha * length * length:.4f}")
    print(f"step against the gradient: f goes from {f(p):.4f} to {f(p - alpha * analytic):.4f}")
    print(f"step along the gradient:   f goes from {f(p):.4f} to {f(p + alpha * analytic):.4f}")


if __name__ == "__main__":
    main()
