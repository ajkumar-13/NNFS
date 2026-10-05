"""Post 10: the forward and the central difference, and the sweep over the step size h.

Run from the series root:
    python posts/10-derivatives-partial-derivatives-and-gradients/snippets/finite_differences.py

Contents: the derivative as a rate and not a change (section 2.4); the forward and the central
difference of f(x) = x^3 at x = 2 with h = 0.001, next to the algebra of section 2.5; the sweep
over h from 1e-1 down to 1e-16 in float64 (section 6.1); and the rules of sections 2.1 and 2.2
checked against a central difference (section 6.2).

Needs only the standard library. Nothing here is random, and the sweep uses only float64
additions, subtractions, multiplications and divisions, each of which is correctly rounded, so
its output is the same on every run and on every machine.
"""
import math
import sys


def forward_difference(f, x, h):
    """Slope of the straight line through (x, f(x)) and (x + h, f(x + h))."""
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
    """Slope of the straight line through (x - h, f(x - h)) and (x + h, f(x + h))."""
    return (f(x + h) - f(x - h)) / (2 * h)


def cube(x):
    return x * x * x


STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
         1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16)


def main():
    x, exact = 2.0, 12.0     # f(x) = x^3, so f'(x) = 3x^2 and f'(2) = 12

    print("== Section 2.4: a derivative is a rate, not a change. f(x) = x^3 at x = 2, slope 12")
    for step in (0.1, 0.01, 0.001):
        change = cube(x + step) - cube(x)
        print(f"step {step:<5}  slope * step = {exact * step:.6f}   actual change in f = {change:.6f}")

    print()
    print("== Section 2.5: forward against central at h = 0.001")
    h = 0.001
    fwd = forward_difference(cube, x, h)
    cen = central_difference(cube, x, h)
    print(f"forward difference  {fwd:.6f}   error {abs(fwd - exact):.3e}   algebra: 3xh + h^2 = {3 * x * h + h * h:.3e}")
    print(f"central difference  {cen:.6f}   error {abs(cen - exact):.3e}   algebra: h^2       = {h * h:.3e}")

    print()
    print("== Section 6.1: the sweep over h in float64 (error = distance from the exact 12)")
    print(f"{'h':>5}   {'forward':>16}   {'error':>7}   {'central':>16}   {'error':>7}")
    rows = []
    for h in STEPS:
        fwd = forward_difference(cube, x, h)
        cen = central_difference(cube, x, h)
        rows.append((h, abs(fwd - exact), abs(cen - exact)))
        print(f"{h:.0e}   {fwd:16.12f}   {abs(fwd - exact):.1e}   {cen:16.12f}   {abs(cen - exact):.1e}")
    best_fwd = min(rows, key=lambda r: r[1])
    best_cen = min(rows, key=lambda r: r[2])
    print(f"smallest forward error on this grid: {best_fwd[1]:.1e} at h = {best_fwd[0]:.0e}")
    print(f"smallest central error on this grid: {best_cen[2]:.1e} at h = {best_cen[0]:.0e}")
    print(f"the smallest forward error is {best_fwd[1] / best_cen[2]:.0f} times the smallest central error")
    print(f"float64 machine epsilon: {sys.float_info.epsilon:.3e}")
    print(f"gap between neighbouring float64 numbers near f(2) = 8: {math.ulp(8.0):.3e}")
    print(f"2.0 + 1e-16 == 2.0 is {2.0 + 1e-16 == 2.0}")
    print(f"a rounding error of 1e-15 / h equals the central truncation error h^2 at h = {1e-15 ** (1 / 3):.1e}")
    print(f"a rounding error of 1e-15 / h equals the forward truncation error 6h at h = {math.sqrt(1e-15 / 6):.1e}")

    print()
    print("== Section 6.2: the rules of sections 2.1 and 2.2 against a central difference, h = 1e-5")
    checks = [
        ("7 (a constant)",    lambda t: 7.0,                           lambda t: 0.0),
        ("x",                 lambda t: t,                             lambda t: 1.0),
        ("x^2",               lambda t: t * t,                         lambda t: 2 * t),
        ("2x^2",              lambda t: 2 * t * t,                     lambda t: 4 * t),
        ("x^3",               lambda t: t * t * t,                     lambda t: 3 * t * t),
        ("5x^4",              lambda t: 5 * t * t * t * t,             lambda t: 20 * t * t * t),
        ("x^3 + 2x^2 + 6",    lambda t: t * t * t + 2 * t * t + 6,     lambda t: 3 * t * t + 4 * t),
        ("5x^3 + 2x",         lambda t: 5 * t * t * t + 2 * t,         lambda t: 15 * t * t + 2),
        ("(2x + 1)^2",        lambda t: (2 * t + 1) * (2 * t + 1),     lambda t: 8 * t + 4),
    ]
    at = 1.5
    print(f"every row at x = {at}")
    print(f"{'f(x)':<16} {'rule':>6}   {'central':>14}   {'gap':>7}")
    worst = 0.0
    for label, f, rule in checks:
        numeric = central_difference(f, at, 1e-5)
        worst = max(worst, abs(numeric - rule(at)))
        print(f"{label:<16} {rule(at):6.2f}   {numeric:14.9f}   {abs(numeric - rule(at)):.1e}")
    print(f"largest gap between a rule and its central difference: {worst:.1e}")
    wrong = 2 * (2 * at + 1)
    print(f"the tempting answer 2(2x + 1) for (2x + 1)^2 gives {wrong:.2f} at x = {at}, half of the measured {central_difference(checks[-1][1], at, 1e-5):.2f}")

    print()
    print("== Hero figure: f(x) = x^2, slope 2x at the three marked points")
    for point in (-1.5, 0.5, 2.0):
        print(f"x = {point:4.1f}   2x = {2 * point:4.1f}   central difference = {central_difference(lambda t: t * t, point, 1e-5):9.6f}")


if __name__ == "__main__":
    main()
