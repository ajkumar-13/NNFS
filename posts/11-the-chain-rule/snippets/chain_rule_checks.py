"""Post 11: the chain rule on scalar functions, each result checked against a central difference.

Run from the series root:
    python posts/11-the-chain-rule/snippets/chain_rule_checks.py

Contents: the first example of section 2, (2x + 1)^2, and the reparameterisation w = 2u left
open by post 10; the distance, time and fuel example of section 3; the polynomial 3(2x^2)^5 of
section 4, with the outer derivative taken at the right and at the wrong point; a chain of
three functions (section 5); a variable that reaches the output along two paths (section 5.1);
and ReLU(2x - 1) on both sides of its corner and on it (section 9).

Needs only the standard library. Nothing here is random, so the output is the same on every run.
"""


def central_difference(f, x, h=1e-5):
    """Slope of the straight line through (x - h, f(x - h)) and (x + h, f(x + h)). Post 10."""
    return (f(x + h) - f(x - h)) / (2 * h)


def chain_derivative(functions, derivatives, x):
    """Derivative at x of the composition that applies functions[0] first.

    Returns the derivative and the list of local derivatives, innermost first.
    """
    value = x
    factors = []
    for function, derivative in zip(functions, derivatives):
        factors.append(derivative(value))    # local slope, at the value this function receives
        value = function(value)              # then move one function along the chain
    product = 1.0
    for factor in reversed(factors):         # start at the output and multiply back to the input
        product *= factor
    return product, factors


def compose(functions):
    """The composition as one function, for the finite-difference check."""
    def composed(x):
        for function in functions:
            x = function(x)
        return x
    return composed


def report(label, functions, derivatives, x):
    slope, factors = chain_derivative(functions, derivatives, x)
    measured = central_difference(compose(functions), x)
    shown = " * ".join(f"{factor:g}" for factor in reversed(factors))
    print(f"{label}")
    print(f"  local derivatives, outermost first: {shown}   ({len(factors)} factors)")
    print(f"  chain rule {slope:.6f}   central difference {measured:.6f}   relative gap {abs(measured - slope) / abs(slope):.1e}")
    return slope


def relu(z):
    return max(0.0, z)


def relu_slope(z):
    return 1.0 if z > 0 else 0.0


def main():
    print("== Section 2: y = (2x + 1)^2 at x = 1.5, with z = 2x + 1")
    inner, outer = (lambda x: 2 * x + 1), (lambda z: z * z)
    d_inner, d_outer = (lambda x: 2.0), (lambda z: 2 * z)
    report("y = f(g(x)), g(x) = 2x + 1, f(z) = z^2", [inner, outer], [d_inner, d_outer], 1.5)
    print(f"  the outer slope alone, 2(2x + 1), is {d_outer(inner(1.5)):g}: half of the measured slope")

    print()
    print("== Section 2: the reparameterisation of post 10, L = (w - 1)^2 with w = 2u, at u = 1")
    report("L = f(g(u)), g(u) = 2u, f(w) = (w - 1)^2",
           [lambda u: 2 * u, lambda w: (w - 1) ** 2], [lambda u: 2.0, lambda w: 2 * (w - 1)], 1.0)

    print()
    print("== Section 3: time -> distance -> fuel, at x = 3 hours")
    report("z = 60x km, y = z / 30 litres", [lambda x: 60 * x, lambda z: z / 30], [lambda x: 60.0, lambda z: 1 / 30], 3.0)
    print(f"  after 3 hours: {60 * 3.0:g} km driven, {60 * 3.0 / 30:g} litres burnt")

    print()
    print("== Section 4: y = 3(2x^2)^5 at x = 1")
    g, f = (lambda x: 2 * x * x), (lambda z: 3 * z ** 5)
    dg, df = (lambda x: 4 * x), (lambda z: 15 * z ** 4)
    report("g(x) = 2x^2, f(z) = 3z^5", [g, f], [dg, df], 1.0)
    print(f"  the closed form 960x^9 at x = 1: {960 * 1.0 ** 9:g}")
    print(f"  outer derivative at the wrong point, f'(x) * g'(x): {df(1.0) * dg(1.0):g}")
    for x in (0.5, 2.0):
        slope, _ = chain_derivative([g, f], [dg, df], x)
        print(f"  x = {x}: chain rule {slope:g}, 960x^9 = {960 * x ** 9:g}")

    print()
    print("== Section 5: three functions, y = 2((3x + 1)^2)^3 at x = 1")
    report("h(x) = 3x + 1, g(u) = u^2, f(v) = 2v^3",
           [lambda x: 3 * x + 1, lambda u: u * u, lambda v: 2 * v ** 3],
           [lambda x: 3.0, lambda u: 2 * u, lambda v: 6 * v * v], 1.0)
    print(f"  expanded first: y = 2(3x + 1)^6, so dy/dx = 36(3x + 1)^5 = {36 * 4 ** 5:g} at x = 1")

    print()
    print("== Section 5.1: two paths, L = u + v with u = 2x and v = x^2, at x = 3")
    x = 3.0
    path_u, path_v = 1.0 * 2.0, 1.0 * (2 * x)
    measured = central_difference(lambda t: 2 * t + t * t, x)
    print(f"  path through u: 1 * 2 = {path_u:g}   path through v: 1 * 2x = {path_v:g}   sum {path_u + path_v:g}")
    print(f"  central difference {measured:.6f}")

    print()
    print("== Section 9: f(x) = ReLU(2x - 1), whose corner is at x = 0.5")
    print("     x   inner 2x - 1   ReLU slope there   chain rule   central difference")
    for x in (0.0, 0.5, 1.0, 2.0):
        slope, factors = chain_derivative([lambda t: 2 * t - 1, relu], [lambda t: 2.0, relu_slope], x)
        measured = central_difference(lambda t: relu(2 * t - 1), x)
        print(f"  {x:4.1f}   {2 * x - 1:12.1f}   {factors[1]:16.1f}   {slope:10.1f}   {measured:18.6f}")


if __name__ == "__main__":
    main()
