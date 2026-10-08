"""Tests for feynkit.milnor.torus_cut_milnor_number in smooth charts and on the Newton path.

Oracles are hand examples. In a chart C^r_y x (C^*)^k_t, beta(x) is the sum over subsets
I of the y coordinates of (-1)^|I| chi~ of the Milnor fibre of g restricted to y_I = 0, and
mu^T = (-1)^(N-1) beta, N the dimension of the polytope (see TorusCutMilnorNumber).

Hand values used below:
- chi~ of a germ with an isolated critical point is (-1)^m mu in m variables, mu the Milnor
  number: s^2 + y^2 has mu = 1 and s^2 has mu = 1 in one variable, so beta = -1 - 1 = -2.
- A node u v times a unit, times a free line, has Milnor fibre C^* x ball, so chi = 0 and
  chi~ = -1 (see test_normal_crossing_curve).
- A germ that is a submersion has a contractible Milnor fibre, so chi~ = 0.
- The germ y s has Milnor fibre {y s = eps}, which lies in T and is C^*: chi = 0, so beta = 0.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import time
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import milnor as milnor_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.milnor import TorusCutMilnorNumber, torus_cut_milnor_number
from feynkit.polytope import polytope_data
from feynkit.toric import orbit_chart

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

X, Y, Z = sp.symbols("x y z")
METHODS = ("smooth chart", "newton")


def terms_of(polynomial: sp.Expr, variables: list[sp.Symbol]) -> list[tuple[int, ...]]:
    """The exponent vectors of the terms of a polynomial."""
    return sorted(sp.Poly(sp.expand(polynomial), *variables).monoms())


def box_germ(k: int, r: int, seed: int) -> tuple[sp.Expr, list[sp.Symbol], list[tuple[int, ...]]]:
    """A random polynomial on the box [0, 2]^k x [0, 3]^r whose chart at the face y = 0 is the
    identity chart: t = x_1, ..., x_k and y = x_(k+1), ..., x_(k+r).

    All the corners of the box are in the support, so the facets through the face y = 0 are the
    coordinate facets and its normal cone is the orthant; exponents 2 and 3 in y make the lattice
    Z^(k + r). The face polynomial is sum (t_i - 1)^2 (times prod (t_i - 1)^2 for k > 1), which
    has a singular point at t = 1; terms of y-degree below 2 are left out, so that g is not a
    submersion in y. Returns G, the variables and the terms on the face.
    """
    rng = random.Random(seed)
    xs = list(sp.symbols(f"x1:{k + r + 1}"))
    t, y = xs[:k], xs[k:]
    u = [ti - 1 for ti in t]
    g = sum(ui**2 for ui in u) + (sp.prod([ui**2 for ui in u]) if k > 1 else 0)
    for b in itertools.product(range(4), repeat=r):
        if sum(b) < 2:
            continue
        for a in itertools.product(range(3), repeat=k):
            corner = all(e in (0, 2) for e in a) and all(e in (0, 3) for e in b)
            if not corner and rng.random() < 0.85:
                continue
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            g += (
                c
                * sp.prod([ti**e for ti, e in zip(t, a, strict=True)])
                * sp.prod([yi**e for yi, e in zip(y, b, strict=True)])
            )
    g = sp.expand(g)
    face = [m for m in terms_of(g, xs) if not any(m[k:])]
    return g, xs, face


@requires_singular
@pytest.mark.parametrize("k, r", [(1, 1), (1, 2), (2, 1)])
@pytest.mark.parametrize("seed", [0, 1])
def test_newton_path_agrees_with_smooth_chart_on_random_germs(k: int, r: int, seed: int) -> None:
    """Item 1: the two routes give the same beta and mu^T on seeded random germs on C^r x
    (C^*)^k that are nondegenerate in the chart coordinates."""
    g, xs, face = box_germ(k, r, seed)
    chart = torus_cut_milnor_number(g, xs, face, [1] * k, method="smooth chart", seed=seed)
    newton = torus_cut_milnor_number(g, xs, face, [1] * k, method="newton")
    assert chart.reason is None and newton.reason is None
    assert chart.method == "smooth chart" and newton.method == "newton"
    assert (chart.value, chart.beta) == (newton.value, newton.beta)
    # N = k + r, so mu^T = (-1)^(k + r - 1) beta.
    assert chart.value == (-1) ** (k + r - 1) * chart.beta
    assert chart.value != 0
    assert len(chart.terms) == 2**r


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_normal_crossing_curve(method: str) -> None:
    """Item 2: mu^T = -1 along a normal-crossing curve.

    G = (x - 1)(y - 1)(1 + z) has the full cube as Newton polytope, N = 3. Near (1, 1, 1) it is
    u v times the unit 2 + (z - 1) with u = x - 1 and v = y - 1, so V(G) is {u v = 0} times
    the z line: a curve of normal crossings of two smooth sheets. The Milnor fibre {u v = eps}
    in the (u, v) plane is C^*, and the fibre of G is that times a ball, so chi(F) = 0 and
    chi~(F) = -1. The point is in T (r = 0), so beta = chi~(F) = -1 and
    mu^T = (-1)^(3 - 1) (-1) = -1.
    """
    g = (X - 1) * (Y - 1) * (1 + Z)
    result = torus_cut_milnor_number(g, [X, Y, Z], terms_of(g, [X, Y, Z]), [1, 1, 1], method=method)
    assert result.reason is None
    assert (result.value, result.beta) == (-1, -1)
    assert result.method == method


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_a_2_singularity_in_the_torus(method: str) -> None:
    """G = (x - 1)^2 + (y - 1)^3, N = 2, at (1, 1) in T: an A_2 point, mu = 2. In two
    variables chi~ = (-1)^1 mu = -2, and the point is in T, so beta = -2 and
    mu^T = (-1)^(2 - 1) (-2) = 2 = mu."""
    g = (X - 1) ** 2 + (Y - 1) ** 3
    result = torus_cut_milnor_number(g, [X, Y], terms_of(g, [X, Y]), [1, 1], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (-2, 2)


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_submersion(method: str) -> None:
    """G = (x - 1) + (y - 1)^2 + (x - 1)(y - 1) has a non-zero differential at (1, 1), so its
    Milnor fibre is contractible: beta = mu^T = 0."""
    g = (X - 1) + (Y - 1) ** 2 + (X - 1) * (Y - 1)
    result = torus_cut_milnor_number(g, [X, Y], terms_of(g, [X, Y]), [1, 1], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (0, 0)


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_a_boundary_point_with_a_morse_fibre(method: str) -> None:
    """G = (x - 1)^2 + y^2 + y^3 with the face y = 0 (an edge of the triangle with vertices
    (0, 0), (2, 0), (0, 3)); N = 2, r = 1 and g = s^2 + y^2 (1 + y) at s = 0.

    I = {}: s^2 + y^2 (unit) is a Morse point of a function of two variables, chi~ = -1.
    I = {1}: s^2 alone, chi~ = +1 (the fibre is two points). beta = -1 - 1 = -2, so
    mu^T = (-1)^1 (-2) = 2, the sum of the Milnor numbers of the restrictions (1 + 1).
    """
    g = (X - 1) ** 2 + Y**2 + Y**3
    face = [m for m in terms_of(g, [X, Y]) if m[1] == 0]
    result = torus_cut_milnor_number(g, [X, Y], face, [1], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (-2, 2)
    if method == "smooth chart":
        assert dict(result.terms) == {(): -1, (0,): 1}
    else:
        # chi of the pieces of the fibre with y != 0: with s = 0 the fibre y^2 (1 + y) = eps has
        # two points near 0; with s != 0 the Milnor fibre, a cylinder over a circle, has chi = 0
        # and loses the two points with y = 0 (s^2 = eps) and the two with s = 0, so -4.
        assert dict(result.terms) == {(): 2, (0,): -4}


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_a_morse_point_times_a_line(method: str) -> None:
    """G = ((x - 1)^2 + y^2 + y^3)(1 + z), N = 3, with the face y = 0 and the point
    (x, z) = (1, 1), where 1 + z = 2 is a unit. The critical locus of g is a line (the z
    direction), transversally the Morse point of the previous test.

    The product with a line does not change the Milnor fibre up to homotopy, so the terms
    are those of the previous test: chi~ = -1 for I = {} and +1 for I = {1}, beta = -2, but now
    mu^T = (-1)^(3 - 1) (-2) = -2.
    """
    g = (X - 1) ** 2 * (1 + Z) + Y**2 * (1 + Z) + Y**3 * (1 + Z)
    face = [m for m in terms_of(g, [X, Y, Z]) if m[1] == 0]
    result = torus_cut_milnor_number(g, [X, Y, Z], face, [1, 1], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (-2, -2)
    if method == "smooth chart":
        assert dict(result.terms) == {(): -1, (0,): 1}


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_a_germ_nonzero_in_y_only(method: str) -> None:
    """G = (x - 1)^2 + y with face y = 0: g = s^2 + y is a submersion in y, so the restriction
    to y = 0 is s^2 with chi~ = 1 and beta = 0 - 1 = -1, mu^T = (-1)^1 (-1) = 1.

    (The lattice spanned by the exponents is Z^2: (0, 0), (1, 0), (2, 0), (0, 1).)
    """
    g = (X - 1) ** 2 + Y
    face = [m for m in terms_of(g, [X, Y]) if m[1] == 0]
    result = torus_cut_milnor_number(g, [X, Y], face, [1], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (-1, 1)


@requires_singular
def test_a_restriction_that_is_identically_zero() -> None:
    """g = y s, a node whose restriction to y = 0 is the zero polynomial.

    The fibre of the zero function is empty, chi~ = -1, and beta = chi~(y s) - (-1) with
    chi~(y s) = -1 (a node in two variables), so beta = 0: the Milnor fibre {y s = eps} lies
    in T and is C^*, chi = 0 = beta + 1_T(x), 1_T(x) = 0. A face polynomial is never zero, so
    this cannot come from torus_cut_milnor_number; the chart-level helper is called directly.
    """
    local = {(1, 1): Fraction(1)}
    result = milnor_module._smooth_chart(local, 1, 1, [Fraction(0)], [0], 0, 60, 1)
    assert dict(result.terms) == {(): -1, (0,): -1}
    assert (result.beta, result.value) == (0, 0)


def test_a_point_off_the_hypersurface_is_a_caller_error() -> None:
    """Where the face polynomial does not vanish the numbers are not defined here."""
    g = (X - 1) ** 2 + (Y - 1) ** 3
    for method in METHODS:
        with pytest.raises(ValidationError, match="closure of V"):
            torus_cut_milnor_number(g, [X, Y], terms_of(g, [X, Y]), [2, 1], method=method)


@requires_singular
@pytest.mark.parametrize("method", METHODS)
def test_coordinates_follow_the_chart_basis_away_from_one(method: str) -> None:
    """G = (x y - 2)^2 + y^2 + y^3 with the face x = y (the exponents (0,0), (1,1), (2,2)).

    The lattice chart basis is not the identity here: the chart coordinate along the face is
    t = (x y)^(-1) (the result carries the chart), so the face polynomial (x y - 2)^2 =
    (1/t - 2)^2 vanishes at t = 1/2 and not at t = 2. At t = 1/2 the germ is the Morse
    point of the earlier test, beta = -2, mu^T = 2; at t = 2 the point is not on V(G).
    """
    g = (X * Y - 2) ** 2 + Y**2 + Y**3
    face = [m for m in terms_of(g, [X, Y]) if m[0] == m[1]]
    result = torus_cut_milnor_number(g, [X, Y], face, [Fraction(1, 2)], method=method)
    assert result.reason is None
    assert (result.beta, result.value) == (-2, 2)
    assert result.chart is not None and result.chart.normal_dimension == 1
    with pytest.raises(ValidationError, match="closure of V"):
        torus_cut_milnor_number(g, [X, Y], face, [2], method=method)


@requires_singular
def test_the_chart_can_be_passed_and_is_checked() -> None:
    g = (X * Y - 2) ** 2 + Y**2 + Y**3
    points = terms_of(g, [X, Y])
    face = [m for m in points if m[0] == m[1]]
    data = polytope_data(points)
    indices = [data.points.index(m) for m in face]
    chart = orbit_chart(data, tuple(sorted(indices)))
    result = torus_cut_milnor_number(g, [X, Y], face, [Fraction(1, 2)], chart=chart)
    assert result.chart == chart and (result.beta, result.value) == (-2, 2)
    other = orbit_chart(data, tuple(sorted(indices)))
    wrong = dataclasses.replace(other, vertex=(5, 5))
    with pytest.raises(ValidationError):
        torus_cut_milnor_number(g, [X, Y], face, [Fraction(1, 2)], chart=wrong)
    with pytest.raises(ValidationError):
        torus_cut_milnor_number(g, [X, Y], face, [Fraction(1, 2)], chart="chart")  # type: ignore[arg-type]


@requires_singular
def test_terms_and_primes_of_the_two_routes() -> None:
    g = (X - 1) ** 2 + Y**2 + Y**3
    face = [m for m in terms_of(g, [X, Y]) if m[1] == 0]
    chart = torus_cut_milnor_number(g, [X, Y], face, [1])
    assert isinstance(chart, TorusCutMilnorNumber)
    assert chart.prime is None or chart.prime < 2**29
    assert [subset for subset, _ in chart.terms] == [(), (0,)]
    newton = torus_cut_milnor_number(g, [X, Y], face, [1], method="newton")
    assert newton.prime is None
    assert [subset for subset, _ in newton.terms] == [(), (0,)]


# --- the undecided paths ---------------------------------------------------------------------


@requires_singular
def test_newton_path_is_undecided_on_a_degenerate_germ() -> None:
    """G = (x - y)^2 + (x - 1)^3 at (1, 1): in the coordinates s = x - 1, s' = y - 1 the
    face polynomial (s - s')^2 of the edge from (2, 0) to (0, 2) has singular points in the
    torus, so Cor. 3.6 does not apply. The germ itself is u^2 + s^3, an A_2 point, mu^T = 2
    (beta = chi~ = -2, N = 2), which the smooth-chart route finds."""
    g = (X - Y) ** 2 + (X - 1) ** 3
    face = terms_of(g, [X, Y])
    newton = torus_cut_milnor_number(g, [X, Y], face, [1, 1], method="newton")
    assert newton.value is None and newton.beta is None and newton.terms == ()
    assert newton.reason is not None and "nondegeneracy" in newton.reason
    chart = torus_cut_milnor_number(g, [X, Y], face, [1, 1], method="smooth chart")
    assert (chart.beta, chart.value) == (-2, 2)


@requires_singular
def test_a_timeout_becomes_a_reason() -> None:
    g = (X - 1) * (Y - 1) * (1 + Z)
    for method in METHODS:
        result = torus_cut_milnor_number(
            g, [X, Y, Z], terms_of(g, [X, Y, Z]), [1, 1, 1], method=method, timeout=0.001
        )
        assert result.value is None and result.beta is None
        assert result.reason is not None and "timeout" in result.reason


def test_an_undecided_error_from_the_lê_computation_becomes_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise ComputationError("undecided: two flags disagree, for the test")

    monkeypatch.setattr(milnor_module, "_decide", refuse)
    g = (X - 1) ** 2 + (Y - 1) ** 3
    result = torus_cut_milnor_number(g, [X, Y], terms_of(g, [X, Y]), [1, 1])
    assert (result.value, result.beta, result.terms, result.prime) == (None, None, (), None)
    assert result.reason is not None and "two flags disagree, for the test" in result.reason
    assert not result.reason.startswith("undecided: ")


def test_another_computation_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ComputationError("a bug")

    monkeypatch.setattr(milnor_module, "_decide", broken)
    g = (X - 1) ** 2 + (Y - 1) ** 3
    with pytest.raises(ComputationError, match="a bug"):
        torus_cut_milnor_number(g, [X, Y], terms_of(g, [X, Y]), [1, 1])


def test_compact_faces_are_bounded_by_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    real = milnor_module.polytope_data

    def slow(points: object) -> object:
        time.sleep(0.6)
        return real(points)  # type: ignore[arg-type]

    monkeypatch.setattr(milnor_module, "polytope_data", slow)
    g = (X - 1) ** 2 + (Y - 1) ** 3
    start = time.monotonic()
    result = torus_cut_milnor_number(
        g, [X, Y], terms_of(g, [X, Y]), [1, 1], method="newton", timeout=0.5
    )
    assert time.monotonic() - start < 10
    assert result.value is None and "compact faces" in (result.reason or "")


def test_a_cone_that_is_not_smooth_is_undecided() -> None:
    """The vertex (0, 0) of conv((0, 0), (2, 1), (1, 2), (1, 1)) has a cone of multiplicity 3."""
    g = 1 + X**2 * Y + X * Y**2 + X * Y
    result = torus_cut_milnor_number(g, [X, Y], [(0, 0)], [], method="smooth chart")
    assert result.value is None and result.reason is not None
    assert "not smooth" in result.reason


# --- bad input -------------------------------------------------------------------------------


def test_bad_input_raises() -> None:
    g = (X - 1) ** 2 + Y**2 + Y**3
    face = [(0, 0), (1, 0), (2, 0)]
    good = {"polynomial": g, "variables": [X, Y], "face": face, "coordinates": [1]}

    def call(**changes: object) -> TorusCutMilnorNumber:
        return torus_cut_milnor_number(**{**good, **changes})  # type: ignore[arg-type]

    bad = [
        {"polynomial": sp.Integer(0)},
        {"polynomial": sp.sqrt(2) * X + Y},
        {"polynomial": 1 / (X + 1) + Y},
        {"polynomial": Z + X},
        {"variables": []},
        {"variables": [X, X]},
        {"variables": [X, 3]},
        {"face": [(0, 0), (1, 0)]},
        {"face": [(5, 5)]},
        {"face": [(0, "a")]},
        {"face": "abc"},
        {"coordinates": [1, 1]},
        {"coordinates": []},
        {"coordinates": [0]},
        {"coordinates": [0.5]},
        {"coordinates": "1"},
        {"seed": 1.5},
        {"seed": True},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": "5"},
        {"method": "subdivision"},
        {"method": "other"},
    ]
    for changes in bad:
        with pytest.raises(ValidationError):
            call(**changes)


Y1, Y2, Y3, X2 = sp.symbols("y1 y2 y3 x2")


@requires_singular
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "name, g, variables, k, expected",
    [
        # Every restriction is Morse (the y1^3 term makes the lattice Z^3): beta = mu^T = 4.
        (
            "r = 2",
            (X - 1) ** 2 + Y1 * Y2 + Y1**2 + Y2**2 + Y1**3,
            [X, Y1, Y2],
            1,
            (4, 4),
        ),
        # r = 3, Morse in every y-subset; mu^T = sum of the Milnor numbers 1 * 2^3 = 8.
        (
            "r = 3",
            (X - 1) ** 2 + Y1**2 + Y2**2 + Y3**2 + Y1 * Y2 + Y2 * Y3 + Y1 * Y3 + Y1**3,
            [X, Y1, Y2, Y3],
            1,
            (-8, 8),
        ),
        # k = 2, r = 2: four variables, 2^2 Morse restrictions, mu^T = 4.
        (
            "k = 2, r = 2",
            (X - 1) ** 2 + (X2 - 1) ** 2 + Y1**2 + Y2**2 + Y1 * Y2 + Y1**3,
            [X, X2, Y1, Y2],
            2,
            (-4, 4),
        ),
    ],
)
def test_morse_germs_with_several_normal_coordinates(
    method: str,
    name: str,
    g: sp.Expr,
    variables: list[sp.Symbol],
    k: int,
    expected: tuple[int, int],
) -> None:
    """beta = sum over I of (-1)^|I| chi~, each restriction a Morse point of m variables with
    chi~ = (-1)^(m-1); mu^T = (-1)^(N-1) beta with N = the number of variables. The tuple is
    (beta, mu^T)."""
    face = [m for m in terms_of(g, variables) if not any(m[k:])]
    result = torus_cut_milnor_number(g, variables, face, [1] * k, method=method)
    assert result.reason is None, name
    assert (result.beta, result.value) == expected
