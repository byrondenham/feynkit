"""Tests for feynkit.strata.singular_strata and singular_strata_from_polynomial.

Oracles are hand examples in smooth charts, where beta is the signed sum over the subsets I of
the normal coordinates of chi~ of the Milnor fibre of g restricted to y_I = 0 (see
:class:`feynkit.milnor.TorusCutMilnorNumber`):
- a node u v times a unit, times a free line, has Milnor fibre C^* times a ball, so chi~ = -1;
- u^2 times a unit has a Milnor fibre of two sheets, so chi~ = 1;
- u^2 w, with w a coordinate, has Milnor fibre {u^2 w = eps}, which is C^*, so chi~ = -1;
- h^2 for a smooth h has chi~ = 1, and for h a quadric cone in three variables at its vertex the
  fibre is two spheres, chi~ = 3.
Where a point is the only difference, the value there is compared with
torus_cut_milnor_number called directly in the chart of the stratum.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import strata as strata_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.milnor import TorusCutMilnorNumber, torus_cut_milnor_number
from feynkit.polytope import polytope_data
from feynkit.strata import StrataAnalysis, Stratum, singular_strata, singular_strata_from_polynomial
from feynkit.toric import NormalCone, normal_cone, smooth_subdivision

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

X1, X2, X3 = sp.symbols("x1 x2 x3")


def points_of(g: sp.Expr, variables: list[sp.Symbol]) -> tuple[tuple[int, ...], ...]:
    return polytope_data(sorted(sp.Poly(sp.expand(g), *variables).monoms())).points


def top_face(g: sp.Expr, variables: list[sp.Symbol]) -> list[int]:
    return list(range(len(points_of(g, variables))))


def analyse(
    g: sp.Expr,
    variables: list[sp.Symbol],
    face: Callable[[tuple[int, ...]], bool] | None = None,
    **kw: Any,
) -> StrataAnalysis:
    """The strata of the face of the points where ``face`` holds, the top face by default."""
    g = sp.expand(g)
    points = points_of(g, variables)
    indices = [i for i, p in enumerate(points) if face is None or face(p)]
    return singular_strata_from_polynomial(g, variables, faces=[indices], timeout=60, **kw)


def point_of(stratum: Stratum) -> list[Fraction]:
    """The coordinates of a stratum that is one rational point."""
    t = sorted(set().union(*(g.free_symbols for g in stratum.generators)), key=str)
    solution = sp.solve(list(stratum.generators), t, dict=True)[0]
    return [Fraction(int(solution[x].p), int(solution[x].q)) for x in t]


def by_dimension(analysis: StrataAnalysis) -> dict[int, list[Stratum]]:
    out: dict[int, list[Stratum]] = {}
    for s in analysis.strata:
        assert s.dimension is not None
        out.setdefault(s.dimension, []).append(s)
    return out


# --- hand examples in smooth charts --------------------------------------------------------


@requires_singular
def test_normal_crossing_curves_and_their_triple_point() -> None:
    """G = (x - 1)(y - 1)(1 + z): the singular locus of the top face is three lines through
    (1, 1, -1), reducible. Along a line the germ is a node times a unit times a line, so beta = -1
    and mu^T = (-1)^(3-1) beta = -1. At the common point G = u v w with a unit 1, whose fibre is
    (C^*)^2, chi~ = -1, so mu^T = -1 there too. The point lies on three lines, so it is a stratum
    of its own."""
    g = (X1 - 1) * (X2 - 1) * (1 + X3)
    a = analyse(g, [X1, X2, X3])
    assert a.complete and a.failures == ()
    assert a.primes
    pieces = by_dimension(a)
    lines, points = pieces[1], pieces[0]
    assert len(lines) == 3 and len(points) == 1
    assert [s.mu_t for s in lines] == [-1, -1, -1]
    assert all(len(s.removed) == 1 for s in lines)
    assert points[0].mu_t == -1 and point_of(points[0]) == [1, 1, -1]
    assert all(s.reason is None and s.chart is not None and s.euler is None for s in a.strata)
    assert a.seed == 0


@requires_singular
def test_a_surface_stratum() -> None:
    """G = (x - 1)^2 (y + z + 1): the top face is singular along the plane x = 1, where the germ is
    u^2 times a unit, chi~ = 1 and mu^T = (-1)^(3-1) 1 = 1. Along the curve u = 0, w = 0 it is
    u^2 w, chi~ = -1."""
    g = (X1 - 1) ** 2 * (X2 + X3 + 1)
    a = analyse(g, [X1, X2, X3])
    assert a.complete
    pieces = by_dimension(a)
    surface, curve = pieces[2][0], pieces[1][0]
    assert (surface.mu_t, curve.mu_t) == (1, -1)
    assert surface.degree == 1 and len(surface.removed) == 1
    assert curve.removed == ()


@requires_singular
def test_a_curve_whose_value_jumps_at_a_point() -> None:
    """G = u^2 - v^2 (v + w^2) with u = x - 1, v = y - 1, w = z - 1: the singular locus u = v = 0
    is a line, transversally a node, so mu^T = -1 along it, except at w = 0, where the value jumps.
    The value at the point is that of torus_cut_milnor_number, called directly there."""
    g = (X1 - 1) ** 2 - (X2 - 1) ** 2 * ((X2 - 1) + (X3 - 1) ** 2)
    a = analyse(g, [X1, X2, X3])
    assert a.complete
    pieces = by_dimension(a)
    curve, point = pieces[1][0], pieces[0][0]
    assert curve.mu_t == -1 and point.mu_t != -1 and point.removed == ()
    assert curve.removed == (point.generators,)
    sp_g = sp.expand(g)
    face = [tuple(m) for m in points_of(sp_g, [X1, X2, X3])]
    direct = torus_cut_milnor_number(
        sp_g, [X1, X2, X3], face, point_of(point), chart=point.chart, timeout=60
    )
    assert direct.reason is None and direct.value == point.mu_t


@requires_singular
def test_the_extended_candidates_separate_jumps_on_a_larger_critical_component() -> None:
    """G = h^2 (1 + x1) with h = (x1 - 1) x3 - (x2 - 2)^2, on the facet x3 = 0, where the chart has
    one normal coordinate y = x3 and g_F = (x2 - 2)^4 (1 + x1).

    Its singular locus is the curve A = {x2 = 2}, which lies in the critical locus of g, the
    surface h = 0 (a quadric cone with vertex (1, 2, 0) on A). Along A, h is smooth and g is
    h^2 times a unit: the term for I = {} is chi~ = 1, and the term for I = {y}, g_F, is
    (x2 - 2)^4 times a unit, four sheets, chi~ = 3, so beta = -2 and mu^T = (-1)^2 (-2) = -2. At the
    vertex of the cone the first term is two spheres, chi~ = 3, so beta = 0. At x1 = -1 the factor
    1 + x1 vanishes, g is a node times a unit in both terms, chi~ = -1, beta = 0. The pair (A, {})
    is of the kind where A lies in a larger component of the critical locus, and the jump at the
    vertex is on A meet Sing C.
    """
    g = (((X1 - 1) * X3 - (X2 - 2) ** 2) ** 2) * (1 + X1)
    pairs: list[tuple[str, list[str]]] = []
    original = strata_module._Face.pair

    def spy(self: Any, member: Any, subset: tuple[int, ...], flag: int) -> Any:
        found = original(self, member, subset, flag)
        pairs.append((found[1], found[0]))
        return found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(strata_module._Face, "pair", spy)
        a = analyse(g, [X1, X2, X3], lambda p: p[2] == 0)
    assert a.complete and a.failures == ()
    pieces = by_dimension(a)
    curve, points = pieces[1][0], pieces[0]
    assert curve.mu_t == -2 and len(curve.removed) == 2
    assert sorted(point_of(p)[0] for p in points) == [-1, 1] and [p.mu_t for p in points] == [0, 0]
    assert "E" in {kind for kind, _ in pairs}


@requires_singular
def test_the_condition_fails_where_the_critical_component_is_singular_along_the_member() -> None:
    """G = ((x - 1)^2 - (y - 1)^3)^2 (1 + z) has the line x = y = 1 in its singular locus, and in
    the critical locus of g a component, the cuspidal cylinder h = 0, which is singular along that
    line. The condition fails there, so complete is False and the pair is listed."""
    g = (((X1 - 1) ** 2 - (X2 - 1) ** 3) ** 2) * (1 + X3)
    a = analyse(g, [X1, X2, X3])
    assert not a.complete
    reasons = [f.reason for f in a.failures]
    assert any("singular locus of a component" in r for r in reasons)
    failing = [f for f in a.failures if f.dimension == 1]
    assert failing and failing[0].coordinates == () and failing[0].face == tuple(range(34))
    assert set(map(str, failing[0].generators)) == {"t1 - 1", "t2 - 1"}


# --- points over a number field -------------------------------------------------------------


@requires_singular
@pytest.mark.parametrize(
    ("irrational", "rational", "degree"),
    [
        ((X1**2 - 2) ** 2 * (X1 + 1), (X1**2 - 1) ** 2 * (X1 + 3), 2),
        ((X1**3 - 2) ** 2 * (X1 + 1), (X1**3 - 1) ** 2 * (X1 + 3), 3),
        ((X1**2 + 1) ** 2 * (X1 + 2), (X1**2 - 1) ** 2 * (X1 + 3), 2),
    ],
    ids=["sqrt2", "cbrt2", "i"],
)
def test_conjugate_points_are_decided_over_their_number_field(
    irrational: sp.Expr, rational: sp.Expr, degree: int
) -> None:
    """G = h^2 (x + 1) with h irreducible: the double roots are conjugate points, one member of
    that degree, an A_1 point of a function of one variable. beta comes from one root over Q(a),
    twice; it is that of the same polynomial with rational roots."""
    a = analyse(irrational, [X1])
    (stratum,) = a.strata
    assert stratum.degree == degree and stratum.dimension == 0
    assert stratum.reason is None and a.complete
    want = analyse(rational, [X1])
    assert [t.mu_t for t in want.strata] == [stratum.mu_t] * len(want.strata)
    assert stratum.mu_t == 1
    assert a.primes and all(p < 2**29 for p in a.primes)


@requires_singular
def test_a_curve_without_rational_points_is_decided_at_two_points_over_number_fields() -> None:
    """G = h^2, h = x^2 + x + y^2 + y + 1, which has no real point, so no rational point: the
    singular locus of the top face is the curve V(h), and the points off the smaller members
    (the conjugate pairs where it meets the lines x^2 + x + 1 = 0, ...) are algebraic. Its value
    is that of the double line (x + y - 1)^2."""
    h = X1**2 + X1 + X2**2 + X2 + 1
    a = analyse(h**2, [X1, X2])
    curves = by_dimension(a)[1]
    assert len(curves) == 1 and curves[0].reason is None
    want = by_dimension(analyse((X1 + X2 - 1) ** 2, [X1, X2]))[1]
    assert curves[0].mu_t == want[0].mu_t == -1
    assert a.complete
    assert all(s.mu_t == 0 for s in a.strata if s.dimension == 0)


@requires_singular
def test_a_member_with_too_few_points_is_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def none_found(self: Any, comp: Any, avoid: Any, want: int) -> list:
        return []

    monkeypatch.setattr(strata_module._Face, "points", none_found)
    a = analyse((X1 - 1) ** 2 * (X2 + 2), [X1, X2])
    assert not a.complete
    assert all("points off the smaller members" in (s.reason or "") for s in a.strata)


# --- undecided paths -----------------------------------------------------------------------


@requires_singular
def test_an_undecided_beta_becomes_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def undecided(*args: Any, **kwargs: Any) -> TorusCutMilnorNumber:
        return TorusCutMilnorNumber(
            None, None, "smooth chart", (), None, "restriction to y_i = 0 for i in []: timed out"
        )

    monkeypatch.setattr(strata_module, "torus_cut_milnor_number", undecided)
    a = analyse((X1 - 1) * (X2 - 1) * (1 + X3), [X1, X2, X3])
    assert not a.complete
    assert a.strata and all(s.mu_t is None and s.reason for s in a.strata)
    assert all("timed out" in (s.reason or "") for s in a.strata)


@requires_singular
def test_two_points_that_disagree_are_undecided(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = itertools.count()

    def moving(*args: Any, **kwargs: Any) -> TorusCutMilnorNumber:
        n = next(counter)
        return TorusCutMilnorNumber(n, n, "smooth chart", (), 536870909, None)

    monkeypatch.setattr(strata_module, "torus_cut_milnor_number", moving)
    a = analyse((X1 - 1) ** 2 * (X2 + X3 + 1), [X1, X2, X3])
    assert not a.complete
    surface = by_dimension(a)[2][0]
    assert surface.mu_t is None and "differs" in (surface.reason or "")


@requires_singular
def test_a_candidate_computation_that_times_out_is_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow(self: Any, member: Any, subset: tuple[int, ...], flag: int) -> Any:
        raise ComputationError("undecided: Singular ran past timeout=1 s")

    monkeypatch.setattr(strata_module._Face, "pair", slow)
    a = analyse((X1 - 1) * (X2 - 1) * (1 + X3), [X1, X2, X3])
    assert not a.complete
    lines = by_dimension(a)[1]
    assert all(s.mu_t is None and "ran past timeout" in (s.reason or "") for s in lines)


@requires_singular
def test_a_failing_singular_run_on_a_face_is_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise ComputationError("undecided: Singular ran past timeout=1 s")

    monkeypatch.setattr(strata_module._Face, "singular_locus", failing)
    a = analyse((X1 - 1) * (X2 - 1) * (1 + X3), [X1, X2, X3])
    (stratum,) = a.strata
    assert stratum.dimension is None and stratum.reason == "Singular ran past timeout=1 s"
    assert not a.complete


# --- Feynman integrals ----------------------------------------------------------------------


@requires_singular
def test_a_generic_point_of_a_massive_bubble_has_no_strata() -> None:
    """At a generic point no face polynomial is singular in its orbit (Kouchnirenko), so the
    analysis is empty and complete."""
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    symbols = {s.name: s for s in fi.symanzik.g.free_symbols if s.name in ("m_1", "m_2", "s")}
    a = singular_strata(fi, {symbols["m_1"] ** 2: 1, symbols["m_2"] ** 2: 4, symbols["s"]: 7})
    assert a.strata == () and a.failures == () and a.complete
    assert len(a.points) == len(fi.newton_polytope.points)


def test_kinematic_constraints_are_refused() -> None:
    s = sp.Symbol("s", real=True)
    fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
    with pytest.raises(ValidationError, match="kinematic_constraints"):
        singular_strata(fi, {})


# --- subdivision limit and validation ------------------------------------------------------


def octahedron_cone() -> NormalCone:
    points = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    data = polytope_data(points)
    vertex = next(f for d, f in data.faces if d == 0)
    return normal_cone(data, vertex)


def test_the_subdivision_timeout_is_undecided() -> None:
    cone = octahedron_cone()
    assert not cone.smooth
    with pytest.raises(ComputationError, match="undecided: .*timeout"):
        smooth_subdivision(cone, timeout=1e-9)
    assert smooth_subdivision(cone, timeout=60) == smooth_subdivision(cone)


def test_the_subdivision_timeout_is_validated() -> None:
    cone = octahedron_cone()
    for bad in (0, -1, True, "1"):
        with pytest.raises(ValidationError, match="timeout"):
            smooth_subdivision(cone, timeout=bad)  # type: ignore[arg-type]


@requires_singular
def test_bad_input_raises() -> None:
    g = (X1 - 1) * (X2 - 1)
    with pytest.raises(ValidationError, match="seed"):
        singular_strata_from_polynomial(g, [X1, X2], seed="0")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="timeout"):
        singular_strata_from_polynomial(g, [X1, X2], timeout=0)
    with pytest.raises(ValidationError, match="not the point indices of a face"):
        singular_strata_from_polynomial(g, [X1, X2], faces=[[0, 3]])
    with pytest.raises(ValidationError, match="faces"):
        singular_strata_from_polynomial(g, [X1, X2], faces="top")  # type: ignore[arg-type]
    a = sp.Symbol("a")
    with pytest.raises(ValidationError, match="no value"):
        singular_strata_from_polynomial(a * g, [X1, X2])


@requires_singular
def test_missing_singular_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strata_module, "_singular_binary", lambda: None)
    with pytest.raises(RuntimeError, match="Singular"):
        singular_strata_from_polynomial((X1 - 1) * (X2 - 1), [X1, X2])


def test_stratum_is_frozen() -> None:
    s = Stratum((0,), 0, (), 1, (), None, None, None, "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.reason = "y"  # type: ignore[misc]
