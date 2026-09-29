"""
Lattice invariants against textbook values, brute force and each other.

The brute force tests membership of every integer point of a box in kP,
using the facets and affine hull of polytope_data, and membership in the
support lattice by solving for chart coordinates with SymPy.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from sympy.matrices.normalforms import hermite_normal_form

from feynkit import _exact
from feynkit import lattice_invariants as li
from feynkit.core.exceptions import ValidationError
from feynkit.lattice_invariants import (
    count_lattice_points,
    ehrhart_polynomial,
    gorenstein_index,
    h_star_vector,
    invariant_chart,
    lattice_points,
    lattice_width,
    polar_dual,
)
from feynkit.polytope import polytope_data

K = sp.Symbol("k")

Points = list[tuple[int, ...]]


def simplex(d: int) -> Points:
    return [(0,) * d] + [tuple(int(i == j) for j in range(d)) for i in range(d)]


def cube(d: int) -> Points:
    return list(itertools.product((0, 1), repeat=d))


def reeve(r: int) -> Points:
    return [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, r)]


def values(poly: sp.Poly, ks: Sequence[int]) -> list[sp.Rational]:
    return [poly.eval(k) for k in ks]


def support_lattice(points: Points) -> Callable[[Sequence[int], int], bool]:
    """A test of x - k alpha_1 for membership in the lattice L the differences span.

    The columns of SymPy's Hermite normal form H of the differences are a
    basis of L, so a vector v of the linear span of L lies in L exactly when
    (H^T H)^-1 H^T v is integral.
    """
    base = points[0]
    differences = sp.Matrix([[a - b for a, b in zip(p, base, strict=True)] for p in points])
    hermite = hermite_normal_form(differences.T)
    left = [
        [Fraction(int(x.p), int(x.q)) for x in row]
        for row in ((hermite.T * hermite).inv() * hermite.T).tolist()
    ]

    def member(x: Sequence[int], k: int) -> bool:
        v = [a - k * b for a, b in zip(x, base, strict=True)]
        return all(sum(a * b for a, b in zip(row, v, strict=True)).denominator == 1 for row in left)

    return member


def brute_force(points: Points, k: int) -> dict[tuple[bool, str], list[tuple[int, ...]]]:
    """The lattice points of kP, and its interior points, in both lattices.

    Every integer point of the bounding box of kP is tested against the
    affine hull and the relative facets of polytope_data, and for the support
    lattice against L.
    """
    data = polytope_data(points)
    member = support_lattice(points)
    n = data.ambient_dimension
    box = [
        range(k * min(p[i] for p in points), k * max(p[i] for p in points) + 1) for i in range(n)
    ]
    found: dict[tuple[bool, str], list[tuple[int, ...]]] = {
        key: [] for key in itertools.product((False, True), ("support", "ambient"))
    }
    for x in itertools.product(*box):
        if any(k * h[0] + _exact.dot(h[1:], x) for h in data.affine_hull):
            continue
        slack = [k * f.offset - _exact.dot(f.normal, x) for f in data.relative_facets]
        if any(s < 0 for s in slack):
            continue
        interior = all(s > 0 for s in slack)
        in_support = member(x, k)
        for strict, lattice in found:
            if (interior or not strict) and (in_support or lattice == "ambient"):
                found[strict, lattice].append(tuple(x))
    return found


def random_polytope(rng: random.Random, d: int, spread: int = 2) -> Points:
    while True:
        count = rng.randint(d + 1, d + 4)
        pts = [tuple(rng.randint(-spread, spread) for _ in range(d)) for _ in range(count)]
        if _exact.affine_rank(pts) == d:
            return pts


def random_embedding(rng: random.Random, points: Points, extra: int) -> Points:
    """The points under a random injective affine map Z^d -> Z^(d + extra)."""
    d = len(points[0])
    while True:
        rows = [[rng.randint(-1, 2) for _ in range(d)] for _ in range(d + extra)]
        if _exact.rank(rows) == d:
            break
    shift = [rng.randint(-1, 1) for _ in rows]
    return [
        tuple(_exact.dot(row, p) + t for row, t in zip(rows, shift, strict=True)) for p in points
    ]


# --- textbook polytopes -----------------------------------------------------


class TestTextbook:
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_unit_simplex(self, d: int) -> None:
        expected = sp.expand(sp.expand_func(sp.binomial(K + d, d)))
        assert sp.expand(ehrhart_polynomial(simplex(d)).as_expr() - expected) == 0
        assert h_star_vector(simplex(d)) == (1,) + (0,) * d
        for k in (1, 2, 3):
            assert count_lattice_points(simplex(d), k) == math.comb(k + d, d)
            assert count_lattice_points(simplex(d), k, interior=True) == math.comb(k - 1, d)

    @pytest.mark.parametrize(
        ("d", "eulerian"), [(1, (1, 0)), (2, (1, 1, 0)), (3, (1, 4, 1, 0)), (4, (1, 11, 11, 1, 0))]
    )
    def test_unit_cube(self, d: int, eulerian: tuple[int, ...]) -> None:
        assert sp.expand(ehrhart_polynomial(cube(d)).as_expr() - (K + 1) ** d) == 0
        assert h_star_vector(cube(d)) == eulerian
        for k in (1, 2, 3):
            assert count_lattice_points(cube(d), k) == (k + 1) ** d
            assert count_lattice_points(cube(d), k, interior=True) == (k - 1) ** d

    @pytest.mark.parametrize("r", [1, 2, 3, 5, 12])
    def test_reeve_tetrahedra(self, r: int) -> None:
        # The vertices span Z^2 x rZ, so the textbook values hold in the ambient lattice; in
        # the support lattice the tetrahedron is a unimodular simplex.
        expected = sp.Rational(r, 6) * K**3 + K**2 + (2 - sp.Rational(r, 6)) * K + 1
        poly = ehrhart_polynomial(reeve(r), lattice="ambient")
        assert poly.gens == (K,)
        assert poly.get_domain() == sp.QQ
        assert sp.expand(poly.as_expr() - expected) == 0
        assert h_star_vector(reeve(r), lattice="ambient") == (1, 0, r - 1, 0)
        assert lattice_points(reeve(r), lattice="ambient") == sorted(reeve(r))
        assert count_lattice_points(reeve(r), interior=True, lattice="ambient") == 0
        simplex_polynomial = sp.expand_func(sp.binomial(K + 3, 3))
        assert sp.expand(ehrhart_polynomial(reeve(r)).as_expr() - simplex_polynomial) == 0

    def test_a_point(self) -> None:
        assert lattice_points([(1, 2)]) == [(1, 2)]
        assert lattice_points([(1, 2)], 3) == [(3, 6)]
        assert count_lattice_points([(1, 2), (1, 2)], 2, interior=True) == 1
        assert ehrhart_polynomial([(1, 2)]).as_expr() == 1
        assert h_star_vector([(1, 2)]) == (1,)

    def test_a_segment(self) -> None:
        points = [(0,), (3,), (1,)]
        assert lattice_points(points, 2) == [(i,) for i in range(7)]
        assert lattice_points(points, 2, interior=True) == [(i,) for i in range(1, 6)]
        assert sp.expand(ehrhart_polynomial(points).as_expr() - (3 * K + 1)) == 0
        assert h_star_vector(points) == (1, 2)
        # Without (1,) the support lattice is 3Z, and the segment has length 1 in it.
        assert lattice_points([(0,), (3,)], 2) == [(0,), (3,), (6,)]
        assert h_star_vector([(0,), (3,)]) == (1, 0)
        assert h_star_vector([(0,), (3,)], lattice="ambient") == (1, 2)


# --- the two lattices -------------------------------------------------------


class TestLattices:
    def test_a_triangle_whose_differences_span_an_index_4_lattice(self) -> None:
        points = [(0, 0), (2, 0), (0, 2)]
        support = ehrhart_polynomial(points).as_expr()
        ambient = ehrhart_polynomial(points, lattice="ambient").as_expr()
        assert sp.expand(support - sp.expand_func(sp.binomial(K + 2, 2))) == 0
        assert sp.expand(ambient - sp.expand_func(sp.binomial(2 * K + 2, 2))) == 0
        assert lattice_points(points) == sorted(points)
        assert lattice_points(points, lattice="ambient") == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (2, 0),
        ]
        assert h_star_vector(points) == (1, 0, 0)
        assert h_star_vector(points, lattice="ambient") == (1, 3, 0)

    def test_a_segment_in_three_dimensions(self) -> None:
        points = [(1, 0, 2), (3, 2, 4)]
        assert lattice_points(points) == [(1, 0, 2), (3, 2, 4)]
        assert lattice_points(points, 2) == [(2, 0, 4), (4, 2, 6), (6, 4, 8)]
        assert lattice_points(points, lattice="ambient") == [(1, 0, 2), (2, 1, 3), (3, 2, 4)]
        chart = invariant_chart(points, lattice="ambient")
        assert chart.basis == ((1, 1, 1),)
        assert [chart.to_ambient(c) for c in chart.coordinates] == points
        assert invariant_chart(points) == polytope_data(points).chart

    def test_dilates_of_a_triangle_off_the_origin(self) -> None:
        points = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        found = lattice_points(points, 2)
        assert found == sorted(x for x in itertools.product(range(3), repeat=3) if sum(x) == 2)
        assert lattice_points(points, 2, lattice="ambient") == found

    def test_the_ambient_chart_of_a_full_dimensional_polytope_is_a_translation(self) -> None:
        points = [(3, -1), (5, 1), (3, 3)]
        chart = invariant_chart(points, lattice="ambient")
        assert chart.basis == ((1, 0), (0, 1))
        assert chart.origin == (3, -1)
        assert abs(_exact.determinant(invariant_chart(points).basis)) == 8


# --- brute force ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_counts_match_brute_force_in_full_dimension(seed: int) -> None:
    rng = random.Random(seed)
    points = random_polytope(rng, 1 + seed % 3)
    for k in (1, 2, 3):
        for (interior, lattice), expected in brute_force(points, k).items():
            got = lattice_points(points, k, interior=interior, lattice=lattice)
            assert got == expected, (points, k, interior, lattice)
            count = count_lattice_points(points, k, interior=interior, lattice=lattice)
            assert count == len(expected)


@pytest.mark.parametrize("seed", range(16))
def test_counts_match_brute_force_below_full_dimension(seed: int) -> None:
    rng = random.Random(1000 + seed)
    points = random_embedding(rng, random_polytope(rng, 1 + seed % 2, spread=1), 1 + seed % 2)
    for k in (1, 2):
        for (interior, lattice), expected in brute_force(points, k).items():
            got = lattice_points(points, k, interior=interior, lattice=lattice)
            assert got == expected, (points, k, interior, lattice)


@pytest.mark.parametrize("seed", range(20))
def test_ehrhart_polynomial_matches_counts_beyond_the_interpolated_dilates(seed: int) -> None:
    rng = random.Random(2000 + seed)
    d = 1 + seed % 4
    points = random_polytope(rng, d)
    if seed % 5 == 0:
        points = random_embedding(rng, points, 1)
    for lattice in ("support", "ambient"):
        poly = ehrhart_polynomial(points, lattice=lattice)
        assert poly.degree() == d
        ks = range(1, d + 2)
        assert values(poly, ks) == [count_lattice_points(points, k, lattice=lattice) for k in ks]
        interior = [count_lattice_points(points, k, interior=True, lattice=lattice) for k in ks]
        assert values(poly, [-k for k in ks]) == [(-1) ** d * c for c in interior]
        h_star = h_star_vector(points, lattice=lattice)
        assert len(h_star) == d + 1
        assert all(h >= 0 for h in h_star)
        assert h_star[d] == interior[0]
        assert poly.LC() * math.factorial(d) == sum(h_star)


# --- Gorenstein index, reflexivity and the polar dual ------------------------


def cross_polytope(d: int) -> Points:
    return [tuple(s * int(i == j) for j in range(d)) for i in range(d) for s in (1, -1)]


def signs(d: int) -> Points:
    return list(itertools.product((-1, 1), repeat=d))


def palindromic_index(h_star: tuple[int, ...]) -> int | None:
    """d + 1 - s when h* is palindromic of degree s, otherwise None.

    P is Gorenstein of index r exactly when h* is palindromic of degree
    d + 1 - r, which gives an independent check of gorenstein_index.
    """
    s = max(i for i, h in enumerate(h_star) if h)
    top = h_star[: s + 1]
    return len(h_star) - s if top == top[::-1] else None


def interior_point(points: Points) -> tuple[int, ...]:
    (p,) = lattice_points(points, interior=True, lattice="ambient")
    return p


class TestGorenstein:
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_cross_polytope_and_cube(self, d: int) -> None:
        # With the origin among the points the support lattice is Z^d; the vertices of the cube
        # alone span 2Z^d, so the cube is reflexive in the ambient lattice only.
        octahedron = [(0,) * d, *cross_polytope(d)]
        assert gorenstein_index(octahedron) == 1
        assert polar_dual(octahedron) == tuple(signs(d))
        assert gorenstein_index(cross_polytope(d), lattice="ambient") == 1
        assert gorenstein_index(signs(d), lattice="ambient") == 1
        assert polar_dual(signs(d), lattice="ambient") == tuple(sorted(cross_polytope(d)))
        assert gorenstein_index(signs(d)) == 2
        assert polar_dual(signs(d)) is None

    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_unit_simplex(self, d: int) -> None:
        assert gorenstein_index(simplex(d)) == d + 1
        assert polar_dual(simplex(d)) is None

    @pytest.mark.parametrize(("r", "index"), [(1, 4), (2, 2), (3, None), (5, None)])
    def test_reeve_tetrahedra(self, r: int, index: int | None) -> None:
        assert gorenstein_index(reeve(r), lattice="ambient") == index
        assert gorenstein_index(reeve(r)) == 4

    def test_a_point_and_segments(self) -> None:
        assert gorenstein_index([(5, 5)]) == 1
        assert polar_dual([(5, 5)]) == ((),)
        assert gorenstein_index([(0,), (1,)]) == 2
        assert gorenstein_index([(4,), (6,), (5,)]) == 1
        assert polar_dual([(4,), (6,), (5,)]) == ((-1,), (1,))
        assert gorenstein_index([(0,), (3,)], lattice="ambient") is None

    @pytest.mark.parametrize("seed", range(20))
    def test_agrees_with_the_h_star_vector(self, seed: int) -> None:
        rng = random.Random(3000 + seed)
        points = random_polytope(rng, 2 + seed % 2, spread=1)
        for lattice in ("support", "ambient"):
            index = gorenstein_index(points, lattice=lattice)
            assert index == palindromic_index(h_star_vector(points, lattice=lattice))
            assert (polar_dual(points, lattice=lattice) is not None) == (index == 1)


def test_the_dual_of_the_dual_is_the_polytope() -> None:
    rng = random.Random(4)
    for d, spread in [(2, 2), (3, 1)]:
        box = list(itertools.product(range(-spread, spread + 1), repeat=d))
        found = 0
        while found < 12:
            points = [(0,) * d, *rng.sample(box, rng.randint(d + 1, d + 5))]
            if _exact.affine_rank(points) < d or gorenstein_index(points, lattice="ambient") != 1:
                continue
            found += 1
            dual = polar_dual(points, lattice="ambient")
            assert dual is not None
            assert gorenstein_index(dual, lattice="ambient") == 1
            p = interior_point(points)
            vertices = polytope_data(points).vertices
            expected = sorted(tuple(a - b for a, b in zip(v, p, strict=True)) for v in vertices)
            assert polar_dual(dual, lattice="ambient") == tuple(expected), points


# --- lattice width ----------------------------------------------------------


def width(points: Sequence[Sequence[int]], u: Sequence[int]) -> int:
    heights = [_exact.dot(u, p) for p in points]
    return max(heights) - min(heights)


def brute_force_width(points: Points) -> int:
    """The least width over every direction in a box that holds every candidate.

    With w0 the least width over the coordinate directions and M the
    differences from v_0 of affinely independent points v_1, ..., v_d, a
    direction u of width at most w0 has |M u| <= w0 entrywise, so |u_i| is
    at most w0 times the largest absolute row sum of M^-1. The box is the
    smallest such bound over all choices of the points.
    """
    d = len(points[0])
    w0 = min(width(points, [int(i == j) for j in range(d)]) for i in range(d))
    radius = None
    for chosen in itertools.combinations(sorted(set(points)), d + 1):
        m = sp.Matrix([[a - b for a, b in zip(q, chosen[0], strict=True)] for q in chosen[1:]])
        if m.det() == 0:
            continue
        inverse = m.inv()
        bound = int(w0 * max(sum(abs(x) for x in inverse.row(i)) for i in range(d)))
        radius = bound if radius is None else min(radius, bound)
    assert radius is not None
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    box = np.stack(np.meshgrid(*[axis] * d, indexing="ij"), axis=-1).reshape(-1, d)
    box = box[np.any(box != 0, axis=1)]
    heights = box @ np.array(points, dtype=np.int64).T
    return int((heights.max(axis=1) - heights.min(axis=1)).min())


class TestWidth:
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_textbook(self, d: int) -> None:
        assert lattice_width(simplex(d)) == (1, (1,) + (0,) * (d - 1))
        assert lattice_width(cube(d)) == (1, (1,) + (0,) * (d - 1))
        assert lattice_width([(0,) * d, *cross_polytope(d)]) == (2, (1,) + (0,) * (d - 1))
        assert lattice_width(signs(d), lattice="ambient")[0] == 2
        assert lattice_width(signs(d))[0] == 1

    def test_a_point_and_a_segment(self) -> None:
        assert lattice_width([(3, 1)]) == (0, ())
        assert lattice_width([(0,), (5,), (2,)]) == (5, (1,))

    def test_a_direction_off_the_axes(self) -> None:
        # A strip between y = x - 1 and y = x: width 5 and 4 along the axes, 1 along (1, -1).
        points = [(0, 0), (4, 4), (1, 0), (5, 4), (1, 1)]
        assert lattice_width(points) == (1, (1, -1))
        assert brute_force_width(points) == 1

    def test_facet_normals_are_tried_before_the_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Between the facets x + y + z >= 2 and x + y + z <= 3, as G = U + F lies for two
        # loops; every coordinate direction has width 3.
        points = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (3, 0, 0), (0, 3, 0), (0, 0, 3)]

        polytope = li._prepare(points, "support")

        def no_search(*args: object, **kwargs: object) -> None:
            raise AssertionError("the width search ran")

        monkeypatch.setattr(li, "_body", no_search)
        assert li._width(polytope) == (1, (1, 1, 1))

    @pytest.mark.parametrize("seed", range(30))
    def test_brute_force(self, seed: int) -> None:
        rng = random.Random(5000 + seed)
        d = 1 + seed % 3
        points = random_polytope(rng, d, spread=3 if d < 3 else 2)
        if seed % 4 == 0:
            points = random_embedding(rng, points, 1)
        for lattice in ("support", "ambient"):
            found, direction = lattice_width(points, lattice=lattice)
            chart = invariant_chart(points, lattice=lattice)
            coordinates = list(chart.coordinates)
            assert found == brute_force_width(coordinates), (points, lattice)
            assert width(coordinates, direction) == found
            assert next(x for x in direction if x) > 0


# --- arguments --------------------------------------------------------------


class TestArguments:
    def test_no_points(self) -> None:
        with pytest.raises(ValidationError, match="at least one point"):
            count_lattice_points([])

    def test_unknown_lattice(self) -> None:
        with pytest.raises(ValidationError, match="'support' or 'ambient'"):
            ehrhart_polynomial(simplex(2), lattice="saturated")  # type: ignore[arg-type]

    @pytest.mark.parametrize("k", [0, -1, 1.5, True])
    def test_dilation_factor(self, k: object) -> None:
        with pytest.raises(ValidationError, match="positive integer"):
            lattice_points(simplex(2), k)  # type: ignore[arg-type]

    def test_integral_floats_and_numpy_integers_are_accepted(self) -> None:
        assert count_lattice_points(np.array(simplex(2)), np.int64(2)) == 6
        assert count_lattice_points(simplex(2), 2.0) == 6  # type: ignore[arg-type]
