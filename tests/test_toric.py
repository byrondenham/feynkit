"""Tests for feynkit.toric: normal cones, smooth orbit charts and smooth subdivisions.

Every example is a hand example: small polytopes whose normal cones and local
equations are written out below.
"""

from __future__ import annotations

import itertools
import random
import time
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import _exact
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.polytope import polytope_data
from feynkit.toric import (
    NormalCone,
    OrbitChart,
    normal_cone,
    orbit_chart,
    orbit_cones,
    smooth_subdivision,
)

SIMPLEX = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
CUBE = list(itertools.product((0, 1), repeat=3))
# The octahedron with its centre, so that the points generate Z^3 and its vertex cones are not
# smooth: the cone at (1, 0, 0) has the four rays (-1, +-1, +-1) over a square.
OCTAHEDRON = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0)]
# conv((0,0), (2,1), (1,2)) with its interior point (1,1): every vertex cone has multiplicity 3.
TRIANGLE_Z3 = [(0, 0), (2, 1), (1, 2), (1, 1)]
UNIT_TRIANGLE = [(0, 0), (1, 0), (0, 1)]


def _coordinates(data, index):
    return data.chart.coordinates[index]


def _in_normal_cone(data, face, u):
    """Whether u is an inner normal at the face: u . c is minimal over the polytope, on the face."""
    values = [sum(a * b for a, b in zip(u, c, strict=True)) for c in data.chart.coordinates]
    return all(values[i] == min(values) for i in face)


def _coefficients(cone_rays, u):
    """Coefficients of u over independent rays (exact), or None when u is outside their span."""
    from feynkit.toric import _coefficients as solve

    return solve(cone_rays, u)


def _in_piece(piece, u):
    c = _coefficients(piece.rays, u)
    return c is not None and all(x >= 0 for x in c), c


def _vertex_faces(data):
    return [idx for dim, idx in data.faces if dim == 0]


# --- item 1: subdivisions ----------------------------------------------------


def _fan_problems(data, face, pieces, samples=40):
    """Exact checks that the pieces form a fan with union the normal cone of the face.

    Every ray lies in the cone; each facet of a piece (a set of its rays) lies in exactly two
    pieces, on opposite sides, when its relative interior meets the interior of the cone, and in
    exactly one piece otherwise; a generic rational point of the cone lies in exactly one piece.
    Returns the list of problems found.
    """
    from feynkit.toric import _coefficients as solve
    from feynkit.toric import _minimal_set, _saturated_basis

    cone = normal_cone(data, face)
    problems = []
    k = len(pieces[0].rays)
    lattice = _saturated_basis(cone.rays, cone.lattice_dimension)

    def coordinates(u):
        return solve(lattice, u)

    for piece in pieces:
        for u in piece.rays:
            if not _in_normal_cone(data, face, u):
                problems.append(("ray outside the cone", u))
    sides = {}
    for piece in pieces:
        for i, apex in enumerate(piece.rays):
            facet = tuple(r for j, r in enumerate(piece.rays) if j != i)
            rows = [coordinates(u) for u in (*facet, apex)]
            det = sp.Matrix(
                [[sp.Rational(x.numerator, x.denominator) for x in row] for row in rows]
            ).det()
            sides.setdefault(facet, []).append(det)
    for facet, dets in sides.items():
        total = [sum(u[j] for u in facet) for j in range(cone.lattice_dimension)]
        interior = set(face) == _minimal_set(data, total)
        if interior:
            if len(dets) != 2 or dets[0] * dets[1] >= 0:
                problems.append(("interior facet", facet, dets))
        elif len(dets) != 1:
            problems.append(("boundary facet", facet, dets))
    rng = random.Random(1)
    for _ in range(samples):
        weights = [Fraction(rng.randint(1, 50), rng.randint(1, 50)) for _ in cone.rays]
        point = [
            sum(w * r[j] for w, r in zip(weights, cone.rays, strict=True))
            for j in range(cone.lattice_dimension)
        ]
        hits = []
        for piece in pieces:
            c = solve(piece.rays, point)
            if c is not None and all(x >= 0 for x in c):
                hits.append(c)
        if len(hits) != 1 or any(x == 0 for x in hits[0]):
            problems.append(("point covered", point, len(hits)))
    assert k == len(pieces[0].rays)
    return problems


def _check_subdivision(data, face, seed=0):
    cone = normal_cone(data, face)
    pieces = smooth_subdivision(cone, seed=seed)
    assert all(p.smooth and p.simplicial and p.multiplicity == 1 for p in pieces)
    assert all(len(p.rays) == len(pieces[0].rays) for p in pieces)
    assert _fan_problems(data, face, pieces) == []
    return pieces


@pytest.mark.parametrize("points", [SIMPLEX, CUBE])
def test_smooth_vertex_cones_are_returned_unchanged(points):
    data = polytope_data(points)
    for face in _vertex_faces(data):
        cone = normal_cone(data, face)
        assert cone.smooth and cone.simplicial and len(cone.rays) == 3
        assert smooth_subdivision(cone, seed=3) == (cone,)


def test_octahedron_vertex_cone_is_split_into_unimodular_cones():
    data = polytope_data(OCTAHEDRON)
    face = (0,)
    cone = normal_cone(data, face)
    assert not cone.simplicial and not cone.smooth and len(cone.rays) == 4
    pieces = _check_subdivision(data, face)
    assert len(pieces) >= 2


def test_octahedron_pieces_sum_to_the_volume_of_the_cone():
    # With l(u) = -u_1, a full-dimensional simplicial cone cuts {l = 1} in a simplex of volume
    # |det| / (3! prod l(u_i)). The original cone is two triangles on the square with corners
    # (-1, +-1, +-1), each of determinant 4 and l = 1 on every ray.
    reference = Fraction(4, 1) + Fraction(4, 1)
    data = polytope_data(OCTAHEDRON)
    cone = normal_cone(data, (0,))
    assert {u[0] for u in cone.rays} == {-1}
    for seed in range(6):
        total = Fraction(0)
        for p in smooth_subdivision(cone, seed=seed):
            det = abs(_exact.determinant([list(r) for r in p.rays]))
            assert det == 1
            weight = 1
            for r in p.rays:
                weight *= -r[0]
            total += Fraction(det, weight)
        assert total == reference


def test_two_seeds_give_different_subdivisions():
    data = polytope_data(OCTAHEDRON)
    cone = normal_cone(data, (0,))
    results = {tuple(p.rays for p in smooth_subdivision(cone, seed=s)) for s in range(8)}
    assert len(results) >= 2
    # the same seed repeats
    assert smooth_subdivision(cone, seed=5) == smooth_subdivision(cone, seed=5)


def test_stellar_step_on_a_simplicial_cone_of_multiplicity_three():
    data = polytope_data(TRIANGLE_Z3)
    for face in _vertex_faces(data):
        cone = normal_cone(data, face)
        assert cone.simplicial and not cone.smooth and cone.multiplicity == 3
        _check_subdivision(data, face)


def test_octahedron_every_vertex_and_a_four_dimensional_cross_polytope():
    data = polytope_data(OCTAHEDRON)
    for face in _vertex_faces(data):
        for p in smooth_subdivision(normal_cone(data, face), seed=1):
            assert p.smooth
    cross = [tuple(s * int(i == j) for j in range(4)) for i in range(4) for s in (1, -1)]
    cross.append((0, 0, 0, 0))
    data4 = polytope_data(cross)
    cone = normal_cone(data4, (0,))
    assert len(cone.rays) == 8 and not cone.simplicial
    pieces = _check_subdivision(data4, (0,))
    assert all(p.smooth for p in pieces)
    # an edge: its cone spans only a three-dimensional subspace of Z^4
    edge = next(idx for dim, idx in data4.faces if dim == 1)
    edge_cone = normal_cone(data4, edge)
    assert len(edge_cone.rays) > 3
    edge_pieces = _check_subdivision(data4, edge)
    assert all(len(p.rays) == 3 and p.smooth for p in edge_pieces)


# --- item 2: normal cones ----------------------------------------------------


def test_normal_cones_of_the_unit_triangle_for_every_dimension():
    data = polytope_data(UNIT_TRIANGLE)
    assert data.chart.coordinates == ((0, 0), (1, 0), (0, 1))
    by_face = {idx: normal_cone(data, idx) for _d, idx in data.faces}
    top = by_face[(0, 1, 2)]
    assert top.rays == () and top.simplicial and top.smooth
    # edges: one ray each, the inner normals (0, 1), (1, 0) and (-1, -1)
    assert by_face[(0, 1)].rays == ((0, 1),)
    assert by_face[(0, 2)].rays == ((1, 0),)
    assert by_face[(1, 2)].rays == ((-1, -1),)
    # vertices: two rays, in the order of the facets
    assert set(by_face[(0,)].rays) == {(0, 1), (1, 0)}
    assert set(by_face[(1,)].rays) == {(0, 1), (-1, -1)}
    assert set(by_face[(2,)].rays) == {(1, 0), (-1, -1)}
    assert all(c.simplicial and c.smooth and c.multiplicity == 1 for c in by_face.values())
    assert normal_cone(data, [2, 1]) == by_face[(1, 2)]


def test_simplicial_non_smooth_cone():
    data = polytope_data(TRIANGLE_Z3)
    assert data.chart.coordinates == ((0, 0), (2, 1), (1, 2), (1, 1))
    cone = normal_cone(data, (0,))
    assert set(cone.rays) == {(-1, 2), (2, -1)}
    assert cone.simplicial and not cone.smooth and cone.multiplicity == 3
    # its edges are smooth rays
    assert normal_cone(data, (0, 1)).smooth
    assert normal_cone(data, (0, 1, 2, 3)).rays == ()


def test_non_simplicial_cone_flags():
    data = polytope_data(OCTAHEDRON)
    cone = normal_cone(data, (0,))
    assert not cone.simplicial and not cone.smooth and cone.multiplicity is None
    # an edge of the octahedron lies on two facets: two rays
    edge = next(idx for dim, idx in data.faces if dim == 1)
    assert len(normal_cone(data, edge).rays) == 2
    # the cube's top face, facets and vertices
    cube = polytope_data(CUBE)
    assert [len(normal_cone(cube, idx).rays) for dim, idx in cube.faces if dim == 3] == [0]
    assert {len(normal_cone(cube, idx).rays) for dim, idx in cube.faces if dim == 2} == {1}
    assert {len(normal_cone(cube, idx).rays) for dim, idx in cube.faces if dim == 1} == {2}
    assert {len(normal_cone(cube, idx).rays) for dim, idx in cube.faces if dim == 0} == {3}


def test_normal_cone_of_a_lower_dimensional_polytope_uses_its_lattice_chart():
    data = polytope_data([(0, 0, 0), (1, 1, 0), (2, 2, 0)])
    assert data.dimension == 1 and len(data.chart.basis) == 1
    assert normal_cone(data, (0,)).rays == ((1,),)
    assert normal_cone(data, (2,)).rays == ((-1,),)
    assert normal_cone(data, (0, 1, 2)).rays == ()


def test_normal_cone_validation():
    data = polytope_data(UNIT_TRIANGLE)
    with pytest.raises(ValidationError):
        normal_cone(data, (0, 3))
    with pytest.raises(ValidationError):
        normal_cone(data, ())
    with pytest.raises(ValidationError):
        normal_cone(data, "ab")
    with pytest.raises(ValidationError):
        normal_cone(UNIT_TRIANGLE, (0,))
    with pytest.raises(ValidationError):
        smooth_subdivision("cone")
    with pytest.raises(ValidationError):
        smooth_subdivision(normal_cone(polytope_data(OCTAHEDRON), (0,)), seed=1.5)


# --- item 3: orbit charts ----------------------------------------------------


def test_chart_at_a_vertex_of_the_unit_triangle():
    # G = a + b x + c y at the vertex (0, 0): y_1 = x, y_2 = y up to the order of the rays.
    data = polytope_data(UNIT_TRIANGLE)
    chart = orbit_chart(data, (0,))
    assert isinstance(chart, OrbitChart)
    assert chart.torus_basis == () and chart.vertex == (0, 0)
    assert chart.normal_dimension == 2 and chart.torus_dimension == 0
    g = chart.local_equation({(0, 0): "a", (1, 0): "b", (0, 1): "c"})
    index = {ray: i for i, ray in enumerate(chart.rays)}
    # the ray (1, 0) is the normal of the edge x = 0, so y_(1,0) is the coordinate x
    e1, e2 = index[(1, 0)], index[(0, 1)]
    assert g == {
        tuple(1 if i == e1 else 0 for i in range(2)): "b",
        tuple(1 if i == e2 else 0 for i in range(2)): "c",
        (0, 0): "a",
    }


def test_chart_at_another_vertex_shifts_by_the_vertex():
    # At (1, 0): x^(-v) G = a/x + b + c y/x, with rays (-1, -1) and (0, 1) giving
    # y_(-1,-1) = 1/x ... evaluated as exponents <u, alpha - v>.
    data = polytope_data(UNIT_TRIANGLE)
    chart = orbit_chart(data, (1,))
    assert chart.vertex == (1, 0)
    g = chart.local_equation({(0, 0): 2, (1, 0): 3, (0, 1): 5})
    i, j = chart.rays.index((-1, -1)), chart.rays.index((0, 1))
    point = [0, 0]
    point[i] = 1
    other = [0, 0]
    other[j] = 1
    assert g == {tuple(point): 2, (0, 0): 3, tuple(other): 5}


def test_chart_on_an_edge_has_one_torus_coordinate():
    # Edge {(0, 0), (1, 0)}: ray (0, 1), y = y-coordinate; the torus coordinate is x^(+-1).
    data = polytope_data(UNIT_TRIANGLE)
    chart = orbit_chart(data, (0, 1))
    assert chart.rays == ((0, 1),) and len(chart.torus_basis) == 1
    w = chart.torus_basis[0]
    assert w[0] in (1, -1)
    assert abs(_exact.determinant([list(w), list(chart.rays[0])])) == 1
    g = chart.local_equation({(0, 0): 1, (1, 0): 4, (0, 1): 7})
    # g = 1 + 4 t^(w_1) + 7 y t^(w_2), t^(w . alpha) being the torus part of x^alpha
    assert g == {(0, 0): 1, (w[0], 0): 4, (w[1], 1): 7}
    # the torus polynomial of the edge is the y = 0 part
    assert {k: c for k, c in g.items() if k[1] == 0} == {(0, 0): 1, (w[0], 0): 4}


def test_chart_of_the_top_face_is_the_torus():
    data = polytope_data(UNIT_TRIANGLE)
    chart = orbit_chart(data, (0, 1, 2))
    assert chart.rays == () and chart.normal_dimension == 0 and len(chart.torus_basis) == 2
    g = chart.local_equation({(0, 0): 1, (1, 0): 2, (0, 1): 3})
    assert sorted(g.values()) == [1, 2, 3] and all(len(k) == 2 for k in g)
    # a basis: the exponent map is invertible
    t = chart.torus_basis
    assert abs(_exact.determinant([list(r) for r in t])) == 1


def test_chart_on_a_cube_edge_with_a_hand_equation():
    # Cube [0,1]^3, edge from (0,0,0) to (1,0,0): rays (0,1,0), (0,0,1) (inner normals), torus
    # coordinate x. G = 1 + x + y + x y z: g = 1 + t^a + y1 + t^a y1 y2, in the order of the rays.
    data = polytope_data(CUBE)
    edge = tuple(i for i, c in enumerate(data.chart.coordinates) if c in {(0, 0, 0), (1, 0, 0)})
    assert len(edge) == 2
    chart = orbit_chart(data, edge)
    assert set(chart.rays) == {(0, 1, 0), (0, 0, 1)} and len(chart.torus_basis) == 1
    w = chart.torus_basis[0]
    assert w[0] in (1, -1) and abs(_exact.determinant([list(w), *map(list, chart.rays)])) == 1
    a, b = chart.rays.index((0, 1, 0)), chart.rays.index((0, 0, 1))

    def key(t, ey, ez):
        e = [0, 0]
        e[a], e[b] = ey, ez
        return (t, *e)

    g = chart.local_equation({(0, 0, 0): 1, (1, 0, 0): 1, (0, 1, 0): 1, (1, 1, 1): 1})
    # the point (1, 1, 1) is v + (1,1,1): the torus exponent is w . (1, 1, 1) = w_1 + w_2 + w_3
    assert g == {
        key(0, 0, 0): 1,
        key(w[0], 0, 0): 1,
        key(w[1], 1, 0): 1,
        key(sum(w), 1, 1): 1,
    }


def test_local_equation_sums_terms_and_drops_zeros():
    data = polytope_data(UNIT_TRIANGLE)
    chart = orbit_chart(data, (0,))
    g = chart.local_equation({(0, 0): 0, (1, 0): Fraction(1, 2)})
    assert list(g.values()) == [Fraction(1, 2)]


def test_orbit_chart_rejects_non_smooth_cones_and_bad_input():
    data = polytope_data(TRIANGLE_Z3)
    with pytest.raises(ValidationError, match="not smooth"):
        orbit_chart(data, (0,))
    with pytest.raises(ValidationError, match="not smooth"):
        orbit_chart(polytope_data(OCTAHEDRON), (0,))
    with pytest.raises(ValidationError):
        orbit_chart(data, (0, 2, 3))
    with pytest.raises(ValidationError):
        orbit_chart("polytope", (0,))
    # a smooth edge of the same polygon is fine
    assert orbit_chart(data, (0, 1)).rays == ((-1, 2),)
    chart = orbit_chart(polytope_data(UNIT_TRIANGLE), (0,))
    with pytest.raises(ValidationError):
        chart.local_equation({(0, 0, 0): 1})
    with pytest.raises(ValidationError):
        chart.local_equation({(0.5, 0): 1})
    with pytest.raises(ValidationError, match="negative"):
        chart.local_equation({(-1, 0): 1})
    with pytest.raises(ValidationError):
        chart.local_equation([(0, 0)])


def test_cone_dataclass_is_frozen():
    cone = normal_cone(polytope_data(UNIT_TRIANGLE), (0,))
    assert isinstance(cone, NormalCone)
    with pytest.raises(AttributeError):
        cone.smooth = False  # type: ignore[misc]


# --- fan checks, large cases, charts of pieces --------------------------------


def _mutant_pieces(cone, seed):
    """A wrong subdivision: star-subdivide only the cone of largest multiplicity, never the fan."""
    from feynkit.toric import _make_cone, _multiplicity, _parallelepiped_points, _triangulate

    d = cone.lattice_dimension
    rng = random.Random(seed)
    cones = _triangulate(cone.rays, rng)
    while True:
        worst = max(cones, key=lambda c: (_multiplicity(c, d), c))
        if _multiplicity(worst, d) == 1:
            break
        point, lam = rng.choice(_parallelepiped_points(worst, d))
        cones.remove(worst)
        cones.extend(
            tuple(r for r in worst if r != u) + (point,)
            for u, f in zip(worst, lam, strict=True)
            if f != 0
        )
    return tuple(_make_cone(cone.face, c, d) for c in sorted(tuple(sorted(c)) for c in cones))


def test_fan_check_rejects_a_subdivision_that_only_splits_the_worst_cone():
    cross = [tuple(s * int(i == j) for j in range(4)) for i in range(4) for s in (1, -1)]
    data = polytope_data([*cross, (0, 0, 0, 0)])
    cone = normal_cone(data, (0,))
    assert _fan_problems(data, (0,), smooth_subdivision(cone, seed=0)) == []
    bad = [s for s in range(6) if _fan_problems(data, (0,), _mutant_pieces(cone, s))]
    assert len(bad) >= 5


def test_fan_check_rejects_overlaps_and_gaps():
    data = polytope_data(OCTAHEDRON)
    cone = normal_cone(data, (0,))
    pieces = smooth_subdivision(cone, seed=0)
    assert _fan_problems(data, (0,), pieces[:-1])
    assert _fan_problems(data, (0,), (*pieces, pieces[0]))


# Three of the six points are far from the others; the vertex cone at (1,) is simplicial of
# multiplicity 128 and the one at (0,) has five rays.
LARGE = [(0, 3, 1), (-1, -3, 3), (1, -3, -3), (2, 3, 0), (2, 3, 3), (2, 2, -3)]


def test_large_multiplicity_cones_are_subdivided_quickly():
    data = polytope_data(LARGE)
    start = time.perf_counter()
    for dim, face in data.faces:
        cone = normal_cone(data, face)
        for seed in (0, 1):
            pieces = smooth_subdivision(cone, seed=seed)
            assert all(p.smooth for p in pieces)
            assert len(pieces) <= 400
            if dim == 0:
                assert _fan_problems(data, face, pieces, samples=10) == []
    assert time.perf_counter() - start < 30
    assert {normal_cone(data, f).multiplicity for d_, f in data.faces if d_ == 0} >= {128}


def test_cone_cap_raises_undecided():
    data = polytope_data(LARGE)
    cone = normal_cone(data, (1,))
    assert cone.multiplicity == 128
    with pytest.raises(ComputationError, match="^undecided: "):
        smooth_subdivision(cone, seed=0, max_cones=2)
    with pytest.raises(ValidationError):
        smooth_subdivision(cone, max_cones=1.5)


def test_chart_of_a_piece_shares_the_torus_coordinates():
    data = polytope_data(TRIANGLE_Z3)
    face = (0,)
    pieces = smooth_subdivision(normal_cone(data, face), seed=0)
    charts = [orbit_chart(data, face, cone=p) for p in pieces]
    assert all(c.rays == p.rays for c, p in zip(charts, pieces, strict=True))
    assert all(c.torus_basis == () and c.vertex == (0, 0) for c in charts)
    # the edge cone of the octahedron: a non-smooth face, two torus directions shared by pieces
    four = [tuple(s * int(i == j) for j in range(4)) for i in range(4) for s in (1, -1)]
    d4 = polytope_data([*four, (0, 0, 0, 0)])
    edge = next(idx for dim, idx in d4.faces if dim == 1)
    edge_pieces = smooth_subdivision(normal_cone(d4, edge), seed=2)
    bases = {orbit_chart(d4, edge, cone=p).torus_basis for p in edge_pieces}
    assert len(bases) == 1 and len(next(iter(bases))) == 1
    for p in edge_pieces:
        chart = orbit_chart(d4, edge, cone=p)
        rows = [list(w) for w in (*chart.torus_basis, *chart.rays)]
        assert abs(_exact.determinant(rows)) == 1
    # a piece of the polytope's own cone on a smooth face: same torus basis as the default
    smooth_face = (0, 1)
    base = orbit_chart(data, smooth_face)
    assert orbit_chart(data, smooth_face, cone=normal_cone(data, smooth_face)) == base


def test_chart_of_a_piece_rejects_bad_cones():
    data = polytope_data(TRIANGLE_Z3)
    face = (0,)
    pieces = smooth_subdivision(normal_cone(data, face), seed=0)
    with pytest.raises(ValidationError):
        orbit_chart(data, (1,), cone=pieces[0])
    with pytest.raises(ValidationError, match="not smooth"):
        orbit_chart(data, face, cone=normal_cone(data, face))
    outside = NormalCone(face, ((1, 0), (0, -1)), True, True, 2)
    with pytest.raises(ValidationError, match="normal cone"):
        orbit_chart(data, face, cone=outside)
    short = NormalCone(face, ((-1, 2),), True, True, 2)
    with pytest.raises(ValidationError):
        orbit_chart(data, face, cone=short)
    with pytest.raises(ValidationError):
        orbit_chart(data, face, cone="piece")


def test_orbit_cones_of_a_subdivision():
    data = polytope_data(TRIANGLE_Z3)
    face = (0,)
    pieces = smooth_subdivision(normal_cone(data, face), seed=0)
    cones = orbit_cones(data, pieces)
    # the cones with relative interior in that of the 2-dimensional cone: the new interior
    # rays and the 2-dimensional pieces; the two boundary rays are on the boundary
    original = set(normal_cone(data, face).rays)
    assert all(len(set(c) & original) < len(c) or len(c) == 2 for c in cones)
    assert len(cones) == len(set(cones))
    assert sum(len(c) == 2 for c in cones) == len(pieces)
    assert all(len(c) in (1, 2) for c in cones)
    # the polytope itself: the cone {0} only
    top = tuple(range(len(data.points)))
    assert orbit_cones(data, [normal_cone(data, top)]) == ((),)
    with pytest.raises(ValidationError):
        orbit_cones(data, [])
    with pytest.raises(ValidationError):
        orbit_cones(data, [*pieces, normal_cone(data, (1,))])
