from __future__ import annotations

import random

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, _exact
from feynkit.core.exceptions import ValidationError
from feynkit.polytope import (
    Facet,
    LatticeChart,
    PolytopeData,
    faces,
    lattice_chart,
    lattice_coordinates,
    polytope_data,
)


class TestMasslessBubble:
    """Newt(G) for G = u_1 + u_2 - s u_1 u_2 / mu^2: the triangle (1,0), (0,1), (1,1)."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls) -> PolytopeData:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        return polytope_data(fi.newton_polytope.points)

    def test_dimensions_and_vertices(self, data: PolytopeData) -> None:
        assert data.ambient_dimension == 2
        assert data.dimension == 2
        assert data.is_full_dimensional
        assert set(data.vertices) == {(1, 0), (0, 1), (1, 1)}

    def test_facet_inequalities(self, data: PolytopeData) -> None:
        # x <= 1, y <= 1, -x - y <= -1, each with a primitive outward normal.
        got = {(f.normal, f.offset) for f in data.facets}
        assert got == {((1, 0), 1), ((0, 1), 1), ((-1, -1), -1)}

    def test_every_point_satisfies_every_facet(self, data: PolytopeData) -> None:
        for f in data.facets:
            for p in data.points:
                assert sum(m * x for m, x in zip(f.normal, p, strict=True)) <= f.offset

    def test_f_vector_and_volume(self, data: PolytopeData) -> None:
        assert data.f_vector == (3, 3, 1)
        assert data.normalized_volume == 1

    def test_facet_points_lie_on_their_hyperplane(self, data: PolytopeData) -> None:
        for f in data.facets:
            assert isinstance(f, Facet)
            for i in f.point_indices:
                p = data.points[i]
                assert sum(m * x for m, x in zip(f.normal, p, strict=True)) == f.offset


class TestOneMassTriangle:
    def test_counts_match_landau_faces(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        data = polytope_data(fi.newton_polytope.points)
        assert data.dimension == 3
        assert len(data.faces) == 27  # the count the Landau tests report
        assert len(data.facets) == data.f_vector[2]
        assert data.normalized_volume == 5  # A = 4 x 7, one-mass triangle


class TestDegenerate:
    def test_segment_is_not_full_dimensional(self) -> None:
        data = polytope_data([(0, 0), (1, 1), (2, 2)])
        assert data.dimension == 1
        assert not data.is_full_dimensional
        assert data.facets == ()
        assert set(data.vertex_indices) == {0, 2}
        assert data.normalized_volume == 2

    def test_moved_helpers_still_work(self) -> None:
        pts = np.array([[0, 0], [1, 0], [0, 1]])
        assert (2, (0, 1, 2)) in faces(pts)
        assert lattice_coordinates(np.array([[0, 0], [2, 2], [4, 4]])) == [(0,), (1,), (2,)]


def _sympy_lattice_coordinates(pts: np.ndarray) -> list[tuple[int, ...]]:
    """The SymPy construction lattice_coordinates used before the exact core."""
    if len(pts) == 1:
        return [()]
    from sympy.matrices.normalforms import hermite_normal_form

    diffs = sp.Matrix([[int(x) for x in row] for row in (pts[1:] - pts[0])])
    hnf = hermite_normal_form(diffs.T).T
    basis = sp.Matrix([row for row in hnf.tolist() if any(x != 0 for x in row)])
    if basis.rows == 0:
        return [() for _ in pts]
    coords = []
    for row in pts - pts[0]:
        target = sp.Matrix([[int(x) for x in row]])
        sol = (
            basis.T.solve_least_squares(target.T)
            if basis.rows < basis.cols
            else basis.T.solve(target.T)
        )
        coords.append(tuple(int(sp.nsimplify(x)) for x in sol))
    mins = [min(c[i] for c in coords) for i in range(basis.rows)]
    return [tuple(c[i] - mins[i] for i in range(basis.rows)) for c in coords]


def embedded_configurations(seed: int, count: int) -> list[list[tuple[int, ...]]]:
    """Distinct random points of dimension 1 to 4, mapped into Z^(d + k), k <= 2, injectively."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        d, extra = rng.randint(1, 4), rng.randint(0, 2)
        size = rng.randint(d + 1, d + 6)
        pts = sorted({tuple(rng.randint(-2, 2) for _ in range(d)) for _ in range(size)})
        while True:
            m = [[rng.randint(-3, 3) for _ in range(d)] for _ in range(d + extra)]
            if sp.Matrix(m).rank() == d:
                break
        shift = [rng.randint(-3, 3) for _ in range(d + extra)]
        out.append(
            [tuple(s + _exact.dot(r, p) for r, s in zip(m, shift, strict=True)) for p in pts]
        )
    return out


class TestLatticeChart:
    def test_examples_from_the_design_note(self) -> None:
        chart = lattice_chart([(0, 0, 0), (2, 0, 0), (0, 2, 0)])
        assert chart == LatticeChart(
            origin=(0, 0, 0), basis=((2, 0, 0), (0, 2, 0)), coordinates=((0, 0), (1, 0), (0, 1))
        )
        plane = lattice_chart([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, -1)])
        assert plane.basis == ((-1, 1, 0), (-1, 0, 1))
        assert plane.coordinates == ((0, 1), (1, 1), (0, 2), (1, 0))
        assert plane.origin == (2, 0, -1)

    def test_points_are_origin_plus_basis_times_coordinates(self) -> None:
        for pts in embedded_configurations(7, 100):
            chart = lattice_chart(pts)
            assert [chart.to_ambient(c) for c in chart.coordinates] == pts
            for t in range(len(chart.basis)):
                assert min(c[t] for c in chart.coordinates) == 0

    def test_coordinates_match_the_sympy_construction(self) -> None:
        for pts in embedded_configurations(8, 200):
            arr = np.array(pts)
            assert lattice_coordinates(arr) == _sympy_lattice_coordinates(arr)

    def test_coordinates_generate_the_lattice(self) -> None:
        for pts in embedded_configurations(9, 100):
            chart = lattice_chart(pts)
            d = len(chart.basis)
            base = chart.coordinates[0]
            diffs = [[a - b for a, b in zip(c, base, strict=True)] for c in chart.coordinates[1:]]
            assert _exact.smith_invariants(diffs, d) == [1] * d

    def test_single_point_and_bad_input(self) -> None:
        chart = lattice_chart([(3, 4)])
        assert chart == LatticeChart(origin=(3, 4), basis=(), coordinates=((),))
        assert lattice_coordinates(np.array([[3, 4], [3, 4]])) == [(), ()]
        with pytest.raises(ValidationError, match="at least one point"):
            lattice_chart([])
        with pytest.raises(ValidationError, match="length"):
            chart.to_ambient((1,))
