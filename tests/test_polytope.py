from __future__ import annotations

import random
import sys
import types

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, _exact, polytope
from feynkit.core.exceptions import ComputationError, ValidationError
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
        with pytest.raises(ValidationError, match="point 1 has 3 coordinates, point 0 has 2"):
            lattice_chart([(0, 0), (1, 2, 3)])

    def test_repeated_points_in_a_lower_dimensional_chart(self) -> None:
        # Point 2 repeats the first point and point 4 repeats point 1.
        pts = [(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, -1)]
        chart = lattice_chart(pts)
        assert chart == LatticeChart(
            origin=(2, 0, -1),
            basis=((-1, 1, 0), (-1, 0, 1)),
            coordinates=((0, 1), (1, 1), (0, 1), (0, 2), (1, 1), (1, 0)),
        )
        assert [chart.to_ambient(c) for c in chart.coordinates] == pts
        doubled = lattice_chart([(0, 0, 0), (2, 0, 0), (0, 0, 0), (0, 2, 0), (2, 0, 0)])
        assert doubled.basis == ((2, 0, 0), (0, 2, 0))
        assert doubled.coordinates == ((0, 0), (1, 0), (0, 0), (0, 1), (1, 0))

    def test_int64_input_round_trips_exactly(self) -> None:
        # The differences and the basis overflow int64; the chart must not.
        arr = np.array([[2**62 + 5, 0], [-(2**62) - 5, 1], [0, 2]], dtype=np.int64)
        chart = lattice_chart(arr)
        coordinates = lattice_coordinates(arr)
        assert coordinates == list(chart.coordinates)
        expected = [tuple(p) for p in arr.tolist()]
        assert [chart.to_ambient(c) for c in coordinates] == expected
        assert [chart.to_ambient(np.array(c, dtype=np.int64)) for c in coordinates] == expected

    def test_to_ambient_takes_integer_entries_exactly(self) -> None:
        chart = lattice_chart([(0, 0), (3, 1)])
        image = chart.to_ambient(np.array([2**62], dtype=np.int64))
        assert image == (3 * 2**62, 2**62)
        assert all(type(x) is int for x in image)
        assert chart.to_ambient((2.0,)) == (6, 2)
        with pytest.raises(ValidationError, match="non-integer"):
            chart.to_ambient((0.5,))


SAMPLE_DIAGRAMS = [
    "11e|e|:zz",
    "12e|2e|e|:zzz",
    "12e|2e|e|:nzz",
    "12e|3e|3e|e|:zzzz",
    "12e|3e|3e|e|:nnnn",
    "111e|e|:nnn",
    "12e|23|3|e|:nnnnn",
]


def diagram_points(cnickel: str) -> list[tuple[int, ...]]:
    """The Newton polytope points of G = U + F; tadpoles need use_mandelstam=False."""
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=not cnickel.startswith("0|"))
    return [tuple(int(x) for x in p) for p in fi.newton_polytope.points]


class TestExactFaces:
    def test_segment_endpoints_hold_every_repeated_point(self) -> None:
        assert faces([(0, 0), (2, 2), (2, 2)]) == [(0, (0,)), (0, (1, 2)), (1, (0, 1, 2))]
        assert faces([(0, 0), (1, 1), (1, 1), (2, 2)]) == [
            (0, (0,)),
            (0, (3,)),
            (1, (0, 1, 2, 3)),
        ]

    def test_points_and_empty_input(self) -> None:
        assert faces([(1, 2), (1, 2)]) == [(0, (0, 1))]
        assert faces([]) == []

    def test_lower_dimensional_parallelogram(self) -> None:
        assert faces([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, -1)]) == [
            (0, (0,)),
            (0, (1,)),
            (0, (2,)),
            (0, (3,)),
            (1, (0, 2)),
            (1, (0, 3)),
            (1, (1, 2)),
            (1, (1, 3)),
            (2, (0, 1, 2, 3)),
        ]

    def test_non_integer_input_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            faces([(0.5, 0), (1, 0)])
        with pytest.raises(ValidationError, match="non-integer"):
            polytope_data([(0, 0.5), (1, 0)])

    def test_ragged_input_raises(self) -> None:
        with pytest.raises(ValidationError, match="point 1 has 1 coordinate, point 0 has 2"):
            faces([(0, 0), (1,)])
        with pytest.raises(ValidationError, match="point 1 has 1 coordinate, point 0 has 2"):
            polytope_data([(0, 0), (1,)])

    def test_numpy_input(self) -> None:
        assert polytope_data(np.array([[0, 0], [1, 0], [0, 1]])).f_vector == (3, 3, 1)

    def test_segment_in_z1_keeps_its_facets(self) -> None:
        data = polytope_data([(1,), (2,)])
        assert data.is_full_dimensional
        assert [(f.normal, f.offset, f.point_indices) for f in data.facets] == [
            ((-1,), -1, (0,)),
            ((1,), 2, (1,)),
        ]


class _FakeCone:
    """Stands in for PyNormaliz.Cone: support hyperplanes (-m, b), so -m . x + b >= 0 on P."""

    skip = 0

    def __init__(self, *, vertices: list[list[int]]) -> None:
        assert all(v[-1] == 1 for v in vertices)
        points = [tuple(v[:-1]) for v in vertices]
        self._forms = [[*(-m for m in h.normal), h.offset] for h in _exact.beneath_beyond(points)]

    def SupportHyperplanes(self) -> list[list[int]]:
        return self._forms[self.skip :]


class _FakeConeMissingOne(_FakeCone):
    skip = 1


def _fake_pynormaliz(monkeypatch: pytest.MonkeyPatch, cone: type) -> None:
    module = types.ModuleType("PyNormaliz")
    module.Cone = cone
    monkeypatch.setitem(sys.modules, "PyNormaliz", module)
    monkeypatch.setattr(polytope, "_pynormaliz_available", lambda: True)


requires_normaliz = pytest.mark.skipif(
    not polytope._pynormaliz_available(), reason="PyNormaliz not installed"
)


class TestBackends:
    @pytest.mark.parametrize("cnickel", SAMPLE_DIAGRAMS)
    def test_qhull_candidates_certify_and_agree(self, cnickel: str) -> None:
        points = diagram_points(cnickel)
        halfspaces, _, generator = polytope._certified_hull(points, "qhull")
        assert generator == "qhull"
        assert sorted(halfspaces) == sorted(_exact.beneath_beyond(points))
        assert polytope_data(points, backend="qhull") == polytope_data(points, backend="python")

    def test_auto_is_python(self) -> None:
        assert polytope._resolve_backend("auto") == "python"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ComputationError, match="Unknown backend 'bogus'"):
            polytope_data([(0, 0), (1, 0), (0, 1)], backend="bogus")
        with pytest.raises(ComputationError, match="Unknown backend"):
            faces([], backend="bogus")

    def test_normaliz_without_pynormaliz_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(polytope, "_pynormaliz_available", lambda: False)
        with pytest.raises(ComputationError, match="PyNormaliz"):
            polytope_data([(0, 0), (1, 0), (0, 1)], backend="normaliz")

    def test_qhull_list_missing_a_facet_is_rejected_and_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        points = diagram_points("12e|3e|3e|e|:zzzz")
        complete = polytope._qhull_candidates(points)
        assert complete is not None and len(complete) == 10
        with pytest.raises(_exact.CertificateError):
            _exact.certified_lattice(points, [h.mask for h in complete[1:]], 4)
        monkeypatch.setattr(polytope, "_qhull_candidates", lambda coords: complete[1:])
        halfspaces, _, generator = polytope._certified_hull(points, "qhull")
        assert generator == "python"
        assert sorted(halfspaces) == sorted(_exact.beneath_beyond(points))
        assert polytope_data(points, backend="qhull") == polytope_data(points, backend="python")

    def test_qhull_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(polytope, "_qhull_candidates", lambda coords: None)
        assert polytope._certified_hull(diagram_points("12e|2e|e|:zzz"), "qhull")[2] == "python"

    def test_qhull_falls_back_on_coordinates_too_large_for_a_float(self) -> None:
        points = [(0, 0), (2**1100, 0), (0, 1), (1, 1)]
        assert polytope._qhull_candidates(points) is None
        assert polytope._certified_hull(points, "qhull")[2] == "python"
        assert faces(points, backend="qhull") == faces(points)

    def test_python_path_certificate_failure_is_a_bug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        points = diagram_points("12e|2e|e|:zzz")
        complete = _exact.beneath_beyond(points)
        monkeypatch.setattr(_exact, "beneath_beyond", lambda coords: complete[1:])
        with pytest.raises(ComputationError, match=r"completeness certificate.*\(C"):
            polytope_data(points)

    def test_normaliz_candidates_are_verified_and_certified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_pynormaliz(monkeypatch, _FakeCone)
        points = diagram_points("12e|3e|3e|e|:zzzz")
        assert polytope._certified_hull(points, "normaliz")[2] == "normaliz"
        assert polytope_data(points, backend="normaliz") == polytope_data(points, backend="python")

    def test_incomplete_normaliz_list_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_pynormaliz(monkeypatch, _FakeConeMissingOne)
        points = diagram_points("12e|3e|3e|e|:zzzz")
        assert polytope._certified_hull(points, "normaliz")[2] == "python"

    @requires_normaliz
    @pytest.mark.parametrize("cnickel", SAMPLE_DIAGRAMS)
    def test_normaliz_agrees_with_python(self, cnickel: str) -> None:
        points = diagram_points(cnickel)
        assert polytope._certified_hull(points, "normaliz")[2] == "normaliz"
        assert polytope_data(points, backend="normaliz") == polytope_data(points, backend="python")
