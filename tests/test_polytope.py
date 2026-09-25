from __future__ import annotations

import math
import random
import sys
import types
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, _exact, polytope
from feynkit.a_configuration import AConfiguration
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.polytope import (
    Facet,
    LatticeChart,
    PolytopeData,
    faces,
    lattice_chart,
    lattice_coordinates,
    normalized_volume,
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


def embed(points: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Points of Z^n, n >= 2, under an injective affine map into Z^(n + 2).

    The map is x -> M x + t with M the rows of diag(2, 1, ..., 1), then
    (0, 1, ..., 1) and (4, 0, ..., 0, -1). The image of Z^n has index 2 in its
    saturation, so the chart path meets a lattice that is not saturated.
    """
    n = len(points[0])
    rows = [[(2 if i == 0 else 1) if i == j else 0 for j in range(n)] for i in range(n)]
    rows += [[0] + [1] * (n - 1), [4] + [0] * (n - 2) + [-1]]
    shift = [3, -1] + [0] * (n - 2) + [5, 7]
    return [tuple(t + _exact.dot(r, p) for r, t in zip(rows, shift, strict=True)) for p in points]


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

    def test_repeated_points_lie_on_every_face_they_belong_to(self) -> None:
        # A square pyramid, with the midpoint of a base edge and an interior point.
        distinct = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (1, 1, 2), (1, 0, 0), (1, 1, 1)]
        order = [0, 5, 4, 1, 5, 2, 3, 6, 0, 4, 6, 3]
        points = [distinct[i] for i in order]
        expected = sorted(
            (k, tuple(j for j, i in enumerate(order) if i in idx)) for k, idx in faces(distinct)
        )
        assert len(expected) == 19
        for pts in (points, embed(points)):
            assert faces(pts) == expected
            assert faces(pts, backend="qhull") == expected
            assert normalized_volume(pts) == 16
        facets = {f.point_indices for f in polytope_data(points).facets}
        assert facets == {idx for k, idx in expected if k == 2}

    def test_non_sequence_and_non_2d_input_raise(self) -> None:
        for bad in (5, [1, 2], np.zeros(3), np.zeros((2, 2, 2))):
            with pytest.raises(ValidationError, match="sequence|two-dimensional"):
                faces(bad)
            with pytest.raises(ValidationError, match="sequence|two-dimensional"):
                lattice_chart(bad)
            with pytest.raises(ValidationError, match="sequence|two-dimensional"):
                polytope_data(bad)
            with pytest.raises(ValidationError, match="sequence|two-dimensional"):
                normalized_volume(bad)

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

    @pytest.mark.parametrize("backend", ["qhull", "normaliz"])
    @pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "12e|3e|3e|e|:nnnn"])
    def test_chart_path_through_every_generator(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, cnickel: str
    ) -> None:
        if backend == "normaliz":
            _fake_pynormaliz(monkeypatch, _FakeCone)
        original = diagram_points(cnickel)
        points = embed(original)
        hull = polytope._hull(points, backend)
        assert hull.in_chart and hull.dimension >= 2 and hull.generator == backend
        data = polytope_data(points, backend=backend)
        assert data == polytope_data(points, backend="python")
        assert data.dimension < data.ambient_dimension
        assert data.faces == polytope_data(original).faces
        assert data.normalized_volume == polytope_data(original).normalized_volume

    @requires_normaliz
    @pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "12e|3e|3e|e|:nnnn"])
    def test_chart_path_through_pynormaliz(self, cnickel: str) -> None:
        points = embed(diagram_points(cnickel))
        hull = polytope._hull(points, "normaliz")
        assert hull.in_chart and hull.generator == "normaliz"
        assert polytope_data(points, backend="normaliz") == polytope_data(points, backend="python")

    @pytest.mark.parametrize(
        "error",
        [
            type("NormalizError", (Exception,), {}),
            type("NormalizInterfaceError", (Exception,), {}),
            RuntimeError,
        ],
    )
    @pytest.mark.parametrize("method", ["__init__", "SupportHyperplanes"])
    def test_normaliz_runtime_errors_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, error: type[Exception], method: str
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise error("raised inside PyNormaliz")

        _fake_pynormaliz(monkeypatch, type("FailingCone", (_FakeCone,), {method: fail}))
        points = diagram_points("12e|3e|3e|e|:zzzz")
        assert polytope._normaliz_candidates(points) is None
        assert polytope._certified_hull(points, "normaliz")[2] == "python"
        assert polytope_data(points, backend="normaliz") == polytope_data(points, backend="python")


# A steep set on which a floating hull loses the volume: (177678, -2) repeats, and the hull is
# the pentagon on points 8, 4, 3, 2, 7, in counterclockwise order.
STEEP = [
    (177678, -2),
    (-88840, 1),
    (-177677, 2),
    (266518, -3),
    (266516, -3),
    (88840, -1),
    (177678, -2),
    (-177679, 2),
    (-10000000000, 0),
]


class TestNormalizedVolume:
    def test_matches_polytope_data_and_every_backend(self) -> None:
        for cnickel in SAMPLE_DIAGRAMS:
            points = diagram_points(cnickel)
            volume = polytope_data(points).normalized_volume
            assert normalized_volume(points) == volume
            assert normalized_volume(points, backend="qhull") == volume

    def test_bms_volumes(self) -> None:
        from feynkit import bms_simplex_a_config

        for n, volume in [(3, 4), (4, 8), (5, 16)]:
            points = [tuple(int(x) for x in p) for p in bms_simplex_a_config(n).affine_points]
            assert polytope_data(points).normalized_volume == volume

    def test_bad_input_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one point"):
            normalized_volume([])
        with pytest.raises(ComputationError, match="Unknown backend"):
            normalized_volume([(0, 0), (1, 0), (0, 1)], backend="bogus")

    def test_steep_set_is_exact(self) -> None:
        # The differences span Z^2, so the volume is twice the area of the hull, which the
        # shoelace formula gives as 50000000015. A floating hull gave 50000000005.
        hull = [STEEP[i] for i in (8, 4, 3, 2, 7)]
        twice_area = sum(
            p[0] * q[1] - q[0] * p[1] for p, q in zip(hull, hull[1:] + hull[:1], strict=True)
        )
        assert twice_area == 50000000015
        assert polytope_data(STEEP).vertex_indices == (2, 3, 4, 7, 8)
        for backend in ("python", "qhull"):
            assert normalized_volume(STEEP, backend=backend) == twice_area
            assert polytope_data(STEEP, backend=backend).normalized_volume == twice_area


def assert_lattice_forms(data: PolytopeData) -> None:
    """Check each relative facet's lattice form, chart invariants and ambient inequality."""
    for f in data.relative_facets:
        form = f.lattice_form
        values = [
            form[0] + sum(c * x for c, x in zip(form[1:], p, strict=True)) for p in data.points
        ]
        assert all(v.denominator == 1 and v >= 0 for v in values)
        assert math.gcd(*(int(v) for v in values)) == 1
        assert tuple(j for j, v in enumerate(values) if v == 0) == f.point_indices
        assert f.lattice_normal is not None and f.lattice_offset is not None
        assert math.gcd(*f.normal) == 1 and math.gcd(*f.lattice_normal) == 1
        assert tuple(_exact.dot(b, f.normal) for b in data.chart.basis) == tuple(
            f.lattice_index * m for m in f.lattice_normal
        )
        assert (
            f.offset - _exact.dot(f.normal, data.chart.origin) == f.lattice_index * f.lattice_offset
        )
        heights = [f.offset - _exact.dot(f.normal, p) for p in data.points]
        assert all(h >= 0 for h in heights)
        assert math.gcd(*heights) == f.lattice_index
    assert len(data.affine_hull) == data.ambient_dimension - data.dimension
    for row in data.affine_hull:
        assert all(row[0] + _exact.dot(row[1:], p) == 0 for p in data.points)
    if data.is_full_dimensional:
        assert data.relative_facets == data.facets
    else:
        assert data.facets == ()


class TestLatticeForms:
    def test_lattice_indices_differ_per_facet(self) -> None:
        data = polytope_data([(0, 0), (2, 0), (0, 1)])
        assert data.smith_invariants == (1, 2)
        assert data.sublattice_index == 2
        assert [(f.normal, f.offset, f.lattice_index) for f in data.facets] == [
            ((0, -1), 0, 1),
            ((-1, 0), 0, 2),
            ((1, 2), 2, 2),
        ]
        assert data.facets[1].lattice_form == (Fraction(0), Fraction(1, 2), Fraction(0))
        assert_lattice_forms(data)

    def test_index_two_lattice_with_primitive_facets(self) -> None:
        data = polytope_data([(0, 0), (4, 0), (2, 2), (2, 1)])
        assert data.sublattice_index == 2
        assert [f.lattice_index for f in data.facets] == [1, 1, 1]
        assert_lattice_forms(data)

    def test_trivial_smith_invariants_give_index_one(self) -> None:
        for cnickel in SAMPLE_DIAGRAMS:
            data = polytope_data(diagram_points(cnickel))
            assert data.sublattice_index == 1
            assert all(f.lattice_index == 1 for f in data.facets)
            assert_lattice_forms(data)

    def test_facet_defaults(self) -> None:
        f = Facet((1, 0), 1, (0,))
        assert f.lattice_normal is None and f.lattice_offset is None and f.lattice_index == 1
        assert f.homogenised_form == (1, -1, 0)
        assert f.lattice_form == f.homogenised_form

    def test_full_dimensional_lift_reproduces_the_facets(self) -> None:
        cases = [
            [(0, 0), (2, 0), (0, 1)],
            [(0, 0), (4, 0), (2, 2), (2, 1)],
            diagram_points("12e|2e|e|:nzz"),
        ]
        for points in cases:
            data = polytope_data(points)
            lift = polytope._lift_data(list(data.points), data.chart)
            for f in data.facets:
                assert polytope._lift_form(lift, f.lattice_normal, f.lattice_offset) == (
                    f.normal,
                    f.offset,
                    f.lattice_index,
                )

    def test_smith_invariants_match_aconfiguration(self) -> None:
        for points in (
            [(0, 0), (2, 0), (0, 1)],
            [(0, 0, 0), (2, 0, 0), (0, 2, 0)],
            diagram_points("12e|2e|e|:zzz"),
        ):
            cfg = AConfiguration(np.array(points).T, is_homogenized=False)
            assert list(polytope_data(points).smith_invariants) == cfg.smith_invariants


class TestLowerDimensional:
    def test_parallelogram_in_r3(self) -> None:
        data = polytope_data([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, -1)])
        assert data.dimension == 2 and data.facets == ()
        assert data.affine_hull == ((-1, 1, 1, 1),)
        assert [(f.normal, f.offset, f.point_indices) for f in data.relative_facets] == [
            ((0, -1, 0), 0, (0, 2)),
            ((1, 0, 0), 1, (0, 3)),
            ((-1, 0, 0), 0, (1, 2)),
            ((0, 1, 0), 1, (1, 3)),
        ]
        assert data.normalized_volume == 2
        assert_lattice_forms(data)

    def test_scaled_triangle_lifts_with_index_two(self) -> None:
        data = polytope_data([(0, 0, 0), (2, 0, 0), (0, 2, 0)])
        assert data.chart.coordinates == ((0, 0), (1, 0), (0, 1))
        assert data.affine_hull == ((0, 0, 0, 1),)
        assert data.smith_invariants == (2, 2) and data.sublattice_index == 4
        f = {g.point_indices: g for g in data.relative_facets}[(0, 2)]
        assert (f.normal, f.offset, f.lattice_index) == ((-1, 0, 0), 0, 2)
        assert (f.lattice_normal, f.lattice_offset) == ((-1, 0), 0)
        assert data.normalized_volume == 1
        assert_lattice_forms(data)

    def test_segment_and_point(self) -> None:
        seg = polytope_data([(0, 0), (1, 1), (2, 2)])
        assert seg.facets == ()
        assert seg.affine_hull == ((0, -1, 1),)
        assert [
            (f.normal, f.offset, f.point_indices, f.lattice_index) for f in seg.relative_facets
        ] == [
            ((-1, 0), 0, (0,), 1),
            ((1, 0), 2, (2,), 1),
        ]
        point = polytope_data([(3, 4)])
        assert point.relative_facets == ()
        assert point.affine_hull == ((-3, 1, 0), (-4, 0, 1))
        assert point.chart == LatticeChart(origin=(3, 4), basis=(), coordinates=((),))

    def test_random_embeddings(self) -> None:
        for points in embedded_configurations(10, 120):
            assert_lattice_forms(polytope_data(points))

    def test_wide_input_keeps_the_affine_hull_small(self) -> None:
        # 1500 points x = (c, R c + s) of a 4-dimensional affine subspace of Z^15, with c in
        # [-5, 5]^4 and the 16 corners of that box among them, so that P is the box. The forms
        # vanishing on the differences have the basis (-R_i, e_i), which is already a Hermite
        # normal form: the last non-zero entry of column i is a 1 in row 4 + i, with zeros to
        # its right. affine_hull is therefore exactly the rows (-s_i, -R_i, e_i), and no entry
        # exceeds 3 in absolute value, however many points there are.
        rng = random.Random(15)
        d, n = 4, 15
        r = [[rng.randint(-3, 3) for _ in range(d)] for _ in range(n - d)]
        s = [rng.randint(-3, 3) for _ in range(n - d)]
        corners = [tuple(10 * ((k >> i) & 1) - 5 for i in range(d)) for k in range(2**d)]
        inner = [tuple(rng.randint(-5, 5) for _ in range(d)) for _ in range(1500 - 2**d)]
        points = [
            (*c, *(si + _exact.dot(ri, c) for ri, si in zip(r, s, strict=True)))
            for c in corners + inner
        ]
        data = polytope_data(points)
        assert data.dimension == d and data.f_vector == (16, 32, 24, 8, 1)
        assert data.affine_hull == tuple(
            (-si, *(-x for x in ri), *(int(j == i) for j in range(n - d)))
            for i, (ri, si) in enumerate(zip(r, s, strict=True))
        )
        assert max(abs(x) for row in data.affine_hull for x in row) <= 3
        assert_lattice_forms(data)
