"""Tests for the exact integer routines in feynkit._exact."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from scipy.spatial import ConvexHull
from sympy.matrices.normalforms import hermite_normal_form as sympy_hermite
from sympy.matrices.normalforms import invariant_factors

from feynkit import _exact
from feynkit.core.exceptions import ComputationError, ValidationError


def _matrix(seed: int) -> list[list[int]]:
    """A random small integer matrix, rank-deficient about half the time in one of two ways.

    A row can be a combination of two others (row-rank deficiency), or a
    column can be a multiple of the first column (column-rank deficiency).
    """
    rng = random.Random(seed)
    m, n = rng.randint(1, 6), rng.randint(1, 7)
    rows = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(m)]
    if m > 1 and rng.random() < 0.3:
        rows[-1] = [a + 2 * b for a, b in zip(rows[0], rows[1], strict=True)]
    if n > 1 and rng.random() < 0.3:
        c = rng.randrange(1, n)
        for row in rows:
            row[c] = 3 * row[0]
    return rows


def _square(seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    n = rng.randint(0, 6)
    return [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]


MATRICES = [_matrix(seed) for seed in range(300)]
SQUARES = [_square(seed) for seed in range(100)]


class TestIntegerPoints:
    def test_converts_arrays_and_sequences(self) -> None:
        points = _exact.integer_points(np.array([[0, 1], [2, 3]], dtype=np.int64))
        assert points == [(0, 1), (2, 3)]
        assert all(type(x) is int for p in points for x in p)
        assert _exact.integer_points([(0, 1.0), [2, sp.Integer(3)]]) == [(0, 1), (2, 3)]
        assert _exact.integer_points([]) == []
        assert _exact.integer_points(np.zeros((0, 3), dtype=int)) == []

    def test_accepts_sympy_and_fraction_integral_values(self) -> None:
        assert _exact.integer_points([(sp.Float(1.0), sp.Rational(4, 2))]) == [(1, 2)]
        assert _exact.integer_points([(Fraction(6, 3),)]) == [(2,)]

    def test_rejects_non_integers_and_ragged_input(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.integer_points([(0, 0.5)])
        with pytest.raises(ValidationError, match="coordinates"):
            _exact.integer_points([(0, 0), (1,)])
        with pytest.raises(ValidationError, match="two-dimensional"):
            _exact.integer_points(np.array([1, 2, 3]))

    def test_rejects_non_sequence_input(self) -> None:
        with pytest.raises(ValidationError, match="sequence"):
            _exact.integer_points(5)  # type: ignore[arg-type]

    def test_rationals_beyond_float_range(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.integer_points([(Fraction(10**400, 3),)])
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.integer_points([(sp.Rational(10**400, 3),)])
        assert _exact.integer_points([(Fraction(3 * 10**400, 3),)]) == [(10**400,)]

    def test_mask_indices(self) -> None:
        assert _exact.mask_indices(0) == []
        assert _exact.mask_indices(0b101001) == [0, 3, 5]

    def test_mask_indices_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            _exact.mask_indices(-1)


class TestDeterminantAndRank:
    def test_determinant_matches_sympy(self) -> None:
        for matrix in SQUARES:
            assert _exact.determinant(matrix) == sp.Matrix(matrix).det()

    def test_rank_matches_sympy(self) -> None:
        for rows in MATRICES:
            assert _exact.rank(rows) == sp.Matrix(rows).rank()

    def test_small_cases(self) -> None:
        assert _exact.determinant([]) == 1
        assert _exact.determinant([[0, 1], [1, 0]]) == -1
        assert _exact.determinant([[2, 4], [1, 2]]) == 0
        assert _exact.rank([]) == 0
        assert _exact.rank([[0, 0, 0]]) == 0

    def test_rank_rejects_ragged_rows(self) -> None:
        with pytest.raises(ValidationError, match="row 1 has 2"):
            _exact.rank([[1, 2, 3], [1, 2]])
        with pytest.raises(ValidationError, match="row 2 has 4"):
            _exact.rank([[1, 2, 3], [4, 5, 6], [1, 2, 3, 4]])

    def test_rejects_non_integer_entries(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.determinant([[1, 0], [0, 1.5]])
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.rank([[1, 2], [0.5, 1]])
        assert _exact.determinant([[2.0, 0], [0, np.int64(3)]]) == 6

    def test_determinant_rejects_non_square(self) -> None:
        with pytest.raises(ValidationError, match="square"):
            _exact.determinant([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValidationError, match="square"):
            _exact.determinant([[1, 2], [3, 4], [5, 6]])


class TestDot:
    def test_dot_rejects_vectors_of_different_length(self) -> None:
        with pytest.raises(ValidationError, match="length"):
            _exact.dot([1, 2], [1])

    def test_dot_rejects_non_integers(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.dot([1, 2], [1.5, 4])
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.dot([Fraction(1, 2)], [2])

    def test_dot_converts_numpy_integers_exactly(self) -> None:
        # 2**80 overflows int64, so the product must be taken in Python integers.
        u = np.array([2**40, 3], dtype=np.int64)
        result = _exact.dot(u, u)
        assert result == 2**80 + 9
        assert type(result) is int
        assert _exact.dot([2.0, 1], [3, 1]) == 7


class TestAffineSpan:
    def test_affine_rank(self) -> None:
        assert _exact.affine_rank([]) == 0
        assert _exact.affine_rank([(1, 2)]) == 0
        assert _exact.affine_rank([(0, 0), (1, 1), (2, 2)]) == 1
        assert _exact.affine_rank([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]) == 2
        cube_corner = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        assert _exact.affine_rank(cube_corner, bound=1) == 1

    def test_affine_rank_matches_sympy(self) -> None:
        for rows in MATRICES:
            diffs = [[a - b for a, b in zip(r, rows[0], strict=True)] for r in rows[1:]]
            assert _exact.affine_rank(rows) == (sp.Matrix(diffs).rank() if diffs else 0)

    def test_affine_basis_is_greedy_in_the_given_order(self) -> None:
        pts = [(0, 0), (0, 0), (1, 1), (2, 2), (1, 0), (5, 7)]
        assert _exact.affine_basis(pts, range(len(pts)), 3) == [0, 2, 4]
        assert _exact.affine_basis(pts, [3, 2, 1], 3) == [3, 2]
        assert _exact.affine_basis(pts, range(len(pts)), 0) == []

    def test_affine_rank_does_not_overflow_numpy_int64(self) -> None:
        # 2**32 overflows an int64 product once combined with another large entry;
        # the true affine rank of these three points is 2.
        points = np.array([(0, 0, 0), (2**32, 1, 0), (2**32, 2**32 + 1, 0)], dtype=np.int64)
        assert _exact.affine_rank(points) == 2

    def test_affine_span_rejects_ragged_points(self) -> None:
        with pytest.raises(ValidationError, match="coordinates"):
            _exact.affine_rank([(0, 0), (1, 2, 3)])
        span = _exact.AffineSpan()
        span.add((0, 0))
        with pytest.raises(ValidationError, match="coordinates"):
            span.add((1, 2, 3))

    def test_affine_span_rejects_non_integers(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.affine_rank([(0, 0), (1.5, 0)])
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.affine_rank([(0.5, 0), (1, 1)])
        assert _exact.affine_rank(np.array([(0, 0), (2**40, 1)], dtype=np.int64)) == 1


class TestHyperplaneNormal:
    def test_primitive_normals(self) -> None:
        assert _exact.hyperplane_normal([(0, 0), (2, 4)]) in {(2, -1), (-2, 1)}
        normal = _exact.hyperplane_normal([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        assert normal in {(1, 1, 1), (-1, -1, -1)}

    def test_normal_is_orthogonal_and_primitive(self) -> None:
        rng = random.Random(3)
        for _ in range(100):
            d = rng.randint(2, 6)
            pts = [tuple(rng.randint(-3, 3) for _ in range(d)) for _ in range(d)]
            if _exact.affine_rank(pts) != d - 1:
                continue
            normal = _exact.hyperplane_normal(pts)
            assert math.gcd(*normal) == 1
            assert all(_exact.dot(normal, p) == _exact.dot(normal, pts[0]) for p in pts)

    def test_dependent_points_raise(self) -> None:
        with pytest.raises(ComputationError, match="hyperplane"):
            _exact.hyperplane_normal([(0, 0, 0), (1, 1, 1), (2, 2, 2)])

    def test_no_points_raises(self) -> None:
        with pytest.raises(ComputationError, match="hyperplane"):
            _exact.hyperplane_normal([])

    def test_rejects_ragged_points(self) -> None:
        with pytest.raises(ValidationError, match="coordinates"):
            _exact.hyperplane_normal([(0, 0), (1, 2, 3)])

    def test_rejects_non_integers(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.hyperplane_normal([(0, 0), (1, 0.5)])
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.hyperplane_normal([(0.5, 0), (1, 1)])

    def test_orientation_sign_convention(self) -> None:
        # det([x; diffs]) = g * (normal . x) for every x, with g > 0: this pins
        # the sign of hyperplane_normal to the cofactor expansion along the
        # first row, not just its magnitude.
        rng = random.Random(2024)
        done = 0
        while done < 50:
            d = rng.randint(2, 5)
            pts = [tuple(rng.randint(-4, 4) for _ in range(d)) for _ in range(d)]
            if _exact.affine_rank(pts) != d - 1:
                continue
            done += 1
            normal = _exact.hyperplane_normal(pts)
            diffs = [[a - b for a, b in zip(p, pts[0], strict=True)] for p in pts[1:]]
            minors = [
                (-1) ** j * _exact.determinant([row[:j] + row[j + 1 :] for row in diffs])
                for j in range(d)
            ]
            g = math.gcd(*minors)
            assert g > 0
            for _ in range(3):
                x = [rng.randint(-4, 4) for _ in range(d)]
                det = _exact.determinant([x, *diffs])
                assert det == g * _exact.dot(normal, x)


class TestHermiteNormalForm:
    def test_matches_sympy(self) -> None:
        for rows in MATRICES:
            if sp.Matrix(rows).rank() == 0:
                continue
            expected = sympy_hermite(sp.Matrix(rows)).tolist()
            assert _exact.hermite_normal_form(rows, len(rows[0])) == expected

    def test_transform_is_unimodular_and_splits_off_the_kernel(self) -> None:
        for rows in MATRICES:
            ncols = len(rows[0])
            form = _exact.hermite_normal_form_with_transform(rows, ncols)
            assert form.hermite == _exact.hermite_normal_form(rows, ncols)
            assert abs(_exact.determinant(form.transform)) == 1
            product = [
                [sum(r[k] * form.transform[k][j] for k in range(ncols)) for j in range(ncols)]
                for r in rows
            ]
            zero = ncols - form.rank
            assert all(row[:zero] == [0] * zero for row in product)
            assert [row[zero:] for row in product] == form.hermite
            assert len(form.kernel) == zero
            assert all(_exact.dot(r, v) == 0 for r in rows for v in form.kernel)

    def test_empty_matrix(self) -> None:
        form = _exact.hermite_normal_form_with_transform([], 3)
        assert form.rank == 0
        assert form.kernel == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert form.image == []

    def test_solve_and_reduce(self) -> None:
        h = _exact.hermite_normal_form([[2, 0], [0, 2], [0, 0]], 2)
        assert h == [[2, 0], [0, 2], [0, 0]]
        assert _exact.solve_hermite(h, [1, 2, 0]) == [Fraction(1, 2), Fraction(1)]
        assert _exact.solve_hermite(h, [1, 2, 1]) is None
        k = _exact.hermite_normal_form([[1], [1], [1]], 1)
        assert _exact.pivot_rows(k) == [2]
        assert _exact.reduce_modulo([3, 5, 7], k) == [-4, -2, 0]

    def test_rejects_wrong_width(self) -> None:
        with pytest.raises(ValidationError, match="ncols"):
            _exact.hermite_normal_form([[1, 2, 3]], 2)
        with pytest.raises(ValidationError, match="ncols"):
            _exact.hermite_normal_form_with_transform([[1, 2]], 3)

    def test_rejects_non_integer_entries(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.hermite_normal_form([[1, 2.5]], 2)

    def test_solve_and_reduce_properties(self) -> None:
        rng = random.Random(9001)
        done = 0
        while done < 50:
            rows = _matrix(rng.randint(0, 10**9))
            ncols = len(rows[0])
            h = _exact.hermite_normal_form(rows, ncols)
            width = len(h[0]) if h else 0
            if width == 0:
                continue
            done += 1
            m = len(h)
            pivots = _exact.pivot_rows(h)
            assert pivots == [max(i for i in range(m) if h[i][t]) for t in range(width)]

            # an integer combination of the columns is recovered exactly
            z = [rng.randint(-4, 4) for _ in range(width)]
            target = [sum(h[i][t] * z[t] for t in range(width)) for i in range(m)]
            assert _exact.solve_hermite(h, target) == [Fraction(x) for x in z]

            # an arbitrary integer target: None exactly when it lies outside the span
            v = [rng.randint(-4, 4) for _ in range(m)]
            rank_h = sp.Matrix(h).rank()
            in_span = sp.Matrix.hstack(sp.Matrix(h), sp.Matrix(v)).rank() == rank_h
            got = _exact.solve_hermite(h, v)
            assert (got is not None) == in_span
            if got is not None:
                assert [sum(h[i][t] * got[t] for t in range(width)) for i in range(m)] == v

            # reduce_modulo: same coset, canonical range, invariant under lattice shifts
            red = _exact.reduce_modulo(v, h)
            diff = [x - y for x, y in zip(v, red, strict=True)]
            coeff = _exact.solve_hermite(h, diff)
            assert coeff is not None and all(c.denominator == 1 for c in coeff)
            for t, i in enumerate(pivots):
                assert 0 <= red[i] < h[i][t]
            shift = [rng.randint(-5, 5) for _ in range(width)]
            moved = [v[i] + sum(h[i][t] * shift[t] for t in range(width)) for i in range(m)]
            assert _exact.reduce_modulo(moved, h) == red

    def test_image_maps_onto_the_hermite_columns(self) -> None:
        rng = random.Random(4242)
        for _ in range(50):
            rows = _matrix(rng.randint(0, 10**9))
            ncols = len(rows[0])
            form = _exact.hermite_normal_form_with_transform(rows, ncols)
            for t, v in enumerate(form.image):
                assert [_exact.dot(r, v) for r in rows] == [row[t] for row in form.hermite]


class TestSmithInvariants:
    def test_matches_sympy(self) -> None:
        for rows in MATRICES:
            expected = [abs(int(x)) for x in invariant_factors(sp.Matrix(rows)) if x != 0]
            assert _exact.smith_invariants(rows, len(rows[0])) == expected

    def test_known_values(self) -> None:
        assert _exact.smith_invariants([[2, 0], [0, 1]], 2) == [1, 2]
        assert _exact.smith_invariants([[4, 0], [2, 2], [2, 1]], 2) == [1, 2]
        assert _exact.smith_invariants([], 3) == []
        assert _exact.smith_invariants([[0, 0]], 2) == []

    def test_rejects_wrong_width(self) -> None:
        with pytest.raises(ValidationError, match="ncols"):
            _exact.smith_invariants([[2, 4, 3]], 2)

    def test_rejects_non_integer_entries(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.smith_invariants([[2, 0.5]], 2)


def _facet_set(
    halfspaces: list[_exact.Halfspace],
) -> set[tuple[tuple[int, ...], int, tuple[int, ...]]]:
    return {(h.normal, h.offset, tuple(_exact.mask_indices(h.mask))) for h in halfspaces}


class TestVerifyFacet:
    def test_accepts_orients_and_records_the_tight_set(self) -> None:
        square = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 0)]
        assert _exact.verify_facet(square, [0, 1, 4], range(5)) == _exact.Halfspace(
            (0, -1), 0, 0b10011
        )

    def test_rejects_a_hyperplane_with_points_on_both_sides(self) -> None:
        square = [(0, 0), (2, 0), (0, 2), (2, 2)]
        assert _exact.verify_facet(square, [0, 3], range(4)) is None

    def test_rejects_too_few_independent_points(self) -> None:
        cube = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        assert _exact.verify_facet(cube, [0, 1], range(8)) is None


class TestBeneathBeyond:
    def test_square_with_boundary_and_interior_points(self) -> None:
        pts = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (1, 1)]
        assert _facet_set(_exact.beneath_beyond(pts)) == {
            ((0, -1), 0, (0, 1, 4)),
            ((-1, 0), 0, (0, 2)),
            ((1, 0), 2, (1, 3)),
            ((0, 1), 2, (2, 3)),
        }

    def test_repeated_points_share_their_faces(self) -> None:
        pts = [(0, 0), (1, 0), (0, 1), (1, 0)]
        assert _facet_set(_exact.beneath_beyond(pts)) == {
            ((0, -1), 0, (0, 1, 3)),
            ((-1, 0), 0, (0, 2)),
            ((1, 1), 1, (1, 2, 3)),
        }

    def test_octahedron(self) -> None:
        pts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        got = _facet_set(_exact.beneath_beyond(pts))
        assert {(m, b) for m, b, _ in got} == {
            ((a, b, c), 1) for a in (1, -1) for b in (1, -1) for c in (1, -1)
        }
        assert all(len(idx) == 3 for _, _, idx in got)

    def test_square_pyramid_keeps_its_square_base_whole(self) -> None:
        pts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (1, 1, 1)]
        got = _facet_set(_exact.beneath_beyond(pts))
        assert ((0, 0, -1), 0, (0, 1, 2, 3)) in got
        assert len(got) == 5

    def test_needs_full_dimensional_points(self) -> None:
        with pytest.raises(ComputationError, match="full-dimensional"):
            _exact.beneath_beyond([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        with pytest.raises(ComputationError, match="at least 2"):
            _exact.beneath_beyond([(0,), (1,)])

    def test_needs_at_least_one_point(self) -> None:
        with pytest.raises(ComputationError, match="at least one point"):
            _exact.beneath_beyond([])

    def test_matches_qhull_on_random_configurations(self) -> None:
        rng = random.Random(11)
        checked = 0
        for _ in range(150):
            d = rng.randint(2, 5)
            count = rng.randint(d + 1, d + 12)
            pts = sorted({tuple(rng.randint(-3, 3) for _ in range(d)) for _ in range(count)})
            if _exact.affine_rank(pts) != d:
                continue
            halfspaces = _exact.beneath_beyond(pts)
            for h in halfspaces:
                assert all(_exact.dot(h.normal, p) <= h.offset for p in pts)
                tight = [pts[j] for j in _exact.mask_indices(h.mask)]
                assert _exact.affine_rank(tight) == d - 1
            x = np.array(pts, dtype=float)
            hull = ConvexHull(x)
            expected = {
                frozenset(np.flatnonzero(np.abs(x @ eq[:-1] + eq[-1]) < 1e-7).tolist())
                for eq in hull.equations
            }
            assert {frozenset(_exact.mask_indices(h.mask)) for h in halfspaces} == expected
            checked += 1
        assert checked >= 140


CUBE = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
CUBE_FACETS = [
    ((-1, 0, 0), 0),
    ((1, 0, 0), 1),
    ((0, -1, 0), 0),
    ((0, 1, 0), 1),
    ((0, 0, -1), 0),
    ((0, 0, 1), 1),
]
PYRAMID = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (1, 1, 1)]
PYRAMID_FACETS = [((0, 0, -1), 0), ((0, -1, 1), 0), ((1, 0, 1), 2), ((0, 1, 1), 2), ((-1, 0, 1), 0)]
OCTAHEDRON = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
OCTAHEDRON_FACETS = [((a, b, c), 1) for a in (1, -1) for b in (1, -1) for c in (1, -1)]


def _masks(points: list[tuple[int, ...]], forms: list[tuple[tuple[int, ...], int]]) -> list[int]:
    """Tight-set masks of valid inequalities, checked valid on the points."""
    masks = []
    for normal, offset in forms:
        assert all(_exact.dot(normal, p) <= offset for p in points)
        masks.append(sum(1 << j for j, p in enumerate(points) if _exact.dot(normal, p) == offset))
    return masks


def _f_vector(dims: dict[int, int]) -> list[int]:
    counts = [0] * (max(dims.values()) + 1)
    for k in dims.values():
        counts[k] += 1
    return counts


class TestFaceDimensions:
    def test_cube(self) -> None:
        dims = _exact.face_dimensions(CUBE, _masks(CUBE, CUBE_FACETS), 3)
        assert _f_vector(dims) == [8, 12, 6, 1]

    def test_incomplete_pyramid_satisfies_euler_poincare(self) -> None:
        dims = _exact.face_dimensions(PYRAMID, _masks(PYRAMID, PYRAMID_FACETS[:-1]), 3)
        f = _f_vector(dims)
        assert f == [3, 5, 4, 1]
        assert sum((-1) ** k * c for k, c in enumerate(f)) == 1


class TestCertificate:
    def test_complete_lists_pass(self) -> None:
        cases = [(CUBE, CUBE_FACETS), (PYRAMID, PYRAMID_FACETS), (OCTAHEDRON, OCTAHEDRON_FACETS)]
        for points, forms in cases:
            masks = _masks(points, forms)
            lattice = _exact.certified_lattice(points, masks, 3)
            assert set(lattice.phi[lattice.top]) == set(masks)

    def test_two_edges_of_a_triangle_fail_c1(self) -> None:
        tri = [(0, 0), (1, 0), (0, 1)]
        with pytest.raises(_exact.CertificateError, match=r"\(C1\)"):
            _exact.certified_lattice(tri, _masks(tri, [((0, -1), 0), ((-1, 0), 0)]), 2)

    def test_cube_missing_one_facet_fails_c1(self) -> None:
        with pytest.raises(_exact.CertificateError, match=r"\(C1\)"):
            _exact.certified_lattice(CUBE, _masks(CUBE, CUBE_FACETS[:1] + CUBE_FACETS[2:]), 3)

    def test_cube_missing_four_facets_fails_c2(self) -> None:
        with pytest.raises(_exact.CertificateError, match=r"\(C2\)"):
            _exact.certified_lattice(CUBE, _masks(CUBE, CUBE_FACETS[:2]), 3)

    def test_pyramid_missing_a_triangle_fails_although_euler_poincare_holds(self) -> None:
        with pytest.raises(_exact.CertificateError, match=r"\(C1\)"):
            _exact.certified_lattice(PYRAMID, _masks(PYRAMID, PYRAMID_FACETS[:-1]), 3)

    def test_octahedron_missing_one_facet_fails_only_c3(self) -> None:
        with pytest.raises(_exact.CertificateError, match=r"\(C3\)"):
            _exact.certified_lattice(OCTAHEDRON, _masks(OCTAHEDRON, OCTAHEDRON_FACETS[1:]), 3)

    def test_certificate_error_is_a_computation_error(self) -> None:
        assert issubclass(_exact.CertificateError, ComputationError)


class TestPullingSimplices:
    def test_square_splits_into_two_triangles(self) -> None:
        square = [(0, 0), (1, 0), (0, 1), (1, 1)]
        masks = _masks(square, [((0, -1), 0), ((-1, 0), 0), ((1, 0), 1), ((0, 1), 1)])
        lattice = _exact.certified_lattice(square, masks, 2)
        simplices = _exact.pulling_simplices(lattice, square)
        assert sorted(simplices) == [(0, 1, 3), (0, 2, 3)]
        assert _exact.simplex_volume(square, simplices, 1) == 2

    def test_simplices_do_not_depend_on_the_input_order(self) -> None:
        def simplices_of(points: list[tuple[int, ...]]) -> set[frozenset[tuple[int, ...]]]:
            lattice = _exact.certified_lattice(points, _masks(points, OCTAHEDRON_FACETS), 3)
            return {
                frozenset(points[i] for i in s) for s in _exact.pulling_simplices(lattice, points)
            }

        assert simplices_of(OCTAHEDRON) == simplices_of(OCTAHEDRON[::-1])
        lattice = _exact.certified_lattice(OCTAHEDRON, _masks(OCTAHEDRON, OCTAHEDRON_FACETS), 3)
        simplices = _exact.pulling_simplices(lattice, OCTAHEDRON)
        assert _exact.simplex_volume(OCTAHEDRON, simplices, 1) == 8

    def test_volume_checks_the_sublattice_index(self) -> None:
        tri = [(0, 0), (2, 0), (0, 1)]
        masks = _masks(tri, [((0, -1), 0), ((-1, 0), 0), ((1, 2), 2)])
        simplices = _exact.pulling_simplices(_exact.certified_lattice(tri, masks, 2), tri)
        assert _exact.simplex_volume(tri, simplices, 2) == 1
        with pytest.raises(ComputationError, match="sublattice index 4"):
            _exact.simplex_volume(tri, simplices, 4)
