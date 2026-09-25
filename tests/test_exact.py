"""Tests for the exact integer routines in feynkit._exact."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from sympy.matrices.normalforms import hermite_normal_form as sympy_hermite
from sympy.matrices.normalforms import invariant_factors

from feynkit import _exact
from feynkit.core.exceptions import ComputationError, ValidationError


def _matrix(seed: int) -> list[list[int]]:
    """A random small integer matrix, rank-deficient about a third of the time."""
    rng = random.Random(seed)
    m, n = rng.randint(1, 6), rng.randint(1, 7)
    rows = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(m)]
    if m > 1 and rng.random() < 0.3:
        rows[-1] = [a + 2 * b for a, b in zip(rows[0], rows[1], strict=True)]
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

    def test_rejects_non_integers_and_ragged_input(self) -> None:
        with pytest.raises(ValidationError, match="non-integer"):
            _exact.integer_points([(0, 0.5)])
        with pytest.raises(ValidationError, match="coordinates"):
            _exact.integer_points([(0, 0), (1,)])
        with pytest.raises(ValidationError, match="two-dimensional"):
            _exact.integer_points(np.array([1, 2, 3]))

    def test_mask_indices(self) -> None:
        assert _exact.mask_indices(0) == []
        assert _exact.mask_indices(0b101001) == [0, 3, 5]


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
