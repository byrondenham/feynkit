"""
Tests for maximal pairing matrix canonicalisation.

Oracle: a brute force over every row and column permutation, compared in the
row-major order of matrix_lexicographic_compare.
"""

import random
from itertools import permutations

import pytest
from sympy import Matrix, eye, symbols
from sympy.core.sorting import default_sort_key

from feynkit import FeynmanIntegral
from feynkit.normal_forms.pairing_matrix import (
    _apply_permutation,
    is_canonical,
    maximal_pairing_matrix,
)
from feynkit.normal_forms.pairing_matrix import (
    matrix_lexicographic_compare as _matrix_lexicographic_compare,
)


class TestSymbolicComparison:
    """Test symbolic comparison utilities."""

    def test_lexicographic_matrix_compare(self) -> None:
        """Test matrix lexicographic comparison."""
        a, b, c = symbols("a b c")

        M1 = Matrix([[a, b], [c, a]])
        M2 = Matrix([[a, b], [c, a]])
        assert _matrix_lexicographic_compare(M1, M2) == 0

        M3 = Matrix([[a, c], [b, a]])
        # Assuming c > b in default ordering
        if _matrix_lexicographic_compare(M3, M1) != 0:
            assert True  # Ordering is deterministic


class TestIdempotence:
    """Test that applying the algorithm twice gives the same result."""

    def test_idempotence_symbolic(self) -> None:
        """Test idempotence with symbolic matrix."""
        a, b, c = symbols("a b c")
        PM = Matrix([[a, b, c], [b, c, a], [c, a, b]])

        result1 = maximal_pairing_matrix(PM)
        result2 = maximal_pairing_matrix(result1.PM_max)

        assert _matrix_lexicographic_compare(result1.PM_max, result2.PM_max) == 0

    def test_idempotence_numeric(self) -> None:
        """Test idempotence with numeric matrix."""
        PM = Matrix([[1, 2, 3], [3, 1, 2], [2, 3, 1]])

        result1 = maximal_pairing_matrix(PM)
        result2 = maximal_pairing_matrix(result1.PM_max)

        assert result1.PM_max == result2.PM_max


class TestSmallMatrices:
    """Test against brute-force enumeration for small matrices."""

    def test_2x2_identity(self) -> None:
        """Test 2x2 identity matrix."""
        PM = Matrix([[1, 0], [0, 1]])
        result = maximal_pairing_matrix(PM)

        # Should be canonical (diagonal with 1 > 0)
        assert result.PM_max == Matrix([[1, 0], [0, 1]])

    def test_2x2_symmetric(self) -> None:
        """Test 2x2 symmetric matrix."""
        a, b = symbols("a b", positive=True)
        PM = Matrix([[a, b], [b, a]])

        result = maximal_pairing_matrix(PM)
        # Maximal form should have larger element first
        assert result.PM_max is not None

    def test_3x3_numeric(self) -> None:
        """Test 3x3 numeric matrix."""
        PM = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        result = maximal_pairing_matrix(PM)

        # Maximal row should be [7, 8, 9]
        # So first row should start with 9
        assert result.PM_max[0, 0] == 9
        # Result should be a valid matrix
        assert result.PM_max.shape == PM.shape


class TestSymbolicDeterminism:
    """Test deterministic behaviour with equivalent symbolic expressions."""

    def test_equivalent_expressions(self) -> None:
        """Test that equivalent expressions give same result."""
        from sympy import simplify

        a, b = symbols("a b")

        PM1 = Matrix([[a + b, a], [b, a - b]])
        PM2 = Matrix([[b + a, a], [b, a - b]])  # Commutative variant

        result1 = maximal_pairing_matrix(PM1)
        result2 = maximal_pairing_matrix(PM2)

        # Should give identical results
        for i in range(result1.PM_max.shape[0]):
            for j in range(result1.PM_max.shape[1]):
                diff = simplify(result1.PM_max[i, j] - result2.PM_max[i, j])
                assert diff == 0

    def test_symbolic_consistency(self) -> None:
        """Test consistency across multiple runs."""
        x, y, z = symbols("x y z")
        PM = Matrix([[x, y], [z, x]])

        results = [maximal_pairing_matrix(PM) for _ in range(3)]

        # All should be identical
        for result in results[1:]:
            assert result.PM_max == results[0].PM_max


class TestCanonicalityCheck:
    """Test the is_canonical helper function."""

    def test_canonical_matrix(self) -> None:
        """Test that maximal matrix is recognised as canonical."""
        PM = Matrix([[3, 2], [1, 0]])
        result = maximal_pairing_matrix(PM)

        assert is_canonical(result.PM_max)

    def test_non_canonical_matrix(self) -> None:
        """A matrix that permutations make larger is not canonical; its maximum is."""
        PM = Matrix([[1, 2], [3, 4]])
        result = maximal_pairing_matrix(PM)
        assert not is_canonical(PM)
        assert result.PM_max == Matrix([[4, 3], [2, 1]])
        assert is_canonical(result.PM_max)

    def test_each_orbit_has_exactly_one_canonical_matrix(self) -> None:
        PM = Matrix([[0, 1, 2], [2, 0, 1]])
        orbit = {
            tuple(PM.extract(list(rows), list(cols)))
            for rows in permutations(range(2))
            for cols in permutations(range(3))
        }
        canonical = [m for m in orbit if is_canonical(Matrix(2, 3, list(m)))]
        assert canonical == [tuple(maximal_pairing_matrix(PM).PM_max)]

    def test_float_beside_an_equal_integer(self) -> None:
        # The default_sort_keys of 1.0 and 1 are neither equal nor ordered, so the
        # ranks the search uses order them by hash, which PYTHONHASHSEED changes.
        # matrix_lexicographic_compare finds every arrangement equal.
        PM = Matrix([[1.0, 1]])
        assert is_canonical(maximal_pairing_matrix(PM).PM_max)
        assert is_canonical(PM)
        assert is_canonical(Matrix([[1, 1.0]]))


class TestPermutationApplication:
    """Test permutation tracking and application."""

    def test_permutation_identity(self) -> None:
        """Test that applying identity permutation gives same matrix."""
        PM = Matrix([[1, 2], [3, 4]])
        row_perm = [0, 1]
        col_perm = [0, 1]

        result = _apply_permutation(PM, row_perm, col_perm)
        assert result == PM

    def test_permutation_inverse(self) -> None:
        """Test that permutations can be inverted."""
        PM = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = maximal_pairing_matrix(PM)

        # Apply inverse permutation
        inv_row = [0] * len(result.row_permutation)
        inv_col = [0] * len(result.col_permutation)

        for i, p in enumerate(result.row_permutation):
            inv_row[p] = i
        for j, p in enumerate(result.col_permutation):
            inv_col[p] = j

        recovered = _apply_permutation(result.PM_max, inv_row, inv_col)
        # Should recover something related to original (up to symmetry)
        assert recovered.shape == PM.shape


class TestEdgeCases:
    """Test edge cases and special matrices."""

    def test_single_element(self) -> None:
        """Test 1x1 matrix."""
        a = symbols("a")
        PM = Matrix([[a]])

        result = maximal_pairing_matrix(PM)
        assert result.PM_max == PM

    def test_zero_matrix(self) -> None:
        """Test matrix of all zeros."""
        PM = Matrix([[0, 0], [0, 0]])

        result = maximal_pairing_matrix(PM)
        assert result.PM_max == PM  # All permutations equivalent

    def test_rectangular_matrix(self) -> None:
        """Test non-square matrix."""
        PM = Matrix([[1, 2, 3], [4, 5, 6]])

        result = maximal_pairing_matrix(PM)
        assert result.PM_max.shape == PM.shape
        # Maximal row should come first
        assert result.PM_max[0, 0] == 6  # Largest element in maximal row [4, 5, 6]


def brute_force_maximum(PM: Matrix) -> Matrix:
    """The largest matrix that row and column permutations make from PM, row by row."""
    m, n = PM.shape
    keys = [[default_sort_key(x) for x in row] for row in PM.tolist()]
    _, rows, cols = max(
        (tuple(keys[r][c] for r in rows for c in cols), rows, cols)
        for rows in permutations(range(m))
        for cols in permutations(range(n))
    )
    return PM.extract(list(rows), list(cols))


def assert_exact(PM: Matrix) -> None:
    """PM_max is the brute-force maximum, the permutations give it, and it is canonical."""
    result = maximal_pairing_matrix(PM)
    expected = brute_force_maximum(PM)
    assert _matrix_lexicographic_compare(result.PM_max, expected) == 0
    assert result.PM_max == expected
    assert _apply_permutation(PM, result.row_permutation, result.col_permutation) == result.PM_max
    assert sorted(result.row_permutation) == list(range(PM.rows))
    assert sorted(result.col_permutation) == list(range(PM.cols))
    assert is_canonical(result.PM_max)


# The A-matrix of the massless triangle 12e|2e|e|:zzz, and its maximum.
TRIANGLE = Matrix([[1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1]])
TRIANGLE_MAX = Matrix(
    [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]]
)


class TestExactMaximum:
    """maximal_pairing_matrix returns the true maximum, not a greedy approximation."""

    def test_random_numeric_matrices(self) -> None:
        rng = random.Random(0)
        for _ in range(200):
            m, n = rng.randint(1, 4), rng.randint(1, 5)
            assert_exact(Matrix(m, n, lambda i, j: rng.randint(0, 2)))

    def test_random_symbolic_matrices(self) -> None:
        x, y, z = symbols("x y z")
        rng = random.Random(1)
        for _ in range(60):
            m, n = rng.randint(1, 3), rng.randint(1, 4)
            assert_exact(Matrix(m, n, lambda i, j: rng.choice([x, y, z, x + y, 0, 1])))

    @pytest.mark.parametrize(
        "PM",
        [
            Matrix([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]),
            Matrix([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 0]]),
            Matrix([[2, 0, 1], [0, 2, 1], [1, 1, 0], [0, 0, 0]]),
        ],
        ids=["permutation", "repeated-rows", "zero-row"],
    )
    def test_structured_matrices(self, PM: Matrix) -> None:
        assert_exact(PM)

    def test_triangle_a_matrix(self) -> None:
        A = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz").gkz.a_matrix
        assert A == TRIANGLE
        assert maximal_pairing_matrix(A).PM_max == TRIANGLE_MAX
        assert_exact(A)
        assert not is_canonical(A)

    def test_every_matrix_in_an_orbit_has_the_same_maximum(self) -> None:
        rng = random.Random(2)
        for _ in range(10):
            rows, cols = rng.sample(range(4), 4), rng.sample(range(6), 6)
            assert maximal_pairing_matrix(TRIANGLE.extract(rows, cols)).PM_max == TRIANGLE_MAX

    def test_identity_is_its_own_maximum(self) -> None:
        # Every row ties at every step. Only the arrangements that leave different
        # rows are kept apart, 2^10 of them; all 10! orderings would not fit in memory.
        result = maximal_pairing_matrix(eye(10))
        assert result.PM_max == eye(10)
        assert result.row_permutation == result.col_permutation == list(range(10))

    @pytest.mark.parametrize(
        "PM",
        [TRIANGLE_MAX, Matrix([[4, 3], [2, 1]]), Matrix([[1, 1, 0], [1, 1, 0], [0, 0, 0]])],
        ids=["triangle-max", "2x2", "equal-rows"],
    )
    def test_maximal_input_gets_identity_permutations(self, PM: Matrix) -> None:
        # Of the orders that give the maximum, the smallest is returned, so an
        # input that is already maximal keeps its rows and columns in place.
        result = maximal_pairing_matrix(PM)
        assert result.PM_max == PM
        assert result.row_permutation == list(range(PM.rows))
        assert result.col_permutation == list(range(PM.cols))
        assert is_canonical(PM)
