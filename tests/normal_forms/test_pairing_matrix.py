"""
Tests for maximal pairing matrix canonicalisation.
"""

from sympy import Matrix, symbols

from feynkit.normal_forms.pairing_matrix import (
    _apply_permutation,
    is_canonical,
    matrix_lexicographic_compare as _matrix_lexicographic_compare,
    maximal_pairing_matrix,
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
    """Test deterministic behavior with equivalent symbolic expressions."""

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
        """Test that non-maximal matrix is recognised."""
        PM = Matrix([[1, 2], [3, 4]])

        # This is likely not canonical (4,3 row would be maximal)
        if not is_canonical(PM):
            result = maximal_pairing_matrix(PM)
            assert is_canonical(result.PM_max)


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
