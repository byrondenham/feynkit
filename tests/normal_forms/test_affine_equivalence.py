"""Tests for the polytope-equivalence verbs."""

from sympy import Matrix

from feynkit import PolytopeEquivalence
from feynkit.normal_forms import is_affinely_equivalent


class TestAffineEquivalence:
    """Tests for the broader rational equivalence (brute-force backend)."""

    def test_self_equivalence(self) -> None:
        points = Matrix([[0, 0], [1, 0], [0, 1]])
        result = is_affinely_equivalent(points, points)
        assert isinstance(result, PolytopeEquivalence)
        assert result.equivalent is True
        assert result.relation == "affine_polytope"

    def test_translated_and_scaled_polytope(self) -> None:
        """A 2D triangle and its 2x-scaled, translated image are affinely equivalent."""
        points_a = Matrix([[0, 0], [1, 0], [0, 1]])
        points_b = Matrix([[2, 1], [4, 1], [2, 3]])
        assert is_affinely_equivalent(points_a, points_b).equivalent is True

    def test_distinguishes_collinear_from_triangle(self) -> None:
        triangle = Matrix([[0, 0], [1, 0], [0, 1]])
        collinear = Matrix([[2, 1], [5, 1], [8, 1]])
        assert is_affinely_equivalent(triangle, collinear).equivalent is False

    def test_unordered_correspondence(self) -> None:
        """Reordered vertices should still be detected as the same polytope."""
        points_a = Matrix([[0, 0], [1, 0], [0, 1]])
        points_b = Matrix([[2, 1], [2, 3], [4, 1]])
        assert is_affinely_equivalent(points_a, points_b).equivalent is True
