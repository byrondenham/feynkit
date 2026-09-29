"""Tests for the polytope-equivalence verbs."""

from sympy import Matrix

from feynkit import FeynmanIntegral, PolytopeEquivalence
from feynkit.normal_forms import is_affinely_equivalent, is_point_config_equivalent


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


class TestBelowFullDimension:
    """The witness is the lift of the map between the lattice charts, so it is invertible."""

    def test_three_points_and_themselves(self) -> None:
        # A triangle in R^4. Solving in R^4 left free parameters, set to 0: the witness
        # for the points and themselves was singular.
        points = Matrix([[1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]])
        rows = {tuple(points.row(i)) for i in range(points.rows)}
        for test in (is_affinely_equivalent, is_point_config_equivalent):
            result = test(points, points)
            assert result.equivalent
            assert result.determinant != 0
            M, t = result.witness_map, result.translation
            assert {tuple(M * points.row(i).T + t) for i in range(points.rows)} == rows

    def test_every_segment(self) -> None:
        assert is_affinely_equivalent(Matrix([[0, 0], [1, 1]]), Matrix([[0, 0], [3, 6]])).equivalent

    def test_unimodular_image(self) -> None:
        points = FeynmanIntegral.from_cnickel("012e|2e|e|:znnn").newton_polytope.points
        U = Matrix([[0, 1, 0, 0], [1, 1, 0, 0], [0, 2, 1, 0], [1, 0, 1, 1]])
        t = Matrix([3, -1, 2, 0])
        image = [tuple(U * Matrix(p) + t) for p in points]
        for test in (is_affinely_equivalent, is_point_config_equivalent):
            result = test(Matrix(points), Matrix(image))
            assert result.equivalent
            assert abs(result.determinant) == 1
