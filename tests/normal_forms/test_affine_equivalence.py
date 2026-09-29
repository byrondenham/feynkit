"""Tests for the polytope-equivalence verbs."""

import pytest
from sympy import Matrix, Rational, eye

from feynkit import FeynmanIntegral, PolytopeEquivalence, ValidationError
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

    def test_a_repeated_point(self) -> None:
        # The free parameters set to 0 gave a singular witness for a point repeated.
        a, b = Matrix([[1, 1], [1, 1]]), Matrix([[2, 3], [2, 3]])
        for test in (is_affinely_equivalent, is_point_config_equivalent):
            result = test(a, b)
            assert result.equivalent
            assert result.witness_map == eye(2)
            assert result.translation == Matrix([1, 2])
            assert result.determinant == 1


def _maps_onto(result: PolytopeEquivalence, points_a: Matrix, points_b: Matrix) -> bool:
    """Whether the witness maps the rows of points_a onto those of points_b, as multisets."""
    M, t = result.witness_map, result.translation
    image = sorted(tuple(M * points_a.row(i).T + t) for i in range(points_a.rows))
    return image == sorted(tuple(points_b.row(i)) for i in range(points_b.rows))


R = Rational


class TestRationalPoints:
    """Each configuration is scaled to integers, and the witness scaled back."""

    @pytest.mark.parametrize(
        ("points_a", "points_b"),
        [
            # A triangle in full dimension, whose coordinates were read as their integer parts.
            ([[0, 0], [R(1, 2), 0], [0, R(1, 3)]], [[0, 0], [1, 0], [0, 1]]),
            # A segment with its midpoint, and a triangle, below full dimension.
            ([[0, 0], [R(1, 2), R(1, 2)], [1, 1]], [[0, 0], [1, 1], [2, 2]]),
            ([[0, 0, 0], [R(1, 3), 0, 1], [0, R(2, 3), 1]], [[1, 1, 0], [2, 1, 0], [1, 2, 0]]),
        ],
    )
    def test_both_affine_tests(self, points_a: list, points_b: list) -> None:
        a, b = Matrix(points_a), Matrix(points_b)
        for test in (is_affinely_equivalent, is_point_config_equivalent):
            result = test(a, b)
            assert result.equivalent
            assert result.determinant != 0
            assert _maps_onto(result, a, b)
            reverse = test(b, a)
            assert reverse.equivalent
            assert _maps_onto(reverse, b, a)

    def test_floats_that_are_not_integers(self) -> None:
        # They are rejected, not rounded; integral floats count as integers.
        unit = Matrix([[0, 0], [1, 0], [0, 1]])
        with pytest.raises(ValidationError, match="not an exact rational"):
            is_affinely_equivalent(Matrix([[0, 0], [0.5, 0], [0, 1]]), unit)
        with pytest.raises(ValidationError, match="not an exact rational"):
            is_point_config_equivalent(Matrix([[0, 0], [0.5, 0.5]]), Matrix([[0, 0], [1, 1]]))
        assert is_affinely_equivalent(Matrix([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]), unit).equivalent
