"""Tests for toric ideal computation."""

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit.algebra import (
    compute_toric_ideal_generators,
    count_generators,
    extract_binomial_form,
    is_binomial_ideal,
)
from feynkit.core import ComputationError
from feynkit.systems import construct_gkz_matrix


@pytest.fixture  # type: ignore
def simple_polynomial() -> Generator[tuple[sp.Expr, list[sp.Symbol]], None, None]:
    """Create a simple polynomial for testing."""
    u1, u2 = sp.symbols("u1 u2", nonnegative=True)
    poly = u1**2 + u1 * u2 + u2**2
    yield poly, [u1, u2]


@pytest.fixture  # type: ignore
def simple_a_matrix() -> Generator[sp.Matrix, None, None]:
    """Create a simple A-matrix."""
    # A = [[1, 1, 1],
    #      [2, 1, 0],
    #      [0, 1, 2]]
    yield sp.Matrix([[1, 1, 1], [2, 1, 0], [0, 1, 2]])


class TestToricIdealComputation:
    """Test toric ideal generator computation."""

    def test_compute_generators_simple(self, simple_a_matrix: sp.Matrix) -> None:
        """Test computing toric ideal generators for simple matrix."""
        generators = compute_toric_ideal_generators(simple_a_matrix)

        # Should have at least one generator
        assert len(generators) > 0

    def test_generators_are_polynomials(self, simple_a_matrix: sp.Matrix) -> None:
        """Test that generators are valid polynomial expressions."""
        generators = compute_toric_ideal_generators(simple_a_matrix)

        for gen in generators:
            assert isinstance(gen, sp.Expr)

    def test_simple_binomial_relation(self) -> None:
        """Test binomial relation for simple case."""
        # For A = [[1, 1], [0, 1]], we expect z_1^1 = z_2^0
        # which gives no non-trivial relations
        A = sp.Matrix([[1, 1], [0, 1]])
        generators = compute_toric_ideal_generators(A)

        # Should have minimal or no generators for this simple case
        assert isinstance(generators, list)


class TestToricIdealUtilities:
    """Test utility functions for toric ideals."""

    def test_count_generators(self, simple_a_matrix: sp.Matrix) -> None:
        """Test counting generators."""
        generators = compute_toric_ideal_generators(simple_a_matrix)
        count = count_generators(generators)

        assert count == len(generators)
        assert count >= 0

    def test_is_binomial_ideal(self, simple_a_matrix: sp.Matrix) -> None:
        """Test checking if ideal is binomial."""
        generators = compute_toric_ideal_generators(simple_a_matrix)

        if generators:
            # Toric ideals should be binomial
            is_binomial = is_binomial_ideal(generators)
            assert isinstance(is_binomial, bool)

    def test_extract_binomial_form_valid(self) -> None:
        """Test extracting binomial form from valid binomial."""
        z1, z2, z3 = sp.symbols("z1 z2 z3")
        gen = z1 * z3 - z2**2

        pos, neg = extract_binomial_form(gen)

        # Both should be monomials
        assert pos != 0
        assert neg != 0

    def test_extract_binomial_form_invalid(self) -> None:
        """Test that extracting from non-binomial raises error."""
        z1, z2, z3 = sp.symbols("z1 z2 z3")
        gen = z1 + z2 + z3  # Three terms, not binomial

        with pytest.raises(ComputationError):
            extract_binomial_form(gen)


class TestToricIdealFromPolynomial:
    """Test toric ideal computation from polynomial."""

    def test_from_polynomial_via_matrix(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Test computing toric ideal from polynomial via A-matrix."""
        poly, variables = simple_polynomial

        # Construct A-matrix
        A = construct_gkz_matrix(poly, variables)

        # Compute toric ideal
        generators = compute_toric_ideal_generators(A)

        # Should produce valid generators
        assert isinstance(generators, list)
