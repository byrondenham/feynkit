"""Tests for the polynomial helpers in feynkit.polynomials.operations.

Every expected value below is computed by hand and stated in the test, so a
failure points at the library rather than at a re-implementation of it.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import PolynomialError
from feynkit.polynomials import (
    extract_coefficient,
    homogenise_polynomial,
    simplify_rational_function,
)


@pytest.fixture  # type: ignore[misc]
def xy() -> Generator[tuple[sp.Symbol, sp.Symbol], None, None]:
    """Two plain variables for the small worked examples."""
    yield sp.symbols("x y")


@pytest.fixture  # type: ignore[misc]
def bubble_g() -> Generator[tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol], None, None]:
    """Lee-Pomeransky polynomial of the massless bubble.

    G = u_1 + u_2 - s u_1 u_2 / mu^2, together with its variables and the two
    kinematic symbols that appear in the coefficients.
    """
    integral = FeynmanIntegral.from_cnickel("11e|e|:zz")
    g = sp.expand(integral.symanzik.g)
    variables = list(integral.symanzik.lp_parameters)
    yield g, variables, sp.Symbol("s", real=True), integral.graph.energy_scale


class TestHomogenisePolynomial:
    def test_constant_term_picks_up_the_full_power_of_h(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """3 x^2 y + 2 x y^2 + 5 has total degree 3.

        The two cubic terms are already of top degree, so they are untouched;
        the constant 5 is of degree 0 and gains h^3. By hand the result is
        3 x^2 y + 2 x y^2 + 5 h^3.
        """
        x, y = xy
        h = sp.Symbol("h", positive=True)

        result = homogenise_polynomial(3 * x**2 * y + 2 * x * y**2 + 5, [x, y])

        assert sp.expand(result - (3 * x**2 * y + 2 * x * y**2 + 5 * h**3)) == 0

    def test_every_monomial_of_the_result_has_the_same_total_degree(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """x^2 + y has degree 2, so y gains one power of h and x^2 none."""
        x, y = xy
        h = sp.Symbol("h", positive=True)

        result = homogenise_polynomial(x**2 + y, [x, y])

        assert sp.expand(result - (x**2 + h * y)) == 0
        degrees = {sum(monom) for monom in sp.Poly(result, x, y, h).monoms()}
        assert degrees == {2}

    def test_already_homogeneous_polynomial_gains_no_h(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """x^2 + x y + y^2 is homogeneous of degree 2, so h^0 multiplies each term."""
        x, y = xy
        poly = x**2 + x * y + y**2

        result = homogenise_polynomial(poly, [x, y])

        assert sp.expand(result - poly) == 0
        assert sp.Symbol("h", positive=True) not in result.free_symbols

    def test_bubble_lee_pomeransky_polynomial(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """G = u_1 + u_2 - s u_1 u_2 / mu^2 has degree 2 in (u_1, u_2).

        The two linear terms each gain one power of h and the quadratic term is
        left alone, so by hand the homogenisation is
        h u_1 + h u_2 - s u_1 u_2 / mu^2.
        """
        g, variables, s, mu = bubble_g
        u1, u2 = variables
        h = sp.Symbol("h", positive=True)

        result = homogenise_polynomial(g, variables)

        assert sp.expand(result - (h * u1 + h * u2 - s * u1 * u2 / mu**2)) == 0

    def test_non_polynomial_input_raises(self, xy: tuple[sp.Symbol, sp.Symbol]) -> None:
        """1/x is not a polynomial in x, so sympy's Poly constructor fails."""
        x, _ = xy

        with pytest.raises(PolynomialError, match="Failed to homogenise polynomial"):
            homogenise_polynomial(1 / x, [x])


class TestExtractCoefficient:
    def test_extracts_each_coefficient_of_a_worked_example(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """For 3 x^2 y + 2 x y^2 + 5 the coefficients are 3, 2 and 5."""
        x, y = xy
        poly = 3 * x**2 * y + 2 * x * y**2 + 5

        assert extract_coefficient(poly, [x, y], (2, 1)) == 3
        assert extract_coefficient(poly, [x, y], (1, 2)) == 2
        assert extract_coefficient(poly, [x, y], (0, 0)) == 5

    def test_absent_monomial_gives_exactly_zero(self, xy: tuple[sp.Symbol, sp.Symbol]) -> None:
        """x^5 y^5 does not occur in the polynomial, so the answer is the integer 0."""
        x, y = xy
        poly = 3 * x**2 * y + 2 * x * y**2 + 5

        result = extract_coefficient(poly, [x, y], (5, 5))

        assert result == 0
        assert isinstance(result, sp.Integer)

    def test_variable_order_fixes_the_exponent_tuple(self, xy: tuple[sp.Symbol, sp.Symbol]) -> None:
        """With the variables given as [y, x] the monomial x^2 y is the tuple (1, 2)."""
        x, y = xy
        poly = 3 * x**2 * y + 2 * x * y**2

        assert extract_coefficient(poly, [y, x], (1, 2)) == 3
        assert extract_coefficient(poly, [y, x], (2, 1)) == 2

    def test_symbolic_coefficients_of_the_bubble(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """In G = u_1 + u_2 - s u_1 u_2 / mu^2 the u_1 u_2 coefficient is -s/mu^2."""
        g, variables, s, mu = bubble_g

        assert sp.simplify(extract_coefficient(g, variables, (1, 1)) + s / mu**2) == 0
        assert extract_coefficient(g, variables, (1, 0)) == 1
        assert extract_coefficient(g, variables, (0, 1)) == 1
        assert extract_coefficient(g, variables, (0, 0)) == 0

    def test_non_polynomial_input_raises(self, xy: tuple[sp.Symbol, sp.Symbol]) -> None:
        """sin(x) is not a polynomial in x, so the Poly constructor fails."""
        x, _ = xy

        with pytest.raises(PolynomialError, match="Failed to extract coefficient"):
            extract_coefficient(sp.sin(x), [x], (1,))


class TestSimplifyRationalFunction:
    def test_cancels_a_common_factor(self, xy: tuple[sp.Symbol, sp.Symbol]) -> None:
        """(x^2 - 1)/(x - 1) = (x - 1)(x + 1)/(x - 1) = x + 1."""
        x, _ = xy

        assert simplify_rational_function((x**2 - 1) / (x - 1)) == x + 1

    def test_puts_a_sum_of_reciprocals_over_a_common_denominator(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """1/x + 1/y = (x + y)/(x y)."""
        x, y = xy

        result = simplify_rational_function(1 / x + 1 / y)

        assert sp.simplify(result - (x + y) / (x * y)) == 0
        numerator, denominator = result.as_numer_denom()
        assert sp.expand(numerator - (x + y)) == 0
        assert sp.expand(denominator - x * y) == 0

    def test_final_cancel_undoes_the_intermediate_factorisation(
        self, xy: tuple[sp.Symbol, sp.Symbol]
    ) -> None:
        """x^2 + 2 x y + y^2 factors as (x + y)^2, but cancel re-expands it.

        The function applies simplify, then factor, then cancel, and cancel
        returns a quotient with an expanded numerator. The result is therefore
        the expanded sum, not the square.
        """
        x, y = xy

        result = simplify_rational_function(x**2 + 2 * x * y + y**2)

        assert result == x**2 + 2 * x * y + y**2
        assert result.is_Add
        assert not result.is_Pow

    def test_returns_the_input_unchanged_when_simplification_fails(self) -> None:
        """An object sympy cannot handle falls through to the except branch."""

        class NotAnExpression:
            pass

        value = NotAnExpression()

        assert simplify_rational_function(value) is value  # type: ignore[arg-type]
