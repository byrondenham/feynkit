"""Tests for the GKZ A-matrix helpers in feynkit.systems.gkz.

Each column of the A-matrix is a monomial of the Lee-Pomeransky polynomial,
written as [1, alpha_1, ..., alpha_n] with the homogenising 1 on top. The order
of the columns follows the order in which sympy enumerates the terms of the
polynomial, so the assertions below compare column sets rather than matrices
where the order is not part of the contract.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import MatrixError, PolynomialError
from feynkit.systems import construct_gkz_matrix, extract_monomial_support
from feynkit.systems.gkz import construct_gkz_matrix_from_exponents, validate_gkz_matrix


def columns(matrix: sp.Matrix) -> set[tuple[int, ...]]:
    """Return the columns of a matrix as a set of integer tuples."""
    return {tuple(int(entry) for entry in matrix.col(j)) for j in range(matrix.cols)}


@pytest.fixture  # type: ignore[misc]
def bubble() -> Generator[tuple[sp.Expr, list[sp.Symbol]], None, None]:
    """G and the Lee-Pomeransky variables of the massless bubble.

    G = u_1 + u_2 - s u_1 u_2 / mu^2.
    """
    integral = FeynmanIntegral.from_cnickel("11e|e|:zz")
    yield sp.expand(integral.symanzik.g), list(integral.symanzik.lp_parameters)


@pytest.fixture  # type: ignore[misc]
def one_mass_triangle() -> Generator[tuple[sp.Expr, list[sp.Symbol]], None, None]:
    """G and the Lee-Pomeransky variables of the one-mass triangle.

    With only the first edge massive,

        G = u_1 + u_2 + u_3
            + m_1^2 u_1^2 / mu^2
            + (m_1^2 - p1^2) u_1 u_2 / mu^2
            + (m_1^2 - p2^2) u_1 u_3 / mu^2
            - p3^2 u_2 u_3 / mu^2,

    which has seven monomials.
    """
    integral = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
    yield sp.expand(integral.symanzik.g), list(integral.symanzik.lp_parameters)


class TestConstructGkzMatrix:
    def test_bubble_columns(self, bubble: tuple[sp.Expr, list[sp.Symbol]]) -> None:
        """The monomials u_1 u_2, u_1 and u_2 give the columns
        (1, 1, 1), (1, 1, 0) and (1, 0, 1)."""
        g, variables = bubble

        a_matrix = construct_gkz_matrix(g, variables)

        assert a_matrix.shape == (3, 3)
        assert columns(a_matrix) == {(1, 1, 1), (1, 1, 0), (1, 0, 1)}

    def test_one_mass_triangle_columns(
        self, one_mass_triangle: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Three linear and four quadratic monomials give seven columns.

        The quadratic ones are u_1^2 (from the mass on edge 1), u_1 u_2,
        u_1 u_3 and u_2 u_3; the missing u_2^2 and u_3^2 are exactly the
        massless edges.
        """
        g, variables = one_mass_triangle

        a_matrix = construct_gkz_matrix(g, variables)

        assert a_matrix.shape == (4, 7)
        assert columns(a_matrix) == {
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (1, 2, 0, 0),
            (1, 1, 1, 0),
            (1, 1, 0, 1),
            (1, 0, 1, 1),
        }

    def test_small_polynomial_columns(self) -> None:
        """u_1^2 + u_1 u_2 + u_2^2 gives (1, 2, 0), (1, 1, 1) and (1, 0, 2)."""
        u1, u2 = sp.symbols("u1 u2", nonnegative=True)

        a_matrix = construct_gkz_matrix(u1**2 + u1 * u2 + u2**2, [u1, u2])

        assert a_matrix.shape == (3, 3)
        assert columns(a_matrix) == {(1, 2, 0), (1, 1, 1), (1, 0, 2)}

    def test_first_row_is_all_ones(
        self, one_mass_triangle: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        g, variables = one_mass_triangle

        a_matrix = construct_gkz_matrix(g, variables)

        assert list(a_matrix.row(0)) == [1] * a_matrix.cols

    def test_column_order_follows_the_monomial_support(
        self, one_mass_triangle: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Column j is [1] followed by the exponent vector of monomial j."""
        g, variables = one_mass_triangle

        a_matrix = construct_gkz_matrix(g, variables)
        support = extract_monomial_support(g, variables)

        for j, (exponents, _) in enumerate(support):
            assert tuple(a_matrix.col(j)) == (1, *exponents)

    def test_coefficients_do_not_enter_the_matrix(
        self, bubble: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Only the exponents matter, so u_1 + u_2 + c u_1 u_2 has the bubble's
        A-matrix whatever the symbol c stands for."""
        g, variables = bubble
        u1, u2 = variables
        c = sp.Symbol("c", positive=True)

        assert columns(construct_gkz_matrix(u1 + u2 + c * u1 * u2, variables)) == columns(
            construct_gkz_matrix(g, variables)
        )

    def test_empty_support_raises(self) -> None:
        """The zero polynomial has no monomials at all."""
        x = sp.Symbol("x")

        with pytest.raises(PolynomialError, match="empty support"):
            construct_gkz_matrix(sp.Integer(0), [x])


class TestConstructGkzMatrixFromExponents:
    def test_builds_the_bubble_matrix_directly(self) -> None:
        """The exponent vectors (1, 1), (1, 0) and (0, 1) prepend a row of ones."""
        a_matrix = construct_gkz_matrix_from_exponents([(1, 1), (1, 0), (0, 1)], 2)

        assert a_matrix.shape == (3, 3)
        assert a_matrix == sp.Matrix([[1, 1, 1], [1, 1, 0], [1, 0, 1]])

    def test_agrees_with_the_polynomial_route(
        self, one_mass_triangle: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Feeding the support of G back in reproduces construct_gkz_matrix."""
        g, variables = one_mass_triangle
        support = extract_monomial_support(g, variables)

        direct = construct_gkz_matrix_from_exponents(
            [exponents for exponents, _ in support], len(variables)
        )

        assert direct == construct_gkz_matrix(g, variables)

    def test_extra_variables_give_zero_rows(self) -> None:
        """Declaring three variables for two-component exponents pads with zeros."""
        a_matrix = construct_gkz_matrix_from_exponents([(1, 1), (1, 0)], 3)

        assert a_matrix.shape == (4, 2)
        assert a_matrix == sp.Matrix([[1, 1], [1, 1], [1, 0], [0, 0]])

    def test_empty_exponent_list_raises(self) -> None:
        with pytest.raises(PolynomialError, match="Empty exponent vector list"):
            construct_gkz_matrix_from_exponents([], 2)


class TestValidateGkzMatrix:
    def test_accepts_the_bubble_matrix(self, bubble: tuple[sp.Expr, list[sp.Symbol]]) -> None:
        g, variables = bubble

        assert validate_gkz_matrix(construct_gkz_matrix(g, variables)) is None

    def test_accepts_the_one_mass_triangle_matrix(
        self, one_mass_triangle: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        g, variables = one_mass_triangle

        assert validate_gkz_matrix(construct_gkz_matrix(g, variables)) is None

    def test_rejects_a_single_row(self) -> None:
        """A GKZ matrix needs the homogenising row plus at least one variable row."""
        with pytest.raises(MatrixError, match="at least 2 rows"):
            validate_gkz_matrix(sp.Matrix([[1, 1, 1]]))

    def test_rejects_an_empty_column_set(self) -> None:
        with pytest.raises(MatrixError, match="at least 1 column"):
            validate_gkz_matrix(sp.zeros(3, 0))

    def test_rejects_a_first_row_that_is_not_all_ones(self) -> None:
        with pytest.raises(MatrixError, match=r"First row must be all ones"):
            validate_gkz_matrix(sp.Matrix([[1, 2], [0, 1]]))

    def test_rejects_a_negative_exponent(self) -> None:
        with pytest.raises(MatrixError, match="non-negative integers"):
            validate_gkz_matrix(sp.Matrix([[1, 1], [0, -1]]))

    def test_rejects_a_fractional_exponent(self) -> None:
        with pytest.raises(MatrixError, match="non-negative integers"):
            validate_gkz_matrix(sp.Matrix([[1, 1], [0, sp.Rational(1, 2)]]))

    def test_rejects_a_symbolic_exponent(self) -> None:
        """A free symbol is not known to be an integer, so it is rejected."""
        with pytest.raises(MatrixError, match="non-negative integers"):
            validate_gkz_matrix(sp.Matrix([[1, 1], [0, sp.Symbol("t")]]))
