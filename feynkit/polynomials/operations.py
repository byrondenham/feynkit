"""
Polynomial manipulation utilities for Feynman integrals.

Provides helper functions for common polynomial operations used throughout feynkit.
"""

from __future__ import annotations

import sympy as sp

from ..core.exceptions import PolynomialError


def homogenise_polynomial(poly: sp.Expr, variables: list[sp.Symbol]) -> sp.Expr:
    """
    Homogenise a polynomial by adding a homogenising variable.

    Parameters
    ----------
    poly : sp.Expr
        Polynomial to homogenise.
    variables : List[sp.Symbol]
        Variables in the polynomial.

    Returns
    -------
    sp.Expr
        Homogenised polynomial.

    Notes
    -----
    A polynomial is homogeneous if all terms have the same total degree.
    To homogenise, we introduce a new variable and multiply each term
    by appropriate powers to make all terms have the same degree.
    """
    try:
        P = sp.Poly(poly, *variables)
        max_degree = P.total_degree()

        # Create homogenising variable
        h = sp.Symbol("h", positive=True)

        # Homogenise each term
        result = sp.Integer(0)
        for monom, coeff in P.terms():
            term_degree = sum(monom)
            homogenising_power = max_degree - term_degree
            result += (
                coeff
                * sp.prod([v**e for v, e in zip(variables, monom)])
                * h**homogenising_power
            )

        return result
    except Exception as e:
        raise PolynomialError(f"Failed to homogenise polynomial: {e}") from e


def extract_coefficient(
    poly: sp.Expr,
    variables: list[sp.Symbol],
    target_monomial: tuple[int, ...],
) -> sp.Expr:
    """
    Extract the coefficient of a specific monomial from a polynomial.

    Parameters
    ----------
    poly : sp.Expr
        Polynomial from which to extract coefficient.
    variables : List[sp.Symbol]
        Variables in the polynomial (order matters).
    target_monomial : tuple[int, ...]
        Exponent tuple of the target monomial.

    Returns
    -------
    sp.Expr
        Coefficient of the target monomial, or 0 if not present.

    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> poly = 3*x**2*y + 2*x*y**2 + 5
    >>> coeff = extract_coefficient(poly, [x, y], (2, 1))
    >>> print(coeff)
    3
    """
    try:
        P = sp.Poly(poly, *variables)
        coeffs = dict(P.terms())
        return coeffs.get(target_monomial, sp.Integer(0))
    except Exception as e:
        raise PolynomialError(f"Failed to extract coefficient: {e}") from e


def simplify_rational_function(expr: sp.Expr) -> sp.Expr:
    """
    Simplify a rational function (ratio of polynomials).

    Parameters
    ----------
    expr : sp.Expr
        Expression to simplify.

    Returns
    -------
    sp.Expr
        Simplified expression.

    Notes
    -----
    This applies multiple simplification strategies:
    1. Standard simplify
    2. Factor
    3. Cancel common factors
    """
    try:
        # Apply various simplification methods
        result = sp.simplify(expr)
        result = sp.factor(result)
        result = sp.cancel(result)
        return result
    except Exception:
        # If simplification fails, return original
        return expr
