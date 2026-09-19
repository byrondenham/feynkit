"""
Variable transformations for polynomial manipulations.

Provides functions to perform common variable transformations used in
Feynman integral computations, such as inversions and rescalings.
"""

from __future__ import annotations

import sympy as sp


def invert_variables(
    expr: sp.Expr,
    variables: list[sp.Symbol],
) -> sp.Expr:
    """
    Apply inversion transformation: x_i -> 1/x_i to all variables.

    Parameters
    ----------
    expr : sp.Expr
        Expression to transform.
    variables : List[sp.Symbol]
        Variables to invert.

    Returns
    -------
    sp.Expr
        Transformed expression with x_i -> 1/x_i.

    Examples
    --------
    >>> a1, a2 = sp.symbols('a1 a2', positive=True)
    >>> expr = a1 + a2
    >>> result = invert_variables(expr, [a1, a2])
    >>> print(result)
    1/a1 + 1/a2
    """
    inversion_map = {var: 1 / var for var in variables}
    return expr.subs(inversion_map)


def rescale_variables(
    expr: sp.Expr,
    variable_map: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """
    Apply arbitrary rescaling to variables.

    Parameters
    ----------
    expr : sp.Expr
        Expression to transform.
    variable_map : Dict[sp.Symbol, sp.Expr]
        Dictionary mapping variables to their rescaled expressions.

    Returns
    -------
    sp.Expr
        Transformed expression.

    Examples
    --------
    >>> a1, a2, lambda_var = sp.symbols('a1 a2 lambda', positive=True)
    >>> expr = a1**2 + a2**2
    >>> # Rescale: a1 -> lambda*a1, a2 -> lambda*a2
    >>> result = rescale_variables(expr, {a1: lambda_var*a1, a2: lambda_var*a2})
    >>> print(result)
    lambda**2*a1**2 + lambda**2*a2**2
    """
    return expr.subs(variable_map)


def projective_transformation(
    expr: sp.Expr,
    _variables: list[sp.Symbol],
    normalisation_var: sp.Symbol,
) -> sp.Expr:
    """
    Apply projective transformation by setting one variable to 1.

    Parameters
    ----------
    expr : sp.Expr
        Homogeneous expression to transform.
    variables : List[sp.Symbol]
        All variables in the expression.
    normalisation_var : sp.Symbol
        Variable to set to 1 (typically the last one).

    Returns
    -------
    sp.Expr
        Expression in projective coordinates.

    Notes
    -----
    This is commonly used when working with homogeneous polynomials
    to reduce to projective space by setting one coordinate to 1.

    Examples
    --------
    >>> a1, a2, a3 = sp.symbols('a1 a2 a3', positive=True)
    >>> expr = a1**2 + a2**2 + a3**2
    >>> result = projective_transformation(expr, [a1, a2, a3], a3)
    >>> print(result)
    a1**2 + a2**2 + 1
    """
    return expr.subs(normalisation_var, sp.Integer(1))
