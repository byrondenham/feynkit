"""
Monomial support extraction and analysis.

Provides functions to extract and analyse the monomial support of polynomials,
which forms the basis for constructing GKZ A-matrices.
"""

from __future__ import annotations

import sympy as sp

from ..core.exceptions import PolynomialError


def extract_monomial_support(
    polynomial: sp.Expr,
    variables: list[sp.Symbol],
) -> list[tuple[tuple[int, ...], sp.Expr]]:
    """
    Extract the monomial support of a polynomial.

    The monomial support consists of all monomial exponent vectors (multi-indices)
    that appear in the polynomial with nonzero coefficients. This encodes the
    combinatorial structure needed for the GKZ A-matrix.

    Parameters
    ----------
    polynomial : sp.Expr
        Polynomial expression to analyse (typically the Lee-Pomeransky G polynomial).
    variables : List[sp.Symbol]
        List of variables with respect to which the polynomial is expanded.
        These are typically the Lee-Pomeransky parameters u_i.

    Returns
    -------
    List[Tuple[Tuple[int, ...], sp.Expr]]
        List of (exponent_vector, coefficient) pairs, where:
        - exponent_vector: Tuple of non-negative integers (alpha_1, alpha_2, ..., alpha_n)
          representing the monomial u_1^alpha_1 * ... * u_n^alpha_n
        - coefficient: Simplified symbolic coefficient for this monomial

    Raises
    ------
    PolynomialError
        If polynomial expansion or support extraction fails.

    Notes
    -----
    For a polynomial G(u) = sum c_alpha * u^alpha, this function returns the set:
        {(alpha, c_alpha) : c_alpha != 0}

    The monomial support forms the columns of the GKZ A-matrix (after adding
    a homogenising row of ones).

    Examples
    --------
    >>> import sympy as sp
    >>> u1, u2 = sp.symbols('u1 u2', nonnegative=True)
    >>> poly = u1**2 + 3*u1*u2 + 2*u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> for exp_vec, coeff in support:
    ...     print(f"{exp_vec}: {coeff}")
    (2, 0): 1
    (1, 1): 3
    (0, 2): 2

    >>> # For Symanzik polynomial
    >>> a1, a2, a3 = sp.symbols('a1 a2 a3', nonnegative=True)
    >>> U = a1*a2 + a2*a3 + a3*a1  # Triangle graph U polynomial
    >>> support = extract_monomial_support(U, [a1, a2, a3])
    >>> # Returns: [((1,1,0), 1), ((0,1,1), 1), ((1,0,1), 1)]

    References
    ----------
    .. [1] Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994).
            "Discriminants, Resultants and Multidimensional Determinants."
            Birkhäuser.
    """
    try:
        var_idx = {v: i for i, v in enumerate(variables)}
        n = len(variables)
        support: dict[tuple[int, ...], sp.Expr] = {}
        for term in sp.Add.make_args(sp.expand(polynomial)):
            exponents = [0] * n
            coeff: sp.Expr = sp.Integer(1)
            for factor in sp.Mul.make_args(term):
                if factor.is_Pow:
                    base, exp = factor.as_base_exp()
                    if base in var_idx:
                        exponents[var_idx[base]] = int(exp)
                    else:
                        coeff = coeff * factor
                elif factor in var_idx:
                    exponents[var_idx[factor]] = 1
                else:
                    coeff = coeff * factor
            key = tuple(exponents)
            support[key] = support.get(key, sp.Integer(0)) + coeff
        return [(k, v) for k, v in support.items() if v != 0]

    except Exception as e:
        raise PolynomialError(f"Failed to extract monomial support: {e}") from e


def count_monomials(support: list[tuple[tuple[int, ...], sp.Expr]]) -> int:
    """
    Count the number of monomials in the support.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support from extract_monomial_support().

    Returns
    -------
    int
        Number of monomials in the support.

    Examples
    --------
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = u1**2 + u1*u2 + u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> count_monomials(support)
    3
    """
    return len(support)


def get_max_degree(support: list[tuple[tuple[int, ...], sp.Expr]]) -> int:
    """
    Get the maximum total degree of monomials in the support.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support.

    Returns
    -------
    int
        Maximum total degree.

    Examples
    --------
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = u1**3 + u1*u2 + u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> get_max_degree(support)
    3
    """
    if not support:
        return 0

    return max(sum(exponents) for exponents, _ in support)


def filter_support_by_degree(
    support: list[tuple[tuple[int, ...], sp.Expr]],
    target_degree: int,
) -> list[tuple[tuple[int, ...], sp.Expr]]:
    """
    Filter support to include only monomials of a specific total degree.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support.
    target_degree : int
        Target total degree.

    Returns
    -------
    List[Tuple[Tuple[int, ...], sp.Expr]]
        Filtered support containing only monomials with total degree = target_degree.

    Examples
    --------
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = u1**2 + u1*u2 + u2**2 + u1
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> degree_2 = filter_support_by_degree(support, 2)
    >>> len(degree_2)
    3
    """
    return [(exp_vec, coeff) for exp_vec, coeff in support if sum(exp_vec) == target_degree]
