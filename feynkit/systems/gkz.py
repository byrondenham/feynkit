"""
GKZ (Gelfand-Kapranov-Zelevinsky) hypergeometric system construction.

This module constructs the GKZ A-hypergeometric system associated with a Feynman
integral in Lee-Pomeransky representation. The GKZ system provides a system of
partial differential equations satisfied by the Feynman integral.
"""

import sympy as sp

from ..core.exceptions import MatrixError, PolynomialError
from .monomial import extract_monomial_support


def construct_gkz_matrix(
    polynomial: sp.Expr,
    variables: list[sp.Symbol],
) -> sp.Matrix:
    """
    Construct the GKZ A-matrix from a polynomial.

    The A-matrix is fundamental to the GKZ hypergeometric system. Each column
    corresponds to a monomial in the polynomial's support, encoding its exponent
    vector with a homogenising row of ones prepended.

    Parameters
    ----------
    polynomial : sp.Expr
        Polynomial from which to construct the A-matrix.
        Typically the Lee-Pomeransky G(u) = U(u) + F(u).
    variables : List[sp.Symbol]
        Variables of the polynomial (Lee-Pomeransky parameters).

    Returns
    -------
    sp.Matrix
        The A-matrix of size (n+1) x m, where:
        - n = len(variables) = number of variables
        - m = number of monomials in support

        Structure:
        Row 0: [1, 1, ..., 1] (homogenising row)
        Row i (i >= 1): [alpha_{i-1,1}, alpha_{i-1,2}, ..., alpha_{i-1,m}]
                        (i-th exponent for each monomial)

    Raises
    ------
    PolynomialError
        If monomial support extraction fails.
    MatrixError
        If matrix construction fails.

    Notes
    -----
    For a polynomial G(u) = sum_j c_j u^alpha_j with support {alpha_1, alpha_2, ..., alpha_m},
    the A-matrix is:

            [     1        1       ...      1     ]
            | alpha1[0] alpha2[0]  ...  alpham[0] |
            | alpha1[1] alpha2[1]  ...  alpham[1] |
            |     :        :                :     |
            [alpha1[n-1]alpha2[n-1]... alpham[n-1]]

    The columns of A generate a monoid in mathbb{Z}^{n+1}, and the GKZ system
    studies functions with prescribed homogeneity under this monoid action.

    Examples
    --------
    >>> import sympy as sp
    >>> u1, u2 = sp.symbols('u1 u2', nonnegative=True)
    >>> poly = u1**2 + u1*u2 + u2**2
    >>> A = construct_gkz_matrix(poly, [u1, u2])
    >>> print(A)
    Matrix([
        [1, 1, 1],      # Homogenising row
        [2, 1, 0],      # u1 exponents
        [0, 1, 2]       # u2 exponents
    ])

    References
    ----------
    .. [1] Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1989).
            "Hypergeometric functions and toric varieties."
            Funct. Anal. Appl. 23, 94-106.
    .. [2] de la Cruz, L. (2019). "Feynman integrals as A-hypergeometric functions."
            JHEP 12, 123.
    """
    try:
        # Get monomial support: list of (exponent_tuple, coefficient)
        support = extract_monomial_support(polynomial, variables)

        if not support:
            raise PolynomialError("Polynomial has empty support")

        num_monomials = len(support)  # Number of columns (m)
        num_variables = len(variables)  # Number of variables (n)

        # Initialise (n+1) x m matrix
        A = sp.zeros(num_variables + 1, num_monomials)

        # Fill the A-matrix column by column
        for j, (alpha, _) in enumerate(support):
            # First row: homogenising ones
            A[0, j] = 1

            # Remaining rows: exponent components
            for i in range(num_variables):
                A[i + 1, j] = alpha[i]

        return sp.Matrix(A)

    except PolynomialError:
        raise
    except Exception as e:
        raise MatrixError(f"Failed to construct GKZ A-matrix: {e}") from e


def construct_gkz_matrix_from_exponents(
    exponent_vecs: list[tuple[int, ...]],
    n_vars: int,
) -> sp.Matrix:
    """
    Build the GKZ A-matrix directly from pre-computed exponent vectors.

    Avoids symbolic polynomial construction entirely; each column is
    ``[1, alpha[0], alpha[1], ..., alpha[n-1]]``.
    """
    if not exponent_vecs:
        raise PolynomialError("Empty exponent vector list")
    m = len(exponent_vecs)
    A = sp.zeros(n_vars + 1, m)
    for j, alpha in enumerate(exponent_vecs):
        A[0, j] = 1
        for i, exp in enumerate(alpha):
            A[i + 1, j] = exp
    return A


def validate_gkz_matrix(A: sp.Matrix) -> None:
    """
    Validate that a matrix is a valid GKZ A-matrix.

    Parameters
    ----------
    A : sp.Matrix
        Matrix to validate.

    Raises
    ------
    MatrixError
        If the matrix is not a valid GKZ A-matrix.

    Notes
    -----
    A valid GKZ A-matrix must:
    - Have at least 2 rows
    - Have first row all ones
    - Have all remaining entries non-negative integers
    """
    if A.rows < 2:
        raise MatrixError("GKZ A-matrix must have at least 2 rows")

    if A.cols < 1:
        raise MatrixError("GKZ A-matrix must have at least 1 column")

    # Check first row is all ones
    for j in range(A.cols):
        if A[0, j] != 1:
            raise MatrixError(f"First row must be all ones, but A[0,{j}] = {A[0, j]}")

    # Check remaining entries are non-negative integers
    for i in range(1, A.rows):
        for j in range(A.cols):
            value = A[i, j]
            if not value.is_integer or value < 0:
                raise MatrixError(
                    f"A-matrix entries must be non-negative integers, but A[{i},{j}] = {value}"
                )
