"""
Euler differential operators for GKZ hypergeometric systems.

Provides functions to generate Euler operators and the associated differential
equations that Feynman integrals satisfy.
"""

from __future__ import annotations

import sympy as sp

from ..core.exceptions import ValidationError


def create_euler_operators(
    a_matrix: sp.Matrix,
    z_variables: list[sp.Symbol],
) -> list[sp.Expr]:
    """
    Create Euler differential operators from the A-matrix.

    Each row of the A-matrix defines one Euler operator:
        Ehat_r = sum_j A[r,j] * z_j * partial/partial z_j

    Parameters
    ----------
    a_matrix : sp.Matrix
        The GKZ A-matrix of size (n+1) x m.
    z_variables : List[sp.Symbol]
        Differential variables [z_1, z_2, ..., z_m].

    Returns
    -------
    List[sp.Expr]
        List of Euler operators, one for each row of A.

    Raises
    ------
    ValidationError
        If number of z_variables doesn't match A-matrix columns.

    Notes
    -----
    The Euler operators act on a function Phi(z_1, ..., z_m) as:
        Ehat_r * Phi = sum_j A[r,j] * z_j * partial Phi/partial z_j

    Examples
    --------
    >>> import sympy as sp
    >>> A = sp.Matrix([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
    >>> z1, z2, z3 = sp.symbols('z1 z2 z3')
    >>> operators = create_euler_operators(A, [z1, z2, z3])
    >>> len(operators)
    3
    """
    num_rows = a_matrix.rows
    num_cols = a_matrix.cols

    if len(z_variables) != num_cols:
        raise ValidationError(
            f"Number of z variables ({len(z_variables)}) must match "
            f"number of A-matrix columns ({num_cols})"
        )

    # Define Phi as a symbolic function of all z variables
    phi = sp.Function("Phi")(*z_variables)

    operators = []

    for r in range(num_rows):
        # Construct Euler operator for row r
        operator = sum(
            a_matrix[r, j] * z_variables[j] * sp.Derivative(phi, z_variables[j])
            for j in range(num_cols)
        )
        operators.append(operator)

    return operators


def create_euler_equations(
    a_matrix: sp.Matrix,
    beta_parameters: list[sp.Expr],
    z_variables: list[sp.Symbol],
) -> list[sp.Equality]:
    """
    Generate Euler differential equations for the GKZ hypergeometric system.

    The Euler equations (also called A-hypergeometric equations or Horn equations) are
    a system of linear partial differential equations that the Feynman integral satisfies.
    Each row of the A-matrix gives one Euler operator equation.

    Parameters
    ----------
    a_matrix : sp.Matrix
        The GKZ A-matrix of size (n+1) x m encoding monomial exponents.
    beta_parameters : List[sp.Expr]
        Parameter vector of length (n+1), typically:
        - beta_0 = -D/2 (homogeneity of G^{-D/2} under an overall rescaling)
        - beta_i = -nu_i for i >= 1 (rescaling of the i-th Lee-Pomeransky variable)
        See de la Cruz (2019) eq. for I_{g_r}(kappa), kappa = (-d/2, -alpha), and
        Klausen (2020) Thm 3.1, which proves J(s^{a_b} z) = s^{-nu_b} J(z).
    z_variables : List[sp.Symbol]
        Variables for the differential equations (z_1, z_2, ..., z_m).
        These represent coefficients in the polynomial expansion.

    Returns
    -------
    List[sp.Equality]
        List of (n+1) differential equations of the form:
            sum_j A[r,j] * z_j * partial Phi/partial z_j = beta_r * Phi
        for r = 0, 1, ..., n.

    Raises
    ------
    ValidationError
        If dimensions are inconsistent.

    Notes
    -----
    The r-th Euler equation is:
        Ehat_r * Phi = beta_r * Phi
    where the Euler operator Ehat_r is defined as:
        Ehat_r = sum_{j=1}^m A[r,j] * z_j * partial/partial z_j

    These equations express homogeneity properties of the Feynman integral
    under rescalings associated with the monoid structure.

    For Feynman integrals:
    - The first equation (r=0) relates to overall dimensional analysis
    - The remaining equations (r >= 1) relate to shifts in propagator indices

    The function Phi(z_1, ..., z_m) represents the Feynman integral as a function
    of the monomial coefficients.

    Examples
    --------
    >>> import sympy as sp
    >>> A = sp.Matrix([[1, 1], [2, 0], [0, 2]])
    >>> beta = [sp.Symbol('b0'), sp.Symbol('b1'), sp.Symbol('b2')]
    >>> z1, z2 = sp.symbols('z1 z2')
    >>> equations = create_euler_equations(A, beta, [z1, z2])
    >>> len(equations)
    3

    References
    ----------
    .. [1] Saito, M., Sturmfels, B., Takayama, N. (2000).
            "Gröbner Deformations of Hypergeometric Differential Equations."
            Springer.
    .. [2] Klausen, R.P. (2022). "Hypergeometric Series Representations of
            Feynman Integrals by GKZ Hypergeometric Systems." arXiv:2205.14393.
    """
    num_rows = a_matrix.rows
    num_cols = a_matrix.cols

    if len(z_variables) != num_cols:
        raise ValidationError(
            f"Number of z variables ({len(z_variables)}) must match "
            f"number of A-matrix columns ({num_cols})"
        )

    if len(beta_parameters) != num_rows:
        raise ValidationError(
            f"Number of beta parameters ({len(beta_parameters)}) must match "
            f"number of A-matrix rows ({num_rows})"
        )

    # Define Phi as a symbolic function of all z variables
    phi = sp.Function("Phi")(*z_variables)

    equations = []

    # Generate one equation for each row of A
    for r in range(num_rows):
        # Left-hand side: Euler operator Ehat_r applied to Phi
        # Ehat_r * Phi = sum_j A[r,j] * z_j * dPhi/dz_j
        lhs = sum(
            a_matrix[r, j] * z_variables[j] * sp.Derivative(phi, z_variables[j])
            for j in range(num_cols)
        )

        # Right-hand side: beta_r * Phi
        rhs = beta_parameters[r] * phi

        # Create equation: lhs = rhs
        equations.append(sp.Eq(lhs, rhs))

    return equations


def format_euler_equation(equation: sp.Equality, simplify_lhs: bool = True) -> str:
    """
    Format an Euler equation as a readable string.

    Parameters
    ----------
    equation : sp.Equality
        Euler equation to format.
    simplify_lhs : bool, default True
        Whether to simplify the left-hand side before formatting.

    Returns
    -------
    str
        Formatted equation string.

    Examples
    --------
    >>> import sympy as sp
    >>> z1, z2 = sp.symbols('z1 z2')
    >>> phi = sp.Function('Phi')(z1, z2)
    >>> lhs = z1 * sp.Derivative(phi, z1) + z2 * sp.Derivative(phi, z2)
    >>> rhs = sp.Symbol('beta') * phi
    >>> eq = sp.Eq(lhs, rhs)
    >>> print(format_euler_equation(eq))
    """
    lhs = equation.lhs
    rhs = equation.rhs

    if simplify_lhs:
        lhs = sp.simplify(lhs)

    return f"{lhs} = {rhs}"
