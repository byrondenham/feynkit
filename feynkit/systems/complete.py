"""
Complete GKZ hypergeometric system construction.

Provides high-level functions to construct the complete GKZ system from
a Lee-Pomeransky polynomial, including A-matrix, beta parameters, and
Euler equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..core.exceptions import ValidationError
from .euler import create_euler_equations
from .gkz import construct_gkz_matrix
from .monomial import extract_monomial_support


@dataclass
class GKZSystem:
    """
    Complete GKZ hypergeometric system.

    Attributes
    ----------
    a_matrix : sp.Matrix
        The GKZ A-matrix of size (n+1) × m.
    z_variables : List[sp.Symbol]
        Differential variables [z_1, z_2, ..., z_m].
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support: [(exponent_vector, coefficient), ...].
    beta_parameters : List[sp.Expr]
        Parameter vector [beta_0, beta_1, ..., beta_n].
    euler_equations : List[sp.Equality]
        System of (n+1) Euler differential equations.
    dimension : sp.Expr
        Spacetime dimension D.
    propagator_exponents : List[sp.Expr]
        List of propagator exponents [nu_1, nu_2, ..., nu_n].
    """

    a_matrix: sp.Matrix
    z_variables: list[sp.Symbol]
    support: list[tuple[tuple[int, ...], sp.Expr]]
    beta_parameters: list[sp.Expr]
    euler_equations: list[sp.Equality]
    dimension: sp.Expr
    propagator_exponents: list[sp.Expr]

    def __str__(self) -> str:
        """Return a readable string representation."""
        lines = [
            "GKZ Hypergeometric System",
            "=" * 60,
            f"A-matrix shape: {self.a_matrix.rows} × {self.a_matrix.cols}",
            f"Number of monomials: {len(self.support)}",
            f"Number of variables: {len(self.z_variables)}",
            f"\nA-matrix:\n{self.a_matrix}",
            f"\nBeta parameters: {self.beta_parameters}",
            f"\nEuler equations ({len(self.euler_equations)} equations):",
        ]

        for i, eq in enumerate(self.euler_equations):
            lines.append(f"  [{i}] {eq}")

        return "\n".join(lines)


def create_gkz_system(
    g_polynomial: sp.Expr,
    u_variables: list[sp.Symbol],
    dimension: sp.Expr,
    propagator_exponents: list[sp.Expr],
) -> GKZSystem:
    """
    Construct complete GKZ hypergeometric system from Lee-Pomeransky representation.

    Builds the full GKZ A-hypergeometric system associated with a Feynman integral,
    including the A-matrix, parameter vector beta, differential variables z, monomial
    support, and Euler differential equations.

    Parameters
    ----------
    g_polynomial : sp.Expr
        The Lee-Pomeransky polynomial G(u) = U(u) + F(u).
        This polynomial encodes the complete structure of the Feynman integral.
    u_variables : List[sp.Symbol]
        Lee-Pomeransky parameters [u_1, u_2, ..., u_n], one per internal edge.
    dimension : sp.Expr
        Spacetime dimension (e.g., symbolic D or 4 - 2 epsilon for dimensional regularisation).
    propagator_exponents : List[sp.Expr]
        List of propagator exponents [nu_1, nu_2, ..., nu_n] for each internal edge.
        Order must match u_variables.

    Returns
    -------
    GKZSystem
        Complete GKZ system with all components.

    Raises
    ------
    ValidationError
        If inputs are inconsistent.

    Notes
    -----
    The parameter vector beta is constructed as:
        beta = [sum nu_i - D/2, nu_1, nu_2, ..., nu_n]

    This choice ensures that:
    1. The homogeneity degree beta_0 matches the overall dimensional behavior
    2. The individual beta_i track the propagator index shifts

    Examples
    --------
    >>> import sympy as sp
    >>> # For a bubble diagram in Lee-Pomeransky representation
    >>> u1, u2 = sp.symbols('u1 u2', nonnegative=True)
    >>> D = sp.Symbol('D', positive=True)
    >>> nu1, nu2 = sp.symbols('nu1 nu2', positive=True)
    >>>
    >>> # Suppose G(u) = u1 + u2 + s*u1*u2/mu^2 (simplified)
    >>> s, mu = sp.symbols('s mu', positive=True)
    >>> G = u1 + u2 + s*u1*u2/mu**2
    >>>
    >>> gkz = create_gkz_system(G, [u1, u2], D, [nu1, nu2])
    >>> print(gkz)

    References
    ----------
    .. [1] Bitoun, T., Bogner, C., Klausen, R.P., Panzer, E. (2019).
            "Feynman integral relations from parametric annihilators."
            Lett. Math. Phys. 109, 497-564.
    .. [2] Klausen, R.P. (2020). "Kinematic singularities of Feynman integrals
            and principal A-determinants." JHEP 02, 004.
    """
    # Validate inputs
    if len(u_variables) != len(propagator_exponents):
        raise ValidationError(
            f"Number of u variables ({len(u_variables)}) must match "
            f"number of propagator exponents ({len(propagator_exponents)})"
        )

    # Extract monomial support from G(u)
    support = extract_monomial_support(g_polynomial, u_variables)
    num_monomials = len(support)

    # Construct A-matrix
    a_matrix = construct_gkz_matrix(g_polynomial, u_variables)

    # Create differential variables z_j
    z_variables = [sp.Symbol(f"z_{j + 1}") for j in range(num_monomials)]

    # Construct parameter vector beta
    # beta = [sum_i nu_i - D/2, nu_1, nu_2, ..., nu_n]
    nu_sum = sum(propagator_exponents)
    beta_parameters = [nu_sum - dimension / 2] + list(propagator_exponents)

    # Generate Euler differential equations
    euler_equations = create_euler_equations(a_matrix, beta_parameters, z_variables)

    return GKZSystem(
        a_matrix=a_matrix,
        z_variables=z_variables,
        support=support,
        beta_parameters=beta_parameters,
        euler_equations=euler_equations,
        dimension=dimension,
        propagator_exponents=propagator_exponents,
    )


def extract_coefficients_from_support(
    support: list[tuple[tuple[int, ...], sp.Expr]],
) -> list[sp.Expr]:
    """
    Extract just the coefficients from monomial support.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support.

    Returns
    -------
    List[sp.Expr]
        List of coefficients in the same order as support.

    Examples
    --------
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = 2*u1**2 + 3*u1*u2 + u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> coeffs = extract_coefficients_from_support(support)
    >>> print(coeffs)
    [2, 3, 1]
    """
    return [coeff for _, coeff in support]


def extract_exponents_from_support(
    support: list[tuple[tuple[int, ...], sp.Expr]],
) -> list[tuple[int, ...]]:
    """
    Extract just the exponent vectors from monomial support.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], sp.Expr]]
        Monomial support.

    Returns
    -------
    List[Tuple[int, ...]]
        List of exponent vectors.

    Examples
    --------
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = u1**2 + u1*u2 + u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> exponents = extract_exponents_from_support(support)
    >>> print(exponents)
    [(2, 0), (1, 1), (0, 2)]
    """
    return [exp_vec for exp_vec, _ in support]
