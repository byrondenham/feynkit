"""
Symanzik polynomial computation for Feynman integrals.

This module computes the Symanzik polynomials U and F, which are central to
the parametric representation of Feynman integrals. These polynomials encode
topological and kinematic information about the Feynman diagram.
"""

import sympy as sp

from ..core.exceptions import PolynomialError
from ..core.graph import Graph
from ..kinematics.momentum import get_momentum_product


def calculate_symanzik_polynomials(
    graph: Graph,
    momentum_products: dict[tuple[int, int], sp.Expr],
) -> tuple[sp.Expr, sp.Expr]:
    """
    Compute the Symanzik U and F polynomials for a Feynman graph.

    The Symanzik polynomials are fundamental to the parametric representation of
    Feynman integrals:

    - U(a): First Symanzik polynomial, depends only on graph topology and Schwinger
      parameters {a_i}. Encodes spanning trees of the graph.

    - F(a): Second Symanzik polynomial, includes masses and external momenta.
      Encodes two-trees and kinematic information.

    Parameters
    ----------
    graph : Graph
        The Feynman graph for which to compute U and F.
    momentum_products : Dict[Tuple[int, int], sp.Expr]
        Dictionary of momentum dot products for external legs, typically from
        create_momentum_products(). Maps pairs (i, j) to p_i · p_j or s_{ij}/2.

    Returns
    -------
    U : sp.Expr
        First Symanzik polynomial U(a_1, ..., a_n), a homogeneous polynomial
        in the Schwinger parameters. Represents sum over spanning trees.
    F : sp.Expr
        Second Symanzik polynomial F(a_1, ..., a_n), includes contributions from
        propagator masses and external momentum invariants.

    Raises
    ------
    PolynomialError
        If the graph has no degree-1 b-term in W polynomial (indicates invalid
        graph topology for this construction).

    Notes
    -----
    Computation Algorithm:

    The computation proceeds by:
    1. Expanding the W polynomial by external parameters b_j
    2. Extracting U from the coefficient of degree-1 terms (after inversion a_i → 1/a_i)
    3. Computing F_0 from degree-2 terms weighted by momentum products
    4. Adding mass contributions: m_i^2 * a_i for each propagator

    Normalisation:

    The polynomials are normalised by the energy scale μ:
    - Momentum terms are scaled by 1/μ^2
    - Mass terms are scaled by m_i^2/μ^2

    Physical Interpretation:

    For a bubble diagram with 2 external legs:
        U = a_1 + a_2
        F = a_1 * a_2 * s / μ^2 + (a_1 * m_1^2 + a_2 * m_2^2) / μ^2

    Examples
    --------
    >>> import sympy as sp
    >>> from feynkit.core import Edge, Graph
    >>> from feynkit.kinematics import create_momentum_products
    >>> from feynkit.polynomials import calculate_symanzik_polynomials
    >>>
    >>> # Create bubble diagram
    >>> m1, m2 = sp.symbols('m1 m2', nonnegative=True)
    >>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1)
    >>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2)
    >>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    >>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    >>>
    >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    >>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
    >>>
    >>> U, F = calculate_symanzik_polynomials(graph, p_dot)
    >>> print(f"U = {U}")
    >>> print(f"F = {F}")

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    .. [2] Symanzik, K. (1971). "Small distance behaviour in field theory and power counting."
            Commun. Math. Phys. 18, 227-246.
    """
    # Expand W polynomial by external leg parameters b_j
    w_coefficients = graph.expand_w_by_external_parameters()

    # Get list of Schwinger parameters for internal edges
    internal_edges = graph.get_internal_edges()
    schwinger_params = [graph.schwinger_parameters[e.idx] for e in internal_edges]

    # Compute product of all Schwinger parameters: prod a_i
    product_of_params = sp.prod(schwinger_params) if schwinger_params else sp.Interger(1)

    # Create inversion map: a_i → 1/a_i for variable transformation
    inversion_map = {param: 1 / param for param in schwinger_params}

    # === Extract U polynomial from degree-1 term ===
    # Find coefficient of any single b_j term (degree-1 in b parameters)
    degree_1_key = next((key for key in w_coefficients if len(key) == 1), None)

    if degree_1_key is None:
        raise PolynomialError(
            "No degree-1 b-term in W polynomial. "
            "This indicates invalid graph topology for Symanzik polynomial extraction."
        )

    # U is obtained by: prod a_i * coeff[degree-1 term] with substitution a_i → 1/a_i
    # This gives a polynomial homogeneous in {a_i}
    u_polynomial = sp.simplify(
        sp.factor(product_of_params * w_coefficients[degree_1_key].subs(inversion_map))
    )

    # === Extract F_0 from degree-2 terms ===
    momentum_contribution = sp.Integer(0)

    for key, coefficient in w_coefficients.items():
        if len(key) == 2:
            # Degree-2 term corresponds to b_j * b_k, encoding (p_j + p_k)^2
            j, k = key

            # Get momentum dot product p_j · p_k (or s_{jk}/2)
            p_jk = get_momentum_product(momentum_products, j, k)

            # Add contribution: (p_j · p_k / μ^2) * coefficient
            momentum_contribution += (p_jk / (graph.energy_scale**2)) * sp.simplify(
                coefficient.subs(inversion_map)
            )

    # Transform to get F_0: prod a_i * sum of weighted degree-2 terms
    f_0 = sp.simplify(sp.factor(product_of_params * momentum_contribution))

    # === Add mass contributions ===
    mass_contribution = sp.Integer(0)

    for edge in internal_edges:
        param = graph.schwinger_parameters[edge.idx]
        mass_squared = edge.get_mass() ** 2

        # Add term: a_i * (m_i^2 / μ^2)
        mass_contribution += param * (mass_squared / (graph.energy_scale**2))

    # === Complete F polynomial ===
    # F = F_0 + U * (mass contributions)
    # The factor of U ensures correct homogeneity in Schwinger parameters
    f_polynomial = sp.simplify(sp.factor(f_0 + u_polynomial * mass_contribution))

    return u_polynomial, f_polynomial


def extract_u_from_w(
    w_coefficients: dict[tuple[int, ...], sp.Expr],
    schwinger_params: list[sp.Symbol],
) -> sp.Expr:
    """
    Extract the U polynomial from W polynomial coefficients.

    This is a helper function that performs the variable transformation
    and extraction logic for the U polynomial.

    Parameters
    ----------
    w_coefficients : Dict[Tuple[int, ...], sp.Expr]
        W polynomial expanded by external parameters.
    schwinger_params : list[sp.Symbol]
        List of Schwinger parameters.

    Returns
    -------
    sp.Expr
        The U polynomial.

    Raises
    ------
    PolynomialError
        If no degree-1 term is found.
    """
    # Find degree-1 coefficient
    degree_1_key = next((key for key in w_coefficients if len(key) == 1), None)

    if degree_1_key is None:
        raise PolynomialError("No degree-1 b-term found in W polynomial")

    # Product of all Schwinger parameters
    product_of_params = sp.prod(schwinger_params) if schwinger_params else sp.Integer(1)

    # Inversion map
    inversion_map = {param: 1 / param for param in schwinger_params}

    # Extract U
    u_polynomial = sp.simplify(
        sp.factor(product_of_params * w_coefficients[degree_1_key].subs(inversion_map))
    )

    return u_polynomial
