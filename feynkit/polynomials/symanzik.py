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
from .spanning_trees import separating_2forest_poly, spanning_tree_poly


def _reverse_monomials(poly_expr: sp.Expr, variables: list[sp.Symbol]) -> sp.Expr:
    """
    Compute ``(∏ xᵢ) · P(1/x₁, …, 1/xₙ)`` without rational arithmetic.

    Every monomial in a spanning-tree or spanning-2-forest polynomial is
    square-free (each variable appears with exponent 0 or 1), so the map
    ``eᵢ → 1 − eᵢ`` sends valid exponent vectors to valid exponent vectors.
    Working directly on the :class:`~sympy.Poly` monomial list avoids
    creating rational intermediate expressions.
    """
    if poly_expr == 0:
        return sp.Integer(0)
    poly = sp.Poly(poly_expr, *variables, domain="EX")
    terms: dict[tuple[int, ...], sp.Expr] = {}
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        terms[tuple(1 - e for e in monom)] = coeff
    if not terms:
        return sp.Integer(0)
    return sp.Poly.from_dict(terms, *variables, domain="EX").as_expr()


def _calculate_symanzik_polynomials(
    graph: Graph,
    momentum_products: dict[tuple[int, int], sp.Expr],
) -> tuple[sp.Expr, sp.Expr]:
    """
    Compute the Symanzik U and F polynomials for a Feynman graph.

    The Symanzik polynomials are fundamental to the parametric representation
    of Feynman integrals:

    - U(a): First Symanzik polynomial, depends only on graph topology and
      Schwinger parameters {a_i}. Encodes spanning trees of the graph.

    - F(a): Second Symanzik polynomial, includes masses and external momenta.
      Encodes two-trees and kinematic information.

    Parameters
    ----------
    graph : Graph
        The Feynman graph for which to compute U and F.
    momentum_products : Dict[Tuple[int, int], sp.Expr]
        Dictionary of momentum dot products for external legs.

    Returns
    -------
    U : sp.Expr
        First Symanzik polynomial U(a_1, …, a_n).
    F : sp.Expr
        Second Symanzik polynomial F(a_1, …, a_n).

    Raises
    ------
    PolynomialError
        If spanning-tree enumeration fails (e.g. disconnected internal graph).

    Notes
    -----
    **Algorithm:**

    Let C = sum over spanning trees T of ∏_{e∈T} aₑ  (spanning-tree polynomial
    of the *internal* graph).  Then:

        U = (∏ aᵢ) · C(1/a₁, …, 1/aₙ)

    which is computed without rational arithmetic via :func:`_reverse_monomials`.

    Similarly, for each pair of external legs (j, k) attached to distinct
    internal vertices vⱼ, vₖ, let Q_{jk} = sum over spanning 2-forests that
    separate vⱼ from vₖ of ∏_{e in forest} aₑ.  Then:

        F₀ = Σ_{j<k} (pⱼ·pₖ / μ²) · (∏ aᵢ) · Q_{jk}(1/a)

    and F = F₀ + U · Σᵢ mᵢ²/μ² · aᵢ.

    This completely avoids a symbolic Laplacian-matrix determinant.

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    .. [2] Symanzik, K. (1971). Commun. Math. Phys. 18, 227–246.
    """
    internal_edges = graph.get_internal_edges()
    n_int = graph.internal_vertices

    if not internal_edges:
        raise PolynomialError("Graph has no internal edges.")

    # 0-indexed edge pairs for enumeration
    edge_pairs = [(e.v1 - 1, e.v2 - 1) for e in internal_edges]
    params = [graph.schwinger_parameters[e.idx] for e in internal_edges]

    # === U polynomial ===
    # C = spanning-tree polynomial of the internal graph
    C = spanning_tree_poly(n_int, edge_pairs, params)
    if C == 0:
        raise PolynomialError(
            "No spanning trees found. Check that the internal graph is connected."
        )
    u_polynomial = _reverse_monomials(C, params)

    # === F₀: kinematic contribution from spanning 2-forests ===
    # Map external leg index j (1-based) → 0-indexed internal vertex
    leg_to_vertex: dict[int, int] = {}
    for ext_edge in graph.get_external_edges():
        leg_number = ext_edge.v2 - n_int
        leg_to_vertex[leg_number] = ext_edge.v1 - 1

    f_0 = sp.Integer(0)
    n_ext = graph.external_legs
    mu2 = graph.energy_scale**2

    for j in range(1, n_ext + 1):
        for k in range(j + 1, n_ext + 1):
            vj = leg_to_vertex[j]
            vk = leg_to_vertex[k]
            if vj == vk:
                continue  # same internal vertex → no 2-forest can separate them

            p_jk = get_momentum_product(momentum_products, j, k)
            if p_jk == 0:
                continue

            Q_jk = separating_2forest_poly(n_int, edge_pairs, vj, vk, params)
            if Q_jk != 0:
                f_0 += (p_jk / mu2) * _reverse_monomials(Q_jk, params)

    f_0 = sp.expand(f_0)

    # === Mass contribution ===
    mass_contribution = sp.Integer(0)
    for edge in internal_edges:
        param = graph.schwinger_parameters[edge.idx]
        mass_contribution += param * (edge.get_mass() ** 2 / mu2)

    f_polynomial = sp.expand(f_0 + u_polynomial * mass_contribution)

    return u_polynomial, f_polynomial


def extract_u_from_w(
    w_coefficients: dict[tuple[int, ...], sp.Expr],
    schwinger_params: list[sp.Symbol],
) -> sp.Expr:
    """
    Extract the U polynomial from W polynomial coefficients.

    Legacy helper retained for external callers; new code should use
    :func:`_calculate_symanzik_polynomials` directly.
    """
    degree_1_key = next((key for key in w_coefficients if len(key) == 1), None)
    if degree_1_key is None:
        raise PolynomialError("No degree-1 b-term found in W polynomial")

    coeff = w_coefficients[degree_1_key]
    return _reverse_monomials(coeff, schwinger_params)
