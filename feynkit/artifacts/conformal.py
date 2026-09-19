"""
A-configurations for momentum-space conformal field theory.

Factories
---------
massless_polygon_a_config(n)
    GKZ A-configuration of the 1-loop massless n-gon (C_n = n-cycle).
    For n=3 this is the triangle; for n=4, the box.
    Uses n LP variables (one per internal edge) and lives in R ^n.

bms_simplex_a_config(n)
    GKZ A-configuration of the Bzowski-McFadden-Skenderis n-point conformal
    simplex integral (Bzowski-McFadden-Skenderis 2021; Caloro 2024).
    The G polynomial has 2n monomials in n variables:
      - n lower monomials prod_{j!=i} u_j  (degree n-1, "K-function denominators")
      - n upper monomials u_i^2 prod_{j!=i} u_j  (degree n+1, "scaling terms")
    For n=3 this reproduces triple_k_a_config().

complete_graph_a_config(n)
    GKZ A-configuration of the complete graph K_n with one external leg per
    vertex (the "n-simplex Feynman graph").
    K_n has C(n,2) internal edges, so C(n,2) LP variables and ambient dim
    C(n,2).  For n=3, K_3 = C_3 = triangle, coinciding with
    massless_polygon_a_config(3).

conformal_companion_a_config(n)
    Candidate companion A-configuration to bms_simplex_a_config(n), designed to
    admit a det=2 finite-index map to BMS_n.  For n=3 this is exactly the
    massless triangle.  The 2n monomials are:
      - n lower monomials prod_{j!=i} u_j  (degree n-1, same as BMS_n lower)
      - n upper monomials u_i            (degree 1, standard basis)
    G polynomial: sum_i prod_{j!=i} u_j + sum_i p_i^2 u_i
"""

from __future__ import annotations

import sympy as sp

from ..a_configuration import AConfiguration
from ..core.edge import Edge
from ..core.graph import Graph
from ..integral import FeynmanIntegral

_ZERO = sp.Integer(0)


# -- massless n-gon (C_n) ------------------------------------------------------


def massless_polygon_a_config(n: int) -> AConfiguration:
    """
    GKZ A-configuration of the massless 1-loop n-gon (n-cycle C_n).

    The n-gon has n internal edges forming a cycle and n external legs, one
    per vertex.  The Lee-Pomeransky G polynomial lives in n variables (one per
    internal edge).

    Parameters
    ----------
    n
        Number of sides (= external legs).  Must be >= 3.

    Returns
    -------
    AConfiguration
        Homogenised (n+1) x N A-matrix derived from feynkit's LP pipeline.
    """
    if n < 3:
        raise ValueError(f"n-gon requires n >= 3, got {n}")
    edges = []
    for i in range(1, n + 1):
        j = (i % n) + 1
        edges.append(Edge(idx=i, v1=i, v2=j, is_internal=True, mass=_ZERO))
    for i in range(1, n + 1):
        edges.append(Edge(idx=n + i, v1=i, v2=n + i, is_internal=False))
    graph = Graph(internal_vertices=n, external_legs=n, edges=edges)
    fi = FeynmanIntegral(graph)
    return AConfiguration(fi.gkz.a_matrix, is_homogenized=True)


# -- BMS n-point conformal simplex ---------------------------------------------


def _bms_g_polynomial(n: int) -> sp.Expr:
    """
    Derive the G polynomial for the n-Bessel conformal simplex integral via
    Schwinger parameterisation.

    Starting from

        I_n = int_0^inf r^{beta_0-1} prod_{i=1}^n K_{nu_i}(p_i r) dr

    with the integral representation

        K_nu(p r) = (p/2)^nu / 2 * int_0^inf u^{-nu-1} exp(-r(u + p^2/(4u))) du

    the product of n K-functions introduces n Schwinger parameters u_1,...,u_n and
    after integrating over r yields Gamma(beta_0) G_0^{-beta_0} with

        G_0(u) = sum_i u_i + sum_i p_i^2/(4u_i).

    Multiplying G_0 by 4 prod_j u_j to clear denominators, and absorbing the
    overall factor 4 prod u_j into the Schwinger weight, gives the polynomial

        G(u) = sum_i p_i^2 prod_{j!=i} u_j  +  4 sum_i u_i^2 prod_{j!=i} u_j

    where the coefficient 4 is absorbed into the GKZ coefficient of the upper
    monomials when all N=2n coefficients are treated as independent variables.

    Returns G as a SymPy expression in symbols u_1,...,u_n and p_1sq,...,p_nsq.
    """
    u = [sp.Symbol(f"u_{i + 1}") for i in range(n)]
    p_sq = [sp.Symbol(f"p_{i + 1}sq") for i in range(n)]
    prod_u = sp.Mul(*u)
    lower = sum(p_sq[i] * prod_u / u[i] for i in range(n))
    upper = 4 * sum(u) * prod_u  # 4 (sum u_i)(prod u_j) = 4 sum u_i^2 prod_{j!=i} u_j
    return sp.expand(lower + upper)


def bms_simplex_a_config(n: int) -> AConfiguration:
    """
    GKZ A-configuration of the BMS n-point conformal simplex (contact) integral.

    Derivation
    ----------
    The integral is the n-Bessel K-function form of the contact Witten diagram:

        I_n(p_1,...,p_n) = int_0^inf r^{beta_0-1} prod_i K_{nu_i}(p_i r) dr

    Using K_nu(pr) = (p/2)^nu/2 int_0^inf u^{-nu-1} exp(-r(u + p^2/(4u))) du for each
    factor, taking the product over i, and integrating out r gives

        I_n ~ Gamma(beta_0) int prod_i du_i u_i^{beta_0-nu_i-1} G_0^{-beta_0}

    with G_0 = sum_i u_i + sum_i p_i^2/(4u_i).  Multiplying G_0 by 4 prod_j u_j (which
    shifts the u-exponents) yields the Lee-Pomeransky-style polynomial

        G(u) = sum_i p_i^2 prod_{j!=i} u_j  +  4 sum_i u_i^2 prod_{j!=i} u_j       (*)

    Promoting all 2n coefficients of (*) to independent GKZ variables a_j
    gives the generalised integral I_A(beta, a), whose A-matrix is read off from
    the 2n monomial exponent vectors:

        lower : e_i  has 0 in position i, 1 elsewhere  (degree n-1)
        upper : f_i  has 2 in position i, 1 elsewhere  (degree n+1)

    The A-matrix is derived programmatically from the monomial support of G
    (see ``_bms_g_polynomial``).

    Physical context (BMS 2021, Caloro 2024)
    -----------------------------------------
    For equal conformal dimensions Delta_i = Delta, the n-point contact Witten diagram
    reduces to I_n after applying the star-mesh duality of Caloro (2024).
    The holonomic rank vol_0 = 2^{n-1} counts the independent A-hypergeometric
    series at generic beta.  The Smith invariants [1,...,1,2] reflect that the
    monomial support spans an index-2 sublattice of Z ^n (the even-parity
    sublattice).  For n=3 this recovers the triple-K A-configuration.

    Parameters
    ----------
    n
        Number of external points.  Must be >= 2.

    Returns
    -------
    AConfiguration
        Homogenised (n+1) x 2n A-matrix derived from the G polynomial (*).
    """
    if n < 2:
        raise ValueError(f"BMS simplex requires n >= 2, got {n}")

    G = _bms_g_polynomial(n)
    u = [sp.Symbol(f"u_{i + 1}") for i in range(n)]

    # Extract exponent vectors from the monomial support of G
    poly = sp.Poly(G, *u)
    monoms = sorted(poly.monoms())  # sorted for deterministic column order

    # Build homogenised A-matrix: first row all-ones, then one row per variable
    hom_row = [1] * len(monoms)
    coord_rows = [[m[k] for m in monoms] for k in range(n)]
    A = sp.Matrix([hom_row] + coord_rows)
    return AConfiguration(A, is_homogenized=True)


# -- K_n complete-graph LP -----------------------------------------------------


def complete_graph_a_config(n: int) -> AConfiguration:
    """
    GKZ A-configuration of the complete graph K_n with one external leg per
    vertex.

    K_n has C(n,2) = n(n-1)/2 internal edges and n external legs.  The
    Lee-Pomeransky G polynomial lives in C(n,2) variables (one per internal
    edge) and the A-matrix has ambient dimension C(n,2).

    For n=3, K_3 = C_3 = triangle, so this coincides with
    ``massless_polygon_a_config(3)``.
    For n=4, K_4 is a 3-loop tetrahedron graph (C(4,2)=6 LP variables,
    ambient dimension 6), much larger than the n=4 BMS simplex (ambient dim
    4).

    Parameters
    ----------
    n
        Number of vertices.  Must be >= 3.

    Returns
    -------
    AConfiguration
        Homogenised A-matrix derived from feynkit's LP pipeline.
    """
    if n < 3:
        raise ValueError(f"K_n requires n >= 3, got {n}")
    edges = []
    idx = 1
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            edges.append(Edge(idx=idx, v1=i, v2=j, is_internal=True, mass=_ZERO))
            idx += 1
    for v in range(1, n + 1):
        edges.append(Edge(idx=idx, v1=v, v2=n + v, is_internal=False))
        idx += 1
    graph = Graph(internal_vertices=n, external_legs=n, edges=edges)
    fi = FeynmanIntegral(graph)
    return AConfiguration(fi.gkz.a_matrix, is_homogenized=True)


# -- conformal companion -------------------------------------------------------


def conformal_companion_a_config(n: int) -> AConfiguration:
    """
    Candidate companion to bms_simplex_a_config(n) for the finite-index map.

    For n=3 this coincides with the massless triangle (massless_polygon_a_config(3)).

    The 2n monomials are:
      lower : prod_{j!=i} u_j  (degree n-1, same as BMS_n lower / U(C_n)),  i=1,...,n
      upper : u_i            (degree 1, standard basis),                   i=1,...,n

    G polynomial: G(u) = sum_i prod_{j!=i} u_j  +  sum_i p_i^2 u_i

    This is the natural higher-dimensional analogue of the triangle A-configuration
    when searching for companions related to BMS_n by a finite-index (det=2) affine map.

    Parameters
    ----------
    n
        Number of variables / external points.  Must be >= 2.

    Returns
    -------
    AConfiguration
        Homogenised (n+1) x 2n A-matrix.
    """
    if n < 3:
        raise ValueError(f"conformal companion requires n >= 3, got {n}")
    monomials = []
    for k in range(n):
        # lower: all-ones except position k -> prod_{j!=k} u_j
        monomials.append(tuple(0 if j == k else 1 for j in range(n)))
    for k in range(n):
        # upper: standard basis vector ehat_k -> u_k
        monomials.append(tuple(1 if j == k else 0 for j in range(n)))
    hom_row = [1] * (2 * n)
    coord_rows = [[m[r] for m in monomials] for r in range(n)]
    A = sp.Matrix([hom_row] + coord_rows)
    return AConfiguration(A, is_homogenized=True)
