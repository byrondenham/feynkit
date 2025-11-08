"""
GKZ hypergeometric systems module for Feynman integrals.

Provides tools for constructing and analysing GKZ A-hypergeometric systems,
including monomial support extraction, A-matrix construction, and Euler operators.

Classes
-------
GKZSystem
    Complete GKZ hypergeometric system with all components.

Functions
---------
extract_monomial_support
    Extract monomial support from a polynomial.
construct_gkz_matrix
    Construct GKZ A-matrix from polynomial support.
create_euler_operators
    Create Euler differential operators from A-matrix.
create_euler_equations
    Generate Euler differential equations.
create_gkz_system
    Construct complete GKZ system from Lee-Pomeransky polynomial.
extract_coefficients_from_support
    Extract coefficients from support.
extract_exponents_from_support
    Extract exponent vectors from support.

Examples
--------
>>> from feynkit.systems import create_gkz_system
>>> from feynkit import Edge, Graph, create_momentum_products, create_parametrisations
>>> import sympy as sp
>>>
>>> # Create bubble diagram
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>>
>>> # Setup parameters
>>> D = sp.Symbol('D', positive=True)
>>> nu1, nu2 = sp.symbols('nu1 nu2', positive=True)
>>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
>>>
>>> # Create parametrisations and get G polynomial
>>> all_param = create_parametrisations(graph, D, 1, {1: nu1, 2: nu2}, p_dot)
>>> G = all_param.lee_pomeransky.get_g_polynomial()
>>>
>>> # Create GKZ system
>>> u_vars = all_param.lee_pomeransky.compute().parameters
>>> gkz = create_gkz_system(G, u_vars, D, [nu1, nu2])
>>> print(gkz)
"""

from .complete import (
    GKZSystem,
    create_gkz_system,
    extract_coefficients_from_support,
    extract_exponents_from_support,
)
from .euler import create_euler_equations, create_euler_operators, format_euler_equation
from .gkz import construct_gkz_matrix, validate_gkz_matrix
from .monomial import (
    count_monomials,
    extract_monomial_support,
    filter_support_by_degree,
    get_max_degree,
)

__all__ = [
    # Main system
    "GKZSystem",
    "create_gkz_system",
    # Monomial support
    "extract_monomial_support",
    "count_monomials",
    "get_max_degree",
    "filter_support_by_degree",
    # GKZ matrix
    "construct_gkz_matrix",
    "validate_gkz_matrix",
    # Euler operators
    "create_euler_operators",
    "create_euler_equations",
    "format_euler_equation",
    # Utilities
    "extract_coefficients_from_support",
    "extract_exponents_from_support",
]
