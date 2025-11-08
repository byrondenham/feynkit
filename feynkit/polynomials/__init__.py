"""
Polynomial computation module for Feynman integrals.

Provides functions to compute and manipulate Symanzik polynomials and
related polynomial structures.

Functions
---------
calculate_symanzik_polynomials
    Compute U and F Symanzik polynomials from a Feynman graph.
extract_u_from_w
    Extract U polynomial from W polynomial coefficients.
homogenise_polynomial
    Make a polynomial homogeneous.
extract_coefficient
    Extract coefficient of a specific monomial.
simplify_rational_function
    Simplify rational expressions.
invert_variables
    Apply variable inversion x_i → 1/x_i.
rescale_variables
    Apply variable rescaling.
projective_transformation
    Transform to projective coordinates.

Examples
--------
>>> from feynkit.polynomials import calculate_symanzik_polynomials
>>> from feynkit.kinematics import create_momentum_products
>>> from feynkit.core import Edge, Graph
>>>
>>> # Create bubble diagram
>>> import sympy as sp
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>>
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
>>>
>>> U, F = calculate_symanzik_polynomials(graph, p_dot)
"""

from .operations import extract_coefficient, homogenise_polynomial, simplify_rational_function
from .symanzik import calculate_symanzik_polynomials, extract_u_from_w
from .transforms import invert_variables, projective_transformation, rescale_variables

__all__ = [
    # Main functions
    "calculate_symanzik_polynomials",
    "extract_u_from_w",
    # Operations
    "homogenise_polynomial",
    "extract_coefficient",
    "simplify_rational_function",
    # Transforms
    "invert_variables",
    "rescale_variables",
    "projective_transformation",
]
