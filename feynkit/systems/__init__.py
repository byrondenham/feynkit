"""
GKZ hypergeometric systems module for Feynman integrals.

The complete GKZ system associated with a Feynman integral is obtained via
:attr:`feynkit.FeynmanIntegral.gkz`. This module exposes the lower-level
helpers for working with monomial supports, A-matrices, and Euler operators
in their own right.

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
extract_coefficients_from_support
    Extract coefficients from support.
extract_exponents_from_support
    Extract exponent vectors from support.
"""

from .complete import (
    GKZSystem,
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
    "GKZSystem",
    "extract_monomial_support",
    "count_monomials",
    "get_max_degree",
    "filter_support_by_degree",
    "construct_gkz_matrix",
    "validate_gkz_matrix",
    "create_euler_operators",
    "create_euler_equations",
    "format_euler_equation",
    "extract_coefficients_from_support",
    "extract_exponents_from_support",
]
