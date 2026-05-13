"""
Polynomial computation module for Feynman integrals.

Provides utilities for manipulating Symanzik / Lee–Pomeransky polynomials
and related polynomial expressions. Computation of the U/F polynomials of a
graph is handled by :class:`feynkit.FeynmanIntegral`; this module exposes
the lower-level transforms and accessors.

Functions
---------
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
"""

from .operations import extract_coefficient, homogenise_polynomial, simplify_rational_function
from .symanzik import extract_u_from_w
from .transforms import invert_variables, projective_transformation, rescale_variables

__all__ = [
    "extract_u_from_w",
    "homogenise_polynomial",
    "extract_coefficient",
    "simplify_rational_function",
    "invert_variables",
    "rescale_variables",
    "projective_transformation",
]
