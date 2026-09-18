"""
Algebraic tools module for Feynman integrals.

Provides algebraic methods including toric ideal computation, Gröbner basis
algorithms, and syzygy analysis.

Functions
---------
compute_toric_ideal_generators
    Compute generators of the toric ideal from A-matrix.
count_generators
    Count number of toric ideal generators.
is_binomial_ideal
    Check if ideal is binomial.
extract_binomial_form
    Extract monomials from a binomial generator.
compute_groebner_basis
    Compute Gröbner basis for polynomial ideal.
reduce_polynomial
    Reduce polynomial modulo Gröbner basis.
is_in_ideal
    Check if polynomial is in ideal.

Examples
--------
>>> from feynkit.algebra import compute_toric_ideal_generators
>>> from feynkit.systems import construct_gkz_matrix
>>> import sympy as sp
>>>
>>> # Create simple polynomial
>>> u1, u2 = sp.symbols('u1 u2')
>>> poly = u1**2 + u1*u2 + u2**2
>>>
>>> # Construct A-matrix and compute toric ideal
>>> A = construct_gkz_matrix(poly, [u1, u2])
>>> generators = compute_toric_ideal_generators(A)
>>> for gen in generators:
...     print(gen)
"""

from .groebner import (
    compute_groebner_basis,
    ideal_quotient,
    intersect_ideals,
    is_in_ideal,
    reduce_polynomial,
)
from .monomial_change import (
    apply_monomial_change,
    monomial_substitution_from_affine,
    overall_monomial_factor,
    transform_exponent_vector,
    transform_support,
)
from .syzygy import compute_syzygy_module, trivial_syzygy
from .toric import (
    compute_toric_ideal_generators,
    count_generators,
    extract_binomial_form,
    is_binomial_ideal,
)

__all__ = [
    # Toric ideals
    "compute_toric_ideal_generators",
    "count_generators",
    "is_binomial_ideal",
    "extract_binomial_form",
    # Gröbner bases
    "compute_groebner_basis",
    "reduce_polynomial",
    "is_in_ideal",
    "intersect_ideals",
    "ideal_quotient",
    # Monomial changes
    "monomial_substitution_from_affine",
    "overall_monomial_factor",
    "apply_monomial_change",
    "transform_exponent_vector",
    "transform_support",
    # Syzygies
    "compute_syzygy_module",
    "trivial_syzygy",
]
