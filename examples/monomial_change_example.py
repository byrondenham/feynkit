"""Example: affine map on exponents to Laurent monomial variable change."""

import sympy as sp

from feynkit.algebra import monomial_substitution_from_affine, overall_monomial_factor

x1, x2, x3 = sp.symbols("x1 x2 x3")
y1, y2, y3 = sp.symbols("y1 y2 y3")

M = sp.Matrix([
    [0, -1, -1],
    [-1, 0, -1],
    [-1, -1, 0],
])
c = sp.Matrix([1, 1, 1])

subs = monomial_substitution_from_affine(M, [x1, x2, x3], [y1, y2, y3])
factor = overall_monomial_factor(c, [y1, y2, y3])

print(subs)
print(f"factor = {factor}")
print("Note: this can be non-unimodular and gives Laurent monomials, not lattice automorphisms.")
