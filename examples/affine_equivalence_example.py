"""Example: comparing convex lattice polytopes via the equivalence verbs.

Two complementary verbs are available:

- ``is_unimodular_equivalent`` (Liu–Cai, arXiv:2506.23846) — the natural
  relation for *lattice* polytopes; allows only ``U ∈ GL_n(ℤ)`` and
  integer translations. Returns a witness matrix on success.
- ``is_affinely_equivalent`` — broader equivalence over the rationals; the
  brute-force backend is exact but slow on large polytopes.
"""

import sympy as sp

from feynkit.normal_forms import is_affinely_equivalent, is_unimodular_equivalent


# Source polytope: unit square in the plane.
points_a = [(0, 0), (1, 0), (1, 1), (0, 1)]

# Image under a unimodular shear x -> x + y, y -> y, then translation by (3, -1).
points_b = [(3, -1), (4, -1), (5, 0), (4, 0)]

# Same square but slightly off-lattice: not unimodularly equivalent.
points_c_rational = sp.Matrix([[0, 0], [sp.Rational(3, 2), 0],
                               [sp.Rational(3, 2), 1], [0, 1]])

print("Unimodular equivalence (Liu–Cai)")
print("=" * 60)

result = is_unimodular_equivalent(points_a, points_b)
print(f"unit square → sheared+translated square : {result.equivalent}")
if result.equivalent:
    print(f"  witness map U =\n{sp.pretty(result.witness_map)}")
    print(f"  vertex correspondence: {result.vertex_correspondence}")

different = [(0, 0), (1, 0), (2, 0), (0, 1)]  # not affinely equivalent (collinear edge)
result = is_unimodular_equivalent(points_a, different)
print(f"\nunit square → collinear-edge polygon    : {result.equivalent}")

print()
print("Affine equivalence (brute-force, broader)")
print("=" * 60)

result = is_affinely_equivalent(points_a, points_b)
print(f"unit square → sheared+translated square : {result.equivalent}")

result = is_affinely_equivalent(sp.Matrix(points_a), points_c_rational)
print(f"unit square → 1.5x-scaled rectangle     : {result.equivalent}")
