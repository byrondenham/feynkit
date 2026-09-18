"""Example: Computing toric ideal for a Feynman integral."""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.algebra import is_binomial_ideal

# Create bubble diagram
m1, m2 = sp.symbols("m1 m2", nonnegative=True)
nu1, nu2 = sp.symbols("nu1 nu2", positive=True)

e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
integral = FeynmanIntegral(graph, propagator_exponents={1: nu1, 2: nu2})

print("=" * 80)
print("TORIC IDEAL COMPUTATION FOR BUBBLE DIAGRAM")
print("=" * 80)

print(f"\nG polynomial: {integral.symanzik.g}")

print("\nA-matrix:")
print(integral.gkz.a_matrix)

print("\n" + "=" * 80)
print("COMPUTING TORIC IDEAL")
print("=" * 80)

generators = integral.toric_ideal.generators

print(f"\nNumber of generators: {len(generators)}")
print(f"Is binomial ideal: {is_binomial_ideal(generators) if generators else 'N/A'}")

if generators:
    print("\nGenerators:")
    for i, gen in enumerate(generators):
        print(f"  [{i}] {gen} = 0")
else:
    print("\nNo non-trivial generators (free polynomial ring)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
The toric ideal generators represent polynomial relations among the
monomial coefficients of the G polynomial. These relations correspond
to integration-by-parts (IBP) identities for the Feynman integral.

If the ideal is trivial (no generators), then all monomial coefficients
are algebraically independent, and the integral is a "master integral"
that cannot be reduced further using IBP identities.
""")
