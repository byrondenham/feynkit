"""Example: Constructing GKZ hypergeometric system for a bubble diagram."""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph

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
print("GKZ HYPERGEOMETRIC SYSTEM FOR BUBBLE DIAGRAM")
print("=" * 80)

print("\nLee-Pomeransky G polynomial:")
print(f"  G = {integral.symanzik.g}")

print("\n" + "=" * 80)
print(integral.gkz)

print("\n" + "=" * 80)
print("MONOMIAL SUPPORT")
print("=" * 80)
for i, (exp_vec, coeff) in enumerate(integral.gkz.support):
    print(f"  [{i}] u^{exp_vec}: coefficient = {coeff}")

print("\n" + "=" * 80)
