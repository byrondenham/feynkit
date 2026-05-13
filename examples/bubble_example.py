"""Example: Computing Symanzik polynomials for a bubble diagram."""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph

# Create mass and exponent symbols
m1, m2 = sp.symbols("m1 m2", nonnegative=True)
nu1, nu2 = sp.symbols("nu1 nu2", positive=True)

# Create bubble diagram edges
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

# Build graph and unified integral object
graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
integral = FeynmanIntegral(graph, propagator_exponents={1: nu1, 2: nu2})

print("=" * 60)
print("Bubble Diagram Analysis")
print("=" * 60)
print(f"\nGraph: {graph}")
print(f"Loop count: {integral.loop_count}")
print(f"\nMomentum products: {integral.momentum_products}")

# Symanzik polynomials are accessed as a single grouped property.
print("\nSymanzik U polynomial:")
print(f"  U = {integral.symanzik.u}")
print("\nSymanzik F polynomial:")
print(f"  F = {integral.symanzik.f}")

print("\n" + "=" * 60)
