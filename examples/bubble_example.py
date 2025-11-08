"""Example: Computing Symanzik polynomials for a bubble diagram."""

import sympy as sp

from feynkit import Edge, Graph, calculate_symanzik_polynomials, create_momentum_products

# Create mass and exponent symbols
m1, m2 = sp.symbols("m1 m2", nonnegative=True)
nu1, nu2 = sp.symbols("nu1 nu2", positive=True)

# Create bubble diagram edges
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

# Build graph
graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

print("=" * 60)
print("Bubble Diagram Analysis")
print("=" * 60)
print(f"\nGraph: {graph}")
print(f"Loop count: {graph.get_loop_count()}")

# Create momentum products using Mandelstam variables
p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
print(f"\nMomentum products: {p_dot}")

# Calculate Symanzik polynomials
U, F = calculate_symanzik_polynomials(graph, p_dot)

print("\nSymanzik U polynomial:")
print(f"  U = {U}")
print("\nSymanzik F polynomial:")
print(f"  F = {F}")

print("\n" + "=" * 60)
