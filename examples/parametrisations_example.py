"""Example: Computing all three parametric representations for a bubble diagram."""

import sympy as sp

from feynkit import Edge, Graph, create_momentum_products, create_parametrisations

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

print("=" * 80)
print("PARAMETRIC REPRESENTATIONS FOR BUBBLE DIAGRAM")
print("=" * 80)
print(f"\nGraph: {graph}")

# Setup parameters
D = sp.Symbol("D", positive=True)
nus = {1: nu1, 2: nu2}
p_dot = create_momentum_products(n_external=2, use_mandelstam=True)

# Create all parametrisations
all_param = create_parametrisations(graph, D, 1, nus, p_dot)

print("\nSymanzik Polynomials:")
print(f"  U = {all_param.u_polynomial}")
print(f"  F = {all_param.f_polynomial}")

# Compute and display each parametrisation
print("\n" + "=" * 80)
print("SCHWINGER PARAMETRISATION")
print("=" * 80)
schwinger = all_param.schwinger.compute()
print(schwinger)

print("\n" + "=" * 80)
print("FEYNMAN PARAMETRISATION")
print("=" * 80)
feynman = all_param.feynman.compute()
print(feynman)

print("\n" + "=" * 80)
print("LEE-POMERANSKY PARAMETRISATION")
print("=" * 80)
lee_pom = all_param.lee_pomeransky.compute()
print(lee_pom)

# Show G polynomial
print("\nG Polynomial (Lee-Pomeransky):")
print(f"  G = {all_param.lee_pomeransky.get_g_polynomial()}")

print("\n" + "=" * 80)
