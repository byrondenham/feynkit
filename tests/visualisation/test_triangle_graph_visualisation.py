"""Test script to verify triangle graph polytope visualisation."""

import sympy as sp

from feynkit import Edge, Graph, create_momentum_products, create_parametrisations
from feynkit.systems import create_gkz_system
from feynkit.visualisation import save_polytope_tikz, visualise_newton_polytope

print("=" * 80)
print("TRIANGLE FEYNMAN GRAPH - NEWTON POLYTOPE VISUALISATION")
print("=" * 80)

# Create triangle graph (3 internal vertices, 3 internal edges, 3 external legs)
m1, m2, m3 = sp.symbols("m1 m2 m3", nonnegative=True)
nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3", positive=True)

# Internal propagators forming a triangle
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=3, is_internal=True, mass=m2, nu=nu2)
e3 = Edge(idx=3, v1=3, v2=1, is_internal=True, mass=m3, nu=nu3)

# External legs
ex1 = Edge(idx=4, v1=1, v2=4, is_internal=False)
ex2 = Edge(idx=5, v1=2, v2=5, is_internal=False)
ex3 = Edge(idx=6, v1=3, v2=6, is_internal=False)

# Build graph
graph = Graph(internal_vertices=3, external_legs=3, edges=[e1, e2, e3, ex1, ex2, ex3])

print(f"\nGraph: {graph}")
print(f"Loop count: {graph.get_loop_count()}")

# Setup parameters
D = sp.Symbol("D", positive=True)
nus = {1: nu1, 2: nu2, 3: nu3}
p_dot = create_momentum_products(n_external=3, use_mandelstam=True)

print(f"\nMomentum products: {p_dot}")

# Create parametrisations
all_param = create_parametrisations(graph, D, 1, nus, p_dot)

print("\nSymanzik polynomials:")
print(f"  U = {all_param.u_polynomial}")
print(f"  F = {all_param.f_polynomial}")

# Get Lee-Pomeransky polynomial
G = all_param.lee_pomeransky.get_g_polynomial()
u_vars = all_param.lee_pomeransky.compute().parameters

print("\nLee-Pomeransky G polynomial:")
print(f"  G = {G}")

# Create GKZ system
gkz = create_gkz_system(G, u_vars, D, [nu1, nu2, nu3])

print(f"\nMonomial support ({len(gkz.support)} monomials):")
for i, (exp_vec, coeff) in enumerate(gkz.support):
    print(f"  [{i}] u^{exp_vec}: {coeff}")

print(f"\nA-matrix shape: {gkz.a_matrix.shape}")
print(f"A-matrix:\n{gkz.a_matrix}")

# Generate TikZ visualisation
print("\n" + "=" * 80)
print("GENERATING TIKZ VISUALISATION")
print("=" * 80)

tikz_code = visualise_newton_polytope(
    gkz.support,
    title="Triangle Graph Newton Polytope",
    show_labels=True,
    show_fill=False,
)

print("\nTikZ code preview (first 500 chars):")
print("-" * 80)
print(tikz_code[:500] + "...")
print("-" * 80)

# Count edges in the output
visible_count = tikz_code.count("\\draw[thick,black]")
hidden_count = tikz_code.count("\\draw[dashed,gray]")

print("\nEdge statistics:")
print(f"  Visible edges (solid black): {visible_count}")
print(f"  Hidden edges (dashed gray): {hidden_count}")
print(f"  Total edges drawn: {visible_count + hidden_count}")

# Save to file
standalone_file = save_polytope_tikz(
    gkz.support,
    "triangle_graph_polytope.tex",
    title="Triangle Graph Newton Polytope",
    standalone=True,
    show_labels=True,
    show_fill=False,
)

print(f"\nSaved standalone document: {standalone_file}")
print("  Compile with: pdflatex triangle_graph_polytope.tex")

# Also save just the code
code_file = save_polytope_tikz(
    gkz.support,
    "triangle_graph_polytope_code.tex",
    title="Triangle Graph Newton Polytope",
    standalone=False,
    show_labels=True,
    show_fill=False,
)

print(f"Saved TikZ code: {code_file}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print(
    """
The Newton polytope is the convex hull of the monomial support of G(u).
For a triangle graph, the G polynomial will have multiple monomials
corresponding to different Feynman parameter configurations.

The polytope edges should represent the combinatorial structure of the
polynomial's support. Some edges may be "hidden" (behind faces when
viewed from a particular angle) and are drawn as dashed gray lines.
"""
)
