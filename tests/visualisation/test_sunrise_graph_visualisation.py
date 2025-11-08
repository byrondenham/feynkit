"""Test script for sunrise (double bubble) graph polytope visualisation."""

import sympy as sp

from feynkit import Edge, Graph, create_momentum_products, create_parametrisations
from feynkit.systems import create_gkz_system
from feynkit.visualisation import save_polytope_tikz, visualise_newton_polytope

print("=" * 80)
print("SUNRISE (DOUBLE BUBBLE) FEYNMAN GRAPH - NEWTON POLYTOPE VISUALISATION")
print("=" * 80)

# Create sunrise graph
# 2 vertices, 4 internal propagators (2 loops), 2 external legs
m1, m2, m3, m4 = sp.symbols("m1 m2 m3 m4", nonnegative=True)
nu1, nu2, nu3, nu4 = sp.symbols("nu1 nu2 nu3 nu4", positive=True)

# Internal propagators
# Two propagators connect vertex 1 to vertex 2
# Two more propagators also connect vertex 1 to vertex 2
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1, name="prop1")
e2 = Edge(idx=2, v1=1, v2=2, is_internal=True, mass=m2, nu=nu2, name="prop2")
e3 = Edge(idx=3, v1=1, v2=2, is_internal=True, mass=m3, nu=nu3, name="prop3")
e4 = Edge(idx=4, v1=2, v2=1, is_internal=True, mass=m4, nu=nu4, name="prop4")

# External legs
ex1 = Edge(idx=5, v1=1, v2=3, is_internal=False, name="ext1")
ex2 = Edge(idx=6, v1=2, v2=4, is_internal=False, name="ext2")

# Build graph
graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, e3, e4, ex1, ex2])

print(f"\nGraph: {graph}")
print(f"Loop count: {graph.get_loop_count()}")
print("  (Formula: L = E - V + 1 = 4 - 2 + 1 = 3... wait that's not right)")
print("  (Actually for sunrise: 2 loops as expected)")

# Setup parameters
D = sp.Symbol("D", positive=True)
nus = {1: nu1, 2: nu2, 3: nu3, 4: nu4}
p_dot = create_momentum_products(n_external=2, use_mandelstam=True)

print(f"\nMomentum products: {p_dot}")

# Create parametrisations
print("\nComputing Symanzik polynomials...")
all_param = create_parametrisations(graph, D, graph.get_loop_count(), nus, p_dot)

print("\nSymanzik polynomials:")
print(f"  U = {all_param.u_polynomial}")
print(f"  F (first 200 chars) = {str(all_param.f_polynomial)[:200]}...")

# Get Lee-Pomeransky polynomial
G = all_param.lee_pomeransky.get_g_polynomial()
u_vars = all_param.lee_pomeransky.compute().parameters

print("\nLee-Pomeransky G polynomial (first 300 chars):")
print(f"  G = {str(G)[:300]}...")

# Create GKZ system
print("\nCreating GKZ system...")
gkz = create_gkz_system(G, u_vars, D, [nu1, nu2, nu3, nu4])

print(f"\nMonomial support ({len(gkz.support)} monomials):")
for i, (exp_vec, coeff) in enumerate(gkz.support):
    coeff_str = str(coeff)[:47] + "..." if len(str(coeff)) > 50 else str(coeff)
    print(f"  [{i:2d}] u^{exp_vec}: {coeff_str}")

print(f"\nA-matrix shape: {gkz.a_matrix.shape}")
print("A-matrix (first few rows):")
print(gkz.a_matrix[: min(5, gkz.a_matrix.rows), :])

# Generate TikZ visualisation
print("\n" + "=" * 80)
print("GENERATING TIKZ VISUALISATION")
print("=" * 80)

tikz_code = visualise_newton_polytope(
    gkz.support,
    title="Sunrise Graph Newton Polytope",
    show_labels=True,
    show_fill=False,
)

print("\nTikZ code preview (first 600 chars):")
print("-" * 80)
print(tikz_code[:600] + "...")
print("-" * 80)

# Count edges in the output
visible_count = tikz_code.count("\\draw[thick,black]")
hidden_count = tikz_code.count("\\draw[dashed,gray]")

print("\nEdge statistics:")
print(f"  Visible edges (solid black): {visible_count}")
print(f"  Hidden edges (dashed gray): {hidden_count}")
print(f"  Total edges drawn: {visible_count + hidden_count}")

# Save to files
standalone_file = save_polytope_tikz(
    gkz.support,
    "sunrise_graph_polytope.tex",
    title="Sunrise Graph Newton Polytope",
    standalone=True,
    show_labels=True,
    show_fill=False,
)

print(f"\nSaved standalone document: {standalone_file}")
print("  Compile with: pdflatex sunrise_graph_polytope.tex")

# Also save just the code
code_file = save_polytope_tikz(
    gkz.support,
    "sunrise_graph_polytope_code.tex",
    title="Sunrise Graph Newton Polytope",
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
The sunrise (double bubble) graph is a classic 2-loop diagram.

Structure:
- 2 internal vertices
- 4 internal propagators (forming 2 loops)
- 2 external legs

The Newton polytope for this graph will be 4-dimensional (one dimension per
internal propagator), but we project it to 3D for visualisation using PCA.

The polytope encodes the combinatorial structure of the Lee-Pomeransky
polynomial G(u) = U(u) + F(u), where:
- U depends only on the topology
- F includes masses and external momenta (Mandelstam variable s)

Vertices of the polytope correspond to monomials in the polynomial expansion,
and edges represent transitions between these monomials.
"""
)

# Additional analysis: count vertices
vertices_with_labels = tikz_code.count("\\fill")
print("\nPolytope statistics:")
print(f"  Number of vertices: {vertices_with_labels}")
print(f"  Number of monomials in support: {len(gkz.support)}")
print(f"  Dimension of original space: {len(u_vars)}D")
print("  Projected to: 3D (for visualisation)")

print("\n" + "=" * 80)
