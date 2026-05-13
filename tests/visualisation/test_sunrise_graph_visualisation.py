"""Test script for sunrise (double bubble) graph polytope visualisation."""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.visualisation import save_polytope_tikz

print("=" * 80)
print("SUNRISE (DOUBLE BUBBLE) FEYNMAN GRAPH - NEWTON POLYTOPE VISUALISATION")
print("=" * 80)

# Sunrise graph: 2 vertices, 4 internal propagators (3 loops), 2 external legs.
m1, m2, m3, m4 = sp.symbols("m1 m2 m3 m4", nonnegative=True)
nu1, nu2, nu3, nu4 = sp.symbols("nu1 nu2 nu3 nu4", positive=True)

e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1, name="prop1")
e2 = Edge(idx=2, v1=1, v2=2, is_internal=True, mass=m2, nu=nu2, name="prop2")
e3 = Edge(idx=3, v1=1, v2=2, is_internal=True, mass=m3, nu=nu3, name="prop3")
e4 = Edge(idx=4, v1=2, v2=1, is_internal=True, mass=m4, nu=nu4, name="prop4")
ex1 = Edge(idx=5, v1=1, v2=3, is_internal=False, name="ext1")
ex2 = Edge(idx=6, v1=2, v2=4, is_internal=False, name="ext2")

graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, e3, e4, ex1, ex2])
integral = FeynmanIntegral(
    graph,
    propagator_exponents={1: nu1, 2: nu2, 3: nu3, 4: nu4},
)

print(f"\nGraph: {graph}")
print(f"Loop count: {integral.loop_count}")
print(f"\nMomentum products: {integral.momentum_products}")

print("\nComputing Symanzik polynomials...")
print("\nSymanzik polynomials:")
print(f"  U = {integral.symanzik.u}")
print(f"  F (first 200 chars) = {str(integral.symanzik.f)[:200]}...")

print("\nLee-Pomeransky G polynomial (first 300 chars):")
print(f"  G = {str(integral.symanzik.g)[:300]}...")

print("\nCreating GKZ system...")
gkz = integral.gkz

print(f"\nMonomial support ({len(gkz.support)} monomials):")
for i, (exp_vec, coeff) in enumerate(gkz.support):
    coeff_str = str(coeff)[:47] + "..." if len(str(coeff)) > 50 else str(coeff)
    print(f"  [{i:2d}] u^{exp_vec}: {coeff_str}")

print(f"\nA-matrix shape: {gkz.a_matrix.shape}")
print("A-matrix (first few rows):")
print(gkz.a_matrix[: min(5, gkz.a_matrix.rows), :])

print("\n" + "=" * 80)
print("GENERATING TIKZ VISUALISATION")
print("=" * 80)

tikz_code = integral.visualise_polytope(
    title="Sunrise Graph Newton Polytope",
    show_labels=True,
    show_fill=False,
)

print("\nTikZ code preview (first 600 chars):")
print("-" * 80)
print(tikz_code[:600] + "...")
print("-" * 80)

visible_count = tikz_code.count("\\draw[thick,black]")
hidden_count = tikz_code.count("\\draw[dashed,gray]")

print("\nEdge statistics:")
print(f"  Visible edges (solid black): {visible_count}")
print(f"  Hidden edges (dashed gray): {hidden_count}")
print(f"  Total edges drawn: {visible_count + hidden_count}")

standalone_file = save_polytope_tikz(
    integral.newton_polytope.support,
    "sunrise_graph_polytope.tex",
    title="Sunrise Graph Newton Polytope",
    standalone=True,
    show_labels=True,
    show_fill=False,
)
print(f"\nSaved standalone document: {standalone_file}")
print("  Compile with: pdflatex sunrise_graph_polytope.tex")

code_file = save_polytope_tikz(
    integral.newton_polytope.support,
    "sunrise_graph_polytope_code.tex",
    title="Sunrise Graph Newton Polytope",
    standalone=False,
    show_labels=True,
    show_fill=False,
)
print(f"Saved TikZ code: {code_file}")

vertices_with_labels = tikz_code.count("\\fill")
print("\nPolytope statistics:")
print(f"  Number of vertices: {vertices_with_labels}")
print(f"  Number of monomials in support: {len(gkz.support)}")
print(f"  Dimension of original space: {len(integral.symanzik.lp_parameters)}D")
print("  Projected to: 3D (for visualisation)")

print("\n" + "=" * 80)
