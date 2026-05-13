"""Example: Visualising Newton polytope for a Feynman integral."""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.visualisation import save_polytope_tikz

# Create triangle diagram
m1, m2, m3 = sp.symbols("m1 m2 m3", nonnegative=True)
nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3", positive=True)

e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
e2 = Edge(idx=2, v1=2, v2=3, is_internal=True, mass=m2, nu=nu2)
e3 = Edge(idx=3, v1=3, v2=1, is_internal=True, mass=m3, nu=nu3)
ex1 = Edge(idx=4, v1=1, v2=4, is_internal=False)
ex2 = Edge(idx=5, v1=2, v2=5, is_internal=False)
ex3 = Edge(idx=6, v1=3, v2=6, is_internal=False)

graph = Graph(internal_vertices=3, external_legs=3, edges=[e1, e2, e3, ex1, ex2, ex3])
integral = FeynmanIntegral(graph, propagator_exponents={1: nu1, 2: nu2, 3: nu3})

print("=" * 80)
print("NEWTON POLYTOPE VISUALISATION FOR TRIANGLE DIAGRAM")
print("=" * 80)

print(f"\nG polynomial: {integral.symanzik.g}")

print(f"\nMonomial support ({len(integral.newton_polytope.support)} monomials):")
for i, (exp_vec, coeff) in enumerate(integral.newton_polytope.support):
    print(f"  [{i}] u^{exp_vec}: {coeff}")

print("\n" + "=" * 80)
print("GENERATING TIKZ CODE")
print("=" * 80)

tikz_code = integral.visualise_polytope(
    title="Triangle Diagram Newton Polytope",
    show_labels=True,
    show_fill=False,
)

print("\nGenerated TikZ code:")
print("-" * 80)
print(tikz_code)
print("-" * 80)

print("\nNote: Hidden edges (behind the polytope) are shown as dashed gray lines.")
print("      Visible edges (in front) are shown as solid black lines.")

print("\n" + "=" * 80)
print("SAVING TO FILES")
print("=" * 80)

# Save standalone document (can be compiled with pdflatex)
standalone_file = save_polytope_tikz(
    integral.newton_polytope.support,
    "triangle_polytope_standalone.tex",
    title="Triangle Diagram Newton Polytope",
    standalone=True,
    show_labels=True,
    show_fill=False,
)
print(f"\nSaved standalone document: {standalone_file}")
print("  Compile with: pdflatex triangle_polytope_standalone.tex")

# Save just the TikZ code (for inclusion in larger documents)
code_file = save_polytope_tikz(
    integral.newton_polytope.support,
    "triangle_polytope_code.tex",
    title="Triangle Diagram Newton Polytope",
    standalone=False,
    show_labels=True,
    show_fill=False,
)
print(f"\nSaved TikZ code: {code_file}")
print("  Include in document with: \\input{triangle_polytope_code.tex}")

print("\n" + "=" * 80)
