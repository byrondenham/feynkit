"""Complete analysis example: Generate full LaTeX and text reports with polytope."""

import sympy as sp

from feynkit import Edge, Graph, create_momentum_products, create_parametrisations
from feynkit.algebra import compute_toric_ideal_generators
from feynkit.io import (
    create_analysis_document,
    create_analysis_report,
    save_latex_document,
    save_text_report,
)
from feynkit.systems import create_gkz_system
from feynkit.visualisation import visualise_newton_polytope

print("=" * 80)
print("COMPLETE FEYNMAN INTEGRAL ANALYSIS")
print("=" * 80)

# Create triangle graph
m1, m2, m3 = sp.symbols("m1 m2 m3", nonnegative=True)
nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3", positive=True)

e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1, name="propagator_12")
e2 = Edge(idx=2, v1=2, v2=3, is_internal=True, mass=m2, nu=nu2, name="propagator_23")
e3 = Edge(idx=3, v1=3, v2=1, is_internal=True, mass=m3, nu=nu3, name="propagator_31")
ex1 = Edge(idx=4, v1=1, v2=4, is_internal=False, name="external_1")
ex2 = Edge(idx=5, v1=2, v2=5, is_internal=False, name="external_2")
ex3 = Edge(idx=6, v1=3, v2=6, is_internal=False, name="external_3")

graph = Graph(internal_vertices=3, external_legs=3, edges=[e1, e2, e3, ex1, ex2, ex3])

print(f"\nGraph: {graph}")

# Setup parameters
D = sp.Symbol("D", positive=True)
nus = {1: nu1, 2: nu2, 3: nu3}
p_dot = create_momentum_products(n_external=3, use_mandelstam=True)

# Create parametrisations
print("\nComputing parametrisations...")
all_param = create_parametrisations(graph, D, 1, nus, p_dot)

# Get all three parametrisations
schwinger_result = all_param.schwinger.compute()
feynman_result = all_param.feynman.compute()
lee_pom_result = all_param.lee_pomeransky.compute()

param_results = {
    "Schwinger": schwinger_result,
    "Feynman": feynman_result,
    "Lee-Pomeransky": lee_pom_result,
}

# Create GKZ system
print("Computing GKZ system...")
G = all_param.lee_pomeransky.get_g_polynomial()
u_vars = lee_pom_result.parameters
gkz = create_gkz_system(G, u_vars, D, [nu1, nu2, nu3])

# Compute toric ideal
print("Computing toric ideal...")
toric_gens = compute_toric_ideal_generators(gkz.a_matrix)
print(f"  Found {len(toric_gens)} generator(s)")

# Generate Newton polytope visualisation
print("Generating Newton polytope...")
polytope_tikz_code = visualise_newton_polytope(
    gkz.support,
    title="Newton Polytope",
    show_labels=True,
    show_fill=False,
)

# Wrap in figure environment for LaTeX
polytope_figure = "\n".join(
    [
        "\\begin{figure}[htbp]",
        "\\centering",
        polytope_tikz_code,
        "\\caption{Newton polytope of the Lee-Pomeransky polynomial $G(u) = U(u) + F(u)$. "
        "Vertices represent monomial exponent vectors, solid edges are visible from the viewing angle, "
        "and dashed edges are hidden behind faces.}",
        "\\label{fig:newton_polytope}",
        "\\end{figure}",
    ]
)

# Generate LaTeX document
print("\n" + "=" * 80)
print("GENERATING LATEX DOCUMENT")
print("=" * 80)

latex_doc = create_analysis_document(
    graph=graph,
    u_polynomial=all_param.u_polynomial,
    f_polynomial=all_param.f_polynomial,
    gkz_system=gkz,
    parametrisation_results=param_results,
    toric_generators=toric_gens,
    polytope_tikz=polytope_figure,
    title="Triangle Graph Analysis",
    author="Feynkit Analysis",
)

latex_file = save_latex_document(latex_doc, "triangle_analysis.tex")
print(f"\nSaved LaTeX document: {latex_file}")
print("  Compile with: pdflatex triangle_analysis.tex")
print("  (Run twice to resolve cross-references)")

# Generate text report
print("\n" + "=" * 80)
print("GENERATING TEXT REPORT")
print("=" * 80)

text_report = create_analysis_report(
    graph=graph,
    u_polynomial=all_param.u_polynomial,
    f_polynomial=all_param.f_polynomial,
    gkz_system=gkz,
    parametrisation_results=param_results,
    toric_generators=toric_gens,
    title="Triangle Graph Analysis Report",
)

text_file = save_text_report(text_report, "triangle_analysis.txt")
print(f"\nSaved text report: {text_file}")

# Print preview of text report
print("\n" + "=" * 80)
print("TEXT REPORT PREVIEW (first 1500 characters)")
print("=" * 80)
print(text_report[:1500])
print("...")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(f"  - {latex_file} (LaTeX document)")
print(f"  - {text_file} (Text report)")
print("\nYou can now:")
print(f"  1. Compile the LaTeX: pdflatex {latex_file}")
print(f"  2. Run again for refs: pdflatex {latex_file}")
print(f"  3. View the text report: cat {text_file}")
print("=" * 80)
