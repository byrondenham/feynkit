"""
Automorphism survey: polytope and graph symmetries of standard Feynman integrals.

For each diagram:
  1. Compute the unimodular automorphism group Aut(P) of the Newton polytope of G.
  2. Compute the graph automorphism group (vertex permutations preserving topology+mass).
  3. Identify the coefficient-preserving subgroup, whose elements relate parameter vectors.

Run with:
    uv run python examples/automorphism_survey.py
"""

from feynkit import FeynmanIntegral
from feynkit.normal_forms.polytope_automorphisms import (
    coefficient_preserving_indices,
    compute_graph_automorphisms,
    compute_polytope_automorphisms,
)

DIAGRAMS = [
    ("Massless bubble", "11e|e|:zz"),
    ("Massive bubble", "11e|e|:nn"),
    ("One-mass bubble", "11e|e|:nz"),
    ("Massless triangle", "12e|2e|e|:zzz"),
    ("One-mass triangle", "12e|2e|e|:nzz"),
    ("Two-mass triangle", "12e|2e|e|:nnz"),
    ("All-mass triangle", "12e|2e|e|:nnn"),
    ("Massless box", "12e|3e|3e|e|:zzzz"),
    ("One-mass box", "12e|3e|3e|e|:nzzz"),
    ("Massless 3-prop banana", "111e|e|:zzz"),
    ("Massive 3-prop banana", "111e|e|:nnn"),
    ("Massless 4-prop banana", "1111e|e|:zzzz"),
]

SEP = "-" * 72


def _describe_group(order: int) -> str:
    known = {
        1: "trivial",
        2: "Z/2",
        4: "V_4 or Z/4",
        6: "S_3 or Z/6",
        8: "D_4 or Z/8",
        12: "A_4 or D_6",
        24: "S_4",
        48: "B_3",
        120: "B_4?",
    }
    return known.get(order, "?")


print(SEP)
print("Feynkit, Automorphism Survey")
print(SEP)
print(f"{'Diagram':<28} {'|Aut(P)|':>8} {'|Aut(G)|':>8} {'|Coeff-pres.|':>14}  Notes")
print(SEP)

for label, cnickel in DIAGRAMS:
    try:
        fi = FeynmanIntegral.from_cnickel(cnickel)
    except Exception as exc:
        print(f"{label:<28}  ERROR: {exc}")
        continue

    try:
        pts = fi.newton_polytope.points
        poly_auts = compute_polytope_automorphisms(pts)
        graph_auts = compute_graph_automorphisms(fi.graph)
        cp_idx = coefficient_preserving_indices(fi, poly_auts)
        poly_ord = poly_auts.order
        graph_ord = len(graph_auts)
        cp_ord = len(cp_idx)
        n_orbits = len(poly_auts.vertex_orbits)
        note = f"({_describe_group(poly_ord)}), {n_orbits} orbit{'s' if n_orbits != 1 else ''}"
        print(f"{label:<28} {poly_ord:>8} {graph_ord:>8} {cp_ord:>14}  {note}")
    except Exception as exc:
        print(f"{label:<28}  FAILED: {exc}")

print(SEP)
print()
print("Definitions")
print("-----------")
print("|Aut(P)|   Unimodular automorphism group of Newton polytope of G.")
print("           (U, t) with U in GL_n(Z), |det U|=1, {Up+t : p in P} = P.")
print()
print("|Aut(G)|   Vertex permutations of the Feynman graph preserving")
print("           topology and mass colouring. For multi-edge graphs (bananas)")
print("           this misses edge-permutation symmetries.")
print()
print("|Coeff-pres.| Automorphisms (U,t) in Aut(P) that also preserve the")
print("           coefficients of G. Each gives I_A(beta, z) = I_A(T beta, z),")
print("           with T = [[1, 0], [t, U]], a relation between parameter vectors")
print("           at one kinematic point.")
print()

# Detailed view for one interesting case.
print(SEP)
print("Detail: massless bubble")
print(SEP)
fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
print(f"G = {fi.symanzik.g}")
print()
auts = fi.polytope_automorphisms
print(f"|Aut(P)| = {auts.order}  (S_3: automorphisms of a 2-simplex)")
print(f"Vertex orbits: {auts.vertex_orbits}  (single orbit, all points equivalent)")
print()
cp = coefficient_preserving_indices(fi, auts)
print(f"Coefficient-preserving indices: {cp}")
print("Maps:")
for k in cp:
    U, t = auts.maps[k]
    vperm = auts.vertex_permutations[k]
    print(f"  k={k}: U={U.tolist()}, t={t.T.tolist()}, vperm={vperm}")
