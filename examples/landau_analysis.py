"""
Landau singularity analysis via the principal A-determinant.

Demonstrates feynkit.landau for standard Feynman diagrams and the BMS
conformal simplex family.  For each integral the reduced principal
A-determinant is computed face by face; its irreducible factors are the
candidate Landau surfaces of the integral.

Usage
-----
    python examples/landau_analysis.py          # bubbles, triangle, BMS_3..5 (seconds)
    python examples/landau_analysis.py --all    # also the massless box (several minutes)
"""

from __future__ import annotations

import argparse

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.artifacts.conformal import _bms_g_polynomial
from feynkit.landau import LandauAnalysis, landau_analysis, landau_analysis_from_polynomial

# --- diagram constructors ----------------------------------------------------


def _bubble(m1, m2) -> FeynmanIntegral:
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1),
        Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2),
        Edge(idx=3, v1=1, v2=3, is_internal=False),
        Edge(idx=4, v1=2, v2=4, is_internal=False),
    ]
    g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    return FeynmanIntegral(g)


def _massless_triangle() -> FeynmanIntegral:
    z = sp.Integer(0)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z),
        Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z),
        Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z),
        Edge(idx=4, v1=1, v2=4, is_internal=False),
        Edge(idx=5, v1=2, v2=5, is_internal=False),
        Edge(idx=6, v1=3, v2=6, is_internal=False),
    ]
    g = Graph(internal_vertices=3, external_legs=3, edges=edges)
    return FeynmanIntegral(g)


def _massless_box() -> FeynmanIntegral:
    z = sp.Integer(0)
    nu = sp.symbols("nu1:5", positive=True)
    internal = [
        Edge(idx=i + 1, v1=(i % 4) + 1, v2=((i + 1) % 4) + 1, is_internal=True, mass=z, nu=nu[i])
        for i in range(4)
    ]
    external = [Edge(idx=5 + i, v1=i + 1, v2=5 + i, is_internal=False) for i in range(4)]
    g = Graph(internal_vertices=4, external_legs=4, edges=internal + external)
    return FeynmanIntegral(g)


# --- formatting helpers ------------------------------------------------------


def _show(name: str, result: LandauAnalysis) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")
    print(
        f"  Faces:  {len(result.face_discriminants)}  (non-trivial: {sum(1 for f in result.face_discriminants if f.discriminant != 1)})"
    )
    print(f"  # surfaces:   {len(result.landau_surfaces)}")
    if result.landau_surfaces:
        print("  Landau surfaces (irreducible factors of E_A):")
        for i, surf in enumerate(result.landau_surfaces, 1):
            print(f"    [{i}]  {surf}  =  0")
    else:
        print("  No kinematic Landau surfaces (integral has no normal threshold).")


# --- main --------------------------------------------------------------------


def main(include_box: bool = False) -> None:
    print("=" * 60)
    print("  LANDAU SINGULARITY ANALYSIS ,  feynkit.landau")
    print("  Reduced principal A-determinant E_A(G) over all faces")
    print("=" * 60)
    print()
    print("  Theory: the zero locus of E_A in kinematic space")
    print("  gives the leading Landau singularity surfaces of the")
    print("  Feynman integral (normal thresholds and IR singularities).")
    print()
    print("  Ref: Gelfand-Kapranov-Zelevinsky (1994) section 10.1")

    # -- 1. Massless bubble ---------------------------------------------------
    mb = _bubble(sp.Integer(0), sp.Integer(0))
    _show("Massless bubble  [m1 = m2 = 0]", landau_analysis(mb))

    # -- 2. Massive bubble ----------------------------------------------------
    m1, m2 = sp.symbols("m1 m2", nonnegative=True)
    bubble = _bubble(m1, m2)
    result_bubble = landau_analysis(bubble)
    _show("Massive bubble  [m1, m2 free]", result_bubble)
    # Verify zeros
    s = sp.Symbol("s", real=True)
    lp = result_bubble.principal_a_determinant
    t_norm = sp.simplify(lp.subs(s, (m1 + m2) ** 2))
    t_pseudo = sp.simplify(lp.subs(s, (m1 - m2) ** 2))
    print("\n  Verification:")
    print(f"    E_A at threshold   s = (m1+m2)^2  ->  {t_norm}")
    print(f"    E_A at pseudothres s = (m1-m2)^2  ->  {t_pseudo}")

    # -- 3. Equal-mass bubble -------------------------------------------------
    m = sp.Symbol("m", nonnegative=True)
    _show("Equal-mass bubble  [m1 = m2 = m]", landau_analysis(_bubble(m, m)))

    # -- 4. Massless triangle -------------------------------------------------
    _show("Massless triangle  [all propagators massless]", landau_analysis(_massless_triangle()))
    print()
    print("  The factors p_i^2 = 0 are the vertex contributions (external masses);")
    print("  the Gram determinant lambda(p_1^2, p_2^2, p_3^2) is the second-type singularity.")

    # -- 5. Massless box (opt-in: the discriminants take several minutes) ----
    if include_box:
        print("\n  [Computing massless box: this takes several minutes...]")
        _show(
            "Massless box  [4 massless propagators, 4 external legs]",
            landau_analysis(_massless_box()),
        )
    else:
        print("\n  [Massless box skipped; rerun with --all to include it]")

    # -- 6. BMS_3 conformal simplex -------------------------------------------
    print("\n  -- Conformal family (BMS simplex, Bzowski-McFadden-Skenderis) --")
    for n in range(3, 6):
        g = _bms_g_polynomial(n)
        params = [sp.Symbol(f"u_{i+1}") for i in range(n)]
        result_bms = landau_analysis_from_polynomial(g, params)
        _show(f"BMS_{n}  ({n}-point conformal simplex)", result_bms)

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print("  The principal A-determinant recovers:")
    print()
    print("  - Bubble:    normal threshold  s = (m1+m2)^2")
    print("               pseudothreshold   s = (m1-m2)^2, and p^2 = 0")
    print("  - Triangle:  p_i^2 = 0 and the Gram determinant")
    print("  - BMS_n:     null-momentum singularities p_i^2 = 0")
    print("               (conformal IR singularities)")
    print()
    print("  All results are derived purely from the Newton polytope")
    print("  of the Lee-Pomeransky G polynomial via edge discriminants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument("--all", action="store_true", help="also analyse the massless box (slow)")
    main(include_box=parser.parse_args().all)
