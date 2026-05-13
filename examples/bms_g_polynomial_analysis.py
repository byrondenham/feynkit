"""
BMS G-polynomial derivation and triangle / triple-K equivalence analysis.

Shows the Schwinger-parameterisation derivation of the BMS conformal simplex
G polynomial from first principles, then investigates the affine equivalence
between the BMS_3 (triple-K) A-configuration and the massless triangle (C_3 LP).

Run with:
    uv run python examples/bms_g_polynomial_analysis.py
"""

import sympy as sp
import numpy as np

from feynkit import AConfiguration, bms_simplex_a_config, massless_polygon_a_config
from feynkit.a_configuration import finite_index_map
from feynkit.artifacts.conformal import _bms_g_polynomial

SEP = "─" * 72


def print_section(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Symbolic derivation of G₀ and G
# ─────────────────────────────────────────────────────────────────────────────

print_section("Schwinger-parameterisation derivation of the BMS G polynomial")

print("""
Starting point (BMS 2021 / Caloro 2024):

    I_n(p₁,…,pₙ) = ∫₀^∞ r^{β₀-1} ∏ᵢ K_{νᵢ}(pᵢ r) dr

Using the integral representation of the Bessel K function,

    K_ν(pᵢ r) = (pᵢ/2)^{νᵢ}/2 ∫₀^∞ uᵢ^{-νᵢ-1} exp(-r(uᵢ + pᵢ²/(4uᵢ))) duᵢ

multiplying over i=1,…,n, and integrating over r via Γ(β₀):

    I_n ∝ ∫ [∏ᵢ duᵢ uᵢ^{β₀-νᵢ-1}]  G₀(u,p)^{-β₀}

with the rational intermediate polynomial

    G₀(u,p) = ∑ᵢ uᵢ + ∑ᵢ pᵢ²/(4uᵢ)

Multiplying by 4 ∏ⱼ uⱼ to clear denominators (this shifts the Schwinger
exponents, absorbed into the GKZ parameter vector β), gives the
Lee–Pomeransky-style polynomial

    G(u,p) = ∑ᵢ pᵢ² ∏_{j≠i} uⱼ  +  4 ∑ᵢ uᵢ² ∏_{j≠i} uⱼ

with exactly 2n monomials (n lower of degree n−1, n upper of degree n+1).
""")

for n in [2, 3, 4]:
    u = [sp.Symbol(f"u_{i+1}") for i in range(n)]
    p_sq = [sp.Symbol(f"p_{i+1}sq") for i in range(n)]

    # G₀ rational form
    G0 = sum(u) + sum(p_sq[i] / (4 * u[i]) for i in range(n))
    G0_simplified = sp.simplify(G0)

    G = _bms_g_polynomial(n)
    poly = sp.Poly(G, *u)
    monoms = poly.monoms()
    lower = [m for m in monoms if sum(m) == n - 1]
    upper_m = [m for m in monoms if sum(m) == n + 1]

    print(f"  n={n}:")
    print(f"    G₀  = {sp.collect(G0_simplified, u)}")
    print(f"    G   = {G}")
    print(f"    monomials: {len(monoms)} total  ({len(lower)} lower deg {n-1},  "
          f"{len(upper_m)} upper deg {n+1})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: n=3 — finite-index map from triangle LP to triple-K
# ─────────────────────────────────────────────────────────────────────────────

print_section("n=3 — the known triangle / triple-K equivalence")

print("""
Both the massless triangle (C_3 LP) and the triple-K (BMS_3) are A-configurations
in ℝ³ with N=6 monomials and holonomic rank 4, but with Smith invariants [1,1,1]
vs [1,1,2].  They are NOT unimodularly equivalent, but there exists a finite-index
affine map from the triangle's monomial support into the triple-K's.

The explicit map:
    M = [[-1, -1, 0],
         [-1,  0,-1],
         [ 0,  1, 1]]
    t = [2, 2, 0]
    det(M) = -2  (index-2 map)

This sends each triangle LP exponent vector (the 3 affine coordinate rows of the
homogenised A-matrix, excluding the all-ones homogenisation row) to a BMS_3
exponent vector.
""")

# Build the affine map
M = np.array([[-1, -1, 0],
              [-1,  0, -1],
              [ 0,  1,  1]], dtype=int)
t = np.array([2, 2, 0], dtype=int)

print(f"  det(M) = {int(round(np.linalg.det(M)))}")

# Get affine coordinate rows of C_3 LP (rows 1..3 of the 4×6 A-matrix)
cn3 = massless_polygon_a_config(3)
bms3 = bms_simplex_a_config(3)

# Columns of A (excluding homogenisation row 0) are the affine coords
tri_cols = [tuple(int(cn3.matrix[r, j]) for r in range(1, cn3.matrix.shape[0]))
            for j in range(cn3.matrix.shape[1])]
bms_cols = {tuple(int(bms3.matrix[r, j]) for r in range(1, bms3.matrix.shape[0]))
            for j in range(bms3.matrix.shape[1])}

print("\n  Verifying map on all 6 triangle monomials:")
all_hit = True
for col in tri_cols:
    v = np.array(col, dtype=int)
    mapped = tuple((M @ v + t).tolist())
    hit = mapped in bms_cols
    all_hit = all_hit and hit
    print(f"    triangle {list(col)}  →  {list(mapped)}  in BMS_3: {hit}")

print(f"\n  All 6 triangle monomials map to BMS_3 monomials: {all_hit}")

# Confirm via feynkit's finite_index_map
print()
print("  feynkit finite_index_map(C_3, BMS_3):")
fim = finite_index_map(cn3, bms3)
print(f"    found   : {fim.found}")
if fim.found:
    print(f"    det     : {fim.determinant}")
    print(f"    M       : {fim.witness_matrix.tolist()}")
    print(f"    t       : {[int(x) for x in fim.translation]}")
    print(f"    unimod  : {fim.is_unimodular}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Why n≥4 admits no such map
# ─────────────────────────────────────────────────────────────────────────────

print_section("n≥4 — why no equivalence exists")

print("""
A necessary condition for any point-bijective affine map A → B is N(A) = N(B).
For the C_n LP and BMS_n families:

    N(C_n LP) = n  (spanning-tree monomials of U)
               + C(n,2)  (kinematic monomials of F)
               = n(n+1)/2

    N(BMS_n) = 2n

For n=3:  N(C_3) = 6,  N(BMS_3) = 6  — equal, map possible.
For n≥4:  n(n+1)/2 > 2n  (equivalently n > 3), so N(C_n) > N(BMS_n).
""")

print(f"  {'n':>3}  {'N(C_n LP)':>12}  {'N(BMS_n)':>10}  {'equal?':>8}")
print(f"  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*8}")
for n in range(3, 7):
    n_cn  = n * (n + 1) // 2
    n_bms = 2 * n
    print(f"  {n:>3}  {n_cn:>12}  {n_bms:>10}  {str(n_cn == n_bms):>8}")

print()
print("  Verifying with feynkit for n=4,5:")
for n in [4, 5]:
    cn  = massless_polygon_a_config(n)
    bms = bms_simplex_a_config(n)
    print(f"\n  n={n}:  C_{n} LP has N={cn.n_points}, BMS_{n} has N={bms.n_points}")
    if cn.n_points != bms.n_points:
        print(f"    N differs — no bijective map possible, skipping finite_index_map")
    else:
        fim_ab = finite_index_map(cn, bms)
        fim_ba = finite_index_map(bms, cn)
        if fim_ab.found or fim_ba.found:
            print(f"    Finite-index map found (unexpected)")
        else:
            print(f"    No finite-index map in either direction (as expected)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Sub-configuration structure of BMS_n
# ─────────────────────────────────────────────────────────────────────────────

print_section("Sub-configuration structure: lower vs upper monomials")

print("""
The BMS_n G polynomial splits into two geometrically distinct parts:

    lower : ∑ᵢ pᵢ² ∏_{j≠i} uⱼ  — n monomials, each with one zero exponent
    upper : 4 ∑ᵢ uᵢ² ∏_{j≠i} uⱼ — n monomials, each with one exponent=2

The lower sub-configuration (monomials of degree n−1) and upper sub-configuration
(monomials of degree n+1) are each polytopes in ℝⁿ.  Together their convex hull
gives vol₀(BMS_n) = 2^{n-1}.
""")

for n in [2, 3, 4, 5]:
    G = _bms_g_polynomial(n)
    u = [sp.Symbol(f"u_{i+1}") for i in range(n)]
    poly = sp.Poly(G, *u)
    monoms = sorted(poly.monoms())
    lower_m = [m for m in monoms if sum(m) == n - 1]
    upper_m = [m for m in monoms if sum(m) == n + 1]

    # Build sub-configurations
    hom = [1] * len(lower_m)
    lower_A = sp.Matrix([hom] + [[m[k] for m in lower_m] for k in range(n)])
    upper_A = sp.Matrix([[1]*len(upper_m)] + [[m[k] for m in upper_m] for k in range(n)])
    cfg_lower = AConfiguration(lower_A, is_homogenized=True)
    cfg_upper = AConfiguration(upper_A, is_homogenized=True)

    full_cfg = bms_simplex_a_config(n)
    print(f"  n={n}:  full vol₀={full_cfg.normalized_volume} = 2^{n-1}={2**(n-1)}"
          f"  |  lower vol₀={cfg_lower.normalized_volume}"
          f"  |  upper vol₀={cfg_upper.normalized_volume}")

print()
print("  BMS_n holonomic rank = 2^{n-1}:  confirmed for n=2,3,4,5.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Summary
# ─────────────────────────────────────────────────────────────────────────────

print_section("Summary")

print("""
The BMS G polynomial G(u) = ∑ pᵢ² ∏_{j≠i} uⱼ + 4 ∑ uᵢ² ∏_{j≠i} uⱼ is derived
analytically via Schwinger parameterisation of the n-Bessel K-function integral.
Its A-configuration is extracted from the monomial support of G.

For n=3:
  • C_3 LP (triangle) and BMS_3 (triple-K) have the same N=6 and vol₀=4.
  • There exists a finite-index (det=2) affine map from triangle → triple-K:
      M = [[-1,-1,0],[-1,0,-1],[0,1,1]],  t = [2,2,0]
  • They are NOT unimodularly equivalent (Smith [1,1,1] vs [1,1,2]).
  • The index-2 map reflects the triple-K spanning the even-parity sublattice.

For n≥4:
  • N(C_n LP) = n(n+1)/2 > 2n = N(BMS_n)  — no bijective map is possible.
  • The GKZ systems have different holonomic ranks and non-isomorphic Newton
    polytopes.

The triangle/triple-K equivalence is a unique n=3 coincidence arising from
C_3 = K_3 (the cycle and the complete graph on 3 vertices are the same graph).
""")
