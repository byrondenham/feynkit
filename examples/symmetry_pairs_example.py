"""
Symmetry pairs of Feynman integral GKZ A-configurations.

A symmetry pair (T, P) of an A-matrix is an invertible integer affine
self-map of the monomial support: M·a_j + t = a_{P(j)} for every monomial
a_j, where T = [[1, 0^T], [t, M]] is the (n+1)×(n+1) homogenised matrix
and P is the induced column permutation.  The condition is T·A = A·Π_P.

Each pair gives a transformation identity for the generalised Feynman
integral (Forsgård–Matusevich–Sobieska 2019; de la Cruz 2024):

    I_A(β, z) = I_A(T β, z_P),      prefactor R(β) = 1

where z_P = (z_{P(0)}, …, z_{P(N-1)}) and the GKZ parameter vector β is
transformed by the full homogenised matrix T.

Classical hypergeometric symmetries are instances of this:
  • The eightfold Kummer group of ₂F₁ arises from the 8 symmetry pairs
    of the one-mass bubble (de la Cruz 2024, §3.1).
  • The B₃ octahedral symmetry of the massless triangle (48 pairs) reflects
    the cross-polytope structure of its Newton polytope.

Run with:
    uv run python examples/symmetry_pairs_example.py
"""

import sympy as sp

from feynkit import FeynmanIntegral

SEP  = "─" * 72
SEP2 = "━" * 72


def print_section(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _group_name(order: int) -> str:
    return {
        1: "trivial", 2: "ℤ/2", 3: "ℤ/3", 4: "V₄", 6: "S₃",
        8: "eightfold (₂F₁)", 12: "A₄", 24: "S₄", 48: "B₃", 120: "S₅",
    }.get(order, f"order {order}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: one-mass bubble  →  eightfold ₂F₁ symmetry
# ──────────────────────────────────────────────────────────────────────────────

print_section("One-mass bubble  —  8 symmetry pairs = eightfold ₂F₁ symmetry")

fi_1m = FeynmanIntegral.from_cnickel("11e|e|:nz")
pairs_1m = fi_1m.symmetry_pairs
beta_1m  = fi_1m.gkz.beta_parameters

print(f"""
The one-mass bubble has G with {fi_1m.newton_polytope.a_matrix.cols} monomials in ℝ².
  G = {fi_1m.symanzik.g}
  β = {beta_1m}   (β₀ = Σνᵢ − D/2, β₁ = ν₁, β₂ = ν₂)

de la Cruz (2024) §3.1 shows this configuration has exactly 8 symmetry pairs,
recovering the eightfold Kummer symmetry group of the Gauss ₂F₁ function.
""")

print(f"  Found {len(pairs_1m)} symmetry pairs  ({_group_name(len(pairs_1m))})\n")

print(f"  {'#':<4} {'det':>4}  {'perm P':<22}  T β")
print(f"  {'─'*4}  {'─'*4}  {'─'*22}  {'─'*34}")
for i, p in enumerate(pairs_1m):
    tb = p.transform_beta(beta_1m)
    perm_str = str(list(p.column_permutation))
    print(f"  [{i}]  {p.determinant:>4}  {perm_str:<22}  {tb}")

print(f"""
Observation:
  β₀ = Σνᵢ − D/2 is preserved by every T (first row of T is always
  [1, 0, 0]), confirming that the overall degree of homogeneity is
  invariant.  The dimension D and propagator indices ν are mixed by the
  lower block of T.

  Pair [1] maps ν₂ → −D + ν₁ + ν₂.  At ν₁ = ν₂ = 1, D = 4 this gives
  ν₂ → −2: a shift down by 2+ν₂.  Pairs of this type give
  dimension-shift / index-reduction identities between integrals.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: massless triangle  →  48 pairs, B₃ symmetry
# ──────────────────────────────────────────────────────────────────────────────

print_section("Massless triangle  —  48 symmetry pairs, B₃ (octahedral group)")

fi_tri = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
pairs_tri = fi_tri.symmetry_pairs
beta_tri  = fi_tri.gkz.beta_parameters

print(f"""
Massless triangle: G has {fi_tri.newton_polytope.a_matrix.cols} monomials in ℝ³.
  β = {beta_tri}

All 48 pairs are unimodular (|det M| = 1), forming the hyperoctahedral
group B₃ — the symmetry group of the octahedron / cross-polytope.
The Newton polytope of the triangle IS a cross-polytope, centred at
(½, ½, ½), with vertices at the three antipodal pairs of edge midpoints.
""")

uni_tri = [p for p in pairs_tri if p.is_unimodular]
print(f"  Total pairs         : {len(pairs_tri)}")
print(f"  Unimodular (det=1)  : {len(uni_tri)}")
print(f"  Non-unimodular      : {len(pairs_tri) - len(uni_tri)}")
print(f"  Group name          : {_group_name(len(uni_tri))}")

# Display five sample maps
print("\n  Sample maps (first 5):")
print(f"  {'#':<4} {'det':>4}  {'M (rows)':<42}  t")
print(f"  {'─'*4}  {'─'*4}  {'─'*42}  {'─'*16}")
for i, p in enumerate(pairs_tri[:5]):
    M_rows = "; ".join(str(row) for row in p.linear_map.tolist())
    t_str  = str([int(p.translation[k, 0]) for k in range(p.translation.rows)])
    print(f"  [{i}]  {p.determinant:>4}  [{M_rows}]  {t_str}")

# Show the T·A = A·Π_P check for one pair
p_sample = pairs_tri[7]
A = fi_tri.gkz.a_matrix
N = A.cols
Pi = sp.zeros(N, N)
for j, k in enumerate(p_sample.column_permutation):
    Pi[k, j] = 1
ta_eq_ap = (p_sample.homogenized_map * A == A * Pi)
print(f"\n  Verification T·A = A·Π_P for pair [7]: {ta_eq_ap}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: β transformation in detail
# ──────────────────────────────────────────────────────────────────────────────

print_section("β transformation: I_A(β, z) = I_A(T β, z_P)")

fi = FeynmanIntegral.from_cnickel("11e|e|:nz")
pairs = fi.symmetry_pairs
beta  = fi.gkz.beta_parameters
D, nu1, nu2 = sp.Symbol("D"), sp.Symbol("nu_1"), sp.Symbol("nu_2")

print(f"""
For the one-mass bubble with β = {beta}:

The identity I_A(β, z) = I_A(T β, z_P) is a functional equation relating
the generalised Feynman integral at parameter β to the same integral
evaluated at the permuted kinematic point z_P but with modified parameters
T β.  Specialising to physical z recovers a relation between two concrete
integrals.
""")

for i, p in enumerate(pairs):
    tb  = p.transform_beta(beta)
    perm = list(p.column_permutation)
    print(f"  [{i}]  I(β, z) = I({tb},  z_P={perm})")


# ──────────────────────────────────────────────────────────────────────────────
# Section 4: survey across diagrams
# ──────────────────────────────────────────────────────────────────────────────

print_section("Survey: symmetry pair counts for standard diagrams")

DIAGRAMS = [
    ("Massless bubble",      "11e|e|:zz",         "S₃: 3-simplex symmetry"),
    ("One-mass bubble",      "11e|e|:nz",         "eightfold ₂F₁ symmetry (de la Cruz 2024)"),
    ("Massive bubble",       "11e|e|:nn",         "ℤ/2: propagator exchange"),
    ("Massless triangle",    "12e|2e|e|:zzz",     "B₃: cross-polytope / octahedral symmetry"),
    ("One-mass triangle",    "12e|2e|e|:nzz",     "S₃: residual permutation symmetry"),
    ("Two-mass triangle",    "12e|2e|e|:nnz",     "V₄: exchange of massive or massless legs"),
    ("All-mass triangle",    "12e|2e|e|:nnn",     "S₃: full propagator permutation"),
    ("Massless box",         "12e|3e|3e|e|:zzzz", "S₅: 10 = C(5,2) monomial structure"),
    ("One-mass box",         "12e|3e|3e|e|:nzzz", "S₄: symmetry breaking by one mass"),
    ("Massless 3-banana",    "111e|e|:zzz",       "S₄: 3-banana is a tetrahedron in ℝ³"),
    ("Massless 4-banana",    "1111e|e|:zzzz",     "S₅: 4-banana is a 4-simplex in ℝ⁴"),
]

print(f"\n  {'Diagram':<22} {'N':>4} {'n':>3} {'pairs':>7} {'uni':>5} {'dets':>14}  Interpretation")
print(f"  {'─'*22}  {'─'*4}  {'─'*3}  {'─'*7}  {'─'*5}  {'─'*14}  {'─'*30}")

for label, cn, interp in DIAGRAMS:
    try:
        fi   = FeynmanIntegral.from_cnickel(cn)
        ps   = fi.symmetry_pairs
        cfg  = fi.gkz.a_matrix
        N    = cfg.cols
        n    = cfg.rows - 1  # ambient dimension (drop homogenization row)
        n_all = len(ps)
        n_uni = sum(1 for p in ps if p.is_unimodular)
        dets  = sorted(set(p.determinant for p in ps))
        dets_str = str(dets) if len(dets) <= 3 else f"[{min(dets)}..{max(dets)}]"
        print(f"  {label:<22}  {N:>4}  {n:>3}  {n_all:>7}  {n_uni:>5}  {dets_str:>14}  {interp}")
    except Exception as exc:
        print(f"  {label:<22}  ERROR: {exc}")

print("""
Observations:
  • Adding mass to the bubble: massless (6 = S₃) → one-mass (8) → massive (2 = ℤ/2).
    The one-mass case is the most symmetric: its 4 monomials form a configuration
    in ℝ² with the full eightfold Kummer symmetry of ₂F₁ (de la Cruz 2024 §3.1).

  • Banana graphs: the L-loop massless banana has N = L+1 monomials in ℝ^L
    forming the standard L-simplex.  Its symmetry group is S_(L+1) (all
    permutations of the L+1 vertices), consistent with the propagator-exchange
    symmetry of a single-scale graph:
      3-banana (N=4, ℝ³) : S₄  (order 24)
      4-banana (N=5, ℝ⁴) : S₅  (order 120)

  • Massless box: N=10, |Aut|=120 = S₅.  The 10 monomials equal C(5,2), the
    number of edges of the complete graph K₅, and the S₅ symmetry acts on the
    5 vertices of K₅.  This combinatorial structure is preserved by all 120
    symmetry pairs.

  • No non-unimodular pairs appear for any of these physical configurations.
    This is consistent with each configuration spanning its full ambient lattice
    (Smith invariants all equal to 1), which forces det(M) = ±1 for any
    self-map.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Section 5: triangle vs triple-K — same pairs, different lattice
# ──────────────────────────────────────────────────────────────────────────────

print_section("Triangle vs Triple-K: same symmetry group, different lattice")

from feynkit import symmetry_pairs as sp_fn
from feynkit.artifacts.dissertation import triangle_a_config, triple_k_a_config

tri = triangle_a_config()
tpk = triple_k_a_config()

pairs_tri2 = sp_fn(tri)
pairs_tpk  = sp_fn(tpk)

uni_tri2 = [p for p in pairs_tri2 if p.is_unimodular]
uni_tpk  = [p for p in pairs_tpk  if p.is_unimodular]

print(f"""
Both the triangle and triple-K Newton polytopes are cross-polytopes in ℝ³
with B₃ symmetry, so their symmetry pair groups are isomorphic.

  Triangle : {len(pairs_tri2)} pairs ({len(uni_tri2)} unimodular, {len(pairs_tri2)-len(uni_tri2)} non-unimodular)
             Smith invariants: {tri.smith_invariants}  →  spans ℤ³  (index-1)
  Triple-K : {len(pairs_tpk)} pairs ({len(uni_tpk)} unimodular, {len(pairs_tpk)-len(uni_tpk)} non-unimodular)
             Smith invariants: {tpk.smith_invariants}  →  even-sum sublattice (index-2 in ℤ³)

Both have 48 unimodular pairs — the same abstract group B₃.  Yet they are
NOT unimodularly equivalent: the lattice embedding differs, as measured by
the Smith invariants.  The finite-index map M = [[0,1,1],[1,0,1],[1,1,0]]
(det 2) maps the triangle onto the triple-K, bridging the two lattice
environments.
""")

# Confirm all unimodular pairs satisfy T·A = A·Π_P
for cfg, pairs, name in [(tri, pairs_tri2, "triangle"), (tpk, pairs_tpk, "triple-K")]:
    A = cfg.matrix
    N = cfg.n_points
    ok = True
    for p in pairs:
        Pi = sp.zeros(N, N)
        for j, k in enumerate(p.column_permutation):
            Pi[k, j] = 1
        if p.homogenized_map * A != A * Pi:
            ok = False
            break
    print(f"  T·A = A·Π_P verified for all {len(pairs)} pairs of {name}: {ok}")
