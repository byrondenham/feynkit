"""
Conformal simplex vs Lee-Pomeransky GKZ A-configurations.

Investigates whether the n=3 triangle / triple-K equivalence extends to n=4
and beyond by comparing:

  C_n LP    The massless 1-loop n-gon (C_n = n-cycle graph), computed by
            feynkit from the graph Lee-Pomeransky polynomial.  Uses n LP
            variables (one per internal edge).

  BMS_n     The Bzowski-McFadden-Skenderis n-point conformal simplex
            (BMS 2021; Caloro 2024).  G polynomial has 2n monomials in n
            variables: n "denominator" terms prod_{j!=i} u_j and n "scaling"
            terms u_i^2 prod_{j!=i} u_j.  For n=3 this is the triple-K.

  K_n LP    The complete graph K_n (n vertices, C(n,2) internal edges, one
            external leg per vertex).  For n=3, K_3 = C_3 = triangle.
            For n=4, K_4 is a 3-loop tetrahedron with 6 LP variables.

Run with:
    uv run python examples/conformal_simplex_comparison.py
"""

from feynkit import (
    AConfiguration,
    bms_simplex_a_config,
    complete_graph_a_config,
    massless_polygon_a_config,
)
from feynkit.a_configuration import finite_index_map
from feynkit.artifacts.dissertation import triangle_a_config, triple_k_a_config

SEP = "-" * 72

_AUT_THRESHOLD = 12  # skip automorphism computation for N above this (N=15 takes ~2min)


def print_section(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def describe(label: str, cfg: AConfiguration) -> None:
    print(f"\n  {label}")
    print(f"    A-matrix shape   : {cfg.matrix.shape[0]} x {cfg.matrix.shape[1]}")
    print(f"    N (monomials)    : {cfg.n_points}")
    print(f"    ambient_dim      : {cfg.ambient_dim}")
    print(f"    affine_dim       : {cfg.affine_dim}")
    print(f"    Smith invariants : {cfg.smith_invariants}")
    print(
        f"    normalised vol   : {cfg.normalized_volume}  (= GKZ holonomic rank for generic beta)"
    )
    if cfg.n_points <= _AUT_THRESHOLD:
        print(f"    |Aut(P)|         : {cfg.automorphisms().order}")
    else:
        print(f"    |Aut(P)|         : (skipped, N={cfg.n_points} too large)")


def compare(label_a: str, cfg_a: AConfiguration, label_b: str, cfg_b: AConfiguration) -> None:
    print(f"\n  {label_a}  vs  {label_b}")

    if cfg_a.ambient_dim != cfg_b.ambient_dim:
        print(
            f"    ambient dims differ ({cfg_a.ambient_dim} vs {cfg_b.ambient_dim})"
            " , no direct comparison"
        )
        return
    if cfg_a.n_points != cfg_b.n_points:
        print(
            f"    N differs ({cfg_a.n_points} vs {cfg_b.n_points})"
            " , no bijective map possible, not equivalent in any sense"
        )
        return

    r_uni = cfg_a.is_unimodular_equivalent_to(cfg_b)
    print(f"    Unimodular equiv : {r_uni.equivalent}", end="")
    if r_uni.equivalent and r_uni.witness_map is not None:
        print(f"  (witness U = {r_uni.witness_map.tolist()})", end="")
    print()

    if not r_uni.equivalent:
        r_aff = cfg_a.is_affinely_equivalent_to(cfg_b)
        print(f"    Affine equiv     : {r_aff.equivalent}")

        fim_ab = finite_index_map(cfg_a, cfg_b)
        fim_ba = finite_index_map(cfg_b, cfg_a)
        if fim_ab.found:
            print(
                f"    Finite-index map : {label_a} -> {label_b}  "
                f"det={fim_ab.determinant}  unimodular={fim_ab.is_unimodular}"
            )
            print(f"      M = {fim_ab.witness_matrix.tolist()}")
            print(f"      t = {[int(x) for x in fim_ab.translation]}")
        elif fim_ba.found:
            print(
                f"    Finite-index map : {label_b} -> {label_a}  "
                f"det={fim_ba.determinant}  unimodular={fim_ba.is_unimodular}"
            )
            print(f"      M = {fim_ba.witness_matrix.tolist()}")
        else:
            print("    Finite-index map : not found in either direction")


# ------------------------------------------------------------------------------
# n=3: the known case
# ------------------------------------------------------------------------------

print_section("n=3  (the known case)")

print("""
For n=3 the 1-loop triangle C_3 = K_3 (complete graph coincides with the
cycle), so the three families all converge.  The massless triangle and the
triple-K (BMS 3-point conformal simplex) are rationally affinely equivalent
but NOT unimodularly equivalent.
""")

tri = triangle_a_config()  # C_3 LP = K_3 LP (from dissertation)
tpk = triple_k_a_config()  # BMS n=3 (from dissertation)
bms3 = bms_simplex_a_config(3)  # must equal triple-K
cn3 = massless_polygon_a_config(3)  # must equal triangle
kn3 = complete_graph_a_config(3)  # must equal triangle (K_3=C_3)

assert tpk.smith_invariants == bms3.smith_invariants, "BMS_3 must match triple-K"
assert tri.smith_invariants == cn3.smith_invariants, "C_3 must match triangle"
assert tri.smith_invariants == kn3.smith_invariants, "K_3 must match triangle"
assert tri.normalized_volume == cn3.normalized_volume, "C_3 vol must match triangle"

describe("C_3 LP (triangle, from feynkit graph)", cn3)
describe("BMS n=3 (triple-K)", bms3)

print()
print("  Comparison:")
compare("C_3 LP (triangle)", cn3, "BMS n=3 (triple-K)", bms3)


# ------------------------------------------------------------------------------
# n=4: does the relation extend?
# ------------------------------------------------------------------------------

print_section("n=4  (does the relation extend?)")

print("""
For n=4, C_4 (the box) and K_4 are different graphs:
  C_4 = 4-cycle (box):  4 internal edges, 1 loop, 4 LP variables
  K_4 = complete graph: 6 internal edges, 3 loops, 6 LP variables

The BMS 4-point simplex uses 4 LP variables (one per external momentum).
Only C_4 lives in the same 4-dimensional ambient space as BMS_4.
""")

cn4 = massless_polygon_a_config(4)  # box, 4 LP vars
bms4 = bms_simplex_a_config(4)  # BMS 4-point, 4 LP vars
kn4 = complete_graph_a_config(4)  # K_4, 6 LP vars

describe("C_4 LP (massless box, 1-loop)", cn4)
describe("BMS n=4 (quadruple-K structure)", bms4)
describe("K_4 LP (complete graph, 3-loop)", kn4)

print()
print("  Comparisons:")
compare("C_4 LP (box)", cn4, "BMS n=4", bms4)
print()
compare("K_4 LP", kn4, "BMS n=4", bms4)


# ------------------------------------------------------------------------------
# n=5: further check
# ------------------------------------------------------------------------------

print_section("n=5  (further check)")

cn5 = massless_polygon_a_config(5)
bms5 = bms_simplex_a_config(5)

describe("C_5 LP (massless pentagon, 1-loop)", cn5)
describe("BMS n=5", bms5)

print()
print("  Comparison:")
compare("C_5 LP (pentagon)", cn5, "BMS n=5", bms5)


# ------------------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------------------

print_section("Summary table")

print(
    f"\n  {'n':>2}  {'config':<14}  {'N':>4}  {'dim':>4}  {'Smith':<16}  {'vol':>6}  {'|Aut|':>7}"
)
print(f"  {'-'*2}  {'-'*14}  {'-'*4}  {'-'*4}  {'-'*16}  {'-'*6}  {'-'*7}")

rows = []
for n in [3, 4, 5]:
    cn = massless_polygon_a_config(n)
    bms = bms_simplex_a_config(n)
    rows.append((n, f"C_{n} LP", cn))
    rows.append((n, f"BMS n={n}", bms))

for n, label, cfg in rows:
    smith_str = str(cfg.smith_invariants)
    aut_str = str(cfg.automorphisms().order) if cfg.n_points <= _AUT_THRESHOLD else " - "
    print(
        f"  {n:>2}  {label:<14}  {cfg.n_points:>4}  {cfg.ambient_dim:>4}  "
        f"{smith_str:<16}  {cfg.normalized_volume:>6}  {aut_str:>7}"
    )


# ------------------------------------------------------------------------------
# Diagnostic: why does n=3 work?
# ------------------------------------------------------------------------------

cn4_n = massless_polygon_a_config(4).n_points
bms4_n = bms_simplex_a_config(4).n_points
cn5_n = massless_polygon_a_config(5).n_points
bms5_n = bms_simplex_a_config(5).n_points
cn4_vol = massless_polygon_a_config(4).normalized_volume
bms4_vol = bms_simplex_a_config(4).normalized_volume

print_section("Why n=3 is special")

print(f"""
The triangle/triple-K equivalence (n=3) rests on three coincidences:

1.  C_3 = K_3.  The 3-cycle and the complete graph on 3 vertices are the
    same graph (triangle).  C_n and K_n differ for n >= 4:
      C_4 has 4 internal edges;  K_4 has C(4,2) = 6.
    As a consequence, C_3 and K_3 LP have the same 3 LP variables,
    matching the 3 LP variables of BMS_3.  For n >= 4, C_n has n LP
    variables but K_n has C(n,2) > n, giving mismatched ambient dimensions
    between K_n LP and BMS_n.

2.  Same monomial count.  C_3 G polynomial has U + F with 3 + 3 = 6
    monomials; BMS_3 (triple-K) also has 6 monomials (3 lower + 3 upper).
    This enables a point-bijective comparison.  For n >= 4 the counts
    diverge:
      n=4:  C_4 has {cn4_n} monomials vs BMS_4 has {bms4_n}
      n=5:  C_5 has {cn5_n} monomials vs BMS_5 has {bms5_n}
    Different monomial counts immediately rule out any bijective equivalence.

3.  Same normalised volume.  Both C_3 and BMS_3 have vol_0 = 4, giving the
    same GKZ holonomic rank for generic beta.
    For n=4 the volumes differ:  vol_0(C_4) = {cn4_vol}  vs  vol_0(BMS_4) = {bms4_vol}.
    Different volumes -> different holonomic ranks for generic beta ->
    different numbers of independent hypergeometric series in the GKZ
    solution space.

The G polynomial of the triangle C_3 has the same number of monomials as the
triple-K because:
  U(C_3) = u1 + u2 + u3  (3 monomials, from the 3 spanning trees)
  F(C_3) contains 3 bilinear terms (one per Mandelstam invariant)
  Triple-K has 3 "K-denominator" + 3 "scaling" monomials

For n=4:
  U(C_4) = u1 + u2 + u3 + u4  (4 degree-1 monomials, from the 4 spanning trees)
  F(C_4) contains 6 degree-2 terms (one per pair of edges: p_1^2, ..., p_4^2, s_12, s_23)
  Total C_4 monomials = {cn4_n} = 4 + 6
  BMS_4 monomials = {bms4_n} = 4 + 4

The mismatch of {cn4_n} vs {bms4_n} arises because C_4 has C(4,2)=6 kinematic
invariants (more than 4), while BMS_4 has exactly n=4 "scaling" monomials
and n=4 "denominator" monomials.

Conclusion: the triangle/triple-K relation is a unique n=3 coincidence with
no direct analogue at n >= 4 at the level of A-configurations and Newton
polytopes.  At n = 4 and 5 the GKZ systems for C_n LP and BMS_n are
inequivalent: they have different holonomic ranks for generic beta,
different numbers of independent hypergeometric-series solutions, and
non-isomorphic Newton polytopes.
""")
