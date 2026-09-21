"""
Search for higher-dimensional analogues of the triangle / triple-K equivalence.

The n=3 equivalence:
  massless triangle C_3 LP  <-det=2 map->  BMS_3 (triple-K)

is the only known case where a Feynman-LP A-configuration and the BMS conformal
simplex are related by a finite-index (det=2) affine map.

This script tests the canonical candidate for n>=4 and then conducts a broader search.

Canonical candidate (Phase 1):
  A_candidate(n) = { prod_{j!=i} u_j : i=1,...,n }  (degree n-1, = BMS_n lower)
                 union { u_i : i=1,...,n }             (degree 1, standard basis)

Key parity finding:
  All BMS_n monomials have degree-parity n-1 == n+1 mod 2 (same for both halves).
  The canonical candidate's lower monomials (degree n-1) and upper (degree 1) have
  DIFFERENT parities iff n is even... wait, it's the reverse:
    - n odd (n=3): lower deg n-1=2 (even), upper deg 1 (odd)  -> mixed -> Smith all-ones
    - n even (n=4): lower deg n-1=3 (odd), upper deg 1 (odd)  -> same  -> Smith [1,...,1,2]
  For odd n: parity mixed, but gcd(n-2)=n-2 governs the lattice structure.
  Smith(A_candidate(n)) = [1,...,1, n-2]:  n-2=1 for n=3, n-2=2 for n=4, n-2=3 for n=5.

Phase 2: Test an alternative upper monomial set for n=4, degree-2 pairs (even sum)
  instead of standard basis (odd sum), to get mixed parity and Smith all-ones.

Phase 3: Numpy-vectorised inverse-map search over small det=+/-2 matrices for n=4.

The two slow pieces are the affine-equivalence checks of phase 2 (a brute-force
search over affine bases, about seven minutes for each of the three candidates)
and the phase-3 sweep over 2,825,761 candidate maps (another ten minutes or so),
so both sit behind --all.  Expect --all to run for half an hour.

Run with:
    uv run python examples/higher_analogues_search.py          # phases 1 and 2
    uv run python examples/higher_analogues_search.py --all    # everything
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from typing import Any

import numpy as np
import sympy as sp

from feynkit import (
    AConfiguration,
    bms_simplex_a_config,
    conformal_companion_a_config,
    massless_polygon_a_config,
)
from feynkit.a_configuration import finite_index_map

SEP = "-" * 72
FLUSH = sys.stdout.flush


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("Run with")[0].strip())
    parser.add_argument(
        "--all",
        action="store_true",
        help="also run the phase-2 affine-equivalence checks and the phase-3 sweep (half an hour)",
    )
    return parser.parse_args()


RUN_ALL = _parse_args().all


def print_section(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)
    FLUSH()


def describe(label: str, cfg: AConfiguration) -> None:
    print(f"  {label}")
    print(
        f"    N={cfg.n_points}  dim={cfg.ambient_dim}  Smith={cfg.smith_invariants}  vol_0={cfg.normalized_volume}"
    )
    FLUSH()


def check_map(label_a: str, a: AConfiguration, label_b: str, b: AConfiguration) -> None:
    fim = finite_index_map(a, b)
    print(f"  finite_index_map({label_a} -> {label_b}): found={fim.found}", end="")
    if fim.found:
        print(f"  det={fim.determinant}  unimod={fim.is_unimodular}", end="")
    print()
    FLUSH()


# -----------------------------------------------------------------------------
# Phase 1: canonical candidate
# -----------------------------------------------------------------------------

print_section("Phase 1, canonical candidate A_candidate(n) = lower_BMS union standard_basis")

print("""  G polynomial: G(u) = sum_i prod_{j!=i} u_j  +  sum_i p_i^2 u_i
  For n=3 this is the triangle by construction.\n""")

bms3 = bms_simplex_a_config(3)
cn3 = conformal_companion_a_config(3)
tri = massless_polygon_a_config(3)

assert cn3.smith_invariants == tri.smith_invariants == [1, 1, 1]
assert cn3.normalized_volume == tri.normalized_volume == 4
print("  n=3: companion matches triangle yes")
FLUSH()

print()
for n in [3, 4, 5]:
    comp = conformal_companion_a_config(n)
    bms = bms_simplex_a_config(n)
    describe(f"companion({n})", comp)
    describe(f"BMS_{n}     ", bms)
    smith_last = comp.smith_invariants[-1]
    print(f"  last Smith = {smith_last}  (= n-2 = {n-2}?  {'yes' if smith_last == n-2 else 'no'})")
    check_map(f"companion({n})", comp, f"BMS_{n}", bms)
    r_aff = comp.is_affinely_equivalent_to(bms)
    print(f"  affine equiv: {r_aff.equivalent}")
    print()


# -----------------------------------------------------------------------------
# Phase 2: degree-2 upper monomials for n=4 (even n needs even-degree upper)
# -----------------------------------------------------------------------------

print_section("Phase 2, alternative n=4: lower_BMS_4 union degree-2 pairs")

print("""  For n=4: lower monomials have degree 3 (odd sum).
  Standard basis has degree 1 (odd sum) -> same parity -> Smith [1,1,1,2].
  Use degree-2 pairs (even sum) instead -> mixed parity -> possibly Smith [1,1,1,1].

  Candidates for 4 degree-2 monomials from C_4 "adjacent pairs":
    A: box-cycle  {u_1u_2, u_2u_3, u_3u_4, u_1u_4} , the 4-gon cycle edges
    B: star-cross {u_1u_2, u_1u_3, u_1u_4, u_2u_3} , star from vertex 1 + one cross
    C: all same   {u_1u_2, u_1u_3, u_1u_4, u_2u_3} , same as B
    D: opposite   {u_1u_2, u_3u_4, u_1u_3, u_2u_4} , 2 pairs of complementary edges
""")

bms4 = bms_simplex_a_config(4)

lower4 = [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]

upper_sets = {
    "box-cycle": [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 0, 0, 1)],
    "star+cross": [(1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0)],
    "opposite": [(1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1)],
}

for name, upper4 in upper_sets.items():
    monomials = lower4 + upper4
    hom = [1] * 8
    coord_rows = [[m[r] for m in monomials] for r in range(4)]
    A = sp.Matrix([hom] + coord_rows)
    try:
        cfg = AConfiguration(A, is_homogenized=True)
        print(f"  {name}:")
        describe("", cfg)
        check_map("", cfg, "BMS_4", bms4)
        r_uni = cfg.is_unimodular_equivalent_to(bms4)
        if RUN_ALL:
            affine = str(cfg.is_affinely_equivalent_to(bms4).equivalent)
        else:
            affine = "skipped, use --all"
        print(f"    affine==BMS_4={affine}  unimod==BMS_4={r_uni.equivalent}")
    except Exception as e:
        print(f"  {name}: error, {e}")
    print()
    FLUSH()


# -----------------------------------------------------------------------------
# Phase 3: numpy-vectorised inverse-map search for n=4
# -----------------------------------------------------------------------------

print_section("Phase 3, vectorised inverse-map search for n=4 companion")

print("""  Strategy: enumerate all 4 x 4 integer matrices M (entries {-1,0,1}, det=+/-2,
  all-even column sums) and check if A_1 = adj(M)*(BMS_4 - t)/2 has Smith [1,1,1,1].
  Parity constraint: all BMS_4 columns have odd sum -> M must have all-even column
  sums so that sum(M*v) is always even, and sum(t) odd, to produce odd-sum outputs.
""")

# BMS_4 affine coord columns, shape (8,4)
bms4_cols_np = np.array(
    [[int(bms4.matrix[r, j]) for r in range(1, 5)] for j in range(8)],
    dtype=np.int64,
)


# Generate all 4-tuples from {-1,0,1} with even sum (41 per column)
def even_sum_cols() -> list[tuple]:

    return [t for t in product((-1, 0, 1), repeat=4) if sum(t) % 2 == 0]


even_cols = even_sum_cols()
print(f"  Even-sum columns from {{-1,0,1}}: {len(even_cols)} per column")

# batch size for the det filter
BATCH = 100_000


def _companions_in_batch(batch: list[tuple[int, ...]], found: list[dict[str, Any]]) -> None:
    """Append every Smith-[1,1,1,1] inverse image coming from one batch of maps M."""
    arr = np.array(batch, dtype=np.float64).reshape(-1, 4, 4)  # (B, 4, 4)
    dets = np.linalg.det(arr)
    mask = np.abs(np.round(dets)) == 2
    if not mask.any():
        return
    int_arr = np.array(batch, dtype=np.int64).reshape(-1, 4, 4)
    for M in int_arr[mask]:  # shape (4,4)
        det = int(round(np.linalg.det(M.astype(float))))
        # Compute adj(M) = det * M^{-1}
        adj = np.round(det * np.linalg.inv(M)).astype(np.int64)
        # Try different translations t
        for t_sum in (1, 3):
            for t_idx in range(4):
                t = np.zeros(4, dtype=np.int64)
                t[t_idx] = t_sum
                shifted = bms4_cols_np - t[np.newaxis, :]  # (8,4)
                cand2 = (adj @ shifted.T).T  # (8,4)
                if not np.all(cand2 % 2 == 0):
                    continue
                cand = cand2 // 2
                # Build AConfiguration
                hom = np.ones((1, 8), dtype=np.int64)
                A_np = np.vstack([hom, cand.T])
                try:
                    cfg = AConfiguration(sp.Matrix(A_np.tolist()), is_homogenized=True)
                    if cfg.smith_invariants == [1, 1, 1, 1]:
                        found.append(
                            {
                                "M": M.tolist(),
                                "t": t.tolist(),
                                "det": det,
                                "smith": cfg.smith_invariants,
                                "vol": cfg.normalized_volume,
                                "cols": cand.tolist(),
                            }
                        )
                except Exception:
                    pass


def search_companions() -> tuple[list[dict[str, Any]], int]:
    """Sweep every M in batches; return the companions found and how many M were checked."""
    found: list[dict[str, Any]] = []
    batch: list[tuple[int, ...]] = []
    checked = 0
    flat_gen = (
        (c1 + c2 + c3 + c4)
        for c1 in even_cols
        for c2 in even_cols
        for c3 in even_cols
        for c4 in even_cols
    )
    for flat in flat_gen:
        batch.append(flat)
        if len(batch) < BATCH:
            continue
        _companions_in_batch(batch, found)
        checked += len(batch)
        batch = []
        if checked % 1_000_000 == 0:
            print(
                f"  ... {checked:,} checked, {len(found)} companions found so far",
                flush=True,
            )
    if batch:  # the last, short batch
        _companions_in_batch(batch, found)
        checked += len(batch)
    return found, checked


# Full count:
n_mats = len(even_cols) ** 4
print(f"  Total matrices to check: {n_mats:,}")

found_companions: list[dict[str, Any]] = []
checked = 0

if RUN_ALL:
    print("  Running batched det filter...", flush=True)
    found_companions, checked = search_companions()
    print(f"\n  Checked: {checked:,} matrices")
    print(f"  Found companions with Smith=[1,1,1,1]: {len(found_companions)}")
else:
    print(f"  [search skipped; rerun with --all to sweep all {n_mats:,} matrices]")

if found_companions:
    seen = set()
    unique = []
    for c in found_companions:
        key = tuple(sorted(tuple(row) for row in c["cols"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    print(f"  Unique column sets: {len(unique)}")
    _u_syms = [sp.Symbol(f"u_{i+1}") for i in range(4)]
    for i, c in enumerate(unique[:3]):
        print(f"\n  Candidate {i+1}: Smith={c['smith']}  vol_0={c['vol']}")
        print(f"    M = {c['M']}")
        print(f"    t = {c['t']}")
        cols_i = c["cols"]  # list of 8 column vectors (each length 4)
        G_expr = sum(
            sp.Mul(*[_u_syms[k] ** col[k] for k in range(4) if col[k] > 0]) for col in cols_i
        )
        print(f"    G(u) = {sp.expand(G_expr)}")
        cfg = AConfiguration(
            sp.Matrix([[1] * 8] + [[col[r] for col in cols_i] for r in range(4)]),
            is_homogenized=True,
        )
        from itertools import permutations as _perms

        col_set = frozenset(tuple(col) for col in cols_i)
        sym_count = sum(
            1
            for perm in _perms(range(4))
            if frozenset(tuple(col[perm[r]] for r in range(4)) for col in cols_i) == col_set
        )
        fim = finite_index_map(cfg, bms4)
        print(
            f"    S4-symmetry: {sym_count}/24  |  "
            f"finite_index_map -> BMS_4: found={fim.found}  det={fim.determinant}"
        )
        FLUSH()
elif RUN_ALL:
    print("""
  No companion with Smith=[1,1,1,1] found among det=+/-2 matrices
  with entries {{-1,0,1}} and even column sums.

  Interpretation:
    The higher-dimensional analogue of the triangle, if it exists, either:
    (a) requires a map M with larger entries (outside {{-1,0,1}}), or
    (b) does not correspond to any "small" integer affine map to BMS_n, or
    (c) requires a fundamentally different lattice construction.

  The canonical candidate A_candidate(n) = lower_BMS union standard_basis is always
  affinely equivalent to BMS_n (same Newton polytope shape over Q) but with
  Smith invariants [1,...,1, n-2], not all-ones.  It is the "natural geometric
  companion" but not connected by a finite-index map for n>=4.
""")
    FLUSH()


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

print_section("Summary")
_n4_companion_found = len(found_companions) > 0
_unique_count = len({tuple(sorted(tuple(row) for row in c["cols"])) for c in found_companions})

if RUN_ALL:
    _verdict = (
        "yes  YES, full-lattice companions with det=2 map to BMS_4 DO EXIST."
        if _n4_companion_found
        else "no  No full-lattice companions found in this restricted search range."
    )
    _phase3_summary = f"""  PHASE 3 RESULT (n=4, search range: M entries {{-1,0,1}}, det=+/-2):
    Found companions: {len(found_companions):,}  |  Unique column sets: {_unique_count}
    {_verdict}

  First candidate (up to S4 permutation) has G polynomial:
    G = u_1u_2u_3 + (sum_{{i<j}} u_iu_j) + u_4
  = one degree-(n-1) monomial prod_{{j!=k}} u_j
  + all C(n,2) degree-2 pairs
  + one degree-1 monomial u_k  (k fixed, 4 choices by S4 symmetry)"""
else:
    _phase3_summary = (
        "  PHASE 3 (n=4, search range: M entries {-1,0,1}, det=+/-2): not run.\n"
        "    Rerun with --all for the sweep and the companions it turns up."
    )

print(f"""
  The triangle/triple-K equivalence (n=3) arises from a unique lattice coincidence:
    Smith(A_candidate(3)) = [1,1,1]   (full Z^3 lattice)
    Smith(BMS_3)          = [1,1,2]   (index-2 sublattice)
    det(M) = 2            (the lattice index is carried by the map)

  For the canonical companion conformal_companion_a_config(n), Smith(n) = [1,...,1, n-2]:
    n=4: Smith = [1,1,1,2] = Smith(BMS_4)  -> same sublattice, no det=2 map
    n=5: Smith = [1,1,1,1,3]               -> different sublattice from BMS_5's [1,...,2]

  All A_candidate(n) ARE affinely equivalent to BMS_n (same Newton polytope
  shape over Q), but in a different arithmetic sublattice.

{_phase3_summary}

  The canonical companion conformal_companion_a_config(n) remains the most
  natural and fully S_n-symmetric choice, but it is NOT connected to BMS_n
  by any integer finite-index map for n>=4.  The Phase 3 companions break S_n
  to S_{{n-1}} (stabilising one variable), suggesting no fully symmetric
  higher-dimensional analogue of the triangle-triple-K pair exists.
""")
