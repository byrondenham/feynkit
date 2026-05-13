"""
Experiment: BMS n-point conformal simplex vs. n-point contact diagram.

For n = 3, 4, 5 we compare:

  Object 1 — BMS_n simplex:   bms_simplex_a_config(n)
  Object 2 — contact diagram: A = [[1,...,1, 1,...,1], [-I_n | I_n]]

For each object we print the Newton polytope data, then run two equivalence checks:

  1. Polytope equivalence  — hull vertices only.
     First tries finite-index integer map (finite_index_map on vertex sets),
     then falls back to rational-affine brute force (is_affinely_equivalent).

  2. Point-config equivalence — all A-columns (GKZ condition).
     Uses finite_index_map on all affine points.

Summary table at the end.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, "/home/byron/docs/ph32048/feynkit")

from feynkit.a_configuration import AConfiguration, FiniteIndexResult, finite_index_map
from feynkit.artifacts.conformal import bms_simplex_a_config
from feynkit.normal_forms._invariants import hull_vertex_indices
from feynkit.normal_forms.affine_equivalence import is_affinely_equivalent


# ──────────────────────────────────────────────────────────────────────────────
# Contact diagram factory
# ──────────────────────────────────────────────────────────────────────────────

def contact_a_config(n: int) -> AConfiguration:
    """
    A-matrix for the n-point contact diagram:

        A = [[1, 1, ..., 1, 1, 1, ..., 1],   ← 2n ones (homogenisation row)
             [-I_n | I_n]]                    ← n × 2n

    Columns are -e_1,...,-e_n, e_1,...,e_n in ℝ^n.
    """
    top = [1] * (2 * n)
    bottom_neg = (-np.eye(n, dtype=int)).tolist()
    bottom_pos = np.eye(n, dtype=int).tolist()
    rows = [top] + [bottom_neg[i] + bottom_pos[i] for i in range(n)]
    A = sp.Matrix(rows)
    return AConfiguration(A, is_homogenized=True)


# ──────────────────────────────────────────────────────────────────────────────
# Polytope invariants helper
# ──────────────────────────────────────────────────────────────────────────────

def _normalized_volume(pts: np.ndarray) -> int | str:
    """Normalised lattice volume of the convex hull of pts (rows = points)."""
    from scipy.spatial import ConvexHull, QhullError

    n = pts.shape[1]
    arr = pts.astype(float)
    aff_dim = int(np.linalg.matrix_rank((arr[1:] - arr[0]).astype(float)))
    if aff_dim == 0:
        return 1
    if aff_dim < n:
        centroid = arr.mean(axis=0)
        deltas = arr - centroid
        _, _, vt = np.linalg.svd(deltas, full_matrices=False)
        coords = deltas @ vt[:aff_dim, :].T
    else:
        coords = arr
    try:
        hull = ConvexHull(coords)
        eucl = float(hull.volume)
    except QhullError:
        return "QhullError"
    return int(round(math.factorial(aff_dim) * eucl))


def describe(cfg: AConfiguration, label: str) -> dict:
    """Compute and print the Newton polytope data for a configuration."""
    pts = cfg.affine_points  # n_monomials × n_dim
    n_monomials = cfg.n_points
    amb_dim = cfg.ambient_dim

    v_idx = hull_vertex_indices(pts)
    verts = pts[v_idx]
    n_verts = len(v_idx)
    n_interior = n_monomials - n_verts
    vol = _normalized_volume(verts)
    smith = cfg.smith_invariants

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Monomials (columns of A) : {n_monomials}")
    print(f"  Ambient dim              : {amb_dim}")
    print(f"  Newton polytope vertices : {n_verts}")
    print(f"  Interior lattice points  : {n_interior}")
    print(f"  Normalised volume        : {vol}")
    print(f"  Smith invariants of A    : {smith}")
    print(f"  Vertex matrix (rows):")
    for row in verts.tolist():
        print(f"    {row}")

    return {
        "n_monomials": n_monomials,
        "amb_dim": amb_dim,
        "n_verts": n_verts,
        "n_interior": n_interior,
        "vol": vol,
        "smith": smith,
        "verts": verts,
        "pts": pts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Equivalence search
# ──────────────────────────────────────────────────────────────────────────────

def _classify(det: int | None) -> str:
    if det is None:
        return "—"
    if abs(det) == 1:
        return "unimodular"
    if isinstance(det, int) or (isinstance(det, sp.Expr) and det.is_Integer):
        return "integer-affine"
    return "rational-affine"


def polytope_equivalence(
    d1: dict, d2: dict, label: str
) -> dict:
    """
    Check equivalence at the polytope level (hull vertices only).

    1. Try finite-index integer map (handles unimodular and integer-affine).
    2. If not found, try rational-affine brute force.
    """
    V1 = d1["verts"]  # (n_verts × n_dim), rows = points
    V2 = d2["verts"]
    # Pass numpy arrays directly — finite_index_map treats them as (n_pts × n_dim)
    cfg1 = V1
    cfg2 = V2

    # Quick invariant checks.
    if d1["n_verts"] != d2["n_verts"]:
        print(f"\n  [Polytope] {label}: FAIL — vertex counts differ "
              f"({d1['n_verts']} vs {d2['n_verts']})")
        return {"level": "polytope", "reason": "vertex_count_mismatch", "equivalent": False}
    if d1["amb_dim"] != d2["amb_dim"]:
        print(f"\n  [Polytope] {label}: FAIL — ambient dimension mismatch")
        return {"level": "polytope", "reason": "dim_mismatch", "equivalent": False}
    if d1["vol"] != d2["vol"]:
        print(f"\n  [Polytope] {label}: FAIL — normalised volume differs "
              f"({d1['vol']} vs {d2['vol']})")
        return {"level": "polytope", "reason": "volume_mismatch", "equivalent": False}

    # Integer affine map on vertices.
    fi = finite_index_map(cfg1, cfg2)
    if fi.found:
        M = fi.witness_matrix
        t = fi.translation
        det = fi.determinant
        cls = _classify(det)
        print(f"\n  [Polytope] {label}: EQUIVALENT ({cls})")
        print(f"    det M = {det}")
        print(f"    M =\n{sp.pretty(M)}")
        print(f"    t = {t.T.tolist()[0]}")
        return {"level": "polytope", "equivalent": True, "classification": cls,
                "M": M, "t": t, "det": det}

    # Rational-affine brute force.
    res = is_affinely_equivalent(V1.tolist(), V2.tolist())
    if res.equivalent:
        M = res.witness_map
        t = res.translation
        det = res.determinant
        cls = "rational-affine"
        print(f"\n  [Polytope] {label}: EQUIVALENT ({cls})")
        print(f"    det M = {det}")
        if M is not None:
            print(f"    M =\n{sp.pretty(M)}")
        if t is not None:
            print(f"    t = {t.T.tolist()}")
        return {"level": "polytope", "equivalent": True, "classification": cls,
                "M": M, "t": t, "det": det}

    print(f"\n  [Polytope] {label}: NO EQUIVALENCE FOUND "
          f"(invariants matched: vol={d1['vol']}, verts={d1['n_verts']})")
    return {"level": "polytope", "equivalent": False, "reason": "no_map_found"}


def point_config_equivalence(d1: dict, d2: dict, label: str) -> dict:
    """Check equivalence at the full A-column level (GKZ condition)."""
    cfg1 = d1["pts"]  # (n_pts × n_dim) numpy arrays passed directly
    cfg2 = d2["pts"]

    if d1["n_monomials"] != d2["n_monomials"]:
        print(f"  [Point-config] {label}: FAIL — monomial counts differ")
        return {"level": "point_config", "reason": "count_mismatch", "equivalent": False}
    if d1["amb_dim"] != d2["amb_dim"]:
        print(f"  [Point-config] {label}: FAIL — ambient dimension mismatch")
        return {"level": "point_config", "reason": "dim_mismatch", "equivalent": False}

    fi = finite_index_map(cfg1, cfg2)
    if fi.found:
        det = fi.determinant
        cls = _classify(det)
        print(f"  [Point-config] {label}: EQUIVALENT ({cls}), det M = {det}")
        return {"level": "point_config", "equivalent": True, "classification": cls,
                "det": det, "M": fi.witness_matrix, "t": fi.translation}

    print(f"  [Point-config] {label}: NO EQUIVALENCE (no integer affine map on full point sets)")
    return {"level": "point_config", "equivalent": False}


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    summary_rows = []

    for n in [3, 4, 5]:
        print(f"\n{'═'*70}")
        print(f"  n = {n}")
        print(f"{'═'*70}")

        bms = bms_simplex_a_config(n)
        cdiag = contact_a_config(n)

        d_bms = describe(bms, f"BMS_{n} simplex (bms_simplex_a_config)")
        d_cd = describe(cdiag, f"Contact diagram (A=[1|1; -I|I])")

        polytope_res = polytope_equivalence(d_bms, d_cd, f"BMS_{n} vs Contact_{n}")
        point_res = point_config_equivalence(d_bms, d_cd, f"BMS_{n} vs Contact_{n}")

        summary_rows.append({
            "n": n,
            "bms_verts": d_bms["n_verts"],
            "cd_verts": d_cd["n_verts"],
            "bms_vol": d_bms["vol"],
            "cd_vol": d_cd["vol"],
            "bms_smith": d_bms["smith"],
            "cd_smith": d_cd["smith"],
            "poly_equiv": polytope_res["equivalent"],
            "poly_class": polytope_res.get("classification", polytope_res.get("reason", "—")),
            "poly_det": polytope_res.get("det", "—"),
            "pc_equiv": point_res["equivalent"],
            "pc_class": point_res.get("classification", point_res.get("reason", "—")),
            "pc_det": point_res.get("det", "—"),
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  SUMMARY TABLE")
    print(f"{'═'*70}")
    header = (f"{'n':>2}  {'V_bms':>5}  {'V_cd':>4}  "
              f"{'Vol_bms':>7}  {'Vol_cd':>6}  "
              f"{'Smith_bms':<14}  {'Smith_cd':<12}  "
              f"{'Poly-equiv?':>11}  {'Class':>15}  {'det M':>8}  "
              f"{'PC-equiv?':>9}  {'PC-class':>15}")
    print(header)
    print("─" * len(header))
    for r in summary_rows:
        print(
            f"{r['n']:>2}  {r['bms_verts']:>5}  {r['cd_verts']:>4}  "
            f"{str(r['bms_vol']):>7}  {str(r['cd_vol']):>6}  "
            f"{str(r['bms_smith']):<14}  {str(r['cd_smith']):<12}  "
            f"{str(r['poly_equiv']):>11}  {str(r['poly_class']):>15}  "
            f"{str(r['poly_det']):>8}  "
            f"{str(r['pc_equiv']):>9}  {str(r['pc_class']):>15}"
        )

    print(f"\n{'═'*70}")
    print("Key:")
    print("  V_*   = number of Newton polytope vertices")
    print("  Vol_* = normalised lattice volume of Newton polytope")
    print("  Smith = Smith normal form diagonal of A")
    print("  Poly-equiv = affine equivalence on hull vertices only")
    print("  PC-equiv   = affine equivalence on all A-columns (GKZ condition)")
    print("  Class: unimodular (|det|=1), integer-affine (|det|>1, integer),")
    print("         rational-affine (rational det), or failure reason")
    print()


if __name__ == "__main__":
    main()
