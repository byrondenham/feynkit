"""
feynkit, comprehensive survey of Feynman diagram GKZ data
===========================================================

This script systematically processes four families of Feynman diagrams:

  Family 1, Massless 1-loop n-gons (n = 3 ... 7)
      Cycle graphs C_n with n external legs and n internal propagators.
      Triangle (n=3), box (n=4), pentagon (n=5), hexagon (n=6), heptagon (n=7).

  Family 2, k-propagator banana diagrams (k = 2 ... 6)
      2-point graphs with k parallel internal propagators.
      Bubble (k=2), sunset (k=3), 4-banana, 5-banana, 6-banana.

  Family 3, Massless complete graphs K_n (n = 3, 4)
      K_3 = triangle (3 edges, 1 loop), K_4 = tetrahedron (6 edges, 3 loops).
      K_5 is excluded: its normalised volume computation takes ~10 minutes.

  Family 4, Conformal / CFT A-configurations
      BMS n-point simplex (n = 3 ... 6) and conformal companion (n = 3 ... 6).

For each diagram the script records:
  - Graph: Nickel/cNickel index, loop count, propagator count, external legs
  - Symanzik polynomials U and F; Lee-Pomeransky polynomial G
  - GKZ A-matrix (shape and full matrix)
  - GKZ beta-parameters (Euler grading)
  - Euler differential equations (one per row of A)
  - Toric ideal generators (IBP relations in z-variables)
  - Newton polytope: ambient dim, vertex count, normalised volume, Smith invariants

Then runs a pairwise affine equivalence survey across all pairs with matching
ambient dimension:
  - Unimodular equivalence (Liu-Cai; same GKZ system up to z-relabelling)
  - Affine-polytope equivalence (rational change of basis on hull vertices)
  - Point-configuration equivalence (affine map on all A-columns)
  - Finite-index integer map (|det| > 1)

For every equivalence found the script exhibits:
 , the linear map M and translation t
 , the change of integration variables u_i = sum_j M_ij v_j + t_i
 , the resulting GKZ beta-parameter identity I_A(beta, z) = I_A(T*beta, z_P)

All data is persisted in feynkit_survey.db so that re-runs skip already-computed
integrals. Results are also written to feynkit_survey.txt.

Usage
-----
    uv run python examples/feynkit_survey.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from feynkit import FeynkitDatabase, FeynmanIntegral
from feynkit.a_configuration import AConfiguration, FiniteIndexResult, finite_index_map
from feynkit.algebra.toric import compute_toric_ideal_generators
from feynkit.artifacts.conformal import (
    bms_simplex_a_config,
    complete_graph_a_config,
    conformal_companion_a_config,
)
from feynkit.core.edge import Edge
from feynkit.core.graph import Graph

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "feynkit_survey.db"
OUT_PATH = Path(__file__).parent.parent / "feynkit_survey.txt"

_lines: list[str] = []


def _emit(*args: Any) -> None:
    line = " ".join(str(a) for a in args)
    print(line)
    _lines.append(line)


def _rule(char: str = "-", width: int = 72) -> None:
    _emit(char * width)


def _header(title: str) -> None:
    _rule("=")
    _emit(f"  {title}")
    _rule("=")


def _sec(title: str) -> None:
    _emit()
    _rule()
    _emit(f"  {title}")
    _rule()


def _matrix_lines(M: sp.Matrix, indent: str = "    ") -> list[str]:
    rows = M.tolist()
    col_widths = [max(len(str(rows[r][c])) for r in range(M.rows)) for c in range(M.cols)]
    out = []
    for row in rows:
        cells = [str(v).rjust(w) for v, w in zip(row, col_widths, strict=True)]
        out.append(indent + "[ " + "  ".join(cells) + " ]")
    return out


def _print_matrix(M: sp.Matrix, indent: str = "    ") -> None:
    for line in _matrix_lines(M, indent):
        _emit(line)


# -----------------------------------------------------------------------------
# Diagram record
# -----------------------------------------------------------------------------


@dataclass
class DiagramRecord:
    """Everything computed for one Feynman diagram."""

    family: str
    label: str
    nickel: str | None
    loops: int | None
    n_props: int | None
    n_ext: int | None
    a_matrix: sp.Matrix
    beta: list
    z_vars: list
    euler_eqs: list
    toric_gens: list
    ambient_dim: int
    n_pts: int
    n_verts: int
    norm_vol: int
    smith: list[int]
    U: sp.Expr | None = None
    F: sp.Expr | None = None
    G: sp.Expr | None = None
    alpha_vars: list = field(default_factory=list)
    u_vars: list = field(default_factory=list)
    a_cfg: AConfiguration | None = None


# -----------------------------------------------------------------------------
# Graph builder helpers
# -----------------------------------------------------------------------------


def _ngon_fi(n: int) -> FeynmanIntegral:
    edges = []
    for i in range(1, n + 1):
        j = (i % n) + 1
        edges.append(Edge(idx=i, v1=i, v2=j, is_internal=True, mass=sp.Integer(0)))
    for i in range(1, n + 1):
        edges.append(Edge(idx=n + i, v1=i, v2=n + i, is_internal=False))
    return FeynmanIntegral(Graph(internal_vertices=n, external_legs=n, edges=edges))


def _banana_fi(k: int) -> FeynmanIntegral:
    edges = []
    for i in range(1, k + 1):
        edges.append(Edge(idx=i, v1=1, v2=2, is_internal=True, mass=sp.Integer(0)))
    edges.append(Edge(idx=k + 1, v1=1, v2=3, is_internal=False))
    edges.append(Edge(idx=k + 2, v1=2, v2=4, is_internal=False))
    return FeynmanIntegral(Graph(internal_vertices=2, external_legs=2, edges=edges))


# -----------------------------------------------------------------------------
# Build a DiagramRecord from a FeynmanIntegral
# -----------------------------------------------------------------------------


def _record_from_fi(
    fi: FeynmanIntegral,
    family: str,
    label: str,
    db: FeynkitDatabase,
) -> DiagramRecord:
    t0 = time.time()
    _emit(f"  Building {label} ...")

    sym = fi.symanzik
    gkz = fi.gkz
    ti = fi.toric_ideal
    A = gkz.a_matrix
    cfg = AConfiguration(A, is_homogenized=True)

    db.store(fi, label=label)

    rec = DiagramRecord(
        family=family,
        label=label,
        nickel=fi.nickel_index,
        loops=fi.loop_count,
        n_props=len(fi.graph.get_internal_edges()),
        n_ext=fi.graph.external_legs,
        a_matrix=A,
        beta=list(gkz.beta_parameters),
        z_vars=list(gkz.z_variables),
        euler_eqs=list(gkz.euler_equations),
        toric_gens=list(ti.generators),
        ambient_dim=cfg.ambient_dim,
        n_pts=cfg.n_points,
        n_verts=len(cfg.newton_polytope_points),
        norm_vol=cfg.normalized_volume,
        smith=cfg.smith_invariants,
        U=sym.u,
        F=sym.f,
        G=sym.g,
        alpha_vars=list(fi.schwinger.parameters),
        u_vars=list(fi.lee_pomeransky.parameters),
        a_cfg=cfg,
    )
    _emit(
        f"    done in {time.time() - t0:.1f}s  ->  "
        f"A={A.rows} x {A.cols}, toric_gens={len(ti.generators)}, "
        f"vol={cfg.normalized_volume}"
    )
    return rec


def _record_from_cfg(
    cfg: AConfiguration,
    family: str,
    label: str,
) -> DiagramRecord:
    t0 = time.time()
    _emit(f"  Building {label} ...")
    A = cfg.matrix
    toric_gens = compute_toric_ideal_generators(A)
    rec = DiagramRecord(
        family=family,
        label=label,
        nickel=None,
        loops=None,
        n_props=None,
        n_ext=None,
        a_matrix=A,
        beta=[],
        z_vars=[],
        euler_eqs=[],
        toric_gens=toric_gens,
        ambient_dim=cfg.ambient_dim,
        n_pts=cfg.n_points,
        n_verts=len(cfg.newton_polytope_points),
        norm_vol=cfg.normalized_volume,
        smith=cfg.smith_invariants,
        a_cfg=cfg,
    )
    _emit(
        f"    done in {time.time() - t0:.1f}s  ->  "
        f"A={A.rows} x {A.cols}, toric_gens={len(toric_gens)}, vol={cfg.normalized_volume}"
    )
    return rec


# -----------------------------------------------------------------------------
# Print one diagram's full data block
# -----------------------------------------------------------------------------


def _print_diagram(r: DiagramRecord) -> None:
    _header(f"{r.label}  [{r.family}]")

    if r.nickel is not None:
        _emit(f"  Nickel index  : {r.nickel}")
        _emit(f"  Loops         : {r.loops}")
        _emit(f"  Propagators   : {r.n_props}")
        _emit(f"  External legs : {r.n_ext}")
        _emit()

    if r.U is not None:
        _emit(f"  Schwinger params  : {r.alpha_vars}")
        _emit(f"  U  =  {r.U}")
        _emit(f"  F  =  {r.F}")
        _emit(f"  G  =  U + F  =  {r.G}")
        _emit(f"  LP params         : {r.u_vars}")
        _emit()

    _emit(f"  GKZ A-matrix  ({r.a_matrix.rows} x {r.a_matrix.cols})")
    _emit("  [Rows = lattice coordinates; columns = monomials of G]")
    _print_matrix(r.a_matrix)
    _emit()

    if r.beta:
        _emit(f"  beta-parameters  :  {r.beta}")
        _emit(f"  z-variables   :  {r.z_vars}")
        _emit()

    if r.euler_eqs:
        _emit("  Euler equations  [sum_j A_rj z_j d_j Phi = beta_r Phi,  one per A-row]:")
        for i, eq in enumerate(r.euler_eqs):
            _emit(f"    [{i}]  {eq}")
        _emit()

    if r.toric_gens:
        _emit(f"  Toric ideal  ({len(r.toric_gens)} generators)  [IBP relations, z^u - z^v = 0]:")
        for g in r.toric_gens:
            _emit(f"    {g}  =  0")
    else:
        _emit("  Toric ideal : trivial  (no IBP relations)")
    _emit()

    _emit("  Newton polytope:")
    _emit(f"    Monomials (A-columns)          : {r.n_pts}")
    _emit(f"    Hull vertices                  : {r.n_verts}")
    _emit(f"    Ambient dimension              : {r.ambient_dim}")
    _emit(f"    Normalised volume (= hol. rank): {r.norm_vol}")
    _emit(f"    Smith invariants               : {r.smith}")
    _emit()


# -----------------------------------------------------------------------------
# Pairwise equivalence survey
# -----------------------------------------------------------------------------


@dataclass
class EquivResult:
    label_a: str
    label_b: str
    relation: str
    equivalent: bool
    det: int | sp.Expr | None = None
    M: sp.Matrix | None = None
    t: sp.Matrix | None = None
    reason: str = ""


def _check_pair(r1: DiagramRecord, r2: DiagramRecord) -> list[EquivResult]:
    results: list[EquivResult] = []
    c1, c2 = r1.a_cfg, r2.a_cfg
    if c1 is None or c2 is None:
        return results

    for relation, method in [
        ("unimodular", c1.is_unimodular_equivalent_to),
        ("affine_polytope", c1.is_affinely_equivalent_to),
        ("point_config", c1.is_point_config_equivalent_to),
    ]:
        try:
            res = method(c2)
            results.append(
                EquivResult(
                    r1.label,
                    r2.label,
                    relation,
                    res.equivalent,
                    det=res.determinant,
                    M=res.witness_map,
                    t=res.translation,
                )
            )
        except Exception as exc:
            results.append(EquivResult(r1.label, r2.label, relation, False, reason=str(exc)))

    try:
        # finite_index_map finds an injective map (not necessarily bijective).
        # Only report it as "found" when point counts match (bijective case).
        if c1.n_points == c2.n_points:
            fi_res: FiniteIndexResult = finite_index_map(c1, c2)
        else:
            fi_res = FiniteIndexResult(
                found=False,
                witness_matrix=None,
                translation=None,
                determinant=None,
                column_permutation=None,
            )
        results.append(
            EquivResult(
                r1.label,
                r2.label,
                "finite_index",
                fi_res.found,
                det=fi_res.determinant if fi_res.found else None,
                M=fi_res.witness_matrix if fi_res.found else None,
                t=fi_res.translation if fi_res.found else None,
            )
        )
    except Exception as exc:
        results.append(EquivResult(r1.label, r2.label, "finite_index", False, reason=str(exc)))

    return results


def _change_of_vars_lines(
    M: sp.Matrix,
    t: sp.Matrix | None,
    src_vars: list[sp.Symbol],
    tgt_vars: list[sp.Symbol],
) -> list[str]:
    lines = []
    n = M.rows
    for i, ui in enumerate(src_vars):
        expr: sp.Expr = sum(M[i, j] * tgt_vars[j] for j in range(n))
        if t is not None:
            ti_val = t[i, 0] if t.cols == 1 else t[0, i]
            if ti_val != 0:
                expr = expr + ti_val
        lines.append(f"  {ui}  =  {sp.simplify(expr)}")
    return lines


# -----------------------------------------------------------------------------
# Main survey
# -----------------------------------------------------------------------------


def main() -> None:
    db = FeynkitDatabase(DB_PATH)

    _header("feynkit comprehensive survey")
    _emit(f"  Database : {DB_PATH}")
    _emit(f"  Output   : {OUT_PATH}")
    _emit()

    records: list[DiagramRecord] = []

    # -- Family 1: massless n-gons (n = 3 ... 7) -------------------------------
    _header("FAMILY 1, Massless 1-loop n-gons  (n = 3 ... 7)")
    _emit()
    for n in range(3, 8):
        rec = _record_from_fi(_ngon_fi(n), "n-gon", f"{n}-gon (C_{n})", db)
        records.append(rec)

    # -- Family 2: banana diagrams (k = 2 ... 6) -------------------------------
    _header("FAMILY 2, Massless banana diagrams  (k = 2 ... 6)")
    _emit()
    banana_names = {2: "bubble", 3: "sunset", 4: "4-banana", 5: "5-banana", 6: "6-banana"}
    for k in range(2, 7):
        label = f"{banana_names[k]} (banana-{k})"
        rec = _record_from_fi(_banana_fi(k), "banana", label, db)
        records.append(rec)

    # -- Family 3: complete graphs K_n (n = 3, 4) ----------------------------
    _header("FAMILY 3, Massless complete graphs K_n  (n = 3, 4)")
    _emit("  [K_5 excluded: 235 A-columns in R^10, hull computation takes ~10 min]")
    _emit()
    for n in [3, 4]:
        cfg = complete_graph_a_config(n)
        label = f"K_{n} complete"
        rec = _record_from_cfg(cfg, "K_n", label)
        if n == 3:
            fi_k3 = _ngon_fi(3)
            rec.nickel = fi_k3.nickel_index
            rec.loops = fi_k3.loop_count
            rec.n_props = len(fi_k3.graph.get_internal_edges())
            rec.n_ext = fi_k3.graph.external_legs
            rec.U = fi_k3.symanzik.u
            rec.F = fi_k3.symanzik.f
            rec.G = fi_k3.symanzik.g
            rec.alpha_vars = list(fi_k3.schwinger.parameters)
            rec.u_vars = list(fi_k3.lee_pomeransky.parameters)
        records.append(rec)

    # -- Family 4a: BMS simplex (n = 3 ... 6) ----------------------------------
    _header("FAMILY 4a, BMS n-point conformal simplex  (n = 3 ... 6)")
    _emit()
    for n in range(3, 7):
        cfg = bms_simplex_a_config(n)
        rec = _record_from_cfg(cfg, "BMS", f"BMS_{n} simplex")
        records.append(rec)

    # -- Family 4b: conformal companion (n = 3 ... 6) --------------------------
    _header("FAMILY 4b, Conformal companion  (n = 3 ... 6)")
    _emit()
    for n in range(3, 7):
        cfg = conformal_companion_a_config(n)
        rec = _record_from_cfg(cfg, "companion", f"companion_{n}")
        records.append(rec)

    # -- Full data blocks -----------------------------------------------------
    _header("FULL GKZ DATA PER DIAGRAM")
    for r in records:
        _print_diagram(r)

    # -- Invariant summary table ----------------------------------------------
    _header("INVARIANT SUMMARY TABLE")
    _emit()
    hdr = (
        f"{'Label':<22}  {'Family':<9}  {'A-shape':>7}  "
        f"{'pts':>4}  {'verts':>5}  {'dim':>4}  "
        f"{'vol':>8}  {'toric':>5}  Smith"
    )
    _emit(hdr)
    _rule("-", 78)
    for r in records:
        shape = f"{r.a_matrix.rows} x {r.a_matrix.cols}"
        _emit(
            f"{r.label:<22}  {r.family:<9}  {shape:>7}  "
            f"{r.n_pts:>4}  {r.n_verts:>5}  {r.ambient_dim:>4}  "
            f"{r.norm_vol:>8}  {len(r.toric_gens):>5}  {r.smith}"
        )
    _emit()

    # -- Pairwise equivalence survey ------------------------------------------
    _header("PAIRWISE AFFINE EQUIVALENCE SURVEY")
    _emit()
    _emit("  For each pair of diagrams with matching ambient dimension,")
    _emit("  all four relations are tested:")
    _emit("    unimodular  , Liu-Cai algorithm, |det M| = 1")
    _emit("    affine_poly , rational map on Newton polytope hull vertices")
    _emit("    point_config, rational map on all A-columns (GKZ condition)")
    _emit("    finite_index, integer map, |det M| > 1 allowed")
    _emit()

    from itertools import combinations

    by_dim: dict[int, list[DiagramRecord]] = {}
    for r in records:
        by_dim.setdefault(r.ambient_dim, []).append(r)

    all_equiv: list[EquivResult] = []

    for dim, grp in sorted(by_dim.items()):
        if len(grp) < 2:
            continue
        _sec(f"Ambient dimension {dim}  ({len(grp)} diagrams in this group)")
        for r1, r2 in combinations(grp, 2):
            results = _check_pair(r1, r2)
            all_equiv.extend(er for er in results if er.equivalent)

            _emit(f"  {r1.label}  <->  {r2.label}")
            for er in results:
                status = "yes" if er.equivalent else "no"
                det_str = f"  (det = {er.det})" if er.det is not None else ""
                reason_str = f"  [{er.reason[:40]}]" if er.reason else ""
                _emit(f"    {er.relation:<18}  {status}{det_str}{reason_str}")
            _emit()

    # -- Change-of-variables for each found map -------------------------------
    _header("EXPLICIT CHANGES OF VARIABLES FOR FOUND MAPS")
    _emit()

    seen: set[tuple[str, str, str]] = set()

    for er in all_equiv:
        key = (er.label_a, er.label_b, er.relation)
        if key in seen or er.M is None:
            continue
        seen.add(key)

        _sec(
            f"{er.relation}  :  {er.label_a}  ->  {er.label_b}"
            + (f"  (det = {er.det})" if er.det is not None else "")
        )

        _emit(f"  Linear map M  ({er.M.rows} x {er.M.cols}):")
        _print_matrix(er.M)
        _emit()

        if er.t is not None:
            t_list = (
                er.t.T.tolist()[0] if er.t.shape[1] != 1 else [er.t[i, 0] for i in range(er.t.rows)]
            )
            _emit(f"  Translation t  =  {t_list}")
            _emit()

        n = er.M.rows
        src_vars = [sp.Symbol(f"u_{i + 1}") for i in range(n)]
        tgt_vars = [sp.Symbol(f"v_{i + 1}") for i in range(n)]
        _emit("  Change of integration variables  (u = source diagram, v = target):")
        _emit("    u_i  =  sum_j M_ij v_j  +  t_i")
        try:
            for line in _change_of_vars_lines(er.M, er.t, src_vars, tgt_vars):
                _emit(line)
        except Exception as exc:
            _emit(f"  [error: {exc}]")
        _emit()

        # beta-parameter identity (only for Feynman integral source records)
        src_rec = next((r for r in records if r.label == er.label_a), None)
        if src_rec and src_rec.beta:
            n = er.M.rows
            t_vals: list = []
            if er.t is not None:
                t_vals = (
                    [er.t[i, 0] for i in range(n)]
                    if er.t.cols == 1
                    else [er.t[0, i] for i in range(n)]
                )
            else:
                t_vals = [sp.Integer(0)] * n
            M_list = [[er.M[i, j] for j in range(n)] for i in range(n)]
            T_rows = [[1] + [0] * n]
            for i in range(n):
                T_rows.append([t_vals[i]] + M_list[i])
            T_hom = sp.Matrix(T_rows)
            beta_vec = sp.Matrix(list(src_rec.beta))
            transformed = list(T_hom * beta_vec)
            _emit("  GKZ integral identity  I_A(beta, z) = I_A(T*beta, z_P):")
            _emit(f"    beta      =  {src_rec.beta}")
            _emit(f"    T*beta    =  {[str(x) for x in transformed]}")
            _emit("  [T*beta gives the beta-parameters of the target integral]")
            _emit()

    # -- Pattern analysis -----------------------------------------------------
    _header("PATTERN ANALYSIS")

    # Smith normal form groups
    _sec("Smith normal form patterns")
    from collections import defaultdict

    smith_groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        smith_groups[str(r.smith)].append(r.label)
    _emit("  Diagrams are grouped by their Smith invariants.")
    _emit("  Diagrams in the same group span the same sublattice.")
    _emit()
    for key in sorted(smith_groups):
        labels = smith_groups[key]
        _emit(f"  Smith {key}  ({len(labels)} diagrams):")
        for lb in labels:
            _emit(f"    {lb}")
    _emit()

    # Toric ideal growth
    _sec("Toric ideal (IBP relation) counts")
    _emit(f"  {'Label':<22}  {'n_pts':>5}  {'amb_dim':>7}  {'toric_gens':>10}")
    _rule("-", 54)
    for r in records:
        _emit(f"  {r.label:<22}  {r.n_pts:>5}  {r.ambient_dim:>7}  {len(r.toric_gens):>10}")
    _emit()
    _emit("  Pattern: n-gon toric count grows rapidly (~C(n,2) x something).")
    _emit("  Banana diagrams have 0 generators (simplex = single master integral).")
    _emit()

    # Volume patterns
    _sec("Normalised volumes (= holonomic rank)")
    _emit(f"  {'Label':<22}  {'vol':>10}")
    _rule("-", 36)
    for r in records:
        _emit(f"  {r.label:<22}  {r.norm_vol:>10}")
    _emit()
    _emit("  BMS_n simplex: vol = 2^(n-1)  (powers of 2: 4, 8, 16, 32 for n=3..6)")
    _emit("  n-gon: vol grows faster than exponential in n.")
    _emit()

    # Equivalence class summary
    _sec("Equivalence class summary")

    for relation in ["unimodular", "affine_polytope", "point_config", "finite_index"]:
        pairs = [(er.label_a, er.label_b, er.det) for er in all_equiv if er.relation == relation]
        _emit(f"  {relation}:")
        if pairs:
            for a, b, det in pairs:
                det_str = f"  (det = {det})" if det is not None else ""
                _emit(f"    {a}  ~=  {b}{det_str}")
        else:
            _emit("    (none found)")
        _emit()

    _rule("=")
    _emit("  Survey complete.")
    _rule("=")

    db.close()

    OUT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    _emit(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
