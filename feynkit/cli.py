"""
feynkit command-line interface.

Usage
-----
    fk <cnickel>                        # full analysis of one diagram
    fk <cnickel> --gkz --newton         # only GKZ and Newton polytope
    fk <cnickel1> <cnickel2>            # equivalence analysis of two diagrams

Section flags (single diagram only; omit all to run everything)
---------------------------------------------------------------
    -s / --symanzik     Symanzik polynomials U, F, G
    -p / --params       Integral parametrisations (Schwinger, Feynman, Lee-Pom.)
    -g / --gkz          GKZ A-matrix and Euler equations
    -t / --toric        Toric ideal (IBP generators)
    -n / --newton       Newton polytope (vertices, volume, Smith invariants)
    -S / --symmetries   Polytope automorphisms and symmetry pairs

Examples
--------
    fk "12e|2e|e|:zzz"
    fk "12e|2e|e|:zzz" --gkz --newton
    fk 12e|2e|e|                         # bare topology (massless assumed)
    fk "12e|2e|e|:zzz" "11e|e|:zz"      # triangle vs bubble
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rule(char: str = "─", width: int = 68) -> None:
    print(char * width)


def _header(title: str) -> None:
    _rule("═")
    print(f"  {title}")
    _rule("═")


def _sec(title: str) -> None:
    print()
    _rule()
    print(f"  {title}")
    _rule()


def _kv(key: str, val: object, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:<28} {val}")


def _matrix(M: sp.Matrix, indent: int = 4) -> None:
    pad = " " * indent
    rows = M.tolist()
    widths = [max(len(str(rows[r][c])) for r in range(M.rows)) for c in range(M.cols)]
    for row in rows:
        cells = [str(v).rjust(w) for v, w in zip(row, widths)]
        print(pad + "[ " + "  ".join(cells) + " ]")


def _fmt_euler(eq: sp.Eq) -> str:
    """Render an Euler equation as 'z_j ∂_j + ... = β' (operator form, no Φ)."""
    lhs_parts = []
    for term in eq.lhs.as_ordered_terms():
        derivs = [a for a in sp.preorder_traversal(term) if isinstance(a, sp.Derivative)]
        if derivs:
            var = derivs[0].variables[0]
            idx = str(var).split("_", 1)[1]
            lhs_parts.append(f"z_{idx} ∂_{idx}")

    phi_atoms = [
        a for a in sp.preorder_traversal(eq.rhs) if isinstance(a, sp.core.function.AppliedUndef)
    ]
    beta = sp.expand(eq.rhs / phi_atoms[0]) if phi_atoms else eq.rhs

    return " + ".join(lhs_parts) + "  =  " + str(beta)


# ─────────────────────────────────────────────────────────────────────────────
# Single-diagram analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyse_one(cnickel: str, db_path: Path, sections: set[str]) -> None:
    from feynkit import FeynkitDatabase, FeynmanIntegral
    from feynkit.a_configuration import AConfiguration

    show_all = len(sections) == 0

    def _show(name: str) -> bool:
        return show_all or name in sections

    t0 = time.time()
    fi = FeynmanIntegral.from_cnickel(cnickel)
    db = FeynkitDatabase(db_path)

    # ── graph (always shown) ─────────────────────────────────────────────────
    _header(f"Feynman integral  {fi.cnickel}")
    _kv("Nickel index", fi.nickel_index)
    _kv("Loop count", fi.loop_count)
    _kv("Propagators", len(fi.graph.get_internal_edges()))
    _kv("External legs", fi.graph.external_legs)

    # ── Symanzik polynomials ─────────────────────────────────────────────────
    if _show("symanzik"):
        _sec("Symanzik polynomials")
        sym = fi.symanzik
        _kv("Schwinger params", fi.schwinger.parameters)
        print(f"  U  =  {sym.u}")
        print(f"  F  =  {sym.f}")
        print(f"  G  =  U + F  =  {sym.g}")

    # ── Parametrisations ─────────────────────────────────────────────────────
    if _show("params"):
        _sec("Integral parametrisations")
        sch = fi.schwinger
        feyn = fi.feynman
        lp = fi.lee_pomeransky
        print(f"  Schwinger   params  {sch.parameters}")
        print(f"              prefactor  {sch.prefactor}")
        print(f"              measure    {sch.measure}")
        print()
        print(f"  Feynman     params  {feyn.parameters}")
        print(f"              prefactor  {feyn.prefactor}")
        print(f"              constraint {feyn.constraints}")
        print()
        print(f"  Lee–Pom.    params  {lp.parameters}")
        print(f"              prefactor  {lp.prefactor}")

    # ── GKZ system ───────────────────────────────────────────────────────────
    if _show("gkz"):
        _sec("GKZ hypergeometric system")
        gkz = fi.gkz
        A = gkz.a_matrix
        print(f"  A-matrix  ({A.rows} × {A.cols})" f"  [rows = coordinates; cols = monomials of G]")
        _matrix(A)
        print()
        _kv("β-parameters", gkz.beta_parameters)
        _kv("z-variables", gkz.z_variables)
        print()
        print("  Euler equations  (Σ_j A_rj z_j ∂_j = β_r):")
        for i, eq in enumerate(gkz.euler_equations):
            print(f"    [{i}]  {_fmt_euler(eq)}")

    # ── Toric ideal ──────────────────────────────────────────────────────────
    if _show("toric"):
        _sec("Toric ideal  (IBP relations in z-space)")
        ti = fi.toric_ideal
        gens = ti.generators
        if gens:
            print(f"  {len(gens)} generator(s)  [z^u - z^v = 0  ↔  A·u = A·v]:")
            for g in gens:
                print(f"    {g}  =  0")
        else:
            print("  Trivial  (no IBP relations — single master integral)")

    # ── Newton polytope ──────────────────────────────────────────────────────
    if _show("newton"):
        _sec("Newton polytope")
        cfg = AConfiguration(fi.gkz.a_matrix, is_homogenized=True)
        _kv("Monomials (A-columns)", cfg.n_points)
        _kv("Hull vertices", len(cfg.newton_polytope_points))
        _kv("Ambient dimension", cfg.ambient_dim)
        _kv("Affine dimension", cfg.affine_dim)
        _kv("Normalised volume (holonomic rank)", cfg.normalized_volume)
        _kv("Smith invariants", cfg.smith_invariants)
        model = cfg.intrinsic_model()
        _kv("Lattice base point", model.base_point)

    # ── Automorphisms / symmetry pairs ───────────────────────────────────────
    if _show("symmetries"):
        _sec("Symmetries")
        aut = fi.polytope_automorphisms
        _kv("|Aut(P)|  (polytope automorphisms)", aut.order)
        _kv("|Aut(graph)|", len(fi.graph_automorphisms))
        _kv("Vertex orbits under Aut(P)", aut.vertex_orbits)
        sym_pairs = fi.symmetry_pairs
        uni_pairs = [p for p in sym_pairs if p.is_unimodular]
        fi_pairs = [p for p in sym_pairs if not p.is_unimodular]
        _kv("Symmetry pairs total", len(sym_pairs))
        _kv("  unimodular (|det|=1)", len(uni_pairs))
        _kv("  finite-index (|det|>1)", len(fi_pairs))
        if fi_pairs:
            for p in fi_pairs[:3]:
                print(f"    det={p.determinant}  perm={p.column_permutation}")

    # ── Database (always) ────────────────────────────────────────────────────
    _sec("Database")
    rec = db.store(fi, label=fi.cnickel)
    print(f"  Stored: {rec}")
    existing = db.find_equivalent(fi, relation="unimodular")
    if existing:
        print(f"  Unimodular equivalents in DB ({len(existing)}):")
        for r in existing:
            lbl = r.label or r.cnickel or r.fingerprint[:8]
            print(f"    {lbl}  (A={r.n_rows}×{r.n_cols})")
    else:
        print("  No unimodular equivalents in DB.")

    db.close()
    _rule("═")
    print(f"  Done in {time.time()-t0:.1f}s")
    _rule("═")


# ─────────────────────────────────────────────────────────────────────────────
# Two-diagram equivalence analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyse_pair(cn1: str, cn2: str, db_path: Path) -> None:
    from feynkit import FeynkitDatabase, FeynmanIntegral
    from feynkit.a_configuration import AConfiguration, FiniteIndexResult, finite_index_map

    t0 = time.time()
    fi1 = FeynmanIntegral.from_cnickel(cn1)
    fi2 = FeynmanIntegral.from_cnickel(cn2)
    db = FeynkitDatabase(db_path)

    _header("Equivalence analysis")
    print(f"  A  =  {fi1.cnickel}")
    print(f"  B  =  {fi2.cnickel}")

    # ── per-diagram summaries ─────────────────────────────────────────────────
    def _brief(fi: FeynmanIntegral, tag: str) -> AConfiguration:
        gkz = fi.gkz
        A = gkz.a_matrix
        cfg = AConfiguration(A, is_homogenized=True)
        ti = fi.toric_ideal
        _sec(f"Diagram {tag}  —  {fi.cnickel}")
        _kv("Nickel", fi.nickel_index)
        _kv(
            "Loops / props / ext",
            f"{fi.loop_count} / {len(fi.graph.get_internal_edges())} / {fi.graph.external_legs}",
        )
        print(f"  U  =  {fi.symanzik.u}")
        print(f"  F  =  {fi.symanzik.f}")
        print(f"  G  =  {fi.symanzik.g}")
        print()
        print(f"  A-matrix  ({A.rows} × {A.cols}):")
        _matrix(A)
        print()
        _kv("β-parameters", gkz.beta_parameters)
        _kv("Toric generators", len(ti.generators))
        _kv(
            "Monomials / verts / dim",
            f"{cfg.n_points} / {len(cfg.newton_polytope_points)} / {cfg.ambient_dim}",
        )
        _kv("Normalised volume", cfg.normalized_volume)
        _kv("Smith invariants", cfg.smith_invariants)
        db.store(fi, label=fi.cnickel)
        return cfg

    cfg1 = _brief(fi1, "A")
    cfg2 = _brief(fi2, "B")

    # ── quick pre-filter ──────────────────────────────────────────────────────
    _sec("Equivalence checks")
    if cfg1.ambient_dim != cfg2.ambient_dim:
        print(f"  Ambient dimension mismatch ({cfg1.ambient_dim} vs {cfg2.ambient_dim}).")
        print("  No affine equivalence is possible between spaces of different dimension.")
        db.close()
        _rule("═")
        print(f"  Done in {time.time()-t0:.1f}s")
        _rule("═")
        return

    results = []
    for relation, method in [
        ("unimodular", cfg1.is_unimodular_equivalent_to),
        ("affine_polytope", cfg1.is_affinely_equivalent_to),
        ("point_config", cfg1.is_point_config_equivalent_to),
    ]:
        res = method(cfg2)
        status = "YES" if res.equivalent else "no"
        det_str = f"  (det = {res.determinant})" if res.determinant is not None else ""
        print(f"  {relation:<22} {status}{det_str}")
        results.append((relation, res))

    # finite-index (only when point counts match)
    if cfg1.n_points == cfg2.n_points:
        fi_res: FiniteIndexResult = finite_index_map(cfg1, cfg2)
        status = "YES" if fi_res.found else "no"
        det_str = f"  (det = {fi_res.determinant})" if fi_res.found else ""
        print(f"  {'finite_index':<22} {status}{det_str}")
    else:
        fi_res = FiniteIndexResult(found=False)
        print(
            f"  {'finite_index':<22} n/a  (different monomial counts: {cfg1.n_points} vs {cfg2.n_points})"
        )

    # ── change of variables for any found map ────────────────────────────────
    any_found = any(r.equivalent for _, r in results) or fi_res.found
    if not any_found:
        print()
        print("  No equivalence found between A and B.")
        db.close()
        _rule("═")
        print(f"  Done in {time.time()-t0:.1f}s")
        _rule("═")
        return

    for relation, res in results:
        if not res.equivalent or res.witness_map is None:
            continue
        _sec(f"Change of variables  [{relation}]")
        M = res.witness_map
        t = res.translation
        n = M.rows
        print(f"  Linear map M  ({M.rows} × {M.cols}):")
        _matrix(M)
        if t is not None:
            t_list = [t[i, 0] for i in range(n)] if t.cols == 1 else [t[0, i] for i in range(n)]
            print(f"  Translation t  =  {t_list}")
        print()
        src_vars = [sp.Symbol(f"u_{i+1}") for i in range(n)]
        tgt_vars = [sp.Symbol(f"v_{i+1}") for i in range(n)]
        print("  u_i (diagram A)  =  Σ_j M_ij v_j + t_i  (diagram B variables v_j):")
        for i, ui in enumerate(src_vars):
            expr = sum(M[i, j] * tgt_vars[j] for j in range(n))
            if t is not None:
                ti_val = t[i, 0] if t.cols == 1 else t[0, i]
                if ti_val != 0:
                    expr = expr + ti_val
            print(f"    {ui}  =  {sp.simplify(expr)}")
        print()
        beta1 = list(fi1.gkz.beta_parameters)
        t_vals = (
            [t[i, 0] for i in range(n)]
            if t is not None and t.cols == 1
            else [t[0, i] for i in range(n)] if t is not None else [sp.Integer(0)] * n
        )
        T_rows = [[sp.Integer(1)] + [sp.Integer(0)] * n]
        for i in range(n):
            T_rows.append([t_vals[i]] + [M[i, j] for j in range(n)])
        T_hom = sp.Matrix(T_rows)
        transformed = list(T_hom * sp.Matrix(beta1))
        print("  GKZ identity  I_A(β, z) = I_A(T·β, z_P):")
        print(f"    β    =  {beta1}")
        print(f"    T·β  =  {[str(x) for x in transformed]}")

    if fi_res.found and fi_res.witness_matrix is not None:
        M, t = fi_res.witness_matrix, fi_res.translation
        n = M.rows
        _sec(f"Change of variables  [finite_index, det = {fi_res.determinant}]")
        print(f"  Linear map M  ({M.rows} × {M.cols}):")
        _matrix(M)
        if t is not None:
            t_list = [t[i, 0] for i in range(n)] if t.cols == 1 else [t[0, i] for i in range(n)]
            print(f"  Translation t  =  {t_list}")
        print()
        src_vars = [sp.Symbol(f"u_{i+1}") for i in range(n)]
        tgt_vars = [sp.Symbol(f"v_{i+1}") for i in range(n)]
        print("  u_i (A)  =  Σ_j M_ij v_j + t_i  (B variables v_j):")
        for i, ui in enumerate(src_vars):
            expr = sum(M[i, j] * tgt_vars[j] for j in range(n))
            if t is not None:
                ti_val = t[i, 0] if t.cols == 1 else t[0, i]
                if ti_val != 0:
                    expr = expr + ti_val
            print(f"    {ui}  =  {sp.simplify(expr)}")

    db.close()
    _rule("═")
    print(f"  Done in {time.time()-t0:.1f}s")
    _rule("═")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fk",
        description="feynkit: analyse Feynman integrals from cNickel strings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
section flags (single diagram; omit all to run every section):
  -s / --symanzik     Symanzik polynomials U, F, G
  -p / --params       Parametrisations (Schwinger, Feynman, Lee-Pom.)
  -g / --gkz          GKZ A-matrix and Euler equations
  -t / --toric        Toric ideal (IBP generators in z-space)
  -n / --newton       Newton polytope (vertices, volume, Smith invariants)
  -S / --symmetries   Polytope automorphisms and symmetry pairs

examples:
  fk "12e|2e|e|:zzz"                      # full analysis
  fk "12e|2e|e|:zzz" -g -n                # GKZ and Newton polytope only
  fk "e1234|e234|e34|e4|e|" -n -S         # large diagram: polytope + symmetries
  fk 12e|2e|e|                             # bare topology (massless assumed)
  fk "12e|2e|e|:zzz" "11e|e|:zz"          # equivalence of two diagrams
        """,
    )
    parser.add_argument(
        "diagrams",
        nargs="+",
        metavar="CNICKEL",
        help="one or two cNickel strings",
    )
    parser.add_argument(
        "--db",
        default="feynkit.db",
        metavar="PATH",
        help="SQLite database path (default: feynkit.db)",
    )

    sec = parser.add_argument_group(
        "sections",
        "which sections to show for a single diagram " "(omit all flags to show every section)",
    )
    sec.add_argument("-s", "--symanzik", action="store_true", help="Symanzik polynomials")
    sec.add_argument("-p", "--params", action="store_true", help="Integral parametrisations")
    sec.add_argument("-g", "--gkz", action="store_true", help="GKZ A-matrix + Euler equations")
    sec.add_argument("-t", "--toric", action="store_true", help="Toric ideal (IBP generators)")
    sec.add_argument("-n", "--newton", action="store_true", help="Newton polytope")
    sec.add_argument(
        "-S", "--symmetries", action="store_true", help="Polytope automorphisms + symmetry pairs"
    )

    args = parser.parse_args(argv)

    if len(args.diagrams) > 2:
        parser.error("At most two diagrams are supported.")

    db_path = Path(args.db)

    if len(args.diagrams) == 1:
        sections = {
            name
            for name in ("symanzik", "params", "gkz", "toric", "newton", "symmetries")
            if getattr(args, name)
        }
        analyse_one(args.diagrams[0], db_path, sections)
    else:
        analyse_pair(args.diagrams[0], args.diagrams[1], db_path)


if __name__ == "__main__":
    main()
