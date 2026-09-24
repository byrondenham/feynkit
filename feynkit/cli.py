"""
feynkit command-line interface.

Usage
-----
    fk analyse CNICKEL [options]    one diagram, section by section
    fk compare A B [options]        two diagrams: equivalence checks and GKZ identities
    fk CNICKEL [options]            the bare form of fk analyse
    fk A B [options]                the bare form of fk compare
    fk --version

Run fk analyse --help or fk compare --help for the options.

Examples
--------
    fk analyse "12e|2e|e|:zzz"
    fk analyse "12e|2e|e|:zzz" --gkz --newton
    fk analyse "12e|2e|e|"                     # bare topology: every propagator massless
    fk compare "12e|2e|e|:zzz" "11e|e|:zz"     # triangle against bubble

Quote every CNickel string: an unquoted | is a shell pipe.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import sympy as sp

from feynkit.core.constants import __version__

COMMANDS = ("analyse", "compare")
SECTION_FLAGS = ("symanzik", "params", "gkz", "toric", "newton", "symmetries")

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def _rule(char: str = "-", width: int = 68) -> None:
    print(char * width)


def _header(title: str) -> None:
    _rule("=")
    print(f"  {title}")
    _rule("=")


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
        cells = [str(v).rjust(w) for v, w in zip(row, widths, strict=True)]
        print(pad + "[ " + "  ".join(cells) + " ]")


def _fmt_euler(eq: sp.Eq) -> str:
    """Render an Euler equation as 'z_j d_j + ... = beta' (operator form, no Phi)."""
    lhs_parts = []
    for term in eq.lhs.as_ordered_terms():
        derivs = [a for a in sp.preorder_traversal(term) if isinstance(a, sp.Derivative)]
        if derivs:
            var = derivs[0].variables[0]
            idx = str(var).split("_", 1)[1]
            lhs_parts.append(f"z_{idx} d_{idx}")

    phi_atoms = [
        a for a in sp.preorder_traversal(eq.rhs) if isinstance(a, sp.core.function.AppliedUndef)
    ]
    beta = sp.expand(eq.rhs / phi_atoms[0]) if phi_atoms else eq.rhs

    return " + ".join(lhs_parts) + "  =  " + str(beta)


# -----------------------------------------------------------------------------
# Maps between two configurations
# -----------------------------------------------------------------------------


def _column_permutation(
    src_pts: Sequence[Sequence[int]],
    tgt_pts: Sequence[Sequence[int]],
    M: sp.Matrix,
    t: sp.Matrix,
) -> list[int] | None:
    """P, counted from 0, with M a_j + t = b_P(j) for every column; None if P is no bijection."""
    free: dict[tuple[sp.Expr, ...], list[int]] = {}
    for k, b in enumerate(tgt_pts):
        free.setdefault(tuple(sp.Integer(int(x)) for x in b), []).append(k)
    perm = []
    for a in src_pts:
        slots = free.get(tuple(M * sp.Matrix([int(x) for x in a]) + t))
        if not slots:
            return None
        perm.append(slots.pop(0))
    return perm if len(perm) == len(tgt_pts) else None


def _gkz_identity(
    src_pts: Sequence[Sequence[int]],
    tgt_pts: Sequence[Sequence[int]],
    M: sp.Matrix,
    t: sp.Matrix | None,
    beta: Sequence[sp.Expr],
    perm: Sequence[int] | None = None,
) -> tuple[list[int], sp.Expr, list[sp.Expr]] | None:
    """
    P, |det M| and T beta for I_A(beta, z_P) = |det M| I_B(T beta, z).

    Here M a_j + t = b_P(j) for every column a_j of A, P counts from 0 and
    T = [[1, 0], [t, M]]; the substitution u_i = prod_k v_k^(M_ki) proves it.
    Returns None when M is singular or the map is not a bijection between the
    columns, since then no identity follows.  ``perm`` is used as P when given.
    """
    M = sp.Matrix(M)
    n = M.rows
    t_col = sp.zeros(n, 1) if t is None else sp.Matrix(t).reshape(n, 1)
    det = M.det()
    if det == 0:
        return None
    if perm is None:
        P = _column_permutation(src_pts, tgt_pts, M, t_col)
    else:
        P = list(perm) if sorted(perm) == list(range(len(tgt_pts))) else None
    if P is None:
        return None
    T = sp.Matrix.vstack(sp.Matrix([[1] + [0] * n]), t_col.row_join(M))
    return P, abs(det), list(T * sp.Matrix(list(beta)))


def _substitution(M: sp.Matrix) -> list[str]:
    """The substitution u_i = prod_k v_k^(M_ki), one string per u_i."""
    lines = []
    for i in range(M.cols):
        factors = []
        for k in range(M.rows):
            e = M[k, i]
            if e == 0:
                continue
            v = f"v_{k + 1}"
            if e == 1:
                factors.append(v)
            elif e.is_Integer and e > 0:
                factors.append(f"{v}^{e}")
            else:
                factors.append(f"{v}^({e})")
        lines.append(f"u_{i + 1}  =  {' '.join(factors)}")
    return lines


# -----------------------------------------------------------------------------
# Single-diagram analysis
# -----------------------------------------------------------------------------


def analyse_one(cnickel: str, db_path: Path, sections: set[str]) -> None:
    from feynkit import FeynkitDatabase, FeynmanIntegral
    from feynkit.a_configuration import AConfiguration

    show_all = len(sections) == 0

    def _show(name: str) -> bool:
        return show_all or name in sections

    t0 = time.time()
    fi = FeynmanIntegral.from_cnickel(cnickel)
    db = FeynkitDatabase(db_path)

    # -- graph (always shown) -------------------------------------------------
    _header(f"Feynman integral  {fi.cnickel}")
    _kv("Nickel index", fi.nickel_index)
    _kv("Loop count", fi.loop_count)
    _kv("Propagators", len(fi.graph.get_internal_edges()))
    _kv("External legs", fi.graph.external_legs)

    # -- Symanzik polynomials -------------------------------------------------
    if _show("symanzik"):
        _sec("Symanzik polynomials")
        sym = fi.symanzik
        _kv("Schwinger params", fi.schwinger.parameters)
        print(f"  U  =  {sym.u}")
        print(f"  F  =  {sym.f}")
        print(f"  G  =  U + F  =  {sym.g}")

    # -- Parametrisations -----------------------------------------------------
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
        print(f"  Lee-Pom.    params  {lp.parameters}")
        print(f"              prefactor  {lp.prefactor}")

    # -- GKZ system -----------------------------------------------------------
    if _show("gkz"):
        _sec("GKZ hypergeometric system")
        gkz = fi.gkz
        A = gkz.a_matrix
        print(f"  A-matrix  ({A.rows} x {A.cols})" f"  [rows = coordinates; cols = monomials of G]")
        _matrix(A)
        print()
        _kv("beta-parameters", gkz.beta_parameters)
        _kv("z-variables", gkz.z_variables)
        print()
        print("  Euler equations  (sum_j A_rj z_j d_j = beta_r):")
        for i, eq in enumerate(gkz.euler_equations):
            print(f"    [{i}]  {_fmt_euler(eq)}")

    # -- Toric ideal ----------------------------------------------------------
    if _show("toric"):
        _sec("Toric ideal  (generators: an analogue of IBP relations)")
        ti = fi.toric_ideal
        gens = ti.generators
        if gens:
            print(f"  {len(gens)} generator(s)  [z^u - z^v = 0  <->  A*u = A*v]:")
            for g in gens:
                print(f"    {g}  =  0")
        else:
            print("  Trivial  (the zero ideal)")

    # -- Newton polytope ------------------------------------------------------
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

    # -- Automorphisms / symmetry pairs ---------------------------------------
    if _show("symmetries"):
        _sec("Symmetries")
        aut = fi.polytope_automorphisms
        _kv("|Aut(P)|  (polytope automorphisms)", aut.order)
        _kv("|Aut(graph)|", len(fi.graph_automorphisms))
        _kv("Vertex orbits under Aut(P)", aut.vertex_orbits)
        _kv("Symmetry pairs  (|det M|=1)", len(fi.symmetry_pairs))

    # -- Database (always) ----------------------------------------------------
    _sec("Database")
    rec = db.store(fi, label=fi.cnickel)
    print(f"  Stored: {rec}")
    existing = db.find_equivalent(fi, relation="unimodular")
    if existing:
        print(f"  Unimodular equivalents in DB ({len(existing)}):")
        for r in existing:
            lbl = r.label or r.cnickel or r.fingerprint[:8]
            print(f"    {lbl}  (A={r.n_rows} x {r.n_cols})")
    else:
        print("  No unimodular equivalents in DB.")

    db.close()
    _rule("=")
    print(f"  Done in {time.time()-t0:.1f}s")
    _rule("=")


# -----------------------------------------------------------------------------
# Two-diagram equivalence analysis
# -----------------------------------------------------------------------------


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

    # -- per-diagram summaries -------------------------------------------------
    def _brief(fi: FeynmanIntegral, tag: str) -> AConfiguration:
        gkz = fi.gkz
        A = gkz.a_matrix
        cfg = AConfiguration(A, is_homogenized=True)
        ti = fi.toric_ideal
        _sec(f"Diagram {tag} ,  {fi.cnickel}")
        _kv("Nickel", fi.nickel_index)
        _kv(
            "Loops / props / ext",
            f"{fi.loop_count} / {len(fi.graph.get_internal_edges())} / {fi.graph.external_legs}",
        )
        print(f"  U  =  {fi.symanzik.u}")
        print(f"  F  =  {fi.symanzik.f}")
        print(f"  G  =  {fi.symanzik.g}")
        print()
        print(f"  A-matrix  ({A.rows} x {A.cols}):")
        _matrix(A)
        print()
        _kv("beta-parameters", gkz.beta_parameters)
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

    # -- quick pre-filter ------------------------------------------------------
    _sec("Equivalence checks")
    if cfg1.ambient_dim != cfg2.ambient_dim:
        print(f"  Ambient dimension mismatch ({cfg1.ambient_dim} vs {cfg2.ambient_dim}).")
        print("  No affine equivalence is possible between spaces of different dimension.")
        db.close()
        _rule("=")
        print(f"  Done in {time.time()-t0:.1f}s")
        _rule("=")
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

    # -- witness maps and the identities they give ----------------------------
    any_found = any(r.equivalent for _, r in results) or fi_res.found
    if not any_found:
        print()
        print("  No equivalence found between A and B.")
        db.close()
        _rule("=")
        print(f"  Done in {time.time()-t0:.1f}s")
        _rule("=")
        return

    # A map of the hull vertices implies no GKZ identity unless every column is
    # a vertex, and then point_config holds too and prints the identity itself.
    pts1 = cfg1.affine_points.tolist()
    pts2 = cfg2.affine_points.tolist()
    beta1 = list(fi1.gkz.beta_parameters)
    z2 = list(fi2.gkz.z_variables)
    all_vertices = all(len(c.newton_polytope_points) == c.n_points for c in (cfg1, cfg2))

    def _witness(relation: str, M: sp.Matrix, t: sp.Matrix | None) -> None:
        _sec(f"Witness map  [{relation}]")
        print(f"  Linear map M  ({M.rows} x {M.cols}):")
        _matrix(M)
        if t is not None:
            print(f"  Translation t  =  {list(t)}")
        print()

    def _identity(M: sp.Matrix, t: sp.Matrix | None, perm: list[int] | None = None) -> None:
        identity = _gkz_identity(pts1, pts2, M, t, beta1, perm)
        if identity is None:
            print("  M is singular or not a bijection on the columns: no GKZ identity follows.")
            return
        P, factor, t_beta = identity
        print("  M a_j + t = b_P(j) for each column a_j of A (b_k: columns of B, from 1):")
        print(f"    P       =  {[k + 1 for k in P]}")
        print("  Substitution u_i = prod_k v_k^(M_ki), u of diagram A, v of diagram B:")
        for line in _substitution(M):
            print(f"    {line}")
        print("  GKZ identity I_A(beta, z_P) = |det M| I_B(T beta, z), no Gamma prefactors:")
        print(f"    |det M| =  {factor}")
        print(f"    z_P     =  ({', '.join(str(z2[k]) for k in P)})")
        print(f"    beta    =  {beta1}")
        print(f"    T beta  =  {t_beta}")

    for relation, res in results:
        if not res.equivalent or res.witness_map is None:
            continue
        _witness(relation, res.witness_map, res.translation)
        if relation == "point_config":
            _identity(res.witness_map, res.translation)
        elif all_vertices:
            print("  Relates the Newton polytopes only; every column is a vertex, so")
            print("  point_config gives the identity.")
        else:
            print("  Relates the Newton polytopes only; not every column is a vertex, so")
            print("  no GKZ identity follows.")

    if fi_res.found and fi_res.witness_matrix is not None:
        _witness("finite_index", fi_res.witness_matrix, fi_res.translation)
        _identity(fi_res.witness_matrix, fi_res.translation, fi_res.column_permutation)

    db.close()
    _rule("=")
    print(f"  Done in {time.time()-t0:.1f}s")
    _rule("=")


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

# Options that take a value. The bare form needs them to tell a value from a
# second diagram: fk "12e|2e|e|" --db x.db names one diagram, not two.
_VALUE_OPTIONS = frozenset({"--db"})

_MAIN_EPILOG = """\
examples:
  fk analyse "12e|2e|e|:zzz"                   every section of the massless triangle
  fk analyse "12e|2e|e|:zzz" -g -n             GKZ system and Newton polytope only
  fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"   the mass on two different propagators
  fk "12e|2e|e|:zzz"                           the bare form of fk analyse

Run fk analyse --help or fk compare --help for their options.
Quote every CNickel string: an unquoted | is a shell pipe.
"""

_ANALYSE_EPILOG = """\
examples:
  fk analyse "12e|2e|e|:zzz"                   every section
  fk analyse "12e|2e|e|:zzz" -g -n             GKZ system and Newton polytope only
  fk analyse "12e|3e|3e|e|:zzzz" -n -S         the massless box: polytope and symmetries
  fk analyse "12e|2e|e|"                       bare topology: every propagator massless

Quote every CNickel string: an unquoted | is a shell pipe.
"""

_COMPARE_EPILOG = """\
examples:
  fk compare "12e|2e|e|:zzz" "11e|e|:zz"       triangle against bubble
  fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"   the mass on two different propagators

Quote every CNickel string: an unquoted | is a shell pipe.
"""


class _Parsers(NamedTuple):
    """The fk parser and the parsers of its two subcommands."""

    main: argparse.ArgumentParser
    analyse: argparse.ArgumentParser
    compare: argparse.ArgumentParser


def _build_parser() -> _Parsers:
    """The fk parser; the options both subcommands take come from one parent."""
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument(
        "--db",
        default="feynkit.db",
        metavar="PATH",
        help="SQLite database for the results (default: feynkit.db in the working directory)",
    )

    parser = argparse.ArgumentParser(
        prog="fk",
        description="Analyse Feynman integrals given as CNickel strings.",
        epilog=_MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"fk {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    analyse = commands.add_parser(
        "analyse",
        parents=[common],
        help="analyse one diagram",
        description="Analyse one diagram: every section, or those the section flags choose.",
        epilog=_ANALYSE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    analyse.add_argument(
        "cnickel", metavar="CNICKEL", help='CNickel string, quoted, e.g. "12e|2e|e|:nzz"'
    )
    shown = analyse.add_argument_group(
        "sections", "sections to print; all of them when no flag is given"
    )
    shown.add_argument("-s", "--symanzik", action="store_true", help="Symanzik polynomials U, F, G")
    shown.add_argument(
        "-p",
        "--params",
        action="store_true",
        help="Schwinger, Feynman and Lee-Pomeransky parametrisations",
    )
    shown.add_argument("-g", "--gkz", action="store_true", help="GKZ A-matrix and Euler equations")
    shown.add_argument("-t", "--toric", action="store_true", help="toric ideal of the A-matrix")
    shown.add_argument(
        "-n",
        "--newton",
        action="store_true",
        help="Newton polytope: vertices, normalised volume, Smith invariants",
    )
    shown.add_argument(
        "-S", "--symmetries", action="store_true", help="polytope automorphisms and symmetry pairs"
    )

    compare = commands.add_parser(
        "compare",
        parents=[common],
        help="compare two diagrams",
        description=(
            "Compare two diagrams: their GKZ data, four equivalence checks and the "
            "identity that each map of every column gives."
        ),
        epilog=_COMPARE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    compare.add_argument("first", metavar="A", help="first CNickel string, quoted")
    compare.add_argument("second", metavar="B", help="second CNickel string, quoted")
    return _Parsers(parser, analyse, compare)


def _positionals(argv: Sequence[str]) -> list[str]:
    """The positional arguments of argv, skipping the value of each option that takes one."""
    found: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token == "--":
            found.extend(tokens)
        elif token.startswith("-"):
            if token in _VALUE_OPTIONS:
                next(tokens, None)
        else:
            found.append(token)
    return found


def _with_command(argv: Sequence[str]) -> list[str]:
    """argv with the subcommand that the bare form fk CNICKEL or fk A B stands for.

    argparse cannot give one slot to either a subcommand or a positional, so
    the bare form is rewritten before parsing. When the first positional is not
    a command name, two positionals mean compare and any other number means
    analyse, which then reports surplus arguments itself. argv without
    positionals, such as --help or --version, is left alone.
    """
    args = list(argv)
    positionals = _positionals(args)
    if not positionals or positionals[0] in COMMANDS:
        return args
    return ["compare" if len(positionals) == 2 else "analyse", *args]


def _section_flags(args: argparse.Namespace) -> set[str]:
    """The sections that the section flags of analyse choose."""
    return {name for name in SECTION_FLAGS if getattr(args, name)}


def main(argv: Sequence[str] | None = None) -> None:
    """Run fk; argparse exits with status 2 on a usage error."""
    parsers = _build_parser()
    args = parsers.main.parse_args(_with_command(sys.argv[1:] if argv is None else argv))
    db_path = Path(args.db)
    if args.command == "analyse":
        analyse_one(args.cnickel, db_path, _section_flags(args))
    else:
        analyse_pair(args.first, args.second, db_path)


if __name__ == "__main__":
    main()
