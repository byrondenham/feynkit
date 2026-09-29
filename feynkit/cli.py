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
    fk analyse "11e|e|:nn" --torus-count       # candidate master count from point counts
    fk analyse "12e|2e|e|"                     # bare topology: every propagator massless
    fk compare "12e|2e|e|:zzz" "11e|e|:zz"     # triangle against bubble

Quote every CNickel string: an unquoted | is a shell pipe.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NoReturn

import sympy as sp

from feynkit.a_configuration import AConfiguration, FiniteIndexResult, finite_index_map
from feynkit.core.constants import __version__
from feynkit.core.exceptions import FeynkitError, ValidationError
from feynkit.core.graph import Graph
from feynkit.database import FeynkitDatabase
from feynkit.integral import FeynmanIntegral
from feynkit.io.report import DEFAULT_SECTIONS, SECTION_NAMES, AnalysisReport
from feynkit.io.report_latex import render_latex
from feynkit.io.report_text import render_text
from feynkit.kinematics.classes import IMPOSABLE_CLASSES, KinematicClass
from feynkit.landau import LandauAnalysis
from feynkit.point_count import TorusCount
from feynkit.polytope import polytope_data

SECTION_FLAGS = ("symanzik", "params", "gkz", "toric", "newton", "symmetries")
# The seed and budget of the point counts when --seed and --torus-budget are not given.
DEFAULT_TORUS_SEED = 0
DEFAULT_TORUS_BUDGET = 2 * 10**9
# 128 + SIGPIPE: the status of a tool that a closed pipe stops, as the shell reports it.
EXIT_BROKEN_PIPE = 141
# The status of fk compare when no check finds an equivalence. diff uses 1 for
# a difference, but fk uses 1 for errors.
EXIT_NOT_EQUIVALENT = 3
# The budget of the point counts as their errors name it, after the keyword of
# count_torus_points; fk calls it --torus-budget.
_MAX_EVALUATIONS = re.compile(r"max_evaluations=(\d+)")

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


def _fmt_euler(row: Sequence[sp.Expr], z_variables: Sequence[sp.Symbol], beta: sp.Expr) -> str:
    """The Euler equation of one row of A, 'A_r1 z_1 d_1 + ...  =  beta_r', without Phi.

    Terms with a zero coefficient are left out, and a coefficient of 1 is not written.
    """
    terms: list[str] = []
    for coefficient, z in zip(row, z_variables, strict=True):
        if coefficient == 0:
            continue
        index = str(z).split("_", 1)[1]
        size = abs(coefficient)
        term = f"z_{index} d_{index}" if size == 1 else f"{size} z_{index} d_{index}"
        if terms:
            terms.append(f"+ {term}" if coefficient > 0 else f"- {term}")
        else:
            terms.append(term if coefficient > 0 else f"-{term}")
    return f"{' '.join(terms) or '0'}  =  {beta}"


def _done(t0: float) -> None:
    """Close the output with the elapsed time, and flush it."""
    _rule("=")
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    _rule("=")
    sys.stdout.flush()


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
# Loading, errors, progress and the database
# -----------------------------------------------------------------------------

CNICKEL_GRAMMAR = (
    "expected TOPOLOGY or TOPOLOGY:COLOURS, where TOPOLOGY has one '|'-terminated entry "
    "per vertex naming the vertices it joins (digits) and its external legs (e), and "
    "COLOURS one mass code per propagator (z massless, n massive), "
    'as in fk analyse "12e|2e|e|:nzz"'
)


class CliError(Exception):
    """A bad input or a failed write, which main reports in one line on stderr."""


def _load(cnickel: str, kinematics: KinematicClass | None = None) -> FeynmanIntegral:
    """The integral of a CNickel string, with a kinematic class imposed if one is given.

    A string that does not parse raises CliError with the grammar. The
    Mandelstam invariants need two external legs, so a graph with fewer, such
    as the massive tadpole 0|:n, gets generic momentum products instead. Any
    other failure to build the integral raises CliError without the grammar,
    and so does a class that FeynmanIntegral.with_kinematics refuses.
    """
    try:
        graph = Graph.from_cnickel(cnickel)
    except ValueError as exc:
        raise CliError(f"cannot parse CNickel {cnickel!r}: {exc}; {CNICKEL_GRAMMAR}") from exc
    try:
        integral = FeynmanIntegral(graph, use_mandelstam=graph.external_legs >= 2)
    except (ValueError, FeynkitError) as exc:
        raise CliError(f"cannot build the integral of CNickel {cnickel!r}: {exc}") from exc
    if kinematics is None:
        return integral
    try:
        return integral.with_kinematics(kinematics)
    except ValidationError as exc:
        raise CliError(f"CNickel {cnickel!r}: {exc}") from exc


def _fail(message: str) -> NoReturn:
    """Print message as one line on stderr and exit with status 1."""
    sys.stdout.flush()
    text = " ".join(message.splitlines())
    print(f"fk: error: {text}", file=sys.stderr)
    raise SystemExit(1)


def _open_database(stack: ExitStack, path: Path | None) -> FeynkitDatabase | None:
    """The database at path, closed when stack unwinds; None when path is None."""
    return None if path is None else stack.enter_context(FeynkitDatabase(path))


@contextmanager
def _stage(name: str, verbose: bool) -> Iterator[None]:
    """Run one stage, then flush stdout and, if verbose, print its time on stderr."""
    start = time.perf_counter()
    yield
    sys.stdout.flush()
    if verbose:
        print(f"fk: {name} {time.perf_counter() - start:.2f}s", file=sys.stderr)


def _label(given: str, fi: FeynmanIntegral) -> str:
    """The CNickel string as given, with the canonical form beside it when they differ."""
    return given if given == fi.cnickel else f"{given}  (canonical form {fi.cnickel})"


# -----------------------------------------------------------------------------
# Single-diagram analysis
# -----------------------------------------------------------------------------


def _print_graph(given: str, fi: FeynmanIntegral) -> None:
    _header(f"Feynman integral  {_label(given, fi)}")
    _kv("Nickel index", fi.nickel_index)
    _kv("Loop count", fi.loop_count)
    _kv("Propagators", len(fi.graph.get_internal_edges()))
    _kv("External legs", fi.graph.external_legs)
    _kv("Kinematic class", fi.kinematic_class)


def _print_symanzik(fi: FeynmanIntegral) -> None:
    _sec("Symanzik polynomials")
    sym = fi.symanzik
    _kv("Schwinger params", sym.schwinger_parameters)
    print(f"  U  =  {sym.u}")
    print(f"  F  =  {sym.f}")
    print(f"  G  =  U + F  =  {sym.g}")


def _print_params(fi: FeynmanIntegral) -> None:
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


def _print_gkz(fi: FeynmanIntegral) -> None:
    _sec("GKZ hypergeometric system")
    gkz = fi.gkz
    A = gkz.a_matrix
    print(f"  A-matrix  ({A.rows} x {A.cols})  [rows = coordinates; cols = monomials of G]")
    _matrix(A)
    print()
    _kv("beta-parameters", gkz.beta_parameters)
    _kv("z-variables", gkz.z_variables)
    print()
    print("  Euler equations  (sum_j A_rj z_j d_j = beta_r):")
    for r, beta in enumerate(gkz.beta_parameters):
        print(f"    [{r}]  {_fmt_euler(list(A.row(r)), gkz.z_variables, beta)}")


def _print_toric(fi: FeynmanIntegral) -> None:
    _sec("Toric ideal  (generators: an analogue of IBP relations)")
    gens = fi.toric_ideal.generators
    if gens:
        print(f"  {len(gens)} generator(s)  [z^u - z^v = 0  <->  A*u = A*v]:")
        for g in gens:
            print(f"    {g}  =  0")
    else:
        print("  Trivial  (the zero ideal)")


def _vertices_and_volume(cfg: AConfiguration) -> tuple[int, int]:
    """The number of vertices and the normalised volume of the Newton polytope of cfg.

    Both come from one polytope_data, as in the report, in every dimension.
    """
    data = polytope_data(cfg.affine_points.tolist())
    return len(data.vertex_indices), data.normalized_volume


def _print_newton(fi: FeynmanIntegral) -> None:
    _sec("Newton polytope")
    cfg = AConfiguration(fi.gkz.a_matrix, is_homogenized=True)
    vertices, volume = _vertices_and_volume(cfg)
    _kv("Monomials (A-columns)", cfg.n_points)
    _kv("Hull vertices", vertices)
    _kv("Ambient dimension", cfg.ambient_dim)
    _kv("Affine dimension", cfg.affine_dim)
    _kv("Scaleless", "yes" if fi.is_scaleless else "no")
    # Below full dimension the rows of A are dependent, so for generic beta the
    # Euler equations contradict each other and the rank is 0, not the volume.
    full = cfg.affine_dim == cfg.ambient_dim
    _kv("Normalised volume", f"{volume}  (the holonomic rank for generic beta)" if full else volume)
    _kv("Smith invariants", cfg.smith_invariants)
    _kv("Lattice base point", cfg.intrinsic_model().base_point)
    _print_lattice_invariants(fi, full)


def _print_lattice_invariants(fi: FeynmanIntegral, full: bool) -> None:
    """The lattice invariants of the Newton polytope and, when NA is normal, the certificate."""
    found = fi.lattice_invariants()
    index = found.gorenstein_index
    gorenstein = "none" if index is None else str(index)
    _kv("Lattice points", f"{found.lattice_points}  ({found.interior_points} interior)")
    _kv("h*-vector", found.h_star)
    _kv("Gorenstein index", f"{gorenstein}  (reflexive)" if index == 1 else gorenstein)
    _kv("Lattice width", found.lattice_width)
    _kv("Integer decomposition (IDP)", "yes" if found.idp else "no")
    if found.normal:
        _kv("Normal configuration", "yes")
        if full:
            print("  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972)")
            print("  and there are no rank jumps (Matusevich, Miller and Walther 2005).")
        else:
            print("  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972).")
        return
    reasons = []
    if not found.idp:
        reasons.append("P is not IDP")
    if not found.support_is_saturated:
        missing = found.lattice_points - len({tuple(p) for p in fi.newton_polytope.points})
        reasons.append(
            f"the support misses {missing} lattice point{'s' if missing != 1 else ''} of P"
        )
    _kv("Normal configuration", f"no  ({'; '.join(reasons)})")


def _print_symmetries(fi: FeynmanIntegral) -> None:
    _sec("Symmetries")
    cfg = AConfiguration(fi.gkz.a_matrix, is_homogenized=True)
    full = cfg.affine_dim == cfg.ambient_dim
    aut = fi.polytope_automorphisms
    kind = "polytope automorphisms" if full else "automorphisms of P in its affine hull"
    _kv("|Aut(P)|", f"{aut.order}  ({kind})")
    _kv("|Aut(graph)|", len(fi.graph_automorphisms))
    _kv("Vertex orbits under Aut(P)", aut.vertex_orbits)
    _kv("Symmetry pairs  (|det M|=1)", len(fi.symmetry_pairs))
    if not full:
        print("  P is not full-dimensional: the identities of the pairs hold only trivially.")


def _print_database(fi: FeynmanIntegral, db: FeynkitDatabase) -> None:
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


def _print_torus(count: TorusCount) -> None:
    _sec("Candidate Euler characteristic from point counts")
    if count.point:
        point = ", ".join(f"{key} = {value}" for key, value in count.point)
        origin = "given" if count.seed is None else f"seed {count.seed}"
        _kv("Kinematic point", f"{point}  ({origin})")
    else:
        _kv("Kinematic point", "none needed: G has no kinematic symbols")
    if count.on_shell:
        _kv("On shell", ", ".join(f"{key} = {value}" for key, value in count.on_shell))
    if count.on_landau_surface:
        print(
            "  A coefficient of G or a face discriminant vanishes at the point, where C can be "
            "smaller than for generic kinematics."
        )
    _kv("Solved for", count.eliminated)
    excluded = ", ".join(map(str, count.excluded_primes))
    _kv("Excluded primes", f"{excluded}  (up to {count.max_prime})")
    fit = set(count.fit_primes)
    _kv("Counts #V(F_p) to fit", ", ".join(f"{p}: {n}" for p, n in count.counts if p in fit))
    checks = ", ".join(f"{p}: {n}" for p, n in count.counts if p not in fit)
    _kv("Counts #V(F_p) to check", checks or "none")
    if count.skipped_faces:
        _kv("Skipped faces", f"{count.skipped_faces}  (by the Landau analysis)")
    if count.candidate_polynomial is None:
        print(f"  No candidate: {count.reason}.")
    else:
        q = sp.Symbol("q")
        _kv("Candidate P(q)", sp.Add(*(c * q**i for i, c in enumerate(count.candidate_polynomial))))
        _kv("Candidate chi(X) = -P(1)", count.candidate_euler_characteristic)
        _kv("Candidate master count", count.candidate_master_count)
    print("  A fit on finitely many primes is evidence, not a proof.")


_PRINTERS: dict[str, Callable[[FeynmanIntegral], None]] = {
    "symanzik": _print_symanzik,
    "params": _print_params,
    "gkz": _print_gkz,
    "toric": _print_toric,
    "newton": _print_newton,
    "symmetries": _print_symmetries,
}


# -----------------------------------------------------------------------------
# Analysis report
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportOptions:
    """Which report sections analyse builds, and where it writes the report."""

    sections: tuple[str, ...] = DEFAULT_SECTIONS
    latex: Path | None = None
    text: Path | None = None
    as_json: bool = False
    torus_seed: int = DEFAULT_TORUS_SEED
    torus_budget: int = DEFAULT_TORUS_BUDGET

    @property
    def writes_files(self) -> bool:
        return self.latex is not None or self.text is not None


def _integer(text: str, least: int) -> int:
    """Parse an integer option of at least ``least``; anything else is a usage error."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer; got {text!r}") from None
    if value < least:
        raise argparse.ArgumentTypeError(f"must be at least {least}; got {value}")
    return value


def _seed(text: str) -> int:
    """Parse --seed, 0 or more: random.Random seeds with the absolute value of an integer, so
    -1 would draw the point of seed 1 and report it as seed -1."""
    return _integer(text, 0)


def _budget(text: str) -> int:
    """Parse --torus-budget, 1 or more evaluations of G."""
    return _integer(text, 1)


def _kinematic_class(text: str) -> KinematicClass:
    """Parse --kinematics: a kinematic class that FeynmanIntegral.with_kinematics imposes."""
    if text not in IMPOSABLE_CLASSES:
        raise argparse.ArgumentTypeError(
            f"unknown kinematic class {text!r}; choose from {', '.join(IMPOSABLE_CLASSES)}"
        )
    return text


def _section_list(text: str) -> tuple[str, ...]:
    """Parse --sections: comma-separated names from SECTION_NAMES, kept in report order."""
    names = {name.strip() for name in text.split(",") if name.strip()}
    choices = ", ".join(SECTION_NAMES)
    if not names:
        raise argparse.ArgumentTypeError(f"no section given; choose from {choices}")
    unknown = sorted(names - set(SECTION_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown section {', '.join(unknown)}; choose from {choices}"
        )
    return tuple(name for name in SECTION_NAMES if name in names)


def _write(path: Path, content: str, kind: str) -> None:
    """Write one report file; a failure raises CliError."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot write the {kind} report to {path}: {exc.strerror or exc}") from exc


def _check_writable(path: Path, kind: str) -> None:
    """Raise CliError, as _write would, when the report file plainly cannot be written.

    It runs before the analysis, which can take minutes. _write still reports
    what it misses, such as a directory removed in the meantime.
    """
    # os.path answers False where stat is denied; Path raises before Python 3.14.
    parent = path.parent
    if os.path.isdir(path):
        code = errno.EISDIR
    elif not os.path.isdir(parent):
        # stat fails as opening the file would: the parent is missing, lies
        # under a file or sits in a directory that cannot be searched.
        try:
            os.stat(parent)
        except OSError as exc:
            code = exc.errno or errno.ENOENT
        else:
            code = errno.ENOTDIR
    elif not (
        os.access(path, os.W_OK) if os.path.exists(path) else os.access(parent, os.W_OK | os.X_OK)
    ):
        code = errno.EACCES
    else:
        return
    raise CliError(f"cannot write the {kind} report to {path}: {os.strerror(code)}")


def _write_reports(report: AnalysisReport, options: ReportOptions, *, announce: bool) -> None:
    """Write the LaTeX and text reports asked for, saying so on stdout when announce is set."""
    if options.latex is not None:
        _write(options.latex, render_latex(report), "LaTeX")
        if announce:
            print(f"  Wrote the LaTeX report to {options.latex}")
    if options.text is not None:
        _write(options.text, render_text(report), "text")
        if announce:
            print(f"  Wrote the text report to {options.text}")


def _json_value(value: str) -> bool | int | str | None:
    """A summary value as an integer when it is one, a boolean for "yes" and "no", None for
    "none", otherwise as given."""
    if value == "none":
        return None
    if value in ("yes", "no"):
        return value == "yes"
    try:
        return int(value)
    except ValueError:
        return value


def _summary_json(
    given: str, fi: FeynmanIntegral, report: AnalysisReport, sections: Sequence[str]
) -> str:
    """The report's summary as JSON, keyed in snake case, with the CNickel strings."""
    summary = {
        label.lower().replace(" ", "_"): _json_value(value) for label, value in report.summary()
    }
    payload = {
        "input": given,
        "cnickel": fi.cnickel,
        "sections": list(sections),
        "summary": summary,
    }
    return json.dumps(payload, indent=2)


def analyse_one(
    cnickel: str,
    db_path: Path | None,
    sections: set[str],
    *,
    report: ReportOptions | None = None,
    verbose: bool = False,
    kinematics: KinematicClass | None = None,
) -> None:
    """Analyse one diagram: every section or the chosen ones, then the reports asked for.

    ``kinematics`` is a kinematic class to impose on the integral of the
    string. With --json, only the report's summary is printed, as JSON. Stdout is
    flushed after each stage; verbose prints each stage's time on stderr. The
    point counts are printed only when ``sections`` holds "torus". They run once
    when the report holds them too, and otherwise take the report's Landau
    analysis when it has one.
    """
    options = report if report is not None else ReportOptions()
    t0 = time.perf_counter()
    with _stage("parse", verbose):
        fi = _load(cnickel, kinematics)
    built: list[AnalysisReport] = []

    def build() -> AnalysisReport:
        if not built:
            built.append(
                AnalysisReport.from_integral(
                    fi,
                    options.sections,
                    torus_seed=options.torus_seed,
                    torus_budget=options.torus_budget,
                )
            )
        return built[0]

    with ExitStack() as stack:
        db = _open_database(stack, db_path)
        if options.as_json:
            with _stage("report", verbose):
                _write_reports(build(), options, announce=False)
            if db is not None:
                with _stage("database", verbose):
                    db.store(fi, label=fi.cnickel)
            print(_summary_json(cnickel, fi, build(), options.sections))
            return
        with _stage("graph", verbose):
            _print_graph(cnickel, fi)
        for name in SECTION_FLAGS:
            if not sections or name in sections:
                with _stage(name, verbose):
                    _PRINTERS[name](fi)
        if "torus" in sections:
            with _stage("torus", verbose):
                count = None
                landau: LandauAnalysis | None = None
                if options.writes_files and "torus" in options.sections:
                    count = build().torus
                elif options.writes_files and "landau" in options.sections:
                    section = build().landau
                    landau = section.analysis if section is not None else None
                if count is None:
                    count = fi.torus_count(
                        seed=options.torus_seed,
                        max_evaluations=options.torus_budget,
                        landau=landau,
                    )
                _print_torus(count)
        if options.writes_files:
            with _stage("report", verbose):
                _sec("Report")
                _write_reports(build(), options, announce=True)
        if db is not None:
            with _stage("database", verbose):
                _print_database(fi, db)
    _done(t0)


# -----------------------------------------------------------------------------
# Two-diagram equivalence analysis
# -----------------------------------------------------------------------------


def analyse_pair(cn1: str, cn2: str, db_path: Path | None, *, verbose: bool = False) -> bool:
    """Compare two diagrams: GKZ data, four equivalence checks and the identities that follow.

    Returns whether any check found an equivalence.
    """
    t0 = time.perf_counter()
    with _stage("parse", verbose):
        fi1, fi2 = _load(cn1), _load(cn2)
    with ExitStack() as stack:
        db = _open_database(stack, db_path)
        found = _compare(cn1, fi1, cn2, fi2, db, verbose=verbose)
    _done(t0)
    return found


def _compare(
    cn1: str,
    fi1: FeynmanIntegral,
    cn2: str,
    fi2: FeynmanIntegral,
    db: FeynkitDatabase | None,
    *,
    verbose: bool,
) -> bool:
    """Print both diagrams, the four checks and each map found; return whether one was."""

    # -- per-diagram summaries -------------------------------------------------
    def _brief(given: str, fi: FeynmanIntegral, tag: str) -> tuple[AConfiguration, int]:
        gkz = fi.gkz
        A = gkz.a_matrix
        cfg = AConfiguration(A, is_homogenized=True)
        ti = fi.toric_ideal
        _sec(f"Diagram {tag}  {_label(given, fi)}")
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
        vertices, volume = _vertices_and_volume(cfg)
        _kv("Monomials / verts / dim", f"{cfg.n_points} / {vertices} / {cfg.ambient_dim}")
        _kv("Normalised volume", volume)
        _kv("Smith invariants", cfg.smith_invariants)
        if db is not None:
            db.store(fi, label=fi.cnickel)
        return cfg, vertices

    with _stage("diagram A", verbose):
        _header("Equivalence analysis")
        print(f"  A  =  {_label(cn1, fi1)}")
        print(f"  B  =  {_label(cn2, fi2)}")
        cfg1, vertices1 = _brief(cn1, fi1, "A")
    with _stage("diagram B", verbose):
        cfg2, vertices2 = _brief(cn2, fi2, "B")

    with _stage("equivalence checks", verbose):
        # -- quick pre-filter ----------------------------------------------------
        _sec("Equivalence checks")
        if cfg1.ambient_dim != cfg2.ambient_dim:
            print(f"  Ambient dimension mismatch ({cfg1.ambient_dim} vs {cfg2.ambient_dim}).")
            print("  No affine equivalence is possible between spaces of different dimension.")
            return False

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
        fi_res = FiniteIndexResult(found=False)
        if cfg1.n_points != cfg2.n_points:
            print(
                f"  {'finite_index':<22} n/a  (different monomial counts: {cfg1.n_points} vs {cfg2.n_points})"
            )
        else:
            fi_res = finite_index_map(cfg1, cfg2)
            status = "YES" if fi_res.found else "no"
            det_str = f"  (det = {fi_res.determinant})" if fi_res.found else ""
            print(f"  {'finite_index':<22} {status}{det_str}")

        # -- witness maps and the identities they give --------------------------
        any_found = any(r.equivalent for _, r in results) or fi_res.found
        if not any_found:
            print()
            print("  No equivalence found between A and B.")
            return False

    with _stage("witness maps", verbose):
        # A map of the hull vertices implies no GKZ identity unless every column
        # is a vertex, and then point_config holds too and prints the identity.
        pts1 = cfg1.affine_points.tolist()
        pts2 = cfg2.affine_points.tolist()
        beta1 = list(fi1.gkz.beta_parameters)
        z2 = list(fi2.gkz.z_variables)
        all_vertices = vertices1 == cfg1.n_points and vertices2 == cfg2.n_points
        # Below full dimension a map's identity holds only trivially: for generic beta
        # the GKZ system has no non-zero solutions.
        trivial = cfg1.affine_dim < cfg1.ambient_dim

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

        def _trivial() -> None:
            if trivial:
                print("  The Newton polytopes are not full-dimensional: the identity holds")
                print("  only trivially.")

        for relation, res in results:
            if not res.equivalent or res.witness_map is None:
                continue
            _witness(relation, res.witness_map, res.translation)
            if relation == "point_config":
                _identity(res.witness_map, res.translation)
                _trivial()
            elif all_vertices:
                print("  Relates the Newton polytopes only; every column is a vertex, so")
                print("  point_config gives the identity.")
            else:
                print("  Relates the Newton polytopes only; not every column is a vertex, so")
                print("  no GKZ identity follows.")

        if fi_res.found and fi_res.witness_matrix is not None:
            _witness("finite_index", fi_res.witness_matrix, fi_res.translation)
            _identity(fi_res.witness_matrix, fi_res.translation, fi_res.column_permutation)
            _trivial()
    return True


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

# Options that take a value. The bare form needs them to tell a value from a
# second diagram: fk "12e|2e|e|" --db x.db names one diagram, not two.
_VALUE_OPTIONS = frozenset(
    {"--db", "--latex", "--text", "--sections", "--seed", "--torus-budget", "--kinematics"}
)

_MAIN_EPILOG = """\
examples:
  fk analyse "12e|2e|e|:zzz"                   every section of the massless triangle
  fk analyse "12e|2e|e|:zzz" -g -n             GKZ system and Newton polytope only
  fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"   the mass on two different propagators
  fk "12e|2e|e|:zzz"                           the bare form of fk analyse
  fk "12e|2e|e|:nzz" "12e|2e|e|:znz"           the bare form of fk compare

Run fk analyse --help or fk compare --help for their options.
Quote every CNickel string: an unquoted | is a shell pipe.

Exit status: 0 on success; 1, with one line on stderr, for a CNickel string
that does not parse or a feynkit, database or file error; 2 for a usage error;
3 when fk compare finds no equivalence; 141, with nothing on stderr, when the
reader of the output closes the pipe.
"""

_ANALYSE_EPILOG = """\
examples:
  fk analyse "12e|2e|e|:zzz"                   every section
  fk analyse "12e|2e|e|:zzz" -g -n             GKZ system and Newton polytope only
  fk analyse "12e|3e|3e|e|:zzzz" -n -S         the massless box: polytope and symmetries
  fk analyse "12e|2e|e|"                       bare topology: every propagator massless
  fk analyse "11e|e|:nn" --torus-count         the point counts alone; slow for larger graphs
  fk analyse "12e|2e|e|:nnn" --latex triangle.tex --text triangle.txt
  fk analyse "12e|2e|e|:nzz" --json --sections gkz,polytope --no-db
  fk analyse "12e|3e|3e|e|:zzzz" --kinematics massless_on_shell -n

Quote every CNickel string: an unquoted | is a shell pipe.
"""

_COMPARE_DESCRIPTION = """\
Compare two diagrams. fk compare prints a summary of each, then tests four
equivalences between their A-configurations: unimodular, affine_polytope,
point_config and finite_index. It prints each map it finds. A point_config
or finite_index map sends every column of one A-matrix to a column of the
other, and for such a map fk compare also prints the identity between the
two integrals. It exits with status 3 when no check finds an equivalence.
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
    database = common.add_mutually_exclusive_group()
    database.add_argument(
        "--db",
        default="feynkit.db",
        metavar="PATH",
        help="SQLite database for the results (default: feynkit.db in the working directory)",
    )
    database.add_argument("--no-db", action="store_true", help="use no database")
    common.add_argument(
        "-v", "--verbose", action="store_true", help="print the time of each stage on stderr"
    )

    parser = argparse.ArgumentParser(
        prog="fk",
        description="Analyse Feynman integrals given as CNickel strings.",
        epilog=_MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        # _parse_args reports the errors of this parser, to explain a bad command.
        exit_on_error=False,
    )
    parser.add_argument("--version", action="version", version=f"fk {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    analyse = commands.add_parser(
        "analyse",
        aliases=["analyze"],
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
    analyse.add_argument(
        "--kinematics",
        type=_kinematic_class,
        metavar="CLASS",
        help="kinematic class to impose; by default the kinematics of the CNickel string",
    )
    shown = analyse.add_argument_group(
        "sections", "sections to print; all but --torus-count when no flag is given"
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
        help=(
            "Newton polytope: vertices, whether scaleless, normalised volume, Smith invariants, "
            "lattice invariants"
        ),
    )
    shown.add_argument(
        "-S", "--symmetries", action="store_true", help="polytope automorphisms and symmetry pairs"
    )
    shown.add_argument(
        "--torus-count",
        action="store_true",
        help="print the candidate Euler characteristic from finite-field point counts; slow",
    )
    counting = analyse.add_argument_group(
        "point counts", "for --torus-count and the torus report section"
    )
    counting.add_argument("--seed", type=_seed, metavar="N", help="seed for the kinematic point")
    counting.add_argument("--torus-budget", type=_budget, metavar="N", help="maximum evaluations")
    report = analyse.add_argument_group(
        "report", "the analysis report of FeynmanIntegral.to_latex and to_text"
    )
    report.add_argument(
        "--latex", type=Path, metavar="FILE", help="write the report as a LaTeX document"
    )
    report.add_argument("--text", type=Path, metavar="FILE", help="write the report as plain text")
    report.add_argument(
        "--json",
        action="store_true",
        help="print a JSON summary of the report on stdout and nothing else",
    )
    report.add_argument(
        "--sections",
        type=_section_list,
        metavar="NAMES",
        help=(
            f"comma-separated report sections from {', '.join(SECTION_NAMES)}; "
            "all but torus by default"
        ),
    )

    compare = commands.add_parser(
        "compare",
        parents=[common],
        help="compare two diagrams",
        description=_COMPARE_DESCRIPTION,
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
    the bare form is rewritten before parsing. Every CNickel string fk prints
    or documents has a |, so a first positional with one starts the bare form:
    two positionals mean compare and any other number means analyse, which
    then reports surplus arguments itself. Otherwise argv is left alone, so
    that argparse takes a command such as analyze and reports a misspelt one,
    and argv without positionals, such as --help or --version, is left alone
    too.
    """
    args = list(argv)
    positionals = _positionals(args)
    if not positionals or "|" not in positionals[0]:
        return args
    return ["compare" if len(positionals) == 2 else "analyse", *args]


def _cnickel_hint(positionals: Sequence[str]) -> str:
    """A line for the error on a first word that looks like a CNickel string without a |.

    The parser reads 0:n as the tadpole 0|:n, but _with_command needs a | to
    see the bare form, so argparse takes 0:n for a misspelt command.
    """
    if not positionals or ":" not in positionals[0]:
        return ""
    command = "compare" if len(positionals) == 2 else "analyse"
    quoted = " ".join(f'"{word}"' for word in positionals)
    return f"\na CNickel string without a | needs the command: fk {command} {quoted}"


def _reject_options_before(
    parser: argparse.ArgumentParser, words: Sequence[str], command: str
) -> None:
    """Exit through parser when options of its command come before the command in words.

    fk's own parser does not know them: it leaves a flag such as --no-db over,
    and takes the value of an option such as --db PATH for the command. The
    error shows each option, with its value, after the command.
    """
    misplaced: list[str] = []
    before = iter(words[: words.index(command)])
    for word in before:
        if word.split("=", 1)[0] in parser._option_string_actions:
            misplaced.append(word)
            value = next(before, None) if word in _VALUE_OPTIONS else None
            if value is not None:
                misplaced.append(value)
    if misplaced:
        options = " ".join(misplaced)
        parser.error(f"{options} must follow the command: {parser.prog} {options} ...")


def _parse_args(parsers: _Parsers, argv: Sequence[str]) -> argparse.Namespace:
    """The arguments of argv, with the bare form rewritten and analyze read as analyse.

    An unknown option or a surplus positional is a usage error of the
    subcommand, so argparse prints that subcommand's usage, not fk's. An
    option of the subcommand given before it, as in fk --no-db analyse X, is
    told to follow the command.
    """
    words = _with_command(argv)
    try:
        args, extra = parsers.main.parse_known_args(words)
    except argparse.ArgumentError as exc:
        hint = ""
        if exc.argument_name == "COMMAND":
            # A word that is no command: the value of an option given before
            # the command, or a CNickel string without a |.
            positionals = _positionals(words)
            if positionals and positionals[0] in ("analyse", "analyze", "compare"):
                parser = parsers.compare if positionals[0] == "compare" else parsers.analyse
                _reject_options_before(parser, words, positionals[0])
            hint = _cnickel_hint(positionals)
        parsers.main.error(f"{exc}{hint}")
    command = args.command
    if command == "analyze":
        args.command = "analyse"
    if extra:
        parser = parsers.analyse if args.command == "analyse" else parsers.compare
        _reject_options_before(parser, words, command)
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args


def _section_flags(args: argparse.Namespace) -> set[str]:
    """The sections that the section flags of analyse choose; --torus-count chooses "torus"."""
    chosen = {name for name in SECTION_FLAGS if getattr(args, name)}
    return chosen | {"torus"} if args.torus_count else chosen


def _report_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ReportOptions:
    """The report options of analyse, after the checks argparse cannot express.

    A usage error exits through the parser; a report file that cannot be
    written raises CliError, before any analysis.
    """
    if args.json and _section_flags(args):
        parser.error("--json prints only the summary; drop the section flags")
    if args.sections is not None and not (args.latex or args.text or args.json):
        parser.error("--sections chooses report sections; add --latex, --text or --json")
    counts = args.torus_count or (args.sections is not None and "torus" in args.sections)
    for option, value in (("--seed", args.seed), ("--torus-budget", args.torus_budget)):
        if value is not None and not counts:
            parser.error(
                f"{option} applies to the point counts; add --torus-count or name torus in "
                "--sections"
            )
    for path, kind in ((args.latex, "LaTeX"), (args.text, "text")):
        if path is not None:
            _check_writable(path, kind)
    return ReportOptions(
        sections=DEFAULT_SECTIONS if args.sections is None else args.sections,
        latex=args.latex,
        text=args.text,
        as_json=args.json,
        torus_seed=DEFAULT_TORUS_SEED if args.seed is None else args.seed,
        torus_budget=DEFAULT_TORUS_BUDGET if args.torus_budget is None else args.torus_budget,
    )


def _run(parsers: _Parsers, args: argparse.Namespace) -> int:
    """Run the subcommand of args, reporting the errors main describes in one line.

    Returns the exit status: EXIT_NOT_EQUIVALENT when fk compare finds no
    equivalence, otherwise 0.
    """
    db_path = None if args.no_db else Path(args.db)
    try:
        if args.command == "analyse":
            options = _report_options(parsers.analyse, args)
            analyse_one(
                args.cnickel,
                db_path,
                _section_flags(args),
                report=options,
                verbose=args.verbose,
                kinematics=args.kinematics,
            )
            return 0
        found = analyse_pair(args.first, args.second, db_path, verbose=args.verbose)
        return 0 if found else EXIT_NOT_EQUIVALENT
    except CliError as exc:
        _fail(str(exc))
    except FeynkitError as exc:
        _fail(_MAX_EVALUATIONS.sub(r"--torus-budget \1", str(exc)) or type(exc).__name__)
    except sqlite3.Error as exc:
        _fail(f"database {args.db}: {exc}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run fk.

    A CNickel string that does not parse, a feynkit error or a database error
    is reported in one line on stderr with exit status 1, and argparse exits
    with status 2 on a usage error. fk compare exits with status 3 when none of
    its checks finds an equivalence, as when the ambient dimensions differ, and
    with 0 when one does. When the reader of stdout closes the pipe early, as
    head does, fk stops without a message and exits with status 141, which is
    128 + SIGPIPE, the status a shell reports for cat or grep in the same
    place. Any other exception is a bug and keeps its traceback.
    """
    parsers = _build_parser()
    args = _parse_args(parsers, sys.argv[1:] if argv is None else argv)
    try:
        status = _run(parsers, args)
    except BrokenPipeError:
        # The database is closed by now. Point stdout at devnull so that the
        # interpreter's final flush does not fail again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(EXIT_BROKEN_PIPE) from None
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
