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
import ast
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
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple, NoReturn

import sympy as sp

from feynkit import _exact
from feynkit.a_configuration import AConfiguration, FiniteIndexResult, finite_index_map
from feynkit.core.constants import __version__
from feynkit.core.exceptions import FeynkitError, ValidationError
from feynkit.core.graph import Graph
from feynkit.database import FeynkitDatabase
from feynkit.face_identification import identify_faces
from feynkit.face_lattice import DecoratedFaceLattice, Epsilon
from feynkit.integral import FeynmanIntegral
from feynkit.io.report import (
    DEFAULT_SECTIONS,
    FACE_CODIMENSION,
    LATTICE_BUDGET,
    NORMALIZ_TIMEOUT,
    SECTION_NAMES,
    AnalysisReport,
    Faces,
)
from feynkit.io.report_latex import render_latex
from feynkit.io.report_text import render_text
from feynkit.io.sections.degeneracy import Degeneracy, decide
from feynkit.io.sections.face_lattice import face_rows, kind_counts
from feynkit.io.sections.faces import face_counts, facet_graph
from feynkit.io.sections.resonance import epsilon_set
from feynkit.kinematics.classes import IMPOSABLE_CLASSES, KinematicClass
from feynkit.landau import LandauAnalysis
from feynkit.point_count import TorusCount
from feynkit.polytope import polytope_data
from feynkit.resonance import choose_d0, classify_facets, span_epsilons

SECTION_FLAGS = ("symanzik", "params", "gkz", "toric", "newton", "resonance", "symmetries")
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


_ASSIGNMENT_GRAMMAR = "expected SYMBOL=VALUE, such as p4^2=0"
_WORD = re.compile(r"[A-Za-z_]\w*")


def _assignment(text: str) -> tuple[str, str]:
    """Parse --set: SYMBOL=VALUE, both sides non-empty and stripped."""
    name, equals, value = text.partition("=")
    name, value = name.strip(), value.strip()
    if not (equals and name and value):
        raise argparse.ArgumentTypeError(f"{_ASSIGNMENT_GRAMMAR}; got {text!r}")
    return name, value


def _unknown_symbol(name: str, cnickel: str, symbols: Sequence[sp.Symbol]) -> CliError:
    listed = ", ".join(str(symbol) for symbol in symbols) or "none"
    return CliError(
        f"unknown kinematic symbol {name!r}; the symbols of CNickel {cnickel!r} are {listed}"
    )


def _exact_value(text: str, symbols: dict[str, sp.Symbol], cnickel: str) -> sp.Expr:
    """The value of a --set as a linear form in the symbols, read exactly.

    Integers and quotients of them, the symbols, + - * / and brackets are read.
    Anything else raises CliError: a float or a function is not exact, an
    unknown name is not a symbol, and a product or quotient of symbols, or a
    power, is not linear.
    """
    slots: dict[str, sp.Symbol] = {}
    replaced = text
    for index, name in enumerate(sorted(symbols, key=len, reverse=True)):
        slot = f"_slot{index}_"
        if name in replaced:
            replaced = replaced.replace(name, slot)
            slots[slot] = symbols[name]
    if re.search(r"\d\.|\.\d|\d[eE][+-]?\d", replaced):
        raise CliError(f"cannot read {text!r} exactly; write fractions such as 1/2, not decimals")
    for word in _WORD.findall(replaced):
        if word not in slots:
            if re.search(rf"{word}\s*\(", replaced):
                raise CliError(
                    f"cannot read {word}(...) exactly; use integers, fractions and symbols"
                )
            raise _unknown_symbol(word, cnickel, list(symbols.values()))
    try:
        tree = ast.parse(replaced, mode="eval")
    except SyntaxError:
        raise CliError(f"cannot read {text!r}; use integers, fractions and symbols") from None

    def value(node: ast.expr) -> sp.Expr:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.Constant):
            raise CliError(
                f"cannot read {text!r} exactly; write fractions such as 1/2, not decimals"
            )
        if isinstance(node, ast.Name):
            return slots[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = value(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, ast.Add | ast.Sub | ast.Mult | ast.Div
        ):
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise CliError(f"cannot read {text!r}: division by zero")
            return left / right
        if isinstance(node, ast.BinOp):
            raise CliError(f"not linear: {text!r} has a power")
        raise CliError(f"cannot read {text!r}; use integers, fractions and symbols")

    result = sp.expand(value(tree.body))
    if (
        not result.is_polynomial(*slots.values())
        or sp.Poly(result, *symbols.values()).total_degree() > 1
    ):
        raise CliError(f"not linear: {text!r} is not a linear combination of the symbols")
    return result


def _substitute(
    integral: FeynmanIntegral, cnickel: str, assignments: Sequence[tuple[str, str]]
) -> tuple[FeynmanIntegral, list[str]]:
    """The integral with each --set applied in order, and the substitutions as stated.

    Each assignment names a symbol of the momentum products as they stand after
    the earlier ones. A failure raises CliError naming the assignment.
    """
    stated: list[str] = []
    for name, text in assignments:
        products = integral.momentum_products
        symbols = {
            str(symbol): symbol
            for symbol in sorted(
                set().union(*(sp.sympify(v).free_symbols for v in products.values())),
                key=str,
            )
        }
        try:
            if name not in symbols:
                raise _unknown_symbol(name, cnickel, list(symbols.values()))
            target = symbols[name]
            value = _exact_value(text, symbols, cnickel)
            if value.has(target):
                raise CliError(f"the value contains {name} itself")
        except CliError as exc:
            raise CliError(f"--set '{name}={text}': {exc}") from None
        integral = integral.with_(
            momentum_products={
                k: sp.expand(sp.sympify(v).subs(target, value)) for k, v in products.items()
            }
        )
        stated.append(f"{name} = {value}")
    return integral, stated


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


def _print_graph(given: str, fi: FeynmanIntegral, substitutions: Sequence[str] = ()) -> None:
    _header(f"Feynman integral  {_label(given, fi)}")
    _kv("Nickel index", fi.nickel_index)
    _kv("Loop count", fi.loop_count)
    _kv("Propagators", len(fi.graph.get_internal_edges()))
    _kv("External legs", fi.graph.external_legs)
    _kv("Kinematic class", fi.kinematic_class)
    if substitutions:
        _kv("Substitutions", ", ".join(substitutions))


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


def _yes_no(value: bool | None) -> str:
    return "not computed" if value is None else "yes" if value else "no"


def _print_lattice_invariants(fi: FeynmanIntegral, full: bool) -> None:
    """The lattice invariants of the Newton polytope and, when NA is normal, the certificate."""
    found = fi.lattice_invariants(budget=LATTICE_BUDGET, timeout=NORMALIZ_TIMEOUT)
    index = found.gorenstein_index
    gorenstein = "none" if index is None else str(index)
    _kv("Lattice points", f"{found.lattice_points}  ({found.interior_points} interior)")
    _kv("h*-vector", "not computed" if found.h_star is None else found.h_star)
    _kv("Gorenstein index", f"{gorenstein}  (reflexive)" if index == 1 else gorenstein)
    _kv("Lattice width", found.lattice_width)
    _kv("Integer decomposition (IDP)", _yes_no(found.idp))
    if found.normal is None:
        _kv(
            "Normal configuration",
            "not computed  (install PyNormaliz, or call lattice_invariants directly)",
        )
        return
    if found.normal:
        _kv("Normal configuration", "yes")
        if full:
            print("  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972)")
            print("  and there are no rank jumps (Matusevich, Miller and Walther 2005).")
        else:
            print("  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972).")
        return
    reasons = []
    if found.idp is False:
        reasons.append("P is not IDP")
    if not found.support_is_saturated:
        missing = found.lattice_points - len({tuple(p) for p in fi.newton_polytope.points})
        reasons.append(
            f"the support misses {missing} lattice point{'s' if missing != 1 else ''} of P"
        )
    _kv("Normal configuration", f"no  ({'; '.join(reasons)})")


# Where D_0 comes from, as the terminal says it.
_D0_ORIGIN = {
    "given": "given by --d0",
    "dimension": "read from the dimension of the integral",
    "default": "the default",
}


def _print_d0_and_powers(fi: FeynmanIntegral, d0: Fraction, source: str) -> list[int]:
    """Print D_0 and the powers, 1 on every edge when the exponents are not all integers, and
    return the powers."""
    _kv("D_0", f"{d0}  ({_D0_ORIGIN[source]})")
    edges = fi.graph.get_internal_edges()
    try:
        powers = [_exact._as_int(fi.propagator_exponents[e.idx], "") for e in edges]
        note = ""
    except ValidationError:
        powers, note = [1] * len(edges), "  (the exponents are not all integers, so 1 is used)"
    _kv("Powers nu_e", f"({', '.join(map(str, powers))}){note}")
    return powers


def _print_resonance(fi: FeynmanIntegral, given: Fraction | None) -> None:
    """Each facet: its inequality, l_F(beta), and where it is resonant and admissible."""
    d0, source = choose_d0(given, fi.dimension)
    _sec(f"Resonance  (D = {d0} - 2 epsilon)")
    powers = _print_d0_and_powers(fi, d0, source)
    edges = fi.graph.get_internal_edges()
    data = polytope_data(fi.newton_polytope.points)
    if not data.is_full_dimensional:
        _kv("beta in the span of A for", epsilon_set(span_epsilons(data, powers, d0), latex=False))
        print("  P is not full-dimensional: the facets are relative to its affine hull.")
    x = [sp.Symbol(f"x_{e.idx}") for e in edges]
    nu = [sp.Symbol(f"nu_{e.idx}") for e in edges]
    for k, record in enumerate(classify_facets(data, powers, d0), start=1):
        lhs = sp.Add(*(m * v for m, v in zip(record.facet.normal, x, strict=True)))
        form = record.form.expression(sp.Symbol("D"), nu)
        zero = "yes" if record.resonant_at_zero else "no"
        reducible = "yes" if record.reducible else "not decided"
        print(f"  F_{k:<3} {lhs} <= {record.facet.offset}    l_F(beta) = {form}")
        print(
            f"        resonant: {epsilon_set(record.resonant, latex=False)} (at epsilon = 0: "
            f"{zero}); admissible: {epsilon_set(record.admissible, latex=False)}; "
            f"reducible: {reducible}"
        )


def _print_faces(fi: FeynmanIntegral) -> None:
    """The graph of each facet and the number of faces in each class."""
    _sec(f"Faces as graphs  (up to codimension {FACE_CODIMENSION})")
    data = polytope_data(fi.newton_polytope.points)
    found = identify_faces(fi, max_codimension=FACE_CODIMENSION, data=data)
    if not data.is_full_dimensional:
        print("  P is not full-dimensional: its faces are not identified.")
        return
    x = [sp.Symbol(f"x_{e.idx}") for e in fi.graph.get_internal_edges()]
    facets = [face for face in found if face.facet is not None]
    for k, face in enumerate(facets, start=1):
        assert face.facet is not None
        inequality = f"{sp.Add(*(m * v for m, v in zip(face.facet.normal, x, strict=True)))}"
        inequality += f" <= {face.facet.offset}"
        if face.verified:
            graph = face.name()
        elif face.kind == "support_product":
            graph = f"support product of {face.name()}"
        else:
            graph = f"unidentified; predicted {face.name()}"
        print(f"  F_{k:<3} {inequality:<32} {graph}")
    section = Faces(faces=found, max_codimension=FACE_CODIMENSION, full_dimensional=True)
    columns = "".join(f"{f'codim {c}':>9}" for c in range(FACE_CODIMENSION + 1))
    print(f"  {'Class':<14}{columns}")
    for name, counts in face_counts(section):
        print(f"  {name:<14}{''.join(f'{n:>9}' for n in counts)}")
    if not all(face.verified for face in found):
        print("  The report (--latex or --text) writes out G|_F and the prediction.")


def _centres(lattice: DecoratedFaceLattice, eps: Epsilon) -> str:
    """The number of resonance centres at eps, or the empty face, and whether the system is
    reducible there."""
    centres = lattice.centres(eps)
    if not centres:
        return "none: no face is resonant, beta lies outside the span of A"
    reducible = lattice.reducible(eps)
    verdict = "not decided" if reducible is None else "yes" if reducible else "no"
    found = "the empty face" if not centres[0].point_indices else str(len(centres))
    return f"{found}; reducible: {verdict}"


def _print_centres(lattice: DecoratedFaceLattice, eps: Epsilon) -> None:
    """The resonance centres at eps, unless there are none or the empty face is the centre."""
    centres = lattice.centres(eps)
    if centres and centres[0].point_indices:
        for dimension, facets, graph in face_rows(centres, latex=False):
            print(f"    dimension {dimension:<3} facets {facets:<20} {graph}")


def _print_face_lattice(fi: FeynmanIntegral, given: Fraction | None, check: bool) -> None:
    """The faces by dimension and resonant set, the centres at generic eps and at eps = 0 with
    the verdict on reducibility, and with check the comparison with the Cayley side."""
    d0, source = choose_d0(given, fi.dimension)
    _sec(f"Face resonance and reducibility  (D = {d0} - 2 epsilon)")
    powers = _print_d0_and_powers(fi, d0, source)
    edges = [e.idx for e in fi.graph.get_internal_edges()]
    lattice = fi.face_lattice(
        d0, nu=dict(zip(edges, powers, strict=True)), identify_codimension=FACE_CODIMENSION
    )
    columns = (("Faces", 7), ("every eps", 11), ("progression", 13), ("one value", 11), ("none", 6))
    print(f"  {'Dimension':<11}" + "".join(f"{name:>{w}}" for name, w in columns))
    for k, row in kind_counts(lattice):
        cells = (sum(row), *row)
        print(f"  {k:<11}" + "".join(f"{n:>{w}}" for n, (_, w) in zip(cells, columns, strict=True)))
    _kv("Centres at generic epsilon", _centres(lattice, "generic"))
    _print_centres(lattice, "generic")
    _kv("Centres at epsilon = 0", _centres(lattice, 0))
    _print_centres(lattice, 0)
    if not lattice.full_rank:
        print("  A does not have full rank, so reducibility is not decided.")
    pyramids = sum(face.pyramid and face.codimension > 0 for face in lattice.faces)
    if pyramids:
        _kv("Pyramid faces besides P", pyramids)
    defects = sum(face.lattice_defect != 1 for face in lattice.faces)
    if defects:
        _kv("Faces with a lattice defect", defects)
    if check:
        found = lattice.check_schwinger()
        if found.agrees:
            _kv("Cayley side", "agrees face by face")
        else:
            _kv("Cayley side", f"differs on {len(found.mismatches)} faces")


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


def _print_degeneracy(section: Degeneracy) -> None:
    """The degenerate faces at the point of the point counts, as the report section has them."""
    _sec("Degenerate faces  (at the kinematic point of the point counts)")
    analysis = section.analysis
    if analysis is None:
        print("  Not decided: faces of dimension 2 or more need Singular, which was not found.")
        return
    if analysis.point:
        point = ", ".join(f"{key} = {value}" for key, value in analysis.point)
        _kv("Kinematic point", f"{point}  (seed {section.seed})")
    else:
        _kv("Kinematic point", "none needed: G has no kinematic symbols")
    if analysis.support_loss:
        _kv("Support loss", f"{analysis.support_loss}  (coefficients vanish at the point)")
    _kv("Normalised volume", analysis.volume)
    degenerate = analysis.degenerate_faces
    _kv("Degenerate faces", len(degenerate))
    for face, found in zip(degenerate, section.identifications, strict=True):
        tau = "-" if face.tjurina is None else face.tjurina
        graph = facet_graph(found, latex=False) if found is not None else "-"
        print(
            f"    dimension {face.dimension:<3} points {len(face.point_indices):<4} "
            f"dim Sing {face.singular_dimension:<3} tau {str(tau):<4} {graph}"
        )
    undecided = analysis.undecided_faces
    if undecided:
        _kv("Undecided faces", f"{len(undecided)}  (Singular ran past its time limit)")
    elif not degenerate and analysis.volume:
        print("  No face is degenerate, so |chi(X)| = N! Vol(P_z) = " f"{analysis.volume}.")
    if analysis.prime is not None:
        checked = [f for f in analysis.faces if f.modular is not None and f.degenerate is not None]
        differ = sum(f.modular != f.degenerate for f in checked)
        verdict = "agrees on every face" if not differ else f"differs on {differ} faces"
        _kv("Check over F_p", f"{verdict}  (p = {analysis.prime})")


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
    limits: bool = False
    d0: Fraction | None = None
    check_schwinger: bool = False

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


def _d0(text: str) -> Fraction:
    """Parse --d0: an integer or a fraction such as 7/2, read exactly."""
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(
            f"expected an integer or a fraction such as 7/2; got {text!r}"
        ) from None


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


def _write_reports(
    report: AnalysisReport,
    options: ReportOptions,
    *,
    announce: bool,
    substitutions: Sequence[str] = (),
) -> None:
    """Write the LaTeX and text reports asked for, saying so on stdout when announce is set."""
    if options.latex is not None:
        _write(options.latex, render_latex(report, substitutions=substitutions), "LaTeX")
        if announce:
            print(f"  Wrote the LaTeX report to {options.latex}")
    if options.text is not None:
        _write(options.text, render_text(report, substitutions=substitutions), "text")
        if announce:
            print(f"  Wrote the text report to {options.text}")


def _json_value(value: str) -> bool | int | str | None:
    """A summary value as an integer when it is one, a boolean for "yes" and "no", None for
    "none", "not computed" and "not decided", otherwise as given."""
    if value in ("none", "not computed", "not decided"):
        return None
    if value in ("yes", "no"):
        return value == "yes"
    try:
        return int(value)
    except ValueError:
        return value


def _summary_json(
    given: str,
    fi: FeynmanIntegral,
    report: AnalysisReport,
    sections: Sequence[str],
    substitutions: Sequence[str] = (),
) -> str:
    """The report's summary as JSON, keyed in snake case, with the CNickel strings.

    The substitutions of --set are listed under "substitutions" when there are any.
    """
    summary = {
        label.lower().replace(" ", "_"): _json_value(value) for label, value in report.summary()
    }
    payload = {
        "input": given,
        "cnickel": fi.cnickel,
        "sections": list(sections),
        "summary": summary,
    }
    if substitutions:
        payload["substitutions"] = list(substitutions)
    return json.dumps(payload, indent=2)


def analyse_one(
    cnickel: str,
    db_path: Path | None,
    sections: set[str],
    *,
    report: ReportOptions | None = None,
    verbose: bool = False,
    kinematics: KinematicClass | None = None,
    substitutions: Sequence[tuple[str, str]] = (),
) -> None:
    """Analyse one diagram: every section or the chosen ones, then the reports asked for.

    ``kinematics`` is a kinematic class to impose on the integral of the
    string, and ``substitutions`` are (symbol, value) pairs of --set, applied
    after it and stated in the headers. With --json, only the report's summary is printed, as JSON. Stdout is
    flushed after each stage; verbose prints each stage's time on stderr. The
    point counts are printed only when ``sections`` holds "torus". They run once
    when the report holds them too, and otherwise take the report's Landau
    analysis when it has one. The degenerate faces are printed only when
    ``sections`` holds "degeneracy", and are shared with the report in the same
    way.
    """
    options = report if report is not None else ReportOptions()
    t0 = time.perf_counter()
    with _stage("parse", verbose):
        fi = _load(cnickel, kinematics)
        fi, stated = _substitute(fi, cnickel, substitutions)
    built: list[AnalysisReport] = []

    def build() -> AnalysisReport:
        if not built:
            built.append(
                AnalysisReport.from_integral(
                    fi,
                    options.sections,
                    torus_seed=options.torus_seed,
                    torus_budget=options.torus_budget,
                    limits=True if options.limits else None,
                    d0=options.d0,
                    check_schwinger=options.check_schwinger,
                )
            )
        return built[0]

    with ExitStack() as stack:
        db = _open_database(stack, db_path)
        if options.as_json:
            with _stage("report", verbose):
                _write_reports(build(), options, announce=False, substitutions=stated)
            if db is not None:
                with _stage("database", verbose):
                    db.store(fi, label=fi.cnickel)
            print(_summary_json(cnickel, fi, build(), options.sections, stated))
            return
        with _stage("graph", verbose):
            _print_graph(cnickel, fi, stated)
        for name in SECTION_FLAGS:
            if not sections or name in sections:
                with _stage(name, verbose):
                    if name == "resonance":
                        _print_resonance(fi, options.d0)
                    else:
                        _PRINTERS[name](fi)
        if not sections or "faces" in sections:
            with _stage("faces", verbose):
                _print_faces(fi)
        if not sections or "face_lattice" in sections:
            with _stage("face lattice", verbose):
                _print_face_lattice(fi, options.d0, "face_lattice" in sections)
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
        if "degeneracy" in sections:
            with _stage("degeneracy", verbose):
                faces: Degeneracy | None = None
                shared: LandauAnalysis | None = None
                if options.writes_files and "degeneracy" in options.sections:
                    faces = build().degeneracy
                elif options.writes_files and "landau" in options.sections:
                    found = build().landau
                    shared = found.analysis if found is not None else None
                if faces is None:
                    faces = decide(fi, seed=options.torus_seed, landau=shared)
                _print_degeneracy(faces)
        if options.writes_files:
            with _stage("report", verbose):
                _sec("Report")
                _write_reports(build(), options, announce=True, substitutions=stated)
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
    {
        "--db",
        "--latex",
        "--text",
        "--sections",
        "--seed",
        "--torus-budget",
        "--kinematics",
        "--set",
        "--d0",
    }
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
  fk analyse "12e|22e|e|:nnnn" -f              the graphs of the parachute's faces
  fk analyse "111e|e|:nnn" -D                  the degenerate faces of the massive sunrise

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
    analyse.add_argument(
        "--set",
        dest="substitutions",
        action="append",
        default=[],
        type=_assignment,
        metavar="SYMBOL=VALUE",
        help=(
            "set a kinematic symbol to 0, a rational number, another symbol or a linear "
            "combination of them, after --kinematics; repeatable, as in --set 'p4^2=0'"
        ),
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
        "-r",
        "--resonance",
        action="store_true",
        help=(
            "where each facet of the Newton polytope is resonant or admissible as "
            "D = D_0 - 2 epsilon varies, and whether it makes the GKZ system reducible"
        ),
    )
    shown.add_argument(
        "--d0",
        type=_d0,
        metavar="VALUE",
        help=(
            "D_0 of --resonance and of the report's resonance section, an integer or a "
            "fraction such as 7/2; 4 by default"
        ),
    )
    shown.add_argument(
        "-f",
        "--faces",
        action="store_true",
        help=(
            f"the graph of each face of the Newton polytope up to codimension {FACE_CODIMENSION}, "
            "a product of Symanzik polynomials of minors checked exactly; it also adds the "
            "faces section to reports whose --sections leave it out"
        ),
    )
    shown.add_argument(
        "-L",
        "--face-lattice",
        action="store_true",
        help=(
            "where every face of the Newton polytope is resonant, the resonance centres at "
            "generic epsilon and at epsilon = 0, and whether the GKZ system is reducible; it "
            "also compares the Cayley configuration of the Schwinger representation face by "
            "face, and adds the face_lattice section, with that comparison, to the reports"
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
    shown.add_argument(
        "-D",
        "--degeneracy",
        action="store_true",
        help=(
            "the faces of the Newton polytope on which G has a singular point in the torus, "
            "decided exactly at the kinematic point of the point counts; it needs Singular, "
            "and adds the degeneracy section to reports whose --sections leave it out"
        ),
    )
    counting = analyse.add_argument_group(
        "point counts", "for --torus-count and the torus report section"
    )
    counting.add_argument(
        "--seed",
        type=_seed,
        metavar="N",
        help="seed for the kinematic point, which --degeneracy shares",
    )
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
        "--limits",
        action="store_true",
        help=(
            "look for limit surfaces in the Landau section: the factors of the parent family's "
            "surfaces that the kinematics restrict; it analyses the parent family as well"
        ),
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
    """The sections that the section flags of analyse choose; --faces chooses "faces",
    --face-lattice "face_lattice", --torus-count "torus" and --degeneracy "degeneracy"."""
    chosen = {name for name in SECTION_FLAGS if getattr(args, name)}
    chosen |= {"faces"} if args.faces else set()
    chosen |= {"face_lattice"} if args.face_lattice else set()
    chosen |= {"degeneracy"} if args.degeneracy else set()
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
    if args.limits and not (args.latex or args.text or args.json):
        parser.error("--limits changes the report; add --latex, --text or --json")
    flags = _section_flags(args)
    printed = not flags or bool({"resonance", "face_lattice"} & flags)
    if args.d0 is not None and not (printed or args.latex or args.text or args.json):
        parser.error(
            "--d0 applies to the resonance and face-lattice sections; add -r, -L, --latex, "
            "--text or --json"
        )
    named = set(args.sections or ())
    counts = args.torus_count or "torus" in named
    if args.torus_budget is not None and not counts:
        parser.error(
            "--torus-budget applies to the point counts; add --torus-count or name torus in "
            "--sections"
        )
    if args.seed is not None and not (counts or args.degeneracy or "degeneracy" in named):
        parser.error(
            "--seed applies to the point counts and the degenerate faces; add --torus-count "
            "or --degeneracy, or name torus or degeneracy in --sections"
        )
    for path, kind in ((args.latex, "LaTeX"), (args.text, "text")):
        if path is not None:
            _check_writable(path, kind)
    sections = DEFAULT_SECTIONS if args.sections is None else args.sections
    if args.faces and "faces" not in sections:
        sections = (*sections, "faces")
    if args.face_lattice and "face_lattice" not in sections:
        sections = (*sections, "face_lattice")
    if args.degeneracy and "degeneracy" not in sections:
        sections = (*sections, "degeneracy")
    return ReportOptions(
        sections=sections,
        latex=args.latex,
        text=args.text,
        as_json=args.json,
        torus_seed=DEFAULT_TORUS_SEED if args.seed is None else args.seed,
        torus_budget=DEFAULT_TORUS_BUDGET if args.torus_budget is None else args.torus_budget,
        limits=args.limits,
        d0=args.d0,
        check_schwinger=args.face_lattice,
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
                substitutions=args.substitutions,
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
